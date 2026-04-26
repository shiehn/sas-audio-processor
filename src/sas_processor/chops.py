"""Beat-aligned audio chopping.

Used by the v1 transition generator (signals-and-sorcery) to extract
1-, 2-, or 4-beat segments from per-layer WAV stems. The transition
generator invokes `trim-range` once per event; each call produces a tiny
WAV that is later concatenated + mixed back into the final transition.

Design notes:
  - Sample rate is read from the input file (we do not resample).
  - Start/duration are expressed in BEATS (not bars, not seconds) so the
    caller can specify quarter-bar chops without coordinating bar math
    with the Python side.
  - `trim_audio` from `processor.py` already zero-pads when the requested
    range exceeds the input — we rely on that behavior to make end-of-file
    chops safe without special-casing them here.
  - Optional linear fade-in / fade-out envelopes applied after trim.
    Used by the transition generator for sample crossfades (scene-A
    samples fade out, scene-B samples fade in across the transition)
    and by LLM-emitted events that want a soft overlap between layers.
  - Optional `reverse`, `stutter_repeats`, and `volume` fill effects.
    These are first-class TransitionEvent fields in the plan JSON so
    re-rendering is deterministic. All three preserve the chop's total
    sample count (the "length invariant") — only content changes.
    Order of operations: trim → reverse → stutter → fades → volume.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import soundfile as sf

from sas_processor.io_utils import _log_event, atomic_sf_write
from sas_processor.processor import trim_audio


def trim_range(
    input_path: str,
    output_path: str,
    bpm: float,
    meter: int,
    start_beat: float,
    duration_beats: float,
    fade_in_beats: float = 0.0,
    fade_out_beats: float = 0.0,
    reverse: bool = False,
    stutter_repeats: int = 1,
    volume: float = 1.0,
) -> dict:
    """Write a chop of `duration_beats` starting at `start_beat` from `input_path`.

    Args:
        input_path:      Source WAV file.
        output_path:     Destination WAV file (parent directory must exist).
        bpm:             Tempo of the source stem (must match how it was rendered).
        meter:           Beats per bar (informational — kept for API symmetry
                         with split_audio_bars; `trim_range` itself only uses bpm).
        start_beat:      Beat offset (0-indexed) from the start of the input.
        duration_beats:  Length of the chop, in beats. For v1 this is 1, 2, or 4.
        fade_in_beats:   If > 0, apply a linear fade from 0 → 1 over the first
                         `fade_in_beats` beats of the chop. Default 0 (no fade).
        fade_out_beats:  If > 0, apply a linear fade from 1 → 0 over the last
                         `fade_out_beats` beats of the chop. Default 0 (no fade).
        reverse:         If True, reverse the trimmed chop along the time axis
                         BEFORE applying fades. Preserves length. Used by the
                         "reverse sweep" fill at the end of scene A.
        stutter_repeats: If > 1, tile the chop's first (num_samples / N) samples
                         N times to fill exactly `num_samples`. Used by the
                         "stutter roll" fill. N=1 (default) is a no-op.
        volume:          Scalar gain (0..1) applied to the final chop AFTER
                         fades and stutter. Used for ghost hits (0.4) and kick
                         drops (0.0). Default 1.0 (no change).

    Returns:
        Dict with `output`, `samples`, `duration_s`, `sample_rate`,
        `channels`, `start_sample`, `fade_in_samples`, `fade_out_samples`,
        `reverse`, `stutter_repeats`, `volume`. Suitable for emit_json.

    Raises:
        ValueError  on invalid bpm / meter / duration / fade / fill args.
        FileNotFoundError on missing input.
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if meter <= 0:
        raise ValueError(f"meter must be > 0, got {meter}")
    if duration_beats <= 0:
        raise ValueError(f"duration_beats must be > 0, got {duration_beats}")
    if start_beat < 0:
        raise ValueError(f"start_beat must be >= 0, got {start_beat}")
    if fade_in_beats < 0:
        raise ValueError(f"fade_in_beats must be >= 0, got {fade_in_beats}")
    if fade_out_beats < 0:
        raise ValueError(f"fade_out_beats must be >= 0, got {fade_out_beats}")
    if fade_in_beats + fade_out_beats > duration_beats:
        raise ValueError(
            f"fade_in_beats ({fade_in_beats}) + fade_out_beats ({fade_out_beats}) "
            f"must not exceed duration_beats ({duration_beats})"
        )
    if not isinstance(stutter_repeats, (int, np.integer)) or stutter_repeats < 1:
        raise ValueError(
            f"stutter_repeats must be a positive integer, got {stutter_repeats!r}"
        )
    if not (0.0 <= volume <= 1.0):
        raise ValueError(f"volume must be in [0, 1], got {volume}")

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    audio, sr = sf.read(input_path, always_2d=False)

    # Preserve the input WAV's subtype (PCM_16 / PCM_24 / FLOAT) so the
    # chop matches the source stem's bit depth. Without this, soundfile
    # defaults to PCM_16 on write, which produces a bit-depth mismatch
    # when the chops are later concatenated with silence WAVs that the TS
    # transition renderer sizes to the stem's actual format (24-bit from
    # Tracktion). The wav-concat / wav-mix utilities then refuse the mix.
    src_info = sf.info(input_path)

    samples_per_beat = (60.0 / bpm) * sr
    start_sample = int(round(start_beat * samples_per_beat))
    num_samples = int(round(duration_beats * samples_per_beat))
    fade_in_samples = int(round(fade_in_beats * samples_per_beat))
    fade_out_samples = int(round(fade_out_beats * samples_per_beat))

    chop = trim_audio(audio, start_sample, num_samples)

    if reverse:
        # Stride-flip on the time axis. Copy to detach from the negative
        # stride view — downstream ops (fade, volume) allocate a new array
        # anyway, but the copy is cheap and avoids any stride surprises.
        chop = chop[::-1].copy()

    if stutter_repeats > 1:
        chop = _apply_stutter(chop, num_samples, stutter_repeats)

    if fade_in_samples > 0 or fade_out_samples > 0:
        chop = _apply_linear_fades(chop, fade_in_samples, fade_out_samples)

    if volume != 1.0:
        chop = _apply_volume(chop, volume)

    out = Path(output_path)
    # Track whether we created the parent dir so we can roll it back on
    # write failure without disturbing a directory the caller already had.
    created_parent = not out.parent.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_sf_write(output_path, chop, sr, subtype=src_info.subtype)
    except BaseException:
        if created_parent:
            # Best-effort rollback. rmdir only succeeds if the dir is empty
            # (the atomic-write helper already removed its temp on failure),
            # which is exactly the safe condition.
            rmdir_ok = True
            try:
                out.parent.rmdir()
            except OSError:
                rmdir_ok = False
            _log_event(
                "trim_range_parent_dir_rollback",
                parent_dir=str(out.parent),
                rmdir_ok=rmdir_ok,
            )
        raise

    channels = 1 if chop.ndim == 1 else int(chop.shape[1])

    return {
        "output": output_path,
        "samples": num_samples,
        "duration_s": num_samples / sr,
        "sample_rate": int(sr),
        "channels": channels,
        "start_sample": start_sample,
        "fade_in_samples": fade_in_samples,
        "fade_out_samples": fade_out_samples,
        "reverse": bool(reverse),
        "stutter_repeats": int(stutter_repeats),
        "volume": float(volume),
    }


def _apply_linear_fades(
    audio: np.ndarray,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    """Apply linear fade envelopes to a PCM buffer.

    Fade-in ramps the first `fade_in_samples` samples from 0 → 1. Fade-out
    ramps the last `fade_out_samples` samples from 1 → 0. Both are
    independent; overlap is prevented by `trim_range`'s caller-side
    validation (`fade_in + fade_out <= duration`).

    Works on mono (1-D) and multi-channel (2-D with shape (n, channels))
    buffers. Returns a new array — does not mutate the input.

    The buffer's dtype is preserved; we multiply in float to avoid overflow
    and then restore the original dtype for the caller. For float inputs
    this is a no-op.
    """
    if fade_in_samples == 0 and fade_out_samples == 0:
        return audio
    if audio.size == 0:
        return audio

    n = audio.shape[0]
    original_dtype = audio.dtype

    # Work in float64 to avoid quantization during the ramp, then cast back.
    work = audio.astype(np.float64, copy=True)

    if fade_in_samples > 0:
        k = min(fade_in_samples, n)
        ramp = np.linspace(0.0, 1.0, num=k, endpoint=False)
        if work.ndim == 1:
            work[:k] *= ramp
        else:
            work[:k] *= ramp[:, np.newaxis]

    if fade_out_samples > 0:
        k = min(fade_out_samples, n)
        ramp = np.linspace(1.0, 0.0, num=k, endpoint=False)
        if work.ndim == 1:
            work[-k:] *= ramp
        else:
            work[-k:] *= ramp[:, np.newaxis]

    # Restore original dtype. For integer PCM types soundfile converts
    # float back to int when writing; for float types no conversion is
    # needed. We use astype to match the input dtype exactly.
    if np.issubdtype(original_dtype, np.floating):
        return work.astype(original_dtype)
    return work.astype(original_dtype)


def _apply_stutter(
    audio: np.ndarray,
    num_samples: int,
    stutter_repeats: int,
) -> np.ndarray:
    """Tile the chop's first (num_samples // N) samples N times to fill
    exactly `num_samples`.

    The "stutter" fill effect: a single short slice of audio is repeated
    N times across the same destination slot, producing classic drum-roll
    stutter. The resulting array is trimmed back to `num_samples` so the
    length invariant holds (tile rounding never adds samples beyond the
    original request).

    Works on mono (1-D) and stereo (2-D, shape (n, channels)) buffers.
    Returns a new array — does not mutate the input.
    """
    if stutter_repeats <= 1:
        return audio
    if audio.size == 0:
        return audio

    slice_samples = num_samples // stutter_repeats
    if slice_samples <= 0:
        # Degenerate case: stutter count exceeds sample budget. Return
        # the unchanged audio rather than producing an empty result — the
        # length invariant matters more than the fill firing.
        return audio

    if audio.ndim == 1:
        one = audio[:slice_samples]
        tiled = np.tile(one, stutter_repeats)
    else:
        one = audio[:slice_samples, :]
        tiled = np.tile(one, (stutter_repeats, 1))

    # Integer division can UNDER-shoot num_samples (e.g. 44100 // 8 = 5512,
    # tiled × 8 = 44096). Pad the tail with zeros to land on exactly
    # num_samples so the length invariant holds for any N.
    if tiled.shape[0] < num_samples:
        pad = num_samples - tiled.shape[0]
        if audio.ndim == 1:
            tiled = np.concatenate([tiled, np.zeros(pad, dtype=audio.dtype)])
        else:
            channels = audio.shape[1]
            tiled = np.concatenate(
                [tiled, np.zeros((pad, channels), dtype=audio.dtype)], axis=0
            )

    # If tile still over-shot (shouldn't, but belt + suspenders) trim back.
    return tiled[:num_samples]


def _apply_volume(audio: np.ndarray, volume: float) -> np.ndarray:
    """Scalar-multiply the audio buffer by `volume` (0..1).

    Works in float64 to avoid integer overflow, then casts back to the
    input dtype. For float inputs the intermediate cast is still done
    but is effectively free (astype is a no-op when dtype matches).
    """
    if volume == 1.0:
        return audio
    if audio.size == 0:
        return audio

    original_dtype = audio.dtype
    work = audio.astype(np.float64, copy=True) * volume
    return work.astype(original_dtype)
