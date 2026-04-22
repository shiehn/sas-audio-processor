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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

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

    Returns:
        Dict with `output`, `samples`, `duration_s`, `sample_rate`,
        `channels`, `start_sample`, `fade_in_samples`, `fade_out_samples`.
        Suitable for emit_json.

    Raises:
        ValueError  on invalid bpm / meter / duration / fade args.
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

    if fade_in_samples > 0 or fade_out_samples > 0:
        chop = _apply_linear_fades(chop, fade_in_samples, fade_out_samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, chop, sr, subtype=src_info.subtype)

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
