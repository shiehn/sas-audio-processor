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
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from sas_processor.processor import trim_audio


def trim_range(
    input_path: str,
    output_path: str,
    bpm: float,
    meter: int,
    start_beat: float,
    duration_beats: float,
) -> dict:
    """Write a chop of `duration_beats` starting at `start_beat` from `input_path`.

    Args:
        input_path:     Source WAV file.
        output_path:    Destination WAV file (parent directory must exist).
        bpm:            Tempo of the source stem (must match how it was rendered).
        meter:          Beats per bar (informational — kept for API symmetry
                        with split_audio_bars; `trim_range` itself only uses bpm).
        start_beat:     Beat offset (0-indexed) from the start of the input.
        duration_beats: Length of the chop, in beats. For v1 this is 1, 2, or 4.

    Returns:
        Dict with `output`, `samples`, `duration_s`, `sample_rate`,
        `channels`, `start_sample`. Suitable for emit_json.

    Raises:
        ValueError  on invalid bpm / meter / duration.
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

    chop = trim_audio(audio, start_sample, num_samples)

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
    }
