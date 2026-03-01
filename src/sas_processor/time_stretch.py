"""Time-stretching audio to match a target BPM using librosa."""

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


@dataclass
class TimeStretchResult:
    """Result of a time-stretch operation."""
    success: bool
    output_path: str
    source_bpm: float
    target_bpm: float
    rate: float
    duration_seconds: float
    error: Optional[str] = None
    error_code: Optional[str] = None


def time_stretch_audio(
    input_path: str,
    output_path: str,
    source_bpm: float,
    target_bpm: float,
) -> TimeStretchResult:
    """
    Time-stretch audio to match a target BPM using librosa's phase vocoder.

    For ±10 BPM changes (~5-10%), quality is very good.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output WAV file
        source_bpm: Original BPM of the audio
        target_bpm: Desired BPM after stretching

    Returns:
        TimeStretchResult with operation details
    """
    error_result = TimeStretchResult(
        success=False,
        output_path=output_path,
        source_bpm=source_bpm,
        target_bpm=target_bpm,
        rate=0.0,
        duration_seconds=0.0,
    )

    # Validate BPM values
    if source_bpm <= 0 or source_bpm > 999:
        error_result.error = f"Invalid source BPM: {source_bpm}"
        error_result.error_code = "INVALID_SOURCE_BPM"
        return error_result

    if target_bpm <= 0 or target_bpm > 999:
        error_result.error = f"Invalid target BPM: {target_bpm}"
        error_result.error_code = "INVALID_TARGET_BPM"
        return error_result

    rate = target_bpm / source_bpm

    # Sanity check: don't allow extreme stretching
    if rate < 0.5 or rate > 2.0:
        error_result.error = f"Stretch rate {rate:.2f} is too extreme (must be 0.5-2.0)"
        error_result.error_code = "EXTREME_STRETCH"
        return error_result

    try:
        # Load audio preserving original sample rate and bit depth info
        info = sf.info(input_path)
        audio, sr = sf.read(input_path, dtype='float64')
        subtype = info.subtype

        # Handle both mono and stereo
        if audio.ndim == 1:
            # Mono: librosa expects (samples,) for mono
            stretched = librosa.effects.time_stretch(audio, rate=rate)
        else:
            # Stereo: stretch each channel independently
            channels = []
            for ch in range(audio.shape[1]):
                stretched_ch = librosa.effects.time_stretch(audio[:, ch], rate=rate)
                channels.append(stretched_ch)
            stretched = np.column_stack(channels)

        # Calculate output duration
        if stretched.ndim == 1:
            duration_seconds = len(stretched) / sr
        else:
            duration_seconds = stretched.shape[0] / sr

        # Write output preserving original format
        sf.write(output_path, stretched, sr, subtype=subtype)

        return TimeStretchResult(
            success=True,
            output_path=output_path,
            source_bpm=source_bpm,
            target_bpm=target_bpm,
            rate=rate,
            duration_seconds=duration_seconds,
        )

    except sf.SoundFileError as e:
        error_result.error = f"Audio file error: {e}"
        error_result.error_code = "AUDIO_FILE_ERROR"
        return error_result
    except MemoryError:
        error_result.error = "Out of memory - audio file may be too large"
        error_result.error_code = "MEMORY_ERROR"
        return error_result
    except Exception as e:
        error_result.error = f"Time stretch failed: {str(e)}"
        error_result.error_code = "STRETCH_ERROR"
        return error_result
