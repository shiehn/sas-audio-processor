"""Core audio processing: loading, trimming, and saving."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from sas_processor.beat_detection import get_downbeat_sample


# Minimum free disk space required (10 MB as safety buffer)
MIN_FREE_SPACE_BYTES = 10 * 1024 * 1024


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    success: bool
    output_path: str
    downbeat_time: float
    original_duration: float
    output_duration: float
    sample_rate: int
    error: Optional[str] = None
    error_code: Optional[str] = None


def _make_error_result(output_path: str, error: str, error_code: str) -> ProcessingResult:
    """Create a failed ProcessingResult with the given error."""
    return ProcessingResult(
        success=False,
        output_path=output_path,
        downbeat_time=0,
        original_duration=0,
        output_duration=0,
        sample_rate=0,
        error=error,
        error_code=error_code
    )


def _check_input_file(input_path: str) -> Optional[str]:
    """
    Validate input file exists and is readable.

    Returns:
        Error message if validation fails, None if OK.
    """
    input_file = Path(input_path)

    if not input_file.exists():
        return f"Input file not found: {input_path}"

    if not input_file.is_file():
        return f"Input path is not a file: {input_path}"

    if not os.access(input_path, os.R_OK):
        return f"Input file is not readable (permission denied): {input_path}"

    if input_file.stat().st_size == 0:
        return f"Input file is empty: {input_path}"

    if not input_file.suffix.lower() == '.wav':
        return f"Only WAV files are supported, got: {input_file.suffix}"

    return None


def _check_output_path(output_path: str, estimated_size_bytes: int = 0) -> Optional[str]:
    """
    Validate output path is writable and has sufficient disk space.

    Args:
        output_path: Path where output will be written
        estimated_size_bytes: Estimated output file size (0 to skip size check)

    Returns:
        Error message if validation fails, None if OK.
    """
    output_file = Path(output_path)
    output_dir = output_file.parent

    # Check output directory exists
    if not output_dir.exists():
        return f"Output directory does not exist: {output_dir}"

    if not output_dir.is_dir():
        return f"Output parent path is not a directory: {output_dir}"

    # Check write permission on directory
    if not os.access(output_dir, os.W_OK):
        return f"Output directory is not writable (permission denied): {output_dir}"

    # Check if output file already exists and is writable
    if output_file.exists():
        if not os.access(output_path, os.W_OK):
            return f"Output file exists but is not writable (permission denied): {output_path}"

    # Check available disk space
    try:
        disk_usage = shutil.disk_usage(output_dir)
        required_space = max(estimated_size_bytes, MIN_FREE_SPACE_BYTES)
        if disk_usage.free < required_space:
            free_mb = disk_usage.free / (1024 * 1024)
            required_mb = required_space / (1024 * 1024)
            return f"Insufficient disk space: {free_mb:.1f} MB free, need {required_mb:.1f} MB"
    except OSError as e:
        return f"Unable to check disk space: {e}"

    return None


def _verify_output_written(output_path: str) -> Optional[str]:
    """
    Verify output file was written successfully.

    Returns:
        Error message if verification fails, None if OK.
    """
    output_file = Path(output_path)

    if not output_file.exists():
        return f"Output file was not created: {output_path}"

    if output_file.stat().st_size == 0:
        return f"Output file is empty (write may have failed): {output_path}"

    # Try to read the file info to verify it's valid audio
    try:
        info = sf.info(output_path)
        if info.frames == 0:
            return f"Output file has no audio frames: {output_path}"
    except Exception as e:
        return f"Output file is not valid audio: {e}"

    return None


def load_audio(input_path: str) -> tuple[np.ndarray, int, str]:
    """
    Load audio file preserving original sample rate and format.

    Args:
        input_path: Path to input WAV file

    Returns:
        Tuple of (audio_data, sample_rate, subtype)
        subtype indicates bit depth (e.g., 'PCM_16', 'PCM_24', 'FLOAT')
    """
    info = sf.info(input_path)
    audio, sr = sf.read(input_path, dtype='float64')
    return audio, sr, info.subtype


def save_audio(audio: np.ndarray, output_path: str, sr: int, subtype: str) -> None:
    """
    Save audio file preserving original format.

    Args:
        audio: Audio data
        output_path: Path to output WAV file
        sr: Sample rate
        subtype: Original subtype (bit depth)
    """
    sf.write(output_path, audio, sr, subtype=subtype)


def calculate_bar_samples(sr: int, bpm: float, bars: int, meter: int = 4) -> int:
    """
    Calculate the number of samples for a given number of bars.

    Args:
        sr: Sample rate
        bpm: Beats per minute
        bars: Number of bars
        meter: Beats per bar (default 4)

    Returns:
        Number of samples
    """
    samples_per_beat = (60.0 / bpm) * sr
    samples_per_bar = samples_per_beat * meter
    return int(samples_per_bar * bars)


def trim_audio(
    audio: np.ndarray,
    start_sample: int,
    num_samples: int
) -> np.ndarray:
    """
    Trim audio to specified range.

    Args:
        audio: Audio data (1D or 2D)
        start_sample: Start position in samples
        num_samples: Number of samples to extract

    Returns:
        Trimmed audio
    """
    end_sample = start_sample + num_samples

    # Handle both mono and stereo
    if audio.ndim == 1:
        # Mono
        if end_sample > len(audio):
            # Pad with zeros if needed
            result = np.zeros(num_samples, dtype=audio.dtype)
            available = len(audio) - start_sample
            if available > 0:
                result[:available] = audio[start_sample:start_sample + available]
            return result
        return audio[start_sample:end_sample]
    else:
        # Stereo (samples, channels)
        if end_sample > audio.shape[0]:
            # Pad with zeros if needed
            result = np.zeros((num_samples, audio.shape[1]), dtype=audio.dtype)
            available = audio.shape[0] - start_sample
            if available > 0:
                result[:available] = audio[start_sample:start_sample + available]
            return result
        return audio[start_sample:end_sample]


def _estimate_output_size(num_samples: int, channels: int, bytes_per_sample: int = 3) -> int:
    """Estimate output file size in bytes (default assumes 24-bit audio)."""
    # WAV header is ~44 bytes, plus audio data
    return 44 + (num_samples * channels * bytes_per_sample)


def process_audio(
    input_path: str,
    output_path: str,
    bpm: float,
    bars: int,
    meter: int = 4,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> ProcessingResult:
    """
    Process audio file: detect downbeat and trim to specified bars.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output WAV file
        bpm: Known BPM of the audio
        bars: Number of bars to extract
        meter: Beats per bar (default 4 for 4/4 time)
        progress_callback: Optional callback(stage, percent)

    Returns:
        ProcessingResult with details of the operation
    """
    def report_progress(stage: str, percent: int) -> None:
        if progress_callback:
            progress_callback(stage, percent)

    try:
        # Validate input file
        report_progress("validating", 0)
        input_error = _check_input_file(input_path)
        if input_error:
            return _make_error_result(output_path, input_error, "INPUT_ERROR")

        # Initial output path validation (before we know the exact size)
        output_error = _check_output_path(output_path, estimated_size_bytes=0)
        if output_error:
            return _make_error_result(output_path, output_error, "OUTPUT_ERROR")

        # Load audio
        report_progress("loading", 10)
        try:
            audio, sr, subtype = load_audio(input_path)
        except sf.SoundFileError as e:
            return _make_error_result(
                output_path,
                f"Failed to read audio file (may be corrupted or invalid format): {e}",
                "READ_ERROR"
            )

        original_duration = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        # Validate audio has content
        if original_duration < 0.1:
            return _make_error_result(
                output_path,
                f"Audio file is too short: {original_duration:.3f}s",
                "INPUT_ERROR"
            )

        report_progress("detecting", 30)
        # Detect downbeat
        try:
            downbeat_sample, beat_samples = get_downbeat_sample(audio, sr, bpm, meter)
        except Exception as e:
            return _make_error_result(
                output_path,
                f"Beat detection failed: {e}",
                "DETECTION_ERROR"
            )
        downbeat_time = downbeat_sample / sr

        # Calculate how many samples we need
        report_progress("trimming", 60)
        num_samples = calculate_bar_samples(sr, bpm, bars, meter)

        # Check disk space with estimated output size
        estimated_size = _estimate_output_size(num_samples, channels)
        output_error = _check_output_path(output_path, estimated_size_bytes=estimated_size)
        if output_error:
            return _make_error_result(output_path, output_error, "OUTPUT_ERROR")

        # Trim audio starting at downbeat
        trimmed = trim_audio(audio, downbeat_sample, num_samples)
        output_duration = len(trimmed) / sr if trimmed.ndim == 1 else trimmed.shape[0] / sr

        # Save output
        report_progress("writing", 80)
        try:
            save_audio(trimmed, output_path, sr, subtype)
        except sf.SoundFileError as e:
            return _make_error_result(
                output_path,
                f"Failed to write output file: {e}",
                "WRITE_ERROR"
            )
        except OSError as e:
            return _make_error_result(
                output_path,
                f"I/O error writing output file: {e}",
                "WRITE_ERROR"
            )

        # Verify output was written successfully
        report_progress("verifying", 95)
        verify_error = _verify_output_written(output_path)
        if verify_error:
            return _make_error_result(output_path, verify_error, "VERIFY_ERROR")

        report_progress("complete", 100)

        return ProcessingResult(
            success=True,
            output_path=output_path,
            downbeat_time=downbeat_time,
            original_duration=original_duration,
            output_duration=output_duration,
            sample_rate=sr
        )

    except MemoryError:
        return _make_error_result(
            output_path,
            "Out of memory - audio file may be too large",
            "MEMORY_ERROR"
        )
    except KeyboardInterrupt:
        return _make_error_result(
            output_path,
            "Processing was interrupted",
            "INTERRUPTED"
        )
    except Exception as e:
        return _make_error_result(
            output_path,
            f"Unexpected error: {str(e)}",
            "UNEXPECTED_ERROR"
        )
