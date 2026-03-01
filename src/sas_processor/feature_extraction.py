"""Feature extraction for audio analysis (BPM detection)."""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    """Extracted audio features."""

    bpm: float
    duration_seconds: float
    sample_rate: int
    channels: int


def extract_features(audio_path: str) -> AudioFeatures:
    """
    Extract audio features from a file (BPM detection).

    Uses librosa.beat.beat_track() with bpm=None for automatic tempo estimation.

    Args:
        audio_path: Path to the audio file to analyze.

    Returns:
        AudioFeatures with detected BPM, duration, sample rate, and channel count.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If audio loading or analysis fails.
    """
    # Load audio file (sr=None preserves original sample rate)
    y, sr = librosa.load(audio_path, sr=None, mono=False)

    # Determine channel count and create mono version for analysis
    if y.ndim > 1:
        y_mono = np.mean(y, axis=0)
        channels = y.shape[0]
    else:
        y_mono = y
        channels = 1

    # Calculate duration
    duration_seconds = len(y_mono) / sr

    # Detect BPM (bpm=None triggers automatic detection)
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr, units="frames")
    # librosa may return scalar or array depending on version
    detected_bpm = float(np.atleast_1d(tempo)[0])

    return AudioFeatures(
        bpm=round(detected_bpm, 1),
        duration_seconds=round(duration_seconds, 3),
        sample_rate=sr,
        channels=channels,
    )
