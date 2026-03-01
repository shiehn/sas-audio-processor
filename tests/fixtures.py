"""
Test fixtures for loading real audio samples from librosa.

This module provides utilities to load librosa's built-in sample audio files
and convert them to WAV format for testing the audio processor.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


@dataclass
class SampleMetadata:
    """Metadata for a librosa sample audio file."""
    name: str
    genre: str
    description: str
    has_clear_beats: bool
    estimated_bpm: Optional[float] = None  # None means we'll estimate it
    is_stereo: bool = False


# Metadata for librosa's built-in sample files
LIBROSA_SAMPLES = {
    'nutcracker': SampleMetadata(
        name='nutcracker',
        genre='classical',
        description='Tchaikovsky - Dance of the Sugar Plum Fairy',
        has_clear_beats=True,
        estimated_bpm=150.0,  # Approximate
        is_stereo=False,
    ),
    'choice': SampleMetadata(
        name='choice',
        genre='electronic',
        description='Admiral Bob - Choice (Drum & Bass)',
        has_clear_beats=True,
        estimated_bpm=170.0,  # Drum & Bass typically 160-180
        is_stereo=False,
    ),
    'vibeace': SampleMetadata(
        name='vibeace',
        genre='electronic',
        description='Kevin MacLeod - Vibe Ace (Synth)',
        has_clear_beats=True,
        estimated_bpm=None,  # Will estimate
        is_stereo=False,
    ),
    'brahms': SampleMetadata(
        name='brahms',
        genre='classical',
        description='Brahms - Hungarian Dance #5',
        has_clear_beats=True,
        estimated_bpm=100.0,  # Approximate
        is_stereo=False,
    ),
    'trumpet': SampleMetadata(
        name='trumpet',
        genre='instrumental',
        description='Mihai Sorohan - Trumpet Loop',
        has_clear_beats=False,  # Solo instrument, less clear beats
        estimated_bpm=None,
        is_stereo=False,  # Actually mono despite the name
    ),
    'fishin': SampleMetadata(
        name='fishin',
        genre='folk',
        description="Karissa Hobbs - Let's Go Fishin'",
        has_clear_beats=True,
        estimated_bpm=None,
        is_stereo=False,
    ),
}


def get_librosa_sample_path(name: str) -> str:
    """
    Get the path to a librosa sample file.

    Downloads the sample if not already cached.

    Args:
        name: Name of the librosa sample (e.g., 'nutcracker', 'choice')

    Returns:
        Path to the audio file
    """
    return librosa.ex(name)


def load_librosa_sample(
    name: str,
    sr: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Load a librosa sample audio file.

    Args:
        name: Name of the librosa sample
        sr: Target sample rate (None = native rate)
        mono: Convert to mono if True
        duration: Only load this many seconds
        offset: Start reading at this time (seconds)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    path = get_librosa_sample_path(name)
    return librosa.load(path, sr=sr, mono=mono, duration=duration, offset=offset)


def save_librosa_sample_as_wav(
    name: str,
    output_path: str,
    sr: int = 44100,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
    subtype: str = 'PCM_24',
) -> str:
    """
    Load a librosa sample and save it as a WAV file.

    Args:
        name: Name of the librosa sample
        output_path: Path to save the WAV file
        sr: Target sample rate
        mono: Convert to mono if True
        duration: Only load this many seconds
        offset: Start reading at this time (seconds)
        subtype: WAV subtype (e.g., 'PCM_16', 'PCM_24', 'FLOAT')

    Returns:
        Path to the saved WAV file
    """
    audio, sample_rate = load_librosa_sample(
        name, sr=sr, mono=mono, duration=duration, offset=offset
    )

    sf.write(output_path, audio, sample_rate, subtype=subtype)
    return output_path


def estimate_bpm(audio: np.ndarray, sr: int) -> float:
    """
    Estimate the BPM of an audio signal using librosa.

    Args:
        audio: Audio samples
        sr: Sample rate

    Returns:
        Estimated BPM
    """
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # Handle both scalar and array returns
    if hasattr(tempo, '__len__'):
        return float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)


def get_sample_with_estimated_bpm(name: str, sr: int = 44100) -> tuple[np.ndarray, int, float]:
    """
    Load a librosa sample and estimate its BPM.

    Args:
        name: Name of the librosa sample
        sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate, estimated_bpm)
    """
    audio, sample_rate = load_librosa_sample(name, sr=sr)

    metadata = LIBROSA_SAMPLES.get(name)
    if metadata and metadata.estimated_bpm:
        bpm = metadata.estimated_bpm
    else:
        bpm = estimate_bpm(audio, sample_rate)

    return audio, sample_rate, bpm


class TempWavFile:
    """
    Context manager for creating temporary WAV files from librosa samples.

    Usage:
        with TempWavFile('nutcracker') as wav_path:
            result = process_audio(wav_path, ...)
    """

    def __init__(
        self,
        sample_name: str,
        sr: int = 44100,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
        subtype: str = 'PCM_24',
    ):
        self.sample_name = sample_name
        self.sr = sr
        self.mono = mono
        self.duration = duration
        self.offset = offset
        self.subtype = subtype
        self._temp_dir = None
        self._wav_path = None

    def __enter__(self) -> str:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._wav_path = str(Path(self._temp_dir.name) / f"{self.sample_name}.wav")

        save_librosa_sample_as_wav(
            self.sample_name,
            self._wav_path,
            sr=self.sr,
            mono=self.mono,
            duration=self.duration,
            offset=self.offset,
            subtype=self.subtype,
        )

        return self._wav_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._temp_dir:
            self._temp_dir.cleanup()
        return False


def create_test_wav_from_sample(
    sample_name: str,
    tmp_path: Path,
    filename: Optional[str] = None,
    **kwargs,
) -> tuple[str, float]:
    """
    Create a test WAV file from a librosa sample in a pytest tmp_path.

    Args:
        sample_name: Name of the librosa sample
        tmp_path: pytest tmp_path fixture
        filename: Optional custom filename (default: {sample_name}.wav)
        **kwargs: Additional arguments to save_librosa_sample_as_wav

    Returns:
        Tuple of (wav_path, estimated_bpm)
    """
    if filename is None:
        filename = f"{sample_name}.wav"

    wav_path = str(tmp_path / filename)

    # Get default kwargs
    sr = kwargs.pop('sr', 44100)

    save_librosa_sample_as_wav(sample_name, wav_path, sr=sr, **kwargs)

    # Get BPM
    metadata = LIBROSA_SAMPLES.get(sample_name)
    if metadata and metadata.estimated_bpm:
        bpm = metadata.estimated_bpm
    else:
        audio, _ = load_librosa_sample(sample_name, sr=sr)
        bpm = estimate_bpm(audio, sr)

    return wav_path, bpm


def get_all_sample_names() -> list[str]:
    """Get list of all available librosa sample names."""
    return list(LIBROSA_SAMPLES.keys())


def get_samples_with_clear_beats() -> list[str]:
    """Get list of samples that have clear rhythmic beats."""
    return [
        name for name, meta in LIBROSA_SAMPLES.items()
        if meta.has_clear_beats
    ]
