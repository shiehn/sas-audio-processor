"""
Pytest configuration and shared fixtures for SAS Audio Processor tests.
"""

import os
from pathlib import Path

import pytest

from tests.fixtures import (
    LIBROSA_SAMPLES,
    create_test_wav_from_sample,
    get_all_sample_names,
    get_samples_with_clear_beats,
)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "real_audio: tests that use real audio files from librosa"
    )
    config.addinivalue_line(
        "markers", "slow: tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "binary: tests that require the compiled PyInstaller binary"
    )


@pytest.fixture
def binary_path():
    """
    Get path to the compiled sas-processor binary.

    Returns None if the binary doesn't exist.
    """
    project_root = Path(__file__).parent.parent
    binary = project_root / "dist" / "sas-processor" / "sas-processor"

    if binary.exists():
        return str(binary)
    return None


@pytest.fixture
def sample_names():
    """Get list of all available librosa sample names."""
    return get_all_sample_names()


@pytest.fixture
def samples_with_beats():
    """Get list of samples that have clear rhythmic beats."""
    return get_samples_with_clear_beats()


@pytest.fixture
def sample_metadata():
    """Get metadata dictionary for all librosa samples."""
    return LIBROSA_SAMPLES


@pytest.fixture
def nutcracker_wav(tmp_path):
    """Create a WAV file from the nutcracker sample."""
    wav_path, bpm = create_test_wav_from_sample('nutcracker', tmp_path)
    return wav_path, bpm


@pytest.fixture
def choice_wav(tmp_path):
    """Create a WAV file from the choice (drum & bass) sample."""
    wav_path, bpm = create_test_wav_from_sample('choice', tmp_path)
    return wav_path, bpm


@pytest.fixture
def vibeace_wav(tmp_path):
    """Create a WAV file from the vibeace (electronic) sample."""
    wav_path, bpm = create_test_wav_from_sample('vibeace', tmp_path)
    return wav_path, bpm


@pytest.fixture
def brahms_wav(tmp_path):
    """Create a WAV file from the brahms (classical) sample."""
    wav_path, bpm = create_test_wav_from_sample('brahms', tmp_path)
    return wav_path, bpm


@pytest.fixture
def trumpet_wav(tmp_path):
    """Create a WAV file from the trumpet (stereo) sample."""
    wav_path, bpm = create_test_wav_from_sample('trumpet', tmp_path, mono=False)
    return wav_path, bpm


@pytest.fixture
def fishin_wav(tmp_path):
    """Create a WAV file from the fishin (folk) sample."""
    wav_path, bpm = create_test_wav_from_sample('fishin', tmp_path)
    return wav_path, bpm


@pytest.fixture
def output_wav(tmp_path):
    """Get a path for output WAV file."""
    return str(tmp_path / "output.wav")
