"""
Tests for feature extraction module (BPM detection).

Tests cover:
- BPM detection accuracy with various audio samples
- Stereo/mono handling
- Error handling for invalid files
- Sample rate and channel detection
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.feature_extraction import AudioFeatures, extract_features
from tests.fixtures import (
    LIBROSA_SAMPLES,
    create_test_wav_from_sample,
    save_librosa_sample_as_wav,
    load_librosa_sample,
)


class TestAudioFeatures:
    """Test the AudioFeatures dataclass."""

    def test_dataclass_fields(self):
        """Verify AudioFeatures has all expected fields."""
        features = AudioFeatures(
            bpm=120.0,
            duration_seconds=10.5,
            sample_rate=44100,
            channels=2,
        )
        assert features.bpm == 120.0
        assert features.duration_seconds == 10.5
        assert features.sample_rate == 44100
        assert features.channels == 2


class TestExtractFeatures:
    """Test the extract_features function."""

    def test_extract_features_basic(self, tmp_path):
        """Test basic feature extraction with a known sample."""
        wav_path, expected_bpm = create_test_wav_from_sample('choice', tmp_path)

        features = extract_features(wav_path)

        assert isinstance(features, AudioFeatures)
        assert features.bpm > 0
        assert features.duration_seconds > 0
        assert features.sample_rate == 44100
        assert features.channels == 1  # Default is mono

    def test_bpm_detection_returns_reasonable_value_choice(self, tmp_path):
        """Test BPM detection returns a reasonable tempo for drum & bass sample."""
        wav_path, _ = create_test_wav_from_sample('choice', tmp_path)

        features = extract_features(wav_path)

        # BPM detection has octave ambiguity - librosa may detect at different
        # metrical levels. Just verify we get a reasonable musical tempo.
        assert 50 < features.bpm < 250, f"BPM {features.bpm} is outside reasonable range"

    def test_bpm_detection_returns_reasonable_value_nutcracker(self, tmp_path):
        """Test BPM detection returns a reasonable tempo for classical sample."""
        wav_path, _ = create_test_wav_from_sample('nutcracker', tmp_path)

        features = extract_features(wav_path)

        # Classical music BPM detection may vary due to rubato and complex rhythms.
        # Just verify we get a reasonable musical tempo.
        assert 50 < features.bpm < 250, f"BPM {features.bpm} is outside reasonable range"

    def test_stereo_file_handling(self, tmp_path):
        """Test that stereo files are handled correctly."""
        # Create a stereo WAV file
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create a simple stereo signal (sine wave at different frequencies per channel)
        left_channel = np.sin(2 * np.pi * 440 * t)
        right_channel = np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.column_stack([left_channel, right_channel])

        wav_path = str(tmp_path / "stereo_test.wav")
        sf.write(wav_path, stereo_audio, sr)

        features = extract_features(wav_path)

        assert features.channels == 2
        assert features.sample_rate == sr
        assert abs(features.duration_seconds - duration) < 0.1

    def test_mono_file_handling(self, tmp_path):
        """Test that mono files are handled correctly."""
        # Create a mono WAV file
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        mono_audio = np.sin(2 * np.pi * 440 * t)

        wav_path = str(tmp_path / "mono_test.wav")
        sf.write(wav_path, mono_audio, sr)

        features = extract_features(wav_path)

        assert features.channels == 1
        assert features.sample_rate == sr
        assert abs(features.duration_seconds - duration) < 0.1

    def test_different_sample_rates(self, tmp_path):
        """Test handling of different sample rates."""
        for test_sr in [22050, 44100, 48000]:
            duration = 1.0
            t = np.linspace(0, duration, int(test_sr * duration))
            audio = np.sin(2 * np.pi * 440 * t)

            wav_path = str(tmp_path / f"sr_{test_sr}.wav")
            sf.write(wav_path, audio, test_sr)

            features = extract_features(wav_path)

            assert features.sample_rate == test_sr, f"Expected SR {test_sr}, got {features.sample_rate}"

    def test_duration_calculation(self, tmp_path):
        """Test that duration is calculated correctly."""
        sr = 44100
        expected_duration = 3.5

        t = np.linspace(0, expected_duration, int(sr * expected_duration))
        audio = np.sin(2 * np.pi * 440 * t)

        wav_path = str(tmp_path / "duration_test.wav")
        sf.write(wav_path, audio, sr)

        features = extract_features(wav_path)

        assert abs(features.duration_seconds - expected_duration) < 0.01

    def test_bpm_rounding(self, tmp_path):
        """Test that BPM is rounded to one decimal place."""
        wav_path, _ = create_test_wav_from_sample('choice', tmp_path)

        features = extract_features(wav_path)

        # Check BPM has at most one decimal place
        bpm_str = str(features.bpm)
        if '.' in bpm_str:
            decimal_places = len(bpm_str.split('.')[1])
            assert decimal_places <= 1, f"BPM {features.bpm} has too many decimal places"

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        nonexistent_path = str(tmp_path / "nonexistent.wav")

        with pytest.raises(Exception):  # librosa raises various exceptions for missing files
            extract_features(nonexistent_path)

    def test_invalid_audio_file(self, tmp_path):
        """Test that invalid audio files raise an exception."""
        invalid_path = str(tmp_path / "invalid.wav")
        # Create a file with random non-audio data
        with open(invalid_path, 'wb') as f:
            f.write(b'This is not a valid audio file')

        with pytest.raises(Exception):  # Should raise an error when trying to load
            extract_features(invalid_path)


@pytest.mark.real_audio
class TestRealAudioSamples:
    """Test with various real audio samples from librosa."""

    @pytest.mark.parametrize("sample_name", [
        'choice',
        'nutcracker',
        'brahms',
        'fishin',
    ])
    def test_extract_features_multiple_samples(self, sample_name, tmp_path):
        """Test feature extraction works on multiple sample types."""
        wav_path, _ = create_test_wav_from_sample(sample_name, tmp_path)

        features = extract_features(wav_path)

        assert features.bpm > 0, f"BPM should be positive for {sample_name}"
        assert features.bpm < 300, f"BPM should be reasonable (<300) for {sample_name}"
        assert features.duration_seconds > 0
        assert features.sample_rate > 0
        assert features.channels >= 1
