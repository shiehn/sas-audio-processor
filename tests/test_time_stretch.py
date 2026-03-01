"""Tests for time-stretching functionality."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.time_stretch import time_stretch_audio, TimeStretchResult
from tests.fixtures import create_test_wav_from_sample


class TestTimeStretchAudio:
    """Tests for the time_stretch_audio function."""

    def test_stretch_faster(self, tmp_path):
        """Stretching to a faster BPM should produce shorter audio."""
        wav_path, bpm = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        source_bpm = 100.0
        target_bpm = 110.0

        result = time_stretch_audio(wav_path, output_path, source_bpm, target_bpm)

        assert result.success is True
        assert result.rate == pytest.approx(1.1, abs=0.001)
        assert Path(result.output_path).exists()

        # Verify output is shorter than input (faster tempo = shorter)
        original_info = sf.info(wav_path)
        stretched_info = sf.info(output_path)
        assert stretched_info.duration < original_info.duration

    def test_stretch_slower(self, tmp_path):
        """Stretching to a slower BPM should produce longer audio."""
        wav_path, bpm = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        source_bpm = 100.0
        target_bpm = 90.0

        result = time_stretch_audio(wav_path, output_path, source_bpm, target_bpm)

        assert result.success is True
        assert result.rate == pytest.approx(0.9, abs=0.001)
        assert Path(result.output_path).exists()

        # Verify output is longer than input (slower tempo = longer)
        original_info = sf.info(wav_path)
        stretched_info = sf.info(output_path)
        assert stretched_info.duration > original_info.duration

    def test_stretch_preserves_sample_rate(self, tmp_path):
        """Time-stretching should preserve the original sample rate."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        result = time_stretch_audio(wav_path, output_path, 100.0, 105.0)

        assert result.success is True
        original_info = sf.info(wav_path)
        stretched_info = sf.info(output_path)
        assert stretched_info.samplerate == original_info.samplerate

    def test_stretch_stereo(self, tmp_path):
        """Time-stretching should handle stereo audio."""
        # Create a stereo WAV file
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)
        stereo_audio = np.random.randn(samples, 2).astype(np.float64) * 0.1
        wav_path = str(tmp_path / "stereo.wav")
        sf.write(wav_path, stereo_audio, sr, subtype='PCM_24')

        output_path = str(tmp_path / "stretched.wav")
        result = time_stretch_audio(wav_path, output_path, 120.0, 130.0)

        assert result.success is True
        stretched_info = sf.info(output_path)
        assert stretched_info.channels == 2

    def test_rate_calculation(self, tmp_path):
        """Rate should be target_bpm / source_bpm."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        result = time_stretch_audio(wav_path, output_path, 100.0, 120.0)

        assert result.success is True
        assert result.rate == pytest.approx(1.2, abs=0.001)
        assert result.source_bpm == 100.0
        assert result.target_bpm == 120.0

    def test_duration_matches_rate(self, tmp_path):
        """Output duration should be approximately input_duration / rate."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        source_bpm = 100.0
        target_bpm = 110.0
        rate = target_bpm / source_bpm

        result = time_stretch_audio(wav_path, output_path, source_bpm, target_bpm)

        assert result.success is True
        original_info = sf.info(wav_path)
        expected_duration = original_info.duration / rate
        assert result.duration_seconds == pytest.approx(expected_duration, rel=0.05)

    def test_invalid_source_bpm(self, tmp_path):
        """Invalid source BPM should return error."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        result = time_stretch_audio(wav_path, output_path, 0.0, 120.0)
        assert result.success is False
        assert result.error_code == "INVALID_SOURCE_BPM"

        result = time_stretch_audio(wav_path, output_path, -10.0, 120.0)
        assert result.success is False
        assert result.error_code == "INVALID_SOURCE_BPM"

    def test_invalid_target_bpm(self, tmp_path):
        """Invalid target BPM should return error."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        result = time_stretch_audio(wav_path, output_path, 120.0, 0.0)
        assert result.success is False
        assert result.error_code == "INVALID_TARGET_BPM"

    def test_extreme_stretch_rejected(self, tmp_path):
        """Extreme stretch ratios should be rejected."""
        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        # rate = 300/100 = 3.0 (too extreme)
        result = time_stretch_audio(wav_path, output_path, 100.0, 300.0)
        assert result.success is False
        assert result.error_code == "EXTREME_STRETCH"

    def test_file_not_found(self, tmp_path):
        """Non-existent input file should return error."""
        output_path = str(tmp_path / "stretched.wav")
        result = time_stretch_audio("/nonexistent/file.wav", output_path, 100.0, 110.0)
        assert result.success is False


@pytest.mark.binary
class TestTimeStretchCLI:
    """Tests for the --time-stretch CLI mode."""

    def test_cli_time_stretch_json(self, binary_path, tmp_path):
        """CLI --time-stretch with --json should return proper JSON."""
        if binary_path is None:
            pytest.skip("Binary not available")

        wav_path, _ = create_test_wav_from_sample('brahms', tmp_path)
        output_path = str(tmp_path / "stretched.wav")

        result = subprocess.run(
            [binary_path, '--time-stretch', '--json',
             '--input', wav_path, '--output', output_path,
             '--source-bpm', '100', '--target-bpm', '110'],
            capture_output=True, text=True, timeout=60,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout.strip().split('\n')[-1])
        assert data['type'] == 'time_stretch'
        assert data['success'] is True
        assert data['source_bpm'] == 100.0
        assert data['target_bpm'] == 110.0
        assert 'duration_seconds' in data
        assert 'rate' in data
        assert Path(data['output']).exists()

    def test_cli_version_bumped(self, binary_path):
        """CLI --ping should return version 1.1.0."""
        if binary_path is None:
            pytest.skip("Binary not available")

        result = subprocess.run(
            [binary_path, '--ping'],
            capture_output=True, text=True, timeout=10,
        )

        assert result.returncode == 0
        data = json.loads(result.stdout.strip())
        assert data['version'] == '1.1.0'
