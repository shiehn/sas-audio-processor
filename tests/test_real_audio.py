"""
End-to-end tests using real audio files from librosa's sample library.

These tests validate that the audio processor works correctly with actual
music files across various genres, not just synthetic click tracks.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.processor import process_audio
from tests.fixtures import (
    LIBROSA_SAMPLES,
    TempWavFile,
    create_test_wav_from_sample,
    estimate_bpm,
    get_samples_with_clear_beats,
    load_librosa_sample,
)


# =============================================================================
# Tests by Genre
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioGenres:
    """Test processing of real audio files across different genres."""

    def test_classical_nutcracker(self, nutcracker_wav, output_wav):
        """Test processing classical music (Tchaikovsky)."""
        input_path, bpm = nutcracker_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success, f"Failed to process nutcracker: {result.error}"
        assert Path(output_wav).exists()

        # Verify output is valid audio
        output_audio, out_sr = sf.read(output_wav)
        assert len(output_audio) > 0
        assert out_sr == 44100

        # Verify duration is approximately correct (4 bars)
        expected_duration = 4 * 4 * (60.0 / bpm)
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.5, \
            f"Duration mismatch: expected {expected_duration:.2f}s, got {actual_duration:.2f}s"

    def test_electronic_choice(self, choice_wav, output_wav):
        """Test processing drum & bass / electronic music."""
        input_path, bpm = choice_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success, f"Failed to process choice: {result.error}"
        assert Path(output_wav).exists()

        # Verify output
        output_audio, out_sr = sf.read(output_wav)
        expected_duration = 4 * 4 * (60.0 / bpm)
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.5

    def test_electronic_vibeace(self, vibeace_wav, output_wav):
        """Test processing synth/electronic music."""
        input_path, bpm = vibeace_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success, f"Failed to process vibeace: {result.error}"
        assert Path(output_wav).exists()

    def test_classical_brahms(self, brahms_wav, output_wav):
        """Test processing classical dance music (Brahms)."""
        input_path, bpm = brahms_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success, f"Failed to process brahms: {result.error}"
        assert Path(output_wav).exists()

    def test_folk_fishin(self, fishin_wav, output_wav):
        """Test processing folk music with vocals."""
        input_path, bpm = fishin_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success, f"Failed to process fishin: {result.error}"
        assert Path(output_wav).exists()

    def test_instrumental_trumpet(self, trumpet_wav, output_wav):
        """Test processing instrumental music (trumpet)."""
        input_path, bpm = trumpet_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=2,  # Shorter since trumpet sample is small
            meter=4
        )

        assert result.success, f"Failed to process trumpet: {result.error}"
        assert Path(output_wav).exists()

    def test_stereo_audio_preservation(self, tmp_path, output_wav):
        """Test that stereo audio is preserved through processing."""
        # Create a synthetic stereo test file
        sr = 44100
        duration = 10.0
        bpm = 120.0

        # Create stereo audio with different content in each channel
        t = np.linspace(0, duration, int(sr * duration))
        left_channel = np.sin(2 * np.pi * 440 * t)  # 440 Hz
        right_channel = np.sin(2 * np.pi * 880 * t)  # 880 Hz
        stereo_audio = np.column_stack([left_channel, right_channel])

        input_path = str(tmp_path / "stereo_test.wav")
        sf.write(input_path, stereo_audio, sr, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=2,
            meter=4
        )

        assert result.success, f"Failed to process stereo audio: {result.error}"
        assert Path(output_wav).exists()

        # Verify stereo is preserved
        output_audio, _ = sf.read(output_wav)
        assert output_audio.ndim == 2, "Stereo should be preserved"
        assert output_audio.shape[1] == 2, "Should have 2 channels"


# =============================================================================
# Process All Samples
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioAllSamples:
    """Test that all librosa samples can be processed successfully."""

    @pytest.mark.parametrize("sample_name", get_samples_with_clear_beats())
    def test_process_sample(self, sample_name, tmp_path):
        """Test processing each librosa sample with clear beats."""
        input_path, bpm = create_test_wav_from_sample(sample_name, tmp_path)
        output_path = str(tmp_path / "output.wav")

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=bpm,
            bars=2,  # Use 2 bars to keep test fast
            meter=4
        )

        assert result.success, f"Failed to process {sample_name}: {result.error}"
        assert Path(output_path).exists(), f"Output file not created for {sample_name}"

        # Verify output is valid audio
        output_audio, out_sr = sf.read(output_path)
        assert len(output_audio) > 0, f"Output audio is empty for {sample_name}"


# =============================================================================
# Sample Rate and Bit Depth Tests
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioFormats:
    """Test format preservation with real audio files."""

    def test_sample_rate_preservation(self, tmp_path):
        """Test that sample rate is preserved."""
        # Test with different sample rates
        for sr in [22050, 44100, 48000]:
            input_path, bpm = create_test_wav_from_sample(
                'nutcracker', tmp_path,
                filename=f"input_{sr}.wav",
                sr=sr
            )
            output_path = str(tmp_path / f"output_{sr}.wav")

            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=2,
                meter=4
            )

            assert result.success
            assert result.sample_rate == sr

            # Verify output sample rate
            info = sf.info(output_path)
            assert info.samplerate == sr, f"Sample rate not preserved for {sr}Hz"

    def test_bit_depth_preservation(self, tmp_path):
        """Test that bit depth is preserved."""
        for subtype in ['PCM_16', 'PCM_24', 'FLOAT']:
            input_path, bpm = create_test_wav_from_sample(
                'nutcracker', tmp_path,
                filename=f"input_{subtype}.wav",
                subtype=subtype
            )
            output_path = str(tmp_path / f"output_{subtype}.wav")

            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=2,
                meter=4
            )

            assert result.success

            # Verify output bit depth
            input_info = sf.info(input_path)
            output_info = sf.info(output_path)
            assert output_info.subtype == input_info.subtype, \
                f"Bit depth not preserved for {subtype}"


# =============================================================================
# Edge Cases with Real Audio
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioEdgeCases:
    """Test edge cases using real audio files."""

    def test_with_offset(self, tmp_path):
        """Test processing audio starting from an offset."""
        # Load sample with 2 second offset
        input_path, bpm = create_test_wav_from_sample(
            'nutcracker', tmp_path,
            offset=2.0,
            duration=10.0
        )
        output_path = str(tmp_path / "output.wav")

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=bpm,
            bars=2,
            meter=4
        )

        assert result.success

    def test_short_excerpt(self, tmp_path):
        """Test processing a very short audio excerpt."""
        input_path, bpm = create_test_wav_from_sample(
            'choice', tmp_path,
            duration=3.0  # Only 3 seconds
        )
        output_path = str(tmp_path / "output.wav")

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=bpm,
            bars=1,
            meter=4
        )

        assert result.success

    def test_different_bar_counts(self, tmp_path):
        """Test extracting different numbers of bars from real audio."""
        input_path, bpm = create_test_wav_from_sample('choice', tmp_path)

        for bars in [1, 2, 4, 8]:
            output_path = str(tmp_path / f"output_{bars}bars.wav")

            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=bars,
                meter=4
            )

            assert result.success, f"Failed for {bars} bars"

            # Verify duration
            output_audio, out_sr = sf.read(output_path)
            expected_duration = bars * 4 * (60.0 / bpm)
            actual_duration = len(output_audio) / out_sr
            assert abs(actual_duration - expected_duration) < 0.5, \
                f"Duration mismatch for {bars} bars"


# =============================================================================
# Downbeat Detection Accuracy
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioDownbeatAccuracy:
    """Test downbeat detection accuracy with real audio."""

    def test_downbeat_is_within_audio(self, nutcracker_wav, output_wav):
        """Test that detected downbeat is within the audio bounds."""
        input_path, bpm = nutcracker_wav

        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert result.success
        assert result.downbeat_time >= 0
        assert result.downbeat_time < result.original_duration

    def test_downbeat_consistency(self, tmp_path):
        """Test that downbeat detection is consistent across runs."""
        input_path, bpm = create_test_wav_from_sample('choice', tmp_path)

        downbeat_times = []
        for i in range(3):
            output_path = str(tmp_path / f"output_{i}.wav")
            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=4,
                meter=4
            )
            assert result.success
            downbeat_times.append(result.downbeat_time)

        # All downbeat times should be the same
        assert all(t == downbeat_times[0] for t in downbeat_times), \
            f"Inconsistent downbeat times: {downbeat_times}"


# =============================================================================
# CLI Tests with Real Audio
# =============================================================================

@pytest.mark.real_audio
class TestRealAudioCLI:
    """Test CLI with real audio files."""

    def test_cli_with_nutcracker(self, nutcracker_wav, output_wav):
        """Test CLI processing of classical music."""
        input_path, bpm = nutcracker_wav

        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', input_path,
             '--output', output_wav,
             '--bpm', str(bpm),
             '--bars', '4'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert Path(output_wav).exists()

        # Parse JSON output
        lines = result.stdout.strip().split('\n')
        json_objects = [json.loads(line) for line in lines if line.strip()]

        trim_events = [j for j in json_objects if j.get('type') == 'trim']
        assert len(trim_events) == 1
        assert trim_events[0]['success'] is True

    def test_cli_with_electronic(self, choice_wav, output_wav):
        """Test CLI processing of electronic music."""
        input_path, bpm = choice_wav

        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', input_path,
             '--output', output_wav,
             '--bpm', str(bpm),
             '--bars', '4',
             '--verbose'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert Path(output_wav).exists()


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.real_audio
@pytest.mark.slow
class TestRealAudioPerformance:
    """Performance tests with real audio (marked as slow)."""

    def test_processing_time(self, nutcracker_wav, output_wav):
        """Test that processing completes in reasonable time."""
        import time

        input_path, bpm = nutcracker_wav

        start = time.time()
        result = process_audio(
            input_path=input_path,
            output_path=output_wav,
            bpm=bpm,
            bars=8,
            meter=4
        )
        elapsed = time.time() - start

        assert result.success
        # Should complete in under 10 seconds for normal audio
        assert elapsed < 10.0, f"Processing took too long: {elapsed:.2f}s"

    def test_multiple_samples_sequentially(self, tmp_path):
        """Test processing multiple samples sequentially."""
        samples = get_samples_with_clear_beats()

        for sample_name in samples:
            input_path, bpm = create_test_wav_from_sample(sample_name, tmp_path)
            output_path = str(tmp_path / f"output_{sample_name}.wav")

            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=2,
                meter=4
            )

            assert result.success, f"Failed for {sample_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
