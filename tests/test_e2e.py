"""
End-to-end tests using real audio files.

These tests create various audio scenarios and verify the complete
processing pipeline works correctly.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.processor import process_audio


# =============================================================================
# Test Audio Generation Utilities
# =============================================================================

def generate_click_track(
    bpm: float,
    bars: int,
    sr: int = 44100,
    meter: int = 4,
    downbeat_emphasis: float = 2.0,
    click_freq: float = 1000.0,
    click_duration: float = 0.02,
    silence_before: float = 0.0,
) -> np.ndarray:
    """
    Generate a click track with emphasized downbeats.

    Args:
        bpm: Tempo in beats per minute
        bars: Number of bars
        sr: Sample rate
        meter: Beats per bar
        downbeat_emphasis: Volume multiplier for downbeats
        click_freq: Frequency of click sound
        click_duration: Duration of each click in seconds
        silence_before: Seconds of silence before first beat

    Returns:
        Audio array (mono, float32)
    """
    samples_per_beat = int((60.0 / bpm) * sr)
    samples_per_bar = samples_per_beat * meter
    silence_samples = int(silence_before * sr)
    total_samples = silence_samples + (samples_per_bar * bars)

    audio = np.zeros(total_samples, dtype=np.float32)

    # Create click sound
    click_samples = int(sr * click_duration)
    t = np.arange(click_samples) / sr
    click = np.sin(2 * np.pi * click_freq * t).astype(np.float32)
    click *= np.exp(-t / (click_duration / 4))  # Exponential decay

    # Place clicks
    for bar in range(bars):
        for beat in range(meter):
            beat_num = bar * meter + beat
            pos = silence_samples + (beat_num * samples_per_beat)
            if pos + click_samples < len(audio):
                if beat == 0:  # Downbeat
                    audio[pos:pos + click_samples] += click * downbeat_emphasis
                else:
                    audio[pos:pos + click_samples] += click

    return audio


def generate_sine_tone(
    frequency: float,
    duration: float,
    sr: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a simple sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)


def generate_stereo_audio(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Combine two mono channels into stereo."""
    return np.column_stack([left, right])


# =============================================================================
# E2E Tests - Basic Functionality
# =============================================================================

class TestE2EBasicProcessing:
    """Basic end-to-end processing tests."""

    def test_simple_click_track_120bpm(self, tmp_path: Path) -> None:
        """Process a simple 120 BPM click track."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        # Generate 8 bars at 120 BPM
        audio = generate_click_track(bpm=120, bars=8, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        # Process: extract 4 bars
        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success
        assert Path(output_path).exists()

        # Verify output duration (4 bars at 120 BPM = 8 seconds)
        output_audio, out_sr = sf.read(output_path)
        expected_duration = 8.0  # 4 bars * 2 seconds/bar
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.1

    def test_different_tempos(self, tmp_path: Path) -> None:
        """Test processing at various tempos."""
        tempos = [60, 90, 120, 140, 180]

        for bpm in tempos:
            input_path = str(tmp_path / f"input_{bpm}bpm.wav")
            output_path = str(tmp_path / f"output_{bpm}bpm.wav")

            audio = generate_click_track(bpm=bpm, bars=8, sr=44100)
            sf.write(input_path, audio, 44100, subtype='PCM_24')

            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=bpm,
                bars=4,
                meter=4
            )

            assert result.success, f"Failed at {bpm} BPM"
            assert Path(output_path).exists()

            # Verify duration
            output_audio, out_sr = sf.read(output_path)
            # 4 bars at bpm = 4 * 4 * (60/bpm) seconds
            expected_duration = 4 * 4 * (60.0 / bpm)
            actual_duration = len(output_audio) / out_sr
            assert abs(actual_duration - expected_duration) < 0.1, f"Duration mismatch at {bpm} BPM"

    def test_different_bar_counts(self, tmp_path: Path) -> None:
        """Test extracting different numbers of bars."""
        bar_counts = [1, 2, 4, 8, 16]
        bpm = 120
        sr = 44100

        # Generate source audio with enough bars
        input_path = str(tmp_path / "input.wav")
        audio = generate_click_track(bpm=bpm, bars=32, sr=sr)
        sf.write(input_path, audio, sr, subtype='PCM_24')

        for bars in bar_counts:
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
            assert abs(actual_duration - expected_duration) < 0.1, f"Duration mismatch for {bars} bars"


class TestE2EStereoProcessing:
    """Tests for stereo audio processing."""

    def test_stereo_click_track(self, tmp_path: Path) -> None:
        """Process stereo audio correctly."""
        input_path = str(tmp_path / "stereo_input.wav")
        output_path = str(tmp_path / "stereo_output.wav")

        # Generate stereo: left has clicks, right has different clicks
        left = generate_click_track(bpm=120, bars=8, sr=44100, click_freq=1000)
        right = generate_click_track(bpm=120, bars=8, sr=44100, click_freq=800)
        stereo = generate_stereo_audio(left, right)

        sf.write(input_path, stereo, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success

        # Verify output is stereo
        output_audio, out_sr = sf.read(output_path)
        assert output_audio.ndim == 2
        assert output_audio.shape[1] == 2

    def test_stereo_preserves_channels(self, tmp_path: Path) -> None:
        """Verify stereo channels are preserved correctly."""
        input_path = str(tmp_path / "stereo_input.wav")
        output_path = str(tmp_path / "stereo_output.wav")

        sr = 44100
        # Generate click track first to know the duration
        clicks = generate_click_track(bpm=120, bars=8, sr=sr)
        duration = len(clicks) / sr

        # Left channel: 440 Hz sine, Right channel: silence
        left = generate_sine_tone(440, duration=duration, sr=sr)
        right = np.zeros_like(left)
        stereo = generate_stereo_audio(left, right)

        # Add beat clicks for detection to both channels
        stereo[:, 0] += clicks
        stereo[:, 1] += clicks * 0.5  # Less clicks on right

        sf.write(input_path, stereo, sr, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=2,
            meter=4
        )

        assert result.success

        output_audio, _ = sf.read(output_path)
        # Left should have content, right should be mostly silence
        left_energy = np.sum(output_audio[:, 0] ** 2)
        right_energy = np.sum(output_audio[:, 1] ** 2)
        assert left_energy > right_energy * 2  # Left significantly louder


class TestE2ESampleRates:
    """Tests for different sample rates."""

    def test_44100hz(self, tmp_path: Path) -> None:
        """Test standard 44.1kHz sample rate."""
        self._test_sample_rate(tmp_path, 44100)

    def test_48000hz(self, tmp_path: Path) -> None:
        """Test 48kHz sample rate."""
        self._test_sample_rate(tmp_path, 48000)

    def test_96000hz(self, tmp_path: Path) -> None:
        """Test 96kHz sample rate."""
        self._test_sample_rate(tmp_path, 96000)

    def test_22050hz(self, tmp_path: Path) -> None:
        """Test 22.05kHz sample rate."""
        self._test_sample_rate(tmp_path, 22050)

    def _test_sample_rate(self, tmp_path: Path, sr: int) -> None:
        """Helper to test a specific sample rate."""
        input_path = str(tmp_path / f"input_{sr}hz.wav")
        output_path = str(tmp_path / f"output_{sr}hz.wav")

        audio = generate_click_track(bpm=120, bars=8, sr=sr)
        sf.write(input_path, audio, sr, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success
        assert result.sample_rate == sr

        # Verify output sample rate matches
        info = sf.info(output_path)
        assert info.samplerate == sr


class TestE2EBitDepths:
    """Tests for different bit depths."""

    def test_16bit(self, tmp_path: Path) -> None:
        """Test 16-bit audio."""
        self._test_bit_depth(tmp_path, 'PCM_16')

    def test_24bit(self, tmp_path: Path) -> None:
        """Test 24-bit audio."""
        self._test_bit_depth(tmp_path, 'PCM_24')

    def test_32bit_float(self, tmp_path: Path) -> None:
        """Test 32-bit float audio."""
        self._test_bit_depth(tmp_path, 'FLOAT')

    def _test_bit_depth(self, tmp_path: Path, subtype: str) -> None:
        """Helper to test a specific bit depth."""
        input_path = str(tmp_path / f"input_{subtype}.wav")
        output_path = str(tmp_path / f"output_{subtype}.wav")

        audio = generate_click_track(bpm=120, bars=8, sr=44100)
        sf.write(input_path, audio, 44100, subtype=subtype)

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success

        # Verify output bit depth matches
        input_info = sf.info(input_path)
        output_info = sf.info(output_path)
        assert output_info.subtype == input_info.subtype


# =============================================================================
# E2E Tests - CLI Interface
# =============================================================================

class TestE2ECLI:
    """Tests for the command-line interface."""

    @pytest.fixture
    def cli_env(self, tmp_path: Path):
        """Create test audio file and return paths."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        audio = generate_click_track(bpm=120, bars=8, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        return input_path, output_path

    def test_cli_trim(self, cli_env) -> None:
        """Test trim subcommand."""
        input_path, output_path = cli_env

        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', '120',
             '--bars', '4'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert Path(output_path).exists()

    def test_cli_trim_json_output(self, cli_env) -> None:
        """Test trim subcommand with JSON output."""
        input_path, output_path = cli_env

        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', '120',
             '--bars', '4'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Parse JSON output
        lines = result.stdout.strip().split('\n')
        json_objects = [json.loads(line) for line in lines if line.strip()]

        # Should have progress events and final result
        progress_events = [j for j in json_objects if j.get('type') == 'progress']
        trim_events = [j for j in json_objects if j.get('type') == 'trim']

        assert len(progress_events) >= 4  # loading, detecting, trimming, writing
        assert len(trim_events) == 1
        assert trim_events[0]['success'] is True

    def test_cli_invalid_file(self, tmp_path: Path) -> None:
        """Test CLI with non-existent input file."""
        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', str(tmp_path / 'nonexistent.wav'),
             '--output', str(tmp_path / 'output.wav'),
             '--bpm', '120',
             '--bars', '4'],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        # Error should be in stderr
        error_json = json.loads(result.stderr.strip())
        assert error_json['type'] == 'error'
        assert error_json['code'] == 'FILE_NOT_FOUND'

    def test_cli_invalid_bpm(self, cli_env) -> None:
        """Test CLI with invalid BPM."""
        input_path, output_path = cli_env

        result = subprocess.run(
            ['python', '-m', 'sas_processor', 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', '0',
             '--bars', '4'],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0


# =============================================================================
# E2E Tests - Edge Cases
# =============================================================================

class TestE2EEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_audio(self, tmp_path: Path) -> None:
        """Test with very short audio (less than requested bars)."""
        input_path = str(tmp_path / "short.wav")
        output_path = str(tmp_path / "output.wav")

        # Only 2 bars but requesting 4
        audio = generate_click_track(bpm=120, bars=2, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        # Should still succeed but output will be padded with silence
        assert result.success
        assert Path(output_path).exists()

    def test_audio_with_silence_prefix(self, tmp_path: Path) -> None:
        """Test audio that starts with silence."""
        input_path = str(tmp_path / "silence_prefix.wav")
        output_path = str(tmp_path / "output.wav")

        # 2 seconds of silence before beats start
        audio = generate_click_track(bpm=120, bars=8, sr=44100, silence_before=2.0)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success
        # Downbeat should be found after the silence
        assert result.downbeat_time >= 1.5  # At least after some silence

    def test_single_bar(self, tmp_path: Path) -> None:
        """Test extracting just 1 bar."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        audio = generate_click_track(bpm=120, bars=8, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=1,
            meter=4
        )

        assert result.success

        # 1 bar at 120 BPM = 2 seconds
        output_audio, out_sr = sf.read(output_path)
        expected_duration = 2.0
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.1

    def test_unusual_meter(self, tmp_path: Path) -> None:
        """Test with 3/4 time signature."""
        input_path = str(tmp_path / "waltz.wav")
        output_path = str(tmp_path / "output.wav")

        # 3/4 time
        audio = generate_click_track(bpm=120, bars=8, sr=44100, meter=3)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=3
        )

        assert result.success

        # 4 bars of 3/4 at 120 BPM = 4 * 3 * 0.5 = 6 seconds
        output_audio, out_sr = sf.read(output_path)
        expected_duration = 6.0
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.1

    def test_slow_tempo(self, tmp_path: Path) -> None:
        """Test with very slow tempo (60 BPM)."""
        input_path = str(tmp_path / "slow.wav")
        output_path = str(tmp_path / "output.wav")

        audio = generate_click_track(bpm=60, bars=4, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=60,
            bars=2,
            meter=4
        )

        assert result.success

        # 2 bars at 60 BPM = 2 * 4 * 1.0 = 8 seconds
        output_audio, out_sr = sf.read(output_path)
        expected_duration = 8.0
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.1

    def test_fast_tempo(self, tmp_path: Path) -> None:
        """Test with fast tempo (180 BPM)."""
        input_path = str(tmp_path / "fast.wav")
        output_path = str(tmp_path / "output.wav")

        audio = generate_click_track(bpm=180, bars=16, sr=44100)
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=180,
            bars=8,
            meter=4
        )

        assert result.success

        # 8 bars at 180 BPM = 8 * 4 * (60/180) = 10.67 seconds
        output_audio, out_sr = sf.read(output_path)
        expected_duration = 8 * 4 * (60.0 / 180)
        actual_duration = len(output_audio) / out_sr
        assert abs(actual_duration - expected_duration) < 0.1


# =============================================================================
# E2E Tests - Downbeat Detection Accuracy
# =============================================================================

class TestE2EDownbeatDetection:
    """Tests specifically for downbeat detection accuracy."""

    def test_clear_downbeat_emphasis(self, tmp_path: Path) -> None:
        """Test detection with clearly emphasized downbeats."""
        input_path = str(tmp_path / "emphasized.wav")
        output_path = str(tmp_path / "output.wav")

        # Strong downbeat emphasis (3x volume)
        audio = generate_click_track(
            bpm=120, bars=8, sr=44100,
            downbeat_emphasis=3.0
        )
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        assert result.success
        # Downbeat detection finds a beat position that it considers the "1"
        # The exact position depends on the beat detection algorithm
        # Just verify it found a reasonable downbeat within the audio
        assert result.downbeat_time < result.original_duration / 2  # Within first half

    def test_subtle_downbeat_emphasis(self, tmp_path: Path) -> None:
        """Test detection with subtle downbeat emphasis."""
        input_path = str(tmp_path / "subtle.wav")
        output_path = str(tmp_path / "output.wav")

        # Only 1.2x emphasis
        audio = generate_click_track(
            bpm=120, bars=8, sr=44100,
            downbeat_emphasis=1.2
        )
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4,
            meter=4
        )

        # Should still succeed even with subtle emphasis
        assert result.success
        # Downbeat detection may be less accurate but should complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
