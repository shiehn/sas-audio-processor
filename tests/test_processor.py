"""Tests for SAS Audio Processor."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.beat_detection import detect_beats, find_downbeat, get_downbeat_sample
from sas_processor.processor import (
    calculate_bar_samples,
    load_audio,
    process_audio,
    save_audio,
    trim_audio,
)


def create_test_wav(path: str, duration: float = 5.0, sr: int = 44100) -> None:
    """Create a test WAV file with a simple sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Create a 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr, subtype='PCM_24')


def create_test_wav_with_beats(
    path: str,
    bpm: float = 120.0,
    bars: int = 4,
    sr: int = 44100
) -> None:
    """Create a test WAV file with audible beat clicks."""
    samples_per_beat = int((60.0 / bpm) * sr)
    samples_per_bar = samples_per_beat * 4
    total_samples = samples_per_bar * bars

    audio = np.zeros(total_samples, dtype=np.float32)

    # Add click at each beat
    click_duration = int(sr * 0.01)  # 10ms click
    click = np.sin(2 * np.pi * 1000 * np.arange(click_duration) / sr)
    click *= np.exp(-np.arange(click_duration) / (click_duration / 4))  # Decay

    for beat in range(bars * 4):
        pos = beat * samples_per_beat
        if pos + click_duration < len(audio):
            # Make downbeats louder
            if beat % 4 == 0:
                audio[pos:pos + click_duration] += click * 1.0
            else:
                audio[pos:pos + click_duration] += click * 0.5

    sf.write(path, audio, sr, subtype='PCM_24')


class TestLoadSaveAudio:
    """Tests for audio loading and saving."""

    def test_load_audio(self, tmp_path: Path) -> None:
        """Test loading a WAV file."""
        wav_path = str(tmp_path / "test.wav")
        create_test_wav(wav_path)

        audio, sr, subtype = load_audio(wav_path)

        assert sr == 44100
        assert len(audio) > 0
        assert subtype == 'PCM_24'

    def test_save_audio(self, tmp_path: Path) -> None:
        """Test saving a WAV file."""
        wav_path = str(tmp_path / "output.wav")
        audio = np.zeros(44100, dtype=np.float64)
        save_audio(audio, wav_path, 44100, 'PCM_24')

        assert Path(wav_path).exists()
        loaded, sr = sf.read(wav_path)
        assert sr == 44100
        assert len(loaded) == 44100


class TestBeatDetection:
    """Tests for beat detection functions."""

    def test_detect_beats_returns_samples(self, tmp_path: Path) -> None:
        """Test that detect_beats returns sample positions."""
        wav_path = str(tmp_path / "beats.wav")
        create_test_wav_with_beats(wav_path, bpm=120, bars=4)

        audio, sr = sf.read(wav_path)
        beats = detect_beats(audio, sr, bpm=120)

        assert len(beats) > 0
        assert all(isinstance(b, (int, np.integer)) for b in beats)

    def test_get_downbeat_sample(self, tmp_path: Path) -> None:
        """Test getting the downbeat position."""
        wav_path = str(tmp_path / "beats.wav")
        create_test_wav_with_beats(wav_path, bpm=120, bars=4)

        audio, sr = sf.read(wav_path)
        downbeat_sample, beats = get_downbeat_sample(audio, sr, bpm=120)

        assert isinstance(downbeat_sample, int)
        assert downbeat_sample >= 0


class TestTrimAudio:
    """Tests for audio trimming."""

    def test_trim_mono(self) -> None:
        """Test trimming mono audio."""
        audio = np.arange(1000, dtype=np.float64)
        trimmed = trim_audio(audio, 100, 200)

        assert len(trimmed) == 200
        assert trimmed[0] == 100
        assert trimmed[-1] == 299

    def test_trim_stereo(self) -> None:
        """Test trimming stereo audio."""
        audio = np.column_stack([
            np.arange(1000, dtype=np.float64),
            np.arange(1000, 2000, dtype=np.float64)
        ])
        trimmed = trim_audio(audio, 100, 200)

        assert trimmed.shape == (200, 2)
        assert trimmed[0, 0] == 100
        assert trimmed[0, 1] == 1100

    def test_trim_with_padding(self) -> None:
        """Test trimming beyond audio length pads with zeros."""
        audio = np.ones(100, dtype=np.float64)
        trimmed = trim_audio(audio, 50, 100)

        assert len(trimmed) == 100
        assert trimmed[49] == 1.0
        assert trimmed[50] == 0.0


class TestCalculateBarSamples:
    """Tests for bar sample calculation."""

    def test_calculate_120bpm_44100hz(self) -> None:
        """Test sample calculation at 120 BPM, 44100 Hz."""
        # At 120 BPM, one beat = 0.5s = 22050 samples
        # One bar (4 beats) = 2s = 88200 samples
        samples = calculate_bar_samples(sr=44100, bpm=120, bars=1)
        assert samples == 88200

    def test_calculate_multiple_bars(self) -> None:
        """Test sample calculation for multiple bars."""
        samples_1 = calculate_bar_samples(sr=44100, bpm=120, bars=1)
        samples_4 = calculate_bar_samples(sr=44100, bpm=120, bars=4)
        assert samples_4 == samples_1 * 4


class TestProcessAudio:
    """Integration tests for the full processing pipeline."""

    def test_process_audio_success(self, tmp_path: Path) -> None:
        """Test full audio processing pipeline."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        create_test_wav_with_beats(input_path, bpm=120, bars=8)

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert result.success
        assert Path(output_path).exists()
        assert result.downbeat_time >= 0
        assert result.output_duration > 0

    def test_process_audio_file_not_found(self, tmp_path: Path) -> None:
        """Test processing with non-existent file."""
        result = process_audio(
            input_path=str(tmp_path / "nonexistent.wav"),
            output_path=str(tmp_path / "output.wav"),
            bpm=120,
            bars=4
        )

        assert not result.success
        assert "not found" in result.error.lower()
        assert result.error_code == "INPUT_ERROR"

    def test_process_audio_preserves_sample_rate(self, tmp_path: Path) -> None:
        """Test that processing preserves the original sample rate."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        # Create at 48kHz
        t = np.linspace(0, 2, 48000 * 2, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(input_path, audio, 48000, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=1
        )

        assert result.success
        assert result.sample_rate == 48000

        # Verify output sample rate
        info = sf.info(output_path)
        assert info.samplerate == 48000


class TestProcessAudioIOErrors:
    """Tests for I/O error handling in process_audio."""

    def test_input_file_not_readable(self, tmp_path: Path) -> None:
        """Test error when input file exists but is not readable."""
        import os
        import stat

        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")

        create_test_wav(input_path)

        # Remove read permission
        os.chmod(input_path, 0o000)

        try:
            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=120,
                bars=4
            )

            assert not result.success
            assert "permission" in result.error.lower() or "readable" in result.error.lower()
            assert result.error_code == "INPUT_ERROR"
        finally:
            # Restore permissions for cleanup
            os.chmod(input_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_input_file_empty(self, tmp_path: Path) -> None:
        """Test error when input file is empty."""
        input_path = str(tmp_path / "empty.wav")
        output_path = str(tmp_path / "output.wav")

        # Create empty file
        Path(input_path).touch()

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert not result.success
        assert result.error_code == "INPUT_ERROR"

    def test_input_not_wav_file(self, tmp_path: Path) -> None:
        """Test error when input file is not a WAV."""
        input_path = str(tmp_path / "input.mp3")
        output_path = str(tmp_path / "output.wav")

        # Create a fake file with wrong extension
        Path(input_path).write_text("not a wav file")

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert not result.success
        assert "wav" in result.error.lower() or "supported" in result.error.lower()
        assert result.error_code == "INPUT_ERROR"

    def test_output_directory_not_exists(self, tmp_path: Path) -> None:
        """Test error when output directory doesn't exist."""
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "nonexistent_dir" / "output.wav")

        create_test_wav(input_path)

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert not result.success
        assert "directory" in result.error.lower() or "not exist" in result.error.lower()
        assert result.error_code == "OUTPUT_ERROR"

    def test_output_directory_not_writable(self, tmp_path: Path) -> None:
        """Test error when output directory is not writable."""
        import os
        import stat

        input_path = str(tmp_path / "input.wav")
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        output_path = str(readonly_dir / "output.wav")

        create_test_wav(input_path)

        # Make directory read-only
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            result = process_audio(
                input_path=input_path,
                output_path=output_path,
                bpm=120,
                bars=4
            )

            assert not result.success
            assert "permission" in result.error.lower() or "writable" in result.error.lower()
            assert result.error_code == "OUTPUT_ERROR"
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, stat.S_IRWXU)

    def test_corrupted_wav_file(self, tmp_path: Path) -> None:
        """Test error when WAV file is corrupted."""
        input_path = str(tmp_path / "corrupted.wav")
        output_path = str(tmp_path / "output.wav")

        # Create a file with .wav extension but invalid content
        Path(input_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt corrupted data")

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert not result.success
        assert result.error_code == "READ_ERROR"

    def test_very_short_audio(self, tmp_path: Path) -> None:
        """Test error when audio file is too short."""
        input_path = str(tmp_path / "short.wav")
        output_path = str(tmp_path / "output.wav")

        # Create a very short audio file (less than 0.1 seconds)
        audio = np.zeros(1000, dtype=np.float32)  # ~22ms at 44100Hz
        sf.write(input_path, audio, 44100, subtype='PCM_24')

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=120,
            bars=4
        )

        assert not result.success
        assert "too short" in result.error.lower()
        assert result.error_code == "INPUT_ERROR"

    def test_error_code_propagated(self, tmp_path: Path) -> None:
        """Test that error codes are properly set for different errors."""
        output_path = str(tmp_path / "output.wav")

        # Test INPUT_ERROR
        result = process_audio(
            input_path="/nonexistent/file.wav",
            output_path=output_path,
            bpm=120,
            bars=4
        )
        assert result.error_code == "INPUT_ERROR"

        # Test OUTPUT_ERROR
        input_path = str(tmp_path / "input.wav")
        create_test_wav(input_path)

        result = process_audio(
            input_path=input_path,
            output_path="/nonexistent_dir/output.wav",
            bpm=120,
            bars=4
        )
        assert result.error_code == "OUTPUT_ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
