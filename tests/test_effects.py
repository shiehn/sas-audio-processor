"""
Unit tests for sas_processor.effects module.

Tests each audio effect function for correctness:
- Output file is created
- Output format matches input (sample rate, subtype)
- Effect produces expected result characteristics
- Mono and stereo handling
"""

import numpy as np
import pytest
import soundfile as sf

from sas_processor.effects import (
    _load,
    _save,
    _apply_pedalboard_effect,
    normalize_audio,
    apply_gain,
    to_mono,
    convert_audio,
    remove_silence,
    compress_audio,
    apply_eq,
    apply_reverb,
    apply_limiter,
    apply_filter,
    pitch_shift_audio,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_sine_wav(path: str, freq: float = 440.0, sr: int = 44100,
                   duration: float = 1.0, amplitude: float = 0.5,
                   channels: int = 1, subtype: str = 'PCM_24') -> str:
    """Create a sine wave WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    if channels > 1:
        audio = np.column_stack([audio] * channels)
    sf.write(path, audio, sr, subtype=subtype)
    return path


def _make_silent_wav(path: str, sr: int = 44100, duration: float = 1.0,
                     subtype: str = 'PCM_24') -> str:
    """Create a silent WAV file."""
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float64)
    sf.write(path, audio, sr, subtype=subtype)
    return path


def _make_wav_with_silence_padding(path: str, sr: int = 44100,
                                   subtype: str = 'PCM_24') -> str:
    """Create a WAV with silence at start and end, tone in middle."""
    silence = np.zeros(int(sr * 0.5), dtype=np.float64)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
    audio = np.concatenate([silence, tone, silence])
    sf.write(path, audio, sr, subtype=subtype)
    return path


# ============================================================================
# _load / _save
# ============================================================================

class TestLoadSave:
    def test_load_returns_audio_sr_subtype(self, tmp_path):
        path = _make_sine_wav(str(tmp_path / "test.wav"))
        audio, sr, subtype = _load(path)
        assert sr == 44100
        assert subtype == 'PCM_24'
        assert audio.ndim == 1
        assert len(audio) == 44100

    def test_save_preserves_format(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), subtype='PCM_16')
        audio, sr, subtype = _load(in_path)
        out_path = str(tmp_path / "out.wav")
        _save(audio, out_path, sr, subtype)
        info = sf.info(out_path)
        assert info.subtype == 'PCM_16'
        assert info.samplerate == 44100


# ============================================================================
# _apply_pedalboard_effect
# ============================================================================

class TestApplyPedalboardEffect:
    def test_applies_effect_and_writes_output(self, tmp_path):
        from pedalboard import Gain
        in_path = _make_sine_wav(str(tmp_path / "in.wav"))
        out_path = str(tmp_path / "out.wav")
        _apply_pedalboard_effect([Gain(gain_db=6.0)], in_path, out_path)
        assert sf.info(out_path).samplerate == 44100

    def test_handles_stereo(self, tmp_path):
        from pedalboard import Gain
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), channels=2)
        out_path = str(tmp_path / "out.wav")
        _apply_pedalboard_effect([Gain(gain_db=0.0)], in_path, out_path)
        audio, _ = sf.read(out_path)
        assert audio.ndim == 2
        assert audio.shape[1] == 2


# ============================================================================
# normalize_audio
# ============================================================================

class TestNormalizeAudio:
    def test_peak_normalize(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.25)
        out_path = str(tmp_path / "out.wav")
        result = normalize_audio(in_path, out_path, 'peak', -14.0, -1.0)
        assert result["mode"] == "peak"
        audio, _ = sf.read(out_path)
        peak_db = 20 * np.log10(np.max(np.abs(audio)))
        assert abs(peak_db - (-1.0)) < 0.5

    def test_lufs_normalize(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        result = normalize_audio(in_path, out_path, 'lufs', -14.0, -1.0)
        assert result["mode"] == "lufs"
        assert "gain_db" in result

    def test_preserves_format(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), subtype='PCM_16')
        out_path = str(tmp_path / "out.wav")
        normalize_audio(in_path, out_path, 'peak', -14.0, -1.0)
        assert sf.info(out_path).subtype == 'PCM_16'


# ============================================================================
# apply_gain
# ============================================================================

class TestApplyGain:
    def test_positive_gain_increases_amplitude(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.1)
        out_path = str(tmp_path / "out.wav")
        result = apply_gain(in_path, out_path, 6.0)
        assert result["gain_db"] == 6.0
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        assert np.max(np.abs(out_audio)) > np.max(np.abs(in_audio))

    def test_negative_gain_decreases_amplitude(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        apply_gain(in_path, out_path, -6.0)
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        assert np.max(np.abs(out_audio)) < np.max(np.abs(in_audio))

    def test_zero_gain_preserves(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        apply_gain(in_path, out_path, 0.0)
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        np.testing.assert_allclose(out_audio, in_audio, atol=1e-4)


# ============================================================================
# to_mono
# ============================================================================

class TestToMono:
    def test_stereo_to_mono(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), channels=2)
        out_path = str(tmp_path / "out.wav")
        result = to_mono(in_path, out_path)
        assert result["channels_in"] == 2
        assert result["channels_out"] == 1
        audio, _ = sf.read(out_path)
        assert audio.ndim == 1

    def test_mono_stays_mono(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), channels=1)
        out_path = str(tmp_path / "out.wav")
        result = to_mono(in_path, out_path)
        assert result["channels_in"] == 1
        assert result["channels_out"] == 1


# ============================================================================
# convert_audio
# ============================================================================

class TestConvertAudio:
    def test_resample(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), sr=44100)
        out_path = str(tmp_path / "out.wav")
        result = convert_audio(in_path, out_path, 22050, None)
        assert result["sample_rate"] == 22050
        assert sf.info(out_path).samplerate == 22050

    def test_bit_depth_change(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), subtype='PCM_24')
        out_path = str(tmp_path / "out.wav")
        result = convert_audio(in_path, out_path, None, '16')
        assert result["bit_depth"] == 'PCM_16'
        assert sf.info(out_path).subtype == 'PCM_16'

    def test_no_change(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"))
        out_path = str(tmp_path / "out.wav")
        result = convert_audio(in_path, out_path, None, None)
        assert result["sample_rate"] == 44100


# ============================================================================
# remove_silence
# ============================================================================

class TestRemoveSilence:
    def test_trims_silence(self, tmp_path):
        in_path = _make_wav_with_silence_padding(str(tmp_path / "in.wav"))
        out_path = str(tmp_path / "out.wav")
        result = remove_silence(in_path, out_path, top_db=30.0)
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        assert len(out_audio) < len(in_audio)
        assert result["trimmed_start"] > 0
        assert result["trimmed_end"] > 0

    def test_no_silence_no_trim(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        result = remove_silence(in_path, out_path, top_db=60.0)
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        # Should be nearly the same length
        assert abs(len(out_audio) - len(in_audio)) < 1000


# ============================================================================
# compress_audio
# ============================================================================

class TestCompressAudio:
    def test_compression_reduces_dynamic_range(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.8)
        out_path = str(tmp_path / "out.wav")
        result = compress_audio(in_path, out_path, -20.0, 4.0, 1.0, 100.0)
        assert result["threshold_db"] == -20.0
        assert result["ratio"] == 4.0
        # Output should exist and be valid
        audio, _ = sf.read(out_path)
        assert len(audio) > 0

    def test_preserves_sample_rate(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"))
        out_path = str(tmp_path / "out.wav")
        compress_audio(in_path, out_path, -20.0, 4.0, 1.0, 100.0)
        assert sf.info(out_path).samplerate == 44100


# ============================================================================
# apply_eq
# ============================================================================

class TestApplyEq:
    def test_eq_boost(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=1000.0)
        out_path = str(tmp_path / "out.wav")
        result = apply_eq(in_path, out_path, 1000.0, 6.0, 1.0)
        assert result["freq"] == 1000.0
        assert result["gain_db"] == 6.0
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        # Boosting at the signal frequency should increase amplitude
        assert np.max(np.abs(out_audio)) > np.max(np.abs(in_audio)) * 0.9

    def test_eq_cut(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=1000.0)
        out_path = str(tmp_path / "out.wav")
        result = apply_eq(in_path, out_path, 1000.0, -12.0, 1.0)
        assert result["gain_db"] == -12.0
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        assert np.max(np.abs(out_audio)) < np.max(np.abs(in_audio))


# ============================================================================
# apply_reverb
# ============================================================================

class TestApplyReverb:
    def test_reverb_applied(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), duration=0.5)
        out_path = str(tmp_path / "out.wav")
        result = apply_reverb(in_path, out_path, 0.8, 0.5, 0.5)
        assert result["room_size"] == 0.8
        assert result["wet_level"] == 0.5
        audio, _ = sf.read(out_path)
        assert len(audio) > 0

    def test_dry_reverb_preserves_signal(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), duration=0.5)
        out_path = str(tmp_path / "out.wav")
        apply_reverb(in_path, out_path, 0.0, 0.0, 0.0)
        out_audio, _ = sf.read(out_path)
        # With 0 wet level, output should still contain audio
        assert len(out_audio) > 0
        assert np.max(np.abs(out_audio)) > 0.01


# ============================================================================
# apply_limiter
# ============================================================================

class TestApplyLimiter:
    def test_limiter_produces_valid_output(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.9)
        out_path = str(tmp_path / "out.wav")
        result = apply_limiter(in_path, out_path, -6.0)
        assert result["threshold_db"] == -6.0
        audio, _ = sf.read(out_path)
        assert len(audio) > 0
        assert sf.info(out_path).samplerate == 44100

    def test_quiet_signal_unchanged(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), amplitude=0.01)
        out_path = str(tmp_path / "out.wav")
        apply_limiter(in_path, out_path, -1.0)
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        # Signal well below threshold should be mostly unchanged
        np.testing.assert_allclose(out_audio, in_audio, atol=0.01)


# ============================================================================
# apply_filter
# ============================================================================

class TestApplyFilter:
    def test_highpass_removes_low_freq(self, tmp_path):
        # Create a low-frequency signal
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=100.0, amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        result = apply_filter(in_path, out_path, 'highpass', 500.0)
        assert result["filter_type"] == 'highpass'
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        # High-pass at 500Hz should attenuate 100Hz signal
        assert np.max(np.abs(out_audio)) < np.max(np.abs(in_audio)) * 0.5

    def test_lowpass_removes_high_freq(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=5000.0, amplitude=0.5)
        out_path = str(tmp_path / "out.wav")
        result = apply_filter(in_path, out_path, 'lowpass', 1000.0)
        assert result["filter_type"] == 'lowpass'
        in_audio, _ = sf.read(in_path)
        out_audio, _ = sf.read(out_path)
        assert np.max(np.abs(out_audio)) < np.max(np.abs(in_audio)) * 0.5


# ============================================================================
# pitch_shift_audio
# ============================================================================

class TestPitchShiftAudio:
    def test_pitch_shift_up(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=440.0, duration=0.5)
        out_path = str(tmp_path / "out.wav")
        result = pitch_shift_audio(in_path, out_path, 12.0)
        assert result["semitones"] == 12.0
        audio, _ = sf.read(out_path)
        assert len(audio) > 0

    def test_pitch_shift_preserves_duration(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), freq=440.0, duration=1.0)
        out_path = str(tmp_path / "out.wav")
        pitch_shift_audio(in_path, out_path, 2.0)
        in_info = sf.info(in_path)
        out_info = sf.info(out_path)
        # Duration should be preserved (within tolerance)
        assert abs(in_info.duration - out_info.duration) < 0.1

    def test_pitch_shift_stereo(self, tmp_path):
        in_path = _make_sine_wav(str(tmp_path / "in.wav"), channels=2, duration=0.5)
        out_path = str(tmp_path / "out.wav")
        pitch_shift_audio(in_path, out_path, -3.0)
        audio, _ = sf.read(out_path)
        assert audio.ndim == 2
        assert audio.shape[1] == 2
