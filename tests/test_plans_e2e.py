"""
End-to-end tests for every CLI subcommand (one per DeclarAgent plan).

Each test invokes `python -m sas_processor <subcommand>` as a subprocess,
simulating how DeclarAgent plans call the CLI.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# =============================================================================
# Helpers
# =============================================================================

def _run_cli(*args: str, expect_success: bool = True) -> dict:
    """Run sas-processor CLI and return parsed JSON result."""
    result = subprocess.run(
        [sys.executable, '-m', 'sas_processor', *args],
        capture_output=True,
        text=True,
    )
    if expect_success:
        assert result.returncode == 0, (
            f"CLI failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Return last JSON line from stdout (the result)
        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        assert lines, f"No output from CLI. stderr: {result.stderr}"
        return json.loads(lines[-1])
    else:
        assert result.returncode != 0
        return json.loads(result.stderr.strip()) if result.stderr.strip() else {}


def _make_wav(path: str, duration: float = 2.0, sr: int = 44100,
              freq: float = 440.0, stereo: bool = False) -> str:
    """Create a test WAV file with a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    if stereo:
        audio = np.column_stack([audio, audio * 0.8])
    sf.write(path, audio, sr, subtype='PCM_24')
    return path


def _make_click_track(path: str, bpm: float = 120.0, bars: int = 8,
                      sr: int = 44100) -> str:
    """Create a test WAV with beat clicks."""
    samples_per_beat = int((60.0 / bpm) * sr)
    total_samples = samples_per_beat * 4 * bars
    audio = np.zeros(total_samples, dtype=np.float32)

    click_dur = int(sr * 0.01)
    click = np.sin(2 * np.pi * 1000 * np.arange(click_dur) / sr).astype(np.float32)
    click *= np.exp(-np.arange(click_dur, dtype=np.float32) / (click_dur / 4))

    for beat in range(bars * 4):
        pos = beat * samples_per_beat
        if pos + click_dur < len(audio):
            amp = 1.0 if beat % 4 == 0 else 0.5
            audio[pos:pos + click_dur] += click * amp

    sf.write(path, audio, sr, subtype='PCM_24')
    return path


# =============================================================================
# Plan: ping
# =============================================================================

class TestPingPlan:
    def test_ping(self) -> None:
        result = _run_cli('ping')
        assert result['type'] == 'ping'
        assert result['success'] is True
        assert 'version' in result


# =============================================================================
# Plan: analyze
# =============================================================================

class TestAnalyzePlan:
    def test_analyze(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=8)
        result = _run_cli('analyze', '--input', wav)
        assert result['type'] == 'analyze'
        assert result['success'] is True
        assert 'bpm' in result
        assert result['bpm'] > 0
        assert 'duration_seconds' in result
        assert 'sample_rate' in result
        assert 'channels' in result


# =============================================================================
# Plan: trim
# =============================================================================

class TestTrimPlan:
    def test_trim(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=8)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('trim', '--input', wav, '--output', out,
                          '--bpm', '120', '--bars', '4')
        assert result['type'] == 'trim'
        assert result['success'] is True
        assert Path(out).exists()
        # 4 bars at 120 BPM = 8 seconds
        info = sf.info(out)
        assert abs(info.duration - 8.0) < 0.5

    def test_trim_invalid_bpm(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('trim', '--input', wav, '--output', out,
                          '--bpm', '0', '--bars', '4', expect_success=False)
        assert result.get('code') == 'INVALID_BPM'


# =============================================================================
# Plan: time-stretch
# =============================================================================

class TestTimeStretchPlan:
    def test_time_stretch(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('time-stretch', '--input', wav, '--output', out,
                          '--source-bpm', '120', '--target-bpm', '130')
        assert result['type'] == 'time-stretch'
        assert result['success'] is True
        assert Path(out).exists()


# =============================================================================
# Plan: normalize
# =============================================================================

class TestNormalizePlan:
    def test_normalize_lufs(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('normalize', '--input', wav, '--output', out,
                          '--mode', 'lufs', '--target-lufs', '-14')
        assert result['type'] == 'normalize'
        assert result['success'] is True
        assert Path(out).exists()
        assert result['mode'] == 'lufs'

    def test_normalize_peak(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('normalize', '--input', wav, '--output', out,
                          '--mode', 'peak', '--target-peak', '-1')
        assert result['type'] == 'normalize'
        assert result['success'] is True
        assert result['mode'] == 'peak'


# =============================================================================
# Plan: gain
# =============================================================================

class TestGainPlan:
    def test_gain_positive(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('gain', '--input', wav, '--output', out, '--db', '3')
        assert result['type'] == 'gain'
        assert result['success'] is True
        assert result['gain_db'] == 3.0

    def test_gain_negative(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('gain', '--input', wav, '--output', out, '--db', '-6')
        assert result['success'] is True
        assert result['gain_db'] == -6.0


# =============================================================================
# Plan: mono
# =============================================================================

class TestMonoPlan:
    def test_stereo_to_mono(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), stereo=True)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('mono', '--input', wav, '--output', out)
        assert result['type'] == 'mono'
        assert result['success'] is True
        assert result['channels_in'] == 2
        assert result['channels_out'] == 1
        # Verify output is actually mono
        audio, _ = sf.read(out)
        assert audio.ndim == 1

    def test_mono_to_mono(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), stereo=False)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('mono', '--input', wav, '--output', out)
        assert result['success'] is True
        assert result['channels_in'] == 1


# =============================================================================
# Plan: convert
# =============================================================================

class TestConvertPlan:
    def test_convert_sample_rate(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), sr=44100)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('convert', '--input', wav, '--output', out,
                          '--sample-rate', '48000')
        assert result['type'] == 'convert'
        assert result['success'] is True
        assert result['sample_rate'] == 48000
        info = sf.info(out)
        assert info.samplerate == 48000

    def test_convert_bit_depth(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('convert', '--input', wav, '--output', out,
                          '--bit-depth', '16')
        assert result['success'] is True
        info = sf.info(out)
        assert info.subtype == 'PCM_16'


# =============================================================================
# Plan: silence-remove
# =============================================================================

class TestSilenceRemovePlan:
    def test_silence_remove(self, tmp_path: Path) -> None:
        # Create audio with silence at start and end
        sr = 44100
        silence = np.zeros(sr, dtype=np.float32)  # 1 second silence
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
        audio = np.concatenate([silence, tone, silence])
        wav = str(tmp_path / 'in.wav')
        sf.write(wav, audio, sr, subtype='PCM_24')

        out = str(tmp_path / 'out.wav')
        result = _run_cli('silence-remove', '--input', wav, '--output', out)
        assert result['type'] == 'silence-remove'
        assert result['success'] is True
        # Output should be shorter than input
        out_audio, _ = sf.read(out)
        assert len(out_audio) < len(audio)


# =============================================================================
# Plan: compress
# =============================================================================

class TestCompressPlan:
    def test_compress(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('compress', '--input', wav, '--output', out,
                          '--threshold', '-20', '--ratio', '4')
        assert result['type'] == 'compress'
        assert result['success'] is True
        assert Path(out).exists()
        assert result['threshold_db'] == -20.0
        assert result['ratio'] == 4.0


# =============================================================================
# Plan: eq
# =============================================================================

class TestEqPlan:
    def test_eq_boost(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('eq', '--input', wav, '--output', out,
                          '--freq', '1000', '--gain-db', '6', '--q', '1.0')
        assert result['type'] == 'eq'
        assert result['success'] is True
        assert result['freq'] == 1000.0
        assert result['gain_db'] == 6.0

    def test_eq_cut(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('eq', '--input', wav, '--output', out,
                          '--freq', '500', '--gain-db', '-6', '--q', '2.0')
        assert result['success'] is True
        assert result['gain_db'] == -6.0


# =============================================================================
# Plan: reverb
# =============================================================================

class TestReverbPlan:
    def test_reverb(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('reverb', '--input', wav, '--output', out,
                          '--room-size', '0.7', '--wet-level', '0.3')
        assert result['type'] == 'reverb'
        assert result['success'] is True
        assert Path(out).exists()


# =============================================================================
# Plan: limit
# =============================================================================

class TestLimitPlan:
    def test_limit(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('limit', '--input', wav, '--output', out,
                          '--threshold', '-1')
        assert result['type'] == 'limit'
        assert result['success'] is True
        assert result['threshold_db'] == -1.0


# =============================================================================
# Plan: filter
# =============================================================================

class TestFilterPlan:
    def test_highpass(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('filter', '--input', wav, '--output', out,
                          '--type', 'highpass', '--cutoff-hz', '200')
        assert result['type'] == 'filter'
        assert result['success'] is True
        assert result['filter_type'] == 'highpass'
        assert result['cutoff_hz'] == 200.0

    def test_lowpass(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('filter', '--input', wav, '--output', out,
                          '--type', 'lowpass', '--cutoff-hz', '5000')
        assert result['success'] is True
        assert result['filter_type'] == 'lowpass'


# =============================================================================
# Plan: pitch-shift
# =============================================================================

class TestPitchShiftPlan:
    def test_pitch_shift_up(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('pitch-shift', '--input', wav, '--output', out,
                          '--semitones', '2')
        assert result['type'] == 'pitch-shift'
        assert result['success'] is True
        assert result['semitones'] == 2.0

    def test_pitch_shift_down(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('pitch-shift', '--input', wav, '--output', out,
                          '--semitones', '-3')
        assert result['success'] is True
        assert result['semitones'] == -3.0


# =============================================================================
# Plan: detect-key
# =============================================================================

class TestDetectKeyPlan:
    def test_detect_key(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0)
        result = _run_cli('detect-key', '--input', wav)
        assert result['type'] == 'detect-key'
        assert result['success'] is True
        assert 'key' in result
        assert 'mode' in result
        assert result['mode'] in ('major', 'minor')
        assert 'confidence' in result


# =============================================================================
# Plan: loudness
# =============================================================================

class TestLoudnessPlan:
    def test_loudness(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'))
        result = _run_cli('loudness', '--input', wav)
        assert result['type'] == 'loudness'
        assert result['success'] is True
        assert 'lufs' in result
        assert 'peak_db' in result
        assert result['lufs'] < 0  # LUFS is always negative for normal audio


# =============================================================================
# Plan: onset-detect
# =============================================================================

class TestOnsetDetectPlan:
    def test_onset_detect(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=4)
        result = _run_cli('onset-detect', '--input', wav)
        assert result['type'] == 'onset-detect'
        assert result['success'] is True
        assert 'onset_count' in result
        assert result['onset_count'] > 0
        assert 'onset_times' in result
        assert len(result['onset_times']) == result['onset_count']


# =============================================================================
# Plan: split-bars
# =============================================================================

class TestSplitBarsPlan:
    def test_split_bars(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=4)
        out_dir = str(tmp_path / 'chunks')
        result = _run_cli('split-bars', '--input', wav, '--output-dir', out_dir,
                          '--bpm', '120', '--bars-per-chunk', '1')
        assert result['type'] == 'split-bars'
        assert result['success'] is True
        assert result['chunks'] == 4
        assert len(result['files']) == 4
        for f in result['files']:
            assert Path(f).exists()

    def test_split_bars_multi(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=8)
        out_dir = str(tmp_path / 'chunks')
        result = _run_cli('split-bars', '--input', wav, '--output-dir', out_dir,
                          '--bpm', '120', '--bars-per-chunk', '2')
        assert result['success'] is True
        assert result['chunks'] == 4
        assert result['bars_per_chunk'] == 2


# =============================================================================
# Plan: melody-to-midi
# =============================================================================

class TestMelodyToMidiPlan:
    @pytest.mark.slow
    def test_melody_to_midi(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0, freq=440.0)
        out = str(tmp_path / 'out.mid')
        result = _run_cli('melody-to-midi', '--input', wav, '--output', out)
        assert result['type'] == 'melody-to-midi'
        assert result['success'] is True
        assert Path(out).exists()
        assert 'note_count' in result


# =============================================================================
# Plan: master-track (composite: normalize -> compress -> limit)
# =============================================================================

class TestMasterTrackPlan:
    def test_master_track_chain(self, tmp_path: Path) -> None:
        """Simulate the master-track composite plan."""
        wav = _make_wav(str(tmp_path / 'in.wav'))
        norm_out = str(tmp_path / 'norm.wav')
        comp_out = str(tmp_path / 'comp.wav')
        final_out = str(tmp_path / 'master.wav')

        # Step 1: Normalize
        r1 = _run_cli('normalize', '--input', wav, '--output', norm_out,
                       '--mode', 'lufs', '--target-lufs', '-14')
        assert r1['success']

        # Step 2: Compress
        r2 = _run_cli('compress', '--input', norm_out, '--output', comp_out,
                       '--threshold', '-20', '--ratio', '4')
        assert r2['success']

        # Step 3: Limit
        r3 = _run_cli('limit', '--input', comp_out, '--output', final_out,
                       '--threshold', '-1')
        assert r3['success']
        assert Path(final_out).exists()


# =============================================================================
# Plan: sample-prep (composite: analyze -> trim -> normalize -> convert)
# =============================================================================

class TestSamplePrepPlan:
    def test_sample_prep_chain(self, tmp_path: Path) -> None:
        """Simulate the sample-prep composite plan."""
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=8)
        trim_out = str(tmp_path / 'trimmed.wav')
        norm_out = str(tmp_path / 'normed.wav')
        final_out = str(tmp_path / 'final.wav')

        # Step 1: Analyze
        r1 = _run_cli('analyze', '--input', wav)
        assert r1['success']
        bpm = str(round(r1['bpm']))

        # Step 2: Trim
        r2 = _run_cli('trim', '--input', wav, '--output', trim_out,
                       '--bpm', bpm, '--bars', '4')
        assert r2['success']

        # Step 3: Normalize
        r3 = _run_cli('normalize', '--input', trim_out, '--output', norm_out,
                       '--mode', 'lufs', '--target-lufs', '-14')
        assert r3['success']

        # Step 4: Convert
        r4 = _run_cli('convert', '--input', norm_out, '--output', final_out,
                       '--sample-rate', '44100', '--bit-depth', '24')
        assert r4['success']
        assert Path(final_out).exists()


# =============================================================================
# Plan: tempo-match (composite: analyze -> time-stretch)
# =============================================================================

class TestTempoMatchPlan:
    def test_tempo_match_chain(self, tmp_path: Path) -> None:
        """Simulate the tempo-match composite plan."""
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=8)
        out = str(tmp_path / 'matched.wav')

        # Step 1: Analyze to get BPM
        r1 = _run_cli('analyze', '--input', wav)
        assert r1['success']
        source_bpm = str(round(r1['bpm']))

        # Step 2: Time-stretch to target
        r2 = _run_cli('time-stretch', '--input', wav, '--output', out,
                       '--source-bpm', source_bpm, '--target-bpm', '130')
        assert r2['success']
        assert Path(out).exists()


# =============================================================================
# Plan: full-analysis (composite: analyze + detect-key + loudness)
# =============================================================================

class TestFullAnalysisPlan:
    def test_full_analysis_chain(self, tmp_path: Path) -> None:
        """Simulate the full-analysis composite plan."""
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0)

        r1 = _run_cli('analyze', '--input', wav)
        assert r1['success']
        assert 'bpm' in r1

        r2 = _run_cli('detect-key', '--input', wav)
        assert r2['success']
        assert 'key' in r2

        r3 = _run_cli('loudness', '--input', wav)
        assert r3['success']
        assert 'lufs' in r3


# =============================================================================
# Plan: melody-extract (composite: normalize -> melody-to-midi)
# =============================================================================

class TestMelodyExtractPlan:
    @pytest.mark.slow
    def test_melody_extract_chain(self, tmp_path: Path) -> None:
        """Simulate the melody-extract composite plan."""
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0, freq=440.0)
        norm_out = str(tmp_path / 'normed.wav')
        midi_out = str(tmp_path / 'melody.mid')

        # Step 1: Normalize
        r1 = _run_cli('normalize', '--input', wav, '--output', norm_out,
                       '--mode', 'lufs', '--target-lufs', '-14')
        assert r1['success']

        # Step 2: Extract melody to MIDI
        r2 = _run_cli('melody-to-midi', '--input', norm_out, '--output', midi_out)
        assert r2['success']
        assert Path(midi_out).exists()


# =============================================================================
# Error handling tests
# =============================================================================

class TestArtifactVerification:
    """Verify output file properties (sample rate, channels, duration)."""

    def test_time_stretch_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=3.0, sr=44100)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('time-stretch', '--input', wav, '--output', out,
                          '--source-bpm', '120', '--target-bpm', '130')
        assert result['success'] is True
        info = sf.info(out)
        assert info.samplerate == 44100
        # Faster tempo => shorter duration
        assert info.duration < 3.0

    def test_normalize_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), sr=44100)
        out = str(tmp_path / 'out.wav')
        _run_cli('normalize', '--input', wav, '--output', out,
                 '--mode', 'lufs', '--target-lufs', '-14')
        info = sf.info(out)
        assert info.samplerate == 44100

    def test_gain_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), sr=44100, stereo=True)
        out = str(tmp_path / 'out.wav')
        _run_cli('gain', '--input', wav, '--output', out, '--db', '3')
        info = sf.info(out)
        assert info.samplerate == 44100
        assert info.channels == 2

    def test_compress_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=2.0)
        out = str(tmp_path / 'out.wav')
        _run_cli('compress', '--input', wav, '--output', out,
                 '--threshold', '-20', '--ratio', '4')
        info = sf.info(out)
        assert info.samplerate == 44100
        assert abs(info.duration - 2.0) < 0.1

    def test_reverb_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=1.0)
        out = str(tmp_path / 'out.wav')
        _run_cli('reverb', '--input', wav, '--output', out,
                 '--room-size', '0.5', '--wet-level', '0.3')
        info = sf.info(out)
        assert info.samplerate == 44100
        # Reverb adds a tail, so output should be >= input duration
        assert info.duration >= 1.0

    def test_pitch_shift_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=2.0)
        out = str(tmp_path / 'out.wav')
        _run_cli('pitch-shift', '--input', wav, '--output', out,
                 '--semitones', '2')
        info = sf.info(out)
        assert info.samplerate == 44100
        assert abs(info.duration - 2.0) < 0.5

    def test_filter_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=2.0)
        out = str(tmp_path / 'out.wav')
        _run_cli('filter', '--input', wav, '--output', out,
                 '--type', 'highpass', '--cutoff-hz', '200')
        info = sf.info(out)
        assert info.samplerate == 44100
        assert abs(info.duration - 2.0) < 0.1

    def test_limit_output_properties(self, tmp_path: Path) -> None:
        wav = _make_wav(str(tmp_path / 'in.wav'), duration=2.0)
        out = str(tmp_path / 'out.wav')
        _run_cli('limit', '--input', wav, '--output', out, '--threshold', '-1')
        info = sf.info(out)
        assert info.samplerate == 44100
        # Verify peak is within threshold
        audio, _ = sf.read(out)
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        assert peak_db <= 0.0  # Should not exceed 0 dBFS


class TestErrorHandling:
    def test_missing_input_file(self, tmp_path: Path) -> None:
        result = _run_cli('normalize', '--input', str(tmp_path / 'nope.wav'),
                          '--output', str(tmp_path / 'out.wav'),
                          expect_success=False)
        assert result.get('code') == 'FILE_NOT_FOUND'

    def test_missing_input_gain(self, tmp_path: Path) -> None:
        result = _run_cli('gain', '--input', str(tmp_path / 'nope.wav'),
                          '--output', str(tmp_path / 'out.wav'),
                          '--db', '3', expect_success=False)
        assert result.get('code') == 'FILE_NOT_FOUND'

    def test_missing_input_analyze(self, tmp_path: Path) -> None:
        result = _run_cli('analyze', '--input', str(tmp_path / 'nope.wav'),
                          expect_success=False)
        assert result.get('code') == 'FILE_NOT_FOUND'

    def test_missing_input_detect_key(self, tmp_path: Path) -> None:
        result = _run_cli('detect-key', '--input', str(tmp_path / 'nope.wav'),
                          expect_success=False)
        assert result.get('code') == 'FILE_NOT_FOUND'

    def test_missing_input_compress(self, tmp_path: Path) -> None:
        result = _run_cli('compress', '--input', str(tmp_path / 'nope.wav'),
                          '--output', str(tmp_path / 'out.wav'),
                          '--threshold', '-20', '--ratio', '4',
                          expect_success=False)
        assert result.get('code') == 'FILE_NOT_FOUND'

    def test_trim_invalid_bars(self, tmp_path: Path) -> None:
        wav = _make_click_track(str(tmp_path / 'in.wav'))
        out = str(tmp_path / 'out.wav')
        result = _run_cli('trim', '--input', wav, '--output', out,
                          '--bpm', '120', '--bars', '0', expect_success=False)
        assert result.get('code') == 'INVALID_BARS'

    def test_ping_no_args(self) -> None:
        """Ping requires no arguments."""
        result = _run_cli('ping')
        assert result['success'] is True

    def test_help_output(self) -> None:
        """Verify --help works for main group."""
        proc = subprocess.run(
            [sys.executable, '-m', 'sas_processor', '--help'],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert 'Audio processing tools' in proc.stdout

    def test_subcommand_help(self) -> None:
        """Verify --help works for individual subcommands."""
        subcommands = ['trim', 'normalize', 'gain', 'compress', 'eq',
                       'reverb', 'limit', 'filter', 'pitch-shift',
                       'time-stretch', 'analyze', 'detect-key', 'loudness',
                       'onset-detect', 'split-bars', 'melody-to-midi',
                       'mono', 'convert', 'silence-remove', 'ping']
        for cmd in subcommands:
            proc = subprocess.run(
                [sys.executable, '-m', 'sas_processor', cmd, '--help'],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"--help failed for {cmd}"
            assert '--input' in proc.stdout or cmd == 'ping', (
                f"--help for {cmd} missing expected content"
            )

    def test_verbose_flag(self, tmp_path: Path) -> None:
        """Verify --verbose flag is accepted on trim."""
        wav = _make_click_track(str(tmp_path / 'in.wav'), bpm=120, bars=4)
        out = str(tmp_path / 'out.wav')
        result = _run_cli('trim', '--input', wav, '--output', out,
                          '--bpm', '120', '--bars', '2', '--verbose')
        assert result['success'] is True

    def test_stderr_json_on_error(self, tmp_path: Path) -> None:
        """Verify stderr contains valid JSON error for multiple subcommands."""
        missing = str(tmp_path / 'nope.wav')
        for cmd_args in [
            ['normalize', '--input', missing, '--output', str(tmp_path / 'o.wav')],
            ['gain', '--input', missing, '--output', str(tmp_path / 'o.wav'), '--db', '3'],
            ['analyze', '--input', missing],
            ['loudness', '--input', missing],
        ]:
            proc = subprocess.run(
                [sys.executable, '-m', 'sas_processor', *cmd_args],
                capture_output=True, text=True,
            )
            assert proc.returncode != 0
            err = json.loads(proc.stderr.strip())
            assert err['type'] == 'error'
            assert 'code' in err
            assert 'message' in err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
