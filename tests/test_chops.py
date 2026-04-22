"""Tests for beat-aligned chopping (v1 transition generator)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sas_processor.chops import trim_range


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sine_4bar_120bpm(tmp_path: Path) -> Path:
    """A 4-bar 120 BPM sine wave at 44.1 kHz stereo. 8s total."""
    sr = 44100
    bpm = 120.0
    meter = 4
    bars = 4
    duration_s = (60.0 / bpm) * meter * bars        # 8s
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    sine = 0.25 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)
    out = tmp_path / "in.wav"
    sf.write(out, stereo, sr)
    return out


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------

def test_trim_range_produces_chop_of_correct_length(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    out = tmp_path / "chop.wav"
    result = trim_range(str(sine_4bar_120bpm), str(out), bpm=120.0, meter=4, start_beat=0, duration_beats=2)

    # 2 beats at 120 BPM at 44100 Hz = 2 * (60/120) * 44100 = 44100 samples.
    assert result["samples"] == 44100
    assert result["sample_rate"] == 44100
    assert result["channels"] == 2
    assert Path(out).exists()
    # Read the output and check sample count matches.
    data, sr = sf.read(str(out), always_2d=True)
    assert sr == 44100
    assert abs(data.shape[0] - 44100) <= 1           # ±1 sample tolerance for rounding


def test_trim_range_whole_bar_at_120bpm(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    """Whole-bar chop from bar 2 (beats 4..7) — 4 beats = 88200 samples."""
    out = tmp_path / "chop.wav"
    result = trim_range(str(sine_4bar_120bpm), str(out), bpm=120.0, meter=4, start_beat=4, duration_beats=4)

    assert result["samples"] == 88200
    assert result["start_sample"] == 88200           # beat 4 × 22050 samples/beat
    data, _ = sf.read(str(out), always_2d=True)
    assert abs(data.shape[0] - 88200) <= 1


def test_trim_range_quarter_bar(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    """Quarter-bar (1 beat) chop — smallest allowed v1 length."""
    out = tmp_path / "chop.wav"
    result = trim_range(str(sine_4bar_120bpm), str(out), bpm=120.0, meter=4, start_beat=3, duration_beats=1)

    assert result["samples"] == 22050
    assert result["duration_s"] == pytest.approx(0.5, abs=0.001)


def test_trim_range_past_eof_zero_pads(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    """Requesting a chop that extends past EOF returns the expected length
    (zero-padded) rather than silently truncating. Prevents short final chops."""
    out = tmp_path / "chop.wav"
    # Input is 16 beats long; request starts at beat 15 with 4 beats duration → 3 beats past EOF
    result = trim_range(str(sine_4bar_120bpm), str(out), bpm=120.0, meter=4, start_beat=15, duration_beats=4)

    assert result["samples"] == 88200
    data, _ = sf.read(str(out), always_2d=True)
    assert abs(data.shape[0] - 88200) <= 1
    # The tail should be silence (zero-padded) — check the last 0.5s is near-zero.
    tail = data[-22050:]
    assert float(np.max(np.abs(tail))) < 0.01


def test_trim_range_rejects_bad_args(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    out = tmp_path / "chop.wav"
    with pytest.raises(ValueError):
        trim_range(str(sine_4bar_120bpm), str(out), bpm=0, meter=4, start_beat=0, duration_beats=1)
    with pytest.raises(ValueError):
        trim_range(str(sine_4bar_120bpm), str(out), bpm=120, meter=4, start_beat=0, duration_beats=0)
    with pytest.raises(ValueError):
        trim_range(str(sine_4bar_120bpm), str(out), bpm=120, meter=0, start_beat=0, duration_beats=1)
    with pytest.raises(ValueError):
        trim_range(str(sine_4bar_120bpm), str(out), bpm=120, meter=4, start_beat=-1, duration_beats=1)


def test_trim_range_missing_input_raises(tmp_path: Path) -> None:
    out = tmp_path / "chop.wav"
    with pytest.raises(FileNotFoundError):
        trim_range(str(tmp_path / "nope.wav"), str(out), bpm=120, meter=4, start_beat=0, duration_beats=1)


def test_trim_range_creates_parent_dirs(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    """Output path's parent dir is auto-created so callers can nest freely."""
    out = tmp_path / "nested" / "sub" / "chop.wav"
    trim_range(str(sine_4bar_120bpm), str(out), bpm=120, meter=4, start_beat=0, duration_beats=1)
    assert out.exists()


def test_trim_range_linear_fade_in(tmp_path: Path) -> None:
    """Fade-in ramps the first N beats from 0 -> 1.

    Build a constant DC signal so we can check the envelope directly in
    the output. After a 1-beat fade-in over a 4-beat chop at 120 BPM
    (22050 samples / beat), the first sample should be ~0 and samples
    past 22050 should match the DC level exactly.
    """
    sr = 44100
    bpm = 120.0
    frames = int(sr * 4)      # 4 seconds, plenty of headroom
    # Use a non-zero constant (0.5) so fade math is observable.
    samples = np.full(frames, 0.5, dtype=np.float32)

    src = tmp_path / "dc.wav"
    sf.write(src, samples, sr, subtype="FLOAT")
    out = tmp_path / "faded.wav"

    trim_range(str(src), str(out), bpm=bpm, meter=4,
               start_beat=0, duration_beats=4, fade_in_beats=1)

    data, _ = sf.read(str(out), always_2d=False)
    # Sample 0 is first fade sample → ~0 (linspace starts at 0).
    assert abs(data[0]) < 1e-4
    # Mid-fade (sample 11025 = half of 22050) should be ~0.25 (= 0.5 * 0.5).
    assert abs(data[11025] - 0.25) < 0.05
    # Just past fade end — full amplitude (0.5).
    assert abs(data[22100] - 0.5) < 1e-4
    # Well past fade — still full amplitude.
    assert abs(data[60000] - 0.5) < 1e-4


def test_trim_range_linear_fade_out(tmp_path: Path) -> None:
    """Fade-out ramps the last N beats from 1 -> 0."""
    sr = 44100
    bpm = 120.0
    frames = int(sr * 4)
    samples = np.full(frames, 0.5, dtype=np.float32)

    src = tmp_path / "dc.wav"
    sf.write(src, samples, sr, subtype="FLOAT")
    out = tmp_path / "faded.wav"

    trim_range(str(src), str(out), bpm=bpm, meter=4,
               start_beat=0, duration_beats=4, fade_out_beats=1)

    data, _ = sf.read(str(out), always_2d=False)
    total = len(data)
    # Full amplitude in the non-fade region.
    assert abs(data[0] - 0.5) < 1e-4
    assert abs(data[total - 44000] - 0.5) < 1e-4
    # Fade-out region: last sample is ~0 (ramp ends just short of 0).
    assert abs(data[-1]) < 0.01
    # Mid-fade: ~0.25.
    assert abs(data[total - 11025] - 0.25) < 0.05


def test_trim_range_combined_fade_in_and_out(tmp_path: Path) -> None:
    """Both fades applied together — non-fade middle stays at full amplitude."""
    sr = 44100
    bpm = 120.0
    frames = int(sr * 4)
    samples = np.full(frames, 0.5, dtype=np.float32)

    src = tmp_path / "dc.wav"
    sf.write(src, samples, sr, subtype="FLOAT")
    out = tmp_path / "faded.wav"

    trim_range(str(src), str(out), bpm=bpm, meter=4,
               start_beat=0, duration_beats=4,
               fade_in_beats=1, fade_out_beats=1)

    data, _ = sf.read(str(out), always_2d=False)
    total = len(data)
    # Fade-in start ~0.
    assert abs(data[0]) < 1e-4
    # Middle (beat 2, sample 44100) — full amplitude.
    assert abs(data[44100] - 0.5) < 1e-4
    # Fade-out end ~0.
    assert abs(data[-1]) < 0.01


def test_trim_range_fades_sum_validation(tmp_path: Path) -> None:
    """fade_in + fade_out must not exceed duration_beats."""
    sr = 44100
    samples = np.zeros(sr * 4, dtype=np.float32)
    src = tmp_path / "z.wav"
    sf.write(src, samples, sr, subtype="FLOAT")
    out = tmp_path / "out.wav"

    with pytest.raises(ValueError, match="must not exceed duration"):
        trim_range(str(src), str(out), bpm=120.0, meter=4,
                   start_beat=0, duration_beats=2,
                   fade_in_beats=1.5, fade_out_beats=1.5)


def test_cli_trim_range_accepts_fade_flags(tmp_path: Path) -> None:
    """CLI path wires --fade-in-beats / --fade-out-beats through to trim_range."""
    sr = 44100
    samples = np.full(sr * 4, 0.5, dtype=np.float32)
    src = tmp_path / "dc.wav"
    sf.write(src, samples, sr, subtype="FLOAT")
    out = tmp_path / "faded.wav"

    proc = subprocess.run(
        [sys.executable, "-m", "sas_processor.cli", "trim-range",
         "--input", str(src),
         "--output", str(out),
         "--bpm", "120",
         "--meter", "4",
         "--start-beat", "0",
         "--duration-beats", "2",
         "--fade-in-beats", "0.5",
         "--fade-out-beats", "0.5"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    last = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(last)
    assert payload["success"] is True
    assert payload["fade_in_samples"] == int(round(0.5 * (60.0 / 120.0) * sr))
    assert payload["fade_out_samples"] == int(round(0.5 * (60.0 / 120.0) * sr))
    # Spot check the output has non-full amplitude at start and end.
    data, _ = sf.read(str(out), always_2d=False)
    assert abs(data[0]) < 1e-4
    assert abs(data[-1]) < 0.01


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_trim_range_preserves_input_subtype(tmp_path: Path, subtype: str) -> None:
    """Regression: chops MUST inherit the input WAV's subtype (bit depth /
    encoding). Without this, soundfile defaults to PCM_16 on write and
    chops from a 24-bit Tracktion stem come out as 16-bit, causing the
    TS-side wav-concat / wav-mix utility to refuse to concat them with
    the 24-bit silence WAVs the transition renderer generates.
    """
    sr = 44100
    duration_s = 1.0
    n = int(sr * duration_s)
    sine = 0.25 * np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)

    src = tmp_path / f"in-{subtype}.wav"
    sf.write(src, stereo, sr, subtype=subtype)

    out = tmp_path / f"chop-{subtype}.wav"
    trim_range(str(src), str(out), bpm=120.0, meter=4, start_beat=0, duration_beats=1)

    assert sf.info(str(out)).subtype == subtype


# -----------------------------------------------------------------------------
# CLI tests (exercise the subprocess path used by the TS wrapper)
# -----------------------------------------------------------------------------

def test_cli_trim_range_emits_json(sine_4bar_120bpm: Path, tmp_path: Path) -> None:
    out = tmp_path / "chop.wav"
    proc = subprocess.run(
        [sys.executable, "-m", "sas_processor.cli", "trim-range",
         "--input", str(sine_4bar_120bpm),
         "--output", str(out),
         "--bpm", "120",
         "--meter", "4",
         "--start-beat", "2",
         "--duration-beats", "2"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    # Last stdout line is the JSON result.
    last = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(last)
    assert payload["type"] == "trim-range"
    assert payload["success"] is True
    assert payload["samples"] == 44100
    assert payload["sample_rate"] == 44100
    assert Path(payload["output"]).exists()


def test_cli_trim_range_reports_error_on_missing_input(tmp_path: Path) -> None:
    out = tmp_path / "chop.wav"
    proc = subprocess.run(
        [sys.executable, "-m", "sas_processor.cli", "trim-range",
         "--input", str(tmp_path / "nope.wav"),
         "--output", str(out),
         "--bpm", "120",
         "--meter", "4",
         "--start-beat", "0",
         "--duration-beats", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    # The _validate_input_file path emits FILE_NOT_FOUND to stderr.
    assert "FILE_NOT_FOUND" in proc.stderr or "not found" in proc.stderr.lower()
