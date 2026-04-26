"""Tests for the atomic write helpers (`io_utils.py`).

Goal: verify that every audio-producing path is atomic from the caller's
perspective — a mid-write failure leaves the destination either fully
written with the new content or untouched (with the prior content if any).
Half-written files are never visible at the final path.

These tests exercise the helpers directly and through the higher-level
effects / processor / chops APIs to catch wiring regressions.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from sas_processor import io_utils
from sas_processor.io_utils import (
    atomic_midi_write,
    atomic_sf_write,
    atomic_write_path,
    cleanup_inflight_temps,
)


def _make_silence_wav(path: Path, duration_s: float = 0.5, sr: int = 44100) -> None:
    """Write a tiny non-trivial WAV that downstream sf.read can validate."""
    n = int(duration_s * sr)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr, subtype="PCM_16", format="WAV")


# ---------- core helper ---------------------------------------------------


class TestAtomicWritePath:
    def test_clean_exit_replaces_destination(self, tmp_path):
        final = tmp_path / "out.wav"
        with atomic_write_path(final) as tmp:
            # Simulate a writer producing real WAV content at the temp path.
            sf.write(str(tmp), np.zeros(100, dtype=np.float32), 44100,
                     subtype="PCM_16", format="WAV")
        assert final.exists()
        assert final.stat().st_size > 0
        # No leftover .tmp
        leftovers = list(tmp_path.glob("*.tmp.*"))
        assert leftovers == [], f"Unexpected temp files: {leftovers}"

    def test_exception_inside_block_preserves_prior_destination(self, tmp_path):
        final = tmp_path / "out.wav"
        # Pre-populate the destination with known content.
        _make_silence_wav(final, duration_s=0.25, sr=44100)
        prior_size = final.stat().st_size
        prior_bytes = final.read_bytes()

        with pytest.raises(RuntimeError, match="boom"):
            with atomic_write_path(final) as tmp:
                # Write a partial file at temp, then fail before the block exits.
                sf.write(str(tmp), np.zeros(100, dtype=np.float32), 44100,
                         subtype="PCM_16", format="WAV")
                raise RuntimeError("boom — caller crashed mid-write")

        # The destination must be untouched.
        assert final.stat().st_size == prior_size
        assert final.read_bytes() == prior_bytes
        # Temp must be cleaned up.
        leftovers = list(tmp_path.glob("*.tmp.*"))
        assert leftovers == [], f"Unexpected temp files: {leftovers}"

    def test_exception_with_no_prior_destination_leaves_no_file(self, tmp_path):
        final = tmp_path / "fresh.wav"
        assert not final.exists()

        with pytest.raises(ValueError, match="boom"):
            with atomic_write_path(final) as tmp:
                sf.write(str(tmp), np.zeros(100, dtype=np.float32), 44100,
                         subtype="PCM_16", format="WAV")
                raise ValueError("boom")

        assert not final.exists()
        leftovers = list(tmp_path.glob("*.tmp.*"))
        assert leftovers == [], f"Unexpected temp files: {leftovers}"

    def test_temps_namespace_per_pid_and_uuid(self, tmp_path):
        """Two calls into atomic_write_path on the same final must produce
        distinct temp paths so they cannot collide."""
        final = tmp_path / "out.wav"

        # We'll capture the temp paths yielded by two sequential opens of the
        # same final path — they must differ even with the same PID.
        temps: list[Path] = []
        with atomic_write_path(final) as tmp1:
            temps.append(tmp1)
            sf.write(str(tmp1), np.zeros(50, dtype=np.float32), 44100,
                     subtype="PCM_16", format="WAV")
        with atomic_write_path(final) as tmp2:
            temps.append(tmp2)
            sf.write(str(tmp2), np.zeros(50, dtype=np.float32), 44100,
                     subtype="PCM_16", format="WAV")

        assert temps[0] != temps[1]
        # Both should sit beside `final`, not in a subdir.
        for t in temps:
            assert t.parent == final.parent

    def test_inflight_set_cleared_on_success(self, tmp_path):
        # Belt-and-suspenders: after a clean write, the in-flight registry
        # should be empty (so a later signal handler doesn't re-delete a
        # path that is now the LIVE destination).
        cleanup_inflight_temps()  # clear any state from prior tests
        final = tmp_path / "out.wav"
        with atomic_write_path(final) as tmp:
            sf.write(str(tmp), np.zeros(50, dtype=np.float32), 44100,
                     subtype="PCM_16", format="WAV")
        assert len(io_utils._INFLIGHT_TEMPS) == 0

    def test_inflight_set_cleared_on_failure(self, tmp_path):
        cleanup_inflight_temps()
        final = tmp_path / "out.wav"
        with pytest.raises(RuntimeError):
            with atomic_write_path(final) as tmp:
                sf.write(str(tmp), np.zeros(50, dtype=np.float32), 44100,
                         subtype="PCM_16", format="WAV")
                raise RuntimeError("crash")
        assert len(io_utils._INFLIGHT_TEMPS) == 0


# ---------- atomic_sf_write -----------------------------------------------


class TestAtomicSfWrite:
    def test_writes_valid_wav(self, tmp_path):
        out = tmp_path / "out.wav"
        atomic_sf_write(out, np.zeros(1000, dtype=np.float32), 44100, subtype="PCM_16")
        info = sf.info(str(out))
        assert info.samplerate == 44100
        assert info.frames == 1000

    def test_failure_does_not_corrupt_prior_file(self, tmp_path):
        out = tmp_path / "out.wav"
        _make_silence_wav(out, duration_s=0.5, sr=44100)
        prior = out.read_bytes()

        # Force sf.write to fail to simulate a bad subtype, OOM, etc.
        with patch("soundfile.write", side_effect=RuntimeError("simulated write failure")):
            with pytest.raises(RuntimeError, match="simulated"):
                atomic_sf_write(out, np.zeros(100, dtype=np.float32), 44100, subtype="PCM_16")

        # Destination preserved.
        assert out.read_bytes() == prior
        # No temp leftover.
        assert list(tmp_path.glob("*.tmp.*")) == []


# ---------- atomic_midi_write ---------------------------------------------


class _FakeMidi:
    """Stand-in for pretty_midi.PrettyMIDI — basic_pitch's predict() returns
    one of these. We only need .write(path)."""

    def __init__(self, payload: bytes = b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x60") -> None:
        self.payload = payload
        self.fail_with: BaseException | None = None

    def write(self, path: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        with open(path, "wb") as f:
            f.write(self.payload)


class TestAtomicMidiWrite:
    def test_writes_midi_atomically(self, tmp_path):
        out = tmp_path / "melody.mid"
        atomic_midi_write(out, _FakeMidi())
        assert out.exists()
        assert out.read_bytes().startswith(b"MThd")

    def test_failure_preserves_prior_midi(self, tmp_path):
        out = tmp_path / "melody.mid"
        out.write_bytes(b"PRIOR MIDI CONTENT")
        prior = out.read_bytes()

        bad = _FakeMidi()
        bad.fail_with = RuntimeError("midi crash")
        with pytest.raises(RuntimeError, match="midi crash"):
            atomic_midi_write(out, bad)

        assert out.read_bytes() == prior
        assert list(tmp_path.glob("*.tmp.*")) == []


# ---------- end-to-end through chops + analysis ---------------------------


class TestAtomicityThroughChops:
    """Belt-and-suspenders: confirm chops.trim_range itself fails atomically."""

    def test_trim_range_failure_does_not_create_partial_output(self, tmp_path):
        """If atomic_sf_write raises, the destination must not be created."""
        from sas_processor.chops import trim_range
        from tests.fixtures import create_test_wav_from_sample

        wav_path, _ = create_test_wav_from_sample("brahms", tmp_path)
        out_path = tmp_path / "chop.wav"

        with patch("sas_processor.chops.atomic_sf_write",
                   side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                trim_range(
                    wav_path, str(out_path), bpm=120.0, meter=4,
                    start_beat=0.0, duration_beats=4.0,
                )

        assert not out_path.exists()


class TestAtomicityThroughSplitBars:
    """split_audio_bars must roll back partial bar files and (if it created
    the directory) the directory itself, on mid-loop failure."""

    def test_split_bars_rolls_back_on_partial_failure(self, tmp_path):
        from sas_processor.analysis import split_audio_bars
        from tests.fixtures import create_test_wav_from_sample

        wav_path, _ = create_test_wav_from_sample("brahms", tmp_path)
        out_dir = tmp_path / "bars-fresh"
        assert not out_dir.exists()

        # Patch atomic_sf_write to succeed for the first 2 bars then raise.
        original = io_utils.atomic_sf_write
        call_count = {"n": 0}

        def fake_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 2:
                raise RuntimeError("simulated mid-loop failure")
            return original(*args, **kwargs)

        with patch("sas_processor.analysis.atomic_sf_write", side_effect=fake_write):
            with pytest.raises(RuntimeError, match="simulated mid-loop"):
                split_audio_bars(wav_path, str(out_dir), bpm=120.0,
                                 bars_per_chunk=1, meter=4)

        # The dir we created should be gone (along with all 2 successful bars).
        # If for some reason rmdir failed because rollback missed a file, we
        # still want the test to fail loudly.
        assert not out_dir.exists(), (
            f"Expected freshly-created dir to be rolled back, but it exists "
            f"with: {sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []}"
        )

    def test_split_bars_preserves_caller_provided_dir_on_failure(self, tmp_path):
        """If the caller already had the output dir, we don't delete it
        (only the bar files we wrote)."""
        from sas_processor.analysis import split_audio_bars
        from tests.fixtures import create_test_wav_from_sample

        wav_path, _ = create_test_wav_from_sample("brahms", tmp_path)
        out_dir = tmp_path / "bars-pre-existing"
        out_dir.mkdir()
        # Drop a file that pre-dates our call — must survive the failure.
        sentinel = out_dir / "sentinel.txt"
        sentinel.write_text("do not delete me")

        original = io_utils.atomic_sf_write
        call_count = {"n": 0}

        def fake_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 2:
                raise RuntimeError("simulated mid-loop failure")
            return original(*args, **kwargs)

        with patch("sas_processor.analysis.atomic_sf_write", side_effect=fake_write):
            with pytest.raises(RuntimeError):
                split_audio_bars(wav_path, str(out_dir), bpm=120.0,
                                 bars_per_chunk=1, meter=4)

        assert out_dir.exists(), "Caller-provided dir must NOT be deleted"
        assert sentinel.exists() and sentinel.read_text() == "do not delete me"
        # No partial bar files should remain.
        bar_files = sorted(out_dir.glob("bar_*.wav"))
        assert bar_files == [], f"Unexpected leftover bar files: {bar_files}"


# ---------- in-flight registry sweep --------------------------------------


class TestCleanupInflightTemps:
    def test_cleanup_unlinks_registered_temps(self, tmp_path):
        cleanup_inflight_temps()  # clear state

        # Manually fabricate two "in-flight" temps (mimicking what
        # atomic_write_path would have done before being interrupted).
        t1 = tmp_path / "leftover1.wav.tmp.999.aaa"
        t2 = tmp_path / "leftover2.wav.tmp.999.bbb"
        t1.write_bytes(b"partial1")
        t2.write_bytes(b"partial2")
        io_utils._register_temp(t1)
        io_utils._register_temp(t2)

        removed = cleanup_inflight_temps()
        assert removed == 2
        assert not t1.exists()
        assert not t2.exists()

    def test_cleanup_idempotent(self, tmp_path):
        cleanup_inflight_temps()
        # Running with no in-flight temps is a no-op.
        assert cleanup_inflight_temps() == 0
        assert cleanup_inflight_temps() == 0

    def test_cleanup_swallows_already_gone(self, tmp_path):
        cleanup_inflight_temps()
        t = tmp_path / "vanished.wav.tmp.999.xyz"
        # Register a path that doesn't actually exist on disk.
        io_utils._register_temp(t)
        # Must not raise, must report 0 removed (the unlink raised ENOENT).
        removed = cleanup_inflight_temps()
        assert removed == 0


# ---------- stderr-JSON logging on recovery paths -------------------------

import json as _json


class TestRecoveryPathLogging:
    """The recovery paths emit stderr-JSON lines so the Electron caller's
    spawn-stderr capture surfaces them in `output.log`. Without these,
    rollbacks were silent — debugging a failed render meant guessing what
    the helper actually did. These tests prove the lines fire."""

    def _parse_log_events(self, capsys, event_name: str) -> list[dict]:
        """Read captured stderr, parse each JSON line, return events whose
        `event` field matches `event_name`. Lines that aren't JSON (other
        stderr output) are ignored."""
        captured = capsys.readouterr().err
        events: list[dict] = []
        for line in captured.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("event") == event_name:
                events.append(obj)
        return events

    def test_atomic_write_logs_rollback_on_exception(self, tmp_path, capsys):
        out = tmp_path / "out.wav"
        with pytest.raises(RuntimeError, match="boom"):
            with atomic_write_path(out) as tmp:
                sf.write(str(tmp), np.zeros(100, dtype=np.float32), 44100,
                         subtype="PCM_16", format="WAV")
                raise RuntimeError("boom")

        events = self._parse_log_events(capsys, "atomic_write_rolled_back")
        assert len(events) == 1, f"Expected one rollback log, got: {events}"
        ev = events[0]
        assert ev["final_path"] == str(out)
        assert ev["temp_unlinked"] is True
        assert ev["error_type"] == "RuntimeError"

    def test_atomic_write_does_not_log_on_success(self, tmp_path, capsys):
        out = tmp_path / "out.wav"
        atomic_sf_write(out, np.zeros(100, dtype=np.float32), 44100, subtype="PCM_16")
        events = self._parse_log_events(capsys, "atomic_write_rolled_back")
        assert events == [], "No rollback log expected on the happy path"

    def test_cleanup_inflight_logs_when_it_reclaims(self, tmp_path, capsys):
        cleanup_inflight_temps()  # clear pre-existing state
        capsys.readouterr()  # drain prior stderr

        # Pre-register two real temps to be reclaimed.
        t1 = tmp_path / "leftover1.tmp"
        t2 = tmp_path / "leftover2.tmp"
        t1.write_bytes(b"x")
        t2.write_bytes(b"y")
        io_utils._register_temp(t1)
        io_utils._register_temp(t2)

        cleanup_inflight_temps()

        events = self._parse_log_events(capsys, "inflight_temps_cleaned")
        assert len(events) == 1
        ev = events[0]
        assert ev["registered"] == 2
        assert ev["removed"] == 2
        assert ev["missing"] == 0

    def test_cleanup_inflight_silent_when_nothing_to_do(self, tmp_path, capsys):
        cleanup_inflight_temps()  # ensure registry empty
        capsys.readouterr()       # drain prior stderr

        cleanup_inflight_temps()  # genuine no-op

        events = self._parse_log_events(capsys, "inflight_temps_cleaned")
        assert events == [], "No log expected when registry is empty"

    def test_split_audio_bars_logs_rollback_on_failure(self, tmp_path, capsys):
        from sas_processor.analysis import split_audio_bars
        from tests.fixtures import create_test_wav_from_sample

        wav_path, _ = create_test_wav_from_sample("brahms", tmp_path)
        out_dir = tmp_path / "bars-rollback-log"

        original = io_utils.atomic_sf_write
        call_count = {"n": 0}

        def fake_write(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 2:
                raise RuntimeError("simulated mid-loop failure")
            return original(*args, **kwargs)

        with patch("sas_processor.analysis.atomic_sf_write", side_effect=fake_write):
            with pytest.raises(RuntimeError):
                split_audio_bars(wav_path, str(out_dir), bpm=120.0,
                                 bars_per_chunk=1, meter=4)

        events = self._parse_log_events(capsys, "split_audio_bars_rolled_back")
        assert len(events) == 1, f"Expected rollback log, got: {events}"
        ev = events[0]
        assert ev["output_dir"] == str(out_dir)
        assert ev["bars_written"] == 2
        assert ev["bars_unlinked"] == 2
        assert ev["dir_was_created_by_us"] is True
        assert ev["dir_removed"] is True

    def test_trim_range_logs_parent_dir_rollback(self, tmp_path, capsys):
        from sas_processor.chops import trim_range
        from tests.fixtures import create_test_wav_from_sample

        wav_path, _ = create_test_wav_from_sample("brahms", tmp_path)
        # Use a fresh parent dir so trim_range creates it itself.
        out_path = tmp_path / "fresh-parent" / "chop.wav"

        with patch("sas_processor.chops.atomic_sf_write",
                   side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError):
                trim_range(
                    wav_path, str(out_path), bpm=120.0, meter=4,
                    start_beat=0.0, duration_beats=4.0,
                )

        events = self._parse_log_events(capsys, "trim_range_parent_dir_rollback")
        assert len(events) == 1
        ev = events[0]
        assert ev["parent_dir"] == str(out_path.parent)
        assert ev["rmdir_ok"] is True
