"""Concurrent invocation safety tests.

Two `sas-processor` calls writing to the SAME output path (programmer
error) must produce one valid file and no half-written corruption. The
atomic-write helper makes this safe: each process writes to its own
PID/UUID-suffixed temp, then os.replace promotes one of them onto the
final path. The "loser" gets its temp cleaned up on its own exit.

Disjoint-output cases (the normal case — every render targets a unique
path) are also covered as a regression guard for the atomic-write wiring.

We use multiprocessing rather than threads so the writes are genuinely
parallel (separate Python processes, no GIL serialization).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import soundfile as sf

from sas_processor.io_utils import atomic_sf_write


def _writer_worker(
    output_path: str,
    duration_s: float,
    sr: int,
    delay_s: float,
    fill_value: float,
) -> Tuple[bool, str]:
    """Sub-process body: write a WAV with constant amplitude `fill_value`
    after `delay_s` (lets the parent stagger two workers).

    The fill_value lets us tell which writer "won" the race when both
    target the same path.
    """
    try:
        if delay_s > 0:
            time.sleep(delay_s)
        n = int(duration_s * sr)
        audio = np.full(n, fill_value, dtype=np.float32)
        atomic_sf_write(output_path, audio, sr, subtype="PCM_16")
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _read_first_sample(path: Path) -> float:
    """Return the first audio sample of a WAV — used to identify which
    writer's content survived the concurrent race."""
    audio, _ = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return float(audio[0])


class TestConcurrentSamePath:
    def test_two_writers_same_path_produce_one_valid_file(self, tmp_path):
        """Both processes call atomic_sf_write on the SAME output path.
        Result must be exactly one valid WAV at that path, no .tmp leftover,
        and the content must match exactly one of the two writers (last
        os.replace wins — never an interleave)."""
        out = tmp_path / "race.wav"

        # Writer A fills with 0.25; B fills with 0.75. They start "at the
        # same time" (no delay) and race os.replace.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            results = pool.starmap(
                _writer_worker,
                [
                    (str(out), 0.25, 44100, 0.0, 0.25),
                    (str(out), 0.25, 44100, 0.0, 0.75),
                ],
            )

        for ok, msg in results:
            assert ok, f"Worker errored: {msg}"

        # Exactly one file at the final path.
        assert out.exists()
        # No leftover temp files anywhere in the dir.
        leftover_temps = sorted(tmp_path.glob("*.tmp.*"))
        assert leftover_temps == [], (
            f"Expected no leftover .tmp files, got: {[p.name for p in leftover_temps]}"
        )

        # Final file must be readable + match exactly one of the two
        # writers (no interleaving, no corruption).
        first = _read_first_sample(out)
        # Allow PCM_16 quantization slop.
        assert first == pytest.approx(0.25, abs=1e-3) or first == pytest.approx(0.75, abs=1e-3), (
            f"Final content matched neither writer (got first sample={first}); "
            f"this suggests interleaving or corruption."
        )

    def test_writer_b_wins_when_started_after_a(self, tmp_path):
        """Sequential-but-overlapping write: A starts, B starts ~50ms
        later, both target the same path. B must win — its content lands
        last via os.replace."""
        out = tmp_path / "ordered.wav"

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            results = pool.starmap(
                _writer_worker,
                [
                    # A is faster (shorter audio, no delay) but B's longer
                    # write still finishes second because of the head-start
                    # gap; what we really care about is "the file is one
                    # valid file with B's content" once both finish.
                    (str(out), 0.10, 44100, 0.00, 0.25),  # A
                    (str(out), 0.10, 44100, 0.20, 0.75),  # B (200ms later)
                ],
            )
        for ok, msg in results:
            assert ok, msg

        assert out.exists()
        first = _read_first_sample(out)
        # B started 200ms after A, A's file is short — B should land last.
        assert first == pytest.approx(0.75, abs=1e-3), (
            f"Expected B's content (0.75) to win; got first sample={first}"
        )
        assert sorted(tmp_path.glob("*.tmp.*")) == []


class TestConcurrentDisjointPaths:
    def test_concurrent_writes_to_different_paths_succeed(self, tmp_path):
        """Sanity check: the normal (correct) case — many concurrent
        renders to distinct paths — all succeed and produce valid WAVs."""
        ctx = mp.get_context("spawn")
        out_paths = [tmp_path / f"distinct_{i:02d}.wav" for i in range(8)]
        fills = [i / 10.0 for i in range(1, 9)]  # 0.1, 0.2, ..., 0.8

        with ctx.Pool(processes=4) as pool:
            results = pool.starmap(
                _writer_worker,
                [(str(p), 0.1, 44100, 0.0, f) for p, f in zip(out_paths, fills)],
            )
        for ok, msg in results:
            assert ok, msg

        for p, fill in zip(out_paths, fills):
            assert p.exists(), f"Missing output: {p}"
            assert _read_first_sample(p) == pytest.approx(fill, abs=1e-3)

        # Zero leftover temps anywhere in the dir.
        assert sorted(tmp_path.glob("*.tmp.*")) == []
