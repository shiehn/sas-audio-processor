"""Tests for the SIGTERM/SIGINT cleanup path in `cli.py`.

The CLI registers SIGTERM/SIGINT/SIGHUP handlers at module init that sweep
`io_utils._INFLIGHT_TEMPS` before exiting. These tests verify that the
sweep actually runs and removes the in-flight temp files on signal receipt.

The signal handler can't be exercised purely in-process (signal.signal
needs to fire from kernel-space, not just be invoked) so we spawn a
subprocess that:
  1. Manually registers a fake "in-flight" temp via io_utils._register_temp
  2. Sends itself SIGTERM (or waits for an external SIGTERM)
  3. Exits via the handler's sys.exit path

We then assert from the parent that the temp was deleted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def _python_bin() -> str:
    """Use the same interpreter that runs pytest (which has sas_processor on path)."""
    return sys.executable


def _project_root() -> Path:
    """sas-audio-processor/ — needed so the spawned interpreter can import sas_processor."""
    return Path(__file__).resolve().parent.parent


class TestSigtermCleanup:
    def test_sigterm_during_inflight_temp_cleans_up(self, tmp_path):
        """Spawn a child that registers a temp then loops; SIGTERM it; assert
        the temp gets unlinked by the signal handler before the child exits."""
        temp_file = tmp_path / "inflight.wav.tmp.test"
        temp_file.write_bytes(b"partial wav")
        marker_file = tmp_path / "child-ready.txt"

        script = textwrap.dedent(
            f"""
            import sys, time, os
            sys.path.insert(0, {str(_project_root() / 'src')!r})

            # Importing cli installs the SIGTERM/SIGINT handlers + atexit.
            from sas_processor import cli  # noqa: F401
            from sas_processor import io_utils

            inflight = {str(temp_file)!r}
            marker = {str(marker_file)!r}

            io_utils._register_temp(__import__('pathlib').Path(inflight))
            with open(marker, 'w') as f:
                f.write('ready')

            # Block until killed.
            while True:
                time.sleep(0.05)
            """
        )

        proc = subprocess.Popen([_python_bin(), "-c", script])
        try:
            # Wait for the child to register its temp and signal readiness.
            for _ in range(100):
                if marker_file.exists():
                    break
                time.sleep(0.05)
            else:
                proc.kill()
                pytest.fail("Child never wrote ready marker")

            assert temp_file.exists(), "Temp should still exist while child is alive"

            # Send SIGTERM and wait for the child to exit.
            proc.send_signal(signal.SIGTERM)
            exit_code = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        # Conventional 128 + signum (SIGTERM = 15 → 143) per cli.py handler.
        assert exit_code == 128 + signal.SIGTERM, (
            f"Expected exit code {128 + signal.SIGTERM} from SIGTERM handler, "
            f"got {exit_code}"
        )
        # The handler's job: temp must be gone.
        assert not temp_file.exists(), (
            "SIGTERM handler should have unlinked the in-flight temp before exit"
        )

    def test_sigint_also_cleans_up(self, tmp_path):
        """Same as above but for SIGINT (Ctrl-C / dev path)."""
        temp_file = tmp_path / "inflight-sigint.wav.tmp.test"
        temp_file.write_bytes(b"partial")
        marker_file = tmp_path / "child-ready.txt"

        script = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(_project_root() / 'src')!r})
            from sas_processor import cli  # noqa: F401
            from sas_processor import io_utils

            io_utils._register_temp(__import__('pathlib').Path({str(temp_file)!r}))
            with open({str(marker_file)!r}, 'w') as f:
                f.write('ready')
            while True:
                time.sleep(0.05)
            """
        )

        proc = subprocess.Popen([_python_bin(), "-c", script])
        try:
            for _ in range(100):
                if marker_file.exists():
                    break
                time.sleep(0.05)
            else:
                proc.kill()
                pytest.fail("Child never wrote ready marker")

            proc.send_signal(signal.SIGINT)
            exit_code = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        assert exit_code == 128 + signal.SIGINT
        assert not temp_file.exists()

    def test_atexit_cleans_up_on_natural_exit(self, tmp_path):
        """Even without a signal, the atexit hook should sweep in-flight
        temps when the process exits cleanly. This guards against a code
        path that registers a temp but exits without calling the helper."""
        temp_file = tmp_path / "inflight-atexit.wav.tmp.test"
        temp_file.write_bytes(b"partial")

        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(_project_root() / 'src')!r})
            from sas_processor import cli  # noqa: F401
            from sas_processor import io_utils

            io_utils._register_temp(__import__('pathlib').Path({str(temp_file)!r}))
            # Exit naturally — atexit should fire.
            sys.exit(0)
            """
        )

        proc = subprocess.run([_python_bin(), "-c", script], timeout=10)
        assert proc.returncode == 0
        assert not temp_file.exists(), (
            "atexit handler should have swept the registered temp"
        )
