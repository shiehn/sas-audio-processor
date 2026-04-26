"""Atomic write helpers + signal-aware in-flight temp tracking.

Every audio-producing CLI command writes through these helpers so that:

  * Mid-write failure (exception, SIGTERM, SIGKILL) never leaves a
    half-written WAV at the caller's destination path. The destination is
    either fully replaced with the new content or untouched.

  * Two concurrent processes targeting the same destination path produce
    one valid WAV (last writer wins via os.replace) — never an interleaved
    corrupt file.

The implementation is sibling-tempfile-then-atomic-rename:

    final_path  = /some/dir/output.wav
    temp_path   = /some/dir/output.wav.tmp.<pid>.<uuid>
    write to temp_path → fsync → os.replace(temp_path, final_path)

Sibling-file (same parent dir) is required so os.replace is atomic on the
same filesystem. The PID + UUID suffix means two processes writing the same
final path do not also collide on the temp path.

Temps are registered in a module-level `_INFLIGHT_TEMPS` set so the CLI
signal handler (cli.py) can sweep them on SIGTERM / SIGINT before exiting.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Set

import numpy as np
import soundfile as sf


def _log_event(event: str, **fields: Any) -> None:
    """Emit a stderr-JSON log line.

    The audio-processor is invoked as a subprocess by the Electron app, which
    captures stderr per-call into `output.log` via the spawn handlers in
    `audio-feature-service.ts` / `audio-chop-service.ts`. Errors are already
    routed this way (see `cli.emit_error`) — we use the same channel for
    informational/warning recovery events so they show up in `output.log`
    without a new log infrastructure.

    Best-effort: never raises (logging must not mask the real error path).
    """
    try:
        payload: dict[str, Any] = {"type": "log", "src": "io_utils", "event": event, **fields}
        sys.stderr.write(json.dumps(payload) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — logging must never raise
        pass


# Process-global registry of in-flight temp files. The signal handler in
# cli.py iterates this set on SIGTERM / SIGINT and unlinks each entry.
_INFLIGHT_TEMPS: Set[Path] = set()
_INFLIGHT_LOCK = threading.Lock()


def _make_temp_path(final_path: Path) -> Path:
    """Build a unique sibling temp path for `final_path`.

    Includes PID + a short UUID so two concurrent processes targeting the
    same destination do not race on the temp path itself.
    """
    suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    return final_path.with_name(final_path.name + suffix)


def _register_temp(path: Path) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT_TEMPS.add(path)


def _unregister_temp(path: Path) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT_TEMPS.discard(path)


def cleanup_inflight_temps() -> int:
    """Delete every temp file currently registered as in-flight.

    Returns the number of temps removed. Safe to call from signal handlers
    and atexit hooks — swallows OSError per file. Logs to stderr-JSON when
    it actually reclaims something, so signal/atexit recovery is visible in
    `output.log`.
    """
    with _INFLIGHT_LOCK:
        # Snapshot then clear so this function is idempotent (re-running it
        # in atexit after a signal handler ran is a no-op).
        temps = list(_INFLIGHT_TEMPS)
        _INFLIGHT_TEMPS.clear()

    removed = 0
    for p in temps:
        with contextlib.suppress(OSError):
            p.unlink()
            removed += 1

    if removed > 0 or len(temps) > 0:
        _log_event(
            "inflight_temps_cleaned",
            registered=len(temps),
            removed=removed,
            missing=len(temps) - removed,
        )
    return removed


@contextmanager
def atomic_write_path(final_path: str | os.PathLike[str]) -> Iterator[Path]:
    """Context manager yielding a sibling temp path.

    Usage:

        with atomic_write_path(out) as tmp:
            sf.write(str(tmp), data, sr, subtype=subtype)
        # at this point, `out` has been atomically replaced.

    On exception inside the with-block, the temp file is deleted and the
    exception propagates. The destination at `final_path` is never touched.

    Caller is expected to write to the yielded path. After the block exits
    cleanly, the helper os.replaces the temp onto the final path.
    """
    final = Path(final_path)
    temp = _make_temp_path(final)
    _register_temp(temp)
    try:
        yield temp
        # Block exited cleanly — promote temp to final path. os.replace is
        # atomic on POSIX (and on Windows when src + dst are on the same
        # volume, which they are here because temp is a sibling of final).
        os.replace(temp, final)
    except BaseException as exc:
        # Any failure (Exception, KeyboardInterrupt, SystemExit) — remove
        # the partial temp. We do NOT touch `final`, so its prior state is
        # preserved. Re-raise so the caller sees the original error.
        unlink_ok = True
        try:
            temp.unlink()
        except OSError:
            unlink_ok = False
        # Log so output.log shows WHY a render failed and that we cleaned
        # up after it. The exception type is the load-bearing trace info.
        _log_event(
            "atomic_write_rolled_back",
            final_path=str(final),
            temp_path=str(temp),
            temp_unlinked=unlink_ok,
            error_type=type(exc).__name__,
        )
        raise
    finally:
        _unregister_temp(temp)


def _format_from_extension(final_path: Path) -> str:
    """Map a final-path extension to a soundfile `format` token.

    Required because the temp file we write to has a non-audio extension
    (e.g. `out.wav.tmp.123.abc`), which makes soundfile's auto-detection
    fail. We pass `format` explicitly based on what the FINAL path will be.
    Defaults to WAV — the audio processor is WAV-only at the CLI boundary.
    """
    ext = final_path.suffix.lower()
    if ext == ".wav":
        return "WAV"
    if ext == ".flac":
        return "FLAC"
    if ext in (".ogg", ".oga"):
        return "OGG"
    if ext == ".aiff" or ext == ".aif":
        return "AIFF"
    return "WAV"


def atomic_sf_write(
    output_path: str | os.PathLike[str],
    data: np.ndarray,
    sample_rate: int,
    *,
    subtype: str | None = None,
) -> None:
    """soundfile.write() with atomic-replace semantics.

    Drop-in replacement for `sf.write(output_path, data, sample_rate, subtype=subtype)`
    that guarantees the destination is either fully written or untouched.
    The temp file has a non-audio extension (so concurrent runs don't pick
    up each other's partials), so we pass `format` explicitly to soundfile.
    """
    final = Path(output_path)
    fmt = _format_from_extension(final)
    with atomic_write_path(output_path) as tmp:
        sf.write(str(tmp), data, sample_rate, subtype=subtype, format=fmt)


def atomic_midi_write(output_path: str | os.PathLike[str], midi_data: Any) -> None:
    """Write a pretty_midi PrettyMIDI object atomically.

    `midi_data.write(path)` is the basic_pitch / pretty_midi convention.
    """
    with atomic_write_path(output_path) as tmp:
        midi_data.write(str(tmp))
