"""Bounded retry policy for `.failed-*` deferred-capture evidence files.

drain_deferred_captures must retry a failed deferred-capture file up to
FAILED_MAX_ATTEMPTS times with exponential backoff, then move it to
.permanent-failed-<ts>.jsonl and emit a `permanent_capture_failure` event at
severity=critical. Permanent-failed files MUST never be reprocessed.

Filename conventions tested:

* Clean file (no .failed- substring) failing for the first time becomes
  ``<basename>.failed-<ts>-attempt-1.jsonl`` (guards the off-by-one fix —
  must NOT become attempt-2 on its first failure).
* Legacy ``.failed-<ts>.jsonl`` shape (pre-existing files without the
  ``-attempt-N`` suffix) is interpreted as attempt-1 and next failure becomes
  attempt-2.
* After 3 failed attempts the file is renamed to
  ``.permanent-failed-<ts>.jsonl`` and a critical event is emitted.
* Permanent-failed files are skipped silently by subsequent drain passes.
* Backoff is per-attempt: 60s, 120s, 240s before the next retry pass touches
  the file. A file whose mtime is younger than its backoff is left untouched.

All tests use a tmp HOME so the production user state is untouched.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """HOME=tmp_path + fail-backend keyring + crypto passphrase + isolated store."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-retry-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _make_clean_jsonl(deferred_dir: Path, session_id: str, ts_suffix: int) -> Path:
    """Write a minimal v1 deferred-capture JSONL file (header + one event)."""
    import json

    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"sess-{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": "event text long enough to pass MIN_CAPTURE length checks",
        "cue": "test cue retry",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    out.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    return out


def _force_insert_failed(monkeypatch) -> None:
    """Make ``capture_turn`` return insert-failed regardless of input."""

    def _stub(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "skipped", "reason": "insert-failed:test"}

    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "capture_turn", _stub)


def _open_isolated_store():
    from iai_mcp.store import MemoryStore

    return MemoryStore()


def test_retry_after_backoff(iai_home, monkeypatch):
    """A ``.failed-<ts>-attempt-1.jsonl`` with mtime > 61s ago is reprocessed."""
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "retry", 1000000000)

    attempt_1 = fpath.with_name("sess-retry-1000000000.failed-1000000000-attempt-1.jsonl")
    fpath.rename(attempt_1)
    aged = time.time() - 70
    os.utime(attempt_1, (aged, aged))

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not attempt_1.exists(), "attempt-1 file must be renamed forward"
    attempt_2 = list(deferred_dir.glob("sess-retry-1000000000.failed-*-attempt-2.jsonl"))
    assert len(attempt_2) == 1, f"expected exactly one attempt-2 file, got {attempt_2}"


def test_permanent_after_3_attempts(iai_home, monkeypatch):
    """A file at attempt-3 that fails its retry pass becomes permanent-failed."""
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "perm", 2000000000)
    attempt_3 = fpath.with_name("sess-perm-2000000000.failed-2000000000-attempt-3.jsonl")
    fpath.rename(attempt_3)
    aged = time.time() - 300
    os.utime(attempt_3, (aged, aged))

    _force_insert_failed(monkeypatch)

    write_event_calls: list[tuple[str, dict, dict]] = []

    def _stub_write_event(_store, kind, data, **kwargs):
        write_event_calls.append((kind, data, kwargs))

    import iai_mcp.events as events_mod

    monkeypatch.setattr(events_mod, "write_event", _stub_write_event)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not attempt_3.exists(), "attempt-3 file must be renamed away"

    perm = list(deferred_dir.glob("sess-perm-2000000000.permanent-failed-*.jsonl"))
    assert len(perm) == 1, f"expected 1 .permanent-failed-* file, got {perm}"

    perm_events = [c for c in write_event_calls if c[0] == "permanent_capture_failure"]
    assert len(perm_events) == 1, (
        f"expected exactly 1 permanent_capture_failure event, got {write_event_calls}"
    )
    kind, data, kwargs = perm_events[0]
    assert kwargs.get("severity") == "critical", kwargs
    assert data.get("attempts") == 3, data
    assert "file" in data and ".permanent-failed-" in data["file"], data


def test_skip_permanent_failed(iai_home, monkeypatch):
    """``.permanent-failed-*.jsonl`` files survive subsequent drain passes."""
    from iai_mcp.capture import drain_deferred_captures
    import json

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    perm = deferred_dir / "sess-doomed-3000000000.permanent-failed-3000000000.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": "doomed",
        "cwd": "/tmp",
    }
    event = {
        "text": "this content stays preserved on disk forever",
        "cue": "perm cue",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    perm.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert perm.exists(), "permanent-failed file must not be unlinked"
    assert perm.name.startswith("sess-doomed-3000000000.permanent-failed-"), perm


def test_clean_file_first_failure_becomes_attempt_1(iai_home, monkeypatch):
    """A file with no ``.failed-`` substring failing for the first time goes
    to ``-attempt-1``, NOT ``-attempt-2``."""
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _make_clean_jsonl(deferred_dir, "first", 4000000000)
    assert ".failed-" not in fpath.name

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not fpath.exists(), "clean file must be renamed away on first failure"

    attempt_1 = list(deferred_dir.glob("sess-first-4000000000.failed-*-attempt-1.jsonl"))
    assert len(attempt_1) == 1, f"expected exactly one attempt-1 file, got {attempt_1}"
    attempt_other = list(deferred_dir.glob("sess-first-4000000000.failed-*-attempt-[2-9].jsonl"))
    assert len(attempt_other) == 0, (
        f"first failure must become attempt-1, not {attempt_other}"
    )


def test_legacy_failed_shape_becomes_attempt_2(iai_home, monkeypatch):
    """A pre-existing legacy-shape file counts as prior_attempt=1 so its next
    failure becomes attempt-2."""
    from iai_mcp.capture import drain_deferred_captures
    import json

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    legacy = deferred_dir / "sess-legacy-5000000000.failed-5000000000.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-13T00:00:00Z",
        "session_id": "legacy",
        "cwd": "/tmp",
    }
    event = {
        "text": "legacy shape from before the retry policy landed",
        "cue": "legacy cue",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T00:00:00Z",
    }
    legacy.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
    aged = time.time() - 70
    os.utime(legacy, (aged, aged))

    _force_insert_failed(monkeypatch)
    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert not legacy.exists(), "legacy file must be renamed forward"

    attempt_2 = list(deferred_dir.glob("sess-legacy-5000000000.failed-*-attempt-2.jsonl"))
    assert len(attempt_2) == 1, f"expected exactly one attempt-2 file, got {attempt_2}"
