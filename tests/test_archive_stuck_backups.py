"""Daemon-startup archiver for ``lifecycle_state.json.HIBERNATION-stuck*.bak``.

The lifecycle recovery path historically left ``.bak`` artifacts next to
``lifecycle_state.json`` in ``~/.iai-mcp/``. ``archive_stuck_backups`` moves
each matching file into ``~/.iai-mcp/archive/`` with an mtime-stamped name,
creating the archive directory at mode ``0o700`` if absent. Idempotent: a
second pass over the same destination skips the move and reports it as a
collision.
"""
from __future__ import annotations

import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path


def test_archive_moves_bak_file(tmp_path):
    from iai_mcp.archive_backups import archive_stuck_backups

    state_dir = tmp_path / ".iai-mcp"
    state_dir.mkdir(parents=True, exist_ok=True)

    bak = state_dir / "lifecycle_state.json.HIBERNATION-stuck.bak"
    payload = b"recovery snapshot bytes"
    bak.write_bytes(payload)

    pinned = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(bak, (pinned, pinned))

    result = archive_stuck_backups(state_dir=state_dir)

    assert result == {"moved": 1, "skipped_existing": 0}, result
    assert not bak.exists(), "source should be removed after move"

    archive_dir = state_dir / "archive"
    assert archive_dir.is_dir()
    archive_mode = stat.S_IMODE(archive_dir.stat().st_mode)
    assert archive_mode == 0o700, oct(archive_mode)

    expected_name = "lifecycle_state.json.HIBERNATION-stuck.bak-20260513T120000Z.bak"
    expected = archive_dir / expected_name
    assert expected.exists(), f"expected archived file at {expected}"
    assert expected.read_bytes() == payload


def test_archive_idempotent(tmp_path):
    from iai_mcp.archive_backups import archive_stuck_backups

    state_dir = tmp_path / ".iai-mcp"
    state_dir.mkdir(parents=True, exist_ok=True)

    bak = state_dir / "lifecycle_state.json.HIBERNATION-stuck.bak"
    bak.write_bytes(b"first")
    pinned = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(bak, (pinned, pinned))

    first = archive_stuck_backups(state_dir=state_dir)
    assert first == {"moved": 1, "skipped_existing": 0}

    bak.write_bytes(b"second")
    os.utime(bak, (pinned, pinned))

    second = archive_stuck_backups(state_dir=state_dir)
    assert second == {"moved": 0, "skipped_existing": 1}, second
    assert bak.exists(), "colliding source should remain on disk"

    bak.unlink()
    third = archive_stuck_backups(state_dir=state_dir)
    assert third == {"moved": 0, "skipped_existing": 0}, third
