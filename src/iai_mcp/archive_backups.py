"""Daemon-startup archiver for stuck lifecycle-state backup files.

A historical recovery path left ``lifecycle_state.json.HIBERNATION-stuck.bak``
artifacts in ``~/.iai-mcp/``. No current code generates them, but operator
hygiene calls for moving any leftover ``.bak`` files into an ``archive/``
subdirectory so the state directory stays clean.

``archive_stuck_backups`` runs once per daemon boot. It is idempotent
(collisions are skipped, never overwritten) and fail-safe (per-file
exceptions are swallowed after a warning log line).
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_GLOB_PATTERN: str = "lifecycle_state.json.HIBERNATION-stuck*.bak"
_ARCHIVE_DIR_NAME: str = "archive"


def archive_stuck_backups(
    state_dir: Path | None = None,
    archive_dir: Path | None = None,
) -> dict[str, int]:
    """Move HIBERNATION-stuck backup files into a dated archive directory.

    Each matching file's mtime is rendered as ``YYYYMMDDTHHMMSSZ`` and
    appended to the original basename:
    ``<name>-<mtime_iso_date>.bak``. The archive directory is created at
    mode ``0o700`` when absent. Files whose archive destination already
    exists are left in place and counted under ``skipped_existing``.

    Returns ``{"moved": <int>, "skipped_existing": <int>}``.
    """
    if state_dir is None:
        state_dir = Path.home() / ".iai-mcp"
    if archive_dir is None:
        archive_dir = state_dir / _ARCHIVE_DIR_NAME

    counts = {"moved": 0, "skipped_existing": 0}

    try:
        if not state_dir.is_dir():
            return counts
        matches = sorted(state_dir.glob(_GLOB_PATTERN))
        if not matches:
            return counts
        archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            archive_dir.chmod(0o700)
        except OSError:
            log.warning("archive_stuck_backups: chmod failed for %s", archive_dir)
    except Exception:  # noqa: BLE001 -- top-level fail-safe
        log.warning("archive_stuck_backups: setup failed", exc_info=True)
        return counts

    for src in matches:
        try:
            mtime = src.stat().st_mtime
            stamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            dst_name = f"{src.name}-{stamp}.bak"
            dst = archive_dir / dst_name
            if dst.exists():
                counts["skipped_existing"] += 1
                continue
            shutil.move(str(src), str(dst))
            counts["moved"] += 1
        except Exception:  # noqa: BLE001 -- per-file fail-safe
            log.warning(
                "archive_stuck_backups: failed to archive %s", src, exc_info=True
            )

    return counts
