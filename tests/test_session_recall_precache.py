"""Contract — compose/emit split + precache + hook fallback.

(a) Daemon helper `_write_session_start_cache` writes a non-empty markdown
    file at the configured cache path; the content equals
    `format_payload_as_markdown(_compose_session_start_payload(...))` capped
    to 10000 chars; and the helper call inserts ZERO new `session_started`
    events into the events table (proves the compose helper, not the
    emit-bearing public wrapper, is what the writer calls).
(b) When the cache file exists, is non-empty, and mtime is fresh, the hook
    prints the cache content, logs `cache-hit fresh`, and exits 0 WITHOUT
    invoking the CLI.
(c) When the cache file is absent, the hook falls through to the live CLI
    path. The CLI's stdout is forwarded AND the hook's log file for today
    contains the substring `cache-miss absent` (this phrase is the
    discriminator that keeps the test RED on current `main`).
(d) When the cache file exists and mtime is 25h old, the hook still
    prints the cache content, logs `cache-hit age=`, and exits 0 WITHOUT
    invoking the CLI (no freshness expiry).
"""
from __future__ import annotations

import datetime as _dt
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


HOOK_PATH = Path(__file__).resolve().parent.parent / "deploy" / "hooks" / "iai-mcp-session-recall.sh"
CACHE_REL = ".iai-mcp/.session-start-payload.cached.md"
SENTINEL = "SENTINEL_LIVE_PATH_OK"


def _today_log_path(home: Path) -> Path:
    """Return the hook's log file path for the current UTC date."""
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    return home / ".iai-mcp" / "logs" / f"recall-{today}.log"


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _make_stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-mcp"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_hook(home: Path, *, extra_env: dict[str, str] | None = None,
              stdin_payload: str = '{"session_id":"x","source":"startup","cwd":"/tmp","transcript_path":""}',
              timeout: float = 10.0):
    env = os.environ.copy()
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=stdin_payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _count_session_started_events(store) -> int:
    """Count rows in the events table where kind='session_started'.

    Uses `query_events` (the public read path) to stay symmetric with how
    M6 trajectory measures the same metric. Returns 0 on absence.
    """
    from iai_mcp.events import query_events
    rows = query_events(store, kind="session_started", limit=10000)
    return len(list(rows))


# ---------------------------------------------------------------------- (a)

def test_daemon_writes_cache_on_rem_completion(tmp_path, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp import retrieve
    from iai_mcp.session import (
        _compose_session_start_payload,
        format_payload_as_markdown,
    )

    store = _fresh_store(tmp_path, monkeypatch)

    # Seed the store using the canonical pattern from
    # tests/test_cmd_session_start_formats_4_segments.py. Use BOTH:
    # _seed_l0_identity for L0 and an inline _seed_pinned_l1(store, 3) for L1
    # so the rendered payload has real content under wake_depth=standard.
    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    from datetime import datetime, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    _now = datetime.now(timezone.utc)
    for _i in range(3):
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {_i}: high-detail context.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=_now,
            updated_at=_now,
            tags=[],
            language="en",
        ))

    # Baseline event count before the writer runs.
    events_before = _count_session_started_events(store)

    cache_path = tmp_path / "session-start-payload.cached.md"
    daemon_mod._write_session_start_cache(store, cache_path=cache_path)

    # File contract.
    assert cache_path.exists(), "cache file was not created"
    content = cache_path.read_text(encoding="utf-8")
    assert content, "cache file is empty — wake_depth=standard should produce content"
    assert len(content) <= 10000, "cache content exceeds 10000-char cap"

    # Parity with what the hook will later read — built via the EMIT-FREE
    # compose helper, not assemble_session_start.
    _g, asgn, rc = retrieve.build_runtime_graph(store)
    payload = _compose_session_start_payload(
        store, asgn, rc,
        session_id="precache",
        profile_state={"wake_depth": "standard"},
    )
    expected = format_payload_as_markdown(payload)[:10000]
    assert content == expected, "cache content does not match _compose_session_start_payload output"

    # CRITICAL: the writer must NOT have emitted any session_started events.
    events_after = _count_session_started_events(store)
    assert events_after == events_before, (
        f"precache writer emitted {events_after - events_before} session_started "
        f"event(s); should be 0 (compose-vs-emit split is broken)"
    )


# ---------------------------------------------------------------------- (b)

def test_hook_reads_cache_when_fresh(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    # Match real payload shape: format_payload_as_markdown joins blocks with
    # "\n\n" and emits no trailing newline, so the daemon-written cache file
    # has no trailing newline either. Bash command substitution
    # (`out=$(head -c ... "$file")`) strips trailing newlines, so a cache
    # that omitted the trailing newline round-trips byte-for-byte.
    cache_content = "# L0 identity\nfresh-cache-content-marker"
    (home / CACHE_REL).write_text(cache_content)
    # mtime defaults to now — already fresh.

    # Boom-on-exec stub: if the hook ever calls the CLI, the test fails.
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"hook did not return cache verbatim. stdout={proc.stdout!r}"
    )
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout, (
        "hook called the CLI instead of reading the cache"
    )

    # Log-line discriminator: keeps the test honest if the hook accidentally
    # routes via the CLI path with an empty CLI stdout.
    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-hit age=" in log_text, (
        f"expected 'cache-hit age=' marker in log; got:\n{log_text}"
    )


# ---------------------------------------------------------------------- (c)

def test_hook_falls_back_when_cache_absent(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    # No cache file written.

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\nprintf '%s' '{SENTINEL}'\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert SENTINEL in proc.stdout, (
        f"fallback CLI sentinel not in stdout. stdout={proc.stdout!r}"
    )

    # HARD DISCRIMINATOR: this phrase is only ever written by the cache-first
    # branch's absent path, which does not exist on current `main`. Without
    # this assertion the test passes on `main` (the sentinel-printing stub
    # produces SENTINEL on either the cache-less or cache-present codepath),
    # which would mean the test is not actually RED. Asserting on the log
    # marker pins the test to the new behaviour.
    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-miss absent" in log_text, (
        f"expected 'cache-miss absent' marker in log; got:\n{log_text}"
    )


# ---------------------------------------------------------------------- (d)

def test_hook_serves_stale_cache(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    stale_cache = home / CACHE_REL
    cache_content = "# stale\nold content that must be ignored"
    stale_cache.write_text(cache_content)
    # Backdate mtime to 25 hours ago.
    twenty_five_h_ago = time.time() - (25 * 3600)
    os.utime(stale_cache, (twenty_five_h_ago, twenty_five_h_ago))

    # BOOM-on-exec stub: if the hook ever calls the CLI, the test fails.
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho CLI_SHOULD_NOT_BE_CALLED\nexit 0\n")
    (home / ".iai-mcp" / ".cli-path").write_text(str(stub_dir / "iai-mcp"))

    proc = _run_hook(home)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == cache_content, (
        f"hook did not return cache verbatim. stdout={proc.stdout!r}"
    )
    assert "CLI_SHOULD_NOT_BE_CALLED" not in proc.stdout, (
        "hook called the CLI instead of reading the stale cache"
    )

    log_path = _today_log_path(home)
    assert log_path.exists(), f"hook log missing: {log_path}"
    log_text = log_path.read_text(encoding="utf-8")
    assert "cache-hit age=" in log_text, (
        f"expected 'cache-hit age=' marker in log; got:\n{log_text}"
    )
    assert "cache-miss stale" not in log_text, (
        f"deleted marker 'cache-miss stale' still present in log; got:\n{log_text}"
    )
