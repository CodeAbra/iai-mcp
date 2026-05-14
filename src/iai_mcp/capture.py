"""memory_capture — WRITE-side ambient capture.

This module provides two entry points for capturing conversation content
into the iai-mcp store:

1. `capture_turn(store, cue, text, tier, session_id)`:
   in-session, explicit. Called via MCP tool `memory_capture` when Claude
   detects a surprising correction, load-bearing decision, or lesson.

2. `capture_transcript(store, transcript_path, session_id)`:
   end-of-session, ambient. Called by `~/.claude/hooks/iai-mcp-session-capture.sh`
   Stop-hook on SessionEnd. Reads Claude Code JSONL transcript, extracts
   user + assistant turns, filters through shield + dedup, inserts records.

Both paths respect:
- Shield: HARD_BLOCK drops the record; FLAG_FOR_REVIEW stores with tag
  (policy: user chose visibility over paranoia, 2026-04-20).
- Dedup: if query_similar returns a hit with cos >= DEDUP_THRESHOLD
  (0.95), we reinforce instead of insert (boost Hebbian edge).
- Language: detected via langdetect; falls back to 'en' on ambiguity.
- Encryption: goes through the standard store.insert() path which handles
  AES-256-GCM column encryption.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# Per-pass event cap for drain_deferred_captures. When a single file would
# push the running total past this threshold, the unprocessed remainder is
# rewritten to {basename}.partial.jsonl (header preserved) for the next pass.
MAX_DRAIN_EVENTS_PER_RUN = 5000

# Matches the active-writer marker name shape exactly: `{anything}.live.jsonl`
# with no hyphen-epoch between `.live` and `.jsonl`. The Stop-hook rename
# target `{id}.live-{epoch}.jsonl` does NOT match (the `-` after `.live`
# breaks the pattern).
_LIVE_ACTIVE_RE = re.compile(r"\.live\.jsonl$")

# Deviation: blocking import cost — `iai_mcp.embed` pulls in transformers +
# torch (~2.9s cold import). Loading capture.py for the `--no-spawn` deferred
# path (which never embeds anything) would exceed the 2s wall-clock budget.
# Moved to lazy import inside `capture_turn` — keeps the write_deferred_captures
# cold path under ~1s. `from __future__ import annotations` (line 29) keeps
# type hints intact without runtime import. `MemoryStore` left at module top —
# its 0.4s import is acceptable.
from iai_mcp.store import MemoryStore
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    TIER_ENUM,
    MemoryRecord,
)

log = logging.getLogger(__name__)

DEDUP_COS_THRESHOLD = 0.95
MIN_CAPTURE_LEN = 12
MAX_CAPTURE_LEN = 8000

# Bounded retry policy for `.failed-*` deferred-capture evidence files.
# A file is retried up to FAILED_MAX_ATTEMPTS times with exponential
# backoff (60s, 120s, 240s); after that it transitions to
# .permanent-failed-<ts>.jsonl as a terminal evidence state and a
# `permanent_capture_failure` event is emitted at severity=critical.
FAILED_MAX_ATTEMPTS: int = 3
FAILED_BACKOFF_BASE_SEC: float = 60.0

_FAILED_ATTEMPT_RE = re.compile(r"-attempt-(\d+)\.jsonl$")
_FAILED_SHAPE_RE = re.compile(r"^(.+?)\.failed-(\d+)(?:-attempt-\d+)?\.jsonl$")


def _parse_failed_attempt(name: str) -> int:
    """Return the prior-attempt count encoded in a deferred-capture filename.

    Mapping:
      * ``<base>.failed-<ts>-attempt-<n>.jsonl`` -> n
      * ``<base>.failed-<ts>.jsonl`` (legacy shape) -> 1
      * any name without ``.failed-`` (clean file) -> 0
    """
    m = _FAILED_ATTEMPT_RE.search(name)
    if m:
        return int(m.group(1))
    if ".failed-" in name:
        return 1
    return 0


def _advance_failed_path(
    fpath: Path,
    store: "MemoryStore",
    *,
    first_error: str,
    log_path: Path,
) -> Path:
    """Rename ``fpath`` forward to the next-attempt or terminal evidence shape.

    Clean -> attempt-1, attempt-N -> attempt-(N+1), and on the fourth
    failure (would-be-attempt-4) the file transitions to
    ``.permanent-failed-<ts>.jsonl`` and a ``permanent_capture_failure``
    event is emitted at severity=critical.
    """
    prior_attempt = _parse_failed_attempt(fpath.name)
    next_attempt = prior_attempt + 1
    m = _FAILED_SHAPE_RE.match(fpath.name)
    if m:
        base = m.group(1)
        ts_str = m.group(2)
    else:
        base = fpath.stem
        ts_str = str(int(time.time()))
    if next_attempt > FAILED_MAX_ATTEMPTS:
        new_name = f"{base}.permanent-failed-{ts_str}.jsonl"
        failed_path = fpath.with_name(new_name)
        fpath.rename(failed_path)
        try:
            from iai_mcp.events import write_event

            write_event(
                store,
                "permanent_capture_failure",
                {
                    "file": new_name,
                    "first_error": first_error,
                    "attempts": FAILED_MAX_ATTEMPTS,
                },
                severity="critical",
                domain="ops",
            )
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"permanent_capture_failure-event-skipped {new_name}\n"
                    )
            except Exception:
                pass
        return failed_path
    new_name = f"{base}.failed-{ts_str}-attempt-{next_attempt}.jsonl"
    failed_path = fpath.with_name(new_name)
    fpath.rename(failed_path)
    return failed_path


def _detect_language(text: str) -> str:
    """Best-effort ISO-639-1 via langdetect; 'en' on any failure."""
    try:
        from langdetect import detect  # lazy: already a project dep

        code = detect(text[:500])
        return code if len(code) == 2 else "en"
    except Exception:
        return "en"


def _run_shield(text: str) -> tuple[str, list[str]]:
    """Run shield; return (verdict, tags) where verdict in HARD_BLOCK|FLAG|OK."""
    try:
        from iai_mcp.shield import evaluate

        result = evaluate(text)
        verdict = getattr(result, "verdict", "OK")
        tags = list(getattr(result, "tags", []) or [])
        return verdict, tags
    except Exception:
        return "OK", []


def capture_turn(
    store: MemoryStore,
    *,
    cue: str,
    text: str,
    tier: str = "episodic",
    session_id: str = "-",
    role: str = "user",
) -> dict[str, Any]:
    """Write a single conversation turn to the iai-mcp store.

    Returns {"status": "inserted|reinforced|skipped", "record_id": uuid-or-null,
             "reason": short-string}.
    """
    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    if len(text) < MIN_CAPTURE_LEN:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    verdict, shield_tags = _run_shield(text)
    if verdict == "HARD_BLOCK":
        return {"status": "skipped", "record_id": None, "reason": "shield HARD_BLOCK"}

    # Lazy import: keeps the cold module-import cost low for the
    # `--no-spawn` deferred path which never embeds.
    from iai_mcp.embed import embedder_for_store

    emb = embedder_for_store(store).embed(cue or text)
    embedding = list(emb)

    # Dedup: query_similar against existing records at the same tier.
    # query_similar accepts a `tier` kwarg natively, returns
    # list[tuple[MemoryRecord, float]] (legacy contract, unchanged shape --
    # we unpack the tuple correctly in the loop body), and the dedup hit
    # reinforces via the typed `reinforce_record` wrapper (single-uuid
    # argument shape against a single-uuid API).
    try:
        neighbours = store.query_similar(embedding, k=3, tier=tier)
    except (ValueError, IOError) as exc:
        # Genuinely-recoverable cases only: bad tier validation surfaces as
        # ValueError (already caught by query_similar's pre-I/O guard); transient
        # LanceDB I/O surfaces as IOError. A TypeError from a wrong call shape
        # MUST surface in tests -- the silent `except Exception: pass` blanket
        # is removed deliberately.
        log.warning(
            "capture_dedup_query_failed",
            extra={"err_type": type(exc).__name__, "err": str(exc)[:120]},
        )
        neighbours = []

    for record, score in neighbours:  # tuple-unpack (MemoryRecord, float)
        if score >= DEDUP_COS_THRESHOLD:
            # Single-record reinforcement: route through reinforce_record,
            # NOT boost_edges([UUID(...)]) which expects pairs.
            try:
                store.reinforce_record(record.id)
            except (ValueError, IOError) as exc:
                # Reinforce is best-effort observability; log and continue
                # so the duplicate is still detected even if the LTP write
                # fails. Same narrowed-except discipline as the query above.
                log.warning(
                    "capture_dedup_reinforce_failed",
                    extra={
                        "err_type": type(exc).__name__,
                        "record_id": str(record.id),
                    },
                )
            return {
                "status": "reinforced",
                "record_id": str(record.id),
                "reason": f"cos={score:.3f} >= {DEDUP_COS_THRESHOLD}",
            }

    tags = ["capture", f"role:{role}"]
    if verdict == "FLAG_FOR_REVIEW":
        tags.append("shield:flagged")
        tags.extend(f"shield:{t}" for t in shield_tags[:3])

    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": cue or "(auto-capture)",
                     "session_id": session_id, "role": role}],
        created_at=now,
        updated_at=now,
        tags=tags,
        language=_detect_language(text),
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )

    try:
        store.insert(rec)
    except Exception as e:
        log.exception("capture_turn insert failed")
        return {"status": "skipped", "record_id": None, "reason": f"insert-failed: {type(e).__name__}"}

    return {"status": "inserted", "record_id": str(rec.id), "reason": f"tier={tier}"}


def capture_transcript(
    store: MemoryStore,
    transcript_path: Path | str,
    *,
    session_id: str = "-",
    max_turns: int = 200,
) -> dict[str, Any]:
    """Read a Claude Code JSONL transcript, capture user + assistant turns.

    Returns {"inserted": N, "reinforced": M, "skipped": K, "errors": E}.
    """
    path = Path(transcript_path).expanduser()
    if not path.exists():
        return {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 1,
                "reason": f"transcript not found: {path}"}

    counts = {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 0}
    seen = 0
    with path.open() as fh:
        for line in fh:
            if seen >= max_turns:
                break
            seen += 1
            try:
                obj = json.loads(line)
            except Exception:
                counts["errors"] += 1
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
            role = obj.get("type") or msg.get("role", "")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # Claude Code messages use block format; collect text blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts).strip()
            else:
                text = str(content).strip()
            if not text:
                continue
            result = capture_turn(
                store,
                cue=f"session {session_id} turn {seen}",
                text=text,
                tier="episodic",
                session_id=session_id,
                role=role,
            )
            status = result.get("status", "skipped")
            if status in counts:
                counts[status] += 1
            else:
                counts["skipped"] += 1

    return counts


def _parse_transcript_line(line: str) -> tuple[str, str] | None:
    """Parse a Claude Code transcript JSONL line into (role, text).

    Returns None if the line is not a user/assistant turn, has empty text,
    or cannot be JSON-decoded. Used by both `capture_transcript` and the
    deferred-write paths to keep parsing rules in one place.
    """
    try:
        obj = json.loads(line)
    except Exception:
        return None
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    role = obj.get("type") or msg.get("role", "")
    if role not in {"user", "assistant"}:
        return None
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(parts).strip()
    else:
        text = str(content).strip()
    if not text:
        return None
    return role, text


def write_deferred_event(
    session_id: str,
    role: str,
    text: str,
    *,
    cwd: str | None = None,
) -> Path:
    """Append a single JSONL event to `{session_id}.live.jsonl`.

    Creates the file with a header on first call; appends events on
    subsequent calls. Pure file IO — no daemon socket, no embedder,
    no shield, no LanceDB.

    The drain function skips files matching the exact `*.live.jsonl`
    suffix, so the writer/drain race is structurally impossible while
    this file is the active marker. The Stop hook renames the file to
    `{session_id}.live-{epoch}.jsonl` at session end; the drain then
    picks it up.

    Format invariants are duplicated by the per-turn shell hook at
    deploy/hooks/iai-mcp-turn-capture.sh (system-python inline) — keep
    header/event keys in sync.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    need_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a") as fh:
        if need_header:
            header = {
                "version": 1,
                "deferred_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "cwd": cwd or os.getcwd(),
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        event = {
            "text": text,
            "cue": f"session {session_id} turn",
            "tier": "episodic",
            "role": role,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Deferred-captures writer for `--no-spawn` hook mode
# ---------------------------------------------------------------------------


def write_deferred_captures(
    session_id: str,
    transcript_path: Path | str,
    *,
    cwd: str | None = None,
    max_turns: int = 200,
) -> Path:
    """Defer transcript capture by writing events to a JSONL file under
    ``~/.iai-mcp/.deferred-captures/``. Returns the path written.

    Used by ``iai-mcp capture-transcript --no-spawn`` when the daemon is
    unreachable. The Stop hook calls this so it never blocks session teardown
    waiting for a daemon spawn.

    The daemon's drain loop (in daemon.py / WAKE handler) consumes these on
    next WAKE. Format is JSONL v1:

    - Line 1: header ``{"version":1,"deferred_at":<ISO>,"session_id":<id>,"cwd":<path>}``
    - Lines 2..N: one event per user/assistant turn
      ``{"text":<verbatim>,"cue":<short>,"tier":"episodic","role":<u|a>,"ts":<ISO>}``

    Pure-write: no MemoryStore touch, no socket touch, no daemon import.
    Uses ``Path.home()`` at call time so HOME-monkeypatched tests get the
    right tmp dir. Idempotent ``mkdir(parents=True, exist_ok=True)``.

    Args:
        session_id: Claude Code session id (provenance + filename component).
        transcript_path: path to the JSONL transcript file (or non-existent —
            we write the header then return; daemon drain treats as no-op).
        cwd: optional CWD override for the header (defaults to ``os.getcwd()``).
        max_turns: cap on transcript turns to emit (default 200, matches
            ``capture_transcript`` semantics).

    Returns:
        ``Path`` of the written ``.jsonl`` file.

    Notes:
        - Filename pattern ``{session_id}-{int(time.time())}.jsonl`` — the
          unix-ts suffix avoids collisions if the same session captures
          multiple times.
        - Reuses the same parsing logic as ``capture_transcript`` so the
          deferred path and the inline path stay consistent.
        - Returns even on missing transcript (writes header only) — daemon
          drain treats as no-op. Hook MUST never raise here.
        - Stdlib only: ``json``, ``time``, ``pathlib.Path``, ``datetime``,
          ``os``.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{session_id}-{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        # Header (line 1, version=1 forward-compat marker).
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": cwd or os.getcwd(),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        # Read transcript and emit one event per user/assistant turn.
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return out_path  # empty body — daemon drain will treat as no-op
        seen = 0
        with path.open() as src:
            for line in src:
                if seen >= max_turns:
                    break
                seen += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
                role = obj.get("type") or msg.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    text = "\n".join(text_parts).strip()
                else:
                    text = str(content).strip()
                if not text:
                    continue
                event = {
                    "text": text,
                    "cue": f"session {session_id} turn {seen}",
                    "tier": "episodic",
                    "role": role,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Deferred-captures drain (READ side, daemon-resident)
# ---------------------------------------------------------------------------


def drain_deferred_captures(store: MemoryStore) -> dict[str, int]:
    """Consume ``~/.iai-mcp/.deferred-captures/*.jsonl`` produced by
    ``iai-mcp capture-transcript --no-spawn``.

    For each ``.jsonl`` file in the deferred-captures dir:

    * Read line 1 (header). If ``version > 1`` (forward-compat guard), log a
      "skip" line to ``~/.iai-mcp/logs/deferred-drain-YYYY-MM-DD.log`` and
      leave the file in place — a future daemon version will know how to
      handle it.
    * For each event line (lines 2..N), call ``capture_turn(store, ...)``
      and inspect its return-status dict:
      - status="inserted"  → events_inserted += 1
      - status="reinforced" → events_reinforced += 1
      - status="skipped" with reason matching ^insert-failed:* (capture_turn
        path where store.insert raised) → events_skipped_insert_failed += 1
        and the WHOLE FILE is treated as failed: renamed to
        .failed-<ts>.jsonl, NOT unlinked.
      - status="skipped" with any other reason (shield HARD_BLOCK, too short,
        invalid tier — all *intentional* drops) → events_skipped_intentional
        += 1.
    * On full success (zero insert-failed events): delete the file,
      files_drained += 1.
    * On any insert-failed event: rename the file to
      ``<basename>.failed-<unix_ts>.jsonl`` (preserves evidence for manual
      inspection), log a "insert-failed" line with the first error,
      files_failed += 1.
    * On parser/header exception: same outer rename + log path as before
      (existing behaviour), files_failed += 1.
    * On 0-byte / empty file: delete it (no-op header-only deferral).

    Idempotent: re-running on a directory with no ``.jsonl`` files (or no
    deferred-captures dir at all) returns zero counts without error.

    Returns dict with keys:
        files_drained, files_failed,
        events_inserted, events_reinforced,
        events_skipped_intentional, events_skipped_insert_failed.

    Notes:
        - Uses ``Path.home()`` at call time so HOME-monkeypatched tests get
          the right tmp dir.
        - Stdlib only — no new deps.
        - Caller (daemon.main / _tick_body) MUST wrap in try/except so a
          drain crash never propagates into the asyncio event loop. This
          function itself catches per-file exceptions defensively.
        - The ``store`` argument is the same MemoryStore instance the
          daemon uses for all other writes (so connection/lock semantics
          are consistent). Drain MUST run inside ``asyncio.to_thread`` from
          async callers because ``capture_turn`` does sync LanceDB I/O.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    log_dir = Path.home() / ".iai-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir / f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    counts = {
        "files_drained": 0,
        "files_failed": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped_intentional": 0,
        "events_skipped_insert_failed": 0,
    }
    if not deferred_dir.exists():
        return counts
    total_events_processed = 0
    cap_hit = False
    # Iterate every .jsonl entry but skip exact active-writer markers and
    # *.failed-*.jsonl files (preserved evidence from prior drains).
    candidates = []
    for fpath in sorted(deferred_dir.iterdir()):
        if fpath.suffix != ".jsonl":
            continue
        if _LIVE_ACTIVE_RE.search(fpath.name):
            continue
        # Terminal evidence — never reprocess.
        if ".permanent-failed-" in fpath.name:
            continue
        # `.failed-<ts>[-attempt-N].jsonl` is a retry candidate when its
        # per-attempt backoff has elapsed. Younger files are left untouched
        # so a flaky downstream can recover before the file ages forward.
        if ".failed-" in fpath.name:
            attempt_n = _parse_failed_attempt(fpath.name)
            backoff_sec = FAILED_BACKOFF_BASE_SEC * (2 ** (attempt_n - 1))
            try:
                file_mtime = fpath.stat().st_mtime
            except OSError:
                continue
            if (time.time() - file_mtime) < backoff_sec:
                continue
        candidates.append(fpath)

    for fpath in candidates:
        if cap_hit:
            break
        file_had_insert_failure = False
        file_first_error: str | None = None
        try:
            with fpath.open() as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
            if not lines:
                # Empty file (e.g. partial write that never got header) — drop.
                fpath.unlink()
                continue
            header = json.loads(lines[0])
            if header.get("version", 0) > 1:
                # Forward-compat guard: leave the file in place; a future
                # daemon revision will know the format. Log + continue.
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} skip {fpath.name}: "
                        f"version={header.get('version')}\n"
                    )
                continue
            session_id = header.get("session_id", "-")
            event_lines = lines[1:]
            processed_in_file = 0
            for idx, ln in enumerate(event_lines):
                if total_events_processed >= MAX_DRAIN_EVENTS_PER_RUN:
                    # Cap reached mid-file — write the unprocessed remainder
                    # to {basename}.partial.jsonl atomically and unlink the
                    # original only after the partial is durable on disk.
                    remainder = event_lines[idx:]
                    partial_path = fpath.with_suffix(".partial.jsonl")
                    tmp_path = fpath.with_suffix(".partial.tmp")
                    with tmp_path.open("w") as ph:
                        ph.write(lines[0] + "\n")
                        for r in remainder:
                            ph.write(r + "\n")
                        ph.flush()
                        os.fsync(ph.fileno())
                    os.replace(tmp_path, partial_path)
                    fpath.unlink()
                    counts["files_drained"] += 1
                    cap_hit = True
                    break
                ev = json.loads(ln)
                # Reuse capture_turn so the deferred path lands in the same
                # shield + dedup + encryption pipeline as live captures.
                result = capture_turn(
                    store,
                    cue=ev.get("cue", ""),
                    text=ev.get("text", ""),
                    tier=ev.get("tier", "episodic"),
                    session_id=session_id,
                    role=ev.get("role", "user"),
                )
                status = result.get("status", "skipped")
                reason = result.get("reason", "")
                if status == "inserted":
                    counts["events_inserted"] += 1
                    try:
                        from iai_mcp.memory_bank import append_recent_record

                        rid_str = result.get("record_id")
                        if rid_str:
                            rec = store.get(UUID(rid_str))
                            if rec is not None:
                                append_recent_record(store, rec)
                    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
                        log.warning(
                            "bank-recent append failed for record %s",
                            result.get("record_id"),
                            exc_info=True,
                        )
                elif status == "reinforced":
                    counts["events_reinforced"] += 1
                elif status == "skipped" and reason.startswith("insert-failed:"):
                    counts["events_skipped_insert_failed"] += 1
                    file_had_insert_failure = True
                    if file_first_error is None:
                        file_first_error = reason
                else:
                    counts["events_skipped_intentional"] += 1
                total_events_processed += 1
                processed_in_file += 1
            if cap_hit:
                break
            if file_had_insert_failure:
                # Preserve the file as evidence — at least one event hit
                # the insert-failed code path inside capture_turn (store.insert
                # raised, capture_turn swallowed and returned status=skipped
                # reason=insert-failed:*).
                failed_path = _advance_failed_path(
                    fpath,
                    store,
                    first_error=file_first_error or "unknown",
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} insert-failed "
                        f"{fpath.name}: first_error={file_first_error}\n"
                    )
                counts["files_failed"] += 1
            else:
                fpath.unlink()
                counts["files_drained"] += 1
        except Exception as e:  # noqa: BLE001 -- per-file isolation, never raise
            try:
                # Preserve evidence: rename so the next drain pass skips it
                # AND a human can inspect the failure.
                failed_path = _advance_failed_path(
                    fpath,
                    store,
                    first_error=file_first_error or repr(e),
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} failed "
                        f"{fpath.name}: {type(e).__name__}: {e}\n"
                    )
            except Exception:
                pass
            counts["files_failed"] += 1
    try:
        from iai_mcp.memory_bank import prune_recent_windows

        prune_recent_windows()
    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning("bank-recent prune failed", exc_info=True)
    return counts
