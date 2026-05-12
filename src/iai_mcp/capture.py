"""Plan 06 memory_capture (WRITE-side ambient gap closure).

Context: prior phases shipped ambient READ (session_start compact handle) and
ambient daemon (sleep cycles, REM, overnight digest). WRITE-side capture of
conversation content was architectural gap — nothing in iai-mcp automatically
ingested what the user said or what Claude decided during a session.

This module closes that gap with two entry points:

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# R3 deviation [Rule 3 - blocking import cost]: `iai_mcp.embed` pulls
# in transformers + torch (~2.9s cold import). Loading capture.py for the
# `--no-spawn` deferred path (which never embeds anything) blew the R3 2s
# wall-clock budget. Moved to lazy import inside `capture_turn` — keeps the
# write_deferred_captures cold path under ~1s. `from __future__ import
# annotations` (line 29) keeps type hints intact without runtime import.
# `MemoryStore` left at module top — its 0.4s import is acceptable.
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


def _content_to_text(content: Any) -> str:
    """Extract plain text from Claude and Codex content block shapes."""
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"text", "input_text", "output_text"}:
                text_parts.append(str(block.get("text", "")))
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _extract_transcript_turn(obj: dict[str, Any]) -> tuple[str, str] | None:
    """Return (role, text) for supported Claude Code and Codex transcript rows."""
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    obj_type = obj.get("type")

    if obj_type == "response_item":
        # Codex response_item message rows can include injected context and
        # duplicate cleaner event_msg rows. Capture only event_msg turns.
        return None

    if obj_type == "event_msg":
        payload_type = payload.get("type")
        if payload_type == "user_message":
            text = _content_to_text(payload.get("message", ""))
            return ("user", text) if text else None
        if payload_type == "agent_message":
            text = _content_to_text(payload.get("message", ""))
            return ("assistant", text) if text else None
        return None

    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    role = obj_type if obj_type in {"user", "assistant"} else str(msg.get("role", ""))
    if role not in {"user", "assistant"}:
        return None
    text = _content_to_text(msg.get("content", ""))
    return (role, text) if text else None


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
    # `--no-spawn` deferred path (Phase 7.1 R3) which never embeds.
    from iai_mcp.embed import embedder_for_store

    emb = embedder_for_store(store).embed(cue or text)
    embedding = list(emb)

    # Dedup: query_similar against existing records at the same tier.
    # Phase 07.11-01 / query_similar accepts a `tier` kwarg natively
    # (Bug A fix), returns list[tuple[MemoryRecord, float]] (legacy contract,
    # unchanged shape -- we unpack the tuple correctly in the loop body, Bug B
    # fix), and the dedup hit reinforces via the typed `reinforce_record`
    # wrapper (Bug C fix -- single-uuid argument shape against a single-uuid
    # API).
    try:
        neighbours = store.query_similar(embedding, k=3, tier=tier)
    except (ValueError, IOError) as exc:
        # Genuinely-recoverable cases only: bad tier validation surfaces as
        # ValueError (already caught by query_similar's pre-I/O guard); transient
        # LanceDB I/O surfaces as IOError. A TypeError from a wrong call shape
        # MUST surface in tests -- the silent `except Exception: pass` blanket
        # is removed deliberately (D-01 contract).
        log.warning(
            "capture_dedup_query_failed",
            extra={"err_type": type(exc).__name__, "err": str(exc)[:120]},
        )
        neighbours = []

    for record, score in neighbours:  # tuple-unpack -- fix for Bug B
        if score >= DEDUP_COS_THRESHOLD:
            # Single-record reinforcement: route through reinforce_record
            #, NOT boost_edges([UUID(...)]) which expects pairs.
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
            turn = _extract_transcript_turn(obj)
            if turn is None:
                continue
            role, text = turn
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


# ---------------------------------------------------------------------------
# R3 / D7.1-04: deferred-captures writer for `--no-spawn` hook mode
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

    Used by ``iai-mcp capture-transcript --no-spawn`` (R3, D7.1-04) when the
    daemon is unreachable. The Stop hook calls this so it never blocks
    session teardown waiting for a daemon spawn (the third spawn vector
    forensic anomaly #3 from ``report-20260426-150300.md``).

    The daemon's drain loop (Plan 07.1-05b, in daemon.py / WAKE handler)
    consumes these on next WAKE. Format is JSONL v1 per D7.1-04:

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
        - Stdlib only: ``json``, ``time``, ``pathlib.Path``, ``datetime``, ``os``.
    """
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out_path = deferred_dir / f"{session_id}-{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        # Header (line 1, version=1 forward-compat marker per D7.1-04).
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
                turn = _extract_transcript_turn(obj)
                if turn is None:
                    continue
                role, text = turn
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
# R3 / D7.1-04: deferred-captures drain (READ side, daemon-resident)
# ---------------------------------------------------------------------------


def drain_deferred_captures(store: MemoryStore) -> dict[str, int]:
    """Consume ``~/.iai-mcp/.deferred-captures/*.jsonl`` produced by
    ``iai-mcp capture-transcript --no-spawn`` (Plan 07.1-05 WRITE side).

    For each ``.jsonl`` file in the deferred-captures dir:

    * Read line 1 (header). If ``version > 1`` (forward-compat guard), log a
      "skip" line to ``~/.iai-mcp/logs/deferred-drain-YYYY-MM-DD.log`` and
      leave the file in place — a future daemon version will know how to
      handle it.
    * For each event line (lines 2..N), call ``capture_turn(store, ...)``
      and inspect its return-status dict. W2 / D-02:
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
    for fpath in sorted(deferred_dir.glob("*.jsonl")):
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
            for ln in event_lines:
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
                elif status == "reinforced":
                    counts["events_reinforced"] += 1
                elif status == "skipped" and reason.startswith("insert-failed:"):
                    counts["events_skipped_insert_failed"] += 1
                    file_had_insert_failure = True
                    if file_first_error is None:
                        file_first_error = reason
                else:
                    counts["events_skipped_intentional"] += 1
            if file_had_insert_failure:
                # preserve the file as evidence — at least one
                # event hit the insert-failed code path inside capture_turn
                # (store.insert raised, capture_turn swallowed and returned
                # status=skipped reason=insert-failed:*). Pre-07.9 the file
                # was unlinked here and the data was silently lost.
                failed_path = fpath.with_suffix(f".failed-{int(time.time())}.jsonl")
                fpath.rename(failed_path)
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
                failed_path = fpath.with_suffix(f".failed-{int(time.time())}.jsonl")
                fpath.rename(failed_path)
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} failed "
                        f"{fpath.name}: {type(e).__name__}: {e}\n"
                    )
            except Exception:
                pass
            counts["files_failed"] += 1
    return counts
