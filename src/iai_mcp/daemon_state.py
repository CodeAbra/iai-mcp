# Canonical FSM state lives in ``lifecycle_state.json`` (LifecycleStateRecord).
# This module's ``fsm_state`` field is the legacy view; ``fsm_reconcile`` at
# daemon startup compares the two and warns on disagreement.
"""Atomic daemon state persistence.

State file at ~/.iai-mcp/.daemon-state.json holds:
- fsm_state               -- WAKE / TRANSITIONING / SLEEP / DREAMING
- daemon_started_at       -- ISO8601 UTC
- last_digest_shown_at    -- ISO8601 UTC, used by morning digest gate
- pending_digest          -- dict ready to surface in next memory_recall
- last_learned_at         -- last quiet-window learn timestamp
- last_session_ts         -- last observed session_started event ts

All writes via tempfile + os.replace (POSIX atomic rename). Crash-mid-write
leaves the old file intact; readers either see old complete or new complete,
never partial.

T-04-01 mitigation: atomic rename precludes torn writes.
T-04-07 mitigation: file mode 0o600 user-only.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

# morning-digest gating threshold. The digest is surfaced only when it
# has been at least this many hours since the last show (or has never shown).
DIGEST_SHOW_THRESHOLD_HOURS: int = 18

# first_turn_pending eviction guards. A session is considered stale once it
# has sat in the dict for longer than FIRST_TURN_TTL_HOURS -- typically it
# means the client died before consuming the flag, so the entry will never
# be popped by ``consume_first_turn``. MAX_FIRST_TURN_ENTRIES caps the dict
# as a secondary safety net when many sessions open in a short window.
FIRST_TURN_TTL_HOURS: int = 24
MAX_FIRST_TURN_ENTRIES: int = 100


def load_state() -> dict:
    """Read the state file; return {} if missing or malformed (self-heal)."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt file -- return empty dict; next save_state writes fresh.
        return {}


def save_state(state: dict) -> None:
    """Atomically persist state via tempfile + os.replace.

    Semantics:
    - Creates parent dir if missing.
    - Writes to a sibling temp file in the same directory (required so
      os.replace can do an atomic rename on the same filesystem).
    - fsync the file contents before rename so the data is on disk.
    - chmod 0o600 before the swap so the visible file is never world-readable.
    - On exception: unlink the temp file so `/tmp` doesn't accumulate.
    """
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".daemon-state.",
        suffix=".tmp",
        dir=str(STATE_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, STATE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def prune_stale_first_turn(
    state: dict,
    now: datetime | None = None,
    ttl_hours: int = FIRST_TURN_TTL_HOURS,
    max_entries: int = MAX_FIRST_TURN_ENTRIES,
) -> int:
    """Evict first_turn_pending entries older than ``ttl_hours`` and cap the
    dict at ``max_entries`` (keep newest by timestamp). Returns the number
    of entries removed.

    Accepts legacy values ``True`` / ``False`` as "unknown timestamp" and
    stamps them with ``now`` so they age out on the next prune. Idempotent;
    safe to call on every save.
    """
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return 0

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(hours=ttl_hours)

    def _as_dt(value: object) -> datetime:
        """Parse stored value into an aware datetime; unknown -> epoch (evict).

        Legacy bool / malformed strings are treated as "stale, evict now" —
        they cannot be aged sensibly without a real timestamp, and the
        former "stamp with current" behaviour kept the dict from ever
        draining when clients died before writing ISO timestamps.
        """
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return datetime.fromtimestamp(0, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)

    # Normalise every entry to an ISO timestamp string so downstream
    # callers see a consistent value shape after the first prune.
    removed = 0
    for sid, value in list(pending.items()):
        dt = _as_dt(value)
        if dt < cutoff:
            pending.pop(sid, None)
            removed += 1
        elif not isinstance(value, str):
            pending[sid] = dt.isoformat()

    # Secondary cap — keep the newest ``max_entries`` by timestamp.
    if len(pending) > max_entries:
        ordered = sorted(
            pending.items(),
            key=lambda kv: _as_dt(kv[1]),
            reverse=True,
        )
        keep = dict(ordered[:max_entries])
        removed += len(pending) - len(keep)
        state["first_turn_pending"] = keep

    return removed


def mark_session_opened(state: dict, session_id: str) -> None:
    """ / D5-03: mark first_turn_pending for a session.

    Stores the opening timestamp as the dict value so ``prune_stale_first_turn``
    can evict entries whose client died before consuming the flag. Opportunistic
    prune on every mark keeps the dict bounded without a dedicated reaper.

    Idempotent. Persistence is the caller's responsibility (typical callers:
    concurrency socket handler; tests directly).
    """
    if not isinstance(session_id, str) or not session_id:
        return
    pending = state.setdefault("first_turn_pending", {})
    pending[session_id] = datetime.now(timezone.utc).isoformat()
    prune_stale_first_turn(state)


def consume_first_turn(state: dict, session_id: str) -> bool:
    """Return True iff first call for session; atomic pop+save.

    D5-03: the first memory_recall in a session consumes the
    flag so subsequent recalls bypass the first-turn hook.
    """
    try:
        pending = state.get("first_turn_pending")
        if not isinstance(pending, dict):
            return False
        if pending.pop(session_id, False):
            try:
                save_state(state)
            except Exception:
                # save failure is non-fatal — returning True still triggers
                # the hook exactly once in-process; cross-process atomicity
                # is best-effort.
                pass
            return True
        return False
    except Exception:
        return False


# R3 (per / / ): a per-tick + startup
# reaper for stale `first_turn_pending` entries with a 1-hour TTL and a
# tuple return shape (updated_state, dropped_session_ids).
#
# Distinct from `prune_stale_first_turn` above which has a 24h ceiling and
# is opportunistically invoked from `mark_session_opened`. Both helpers
# coexist by design (researcher finding #1 + advisor recommendation):
# - `prune_stale_first_turn` keeps its 24h opportunistic path on session-open;
# - `prune_first_turn_pending` is the per-tick + startup reaper that needs
#   the dropped IDs back so the caller can emit
# `kind=first_turn_pending_expired` events .
#
# Pure function — no I/O. Caller is responsible for `save_state(state)`
# and the event emit. Idempotent; safe on empty/missing input.

FIRST_TURN_PENDING_TTL_SEC_DEFAULT: float = 3600.0  # D7.2-08 1h default


def prune_first_turn_pending(
    state: dict,
    now: datetime | None = None,
    ttl_sec: float = FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
) -> tuple[dict, list[str]]:
    """R3: drain stale `first_turn_pending` entries.

    Returns (updated_state_dict, dropped_session_ids). Pure function —
    does NOT call save_state; does NOT emit events. Caller decides
    persistence + event emission.

    Eviction rules:
    - String value parsed as ISO timestamp; entry evicts if (now - ts) >= ttl_sec.
    - Non-string value (legacy bool / dict / None) treated as stale → evict.
      Matches the established behavior of `prune_stale_first_turn` for
      legacy entries (cannot be aged sensibly without a timestamp).
    - Naive timestamps assumed UTC.
    - Malformed ISO strings → evict (defensive against corruption).

    Distinct from `prune_stale_first_turn` (24h default, returns int);
    this helper is per-tick + startup with a shorter TTL and visibility
    into which sessions were dropped ( event payload needs the
    session_ids list).
    """
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return state, []

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(seconds=ttl_sec)

    dropped: list[str] = []
    fresh: dict = {}
    for sid, value in pending.items():
        if isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                dropped.append(sid)
                continue
            if ts < cutoff:
                dropped.append(sid)
                continue
            fresh[sid] = value
        else:
            # Legacy bool / dict / None / number — no recoverable timestamp.
            dropped.append(sid)

    state["first_turn_pending"] = fresh
    return state, dropped


def get_pending_digest(state: dict, now: datetime) -> dict | None:
    """: return pending morning digest if eligible, else None.

    Eligibility gate: >= DIGEST_SHOW_THRESHOLD_HOURS since last_digest_shown_at
    OR never shown. When returned, the digest is consumed from state and
    last_digest_shown_at is advanced to `now`; state is persisted via
    save_state so the same digest never appears twice in the same 18h window.
    """
    last_shown = state.get("last_digest_shown_at")
    if last_shown:
        try:
            last_dt = datetime.fromisoformat(last_shown)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
            if now_cmp - last_dt < timedelta(hours=DIGEST_SHOW_THRESHOLD_HOURS):
                return None
        except (TypeError, ValueError):
            # Malformed timestamp -- treat as never shown, fall through.
            pass

    digest = state.get("pending_digest")
    if not digest:
        return None

    now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    state["last_digest_shown_at"] = now_cmp.isoformat()
    state.pop("pending_digest", None)
    save_state(state)
    return digest
