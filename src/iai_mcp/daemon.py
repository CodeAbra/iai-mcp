"""IAI-MCP Sleep Daemon main entry point.


Constitutional guards:
- C1 HUMAN-FIRST: daemon NEVER starts heavy ops while ANY MCP active. _tick_body
  calls `lock.try_acquire_exclusive` and yields immediately on False. Between
  REM cycles, `_check_still_exclusive(lock)` probes `holds_exclusive_nb` and
  the cycle loop breaks if MCP acquired a shared lock mid-night.
- C-USER-CONSENT: daemon NEVER initiates sleep mode without explicit user
  consent; consent gate lives in bedtime.py.
- C3: ZERO API cost. This module does NOT reference the paid-API env var;
  claude_cli.py is wired with env scrubbed at subprocess creation.
- C4: Clean uninstall via signal.SIGTERM -> shutdown event -> task cancel +
  lock.close + state persisted. launchd/systemd stop this daemon cleanly.
- C5: Literal preservation -- daemon never assigns to record.literal_surface.
  Called modules (sleep.py / schema.py) respect this by design.
  Grep-guarded by tests/test_constitutional_guards.py.
- C6: S5 audit runs read-only (MVCC); spawned as an independent task alongside
  the scheduler so it continues even when the scheduler is blocked on a heavy op.


The scheduler tick loop only emits `tick_error` events on exception; it never
crashes. _tick_body implements: empty-store shortcut, quiet-window re-learn,
bootstrap fallback, lock acquire with C1 yield, N-cycle REM loop via
`dream.run_rem_cycle`, FSM transitions, pending_digest accumulation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

from iai_mcp import s4
from iai_mcp.concurrency import ProcessLock, serve_control_socket  # noqa: F401 -- kept for backward compat (serve_control_socket STAYS in concurrency.py for the test suite)
from iai_mcp.daemon_state import load_state, save_state
from iai_mcp.dream import run_rem_cycle
from iai_mcp.events import write_event
from iai_mcp import maintenance as _maintenance
from iai_mcp.identity_audit import continuous_audit
from iai_mcp.maintenance import optimize_lance_storage
from iai_mcp.quiet_window import (
    BUCKET_COUNT,
    BUCKET_MINUTES,
    learn_quiet_window,
    should_bootstrap_trigger,
    should_relearn,
)
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.tz import load_user_tz

# ---------------------------------------------------------------------------
# State machine constants
# ---------------------------------------------------------------------------

STATE_WAKE: str = "WAKE"
STATE_TRANSITIONING: str = "TRANSITIONING"
STATE_SLEEP: str = "SLEEP"
STATE_DREAMING: str = "DREAMING"

# Valid FSM edges. DREAMING must return via SLEEP on wake.
VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_WAKE: {STATE_TRANSITIONING},
    STATE_TRANSITIONING: {STATE_SLEEP, STATE_WAKE},
    STATE_SLEEP: {STATE_DREAMING, STATE_WAKE},
    STATE_DREAMING: {STATE_SLEEP},
}

# Scheduler tick cadence (seconds). Light tick every 30s; hourly / 3h / 24h
# periodic work is gated inside _tick_body by last-ran timestamps.
TICK_INTERVAL_SEC: int = 30

# default cycle count per quiet window (biologically typical 4-5).
DEFAULT_CYCLE_COUNT: int = 4

# Hourly cadence for the S4 offline pass (FSRS wall-clock decay + viability scan).
# Matches the sigma snapshot cadence in identity_audit so the daemon has a single
# coherent "hourly heartbeat" of diagnostics.
S4_OFFLINE_INTERVAL_SEC: int = 60 * 60

# .6 W1: startup grace period before the FIRST iteration of
# `_s4_offline_loop`. The S4 offline pass walks the full graph and on cold
# caches calls `runtime_graph_cache.save -> json.dumps`, materialising a
# multi-GB intermediate string (: py-spy 2026-04-29 PID 7959
# RSS 7.6GB). Default = S4_OFFLINE_INTERVAL_SEC (1h, matching steady-state
# cadence). Set to 0 for tests / explicit warm-start. Env override
# IAI_MCP_S4_FIRST_ITER_GRACE_SEC.
S4_FIRST_ITER_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_S4_FIRST_ITER_GRACE_SEC", str(S4_OFFLINE_INTERVAL_SEC)),
)


# ---------------------------------------------------------------------------
# WAKE -> DROWSY drain edge helpers
# ---------------------------------------------------------------------------


def _should_drain_on_drowsy_edge(prev, current) -> bool:
    """True iff this is the edge into DROWSY (prev=WAKE, current=DROWSY)."""
    from iai_mcp.lifecycle_state import LifecycleState as _L
    return prev is _L.WAKE and current is _L.DROWSY


def _run_drowsy_drain(store, *, drain_fn, write_event_fn) -> None:
    """Run drain and emit one bookkeeping event.


    Writes ``deferred_drain_drowsy`` only when work was done; on exception
    swallows and writes ``deferred_drain_failed`` with ``phase='drowsy'``.
    Silent on zero-work to avoid log noise.
    """
    try:
        result = drain_fn(store)
    except Exception as e:  # noqa: BLE001 -- lifecycle_tick MUST NOT crash
        try:
            write_event_fn(
                store,
                "deferred_drain_failed",
                {"error": str(e)[:200], "phase": "drowsy"},
                severity="warning",
            )
        except Exception:
            pass
        return
    if not isinstance(result, dict):
        return
    if result.get("files_drained") or result.get("files_failed"):
        try:
            write_event_fn(
                store,
                "deferred_drain_drowsy",
                result,
                severity="info",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SessionStart precache
# ---------------------------------------------------------------------------

# Cached recall payload written once per REM-loop completion so the
# SessionStart hook reads a file instead of dispatching a JSON-RPC
# call into core (which blocks on the exclusive store lock during
# DREAMING). Path mirrors the hook's $HOME/.iai-mcp/.session-start-payload.cached.md.
SESSION_START_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"
SESSION_START_CACHE_MAX_CHARS = 10000


def _write_session_start_cache(store, *, cache_path: Path = SESSION_START_CACHE_PATH) -> None:
    """Best-effort: write the session-start markdown payload to cache_path.

    Atomic via tmp-file + fsync + os.replace. Any exception is swallowed
    after a best-effort write_event(..., severity="warning"). This MUST
    NOT propagate into the REM loop.

    Uses `_compose_session_start_payload` (emit-free) so no
    `session_started` event is written per REM cycle. The live
    `core.dispatch` path continues to use `assemble_session_start` and
    still emits one event per real session.
    """
    try:
        from iai_mcp import retrieve
        from iai_mcp.session import (
            _compose_session_start_payload,
            format_payload_as_markdown,
        )

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id="precache",
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if not rendered:
            return
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, cache_path)
    except Exception as exc:  # noqa: BLE001 -- cache write MUST NOT crash the REM loop
        try:
            write_event(
                store,
                "session_start_cache_write_failed",
                {"error": str(exc)[:200]},
                severity="warning",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# State machine transitions (separated so tests can exercise directly)
# ---------------------------------------------------------------------------

def transition(state: dict, new_fsm: str) -> None:
    """Attempt the WAKE/TRANSITIONING/SLEEP/DREAMING edge.


    Raises ValueError when the edge is not in VALID_TRANSITIONS. Persists
    the new fsm_state + fsm_transition_at via save_state.
    """
    current = state.get("fsm_state", STATE_WAKE)
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_fsm not in allowed:
        raise ValueError(
            f"Illegal transition {current} -> {new_fsm}; allowed: {sorted(allowed)}"
        )
    state["fsm_state"] = new_fsm
    state["fsm_transition_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


# ---------------------------------------------------------------------------
# Helpers used by _tick_body
# ---------------------------------------------------------------------------

def _store_is_empty(store: MemoryStore) -> bool:
    """Return True when the records table is empty (Pitfall 4 shortcut)."""
    try:
        return store.db.open_table("records").count_rows() == 0
    except Exception:
        return True


def _is_inside_window(
    window: tuple[int, int] | list | None,
    now: datetime,
    tz,
) -> bool:
    """Return True when the current local time falls inside the learned quiet
    window. Handles wrap-around across local midnight (e.g. 22:00 -> 06:00)."""
    if not window:
        return False
    try:
        start, duration = int(window[0]), int(window[1])
    except (TypeError, ValueError, IndexError):
        return False
    if duration <= 0:
        return False
    now_local = now.astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // BUCKET_MINUTES
    end = (start + duration) % BUCKET_COUNT
    if start < end:
        return start <= cur_bucket < end
    # Wrap-around (e.g. start=44 (22:00), duration=16, end=(44+16)%48=12 (06:00))
    return cur_bucket >= start or cur_bucket < end


def _check_still_exclusive(lock: ProcessLock) -> bool:
    """Verify the daemon still holds the exclusive lock between REM cycles.


     HUMAN-FIRST: if an MCP client acquired a shared lock mid-night
    (e.g. user opened Claude Code between our REM cycles), the daemon
    must yield cooperatively BEFORE starting the next cycle.


    Delegates to `ProcessLock.holds_exclusive_nb` (Task 1). That
    method re-tries `fcntl.flock(LOCK_EX | LOCK_NB)` on our existing fd:
    - Still holding exclusive: re-acquire is a no-op success -> True.
    - MCP grabbed shared in between: EWOULDBLOCK -> False -> daemon yields.
    """
    return lock.holds_exclusive_nb()


# ---------------------------------------------------------------------------
# removed `_should_yield_to_mcp` /
# `MCP_RECENT_ACTIVITY_WINDOW_SEC`. The in-process C1 yield
# helper deferred REM cycles when MCP traffic was active or recent. Phase
# 10.6 supersedes this gate with the lifecycle state machine: when wrapper
# heartbeats are FRESH the daemon is in WAKE state and the sleep_pipeline
# is never run; SLEEP-state work is bounded-deferred via the lifecycle
# tick's `interrupt_check`. REM cycles in `_tick_body` therefore run
# without an explicit yield gate — they remain gated by the existing
# ProcessLock fcntl flock + Lance MVCC and trigger only inside the
# learned quiet window.
# ---------------------------------------------------------------------------


def _update_pending_digest(state: dict, cycle_result: dict) -> None:
    """Accumulate per-cycle outputs into the morning digest (, )."""
    digest = state.get("pending_digest") or {
        "rem_cycles_completed": 0,
        "episodes_processed": 0,
        "schemas_induced_tier0": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out_cycles": 0,
    }
    digest["rem_cycles_completed"] = int(digest.get("rem_cycles_completed", 0)) + 1
    digest["episodes_processed"] = int(digest.get("episodes_processed", 0)) + int(
        cycle_result.get("summaries_created", 0) or 0
    )
    digest["schemas_induced_tier0"] = int(digest.get("schemas_induced_tier0", 0)) + int(
        cycle_result.get("schema_candidates", 0) or 0
    )
    if cycle_result.get("claude_call_used"):
        digest["claude_call_used"] = True
        digest["main_insight_text"] = cycle_result.get("main_insight_text")
    if cycle_result.get("timed_out"):
        digest["timed_out_cycles"] = int(digest.get("timed_out_cycles", 0)) + 1
    state["pending_digest"] = digest


# ---------------------------------------------------------------------------
# Scheduler tick body
# ---------------------------------------------------------------------------

async def _tick_body(
    store: MemoryStore,
    lock: ProcessLock,
    state: dict,
    *,
    mcp_socket: SocketServer | None = None,
) -> None:
    """One scheduler tick. Runs every TICK_INTERVAL_SEC (30s).


    Decision tree:
    0.5 (b): drain first_turn_pending entries older
        than 1 h. Runs FIRST so stale entries get cleared regardless of any
        yield/pause downstream. Helper called with explicit `now=` kwarg so

        its behaviour is fully driven by this tick's clock. Emits
        `first_turn_pending_expired` event when entries are dropped .
    -1. REMOVED (was in-process C1 yield via
        `_should_yield_to_mcp(mcp_socket)`). Lifecycle state machine
        supersedes this gate: REM cycles only run inside the learned
        quiet window where MCP traffic is rare. ProcessLock + Lance
        MVCC remain the secondary guards. The `mcp_socket` kwarg is
        retained as accepted-and-ignored so existing tests keep working.
    0. scheduler_paused -> skip immediately (gap-fill).
    1. Empty store -> short-circuit (Pitfall 4).
    2. Re-learn quiet window if 24h elapsed .
    3. Determine if we are inside the learned window OR the 2h-idle bootstrap
       OR a user_sleep_request / force_rem_request is pending (gap-fill).
       Otherwise return without lock acquire.
    4. C1 gate: try_acquire_exclusive. If False -> emit `daemon_yielded`
       with reason=mcp_active, return.

    5. Transition WAKE -> TRANSITIONING -> SLEEP.
    6. Loop up to DEFAULT_CYCLE_COUNT REM cycles via `run_rem_cycle`. Between
       cycles, probe `_check_still_exclusive` AND `force_wake_request`. On
       either: emit `daemon_yielded` and break.
    7. Transition SLEEP -> WAKE, release lock, persist state.


    Exceptions inside the REM loop surface as `rem_cycle_error` events
    emitted by dream.run_rem_cycle itself; this function's try/finally
    guarantees the lock is released even on an unexpected raise.
    """
    # --- Step 0.5: per-tick prune ---------------
    # Drain stale first_turn_pending entries (older than 1 h) on every tick.
    # Runs BEFORE any yield/pause/empty-store gate so stale entries clear
    # even when the rest of the tick would skip. Pure-in-memory walk +
    # at most one save_state + at most one event emit, all wrapped in
    # try/except so a malformed state never blocks the tick.
    #

    # Explicit `now=datetime.now(timezone.utc)` kwarg threads this tick's
    # clock into the helper; the helper does NOT call datetime.now itself
    # along this path, which keeps the function pure and trivially testable
    # by passing a fixed `NOW` directly.
    try:
        from iai_mcp.daemon_state import (
            FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
            prune_first_turn_pending,
        )

        state, dropped = prune_first_turn_pending(
            state, now=datetime.now(timezone.utc),
        )
        if dropped:
            try:
                save_state(state)
            except Exception:
                pass
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "first_turn_pending_expired",
                    {
                        "dropped_count": len(dropped),
                        "session_ids": dropped,
                        "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                        "phase": "tick",
                    },
                    severity="info",
                )
            except Exception:
                pass
    except Exception:
        # Defense-in-depth: drain MUST NOT crash the tick. .1
        # established the discipline of swallowing exceptions in
        # auxiliary tick steps to preserve C1 cooperative scheduling.
        pass

    # --- Step -1: REMOVED in .6 ------------------------------------
    # The in-process C1 HUMAN-FIRST yield (was
    # `_should_yield_to_mcp(mcp_socket)`) is gone. The lifecycle state
    # machine + sleep_pipeline supersede it: REM cycles only run inside
    # the learned quiet window, where MCP traffic is rare; the daemon's
    # ProcessLock + Lance MVCC remain the secondary guards if traffic
    # arrives mid-cycle.

    # --- Step 0: scheduler_paused gate (gap-fill) ----------------------
    if state.get("scheduler_paused") is True:
        try:
            await asyncio.to_thread(
                write_event,
                store,
                "daemon_tick_skipped",
                {"reason": "paused"},
                severity="info",
            )
        except Exception:
            pass
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "paused"
        try:
            save_state(state)
        except Exception:
            pass
        return

    # --- Step 1: empty store shortcut ---------------------------------------
    if _store_is_empty(store):
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "empty_store"
        save_state(state)
        return

    now = datetime.now(timezone.utc)
    try:
        tz = load_user_tz()
    except Exception:
        # Config unreadable; fall back to UTC so we still run.
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("UTC")

    # --- Step 2: re-learn quiet window every 24h ----------------------------
    last_learned_raw = state.get("quiet_window_learned_at")
    last_learned_dt: datetime | None = None
    if last_learned_raw:
        try:
            last_learned_dt = datetime.fromisoformat(last_learned_raw)
        except (TypeError, ValueError):
            last_learned_dt = None
    if should_relearn(last_learned_dt, now):
        try:
            window = await asyncio.to_thread(learn_quiet_window, store, now, tz)
        except Exception:
            window = None
        state["quiet_window"] = list(window) if window else None
        state["quiet_window_learned_at"] = now.isoformat()
        save_state(state)

    # --- Step 3: decide whether to run at all -------------------------------
    # gap-fill: user_sleep_request or force_rem_request bypass the
    # quiet-window + idle-bootstrap checks. They are explicit user / operator
    # overrides and must run immediately when the daemon can take the lock.
    user_sleep_req = state.get("user_sleep_request") or {}
    force_rem_req = state.get("force_rem_request") or {}
    user_sleep_pending = bool(user_sleep_req.get("pending"))
    force_rem_pending = bool(force_rem_req.get("pending"))

    window = state.get("quiet_window")
    in_window = _is_inside_window(window, now, tz) if window else False

    if not in_window and not user_sleep_pending and not force_rem_pending:
        last_session_raw = state.get("last_session_ts") or state.get("last_session_started_at")
        last_session_dt: datetime | None = None
        if last_session_raw:
            try:
                last_session_dt = datetime.fromisoformat(last_session_raw)
            except (TypeError, ValueError):
                last_session_dt = None
        if not should_bootstrap_trigger(last_session_dt, now):
            state["last_tick_at"] = now.isoformat()
            state["last_tick_skipped_reason"] = "outside_window"
            save_state(state)
            return

    # --- Step 4: C1 gate -- exclusive lock acquisition ----------------------
    if not lock.try_acquire_exclusive():
        try:
            await asyncio.to_thread(
                write_event,
                store,
                "daemon_yielded",
                {"reason": "mcp_active"},
                severity="info",
            )
        except Exception:
            pass
        state["last_tick_at"] = now.isoformat()
        state["last_tick_skipped_reason"] = "mcp_active"
        save_state(state)
        return

    # --- Step 5-7: run REM cycles under the lock ----------------------------
    state.pop("last_tick_skipped_reason", None)

    # Clear user_sleep_request.pending the moment we commit to entering SLEEP:
    # the request has been honored and must not re-trigger on subsequent ticks
    # if the scheduler wakes and re-enters WAKE normally.
    if user_sleep_pending:
        req = state.get("user_sleep_request") or {}
        req["pending"] = False
        req["honored_at"] = now.isoformat()
        state["user_sleep_request"] = req
        save_state(state)

    try:
        transition(state, STATE_TRANSITIONING)
        transition(state, STATE_SLEEP)

        session_id = f"daemon-{now.isoformat()}"

        # gap-fill: force_rem_request runs ONE out-of-schedule REM cycle.
        # Clear the flag first so a raise inside run_rem_cycle doesn't loop.
        if force_rem_pending:
            req = state.get("force_rem_request") or {}
            req["pending"] = False
            req["honored_at"] = now.isoformat()
            state["force_rem_request"] = req
            save_state(state)
            total_cycles = 1
        else:
            total_cycles = int(state.get("rem_cycle_count") or DEFAULT_CYCLE_COUNT)
        claude_enabled = bool(state.get("claude_enabled", True))
        completed = 0

        for i in range(1, total_cycles + 1):
            transition(state, STATE_DREAMING)
            try:
                result = await run_rem_cycle(
                    store,
                    i,
                    total_cycles,
                    session_id,
                    is_last=(i == total_cycles),
                    claude_enabled=claude_enabled,
                )
            except Exception as exc:  # noqa: BLE001 -- dream already catches; double-guard
                try:
                    await asyncio.to_thread(
                        write_event,
                        store,
                        "rem_cycle_error",
                        {"cycle": i, "error": str(exc)[:500]},
                        severity="critical",
                    )
                except Exception:
                    pass
                result = {"cycle": i, "timed_out": False}

            _update_pending_digest(state, result)
            save_state(state)
            transition(state, STATE_SLEEP)
            completed = i

            # gap-fill: force_wake_request between cycles.
            force_wake_req = state.get("force_wake_request") or {}
            if force_wake_req.get("pending") is True:
                try:
                    await asyncio.to_thread(
                        write_event,
                        store,
                        "daemon_yielded",
                        {
                            "reason": "force_wake_requested",
                            "completed_cycles": completed,
                        },
                        severity="info",
                    )
                except Exception:
                    pass
                force_wake_req["pending"] = False
                force_wake_req["honored_at"] = datetime.now(timezone.utc).isoformat()
                state["force_wake_request"] = force_wake_req
                save_state(state)
                break

            # Between cycles: cooperative yield probe.
            if not _check_still_exclusive(lock):
                try:
                    await asyncio.to_thread(
                        write_event,
                        store,
                        "daemon_yielded",
                        {
                            "reason": "mcp_reacquired_mid_night",
                            "completed_cycles": completed,
                        },
                        severity="info",
                    )
                except Exception:
                    pass
                break

        # Write the recall payload once per REM-loop completion so the
        # SessionStart hook can read a file instead of dispatching a
        # JSON-RPC call into core (which would block on the exclusive
        # store lock during DREAMING). Best-effort; never raises.
        try:
            await asyncio.to_thread(_write_session_start_cache, store)
        except Exception:
            pass

        try:
            from iai_mcp.memory_bank import write_processed_salience_top_n

            await asyncio.to_thread(write_processed_salience_top_n, store)
        except Exception:
            pass

        transition(state, STATE_WAKE)

        # drain deferred-captures on every WAKE
        # transition. While the daemon was in SLEEP/DREAMING, Stop hooks
        # may have written --no-spawn deferral files to
        # ~/.iai-mcp/.deferred-captures/ (the daemon's MCP socket is open
        # but the heavy SLEEP work runs under the exclusive lock). This
        # second drain catches anything that piled up since startup-drain.
        # Runs inside the existing try/finally so the lock release happens
        # even if drain raises (defense-in-depth on top of drain's own
        # per-file try/except).
        try:
            from iai_mcp.capture import drain_deferred_captures

            wake_drain = await asyncio.to_thread(drain_deferred_captures, store)
            if wake_drain["files_drained"] or wake_drain["files_failed"]:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "deferred_drain_wake",
                    wake_drain,
                    severity="info",
                )
        except Exception as e:  # noqa: BLE001 -- drain MUST NOT crash tick
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "deferred_drain_failed",
                    {"error": str(e)[:200], "phase": "wake"},
                    severity="warning",
                )
            except Exception:
                pass

        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_completed_cycles"] = completed
        save_state(state)
    finally:
        try:
            lock.release()
        except Exception:
            pass


async def _scheduler_tick(
    store: MemoryStore,
    lock: ProcessLock,
    state: dict,
    *,
    tick_body: Callable[..., Awaitable[None]] | None = None,
    mcp_socket: SocketServer | None = None,
) -> None:
    """Run _tick_body every TICK_INTERVAL_SEC.


    An individual tick failure MUST NOT crash the daemon. We catch all
    exceptions, write a `tick_error` event (best-effort; even the event

    write is wrapped), and keep looping.


    LOCKED: when invoked from daemon.main, mcp_socket is
    threaded through to _tick_body so the in-process C1 HUMAN-FIRST yield
    can probe mcp_socket.last_activity_ts and active_connections between
    REM cycles. Legacy unit tests that pass a custom tick_body keep working
    — both built-in _tick_body and tick_body callables are invoked with
    keyword-only mcp_socket.
    """
    body = tick_body or _tick_body
    while True:
        try:
            await body(store, lock, state, mcp_socket=mcp_socket)
        except TypeError:
            # Legacy tick_body callables that pre-date may not accept
            # the keyword-only mcp_socket arg. Fall back to the 3-arg form so
            # existing tests keep passing without modification.
            try:
                await body(store, lock, state)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    write_event(
                        store,
                        "tick_error",
                        {"error": str(exc), "type": type(exc).__name__},
                        severity="warning",
                    )
                except Exception:
                    pass
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 -- daemon must never die mid-tick
            try:
                write_event(
                    store,
                    "tick_error",
                    {"error": str(exc), "type": type(exc).__name__},
                    severity="warning",
                )
            except Exception:
                pass
        try:
            await asyncio.sleep(TICK_INTERVAL_SEC)
        except asyncio.CancelledError:
            break


# ---------------------------------------------------------------------------
# S4 offline-pass loop (hourly viability scan, Warning 6)
# ---------------------------------------------------------------------------

async def _s4_offline_loop(store: MemoryStore, shutdown: asyncio.Event) -> None:
    """Hourly S4 viability scan -- contradictions, drift, stale goals, hit_rate.


    FSRS decay is applied by WALL-CLOCK elapsed time since last_reviewed (not
    per access count), so this loop only needs a wall-clock cadence; it does
    NOT iterate records or advance per-read counters. That keeps the loop
    cheap enough to run concurrent with other daemon work via LanceDB MVCC.


    W1 / , : a startup grace period delays the FIRST
    iteration so a freshly-spawned daemon does not immediately run the heavy
    S4 viability scan before draining deferred captures. Configured via
    S4_FIRST_ITER_GRACE_SEC (env IAI_MCP_S4_FIRST_ITER_GRACE_SEC). Cancellation
    semantics: if shutdown fires during the grace wait, the loop returns
    cleanly (no work performed, no exception).
    """
    if S4_FIRST_ITER_GRACE_SEC > 0:
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_FIRST_ITER_GRACE_SEC
            )
            # Shutdown fired during grace -- return without running S4.
            return
        except asyncio.TimeoutError:
            pass  # Grace elapsed; fall through to the regular loop.
    while not shutdown.is_set():
        try:
            await asyncio.to_thread(s4.run_offline_pass, store)
        except Exception as exc:  # noqa: BLE001 -- never die on offline-pass failure
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "s4_offline_pass_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:
                pass
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_OFFLINE_INTERVAL_SEC
            )
            break
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# HIPPEA activation cascade loop ( / D5-05)
# ---------------------------------------------------------------------------

# Poll cadence for the cascade loop. Short enough that a session_open event
# queued by the TS wrapper gets served within a few seconds; long enough
# that an idle loop doesn't spin the CPU.
HIPPEA_CASCADE_POLL_SEC: float = 5.0

# minimum interval between cascade body executions.
# Default 60s = 12x the 5s poll cadence; gates heavy work without dropping
# `pending` flags. Env override IAI_MCP_HIPPEA_MIN_INTERVAL_SEC.
HIPPEA_CASCADE_MIN_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPEA_MIN_INTERVAL_SEC", "60.0"),
)

# timestamp of the most recent cascade body
# completion (success or exception). Module-level mutable; the cascade
# loop declares `global _last_cascade_completed_at` to write. Ephemeral
# by design — daemon restart resets to 0.0 (subsequent pending=true
# triggers immediately on first poll, which is fine because the only
# time pending=true persists across restart is when the user opened a
# session, was disconnected, then the daemon rebooted).
_last_cascade_completed_at: float = 0.0


# ---------------------------------------------------------------------------
# : CPU watchdog (observation-only)
# ---------------------------------------------------------------------------
# Polls own-process CPU every WATCHDOG_POLL_SEC; emits `daemon_cpu_overload`
# (severity=critical) on sustained > WATCHDOG_THRESHOLD_PERCENT for 2
# consecutive samples (= WATCHDOG_POLL_SEC * 2 seconds sustained). The 71-
# minute blind period from 2026-04-27 (99-363% CPU, zero events) cannot
# recur. LOCKED: observation-only — no SIGTERM, no os.kill, no
# launchctl. Triage / repair is user-driven (Activity Monitor + launchctl
# unload -w). Auto-kill risks data loss + breaks C1 HUMAN-FIRST.
WATCHDOG_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_POLL_SEC", "30.0"),
)
WATCHDOG_THRESHOLD_PERCENT: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_THRESHOLD_PERCENT", "50.0"),
)
WATCHDOG_EVENT_COOLDOWN_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_EVENT_COOLDOWN_SEC", "300.0"),
)
WATCHDOG_SAMPLE_WINDOW: int = 4

# timestamp of the most recent overload event emit.
# Module-level mutable; `_cpu_watchdog_loop` declares `global` to write.
# Ephemeral — daemon restart resets to 0.0 so the first overload after
# restart can fire without waiting out a stale cooldown.
_last_overload_event_at: float = 0.0

# .2 R5: monotonic boot timestamp; populated in main after the
# daemon's wall-clock `daemon_started_at` stamp. Used by the watchdog to
# include `uptime_sec` in the overload payload. None until first stamped.
_daemon_started_monotonic: float | None = None


# ---------------------------------------------------------------------------
# REMOVED .8 RSS-watchdog
# restart-policy block (`_should_restart`, `_clean_shutdown_for_restart`,
# `_rss_watchdog_loop`, env vars `IAI_MCP_RSS_RESTART_THRESHOLD_MB`,
# `IAI_MCP_TTL_RESTART_HOURS`, `IAI_MCP_COLD_START_GRACE_SEC`). The
# lifecycle state machine + sleep_pipeline supersede this loop:
# Hibernation (process kill, RSS=0) is the new mechanism for unbounded
# RSS / long-uptime collapse. The plist's `KeepAlive={"Crashed": true}`
# (.6 plist update) ensures graceful exit 0 stays dead until
# wrapper kickstart, so periodic restart is no longer a concern.
# ---------------------------------------------------------------------------


async def _hippea_cascade_loop(store, shutdown: asyncio.Event) -> None:
    """5th daemon task. Polls `hippea_cascade_request` and
    pre-warms the HIPPEA LRU on pending.


    Constitutional invariants:
    - C1 HUMAN-FIRST: yields on shutdown within 5s (via asyncio.wait_for).
    - C3 ZERO API COST: no Anthropic SDK import; pure-local salience math.
    - C6 READ-ONLY: cascade is read-only against the store. The ONLY writes
      by this loop are (a) clearing the request flag in state and (b) emitting
      a `hippea_cascade_completed` diagnostic event. Neither mutates
      MemoryRecord rows.


    `retrieve.build_runtime_graph(store)` is now
    wrapped in `await asyncio.to_thread(...)` — previously the bare-sync
    call blocked the asyncio event loop for 8-13 s while it traversed
    NetworkX. Wrapping unblocks every other coroutine on the loop
    (socket_server.handle, _tick_body, _s4_offline_loop, audit_task).


    : cascade body is gated by a 60 s
    minimum-interval cooldown (`HIPPEA_CASCADE_MIN_INTERVAL_SEC`). When
    cooldown blocks, `pending=true` STAYS set (the cooldown gates work,
    does not consume requests). Next poll re-checks. Worst-case under
    perpetual `pending=true`: ≤ 1 cascade per 60 s.
    """
    # .2 R2 / Pitfall 3: explicit `global` so the assignment in the
    # finally block updates module-level state, not a local binding. Without
    # this declaration the cooldown is silently broken.
    global _last_cascade_completed_at

    # Local imports isolate cascade machinery from daemon boot-time cost.
    from iai_mcp import retrieve
    from iai_mcp.daemon_state import load_state, save_state
    from iai_mcp.hippea_cascade import run_cascade

    while not shutdown.is_set():
        try:
            state = load_state()
            req = state.get("hippea_cascade_request") or {}
            if req.get("pending"):
                # .2 R2 cooldown gate . If cascade body ran
                # within the last MIN_INTERVAL seconds, skip the body but
                # leave `pending=true` so the next eligible poll runs it.
                elapsed = time.monotonic() - _last_cascade_completed_at
                if elapsed < HIPPEA_CASCADE_MIN_INTERVAL_SEC:
                    # Cooldown gates execution; pending stays set until
                    # cascade actually runs. No event emit (would flood
                    # the ledger every 5 s).
                    pass
                else:
                    try:
                        assignment = None
                        try:
                            # wrap heavy sync call.
                            # Returns the 3-tuple (graph, assignment, rich_club)
                            # intact through to_thread.
                            _graph, assignment, _rc = await asyncio.to_thread(
                                retrieve.build_runtime_graph, store,
                            )
                        except Exception:
                            assignment = None
                        stats: dict = {
                            "communities_selected": 0, "records_warmed": 0,
                        }
                        if assignment is not None:
                            try:
                                # run_cascade is async-clean ;
                                # direct await is correct.
                                stats = await run_cascade(store, assignment)
                            except Exception:
                                stats = {
                                    "communities_selected": 0,
                                    "records_warmed": 0,
                                }
                        try:
                            await asyncio.to_thread(
                                write_event,
                                store,
                                "hippea_cascade_completed",
                                {
                                    "session_id": req.get("session_id", ""),
                                    **stats,
                                },
                                severity="info",
                            )
                        except Exception:
                            pass
                        # Clear the request flag so we don't re-run the same
                        # cascade. Pitfall 5 (daemon_state.save_state
                        # concurrency): the main tick loop may also write
                        # state concurrently; we re-read just before clearing
                        # to minimise lost-write windows.
                        try:
                            state = load_state()
                            state["hippea_cascade_request"] = {"pending": False}
                            save_state(state)
                        except Exception:
                            pass
                    finally:
                        # stamp end-of-cascade
                        # timestamp regardless of success/exception. Updates
                        # module-level state via the `global` declaration
                        # at the top of the function body.
                        _last_cascade_completed_at = time.monotonic()
        except Exception:
            # Any error in the outer body must not terminate the task
            # (C1: cooperative shutdown only).
            pass
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=HIPPEA_CASCADE_POLL_SEC,
            )
            # shutdown fired -> exit loop
            break
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# : CPU watchdog body (observation-only)
# ---------------------------------------------------------------------------

def _watchdog_active_task_names() -> list[str]:
    """Best-effort `active_tasks` payload.


    Returns up to 5 names of currently-running asyncio tasks (excluding
    done tasks). Falls back to '?' on empty get_name. Wrapped in
    try/except so an introspection failure never blocks the event emit.

    """
    out: list[str] = []
    try:
        for t in asyncio.all_tasks():
            if t.done():
                continue
            name = t.get_name() or "?"
            out.append(name)
    except Exception:  # noqa: BLE001 -- introspection failure non-fatal
        pass
    return out[:5]


async def _cpu_watchdog_loop(store, shutdown: asyncio.Event) -> None:
    """: observation-only CPU watchdog.


    Polls own-process CPU every WATCHDOG_POLL_SEC seconds via
    psutil.Process(os.getpid).cpu_percent(interval=None). When the
    last 2 samples both exceed WATCHDOG_THRESHOLD_PERCENT (default 50),
    emits `daemon_cpu_overload` event with severity=critical containing
    fsm_state, cpu_samples_pct, uptime_sec, active_tasks, threshold_pct,
    sustained_sec.


    Per-event cooldown WATCHDOG_EVENT_COOLDOWN_SEC (default 300s) prevents
    ledger flood under prolonged overload — at most one event per 5 min.


    : OBSERVATION-ONLY. No SIGTERM, no os.kill, no launchctl.
    The only side-effect is a write_event call. Triage / repair is
    user-driven (Activity Monitor, launchctl unload -w). Auto-kill
    risks data loss + breaks C1 HUMAN-FIRST. may add a soft-
    yield signal; 7.2 stays pure-observation.



    Pitfall 1 mitigation: prime the meter ONCE before the polling loop
    so the first real sample at t=POLL_SEC is a meaningful delta, not
    a 0.0 baseline-priming response.
    """
    # Pitfall 3: explicit `global` so cooldown timestamp updates module
    # state, not a local binding.
    global _last_overload_event_at

    # Local imports per RESEARCH Pitfall 5: keep daemon boot cheap.
    from collections import deque

    import psutil

    proc = psutil.Process(os.getpid())
    # Pitfall 1: prime psutil's internal CPU meter — first cpu_percent
    # call returns 0.0 (no prior measurement to delta against). Discard.
    try:
        proc.cpu_percent(interval=None)
    except Exception:  # noqa: BLE001 -- prime failure non-fatal
        pass

    samples: deque[float] = deque(maxlen=WATCHDOG_SAMPLE_WINDOW)

    while not shutdown.is_set():
        # Sleep for one poll interval (or break early on shutdown).
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=WATCHDOG_POLL_SEC,
            )
            break
        except asyncio.TimeoutError:
            pass

        # Sample own-process CPU (delta vs prior call).
        try:
            cpu_pct = proc.cpu_percent(interval=None)
            samples.append(cpu_pct)
        except Exception:  # noqa: BLE001 -- psutil flakiness must not crash
            continue

        # Trigger: 2 consecutive samples both > threshold (= sustained
        # WATCHDOG_POLL_SEC * 2 seconds).
        if (
            len(samples) >= 2
            and samples[-1] > WATCHDOG_THRESHOLD_PERCENT
            and samples[-2] > WATCHDOG_THRESHOLD_PERCENT
        ):
            now_mono = time.monotonic()
            # cooldown: at most 1 event per 5 min.
            if (now_mono - _last_overload_event_at) < WATCHDOG_EVENT_COOLDOWN_SEC:
                continue

            fsm_state = "?"
            try:
                state = load_state()
                fsm_state = state.get("fsm_state", "?")
            except Exception:  # noqa: BLE001 -- introspection only
                pass

            uptime_sec: float | None = None
            if _daemon_started_monotonic is not None:
                uptime_sec = round(now_mono - _daemon_started_monotonic, 1)

            payload = {
                "fsm_state": fsm_state,
                "cpu_samples_pct": list(samples),
                "uptime_sec": uptime_sec,
                "active_tasks": _watchdog_active_task_names(),
                "threshold_pct": WATCHDOG_THRESHOLD_PERCENT,
                "sustained_sec": int(WATCHDOG_POLL_SEC * 2),
            }

            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "daemon_cpu_overload",
                    payload,
                    severity="critical",
                )
            except Exception:  # noqa: BLE001 -- ledger emit failure non-fatal
                continue

            _last_overload_event_at = now_mono


# ---------------------------------------------------------------------------
# REMOVED the .8 RSS watchdog +
# clean-shutdown restart trigger block. `_resolve_shutdown_exit_code`
# (75/0 sentinel decision), `_clean_shutdown_for_restart` (os._exit(75)),
# `_rss_watchdog_loop` (RSS polling + TTL trigger) are all gone.
#

# The lifecycle state machine + sleep_pipeline supersede this design.
# Hibernation kills the process with exit 0 (graceful) and the plist's
# `KeepAlive={"Crashed": true}` ensures launchd does NOT auto-respawn
# on graceful exit; the wrapper kickstart is the wake mechanism.
#

# The user-stop sentinel from 541c874 is PRESERVED but simplified.
# `iai-mcp daemon stop` still writes `user_requested_shutdown=True`
# to `.daemon-state.json` before SIGTERM; the daemon's main finally
# block clears the sentinel from the on-disk file (so a stale flag
# cannot leak across boots) but the exit code is now uniformly 0
# regardless of who triggered the shutdown.
# ---------------------------------------------------------------------------

# Sentinel key in .daemon-state.json. Preserved from 541c874. Phase
# 10.6 simplifies the read semantics: the daemon's main finally
# block clears the on-disk flag so it does not leak across boots; the
# exit code no longer branches on it (always 0).
_USER_SHUTDOWN_FLAG = "user_requested_shutdown"


def _clear_user_shutdown_sentinel(state: dict) -> None:
    """Clear the on-disk + in-memory ``user_requested_shutdown`` flag.


    Cross-process invariant (preserved from 541c874): the CLI
    ``iai-mcp daemon stop`` runs in a SEPARATE process from the daemon
    and writes the sentinel to ``.daemon-state.json`` BEFORE sending
    SIGTERM. The daemon's in-memory ``state`` dict was loaded at boot
    time and is never re-read on signal — so the disk-side flag must
    be cleared explicitly here, not just popped from the memory dict.


    change: the function ONLY clears the sentinel; it does
    NOT decide an exit code. main always returns 0 on graceful
    shutdown, regardless of who triggered it. launchd's
    ``KeepAlive={"Crashed": true}`` plist ensures graceful exit 0
    stays dead until wrapper kickstart fires.


    Read failure is fail-safe: ignored. The next ``save_state`` from
    main will overwrite the on-disk record anyway.
    """
    try:
        on_disk = load_state()
        if _USER_SHUTDOWN_FLAG in on_disk:
            on_disk.pop(_USER_SHUTDOWN_FLAG, None)
            save_state(on_disk)
    except Exception:
        # Disk read/write failure must NOT block shutdown.
        pass
    state.pop(_USER_SHUTDOWN_FLAG, None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    """Open store + lock, prewarm embedder, serve socket, tick forever.


    Returns 0 on clean shutdown (signal-driven OR Hibernation transition);
    returns 1 only on LifecycleLockConflict (a same-host live-PID conflict);

    raises SystemExit(2) on partial-migration boot block. Signals

    SIGTERM/SIGINT/SIGHUP all set the shutdown event.


    Tasks spawned (post-Phase-10.6):
    - mcp_socket_task: SocketServer.serve — SOLE binder of
                             ~/.iai-mcp/.daemon.sock.
    - tick_task: scheduler tick loop (_scheduler_tick + _tick_body)
                             for legacy REM cycles. The _should_yield_to_mcp gate inside
                             _tick_body has been removed; the lifecycle
                             state machine supersedes the in-process yield.
    - audit_task: continuous_audit (C6, MVCC reads).
    - s4_task: hourly S4 offline pass.
    - cascade_task: HIPPEA activation-cascade
                             pre-warmer.
    - cpu_watchdog_task: observation-only CPU watchdog.
    - lifecycle_tick_task: drives the
                             WAKE/DROWSY/SLEEP/HIBERNATION state machine
                             every 30 s; runs sleep_pipeline on SLEEP
                             entry; sets the global shutdown event on
                             HIBERNATION (with shadow_run=False).


    Removed in Task 1.4:
    - idle_propagator_task (was the bridge from socket idle_watcher to
                             the global shutdown event; idle_watcher itself
                             gone).
    - rss_watchdog_task (RSS-watchdog; Hibernation now
                             provides "kill the process to drop RSS").
    """
    # F-05: the daemon is a long-lived reader while MCP tool calls write
    # to the same LanceDB directory from short-lived processes. Without
    # an explicit consistency interval the daemon's connection pins the
    # manifest snapshot it read at startup and every tick's
    # ``_store_is_empty`` check keeps returning True even after writers
    # have populated the store. ``timedelta(seconds=0)`` gives strong
    # consistency — each read re-checks the latest committed version at
    # negligible cost (one manifest stat per query) and restores the
    # tick body's ability to see work.
    store = MemoryStore(read_consistency_interval=timedelta(seconds=0))

    try:
        from iai_mcp.crypto_key_watch import check_crypto_key_file_rotation_event

        check_crypto_key_file_rotation_event(store)
    except Exception:
        pass

    # boot-time partial-migration detector. Closes the
    # V2-07 anti-pattern of declared-but-unwired knobs — the rollback handler
    # in migrate.py only fires if it's actually called from the boot path.
    # Placed BEFORE the embedder prewarm so a partial-state boot short-
    # circuits before paying the ~10s model-load cost.
    #

    # State machine (see migrate.detect_partial_migration):
    # - clean / unknown -> proceed to ready advertisement.
    # - needs_cleanup -> drop records_old_<ts>, then proceed.
    # - needs_rollback -> STOP daemon; surface remediation prompt.
    # - partial_swap_inconsistent -> STOP daemon; surface remediation prompt
    # (manual recovery; no rollback anchor).
    from iai_mcp.migrate import detect_partial_migration
    _migration_state = detect_partial_migration(store.db)
    if _migration_state["state"] == "partial_swap_inconsistent":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback to restore from "
                        "records_old_<ts>, then iai-mcp daemon-start."
                    ),
                }) + "\n"
            )
        except Exception:
            pass
        raise SystemExit(2)
    if _migration_state["state"] == "needs_rollback":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback (discard the partial "
                        "staging) OR iai-mcp migrate --resume (continue "
                        "from migration_progress.json checkpoint)."
                    ),
                }) + "\n"
            )
        except Exception:
            pass
        raise SystemExit(2)
    if _migration_state["state"] == "needs_cleanup":
        # Successful swap from a previous boot; drop the old table now.
        for _old_name in _migration_state.get("old_tables", []):
            try:
                store.db.drop_table(_old_name)
            except Exception as _exc:
                try:
                    sys.stderr.write(
                        json.dumps({
                            "event": "migrate_cleanup_failed",
                            "table": _old_name,
                            "err": str(_exc)[:120],
                        }) + "\n"
                    )
                except Exception:
                    pass

    # Pitfall 3 prewarm: avoid 10s cold-start in the first REM cycle by
    # loading the embedder model into RAM at boot. The warmup text is
    # trivial; we only care about model-load side-effect.
    try:
        from iai_mcp.embed import embedder_for_store
        embedder_for_store(store).embed("warmup")
    except Exception as exc:  # noqa: BLE001 -- prewarm failure is non-fatal
        try:
            write_event(store, "prewarm_failed", {"error": str(exc)}, severity="warning")
        except Exception:
            pass

    # Detect drift between the canonical and legacy state files. Detect-only:
    # the canonical lifecycle_state.json is the source of truth. A mismatch
    # is surfaced via a warning-severity event and a stdlib log line so an
    # operator can audit; the daemon does not auto-correct.
    try:
        from iai_mcp.fsm_reconcile import reconcile_fsm_state

        _drift_report = reconcile_fsm_state()
        if _drift_report.get("drift") is True:
            log.warning(
                "fsm_drift_detected canonical=%s legacy=%s",
                _drift_report.get("canonical"),
                _drift_report.get("legacy"),
            )
            try:
                write_event(
                    store,
                    "fsm_drift_detected",
                    _drift_report,
                    severity="warning",
                    domain="ops",
                )
            except Exception:  # noqa: BLE001 -- fail-safe
                pass
    except Exception:  # noqa: BLE001 -- fail-safe boundary
        pass

    # Archive any HIBERNATION-stuck.bak recovery artifacts left by prior
    # daemon lives. Pure disk hygiene; fail-safe.
    try:
        from iai_mcp.archive_backups import archive_stuck_backups

        archive_stuck_backups()
    except Exception:  # noqa: BLE001 -- fail-safe boundary
        pass

    lock = ProcessLock()

    # acquire the WAKE-13 single-machine
    # lockfile (~/.iai-mcp/.locked). This is DISTINCT from `lock`
    # (ProcessLock fcntl flock that guards LanceDB writers); the
    # lifecycle lock is a higher-level, human-readable singleton marker
    # for the lifecycle state machine. A live-PID conflict on the same
    # host raises LifecycleLockConflict and we exit 1; dead-PID or
    # foreign-host scenarios are silently overwritten.
    from iai_mcp.lifecycle_lock import LifecycleLock, LifecycleLockConflict

    lifecycle_lock = LifecycleLock()
    try:
        lifecycle_lock.acquire()
    except LifecycleLockConflict as exc:
        sys.stderr.write(f"daemon already running: {exc}\n")
        return 1

    state = load_state()
    state.setdefault("fsm_state", STATE_WAKE)
    state["daemon_started_at"] = datetime.now(timezone.utc).isoformat()
    # .2 R5: stamp monotonic boot time so CPU watchdog payload
    # can include uptime_sec. Module-level global; written here only.
    global _daemon_started_monotonic
    _daemon_started_monotonic = time.monotonic()
    # (a) revised: stamp daemon_pid into the state file so
    # `iai-mcp doctor` check (a) can read the live PID. The fcntl `.lock`
    # file holds zero PID bytes, so a separate source of truth is required.
    # On graceful shutdown the finally block clears this key (see below).
    state["daemon_pid"] = os.getpid()
    save_state(state)
    write_event(store, "daemon_started", {"state": state["fsm_state"]})

    # .5 L5: consume any pending wake.signal written by the MCP
    # wrapper while the daemon was down. .6
    # Task 1.5 wires the result into the lifecycle state machine: a
    # consumed wake_signal dispatches WAKE_SIGNAL to the LSM (which
    # transitions HIBERNATION -> WAKE if needed; no-op on cold boot
    # where current_state is already WAKE).
    _wake_was_pending = False
    try:
        from pathlib import Path as _Path

        from iai_mcp.wake_handler import WakeHandler

        _wake_signal_path = _Path("~/.iai-mcp/wake.signal").expanduser()
        if WakeHandler(_wake_signal_path).consume_wake_signal():
            _wake_was_pending = True
            write_event(
                store, "wake_signal_consumed", {"phase": "startup"}, severity="info"
            )
    except Exception:
        # Defensive: never block daemon boot on a wake-handler error.
        pass

    # drain any capture-queue records
    # buffered by the wrapper while the daemon was hibernated. The
    # queue is the durable WRITE-side buffer that makes Hibernation
    # viable . Records are routed back through the
    # existing capture path so the verbatim contract is
    # preserved end-to-end.
    try:
        from iai_mcp.capture import capture_turn as _capture_turn
        from iai_mcp.capture_queue import CaptureQueue

        _capture_queue = CaptureQueue()
        # Bind store via closure; map the queue's record envelope to
        # capture_turn's keyword-only signature (cue, text, tier,
        # session_id, role). The queue's records originate from the
        # wrapper's memory_capture path which already populates these
        # fields verbatim.
        def _capture_handler(record: dict) -> None:
            kwargs = {
                "cue": record.get("cue", ""),
                "text": record.get("text", record.get("surface", "")),
                "tier": record.get("tier", "episodic"),
                "session_id": record.get("session_id", "-"),
                "role": record.get("role", "user"),
            }
            _capture_turn(store, **kwargs)

        ingested = await asyncio.to_thread(
            _capture_queue.ingest_pending, _capture_handler,
        )
        if ingested > 0:
            write_event(
                store,
                "capture_queue_drained",
                {"phase": "startup", "ingested": ingested},
                severity="info",
            )
    except Exception as exc:  # noqa: BLE001 -- never block boot on queue drain
        try:
            write_event(
                store,
                "capture_queue_drain_failed",
                {"phase": "startup", "error": str(exc)[:200]},
                severity="warning",
            )
        except Exception:
            pass

    # startup-prune: drain any first_turn_pending
    # entries that are older than FIRST_TURN_PENDING_TTL_SEC_DEFAULT (1h).
    # The user's machine on 2026-04-27 had 11 stale entries (oldest 16h+)
    # before launchctl unload — each one perpetually retriggered the HIPPEA
    # cascade. Pruning at boot resets the slate; the per-tick prune (in
    # _tick_body Step 0.5) keeps it clean during long-running daemons.
    #

    # We pass an explicit `now=` kwarg (rather than letting the helper
    # default to `datetime.now(timezone.utc)`) so the helper's behaviour
    # is fully deterministic from the caller's perspective. Tests of the
    # wire-in can supply a fixed `NOW` and assert the helper output
    # directly without datetime monkeypatching.
    try:
        from iai_mcp.daemon_state import (
            FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
            prune_first_turn_pending,
        )

        state, dropped = prune_first_turn_pending(
            state, now=datetime.now(timezone.utc),
        )
        if dropped:
            save_state(state)
            try:
                write_event(
                    store,
                    "first_turn_pending_expired",
                    {
                        "dropped_count": len(dropped),
                        "session_ids": dropped,
                        "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                        "phase": "startup",
                    },
                    severity="info",
                )
            except Exception:
                pass
    except Exception:
        # Drain failure must never block daemon startup.
        # established this exception-isolation discipline for startup-side
        # work.
        pass

    # drain any deferred-captures JSONL files that
    # piled up in ~/.iai-mcp/.deferred-captures/ while we were down. Stop-hook
    # invocations of `iai-mcp capture-transcript --no-spawn` defer to disk
    # when the daemon socket is unreachable; this is the daemon-side reader
    # that ingests them on next boot. Runs ONCE at startup; the WAKE-from-
    # SLEEP transition inside _tick_body re-runs drain to catch files
    # written while the daemon was asleep but not yet exited.
    #

    # Wrapped in try/except that NEVER propagates: a malformed deferred
    # file or a bug in capture_turn must not block daemon startup. Per-
    # file errors are isolated inside drain_deferred_captures (renames the
    # offender to .failed-<ts>.jsonl).
    try:
        from iai_mcp.capture import drain_deferred_captures

        drain_counts = await asyncio.to_thread(drain_deferred_captures, store)
        if drain_counts["files_drained"] or drain_counts["files_failed"]:
            write_event(
                store,
                "deferred_drain_startup",
                drain_counts,
                severity="info",
            )
    except Exception as e:  # noqa: BLE001 -- drain MUST NOT crash daemon
        try:
            write_event(
                store,
                "deferred_drain_failed",
                {"error": str(e)[:200], "phase": "startup"},
                severity="warning",
            )
        except Exception:
            pass

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread: no signal handlers.
            pass

    # one-shot Lance storage optimize at startup,
    # BEFORE the SocketServer binds and any tasks are created. Rationale:
    # (a) collapses any pre-existing version bloat before the first task
    # touches records.lance (the smoking-gun forensic case 2026-04-27 was
    # 10,841 versions / 3.66 GB on records.lance accumulated over 9 days);
    # (b) by definition no MCP client has connected yet so the 33-second
    # I/O cannot interfere with any user-facing work; (c) the helper itself
    # never raises and the wrapping try/except is belt-and-braces
    # so a corrupt LanceDB cannot block daemon boot.
    #

    # REMOVED the
    # IAI_MCP_SKIP_STARTUP_OPTIMIZE env override path. Sleep_pipeline
    # step 4 (OPTIMIZE_LANCE) and step 5 (COMPACT_RECORDS) handle
    # version-bloat collapse during the SLEEP state, so the synchronous
    # boot-time call no longer needs an opt-out for cold-start latency.
    try:
        startup_t0 = time.monotonic()
        startup_report = await asyncio.to_thread(optimize_lance_storage, store)
        await asyncio.to_thread(
            write_event,
            store,
            "lance_storage_optimized",
            {
                "phase": "startup",
                "retention_days": (
                    _maintenance.LANCE_OPTIMIZE_RETENTION_SEC / 86400.0
                ),
                "per_table": startup_report,
                "total_elapsed_sec": round(time.monotonic() - startup_t0, 3),
            },
            severity="info",
        )
    except Exception:
        # maintenance MUST NOT crash daemon boot.
        pass

    # (lines 83-85, LOCKED): SocketServer is the SINGLE
    # binder of ~/.iai-mcp/.daemon.sock. The pre-Phase-7 concurrency.serve_control_socket
    # has been REMOVED from this gather block — both servers calling
    # asyncio.start_unix_server on the same SOCKET_PATH would EADDRINUSE on the
    # second bind and the daemon would fail to start. Backward compat for the 7
    # control messages is preserved inside SocketServer.handle's
    # dispatcher fork (jsonrpc=='2.0' → core.dispatch; 'type' in
    # CONTROL_MSG_TYPES → forward to concurrency._dispatch_socket_request).
    # concurrency.serve_control_socket function STAYS defined in concurrency.py
    # for test-compat per D7-17 final paragraph; scheduled for cleanup
    # once the 1226-test suite is migrated.
    #

    # , R1: full MCP-method routing over unix socket.
    # idle_secs defaults to env IAI_DAEMON_IDLE_SHUTDOWN_SECS or 1800 (D7-05).
    mcp_socket = SocketServer(store, lock=lock, state=state)
    mcp_socket_task = asyncio.create_task(mcp_socket.serve())

    # REMOVED `_propagate_idle_shutdown`
    # bridge task. The socket-side `idle_watcher` (which set
    # mcp_socket.shutdown_event after IDLE_CHECK_INTERVAL_SECS of
    # inactivity) has also been removed in this phase. The lifecycle
    # state machine (Task 1.5) takes over the "idle daemon -> shut
    # down" responsibility via the heartbeat scanner + idle detector
    # + Hibernation transition.

    # initialise the lifecycle state
    # machine + heartbeat scanner + idle detector + sleep pipeline.
    # All four are stdlib-only / no new deps. The state machine reads
    # / writes ~/.iai-mcp/lifecycle_state.json via fcntl flock. Task
    # 1.6 flips the LSM default to shadow_run=False so HIBERNATION
    # transitions actually exit the daemon process.
    from iai_mcp.heartbeat_scanner import HeartbeatScanner as _HeartbeatScanner
    from iai_mcp.idle_detector import IdleDetector as _IdleDetector
    from iai_mcp.lifecycle import (
        LifecycleEvent as _LifecycleEvent,
    )
    from iai_mcp.lifecycle import (
        LifecycleStateMachine as _LifecycleStateMachine,
    )
    from iai_mcp.lifecycle_state import LifecycleState as _LifecycleState
    from iai_mcp.sleep_pipeline import SleepPipeline as _SleepPipeline

    # Honor IAI_MCP_STORE for the wrappers dir resolution (test isolation
    # + multi-tenant deployments). Falls back to ~/.iai-mcp/wrappers in
    # production where the env var is unset.
    from pathlib import Path as _PathHere
    _store_root = os.environ.get("IAI_MCP_STORE")
    _wrappers_dir = (
        _PathHere(_store_root) if _store_root else _PathHere.home() / ".iai-mcp"
    ) / "wrappers"
    _heartbeat_scanner = _HeartbeatScanner(_wrappers_dir)
    _idle_detector = _IdleDetector()
    _sleep_pipeline = _SleepPipeline(store=store)
    # The state machine constructor reads its shadow_run default from
    # the class signature (flipped to False in Task 1.6). Tests can
    # override by passing an explicit kwarg.
    _state_machine = _LifecycleStateMachine()

    # If the wrapper kicked us via wake.signal AND our last persisted
    # state was HIBERNATION, dispatch WAKE_SIGNAL so the LSM
    # transitions back to WAKE atomically with the kickstart.
    if _wake_was_pending:
        try:
            _state_machine.dispatch(_LifecycleEvent.WAKE_SIGNAL)
        except Exception:
            pass

    tick_task = asyncio.create_task(
        _scheduler_tick(store, lock, state, mcp_socket=mcp_socket)
    )
    audit_task = asyncio.create_task(
        # dropped the `socket=` kwarg
        # — `_should_yield_to_mcp` is gone. `continuous_audit`'s
        # periodic Lance optimize body now runs unconditionally once
        # the cooldown gate passes; SLEEP-state coexistence is
        # provided by the lifecycle state machine instead.
        continuous_audit(store, shutdown)
    )
    s4_task = asyncio.create_task(
        _s4_offline_loop(store, shutdown)
    )
    # HIPPEA activation-cascade loop.
    cascade_task = asyncio.create_task(
        _hippea_cascade_loop(store, shutdown)
    )

    # CPU watchdog (observation-only).
    cpu_watchdog_task = asyncio.create_task(
        _cpu_watchdog_loop(store, shutdown)
    )
    # REMOVED the rss_watchdog_task.
    # `_rss_watchdog_loop` / `_clean_shutdown_for_restart` /
    # `_should_restart` were the legacy mechanism for unbounded RSS;
    # the lifecycle state machine's Hibernation transition now
    # provides the same "kill the process to drop RSS" behaviour as a
    # natural consequence of the WAKE -> DROWSY -> SLEEP -> HIBERNATION
    # progression.

    # lifecycle TICK loop.
    # Cadence: 30 seconds (no busy loops; idle CPU near zero).
    # Responsibilities per CONTEXT 10.6:
    # 1. Poll heartbeat scanner + idle detector.
    # 2. Dispatch HEARTBEAT_REFRESH / IDLE_5MIN / IDLE_30MIN events
    # to the state machine based on observed activity.
    # 3. When state == SLEEP, run sleep_pipeline.run with an
    # `interrupt_check` lambda that reads MCP socket activity.
    # On natural completion, dispatch SLEEP_CYCLE_DONE so the
    # state machine transitions to HIBERNATION.
    # 4. When state == HIBERNATION (with shadow_run=False), set
    # the global shutdown event so main exits gracefully.

    LIFECYCLE_TICK_INTERVAL_SEC: float = 30.0
    DROWSY_AFTER_SEC: float = float(
        os.environ.get("LIFECYCLE_DROWSY_AFTER_SEC", "300")
    )  # 5 min
    HIBERNATE_AFTER_SEC: float = float(
        os.environ.get("LIFECYCLE_HIBERNATE_AFTER_SEC", "7200")
    )  # 2 h (state machine HIBERNATION_GRACE_EXPIRED future-phase)
    SLEEP_HEARTBEAT_IDLE_SEC: float = float(
        os.environ.get("LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC", "1800")
    )  # 30 min — for IDLE_30MIN dispatch threshold
    # Window inside which an MCP touch / open connection means the
    # daemon should defer the next sleep_pipeline chunk (interrupt).
    INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC: float = 30.0

    # Track when WAKE last had heartbeat activity; the lifecycle
    # state machine's last_activity_ts in lifecycle_state.json is
    # the persistent-side record, but we also keep a monotonic
    # baseline here for the IDLE_5MIN / IDLE_30MIN thresholds.
    _last_active_monotonic: list[float] = [time.monotonic()]
    # Previous-tick lifecycle state for WAKE -> DROWSY edge detection.
    _prev_lifecycle_state: list = [_LifecycleState.WAKE]

    async def lifecycle_tick() -> None:
        """Periodic lifecycle event dispatcher.


        Called every LIFECYCLE_TICK_INTERVAL_SEC seconds (30 s).
        Cancellation-safe via asyncio.wait_for(shutdown.wait, ...).
        """
        while not shutdown.is_set():
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=LIFECYCLE_TICK_INTERVAL_SEC,
                )
                return  # shutdown fired
            except asyncio.TimeoutError:
                pass

            try:
                # 1. Probe heartbeat scanner + idle detector.
                scanner_active = await asyncio.to_thread(
                    _heartbeat_scanner.is_active,
                )
                heartbeat_idle = await asyncio.to_thread(
                    _heartbeat_scanner.heartbeat_idle_30min,
                )
                sleep_eligible = await asyncio.to_thread(
                    _idle_detector.sleep_eligible, heartbeat_idle,
                )

                now_mono = time.monotonic()
                idle_elapsed = now_mono - _last_active_monotonic[0]

                if scanner_active:
                    # Wrapper is alive — refresh activity baseline
                    # and dispatch HEARTBEAT_REFRESH (DROWSY -> WAKE).
                    _last_active_monotonic[0] = now_mono
                    _state_machine.dispatch(
                        _LifecycleEvent.HEARTBEAT_REFRESH,
                    )
                elif idle_elapsed >= SLEEP_HEARTBEAT_IDLE_SEC and sleep_eligible:
                    # 30 min idle + hardware confirmation → request
                    # SLEEP transition. Payload guard satisfies the
                    # transition-table requirement.
                    _state_machine.dispatch(
                        _LifecycleEvent.IDLE_30MIN,
                        sleep_eligible=True,
                    )
                elif idle_elapsed >= DROWSY_AFTER_SEC:
                    # 5 min idle → DROWSY (no-op if already there).
                    _state_machine.dispatch(_LifecycleEvent.IDLE_5MIN)

                # 2. If state is now SLEEP, run the sleep pipeline
                # with bounded deferral.
                current = _state_machine.current_state
                # WAKE -> DROWSY edge: drain deferred captures once per
                # entry. Guarded by _prev_lifecycle_state so consecutive
                # DROWSY ticks do not re-fire.
                if _should_drain_on_drowsy_edge(_prev_lifecycle_state[0], current):
                    try:
                        from iai_mcp.capture import drain_deferred_captures

                        await asyncio.to_thread(
                            _run_drowsy_drain,
                            store,
                            drain_fn=drain_deferred_captures,
                            write_event_fn=write_event,
                        )
                    except Exception:
                        pass
                _prev_lifecycle_state[0] = current
                if current is _LifecycleState.SLEEP:
                    def _interrupt_check() -> bool:
                        # Bounded deferral: fire the interrupt if
                        # MCP traffic is active or recent.
                        if mcp_socket.active_connections > 0:
                            return True
                        elapsed = (
                            time.monotonic() - mcp_socket.last_activity_ts
                        )
                        return elapsed < INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC

                    result = await asyncio.to_thread(
                        _sleep_pipeline.run, _interrupt_check,
                    )
                    if (
                        not result.get("interrupted", False)
                        and result.get("failed_step") is None
                        and not result.get("quarantine_triggered", False)
                        and len(result.get("completed_steps", [])) >= 5
                    ):
                        # Natural completion of all 5 steps → maybe
                        # transition to HIBERNATION.
                        # `still_idle` payload guard: re-check idle
                        # AFTER the pipeline ran (it may have run
                        # for several seconds; user activity may
                        # have arrived in between).
                        still_idle_now = await asyncio.to_thread(
                            _heartbeat_scanner.heartbeat_idle_30min,
                        )
                        sleep_eligible_now = await asyncio.to_thread(
                            _idle_detector.sleep_eligible, still_idle_now,
                        )
                        _state_machine.dispatch(
                            _LifecycleEvent.SLEEP_CYCLE_DONE,
                            still_idle=(still_idle_now and sleep_eligible_now),
                        )

                # 3. If state is HIBERNATION and shadow_run=False,
                # set the global shutdown event. main's finally
                # block will release the lifecycle lock and exit 0.
                current = _state_machine.current_state
                if (
                    current is _LifecycleState.HIBERNATION
                    and not _state_machine.shadow_run
                ):
                    try:
                        write_event(
                            store,
                            "lifecycle_hibernation_exit",
                            {
                                "reason": "lifecycle_tick_hibernation",
                                "shadow_run": False,
                            },
                            severity="info",
                        )
                    except Exception:
                        pass
                    shutdown.set()
                    return
            except Exception:  # noqa: BLE001 -- lifecycle tick must NEVER crash
                # Defensive: any error in the lifecycle tick should
                # not bring down the daemon. The next tick gets a
                # fresh chance.
                pass

    lifecycle_tick_task = asyncio.create_task(lifecycle_tick())

    try:
        await shutdown.wait()
    finally:
        # simplified shutdown set.
        # `idle_propagator_task` and `rss_watchdog_task` are gone; the
        # remaining 6 tasks (mcp_socket + tick + audit + s4 + cascade
        # + cpu_watchdog) form the cancel set. Trigger SocketServer's
        # graceful drain explicitly so connections close before the
        # asyncio.Server is torn down by task cancellation.
        try:
            mcp_socket.shutdown_event.set()
        except Exception:
            pass
        _cancel_targets = [
            tick_task, audit_task, s4_task, cascade_task,
            mcp_socket_task,
            cpu_watchdog_task,
            lifecycle_tick_task,
        ]
        for t in _cancel_targets:
            t.cancel()
        # Drain task exceptions silently: we're shutting down.
        await asyncio.gather(*_cancel_targets, return_exceptions=True)
        try:
            write_event(store, "daemon_stopped", {"state": state.get("fsm_state")})
        except Exception:
            pass
        # Persist final state so next boot sees a clean shutdown marker.
        # clear the on-disk
        # user_requested_shutdown sentinel so it does not leak across
        # boots. Exit code is uniformly 0 — the plist's KeepAlive=
        # {"Crashed": true} ensures graceful 0 stays dead until wrapper
        # kickstart.
        _clear_user_shutdown_sentinel(state)
        try:
            state.pop("daemon_pid", None)
            state["daemon_stopped_at"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
        except Exception:
            pass
        # release the lifecycle lockfile
        # so the next daemon boot can acquire cleanly. release is
        # idempotent.
        try:
            lifecycle_lock.release()
        except Exception:
            pass
        # Clean uninstall invariant (C4): release + close the fcntl fd.
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
