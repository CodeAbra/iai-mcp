"""iai-mcp CLI: health + migrate + trajectory + audit + crypto + daemon.

Plan 02-05 gave us `audit`. added crypto. Plan 04-05
adds the `daemon` subcommand group.

Commands:
- `iai-mcp health`           -- print the most recent llm_health event in user-local TZ
- `iai-mcp migrate`          -- Phase 1->2 migration OR v2->v3
                                encryption migration (chosen by --from / --to)
- `iai-mcp trajectory`       -- aggregate M1..M6 trajectory events
- `iai-mcp audit`            -- (Plan 02-05 OPS-07) identity + shield audit log
- `iai-mcp crypto status`           -- (Plan 02-08, rewritten 07.10) file-backend key status
- `iai-mcp crypto rotate`           -- rotate AES-256-GCM key
- `iai-mcp crypto migrate-to-file`  -- (Plan 07.10) one-time migration from Keychain to file
- `iai-mcp crypto init`             -- (Plan 07.10) fresh-install: generate a new key file
- `iai-mcp crypto recover-with-prior-key` -- re-encrypt records after wrong-key rotation (32-byte prior key file)
- `iai-mcp crypto redact-undecryptable` -- replace surfaces that fail decrypt with a redacted marker
- `iai-mcp daemon install`   -- (Plan 04-05 DAEMON-10) silent install + consent
- `iai-mcp daemon uninstall` -- C4 clean uninstall (plist/unit + 3 state files)
- `iai-mcp daemon start|stop|status|logs|force-rem|pause|resume|configure`

All timestamps render in the user's IANA timezone via
`iai_mcp.tz.load_user_tz() + to_local()`. Storage remains UTC.

OPS-07 audit privacy: shield match patterns are REDACTED to the MATCH COUNT
in CLI output (T-02-05-02 info-disclosure mitigation). Full payload remains
in the events table for forensics.

Constitutional guards (Plan 04-05 daemon group):
- C3 / ZERO API costs. The paid-API env-var token is forbidden in
  daemon-side modules; this CLI delegates LLM-aware operations to the
  daemon process which uses `claude -p` subprocess (subscription only).
- C4: `daemon uninstall` MUST remove plist/unit AND ~/.iai-mcp/.lock,
  ~/.iai-mcp/.daemon.sock, ~/.iai-mcp/.daemon-state.json -- verified by
  tests/shell/test_launchd_install.sh and tests/test_cli_daemon.py.
- Pitfall 5 (launchd PATH): install renders the plist with absolute
  `sys.executable` substituted -- launchd has no PATH, relative `python3`
  would resolve to /usr/bin/python3 even if user installed in /opt/python.
- Pitfall 8 (systemd linger): install probes `loginctl show-user --property=Linger`
  on Linux; if Linger=no, runs `loginctl enable-linger $USER` and re-verifies.
  PAM-variant systems may silently refuse, hence the post-enable check + WARN.
- Subprocess invocation: argv-list form ALWAYS, never shell=True. launchctl /
  systemctl / loginctl / tail / journalctl all receive list args.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# R9: top-level `iai-mcp doctor` handler (D7-10 placement
# precedent — alongside `iai-mcp schema-cleanup` rather than nested under
# `iai-mcp daemon`). doctor.py imports the daemon-state path constants
# lazily inside its check functions, so this top-level import is acyclic.
from iai_mcp.doctor import cmd_doctor

# ---------------------------------------------------------------------------
# constants -- daemon CLI group
# ---------------------------------------------------------------------------

# Re-export the daemon-side state paths so tests + uninstall can clear them
# in lock-step with `iai_mcp.concurrency` / `iai_mcp.daemon_state`. These
# duplicate Path.home() lookups so monkeypatching Path.home in tests works.
LOCK_PATH: Path = Path.home() / ".iai-mcp" / ".lock"
SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"
STATE_PATH: Path = Path.home() / ".iai-mcp" / ".daemon-state.json"

# Deploy artefact targets (Plan 04-01 created the templates; we install copies
# into the user's per-user system-level dirs).
LAUNCHD_TARGET: Path = Path.home() / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist"
SYSTEMD_TARGET: Path = Path.home() / ".config" / "systemd" / "user" / "iai-mcp-daemon.service"

# Repo-relative templates shipped with the package.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
LAUNCHD_TEMPLATE: Path = _PROJECT_ROOT / "deploy" / "launchd" / "com.iai-mcp.daemon.plist"
SYSTEMD_TEMPLATE: Path = _PROJECT_ROOT / "deploy" / "systemd" / "iai-mcp-daemon.service"

DAEMON_LABEL: str = "com.iai-mcp.daemon"
SERVICE_NAME: str = "iai-mcp-daemon.service"

# first-run consent banner. Wording cites RAM cost, Claude budget cap,
# opt-out command. Aborts unless user types lowercase 'y' (strict).
CONSENT_BANNER: str = """\
==============================================================================
IAI-MCP Sleep Daemon -- First Install Consent
==============================================================================

The sleep daemon runs in the background between Claude Code sessions to
perform neural consolidation (REM cycles, schema induction, drift detection).

Resource cost:
  - RAM: ~400 MB (bge-small-en-v1.5 embedding model kept warm to avoid cold-start;
    rises to ~2 GB if the opt-in bge-m3 model is selected via IAI_MCP_EMBED_MODEL)
  - CPU: brief bursts during REM cycles inside your learned quiet window
  - Disk: ~50MB/week in event logs + schema candidates

Claude subscription impact:
  - Max 1 `claude -p` call per night ("lucid moment" main insight)
  - Hard cap: 1% of daily subscription quota, 7% weekly buffer
  - ZERO API costs (no paid-API key -- uses your subscription only)

Opt out anytime:
  iai-mcp daemon uninstall

Continue? [y/N]: """


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _render_launchd_plist() -> str:
    """Pitfall 5: substitute the literal `/usr/local/bin/python3` placeholder
    AND `{USERNAME}` token in the template with sys.executable + actual user.
    """
    text = LAUNCHD_TEMPLATE.read_text()
    username = os.environ.get("USER") or Path.home().name
    text = text.replace("/usr/local/bin/python3", sys.executable)
    text = text.replace("{USERNAME}", username)
    return text


def _render_systemd_unit() -> str:
    """Pitfall 5 (systemd variant): substitute `/usr/bin/python3` template
    placeholder with the actual sys.executable so systemd resolves the right
    interpreter even when the user's venv lives outside /usr.
    """
    text = SYSTEMD_TEMPLATE.read_text()
    text = text.replace("/usr/bin/python3", sys.executable)
    return text


def _try_short_timeout_connect(timeout_ms: int = 250) -> bool:
    """Probe daemon socket reachability with a hard timeout. Returns True if
    connect succeeded. Used by ``capture-transcript --no-spawn`` (R3) to
    decide between inline ingest vs JSONL defer — hook is best-effort and
    must NEVER block session teardown waiting on a 5s cold-start.

    Honors the ``IAI_DAEMON_SOCKET_PATH`` env override (test isolation +
    HIGH-4 lock from Plan 07-04). Closes the probe socket immediately —
    we never write a request, only check that connect(2) returns.
    """
    import socket as _socket

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(timeout_ms / 1000.0)
    try:
        s.connect(sock_path)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError, _socket.timeout):
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _prompt_consent(stream_out=None) -> bool:
    """print the consent banner, read one line from stdin, return True
    only if the response stripped + lowercased equals exactly 'y'.

    Resolve sys.stderr at call time (NOT at module import) so pytest's capsys
    fixture can intercept the banner -- capsys swaps sys.stderr after our
    module is imported.
    """
    if stream_out is None:
        stream_out = sys.stderr
    print(CONSENT_BANNER, file=stream_out, end="")
    stream_out.flush()
    try:
        response = input("")
    except EOFError:
        return False
    return response.strip().lower() == "y"


def _record_consent_receipt() -> None:
    """D-10 audit trail: write a timestamped JSON receipt under
    ~/.iai-mcp/.consent-<ts>.json so a forensic review can verify the user
    actually consented (not bypassed via --yes). Failure to write the receipt
    is logged to stderr but never blocks the install."""
    state_dir = LOCK_PATH.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "consent": True,
        "ts": ts,
        "executable": sys.executable,
        "platform": platform.system(),
        "user": os.environ.get("USER") or "",
    }
    safe_ts = ts.replace(":", "").replace("-", "").replace(".", "")
    receipt = state_dir / f".consent-{safe_ts}.json"
    try:
        receipt.write_text(json.dumps(payload, indent=2))
        os.chmod(receipt, 0o600)
    except OSError as exc:
        print(f"warning: could not write consent receipt: {exc}", file=sys.stderr)


def _remove_state_files() -> None:
    """C4 invariant: clean uninstall removes ALL daemon-created state files."""
    for p in (LOCK_PATH, SOCKET_PATH, STATE_PATH):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"warning: could not remove {p}: {exc}", file=sys.stderr)


def _send_socket_request(req: dict, *, timeout: float = 30.0) -> dict | None:
    """One-shot NDJSON request/response over the daemon control socket.

    Returns None when the daemon is unreachable (socket missing, connection
    refused). Raises asyncio.TimeoutError if the daemon accepted the
    connection but never replied within `timeout` seconds.
    """

    async def _runner() -> dict | None:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(SOCKET_PATH)),
                timeout=5.0,
            )
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        except OSError:
            return None
        try:
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    return asyncio.run(_runner())


# ---------------------------------------------------------------------------
# daemon subcommand handlers
# ---------------------------------------------------------------------------


def cmd_daemon_install(args: argparse.Namespace) -> int:
    """DAEMON-10 install: render plist/unit, drop into per-user system path,
    enable via launchctl bootstrap or systemctl --user enable --now.

    --dry-run prints the would-be path + rendered contents and exits.
    --yes skips the consent banner.
    """
    dry_run = bool(getattr(args, "dry_run", False))
    yes = bool(getattr(args, "yes", False))

    if not yes and not dry_run:
        if not _prompt_consent():
            print("Install cancelled.", file=sys.stderr)
            return 1
        _record_consent_receipt()

    if _is_macos():
        content = _render_launchd_plist()
        target = LAUNCHD_TARGET
    elif _is_linux():
        content = _render_systemd_unit()
        target = SYSTEMD_TARGET
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1

    if dry_run:
        print(f"# Would install to: {target}")
        print(content)
        return 0

    # Write the rendered file; idempotent re-install is fine (overwrite).
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    try:
        os.chmod(target, 0o644)
    except OSError:
        pass

    uid = os.getuid()
    if _is_macos():
        # Idempotent bootstrap: bootout first if a previous version is loaded.
        # Both calls are best-effort; a fresh system has nothing to bootout.
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0 and result.stderr:
            print(
                f"warning: launchctl bootstrap returned {result.returncode}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
        subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{DAEMON_LABEL}"],
            check=False, capture_output=True,
        )
    else:
        # Linux: probe loginctl Linger state (Pitfall 8). If not enabled, try
        # to enable; if still not enabled after that, warn loudly.
        user = os.environ.get("USER") or ""
        linger_probe = subprocess.run(
            ["loginctl", "show-user", user, "--property=Linger"],
            check=False, capture_output=True, text=True,
        )
        if "Linger=yes" not in linger_probe.stdout:
            subprocess.run(
                ["loginctl", "enable-linger", user],
                check=False, capture_output=True,
            )
            linger_recheck = subprocess.run(
                ["loginctl", "show-user", user, "--property=Linger"],
                check=False, capture_output=True, text=True,
            )
            if "Linger=yes" not in linger_recheck.stdout:
                print(
                    "WARNING: loginctl enable-linger did not take effect -- "
                    "daemon may die at logout",
                    file=sys.stderr,
                )
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", SERVICE_NAME],
            check=False, capture_output=True,
        )

    print(f"Installed to {target}")
    return 0


def cmd_daemon_uninstall(args: argparse.Namespace) -> int:
    """C4 invariant: clean removal of plist/unit + ALL state files."""
    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Uninstall IAI-MCP daemon? "
                "(removes plist/unit + state files) [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Uninstall cancelled.", file=sys.stderr)
            return 1

    uid = os.getuid()
    if _is_macos():
        if LAUNCHD_TARGET.exists():
            subprocess.run(
                ["launchctl", "bootout", f"gui/{uid}", str(LAUNCHD_TARGET)],
                check=False, capture_output=True,
            )
            try:
                LAUNCHD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove plist: {exc}", file=sys.stderr)
    elif _is_linux():
        if SYSTEMD_TARGET.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", SERVICE_NAME],
                check=False, capture_output=True,
            )
            try:
                SYSTEMD_TARGET.unlink()
            except OSError as exc:
                print(f"warning: could not remove unit: {exc}", file=sys.stderr)
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False, capture_output=True,
            )

    _remove_state_files()
    print("Daemon uninstalled. State files removed.")
    return 0


def cmd_daemon_start(args: argparse.Namespace) -> int:
    uid = os.getuid()
    if _is_macos():
        subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{DAEMON_LABEL}"],
            check=False,
        )
    elif _is_linux():
        subprocess.run(
            ["systemctl", "--user", "start", SERVICE_NAME],
            check=False,
        )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_stop(args: argparse.Namespace) -> int:
    """Stop the singleton iai-mcp daemon (user-initiated shutdown).

    Sends SIGTERM to the daemon via launchctl (macOS) or systemctl --user
    (Linux). The daemon exits 0 on graceful shutdown; the supervisor
    respawns only on crash via the plist's `KeepAlive.Crashed=true`
    contract (commit 0cdc6a9). A user-initiated stop therefore takes the
    daemon down for good — no respawn — until the user explicitly starts
    it again.

    As informational telemetry only, we also write a
    `user_requested_shutdown=True` sentinel to .daemon-state.json before
    sending the signal. The daemon clears the sentinel on graceful
    shutdown via `_clear_user_shutdown_sentinel` (daemon.py:1002, called
    from main at daemon.py:1670). The sentinel is NOT consumed for any
    control-flow decision — it exists purely so post-mortem inspection of
    .daemon-state.json can distinguish a user-stop from other shutdown
    paths. The sentinel write is best-effort: a state-file failure must
    NOT block the SIGTERM (the user explicitly wants the daemon down).
    """
    # Best-effort sentinel write: we do NOT abort on failure.
    try:
        from iai_mcp.daemon_state import load_state, save_state

        state = load_state()
        state["user_requested_shutdown"] = True
        save_state(state)
    except Exception:
        # Persistence failure must not block the SIGTERM (user explicitly
        # wants the daemon down). Worst case: one extra respawn cycle.
        pass

    uid = os.getuid()
    if _is_macos():
        subprocess.run(
            ["launchctl", "kill", "SIGTERM", f"gui/{uid}/{DAEMON_LABEL}"],
            check=False,
        )
    elif _is_linux():
        subprocess.run(
            ["systemctl", "--user", "stop", SERVICE_NAME],
            check=False,
        )
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_status(args: argparse.Namespace) -> int:
    """Socket round-trip + version-skew detection."""
    try:
        resp = _send_socket_request({"type": "status"}, timeout=10.0)
    except asyncio.TimeoutError:
        print("daemon not responding", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- surface socket errors cleanly
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if resp is None:
        print("daemon not running")
        return 1

    # Version skew check: compare daemon's reported version with installed.
    try:
        from iai_mcp import __version__ as installed_version
    except Exception:
        installed_version = "unknown"
    daemon_version = resp.get("version", "unknown")
    if (
        daemon_version != "unknown"
        and installed_version != "unknown"
        and daemon_version != installed_version
    ):
        print(
            f"WARNING: daemon version {daemon_version} != "
            f"installed {installed_version} -- run iai-mcp daemon "
            f"stop && iai-mcp daemon start to restart",
            file=sys.stderr,
        )

    for k, v in resp.items():
        print(f"{k}: {v}")
    return 0


def cmd_daemon_logs(args: argparse.Namespace) -> int:
    follow = bool(getattr(args, "follow", False))
    lines = int(getattr(args, "lines", 50))
    if _is_macos():
        path = Path.home() / "Library" / "Logs" / "iai-mcp-daemon.stderr.log"
        argv = ["tail"]
        if follow:
            argv.append("-f")
        argv.extend(["-n", str(lines), str(path)])
        subprocess.run(argv, check=False)
    elif _is_linux():
        argv = ["journalctl", "--user", "-u", SERVICE_NAME, "-n", str(lines)]
        if follow:
            argv.append("-f")
        subprocess.run(argv, check=False)
    else:
        print(f"Unsupported OS: {platform.system()}", file=sys.stderr)
        return 1
    return 0


def cmd_daemon_force_rem(args: argparse.Namespace) -> int:
    """D-18 cooperative force: wait up to 15min for current cycle to finish."""
    try:
        resp = _send_socket_request({"type": "force_rem"}, timeout=15 * 60)
    except asyncio.TimeoutError:
        print("force_rem timed out after 15 minutes", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(json.dumps(resp))
    return 0


def cmd_daemon_pause(args: argparse.Namespace) -> int:
    seconds = int(args.seconds)
    try:
        resp = _send_socket_request(
            {"type": "pause", "seconds": seconds}, timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print(f"paused for {seconds}s")
    return 0


def cmd_daemon_resume(args: argparse.Namespace) -> int:
    try:
        resp = _send_socket_request({"type": "resume"}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if resp is None:
        print("daemon not running")
        return 1
    print("resumed")
    return 0


def cmd_daemon_configure(args: argparse.Namespace) -> int:
    """per-setting overrides written to ~/.iai-mcp/.daemon-state.json.

    Subcommands:
      - set-budget <float>          -- daily_quota_pct_override
      - set-cycle-count <int>       -- cycle_count_override
      - set-quiet-window HH:MM-HH:MM -- quiet_window_manual_override
      - disable-claude              -- claude_enabled = False (force Tier-0)
      - enable-claude               -- claude_enabled = True
    """
    from iai_mcp.daemon_state import load_state, save_state

    key = args.key
    value = getattr(args, "value", None)
    state = load_state()

    if key == "set-budget":
        if value is None:
            print("set-budget requires a float value", file=sys.stderr)
            return 2
        state["daily_quota_pct_override"] = float(value)
    elif key == "set-cycle-count":
        if value is None:
            print("set-cycle-count requires an int value", file=sys.stderr)
            return 2
        state["cycle_count_override"] = int(value)
    elif key == "set-quiet-window":
        if value is None or "-" not in value:
            print(
                "set-quiet-window requires HH:MM-HH:MM format",
                file=sys.stderr,
            )
            return 2
        start, end = value.split("-", 1)
        state["quiet_window_manual_override"] = [start.strip(), end.strip()]
    elif key == "disable-claude":
        state["claude_enabled"] = False
    elif key == "enable-claude":
        state["claude_enabled"] = True
    else:
        print(f"unknown configure key: {key}", file=sys.stderr)
        return 2

    save_state(state)
    print(f"{key} -> {value if value is not None else 'toggled'}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Show the most recent llm_health event in the user's local timezone."""
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore
    from iai_mcp.tz import load_user_tz, to_local

    store = MemoryStore()
    tz = load_user_tz()
    events = query_events(store, kind="llm_health", limit=1)
    if not events:
        print("llm_health: no events recorded")
        return 0
    e = events[0]
    local = to_local(e["ts"], tz)
    severity = e.get("severity") or "?"
    print(f"llm_health: {severity} at {local.isoformat()} ({tz.key})")
    print(f"  data: {e['data']}")
    return 0


def cmd_capture_transcript(args: argparse.Namespace) -> int:
    """Plan 06: batch-capture a Claude Code JSONL transcript into the store.

    Called by ~/.claude/hooks/iai-mcp-session-capture.sh on Stop event.
    Fail-safe by design: any exception logs and returns 0 so the hook never
    blocks session teardown.

    ``--no-spawn`` ALWAYS writes a deferred-captures JSONL file
    under ``~/.iai-mcp/.deferred-captures/<id>-<ts>.jsonl`` (D7.1-04 format)
    and exits 0 within 2s — NEVER spawning the daemon, NEVER importing
    ``iai_mcp.capture.capture_transcript`` (which transitively loads
    ``sentence_transformers`` / bge-small-en-v1.5 in a brand-new subprocess).
    The daemon's WAKE drain loop (Phase 7.1 R3 / Plan 07.1-06, in
    ``daemon.main()`` startup + ``_tick_body`` head) consumes the deferred
    file later with the daemon-process embedder that's already loaded.

    Default mode (without ``--no-spawn``) keeps inline-ingest
    behaviour unchanged — user-explicit ``iai-mcp capture-transcript``
    invocations still embed eagerly as documented.
    """
    import json
    import sys as _sys

    no_spawn = bool(getattr(args, "no_spawn", False))

    if no_spawn:
        # hook is best-effort. ALWAYS defer; the 250ms socket probe
        # and the reachable-inline branch are gone. Even when the daemon is
        # reachable we still write the JSONL file — the daemon's WAKE drain
        # picks it up within seconds with its already-loaded embedder, which
        # is dramatically cheaper than cold-loading bge-small-en-v1.5 in 286+
        # short-lived Stop-hook subprocesses per day.
        from iai_mcp.capture import write_deferred_captures

        try:
            out = write_deferred_captures(
                session_id=args.session_id,
                transcript_path=args.transcript_path,
                cwd=os.getcwd(),
                max_turns=args.max_turns,
            )
            print(json.dumps({"status": "deferred", "path": str(out)}, ensure_ascii=False))
            return 0
        except Exception as e:
            # Fail-safe: hook MUST exit 0. Log to stderr, return 0.
            print(
                f"capture-transcript --no-spawn: failed {type(e).__name__}: {e}",
                file=_sys.stderr,
            )
            return 0

    # Default path (no --no-spawn): existing behavior, unchanged.
    from iai_mcp.capture import capture_transcript
    from iai_mcp.store import MemoryStore

    try:
        store = MemoryStore()
        counts = capture_transcript(
            store,
            args.transcript_path,
            session_id=args.session_id,
            max_turns=args.max_turns,
        )
        print(json.dumps(counts, ensure_ascii=False))
        return 0
    except Exception as e:
        print(f"capture-transcript: failed {type(e).__name__}: {e}", file=_sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# Plan 06 capture-hooks installer (makes ambient WRITE-capture portable).
# ---------------------------------------------------------------------------

def _capture_hook_paths() -> tuple[Path, Path, Path]:
    """Return (hook_src_in_repo, hook_dst_in_home, settings_path)."""
    from pathlib import Path as _P
    import iai_mcp

    pkg_dir = _P(iai_mcp.__file__).resolve().parent
    # repo layout: <repo>/src/iai_mcp/cli.py -> <repo>/deploy/hooks/...
    repo_root = pkg_dir.parent.parent
    src = repo_root / "deploy" / "hooks" / "iai-mcp-session-capture.sh"
    dst = _P.home() / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    settings = _P.home() / ".claude" / "settings.json"
    return src, dst, settings


def _codex_capture_hook_paths() -> tuple[Path, Path, Path]:
    """Return (hook_src_in_repo, hook_dst_in_home, codex_hooks_json)."""
    from pathlib import Path as _P
    import iai_mcp

    pkg_dir = _P(iai_mcp.__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    src = repo_root / "deploy" / "hooks" / "iai-mcp-codex-session-capture.sh"
    dst = _P.home() / ".codex" / "hooks" / "iai-mcp-codex-session-capture.sh"
    settings = _P.home() / ".codex" / "hooks.json"
    return src, dst, settings


def _claude_desktop_config_path() -> Path | None:
    """Locate the Claude Desktop app config file, or None if Desktop isn't
    installed. Claude Desktop and Claude Code CLI use SEPARATE config files:

      - Claude Code CLI:  ~/.claude.json (managed by `claude mcp add`)
      - Claude Desktop:   platform-specific path (this function)

    So MCP registered via `claude mcp add` is NOT visible to Desktop, which
    is why iai-mcp has to be registered in both configs independently.
    """
    import platform as _plat
    home = Path.home()
    sysname = _plat.system()
    if sysname == "Darwin":
        p = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sysname == "Windows":
        appdata = os.environ.get("APPDATA") or str(home / "AppData" / "Roaming")
        p = Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:  # Linux / BSD
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(home / ".config")
        p = Path(xdg) / "Claude" / "claude_desktop_config.json"
    return p if p.parent.exists() else None


def _build_iai_mcp_server_entry(repo_root: Path) -> dict:
    """Build the mcpServers entry for iai-mcp, with absolute paths to the
    current install's wrapper + venv python. Same shape works for both
    Claude Code's ~/.claude.json and Claude Desktop's claude_desktop_config.json.
    """
    wrapper = repo_root / "mcp-wrapper" / "dist" / "index.js"
    # Best-effort guess at venv python: <repo>/.venv/bin/python if present.
    venv_py = repo_root / ".venv" / "bin" / "python"
    iai_mcp_python = str(venv_py) if venv_py.exists() else sys.executable
    iai_mcp_store = str(Path.home() / ".iai-mcp")
    return {
        "command": "node",
        "args": [str(wrapper)],
        "env": {
            "IAI_MCP_PYTHON": iai_mcp_python,
            "IAI_MCP_STORE": iai_mcp_store,
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }


def _patch_claude_desktop_config(action: str) -> str:
    """action: 'install' | 'uninstall'. Returns a status message for logging.

    install: add/overwrite mcpServers.iai-mcp in the Desktop config.
    uninstall: remove mcpServers.iai-mcp; leave other servers + preferences
    untouched. Idempotent. If Desktop isn't installed, return a skip message.
    """
    import json as _json
    import iai_mcp as _pkg
    repo_root = Path(_pkg.__file__).resolve().parent.parent.parent

    cfg_path = _claude_desktop_config_path()
    if cfg_path is None:
        return "Claude Desktop: not installed (no config dir) — skipped"

    if not cfg_path.exists():
        if action == "uninstall":
            return f"Claude Desktop: {cfg_path} absent — skipped"
        # install: create minimal config with just our entry.
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"mcpServers": {"iai-mcp": _build_iai_mcp_server_entry(repo_root)}}
        cfg_path.write_text(_json.dumps(data, indent=2))
        return f"Claude Desktop: created {cfg_path} with iai-mcp registered"

    try:
        data = _json.loads(cfg_path.read_text())
    except Exception as e:
        return f"Claude Desktop: {cfg_path} unreadable ({type(e).__name__}) — skipped"

    servers = data.setdefault("mcpServers", {})

    if action == "uninstall":
        if "iai-mcp" in servers:
            servers.pop("iai-mcp", None)
            cfg_path.write_text(_json.dumps(data, indent=2))
            return f"Claude Desktop: removed iai-mcp from {cfg_path}"
        return "Claude Desktop: iai-mcp not in config — no change"

    # install
    new_entry = _build_iai_mcp_server_entry(repo_root)
    if servers.get("iai-mcp") == new_entry:
        return f"Claude Desktop: {cfg_path} already has iai-mcp — no change"
    servers["iai-mcp"] = new_entry
    cfg_path.write_text(_json.dumps(data, indent=2))
    return f"Claude Desktop: patched {cfg_path} (iai-mcp registered)"


_CAPTURE_HOOK_MARKER = "iai-mcp-session-capture.sh"
_CODEX_CAPTURE_HOOK_MARKER = "iai-mcp-codex-session-capture.sh"


def _load_settings(path):
    import json as _json
    if not path.exists():
        return {}
    try:
        return _json.loads(path.read_text())
    except Exception:
        return {}


def _target_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "target", "claude") or "claude")


def _target_includes(target: str, name: str) -> bool:
    return target == "all" or target == name


def _patch_codex_hooks_config(action: str, command: str | None = None) -> str:
    """Install/uninstall the Codex Stop hook in ~/.codex/hooks.json."""
    import json as _json

    _, dst, settings = _codex_capture_hook_paths()
    data = _load_settings(settings)
    if not isinstance(data, dict):
        data = {}
    hooks = data.setdefault("hooks", {})
    stop_list = hooks.setdefault("Stop", [])

    def has_marker(entry: dict) -> bool:
        return any(
            _CODEX_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or [])
        )

    if action == "uninstall":
        kept = [entry for entry in stop_list if not has_marker(entry)]
        if len(kept) == len(stop_list):
            return f"Codex: no Stop entry to remove in {settings}"
        if kept:
            hooks["Stop"] = kept
        else:
            hooks.pop("Stop", None)
        if not hooks:
            data.pop("hooks", None)
        settings.write_text(_json.dumps(data, indent=2))
        return f"Codex: patched {settings} (Stop hook removed)"

    if any(has_marker(entry) for entry in stop_list):
        return f"Codex: {settings} already has Stop hook - no change"

    stop_list.append({
        "matcher": "",
        "hooks": [{
            "type": "command",
            "command": command or f"bash {dst}",
            "timeout": 35,
        }],
    })
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(_json.dumps(data, indent=2))
    return f"Codex: patched {settings} (Stop hook registered)"


def _install_codex_capture_hook() -> int:
    """Install the Codex Stop hook without touching Claude config."""
    import shlex
    import shutil
    import stat

    src, dst, _settings = _codex_capture_hook_paths()
    if not src.exists():
        print(f"ERROR: hook template missing in repo: {src}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"Codex: installed {dst}")
    print(_patch_codex_hooks_config("install", f"bash {shlex.quote(str(dst))}"))
    return 0


def _uninstall_codex_capture_hook() -> int:
    """Remove the Codex Stop hook script and config entry."""
    _, dst, _settings = _codex_capture_hook_paths()
    if dst.exists():
        dst.unlink()
        print(f"Codex: removed {dst}")
    else:
        print(f"Codex: not present {dst}")
    print(_patch_codex_hooks_config("uninstall"))
    return 0


def _codex_capture_hook_status() -> bool:
    """Print and return whether the Codex Stop hook is installed and wired."""
    src, dst, settings = _codex_capture_hook_paths()
    print(f"Codex template: {src}  {'PRESENT' if src.exists() else 'MISSING'}")
    print(f"Codex installed: {dst}  {'PRESENT' if dst.exists() else 'MISSING'}")
    data = _load_settings(settings)
    stop_list = data.get("hooks", {}).get("Stop", [])
    wired = any(
        any(_CODEX_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    print(f"Codex hooks.json: {settings}  {'WIRED' if wired else 'NOT WIRED'}")
    return dst.exists() and wired


def cmd_capture_hooks_install(args: argparse.Namespace) -> int:
    """Copy the Stop hook into ~/.claude/hooks/ and register it in settings.json."""
    import json as _json
    import shutil
    import stat

    target = _target_from_args(args)
    if _target_includes(target, "codex") and _install_codex_capture_hook() != 0:
        return 1
    if not _target_includes(target, "claude"):
        print("\nNext: restart Codex so it picks up the hook registration.")
        print("If hooks are disabled by policy, enable [features].hooks = true.")
        print("Verify: iai-mcp capture-hooks status --target codex")
        return 0

    src, dst, settings = _capture_hook_paths()
    if not src.exists():
        print(f"ERROR: hook template missing in repo: {src}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed: {dst}")

    settings.parent.mkdir(parents=True, exist_ok=True)
    data = _load_settings(settings)
    data.setdefault("hooks", {})
    stop_list = data["hooks"].setdefault("Stop", [])

    hook_cmd = f"bash {dst}"
    # Idempotent: skip if an identical command is already wired up.
    already = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    if already:
        print("settings.json already has Stop hook — no change")
    else:
        stop_list.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 35}]})
        settings.write_text(_json.dumps(data, indent=2))
        print(f"patched: {settings} (Stop hook registered)")

    # Claude Desktop is a separate app with its own mcpServers config —
    # register iai-mcp there too so ambient memory works for BOTH surfaces.
    desktop_msg = _patch_claude_desktop_config("install")
    print(desktop_msg)

    if _target_includes(target, "codex"):
        print("\nNext: restart Codex, Claude Code, and Claude Desktop.")
        print("If Codex hooks are disabled by policy, enable [features].hooks = true.")
    else:
        print("\nNext: fully quit + relaunch Claude Code AND Claude Desktop")
        print("      so both pick up the registration (macOS: `killall Claude`).")
    print("Verify: iai-mcp capture-hooks status")
    return 0


def cmd_capture_hooks_uninstall(args: argparse.Namespace) -> int:
    """Remove the Stop hook script and its settings.json entry (idempotent)."""
    import json as _json

    target = _target_from_args(args)
    if _target_includes(target, "codex"):
        _uninstall_codex_capture_hook()
    if not _target_includes(target, "claude"):
        return 0

    _, dst, settings = _capture_hook_paths()
    if dst.exists():
        dst.unlink()
        print(f"removed: {dst}")
    else:
        print(f"(not present) {dst}")

    if settings.exists():
        data = _load_settings(settings)
        stop_list = data.get("hooks", {}).get("Stop", [])
        kept = [
            entry for entry in stop_list
            if not any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
                       for h in (entry.get("hooks") or []))
        ]
        if len(kept) != len(stop_list):
            if kept:
                data["hooks"]["Stop"] = kept
            else:
                data["hooks"].pop("Stop", None)
            settings.write_text(_json.dumps(data, indent=2))
            print(f"patched: {settings} (Stop entry removed)")
        else:
            print(f"(no Stop entry to remove) {settings}")

    # Also unregister from Claude Desktop config.
    desktop_msg = _patch_claude_desktop_config("uninstall")
    print(desktop_msg)

    return 0


def cmd_capture_hooks_status(args: argparse.Namespace) -> int:
    """Show whether the Stop hook is installed and active on both surfaces."""
    import json as _json

    target = _target_from_args(args)
    codex_ok = True
    if _target_includes(target, "codex"):
        codex_ok = _codex_capture_hook_status()
        if not _target_includes(target, "claude"):
            if codex_ok:
                print("\nstatus: ACTIVE - Codex ambient capture will fire on Stop")
                return 0
            print("\nstatus: INACTIVE - Codex not fully wired. Run: iai-mcp capture-hooks install --target codex")
            return 1

    src, dst, settings = _capture_hook_paths()
    print(f"repo template: {src}  {'PRESENT' if src.exists() else 'MISSING'}")
    print(f"installed at:  {dst}  {'PRESENT' if dst.exists() else 'MISSING'}")

    data = _load_settings(settings)
    stop_list = data.get("hooks", {}).get("Stop", [])
    wired = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    print(f"Claude Code settings.json: {settings}  {'WIRED' if wired else 'NOT WIRED'}")

    # Claude Desktop (separate config file, separate app).
    desktop_cfg = _claude_desktop_config_path()
    if desktop_cfg is None:
        desktop_line = "Claude Desktop: not installed"
        desktop_wired = False
    elif not desktop_cfg.exists():
        desktop_line = f"Claude Desktop: {desktop_cfg} MISSING"
        desktop_wired = False
    else:
        try:
            d = _json.loads(desktop_cfg.read_text())
            desktop_wired = "iai-mcp" in d.get("mcpServers", {})
            desktop_line = f"Claude Desktop: {desktop_cfg}  {'WIRED' if desktop_wired else 'NOT WIRED'}"
        except Exception:
            desktop_line = f"Claude Desktop: {desktop_cfg} (unreadable)"
            desktop_wired = False
    print(desktop_line)

    ok = dst.exists() and wired
    # Desktop wiring is a bonus, not a requirement — if Desktop isn't
    # installed there's no surface to wire up. Only flag INACTIVE when
    # Desktop IS installed but not wired.
    desktop_problem = desktop_cfg is not None and desktop_cfg.exists() and not desktop_wired

    if ok and not desktop_problem and codex_ok:
        print(f"\nstatus: ACTIVE — ambient capture will fire on every Stop event "
              f"(Claude Code{'; Desktop also wired' if desktop_wired else ''})")
        return 0
    msg = []
    if not ok:
        msg.append("Claude Code not fully wired")
    if not codex_ok:
        msg.append("Codex not fully wired")
    if desktop_problem:
        msg.append("Claude Desktop present but iai-mcp not registered")
    print(f"\nstatus: INACTIVE — {'; '.join(msg)}. Run: iai-mcp capture-hooks install")
    return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    """Run the appropriate migration based on --from / --to version pair,
    OR a Plan 07.11-03 / crash-safe-reembed action (--resume / --rollback).

    Supported:
      --from=1 --to=2   -> Phase 2
      --from=2 --to=3   encryption-at-rest migration
      --from=3 --to=4   TEM factorization
      --rollback        Plan 07.11-03 drop records_v_new and (if needed)
                        restore records from records_old_<ts>. Routes to
                        migrate._rollback. Exit codes: 0 success, 1 user-
                        correctable error, 2 unrecoverable.
      --resume          Plan 07.11-03 continue an interrupted reembed
                        migration from migration_progress.json. Routes to
                        migrate._resume with the live store's embedder.
                        Same exit-code contract.

    Anything else returns exit code 2 with a clear error message.
    """
    from iai_mcp.store import MemoryStore
    store = MemoryStore()

    # Plan 07.11-03 / rollback / resume entry points. Mutually exclusive
    # with the --from/--to dispatch below; checked first so they short-circuit.
    if bool(getattr(args, "rollback", False)):
        from iai_mcp import migrate
        return migrate._rollback(store.db, store)
    if bool(getattr(args, "resume", False)):
        # Resume requires the same target embedder the original migration
        # used. The simplest contract: resume to the embedder configured in
        # the running environment (IAI_MCP_EMBED_MODEL / IAI_MCP_EMBED_DIM).
        # The progress-file's saved_target_dim is cross-checked in
        # migrate._resume — a mismatch returns rc=1.
        from iai_mcp import migrate
        from iai_mcp.embed import embedder_for_store
        target = embedder_for_store(store)
        return migrate._resume(store.db, store, target)

    from_v = int(getattr(args, "from_", 1))
    to_v = int(getattr(args, "to", 2))
    dry_run = bool(getattr(args, "dry_run", False))
    verbose = bool(getattr(args, "verbose", False))

    def _progress(i: int, n: int) -> None:
        if verbose:
            print(f"[{i + 1}/{n}] migrating...")

    if from_v == 1 and to_v == 2:
        from iai_mcp.migrate import migrate_v1_to_v2
        result = migrate_v1_to_v2(store, dry_run=dry_run, progress=_progress)
        prefix = "would migrate" if dry_run else "migrated"
        print(
            f"{prefix} {result['records_migrated']} records in "
            f"{result['duration_sec']:.2f}s "
            f"({result['previous_model']} -> {result['new_model']})"
        )
        return 0

    if from_v == 2 and to_v == 3:
        from iai_mcp.migrate import migrate_encryption_v2_to_v3
        result = migrate_encryption_v2_to_v3(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would encrypt" if dry_run else "encrypted"
        print(
            f"{prefix} {result['records_migrated']} records + "
            f"{result['events_migrated']} events in "
            f"{result['duration_sec']:.2f}s "
            f"(AES-256-GCM, iai:enc:v1:)"
        )
        return 0

    if from_v == 3 and to_v == 4:
        # CONN-05: TEM factorization migration. Renames the
        # legacy `hd_vector_json` (pa.string()) column to `structure_hv`
        # (pa.binary()) and backfills every row via tem.bind_structure().
        from iai_mcp.migrate import migrate_hd_vector_to_structure_hv_v3_to_v4
        result = migrate_hd_vector_to_structure_hv_v3_to_v4(
            store, dry_run=dry_run, progress=_progress
        )
        prefix = "would rename" if dry_run else "renamed"
        print(
            f"{prefix} {result['updated']} records' "
            f"hd_vector_json->structure_hv column in "
            f"{result['duration_ms'] / 1000:.2f}s "
            f"(schema v3->v4, TEM factorization, D=10000 BSC packed)"
        )
        return 0

    print(
        f"unsupported migration --from={from_v} --to={to_v}; "
        f"supported: 1->2 (Plan 02-01 schema), 2->3 (Plan 02-08 encryption), "
        f"3->4 (Plan 03-01 TEM factorization)",
        file=sys.stderr,
    )
    return 2


def cmd_crypto_status(args: argparse.Namespace) -> int:
    """Phase 07.10 report file-backend key state (no keyring mention).

    Output is a single JSON document with the file-backend invariants:
      - backend = "file"
      - path = absolute key-file path
      - present = file exists
      - mode = "0o600" + mode_secure flag (true iff group/world bits are zero)
      - uid + uid_matches_process flag
      - length_bytes + length_valid (== KEY_BYTES)
      - passphrase_fallback_set (whether IAI_MCP_CRYPTO_PASSPHRASE is set)
      - hint when the file is missing (D-04 dual-remediation message)

    Never prints the key bytes (D-09 information-disclosure mitigation).
    No "keyring" string in the output (D-09 — keyring backend retired).
    """
    import json as _json
    import os as _os

    from iai_mcp.crypto import CIPHERTEXT_PREFIX, CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()

    present = path.exists()
    status: dict[str, object] = {
        "user_id": user_id,
        "backend": "file",
        "path": str(path),
        "present": present,
        "algorithm": "AES-256-GCM",
        "format": CIPHERTEXT_PREFIX,
    }

    if present:
        st = path.stat()
        mode_octal = f"0o{st.st_mode & 0o777:03o}"
        length = st.st_size
        status["mode"] = mode_octal
        status["mode_secure"] = (st.st_mode & 0o077 == 0)
        status["uid"] = st.st_uid
        status["uid_matches_process"] = (st.st_uid == _os.geteuid())
        status["length_bytes"] = length
        status["length_valid"] = (length == KEY_BYTES)
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
    else:
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
        status["hint"] = (
            "no key file. Run `iai-mcp crypto migrate-to-file` "
            "(existing Keychain key) or `iai-mcp crypto init` "
            "(fresh install), or set IAI_MCP_CRYPTO_PASSPHRASE."
        )

    print(_json.dumps(status, indent=2))
    return 0


def cmd_crypto_rotate(args: argparse.Namespace) -> int:
    """Plan 02-08 (Phase 07.10 update): rotate the encryption key +
    re-encrypt every record.

    Flow:
    1. Load current key + decrypt all records into in-memory MemoryRecord list.
    2. Rotate the key file (writes a fresh 32 bytes via _try_file_set, atomic
       temp+rename, mode 0o600). also invalidates the cached
       AESGCM bound to the old key (Phase 07.7 store.py:391 cached_property)
       so subsequent encrypts use the fresh key.
    3. Re-encrypt every record with the new key via a delete+insert cycle.

    Events data_json is also re-encrypted (mirrors v2->v3 behaviour).
    """
    import json as _json

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import (
        EVENTS_TABLE,
        MemoryStore,
        RECORDS_TABLE,
        _uuid_literal,
    )

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)

    # 1) Read everything under the old key (decryption is automatic).
    decrypted_records = store.all_records()

    # Decrypt events payloads up front so we can re-encrypt after rotation.
    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    decrypted_events: list[dict] = []
    from iai_mcp.crypto import decrypt_field, is_encrypted
    for _, row in events_df.iterrows():
        raw = row.get("data_json") or "{}"
        eid = str(row["id"])
        if is_encrypted(raw):
            try:
                raw = decrypt_field(
                    raw, store._key(), associated_data=eid.encode("ascii")
                )
            except Exception:
                raw = "{}"
        decrypted_events.append({"id": eid, "data_json": raw})

    # 2) Rotate the key (this flips store._crypto_key via wrapper cache).
    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key  # Force subsequent encrypts under the fresh key.
    # invalidate the cached AESGCM bound to the old key
    # (Phase 07.7 cached_property at store.py:391). Without this, the next
    # encrypt would use AESGCM(old_key) and produce ciphertext that cannot
    # be decrypted under new_key.
    store._invalidate_aesgcm_cache()

    # 3) Re-encrypt every record via delete + insert (MVCC-safe).
    tbl = store.db.open_table(RECORDS_TABLE)
    record_count = 0
    for rec in decrypted_records:
        try:
            tbl.delete(f"id = '{_uuid_literal(rec.id)}'")
        except Exception:
            pass
        # store.insert() encrypts using the new cached key.
        try:
            store.insert(rec)
            record_count += 1
        except Exception:
            continue

    # Re-encrypt events data_json under the new key.
    event_count = 0
    for ev in decrypted_events:
        ad = ev["id"].encode("ascii")
        new_ct = encrypt_field(ev["data_json"], new_key, associated_data=ad)
        try:
            events_tbl.update(
                where=f"id = '{ev['id']}'",
                values={"data_json": new_ct},
            )
            event_count += 1
        except Exception:
            continue

    print(
        _json.dumps(
            {
                "status": "rotated",
                "user_id": user_id,
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
                "algorithm": "AES-256-GCM",
                "format": "iai:enc:v1:",
            },
            indent=2,
        )
    )
    try:
        from iai_mcp.crypto_key_watch import sync_crypto_key_watcher_to_disk
        from iai_mcp.events import write_event

        write_event(
            store,
            kind="crypto_key_rotated",
            data={
                "source": "cli_rotate",
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
            },
            severity="info",
        )
        sync_crypto_key_watcher_to_disk(store)
    except Exception:
        pass
    return 0


def cmd_crypto_recover_prior_key(args: argparse.Namespace) -> int:
    """Re-stage all records and swap after decrypting with a prior AES key."""
    import json as _json

    from iai_mcp.crypto import KEY_BYTES
    from iai_mcp.migrate import migrate_crypto_recover_prior_key
    from iai_mcp.store import MemoryStore

    path: Path = args.prior_key_file
    try:
        prior = path.read_bytes()
    except OSError as exc:
        print(f"cannot read prior key file: {exc}", file=sys.stderr)
        return 1
    if len(prior) != KEY_BYTES:
        print(
            f"prior key file must be exactly {KEY_BYTES} bytes, got {len(prior)}",
            file=sys.stderr,
        )
        return 1
    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_crypto_recover_prior_key(
            store, prior, dry_run=bool(getattr(args, "dry_run", False)),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_redact_undecryptable(args: argparse.Namespace) -> int:
    """CLI entry for literal_surface redaction when decrypt fails."""
    import json as _json

    from iai_mcp.migrate import migrate_redact_undecryptable_records
    from iai_mcp.store import MemoryStore

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_redact_undecryptable_records(store)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_migrate_to_file(args: argparse.Namespace) -> int:
    """Phase 07.10 one-time migration from macOS Keychain to file backend.

    Reads the existing key from the macOS Keychain (the call that hangs in
    launchd context — this command MUST be run from an interactive Terminal
    so the Keychain ACL prompt can appear and the user can click "Always Allow"),
    writes it to ``{store_root}/.crypto.key``, verifies a round-trip read.

    Idempotent: a valid existing file is a no-op success that does NOT touch
    keyring (D-08, case 9). If the file exists but is malformed, the
    command refuses with a clear error pointing at the file path; user must
    remove the file manually before retrying.

    Default ``--keep-keychain`` leaves the keyring entry in place (lower-risk
    default; user can manually delete via Keychain Access.app).
    ``--delete-keychain`` deletes the entry only AFTER round-trip verification
    succeeds.
    """
    import base64 as _b64
    # LOCAL import: crypto.py + everything else stays keyring-free at module
    # scope. The migration command itself is the ONLY in-process code path that
    # imports keyring.
    import keyring as _keyring
    import keyring.errors as _keyring_errors

    from iai_mcp.crypto import (
        CryptoKey,
        CryptoKeyError,
        KEY_BYTES,
        SERVICE_NAME_DEFAULT,
    )

    user_id = getattr(args, "user_id", None) or "default"
    keep_keychain = getattr(args, "keep_keychain", True)

    ck = CryptoKey(user_id=user_id)

    # Idempotent path (D-08, case 9): if the file is already valid, exit
    # 0 without touching keyring.
    try:
        existing = ck._try_file_get()
    except CryptoKeyError as exc:
        print(
            f"refusing: existing key file is malformed: {exc}",
            file=sys.stderr,
        )
        return 1
    if existing is not None:
        print(f"already migrated: {ck._key_file_path()}")
        return 0

    # Read from macOS Keychain (this is THE call that hangs in launchd;
    # interactive Terminal only).
    try:
        encoded = _keyring.get_password(SERVICE_NAME_DEFAULT, user_id)
    except _keyring_errors.NoKeyringError:
        print(
            "no keyring backend available; nothing to migrate. "
            "If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=sys.stderr,
        )
        return 1
    except _keyring_errors.KeyringError as exc:
        print(f"keyring read failed: {exc}", file=sys.stderr)
        return 1
    if encoded is None:
        print(
            f"no key found in keyring for user_id={user_id!r}. "
            f"If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=sys.stderr,
        )
        return 1

    try:
        source = _b64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception as exc:
        print(f"keyring entry is malformed: {exc}", file=sys.stderr)
        return 1
    if len(source) != KEY_BYTES:
        print(
            f"keyring entry has wrong length {len(source)} (expected {KEY_BYTES})",
            file=sys.stderr,
        )
        return 1

    # Write via the atomic helper.
    try:
        ck._try_file_set(source)
    except Exception as exc:
        print(f"failed to write key file: {exc}", file=sys.stderr)
        return 1

    # Round-trip verification: read what we just wrote, byte-compare.
    try:
        roundtrip = ck._try_file_get()
    except CryptoKeyError as exc:
        # Read-back failed; remove the partial file.
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(f"round-trip verification failed: {exc}", file=sys.stderr)
        return 1
    if roundtrip != source:
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(
            "round-trip verification failed: bytes differ", file=sys.stderr
        )
        return 1

    # Success path.
    path = ck._key_file_path()
    print(f"migrated: {path} (mode 0o600, {KEY_BYTES} bytes)")

    if not keep_keychain:
        try:
            _keyring.delete_password(SERVICE_NAME_DEFAULT, user_id)
            print(f"deleted keyring entry for user_id={user_id!r}")
        except _keyring_errors.PasswordDeleteError:
            # Already absent — treat as success.
            pass
        except _keyring_errors.KeyringError as exc:
            # Non-fatal: file is written + verified, keyring delete failed;
            # print warning and continue (exit 0).
            print(
                f"warning: failed to delete keyring entry: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            "keyring entry kept (default). "
            "To remove manually, run "
            "`iai-mcp crypto migrate-to-file --delete-keychain` "
            "or use macOS Keychain Access.app."
        )

    return 0


def cmd_crypto_init(args: argparse.Namespace) -> int:
    """Phase 07.10 generate a fresh ``.crypto.key`` (fresh installs only).

    Refuses if the file already exists (any state, valid or malformed). The
    ONLY code path in the project that creates a fresh key — daemon
    refusal-to-start explicitly forbids silent key generation.

    To rotate an existing key, use ``iai-mcp crypto rotate``. To wipe and
    start over, the user must remove the file manually before re-running
    ``crypto init``.
    """
    import secrets as _secrets

    from iai_mcp.crypto import CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()
    if path.exists():
        print(
            f"refusing: key file already exists at {path}. "
            f"To rotate, run `iai-mcp crypto rotate`. "
            f"To wipe and start over, remove the file manually first.",
            file=sys.stderr,
        )
        return 1
    fresh = _secrets.token_bytes(KEY_BYTES)
    ck._try_file_set(fresh)
    print(f"created: {path} (mode 0o600, {KEY_BYTES} bytes)")
    return 0


def cmd_topology(args: argparse.Namespace) -> int:
    """Plan 03-02 CONN-07: print live small-world topology snapshot.

    One key:value line per metric:

        C: <average clustering>
        L: <characteristic path length>
        sigma: <fast_sigma() | "insufficient_data">
        communities: <Leiden community count>
        rich_club_ratio: <|rich_club| / N>
        N: <node count>
        regime: <"developmental" | "mid_life_drift" | "healthy" | "insufficient_data">

    sigma is a CYBERNETIC DIAGNOSTIC; never a routing decision (constitutional
    guard). The CLI is a print-only command -- no event writes,
    no state mutation. compute_and_emit() runs in S4's offline pass instead
    (see `iai_mcp.s4.run_offline_pass`).
    """
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.sigma import compute_topology_snapshot
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    graph, _assignment, _rich_club = build_runtime_graph(store)
    snap = compute_topology_snapshot(graph)

    def _fmt(v) -> str:
        if v is None:
            return "insufficient_data"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    print(f"C: {_fmt(snap.get('C'))}")
    print(f"L: {_fmt(snap.get('L'))}")
    print(f"sigma: {_fmt(snap.get('sigma'))}")
    print(f"communities: {_fmt(snap.get('community_count'))}")
    print(f"rich_club_ratio: {_fmt(snap.get('rich_club_ratio'))}")
    print(f"N: {_fmt(snap.get('N'))}")
    print(f"regime: {_fmt(snap.get('regime'))}")
    return 0


def cmd_trajectory(args: argparse.Namespace) -> int:
    """Aggregate M1..M6 trajectory events (D-32, OPS-08, Plan 02-04)."""
    from datetime import datetime, timedelta, timezone

    from iai_mcp.store import MemoryStore
    from iai_mcp.trajectory import METRIC_NAMES, aggregate_trajectory

    store = MemoryStore()
    weeks = getattr(args, "since", None)
    since = None
    if weeks is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(weeks))
    data = aggregate_trajectory(store, since=since)
    if not any(data.get(m) for m in METRIC_NAMES):
        print("no trajectory data recorded")
        return 0
    for metric in METRIC_NAMES:
        points = data.get(metric, [])
        if not points:
            print(f"{metric.upper()}: (no data)")
            continue
        values = [v for _, v in points]
        n = len(values)
        mean = sum(values) / n
        print(
            f"{metric.upper()}: n={n} mean={mean:.3f} "
            f"min={min(values):.3f} max={max(values):.3f}"
        )
    return 0


def _redact_shield_data(data: dict) -> str:
    """Render a shield event's data dict with matched-pattern redaction.

    T-02-05-02: shield_rejection / shield_flag events store the matched
    patterns. CLI output shows ONLY the count to avoid leaking the shield's
    signal-word dictionary to attackers inspecting logs.
    """
    matched = data.get("matched") or []
    tier = data.get("tier", "-")
    record_id = data.get("record_id", "-")
    action = data.get("action", "-")
    return (
        f"tier={tier} action={action} "
        f"matched_count={len(matched)} record_id={record_id}"
    )


def _format_audit_event(event: dict, tz) -> str:
    """Single-line audit event rendering in the user's local TZ."""
    from iai_mcp.tz import to_local

    ts = event.get("ts")
    try:
        local_ts = to_local(ts, tz) if ts is not None else None
    except Exception:
        local_ts = None
    ts_str = local_ts.isoformat() if local_ts is not None else str(ts)

    kind = event.get("kind", "?")
    sev = event.get("severity") or "-"
    data = event.get("data") or {}
    if kind in ("shield_rejection", "shield_flag", "shield_log"):
        data_str = _redact_shield_data(data)
    else:
        data_str = str(data)[:200]
    return f"[{ts_str}] {kind:32s} [{sev:8s}] {data_str}"


def cmd_audit(args: argparse.Namespace) -> int:
    """Render identity-event audit log.

    Accepts a sub-command via the `audit_sub` attribute:
      - None / 'all'      -- full audit (s5_* + shield_* + drift alerts)
      - 'shield'          -- shield events only
      - 'drift'           -- runs detect_drift_anomaly + prints status
      - 'identity'        -- s5_* events only (no shield)

    Shared flags: --since WEEKS, --severity SEV.
    """
    from datetime import datetime, timedelta, timezone

    from iai_mcp.s5 import (
        AUDIT_EVENT_KINDS,
        audit_identity_events,
        detect_drift_anomaly,
    )
    from iai_mcp.store import MemoryStore
    from iai_mcp.tz import load_user_tz

    store = MemoryStore()
    tz = load_user_tz()

    since_raw = getattr(args, "since", None)
    since = None
    if since_raw is not None:
        since = datetime.now(timezone.utc) - timedelta(weeks=int(since_raw))

    sub = getattr(args, "audit_sub", None)

    # Subcommand: drift -- runs detection + reports.
    if sub == "drift":
        alerts = detect_drift_anomaly(store)
        if not alerts:
            print("drift: no anomaly detected (M4 variance stable)")
        else:
            for a in alerts:
                print(
                    f"drift: variance increasing across "
                    f"{a.get('window_sessions')} sessions; "
                    f"first={a.get('first_value'):.3f} "
                    f"last={a.get('last_value'):.3f}"
                )
        return 0

    # Subcommand: shield -- only shield-family events.
    if sub == "shield":
        kinds = ("shield_rejection", "shield_flag", "shield_log")
        events = audit_identity_events(store, since=since, kinds=kinds)
        severity = getattr(args, "severity", None)
        if severity:
            events = [e for e in events if e.get("severity") == severity]
        if not events:
            print("audit shield: no events recorded")
            return 0
        for e in events:
            print(_format_audit_event(e, tz))
        return 0

    # Subcommand: identity -- only s5_* + cross-lingual warnings.
    if sub == "identity":
        kinds = (
            "s5_invariant_update",
            "s5_invariant_proposal",
            "s5_cooldown_block",
            "s5_drift_alert",
            "identity_cross_lingual_warning",
        )
        events = audit_identity_events(store, since=since, kinds=kinds)
        severity = getattr(args, "severity", None)
        if severity:
            events = [e for e in events if e.get("severity") == severity]
        if not events:
            print("audit identity: no events recorded")
            return 0
        for e in events:
            print(_format_audit_event(e, tz))
        return 0

    # Default: full audit.
    events = audit_identity_events(store, since=since, kinds=AUDIT_EVENT_KINDS)
    severity = getattr(args, "severity", None)
    if severity:
        events = [e for e in events if e.get("severity") == severity]
    if not events:
        print("No identity events recorded")
        return 0
    for e in events:
        print(_format_audit_event(e, tz))
    return 0


def cmd_schema_cleanup(args: argparse.Namespace) -> int:
    """Plan 06-05 R8: schema-cleanup CLI dispatch.

    Soft-deletes duplicate schema records that accumulated in production
    stores BEFORE made `persist_schema` idempotent.

    Default mode is --dry-run (Beer VSM S2 anti-oscillation reversibility).
    --apply requires the explicit flag; no interactive prompts so the
    flow is reproducible and testable.

    `--store-path` targets the IAI root directory (the path passed to
    MemoryStore() — contains the `lancedb/` subdir with the actual tables).
    Default is ~/.iai-mcp (matches MemoryStore() no-args default per
    DEFAULT_STORAGE_PATH).
    """
    from iai_mcp.migrate import cleanup_schema_duplicates
    from iai_mcp.store import MemoryStore

    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        # Match MemoryStore() default semantics: store.root = ~/.iai-mcp
        # (the IAI root); LanceDB tables live at store.root / "lancedb".
        store_path = Path.home() / ".iai-mcp"

    if not store_path.exists():
        print(
            f"error: store path does not exist: {store_path}",
            file=sys.stderr,
        )
        return 2

    apply = bool(getattr(args, "apply", False))

    store = MemoryStore(path=store_path)
    summary = cleanup_schema_duplicates(
        store, apply=apply, store_path=store_path,
    )

    mode_str = summary.get("mode", "dry-run")
    print(f"iai-mcp schema-cleanup [{mode_str}]")
    print(f"  groups (patterns with N>1 duplicates): {summary.get('groups', 0)}")
    print(f"  keepers (one per group):               {summary.get('keepers', 0)}")
    print(
        f"  pruned (soft-deleted, tier=semantic_pruned): "
        f"{summary.get('pruned', 0)}"
    )
    print(
        f"  edges to reinforce onto keepers:       "
        f"{summary.get('edges_reinforced', 0)}"
    )
    if summary.get("snapshot_dir"):
        print(f"  snapshot directory:                    {summary['snapshot_dir']}")
    if mode_str == "dry-run" and summary.get("groups", 0) > 0:
        print()
        print("  Run with --apply to execute.")
    return 0


# ---------------------------------------------------------------------------
# Plan 07.14-01 one-shot LanceDB compaction CLI
# ---------------------------------------------------------------------------
#
# Root-cause fix for the runaway records.lance version-manifest pile that
# dominates daemon cold-start time. Re-uses the existing
# `optimize_lance_storage(retention=timedelta(days=0))` helper from
# `iai_mcp.maintenance` (D7.3-09 never-raises contract) wrapped in:
#   - daemon-stopped pre-flight (psutil cmdline check rules out PID-recycle)
#   - record-id set equality assertion (verbatim-recall invariant; #2)
#   - audit JSON trail (UTC ISO timestamp; mirrors `.consent-{ts}.json` shape)
#
# This CLI runs WITH DAEMON STOPPED, so `_should_yield_to_mcp` is irrelevant
# (D-05 #1). Per #4 the optimize call is pure storage compaction —
# never reads or paraphrases stored `literal_surface`.
# ---------------------------------------------------------------------------


def _maintenance_compact_preflight_daemon_alive() -> str | None:
    """Return None if the daemon is NOT alive (safe to proceed); return a
    friendly error string if alive (caller prints to stderr + returns 1).

    Defense in depth: read `~/.iai-mcp/.daemon-state.json`, extract
    `daemon_pid`. If absent, daemon is not alive → None. If present, check
    `os.kill(pid, 0)` (does NOT signal — only checks process existence).
    If alive, confirm `psutil.Process(pid).cmdline()` contains
    `iai_mcp.daemon` to rule out PID-recycle false positives.
    """
    import json as _json
    import os as _os

    if not STATE_PATH.exists():
        return None
    try:
        state = _json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    pid = state.get("daemon_pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        _os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    except OSError:
        return None
    # Process exists. Confirm it is iai_mcp.daemon (not PID recycle).
    try:
        import psutil
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline())
    except Exception:
        # If psutil cannot inspect, conservatively treat as alive — REFUSE.
        return (
            f"daemon running (pid {pid}); run `iai-mcp daemon stop` "
            f"first, then retry"
        )
    if "iai_mcp.daemon" not in cmdline:
        return None  # PID recycle — not our daemon.
    return (
        f"daemon running (pid {pid}); run `iai-mcp daemon stop` first, "
        f"then retry"
    )


def _maintenance_compact_metrics(
    records_lance_dir: Path,
    store: object | None = None,
) -> dict:
    """Capture metrics for the records table.

    Returns dict with keys: versions_count, size_mb, records_count,
    record_id_set. `store` may be None on the dry-run pass when caller
    only walks the directory; on the apply pass it must be a live
    MemoryStore so we can read tbl.count_rows() and the record-id set
    via tbl.to_pandas(columns=['id']).
    """
    versions_count = 0
    versions_dir = records_lance_dir / "_versions"
    if versions_dir.exists():
        versions_count = sum(
            1 for _ in versions_dir.glob("*.manifest")
        )
    size_bytes = 0
    for p in records_lance_dir.rglob("*"):
        try:
            if p.is_file():
                size_bytes += p.stat().st_size
        except OSError:
            continue
    size_mb = round(size_bytes / (1024 * 1024), 1)
    records_count = 0
    record_id_set: set[str] = set()
    if store is not None:
        try:
            tbl = store.db.open_table("records")
            records_count = int(tbl.count_rows())
            df = tbl.to_pandas(columns=["id"])
            record_id_set = {str(x) for x in df["id"].tolist()}
        except Exception:
            pass
    return {
        "versions_count": versions_count,
        "size_mb": size_mb,
        "records_count": records_count,
        "record_id_set": record_id_set,
    }


def _maintenance_compact_dry_run(
    store_path: Path, records_lance_dir: Path,
) -> int:
    """--dry-run: open the store, capture pre-metrics, print JSON; do NOT
    call optimize, do NOT write an audit file.
    """
    import json as _json
    from iai_mcp.store import MemoryStore

    store = None
    try:
        store = MemoryStore(path=store_path)
    except Exception as exc:
        print(
            f"warning: could not open MemoryStore (records_count + "
            f"record_id_set will be 0): {exc}",
            file=sys.stderr,
        )
    metrics = _maintenance_compact_metrics(records_lance_dir, store=store)
    out = {
        "mode": "dry-run",
        "metrics": {
            "pre": {
                k: v for k, v in metrics.items() if k != "record_id_set"
            },
            "post": None,
        },
        "would_invoke": "optimize_lance_storage(retention=0d)",
    }
    print(_json.dumps(out, indent=2))
    return 0


def _maintenance_compact_apply(
    store_path: Path, records_lance_dir: Path,
) -> int:
    """--apply: open store, capture pre-metrics, call optimize(retention=0d)
    on records/edges/events via the existing helper, capture post-metrics,
    assert record-id set equality on the records table, write audit file.
    """
    import json as _json
    import time as _time
    from datetime import datetime, timedelta, timezone
    from iai_mcp.maintenance import optimize_lance_storage
    from iai_mcp.store import MemoryStore

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = (
        Path.home() / ".iai-mcp" / f".maintenance-compact-{ts}.json"
    )

    store = MemoryStore(path=store_path)
    pre_metrics = _maintenance_compact_metrics(
        records_lance_dir, store=store,
    )
    pre_id_set = pre_metrics["record_id_set"]

    t0 = _time.monotonic()
    report = optimize_lance_storage(
        store, retention=timedelta(days=0),
    )
    elapsed = round(_time.monotonic() - t0, 3)

    # Post: re-open store for fresh metadata view (helper docstring D7.3-09
    # mentions some LanceDB versions cache table metadata on the original
    # handle until refresh).
    store_after = MemoryStore(path=store_path)
    post_metrics = _maintenance_compact_metrics(
        records_lance_dir, store=store_after,
    )
    post_id_set = post_metrics["record_id_set"]

    # Verbatim-recall invariant — record-id set equality (D-05 #2).
    if pre_id_set != post_id_set:
        missing = pre_id_set - post_id_set
        extra = post_id_set - pre_id_set
        failed_path = (
            Path.home() / ".iai-mcp"
            / f".maintenance-compact-FAILED-{ts}.json"
        )
        failed_payload = {
            "command": "iai-mcp maintenance compact-records --apply",
            "timestamp_utc": ts,
            "status": "aborted",
            "reason": "record_id_set divergence post-optimize",
            "metrics_pre": {
                k: v for k, v in pre_metrics.items()
                if k != "record_id_set"
            },
            "metrics_post": {
                k: v for k, v in post_metrics.items()
                if k != "record_id_set"
            },
            "missing_ids_count": len(missing),
            "extra_ids_count": len(extra),
            "missing_ids_sample": list(sorted(missing))[:10],
            "extra_ids_sample": list(sorted(extra))[:10],
            "optimize_report": report,
            "elapsed_sec": elapsed,
        }
        try:
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(_json.dumps(failed_payload, indent=2))
        except OSError:
            pass
        print(
            f"ABORT: record_id_set divergence — missing={len(missing)} "
            f"extra={len(extra)}; details written to {failed_path}",
            file=sys.stderr,
        )
        return 1

    payload = {
        "command": "iai-mcp maintenance compact-records --apply",
        "timestamp_utc": ts,
        "status": "ok",
        "metrics_pre": {
            k: v for k, v in pre_metrics.items() if k != "record_id_set"
        },
        "metrics_post": {
            k: v for k, v in post_metrics.items() if k != "record_id_set"
        },
        "elapsed_sec": elapsed,
        "optimize_report": report,
    }
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(_json.dumps(payload, indent=2))
    except OSError as exc:
        print(
            f"warning: could not write audit file {audit_path}: {exc}",
            file=sys.stderr,
        )
    print(_json.dumps({
        "mode": "apply",
        "metrics": {
            "pre": payload["metrics_pre"],
            "post": payload["metrics_post"],
        },
        "elapsed_sec": elapsed,
        "audit_file": str(audit_path),
        "status": "ok",
    }, indent=2))
    return 0


def cmd_maintenance_compact_records(args: argparse.Namespace) -> int:
    """Plan 07.14-01 one-shot LanceDB compaction CLI.

    Pre-flight: refuse if the daemon process is alive (PID + cmdline check).
    Mode: `--dry-run` (default) prints metrics-only JSON; `--apply --yes`
    runs `optimize_lance_storage(retention=timedelta(days=0))` on the
    records/edges/events tables, asserts record-id set equality on the
    records table, and writes an audit JSON.

    Exit codes: 0 ok, 1 pre-flight refusal or invariant abort, 2 wrong-flag
    combo (apply without yes on a non-tty).

    This CLI runs with the daemon stopped, so `_should_yield_to_mcp` is
    irrelevant. Per #4 the optimize call never paraphrases or smooths
    stored content — it is pure storage compaction.
    """
    # Resolve store path (same convention as cmd_schema_cleanup line 1708).
    if args.store_path is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    records_lance_dir = store_path / "lancedb" / "records.lance"
    if not records_lance_dir.exists():
        print(
            f"error: records.lance not found at {records_lance_dir}",
            file=sys.stderr,
        )
        return 1

    apply = bool(getattr(args, "apply", False))
    yes = bool(getattr(args, "yes", False))
    # Default to dry-run when neither flag set.
    if not apply:
        # Treat `--dry-run` and "neither flag" identically.
        return _maintenance_compact_dry_run(store_path, records_lance_dir)

    # --apply path: pre-flight + optional consent + optimize + invariant.
    # Pre-flight 1: daemon alive?
    refusal = _maintenance_compact_preflight_daemon_alive()
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 1

    # Pre-flight 2: --apply on non-tty without --yes is refused.
    if not yes and not sys.stdin.isatty():
        print(
            "error: --apply on non-tty requires --yes (refusing to proceed "
            "without interactive consent or explicit --yes)",
            file=sys.stderr,
        )
        return 2

    # Pre-flight 3: interactive consent (mirrors cmd_daemon_install D-21).
    if not yes:
        prompt = (
            "About to compact records.lance via optimize(cleanup_older_than="
            "0d). Daemon must be stopped. Type 'y' to proceed: "
        )
        try:
            response = input(prompt)
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("aborted: user did not consent", file=sys.stderr)
            return 1

    return _maintenance_compact_apply(store_path, records_lance_dir)


# ---------------------------------------------------------------------------
# -- iai-mcp lifecycle status
# ---------------------------------------------------------------------------

def _format_relative(ts_iso: str, now: datetime | None = None) -> str:
    """Render a friendly elapsed string for an ISO-8601 UTC timestamp.

    Output examples: "12 minutes", "3 hours", "2 days". Used by
    `cmd_lifecycle_status` to mirror the spec's "(12 minutes)" suffix
    next to the `since:` line.
    """
    try:
        ts = datetime.fromisoformat(ts_iso)
    except (TypeError, ValueError):
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    delta = moment - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def cmd_lifecycle_force_unlock(args: argparse.Namespace) -> int:
    """Phase 10.6 Plan 10.6-01 Task 1.2: clear ``~/.iai-mcp/.locked``.

    Operator-facing recovery path for a stale lockfile that the
    daemon's own dead-PID takeover did not clear (e.g. cross-host
    iCloud/NFS sync where the user wants to wipe the foreign
    hostname BEFORE booting a new daemon, or a corrupt schema
    bump that the operator wants to inspect).

    Output: prints the prior payload (PID + hostname + started_at)
    so the operator can confirm what was cleared. ``--yes`` skips
    the interactive [y/N] prompt; tests pass ``--yes`` to avoid
    blocking on input().

    Exit codes:
      0 -- file cleared (or absent already, which is also "clear")
      1 -- user declined the prompt
    """
    from iai_mcp.lifecycle_lock import DEFAULT_LOCK_PATH, LifecycleLock

    # Resolve the lock-path. Tests inject ``args.lock_path`` to point
    # at a tmp file; production callers fall through to the default.
    lock_path = getattr(args, "lock_path", None)
    if lock_path is not None:
        lock = LifecycleLock(Path(lock_path))
    else:
        lock = LifecycleLock(DEFAULT_LOCK_PATH)

    existing = lock.read()
    if existing is None:
        print("No lockfile present; nothing to unlock.")
        return 0

    # Diagnostic surface so the operator can verify what they are clearing.
    print(
        f"Existing lockfile: pid={existing['pid']} "
        f"hostname={existing['hostname']} "
        f"started_at={existing['started_at']}"
    )

    yes = bool(getattr(args, "yes", False))
    if not yes:
        try:
            response = input(
                "Force unlock and remove the lockfile? [y/N]: "
            )
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Force-unlock cancelled.", file=sys.stderr)
            return 1

    previous = lock.force_unlock()
    if previous is None:
        # Race: file vanished between our read and unlink. Same exit
        # status -- the desired end state ("no lockfile") is reached.
        print("Lockfile already removed by another process.")
        return 0
    print("Lockfile removed.")
    return 0


def cmd_lifecycle_status(args: argparse.Namespace) -> int:
    """print formatted snapshot of `lifecycle_state.json`.

    Returns 0 unless the
    state file is unreadable in a way that bypasses the self-heal
    path (rare; load_state recovers from missing/corrupt files by
    returning a fresh default WAKE record).
    """
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, load_state

    record = load_state(LIFECYCLE_STATE_PATH)
    print(f"state: {record['current_state']}")
    print(
        f"since: {record['since_ts']} "
        f"({_format_relative(record['since_ts'])})"
    )
    print(f"last_activity: {record['last_activity_ts']}")
    print(f"wrapper_event_seq: {record['wrapper_event_seq']}")

    progress = record.get("sleep_cycle_progress")
    if progress is None:
        print("sleep_cycle_progress: none")
    else:
        step = progress.get("last_completed_step", 0)
        attempt = progress.get("attempt", 0)
        last_error = progress.get("last_error") or "none"
        started_at = progress.get("started_at", "?")
        print(
            f"sleep_cycle_progress: step={step} attempt={attempt} "
            f"last_error={last_error} started_at={started_at}"
        )

    quarantine = record.get("quarantine")
    if quarantine is None:
        print("quarantine: none")
    else:
        print(
            f"quarantine: until={quarantine['until_ts']} "
            f"reason={quarantine['reason']} since={quarantine['since_ts']}"
        )

    shadow = record.get("shadow_run", True)
    if shadow:
        print(
            "shadow_run: true (legacy RSS-watchdog still owns shutdown "
            "-- until Phase 10.6)"
        )
    else:
        print("shadow_run: false")

    return 0


# ---------------------------------------------------------------------------
# Plan 10.3-01 Task 1.5 -- iai-mcp maintenance sleep-cycle
# ---------------------------------------------------------------------------
#
# CLI surface for the SleepPipeline. Two flags:
#   --force              Run even when quarantined (operator override).
#   --reset-quarantine   Clear quarantine first; then run normally.
#
# Output format: one line per
# step in `[N/5] step_name ... ok (Ms)` format, plus a final summary
# line. On quarantine without --force, exits non-zero with an
# informational message pointing at --force / --reset-quarantine.
# ---------------------------------------------------------------------------


def cmd_maintenance_sleep_cycle(args: argparse.Namespace) -> int:
    """Plan 10.3-01 Task 1.5: run the sleep pipeline once.

    Exit codes:
      0 — success (5/5 steps complete) OR auto-recovery succeeded
      1 — quarantined and --force not specified, OR a step failed
      2 — store could not be opened (rare; same convention as
          other maintenance subcommands)

    The pipeline is invoked synchronously and prints a step-by-step
    progress trail. Output is plain text (NOT JSON) so the operator can
    follow along in a terminal; structured event-log entries cover
    machine-readable telemetry needs.

    No daemon-stopped pre-flight: unlike `compact-records`, the sleep
    pipeline calls Lance optimize on a 1-day retention window (NOT
    retention=0d for steps 1-4), so coexistence with the daemon's own
    `optimize_lance_storage` periodic call is safe (LanceDB MVCC).
    Step 5 (compact_records) does use retention=0d but the pipeline
    runs CLI-only in — daemon coexistence is the Phase
    10.4/10.5 wiring concern.
    """
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
    from iai_mcp.sleep_pipeline import SleepPipeline, SleepStep
    from iai_mcp.store import MemoryStore

    # Resolve store path the same way other maintenance commands do.
    if getattr(args, "store_path", None) is not None:
        store_path = Path(args.store_path).expanduser()
    else:
        store_path = Path.home() / ".iai-mcp"

    try:
        store = MemoryStore(path=store_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: could not open MemoryStore at {store_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=LIFECYCLE_STATE_PATH,
        event_log=LifecycleEventLog(),
    )

    reset_quarantine = bool(getattr(args, "reset_quarantine", False))
    force = bool(getattr(args, "force", False))

    if reset_quarantine:
        if pipeline.is_quarantined():
            pipeline.reset_quarantine()
            print("Quarantine cleared.")
        else:
            print("Quarantine not active; --reset-quarantine had no effect.")

    # Quarantine gate (when --force is NOT passed).
    if pipeline.is_quarantined() and not force:
        from iai_mcp.lifecycle_state import load_state

        record = load_state(LIFECYCLE_STATE_PATH)
        quarantine = record.get("quarantine") or {}
        until_ts = quarantine.get("until_ts", "?")
        reason = quarantine.get("reason", "unknown")
        print(
            f"Sleep cycle quarantined until {until_ts}.",
            file=sys.stderr,
        )
        print(f"Reason: {reason}", file=sys.stderr)
        print(
            "Use --force to override OR --reset-quarantine to clear.",
            file=sys.stderr,
        )
        return 1

    # Step-name -> 1..5 index for the progress prefix.
    step_index = {
        SleepStep.SCHEMA_MINE: 1,
        SleepStep.KNOB_TUNE: 2,
        SleepStep.DREAM_DECAY: 3,
        SleepStep.OPTIMIZE_LANCE: 4,
        SleepStep.COMPACT_RECORDS: 5,
    }

    print("Sleep cycle started.")
    # Run via force_run() if --force was passed, else run().
    runner = pipeline.force_run if force else pipeline.run
    result = runner()

    # Render per-step lines. Note: result["completed_steps"] is the list
    # of steps THIS invocation completed (resumes do NOT replay prior
    # steps), so the prefix is the index of the SleepStep, not its
    # position in completed_steps.
    for step in result["completed_steps"]:
        idx = step_index.get(step, "?")
        # We do not have per-step durations from the result dict (only
        # `duration_sec` for the whole run). Print "ok" without timing
        # to keep the line shape stable; precise per-step timings live
        # in the lifecycle event log under sleep_step_completed.
        print(f"[{idx}/5] {step.name.lower()} ... ok")

    duration = result.get("duration_sec", 0.0)
    failed = result.get("failed_step")
    interrupted = result.get("interrupted", False)
    quarantine_triggered = result.get("quarantine_triggered", False)

    if failed is not None:
        idx = step_index.get(failed, "?")
        err = result.get("error") or "unknown"
        print(
            f"[{idx}/5] {failed.name.lower()} ... FAILED: {err}",
            file=sys.stderr,
        )
        if quarantine_triggered:
            print(
                "Sleep cycle quarantined for 24h after 3rd consecutive "
                "failure of this step. Use --reset-quarantine to clear.",
                file=sys.stderr,
            )
        else:
            print(
                "Sleep cycle aborted; rerun to retry from this step.",
                file=sys.stderr,
            )
        return 1

    if interrupted:
        print(
            f"Sleep cycle deferred (bounded interrupt; "
            f"{duration:.1f}s elapsed). Resume on next invocation.",
        )
        return 0

    print(f"Sleep cycle complete ({duration:.1f}s total).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iai-mcp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="show LLM health status")
    h.set_defaults(func=cmd_health)

    m = sub.add_parser(
        "migrate",
        help=(
            "migrate records: 1->2 or 2->3 (Plan 02-08 encryption); "
            "OR --resume / --rollback a partial reembed migration (Plan 07.11-03)"
        ),
    )
    m.add_argument("--from", dest="from_", type=int, default=1)
    m.add_argument("--to", type=int, default=2)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--verbose", "-v", action="store_true")
    # Plan 07.11-03 / crash-safe-reembed entry points. Additive flags;
    # --from/--to dispatch is unchanged when neither --resume nor --rollback
    # is passed.
    m.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial reembed migration from migration_progress.json checkpoint.",
    )
    m.add_argument(
        "--rollback",
        action="store_true",
        help=(
            "Roll back a partial reembed migration: drop records_v_new and "
            "(if needed) restore records from records_old_<ts>."
        ),
    )
    m.set_defaults(func=cmd_migrate)

    # crypto subcommand.
    c = sub.add_parser(
        "crypto",
        help="encryption key management (Plan 02-08, SEC-ENCRYPTION-AT-REST)",
    )
    crypto_sub = c.add_subparsers(dest="crypto_cmd", required=True)

    cs = crypto_sub.add_parser(
        "status",
        help=(
            "(Plan 07.10) show file-backend key status: backend, path, "
            "mode, uid, length validation, passphrase-fallback flag"
        ),
    )
    cs.add_argument("--user-id", dest="user_id", default="default")
    cs.set_defaults(func=cmd_crypto_status)

    cr = crypto_sub.add_parser(
        "rotate", help="rotate encryption key + re-encrypt all records"
    )
    cr.add_argument("--user-id", dest="user_id", default="default")
    cr.set_defaults(func=cmd_crypto_rotate)

    # W3: migrate-to-file + init subcommands.
    mtf = crypto_sub.add_parser(
        "migrate-to-file",
        help=(
            "(Plan 07.10) one-time: read existing key from macOS Keychain "
            "and write to .crypto.key file (interactive Terminal only)"
        ),
    )
    mtf.add_argument("--user-id", dest="user_id", default="default")
    mtf_group = mtf.add_mutually_exclusive_group()
    mtf_group.add_argument(
        "--keep-keychain",
        dest="keep_keychain",
        action="store_true",
        default=True,
        help="leave the existing macOS Keychain entry in place (default)",
    )
    mtf_group.add_argument(
        "--delete-keychain",
        dest="keep_keychain",
        action="store_false",
        help="delete the macOS Keychain entry after successful migration",
    )
    mtf.set_defaults(func=cmd_crypto_migrate_to_file)

    ci = crypto_sub.add_parser(
        "init",
        help=(
            "(Plan 07.10) generate a fresh .crypto.key file "
            "(fresh installs only — refuses if file exists)"
        ),
    )
    ci.add_argument("--user-id", dest="user_id", default="default")
    ci.set_defaults(func=cmd_crypto_init)

    rwpk = crypto_sub.add_parser(
        "recover-with-prior-key",
        help=(
            "stage all records, decrypt literal/provenance/gain with current "
            "then prior key, re-encrypt under current key; atomic Lance swap"
        ),
    )
    rwpk.add_argument(
        "--prior-key-file",
        type=Path,
        required=True,
        help="path to exactly 32 raw AES key bytes (same format as .crypto.key)",
    )
    rwpk.add_argument("--user-id", dest="user_id", default="default")
    rwpk.add_argument(
        "--dry-run",
        action="store_true",
        help="report rows that need the prior key without mutating tables",
    )
    rwpk.set_defaults(func=cmd_crypto_recover_prior_key)

    cred = crypto_sub.add_parser(
        "redact-undecryptable",
        help=(
            "replace literal_surface that fails AES-GCM decrypt with a redacted "
            "marker (preserves embeddings, edges, metadata)"
        ),
    )
    cred.add_argument("--user-id", dest="user_id", default="default")
    cred.set_defaults(func=cmd_crypto_redact_undecryptable)

    t = sub.add_parser(
        "trajectory",
        help="aggregate M1..M6 trajectory events (D-32, OPS-08)",
    )
    t.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    t.set_defaults(func=cmd_trajectory)

    # CONN-07: live topology snapshot (sigma + C + L + community + rich-club).
    topo = sub.add_parser(
        "topology",
        help=(
            "live small-world topology snapshot (Plan 03-02 CONN-07): "
            "C, L, sigma, communities, rich-club ratio, N, regime"
        ),
    )
    topo.set_defaults(func=cmd_topology)

    # Plan 06 WRITE-side ambient: capture a Claude Code JSONL transcript
    # into the store (called by ~/.claude/hooks/iai-mcp-session-capture.sh).
    cap = sub.add_parser(
        "capture-transcript",
        help=(
            "batch-capture a Claude Code JSONL transcript into episodic tier. "
            "Used by the Stop hook for ambient WRITE-side observation capture."
        ),
    )
    cap.add_argument("transcript_path", help="path to the Claude Code JSONL transcript file")
    cap.add_argument("--session-id", default="-", help="session id for provenance")
    cap.add_argument("--max-turns", type=int, default=200,
                     help="cap on turns to scan (default 200; older turns skipped)")
    cap.add_argument(
        "--no-spawn",
        action="store_true",
        default=False,
        help=(
            "Hook-only mode: try connect with 250ms timeout. On miss, write "
            "transcript to ~/.iai-mcp/.deferred-captures/ and exit 0 within 2s. "
            "NEVER spawn daemon. Used by ~/.claude/hooks/iai-mcp-session-capture.sh "
            "to eliminate spawn vector (Phase 7.1 R3 / D7.1-04)."
        ),
    )
    cap.set_defaults(func=cmd_capture_transcript)

    # Plan 06 ambient-capture installer: drops the Stop hook into Claude Code
    # or Codex hook config. Makes a fresh
    # install of iai-mcp on another machine a two-step flow:
    #   pip install -e ".[dev,compress]"
    #   iai-mcp capture-hooks install
    ch = sub.add_parser(
        "capture-hooks",
        help="install/uninstall/status Stop hooks for ambient session capture",
    )
    ch_sub = ch.add_subparsers(dest="capture_hooks_cmd", required=True)
    for name, helptext, func in (
        ("install", "copy and register the Stop hook", cmd_capture_hooks_install),
        ("uninstall", "remove the Stop hook and config entry", cmd_capture_hooks_uninstall),
        ("status", "show whether the Stop hook is installed and active", cmd_capture_hooks_status),
    ):
        hook_cmd = ch_sub.add_parser(name, help=helptext)
        hook_cmd.add_argument(
            "--target",
            choices=["claude", "codex", "all"],
            default="claude",
            help="hook target to manage (default: claude)",
        )
        hook_cmd.set_defaults(func=func)

    # audit subcommand + sub-subcommands.
    a = sub.add_parser(
        "audit",
        help="identity + shield audit log (OPS-07, D-30)",
    )
    a.add_argument(
        "--since",
        type=int,
        default=None,
        help="weeks back to include (default: all history)",
    )
    a.add_argument(
        "--severity",
        choices=["info", "warning", "critical"],
        default=None,
        help="filter by severity",
    )
    audit_sub = a.add_subparsers(dest="audit_sub")
    for name, helptext in (
        ("shield", "shield-only audit (match counts redacted)"),
        ("drift", "detect M4 drift anomaly and surface it"),
        ("identity", "s5_* identity events only"),
    ):
        sp = audit_sub.add_parser(name, help=helptext)
        sp.add_argument("--since", type=int, default=None)
        sp.add_argument(
            "--severity",
            choices=["info", "warning", "critical"],
            default=None,
        )
    a.set_defaults(func=cmd_audit)

    # daemon subcommand group (DAEMON-10 + DAEMON-12).
    d = sub.add_parser(
        "daemon",
        help="sleep daemon: install/uninstall/start/stop/status/logs/...",
    )
    daemon_sub = d.add_subparsers(dest="daemon_cmd", required=True)

    di = daemon_sub.add_parser(
        "install",
        help=(
            "install launchd plist (macOS) / systemd user unit (Linux); "
            "first-run consent banner per unless --yes"
        ),
    )
    di.add_argument(
        "--dry-run",
        action="store_true",
        help="print plist/unit contents without writing or invoking launchctl/systemctl",
    )
    di.add_argument(
        "--yes", "-y",
        action="store_true",
        help="skip the consent banner (records --yes audit-trail still)",
    )
    di.set_defaults(func=cmd_daemon_install)

    du = daemon_sub.add_parser(
        "uninstall",
        help="C4 clean uninstall: remove plist/unit + 3 state files",
    )
    du.add_argument("--yes", "-y", action="store_true")
    du.set_defaults(func=cmd_daemon_uninstall)

    daemon_sub.add_parser(
        "start", help="launchctl kickstart / systemctl --user start",
    ).set_defaults(func=cmd_daemon_start)

    daemon_sub.add_parser(
        "stop", help="launchctl kill SIGTERM / systemctl --user stop",
    ).set_defaults(func=cmd_daemon_stop)

    daemon_sub.add_parser(
        "status",
        help=(
            "socket round-trip: print daemon FSM state, uptime, version "
            "(warns on version skew vs installed package)"
        ),
    ).set_defaults(func=cmd_daemon_status)

    dlogs = daemon_sub.add_parser(
        "logs",
        help="tail daemon log file (macOS Library/Logs) or journalctl (Linux)",
    )
    dlogs.add_argument("-f", "--follow", action="store_true")
    dlogs.add_argument("-n", "--lines", type=int, default=50)
    dlogs.set_defaults(func=cmd_daemon_logs)

    daemon_sub.add_parser(
        "force-rem",
        help="D-18 cooperative force: trigger one REM cycle out-of-schedule",
    ).set_defaults(func=cmd_daemon_force_rem)

    dpause = daemon_sub.add_parser(
        "pause", help="pause daemon scheduler for N seconds",
    )
    dpause.add_argument("seconds", type=int)
    dpause.set_defaults(func=cmd_daemon_pause)

    daemon_sub.add_parser(
        "resume", help="resume daemon scheduler after a pause",
    ).set_defaults(func=cmd_daemon_resume)

    dconf = daemon_sub.add_parser(
        "configure",
        help=(
            "D-22 per-setting override: set-budget / set-cycle-count / "
            "set-quiet-window / disable-claude / enable-claude"
        ),
    )
    dconf.add_argument(
        "key",
        choices=[
            "set-budget",
            "set-cycle-count",
            "set-quiet-window",
            "disable-claude",
            "enable-claude",
        ],
    )
    dconf.add_argument("value", nargs="?", default=None)
    dconf.set_defaults(func=cmd_daemon_configure)

    # R8: schema-cleanup top-level subcommand. NOT under
    # `iai-mcp migrate ...` — `migrate` namespace is reserved for v-bump
    # schema migrations (v3 -> v4 etc); this is a maintenance op.
    sc = sub.add_parser(
        "schema-cleanup",
        help=(
            "soft-delete duplicate schema records (Plan 06-05 R8). Default "
            "mode is --dry-run; --apply snapshots the LanceDB dir and "
            "performs the cleanup. Idempotent (re-running is a no-op)."
        ),
    )
    sc_mode = sc.add_mutually_exclusive_group()
    sc_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print the cleanup diff without mutating the store",
    )
    sc_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="snapshot the LanceDB dir + soft-delete duplicates",
    )
    sc.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; LanceDB tables "
            "live at <store-path>/lancedb)"
        ),
    )
    sc.set_defaults(func=cmd_schema_cleanup)

    # Plan 07.14-01 top-level `maintenance` subcommand for one-shot
    # Lance compaction. Same placement precedent as `schema-cleanup` and
    # `doctor` — top-level discoverability matters for first-touch ops.
    mtn = sub.add_parser(
        "maintenance",
        help=(
            "one-shot maintenance ops (Plan 07.14-01). Currently: "
            "compact-records (drain LanceDB version-manifest pile)."
        ),
    )
    mtn_sub = mtn.add_subparsers(dest="maintenance_cmd", required=True)
    mtn_compact = mtn_sub.add_parser(
        "compact-records",
        help=(
            "compact records.lance via optimize(cleanup_older_than=0d). "
            "DAEMON MUST BE STOPPED. Default --dry-run; --apply requires "
            "--yes for non-tty."
        ),
    )
    mtn_compact_mode = mtn_compact.add_mutually_exclusive_group()
    mtn_compact_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="(default) print metrics-only JSON; do NOT call optimize",
    )
    mtn_compact_mode.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="run optimize(cleanup_older_than=0d) on records/edges/events",
    )
    mtn_compact.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip the interactive 'y/N' prompt",
    )
    mtn_compact.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; LanceDB tables "
            "live at <store-path>/lancedb). Mirrors `schema-cleanup` flag."
        ),
    )
    mtn_compact.set_defaults(func=cmd_maintenance_compact_records)

    # Plan 10.3-01 Task 1.5: maintenance sleep-cycle subcommand.
    # Runs the 5-step SleepPipeline (schema_mine -> knob_tune ->
    # dream_decay -> optimize_lance -> compact_records) once, with
    # quarantine gating + bounded-deferral support.
    mtn_sleep = mtn_sub.add_parser(
        "sleep-cycle",
        help=(
            "(Phase 10.3) run the 5-step sleep pipeline once: "
            "schema_mine, knob_tune, dream_decay, optimize_lance, "
            "compact_records. 3-strike auto-quarantine; use --force "
            "to override, --reset-quarantine to clear."
        ),
    )
    mtn_sleep.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="run even if quarantined (operator override)",
    )
    mtn_sleep.add_argument(
        "--reset-quarantine",
        dest="reset_quarantine",
        action="store_true",
        default=False,
        help="clear quarantine state before running",
    )
    mtn_sleep.add_argument(
        "--store-path",
        dest="store_path",
        default=None,
        help=(
            "IAI root directory (defaults to ~/.iai-mcp; LanceDB tables "
            "live at <store-path>/lancedb)"
        ),
    )
    mtn_sleep.set_defaults(func=cmd_maintenance_sleep_cycle)

    # R9: doctor top-level subcommand (D7-10 — same placement
    # precedent as `iai-mcp schema-cleanup`, NOT nested under
    # `iai-mcp daemon`). First-touch recovery tool — top-level
    # discoverability matters when the user sees `daemon_unreachable`.
    doc = sub.add_parser(
        "doctor",
        help=(
            "Diagnose daemon health (7 checks; (g) duplicate-binder detection "
            "added in R6). With --apply, attempt safe repairs "
            "(unlink stale socket, kill duplicate binders, cleanup orphans, "
            "respawn daemon). With --apply --yes, skip confirmations. "
            "Exit 0=all green, 1=any FAIL, 2=--apply tried but FAIL persists."
        ),
    )
    # --apply is additive (NOT a mode switch like dry-run/apply on
    # schema-cleanup), so no mutually-exclusive group; --yes is a sub-modifier
    # that cmd_doctor checks for warning-and-ignore semantics if used alone.
    doc.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="attempt safe repairs after diagnosis; prompts before each destructive action",
    )
    doc.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="(use with --apply) skip confirmation prompts; equivalent to typing 'y' to all",
    )
    doc.set_defaults(func=cmd_doctor)

    # -- iai-mcp lifecycle status. Top-level placement
    # follows the `doctor` / `maintenance` precedent: first-touch
    # observability matters and the user types it directly.
    lc = sub.add_parser(
        "lifecycle",
        help=(
            "(Plan 10.1) inspect lifecycle state machine "
            "(WAKE/DROWSY/SLEEP/HIBERNATION). Currently: status."
        ),
    )
    lc_sub = lc.add_subparsers(dest="lifecycle_cmd", required=True)
    lc_status = lc_sub.add_parser(
        "status",
        help=(
            "print current lifecycle state, since-ts, last activity, "
            "wrapper event seq, sleep-cycle progress, quarantine, and "
            "shadow_run flag"
        ),
    )
    lc_status.set_defaults(func=cmd_lifecycle_status)

    # Plan 10.6-01 Task 1.2: force-unlock recovery for
    # ~/.iai-mcp/.locked. Operator path; daemon-side dead-PID takeover
    # handles the common case automatically.
    lc_unlock = lc_sub.add_parser(
        "force-unlock",
        help=(
            "(Plan 10.6) clear a stale ~/.iai-mcp/.locked lockfile and "
            "print the prior PID / hostname / started_at"
        ),
    )
    lc_unlock.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive [y/N] prompt",
    )
    lc_unlock.set_defaults(func=cmd_lifecycle_force_unlock)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
