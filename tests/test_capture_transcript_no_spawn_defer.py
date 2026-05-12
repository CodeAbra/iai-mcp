"""Phase 7.5 acceptance — `iai-mcp capture-transcript --no-spawn` ALWAYS defers.

Closes the embedder cold-load amplification documented in SPEC 07.5: every
Stop-hook invocation (286/day on 2026-04-27) was loading bge-small-en-v1.5
in a brand-new Python subprocess on the daemon-reachable path. Forensic
evidence: stderr `Loading weights: 0%|...| 0/391 ...|██| 391/391` × 10 +
`leaked semaphore objects at shutdown` × 7.

Fix: `cmd_capture_transcript` `--no-spawn` branch in `src/iai_mcp/cli.py`
no longer probes the socket and no longer imports
`iai_mcp.capture.capture_transcript` / `iai_mcp.store.MemoryStore`. It
unconditionally calls `write_deferred_captures(...)` and prints
`{"status": "deferred", "path": "..."}`. The daemon's WAKE drain (Phase
7.1 R3 / Plan 07.1-06) consumes deferred files with the daemon's
already-loaded embedder.

Test matrix:
- Test 1: subprocess + reachable mock socket (real AF_UNIX listener) →
  status="deferred", stderr has ZERO `Loading weights` and ZERO
  `sentence_transformers` mentions. The reachable case used to inline-embed;
  now it must defer just like the unreachable case.
- Test 2: subprocess + unreachable socket (back-compat) → identical output.
  Locks down that the new always-defer path doesn't regress the existing
  unreachable behaviour.
- Test 3: subprocess + fresh interpreter introspects `sys.modules` AFTER the
  CLI handler runs end-to-end → asserts `iai_mcp.embed` and
  `sentence_transformers` are NOT loaded. Subprocess required because other
  pytest tests in the same session may pre-load `iai_mcp.embed`, which
  poisons in-process `sys.modules` checks.
- Test 4: in-process source-string scan of the modified function body →
  asserts the `--no-spawn` block contains zero `capture_transcript` /
  `MemoryStore` import statements. Cheap structural lockdown so the inline
  path can't be reintroduced without breaking a test (SPEC A1).

Test isolation:
- HOME=tmp_path so `Path.home()` resolves to a fresh dir; the user's
  real ~/.iai-mcp/.deferred-captures/ is never touched.
- IAI_DAEMON_SOCKET_PATH=/tmp/iai-no-spawn-defer-<pid>-<n>/d.sock so the
  reachable case binds a real listener and the unreachable case points to
  a non-existent path.
- Subprocess invocation: `[sys.executable, '-m', 'iai_mcp.cli', ...]` with
  PYTHONPATH set; we don't depend on the `iai-mcp` console script being on
  PATH (matches the test_capture_transcript_no_spawn.py pattern).
"""
from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# POSIX-only: subprocess + AF_UNIX socket; matches the existing module's gate.
pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX",
)


# ---------------------------------------------------------------------------
# Shared helpers (kept local to keep this module standalone — the canonical
# pattern lives in test_capture_transcript_no_spawn.py but cross-importing
# would couple two unrelated test modules).
# ---------------------------------------------------------------------------


def _isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Build env that isolates HOME + socket path to tmp_path.

    Returns (env_dict, deferred_dir, sock_path).

    `sock_path` is created and `deferred_dir` is the on-disk location where
    `write_deferred_captures` will land its JSONL when HOME is honored.
    """
    sock_dir = Path(f"/tmp/iai-no-spawn-defer-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    # Defense-in-depth: if the inline path is somehow exercised, force the
    # fail-backend so we don't hang on the real keychain prompt.
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-no-spawn-defer-pass"
    # Make the spawned python find iai_mcp without an editable install.
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    return env, iai_dir / ".deferred-captures", sock_path


def _make_transcript(tmp_path: Path) -> Path:
    """Write a 3-turn Claude Code-style JSONL transcript."""
    turns = [
        {"type": "user", "message": {"role": "user", "content": "hello phase 7 5"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "ack always defer"}},
        {"type": "user", "message": {"role": "user", "content": "third defer turn"}},
    ]
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(t) for t in turns) + "\n")
    return transcript_path


def _make_codex_transcript(tmp_path: Path) -> Path:
    turns = [
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "noisy injected instructions should not be captured",
                    }
                ],
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "codex event user turn"},
        },
        {
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "codex assistant turn"},
        },
    ]
    transcript_path = tmp_path / "codex-transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(t) for t in turns) + "\n")
    return transcript_path


def _run_no_spawn(env: dict[str, str], transcript_path: Path) -> subprocess.CompletedProcess:
    """Invoke `iai-mcp capture-transcript --no-spawn <transcript>` via
    `python -m iai_mcp.cli`. 5s wall-clock budget — comfortably above the 2s
    contract the implementation must meet.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "capture-transcript",
            "--no-spawn",
            "--session-id",
            "test-phase75",
            str(transcript_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _bind_listener(sock_path: Path) -> socket.socket:
    """Bind an AF_UNIX listener at `sock_path` so `_try_short_timeout_connect`
    would return True if the OLD code path were reached. Caller must close
    the returned socket and unlink the path; use try/finally."""
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.listen(1)
    return s


# ---------------------------------------------------------------------------
# Test 1: reachable mock socket — must STILL defer (not inline-insert).
# This is the load-bearing acceptance: the OLD behaviour on this
# branch was inline ingest with embedder cold-load. NEW behaviour: defer.
# ---------------------------------------------------------------------------


def test_no_spawn_reachable_defers_not_inserts(tmp_path):
    """Phase 7.5 R1: even with the daemon socket reachable, --no-spawn
    writes a deferred-captures JSONL and exits 0 with status="deferred"."""
    env, deferred_dir, sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    listener = _bind_listener(sock_path)
    try:
        proc = _run_no_spawn(env, transcript)
    finally:
        listener.close()
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"

    # Must be JSON-parsable AND have status="deferred" (NOT "inserted": N).
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", (
        f"reachable case must defer under Phase 7.5; got {payload!r}"
    )
    assert "path" in payload, payload
    assert "inserted" not in payload, (
        f"inline-ingest path must not run under --no-spawn; got {payload!r}"
    )

    # Empirical proof the embedder did NOT cold-load: stderr is clean.
    # `sentence_transformers` writes a tqdm progress bar containing
    # `Loading weights` when bge-small-en-v1.5 first loads.
    assert "Loading weights" not in proc.stderr, (
        f"embedder cold-loaded on reachable --no-spawn path (Phase 7.5 broken):\n"
        f"{proc.stderr}"
    )
    assert "sentence_transformers" not in proc.stderr, (
        f"sentence_transformers touched on reachable --no-spawn path:\n"
        f"{proc.stderr}"
    )

    # File-on-disk side-effect: deferred JSONL exists with v1 header.
    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferred file, got {files}"
    header = json.loads(files[0].read_text().splitlines()[0])
    assert header["version"] == 1
    assert header["session_id"] == "test-phase75"


# ---------------------------------------------------------------------------
# Test 2: unreachable socket — back-compat. Same output as Test 1.
# ---------------------------------------------------------------------------


def test_no_spawn_unreachable_still_defers(tmp_path):
    """Back-compat guard: --no-spawn with daemon UNREACHABLE behaves
    identically to the reachable case (both defer). Locks down that the
    new always-defer path doesn't regress existing behaviour."""
    env, deferred_dir, sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    # No listener bound; sock_path does not exist on disk.
    assert not sock_path.exists()

    proc = _run_no_spawn(env, transcript)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", payload
    assert "inserted" not in payload, payload

    # Same stderr cleanliness invariant.
    assert "Loading weights" not in proc.stderr, proc.stderr
    assert "sentence_transformers" not in proc.stderr, proc.stderr

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferred file, got {files}"


def test_no_spawn_extracts_codex_transcript_turns(tmp_path):
    env, deferred_dir, _sock_path = _isolated_env(tmp_path)
    transcript = _make_codex_transcript(tmp_path)

    proc = _run_no_spawn(env, transcript)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", payload

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferred file, got {files}"
    events = [json.loads(line) for line in files[0].read_text().splitlines()[1:]]
    assert [event["role"] for event in events] == ["user", "assistant"]
    assert [event["text"] for event in events] == [
        "codex event user turn",
        "codex assistant turn",
    ]


# ---------------------------------------------------------------------------
# Test 3: fresh subprocess introspects sys.modules to prove no embedder load.
# In-process is unreliable because pytest sessions pre-load iai_mcp.embed via
# other test modules (test_recall_cue_router, test_active_inference_gate,
# test_invariant_anchor_edges, test_schema_instance_of_edges).
# ---------------------------------------------------------------------------


def test_no_spawn_zero_embedder_imports_in_fresh_process(tmp_path):
    """Phase 7.5 R1 (import-isolation): in a brand-new Python interpreter,
    invoking the `--no-spawn` CLI handler end-to-end leaves
    `iai_mcp.embed` and `sentence_transformers` UNLOADED. Direct evidence
    the heavy-import path is severed."""
    env, deferred_dir, _sock_path = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    # Inline driver script: invoke main(), then dump the loaded module names
    # we care about as a single-line JSON.
    driver = (
        "import sys, json\n"
        "from iai_mcp.cli import main\n"
        "rc = main([\n"
        "  'capture-transcript', '--no-spawn',\n"
        "  '--session-id', 'test-phase75-fresh',\n"
        f"  {str(transcript)!r},\n"
        "])\n"
        "loaded = sorted(\n"
        "  k for k in sys.modules\n"
        "  if k == 'iai_mcp.embed' or k.startswith('iai_mcp.embed.')\n"
        "  or k == 'sentence_transformers' or k.startswith('sentence_transformers.')\n"
        "  or k == 'torch' or k.startswith('torch.')\n"
        "  or k == 'transformers' or k.startswith('transformers.')\n"
        ")\n"
        "print('IAIMCP75_DUMP=' + json.dumps({'rc': rc, 'loaded': loaded}))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", driver],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, f"driver failed: stderr={proc.stderr!r}"

    # Find the dump line; CLI may emit its own JSON to stdout first.
    dump_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("IAIMCP75_DUMP=")]
    assert len(dump_lines) == 1, f"expected 1 dump line, got {dump_lines!r}"
    dump = json.loads(dump_lines[0][len("IAIMCP75_DUMP=") :])

    assert dump["rc"] == 0, f"main() returned {dump['rc']}"

    loaded = set(dump["loaded"])
    # The load-bearing assertions: heavy embedder and ML deps NOT touched.
    forbidden = {m for m in loaded if (
        m == "iai_mcp.embed" or m.startswith("iai_mcp.embed.")
        or m == "sentence_transformers" or m.startswith("sentence_transformers.")
    )}
    assert not forbidden, (
        f"--no-spawn must not import embedder/ML deps; loaded: {sorted(forbidden)}"
    )

    # Side-effect: deferred file landed on disk in the fresh interpreter run.
    assert any(deferred_dir.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# Test 4: structural lockdown — the modified function body must not contain
# the reintroduced inline imports. Cheap, in-process, regression-proof
# (SPEC A1: "Verified by static grep on the modified function").
# ---------------------------------------------------------------------------


def test_no_spawn_branch_has_no_inline_imports():
    """Phase 7.5 A1 lockdown: the `if no_spawn:` block in
    `cmd_capture_transcript` contains zero imports of
    `iai_mcp.capture.capture_transcript` and `iai_mcp.store.MemoryStore`.
    Prevents quiet reintroduction of the inline-embed path."""
    cli_src = (REPO / "src" / "iai_mcp" / "cli.py").read_text()

    # Locate the function body.
    fn_match = re.search(
        r"^def cmd_capture_transcript\(.*?\n(.*?)^def ",
        cli_src,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert fn_match, "could not locate cmd_capture_transcript in cli.py"
    fn_body = fn_match.group(1)

    # Slice the `if no_spawn:` branch — everything between the `if no_spawn:`
    # line and the next un-indented (or 4-space indented) `# Default path`
    # marker. The default-mode path lives below that marker and IS allowed
    # to import capture_transcript + MemoryStore.
    no_spawn_match = re.search(
        r"^    if no_spawn:\n(.*?)^    # Default path",
        fn_body,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert no_spawn_match, (
        "could not isolate `if no_spawn:` block; layout drifted from fix"
    )
    no_spawn_block = no_spawn_match.group(1)

    # The branch must reference write_deferred_captures and nothing else
    # heavy.
    assert "write_deferred_captures" in no_spawn_block, (
        "no_spawn branch must call write_deferred_captures"
    )

    # Forbidden inline-ingest imports.
    assert "from iai_mcp.capture import capture_transcript" not in no_spawn_block, (
        "Phase 7.5 regression: capture_transcript reintroduced into "
        "--no-spawn branch (would trigger embedder cold-load on every "
        "Stop-hook fire)"
    )
    assert "from iai_mcp.store import MemoryStore" not in no_spawn_block, (
        "Phase 7.5 regression: MemoryStore reintroduced into --no-spawn "
        "branch"
    )

    # Defensive: no probe call either — the SPEC removes it from this branch.
    assert "_try_short_timeout_connect" not in no_spawn_block, (
        "socket probe must be gone from --no-spawn branch (the "
        "probe was the gate that selected the inline path)"
    )
