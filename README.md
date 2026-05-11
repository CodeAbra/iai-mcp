<p align="center">
  <img src="logo.png" alt="IAI-MCP" width="600">
</p>


<h3 align="center">The best-benchmarked open-source memory system for AI coding assistants.</h3>
<p align="center">Every claim ships with the harness that proves it. Run the benchmarks yourself.</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python 3.11 | 3.12">
  <img src="https://img.shields.io/badge/platform-macOS-lightgrey.svg" alt="Platform: macOS">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/verbatim%20recall-%E2%89%A599%25%20at%2010k-brightgreen.svg" alt="Verbatim recall >= 99%">
  <img src="https://img.shields.io/badge/p95%20latency-%3C100ms-brightgreen.svg" alt="p95 < 100ms">
  <img src="https://img.shields.io/badge/encryption-AES--256--GCM-green.svg" alt="AES-256-GCM">
  <img src="https://img.shields.io/badge/local--only-no%20telemetry-green.svg" alt="Local only, no telemetry">
  <img src="https://img.shields.io/badge/MCP-compatible-purple.svg" alt="MCP compatible">
  <a href="https://glama.ai/mcp/servers/CodeAbra/iai-mcp"><img src="https://glama.ai/mcp/servers/CodeAbra/iai-mcp/badges/score.svg" alt="Glama MCP score"></a>
</p>

---

# iai-mcp

*Independent Autistic Intelligence — a local memory layer for Claude (and other MCP-compatible assistants).*

## Table of contents

- [What it is](#what-it-is)
- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Doctor](#doctor)
- [Notes for AI assistants](#notes-for-ai-assistants-helping-with-installation)
- [Status and limitations](#status-and-limitations)
- [Compatibility](#compatibility)
- [About the name](#about-the-name)
- [Authors](#authors)
- [License](#license)
- [Contributing](#contributing)

---

## What it is

A local server that speaks the [MCP protocol](https://modelcontextprotocol.io) and gives Claude, and any other MCP-compatible assistant, a long-term memory. It captures every turn of every session verbatim, organizes those captures over time into a personal map of who you are, and serves a small slice of relevant memory back at the start of each new conversation. You never have to say *"remember this"* or *"what did we say last time?"*.

I built this for myself. It worked. I've been running it daily for months, and now I'm sharing it. The benchmarks were mostly for my own curiosity. I wanted to know if it actually works or if I'd just gotten used to it.

---

## Quick start

### Prerequisites

- macOS (Apple Silicon tested)
- Python 3.11 or 3.12
- Node.js 18+
- [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) or Codex CLI as the MCP host
- ~500 MB free disk

Windows and Linux not supported yet but I'm working on it.

### Install

```bash
git clone https://github.com/CodeAbra/iai-mcp.git
cd iai-mcp
bash scripts/install.sh
```

The installer creates a Python venv, installs dependencies (LanceDB, sentence-transformers, torch-hd, NetworkX, igraph), builds the TypeScript MCP wrapper, pre-downloads the default embedding model (~130 MB), symlinks the CLI to `~/.local/bin/iai-mcp`, and on macOS registers the daemon with launchd.

Make sure `~/.local/bin` is on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc or ~/.bashrc
iai-mcp --version
```

### Install the capture hook

This is what makes capture ambient. Without it you'd have to save memories by hand.

Claude Code is the default target:

```bash
iai-mcp capture-hooks install
```

For Codex:

```bash
iai-mcp capture-hooks install --target codex
```

To install both:

```bash
iai-mcp capture-hooks install --target all
```

Check status with:

```bash
iai-mcp capture-hooks status --target all
```

Manual Claude Code setup is equivalent to:

```bash
mkdir -p ~/.claude/hooks
cp deploy/hooks/iai-mcp-session-capture.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/iai-mcp-session-capture.sh
```

Register in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/iai-mcp-session-capture.sh"
          }
        ]
      }
    ]
  }
}
```

### Connect your MCP host

Claude Code:

```bash
claude mcp add iai-mcp -- node "$(pwd)/mcp-wrapper/dist/index.js"
```

Or edit `~/.claude.json` directly:

```json
{
  "mcpServers": {
    "iai-mcp": {
      "command": "node",
      "args": ["/absolute/path/to/iai-mcp/mcp-wrapper/dist/index.js"]
    }
  }
}
```

Use the absolute path. `~` and `$HOME` won't expand here.

For Claude Desktop (untested), edit `~/Library/Application Support/Claude/claude_desktop_config.json`.

Codex CLI:

```toml
[mcp_servers.iai-mcp]
command = "node"
args = ["/absolute/path/to/iai-mcp/mcp-wrapper/dist/index.js"]

[mcp_servers.iai-mcp.env]
IAI_MCP_PYTHON = "/absolute/path/to/iai-mcp/.venv/bin/python"
IAI_MCP_STORE = "/Users/you/.iai-mcp"
TRANSFORMERS_VERBOSITY = "error"
TOKENIZERS_PARALLELISM = "false"
```

Codex hooks are stable in current Codex CLI builds. If hooks are disabled by
local policy or an older install, enable `[features].hooks = true` in
`~/.codex/config.toml`.

### Verify

```bash
iai-mcp doctor
iai-mcp daemon status
```

Restart Claude Code. Start a session, do some work, exit. Then:

```bash
tail ~/.iai-mcp/logs/capture-$(date -u +%Y-%m-%d).log
```

You should see a `rc=0` line. That's your first memory.

---

## Usage

You do not call `iai-mcp` directly during a session. Once it's connected:

Capture is automatic. Every turn, yours and the assistant's, is recorded verbatim with timestamps and session metadata. You don't say *"remember this."*

Recall is automatic. When a new session starts, the daemon assembles a small relevant slice of your history and injects it into the conversation prefix. You don't say *"what did we say."*

Consolidation runs idle. Between sessions, the daemon merges duplicates, strengthens recall pathways for things retrieved often, and prunes weak edges. The system gets quietly better at remembering you over time.

After a few weeks of regular use the difference becomes noticeable. The assistant stops asking the same orientation questions, references things you mentioned in passing, and adapts to your style without being told.

---

## How it works

The daemon is a Python process that runs in the background. Your MCP client connects to it via a Unix socket. No network exposure.

Memory is stored in three tiers:

*Episodic* is verbatim, timestamped fragments of what was said. Write-once, never overwritten or rewritten.

*Semantic* is summaries induced from clusters of related episodes during idle-time consolidation.

*Procedural* is a small set of stable parameters about you, learned over time: preferences, style cues, recurring patterns. Eleven sealed knobs that shift based on what works.

A background pass runs periodically (sleep cycles): it clusters episodes, builds semantic summaries, decays old unreinforced connections, and reinforces frequently co-retrieved paths. Things you haven't revisited fade naturally. There's an optional "insight of the day" step that makes one Anthropic API call, but it's off by default.

Recall combines three signals: semantic similarity, graph-link strength, and recency. All ranked together.

All records are encrypted at rest with AES-256-GCM. The key lives in `~/.iai-mcp/.key` (mode 0600). Back it up. Lose the key, lose the memories.

Everything lives at `~/.iai-mcp/`. Embeddings are computed locally with `bge-small-en-v1.5`. The only data that leaves the machine is your normal conversation with whatever LLM API your client uses.

```
Claude Code  <--MCP-stdio-->  TypeScript wrapper  <--UNIX socket-->  Python daemon  <-->  LanceDB
```

---

## Benchmarks

I made these because I wanted honest numbers. Every harness ships in `bench/`. Run them on your machine, get your own results.

| Metric | Target | Measured |
|---|---|---|
| Verbatim recall (byte-exact) | >=99% | >=99% at N=10k |
| Recall p95 latency | <100 ms | <100 ms at N=10k |
| RAM at steady state | <=300 MB | ~150-300 MB |
| Session-start tokens (warm cache) | <=3,000 | <=3,000 |
| Session-start tokens (cold) | <=8,000 | <=8,000 |

```bash
python -m bench.verbatim                     # verbatim fidelity
python -m bench.neural_map                   # recall latency
python -m bench.memory_footprint             # RAM usage
python -m bench.tokens                       # session-start cost
python -m bench.total_session_cost           # full 10-turn cost
python -m bench.trajectory                   # 30-session corpus
python -m bench.contradiction_longitudinal   # falsifiability
python -m bench.longmemeval_blind            # LongMemEval-S blind run
```

The LongMemEval-S run is blind on purpose. No dataset-specific tuning, no hyperparameter sweep. The numbers are what they are.

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `IAI_MCP_STORE` | `~/.iai-mcp/` | Data directory |
| `IAI_MCP_EMBED_MODEL` | `bge-small-en-v1.5` | Embedding model. `bge-m3` for multilingual at ~3x size. |

Switching embedders requires re-embedding the store: `iai-mcp migrate reembed`.

---

## Doctor

`iai-mcp doctor` runs 14 checks against the daemon, the store, and the runtime state. Output is one line per check: PASS, WARN, or FAIL.

```bash
iai-mcp doctor
```

What it checks:

| # | Check | What it means |
|---|---|---|
| a | Daemon alive | Is the daemon process running? |
| b | Socket fresh | Can the UNIX socket accept a connection? |
| c | Lock healthy | Is the process lock held correctly? |
| d | No orphan core | No leftover stdio core process without a daemon |
| e | State file valid | `.daemon-state.json` parses and has expected fields |
| f | LanceDB readable | Can the records table be opened and queried? |
| g | No duplicate binders | Only one process is bound to the socket |
| h | Crypto file state | Encryption key exists, correct permissions (0600) |
| i | Lance versions count | LanceDB version manifests aren't piling up |
| j | Lifecycle current state | Current FSM state is valid |
| k | Lifecycle history 24h | Recent lifecycle transitions look sane |
| l | Sleep cycle status | Last sleep cycle completed or is running normally |
| m | Heartbeat scanner | Wrapper heartbeat files are fresh |
| n | HID idle source | Idle detection source is available |

14/14 PASS is healthy. 13/14 with check (b) failing during a sleep cycle is also normal (the socket is busy during consolidation). Multiple FAILs or a FAIL on (a) or (f) means something is actually wrong.

---

## Notes for AI assistants helping with installation

If you're an AI model helping a user install or debug iai-mcp, read this before diagnosing anything.

The daemon has four FSM states: `WAKE`, `TRANSITIONING`, `SLEEP`, `DREAMING`. Separately, the wrapper-layer lifecycle has four states: `WAKE`, `DROWSY`, `SLEEP`, `HIBERNATION`. These are two different state machines in two different files. Don't confuse them.

Things that look broken but are not:

- `last_tick_at` hasn't changed in hours. This is a sleep-cycle tick timestamp, not a process heartbeat. The daemon can be alive and healthy with a stale `last_tick_at`. Check `iai-mcp daemon status` instead.
- `fsm_state: DREAMING` for hours. Normal. The daemon is running a sleep cycle. Wake transitions are event-driven (a new session arriving wakes it), not time-driven.
- `status: deferred` in the capture log. This is the only path the Stop hook takes. It writes captures to `~/.iai-mcp/.deferred-captures/` and the daemon picks them up on its next tick. This is by design, not a failure.
- `iai-mcp doctor` shows 13/14 PASS with check (b) "socket unreachable." Expected when the daemon FSM is in SLEEP or DREAMING. The socket is intentionally unresponsive during consolidation passes. This is healthy.
- Cold start takes a few seconds. The daemon needs to initialize the LanceDB store and load the embedding model into memory. Don't kill it because it hasn't responded in 5 seconds.

Actual problems look like: `iai-mcp daemon status` returning `ok: False`, doctor showing multiple FAIL results, or the daemon process not existing at all (`pgrep -f iai_mcp.daemon` returns nothing).

When in doubt, run `iai-mcp doctor` and read what it says. The output is self-explanatory.

---

## Status and limitations

This is experimental. I built it for myself, it works on my machine, and I'm sharing it because it might be useful to you. No SLA, no support guarantee. Breaking changes are possible between versions. Pin a commit hash if you depend on stability.

Limitations worth knowing about:

- The default embedding model is English-only. The assistant translates to English on the way into memory. The opt-in `bge-m3` model removes this constraint at a cost of ~3x storage and slower indexing.
- No cross-machine sync. The data lives where the daemon runs. Backup is `cp -a ~/.iai-mcp/` somewhere safe.
- No GUI. Inspection happens through CLI subcommands (`iai-mcp doctor`, `iai-mcp daemon status`, `iai-mcp topology`).
- Cold start on a freshly booted machine takes a few seconds while the daemon initializes caches.
- Recall quality on the first ~10 sessions is mediocre. The system needs material to consolidate before it gets useful.

---

## Compatibility

Claude Code is the primary host, validated in daily use.

Claude Desktop should work (uses `claude_desktop_config.json` instead of `~/.claude.json`) but hasn't been tested end to end.

Codex CLI supports the MCP wrapper and ambient capture through a `Stop` hook.

Other MCP-over-stdio hosts speak the same protocol and should work in principle. Not tested.

If you get it running on something else, open an issue or PR.

---

## About the name

*IAI* stands for Independent Autistic Intelligence.

- **Independent.** Fully local. The daemon runs on your machine, embeddings are computed locally, no telemetry, no cloud dependency. Your memory is your data and stays your data.
- **Autistic.** Describes the memory style, not a diagnosis or a metaphor. The memory is built around verbatim recall, attention to specific cues, and refusal to smooth rare events into typical ones. Most memory systems compress and summarize aggressively, aiming to give the assistant a *gist* of the past. This one preserves what was actually said and surfaces it on a precise cue. The trade-off is intentional: more storage and a stricter retrieval interface, in exchange for not losing details.
- **Intelligence.** Used in the systems sense, something that observes, adapts, and stays viable over time, not the marketing sense.

---

## Authors

By Areg Aramovich Noya, in collaboration with the team at [lcgc.dev](https://lcgc.dev).

I built this because I needed it. It works for me. If it works for you, take it.

## License

[MIT](LICENSE)

## Contributing

Issues and PRs welcome. If your change touches retrieval, capture, or consolidation, include bench re-runs.
