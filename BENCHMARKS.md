# IAI-MCP benchmarks

This document describes the 6 benchmark axes the project tracks, the targets the design was built against, and how to reproduce the numbers reported in the README.

The benches are **measurements, not a leaderboard.** No winners, no losers — they exist so the design choices in `src/iai_mcp/` can be checked against honest numbers, and so anyone reproducing the work has a fixed methodology to follow.

## Philosophy

- **Self-reference is primary.** Every bench runs IAI-MCP and reports a number. The number is judged against the project's own target (see table below), not against external tools.
- **Reference tools are context, not competition.** Where applicable, the same workload is also run through two reference memory tools so the absolute number has scale. Reference tools were designed for different goals and are not graded against IAI-MCP's targets.
- **Honest disclosure beats favorable framing.** If a number misses its target, the report says so. If a comparison favors a reference tool on an axis the reference was built for, the report says so.

## The 6 axes

| Axis | Module | What it measures | Project target (self-reference) |
|------|--------|------------------|----------------------------------|
| **M-01** | `bench/tokens.py` | Session-start billable token budget (fresh / warm p50 / warm p95) | ≤3000 tokens warm |
| **M-02** | `bench/neural_map.py` | `pipeline_recall` end-to-end latency at {100, 1k, 5k, 10k} store sizes | p95 < 100 ms at 10k records |
| **M-03** | `bench/memory_footprint.py` | Steady-state RSS after loading N=10k records + bge-m3 embedder + graph runtime | No target — honest disclosure of the design's RAM cost |
| **M-04** | `bench/verbatim.py` | Byte-exact recall of pinned records under noise (gap × noise_per_session) | ≥99% byte-exact across session gaps |
| **M-05** | `bench/trajectory.py` | M1..M6 internal metrics across a 30-session synthetic corpus (curiosity, precision, tokens, variance, context-repeat) | Curve moves in the predicted direction (M1 ↓, M2 ↑, M3 ↓, M4 ↓, M5 ↓, M6 ≥0.9 by ~S20) |
| **M-06** | `bench/multilingual_fidelity.py` | Verbatim recall accuracy per language (en / ru / ja / ar / de / fr / es / zh) | Parity across 8 languages — per-language number, no aggregate shortcut |
| **M-07** | `bench/total_session_cost.py` | Full 10-turn session cost in tokens (system + tool descriptions + tool-call payloads + tool-result bodies) | Reported as `wake_depth=minimal` / `standard` / `deep` |
| **M-08** | `bench/longmemeval_blind.py` | LongMemEval-S blind retrieval (R@5, R@10, retrieve vs full pipeline) | Report only — third-party fixed-corpus benchmark |

> M-03, M-06 and M-08 each have their own caveats: M-03 has no target (it's a disclosure axis), M-06 reports per-language without aggregating, and M-08 uses an external dataset and is treated as report-only because the fixture is not under project control.

## Reference tools (comparator context)

The bench harness supports comparing IAI-MCP against two reference memory tools where the axis makes sense:

- **mempalace** — verbatim-text local memory tool. Good reference on M-04 verbatim, M-05 cross-session, M-06 multilingual because it stores literal text.
- **claude-mem** — Anthropic's in-session memory. Good reference on M-01 tokens and M-02 latency. M-04 verbatim is not what claude-mem was built for, so that number is reported as curiosity-only.

Reference numbers are reported alongside IAI-MCP numbers in the bench output JSON. Both reference tools are run via small adapter scripts under `bench/adapters/` that emit JSON in the matching `bench/<axis>.py` schema. On machines where adapters are missing, the per-axis bench falls back to IAI-MCP-only and the report carries an honest "reference not measured" disclosure for that row.

## Running benchmarks

### Prerequisites

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install tiktoken              # for M-01 offline tokeniser mode
```

Confirm tests pass before running benches (regression guard):

```bash
pytest -q
```

### Per-axis invocation

Each bench module is a runnable Python script that emits a single JSON line to its `--output` file. Cheapest axis first:

```bash
# M-01 — token budget (~5 minutes)
python -m bench.tokens --output bench_out/m01.json

# M-02 — latency, scaled across store sizes (~30 minutes)
python -m bench.neural_map --sizes 100,1000,5000,10000 --output bench_out/m02.json

# M-03 — RAM footprint (~10 minutes)
python -m bench.memory_footprint --n-records 10000 --output bench_out/m03.json

# M-04 — verbatim, across three session gaps (~45 minutes)
for gap in 5 20 100; do
  python -m bench.verbatim --gap "$gap" --n-pinned 50 --output "bench_out/m04_gap${gap}.json"
done

# M-05 — cross-session curve over 30-session corpus (~2 hours)
python -m bench.trajectory --n-sessions 30 --output bench_out/m05.json

# M-06 — per-language verbatim (~30 minutes)
python -m bench.multilingual_fidelity \
  --languages en,ru,ja,ar,de,fr,es,zh \
  --output bench_out/m06.json

# M-07 — total session cost (10-turn script)
python -m bench.total_session_cost --wake-depth standard --output bench_out/m07.json

# M-08 — LongMemEval-S blind retrieval
# Requires IAI_MCP_CRYPTO_PASSPHRASE in env; pre-flight enforces this.
IAI_MCP_CRYPTO_PASSPHRASE=bench python -m bench.longmemeval_blind \
  --split test --limit 30 --output bench_out/m08.json
```

Bench modules NOT runnable on machines without GPU / dataset / external API are skipped with an explicit message; the harness never silently produces a 0.

### Comparator runs

When a reference adapter is present under `bench/adapters/`, pass its path to the same axis script via `--ref-<tool>` flags:

```bash
python -m bench.total_session_cost --ref-mempalace 7000 --ref-claude-mem 5000 \
  --wake-depth standard --output bench_out/m07_full.json
```

The output JSON's `refs` field carries the reference numbers alongside IAI-MCP's number for downstream rendering.

## Reading the results

Each bench writes one JSON file. The structure is documented inline in each `bench/<axis>.py` docstring and is stable across versions (additive only). Per-axis interpretation:

- **M-01** — three rows: fresh / warm p50 / warm p95. The number is total tokens an MCP host would pay at session start. Compare against the IAI-MCP target (≤3000 warm) and against any reference numbers in `refs`.
- **M-02** — per-store-size latency at p50 and p95. Add per-stage timings (embed / gate / seeds / spread / rank) if `neural_map.py` was run with `--per-stage`. Reference adapters with API-roundtrip latency carry that explicitly in their JSON; do not blame implementation for network cost.
- **M-03** — single RSS number in MB. Honest disclosure: this is what the bge-m3 embedder + graph runtime cost in RAM. There is no target; reading the number is the entire point.
- **M-04** — three gap numbers. Each must hit ≥0.99 byte-exact for the project to consider verbatim claims honest. If any gap drops, flag explicitly.
- **M-05** — curve plot data. Each metric reports a per-session series. Report whether the predicted direction holds across the 30 sessions.
- **M-06** — per-language accuracy. No aggregate. If any language drops below 0.9, flag it under "Anomalies".
- **M-07** — three rows (minimal / standard / deep wake-depth). Per-turn cost decomposition.
- **M-08** — R@5, R@10 means; retrieve vs full-pipeline lift. Hard-coded fixture, run-to-run reproducibility is the point.

## Report template

A bench report is a single markdown file with one section per axis. Recommended sections:

1. **Headline numbers** — one line per axis: target vs measured.
2. **Per-axis tables** — full JSON output rendered as a table.
3. **Reference comparison** — when adapters were run, a side-by-side row per reference tool.
4. **Anomalies** — anything missed-target or surprising. Be specific.
5. **Environment** — Python version, OS, RAM, model checkpoints used, IAI-MCP commit hash.
6. **Caveats** — anything the bench couldn't measure (missing adapter, skipped axis, etc.).

No winner language. No leaderboard framing. The measurements are the report; the design judgements are downstream.

## Honest disclosures

- `bench/total_session_cost.py` is a **simulated** 10-turn script — it reproduces the token composition (system overhead + tool descriptions + tool-call payloads + tool-result bodies) a real MCP runtime would emit for the turn kinds. Real runtime adds JSON-RPC envelope overhead (~30-50 tok/turn); the simulation excludes that. Bench rows MUST disclose this caveat alongside the row.
- `bench/longmemeval_blind.py` uses a fixed LongMemEval-S dataset commit; results are reproducible across hosts. Pre-flight requires `IAI_MCP_CRYPTO_PASSPHRASE` in env (per-row tmp stores are isolated and don't inherit the home keychain) and exits loudly if missing — the bench will not silently report `0.000` because of an env miss.
- M-03 RAM is a steady-state RSS snapshot. Cold-start RSS is lower (~600 MB) before the bge-m3 embedder loads.
