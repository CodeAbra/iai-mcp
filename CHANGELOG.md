# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-05-13

### Security

- Precache file (`~/.iai-mcp/.session-start-payload.cached.md`) now created with mode 0600 instead of process umask default (was 0644 world-readable).

## [0.3.1] — 2026-05-13

### Added

- **Session-start precache**: the daemon writes the recall payload to a cache file (`~/.iai-mcp/.session-start-payload.cached.md`) once per REM-loop completion. The SessionStart hook reads this file when fresh (mtime < 24 h), avoiding a JSON-RPC call into core that would block on the exclusive store lock during DREAMING.
- 4 new regression tests covering the precache writer, cache-hit, cache-miss-absent, and cache-miss-stale paths.

### Changed

- `assemble_session_start` refactored into an emit-free `_compose_session_start_payload` helper plus a thin wrapper that adds the `session_started` event. Public API and return type unchanged.

## [0.3.0] — 2026-05-12

### Added

- **Per-turn ambient capture** via a new `UserPromptSubmit` hook (`deploy/hooks/iai-mcp-turn-capture.sh`). Each prompt and the preceding assistant turn(s) are appended to a per-session `.live.jsonl` buffer as pure file IO (~5 ms, no daemon RPC, no embedder). The Stop hook atomically renames the buffer at session end; the daemon drains it through the full pipeline on the next idle edge.
- **Session-start recall injection** via a new `SessionStart` hook (`deploy/hooks/iai-mcp-session-recall.sh`). On session open the hook calls `iai-mcp session-start` and pipes the assembled memory prefix (L0 identity, L1 critical facts, L2 communities, global rich-club) to stdout, capped at 10 000 chars. Claude Code injects it as `additionalContext`. Fail-safe: empty store or unreachable daemon exits 0 with empty stdout.
- **New CLI command `iai-mcp session-start`** exposes the payload formatter for manual or debug use. Connects to the daemon socket with a 5 s connect / 30 s read timeout.
- **New CLI command `iai-mcp capture-turn-deferred`** exposes the per-turn writer for manual or debug use.
- **3-hook installer**: `iai-mcp capture-hooks install` now wires `UserPromptSubmit`, `Stop`, and `SessionStart` hooks into `~/.claude/settings.json`. Uninstall and status report all three.
- **Daemon DROWSY drain**: the daemon now drains the deferred-captures buffer on the `WAKE → DROWSY` lifecycle edge (5-min idle) in addition to the existing post-REM drain. Buffers no longer sit indefinitely when a quiet window doesn't fire.
- **Auto-provision `.crypto.key`**: `iai-mcp daemon install` and `scripts/install.sh` auto-generate `~/.iai-mcp/.crypto.key` on fresh installs. Idempotent; the `IAI_MCP_CRYPTO_PASSPHRASE` fallback is preserved.
- **Drain cap**: each drain pass is capped at 5 000 events. Remainder is written to `*.partial.jsonl` for the next pass.
- README: headless/VPS deployment section, AVX2 requirement, troubleshooting table.

### Changed

- **Capture hooks section** in README rewritten for the 3-hook model.

## [0.2.0] — 2026-05-12

### Added

- **Opt-in int8 embedding quantization** via the `IAI_MCP_EMBED_QUANTIZE=int8` environment variable. The default `fp32` path is unchanged. Round-trip cosine similarity ≥ 0.99 on `bge-small-en-v1.5` in tests. New `Embedder.embed_quantized()` surface returns a `QuantizedVector` with per-vector `scale` and `zero_point` calibration.
- **Derived temporal validity**: `memory_recall` hits and anti-hits now carry `valid_from` and `valid_to` fields derived at recall time from the contradiction-edge graph. `valid_from` defaults to the record's `created_at`; `valid_to` is set only when a newer record contradicts it. Both default to `None` on paths that don't enrich (back-compat preserved).
- **MCP tool annotations and outputSchema** on every tool. Each tool now declares `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`, and `title` annotations plus a structured `outputSchema`. Lifts Glama TDQS from C to B.
- **`BENCHMARKS.md`** — public methodology document covering the eight project benchmarks (M-01 token budget, M-02 latency, M-03 RSS, M-04 verbatim, M-05 trajectory, M-06 multilingual, M-07 session cost, M-08 LongMemEval-S).
- **Bench harness reliability**: `bench/longmemeval_blind.py` now supports `--resume` and `--fresh` flags, auto-cleans errored checkpoints by default, requires an explicit `IAI_MCP_STORE_PASSPHRASE` for the encrypted store, and classifies errored rows separately from genuine misses in the summary.
- **Codex CLI** as an optional `capture-hooks` target for ambient Stop-hook capture. New: `iai-mcp capture-hooks install --target codex|claude|all` and `iai-mcp capture-hooks status --target all`.
- README documents Claude Code and Codex setup paths for capture hooks and MCP wiring.

### Changed

- **Behavior — stale downweight on recall.** Records contradicted by a newer record are now downweighted (not hidden) in both `hits` and `anti_hits`. Score is multiplied by `STALE_DOWNWEIGHT_FACTOR`, and the `reason` field carries a ` · stale` suffix. Top-K ranking may shift compared to v0.1.0 — fresh lower-cosine records can outrank stale higher-cosine ones. Audit trail preserved (records are not removed).
- **API contract — deterministic `overnight_digest`.** The `overnight_digest` block in `memory_recall` responses is now deterministic: same inputs produce the same shape and field set. When no REM cycle has run, the digest is a zeroed default instead of a partial dict. Same top-level keys returned over both stdio and socket transports.
- **API contract — `camouflaging_status` outputSchema fields renamed** to match the actual Python response. `formality_trend` → `trajectory_slope`, `anomaly_score` → `current_mean`, plus new `sample_count: integer`. Permissive JSON Schema consumers were already tolerant; strict-validation consumers must update.

### Known fragile surfaces

- `IAI_MCP_EMBED_QUANTIZE` accepts only `int8` (lowercase) or unset. Any other value — including `INT8`, `int4`, or typos — causes the daemon to fail loud at startup with a `ValueError`. This is intentional; no silent fallback to `fp32`.
- New `valid_from` and `valid_to` keys in `hits[]` and `anti_hits[]` are additive (default `None`). Strict JSON Schema consumers that validate with `additionalProperties: false` will reject the response shape until they widen their schema.
- The `_knobs_applied` field is present in the `memory_recall` response but is not yet declared in the tool's `outputSchema`. Known debt; will be addressed in a follow-up release.

### Acknowledgements

- Reddit user [u/BeginningReflection4](https://www.reddit.com/user/BeginningReflection4) — feedback and testing that shaped this release.

## [0.1.0] — 2026-05-11

Initial public release. Local memory daemon for MCP-over-stdio hosts. Verbatim recall, ambient capture, sleep-cycle consolidation, encrypted-at-rest LanceDB store, configurable operating profile.

[0.3.0]: https://github.com/CodeAbra/iai-mcp/releases/tag/v0.3.0
[0.2.0]: https://github.com/CodeAbra/iai-mcp/releases/tag/v0.2.0
[0.1.0]: https://github.com/CodeAbra/iai-mcp/releases/tag/v0.1.0
