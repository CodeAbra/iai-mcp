"""Plan 05-11 blind-run orchestrator — / M-08.

Runs LongMemEval-S through IAI-MCP's public API (MemoryStore.insert +
retrieve.recall) in strict blind mode: no per-dataset tuning, no
hyperparameter sweep, no late adjustment after seeing numbers. This is
the external honesty axis for Phase 5.

## Row-level protocol

One evaluation row in LongMemEval-S contains:

    { "question", "answer_session_ids" (gold),
      "haystack_session_ids", "haystack_sessions" (the full history) }

Per row the orchestrator does:

    1. fresh tmp MemoryStore (per-row isolation; no cross-row leakage)
    2. enable async writes (Plan 05-10 — keeps RAM bounded on a
       16GB M1 laptop)
    3. embed + insert every turn of every haystack session; each record
       is tagged with ``session:<session_id>`` so the orchestrator can
       score at the dataset's native session-ID granularity.
    4. disable async writes (flushes the queue; the store now holds the
       full haystack).
    5. build_runtime_graph once (Plan 05-09 cache amortises cold start
       across rows via the shared runtime graph cache dir).
    6. call retrieve.recall for the eval query, with k_hits=10.
    7. compute R@5 / R@10 at session-ID granularity (the standard
       LongMemEval metric): a retrieved record "hits" if its ``session:``
       tag is in answer_session_ids. R@k is 1.0 if any top-k hits, else 0.
    8. measure per-query token cost via bench.tokens counters.

## CLI

    python bench/longmemeval_blind.py \\
        --split S \\
        [--limit N] \\
        [--granularity {session, turn}] \\
        [--dataset {cleaned, raw}] \\
        [--qid-include csv] \\
        --out /tmp/p11_lme_full.json

Phase 9 added two methodology-alignment flags:

    --granularity session   (default; one record per session,
                             content = "\\n".join(user-only turns))
    --granularity turn      (v1/v2 reproducer; one record per turn)
    --dataset cleaned       (default; xiaowu0162/longmemeval-cleaned)
    --dataset raw           (v1/v2 reproducer; xiaowu0162/longmemeval
                             rev 2ec2a557f339)
    --qid-include csv       optional comma-separated question_ids; when
                             set, only those rows run (used by smoke
                             tests for per-qid baseline verification)

## Output JSON keys

    {
      "split": "S",
      "dataset_id": "xiaowu0162/longmemeval-cleaned" | "xiaowu0162/longmemeval",
      "revision": "<40-hex>",
      "granularity": "session" | "turn",
      "dataset_choice": "cleaned" | "raw",
      "n_rows": int,                 # rows actually evaluated
      "r_at_5": float,               # session-ID R@5, mean across rows
      "r_at_10": float,              # session-ID R@10, mean across rows
      "token_p50": int,              # per-query cue-text tokens, median
      "token_p95": int,              # per-query cue-text tokens, p95
      "session_tokens_mean": float,  # mean per-row inserted text tokens
                                     # (proxy for the rows' storage footprint)
      "errors": [{"question_id": str, "error_class": str, "error": str}],
      "hard_limit": int | null,
      "note": str
    }

## discipline

The run is ONE-SHOT. If a bug crashes a row, it's logged in ``errors``
and counted as a MISS against R@k (not silently dropped). The published
number is whatever came out. Disclosures (small-N, hardware limit,
English-only embedder, etc.) live in the published bench report and
05-11-SUMMARY.md — they don't get folded back into this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

# Silence the "UNEXPECTED embeddings.position_ids" noise from
# sentence-transformers so the blind-run stderr stays focused on errors.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# IAI-MCP imports — public API only (plan directive).
from iai_mcp.embed import Embedder, embedder_for_store
from iai_mcp.pipeline import recall_for_benchmark
from iai_mcp.retrieve import build_runtime_graph, recall as retrieve_recall
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

# Adapter (ships alongside this script).
from bench.adapters.longmemeval import (
    DATASET_ID,
    PINNED_REVISION,
    LMESession,
    LongMemEvalAdapter,
)

# Token counter (reuses bench/tokens.py three-tier helper).
from bench.tokens import _char4_count, _tiktoken_count


def _count_tokens(text: str) -> int:
    """Prefer tiktoken-cl100k proxy; fall back to char4."""
    try:
        return _tiktoken_count(text)
    except Exception:  # pragma: no cover
        return _char4_count(text)


def preflight_crypto_or_exit() -> None:
    """Block the run loudly if crypto is unconfigured.

    Bench uses a FRESH per-row ``MemoryStore`` rooted in a unique tmp
    directory (``/tmp/lme_blind_*/row-*/lancedb/``). Per-row tmp dirs
    have no pre-existing ``.crypto.key`` file, and the home-directory
    ``~/.iai-mcp/.crypto.key`` does NOT propagate into per-row tmp
    stores. The only crypto state that reaches the per-row store is
    ``IAI_MCP_CRYPTO_PASSPHRASE`` from the process environment.

    Therefore D1 requires the env var; the file-backend keychain at the
    user's home is irrelevant to bench execution and was a false positive
    in the original pre-flight (smoke-caught 2026-05-11).

    On failure: prints a message to stderr explaining the per-row
    isolation pattern and naming the env var, then raises
    ``SystemExit(2)``.
    """
    if os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        return
    print(
        "[LME] FATAL: crypto unconfigured. Bench uses per-row isolated "
        "MemoryStore tmp dirs; the home-directory ~/.iai-mcp/.crypto.key "
        "keychain does NOT propagate to per-row tmp stores. "
        "Set IAI_MCP_CRYPTO_PASSPHRASE in env before invoking the harness.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


def _percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p / 100.0))))
    return s[k]


def _make_record(
    content: str,
    session_id: str,
    role: str,
    embedding: list[float],
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    from uuid import uuid4

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=content,
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
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[
            "longmemeval",
            f"role:{role}",
            f"session:{session_id}",
        ],
        language="en",
    )


def _run_one_row(
    row_id: str,
    question: str,
    question_type: str,
    answer_session_ids: set[str],
    sessions: list[LMESession],
    tmp_root: Path,
    granularity: str = "turn",
    embedder_key: str = "bge-small-en-v1.5",
) -> dict[str, Any]:
    """Execute the per-row protocol. Returns a dict with r_at_5/r_at_10
    for BOTH retrieve_recall (flat-cosine baseline, matches Phase 5
    n=30) AND recall_for_benchmark (full graph-native architecture; Phase
    8 entry-point split), token counts plus timing info. Raises
    only on programmer errors; dataset/runtime errors are caught by the
    caller.

    bench/lme500 protocol: prong X = retrieve_recall, prong Y =
    recall_for_benchmark. Both share the same insert phase + retrieved-set
    mapping, so the architecture-vs-baseline delta is attributable to
    the recall function only, not retrieval-side variance.

    ``granularity`` controls corpus construction.
        "turn"    -> one record per turn (v1/v2 baseline; ~500 records/row)
        "session" -> one record per session whose content is
                     "\\n".join(user-only turns), matching mempalace's
                     reference verbatim (~53 records/row).
    """
    t0 = time.time()

    # Fresh store in a per-row tmp dir.
    store_dir = tmp_root / f"row-{row_id}"
    store_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_dir / "lancedb")

    # async writes: coalesce LanceDB appends across the row.
    # enable_async_writes is a coroutine — drive it from a fresh loop so
    # the surrounding orchestrator stays sync.
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))

    # count inserted tokens as a rough storage footprint.
    inserted_text_tokens = 0

    # route through the explicit registry key so the
    # embedder ablation experiment can swap to all-MiniLM-L6-v2 without
    # touching the production-default resolver (embedder_for_store kept
    # imported for backward-compat; not called on this path).
    embedder = Embedder(model_key=embedder_key)
    _ = embedder_for_store  # silence unused-import warning when the prod path is bypassed

    # --------- INSERT phase ---------
    # One pass over all haystack sessions for this row. Each MemoryRecord is
    # tagged with its session_id so R@k can score at the dataset's native
    # session granularity. splits this into two paths:
    #   - "turn"    (v1/v2 baseline; one record per turn, both roles)
    #   - "session" (mempalace-aligned; one record per session, user-only
    #                turns joined with "\n"; ~10x fewer records per row)
    id_to_session: dict[str, str] = {}  # record_id.hex -> session_id
    if granularity == "session":
        # Session-granularity (D-01, mempalace-aligned): ONE record per
        # session, content = "\n".join(user-only turns). Skip sessions
        # with no user turns. Verbatim shape match with mempalace's
        # benchmarks/longmemeval_bench.py reference loop.
        for sess in sessions:
            user_turns = [
                str(turn.get("content", "")).strip()
                for turn in sess.turns
                if str(turn.get("role", "user")) == "user"
                and str(turn.get("content", "")).strip()
            ]
            if not user_turns:
                continue
            doc_text = "\n".join(user_turns)
            vec = embedder.embed(doc_text)
            rec = _make_record(
                content=doc_text,
                session_id=sess.session_id,
                role="user",
                embedding=vec,
            )
            store.insert(rec)
            id_to_session[str(rec.id)] = sess.session_id
            inserted_text_tokens += _count_tokens(doc_text)
    else:
        # Turn-granularity (v1/v2 baseline; bytes-identical loop body).
        for sess in sessions:
            for turn in sess.turns:
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                vec = embedder.embed(content)
                rec = _make_record(
                    content=content,
                    session_id=sess.session_id,
                    role=str(turn.get("role", "user")),
                    embedding=vec,
                )
                store.insert(rec)
                id_to_session[str(rec.id)] = sess.session_id
                inserted_text_tokens += _count_tokens(content)

    # Flush the async queue before recall. disable_async_writes is a
    # coroutine too — drive from a fresh loop.
    asyncio.run(store.disable_async_writes())
    t_after_insert = time.time()

    # --------- Build runtime graph (Plan 05-09 cache warms cold-start) ---------
    # bench/lme500: capture the (graph, assignment, rich_club) tuple so
    # recall_for_benchmark (prong Y) can reuse it. retrieve_recall (prong X)
    # is unaffected by graph build success/failure.
    graph = None
    assignment = None
    rich_club = None
    try:
        graph, assignment, rich_club = build_runtime_graph(store)
    except Exception as exc:  # pragma: no cover — cache helpers should be robust
        # Don't fail the row on graph build; retrieve_recall is still
        # callable from the flat store. recall_for_benchmark will be skipped
        # for this row and counted as miss for the Y prong.
        print(
            f"[LME] row={row_id} build_runtime_graph failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    t_after_graph = time.time()

    # --------- Prong X: retrieve_recall (flat-cosine, baseline) ---------
    cue_embedding = embedder.embed(question)
    resp_x = retrieve_recall(
        store=store,
        cue_embedding=cue_embedding,
        cue_text=question,
        session_id=f"lme-{row_id}",
        budget_tokens=1500,
        k_hits=10,
        k_anti=0,
    )
    t_after_x = time.time()

    # --------- Prong Y: recall_for_benchmark (full graph-native architecture) ---------
    # entry-point split: bench harness uses the top-K contract
    # (k_hits=10, no budget_tokens). mode="concept" preserved verbatim — the
    # bench is concept-shaped per BENCH_PROTOCOL_lme500.md and the D-02
    # `_gate_bias_for_mode("concept") == 0.1` bias is what v2 measurements observe.
    resp_y = None
    pipeline_error: str | None = None
    if graph is not None:
        try:
            resp_y = recall_for_benchmark(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=question,
                session_id=f"lme-{row_id}",
                k_hits=10,
                profile_state=None,
                turn=0,
                mode="concept",
            )
        except Exception as exc:
            pipeline_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(
                f"[LME] row={row_id} recall_for_benchmark failed: "
                f"{pipeline_error}",
                file=sys.stderr,
            )
    else:
        pipeline_error = "graph_build_failed"
    t_after_y = time.time()

    def _retrieved_session_ids(resp) -> list[str]:
        if resp is None:
            return []
        out: list[str] = []
        for hit in resp.hits:
            sid = id_to_session.get(str(hit.record_id))
            if sid is not None:
                out.append(sid)
        return out

    sids_x = _retrieved_session_ids(resp_x)
    sids_y = _retrieved_session_ids(resp_y)

    # LongMemEval-standard R@k at session-ID granularity: hit-at-k.
    #   R@k = 1.0 if any of the top-k retrieved records belongs to a gold
    #   session, else 0.0. Aggregated across rows by the caller.
    def _hit_at_k(sids: list[str], k: int) -> float:
        top = sids[:k]
        return 1.0 if any(s in answer_session_ids for s in top) else 0.0

    r5_x = _hit_at_k(sids_x, 5)
    r10_x = _hit_at_k(sids_x, 10)
    r5_y = _hit_at_k(sids_y, 5) if resp_y is not None else 0.0
    r10_y = _hit_at_k(sids_y, 10) if resp_y is not None else 0.0

    query_tokens = _count_tokens(question)

    return {
        "question_id": row_id,
        "question_type": question_type,
        # Prong X — retrieve_recall (flat-cosine baseline, line-by-line)
        "r_at_5_retrieve": r5_x,
        "r_at_10_retrieve": r10_x,
        # Prong Y — recall_for_benchmark (full graph-native pipeline; D-07)
        "r_at_5_pipeline": r5_y,
        "r_at_10_pipeline": r10_y,
        "pipeline_error": pipeline_error,
        # Shared
        "query_tokens": query_tokens,
        "inserted_text_tokens": inserted_text_tokens,
        "n_haystack_sessions": len(sessions),
        "n_turns_inserted": len(id_to_session),
        "timing_seconds": {
            "insert": round(t_after_insert - t0, 2),
            "graph": round(t_after_graph - t_after_insert, 2),
            "recall_retrieve": round(t_after_x - t_after_graph, 2),
            "recall_pipeline": round(t_after_y - t_after_x, 2),
            "total": round(t_after_y - t0, 2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="S",
        choices=["S", "M", "oracle"],
        help="LongMemEval split (Plan 05-11 runs S)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "practical-cap on rows evaluated. LongMemEval-S = 500 rows; "
            "at ~500 turns/row and 11ms/embed on a 16GB M1 laptop, the "
            "full 500-row run is multi-hour. --limit lets the blind pilot "
            "finish; the SUMMARY discloses the cap honestly."
        ),
    )
    parser.add_argument(
        "--out",
        default="/tmp/p11_lme_full.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "JSONL checkpoint path for crash-resume; default = <out>.jsonl. "
            "Each completed (or errored) row is appended with fsync as one "
            "JSON line. On restart, rows whose question_id already appears "
            "in the checkpoint are skipped."
        ),
    )
    # granularity flag with mempalace-aligned default.
    parser.add_argument(
        "--granularity",
        choices=["session", "turn"],
        default="session",
        help=(
            "corpus-construction granularity. "
            "'session' (default, v3): one record per session, "
            "content = '\\n'.join(user-only turns) — matches mempalace's "
            "reference. 'turn': one record per turn (v1/v2 baseline; "
            "use with --dataset raw to reproduce v2's 0.956)."
        ),
    )
    # dataset choice flag with mempalace-aligned default.
    parser.add_argument(
        "--dataset",
        choices=["cleaned", "raw"],
        default="cleaned",
        help=(
            "dataset variant. 'cleaned' (default, v3): "
            "xiaowu0162/longmemeval-cleaned, SHA pinned via repo_info(). "
            "'raw' (v1/v2 baseline): xiaowu0162/longmemeval rev "
            "2ec2a557f339... — use with --granularity turn to reproduce "
            "v2's 0.956."
        ),
    )
    # Step B: per-qid filter for the v2-baseline
    # smoke reproducer. Applied AFTER --limit so a future caller passing
    # both flags gets a deterministic intersection (limit narrows by row
    # count, qid-include narrows by id). Default None preserves v1/v2 behaviour.
    parser.add_argument(
        "--qid-include",
        default=None,
        help=(
            "comma-separated list of question_ids; if set, only these "
            "rows run (used by smoke tests for per-qid baseline "
            "verification). Applied after --limit."
        ),
    )
    # bench-only embedder swap. Default preserves v3
    # baseline (bge-small-en-v1.5). all-MiniLM-L6-v2 is mempalace's ChromaDB
    # default — used for the embedder-axis ablation in v3.1. Production
    # embedder is unchanged regardless of this flag (English-Only Brain lock
    # from / Plan 05-08; the Embedder.__init__ kwarg is the only
    # entry point that surfaces the registry's all-MiniLM-L6-v2 entry).
    parser.add_argument(
        "--embedder",
        choices=["bge-small-en-v1.5", "all-MiniLM-L6-v2"],
        default="bge-small-en-v1.5",
        help=(
            "embedder model_key. 'bge-small-en-v1.5' (default, v3 "
            "baseline) routes via the production English-only embedder. "
            "'all-MiniLM-L6-v2' (Phase 9.1 ablation) is mempalace's "
            "ChromaDB default — bench-only swap, production unchanged."
        ),
    )
    # Checkpoint disposition flags. Default behaviour (no flag):
    # auto-clean when the prior checkpoint contains ERROR rows, keep it
    # otherwise. --resume opts into the old keep-no-matter-what behaviour;
    # --fresh force-cleans even a SUCCESS-only checkpoint. The two are
    # mutually exclusive because the auto-clean default ALREADY handles the
    # common "errored run, retry from scratch" case.
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep the existing checkpoint as-is even if it contains rows "
            "classified as ERROR. Without this flag, a checkpoint containing "
            "prior errors is auto-cleaned (the default; errors are usually a "
            "fail-fast crypto / env miss and the right action is to fix the "
            "environment and rerun from scratch). Mutually exclusive with "
            "--fresh."
        ),
    )
    resume_group.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Force-clean the checkpoint even if it contains no errors. "
            "Mutually exclusive with --resume."
        ),
    )
    args = parser.parse_args(argv)

    # Fail-loud crypto pre-flight BEFORE any adapter load / row work.
    # If IAI_MCP_CRYPTO_PASSPHRASE is not set, per-row store.insert(...)
    # would raise on the encryption path; the row-level except previously
    # folded those into R@5 as silent MISS, producing clean-looking 0.000
    # JSON. Now: SystemExit(2) before any row is touched.
    preflight_crypto_or_exit()

    print(
        f"[LME] blind run starting "
        f"split={args.split} limit={args.limit} "
        f"granularity={args.granularity} dataset={args.dataset} "
        f"embedder={args.embedder} "
        f"out={args.out}",
        file=sys.stderr,
        flush=True,
    )

    # branch the adapter on --dataset.
    if args.dataset == "cleaned":
        from bench.adapters.longmemeval_cleaned import (
            CLEANED_DATASET_ID,
            CleanedLongMemEvalAdapter,
        )
        adapter = CleanedLongMemEvalAdapter()
        dataset_id_emit = CLEANED_DATASET_ID
        revision_emit = adapter.revision
    else:
        adapter = LongMemEvalAdapter()
        dataset_id_emit = DATASET_ID
        revision_emit = PINNED_REVISION
    # Adapter yields one LMESession per haystack session, but the
    # blind-run protocol needs rows (one question + all its haystack
    # sessions). Group by question_id (carried inside queries[0]).
    grouped: dict[str, dict[str, Any]] = {}
    row_order: list[str] = []
    for lme_session in adapter.load_dataset(split=args.split):
        q = lme_session.queries[0]
        qid = q["question_id"]
        if qid not in grouped:
            grouped[qid] = {
                "question": q["query"],
                "question_type": q.get("question_type", "unknown"),
                "answer_session_ids": set(q.get("relevant_turn_ids", [])),
                "sessions": [],
            }
            row_order.append(qid)
        grouped[qid]["sessions"].append(lme_session)

    if args.limit is not None:
        row_order = row_order[: args.limit]

    # Step B: --qid-include filter applied AFTER
    # --limit so a future caller passing both flags gets a deterministic
    # intersection. The default None path is a no-op for backward compat.
    if args.qid_include is not None:
        wanted = {q.strip() for q in str(args.qid_include).split(",") if q.strip()}
        row_order = [qid for qid in row_order if qid in wanted]
        print(
            f"[LME] qid-include filter: kept {len(row_order)} of "
            f"{len(wanted)} requested qids",
            file=sys.stderr,
            flush=True,
        )

    tmp_root = Path(tempfile.mkdtemp(prefix="lme_blind_"))
    print(f"[LME] per-row stores rooted at {tmp_root}", file=sys.stderr, flush=True)

    per_row: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    # bench/lme500: track BOTH prongs (X = retrieve_recall, Y = recall_for_benchmark).
    r5_x_values: list[float] = []
    r10_x_values: list[float] = []
    r5_y_values: list[float] = []
    r10_y_values: list[float] = []
    query_tokens: list[int] = []
    session_tokens: list[int] = []

    # bench/lme500: per-row JSONL checkpoint for crash resume.
    # Each row's full result is appended with flush + fsync, so a kill at
    # row N preserves rows 1..N-1 fully. Restart skips rows already in the
    # checkpoint (matched by question_id).
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(str(args.out) + ".jsonl")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint disposition gate. Decide BEFORE the resume scan whether
    # to keep the file or unlink and start fresh. Three paths:
    #   1. --fresh           -> always unlink (even on SUCCESS-only).
    #   2. prior errors + no --resume -> auto-clean + verbatim phrase.
    #   3. otherwise         -> leave checkpoint; the existing resume
    #                           scan re-checks `.exists()` and picks it up.
    if checkpoint_path.exists():
        prior_error_count = 0
        prior_total_count = 0
        with open(checkpoint_path, "r", encoding="utf-8") as cp_dispo:
            for line in cp_dispo:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupt line — the existing resume scan already
                    # logs + skips these. Don't count as error for
                    # disposition (the next resume scan will warn).
                    continue
                prior_total_count += 1
                # Recognise BOTH the new `classification == "ERROR"`
                # shape AND the legacy `"error" in rec` shape.
                if (
                    rec.get("classification") == "ERROR"
                    or "error" in rec
                ):
                    prior_error_count += 1

        if args.fresh:
            print(
                f"[LME] --fresh: discarding {prior_total_count}-row "
                f"checkpoint at {checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )
            checkpoint_path.unlink()
        elif prior_error_count > 0 and not args.resume:
            # Verbatim phrase contract — grepped literally by reviewer
            # tooling. Do NOT pluralise "1 errors" / restructure.
            print(
                f"[LME] Resuming from prior run with {prior_error_count} "
                f"errors; starting fresh. Pass --resume to keep checkpoint.",
                file=sys.stderr,
                flush=True,
            )
            checkpoint_path.unlink()
        # else: keep checkpoint; the existing resume scan below picks it up.

    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as cp_f:
            for line in cp_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[LME] WARN: skipping corrupt checkpoint line: {line[:80]!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                qid = rec.get("question_id")
                if not qid:
                    continue
                completed_ids.add(qid)
                # Recognise BOTH the new top-level `classification == "ERROR"`
                # shape AND the legacy `"error" in rec` shape (already-on-disk
                # checkpoints written before this patch landed).
                is_error_row = (
                    rec.get("classification") == "ERROR"
                    or (
                        "error" in rec and isinstance(rec.get("error"), dict)
                    )
                )
                if is_error_row:
                    # Resumed error row: count as full miss for both prongs.
                    errors.append(
                        {
                            "question_id": qid,
                            "error_class": rec["error"].get("error_class", "Unknown"),
                            "error": rec["error"].get("error", ""),
                        }
                    )
                    r5_x_values.append(0.0)
                    r10_x_values.append(0.0)
                    r5_y_values.append(0.0)
                    r10_y_values.append(0.0)
                    query_tokens.append(0)
                    session_tokens.append(0)
                else:
                    # Resumed success row.
                    per_row.append(rec)
                    r5_x_values.append(float(rec.get("r_at_5_retrieve", 0.0)))
                    r10_x_values.append(float(rec.get("r_at_10_retrieve", 0.0)))
                    r5_y_values.append(float(rec.get("r_at_5_pipeline", 0.0)))
                    r10_y_values.append(float(rec.get("r_at_10_pipeline", 0.0)))
                    query_tokens.append(int(rec.get("query_tokens", 0)))
                    session_tokens.append(int(rec.get("inserted_text_tokens", 0)))
    if completed_ids:
        print(
            f"[LME] resume: {len(completed_ids)} rows already in checkpoint "
            f"{checkpoint_path}; processing {len(row_order) - len(completed_ids)} remaining",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[LME] checkpoint: writing per-row durable JSONL to {checkpoint_path}",
            file=sys.stderr,
            flush=True,
        )

    def _checkpoint_append(rec: dict[str, Any]) -> None:
        """Append one row record to the checkpoint, flush+fsync for durability."""
        with open(checkpoint_path, "a", encoding="utf-8") as cp_a:
            cp_a.write(json.dumps(rec) + "\n")
            cp_a.flush()
            os.fsync(cp_a.fileno())

    run_t0 = time.time()
    for i, qid in enumerate(row_order):
        if qid in completed_ids:
            continue
        row = grouped[qid]
        try:
            res = _run_one_row(
                row_id=qid,
                question=row["question"],
                question_type=row["question_type"],
                answer_session_ids=row["answer_session_ids"],
                sessions=row["sessions"],
                tmp_root=tmp_root,
                granularity=args.granularity,
                embedder_key=args.embedder,
            )
            per_row.append(res)
            r5_x_values.append(res["r_at_5_retrieve"])
            r10_x_values.append(res["r_at_10_retrieve"])
            r5_y_values.append(res["r_at_5_pipeline"])
            r10_y_values.append(res["r_at_10_pipeline"])
            query_tokens.append(res["query_tokens"])
            session_tokens.append(res["inserted_text_tokens"])
            # Splice classification at append time so the on-disk JSONL
            # agrees with the in-memory per_row list. Keep `_run_one_row`
            # itself pure — it knows nothing about the ERROR / SUCCESS
            # taxonomy.
            res_with_class = dict(res)
            res_with_class["classification"] = "SUCCESS"
            _checkpoint_append(res_with_class)
            elapsed = time.time() - run_t0
            print(
                f"[LME] row {i+1}/{len(row_order)} qid={qid} "
                f"qtype={res['question_type']} "
                f"R@5_x={res['r_at_5_retrieve']:.0f} R@5_y={res['r_at_5_pipeline']:.0f} "
                f"R@10_x={res['r_at_10_retrieve']:.0f} R@10_y={res['r_at_10_pipeline']:.0f} "
                f"t_row={res['timing_seconds']['total']:.1f}s "
                f"t_total={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            # T-05-11-04 mitigation: log + count as miss, do
            # NOT silently drop.
            err_payload = {
                "error_class": type(exc).__name__,
                "error": str(exc)[:500],
            }
            errors.append({"question_id": qid, **err_payload})
            # Counted as a full miss for both prongs — preserves
            # "count against R@5 as 0" from the plan text.
            r5_x_values.append(0.0)
            r10_x_values.append(0.0)
            r5_y_values.append(0.0)
            r10_y_values.append(0.0)
            query_tokens.append(0)
            session_tokens.append(0)
            # Persist the error row to checkpoint so a restart skips it.
            # Top-level classification="ERROR" tag so the resume scan +
            # auto-clean disposition can distinguish ERROR rows from
            # genuine MISS rows without parsing the embedded error
            # payload. The legacy `error` payload is preserved for
            # backward compat with already-on-disk checkpoints.
            _checkpoint_append(
                {
                    "question_id": qid,
                    "question_type": row.get("question_type", "unknown"),
                    "classification": "ERROR",
                    "error": err_payload,
                }
            )
            print(
                f"[LME] ERROR row={qid}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
        finally:
            # Free disk aggressively — many rows × ~500 turns per store
            # adds up even on 64GB.
            row_dir = tmp_root / f"row-{qid}"
            if row_dir.exists():
                shutil.rmtree(row_dir, ignore_errors=True)

    shutil.rmtree(tmp_root, ignore_errors=True)

    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    # Honest-disclosure summary triple. Derived from already-tracked
    # structures so resumed rows (which append to per_row / errors in the
    # checkpoint scan above) are counted alongside freshly-run rows.
    # `r_at_5_retrieve` is the canonical hit indicator — matches the
    # `r5_x_values` aggregate used for R@5_retrieve below.
    n_hits = sum(
        1
        for r in per_row
        if float(r.get("r_at_5_retrieve", 0.0)) == 1.0
    )
    n_misses = sum(
        1
        for r in per_row
        if float(r.get("r_at_5_retrieve", 0.0)) == 0.0
    )
    n_errors = len(errors)

    out = {
        "split": args.split,
        "dataset_id": dataset_id_emit,
        "revision": revision_emit,
        # reproducibility fields:
        "granularity": args.granularity,
        "dataset_choice": args.dataset,
        # embedder identity pinned for v3.1 ablation reproducibility.
        # Default "bge-small-en-v1.5" reproduces v3 baseline; "all-MiniLM-L6-v2"
        # is the embedder-axis ablation toggle (mempalace ChromaDB default).
        "embedder_model_key": args.embedder,
        "embedder_hf_id": Embedder(model_key=args.embedder).model_name,
        "n_rows": len(row_order),
        # Prong X — retrieve_recall (flat-cosine baseline, line-by-line)
        "r_at_5_retrieve": _mean(r5_x_values),
        "r_at_10_retrieve": _mean(r10_x_values),
        # Prong Y — recall_for_benchmark (full graph-native architecture; D-07)
        "r_at_5_pipeline": _mean(r5_y_values),
        "r_at_10_pipeline": _mean(r10_y_values),
        # Architecture lift (Y - X)
        "r_at_5_lift": _mean(r5_y_values) - _mean(r5_x_values),
        "r_at_10_lift": _mean(r10_y_values) - _mean(r10_x_values),
        "token_p50": _percentile(query_tokens, 50),
        "token_p95": _percentile(query_tokens, 95),
        "session_tokens_mean": (
            statistics.fmean(session_tokens) if session_tokens else 0.0
        ),
        "errors": errors,
        # ERROR-vs-MISS honest-disclosure triple. R@5 / R@10 means stay
        # computed over the union of success + error rows for backward
        # compat; the triple below is what the reviewer asked for to make
        # silent zeros impossible.
        "n_hits": n_hits,
        "n_misses": n_misses,
        "n_errors": n_errors,
        "hard_limit": args.limit,
        "metric_def": (
            "Session-ID hit-at-k: R@k = 1.0 if any of top-k retrieved records "
            "belongs to a gold session_id, else 0.0 (LongMemEval standard)."
        ),
        "per_row": per_row,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": round(time.time() - run_t0, 2),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ERROR-vs-MISS triple on the DONE line so the reviewer can see at a
    # glance whether a 0.000 R@5 was a real product miss or an environment
    # misconfiguration. `errors=N` retained for grep-compat with already-
    # saved bench logs; the new `hits=N misses=N errors=N` triple is the
    # honest-disclosure surface.
    print(
        f"[LME] DONE n_rows={out['n_rows']} "
        f"R@5_retrieve={out['r_at_5_retrieve']:.3f} "
        f"R@5_pipeline={out['r_at_5_pipeline']:.3f} "
        f"lift_R@5={out['r_at_5_lift']:+.3f} "
        f"R@10_retrieve={out['r_at_10_retrieve']:.3f} "
        f"R@10_pipeline={out['r_at_10_pipeline']:.3f} "
        f"lift_R@10={out['r_at_10_lift']:+.3f} "
        f"hits={n_hits} misses={n_misses} errors={n_errors} "
        f"-> {args.out}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
