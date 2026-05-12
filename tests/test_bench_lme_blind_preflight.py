"""D1 — pre-flight crypto check + ERROR-vs-MISS classification + summary counters.

Background: today, running `bench/longmemeval_blind.py` without
`IAI_MCP_CRYPTO_PASSPHRASE` set (and no `.crypto.key` file) produces a
clean-looking JSON saying we scored 0/500 — every row errors out inside
`store.insert` on the encryption path and gets folded into R@5 / R@10 as
a MISS. The harness writes a final JSON with `r_at_5 == 0.0`, no loud
signal that crypto was the real problem.

This file pins five contracts:

D1.1 — pre-flight crypto check:
    1. `test_preflight_exits_when_no_crypto` — no env var, no key file =>
       exits with code 2 BEFORE any adapter / row work; the output JSON
       is never created.
    2. `test_preflight_passes_with_passphrase` — `IAI_MCP_CRYPTO_PASSPHRASE`
       set => happy path.
    3. `test_preflight_passes_with_key_file` — `.crypto.key` in store
       root => happy path.

D1.2 — ERROR-vs-MISS classification + summary line:
    4. `test_error_row_classified_as_error_not_miss` — per-row errors
       written to checkpoint JSONL with `"classification": "ERROR"`;
       output JSON carries `n_hits` / `n_misses` / `n_errors` as three
       separate top-level integers.
    5. `test_summary_line_separates_errors_from_misses` — stderr DONE
       line contains `hits=N misses=N errors=N` in that order.

Adapter calls are stubbed via `_patch_adapter`; row execution is stubbed
via `_patch_run_one_row`. No HuggingFace network access at any point.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Shared mocking helpers (referenced verbatim from the PLAN's
# <adapter_mocking_fixture> block). Duplicated inline in
# tests/test_bench_lme_blind_checkpoint.py per the plan note ("duplication
# is fine for a quick-mode plan").
# --------------------------------------------------------------------------- #


class _StubLMESession:
    """Minimal stand-in for bench.adapters.longmemeval.LMESession.

    The blind-run grouping loop reads ``q = lme_session.queries[0]`` and
    pulls ``question_id`` / ``query`` / ``question_type`` /
    ``relevant_turn_ids`` off it. For tests that route through
    `_run_one_row` we monkeypatch `_run_one_row` itself, so the session
    payload only needs to be structurally valid — no real turns required.
    """

    def __init__(self, qid: str, question_type: str = "test") -> None:
        self.queries = [
            {
                "question_id": qid,
                "query": f"q for {qid}",
                "question_type": question_type,
                "relevant_turn_ids": [f"sess-{qid}"],
            }
        ]
        self.session_id = f"sess-{qid}"
        self.turns = []  # _run_one_row is monkeypatched in tests


def _patch_adapter(monkeypatch, qids: list[str] | None = None) -> None:
    """Replace BOTH LongMemEvalAdapter.load_dataset AND
    CleanedLongMemEvalAdapter.load_dataset with a stub iterator.

    `qids=None` / `qids=[]` yields nothing (empty-rows happy path tests).
    Non-empty `qids` yields one `_StubLMESession` per qid.
    """
    sessions = [_StubLMESession(qid) for qid in (qids or [])]

    def _stub_load_dataset(self, split="S"):
        yield from sessions

    from bench.adapters.longmemeval import LongMemEvalAdapter
    from bench.adapters.longmemeval_cleaned import CleanedLongMemEvalAdapter

    monkeypatch.setattr(
        LongMemEvalAdapter, "load_dataset", _stub_load_dataset, raising=True
    )
    monkeypatch.setattr(
        CleanedLongMemEvalAdapter,
        "load_dataset",
        _stub_load_dataset,
        raising=True,
    )


def _patch_run_one_row(
    monkeypatch,
    raise_on_indices: set[int],
    success_template: dict | None = None,
) -> list[int]:
    """Wrap `bench.longmemeval_blind._run_one_row` so calls at indices in
    `raise_on_indices` raise `RuntimeError('synthetic')`; other calls return
    a deep-copied `success_template` with the row's `question_id` spliced
    in. Returns the call counter (list-of-one int) so a test can assert on
    the number of calls.

    Default `success_template` gives `r_at_5_retrieve=0.0` (a genuine MISS)
    so `test_summary_line_separates_errors_from_misses` can mix ERROR +
    MISS without extra wiring.
    """
    counter = [0]
    default_success = success_template or {
        "question_id": None,
        "question_type": "test",
        "r_at_5_retrieve": 0.0,
        "r_at_10_retrieve": 0.0,
        "r_at_5_pipeline": 0.0,
        "r_at_10_pipeline": 0.0,
        "pipeline_error": None,
        "query_tokens": 0,
        "inserted_text_tokens": 0,
        "n_haystack_sessions": 0,
        "n_turns_inserted": 0,
        "timing_seconds": {
            "insert": 0.0,
            "graph": 0.0,
            "recall_retrieve": 0.0,
            "recall_pipeline": 0.0,
            "total": 0.0,
        },
    }

    def _wrapped(
        *,
        row_id,
        question,
        question_type,
        answer_session_ids,
        sessions,
        tmp_root,
        granularity,
        embedder_key,
    ):
        idx = counter[0]
        counter[0] += 1
        if idx in raise_on_indices:
            raise RuntimeError("synthetic")
        out = dict(default_success)
        out["question_id"] = row_id
        out["question_type"] = question_type
        return out

    import bench.longmemeval_blind as mod

    monkeypatch.setattr(mod, "_run_one_row", _wrapped, raising=True)
    return counter


# --------------------------------------------------------------------------- #
# D1.1 — pre-flight crypto check
# --------------------------------------------------------------------------- #


def test_preflight_exits_when_no_passphrase(tmp_path, monkeypatch, capsys):
    """No `IAI_MCP_CRYPTO_PASSPHRASE` env var => SystemExit before adapter load.

    Smoke-caught 2026-05-11: bench creates per-row tmp MemoryStores
    (`/tmp/lme_blind_*/row-*/lancedb/`); a home-directory `.crypto.key`
    does NOT propagate into those tmp dirs. Only the env passphrase
    reaches the per-row store. Pre-flight must require the env var
    and NOT be satisfied by file-backend keychains at the user home.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    with pytest.raises(SystemExit) as ei:
        mod.main(["--limit", "1", "--out", str(out_path)])

    assert ei.value.code != 0, "pre-flight must exit with non-zero code"
    assert ei.value.code == 2

    captured = capsys.readouterr()
    err = captured.err
    assert "IAI_MCP_CRYPTO_PASSPHRASE" in err, (
        "stderr must name the passphrase env var: " + err
    )
    assert "per-row" in err, (
        "stderr must explain the per-row isolation pattern: " + err
    )
    # Output JSON must NOT have been created — pre-flight blocks before write.
    assert not out_path.exists(), (
        "pre-flight must short-circuit before output JSON is written"
    )


def test_preflight_passes_with_passphrase(tmp_path, monkeypatch):
    """Passphrase env var set => happy path (n_rows == 0 via empty adapter)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=[])

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists(), "happy path must write the output JSON"
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


def test_preflight_rejects_keyfile_only(tmp_path, monkeypatch, capsys):
    """`.crypto.key` file in IAI_MCP_STORE but NO env passphrase => still exits.

    Bench per-row tmp stores do not see the home-directory keychain.
    Pre-flight must fail loud even when the user has run `iai-mcp crypto
    init` for the daemon — bench requires the env passphrase explicitly.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    with pytest.raises(SystemExit) as ei:
        mod.main(["--limit", "1", "--out", str(out_path)])

    assert ei.value.code == 2
    captured = capsys.readouterr()
    assert "IAI_MCP_CRYPTO_PASSPHRASE" in captured.err
    assert not out_path.exists()


# --------------------------------------------------------------------------- #
# D1.2 — ERROR-vs-MISS classification + summary counters
# --------------------------------------------------------------------------- #


def test_error_row_classified_as_error_not_miss(tmp_path, monkeypatch):
    """Per-row errors written with `classification == "ERROR"`; output JSON
    carries `n_hits`, `n_misses`, `n_errors` as three separate top-level
    integers.

    Pre-Task-1 the error row writes `"error": {...}` only — no
    `classification` field, no `n_hits` / `n_misses` / `n_errors` summary
    triple. This test pins both the JSONL shape and the summary triple.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=["q1", "q2"])
    _patch_run_one_row(monkeypatch, raise_on_indices={0, 1})

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "2", "--out", str(out_path)])
    assert rc == 0

    # Checkpoint JSONL — both rows must carry the new top-level classification.
    cp_path = tmp_path / "o.json.jsonl"
    assert cp_path.exists(), "checkpoint JSONL must be written"
    lines = [
        json.loads(line)
        for line in cp_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    for rec in lines:
        assert rec.get("classification") == "ERROR", (
            "every errored row must carry classification=ERROR: " + repr(rec)
        )
        # Backward-compat: existing `error` payload preserved.
        assert isinstance(rec.get("error"), dict)
        assert "error_class" in rec["error"]

    # Output JSON — summary triple at top level.
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert len(out["errors"]) == 2
    assert out["n_errors"] == 2
    assert out["n_misses"] == 0
    assert out["n_hits"] == 0


def test_summary_line_separates_errors_from_misses(
    tmp_path, monkeypatch, capsys
):
    """Stderr DONE line contains `hits=N misses=N errors=N` in that order.

    Setup: 3 rows. Rows 0 and 1 raise (-> 2 errors). Row 2 succeeds with
    `r_at_5_retrieve=0.0` (the default-template MISS). Expect:
      hits=0 misses=1 errors=2.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    _patch_adapter(monkeypatch, qids=["q1", "q2", "q3"])
    _patch_run_one_row(monkeypatch, raise_on_indices={0, 1})

    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "3", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    # All three counts must appear in the same line, in this order.
    # Use a regex-flexible substring search: locate `hits=0`, then ensure
    # `misses=1` follows in the same DONE block, then `errors=2`.
    assert "hits=0" in err, "hits count missing from DONE line: " + err
    assert "misses=1" in err, "misses count missing from DONE line: " + err
    assert "errors=2" in err, "errors count missing from DONE line: " + err
    # Order check: hits before misses before errors (substring indices).
    hi = err.index("hits=0")
    mi = err.index("misses=1")
    ei_ = err.index("errors=2")
    assert hi < mi < ei_, (
        "DONE summary must list hits / misses / errors in that order; "
        f"saw indices {hi}/{mi}/{ei_} in: {err}"
    )
