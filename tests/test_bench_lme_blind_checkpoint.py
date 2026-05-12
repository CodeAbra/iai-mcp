"""D2 — checkpoint disposition flags (--resume / --fresh) + auto-clean default.

Background: today, after a failed run (e.g. crypto trap from D1), the
harness checkpoints the errored rows to ``<out>.jsonl``. The next
invocation silently re-skips those rows, so the zero persists across
retries. The user has to manually ``rm <out>.jsonl`` to recover.

This file pins five contracts:

D2.1 — auto-clean by default when prior errors present:
    1. ``test_checkpoint_auto_cleans_when_prior_errors`` — pre-existing
       checkpoint with at least one ERROR row, no flag => auto-clean and
       restart, with the verbatim stderr phrase.

D2.2 — --resume keeps checkpoint despite errors:
    2. ``test_resume_flag_keeps_checkpoint_with_errors`` — same precondition
       + --resume => existing behaviour preserved (no auto-clean).

D2.3 — --fresh force-cleans clean checkpoint:
    3. ``test_fresh_flag_force_cleans_clean_checkpoint`` — checkpoint with
       only SUCCESS rows + --fresh => force-clean and restart.

Plus argparse contract checks:
    4. ``test_fresh_and_resume_mutually_exclusive`` — passing both flags
       triggers argparse's mutually-exclusive-group error.
    5. ``test_default_behavior_clean_checkpoint_no_errors_keeps_it`` —
       no errors + no --fresh + no --resume => checkpoint preserved
       (auto-clean does NOT fire on a clean checkpoint).

Adapter calls stubbed via `_patch_adapter` (no HF network access).
"""
from __future__ import annotations

import json
import os

import pytest


# --------------------------------------------------------------------------- #
# Shared mocking helpers — duplicated from
# tests/test_bench_lme_blind_preflight.py per the plan note ("duplication
# is fine for a quick-mode plan").
# --------------------------------------------------------------------------- #


class _StubLMESession:
    """Minimal stand-in for bench.adapters.longmemeval.LMESession."""

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
        self.turns = []


def _patch_adapter(monkeypatch, qids: list[str] | None = None) -> None:
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


# Convenience: build a SUCCESS-shaped checkpoint line consistent with the
# post-Task-1 on-disk format. Mirrors the dict that
# `bench.longmemeval_blind._run_one_row` returns plus the spliced
# `classification: "SUCCESS"` field.
def _success_row(qid: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "test",
        "classification": "SUCCESS",
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


def _error_row(qid: str) -> dict:
    return {
        "question_id": qid,
        "question_type": "test",
        "classification": "ERROR",
        "error": {
            "error_class": "RuntimeError",
            "error": "synthetic",
        },
    }


# --------------------------------------------------------------------------- #
# D2.1 — auto-clean default when prior errors present
# --------------------------------------------------------------------------- #


def test_checkpoint_auto_cleans_when_prior_errors(
    tmp_path, monkeypatch, capsys
):
    """Pre-existing checkpoint with one ERROR row => auto-clean + restart.

    Verbatim stderr phrase contract:
        ``Resuming from prior run with 1 errors; starting fresh. Pass --resume to keep checkpoint.``

    The string is grepped literally by reviewer-facing tooling — do not
    "fix" the "1 errors" grammar.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_success_row("q-ok-1")) + "\n")
        f.write(json.dumps(_error_row("q-err-1")) + "\n")
    cp_size_before = cp_path.stat().st_size
    assert cp_size_before > 0, "pre-flight: precondition checkpoint nonempty"

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    expected = (
        "Resuming from prior run with 1 errors; starting fresh. "
        "Pass --resume to keep checkpoint."
    )
    assert expected in err, (
        "verbatim auto-clean phrase missing from stderr; got:\n" + err
    )

    # On-disk checkpoint either deleted OR truncated to 0 bytes.
    if cp_path.exists():
        assert cp_path.stat().st_size == 0, (
            "checkpoint must be empty after auto-clean; size="
            f"{cp_path.stat().st_size}"
        )

    # Output JSON reflects the fresh-restart state (n_rows=0 with empty adapter).
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


# --------------------------------------------------------------------------- #
# D2.2 — --resume keeps checkpoint despite errors
# --------------------------------------------------------------------------- #


def test_resume_flag_keeps_checkpoint_with_errors(
    tmp_path, monkeypatch, capsys
):
    """--resume preserves the checkpoint even when it contains ERROR rows.

    Same precondition as the auto-clean test, but with --resume on the
    command line. Expect:
      - stderr does NOT contain 'starting fresh' (the auto-clean message),
      - the existing 'resume: N rows already in checkpoint' log line fires,
      - on-disk checkpoint is byte-identical to what we wrote.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    payload = (
        json.dumps(_success_row("q-ok-1"))
        + "\n"
        + json.dumps(_error_row("q-err-1"))
        + "\n"
    )
    cp_path.write_text(payload, encoding="utf-8")

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(
        ["--limit", "0", "--out", str(out_path), "--resume"]
    )
    assert rc == 0

    err = capsys.readouterr().err
    assert "starting fresh" not in err, (
        "auto-clean must NOT fire when --resume is passed; stderr was:\n" + err
    )
    # Existing resume log line (or its post-Task-1 equivalent).
    assert "resume:" in err, (
        "expected the 'resume: N rows already in checkpoint' log line: "
        + err
    )

    # On-disk checkpoint preserved byte-for-byte.
    assert cp_path.exists()
    assert cp_path.read_text(encoding="utf-8") == payload, (
        "--resume must not mutate the checkpoint"
    )


# --------------------------------------------------------------------------- #
# D2.3 — --fresh force-cleans clean checkpoint
# --------------------------------------------------------------------------- #


def test_fresh_flag_force_cleans_clean_checkpoint(
    tmp_path, monkeypatch, capsys
):
    """--fresh force-cleans even when no ERROR rows present.

    Symmetric with the existing `rm <out>.jsonl` workaround. Documented as
    opt-in destructive in the --fresh help text.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_success_row("q-ok-1")) + "\n")
    assert cp_path.stat().st_size > 0

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(
        ["--limit", "0", "--out", str(out_path), "--fresh"]
    )
    assert rc == 0

    err = capsys.readouterr().err
    # Plan asks for `[LME] --fresh: discarding N-row checkpoint` shape.
    assert "--fresh" in err and "discarding" in err, (
        "expected --fresh force-clean log line; got:\n" + err
    )

    # Checkpoint must be gone or empty.
    if cp_path.exists():
        assert cp_path.stat().st_size == 0

    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    assert out["n_rows"] == 0


# --------------------------------------------------------------------------- #
# Argparse contract — mutual exclusion
# --------------------------------------------------------------------------- #


def test_fresh_and_resume_mutually_exclusive(tmp_path, monkeypatch, capsys):
    """--fresh and --resume cannot both be passed.

    Use argparse's add_mutually_exclusive_group; failure is a
    non-zero exit with an error message naming both flags.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    with pytest.raises(SystemExit) as ei:
        mod.main(
            [
                "--limit",
                "0",
                "--out",
                str(out_path),
                "--fresh",
                "--resume",
            ]
        )
    # argparse's mutually-exclusive-group exits with code 2.
    assert ei.value.code != 0
    err = capsys.readouterr().err
    # Tolerate either order in argparse's error formatting (--fresh /
    # --resume not allowed with ...).
    assert "--fresh" in err and "--resume" in err, (
        "mutual-exclusion error must name both flags: " + err
    )


# --------------------------------------------------------------------------- #
# Default — clean checkpoint without flag is preserved
# --------------------------------------------------------------------------- #


def test_default_behavior_clean_checkpoint_no_errors_keeps_it(
    tmp_path, monkeypatch, capsys
):
    """No errors + no --fresh + no --resume => checkpoint preserved.

    Auto-clean must trigger ONLY when prior errors are present; a clean
    SUCCESS-only checkpoint is the normal "resume an interrupted run"
    case and must keep its existing semantics.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    cp_path = tmp_path / "o.json.jsonl"
    payload = json.dumps(_success_row("q-ok-1")) + "\n"
    cp_path.write_text(payload, encoding="utf-8")

    _patch_adapter(monkeypatch, qids=[])
    out_path = tmp_path / "o.json"

    import bench.longmemeval_blind as mod

    rc = mod.main(["--limit", "0", "--out", str(out_path)])
    assert rc == 0

    err = capsys.readouterr().err
    assert "starting fresh" not in err, (
        "auto-clean must NOT fire on a clean SUCCESS-only checkpoint: " + err
    )
    assert "resume:" in err, (
        "expected the existing resume log line on clean checkpoint: " + err
    )

    # Checkpoint preserved.
    assert cp_path.exists()
    assert cp_path.read_text(encoding="utf-8") == payload
