"""Contract — bank-recall CLI substring helpers + subprocess JSON surface.

Six tests in one file:

  1. ``read_processed_records()`` yields every JSON line in
     ``~/.iai-mcp/.memory-bank/processed/salience-top-N.jsonl`` as a dict.

  2. ``read_processed_records()`` silently skips malformed JSON lines and
     keeps yielding the good ones.

  3. ``read_recent_records(key=...)`` decrypts every line in
     ``window-YYYY-MM-DD.jsonl`` using AAD derived from the filename date.

  4. ``bank_recall_substring(query, limit, key=...)`` case-insensitive
     substring-matches across processed + recent, ranks processed before
     recent (then salience DESC inside processed, ts DESC inside recent),
     and returns the 7-key top-level response shape that mirrors the
     daemon's memory_recall.

  5. ``bank_recall_substring`` honors ``limit`` and ranks by salience
     DESC inside the processed tier when more than ``limit`` records
     match.

  6. ``iai-mcp bank-recall --query <q> --limit <n> --json`` (subprocess)
     emits a JSON object to stdout with the seven top-level keys and
     exits 0.
"""
from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.crypto import CryptoKey, decrypt_field, encrypt_field, is_encrypted
from iai_mcp.memory_bank import (
    append_recent_record,
    bank_recall_substring,
    read_processed_records,
    read_recent_records,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """HOME=tmp_path + keyring fail-backend + crypto passphrase."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-recent-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))

    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _recent_dir(home: Path) -> Path:
    return home / ".iai-mcp" / ".memory-bank" / "recent"


def _processed_dir(home: Path) -> Path:
    return home / ".iai-mcp" / ".memory-bank" / "processed"


def _make_record(
    *,
    embed_dim: int,
    text: str = "hello world",
    tier: str = "episodic",
    role: str = "user",
    rec_id: UUID | None = None,
) -> MemoryRecord:
    """Build a valid MemoryRecord with a deterministic float32-exact embedding."""
    rid = rec_id if rec_id is not None else uuid4()
    embedding = np.linspace(0.0, 1.0, embed_dim).astype(np.float32).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=text,
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
        provenance=[
            {
                "ts": now.isoformat(),
                "cue": "test",
                "session_id": "s1",
                "role": role,
            }
        ],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        schema_version=SCHEMA_VERSION_CURRENT,
    )


def _write_processed(
    home: Path, rows: list[dict[str, object]], *, raw_extra: str = ""
) -> Path:
    """Write rows (one per line) into salience-top-N.jsonl."""
    pdir = _processed_dir(home)
    pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(pdir, 0o700)
    target = pdir / "salience-top-N.jsonl"
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    if raw_extra:
        body += raw_extra
    target.write_text(body, encoding="utf-8")
    os.chmod(target, 0o600)
    return target


def _make_processed_row(
    *,
    text: str,
    salience: float,
    embed_dim: int = 4,
    tier: str = "episodic",
    ts: str | None = None,
    rec_id: UUID | None = None,
) -> dict[str, object]:
    """Build one processed-tier row matching the writer schema."""
    rid = rec_id if rec_id is not None else uuid4()
    emb = np.linspace(0.0, 1.0, embed_dim).astype(np.float32).tobytes()
    return {
        "id": str(rid),
        "text": text,
        "embedding_b64": base64.b64encode(emb).decode("ascii"),
        "tier": tier,
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "salience": float(salience),
    }


def test_read_processed_records_yields_lines(iai_home):
    rows = [
        _make_processed_row(text="alpha record", salience=0.9),
        _make_processed_row(text="beta record", salience=0.5),
        _make_processed_row(text="gamma record", salience=0.1),
    ]
    _write_processed(iai_home, rows)

    out = list(read_processed_records())
    assert len(out) == 3, f"expected 3 rows, got {len(out)}"
    expected_keys = {"id", "text", "embedding_b64", "tier", "ts", "salience"}
    for d in out:
        assert set(d.keys()) == expected_keys, (
            f"schema mismatch: got {set(d.keys())}, expected {expected_keys}"
        )
    assert [d["text"] for d in out] == [
        "alpha record",
        "beta record",
        "gamma record",
    ]


def test_read_processed_records_skips_bad_lines(iai_home, caplog):
    pdir = _processed_dir(iai_home)
    pdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(pdir, 0o700)
    target = pdir / "salience-top-N.jsonl"

    good_a = _make_processed_row(text="apple", salience=0.7)
    good_b = _make_processed_row(text="banana", salience=0.5)
    good_c = _make_processed_row(text="cherry", salience=0.3)

    body = (
        json.dumps(good_a, separators=(",", ":")) + "\n"
        + "not-json\n"
        + json.dumps(good_b, separators=(",", ":")) + "\n"
        + '{"id":"u' + "\n"
        + json.dumps(good_c, separators=(",", ":")) + "\n"
    )
    target.write_text(body, encoding="utf-8")
    os.chmod(target, 0o600)

    out = list(read_processed_records())
    assert len(out) == 3, f"expected 3 valid rows, got {len(out)}"
    assert [d["text"] for d in out] == ["apple", "banana", "cherry"]


def test_read_recent_records_decrypts_with_filename_aad(iai_home):
    store = MemoryStore()
    rec = _make_record(embed_dim=store.embed_dim, text="hello carrot")
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)

    append_recent_record(store, rec, now=fixed_now)

    out = list(read_recent_records(key=store._key()))
    assert len(out) == 1, f"expected 1 row, got {len(out)}"
    obj = out[0]
    expected_keys = {"id", "text", "embedding_b64", "tier", "ts", "role"}
    assert set(obj.keys()) == expected_keys, (
        f"schema mismatch: got {set(obj.keys())}, expected {expected_keys}"
    )
    assert obj["text"] == "hello carrot"
    assert obj["id"] == str(rec.id)
    assert obj["role"] == "user"


def test_bank_recall_substring_matches_processed_and_recent(iai_home):
    store = MemoryStore()

    rows = [
        _make_processed_row(text="alpha carrot pie", salience=0.8),
        _make_processed_row(text="beta soup", salience=0.6),
    ]
    _write_processed(iai_home, rows)

    rec = _make_record(embed_dim=store.embed_dim, text="alpha grain bowl")
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)
    append_recent_record(store, rec, now=fixed_now)

    result = bank_recall_substring("alpha", limit=20, key=store._key())

    assert set(result.keys()) >= {
        "hits",
        "anti_hits",
        "activation_trace",
        "budget_used",
        "cue_mode",
        "patterns_observed",
        "_knobs_applied",
    }
    assert result["anti_hits"] == []
    assert result["cue_mode"] == "verbatim"
    assert result["_knobs_applied"] == {}
    assert result["activation_trace"] == []
    assert result["budget_used"] == 0
    assert result["patterns_observed"] == []

    hits = result["hits"]
    assert len(hits) == 2, f"expected 2 hits, got {len(hits)}"

    h0, h1 = hits
    assert h0["literal_surface"] == "alpha carrot pie", (
        f"first hit must be processed; got {h0}"
    )
    assert h0["reason"] == "bank-substring-match (processed)"
    assert h0["score"] == pytest.approx(0.8)
    assert h0["adjacent_suggestions"] == []
    assert h0["valid_from"] is None
    assert h0["valid_to"] is None
    assert set(h0.keys()) == {
        "record_id",
        "score",
        "reason",
        "literal_surface",
        "adjacent_suggestions",
        "valid_from",
        "valid_to",
    }

    assert h1["literal_surface"] == "alpha grain bowl"
    assert h1["reason"] == "bank-substring-match (recent)"
    assert h1["score"] == 0.0
    assert h1["record_id"] == str(rec.id)


def test_bank_recall_substring_respects_limit(iai_home):
    store = MemoryStore()

    rows = [
        _make_processed_row(text=f"needle row {i}", salience=float(i))
        for i in range(30)
    ]
    _write_processed(iai_home, rows)

    result = bank_recall_substring("needle", limit=5, key=store._key())
    hits = result["hits"]
    assert len(hits) == 5, f"expected 5 hits, got {len(hits)}"

    expected_scores = [29.0, 28.0, 27.0, 26.0, 25.0]
    actual_scores = [h["score"] for h in hits]
    assert actual_scores == expected_scores, (
        f"expected {expected_scores}, got {actual_scores}"
    )


def test_cli_bank_recall_emits_json_to_stdout(iai_home):
    rows = [
        _make_processed_row(text="carrot stew", salience=0.7),
        _make_processed_row(text="carrot soup", salience=0.4),
    ]
    _write_processed(iai_home, rows)

    import iai_mcp as _iai_mcp_pkg

    pkg_root = Path(_iai_mcp_pkg.__file__).resolve().parent.parent
    sub_env = {**os.environ}
    existing_pp = sub_env.get("PYTHONPATH", "")
    sub_env["PYTHONPATH"] = (
        f"{pkg_root}{os.pathsep}{existing_pp}" if existing_pp else str(pkg_root)
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "bank-recall",
            "--query",
            "carrot",
            "--limit",
            "10",
            "--json",
        ],
        capture_output=True,
        check=False,
        env=sub_env,
        text=True,
    )

    assert result.returncode == 0, (
        f"exit={result.returncode} stderr={result.stderr!r}"
    )
    payload = json.loads(result.stdout)
    expected_top = {
        "hits",
        "anti_hits",
        "activation_trace",
        "budget_used",
        "cue_mode",
        "patterns_observed",
        "_knobs_applied",
    }
    assert expected_top.issubset(set(payload.keys())), (
        f"missing keys: {expected_top - set(payload.keys())}"
    )

    hits = payload["hits"]
    assert len(hits) == 2, f"expected 2 hits, got {len(hits)}"
    expected_hit_keys = {
        "record_id",
        "score",
        "reason",
        "literal_surface",
        "adjacent_suggestions",
        "valid_from",
        "valid_to",
    }
    for h in hits:
        assert set(h.keys()) == expected_hit_keys, (
            f"hit schema mismatch: got {set(h.keys())}"
        )
