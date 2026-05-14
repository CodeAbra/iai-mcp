"""Contract — processed memory-bank salience-top-N JSONL writer.

Four properties under test:
  1. After invocation the JSONL file exists at the expected path under
     ~/.iai-mcp/.memory-bank/processed/ with file mode 0o600, parent dir
     mode 0o700, JSON lines sorted by descending salience, line count
     bounded by min(record_count, n).
  2. Each line's base64 embedding decodes to exactly `store.embed_dim * 4`
     bytes (float32) and round-trips back to the original embedding under
     exact float32 equality.
  3. Parent directory is chmod 0o700 both on first creation and when a
     pre-existing directory has more permissive bits (umask clobber).
  4. Records with malformed embeddings (None / wrong length) are skipped
     with a WARNING-level log on the `iai_mcp.memory_bank` logger; valid
     records still land in the file in the correct count.

All tests stub out the runtime graph via `monkeypatch.setattr` and redirect
the writer's `Path.home()` via `monkeypatch.setenv("HOME", tmp_path)`.
No real `MemoryStore`, no real LanceDB I/O.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import UUID, uuid4

import numpy as np
import pytest


@dataclass
class _StubRec:
    id: UUID
    tier: str
    literal_surface: str
    embedding: list[float]
    created_at: datetime


class _StubStore:
    def __init__(self, records: list[_StubRec], embed_dim: int) -> None:
        self._records = list(records)
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def all_records(self) -> list[_StubRec]:
        return list(self._records)

    def iter_records(
        self,
        *,
        columns: list[str] | None = None,
        batch_size: int = 1024,
        where: str | None = None,
    ) -> Iterable[_StubRec]:
        yield from self._records


class _FakeNX:
    def __init__(self, centrality_by_id: dict[str, float]) -> None:
        self._data = dict(centrality_by_id)

    def nodes(self, data: bool = False):
        if data:
            return [(k, {"centrality": v}) for k, v in self._data.items()]
        return list(self._data.keys())


class _FakeGraph:
    def __init__(self, centrality_by_id: dict[str, float]) -> None:
        self._nx = _FakeNX(centrality_by_id)


def _processed_dir(home: Path) -> Path:
    return home / ".iai-mcp" / ".memory-bank" / "processed"


def _target_path(home: Path) -> Path:
    return _processed_dir(home) / "salience-top-N.jsonl"


def _make_records(count: int, *, embed_dim: int) -> list[_StubRec]:
    """Build `count` valid records with float32-exact embeddings."""
    now = datetime.now(timezone.utc)
    recs: list[_StubRec] = []
    for i in range(count):
        embedding = [float(i + j) for j in range(embed_dim)]
        recs.append(
            _StubRec(
                id=uuid4(),
                tier="semantic",
                literal_surface=f"Record number {i}",
                embedding=embedding,
                created_at=now,
            )
        )
    return recs


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw:
        return []
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


@pytest.mark.parametrize("m_records", [0, 3, 14])
def test_processed_salience_top_n_written_at_rem_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, m_records: int
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    embed_dim = 8
    records = _make_records(m_records, embed_dim=embed_dim)
    store = _StubStore(records, embed_dim=embed_dim)

    centrality_map: dict[str, float] = {}
    for idx, rec in enumerate(records):
        centrality_map[str(rec.id)] = float(m_records - idx)

    monkeypatch.setattr(
        "iai_mcp.retrieve.build_runtime_graph",
        lambda _store: (_FakeGraph(centrality_map), None, None),
    )

    from iai_mcp.memory_bank import write_processed_salience_top_n

    write_processed_salience_top_n(store, n=4)

    target = _target_path(tmp_path)
    assert target.exists(), f"expected file at {target}"

    file_mode = oct(stat.S_IMODE(os.stat(target).st_mode))
    assert file_mode == "0o600", f"file mode {file_mode} != 0o600"

    lines = _read_jsonl(target)
    expected_count = min(m_records, 4)
    assert len(lines) == expected_count, (
        f"expected {expected_count} lines, got {len(lines)}"
    )

    required_keys = {"id", "text", "embedding_b64", "tier", "ts", "salience"}
    for ln in lines:
        assert required_keys.issubset(ln.keys()), (
            f"missing keys {required_keys - set(ln.keys())} in {ln}"
        )

    saliences = [float(ln["salience"]) for ln in lines]
    assert saliences == sorted(saliences, reverse=True), (
        f"saliences not descending: {saliences}"
    )


def test_processed_salience_top_n_embedding_dimension_matches_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    embed_dim = 8
    records = _make_records(3, embed_dim=embed_dim)
    store = _StubStore(records, embed_dim=embed_dim)

    centrality_map = {str(rec.id): float(i) for i, rec in enumerate(records)}
    monkeypatch.setattr(
        "iai_mcp.retrieve.build_runtime_graph",
        lambda _store: (_FakeGraph(centrality_map), None, None),
    )

    from iai_mcp.memory_bank import write_processed_salience_top_n

    write_processed_salience_top_n(store, n=10)

    target = _target_path(tmp_path)
    lines = _read_jsonl(target)
    assert len(lines) == 3

    by_id = {ln["id"]: ln for ln in lines}
    for rec in records:
        ln = by_id[str(rec.id)]
        decoded = base64.b64decode(ln["embedding_b64"])
        assert len(decoded) == embed_dim * 4, (
            f"decoded {len(decoded)} bytes != {embed_dim * 4} (float32)"
        )
        decoded_arr = np.frombuffer(decoded, dtype=np.float32).tolist()
        expected = np.asarray(rec.embedding, dtype=np.float32).tolist()
        assert decoded_arr == expected, (
            f"round-trip mismatch for record {rec.id}: "
            f"{decoded_arr} != {expected}"
        )


@pytest.mark.parametrize("preexisting_mode", [None, 0o755])
def test_processed_parent_dir_mode_is_owner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting_mode: int | None,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    if preexisting_mode is not None:
        pdir = _processed_dir(tmp_path)
        pdir.mkdir(parents=True, exist_ok=True)
        os.chmod(pdir, preexisting_mode)
        assert oct(stat.S_IMODE(os.stat(pdir).st_mode)) == oct(preexisting_mode)

    store = _StubStore([], embed_dim=8)
    monkeypatch.setattr(
        "iai_mcp.retrieve.build_runtime_graph",
        lambda _store: (_FakeGraph({}), None, None),
    )

    from iai_mcp.memory_bank import write_processed_salience_top_n

    write_processed_salience_top_n(store, n=4)

    pdir = _processed_dir(tmp_path)
    mode_after = oct(stat.S_IMODE(os.stat(pdir).st_mode))
    assert mode_after == "0o700", (
        f"parent dir mode {mode_after} != 0o700"
    )


def test_processed_skips_bad_embedding_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    embed_dim = 8
    valid_records = _make_records(3, embed_dim=embed_dim)
    bad_rec = _StubRec(
        id=uuid4(),
        tier="semantic",
        literal_surface="bad record with empty embedding",
        embedding=[],
        created_at=datetime.now(timezone.utc),
    )
    records = list(valid_records) + [bad_rec]
    store = _StubStore(records, embed_dim=embed_dim)

    centrality_map = {str(rec.id): float(i + 1) for i, rec in enumerate(records)}
    monkeypatch.setattr(
        "iai_mcp.retrieve.build_runtime_graph",
        lambda _store: (_FakeGraph(centrality_map), None, None),
    )

    from iai_mcp.memory_bank import write_processed_salience_top_n

    with caplog.at_level(logging.WARNING, logger="iai_mcp.memory_bank"):
        write_processed_salience_top_n(store, n=10)

    target = _target_path(tmp_path)
    lines = _read_jsonl(target)
    assert len(lines) == 3, f"expected 3 valid lines, got {len(lines)}"

    written_ids = {ln["id"] for ln in lines}
    assert str(bad_rec.id) not in written_ids, (
        f"bad record {bad_rec.id} was written despite malformed embedding"
    )

    warning_records = [
        rec for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    assert warning_records, "expected at least one WARNING log record"
    matching = [
        rec
        for rec in warning_records
        if str(bad_rec.id) in rec.getMessage()
        or "dim" in rec.getMessage().lower()
    ]
    assert matching, (
        f"no WARNING referenced the bad record id or 'dim'; got: "
        f"{[r.getMessage() for r in warning_records]}"
    )
