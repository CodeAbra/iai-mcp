"""Contract — bank/recent encrypted append + retention sweep.

Five properties under test:

  1. ``append_recent_record(store, rec, now=...)`` creates
     ``~/.iai-mcp/.memory-bank/recent/window-YYYY-MM-DD.jsonl`` (UTC date
     from ``now``) with file mode ``0o600``, parent dir mode ``0o700``,
     and writes exactly one newline-terminated line.

  2. Each appended line is a single ``iai:enc:v1:`` ciphertext that
     decrypts under ``store._key()`` + ``store._ad(rec.id)`` to a UTF-8
     JSON object with exactly six keys
     ``{id, text, embedding_b64, tier, ts, role}``. ``embedding_b64``
     round-trips to the original float32 embedding byte-for-byte. AAD
     binding is active — a wrong AAD raises ``InvalidTag``.

  3. ``append_recent_record`` is concurrency-safe under a module-level
     ``threading.Lock`` — 4 threads each appending 10 distinct records
     produce exactly 40 decryptable lines with no torn writes and no
     duplicate or missing record ids.

  4. ``prune_recent_windows(keep_days=30, now=...)`` unlinks window
     files whose filename-date is older than ``now - keep_days``,
     leaves newer files alone, skips malformed filenames silently,
     and is idempotent.

  5. Integration smoke: ``drain_deferred_captures`` mirrors every
     ``status='inserted'`` record into the bank/recent window file
     for today's UTC date in addition to writing to LanceDB.

  6. Cold reader: AAD derives purely from the window filename,
     no prior knowledge of record ids is required to decrypt.
"""
from __future__ import annotations

import base64
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from iai_mcp.capture import drain_deferred_captures
from iai_mcp.crypto import decrypt_field, encrypt_field, is_encrypted
from iai_mcp.memory_bank import append_recent_record, prune_recent_windows
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


def _write_dummy_window_file(path: Path, store: MemoryStore, rec_id: UUID) -> None:
    """Write a single legit ciphertext line into a window file."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    payload = {"id": str(rec_id), "text": "dummy"}
    ct = encrypt_field(
        json.dumps(payload, separators=(",", ":")),
        store._key(),
        associated_data=store._ad(rec_id),
    )
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (ct + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def test_recent_append_creates_dated_window_file(iai_home):
    store = MemoryStore()
    rec = _make_record(embed_dim=store.embed_dim, text="hello world")
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)

    append_recent_record(store, rec, now=fixed_now)

    target = _recent_dir(iai_home) / "window-2026-05-13.jsonl"
    assert target.exists(), f"expected window file at {target}"

    file_mode = stat.S_IMODE(os.stat(target).st_mode)
    parent_mode = stat.S_IMODE(os.stat(target.parent).st_mode)
    assert file_mode == 0o600, f"file mode = 0o{file_mode:o}, expected 0o600"
    assert parent_mode == 0o700, f"parent mode = 0o{parent_mode:o}, expected 0o700"

    body = target.read_text(encoding="utf-8")
    lines = body.splitlines(keepends=True)
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}"
    assert lines[0].endswith("\n"), "line must be newline-terminated"


def test_recent_append_format_is_iai_enc_v1(iai_home):
    store = MemoryStore()
    rec = _make_record(embed_dim=store.embed_dim, text="hello world")
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)

    append_recent_record(store, rec, now=fixed_now)

    target = _recent_dir(iai_home) / "window-2026-05-13.jsonl"
    raw_line = target.read_text(encoding="utf-8").rstrip("\n")

    assert raw_line.startswith("iai:enc:v1:"), "ciphertext must carry version prefix"
    assert is_encrypted(raw_line), "is_encrypted() guard must accept the line"

    window_aad = b"2026-05-13"
    plaintext = decrypt_field(raw_line, store._key(), associated_data=window_aad)
    obj = json.loads(plaintext)

    expected_keys = {"id", "text", "embedding_b64", "tier", "ts", "role"}
    assert set(obj.keys()) == expected_keys, (
        f"schema mismatch: got {set(obj.keys())}, expected {expected_keys}"
    )

    assert obj["id"] == str(rec.id)
    assert obj["text"] == "hello world"
    assert obj["tier"] == "episodic"
    assert obj["role"] == "user"

    emb_bytes = base64.b64decode(obj["embedding_b64"])
    assert len(emb_bytes) == store.embed_dim * 4, (
        f"embedding_b64 should decode to {store.embed_dim * 4} bytes, "
        f"got {len(emb_bytes)}"
    )

    round_trip = np.frombuffer(emb_bytes, dtype=np.float32).tolist()
    assert round_trip == rec.embedding, "float32 embedding must round-trip byte-for-byte"

    with pytest.raises(InvalidTag):
        decrypt_field(raw_line, store._key(), associated_data=b"wrong-aad")
    with pytest.raises(InvalidTag):
        decrypt_field(raw_line, store._key(), associated_data=store._ad(rec.id))
    with pytest.raises(InvalidTag):
        decrypt_field(raw_line, store._key(), associated_data=b"2026-05-12")


def test_recent_append_serializes_appends_under_concurrency(iai_home):
    store = MemoryStore()
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)
    embed_dim = store.embed_dim

    records: list[MemoryRecord] = [
        _make_record(embed_dim=embed_dim, text=f"thread-record-{i}", rec_id=uuid4())
        for i in range(40)
    ]
    expected_ids = {str(r.id) for r in records}

    def _append_batch(batch: list[MemoryRecord]) -> None:
        for r in batch:
            append_recent_record(store, r, now=fixed_now)

    batches = [records[i : i + 10] for i in range(0, 40, 10)]
    assert len(batches) == 4 and all(len(b) == 10 for b in batches)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_append_batch, b) for b in batches]
        for f in futures:
            f.result()

    target = _recent_dir(iai_home) / "window-2026-05-13.jsonl"
    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 40, f"expected 40 lines, got {len(lines)}"
    for ln in lines:
        assert ln.startswith("iai:enc:v1:"), "torn write detected — line missing prefix"

    window_aad = b"2026-05-13"
    key = store._key()
    seen_ids: set[str] = set()

    for ln in lines:
        plaintext = decrypt_field(ln, key, associated_data=window_aad)
        obj = json.loads(plaintext)
        seen_ids.add(obj["id"])

    assert seen_ids == expected_ids, (
        f"missing {expected_ids - seen_ids} or extras {seen_ids - expected_ids}"
    )


def test_recent_prune_deletes_files_older_than_keep_days(iai_home):
    store = MemoryStore()
    recent = _recent_dir(iai_home)
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)

    old_file = recent / "window-2025-01-01.jsonl"
    one_day = recent / "window-2026-05-12.jsonl"
    today = recent / "window-2026-05-13.jsonl"

    rid1, rid2, rid3 = uuid4(), uuid4(), uuid4()
    _write_dummy_window_file(old_file, store, rid1)
    _write_dummy_window_file(one_day, store, rid2)
    _write_dummy_window_file(today, store, rid3)

    deleted = prune_recent_windows(keep_days=30, now=fixed_now)
    assert deleted == 1, f"expected 1 deletion, got {deleted}"
    assert not old_file.exists(), "old window file should be unlinked"
    assert one_day.exists(), "1-day-old window must survive 30-day retention"
    assert today.exists(), "today's window must survive"

    deleted_again = prune_recent_windows(keep_days=30, now=fixed_now)
    assert deleted_again == 0

    bogus_a = recent / "notawindow.jsonl"
    bogus_b = recent / "window-bogus.jsonl"
    bogus_a.write_text("garbage")
    bogus_b.write_text("garbage")
    os.chmod(bogus_a, 0o600)
    os.chmod(bogus_b, 0o600)

    deleted_third = prune_recent_windows(keep_days=30, now=fixed_now)
    assert deleted_third == 0, "malformed filenames must not trigger deletion"
    assert bogus_a.exists() and bogus_b.exists(), "malformed filenames must be skipped"


def test_drain_writes_to_lancedb_and_bank_recent(iai_home):
    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    deferred_file = deferred_dir / "test-session.jsonl"
    header = {"version": 1, "session_id": "test-session"}
    event = {
        "text": "integration smoke text",
        "cue": "test",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-13T12:00:00+00:00",
    }
    deferred_file.write_text(
        json.dumps(header, separators=(",", ":")) + "\n"
        + json.dumps(event, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    store = MemoryStore()
    counts = drain_deferred_captures(store)

    assert counts["events_inserted"] == 1, f"counts={counts}"
    assert counts["files_drained"] == 1, f"counts={counts}"
    assert not deferred_file.exists(), "drained file must be unlinked"

    records = store.all_records()
    matching = [r for r in records if r.literal_surface == "integration smoke text"]
    assert len(matching) >= 1, "drain must persist the record in LanceDB"

    recent = _recent_dir(iai_home)
    windows = list(recent.glob("window-*.jsonl"))
    assert len(windows) == 1, (
        f"expected exactly one bank/recent window file, got {len(windows)}: "
        f"{[w.name for w in windows]}"
    )

    window_file = windows[0]
    lines = [ln for ln in window_file.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1, f"expected 1 line in {window_file.name}, got {len(lines)}"

    name = window_file.name
    date_part = name[len("window-") : -len(".jsonl")]
    window_aad = date_part.encode("utf-8")
    plaintext = decrypt_field(lines[0], store._key(), associated_data=window_aad)
    obj = json.loads(plaintext)
    assert obj["text"] == "integration smoke text"
    inserted = matching[0]
    assert obj["id"] == str(inserted.id)


def test_recent_append_decrypts_without_knowing_record_id(iai_home):
    """A cold reader must be able to decrypt each window line by deriving the
    AAD from the filename alone -- never from a record id."""
    store = MemoryStore()
    rec = _make_record(embed_dim=store.embed_dim, text="cold reader payload")
    fixed_now = datetime(2026, 5, 13, tzinfo=timezone.utc)

    append_recent_record(store, rec, now=fixed_now)

    recent = _recent_dir(iai_home)
    windows = list(recent.glob("window-*.jsonl"))
    assert len(windows) == 1, f"expected one window file, got {len(windows)}"
    window_file = windows[0]

    name = window_file.name
    date_part = name[len("window-") : -len(".jsonl")]
    window_aad = date_part.encode("utf-8")

    raw_line = window_file.read_text(encoding="utf-8").rstrip("\n")
    assert raw_line.startswith("iai:enc:v1:"), "envelope prefix missing"

    plaintext = decrypt_field(raw_line, store._key(), associated_data=window_aad)
    obj = json.loads(plaintext)

    assert obj["text"] == "cold reader payload"
    assert obj["id"] == str(rec.id)

    with pytest.raises(InvalidTag):
        decrypt_field(raw_line, store._key(), associated_data=store._ad(rec.id))
