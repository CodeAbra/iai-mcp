"""Processed memory-bank writers.

Writes denormalized read-side caches under ~/.iai-mcp/.memory-bank/processed/.
Read-side decoupled from write-side: the live LanceDB store remains the
authoritative writer; this module produces stable read-side snapshots for
cache-warmer and fallback paths.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKey, decrypt_field, encrypt_field


log = logging.getLogger(__name__)


# Process-local lock serializes concurrent appends to the current window file.
# POSIX O_APPEND is atomic per-write only for writes <= PIPE_BUF (~4 KB on
# Linux/macOS); encrypted JSONL lines for long literal_surface exceed that.
# A module-level Lock keeps every append from a single daemon process
# ordered and crash-consistent. Cross-process append concurrency is out of
# scope (single-daemon writer model).
_RECENT_APPEND_LOCK = threading.Lock()

_RECENT_WINDOW_PREFIX = "window-"
_RECENT_WINDOW_SUFFIX = ".jsonl"
_RECENT_KEEP_DAYS_DEFAULT = 30


def write_processed_salience_top_n(store: Any, n: int = 1000) -> None:
    """Write the top-N records by runtime-graph salience to a JSONL cache.

    Best-effort: any exception inside the body is logged at WARNING and
    swallowed. The caller (REM-completion handler in the daemon) must
    never crash because of writer failure.

    Parameters
    ----------
    store
        Anything that exposes ``embed_dim`` (int) and ``iter_records()``
        yielding records with attributes ``id``, ``tier``,
        ``literal_surface``, ``embedding`` (sequence of floats), and
        ``created_at`` (timezone-aware datetime).
    n
        Maximum number of records to write. When fewer valid records
        exist the file holds all available records without padding.
    """
    try:
        from iai_mcp import retrieve

        embed_dim = int(store.embed_dim)
        graph, _assignment, _rc = retrieve.build_runtime_graph(store)

        centrality_by_id: dict[str, float] = {
            str(nid): float(attrs.get("centrality", 0.0) or 0.0)
            for nid, attrs in graph._nx.nodes(data=True)
        }

        processed_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(processed_dir, 0o700)
        target = processed_dir / "salience-top-N.jsonl"

        entries: list[tuple[float, dict[str, Any]]] = []
        for rec in store.iter_records():
            embedding = getattr(rec, "embedding", None)
            if embedding is None or len(embedding) != embed_dim:
                actual_len = -1 if embedding is None else len(embedding)
                log.warning(
                    "skipping record %s: embedding dim %s != store.embed_dim %s",
                    getattr(rec, "id", "<unknown-id>"),
                    actual_len,
                    embed_dim,
                )
                continue

            try:
                emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
            except (TypeError, ValueError) as exc:
                log.warning(
                    "skipping record %s: embedding not float-coercible (%s)",
                    getattr(rec, "id", "<unknown-id>"),
                    exc,
                )
                continue

            embedding_b64 = base64.b64encode(emb_bytes).decode("ascii")
            salience = centrality_by_id.get(str(rec.id), 0.0)
            entries.append(
                (
                    salience,
                    {
                        "id": str(rec.id),
                        "text": (rec.literal_surface or "")[:200],
                        "embedding_b64": embedding_b64,
                        "tier": rec.tier,
                        "ts": rec.created_at.isoformat(),
                        "salience": float(salience),
                    },
                )
            )

        entries.sort(key=lambda pair: pair[0], reverse=True)
        top = entries[: max(0, int(n))]
        payload_lines = [
            json.dumps(line, separators=(",", ":")) for _s, line in top
        ]
        body = "\n".join(payload_lines)
        if payload_lines:
            body += "\n"

        fd, tmp = tempfile.mkstemp(
            prefix=".salience-top-n.",
            suffix=".tmp",
            dir=str(processed_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning("processed salience writer failed: %s", exc)


def append_recent_record(
    store: Any,
    record: Any,
    *,
    now: datetime | None = None,
) -> None:
    """Append one AES-256-GCM encrypted JSONL line to today's window file.

    Schema parallel to the processed writer minus ``salience`` (real
    centrality is only known at REM-time; drain time has nothing meaningful
    to write there, so the key is omitted to keep the schema honest).

    Parameters
    ----------
    store
        Live ``MemoryStore``. Used for ``_key()`` (32-byte AES key). AAD
        is bound to the window filename's date string (UTF-8 bytes of
        ``YYYY-MM-DD``) so that a cold reader can derive AAD from the
        filename without first having to decrypt a record id. Anti-swap
        protection is per-day: ciphertext cannot be moved between window
        files without GCM tag invalidation; within the same day, line
        order is not authenticated (acceptable under the local
        single-user threat model).
    record
        Just-inserted ``MemoryRecord``. Must carry: ``id`` (UUID),
        ``literal_surface`` (str), ``embedding`` (sequence of floats with
        length ``store.embed_dim``), ``tier`` (str), ``created_at``
        (timezone-aware datetime), ``provenance`` (list[dict] whose last
        entry may hold ``role``).
    now
        UTC datetime; if ``None``, defaults to ``datetime.now(timezone.utc)``.
        Tests inject a deterministic clock; production passes ``None``.
    """
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    recent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(recent_dir, 0o700)

    ts = now or datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    target = recent_dir / f"{_RECENT_WINDOW_PREFIX}{date_str}{_RECENT_WINDOW_SUFFIX}"
    window_aad = date_str.encode("utf-8")

    role = "user"
    if record.provenance:
        role = record.provenance[-1].get("role", "user") or "user"

    emb_bytes = np.asarray(record.embedding, dtype=np.float32).tobytes()
    embedding_b64 = base64.b64encode(emb_bytes).decode("ascii")

    payload = {
        "id": str(record.id),
        "text": record.literal_surface,
        "embedding_b64": embedding_b64,
        "tier": record.tier,
        "ts": record.created_at.isoformat(),
        "role": role,
    }
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        associated_data=window_aad,
    )

    line = (ciphertext + "\n").encode("utf-8")

    with _RECENT_APPEND_LOCK:
        fd = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)


def prune_recent_windows(
    *,
    keep_days: int = _RECENT_KEEP_DAYS_DEFAULT,
    now: datetime | None = None,
) -> int:
    """Unlink window files whose filename-date is older than now - keep_days.

    Filename-date is canonical (NOT mtime -- a backup tool could touch the
    file's mtime and break retention). Files that don't match the
    ``window-YYYY-MM-DD.jsonl`` shape are skipped silently. Returns the
    count of files unlinked. Caller (drain) wraps in try/except.
    """
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    if not recent_dir.exists():
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=keep_days)
    cutoff_date = cutoff.date()
    deleted = 0
    for fpath in recent_dir.iterdir():
        if not fpath.is_file():
            continue
        name = fpath.name
        if not (
            name.startswith(_RECENT_WINDOW_PREFIX)
            and name.endswith(_RECENT_WINDOW_SUFFIX)
        ):
            continue
        date_part = name[len(_RECENT_WINDOW_PREFIX) : -len(_RECENT_WINDOW_SUFFIX)]
        try:
            file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff_date:
            try:
                fpath.unlink()
                deleted += 1
            except OSError:
                log.warning("prune_recent_windows: failed to unlink %s", fpath)
    return deleted


def read_processed_records() -> Iterator[dict[str, Any]]:
    """Yield each JSON line from the processed-tier file as a dict.

    Reads ``~/.iai-mcp/.memory-bank/processed/salience-top-N.jsonl``.
    Returns an empty iterator if the file does not exist. Malformed
    lines are skipped silently with a WARNING log.
    """
    processed_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "processed"
    target = processed_dir / "salience-top-N.jsonl"
    if not target.exists():
        return
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("read_processed_records: cannot read %s: %s", target, exc)
        return
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning(
                "read_processed_records: skipping malformed line %d: %s",
                line_no,
                exc,
            )
            continue
        if not isinstance(obj, dict):
            log.warning(
                "read_processed_records: skipping non-object line %d",
                line_no,
            )
            continue
        yield obj


def read_recent_records(*, key: bytes | None = None) -> Iterator[dict[str, Any]]:
    """Yield each decrypted JSON line from every window-*.jsonl file.

    Iterates ``~/.iai-mcp/.memory-bank/recent/window-YYYY-MM-DD.jsonl``
    files in newest-first filename order. The AAD passed to
    ``decrypt_field`` is the window-file's date portion encoded as
    UTF-8 bytes -- matching the write-side at ``append_recent_record``.

    When ``key`` is None, the master key is derived lazily via
    ``CryptoKey(user_id="default").get_or_create()`` so that the CLI
    handler can call this helper without an open MemoryStore.

    Lines that fail to decrypt (InvalidTag), fail to parse as JSON, or
    are not iai:enc:v1:-prefixed are skipped silently with a WARNING
    log. The read path is best-effort.
    """
    recent_dir = Path.home() / ".iai-mcp" / ".memory-bank" / "recent"
    if not recent_dir.exists():
        return

    resolved_key = key
    if resolved_key is None:
        ck = CryptoKey(user_id="default")
        resolved_key = ck.get_or_create()

    candidates: list[tuple[Path, bytes]] = []
    for fpath in recent_dir.iterdir():
        if not fpath.is_file():
            continue
        name = fpath.name
        if not (
            name.startswith(_RECENT_WINDOW_PREFIX)
            and name.endswith(_RECENT_WINDOW_SUFFIX)
        ):
            continue
        date_part = name[len(_RECENT_WINDOW_PREFIX) : -len(_RECENT_WINDOW_SUFFIX)]
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        candidates.append((fpath, date_part.encode("utf-8")))

    candidates.sort(key=lambda pair: pair[0].name, reverse=True)

    for fpath, window_aad in candidates:
        try:
            text = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("read_recent_records: cannot read %s: %s", fpath, exc)
            continue
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                plaintext = decrypt_field(
                    line, resolved_key, associated_data=window_aad
                )
            except (InvalidTag, ValueError) as exc:
                log.warning(
                    "read_recent_records: skipping line %d in %s: %s",
                    line_no,
                    fpath.name,
                    exc,
                )
                continue
            try:
                obj = json.loads(plaintext)
            except json.JSONDecodeError as exc:
                log.warning(
                    "read_recent_records: bad JSON at line %d in %s: %s",
                    line_no,
                    fpath.name,
                    exc,
                )
                continue
            if not isinstance(obj, dict):
                log.warning(
                    "read_recent_records: skipping non-object line %d in %s",
                    line_no,
                    fpath.name,
                )
                continue
            yield obj


def bank_recall_substring(
    query: str,
    limit: int = 20,
    *,
    include_processed: bool = True,
    include_recent: bool = True,
    key: bytes | None = None,
) -> dict[str, Any]:
    """Case-insensitive substring scan over the processed + recent tiers.

    Returns a dict that mirrors the daemon's memory_recall response
    shape so the wrapper's bank-fallback path is wire-compatible:

        {
            "hits": [...],           # ranked: processed then recent
            "anti_hits": [],
            "activation_trace": [],
            "budget_used": 0,
            "cue_mode": "verbatim",
            "patterns_observed": [],
            "_knobs_applied": {},
        }

    Ranking
    -------
    1. Processed-tier hits come before recent-tier hits.
    2. Inside processed: ``salience`` DESC.
    3. Inside recent: ``ts`` DESC (newest first).
    4. Truncate to ``limit`` after sorting.
    """
    needle = (query or "").lower()
    processed_hits: list[tuple[float, dict[str, Any]]] = []
    recent_hits: list[tuple[str, dict[str, Any]]] = []

    if include_processed:
        for row in read_processed_records():
            text = str(row.get("text", "") or "")
            if needle and needle not in text.lower():
                continue
            try:
                salience = float(row.get("salience", 0.0) or 0.0)
            except (TypeError, ValueError):
                salience = 0.0
            hit = {
                "record_id": str(row.get("id", "")),
                "score": salience,
                "reason": "bank-substring-match (processed)",
                "literal_surface": text,
                "adjacent_suggestions": [],
                "valid_from": None,
                "valid_to": None,
            }
            processed_hits.append((salience, hit))

    if include_recent:
        for row in read_recent_records(key=key):
            text = str(row.get("text", "") or "")
            if needle and needle not in text.lower():
                continue
            ts = str(row.get("ts", "") or "")
            hit = {
                "record_id": str(row.get("id", "")),
                "score": 0.0,
                "reason": "bank-substring-match (recent)",
                "literal_surface": text,
                "adjacent_suggestions": [],
                "valid_from": None,
                "valid_to": None,
            }
            recent_hits.append((ts, hit))

    processed_hits.sort(key=lambda pair: pair[0], reverse=True)
    recent_hits.sort(key=lambda pair: pair[0], reverse=True)

    ordered = [h for _s, h in processed_hits] + [h for _t, h in recent_hits]
    capped = ordered[: max(0, int(limit))]

    return {
        "hits": capped,
        "anti_hits": [],
        "activation_trace": [],
        "budget_used": 0,
        "cue_mode": "verbatim",
        "patterns_observed": [],
        "_knobs_applied": {},
    }
