"""Pure-read drift detection between canonical and legacy FSM state files.

The canonical lifecycle state machine (WAKE / DROWSY / SLEEP / HIBERNATION)
persists to ``~/.iai-mcp/lifecycle_state.json``. The historical
``~/.iai-mcp/.daemon-state.json`` carries an ``fsm_state`` field with the
older vocabulary (WAKE / TRANSITIONING / SLEEP / DREAMING).

Both files coexist during the migration window. ``reconcile_fsm_state``
compares the two and returns whether the declared states agree according
to the documented mapping. It is a read-only diagnostic -- the caller
decides how to surface a drift report (log, event emission, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

_NO_DRIFT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("WAKE", "WAKE"),
        ("SLEEP", "SLEEP"),
        ("SLEEP", "DREAMING"),
        ("DROWSY", "TRANSITIONING"),
    }
)


def _read_canonical(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("current_state")
    return value if isinstance(value, str) and value else None


def _read_legacy(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("fsm_state")
    return value if isinstance(value, str) and value else None


def reconcile_fsm_state(
    canonical_path: Path | None = None,
    legacy_path: Path | None = None,
) -> dict[str, str | bool | None]:
    """Return a drift report comparing the canonical and legacy state files.

    Both file paths default to the production locations under
    ``~/.iai-mcp/``. Pure read: never writes, never logs, never emits
    events. Corrupt or unreadable files are treated as missing.

    Returns a dict with keys:
      * ``canonical`` -- the canonical lifecycle state name, or ``None``
        when the file is absent / unreadable / malformed.
      * ``legacy`` -- the legacy ``fsm_state`` value, or ``None`` likewise.
      * ``drift`` -- ``True`` only when both sides are populated and the
        (canonical, legacy) pair is not in the no-drift mapping table.
    """
    if canonical_path is None:
        from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH

        canonical_path = LIFECYCLE_STATE_PATH
    if legacy_path is None:
        from iai_mcp.daemon_state import STATE_PATH

        legacy_path = STATE_PATH

    canonical = _read_canonical(canonical_path)
    legacy = _read_legacy(legacy_path)

    if canonical is None or legacy is None:
        drift = False
    elif canonical == "HIBERNATION":
        drift = False
    else:
        drift = (canonical, legacy) not in _NO_DRIFT_PAIRS

    return {"canonical": canonical, "legacy": legacy, "drift": drift}
