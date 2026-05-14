"""Detect-only reconciliation between canonical and legacy daemon-state files.

The canonical lifecycle state lives in ``~/.iai-mcp/lifecycle_state.json``
with values WAKE / DROWSY / SLEEP / HIBERNATION. The legacy
file ``~/.iai-mcp/.daemon-state.json`` retains historical values
WAKE / TRANSITIONING / SLEEP / DREAMING.

``reconcile_fsm_state`` reads both files (if present) and reports whether
their declared lifecycle states disagree. It is a pure read — no writes,
no logging, no event emission; the caller decides how to surface drift.

Mapping table (no drift):
  canonical=WAKE        <-> legacy=WAKE         (steady)
  canonical=SLEEP       <-> legacy=SLEEP        (steady)
  canonical=SLEEP       <-> legacy=DREAMING     (DREAMING is a sub-state)
  canonical=DROWSY      <-> legacy=TRANSITIONING
  canonical=HIBERNATION <-> any legacy value    (legacy did not model it)

Any other combo => drift=True.
Both files missing or one side missing => drift=False (one-sided is not
drift; it is the bootstrapping condition).
"""
from __future__ import annotations

import json
from pathlib import Path


def _write_canonical(path: Path, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_state": state,
        "since_ts": "2026-05-13T00:00:00+00:00",
        "last_activity_ts": "2026-05-13T00:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
    }
    path.write_text(json.dumps(payload))


def _write_legacy(path: Path, fsm_state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fsm_state": fsm_state}
    path.write_text(json.dumps(payload))


def test_reconcile_detects_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    _write_canonical(canonical, "WAKE")
    _write_legacy(legacy, "SLEEP")

    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)

    assert report == {"canonical": "WAKE", "legacy": "SLEEP", "drift": True}


def test_no_drift_returns_drift_false(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"

    _write_canonical(canonical, "WAKE")
    _write_legacy(legacy, "WAKE")
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report["drift"] is False
    assert report["canonical"] == "WAKE"
    assert report["legacy"] == "WAKE"


def test_both_files_missing_returns_no_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report == {"canonical": None, "legacy": None, "drift": False}


def test_one_side_missing_returns_no_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    _write_canonical(canonical, "SLEEP")
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report["drift"] is False
    assert report["canonical"] == "SLEEP"
    assert report["legacy"] is None


def test_dreaming_maps_to_sleep_no_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    _write_canonical(canonical, "SLEEP")
    _write_legacy(legacy, "DREAMING")
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report["drift"] is False


def test_transitioning_maps_to_drowsy_no_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    _write_canonical(canonical, "DROWSY")
    _write_legacy(legacy, "TRANSITIONING")
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report["drift"] is False


def test_hibernation_accepts_any_legacy_no_drift(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    for legacy_state in ("WAKE", "TRANSITIONING", "SLEEP", "DREAMING"):
        _write_canonical(canonical, "HIBERNATION")
        _write_legacy(legacy, legacy_state)
        report = reconcile_fsm_state(
            canonical_path=canonical, legacy_path=legacy
        )
        assert report["drift"] is False, (legacy_state, report)


def test_corrupt_file_treated_as_missing(tmp_path):
    from iai_mcp.fsm_reconcile import reconcile_fsm_state

    canonical = tmp_path / "lifecycle_state.json"
    legacy = tmp_path / ".daemon-state.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("not valid json {{{")
    _write_legacy(legacy, "SLEEP")
    report = reconcile_fsm_state(canonical_path=canonical, legacy_path=legacy)
    assert report["drift"] is False
    assert report["canonical"] is None
    assert report["legacy"] == "SLEEP"
