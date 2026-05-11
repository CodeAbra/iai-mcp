from __future__ import annotations

import argparse
import json
import subprocess


def _args(target: str = "codex") -> argparse.Namespace:
    return argparse.Namespace(target=target)


def test_codex_capture_hook_install_status_uninstall(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))

    from iai_mcp.cli import (
        cmd_capture_hooks_install,
        cmd_capture_hooks_status,
        cmd_capture_hooks_uninstall,
    )

    assert cmd_capture_hooks_install(_args()) == 0
    out = capsys.readouterr().out
    assert "Codex: installed" in out

    hook_path = tmp_path / ".codex" / "hooks" / "iai-mcp-codex-session-capture.sh"
    hooks_json = tmp_path / ".codex" / "hooks.json"
    assert hook_path.exists()

    data = json.loads(hooks_json.read_text())
    stop_entries = data["hooks"]["Stop"]
    commands = [
        hook["command"] for entry in stop_entries for hook in entry.get("hooks", [])
    ]
    assert any("iai-mcp-codex-session-capture.sh" in command for command in commands)
    assert "codex_hooks" not in hooks_json.read_text()

    assert cmd_capture_hooks_status(_args()) == 0
    assert "status: ACTIVE" in capsys.readouterr().out

    assert cmd_capture_hooks_uninstall(_args()) == 0
    assert not hook_path.exists()
    data = json.loads(hooks_json.read_text())
    assert "Stop" not in data.get("hooks", {})


def test_codex_capture_hook_uninstall_preserves_unrelated_stop_hooks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    hooks_json = tmp_path / ".codex" / "hooks.json"
    hooks_json.parent.mkdir(parents=True)
    hooks_json.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash ~/.codex/hooks/keep.sh",
                                }
                            ]
                        },
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash ~/.codex/hooks/iai-mcp-codex-session-capture.sh",
                                }
                            ]
                        },
                    ]
                }
            }
        )
    )

    from iai_mcp.cli import cmd_capture_hooks_uninstall

    assert cmd_capture_hooks_uninstall(_args()) == 0
    data = json.loads(hooks_json.read_text())
    stop_entries = data["hooks"]["Stop"]
    commands = [
        hook["command"] for entry in stop_entries for hook in entry.get("hooks", [])
    ]
    assert commands == ["bash ~/.codex/hooks/keep.sh"]


def test_codex_stop_hook_stdout_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    from iai_mcp.cli import cmd_capture_hooks_install

    assert cmd_capture_hooks_install(_args()) == 0
    hook_path = tmp_path / ".codex" / "hooks" / "iai-mcp-codex-session-capture.sh"

    payload = {
        "cwd": str(tmp_path),
        "hook_event_name": "Stop",
        "last_assistant_message": None,
        "model": "gpt-test",
        "permission_mode": "default",
        "session_id": "test-session",
        "stop_hook_active": False,
        "transcript_path": None,
        "turn_id": "turn-1",
    }
    proc = subprocess.run(
        ["bash", str(hook_path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert proc.returncode == 0
    assert proc.stdout == ""
