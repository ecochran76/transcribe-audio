from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_intelligence_ledger


def test_create_run_writes_host_owned_ledger(tmp_path: Path) -> None:
    payload = app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="context-reread",
        purpose="Prepare a supervised contextual reread.",
        document_id="doc_123",
        run_id="test-run",
    )

    run = payload["run"]
    run_dir = tmp_path / "app-intelligence-runs" / "test-run"

    assert run["schema_version"] == app_intelligence_ledger.SCHEMA_VERSION
    assert run["provider"] == "codex-app-server"
    assert run["phase"] == "prepared"
    assert run["policy"]["host_owns_control_flow"] is True
    assert run["policy"]["structured_decisions_required"] is True
    assert run["policy"]["remote_transport"] == "forbidden_without_auth_review"
    assert "start_app_server_session" in run["policy"]["allowed_actions"]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "branches").is_dir()
    assert payload["events"][0]["event_type"] == "run_prepared"


def test_list_and_show_runs_are_user_scoped(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="speaker-review",
        purpose="Prepare speaker disambiguation.",
        run_id="speaker-run",
    )

    listing = app_intelligence_ledger.list_runs(state_root=tmp_path)
    shown = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="speaker-run")

    assert listing["total"] == 1
    assert listing["items"][0]["run_id"] == "speaker-run"
    assert shown["run"]["workflow"] == "speaker-review"


def test_cli_create_outputs_json(tmp_path: Path, capsys) -> None:
    exit_code = app_intelligence_ledger.main(
        [
            "--state-dir",
            str(tmp_path),
            "create",
            "--workflow",
            "memory-review",
            "--purpose",
            "Prepare memory harvest review.",
            "--run-id",
            "memory-run",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["run"]["run_id"] == "memory-run"
