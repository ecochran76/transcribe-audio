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


def test_session_start_preflight_is_non_starting_and_can_append_event(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="app-supervisor",
        purpose="Prepare supervised work.",
        run_id="preflight-run",
    )

    dry_run = app_intelligence_ledger.session_start_preflight(
        state_root=tmp_path,
        run_id="preflight-run",
        provider_ready=True,
        provider_status="ready",
        approval_token=app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN,
    )

    assert dry_run["ok"] is True
    assert dry_run["will_start_session"] is False
    assert dry_run["will_write_event"] is False
    assert dry_run["future_required_approval_token_for_session_start"] == app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN

    appended = app_intelligence_ledger.session_start_preflight(
        state_root=tmp_path,
        run_id="preflight-run",
        provider_ready=True,
        provider_status="ready",
        approval_token=app_intelligence_ledger.SESSION_START_PREFLIGHT_EVENT_TOKEN,
        append_event_log=True,
    )
    shown = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="preflight-run")

    assert appended["event"]["event_type"] == "session_start_preflight"
    assert shown["events"][-1]["event_type"] == "session_start_preflight"
    assert shown["run"]["phase"] == "prepared"


def test_session_start_preflight_event_append_requires_event_token(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="app-supervisor",
        purpose="Prepare supervised work.",
        run_id="preflight-run",
    )

    try:
        app_intelligence_ledger.session_start_preflight(
            state_root=tmp_path,
            run_id="preflight-run",
            provider_ready=True,
            provider_status="ready",
            approval_token=app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN,
            append_event_log=True,
        )
    except ValueError as exc:
        assert app_intelligence_ledger.SESSION_START_PREFLIGHT_EVENT_TOKEN in str(exc)
    else:
        raise AssertionError("Expected preflight event append to require event token.")


def test_mark_session_started_updates_ledger_without_thread_turn(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="app-supervisor",
        purpose="Prepare supervised work.",
        run_id="session-run",
    )
    requested = app_intelligence_ledger.record_session_start_requested(
        state_root=tmp_path,
        run_id="session-run",
        transport="stdio",
        approval_token=app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN,
    )
    started = app_intelligence_ledger.mark_session_started(
        state_root=tmp_path,
        run_id="session-run",
        transport="stdio",
        codex_bin="/usr/local/bin/codex",
        start_result={"ok": True, "returncode": 0},
        version_result={"ok": True, "stdout": "codex-cli 0.131.0\n"},
    )
    shown = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="session-run")

    assert requested["event_type"] == "app_server_session_start_requested"
    assert started["run"]["phase"] == "session_started"
    assert started["run"]["status"] == "running"
    assert started["run"]["state"]["active_codex_thread_id"] is None
    assert started["run"]["state"]["app_server"]["model_turn_started"] is False
    assert shown["events"][-1]["event_type"] == "app_server_session_started"


def test_prepare_model_turn_packet_writes_review_artifact_without_send(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="contextual_reread",
        purpose="Prepare supervised work.",
        document_id="doc_123",
        run_id="packet-run",
    )
    app_intelligence_ledger.mark_session_started(
        state_root=tmp_path,
        run_id="packet-run",
        transport="stdio",
        codex_bin="/usr/local/bin/codex",
        start_result={"ok": True, "returncode": 0},
        version_result={"ok": True, "stdout": "codex-cli 0.131.0\n"},
    )

    prepared = app_intelligence_ledger.prepare_model_turn_packet(
        state_root=tmp_path,
        run_id="packet-run",
        task="contextual_reread",
        route={"provider": "codex-app-server", "model": ""},
        document={"id": "doc_123", "title": "Weekly Product Sync"},
        prompt_text="Review this transcript.",
        approval_token=app_intelligence_ledger.MODEL_TURN_PREFLIGHT_TOKEN,
    )
    shown = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert prepared["will_send_prompt"] is False
    assert Path(prepared["packet_path"]).exists()
    assert Path(prepared["prompt_path"]).read_text(encoding="utf-8") == "Review this transcript."
    assert prepared["packet"]["future_required_approval_token_for_send"] == app_intelligence_ledger.MODEL_TURN_SEND_TOKEN
    assert shown["run"]["prompt_packets"][0]["sent"] is False
    assert shown["events"][-1]["event_type"] == "model_turn_preflight_prepared"


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
