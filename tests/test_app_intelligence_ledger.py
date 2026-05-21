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

    reviewed = app_intelligence_ledger.read_model_turn_packet(
        state_root=tmp_path,
        run_id="packet-run",
        packet_id=prepared["packet"]["packet_id"],
    )

    assert reviewed["will_send_prompt"] is False
    assert reviewed["future_required_approval_token_for_send"] == app_intelligence_ledger.MODEL_TURN_SEND_TOKEN
    assert reviewed["prompt_text"] == "Review this transcript."
    assert reviewed["packet"]["packet_id"] == prepared["packet"]["packet_id"]

    send_preflight = app_intelligence_ledger.model_turn_send_preflight(
        state_root=tmp_path,
        run_id="packet-run",
        packet_id=prepared["packet"]["packet_id"],
        approval_token=app_intelligence_ledger.MODEL_TURN_SEND_TOKEN,
    )
    shown_after_send_preflight = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert send_preflight["ok"] is True
    assert send_preflight["will_send_prompt"] is False
    assert send_preflight["will_write_event"] is False
    assert send_preflight["checks"]["packet_not_sent"] is True
    assert send_preflight["prompt_char_count"] == len("Review this transcript.")
    assert shown_after_send_preflight["run"]["prompt_packets"][0]["sent"] is False
    assert [event["event_id"] for event in shown_after_send_preflight["events"]] == [event["event_id"] for event in shown["events"]]

    app_intelligence_ledger.append_codex_event(
        state_root=tmp_path,
        run_id="packet-run",
        payload={"method": "turn/started", "params": {"turn": {"id": "turn_123"}}},
    )
    started = app_intelligence_ledger.record_model_turn_started(
        state_root=tmp_path,
        run_id="packet-run",
        packet_id=prepared["packet"]["packet_id"],
        thread_id="thread_123",
        turn_id="turn_123",
        app_server_result={"captured_event_count": 1},
    )
    shown_after_start = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert started["run"]["phase"] == "model_turn_started"
    assert started["run"]["state"]["active_codex_thread_id"] == "thread_123"
    assert started["run"]["state"]["latest_turn_id"] == "turn_123"
    assert started["run"]["prompt_packets"][0]["sent"] is True
    assert shown_after_start["events"][-1]["event_type"] == "model_turn_started"
    assert shown_after_start["codex_events_count"] == 1

    captured = app_intelligence_ledger.record_model_turn_status(
        state_root=tmp_path,
        run_id="packet-run",
        thread_id="thread_123",
        turn_id="turn_123",
        status_payload={
            "status": "completed",
            "completed": True,
            "output_text": json.dumps(
                {
                    "action": "ask_for_human_review",
                    "rationale": "The turn needs operator review before any host action.",
                    "confidence": 0.72,
                    "review_flags": ["human_review_required"],
                    "recommended_next_prompt": "Ask for missing context.",
                }
            ),
            "thread_read_response": {},
            "turns_list_response": {},
            "items_list_response": {},
        },
        approval_token=app_intelligence_ledger.MODEL_TURN_STATUS_TOKEN,
    )
    shown_after_status = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert captured["completed"] is True
    assert captured["will_execute_structured_decision"] is False
    assert Path(captured["artifact_path"]).exists()
    assert shown_after_status["run"]["phase"] == "model_turn_completed"
    assert shown_after_status["run"]["latest_model_turn_status"]["output_char_count"] > 0
    assert shown_after_status["events"][-1]["event_type"] == "model_turn_status_captured"

    decision = app_intelligence_ledger.validate_latest_structured_decision(
        state_root=tmp_path,
        run_id="packet-run",
        approval_token=app_intelligence_ledger.STRUCTURED_DECISION_VALIDATE_TOKEN,
    )
    shown_after_decision = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert decision["valid"] is True
    assert decision["will_execute_host_action"] is False
    assert decision["decision"]["action"] == "ask_for_human_review"
    assert Path(decision["artifact_path"]).exists()
    assert shown_after_decision["run"]["decisions"][0]["status"] == "validated"
    assert shown_after_decision["events"][-1]["event_type"] == "structured_decision_validated"

    applied = app_intelligence_ledger.apply_validated_structured_decision(
        state_root=tmp_path,
        run_id="packet-run",
        decision_id=decision["decision_id"],
        approval_token=app_intelligence_ledger.STRUCTURED_DECISION_APPLY_TOKEN,
        reviewer="test-operator",
        note="Route to the review queue surface.",
    )
    shown_after_apply = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert applied["ok"] is True
    assert applied["decision_action"] == "ask_for_human_review"
    assert applied["applied_ledger_state"] is True
    assert applied["will_execute_external_action"] is False
    assert applied["will_execute_downstream_action"] is False
    assert applied["will_execute_write_bearing_action"] is False
    assert applied["will_fork_or_rollback"] is False
    assert Path(applied["artifact_path"]).exists()
    assert shown_after_apply["run"]["phase"] == "human_review_requested"
    assert shown_after_apply["run"]["status"] == "needs_human_review"
    assert shown_after_apply["run"]["decisions"][0]["status"] == "applied"
    assert shown_after_apply["events"][-1]["event_type"] == "structured_decision_applied"

    annotated = app_intelligence_ledger.record_human_review_decision(
        state_root=tmp_path,
        run_id="packet-run",
        decision_id=decision["decision_id"],
        review_action="annotate",
        approval_token=app_intelligence_ledger.HUMAN_REVIEW_DECISION_TOKEN,
        reviewer="test-operator",
        note="Need a second source check.",
    )
    resolved = app_intelligence_ledger.record_human_review_decision(
        state_root=tmp_path,
        run_id="packet-run",
        decision_id=decision["decision_id"],
        review_action="resolve",
        approval_token=app_intelligence_ledger.HUMAN_REVIEW_DECISION_TOKEN,
        reviewer="test-operator",
        note="Reviewed and accepted no further action.",
    )
    reopened = app_intelligence_ledger.record_human_review_decision(
        state_root=tmp_path,
        run_id="packet-run",
        decision_id=decision["decision_id"],
        review_action="reopen",
        approval_token=app_intelligence_ledger.HUMAN_REVIEW_DECISION_TOKEN,
        reviewer="test-operator",
        note="Reopened for another operator pass.",
    )
    shown_after_review = app_intelligence_ledger.response_for_run(state_root=tmp_path, run_id="packet-run")

    assert annotated["human_review_status"] == "open"
    assert resolved["human_review_status"] == "resolved"
    assert reopened["human_review_status"] == "open"
    assert shown_after_review["run"]["decisions"][0]["human_review"]["status"] == "open"
    assert len(shown_after_review["run"]["decisions"][0]["human_review"]["notes"]) == 3
    assert shown_after_review["events"][-1]["event_type"] == "human_review_decision_recorded"


def test_structured_decision_validation_rejects_non_decision_output(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="contextual_reread",
        purpose="Validate bad decision output.",
        run_id="bad-decision-run",
    )
    app_intelligence_ledger.mark_session_started(
        state_root=tmp_path,
        run_id="bad-decision-run",
        transport="stdio",
        codex_bin="/usr/local/bin/codex",
        start_result={"ok": True},
        version_result={"ok": True},
    )
    app_intelligence_ledger.record_model_turn_started(
        state_root=tmp_path,
        run_id="bad-decision-run",
        packet_id="packet-1",
        thread_id="thread_bad",
        turn_id="turn_bad",
        app_server_result={},
    )
    app_intelligence_ledger.record_model_turn_status(
        state_root=tmp_path,
        run_id="bad-decision-run",
        thread_id="thread_bad",
        turn_id="turn_bad",
        status_payload={"status": "completed", "completed": True, "output_text": '{"summary":"not a decision"}'},
        approval_token=app_intelligence_ledger.MODEL_TURN_STATUS_TOKEN,
    )

    decision = app_intelligence_ledger.validate_latest_structured_decision(
        state_root=tmp_path,
        run_id="bad-decision-run",
        approval_token=app_intelligence_ledger.STRUCTURED_DECISION_VALIDATE_TOKEN,
    )

    assert decision["valid"] is False
    assert decision["will_execute_host_action"] is False
    assert decision["errors"]
    assert decision["run"]["decisions"][0]["status"] == "rejected"


def test_structured_decision_apply_blocks_write_bearing_actions(tmp_path: Path) -> None:
    app_intelligence_ledger.create_run(
        state_root=tmp_path,
        workflow="contextual_reread",
        purpose="Validate fork decision output.",
        run_id="fork-decision-run",
    )
    app_intelligence_ledger.mark_session_started(
        state_root=tmp_path,
        run_id="fork-decision-run",
        transport="stdio",
        codex_bin="/usr/local/bin/codex",
        start_result={"ok": True},
        version_result={"ok": True},
    )
    app_intelligence_ledger.record_model_turn_started(
        state_root=tmp_path,
        run_id="fork-decision-run",
        packet_id="packet-1",
        thread_id="thread_fork",
        turn_id="turn_fork",
        app_server_result={},
    )
    app_intelligence_ledger.record_model_turn_status(
        state_root=tmp_path,
        run_id="fork-decision-run",
        thread_id="thread_fork",
        turn_id="turn_fork",
        status_payload={
            "status": "completed",
            "completed": True,
            "output_text": json.dumps(
                {
                    "action": "fork_branches",
                    "rationale": "Explore competing context routes.",
                    "confidence": 0.81,
                    "review_flags": [],
                    "branch_count": 2,
                    "experiments": ["drive-heavy", "graph-heavy"],
                }
            ),
        },
        approval_token=app_intelligence_ledger.MODEL_TURN_STATUS_TOKEN,
    )
    decision = app_intelligence_ledger.validate_latest_structured_decision(
        state_root=tmp_path,
        run_id="fork-decision-run",
        approval_token=app_intelligence_ledger.STRUCTURED_DECISION_VALIDATE_TOKEN,
    )

    try:
        app_intelligence_ledger.apply_validated_structured_decision(
            state_root=tmp_path,
            run_id="fork-decision-run",
            decision_id=decision["decision_id"],
            approval_token=app_intelligence_ledger.STRUCTURED_DECISION_APPLY_TOKEN,
        )
    except ValueError as exc:
        assert "only records ledger-only" in str(exc)
    else:
        raise AssertionError("Expected fork_branches to be blocked by the ledger-only apply endpoint.")


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
