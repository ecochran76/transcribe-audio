from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_generation3_preparation_stop as stop


def _state() -> dict:
    return {
        "attempted_unit_count": 7,
        "completed_p1_unit_count": 6,
        "completed_p2_unit_count": 6,
        "completed_p2_method_count": 30,
        "failed_run_id": stop.EXPECTED_FAILED_RUN_ID,
        "failed_source_sha256": stop.EXPECTED_FAILED_SOURCE_SHA256,
        "source_duration_seconds": stop.OBSERVED_SOURCE_DURATION_SECONDS,
        "decoded_duration_seconds": stop.OBSERVED_DECODED_DURATION_SECONDS,
        "duration_drift_seconds": 89.776791,
        "duration_tolerance_seconds": 0.05,
        "exception_class": stop.EXPECTED_EXCEPTION_CLASS,
        "exception_message": stop.EXPECTED_EXCEPTION_MESSAGE,
        "explicit_absences": {
            "failed_p1_outputs": True, "seventh_p2_unit": True,
            "preparation_application": True, "preparation_receipt": True,
            "condition_freeze": True, "evaluation_windows": True,
            "exact_trials": True, "scores": True, "metrics": True,
            "terminal_decision": True,
        },
        "artifact_inventory": [{"relative_path": "p1/example", "sha256": "a" * 64, "bytes": 1}],
        "artifact_inventory_sha256": "b" * 64,
    }


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(stop, "_validate_parent", lambda paths: {"status": "frozen"})
    monkeypatch.setattr(stop, "_partial_state", lambda paths: _state())
    monkeypatch.setattr(stop, "_repository_authority", lambda: {
        "commit": "c" * 40, "module_sha256": "d" * 64,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    })
    return tmp_path


def test_preview_authorizes_only_terminal_stop(isolated: Path) -> None:
    preview = stop.preview_generation3_preparation_stop(runtime_root=isolated)
    assert preview["status"] == "terminal_stop_required"
    assert preview["reason_code"] == stop.EXPECTED_REASON_CODE
    assert preview["failure_observation"]["duration_drift_seconds"] == 89.776791
    assert preview["failure_observation"]["did_recompute_audio"] is False
    assert preview["authorized_actions"]["record_terminal_stop"] is True
    assert all(
        value is False for key, value in preview["authorized_actions"].items()
        if key != "record_terminal_stop"
    )
    assert preview["contains_paths"] is False
    portable = stop.portable_stop_projection(preview)
    assert "failure_observation" not in portable
    assert portable["failure_observation_sha256"] == preview["failure_observation_sha256"]
    assert portable["contains_private_evaluation"] is False


def test_apply_writes_only_sibling_terminal_packet(isolated: Path) -> None:
    preview = stop.preview_generation3_preparation_stop(runtime_root=isolated)
    receipt = stop.apply_generation3_preparation_stop(
        preview, expected_content_sha256=preview["content_sha256"],
        runtime_root=isolated,
    )
    paths = stop._paths(isolated)
    assert paths["manifest"].is_file()
    assert paths["stop_receipt"].is_file()
    assert not paths["application"].exists()
    assert not paths["receipt"].exists()
    assert paths["stop"].parent.name == "terminal-stops"
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["private_failure_evidence"]["exception_class"] == "AudioDerivativeError"
    assert not any(manifest["authorized_actions_after_stop"].values())
    assert set(receipt["action_vector"]) == set(stop.POST_STOP_ACTIONS)
    assert not any(receipt["action_vector"].values())
    assert "duration_drift_seconds" not in receipt
    assert "duration_tolerance_seconds" not in receipt
    assert "private_failure_evidence" not in receipt


def test_replay_is_full_body_and_idempotent(isolated: Path) -> None:
    preview = stop.preview_generation3_preparation_stop(runtime_root=isolated)
    stop.apply_generation3_preparation_stop(
        preview, expected_content_sha256=preview["content_sha256"],
        runtime_root=isolated,
    )
    replay = stop.replay_generation3_preparation_stop(runtime_root=isolated)
    assert replay["idempotent_replay"] is True
    assert replay["replay_mode"] == "full_body_without_audio_execution"
    second = stop.apply_generation3_preparation_stop(
        preview, expected_content_sha256=preview["content_sha256"],
        runtime_root=isolated,
    )
    assert second["idempotent_replay"] is True


def test_stale_preview_is_rejected(isolated: Path) -> None:
    preview = stop.preview_generation3_preparation_stop(runtime_root=isolated)
    stale = {**preview, "failure_observation_sha256": "0" * 64}
    with pytest.raises(stop.Generation3PreparationStopError, match="stale"):
        stop.apply_generation3_preparation_stop(
            stale, expected_content_sha256=preview["content_sha256"],
            runtime_root=isolated,
        )


def test_replay_rejects_coordinated_terminal_mutation(isolated: Path) -> None:
    preview = stop.preview_generation3_preparation_stop(runtime_root=isolated)
    stop.apply_generation3_preparation_stop(
        preview, expected_content_sha256=preview["content_sha256"], runtime_root=isolated,
    )
    paths = stop._paths(isolated)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["authorized_actions_after_stop"].pop("retry_preparation")
    paths["manifest"].write_text(json.dumps(manifest))
    receipt = json.loads(paths["stop_receipt"].read_text())
    receipt["manifest_sha256"] = stop.sha256_file(paths["manifest"])
    paths["stop_receipt"].write_text(json.dumps(receipt))
    with pytest.raises(stop.Generation3PreparationStopError, match="drifted"):
        stop.replay_generation3_preparation_stop(runtime_root=isolated)


def test_live_partial_state_is_read_only() -> None:
    state = stop._partial_state(stop._paths())
    assert state["attempted_unit_count"] == 7
    assert state["completed_p1_unit_count"] == 6
    assert state["completed_p2_method_count"] == 30
    assert state["artifact_inventory_sha256"].isalnum()
