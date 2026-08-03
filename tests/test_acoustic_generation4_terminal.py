import json

import pytest

import acoustic_generation4_terminal as terminal


REPOSITORY = {
    "commit": "1" * 40,
    "module_sha256": "2" * 64,
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}

FAILURE = {
    "g2_preview_sha256": terminal.G2_PREVIEW_SHA256,
    "g2_manifest_sha256": terminal.G2_MANIFEST_SHA256,
    "failed_source_sha256": "a" * 64,
    "failed_source_path": "/private/source.m4a",
    "failed_p1_run_id": "audio-run-one",
    "failed_p1_dry_run_sha256": "b" * 64,
    "source_duration_seconds": 10.2,
    "decoded_duration_seconds": 10.0,
    "duration_drift_seconds": 0.2,
    "frozen_tolerance_seconds": 0.05,
    "completed_p1_p2_case_count": 3,
    "failed_case_count": 1,
    "not_attempted_after_stop_count": 3,
    "completed_p2": [],
}


def _bind(monkeypatch) -> None:
    monkeypatch.setattr(terminal, "_failure_evidence", lambda: dict(FAILURE))
    monkeypatch.setattr(terminal, "_repository_authority", lambda: dict(REPOSITORY))


def test_terminal_preview_applies_frozen_stop_precedence(monkeypatch) -> None:
    _bind(monkeypatch)

    preview = terminal.preview_generation4_terminal()
    portable = terminal._portable(preview)

    assert preview["terminal_decision"] == "stop"
    assert preview["terminal_stage"] == "G3_blind_preparation"
    assert preview["policy_precedence"] == 1
    assert preview["duration_drift_seconds"] > preview["frozen_tolerance_seconds"]
    assert not any(preview["action_vector"].values())
    assert preview["did_reveal_gold_to_prediction_workers"] is False
    assert preview["did_send_prediction_turn"] is False
    assert preview["did_load_or_run_biometric_models"] is False
    assert "private_evidence" not in portable
    assert portable["contains_paths"] is False
    assert portable["contains_private_membership"] is False


def test_terminal_apply_replay_is_private_and_idempotent(tmp_path, monkeypatch) -> None:
    _bind(monkeypatch)
    preview = terminal.preview_generation4_terminal()

    applied = terminal.apply_generation4_terminal(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = terminal.replay_generation4_terminal(
        preview["content_sha256"], runtime_root=tmp_path
    )

    paths = terminal._paths(tmp_path, preview["content_sha256"])
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    receipt = json.loads(paths["receipt"].read_text())
    assert "private_evidence" not in receipt


def test_terminal_replay_rejects_failure_drift(tmp_path, monkeypatch) -> None:
    _bind(monkeypatch)
    preview = terminal.preview_generation4_terminal()
    terminal.apply_generation4_terminal(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    monkeypatch.setattr(
        terminal,
        "_failure_evidence",
        lambda: {**FAILURE, "duration_drift_seconds": 0.3},
    )

    with pytest.raises(terminal.Generation4TerminalError, match="drifted"):
        terminal.replay_generation4_terminal(
            preview["content_sha256"], runtime_root=tmp_path
        )
