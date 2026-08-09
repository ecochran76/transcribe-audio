from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

import speaker_identity_plan0063_a1_authority as authority
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    write_immutable_private_json,
)


def _sqlite_root(root: Path, database_name: str) -> Path:
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    database = root / database_name
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE baseline (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO baseline(value) VALUES ('unchanged')")
        connection.commit()
    database.chmod(0o600)
    return root


def _repository(commit: str = "1" * 40) -> dict:
    return {
        "commit": commit,
        "upstream": commit,
        "ahead": 0,
        "behind": 0,
        "clean": True,
        "modules": {"authority.py": "2" * 64},
    }


def _arrange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    test_mode: bool = False,
) -> dict:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    transition_core = {
        "schema_version": "transcribe-audio.plan0063-reviewed-transition.v1",
        "status": "reviewed_transition_ready_for_private_rehearsal",
        "review_content_sha256": "3" * 64,
        "review_submission_sha256": "4" * 64,
        "metrics": {
            "canonical_person_count": 5,
            "slot_binding_count": 9,
            "active_voice_binding_count": 1,
        },
        "rehearsal_allowed": True,
        "a1_authorized": False,
        "live_mutation_count": 0,
    }
    transition = {
        **transition_core,
        "content_sha256": canonical_artifact_hash(transition_core),
    }
    transition_sha256 = transition["content_sha256"]
    rehearsal_paths = authority.canonical_rehearsal.rehearsal_paths(
        runtime_root, transition_sha256
    )
    ensure_private_tree(runtime_root, rehearsal_paths["run"])
    write_immutable_private_json(rehearsal_paths["transition"], transition)

    receipt_core = {
        "schema_version": authority.biometric_rehearsal.COMPLETE_RECEIPT_SCHEMA,
        "status": "complete_private_apply_and_rollback_proved",
        "transition_sha256": transition_sha256,
        "logical_transition_apply_count": 1,
        "logical_transition_rollback_count": 1,
        "applied_person_count": 5,
        "applied_reference_count": 4,
        "applied_profile_count": 12,
        "applied_source_count": 20,
        "test_mode": test_mode,
        "a1_request_ready": not test_mode,
        "a1_authorized": False,
        "live_mutation_count": 0,
    }
    receipt = {
        **receipt_core,
        "content_sha256": canonical_artifact_hash(receipt_core),
    }
    receipt_path = rehearsal_paths["run"] / "complete-receipt.json"
    write_immutable_private_json(receipt_path, receipt)

    def replay_complete_private_rehearsal(**kwargs):
        assert kwargs["transition_sha256"] == transition_sha256
        return {
            **receipt,
            "receipt_path": str(receipt_path),
            "idempotent_replay": True,
        }

    monkeypatch.setattr(
        authority.biometric_rehearsal,
        "replay_complete_private_rehearsal",
        replay_complete_private_rehearsal,
    )
    monkeypatch.setattr(
        authority.canonical_rehearsal,
        "validate_reviewed_transition",
        lambda value: value["content_sha256"],
    )
    monkeypatch.setattr(authority, "_repository_authority", _repository)

    live_store_root = _sqlite_root(tmp_path / "knowledge", "transcripts.sqlite3")
    live_reference_root = _sqlite_root(
        tmp_path / "references", authority.biometric_rehearsal.REFERENCE_DATABASE_NAME
    )
    live_profile_root = _sqlite_root(
        tmp_path / "profiles", authority.biometric_rehearsal.PROFILE_DATABASE_NAME
    )
    return {
        "runtime_root": runtime_root,
        "transition_sha256": transition_sha256,
        "live_store_root": live_store_root,
        "live_reference_root": live_reference_root,
        "live_profile_root": live_profile_root,
    }


def _build(arguments: dict) -> dict:
    return authority.build_a1_request(**arguments)


def test_a1_request_is_exact_private_nonapplying_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arrange(tmp_path, monkeypatch)
    request = _build(arguments)

    assert request["status"] == "awaiting_literal_operator_authorization"
    assert request["a1_authorized"] is False
    assert request["live_mutation_count"] == 0
    assert request["expected_apply_counts"] == {
        "canonical_people": 5,
        "slot_bindings": 9,
        "voice_bindings": 1,
        "references": 4,
        "profiles": 12,
        "sources": 20,
    }
    assert request["requested_actions"]["register_biometric_references"] is True
    assert request["requested_actions"]["write_provider_records"] is False
    assert request["requested_actions"]["write_graphiti"] is False
    assert request["requested_actions"]["quiesce_transcript_services"] is True
    assert request["requested_actions"]["restore_transcript_services"] is True
    request_path = Path(request["request_path"])
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(request_path.parent.stat().st_mode) == 0o700

    block = authority.render_a1_answer_block(request)
    assert block.splitlines() == [
        f"PLAN0063_A1_SCHEMA={authority.A1_SUBMISSION_SCHEMA}",
        f"PLAN0063_A1_REQUEST_SHA256={request['request_sha256']}",
        f"PLAN0063_A1_TRANSITION_SHA256={arguments['transition_sha256']}",
        "PLAN0063_A1_REHEARSAL_SHA256="
        + request["rehearsal_receipt_content_sha256"],
        f"PLAN0063_A1_DECISION={authority.APPROVAL_DECISION}",
    ]

    replay = _build(arguments)
    assert replay["request_sha256"] == request["request_sha256"]
    assert replay["idempotent_replay"] is True


def test_a1_request_rejects_test_mode_rehearsal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arrange(tmp_path, monkeypatch, test_mode=True)
    with pytest.raises(
        authority.Plan0063A1AuthorityError,
        match="production-mode complete private rehearsal",
    ):
        _build(arguments)


def test_a1_request_replay_rejects_live_state_or_repository_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arrange(tmp_path, monkeypatch)
    request = _build(arguments)
    database = arguments["live_store_root"] / "transcripts.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO baseline(value) VALUES ('drift')")
        connection.commit()
    with pytest.raises(
        authority.Plan0063A1AuthorityError,
        match="current exact authority",
    ):
        authority.replay_a1_request(**arguments)

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM baseline WHERE value = 'drift'")
        connection.commit()
    monkeypatch.setattr(authority, "_repository_authority", lambda: _repository("5" * 40))
    with pytest.raises(
        authority.Plan0063A1AuthorityError,
        match="current exact authority",
    ):
        authority.replay_a1_request(**arguments)
    assert request["live_mutation_count"] == 0


def test_literal_a1_authorization_freezes_and_replays_without_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arrange(tmp_path, monkeypatch)
    request = _build(arguments)
    block = authority.render_a1_answer_block(request)

    frozen = authority.freeze_a1_authorization(
        block,
        expected_request_sha256=request["request_sha256"],
        authorized_at="2026-08-09T18:30:00Z",
        **arguments,
    )
    assert frozen["status"] == "authorized_for_one_exact_live_apply"
    assert frozen["a1_authorized"] is True
    assert frozen["live_mutation_count"] == 0
    assert frozen["authorized_actions"]["perform_external_write"] is False
    assert stat.S_IMODE(Path(frozen["authority_path"]).stat().st_mode) == 0o600

    replay = authority.freeze_a1_authorization(
        block,
        expected_request_sha256=request["request_sha256"],
        authorized_at="2026-08-09T18:30:00Z",
        **arguments,
    )
    assert replay["authority_sha256"] == frozen["authority_sha256"]
    assert replay["idempotent_replay"] is True


def test_literal_a1_authorization_rejects_any_changed_or_extra_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arrange(tmp_path, monkeypatch)
    request = _build(arguments)
    block = authority.render_a1_answer_block(request)

    with pytest.raises(
        authority.Plan0063A1AuthorityError,
        match="must match the exact requested block",
    ):
        authority.freeze_a1_authorization(
            block.replace(authority.APPROVAL_DECISION, "decline") + "\nEXTRA=value",
            expected_request_sha256=request["request_sha256"],
            **arguments,
        )
