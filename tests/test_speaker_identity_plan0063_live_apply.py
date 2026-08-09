from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

import speaker_identity_plan0063_live_apply as live_apply
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


class FakeServices:
    def __init__(self) -> None:
        self.active = True
        self.history: list[str] = []

    def snapshot(self) -> dict:
        state = "active" if self.active else "inactive"
        sub_state = "running" if self.active else "dead"
        return {
            "services": {
                service: {
                    "active_state": state,
                    "sub_state": sub_state,
                    "nrestarts": 0,
                }
                for service in live_apply.SERVICES
            }
        }

    def quiesce(self) -> dict:
        self.history.append("quiesce")
        self.active = False
        return self.snapshot()

    def restore(self) -> dict:
        self.history.append("restore")
        self.active = True
        return self.snapshot()


def _insert(root: Path, database_name: str, value: str) -> None:
    with sqlite3.connect(root / database_name) as connection:
        connection.execute("INSERT INTO baseline(value) VALUES (?)", (value,))
        connection.commit()


def _arrange(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    live_store_root = _sqlite_root(tmp_path / "knowledge", "transcripts.sqlite3")
    live_reference_root = _sqlite_root(
        tmp_path / "references", live_apply.biometric_rehearsal.REFERENCE_DATABASE_NAME
    )
    live_profile_root = _sqlite_root(
        tmp_path / "profiles", live_apply.biometric_rehearsal.PROFILE_DATABASE_NAME
    )
    transition_core = {
        "schema_version": "transcribe-audio.plan0063-reviewed-transition.v1",
        "status": "reviewed_transition_ready_for_private_rehearsal",
        "metrics": {
            "canonical_person_count": 1,
            "slot_binding_count": 1,
            "external_identity_count": 0,
            "reviewed_voice_binding_count": 1,
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
    transition_paths = live_apply.canonical_rehearsal.rehearsal_paths(
        runtime_root, transition_sha256
    )
    ensure_private_tree(runtime_root, transition_paths["run"])
    write_immutable_private_json(transition_paths["transition"], transition)
    monkeypatch.setattr(
        live_apply.canonical_rehearsal,
        "validate_reviewed_transition",
        lambda value: value["content_sha256"],
    )

    baseline = live_apply._live_snapshots(
        live_store_root=live_store_root,
        live_reference_root=live_reference_root,
        live_profile_root=live_profile_root,
    )
    request_core = {
        "expected_live_state": {
            "snapshot_sha256s": live_apply._snapshot_hashes(baseline)
        }
    }
    a1_root = runtime_root / f"a1-{transition_sha256[:20]}"
    ensure_private_tree(runtime_root, a1_root)
    request_path = a1_root / "private-request.json"
    write_immutable_private_json(request_path, request_core)
    authority_core = {
        "schema_version": "test-a1-authority",
        "transition_sha256": transition_sha256,
        "a1_authorized": True,
    }
    authority_document = {
        **authority_core,
        "content_sha256": canonical_artifact_hash(authority_core),
    }
    authority_path = a1_root / "private-authority.json"
    write_immutable_private_json(authority_path, authority_document)
    authority_sha256 = authority_document["content_sha256"]
    request_sha256 = "7" * 64
    expected_counts = {
        "canonical_people": 1,
        "slot_bindings": 1,
        "voice_bindings": 1,
        "references": 1,
        "profiles": 1,
        "sources": 1,
    }

    def replay_authorization(*args, **kwargs):
        assert args == (transition_sha256,)
        assert kwargs["expected_request_sha256"] == request_sha256
        return {
            "authority_sha256": authority_sha256,
            "authority_path": str(authority_path),
            "request_path": str(request_path),
            "a1_authorized": True,
            "authorization_scope": "one_exact_plan0063_local_live_apply",
            "authorized_actions": dict(live_apply.a1.AUTHORIZED_ACTIONS),
            "expected_apply_counts": expected_counts,
            "live_mutation_count": 0,
        }

    monkeypatch.setattr(
        live_apply.a1, "replay_a1_authorization", replay_authorization
    )
    services = FakeServices()
    return {
        "call": {
            "authority_sha256": authority_sha256,
            "transition_sha256": transition_sha256,
            "expected_request_sha256": request_sha256,
            "live_store_root": live_store_root,
            "live_reference_root": live_reference_root,
            "live_profile_root": live_profile_root,
            "runtime_root": runtime_root,
            "service_controller": services,
            "test_mode": True,
        },
        "services": services,
        "baseline": baseline,
    }


def _successful_knowledge(transition, *, live_store_root):
    _insert(live_store_root, "transcripts.sqlite3", "knowledge-applied")
    return {
        "expected_counts": {
            "knowledge_people": 1,
            "knowledge_source_records": 1,
        },
        "person_receipts": [{"status": "saved"}],
        "observation_receipt": {"status": "saved"},
        "profile_receipt": {"status": "rebuilt"},
    }


def _successful_biometrics(
    transition,
    *,
    live_reference_root,
    live_profile_root,
    adapters,
    test_mode,
    baseline,
):
    assert test_mode is True
    _insert(
        live_reference_root,
        live_apply.biometric_rehearsal.REFERENCE_DATABASE_NAME,
        "reference-applied",
    )
    _insert(
        live_profile_root,
        live_apply.biometric_rehearsal.PROFILE_DATABASE_NAME,
        "profile-applied",
    )
    return {
        "created_references": [{"profile_id": "reference-one"}],
        "created_profiles": [{"profile_id": "profile-one"}],
        "reference_count": 1,
        "profile_count": 1,
        "source_count": 1,
    }


def test_a1_gated_live_apply_is_one_shot_backed_up_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arranged = _arrange(tmp_path, monkeypatch)
    monkeypatch.setattr(live_apply, "_apply_knowledge", _successful_knowledge)
    monkeypatch.setattr(live_apply, "_apply_biometrics", _successful_biometrics)

    receipt = live_apply.apply_live_transition(**arranged["call"])
    assert receipt["status"] == "live_apply_completed"
    assert receipt["logical_live_apply_count"] == 1
    assert receipt["logical_live_rollback_count"] == 0
    assert receipt["test_mode"] is True
    assert receipt["live_mutation_count"] == 0
    assert receipt["unauthorized_effect_count"] == 0
    assert arranged["services"].history == ["quiesce", "restore"]
    assert Path(receipt["receipt_path"]).is_file()
    assert stat.S_IMODE(Path(receipt["receipt_path"]).stat().st_mode) == 0o600

    replay = live_apply.apply_live_transition(**arranged["call"])
    assert replay["receipt_sha256"] == receipt["receipt_sha256"]
    assert replay["idempotent_replay"] is True
    assert arranged["services"].history == ["quiesce", "restore"]


def test_failed_live_apply_exactly_restores_all_stores_and_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arranged = _arrange(tmp_path, monkeypatch)
    unrelated = arranged["call"]["live_profile_root"] / "acquisitions"
    unrelated.mkdir(mode=0o750)
    unrelated.chmod(0o750)
    unrelated_file = unrelated / "untouched.bin"
    unrelated_file.write_bytes(b"outside-selected-profile-state")
    unrelated_file.chmod(0o640)
    monkeypatch.setattr(live_apply, "_apply_knowledge", _successful_knowledge)

    def fail_biometrics(
        transition,
        *,
        live_reference_root,
        live_profile_root,
        adapters,
        test_mode,
        baseline,
    ):
        _insert(
            live_reference_root,
            live_apply.biometric_rehearsal.REFERENCE_DATABASE_NAME,
            "partial-reference",
        )
        raise ValueError("synthetic failure")

    monkeypatch.setattr(live_apply, "_apply_biometrics", fail_biometrics)
    with pytest.raises(
        live_apply.Plan0063LiveApplyError,
        match="exact rollback completed",
    ):
        live_apply.apply_live_transition(**arranged["call"])

    current = live_apply._live_snapshots(
        live_store_root=arranged["call"]["live_store_root"],
        live_reference_root=arranged["call"]["live_reference_root"],
        live_profile_root=arranged["call"]["live_profile_root"],
    )
    assert current == arranged["baseline"]
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o750
    assert stat.S_IMODE(unrelated_file.stat().st_mode) == 0o640
    assert unrelated_file.read_bytes() == b"outside-selected-profile-state"
    assert arranged["services"].active is True
    assert arranged["services"].history == [
        "quiesce",
        "quiesce",
        "restore",
    ]
    replay = live_apply.replay_live_apply(
        arranged["call"]["transition_sha256"],
        live_store_root=arranged["call"]["live_store_root"],
        live_reference_root=arranged["call"]["live_reference_root"],
        live_profile_root=arranged["call"]["live_profile_root"],
        runtime_root=arranged["call"]["runtime_root"],
        service_controller=arranged["services"],
    )
    assert replay["status"] == "live_apply_failed_and_exactly_restored"
    assert replay["logical_live_rollback_count"] == 1


def test_live_apply_rejects_custom_production_controls_before_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arranged = _arrange(tmp_path, monkeypatch)
    call = {**arranged["call"], "test_mode": False}
    with pytest.raises(
        live_apply.Plan0063LiveApplyError,
        match="test-only",
    ):
        live_apply.apply_live_transition(**call)


def test_test_mode_live_apply_rejects_production_store_target() -> None:
    with pytest.raises(
        live_apply.Plan0063LiveApplyError,
        match="cannot target a production state root",
    ):
        live_apply._validate_test_targets(
            live_store_root=Path("~/.transcripts"),
            live_reference_root=Path("/tmp/test-reference-root"),
            live_profile_root=Path("/tmp/test-profile-root"),
        )
