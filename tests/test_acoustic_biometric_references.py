from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import acoustic_biometric_references as biometric_authority

from acoustic_biometric_references import (
    APPROVAL_SCHEMA,
    INVALIDATION_SCHEMA,
    MATERIALIZATION_SCHEMA,
    PROMOTION_SCHEMA,
    BiometricReferenceError,
    acknowledge_descendant_invalidation,
    acknowledge_descendant_promotion,
    apply_change as _apply_change,
    descendant_is_eligible,
    dry_run as _dry_run,
    register_descendant,
    request_descendant_invalidation,
    replay_reference,
    resolve_eligible_reference,
    source_set_sha256,
)
from acoustic_audio_derivatives import (
    ensure_private_tree,
    sha256_file,
    write_immutable_private_json,
)


def dry_run(*args, **kwargs):
    kwargs.setdefault("test_mode", True)
    return _dry_run(*args, **kwargs)


def apply_change(*args, **kwargs):
    kwargs.setdefault("test_mode", True)
    return _apply_change(*args, **kwargs)


def p4_anchor(root: Path, receipt: dict) -> tuple[Path, Path]:
    authority_root = root.parent / "p4-authority"
    ensure_private_tree(authority_root, authority_root)
    canonical = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    path = authority_root / f"{hashlib.sha256(canonical).hexdigest()}.json"
    write_immutable_private_json(path, receipt)
    return path, authority_root


def materialization_receipt(
    resolved: dict, descendant_id: str, artifact_sha256: str
) -> dict:
    return {
        "schema_version": MATERIALIZATION_SCHEMA,
        "status": "staged",
        "profile_id": resolved["profile_id"],
        "generation_id": resolved["generation_id"],
        "generation_sha256": resolved["generation_sha256"],
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha256,
        "staging_ref_sha256": "9" * 64,
        "eligible_for_use": False,
        "will_perform_external_write": False,
        "created_at": "2026-07-31T12:01:00Z",
    }


def registration_token(
    resolved: dict, descendant_id: str, artifact_sha256: str
) -> str:
    materialization = materialization_receipt(
        resolved, descendant_id, artifact_sha256
    )
    canonical = json.dumps(
        materialization,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return (
        f"REGISTER_BIOMETRIC_DESCENDANT:{resolved['generation_id']}:"
        f"{descendant_id}:{artifact_sha256}:{hashlib.sha256(canonical).hexdigest()}"
    )


def promote_descendant(root: Path, registered: dict) -> dict:
    receipt = {
        "schema_version": PROMOTION_SCHEMA,
        "status": "promoted",
        "descendant_id": registered["descendant_id"],
        "artifact_sha256": registered["artifact_sha256"],
        "materialization_receipt_sha256": registered[
            "materialization_receipt_sha256"
        ],
        "eligible_for_use": True,
        "will_perform_external_write": False,
        "promoted_at": "2026-07-31T12:02:00Z",
    }
    authority_path, authority_root = p4_anchor(root, receipt)
    return acknowledge_descendant_promotion(
        registered["descendant_id"],
        receipt,
        authority_receipt_path=authority_path,
        p4_authority_root=authority_root,
        approval_token=registered["required_promotion_token"],
        runtime_root=root,
    )


def invalidate_descendant(
    root: Path, descendant_id: str, artifact_sha256: str, reason: str
) -> dict:
    receipt = {
        "schema_version": INVALIDATION_SCHEMA,
        "status": "invalidated",
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha256,
        "reason": reason,
        "evidence_sha256": "8" * 64,
        "will_perform_external_write": False,
        "acknowledged_at": "2099-07-31T12:03:00Z",
    }
    authority_path, authority_root = p4_anchor(root, receipt)
    return acknowledge_descendant_invalidation(
        descendant_id,
        receipt,
        authority_receipt_path=authority_path,
        p4_authority_root=authority_root,
        approval_token=(
            f"ACK_BIOMETRIC_DESCENDANT_INVALIDATION:{descendant_id}:"
            f"{artifact_sha256}:{reason}"
        ),
        runtime_root=root,
    )


def source(reference_id: str, *, session: str, offset: float = 0.0) -> dict:
    suffix = reference_id[-1]
    result = {
        "reference_id": reference_id,
        "source_blob_id": f"blob-{suffix}",
        "source_sha256": suffix * 64,
        "recording_id": f"recording-{suffix}",
        "conversation_id": f"conversation-{suffix}",
        "speaker_label_id": f"speaker-label-{suffix}",
        "session_id": session,
        "start_seconds": offset,
        "end_seconds": offset + 2.0,
        "source_duration_seconds": 20.0,
        "quality_evidence": {
            "evidence_id": f"quality-{suffix}",
            "sha256": ("a" if suffix != "a" else "b") * 64,
        },
        "device_class": "synthetic-fixture",
        "acoustic_conditions": ["synthetic"],
    }
    result["fixture_authority"] = {
        "schema_version": "transcribe-audio.synthetic-reference-fixture.v1",
        "fixture_id": reference_id,
        "source_sha256": result["source_sha256"],
        "source_duration_seconds": result["source_duration_seconds"],
        "quality_evidence_sha256": result["quality_evidence"]["sha256"],
    }
    return result


def approval(
    action: str,
    profile_id: str,
    person_ref_id: str,
    *,
    sources: list[dict] | None,
    expected_generation_id: str | None,
    suffix: str,
) -> dict:
    return {
        "schema_version": APPROVAL_SCHEMA,
        "approval_id": f"bio-approval-{suffix}",
        "reviewer_ref_id": "reviewer-ref-synthetic",
        "reviewed_at": "2026-07-31T12:00:00Z",
        "purpose": f"biometric_reference_{action}",
        "scope": {
            "profile_id": profile_id,
            "person_ref_id": person_ref_id,
            "source_set_sha256": (
                source_set_sha256(sources, test_mode=True)
                if sources is not None
                else None
            ),
            "expected_generation_id": expected_generation_id,
        },
    }


def plan_and_apply(
    root: Path,
    action: str,
    *,
    profile_id: str,
    person_ref_id: str,
    sources: list[dict] | None,
    expected_generation_id: str | None,
    suffix: str,
) -> tuple[dict, dict]:
    plan = dry_run(
        action,
        profile_id=profile_id,
        person_ref_id=person_ref_id,
        sources=sources,
        approval=approval(
            action,
            profile_id,
            person_ref_id,
            sources=sources,
            expected_generation_id=expected_generation_id,
            suffix=suffix,
        ),
        test_mode=True,
        runtime_root=root,
    )
    receipt = apply_change(
        plan["run_id"],
        approval_token=plan["required_approval_token"],
        sources=sources,
        test_mode=True,
        runtime_root=root,
    )
    return plan, receipt


def create_profile(
    root: Path,
    *,
    profile_id: str = "reference-profile-one",
    person_ref_id: str = "person-ref-one",
) -> tuple[dict, dict]:
    sources = [
        source("reference-a", session="session-one"),
        source("reference-b", session="session-two", offset=3.0),
    ]
    return plan_and_apply(
        root,
        "create",
        profile_id=profile_id,
        person_ref_id=person_ref_id,
        sources=sources,
        expected_generation_id=None,
        suffix="create",
    )


def transition(
    root: Path,
    action: str,
    replay: dict,
    *,
    sources: list[dict] | None = None,
    suffix: str,
) -> tuple[dict, dict]:
    return plan_and_apply(
        root,
        action,
        profile_id=replay["profile_id"],
        person_ref_id=replay["person_ref_id"],
        sources=sources,
        expected_generation_id=replay["head_generation_id"],
        suffix=suffix,
    )


def test_p4_can_request_descendant_invalidation(tmp_path: Path) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    resolved = resolve_eligible_reference(created["person_ref_id"], runtime_root=root)
    descendant_id = "materialized-profile-p4-withdraw"
    artifact_sha256 = "e" * 64
    materialization = materialization_receipt(
        resolved, descendant_id, artifact_sha256
    )
    authority_path, authority_root = p4_anchor(root, materialization)
    registered = register_descendant(
        resolved["profile_id"],
        resolved["generation_id"],
        descendant_id,
        artifact_sha256,
        materialization_receipt=materialization,
        authority_receipt_path=authority_path,
        p4_authority_root=authority_root,
        approval_token=registration_token(
            resolved, descendant_id, artifact_sha256
        ),
        runtime_root=root,
    )
    promote_descendant(root, registered)
    assert descendant_is_eligible(descendant_id, runtime_root=root)

    requested = request_descendant_invalidation(
        descendant_id,
        reason="p4_profile_withdrawn",
        approval_token=(
            f"INVALIDATE_BIOMETRIC_DESCENDANT:{descendant_id}:"
            f"{artifact_sha256}:p4_profile_withdrawn"
        ),
        runtime_root=root,
    )

    assert requested["state"] == "invalidation_pending"
    assert requested["required_acknowledgment_token"] == (
        f"ACK_BIOMETRIC_DESCENDANT_INVALIDATION:{descendant_id}:"
        f"{artifact_sha256}:p4_profile_withdrawn"
    )
    assert descendant_is_eligible(descendant_id, runtime_root=root) is False
    invalidate_descendant(
        root, descendant_id, artifact_sha256, "p4_profile_withdrawn"
    )
    assert descendant_is_eligible(descendant_id, runtime_root=root) is False


def test_synthetic_reference_lifecycle_and_descendant_revocation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    plan, created = create_profile(root)
    assert created["status"] == "success"
    assert created["lifecycle_state"] == "active"
    assert created["will_read_audio"] is False
    assert created["will_run_model"] is False
    assert created["will_create_embedding"] is False
    repeated = apply_change(
        plan["run_id"],
        approval_token=plan["required_approval_token"],
        runtime_root=root,
    )
    assert repeated["idempotent_replay"] is True
    active = replay_reference(created["profile_id"], runtime_root=root)
    assert active["lifecycle_state"] == "verified_active"
    assert active["eligible_for_materialization"] is True
    resolved = resolve_eligible_reference("person-ref-one", runtime_root=root)
    reference = resolved["reference"]
    assert len(reference["sources"]) == 2
    assert "eligible_for_scoring" not in reference
    assert "embedding_model_revision" not in reference
    assert "private_embedding_ref" not in reference

    descendant_one = "materialized-profile-one"
    descendant_sha = "e" * 64
    register_token = registration_token(
        resolved, descendant_one, descendant_sha
    )
    materialization = materialization_receipt(
        resolved, descendant_one, descendant_sha
    )
    authority_path, authority_root = p4_anchor(root, materialization)
    registered = register_descendant(
        resolved["profile_id"],
        resolved["generation_id"],
        descendant_one,
        descendant_sha,
        materialization_receipt=materialization,
        authority_receipt_path=authority_path,
        p4_authority_root=authority_root,
        approval_token=register_token,
        runtime_root=root,
    )
    assert registered["idempotent_replay"] is False
    assert (
        register_descendant(
            resolved["profile_id"],
            resolved["generation_id"],
            descendant_one,
            descendant_sha,
            materialization_receipt=materialization_receipt(
                resolved, descendant_one, descendant_sha
            ),
            authority_receipt_path=authority_path,
            p4_authority_root=authority_root,
            approval_token=register_token,
            runtime_root=root,
        )["idempotent_replay"]
        is True
    )
    assert not descendant_is_eligible(descendant_one, runtime_root=root)
    promote_descendant(root, registered)
    assert descendant_is_eligible(descendant_one, runtime_root=root)

    replacement_sources = [
        source("reference-c", session="session-three"),
        source("reference-d", session="session-four", offset=4.0),
    ]
    supersede_plan, superseded = transition(
        root,
        "supersede",
        active,
        sources=replacement_sources,
        suffix="supersede",
    )
    assert supersede_plan["required_approval_token"] == (
        f"SUPERSEDE_BIOMETRIC_REFERENCE:{active['head_generation_id']}:"
        f"{supersede_plan['run_id']}:{supersede_plan['dry_run_sha256']}"
    )
    assert superseded["lifecycle_state"] == "active"
    assert superseded["generation_id"] != active["head_generation_id"]
    assert not descendant_is_eligible(descendant_one, runtime_root=root)
    invalidate_descendant(
        root, descendant_one, descendant_sha, "reference_superseded"
    )
    current = replay_reference(created["profile_id"], runtime_root=root)
    assert current["generation_count"] == 2
    assert current["eligible_for_materialization"] is True

    descendant_two = "materialized-profile-two"
    descendant_two_sha = "f" * 64
    second_resolved = {
        "profile_id": current["profile_id"],
        "generation_id": current["head_generation_id"],
        "generation_sha256": current["head_manifest_sha256"],
    }
    second_materialization = materialization_receipt(
        second_resolved, descendant_two, descendant_two_sha
    )
    second_authority_path, second_authority_root = p4_anchor(
        root, second_materialization
    )
    second_registration = register_descendant(
        current["profile_id"],
        current["head_generation_id"],
        descendant_two,
        descendant_two_sha,
        materialization_receipt=second_materialization,
        authority_receipt_path=second_authority_path,
        p4_authority_root=second_authority_root,
        approval_token=registration_token(
            second_resolved,
            descendant_two,
            descendant_two_sha,
        ),
        runtime_root=root,
    )
    promote_descendant(root, second_registration)
    withdraw_plan, withdrawn = transition(
        root, "withdraw", current, suffix="withdraw"
    )
    assert withdraw_plan["required_approval_token"] == (
        f"WITHDRAW_BIOMETRIC_REFERENCE:{current['head_generation_id']}:"
        f"{withdraw_plan['dry_run_sha256']}"
    )
    assert withdrawn["lifecycle_state"] == "withdrawn"
    assert not descendant_is_eligible(descendant_two, runtime_root=root)
    inactive = replay_reference(created["profile_id"], runtime_root=root)
    assert inactive["status"] == "blocked"
    assert inactive["reason_code"] == "descendant_invalidation_pending"
    invalidate_descendant(
        root, descendant_two, descendant_two_sha, "reference_withdrawn"
    )
    inactive = replay_reference(created["profile_id"], runtime_root=root)
    assert inactive["status"] == "success"
    assert inactive["lifecycle_state"] == "verified_withdrawn"
    with pytest.raises(BiometricReferenceError, match="no eligible"):
        resolve_eligible_reference("person-ref-one", runtime_root=root)

    delete_plan, deleted = transition(root, "delete", inactive, suffix="delete")
    assert delete_plan["required_approval_token"] == (
        f"DELETE_BIOMETRIC_REFERENCE:{inactive['profile_id']}:"
        f"{delete_plan['dry_run_sha256']}"
    )
    assert deleted["lifecycle_state"] == "deleted"
    tombstone = replay_reference(created["profile_id"], runtime_root=root)
    assert tombstone["lifecycle_state"] == "verified_deleted"
    assert tombstone["eligible_for_materialization"] is False
    with pytest.raises(BiometricReferenceError, match="already exists"):
        dry_run(
            "create",
            profile_id=created["profile_id"],
            person_ref_id="person-ref-one",
            sources=replacement_sources,
            approval=approval(
                "create",
                created["profile_id"],
                "person-ref-one",
                sources=replacement_sources,
                expected_generation_id=None,
                suffix="resurrect",
            ),
            runtime_root=root,
        )

    database = root / "references.sqlite3"
    with sqlite3.connect(database) as connection:
        manifests = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT manifest_json FROM generations ORDER BY sequence"
            )
        ]
    assert all("sources" not in manifest for manifest in manifests)
    assert all("approval" not in manifest for manifest in manifests)
    assert b"recording-a" not in database.read_bytes()
    assert b"conversation-a" not in database.read_bytes()
    assert "recording-a" not in Path(plan["dry_run_path"]).read_text(
        encoding="utf-8"
    )
    for path in root.rglob("*"):
        expected_mode = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode


def test_validation_rejects_non_biometric_approval_private_fields_and_bounds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    sources = [source("reference-a", session="session-one")]
    ordinary_confirmation = approval(
        "create",
        "reference-profile-one",
        "person-ref-one",
        sources=sources,
        expected_generation_id=None,
        suffix="ordinary",
    )
    ordinary_confirmation["purpose"] = "speaker_identity_confirmation"
    with pytest.raises(BiometricReferenceError, match="purpose"):
        dry_run(
            "create",
            profile_id="reference-profile-one",
            person_ref_id="person-ref-one",
            sources=sources,
            approval=ordinary_confirmation,
            runtime_root=root,
        )

    private_sources = [
        {**sources[0], "contact_email": "private@example.invalid"}
    ]
    with pytest.raises(BiometricReferenceError, match="forbidden"):
        source_set_sha256(private_sources, test_mode=True)

    invalid_sources = [dict(sources[0])]
    invalid_sources[0]["end_seconds"] = float("nan")
    with pytest.raises(BiometricReferenceError, match="bounds"):
        source_set_sha256(invalid_sources, test_mode=True)

    duplicate_sources = [sources[0], dict(sources[0])]
    duplicate_sources[1]["reference_id"] = "reference-b"
    duplicate_sources[1]["fixture_authority"] = {
        **duplicate_sources[1]["fixture_authority"],
        "fixture_id": "reference-b",
    }
    with pytest.raises(BiometricReferenceError, match="duplicated"):
        source_set_sha256(duplicate_sources, test_mode=True)

    asserted_lineage = [dict(sources[0])]
    asserted_lineage[0].pop("fixture_authority")
    asserted_lineage[0]["lineage"] = {
        "authority": "p1_audio_derivative_replay",
        "run_id": "audio-run-asserted",
        "runtime_root": str(tmp_path / "missing-p1"),
        "replay_receipt_sha256": "7" * 64,
        "validation_status": "verified_active",
    }
    with pytest.raises(BiometricReferenceError, match="lineage request"):
        source_set_sha256(asserted_lineage)


def test_cross_person_source_claim_and_wrong_token_fail_without_state_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    claimed = [source("reference-a", session="session-one")]
    plan = dry_run(
        "create",
        profile_id="reference-profile-two",
        person_ref_id="person-ref-two",
        sources=claimed,
        approval=approval(
            "create",
            "reference-profile-two",
            "person-ref-two",
            sources=claimed,
            expected_generation_id=None,
            suffix="cross-person",
        ),
        runtime_root=root,
    )
    with pytest.raises(BiometricReferenceError, match="requires token"):
        apply_change(plan["run_id"], approval_token="", runtime_root=root)
    with pytest.raises(BiometricReferenceError, match="another person"):
        apply_change(
            plan["run_id"],
            approval_token=plan["required_approval_token"],
            sources=claimed,
            runtime_root=root,
        )
    assert replay_reference(created["profile_id"], runtime_root=root)[
        "lifecycle_state"
    ] == "verified_active"
    with sqlite3.connect(root / "references.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM profiles WHERE profile_id = 'reference-profile-two'"
        ).fetchone()[0] == 0


def test_concurrent_supersede_has_one_cas_winner(tmp_path: Path) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    current = replay_reference(created["profile_id"], runtime_root=root)
    plans: list[tuple[dict, list[dict]]] = []
    for suffix in ("c", "d"):
        sources = [source(f"reference-{suffix}", session=f"session-{suffix}")]
        plans.append(
            (
                dry_run(
                    "supersede",
                    profile_id=current["profile_id"],
                    sources=sources,
                    approval=approval(
                        "supersede",
                        current["profile_id"],
                        current["person_ref_id"],
                        sources=sources,
                        expected_generation_id=current["head_generation_id"],
                        suffix=f"race-{suffix}",
                    ),
                    runtime_root=root,
                ),
                sources,
            )
        )

    def attempt(item: tuple[dict, list[dict]]) -> str:
        plan, sources = item
        try:
            apply_change(
                plan["run_id"],
                approval_token=plan["required_approval_token"],
                sources=sources,
                runtime_root=root,
            )
        except BiometricReferenceError as exc:
            return str(exc)
        return "success"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, plans))
    assert outcomes.count("success") == 1
    assert sum("changed after dry run" in outcome for outcome in outcomes) == 1
    replay = replay_reference(created["profile_id"], runtime_root=root)
    assert replay["generation_count"] == 2
    assert replay["eligible_for_materialization"] is True


def test_transaction_abort_rolls_back_partial_create(tmp_path: Path) -> None:
    root = tmp_path / "references"
    create_profile(root)
    sources = [source("reference-c", session="session-three")]
    plan = dry_run(
        "create",
        profile_id="reference-profile-two",
        person_ref_id="person-ref-two",
        sources=sources,
        approval=approval(
            "create",
            "reference-profile-two",
            "person-ref-two",
            sources=sources,
            expected_generation_id=None,
            suffix="abort",
        ),
        runtime_root=root,
    )
    database = root / "references.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER abort_second_event BEFORE INSERT ON events
            WHEN NEW.profile_id = 'reference-profile-two'
            BEGIN SELECT RAISE(ABORT, 'synthetic crash'); END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic crash"):
        apply_change(
            plan["run_id"],
            approval_token=plan["required_approval_token"],
            sources=sources,
            runtime_root=root,
        )
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM profiles WHERE profile_id = 'reference-profile-two'"
        ).fetchone()[0] == 0
        connection.execute("DROP TRIGGER abort_second_event")


def test_tamper_permissions_and_hardlinks_fail_replay(tmp_path: Path) -> None:
    tamper_root = tmp_path / "tamper"
    _, created = create_profile(tamper_root)
    database = tamper_root / "references.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE events SET payload_json = '{}' WHERE profile_id = ? AND sequence = 1",
            (created["profile_id"],),
        )
    with pytest.raises(BiometricReferenceError, match="event chain"):
        replay_reference(created["profile_id"], runtime_root=tamper_root)

    permission_root = tmp_path / "permissions"
    _, permission_created = create_profile(permission_root)
    permission_database = permission_root / "references.sqlite3"
    permission_database.chmod(0o644)
    with pytest.raises(BiometricReferenceError):
        replay_reference(permission_created["profile_id"], runtime_root=permission_root)

    hardlink_root = tmp_path / "hardlink"
    _, hardlink_created = create_profile(hardlink_root)
    hardlink_database = hardlink_root / "references.sqlite3"
    os.link(hardlink_database, hardlink_root / "database-alias.sqlite3")
    with pytest.raises(BiometricReferenceError, match="hard-linked"):
        replay_reference(hardlink_created["profile_id"], runtime_root=hardlink_root)


def test_dry_run_and_apply_source_drift_fail_before_profile_creation(
    tmp_path: Path,
) -> None:
    sources = [source("reference-a", session="session-one")]
    tamper_root = tmp_path / "plan-tamper"
    plan = dry_run(
        "create",
        profile_id="reference-profile-one",
        person_ref_id="person-ref-one",
        sources=sources,
        approval=approval(
            "create",
            "reference-profile-one",
            "person-ref-one",
            sources=sources,
            expected_generation_id=None,
            suffix="tamper",
        ),
        runtime_root=tamper_root,
    )
    plan_path = Path(plan["dry_run_path"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["will_read_audio"] = True
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    forged_token = (
        f"CREATE_BIOMETRIC_REFERENCE:{plan['run_id']}:{sha256_file(plan_path)}"
    )
    with pytest.raises(BiometricReferenceError, match="exceeds P3 scope"):
        apply_change(
            plan["run_id"],
            approval_token=forged_token,
            sources=sources,
            runtime_root=tamper_root,
        )
    assert not (tamper_root / "references.sqlite3").exists()

    drift_root = tmp_path / "source-drift"
    drift_plan = dry_run(
        "create",
        profile_id="reference-profile-one",
        person_ref_id="person-ref-one",
        sources=sources,
        approval=approval(
            "create",
            "reference-profile-one",
            "person-ref-one",
            sources=sources,
            expected_generation_id=None,
            suffix="drift",
        ),
        runtime_root=drift_root,
    )
    changed = [dict(sources[0])]
    changed[0]["session_id"] = "different-session"
    with pytest.raises(BiometricReferenceError, match="differs from the dry run"):
        apply_change(
            drift_plan["run_id"],
            approval_token=drift_plan["required_approval_token"],
            sources=changed,
            runtime_root=drift_root,
        )
    assert list((drift_root / "attempts").glob("*.json"))
    with sqlite3.connect(drift_root / "references.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0


def test_writable_database_rejects_dangling_symlink_and_late_hardlink(
    tmp_path: Path,
) -> None:
    sources = [source("reference-a", session="session-one")]
    symlink_root = tmp_path / "symlink"
    plan = dry_run(
        "create",
        profile_id="reference-profile-one",
        person_ref_id="person-ref-one",
        sources=sources,
        approval=approval(
            "create",
            "reference-profile-one",
            "person-ref-one",
            sources=sources,
            expected_generation_id=None,
            suffix="symlink",
        ),
        runtime_root=symlink_root,
    )
    external = tmp_path / "escaped.sqlite3"
    (symlink_root / "references.sqlite3").symlink_to(external)
    with pytest.raises(BiometricReferenceError, match="must not be a symlink"):
        apply_change(
            plan["run_id"],
            approval_token=plan["required_approval_token"],
            sources=sources,
            runtime_root=symlink_root,
        )
    assert not external.exists()

    hardlink_root = tmp_path / "hardlink-apply"
    _, created = create_profile(hardlink_root)
    current = replay_reference(created["profile_id"], runtime_root=hardlink_root)
    replacement = [source("reference-c", session="session-three")]
    supersede_plan = dry_run(
        "supersede",
        profile_id=current["profile_id"],
        sources=replacement,
        approval=approval(
            "supersede",
            current["profile_id"],
            current["person_ref_id"],
            sources=replacement,
            expected_generation_id=current["head_generation_id"],
            suffix="hardlink",
        ),
        runtime_root=hardlink_root,
    )
    os.link(
        hardlink_root / "references.sqlite3",
        hardlink_root / "references-alias.sqlite3",
    )
    with pytest.raises(BiometricReferenceError, match="hard-linked"):
        apply_change(
            supersede_plan["run_id"],
            approval_token=supersede_plan["required_approval_token"],
            sources=replacement,
            runtime_root=hardlink_root,
        )


def test_overlap_claim_and_approval_reuse_are_scope_bound(tmp_path: Path) -> None:
    root = tmp_path / "references"
    create_profile(root)
    overlapping = source("reference-z", session="changed-session")
    overlapping["source_sha256"] = "a" * 64
    overlapping["speaker_label_id"] = "changed-speaker-label"
    overlapping["recording_id"] = "changed-recording"
    overlapping["conversation_id"] = "changed-conversation"
    overlapping["start_seconds"] = "0"
    overlapping["end_seconds"] = "2.0"
    overlapping["fixture_authority"] = {
        **overlapping["fixture_authority"],
        "source_sha256": overlapping["source_sha256"],
    }
    overlap_sources = [overlapping]
    overlap_plan = dry_run(
        "create",
        profile_id="reference-profile-two",
        person_ref_id="person-ref-two",
        sources=overlap_sources,
        approval=approval(
            "create",
            "reference-profile-two",
            "person-ref-two",
            sources=overlap_sources,
            expected_generation_id=None,
            suffix="overlap",
        ),
        runtime_root=root,
    )
    with pytest.raises(BiometricReferenceError, match="another person"):
        apply_change(
            overlap_plan["run_id"],
            approval_token=overlap_plan["required_approval_token"],
            sources=overlap_sources,
            runtime_root=root,
        )

    distinct = [source("reference-c", session="session-three")]
    reused = approval(
        "create",
        "reference-profile-three",
        "person-ref-three",
        sources=distinct,
        expected_generation_id=None,
        suffix="create",
    )
    reuse_plan = dry_run(
        "create",
        profile_id="reference-profile-three",
        person_ref_id="person-ref-three",
        sources=distinct,
        approval=reused,
        runtime_root=root,
    )
    with pytest.raises(BiometricReferenceError, match="already consumed"):
        apply_change(
            reuse_plan["run_id"],
            approval_token=reuse_plan["required_approval_token"],
            sources=distinct,
            runtime_root=root,
        )


def test_historical_replay_and_idempotence_survive_new_profile_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    create_plan, created = create_profile(root)
    active = replay_reference(created["profile_id"], runtime_root=root)
    _, withdrawn = transition(root, "withdraw", active, suffix="withdraw-old")
    inactive = replay_reference(withdrawn["profile_id"], runtime_root=root)
    transition(root, "delete", inactive, suffix="delete-old")

    new_sources = [source("reference-c", session="session-three")]
    _, replacement = plan_and_apply(
        root,
        "create",
        profile_id="reference-profile-new",
        person_ref_id="person-ref-one",
        sources=new_sources,
        expected_generation_id=None,
        suffix="reenroll",
    )
    assert replay_reference(created["profile_id"], runtime_root=root)[
        "lifecycle_state"
    ] == "verified_deleted"
    assert replay_reference(replacement["profile_id"], runtime_root=root)[
        "lifecycle_state"
    ] == "verified_active"
    historical = apply_change(
        create_plan["run_id"],
        approval_token=create_plan["required_approval_token"],
        runtime_root=root,
    )
    assert historical["historical_lifecycle_state"] == "active"
    assert historical["lifecycle_state"] == "verified_deleted"


def test_coordinated_descendant_row_tamper_cannot_replace_receipt_anchor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    resolved = resolve_eligible_reference("person-ref-one", runtime_root=root)
    descendant_id = "materialized-profile-one"
    artifact_sha = "e" * 64
    materialization = materialization_receipt(
        resolved, descendant_id, artifact_sha
    )
    authority_path, authority_root = p4_anchor(root, materialization)
    registered = register_descendant(
        resolved["profile_id"],
        resolved["generation_id"],
        descendant_id,
        artifact_sha,
        materialization_receipt=materialization,
        authority_receipt_path=authority_path,
        p4_authority_root=authority_root,
        approval_token=registration_token(
            resolved, descendant_id, artifact_sha
        ),
        runtime_root=root,
    )
    promote_descendant(root, registered)
    database = root / "references.sqlite3"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT promotion_receipt_json FROM descendants WHERE descendant_id = ?",
            (descendant_id,),
        ).fetchone()
        forged = json.loads(row[0])
        forged["promoted_at"] = "2026-07-31T13:00:00Z"
        canonical = json.dumps(
            forged, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        connection.execute(
            """
            UPDATE descendants SET promotion_receipt_json = ?,
            promotion_receipt_sha256 = ? WHERE descendant_id = ?
            """,
            (
                json.dumps(forged, sort_keys=True, separators=(",", ":")),
                hashlib.sha256(canonical).hexdigest(),
                descendant_id,
            ),
        )
    with pytest.raises(BiometricReferenceError, match="Private artifact|anchor|authority"):
        replay_reference(created["profile_id"], runtime_root=root)


@pytest.mark.parametrize("table", ["descendants", "idempotency", "approvals"])
def test_replay_detects_deleted_append_only_inventory_rows(
    tmp_path: Path, table: str
) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    if table == "descendants":
        resolved = resolve_eligible_reference("person-ref-one", runtime_root=root)
        descendant_id = "materialized-profile-one"
        artifact_sha = "e" * 64
        receipt = materialization_receipt(resolved, descendant_id, artifact_sha)
        authority_path, authority_root = p4_anchor(root, receipt)
        register_descendant(
            resolved["profile_id"], resolved["generation_id"], descendant_id,
            artifact_sha, materialization_receipt=receipt,
            authority_receipt_path=authority_path,
            p4_authority_root=authority_root,
            approval_token=registration_token(resolved, descendant_id, artifact_sha),
            runtime_root=root,
        )
    with sqlite3.connect(root / "references.sqlite3") as connection:
        connection.execute(f"DELETE FROM {table}")
    with pytest.raises(BiometricReferenceError, match="inventory|approval claim"):
        replay_reference(created["profile_id"], runtime_root=root)


def test_delete_requires_withdraw_and_descendant_acknowledgment(tmp_path: Path) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    active = replay_reference(created["profile_id"], runtime_root=root)
    resolved = resolve_eligible_reference("person-ref-one", runtime_root=root)
    descendant_id = "materialized-profile-one"
    artifact_sha = "e" * 64
    receipt = materialization_receipt(resolved, descendant_id, artifact_sha)
    authority_path, authority_root = p4_anchor(root, receipt)
    registered = register_descendant(
        resolved["profile_id"], resolved["generation_id"], descendant_id,
        artifact_sha, materialization_receipt=receipt,
        authority_receipt_path=authority_path, p4_authority_root=authority_root,
        approval_token=registration_token(resolved, descendant_id, artifact_sha),
        runtime_root=root,
    )
    promote_descendant(root, registered)
    delete_approval = approval(
        "delete", active["profile_id"], active["person_ref_id"], sources=None,
        expected_generation_id=active["head_generation_id"], suffix="early-delete",
    )
    with pytest.raises(BiometricReferenceError, match="withdrawn before deletion"):
        dry_run(
            "delete", profile_id=active["profile_id"], approval=delete_approval,
            runtime_root=root,
        )
    _, withdrawn = transition(root, "withdraw", active, suffix="withdraw-first")
    inactive = replay_reference(withdrawn["profile_id"], runtime_root=root)
    with pytest.raises(BiometricReferenceError, match="acknowledgments"):
        dry_run(
            "delete", profile_id=inactive["profile_id"],
            approval=approval(
                "delete", inactive["profile_id"], inactive["person_ref_id"],
                sources=None, expected_generation_id=inactive["head_generation_id"],
                suffix="pending-delete",
            ), runtime_root=root,
        )
    invalidate_descendant(root, descendant_id, artifact_sha, "reference_withdrawn")
    transition(
        root, "delete", replay_reference(created["profile_id"], runtime_root=root),
        suffix="acknowledged-delete",
    )


def test_production_source_requires_lineage_and_source_manifest_is_authoritative(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    synthetic = source("reference-a", session="session-one")
    production = dict(synthetic)
    production.pop("fixture_authority")
    with pytest.raises(BiometricReferenceError, match="replay-validated lineage"):
        source_set_sha256([production])
    plan, _ = create_profile(root)
    manifest_path = Path(plan["source_manifest_path"])
    os.chmod(manifest_path, 0o600)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(BiometricReferenceError, match="source manifest"):
        apply_change(
            plan["run_id"], approval_token=plan["required_approval_token"],
            sources=[
                source("reference-a", session="session-one"),
                source("reference-b", session="session-two", offset=3.0),
            ], runtime_root=root,
        )


def test_post_commit_receipt_publication_recovers_from_prepared_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "references"
    sources = [source("reference-a", session="session-one")]
    plan = dry_run(
        "create", profile_id="reference-profile-one", person_ref_id="person-ref-one",
        sources=sources,
        approval=approval(
            "create", "reference-profile-one", "person-ref-one", sources=sources,
            expected_generation_id=None, suffix="commit-recovery",
        ), runtime_root=root,
    )
    original = biometric_authority._promote_staged_receipt
    with monkeypatch.context() as scoped:
        scoped.setattr(
            biometric_authority, "_promote_staged_receipt",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupt")),
        )
        with pytest.raises(RuntimeError, match="interrupt"):
            apply_change(
                plan["run_id"], approval_token=plan["required_approval_token"],
                sources=sources, runtime_root=root,
            )
    monkeypatch.setattr(biometric_authority, "_promote_staged_receipt", original)
    recovered = apply_change(
        plan["run_id"], approval_token=plan["required_approval_token"],
        runtime_root=root,
    )
    assert recovered["idempotent_replay"] is True
    assert Path(recovered["receipt_anchor_path"]).is_file()


def test_reference_manifest_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    with sqlite3.connect(root / "references.sqlite3") as connection:
        row = connection.execute(
            "SELECT generation_id, manifest_json FROM generations"
        ).fetchone()
        manifest = json.loads(row[1])
        manifest["unexpected"] = True
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        connection.execute(
            "UPDATE generations SET manifest_json = ?, manifest_sha256 = ? WHERE generation_id = ?",
            (canonical.decode("utf-8"), hashlib.sha256(canonical).hexdigest(), row[0]),
        )
    with pytest.raises(BiometricReferenceError, match="schema"):
        replay_reference(created["profile_id"], runtime_root=root)


def test_generation_manifest_cannot_replace_immutable_creation_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    with sqlite3.connect(root / "references.sqlite3") as connection:
        row = connection.execute(
            "SELECT generation_id, manifest_json FROM generations"
        ).fetchone()
        manifest = json.loads(row[1])
        manifest["created_at"] = "2026-07-31T12:59:59Z"
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        connection.execute(
            "UPDATE generations SET manifest_json = ?, manifest_sha256 = ? WHERE generation_id = ?",
            (canonical.decode("utf-8"), hashlib.sha256(canonical).hexdigest(), row[0]),
        )
    with pytest.raises(BiometricReferenceError, match="creation binding"):
        replay_reference(created["profile_id"], runtime_root=root)


def test_public_eligibility_queries_are_filesystem_read_only(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    assert descendant_is_eligible("missing-descendant", runtime_root=missing_root) is False
    assert not missing_root.exists()
    with pytest.raises(BiometricReferenceError, match="does not exist"):
        resolve_eligible_reference("person-ref-one", runtime_root=missing_root)
    assert not missing_root.exists()

    root = tmp_path / "references"
    _, created = create_profile(root)
    database = root / "references.sqlite3"
    before = database.stat()
    resolved = resolve_eligible_reference("person-ref-one", runtime_root=root)
    assert resolved["profile_id"] == created["profile_id"]
    assert descendant_is_eligible("missing-descendant", runtime_root=root) is False
    after = database.stat()
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
    )


def test_tombstone_hashes_and_delete_timestamp_bind_to_delete_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    _, created = create_profile(root)
    active = replay_reference(created["profile_id"], runtime_root=root)
    _, withdrawn = transition(root, "withdraw", active, suffix="tombstone-withdraw")
    inactive = replay_reference(withdrawn["profile_id"], runtime_root=root)
    transition(root, "delete", inactive, suffix="tombstone-delete")
    with sqlite3.connect(root / "references.sqlite3") as connection:
        row = connection.execute(
            "SELECT generation_id, manifest_json FROM generations ORDER BY sequence LIMIT 1"
        ).fetchone()
        tombstone = json.loads(row[1])
        tombstone["deleted_at"] = "2026-07-31T23:59:59Z"
        canonical = json.dumps(
            tombstone, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        connection.execute(
            "UPDATE generations SET manifest_json = ?, manifest_sha256 = ? WHERE generation_id = ?",
            (canonical.decode("utf-8"), hashlib.sha256(canonical).hexdigest(), row[0]),
        )
    with pytest.raises(BiometricReferenceError, match="tombstone"):
        replay_reference(created["profile_id"], runtime_root=root)
