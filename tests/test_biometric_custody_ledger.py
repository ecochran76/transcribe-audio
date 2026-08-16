from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
from biometric_custody_ledger import BiometricCustodyLedger


def _ledger(tmp_path: Path) -> BiometricCustodyLedger:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    private_root = tmp_path / "private-biometric"
    private_root.mkdir(mode=0o700)
    return BiometricCustodyLedger(tmp_path, private_root=private_root)


def _a3_fixture(name: str) -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "dev"
        / "fixtures"
        / "plan-0072-a3"
        / name
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(
    ledger: BiometricCustodyLedger,
    *,
    suffix: str,
    reviewed_person_id: str = "",
    eligible: bool = True,
) -> object:
    payload = f"synthetic-sample-{suffix}".encode()
    private_ref = ledger.store_private_object(
        object_id=f"sample-object-{suffix}",
        payload=payload,
    )
    return ledger.register_sample(
        conversation_id=f"conversation-{suffix}",
        recording_id=f"recording-{suffix}",
        speaker_ref="SPEAKER_1",
        start_ms=0,
        end_ms=1200,
        source_media_sha256=hashlib.sha256(f"media-{suffix}".encode()).hexdigest(),
        sample_sha256=hashlib.sha256(payload).hexdigest(),
        quality={
            "eligible": eligible,
            "reason_codes": [] if eligible else ["synthetic_quality_failure"],
        },
        preparation_lineage={"recipe_version": "synthetic-v1"},
        review_state="reviewed" if reviewed_person_id else "unreviewed",
        person_id=reviewed_person_id,
        review_authority_id=(f"review-{suffix}" if reviewed_person_id else ""),
        consent_authority=("consent-redacted-v1" if reviewed_person_id else ""),
        private_object_id=private_ref.object_id,
        private_object_sha256=private_ref.sha256,
        created_at="2026-08-16T21:10:00Z",
    )


def test_unreviewed_sample_is_person_unbound_and_private_object_is_hash_bound(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    payload = b"synthetic-not-a-real-voice-sample"
    private_ref = ledger.store_private_object(
        object_id="sample-object-redacted-1",
        payload=payload,
    )

    sample = ledger.register_sample(
        conversation_id="00000000-0000-4000-8000-000000000301",
        recording_id="00000000-0000-4000-8000-000000000302",
        speaker_ref="SPEAKER_1",
        start_ms=100,
        end_ms=1800,
        source_media_sha256="a" * 64,
        sample_sha256=hashlib.sha256(payload).hexdigest(),
        quality={"eligible": True, "reason_codes": []},
        preparation_lineage={"recipe_version": "synthetic-v1"},
        review_state="unreviewed",
        private_object_id=private_ref.object_id,
        private_object_sha256=private_ref.sha256,
        created_at="2026-08-16T21:00:00Z",
    )

    loaded = ledger.load_sample(sample.sample_id)
    object_path = tmp_path / "private-biometric" / "objects" / private_ref.object_id
    assert loaded["person_id"] is None
    assert loaded["review_authority_id"] is None
    assert "private-biometric" not in json.dumps(loaded)
    assert object_path.read_bytes() == payload
    assert stat.S_IMODE(object_path.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="reviewed identity"):
        ledger.register_sample(
            conversation_id="00000000-0000-4000-8000-000000000301",
            recording_id="00000000-0000-4000-8000-000000000302",
            speaker_ref="SPEAKER_1",
            start_ms=2000,
            end_ms=3000,
            source_media_sha256="a" * 64,
            sample_sha256="b" * 64,
            quality={"eligible": True, "reason_codes": []},
            preparation_lineage={"recipe_version": "synthetic-v1"},
            review_state="unreviewed",
            person_id="person-redacted-1",
            private_object_id=private_ref.object_id,
            private_object_sha256=private_ref.sha256,
            created_at="2026-08-16T21:01:00Z",
        )


def test_reviewed_sample_requires_authority_and_is_append_only(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    payload = b"synthetic-reviewed-sample"
    private_ref = ledger.store_private_object(
        object_id="sample-object-redacted-2",
        payload=payload,
    )
    sample = ledger.register_sample(
        conversation_id="00000000-0000-4000-8000-000000000303",
        recording_id="00000000-0000-4000-8000-000000000304",
        speaker_ref="SPEAKER_2",
        start_ms=0,
        end_ms=1500,
        source_media_sha256="c" * 64,
        sample_sha256=hashlib.sha256(payload).hexdigest(),
        quality={"eligible": True, "reason_codes": []},
        preparation_lineage={"recipe_version": "synthetic-v1"},
        review_state="reviewed",
        person_id="person-redacted-2",
        review_authority_id="review-redacted-2",
        consent_authority="consent-redacted-v1",
        private_object_id=private_ref.object_id,
        private_object_sha256=private_ref.sha256,
        created_at="2026-08-16T21:02:00Z",
    )

    assert ledger.load_sample(sample.sample_id)["person_id"] == (
        "person-redacted-2"
    )
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute(
                "UPDATE knowledge_voice_samples SET person_id = 'tampered'"
            )


def test_anonymous_cluster_memberships_are_soft_ranked_and_reversible(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    primary = _sample(ledger, suffix="cluster-primary")
    alternative = _sample(ledger, suffix="cluster-alternative")

    first = ledger.record_cluster_version(
        cluster_id="anonymous-cluster-redacted-1",
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": primary.sample_id,
                "rank": 1,
                "score": 0.91,
                "evidence_ids": ["pairwise-redacted-1"],
                "membership_state": "candidate",
            },
            {
                "sample_id": alternative.sample_id,
                "rank": 2,
                "score": 0.72,
                "evidence_ids": ["pairwise-redacted-2"],
                "membership_state": "candidate",
            },
        ),
        status="candidate",
        created_at="2026-08-16T21:11:00Z",
    )
    reviewed = ledger.record_cluster_version(
        cluster_id="anonymous-cluster-redacted-1",
        predecessor_version_id=first.cluster_version_id,
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": primary.sample_id,
                "rank": 1,
                "score": 0.91,
                "evidence_ids": ["pairwise-redacted-1"],
                "membership_state": "confirmed",
            },
            {
                "sample_id": alternative.sample_id,
                "rank": 2,
                "score": 0.72,
                "evidence_ids": ["pairwise-redacted-2"],
                "membership_state": "rejected",
            },
        ),
        status="reviewed",
        created_at="2026-08-16T21:12:00Z",
    )

    loaded = ledger.load_cluster_version(reviewed.cluster_version_id)
    assert loaded["predecessor_version_id"] == first.cluster_version_id
    assert [item["membership_state"] for item in loaded["memberships"]] == [
        "confirmed",
        "rejected",
    ]
    assert all("person_id" not in item for item in loaded["memberships"])
    assert ledger.load_sample(primary.sample_id)["person_id"] is None
    assert ledger.load_sample(alternative.sample_id)["person_id"] is None


def test_profile_version_requires_exact_reviewed_allowlist_and_activation_event(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "person-redacted-profile-1"
    first = _sample(ledger, suffix="profile-first", reviewed_person_id=person_id)
    second = _sample(ledger, suffix="profile-second", reviewed_person_id=person_id)
    unreviewed = _sample(ledger, suffix="profile-unreviewed")
    ineligible = _sample(
        ledger,
        suffix="profile-ineligible",
        reviewed_person_id=person_id,
        eligible=False,
    )
    family = ledger.register_profile_family(
        person_id=person_id,
        family_key="studio-en-us",
        conditions={"device": "synthetic", "language": "en-US"},
        created_at="2026-08-16T21:13:00Z",
    )
    profile_payload = b"synthetic-profile-payload"
    private_ref = ledger.store_private_object(
        object_id="profile-object-redacted-1",
        payload=profile_payload,
    )
    with pytest.raises(ValueError, match="reviewed eligible samples"):
        ledger.build_profile_version(
            profile_family_id=family.profile_family_id,
            sample_ids=(first.sample_id, unreviewed.sample_id),
            evaluation_id="evaluation-redacted-1",
            model_revision="model-synthetic-v1",
            recipe_revision="recipe-synthetic-v1",
            private_object_id=private_ref.object_id,
            private_object_sha256=private_ref.sha256,
            created_at="2026-08-16T21:14:00Z",
        )
    with pytest.raises(ValueError, match="reviewed eligible samples"):
        ledger.build_profile_version(
            profile_family_id=family.profile_family_id,
            sample_ids=(first.sample_id, ineligible.sample_id),
            evaluation_id="evaluation-redacted-1",
            model_revision="model-synthetic-v1",
            recipe_revision="recipe-synthetic-v1",
            private_object_id=private_ref.object_id,
            private_object_sha256=private_ref.sha256,
            created_at="2026-08-16T21:14:30Z",
        )

    profile = ledger.build_profile_version(
        profile_family_id=family.profile_family_id,
        sample_ids=(first.sample_id, second.sample_id),
        evaluation_id="evaluation-redacted-1",
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        private_object_id=private_ref.object_id,
        private_object_sha256=private_ref.sha256,
        created_at="2026-08-16T21:15:00Z",
    )
    assert ledger.profile_state(profile.profile_version_id)["status"] == "pending"
    activation = ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="activate",
        reason_code="reviewed_evaluation_pass",
        authority_id="dashboard-review-redacted-1",
        idempotency_key="activate-profile-redacted-1",
        created_at="2026-08-16T21:16:00Z",
    )

    state = ledger.profile_state(profile.profile_version_id)
    assert activation.action == "activate"
    assert state["status"] == "active"
    assert tuple(state["sample_ids"]) == tuple(
        sorted((first.sample_id, second.sample_id))
    )
    rebuilt_equal = ledger.store_private_object(
        object_id="profile-object-redacted-1-rebuild",
        payload=profile_payload,
    )
    equal_receipt = ledger.verify_profile_rebuild(
        profile_version_id=profile.profile_version_id,
        rebuilt_object_id=rebuilt_equal.object_id,
        rebuilt_object_sha256=rebuilt_equal.sha256,
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        created_at="2026-08-16T21:17:00Z",
    )
    rebuilt_drift = ledger.store_private_object(
        object_id="profile-object-redacted-1-drift",
        payload=b"synthetic-profile-payload-drift",
    )
    drift_receipt = ledger.verify_profile_rebuild(
        profile_version_id=profile.profile_version_id,
        rebuilt_object_id=rebuilt_drift.object_id,
        rebuilt_object_sha256=rebuilt_drift.sha256,
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        created_at="2026-08-16T21:18:00Z",
    )
    superseded = ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="supersede",
        reason_code="successor_selected",
        authority_id="dashboard-review-redacted-2",
        idempotency_key="supersede-profile-redacted-1",
        supersedes_event_id=activation.event_id,
        created_at="2026-08-16T21:19:00Z",
    )
    assert ledger.profile_state(profile.profile_version_id)["status"] == (
        "superseded"
    )
    ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="rollback",
        reason_code="successor_rollback",
        authority_id="dashboard-review-redacted-3",
        idempotency_key="rollback-profile-redacted-1",
        supersedes_event_id=superseded.event_id,
        created_at="2026-08-16T21:20:00Z",
    )

    assert equal_receipt.byte_equal is True
    assert drift_receipt.byte_equal is False
    assert ledger.profile_state(profile.profile_version_id)["status"] == "active"


def test_person_exclusion_invalidates_samples_and_profile_without_deleting_bytes(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "person-redacted-exclusion-1"
    sample = _sample(ledger, suffix="exclude-person", reviewed_person_id=person_id)
    family = ledger.register_profile_family(
        person_id=person_id,
        family_key="default",
        conditions={},
        created_at="2026-08-16T21:20:00Z",
    )
    profile_ref = ledger.store_private_object(
        object_id="profile-object-exclusion-1",
        payload=b"synthetic-profile-exclusion",
    )
    profile = ledger.build_profile_version(
        profile_family_id=family.profile_family_id,
        sample_ids=(sample.sample_id,),
        evaluation_id="evaluation-exclusion-1",
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        private_object_id=profile_ref.object_id,
        private_object_sha256=profile_ref.sha256,
        created_at="2026-08-16T21:21:00Z",
    )
    active = ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="activate",
        reason_code="reviewed_evaluation_pass",
        authority_id="review-exclusion-1",
        idempotency_key="activate-exclusion-1",
        created_at="2026-08-16T21:22:00Z",
    )

    preview = ledger.preview_effect(
        mode="exclude",
        target_type="person",
        target_id=person_id,
    )
    receipt = ledger.apply_effect(
        preview=preview,
        authority_id="privacy-control-redacted-1",
        idempotency_key="exclude-person-redacted-1",
        created_at="2026-08-16T21:23:00Z",
    )

    assert receipt.status == "applied"
    replay = ledger.apply_effect(
        preview=preview,
        authority_id="privacy-control-redacted-1",
        idempotency_key="exclude-person-redacted-1",
        created_at="2026-08-16T21:23:00Z",
    )
    assert replay.status == "unchanged"
    assert replay.sample_event_ids == receipt.sample_event_ids
    with pytest.raises(ValueError, match="idempotency drifted"):
        ledger.apply_effect(
            preview=ledger.preview_effect(
                mode="exclude",
                target_type="profile",
                target_id=profile.profile_version_id,
            ),
            authority_id="privacy-control-redacted-1",
            idempotency_key="exclude-person-redacted-1",
            created_at="2026-08-16T21:23:00Z",
        )
    assert ledger.sample_state(sample.sample_id)["exclusion_state"] == "excluded"
    assert ledger.profile_state(profile.profile_version_id)["status"] == (
        "invalidated"
    )
    assert (
        tmp_path
        / "private-biometric"
        / "objects"
        / "sample-object-exclude-person"
    ).is_file()
    restored = ledger.record_sample_event(
        sample_id=sample.sample_id,
        event_type="restore",
        actor_id="operator-redacted",
        authority_id="privacy-control-redacted-2",
        idempotency_key="restore-person-sample-redacted-1",
        supersedes_event_id=receipt.sample_event_ids[0],
        created_at="2026-08-16T21:24:00Z",
    )
    assert restored.event_type == "restore"
    assert ledger.sample_state(sample.sample_id)["exclusion_state"] == "included"
    assert ledger.profile_state(profile.profile_version_id)["status"] == (
        "invalidated"
    )
    assert active.event_id != ledger.profile_state(profile.profile_version_id)[
        "event_id"
    ]


def test_sample_deletion_removes_active_bytes_and_preserves_minimal_tombstone(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "person-redacted-deletion-1"
    deleted_sample = _sample(
        ledger,
        suffix="delete-target",
        reviewed_person_id=person_id,
    )
    retained_sample = _sample(
        ledger,
        suffix="delete-retained",
        reviewed_person_id=person_id,
    )
    cluster = ledger.record_cluster_version(
        cluster_id="anonymous-cluster-deletion-1",
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": deleted_sample.sample_id,
                "rank": 1,
                "score": 0.9,
                "evidence_ids": ["pairwise-delete-1"],
                "membership_state": "candidate",
            },
            {
                "sample_id": retained_sample.sample_id,
                "rank": 2,
                "score": 0.8,
                "evidence_ids": ["pairwise-delete-2"],
                "membership_state": "candidate",
            },
        ),
        status="candidate",
        created_at="2026-08-16T21:25:00Z",
    )
    family = ledger.register_profile_family(
        person_id=person_id,
        family_key="deletion-family",
        conditions={},
        created_at="2026-08-16T21:26:00Z",
    )
    profile_ref = ledger.store_private_object(
        object_id="profile-object-deletion-1",
        payload=b"synthetic-profile-deletion",
    )
    profile = ledger.build_profile_version(
        profile_family_id=family.profile_family_id,
        sample_ids=(deleted_sample.sample_id, retained_sample.sample_id),
        evaluation_id="evaluation-deletion-1",
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        private_object_id=profile_ref.object_id,
        private_object_sha256=profile_ref.sha256,
        created_at="2026-08-16T21:27:00Z",
    )
    ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="activate",
        reason_code="reviewed_evaluation_pass",
        authority_id="review-deletion-1",
        idempotency_key="activate-deletion-1",
        created_at="2026-08-16T21:28:00Z",
    )
    preview = ledger.preview_effect(
        mode="delete",
        target_type="sample",
        target_id=deleted_sample.sample_id,
    )

    receipt = ledger.apply_effect(
        preview=preview,
        authority_id="deletion-authority-redacted-1",
        idempotency_key="delete-sample-redacted-1",
        created_at="2026-08-16T21:29:00Z",
    )
    replay = ledger.apply_effect(
        preview=preview,
        authority_id="deletion-authority-redacted-1",
        idempotency_key="delete-sample-redacted-1",
        created_at="2026-08-16T21:29:00Z",
    )

    assert receipt.status == "applied"
    assert replay.status == "unchanged"
    assert ledger.sample_state(deleted_sample.sample_id)["exclusion_state"] == (
        "deleted"
    )
    assert ledger.profile_state(profile.profile_version_id)["status"] == (
        "invalidated"
    )
    memberships = ledger.load_cluster_version(cluster.cluster_version_id)[
        "memberships"
    ]
    assert memberships[0]["effective_membership_state"] == "excluded"
    assert memberships[1]["effective_membership_state"] == "candidate"
    assert not (
        tmp_path
        / "private-biometric"
        / "objects"
        / "sample-object-delete-target"
    ).exists()
    assert (
        tmp_path
        / "private-biometric"
        / "objects"
        / "sample-object-delete-retained"
    ).is_file()
    tombstone = ledger.load_deletion_tombstone(receipt.tombstone_id)
    assert tombstone["target_type"] == "sample"
    assert tombstone["backup_disposition"] == "exclude_from_future_backups"
    assert (
        tombstone["historical_backup_disposition"]
        == "expire_on_retention_schedule"
    )
    assert tombstone["deleted_object_hashes"] == sorted(
        [deleted_sample.sample_sha256, profile_ref.sha256]
    )
    assert not (
        tmp_path
        / "private-biometric"
        / "objects"
        / "profile-object-deletion-1"
    ).exists()
    assert "object_id" not in json.dumps(tombstone)
    assert "private-biometric" not in json.dumps(tombstone)


def test_private_custody_rejects_broad_permissions_and_path_traversal(
    tmp_path: Path,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    broad = tmp_path / "broad-private"
    broad.mkdir(mode=0o755)
    broad.chmod(0o755)
    with pytest.raises(ValueError, match="group/world"):
        BiometricCustodyLedger(tmp_path, private_root=broad)
    broad.chmod(0o700)
    ledger = BiometricCustodyLedger(tmp_path, private_root=broad)
    with pytest.raises(ValueError, match="object ID"):
        ledger.store_private_object(object_id="../escape", payload=b"synthetic")


def test_effect_preview_rejects_newly_changed_scope_without_partial_effects(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = "person-stale-preview"
    first = _sample(ledger, suffix="stale-first", reviewed_person_id=person_id)
    preview = ledger.preview_effect(
        mode="exclude",
        target_type="person",
        target_id=person_id,
    )
    second = _sample(ledger, suffix="stale-second", reviewed_person_id=person_id)

    with pytest.raises(ValueError, match="stale"):
        ledger.apply_effect(
            preview=preview,
            authority_id="privacy-stale-authority",
            idempotency_key="stale-effect-1",
            created_at="2026-08-16T21:35:00Z",
        )

    assert ledger.sample_state(first.sample_id)["exclusion_state"] == "included"
    assert ledger.sample_state(second.sample_id)["exclusion_state"] == "included"


def test_unclustered_inventory_uses_current_soft_memberships(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    clustered = _sample(ledger, suffix="inventory-clustered")
    unclustered = _sample(ledger, suffix="inventory-unclustered")
    ledger.record_cluster_version(
        cluster_id="cluster-inventory-1",
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": clustered.sample_id,
                "rank": 1,
                "score": 0.8,
                "evidence_ids": ["inventory-evidence-1"],
                "membership_state": "candidate",
            },
        ),
        status="candidate",
        created_at="2026-08-16T21:36:00Z",
    )

    assert ledger.list_unclustered_samples() == (unclustered.sample_id,)


def test_confirmed_cluster_anchor_requeues_only_material_unreviewed_changes(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    anchor = _sample(
        ledger,
        suffix="rescore-anchor",
        reviewed_person_id="person-rescore-anchor",
    )
    material = _sample(ledger, suffix="rescore-material")
    stable = _sample(ledger, suffix="rescore-stable")
    cluster = ledger.record_cluster_version(
        cluster_id="cluster-rescore-1",
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": anchor.sample_id,
                "rank": 1,
                "score": 0.95,
                "evidence_ids": ["rescore-evidence-anchor"],
                "membership_state": "confirmed",
            },
            {
                "sample_id": material.sample_id,
                "rank": 2,
                "score": 0.60,
                "evidence_ids": ["rescore-evidence-material"],
                "membership_state": "candidate",
            },
            {
                "sample_id": stable.sample_id,
                "rank": 3,
                "score": 0.70,
                "evidence_ids": ["rescore-evidence-stable"],
                "membership_state": "candidate",
            },
        ),
        status="reviewed",
        created_at="2026-08-16T21:37:00Z",
    )

    receipt = ledger.record_cluster_rescore(
        cluster_version_id=cluster.cluster_version_id,
        anchor_sample_id=anchor.sample_id,
        score_updates=(
            {"sample_id": material.sample_id, "old_score": 0.60, "new_score": 0.82},
            {"sample_id": stable.sample_id, "old_score": 0.70, "new_score": 0.74},
        ),
        material_threshold=0.10,
        processing_version="cluster-rescore-synthetic-v1",
        created_at="2026-08-16T21:38:00Z",
    )

    assert receipt.requeued_sample_ids == (material.sample_id,)
    assert ledger.load_sample(material.sample_id)["person_id"] is None
    assert ledger.load_sample(stable.sample_id)["person_id"] is None


def test_plan0072_a3_redacted_fixture_replays_custody_and_deletion(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    fixture = _a3_fixture("custody-replay.json")
    deletion = _a3_fixture("deletion-preview.json")
    receipts = {
        str(item["suffix"]): _sample(
            ledger,
            suffix=str(item["suffix"]),
            reviewed_person_id=str(item["person_id"]),
        )
        for item in fixture["samples"]
    }
    memberships = tuple(
        {
            "sample_id": receipts[str(item["suffix"])].sample_id,
            "rank": rank,
            "score": item["score"],
            "evidence_ids": [f"fixture-evidence-{rank}"],
            "membership_state": item["membership_state"],
        }
        for rank, item in enumerate(fixture["samples"], start=1)
    )
    cluster_input = fixture["cluster"]
    cluster = ledger.record_cluster_version(
        cluster_id=str(cluster_input["cluster_id"]),
        algorithm_version=str(cluster_input["algorithm_version"]),
        memberships=memberships,
        status=str(cluster_input["status"]),
        created_at=str(fixture["created_at"]),
    )
    profile_input = fixture["profile"]
    reviewed = receipts["fixture-reviewed"]
    family = ledger.register_profile_family(
        person_id="person-redacted-fixture",
        family_key=str(profile_input["family_key"]),
        conditions=profile_input["conditions"],
        created_at=str(fixture["created_at"]),
    )
    profile_ref = ledger.store_private_object(
        object_id="profile-object-redacted-fixture",
        payload=b"synthetic-profile-redacted-fixture",
    )
    profile = ledger.build_profile_version(
        profile_family_id=family.profile_family_id,
        sample_ids=(reviewed.sample_id,),
        evaluation_id=str(profile_input["evaluation_id"]),
        model_revision=str(profile_input["model_revision"]),
        recipe_revision=str(profile_input["recipe_revision"]),
        private_object_id=profile_ref.object_id,
        private_object_sha256=profile_ref.sha256,
        created_at=str(fixture["created_at"]),
    )

    target = receipts[str(deletion["target_selector"])]
    preview = ledger.preview_effect(
        mode=str(deletion["mode"]),
        target_type=str(deletion["target_type"]),
        target_id=target.sample_id,
    )
    effect = ledger.apply_effect(
        preview=preview,
        authority_id="authority-redacted-fixture",
        idempotency_key="delete-redacted-fixture",
        created_at=str(fixture["created_at"]),
    )
    expected = deletion["expected"]
    tombstone = ledger.load_deletion_tombstone(effect.tombstone_id)
    loaded_cluster = ledger.load_cluster_version(cluster.cluster_version_id)

    assert ledger.sample_state(target.sample_id)["exclusion_state"] == expected[
        "sample_state"
    ]
    assert loaded_cluster["memberships"][1]["effective_membership_state"] == (
        expected["retained_cluster_membership_state"]
    )
    assert tombstone["backup_disposition"] == expected["backup_disposition"]
    assert tombstone["historical_backup_disposition"] == expected[
        "historical_backup_disposition"
    ]
    assert ledger.profile_state(profile.profile_version_id)["status"] == "pending"
    assert "private-biometric" not in json.dumps(tombstone)


@pytest.mark.parametrize(
    ("target_type", "sample_deleted", "profile_status", "cluster_status"),
    (
        ("sample", True, "invalidated", "active"),
        ("recording", True, "invalidated", "active"),
        ("person", True, "deleted", "active"),
        ("profile", False, "deleted", "active"),
        ("cluster", False, "active", "deleted"),
    ),
)
def test_every_initial_deletion_scope_has_bounded_effects(
    tmp_path: Path,
    target_type: str,
    sample_deleted: bool,
    profile_status: str,
    cluster_status: str,
) -> None:
    ledger = _ledger(tmp_path)
    person_id = f"person-scope-{target_type}"
    sample = _sample(
        ledger,
        suffix=f"scope-{target_type}",
        reviewed_person_id=person_id,
    )
    cluster_id = f"cluster-scope-{target_type}"
    cluster = ledger.record_cluster_version(
        cluster_id=cluster_id,
        algorithm_version="cluster-synthetic-v1",
        memberships=(
            {
                "sample_id": sample.sample_id,
                "rank": 1,
                "score": 0.9,
                "evidence_ids": [f"evidence-scope-{target_type}"],
                "membership_state": "candidate",
            },
        ),
        status="candidate",
        created_at="2026-08-16T21:30:00Z",
    )
    family = ledger.register_profile_family(
        person_id=person_id,
        family_key="scope-family",
        conditions={},
        created_at="2026-08-16T21:31:00Z",
    )
    profile_ref = ledger.store_private_object(
        object_id=f"profile-object-scope-{target_type}",
        payload=f"profile-scope-{target_type}".encode(),
    )
    profile = ledger.build_profile_version(
        profile_family_id=family.profile_family_id,
        sample_ids=(sample.sample_id,),
        evaluation_id=f"evaluation-scope-{target_type}",
        model_revision="model-synthetic-v1",
        recipe_revision="recipe-synthetic-v1",
        private_object_id=profile_ref.object_id,
        private_object_sha256=profile_ref.sha256,
        created_at="2026-08-16T21:32:00Z",
    )
    ledger.record_profile_event(
        profile_version_id=profile.profile_version_id,
        action="activate",
        reason_code="reviewed_evaluation_pass",
        authority_id=f"review-scope-{target_type}",
        idempotency_key=f"activate-scope-{target_type}",
        created_at="2026-08-16T21:33:00Z",
    )
    target_ids = {
        "sample": sample.sample_id,
        "recording": f"recording-scope-{target_type}",
        "person": person_id,
        "profile": profile.profile_version_id,
        "cluster": cluster_id,
    }
    preview = ledger.preview_effect(
        mode="delete",
        target_type=target_type,
        target_id=target_ids[target_type],
    )

    receipt = ledger.apply_effect(
        preview=preview,
        authority_id=f"delete-authority-scope-{target_type}",
        idempotency_key=f"delete-scope-{target_type}",
        created_at="2026-08-16T21:34:00Z",
    )

    assert bool(
        ledger.sample_state(sample.sample_id)["exclusion_state"] == "deleted"
    ) is sample_deleted
    assert ledger.profile_state(profile.profile_version_id)["status"] == (
        profile_status
    )
    assert ledger.cluster_state(cluster_id)["status"] == cluster_status
    assert ledger.load_deletion_tombstone(receipt.tombstone_id)["target_type"] == (
        target_type
    )
