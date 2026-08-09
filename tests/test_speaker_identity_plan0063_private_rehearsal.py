from __future__ import annotations

import copy
from pathlib import Path

import pytest

import speaker_identity_plan0063_human_review as review
import speaker_identity_plan0063_private_rehearsal as rehearsal
import transcript_store


def _review_inputs() -> tuple[dict, dict, dict[str, str]]:
    person_slots = (
        ("Person Alpha", ("doc0::SPEAKER_1", "doc0::SPEAKER_2")),
        ("Person Bravo", ("doc1::SPEAKER_1",)),
        ("Person Charlie", ("doc2::SPEAKER_1",)),
        ("Person Delta", ("doc3::SPEAKER_1", "doc3::SPEAKER_2")),
        (
            "Person Echo",
            (
                "47ea79857aa1ac2d1d79::SPEAKER_2",
                "47ea79857aa1ac2d1d79::SPEAKER_3",
            ),
        ),
        ("Person Foxtrot", ("doc5::SPEAKER_1",)),
    )
    people = []
    slot_identities = []
    for person_index, (name, slots) in enumerate(person_slots, 1):
        proposed_person_id = f"provisional-person-{person_index:024x}"
        people.append(
            {
                "proposed_person_id": proposed_person_id,
                "member_names": [name],
                "member_slot_ids": list(slots),
                "member_slot_person_ids": [
                    f"slot-person-{person_index:02d}-{slot_index:02d}"
                    for slot_index, _ in enumerate(slots, 1)
                ],
            }
        )
        for slot_index, slot_id in enumerate(slots, 1):
            slot_identities.append(
                {
                    "slot_id": slot_id,
                    "slot_person_id": f"slot-person-{person_index:02d}-{slot_index:02d}",
                    "name": name,
                    "normalized_name": name.casefold(),
                    "email": (
                        f"person{person_index}@example.test"
                        if slot_index == 1 or person_index == 4
                        else ""
                    ),
                    "organization": "Example Organization",
                    "decision_type": "operator_reviewed",
                }
            )

    merge_people = (people[0], people[3], people[4])
    merges = [
        {
            "merge_proposal_id": f"person-merge-{index:024x}",
            "proposed_person_id": person["proposed_person_id"],
            "basis": "operator_reviewed_same_name",
            "member_slot_ids": person["member_slot_ids"],
            "decision": "pending",
        }
        for index, person in enumerate(merge_people, 1)
    ]
    reconciled = {
        "content_sha256": review.RECONCILIATION_SHA256,
        "status": "pending_human_grouping_and_binding_review",
        "source_submission_sha256": "a" * 64,
        "negative_actions": review.NEGATIVE_ACTIONS,
        "slot_identities": slot_identities,
        "person_proposals": people,
        "merge_proposals": merges,
        "voice_binding_proposals": [
            {
                "binding_proposal_id": f"voice-person-binding-{1:024x}",
                "proposed_person_id": people[3]["proposed_person_id"],
                "acoustic_subject_id": "subject-example",
                "slot_id": people[3]["member_slot_ids"][0],
                "decision": "pending",
            }
        ],
    }

    candidate_indexes = (0, 1, 2, 4, 5)
    source_counts = (6, 6, 2, 6, 6)
    source_proposals = []
    clip_hashes: dict[str, str] = {}
    reference_number = 1
    for person_index, source_count in zip(candidate_indexes, source_counts):
        person = people[person_index]
        windows = []
        for window_index in range(source_count):
            reference_id = f"review-window-{reference_number:024x}"
            slot_id = person["member_slot_ids"][
                window_index % len(person["member_slot_ids"])
            ]
            speaker_label = slot_id.split("::", 1)[1]
            clip_hashes[reference_id] = f"{reference_number:064x}"
            windows.append(
                {
                    "reference_id": reference_id,
                    "recording_id": f"recording-{person_index}",
                    "source_blob_id": f"blob-{person_index}",
                    "slot_id": slot_id,
                    "speaker_label_id": speaker_label,
                    "start_seconds": float(reference_number * 10),
                    "end_seconds": float(reference_number * 10 + 5),
                    "source_sha256": f"{person_index + 1:064x}",
                    "audio_quality_sha256": f"{person_index + 11:064x}",
                    "lineage": {"authority": "p1_audio_derivative_replay"},
                    "future_holdout_excluded": True,
                    "data_split": "development_training_candidate",
                }
            )
            reference_number += 1
        source_proposals.append(
            {
                "proposed_person_id": person["proposed_person_id"],
                "member_slot_ids": person["member_slot_ids"],
                "device_metadata_status": "unverified",
                "status": "source_feasible_pending_human_review",
                "enrollment_authorized": False,
                "source_windows": windows,
            }
        )
    feasibility = {
        "content_sha256": review.FEASIBILITY_SHA256,
        "status": "source_feasibility_ready_pending_human_review",
        "reconciliation_content_sha256": review.RECONCILIATION_SHA256,
        "negative_actions": review.NEGATIVE_ACTIONS,
        "person_source_proposals": source_proposals,
    }
    return reconciled, feasibility, clip_hashes


def _manifest_and_submission() -> tuple[dict, dict, dict, dict]:
    reconciled, feasibility, clip_hashes = _review_inputs()
    manifest = review.build_review_manifest(
        reconciled,
        feasibility,
        clip_sha256_by_reference=clip_hashes,
        repository_authority={"commit": "fixture"},
    )
    decisions = [
        *manifest["merge_reviews"],
        *manifest["binding_reviews"],
        *[
            window
            for person in manifest["source_reviews"]
            for window in person["windows"]
        ],
    ]
    lines = [
        f"PLAN0063_SCHEMA={review.SUBMISSION_SCHEMA}",
        f"PLAN0063_P2_CONTENT_SHA256={manifest['reconciliation_content_sha256']}",
        f"PLAN0063_P3_CONTENT_SHA256={manifest['feasibility_content_sha256']}",
        f"PLAN0063_P4_CONTENT_SHA256={manifest['content_sha256']}",
        *[f"{item['decision_key']}={item['choices'][0]}" for item in decisions],
    ]
    submission = review.parse_review_submission("\n".join(lines), manifest)
    return reconciled, feasibility, manifest, submission


def _transition() -> dict:
    reconciled, feasibility, manifest, submission = _manifest_and_submission()
    return rehearsal.build_reviewed_transition(
        review_manifest=manifest,
        reconciliation=reconciled,
        feasibility=feasibility,
        submission=submission,
        reviewed_at="2026-08-09T12:00:00Z",
    )


def _initialize_legacy_store(root: Path) -> None:
    with transcript_store.connect(root) as connection:
        transcript_store.init_db(connection)
        connection.execute(
            """
            INSERT INTO contacts (
                id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "contact-fixture",
                "Existing Contact",
                "existing@example.test",
                "",
                "{}",
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO speaker_assignments (
                id, conversation_key, document_id, speaker_label, contact_id,
                contact_label, status, confidence, evidence_json, created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "assignment-fixture",
                "conversation-fixture",
                "document-fixture",
                "SPEAKER_1",
                "contact-fixture",
                "Existing Contact",
                "confirmed",
                1.0,
                "[]",
                "2026-08-09T00:00:00Z",
                "2026-08-09T00:00:00Z",
            ),
        )
        connection.commit()


def test_reviewed_transition_resolves_merges_binding_and_sources() -> None:
    transition = _transition()

    assert transition["status"] == "reviewed_transition_ready_for_private_rehearsal"
    assert transition["metrics"] == {
        "canonical_person_count": 6,
        "slot_binding_count": 9,
        "external_identity_count": 6,
        "accepted_merge_count": 3,
        "rejected_merge_count": 0,
        "active_voice_binding_count": 1,
        "reviewed_voice_binding_count": 1,
        "included_source_count": 26,
        "excluded_source_count": 0,
        "enrollment_unit_count": 5,
        "source_feasible_enrollment_unit_count": 5,
    }
    assert transition["rehearsal_allowed"] is True
    assert transition["a1_authorized"] is False
    assert transition["live_mutation_count"] == 0


def test_rejected_merge_that_exceeds_bound_fails_closed() -> None:
    reconciled, feasibility, manifest, submission = _manifest_and_submission()
    changed = copy.deepcopy(submission)
    first_merge = next(
        item
        for item in changed["decisions"]
        if item["decision_key"].startswith("MERGE::")
    )
    first_merge["decision"] = "reject"
    core = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = rehearsal.canonical_artifact_hash(core)

    transition = rehearsal.build_reviewed_transition(
        review_manifest=manifest,
        reconciliation=reconciled,
        feasibility=feasibility,
        submission=changed,
        reviewed_at="2026-08-09T12:00:00Z",
    )

    assert transition["status"] == "reviewed_transition_exceeds_enrollment_bound"
    assert transition["metrics"]["canonical_person_count"] == 7
    assert transition["metrics"]["enrollment_unit_count"] == 6
    assert transition["rehearsal_allowed"] is False


def test_private_knowledge_apply_rolls_back_exactly_and_replays(
    tmp_path: Path,
) -> None:
    transition = _transition()
    live_root = tmp_path / "live-store"
    runtime_root = tmp_path / "private-runtime"
    _initialize_legacy_store(live_root)
    live_database = transcript_store.db_path(live_root)
    live_before = rehearsal.sha256_file(live_database)

    receipt = rehearsal.rehearse_knowledge_copy(
        transition,
        live_store_root=live_root,
        runtime_root=runtime_root,
    )

    assert receipt["copy_apply_count"] == 1
    assert receipt["copy_rollback_count"] == 1
    assert receipt["biometric_rehearsal_complete"] is False
    assert receipt["live_mutation_count"] == 0
    assert receipt["idempotent_replay"] is False
    assert rehearsal.sha256_file(live_database) == live_before

    replay = rehearsal.rehearse_knowledge_copy(
        transition,
        live_store_root=live_root,
        runtime_root=runtime_root,
    )
    assert replay["idempotent_replay"] is True
    assert replay["content_sha256"] == receipt["content_sha256"]
    assert rehearsal.sha256_file(live_database) == live_before


def test_private_rehearsal_rejects_transition_over_bound(tmp_path: Path) -> None:
    transition = _transition()
    transition["status"] = "reviewed_transition_exceeds_enrollment_bound"
    transition["rehearsal_allowed"] = False
    core = {key: value for key, value in transition.items() if key != "content_sha256"}
    transition["content_sha256"] = rehearsal.canonical_artifact_hash(core)
    live_root = tmp_path / "live-store"
    _initialize_legacy_store(live_root)

    with pytest.raises(
        rehearsal.Plan0063PrivateRehearsalError,
        match="not eligible",
    ):
        rehearsal.rehearse_knowledge_copy(
            transition,
            live_store_root=live_root,
            runtime_root=tmp_path / "runtime",
        )
