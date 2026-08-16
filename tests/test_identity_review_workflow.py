from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import transcript_store
from conversation_knowledge_store import ConversationKnowledgeStore
from identity_review_workflow import IdentityReviewWorkflow, StaleReviewSubmission


SHA_A = "a" * 64
SHA_B = "b" * 64


def queue_item(*, item_id: str = "queue-1", projection_version: str = "1") -> dict:
    return {
        "schema_version": "transcribe-audio.identity-review-queue-item.v1",
        "queue_item_id": item_id,
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "original_recording_filename": "Monday planning.m4a",
        "source_artifact_sha256": SHA_A,
        "source_media_sha256": SHA_B,
        "processing_run_id": "run-1",
        "model_versions": ["identity-model-v1"],
        "rubric_versions": ["identity-rubric-v1"],
        "profile_versions": ["profile-v1"],
        "calendar_candidates": [
            {
                "candidate_id": "event-1",
                "label": "Monday planning",
                "association_strength": 0.83,
                "attendees": ["Alex Example", "Morgan Example"],
            }
        ],
        "participant_hypotheses": [
            {"hypothesis_id": "participant-1", "label": "Alex Example", "kind": "participant"}
        ],
        "speakers": [
            {
                "speaker_ref": "SPEAKER_01",
                "proposal_id": "proposal-1",
                "best_guess": {"person_id": "person-1", "label": "Alex Example", "strength": 0.76},
                "alternatives": [{"person_id": "person-2", "label": "Morgan Example", "strength": 0.42}],
                "evidence": [{"pillar": "calendar", "direction": "supporting", "summary": "Attendee snapshot"}],
                "audio": {"media_url": "/api/blobs/blob-1", "start_ms": 1250, "end_ms": 6900},
            }
        ],
        "review_state": "unreviewed",
        "decision_history": [],
        "effect_preview_ref": "",
        "projection_version": projection_version,
        "created_at": "2026-08-16T18:00:00Z",
    }


def submission(*, item_id: str = "queue-1", version: str = "1", key: str = "decision-1") -> dict:
    return {
        "schema_version": "transcribe-audio.identity-review-submission.v1",
        "submission_id": "submission-1",
        "queue_item_id": item_id,
        "conversation_id": "conversation-1",
        "proposal_id": "proposal-1",
        "action": "choose_existing_person",
        "expected_projection_version": version,
        "decision_payload": {"person_id": "person-1", "speaker_ref": "SPEAKER_01"},
        "comment": "Confirmed from the reviewed evidence.",
        "idempotency_key": key,
        "reviewer": "operator",
        "decided_at": "2026-08-16T18:05:00Z",
    }


@pytest.fixture
def workflow(tmp_path: Path) -> IdentityReviewWorkflow:
    store = ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    return IdentityReviewWorkflow(tmp_path)


def test_v8_migration_is_additive_and_rolls_back_to_v7(tmp_path: Path) -> None:
    store = ConversationKnowledgeStore(tmp_path)
    receipt = store.migrate(backup=False)

    assert receipt.to_version == 8
    assert receipt.applied_versions == tuple(range(1, 9))
    with transcript_store.connect(tmp_path) as con:
        names = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'knowledge_identity_review_%'"
            )
        }
    assert {
        "knowledge_identity_review_queue",
        "knowledge_identity_review_submissions",
        "knowledge_identity_review_effect_previews",
    }.issubset(names)

    rolled_back = store.rollback(target_version=7, backup=False)
    assert rolled_back.rolled_back_versions == (8,)
    assert store.schema_status().schema_version == 7


def test_queue_projection_preserves_original_filename_and_filters(workflow: IdentityReviewWorkflow) -> None:
    workflow.project_queue_item(queue_item(), priority=90, impact_score=0.8)
    workflow.project_queue_item(
        {
            **queue_item(item_id="queue-2"),
            "conversation_id": "conversation-2",
            "recording_id": "recording-2",
            "original_recording_filename": "Customer interview.wav",
            "review_state": "unresolved",
        },
        priority=40,
        impact_score=0.3,
    )

    page = workflow.list_queue(limit=1, offset=0, state="unreviewed", query="Monday")

    assert page["total"] == 1
    assert page["limit"] == 1
    assert page["offset"] == 0
    assert page["items"][0]["original_recording_filename"] == "Monday planning.m4a"
    assert page["items"][0]["source_artifact_sha256"] == SHA_A
    assert page["items"][0]["speakers"][0]["audio"]["start_ms"] == 1250
    assert "stored_path" not in str(page)


def test_preview_is_zero_effect_and_submit_is_idempotent_and_stale_safe(
    workflow: IdentityReviewWorkflow,
) -> None:
    workflow.project_queue_item(queue_item(), priority=90, impact_score=0.8)

    preview = workflow.preview_submission(submission())
    assert preview["effect_mode"] == "preview_only"
    assert preview["provider_write_count"] == 0
    assert preview["raw_deletion_count"] == 0
    assert preview["proposed_effects"][0]["effect_type"] == "speaker_identity_decision"

    accepted = workflow.record_submission(submission())
    replay = workflow.record_submission(submission())
    assert {key: value for key, value in accepted.items() if key != "idempotent_replay"} == {
        key: value for key, value in replay.items() if key != "idempotent_replay"
    }
    assert accepted["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert accepted["projection_version"] == "2"
    assert accepted["effect_preview"]["provider_write_count"] == 0
    assert workflow.get_queue_item("queue-1")["decision_history"][0]["submission_id"] == "submission-1"

    with pytest.raises(StaleReviewSubmission, match="expected projection version 1"):
        workflow.record_submission(submission(key="decision-2"))


def test_people_projection_aggregates_sources_roles_and_relationships(
    workflow: IdentityReviewWorkflow,
) -> None:
    with transcript_store.connect(workflow.root) as con:
        con.execute(
            """
            INSERT INTO knowledge_identity_people_projection (
              person_id, status, primary_name, aliases_json, merged_into_person_id,
              input_watermark, metadata_json, built_at
            ) VALUES ('person-1', 'reviewed', 'Alex Example', '["Alex E."]', '', 'watermark-1', '{}', '2026-08-16T18:00:00Z')
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_identity_source_projection (
              source_record_id, person_id, source_profile_id, provider_kind, account_id,
              tenant_id, record_type, external_ref, label, source_event_at, observed_at,
              content_hash, resolution_status, input_watermark, metadata_json, built_at
            ) VALUES (
              'source-1', 'person-1', 'fixture-profile', 'fixture', 'fixture-account',
              'fixture-tenant', 'contact', 'contact-1', 'Alex Example', '',
              '2026-08-16T18:00:00Z', ?, 'reviewed', 'watermark-1', '{}', '2026-08-16T18:00:00Z'
            )
            """,
            (SHA_A,),
        )
        con.execute(
            """
            INSERT INTO knowledge_identity_role_projection (
              role_id, person_id, role_type, organization_id, project_id, matter_id,
              conversation_id, starts_at, ends_at, status, evidence_ids_json,
              input_watermark, metadata_json, built_at
            ) VALUES (
              'role-1', 'person-1', 'project_lead', 'org-1', '', '', '', '', '',
              'reviewed', '["evidence-1"]', 'watermark-1', '{}', '2026-08-16T18:00:00Z'
            )
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_identity_relationship_projection (
              relationship_id, relationship_type, subject_type, subject_id, object_type,
              object_id, directionality, inverse_relationship_id, starts_at, ends_at,
              status, evidence_ids_json, input_watermark, metadata_json, built_at
            ) VALUES (
              'relationship-1', 'works_with', 'person', 'person-1', 'person', 'person-2',
              'symmetric', '', '', '', 'reviewed', '["evidence-2"]', 'watermark-1', '{}',
              '2026-08-16T18:00:00Z'
            )
            """
        )
        con.commit()

    payload = workflow.list_people(query="alex", status="reviewed", limit=10, offset=0)

    assert payload["total"] == 1
    person = payload["items"][0]
    assert person["primary_name"] == "Alex Example"
    assert person["aliases"] == ["Alex E."]
    assert person["source_records"][0]["provider_kind"] == "fixture"
    assert person["roles"][0]["role_type"] == "project_lead"
    assert person["relationships"][0]["relationship_type"] == "works_with"
