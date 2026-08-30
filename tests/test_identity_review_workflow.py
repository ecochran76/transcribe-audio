from __future__ import annotations

import json
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
    return IdentityReviewWorkflow(
        tmp_path, gold_root=tmp_path / "speaker-evaluation-campaigns"
    )


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


def test_queue_read_enriches_recording_and_diarization_metadata(
    workflow: IdentityReviewWorkflow,
) -> None:
    workflow.project_queue_item(queue_item(), priority=90, impact_score=0.8)
    payload = {
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "event": {"summary": "1042: Monday planning"},
        "utterances": [
            {"speaker": "SPEAKER_01", "start": 1250, "end": 6900, "text": "First sample."},
            {"speaker": "SPEAKER_01", "start": 8000, "end": 11000, "text": "Second sample."},
        ],
    }
    with transcript_store.connect(workflow.root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO documents (
              id, kind, title, source_path, stored_path, artifact_sha256,
              generated_at, text_content, json_payload, metadata_json,
              embedding_json, embedding_provider, embedding_model, created_at, updated_at
            ) VALUES (?, 'transcript', 'AssemblyAI Transcript', '/source.json', '/stored.json', ?,
              '2026-08-15T14:30:00Z', 'text', ?, ?, '[]', 'hash', 'hash-v1',
              '2026-08-15T14:30:00Z', '2026-08-15T14:30:00Z')
            """,
            (
                "document-1",
                SHA_A,
                json.dumps(payload),
                json.dumps({"media_blob": {"playback_url": "/api/blobs/blob-1"}}),
            ),
        )
        con.commit()

    display = workflow.list_queue()["items"][0]["display"]

    assert display["title"] == "Monday planning"
    assert display["recorded_at"] == "2026-08-15T14:30:00Z"
    assert display["duration_ms"] == 11000
    assert display["utterance_count"] == 2
    assert display["media_url"] == "/api/blobs/blob-1"
    assert display["diarization"][0] == {
        "speaker_ref": "SPEAKER_01",
        "utterance_count": 2,
        "talk_time_ms": 8650,
        "sample_segments": [
            {"start_ms": 1250, "end_ms": 6900, "text": "First sample."},
            {"start_ms": 8000, "end_ms": 11000, "text": "Second sample."},
        ],
    }


def test_queue_read_reconciles_exact_operator_gold_without_mutating_proposal(
    workflow: IdentityReviewWorkflow,
) -> None:
    workflow.project_queue_item(queue_item(), priority=90, impact_score=0.8)
    payload = {
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "utterances": [
            {
                "speaker": "SPEAKER_01",
                "start": 1250,
                "end": 6900,
                "text": "Reviewed sample.",
            }
        ],
    }
    with transcript_store.connect(workflow.root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO documents (
              id, kind, title, source_path, stored_path, artifact_sha256,
              generated_at, text_content, json_payload, metadata_json,
              embedding_json, embedding_provider, embedding_model, created_at, updated_at
            ) VALUES (?, 'transcript', 'AssemblyAI Transcript', '/source.json', '/stored.json', ?,
              '2026-08-15T14:30:00Z', 'text', ?, '{}', '[]', 'hash', 'hash-v1',
              '2026-08-15T14:30:00Z', '2026-08-15T14:30:00Z')
            """,
            ("document-1", SHA_A, json.dumps(payload)),
        )
        con.commit()

    campaign_id = "campaign-0123456789abcdefabcd"
    campaign_dir = workflow.gold_root / campaign_id
    gold_path = campaign_dir / "gold" / "document-1" / "gold-1.json"
    gold_path.parent.mkdir(parents=True)
    gold_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.speaker-evaluation-gold.v1",
                "campaign_id": campaign_id,
                "gold_id": "gold-1",
                "document_id": "document-1",
                "disposition": "eligible_known",
                "reviewed_at": "2026-08-16T18:05:00Z",
                "reviewer": "Eric Cochran",
                "review_method": "transcript_and_calendar",
                "prediction_visibility": "excluded",
                "people": [
                    {
                        "person_ground_truth_id": "person-alex-example",
                        "name": "Alex Example",
                        "email": "alex@example.test",
                    }
                ],
                "speaker_outcomes": [
                    {
                        "speaker_label": "SPEAKER_01",
                        "outcome": "person",
                        "person_ground_truth_id": "person-alex-example",
                    }
                ],
                "same_person_label_groups": [],
            }
        ),
        encoding="utf-8",
    )
    index_path = campaign_dir / "gold" / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "gold_id": "gold-1",
                        "document_id": "document-1",
                        "reviewed_at": "2026-08-16T18:05:00Z",
                        "path": str(gold_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    item = workflow.list_queue()["items"][0]
    review = item["display"]["operator_review"]

    assert review["status"] == "reviewed"
    assert review["matched_speaker_count"] == 1
    assert review["speaker_outcomes"] == [
        {
            "speaker_ref": "SPEAKER_01",
            "outcome": "person",
            "label": "Alex Example",
            "person_ground_truth_id": "person-alex-example",
            "mixed_components": [],
        }
    ]
    assert item["speakers"][0]["best_guess"]["label"] == "Alex Example"

    people = workflow.list_people(kind="reviewed_speaker")
    assert people["total"] == 1
    reviewed_person = people["items"][0]
    assert reviewed_person["identity_kind"] == "reviewed_speaker"
    assert reviewed_person["source_identity_id"] == "person-alex-example"
    assert reviewed_person["speaker_review_count"] == 1
    assert reviewed_person["recording_count"] == 1
    assert reviewed_person["review_occurrences"][0]["recording_filename"] == "Monday planning.m4a"


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


def test_people_projection_exposes_shadow_graph_hypotheses_without_accepting_them(
    workflow: IdentityReviewWorkflow,
) -> None:
    metadata = {
        "contact_class": "person_candidate",
        "calendar_attendee": {"appearances": []},
        "enrichment": {
            "source_records": [
                {
                    "provider": "gws",
                    "profile": "fixture",
                    "record_type": "gws_contact",
                    "source_record_id": "people/alex",
                    "label": "Alex Example",
                    "organizations": ["Example Labs"],
                    "roles": [
                        {
                            "title": "Research Director",
                            "organization": "Example Labs",
                            "department": "Research",
                            "current": True,
                        }
                    ],
                    "match_basis": "exact_email",
                }
            ]
        },
    }
    with transcript_store.connect(workflow.root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (
              id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (
              'contact-alex', 'Alex Example', 'alex@example.test', '', ?,
              '2026-08-29T00:00:00Z', '2026-08-29T00:00:00Z'
            )
            """,
            (json.dumps(metadata),),
        )
        con.commit()

    payload = workflow.list_people(kind="local_contact", limit=10)

    assert payload["graph_discovery"]["authority_mode"] == "shadow_hypotheses_only"
    assert payload["graph_discovery"]["role_hypothesis_count"] == 1
    assert payload["graph_discovery"]["affiliation_hypothesis_count"] == 1
    assert payload["graph_discovery"]["accepted_effect_count"] == 0
    contact = payload["items"][0]
    assert contact["role_hypotheses"][0]["display_value"] == "Research Director"
    assert contact["relationship_hypotheses"][0]["relationship_type"] == "AFFILIATED_WITH"
    assert contact["roles"] == []
    assert contact["relationships"] == []


def test_people_projection_bridges_current_profiles_and_local_contacts_without_linking(
    workflow: IdentityReviewWorkflow,
) -> None:
    with transcript_store.connect(workflow.root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO knowledge_people (
              id, status, primary_name, metadata_json, created_at, updated_at
            ) VALUES (
              'canonical-person-1', 'reviewed', 'Alex Example', '{}',
              '2026-08-16T18:00:00Z', '2026-08-16T18:00:00Z'
            )
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_source_records (
              id, person_id, source_profile_id, provider_kind, account_id,
              tenant_id, external_ref, label, relationship_scope,
              identifier_authority, observed_at, content_hash, metadata_json,
              created_at, updated_at
            ) VALUES (
              'source-legacy-1', 'canonical-person-1', 'profile-1', 'human_review',
              '', '', 'review-1', 'Alex Example', 'person', 'reviewed',
              '2026-08-16T18:00:00Z', ?, '{}',
              '2026-08-16T18:00:00Z', '2026-08-16T18:00:00Z'
            )
            """,
            (SHA_A,),
        )
        con.execute(
            """
            INSERT INTO knowledge_current_person_profiles (
              person_id, resolution_status, primary_name, aliases_json,
              source_record_ids_json, observation_ids_json, input_watermark,
              metadata_json, built_at
            ) VALUES (
              'canonical-person-1', 'reviewed', 'Alex Example', '["Alex E."]',
              '["source-legacy-1"]', '[]', 'profile-watermark', '{}',
              '2026-08-16T18:00:00Z'
            )
            """
        )
        con.execute(
            """
            INSERT INTO contacts (
              id, label, email, external_ref, metadata_json, created_at, updated_at
            ) VALUES (
              'contact-1', 'Alex Example', 'alex@example.test', '',
              '{"source":"context_workbench_operator_input"}',
              '2026-08-16T18:00:00Z', '2026-08-16T18:00:00Z'
            )
            """
        )
        con.commit()

    payload = workflow.list_people(query="alex", limit=10)

    assert payload["total"] == 2
    assert payload["counts"] == {
        "canonical_person": 1,
        "local_contact": 1,
        "reviewed_speaker": 0,
    }
    assert {item["identity_kind"] for item in payload["items"]} == {
        "canonical_person",
        "local_contact",
    }
    assert all(item["possible_related_records"] for item in payload["items"])
    canonical = workflow.list_people(kind="canonical_person")["items"][0]
    assert canonical["aliases"] == ["Alex E."]
    assert canonical["source_records"][0]["provider_kind"] == "human_review"
    contact = workflow.list_people(kind="local_contact")["items"][0]
    assert contact["source_records"][0]["resolution_status"] == "unlinked"
    assert contact["contact_methods"] == [
        {"kind": "email", "value": "alex@example.test"}
    ]


def test_people_projection_rejects_unknown_record_kind(
    workflow: IdentityReviewWorkflow,
) -> None:
    with pytest.raises(ValueError, match="Unsupported Contacts record type"):
        workflow.list_people(kind="guessed_person")


def test_people_projection_exposes_calendar_contact_evidence(
    workflow: IdentityReviewWorkflow,
) -> None:
    metadata = {
        "source": "calendar_attendee",
        "resolution_status": "review_required",
        "contact_class": "shared_or_role_address",
        "identity_boundary": "exact_email_source_join_not_person_or_speaker_proof",
        "calendar_attendee": {
            "aliases": ["Support"],
            "occurrence_count": 2,
            "recording_count": 1,
            "appearances": [
                {
                    "document_id": "doc-1",
                    "recording_title": "Support call",
                    "recording_filename": "support-call.m4a",
                    "recorded_at": "2026-08-20T14:30:00Z",
                }
            ],
        },
        "enrichment": {
            "phones": ["+1 555 0100"],
            "organizations": ["Example Labs"],
            "source_records": [
                {
                    "provider": "gws",
                    "profile": "default",
                    "record_type": "gws_contact",
                    "source_record_id": "people/support",
                    "label": "Support",
                }
            ],
        },
    }
    with transcript_store.connect(workflow.root) as con:
        transcript_store.init_db(con)
        con.execute(
            """
            INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
            VALUES ('contact-calendar', 'Support', 'support@example.test', '', ?,
                    '2026-08-20T14:30:00Z', '2026-08-20T14:30:00Z')
            """,
            (json.dumps(metadata),),
        )
        con.commit()

    contact = workflow.list_people(kind="local_contact", limit=500)["items"][0]
    assert contact["status"] == "review_required"
    assert contact["contact_class"] == "shared_or_role_address"
    assert contact["recording_count"] == 1
    assert contact["attendee_occurrence_count"] == 2
    assert contact["calendar_occurrences"][0]["recording_filename"] == "support-call.m4a"
    assert {method["kind"] for method in contact["contact_methods"]} == {"email", "phone"}
    assert contact["organizations"] == ["Example Labs"]
    assert contact["source_records"][1]["resolution_status"] == "exact_email_observation"
