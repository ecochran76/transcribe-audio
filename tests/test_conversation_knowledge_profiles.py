from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_profiles
import conversation_knowledge_store


CONVERSATION_ID = "00000000-0000-4000-8000-000000000301"
PERSON_ID = "00000000-0000-4000-8000-000000000302"
SAME_NAME_PERSON_ID = "00000000-0000-4000-8000-000000000303"
EVALUATION_ID = "00000000-0000-4000-8000-000000000304"


def _decision(
    suffix: int,
    *,
    proposal_id: str,
    action: str,
    supersedes: str = "",
    asserted: dict[str, str] | None = None,
) -> conversation_knowledge_store.ReviewDecisionRecord:
    return conversation_knowledge_store.ReviewDecisionRecord(
        decision_id=f"00000000-0000-4000-8000-{suffix:012d}",
        evaluation_id=EVALUATION_ID,
        proposal_id=proposal_id,
        action=action,
        reviewer="operator",
        method="manual",
        decided_at=f"2026-07-26T15:{suffix % 60:02d}:00Z",
        supersedes_decision_id=supersedes,
        reviewer_asserted_identity=asserted or {},
    )


def _projector(
    tmp_path: Path,
) -> conversation_knowledge_profiles.ConversationProfileProjector:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    store.save_conversation_snapshot(
        conversation_knowledge_store.ConversationSnapshot(
            conversation=conversation_knowledge_store.ConversationRecord(
                conversation_id=CONVERSATION_ID,
                title="Reviewed profile fixture",
                starts_at="2026-07-26T09:00:00-05:00",
            ),
        )
    )
    store.save_person_snapshot(
        conversation_knowledge_store.PersonSnapshot(
            person=conversation_knowledge_store.PersonRecord(
                person_id=PERSON_ID,
                status="reviewed",
                primary_name="Alex Example",
            ),
            source_records=(
                conversation_knowledge_store.SourceRecord(
                    source_record_id="gws-alex",
                    person_id=PERSON_ID,
                    source_profile_id="gws-personal",
                    provider_kind="gws",
                    account_id="personal@example.com",
                    tenant_id="",
                    external_ref="people/alex",
                    label="Alex Example",
                    relationship_scope="personal_interaction",
                    identifier_authority="email",
                    observed_at="2026-07-25T14:00:00Z",
                    content_hash="gws-alex-hash",
                ),
                conversation_knowledge_store.SourceRecord(
                    source_record_id="odoo-alex",
                    person_id=PERSON_ID,
                    source_profile_id="odoo-company",
                    provider_kind="odoo",
                    account_id="",
                    tenant_id="company-prod",
                    external_ref="res.partner:7",
                    label="Alexander Example",
                    relationship_scope="company_interaction",
                    identifier_authority="provider_id",
                    observed_at="2026-07-25T14:05:00Z",
                    content_hash="odoo-alex-hash",
                ),
            ),
        )
    )
    store.save_person_snapshot(
        conversation_knowledge_store.PersonSnapshot(
            person=conversation_knowledge_store.PersonRecord(
                person_id=SAME_NAME_PERSON_ID,
                status="ambiguous",
                primary_name="Alex Example",
            ),
            source_records=(
                conversation_knowledge_store.SourceRecord(
                    source_record_id="other-alex",
                    person_id=SAME_NAME_PERSON_ID,
                    source_profile_id="odoo-other",
                    provider_kind="odoo",
                    account_id="",
                    tenant_id="other-prod",
                    external_ref="res.partner:9",
                    label="Alex Example",
                    relationship_scope="other_company_interaction",
                    identifier_authority="provider_id",
                    observed_at="2026-07-25T14:10:00Z",
                    content_hash="other-alex-hash",
                ),
            ),
        )
    )
    evaluation = conversation_knowledge_store.EvaluationRecord(
        evaluation_id=EVALUATION_ID,
        conversation_id=CONVERSATION_ID,
        evaluation_type="speaker_identity",
        schema_version="speaker-identity.v1",
        status="complete",
        created_at="2026-07-26T15:00:00Z",
        payload={
            "evaluation_id": EVALUATION_ID,
            "proposals": [
                {
                    "proposal_id": "proposal-confirm",
                    "person_id": PERSON_ID,
                    "speaker_labels": ["A", "B"],
                    "review_flags": ["possible_diarization_split"],
                    "identity": {
                        "name": "Alex Example",
                        "organization": "Example Co",
                    },
                    "project": "Alpha Catalyst",
                    "topics": ["catalyst procurement"],
                    "terms": ["rare catalyst"],
                },
                {
                    "proposal_id": "proposal-mixed",
                    "person_id": PERSON_ID,
                    "speaker_labels": ["C"],
                    "review_flags": ["mixed_speaker_label"],
                    "utterance_assignments": [
                        {"utterance_id": "u1", "person_id": PERSON_ID},
                        {
                            "utterance_id": "u2",
                            "person_id": SAME_NAME_PERSON_ID,
                        },
                    ],
                },
                {
                    "proposal_id": "proposal-defer",
                    "person_id": "",
                    "speaker_labels": ["D"],
                },
            ],
        },
    )
    rejected = _decision(
        305,
        proposal_id="proposal-mixed",
        action="reject",
    )
    store.save_processing_history(
        conversation_knowledge_store.ProcessingHistory(
            conversation_id=CONVERSATION_ID,
            current_evaluation_id=EVALUATION_ID,
            evaluations=(evaluation,),
            review_decisions=(
                _decision(
                    306,
                    proposal_id="proposal-confirm",
                    action="confirm",
                    asserted={
                        "name": "Alex Example",
                        "organization": "Example Co",
                    },
                ),
                rejected,
                _decision(
                    307,
                    proposal_id="proposal-defer",
                    action="defer",
                ),
                _decision(
                    308,
                    proposal_id="proposal-mixed",
                    action="confirm",
                    supersedes=rejected.decision_id,
                ),
            ),
        )
    )
    return conversation_knowledge_profiles.ConversationProfileProjector(
        tmp_path
    )


def _observation_digest(tmp_path: Path) -> tuple[int, str]:
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        rows = con.execute(
            """
            SELECT id, observation_type, subject_type, subject_id, source_type,
                   source_id, COALESCE(conversation_id, ''), observed_at,
                   review_state, payload_json, content_hash
            FROM knowledge_observations
            ORDER BY id
            """
        ).fetchall()
    return len(rows), json.dumps(rows, separators=(",", ":"))


def test_reviewed_outcomes_become_immutable_typed_observations(
    tmp_path: Path,
) -> None:
    projector = _projector(tmp_path)

    first = projector.append_reviewed_observations(CONVERSATION_ID)
    before = _observation_digest(tmp_path)
    second = projector.append_reviewed_observations(CONVERSATION_ID)
    after = _observation_digest(tmp_path)
    types = {
        item.observation_type
        for item in conversation_knowledge_store.ConversationKnowledgeStore(
            tmp_path
        ).load_observations(CONVERSATION_ID)
    }

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert before == after
    assert {
        "speaker_identity_confirmed",
        "speaker_identity_rejected",
        "speaker_identity_deferred",
        "review_decision_superseded",
        "diarization_split",
        "mixed_speaker",
        "reviewer_asserted_identity",
    } <= types


def test_projection_rebuild_is_deterministic_and_preserves_ambiguity(
    tmp_path: Path,
) -> None:
    projector = _projector(tmp_path)
    projector.append_reviewed_observations(CONVERSATION_ID)
    observations_before = _observation_digest(tmp_path)

    first = projector.rebuild()
    people_before = projector.load_person_profiles()
    affinities_before = projector.load_affinity_profiles()
    second = projector.rebuild()
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        con.execute("DELETE FROM knowledge_affinity_profiles")
        con.execute("DELETE FROM knowledge_current_person_profiles")
        con.execute(
            """
            DELETE FROM knowledge_projection_state
            WHERE projection_name = 'reviewed-affinity-profiles'
            """
        )
        con.commit()
    rebuilt = projector.rebuild()

    assert first.status == "inserted"
    assert second.status == "unchanged"
    assert rebuilt.status == "inserted"
    assert projector.load_person_profiles() == people_before
    assert projector.load_affinity_profiles() == affinities_before
    assert _observation_digest(tmp_path) == observations_before
    assert {
        profile.person_id for profile in people_before
    } == {PERSON_ID, SAME_NAME_PERSON_ID}
    assert [profile.primary_name for profile in people_before].count(
        "Alex Example"
    ) == 2
    assert all(profile.observation_ids for profile in people_before)
    assert all(profile.input_watermark for profile in people_before)
    assert {
        profile.affinity_type for profile in affinities_before
    } >= {
        "interaction",
        "organization",
        "project",
        "source_relationship",
        "terminology",
        "topic",
    }
    assert all(profile.observation_ids for profile in affinities_before)
    assert all(profile.input_watermark for profile in affinities_before)
    source_profiles = {
        profile.object_id
        for profile in affinities_before
        if profile.affinity_type == "source_relationship"
    }
    assert source_profiles == {
        "gws-personal",
        "odoo-company",
        "odoo-other",
    }


def test_v3_rollback_preserves_evidence_schema(tmp_path: Path) -> None:
    projector = _projector(tmp_path)
    projector.append_reviewed_observations(CONVERSATION_ID)
    projector.rebuild()
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)

    receipt = store.rollback(target_version=2, backup=False)

    assert receipt.rolled_back_versions == (3,)
    assert store.schema_status().schema_version == 2
    assert store.load_observations(CONVERSATION_ID)
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        evidence_table = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE name = 'knowledge_evidence_snapshots'
            """
        ).fetchone()
        profile_table = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE name = 'knowledge_current_person_profiles'
            """
        ).fetchone()
    assert evidence_table is not None
    assert profile_table is None
