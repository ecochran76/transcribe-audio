from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_evidence
import conversation_knowledge_store


CONVERSATION_ID = "00000000-0000-4000-8000-000000000201"
PERSON_ID = "00000000-0000-4000-8000-000000000202"
GWS_EVIDENCE_ID = "00000000-0000-4000-8000-000000000203"
ODOO_EVIDENCE_ID = "00000000-0000-4000-8000-000000000204"
OTHER_EVIDENCE_ID = "00000000-0000-4000-8000-000000000205"
REQUEST_ID = "00000000-0000-4000-8000-000000000206"
BUNDLE_ID = "00000000-0000-4000-8000-000000000207"
CONCEPT_ID = "00000000-0000-4000-8000-000000000208"
MENTION_ID = "00000000-0000-4000-8000-000000000209"


def test_unlinked_provider_record_is_preserved_as_metadata_without_fk_failure(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    snapshot = replace(
        _snapshots()[0],
        evidence_id="00000000-0000-4000-8000-000000000211",
        source_record_id="people/unresolved-provider-record",
        structured_metadata={
            "provider_record_id": "people/unresolved-provider-record",
        },
        content_hash="unlinked-provider-evidence-hash",
    )

    assert repository.save_snapshot(snapshot) == "inserted"
    assert repository.save_snapshot(snapshot) == "unchanged"
    stored = repository.load_snapshot(snapshot.evidence_id)

    assert stored is not None
    assert stored.source_record_id == ""
    assert stored.structured_metadata["provider_record_id"] == (
        "people/unresolved-provider-record"
    )


def _repository(
    tmp_path: Path,
) -> conversation_knowledge_evidence.ConversationEvidenceRepository:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    store.save_conversation_snapshot(
        conversation_knowledge_store.ConversationSnapshot(
            conversation=conversation_knowledge_store.ConversationRecord(
                conversation_id=CONVERSATION_ID,
                title="Evidence fixture",
                starts_at="2026-07-26T09:00:00-05:00",
            ),
        )
    )
    store.save_person_snapshot(
        conversation_knowledge_store.PersonSnapshot(
            person=conversation_knowledge_store.PersonRecord(
                person_id=PERSON_ID,
                status="reviewed",
                primary_name="Example Person",
            ),
            source_records=(
                conversation_knowledge_store.SourceRecord(
                    source_record_id="gws-contact-1",
                    person_id=PERSON_ID,
                    source_profile_id="gws-personal",
                    provider_kind="gws",
                    account_id="personal@example.com",
                    tenant_id="",
                    external_ref="people/1",
                    label="Example Person",
                    relationship_scope="personal_interaction",
                    identifier_authority="email",
                    observed_at="2026-07-26T14:00:00Z",
                    content_hash="gws-contact-hash",
                ),
                conversation_knowledge_store.SourceRecord(
                    source_record_id="odoo-contact-1",
                    person_id=PERSON_ID,
                    source_profile_id="odoo-company",
                    provider_kind="odoo",
                    account_id="",
                    tenant_id="company-prod",
                    external_ref="res.partner:1",
                    label="Example Person",
                    relationship_scope="company_interaction",
                    identifier_authority="provider_id",
                    observed_at="2026-07-26T14:00:00Z",
                    content_hash="odoo-contact-hash",
                ),
            ),
            external_identities=(
                conversation_knowledge_store.ExternalIdentityRecord(
                    external_identity_id=(
                        "00000000-0000-4000-8000-000000000210"
                    ),
                    person_id=PERSON_ID,
                    source_record_id="gws-contact-1",
                    identity_kind="email",
                    normalized_value="person@example.com",
                    display_value="person@example.com",
                    authority="gws",
                    verified=True,
                ),
                conversation_knowledge_store.ExternalIdentityRecord(
                    external_identity_id=(
                        "00000000-0000-4000-8000-000000000211"
                    ),
                    person_id=PERSON_ID,
                    source_record_id="odoo-contact-1",
                    identity_kind="email",
                    normalized_value="person@example.com",
                    display_value="person@example.com",
                    authority="odoo",
                    verified=True,
                ),
            ),
        )
    )
    return conversation_knowledge_evidence.ConversationEvidenceRepository(
        tmp_path
    )


def _snapshots() -> tuple[
    conversation_knowledge_evidence.EvidenceSnapshotRecord,
    conversation_knowledge_evidence.EvidenceSnapshotRecord,
    conversation_knowledge_evidence.EvidenceSnapshotRecord,
]:
    return (
        conversation_knowledge_evidence.EvidenceSnapshotRecord(
            evidence_id=GWS_EVIDENCE_ID,
            source_record_id="gws-contact-1",
            source_profile_id="gws-personal",
            provider_kind="gws",
            account_id="personal@example.com",
            tenant_id="",
            source_type="gmail_message",
            capability="mail",
            snippet="Alpha catalyst procurement was discussed by email.",
            structured_metadata={"thread_id": "thread-1"},
            source_event_at="2026-07-25T14:00:00Z",
            observed_at="2026-07-25T14:05:00Z",
            retrieved_at="2026-07-25T14:06:00Z",
            temporal_class="contemporaneous",
            source_uri="gmail:message-1",
            content_hash="gws-evidence-hash",
            independence_group_id="interaction-email-1",
            freshness_state="current",
            embedding=(1.0, 0.0),
            embedding_provider="fixture",
            embedding_model="fixture-v1",
        ),
        conversation_knowledge_evidence.EvidenceSnapshotRecord(
            evidence_id=ODOO_EVIDENCE_ID,
            source_record_id="odoo-contact-1",
            source_profile_id="odoo-company",
            provider_kind="odoo",
            account_id="",
            tenant_id="company-prod",
            source_type="crm_log_note",
            capability="log_notes",
            snippet="The company log note references the Alpha catalyst project.",
            structured_metadata={"model": "res.partner"},
            source_event_at="2026-07-24T14:00:00Z",
            observed_at="2026-07-27T14:05:00Z",
            retrieved_at="2026-07-27T14:06:00Z",
            temporal_class="later_retrieved",
            source_uri="odoo:res.partner:1:note:4",
            content_hash="odoo-evidence-hash",
            independence_group_id="interaction-log-note-1",
            freshness_state="current",
            embedding=(0.9, 0.1),
            embedding_provider="fixture",
            embedding_model="fixture-v1",
        ),
        conversation_knowledge_evidence.EvidenceSnapshotRecord(
            evidence_id=OTHER_EVIDENCE_ID,
            source_record_id="",
            source_profile_id="odoo-other",
            provider_kind="odoo",
            account_id="",
            tenant_id="other-prod",
            source_type="crm_log_note",
            capability="log_notes",
            snippet="Alpha must not cross the tenant boundary.",
            structured_metadata={},
            source_event_at="2026-07-24T14:00:00Z",
            observed_at="2026-07-24T14:05:00Z",
            retrieved_at="2026-07-24T14:06:00Z",
            temporal_class="contemporaneous",
            source_uri="odoo:other:note:1",
            content_hash="other-evidence-hash",
            independence_group_id="interaction-other-1",
            freshness_state="current",
            embedding=(0.0, 1.0),
            embedding_provider="fixture",
            embedding_model="fixture-v1",
        ),
    )


def test_evidence_query_enforces_scope_capability_and_temporal_policy(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    for snapshot in _snapshots():
        assert repository.save_snapshot(snapshot) == "inserted"
        assert repository.save_snapshot(snapshot) == "unchanged"

    gws_scope = conversation_knowledge_evidence.EvidenceScope(
        source_profile_id="gws-personal",
        account_id="personal@example.com",
        tenant_id="",
    )
    odoo_scope = conversation_knowledge_evidence.EvidenceScope(
        source_profile_id="odoo-company",
        account_id="",
        tenant_id="company-prod",
    )

    gws_results = repository.search_snapshots(
        "alpha catalyst",
        scopes=(gws_scope,),
        capabilities=("mail",),
        as_of="2026-07-26T14:00:00Z",
        hindsight_policy="exclude",
    )
    combined_without_hindsight = repository.search_snapshots(
        "alpha catalyst",
        scopes=(gws_scope, odoo_scope),
        capabilities=("mail", "log_notes"),
        as_of="2026-07-26T14:00:00Z",
        hindsight_policy="exclude",
    )
    combined_with_later = repository.search_snapshots(
        "alpha catalyst",
        scopes=(gws_scope, odoo_scope),
        capabilities=("mail", "log_notes"),
        as_of="2026-07-26T14:00:00Z",
        hindsight_policy="allow_later_retrieved",
    )

    assert [item.evidence_id for item in gws_results] == [GWS_EVIDENCE_ID]
    assert [item.evidence_id for item in combined_without_hindsight] == [
        GWS_EVIDENCE_ID
    ]
    assert {item.evidence_id for item in combined_with_later} == {
        GWS_EVIDENCE_ID,
        ODOO_EVIDENCE_ID,
    }
    assert OTHER_EVIDENCE_ID not in {
        item.evidence_id for item in combined_with_later
    }
    assert repository.semantic_snapshots(
        (1.0, 0.0),
        scopes=(gws_scope, odoo_scope),
        capabilities=("mail", "log_notes"),
        as_of="2026-07-26T14:00:00Z",
        hindsight_policy="allow_later_retrieved",
    )[0].evidence_id == GWS_EVIDENCE_ID


def test_evidence_query_treats_hyphenated_prefix_terms_as_literals(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.save_snapshot(_snapshots()[0])
    gws_scope = conversation_knowledge_evidence.EvidenceScope(
        source_profile_id="gws-personal",
        account_id="personal@example.com",
        tenant_id="",
    )

    results = repository.search_snapshots(
        "alpha board-member",
        scopes=(gws_scope,),
        capabilities=("mail",),
        as_of="2026-07-26T14:00:00Z",
        hindsight_policy="exclude",
    )

    assert [item.evidence_id for item in results] == [GWS_EVIDENCE_ID]


def test_snapshot_contract_rejects_unbounded_provider_content(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    snapshot = _snapshots()[0]

    with pytest.raises(ValueError, match="bounded snippet"):
        repository.save_snapshot(
            conversation_knowledge_evidence.EvidenceSnapshotRecord(
                **{
                    **snapshot.__dict__,
                    "evidence_id": (
                        "00000000-0000-4000-8000-000000000212"
                    ),
                    "snippet": "x"
                    * (
                        conversation_knowledge_evidence.MAX_EVIDENCE_SNIPPET_CHARS
                        + 1
                    ),
                }
            )
        )
    with pytest.raises(ValueError, match="structured metadata"):
        repository.save_snapshot(
            conversation_knowledge_evidence.EvidenceSnapshotRecord(
                **{
                    **snapshot.__dict__,
                    "evidence_id": (
                        "00000000-0000-4000-8000-000000000213"
                    ),
                    "structured_metadata": {
                        "body": "x"
                        * (
                            conversation_knowledge_evidence.MAX_EVIDENCE_METADATA_CHARS
                            + 1
                        )
                    },
                }
            )
        )
    with pytest.raises(ValueError, match="raw provider body"):
        repository.save_snapshot(
            conversation_knowledge_evidence.EvidenceSnapshotRecord(
                **{
                    **snapshot.__dict__,
                    "evidence_id": (
                        "00000000-0000-4000-8000-000000000214"
                    ),
                    "structured_metadata": {"raw_body": "short but forbidden"},
                }
            )
        )


def test_exact_identity_lookup_preserves_source_scope(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    gws_matches = repository.find_people_by_external_identity(
        "email",
        "PERSON@example.com",
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-personal",
                account_id="personal@example.com",
                tenant_id="",
            ),
        ),
    )
    odoo_matches = repository.find_people_by_external_identity(
        "email",
        "person@example.com",
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="odoo-company",
                account_id="",
                tenant_id="company-prod",
            ),
        ),
    )

    assert gws_matches[0].person_id == PERSON_ID
    assert gws_matches[0].source_record_id == "gws-contact-1"
    assert odoo_matches[0].person_id == PERSON_ID
    assert odoo_matches[0].source_record_id == "odoo-contact-1"


def test_concepts_requests_and_bundles_are_replayable_and_immutable(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    for snapshot in _snapshots()[:2]:
        repository.save_snapshot(snapshot)
    concept = conversation_knowledge_evidence.ConceptRecord(
        concept_id=CONCEPT_ID,
        concept_type="project",
        normalized_value="alpha catalyst",
        display_value="Alpha Catalyst",
    )
    mention = conversation_knowledge_evidence.ConceptMentionRecord(
        mention_id=MENTION_ID,
        concept_id=CONCEPT_ID,
        conversation_id=CONVERSATION_ID,
        utterance_id="",
        evidence_snapshot_id=GWS_EVIDENCE_ID,
        person_id=PERSON_ID,
        observed_at="2026-07-25T14:05:00Z",
    )
    repository.save_concept(concept, mentions=(mention,))
    with pytest.raises(ValueError, match="mention is immutable"):
        repository.save_concept(
            concept,
            mentions=(
                conversation_knowledge_evidence.ConceptMentionRecord(
                    **{
                        **mention.__dict__,
                        "metadata": {"changed": True},
                    }
                ),
            ),
        )
    request = conversation_knowledge_evidence.RetrievalRequestRecord(
        request_id=REQUEST_ID,
        conversation_id=CONVERSATION_ID,
        recording_ids=(),
        speaker_labels=("A",),
        clue_ids=("clue-1",),
        conversation_at="2026-07-26T14:00:00Z",
        as_of="2026-07-26T14:00:00Z",
        prepared_person_ids=(PERSON_ID,),
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-personal",
                account_id="personal@example.com",
                tenant_id="",
            ),
        ),
        capabilities=("mail",),
        budgets={"max_records": 5, "max_characters": 2000},
        freshness_policy="current_or_stale_labeled",
        hindsight_policy="exclude",
        retrieval_version="retrieval-v1",
        ranking_version="ranking-v1",
        requesting_workflow="speaker_identity",
        run_id="run-1",
        created_at="2026-07-26T14:01:00Z",
    )
    repository.save_retrieval_request(request)
    bundle = conversation_knowledge_evidence.EvidenceBundleRecord.create(
        bundle_id=BUNDLE_ID,
        request_id=REQUEST_ID,
        status="partial",
        items=(
            conversation_knowledge_evidence.EvidenceBundleItem(
                evidence_id=GWS_EVIDENCE_ID,
                disposition="included",
                reason_code="exact_identifier_support",
                rank=1,
                score=1.0,
            ),
            conversation_knowledge_evidence.EvidenceBundleItem(
                evidence_id=ODOO_EVIDENCE_ID,
                disposition="excluded",
                reason_code="outside_temporal_policy",
                rank=0,
                score=0.0,
            ),
        ),
        candidate_person_ids=(PERSON_ID,),
        warnings=("provider_partial_failure",),
        source_failures=(
            {
                "source_profile_id": "odoo-company",
                "reason_code": "provider_unavailable",
            },
        ),
        allowlists={"evidence_ids": [GWS_EVIDENCE_ID]},
        created_at="2026-07-26T14:02:00Z",
    )

    assert repository.save_bundle(bundle) == "inserted"
    assert repository.save_bundle(bundle) == "unchanged"
    assert repository.load_bundle(BUNDLE_ID) == bundle
    assert repository.search_concepts("alpha catalyst") == (concept,)

    with pytest.raises(ValueError, match="immutable"):
        repository.save_bundle(
            conversation_knowledge_evidence.EvidenceBundleRecord(
                **{
                    **bundle.__dict__,
                    "status": "complete",
                }
            )
        )


def test_v2_rollback_preserves_v1_domain_records(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.save_snapshot(_snapshots()[0])
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)

    receipt = store.rollback(target_version=1, backup=False)

    assert receipt.rolled_back_versions == (7, 6, 5, 4, 3, 2)
    assert store.schema_status().schema_version == 1
    assert store.load_conversation_snapshot(CONVERSATION_ID) is not None
    with sqlite3.connect(tmp_path / "transcripts.sqlite3") as con:
        evidence_table = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE name = 'knowledge_evidence_snapshots'
            """
        ).fetchone()
    assert evidence_table is None


def test_failed_v2_migration_leaves_version_one_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(target_version=1, backup=False)
    store.save_conversation_snapshot(
        conversation_knowledge_store.ConversationSnapshot(
            conversation=conversation_knowledge_store.ConversationRecord(
                conversation_id=CONVERSATION_ID,
                title="Version one survives",
            )
        )
    )

    def fail_v2(_con: sqlite3.Connection) -> None:
        raise RuntimeError("forced v2 failure")

    monkeypatch.setattr(store, "_apply_v2", fail_v2)
    with pytest.raises(RuntimeError, match="forced v2 failure"):
        store.migrate(backup=False)

    assert store.schema_status().schema_version == 1
    assert store.load_conversation_snapshot(CONVERSATION_ID) is not None
