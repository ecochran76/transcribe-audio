from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_identity_retrieval
import conversation_knowledge_evidence
import conversation_knowledge_profiles
import conversation_knowledge_store


CONVERSATION_ID = "00000000-0000-4000-8000-000000000401"
RECORDING_ID = "00000000-0000-4000-8000-000000000402"
CLUE_ID = "00000000-0000-4000-8000-000000000403"
PERSON_ID = "00000000-0000-4000-8000-000000000404"
REQUEST_ID = "00000000-0000-4000-8000-000000000405"
ADAPTER_EVIDENCE_ID = "00000000-0000-4000-8000-000000000406"
OUT_OF_SCOPE_EVIDENCE_ID = "00000000-0000-4000-8000-000000000407"


def _evidence(
    suffix: int,
    *,
    independence_group: str,
    snippet: str,
    source_record_id: str = "gws-person-1",
    source_profile_id: str = "gws-personal",
    account_id: str = "owner@example.com",
    tenant_id: str = "",
    stance: str = "",
) -> conversation_knowledge_evidence.EvidenceSnapshotRecord:
    return conversation_knowledge_evidence.EvidenceSnapshotRecord(
        evidence_id=f"00000000-0000-4000-8000-{suffix:012d}",
        source_record_id=source_record_id,
        source_profile_id=source_profile_id,
        provider_kind="gws",
        account_id=account_id,
        tenant_id=tenant_id,
        source_type="gmail_message",
        capability="mail",
        snippet=snippet,
        structured_metadata={"stance": stance} if stance else {},
        source_event_at="2026-07-25T14:00:00Z",
        observed_at="2026-07-25T14:05:00Z",
        retrieved_at="2026-07-25T14:06:00Z",
        temporal_class="contemporaneous",
        source_uri=f"gmail:message-{suffix}",
        content_hash=f"evidence-hash-{suffix}",
        independence_group_id=independence_group,
        freshness_state="current",
        embedding=(1.0, 0.0),
        embedding_provider="fixture",
        embedding_model="fixture-v1",
    )


def _root(tmp_path: Path) -> Path:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    store.save_conversation_snapshot(
        conversation_knowledge_store.ConversationSnapshot(
            conversation=conversation_knowledge_store.ConversationRecord(
                conversation_id=CONVERSATION_ID,
                title="Identity retrieval fixture",
                starts_at="2026-07-26T14:00:00Z",
                metadata={
                    "event": {
                        "summary": "Alpha review",
                        "attendees": [
                            {
                                "displayName": "Example Person",
                                "email": "person@example.com",
                                "responseStatus": "accepted",
                            }
                        ],
                    }
                },
            ),
            recordings=(
                conversation_knowledge_store.RecordingRecord(
                    recording_id=RECORDING_ID,
                    conversation_id=CONVERSATION_ID,
                ),
            ),
            utterances=(
                conversation_knowledge_store.UtteranceRecord(
                    utterance_id=CLUE_ID,
                    conversation_id=CONVERSATION_ID,
                    recording_id=RECORDING_ID,
                    speaker_label="A",
                    ordinal=0,
                    start_ms=0,
                    end_ms=1000,
                    text="We should review Alpha catalyst procurement.",
                ),
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
                    source_record_id="gws-person-1",
                    person_id=PERSON_ID,
                    source_profile_id="gws-personal",
                    provider_kind="gws",
                    account_id="owner@example.com",
                    tenant_id="",
                    external_ref="people/1",
                    label="Example Person",
                    relationship_scope="personal_interaction",
                    identifier_authority="email",
                    observed_at="2026-07-25T14:00:00Z",
                    content_hash="contact-hash",
                ),
                conversation_knowledge_store.SourceRecord(
                    source_record_id="odoo-person-1",
                    person_id=PERSON_ID,
                    source_profile_id="odoo-company",
                    provider_kind="odoo",
                    account_id="",
                    tenant_id="company-prod",
                    external_ref="res.partner:1",
                    label="Example Person",
                    relationship_scope="company_interaction",
                    identifier_authority="email",
                    observed_at="2026-07-25T14:01:00Z",
                    content_hash="odoo-contact-hash",
                ),
            ),
            external_identities=(
                conversation_knowledge_store.ExternalIdentityRecord(
                    external_identity_id=(
                        "00000000-0000-4000-8000-000000000408"
                    ),
                    person_id=PERSON_ID,
                    source_record_id="gws-person-1",
                    identity_kind="email",
                    normalized_value="person@example.com",
                    display_value="person@example.com",
                    authority="gws",
                    verified=True,
                ),
                conversation_knowledge_store.ExternalIdentityRecord(
                    external_identity_id=(
                        "00000000-0000-4000-8000-000000000415"
                    ),
                    person_id=PERSON_ID,
                    source_record_id="odoo-person-1",
                    identity_kind="email",
                    normalized_value="person@example.com",
                    display_value="person@example.com",
                    authority="odoo",
                    verified=True,
                ),
            ),
        )
    )
    store.save_processing_history(
        conversation_knowledge_store.ProcessingHistory(
            conversation_id=CONVERSATION_ID,
            current_evaluation_id="",
        )
    )
    profile_projector = (
        conversation_knowledge_profiles.ConversationProfileProjector(
            tmp_path
        )
    )
    profile_projector.append_reviewed_observations(CONVERSATION_ID)
    profile_projector.rebuild()
    repository = (
        conversation_knowledge_evidence.ConversationEvidenceRepository(
            tmp_path
        )
    )
    repository.save_snapshot(
        _evidence(
            409,
            independence_group="interaction-1",
            snippet="Alpha catalyst procurement is led by Example Person.",
        )
    )
    repository.save_snapshot(
        _evidence(
            410,
            independence_group="interaction-1",
            snippet="The same Alpha fact was copied into another message.",
        )
    )
    repository.save_snapshot(
        _evidence(
            411,
            independence_group="interaction-2",
            snippet="Alpha catalyst was not assigned to Example Person.",
            source_record_id="",
            stance="contradict",
        )
    )
    repository.save_snapshot(
        conversation_knowledge_evidence.EvidenceSnapshotRecord(
            **{
                **_evidence(
                    413,
                    independence_group="interaction-stale",
                    snippet="A stale Alpha association must remain labeled.",
                ).__dict__,
                "freshness_state": "stale",
            }
        )
    )
    return tmp_path


class FakePartialAdapter:
    adapter_id = "fake-partial"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def retrieve(
        self,
        request: conversation_identity_retrieval.ProviderRetrievalRequest,
    ) -> conversation_identity_retrieval.ProviderRetrievalResult:
        self.calls += 1
        repository = (
            conversation_knowledge_evidence.ConversationEvidenceRepository(
                self.root
            )
        )
        assert repository.load_retrieval_request(REQUEST_ID) is not None
        assert "alpha" in request.query_terms
        return conversation_identity_retrieval.ProviderRetrievalResult(
            snapshots=(
                conversation_knowledge_evidence.EvidenceSnapshotRecord(
                    **{
                        **_evidence(
                            406,
                            independence_group="interaction-3",
                            snippet=(
                                "A fresh bounded provider result supports "
                                "the Alpha association."
                            ),
                            source_record_id="",
                        ).__dict__,
                        "evidence_id": ADAPTER_EVIDENCE_ID,
                    }
                ),
                conversation_knowledge_evidence.EvidenceSnapshotRecord(
                    **{
                        **_evidence(
                            407,
                            independence_group="interaction-other",
                            snippet="This result belongs to another tenant.",
                            source_record_id="",
                            source_profile_id="odoo-other",
                            account_id="",
                            tenant_id="other-prod",
                        ).__dict__,
                        "evidence_id": OUT_OF_SCOPE_EVIDENCE_ID,
                    }
                ),
            ),
            failures=(
                {
                    "source_profile_id": "gws-personal",
                    "reason_code": "drive_unavailable",
                },
            ),
            warnings=("provider_partial_result",),
        )


def test_prepare_identity_evidence_is_exact_first_bounded_and_replayable(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    adapter = FakePartialAdapter(root)
    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-personal",
                account_id="owner@example.com",
                tenant_id="",
            ),
        ),
        capabilities=("mail",),
        provider_adapters=(adapter,),
        query_embedding=(1.0, 0.0),
        max_records=3,
        max_characters=2_000,
        max_per_source=3,
        max_provider_calls=1,
        max_relationship_hops=1,
        request_id=REQUEST_ID,
        run_id="run-identity-1",
        requested_at="2026-07-26T14:01:00Z",
    )

    first = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_ID,
        speaker_labels=("A",),
        clue_ids=(CLUE_ID,),
        as_of="2026-07-26T14:00:00Z",
        policy=policy,
        root=root,
    )
    second = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_ID,
        speaker_labels=("A",),
        clue_ids=(CLUE_ID,),
        as_of="2026-07-26T14:00:00Z",
        policy=policy,
        root=root,
    )
    repository = (
        conversation_knowledge_evidence.ConversationEvidenceRepository(root)
    )

    assert adapter.calls == 2
    assert first.persisted_bundle.status == "partial"
    assert first.people[0].person_id == PERSON_ID
    assert first.people[0].match_reasons == ("calendar_attendee_email",)
    assert first.calendar_candidates[0].matched_person_ids == (PERSON_ID,)
    assert first.transcript_clues[0].clue_id == CLUE_ID
    assert first.relationships[0].relationship_type == "source_relationship"
    assert {
        item.object_id for item in first.relationships
    } == {"gws-personal"}
    assert {item.direction for item in first.evidence} == {
        "support",
        "contradict",
    }
    assert any(
        item.reason_code == "duplicate_independence_group"
        for item in first.evidence
    )
    assert any(
        item.reason_code == "outside_freshness_policy"
        for item in first.evidence
    )
    included = [
        item
        for item in first.evidence
        if item.disposition == "included"
    ]
    assert len(included) == 3
    assert len(
        {
            item.snapshot.independence_group_id
            for item in included
        }
    ) == len(included)
    assert "provider_partial_result" in first.warnings
    assert {
        item["reason_code"] for item in first.source_failures
    } == {
        "drive_unavailable",
        "out_of_scope_provider_result",
    }
    assert repository.load_snapshot(OUT_OF_SCOPE_EVIDENCE_ID) is None
    assert (
        repository.load_bundle(first.persisted_bundle.bundle_id)
        == first.persisted_bundle
    )
    assert second.persisted_bundle == first.persisted_bundle


def test_calendar_and_prepared_candidates_survive_without_evidence(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="odoo-empty",
                account_id="",
                tenant_id="empty-prod",
            ),
        ),
        capabilities=("log_notes",),
        prepared_person_ids=(PERSON_ID,),
        max_records=2,
        max_characters=500,
        max_per_source=2,
        max_provider_calls=0,
        max_relationship_hops=0,
        request_id="00000000-0000-4000-8000-000000000412",
        requested_at="2026-07-26T14:03:00Z",
    )

    bundle = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_ID,
        policy=policy,
        root=root,
    )

    assert bundle.calendar_candidates[0].email == "person@example.com"
    assert bundle.calendar_candidates[0].matched_person_ids == ()
    assert bundle.people[0].person_id == PERSON_ID
    assert bundle.people[0].match_reasons == ("prepared_person",)
    assert bundle.evidence == ()
    assert bundle.persisted_bundle.status == "complete"
    assert "no_bounded_evidence" in bundle.warnings


def test_candidate_grouping_preserves_every_permitted_source_affinity(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-personal",
                account_id="owner@example.com",
                tenant_id="",
            ),
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="odoo-company",
                account_id="",
                tenant_id="company-prod",
            ),
        ),
        capabilities=("mail", "log_notes"),
        max_records=2,
        max_characters=1_500,
        max_per_source=2,
        max_provider_calls=0,
        max_relationship_hops=1,
        request_id="00000000-0000-4000-8000-000000000416",
        requested_at="2026-07-26T14:03:30Z",
    )

    bundle = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_ID,
        policy=policy,
        root=root,
    )

    assert bundle.people[0].person_id == PERSON_ID
    assert bundle.people[0].source_profile_ids == (
        "gws-personal",
        "odoo-company",
    )
    assert bundle.people[0].source_record_ids == (
        "gws-person-1",
        "odoo-person-1",
    )
    assert {
        item.object_id for item in bundle.relationships
    } == {"gws-personal", "odoo-company"}


class ExplodingAdapter:
    adapter_id = "exploding"

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self,
        request: conversation_identity_retrieval.ProviderRetrievalRequest,
    ) -> conversation_identity_retrieval.ProviderRetrievalResult:
        self.calls += 1
        raise TimeoutError("fixture timeout")


class UncalledAdapter:
    adapter_id = "uncalled"

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self,
        request: conversation_identity_retrieval.ProviderRetrievalRequest,
    ) -> conversation_identity_retrieval.ProviderRetrievalResult:
        self.calls += 1
        return conversation_identity_retrieval.ProviderRetrievalResult()


def test_provider_exception_is_partial_and_call_budget_is_enforced(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    exploding = ExplodingAdapter()
    uncalled = UncalledAdapter()
    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(
            conversation_knowledge_evidence.EvidenceScope(
                source_profile_id="gws-personal",
                account_id="owner@example.com",
                tenant_id="",
            ),
        ),
        capabilities=("mail",),
        provider_adapters=(exploding, uncalled),
        max_records=2,
        max_characters=1_500,
        max_per_source=2,
        max_provider_calls=1,
        max_relationship_hops=0,
        request_id="00000000-0000-4000-8000-000000000414",
        requested_at="2026-07-26T14:04:00Z",
    )

    bundle = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_ID,
        clue_ids=(CLUE_ID,),
        policy=policy,
        root=root,
    )

    assert exploding.calls == 1
    assert uncalled.calls == 0
    assert bundle.persisted_bundle.status == "partial"
    assert bundle.source_failures == (
        {
            "adapter_id": "exploding",
            "reason_code": "provider_exception",
            "detail": "TimeoutError",
        },
    )
    assert "provider_call_budget_exhausted" in bundle.warnings
    assert all(
        item.reason_code != "provider_absence"
        for item in bundle.persisted_bundle.items
    )
