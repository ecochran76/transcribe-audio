from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_knowledge_store
import conversation_identity_retrieval
from conversation_evidence_fabric import (
    EvidenceAnchor,
    EvidenceFabric,
    EvidenceRequest,
    ProviderRetrievalResult,
)
from conversation_knowledge_evidence import (
    ConversationEvidenceRepository,
    EvidenceScope,
    EvidenceSnapshotRecord,
)
from identity_learning_ledger import IdentityLearningLedger


PERSON_A = "00000000-0000-4000-8000-000000007401"
PERSON_B = "00000000-0000-4000-8000-000000007402"
PERSON_C = "00000000-0000-4000-8000-000000007405"
CONVERSATION_A = "00000000-0000-4000-8000-000000007403"
CONVERSATION_B = "00000000-0000-4000-8000-000000007404"


def _append(
    ledger: IdentityLearningLedger,
    *,
    event_type: str,
    payload: dict[str, object],
    ordinal: int,
) -> None:
    ledger.append_event(
        event_type=event_type,
        payload=payload,
        actor_id="reviewer:fixture",
        occurred_at=f"2026-01-{ordinal:02d}T12:00:00Z",
        idempotency_key=f"plan0074-{ordinal}",
    )


def _fabric(tmp_path: Path) -> EvidenceFabric:
    store = conversation_knowledge_store.ConversationKnowledgeStore(tmp_path)
    store.migrate(backup=False)
    for conversation_id, starts_at in (
        (CONVERSATION_A, "2026-01-03T12:00:00Z"),
        (CONVERSATION_B, "2026-01-10T12:00:00Z"),
    ):
        store.save_conversation_snapshot(
            conversation_knowledge_store.ConversationSnapshot(
                conversation=conversation_knowledge_store.ConversationRecord(
                    conversation_id=conversation_id,
                    title="Plan 0074 fixture",
                    starts_at=starts_at,
                )
            )
        )
    ledger = IdentityLearningLedger(tmp_path)
    _append(
        ledger,
        event_type="person_created",
        payload={
            "person_id": PERSON_A,
            "primary_name": "Alex Example",
            "status": "reviewed",
        },
        ordinal=1,
    )
    _append(
        ledger,
        event_type="person_created",
        payload={
            "person_id": PERSON_B,
            "primary_name": "Morgan Example",
            "status": "reviewed",
        },
        ordinal=2,
    )
    _append(
        ledger,
        event_type="relationship_asserted",
        payload={
            "relationship_id": "relationship-plan0074-reviewed",
            "relationship_type": "works_with",
            "subject_type": "person",
            "subject_id": PERSON_A,
            "object_type": "person",
            "object_id": PERSON_B,
            "directionality": "symmetric",
            "starts_at": "2026-01-03T00:00:00Z",
            "status": "reviewed",
            "evidence_ids": ["evidence-plan0074-a"],
            "metadata": {
                "originating_conversation_id": CONVERSATION_A,
                "accepted_at": "2026-01-04T12:00:00Z",
            },
        },
        ordinal=4,
    )
    _append(
        ledger,
        event_type="relationship_asserted",
        payload={
            "relationship_id": "relationship-plan0074-proposed",
            "relationship_type": "advisor_for",
            "subject_type": "person",
            "subject_id": PERSON_A,
            "object_type": "person",
            "object_id": PERSON_B,
            "status": "proposed",
            "evidence_ids": ["evidence-plan0074-proposed"],
            "metadata": {
                "originating_conversation_id": CONVERSATION_A,
                "accepted_at": "2026-01-04T12:00:00Z",
            },
        },
        ordinal=5,
    )
    ledger.rebuild()
    return EvidenceFabric(tmp_path)


def test_reviewed_relationship_becomes_later_context_without_self_corroboration(
    tmp_path: Path,
) -> None:
    fabric = _fabric(tmp_path)
    request = EvidenceRequest(
        purpose="conversation_understanding",
        conversation_id=CONVERSATION_B,
        anchors=(
            EvidenceAnchor("person", PERSON_A),
            EvidenceAnchor("person", PERSON_B),
        ),
        query_terms=(),
        scopes=(EvidenceScope("local-knowledge", "local", "local"),),
        capabilities=("accepted_relationships",),
        as_of="2026-01-10T12:00:00Z",
        hindsight_policy="exclude",
        allowed_freshness_states=("current",),
        max_records=20,
        max_characters=12_000,
        max_provider_calls=0,
        max_relationship_hops=1,
    )

    later = fabric.collect(request)
    repeated = fabric.collect(request)
    origin = fabric.collect(replace(request, conversation_id=CONVERSATION_A))
    historical = fabric.collect(
        replace(request, as_of="2026-01-03T12:00:00Z")
    )
    explicit_hindsight = fabric.collect(
        replace(
            request,
            as_of="2026-01-03T12:00:00Z",
            hindsight_policy="allow_hindsight",
        )
    )

    assert [item.relationship_id for item in later.relationships] == [
        "relationship-plan0074-reviewed"
    ]
    assert later.relationships[0].evidence_ids == ("evidence-plan0074-a",)
    assert later.knowledge_watermark != "empty"
    assert later.content_hash == repeated.content_hash
    assert origin.relationships == ()
    assert "current_conversation_relationship_excluded" in origin.warnings
    assert historical.relationships == ()
    assert "relationship_after_as_of_excluded" in historical.warnings
    assert [
        item.relationship_id for item in explicit_hindsight.relationships
    ] == ["relationship-plan0074-reviewed"]


def test_provider_collection_persists_only_bounded_in_scope_evidence(
    tmp_path: Path,
) -> None:
    fabric = _fabric(tmp_path)
    scope = EvidenceScope("source-profile", "account", "tenant")

    def snapshot(evidence_id: str, *, source_profile_id: str) -> EvidenceSnapshotRecord:
        return EvidenceSnapshotRecord(
            evidence_id=evidence_id,
            source_record_id="",
            source_profile_id=source_profile_id,
            provider_kind="fixture",
            account_id="account",
            tenant_id="tenant",
            source_type="fixture_document",
            capability="document_search",
            snippet="Bounded project context",
            structured_metadata={"title": "Project context"},
            source_event_at="2026-01-08T12:00:00Z",
            observed_at="2026-01-08T12:00:00Z",
            retrieved_at="2026-01-09T12:00:00Z",
            temporal_class="contemporaneous",
            source_uri="fixture:document",
            content_hash=f"hash-{evidence_id}",
            independence_group_id=f"group-{evidence_id}",
            freshness_state="current",
        )

    permitted = snapshot(
        "00000000-0000-4000-8000-000000007411",
        source_profile_id="source-profile",
    )
    rejected = snapshot(
        "00000000-0000-4000-8000-000000007412",
        source_profile_id="another-profile",
    )

    class FixtureAdapter:
        adapter_id = "fixture-document-adapter"

        def __init__(self, *, reverse: bool) -> None:
            self.reverse = reverse

        def retrieve(self, request: object) -> ProviderRetrievalResult:
            snapshots = (rejected, permitted)
            return ProviderRetrievalResult(
                snapshots=(
                    tuple(reversed(snapshots)) if self.reverse else snapshots
                )
            )

    class FailedAdapter:
        adapter_id = "failed-document-adapter"

        def retrieve(self, request: object) -> ProviderRetrievalResult:
            raise RuntimeError("fixture failure")

    request = EvidenceRequest(
        purpose="people_relationship_discovery",
        conversation_id=CONVERSATION_B,
        anchors=(),
        query_terms=("project context",),
        scopes=(scope,),
        capabilities=("document_search",),
        as_of="2026-01-10T12:00:00Z",
        hindsight_policy="exclude",
        allowed_freshness_states=("current",),
        max_records=20,
        max_characters=12_000,
        max_provider_calls=2,
        max_relationship_hops=0,
    )

    bundle = fabric.collect(
        request,
        adapters=(FixtureAdapter(reverse=False), FailedAdapter()),
    )
    repeated = fabric.collect(
        request,
        adapters=(FixtureAdapter(reverse=True), FailedAdapter()),
    )

    assert [item.evidence_id for item in bundle.provider_snapshots] == [
        permitted.evidence_id
    ]
    assert [item["reason_code"] for item in bundle.source_failures] == [
        "provider_exception",
        "out_of_scope_provider_result",
    ]
    assert bundle.content_hash == repeated.content_hash
    repository = ConversationEvidenceRepository(tmp_path)
    assert repository.load_snapshot(permitted.evidence_id) == permitted
    assert repository.load_snapshot(rejected.evidence_id) is None


def test_speaker_identity_facade_reads_accepted_relationships_through_fabric(
    tmp_path: Path,
) -> None:
    _fabric(tmp_path)
    policy = conversation_identity_retrieval.IdentityEvidencePolicy(
        scopes=(EvidenceScope("local-knowledge", "local", "local"),),
        capabilities=("contacts",),
        prepared_person_ids=(PERSON_A, PERSON_B),
        max_provider_calls=0,
        max_relationship_hops=1,
        requested_at="2026-01-10T12:01:00Z",
    )

    bundle = conversation_identity_retrieval.prepare_identity_evidence(
        CONVERSATION_B,
        as_of="2026-01-10T12:00:00Z",
        policy=policy,
        root=tmp_path,
    )

    assert [item.relationship_type for item in bundle.relationships] == [
        "works_with"
    ]
    assert bundle.relationships[0].observation_ids == ("evidence-plan0074-a",)


def test_relationship_expansion_obeys_the_requested_hop_limit(
    tmp_path: Path,
) -> None:
    fabric = _fabric(tmp_path)
    ledger = IdentityLearningLedger(tmp_path)
    _append(
        ledger,
        event_type="person_created",
        payload={
            "person_id": PERSON_C,
            "primary_name": "Casey Example",
            "status": "reviewed",
        },
        ordinal=6,
    )
    _append(
        ledger,
        event_type="relationship_asserted",
        payload={
            "relationship_id": "relationship-plan0074-second-hop",
            "relationship_type": "reports_to",
            "subject_type": "person",
            "subject_id": PERSON_B,
            "object_type": "person",
            "object_id": PERSON_C,
            "directionality": "directional",
            "starts_at": "2026-01-05T00:00:00Z",
            "status": "reviewed",
            "evidence_ids": ["evidence-plan0074-b"],
            "metadata": {
                "originating_conversation_id": "prior-conversation",
                "accepted_at": "2026-01-06T12:00:00Z",
            },
        },
        ordinal=7,
    )
    ledger.rebuild()
    request = EvidenceRequest(
        purpose="conversation_understanding",
        conversation_id=CONVERSATION_B,
        anchors=(EvidenceAnchor("person", PERSON_A),),
        query_terms=(),
        scopes=(EvidenceScope("local-knowledge", "local", "local"),),
        capabilities=("accepted_relationships",),
        as_of="2026-01-10T12:00:00Z",
        hindsight_policy="exclude",
        allowed_freshness_states=("current",),
        max_records=20,
        max_characters=12_000,
        max_provider_calls=0,
        max_relationship_hops=1,
    )

    one_hop = fabric.collect(request)
    two_hops = fabric.collect(replace(request, max_relationship_hops=2))

    assert [item.relationship_id for item in one_hop.relationships] == [
        "relationship-plan0074-reviewed"
    ]
    assert [item.relationship_id for item in two_hops.relationships] == [
        "relationship-plan0074-reviewed",
        "relationship-plan0074-second-hop",
    ]
