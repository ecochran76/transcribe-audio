from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4, uuid5

import transcript_store
from conversation_knowledge_evidence import (
    ConversationEvidenceRepository,
    EvidenceBundleItem,
    EvidenceBundleRecord,
    EvidenceScope,
    EvidenceSnapshotRecord,
    ExternalIdentityMatch,
    RetrievalRequestRecord,
)
from conversation_knowledge_store import ConversationKnowledgeStore


RETRIEVAL_VERSION = "conversation-identity-retrieval.v1"
RANKING_VERSION = "conversation-identity-ranking.v1"
MAX_PROVIDER_QUERY_TERMS = 24
_IDENTITY_NAMESPACE = UUID("fd5f90be-0f38-43df-ac36-68e7e78c29de")


@dataclass(frozen=True)
class ProviderRetrievalRequest:
    conversation_id: str
    query_terms: tuple[str, ...]
    scopes: tuple[EvidenceScope, ...]
    capabilities: tuple[str, ...]
    as_of: str
    max_records: int
    max_characters: int


@dataclass(frozen=True)
class ProviderRetrievalResult:
    snapshots: tuple[EvidenceSnapshotRecord, ...] = ()
    failures: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()


class HostEvidenceAdapter(Protocol):
    adapter_id: str

    def retrieve(
        self,
        request: ProviderRetrievalRequest,
    ) -> ProviderRetrievalResult: ...


@dataclass(frozen=True)
class IdentityEvidencePolicy:
    scopes: tuple[EvidenceScope, ...]
    capabilities: tuple[str, ...]
    prepared_query_terms: tuple[str, ...] = ()
    prepared_person_ids: tuple[str, ...] = ()
    authoritative_identifiers: tuple[tuple[str, str], ...] = ()
    provider_adapters: tuple[HostEvidenceAdapter, ...] = ()
    query_embedding: tuple[float, ...] = ()
    allowed_freshness_states: tuple[str, ...] = ("current",)
    hindsight_policy: str = "exclude"
    freshness_policy: str = "current_only"
    max_records: int = 20
    max_characters: int = 12_000
    max_per_source: int = 5
    max_provider_calls: int = 4
    max_relationship_hops: int = 1
    request_id: str = ""
    run_id: str = ""
    requested_at: str = ""


@dataclass(frozen=True)
class CalendarIdentityCandidate:
    name: str
    email: str
    response_status: str
    matched_person_ids: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptClue:
    clue_id: str
    speaker_label: str
    text: str
    recording_id: str


@dataclass(frozen=True)
class IdentityCandidate:
    person_id: str
    source_record_ids: tuple[str, ...]
    source_profile_ids: tuple[str, ...]
    match_reasons: tuple[str, ...]
    exact_identities: tuple[str, ...]
    display_name: str = ""


@dataclass(frozen=True)
class RelationshipSummary:
    subject_id: str
    relationship_type: str
    object_type: str
    object_id: str
    display_value: str
    observation_ids: tuple[str, ...]
    input_watermark: str


@dataclass(frozen=True)
class RankedEvidence:
    snapshot: EvidenceSnapshotRecord
    score: float
    direction: str
    features: dict[str, float]
    disposition: str
    reason_code: str


@dataclass(frozen=True)
class PreparedIdentityEvidenceBundle:
    request: RetrievalRequestRecord
    persisted_bundle: EvidenceBundleRecord
    calendar_candidates: tuple[CalendarIdentityCandidate, ...]
    transcript_clues: tuple[TranscriptClue, ...]
    people: tuple[IdentityCandidate, ...]
    relationships: tuple[RelationshipSummary, ...]
    evidence: tuple[RankedEvidence, ...]
    warnings: tuple[str, ...]
    source_failures: tuple[dict[str, str], ...]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _stable_uuid(*parts: str) -> str:
    return str(uuid5(_IDENTITY_NAMESPACE, "\x1f".join(parts)))


def _normalize_email(value: str) -> str:
    return value.strip().casefold()


def _scope_key(scope: EvidenceScope) -> tuple[str, str, str]:
    return (
        scope.source_profile_id,
        scope.account_id,
        scope.tenant_id,
    )


def _event_attendees(event: dict[str, object]) -> tuple[dict[str, str], ...]:
    raw = event.get("attendees")
    if not isinstance(raw, list):
        return ()
    attendees: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            attendees.append(
                {
                    "name": "",
                    "email": _normalize_email(item),
                    "response_status": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        email = _normalize_email(
            str(item.get("email") or item.get("emailAddress") or "")
        )
        name = str(
            item.get("displayName")
            or item.get("name")
            or item.get("label")
            or ""
        ).strip()
        if not email and not name:
            continue
        attendees.append(
            {
                "name": name,
                "email": email,
                "response_status": str(
                    item.get("responseStatus")
                    or item.get("response_status")
                    or ""
                ),
            }
        )
    return tuple(attendees)


def _snapshot_is_allowed(
    snapshot: EvidenceSnapshotRecord,
    policy: IdentityEvidencePolicy,
    *,
    as_of: str,
) -> bool:
    if (
        snapshot.source_profile_id,
        snapshot.account_id,
        snapshot.tenant_id,
    ) not in {_scope_key(scope) for scope in policy.scopes}:
        return False
    if snapshot.capability not in policy.capabilities:
        return False
    if snapshot.freshness_state not in policy.allowed_freshness_states:
        return False
    if snapshot.source_event_at and snapshot.source_event_at > as_of:
        return False
    allowed_temporal = {
        "exclude": {"contemporaneous"},
        "allow_later_retrieved": {
            "contemporaneous",
            "later_retrieved",
        },
        "allow_hindsight": {
            "contemporaneous",
            "later_retrieved",
            "hindsight",
        },
    }.get(policy.hindsight_policy)
    if allowed_temporal is None or snapshot.temporal_class not in allowed_temporal:
        return False
    if (
        snapshot.temporal_class == "contemporaneous"
        and snapshot.observed_at > as_of
    ):
        return False
    return True


def prepare_identity_evidence(
    conversation_id: str,
    *,
    speaker_labels: tuple[str, ...] = (),
    clue_ids: tuple[str, ...] = (),
    as_of: str | None = None,
    policy: IdentityEvidencePolicy | None = None,
    root: Path | None = None,
) -> PreparedIdentityEvidenceBundle:
    """Prepare and persist one host-owned, bounded identity evidence bundle."""
    if policy is None:
        raise ValueError("Identity evidence retrieval requires an explicit policy.")
    if not policy.scopes or not policy.capabilities:
        raise ValueError("Identity evidence policy requires scopes and capabilities.")
    if not policy.allowed_freshness_states:
        raise ValueError("Identity evidence policy requires freshness states.")
    if min(
        policy.max_records,
        policy.max_characters,
        policy.max_per_source,
        policy.max_provider_calls,
    ) < 0:
        raise ValueError("Identity evidence budgets cannot be negative.")
    if policy.max_relationship_hops < 0 or policy.max_relationship_hops > 2:
        raise ValueError("Identity evidence relationship hops must be between 0 and 2.")

    store = ConversationKnowledgeStore(root)
    snapshot = store.load_conversation_snapshot(conversation_id)
    if snapshot is None:
        raise ValueError("Conversation does not exist in the knowledge store.")
    repository = ConversationEvidenceRepository(root)
    effective_as_of = (
        str(as_of or "").strip()
        or snapshot.conversation.starts_at
        or _utc_now()
    )
    requested_at = policy.requested_at or _utc_now()
    request_id = policy.request_id or str(uuid4())
    UUID(request_id)
    event = snapshot.conversation.metadata.get("event")
    event = event if isinstance(event, dict) else {}

    clues = tuple(
        TranscriptClue(
            clue_id=item.utterance_id,
            speaker_label=item.speaker_label,
            text=item.text,
            recording_id=item.recording_id,
        )
        for item in snapshot.utterances
        if item.utterance_id in set(clue_ids)
    )
    missing_clue_ids = sorted(set(clue_ids) - {item.clue_id for item in clues})
    query_terms = _query_terms(
        attendees=_event_attendees(event),
        clues=clues,
        speaker_labels=speaker_labels,
        prepared_terms=policy.prepared_query_terms,
    )
    request = RetrievalRequestRecord(
        request_id=request_id,
        conversation_id=conversation_id,
        recording_ids=tuple(
            item.recording_id for item in snapshot.recordings
        ),
        speaker_labels=tuple(speaker_labels),
        clue_ids=tuple(clue_ids),
        conversation_at=snapshot.conversation.starts_at,
        as_of=effective_as_of,
        prepared_person_ids=tuple(policy.prepared_person_ids),
        scopes=tuple(policy.scopes),
        capabilities=tuple(policy.capabilities),
        budgets={
            "max_records": policy.max_records,
            "max_characters": policy.max_characters,
            "max_per_source": policy.max_per_source,
            "max_provider_calls": policy.max_provider_calls,
            "max_relationship_hops": policy.max_relationship_hops,
            "provider_adapter_ids": [
                adapter.adapter_id
                for adapter in policy.provider_adapters[
                    : policy.max_provider_calls
                ]
            ],
            "allowed_freshness_states": list(
                policy.allowed_freshness_states
            ),
            "authoritative_identifiers": [
                [kind, value]
                for kind, value in policy.authoritative_identifiers
            ],
            "query_terms": list(query_terms),
            "query_embedding": {
                "dimensions": len(policy.query_embedding),
                "sha256": (
                    _canonical_hash(list(policy.query_embedding))
                    if policy.query_embedding
                    else ""
                ),
            },
        },
        freshness_policy=policy.freshness_policy,
        hindsight_policy=policy.hindsight_policy,
        retrieval_version=RETRIEVAL_VERSION,
        ranking_version=RANKING_VERSION,
        requesting_workflow="speaker_identity",
        run_id=policy.run_id,
        created_at=requested_at,
    )
    repository.save_retrieval_request(request)

    source_failures: list[dict[str, str]] = []
    warnings: list[str] = []
    for adapter in policy.provider_adapters[: policy.max_provider_calls]:
        adapter_request = ProviderRetrievalRequest(
            conversation_id=conversation_id,
            query_terms=query_terms,
            scopes=policy.scopes,
            capabilities=policy.capabilities,
            as_of=effective_as_of,
            max_records=policy.max_records,
            max_characters=policy.max_characters,
        )
        try:
            result = adapter.retrieve(adapter_request)
        except Exception as exc:
            source_failures.append(
                {
                    "adapter_id": adapter.adapter_id,
                    "reason_code": "provider_exception",
                    "detail": type(exc).__name__,
                }
            )
            continue
        warnings.extend(result.warnings)
        source_failures.extend(dict(item) for item in result.failures)
        for provider_snapshot in result.snapshots:
            if not _snapshot_is_allowed(
                provider_snapshot,
                policy,
                as_of=effective_as_of,
            ):
                source_failures.append(
                    {
                        "adapter_id": adapter.adapter_id,
                        "reason_code": "out_of_scope_provider_result",
                        "detail": provider_snapshot.evidence_id,
                    }
                )
                continue
            repository.save_snapshot(provider_snapshot)

    people, calendar_candidates = _exact_candidates(
        repository,
        event=event,
        policy=policy,
    )
    candidate_source_ids = tuple(
        sorted(
            {
                source_id
                for person in people
                for source_id in person.source_record_ids
            }
        )
    )
    pool: dict[str, EvidenceSnapshotRecord] = {}
    exact_evidence_ids: set[str] = set()
    if candidate_source_ids:
        for evidence in repository.scoped_snapshots(
            scopes=policy.scopes,
            capabilities=policy.capabilities,
            as_of=effective_as_of,
            hindsight_policy=policy.hindsight_policy,
            source_record_ids=candidate_source_ids,
        ):
            pool[evidence.evidence_id] = evidence
            exact_evidence_ids.add(evidence.evidence_id)
    lexical_ids: set[str] = set()
    for term in query_terms[:24]:
        for evidence in repository.search_snapshots(
            term,
            scopes=policy.scopes,
            capabilities=policy.capabilities,
            as_of=effective_as_of,
            hindsight_policy=policy.hindsight_policy,
            limit=max(policy.max_records * 4, 20),
        ):
            pool[evidence.evidence_id] = evidence
            lexical_ids.add(evidence.evidence_id)
    semantic_rank: dict[str, int] = {}
    if policy.query_embedding:
        for rank, evidence in enumerate(
            repository.semantic_snapshots(
                policy.query_embedding,
                scopes=policy.scopes,
                capabilities=policy.capabilities,
                as_of=effective_as_of,
                hindsight_policy=policy.hindsight_policy,
                limit=max(policy.max_records * 4, 20),
            ),
            start=1,
        ):
            pool[evidence.evidence_id] = evidence
            semantic_rank[evidence.evidence_id] = rank

    relationships = _relationship_summaries(
        root=store.root,
        person_ids=tuple(item.person_id for item in people),
        max_hops=policy.max_relationship_hops,
        scopes=policy.scopes,
        capabilities=policy.capabilities,
    )
    candidate_source_profiles = {
        profile_id
        for person in people
        for profile_id in person.source_profile_ids
    }
    reserved_characters = len(
        json.dumps(
            {
                "calendar_candidates": [
                    item.__dict__ for item in calendar_candidates
                ],
                "transcript_clues": [
                    item.__dict__ for item in clues
                ],
                "people": [item.__dict__ for item in people],
                "relationships": [
                    item.__dict__ for item in relationships
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    ranked = _rank_and_budget(
        tuple(pool.values()),
        exact_evidence_ids=exact_evidence_ids,
        lexical_ids=lexical_ids,
        semantic_rank=semantic_rank,
        candidate_source_profiles=candidate_source_profiles,
        policy=policy,
        reserved_characters=reserved_characters,
    )
    if missing_clue_ids:
        warnings.append("missing_clue_ids")
    if not ranked:
        warnings.append("no_bounded_evidence")
    if len(policy.provider_adapters) > policy.max_provider_calls:
        warnings.append("provider_call_budget_exhausted")
    status = "partial" if source_failures else "complete"
    bundle_items_list: list[EvidenceBundleItem] = []
    included_rank = 0
    for item in ranked:
        if item.disposition == "included":
            included_rank += 1
        bundle_items_list.append(
            EvidenceBundleItem(
                evidence_id=item.snapshot.evidence_id,
                disposition=item.disposition,
                reason_code=item.reason_code,
                rank=(
                    included_rank
                    if item.disposition == "included"
                    else 0
                ),
                score=item.score,
                metadata={
                    "direction": item.direction,
                    "features": item.features,
                    "independence_group_id": (
                        item.snapshot.independence_group_id
                    ),
                    "freshness_state": item.snapshot.freshness_state,
                    "temporal_class": item.snapshot.temporal_class,
                },
            )
        )
    bundle_items = tuple(
        sorted(
            bundle_items_list,
            key=lambda item: (
                0 if item.disposition == "included" else 1,
                item.rank,
                item.evidence_id,
            ),
        )
    )
    bundle_id = _stable_uuid(
        "evidence-bundle",
        request_id,
        _canonical_hash(
            {
                "items": [
                    {
                        "evidence_id": item.evidence_id,
                        "disposition": item.disposition,
                        "reason_code": item.reason_code,
                        "score": item.score,
                    }
                    for item in bundle_items
                ],
                "source_failures": source_failures,
                "warnings": sorted(set(warnings)),
            }
        ),
    )
    persisted = EvidenceBundleRecord.create(
        bundle_id=bundle_id,
        request_id=request_id,
        status=status,
        items=bundle_items,
        candidate_person_ids=tuple(item.person_id for item in people),
        warnings=tuple(sorted(set(warnings))),
        source_failures=tuple(source_failures),
        allowlists={
            "person_ids": [item.person_id for item in people],
            "evidence_ids": [
                item.snapshot.evidence_id
                for item in ranked
                if item.disposition == "included"
            ],
            "speaker_labels": list(speaker_labels),
            "clue_ids": [item.clue_id for item in clues],
        },
        created_at=requested_at,
    )
    repository.save_bundle(persisted)
    return PreparedIdentityEvidenceBundle(
        request=request,
        persisted_bundle=persisted,
        calendar_candidates=calendar_candidates,
        transcript_clues=clues,
        people=people,
        relationships=relationships,
        evidence=ranked,
        warnings=persisted.warnings,
        source_failures=tuple(source_failures),
    )


def _query_terms(
    *,
    attendees: tuple[dict[str, str], ...],
    clues: tuple[TranscriptClue, ...],
    speaker_labels: tuple[str, ...],
    prepared_terms: tuple[str, ...] = (),
) -> tuple[str, ...]:
    values: list[str] = []
    for attendee in attendees:
        values.extend((attendee["email"], attendee["name"]))
    values.extend(prepared_terms)
    for clue in clues:
        values.extend(
            token
            for token in transcript_store.tokens(clue.text)
            if len(token) >= 3
        )
    values.extend(
        label
        for label in speaker_labels
        if len(label.strip()) >= 3
        and not label.strip().casefold().startswith("speaker ")
    )
    seen: set[str] = set()
    return tuple(
        value.strip()
        for value in values
        if value.strip()
        and not (
            value.strip().casefold() in seen
            or seen.add(value.strip().casefold())
        )
    )[:MAX_PROVIDER_QUERY_TERMS]


def _exact_candidates(
    repository: ConversationEvidenceRepository,
    *,
    event: dict[str, object],
    policy: IdentityEvidencePolicy,
) -> tuple[
    tuple[IdentityCandidate, ...],
    tuple[CalendarIdentityCandidate, ...],
]:
    candidate_state: dict[str, dict[str, set[str]]] = {}
    calendar: list[CalendarIdentityCandidate] = []

    def add_match(match: ExternalIdentityMatch, reason: str, identity: str) -> None:
        state = candidate_state.setdefault(
            match.person_id,
            {
                "source_record_ids": set(),
                "source_profile_ids": set(),
                "match_reasons": set(),
                "exact_identities": set(),
            },
        )
        state["source_record_ids"].add(match.source_record_id)
        state["source_profile_ids"].add(match.source_profile_id)
        state["match_reasons"].add(reason)
        state["exact_identities"].add(identity)

    for attendee in _event_attendees(event):
        matches = (
            repository.find_people_by_external_identity(
                "email",
                attendee["email"],
                scopes=policy.scopes,
            )
            if attendee["email"]
            else ()
        )
        for match in matches:
            add_match(
                match,
                "calendar_attendee_email",
                f"email:{attendee['email']}",
            )
        calendar.append(
            CalendarIdentityCandidate(
                name=attendee["name"],
                email=attendee["email"],
                response_status=attendee["response_status"],
                matched_person_ids=tuple(
                    sorted({item.person_id for item in matches})
                ),
            )
        )
    for identity_kind, value in policy.authoritative_identifiers:
        for match in repository.find_people_by_external_identity(
            identity_kind,
            value,
            scopes=policy.scopes,
        ):
            add_match(
                match,
                "authoritative_identifier",
                f"{identity_kind}:{value.strip().casefold()}",
            )
    for person_id in policy.prepared_person_ids:
        state = candidate_state.setdefault(
            person_id,
            {
                "source_record_ids": set(),
                "source_profile_ids": set(),
                "match_reasons": set(),
                "exact_identities": set(),
            },
        )
        state["match_reasons"].add("prepared_person")
    for source in repository.source_records_for_people(
        tuple(sorted(candidate_state)),
        scopes=policy.scopes,
    ):
        state = candidate_state[source.person_id]
        state["source_record_ids"].add(source.source_record_id)
        state["source_profile_ids"].add(source.source_profile_id)
    people = tuple(
        IdentityCandidate(
            person_id=person_id,
            source_record_ids=tuple(
                sorted(state["source_record_ids"])
            ),
            source_profile_ids=tuple(
                sorted(state["source_profile_ids"])
            ),
            match_reasons=tuple(sorted(state["match_reasons"])),
            exact_identities=tuple(sorted(state["exact_identities"])),
            display_name=(
                snapshot.person.primary_name
                if (
                    snapshot := ConversationKnowledgeStore(
                        repository.root
                    ).load_person_snapshot(person_id)
                )
                is not None
                else ""
            ),
        )
        for person_id, state in sorted(candidate_state.items())
    )
    return people, tuple(calendar)


def _relationship_summaries(
    *,
    root: Path,
    person_ids: tuple[str, ...],
    max_hops: int,
    scopes: tuple[EvidenceScope, ...],
    capabilities: tuple[str, ...],
) -> tuple[RelationshipSummary, ...]:
    if not person_ids or max_hops == 0:
        return ()
    placeholders = ",".join("?" for _ in person_ids)
    summaries: list[RelationshipSummary] = []
    with transcript_store.connect(root) as con:
        rows = con.execute(
            f"""
            SELECT *
            FROM knowledge_affinity_profiles
            WHERE subject_type = 'person'
              AND subject_id IN ({placeholders})
            ORDER BY subject_id, affinity_type, object_type,
                     object_id, normalized_value
            LIMIT 200
            """,
            person_ids,
        ).fetchall()
        relationship_rows = con.execute(
            f"""
            SELECT *
            FROM knowledge_relationships
            WHERE subject_type = 'person'
              AND subject_id IN ({placeholders})
            ORDER BY subject_id, relationship_type, object_type, object_id
            LIMIT 200
            """,
            person_ids,
        ).fetchall()
    permitted_scope_keys = {_scope_key(scope) for scope in scopes}
    for row in rows:
        affinity_type = str(row["affinity_type"])
        if affinity_type == "source_relationship":
            metadata = json.loads(str(row["metadata_json"]) or "{}")
            account_ids = {
                str(value)
                for value in metadata.get("account_ids", [])
            }
            tenant_ids = {
                str(value)
                for value in metadata.get("tenant_ids", [])
            }
            if not any(
                str(row["object_id"]) == profile_id
                and account_id in account_ids
                and tenant_id in tenant_ids
                for profile_id, account_id, tenant_id in permitted_scope_keys
            ):
                continue
        elif not {"reviewed_history", "relationships"} & set(capabilities):
            continue
        summaries.append(
            RelationshipSummary(
                subject_id=str(row["subject_id"]),
                relationship_type=affinity_type,
                object_type=str(row["object_type"]),
                object_id=str(row["object_id"]),
                display_value=str(
                    row["display_value"] or row["normalized_value"]
                ),
                observation_ids=tuple(
                    str(item)
                    for item in json.loads(
                        str(row["observation_ids_json"])
                    )
                ),
                input_watermark=str(row["input_watermark"]),
            )
        )
    for row in relationship_rows:
        if not {"reviewed_history", "relationships"} & set(capabilities):
            continue
        observation_id = str(row["source_observation_id"] or "")
        summaries.append(
            RelationshipSummary(
                subject_id=str(row["subject_id"]),
                relationship_type=str(row["relationship_type"]),
                object_type=str(row["object_type"]),
                object_id=str(row["object_id"]),
                display_value="",
                observation_ids=(
                    (observation_id,) if observation_id else ()
                ),
                input_watermark="",
            )
        )
    return tuple(summaries)


def _rank_and_budget(
    snapshots: tuple[EvidenceSnapshotRecord, ...],
    *,
    exact_evidence_ids: set[str],
    lexical_ids: set[str],
    semantic_rank: dict[str, int],
    candidate_source_profiles: set[str],
    policy: IdentityEvidencePolicy,
    reserved_characters: int,
) -> tuple[RankedEvidence, ...]:
    prepared: list[
        tuple[float, str, EvidenceSnapshotRecord, dict[str, float]]
    ] = []
    for snapshot in snapshots:
        features: dict[str, float] = {
            "exact_source_record": (
                100.0 if snapshot.evidence_id in exact_evidence_ids else 0.0
            ),
            "lexical_match": (
                30.0 if snapshot.evidence_id in lexical_ids else 0.0
            ),
            "semantic_rank": (
                max(
                    0.0,
                    25.0
                    - float(semantic_rank[snapshot.evidence_id] - 1),
                )
                if snapshot.evidence_id in semantic_rank
                else 0.0
            ),
            "source_affinity": (
                10.0
                if snapshot.source_profile_id in candidate_source_profiles
                else 0.0
            ),
            "temporal_fit": {
                "contemporaneous": 10.0,
                "later_retrieved": 3.0,
                "hindsight": 0.0,
            }.get(snapshot.temporal_class, 0.0),
        }
        direction = (
            "contradict"
            if str(
                snapshot.structured_metadata.get("stance") or ""
            ).casefold()
            in {"contradict", "contradiction"}
            else "support"
        )
        score = sum(features.values())
        if direction == "contradict":
            features["contradiction_priority"] = 5.0
            score += 5.0
        prepared.append(
            (score, snapshot.evidence_id, snapshot, features)
        )
    prepared.sort(key=lambda item: (-item[0], item[1]))
    included_groups: set[str] = set()
    source_counts: dict[str, int] = {}
    record_count = 0
    character_count = reserved_characters
    results: list[RankedEvidence] = []
    for score, _evidence_id, snapshot, features in prepared:
        direction = (
            "contradict"
            if "contradiction_priority" in features
            else "support"
        )
        packet_chars = len(snapshot.snippet) + len(
            json.dumps(
                snapshot.structured_metadata,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        disposition = "included"
        reason = (
            "ranked_contradicting_evidence"
            if direction == "contradict"
            else "ranked_supporting_evidence"
        )
        if snapshot.freshness_state not in policy.allowed_freshness_states:
            disposition = "excluded"
            reason = "outside_freshness_policy"
        elif snapshot.independence_group_id in included_groups:
            disposition = "excluded"
            reason = "duplicate_independence_group"
        elif record_count >= policy.max_records:
            disposition = "excluded"
            reason = "record_budget_exhausted"
        elif character_count + packet_chars > policy.max_characters:
            disposition = "excluded"
            reason = "character_budget_exhausted"
        elif (
            source_counts.get(snapshot.source_profile_id, 0)
            >= policy.max_per_source
        ):
            disposition = "excluded"
            reason = "per_source_budget_exhausted"
        if disposition == "included":
            record_count += 1
            character_count += packet_chars
            included_groups.add(snapshot.independence_group_id)
            source_counts[snapshot.source_profile_id] = (
                source_counts.get(snapshot.source_profile_id, 0) + 1
            )
        results.append(
            RankedEvidence(
                snapshot=snapshot,
                score=score,
                direction=direction,
                features=features,
                disposition=disposition,
                reason_code=reason,
            )
        )
    return tuple(results)
