from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import transcript_store
from conversation_knowledge_evidence import (
    ConversationEvidenceRepository,
    EvidenceScope,
    EvidenceSnapshotRecord,
)


SUPPORTED_PURPOSES = {
    "people_relationship_discovery",
    "speaker_identity",
    "conversation_understanding",
}
SUPPORTED_HINDSIGHT_POLICIES = {
    "exclude",
    "allow_later_retrieved",
    "allow_hindsight",
}
ACCEPTED_RELATIONSHIP_STATUSES = {"accepted", "reviewed"}
ACCEPTED_ACTIVITY_STATUSES = {"accepted", "confirmed", "observed"}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_time(value: str, *, field_name: str) -> datetime:
    normalized = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone offset.")
    return parsed


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


class EvidenceAdapter(Protocol):
    adapter_id: str

    def retrieve(
        self,
        request: ProviderRetrievalRequest,
    ) -> ProviderRetrievalResult: ...


@dataclass(frozen=True)
class EvidenceAnchor:
    anchor_type: str
    anchor_id: str


@dataclass(frozen=True)
class EvidenceRequest:
    purpose: str
    conversation_id: str
    anchors: tuple[EvidenceAnchor, ...]
    query_terms: tuple[str, ...]
    scopes: tuple[EvidenceScope, ...]
    capabilities: tuple[str, ...]
    as_of: str
    hindsight_policy: str
    allowed_freshness_states: tuple[str, ...]
    max_records: int
    max_characters: int
    max_provider_calls: int
    max_relationship_hops: int


@dataclass(frozen=True)
class AcceptedRelationship:
    relationship_id: str
    relationship_type: str
    subject_type: str
    subject_id: str
    object_type: str
    object_id: str
    directionality: str
    starts_at: str
    ends_at: str
    status: str
    evidence_ids: tuple[str, ...]
    originating_conversation_id: str
    accepted_at: str
    input_watermark: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedActivity:
    observation_id: str
    subject_type: str
    subject_id: str
    channel: str
    occurred_at: str
    participation_status: str
    evidence_status: str
    source_profile_id: str
    account_id: str
    tenant_id: str
    source_record_id: str
    independence_group_id: str
    source_locator: dict[str, Any]
    originating_conversation_id: str
    accepted_at: str
    input_watermark: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBundle:
    purpose: str
    conversation_id: str
    provider_snapshots: tuple[EvidenceSnapshotRecord, ...]
    relationships: tuple[AcceptedRelationship, ...]
    activities: tuple[AcceptedActivity, ...]
    warnings: tuple[str, ...]
    source_failures: tuple[dict[str, str], ...]
    knowledge_watermark: str
    content_hash: str


class EvidenceFabric:
    """Collect bounded source evidence and accepted knowledge behind one seam."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)

    def collect(
        self,
        request: EvidenceRequest,
        *,
        adapters: tuple[EvidenceAdapter, ...] = (),
    ) -> EvidenceBundle:
        self._validate_request(request)
        provider_snapshots, provider_warnings, source_failures = (
            self._collect_provider_snapshots(request, adapters)
        )
        relationships, relationship_warnings, watermark = (
            self._accepted_relationships(request)
        )
        activities, activity_warnings, activity_watermark = (
            self._accepted_activities(request)
        )
        if watermark == "empty":
            watermark = activity_watermark
        warnings = tuple(
            sorted(
                set(provider_warnings)
                .union(relationship_warnings)
                .union(activity_warnings)
            )
        )
        ordered_failures = tuple(
            sorted(
                source_failures,
                key=lambda item: (
                    item.get("adapter_id", ""),
                    item.get("reason_code", ""),
                    item.get("detail", ""),
                ),
            )
        )
        semantic = {
            "purpose": request.purpose,
            "conversation_id": request.conversation_id,
            "provider_snapshots": [
                asdict(item) for item in provider_snapshots
            ],
            "relationships": [asdict(item) for item in relationships],
            "activities": [asdict(item) for item in activities],
            "warnings": list(warnings),
            "source_failures": list(ordered_failures),
            "knowledge_watermark": watermark,
        }
        return EvidenceBundle(
            purpose=request.purpose,
            conversation_id=request.conversation_id,
            provider_snapshots=provider_snapshots,
            relationships=relationships,
            activities=activities,
            warnings=warnings,
            source_failures=ordered_failures,
            knowledge_watermark=watermark,
            content_hash=_canonical_hash(semantic),
        )

    def _collect_provider_snapshots(
        self,
        request: EvidenceRequest,
        adapters: tuple[EvidenceAdapter, ...],
    ) -> tuple[
        tuple[EvidenceSnapshotRecord, ...],
        list[str],
        list[dict[str, str]],
    ]:
        repository = ConversationEvidenceRepository(self.root)
        warnings: list[str] = []
        failures: list[dict[str, str]] = []
        pool: dict[str, EvidenceSnapshotRecord] = {}
        adapter_request = ProviderRetrievalRequest(
            conversation_id=request.conversation_id,
            query_terms=request.query_terms,
            scopes=request.scopes,
            capabilities=request.capabilities,
            as_of=request.as_of,
            max_records=request.max_records,
            max_characters=request.max_characters,
        )
        for adapter in adapters[: request.max_provider_calls]:
            try:
                result = adapter.retrieve(adapter_request)
            except Exception as exc:
                failures.append(
                    {
                        "adapter_id": adapter.adapter_id,
                        "reason_code": "provider_exception",
                        "detail": type(exc).__name__,
                    }
                )
                continue
            warnings.extend(result.warnings)
            failures.extend(dict(item) for item in result.failures)
            for snapshot in result.snapshots:
                if not self._snapshot_is_allowed(snapshot, request):
                    failures.append(
                        {
                            "adapter_id": adapter.adapter_id,
                            "reason_code": "out_of_scope_provider_result",
                            "detail": snapshot.evidence_id,
                        }
                    )
                    continue
                existing = pool.get(snapshot.evidence_id)
                if existing is not None and existing != snapshot:
                    failures.append(
                        {
                            "adapter_id": adapter.adapter_id,
                            "reason_code": "conflicting_provider_result",
                            "detail": snapshot.evidence_id,
                        }
                    )
                    continue
                pool[snapshot.evidence_id] = snapshot
        if len(adapters) > request.max_provider_calls:
            warnings.append("provider_call_budget_exhausted")

        included: list[EvidenceSnapshotRecord] = []
        consumed_characters = 0
        for snapshot in sorted(pool.values(), key=lambda item: item.evidence_id):
            snapshot_size = len(snapshot.snippet) + len(
                _canonical_json(snapshot.structured_metadata)
            )
            if (
                len(included) >= request.max_records
                or consumed_characters + snapshot_size > request.max_characters
            ):
                warnings.append("evidence_budget_exhausted")
                continue
            repository.save_snapshot(snapshot)
            included.append(snapshot)
            consumed_characters += snapshot_size
        return tuple(included), warnings, failures

    @staticmethod
    def _snapshot_is_allowed(
        snapshot: EvidenceSnapshotRecord,
        request: EvidenceRequest,
    ) -> bool:
        scope_keys = {
            (scope.source_profile_id, scope.account_id, scope.tenant_id)
            for scope in request.scopes
        }
        if (
            snapshot.source_profile_id,
            snapshot.account_id,
            snapshot.tenant_id,
        ) not in scope_keys:
            return False
        if snapshot.capability not in request.capabilities:
            return False
        if snapshot.freshness_state not in request.allowed_freshness_states:
            return False
        as_of = _parse_time(request.as_of, field_name="as_of")
        if (
            snapshot.source_event_at
            and _parse_time(
                snapshot.source_event_at,
                field_name="source_event_at",
            )
            > as_of
        ):
            return False
        allowed_temporal = {
            "exclude": {"contemporaneous"},
            "allow_later_retrieved": {"contemporaneous", "later_retrieved"},
            "allow_hindsight": {
                "contemporaneous",
                "later_retrieved",
                "hindsight",
            },
        }[request.hindsight_policy]
        if snapshot.temporal_class not in allowed_temporal:
            return False
        if (
            snapshot.temporal_class == "contemporaneous"
            and _parse_time(snapshot.observed_at, field_name="observed_at") > as_of
        ):
            return False
        return True

    @staticmethod
    def _validate_request(request: EvidenceRequest) -> None:
        if request.purpose not in SUPPORTED_PURPOSES:
            raise ValueError("Evidence purpose is unsupported.")
        if not request.conversation_id:
            raise ValueError("Evidence request requires a conversation ID.")
        if not request.scopes or not request.capabilities:
            raise ValueError("Evidence request requires scopes and capabilities.")
        if any(not scope.source_profile_id for scope in request.scopes):
            raise ValueError("Every evidence scope requires a source profile.")
        if any(not capability for capability in request.capabilities):
            raise ValueError("Evidence capabilities cannot be blank.")
        if any(
            not anchor.anchor_type or not anchor.anchor_id
            for anchor in request.anchors
        ):
            raise ValueError("Evidence anchors require a type and identifier.")
        if not request.allowed_freshness_states:
            raise ValueError("Evidence request requires freshness states.")
        if request.hindsight_policy not in SUPPORTED_HINDSIGHT_POLICIES:
            raise ValueError("Evidence hindsight policy is unsupported.")
        if min(
            request.max_records,
            request.max_characters,
            request.max_provider_calls,
            request.max_relationship_hops,
        ) < 0:
            raise ValueError("Evidence request budgets cannot be negative.")
        if request.max_relationship_hops > 2:
            raise ValueError("Evidence relationship hops must be between 0 and 2.")
        _parse_time(request.as_of, field_name="as_of")

    def _accepted_relationships(
        self,
        request: EvidenceRequest,
    ) -> tuple[tuple[AcceptedRelationship, ...], list[str], str]:
        if (
            "accepted_relationships" not in request.capabilities
            or not request.anchors
            or request.max_relationship_hops == 0
        ):
            return (), [], self._knowledge_watermark()
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_identity_relationship_projection
                ORDER BY relationship_id
                """
            ).fetchall()
        watermark = (
            str(rows[0]["input_watermark"])
            if rows
            else self._knowledge_watermark()
        )
        as_of = _parse_time(request.as_of, field_name="as_of")
        frontier = {
            (anchor.anchor_type, anchor.anchor_id)
            for anchor in request.anchors
        }
        visited_entities = set(frontier)
        visited_relationships: set[str] = set()
        warnings: list[str] = []
        included: list[AcceptedRelationship] = []
        consumed_characters = 0
        for _hop in range(request.max_relationship_hops):
            next_frontier: set[tuple[str, str]] = set()
            for row in rows:
                relationship_id = str(row["relationship_id"])
                if relationship_id in visited_relationships:
                    continue
                subject_key = (
                    str(row["subject_type"]),
                    str(row["subject_id"]),
                )
                object_key = (
                    str(row["object_type"]),
                    str(row["object_id"]),
                )
                if subject_key not in frontier and object_key not in frontier:
                    continue
                if str(row["status"]) not in ACCEPTED_RELATIONSHIP_STATUSES:
                    continue
                metadata = self._json_object(row["metadata_json"])
                originating_conversation_id = str(
                    metadata.get("originating_conversation_id") or ""
                )
                if originating_conversation_id == request.conversation_id:
                    warnings.append("current_conversation_relationship_excluded")
                    continue
                starts_at = str(row["starts_at"] or "")
                ends_at = str(row["ends_at"] or "")
                accepted_at = str(metadata.get("accepted_at") or "")
                if (
                    starts_at
                    and _parse_time(starts_at, field_name="starts_at") > as_of
                ):
                    warnings.append("relationship_after_as_of_excluded")
                    continue
                if (
                    ends_at
                    and _parse_time(ends_at, field_name="ends_at") <= as_of
                ):
                    warnings.append(
                        "relationship_outside_effective_time_excluded"
                    )
                    continue
                if request.hindsight_policy != "allow_hindsight":
                    if not accepted_at:
                        warnings.append("relationship_acceptance_time_missing")
                        continue
                    if _parse_time(accepted_at, field_name="accepted_at") > as_of:
                        warnings.append("relationship_after_as_of_excluded")
                        continue
                relationship = AcceptedRelationship(
                    relationship_id=relationship_id,
                    relationship_type=str(row["relationship_type"]),
                    subject_type=subject_key[0],
                    subject_id=subject_key[1],
                    object_type=object_key[0],
                    object_id=object_key[1],
                    directionality=str(row["directionality"]),
                    starts_at=starts_at,
                    ends_at=ends_at,
                    status=str(row["status"]),
                    evidence_ids=tuple(
                        self._json_list(row["evidence_ids_json"])
                    ),
                    originating_conversation_id=originating_conversation_id,
                    accepted_at=accepted_at,
                    input_watermark=str(row["input_watermark"]),
                    metadata=metadata,
                )
                relationship_size = len(_canonical_json(asdict(relationship)))
                if (
                    len(included) >= request.max_records
                    or consumed_characters + relationship_size
                    > request.max_characters
                ):
                    warnings.append("evidence_budget_exhausted")
                    continue
                included.append(relationship)
                consumed_characters += relationship_size
                visited_relationships.add(relationship_id)
                for entity_key in (subject_key, object_key):
                    if entity_key not in visited_entities:
                        next_frontier.add(entity_key)
            if not next_frontier:
                break
            visited_entities.update(next_frontier)
            frontier = next_frontier
        return (
            tuple(sorted(included, key=lambda item: item.relationship_id)),
            warnings,
            watermark,
        )

    def _accepted_activities(
        self,
        request: EvidenceRequest,
    ) -> tuple[tuple[AcceptedActivity, ...], list[str], str]:
        if "accepted_activity_history" not in request.capabilities or not request.anchors:
            return (), [], self._knowledge_watermark()
        with transcript_store.connect(self.root) as con:
            table = con.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'knowledge_identity_activity_projection'
                """
            ).fetchone()
            if table is None:
                return (), ["accepted_activity_projection_unavailable"], self._knowledge_watermark()
            rows = con.execute(
                """
                SELECT * FROM knowledge_identity_activity_projection
                ORDER BY occurred_at DESC, observation_id
                """
            ).fetchall()
        watermark = str(rows[0]["input_watermark"]) if rows else self._knowledge_watermark()
        anchor_keys = {(anchor.anchor_type, anchor.anchor_id) for anchor in request.anchors}
        scope_keys = {
            (scope.source_profile_id, scope.account_id, scope.tenant_id)
            for scope in request.scopes
        }
        as_of = _parse_time(request.as_of, field_name="as_of")
        included: list[AcceptedActivity] = []
        warnings: list[str] = []
        consumed_characters = 0
        for row in rows:
            if (str(row["subject_type"]), str(row["subject_id"])) not in anchor_keys:
                continue
            if str(row["evidence_status"]) not in ACCEPTED_ACTIVITY_STATUSES:
                continue
            if (
                str(row["source_profile_id"]),
                str(row["account_id"]),
                str(row["tenant_id"]),
            ) not in scope_keys:
                continue
            if str(row["freshness_state"]) not in request.allowed_freshness_states:
                continue
            source_event_at = str(row["source_event_at"] or row["occurred_at"])
            if _parse_time(source_event_at, field_name="source_event_at") > as_of:
                warnings.append("activity_after_as_of_excluded")
                continue
            valid_from = str(row["valid_from"] or "")
            valid_to = str(row["valid_to"] or "")
            if valid_from and _parse_time(valid_from, field_name="valid_from") > as_of:
                warnings.append("activity_outside_effective_time_excluded")
                continue
            if valid_to and _parse_time(valid_to, field_name="valid_to") <= as_of:
                warnings.append("activity_outside_effective_time_excluded")
                continue
            metadata = self._json_object(row["metadata_json"])
            originating_conversation_id = str(
                metadata.get("originating_conversation_id") or ""
            )
            if originating_conversation_id == request.conversation_id:
                warnings.append("current_conversation_activity_excluded")
                continue
            accepted_at = str(metadata.get("accepted_at") or "")
            if request.hindsight_policy != "allow_hindsight":
                if not accepted_at:
                    warnings.append("activity_acceptance_time_missing")
                    continue
                if _parse_time(accepted_at, field_name="accepted_at") > as_of:
                    warnings.append("activity_after_as_of_excluded")
                    continue
            activity = AcceptedActivity(
                observation_id=str(row["observation_id"]),
                subject_type=str(row["subject_type"]),
                subject_id=str(row["subject_id"]),
                channel=str(row["channel"]),
                occurred_at=str(row["occurred_at"]),
                participation_status=str(row["participation_status"]),
                evidence_status=str(row["evidence_status"]),
                source_profile_id=str(row["source_profile_id"]),
                account_id=str(row["account_id"]),
                tenant_id=str(row["tenant_id"]),
                source_record_id=str(row["source_record_id"]),
                independence_group_id=str(row["independence_group_id"]),
                source_locator=self._json_object(row["source_locator_json"]),
                originating_conversation_id=originating_conversation_id,
                accepted_at=accepted_at,
                input_watermark=str(row["input_watermark"]),
                metadata=metadata,
            )
            activity_size = len(_canonical_json(asdict(activity)))
            if (
                len(included) >= request.max_records
                or consumed_characters + activity_size > request.max_characters
            ):
                warnings.append("evidence_budget_exhausted")
                continue
            included.append(activity)
            consumed_characters += activity_size
        return (
            tuple(sorted(included, key=lambda item: (item.occurred_at, item.observation_id), reverse=True)),
            warnings,
            watermark,
        )

    def _knowledge_watermark(self) -> str:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT input_watermark
                FROM knowledge_identity_people_projection
                ORDER BY person_id
                LIMIT 1
                """
            ).fetchone()
        return str(row["input_watermark"]) if row is not None else "empty"

    @staticmethod
    def _json_list(value: object) -> list[str]:
        try:
            parsed = json.loads(str(value or "[]"))
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item) for item in parsed]

    @staticmethod
    def _json_object(value: object) -> dict[str, Any]:
        try:
            parsed = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
