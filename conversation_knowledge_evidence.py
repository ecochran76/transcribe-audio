from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID

import transcript_store
from conversation_knowledge_store import ConversationKnowledgeStore


MAX_EVIDENCE_SNIPPET_CHARS = 4_000
MAX_EVIDENCE_METADATA_CHARS = 8_000
EVIDENCE_SCHEMA_VERSION = 2
_TEMPORAL_CLASSES = {
    "contemporaneous",
    "later_retrieved",
    "hindsight",
}
_HINDSIGHT_CLASSES = {
    "exclude": ("contemporaneous",),
    "allow_later_retrieved": (
        "contemporaneous",
        "later_retrieved",
    ),
    "allow_hindsight": (
        "contemporaneous",
        "later_retrieved",
        "hindsight",
    ),
}
_RAW_PROVIDER_BODY_KEYS = {
    "body",
    "content",
    "full_body",
    "full_content",
    "full_text",
    "message_body",
    "raw",
    "raw_body",
    "raw_content",
}


@dataclass(frozen=True)
class EvidenceScope:
    source_profile_id: str
    account_id: str
    tenant_id: str


@dataclass(frozen=True)
class EvidenceSnapshotRecord:
    evidence_id: str
    source_record_id: str
    source_profile_id: str
    provider_kind: str
    account_id: str
    tenant_id: str
    source_type: str
    capability: str
    snippet: str
    structured_metadata: dict[str, Any]
    source_event_at: str
    observed_at: str
    retrieved_at: str
    temporal_class: str
    source_uri: str
    content_hash: str
    independence_group_id: str
    freshness_state: str
    expires_at: str = ""
    redaction: dict[str, Any] = field(default_factory=dict)
    truncation: dict[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] = ()
    embedding_provider: str = ""
    embedding_model: str = ""


@dataclass(frozen=True)
class ExternalIdentityMatch:
    person_id: str
    source_record_id: str
    source_profile_id: str
    account_id: str
    tenant_id: str
    identity_kind: str
    normalized_value: str
    authority: str
    verified: bool


@dataclass(frozen=True)
class ScopedSourceRecord:
    person_id: str
    source_record_id: str
    source_profile_id: str
    account_id: str
    tenant_id: str


@dataclass(frozen=True)
class ConceptRecord:
    concept_id: str
    concept_type: str
    normalized_value: str
    display_value: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptMentionRecord:
    mention_id: str
    concept_id: str
    conversation_id: str
    utterance_id: str
    evidence_snapshot_id: str
    person_id: str
    observed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRequestRecord:
    request_id: str
    conversation_id: str
    recording_ids: tuple[str, ...]
    speaker_labels: tuple[str, ...]
    clue_ids: tuple[str, ...]
    conversation_at: str
    as_of: str
    prepared_person_ids: tuple[str, ...]
    scopes: tuple[EvidenceScope, ...]
    capabilities: tuple[str, ...]
    budgets: dict[str, Any]
    freshness_policy: str
    hindsight_policy: str
    retrieval_version: str
    ranking_version: str
    requesting_workflow: str
    run_id: str
    created_at: str


@dataclass(frozen=True)
class EvidenceBundleItem:
    evidence_id: str
    disposition: str
    reason_code: str
    rank: int
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceBundleRecord:
    bundle_id: str
    request_id: str
    status: str
    items: tuple[EvidenceBundleItem, ...]
    candidate_person_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    source_failures: tuple[dict[str, Any], ...]
    allowlists: dict[str, Any]
    content_hash: str
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        request_id: str,
        status: str,
        items: tuple[EvidenceBundleItem, ...],
        candidate_person_ids: tuple[str, ...],
        warnings: tuple[str, ...],
        source_failures: tuple[dict[str, Any], ...],
        allowlists: dict[str, Any],
        created_at: str,
    ) -> EvidenceBundleRecord:
        provisional = cls(
            bundle_id=bundle_id,
            request_id=request_id,
            status=status,
            items=items,
            candidate_person_ids=candidate_person_ids,
            warnings=warnings,
            source_failures=source_failures,
            allowlists=allowlists,
            content_hash="",
            created_at=created_at,
        )
        return replace(
            provisional,
            content_hash=_canonical_hash(_bundle_hash_payload(provisional)),
        )


def _uuid(value: str, *, field_name: str) -> str:
    try:
        return str(UUID(str(value or "").strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a durable opaque UUID.") from exc


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _json_list(value: str) -> list[Any]:
    loaded = json.loads(value or "[]")
    return loaded if isinstance(loaded, list) else []


def _canonical_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _scope_payload(scope: EvidenceScope) -> dict[str, str]:
    return {
        "source_profile_id": scope.source_profile_id,
        "account_id": scope.account_id,
        "tenant_id": scope.tenant_id,
    }


def _request_hash_payload(request: RetrievalRequestRecord) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "conversation_id": request.conversation_id,
        "recording_ids": list(request.recording_ids),
        "speaker_labels": list(request.speaker_labels),
        "clue_ids": list(request.clue_ids),
        "conversation_at": request.conversation_at,
        "as_of": request.as_of,
        "prepared_person_ids": list(request.prepared_person_ids),
        "scopes": [_scope_payload(scope) for scope in request.scopes],
        "capabilities": list(request.capabilities),
        "budgets": request.budgets,
        "freshness_policy": request.freshness_policy,
        "hindsight_policy": request.hindsight_policy,
        "retrieval_version": request.retrieval_version,
        "ranking_version": request.ranking_version,
        "requesting_workflow": request.requesting_workflow,
        "run_id": request.run_id,
        "created_at": request.created_at,
    }


def _bundle_hash_payload(bundle: EvidenceBundleRecord) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "request_id": bundle.request_id,
        "status": bundle.status,
        "items": [
            {
                "evidence_id": item.evidence_id,
                "disposition": item.disposition,
                "reason_code": item.reason_code,
                "rank": item.rank,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in bundle.items
        ],
        "candidate_person_ids": list(bundle.candidate_person_ids),
        "warnings": list(bundle.warnings),
        "source_failures": list(bundle.source_failures),
        "allowlists": bundle.allowlists,
        "created_at": bundle.created_at,
    }


class ConversationEvidenceRepository:
    """Persist and query bounded evidence behind explicit isolation scopes."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        status = ConversationKnowledgeStore(self.root).schema_status()
        if status.schema_version < EVIDENCE_SCHEMA_VERSION or status.dirty:
            raise RuntimeError(
                "Conversation evidence schema version 2 is not initialized."
            )

    def save_snapshot(self, snapshot: EvidenceSnapshotRecord) -> str:
        """Append one immutable, bounded evidence snapshot."""
        self._validate_snapshot(snapshot)
        existing = self.load_snapshot(snapshot.evidence_id)
        if existing is not None:
            if existing != snapshot:
                raise ValueError(
                    f"Evidence snapshot is immutable: {snapshot.evidence_id}."
                )
            return "unchanged"
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_independence_groups (
                        id, group_key, metadata_json, created_at
                    )
                    VALUES (?, ?, '{}', ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        snapshot.independence_group_id,
                        snapshot.independence_group_id,
                        snapshot.observed_at,
                    ),
                )
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_snapshots (
                        id, source_record_id, source_profile_id, provider_kind,
                        account_id, tenant_id, source_type, capability, snippet,
                        structured_metadata_json, source_event_at, observed_at,
                        retrieved_at, expires_at, temporal_class, source_uri,
                        content_hash, redaction_json, truncation_json,
                        independence_group_id, freshness_state, embedding_json,
                        embedding_provider, embedding_model, created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        snapshot.evidence_id,
                        snapshot.source_record_id or None,
                        snapshot.source_profile_id,
                        snapshot.provider_kind,
                        snapshot.account_id,
                        snapshot.tenant_id,
                        snapshot.source_type,
                        snapshot.capability,
                        snapshot.snippet,
                        _json_dumps(snapshot.structured_metadata),
                        snapshot.source_event_at,
                        snapshot.observed_at,
                        snapshot.retrieved_at,
                        snapshot.expires_at,
                        snapshot.temporal_class,
                        snapshot.source_uri,
                        snapshot.content_hash,
                        _json_dumps(snapshot.redaction),
                        _json_dumps(snapshot.truncation),
                        snapshot.independence_group_id,
                        snapshot.freshness_state,
                        _json_dumps(list(snapshot.embedding)),
                        snapshot.embedding_provider,
                        snapshot.embedding_model,
                        snapshot.observed_at,
                    ),
                )
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_fts (
                        evidence_id, search_text
                    )
                    VALUES (?, ?)
                    """,
                    (
                        snapshot.evidence_id,
                        " ".join(
                            (
                                snapshot.snippet,
                                _json_dumps(snapshot.structured_metadata),
                                snapshot.source_type,
                                snapshot.capability,
                            )
                        ),
                    ),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return "inserted"

    def load_snapshot(
        self,
        evidence_id: str,
    ) -> EvidenceSnapshotRecord | None:
        evidence_id = _uuid(evidence_id, field_name="evidence_id")
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT *
                FROM knowledge_evidence_snapshots
                WHERE id = ?
                """,
                (evidence_id,),
            ).fetchone()
        return self._snapshot_from_row(row) if row is not None else None

    def search_snapshots(
        self,
        query: str,
        *,
        scopes: tuple[EvidenceScope, ...],
        capabilities: tuple[str, ...],
        as_of: str,
        hindsight_policy: str,
        limit: int = 50,
    ) -> tuple[EvidenceSnapshotRecord, ...]:
        """Search bounded lexical evidence within exact source scopes."""
        self._validate_query(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        fts_query = transcript_store.fts_query(query)
        if not fts_query:
            return ()
        where, parameters = self._query_filter(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                f"""
                SELECT snapshot.*
                FROM knowledge_evidence_fts
                JOIN knowledge_evidence_snapshots AS snapshot
                  ON snapshot.id = knowledge_evidence_fts.evidence_id
                WHERE knowledge_evidence_fts MATCH ? AND {where}
                ORDER BY bm25(knowledge_evidence_fts), snapshot.id
                LIMIT ?
                """,
                (fts_query, *parameters, max(1, min(limit, 500))),
            ).fetchall()
        return tuple(self._snapshot_from_row(row) for row in rows)

    def semantic_snapshots(
        self,
        query_embedding: tuple[float, ...],
        *,
        scopes: tuple[EvidenceScope, ...],
        capabilities: tuple[str, ...],
        as_of: str,
        hindsight_policy: str,
        limit: int = 50,
    ) -> tuple[EvidenceSnapshotRecord, ...]:
        """Rank stored bounded evidence vectors within exact source scopes."""
        self._validate_query(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        if not query_embedding:
            return ()
        where, parameters = self._query_filter(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                f"""
                SELECT snapshot.*
                FROM knowledge_evidence_snapshots AS snapshot
                WHERE {where} AND embedding_json != '[]'
                """,
                parameters,
            ).fetchall()
        ranked = sorted(
            (
                (
                    transcript_store.cosine(
                        list(query_embedding),
                        [
                            float(value)
                            for value in _json_list(str(row["embedding_json"]))
                        ],
                    ),
                    self._snapshot_from_row(row),
                )
                for row in rows
                if len(_json_list(str(row["embedding_json"])))
                == len(query_embedding)
            ),
            key=lambda item: (-item[0], item[1].evidence_id),
        )
        return tuple(item[1] for item in ranked[: max(1, min(limit, 500))])

    def scoped_snapshots(
        self,
        *,
        scopes: tuple[EvidenceScope, ...],
        capabilities: tuple[str, ...],
        as_of: str,
        hindsight_policy: str,
        source_record_ids: tuple[str, ...] = (),
        limit: int = 500,
    ) -> tuple[EvidenceSnapshotRecord, ...]:
        """Load bounded evidence within exact scopes, optionally by source."""
        self._validate_query(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        where, parameters = self._query_filter(
            scopes=scopes,
            capabilities=capabilities,
            as_of=as_of,
            hindsight_policy=hindsight_policy,
        )
        source_clause = ""
        if source_record_ids:
            placeholders = ",".join("?" for _ in source_record_ids)
            source_clause = (
                f" AND snapshot.source_record_id IN ({placeholders})"
            )
            parameters.extend(source_record_ids)
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                f"""
                SELECT snapshot.*
                FROM knowledge_evidence_snapshots AS snapshot
                WHERE {where}{source_clause}
                ORDER BY snapshot.observed_at, snapshot.id
                LIMIT ?
                """,
                (*parameters, max(1, min(limit, 2_000))),
            ).fetchall()
        return tuple(self._snapshot_from_row(row) for row in rows)

    def find_people_by_external_identity(
        self,
        identity_kind: str,
        value: str,
        *,
        scopes: tuple[EvidenceScope, ...],
    ) -> tuple[ExternalIdentityMatch, ...]:
        """Resolve exact identifiers without erasing their source scopes."""
        if not identity_kind or not value or not scopes:
            raise ValueError("Exact identity lookup requires a value and scopes.")
        scope_sql, scope_parameters = self._scope_filter(scopes)
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                f"""
                SELECT identity.*, source.source_profile_id,
                       source.account_id, source.tenant_id
                FROM knowledge_external_identities AS identity
                JOIN knowledge_source_records AS source
                  ON source.id = identity.source_record_id
                WHERE identity.identity_kind = ?
                  AND identity.normalized_value = ?
                  AND ({scope_sql})
                ORDER BY identity.verified DESC, identity.authority, identity.id
                """,
                (
                    identity_kind.strip().casefold(),
                    value.strip().casefold(),
                    *scope_parameters,
                ),
            ).fetchall()
        return tuple(
            ExternalIdentityMatch(
                person_id=str(row["person_id"]),
                source_record_id=str(row["source_record_id"]),
                source_profile_id=str(row["source_profile_id"]),
                account_id=str(row["account_id"]),
                tenant_id=str(row["tenant_id"]),
                identity_kind=str(row["identity_kind"]),
                normalized_value=str(row["normalized_value"]),
                authority=str(row["authority"]),
                verified=bool(row["verified"]),
            )
            for row in rows
        )

    def source_records_for_people(
        self,
        person_ids: tuple[str, ...],
        *,
        scopes: tuple[EvidenceScope, ...],
    ) -> tuple[ScopedSourceRecord, ...]:
        """Return every permitted source affinity for selected people."""
        if not person_ids or not scopes:
            return ()
        for person_id in person_ids:
            _uuid(person_id, field_name="person_id")
        scope_sql, scope_parameters = self._scope_filter(scopes)
        placeholders = ",".join("?" for _ in person_ids)
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                f"""
                SELECT person_id, id, source_profile_id, account_id, tenant_id
                FROM knowledge_source_records AS source
                WHERE person_id IN ({placeholders})
                  AND ({scope_sql})
                ORDER BY person_id, source_profile_id, id
                """,
                (*person_ids, *scope_parameters),
            ).fetchall()
        return tuple(
            ScopedSourceRecord(
                person_id=str(row["person_id"]),
                source_record_id=str(row["id"]),
                source_profile_id=str(row["source_profile_id"]),
                account_id=str(row["account_id"]),
                tenant_id=str(row["tenant_id"]),
            )
            for row in rows
        )

    def save_concept(
        self,
        concept: ConceptRecord,
        *,
        mentions: tuple[ConceptMentionRecord, ...] = (),
    ) -> str:
        """Store one typed concept and immutable bounded mentions."""
        _uuid(concept.concept_id, field_name="concept_id")
        if not all(
            (
                concept.concept_type,
                concept.normalized_value,
                concept.display_value,
            )
        ):
            raise ValueError("Concept type and values are required.")
        now = mentions[0].observed_at if mentions else ""
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT * FROM knowledge_concepts WHERE id = ?",
                (concept.concept_id,),
            ).fetchone()
            if existing is not None:
                loaded = ConceptRecord(
                    concept_id=str(existing["id"]),
                    concept_type=str(existing["concept_type"]),
                    normalized_value=str(existing["normalized_value"]),
                    display_value=str(existing["display_value"]),
                    metadata=_json_object(str(existing["metadata_json"])),
                )
                if loaded != concept:
                    raise ValueError(
                        f"Concept identity is immutable: {concept.concept_id}."
                    )
                status = "unchanged"
            else:
                status = "inserted"
            con.execute("BEGIN IMMEDIATE")
            try:
                if existing is None:
                    con.execute(
                        """
                        INSERT INTO knowledge_concepts (
                            id, concept_type, normalized_value, display_value,
                            metadata_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            concept.concept_id,
                            concept.concept_type,
                            concept.normalized_value,
                            concept.display_value,
                            _json_dumps(concept.metadata),
                            now,
                            now,
                        ),
                    )
                    con.execute(
                        """
                        INSERT INTO knowledge_concept_fts (
                            concept_id, search_text
                        )
                        VALUES (?, ?)
                        """,
                        (
                            concept.concept_id,
                            " ".join(
                                (
                                    concept.concept_type,
                                    concept.normalized_value,
                                    concept.display_value,
                                    _json_dumps(concept.metadata),
                                )
                            ),
                        ),
                    )
                for mention in mentions:
                    self._validate_mention(mention, concept.concept_id)
                    prior = con.execute(
                        """
                        SELECT *
                        FROM knowledge_concept_mentions
                        WHERE id = ?
                        """,
                        (mention.mention_id,),
                    ).fetchone()
                    if prior is not None:
                        loaded_mention = ConceptMentionRecord(
                            mention_id=str(prior["id"]),
                            concept_id=str(prior["concept_id"]),
                            conversation_id=str(
                                prior["conversation_id"] or ""
                            ),
                            utterance_id=str(prior["utterance_id"] or ""),
                            evidence_snapshot_id=str(
                                prior["evidence_snapshot_id"]
                            ),
                            person_id=str(prior["person_id"] or ""),
                            observed_at=str(prior["observed_at"]),
                            metadata=_json_object(
                                str(prior["metadata_json"])
                            ),
                        )
                        if loaded_mention != mention:
                            raise ValueError(
                                "Concept mention is immutable: "
                                f"{mention.mention_id}."
                            )
                        continue
                    con.execute(
                        """
                        INSERT INTO knowledge_concept_mentions (
                            id, concept_id, conversation_id, utterance_id,
                            evidence_snapshot_id, person_id, observed_at,
                            metadata_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            mention.mention_id,
                            mention.concept_id,
                            mention.conversation_id or None,
                            mention.utterance_id or None,
                            mention.evidence_snapshot_id,
                            mention.person_id or None,
                            mention.observed_at,
                            _json_dumps(mention.metadata),
                            mention.observed_at,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return status

    def search_concepts(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> tuple[ConceptRecord, ...]:
        fts_query = transcript_store.fts_query(query)
        if not fts_query:
            return ()
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT concept.*
                FROM knowledge_concept_fts
                JOIN knowledge_concepts AS concept
                  ON concept.id = knowledge_concept_fts.concept_id
                WHERE knowledge_concept_fts MATCH ?
                ORDER BY bm25(knowledge_concept_fts), concept.id
                LIMIT ?
                """,
                (fts_query, max(1, min(limit, 500))),
            ).fetchall()
        return tuple(
            ConceptRecord(
                concept_id=str(row["id"]),
                concept_type=str(row["concept_type"]),
                normalized_value=str(row["normalized_value"]),
                display_value=str(row["display_value"]),
                metadata=_json_object(str(row["metadata_json"])),
            )
            for row in rows
        )

    def save_retrieval_request(
        self,
        request: RetrievalRequestRecord,
    ) -> str:
        """Append an immutable, replayable retrieval request."""
        self._validate_request(request)
        existing = self.load_retrieval_request(request.request_id)
        if existing is not None:
            if existing != request:
                raise ValueError(
                    f"Retrieval request is immutable: {request.request_id}."
                )
            return "unchanged"
        payload = _request_hash_payload(request)
        with transcript_store.connect(self.root) as con:
            con.execute(
                """
                INSERT INTO knowledge_retrieval_requests (
                    id, conversation_id, recording_ids_json,
                    speaker_labels_json, clue_ids_json, conversation_at,
                    as_of, prepared_person_ids_json, scopes_json,
                    capabilities_json, budgets_json, freshness_policy,
                    hindsight_policy, retrieval_version, ranking_version,
                    requesting_workflow, run_id, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.conversation_id,
                    _json_dumps(list(request.recording_ids)),
                    _json_dumps(list(request.speaker_labels)),
                    _json_dumps(list(request.clue_ids)),
                    request.conversation_at,
                    request.as_of,
                    _json_dumps(list(request.prepared_person_ids)),
                    _json_dumps(
                        [_scope_payload(scope) for scope in request.scopes]
                    ),
                    _json_dumps(list(request.capabilities)),
                    _json_dumps(request.budgets),
                    request.freshness_policy,
                    request.hindsight_policy,
                    request.retrieval_version,
                    request.ranking_version,
                    request.requesting_workflow,
                    request.run_id,
                    _canonical_hash(payload),
                    request.created_at,
                ),
            )
            con.commit()
        return "inserted"

    def load_retrieval_request(
        self,
        request_id: str,
    ) -> RetrievalRequestRecord | None:
        request_id = _uuid(request_id, field_name="request_id")
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT *
                FROM knowledge_retrieval_requests
                WHERE id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        scopes = tuple(
            EvidenceScope(
                source_profile_id=str(item.get("source_profile_id") or ""),
                account_id=str(item.get("account_id") or ""),
                tenant_id=str(item.get("tenant_id") or ""),
            )
            for item in _json_list(str(row["scopes_json"]))
            if isinstance(item, dict)
        )
        request = RetrievalRequestRecord(
            request_id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            recording_ids=tuple(
                str(item)
                for item in _json_list(str(row["recording_ids_json"]))
            ),
            speaker_labels=tuple(
                str(item)
                for item in _json_list(str(row["speaker_labels_json"]))
            ),
            clue_ids=tuple(
                str(item) for item in _json_list(str(row["clue_ids_json"]))
            ),
            conversation_at=str(row["conversation_at"]),
            as_of=str(row["as_of"]),
            prepared_person_ids=tuple(
                str(item)
                for item in _json_list(str(row["prepared_person_ids_json"]))
            ),
            scopes=scopes,
            capabilities=tuple(
                str(item)
                for item in _json_list(str(row["capabilities_json"]))
            ),
            budgets=_json_object(str(row["budgets_json"])),
            freshness_policy=str(row["freshness_policy"]),
            hindsight_policy=str(row["hindsight_policy"]),
            retrieval_version=str(row["retrieval_version"]),
            ranking_version=str(row["ranking_version"]),
            requesting_workflow=str(row["requesting_workflow"]),
            run_id=str(row["run_id"]),
            created_at=str(row["created_at"]),
        )
        if str(row["content_hash"]) != _canonical_hash(
            _request_hash_payload(request)
        ):
            raise ValueError(
                f"Retrieval request content hash is invalid: {request_id}."
            )
        return request

    def save_bundle(self, bundle: EvidenceBundleRecord) -> str:
        """Append an immutable content-hashed evidence bundle."""
        existing = self.load_bundle(bundle.bundle_id)
        if existing is not None:
            if existing != bundle:
                raise ValueError(
                    f"Evidence bundle is immutable: {bundle.bundle_id}."
                )
            return "unchanged"
        self._validate_bundle(bundle)
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_evidence_bundles (
                        id, request_id, status, candidate_person_ids_json,
                        warnings_json, source_failures_json, allowlists_json,
                        content_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.bundle_id,
                        bundle.request_id,
                        bundle.status,
                        _json_dumps(list(bundle.candidate_person_ids)),
                        _json_dumps(list(bundle.warnings)),
                        _json_dumps(list(bundle.source_failures)),
                        _json_dumps(bundle.allowlists),
                        bundle.content_hash,
                        bundle.created_at,
                    ),
                )
                for item in bundle.items:
                    con.execute(
                        """
                        INSERT INTO knowledge_evidence_bundle_items (
                            bundle_id, evidence_id, disposition, reason_code,
                            rank, score, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            bundle.bundle_id,
                            item.evidence_id,
                            item.disposition,
                            item.reason_code,
                            item.rank,
                            item.score,
                            _json_dumps(item.metadata),
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return "inserted"

    def load_bundle(
        self,
        bundle_id: str,
    ) -> EvidenceBundleRecord | None:
        bundle_id = _uuid(bundle_id, field_name="bundle_id")
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT *
                FROM knowledge_evidence_bundles
                WHERE id = ?
                """,
                (bundle_id,),
            ).fetchone()
            if row is None:
                return None
            item_rows = con.execute(
                """
                SELECT *
                FROM knowledge_evidence_bundle_items
                WHERE bundle_id = ?
                ORDER BY
                    CASE disposition
                        WHEN 'included' THEN 0
                        ELSE 1
                    END,
                    rank,
                    evidence_id
                """,
                (bundle_id,),
            ).fetchall()
        bundle = EvidenceBundleRecord(
            bundle_id=str(row["id"]),
            request_id=str(row["request_id"]),
            status=str(row["status"]),
            items=tuple(
                EvidenceBundleItem(
                    evidence_id=str(item["evidence_id"]),
                    disposition=str(item["disposition"]),
                    reason_code=str(item["reason_code"]),
                    rank=int(item["rank"]),
                    score=(
                        float(item["score"])
                        if item["score"] is not None
                        else None
                    ),
                    metadata=_json_object(str(item["metadata_json"])),
                )
                for item in item_rows
            ),
            candidate_person_ids=tuple(
                str(item)
                for item in _json_list(
                    str(row["candidate_person_ids_json"])
                )
            ),
            warnings=tuple(
                str(item) for item in _json_list(str(row["warnings_json"]))
            ),
            source_failures=tuple(
                dict(item)
                for item in _json_list(str(row["source_failures_json"]))
                if isinstance(item, dict)
            ),
            allowlists=_json_object(str(row["allowlists_json"])),
            content_hash=str(row["content_hash"]),
            created_at=str(row["created_at"]),
        )
        if bundle.content_hash != _canonical_hash(
            _bundle_hash_payload(bundle)
        ):
            raise ValueError(
                f"Evidence bundle content hash is invalid: {bundle_id}."
            )
        return bundle

    @staticmethod
    def _validate_snapshot(snapshot: EvidenceSnapshotRecord) -> None:
        _uuid(snapshot.evidence_id, field_name="evidence_id")
        if not all(
            (
                snapshot.source_profile_id,
                snapshot.provider_kind,
                snapshot.source_type,
                snapshot.capability,
                snapshot.observed_at,
                snapshot.temporal_class,
                snapshot.content_hash,
                snapshot.independence_group_id,
                snapshot.freshness_state,
            )
        ):
            raise ValueError("Evidence source, scope, time, and hash are required.")
        if snapshot.temporal_class not in _TEMPORAL_CLASSES:
            raise ValueError("Unsupported evidence temporal class.")
        if len(snapshot.snippet) > MAX_EVIDENCE_SNIPPET_CHARS:
            raise ValueError("Evidence bounded snippet exceeds its character cap.")
        if (
            len(_json_dumps(snapshot.structured_metadata))
            > MAX_EVIDENCE_METADATA_CHARS
        ):
            raise ValueError(
                "Evidence structured metadata exceeds its character cap."
            )
        risky_keys = {
            str(key).strip().casefold()
            for key in snapshot.structured_metadata
        } & _RAW_PROVIDER_BODY_KEYS
        if risky_keys:
            raise ValueError(
                "Evidence structured metadata cannot contain raw provider "
                "body fields; use the bounded snippet."
            )
        for vector_value in snapshot.embedding:
            if not isinstance(vector_value, (int, float)):
                raise ValueError("Evidence embedding must be numeric.")

    @staticmethod
    def _validate_query(
        *,
        scopes: tuple[EvidenceScope, ...],
        capabilities: tuple[str, ...],
        as_of: str,
        hindsight_policy: str,
    ) -> None:
        if not scopes or not capabilities or not as_of:
            raise ValueError(
                "Evidence query requires scopes, capabilities, and as_of."
            )
        if hindsight_policy not in _HINDSIGHT_CLASSES:
            raise ValueError("Unsupported hindsight policy.")
        if any(not scope.source_profile_id for scope in scopes):
            raise ValueError("Every evidence scope requires a source profile.")
        if any(not capability for capability in capabilities):
            raise ValueError("Evidence capabilities cannot be blank.")

    @staticmethod
    def _scope_filter(
        scopes: tuple[EvidenceScope, ...],
        *,
        alias: str = "source",
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        parameters: list[str] = []
        for scope in scopes:
            clauses.append(
                f"({alias}.source_profile_id = ? "
                f"AND {alias}.account_id = ? "
                f"AND {alias}.tenant_id = ?)"
            )
            parameters.extend(
                (
                    scope.source_profile_id,
                    scope.account_id,
                    scope.tenant_id,
                )
            )
        return " OR ".join(clauses), parameters

    @classmethod
    def _query_filter(
        cls,
        *,
        scopes: tuple[EvidenceScope, ...],
        capabilities: tuple[str, ...],
        as_of: str,
        hindsight_policy: str,
    ) -> tuple[str, list[Any]]:
        scope_sql, parameters = cls._scope_filter(
            scopes,
            alias="snapshot",
        )
        capability_placeholders = ",".join("?" for _ in capabilities)
        temporal_classes = _HINDSIGHT_CLASSES[hindsight_policy]
        temporal_placeholders = ",".join("?" for _ in temporal_classes)
        contemporaneous_clause = (
            "(snapshot.temporal_class != 'contemporaneous' "
            "OR snapshot.observed_at <= ?)"
        )
        where = (
            f"({scope_sql}) "
            f"AND snapshot.capability IN ({capability_placeholders}) "
            f"AND snapshot.temporal_class IN ({temporal_placeholders}) "
            "AND (snapshot.source_event_at = '' "
            "OR snapshot.source_event_at <= ?) "
            f"AND {contemporaneous_clause}"
        )
        return (
            where,
            [
                *parameters,
                *capabilities,
                *temporal_classes,
                as_of,
                as_of,
            ],
        )

    @staticmethod
    def _snapshot_from_row(row: Any) -> EvidenceSnapshotRecord:
        return EvidenceSnapshotRecord(
            evidence_id=str(row["id"]),
            source_record_id=str(row["source_record_id"] or ""),
            source_profile_id=str(row["source_profile_id"]),
            provider_kind=str(row["provider_kind"]),
            account_id=str(row["account_id"]),
            tenant_id=str(row["tenant_id"]),
            source_type=str(row["source_type"]),
            capability=str(row["capability"]),
            snippet=str(row["snippet"]),
            structured_metadata=_json_object(
                str(row["structured_metadata_json"])
            ),
            source_event_at=str(row["source_event_at"]),
            observed_at=str(row["observed_at"]),
            retrieved_at=str(row["retrieved_at"]),
            expires_at=str(row["expires_at"]),
            temporal_class=str(row["temporal_class"]),
            source_uri=str(row["source_uri"]),
            content_hash=str(row["content_hash"]),
            redaction=_json_object(str(row["redaction_json"])),
            truncation=_json_object(str(row["truncation_json"])),
            independence_group_id=str(row["independence_group_id"]),
            freshness_state=str(row["freshness_state"]),
            embedding=tuple(
                float(value)
                for value in _json_list(str(row["embedding_json"]))
            ),
            embedding_provider=str(row["embedding_provider"]),
            embedding_model=str(row["embedding_model"]),
        )

    @staticmethod
    def _validate_mention(
        mention: ConceptMentionRecord,
        concept_id: str,
    ) -> None:
        _uuid(mention.mention_id, field_name="mention_id")
        if mention.concept_id != concept_id:
            raise ValueError("Concept mention belongs to another concept.")
        if not mention.observed_at:
            raise ValueError("Concept mention observed_at is required.")
        if not any(
            (
                mention.conversation_id,
                mention.utterance_id,
                mention.evidence_snapshot_id,
                mention.person_id,
            )
        ):
            raise ValueError("Concept mention requires a bounded source target.")

    @staticmethod
    def _validate_request(request: RetrievalRequestRecord) -> None:
        _uuid(request.request_id, field_name="request_id")
        _uuid(request.conversation_id, field_name="conversation_id")
        ConversationEvidenceRepository._validate_query(
            scopes=request.scopes,
            capabilities=request.capabilities,
            as_of=request.as_of,
            hindsight_policy=request.hindsight_policy,
        )
        if not all(
            (
                request.freshness_policy,
                request.retrieval_version,
                request.ranking_version,
                request.requesting_workflow,
                request.created_at,
            )
        ):
            raise ValueError("Retrieval policy, versions, workflow, and time are required.")
        if not request.budgets:
            raise ValueError("Retrieval request requires explicit budgets.")

    def _validate_bundle(self, bundle: EvidenceBundleRecord) -> None:
        _uuid(bundle.bundle_id, field_name="bundle_id")
        _uuid(bundle.request_id, field_name="request_id")
        if bundle.status not in {"complete", "partial", "failed"}:
            raise ValueError("Unsupported evidence bundle status.")
        if not bundle.created_at:
            raise ValueError("Evidence bundle created_at is required.")
        if self.load_retrieval_request(bundle.request_id) is None:
            raise ValueError("Evidence bundle references an unknown request.")
        if bundle.content_hash != _canonical_hash(
            _bundle_hash_payload(bundle)
        ):
            raise ValueError("Evidence bundle content hash is invalid.")
        evidence_ids: set[str] = set()
        for item in bundle.items:
            _uuid(item.evidence_id, field_name="evidence_id")
            if item.disposition not in {"included", "excluded"}:
                raise ValueError("Evidence bundle disposition is invalid.")
            if not item.reason_code:
                raise ValueError("Evidence bundle item requires a reason code.")
            if item.evidence_id in evidence_ids:
                raise ValueError("Evidence bundle repeats an evidence item.")
            evidence_ids.add(item.evidence_id)
            if self.load_snapshot(item.evidence_id) is None:
                raise ValueError("Evidence bundle references unknown evidence.")
        if len(_json_dumps(bundle.source_failures)) > MAX_EVIDENCE_METADATA_CHARS:
            raise ValueError("Evidence source failures exceed their bounded cap.")
