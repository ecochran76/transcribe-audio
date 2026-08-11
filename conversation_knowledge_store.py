"""Versioned user-scoped storage for conversation knowledge."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import transcript_store


LATEST_SCHEMA_VERSION = 3
AUTHORITY_MODE_SIDECAR = "sidecar"


@dataclass(frozen=True)
class KnowledgeSchemaStatus:
    schema_version: int
    authority_mode: str
    dirty: bool


@dataclass(frozen=True)
class MigrationReceipt:
    from_version: int
    to_version: int
    applied_versions: tuple[int, ...]
    rolled_back_versions: tuple[int, ...] = ()
    backup_path: str = ""


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    title: str = ""
    starts_at: str = ""
    ends_at: str = ""
    calendar_association_state: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingRecord:
    recording_id: str
    conversation_id: str
    transcript_document_id: str = ""
    source_blob_id: str = ""
    backend: str = ""
    model: str = ""
    captured_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UtteranceRecord:
    utterance_id: str
    conversation_id: str
    recording_id: str
    speaker_label: str
    ordinal: int
    start_ms: int | None
    end_ms: int | None
    text: str
    source_artifact_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationSnapshot:
    conversation: ConversationRecord
    recordings: tuple[RecordingRecord, ...] = ()
    utterances: tuple[UtteranceRecord, ...] = ()


@dataclass(frozen=True)
class PersonRecord:
    person_id: str
    status: str
    primary_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRecord:
    source_record_id: str
    person_id: str
    source_profile_id: str
    provider_kind: str
    account_id: str
    tenant_id: str
    external_ref: str
    label: str
    relationship_scope: str
    identifier_authority: str
    observed_at: str
    content_hash: str
    source_event_at: str = ""
    retrieved_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    freshness_state: str = "current"
    source_uri: str = ""
    source_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalIdentityRecord:
    external_identity_id: str
    person_id: str
    source_record_id: str
    identity_kind: str
    normalized_value: str
    display_value: str
    authority: str
    verified: bool
    valid_from: str = ""
    valid_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonSnapshot:
    person: PersonRecord
    source_records: tuple[SourceRecord, ...] = ()
    external_identities: tuple[ExternalIdentityRecord, ...] = ()


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    conversation_id: str
    evaluation_type: str
    schema_version: str
    status: str
    created_at: str
    model_profile: str = ""
    input_artifact_id: str = ""
    output_artifact_id: str = ""
    evidence_bundle_id: str = ""
    rubric_versions: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewDecisionRecord:
    decision_id: str
    evaluation_id: str
    proposal_id: str
    action: str
    reviewer: str
    method: str
    decided_at: str
    note: str = ""
    supersedes_decision_id: str = ""
    reviewer_asserted_identity: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingHistory:
    conversation_id: str
    current_evaluation_id: str
    evaluations: tuple[EvaluationRecord, ...] = ()
    review_decisions: tuple[ReviewDecisionRecord, ...] = ()


@dataclass(frozen=True)
class SaveReceipt:
    status: str
    conversation_id: str
    recording_count: int
    utterance_count: int


@dataclass(frozen=True)
class PersonSaveReceipt:
    status: str
    person_id: str
    source_record_count: int
    external_identity_count: int


@dataclass(frozen=True)
class ProcessingSaveReceipt:
    status: str
    conversation_id: str
    evaluation_count: int
    review_decision_count: int


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    observation_type: str
    subject_type: str
    subject_id: str
    source_type: str
    source_id: str
    conversation_id: str
    observed_at: str
    source_event_at: str = ""
    retrieved_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    review_state: str = "unreviewed"
    payload: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(frozen=True)
class ObservationSaveReceipt:
    status: str
    conversation_id: str
    observation_count: int


@dataclass(frozen=True)
class ProjectionStateRecord:
    projection_name: str
    scope_type: str
    scope_id: str
    schema_version: str
    input_watermark: str
    built_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def _uuid(value: str, *, field_name: str) -> str:
    try:
        return str(UUID(str(value or "").strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a durable opaque UUID.") from exc


class ConversationKnowledgeStore:
    """Own schema lifecycle and domain persistence behind one interface."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)

    def schema_status(self) -> KnowledgeSchemaStatus:
        with transcript_store.connect(self.root) as con:
            state_exists = con.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'knowledge_store_state'
                """
            ).fetchone()
            if not state_exists:
                return KnowledgeSchemaStatus(
                    schema_version=0,
                    authority_mode=AUTHORITY_MODE_SIDECAR,
                    dirty=False,
                )
            row = con.execute(
                """
                SELECT schema_version, authority_mode, dirty
                FROM knowledge_store_state
                WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                return KnowledgeSchemaStatus(
                    schema_version=0,
                    authority_mode=AUTHORITY_MODE_SIDECAR,
                    dirty=False,
                )
            return KnowledgeSchemaStatus(
                schema_version=int(row["schema_version"]),
                authority_mode=str(row["authority_mode"]),
                dirty=bool(row["dirty"]),
            )

    def migrate(
        self,
        *,
        target_version: int = LATEST_SCHEMA_VERSION,
        backup: bool = True,
    ) -> MigrationReceipt:
        """Apply forward migrations without changing processing authority."""
        before = self.schema_status()
        if target_version < before.schema_version:
            raise ValueError("Use rollback() to move to an earlier schema version.")
        if target_version > LATEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported knowledge schema version: {target_version}."
            )
        backup_path = ""
        if backup and target_version > before.schema_version:
            backup_path = str(
                self._create_backup(f"pre-migrate-v{before.schema_version}")
            )
        applied: list[int] = []
        with transcript_store.connect(self.root) as con:
            if before.schema_version < 1 <= target_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._apply_v1(con)
                    applied.append(1)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
            if before.schema_version < 2 <= target_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._apply_v2(con)
                    applied.append(2)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
            if before.schema_version < 3 <= target_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._apply_v3(con)
                    applied.append(3)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
        return MigrationReceipt(
            from_version=before.schema_version,
            to_version=target_version,
            applied_versions=tuple(applied),
            backup_path=backup_path,
        )

    def rollback(
        self,
        *,
        target_version: int,
        backup: bool = True,
    ) -> MigrationReceipt:
        """Roll back additive knowledge migrations without touching legacy tables."""
        before = self.schema_status()
        if target_version < 0 or target_version > before.schema_version:
            raise ValueError(
                f"Invalid rollback target for version {before.schema_version}: "
                f"{target_version}."
            )
        backup_path = ""
        if backup and target_version < before.schema_version:
            backup_path = str(
                self._create_backup(f"pre-rollback-v{before.schema_version}")
            )
        rolled_back: list[int] = []
        with transcript_store.connect(self.root) as con:
            if target_version < 3 <= before.schema_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._rollback_v3(con)
                    rolled_back.append(3)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
            if target_version < 2 <= before.schema_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._rollback_v2(con)
                    rolled_back.append(2)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
            if target_version < 1 <= before.schema_version:
                con.execute("BEGIN IMMEDIATE")
                try:
                    self._rollback_v1(con)
                    rolled_back.append(1)
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
        return MigrationReceipt(
            from_version=before.schema_version,
            to_version=target_version,
            applied_versions=(),
            rolled_back_versions=tuple(rolled_back),
            backup_path=backup_path,
        )

    def save_conversation_snapshot(
        self,
        snapshot: ConversationSnapshot,
    ) -> SaveReceipt:
        """Replace one complete conversation snapshot transactionally."""
        self._validate_snapshot(snapshot)
        existing = self.load_conversation_snapshot(
            snapshot.conversation.conversation_id
        )
        if existing == snapshot:
            return SaveReceipt(
                status="unchanged",
                conversation_id=snapshot.conversation.conversation_id,
                recording_count=len(snapshot.recordings),
                utterance_count=len(snapshot.utterances),
            )
        now = _utc_now()
        with transcript_store.connect(self.root) as con:
            self._require_current_schema(con)
            con.execute("BEGIN IMMEDIATE")
            try:
                conversation = snapshot.conversation
                con.execute(
                    """
                    INSERT INTO knowledge_conversations (
                        id, title, starts_at, ends_at,
                        calendar_association_state, metadata_json,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        starts_at = excluded.starts_at,
                        ends_at = excluded.ends_at,
                        calendar_association_state =
                            excluded.calendar_association_state,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        conversation.conversation_id,
                        conversation.title,
                        conversation.starts_at,
                        conversation.ends_at,
                        conversation.calendar_association_state,
                        _json_dumps(conversation.metadata),
                        now,
                        now,
                    ),
                )
                con.execute(
                    "DELETE FROM knowledge_utterances WHERE conversation_id = ?",
                    (conversation.conversation_id,),
                )
                con.execute(
                    "DELETE FROM knowledge_recordings WHERE conversation_id = ?",
                    (conversation.conversation_id,),
                )
                for recording in snapshot.recordings:
                    con.execute(
                        """
                        INSERT INTO knowledge_recordings (
                            id, conversation_id, transcript_document_id,
                            source_blob_id, backend, model, captured_at,
                            metadata_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            recording.recording_id,
                            recording.conversation_id,
                            recording.transcript_document_id,
                            recording.source_blob_id,
                            recording.backend,
                            recording.model,
                            recording.captured_at,
                            _json_dumps(recording.metadata),
                            now,
                            now,
                        ),
                    )
                for utterance in snapshot.utterances:
                    con.execute(
                        """
                        INSERT INTO knowledge_utterances (
                            id, conversation_id, recording_id, speaker_label,
                            ordinal, start_ms, end_ms, text,
                            source_artifact_id, metadata_json,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            utterance.utterance_id,
                            utterance.conversation_id,
                            utterance.recording_id,
                            utterance.speaker_label,
                            utterance.ordinal,
                            utterance.start_ms,
                            utterance.end_ms,
                            utterance.text,
                            utterance.source_artifact_id,
                            _json_dumps(utterance.metadata),
                            now,
                            now,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return SaveReceipt(
            status="updated" if existing is not None else "inserted",
            conversation_id=snapshot.conversation.conversation_id,
            recording_count=len(snapshot.recordings),
            utterance_count=len(snapshot.utterances),
        )

    def load_conversation_snapshot(
        self,
        conversation_id: str,
    ) -> ConversationSnapshot | None:
        """Load one complete conversation snapshot."""
        conversation_id = (
            _uuid(conversation_id, field_name="conversation_id")
            if conversation_id
            else ""
        )
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return None
            row = con.execute(
                """
                SELECT *
                FROM knowledge_conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            recording_rows = con.execute(
                """
                SELECT *
                FROM knowledge_recordings
                WHERE conversation_id = ?
                ORDER BY captured_at, id
                """,
                (conversation_id,),
            ).fetchall()
            utterance_rows = con.execute(
                """
                SELECT *
                FROM knowledge_utterances
                WHERE conversation_id = ?
                ORDER BY recording_id, ordinal, id
                """,
                (conversation_id,),
            ).fetchall()
        return ConversationSnapshot(
            conversation=ConversationRecord(
                conversation_id=str(row["id"]),
                title=str(row["title"]),
                starts_at=str(row["starts_at"]),
                ends_at=str(row["ends_at"]),
                calendar_association_state=str(
                    row["calendar_association_state"]
                ),
                metadata=_json_object(str(row["metadata_json"])),
            ),
            recordings=tuple(
                RecordingRecord(
                    recording_id=str(recording["id"]),
                    conversation_id=str(recording["conversation_id"]),
                    transcript_document_id=str(
                        recording["transcript_document_id"]
                    ),
                    source_blob_id=str(recording["source_blob_id"]),
                    backend=str(recording["backend"]),
                    model=str(recording["model"]),
                    captured_at=str(recording["captured_at"]),
                    metadata=_json_object(str(recording["metadata_json"])),
                )
                for recording in recording_rows
            ),
            utterances=tuple(
                UtteranceRecord(
                    utterance_id=str(utterance["id"]),
                    conversation_id=str(utterance["conversation_id"]),
                    recording_id=str(utterance["recording_id"]),
                    speaker_label=str(utterance["speaker_label"]),
                    ordinal=int(utterance["ordinal"]),
                    start_ms=(
                        int(utterance["start_ms"])
                        if utterance["start_ms"] is not None
                        else None
                    ),
                    end_ms=(
                        int(utterance["end_ms"])
                        if utterance["end_ms"] is not None
                        else None
                    ),
                    text=str(utterance["text"]),
                    source_artifact_id=str(utterance["source_artifact_id"]),
                    metadata=_json_object(str(utterance["metadata_json"])),
                )
                for utterance in utterance_rows
            ),
        )

    def save_person_snapshot(
        self,
        snapshot: PersonSnapshot,
    ) -> PersonSaveReceipt:
        """Replace one complete person/source snapshot transactionally."""
        self._validate_person_snapshot(snapshot)
        existing = self.load_person_snapshot(snapshot.person.person_id)
        if existing == snapshot:
            return PersonSaveReceipt(
                status="unchanged",
                person_id=snapshot.person.person_id,
                source_record_count=len(snapshot.source_records),
                external_identity_count=len(snapshot.external_identities),
            )
        now = _utc_now()
        with transcript_store.connect(self.root) as con:
            self._require_current_schema(con)
            con.execute("BEGIN IMMEDIATE")
            try:
                person = snapshot.person
                con.execute(
                    """
                    INSERT INTO knowledge_people (
                        id, status, primary_name, metadata_json,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        status = excluded.status,
                        primary_name = excluded.primary_name,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        person.person_id,
                        person.status,
                        person.primary_name,
                        _json_dumps(person.metadata),
                        now,
                        now,
                    ),
                )
                con.execute(
                    """
                    DELETE FROM knowledge_external_identities
                    WHERE person_id = ?
                    """,
                    (person.person_id,),
                )
                con.execute(
                    """
                    DELETE FROM knowledge_source_records
                    WHERE person_id = ?
                    """,
                    (person.person_id,),
                )
                for source in snapshot.source_records:
                    con.execute(
                        """
                        INSERT INTO knowledge_source_records (
                            id, person_id, source_profile_id, provider_kind,
                            account_id, tenant_id, external_ref, label,
                            relationship_scope, identifier_authority,
                            source_event_at, observed_at, retrieved_at,
                            valid_from, valid_to, freshness_state, source_uri,
                            content_hash, source_context_json, metadata_json,
                            created_at, updated_at
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            source.source_record_id,
                            source.person_id,
                            source.source_profile_id,
                            source.provider_kind,
                            source.account_id,
                            source.tenant_id,
                            source.external_ref,
                            source.label,
                            source.relationship_scope,
                            source.identifier_authority,
                            source.source_event_at,
                            source.observed_at,
                            source.retrieved_at,
                            source.valid_from,
                            source.valid_to,
                            source.freshness_state,
                            source.source_uri,
                            source.content_hash,
                            _json_dumps(source.source_context),
                            _json_dumps(source.metadata),
                            now,
                            now,
                        ),
                    )
                for identity in snapshot.external_identities:
                    con.execute(
                        """
                        INSERT INTO knowledge_external_identities (
                            id, person_id, source_record_id, identity_kind,
                            normalized_value, display_value, authority,
                            verified, valid_from, valid_to, metadata_json,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identity.external_identity_id,
                            identity.person_id,
                            identity.source_record_id,
                            identity.identity_kind,
                            identity.normalized_value,
                            identity.display_value,
                            identity.authority,
                            int(identity.verified),
                            identity.valid_from,
                            identity.valid_to,
                            _json_dumps(identity.metadata),
                            now,
                            now,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return PersonSaveReceipt(
            status="updated" if existing is not None else "inserted",
            person_id=snapshot.person.person_id,
            source_record_count=len(snapshot.source_records),
            external_identity_count=len(snapshot.external_identities),
        )

    def load_person_snapshot(
        self,
        person_id: str,
    ) -> PersonSnapshot | None:
        """Load one complete person/source snapshot."""
        person_id = _uuid(person_id, field_name="person_id")
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return None
            person = con.execute(
                "SELECT * FROM knowledge_people WHERE id = ?",
                (person_id,),
            ).fetchone()
            if person is None:
                return None
            source_rows = con.execute(
                """
                SELECT *
                FROM knowledge_source_records
                WHERE person_id = ?
                ORDER BY id
                """,
                (person_id,),
            ).fetchall()
            identity_rows = con.execute(
                """
                SELECT *
                FROM knowledge_external_identities
                WHERE person_id = ?
                ORDER BY id
                """,
                (person_id,),
            ).fetchall()
        return PersonSnapshot(
            person=PersonRecord(
                person_id=str(person["id"]),
                status=str(person["status"]),
                primary_name=str(person["primary_name"]),
                metadata=_json_object(str(person["metadata_json"])),
            ),
            source_records=tuple(
                SourceRecord(
                    source_record_id=str(source["id"]),
                    person_id=str(source["person_id"]),
                    source_profile_id=str(source["source_profile_id"]),
                    provider_kind=str(source["provider_kind"]),
                    account_id=str(source["account_id"]),
                    tenant_id=str(source["tenant_id"]),
                    external_ref=str(source["external_ref"]),
                    label=str(source["label"]),
                    relationship_scope=str(source["relationship_scope"]),
                    identifier_authority=str(source["identifier_authority"]),
                    observed_at=str(source["observed_at"]),
                    content_hash=str(source["content_hash"]),
                    source_event_at=str(source["source_event_at"]),
                    retrieved_at=str(source["retrieved_at"]),
                    valid_from=str(source["valid_from"]),
                    valid_to=str(source["valid_to"]),
                    freshness_state=str(source["freshness_state"]),
                    source_uri=str(source["source_uri"]),
                    source_context=_json_object(
                        str(source["source_context_json"])
                    ),
                    metadata=_json_object(str(source["metadata_json"])),
                )
                for source in source_rows
            ),
            external_identities=tuple(
                ExternalIdentityRecord(
                    external_identity_id=str(identity["id"]),
                    person_id=str(identity["person_id"]),
                    source_record_id=str(identity["source_record_id"]),
                    identity_kind=str(identity["identity_kind"]),
                    normalized_value=str(identity["normalized_value"]),
                    display_value=str(identity["display_value"]),
                    authority=str(identity["authority"]),
                    verified=bool(identity["verified"]),
                    valid_from=str(identity["valid_from"]),
                    valid_to=str(identity["valid_to"]),
                    metadata=_json_object(str(identity["metadata_json"])),
                )
                for identity in identity_rows
            ),
        )

    def load_reviewed_person_snapshots(
        self,
        *,
        limit: int = 50,
    ) -> tuple[PersonSnapshot, ...]:
        """Load the bounded current roster whose person status is reviewed."""
        if limit < 1 or limit > 100:
            raise ValueError("Reviewed person snapshot limit must be 1..100.")
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return ()
            rows = con.execute(
                """
                SELECT id
                FROM knowledge_people
                WHERE status = 'reviewed'
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        snapshots = tuple(
            snapshot
            for row in rows
            if (snapshot := self.load_person_snapshot(str(row["id"]))) is not None
        )
        if any(snapshot.person.status != "reviewed" for snapshot in snapshots):
            raise RuntimeError("Reviewed person roster changed while it was read.")
        return snapshots

    def save_processing_history(
        self,
        history: ProcessingHistory,
    ) -> ProcessingSaveReceipt:
        """Append immutable evaluation history and advance its current pointer."""
        self._validate_processing_history(history)
        existing = self.load_processing_history(history.conversation_id)
        if existing == history:
            return ProcessingSaveReceipt(
                status="unchanged",
                conversation_id=history.conversation_id,
                evaluation_count=len(history.evaluations),
                review_decision_count=len(history.review_decisions),
            )
        existing_evaluations = {
            item.evaluation_id: item
            for item in (existing.evaluations if existing else ())
        }
        existing_decisions = {
            item.decision_id: item
            for item in (existing.review_decisions if existing else ())
        }
        for evaluation in history.evaluations:
            prior = existing_evaluations.get(evaluation.evaluation_id)
            if prior is not None and prior != evaluation:
                raise ValueError(
                    "Evaluation history is immutable: "
                    f"{evaluation.evaluation_id}."
                )
        for decision in history.review_decisions:
            prior = existing_decisions.get(decision.decision_id)
            if prior is not None and prior != decision:
                raise ValueError(
                    "Review decision history is immutable: "
                    f"{decision.decision_id}."
                )
        all_evaluation_ids = set(existing_evaluations) | {
            item.evaluation_id for item in history.evaluations
        }
        if (
            history.current_evaluation_id
            and history.current_evaluation_id not in all_evaluation_ids
        ):
            raise ValueError(
                "Current evaluation must exist in processing history."
            )
        now = _utc_now()
        with transcript_store.connect(self.root) as con:
            self._require_current_schema(con)
            conversation = con.execute(
                "SELECT 1 FROM knowledge_conversations WHERE id = ?",
                (history.conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ValueError(
                    "Processing history references an unknown conversation."
                )
            con.execute("BEGIN IMMEDIATE")
            try:
                for evaluation in history.evaluations:
                    if evaluation.evaluation_id in existing_evaluations:
                        continue
                    con.execute(
                        """
                        INSERT INTO knowledge_evaluations (
                            id, conversation_id, evaluation_type,
                            schema_version, status, model_profile,
                            input_artifact_id, output_artifact_id,
                            evidence_bundle_id, rubric_versions_json,
                            payload_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evaluation.evaluation_id,
                            evaluation.conversation_id,
                            evaluation.evaluation_type,
                            evaluation.schema_version,
                            evaluation.status,
                            evaluation.model_profile,
                            evaluation.input_artifact_id,
                            evaluation.output_artifact_id,
                            evaluation.evidence_bundle_id,
                            _json_dumps(evaluation.rubric_versions),
                            _json_dumps(evaluation.payload),
                            evaluation.created_at,
                        ),
                    )
                for decision in history.review_decisions:
                    if decision.decision_id in existing_decisions:
                        continue
                    con.execute(
                        """
                        INSERT INTO knowledge_review_decisions (
                            id, evaluation_id, proposal_id, action,
                            reviewer, method, decided_at, note,
                            supersedes_decision_id,
                            reviewer_asserted_identity_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            decision.decision_id,
                            decision.evaluation_id,
                            decision.proposal_id,
                            decision.action,
                            decision.reviewer,
                            decision.method,
                            decision.decided_at,
                            decision.note,
                            decision.supersedes_decision_id,
                            _json_dumps(
                                decision.reviewer_asserted_identity
                            ),
                        ),
                    )
                con.execute(
                    """
                    INSERT INTO knowledge_processing_state (
                        conversation_id, current_evaluation_id, updated_at
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        current_evaluation_id =
                            excluded.current_evaluation_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        history.conversation_id,
                        history.current_evaluation_id,
                        now,
                    ),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return ProcessingSaveReceipt(
            status="updated" if existing is not None else "inserted",
            conversation_id=history.conversation_id,
            evaluation_count=len(history.evaluations),
            review_decision_count=len(history.review_decisions),
        )

    def load_processing_history(
        self,
        conversation_id: str,
    ) -> ProcessingHistory | None:
        """Load immutable evaluations, decisions, and the current pointer."""
        conversation_id = (
            _uuid(conversation_id, field_name="conversation_id")
            if conversation_id
            else ""
        )
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return None
            state = con.execute(
                """
                SELECT current_evaluation_id
                FROM knowledge_processing_state
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if state is None:
                return None
            evaluation_rows = con.execute(
                """
                SELECT *
                FROM knowledge_evaluations
                WHERE conversation_id = ?
                ORDER BY created_at, id
                """,
                (conversation_id,),
            ).fetchall()
            decision_rows = con.execute(
                """
                SELECT decision.*
                FROM knowledge_review_decisions AS decision
                JOIN knowledge_evaluations AS evaluation
                  ON evaluation.id = decision.evaluation_id
                WHERE evaluation.conversation_id = ?
                ORDER BY decision.decided_at, decision.id
                """,
                (conversation_id,),
            ).fetchall()
        return ProcessingHistory(
            conversation_id=conversation_id,
            current_evaluation_id=str(state["current_evaluation_id"]),
            evaluations=tuple(
                EvaluationRecord(
                    evaluation_id=str(evaluation["id"]),
                    conversation_id=str(evaluation["conversation_id"]),
                    evaluation_type=str(evaluation["evaluation_type"]),
                    schema_version=str(evaluation["schema_version"]),
                    status=str(evaluation["status"]),
                    created_at=str(evaluation["created_at"]),
                    model_profile=str(evaluation["model_profile"]),
                    input_artifact_id=str(
                        evaluation["input_artifact_id"]
                    ),
                    output_artifact_id=str(
                        evaluation["output_artifact_id"]
                    ),
                    evidence_bundle_id=str(
                        evaluation["evidence_bundle_id"]
                    ),
                    rubric_versions=_json_object(
                        str(evaluation["rubric_versions_json"])
                    ),
                    payload=_json_object(str(evaluation["payload_json"])),
                )
                for evaluation in evaluation_rows
            ),
            review_decisions=tuple(
                ReviewDecisionRecord(
                    decision_id=str(decision["id"]),
                    evaluation_id=str(decision["evaluation_id"]),
                    proposal_id=str(decision["proposal_id"]),
                    action=str(decision["action"]),
                    reviewer=str(decision["reviewer"]),
                    method=str(decision["method"]),
                    decided_at=str(decision["decided_at"]),
                    note=str(decision["note"]),
                    supersedes_decision_id=str(
                        decision["supersedes_decision_id"]
                    ),
                    reviewer_asserted_identity=_json_object(
                        str(decision["reviewer_asserted_identity_json"])
                    ),
                )
                for decision in decision_rows
            ),
        )

    def save_observations(
        self,
        conversation_id: str,
        observations: tuple[ObservationRecord, ...],
    ) -> ObservationSaveReceipt:
        """Append immutable source observations for one conversation."""
        conversation_id = (
            _uuid(conversation_id, field_name="conversation_id")
            if conversation_id
            else ""
        )
        for observation in observations:
            _uuid(observation.observation_id, field_name="observation_id")
            if observation.conversation_id != conversation_id:
                raise ValueError(
                    "Observation belongs to a different conversation."
                )
            if not all(
                (
                    observation.observation_type,
                    observation.subject_type,
                    observation.subject_id,
                    observation.source_type,
                    observation.source_id,
                    observation.observed_at,
                )
            ):
                raise ValueError(
                    "Observation type, subject, source, and observed_at "
                    "are required."
                )
        existing = {
            item.observation_id: item
            for item in self.load_observations(conversation_id)
        }
        for observation in observations:
            prior = existing.get(observation.observation_id)
            if prior is not None and prior != observation:
                raise ValueError(
                    "Observation history is immutable: "
                    f"{observation.observation_id}."
                )
        missing = [
            item for item in observations if item.observation_id not in existing
        ]
        if not missing:
            return ObservationSaveReceipt(
                status="unchanged",
                conversation_id=conversation_id,
                observation_count=len(observations),
            )
        now = _utc_now()
        with transcript_store.connect(self.root) as con:
            self._require_current_schema(con)
            con.execute("BEGIN IMMEDIATE")
            try:
                for observation in missing:
                    con.execute(
                        """
                        INSERT INTO knowledge_observations (
                            id, observation_type, subject_type, subject_id,
                            source_type, source_id, conversation_id,
                            source_event_at, observed_at, retrieved_at,
                            valid_from, valid_to, review_state, payload_json,
                            content_hash, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            observation.observation_id,
                            observation.observation_type,
                            observation.subject_type,
                            observation.subject_id,
                            observation.source_type,
                            observation.source_id,
                            observation.conversation_id or None,
                            observation.source_event_at,
                            observation.observed_at,
                            observation.retrieved_at,
                            observation.valid_from,
                            observation.valid_to,
                            observation.review_state,
                            _json_dumps(observation.payload),
                            observation.content_hash,
                            now,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return ObservationSaveReceipt(
            status="inserted",
            conversation_id=conversation_id,
            observation_count=len(observations),
        )

    def load_observations(
        self,
        conversation_id: str,
    ) -> tuple[ObservationRecord, ...]:
        """Load immutable source observations for one conversation."""
        conversation_id = (
            _uuid(conversation_id, field_name="conversation_id")
            if conversation_id
            else ""
        )
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return ()
            if conversation_id:
                rows = con.execute(
                    """
                    SELECT *
                    FROM knowledge_observations
                    WHERE conversation_id = ?
                    ORDER BY observed_at, id
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT *
                    FROM knowledge_observations
                    WHERE conversation_id IS NULL
                    ORDER BY observed_at, id
                    """
                ).fetchall()
        return tuple(
            ObservationRecord(
                observation_id=str(row["id"]),
                observation_type=str(row["observation_type"]),
                subject_type=str(row["subject_type"]),
                subject_id=str(row["subject_id"]),
                source_type=str(row["source_type"]),
                source_id=str(row["source_id"]),
                conversation_id=str(row["conversation_id"] or ""),
                source_event_at=str(row["source_event_at"]),
                observed_at=str(row["observed_at"]),
                retrieved_at=str(row["retrieved_at"]),
                valid_from=str(row["valid_from"]),
                valid_to=str(row["valid_to"]),
                review_state=str(row["review_state"]),
                payload=_json_object(str(row["payload_json"])),
                content_hash=str(row["content_hash"]),
            )
            for row in rows
        )

    def save_projection_state(
        self,
        state: ProjectionStateRecord,
    ) -> str:
        """Record one replaceable projection watermark."""
        if not all(
            (
                state.projection_name,
                state.scope_type,
                state.scope_id,
                state.schema_version,
                state.input_watermark,
                state.built_at,
            )
        ):
            raise ValueError("Projection identity, version, and watermark are required.")
        existing = self.load_projection_state(
            state.projection_name,
            state.scope_type,
            state.scope_id,
        )
        if existing == state:
            return "unchanged"
        with transcript_store.connect(self.root) as con:
            self._require_current_schema(con)
            con.execute(
                """
                INSERT INTO knowledge_projection_state (
                    projection_name, scope_type, scope_id, schema_version,
                    input_watermark, built_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(projection_name, scope_type, scope_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    input_watermark = excluded.input_watermark,
                    built_at = excluded.built_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    state.projection_name,
                    state.scope_type,
                    state.scope_id,
                    state.schema_version,
                    state.input_watermark,
                    state.built_at,
                    _json_dumps(state.metadata),
                ),
            )
            con.commit()
        return "updated" if existing is not None else "inserted"

    def load_projection_state(
        self,
        projection_name: str,
        scope_type: str,
        scope_id: str,
    ) -> ProjectionStateRecord | None:
        """Load one projection watermark."""
        with transcript_store.connect(self.root) as con:
            if not self._schema_is_current(con):
                return None
            row = con.execute(
                """
                SELECT *
                FROM knowledge_projection_state
                WHERE projection_name = ? AND scope_type = ? AND scope_id = ?
                """,
                (projection_name, scope_type, scope_id),
            ).fetchone()
        if row is None:
            return None
        return ProjectionStateRecord(
            projection_name=str(row["projection_name"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            schema_version=str(row["schema_version"]),
            input_watermark=str(row["input_watermark"]),
            built_at=str(row["built_at"]),
            metadata=_json_object(str(row["metadata_json"])),
        )

    @staticmethod
    def _schema_is_current(con: sqlite3.Connection) -> bool:
        row = con.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'knowledge_store_state'
            """
        ).fetchone()
        if row is None:
            return False
        state = con.execute(
            """
            SELECT schema_version, dirty
            FROM knowledge_store_state
            WHERE singleton = 1
            """
        ).fetchone()
        return bool(
            state
            and 1 <= int(state["schema_version"]) <= LATEST_SCHEMA_VERSION
            and not bool(state["dirty"])
        )

    @classmethod
    def _require_current_schema(cls, con: sqlite3.Connection) -> None:
        if not cls._schema_is_current(con):
            raise RuntimeError(
                "Conversation knowledge schema is not initialized or is dirty."
            )

    @staticmethod
    def _validate_snapshot(snapshot: ConversationSnapshot) -> None:
        conversation_id = _uuid(
            snapshot.conversation.conversation_id,
            field_name="conversation_id",
        )
        recording_ids: set[str] = set()
        for recording in snapshot.recordings:
            recording_id = _uuid(
                recording.recording_id,
                field_name="recording_id",
            )
            if recording.conversation_id != conversation_id:
                raise ValueError(
                    "Recording belongs to a different conversation."
                )
            if recording_id in recording_ids:
                raise ValueError("Conversation snapshot repeats a recording.")
            recording_ids.add(recording_id)
        utterance_ids: set[str] = set()
        for utterance in snapshot.utterances:
            utterance_id = _uuid(
                utterance.utterance_id,
                field_name="utterance_id",
            )
            if utterance.conversation_id != conversation_id:
                raise ValueError(
                    "Utterance belongs to a different conversation."
                )
            if utterance.recording_id not in recording_ids:
                raise ValueError(
                    "Utterance references an unprepared recording."
                )
            if utterance_id in utterance_ids:
                raise ValueError("Conversation snapshot repeats an utterance.")
            if utterance.ordinal < 0:
                raise ValueError("Utterance ordinal must be non-negative.")
            if (
                utterance.start_ms is not None
                and utterance.end_ms is not None
                and utterance.end_ms < utterance.start_ms
            ):
                raise ValueError(
                    "Utterance end time cannot precede its start time."
                )
            utterance_ids.add(utterance_id)

    @staticmethod
    def _validate_person_snapshot(snapshot: PersonSnapshot) -> None:
        person_id = _uuid(snapshot.person.person_id, field_name="person_id")
        if not snapshot.person.status or not snapshot.person.primary_name:
            raise ValueError("Person status and primary_name are required.")
        source_ids: set[str] = set()
        for source in snapshot.source_records:
            if source.person_id != person_id:
                raise ValueError(
                    "Source record belongs to a different person."
                )
            if (
                not source.source_record_id
                or not source.source_profile_id
                or not source.provider_kind
            ):
                raise ValueError(
                    "Source record requires ID, profile, and provider kind."
                )
            if source.source_record_id in source_ids:
                raise ValueError("Person snapshot repeats a source record.")
            source_ids.add(source.source_record_id)
        identity_ids: set[str] = set()
        for identity in snapshot.external_identities:
            if identity.person_id != person_id:
                raise ValueError(
                    "External identity belongs to a different person."
                )
            if identity.source_record_id not in source_ids:
                raise ValueError(
                    "External identity references an unprepared source record."
                )
            if (
                not identity.external_identity_id
                or not identity.identity_kind
                or not identity.normalized_value
            ):
                raise ValueError(
                    "External identity requires ID, kind, and normalized value."
                )
            if identity.external_identity_id in identity_ids:
                raise ValueError(
                    "Person snapshot repeats an external identity."
                )
            identity_ids.add(identity.external_identity_id)

    @staticmethod
    def _validate_processing_history(history: ProcessingHistory) -> None:
        conversation_id = _uuid(
            history.conversation_id,
            field_name="conversation_id",
        )
        evaluation_ids: set[str] = set()
        for evaluation in history.evaluations:
            evaluation_id = _uuid(
                evaluation.evaluation_id,
                field_name="evaluation_id",
            )
            if evaluation.conversation_id != conversation_id:
                raise ValueError(
                    "Evaluation belongs to a different conversation."
                )
            if evaluation_id in evaluation_ids:
                raise ValueError(
                    "Processing history repeats an evaluation."
                )
            evaluation_ids.add(evaluation_id)
        decision_ids: set[str] = set()
        for decision in history.review_decisions:
            decision_id = _uuid(
                decision.decision_id,
                field_name="decision_id",
            )
            if decision.evaluation_id not in evaluation_ids:
                raise ValueError(
                    "Review decision references an unprepared evaluation."
                )
            if decision.action not in {"confirm", "reject", "defer"}:
                raise ValueError("Unsupported review decision action.")
            if decision_id in decision_ids:
                raise ValueError(
                    "Processing history repeats a review decision."
                )
            decision_ids.add(decision_id)
        if history.current_evaluation_id:
            _uuid(
                history.current_evaluation_id,
                field_name="current_evaluation_id",
            )

    def _create_backup(self, label: str) -> Path:
        source_path = transcript_store.db_path(self.root)
        if not source_path.is_file():
            raise RuntimeError(
                "Conversation knowledge backup requires an existing database."
            )
        backup_dir = self.root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(backup_dir, 0o700)
        target = backup_dir / (
            f"transcripts-{label}-{uuid4().hex[:12]}.sqlite3"
        )
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=backup_dir,
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            with sqlite3.connect(source_path) as source:
                with sqlite3.connect(temporary) as destination:
                    source.backup(destination)
                    integrity = destination.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                    if not integrity or integrity[0] != "ok":
                        raise RuntimeError(
                            "Conversation knowledge backup failed integrity check."
                        )
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        return target

    @staticmethod
    def _apply_v1(con: sqlite3.Connection) -> None:
        now = _utc_now()
        con.execute(
            """
            CREATE TABLE knowledge_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_store_state (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                authority_mode TEXT NOT NULL,
                dirty INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_schema_migrations (version, applied_at)
            VALUES (1, ?)
            """,
            (now,),
        )
        con.execute(
            """
            INSERT INTO knowledge_store_state (
                singleton, schema_version, authority_mode, dirty, updated_at
            )
            VALUES (1, 1, ?, 0, ?)
            """,
            (AUTHORITY_MODE_SIDECAR, now),
        )
        con.execute(
            """
            CREATE TABLE knowledge_conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                starts_at TEXT NOT NULL DEFAULT '',
                ends_at TEXT NOT NULL DEFAULT '',
                calendar_association_state TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_recordings (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                transcript_document_id TEXT NOT NULL DEFAULT '',
                source_blob_id TEXT NOT NULL DEFAULT '',
                backend TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_recordings_conversation
            ON knowledge_recordings(conversation_id)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_utterances (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                recording_id TEXT NOT NULL
                    REFERENCES knowledge_recordings(id) ON DELETE CASCADE,
                speaker_label TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                start_ms INTEGER,
                end_ms INTEGER,
                text TEXT NOT NULL,
                source_artifact_id TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(recording_id, ordinal)
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_utterances_conversation
            ON knowledge_utterances(conversation_id)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_utterances_recording
            ON knowledge_utterances(recording_id)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_people (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                primary_name TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_source_records (
                id TEXT PRIMARY KEY,
                person_id TEXT REFERENCES knowledge_people(id)
                    ON DELETE CASCADE,
                source_profile_id TEXT NOT NULL,
                provider_kind TEXT NOT NULL,
                account_id TEXT NOT NULL DEFAULT '',
                tenant_id TEXT NOT NULL DEFAULT '',
                external_ref TEXT NOT NULL DEFAULT '',
                label TEXT NOT NULL DEFAULT '',
                relationship_scope TEXT NOT NULL DEFAULT '',
                identifier_authority TEXT NOT NULL DEFAULT '',
                source_event_at TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL DEFAULT '',
                retrieved_at TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                freshness_state TEXT NOT NULL DEFAULT 'current',
                source_uri TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                source_context_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_source_records_person
            ON knowledge_source_records(person_id)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_source_records_scope
            ON knowledge_source_records(
                source_profile_id, account_id, tenant_id, provider_kind
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_source_records_external_ref
            ON knowledge_source_records(
                source_profile_id, provider_kind, external_ref
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_external_identities (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL
                    REFERENCES knowledge_people(id) ON DELETE CASCADE,
                source_record_id TEXT NOT NULL
                    REFERENCES knowledge_source_records(id) ON DELETE CASCADE,
                identity_kind TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                display_value TEXT NOT NULL DEFAULT '',
                authority TEXT NOT NULL DEFAULT '',
                verified INTEGER NOT NULL DEFAULT 0,
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(
                    source_record_id, identity_kind, normalized_value
                )
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_external_identities_person
            ON knowledge_external_identities(person_id)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_external_identities_lookup
            ON knowledge_external_identities(
                identity_kind, normalized_value, verified
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_evaluations (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                evaluation_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL,
                model_profile TEXT NOT NULL DEFAULT '',
                input_artifact_id TEXT NOT NULL DEFAULT '',
                output_artifact_id TEXT NOT NULL DEFAULT '',
                evidence_bundle_id TEXT NOT NULL DEFAULT '',
                rubric_versions_json TEXT NOT NULL DEFAULT '{}',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evaluations_conversation
            ON knowledge_evaluations(conversation_id, created_at)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_review_decisions (
                id TEXT PRIMARY KEY,
                evaluation_id TEXT NOT NULL
                    REFERENCES knowledge_evaluations(id) ON DELETE CASCADE,
                proposal_id TEXT NOT NULL,
                action TEXT NOT NULL
                    CHECK (action IN ('confirm', 'reject', 'defer')),
                reviewer TEXT NOT NULL,
                method TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                supersedes_decision_id TEXT NOT NULL DEFAULT '',
                reviewer_asserted_identity_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_review_decisions_evaluation
            ON knowledge_review_decisions(evaluation_id, decided_at)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_processing_state (
                conversation_id TEXT PRIMARY KEY
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                current_evaluation_id TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_relationships (
                id TEXT PRIMARY KEY,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'forward',
                source_observation_id TEXT NOT NULL DEFAULT '',
                review_state TEXT NOT NULL DEFAULT 'unreviewed',
                confidence_numeric REAL,
                confidence_band TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_relationships_subject
            ON knowledge_relationships(
                subject_type, subject_id, relationship_type
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_relationships_object
            ON knowledge_relationships(
                object_type, object_id, relationship_type
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_concepts (
                id TEXT PRIMARY KEY,
                concept_type TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                display_value TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(concept_type, normalized_value)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_concept_mentions (
                id TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL
                    REFERENCES knowledge_concepts(id) ON DELETE CASCADE,
                conversation_id TEXT
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                utterance_id TEXT
                    REFERENCES knowledge_utterances(id) ON DELETE CASCADE,
                evidence_snapshot_id TEXT NOT NULL DEFAULT '',
                person_id TEXT
                    REFERENCES knowledge_people(id) ON DELETE SET NULL,
                observed_at TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_concept_mentions_concept
            ON knowledge_concept_mentions(concept_id, observed_at)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_concept_mentions_conversation
            ON knowledge_concept_mentions(conversation_id)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_observations (
                id TEXT PRIMARY KEY,
                observation_type TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                conversation_id TEXT
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                source_event_at TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL,
                retrieved_at TEXT NOT NULL DEFAULT '',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                review_state TEXT NOT NULL DEFAULT 'unreviewed',
                payload_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_observations_subject
            ON knowledge_observations(
                subject_type, subject_id, observation_type
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_observations_conversation
            ON knowledge_observations(conversation_id, observed_at)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_claims (
                id TEXT PRIMARY KEY,
                claim_type TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_type TEXT NOT NULL DEFAULT '',
                object_id TEXT NOT NULL DEFAULT '',
                value_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                evaluation_id TEXT
                    REFERENCES knowledge_evaluations(id) ON DELETE SET NULL,
                confidence_numeric REAL,
                confidence_band TEXT NOT NULL DEFAULT '',
                rubric_version TEXT NOT NULL DEFAULT '',
                alternatives_json TEXT NOT NULL DEFAULT '[]',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                valid_from TEXT NOT NULL DEFAULT '',
                valid_to TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_claims_subject
            ON knowledge_claims(subject_type, subject_id, claim_type, status)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_projection_state (
                projection_name TEXT NOT NULL,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                input_watermark TEXT NOT NULL,
                built_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(projection_name, scope_type, scope_id)
            )
            """
        )

    @staticmethod
    def _apply_v2(con: sqlite3.Connection) -> None:
        now = _utc_now()
        con.execute(
            """
            CREATE TABLE knowledge_evidence_independence_groups (
                id TEXT PRIMARY KEY,
                group_key TEXT NOT NULL UNIQUE,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_evidence_snapshots (
                id TEXT PRIMARY KEY,
                source_record_id TEXT
                    REFERENCES knowledge_source_records(id) ON DELETE SET NULL,
                source_profile_id TEXT NOT NULL,
                provider_kind TEXT NOT NULL,
                account_id TEXT NOT NULL DEFAULT '',
                tenant_id TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL,
                capability TEXT NOT NULL,
                snippet TEXT NOT NULL DEFAULT '',
                structured_metadata_json TEXT NOT NULL DEFAULT '{}',
                source_event_at TEXT NOT NULL DEFAULT '',
                observed_at TEXT NOT NULL,
                retrieved_at TEXT NOT NULL DEFAULT '',
                expires_at TEXT NOT NULL DEFAULT '',
                temporal_class TEXT NOT NULL,
                source_uri TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                redaction_json TEXT NOT NULL DEFAULT '{}',
                truncation_json TEXT NOT NULL DEFAULT '{}',
                independence_group_id TEXT NOT NULL
                    REFERENCES knowledge_evidence_independence_groups(id),
                freshness_state TEXT NOT NULL,
                embedding_json TEXT NOT NULL DEFAULT '[]',
                embedding_provider TEXT NOT NULL DEFAULT '',
                embedding_model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_scope
            ON knowledge_evidence_snapshots(
                source_profile_id, account_id, tenant_id, capability
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_temporal
            ON knowledge_evidence_snapshots(
                temporal_class, source_event_at, observed_at, retrieved_at
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_source_record
            ON knowledge_evidence_snapshots(source_record_id)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_content_hash
            ON knowledge_evidence_snapshots(content_hash)
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_embedding_profile
            ON knowledge_evidence_snapshots(
                embedding_provider, embedding_model, source_profile_id
            )
            """
        )
        con.execute(
            """
            CREATE VIRTUAL TABLE knowledge_evidence_fts USING fts5(
                evidence_id UNINDEXED,
                search_text,
                tokenize='unicode61'
            )
            """
        )
        con.execute(
            """
            CREATE VIRTUAL TABLE knowledge_concept_fts USING fts5(
                concept_id UNINDEXED,
                search_text,
                tokenize='unicode61'
            )
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_retrieval_requests (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES knowledge_conversations(id) ON DELETE CASCADE,
                recording_ids_json TEXT NOT NULL DEFAULT '[]',
                speaker_labels_json TEXT NOT NULL DEFAULT '[]',
                clue_ids_json TEXT NOT NULL DEFAULT '[]',
                conversation_at TEXT NOT NULL DEFAULT '',
                as_of TEXT NOT NULL,
                prepared_person_ids_json TEXT NOT NULL DEFAULT '[]',
                scopes_json TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                budgets_json TEXT NOT NULL,
                freshness_policy TEXT NOT NULL,
                hindsight_policy TEXT NOT NULL,
                retrieval_version TEXT NOT NULL,
                ranking_version TEXT NOT NULL,
                requesting_workflow TEXT NOT NULL,
                run_id TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_retrieval_requests_conversation
            ON knowledge_retrieval_requests(conversation_id, as_of, created_at)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_evidence_bundles (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL
                    REFERENCES knowledge_retrieval_requests(id)
                    ON DELETE CASCADE,
                status TEXT NOT NULL,
                candidate_person_ids_json TEXT NOT NULL DEFAULT '[]',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                source_failures_json TEXT NOT NULL DEFAULT '[]',
                allowlists_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_evidence_bundles_request
            ON knowledge_evidence_bundles(request_id, created_at)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_evidence_bundle_items (
                bundle_id TEXT NOT NULL
                    REFERENCES knowledge_evidence_bundles(id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL
                    REFERENCES knowledge_evidence_snapshots(id)
                    ON DELETE RESTRICT,
                disposition TEXT NOT NULL
                    CHECK (disposition IN ('included', 'excluded')),
                reason_code TEXT NOT NULL,
                rank INTEGER NOT NULL DEFAULT 0,
                score REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(bundle_id, evidence_id)
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_bundle_items_disposition
            ON knowledge_evidence_bundle_items(
                bundle_id, disposition, reason_code, rank
            )
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_schema_migrations (version, applied_at)
            VALUES (2, ?)
            """,
            (now,),
        )
        con.execute(
            """
            UPDATE knowledge_store_state
            SET schema_version = 2, updated_at = ?
            WHERE singleton = 1
            """,
            (now,),
        )

    @staticmethod
    def _rollback_v2(con: sqlite3.Connection) -> None:
        now = _utc_now()
        con.execute("DROP TABLE IF EXISTS knowledge_evidence_bundle_items")
        con.execute("DROP TABLE IF EXISTS knowledge_evidence_bundles")
        con.execute("DROP TABLE IF EXISTS knowledge_retrieval_requests")
        con.execute("DROP TABLE IF EXISTS knowledge_concept_fts")
        con.execute("DROP TABLE IF EXISTS knowledge_evidence_fts")
        con.execute("DROP TABLE IF EXISTS knowledge_evidence_snapshots")
        con.execute(
            "DROP TABLE IF EXISTS knowledge_evidence_independence_groups"
        )
        con.execute(
            "DELETE FROM knowledge_schema_migrations WHERE version = 2"
        )
        con.execute(
            """
            UPDATE knowledge_store_state
            SET schema_version = 1, updated_at = ?
            WHERE singleton = 1
            """,
            (now,),
        )

    @staticmethod
    def _apply_v3(con: sqlite3.Connection) -> None:
        now = _utc_now()
        con.execute(
            """
            CREATE TABLE knowledge_current_person_profiles (
                person_id TEXT PRIMARY KEY
                    REFERENCES knowledge_people(id) ON DELETE CASCADE,
                resolution_status TEXT NOT NULL,
                primary_name TEXT NOT NULL,
                aliases_json TEXT NOT NULL DEFAULT '[]',
                source_record_ids_json TEXT NOT NULL DEFAULT '[]',
                observation_ids_json TEXT NOT NULL,
                input_watermark TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                built_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_current_people_name
            ON knowledge_current_person_profiles(primary_name, person_id)
            """
        )
        con.execute(
            """
            CREATE TABLE knowledge_affinity_profiles (
                id TEXT PRIMARY KEY,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                affinity_type TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL DEFAULT '',
                normalized_value TEXT NOT NULL DEFAULT '',
                display_value TEXT NOT NULL DEFAULT '',
                support_count INTEGER NOT NULL CHECK (support_count >= 0),
                independent_interaction_count INTEGER NOT NULL
                    CHECK (independent_interaction_count >= 0),
                first_observed_at TEXT NOT NULL DEFAULT '',
                last_observed_at TEXT NOT NULL DEFAULT '',
                review_state TEXT NOT NULL,
                observation_ids_json TEXT NOT NULL,
                input_watermark TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                built_at TEXT NOT NULL,
                UNIQUE(
                    subject_type, subject_id, affinity_type, object_type,
                    object_id, normalized_value
                )
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_affinity_subject
            ON knowledge_affinity_profiles(
                subject_type, subject_id, affinity_type
            )
            """
        )
        con.execute(
            """
            CREATE INDEX idx_knowledge_affinity_object
            ON knowledge_affinity_profiles(
                object_type, object_id, normalized_value, affinity_type
            )
            """
        )
        con.execute(
            """
            INSERT INTO knowledge_schema_migrations (version, applied_at)
            VALUES (3, ?)
            """,
            (now,),
        )
        con.execute(
            """
            UPDATE knowledge_store_state
            SET schema_version = 3, updated_at = ?
            WHERE singleton = 1
            """,
            (now,),
        )

    @staticmethod
    def _rollback_v3(con: sqlite3.Connection) -> None:
        now = _utc_now()
        con.execute("DROP TABLE IF EXISTS knowledge_affinity_profiles")
        con.execute("DROP TABLE IF EXISTS knowledge_current_person_profiles")
        con.execute(
            "DELETE FROM knowledge_schema_migrations WHERE version = 3"
        )
        con.execute(
            """
            UPDATE knowledge_store_state
            SET schema_version = 2, updated_at = ?
            WHERE singleton = 1
            """,
            (now,),
        )

    @staticmethod
    def _rollback_v1(con: sqlite3.Connection) -> None:
        con.execute("DROP TABLE IF EXISTS knowledge_projection_state")
        con.execute("DROP TABLE IF EXISTS knowledge_claims")
        con.execute("DROP TABLE IF EXISTS knowledge_observations")
        con.execute("DROP TABLE IF EXISTS knowledge_concept_mentions")
        con.execute("DROP TABLE IF EXISTS knowledge_concepts")
        con.execute("DROP TABLE IF EXISTS knowledge_relationships")
        con.execute("DROP TABLE IF EXISTS knowledge_processing_state")
        con.execute("DROP TABLE IF EXISTS knowledge_review_decisions")
        con.execute("DROP TABLE IF EXISTS knowledge_evaluations")
        con.execute("DROP TABLE IF EXISTS knowledge_external_identities")
        con.execute("DROP TABLE IF EXISTS knowledge_source_records")
        con.execute("DROP TABLE IF EXISTS knowledge_people")
        con.execute("DROP TABLE IF EXISTS knowledge_utterances")
        con.execute("DROP TABLE IF EXISTS knowledge_recordings")
        con.execute("DROP TABLE IF EXISTS knowledge_conversations")
        con.execute("DROP TABLE IF EXISTS knowledge_store_state")
        con.execute("DROP TABLE IF EXISTS knowledge_schema_migrations")
