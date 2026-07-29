from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import conversation_processing
import transcript_store
from conversation_knowledge_store import (
    ConversationKnowledgeStore,
    ConversationRecord,
    ConversationSnapshot,
    EvaluationRecord,
    ExternalIdentityRecord,
    ObservationRecord,
    PersonRecord,
    PersonSnapshot,
    ProcessingHistory,
    ProjectionStateRecord,
    RecordingRecord,
    ReviewDecisionRecord,
    SourceRecord,
    UtteranceRecord,
)


APPLY_APPROVAL_TOKEN = "apply-sidecar-shadow-projection"
PROJECTION_NAME = "sidecar-shadow"
PROJECTION_SCHEMA_VERSION = "transcribe-audio.sidecar-shadow.v1"
_LEGACY_PROJECTION_NAMESPACE = UUID("a1699f20-b9f3-4c88-b18b-769fc8fbbef3")


@dataclass(frozen=True)
class LegacyProjectionInput:
    source_path: Path
    projection_path: Path
    source_transcript_sha256: str
    projection_sha256: str
    conversation_id: str
    recording_id: str
_IDENTITY_NAMESPACE = UUID("e00b88cb-f121-49f1-96c8-59576b7c735f")


@dataclass(frozen=True)
class ProjectionPlan:
    authority_mode: str
    transcript_path: Path
    sidecar_path: Path | None
    source_transcript_sha256: str
    source_sidecar_sha256: str
    input_watermark: str
    built_at: str
    document_id: str
    conversation_snapshot: ConversationSnapshot
    processing_history: ProcessingHistory
    people: tuple[PersonSnapshot, ...]
    observations: tuple[ObservationRecord, ...]
    counts: dict[str, int] = field(default_factory=dict)
    sidecar_schema_version: str = conversation_processing.SCHEMA_VERSION
    sidecar_recording_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciliationReceipt:
    status: str
    authority_mode: str
    conversation_id: str
    source_transcript_sha256: str
    source_sidecar_sha256: str
    input_watermark: str
    reconciled: bool
    counts: dict[str, int]
    receipt_path: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_uuid(*parts: str) -> str:
    return str(uuid5(_IDENTITY_NAMESPACE, "\x1f".join(parts)))


def _opaque_uuid(value: Any, *, field_name: str) -> str:
    try:
        return str(UUID(str(value or "").strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a durable opaque UUID.") from exc


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise


def _write_immutable_private_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        stable_keys = (
            "schema_version",
            "authority_mode",
            "conversation_id",
            "source_transcript_sha256",
            "source_sidecar_sha256",
            "input_watermark",
            "reconciled",
            "counts",
        )
        if any(existing.get(key) != payload.get(key) for key in stable_keys):
            raise ValueError(f"Immutable receipt conflict: {path}.")
        return
    _write_private_json(path, payload)


def materialize_legacy_projection_input(
    source_path: Path,
    *,
    output_root: Path,
) -> LegacyProjectionInput:
    """Create a deterministic private identity overlay without mutating source."""
    source = source_path.expanduser().resolve(strict=True)
    if not source.name.endswith(conversation_processing.TRANSCRIPT_SUFFIX):
        raise ValueError("Legacy projection source must be a normalized transcript.")
    if conversation_processing.processing_sidecar_path(source).exists():
        raise ValueError(
            "Legacy projection overlay is only valid when no authority sidecar exists."
        )
    source_hash = _sha256_file(source)
    payload = _read_json(source)
    try:
        conversation_id = _opaque_uuid(
            payload.get("conversation_id"),
            field_name="conversation_id",
        )
    except ValueError:
        conversation_id = str(
            uuid5(
                _LEGACY_PROJECTION_NAMESPACE,
                f"conversation:{source_hash}",
            )
        )
    try:
        recording_id = _opaque_uuid(
            payload.get("recording_id"),
            field_name="recording_id",
        )
    except ValueError:
        recording_id = str(
            uuid5(
                _LEGACY_PROJECTION_NAMESPACE,
                f"recording:{source_hash}",
            )
        )
    projected = {
        **payload,
        "schema_version": max(2, int(payload.get("schema_version") or 0)),
        "conversation_id": conversation_id,
        "recording_id": recording_id,
        "projection_identity": {
            "scheme": "legacy-source-sha256-uuid5.v1",
            "source_transcript_sha256": source_hash,
        },
    }
    destination_root = output_root.expanduser().resolve()
    destination = destination_root / f"{source_hash}-{source.name}"
    if destination.exists():
        if _read_json(destination) != projected:
            raise ValueError(f"Immutable legacy projection conflict: {destination}.")
    else:
        _write_private_json(destination, projected)
    return LegacyProjectionInput(
        source_path=source,
        projection_path=destination,
        source_transcript_sha256=source_hash,
        projection_sha256=_sha256_file(destination),
        conversation_id=conversation_id,
        recording_id=recording_id,
    )


class ConversationKnowledgeProjector:
    """Project authoritative sidecars into the local knowledge store."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        self.store = ConversationKnowledgeStore(self.root)

    def preview(
        self,
        transcript_path: Path,
        *,
        document_id: str = "",
    ) -> ProjectionPlan:
        """Build a read-only, hash-bound projection plan."""
        path = transcript_path.expanduser().resolve(strict=True)
        if not path.name.endswith(conversation_processing.TRANSCRIPT_SUFFIX):
            raise ValueError("Projection source must be a normalized transcript.")
        transcript = _read_json(path)
        conversation_id = _opaque_uuid(
            transcript.get("conversation_id"),
            field_name="conversation_id",
        )
        recording_id = _opaque_uuid(
            transcript.get("recording_id"),
            field_name="recording_id",
        )
        transcript_hash = _sha256_file(path)
        self._validate_document(
            document_id,
            transcript_hash=transcript_hash,
        )

        sidecar_path = conversation_processing.processing_sidecar_path(path)
        if sidecar_path.is_file():
            sidecar = _read_json(sidecar_path)
            sidecar_hash = _sha256_file(sidecar_path)
        else:
            sidecar_path = None
            sidecar_hash = ""
            sidecar = {
                "schema_version": conversation_processing.SCHEMA_VERSION,
                "conversation_id": conversation_id,
                "recording_ids": [recording_id],
                "current_evaluation_id": "",
                "evaluations": [],
                "review_decisions": [],
            }
        self._validate_sidecar(
            sidecar,
            conversation_id=conversation_id,
            recording_id=recording_id,
        )
        contacts, assignments = self._legacy_identity_records(
            document_id=document_id,
            transcript_path=path,
        )
        snapshot = self._conversation_snapshot(
            transcript,
            transcript_path=path,
            transcript_hash=transcript_hash,
            document_id=document_id,
        )
        history = self._processing_history(sidecar)
        people, person_ids = self._person_snapshots(contacts)
        observations = self._assignment_observations(
            assignments,
            conversation_id=conversation_id,
            person_ids=person_ids,
        )
        proposal_count = sum(
            len(item.payload.get("proposals", []))
            if isinstance(item.payload.get("proposals"), list)
            else 0
            for item in history.evaluations
        )
        counts = {
            "assignments": len(assignments),
            "contacts": len(people),
            "conversations": 1,
            "decisions": len(history.review_decisions),
            "evaluations": len(history.evaluations),
            "proposals": proposal_count,
            "recordings": len(snapshot.recordings),
            "utterances": len(snapshot.utterances),
        }
        watermark = _canonical_hash(
            {
                "projection_schema_version": PROJECTION_SCHEMA_VERSION,
                "source_sidecar_sha256": sidecar_hash,
                "source_transcript_sha256": transcript_hash,
                "legacy_contacts": contacts,
                "legacy_assignments": assignments,
            }
        )
        return ProjectionPlan(
            authority_mode="sidecar",
            transcript_path=path,
            sidecar_path=sidecar_path,
            source_transcript_sha256=transcript_hash,
            source_sidecar_sha256=sidecar_hash,
            input_watermark=watermark,
            built_at=_utc_now(),
            document_id=document_id,
            conversation_snapshot=snapshot,
            processing_history=history,
            people=people,
            observations=observations,
            counts=counts,
            sidecar_schema_version=str(sidecar["schema_version"]),
            sidecar_recording_ids=tuple(
                str(value) for value in sidecar.get("recording_ids", [])
            ),
        )

    def apply(
        self,
        plan: ProjectionPlan,
        *,
        approval_token: str,
        migrate_backup: bool = True,
    ) -> ReconciliationReceipt:
        """Apply one previewed plan without changing either source artifact."""
        if approval_token != APPLY_APPROVAL_TOKEN:
            raise ValueError("Sidecar shadow projection requires its approval token.")
        self._verify_sources_unchanged(plan)
        self.store.migrate(backup=migrate_backup)
        statuses = [
            self.store.save_conversation_snapshot(
                plan.conversation_snapshot
            ).status
        ]
        for person in plan.people:
            statuses.append(self.store.save_person_snapshot(person).status)
        statuses.append(
            self.store.save_processing_history(plan.processing_history).status
        )
        statuses.append(
            self.store.save_observations(
                plan.processing_history.conversation_id,
                plan.observations,
            ).status
        )
        state = ProjectionStateRecord(
            projection_name=PROJECTION_NAME,
            scope_type="conversation",
            scope_id=plan.processing_history.conversation_id,
            schema_version=PROJECTION_SCHEMA_VERSION,
            input_watermark=plan.input_watermark,
            built_at=plan.built_at,
            metadata={
                "authority_mode": plan.authority_mode,
                "counts": plan.counts,
                "document_id": plan.document_id,
                "recording_ids": list(plan.sidecar_recording_ids),
                "sidecar_schema_version": plan.sidecar_schema_version,
                "source_sidecar_sha256": plan.source_sidecar_sha256,
                "source_transcript_sha256": plan.source_transcript_sha256,
            },
        )
        statuses.append(self.store.save_projection_state(state))
        reconciled = self._reconciles(plan)
        status = (
            "inserted"
            if "inserted" in statuses
            else "updated"
            if "updated" in statuses
            else "unchanged"
        )
        receipt_path = (
            self.root
            / "projection-receipts"
            / (
                f"{plan.processing_history.conversation_id}."
                f"{plan.input_watermark}.json"
            )
        )
        receipt_payload = {
            "schema_version": "transcribe-audio.projection-receipt.v1",
            "status": status,
            "authority_mode": plan.authority_mode,
            "conversation_id": plan.processing_history.conversation_id,
            "source_transcript_sha256": plan.source_transcript_sha256,
            "source_sidecar_sha256": plan.source_sidecar_sha256,
            "input_watermark": plan.input_watermark,
            "reconciled": reconciled,
            "counts": plan.counts,
            "recorded_at": _utc_now(),
        }
        _write_immutable_private_json(receipt_path, receipt_payload)
        if not reconciled:
            raise ValueError("Projected records did not reconcile with their sources.")
        return ReconciliationReceipt(
            status=status,
            authority_mode=plan.authority_mode,
            conversation_id=plan.processing_history.conversation_id,
            source_transcript_sha256=plan.source_transcript_sha256,
            source_sidecar_sha256=plan.source_sidecar_sha256,
            input_watermark=plan.input_watermark,
            reconciled=True,
            counts=dict(plan.counts),
            receipt_path=str(receipt_path),
        )

    def export_sidecar(
        self,
        conversation_id: str,
        output_path: Path,
    ) -> dict[str, Any]:
        """Export projected history without modifying the authority sidecar."""
        history = self.store.load_processing_history(conversation_id)
        state = self.store.load_projection_state(
            PROJECTION_NAME,
            "conversation",
            conversation_id,
        )
        if history is None or state is None:
            raise ValueError("Conversation has no projected processing history.")
        evaluations = [dict(item.payload) for item in history.evaluations]
        decisions: list[dict[str, Any]] = []
        for item in history.review_decisions:
            decision = {
                "decision_id": item.decision_id,
                "evaluation_id": item.evaluation_id,
                "proposal_id": item.proposal_id,
                "action": item.action,
                "reviewer": item.reviewer,
                "decision_method": item.method,
                "decided_at": item.decided_at,
                "reviewer_note": item.note,
                "supersedes_decision_id": item.supersedes_decision_id,
            }
            if item.reviewer_asserted_identity:
                decision["reviewer_asserted_identity"] = dict(
                    item.reviewer_asserted_identity
                )
            decisions.append(decision)
        payload: dict[str, Any] = {
            "schema_version": state.metadata.get(
                "sidecar_schema_version",
                conversation_processing.SCHEMA_VERSION,
            ),
            "conversation_id": history.conversation_id,
            "recording_ids": list(state.metadata.get("recording_ids", [])),
            "current_evaluation_id": history.current_evaluation_id,
            "evaluations": evaluations,
        }
        if decisions:
            payload["review_decisions"] = decisions
        _write_private_json(output_path.expanduser().resolve(), payload)
        return payload

    def _validate_document(
        self,
        document_id: str,
        *,
        transcript_hash: str,
    ) -> None:
        if not document_id:
            return
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT kind, artifact_sha256
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None or str(row["kind"]) != "transcript":
            raise ValueError("Projection document is not an indexed transcript.")
        if str(row["artifact_sha256"]) != transcript_hash:
            raise ValueError("Projection transcript hash does not match its index.")

    @staticmethod
    def _validate_sidecar(
        sidecar: dict[str, Any],
        *,
        conversation_id: str,
        recording_id: str,
    ) -> None:
        if sidecar.get("schema_version") != conversation_processing.SCHEMA_VERSION:
            raise ValueError("Unsupported processing sidecar schema.")
        if _opaque_uuid(
            sidecar.get("conversation_id"),
            field_name="conversation_id",
        ) != conversation_id:
            raise ValueError("Processing sidecar belongs to another conversation.")
        recording_ids = [
            _opaque_uuid(value, field_name="recording_id")
            for value in sidecar.get("recording_ids", [])
        ]
        if recording_id not in recording_ids:
            raise ValueError("Processing sidecar does not include this recording.")
        if not isinstance(sidecar.get("evaluations", []), list):
            raise ValueError("Processing sidecar evaluations must be a list.")
        if not isinstance(sidecar.get("review_decisions", []), list):
            raise ValueError("Processing sidecar review decisions must be a list.")

    def _legacy_identity_records(
        self,
        *,
        document_id: str,
        transcript_path: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with transcript_store.connect(self.root) as con:
            try:
                assignment_rows = con.execute(
                    """
                    SELECT *
                    FROM speaker_assignments
                    WHERE document_id = ? OR conversation_key = ?
                    ORDER BY speaker_label, id
                    """,
                    (document_id, str(transcript_path)),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table" not in str(exc):
                    raise
                assignment_rows = []
            assignments = [dict(row) for row in assignment_rows]
            contact_ids = sorted(
                {
                    str(row.get("contact_id") or "")
                    for row in assignments
                    if str(row.get("contact_id") or "")
                }
            )
            contacts: list[dict[str, Any]] = []
            if contact_ids:
                placeholders = ",".join("?" for _ in contact_ids)
                try:
                    contacts = [
                        dict(row)
                        for row in con.execute(
                            f"""
                            SELECT *
                            FROM contacts
                            WHERE id IN ({placeholders})
                            ORDER BY id
                            """,
                            contact_ids,
                        ).fetchall()
                    ]
                except sqlite3.OperationalError as exc:
                    if "no such table" not in str(exc):
                        raise
        return contacts, assignments

    @staticmethod
    def _conversation_snapshot(
        transcript: dict[str, Any],
        *,
        transcript_path: Path,
        transcript_hash: str,
        document_id: str,
    ) -> ConversationSnapshot:
        conversation_id = str(transcript["conversation_id"])
        recording_id = str(transcript["recording_id"])
        event = transcript.get("event")
        event = event if isinstance(event, dict) else {}
        utterances: list[UtteranceRecord] = []
        for ordinal, item in enumerate(transcript.get("utterances", [])):
            if not isinstance(item, dict):
                raise ValueError("Transcript utterances must be JSON objects.")
            utterances.append(
                UtteranceRecord(
                    utterance_id=_stable_uuid(
                        "utterance",
                        recording_id,
                        str(ordinal),
                    ),
                    conversation_id=conversation_id,
                    recording_id=recording_id,
                    speaker_label=str(
                        item.get("speaker")
                        or item.get("speaker_label")
                        or ""
                    ),
                    ordinal=ordinal,
                    start_ms=(
                        int(item["start"]) if item.get("start") is not None else None
                    ),
                    end_ms=(
                        int(item["end"]) if item.get("end") is not None else None
                    ),
                    text=str(item.get("text") or ""),
                    source_artifact_id=transcript_hash,
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"speaker", "speaker_label", "start", "end", "text"}
                    },
                )
            )
        return ConversationSnapshot(
            conversation=ConversationRecord(
                conversation_id=conversation_id,
                title=str(
                    transcript.get("transcript_title")
                    or event.get("summary")
                    or "Transcript"
                ),
                starts_at=str(transcript.get("recording_start") or ""),
                ends_at=str(transcript.get("recording_end") or ""),
                calendar_association_state="matched" if event else "",
                metadata={
                    "duration_seconds": transcript.get("duration_seconds"),
                    "event": event,
                    "source_transcript_sha256": transcript_hash,
                },
            ),
            recordings=(
                RecordingRecord(
                    recording_id=recording_id,
                    conversation_id=conversation_id,
                    transcript_document_id=document_id,
                    backend=str(transcript.get("backend") or ""),
                    model=str(transcript.get("model") or ""),
                    captured_at=str(transcript.get("recording_start") or ""),
                    metadata={
                        "source_artifact_path": str(transcript_path),
                        "source_transcript_sha256": transcript_hash,
                    },
                ),
            ),
            utterances=tuple(utterances),
        )

    @staticmethod
    def _processing_history(sidecar: dict[str, Any]) -> ProcessingHistory:
        conversation_id = str(sidecar["conversation_id"])
        evaluations: list[EvaluationRecord] = []
        evaluation_ids: set[str] = set()
        for item in sidecar.get("evaluations", []):
            if not isinstance(item, dict):
                raise ValueError("Processing evaluations must be JSON objects.")
            evaluation_id = _opaque_uuid(
                item.get("evaluation_id"),
                field_name="evaluation_id",
            )
            if evaluation_id in evaluation_ids:
                raise ValueError(f"Duplicate evaluation: {evaluation_id}.")
            evaluation_ids.add(evaluation_id)
            evaluations.append(
                EvaluationRecord(
                    evaluation_id=evaluation_id,
                    conversation_id=conversation_id,
                    evaluation_type=str(
                        item.get("evaluation_type")
                        or item.get("type")
                        or "speaker_identity"
                    ),
                    schema_version=str(item.get("schema_version") or ""),
                    status=str(item.get("status") or ""),
                    created_at=str(
                        item.get("created_at")
                        or item.get("evaluated_at")
                        or ""
                    ),
                    model_profile=str(item.get("model_profile") or ""),
                    input_artifact_id=str(item.get("input_artifact_id") or ""),
                    output_artifact_id=str(item.get("output_artifact_id") or ""),
                    evidence_bundle_id=str(item.get("evidence_bundle_id") or ""),
                    rubric_versions=(
                        dict(item.get("rubric_versions"))
                        if isinstance(item.get("rubric_versions"), dict)
                        else {}
                    ),
                    payload=dict(item),
                )
            )
        current = str(sidecar.get("current_evaluation_id") or "")
        if current and current not in evaluation_ids:
            raise ValueError("Current evaluation is absent from sidecar history.")
        decisions: list[ReviewDecisionRecord] = []
        for item in sidecar.get("review_decisions", []):
            if not isinstance(item, dict):
                raise ValueError("Review decisions must be JSON objects.")
            evaluation_id = _opaque_uuid(
                item.get("evaluation_id"),
                field_name="evaluation_id",
            )
            if evaluation_id not in evaluation_ids:
                raise ValueError("Review decision references an unknown evaluation.")
            decisions.append(
                ReviewDecisionRecord(
                    decision_id=_opaque_uuid(
                        item.get("decision_id"),
                        field_name="decision_id",
                    ),
                    evaluation_id=evaluation_id,
                    proposal_id=str(item.get("proposal_id") or ""),
                    action=str(item.get("action") or ""),
                    reviewer=str(item.get("reviewer") or ""),
                    method=str(item.get("decision_method") or ""),
                    decided_at=str(item.get("decided_at") or ""),
                    note=str(item.get("reviewer_note") or ""),
                    supersedes_decision_id=str(
                        item.get("supersedes_decision_id") or ""
                    ),
                    reviewer_asserted_identity=(
                        dict(item.get("reviewer_asserted_identity"))
                        if isinstance(
                            item.get("reviewer_asserted_identity"),
                            dict,
                        )
                        else {}
                    ),
                )
            )
        return ProcessingHistory(
            conversation_id=conversation_id,
            current_evaluation_id=current,
            evaluations=tuple(evaluations),
            review_decisions=tuple(decisions),
        )

    @staticmethod
    def _person_snapshots(
        contacts: list[dict[str, Any]],
    ) -> tuple[tuple[PersonSnapshot, ...], dict[str, str]]:
        snapshots: list[PersonSnapshot] = []
        person_ids: dict[str, str] = {}
        for contact in contacts:
            contact_id = str(contact.get("id") or "")
            person_id = _stable_uuid("legacy-contact-person", contact_id)
            person_ids[contact_id] = person_id
            email = str(contact.get("email") or "").strip()
            source_record_id = _stable_uuid(
                "legacy-contact-source",
                contact_id,
            )
            source_payload = {
                key: contact.get(key)
                for key in (
                    "id",
                    "label",
                    "email",
                    "external_ref",
                    "metadata_json",
                    "created_at",
                    "updated_at",
                )
            }
            identities: tuple[ExternalIdentityRecord, ...] = ()
            if email:
                normalized_email = email.casefold()
                identities = (
                    ExternalIdentityRecord(
                        external_identity_id=_stable_uuid(
                            "legacy-contact-email",
                            contact_id,
                            normalized_email,
                        ),
                        person_id=person_id,
                        source_record_id=source_record_id,
                        identity_kind="email",
                        normalized_value=normalized_email,
                        display_value=email,
                        authority="legacy_contact",
                        verified=False,
                    ),
                )
            snapshots.append(
                PersonSnapshot(
                    person=PersonRecord(
                        person_id=person_id,
                        status="source_record",
                        primary_name=str(contact.get("label") or ""),
                        metadata={"legacy_contact_id": contact_id},
                    ),
                    source_records=(
                        SourceRecord(
                            source_record_id=source_record_id,
                            person_id=person_id,
                            source_profile_id="local-transcript-store",
                            provider_kind="legacy_contact",
                            account_id="user-scoped",
                            tenant_id="",
                            external_ref=str(
                                contact.get("external_ref") or contact_id
                            ),
                            label=str(contact.get("label") or ""),
                            relationship_scope="local_review_history",
                            identifier_authority="legacy_contact_id",
                            observed_at=str(
                                contact.get("updated_at")
                                or contact.get("created_at")
                                or ""
                            ),
                            content_hash=_canonical_hash(source_payload),
                            source_event_at=str(
                                contact.get("created_at") or ""
                            ),
                            metadata={"legacy_contact": source_payload},
                        ),
                    ),
                    external_identities=identities,
                )
            )
        return tuple(snapshots), person_ids

    @staticmethod
    def _assignment_observations(
        assignments: list[dict[str, Any]],
        *,
        conversation_id: str,
        person_ids: dict[str, str],
    ) -> tuple[ObservationRecord, ...]:
        observations: list[ObservationRecord] = []
        for assignment in assignments:
            assignment_id = str(assignment.get("id") or "")
            contact_id = str(assignment.get("contact_id") or "")
            speaker_label = str(assignment.get("speaker_label") or "")
            payload = {
                key: assignment.get(key)
                for key in (
                    "id",
                    "conversation_key",
                    "document_id",
                    "speaker_label",
                    "contact_id",
                    "contact_label",
                    "status",
                    "confidence",
                    "evidence_json",
                    "created_at",
                    "updated_at",
                )
            }
            if contact_id in person_ids:
                payload["person_id"] = person_ids[contact_id]
            observations.append(
                ObservationRecord(
                    observation_id=_stable_uuid(
                        "legacy-speaker-assignment",
                        assignment_id,
                    ),
                    observation_type="speaker_assignment",
                    subject_type="diarized_speaker",
                    subject_id=_stable_uuid(
                        "diarized-speaker",
                        conversation_id,
                        speaker_label,
                    ),
                    source_type="legacy_speaker_assignment",
                    source_id=assignment_id,
                    conversation_id=conversation_id,
                    source_event_at=str(assignment.get("created_at") or ""),
                    observed_at=str(
                        assignment.get("updated_at")
                        or assignment.get("created_at")
                        or ""
                    ),
                    review_state=str(assignment.get("status") or "unreviewed"),
                    payload=payload,
                    content_hash=_canonical_hash(payload),
                )
            )
        return tuple(
            sorted(
                observations,
                key=lambda item: (item.observed_at, item.observation_id),
            )
        )

    def _verify_sources_unchanged(self, plan: ProjectionPlan) -> None:
        transcript_hash = _sha256_file(plan.transcript_path)
        sidecar_hash = (
            _sha256_file(plan.sidecar_path)
            if plan.sidecar_path is not None and plan.sidecar_path.is_file()
            else ""
        )
        if (
            transcript_hash != plan.source_transcript_sha256
            or sidecar_hash != plan.source_sidecar_sha256
        ):
            raise ValueError("Projection source changed after preview.")

    def _reconciles(self, plan: ProjectionPlan) -> bool:
        conversation_id = plan.processing_history.conversation_id
        snapshot = self.store.load_conversation_snapshot(conversation_id)
        history = self.store.load_processing_history(conversation_id)
        observations = self.store.load_observations(conversation_id)
        state = self.store.load_projection_state(
            PROJECTION_NAME,
            "conversation",
            conversation_id,
        )
        if (
            snapshot != plan.conversation_snapshot
            or history != plan.processing_history
            or observations != plan.observations
            or state is None
            or state.input_watermark != plan.input_watermark
        ):
            return False
        if any(
            self.store.load_person_snapshot(person.person.person_id) != person
            for person in plan.people
        ):
            return False
        proposal_count = sum(
            len(item.payload.get("proposals", []))
            if isinstance(item.payload.get("proposals"), list)
            else 0
            for item in history.evaluations
        )
        return {
            "assignments": len(observations),
            "contacts": len(plan.people),
            "conversations": 1,
            "decisions": len(history.review_decisions),
            "evaluations": len(history.evaluations),
            "proposals": proposal_count,
            "recordings": len(snapshot.recordings),
            "utterances": len(snapshot.utterances),
        } == plan.counts
