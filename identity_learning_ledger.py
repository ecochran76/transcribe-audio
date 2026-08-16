"""Append-only identity/contact correction ledger with rebuildable projections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

import transcript_store


EVENT_SCHEMA = "transcribe-audio.identity-ledger-event.v1"
PROJECTION_SCHEMA = "transcribe-audio.identity-ledger-projection.v1"
ONTOLOGY_SCHEMA = "transcribe-audio.identity-contact-ontology.v1"

EVENT_TYPES = {
    "ontology_registered",
    "source_record_observed",
    "external_identity_observed",
    "person_created",
    "source_record_linked",
    "source_record_corrected",
    "alias_added",
    "role_asserted",
    "role_corrected",
    "relationship_asserted",
    "relationship_corrected",
    "reconciliation_proposed",
    "reconciliation_decided",
    "people_merged",
    "person_split",
    "event_reversed",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{uuid5(NAMESPACE_URL, chr(31).join(parts))}"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: object) -> str:
    return str(value or "").strip()


def _list(value: object) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


@dataclass(frozen=True)
class AppendEventReceipt:
    event_id: str
    content_hash: str
    status: str


@dataclass(frozen=True)
class RebuildReceipt:
    event_count: int
    active_event_count: int
    projection_hash: str
    input_watermark: str


@dataclass(frozen=True)
class IdentityOntologyTerm:
    term_kind: str
    term_key: str
    parent_term_key: str = ""
    directionality: str = "not_applicable"
    inverse_term_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyReceipt:
    ontology_version_id: str
    content_hash: str
    term_count: int
    status: str


@dataclass(frozen=True)
class BaselineSourceRecord:
    source_record_id: str
    person_id: str
    source_profile_id: str
    provider_kind: str
    account_id: str
    tenant_id: str
    record_type: str
    external_ref: str
    label: str
    observed_at: str
    content_hash: str
    email: str = ""
    email_verified: bool = False
    phone: str = ""
    phone_verified: bool = False
    person_specific: bool = False
    shared_identifier: bool = False
    source_event_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReconciliationProposal:
    source_record_id: str
    reason_code: str
    candidate_person_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineReconciliationPlan:
    kept_source_record_ids: tuple[str, ...]
    duplicate_source_record_ids: tuple[str, ...]
    auto_links: tuple[tuple[str, str, str], ...]
    proposals: tuple[ReconciliationProposal, ...]
    provider_write_count: int = 0


class IdentityLearningLedger:
    """Own immutable correction events and deterministic current projections."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        self._require_v4()

    def _require_v4(self) -> None:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT schema_version, dirty
                FROM knowledge_store_state
                WHERE singleton = 1
                """
            ).fetchone()
        if row is None or int(row["schema_version"]) < 4 or bool(row["dirty"]):
            raise RuntimeError("Identity learning ledger requires knowledge schema v4.")

    def append_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        actor_id: str,
        occurred_at: str,
        idempotency_key: str,
        subject_type: str = "",
        subject_id: str = "",
        reverses_event_id: str = "",
    ) -> AppendEventReceipt:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported identity ledger event type: {event_type}.")
        if not all((_text(actor_id), _text(occurred_at), _text(idempotency_key))):
            raise ValueError("Identity ledger events require actor, time, and idempotency key.")
        if event_type == "event_reversed" and not _text(reverses_event_id):
            raise ValueError("Reversal events require reverses_event_id.")
        if event_type != "event_reversed" and _text(reverses_event_id):
            raise ValueError("Only reversal events may reference a reversed event.")
        self._validate_event_payload(event_type, payload)
        event_id = _stable_id("identity-event", _text(idempotency_key))
        core = {
            "id": event_id,
            "event_type": event_type,
            "event_schema": EVENT_SCHEMA,
            "occurred_at": _text(occurred_at),
            "actor_id": _text(actor_id),
            "idempotency_key": _text(idempotency_key),
            "subject_type": _text(subject_type),
            "subject_id": _text(subject_id),
            "reverses_event_id": _text(reverses_event_id),
            "payload": dict(payload),
        }
        content_hash = _canonical_hash(core)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT id, content_hash
                FROM knowledge_identity_ledger_events
                WHERE idempotency_key = ?
                """,
                (_text(idempotency_key),),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Identity event idempotency key was reused with "
                        "different content."
                    )
                return AppendEventReceipt(str(existing["id"]), content_hash, "unchanged")
            if reverses_event_id:
                reversed_row = con.execute(
                    """
                    SELECT event_type
                    FROM knowledge_identity_ledger_events
                    WHERE id = ?
                    """,
                    (reverses_event_id,),
                ).fetchone()
                if reversed_row is None:
                    raise ValueError("Reversal references an unknown identity event.")
                if str(reversed_row["event_type"]) == "event_reversed":
                    raise ValueError("Reversal events cannot reverse another reversal event.")
            con.execute(
                """
                INSERT INTO knowledge_identity_ledger_events (
                    id, event_type, event_schema, occurred_at, actor_id,
                    idempotency_key, subject_type, subject_id,
                    reverses_event_id, payload_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    EVENT_SCHEMA,
                    _text(occurred_at),
                    _text(actor_id),
                    _text(idempotency_key),
                    _text(subject_type),
                    _text(subject_id),
                    _text(reverses_event_id) or None,
                    _canonical_json(dict(payload)),
                    content_hash,
                    _text(occurred_at),
                ),
            )
            con.commit()
        return AppendEventReceipt(event_id, content_hash, "inserted")

    def register_ontology(
        self,
        *,
        schema_name: str,
        version: str,
        terms: Sequence[IdentityOntologyTerm],
    ) -> OntologyReceipt:
        if not _text(schema_name) or not _text(version) or not terms:
            raise ValueError("Ontology registration requires schema, version, and terms.")
        keys = {(term.term_kind, _text(term.term_key)) for term in terms}
        if len(keys) != len(terms):
            raise ValueError("Ontology terms must be unique within their kind.")
        for term in terms:
            if term.term_kind not in {"role", "relationship"}:
                raise ValueError("Ontology term kind must be role or relationship.")
            if term.directionality not in {
                "directional",
                "symmetric",
                "not_applicable",
            }:
                raise ValueError("Ontology directionality is invalid.")
            if term.parent_term_key and (
                term.term_kind,
                term.parent_term_key,
            ) not in keys:
                raise ValueError(f"Ontology term has unknown parent: {term.parent_term_key}.")
            if term.inverse_term_key and (
                "relationship",
                term.inverse_term_key,
            ) not in keys:
                raise ValueError(f"Ontology term has unknown inverse: {term.inverse_term_key}.")
            if term.term_kind == "role" and term.directionality != "not_applicable":
                raise ValueError("Role ontology terms do not have directionality.")
            if term.directionality == "symmetric" and term.inverse_term_key:
                raise ValueError("Symmetric relationship terms do not name an inverse.")
        payload = {
            "schema_name": _text(schema_name),
            "version": _text(version),
            "terms": [asdict(term) for term in terms],
        }
        content_hash = _canonical_hash(payload)
        ontology_id = _stable_id(
            "identity-ontology",
            _text(schema_name),
            _text(version),
            content_hash,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                """
                SELECT id, content_hash
                FROM knowledge_identity_ontology_versions
                WHERE schema_name = ? AND version = ?
                """,
                (_text(schema_name), _text(version)),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Ontology version already exists with different content.")
                return OntologyReceipt(str(existing["id"]), content_hash, len(terms), "unchanged")
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_identity_ontology_versions (
                        id, schema_name, version, content_hash, payload_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ontology_id,
                        _text(schema_name),
                        _utc_now(),
                        content_hash,
                        _canonical_json(payload),
                        _text(version),
                    ),
                )
                for term in terms:
                    con.execute(
                        """
                        INSERT INTO knowledge_identity_ontology_terms (
                            ontology_version_id, term_kind, term_key,
                            parent_term_key, directionality, inverse_term_key,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ontology_id,
                            term.term_kind,
                            _text(term.term_key),
                            _text(term.parent_term_key),
                            term.directionality,
                            _text(term.inverse_term_key),
                            _canonical_json(dict(term.metadata)),
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return OntologyReceipt(ontology_id, content_hash, len(terms), "inserted")

    def ontology_terms(self, ontology_version_id: str) -> tuple[dict[str, Any], ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_identity_ontology_terms
                WHERE ontology_version_id = ?
                ORDER BY rowid
                """,
                (ontology_version_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def rebuild(self) -> RebuildReceipt:
        rows = self._load_events()
        reversed_ids = {
            str(row["reverses_event_id"])
            for row in rows
            if row["event_type"] == "event_reversed"
        }
        active = [
            row
            for row in rows
            if row["event_type"] != "event_reversed" and row["id"] not in reversed_ids
        ]
        people: dict[str, dict[str, Any]] = {}
        sources: dict[str, dict[str, Any]] = {}
        external_identities: dict[str, dict[str, Any]] = {}
        roles: dict[str, dict[str, Any]] = {}
        relationships: dict[str, dict[str, Any]] = {}
        reconciliations: dict[str, dict[str, Any]] = {}
        watermark = str(rows[-1]["id"]) if rows else "empty"
        built_at = str(rows[-1]["occurred_at"]) if rows else "1970-01-01T00:00:00Z"
        for row in active:
            payload = json.loads(str(row["payload_json"]))
            event_type = str(row["event_type"])
            event_id = str(row["id"])
            if event_type == "person_created":
                person_id = self._required(payload, "person_id")
                if person_id in people:
                    raise ValueError(f"Person was created more than once: {person_id}.")
                people[person_id] = {
                    "person_id": person_id,
                    "status": _text(payload.get("status")) or "provisional",
                    "primary_name": _text(payload.get("primary_name")),
                    "aliases": [],
                    "merged_into_person_id": "",
                    "metadata": dict(payload.get("metadata") or {}),
                }
            elif event_type == "source_record_observed":
                source_id = self._required(payload, "source_record_id")
                sources[source_id] = self._source_projection(payload, event_id)
                person_id = sources[source_id]["person_id"]
                if person_id:
                    self._require_member(people, person_id, "person")
            elif event_type == "external_identity_observed":
                identity_id = self._required(payload, "external_identity_id")
                source_id = self._required(payload, "source_record_id")
                self._require_member(sources, source_id, "source record")
                external_identities[identity_id] = self._external_projection(payload)
                person_id = external_identities[identity_id]["person_id"]
                if person_id:
                    self._require_member(people, person_id, "person")
            elif event_type == "source_record_linked":
                source_id = self._required(payload, "source_record_id")
                person_id = self._required(payload, "person_id")
                self._require_member(sources, source_id, "source record")
                self._require_member(people, person_id, "person")
                sources[source_id]["person_id"] = person_id
                sources[source_id]["resolution_status"] = "linked"
                for identity in external_identities.values():
                    if identity["source_record_id"] == source_id:
                        identity["person_id"] = person_id
                        identity["status"] = "linked"
            elif event_type == "source_record_corrected":
                source_id = self._required(payload, "source_record_id")
                self._require_member(sources, source_id, "source record")
                self._apply_correction(
                    sources[source_id],
                    payload,
                    allowed={"label", "resolution_status", "metadata"},
                    kind="source record",
                )
            elif event_type == "alias_added":
                person_id = self._required(payload, "person_id")
                self._require_member(people, person_id, "person")
                alias = self._required(payload, "alias")
                if alias not in people[person_id]["aliases"]:
                    people[person_id]["aliases"].append(alias)
            elif event_type == "role_asserted":
                role_id = self._required(payload, "role_id")
                person_id = self._required(payload, "person_id")
                self._require_member(people, person_id, "person")
                roles[role_id] = self._role_projection(payload)
            elif event_type == "role_corrected":
                role_id = self._required(payload, "role_id")
                self._require_member(roles, role_id, "role")
                self._apply_correction(
                    roles[role_id],
                    payload,
                    allowed={
                        "role_type",
                        "organization_id",
                        "project_id",
                        "matter_id",
                        "conversation_id",
                        "starts_at",
                        "ends_at",
                        "status",
                        "evidence_ids",
                        "metadata",
                    },
                    kind="role",
                )
            elif event_type == "relationship_asserted":
                relationship_id = self._required(payload, "relationship_id")
                relationships[relationship_id] = self._relationship_projection(payload)
            elif event_type == "relationship_corrected":
                relationship_id = self._required(payload, "relationship_id")
                self._require_member(relationships, relationship_id, "relationship")
                self._apply_correction(
                    relationships[relationship_id],
                    payload,
                    allowed={
                        "relationship_type",
                        "subject_type",
                        "subject_id",
                        "object_type",
                        "object_id",
                        "directionality",
                        "inverse_relationship_id",
                        "starts_at",
                        "ends_at",
                        "status",
                        "evidence_ids",
                        "metadata",
                    },
                    kind="relationship",
                )
            elif event_type == "reconciliation_proposed":
                proposal_id = self._required(payload, "proposal_id")
                reconciliations[proposal_id] = self._reconciliation_projection(payload)
            elif event_type == "reconciliation_decided":
                proposal_id = self._required(payload, "proposal_id")
                self._require_member(reconciliations, proposal_id, "proposal")
                reconciliations[proposal_id].update(
                    decision_status=_text(payload.get("decision_status")),
                    decided_by=_text(payload.get("decided_by")),
                    decided_at=_text(payload.get("decided_at")),
                )
            elif event_type == "people_merged":
                target = self._required(payload, "target_person_id")
                self._require_member(people, target, "person")
                for source in map(_text, _list(payload.get("source_person_ids"))):
                    self._require_member(people, source, "person")
                    if source == target:
                        raise ValueError("A person cannot be merged into itself.")
                    people[source]["merged_into_person_id"] = target
                    people[source]["status"] = "merged"
                    for source_record in sources.values():
                        if source_record["person_id"] == source:
                            source_record["person_id"] = target
                            source_record["resolution_status"] = "linked"
                    for identity in external_identities.values():
                        if identity["person_id"] == source:
                            identity["person_id"] = target
                            identity["status"] = "linked"
            elif event_type == "person_split":
                source_person = self._required(payload, "source_person_id")
                target_person = self._required(payload, "target_person_id")
                self._require_member(people, source_person, "person")
                self._require_member(people, target_person, "person")
                record_ids = tuple(map(_text, _list(payload.get("source_record_ids"))))
                if not record_ids:
                    raise ValueError("Person split requires explicit source_record_ids.")
                for source_id in record_ids:
                    self._require_member(sources, source_id, "source record")
                    if sources[source_id]["person_id"] != source_person:
                        raise ValueError(
                            "Person split source record is not linked to its "
                            "source person."
                        )
                    sources[source_id]["person_id"] = target_person
                    sources[source_id]["resolution_status"] = "linked"
                    for identity in external_identities.values():
                        if identity["source_record_id"] == source_id:
                            identity["person_id"] = target_person
                            identity["status"] = "linked"
        semantic = {
            "people": people,
            "sources": sources,
            "external_identities": external_identities,
            "roles": roles,
            "relationships": relationships,
            "reconciliations": reconciliations,
            "input_watermark": watermark,
            "projection_schema": PROJECTION_SCHEMA,
        }
        projection_hash = _canonical_hash(semantic)
        self._replace_projections(
            people=people,
            sources=sources,
            external_identities=external_identities,
            roles=roles,
            relationships=relationships,
            reconciliations=reconciliations,
            watermark=watermark,
            built_at=built_at,
        )
        return RebuildReceipt(
            event_count=len(rows),
            active_event_count=len(active),
            projection_hash=projection_hash,
            input_watermark=watermark,
        )

    def projection_snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        tables = {
            "people": ("knowledge_identity_people_projection", "person_id"),
            "sources": ("knowledge_identity_source_projection", "source_record_id"),
            "external_identities": (
                "knowledge_identity_external_projection",
                "external_identity_id",
            ),
            "roles": ("knowledge_identity_role_projection", "role_id"),
            "relationships": (
                "knowledge_identity_relationship_projection",
                "relationship_id",
            ),
            "reconciliations": (
                "knowledge_identity_reconciliation_projection",
                "proposal_id",
            ),
        }
        snapshot: dict[str, dict[str, dict[str, Any]]] = {}
        with transcript_store.connect(self.root) as con:
            for name, (table, key) in tables.items():
                rows = con.execute(f"SELECT * FROM {table} ORDER BY {key}").fetchall()
                snapshot[name] = {str(row[key]): dict(row) for row in rows}
        return snapshot

    def reconcile_baseline(
        self,
        records: Iterable[BaselineSourceRecord],
    ) -> BaselineReconciliationPlan:
        kept: list[BaselineSourceRecord] = []
        duplicates: list[str] = []
        seen_scope: set[tuple[str, ...]] = set()
        for record in records:
            scope = (
                _text(record.provider_kind),
                _text(record.source_profile_id),
                _text(record.account_id),
                _text(record.tenant_id),
                _text(record.record_type),
                _text(record.external_ref),
            )
            if not all((scope[0], scope[1], scope[4], scope[5])):
                raise ValueError("Baseline source records require a complete source scope.")
            if scope in seen_scope:
                duplicates.append(record.source_record_id)
                continue
            seen_scope.add(scope)
            kept.append(record)
        identifiers: dict[tuple[str, str], set[str]] = {}
        for record in kept:
            if not record.person_id or record.shared_identifier or not record.person_specific:
                continue
            for kind, value, verified in self._record_identifiers(record):
                if verified and value:
                    identifiers.setdefault((kind, value), set()).add(record.person_id)
        auto_links: list[tuple[str, str, str]] = []
        proposals: list[ReconciliationProposal] = []
        for record in kept:
            if record.person_id:
                continue
            if record.shared_identifier:
                proposals.append(
                    ReconciliationProposal(
                        record.source_record_id,
                        "shared_identifier_requires_review",
                    )
                )
                continue
            candidate_reasons: dict[str, set[str]] = {}
            for kind, value, verified in self._record_identifiers(record):
                if not (record.person_specific and verified and value):
                    continue
                for person_id in identifiers.get((kind, value), set()):
                    candidate_reasons.setdefault(person_id, set()).add(f"verified_{kind}")
            if len(candidate_reasons) == 1:
                person_id = next(iter(candidate_reasons))
                reason = sorted(candidate_reasons[person_id])[0]
                auto_links.append((record.source_record_id, person_id, reason))
            elif len(candidate_reasons) > 1:
                proposals.append(
                    ReconciliationProposal(
                        record.source_record_id,
                        "conflicting_authoritative_identifiers",
                        tuple(sorted(candidate_reasons)),
                    )
                )
            else:
                proposals.append(
                    ReconciliationProposal(
                        record.source_record_id,
                        "insufficient_authoritative_identifier",
                    )
                )
        return BaselineReconciliationPlan(
            kept_source_record_ids=tuple(sorted(item.source_record_id for item in kept)),
            duplicate_source_record_ids=tuple(sorted(duplicates)),
            auto_links=tuple(sorted(auto_links)),
            proposals=tuple(sorted(proposals, key=lambda item: item.source_record_id)),
        )

    def _load_events(self) -> list[sqlite3.Row]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_identity_ledger_events
                ORDER BY occurred_at, id
                """
            ).fetchall()
        for row in rows:
            core = {
                "id": str(row["id"]),
                "event_type": str(row["event_type"]),
                "event_schema": str(row["event_schema"]),
                "occurred_at": str(row["occurred_at"]),
                "actor_id": str(row["actor_id"]),
                "idempotency_key": str(row["idempotency_key"]),
                "subject_type": str(row["subject_type"]),
                "subject_id": str(row["subject_id"]),
                "reverses_event_id": _text(row["reverses_event_id"]),
                "payload": json.loads(str(row["payload_json"])),
            }
            if _canonical_hash(core) != str(row["content_hash"]):
                raise RuntimeError(f"Identity ledger event hash drifted: {row['id']}.")
        return rows

    @staticmethod
    def _required(payload: Mapping[str, Any], field_name: str) -> str:
        value = _text(payload.get(field_name))
        if not value:
            raise ValueError(f"Identity event requires {field_name}.")
        return value

    @classmethod
    def _validate_event_payload(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if event_type == "source_record_observed":
            if "email" in payload or "phone" in payload:
                raise ValueError(
                    "Source record events cannot store raw email or phone identifiers; "
                    "append a hashed external identity event instead."
                )
            cls._source_projection(payload, "validation")
        elif event_type == "external_identity_observed":
            cls._external_projection(payload)
        elif event_type == "person_created":
            cls._required(payload, "person_id")
            cls._required(payload, "primary_name")
        elif event_type == "alias_added":
            cls._required(payload, "person_id")
            cls._required(payload, "alias")
        elif event_type == "role_asserted":
            cls._required(payload, "role_id")
            cls._required(payload, "person_id")
            cls._required(payload, "role_type")
        elif event_type == "relationship_asserted":
            for field_name in (
                "relationship_id",
                "relationship_type",
                "subject_type",
                "subject_id",
                "object_type",
                "object_id",
            ):
                cls._required(payload, field_name)
            if _text(payload.get("directionality")) not in {
                "",
                "directional",
                "symmetric",
            }:
                raise ValueError("Relationship directionality is invalid.")
        elif event_type in {
            "source_record_corrected",
            "role_corrected",
            "relationship_corrected",
        }:
            identifier = {
                "source_record_corrected": "source_record_id",
                "role_corrected": "role_id",
                "relationship_corrected": "relationship_id",
            }[event_type]
            cls._required(payload, identifier)
            if not isinstance(payload.get("changes"), Mapping) or not payload["changes"]:
                raise ValueError("Correction events require a non-empty changes object.")
        elif event_type == "people_merged":
            cls._required(payload, "target_person_id")
            if not tuple(filter(_text, _list(payload.get("source_person_ids")))):
                raise ValueError("People merge requires source_person_ids.")
        elif event_type == "person_split":
            cls._required(payload, "source_person_id")
            cls._required(payload, "target_person_id")
            if not tuple(filter(_text, _list(payload.get("source_record_ids")))):
                raise ValueError("Person split requires explicit source_record_ids.")

    @staticmethod
    def _require_member(values: Mapping[str, Any], key: str, kind: str) -> None:
        if key not in values:
            raise ValueError(f"Identity event references unknown {kind}: {key}.")

    @staticmethod
    def _source_projection(payload: Mapping[str, Any], event_id: str) -> dict[str, Any]:
        required = (
            "source_record_id",
            "source_profile_id",
            "provider_kind",
            "record_type",
            "external_ref",
            "observed_at",
            "content_hash",
        )
        missing = [field for field in required if not _text(payload.get(field))]
        if missing:
            raise ValueError(f"Source record event is missing: {', '.join(missing)}.")
        person_id = _text(payload.get("person_id"))
        return {
            "source_record_id": _text(payload["source_record_id"]),
            "person_id": person_id,
            "source_profile_id": _text(payload["source_profile_id"]),
            "provider_kind": _text(payload["provider_kind"]),
            "account_id": _text(payload.get("account_id")),
            "tenant_id": _text(payload.get("tenant_id")),
            "record_type": _text(payload["record_type"]),
            "external_ref": _text(payload["external_ref"]),
            "label": _text(payload.get("label")),
            "source_event_at": _text(payload.get("source_event_at")),
            "observed_at": _text(payload["observed_at"]),
            "content_hash": _text(payload["content_hash"]),
            "resolution_status": "linked" if person_id else "unresolved",
            "metadata": dict(payload.get("metadata") or {}),
            "source_event_id": event_id,
        }

    @staticmethod
    def _external_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "external_identity_id",
            "source_record_id",
            "provider_kind",
            "identity_type",
            "identity_value_hash",
            "observed_at",
        )
        missing = [field for field in required if not _text(payload.get(field))]
        if missing:
            raise ValueError(f"External identity event is missing: {', '.join(missing)}.")
        identity_hash = _text(payload["identity_value_hash"])
        if len(identity_hash) != 64 or any(
            character not in "0123456789abcdef" for character in identity_hash
        ):
            raise ValueError("External identity values must use a lowercase SHA-256 hash.")
        person_id = _text(payload.get("person_id"))
        return {
            "external_identity_id": _text(payload["external_identity_id"]),
            "source_record_id": _text(payload["source_record_id"]),
            "person_id": person_id,
            "provider_kind": _text(payload["provider_kind"]),
            "account_id": _text(payload.get("account_id")),
            "tenant_id": _text(payload.get("tenant_id")),
            "identity_type": _text(payload["identity_type"]),
            "identity_value_hash": identity_hash,
            "person_specific": bool(payload.get("person_specific")),
            "verified": bool(payload.get("verified")),
            "shared_identifier": bool(payload.get("shared_identifier")),
            "observed_at": _text(payload["observed_at"]),
            "valid_from": _text(payload.get("valid_from")),
            "valid_to": _text(payload.get("valid_to")),
            "status": "linked" if person_id else "unresolved",
            "metadata": dict(payload.get("metadata") or {}),
        }

    @staticmethod
    def _apply_correction(
        projection: dict[str, Any],
        payload: Mapping[str, Any],
        *,
        allowed: set[str],
        kind: str,
    ) -> None:
        changes = payload.get("changes")
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError(f"{kind.title()} correction requires changes.")
        unsupported = set(changes) - allowed
        if unsupported:
            raise ValueError(
                f"{kind.title()} correction contains unsupported fields: "
                f"{', '.join(sorted(unsupported))}."
            )
        for key, value in changes.items():
            projection[key] = (
                dict(value) if key == "metadata" and isinstance(value, Mapping)
                else _list(value) if key == "evidence_ids"
                else _text(value)
            )

    @staticmethod
    def _role_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "role_id": _text(payload["role_id"]),
            "person_id": _text(payload["person_id"]),
            "role_type": _text(payload.get("role_type")),
            "organization_id": _text(payload.get("organization_id")),
            "project_id": _text(payload.get("project_id")),
            "matter_id": _text(payload.get("matter_id")),
            "conversation_id": _text(payload.get("conversation_id")),
            "starts_at": _text(payload.get("starts_at")),
            "ends_at": _text(payload.get("ends_at")),
            "status": _text(payload.get("status")) or "proposed",
            "evidence_ids": _list(payload.get("evidence_ids")),
            "metadata": dict(payload.get("metadata") or {}),
        }

    @staticmethod
    def _relationship_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "relationship_id": _text(payload["relationship_id"]),
            "relationship_type": _text(payload.get("relationship_type")),
            "subject_type": _text(payload.get("subject_type")),
            "subject_id": _text(payload.get("subject_id")),
            "object_type": _text(payload.get("object_type")),
            "object_id": _text(payload.get("object_id")),
            "directionality": _text(payload.get("directionality")) or "directional",
            "inverse_relationship_id": _text(payload.get("inverse_relationship_id")),
            "starts_at": _text(payload.get("starts_at")),
            "ends_at": _text(payload.get("ends_at")),
            "status": _text(payload.get("status")) or "proposed",
            "evidence_ids": _list(payload.get("evidence_ids")),
            "metadata": dict(payload.get("metadata") or {}),
        }

    @staticmethod
    def _reconciliation_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "proposal_id": _text(payload["proposal_id"]),
            "proposal_type": _text(payload.get("proposal_type")),
            "source_record_ids": _list(payload.get("source_record_ids")),
            "candidate_person_ids": _list(payload.get("candidate_person_ids")),
            "reason_codes": _list(payload.get("reason_codes")),
            "confidence": payload.get("confidence"),
            "decision_status": _text(payload.get("decision_status")) or "pending",
            "decided_by": "",
            "decided_at": "",
            "metadata": dict(payload.get("metadata") or {}),
        }

    @staticmethod
    def _record_identifiers(
        record: BaselineSourceRecord,
    ) -> tuple[tuple[str, str, bool], ...]:
        phone = "".join(character for character in record.phone if character.isdigit())
        return (
            ("email", record.email.strip().casefold(), record.email_verified),
            ("phone", phone, record.phone_verified),
        )

    def _replace_projections(
        self,
        *,
        people: Mapping[str, Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        external_identities: Mapping[str, Mapping[str, Any]],
        roles: Mapping[str, Mapping[str, Any]],
        relationships: Mapping[str, Mapping[str, Any]],
        reconciliations: Mapping[str, Mapping[str, Any]],
        watermark: str,
        built_at: str,
    ) -> None:
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                for table in (
                    "knowledge_identity_reconciliation_projection",
                    "knowledge_identity_relationship_projection",
                    "knowledge_identity_role_projection",
                    "knowledge_identity_external_projection",
                    "knowledge_identity_source_projection",
                    "knowledge_identity_people_projection",
                ):
                    con.execute(f"DELETE FROM {table}")
                for person_id in sorted(people):
                    value = people[person_id]
                    con.execute(
                        """
                        INSERT INTO knowledge_identity_people_projection (
                            person_id, status, primary_name, aliases_json,
                            merged_into_person_id, input_watermark,
                            metadata_json, built_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            person_id,
                            value["status"],
                            value["primary_name"],
                            _canonical_json(sorted(value["aliases"])),
                            value["merged_into_person_id"],
                            watermark,
                            _canonical_json(value["metadata"]),
                            built_at,
                        ),
                    )
                for source_id in sorted(sources):
                    value = sources[source_id]
                    con.execute(
                        """
                        INSERT INTO knowledge_identity_source_projection (
                            source_record_id, person_id, source_profile_id,
                            provider_kind, account_id, tenant_id, record_type,
                            external_ref, label, source_event_at, observed_at, content_hash,
                            resolution_status, input_watermark, metadata_json,
                            built_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            source_id,
                            value["person_id"],
                            value["source_profile_id"],
                            value["provider_kind"],
                            value["account_id"],
                            value["tenant_id"],
                            value["record_type"],
                            value["external_ref"],
                            value["label"],
                            value["source_event_at"],
                            value["observed_at"],
                            value["content_hash"],
                            value["resolution_status"],
                            watermark,
                            _canonical_json(
                                {
                                    **value["metadata"],
                                    "source_event_id": value["source_event_id"],
                                }
                            ),
                            built_at,
                        ),
                    )
                self._insert_external_identities(
                    con,
                    external_identities,
                    watermark,
                    built_at,
                )
                self._insert_roles(con, roles, watermark, built_at)
                self._insert_relationships(con, relationships, watermark, built_at)
                self._insert_reconciliations(con, reconciliations, watermark, built_at)
                con.commit()
            except Exception:
                con.rollback()
                raise

    @staticmethod
    def _insert_external_identities(
        con: sqlite3.Connection,
        identities: Mapping[str, Mapping[str, Any]],
        watermark: str,
        built_at: str,
    ) -> None:
        for identity_id in sorted(identities):
            value = identities[identity_id]
            con.execute(
                """
                INSERT INTO knowledge_identity_external_projection (
                    external_identity_id, source_record_id, person_id,
                    provider_kind, account_id, tenant_id, identity_type,
                    identity_value_hash, person_specific, verified,
                    shared_identifier, observed_at, valid_from, valid_to,
                    status, input_watermark, metadata_json, built_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity_id,
                    value["source_record_id"],
                    value["person_id"],
                    value["provider_kind"],
                    value["account_id"],
                    value["tenant_id"],
                    value["identity_type"],
                    value["identity_value_hash"],
                    int(value["person_specific"]),
                    int(value["verified"]),
                    int(value["shared_identifier"]),
                    value["observed_at"],
                    value["valid_from"],
                    value["valid_to"],
                    value["status"],
                    watermark,
                    _canonical_json(value["metadata"]),
                    built_at,
                ),
            )

    @staticmethod
    def _insert_roles(
        con: sqlite3.Connection,
        roles: Mapping[str, Mapping[str, Any]],
        watermark: str,
        built_at: str,
    ) -> None:
        for role_id in sorted(roles):
            value = roles[role_id]
            con.execute(
                """
                INSERT INTO knowledge_identity_role_projection (
                    role_id, person_id, role_type, organization_id, project_id,
                    matter_id, conversation_id, starts_at, ends_at, status,
                    evidence_ids_json, input_watermark, metadata_json, built_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role_id,
                    value["person_id"],
                    value["role_type"],
                    value["organization_id"],
                    value["project_id"],
                    value["matter_id"],
                    value["conversation_id"],
                    value["starts_at"],
                    value["ends_at"],
                    value["status"],
                    _canonical_json(value["evidence_ids"]),
                    watermark,
                    _canonical_json(value["metadata"]),
                    built_at,
                ),
            )

    @staticmethod
    def _insert_relationships(
        con: sqlite3.Connection,
        relationships: Mapping[str, Mapping[str, Any]],
        watermark: str,
        built_at: str,
    ) -> None:
        for relationship_id in sorted(relationships):
            value = relationships[relationship_id]
            con.execute(
                """
                INSERT INTO knowledge_identity_relationship_projection (
                    relationship_id, relationship_type, subject_type,
                    subject_id, object_type, object_id, directionality,
                    inverse_relationship_id, starts_at, ends_at, status,
                    evidence_ids_json, input_watermark, metadata_json, built_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relationship_id,
                    value["relationship_type"],
                    value["subject_type"],
                    value["subject_id"],
                    value["object_type"],
                    value["object_id"],
                    value["directionality"],
                    value["inverse_relationship_id"],
                    value["starts_at"],
                    value["ends_at"],
                    value["status"],
                    _canonical_json(value["evidence_ids"]),
                    watermark,
                    _canonical_json(value["metadata"]),
                    built_at,
                ),
            )

    @staticmethod
    def _insert_reconciliations(
        con: sqlite3.Connection,
        reconciliations: Mapping[str, Mapping[str, Any]],
        watermark: str,
        built_at: str,
    ) -> None:
        for proposal_id in sorted(reconciliations):
            value = reconciliations[proposal_id]
            con.execute(
                """
                INSERT INTO knowledge_identity_reconciliation_projection (
                    proposal_id, proposal_type, source_record_ids_json,
                    candidate_person_ids_json, reason_codes_json, confidence,
                    decision_status, decided_by, decided_at, input_watermark,
                    metadata_json, built_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    value["proposal_type"],
                    _canonical_json(value["source_record_ids"]),
                    _canonical_json(value["candidate_person_ids"]),
                    _canonical_json(value["reason_codes"]),
                    value["confidence"],
                    value["decision_status"],
                    value["decided_by"],
                    value["decided_at"],
                    watermark,
                    _canonical_json(value["metadata"]),
                    built_at,
                ),
            )
