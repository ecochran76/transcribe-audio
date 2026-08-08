from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import transcript_store
from conversation_knowledge_projection import (
    APPLY_APPROVAL_TOKEN,
    ConversationKnowledgeProjector,
)
from conversation_knowledge_store import (
    LATEST_SCHEMA_VERSION,
    ConversationKnowledgeStore,
)


CONTRACT_VERSION = "transcribe-audio.speaker-identity-shadow-contract.v1"
ACTIVATION_MANIFEST_VERSION = "transcribe-audio.plan0059-activation-manifest.v1"
ACTIVATION_RECEIPT_VERSION = "transcribe-audio.plan0059-activation-receipt.v1"
P1_MANIFEST_VERSION = "transcribe-audio.plan0059-shadow-store-manifest.v1"
P1_RECEIPT_VERSION = "transcribe-audio.plan0059-shadow-store-receipt.v1"
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
GIT_OID_RE = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
SPEAKER_REF_RE = re.compile(r"^SPEAKER_[1-9][0-9]*$")
ISO_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")

NEGATIVE_ACTION_FIELDS = (
    "apply_live_speaker_assignment",
    "create_or_mutate_person",
    "create_or_mutate_contact",
    "create_or_mutate_role",
    "create_or_mutate_relationship",
    "mutate_acoustic_profile_or_reference",
    "write_provider_record",
    "write_graphiti",
    "enable_default_integration",
    "change_knowledge_authority",
    "run_historical_processing",
)

STATE_TRANSITIONS = {
    "pending": {"evidence_collecting"},
    "evidence_collecting": {"evidence_ready", "evidence_partial", "evidence_failed"},
    "evidence_ready": {"proposed", "abstained"},
    "evidence_partial": {"proposed", "abstained"},
    "evidence_failed": {"abstained"},
    "proposed": {"review_required"},
    "abstained": {"review_required"},
    "review_required": {"accepted", "rejected", "unresolved"},
    "accepted": set(),
    "rejected": set(),
    "unresolved": set(),
}

CONFIDENCE_CAPS = {
    "partial_provider_failure": 0.75,
    "dependent_evidence": 0.70,
    "stale_evidence": 0.60,
    "scope_conflict": 0.50,
    "material_contradiction": 0.49,
}


class IdentityOrchestrationError(ValueError):
    """Raised when a shadow identity artifact violates a frozen contract."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(reason_code: str, message: str) -> None:
    raise IdentityOrchestrationError(reason_code, message)


def _opaque(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not OPAQUE_ID_RE.fullmatch(normalized):
        _fail("invalid_identifier", f"{field_name} must be an opaque identifier.")
    return normalized


def _sha256(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not SHA256_RE.fullmatch(normalized):
        _fail("invalid_hash", f"{field_name} must be a lowercase SHA-256 digest.")
    return normalized


def _git_oid(value: str) -> str:
    normalized = str(value or "").strip()
    if not GIT_OID_RE.fullmatch(normalized):
        _fail("invalid_repository_head", "repository_head must be a full Git object ID.")
    return normalized


def _time(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not ISO_TIME_RE.fullmatch(normalized):
        _fail("invalid_time", f"{field_name} must be an ISO-8601 timestamp with offset.")
    return normalized


def _unique(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(value or "").strip() for value in values)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        _fail("duplicate_identifier", f"{field_name} must be non-empty and unique.")
    return normalized


def negative_action_vector() -> dict[str, bool]:
    return {field_name: False for field_name in NEGATIVE_ACTION_FIELDS}


def validate_negative_actions(value: Mapping[str, Any]) -> dict[str, bool]:
    expected = negative_action_vector()
    if dict(value) != expected:
        _fail("forbidden_mutation", "Every Plan 0059 negative action must remain false.")
    return expected


@dataclass(frozen=True)
class EvidenceScope:
    source_type: str
    source_profile: str
    account_id: str
    tenant_id: str
    capabilities: tuple[str, ...]
    as_of: str
    max_records: int
    max_characters: int
    max_per_source: int
    max_provider_calls: int
    max_relationship_hops: int

    def __post_init__(self) -> None:
        for field_name in ("source_type", "source_profile", "account_id", "tenant_id"):
            _opaque(getattr(self, field_name), field_name)
        _unique(self.capabilities, "capabilities")
        _time(self.as_of, "as_of")
        limits = (
            self.max_records,
            self.max_characters,
            self.max_per_source,
            self.max_provider_calls,
            self.max_relationship_hops,
        )
        if any(isinstance(value, bool) or value < 0 for value in limits):
            _fail("invalid_budget", "Retrieval budgets must be non-negative integers.")


@dataclass(frozen=True)
class EvidenceLineage:
    evidence_id: str
    source_record_id: str
    independence_group: str
    source_type: str
    source_event_at: str
    observed_at: str
    retrieved_at: str
    content_sha256: str
    derived_from_evidence_ids: tuple[str, ...] = ()
    proposed_by_current_evaluation: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "source_record_id",
            "independence_group",
            "source_type",
        ):
            _opaque(getattr(self, field_name), field_name)
        for field_name in ("source_event_at", "observed_at", "retrieved_at"):
            _time(getattr(self, field_name), field_name)
        _sha256(self.content_sha256, "content_sha256")
        _unique(self.derived_from_evidence_ids, "derived_from_evidence_ids") if self.derived_from_evidence_ids else ()


@dataclass(frozen=True)
class AcousticSpeakerEvidence:
    speaker_ref: str
    disposition: str
    acoustic_subject_id: str | None
    score: float
    confidence_band: str
    supporting_unit_count: int
    opposing_unit_count: int
    insufficient_unit_count: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not SPEAKER_REF_RE.fullmatch(self.speaker_ref):
            _fail("invalid_speaker_ref", "Acoustic evidence has an invalid speaker_ref.")
        if self.disposition not in {"assign", "review", "abstain"}:
            _fail("invalid_disposition", "Acoustic evidence disposition is invalid.")
        if not 0.0 <= float(self.score) <= 1.0:
            _fail("invalid_score", "Acoustic score must be between zero and one.")
        if self.disposition == "abstain":
            if self.acoustic_subject_id is not None or self.confidence_band != "none":
                _fail("abstention_carries_identity", "Acoustic abstention cannot carry identity.")
        else:
            _opaque(str(self.acoustic_subject_id or ""), "acoustic_subject_id")
            if self.confidence_band not in {"low", "medium", "high"}:
                _fail("invalid_confidence_band", "Acoustic confidence band is invalid.")
        counts = (self.supporting_unit_count, self.opposing_unit_count, self.insufficient_unit_count)
        if any(isinstance(value, bool) or value < 0 for value in counts):
            _fail("invalid_unit_count", "Acoustic unit counts must be non-negative integers.")
        _unique(self.evidence_ids, "acoustic evidence_ids")


@dataclass(frozen=True)
class AcousticEvidenceBundle:
    conversation_id: str
    recording_id: str
    document_id: str
    speaker_refs: tuple[str, ...]
    source_media_sha256: str
    transcript_sha256: str
    execution_sha256: str
    identity_state_sha256: str
    model_versions: tuple[tuple[str, str], ...]
    created_at: str
    evidence: tuple[AcousticSpeakerEvidence, ...]
    lineage: tuple[EvidenceLineage, ...]
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in ("conversation_id", "recording_id", "document_id"):
            _opaque(getattr(self, field_name), field_name)
        speaker_refs = _unique(self.speaker_refs, "speaker_refs")
        if any(not SPEAKER_REF_RE.fullmatch(value) for value in speaker_refs):
            _fail("invalid_speaker_ref", "Bundle speaker_refs are invalid.")
        for field_name in (
            "source_media_sha256",
            "transcript_sha256",
            "execution_sha256",
            "identity_state_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        _time(self.created_at, "created_at")
        if not self.model_versions or any(not key or not value for key, value in self.model_versions):
            _fail("missing_model_version", "Acoustic model versions must be exact.")
        if tuple(item.speaker_ref for item in self.evidence) != speaker_refs:
            _fail("speaker_set_mismatch", "Acoustic rows must exactly match speaker_refs in order.")
        validate_lineage(self.lineage)
        validate_negative_actions(self.negative_actions)

    @property
    def bundle_id(self) -> str:
        return "bundle-" + canonical_artifact_hash(asdict(self))[:32]


@dataclass(frozen=True)
class CanonicalCandidate:
    person_id: str
    source_record_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    score: float
    accepted_relationship_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _opaque(self.person_id, "person_id")
        _unique(self.source_record_ids, "source_record_ids")
        _unique(self.evidence_ids, "candidate evidence_ids")
        if self.accepted_relationship_evidence_ids:
            _unique(self.accepted_relationship_evidence_ids, "relationship evidence_ids")
        if not 0.0 <= float(self.score) <= 1.0:
            _fail("invalid_score", "Candidate score must be between zero and one.")


@dataclass(frozen=True)
class CanonicalCandidateSnapshot:
    conversation_id: str
    document_id: str
    as_of: str
    schema_version: str
    projection_watermark: str
    candidates: tuple[CanonicalCandidate, ...]
    lineage: tuple[EvidenceLineage, ...]
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        _opaque(self.conversation_id, "conversation_id")
        _opaque(self.document_id, "document_id")
        _time(self.as_of, "as_of")
        _opaque(self.schema_version, "schema_version")
        _sha256(self.projection_watermark, "projection_watermark")
        people = tuple(item.person_id for item in self.candidates)
        if len(people) != len(set(people)):
            _fail("duplicate_person_fork", "A candidate snapshot cannot fork one person_id.")
        validate_lineage(self.lineage)
        validate_negative_actions(self.negative_actions)

    @property
    def snapshot_id(self) -> str:
        return "snapshot-" + canonical_artifact_hash(asdict(self))[:32]


@dataclass(frozen=True)
class ContextEvidenceBundle:
    conversation_id: str
    recording_id: str
    document_id: str
    speaker_refs: tuple[str, ...]
    transcript_sha256: str
    scopes: tuple[EvidenceScope, ...]
    retrieval_version: str
    ranking_version: str
    policy_version: str
    included_evidence_ids: tuple[str, ...]
    excluded_evidence: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    source_failures: tuple[tuple[str, str, bool], ...]
    lineage: tuple[EvidenceLineage, ...]
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in ("conversation_id", "recording_id", "document_id"):
            _opaque(getattr(self, field_name), field_name)
        refs = _unique(self.speaker_refs, "speaker_refs")
        if any(not SPEAKER_REF_RE.fullmatch(value) for value in refs):
            _fail("invalid_speaker_ref", "Context bundle speaker_refs are invalid.")
        _sha256(self.transcript_sha256, "transcript_sha256")
        if not self.scopes:
            _fail("missing_scope", "Context evidence requires at least one explicit scope.")
        for field_name in ("retrieval_version", "ranking_version", "policy_version"):
            _opaque(getattr(self, field_name), field_name)
        _unique(self.included_evidence_ids, "included_evidence_ids") if self.included_evidence_ids else ()
        if any(not evidence_id or not reason for evidence_id, reason in self.excluded_evidence):
            _fail("missing_exclusion_reason", "Excluded evidence requires an ID and reason.")
        if any(not source_id or not reason or not isinstance(required, bool) for source_id, reason, required in self.source_failures):
            _fail("invalid_source_failure", "Source failures require source, reason, and required flag.")
        validate_lineage(self.lineage)
        validate_negative_actions(self.negative_actions)

    @property
    def bundle_id(self) -> str:
        return "bundle-" + canonical_artifact_hash(asdict(self))[:32]


@dataclass(frozen=True)
class IdentityEvidenceFactor:
    factor_type: str
    score: float
    evidence_ids: tuple[str, ...]
    independence_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.factor_type not in {"acoustic", "context", "relationship", "contradiction"}:
            _fail("invalid_factor_type", "Identity evaluation factor type is invalid.")
        if not -1.0 <= float(self.score) <= 1.0:
            _fail("invalid_factor_score", "Identity factor score must be between -1 and 1.")
        _unique(self.evidence_ids, "factor evidence_ids")
        _unique(self.independence_groups, "factor independence_groups")


@dataclass(frozen=True)
class IdentityCaseEvaluation:
    evaluation_id: str
    conversation_id: str
    recording_id: str
    document_id: str
    speaker_ref: str
    condition: str
    acoustic_bundle_id: str | None
    context_bundle_id: str | None
    candidate_snapshot_id: str
    candidate_person_ids: tuple[str, ...]
    factors: tuple[IdentityEvidenceFactor, ...]
    outcome: str
    proposed_person_id: str | None
    alternative_person_ids: tuple[str, ...]
    contradiction_evidence_ids: tuple[str, ...]
    base_confidence: float
    capped_confidence: float
    confidence_cap_reasons: tuple[str, ...]
    abstention_reason: str | None
    source_failures: tuple[tuple[str, str, bool], ...]
    policy_version: str
    evaluated_at: str
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in (
            "evaluation_id",
            "conversation_id",
            "recording_id",
            "document_id",
            "candidate_snapshot_id",
            "policy_version",
        ):
            _opaque(getattr(self, field_name), field_name)
        if not SPEAKER_REF_RE.fullmatch(self.speaker_ref):
            _fail("invalid_speaker_ref", "Evaluation speaker_ref is invalid.")
        if self.condition not in {"context_only", "acoustic_only", "combined"}:
            _fail("invalid_condition", "Evaluation condition is invalid.")
        if self.condition == "context_only":
            if self.acoustic_bundle_id is not None or self.context_bundle_id is None:
                _fail("condition_binding_mismatch", "Context-only evaluation has wrong pillar bindings.")
        elif self.condition == "acoustic_only":
            if self.context_bundle_id is not None or self.acoustic_bundle_id is None:
                _fail("condition_binding_mismatch", "Acoustic-only evaluation has wrong pillar bindings.")
        elif self.acoustic_bundle_id is None or self.context_bundle_id is None:
            _fail("condition_binding_mismatch", "Combined evaluation requires both pillar bindings.")
        for value in (self.acoustic_bundle_id, self.context_bundle_id):
            if value is not None:
                _opaque(value, "bundle_id")
        candidates = _unique(self.candidate_person_ids, "candidate_person_ids")
        alternatives = _unique(self.alternative_person_ids, "alternative_person_ids") if self.alternative_person_ids else ()
        if not set(alternatives).issubset(candidates):
            _fail("unknown_alternative", "Evaluation alternatives must be frozen candidates.")
        if set(alternatives) & ({self.proposed_person_id} if self.proposed_person_id else set()):
            _fail("duplicate_alternative", "Proposed person cannot also be an alternative.")
        if self.outcome not in {"proposed", "abstained"}:
            _fail("invalid_evaluation_outcome", "Evaluation must propose or abstain.")
        if self.outcome == "proposed":
            if self.proposed_person_id not in candidates or self.abstention_reason is not None:
                _fail("unsupported_proposal", "Proposal must select one frozen candidate without abstention.")
        elif self.proposed_person_id is not None or not self.abstention_reason:
            _fail("invalid_abstention", "Abstention requires a reason and no person.")
        expected_confidence, expected_reasons = confidence_cap(
            self.base_confidence, self.confidence_cap_reasons
        )
        if abs(float(self.capped_confidence) - expected_confidence) > 1e-12:
            _fail("confidence_cap_mismatch", "Evaluation confidence does not match reason-coded caps.")
        if tuple(self.confidence_cap_reasons) != expected_reasons:
            _fail("confidence_reason_order", "Confidence-cap reasons must be sorted and unique.")
        factor_evidence = {
            evidence_id for factor in self.factors for evidence_id in factor.evidence_ids
        }
        if not set(self.contradiction_evidence_ids).issubset(factor_evidence):
            _fail("unbound_contradiction", "Contradictions must bind an evaluation factor.")
        if any(not source_id or not reason or not isinstance(required, bool) for source_id, reason, required in self.source_failures):
            _fail("invalid_source_failure", "Evaluation failures require source, reason, and required flag.")
        if any(required for _, _, required in self.source_failures) and self.outcome != "abstained":
            _fail("required_failure_proposed", "A required source failure forces abstention.")
        _time(self.evaluated_at, "evaluated_at")
        validate_negative_actions(self.negative_actions)

    @property
    def content_sha256(self) -> str:
        return canonical_artifact_hash(asdict(self))


@dataclass(frozen=True)
class ShadowIdentityDecision:
    decision_id: str
    evaluation_id: str
    speaker_ref: str
    outcome: str
    selected_person_id: str | None
    reviewer: str
    decided_at: str
    evaluation_sha256: str
    reason_code: str
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field_name in ("decision_id", "evaluation_id", "reviewer", "reason_code"):
            _opaque(getattr(self, field_name), field_name)
        if not SPEAKER_REF_RE.fullmatch(self.speaker_ref):
            _fail("invalid_speaker_ref", "Decision speaker_ref is invalid.")
        if self.outcome not in {"accepted", "rejected", "unresolved"}:
            _fail("invalid_decision_outcome", "Shadow decision outcome is invalid.")
        if self.outcome == "accepted":
            _opaque(str(self.selected_person_id or ""), "selected_person_id")
        elif self.selected_person_id is not None:
            _fail("decision_identity_leak", "Rejected or unresolved decisions cannot select a person.")
        _time(self.decided_at, "decided_at")
        _sha256(self.evaluation_sha256, "evaluation_sha256")
        validate_negative_actions(self.negative_actions)

    @property
    def content_sha256(self) -> str:
        return canonical_artifact_hash(asdict(self))


def validate_lineage(lineage: Sequence[EvidenceLineage]) -> None:
    evidence_ids = tuple(item.evidence_id for item in lineage)
    if len(evidence_ids) != len(set(evidence_ids)):
        _fail("duplicate_evidence", "Evidence IDs must be unique.")
    available = set(evidence_ids)
    for item in lineage:
        if item.evidence_id in item.derived_from_evidence_ids:
            _fail("circular_evidence", "Evidence cannot derive from itself.")
        if not set(item.derived_from_evidence_ids).issubset(available):
            _fail("unknown_lineage_parent", "Derived evidence must cite bundle-local parents.")
        if item.proposed_by_current_evaluation:
            _fail("circular_current_run_support", "Current-run proposals cannot support themselves.")


def validate_bundle_bindings(
    acoustic: AcousticEvidenceBundle,
    context: ContextEvidenceBundle,
    candidates: CanonicalCandidateSnapshot,
) -> None:
    if (
        acoustic.conversation_id != context.conversation_id
        or acoustic.conversation_id != candidates.conversation_id
        or acoustic.recording_id != context.recording_id
        or acoustic.document_id != context.document_id
        or acoustic.document_id != candidates.document_id
        or acoustic.speaker_refs != context.speaker_refs
        or acoustic.transcript_sha256 != context.transcript_sha256
    ):
        _fail("binding_mismatch", "Identity-case bundles do not bind the same frozen case.")
    evidence = tuple(acoustic.lineage) + tuple(context.lineage) + tuple(candidates.lineage)
    validate_lineage(evidence)


def confidence_cap(base_confidence: float, reasons: Iterable[str]) -> tuple[float, tuple[str, ...]]:
    if not 0.0 <= float(base_confidence) <= 1.0:
        _fail("invalid_confidence", "Confidence must be between zero and one.")
    selected = tuple(sorted(set(reasons)))
    unknown = set(selected) - set(CONFIDENCE_CAPS)
    if unknown:
        _fail("unknown_confidence_cap", f"Unknown confidence-cap reasons: {sorted(unknown)}")
    cap = min((CONFIDENCE_CAPS[reason] for reason in selected), default=1.0)
    return min(float(base_confidence), cap), selected


@dataclass(frozen=True)
class TransitionReceipt:
    evaluation_id: str
    actor: str
    transitioned_at: str
    prior_state: str
    next_state: str
    input_hashes: tuple[str, ...]
    policy_version: str
    reason_code: str
    negative_actions: Mapping[str, bool]

    def __post_init__(self) -> None:
        _opaque(self.evaluation_id, "evaluation_id")
        _opaque(self.actor, "actor")
        _time(self.transitioned_at, "transitioned_at")
        if self.next_state not in STATE_TRANSITIONS.get(self.prior_state, set()):
            _fail("invalid_state_transition", f"Cannot transition {self.prior_state} -> {self.next_state}.")
        if not self.input_hashes:
            _fail("missing_input_hash", "Every state transition requires input hashes.")
        for value in self.input_hashes:
            _sha256(value, "input_hash")
        _opaque(self.policy_version, "policy_version")
        _opaque(self.reason_code, "reason_code")
        validate_negative_actions(self.negative_actions)

    @property
    def receipt_id(self) -> str:
        return "transition-" + canonical_artifact_hash(asdict(self))[:32]


def _canonical_membership_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _readonly_connection(database_path: Path) -> sqlite3.Connection:
    resolved = database_path.expanduser().resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _activation_rows(connection: sqlite3.Connection, document_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not document_ids:
        _fail("empty_cohort", "Activation requires an exact cohort.")
    placeholders = ",".join("?" for _ in document_ids)
    rows = connection.execute(
        f"""
        SELECT d.id AS document_id,
               d.artifact_sha256,
               d.generated_at,
               COUNT(DISTINCT json_extract(j.value, '$.speaker')) AS speaker_count,
               b.sha256 AS source_media_sha256,
               b.bytes AS source_media_bytes
        FROM documents d
        JOIN document_blobs db ON db.document_id = d.id
        JOIN blobs b ON b.id = db.blob_id
        JOIN json_each(d.json_payload, '$.utterances') j
        WHERE d.id IN ({placeholders})
          AND d.kind = 'transcript'
          AND b.role = 'source_recording'
        GROUP BY d.id, d.artifact_sha256, d.generated_at, b.sha256, b.bytes
        ORDER BY d.generated_at ASC
        """,
        tuple(document_ids),
    ).fetchall()
    result = [dict(row) for row in rows]
    if len(result) != len(document_ids):
        _fail("cohort_binding_missing", "Every cohort document must bind one stored source recording.")
    if {row["document_id"] for row in result} != set(document_ids):
        _fail("cohort_binding_mismatch", "Cohort document membership drifted.")
    for row in result:
        _opaque(str(row["document_id"]), "document_id")
        _sha256(str(row["artifact_sha256"]), "artifact_sha256")
        _sha256(str(row["source_media_sha256"]), "source_media_sha256")
        _time(str(row["generated_at"]), "generated_at")
        if int(row["speaker_count"]) <= 0 or int(row["source_media_bytes"]) <= 0:
            _fail("ineligible_cohort_member", "Cohort sources require speakers and non-empty media.")
    return result


def _knowledge_status(connection: sqlite3.Connection) -> dict[str, Any]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_store_state'"
    ).fetchone()
    if exists is None:
        return {"schema_version": 0, "authority_mode": "sidecar", "dirty": False}
    row = connection.execute(
        "SELECT schema_version, authority_mode, dirty FROM knowledge_store_state WHERE singleton=1"
    ).fetchone()
    if row is None:
        return {"schema_version": 0, "authority_mode": "sidecar", "dirty": False}
    return {
        "schema_version": int(row["schema_version"]),
        "authority_mode": str(row["authority_mode"]),
        "dirty": bool(row["dirty"]),
    }


def freeze_activation(
    *,
    store_root: Path,
    prior_plan0057_manifest: Path,
    runtime_root: Path,
    cohort_document_ids: Sequence[str],
    expected_membership_sha256: str,
    repository_head: str,
    branch: str,
    activated_at: str,
    service_active_state: str,
    service_sub_state: str,
    service_restarts: int,
) -> dict[str, Any]:
    _sha256(expected_membership_sha256, "expected_membership_sha256")
    _git_oid(repository_head)
    _opaque(branch, "branch")
    _time(activated_at, "activated_at")
    database_path = store_root.expanduser().resolve() / "transcripts.sqlite3"
    database_stat = database_path.stat()
    if stat.S_IMODE(database_stat.st_mode) != 0o600:
        _fail("unsafe_live_database_mode", "Live transcript database must be mode 0600.")
    with _readonly_connection(database_path) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            _fail("live_database_integrity", "Live transcript database failed quick_check.")
        rows = _activation_rows(connection, cohort_document_ids)
        membership_sha256 = _canonical_membership_hash(rows)
        if membership_sha256 != expected_membership_sha256:
            _fail("cohort_membership_drift", "Frozen cohort membership hash changed.")
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("documents", "contacts", "speaker_assignments")
        }
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        knowledge_status = _knowledge_status(connection)

    prior_root = prior_plan0057_manifest.expanduser().absolute().parent.parent
    require_private_file(prior_plan0057_manifest, prior_root)
    prior = read_private_object(prior_plan0057_manifest)
    prior_cohort = ((prior.get("preview") or {}).get("private_evidence") or {}).get("cohort") or []
    prior_hashes = {
        str(item.get("source_media_sha256") or "")
        for item in prior_cohort
        if isinstance(item, Mapping)
    }
    current_hashes = {str(item["source_media_sha256"]) for item in rows}
    overlap = sorted(current_hashes & prior_hashes)
    if overlap:
        _fail("prior_source_overlap", "Plan 0059 cohort overlaps Plan 0057 source media.")

    manifest = {
        "schema_version": ACTIVATION_MANIFEST_VERSION,
        "status": "activated_pre_implementation",
        "activated_at": activated_at,
        "repository": {
            "head": repository_head,
            "branch": branch,
            "owned_worktree_count": 1,
            "upstream_behind": 0,
            "upstream_ahead_before_activation": 1,
        },
        "runtime": {
            "service": {
                "active_state": service_active_state,
                "sub_state": service_sub_state,
                "restarts": int(service_restarts),
            },
            "live_database": {
                "path": str(database_path),
                "bytes": database_stat.st_size,
                "mode": stat.S_IMODE(database_stat.st_mode),
                "quick_check": quick_check,
                "page_count": page_count,
                "page_size": page_size,
                "freelist_count": freelist_count,
                "counts": counts,
                "knowledge_status": knowledge_status,
            },
        },
        "cohort": {
            "selection": "chronological_all_post_plan0057_ingested_transcripts_at_activation",
            "members": rows,
            "membership_sha256": membership_sha256,
            "recording_count": len(rows),
            "speaker_ref_count": sum(int(row["speaker_count"]) for row in rows),
            "prior_plan0057_source_overlap_count": 0,
            "contains_human_gold": False,
        },
        "permissions": {
            "acoustic": {
                "existing_local_models_only": True,
                "existing_profiles_only": True,
                "allowlisted_subject_count": 2,
                "network_required": False,
                "enrollment_allowed": False,
                "profile_or_reference_mutation_allowed": False,
            },
            "context": {
                "source_types": ["gws", "odollo", "local"],
                "read_only": True,
                "requires_explicit_account_tenant_capability_as_of_scope": True,
                "max_provider_calls": 4,
                "max_records": 20,
                "max_characters": 12000,
                "max_per_source": 5,
                "max_relationship_hops": 1,
                "provider_write_allowed": False,
            },
        },
        "human_gates": {
            "gold_available_before_all_conditions_frozen": False,
            "decision_required_per_speaker_ref": True,
            "live_apply_allowed": False,
        },
        "finding_ledger": [],
        "review_discovery_passes_used": 0,
        "delegation": {
            "status": "not_spawned",
            "reason": "runtime_policy_disables_proactive_subagents_and_a0_p0_are_tightly_coupled",
        },
        "negative_actions": negative_action_vector(),
    }
    validate_negative_actions(manifest["negative_actions"])
    content_sha256 = canonical_artifact_hash(manifest)
    selected_root = runtime_root.expanduser().absolute()
    run_root = selected_root / f"activation-{content_sha256[:24]}"
    ensure_private_tree(selected_root, run_root)
    manifest_path = run_root / "private-manifest.json"
    stored = write_immutable_private_json(manifest_path, manifest)
    if stored != manifest:
        _fail("activation_replay_drift", "Persisted activation manifest does not match.")
    receipt = {
        "schema_version": ACTIVATION_RECEIPT_VERSION,
        "status": "activated_pre_implementation",
        "content_sha256": content_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "membership_sha256": membership_sha256,
        "recording_count": len(rows),
        "speaker_ref_count": sum(int(row["speaker_count"]) for row in rows),
        "prior_source_overlap_count": 0,
        "negative_actions_preserved": True,
    }
    receipt_path = run_root / "receipt.json"
    replay = receipt_path.exists()
    persisted_receipt = write_immutable_private_json(receipt_path, receipt)
    return {
        **persisted_receipt,
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "idempotent_replay": replay,
    }


def replay_activation(content_sha256: str, *, runtime_root: Path) -> dict[str, Any]:
    _sha256(content_sha256, "content_sha256")
    root = runtime_root.expanduser().absolute()
    run_root = root / f"activation-{content_sha256[:24]}"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    require_private_file(manifest_path, root)
    require_private_file(receipt_path, root)
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    if (
        canonical_artifact_hash(manifest) != content_sha256
        or receipt.get("content_sha256") != content_sha256
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("membership_sha256") != manifest.get("cohort", {}).get("membership_sha256")
        or manifest.get("negative_actions") != negative_action_vector()
    ):
        _fail("activation_replay_invalid", "Activation receipt binding is invalid.")
    return {**receipt, "idempotent_replay": True}


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists():
        _fail("shadow_artifact_conflict", f"Shadow database already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination_path.parent.chmod(0o700)
    try:
        with _readonly_connection(source_path) as source:
            with sqlite3.connect(destination_path) as destination:
                source.backup(destination)
                result = destination.execute("PRAGMA integrity_check").fetchone()
                if not result or result[0] != "ok":
                    _fail("shadow_database_integrity", "SQLite backup failed integrity_check.")
        destination_path.chmod(0o600)
    except Exception:
        if destination_path.exists():
            destination_path.unlink()
        raise


def _table_counts(database_path: Path) -> dict[str, int]:
    with _readonly_connection(database_path) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        return {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in tables
        }


def _quick_check(database_path: Path) -> str:
    with _readonly_connection(database_path) as connection:
        return str(connection.execute("PRAGMA quick_check").fetchone()[0])


def _document_artifact(database_path: Path, document_id: str) -> tuple[Path, str]:
    with _readonly_connection(database_path) as connection:
        row = connection.execute(
            "SELECT source_path, stored_path, artifact_sha256 FROM documents WHERE id=? AND kind='transcript'",
            (document_id,),
        ).fetchone()
    if row is None:
        _fail("shadow_document_missing", "Frozen cohort document is absent from shadow copy.")
    expected = _sha256(str(row["artifact_sha256"]), "artifact_sha256")
    for candidate in (str(row["source_path"] or ""), str(row["stored_path"] or "")):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and path.name.endswith(".transcript.json") and sha256_file(path) == expected:
            return path.resolve(), expected
    _fail("shadow_transcript_unavailable", "Frozen transcript artifact is unavailable or hash-mismatched.")
    raise AssertionError("unreachable")


def _compatibility_reconciliation_preview(database_path: Path) -> dict[str, Any]:
    with _readonly_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT id, label, email, external_ref FROM contacts ORDER BY id"
        ).fetchall()
    candidates: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        contact_id = str(row["id"])
        label = str(row["label"] or "").strip()
        email = str(row["email"] or "").strip().casefold()
        external_ref = str(row["external_ref"] or "").strip()
        normalized_label = " ".join(label.casefold().split())
        group_key = (email, normalized_label)
        groups.setdefault(group_key, []).append(contact_id)
        candidates.append(
            {
                "contact_id": contact_id,
                "label": label,
                "email": email,
                "external_ref": external_ref,
                "candidate_person_id": "person-preview-" + canonical_artifact_hash(
                    {"contact_id": contact_id}
                )[:24],
                "status": "separate_review_only",
            }
        )
    merge_groups = [
        {
            "contact_ids": sorted(contact_ids),
            "action": "review_required",
            "automatic_merge": False,
        }
        for contact_ids in groups.values()
        if len(contact_ids) > 1
    ]
    return {
        "compatibility_contact_count": len(candidates),
        "candidate_count": len(candidates),
        "merge_group_count": len(merge_groups),
        "candidates": candidates,
        "merge_groups": merge_groups,
        "ambiguous_people_remain_separate": True,
        "merge_split_redirect_review_only": True,
    }


def _p1_paths(runtime_root: Path, activation_content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p1-shadow-{activation_content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "source": run / "source-snapshot" / "transcripts.sqlite3",
        "active_root": run / "active-shadow",
        "active": run / "active-shadow" / "transcripts.sqlite3",
        "active_backup": run / "round-trip" / "active-backup.sqlite3",
        "restored_root": run / "round-trip" / "restored-shadow",
        "restored": run / "round-trip" / "restored-shadow" / "transcripts.sqlite3",
        "rollback_root": run / "rollback-rehearsal",
        "rollback": run / "rollback-rehearsal" / "transcripts.sqlite3",
        "exports": run / "exports",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def rehearse_shadow_store(
    *,
    store_root: Path,
    runtime_root: Path,
    activation_content_sha256: str,
) -> dict[str, Any]:
    activation = replay_activation(activation_content_sha256, runtime_root=runtime_root)
    paths = _p1_paths(runtime_root, activation_content_sha256)
    if paths["receipt"].exists():
        return replay_shadow_store(
            activation_content_sha256=activation_content_sha256,
            runtime_root=runtime_root,
        )
    if paths["run"].exists():
        _fail("incomplete_p1_run", "P1 run directory exists without a terminal receipt.")
    ensure_private_tree(paths["root"], paths["run"])
    activation_manifest_path = (
        paths["root"]
        / f"activation-{activation_content_sha256[:24]}"
        / "private-manifest.json"
    )
    require_private_file(activation_manifest_path, paths["root"])
    activation_manifest = read_private_object(activation_manifest_path)
    cohort = ((activation_manifest.get("cohort") or {}).get("members") or [])
    cohort_ids = [str(item.get("document_id") or "") for item in cohort]
    live_database = store_root.expanduser().resolve() / "transcripts.sqlite3"
    _sqlite_backup(live_database, paths["source"])
    source_snapshot_sha256 = sha256_file(paths["source"])
    source_counts = _table_counts(paths["source"])
    expected_counts = (
        ((activation_manifest.get("runtime") or {}).get("live_database") or {}).get("counts")
        or {}
    )
    for table_name in ("contacts", "speaker_assignments"):
        if source_counts.get(table_name) != int(expected_counts.get(table_name, -1)):
            _fail("live_identity_state_drift", f"Live {table_name} changed after activation.")
    with _readonly_connection(paths["source"]) as source_connection:
        fresh_rows = _activation_rows(source_connection, cohort_ids)
    membership_sha256 = _canonical_membership_hash(fresh_rows)
    if membership_sha256 != activation.get("membership_sha256"):
        _fail("cohort_membership_drift", "Cohort bindings changed before P1.")

    _sqlite_backup(paths["source"], paths["active"])
    active_store = ConversationKnowledgeStore(paths["active_root"])
    migration = active_store.migrate(target_version=LATEST_SCHEMA_VERSION, backup=True)
    if active_store.schema_status().schema_version != LATEST_SCHEMA_VERSION:
        _fail("shadow_migration_incomplete", "Active shadow did not reach current schema.")
    projector = ConversationKnowledgeProjector(paths["active_root"])
    projection_receipts: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []
    for document_id in cohort_ids:
        transcript_path, transcript_sha256 = _document_artifact(paths["active"], document_id)
        plan = projector.preview(transcript_path, document_id=document_id)
        first = projector.apply(plan, approval_token=APPLY_APPROVAL_TOKEN, migrate_backup=False)
        replayed = projector.apply(plan, approval_token=APPLY_APPROVAL_TOKEN, migrate_backup=False)
        if not first.reconciled or not replayed.reconciled or replayed.status != "unchanged":
            _fail("projection_replay_failed", "Shadow projection did not reconcile idempotently.")
        export_path = paths["exports"] / f"{plan.processing_history.conversation_id}.processing.json"
        exported = projector.export_sidecar(plan.processing_history.conversation_id, export_path)
        projection_receipts.append(asdict(first))
        replay_receipts.append(asdict(replayed))
        exports.append(
            {
                "conversation_id": plan.processing_history.conversation_id,
                "document_id": document_id,
                "transcript_sha256": transcript_sha256,
                "export_path": str(export_path),
                "export_sha256": sha256_file(export_path),
                "evaluation_count": len(exported.get("evaluations") or []),
                "decision_count": len(exported.get("review_decisions") or []),
            }
        )
    reconciliation = _compatibility_reconciliation_preview(paths["active"])
    active_counts = _table_counts(paths["active"])
    active_counts_sha256 = canonical_artifact_hash(active_counts)
    if _quick_check(paths["active"]) != "ok":
        _fail("active_shadow_integrity", "Active shadow failed quick_check.")

    _sqlite_backup(paths["active"], paths["active_backup"])
    _sqlite_backup(paths["active_backup"], paths["restored"])
    restored_counts = _table_counts(paths["restored"])
    restored_store = ConversationKnowledgeStore(paths["restored_root"])
    if (
        _quick_check(paths["restored"]) != "ok"
        or restored_counts != active_counts
        or restored_store.schema_status() != active_store.schema_status()
    ):
        _fail("round_trip_restore_failed", "Restored shadow does not reconcile with active shadow.")

    _sqlite_backup(paths["active_backup"], paths["rollback"])
    rollback_store = ConversationKnowledgeStore(paths["rollback_root"])
    rollback_receipt = rollback_store.rollback(target_version=0, backup=True)
    rollback_counts = _table_counts(paths["rollback"])
    if (
        rollback_store.schema_status().schema_version != 0
        or _quick_check(paths["rollback"]) != "ok"
        or any(
            rollback_counts.get(table_name) != source_counts.get(table_name)
            for table_name in ("documents", "contacts", "speaker_assignments")
        )
    ):
        _fail("rollback_rehearsal_failed", "Rollback did not preserve legacy state.")

    with _readonly_connection(live_database) as live_after:
        live_contacts = int(live_after.execute("SELECT COUNT(*) FROM contacts").fetchone()[0])
        live_assignments = int(
            live_after.execute("SELECT COUNT(*) FROM speaker_assignments").fetchone()[0]
        )
        live_knowledge_status = _knowledge_status(live_after)
    if (
        live_contacts != int(expected_counts.get("contacts", -1))
        or live_assignments != int(expected_counts.get("speaker_assignments", -1))
        or live_knowledge_status
        != ((activation_manifest.get("runtime") or {}).get("live_database") or {}).get(
            "knowledge_status"
        )
    ):
        _fail("live_state_mutation", "Live identity or knowledge authority changed during P1.")

    manifest = {
        "schema_version": P1_MANIFEST_VERSION,
        "status": "private_shadow_rehearsal_complete",
        "activation_content_sha256": activation_content_sha256,
        "source_snapshot": {
            "path": str(paths["source"]),
            "sha256": source_snapshot_sha256,
            "quick_check": _quick_check(paths["source"]),
            "table_counts": source_counts,
        },
        "migration": asdict(migration),
        "active_shadow": {
            "path": str(paths["active"]),
            "schema_status": asdict(active_store.schema_status()),
            "quick_check": "ok",
            "table_counts": active_counts,
            "table_counts_sha256": active_counts_sha256,
        },
        "projection_receipts": projection_receipts,
        "projection_replays": replay_receipts,
        "exports": exports,
        "reconciliation_preview": reconciliation,
        "round_trip_restore": {
            "backup_path": str(paths["active_backup"]),
            "backup_sha256": sha256_file(paths["active_backup"]),
            "restored_path": str(paths["restored"]),
            "restored_table_counts_sha256": canonical_artifact_hash(restored_counts),
            "reconciled": restored_counts == active_counts,
        },
        "rollback": {
            "receipt": asdict(rollback_receipt),
            "schema_status": asdict(rollback_store.schema_status()),
            "quick_check": "ok",
            "legacy_counts_preserved": True,
        },
        "live_after": {
            "contacts": live_contacts,
            "speaker_assignments": live_assignments,
            "knowledge_status": live_knowledge_status,
        },
        "negative_actions": negative_action_vector(),
    }
    validate_negative_actions(manifest["negative_actions"])
    manifest_content_sha256 = canonical_artifact_hash(manifest)
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P1_RECEIPT_VERSION,
        "status": "private_shadow_rehearsal_complete",
        "activation_content_sha256": activation_content_sha256,
        "content_sha256": manifest_content_sha256,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "source_snapshot_sha256": source_snapshot_sha256,
        "migration_from_version": migration.from_version,
        "migration_to_version": migration.to_version,
        "projection_count": len(projection_receipts),
        "projection_replay_count": len(replay_receipts),
        "export_count": len(exports),
        "compatibility_contact_count": reconciliation["compatibility_contact_count"],
        "ambiguous_people_remain_separate": True,
        "round_trip_reconciled": True,
        "rollback_to_version": rollback_receipt.to_version,
        "legacy_counts_preserved": True,
        "live_identity_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "active_shadow_root": str(paths["active_root"]),
        "idempotent_replay": False,
    }


def replay_shadow_store(
    *, activation_content_sha256: str, runtime_root: Path
) -> dict[str, Any]:
    _sha256(activation_content_sha256, "activation_content_sha256")
    paths = _p1_paths(runtime_root, activation_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    active_counts = _table_counts(paths["active"])
    if (
        receipt.get("activation_content_sha256") != activation_content_sha256
        or receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or manifest.get("active_shadow", {}).get("table_counts") != active_counts
        or manifest.get("active_shadow", {}).get("table_counts_sha256")
        != canonical_artifact_hash(active_counts)
        or _quick_check(paths["active"]) != "ok"
        or _quick_check(paths["restored"]) != "ok"
        or _quick_check(paths["rollback"]) != "ok"
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("p1_replay_invalid", "P1 shadow-store receipt binding is invalid.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "active_shadow_root": str(paths["active_root"]),
        "idempotent_replay": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan 0059 shadow identity contracts and receipts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-activation")
    freeze.add_argument("--store-root", type=Path, required=True)
    freeze.add_argument("--prior-plan0057-manifest", type=Path, required=True)
    freeze.add_argument("--runtime-root", type=Path, required=True)
    freeze.add_argument("--cohort-document-id", action="append", required=True)
    freeze.add_argument("--expected-membership-sha256", required=True)
    freeze.add_argument("--repository-head", required=True)
    freeze.add_argument("--branch", required=True)
    freeze.add_argument("--activated-at", required=True)
    freeze.add_argument("--service-active-state", required=True)
    freeze.add_argument("--service-sub-state", required=True)
    freeze.add_argument("--service-restarts", type=int, required=True)
    replay = subparsers.add_parser("replay-activation")
    replay.add_argument("--runtime-root", type=Path, required=True)
    replay.add_argument("--content-sha256", required=True)
    p1 = subparsers.add_parser("rehearse-shadow-store")
    p1.add_argument("--store-root", type=Path, required=True)
    p1.add_argument("--runtime-root", type=Path, required=True)
    p1.add_argument("--activation-content-sha256", required=True)
    p1_replay = subparsers.add_parser("replay-shadow-store")
    p1_replay.add_argument("--runtime-root", type=Path, required=True)
    p1_replay.add_argument("--activation-content-sha256", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "freeze-activation":
            result = freeze_activation(
                store_root=args.store_root,
                prior_plan0057_manifest=args.prior_plan0057_manifest,
                runtime_root=args.runtime_root,
                cohort_document_ids=args.cohort_document_id,
                expected_membership_sha256=args.expected_membership_sha256,
                repository_head=args.repository_head,
                branch=args.branch,
                activated_at=args.activated_at,
                service_active_state=args.service_active_state,
                service_sub_state=args.service_sub_state,
                service_restarts=args.service_restarts,
            )
        elif args.command == "replay-activation":
            result = replay_activation(args.content_sha256, runtime_root=args.runtime_root)
        elif args.command == "rehearse-shadow-store":
            result = rehearse_shadow_store(
                store_root=args.store_root,
                runtime_root=args.runtime_root,
                activation_content_sha256=args.activation_content_sha256,
            )
        else:
            result = replay_shadow_store(
                activation_content_sha256=args.activation_content_sha256,
                runtime_root=args.runtime_root,
            )
    except IdentityOrchestrationError as exc:
        print(json.dumps({"status": "error", "reason_code": exc.reason_code, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
