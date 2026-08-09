"""Join validated contextual speaker inference with canonical acoustic evidence.

The module is deliberately review-only.  It consumes the existing two-phase
speaker identity contract, creates one speaker-specific candidate snapshot per
diarized label, and returns immutable context-only, acoustic-only, and combined
evaluations.  It never applies assignments or creates people or voice profiles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from speaker_identity_orchestration import (
    AcousticEvidenceBundle,
    CanonicalCandidate,
    CanonicalCandidateSnapshot,
    ContextEvidenceBundle,
    EvidenceLineage,
    EvidenceScope,
    IdentityCaseEvaluation,
    IdentityEvidenceFactor,
    confidence_cap,
    negative_action_vector,
)
from speaker_identity_preprocess import validate_and_score_identity_evaluation


SCHEMA_VERSION = "transcribe-audio.contextual-speaker-identity-join.v1"
POLICY_VERSION = "contextual-canonical-acoustic-review-v1"


class ContextualIdentityJoinError(ValueError):
    """Raised when prepared contextual evidence cannot form a safe join."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(reason_code: str, message: str) -> None:
    raise ContextualIdentityJoinError(reason_code, message)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{uuid5(NAMESPACE_URL, value)}"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    return tuple(
        value
        for raw in values
        if (value := str(raw or "").strip())
        and not (value in seen or seen.add(value))
    )


@dataclass(frozen=True)
class SuggestedPerson:
    name: str
    email: str
    organization: str


@dataclass(frozen=True)
class SpeakerReviewOutcome:
    speaker_ref: str
    source_speaker_label: str
    context_status: str
    context_person_id: str | None
    acoustic_person_id: str | None
    suggestions: tuple[SuggestedPerson, ...]
    review_flags: tuple[str, ...]
    context_evidence_ids: tuple[str, ...]
    reason_code: str


@dataclass(frozen=True)
class ContextualIdentityJoinResult:
    schema_version: str
    policy_version: str
    document_id: str
    context_bundle: ContextEvidenceBundle
    candidate_snapshots: tuple[CanonicalCandidateSnapshot, ...]
    evaluations: tuple[IdentityCaseEvaluation, ...]
    review_outcomes: tuple[SpeakerReviewOutcome, ...]
    validated_evaluation_id: str
    negative_actions: Mapping[str, bool]

    @property
    def content_sha256(self) -> str:
        return _canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_sha256"] = self.content_sha256
        return payload


def _prepared_scopes(packet: Mapping[str, Any]) -> tuple[EvidenceScope, ...]:
    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), Mapping) else {}
    as_of = str(retrieval.get("as_of") or "").strip()
    budgets = retrieval.get("budgets") if isinstance(retrieval.get("budgets"), Mapping) else {}
    if not as_of:
        _fail("missing_retrieval_as_of", "Contextual join requires the prepared retrieval as-of time.")
    raw_contexts = tuple(
        raw
        for raw in packet.get("source_contexts") or []
        if isinstance(raw, Mapping)
    )
    authority_by_source = {
        str(raw.get("source_id") or "").strip(): raw
        for raw in raw_contexts
        if isinstance(raw.get("owner"), Mapping)
        and str(raw.get("source_id") or "").strip()
    }
    retrieval_contexts = tuple(
        raw for raw in raw_contexts if raw.get("retrieval_scope") == "explicit"
    )
    if not retrieval_contexts:
        retrieval_contexts = raw_contexts

    scopes: list[EvidenceScope] = []
    seen: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    for raw in retrieval_contexts:
        source_id = str(raw.get("source_id") or "").strip()
        authority = authority_by_source.get(source_id) or {}
        owner = (
            authority.get("owner")
            if isinstance(authority.get("owner"), Mapping)
            else {}
        )
        capabilities = _unique(
            raw.get("capabilities")
            or raw.get("evidence_capabilities")
            or authority.get("evidence_capabilities")
            or ()
        )
        key = (
            str(raw.get("source_type") or "identity_context").strip(),
            str(
                raw.get("source_profile")
                or raw.get("source_profile_id")
                or source_id
            ).strip(),
            str(raw.get("account_id") or owner.get("id") or "").strip(),
            str(
                raw.get("tenant_id")
                or authority.get("relationship_scope")
                or ""
            ).strip(),
            capabilities,
        )
        if not all((*key[:4], capabilities)):
            _fail("incomplete_retrieval_scope", "Every contextual source requires type, profile, account, tenant, and capabilities.")
        if key in seen:
            continue
        seen.add(key)
        scopes.append(
            EvidenceScope(
                source_type=key[0],
                source_profile=key[1],
                account_id=key[2],
                tenant_id=key[3],
                capabilities=capabilities,
                as_of=as_of,
                max_records=int(budgets.get("max_records") or 0),
                max_characters=int(budgets.get("max_characters") or 0),
                max_per_source=int(budgets.get("max_per_source") or 0),
                max_provider_calls=int(budgets.get("max_provider_calls") or 0),
                max_relationship_hops=int(budgets.get("max_relationship_hops") or 0),
            )
        )
    if not scopes:
        _fail("missing_retrieval_scope", "Contextual join requires at least one prepared source scope.")
    return tuple(scopes)


def _prepared_reference_values(packet: Mapping[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    conversation = packet.get("conversation") if isinstance(packet.get("conversation"), Mapping) else {}
    for value in [conversation.get("conversation_id"), *(conversation.get("recording_ids") or [])]:
        if str(value or "").strip():
            values.append((str(value), "conversation"))
    for speaker in packet.get("speakers") or []:
        if not isinstance(speaker, Mapping):
            continue
        for clue in speaker.get("utterance_clues") or []:
            if isinstance(clue, Mapping) and str(clue.get("utterance_id") or "").strip():
                values.append((str(clue["utterance_id"]), "transcript_clue"))
    calendar = packet.get("calendar_context") if isinstance(packet.get("calendar_context"), Mapping) else {}
    if str(calendar.get("event_id") or "").strip():
        values.append((str(calendar["event_id"]), "calendar_event"))
    for attendee in calendar.get("attendees") or []:
        if not isinstance(attendee, Mapping):
            continue
        for key in ("id", "email"):
            if str(attendee.get(key) or "").strip():
                values.append((str(attendee[key]), "calendar_attendee"))
    for context in packet.get("source_contexts") or []:
        if not isinstance(context, Mapping):
            continue
        for key in ("source_id", "source_profile", "source_profile_id"):
            if str(context.get(key) or "").strip():
                values.append((str(context[key]), "source_context"))
    for person in packet.get("people") or []:
        if not isinstance(person, Mapping):
            continue
        person_id = str(person.get("person_id") or "").strip()
        if person_id:
            values.append((person_id, "canonical_person"))
        for email in person.get("emails") or []:
            if str(email or "").strip():
                values.append((str(email), "canonical_person"))
        for record in person.get("source_records") or []:
            if not isinstance(record, Mapping):
                continue
            for key in ("source_id", "record_id"):
                if str(record.get(key) or "").strip():
                    values.append((str(record[key]), "canonical_person_source"))
    for source in packet.get("provenance_sources") or []:
        if isinstance(source, Mapping) and str(source.get("source_id") or "").strip():
            values.append((str(source["source_id"]), "provider_evidence"))
    for evidence in (
        (packet.get("retrieval") or {}).get("evidence")
        if isinstance(packet.get("retrieval"), Mapping)
        else []
    ) or []:
        if isinstance(evidence, Mapping) and str(evidence.get("evidence_id") or "").strip():
            values.append((str(evidence["evidence_id"]), "provider_evidence"))
    return values


def _prepared_lineage(
    packet: Mapping[str, Any], *, evaluated_at: str
) -> tuple[tuple[EvidenceLineage, ...], dict[str, str]]:
    retrieval = packet.get("retrieval") if isinstance(packet.get("retrieval"), Mapping) else {}
    conversation_at = str(retrieval.get("conversation_at") or evaluated_at)
    retrieval_by_id = {
        str(item.get("evidence_id")): item
        for item in retrieval.get("evidence") or []
        if isinstance(item, Mapping) and str(item.get("evidence_id") or "").strip()
    }
    result: list[EvidenceLineage] = []
    mapped: dict[str, str] = {}
    for original, source_type in _prepared_reference_values(packet):
        if original in mapped:
            continue
        evidence_id = _stable_id("evidence-prepared", packet.get("evaluation_id"), original)
        mapped[original] = evidence_id
        raw = retrieval_by_id.get(original) or {}
        source_event_at = str(raw.get("source_event_at") or conversation_at)
        observed_at = str(raw.get("observed_at") or evaluated_at)
        retrieved_at = str(raw.get("retrieved_at") or evaluated_at)
        independence = str(raw.get("independence_group_id") or "").strip()
        result.append(
            EvidenceLineage(
                evidence_id=evidence_id,
                source_record_id=_stable_id("source-prepared", source_type, original),
                independence_group=(
                    _stable_id("independence", independence)
                    if independence
                    else _stable_id("independence", source_type, original)
                ),
                source_type=source_type,
                source_event_at=source_event_at,
                observed_at=observed_at,
                retrieved_at=retrieved_at,
                content_sha256=str(raw.get("content_hash") or _canonical_hash({"reference": original, "source_type": source_type})),
            )
        )
    if not result:
        _fail("missing_context_lineage", "Validated contextual output has no prepared evidence lineage.")
    return tuple(result), mapped


def _assignment_original_evidence(assignment: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(str(value) for value in assignment.get("transcript_clue_ids") or [])
    values.extend(str(value) for value in assignment.get("provenance_source_ids") or [])
    for factor in assignment.get("factors") or []:
        if isinstance(factor, Mapping):
            values.extend(str(value) for value in factor.get("evidence_ids") or [])
    return _unique(values)


def _context_factor(
    *,
    score: float,
    assignments: Sequence[Mapping[str, Any]],
    mapped_references: Mapping[str, str],
    lineage: Sequence[EvidenceLineage],
) -> IdentityEvidenceFactor:
    ids = _unique(
        mapped_references[value]
        for assignment in assignments
        for value in _assignment_original_evidence(assignment)
        if value in mapped_references
    )
    if not ids:
        ids = (lineage[0].evidence_id,)
    groups = {item.evidence_id: item.independence_group for item in lineage}
    return IdentityEvidenceFactor(
        factor_type="context",
        score=score,
        evidence_ids=ids,
        independence_groups=_unique(groups[value] for value in ids),
    )


def _acoustic_factor(
    *, row: Any, lineage: Sequence[EvidenceLineage]
) -> IdentityEvidenceFactor:
    groups = {item.evidence_id: item.independence_group for item in lineage}
    evidence_ids = _unique(row.evidence_ids)
    if not evidence_ids or any(value not in groups for value in evidence_ids):
        _fail("acoustic_lineage_missing", "Acoustic evidence lacks bundle-local lineage.")
    return IdentityEvidenceFactor(
        factor_type="acoustic",
        score=float(row.score),
        evidence_ids=evidence_ids,
        independence_groups=_unique(groups[value] for value in evidence_ids),
    )


def _suggestions(assignments: Sequence[Mapping[str, Any]]) -> tuple[SuggestedPerson, ...]:
    result: list[SuggestedPerson] = []
    seen: set[tuple[str, str, str]] = set()
    for assignment in assignments:
        raw = assignment.get("suggested_person") if isinstance(assignment.get("suggested_person"), Mapping) else {}
        values = (
            str(raw.get("name") or "").strip(),
            str(raw.get("email") or "").strip().casefold(),
            str(raw.get("organization") or "").strip(),
        )
        if not any(values) or values in seen:
            continue
        seen.add(values)
        result.append(SuggestedPerson(*values))
    return tuple(result)


def _candidate_snapshot(
    *,
    packet: Mapping[str, Any],
    speaker_ref: str,
    person_scores: Mapping[str, float],
    conversation_id: str,
    document_id: str,
    as_of: str,
    projection_watermark: str,
    evaluated_at: str,
) -> CanonicalCandidateSnapshot:
    people = {
        str(person.get("person_id")): person
        for person in packet.get("people") or []
        if isinstance(person, Mapping) and str(person.get("person_id") or "").strip()
    }
    rows: list[CanonicalCandidate] = []
    lineage: list[EvidenceLineage] = []
    for person_id in sorted(person_scores):
        person = people.get(person_id)
        if person is None:
            continue
        evidence_id = _stable_id("evidence-candidate", document_id, speaker_ref, person_id)
        source_record_id = _stable_id("source-canonical-person", person_id)
        lineage.append(
            EvidenceLineage(
                evidence_id=evidence_id,
                source_record_id=source_record_id,
                independence_group=_stable_id("independence-canonical-person", person_id),
                source_type="canonical_person_candidate",
                source_event_at=evaluated_at,
                observed_at=evaluated_at,
                retrieved_at=evaluated_at,
                content_sha256=_canonical_hash(person),
            )
        )
        rows.append(
            CanonicalCandidate(
                person_id=person_id,
                source_record_ids=(source_record_id,),
                evidence_ids=(evidence_id,),
                score=float(person_scores[person_id]),
            )
        )
    return CanonicalCandidateSnapshot(
        conversation_id=conversation_id,
        document_id=document_id,
        as_of=as_of,
        schema_version="speaker-specific-canonical-candidates-v1",
        projection_watermark=projection_watermark,
        candidates=tuple(rows),
        lineage=tuple(lineage),
        negative_actions=negative_action_vector(),
    )


def _context_reason(status: str, coverage_reason: str) -> str:
    if coverage_reason:
        return coverage_reason
    return {
        "unlisted": "context_unlisted_person_requires_review",
        "unresolved": "context_identity_unresolved",
        "conflicting": "context_identity_conflicting",
        "candidate_match": "context_person_not_mapped_to_canonical_person",
    }.get(status, "context_identity_missing")


def join_contextual_identity(
    *,
    document_id: str,
    transcript_sha256: str,
    identity_packet: Mapping[str, Any],
    identity_readout: Mapping[str, Any],
    acoustic_bundle: AcousticEvidenceBundle,
    speaker_ref_bindings: Mapping[str, str],
    acoustic_subject_person_bindings: Mapping[str, str],
    evaluated_at: str,
) -> ContextualIdentityJoinResult:
    """Create review-only three-condition evaluations from the existing workflow.

    The interface validates model output itself.  Context may select only a
    prepared canonical person; acoustic evidence may select only an explicitly
    bound subject whose person is also prepared in the same identity packet.
    Duplicate or missing speaker coverage becomes a reason-coded abstention.
    """

    validated = validate_and_score_identity_evaluation(
        dict(identity_packet), dict(identity_readout)
    )
    readout = validated["readout"]
    conversation = identity_packet.get("conversation") if isinstance(identity_packet.get("conversation"), Mapping) else {}
    conversation_id = str(conversation.get("conversation_id") or "").strip()
    recording_ids = _unique(conversation.get("recording_ids") or ())
    if len(recording_ids) != 1:
        _fail("recording_binding_ambiguous", "Contextual join requires exactly one recording ID.")
    recording_id = recording_ids[0]
    source_speaker_labels = _unique(
        str(item.get("speaker_label") or "")
        for item in identity_packet.get("speakers") or []
        if isinstance(item, Mapping)
    )
    if not source_speaker_labels:
        _fail("missing_speakers", "Contextual join requires prepared speaker labels.")
    normalized_bindings = {
        str(source or "").strip(): str(target or "").strip()
        for source, target in speaker_ref_bindings.items()
    }
    if (
        set(normalized_bindings) != set(source_speaker_labels)
        or set(normalized_bindings.values()) != set(acoustic_bundle.speaker_refs)
        or len(set(normalized_bindings.values())) != len(normalized_bindings)
    ):
        _fail(
            "speaker_ref_binding_mismatch",
            "Every prepared speaker label must bind exactly one acoustic speaker reference.",
        )
    source_by_speaker_ref = {
        target: source for source, target in normalized_bindings.items()
    }
    speaker_refs = tuple(acoustic_bundle.speaker_refs)
    if (
        acoustic_bundle.conversation_id != conversation_id
        or acoustic_bundle.recording_id != recording_id
        or acoustic_bundle.document_id != document_id
        or acoustic_bundle.speaker_refs != speaker_refs
        or acoustic_bundle.transcript_sha256 != transcript_sha256
    ):
        _fail("case_binding_mismatch", "Contextual and acoustic evidence do not bind the same frozen case.")

    scopes = _prepared_scopes(identity_packet)
    context_lineage, mapped_references = _prepared_lineage(
        identity_packet, evaluated_at=evaluated_at
    )
    retrieval = identity_packet.get("retrieval") if isinstance(identity_packet.get("retrieval"), Mapping) else {}
    projection_watermark = str(retrieval.get("bundle_content_hash") or "").strip()
    if len(projection_watermark) != 64:
        projection_watermark = _canonical_hash(identity_packet)
    source_failures = tuple(
        (
            str(item.get("adapter_id") or item.get("source_profile_id") or "provider"),
            str(item.get("reason_code") or "provider_failure"),
            bool(item.get("required", False)),
        )
        for item in retrieval.get("source_failures") or []
        if isinstance(item, Mapping)
    )
    excluded = tuple(
        (
            mapped_references[str(item.get("evidence_id"))],
            str(item.get("reason_code") or "excluded"),
        )
        for item in retrieval.get("evidence") or []
        if isinstance(item, Mapping)
        and str(item.get("evidence_id") or "") in mapped_references
        and str(item.get("disposition") or "") != "included"
    )
    included = tuple(
        item.evidence_id
        for item in context_lineage
        if item.evidence_id not in {value for value, _ in excluded}
    )
    context_bundle = ContextEvidenceBundle(
        conversation_id=conversation_id,
        recording_id=recording_id,
        document_id=document_id,
        speaker_refs=speaker_refs,
        transcript_sha256=transcript_sha256,
        scopes=scopes,
        retrieval_version=str(retrieval.get("retrieval_version") or "speaker-context-retrieval-v1"),
        ranking_version=str(retrieval.get("ranking_version") or "speaker-context-ranking-v1"),
        policy_version=POLICY_VERSION,
        included_evidence_ids=included,
        excluded_evidence=excluded,
        warnings=_unique([*(retrieval.get("warnings") or []), *(readout.get("warnings") or [])]),
        source_failures=source_failures,
        lineage=context_lineage,
        negative_actions=negative_action_vector(),
    )

    assignments = [
        item
        for item in readout.get("speaker_assignments") or []
        if isinstance(item, Mapping)
    ]
    assignments_by_speaker = {
        speaker_ref: [
            assignment
            for assignment in assignments
            if source_by_speaker_ref[speaker_ref]
            in _unique(assignment.get("speaker_labels") or ())
        ]
        for speaker_ref in speaker_refs
    }
    acoustic_by_speaker = {item.speaker_ref: item for item in acoustic_bundle.evidence}
    prepared_people = {
        str(item.get("person_id"))
        for item in identity_packet.get("people") or []
        if isinstance(item, Mapping) and str(item.get("person_id") or "").strip()
    }
    any_required_failure = any(required for _, _, required in source_failures)

    snapshots: list[CanonicalCandidateSnapshot] = []
    evaluations: list[IdentityCaseEvaluation] = []
    review_outcomes: list[SpeakerReviewOutcome] = []
    as_of = str(retrieval.get("as_of") or "")

    for speaker_ref in speaker_refs:
        source_speaker_label = source_by_speaker_ref[speaker_ref]
        selected = assignments_by_speaker[speaker_ref]
        coverage_reason = (
            "context_speaker_coverage_missing"
            if not selected
            else "context_duplicate_speaker_coverage"
            if len(selected) > 1
            else ""
        )
        assignment = selected[0] if len(selected) == 1 else {}
        context_status = str(assignment.get("status") or "unresolved")
        context_confidence = float(
            ((assignment.get("confidence") or {}).get("numeric") or 0)
        ) / 100.0
        context_person_id = (
            str(assignment.get("person_id") or "").strip()
            if context_status == "candidate_match" and not coverage_reason
            else ""
        )
        if context_person_id not in prepared_people:
            context_person_id = ""

        acoustic_row = acoustic_by_speaker[speaker_ref]
        acoustic_person_id = ""
        if acoustic_row.disposition != "abstain" and acoustic_row.acoustic_subject_id:
            acoustic_person_id = str(
                acoustic_subject_person_bindings.get(acoustic_row.acoustic_subject_id) or ""
            ).strip()
            if acoustic_person_id not in prepared_people:
                acoustic_person_id = ""

        person_scores: dict[str, float] = {}
        if context_person_id:
            person_scores[context_person_id] = context_confidence
        if acoustic_person_id:
            person_scores[acoustic_person_id] = max(
                person_scores.get(acoustic_person_id, 0.0), float(acoustic_row.score)
            )
        snapshot = _candidate_snapshot(
            packet=identity_packet,
            speaker_ref=speaker_ref,
            person_scores=person_scores,
            conversation_id=conversation_id,
            document_id=document_id,
            as_of=as_of,
            projection_watermark=projection_watermark,
            evaluated_at=evaluated_at,
        )
        snapshots.append(snapshot)
        candidate_ids = tuple(item.person_id for item in snapshot.candidates)

        context_factor = _context_factor(
            score=context_confidence,
            assignments=selected,
            mapped_references=mapped_references,
            lineage=context_lineage,
        )
        acoustic_factor = _acoustic_factor(
            row=acoustic_row, lineage=acoustic_bundle.lineage
        )
        review_flags = _unique(
            value
            for item in selected
            for value in item.get("review_flags") or []
        )
        suggestions = _suggestions(selected)
        reason_code = coverage_reason or _context_reason(context_status, "")
        review_outcomes.append(
            SpeakerReviewOutcome(
                speaker_ref=speaker_ref,
                source_speaker_label=source_speaker_label,
                context_status=context_status,
                context_person_id=context_person_id or None,
                acoustic_person_id=acoustic_person_id or None,
                suggestions=suggestions,
                review_flags=review_flags,
                context_evidence_ids=context_factor.evidence_ids,
                reason_code=("context_candidate_match" if context_person_id else reason_code),
            )
        )

        context_blocked_reason = (
            coverage_reason
            or ("required_provider_failure" if any_required_failure else "")
            or ("" if context_person_id else _context_reason(context_status, ""))
        )
        acoustic_blocked_reason = (
            ""
            if acoustic_person_id
            else "acoustic_abstained"
            if acoustic_row.disposition == "abstain"
            else "acoustic_subject_not_mapped_to_prepared_person"
        )
        context_proposal = context_person_id if not context_blocked_reason else ""
        acoustic_proposal = acoustic_person_id if not acoustic_blocked_reason else ""

        for condition in ("context_only", "acoustic_only", "combined"):
            if condition == "context_only":
                factors = (context_factor,)
                proposal = context_proposal
                abstention_reason = context_blocked_reason
                base_confidence = context_confidence
                cap_reasons = [
                    "partial_provider_failure" for _ in [0] if source_failures
                ]
                alternatives: tuple[str, ...] = ()
                contradictions: tuple[str, ...] = ()
            elif condition == "acoustic_only":
                factors = (acoustic_factor,)
                proposal = acoustic_proposal
                abstention_reason = acoustic_blocked_reason
                base_confidence = float(acoustic_row.score)
                cap_reasons = []
                alternatives = ()
                contradictions = ()
            else:
                factors = (acoustic_factor, context_factor)
                base_confidence = max(context_confidence, float(acoustic_row.score))
                cap_reasons = [
                    "partial_provider_failure" for _ in [0] if source_failures
                ]
                alternatives = ()
                contradictions = ()
                if coverage_reason or any_required_failure:
                    proposal = ""
                    abstention_reason = coverage_reason or "required_provider_failure"
                elif context_proposal and acoustic_proposal and context_proposal != acoustic_proposal:
                    proposal = ""
                    abstention_reason = "pillar_identity_conflict"
                    cap_reasons.append("material_contradiction")
                    alternatives = _unique((context_proposal, acoustic_proposal))
                    contradictions = _unique(
                        (*context_factor.evidence_ids, *acoustic_factor.evidence_ids)
                    )
                elif context_proposal or acoustic_proposal:
                    proposal = context_proposal or acoustic_proposal
                    abstention_reason = ""
                else:
                    proposal = ""
                    abstention_reason = context_blocked_reason or acoustic_blocked_reason

            capped_confidence, normalized_reasons = confidence_cap(
                base_confidence, cap_reasons
            )
            evaluations.append(
                IdentityCaseEvaluation(
                    evaluation_id=_stable_id(
                        "evaluation-contextual-join",
                        identity_packet.get("evaluation_id"),
                        document_id,
                        speaker_ref,
                        condition,
                    ),
                    conversation_id=conversation_id,
                    recording_id=recording_id,
                    document_id=document_id,
                    speaker_ref=speaker_ref,
                    condition=condition,
                    acoustic_bundle_id=(
                        acoustic_bundle.bundle_id if condition != "context_only" else None
                    ),
                    context_bundle_id=(
                        context_bundle.bundle_id if condition != "acoustic_only" else None
                    ),
                    candidate_snapshot_id=snapshot.snapshot_id,
                    candidate_person_ids=candidate_ids,
                    factors=factors,
                    outcome="proposed" if proposal else "abstained",
                    proposed_person_id=proposal or None,
                    alternative_person_ids=alternatives,
                    contradiction_evidence_ids=contradictions,
                    base_confidence=base_confidence,
                    capped_confidence=capped_confidence,
                    confidence_cap_reasons=normalized_reasons,
                    abstention_reason=abstention_reason or None,
                    source_failures=(
                        source_failures if condition != "acoustic_only" else ()
                    ),
                    policy_version=POLICY_VERSION,
                    evaluated_at=evaluated_at,
                    negative_actions=negative_action_vector(),
                )
            )

    return ContextualIdentityJoinResult(
        schema_version=SCHEMA_VERSION,
        policy_version=POLICY_VERSION,
        document_id=document_id,
        context_bundle=context_bundle,
        candidate_snapshots=tuple(snapshots),
        evaluations=tuple(evaluations),
        review_outcomes=tuple(review_outcomes),
        validated_evaluation_id=str(identity_packet.get("evaluation_id") or ""),
        negative_actions=negative_action_vector(),
    )
