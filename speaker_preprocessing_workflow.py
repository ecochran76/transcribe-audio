#!/usr/bin/env python3
"""Host-owned two-phase App Intelligence preparation for speaker identity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

import app_intelligence_ledger
import intelligence_config
from conversation_identity_retrieval import PreparedIdentityEvidenceBundle
from conversation_processing import (
    append_evaluation,
    append_review_decision,
    ensure_transcript_identity,
    processing_sidecar_path,
)
from speaker_identity_preprocess import (
    build_clue_discovery_packet,
    build_clue_discovery_prompt,
    build_identity_evaluation_packet,
    build_identity_evaluation_prompt,
    validate_clue_discovery_readout,
    validate_and_score_identity_evaluation,
)

REFERENCE_REPAIR_PACKET_SCHEMA_VERSION = (
    "transcribe-audio.speaker-reference-repair-packet.v1"
)
IDENTITY_RETRIEVAL_CONTEXT_SCHEMA_VERSION = (
    "transcribe-audio.identity-retrieval-context.v1"
)
_RETRIEVAL_FAILURE_FIELDS = {
    "adapter_id",
    "source_profile_id",
    "reason_code",
    "detail",
}


def _read_transcript(path: Path) -> dict[str, Any]:
    transcript = ensure_transcript_identity(path)
    if not isinstance(transcript.get("utterances"), list):
        raise ValueError("Speaker preprocessing requires transcript utterances.")
    return transcript


def _route(value: Optional[dict[str, Any]]) -> dict[str, Any]:
    if value is not None:
        route = dict(value)
    else:
        route = intelligence_config.resolve_task_config(
            intelligence_config.TASK_SPEAKER_DISAMBIGUATION
        ).to_dict()
    if route.get("provider") != "codex-app-server":
        raise ValueError("Speaker preprocessing requires the codex-app-server provider.")
    if not str(route.get("model") or "").strip():
        route["model"] = intelligence_config.DEFAULT_CODEX_APP_MODEL
    return route


def captured_run_json(*, state_root: Path, run_id: str) -> dict[str, Any]:
    """Read one completed, host-captured model JSON result from its run ledger."""
    shown = app_intelligence_ledger.response_for_run(
        state_root=state_root,
        run_id=run_id,
        event_limit=0,
    )
    run = shown["run"]
    latest = (
        run.get("latest_model_turn_status")
        if isinstance(run.get("latest_model_turn_status"), dict)
        else {}
    )
    if not latest.get("completed"):
        raise ValueError(f"App Intelligence run has no completed captured turn: {run_id}.")
    run_path = app_intelligence_ledger.run_dir(state_root, run_id).resolve()
    artifact_path = Path(str(latest.get("artifact_path") or "")).resolve()
    try:
        artifact_path.relative_to(run_path)
    except ValueError as exc:
        raise ValueError("Captured model artifact resolves outside its App Intelligence run.") from exc
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return app_intelligence_ledger.extract_json_object(
        str(payload.get("output_text") or "")
    )


def _prepare_run(
    *,
    phase: str,
    document_id: str,
    document_title: str,
    state_root: Path,
    route: dict[str, Any],
    prompt_text: str,
    input_packet: dict[str, Any],
) -> dict[str, Any]:
    created = app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="speaker_preprocessing",
        purpose=f"Prepare reviewed {phase.replace('_', ' ')} for speaker preprocessing.",
        document_id=document_id,
        provider=route["provider"],
    )
    run_id = created["run"]["run_id"]
    prepared = app_intelligence_ledger.prepare_model_turn_packet(
        state_root=state_root,
        run_id=run_id,
        task=f"speaker_{phase}",
        route=route,
        document={"id": document_id, "title": document_title},
        prompt_text=prompt_text,
        approval_token=app_intelligence_ledger.MODEL_TURN_PREFLIGHT_TOKEN,
    )
    input_packet_path = (
        app_intelligence_ledger.run_dir(state_root, run_id)
        / "artifacts"
        / "speaker-preprocessing"
        / f"{phase}.input.json"
    )
    input_packet_path.parent.mkdir(parents=True, exist_ok=True)
    input_packet_path.write_text(
        json.dumps(input_packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "phase": phase,
        "run_id": run_id,
        "route": route,
        "prompt_packet": prepared["packet"],
        "packet_path": prepared["packet_path"],
        "prompt_path": prepared["prompt_path"],
        "input_packet_path": str(input_packet_path),
        "will_send_prompt": False,
        "future_required_approval_token_for_session_start": (
            app_intelligence_ledger.SESSION_START_APPROVAL_TOKEN
        ),
        "future_required_approval_token_for_send": (
            app_intelligence_ledger.MODEL_TURN_SEND_TOKEN
        ),
    }


def prepare_clue_discovery(
    transcript_path: Path,
    *,
    document_id: str,
    state_root: Path,
    source_contexts: Iterable[dict[str, Any]] = (),
    route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare, but never send, the first App Intelligence phase."""
    transcript = _read_transcript(transcript_path)
    packet = build_clue_discovery_packet(
        transcript=transcript,
        source_contexts=source_contexts,
    )
    result = _prepare_run(
        phase="clue_discovery",
        document_id=document_id,
        document_title=str(transcript.get("transcript_title") or transcript_path.stem),
        state_root=state_root,
        route=_route(route),
        prompt_text=build_clue_discovery_prompt(packet),
        input_packet=packet,
    )
    return {**result, "packet": packet}


def prepare_identity_evaluation(
    transcript_path: Path,
    *,
    document_id: str,
    state_root: Path,
    discovery_readout: dict[str, Any],
    person_records: Iterable[dict[str, Any]] = (),
    provenance_sources: Iterable[dict[str, Any]] = (),
    source_contexts: Iterable[dict[str, Any]] = (),
    retrieval_bundle: Optional[PreparedIdentityEvidenceBundle] = None,
    route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare, but never send, the second phase after validated discovery."""
    transcript = _read_transcript(transcript_path)
    person_records = tuple(person_records)
    provenance_sources = tuple(provenance_sources)
    source_contexts = tuple(source_contexts)
    bundle_inputs: Optional[dict[str, Any]] = None
    if retrieval_bundle is not None:
        if person_records or provenance_sources:
            raise ValueError(
                "retrieval_bundle cannot be combined with legacy person_records "
                "or provenance_sources."
            )
        bundle_inputs = _identity_inputs_from_retrieval_bundle(retrieval_bundle)
        source_contexts = (*source_contexts, *bundle_inputs["source_contexts"])
    packet = build_identity_evaluation_packet(
        transcript=transcript,
        discovery_readout=discovery_readout,
        person_records=person_records,
        provenance_sources=(
            bundle_inputs["provenance_sources"]
            if bundle_inputs is not None
            else provenance_sources
        ),
        source_contexts=source_contexts,
    )
    if bundle_inputs is not None:
        request_conversation_id = retrieval_bundle.request.conversation_id
        packet_conversation_id = str(
            (packet.get("conversation") or {}).get("conversation_id") or ""
        )
        if request_conversation_id != packet_conversation_id:
            raise ValueError(
                "retrieval_bundle conversation does not match the transcript."
            )
        packet["people"] = bundle_inputs["people"]
        packet["retrieval"] = bundle_inputs["retrieval"]
    result = _prepare_run(
        phase="identity_evaluation",
        document_id=document_id,
        document_title=str(transcript.get("transcript_title") or transcript_path.stem),
        state_root=state_root,
        route=_route(route),
        prompt_text=build_identity_evaluation_prompt(packet),
        input_packet=packet,
    )
    return {**result, "packet": packet}


def _identity_inputs_from_retrieval_bundle(
    bundle: PreparedIdentityEvidenceBundle,
) -> dict[str, Any]:
    """Adapt one persisted retrieval result into the existing evaluation contract."""
    request = bundle.request
    persisted = bundle.persisted_bundle
    item_by_evidence_id = {
        item.evidence_id: item
        for item in persisted.items
    }
    provenance_sources: list[dict[str, Any]] = []
    evidence_context: list[dict[str, Any]] = []
    for ranked in bundle.evidence:
        snapshot = ranked.snapshot
        item = item_by_evidence_id.get(snapshot.evidence_id)
        if item is None:
            raise ValueError(
                f"retrieval_bundle lacks persisted item: {snapshot.evidence_id}."
            )
        summary = {
            "evidence_id": snapshot.evidence_id,
            "source_profile_id": snapshot.source_profile_id,
            "provider_kind": snapshot.provider_kind,
            "account_id": snapshot.account_id,
            "tenant_id": snapshot.tenant_id,
            "source_type": snapshot.source_type,
            "capability": snapshot.capability,
            "disposition": item.disposition,
            "reason_code": item.reason_code,
            "rank": item.rank,
            "score": item.score,
            "direction": ranked.direction,
            "freshness_state": snapshot.freshness_state,
            "temporal_class": snapshot.temporal_class,
            "source_event_at": snapshot.source_event_at,
            "observed_at": snapshot.observed_at,
            "retrieved_at": snapshot.retrieved_at,
            "content_hash": snapshot.content_hash,
            "independence_group_id": snapshot.independence_group_id,
        }
        evidence_context.append(summary)
        if item.disposition != "included":
            continue
        provenance_sources.append(
            {
                "source_type": snapshot.source_type,
                "source_id": snapshot.evidence_id,
                "label": (
                    str(snapshot.structured_metadata.get("label") or "")
                    or f"{snapshot.provider_kind} {snapshot.capability}"
                ),
                "snippet": snapshot.snippet,
                "profile": snapshot.source_profile_id,
                "tenant": snapshot.tenant_id,
                "account": snapshot.account_id,
                "capability": snapshot.capability,
                "timestamp": snapshot.source_event_at,
                "independence_key": snapshot.independence_group_id,
                "freshness_state": snapshot.freshness_state,
                "temporal_class": snapshot.temporal_class,
                "inclusion_reason": item.reason_code,
                "direction": ranked.direction,
                "content_hash": snapshot.content_hash,
            }
        )

    people: list[dict[str, Any]] = []
    for candidate in bundle.people:
        emails = sorted(
            {
                identity.removeprefix("email:")
                for identity in candidate.exact_identities
                if identity.startswith("email:") and identity.removeprefix("email:")
            }
        )
        people.append(
            {
                "person_id": candidate.person_id,
                "display_name": emails[0] if emails else candidate.person_id,
                "emails": emails,
                "source_records": [
                    {
                        "source_id": source_profile_id,
                        "source_type": "retrieved_identity_source",
                        "record_id": "",
                        "label": "",
                        "email": "",
                    }
                    for source_profile_id in candidate.source_profile_ids
                ]
                + [
                    {
                        "source_id": "",
                        "source_type": "retrieved_identity_record",
                        "record_id": source_record_id,
                        "label": "",
                        "email": "",
                    }
                    for source_record_id in candidate.source_record_ids
                ],
                "match_reasons": list(candidate.match_reasons),
                "exact_identities": list(candidate.exact_identities),
            }
        )

    source_contexts = [
        {
            "source_id": scope.source_profile_id,
            "source_profile": scope.source_profile_id,
            "account_id": scope.account_id,
            "tenant_id": scope.tenant_id,
            "capabilities": list(request.capabilities),
            "retrieval_scope": "explicit",
        }
        for scope in request.scopes
    ]
    return {
        "people": people,
        "provenance_sources": provenance_sources,
        "source_contexts": source_contexts,
        "retrieval": {
            "schema_version": IDENTITY_RETRIEVAL_CONTEXT_SCHEMA_VERSION,
            "request_id": request.request_id,
            "bundle_id": persisted.bundle_id,
            "bundle_content_hash": persisted.content_hash,
            "status": persisted.status,
            "retrieval_version": request.retrieval_version,
            "ranking_version": request.ranking_version,
            "as_of": request.as_of,
            "conversation_at": request.conversation_at,
            "freshness_policy": request.freshness_policy,
            "hindsight_policy": request.hindsight_policy,
            "budgets": dict(request.budgets),
            "evidence": evidence_context,
            "warnings": list(bundle.warnings),
            "source_failures": [
                {
                    key: str(value)
                    for key, value in item.items()
                    if key in _RETRIEVAL_FAILURE_FIELDS
                }
                for item in bundle.source_failures
            ],
            "allowlists": dict(persisted.allowlists),
        },
    }


def _normalized_ids(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    }


def _clue_discovery_reference_issues(
    packet: dict[str, Any],
    readout: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared_by_speaker = {
        str(item.get("speaker_label") or "").strip(): sorted(
            _normalized_ids(
                [
                    clue.get("utterance_id")
                    for clue in item.get("utterance_clues", [])
                    if isinstance(clue, dict)
                ]
            )
        )
        for item in packet.get("speakers", [])
        if isinstance(item, dict)
    }
    all_prepared = sorted(
        {
            clue_id
            for clue_ids in prepared_by_speaker.values()
            for clue_id in clue_ids
        }
    )
    issues: list[dict[str, Any]] = []
    speaker_clues = readout.get("speaker_clues")
    for index, result in enumerate(
        speaker_clues if isinstance(speaker_clues, list) else []
    ):
        if not isinstance(result, dict):
            continue
        label = str(result.get("speaker_label") or "").strip()
        allowed = prepared_by_speaker.get(label, [])
        invalid = sorted(_normalized_ids(result.get("transcript_clue_ids")) - set(allowed))
        if invalid:
            issues.append(
                {
                    "path": f"speaker_clues[{index}].transcript_clue_ids",
                    "invalid_ids": invalid,
                    "allowed_ids": allowed,
                }
            )
    conversation_clues = readout.get("conversation_clues")
    for index, result in enumerate(
        conversation_clues if isinstance(conversation_clues, list) else []
    ):
        if not isinstance(result, dict):
            continue
        invalid = sorted(
            _normalized_ids(result.get("transcript_clue_ids")) - set(all_prepared)
        )
        if invalid:
            issues.append(
                {
                    "path": f"conversation_clues[{index}].transcript_clue_ids",
                    "invalid_ids": invalid,
                    "allowed_ids": all_prepared,
                }
            )
    for collection_name in ("speaker_group_hints", "mixed_speaker_hints"):
        collection = readout.get(collection_name)
        for index, result in enumerate(
            collection if isinstance(collection, list) else []
        ):
            if not isinstance(result, dict):
                continue
            label = str(result.get("speaker_label") or "").strip()
            allowed = (
                prepared_by_speaker.get(label, [])
                if collection_name == "mixed_speaker_hints"
                else all_prepared
            )
            invalid = sorted(
                _normalized_ids(result.get("transcript_clue_ids")) - set(allowed)
            )
            if invalid:
                issues.append(
                    {
                        "path": (
                            f"{collection_name}[{index}].transcript_clue_ids"
                        ),
                        "invalid_ids": invalid,
                        "allowed_ids": allowed,
                    }
                )
    return issues


def _identity_reference_allowlists(packet: dict[str, Any]) -> dict[str, list[str]]:
    utterance_ids = sorted(
        {
            str(clue.get("utterance_id") or "").strip()
            for speaker in packet.get("speakers", [])
            if isinstance(speaker, dict)
            for clue in speaker.get("utterance_clues", [])
            if isinstance(clue, dict) and str(clue.get("utterance_id") or "").strip()
        }
    )
    provenance_source_ids = {
        str(source.get("source_id") or "").strip()
        for source in packet.get("provenance_sources", [])
        if isinstance(source, dict) and str(source.get("source_id") or "").strip()
    }
    provenance_source_ids.update(
        str(context.get("source_id") or "").strip()
        for context in packet.get("source_contexts", [])
        if isinstance(context, dict) and str(context.get("source_id") or "").strip()
    )
    person_ids: set[str] = set()
    evidence_ids = set(utterance_ids)
    conversation = (
        packet.get("conversation")
        if isinstance(packet.get("conversation"), dict)
        else {}
    )
    evidence_ids.update(
        _normalized_ids(conversation.get("recording_ids"))
    )
    conversation_id = str(conversation.get("conversation_id") or "").strip()
    if conversation_id:
        evidence_ids.add(conversation_id)
    calendar = (
        packet.get("calendar_context")
        if isinstance(packet.get("calendar_context"), dict)
        else {}
    )
    event_id = str(calendar.get("event_id") or "").strip()
    if event_id:
        evidence_ids.add(event_id)
    evidence_ids.update(
        str(attendee.get("id") or "").strip()
        for attendee in calendar.get("attendees", [])
        if isinstance(attendee, dict) and str(attendee.get("id") or "").strip()
    )
    evidence_ids.update(provenance_source_ids)
    for person in packet.get("people", []):
        if not isinstance(person, dict):
            continue
        person_id = str(person.get("person_id") or "").strip()
        if person_id:
            person_ids.add(person_id)
            evidence_ids.add(person_id)
        evidence_ids.update(_normalized_ids(person.get("emails")))
        for record in person.get("source_records", []):
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("record_id") or "").strip()
            source_id = str(record.get("source_id") or "").strip()
            if record_id:
                evidence_ids.add(record_id)
            if source_id:
                provenance_source_ids.add(source_id)
    evidence_ids.update(provenance_source_ids)
    return {
        "evidence_ids": sorted(evidence_ids),
        "utterance_ids": utterance_ids,
        "provenance_source_ids": sorted(provenance_source_ids),
        "person_ids": sorted(person_ids),
    }


def _append_invalid_reference_issue(
    issues: list[dict[str, Any]],
    *,
    path: str,
    values: Any,
    allowed_ids: list[str],
    scalar: bool = False,
) -> None:
    if scalar:
        normalized = str(values or "").strip()
        invalid = [] if normalized in set(allowed_ids) else [normalized or "<empty>"]
    else:
        invalid = sorted(_normalized_ids(values) - set(allowed_ids))
    if invalid:
        issues.append(
            {
                "path": path,
                "invalid_ids": invalid,
                "allowed_ids": allowed_ids,
            }
        )


def _identity_evaluation_reference_issues(
    packet: dict[str, Any],
    readout: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed = _identity_reference_allowlists(packet)
    issues: list[dict[str, Any]] = []

    calendar = readout.get("calendar_association")
    if isinstance(calendar, dict):
        factors = calendar.get("factors")
        for factor_index, factor in enumerate(
            factors if isinstance(factors, list) else []
        ):
            if isinstance(factor, dict):
                _append_invalid_reference_issue(
                    issues,
                    path=(
                        "calendar_association.factors"
                        f"[{factor_index}].evidence_ids"
                    ),
                    values=factor.get("evidence_ids"),
                    allowed_ids=allowed["evidence_ids"],
                )
    person_links = readout.get("person_links")
    for link_index, link in enumerate(
        person_links if isinstance(person_links, list) else []
    ):
        if not isinstance(link, dict):
            continue
        factors = link.get("factors")
        for factor_index, factor in enumerate(
            factors if isinstance(factors, list) else []
        ):
            if isinstance(factor, dict):
                _append_invalid_reference_issue(
                    issues,
                    path=(
                        f"person_links[{link_index}].factors"
                        f"[{factor_index}].evidence_ids"
                    ),
                    values=factor.get("evidence_ids"),
                    allowed_ids=allowed["evidence_ids"],
                )
    assignments = readout.get("speaker_assignments")
    for assignment_index, assignment in enumerate(
        assignments if isinstance(assignments, list) else []
    ):
        if not isinstance(assignment, dict):
            continue
        base_path = f"speaker_assignments[{assignment_index}]"
        _append_invalid_reference_issue(
            issues,
            path=f"{base_path}.transcript_clue_ids",
            values=assignment.get("transcript_clue_ids"),
            allowed_ids=allowed["utterance_ids"],
        )
        _append_invalid_reference_issue(
            issues,
            path=f"{base_path}.provenance_source_ids",
            values=assignment.get("provenance_source_ids"),
            allowed_ids=allowed["provenance_source_ids"],
        )
        factors = assignment.get("factors")
        for factor_index, factor in enumerate(
            factors if isinstance(factors, list) else []
        ):
            if isinstance(factor, dict):
                _append_invalid_reference_issue(
                    issues,
                    path=f"{base_path}.factors[{factor_index}].evidence_ids",
                    values=factor.get("evidence_ids"),
                    allowed_ids=allowed["evidence_ids"],
                )
        utterances = assignment.get("utterance_assignments")
        for utterance_index, utterance in enumerate(
            utterances if isinstance(utterances, list) else []
        ):
            if isinstance(utterance, dict):
                _append_invalid_reference_issue(
                    issues,
                    path=(
                        f"{base_path}.utterance_assignments"
                        f"[{utterance_index}].utterance_id"
                    ),
                    values=utterance.get("utterance_id"),
                    allowed_ids=allowed["utterance_ids"],
                    scalar=True,
                )
    return issues


def prepare_reference_repair(
    *,
    phase: str,
    document_id: str,
    document_title: str,
    state_root: Path,
    original_run_id: str,
    original_packet: dict[str, Any],
    rejected_readout: dict[str, Any],
    route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare one unsent corrective turn for invalid prepared-reference IDs."""
    if phase not in {"clue_discovery", "identity_evaluation"}:
        raise ValueError(f"Unsupported speaker reference-repair phase: {phase}.")
    try:
        if phase == "clue_discovery":
            validate_clue_discovery_readout(original_packet, rejected_readout)
        else:
            validate_and_score_identity_evaluation(
                original_packet,
                rejected_readout,
            )
    except ValueError:
        issues = (
            _clue_discovery_reference_issues(original_packet, rejected_readout)
            if phase == "clue_discovery"
            else _identity_evaluation_reference_issues(
                original_packet,
                rejected_readout,
            )
        )
        if not issues:
            raise
    else:
        raise ValueError("Valid model output does not require reference repair.")
    repair_packet = {
        "schema_version": REFERENCE_REPAIR_PACKET_SCHEMA_VERSION,
        "task": f"speaker_{phase}_reference_repair",
        "phase": phase,
        "original_run_id": original_run_id,
        "invalid_reference_fields": issues,
        "rejected_readout": rejected_readout,
        "policy": {
            "reference_only_repair": True,
            "perform_no_retrieval": True,
            "use_only_allowed_ids": True,
            "return_complete_corrected_readout": True,
        },
    }
    prompt_text = (
        "Correct only the invalid prepared-reference fields in the rejected JSON. "
        "Use only the exact allowed_ids listed for each invalid field. Do not add "
        "new evidence, retrieve information, change substantive conclusions, or "
        "refer to any transcript or provenance content not already represented in "
        "the rejected JSON. Return the complete corrected JSON object only.\n\n"
        f"Reference repair packet:\n"
        f"{json.dumps(repair_packet, sort_keys=True, ensure_ascii=False)}"
    )
    result = _prepare_run(
        phase=f"{phase}_reference_repair",
        document_id=document_id,
        document_title=document_title,
        state_root=state_root,
        route=_route(route),
        prompt_text=prompt_text,
        input_packet=repair_packet,
    )
    return {**result, "repair_packet": repair_packet}


def persist_identity_evaluation(
    transcript_path: Path,
    *,
    packet: dict[str, Any],
    readout: dict[str, Any],
    run_references: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Validate, score, and append one immutable review-gated evaluation."""
    validated = validate_and_score_identity_evaluation(packet, readout)
    scored_readout = validated["readout"]
    proposals: list[dict[str, Any]] = []
    for assignment in scored_readout.get("speaker_assignments", []):
        if not isinstance(assignment, dict):
            continue
        proposals.append(
            {
                **assignment,
                "proposal_id": str(assignment.get("proposal_id") or uuid4()),
            }
        )
    evidence_snapshots = [
        {
            "source_type": source.get("source_type"),
            "source_id": source.get("source_id"),
            "label": source.get("label"),
            "snippet": source.get("snippet"),
            "profile": source.get("profile"),
            "tenant": source.get("tenant"),
            "account": source.get("account"),
            "capability": source.get("capability"),
            "email": source.get("email"),
            "timestamp": source.get("timestamp"),
            "independence_key": source.get("independence_key"),
            "freshness_state": source.get("freshness_state"),
            "temporal_class": source.get("temporal_class"),
            "inclusion_reason": source.get("inclusion_reason"),
            "direction": source.get("direction"),
            "content_hash": source.get("content_hash"),
        }
        for source in packet.get("provenance_sources", [])
        if isinstance(source, dict)
    ]
    evaluation = {
        "evaluation_id": packet["evaluation_id"],
        "status": "awaiting_human_confirmation",
        "run_references": dict(run_references or {}),
        "input_schema_versions": {
            "identity_evaluation_packet": packet.get("schema_version"),
            "clue_discovery_readout": (packet.get("discovery_readout") or {}).get(
                "schema_version"
            ),
        },
        "conversation": packet.get("conversation") or {},
        "source_contexts": packet.get("source_contexts") or [],
        "people": packet.get("people") or [],
        "evidence_snapshots": evidence_snapshots,
        "retrieval": packet.get("retrieval") or {},
        "calendar_association": scored_readout.get("calendar_association") or {},
        "person_links": scored_readout.get("person_links") or [],
        "person_group_proposals": validated.get("person_group_proposals") or [],
        "proposals": proposals,
        "warnings": scored_readout.get("warnings") or [],
        "rubric_versions": {
            name: rubric.get("version")
            for name, rubric in (packet.get("rubrics") or {}).items()
            if isinstance(rubric, dict)
        },
        "safe_bulk_confirm_ready": validated["safe_bulk_confirm_ready"],
        "review_state": {
            "pending_count": len(proposals),
            "requires_human_confirmation": True,
            "will_apply_assignments": False,
        },
    }
    return append_evaluation(transcript_path, evaluation)


def confirm_ready_proposals(
    transcript_path: Path,
    *,
    evaluation_id: str,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    """Confirm only high-strength, unflagged Candidate Matches in one conversation."""
    sidecar_path = processing_sidecar_path(transcript_path)
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    evaluation = next(
        (
            item
            for item in record.get("evaluations", [])
            if isinstance(item, dict) and item.get("evaluation_id") == evaluation_id
        ),
        None,
    )
    if evaluation is None:
        raise ValueError(f"Evaluation does not exist: {evaluation_id}.")
    latest_actions: dict[str, str] = {}
    for decision in record.get("review_decisions", []):
        if isinstance(decision, dict) and decision.get("evaluation_id") == evaluation_id:
            latest_actions[str(decision.get("proposal_id") or "")] = str(
                decision.get("action") or ""
            )
    ready = [
        proposal
        for proposal in evaluation.get("proposals", [])
        if isinstance(proposal, dict)
        and proposal.get("status") == "candidate_match"
        and (proposal.get("confidence") or {}).get("numeric", 0) >= 85
        and not proposal.get("review_flags")
        and latest_actions.get(str(proposal.get("proposal_id") or "")) not in {
            "confirm",
            "reject",
        }
    ]
    confirmed_ids: list[str] = []
    for proposal in ready:
        proposal_id = str(proposal.get("proposal_id") or "")
        record = append_review_decision(
            transcript_path,
            evaluation_id=evaluation_id,
            proposal_id=proposal_id,
            action="confirm",
            reviewer=reviewer,
            method="conversation_bulk_ready",
            note=note,
        )
        confirmed_ids.append(proposal_id)
    return {
        "confirmed_proposal_ids": confirmed_ids,
        "skipped_count": len(evaluation.get("proposals", [])) - len(confirmed_ids),
        "record": record,
        "will_perform_external_write": False,
        "will_rewrite_diarization": False,
    }
