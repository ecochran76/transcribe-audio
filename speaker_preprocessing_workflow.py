#!/usr/bin/env python3
"""Host-owned two-phase App Intelligence preparation for speaker identity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

import app_intelligence_ledger
import intelligence_config
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
    validate_and_score_identity_evaluation,
)


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
    route: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare, but never send, the second phase after validated discovery."""
    transcript = _read_transcript(transcript_path)
    packet = build_identity_evaluation_packet(
        transcript=transcript,
        discovery_readout=discovery_readout,
        person_records=person_records,
        provenance_sources=provenance_sources,
        source_contexts=source_contexts,
    )
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
            "email": source.get("email"),
            "timestamp": source.get("timestamp"),
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
