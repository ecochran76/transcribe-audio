#!/usr/bin/env python3
"""Persist conversation-owned preprocessing history beside transcript artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

SCHEMA_VERSION = "transcribe-audio.conversation-processing.v1"
TRANSCRIPT_SUFFIX = ".transcript.json"
PROCESSING_SUFFIX = ".processing.json"


class ConversationProcessingError(ValueError):
    """Raised when transcript or processing identity is invalid."""


def _opaque_id(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        return str(UUID(text))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ConversationProcessingError(f"{field} must be a durable opaque UUID.") from exc


def processing_sidecar_path(transcript_path: Path) -> Path:
    """Return the conversation processing sidecar beside a transcript artifact."""
    name = transcript_path.name
    if not name.endswith(TRANSCRIPT_SUFFIX):
        raise ConversationProcessingError(f"Transcript artifact must end with {TRANSCRIPT_SUFFIX}.")
    return transcript_path.with_name(f"{name[:-len(TRANSCRIPT_SUFFIX)]}{PROCESSING_SUFFIX}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversationProcessingError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConversationProcessingError(f"{path} must contain a JSON object.")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_transcript_identity(transcript_path: Path) -> dict[str, Any]:
    """Lazily add durable IDs to a normalized legacy transcript artifact."""
    transcript = _read_json(transcript_path)
    changed = False
    for field in ("conversation_id", "recording_id"):
        try:
            value = _opaque_id(transcript.get(field), field=field)
        except ConversationProcessingError:
            value = str(uuid4())
            transcript[field] = value
            changed = True
    if int(transcript.get("schema_version") or 0) < 2:
        transcript["schema_version"] = 2
        changed = True
    if changed:
        _atomic_write_json(transcript_path, transcript)
    return transcript


def append_evaluation(
    transcript_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Append one immutable evaluation and make it current for the conversation."""
    transcript = ensure_transcript_identity(transcript_path)
    conversation_id = _opaque_id(transcript.get("conversation_id"), field="conversation_id")
    recording_id = _opaque_id(transcript.get("recording_id"), field="recording_id")
    sidecar_path = processing_sidecar_path(transcript_path)

    if sidecar_path.exists():
        record = _read_json(sidecar_path)
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ConversationProcessingError(f"Unsupported processing schema: {record.get('schema_version')}.")
        if _opaque_id(record.get("conversation_id"), field="conversation_id") != conversation_id:
            raise ConversationProcessingError("Processing sidecar belongs to a different conversation.")
    else:
        record = {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": conversation_id,
            "recording_ids": [],
            "current_evaluation_id": "",
            "evaluations": [],
        }

    recording_ids = [
        _opaque_id(value, field="recording_id")
        for value in record.get("recording_ids", [])
    ]
    if recording_id not in recording_ids:
        recording_ids.append(recording_id)

    prepared = dict(evaluation)
    evaluation_id = _opaque_id(
        prepared.get("evaluation_id") or str(uuid4()),
        field="evaluation_id",
    )
    prepared["evaluation_id"] = evaluation_id
    evaluations = record.get("evaluations")
    if not isinstance(evaluations, list):
        raise ConversationProcessingError("Processing sidecar evaluations must be a list.")
    if any(
        isinstance(item, dict) and item.get("evaluation_id") == evaluation_id
        for item in evaluations
    ):
        raise ConversationProcessingError(f"Evaluation already exists: {evaluation_id}.")

    result = {
        **record,
        "recording_ids": recording_ids,
        "current_evaluation_id": evaluation_id,
        "evaluations": [*evaluations, prepared],
    }
    _atomic_write_json(sidecar_path, result)
    return result


def append_review_decision(
    transcript_path: Path,
    *,
    evaluation_id: str,
    proposal_id: str,
    action: str,
    reviewer: str,
    method: str,
    note: str = "",
    supersedes_decision_id: str = "",
    reviewer_asserted_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one attributable review decision without altering its evaluation."""
    evaluation_id = _opaque_id(evaluation_id, field="evaluation_id")
    proposal_id = str(proposal_id or "").strip()
    action = str(action or "").strip().lower()
    reviewer = str(reviewer or "").strip()
    method = str(method or "").strip()
    if action not in {"confirm", "reject", "defer"}:
        raise ConversationProcessingError("Review action must be confirm, reject, or defer.")
    if not proposal_id or not reviewer or not method:
        raise ConversationProcessingError(
            "Review decision requires proposal_id, reviewer, and decision method."
        )
    sidecar_path = processing_sidecar_path(transcript_path)
    record = _read_json(sidecar_path)
    evaluations = record.get("evaluations")
    evaluation = next(
        (
            item
            for item in evaluations or []
            if isinstance(item, dict) and item.get("evaluation_id") == evaluation_id
        ),
        None,
    )
    if evaluation is None:
        raise ConversationProcessingError(f"Evaluation does not exist: {evaluation_id}.")
    proposals = evaluation.get("proposals")
    if not isinstance(proposals, list) or not any(
        isinstance(item, dict) and item.get("proposal_id") == proposal_id
        for item in proposals
    ):
        raise ConversationProcessingError(f"Proposal does not exist: {proposal_id}.")

    decisions = record.get("review_decisions")
    if decisions is None:
        decisions = []
    if not isinstance(decisions, list):
        raise ConversationProcessingError("Processing sidecar review_decisions must be a list.")
    if supersedes_decision_id and not any(
        isinstance(item, dict) and item.get("decision_id") == supersedes_decision_id
        for item in decisions
    ):
        raise ConversationProcessingError(
            f"Superseded review decision does not exist: {supersedes_decision_id}."
        )
    asserted = reviewer_asserted_identity if isinstance(reviewer_asserted_identity, dict) else {}
    bounded_asserted = {
        key: str(asserted.get(key) or "").strip()
        for key in ("name", "email", "organization")
        if str(asserted.get(key) or "").strip()
    }
    decision = {
        "decision_id": str(uuid4()),
        "evaluation_id": evaluation_id,
        "proposal_id": proposal_id,
        "action": action,
        "reviewer": reviewer,
        "decision_method": method,
        "decided_at": _utc_now(),
        "reviewer_note": str(note or "").strip(),
        "supersedes_decision_id": str(supersedes_decision_id or "").strip(),
    }
    if bounded_asserted:
        decision["reviewer_asserted_identity"] = bounded_asserted
    result = {**record, "review_decisions": [*decisions, decision]}
    _atomic_write_json(sidecar_path, result)
    return result
