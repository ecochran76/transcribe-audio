#!/usr/bin/env python3
"""
Read-only local HTTP API for the transcript review console.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import intelligence_config
import automation_config
import codex_app_server_client
import participant_identity
import provenance_config
from app_intelligence_ledger import (
    append_codex_event as append_app_intelligence_codex_event,
    apply_validated_structured_decision as apply_app_intelligence_structured_decision,
    create_run as create_app_intelligence_run,
    list_runs as list_app_intelligence_runs,
    mark_session_started as mark_app_intelligence_session_started,
    model_turn_send_preflight as preflight_app_intelligence_model_turn_send,
    preflight_fork_branches as preflight_app_intelligence_fork_branches,
    preflight_rollback as preflight_app_intelligence_rollback,
    prepare_model_turn_packet as prepare_app_intelligence_model_turn_packet,
    read_model_turn_packet as read_app_intelligence_model_turn_packet,
    record_human_review_decision as record_app_intelligence_human_review_decision,
    record_model_turn_failed as record_app_intelligence_model_turn_failed,
    record_model_turn_started as record_app_intelligence_model_turn_started,
    record_model_turn_status as record_app_intelligence_model_turn_status,
    record_session_start_failed as record_app_intelligence_session_start_failed,
    record_session_start_requested as record_app_intelligence_session_start_requested,
    response_for_run as get_app_intelligence_run,
    session_start_preflight as preflight_app_intelligence_session_start,
    validate_latest_structured_decision as validate_app_intelligence_structured_decision,
)
from transcript_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_STORE_DIR,
    TranscriptStoreError,
    connect,
    context_for_document,
    db_path,
    init_db,
    legacy_enrichment_queue,
    parse_object_json,
    search_store,
    stable_id,
    store_dir,
    utcish_now,
)
from transcribe_common import TranscriptionError
from routing_artifacts import unique_strings

DEFAULT_API_PORT = 18876
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_BATCH_ENV_FILE = Path("~/.local/state/transcribe-audio/auracall-transcripts.env")
DEFAULT_CODEX_BIN = "codex"
MAX_READINESS_OUTPUT_CHARS = 2000
MAX_APP_ARTIFACT_BYTES = 512 * 1024
APP_SMOKE_RUN_PREFIX = "smoke-replay-manifest"
APP_BROWSER_SMOKE_DIRNAME = "browser-smokes"
APP_SMOKE_JOB_DIRNAME = "smoke-jobs"
APP_SMOKE_JOB_TOKEN = "RUN_APP_SMOKE_JOB"
APP_SMOKE_CLEANUP_TOKEN = "CLEANUP_APP_SMOKE_ARTIFACTS"
APP_SMOKE_JOB_TIMEOUT_SECONDS = 180
MAX_SMOKE_JOB_TAIL_CHARS = 20000
MAX_SMOKE_EVIDENCE_BYTES = 5 * 1024 * 1024
RETRANSCRIPTION_PREFLIGHT_TOKEN = "QUEUE_RETRANSCRIPTION_JOB"
RETRANSCRIPTION_BACKENDS = {"faster_whisper", "assemblyai"}
RETRANSCRIPTION_JOB_DIRNAME = "retranscription-jobs"
CONTEXT_WORKBENCH_TOKEN = "QUEUE_CONTEXT_WORKBENCH_RUN"
DEPOSITION_MEMORY_PREVIEW_TOKEN = "QUEUE_DEPOSITION_MEMORY_PREVIEW"
FIRST_PASS_SUMMARY_SUBMIT_TOKEN = "SUBMIT_FIRST_PASS_SUMMARY_BATCH"
CONTEXT_WORKBENCH_DIRNAME = "conversation-context-runs"
CONTEXT_CONTACT_SELECTION_DIRNAME = "conversation-context-contact-selections"
CONTEXT_CONTACT_SEARCH_CACHE_DIRNAME = "conversation-context-contact-search-cache"
CONTEXT_CONTACT_REFRESH_DIRNAME = "conversation-context-contact-refresh-jobs"
CONTEXT_CONTACT_AFFINITY_DIRNAME = "conversation-context-contact-affinity-cache"
CONTEXT_CONTACT_MERGE_DIRNAME = "conversation-context-contact-merge-decisions"
CONTEXT_INSTRUCTIONS_DIRNAME = "conversation-context-instructions"
PARTICIPANT_IDENTITY_CACHE_DIRNAME = "participant-identity-bundles"
PARTICIPANT_IDENTITY_CACHE_ALGORITHM = "participant-identity-cache-v3"
CONVERSATION_PREVIEW_DIRNAME = "conversation-preview-decisions"


def document_summary(row: sqlite3.Row) -> dict[str, Any]:
    metadata = parse_object_json(row["metadata_json"])
    media_blob = metadata.get("media_blob") if isinstance(metadata.get("media_blob"), dict) else {}
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "source_path": row["source_path"],
        "stored_path": row["stored_path"],
        "generated_at": row["generated_at"],
        "updated_at": row["updated_at"],
        "embedding_provider": row["embedding_provider"],
        "embedding_model": row["embedding_model"],
        "metadata": metadata,
        "media_blob": media_blob,
    }


def list_documents(
    *,
    root: Optional[Path] = None,
    kind: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        where = ""
        params: list[Any] = []
        if kind:
            where = "WHERE kind = ?"
            params.append(kind)
        total = int(con.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0])
        rows = con.execute(
            f"""
            SELECT * FROM documents
            {where}
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
    return {
        "items": [document_summary(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


def conversation_source_key(row: sqlite3.Row) -> str:
    payload = parse_object_json(row["json_payload"])
    metadata = parse_object_json(row["metadata_json"])
    source_artifact_path = str(metadata.get("source_artifact_path") or payload.get("source_artifact_path") or "")
    if row["kind"] != "transcript" and source_artifact_path:
        return source_artifact_path
    return str(row["source_path"] or row["id"])


def latest_document(rows: list[sqlite3.Row]) -> sqlite3.Row:
    return sorted(
        rows,
        key=lambda row: str(row["generated_at"] or row["updated_at"] or ""),
        reverse=True,
    )[0]


def media_blob_for_document(document: dict[str, Any] | None) -> dict[str, Any]:
    if not document:
        return {}
    media_blob = document.get("media_blob")
    return media_blob if isinstance(media_blob, dict) else {}


def conversation_summary(key: str, rows: list[sqlite3.Row]) -> dict[str, Any]:
    transcripts = [row for row in rows if row["kind"] == "transcript"]
    readouts = [row for row in rows if row["kind"] == "readout"]
    contextual_readouts = [row for row in rows if row["kind"] == "contextual_readout"]
    representative_row = (
        latest_document(contextual_readouts)
        if contextual_readouts
        else latest_document(readouts)
        if readouts
        else latest_document(transcripts)
        if transcripts
        else latest_document(rows)
    )
    source_row = latest_document(transcripts) if transcripts else latest_document(rows)
    latest_row = latest_document(rows)
    representative = document_summary(representative_row)
    source = document_summary(source_row)
    latest = document_summary(latest_row)
    media_blob = media_blob_for_document(representative) or media_blob_for_document(source)
    source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    rep_metadata = representative.get("metadata") if isinstance(representative.get("metadata"), dict) else {}
    calendar = (
        source_metadata.get("event", {}).get("summary")
        if isinstance(source_metadata.get("event"), dict)
        else ""
    ) or (
        rep_metadata.get("event", {}).get("summary")
        if isinstance(rep_metadata.get("event"), dict)
        else ""
    ) or (
        rep_metadata.get("route", {}).get("label")
        if isinstance(rep_metadata.get("route"), dict)
        else ""
    )
    return {
        "key": key,
        "title": representative.get("title") or source.get("title") or "Untitled conversation",
        "representative": representative,
        "source": source,
        "latest_artifact": latest,
        "artifacts": [document_summary(row) for row in rows],
        "artifact_count": len(rows),
        "workflow": {
            "transcript": bool(transcripts),
            "summary": bool(readouts),
            "contextual_readout": bool(contextual_readouts),
        },
        "media_blob": media_blob,
        "media_ready": bool(media_blob.get("playback_url")),
        "calendar_label": calendar or "No context yet",
        "updated_at": latest.get("generated_at") or latest.get("updated_at") or "",
    }


def list_conversations(
    *,
    root: Optional[Path] = None,
    kind: str = "",
    query: str = "",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        rows = con.execute(
            """
            SELECT * FROM documents
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            """
        ).fetchall()
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(conversation_source_key(row), []).append(row)
    conversations = [conversation_summary(key, group_rows) for key, group_rows in groups.items()]
    if kind:
        conversations = [
            conversation
            for conversation in conversations
            if any(artifact.get("kind") == kind for artifact in conversation["artifacts"])
        ]
    if query:
        needle = query.lower()
        conversations = [
            conversation
            for conversation in conversations
            if needle
            in " ".join(
                [
                    str(conversation.get("title") or ""),
                    str(conversation.get("calendar_label") or ""),
                    " ".join(str(artifact.get("title") or "") for artifact in conversation["artifacts"]),
                    " ".join(str(artifact.get("source_path") or "") for artifact in conversation["artifacts"]),
                    " ".join(str(row["text_content"] or "") for row in groups[conversation["key"]]),
                ]
            ).lower()
        ]
    conversations.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    total = len(conversations)
    return {
        "schema_version": "transcribe-audio.conversations.v1",
        "items": conversations[offset : offset + limit],
        "limit": limit,
        "offset": offset,
        "total": total,
        "will_read_artifact_files": False,
        "will_return_artifact_content": False,
    }


def compact_label(value: Any, fallback: str = "") -> str:
    if isinstance(value, dict):
        for key in ("name", "label", "email", "title", "text"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return fallback
    return str(value or fallback).strip()


def truncate_text(value: Any, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def speaker_labels_from_transcript(document: dict[str, Any] | None) -> list[str]:
    if not document:
        return []
    payload = document.get("json_payload") if isinstance(document.get("json_payload"), dict) else {}
    labels: list[str] = []
    for utterance in payload.get("utterances") or []:
        if not isinstance(utterance, dict):
            continue
        label = compact_label(utterance.get("speaker"), "Speaker")
        if label and label not in labels:
            labels.append(label)
    text = str(payload.get("transcript_text") or document.get("text_content") or "")
    for line in text.splitlines():
        match = re.match(r"^(.{1,64}?)\s+\[[^\]]+\]:", line)
        if not match:
            continue
        label = match.group(1).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def contact_candidate_from_label(label: str, *, source: str, confidence: float, evidence: str = "") -> dict[str, Any]:
    return {
        "contact_id": stable_id("contact-candidate", label.lower()),
        "label": label,
        "email": "",
        "source": source,
        "confidence": confidence,
        "evidence": evidence,
    }


def contact_row_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "contact_id": row["id"],
        "label": row["label"],
        "email": row["email"],
        "external_ref": row["external_ref"],
        "metadata": parse_object_json(row["metadata_json"]),
        "source": "contact_table",
        "confidence": 0.75,
        "evidence": "Previously reviewed contact record.",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def assignment_row_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "assignment_id": row["id"],
        "conversation_key": row["conversation_key"],
        "document_id": row["document_id"],
        "speaker_label": row["speaker_label"],
        "contact_id": row["contact_id"],
        "contact_label": row["contact_label"],
        "status": row["status"],
        "confidence": row["confidence"],
        "evidence": parse_object_json(row["evidence_json"]) if row["evidence_json"].startswith("{") else json.loads(row["evidence_json"] or "[]"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def participant_identity_cache_dir(state_root: Path) -> Path:
    return state_root.expanduser() / PARTICIPANT_IDENTITY_CACHE_DIRNAME


def participant_identity_cache_path(*, state_root: Path, conversation_key: str, source_document_id: str) -> Path:
    return participant_identity_cache_dir(state_root) / f"{stable_id('participant-identity-cache', conversation_key, source_document_id)}.json"


def identity_cache_fingerprint(
    *,
    conversation_key: str,
    source_document: dict[str, Any] | None,
    participants: list[Any],
    contacts: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    state_root: Path,
) -> str:
    source = source_document or {}
    source_payload = source.get("json_payload") if isinstance(source.get("json_payload"), dict) else {}
    contact_config = participant_identity.load_contact_source_config(state_root)
    contact_aliases = participant_identity.load_contact_aliases(state_root)
    contact_settings = participant_identity.load_contact_settings(state_root)
    payload = {
        "cache_algorithm": PARTICIPANT_IDENTITY_CACHE_ALGORITHM,
        "conversation_key": conversation_key,
        "source_document_id": source.get("id") or "",
        "source_path": source.get("source_path") or "",
        "source_updated_at": source.get("updated_at") or "",
        "source_generated_at": source.get("generated_at") or "",
        "source_event": source_payload.get("event") if isinstance(source_payload, dict) else {},
        "participants": participants,
        "contacts": [
            {
                "id": contact.get("contact_id") or contact.get("id") or "",
                "label": contact.get("label") or "",
                "email": contact.get("email") or "",
                "updated_at": contact.get("updated_at") or "",
            }
            for contact in contacts
        ],
        "assignments": assignments,
        "contact_config": contact_config,
        "contact_aliases": contact_aliases,
        "contact_settings": contact_settings,
    }
    return stable_id("participant-identity-fingerprint", json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def cached_participant_identity_bundle(
    *,
    conversation_key: str,
    source_document: dict[str, Any] | None,
    participants: list[Any],
    contacts: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    state_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source_document_id = str((source_document or {}).get("id") or "")
    if not conversation_key or not source_document_id:
        return None, {"status": "disabled"}
    fingerprint = identity_cache_fingerprint(
        conversation_key=conversation_key,
        source_document=source_document,
        participants=participants,
        contacts=contacts,
        assignments=assignments,
        state_root=state_root,
    )
    path = participant_identity_cache_path(
        state_root=state_root,
        conversation_key=conversation_key,
        source_document_id=source_document_id,
    )
    payload = read_json_file(path) if path.exists() else {}
    if payload.get("fingerprint") == fingerprint and isinstance(payload.get("identity_bundle"), dict):
        bundle = dict(payload["identity_bundle"])
        bundle["cache_status"] = "hit"
        return bundle, {"status": "hit", "path": str(path), "fingerprint": fingerprint}
    return None, {"status": "miss", "path": str(path), "fingerprint": fingerprint}


def write_participant_identity_bundle_cache(
    *,
    identity_bundle: dict[str, Any],
    conversation_key: str,
    source_document: dict[str, Any] | None,
    fingerprint: str,
    state_root: Path,
) -> str:
    source_document_id = str((source_document or {}).get("id") or "")
    if not conversation_key or not source_document_id:
        return ""
    now = utcish_now()
    path = participant_identity_cache_path(
        state_root=state_root,
        conversation_key=conversation_key,
        source_document_id=source_document_id,
    )
    payload = {
        "schema_version": "transcribe-audio.participant-identity-cache.v1",
        "conversation_key": conversation_key,
        "source_document_id": source_document_id,
        "fingerprint": fingerprint,
        "updated_at": now,
        "identity_bundle": identity_bundle,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, payload)
    return str(path)


def conversation_identity_review(
    *,
    conversation_key: str,
    source_document: dict[str, Any] | None,
    participants: list[Any],
    root: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    resolved_state_root = state_root or DEFAULT_STATE_DIR.expanduser()
    with connect(root) as con:
        init_db(con)
        contact_rows = con.execute("SELECT * FROM contacts ORDER BY updated_at DESC, label LIMIT 50").fetchall()
        assignment_rows = con.execute(
            "SELECT * FROM speaker_assignments WHERE conversation_key = ? ORDER BY speaker_label",
            (conversation_key,),
        ).fetchall()
    contacts = [contact_row_summary(row) for row in contact_rows]
    assignments = {row["speaker_label"]: assignment_row_summary(row) for row in assignment_rows}
    source_payload = source_document.get("json_payload") if source_document and isinstance(source_document.get("json_payload"), dict) else {}
    identity_bundle, cache_meta = cached_participant_identity_bundle(
        conversation_key=conversation_key,
        source_document=source_document,
        participants=participants,
        contacts=contacts,
        assignments=assignments,
        state_root=resolved_state_root,
    )
    if identity_bundle is None:
        identity_bundle = participant_identity.build_participant_identity_bundle(
            conversation_key=conversation_key,
            source_document_id=source_document.get("id") if source_document else "",
            transcript=source_payload,
            transcript_text=str(source_document.get("text_content") or "") if source_document else "",
            readout_participants=participants,
            local_contacts=contacts,
            assignments=assignments,
            state_root=resolved_state_root,
        )
        identity_bundle["cache_status"] = "miss"
        cache_path = write_participant_identity_bundle_cache(
            identity_bundle=identity_bundle,
            conversation_key=conversation_key,
            source_document=source_document,
            fingerprint=str(cache_meta.get("fingerprint") or ""),
            state_root=resolved_state_root,
        )
        cache_meta = {**cache_meta, "status": "stored", "path": cache_path}
    speakers = identity_bundle["speakers"]
    return {
        "schema_version": "transcribe-audio.identity-review.v1",
        "conversation_key": conversation_key,
        "source_document_id": source_document.get("id") if source_document else "",
        "speakers": speakers,
        "contacts": contacts[:20],
        "participants": participants,
        "identity_bundle": identity_bundle,
        "identity_cache": cache_meta,
        "pending_count": sum(1 for speaker in speakers if speaker["review_required"]),
        "confirmed_count": sum(1 for speaker in speakers if speaker["status"] == "confirmed"),
        "deferred_count": sum(1 for speaker in speakers if speaker["status"] == "deferred"),
        "will_execute_external_action": False,
        "will_attempt_automatic_disambiguation": False,
    }


def artifact_path_for_document(document: dict[str, Any] | None) -> Path | None:
    if not document:
        return None
    for key in ("source_path", "stored_path"):
        text = str(document.get(key) or "")
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists() and path.is_file():
            return path
    return None


def path_is_under(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def safe_read_runtime_json(path_text: str, *, root: Optional[Path], state_root: Optional[Path]) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    roots = [store_dir(root), DEFAULT_STORE_DIR.expanduser()]
    if state_root:
        roots.append(state_root.expanduser())
    if not path.exists() or not path.is_file() or not path_is_under(path, roots):
        return {}
    return read_json_file(path)


def compact_provenance_source(source: dict[str, Any]) -> dict[str, Any]:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return {
        "source_id": str(source.get("source_id") or ""),
        "source_type": str(source.get("source_type") or ""),
        "label": truncate_text(source.get("label"), 160),
        "snippet": truncate_text(source.get("snippet"), 260),
        "uri": str(source.get("uri") or ""),
        "quality_status": str(metadata.get("quality_status") or ""),
        "quality_score": metadata.get("quality_score"),
        "quality_reason": truncate_text(metadata.get("quality_reason"), 220),
    }


def contextualization_for_document(
    contextual_document: dict[str, Any] | None,
    *,
    root: Optional[Path],
    state_root: Optional[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not contextual_document:
        return {}, {}
    metadata = contextual_document.get("metadata") if isinstance(contextual_document.get("metadata"), dict) else {}
    payload = contextual_document.get("json_payload") if isinstance(contextual_document.get("json_payload"), dict) else {}
    contextualization = (
        metadata.get("contextualization")
        if isinstance(metadata.get("contextualization"), dict)
        else payload.get("contextualization")
        if isinstance(payload.get("contextualization"), dict)
        else {}
    )
    provider = metadata.get("provider") if isinstance(metadata.get("provider"), dict) else {}
    route_payload = safe_read_runtime_json(str(provider.get("route_decision_path") or ""), root=root, state_root=state_root)
    return contextualization, route_payload


def context_workbench_state(
    *,
    detail: dict[str, Any],
    root: Optional[Path],
    state_root: Optional[Path],
) -> dict[str, Any]:
    contextual_document = detail.get("contextual_readout_document")
    contextualization, route_payload = contextualization_for_document(contextual_document, root=root, state_root=state_root)
    selected = (
        contextualization.get("selected_candidate")
        if isinstance(contextualization.get("selected_candidate"), dict)
        else route_payload.get("selected_candidate")
        if isinstance(route_payload.get("selected_candidate"), dict)
        else {}
    )
    route_pack = route_payload.get("provenance_pack") if isinstance(route_payload.get("provenance_pack"), dict) else {}
    quality_profile = (
        contextualization.get("quality_profile")
        if isinstance(contextualization.get("quality_profile"), dict)
        else route_pack.get("quality_profile")
        if isinstance(route_pack.get("quality_profile"), dict)
        else {}
    )
    included = contextualization.get("supporting_context_sources") or route_pack.get("sources") or []
    excluded = route_pack.get("excluded_sources") or []
    warnings = []
    for source in [contextualization.get("warnings") or [], route_payload.get("warnings") or [], route_pack.get("warnings") or []]:
        for item in source:
            text = str(item or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    identity_bundle = (detail.get("identity_review") or {}).get("identity_bundle", {})
    proposed_contacts = identity_bundle.get("contact_candidates") if isinstance(identity_bundle, dict) else []
    if not isinstance(proposed_contacts, list):
        proposed_contacts = []
    contact_selection = context_contact_selection_state(
        detail=detail,
        state_root=state_root or DEFAULT_STATE_DIR.expanduser(),
        proposed_contacts=proposed_contacts,
    )
    operator_context = context_instructions_state(
        detail=detail,
        state_root=state_root or DEFAULT_STATE_DIR.expanduser(),
    )
    identity_warnings = identity_bundle.get("warnings") if isinstance(identity_bundle, dict) else []
    if isinstance(identity_warnings, list):
        for item in identity_warnings:
            text = str(item or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    excluded_count = contextualization.get("excluded_source_count")
    if not isinstance(excluded_count, int):
        excluded_count = len(excluded)
    status = "contextual_readout_ready" if contextual_document else "needs_context"
    return {
        "schema_version": "transcribe-audio.context-workbench.v1",
        "status": status,
        "identity_status": identity_bundle.get("review_status", "unknown") if isinstance(identity_bundle, dict) else "unknown",
        "participant_identity_bundle": identity_bundle if isinstance(identity_bundle, dict) else {},
        "proposed_contact_candidates": proposed_contacts,
        "contact_selection": contact_selection,
        "operator_context": operator_context,
        "context_instructions": operator_context,
        "selected_candidate": selected,
        "confidence": selected.get("confidence"),
        "included_sources": [compact_provenance_source(source) for source in included if isinstance(source, dict)][:20],
        "excluded_sources": [compact_provenance_source(source) for source in excluded if isinstance(source, dict)][:20],
        "included_source_count": len(included) if isinstance(included, list) else 0,
        "excluded_source_count": excluded_count,
        "warnings": warnings,
        "quality_profile": quality_profile,
        "summary_document_id": (detail.get("summary_document") or {}).get("id") if detail.get("summary_document") else "",
        "contextual_readout_document_id": contextual_document.get("id") if contextual_document else "",
        "route_status": contextualization.get("route_status") or route_payload.get("status") or "",
        "will_execute_external_action": False,
        "will_perform_external_write": False,
        "future_required_approval_token_for_queue": CONTEXT_WORKBENCH_TOKEN,
    }


def context_contact_selection_path(*, state_root: Path, conversation_key: str) -> Path:
    return conversation_context_contact_selections_dir(state_root) / f"{stable_id('context-contact-selection', conversation_key)}.json"


def context_instructions_path(*, state_root: Path, conversation_key: str) -> Path:
    return conversation_context_instructions_dir(state_root) / f"{stable_id('context-instructions', conversation_key)}.json"


def compact_context_contact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []
    merged_sources = candidate.get("merged_sources") if isinstance(candidate.get("merged_sources"), list) else []
    merged_contact_ids = candidate.get("merged_contact_ids") if isinstance(candidate.get("merged_contact_ids"), list) else []
    merge_keys = candidate.get("merge_keys") if isinstance(candidate.get("merge_keys"), list) else []
    relationship_affinity = candidate.get("relationship_affinity") if isinstance(candidate.get("relationship_affinity"), dict) else {}
    ranking_reasons = candidate.get("ranking_reasons") if isinstance(candidate.get("ranking_reasons"), list) else []
    compact = {
        "contact_id": str(candidate.get("contact_id") or candidate.get("id") or ""),
        "canonical_key": str(candidate.get("canonical_key") or ""),
        "label": str(candidate.get("label") or candidate.get("contact_label") or ""),
        "email": str(candidate.get("email") or ""),
        "organization": str(candidate.get("organization") or candidate.get("company") or ""),
        "role": str(candidate.get("role") or candidate.get("title") or ""),
        "phone": str(candidate.get("phone") or ""),
        "source": str(candidate.get("source") or ""),
        "source_type": str(candidate.get("source_type") or candidate.get("source") or ""),
        "source_profile": str(candidate.get("source_profile") or ""),
        "confidence": candidate.get("confidence"),
        "dedupe_key": str(candidate.get("dedupe_key") or participant_identity.candidate_dedupe_key(candidate)),
        "split_merge_key": str(candidate.get("split_merge_key") or ""),
        "merge_keys": [str(value) for value in merge_keys if str(value or "")][:20],
        "source_count": int(candidate.get("source_count") or max(1, len(merged_sources))),
        "merged_contact_ids": [str(value) for value in merged_contact_ids if str(value or "")],
        "merged_sources": [
            {
                "contact_id": str(source.get("contact_id") or ""),
                "label": str(source.get("label") or ""),
                "email": str(source.get("email") or ""),
                "original_label": str(source.get("original_label") or ""),
                "original_email": str(source.get("original_email") or ""),
                "source": str(source.get("source") or ""),
                "source_type": str(source.get("source_type") or source.get("source") or ""),
                "source_profile": str(source.get("source_profile") or ""),
                "confidence": source.get("confidence"),
            }
            for source in merged_sources
            if isinstance(source, dict)
        ],
        "evidence": evidence[:4],
        "review_state": str(candidate.get("review_state") or ""),
        "relationship_affinity": relationship_affinity,
        "rank_score": candidate.get("rank_score"),
        "ranking_reasons": [str(value) for value in ranking_reasons if str(value or "")][:4],
    }
    if compact["contact_id"] and compact["contact_id"] not in compact["merged_contact_ids"]:
        compact["merged_contact_ids"].insert(0, compact["contact_id"])
    return compact


def context_contact_candidate_search_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get("label"),
        candidate.get("email"),
        candidate.get("organization"),
        candidate.get("role"),
        candidate.get("source"),
        candidate.get("source_type"),
        candidate.get("source_profile"),
    ]
    merged_sources = candidate.get("merged_sources") if isinstance(candidate.get("merged_sources"), list) else []
    for source in merged_sources:
        if isinstance(source, dict):
            parts.extend([
                source.get("label"),
                source.get("email"),
                source.get("original_label"),
                source.get("original_email"),
                source.get("source_type"),
                source.get("source_profile"),
            ])
    return " ".join(str(part or "").lower() for part in parts)


def context_contact_candidate_ids(candidate: dict[str, Any]) -> list[str]:
    ids = [str(candidate.get("contact_id") or candidate.get("id") or "").strip()]
    merged_contact_ids = candidate.get("merged_contact_ids") if isinstance(candidate.get("merged_contact_ids"), list) else []
    ids.extend(str(value or "").strip() for value in merged_contact_ids)
    return sorted({value for value in ids if value})


def context_contact_merge_state(*, state_root: Path, conversation_key: str) -> dict[str, Any]:
    if not conversation_key:
        return {
            "schema_version": "transcribe-audio.context-contact-merge.v1",
            "status": "missing_conversation_key",
            "decisions": [],
            "merge_decisions": [],
            "split_decisions": [],
            "decision_path": "",
            "will_execute_external_action": False,
            "will_perform_external_write": False,
        }
    path = context_contact_merge_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    merge_decisions = [item for item in decisions if isinstance(item, dict) and item.get("action") == "merge"]
    split_decisions = [item for item in decisions if isinstance(item, dict) and item.get("action") == "split"]
    return {
        "schema_version": "transcribe-audio.context-contact-merge.v1",
        "status": "reviewed" if decisions else "empty",
        "conversation_key": conversation_key,
        "decision_path": str(path) if path.exists() else "",
        "decisions": decisions[-50:],
        "merge_decisions": merge_decisions[-25:],
        "split_decisions": split_decisions[-25:],
        "allowed_actor_types": ["operator", "app_intelligence"],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def apply_context_contact_merge_policy(
    candidates: Iterable[dict[str, Any]],
    *,
    merge_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    decisions = merge_state.get("decisions") if isinstance(merge_state, dict) else []
    if not isinstance(decisions, list) or not decisions:
        return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]

    merge_by_id: dict[str, dict[str, Any]] = {}
    split_by_id: dict[str, str] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        action = str(decision.get("action") or "")
        contact_ids = [
            str(value or "").strip()
            for value in decision.get("contact_ids", [])
            if str(value or "").strip()
        ] if isinstance(decision.get("contact_ids"), list) else []
        if action == "merge" and len(contact_ids) >= 2:
            canonical = decision.get("canonical_candidate") if isinstance(decision.get("canonical_candidate"), dict) else {}
            merge_key = str(decision.get("merge_key") or decision.get("decision_id") or stable_id("contact-merge", *sorted(contact_ids)))
            for contact_id in contact_ids:
                merge_by_id[contact_id] = {
                    "merge_key": merge_key,
                    "canonical_candidate": canonical,
                    "decision_id": decision.get("decision_id") or "",
                }
        elif action == "split" and contact_ids:
            decision_id = str(decision.get("decision_id") or stable_id("contact-split", *sorted(contact_ids)))
            for contact_id in contact_ids:
                split_by_id[contact_id] = f"{decision_id}:{contact_id}"

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        ids = context_contact_candidate_ids(item)
        split_key = next((split_by_id[contact_id] for contact_id in ids if contact_id in split_by_id), "")
        if split_key:
            primary_id = ids[0] if ids else str(item.get("contact_id") or "")
            item["split_merge_key"] = split_key
            item["dedupe_key"] = f"split:{primary_id or split_key}"
            item.pop("canonical_key", None)
        merge_decision = next((merge_by_id[contact_id] for contact_id in ids if contact_id in merge_by_id), None)
        if merge_decision and not split_key:
            canonical = merge_decision.get("canonical_candidate") or {}
            item["canonical_key"] = str(merge_decision.get("merge_key") or "")
            item["dedupe_key"] = f"alias:{item['canonical_key']}"
            if canonical.get("label"):
                item["label"] = str(canonical.get("label") or "")
            if canonical.get("email"):
                item["email"] = str(canonical.get("email") or "")
            evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
            item["evidence"] = [
                *evidence,
                {
                    "kind": "reviewed_contact_merge",
                    "decision_id": merge_decision.get("decision_id") or "",
                    "merge_key": item["canonical_key"],
                },
            ]
        result.append(item)
    return result


def unique_context_contact_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    merge_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    compact_candidates = [
        compact_context_contact_candidate(candidate)
        for candidate in apply_context_contact_merge_policy(candidates, merge_state=merge_state)
        if isinstance(candidate, dict)
    ]
    merged_candidates = participant_identity.ranked_contact_candidates(
        compact_candidates,
        limit=200,
        per_source_profile=200,
    )
    for candidate in merged_candidates:
        compact = compact_context_contact_candidate(candidate)
        contact_id = compact.get("contact_id") or ""
        dedupe_key = compact.get("dedupe_key") or ""
        if contact_id and contact_id in seen_ids:
            continue
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if contact_id:
            seen_ids.add(contact_id)
        if dedupe_key:
            seen_keys.add(dedupe_key)
        result.append(compact)
    return result


def context_contact_dedupe_clusters(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    seen: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = str(candidate.get("dedupe_key") or participant_identity.candidate_dedupe_key(candidate))
        if key:
            grouped.setdefault(key, []).append(candidate)
        merged_sources = candidate.get("merged_sources") if isinstance(candidate.get("merged_sources"), list) else []
        if len(merged_sources) > 1 and key not in seen:
            clusters.append(
                {
                    "dedupe_key": key,
                    "label": candidate.get("label") or candidate.get("email") or "Contact candidate",
                    "email": candidate.get("email") or "",
                    "contact_ids": candidate.get("merged_contact_ids") or [candidate.get("contact_id")],
                    "source_count": len(merged_sources),
                    "sources": merged_sources,
                }
            )
            seen.add(key)
    for key, values in grouped.items():
        if key in seen or len(values) < 2:
            continue
        clusters.append(
            {
                "dedupe_key": key,
                "label": values[0].get("label") or values[0].get("email") or "Contact candidate",
                "email": values[0].get("email") or "",
                "contact_ids": [value.get("contact_id") for value in values if value.get("contact_id")],
                "source_count": len(values),
                "sources": [
                    {
                        "contact_id": value.get("contact_id") or "",
                        "label": value.get("label") or "",
                        "email": value.get("email") or "",
                        "source": value.get("source") or "",
                        "source_type": value.get("source_type") or value.get("source") or "",
                        "source_profile": value.get("source_profile") or "",
                        "confidence": value.get("confidence"),
                    }
                    for value in values
                ],
            }
        )
        seen.add(key)
    return clusters[:20]


def context_contact_search_cache_state(*, state_root: Path, conversation_key: str) -> dict[str, Any]:
    if not conversation_key:
        return {"entries": [], "items": [], "path": "", "status": "missing_conversation_key"}
    path = context_contact_search_cache_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    items: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        values = entry.get("items") if isinstance(entry.get("items"), list) else []
        items.extend(value for value in values if isinstance(value, dict))
    return {
        "schema_version": "transcribe-audio.context-contact-search-cache.v1",
        "status": "hit" if items else "empty",
        "path": str(path) if path.exists() else "",
        "entries": entries[-25:],
        "items": [compact_context_contact_candidate(item) for item in items],
    }


def append_context_contact_search_cache(
    *,
    state_root: Path,
    conversation_key: str,
    query: str,
    items: list[dict[str, Any]],
    source_profiles: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    path = context_contact_search_cache_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    now = utcish_now()
    entry = {
        "query": query.strip(),
        "created_at": now,
        "item_count": len(items),
        "items": [compact_context_contact_candidate(item) for item in items],
        "source_profiles": source_profiles,
        "warnings": warnings,
        "will_execute_external_action": True,
        "will_perform_external_write": False,
    }
    next_payload = {
        "schema_version": "transcribe-audio.context-contact-search-cache.v1",
        "conversation_key": conversation_key,
        "updated_at": now,
        "entries": [*entries, entry][-25:],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, next_payload)
    return {**entry, "cache_path": str(path)}


def parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date")
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def contact_affinity_person_values(candidate: dict[str, Any]) -> dict[str, set[str]]:
    emails: set[str] = set()
    labels: set[str] = set()
    ids = set(context_contact_candidate_ids(candidate))
    for source in [candidate, *(candidate.get("merged_sources") if isinstance(candidate.get("merged_sources"), list) else [])]:
        if not isinstance(source, dict):
            continue
        email = participant_identity.normalize_email(source.get("email") or source.get("original_email") or source.get("label"))
        if email:
            emails.add(email)
            emails.update(participant_identity.email_alias_keys(email))
        for field in ("label", "original_label"):
            label = participant_identity.candidate_name_text(source.get(field))
            if label and not participant_identity.is_anonymous_speaker_label(label):
                labels.add(label.lower())
    return {"ids": ids, "emails": emails, "labels": labels}


def contact_affinity_matches_value(terms: dict[str, set[str]], value: Any) -> bool:
    if isinstance(value, dict):
        raw_values = [
            value.get("email"),
            value.get("emailAddress"),
            value.get("address"),
            value.get("mail"),
            value.get("displayName"),
            value.get("display_name"),
            value.get("name"),
            value.get("label"),
            value.get("summary"),
            value.get("formatted"),
        ]
    else:
        raw_values = [value]
    text = " ".join(str(item or "") for item in raw_values).lower()
    if not text:
        return False
    value_email = participant_identity.normalize_email(text)
    if value_email and participant_identity.email_alias_keys(value_email) & terms["emails"]:
        return True
    if any(email and email in text for email in terms["emails"]):
        return True
    value_tokens = set(participant_identity.person_name_tokens(text))
    if len(value_tokens) < 2:
        return False
    for label in terms["labels"]:
        label_tokens = set(participant_identity.person_name_tokens(label))
        if len(label_tokens) >= 2 and (label_tokens <= value_tokens or value_tokens <= label_tokens):
            return True
    return False


def event_people_values(event: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for field in ("participants", "attendees", "attendee_emails"):
        field_values = event.get(field)
        if isinstance(field_values, list):
            values.extend(field_values)
    matching = event.get("matching_calendars") if isinstance(event.get("matching_calendars"), list) else []
    for item in matching:
        if not isinstance(item, dict):
            continue
        for field in ("participants", "attendees", "attendee_emails"):
            field_values = item.get(field)
            if isinstance(field_values, list):
                values.extend(field_values)
    return values


def event_timestamp(event: dict[str, Any], *, fallback: Any = "") -> datetime | None:
    for value in [
        (event.get("start") or {}).get("dateTime") if isinstance(event.get("start"), dict) else "",
        (event.get("start") or {}).get("date") if isinstance(event.get("start"), dict) else "",
        event.get("start"),
        event.get("created"),
        event.get("updated"),
        fallback,
    ]:
        parsed = parse_timestamp(value)
        if parsed:
            return parsed
    return None


def context_contact_affinity_inputs(*, root: Optional[Path], state_root: Path) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    selection_dir = conversation_context_contact_selections_dir(state_root)
    for path in sorted(selection_dir.glob("*.json"))[-200:]:
        payload = read_json_file(path)
        for decision in payload.get("decisions") if isinstance(payload.get("decisions"), list) else []:
            if isinstance(decision, dict):
                decisions.append(decision)
    calendar_events: list[dict[str, Any]] = []
    try:
        with connect(root) as con:
            init_db(con)
            rows = con.execute(
                """
                SELECT id, kind, title, json_payload, metadata_json, generated_at, updated_at
                FROM documents
                WHERE kind = 'transcript'
                ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC
                LIMIT 1000
                """
            ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        payload = parse_object_json(row["json_payload"])
        metadata = parse_object_json(row["metadata_json"])
        event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        if not event and isinstance(metadata.get("event"), dict):
            event = metadata["event"]
        if not event:
            continue
        calendar_events.append(
            {
                "document_id": row["id"],
                "title": row["title"],
                "timestamp": event_timestamp(event, fallback=row["generated_at"] or row["updated_at"]),
                "people": event_people_values(event),
            }
        )
    return {"decisions": decisions, "calendar_events": calendar_events}


def increment_affinity_window(counts: dict[str, int], timestamp: datetime | None, *, now: datetime) -> None:
    if not timestamp:
        return
    age_days = max(0, (now - timestamp).days)
    if age_days <= 30:
        counts["interaction_count_30d"] += 1
    if age_days <= 90:
        counts["interaction_count_90d"] += 1
    if age_days <= 365:
        counts["interaction_count_365d"] += 1


def relationship_affinity_for_candidate(
    candidate: dict[str, Any],
    *,
    inputs: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    terms = contact_affinity_person_values(candidate)
    counts = {
        "interaction_count_30d": 0,
        "interaction_count_90d": 0,
        "interaction_count_365d": 0,
        "calendar_overlap_count_365d": 0,
        "message_count_30d": 0,
        "message_count_365d": 0,
        "transcript_overlap_count_365d": 0,
        "prior_selected_count": 0,
        "prior_excluded_count": 0,
    }
    last_contacted: datetime | None = None
    last_calendar: datetime | None = None
    evidence: list[str] = []
    calendar_events = inputs.get("calendar_events") if isinstance(inputs.get("calendar_events"), list) else []
    for event in calendar_events:
        if not isinstance(event, dict):
            continue
        people = event.get("people") if isinstance(event.get("people"), list) else []
        if not any(contact_affinity_matches_value(terms, value) for value in people):
            continue
        timestamp = event.get("timestamp") if isinstance(event.get("timestamp"), datetime) else None
        if timestamp and (now - timestamp).days <= 365:
            counts["calendar_overlap_count_365d"] += 1
            counts["transcript_overlap_count_365d"] += 1
            increment_affinity_window(counts, timestamp, now=now)
            if not last_calendar or timestamp > last_calendar:
                last_calendar = timestamp
            if not last_contacted or timestamp > last_contacted:
                last_contacted = timestamp

    decisions = inputs.get("decisions") if isinstance(inputs.get("decisions"), list) else []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        candidate_ids = {str(decision.get("candidate_id") or "").strip()}
        decision_candidate = decision.get("candidate") if isinstance(decision.get("candidate"), dict) else {}
        candidate_ids.update(context_contact_candidate_ids(decision_candidate))
        matches_id = bool(candidate_ids & terms["ids"])
        matches_value = contact_affinity_matches_value(terms, decision_candidate) if decision_candidate else False
        if not matches_id and not matches_value:
            continue
        created_at = parse_timestamp(decision.get("created_at"))
        increment_affinity_window(counts, created_at, now=now)
        counts["transcript_overlap_count_365d"] += 1 if created_at and (now - created_at).days <= 365 else 0
        if not last_contacted or (created_at and created_at > last_contacted):
            last_contacted = created_at
        action = str(decision.get("action") or "")
        if action == "select":
            counts["prior_selected_count"] += 1
        elif action == "exclude":
            counts["prior_excluded_count"] += 1

    existing = candidate.get("relationship_affinity") if isinstance(candidate.get("relationship_affinity"), dict) else {}
    for key in ("message_count_30d", "message_count_365d"):
        try:
            counts[key] += int(existing.get(key) or 0)
        except (TypeError, ValueError):
            pass
    existing_last = parse_timestamp(existing.get("last_contacted_at"))
    if existing_last and (not last_contacted or existing_last > last_contacted):
        last_contacted = existing_last

    if last_contacted:
        age_days = max(0, (now - last_contacted).days)
        if age_days == 0:
            evidence.append("contacted today")
        elif age_days == 1:
            evidence.append("contacted 1 day ago")
        else:
            evidence.append(f"contacted {age_days} days ago")
    else:
        evidence.append("no recent communication")
    if counts["calendar_overlap_count_365d"]:
        evidence.append(f"{counts['calendar_overlap_count_365d']} calendar overlaps")
    if counts["interaction_count_90d"]:
        evidence.append(f"{counts['interaction_count_90d']} interactions in 90d")
    if counts["prior_selected_count"]:
        evidence.append("selected before")
    if counts["prior_excluded_count"]:
        evidence.append("excluded before")

    return {
        "last_contacted_at": last_contacted.isoformat().replace("+00:00", "Z") if last_contacted else "",
        "last_calendar_overlap_at": last_calendar.isoformat().replace("+00:00", "Z") if last_calendar else "",
        **counts,
        "evidence": evidence[:4],
    }


def contact_text_score(candidate: dict[str, Any], query: str) -> float:
    terms = [term for term in re.split(r"\s+", query.strip().lower()) if term]
    if not terms:
        return 0.5
    label = str(candidate.get("label") or "").lower()
    email = str(candidate.get("email") or "").lower()
    text = context_contact_candidate_search_text(candidate)
    if any(term == email for term in terms if "@" in term):
        return 1.0
    if email and any(email.startswith(term) for term in terms):
        return 0.95
    if label and label.startswith(" ".join(terms)):
        return 0.9
    if all(term in text for term in terms):
        return 0.75
    if any(term in text for term in terms):
        return 0.35
    return 0.0


def contact_conversation_score(candidate: dict[str, Any]) -> float:
    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []
    source_type = str(candidate.get("source_type") or candidate.get("source") or "")
    if source_type in {"operator_participant_hint", "operator_input"}:
        return 0.85
    if any(isinstance(item, dict) and "calendar" in str(item.get("source") or item.get("kind") or "") for item in evidence):
        return 0.8
    if source_type in {"gws_contact", "odollo_contact", "local_contact"}:
        return 0.45
    return 0.25


def contact_source_quality_score(candidate: dict[str, Any]) -> float:
    source_type = str(candidate.get("source_type") or candidate.get("source") or "")
    if source_type in {"operator_input", "operator_participant_hint", "local_contact"}:
        return 1.0
    if source_type == "gws_contact":
        return 0.85
    if source_type == "gws_other_contact":
        return 0.65
    if source_type == "gws_directory_person":
        return 0.55
    if source_type == "odollo_contact":
        return 0.75
    return 0.45


def contact_affinity_score(affinity: dict[str, Any], *, now: datetime) -> float:
    last_contacted = parse_timestamp(affinity.get("last_contacted_at"))
    recency_score = 0.0
    if last_contacted:
        recency_score = max(0.0, 1.0 - (max(0, (now - last_contacted).days) / 365.0))
    count_365 = int(affinity.get("interaction_count_365d") or 0)
    frequency_score = min(1.0, math.log1p(max(0, count_365)) / math.log1p(20))
    return round((0.65 * recency_score) + (0.35 * frequency_score), 4)


def operator_history_score(affinity: dict[str, Any]) -> float:
    selected = int(affinity.get("prior_selected_count") or 0)
    excluded = int(affinity.get("prior_excluded_count") or 0)
    return max(-1.0, min(1.0, (0.4 * selected) - (0.6 * excluded)))


def rank_contact_candidate(candidate: dict[str, Any], *, query: str, now: datetime) -> dict[str, Any]:
    affinity = candidate.get("relationship_affinity") if isinstance(candidate.get("relationship_affinity"), dict) else {}
    text_score = contact_text_score(candidate, query)
    conversation_score = contact_conversation_score(candidate)
    affinity_score = contact_affinity_score(affinity, now=now)
    source_quality = contact_source_quality_score(candidate)
    history_score = operator_history_score(affinity)
    rank_score = round(
        (45.0 * text_score)
        + (15.0 * conversation_score)
        + (25.0 * affinity_score)
        + (10.0 * source_quality)
        + (5.0 * history_score),
        3,
    )
    reasons = []
    if text_score >= 0.9:
        reasons.append("strong text match")
    elif text_score > 0:
        reasons.append("text match")
    reasons.extend(str(value) for value in affinity.get("evidence", []) if str(value or ""))
    if source_quality >= 0.85:
        reasons.append("trusted contact source")
    result = dict(candidate)
    result["rank_score"] = rank_score
    result["score_components"] = {
        "text_score": round(text_score, 4),
        "conversation_score": round(conversation_score, 4),
        "affinity_score": round(affinity_score, 4),
        "source_quality_score": round(source_quality, 4),
        "operator_history_score": round(history_score, 4),
    }
    result["ranking_reasons"] = unique_strings(reasons)[:4]
    return result


def compute_contact_affinity_candidates(
    candidates: list[dict[str, Any]],
    *,
    query: str,
    root: Optional[Path],
    state_root: Path,
) -> list[dict[str, Any]]:
    inputs = context_contact_affinity_inputs(root=root, state_root=state_root)
    now = datetime.now(timezone.utc)
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        compact = compact_context_contact_candidate(candidate)
        compact["relationship_affinity"] = relationship_affinity_for_candidate(compact, inputs=inputs, now=now)
        result.append(rank_contact_candidate(compact, query=query, now=now))
    return sorted(
        result,
        key=lambda item: (
            float(item.get("rank_score") or 0.0),
            float(item.get("confidence") or 0.0),
            str(item.get("label") or item.get("email") or "").lower(),
        ),
        reverse=True,
    )


def context_contact_affinity_cache_state(*, state_root: Path, conversation_key: str) -> dict[str, Any]:
    if not conversation_key:
        return {"status": "missing_conversation_key", "items": [], "items_by_id": {}, "path": ""}
    path = context_contact_affinity_cache_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    items_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for contact_id in context_contact_candidate_ids(item):
            items_by_id.setdefault(contact_id, item)
    return {
        "schema_version": "transcribe-audio.context-contact-affinity-cache.v1",
        "status": "hit" if items else "empty",
        "conversation_key": conversation_key,
        "updated_at": str(payload.get("updated_at") or ""),
        "query": str(payload.get("query") or ""),
        "path": str(path) if path.exists() else "",
        "items": items,
        "items_by_id": items_by_id,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def apply_cached_contact_affinity(
    candidates: list[dict[str, Any]],
    affinity_cache: dict[str, Any],
    *,
    query: str = "",
) -> list[dict[str, Any]]:
    items_by_id = affinity_cache.get("items_by_id") if isinstance(affinity_cache.get("items_by_id"), dict) else {}
    now = datetime.now(timezone.utc)
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        cached = next((items_by_id.get(contact_id) for contact_id in context_contact_candidate_ids(candidate) if items_by_id.get(contact_id)), None)
        item = dict(candidate)
        if isinstance(cached, dict):
            item["relationship_affinity"] = cached.get("relationship_affinity") if isinstance(cached.get("relationship_affinity"), dict) else {}
            item["rank_score"] = cached.get("rank_score")
            item["ranking_reasons"] = cached.get("ranking_reasons") if isinstance(cached.get("ranking_reasons"), list) else []
            item["score_components"] = cached.get("score_components") if isinstance(cached.get("score_components"), dict) else {}
        else:
            item["relationship_affinity"] = item.get("relationship_affinity") if isinstance(item.get("relationship_affinity"), dict) else {}
        result.append(rank_contact_candidate(item, query=query or str(affinity_cache.get("query") or ""), now=now))
    return result


def write_context_contact_affinity_cache(
    *,
    state_root: Path,
    conversation_key: str,
    query: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    path = context_contact_affinity_cache_path(state_root=state_root, conversation_key=conversation_key)
    now = utcish_now()
    compact_items = [compact_context_contact_candidate(item) for item in items]
    payload = {
        "schema_version": "transcribe-audio.context-contact-affinity-cache.v1",
        "conversation_key": conversation_key,
        "query": query,
        "updated_at": now,
        "items": compact_items,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, payload)
    return {
        **payload,
        "status": "updated",
        "path": str(path),
        "item_count": len(compact_items),
    }


def refresh_context_contact_affinity(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    query: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    if not conversation_key:
        raise ValueError("Context contact affinity requires a conversation key.")
    lookup = context_contact_lookup(detail)
    query_text = query.strip()
    terms = [term for term in re.split(r"\s+", query_text.lower()) if term]
    candidates = []
    for candidate in lookup.values():
        if terms and not all(term in context_contact_candidate_search_text(candidate) for term in terms):
            continue
        candidates.append(candidate)
    candidates = unique_context_contact_candidates(candidates)[: max(1, limit)]
    ranked = compute_contact_affinity_candidates(
        candidates,
        query=query_text,
        root=root,
        state_root=state_root,
    )
    cache = write_context_contact_affinity_cache(
        state_root=state_root,
        conversation_key=conversation_key,
        query=query_text,
        items=ranked[: max(1, limit)],
    )
    return {
        "schema_version": "transcribe-audio.context-contact-affinity.v1",
        "status": "updated",
        "query": query_text,
        "item_count": len(ranked[: max(1, limit)]),
        "items": ranked[: max(1, limit)],
        "cache_path": cache["path"],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def context_instructions_state(*, detail: dict[str, Any], state_root: Path) -> dict[str, Any]:
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    path = context_instructions_path(state_root=state_root, conversation_key=conversation_key) if conversation_key else None
    payload = read_json_file(path) if path and path.exists() else {}
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    instruction_text = str(payload.get("instruction_text") or "").strip()
    return {
        "schema_version": "transcribe-audio.context-operator-context.v1",
        "status": "provided" if instruction_text else "empty",
        "conversation_key": conversation_key,
        "source_document_id": source_document.get("id") or "",
        "instruction_text": instruction_text,
        "updated_at": str(payload.get("updated_at") or ""),
        "reviewer": str(payload.get("reviewer") or ""),
        "instruction_path": str(path) if path and path.exists() else "",
        "history": history[-10:],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def context_contact_selection_state(
    *,
    detail: dict[str, Any],
    state_root: Path,
    proposed_contacts: list[Any],
) -> dict[str, Any]:
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    merge_state = context_contact_merge_state(state_root=state_root, conversation_key=conversation_key)
    candidates = [
        compact_context_contact_candidate(candidate)
        for candidate in proposed_contacts
        if isinstance(candidate, dict)
    ]
    local_contacts = (detail.get("identity_review") or {}).get("contacts")
    local_candidates = [
        compact_context_contact_candidate(participant_identity.local_contact_candidate(contact))
        for contact in local_contacts
        if isinstance(contact, dict)
    ] if isinstance(local_contacts, list) else []
    search_cache = context_contact_search_cache_state(state_root=state_root, conversation_key=conversation_key)
    search_cache_candidates = [
        compact_context_contact_candidate(candidate)
        for candidate in search_cache.get("items", [])
        if isinstance(candidate, dict)
    ]
    affinity_cache = context_contact_affinity_cache_state(state_root=state_root, conversation_key=conversation_key)
    searchable_candidates = unique_context_contact_candidates(
        [*candidates, *local_candidates, *search_cache_candidates],
        merge_state=merge_state,
    )
    searchable_candidates = apply_cached_contact_affinity(searchable_candidates, affinity_cache)
    candidate_by_id = {candidate["contact_id"]: candidate for candidate in searchable_candidates if candidate.get("contact_id")}
    for candidate in searchable_candidates:
        merged_contact_ids = candidate.get("merged_contact_ids") if isinstance(candidate.get("merged_contact_ids"), list) else []
        for merged_id in merged_contact_ids:
            if str(merged_id or ""):
                candidate_by_id.setdefault(str(merged_id), candidate)
    path = context_contact_selection_path(state_root=state_root, conversation_key=conversation_key) if conversation_key else None
    payload = read_json_file(path) if path and path.exists() else {}
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    latest: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        candidate_id = str(decision.get("candidate_id") or "")
        action = str(decision.get("action") or "")
        decision_candidate = decision.get("candidate") if isinstance(decision.get("candidate"), dict) else {}
        if candidate_id and decision_candidate:
            candidate_by_id.setdefault(candidate_id, compact_context_contact_candidate(decision_candidate))
        if candidate_id and action in {"select", "exclude", "clear"}:
            latest[candidate_id] = decision
    selected_ids = sorted(candidate_id for candidate_id, decision in latest.items() if decision.get("action") == "select")
    excluded_ids = sorted(candidate_id for candidate_id, decision in latest.items() if decision.get("action") == "exclude")
    selected = [candidate_by_id[candidate_id] for candidate_id in selected_ids if candidate_id in candidate_by_id]
    excluded = [candidate_by_id[candidate_id] for candidate_id in excluded_ids if candidate_id in candidate_by_id]
    return {
        "schema_version": "transcribe-audio.context-contact-selection.v1",
        "status": "selected" if selected else "review_needed" if candidates else "no_candidates",
        "conversation_key": conversation_key,
        "source_document_id": source_document.get("id") or "",
        "selection_path": str(path) if path and path.exists() else "",
        "candidate_count": len(candidates),
        "searchable_candidate_count": len(searchable_candidates),
        "search_cache_status": search_cache.get("status", "empty"),
        "search_cache_candidate_count": len(search_cache_candidates),
        "search_cache_path": search_cache.get("path", ""),
        "affinity_cache_status": affinity_cache.get("status", "empty"),
        "affinity_cache_path": affinity_cache.get("path", ""),
        "searchable_candidates": searchable_candidates[:50],
        "merge_state": merge_state,
        "dedupe_clusters": context_contact_dedupe_clusters(searchable_candidates),
        "selected_candidate_ids": selected_ids,
        "excluded_candidate_ids": excluded_ids,
        "selected_candidates": selected,
        "excluded_candidates": excluded,
        "decisions": decisions[-25:],
        "allowed_actor_types": ["operator", "app_intelligence"],
        "app_intelligence_decision_schema": {
            "selection_actions": ["select", "exclude", "clear"],
            "merge_actions": ["merge", "split"],
            "instruction_action": "save_context_instructions",
            "external_writes_allowed": False,
        },
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def deposition_preview_summary(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    candidates = payload.get("memory_candidates") if isinstance(payload.get("memory_candidates"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    return {
        "schema_version": "transcribe-audio.deposition-memory-preview-summary.v1",
        "status": "preview_ready" if payload else "needs_contextual_readout",
        "preview_path": str(path) if path else "",
        "review_required": bool(payload.get("review_required", True)) if payload else True,
        "selected_candidate": payload.get("selected_candidate") if isinstance(payload.get("selected_candidate"), dict) else {},
        "warnings": [str(item) for item in warnings],
        "actions": [
            {
                "action_type": action.get("action_type"),
                "target_kind": action.get("target_kind"),
                "target_id": action.get("target_id"),
                "status": action.get("status"),
                "writes_enabled": bool((action.get("metadata") or {}).get("writes_enabled")),
            }
            for action in actions
            if isinstance(action, dict)
        ],
        "memory_candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "kind": candidate.get("kind"),
                "status": candidate.get("status"),
                "target_group_id": candidate.get("target_group_id"),
                "evidence": truncate_text(candidate.get("evidence"), 220),
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ],
        "action_count": len(actions),
        "memory_candidate_count": len(candidates),
        "will_execute_external_action": False,
        "will_perform_external_write": False,
        "future_required_approval_token_for_queue": DEPOSITION_MEMORY_PREVIEW_TOKEN,
    }


def existing_deposition_preview_for(contextual_document: dict[str, Any] | None) -> tuple[Path | None, dict[str, Any]]:
    readout_path = artifact_path_for_document(contextual_document)
    if not readout_path:
        return None, {}
    name = readout_path.name
    if name.endswith(".contextual.readout.json"):
        candidate = readout_path.with_name(name[: -len(".contextual.readout.json")] + ".deposit-preview.json")
    elif name.endswith(".readout.json"):
        candidate = readout_path.with_name(name[: -len(".readout.json")] + ".deposit-preview.json")
    else:
        candidate = readout_path.with_suffix(".deposit-preview.json")
    if candidate.exists() and candidate.is_file():
        return candidate, read_json_file(candidate)
    return None, {}


def final_preview_state(detail: dict[str, Any]) -> dict[str, Any]:
    path, payload = existing_deposition_preview_for(detail.get("contextual_readout_document"))
    summary = deposition_preview_summary(path, payload)
    identity = (detail.get("identity_review") or {}).get("identity_bundle", {})
    context = detail.get("context_workbench") if isinstance(detail.get("context_workbench"), dict) else {}
    gate_warnings: list[str] = []
    pending_count = int((detail.get("identity_review") or {}).get("pending_count") or 0)
    if pending_count:
        gate_warnings.append(f"{pending_count} speaker identity decision(s) still need review.")
    if isinstance(identity, dict):
        for warning in identity.get("warnings") if isinstance(identity.get("warnings"), list) else []:
            text = str(warning or "").strip()
            if text and text not in gate_warnings:
                gate_warnings.append(text)
    for warning in context.get("warnings") if isinstance(context.get("warnings"), list) else []:
        text = str(warning or "").strip()
        if text and text not in gate_warnings:
            gate_warnings.append(text)
    summary["identity_context_warnings"] = gate_warnings
    summary["identity_status"] = identity.get("review_status", "unknown") if isinstance(identity, dict) else "unknown"
    summary["ready_for_deposition_review"] = bool(payload) and not gate_warnings
    if gate_warnings:
        summary["status"] = "blocked_identity_or_context_review"
        summary["review_required"] = True
        summary["warnings"] = list(dict.fromkeys([*summary.get("warnings", []), *gate_warnings]))
    return summary


def first_pass_summary_state(detail: dict[str, Any]) -> dict[str, Any]:
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    summary_document = detail.get("summary_document") or {}
    status = "summary_ready" if summary_document else "needs_summary"
    return {
        "schema_version": "transcribe-audio.first-pass-summary-workspace.v1",
        "status": status,
        "source_document_id": source_document.get("id") or "",
        "summary_document_id": summary_document.get("id") or "",
        "summary_ready": bool(summary_document),
        "will_execute_external_action": False,
        "will_perform_external_write": False,
        "future_required_approval_token_for_submit": FIRST_PASS_SUMMARY_SUBMIT_TOKEN,
    }


def get_conversation_detail(
    document_id: str,
    *,
    root: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        selected_row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if selected_row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        rows = con.execute(
            """
            SELECT * FROM documents
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            """
        ).fetchall()
    selected_key = conversation_source_key(selected_row)
    group_rows = [row for row in rows if conversation_source_key(row) == selected_key]
    if not group_rows:
        group_rows = [selected_row]
    transcripts = [row for row in group_rows if row["kind"] == "transcript"]
    readouts = [row for row in group_rows if row["kind"] == "readout"]
    contextual_readouts = [row for row in group_rows if row["kind"] == "contextual_readout"]
    transcript_row = latest_document(transcripts) if transcripts else None
    readout_row = latest_document(readouts) if readouts else None
    contextual_row = latest_document(contextual_readouts) if contextual_readouts else None
    selected_detail = get_document(document_id, root=root)
    transcript_detail = get_document(transcript_row["id"], root=root) if transcript_row else None
    readout_detail = get_document(readout_row["id"], root=root) if readout_row else None
    contextual_detail = get_document(contextual_row["id"], root=root) if contextual_row else None
    participants: list[Any] = []
    for detail in [readout_detail, contextual_detail, transcript_detail, selected_detail]:
        payload_participants = (detail.get("json_payload") if detail else {}).get("participants") if detail else None
        if isinstance(payload_participants, list):
            for participant in payload_participants:
                if participant not in participants:
                    participants.append(participant)
    summary = conversation_summary(selected_key, group_rows)
    identity_review = conversation_identity_review(
        conversation_key=selected_key,
        source_document=transcript_detail,
        participants=participants,
        root=root,
        state_root=state_root,
    )
    detail_payload = {
        "schema_version": "transcribe-audio.conversation-detail.v1",
        "conversation": summary,
        "selected_document": selected_detail,
        "transcript_document": transcript_detail,
        "summary_document": readout_detail,
        "contextual_readout_document": contextual_detail,
        "participants": participants,
        "media_blob": summary["media_blob"],
        "first_pass_summary": {},
        "identity_review": identity_review,
        "context_workbench": {},
        "final_preview": {},
        "review_state": {
            "speaker_pending_count": identity_review["pending_count"],
            "context_status": "contextual_readout_ready" if contextual_detail else "needs_context",
            "deposition_preview_status": "unknown",
        },
        "will_read_artifact_files": False,
        "will_return_artifact_content": True,
    }
    detail_payload["first_pass_summary"] = first_pass_summary_state(detail_payload)
    detail_payload["context_workbench"] = context_workbench_state(
        detail=detail_payload,
        root=root,
        state_root=state_root,
    )
    detail_payload["final_preview"] = final_preview_state(detail_payload)
    detail_payload["review_state"]["deposition_preview_status"] = detail_payload["final_preview"]["status"]
    return {
        **detail_payload,
    }


def warm_participant_identity_cache(
    document_id: str,
    *,
    root: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    identity_review = detail.get("identity_review") if isinstance(detail.get("identity_review"), dict) else {}
    cache = identity_review.get("identity_cache") if isinstance(identity_review.get("identity_cache"), dict) else {}
    return {
        "schema_version": "transcribe-audio.participant-identity-cache-warm.v1",
        "status": cache.get("status") or "unknown",
        "conversation_key": (detail.get("conversation") or {}).get("key") or "",
        "source_document_id": (detail.get("transcript_document") or detail.get("selected_document") or {}).get("id") or "",
        "identity_cache": cache,
        "candidate_count": len(((identity_review.get("identity_bundle") or {}).get("contact_candidates") or [])),
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def get_document(document_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        blobs = con.execute(
            """
            SELECT blobs.id, document_blobs.role, blobs.mime_type, blobs.bytes, blobs.sha256
            FROM document_blobs
            JOIN blobs ON document_blobs.blob_id = blobs.id
            WHERE document_blobs.document_id = ?
            ORDER BY document_blobs.role, blobs.id
            """,
            (document_id,),
        ).fetchall()
    payload = parse_object_json(row["json_payload"])
    summary = document_summary(row)
    summary.update(
        {
            "json_payload": payload,
            "text_content": row["text_content"],
            "blobs": [
                {
                    "id": blob["id"],
                    "role": blob["role"],
                    "mime_type": blob["mime_type"],
                    "bytes": blob["bytes"],
                    "sha256": blob["sha256"],
                    "playback_url": f"/api/blobs/{blob['id']}",
                    "download_url": f"/api/blobs/{blob['id']}?download=1",
                }
                for blob in blobs
            ],
        }
    )
    return summary


def get_related_documents(document_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        payload = parse_object_json(row["json_payload"])
        metadata = parse_object_json(row["metadata_json"])
        source_artifact_path = str(metadata.get("source_artifact_path") or payload.get("source_artifact_path") or "")
        source_row = None
        if source_artifact_path:
            source_row = con.execute("SELECT * FROM documents WHERE source_path = ?", (source_artifact_path,)).fetchone()
        derived_rows = con.execute(
            """
            SELECT * FROM documents
            WHERE (
                json_extract(metadata_json, '$.source_artifact_path') = ?
                OR json_extract(json_payload, '$.source_artifact_path') = ?
            )
            AND id != ?
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            """,
            (row["source_path"], row["source_path"], document_id),
        ).fetchall()
    return {
        "document": document_summary(row),
        "source_artifact_path": source_artifact_path,
        "source_document": document_summary(source_row) if source_row else None,
        "derived_documents": [document_summary(derived_row) for derived_row in derived_rows],
    }


def review_queue_dir(state_root: Path) -> Path:
    return state_root.expanduser() / "review-queue"


def conversation_context_runs_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_WORKBENCH_DIRNAME


def conversation_context_contact_selections_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_CONTACT_SELECTION_DIRNAME


def conversation_context_contact_search_cache_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_CONTACT_SEARCH_CACHE_DIRNAME


def context_contact_search_cache_path(*, state_root: Path, conversation_key: str) -> Path:
    return conversation_context_contact_search_cache_dir(state_root) / f"{stable_id('context-contact-search-cache', conversation_key)}.json"


def conversation_context_contact_refresh_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_CONTACT_REFRESH_DIRNAME


def conversation_context_contact_affinity_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_CONTACT_AFFINITY_DIRNAME


def context_contact_affinity_cache_path(*, state_root: Path, conversation_key: str) -> Path:
    return conversation_context_contact_affinity_dir(state_root) / f"{stable_id('context-contact-affinity-cache', conversation_key)}.json"


def conversation_context_contact_merge_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_CONTACT_MERGE_DIRNAME


def context_contact_merge_path(*, state_root: Path, conversation_key: str) -> Path:
    return conversation_context_contact_merge_dir(state_root) / f"{stable_id('context-contact-merge', conversation_key)}.json"


def conversation_context_instructions_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONTEXT_INSTRUCTIONS_DIRNAME


def conversation_preview_dir(state_root: Path) -> Path:
    return state_root.expanduser() / CONVERSATION_PREVIEW_DIRNAME


def record_speaker_identity_review(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    speaker_label: str,
    action: str,
    contact_label: str = "",
    contact_id: str = "",
    email: str = "",
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    if action not in {"confirm", "defer"}:
        raise ValueError("Speaker identity action must be confirm or defer.")
    if not speaker_label.strip():
        raise ValueError("Missing required speaker_label.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str(detail["conversation"]["key"])
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    now = utcish_now()
    assignment_id = stable_id("speaker-assignment", conversation_key, speaker_label)
    resolved_contact_id = ""
    resolved_contact_label = ""
    status = "deferred"
    evidence: list[dict[str, Any]] = []
    if action == "confirm":
        resolved_contact_label = contact_label.strip() or speaker_label.strip()
        resolved_contact_id = contact_id.strip() or stable_id("contact", resolved_contact_label.lower(), email.lower())
        status = "confirmed"
        evidence.append({"source": "operator_review", "note": note})
    else:
        evidence.append({"source": "operator_defer", "note": note})
    with connect(root) as con:
        init_db(con)
        if action == "confirm":
            con.execute(
                """
                INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, '', ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  label = excluded.label,
                  email = excluded.email,
                  metadata_json = excluded.metadata_json,
                  updated_at = excluded.updated_at
                """,
                (
                    resolved_contact_id,
                    resolved_contact_label,
                    email.strip(),
                    json.dumps({"source": "operator_created", "reviewer": reviewer}, sort_keys=True),
                    now,
                    now,
                ),
            )
        con.execute(
            """
            INSERT INTO speaker_assignments (
                id, conversation_key, document_id, speaker_label, contact_id, contact_label,
                status, confidence, evidence_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_key, speaker_label) DO UPDATE SET
              document_id = excluded.document_id,
              contact_id = excluded.contact_id,
              contact_label = excluded.contact_label,
              status = excluded.status,
              confidence = excluded.confidence,
              evidence_json = excluded.evidence_json,
              updated_at = excluded.updated_at
            """,
            (
                assignment_id,
                conversation_key,
                source_document.get("id") or document_id,
                speaker_label,
                resolved_contact_id,
                resolved_contact_label,
                status,
                1.0 if action == "confirm" else None,
                json.dumps(evidence, sort_keys=True),
                now,
                now,
            ),
        )
        audit_id = stable_id("speaker-audit", assignment_id, now, str(uuid.uuid4()))
        con.execute(
            """
            INSERT INTO speaker_assignment_audits (
                id, assignment_id, conversation_key, document_id, speaker_label,
                action, reviewer, note, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                assignment_id,
                conversation_key,
                source_document.get("id") or document_id,
                speaker_label,
                action,
                reviewer,
                note,
                json.dumps(
                    {
                        "contact_id": resolved_contact_id,
                        "contact_label": resolved_contact_label,
                        "email": email,
                    },
                    sort_keys=True,
                ),
                now,
            ),
        )
        con.commit()
    if action == "defer":
        queue_path = review_queue_dir(state_root) / f"{assignment_id}.speaker-review.json"
        write_json_file(
            queue_path,
            {
                "schema_version": "transcribe-audio.speaker-review-item.v1",
                "id": assignment_id,
                "type": "speaker_identity_review",
                "status": "pending",
                "created_at": now,
                "document_id": source_document.get("id") or document_id,
                "representative_document_id": document_id,
                "conversation_key": conversation_key,
                "workflow_stage": "speakers",
                "speaker_label": speaker_label,
                "reason": note or "Speaker/contact identity was deferred for human review.",
                "will_execute_external_action": False,
            },
        )
    refreshed = get_conversation_detail(document_id, root=root, state_root=state_root)
    return {
        "schema_version": "transcribe-audio.speaker-review-action.v1",
        "status": status,
        "assignment_id": assignment_id,
        "audit_recorded": True,
        "identity_review": refreshed.get("identity_review"),
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def context_workbench_preview(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    queue: bool = False,
    approval_token: str = "",
) -> dict[str, Any]:
    if queue and approval_token != CONTEXT_WORKBENCH_TOKEN:
        raise ValueError(f"Queueing a context workbench run requires approval_token={CONTEXT_WORKBENCH_TOKEN}.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    summary_document = detail.get("summary_document") or {}
    contextual_document = detail.get("contextual_readout_document") or {}
    now = utcish_now()
    run_id = stable_id("context-workbench", document_id, now, str(uuid.uuid4()))
    steps = [
        {
            "name": "first_pass_summary",
            "status": "ready" if summary_document else "missing",
            "document_id": summary_document.get("id") or "",
        },
        {
            "name": "route_context",
            "status": "preview_ready" if summary_document else "blocked",
            "will_execute_provider": False,
            "will_write_external_state": False,
        },
        {
            "name": "contextual_reread",
            "status": "materialized" if contextual_document else "queued" if queue else "preview_ready",
            "document_id": contextual_document.get("id") or "",
            "will_execute_provider": False,
            "will_write_external_state": False,
        },
    ]
    manifest = {
        "schema_version": "transcribe-audio.context-workbench-run.v1",
        "run_id": run_id,
        "status": "queued" if queue else "previewed",
        "created_at": now,
        "document_id": document_id,
        "source_document_id": source_document.get("id") or "",
        "summary_document_id": summary_document.get("id") or "",
        "contextual_readout_document_id": contextual_document.get("id") or "",
        "conversation_key": detail["conversation"]["key"],
        "workflow_stage": "context",
        "steps": steps,
        "context_workbench": detail.get("context_workbench") or {},
        "participant_identity_bundle": (detail.get("identity_review") or {}).get("identity_bundle", {}),
        "contact_selection": (detail.get("context_workbench") or {}).get("contact_selection", {}),
        "operator_context": (detail.get("context_workbench") or {}).get("operator_context", {}),
        "will_execute_external_action": False,
        "will_run_provider": False,
        "will_perform_external_write": False,
        "future_required_approval_token_for_materialize": "MATERIALIZE_CONTEXTUAL_READOUT",
    }
    path = conversation_context_runs_dir(state_root) / f"{now.replace(':', '-')}-{run_id}.json"
    write_json_file(path, manifest)
    return {
        "schema_version": "transcribe-audio.context-workbench-action.v1",
        "status": manifest["status"],
        "run_id": run_id,
        "manifest": str(path),
        "steps": steps,
        "context_workbench": manifest["context_workbench"],
        "contact_selection": manifest["contact_selection"],
        "operator_context": manifest["operator_context"],
        "will_execute_external_action": False,
        "will_run_provider": False,
        "will_perform_external_write": False,
    }


def manual_context_contact_candidate(
    manual_candidate: dict[str, Any],
    *,
    conversation_key: str,
    reviewer: str,
) -> dict[str, Any]:
    raw_email = str(manual_candidate.get("email") or "").strip()
    email = participant_identity.normalize_email(raw_email) or raw_email.lower()
    label = str(manual_candidate.get("label") or manual_candidate.get("name") or "").strip()
    if not label and email:
        label = email
    if not label and not email:
        raise ValueError("Manual context contact requires a label or email.")
    contact_id = str(manual_candidate.get("contact_id") or "").strip() or stable_id("contact", label.lower(), email.lower())
    return compact_context_contact_candidate(
        {
            "contact_id": contact_id,
            "label": label,
            "email": email,
            "source": "operator_input",
            "source_type": "operator_input",
            "source_profile": "context_workbench",
            "confidence": 1.0,
            "evidence": [
                {
                    "kind": "operator_input",
                    "source": "context_workbench",
                    "conversation_key": conversation_key,
                    "reviewer": reviewer,
                }
            ],
        }
    )


def upsert_context_contact(candidate: dict[str, Any], *, root: Optional[Path], reviewer: str, now: str) -> None:
    label = str(candidate.get("label") or candidate.get("email") or "").strip()
    contact_id = str(candidate.get("contact_id") or "").strip()
    if not label or not contact_id:
        return
    with connect(root) as con:
        init_db(con)
        con.execute(
            """
            INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, '', ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              label = excluded.label,
              email = excluded.email,
              metadata_json = excluded.metadata_json,
              updated_at = excluded.updated_at
            """,
            (
                contact_id,
                label,
                str(candidate.get("email") or "").strip(),
                json.dumps({"source": "context_workbench_operator_input", "reviewer": reviewer}, sort_keys=True),
                now,
                now,
            ),
        )
        con.commit()


def context_contact_lookup(detail: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context_state = detail.get("context_workbench") if isinstance(detail.get("context_workbench"), dict) else {}
    proposed = context_state.get("proposed_contact_candidates") if isinstance(context_state.get("proposed_contact_candidates"), list) else []
    selection = context_state.get("contact_selection") if isinstance(context_state.get("contact_selection"), dict) else {}
    merge_state = selection.get("merge_state") if isinstance(selection.get("merge_state"), dict) else {}
    selection_candidates = []
    for key in ("searchable_candidates", "selected_candidates", "excluded_candidates"):
        values = selection.get(key) if isinstance(selection.get(key), list) else []
        selection_candidates.extend(value for value in values if isinstance(value, dict))
    local_contacts = (detail.get("identity_review") or {}).get("contacts")
    local_candidates = [
        participant_identity.local_contact_candidate(contact)
        for contact in local_contacts
        if isinstance(contact, dict)
    ] if isinstance(local_contacts, list) else []
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in unique_context_contact_candidates([*proposed, *selection_candidates, *local_candidates], merge_state=merge_state):
        contact_id = str(candidate.get("contact_id") or "")
        if contact_id:
            lookup.setdefault(contact_id, candidate)
        merged_contact_ids = candidate.get("merged_contact_ids") if isinstance(candidate.get("merged_contact_ids"), list) else []
        for merged_id in merged_contact_ids:
            if str(merged_id or ""):
                lookup.setdefault(str(merged_id), candidate)
    return lookup


def record_context_contact_selection(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    candidate_id: str,
    action: str,
    actor_type: str = "operator",
    reviewer: str = "operator",
    note: str = "",
    manual_candidate: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    batch = record_context_contact_selection_batch(
        document_id,
        root=root,
        state_root=state_root,
        actions=[
            {
                "candidate_id": candidate_id,
                "action": action,
                "actor_type": actor_type,
                "reviewer": reviewer,
                "note": note,
                "manual_candidate": manual_candidate,
            }
        ],
    )
    decision = batch["decisions"][0]
    return {
        "schema_version": "transcribe-audio.context-contact-selection-action.v1",
        "status": decision.get("action") or action,
        "decision": decision,
        "selection_path": batch["selection_path"],
        "context_workbench": batch["context_workbench"],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def record_context_contact_selection_batch(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not actions:
        raise ValueError("Context contact selection batch requires at least one action.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    if not conversation_key:
        raise ValueError("Context contact selection requires a conversation key.")
    candidate_by_id = context_contact_lookup(detail)
    path = context_contact_selection_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    new_decisions: list[dict[str, Any]] = []
    updated_at = utcish_now()
    for item in actions:
        if not isinstance(item, dict):
            raise ValueError("Context contact selection batch actions must be objects.")
        action = str(item.get("action") or "").strip().lower()
        if action not in {"select", "exclude", "clear"}:
            raise ValueError("Context contact selection action must be select, exclude, or clear.")
        actor_type = str(item.get("actor_type") or "operator").strip().lower() or "operator"
        if actor_type not in {"operator", "app_intelligence"}:
            raise ValueError("Context contact selection actor_type must be operator or app_intelligence.")
        reviewer = str(item.get("reviewer") or actor_type).strip() or actor_type
        candidate_id = str(item.get("candidate_id") or "").strip()
        manual_payload = item.get("manual_candidate") if isinstance(item.get("manual_candidate"), dict) else {}
        if candidate_id and candidate_id in candidate_by_id:
            candidate = candidate_by_id[candidate_id]
        elif manual_payload:
            candidate = manual_context_contact_candidate(manual_payload, conversation_key=conversation_key, reviewer=reviewer)
            candidate_id = str(candidate.get("contact_id") or "")
        else:
            raise ValueError("Context contact selection candidate_id is not available for this conversation.")
        if not candidate_id:
            raise ValueError("Context contact selection requires candidate_id.")
        now = utcish_now()
        if manual_payload and action == "select":
            upsert_context_contact(candidate, root=root, reviewer=reviewer, now=now)
        new_decisions.append(
            {
                "decision_id": stable_id("context-contact-selection", conversation_key, candidate_id, action, now, str(uuid.uuid4())),
                "candidate_id": candidate_id,
                "action": action,
                "actor_type": actor_type,
                "reviewer": reviewer,
                "note": str(item.get("note") or "").strip(),
                "created_at": now,
                "candidate": candidate,
                "will_execute_external_action": False,
                "will_perform_external_write": False,
            }
        )
    payload = {
        "schema_version": "transcribe-audio.context-contact-selection.v1",
        "conversation_key": conversation_key,
        "document_id": document_id,
        "source_document_id": (detail.get("transcript_document") or detail.get("selected_document") or {}).get("id") or "",
        "updated_at": updated_at,
        "decisions": [*decisions, *new_decisions],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, payload)
    refreshed_context = context_workbench_state(
        detail=detail,
        root=root,
        state_root=state_root,
    )
    return {
        "schema_version": "transcribe-audio.context-contact-selection-batch.v1",
        "status": "recorded",
        "decisions": new_decisions,
        "selection_path": str(path),
        "context_workbench": refreshed_context,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def search_context_contacts(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    query: str = "",
    limit: int = 20,
    mode: str = "cache",
    source_filters: Optional[list[str]] = None,
) -> dict[str, Any]:
    mode = (mode or "cache").strip().lower()
    if mode not in {"cache", "refresh", "sources"}:
        raise ValueError("Context contact search mode must be cache or refresh.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    refresh_entry: dict[str, Any] = {}
    warnings: list[str] = []
    source_profiles: list[dict[str, Any]] = []
    lookup = context_contact_lookup(detail)
    query_text = query.strip().lower()
    terms = [term for term in re.split(r"\s+", query_text) if term]
    source_filter_set = {str(value or "").strip().lower() for value in (source_filters or []) if str(value or "").strip()}
    if mode in {"refresh", "sources"} and len(query_text) >= 2:
        source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
        transcript_payload = source_document.get("json_payload") if isinstance(source_document.get("json_payload"), dict) else {}
        transcript_payload = {
            **transcript_payload,
            "transcript_text": source_document.get("text_content") or transcript_payload.get("transcript_text") or "",
        }
        collect_kwargs: dict[str, Any] = {
            "query_terms": [query_text],
            "transcript": transcript_payload,
            "state_root": state_root,
        }
        if source_filter_set:
            collect_kwargs["source_filters"] = source_filter_set
        provenance_sources, source_profiles, warnings = participant_identity.collect_configured_contact_sources(**collect_kwargs)
        if source_filter_set:
            provenance_sources = [
                source
                for source in provenance_sources
                if str(source.source_type or "").split("_", 1)[0].lower() in source_filter_set
                or str((source.metadata or {}).get("profile") or "").lower() in source_filter_set
            ]
            source_profiles = [
                profile
                for profile in source_profiles
                if str(profile.get("source") or "").lower() in source_filter_set
                or str(profile.get("profile") or "").lower() in source_filter_set
            ]
        evidence_pool = [
            participant_identity.compact_person(
                {"name": query_text},
                source="context_contact_search.query",
            )
        ]
        source_candidates = [
            participant_identity.provenance_candidate(source, evidence_pool=evidence_pool)
            for source in provenance_sources
        ]
        source_candidates = participant_identity.ranked_contact_candidates(
            source_candidates,
            limit=max(1, limit),
            per_source_profile=max(1, limit),
            aliases=participant_identity.load_contact_aliases(state_root),
            min_confidence=0.4,
        )
        if conversation_key:
            refresh_entry = append_context_contact_search_cache(
                state_root=state_root,
                conversation_key=conversation_key,
                query=query,
                items=source_candidates,
                source_profiles=source_profiles,
                warnings=warnings,
            )
            detail = get_conversation_detail(document_id, root=root, state_root=state_root)
            lookup = context_contact_lookup(detail)
    items = []
    for candidate in lookup.values():
        haystack = context_contact_candidate_search_text(candidate)
        if terms and not all(term in haystack for term in terms):
            continue
        items.append(candidate)
    items = unique_context_contact_candidates(items)
    affinity_cache = context_contact_affinity_cache_state(state_root=state_root, conversation_key=conversation_key)
    if mode in {"refresh", "sources"} and items:
        items = compute_contact_affinity_candidates(
            items,
            query=query,
            root=root,
            state_root=state_root,
        )
        write_context_contact_affinity_cache(
            state_root=state_root,
            conversation_key=conversation_key,
            query=query,
            items=items[: max(1, limit)],
        )
        affinity_cache = context_contact_affinity_cache_state(state_root=state_root, conversation_key=conversation_key)
    else:
        items = apply_cached_contact_affinity(items, affinity_cache, query=query)
    items = sorted(
        items,
        key=lambda item: (
            float(item.get("rank_score") or 0.0),
            float(item.get("confidence") or 0.0),
            str(item.get("label") or item.get("email") or "").lower(),
        ),
        reverse=True,
    )[: max(1, limit)]
    selected_items = []
    selection = ((detail.get("context_workbench") or {}).get("contact_selection") or {})
    for candidate in selection.get("selected_candidates") if isinstance(selection.get("selected_candidates"), list) else []:
        if isinstance(candidate, dict):
            selected_items.append(candidate)
    return {
        "schema_version": "transcribe-audio.context-contact-search.v1",
        "status": "ok",
        "query": query,
        "mode": "refresh" if mode in {"refresh", "sources"} else "cache",
        "cache_status": "updated" if refresh_entry else "hit" if items else "miss",
        "source_profiles": source_profiles,
        "warnings": warnings,
        "refreshed_candidate_count": int(refresh_entry.get("item_count") or 0),
        "search_cache_path": refresh_entry.get("cache_path", ""),
        "affinity_cache_status": affinity_cache.get("status", "empty"),
        "affinity_cache_path": affinity_cache.get("path", ""),
        "selected_items": selected_items,
        "items": items,
        "total": len(items),
        "will_execute_external_action": mode in {"refresh", "sources"},
        "will_perform_external_write": False,
    }


def configured_contact_refresh_sources(*, state_root: Path) -> list[dict[str, Any]]:
    config = participant_identity.load_contact_source_config(state_root)
    sources: list[dict[str, Any]] = [
        {
            "source": "calendar",
            "profile": "conversation",
            "label": "Calendar attendees",
            "read_only": True,
            "will_execute_external_action": False,
            "description": "Refresh local calendar attendee candidates already attached to the transcript.",
        }
    ]
    gws_config = config.get("gws") if isinstance(config.get("gws"), dict) else {}
    for profile in gws_config.get("profiles") if isinstance(gws_config.get("profiles"), list) else []:
        if not isinstance(profile, dict) or profile.get("enabled") is False:
            continue
        label = str(profile.get("label") or profile.get("profile") or "gws-default")
        surfaces = profile.get("surfaces") if isinstance(profile.get("surfaces"), list) else ["contacts"]
        sources.append(
            {
                "source": "gws",
                "profile": label,
                "label": f"GWS {label}",
                "surfaces": surfaces,
                "read_only": True,
                "will_execute_external_action": True,
                "description": "Search configured Google People/Contacts surfaces.",
            }
        )
    odollo_config = config.get("odollo") if isinstance(config.get("odollo"), dict) else {}
    for profile in odollo_config.get("profiles") if isinstance(odollo_config.get("profiles"), list) else []:
        if isinstance(profile, str):
            profile = {"label": profile}
        if not isinstance(profile, dict) or profile.get("enabled") is False:
            continue
        label = str(profile.get("label") or profile.get("profile") or "")
        if not label:
            continue
        sources.append(
            {
                "source": "odollo",
                "profile": label,
                "label": f"Odollo {label}",
                "models": ["res.partner"],
                "read_only": True,
                "will_execute_external_action": True,
                "description": "Search configured read-only Odollo contact provenance.",
            }
        )
    msgcli_config = config.get("msgcli") if isinstance(config.get("msgcli"), dict) else {}
    if msgcli_config:
        sources.append(
            {
                "source": "msgcli",
                "profile": str(msgcli_config.get("profile") or "default"),
                "label": "msgcli mail metadata",
                "read_only": True,
                "enabled": bool(msgcli_config.get("enabled")),
                "will_execute_external_action": bool(msgcli_config.get("enabled")),
                "description": "Future bounded mail metadata source for relationship affinity.",
            }
        )
    return sources


def context_contact_refresh_preview(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    query: str = "",
    source_filters: Optional[list[str]] = None,
) -> dict[str, Any]:
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    source_filter_set = {str(value or "").strip().lower() for value in (source_filters or []) if str(value or "").strip()}
    sources = configured_contact_refresh_sources(state_root=state_root)
    if source_filter_set:
        sources = [
            source
            for source in sources
            if str(source.get("source") or "").lower() in source_filter_set
            or str(source.get("profile") or "").lower() in source_filter_set
        ]
    return {
        "schema_version": "transcribe-audio.context-contact-refresh-preview.v1",
        "status": "ready",
        "conversation_key": conversation_key,
        "query": query.strip(),
        "sources": sources,
        "source_count": len(sources),
        "warnings": [] if sources else ["No configured contact refresh sources matched the requested filters."],
        "will_execute_external_action": any(bool(source.get("will_execute_external_action")) for source in sources),
        "will_perform_external_write": False,
    }


def context_contact_refresh_job_path(*, state_root: Path, job_id: str) -> Path:
    return conversation_context_contact_refresh_dir(state_root) / f"{job_id}.json"


def refresh_context_contacts(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    query: str = "",
    limit: int = 30,
    source_filters: Optional[list[str]] = None,
) -> dict[str, Any]:
    preview = context_contact_refresh_preview(
        document_id,
        root=root,
        state_root=state_root,
        query=query,
        source_filters=source_filters,
    )
    now = utcish_now()
    job_id = f"contact-refresh-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    search = search_context_contacts(
        document_id,
        root=root,
        state_root=state_root,
        query=query,
        limit=limit,
        mode="refresh",
        source_filters=source_filters,
    )
    source_counts = [
        {
            "source": source.get("source") or "",
            "profile": source.get("profile") or "",
            "status": "completed",
            "candidate_count": search.get("refreshed_candidate_count", 0),
        }
        for source in preview.get("sources", [])
        if isinstance(source, dict)
    ]
    manifest = {
        "schema_version": "transcribe-audio.context-contact-refresh-job.v1",
        "job_id": job_id,
        "status": "completed",
        "created_at": now,
        "finished_at": utcish_now(),
        "document_id": document_id,
        "conversation_key": preview.get("conversation_key", ""),
        "query": query.strip(),
        "source_filters": source_filters or [],
        "preview": preview,
        "source_counts": source_counts,
        "search": search,
        "warnings": search.get("warnings", []),
        "will_execute_external_action": preview.get("will_execute_external_action", False),
        "will_perform_external_write": False,
    }
    path = context_contact_refresh_job_path(state_root=state_root, job_id=job_id)
    write_json_file(path, manifest)
    return {
        "schema_version": "transcribe-audio.context-contact-refresh.v1",
        "status": "completed",
        "job_id": job_id,
        "job_path": str(path),
        "source_counts": source_counts,
        "items": search.get("items", []),
        "total": search.get("total", 0),
        "warnings": search.get("warnings", []),
        "cache_status": search.get("cache_status", ""),
        "affinity_cache_status": search.get("affinity_cache_status", ""),
        "will_execute_external_action": preview.get("will_execute_external_action", False),
        "will_perform_external_write": False,
    }


def read_context_contact_refresh_job(*, state_root: Path, job_id: str) -> dict[str, Any]:
    path = context_contact_refresh_job_path(state_root=state_root, job_id=job_id)
    payload = read_json_file(path)
    if not payload:
        raise FileNotFoundError(f"Contact refresh job not found: {job_id}")
    return {**payload, "job_path": str(path)}


def record_context_contact_merge_batch(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not actions:
        raise ValueError("Context contact merge batch requires at least one action.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    if not conversation_key:
        raise ValueError("Context contact merge requires a conversation key.")
    path = context_contact_merge_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    decisions = payload.get("decisions") if isinstance(payload.get("decisions"), list) else []
    new_decisions: list[dict[str, Any]] = []
    lookup = context_contact_lookup(detail)
    for item in actions:
        if not isinstance(item, dict):
            raise ValueError("Context contact merge actions must be objects.")
        action = str(item.get("action") or "").strip().lower()
        if action not in {"merge", "split"}:
            raise ValueError("Context contact merge action must be merge or split.")
        actor_type = str(item.get("actor_type") or "operator").strip().lower() or "operator"
        if actor_type not in {"operator", "app_intelligence"}:
            raise ValueError("Context contact merge actor_type must be operator or app_intelligence.")
        reviewer = str(item.get("reviewer") or actor_type).strip() or actor_type
        contact_ids = [
            str(value or "").strip()
            for value in item.get("contact_ids", [])
            if str(value or "").strip()
        ] if isinstance(item.get("contact_ids"), list) else []
        if len(contact_ids) < 2:
            raise ValueError("Context contact merge/split requires at least two contact_ids.")
        canonical_payload = item.get("canonical_candidate") if isinstance(item.get("canonical_candidate"), dict) else {}
        candidate_sources = [lookup[contact_id] for contact_id in contact_ids if contact_id in lookup]
        canonical = compact_context_contact_candidate(canonical_payload) if canonical_payload else {}
        if not canonical and candidate_sources:
            canonical = compact_context_contact_candidate(candidate_sources[0])
        now = utcish_now()
        decision_id = stable_id("context-contact-merge", conversation_key, action, ",".join(sorted(contact_ids)), now, str(uuid.uuid4()))
        new_decisions.append(
            {
                "decision_id": decision_id,
                "action": action,
                "actor_type": actor_type,
                "reviewer": reviewer,
                "note": str(item.get("note") or "").strip(),
                "created_at": now,
                "contact_ids": contact_ids,
                "dedupe_key": str(item.get("dedupe_key") or ""),
                "merge_key": stable_id("reviewed-contact-merge", conversation_key, *sorted(contact_ids)) if action == "merge" else "",
                "canonical_candidate": canonical,
                "sources": [compact_context_contact_candidate(candidate) for candidate in candidate_sources],
                "will_execute_external_action": False,
                "will_perform_external_write": False,
            }
        )
    updated_at = utcish_now()
    payload = {
        "schema_version": "transcribe-audio.context-contact-merge.v1",
        "conversation_key": conversation_key,
        "document_id": document_id,
        "updated_at": updated_at,
        "decisions": [*decisions, *new_decisions],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, payload)
    refreshed = get_conversation_detail(document_id, root=root, state_root=state_root)
    return {
        "schema_version": "transcribe-audio.context-contact-merge-batch.v1",
        "status": "recorded",
        "decisions": new_decisions,
        "merge_path": str(path),
        "context_workbench": refreshed.get("context_workbench") or {},
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def record_context_instructions(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    instruction_text: str,
    actor_type: str = "operator",
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    actor_type = actor_type.strip().lower() or "operator"
    if actor_type not in {"operator", "app_intelligence"}:
        raise ValueError("Context instruction actor_type must be operator or app_intelligence.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
    if not conversation_key:
        raise ValueError("Context instructions require a conversation key.")
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    path = context_instructions_path(state_root=state_root, conversation_key=conversation_key)
    payload = read_json_file(path) if path.exists() else {}
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    now = utcish_now()
    text = instruction_text.strip()
    entry = {
        "entry_id": stable_id("context-instruction", conversation_key, now, str(uuid.uuid4())),
        "instruction_text": text,
        "actor_type": actor_type,
        "reviewer": reviewer.strip() or actor_type,
        "note": note.strip(),
        "created_at": now,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    payload = {
        "schema_version": "transcribe-audio.context-operator-context.v1",
        "conversation_key": conversation_key,
        "document_id": document_id,
        "source_document_id": source_document.get("id") or "",
        "instruction_text": text,
        "actor_type": actor_type,
        "reviewer": reviewer.strip() or actor_type,
        "updated_at": now,
        "history": [*history, entry],
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
    write_json_file(path, payload)
    refreshed_context = context_workbench_state(
        detail=detail,
        root=root,
        state_root=state_root,
    )
    return {
        "schema_version": "transcribe-audio.context-instructions-action.v1",
        "status": "saved" if text else "cleared",
        "entry": entry,
        "instruction_path": str(path),
        "context_workbench": refreshed_context,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def route_path_for_contextual_document(document: dict[str, Any] | None) -> Path | None:
    if not document:
        return None
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    provider = metadata.get("provider") if isinstance(metadata.get("provider"), dict) else {}
    path_text = str(provider.get("route_decision_path") or "")
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    return path if path.exists() and path.is_file() else None


def create_deposition_memory_preview(
    detail: dict[str, Any],
    *,
    state_root: Path,
) -> tuple[Path, dict[str, Any]]:
    contextual_document = detail.get("contextual_readout_document")
    readout_path = artifact_path_for_document(contextual_document)
    if not readout_path:
        raise TranscriptStoreError("No contextual readout artifact is available for preview.")
    from deposition_preview import generate_deposition_preview

    preview_root = conversation_preview_dir(state_root) / "previews"
    route_path = route_path_for_contextual_document(contextual_document)
    transcript_path = artifact_path_for_document(detail.get("transcript_document"))
    args = argparse.Namespace(
        readout=readout_path,
        route=route_path,
        transcript=transcript_path,
        output_dir=preview_root,
        local_root=state_root.expanduser() / "deposition-preview-target",
        drive_folder_id="",
        drive_profile="",
        odoo_profile="",
        odoo_model="",
        odoo_record_id="",
        graphiti_group="transcribe_audio_main",
        include_transcript=False,
    )
    preview_path = generate_deposition_preview(args)
    return preview_path, read_json_file(preview_path)


def queue_deposition_memory_preview(
    document_id: str,
    *,
    root: Optional[Path],
    state_root: Path,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != DEPOSITION_MEMORY_PREVIEW_TOKEN:
        raise ValueError(f"Queueing preview review requires approval_token={DEPOSITION_MEMORY_PREVIEW_TOKEN}.")
    detail = get_conversation_detail(document_id, root=root, state_root=state_root)
    current_preview = final_preview_state(detail)
    if current_preview.get("identity_context_warnings"):
        return {
            "schema_version": "transcribe-audio.deposition-memory-preview-queue.v1",
            "status": "blocked_identity_or_context_review",
            "review_item_path": "",
            "review_item_id": "",
            "final_preview": current_preview,
            "will_execute_external_action": False,
            "will_perform_external_write": False,
        }
    preview_path, preview_payload = create_deposition_memory_preview(detail, state_root=state_root)
    summary = deposition_preview_summary(preview_path, preview_payload)
    now = utcish_now()
    item_id = stable_id("deposition-memory-preview", document_id, str(preview_path), now)
    queue_path = review_queue_dir(state_root) / f"{item_id}.conversation-preview-review.json"
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    write_json_file(
        queue_path,
        {
            "schema_version": "transcribe-audio.conversation-preview-review.v1",
            "id": item_id,
            "type": "deposition_memory_preview",
            "status": "pending",
            "created_at": now,
            "document_id": source_document.get("id") or document_id,
            "representative_document_id": document_id,
            "conversation_key": detail["conversation"]["key"],
            "workflow_stage": "output",
            "label": detail["conversation"]["title"],
            "reason": "Review deposition and memory-harvest preview before any external apply.",
            "preview_path": str(preview_path),
            "action_count": summary["action_count"],
            "memory_candidate_count": summary["memory_candidate_count"],
            "will_execute_external_action": False,
            "will_perform_external_write": False,
        },
    )
    return {
        "schema_version": "transcribe-audio.deposition-memory-preview-queue.v1",
        "status": "queued_for_review",
        "review_item_path": str(queue_path),
        "review_item_id": item_id,
        "final_preview": summary,
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def retranscription_preflight(
    document_id: str,
    *,
    root: Optional[Path] = None,
    state_root: Path = DEFAULT_STATE_DIR,
    backend: str = "faster_whisper",
    output_dir: str = "",
) -> dict[str, Any]:
    selected_backend = backend if backend in RETRANSCRIPTION_BACKENDS else "faster_whisper"
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        source_row = row
        if row["kind"] != "transcript":
            payload = parse_object_json(row["json_payload"])
            metadata = parse_object_json(row["metadata_json"])
            source_artifact_path = str(metadata.get("source_artifact_path") or payload.get("source_artifact_path") or "")
            if source_artifact_path:
                matched = con.execute("SELECT * FROM documents WHERE source_path = ?", (source_artifact_path,)).fetchone()
                if matched is not None:
                    source_row = matched
        blob = con.execute(
            """
            SELECT blobs.*
            FROM document_blobs
            JOIN blobs ON document_blobs.blob_id = blobs.id
            WHERE document_blobs.document_id = ?
              AND document_blobs.role = 'source_recording'
            ORDER BY blobs.id
            LIMIT 1
            """,
            (source_row["id"],),
        ).fetchone()

    planned_output_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else state_root.expanduser() / "retranscriptions" / str(source_row["id"])
    )
    base_name = Path(str(source_row["source_path"])).stem.replace(".transcript", "")
    script_name = "faster_whisper_transcribe.py" if selected_backend == "faster_whisper" else "assembly_transcribe.py"
    source_media_path = str(blob["stored_path"]) if blob is not None else ""
    command = [
        "python",
        script_name,
        source_media_path or "<missing-source-recording>",
        "--output-dir",
        str(planned_output_dir),
        "--text-output",
    ]
    return {
        "schema_version": "transcribe-audio.retranscription-preflight.v1",
        "ok": blob is not None,
        "document": document_summary(row),
        "source_document": document_summary(source_row),
        "selected_backend": selected_backend,
        "source_blob": {
            "id": blob["id"],
            "role": blob["role"],
            "mime_type": blob["mime_type"],
            "bytes": int(blob["bytes"]),
            "sha256": blob["sha256"],
            "playback_url": f"/api/blobs/{blob['id']}",
            "download_url": f"/api/blobs/{blob['id']}?download=1",
        } if blob is not None else None,
        "planned_outputs": {
            "output_dir": str(planned_output_dir),
            "transcript_json": str(planned_output_dir / f"{base_name}.transcript.json"),
            "docx": str(planned_output_dir / f"{base_name} Transcript.docx"),
            "txt": str(planned_output_dir / f"{base_name} Transcript.txt"),
        },
        "command": command,
        "blocking_checks": [] if blob is not None else ["source_recording_blob_missing"],
        "will_queue": False,
        "will_run_transcription": False,
        "will_write_files": False,
        "future_required_approval_token_for_queue": RETRANSCRIPTION_PREFLIGHT_TOKEN,
    }


def get_blob(blob_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM blobs WHERE id = ?", (blob_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No blob found with id {blob_id}")
    path = Path(row["stored_path"])
    if not path.exists() or not path.is_file():
        raise TranscriptStoreError(f"Blob file is missing for id {blob_id}")
    return {
        "id": row["id"],
        "role": row["role"],
        "path": path,
        "mime_type": row["mime_type"] or "application/octet-stream",
        "bytes": int(row["bytes"]),
        "sha256": row["sha256"],
    }


def parse_int(value: str, default: int, *, minimum: int = 0, maximum: int = 500) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key) or []
    return values[0] if values else default


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def retranscription_jobs_dir(state_root: Path) -> Path:
    return state_root.expanduser() / RETRANSCRIPTION_JOB_DIRNAME


def summarize_retranscription_job(path: Path) -> dict[str, Any]:
    payload = read_json_file(path)
    preflight = payload.get("preflight") if isinstance(payload.get("preflight"), dict) else {}
    return {
        "job_id": str(payload.get("job_id") or path.stem),
        "status": str(payload.get("status") or "unknown"),
        "created_at": str(payload.get("created_at") or ""),
        "path": str(path),
        "document_id": str(payload.get("document_id") or ""),
        "source_document_id": str(payload.get("source_document_id") or ""),
        "selected_backend": str(payload.get("selected_backend") or ""),
        "source_blob_id": str(payload.get("source_blob_id") or ""),
        "planned_outputs": preflight.get("planned_outputs") if isinstance(preflight.get("planned_outputs"), dict) else {},
        "will_run_transcription": bool(payload.get("will_run_transcription")),
        "will_write_files": bool(payload.get("will_write_files")),
        "will_execute_external_action": bool(payload.get("will_execute_external_action")),
    }


def enqueue_retranscription_job(
    document_id: str,
    *,
    root: Optional[Path] = None,
    state_root: Path = DEFAULT_STATE_DIR,
    backend: str = "faster_whisper",
    output_dir: str = "",
    approval_token: str = "",
) -> dict[str, Any]:
    if approval_token != RETRANSCRIPTION_PREFLIGHT_TOKEN:
        raise ValueError(f"Queueing requires approval_token={RETRANSCRIPTION_PREFLIGHT_TOKEN}.")
    preflight = retranscription_preflight(
        document_id,
        root=root,
        state_root=state_root,
        backend=backend,
        output_dir=output_dir,
    )
    if not preflight.get("ok"):
        return {
            "schema_version": "transcribe-audio.retranscription-job-queue.v1",
            "ok": False,
            "status": "blocked",
            "preflight": preflight,
            "blocking_checks": preflight.get("blocking_checks") or [],
            "will_create_job_record": False,
            "will_run_transcription": False,
            "will_write_files": False,
            "will_execute_external_action": False,
        }
    job_root = retranscription_jobs_dir(state_root)
    job_root.mkdir(parents=True, exist_ok=True)
    job_id = f"retranscription-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    job_path = job_root / f"{job_id}.json"
    source_document = preflight.get("source_document") if isinstance(preflight.get("source_document"), dict) else {}
    source_blob = preflight.get("source_blob") if isinstance(preflight.get("source_blob"), dict) else {}
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "transcribe-audio.retranscription-job.v1",
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "document_id": document_id,
        "source_document_id": str(source_document.get("id") or ""),
        "selected_backend": str(preflight.get("selected_backend") or ""),
        "source_blob_id": str(source_blob.get("id") or ""),
        "preflight": preflight,
        "approval_token_checked": RETRANSCRIPTION_PREFLIGHT_TOKEN,
        "will_run_transcription": False,
        "will_write_files": False,
        "will_execute_external_action": False,
        "future_required_approval_token_for_run": "RUN_RETRANSCRIPTION_JOB",
    }
    write_json_atomic(job_path, payload)
    return {
        "schema_version": "transcribe-audio.retranscription-job-queue.v1",
        "ok": True,
        "status": "queued",
        "job": summarize_retranscription_job(job_path),
        "preflight": preflight,
        "required_approval_token_checked": RETRANSCRIPTION_PREFLIGHT_TOKEN,
        "will_start_background_job": False,
        "will_run_transcription": False,
        "will_write_files": False,
        "will_execute_external_action": False,
        "future_required_approval_token_for_run": "RUN_RETRANSCRIPTION_JOB",
    }


def app_intelligence_smoke_status(*, state_root: Path, limit: int = 5) -> dict[str, Any]:
    state_root = state_root.expanduser()
    runs_dir = state_root / "app-intelligence-runs"
    browser_smoke_dir = state_root / APP_BROWSER_SMOKE_DIRNAME
    run_dirs = sorted(
        [
            path
            for path in runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(APP_SMOKE_RUN_PREFIX)
        ] if runs_dir.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    reports = sorted(
        [path for path in browser_smoke_dir.glob("*.json") if path.is_file()] if browser_smoke_dir.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    report_items: list[dict[str, Any]] = []
    for path in reports[:limit]:
        payload = read_json_file(path)
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        screenshot_path = Path(str(payload.get("screenshot_path") or ""))
        report_items.append(
            {
                "path": str(path),
                "status": str(payload.get("status") or "unknown"),
                "run_id": str(payload.get("run_id") or ""),
                "created_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "screenshot_path": str(screenshot_path) if str(screenshot_path) else "",
                "screenshot_exists": bool(str(screenshot_path)) and screenshot_path.exists(),
                "missing_checks": payload.get("missing_checks") if isinstance(payload.get("missing_checks"), list) else [],
                "checks": checks,
            }
        )

    run_items: list[dict[str, Any]] = []
    for path in run_dirs[:limit]:
        ledger = read_json_file(path / "run.json")
        run_items.append(
            {
                "run_id": str(ledger.get("run_id") or path.name),
                "workflow": str(ledger.get("workflow") or ""),
                "status": str(ledger.get("status") or ""),
                "phase": str(ledger.get("phase") or ""),
                "path": str(path),
                "updated_at": str(ledger.get("updated_at") or ""),
            }
        )

    return {
        "schema_version": "transcribe-audio.app-smoke-status.v1",
        "state_root": str(state_root),
        "run_prefix": APP_SMOKE_RUN_PREFIX,
        "runs_dir": str(runs_dir),
        "browser_smoke_dir": str(browser_smoke_dir),
        "latest_report": report_items[0] if report_items else None,
        "reports": report_items,
        "report_count": len(reports),
        "runs": run_items,
        "run_count": len(run_dirs),
        "will_read_artifact_content": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def smoke_jobs_dir(state_root: Path) -> Path:
    return state_root.expanduser() / APP_SMOKE_JOB_DIRNAME


def app_smoke_job_path(state_root: Path, job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
        raise ValueError("Invalid smoke job id.")
    return smoke_jobs_dir(state_root) / f"{job_id}.json"


def app_smoke_job_command(
    *,
    job_type: str,
    state_root: Path,
    base_url: str,
    cleanup: bool = True,
    apply_cleanup: bool = False,
) -> tuple[list[str], int, str]:
    repo_root = Path(__file__).resolve().parent
    state_root = state_root.expanduser()
    if job_type == "api_replay_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_app_replay_manifest.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "browser_replay_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_app_replay_manifest_ui.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "first_pass_resume_ui_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_first_pass_batch_resume_ui.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "cleanup_smokes":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "cleanup_app_smokes.py"),
            "--state-root",
            str(state_root),
            "--format",
            "json",
        ]
        if apply_cleanup:
            command.append("--apply")
        return command, 60, APP_SMOKE_CLEANUP_TOKEN if apply_cleanup else APP_SMOKE_JOB_TOKEN
    raise ValueError(f"Unsupported smoke job type: {job_type}")


def write_app_smoke_job(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def cleanup_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_smoke_cleanup_summary(text: str) -> dict[str, Any] | None:
    prefix = "APP_SMOKE_CLEANUP_JSON="
    for line in str(text or "").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "schema_version": str(payload.get("schema_version") or "transcribe-audio.app-smoke-cleanup.v1"),
            "apply": bool(payload.get("apply")),
            "matched_run_count": cleanup_count(payload.get("matched_run_count")),
            "kept_run_count": cleanup_count(payload.get("kept_run_count")),
            "delete_run_count": cleanup_count(payload.get("delete_run_count")),
            "matched_evidence_count": cleanup_count(payload.get("matched_evidence_count")),
            "keep_evidence": cleanup_count(payload.get("keep_evidence")),
            "evidence_days": cleanup_count(payload.get("evidence_days")),
            "delete_evidence_count": cleanup_count(payload.get("delete_evidence_count")),
        }
    return None


def parse_smoke_evidence_summary(text: str) -> dict[str, Any] | None:
    prefixes = [
        "APP_REPLAY_MANIFEST_UI_SMOKE_JSON=",
        "FIRST_PASS_RESUME_UI_SMOKE_JSON=",
    ]
    for line in str(text or "").splitlines():
        prefix = next((item for item in prefixes if line.startswith(item)), "")
        if not prefix:
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        report_path = str(payload.get("report_path") or "")
        screenshot_path = str(payload.get("screenshot_path") or "")
        return {
            "schema_version": str(payload.get("schema_version") or "transcribe-audio.smoke-evidence.v1"),
            "stdout_prefix": prefix.removesuffix("="),
            "status": str(payload.get("status") or "unknown"),
            "report_path": report_path,
            "screenshot_path": screenshot_path,
            "report_url": f"/api/intelligence/smoke-evidence?path={quote(report_path, safe='')}" if report_path else "",
            "screenshot_url": f"/api/intelligence/smoke-evidence?path={quote(screenshot_path, safe='')}" if screenshot_path else "",
            "check_count": len(checks),
            "failed_check_count": len([value for value in checks.values() if value is not True]),
        }
    return None


def summarize_smoke_job(path: Path) -> dict[str, Any]:
    payload = read_json_file(path)
    if not payload:
        return {}
    stdout_path = Path(str(payload.get("stdout_path") or ""))
    stderr_path = Path(str(payload.get("stderr_path") or ""))
    summary = {
        "job_id": str(payload.get("job_id") or path.stem),
        "job_type": str(payload.get("job_type") or ""),
        "status": str(payload.get("status") or "unknown"),
        "created_at": str(payload.get("created_at") or ""),
        "started_at": str(payload.get("started_at") or ""),
        "finished_at": str(payload.get("finished_at") or ""),
        "returncode": payload.get("returncode"),
        "path": str(path),
        "stdout_path": str(stdout_path) if str(stdout_path) else "",
        "stderr_path": str(stderr_path) if str(stderr_path) else "",
        "stdout_exists": bool(str(stdout_path)) and stdout_path.exists(),
        "stderr_exists": bool(str(stderr_path)) and stderr_path.exists(),
        "will_execute_write_bearing_action": bool(payload.get("will_execute_write_bearing_action")),
        "will_execute_external_action": bool(payload.get("will_execute_external_action")),
        "will_read_artifact_content": False,
    }
    cleanup_summary = parse_smoke_cleanup_summary(str(payload.get("stdout_tail") or ""))
    if cleanup_summary is not None:
        summary["cleanup_summary"] = cleanup_summary
    evidence_summary = parse_smoke_evidence_summary(str(payload.get("stdout_tail") or ""))
    if evidence_summary is not None:
        summary["evidence_summary"] = evidence_summary
    return summary


def resolve_smoke_evidence_path(*, state_root: Path, path_value: str) -> Path:
    if not path_value:
        raise ValueError("Missing smoke evidence path.")
    root = (state_root.expanduser() / APP_BROWSER_SMOKE_DIRNAME).resolve()
    candidate = Path(path_value).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError("Smoke evidence path is outside the browser-smokes directory.")
    if resolved.suffix.lower() not in {".json", ".png"}:
        raise ValueError("Smoke evidence must be a JSON report or PNG screenshot.")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    if resolved.stat().st_size > MAX_SMOKE_EVIDENCE_BYTES:
        raise ValueError("Smoke evidence file is too large to serve.")
    return resolved


def resolve_smoke_job_output_path(*, state_root: Path, job: dict[str, Any], stream: str) -> Path:
    if stream not in {"stdout", "stderr"}:
        raise ValueError("stream must be stdout or stderr.")
    root = smoke_jobs_dir(state_root).resolve()
    candidate = Path(str(job.get(f"{stream}_path") or "")).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError("Smoke job output path is outside the smoke job directory.")
    return resolved


def read_app_smoke_job_tail(*, state_root: Path, job_id: str, stream: str = "stderr", chars: int = 4000) -> dict[str, Any]:
    job_path = app_smoke_job_path(state_root, job_id)
    job = read_json_file(job_path)
    if not job:
        raise FileNotFoundError(str(job_path))
    output_path = resolve_smoke_job_output_path(state_root=state_root, job=job, stream=stream)
    if not output_path.exists() or not output_path.is_file():
        text = ""
    else:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    limit = max(1, min(chars, MAX_SMOKE_JOB_TAIL_CHARS))
    return {
        "schema_version": "transcribe-audio.app-smoke-job-tail.v1",
        "job": summarize_smoke_job(job_path),
        "job_id": str(job.get("job_id") or job_path.stem),
        "stream": stream,
        "path": str(output_path),
        "exists": output_path.exists(),
        "bytes": output_path.stat().st_size if output_path.exists() else 0,
        "char_count": len(text),
        "tail_chars": limit,
        "tail": text[-limit:],
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
        "will_read_arbitrary_file": False,
    }


def list_app_smoke_jobs(*, state_root: Path, limit: int = 10) -> dict[str, Any]:
    root = smoke_jobs_dir(state_root)
    paths = sorted(
        [path for path in root.glob("*.json") if path.is_file()] if root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "schema_version": "transcribe-audio.app-smoke-jobs.v1",
        "job_dir": str(root),
        "items": [summarize_smoke_job(path) for path in paths[:limit]],
        "total": len(paths),
        "available_job_types": [
            "api_replay_smoke",
            "browser_replay_smoke",
            "first_pass_resume_ui_smoke",
            "cleanup_smokes",
        ],
        "will_read_artifact_content": False,
    }


def run_app_smoke_job(job_path: Path) -> None:
    payload = read_json_file(job_path)
    if not payload:
        return
    command = payload.get("command") if isinstance(payload.get("command"), list) else []
    timeout = int(payload.get("timeout_seconds") or APP_SMOKE_JOB_TIMEOUT_SECONDS)
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload.update({"status": "running", "started_at": started_at})
    write_app_smoke_job(job_path, payload)
    stdout_path = Path(str(payload.get("stdout_path") or ""))
    stderr_path = Path(str(payload.get("stderr_path") or ""))
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        payload.update(
            {
                "status": "succeeded" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:],
            }
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = subprocess_text(exc.stdout)
        stderr_text = subprocess_text(exc.stderr)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        payload.update(
            {
                "status": "failed",
                "returncode": None,
                "error": f"Timed out after {timeout} seconds.",
                "stdout_tail": stdout_text[-2000:],
                "stderr_tail": stderr_text[-2000:],
            }
        )
    except OSError as exc:
        payload.update({"status": "failed", "returncode": None, "error": str(exc)})
    payload["finished_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_app_smoke_job(job_path, payload)


def enqueue_app_smoke_job(
    *,
    state_root: Path,
    job_type: str,
    approval_token: str,
    base_url: str,
    cleanup: bool = True,
    apply_cleanup: bool = False,
    start_thread: bool = True,
) -> dict[str, Any]:
    command, timeout, required_token = app_smoke_job_command(
        job_type=job_type,
        state_root=state_root,
        base_url=base_url,
        cleanup=cleanup,
        apply_cleanup=apply_cleanup,
    )
    if approval_token != required_token:
        raise ValueError(f"{job_type} requires approval_token={required_token}.")
    root = smoke_jobs_dir(state_root)
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"{job_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    job_path = root / f"{job_id}.json"
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "transcribe-audio.app-smoke-job.v1",
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "created_at": created_at,
        "base_url": base_url.rstrip("/"),
        "command": command,
        "timeout_seconds": timeout,
        "stdout_path": str(root / f"{job_id}.stdout.txt"),
        "stderr_path": str(root / f"{job_id}.stderr.txt"),
        "cleanup": cleanup,
        "apply_cleanup": apply_cleanup,
        "will_execute_external_action": job_type in {"browser_replay_smoke", "first_pass_resume_ui_smoke"},
        "will_execute_write_bearing_action": job_type == "cleanup_smokes" and apply_cleanup,
        "will_read_artifact_content": False,
    }
    write_app_smoke_job(job_path, payload)
    if start_thread:
        threading.Thread(target=run_app_smoke_job, args=(job_path,), daemon=True).start()
    return {
        "schema_version": "transcribe-audio.app-smoke-job-enqueue.v1",
        "job": summarize_smoke_job(job_path),
        "required_approval_token_checked": required_token,
        "will_start_background_job": bool(start_thread),
        "will_execute_arbitrary_shell": False,
    }


def read_app_intelligence_events_file(run_path: Path) -> list[dict[str, Any]]:
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def collect_app_intelligence_artifact_paths(run: dict[str, Any], run_path: Path) -> set[Path]:
    candidates: list[str] = []
    latest_status = run.get("latest_model_turn_status") if isinstance(run.get("latest_model_turn_status"), dict) else {}
    candidates.append(str(latest_status.get("artifact_path") or ""))
    for packet in run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []:
        if isinstance(packet, dict):
            candidates.append(str(packet.get("packet_path") or ""))
            candidates.append(str(packet.get("prompt_path") or ""))
    for decision in run.get("decisions") if isinstance(run.get("decisions"), list) else []:
        if not isinstance(decision, dict):
            continue
        candidates.append(str(decision.get("artifact_path") or ""))
        apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
        candidates.append(str(apply_result.get("artifact_path") or ""))
    for event in read_app_intelligence_events_file(run_path):
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidates.append(str(payload.get("artifact_path") or ""))
        candidates.append(str(payload.get("packet_path") or ""))
        candidates.append(str(payload.get("prompt_path") or ""))

    root = run_path.resolve()
    allowed: set[Path] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = run_path / path
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        allowed.add(resolved)
    return allowed


def app_intelligence_replay_manifest(*, state_root: Path, run_id: str) -> dict[str, Any]:
    shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    run_path = Path(str(shown.get("path") or ""))
    allowed_paths = collect_app_intelligence_artifact_paths(run, run_path)
    events = read_app_intelligence_events_file(run_path)
    seen: set[Path] = set()
    artifacts: list[dict[str, Any]] = []

    def resolve_registered(path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = run_path / path
        resolved = path.resolve()
        if resolved not in allowed_paths:
            return None
        return resolved

    def add_artifact(
        path_value: str,
        *,
        artifact_role: str,
        label: str,
        source: str,
        created_at: str = "",
        event_id: str = "",
        event_type: str = "",
        packet_id: str = "",
        decision_id: str = "",
        codex_turn_id: str = "",
    ) -> None:
        resolved = resolve_registered(path_value)
        if resolved is None or resolved in seen:
            return
        seen.add(resolved)
        exists = resolved.exists() and resolved.is_file()
        suffix = resolved.suffix.lower()
        artifacts.append(
            {
                "artifact_id": f"artifact-{len(artifacts) + 1:03d}",
                "artifact_role": artifact_role,
                "label": label,
                "path": str(resolved),
                "relative_path": str(resolved.relative_to(run_path.resolve())),
                "artifact_type": "json" if suffix == ".json" else "text",
                "exists": exists,
                "bytes": resolved.stat().st_size if exists else None,
                "source": source,
                "created_at": created_at,
                "event_id": event_id,
                "event_type": event_type,
                "packet_id": packet_id,
                "decision_id": decision_id,
                "codex_turn_id": codex_turn_id,
                "can_read_via_artifact_endpoint": exists,
            }
        )

    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        event_type = str(event.get("event_type") or "")
        event_id = str(event.get("event_id") or "")
        created_at = str(event.get("created_at") or "")
        packet_id = str(payload.get("packet_id") or "")
        decision_id = str(payload.get("decision_id") or "")
        codex_turn_id = str(payload.get("codex_turn_id") or "")
        if payload.get("packet_path"):
            add_artifact(
                str(payload.get("packet_path") or ""),
                artifact_role="prompt_packet_json",
                label=f"Prompt packet {packet_id or len(artifacts) + 1}",
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
            )
        if payload.get("prompt_path"):
            add_artifact(
                str(payload.get("prompt_path") or ""),
                artifact_role="prompt_text",
                label=f"Prompt text {packet_id or len(artifacts) + 1}",
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
            )
        artifact_path = str(payload.get("artifact_path") or "")
        if artifact_path:
            if event_type == "model_turn_status_captured":
                role = "model_turn_status"
                label = f"Turn status {codex_turn_id or len(artifacts) + 1}"
            elif event_type == "structured_decision_validated":
                role = "structured_decision_validation"
                label = f"Decision validation {decision_id or len(artifacts) + 1}"
            elif event_type == "structured_decision_applied":
                role = "structured_decision_apply"
                label = f"Decision apply {decision_id or len(artifacts) + 1}"
            elif "preflight" in event_type:
                role = "preflight"
                label = event_type.replace("_", " ").title()
            else:
                role = "event_artifact"
                label = event_type.replace("_", " ").title() or "Event artifact"
            add_artifact(
                artifact_path,
                artifact_role=role,
                label=label,
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
                decision_id=decision_id,
                codex_turn_id=codex_turn_id,
            )

    for packet in run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []:
        if not isinstance(packet, dict):
            continue
        packet_id = str(packet.get("packet_id") or "")
        add_artifact(
            str(packet.get("packet_path") or ""),
            artifact_role="prompt_packet_json",
            label=f"Prompt packet {packet_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(packet.get("created_at") or ""),
            packet_id=packet_id,
        )
        add_artifact(
            str(packet.get("prompt_path") or ""),
            artifact_role="prompt_text",
            label=f"Prompt text {packet_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(packet.get("created_at") or ""),
            packet_id=packet_id,
        )
    latest_status = run.get("latest_model_turn_status") if isinstance(run.get("latest_model_turn_status"), dict) else {}
    add_artifact(
        str(latest_status.get("artifact_path") or ""),
        artifact_role="model_turn_status",
        label=f"Turn status {latest_status.get('codex_turn_id') or len(artifacts) + 1}",
        source="run_ledger",
        created_at=str(latest_status.get("captured_at") or ""),
        codex_turn_id=str(latest_status.get("codex_turn_id") or ""),
    )
    for decision in run.get("decisions") if isinstance(run.get("decisions"), list) else []:
        if not isinstance(decision, dict):
            continue
        decision_id = str(decision.get("decision_id") or "")
        add_artifact(
            str(decision.get("artifact_path") or ""),
            artifact_role="structured_decision_validation",
            label=f"Decision validation {decision_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(decision.get("created_at") or ""),
            decision_id=decision_id,
        )
        apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
        add_artifact(
            str(apply_result.get("artifact_path") or ""),
            artifact_role="structured_decision_apply",
            label=f"Decision apply {decision_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(decision.get("applied_at") or ""),
            decision_id=decision_id,
        )

    return {
        "schema_version": "transcribe-audio.app-intelligence-replay-manifest.v1",
        "run_id": run_id,
        "run_path": str(run_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
        "will_read_artifact_content": False,
    }


def read_app_intelligence_artifact(*, state_root: Path, run_id: str, artifact_path: str) -> dict[str, Any]:
    if not artifact_path:
        raise ValueError("Missing required query parameter: path")
    shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    run_path = Path(str(shown.get("path") or ""))
    requested = Path(artifact_path).expanduser()
    if not requested.is_absolute():
        requested = run_path / requested
    resolved = requested.resolve()
    run_root = run_path.resolve()
    try:
        relative_path = resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("Artifact path resolves outside the App Intelligence run directory.") from exc
    if resolved not in collect_app_intelligence_artifact_paths(run, run_path):
        raise ValueError("Artifact path is not registered in this App Intelligence run ledger or event log.")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("App Intelligence artifact file is missing.")
    size = resolved.stat().st_size
    if size > MAX_APP_ARTIFACT_BYTES:
        raise ValueError(f"Artifact is too large to display through this endpoint: {size} bytes.")
    raw = resolved.read_text(encoding="utf-8")
    parsed_json: Any = None
    artifact_type = "text"
    try:
        parsed_json = json.loads(raw)
        artifact_type = "json"
    except json.JSONDecodeError:
        parsed_json = None
    return {
        "schema_version": "transcribe-audio.app-intelligence-artifact-read.v1",
        "run_id": run_id,
        "path": str(resolved),
        "relative_path": str(relative_path),
        "artifact_type": artifact_type,
        "bytes": size,
        "json": parsed_json,
        "text": raw,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def compact_document_for_prompt(document: dict[str, Any], *, max_text_chars: int = 12000) -> dict[str, Any]:
    text = str(document.get("text_content") or "")
    return {
        "id": document.get("id") or "",
        "kind": document.get("kind") or "",
        "title": document.get("title") or "",
        "source_path": document.get("source_path") or "",
        "generated_at": document.get("generated_at") or "",
        "metadata": document.get("metadata") if isinstance(document.get("metadata"), dict) else {},
        "text_excerpt": text[:max_text_chars],
        "text_truncated": len(text) > max_text_chars,
        "text_chars": len(text),
    }


def model_turn_prompt_text(*, task: str, route: dict[str, Any], document: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the Transcripts App Intelligence worker.",
            "Do not perform external writes. Return structured analysis only.",
            "The host application owns routing, approvals, memory writes, repository writes, and final application.",
            "",
            f"Task: {task}",
            f"Provider route: {json.dumps(route, sort_keys=True, ensure_ascii=False)}",
            "",
            "Document:",
            f"- id: {document.get('id')}",
            f"- kind: {document.get('kind')}",
            f"- title: {document.get('title')}",
            f"- generated_at: {document.get('generated_at')}",
            f"- text_truncated: {document.get('text_truncated')}",
            "",
            "Transcript or readout text:",
            str(document.get("text_excerpt") or ""),
            "",
            "Return JSON with: summary, important_entities, candidate_context_sources, risks, recommended_next_actions, and review_flags.",
        ]
    )


def write_json_file(path: Path, payload: dict[str, Any]) -> Path:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _run_readiness_command(args: list[str], *, timeout: int = 10, probes: Optional[list[str]] = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[:MAX_READINESS_OUTPUT_CHARS],
        }
    stdout = completed.stdout[:MAX_READINESS_OUTPUT_CHARS]
    stderr = completed.stderr[:MAX_READINESS_OUTPUT_CHARS]
    full_text = f"{completed.stdout}\n{completed.stderr}"
    return {
        "args": args,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": len(completed.stdout) > MAX_READINESS_OUTPUT_CHARS,
        "stderr_truncated": len(completed.stderr) > MAX_READINESS_OUTPUT_CHARS,
        "probes": {probe: probe in full_text for probe in probes or []},
    }


def run_codex_command(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[:MAX_READINESS_OUTPUT_CHARS],
        }
    return {
        "args": args,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:MAX_READINESS_OUTPUT_CHARS],
        "stderr": completed.stderr[:MAX_READINESS_OUTPUT_CHARS],
        "stdout_truncated": len(completed.stdout) > MAX_READINESS_OUTPUT_CHARS,
        "stderr_truncated": len(completed.stderr) > MAX_READINESS_OUTPUT_CHARS,
    }


def codex_app_server_readiness(*, codex_bin: str = DEFAULT_CODEX_BIN) -> dict[str, Any]:
    resolved = shutil.which(codex_bin) if not Path(codex_bin).is_absolute() else codex_bin
    if not resolved:
        return {
            "id": "codex-app-server",
            "label": "Codex app-server",
            "status": "unavailable",
            "ready": False,
            "control_plane": "codex-app-server",
            "capabilities": {
                "persistent_sessions": True,
                "branching": True,
                "rollback": True,
                "streamed_events": True,
                "structured_decisions": True,
                "remote_transport": False,
            },
            "transports": ["stdio"],
            "checks": {
                "which": {
                    "args": ["which", codex_bin],
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "codex executable not found",
                }
            },
            "notes": [
                "Use codex app-server for supervised long-lived intelligence runs.",
                "Do not expose websocket transport without explicit auth and network-boundary review.",
            ],
        }

    version = _run_readiness_command([resolved, "--version"])
    app_server_help = _run_readiness_command(
        [resolved, "app-server", "--help"],
        probes=["--listen <URL>", "--ws-auth"],
    )
    schema_help = _run_readiness_command([resolved, "app-server", "generate-json-schema", "--help"])
    ts_help = _run_readiness_command([resolved, "app-server", "generate-ts", "--help"])
    help_probes = app_server_help.get("probes") if isinstance(app_server_help.get("probes"), dict) else {}
    ready = all(check.get("ok") for check in [version, app_server_help, schema_help, ts_help])
    return {
        "id": "codex-app-server",
        "label": "Codex app-server",
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "binary": resolved,
        "control_plane": "codex-app-server",
        "version": str(version.get("stdout") or "").strip(),
        "capabilities": {
            "persistent_sessions": True,
            "branching": True,
            "rollback": True,
            "streamed_events": True,
            "structured_decisions": True,
            "schema_generation": bool(schema_help.get("ok")),
            "typescript_generation": bool(ts_help.get("ok")),
            "remote_transport": bool(help_probes.get("--listen <URL>")),
            "websocket_auth": bool(help_probes.get("--ws-auth")),
        },
        "transports": ["stdio", "unix"] + (["websocket"] if help_probes.get("--listen <URL>") else []),
        "recommended_transport": "stdio",
        "checks": {
            "version": version,
            "app_server_help": app_server_help,
            "generate_json_schema_help": schema_help,
            "generate_ts_help": ts_help,
        },
        "notes": [
            "Use codex app-server for supervised long-lived intelligence runs with a host-owned ledger.",
            "Use codex exec for stateless leaf jobs and CI-style analysis.",
            "Do not expose websocket transport without explicit auth and network-boundary review.",
        ],
    }


def intelligence_provider_registry(*, codex_bin: str = DEFAULT_CODEX_BIN) -> dict[str, Any]:
    providers = [
        {
            "id": "openai-compatible",
            "label": "OpenAI-compatible API",
            "status": "configured-by-env",
            "capabilities": ["summarize", "contextual_reread", "classify", "extract"],
            "control_plane": "direct-http",
            "notes": ["Uses OPENAI_API_KEY and optional OPENAI_BASE_URL at execution time."],
        },
        {
            "id": "auracall",
            "label": "AuraCall",
            "status": "configured-by-runtime-env",
            "capabilities": ["summarize", "contextual_reread", "browser_backed_batch"],
            "control_plane": "openai-compatible-or-response-batch",
            "notes": ["Batch actions remain manifest-scoped and approval-gated."],
        },
        {
            "id": "codex-exec",
            "label": "codex exec",
            "status": "configured-by-cli",
            "capabilities": ["summarize", "contextual_reread", "leaf_analysis"],
            "control_plane": "process",
            "notes": ["Best for bounded one-shot jobs, not durable supervised sessions."],
        },
        codex_app_server_readiness(codex_bin=codex_bin),
        {
            "id": "openclaw",
            "label": "OpenClaw transcripts agent",
            "status": "configured-by-openclaw-runtime",
            "capabilities": ["notification", "routing_review", "agentic_context"],
            "control_plane": "openclaw",
            "notes": ["Use repo-local OpenClaw contracts and runtime checks before writes."],
        },
        {
            "id": "graphiti",
            "label": "Graphiti",
            "status": "advisory-memory",
            "capabilities": ["memory_lookup", "matter_candidates", "reviewed_memory_harvest"],
            "control_plane": "mcp",
            "notes": ["Graphiti-derived claims are advisory until verified against source artifacts."],
        },
        {
            "id": "local-embedder",
            "label": "Local embedder",
            "status": "configured-by-store",
            "capabilities": ["embed", "semantic_search"],
            "control_plane": "local-provider",
            "notes": ["Current store search uses configured embedding provider/model metadata."],
        },
    ]
    return {
        "providers": providers,
        "default_supervisor": "codex-app-server",
        "policy": {
            "host_owns_control_flow": True,
            "structured_decisions_required": True,
            "ledger_required_for_write_bearing_runs": True,
            "remote_transport_requires_auth_review": True,
        },
    }


def review_status(count: int, *, stale_count: int = 0, pending_count: int = 0) -> str:
    if count <= 0 and pending_count <= 0:
        return "clear"
    if pending_count > 0:
        return "pending"
    if stale_count >= count:
        return "stale"
    if stale_count > 0:
        return "mixed"
    return "pending"


def filename_conflict_summary(state_root: Path) -> dict[str, Any]:
    review_dir = state_root / "filename-conflict-reviews"
    review_files = sorted(review_dir.glob("filename-conflict-review-*.json"), key=lambda path: path.stat().st_mtime)
    review_files = [path for path in review_files if "audit" not in path.name]
    if not review_files:
        return {
            "id": "filename_conflicts",
            "label": "Filename conflicts",
            "count": 0,
            "status": "clear",
            "detail": "No filename-conflict review file found.",
            "path": "",
            "decisions": {},
            "total_count": 0,
        }
    path = review_files[-1]
    payload = read_json_file(path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    decisions: dict[str, int] = {}
    pending_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "pending")
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision in {"pending", "needs_investigation"}:
            pending_count += 1
    decision_text = ", ".join(f"{key}: {value}" for key, value in sorted(decisions.items())) or "no items"
    return {
        "id": "filename_conflicts",
        "label": "Filename conflicts",
        "count": pending_count,
        "status": "pending" if pending_count else "clear",
        "detail": f"Latest review has {decision_text}.",
        "path": str(path),
        "decisions": decisions,
        "total_count": len(items),
        "updated_at": payload.get("updated_at") or payload.get("created_at") or "",
    }


def document_id_for_source_path(root: Optional[Path], source_path: str) -> str:
    if not source_path:
        return ""
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT id FROM documents WHERE source_path = ? LIMIT 1", (source_path,)).fetchone()
    return str(row["id"]) if row else ""


def route_review_items(state_root: Path, *, store_root: Optional[Path] = None, limit: int = 50) -> list[dict[str, Any]]:
    review_dir = state_root / "review-queue"
    paths = sorted(review_dir.glob("*.route-review.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in paths[:limit]:
        payload = read_json_file(path)
        route_path = Path(str(payload.get("route_decision_path") or "")).expanduser()
        route_exists = bool(str(route_path)) and route_path.exists()
        route_payload = read_json_file(route_path) if route_exists else {}
        selected = route_payload.get("selected_candidate") if isinstance(route_payload.get("selected_candidate"), dict) else {}
        source_transcript_path = str(route_payload.get("source_transcript_path") or "")
        item = {
            "id": path.stem.removesuffix(".route-review"),
            "bucket": "route_reviews",
            "type": "route_review",
            "label": payload.get("selected_label") or selected.get("label") or "Unselected route",
            "reason": payload.get("reason") or "",
            "created_at": payload.get("created_at") or "",
            "review_path": str(path),
            "route_decision_path": str(route_path) if str(route_path) else "",
            "route_decision_exists": route_exists,
            "status": "pending" if route_exists else "stale_reference",
            "confidence": selected.get("confidence"),
            "target_kind": selected.get("target_kind") or "",
            "document_id": document_id_for_source_path(store_root, source_transcript_path) if store_root else "",
            "workflow_stage": "context",
        }
        items.append(item)
    return items


def speaker_review_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    paths = sorted(
        review_queue_dir(state_root).glob("*.speaker-review.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for path in paths[:limit]:
        payload = read_json_file(path)
        items.append(
            {
                "id": str(payload.get("id") or path.stem),
                "bucket": "speaker_ids",
                "type": "speaker_identity_review",
                "label": str(payload.get("speaker_label") or "Speaker identity"),
                "reason": str(payload.get("reason") or "Speaker/contact identity needs review."),
                "created_at": str(payload.get("created_at") or ""),
                "review_path": str(path),
                "status": str(payload.get("status") or "pending"),
                "document_id": str(payload.get("document_id") or payload.get("representative_document_id") or ""),
                "representative_document_id": str(payload.get("representative_document_id") or ""),
                "workflow_stage": str(payload.get("workflow_stage") or "speakers"),
                "confidence": None,
                "target_kind": "contact",
            }
        )
    return items


def deposition_memory_preview_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    paths = sorted(
        review_queue_dir(state_root).glob("*.conversation-preview-review.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for path in paths[:limit]:
        payload = read_json_file(path)
        items.append(
            {
                "id": str(payload.get("id") or path.stem),
                "bucket": "deposition_memory_preview",
                "type": "deposition_memory_preview",
                "label": str(payload.get("label") or "Deposition and memory preview"),
                "reason": str(payload.get("reason") or "Preview needs operator review before apply."),
                "created_at": str(payload.get("created_at") or ""),
                "review_path": str(path),
                "artifact_path": str(payload.get("preview_path") or ""),
                "status": str(payload.get("status") or "pending"),
                "document_id": str(payload.get("document_id") or payload.get("representative_document_id") or ""),
                "representative_document_id": str(payload.get("representative_document_id") or ""),
                "workflow_stage": str(payload.get("workflow_stage") or "output"),
                "action_count": int(payload.get("action_count") or 0),
                "memory_candidate_count": int(payload.get("memory_candidate_count") or 0),
                "confidence": None,
                "target_kind": "deposition_memory_preview",
            }
        )
    return items


def app_intelligence_human_review_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    runs = list_app_intelligence_runs(state_root=state_root, limit=500)
    items: list[dict[str, Any]] = []
    for summary in runs.get("items", []):
        if not isinstance(summary, dict):
            continue
        run_id = str(summary.get("run_id") or "")
        if not run_id:
            continue
        try:
            shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
        except (FileNotFoundError, ValueError):
            continue
        run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
        decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            if decision.get("action") != "ask_for_human_review":
                continue
            status = str(decision.get("status") or "")
            if status not in {"validated", "applied"}:
                continue
            human_review = decision.get("human_review") if isinstance(decision.get("human_review"), dict) else {}
            human_review_status = str(human_review.get("status") or "open")
            apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
            item_status = "resolved" if human_review_status == "resolved" else "needs_human_review" if status == "applied" else "pending_apply"
            items.append(
                {
                    "id": f"{run_id}:{decision.get('decision_id') or ''}",
                    "bucket": "app_intelligence_human_review",
                    "type": "app_intelligence_human_review",
                    "label": run.get("purpose") or run.get("workflow") or run_id,
                    "reason": "App Intelligence requested human review.",
                    "created_at": decision.get("applied_at") or decision.get("created_at") or run.get("updated_at") or "",
                    "run_id": run_id,
                    "document_id": run.get("document_id") or "",
                    "workflow": run.get("workflow") or "",
                    "decision_id": decision.get("decision_id") or "",
                    "decision_status": status,
                    "human_review_status": human_review_status,
                    "human_review_note_count": len(human_review.get("notes") if isinstance(human_review.get("notes"), list) else []),
                    "status": item_status,
                    "review_path": str(shown.get("path") or ""),
                    "artifact_path": apply_result.get("artifact_path") or decision.get("artifact_path") or "",
                    "confidence": None,
                    "target_kind": "app_intelligence_run",
                }
            )
    return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)[:limit]


def review_queue_summary(*, state_root: Optional[Path] = None, store_root: Optional[Path] = None, limit: int = 50) -> dict[str, Any]:
    runtime_state_root = (state_root or DEFAULT_STATE_DIR).expanduser()
    route_items = route_review_items(runtime_state_root, store_root=store_root, limit=limit)
    app_human_review_items = app_intelligence_human_review_items(runtime_state_root, limit=limit)
    speaker_items = speaker_review_items(runtime_state_root, limit=limit)
    preview_items = deposition_memory_preview_items(runtime_state_root, limit=limit)
    stale_count = sum(1 for item in route_items if not item["route_decision_exists"])
    actionable_count = len(route_items) - stale_count
    filename_bucket = filename_conflict_summary(runtime_state_root)
    legacy_payload = legacy_enrichment_queue(root=store_root, pending_only=True)
    legacy_count = int(legacy_payload.get("selected_count") or 0)
    route_bucket = {
        "id": "route_reviews",
        "label": "Route reviews",
        "count": actionable_count,
        "status": review_status(len(route_items), stale_count=stale_count),
        "detail": f"{actionable_count} actionable, {stale_count} stale local references.",
        "total_count": len(route_items),
        "stale_count": stale_count,
    }
    legacy_bucket = {
        "id": "first_pass_summaries",
        "label": "First-pass summaries",
        "count": legacy_count,
        "status": "pending" if legacy_count else "clear",
        "detail": "Stored transcripts waiting for first-pass summaries.",
        "duplicate_count": legacy_payload.get("duplicate_count") or 0,
        "sample_items": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "generated_at": item.get("generated_at"),
                "has_media_blob": item.get("has_media_blob"),
            }
            for item in legacy_payload.get("items", [])[:5]
            if isinstance(item, dict)
        ],
    }
    app_human_review_bucket = {
        "id": "app_intelligence_human_review",
        "label": "App Intelligence review",
        "count": sum(1 for item in app_human_review_items if item.get("status") != "resolved"),
        "status": "pending" if any(item.get("status") != "resolved" for item in app_human_review_items) else "clear",
        "detail": f"{sum(1 for item in app_human_review_items if item.get('status') != 'resolved')} App Intelligence human-review decisions need operator attention.",
        "pending_apply_count": sum(1 for item in app_human_review_items if item.get("status") == "pending_apply"),
        "needs_review_count": sum(1 for item in app_human_review_items if item.get("status") == "needs_human_review"),
    }
    preview_open_count = sum(1 for item in preview_items if item.get("status") not in {"resolved", "approved", "rejected"})
    memory_candidate_count = sum(int(item.get("memory_candidate_count") or 0) for item in preview_items)
    speaker_open_count = sum(1 for item in speaker_items if item.get("status") not in {"resolved", "confirmed"})
    buckets = [
        route_bucket,
        app_human_review_bucket,
        filename_bucket,
        legacy_bucket,
        {
            "id": "deposition_memory_preview",
            "label": "Deposition previews",
            "count": preview_open_count,
            "status": "pending" if preview_open_count else "clear",
            "detail": f"{preview_open_count} deposition/memory preview item(s) are queued for local human review.",
        },
        {
            "id": "memory_harvest",
            "label": "Memory harvest",
            "count": memory_candidate_count,
            "status": "gated" if memory_candidate_count else "clear",
            "detail": "Preview candidates stay review-gated; live Graphiti writes require a separate approved apply.",
        },
        {
            "id": "speaker_ids",
            "label": "Speaker IDs",
            "count": speaker_open_count,
            "status": "pending" if speaker_open_count else "clear",
            "detail": f"{speaker_open_count} speaker/contact review item(s) are queued from conversation workspaces.",
        },
    ]
    return {
        "state_dir": str(runtime_state_root),
        "store_dir": str(store_dir(store_root)),
        "limit": limit,
        "buckets": buckets,
        "items": [*preview_items, *speaker_items, *app_human_review_items, *route_items][:limit],
        "total_open": sum(int(bucket.get("count") or 0) for bucket in buckets),
    }


def default_prepare_manifest_path(state_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return state_root.expanduser() / "first-pass-summary-batches" / f"first-pass-summary-prepare-{stamp}.json"


def selected_first_pass_summary_manifest_path(state_root: Path, document_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", document_id)[:48] or "conversation"
    return state_root.expanduser() / "first-pass-summary-batches" / f"first-pass-summary-selected-{safe_doc_id}-{stamp}.json"


def resolve_batch_manifest_path(state_root: Path, path_text: str) -> Path:
    if not path_text:
        raise ValueError("Missing required manifest path.")
    root = (state_root.expanduser() / "first-pass-summary-batches").resolve()
    path = Path(path_text).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest path is outside the first-pass summary batch directory.") from exc
    if not path.exists() or not path.is_file():
        raise TranscriptStoreError(f"Manifest not found: {path}")
    return path


def batch_status_counts(batch_status: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw_counts = batch_status.get("counts")
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            if isinstance(value, int):
                counts[str(key)] = value
        if counts:
            return counts
    for job in batch_status.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def batch_action_response(
    *,
    action: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    status: str,
    batch_status: Optional[dict[str, Any]] = None,
    materialized: Optional[list[dict[str, Any]]] = None,
    materialization_errors: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    batch = manifest.get("batch") if isinstance(manifest.get("batch"), dict) else {}
    batch_payload = manifest.get("batch_payload") if isinstance(manifest.get("batch_payload"), dict) else {}
    requests = batch_payload.get("requests") if isinstance(batch_payload.get("requests"), list) else []
    first_request = requests[0] if requests and isinstance(requests[0], dict) else {}
    output_contract = (
        (first_request.get("metadata") or {}).get("outputContract")
        if isinstance(first_request.get("metadata"), dict)
        else {}
    )
    return {
        "action": action,
        "bucket": "first_pass_summaries",
        "status": status,
        "manifest": str(manifest_path),
        "request_count": int(manifest.get("request_count") or 0),
        "dry_run": bool(manifest.get("dry_run")),
        "batch_id": batch.get("id") if batch else None,
        "workflow": (batch_payload.get("metadata") or {}).get("workflow") if isinstance(batch_payload.get("metadata"), dict) else "",
        "artifact_file": output_contract.get("artifactFileName") if isinstance(output_contract, dict) else "",
        "batch_status": batch_status,
        "batch_counts": batch_status_counts(batch_status or {}),
        "materialized": materialized or [],
        "materialization_errors": materialization_errors or [],
    }


def summarize_first_pass_batch_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    batch_payload = payload.get("batch_payload") if isinstance(payload.get("batch_payload"), dict) else {}
    last_status = payload.get("last_status") if isinstance(payload.get("last_status"), dict) else {}
    materialized = payload.get("materialized") if isinstance(payload.get("materialized"), list) else []
    materialization_errors = payload.get("materialization_errors") if isinstance(payload.get("materialization_errors"), list) else []
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except OSError:
        mtime = ""
    status = str(last_status.get("status") or ("submitted" if batch else "prepared"))
    return {
        "schema_version": "transcribe-audio.first-pass-summary-batch-manifest-summary.v1",
        "manifest": str(path),
        "updated_at": mtime,
        "status": status,
        "request_count": int(payload.get("request_count") or 0),
        "dry_run": bool(payload.get("dry_run")),
        "batch_id": str(batch.get("id") or ""),
        "workflow": (batch_payload.get("metadata") or {}).get("workflow") if isinstance(batch_payload.get("metadata"), dict) else "",
        "batch_counts": batch_status_counts(last_status),
        "materialized_count": len(materialized),
        "materialization_error_count": len(materialization_errors),
        "will_read_request_payloads": False,
        "will_read_transcript_content": False,
    }


def list_first_pass_summary_batch_manifests(*, state_root: Path, limit: int = 10) -> dict[str, Any]:
    root = state_root.expanduser() / "first-pass-summary-batches"
    paths = sorted(
        [path for path in root.glob("*.json") if path.is_file()] if root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items = [summarize_first_pass_batch_manifest(path, read_json_file(path)) for path in paths[:limit]]
    return {
        "schema_version": "transcribe-audio.first-pass-summary-batch-manifests.v1",
        "manifest_dir": str(root),
        "items": items,
        "total": len(paths),
        "limit": limit,
        "will_read_request_payloads": False,
        "will_read_transcript_content": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def first_pass_summary_queue_item(detail: dict[str, Any], *, model: str, store_readouts: bool) -> dict[str, Any]:
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    if not source_document:
        raise TranscriptStoreError("No source transcript is linked to this conversation.")
    artifact_path = artifact_path_for_document(source_document)
    if artifact_path is None:
        raise TranscriptStoreError("No readable transcript artifact is available for first-pass summary preparation.")
    payload = source_document.get("json_payload") if isinstance(source_document.get("json_payload"), dict) else {}
    legacy = payload.get("legacy_import") if isinstance(payload.get("legacy_import"), dict) else {}
    source_path = str(artifact_path.expanduser().resolve())
    command = ["python", "summarize_transcript.py", source_path, "--provider", "openai-compatible"]
    if model:
        command.extend(["--model", model])
    if store_readouts:
        command.append("--store")
    return {
        "id": source_document.get("id") or "",
        "title": source_document.get("title") or "",
        "generated_at": source_document.get("generated_at") or "",
        "source_path": source_path,
        "stored_path": source_document.get("stored_path") or "",
        "legacy_source_path": str(legacy.get("source_path") or ""),
        "legacy_source_sha256": str(legacy.get("source_sha256") or ""),
        "source_media_path": str(payload.get("source_media_path") or ""),
        "has_media_blob": bool((detail.get("media_blob") or {}).get("id")),
        "readout_count": 1 if detail.get("summary_document") else 0,
        "contextual_readout_count": 1 if detail.get("contextual_readout_document") else 0,
        "pending_first_pass_readout": not bool(detail.get("summary_document")),
        "command": command,
        "participant_identity_bundle": (detail.get("identity_review") or {}).get("identity_bundle", {}),
    }


def validate_first_pass_manifest_for_conversation(
    *,
    manifest: dict[str, Any],
    detail: dict[str, Any],
) -> None:
    source_document = detail.get("transcript_document") or detail.get("selected_document") or {}
    source_id = str(source_document.get("id") or "")
    if not source_id:
        raise ValueError("Conversation has no source transcript for this first-pass summary manifest.")
    queue = manifest.get("queue") if isinstance(manifest.get("queue"), dict) else {}
    queue_items = queue.get("items") if isinstance(queue.get("items"), list) else []
    batch_payload = manifest.get("batch_payload") if isinstance(manifest.get("batch_payload"), dict) else {}
    requests = batch_payload.get("requests") if isinstance(batch_payload.get("requests"), list) else []
    queue_ids = [str(item.get("id") or "") for item in queue_items if isinstance(item, dict)]
    request_ids = [
        str((request.get("metadata") or {}).get("transcriptDocumentId") or "")
        for request in requests
        if isinstance(request, dict) and isinstance(request.get("metadata"), dict)
    ]
    scoped_ids = [value for value in [*queue_ids, *request_ids] if value]
    if scoped_ids != [source_id] * len(scoped_ids) or not scoped_ids:
        raise ValueError("Manifest is not scoped to the selected conversation source transcript.")


def prepare_selected_first_pass_summary(
    document_id: str,
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    store: bool = True,
    model: str = "",
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    args = argparse.Namespace(env_file=env_file, base_url=None, api_key=None, model=model or None, dispatch_team=None)
    env = auracall_legacy_enrichment_batch.runtime_env(args)
    dispatch_team = auracall_legacy_enrichment_batch.resolve_dispatch_team(args, env)
    resolved_model = auracall_legacy_enrichment_batch.resolve_model(args, env, dispatch_team)
    detail = get_conversation_detail(document_id, root=store_root, state_root=state_root)
    item = first_pass_summary_queue_item(detail, model=resolved_model, store_readouts=store)
    request_payload = auracall_legacy_enrichment_batch.create_request(item, resolved_model, dispatch_team)
    batch_payload = {
        "metadata": {
            "workflow": "transcribe-audio-first-pass-summary",
            "createdAt": auracall_legacy_enrichment_batch.utc_now_iso(),
            "model": resolved_model,
            "dispatchTeam": dispatch_team,
            "storeDir": str(store_dir(store_root)),
            "selectedCount": 1,
            "duplicateCount": 0,
            "scopedDocumentId": item["id"],
            "conversationKey": detail["conversation"]["key"],
        },
        "limits": {
            "maxConcurrentRuns": auracall_legacy_enrichment_batch.DEFAULT_MAX_CONCURRENT_RUNS,
            "maxBrowserInteractionsPerMinute": auracall_legacy_enrichment_batch.DEFAULT_MAX_BROWSER_INTERACTIONS_PER_MINUTE,
        },
        "requests": [request_payload],
    }
    if dispatch_team:
        batch_payload["dispatch"] = {
            "team": dispatch_team,
            "mode": "next_available",
            "projectSync": "none",
        }
    manifest = {
        "object": "transcribe_audio_auracall_batch_manifest",
        "created_at": auracall_legacy_enrichment_batch.utc_now_iso(),
        "model": resolved_model,
        "dispatch_team": dispatch_team,
        "dry_run": True,
        "store": bool(store),
        "batch_url": auracall_legacy_enrichment_batch.resolve_batch_url(args, env),
        "response_base_url": auracall_legacy_enrichment_batch.resolve_base_url(args, env),
        "queue": {
            "store_dir": str(store_dir(store_root)),
            "pending_only": False,
            "dedupe": False,
            "duplicate_count": 0,
            "selected_count": 1,
            "items": [item],
        },
        "request_count": 1,
        "batch_payload": batch_payload,
        "batch": None,
        "document_id": document_id,
        "conversation_key": detail["conversation"]["key"],
    }
    manifest_path = selected_first_pass_summary_manifest_path(state_root, document_id)
    write_json_file(manifest_path, manifest)
    return {
        **batch_action_response(
            action="prepare_selected_first_pass_summary",
            manifest_path=manifest_path,
            manifest=manifest,
            status="prepared",
        ),
        "first_pass_summary": first_pass_summary_state(detail),
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }


def submit_selected_first_pass_summary(
    document_id: str,
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    manifest: str,
    approval_token: str,
) -> dict[str, Any]:
    detail = get_conversation_detail(document_id, root=store_root, state_root=state_root)
    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    validate_first_pass_manifest_for_conversation(manifest=payload, detail=detail)
    submitted = submit_first_pass_summary_batch(
        state_root=state_root,
        env_file=env_file,
        manifest=str(manifest_path),
        approval_token=approval_token,
    )
    return {
        **submitted,
        "first_pass_summary": first_pass_summary_state(detail),
    }


def run_selected_first_pass_summary(
    document_id: str,
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    approval_token: str,
    store: bool = True,
    model: str = "",
) -> dict[str, Any]:
    if approval_token != FIRST_PASS_SUMMARY_SUBMIT_TOKEN:
        raise ValueError(f"Run requires approval_token={FIRST_PASS_SUMMARY_SUBMIT_TOKEN}.")
    prepared = prepare_selected_first_pass_summary(
        document_id,
        state_root=state_root,
        store_root=store_root,
        env_file=env_file,
        store=store,
        model=model,
    )
    submitted = submit_selected_first_pass_summary(
        document_id,
        state_root=state_root,
        store_root=store_root,
        env_file=env_file,
        manifest=str(prepared["manifest"]),
        approval_token=approval_token,
    )
    return {
        **submitted,
        "action": "run_selected_first_pass_summary",
        "prepared": {
            "action": prepared.get("action"),
            "status": prepared.get("status"),
            "manifest": prepared.get("manifest"),
            "request_count": prepared.get("request_count"),
        },
        "one_click": True,
        "will_execute_external_action": True,
        "will_perform_external_write": True,
    }


def selected_first_pass_summary_status(
    document_id: str,
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    manifest: str,
    materialize: bool = False,
) -> dict[str, Any]:
    detail = get_conversation_detail(document_id, root=store_root, state_root=state_root)
    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    validate_first_pass_manifest_for_conversation(manifest=payload, detail=detail)
    status = first_pass_summary_batch_status(
        state_root=state_root,
        store_root=store_root,
        env_file=env_file,
        manifest=str(manifest_path),
        materialize=materialize,
    )
    refreshed = get_conversation_detail(document_id, root=store_root, state_root=state_root)
    return {
        **status,
        "first_pass_summary": first_pass_summary_state(refreshed),
    }


def prepare_first_pass_summary_batch(
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    limit: int = 5,
    store: bool = True,
    model: str = "",
    manifest: Optional[Path] = None,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    manifest_path = manifest or default_prepare_manifest_path(state_root)
    args = argparse.Namespace(
        command="prepare",
        env_file=env_file,
        base_url=None,
        api_key=None,
        store_dir=store_root,
        limit=limit,
        model=model or None,
        dispatch_team=None,
        all=False,
        no_dedupe=False,
        manifest=manifest_path,
        store=store,
        max_concurrent_runs=auracall_legacy_enrichment_batch.DEFAULT_MAX_CONCURRENT_RUNS,
        max_browser_interactions_per_minute=auracall_legacy_enrichment_batch.DEFAULT_MAX_BROWSER_INTERACTIONS_PER_MINUTE,
    )
    exit_code = auracall_legacy_enrichment_batch.enqueue(args, force_dry_run=True)
    if exit_code != 0:
        raise TranscriptStoreError(f"First-pass summary prepare failed with exit code {exit_code}")
    payload = read_json_file(manifest_path)
    return batch_action_response(
        action="prepare_first_pass_summary_batch",
        manifest_path=manifest_path.expanduser(),
        manifest=payload,
        status="prepared",
    )


def submit_first_pass_summary_batch(
    *,
    state_root: Path,
    env_file: Path,
    manifest: str,
    approval_token: str,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    if approval_token != FIRST_PASS_SUMMARY_SUBMIT_TOKEN:
        raise ValueError(f"Submit requires approval_token={FIRST_PASS_SUMMARY_SUBMIT_TOKEN}.")
    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    if payload.get("batch"):
        return batch_action_response(
            action="submit_first_pass_summary_batch",
            manifest_path=manifest_path,
            manifest=payload,
            status="already_submitted",
        )
    batch_payload = payload.get("batch_payload")
    if not isinstance(batch_payload, dict):
        raise ValueError("Manifest does not include a prepared batch payload.")
    if int(payload.get("request_count") or 0) <= 0:
        raise ValueError("Manifest has no prepared requests to submit.")
    args = argparse.Namespace(env_file=env_file, base_url=None, api_key=None)
    env = auracall_legacy_enrichment_batch.runtime_env(args)
    payload["batch"] = auracall_legacy_enrichment_batch.post_batch(
        str(payload["batch_url"]),
        auracall_legacy_enrichment_batch.resolve_api_key(args, env),
        batch_payload,
    )
    payload["dry_run"] = False
    payload["submitted_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_json_file(manifest_path, payload)
    return batch_action_response(
        action="submit_first_pass_summary_batch",
        manifest_path=manifest_path,
        manifest=payload,
        status="submitted",
    )


def first_pass_summary_batch_status(
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    manifest: str,
    materialize: bool = False,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    batch_id = str(batch.get("id") or "")
    if not batch_id:
        return batch_action_response(
            action="first_pass_summary_batch_status",
            manifest_path=manifest_path,
            manifest=payload,
            status="prepared",
        )
    args = argparse.Namespace(env_file=env_file, base_url=None, api_key=None, store=bool(payload.get("store")), store_dir=store_root, output_dir=None)
    env = auracall_legacy_enrichment_batch.runtime_env(args)
    batch_url = f"{str(payload['batch_url']).rstrip('/')}/{batch_id}"
    batch_status = auracall_legacy_enrichment_batch.read_json_url(
        batch_url,
        auracall_legacy_enrichment_batch.resolve_api_key(args, env),
    )
    materialized: list[dict[str, Any]] = []
    materialization_errors: list[dict[str, Any]] = []
    if materialize:
        materialized, materialization_errors = auracall_legacy_enrichment_batch.materialize_completed(args, payload, batch_status)
    payload["last_status"] = batch_status
    if materialized:
        payload["materialized"] = materialized
    if materialization_errors:
        payload["materialization_errors"] = materialization_errors
    write_json_file(manifest_path, payload)
    return batch_action_response(
        action="first_pass_summary_batch_status",
        manifest_path=manifest_path,
        manifest=payload,
        status=str(batch_status.get("status") or "unknown"),
        batch_status=batch_status,
        materialized=materialized,
        materialization_errors=materialization_errors,
    )


def parse_range_header(header: str, size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes="):
        return None
    value = header.removeprefix("bytes=").split(",", 1)[0].strip()
    if not value or "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    if start_text == "":
        suffix = parse_int(end_text, 0, minimum=0, maximum=size)
        if suffix <= 0:
            return None
        return max(size - suffix, 0), size - 1
    start = parse_int(start_text, 0, minimum=0, maximum=max(size - 1, 0))
    end = parse_int(end_text, size - 1, minimum=start, maximum=max(size - 1, 0)) if end_text else size - 1
    if start > end:
        return None
    return start, end


class TranscriptApiHandler(BaseHTTPRequestHandler):
    server_version = "TranscriptApi/0.1"

    @property
    def store_root(self) -> Path:
        return self.server.store_root  # type: ignore[attr-defined]

    @property
    def embedding_provider(self) -> str:
        return self.server.embedding_provider  # type: ignore[attr-defined]

    @property
    def embedding_model(self) -> str:
        return self.server.embedding_model  # type: ignore[attr-defined]

    @property
    def state_root(self) -> Path:
        return self.server.state_root  # type: ignore[attr-defined]

    def local_base_url(self) -> str:
        port = self.server.server_address[1]  # type: ignore[attr-defined]
        return f"http://127.0.0.1:{port}"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self.write_json({"status": "ok", "store_dir": str(self.store_root), "db_path": str(db_path(self.store_root))})
                return
            if parsed.path == "/api/library":
                self.write_json(
                    list_documents(
                        root=self.store_root,
                        kind=first(params, "kind"),
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                        offset=parse_int(first(params, "offset"), 0, minimum=0, maximum=100000),
                    )
                )
                return
            if parsed.path == "/api/conversations":
                self.write_json(
                    list_conversations(
                        root=self.store_root,
                        kind=first(params, "kind"),
                        query=first(params, "q") or first(params, "query"),
                        limit=parse_int(first(params, "limit"), 100, minimum=1, maximum=500),
                        offset=parse_int(first(params, "offset"), 0, minimum=0, maximum=100000),
                    )
                )
                return
            if parsed.path.startswith("/api/conversations/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 4 and parts[3] == "identity-review":
                    self.write_json(
                        get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root)["identity_review"]
                    )
                    return
                if len(parts) == 4 and parts[3] == "first-pass-summary":
                    self.write_json(
                        get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root)["first_pass_summary"]
                    )
                    return
                if len(parts) == 4 and parts[3] == "context-workbench":
                    self.write_json(
                        get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root)["context_workbench"]
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-search":
                    self.write_json(
                        search_context_contacts(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            query=first(params, "q") or first(params, "query"),
                            limit=parse_int(first(params, "limit"), 20, minimum=1, maximum=100),
                            mode=first(params, "mode") or "cache",
                            source_filters=params.get("source") or params.get("sources") or [],
                        )
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-refresh":
                    self.write_json(
                        context_contact_refresh_preview(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            query=first(params, "q") or first(params, "query"),
                            source_filters=params.get("source") or params.get("sources") or [],
                        )
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-affinity":
                    query = first(params, "q") or first(params, "query")
                    detail = get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root)
                    conversation_key = str((detail.get("conversation") or {}).get("key") or "")
                    cache = context_contact_affinity_cache_state(state_root=self.state_root, conversation_key=conversation_key)
                    items = cache.get("items") if isinstance(cache.get("items"), list) else []
                    if query:
                        terms = [term for term in re.split(r"\s+", query.lower()) if term]
                        items = [
                            item for item in items
                            if isinstance(item, dict)
                            and all(term in context_contact_candidate_search_text(item) for term in terms)
                        ]
                    self.write_json(
                        {
                            "schema_version": "transcribe-audio.context-contact-affinity.v1",
                            "status": cache.get("status", "empty"),
                            "query": query,
                            "items": items,
                            "total": len(items),
                            "cache_path": cache.get("path", ""),
                            "will_execute_external_action": False,
                            "will_perform_external_write": False,
                        }
                    )
                    return
                if len(parts) == 6 and parts[3] == "context-workbench" and parts[4] == "contact-refresh":
                    self.write_json(read_context_contact_refresh_job(state_root=self.state_root, job_id=parts[5]))
                    return
                if len(parts) == 4 and parts[3] == "final-preview":
                    self.write_json(
                        get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root)["final_preview"]
                    )
                    return
                if len(parts) == 3:
                    self.write_json(get_conversation_detail(parts[2], root=self.store_root, state_root=self.state_root))
                    return
            if parsed.path == "/api/review-queue":
                self.write_json(
                    review_queue_summary(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                    )
                )
                return
            if parsed.path == "/api/intelligence/providers":
                self.write_json(intelligence_provider_registry(codex_bin=self.server.codex_bin))  # type: ignore[attr-defined]
                return
            if parsed.path == "/api/intelligence/smokes":
                self.write_json(
                    app_intelligence_smoke_status(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 5, minimum=1, maximum=50),
                    )
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/manifests":
                self.write_json(
                    list_first_pass_summary_batch_manifests(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                    )
                )
                return
            if parsed.path == "/api/intelligence/smoke-jobs":
                self.write_json(
                    list_app_smoke_jobs(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                    )
                )
                return
            if parsed.path == "/api/intelligence/smoke-evidence":
                self.write_smoke_evidence(first(params, "path"))
                return
            if parsed.path.startswith("/api/intelligence/smoke-jobs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[4] == "tail":
                    self.write_json(
                        read_app_smoke_job_tail(
                            state_root=self.state_root,
                            job_id=parts[3],
                            stream=first(params, "stream", "stderr"),
                            chars=parse_int(
                                first(params, "chars"),
                                4000,
                                minimum=1,
                                maximum=MAX_SMOKE_JOB_TAIL_CHARS,
                            ),
                        )
                    )
                    return
            if parsed.path == "/api/intelligence/config":
                self.write_json(intelligence_config.all_task_configs())
                return
            if parsed.path == "/api/automation/config":
                self.write_json(automation_config.all_config(state_root=self.state_root))
                return
            if parsed.path == "/api/provenance/config":
                self.write_json(
                    provenance_config.all_config(
                        state_root=self.state_root,
                        profile=first(params, "profile"),
                    )
                )
                return
            if parsed.path == "/api/provenance/config/doctor":
                self.write_json(
                    provenance_config.doctor(
                        state_root=self.state_root,
                        profile=first(params, "profile"),
                    )
                )
                return
            if parsed.path == "/api/intelligence/runs":
                self.write_json(
                    list_app_intelligence_runs(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                    )
                )
                return
            if parsed.path.startswith("/api/intelligence/runs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 6 and parts[4] == "prompt-packets":
                    self.write_json(
                        read_app_intelligence_model_turn_packet(
                            state_root=self.state_root,
                            run_id=parts[3],
                            packet_id=parts[5],
                        )
                    )
                    return
                if len(parts) == 5 and parts[4] == "replay-manifest":
                    self.write_json(
                        app_intelligence_replay_manifest(
                            state_root=self.state_root,
                            run_id=parts[3],
                        )
                    )
                    return
                if len(parts) == 5 and parts[4] == "artifacts":
                    self.write_json(
                        read_app_intelligence_artifact(
                            state_root=self.state_root,
                            run_id=parts[3],
                            artifact_path=first(params, "path"),
                        )
                    )
                    return
                if len(parts) == 4:
                    self.write_json(
                        get_app_intelligence_run(
                            state_root=self.state_root,
                            run_id=parts[3],
                            event_limit=parse_int(first(params, "event_limit"), 50, minimum=0, maximum=500),
                        )
                    )
                    return
            if parsed.path == "/api/search":
                query = first(params, "q") or first(params, "query")
                if not query:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required query parameter: q")
                    return
                self.write_json(
                    {
                        "query": query,
                        "results": search_store(
                            query,
                            root=self.store_root,
                            kind=first(params, "kind"),
                            limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        ),
                    }
                )
                return
            if parsed.path.startswith("/api/documents/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_json(get_document(parts[2], root=self.store_root))
                    return
                if len(parts) == 4 and parts[3] == "related":
                    self.write_json(get_related_documents(parts[2], root=self.store_root))
                    return
                if len(parts) == 4 and parts[3] == "context":
                    chunk_text = first(params, "chunk_index")
                    self.write_json(
                        context_for_document(
                            parts[2],
                            root=self.store_root,
                            chunk_index=int(chunk_text) if chunk_text else None,
                            context_chunks=parse_int(first(params, "context_chunks"), 1, minimum=0, maximum=10),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        )
                    )
                    return
            if parsed.path.startswith("/api/blobs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_blob(parts[2], download=first(params, "download") in {"1", "true", "yes"})
                    return
            if parsed.path.startswith("/api/"):
                self.write_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if self.write_static(parsed.path):
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except TranscriptStoreError as exc:
            self.write_error(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self.write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/conversations/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 4 and parts[3] == "identity-review":
                    body = self.read_json_body()
                    self.write_json(
                        record_speaker_identity_review(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            speaker_label=str(body.get("speaker_label") or ""),
                            action=str(body.get("action") or ""),
                            contact_label=str(body.get("contact_label") or ""),
                            contact_id=str(body.get("contact_id") or ""),
                            email=str(body.get("email") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "first-pass-summary" and parts[4] == "prepare":
                    body = self.read_json_body()
                    self.write_json(
                        prepare_selected_first_pass_summary(
                            parts[2],
                            state_root=self.state_root,
                            store_root=self.store_root,
                            env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                            store=bool(body.get("store", True)),
                            model=str(body.get("model") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "first-pass-summary" and parts[4] == "submit":
                    body = self.read_json_body()
                    self.write_json(
                        submit_selected_first_pass_summary(
                            parts[2],
                            state_root=self.state_root,
                            store_root=self.store_root,
                            env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                            manifest=str(body.get("manifest") or ""),
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "first-pass-summary" and parts[4] == "run":
                    body = self.read_json_body()
                    self.write_json(
                        run_selected_first_pass_summary(
                            parts[2],
                            state_root=self.state_root,
                            store_root=self.store_root,
                            env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                            approval_token=str(body.get("approval_token") or ""),
                            store=bool(body.get("store", True)),
                            model=str(body.get("model") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "first-pass-summary" and parts[4] == "status":
                    body = self.read_json_body()
                    self.write_json(
                        selected_first_pass_summary_status(
                            parts[2],
                            state_root=self.state_root,
                            store_root=self.store_root,
                            env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                            manifest=str(body.get("manifest") or ""),
                            materialize=bool(body.get("materialize", False)),
                        )
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] in {"preview", "queue"}:
                    body = self.read_json_body()
                    self.write_json(
                        context_workbench_preview(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            queue=parts[4] == "queue",
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-selection":
                    body = self.read_json_body()
                    self.write_json(
                        record_context_contact_selection(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            candidate_id=str(body.get("candidate_id") or ""),
                            action=str(body.get("action") or ""),
                            actor_type=str(body.get("actor_type") or "operator"),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                            manual_candidate=body.get("manual_candidate") if isinstance(body.get("manual_candidate"), dict) else None,
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-selection-batch":
                    body = self.read_json_body()
                    self.write_json(
                        record_context_contact_selection_batch(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            actions=body.get("actions") if isinstance(body.get("actions"), list) else [],
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-refresh":
                    body = self.read_json_body()
                    self.write_json(
                        refresh_context_contacts(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            query=str(body.get("query") or body.get("q") or ""),
                            limit=parse_int(str(body.get("limit") or ""), 30, minimum=1, maximum=100),
                            source_filters=body.get("sources") if isinstance(body.get("sources"), list) else [],
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 6 and parts[3] == "context-workbench" and parts[4] == "contact-refresh" and parts[5] == "preview":
                    body = self.read_json_body()
                    self.write_json(
                        context_contact_refresh_preview(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            query=str(body.get("query") or body.get("q") or ""),
                            source_filters=body.get("sources") if isinstance(body.get("sources"), list) else [],
                        )
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "contact-merge-batch":
                    body = self.read_json_body()
                    self.write_json(
                        record_context_contact_merge_batch(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            actions=body.get("actions") if isinstance(body.get("actions"), list) else [],
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 6 and parts[3] == "context-workbench" and parts[4] == "contact-affinity" and parts[5] == "refresh":
                    body = self.read_json_body()
                    self.write_json(
                        refresh_context_contact_affinity(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            query=str(body.get("query") or body.get("q") or ""),
                            limit=parse_int(str(body.get("limit") or ""), 100, minimum=1, maximum=200),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "context-workbench" and parts[4] == "instructions":
                    body = self.read_json_body()
                    self.write_json(
                        record_context_instructions(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            instruction_text=str(body.get("instruction_text") or body.get("text") or ""),
                            actor_type=str(body.get("actor_type") or "operator"),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
                if len(parts) == 5 and parts[3] == "final-preview" and parts[4] == "queue":
                    body = self.read_json_body()
                    self.write_json(
                        queue_deposition_memory_preview(
                            parts[2],
                            root=self.store_root,
                            state_root=self.state_root,
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
            if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/retranscription/preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[3] == "retranscription" and parts[4] == "preflight":
                    body = self.read_json_body()
                    preflight = retranscription_preflight(
                        parts[2],
                        root=self.store_root,
                        state_root=self.state_root,
                        backend=str(body.get("backend") or "faster_whisper"),
                        output_dir=str(body.get("output_dir") or ""),
                    )
                    self.write_json(preflight, status=HTTPStatus.OK if preflight.get("ok") else HTTPStatus.CONFLICT)
                    return
            if parsed.path.startswith("/api/documents/") and parsed.path.endswith("/retranscription/queue"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[3] == "retranscription" and parts[4] == "queue":
                    body = self.read_json_body()
                    queued = enqueue_retranscription_job(
                        parts[2],
                        root=self.store_root,
                        state_root=self.state_root,
                        backend=str(body.get("backend") or "faster_whisper"),
                        output_dir=str(body.get("output_dir") or ""),
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    self.write_json(queued, status=HTTPStatus.CREATED if queued.get("ok") else HTTPStatus.CONFLICT)
                    return
            if parsed.path == "/api/review-queue/first-pass-summaries/prepare":
                body = self.read_json_body()
                limit = parse_int(str(body.get("limit") or ""), 5, minimum=1, maximum=50)
                store = bool(body.get("store", True))
                model = str(body.get("model") or "")
                self.write_json(
                    prepare_first_pass_summary_batch(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        limit=limit,
                        store=store,
                        model=model,
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/submit":
                body = self.read_json_body()
                self.write_json(
                    submit_first_pass_summary_batch(
                        state_root=self.state_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        manifest=str(body.get("manifest") or ""),
                        approval_token=str(body.get("approval_token") or ""),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/status":
                body = self.read_json_body()
                self.write_json(
                    first_pass_summary_batch_status(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        manifest=str(body.get("manifest") or ""),
                        materialize=bool(body.get("materialize", False)),
                    )
                )
                return
            if parsed.path == "/api/intelligence/config/preview":
                body = self.read_json_body()
                self.write_json(
                    intelligence_config.preview_config_update(
                        task=str(body.get("task") or ""),
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                    )
                )
                return
            if parsed.path == "/api/intelligence/config/apply":
                body = self.read_json_body()
                self.write_json(
                    intelligence_config.apply_config_update(
                        task=str(body.get("task") or ""),
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        approval_token=str(body.get("approval_token") or ""),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/automation/config/preview":
                body = self.read_json_body()
                self.write_json(
                    automation_config.preview_config_update(
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        state_root=self.state_root,
                    )
                )
                return
            if parsed.path == "/api/automation/config/apply":
                body = self.read_json_body()
                self.write_json(
                    automation_config.apply_config_update(
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        approval_token=str(body.get("approval_token") or ""),
                        state_root=self.state_root,
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/provenance/config/preview":
                body = self.read_json_body()
                self.write_json(
                    provenance_config.preview_config_update(
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        state_root=self.state_root,
                    )
                )
                return
            if parsed.path == "/api/provenance/config/apply":
                body = self.read_json_body()
                self.write_json(
                    provenance_config.apply_config_update(
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        approval_token=str(body.get("approval_token") or ""),
                        state_root=self.state_root,
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/intelligence/smoke-jobs":
                body = self.read_json_body()
                self.write_json(
                    enqueue_app_smoke_job(
                        state_root=self.state_root,
                        job_type=str(body.get("job_type") or ""),
                        approval_token=str(body.get("approval_token") or ""),
                        base_url=str(body.get("base_url") or self.local_base_url()),
                        cleanup=bool(body.get("cleanup", True)),
                        apply_cleanup=bool(body.get("apply_cleanup", False)),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/intelligence/runs/prepare":
                body = self.read_json_body()
                workflow = str(body.get("workflow") or "").strip()
                purpose = str(body.get("purpose") or "").strip()
                if not workflow:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required field: workflow")
                    return
                if not purpose:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required field: purpose")
                    return
                self.write_json(
                    # Explicit API provider wins; otherwise use task routing.
                    create_app_intelligence_run(
                        state_root=self.state_root,
                        workflow=workflow,
                        purpose=purpose,
                        document_id=str(body.get("document_id") or ""),
                        provider=str(
                            body.get("provider")
                            or intelligence_config.resolve_task_config(
                                str(body.get("task") or intelligence_config.TASK_APP_SUPERVISOR)
                            ).provider
                        ),
                        created_by=str(body.get("created_by") or "operator"),
                        run_id=str(body.get("run_id") or ""),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/send-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "prompt-packets":
                    body = self.read_json_body()
                    preflight = preflight_app_intelligence_model_turn_send(
                        state_root=self.state_root,
                        run_id=parts[3],
                        packet_id=parts[5],
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    self.write_json(preflight, status=HTTPStatus.OK if preflight.get("ok") else HTTPStatus.CONFLICT)
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/send"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "prompt-packets":
                    body = self.read_json_body()
                    run_id = parts[3]
                    packet_id = parts[5]
                    preflight = preflight_app_intelligence_model_turn_send(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    if not preflight.get("ok"):
                        self.write_json(preflight, status=HTTPStatus.CONFLICT)
                        return
                    packet_review = read_app_intelligence_model_turn_packet(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                    )
                    current = get_app_intelligence_run(state_root=self.state_root, run_id=run_id, event_limit=1)
                    current_run = current.get("run") if isinstance(current.get("run"), dict) else {}
                    current_state = current_run.get("state") if isinstance(current_run.get("state"), dict) else {}
                    route = packet_review.get("packet", {}).get("route") if isinstance(packet_review.get("packet"), dict) else {}
                    model = str(route.get("model") or "") if isinstance(route, dict) else ""
                    try:
                        app_server_result = codex_app_server_client.start_model_turn(
                            codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                            cwd=Path(__file__).resolve().parent,
                            prompt_text=str(packet_review.get("prompt_text") or ""),
                            model=model,
                            existing_thread_id=str(current_state.get("active_codex_thread_id") or ""),
                            timeout_seconds=float(body.get("timeout_seconds") or 30),
                        )
                    except Exception as exc:
                        event = record_app_intelligence_model_turn_failed(
                            state_root=self.state_root,
                            run_id=run_id,
                            packet_id=packet_id,
                            error=str(exc),
                        )
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-model-turn-send.v1",
                                "action": "send_model_turn",
                                "ok": False,
                                "preflight": preflight,
                                "event": event,
                                "will_execute_downstream_action": False,
                                "error": str(exc),
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    for codex_event in app_server_result.get("events", []):
                        if isinstance(codex_event, dict):
                            append_app_intelligence_codex_event(state_root=self.state_root, run_id=run_id, payload=codex_event)
                    started = record_app_intelligence_model_turn_started(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                        thread_id=str(app_server_result.get("thread_id") or ""),
                        turn_id=str(app_server_result.get("turn_id") or ""),
                        app_server_result={
                            "thread_start_response": app_server_result.get("thread_start_response") or {},
                            "turn_start_response": app_server_result.get("turn_start_response") or {},
                            "captured_event_count": len(app_server_result.get("events") or []),
                        },
                    )
                    self.write_json(
                        {
                            "schema_version": "transcribe-audio.app-intelligence-model-turn-send.v1",
                            "action": "send_model_turn",
                            "ok": True,
                            "packet_id": packet_id,
                            "codex_thread_id": app_server_result.get("thread_id") or "",
                            "codex_turn_id": app_server_result.get("turn_id") or "",
                            "captured_event_count": len(app_server_result.get("events") or []),
                            "preflight": preflight,
                            "will_execute_downstream_action": False,
                            **started,
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/turn-status"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    run_id = parts[3]
                    current = get_app_intelligence_run(state_root=self.state_root, run_id=run_id, event_limit=1)
                    run = current.get("run") if isinstance(current.get("run"), dict) else {}
                    state = run.get("state") if isinstance(run.get("state"), dict) else {}
                    thread_id = str(body.get("thread_id") or state.get("active_codex_thread_id") or "")
                    turn_id = str(body.get("turn_id") or state.get("latest_turn_id") or "")
                    if not thread_id or not turn_id:
                        self.write_error(HTTPStatus.BAD_REQUEST, "No active Codex thread/turn is recorded for this run.")
                        return
                    try:
                        status_result = codex_app_server_client.inspect_model_turn(
                            codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                            thread_id=thread_id,
                            turn_id=turn_id,
                            timeout_seconds=float(body.get("timeout_seconds") or 30),
                        )
                    except Exception as exc:
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-model-turn-status.v1",
                                "action": "capture_model_turn_status",
                                "ok": False,
                                "codex_thread_id": thread_id,
                                "codex_turn_id": turn_id,
                                "will_execute_structured_decision": False,
                                "error": str(exc),
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    for codex_event in status_result.get("events", []):
                        if isinstance(codex_event, dict):
                            append_app_intelligence_codex_event(state_root=self.state_root, run_id=run_id, payload=codex_event)
                    captured = record_app_intelligence_model_turn_status(
                        state_root=self.state_root,
                        run_id=run_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status_payload=status_result,
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    self.write_json(
                        {
                            **captured,
                            "codex_thread_id": thread_id,
                            "codex_turn_id": turn_id,
                            "captured_event_count": len(status_result.get("events") or []),
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/structured-decision/validate"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 6:
                    body = self.read_json_body()
                    self.write_json(
                        validate_app_intelligence_structured_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/apply"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        apply_app_intelligence_structured_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/human-review"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        record_app_intelligence_human_review_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            review_action=str(body.get("review_action") or ""),
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/fork-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        preflight_app_intelligence_fork_branches(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/rollback-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        preflight_app_intelligence_rollback(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/session-start-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    provider = codex_app_server_readiness(codex_bin=self.server.codex_bin)  # type: ignore[attr-defined]
                    append_event_log = bool(body.get("append_event", False))
                    self.write_json(
                        preflight_app_intelligence_session_start(
                            state_root=self.state_root,
                            run_id=parts[3],
                            provider_ready=bool(provider.get("ready")),
                            provider_status=str(provider.get("status") or ""),
                            approval_token=str(body.get("approval_token") or ""),
                            append_event_log=append_event_log,
                        ),
                        status=HTTPStatus.ACCEPTED if append_event_log else HTTPStatus.OK,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/session-start"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    transport = str(body.get("transport") or "stdio")
                    approval_token = str(body.get("approval_token") or "")
                    provider = codex_app_server_readiness(codex_bin=self.server.codex_bin)  # type: ignore[attr-defined]
                    preflight = preflight_app_intelligence_session_start(
                        state_root=self.state_root,
                        run_id=parts[3],
                        provider_ready=bool(provider.get("ready")),
                        provider_status=str(provider.get("status") or ""),
                        approval_token=approval_token,
                    )
                    if not preflight.get("ok"):
                        self.write_json(preflight, status=HTTPStatus.CONFLICT)
                        return
                    record_app_intelligence_session_start_requested(
                        state_root=self.state_root,
                        run_id=parts[3],
                        transport=transport,
                        approval_token=approval_token,
                    )
                    start_result = run_codex_command(
                        [self.server.codex_bin, "app-server", "daemon", "start"],  # type: ignore[attr-defined]
                        timeout=30,
                    )
                    if not start_result.get("ok"):
                        event = record_app_intelligence_session_start_failed(
                            state_root=self.state_root,
                            run_id=parts[3],
                            transport=transport,
                            error=str(start_result.get("stderr") or start_result.get("stdout") or "unknown start failure"),
                        )
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-session-start.v1",
                                "action": "start_app_server_session",
                                "ok": False,
                                "will_start_model_turn": False,
                                "preflight": preflight,
                                "start_result": start_result,
                                "event": event,
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    version_result = run_codex_command(
                        [self.server.codex_bin, "app-server", "daemon", "version"],  # type: ignore[attr-defined]
                        timeout=15,
                    )
                    started = mark_app_intelligence_session_started(
                        state_root=self.state_root,
                        run_id=parts[3],
                        transport=transport,
                        codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                        start_result=start_result,
                        version_result=version_result,
                    )
                    self.write_json(
                        {
                            "schema_version": "transcribe-audio.app-intelligence-session-start.v1",
                            "action": "start_app_server_session",
                            "ok": True,
                            "will_start_model_turn": False,
                            "transport": transport,
                            "preflight": preflight,
                            "start_result": start_result,
                            "version_result": version_result,
                            **started,
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/model-turn-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    shown = get_app_intelligence_run(state_root=self.state_root, run_id=parts[3], event_limit=1)
                    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
                    task = str(body.get("task") or run.get("workflow") or intelligence_config.TASK_APP_SUPERVISOR)
                    document_id = str(body.get("document_id") or run.get("document_id") or "")
                    if not document_id:
                        self.write_error(HTTPStatus.BAD_REQUEST, "Missing required document_id for model-turn preflight")
                        return
                    document = compact_document_for_prompt(get_document(document_id, root=self.store_root))
                    route = intelligence_config.resolve_task_config(task).to_dict()
                    prompt_text = model_turn_prompt_text(task=task, route=route, document=document)
                    self.write_json(
                        prepare_app_intelligence_model_turn_packet(
                            state_root=self.state_root,
                            run_id=parts[3],
                            task=task,
                            route=route,
                            document=document,
                            prompt_text=prompt_text,
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
            if parsed.path.startswith("/api/"):
                self.write_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except (TranscriptStoreError, TranscriptionError, OSError, json.JSONDecodeError) as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def read_json_body(self) -> dict[str, Any]:
        length = parse_int(self.headers.get("Content-Length", "0"), 0, minimum=0, maximum=1024 * 1024)
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request JSON body must be an object.")
        return payload

    def write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, status: HTTPStatus, message: str) -> None:
        self.write_json({"error": message, "status": status.value}, status=status)

    def write_smoke_evidence(self, path_value: str) -> None:
        path = resolve_smoke_evidence_path(state_root=self.state_root, path_value=path_value)
        body = path.read_bytes()
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_static(self, request_path: str) -> bool:
        static_dir = self.server.static_dir  # type: ignore[attr-defined]
        if static_dir is None:
            return False
        static_root = Path(static_dir)
        if not static_root.exists() or not static_root.is_dir():
            return False
        relative = unquote(request_path).lstrip("/")
        target = static_root / relative if relative else static_root / "index.html"
        if not target.exists() or target.is_dir():
            target = static_root / "index.html"
        try:
            resolved = target.resolve()
            resolved.relative_to(static_root.resolve())
        except ValueError:
            self.write_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return True
        if not resolved.exists() or not resolved.is_file():
            return False
        body = resolved.read_bytes()
        mime_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-store" if resolved.name == "index.html" else "public, max-age=3600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def write_blob(self, blob_id: str, *, download: bool = False) -> None:
        blob = get_blob(blob_id, root=self.store_root)
        size = int(blob["bytes"])
        file_range = parse_range_header(self.headers.get("Range", ""), size)
        status = HTTPStatus.PARTIAL_CONTENT if file_range else HTTPStatus.OK
        start, end = file_range if file_range else (0, max(size - 1, 0))
        length = max(end - start + 1, 0)

        self.send_response(status)
        self.send_header("Content-Type", str(blob["mime_type"]))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if file_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{blob_id}"')
        self.end_headers()

        with Path(blob["path"]).open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class TranscriptApiServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store_root: Path,
        embedding_provider: str,
        embedding_model: str,
        state_root: Path = DEFAULT_STATE_DIR,
        batch_env_file: Path = DEFAULT_BATCH_ENV_FILE,
        codex_bin: str = DEFAULT_CODEX_BIN,
        quiet: bool = False,
        static_dir: Optional[Path] = DEFAULT_STATIC_DIR,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store_root = store_root
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.state_root = state_root.expanduser()
        self.batch_env_file = batch_env_file.expanduser()
        self.codex_bin = codex_bin
        self.quiet = quiet
        self.static_dir = static_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local transcript review API.")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--batch-env-file", type=Path, default=DEFAULT_BATCH_ENV_FILE)
    parser.add_argument("--codex-bin", default=os.environ.get("TRANSCRIPTS_CODEX_BIN", DEFAULT_CODEX_BIN))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def serve(args: argparse.Namespace) -> None:
    root = store_dir(args.store_dir)
    with connect(root) as con:
        init_db(con)
    server = TranscriptApiServer(
        (args.host, args.port),
        TranscriptApiHandler,
        store_root=root,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        state_root=args.state_dir,
        batch_env_file=args.batch_env_file,
        codex_bin=args.codex_bin,
        quiet=bool(args.quiet),
        static_dir=None if args.no_static else args.static_dir,
    )
    print(f"Serving transcript API on http://{args.host}:{args.port} using {db_path(root)}")
    server.serve_forever()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        serve(args)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
