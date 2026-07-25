"""Oldest-forward evaluation campaign orchestration for speaker identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import intelligence_config
import provenance_config
import speaker_identity_preprocess
import transcript_artifact_access
from transcript_store import connect, init_db, store_dir

MANIFEST_SCHEMA_VERSION = "transcribe-audio.speaker-evaluation-campaign-manifest.v1"
DEFAULT_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio")
RUBRIC_VERSIONS = {
    "calendar_association": "calendar-association.v1",
    "person_link": "person-link.v1",
    "speaker_identity": "speaker-identity.v1",
}
DISPOSITION_RULE_VERSION = "oldest-forward-disposition.v1"
GOLD_SCHEMA_VERSION = "transcribe-audio.speaker-evaluation-gold.v1"
GOLD_FREEZE_SCHEMA_VERSION = (
    "transcribe-audio.speaker-evaluation-gold-freeze.v1"
)
APPLY_CAMPAIGN_TOKEN = "APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST"
RECORD_GOLD_TOKEN = "RECORD_SPEAKER_EVALUATION_GOLD"
FREEZE_GOLD_TOKEN = "FREEZE_SPEAKER_EVALUATION_GOLD_BATCH"
GOLD_DISPOSITIONS = {
    "eligible_known",
    "eligible_unknown",
    "duplicate_member",
    "incomplete",
    "spurious_or_non_conversation",
    "artifact_unavailable",
}
CALENDAR_OUTCOMES = {"correct", "partial", "wrong", "none", "uncertain"}
SPEAKER_OUTCOMES = {
    "person",
    "mixed",
    "non_person_audio",
    "unknown_to_reviewer",
    "insufficient_transcript",
}


def _json_object(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recording_time(row: Any, payload: dict[str, Any]) -> tuple[datetime, str, str]:
    candidates = (
        ("recording_start", payload.get("recording_start")),
        ("generated_at", row["generated_at"]),
        ("updated_at", row["updated_at"]),
    )
    for source, value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc), text, source
    return datetime.max.replace(tzinfo=timezone.utc), "", "unavailable"


def _artifact_access(row: Any) -> dict[str, Any]:
    source_path = Path(str(row["source_path"] or "")).expanduser()
    stored_path = Path(str(row["stored_path"] or "")).expanduser()
    source_accessible = source_path.is_file()
    stored_accessible = stored_path.is_file()
    return {
        "source_path": str(row["source_path"] or ""),
        "stored_path": str(row["stored_path"] or ""),
        "source_accessible": source_accessible,
        "stored_accessible": stored_accessible,
        "selected_location": (
            "source"
            if source_accessible
            else "stored"
            if stored_accessible
            else "unavailable"
        ),
    }


def _content_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _repository_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit_result.stdout.strip() or "unavailable",
        "dirty_tree": bool(status_result.stdout.strip()),
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_private_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Campaign artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign artifact must contain an object: {path}")
    return payload


def _campaign_dir(runtime_root: Path, campaign_id: str) -> Path:
    if not re.fullmatch(r"campaign-[a-f0-9]{20}", campaign_id):
        raise ValueError("Campaign ID is invalid.")
    return runtime_root.expanduser() / campaign_id


def _campaign_manifest(runtime_root: Path, campaign_id: str) -> dict[str, Any]:
    path = _campaign_dir(runtime_root, campaign_id) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Campaign manifest does not exist: {campaign_id}")
    return _read_json(path)


def apply_campaign(
    *,
    store_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    state_root: Optional[Path] = None,
    batch_size: int = 10,
    approval_token: str,
) -> dict[str, Any]:
    """Write one reviewed private manifest without starting model work."""
    if approval_token != APPLY_CAMPAIGN_TOKEN:
        raise ValueError(f"Campaign apply requires approval token {APPLY_CAMPAIGN_TOKEN}.")
    manifest = preview_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        state_root=state_root,
        batch_size=batch_size,
    )
    campaign_id = str(manifest["manifest_id"]).replace("manifest-", "campaign-", 1)
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    manifest_path = campaign_dir / "manifest.json"
    applied = {
        **manifest,
        "campaign_id": campaign_id,
        "mode": "applied",
        "applied_at": _utc_now(),
        "will_write_campaign_state": True,
    }
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("manifest_id") != applied["manifest_id"]:
            raise ValueError("Existing campaign manifest does not match this preview.")
        applied = existing
    else:
        _write_private_json(manifest_path, applied)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "manifest_id": applied["manifest_id"],
        "manifest_path": str(manifest_path),
        "batch_size": batch_size,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }


def _manifest_item(
    manifest: dict[str, Any],
    document_id: str,
) -> dict[str, Any]:
    item = next(
        (
            candidate
            for candidate in manifest.get("items") or []
            if isinstance(candidate, dict)
            and str(candidate.get("document_id") or "") == document_id
        ),
        None,
    )
    if item is None:
        raise ValueError("Document is not present in the campaign manifest.")
    return item


def _document_row(store_root: Optional[Path], document_id: str) -> dict[str, Any]:
    selected_store_root = store_dir(store_root)
    with connect(selected_store_root) as con:
        init_db(con)
        row = con.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Transcript document does not exist: {document_id}")
    return dict(row)


def review_case_packet(
    campaign_id: str,
    document_id: str,
    *,
    store_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Return private transcript/calendar clues without loading gold records."""
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    manifest = _campaign_manifest(selected_runtime_root, campaign_id)
    item = _manifest_item(manifest, document_id)
    document = _document_row(store_root, document_id)
    resolved = transcript_artifact_access.resolve_transcript_artifact(
        document,
        store_root=store_root,
    )
    transcript = transcript_artifact_access.read_resolved_transcript(resolved)
    utterances = [
        {
            "speaker": str(utterance.get("speaker") or ""),
            "start": utterance.get("start"),
            "end": utterance.get("end"),
            "text": str(utterance.get("text") or "")[:1_200],
        }
        for utterance in transcript.get("utterances") or []
        if isinstance(utterance, dict)
    ][:200]
    return {
        "schema_version": "transcribe-audio.speaker-evaluation-review-packet.v1",
        "campaign_id": campaign_id,
        "document_id": document_id,
        "chronological_rank": item.get("chronological_rank"),
        "candidate_role": item.get("candidate_role"),
        "artifact": resolved.to_dict(),
        "transcript_title": str(transcript.get("transcript_title") or ""),
        "recording_start": transcript.get("recording_start"),
        "recording_end": transcript.get("recording_end"),
        "event": transcript.get("event"),
        "speaker_labels": list(item.get("speaker_labels") or []),
        "utterances": utterances,
        "private_runtime_artifact": True,
        "will_read_gold_records": False,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }


def _validate_gold_review(
    review: dict[str, Any],
    *,
    speaker_labels: list[str],
) -> dict[str, Any]:
    disposition = str(review.get("disposition") or "")
    if disposition not in GOLD_DISPOSITIONS:
        raise ValueError("Gold disposition is invalid.")
    reviewer = str(review.get("reviewer") or "").strip()
    review_method = str(review.get("review_method") or "").strip()
    if not reviewer or not review_method:
        raise ValueError("Gold review requires reviewer and review_method.")
    calendar_association = str(review.get("calendar_association") or "uncertain")
    if calendar_association not in CALENDAR_OUTCOMES:
        raise ValueError("Calendar association review is invalid.")

    people = review.get("people") or []
    if not isinstance(people, list):
        raise ValueError("Gold people must be a list.")
    person_ids = {
        str(person.get("person_ground_truth_id") or "")
        for person in people
        if isinstance(person, dict)
        and str(person.get("person_ground_truth_id") or "")
    }
    if len(person_ids) != len(people):
        raise ValueError("Each gold person requires a unique person_ground_truth_id.")

    outcomes = review.get("speaker_outcomes") or []
    if not isinstance(outcomes, list):
        raise ValueError("Gold speaker_outcomes must be a list.")
    outcome_labels: list[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("Each speaker outcome must be an object.")
        label = str(outcome.get("speaker_label") or "")
        status = str(outcome.get("outcome") or "")
        person_id = str(outcome.get("person_ground_truth_id") or "")
        if label not in speaker_labels or label in outcome_labels:
            raise ValueError("Gold speaker labels must be prepared and unique.")
        if status not in SPEAKER_OUTCOMES:
            raise ValueError("Gold speaker outcome is invalid.")
        if status == "person" and person_id not in person_ids:
            raise ValueError("Person speaker outcomes must reference a gold person.")
        if status != "person" and person_id:
            raise ValueError("Only person outcomes may reference one gold person.")
        outcome_labels.append(label)
    if disposition == "eligible_known" and sorted(outcome_labels) != sorted(
        speaker_labels
    ):
        raise ValueError(
            "Eligible-known gold requires one outcome for every speaker label."
        )

    groups = review.get("same_person_label_groups") or []
    if not isinstance(groups, list):
        raise ValueError("same_person_label_groups must be a list.")
    for group in groups:
        if (
            not isinstance(group, list)
            or len(group) < 2
            or any(str(label) not in speaker_labels for label in group)
        ):
            raise ValueError(
                "Each same-person group needs at least two prepared labels."
            )
    return {
        "disposition": disposition,
        "calendar_association": calendar_association,
        "people": people,
        "speaker_outcomes": outcomes,
        "same_person_label_groups": groups,
        "reviewer": reviewer,
        "review_method": review_method,
        "notes": str(review.get("notes") or "").strip(),
    }


def _gold_index(campaign_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = campaign_dir / "gold" / "index.json"
    if not path.exists():
        return path, {
            "schema_version": "transcribe-audio.speaker-evaluation-gold-index.v1",
            "records": [],
        }
    index = _read_json(path)
    if not isinstance(index.get("records"), list):
        raise ValueError("Campaign gold index records must be a list.")
    return path, index


def campaign_status(
    campaign_id: str,
    *,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Return gold progress and next review row without exposing gold content."""
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    manifest = _campaign_manifest(selected_runtime_root, campaign_id)
    _index_path, index = _gold_index(campaign_dir)
    latest_by_document: dict[str, dict[str, Any]] = {}
    for record in index["records"]:
        if isinstance(record, dict):
            latest_by_document[str(record.get("document_id") or "")] = record
    latest = list(latest_by_document.values())
    reviewed_ids = set(latest_by_document)
    next_item = next(
        (
            item
            for item in manifest.get("items") or []
            if isinstance(item, dict)
            and item.get("disposition") == "needs_operator_classification"
            and str(item.get("document_id") or "") not in reviewed_ids
        ),
        None,
    )
    freezes = sorted((campaign_dir / "freezes").glob("*.json")) if (
        campaign_dir / "freezes"
    ).is_dir() else []
    disposition_counts = Counter(
        str(record.get("disposition") or "") for record in latest
    )
    return {
        "schema_version": "transcribe-audio.speaker-evaluation-campaign-status.v1",
        "campaign_id": campaign_id,
        "manifest_id": manifest["manifest_id"],
        "batch_size": int(manifest["batch_size"]),
        "reviewed_case_count": len(latest),
        "eligible_known_count": int(
            disposition_counts.get("eligible_known", 0)
        ),
        "reviewed_disposition_counts": dict(sorted(disposition_counts.items())),
        "next_review": (
            {
                "document_id": str(next_item["document_id"]),
                "chronological_rank": int(next_item["chronological_rank"]),
            }
            if next_item
            else None
        ),
        "latest_freeze_path": str(freezes[-1]) if freezes else "",
        "gold_content_included": False,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }


def record_gold_review(
    campaign_id: str,
    document_id: str,
    review: dict[str, Any],
    *,
    store_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    approval_token: str,
    supersedes_gold_id: str = "",
) -> dict[str, Any]:
    """Append one private operator gold record; never rewrite prior reviews."""
    if approval_token != RECORD_GOLD_TOKEN:
        raise ValueError(f"Gold review requires approval token {RECORD_GOLD_TOKEN}.")
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    manifest = _campaign_manifest(selected_runtime_root, campaign_id)
    item = _manifest_item(manifest, document_id)
    index_path, index = _gold_index(campaign_dir)
    records = index["records"]
    prior = [
        record
        for record in records
        if isinstance(record, dict)
        and str(record.get("document_id") or "") == document_id
    ]
    if prior and not supersedes_gold_id:
        raise ValueError("A correction must identify supersedes_gold_id.")
    if supersedes_gold_id and (
        not prior or str(prior[-1].get("gold_id") or "") != supersedes_gold_id
    ):
        raise ValueError("supersedes_gold_id must reference the latest case review.")
    validated = _validate_gold_review(
        review,
        speaker_labels=list(item.get("speaker_labels") or []),
    )
    gold_id = str(uuid4())
    reviewed_at = _utc_now()
    record = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "gold_id": gold_id,
        "campaign_id": campaign_id,
        "manifest_id": manifest["manifest_id"],
        "document_id": document_id,
        "chronological_rank": item["chronological_rank"],
        "artifact_sha256": item["artifact_sha256"],
        **validated,
        "reviewed_at": reviewed_at,
        "supersedes_gold_id": str(supersedes_gold_id or ""),
        "prediction_visibility": "excluded",
    }
    gold_path = campaign_dir / "gold" / document_id / f"{gold_id}.json"
    _write_private_json(gold_path, record)
    index_record = {
        "gold_id": gold_id,
        "document_id": document_id,
        "chronological_rank": item["chronological_rank"],
        "disposition": validated["disposition"],
        "reviewed_at": reviewed_at,
        "supersedes_gold_id": str(supersedes_gold_id or ""),
        "path": str(gold_path),
    }
    _write_private_json(
        index_path,
        {**index, "records": [*records, index_record]},
    )
    return {**record, "gold_path": str(gold_path)}


def freeze_gold_batch(
    campaign_id: str,
    *,
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Freeze exactly K current eligible-known gold records and reserve holdout."""
    if approval_token != FREEZE_GOLD_TOKEN:
        raise ValueError(f"Gold freeze requires approval token {FREEZE_GOLD_TOKEN}.")
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    manifest = _campaign_manifest(selected_runtime_root, campaign_id)
    _index_path, index = _gold_index(campaign_dir)
    latest_by_document: dict[str, dict[str, Any]] = {}
    for record in index["records"]:
        if isinstance(record, dict):
            latest_by_document[str(record.get("document_id") or "")] = record
    ordered = sorted(
        latest_by_document.values(),
        key=lambda record: (
            int(record.get("chronological_rank") or 0),
            str(record.get("document_id") or ""),
        ),
    )
    eligible = [
        record
        for record in ordered
        if record.get("disposition") == "eligible_known"
    ]
    batch_size = int(manifest["batch_size"])
    if len(eligible) < batch_size:
        raise ValueError(
            f"Gold batch needs {batch_size} eligible-known cases; found {len(eligible)}."
        )
    selected = eligible[:batch_size]
    selected_ids = {str(record["document_id"]) for record in selected}
    last_rank = max(int(record["chronological_rank"]) for record in selected)
    holdout_ids = [
        str(item["document_id"])
        for item in manifest.get("items") or []
        if isinstance(item, dict)
        and int(item.get("chronological_rank") or 0) > last_rank
        and item.get("disposition") == "needs_operator_classification"
        and str(item.get("document_id") or "") not in selected_ids
    ][:batch_size]
    freeze_id = str(uuid4())
    frozen_at = _utc_now()
    freeze = {
        "schema_version": GOLD_FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "campaign_id": campaign_id,
        "manifest_id": manifest["manifest_id"],
        "status": "gold_batch_frozen",
        "batch_size": batch_size,
        "gold_case_count": len(selected),
        "gold_ids": [str(record["gold_id"]) for record in selected],
        "document_ids": [str(record["document_id"]) for record in selected],
        "blind_holdout_document_ids": holdout_ids,
        "frozen_at": frozen_at,
        "prediction_visibility": "excluded",
    }
    freeze_path = campaign_dir / "freezes" / f"{freeze_id}.json"
    _write_private_json(freeze_path, freeze)
    return {**freeze, "freeze_path": str(freeze_path)}


def preview_campaign(
    *,
    store_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    state_root: Optional[Path] = None,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Return a deterministic, read-only oldest-forward campaign manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    selected_store_root = store_dir(store_root)
    selected_runtime_root = (runtime_root or DEFAULT_CAMPAIGN_ROOT).expanduser()
    selected_state_root = (state_root or DEFAULT_STATE_ROOT).expanduser()
    with connect(selected_store_root) as con:
        init_db(con)
        rows = con.execute(
            "SELECT * FROM documents WHERE kind = 'transcript'"
        ).fetchall()

    prepared: list[tuple[datetime, str, dict[str, Any]]] = []
    for row in rows:
        payload = _json_object(row["json_payload"])
        sort_time, recording_time, recording_time_source = _recording_time(
            row, payload
        )
        access = _artifact_access(row)
        utterances = (
            payload.get("utterances")
            if isinstance(payload.get("utterances"), list)
            else []
        )
        transcript_text = " ".join(
            str(utterance.get("text") or "").strip()
            for utterance in utterances
            if isinstance(utterance, dict)
        ).strip()
        if access["selected_location"] == "unavailable":
            disposition = "artifact_unavailable"
            disposition_reason = "no_accessible_source_or_stored_artifact"
        elif len(utterances) <= 1:
            disposition = "incomplete"
            disposition_reason = "one_or_zero_utterances"
        elif len(transcript_text) < 250:
            disposition = "incomplete"
            disposition_reason = "transcript_text_under_250_characters"
        else:
            disposition = "needs_operator_classification"
            disposition_reason = "operator_review_required"
        item = {
            "document_id": str(row["id"]),
            "recording_time": recording_time,
            "recording_time_source": recording_time_source,
            "artifact_sha256": str(row["artifact_sha256"] or ""),
            "artifact": access,
            "utterance_count": len(utterances),
            "transcript_text_chars": len(transcript_text),
            "content_fingerprint": _content_fingerprint(transcript_text),
            "speaker_labels": sorted(
                {
                    str(utterance.get("speaker") or "").strip()
                    for utterance in utterances
                    if isinstance(utterance, dict)
                    and str(utterance.get("speaker") or "").strip()
                }
            ),
            "disposition": disposition,
            "disposition_reason": disposition_reason,
        }
        prepared.append((sort_time, str(row["id"]), item))

    prepared.sort(key=lambda entry: (entry[0], entry[1]))
    items = [entry[2] for entry in prepared]
    for rank, item in enumerate(items, start=1):
        item["chronological_rank"] = rank

    fingerprint_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        fingerprint = str(item["content_fingerprint"])
        if fingerprint:
            fingerprint_groups[fingerprint].append(item)
    duplicate_cluster_count = 0
    for fingerprint, group in sorted(fingerprint_groups.items()):
        if len(group) < 2:
            continue
        duplicate_cluster_count += 1
        canonical = group[0]
        cluster_id = f"duplicate-{fingerprint[:16]}"
        for item in group:
            item["duplicate_cluster_id"] = cluster_id
            item["duplicate_evidence"] = "exact_normalized_transcript"
        for item in group[1:]:
            if item["disposition"] == "needs_operator_classification":
                item["disposition"] = "duplicate_member"
                item["disposition_reason"] = "exact_normalized_transcript_duplicate"
                item["duplicate_of_document_id"] = canonical["document_id"]

    reviewable_items = [
        item
        for item in items
        if item["disposition"] == "needs_operator_classification"
    ]
    for index, item in enumerate(reviewable_items):
        if index < batch_size:
            item["candidate_role"] = "gold_review_candidate"
        elif index < batch_size * 2:
            item["candidate_role"] = "blind_holdout_candidate"
        else:
            item["candidate_role"] = "future_candidate"
    for item in items:
        item.setdefault("candidate_role", "excluded_pending_disposition")

    cursor_item = next(
        (
            item
            for item in items
            if item["disposition"] == "needs_operator_classification"
        ),
        None,
    )
    disposition_counts = Counter(item["disposition"] for item in items)
    algorithm = _repository_state()
    model_route = intelligence_config.resolve_task_config(
        intelligence_config.TASK_SPEAKER_DISAMBIGUATION
    ).to_dict()
    provenance_snapshot = provenance_config.all_config(
        state_root=selected_state_root
    )
    provenance_config_fingerprint = _sha256_json(provenance_snapshot)
    manifest_basis = {
        "algorithm": algorithm,
        "batch_size": batch_size,
        "documents": [
            {
                "artifact_sha256": item["artifact_sha256"],
                "disposition": item["disposition"],
                "document_id": item["document_id"],
                "recording_time": item["recording_time"],
            }
            for item in items
        ],
        "model_route": model_route,
        "provenance_config_fingerprint": provenance_config_fingerprint,
        "rubric_versions": RUBRIC_VERSIONS,
        "rule_version": DISPOSITION_RULE_VERSION,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"manifest-{_sha256_json(manifest_basis)[:20]}",
        "mode": "preview",
        "store_root": str(selected_store_root),
        "runtime_root": str(selected_runtime_root),
        "state_root": str(selected_state_root),
        "batch_size": batch_size,
        "algorithm": algorithm,
        "model_route": model_route,
        "provenance_config_fingerprint": provenance_config_fingerprint,
        "provenance_snapshot_policy": "fresh_retrieval",
        "rubric_versions": dict(RUBRIC_VERSIONS),
        "schema_versions": {
            "clue_packet": (
                speaker_identity_preprocess.SPEAKER_CLUE_PACKET_SCHEMA_VERSION
            ),
            "clue_discovery_packet": (
                speaker_identity_preprocess.CLUE_DISCOVERY_PACKET_SCHEMA_VERSION
            ),
            "clue_discovery_readout": (
                speaker_identity_preprocess.CLUE_DISCOVERY_READOUT_SCHEMA_VERSION
            ),
            "identity_evaluation_packet": (
                speaker_identity_preprocess.IDENTITY_EVALUATION_PACKET_SCHEMA_VERSION
            ),
            "speaker_identity_readout": (
                speaker_identity_preprocess.SPEAKER_IDENTITY_READOUT_SCHEMA_VERSION
            ),
            "disposition_rules": DISPOSITION_RULE_VERSION,
        },
        "cursor": {
            "chronological_rank": (
                int(cursor_item["chronological_rank"]) if cursor_item else None
            ),
            "document_id": str(cursor_item["document_id"]) if cursor_item else "",
        },
        "summary": {
            "total_rows": len(items),
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "duplicate_cluster_count": duplicate_cluster_count,
            "gold_review_candidate_count": min(
                len(reviewable_items), batch_size
            ),
            "blind_holdout_candidate_count": min(
                max(len(reviewable_items) - batch_size, 0),
                batch_size,
            ),
        },
        "items": items,
        "will_write_campaign_state": False,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the oldest-forward speaker identity evaluation campaign."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser(
        "preview",
        help="Print a deterministic campaign manifest without writing state.",
    )
    preview.add_argument("--store-root", type=Path)
    preview.add_argument("--runtime-root", type=Path)
    preview.add_argument("--state-root", type=Path)
    preview.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "preview":
        manifest = preview_campaign(
            store_root=args.store_root,
            runtime_root=args.runtime_root,
            state_root=args.state_root,
            batch_size=args.batch_size,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported campaign command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
