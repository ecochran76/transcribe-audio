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
from itertools import combinations
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
    "speaker_identity": "speaker-identity.v2",
}
DISPOSITION_RULE_VERSION = "oldest-forward-disposition.v1"
GOLD_SCHEMA_VERSION = "transcribe-audio.speaker-evaluation-gold.v1"
GOLD_FREEZE_SCHEMA_VERSION = (
    "transcribe-audio.speaker-evaluation-gold-freeze.v1"
)
BLIND_BASELINE_SCHEMA_VERSION = (
    "transcribe-audio.speaker-evaluation-blind-baseline.v1"
)
APPLY_CAMPAIGN_TOKEN = "APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST"
RECORD_GOLD_TOKEN = "RECORD_SPEAKER_EVALUATION_GOLD"
FREEZE_GOLD_TOKEN = "FREEZE_SPEAKER_EVALUATION_GOLD_BATCH"
START_BLIND_BASELINE_TOKEN = "START_SPEAKER_EVALUATION_BLIND_BASELINE"
CAPTURE_BLIND_PREDICTION_TOKEN = (
    "CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION"
)
REVEAL_GOLD_COMPARISON_TOKEN = (
    "REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON"
)
RECORD_REFINEMENT_DECISION_TOKEN = (
    "RECORD_SPEAKER_EVALUATION_REFINEMENT_DECISION"
)
REPLAY_SPEAKER_CONFIDENCE_CALIBRATION_TOKEN = (
    "REPLAY_SPEAKER_CONFIDENCE_CALIBRATION"
)
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


def start_blind_baseline(
    campaign_id: str,
    *,
    freeze_id: str,
    runtime_root: Optional[Path] = None,
    approval_token: str,
    run_kind: str = "baseline",
    parent_baseline_id: str = "",
    hypothesis: str = "",
    evidence_mode: str = "fresh_retrieval",
) -> dict[str, Any]:
    """Start one private prediction run without opening any gold record."""
    if approval_token != START_BLIND_BASELINE_TOKEN:
        raise ValueError(
            "Blind baseline start requires approval token "
            f"{START_BLIND_BASELINE_TOKEN}."
        )
    if not re.fullmatch(r"[0-9a-f-]{36}", freeze_id):
        raise ValueError("Gold freeze ID is invalid.")
    if run_kind not in {"baseline", "refinement", "holdout"}:
        raise ValueError("Blind run kind is invalid.")
    if evidence_mode not in {
        "fresh_retrieval",
        "fresh_retrieval_comparison",
        "preserved_evidence_replay",
    }:
        raise ValueError("Blind run evidence mode is invalid.")
    if run_kind == "refinement" and not parent_baseline_id:
        raise ValueError("Refinement runs require parent_baseline_id.")
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    manifest = _campaign_manifest(selected_runtime_root, campaign_id)
    freeze_path = campaign_dir / "freezes" / f"{freeze_id}.json"
    if not freeze_path.is_file():
        raise ValueError(f"Gold freeze does not exist: {freeze_id}")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("schema_version") != GOLD_FREEZE_SCHEMA_VERSION
        or freeze.get("campaign_id") != campaign_id
        or freeze.get("manifest_id") != manifest.get("manifest_id")
    ):
        raise ValueError("Gold freeze does not belong to this campaign manifest.")
    document_id_key = (
        "blind_holdout_document_ids" if run_kind == "holdout" else "document_ids"
    )
    document_ids = [
        str(document_id)
        for document_id in freeze.get(document_id_key) or []
        if str(document_id)
    ]
    expected_count = (
        len(freeze.get("blind_holdout_document_ids") or [])
        if run_kind == "holdout"
        else int(freeze.get("gold_case_count") or 0)
    )
    if not document_ids or len(document_ids) != expected_count:
        raise ValueError("Gold freeze document count is inconsistent.")
    cases = []
    for document_id in document_ids:
        item = _manifest_item(manifest, document_id)
        cases.append(
            {
                "document_id": document_id,
                "chronological_rank": int(item["chronological_rank"]),
                "artifact_sha256": str(item["artifact_sha256"]),
                "duplicate_cluster_id": str(
                    item.get("duplicate_cluster_id") or ""
                ),
                "status": "awaiting_prediction",
            }
        )
    baseline_id = f"baseline-{uuid4()}"
    baseline_path = campaign_dir / "baselines" / baseline_id / "baseline.json"
    baseline = {
        "schema_version": BLIND_BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "campaign_id": campaign_id,
        "manifest_id": manifest["manifest_id"],
        "freeze_id": freeze_id,
        "status": "awaiting_predictions",
        "run_kind": run_kind,
        "parent_baseline_id": parent_baseline_id,
        "hypothesis": hypothesis,
        "evidence_mode": evidence_mode,
        "started_at": _utc_now(),
        "document_ids": document_ids,
        "cases": cases,
        "captured_prediction_count": 0,
        "batch_size": len(document_ids),
        "algorithm": (
            manifest.get("algorithm") or {}
            if run_kind == "baseline"
            else _repository_state()
        ),
        "model_route": manifest.get("model_route") or {},
        "rubric_versions": manifest.get("rubric_versions") or {},
        "provenance_config_fingerprint": str(
            manifest.get("provenance_config_fingerprint") or ""
        ),
        "prediction_visibility": "blind",
        "will_read_gold_records": False,
        "gold_content_included": False,
        "will_perform_external_write": False,
    }
    _write_private_json(baseline_path, baseline)
    return {**baseline, "baseline_path": str(baseline_path)}


def _blind_baseline_path(
    runtime_root: Path,
    campaign_id: str,
    baseline_id: str,
) -> Path:
    if not re.fullmatch(
        r"baseline-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}",
        baseline_id,
    ):
        raise ValueError("Blind baseline ID is invalid.")
    return (
        _campaign_dir(runtime_root, campaign_id)
        / "baselines"
        / baseline_id
        / "baseline.json"
    )


def blind_baseline_status(
    campaign_id: str,
    *,
    baseline_id: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Return prediction progress without reading freeze or gold artifacts."""
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    baseline_path = _blind_baseline_path(
        selected_runtime_root,
        campaign_id,
        baseline_id,
    )
    if not baseline_path.is_file():
        raise ValueError(f"Blind baseline does not exist: {baseline_id}")
    baseline = _read_json(baseline_path)
    if baseline.get("schema_version") != BLIND_BASELINE_SCHEMA_VERSION:
        raise ValueError("Blind baseline schema is invalid.")
    return {
        **baseline,
        "baseline_path": str(baseline_path),
        "will_read_gold_records": False,
        "gold_content_included": False,
    }


def capture_blind_prediction(
    campaign_id: str,
    *,
    baseline_id: str,
    document_id: str,
    artifact_sha256: str,
    prediction: dict[str, Any],
    run_references: Optional[dict[str, Any]] = None,
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Capture one immutable prediction before any gold comparison is allowed."""
    if approval_token != CAPTURE_BLIND_PREDICTION_TOKEN:
        raise ValueError(
            "Blind prediction capture requires approval token "
            f"{CAPTURE_BLIND_PREDICTION_TOKEN}."
        )
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    baseline_path = _blind_baseline_path(
        selected_runtime_root,
        campaign_id,
        baseline_id,
    )
    if not baseline_path.is_file():
        raise ValueError(f"Blind baseline does not exist: {baseline_id}")
    baseline = _read_json(baseline_path)
    if baseline.get("schema_version") != BLIND_BASELINE_SCHEMA_VERSION:
        raise ValueError("Blind baseline schema is invalid.")
    cases = [
        dict(item)
        for item in baseline.get("cases") or []
        if isinstance(item, dict)
    ]
    case = next(
        (
            item
            for item in cases
            if str(item.get("document_id") or "") == document_id
        ),
        None,
    )
    if case is None:
        raise ValueError("Document is not part of this blind baseline.")
    if case.get("status") == "prediction_captured":
        raise ValueError("Document already has a captured prediction.")
    if str(case.get("artifact_sha256") or "") != artifact_sha256:
        raise ValueError("Prediction artifact hash does not match the baseline.")
    evaluation_id = str(prediction.get("evaluation_id") or "").strip()
    if not evaluation_id:
        raise ValueError("Blind prediction requires an evaluation_id.")
    prediction_id = f"prediction-{uuid4()}"
    captured_at = _utc_now()
    prediction_record = {
        "schema_version": (
            "transcribe-audio.speaker-evaluation-blind-prediction.v1"
        ),
        "prediction_id": prediction_id,
        "baseline_id": baseline_id,
        "campaign_id": campaign_id,
        "document_id": document_id,
        "artifact_sha256": artifact_sha256,
        "prediction": prediction,
        "run_references": dict(run_references or {}),
        "captured_at": captured_at,
        "prediction_visibility": "blind",
        "will_read_gold_records": False,
        "gold_content_included": False,
        "will_perform_external_write": False,
    }
    prediction_path = (
        baseline_path.parent
        / "predictions"
        / document_id
        / f"{prediction_id}.json"
    )
    _write_private_json(prediction_path, prediction_record)
    case.update(
        {
            "status": "prediction_captured",
            "prediction_id": prediction_id,
            "prediction_path": str(prediction_path),
            "captured_at": captured_at,
        }
    )
    captured_count = sum(
        1 for item in cases if item.get("status") == "prediction_captured"
    )
    status = (
        "predictions_complete"
        if captured_count == int(baseline.get("batch_size") or 0)
        else "awaiting_predictions"
    )
    updated = {
        **baseline,
        "status": status,
        "cases": cases,
        "captured_prediction_count": captured_count,
        "predictions_completed_at": captured_at
        if status == "predictions_complete"
        else "",
    }
    _write_private_json(baseline_path, updated)
    return {
        **updated,
        "baseline_path": str(baseline_path),
        "prediction_id": prediction_id,
        "prediction_path": str(prediction_path),
    }


def _normalized_identity_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^a-z0-9@.+-]+", " ", text).split())


def _prediction_person_index(prediction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(person.get("person_id") or ""): person
        for person in prediction.get("people") or []
        if isinstance(person, dict) and str(person.get("person_id") or "")
    }


def _prediction_identity_values(person: dict[str, Any]) -> set[str]:
    values = {
        _normalized_identity_text(
            person.get("display_name")
            or person.get("label")
            or person.get("name")
        )
    }
    emails = person.get("emails") if isinstance(person.get("emails"), list) else []
    values.update(_normalized_identity_text(value) for value in emails)
    values.add(_normalized_identity_text(person.get("email")))
    for source in person.get("source_records") or []:
        if isinstance(source, dict):
            values.add(_normalized_identity_text(source.get("label")))
            values.add(_normalized_identity_text(source.get("email")))
    return {value for value in values if value}


def _gold_identity_values(person: dict[str, Any]) -> set[str]:
    values = {
        _normalized_identity_text(person.get("name")),
        _normalized_identity_text(person.get("email")),
        _normalized_identity_text(person.get("alternate_email")),
    }
    return {value for value in values if value}


def _proposal_matches_gold_person(
    proposal: dict[str, Any],
    *,
    people_by_id: dict[str, dict[str, Any]],
    gold_person: dict[str, Any],
) -> bool:
    candidate_values: set[str] = set()
    person_id = str(proposal.get("person_id") or "")
    if person_id in people_by_id:
        candidate_values.update(
            _prediction_identity_values(people_by_id[person_id])
        )
    suggested = (
        proposal.get("suggested_person")
        if isinstance(proposal.get("suggested_person"), dict)
        else {}
    )
    candidate_values.update(_prediction_identity_values(suggested))
    return bool(candidate_values & _gold_identity_values(gold_person))


def reveal_blind_baseline_comparison(
    campaign_id: str,
    *,
    baseline_id: str,
    runtime_root: Optional[Path] = None,
    approval_token: str,
    allow_reviewed_holdout_replay: bool = False,
) -> dict[str, Any]:
    """Reveal frozen gold only after every blind prediction is immutable."""
    if approval_token != REVEAL_GOLD_COMPARISON_TOKEN:
        raise ValueError(
            "Gold comparison reveal requires approval token "
            f"{REVEAL_GOLD_COMPARISON_TOKEN}."
        )
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    baseline_path = _blind_baseline_path(
        selected_runtime_root,
        campaign_id,
        baseline_id,
    )
    if not baseline_path.is_file():
        raise ValueError(f"Blind baseline does not exist: {baseline_id}")
    baseline = _read_json(baseline_path)
    if baseline.get("status") != "predictions_complete":
        raise ValueError(
            "Gold comparison requires every blind prediction to be captured."
        )
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    freeze_id = str(baseline.get("freeze_id") or "")
    freeze = _read_json(campaign_dir / "freezes" / f"{freeze_id}.json")
    comparison_mode = "blind_reveal"
    prior_holdout_baseline_id = ""
    if baseline.get("run_kind") == "holdout":
        _index_path, gold_index = _gold_index(campaign_dir)
        latest_by_document = {
            str(record.get("document_id") or ""): record
            for record in gold_index.get("records") or []
            if isinstance(record, dict)
        }
        completed_at = str(baseline.get("predictions_completed_at") or "")
        gold_id_by_document = {}
        if allow_reviewed_holdout_replay:
            if baseline.get("evidence_mode") not in {
                "fresh_retrieval_comparison",
                "preserved_evidence_replay",
            }:
                raise ValueError(
                    "Reviewed holdout replay requires an explicit comparison evidence mode."
                )
            current_documents = {
                str(value) for value in baseline.get("document_ids") or []
            }
            prior_candidates = []
            for prior_path in sorted((campaign_dir / "baselines").glob("*/baseline.json")):
                prior = _read_json(prior_path)
                if (
                    prior.get("baseline_id") != baseline_id
                    and prior.get("run_kind") == "holdout"
                    and prior.get("status") == "comparison_complete"
                    and {
                        str(value)
                        for value in prior.get("document_ids") or []
                    }
                    == current_documents
                    and (prior_path.parent / "comparison.json").is_file()
                ):
                    prior_candidates.append(prior)
            if not prior_candidates:
                raise ValueError(
                    "Reviewed holdout replay requires a prior completed comparison "
                    "for the exact holdout cohort."
                )
            prior_holdout_baseline_id = str(
                prior_candidates[-1].get("baseline_id") or ""
            )
            comparison_mode = "reviewed_holdout_replay"
        for document_id in baseline.get("document_ids") or []:
            record = latest_by_document.get(str(document_id))
            if (
                not record
                or (
                    not allow_reviewed_holdout_replay
                    and str(record.get("reviewed_at") or "") < completed_at
                )
            ):
                raise ValueError(
                    "Every holdout case must be reviewed after predictions completed."
                )
            gold_id_by_document[str(document_id)] = str(
                record.get("gold_id") or ""
            )
    else:
        document_ids = [str(value) for value in freeze.get("document_ids") or []]
        gold_ids = [str(value) for value in freeze.get("gold_ids") or []]
        if len(document_ids) != len(gold_ids):
            raise ValueError("Gold freeze identity mapping is inconsistent.")
        gold_id_by_document = dict(zip(document_ids, gold_ids))
    revealed_at = _utc_now()
    calendar_metrics = Counter(
        {"cases": 0, "exact": 0, "high_or_very_high_wrong": 0}
    )
    speaker_metrics = Counter(
        {
            "known_person_labels": 0,
            "top_proposal_correct": 0,
            "correct_person_present": 0,
            "high_or_very_high_wrong": 0,
        }
    )
    mixed_total = 0
    mixed_flagged = 0
    validation_metrics = Counter(
        {
            "predictions": 0,
            "completed": 0,
            "model_output_rejected": 0,
        }
    )
    grouping_true_positive = 0
    grouping_false_positive = 0
    grouping_false_negative = 0
    cases = []
    calendar_expected_status = {
        "correct": "matched",
        "partial": "ambiguous",
        "wrong": "unmatched",
        "none": "unmatched",
        "uncertain": "ambiguous",
    }
    for baseline_case in baseline.get("cases") or []:
        if not isinstance(baseline_case, dict):
            continue
        document_id = str(baseline_case.get("document_id") or "")
        prediction_path = Path(
            str(baseline_case.get("prediction_path") or "")
        )
        prediction_record = _read_json(prediction_path)
        prediction = (
            prediction_record.get("prediction")
            if isinstance(prediction_record.get("prediction"), dict)
            else {}
        )
        validation_metrics["predictions"] += 1
        if prediction.get("status") == "model_output_rejected":
            validation_metrics["model_output_rejected"] += 1
            failure_stage = str(prediction.get("failure_stage") or "unknown")
            validation_metrics[f"stage_{failure_stage}"] += 1
        else:
            validation_metrics["completed"] += 1
        gold_id = gold_id_by_document.get(document_id, "")
        gold_path = campaign_dir / "gold" / document_id / f"{gold_id}.json"
        gold = _read_json(gold_path)
        if (
            gold.get("schema_version") != GOLD_SCHEMA_VERSION
            or str(gold.get("artifact_sha256") or "")
            != str(baseline_case.get("artifact_sha256") or "")
        ):
            raise ValueError("Frozen gold does not match the blind case artifact.")
        if gold.get("disposition") != "eligible_known":
            cases.append(
                {
                    "document_id": document_id,
                    "chronological_rank": baseline_case.get("chronological_rank"),
                    "prediction_id": prediction_record.get("prediction_id"),
                    "gold_id": gold_id,
                    "gold_disposition": gold.get("disposition"),
                    "prediction_captured_at": prediction_record.get("captured_at"),
                    "gold_revealed_at": revealed_at,
                    "evaluation_excluded": True,
                    "exclusion_reason": "non_scorable_gold_disposition",
                    "failure_classes": [],
                }
            )
            continue
        calendar_prediction = (
            prediction.get("calendar_association")
            if isinstance(prediction.get("calendar_association"), dict)
            else {}
        )
        predicted_calendar_status = str(
            calendar_prediction.get("status") or ""
        )
        expected_calendar_status = calendar_expected_status.get(
            str(gold.get("calendar_association") or ""),
            "ambiguous",
        )
        calendar_exact = predicted_calendar_status == expected_calendar_status
        calendar_band = str(
            (
                calendar_prediction.get("confidence")
                if isinstance(calendar_prediction.get("confidence"), dict)
                else {}
            ).get("band")
            or ""
        )
        calendar_metrics["cases"] += 1
        calendar_metrics["exact"] += int(calendar_exact)
        calendar_metrics["high_or_very_high_wrong"] += int(
            not calendar_exact and calendar_band in {"high", "very_high"}
        )

        people_by_id = _prediction_person_index(prediction)
        proposals = [
            proposal
            for proposal in (
                prediction.get("proposals")
                or prediction.get("speaker_assignments")
                or []
            )
            if isinstance(proposal, dict)
        ]
        gold_people = {
            str(person.get("person_ground_truth_id") or ""): person
            for person in gold.get("people") or []
            if isinstance(person, dict)
        }
        gold_group_pairs = {
            tuple(sorted(pair))
            for group in gold.get("same_person_label_groups") or []
            if isinstance(group, list)
            for pair in combinations(
                [str(label) for label in group if str(label)],
                2,
            )
        }
        proposed_labels_by_person: dict[str, set[str]] = defaultdict(set)
        for proposal in proposals:
            if proposal.get("status") != "candidate_match":
                continue
            person_id = str(proposal.get("person_id") or "")
            if not person_id:
                continue
            proposed_labels_by_person[person_id].update(
                str(label)
                for label in proposal.get("speaker_labels") or []
                if str(label)
            )
        predicted_group_pairs = {
            tuple(sorted(pair))
            for labels in proposed_labels_by_person.values()
            for pair in combinations(sorted(labels), 2)
        }
        grouping_true_positive += len(gold_group_pairs & predicted_group_pairs)
        grouping_false_positive += len(
            predicted_group_pairs - gold_group_pairs
        )
        grouping_false_negative += len(
            gold_group_pairs - predicted_group_pairs
        )
        label_results = []
        for outcome in gold.get("speaker_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            label = str(outcome.get("speaker_label") or "")
            label_proposals = [
                proposal
                for proposal in proposals
                if label
                in [
                    str(value)
                    for value in proposal.get("speaker_labels") or []
                ]
            ]
            if outcome.get("outcome") == "mixed":
                mixed_total += 1
                flagged = any(
                    "possible_mixed_speaker"
                    in [str(flag) for flag in proposal.get("review_flags") or []]
                    or len(
                        {
                            str(item.get("person_id") or "")
                            for item in proposal.get("utterance_assignments") or []
                            if isinstance(item, dict)
                            and str(item.get("person_id") or "")
                        }
                    )
                    > 1
                    for proposal in label_proposals
                )
                mixed_flagged += int(flagged)
                label_results.append(
                    {
                        "speaker_label": label,
                        "gold_outcome": "mixed",
                        "mixed_flagged": flagged,
                    }
                )
                continue
            if outcome.get("outcome") != "person":
                continue
            gold_person = gold_people.get(
                str(outcome.get("person_ground_truth_id") or ""),
                {},
            )
            matches = [
                _proposal_matches_gold_person(
                    proposal,
                    people_by_id=people_by_id,
                    gold_person=gold_person,
                )
                for proposal in label_proposals
            ]
            top_correct = bool(matches and matches[0])
            present = any(matches)
            top_confidence = (
                label_proposals[0].get("confidence")
                if label_proposals
                and isinstance(label_proposals[0].get("confidence"), dict)
                else {}
            )
            top_band = str(top_confidence.get("band") or "")
            speaker_metrics["known_person_labels"] += 1
            speaker_metrics["top_proposal_correct"] += int(top_correct)
            speaker_metrics["correct_person_present"] += int(present)
            speaker_metrics["high_or_very_high_wrong"] += int(
                not top_correct and top_band in {"high", "very_high"}
            )
            label_results.append(
                {
                    "speaker_label": label,
                    "gold_outcome": "person",
                    "top_proposal_correct": top_correct,
                    "correct_person_present": present,
                    "top_confidence_band": top_band,
                }
            )
        failure_classes = []
        prediction_failure_class = str(
            prediction.get("failure_class") or ""
        )
        if prediction_failure_class:
            failure_classes.append(prediction_failure_class)
        if not calendar_exact:
            failure_classes.append("calendar_generic_or_canceled")
        if any(
            result.get("gold_outcome") == "person"
            and not result.get("correct_person_present")
            for result in label_results
        ):
            failure_classes.append("candidate_generation")
        if any(
            result.get("gold_outcome") == "mixed"
            and not result.get("mixed_flagged")
            for result in label_results
        ):
            failure_classes.append("mixed_label_detection")
        cases.append(
            {
                "document_id": document_id,
                "chronological_rank": baseline_case.get("chronological_rank"),
                "prediction_id": prediction_record.get("prediction_id"),
                "gold_id": gold_id,
                "prediction_captured_at": prediction_record.get("captured_at"),
                "gold_revealed_at": revealed_at,
                "calendar": {
                    "reviewed": gold.get("calendar_association"),
                    "expected_prediction_status": expected_calendar_status,
                    "predicted_status": predicted_calendar_status,
                    "confidence": calendar_prediction.get("confidence") or {},
                    "exact": calendar_exact,
                },
                "speaker_labels": label_results,
                "failure_classes": failure_classes,
            }
        )
    comparison = {
        "schema_version": (
            "transcribe-audio.speaker-evaluation-baseline-comparison.v1"
        ),
        "comparison_id": f"comparison-{uuid4()}",
        "baseline_id": baseline_id,
        "campaign_id": campaign_id,
        "freeze_id": freeze_id,
        "status": "comparison_complete",
        "gold_revealed_at": revealed_at,
        "comparison_mode": comparison_mode,
        "prior_holdout_baseline_id": prior_holdout_baseline_id,
        "predictions_captured_before_gold_reveal": (
            comparison_mode == "blind_reveal"
            and all(
                str(case.get("prediction_captured_at") or "") <= revealed_at
                for case in cases
            )
        ),
        "metrics": {
            "calendar_association": dict(calendar_metrics),
            "speaker_identity": dict(speaker_metrics),
            "diarization": {
                "reviewed_mixed_labels": mixed_total,
                "mixed_labels_flagged": mixed_flagged,
                "same_person_pair_true_positive": grouping_true_positive,
                "same_person_pair_false_positive": grouping_false_positive,
                "same_person_pair_false_negative": grouping_false_negative,
                "same_person_pair_precision": (
                    grouping_true_positive
                    / (grouping_true_positive + grouping_false_positive)
                    if grouping_true_positive + grouping_false_positive
                    else 0.0
                ),
                "same_person_pair_recall": (
                    grouping_true_positive
                    / (grouping_true_positive + grouping_false_negative)
                    if grouping_true_positive + grouping_false_negative
                    else 0.0
                ),
            },
            "validation": dict(validation_metrics),
        },
        "cases": cases,
        "prediction_visibility": "revealed_after_complete_batch",
        "will_perform_external_write": False,
    }
    comparison_path = baseline_path.parent / "comparison.json"
    if comparison_path.exists():
        raise ValueError("Blind baseline comparison already exists.")
    _write_private_json(comparison_path, comparison)
    _write_private_json(
        baseline_path,
        {
            **baseline,
            "status": "comparison_complete",
            "gold_revealed_at": revealed_at,
            "comparison_id": comparison["comparison_id"],
            "comparison_path": str(comparison_path),
        },
    )
    return {**comparison, "comparison_path": str(comparison_path)}


def replay_speaker_confidence_calibration(
    campaign_id: str,
    *,
    baseline_ids: list[str],
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Replay host confidence calibration without changing sealed predictions."""
    if approval_token != REPLAY_SPEAKER_CONFIDENCE_CALIBRATION_TOKEN:
        raise ValueError(
            "Confidence calibration replay requires approval token "
            f"{REPLAY_SPEAKER_CONFIDENCE_CALIBRATION_TOKEN}."
        )
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    campaign_dir = _campaign_dir(selected_runtime_root, campaign_id)
    selected_baseline_ids = list(dict.fromkeys(str(value) for value in baseline_ids))
    if not selected_baseline_ids:
        raise ValueError("Confidence calibration replay requires baseline IDs.")

    metrics = Counter(
        {
            "reviewed_person_labels": 0,
            "top_proposal_correct": 0,
            "before_high_or_very_high_correct": 0,
            "before_high_or_very_high_wrong": 0,
            "after_high_or_very_high_correct": 0,
            "after_high_or_very_high_wrong": 0,
            "calibrated_labels": 0,
            "calibrated_correct_labels": 0,
            "calibrated_wrong_labels": 0,
        }
    )
    validation_metrics = Counter()
    cohort_receipts = []
    case_receipts = []
    high_bands = {"high", "very_high"}

    for baseline_id in selected_baseline_ids:
        baseline_path = _blind_baseline_path(
            selected_runtime_root,
            campaign_id,
            baseline_id,
        )
        comparison_path = baseline_path.parent / "comparison.json"
        if not baseline_path.is_file() or not comparison_path.is_file():
            raise ValueError(
                f"Calibration replay requires a completed comparison: {baseline_id}."
            )
        baseline = _read_json(baseline_path)
        comparison = _read_json(comparison_path)
        if (
            baseline.get("status") != "comparison_complete"
            or comparison.get("status") != "comparison_complete"
            or comparison.get("baseline_id") != baseline_id
        ):
            raise ValueError(
                f"Calibration replay comparison is incomplete: {baseline_id}."
            )
        comparison_cases = {
            str(case.get("document_id") or ""): case
            for case in comparison.get("cases") or []
            if isinstance(case, dict)
        }
        cohort_metrics = Counter()
        for key, value in (
            (comparison.get("metrics") or {}).get("validation") or {}
        ).items():
            validation_metrics[str(key)] += int(value or 0)

        for baseline_case in baseline.get("cases") or []:
            if not isinstance(baseline_case, dict):
                continue
            document_id = str(baseline_case.get("document_id") or "")
            comparison_case = comparison_cases.get(document_id, {})
            if comparison_case.get("evaluation_excluded"):
                continue
            prediction_record = _read_json(
                Path(str(baseline_case.get("prediction_path") or ""))
            )
            prediction = (
                prediction_record.get("prediction")
                if isinstance(prediction_record.get("prediction"), dict)
                else {}
            )
            proposals = [
                proposal
                for proposal in prediction.get("proposals") or []
                if isinstance(proposal, dict)
            ]
            for label_result in comparison_case.get("speaker_labels") or []:
                if (
                    not isinstance(label_result, dict)
                    or label_result.get("gold_outcome") != "person"
                ):
                    continue
                label = str(label_result.get("speaker_label") or "")
                label_proposals = [
                    proposal
                    for proposal in proposals
                    if label
                    in [
                        str(value)
                        for value in proposal.get("speaker_labels") or []
                    ]
                ]
                top_proposal = label_proposals[0] if label_proposals else {}
                before_band = str(
                    label_result.get("top_confidence_band") or ""
                )
                calibrated = (
                    speaker_identity_preprocess.calibrate_speaker_identity_confidence(
                        top_proposal,
                        (
                            top_proposal.get("confidence")
                            if isinstance(top_proposal.get("confidence"), dict)
                            else {}
                        ),
                    )
                )
                after_band = str(calibrated.get("band") or "")
                correct = bool(label_result.get("top_proposal_correct"))
                applied = bool(
                    (calibrated.get("calibration") or {}).get("applied")
                )
                metrics["reviewed_person_labels"] += 1
                metrics["top_proposal_correct"] += int(correct)
                metrics["before_high_or_very_high_correct"] += int(
                    correct and before_band in high_bands
                )
                metrics["before_high_or_very_high_wrong"] += int(
                    not correct and before_band in high_bands
                )
                metrics["after_high_or_very_high_correct"] += int(
                    correct and after_band in high_bands
                )
                metrics["after_high_or_very_high_wrong"] += int(
                    not correct and after_band in high_bands
                )
                metrics["calibrated_labels"] += int(applied)
                metrics["calibrated_correct_labels"] += int(applied and correct)
                metrics["calibrated_wrong_labels"] += int(
                    applied and not correct
                )
                cohort_metrics["reviewed_person_labels"] += 1
                cohort_metrics["top_proposal_correct"] += int(correct)
                cohort_metrics["before_high_or_very_high_wrong"] += int(
                    not correct and before_band in high_bands
                )
                cohort_metrics["after_high_or_very_high_wrong"] += int(
                    not correct and after_band in high_bands
                )
                case_receipts.append(
                    {
                        "baseline_id": baseline_id,
                        "document_id": document_id,
                        "speaker_label": label,
                        "top_proposal_correct": correct,
                        "before_band": before_band,
                        "after_band": after_band,
                        "calibration_applied": applied,
                        "calibration_reasons": (
                            calibrated.get("calibration") or {}
                        ).get("reasons")
                        or [],
                    }
                )
        cohort_receipts.append(
            {
                "baseline_id": baseline_id,
                "comparison_id": comparison.get("comparison_id"),
                "metrics": dict(cohort_metrics),
            }
        )

    replay_id = f"calibration-replay-{uuid4()}"
    receipt = {
        "schema_version": (
            "transcribe-audio.speaker-confidence-calibration-replay.v1"
        ),
        "replay_id": replay_id,
        "campaign_id": campaign_id,
        "baseline_ids": selected_baseline_ids,
        "status": "complete",
        "created_at": _utc_now(),
        "algorithm": _repository_state(),
        "calibration_version": (
            speaker_identity_preprocess.SPEAKER_CONFIDENCE_CALIBRATION_VERSION
        ),
        "rubric_version": (
            speaker_identity_preprocess.EVIDENCE_RUBRICS["speaker_identity"][
                "version"
            ]
        ),
        "metrics": {
            **dict(metrics),
            "validation": dict(validation_metrics),
        },
        "cohorts": cohort_receipts,
        "cases": case_receipts,
        "source_predictions_mutated": False,
        "will_perform_external_write": False,
    }
    receipt_path = (
        campaign_dir
        / "calibration-replays"
        / f"{replay_id}.json"
    )
    _write_private_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def record_refinement_decision(
    campaign_id: str,
    *,
    baseline_id: str,
    decision: str,
    target_failure_class: str,
    rationale: str,
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    """Record one immutable accept/reject decision for a completed refinement."""
    if approval_token != RECORD_REFINEMENT_DECISION_TOKEN:
        raise ValueError(
            "Refinement decision requires approval token "
            f"{RECORD_REFINEMENT_DECISION_TOKEN}."
        )
    if decision not in {"accepted", "rejected"}:
        raise ValueError("Refinement decision must be accepted or rejected.")
    if not target_failure_class.strip() or not rationale.strip():
        raise ValueError(
            "Refinement decision requires a target failure class and rationale."
        )
    selected_runtime_root = (
        runtime_root or DEFAULT_CAMPAIGN_ROOT
    ).expanduser()
    baseline_path = _blind_baseline_path(
        selected_runtime_root,
        campaign_id,
        baseline_id,
    )
    baseline = _read_json(baseline_path)
    if (
        baseline.get("run_kind") != "refinement"
        or baseline.get("status") != "comparison_complete"
    ):
        raise ValueError(
            "Refinement decision requires a completed refinement comparison."
        )
    parent_baseline_id = str(baseline.get("parent_baseline_id") or "")
    parent_path = _blind_baseline_path(
        selected_runtime_root,
        campaign_id,
        parent_baseline_id,
    )
    parent = _read_json(parent_path)
    if parent.get("status") not in {
        "comparison_complete",
        "refinement_accepted",
        "refinement_rejected",
    }:
        raise ValueError("Parent baseline comparison is not complete.")
    comparison_path = Path(str(baseline.get("comparison_path") or ""))
    parent_comparison_path = Path(str(parent.get("comparison_path") or ""))
    comparison = _read_json(comparison_path)
    parent_comparison = _read_json(parent_comparison_path)

    def metric(payload: dict[str, Any], section: str, name: str) -> int:
        metrics = payload.get("metrics")
        selected = metrics.get(section) if isinstance(metrics, dict) else {}
        value = selected.get(name) if isinstance(selected, dict) else 0
        return int(value or 0)

    selected_metrics = (
        ("validation", "model_output_rejected"),
        ("validation", "stage_clue_discovery_validation"),
        ("validation", "stage_identity_evaluation_validation"),
        ("calendar_association", "exact"),
        ("calendar_association", "high_or_very_high_wrong"),
        ("speaker_identity", "top_proposal_correct"),
        ("speaker_identity", "correct_person_present"),
        ("speaker_identity", "high_or_very_high_wrong"),
    )
    metric_deltas = {
        f"{section}.{name}": (
            metric(comparison, section, name)
            - metric(parent_comparison, section, name)
        )
        for section, name in selected_metrics
    }
    decided_at = _utc_now()
    record = {
        "schema_version": (
            "transcribe-audio.speaker-evaluation-refinement-decision.v1"
        ),
        "decision_id": f"refinement-decision-{uuid4()}",
        "campaign_id": campaign_id,
        "baseline_id": baseline_id,
        "parent_baseline_id": parent_baseline_id,
        "decision": decision,
        "target_failure_class": target_failure_class.strip(),
        "hypothesis": str(baseline.get("hypothesis") or ""),
        "evidence_mode": str(baseline.get("evidence_mode") or ""),
        "rationale": rationale.strip(),
        "metric_deltas": metric_deltas,
        "decided_at": decided_at,
        "will_apply_assignments": False,
        "will_perform_external_write": False,
    }
    decision_path = baseline_path.parent / "refinement-decision.json"
    if decision_path.exists():
        raise ValueError("Refinement decision already exists.")
    _write_private_json(decision_path, record)
    _write_private_json(
        baseline_path,
        {
            **baseline,
            "status": f"refinement_{decision}",
            "refinement_decision_id": record["decision_id"],
            "refinement_decision_path": str(decision_path),
        },
    )
    return {**record, "decision_path": str(decision_path)}


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
