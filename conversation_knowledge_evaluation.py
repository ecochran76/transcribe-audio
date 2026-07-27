"""Private chronological readiness gate for conversation-knowledge evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5


FREEZE_SCHEMA_VERSION = (
    "transcribe-audio.conversation-knowledge-evaluation-freeze.v1"
)
DECISION_SCHEMA_VERSION = (
    "transcribe-audio.conversation-knowledge-evaluation-decision.v1"
)
FREEZE_EVALUATION_TOKEN = "FREEZE_CONVERSATION_KNOWLEDGE_EVALUATION"
RECORD_DECISION_TOKEN = "RECORD_CONVERSATION_KNOWLEDGE_EVALUATION_DECISION"
DEFAULT_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
DEFAULT_EVALUATION_ROOT = Path(
    "~/.local/state/transcribe-audio/conversation-knowledge-evaluations"
)
EVIDENCE_FAMILIES = (
    "calendar_only",
    "transcript_only",
    "provenance_only",
    "accumulated_history",
    "combined",
)
_EVALUATION_NAMESPACE = UUID("250051ba-f710-46ee-8f1f-abd15205843c")
_DECISIONS = {"accept", "refine", "reject", "stop"}
_GATE_STATUSES = {"pass", "fail", "pending", "not_run"}
_FAMILY_STATUSES = {"complete", "partial", "not_run"}
_SENSITIVE_KEYS = {
    "email",
    "name",
    "snippet",
    "speaker_outcomes",
    "transcript",
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Evaluation source is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation source must be a JSON object: {path}")
    return value


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
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
                value,
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


def _repository_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() or "unavailable",
        "dirty_tree": bool(status.stdout.strip()),
    }


def _campaign_dir(campaign_root: Path, campaign_id: str) -> Path:
    if not re.fullmatch(r"campaign-[a-f0-9]{20}", campaign_id):
        raise ValueError("Campaign ID is invalid.")
    return campaign_root.expanduser() / campaign_id


def _freeze_dir(evaluation_root: Path, freeze_id: str) -> Path:
    if not re.fullmatch(r"evaluation-[0-9a-f-]{36}", freeze_id):
        raise ValueError("Evaluation freeze ID is invalid.")
    return evaluation_root.expanduser() / freeze_id


def _private_evaluation_root(evaluation_root: Path) -> Path:
    root = evaluation_root.expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    return root


def _latest_gold_rank(index: dict[str, Any]) -> int:
    records = index.get("records")
    if not isinstance(records, list):
        raise ValueError("Campaign gold index records must be a list.")
    return max(
        (
            int(record.get("chronological_rank") or 0)
            for record in records
            if isinstance(record, dict)
        ),
        default=0,
    )


def freeze_chronological_evaluation(
    campaign_id: str,
    *,
    campaign_root: Path = DEFAULT_CAMPAIGN_ROOT,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    cohort_size: int = 10,
    approval_token: str,
) -> dict[str, Any]:
    """Freeze the next unseen chronological cohort without reading gold bodies."""
    if approval_token != FREEZE_EVALUATION_TOKEN:
        raise ValueError(
            f"Evaluation freeze requires approval token {FREEZE_EVALUATION_TOKEN}."
        )
    if cohort_size < 1 or cohort_size > 100:
        raise ValueError("Evaluation cohort size must be between 1 and 100.")
    campaign_dir = _campaign_dir(campaign_root, campaign_id)
    manifest_path = campaign_dir / "manifest.json"
    gold_index_path = campaign_dir / "gold" / "index.json"
    manifest = _read_json(manifest_path)
    gold_index = _read_json(gold_index_path)
    start_rank = _latest_gold_rank(gold_index)
    ordered = sorted(
        (
            dict(item)
            for item in manifest.get("items") or []
            if isinstance(item, dict)
            and int(item.get("chronological_rank") or 0) > start_rank
        ),
        key=lambda item: (
            int(item.get("chronological_rank") or 0),
            str(item.get("document_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    for item in ordered:
        disposition = str(item.get("disposition") or "")
        if disposition != "needs_operator_classification":
            excluded_counts[disposition or "unclassified"] += 1
            continue
        if len(selected) >= cohort_size:
            break
        document_id = str(item.get("document_id") or "")
        artifact_sha256 = str(item.get("artifact_sha256") or "")
        if not document_id or not re.fullmatch(r"[a-f0-9]{64}", artifact_sha256):
            raise ValueError("Evaluation candidate lacks a valid artifact identity.")
        selected.append(
            {
                "chronological_rank": int(item["chronological_rank"]),
                "document_id": document_id,
                "artifact_sha256": artifact_sha256,
                "duplicate_cluster_id": str(
                    item.get("duplicate_cluster_id") or ""
                ),
                "utterance_count": int(item.get("utterance_count") or 0),
                "prediction_status": "not_started",
                "ground_truth_status": "not_reviewed",
            }
        )
    if len(selected) != cohort_size:
        raise ValueError(
            f"Evaluation cohort needs {cohort_size} unseen cases; found "
            f"{len(selected)}."
        )
    source_hashes = {
        "campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "gold_index_sha256": hashlib.sha256(
            gold_index_path.read_bytes()
        ).hexdigest(),
    }
    stable_identity = {
        "campaign_id": campaign_id,
        "manifest_id": str(manifest.get("manifest_id") or ""),
        "start_after_chronological_rank": start_rank,
        "cohort_size": cohort_size,
        "cases": selected,
        "source_hashes": source_hashes,
        "evidence_families": list(EVIDENCE_FAMILIES),
    }
    freeze_id = "evaluation-" + str(
        uuid5(_EVALUATION_NAMESPACE, _canonical_hash(stable_identity))
    )
    freeze_path = (
        _freeze_dir(_private_evaluation_root(evaluation_root), freeze_id)
        / "freeze.json"
    )
    if freeze_path.exists():
        return {**_read_json(freeze_path), "freeze_path": str(freeze_path)}
    frozen = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        **stable_identity,
        "excluded_disposition_counts": dict(sorted(excluded_counts.items())),
        "repository": _repository_state(),
        "status": "frozen_pending_readiness",
        "prediction_visibility": "unseen",
        "gold_content_included": False,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
        "frozen_at": _utc_now(),
    }
    _write_private_json(freeze_path, frozen)
    return {**frozen, "freeze_path": str(freeze_path)}


def _validate_aggregate(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key).strip()
        if (
            not normalized_key
            or any(term in normalized_key.casefold() for term in _SENSITIVE_KEYS)
        ):
            raise ValueError(f"{field_name} contains a prohibited key.")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError(f"{field_name} values must be aggregate scalars.")
        result[normalized_key] = item
    return result


def record_readiness_decision(
    freeze_id: str,
    *,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    decision: str,
    reason_codes: tuple[str, ...],
    gate_results: tuple[dict[str, str], ...],
    historical_metrics: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    family_results: dict[str, dict[str, Any]],
    successor_scope: tuple[str, ...],
    approval_token: str,
) -> dict[str, Any]:
    """Record one immutable aggregate decision without opening the unseen cases."""
    if approval_token != RECORD_DECISION_TOKEN:
        raise ValueError(
            f"Evaluation decision requires approval token {RECORD_DECISION_TOKEN}."
        )
    if decision not in _DECISIONS:
        raise ValueError(f"Unsupported evaluation decision: {decision}.")
    freeze_dir = _freeze_dir(
        _private_evaluation_root(evaluation_root),
        freeze_id,
    )
    freeze = _read_json(freeze_dir / "freeze.json")
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("Evaluation freeze schema is invalid.")
    gates: list[dict[str, str]] = []
    for gate in gate_results:
        name = str(gate.get("gate") or "").strip()
        status = str(gate.get("status") or "").strip()
        evidence = str(gate.get("evidence") or "").strip()[:500]
        if not name or status not in _GATE_STATUSES:
            raise ValueError("Evaluation gate result is invalid.")
        gates.append({"gate": name, "status": status, "evidence": evidence})
    families: dict[str, dict[str, Any]] = {}
    for family in EVIDENCE_FAMILIES:
        result = family_results.get(family)
        if not isinstance(result, dict):
            raise ValueError(f"Missing evidence-family result: {family}.")
        status = str(result.get("status") or "")
        if status not in _FAMILY_STATUSES:
            raise ValueError(f"Invalid evidence-family status: {family}.")
        families[family] = _validate_aggregate(
            result,
            field_name=f"family_results.{family}",
        )
    decision_payload = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "campaign_id": freeze["campaign_id"],
        "decision": decision,
        "reason_codes": [str(value) for value in reason_codes if str(value)],
        "gate_results": gates,
        "historical_metrics": _validate_aggregate(
            historical_metrics,
            field_name="historical_metrics",
        ),
        "retrieval_metrics": _validate_aggregate(
            retrieval_metrics,
            field_name="retrieval_metrics",
        ),
        "family_results": families,
        "successor_scope": [
            str(value)[:500] for value in successor_scope if str(value).strip()
        ],
        "cohort_prediction_status": "not_started",
        "cohort_remains_unseen": True,
        "automatic_confirmation_enabled": False,
        "database_authority_enabled": False,
        "sidecar_authority_retained": True,
        "will_execute_app_intelligence": False,
        "will_perform_external_write": False,
    }
    decision_payload["content_hash"] = _canonical_hash(decision_payload)
    decision_payload["recorded_at"] = _utc_now()
    decision_path = freeze_dir / "decision.json"
    if decision_path.exists():
        existing = _read_json(decision_path)
        if existing.get("content_hash") != decision_payload["content_hash"]:
            raise ValueError(f"Immutable decision conflict: {decision_path}.")
        return {**existing, "decision_path": str(decision_path)}
    _write_private_json(decision_path, decision_payload)
    return {**decision_payload, "decision_path": str(decision_path)}
