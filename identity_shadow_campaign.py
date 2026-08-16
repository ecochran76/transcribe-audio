"""Deterministic, zero-effect controller for Plan 0072 A6 shadow campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from acoustic_audio_derivatives import (
    AudioDerivativeError,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)
from identity_review_workflow import IdentityReviewWorkflow, IdentityReviewWorkflowError


PREVIEW_SCHEMA = "transcribe-audio.identity-shadow-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.identity-shadow-manifest.v1"
ACTIVATION_RECEIPT_SCHEMA = "transcribe-audio.identity-shadow-activation-receipt.v1"
CASE_RECEIPT_SCHEMA = "transcribe-audio.identity-shadow-case-receipt.v1"
ARRIVAL_REGISTRATION_SCHEMA = "transcribe-audio.identity-shadow-arrival-registration.v1"
ACTIVATE_TOKEN = "ACTIVATE_PLAN_0072_A6_SHADOW"
FINALIZE_TOKEN = "FINALIZE_PLAN_0072_A6_SHADOW"
SCORECARD_RECEIPT_SCHEMA = "transcribe-audio.identity-shadow-window-scorecard.v1"
TERMINAL_RECEIPT_SCHEMA = "transcribe-audio.identity-shadow-terminal-receipt.v1"
HISTORICAL_LIMIT = 25
NEW_ARRIVAL_WINDOW_DAYS = 7
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OPAQUE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
DISPOSITIONS = {
    "eligible",
    "ineligible_artifact_unstable",
    "ineligible_missing_hash",
    "ineligible_policy_exclusion",
}
FORBIDDEN_PRIVATE_FIELDS = {
    "provider_payload",
    "raw_transcript",
    "source_path",
    "stored_path",
    "transcript_text",
}
EFFECT_POLICY = {
    "accepted_identity_effect_count": 0,
    "accepted_profile_effect_count": 0,
    "provider_write_count": 0,
    "raw_deletion_count": 0,
}


class IdentityShadowCampaignError(ValueError):
    """Raised when a shadow campaign cannot preserve its frozen bounds."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: Any, *, field: str) -> tuple[str, datetime]:
    text = str(value or "")
    if not text.endswith("Z"):
        raise IdentityShadowCampaignError(f"{field} must be an exact UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise IdentityShadowCampaignError(f"{field} must be an exact UTC timestamp.") from exc
    if parsed.tzinfo != timezone.utc:
        raise IdentityShadowCampaignError(f"{field} must be an exact UTC timestamp.")
    normalized = parsed.isoformat().replace("+00:00", "Z")
    if normalized != text:
        raise IdentityShadowCampaignError(f"{field} must be normalized.")
    return text, parsed


def _case_descriptor(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], datetime, datetime]:
    forbidden = sorted(set(candidate).intersection(FORBIDDEN_PRIVATE_FIELDS))
    if forbidden:
        raise IdentityShadowCampaignError(
            f"Candidate contains private payload field: {forbidden[0]}."
        )
    required = (
        "conversation_id",
        "recording_id",
        "original_recording_filename",
        "source_artifact_sha256",
        "source_media_sha256",
        "conversation_at",
        "artifact_stabilized_at",
        "cohort",
        "eligible",
        "disposition",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise IdentityShadowCampaignError(
            f"Candidate is missing required field: {missing[0]}."
        )
    cohort = str(candidate["cohort"])
    if cohort not in {"historical", "new_arrival"}:
        raise IdentityShadowCampaignError("Candidate cohort is invalid.")
    if not isinstance(candidate["eligible"], bool):
        raise IdentityShadowCampaignError("Candidate eligibility must be boolean.")
    descriptor: dict[str, Any] = {}
    for field in ("conversation_id", "recording_id", "original_recording_filename"):
        value = str(candidate[field]).strip()
        if not value:
            raise IdentityShadowCampaignError(f"Candidate {field} must be non-empty.")
        descriptor[field] = value
    for field in ("conversation_id", "recording_id"):
        if not OPAQUE_ID_RE.fullmatch(descriptor[field]):
            raise IdentityShadowCampaignError(f"Candidate {field} must be opaque.")
    filename = descriptor["original_recording_filename"]
    if len(filename) > 255 or "/" in filename or "\\" in filename:
        raise IdentityShadowCampaignError(
            "original_recording_filename must not contain a filesystem path."
        )
    for field in ("source_artifact_sha256", "source_media_sha256"):
        value = str(candidate[field])
        if not SHA256_RE.fullmatch(value):
            raise IdentityShadowCampaignError(f"Candidate {field} is invalid.")
        descriptor[field] = value
    conversation_at, conversation_time = _utc_timestamp(
        candidate["conversation_at"], field="conversation_at"
    )
    stabilized_at, stabilized_time = _utc_timestamp(
        candidate["artifact_stabilized_at"], field="artifact_stabilized_at"
    )
    disposition = str(candidate["disposition"])
    if disposition not in DISPOSITIONS or (
        bool(candidate["eligible"]) != (disposition == "eligible")
    ):
        raise IdentityShadowCampaignError("Candidate disposition is invalid.")
    descriptor.update(
        {
            "conversation_at": conversation_at,
            "artifact_stabilized_at": stabilized_at,
            "cohort": cohort,
            "eligible": candidate["eligible"],
            "disposition": disposition,
        }
    )
    descriptor["case_id"] = f"shadow-case-{_hash(descriptor)[:24]}"
    return descriptor, stabilized_time, conversation_time


def preview_shadow_campaign(
    candidates: Iterable[Mapping[str, Any]],
    *,
    activated_at: str,
) -> dict[str, Any]:
    """Freeze a content-addressed A6 selection without touching private sources."""
    activated_text, activated = _utc_timestamp(activated_at, field="activated_at")
    window_end = activated + timedelta(days=NEW_ARRIVAL_WINDOW_DAYS)
    normalized = [_case_descriptor(candidate) for candidate in candidates]
    ids = [item[0]["conversation_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise IdentityShadowCampaignError("Candidate conversation IDs must be unique.")

    historical = sorted(
        (item for item in normalized if item[0]["cohort"] == "historical"),
        key=lambda item: (item[2], item[0]["conversation_id"]),
    )
    eligible_historical = [item for item in historical if item[0]["eligible"]]
    selected_historical = eligible_historical[:HISTORICAL_LIMIT]
    new_arrivals = sorted(
        (
            item
            for item in normalized
            if item[0]["cohort"] == "new_arrival"
            and item[0]["eligible"]
            and activated <= item[1] < window_end
        ),
        key=lambda item: (item[1], item[0]["conversation_id"]),
    )
    preview: dict[str, Any] = {
        "schema_version": PREVIEW_SCHEMA,
        "operation_mode": "shadow",
        "activated_at": activated_text,
        "historical_cases": [item[0] for item in selected_historical],
        "historical_inventory": {
            "candidate_count": len(historical),
            "selected_count": len(selected_historical),
            "deferred_count": max(0, len(eligible_historical) - HISTORICAL_LIMIT),
            "ineligible_count": len(historical) - len(eligible_historical),
        },
        "new_arrival_cases": [item[0] for item in new_arrivals],
        "new_arrival_window": {
            "starts_at": activated_text,
            "ends_at": window_end.isoformat().replace("+00:00", "Z"),
            "duration_days": NEW_ARRIVAL_WINDOW_DAYS,
        },
        "effect_policy": dict(EFFECT_POLICY),
    }
    preview["campaign_id"] = f"identity-shadow-{_hash(preview)[:24]}"
    return preview


def _validate_preview(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(preview)
    if value.get("schema_version") != PREVIEW_SCHEMA:
        raise IdentityShadowCampaignError("Shadow preview schema is invalid.")
    if value.get("operation_mode") != "shadow":
        raise IdentityShadowCampaignError("Shadow preview operation mode is invalid.")
    if value.get("effect_policy") != EFFECT_POLICY:
        raise IdentityShadowCampaignError("Shadow preview effect policy is invalid.")
    historical = value.get("historical_cases")
    if not isinstance(historical, list) or len(historical) != HISTORICAL_LIMIT:
        raise IdentityShadowCampaignError(
            f"Shadow activation requires exactly {HISTORICAL_LIMIT} historical cases."
        )
    campaign_id = value.pop("campaign_id", None)
    expected_campaign_id = f"identity-shadow-{_hash(value)[:24]}"
    value["campaign_id"] = campaign_id
    if campaign_id != expected_campaign_id:
        raise IdentityShadowCampaignError("Shadow preview campaign binding is invalid.")
    return value


def activate_shadow_campaign(
    preview: Mapping[str, Any],
    *,
    expected_preview_sha256: str,
    reviewed_at: str,
    runtime_root: Path,
    approval_token: str,
) -> dict[str, Any]:
    """Activate an operator-reviewed preview into a private immutable ledger."""
    if approval_token != ACTIVATE_TOKEN:
        raise IdentityShadowCampaignError(
            f"Shadow activation requires approval token {ACTIVATE_TOKEN}."
        )
    value = dict(preview)
    actual_preview_sha256 = _hash(value)
    if actual_preview_sha256 != expected_preview_sha256:
        raise IdentityShadowCampaignError("Reviewed preview hash does not match.")
    value = _validate_preview(value)
    reviewed_text, _ = _utc_timestamp(reviewed_at, field="reviewed_at")
    root = runtime_root.expanduser().absolute()
    run_dir = root / str(value["campaign_id"])
    manifest_path = run_dir / "manifest.json"
    receipt_path = run_dir / "activation-receipt.json"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "campaign_id": value["campaign_id"],
        "status": "active_shadow_only",
        "preview_sha256": actual_preview_sha256,
        "preview": value,
        "reviewed_at": reviewed_text,
        "effect_policy": dict(EFFECT_POLICY),
    }
    try:
        ensure_private_tree(root, run_dir)
        write_immutable_private_json(manifest_path, manifest)
        receipt = {
            "schema_version": ACTIVATION_RECEIPT_SCHEMA,
            "campaign_id": value["campaign_id"],
            "status": "active_shadow_only",
            "preview_sha256": actual_preview_sha256,
            "manifest_sha256": _sha256_file(manifest_path),
            "historical_case_count": len(value["historical_cases"]),
            "new_arrival_window": value["new_arrival_window"],
            "effect_policy": dict(EFFECT_POLICY),
            "reviewed_at": reviewed_text,
        }
        stored = write_immutable_private_json(receipt_path, receipt)
    except (AudioDerivativeError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    return {
        **stored,
        "manifest_path": manifest_path,
        "activation_receipt_path": receipt_path,
    }


def _active_campaign(
    campaign_id: str, *, runtime_root: Path
) -> tuple[Path, dict[str, Any]]:
    root = runtime_root.expanduser().absolute()
    run_dir = root / campaign_id
    manifest_path = run_dir / "manifest.json"
    receipt_path = run_dir / "activation-receipt.json"
    try:
        require_private_file(manifest_path, root)
        require_private_file(receipt_path, root)
        manifest = read_private_object(manifest_path)
        activation = read_private_object(receipt_path)
    except (AudioDerivativeError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("status") != "active_shadow_only"
        or activation.get("schema_version") != ACTIVATION_RECEIPT_SCHEMA
        or activation.get("campaign_id") != campaign_id
        or activation.get("status") != "active_shadow_only"
        or activation.get("manifest_sha256") != _sha256_file(manifest_path)
        or manifest.get("preview_sha256") != _hash(manifest.get("preview"))
        or activation.get("preview_sha256") != manifest.get("preview_sha256")
        or manifest.get("effect_policy") != EFFECT_POLICY
        or activation.get("effect_policy") != EFFECT_POLICY
    ):
        raise IdentityShadowCampaignError("Active shadow campaign binding is invalid.")
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise IdentityShadowCampaignError("Active shadow preview is unavailable.")
    _validate_preview(preview)
    return run_dir, manifest


def _nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IdentityShadowCampaignError(f"{field} must be a non-negative integer.")
    return value


def _require_open_campaign(run_dir: Path) -> None:
    if (run_dir / "window-scorecard.json").exists() or (
        run_dir / "terminal-receipt.json"
    ).exists():
        raise IdentityShadowCampaignError("Shadow campaign is already closed to new cases.")


def register_new_arrival(
    campaign_id: str,
    candidate: Mapping[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Append one stabilized arrival that falls inside the frozen A6 window."""
    run_dir, manifest = _active_campaign(campaign_id, runtime_root=runtime_root)
    _require_open_campaign(run_dir)
    descriptor, stabilized_at, _ = _case_descriptor(candidate)
    if descriptor["cohort"] != "new_arrival" or not descriptor["eligible"]:
        raise IdentityShadowCampaignError("New arrival registration requires an eligible arrival.")
    window = manifest["preview"]["new_arrival_window"]
    _, starts_at = _utc_timestamp(window["starts_at"], field="new_arrival_window.starts_at")
    _, ends_at = _utc_timestamp(window["ends_at"], field="new_arrival_window.ends_at")
    if not starts_at <= stabilized_at < ends_at:
        raise IdentityShadowCampaignError("New arrival is outside the frozen window.")
    frozen_cases = [
        *manifest["preview"]["historical_cases"],
        *manifest["preview"]["new_arrival_cases"],
    ]
    if descriptor["conversation_id"] in {
        item["conversation_id"] for item in frozen_cases
    }:
        raise IdentityShadowCampaignError("New arrival duplicates a frozen conversation.")
    existing_arrivals, _ = _registered_arrivals(
        run_dir,
        root=runtime_root.expanduser().absolute(),
        campaign_id=campaign_id,
    )
    for existing in existing_arrivals.values():
        if existing["conversation_id"] == descriptor["conversation_id"]:
            if existing != descriptor:
                raise IdentityShadowCampaignError(
                    "New arrival conflicts with an existing registration."
                )
            break
    receipt = {
        "schema_version": ARRIVAL_REGISTRATION_SCHEMA,
        "campaign_id": campaign_id,
        "status": "registered_for_shadow",
        "case": descriptor,
        "effect_policy": dict(EFFECT_POLICY),
    }
    arrivals_dir = run_dir / "arrivals"
    receipt_path = arrivals_dir / f"{descriptor['case_id']}.json"
    root = runtime_root.expanduser().absolute()
    try:
        ensure_private_tree(root, arrivals_dir)
        stored = write_immutable_private_json(receipt_path, receipt)
    except (AudioDerivativeError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    return {**stored, "registration_path": receipt_path}


def record_shadow_case(
    campaign_id: str,
    result: Mapping[str, Any],
    *,
    runtime_root: Path,
    review_workflow: IdentityReviewWorkflow | None = None,
    evidence_supervisor: Any = None,
) -> dict[str, Any]:
    """Record one terminal case and optionally project its A5 review queue item."""
    run_dir, manifest = _active_campaign(campaign_id, runtime_root=runtime_root)
    _require_open_campaign(run_dir)
    preview = manifest["preview"]
    members = {
        item["case_id"]: item
        for item in [
            *preview["historical_cases"],
            *preview["new_arrival_cases"],
        ]
    }
    case_id = str(result.get("case_id") or "")
    source = members.get(case_id)
    if source is None:
        root = runtime_root.expanduser().absolute()
        registration_path = run_dir / "arrivals" / f"{case_id}.json"
        try:
            require_private_file(registration_path, root)
            registration = read_private_object(registration_path)
        except (AudioDerivativeError, OSError) as exc:
            raise IdentityShadowCampaignError(
                "Shadow case is not in the active manifest or arrival ledger."
            ) from exc
        source = registration.get("case")
        if (
            registration.get("schema_version") != ARRIVAL_REGISTRATION_SCHEMA
            or registration.get("campaign_id") != campaign_id
            or registration.get("status") != "registered_for_shadow"
            or registration.get("effect_policy") != EFFECT_POLICY
            or not isinstance(source, dict)
            or source.get("case_id") != case_id
        ):
            raise IdentityShadowCampaignError("Shadow arrival registration is invalid.")
    if result.get("effect_policy") != EFFECT_POLICY:
        raise IdentityShadowCampaignError("Shadow case must preserve the zero-effect policy.")
    for field in (
        "conversation_id",
        "recording_id",
        "source_artifact_sha256",
        "source_media_sha256",
    ):
        if result.get(field) != source[field]:
            raise IdentityShadowCampaignError(f"Shadow case {field} binding is invalid.")
    processing_run_id = str(result.get("processing_run_id") or "").strip()
    if not processing_run_id:
        raise IdentityShadowCampaignError("Shadow case processing_run_id is required.")
    status = str(result.get("status") or "")
    if status not in {"complete", "partial", "failed", "skipped"}:
        raise IdentityShadowCampaignError("Shadow case status is not terminal.")
    attempt_count = _nonnegative_int(result.get("attempt_count"), field="attempt_count")
    if attempt_count not in {1, 2}:
        raise IdentityShadowCampaignError("Shadow case allows at most two attempts.")
    provider_reads = result.get("provider_reads")
    if not isinstance(provider_reads, Mapping):
        raise IdentityShadowCampaignError("Shadow case provider_reads is invalid.")
    normalized_provider_reads = {
        field: _nonnegative_int(provider_reads.get(field), field=f"provider_reads.{field}")
        for field in ("succeeded", "failed", "transient_retries")
    }
    if normalized_provider_reads["transient_retries"] > 1:
        raise IdentityShadowCampaignError("Shadow case allows one transient provider retry.")
    if evidence_supervisor is None or not hasattr(evidence_supervisor, "load_run"):
        raise IdentityShadowCampaignError("Shadow case requires its A4 supervisor store.")
    try:
        supervisor = evidence_supervisor.load_run(processing_run_id)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise IdentityShadowCampaignError("A4 supervisor run is unavailable.") from exc
    if not isinstance(supervisor, Mapping):
        raise IdentityShadowCampaignError("Shadow case requires its A4 supervisor run.")
    supervisor_effects = supervisor.get("effect_counts")
    if (
        not isinstance(supervisor_effects, Mapping)
        or not supervisor_effects
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value != 0
            for value in supervisor_effects.values()
        )
    ):
        raise IdentityShadowCampaignError("A4 supervisor zero effects are required.")
    supervisor_retry_count = _nonnegative_int(
        supervisor.get("provider_retry_count"), field="supervisor.provider_retry_count"
    )
    supervisor_content_hash = str(supervisor.get("content_hash") or "")
    supervisor_event_id = str(supervisor.get("event_id") or "").strip()
    if (
        supervisor.get("run_id") != processing_run_id
        or supervisor.get("stage") != "complete"
        or supervisor.get("state") != "complete"
        or not SHA256_RE.fullmatch(supervisor_content_hash)
        or not supervisor_event_id
        or supervisor_retry_count != normalized_provider_reads["transient_retries"]
    ):
        raise IdentityShadowCampaignError("A4 supervisor run binding is invalid.")
    latency_ms = _nonnegative_int(result.get("latency_ms"), field="latency_ms")
    if not isinstance(result.get("duplicate_suppressed"), bool):
        raise IdentityShadowCampaignError("duplicate_suppressed must be boolean.")
    if result.get("knowledge_integrity") not in {"preserved", "not_applicable"}:
        raise IdentityShadowCampaignError("Shadow case knowledge integrity is invalid.")
    completed_at, _ = _utc_timestamp(result.get("completed_at"), field="completed_at")

    queue_item = result.get("queue_item")
    queue_item_id = ""
    queue_item_sha256 = ""
    if queue_item is not None:
        if not isinstance(queue_item, Mapping) or review_workflow is None:
            raise IdentityShadowCampaignError(
                "Queue projection requires an identity review workflow and queue item."
            )
        for field in (
            "conversation_id",
            "recording_id",
            "original_recording_filename",
            "source_artifact_sha256",
            "source_media_sha256",
        ):
            if queue_item.get(field) != source[field]:
                raise IdentityShadowCampaignError(f"Queue item {field} binding is invalid.")
        if queue_item.get("processing_run_id") != processing_run_id:
            raise IdentityShadowCampaignError("Queue item processing run binding is invalid.")
        queue_item_id = str(queue_item.get("queue_item_id") or "")
        queue_item_sha256 = _hash(dict(queue_item))

    receipt = {
        "schema_version": CASE_RECEIPT_SCHEMA,
        "campaign_id": campaign_id,
        "case_id": case_id,
        "cohort": source["cohort"],
        "conversation_id": source["conversation_id"],
        "recording_id": source["recording_id"],
        "original_recording_filename": source["original_recording_filename"],
        "source_artifact_sha256": source["source_artifact_sha256"],
        "source_media_sha256": source["source_media_sha256"],
        "processing_run_id": processing_run_id,
        "status": status,
        "attempt_count": attempt_count,
        "provider_reads": normalized_provider_reads,
        "supervisor_run_content_hash": supervisor_content_hash,
        "supervisor_terminal_event_id": supervisor_event_id,
        "supervisor_effect_counts": dict(supervisor_effects),
        "latency_ms": latency_ms,
        "duplicate_suppressed": result["duplicate_suppressed"],
        "knowledge_integrity": result["knowledge_integrity"],
        "queue_item_id": queue_item_id,
        "queue_item_sha256": queue_item_sha256,
        "queue_projection_verified": queue_item is not None,
        "effect_policy": dict(EFFECT_POLICY),
        "completed_at": completed_at,
    }
    cases_dir = run_dir / "cases"
    receipt_path = cases_dir / f"{case_id}.json"
    root = runtime_root.expanduser().absolute()
    try:
        ensure_private_tree(root, cases_dir)
        if receipt_path.exists():
            require_private_file(receipt_path, root)
            if read_private_object(receipt_path) != receipt:
                raise IdentityShadowCampaignError(
                    "Immutable shadow case receipt conflicts with the recorded result."
                )
        if queue_item is not None:
            review_workflow.project_queue_item(queue_item)
        stored = write_immutable_private_json(receipt_path, receipt)
    except (AudioDerivativeError, IdentityReviewWorkflowError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    return {**stored, "case_receipt_path": receipt_path}


def _registered_arrivals(
    run_dir: Path, *, root: Path, campaign_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    arrivals_dir = run_dir / "arrivals"
    if not arrivals_dir.exists():
        return {}, {}
    if (
        arrivals_dir.is_symlink()
        or not arrivals_dir.is_dir()
        or stat.S_IMODE(arrivals_dir.stat().st_mode) != 0o700
    ):
        raise IdentityShadowCampaignError("Shadow arrival directory is not private.")
    cases: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in sorted(arrivals_dir.iterdir()):
        if path.suffix != ".json":
            raise IdentityShadowCampaignError("Unexpected shadow arrival artifact.")
        try:
            require_private_file(path, root)
            registration = read_private_object(path)
        except (AudioDerivativeError, OSError) as exc:
            raise IdentityShadowCampaignError(str(exc)) from exc
        case = registration.get("case")
        if (
            registration.get("schema_version") != ARRIVAL_REGISTRATION_SCHEMA
            or registration.get("campaign_id") != campaign_id
            or registration.get("status") != "registered_for_shadow"
            or registration.get("effect_policy") != EFFECT_POLICY
            or not isinstance(case, dict)
            or path.stem != case.get("case_id")
        ):
            raise IdentityShadowCampaignError("Shadow arrival registration is invalid.")
        case_id = str(case["case_id"])
        if case_id in cases:
            raise IdentityShadowCampaignError("Shadow arrival case ID is duplicated.")
        cases[case_id] = case
        hashes[case_id] = _sha256_file(path)
    return cases, hashes


def _campaign_cases(
    run_dir: Path, manifest: Mapping[str, Any], *, root: Path, campaign_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    frozen = [
        *manifest["preview"]["historical_cases"],
        *manifest["preview"]["new_arrival_cases"],
    ]
    cases = {str(case["case_id"]): dict(case) for case in frozen}
    arrivals, registration_hashes = _registered_arrivals(
        run_dir, root=root, campaign_id=campaign_id
    )
    if set(cases).intersection(arrivals):
        raise IdentityShadowCampaignError("Shadow arrival duplicates a frozen case.")
    conversation_ids = [case["conversation_id"] for case in [*cases.values(), *arrivals.values()]]
    if len(conversation_ids) != len(set(conversation_ids)):
        raise IdentityShadowCampaignError("Shadow campaign conversation is duplicated.")
    cases.update(arrivals)
    return cases, registration_hashes


def _validate_evaluation_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "candidate_recall",
        "correctness",
        "calibration",
        "high_strength_errors",
        "abstention",
        "review_load",
        "workflow_usability",
    }
    if set(value) != required:
        raise IdentityShadowCampaignError("Shadow evaluation metric inventory is incomplete.")
    normalized: dict[str, Any] = {}
    for name in sorted(required):
        metric = value[name]
        if not isinstance(metric, Mapping):
            raise IdentityShadowCampaignError(f"Shadow evaluation metric {name} is invalid.")
        metric_value = dict(metric)
        status_value = metric_value.get("status")
        if status_value == "unavailable":
            if set(metric_value) != {"status", "reason"}:
                raise IdentityShadowCampaignError(
                    f"Unavailable shadow evaluation metric {name} has extra fields."
                )
            if not str(metric_value.get("reason") or "").strip():
                raise IdentityShadowCampaignError(
                    f"Unavailable shadow evaluation metric {name} needs a reason."
                )
        elif status_value == "measured":
            allowed_fields = {
                "status",
                "value",
                "denominator",
                "interval",
                "evaluation_version",
            }
            if not set(metric_value).issubset(allowed_fields):
                raise IdentityShadowCampaignError(
                    f"Measured shadow evaluation metric {name} has extra fields."
                )
            measured = metric_value.get("value")
            if not isinstance(measured, (int, float, bool)):
                raise IdentityShadowCampaignError(
                    f"Measured shadow evaluation metric {name} needs a value."
                )
            if "denominator" in metric_value:
                _nonnegative_int(
                    metric_value["denominator"], field=f"{name}.denominator"
                )
            if "interval" in metric_value:
                interval = metric_value["interval"]
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or any(
                        isinstance(bound, bool)
                        or not isinstance(bound, (int, float))
                        or not 0 <= float(bound) <= 1
                        for bound in interval
                    )
                    or float(interval[0]) > float(interval[1])
                ):
                    raise IdentityShadowCampaignError(
                        f"Measured shadow evaluation metric {name} interval is invalid."
                    )
            if "evaluation_version" in metric_value and not str(
                metric_value["evaluation_version"]
            ).strip():
                raise IdentityShadowCampaignError(
                    f"Measured shadow evaluation metric {name} version is invalid."
                )
        else:
            raise IdentityShadowCampaignError(
                f"Shadow evaluation metric {name} status is invalid."
            )
        normalized[name] = metric_value
    return normalized


def finalize_shadow_campaign(
    campaign_id: str,
    *,
    observed_through: str,
    finalized_at: str,
    evaluation_metrics: Mapping[str, Any],
    runtime_root: Path,
    approval_token: str,
) -> dict[str, Any]:
    """Close a fully observed A6 window into a replayable aggregate receipt."""
    if approval_token != FINALIZE_TOKEN:
        raise IdentityShadowCampaignError(
            f"Shadow finalization requires approval token {FINALIZE_TOKEN}."
        )
    run_dir, manifest = _active_campaign(campaign_id, runtime_root=runtime_root)
    observed_text, observed = _utc_timestamp(observed_through, field="observed_through")
    finalized_text, finalized = _utc_timestamp(finalized_at, field="finalized_at")
    _, window_end = _utc_timestamp(
        manifest["preview"]["new_arrival_window"]["ends_at"],
        field="new_arrival_window.ends_at",
    )
    if observed < window_end:
        raise IdentityShadowCampaignError("The seven-day window has not been fully observed.")
    if finalized < observed:
        raise IdentityShadowCampaignError("finalized_at cannot precede observed_through.")
    normalized_metrics = _validate_evaluation_metrics(evaluation_metrics)
    root = runtime_root.expanduser().absolute()
    cases, registration_hashes = _campaign_cases(
        run_dir, manifest, root=root, campaign_id=campaign_id
    )
    historical_count = sum(case["cohort"] == "historical" for case in cases.values())
    if historical_count != HISTORICAL_LIMIT:
        raise IdentityShadowCampaignError(
            f"Shadow finalization requires exactly {HISTORICAL_LIMIT} historical cases."
        )
    receipts: list[dict[str, Any]] = []
    receipt_hashes: dict[str, str] = {}
    cases_dir = run_dir / "cases"
    for case_id, source in sorted(cases.items()):
        path = cases_dir / f"{case_id}.json"
        try:
            require_private_file(path, root)
            receipt = read_private_object(path)
        except (AudioDerivativeError, OSError) as exc:
            raise IdentityShadowCampaignError(
                f"Shadow case has no terminal receipt: {case_id}."
            ) from exc
        if (
            receipt.get("schema_version") != CASE_RECEIPT_SCHEMA
            or receipt.get("campaign_id") != campaign_id
            or receipt.get("case_id") != case_id
            or receipt.get("conversation_id") != source["conversation_id"]
            or receipt.get("effect_policy") != EFFECT_POLICY
            or receipt.get("status") not in {"complete", "partial", "failed", "skipped"}
        ):
            raise IdentityShadowCampaignError("Shadow case receipt binding is invalid.")
        receipts.append(receipt)
        receipt_hashes[case_id] = _sha256_file(path)
    if cases_dir.exists():
        unexpected = {
            path.name for path in cases_dir.iterdir()
        } - {f"{case_id}.json" for case_id in cases}
        if unexpected:
            raise IdentityShadowCampaignError("Unexpected shadow case receipt exists.")

    total = len(receipts)
    processed = sum(receipt["status"] in {"complete", "partial"} for receipt in receipts)
    latencies = sorted(int(receipt["latency_ms"]) for receipt in receipts)
    status_counts = {
        status: sum(receipt["status"] == status for receipt in receipts)
        for status in ("complete", "partial", "failed", "skipped")
    }
    provider_yield = {
        field: sum(int(receipt["provider_reads"][field]) for receipt in receipts)
        for field in ("succeeded", "failed", "transient_retries")
    }
    scorecard = {
        "pipeline_yield": {
            "processed_count": processed,
            "total_count": total,
            "rate": round(processed / total, 6) if total else 0.0,
        },
        "status_counts": status_counts,
        "provider_yield": provider_yield,
        "latency_ms": {
            "minimum": latencies[0],
            "maximum": latencies[-1],
            "mean": round(sum(latencies) / total, 3),
            "p95": latencies[math.ceil(total * 0.95) - 1],
        },
        "duplicate_control": {
            "suppressed_count": sum(
                bool(receipt["duplicate_suppressed"]) for receipt in receipts
            ),
            "total_count": total,
        },
        "knowledge_integrity": {
            "preserved_count": sum(
                receipt["knowledge_integrity"] == "preserved" for receipt in receipts
            ),
            "violation_count": 0,
        },
        "queue_projection_count": sum(
            bool(receipt["queue_projection_verified"]) for receipt in receipts
        ),
    }
    pending_review = any(
        name != "calibration" and metric["status"] == "unavailable"
        for name, metric in normalized_metrics.items()
    )
    scorecard_path = run_dir / "window-scorecard.json"
    terminal_path = run_dir / "terminal-receipt.json"
    if pending_review and terminal_path.exists():
        raise IdentityShadowCampaignError(
            "Reviewed shadow terminal already exists; pending review cannot reopen it."
        )
    target_path = scorecard_path if pending_review else terminal_path
    predecessor_scorecard_sha256 = (
        _sha256_file(scorecard_path) if not pending_review and scorecard_path.exists() else ""
    )
    receipt = {
        "schema_version": (
            SCORECARD_RECEIPT_SCHEMA if pending_review else TERMINAL_RECEIPT_SCHEMA
        ),
        "campaign_id": campaign_id,
        "status": (
            "shadow_window_complete_pending_review"
            if pending_review
            else "shadow_window_complete_reviewed"
        ),
        "manifest_sha256": _sha256_file(run_dir / "manifest.json"),
        "observed_through": observed_text,
        "historical_case_count": historical_count,
        "new_arrival_case_count": total - historical_count,
        "case_receipt_sha256": receipt_hashes,
        "arrival_registration_sha256": registration_hashes,
        "scorecard": scorecard,
        "evaluation_metrics": normalized_metrics,
        "effect_policy": dict(EFFECT_POLICY),
        "predecessor_scorecard_sha256": predecessor_scorecard_sha256,
        "finalized_at": finalized_text,
    }
    try:
        write_immutable_private_json(target_path, receipt)
    except (AudioDerivativeError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    return replay_shadow_campaign(campaign_id, runtime_root=runtime_root)


def replay_shadow_campaign(
    campaign_id: str, *, runtime_root: Path
) -> dict[str, Any]:
    """Verify the terminal manifest, arrival ledger, and every case receipt."""
    run_dir, manifest = _active_campaign(campaign_id, runtime_root=runtime_root)
    root = runtime_root.expanduser().absolute()
    scorecard_path = run_dir / "window-scorecard.json"
    terminal_path = run_dir / "terminal-receipt.json"
    receipt_path = terminal_path if terminal_path.exists() else scorecard_path
    try:
        require_private_file(receipt_path, root)
        receipt = read_private_object(receipt_path)
    except (AudioDerivativeError, OSError) as exc:
        raise IdentityShadowCampaignError(str(exc)) from exc
    expected_schema = (
        TERMINAL_RECEIPT_SCHEMA
        if receipt_path == terminal_path
        else SCORECARD_RECEIPT_SCHEMA
    )
    expected_status = (
        "shadow_window_complete_reviewed"
        if receipt_path == terminal_path
        else "shadow_window_complete_pending_review"
    )
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("campaign_id") != campaign_id
        or receipt.get("status") != expected_status
        or receipt.get("effect_policy") != EFFECT_POLICY
        or receipt.get("manifest_sha256") != _sha256_file(run_dir / "manifest.json")
    ):
        raise IdentityShadowCampaignError("Shadow terminal receipt binding is invalid.")
    cases, registration_hashes = _campaign_cases(
        run_dir, manifest, root=root, campaign_id=campaign_id
    )
    if receipt.get("arrival_registration_sha256") != registration_hashes:
        raise IdentityShadowCampaignError("Shadow arrival ledger hash changed.")
    predecessor_sha256 = receipt.get("predecessor_scorecard_sha256")
    if receipt_path == terminal_path:
        if scorecard_path.exists():
            try:
                require_private_file(scorecard_path, root)
            except (AudioDerivativeError, OSError) as exc:
                raise IdentityShadowCampaignError(str(exc)) from exc
            if predecessor_sha256 != _sha256_file(scorecard_path):
                raise IdentityShadowCampaignError("Shadow scorecard predecessor changed.")
        elif predecessor_sha256 != "":
            raise IdentityShadowCampaignError("Shadow scorecard predecessor is unavailable.")
    elif predecessor_sha256 != "":
        raise IdentityShadowCampaignError("Pending scorecard cannot cite a predecessor.")
    expected_case_hashes = receipt.get("case_receipt_sha256")
    if not isinstance(expected_case_hashes, dict) or set(expected_case_hashes) != set(cases):
        raise IdentityShadowCampaignError("Shadow terminal case inventory is invalid.")
    for case_id, expected_sha256 in expected_case_hashes.items():
        path = run_dir / "cases" / f"{case_id}.json"
        try:
            require_private_file(path, root)
        except (AudioDerivativeError, OSError) as exc:
            raise IdentityShadowCampaignError(str(exc)) from exc
        if _sha256_file(path) != expected_sha256:
            raise IdentityShadowCampaignError("Shadow case receipt hash changed.")
    return {**receipt, "campaign_receipt_path": receipt_path}
