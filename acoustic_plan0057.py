from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import acoustic_generation3_recalibration as recalibration
import acoustic_generation4_acoustic_contract as acoustic_contract
import acoustic_generation5_e2 as generation5_e2
import acoustic_plan0056_execution as plan0056_execution
import acoustic_plan0056_pilot as plan0056_pilot
import acoustic_plan0056_runner as plan0056_runner
import transcript_store
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)
from acoustic_shadow_evidence import (
    ALLOWLISTED_SUBJECT_IDS,
    activate_shadow_batch,
    build_shadow_bundle,
    canonical_hash,
    load_for_review,
    publish_shadow_bundle,
)


AUTHORITY_SCHEMA = "transcribe-audio.plan0057-shadow-authority.v1"
AUTHORITY_MANIFEST_SCHEMA = "transcribe-audio.plan0057-shadow-authority-manifest.v1"
AUTHORITY_RECEIPT_SCHEMA = "transcribe-audio.plan0057-shadow-authority-receipt.v1"
AUTHORITY_REPLAY_SCHEMA = "transcribe-audio.plan0057-shadow-authority-replay.v1"
EXECUTION_AUTHORITY_SCHEMA = "transcribe-audio.plan0057-execution-authority.v1"
EXECUTION_AUTHORITY_MANIFEST_SCHEMA = (
    "transcribe-audio.plan0057-execution-authority-manifest.v1"
)
EXECUTION_AUTHORITY_RECEIPT_SCHEMA = (
    "transcribe-audio.plan0057-execution-authority-receipt.v1"
)
EXECUTION_AUTHORITY_REPLAY_SCHEMA = (
    "transcribe-audio.plan0057-execution-authority-replay.v1"
)
EXECUTION_SCHEMA = "transcribe-audio.plan0057-shadow-execution.v1"
EXECUTION_RECEIPT_SCHEMA = "transcribe-audio.plan0057-shadow-execution-receipt.v1"
EXECUTION_REPLAY_SCHEMA = "transcribe-audio.plan0057-shadow-execution-replay.v1"
MODULE_PATH = Path(__file__).name
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0057")
DEFAULT_PRIOR_ROOTS = (
    Path("~/.local/state/transcribe-audio/plan-0037"),
    Path("~/.local/state/transcribe-audio/plan-0056"),
)
PLAN0056_SOURCE_START = "2026-08-05T09:00:00-05:00"
CANDIDATE_IDS = tuple(acoustic_contract.CANDIDATE_IDS)
METHOD_IDS = tuple(acoustic_contract.METHOD_IDS)
EXPECTED_THRESHOLD_UNITS = len(CANDIDATE_IDS) * len(METHOD_IDS)
SHA256_LENGTH = 64
P0_NEGATIVE_ACTION_VECTOR = {
    "decode_audio": False,
    "run_local_diarization": False,
    "run_local_models": False,
    "prepare_shadow_proposals": False,
    "read_human_gold": False,
    "publish_read_only_evidence": False,
    "record_human_review": False,
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}
EXECUTION_ACTION_VECTOR = {
    "decode_audio": True,
    "run_local_diarization": True,
    "run_local_models": True,
    "prepare_shadow_proposals": True,
    "read_human_gold": False,
    "publish_read_only_evidence": True,
    "record_human_review": False,
    "apply_speaker_assignments": False,
    "create_or_mutate_identities": False,
    "mutate_profiles_or_references": False,
    "write_external_provider": False,
    "enable_default_integration": False,
    "run_historical_reprocessing": False,
}
SUBJECT_REVIEW_LABELS = {
    "subject-7c24e8f41409c6f517291fe7": "Chris Williams",
    "subject-df34bc192c07bd86566fff12": "Eric Cochran",
}


class Plan0057Error(ValueError):
    """Raised when Plan 0057 cannot preserve its frozen shadow contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == SHA256_LENGTH and all(char in "0123456789abcdef" for char in text)


def _parse_timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise Plan0057Error("A cohort recording timestamp is invalid.") from exc
    if parsed.tzinfo is None:
        raise Plan0057Error("A cohort recording timestamp must include a timezone.")
    return parsed


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Plan0057Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Plan0057Error("Repository must be clean before authority freeze.")
    if str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split() != ["0", "0"]:
        raise Plan0057Error("Repository must be upstream-even before authority freeze.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_PATH}"], binary=True)
    if not isinstance(body, bytes):
        raise Plan0057Error("Committed Plan 0057 authority is unavailable.")
    module_sha256 = hashlib.sha256(body).hexdigest()
    if module_sha256 != _sha256_file(Path(__file__).resolve()):
        raise Plan0057Error("Committed Plan 0057 authority drifted.")
    return {
        "commit": commit,
        "module_sha256": module_sha256,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _normalize_cohort(raw_cohort: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(raw_cohort) != 3:
        raise Plan0057Error("The shadow batch requires exactly three recordings.")
    baseline = _parse_timestamp(PLAN0056_SOURCE_START)
    cohort: list[dict[str, Any]] = []
    required_hashes = (
        "transcript_sha256",
        "source_media_sha256",
    )
    for expected_ordinal, raw in enumerate(raw_cohort, start=1):
        item = dict(raw)
        if (
            item.get("ordinal") != expected_ordinal
            or not str(item.get("document_id") or "")
            or not str(item.get("conversation_key") or "")
            or not str(item.get("transcript_path") or "")
            or not str(item.get("source_media_path") or "")
            or not str(item.get("context_id") or "")
            or any(not _is_sha256(item.get(key)) for key in required_hashes)
            or _parse_timestamp(item.get("recording_start")) <= baseline
            or float(item.get("duration_seconds") or 0.0) < 60.0
        ):
            raise Plan0057Error("A cohort binding is incomplete, stale, or invalid.")
        probe = item.get("probe")
        if (
            not isinstance(probe, Mapping)
            or float(probe.get("duration_seconds") or 0.0) < 60.0
            or abs(
                float(probe.get("duration_seconds") or 0.0)
                - float(item["duration_seconds"])
            )
            > 1.0
        ):
            raise Plan0057Error("A cohort media probe is incomplete or mismatched.")
        cohort.append(item)
    if len({item["document_id"] for item in cohort}) != 3:
        raise Plan0057Error("Cohort document IDs must be unique.")
    if len({item["conversation_key"] for item in cohort}) != 3:
        raise Plan0057Error("Cohort conversation keys must be unique.")
    if len({item["source_media_sha256"] for item in cohort}) != 3:
        raise Plan0057Error("Cohort source media must be unique.")
    if len({item["context_id"] for item in cohort}) < 2:
        raise Plan0057Error("The cohort requires at least two meeting contexts.")
    return cohort


def _normalize_thresholds(
    raw_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    units = [
        {
            "candidate_id": str(item.get("candidate_id") or ""),
            "method_id": str(item.get("method_id") or ""),
            "threshold": float(item["threshold"]),
            "temperature": float(item["temperature"]),
        }
        for item in raw_units
    ]
    expected = {
        (candidate_id, method_id)
        for candidate_id in CANDIDATE_IDS
        for method_id in METHOD_IDS
    }
    if len(units) != EXPECTED_THRESHOLD_UNITS or {
        (item["candidate_id"], item["method_id"]) for item in units
    } != expected:
        raise Plan0057Error("Exactly nine calibrated acoustic units are required.")
    units.sort(key=lambda item: (item["candidate_id"], item["method_id"]))
    return units


def preview_authority(
    *,
    cohort: Sequence[Mapping[str, Any]],
    prior_hashes: set[str],
    prior_json_hashes: Sequence[str],
    profile_inventory: tuple[list[dict[str, Any]], dict[str, Any]],
    identity_state_snapshot: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
    local_runtime: Mapping[str, Any],
    threshold_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact pre-model Plan 0057 authority without side effects."""

    normalized_cohort = _normalize_cohort(cohort)
    if any(item["source_media_sha256"] in prior_hashes for item in normalized_cohort):
        raise Plan0057Error("A cohort source overlaps retained prior evidence.")
    if not prior_json_hashes or any(not _is_sha256(item) for item in prior_json_hashes):
        raise Plan0057Error("Prior evidence inventory is empty or invalid.")
    profiles, profile_summary = profile_inventory
    profile_subjects = {str(item.get("person_ref_id") or "") for item in profiles}
    profile_candidates = {str(item.get("candidate_id") or "") for item in profiles}
    if (
        len(profiles) != 6
        or profile_subjects != ALLOWLISTED_SUBJECT_IDS
        or profile_candidates != set(CANDIDATE_IDS)
        or any(
            sum(item.get("candidate_id") == candidate for item in profiles) != 2
            for candidate in CANDIDATE_IDS
        )
        or profile_summary.get("profile_count") != 6
        or profile_summary.get("subject_count") != 2
        or profile_summary.get("candidate_count") != 3
    ):
        raise Plan0057Error("Exactly six current profiles for two subjects are required.")
    if not _is_sha256(identity_state_snapshot.get("snapshot_sha256")):
        raise Plan0057Error("The identity-state snapshot is invalid.")
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
        or not str(repository_authority.get("commit") or "")
        or not _is_sha256(repository_authority.get("module_sha256"))
    ):
        raise Plan0057Error("Repository authority must be clean and upstream-even.")
    if (
        local_runtime.get("network_required") is not False
        or local_runtime.get("diarization_model_local") is not True
        or local_runtime.get("transcription_model_local") is not True
        or local_runtime.get("compute_device") not in {"cpu", "cuda"}
        or not _is_sha256(local_runtime.get("runtime_sha256"))
    ):
        raise Plan0057Error("The local acoustic runtime is incomplete.")
    units = _normalize_thresholds(threshold_units)
    core = {
        "schema_version": AUTHORITY_SCHEMA,
        "status": "ready_to_freeze_before_models",
        "repository_authority": dict(repository_authority),
        "allowlisted_subject_ids": sorted(ALLOWLISTED_SUBJECT_IDS),
        "profile_summary": dict(profile_summary),
        "identity_state_before": dict(identity_state_snapshot),
        "source_count": len(normalized_cohort),
        "context_count": len({item["context_id"] for item in normalized_cohort}),
        "source_set_sha256": canonical_hash(
            [item["source_media_sha256"] for item in normalized_cohort]
        ),
        "document_set_sha256": canonical_hash(
            [item["document_id"] for item in normalized_cohort]
        ),
        "freshness": {
            "strictly_after": PLAN0056_SOURCE_START,
            "all_sources_strictly_after": True,
            "prior_overlap_count": 0,
        },
        "prior_exclusion": {
            "root_count": len(DEFAULT_PRIOR_ROOTS),
            "json_file_count": len(prior_json_hashes),
            "json_file_set_sha256": canonical_hash(sorted(prior_json_hashes)),
            "excluded_hash_count": len(prior_hashes),
            "excluded_hash_set_sha256": canonical_hash(sorted(prior_hashes)),
        },
        "local_runtime": dict(local_runtime),
        "threshold_units": units,
        "threshold_unit_count": len(units),
        "diarization_policy": {
            "minimum_speakers": 1,
            "maximum_speakers": 6,
            "model_output_labels_are_not_identities": True,
        },
        "review_clip_policy": {
            "minimum_turn_seconds": 2.0,
            "maximum_turn_seconds": 8.0,
            "maximum_turns_per_speaker": 6,
            "target_seconds_per_speaker": 24.0,
            "minimum_usable_seconds_per_speaker": 6.0,
            "sample_rate": 16_000,
            "channels": 1,
        },
        "private_evidence": {
            "cohort": normalized_cohort,
            "profiles": profiles,
        },
        "contains_human_gold": False,
        "contains_display_names": False,
        "action_vector": dict(P0_NEGATIVE_ACTION_VECTOR),
    }
    return {**core, "content_sha256": canonical_hash(core)}


def _authority_paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / "p0" / f"shadow-authority-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _verify_live_files(preview: Mapping[str, Any]) -> None:
    for source in preview.get("private_evidence", {}).get("cohort", []):
        transcript = Path(str(source.get("transcript_path") or ""))
        media = Path(str(source.get("source_media_path") or ""))
        if (
            not transcript.is_file()
            or transcript.is_symlink()
            or _sha256_file(transcript) != source.get("transcript_sha256")
            or not media.is_file()
            or media.is_symlink()
            or _sha256_file(media) != source.get("source_media_sha256")
        ):
            raise Plan0057Error("A frozen cohort file drifted or disappeared.")


def freeze_authority(
    preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    verify_live_files: bool = True,
) -> dict[str, Any]:
    candidate = dict(preview)
    content_sha256 = candidate.pop("content_sha256", None)
    if (
        content_sha256 != expected_content_sha256
        or content_sha256 != canonical_hash(candidate)
        or candidate.get("schema_version") != AUTHORITY_SCHEMA
        or candidate.get("action_vector") != P0_NEGATIVE_ACTION_VECTOR
    ):
        raise Plan0057Error("The Plan 0057 authority preview is invalid or drifted.")
    if verify_live_files:
        _verify_live_files(preview)
    paths = _authority_paths(runtime_root, expected_content_sha256)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_pre_model_authority",
        "preview": dict(preview),
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": AUTHORITY_RECEIPT_SCHEMA,
        "status": "frozen_pre_model_authority",
        "content_sha256": expected_content_sha256,
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "source_count": preview["source_count"],
        "context_count": preview["context_count"],
        "source_set_sha256": preview["source_set_sha256"],
        "document_set_sha256": preview["document_set_sha256"],
        "identity_state_sha256": preview["identity_state_before"]["snapshot_sha256"],
        "threshold_unit_count": preview["threshold_unit_count"],
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_authority(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    verify_live_files: bool = True,
) -> dict[str, Any]:
    paths = _authority_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != expected_content_sha256
        or receipt.get("content_sha256") != expected_content_sha256
        or receipt.get("manifest_sha256") != _sha256_file(paths["manifest"])
    ):
        raise Plan0057Error("Frozen Plan 0057 authority evidence drifted.")
    candidate = dict(preview)
    candidate.pop("content_sha256", None)
    if expected_content_sha256 != canonical_hash(candidate):
        raise Plan0057Error("Frozen Plan 0057 authority content drifted.")
    if verify_live_files:
        _verify_live_files(preview)
    return {
        **receipt,
        "replay_schema_version": AUTHORITY_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def preview_execution_authority(
    *,
    p0_authority: Mapping[str, Any],
    repository_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize the one local batch attempt after a frozen P0 contract."""

    p0_candidate = dict(p0_authority)
    p0_content_sha256 = p0_candidate.pop("content_sha256", None)
    if (
        p0_content_sha256 != canonical_hash(p0_candidate)
        or p0_authority.get("schema_version") != AUTHORITY_SCHEMA
        or p0_authority.get("status") != "ready_to_freeze_before_models"
        or p0_authority.get("action_vector") != P0_NEGATIVE_ACTION_VECTOR
        or p0_authority.get("source_count") != 3
        or p0_authority.get("context_count", 0) < 2
        or p0_authority.get("threshold_unit_count") != EXPECTED_THRESHOLD_UNITS
    ):
        raise Plan0057Error("Frozen P0 authority is incomplete or unsafe.")
    if (
        repository_authority.get("clean") is not True
        or repository_authority.get("upstream_ahead") != 0
        or repository_authority.get("upstream_behind") != 0
        or repository_authority != p0_authority.get("repository_authority")
    ):
        raise Plan0057Error("Execution repository must match frozen P0 authority.")
    core = {
        "schema_version": EXECUTION_AUTHORITY_SCHEMA,
        "status": "ready_to_freeze_before_local_execution",
        "p0_content_sha256": p0_content_sha256,
        "repository_authority": dict(repository_authority),
        "source_count": p0_authority["source_count"],
        "context_count": p0_authority["context_count"],
        "source_set_sha256": p0_authority["source_set_sha256"],
        "document_set_sha256": p0_authority["document_set_sha256"],
        "identity_state_sha256": p0_authority["identity_state_before"][
            "snapshot_sha256"
        ],
        "runtime_sha256": p0_authority["local_runtime"]["runtime_sha256"],
        "threshold_unit_count": p0_authority["threshold_unit_count"],
        "maximum_execution_attempts": 1,
        "action_vector": dict(EXECUTION_ACTION_VECTOR),
        "contains_human_gold": False,
        "contains_display_names": False,
    }
    return {**core, "content_sha256": canonical_hash(core)}


def _execution_authority_paths(
    runtime_root: Path,
    content_sha256: str,
) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / "p1" / f"execution-authority-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def freeze_execution_authority(
    preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    candidate = dict(preview)
    content_sha256 = candidate.pop("content_sha256", None)
    if (
        content_sha256 != expected_content_sha256
        or content_sha256 != canonical_hash(candidate)
        or candidate.get("schema_version") != EXECUTION_AUTHORITY_SCHEMA
        or candidate.get("action_vector") != EXECUTION_ACTION_VECTOR
    ):
        raise Plan0057Error("Execution authority preview is invalid or drifted.")
    paths = _execution_authority_paths(runtime_root, expected_content_sha256)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {
        "schema_version": EXECUTION_AUTHORITY_MANIFEST_SCHEMA,
        "status": "frozen_before_local_execution",
        "preview": dict(preview),
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": EXECUTION_AUTHORITY_RECEIPT_SCHEMA,
        "status": "frozen_before_local_execution",
        "content_sha256": expected_content_sha256,
        "p0_content_sha256": preview["p0_content_sha256"],
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "source_count": preview["source_count"],
        "context_count": preview["context_count"],
        "identity_state_sha256": preview["identity_state_sha256"],
        "maximum_execution_attempts": 1,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_execution_authority(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _execution_authority_paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != expected_content_sha256
        or receipt.get("content_sha256") != expected_content_sha256
        or receipt.get("manifest_sha256") != _sha256_file(paths["manifest"])
        or preview.get("action_vector") != EXECUTION_ACTION_VECTOR
    ):
        raise Plan0057Error("Frozen execution authority evidence drifted.")
    candidate = dict(preview)
    candidate.pop("content_sha256", None)
    if expected_content_sha256 != canonical_hash(candidate):
        raise Plan0057Error("Frozen execution authority content drifted.")
    return {
        **receipt,
        "replay_schema_version": EXECUTION_AUTHORITY_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def build_live_execution_authority(
    p0_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    replay_authority(p0_content_sha256, runtime_root=runtime_root)
    p0_authority = _load_frozen_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
    )
    return preview_execution_authority(
        p0_authority=p0_authority,
        repository_authority=_repository_authority(),
    )


def build_execution_manifest(
    *,
    authority: Mapping[str, Any],
    execution_authority: Mapping[str, Any] | None = None,
    source_results: Sequence[Mapping[str, Any]],
    identity_state_before: Mapping[str, Any],
    identity_state_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate complete batch yield without trusting per-source summaries."""

    cohort = authority.get("private_evidence", {}).get("cohort")
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA
        or authority.get("source_count") != 3
        or not isinstance(cohort, list)
        or len(cohort) != 3
        or identity_state_before != authority.get("identity_state_before")
        or identity_state_after != identity_state_before
    ):
        raise Plan0057Error("Execution authority or identity state is invalid.")
    expected_ids = {str(item["document_id"]) for item in cohort}
    by_document = {
        str(item.get("document_id") or ""): dict(item) for item in source_results
    }
    if len(source_results) != 3 or set(by_document) != expected_ids:
        raise Plan0057Error("The recording execution denominator is incomplete.")
    stop_reasons: list[dict[str, str]] = []
    eligible_speakers = 0
    covered_speakers = 0
    for source in cohort:
        result = by_document[source["document_id"]]
        if (
            result.get("conversation_key") != source["conversation_key"]
            or result.get("source_media_sha256") != source["source_media_sha256"]
        ):
            raise Plan0057Error("A source execution result is unbound.")
        eligible = result.get("eligible_speaker_count")
        covered = result.get("covered_speaker_count")
        proposals = result.get("proposals")
        if (
            isinstance(eligible, bool)
            or not isinstance(eligible, int)
            or eligible < 0
            or isinstance(covered, bool)
            or not isinstance(covered, int)
            or covered < 0
            or not isinstance(proposals, list)
        ):
            raise Plan0057Error("A source execution denominator is invalid.")
        stop_reason = result.get("stop_reason")
        if result.get("entered") is not True:
            if not stop_reason:
                raise Plan0057Error("A non-entered source requires an explicit stop reason.")
        if stop_reason:
            if covered or proposals:
                raise Plan0057Error("A stopped source cannot claim covered speakers.")
            stop_reasons.append(
                {"document_id": source["document_id"], "reason": str(stop_reason)}
            )
        elif eligible != covered or len(proposals) != covered:
            raise Plan0057Error("Every eligible speaker requires proposal or abstention evidence.")
        eligible_speakers += eligible
        covered_speakers += covered
    core = {
        "schema_version": EXECUTION_SCHEMA,
        "status": (
            "stopped_before_complete_review"
            if stop_reasons
            else "complete_pending_human_review"
        ),
        "p0_content_sha256": authority["content_sha256"],
        "execution_authority_content_sha256": (
            execution_authority or authority
        )["content_sha256"],
        "source_results": [by_document[item["document_id"]] for item in cohort],
        "eligible_recording_count": 3,
        "entered_recording_count": sum(
            by_document[item["document_id"]].get("entered") is True for item in cohort
        ),
        "eligible_speaker_count": eligible_speakers,
        "covered_speaker_count": covered_speakers,
        "stop_reasons": stop_reasons,
        "identity_state_before": dict(identity_state_before),
        "identity_state_after": dict(identity_state_after),
        "identity_state_unchanged": True,
        "read_human_gold": False,
        "applied_assignments": False,
        "created_or_mutated_identities": False,
        "mutated_profiles_or_references": False,
        "wrote_external_provider": False,
        "enabled_default_integration": False,
        "ran_historical_reprocessing": False,
        "requires_human_review": not stop_reasons,
    }
    return {**core, "content_sha256": canonical_hash(core)}


def _execution_paths(runtime_root: Path, authority_sha256: str) -> dict[str, Path]:
    authority_paths = _execution_authority_paths(runtime_root, authority_sha256)
    run = authority_paths["run"] / "batch-execution"
    return {
        "root": authority_paths["root"],
        "authority_run": authority_paths["run"],
        "run": run,
        "sources": run / "sources",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
        "review": run / "review",
        "review_index": run / "review" / "index.html",
        "answer_template": run / "review" / "answer-template.json",
    }


def _source_paths(execution_paths: Mapping[str, Path], source: Mapping[str, Any]) -> dict[str, Path]:
    source_key = hashlib.sha256(str(source["document_id"]).encode("utf-8")).hexdigest()[:16]
    run = execution_paths["sources"] / f"{int(source['ordinal']):02d}-{source_key}"
    return {
        "root": execution_paths["root"],
        "run": run,
        "pcm": run / "source-pcm.wav",
        "diarization": run / "diarization.json",
        "clips": run / "clips",
        "transcripts": run / "transcripts",
        "p1": run / "preparation-p1",
        "p2": run / "preparation-p2",
        "matrices": run / "matrices",
        "proposals": run / "proposals.json",
    }


def _load_frozen_preview(
    expected_authority_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    paths = _authority_paths(runtime_root, expected_authority_sha256)
    require_private_file(paths["manifest"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise Plan0057Error("Frozen Plan 0057 authority is invalid.")
    return preview


def _load_frozen_execution_preview(
    expected_authority_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    paths = _execution_authority_paths(runtime_root, expected_authority_sha256)
    require_private_file(paths["manifest"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise Plan0057Error("Frozen Plan 0057 execution authority is invalid.")
    return preview


def _current_identity_state() -> dict[str, Any]:
    return plan0056_pilot.snapshot_identity_state(
        primary_store=plan0056_pilot.DEFAULT_PRIMARY_STORE,
        knowledge_store=plan0056_pilot.DEFAULT_KNOWLEDGE_STORE,
        profile_store=plan0056_pilot.DEFAULT_PROFILE_STORE,
        reference_store=plan0056_pilot.DEFAULT_REFERENCE_STORE,
    )


def _execute_source(
    source: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    execution_paths: Mapping[str, Path],
) -> dict[str, Any]:
    paths = _source_paths(execution_paths, source)
    ensure_private_tree(paths["root"], paths["run"])
    media_path = Path(str(source["source_media_path"]))
    transcript_path = Path(str(source["transcript_path"]))
    if (
        _sha256_file(media_path) != source["source_media_sha256"]
        or _sha256_file(transcript_path) != source["transcript_sha256"]
    ):
        raise Plan0057Error("A source or transcript drifted before execution.")
    plan0056_runner._decode_private_pcm(media_path, paths["pcm"])
    timeline = plan0056_runner._run_local_diarization(
        paths["pcm"],
        model_root=Path(authority["local_runtime"]["diarization_model"]["root"]),
        minimum_speakers=authority["diarization_policy"]["minimum_speakers"],
        maximum_speakers=authority["diarization_policy"]["maximum_speakers"],
        compute_device=authority["local_runtime"]["compute_device"],
    )
    clip_policy = authority["review_clip_policy"]
    selected = plan0056_runner.select_review_segments(
        timeline,
        minimum_turn_seconds=clip_policy["minimum_turn_seconds"],
        maximum_turn_seconds=clip_policy["maximum_turn_seconds"],
        maximum_turns_per_speaker=clip_policy["maximum_turns_per_speaker"],
        target_seconds_per_speaker=clip_policy["target_seconds_per_speaker"],
        minimum_usable_seconds_per_speaker=clip_policy[
            "minimum_usable_seconds_per_speaker"
        ],
    )
    write_immutable_private_json(
        paths["diarization"],
        {"timeline": timeline, "selected": selected},
    )
    bindings = []
    for speaker_ref, segments in selected.items():
        clip = plan0056_runner._write_speaker_clip(
            paths["pcm"],
            paths["clips"] / f"{speaker_ref}.wav",
            segments,
        )
        bindings.append({"speaker_ref": speaker_ref, **clip})
    snapshots = (
        plan0056_execution.DEFAULT_WHISPER_CACHE_ROOT.expanduser().absolute()
        / "snapshots"
    )
    model_snapshots = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(model_snapshots) != 1:
        raise Plan0057Error("The local transcription model snapshot is ambiguous.")
    transcripts = plan0056_runner._transcribe_clips(
        bindings,
        model_snapshot=model_snapshots[0],
    )
    ensure_private_tree(paths["root"], paths["transcripts"])
    transcript_artifact = paths["transcripts"] / "speaker-transcripts.json"
    write_immutable_private_json(transcript_artifact, {"rows": transcripts})
    scoring_preview = {
        "content_sha256": authority["content_sha256"],
        "p0_authority": {
            "allowlisted_subject_ids": authority["allowlisted_subject_ids"]
        },
        "threshold_units": authority["threshold_units"],
    }
    matrices = plan0056_runner._score_matrices(
        bindings,
        preview=scoring_preview,
        paths=paths,
    )
    proposal_evidence = plan0056_execution.proposals_from_matrices(
        matrices,
        expected_speaker_refs=list(selected),
        allowlisted_subject_ids=authority["allowlisted_subject_ids"],
    )
    write_immutable_private_json(paths["proposals"], proposal_evidence)
    transcripts_by_ref = {
        str(item["speaker_ref"]): str(item.get("transcript") or "")
        for item in transcripts
    }
    clips_by_ref = {str(item["speaker_ref"]): item for item in bindings}
    review_rows = []
    for proposal in proposal_evidence["proposals"]:
        speaker_ref = str(proposal["speaker_ref"])
        review_rows.append(
            {
                "speaker_ref": speaker_ref,
                "clip_path": clips_by_ref[speaker_ref]["clip_path"],
                "clip_sha256": clips_by_ref[speaker_ref]["clip_sha256"],
                "transcript": transcripts_by_ref[speaker_ref],
                "proposal": proposal,
            }
        )
    return {
        "document_id": source["document_id"],
        "conversation_key": source["conversation_key"],
        "source_media_sha256": source["source_media_sha256"],
        "entered": True,
        "eligible_speaker_count": len(bindings),
        "covered_speaker_count": len(proposal_evidence["proposals"]),
        "stop_reason": None,
        "proposals": proposal_evidence["proposals"],
        "review_rows": review_rows,
        "artifact_hashes": {
            "source_pcm": _sha256_file(paths["pcm"]),
            "diarization": _sha256_file(paths["diarization"]),
            "speaker_transcripts": _sha256_file(transcript_artifact),
            "proposal_content": proposal_evidence["content_sha256"],
            "matrix_set": canonical_hash(
                [item["content_sha256"] for item in matrices]
            ),
        },
    }


def _render_review_artifact(
    *,
    execution_manifest: Mapping[str, Any],
    execution_paths: Mapping[str, Path],
) -> dict[str, Any]:
    ensure_private_tree(execution_paths["root"], execution_paths["review"])
    cards: list[str] = []
    answer_rows: list[dict[str, Any]] = []
    for result in execution_manifest["source_results"]:
        for row in result.get("review_rows") or []:
            proposal = row["proposal"]
            subject_id = proposal.get("subject_id")
            display = SUBJECT_REVIEW_LABELS.get(subject_id, "No enrolled subject")
            clip_path = Path(str(row["clip_path"]))
            clip_url = "../" + clip_path.relative_to(execution_paths["run"]).as_posix()
            card_id = f"{result['document_id']}::{row['speaker_ref']}"
            cards.append(
                "<article class='card'>"
                f"<h2>{html.escape(str(row['speaker_ref']))}</h2>"
                f"<p><strong>Proposal:</strong> {html.escape(display)} "
                f"({html.escape(str(subject_id or 'abstain'))})</p>"
                f"<p>{html.escape(str(proposal['confidence_band']))} confidence; "
                f"{int(proposal['supporting_unit_count'])} supporting units; "
                f"{int(proposal['opposing_unit_count'])} opposing units.</p>"
                f"<audio controls preload='metadata' src='{html.escape(clip_url)}'></audio>"
                f"<p class='transcript'>{html.escape(str(row['transcript']))}</p>"
                f"<code>{html.escape(card_id)}</code>"
                "</article>"
            )
            answer_rows.append(
                {
                    "document_id": result["document_id"],
                    "speaker_ref": row["speaker_ref"],
                    "actual_identity": None,
                    "review_display_label": None,
                    "allowed_actual_identities": [
                        *sorted(ALLOWLISTED_SUBJECT_IDS),
                        "neither_enrolled",
                        "unknown",
                    ],
                }
            )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plan 0057 acoustic shadow review</title><style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;background:#f4f6f8;color:#17212b}
.notice,.card{background:white;border:1px solid #ccd5df;border-radius:12px;padding:1rem;margin:1rem 0}
.notice{border-left:6px solid #8c5a00}.card audio{width:100%}.transcript{white-space:pre-wrap;color:#354554}
code{word-break:break-all}h1,h2{margin:.2rem 0 .7rem}</style></head><body>
<h1>Plan 0057 enrolled-only acoustic shadow review</h1>
<div class="notice"><strong>Review evidence only.</strong> These proposals do not apply assignments, create identities, or update profiles. Decide every card as one enrolled subject, neither enrolled person, or unknown.</div>
""" + "\n".join(cards) + "</body></html>"
    execution_paths["review_index"].write_text(page, encoding="utf-8")
    execution_paths["review_index"].chmod(0o600)
    template = {
        "schema_version": "transcribe-audio.plan0057-human-review-input.v1",
        "execution_content_sha256": execution_manifest["content_sha256"],
        "decision_count": len(answer_rows),
        "decisions": answer_rows,
        "apply_speaker_assignments": False,
        "create_or_mutate_identities": False,
        "mutate_profiles_or_references": False,
    }
    write_immutable_private_json(execution_paths["answer_template"], template)
    return {
        "index_path": str(execution_paths["review_index"]),
        "index_sha256": _sha256_file(execution_paths["review_index"]),
        "answer_template_path": str(execution_paths["answer_template"]),
        "answer_template_sha256": _sha256_file(execution_paths["answer_template"]),
        "decision_count": len(answer_rows),
    }


def execute_batch(
    expected_execution_authority_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    state_root: Path = Path("~/.local/state/transcribe-audio"),
) -> dict[str, Any]:
    replay_execution_authority(
        expected_execution_authority_sha256,
        runtime_root=runtime_root,
    )
    execution_authority = _load_frozen_execution_preview(
        expected_execution_authority_sha256,
        runtime_root=runtime_root,
    )
    p0_content_sha256 = execution_authority["p0_content_sha256"]
    replay_authority(p0_content_sha256, runtime_root=runtime_root)
    authority = _load_frozen_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
    )
    if _repository_authority() != execution_authority["repository_authority"]:
        raise Plan0057Error("Execution repository differs from frozen authority.")
    paths = _execution_paths(runtime_root, expected_execution_authority_sha256)
    if paths["receipt"].exists():
        return replay_execution(
            expected_execution_authority_sha256,
            runtime_root=runtime_root,
            state_root=state_root,
        )
    if paths["run"].is_dir() and any(paths["run"].iterdir()):
        raise Plan0057Error(
            "Partial execution artifacts require an explicit disposition; automatic retry is forbidden."
        )
    for source in authority["private_evidence"]["cohort"]:
        existing = load_for_review(
            document_id=source["document_id"],
            conversation_key=source["conversation_key"],
            source_path=source["transcript_path"],
            state_root=state_root,
        )
        if existing.get("status") != "absent":
            raise Plan0057Error(
                "A cohort document already has active or rejected acoustic evidence."
            )
    ensure_private_tree(paths["root"], paths["run"])
    before = _current_identity_state()
    if before != authority["identity_state_before"]:
        raise Plan0057Error("Identity or profile state drifted before batch execution.")
    source_results: list[dict[str, Any]] = []
    stopped = False
    for source in authority["private_evidence"]["cohort"]:
        if stopped:
            source_results.append(
                {
                    "document_id": source["document_id"],
                    "conversation_key": source["conversation_key"],
                    "source_media_sha256": source["source_media_sha256"],
                    "entered": False,
                    "eligible_speaker_count": 0,
                    "covered_speaker_count": 0,
                    "stop_reason": "not_entered_after_prior_stop",
                    "proposals": [],
                    "review_rows": [],
                    "artifact_hashes": {},
                }
            )
            continue
        try:
            source_results.append(
                _execute_source(
                    source,
                    authority=authority,
                    execution_paths=paths,
                )
            )
        except (Plan0057Error, OSError, RuntimeError, ValueError) as exc:
            source_results.append(
                {
                    "document_id": source["document_id"],
                    "conversation_key": source["conversation_key"],
                    "source_media_sha256": source["source_media_sha256"],
                    "entered": True,
                    "eligible_speaker_count": 0,
                    "covered_speaker_count": 0,
                    "stop_reason": f"execution_error:{type(exc).__name__}",
                    "proposals": [],
                    "review_rows": [],
                    "artifact_hashes": {},
                }
            )
            stopped = True
    after = _current_identity_state()
    manifest = build_execution_manifest(
        authority=authority,
        execution_authority=execution_authority,
        source_results=source_results,
        identity_state_before=before,
        identity_state_after=after,
    )
    write_immutable_private_json(paths["manifest"], manifest)
    projections: list[dict[str, Any]] = []
    activation: dict[str, Any] | None = None
    review_artifact: dict[str, Any] | None = None
    if manifest["status"] == "complete_pending_human_review":
        by_document = {
            item["document_id"]: item
            for item in authority["private_evidence"]["cohort"]
        }
        for result in manifest["source_results"]:
            source = by_document[result["document_id"]]
            bundle = build_shadow_bundle(
                document_id=result["document_id"],
                conversation_key=result["conversation_key"],
                source_path=source["transcript_path"],
                source_media_sha256=result["source_media_sha256"],
                execution_content_sha256=manifest["content_sha256"],
                identity_state_sha256=after["snapshot_sha256"],
                rows=result["proposals"],
            )
            projections.append(
                publish_shadow_bundle(
                    bundle,
                    source_path=source["transcript_path"],
                    state_root=state_root,
                    activate=False,
                )
            )
        activation = activate_shadow_batch(
            projections,
            execution_content_sha256=manifest["content_sha256"],
            state_root=state_root,
        )
        review_artifact = _render_review_artifact(
            execution_manifest=manifest,
            execution_paths=paths,
        )
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "status": manifest["status"],
        "execution_authority_content_sha256": expected_execution_authority_sha256,
        "p0_content_sha256": p0_content_sha256,
        "execution_content_sha256": manifest["content_sha256"],
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "eligible_recording_count": manifest["eligible_recording_count"],
        "entered_recording_count": manifest["entered_recording_count"],
        "eligible_speaker_count": manifest["eligible_speaker_count"],
        "covered_speaker_count": manifest["covered_speaker_count"],
        "stop_reason_count": len(manifest["stop_reasons"]),
        "projection_count": len(projections),
        "projection_content_sha256s": [item["content_sha256"] for item in projections],
        "activation": activation,
        "review_artifact": review_artifact,
        "identity_state_unchanged": True,
        "requires_human_review": manifest["requires_human_review"],
        "applied_assignments": False,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_execution(
    expected_execution_authority_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    state_root: Path = Path("~/.local/state/transcribe-audio"),
) -> dict[str, Any]:
    replay_execution_authority(
        expected_execution_authority_sha256,
        runtime_root=runtime_root,
    )
    execution_authority = _load_frozen_execution_preview(
        expected_execution_authority_sha256,
        runtime_root=runtime_root,
    )
    p0_content_sha256 = execution_authority["p0_content_sha256"]
    replay_authority(p0_content_sha256, runtime_root=runtime_root)
    authority = _load_frozen_preview(
        p0_content_sha256,
        runtime_root=runtime_root,
    )
    paths = _execution_paths(runtime_root, expected_execution_authority_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if (
        manifest.get("execution_authority_content_sha256")
        != expected_execution_authority_sha256
        or manifest.get("p0_content_sha256") != p0_content_sha256
        or receipt.get("manifest_sha256") != _sha256_file(paths["manifest"])
        or receipt.get("execution_content_sha256") != manifest.get("content_sha256")
        or manifest.get("identity_state_unchanged") is not True
        or manifest.get("applied_assignments") is not False
        or _current_identity_state() != manifest.get("identity_state_after")
    ):
        raise Plan0057Error("Plan 0057 execution evidence drifted.")
    if manifest.get("status") == "complete_pending_human_review":
        cohort = {
            item["document_id"]: item
            for item in authority["private_evidence"]["cohort"]
        }
        for result in manifest["source_results"]:
            source = cohort[result["document_id"]]
            loaded = load_for_review(
                document_id=result["document_id"],
                conversation_key=result["conversation_key"],
                source_path=source["transcript_path"],
                state_root=state_root,
            )
            if (
                loaded.get("status") != "available"
                or loaded.get("execution_content_sha256")
                != manifest["content_sha256"]
            ):
                raise Plan0057Error("Published shadow evidence drifted.")
    return {
        **receipt,
        "replay_schema_version": EXECUTION_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }


def _prior_inventory(prior_roots: Sequence[Path]) -> tuple[set[str], list[str]]:
    prior_hashes: set[str] = set()
    json_hashes: list[str] = []
    for supplied_root in prior_roots:
        root = supplied_root.expanduser().absolute()
        if not root.is_dir() or root.is_symlink():
            raise Plan0057Error("A prior evidence root is unavailable.")
        for path in sorted(root.rglob("*.json")):
            if not path.is_file() or path.is_symlink():
                continue
            found, _parse_mode = plan0056_pilot._evidence_hashes(path)
            prior_hashes.update(found)
            json_hashes.append(_sha256_file(path))
    return prior_hashes, json_hashes


def _live_cohort(
    document_ids: Sequence[str],
    *,
    store_root: Path = transcript_store.DEFAULT_STORE_DIR,
) -> list[dict[str, Any]]:
    if len(document_ids) != 3 or len(set(document_ids)) != 3:
        raise Plan0057Error("Exactly three unique transcript document IDs are required.")
    database = transcript_store.db_path(store_root)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = []
        for ordinal, document_id in enumerate(document_ids, start=1):
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ? AND kind = 'transcript'",
                (document_id,),
            ).fetchone()
            if row is None:
                raise Plan0057Error("A cohort transcript document is unavailable.")
            try:
                payload = json.loads(row["json_payload"] or "{}")
            except json.JSONDecodeError as exc:
                raise Plan0057Error("A cohort transcript payload is invalid.") from exc
            transcript_path = Path(str(row["source_path"] or "")).absolute()
            media_value = payload.get("working_media_path") or payload.get("source_media_path")
            media_path = Path(str(media_value or "")).absolute()
            event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
            if event.get("start") or event.get("end"):
                context_basis = {
                    "event_start": event.get("start"),
                    "event_end": event.get("end"),
                }
            else:
                context_basis = {
                    "conversation_id": payload.get("conversation_id"),
                }
            probe = plan0056_pilot._probe_media(media_path)
            rows.append(
                {
                    "ordinal": ordinal,
                    "document_id": str(row["id"]),
                    "conversation_key": str(row["source_path"] or row["id"]),
                    "transcript_path": str(transcript_path),
                    "transcript_sha256": _sha256_file(transcript_path),
                    "source_media_path": str(media_path),
                    "source_media_sha256": _sha256_file(media_path),
                    "recording_start": str(payload.get("recording_start") or ""),
                    "context_id": canonical_hash(context_basis),
                    "duration_seconds": float(payload.get("duration_seconds") or 0.0),
                    "probe": probe,
                }
            )
        return rows
    finally:
        connection.close()


def build_live_authority(
    document_ids: Sequence[str],
    *,
    store_root: Path = transcript_store.DEFAULT_STORE_DIR,
    prior_roots: Sequence[Path] = DEFAULT_PRIOR_ROOTS,
) -> dict[str, Any]:
    prior_hashes, prior_json_hashes = _prior_inventory(prior_roots)
    profiles = recalibration._active_profiles(
        calibration_root=recalibration.DEFAULT_CALIBRATION_ROOT,
        p3_runtime_root=recalibration.DEFAULT_P3_RUNTIME_ROOT,
    )
    identity_state = plan0056_pilot.snapshot_identity_state(
        primary_store=plan0056_pilot.DEFAULT_PRIMARY_STORE,
        knowledge_store=plan0056_pilot.DEFAULT_KNOWLEDGE_STORE,
        profile_store=plan0056_pilot.DEFAULT_PROFILE_STORE,
        reference_store=plan0056_pilot.DEFAULT_REFERENCE_STORE,
    )
    thresholds = generation5_e2._threshold_authority()["thresholds"]
    return preview_authority(
        cohort=_live_cohort(document_ids, store_root=store_root),
        prior_hashes=prior_hashes,
        prior_json_hashes=prior_json_hashes,
        profile_inventory=profiles,
        identity_state_snapshot=identity_state,
        repository_authority=_repository_authority(),
        local_runtime=plan0056_execution.local_runtime_inventory(),
        threshold_units=thresholds,
    )


def portable_authority(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_SCHEMA,
        "status": preview["status"],
        "content_sha256": preview["content_sha256"],
        "repository_authority": preview["repository_authority"],
        "source_count": preview["source_count"],
        "context_count": preview["context_count"],
        "source_set_sha256": preview["source_set_sha256"],
        "document_set_sha256": preview["document_set_sha256"],
        "identity_state_sha256": preview["identity_state_before"]["snapshot_sha256"],
        "profile_summary": preview["profile_summary"],
        "threshold_unit_count": preview["threshold_unit_count"],
        "freshness": preview["freshness"],
        "prior_exclusion": preview["prior_exclusion"],
        "action_vector": preview["action_vector"],
        "contains_private_paths": False,
        "contains_display_names": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or replay Plan 0057 P0 authority.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "freeze"):
        child = subparsers.add_parser(command)
        child.add_argument("--document-id", action="append", required=True)
        child.add_argument("--store-root", type=Path, default=transcript_store.DEFAULT_STORE_DIR)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
        if command == "freeze":
            child.add_argument("--expected-content-sha256", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--authority-content-sha256", required=True)
    replay.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    replay_execution_authority_parser = subparsers.add_parser(
        "replay-execution-authority"
    )
    replay_execution_authority_parser.add_argument(
        "--authority-content-sha256",
        required=True,
    )
    replay_execution_authority_parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
    )
    for command in ("preview-execution", "freeze-execution"):
        child = subparsers.add_parser(command)
        child.add_argument("--p0-content-sha256", required=True)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
        if command == "freeze-execution":
            child.add_argument("--expected-content-sha256", required=True)
    for command in ("execute", "replay-execution"):
        child = subparsers.add_parser(command)
        child.add_argument("--authority-content-sha256", required=True)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
        child.add_argument(
            "--state-root",
            type=Path,
            default=Path("~/.local/state/transcribe-audio"),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        result = replay_authority(
            args.authority_content_sha256,
            runtime_root=args.runtime_root,
        )
    elif args.command == "replay-execution-authority":
        result = replay_execution_authority(
            args.authority_content_sha256,
            runtime_root=args.runtime_root,
        )
    elif args.command in {"preview-execution", "freeze-execution"}:
        preview = build_live_execution_authority(
            args.p0_content_sha256,
            runtime_root=args.runtime_root,
        )
        if args.command == "preview-execution":
            result = preview
        else:
            result = freeze_execution_authority(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    elif args.command in {"execute", "replay-execution"}:
        operation = execute_batch if args.command == "execute" else replay_execution
        result = operation(
            args.authority_content_sha256,
            runtime_root=args.runtime_root,
            state_root=args.state_root,
        )
    else:
        preview = build_live_authority(args.document_id, store_root=args.store_root)
        if args.command == "preview":
            result = portable_authority(preview)
        else:
            result = freeze_authority(
                preview,
                expected_content_sha256=args.expected_content_sha256,
                runtime_root=args.runtime_root,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
