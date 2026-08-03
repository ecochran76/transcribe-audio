"""Plan 0052 Generation-4 campaign checkpoint authority."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import acoustic_generation3_recalibration as generation3_recalibration
import acoustic_generation3_recalibration_execution as generation3_execution
import acoustic_generation4_media as generation4_media
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation4-campaign-g0-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation4-campaign-g0-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-campaign-g0-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-campaign-g0-replay.v1"
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
MODULE_NAME = "acoustic_generation4_campaign.py"
PLAN_PATH = Path(
    "docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md"
)
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052")
MEDIA_PREVIEW_SHA256 = (
    "af5bcf2d8e60b811bcddbb875dd1044f69a090346c6118525c5c5dd80bc49974"
)
THRESHOLD_EXECUTION_AUTHORITY_SHA256 = (
    "39298c74aab4a773945268cd73fbaabccf88e8e3026a4041ea6eeed29b715b4f"
)

G0_ACTIONS = (
    "run_g1a_cohort_gold_feasibility",
    "run_g1b_acoustic_contract",
    "run_g1c_context_contract",
    "run_j1_design_reconciliation",
    "freeze_g2_envelope",
    "run_g3_blind_baseline",
    "run_g4_augmented_predictions",
    "run_j2_blindness_audit",
    "reveal_gold",
    "run_g5_scoring",
    "run_j3_result_audit",
    "make_g6_terminal_decision",
    "mutate_profiles_or_references",
    "enable_default_integration",
    "run_historical_reprocessing",
)


class Generation4CampaignError(ValueError):
    """Raised when Plan 0052 G0 cannot fail closed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _sha(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(str(value or "")))


def _validated_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    evidence = dict(value)
    media = evidence.get("media")
    profiles = evidence.get("profiles")
    thresholds = evidence.get("thresholds")
    runtime = evidence.get("runtime")
    if not all(isinstance(item, Mapping) for item in (media, profiles, thresholds, runtime)):
        raise Generation4CampaignError("Generation-4 inherited evidence is incomplete.")
    if (
        not all(_sha(media.get(key)) for key in (
            "preview_content_sha256", "manifest_sha256", "qualified_set_sha256"
        ))
        or media.get("candidate_count") != 12
        or media.get("qualified_count") != 10
        or media.get("rejected_count") != 2
        or media.get("reason_counts")
        != {"qualified": 10, "duration_below_minimum": 2}
        or media.get("replay_mode")
        != "full_body_with_source_redecode_no_retained_audio"
        or media.get("idempotent_replay") is not True
        or not all(_sha(profiles.get(key)) for key in (
            "recalibration_content_sha256", "recalibration_manifest_sha256",
            "profile_set_sha256", "model_asset_set_sha256"
        ))
        or profiles.get("profile_count") != 6
        or profiles.get("subject_count") != 2
        or profiles.get("candidate_count") != 3
        or not all(_sha(thresholds.get(key)) for key in (
            "execution_authority_sha256", "score_matrix_sha256",
            "threshold_application_sha256", "threshold_set_sha256"
        ))
        or thresholds.get("threshold_unit_count") != 9
        or thresholds.get("replay_mode")
        != "recomputed_from_persisted_scores_without_audio"
        or thresholds.get("idempotent_replay") is not True
        or runtime != {"speechbrain": "1.1.0", "onnxruntime": "1.24.4"}
    ):
        raise Generation4CampaignError("Generation-4 inherited evidence did not replay exactly.")
    return evidence


def _validated_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    repository = dict(value)
    if (
        not COMMIT_RE.fullmatch(str(repository.get("commit") or ""))
        or repository.get("module_name") != MODULE_NAME
        or not _sha(repository.get("module_sha256"))
        or not _sha(repository.get("plan_sha256"))
        or repository.get("clean") is not True
        or repository.get("upstream_ahead") != 0
        or repository.get("upstream_behind") != 0
    ):
        raise Generation4CampaignError("Generation-4 repository authority is invalid.")
    return repository


def _git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode:
        raise Generation4CampaignError("Generation-4 repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation4CampaignError("Repository must be clean for G0 authority.")
    behind_ahead = str(
        _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    ).split()
    if behind_ahead != ["0", "0"]:
        raise Generation4CampaignError("Repository must be upstream-even for G0 authority.")
    commit = str(_git(["log", "-1", "--format=%H", "--", MODULE_NAME]))
    if not COMMIT_RE.fullmatch(commit) or _git(
        ["merge-base", "--is-ancestor", commit, "HEAD"]
    ) != "":
        raise Generation4CampaignError("G0 module commit is not an ancestor of HEAD.")
    module_blob = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    plan_blob = _git(["show", f"{commit}:{PLAN_PATH.as_posix()}"], binary=True)
    if not isinstance(module_blob, bytes) or not isinstance(plan_blob, bytes):
        raise Generation4CampaignError("G0 repository blobs are unavailable.")
    module_sha = hashlib.sha256(module_blob).hexdigest()
    if sha256_file(Path(__file__).resolve()) != module_sha:
        raise Generation4CampaignError("G0 module authority drifted.")
    return {
        "commit": commit,
        "module_name": MODULE_NAME,
        "module_sha256": module_sha,
        "plan_sha256": hashlib.sha256(plan_blob).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _collect_inherited_evidence() -> dict[str, Any]:
    media = generation4_media.replay_generation4_media_authority(
        MEDIA_PREVIEW_SHA256
    )
    thresholds = generation3_execution.replay_generation3_recalibration_thresholds(
        THRESHOLD_EXECUTION_AUTHORITY_SHA256
    )
    recalibration_path = generation3_recalibration._existing_manifest(
        generation3_recalibration.DEFAULT_RUNTIME_ROOT
    )
    if recalibration_path is None:
        raise Generation4CampaignError("Frozen successor profile authority is unavailable.")
    require_private_file(
        recalibration_path,
        generation3_recalibration.DEFAULT_RUNTIME_ROOT.expanduser().absolute(),
    )
    try:
        recalibration = json.loads(recalibration_path.read_text(encoding="utf-8"))
        active = recalibration["preview"]["active_profile_authority"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Generation4CampaignError(
            "Frozen successor profile authority is unreadable."
        ) from exc
    return {
        "media": {
            "preview_content_sha256": media["preview_content_sha256"],
            "manifest_sha256": media["manifest_sha256"],
            "qualified_set_sha256": media["qualified_set_sha256"],
            "candidate_count": media["candidate_count"],
            "qualified_count": media["qualified_count"],
            "rejected_count": media["rejected_count"],
            "reason_counts": dict(media["reason_counts"]),
            "replay_mode": media["replay_mode"],
            "idempotent_replay": media["idempotent_replay"],
        },
        "profiles": {
            "recalibration_content_sha256": recalibration["content_sha256"],
            "recalibration_manifest_sha256": sha256_file(recalibration_path),
            "profile_set_sha256": active["profile_set_sha256"],
            "model_asset_set_sha256": active["model_asset_set_sha256"],
            "profile_count": active["profile_count"],
            "subject_count": active["subject_count"],
            "candidate_count": active["candidate_count"],
        },
        "thresholds": {
            "execution_authority_sha256": thresholds["execution_authority_sha256"],
            "score_matrix_sha256": thresholds["score_matrix_sha256"],
            "threshold_application_sha256": thresholds[
                "threshold_application_sha256"
            ],
            "threshold_set_sha256": thresholds["threshold_set_sha256"],
            "threshold_unit_count": thresholds["threshold_unit_count"],
            "replay_mode": thresholds["threshold_replay_mode"],
            "idempotent_replay": thresholds["idempotent_replay"],
        },
        "runtime": {
            "speechbrain": importlib.metadata.version("speechbrain"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
        },
    }


def preview_generation4_campaign(
    *,
    collect_evidence: Callable[[], Mapping[str, Any]] = _collect_inherited_evidence,
    collect_repository: Callable[[], Mapping[str, Any]] = _repository_authority,
) -> dict[str, Any]:
    """Replay inherited evidence and return a portable G0 campaign preview."""
    actions = {name: False for name in G0_ACTIONS}
    actions["run_g1a_cohort_gold_feasibility"] = True
    actions["run_g1b_acoustic_contract"] = True
    actions["run_g1c_context_contract"] = True
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "g0_ready_to_freeze",
        "plan_id": "0052",
        "plan_version": 1,
        "inherited_evidence": _validated_evidence(collect_evidence()),
        "repository_authority": _validated_repository(collect_repository()),
        "action_vector": actions,
        "delegation_receipt": {
            "status": "not_spawned",
            "lane": "G0",
            "reason": "critical_path_authority_replay_owned_by_primary",
            "runtime_handle": None,
        },
        "contains_paths": False,
        "contains_private_membership": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "did_read_private_gold": False,
        "did_load_or_run_models": False,
        "did_mutate_profiles_or_references": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation4-campaign-g0-preview-{digest[:24]}",
        "content_sha256": digest,
    }


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / "g0-authorities" / f"generation4-campaign-g0-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _receipt(preview: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "g0_frozen_g1_design_lanes_authorized",
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256,
        "plan_id": preview["plan_id"],
        "plan_version": preview["plan_version"],
        "inherited_evidence": dict(preview["inherited_evidence"]),
        "repository_authority": dict(preview["repository_authority"]),
        "action_vector": dict(preview["action_vector"]),
        "delegation_receipt": dict(preview["delegation_receipt"]),
        "contains_paths": False,
        "contains_private_membership": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "did_read_private_gold": False,
        "did_load_or_run_models": False,
        "did_mutate_profiles_or_references": False,
        "will_perform_external_write": False,
        "mode": "0600",
    }


def apply_generation4_campaign(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    collect_evidence: Callable[[], Mapping[str, Any]] = _collect_inherited_evidence,
    collect_repository: Callable[[], Mapping[str, Any]] = _repository_authority,
) -> dict[str, Any]:
    """Freeze one reviewed G0 preview or replay the exact existing authority."""
    preview = preview_generation4_campaign(
        collect_evidence=collect_evidence,
        collect_repository=collect_repository,
    )
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_content_sha256
    ):
        raise Generation4CampaignError(
            "Reviewed Generation-4 campaign G0 preview is stale."
        )
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation4_campaign(
            expected_content_sha256,
            runtime_root=runtime_root,
            collect_evidence=collect_evidence,
            collect_repository=collect_repository,
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen",
        "preview": preview,
        "contains_paths": False,
        "contains_private_membership": False,
    }
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation4_campaign(
    expected_content_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    collect_evidence: Callable[[], Mapping[str, Any]] = _collect_inherited_evidence,
    collect_repository: Callable[[], Mapping[str, Any]] = _repository_authority,
) -> dict[str, Any]:
    """Replay inherited evidence and the exact immutable G0 checkpoint."""
    preview = preview_generation4_campaign(
        collect_evidence=collect_evidence,
        collect_repository=collect_repository,
    )
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation4CampaignError(
            "Frozen Generation-4 campaign G0 preview drifted."
        )
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen",
        "preview": preview,
        "contains_paths": False,
        "contains_private_membership": False,
    }
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    expected_receipt = _receipt(preview, sha256_file(paths["manifest"]))
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation4CampaignError(
            "Generation-4 campaign G0 authority drifted."
        )
    return {
        **receipt,
        "replay_schema_version": REPLAY_SCHEMA,
        "replay_mode": "full_inherited_authority_replay_without_gold_or_models",
        "idempotent_replay": True,
    }
