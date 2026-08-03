"""Freeze Plan 0054 J0 acceptance as exact R1/R2-only authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_recovery_authority as r0
from acoustic_audio_derivatives import ensure_private_tree, require_private_file, sha256_file, write_immutable_private_json


PREVIEW_SCHEMA = "transcribe-audio.generation5-recovery-j0-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-recovery-j0-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-recovery-j0-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-recovery-j0-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/j0")
R0_PREVIEW_SHA256 = "de59d3da1edce0e0e5e0050582cac442e39096bcb5ca30c1f57aa230e928d307"
R0_MANIFEST_SHA256 = "3fc8f06de8a098d8312fd7bc6dbe3f327dafce90524dff71c04542716641137e"
SELECTED_MEMBERSHIP_SHA256 = "172477eac32dbca0d2f3ffe6599f6b30167b0685ef692bb7ddc4c819bf689eb5"
REVIEWER_HANDLE = "/root/g5_recovery_j0"
MODULE_NAME = Path(__file__).name


class Generation5RecoveryJ0Error(ValueError):
    """Raised when J0 acceptance cannot be bound exactly."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5RecoveryJ0Error("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5RecoveryJ0Error("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *arguments], cwd=Path(__file__).resolve().parent, capture_output=True, text=not binary, check=False)
    if result.returncode:
        raise Generation5RecoveryJ0Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5RecoveryJ0Error("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5RecoveryJ0Error("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5RecoveryJ0Error("Committed J0 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(), "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _r0_preview() -> dict[str, Any]:
    replay = r0.replay_generation5_recovery_authority(R0_PREVIEW_SHA256)
    paths = r0._paths(r0.DEFAULT_RUNTIME_ROOT, R0_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != R0_MANIFEST_SHA256:
        raise Generation5RecoveryJ0Error("R0 replay drifted.")
    preview = _read_json(paths["manifest"]).get("preview")
    if not isinstance(preview, Mapping) or preview.get("selected_membership_sha256") != SELECTED_MEMBERSHIP_SHA256:
        raise Generation5RecoveryJ0Error("R0 membership drifted.")
    return dict(preview)


def preview_generation5_recovery_j0(
    *, r0_preview: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = dict(r0_preview or _r0_preview())
    if parent.get("content_sha256") != R0_PREVIEW_SHA256 or parent.get("selected_membership_sha256") != SELECTED_MEMBERSHIP_SHA256 or parent.get("did_decode_audio") is not False:
        raise Generation5RecoveryJ0Error("Reviewed R0 packet is invalid.")
    findings = {
        "plan0053_terminal_replay": True,
        "accepted_scientific_contract_unchanged": True,
        "prior_exclusion_union_complete": True,
        "selected_overlap_count": 0,
        "strict_recording_start_and_no_fallback": True,
        "metadata_only_probe": True,
        "exact_first_eight_order": True,
        "row_1_negative_rows_2_8_positive": True,
        "membership_and_source_disjoint": True,
        "recovery_seed_and_segment_disjoint": True,
        "expected_reason_map_literal": True,
        "portable_privacy_passed": True,
        "replay_passed": True,
    }
    actions = {
        "verify_exact_r1_membership": True,
        "run_exact_one_pass_r2": True,
        "enumerate_evaluation_candidates": False,
        "access_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "accepted_for_exact_r1_r2_only",
        "review_decision": "PASS",
        "reviewer_handle": REVIEWER_HANDLE,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "r0_preview_sha256": R0_PREVIEW_SHA256,
        "r0_manifest_sha256": R0_MANIFEST_SHA256,
        "selected_membership_sha256": SELECTED_MEMBERSHIP_SHA256,
        "findings": findings,
        "findings_sha256": _canonical_hash(findings),
        "action_vector": actions,
        "did_decode_audio": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in preview.items() if key != "repository_authority"}
    value["schema_version"] = RECEIPT_SCHEMA
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute(); run = root / f"generation5-recovery-j0-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_recovery_j0(reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    preview = preview_generation5_recovery_j0()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5RecoveryJ0Error("Reviewed J0 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_recovery_j0(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_recovery_j0(expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"]); require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"]); receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5RecoveryJ0Error("J0 preview is missing.")
    preview = dict(preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5RecoveryJ0Error("Recorded repository authority is missing.")
    commit = str(repository.get("commit") or "")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if re.fullmatch(r"[a-f0-9]{40}", commit) else b""
    r0.replay_generation5_recovery_authority(R0_PREVIEW_SHA256)
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != repository.get("module_sha256")
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5RecoveryJ0Error("J0 authority drifted.")
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5RecoveryJ0Error("J0 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
