"""Freeze Plan 0054 J2 PASS as E1-only authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_recovery_validation as r2
from acoustic_audio_derivatives import ensure_private_tree, require_private_file, sha256_file, write_immutable_private_json


PREVIEW_SCHEMA = "transcribe-audio.generation5-recovery-j2-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-recovery-j2-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-recovery-j2-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-recovery-j2-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/j2")
R2_PREVIEW_SHA256 = "2ff4304953642e99d2c765779b878fb6d378d77fa2595d80549081add2cd1c00"
R2_MANIFEST_SHA256 = "29cd6e6dfc9b708fa90f807865e86f60179c00d8023cf3d522e8baf772495445"
REVIEWER_HANDLE = "/root/g5_recovery_j2"
MODULE_NAME = Path(__file__).name


class Generation5RecoveryJ2Error(ValueError):
    """Raised when the recovery J2 PASS cannot be frozen exactly."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5RecoveryJ2Error("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5RecoveryJ2Error("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Generation5RecoveryJ2Error("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5RecoveryJ2Error("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5RecoveryJ2Error("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if (
        not re.fullmatch(r"[a-f0-9]{40}", commit)
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve())
    ):
        raise Generation5RecoveryJ2Error("Committed J2 module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(), "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _r2_preview() -> dict[str, Any]:
    replay = r2.replay_generation5_recovery_validation(R2_PREVIEW_SHA256)
    paths = r2._paths(r2.DEFAULT_RUNTIME_ROOT, R2_PREVIEW_SHA256)
    if replay.get("idempotent_replay") is not True or sha256_file(paths["manifest"]) != R2_MANIFEST_SHA256:
        raise Generation5RecoveryJ2Error("R2 replay drifted.")
    preview = _read_json(paths["manifest"])["preview"]
    if (
        preview.get("positive_holdout_pass_count") != 7
        or preview.get("recovery_negative", {}).get("all_expected_rejections_observed") is not True
    ):
        raise Generation5RecoveryJ2Error("R2 denominator drifted.")
    return dict(preview)


def preview_generation5_recovery_j2(
    *,
    r2_preview: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = dict(r2_preview or _r2_preview())
    if parent.get("content_sha256") != R2_PREVIEW_SHA256:
        raise Generation5RecoveryJ2Error("Reviewed R2 packet is invalid.")
    findings = {
        "exact_membership_reproduced": True,
        "positive_holdout_pass_count": 7,
        "literal_negative_reason_pass_count": 11,
        "corrupt_tail_expected_reason": "measurement_error",
        "observed_to_expected_assignment": False,
        "seed_and_segment_disjoint": True,
        "contract_and_tool_binding": True,
        "privacy_and_replay": True,
    }
    actions = {
        "enumerate_e1_candidates": True,
        "establish_private_gold_feasibility": True,
        "run_models_or_predictions": False,
        "reveal_gold_to_workers": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "accepted_for_e1_only",
        "review_decision": "PASS",
        "reviewer_handle": REVIEWER_HANDLE,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "r2_preview_sha256": R2_PREVIEW_SHA256,
        "r2_manifest_sha256": R2_MANIFEST_SHA256,
        "findings": findings,
        "findings_sha256": _canonical_hash(findings),
        "action_vector": actions,
        "did_enumerate_evaluation_candidates": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in preview.items() if key != "repository_authority"}
    value["schema_version"] = RECEIPT_SCHEMA
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-recovery-j2-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_recovery_j2(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_recovery_j2()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5RecoveryJ2Error("Reviewed recovery J2 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_recovery_j2(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_recovery_j2(expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5RecoveryJ2Error("Recovery J2 preview is missing.")
    preview = dict(preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5RecoveryJ2Error("Recorded repository authority is missing.")
    commit = str(repository.get("commit") or "")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if re.fullmatch(r"[a-f0-9]{40}", commit) else b""
    _r2_preview()
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != repository.get("module_sha256")
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5RecoveryJ2Error("Recovery J2 authority drifted.")
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5RecoveryJ2Error("Recovery J2 body drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
