"""Plan 0054 R1/R2 exact-membership one-pass recovery validation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_content_preservation as preservation
import acoustic_content_preservation_adversarial as adversarial
import acoustic_generation5_recovery_authority as r0
import acoustic_generation5_recovery_j0_acceptance as j0
from acoustic_audio_derivatives import ensure_private_tree, require_private_file, sha256_file, write_immutable_private_json


PREVIEW_SCHEMA = "transcribe-audio.generation5-recovery-validation-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-recovery-validation-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-recovery-validation-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-recovery-validation-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/r2")
R0_PREVIEW_SHA256 = j0.R0_PREVIEW_SHA256
R0_MANIFEST_SHA256 = j0.R0_MANIFEST_SHA256
J0_PREVIEW_SHA256 = "d76bb8aeb81d0cd3bafe13cf21c532ceaddc679bbf4edd499abed9ac15e8c521"
J0_MANIFEST_SHA256 = "21a572ea1d3c8f233973e717dc2ee9269c257ea72a56e9c8396710284f1ebfad"
CONTRACT_SHA256 = r0.CONTRACT_SHA256
MODULES = (
    "acoustic_content_preservation.py",
    "acoustic_content_preservation_adversarial.py",
    "acoustic_generation5_recovery_validation.py",
)


class Generation5RecoveryValidationError(ValueError):
    """Raised when exact R2 validation cannot freeze a complete passing body."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5RecoveryValidationError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5RecoveryValidationError("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *arguments], cwd=Path(__file__).resolve().parent, capture_output=True, text=not binary, check=False)
    if result.returncode:
        raise Generation5RecoveryValidationError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5RecoveryValidationError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5RecoveryValidationError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"])); hashes = {}
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation5RecoveryValidationError("Repository commit is invalid.")
    for name in MODULES:
        body = _git(["show", f"{commit}:{name}"], binary=True)
        if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve().parent / name):
            raise Generation5RecoveryValidationError("Committed module drifted.")
        hashes[name] = hashlib.sha256(body).hexdigest()
    return {"commit": commit, "module_sha256": hashes, "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def _parents() -> tuple[dict[str, Any], dict[str, Any]]:
    r0_replay = r0.replay_generation5_recovery_authority(R0_PREVIEW_SHA256)
    j0_replay = j0.replay_generation5_recovery_j0(J0_PREVIEW_SHA256)
    r0_paths = r0._paths(r0.DEFAULT_RUNTIME_ROOT, R0_PREVIEW_SHA256)
    j0_paths = j0._paths(j0.DEFAULT_RUNTIME_ROOT, J0_PREVIEW_SHA256)
    if (
        r0_replay.get("idempotent_replay") is not True
        or j0_replay.get("idempotent_replay") is not True
        or sha256_file(r0_paths["manifest"]) != R0_MANIFEST_SHA256
        or sha256_file(j0_paths["manifest"]) != J0_MANIFEST_SHA256
    ):
        raise Generation5RecoveryValidationError("Parent replay drifted.")
    r0_preview = _read_json(r0_paths["manifest"])["preview"]
    j0_preview = _read_json(j0_paths["manifest"])["preview"]
    return dict(r0_preview), dict(j0_preview)


def _execute_once(r0_preview: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = r0_preview["private_evidence"]["selected_membership"]
    if len(selected) != 8 or selected[0].get("role") != "recovery_negative_source" or any(item.get("role") != "positive_holdout" for item in selected[1:]):
        raise Generation5RecoveryValidationError("Exact R0 roles drifted.")
    positive = []
    for item in selected[1:]:
        measurement = preservation.measure(
            Path(str(item["path"])),
            expected_source_sha256=str(item["source_sha256"]),
            channel_policy_authority_sha256=J0_PREVIEW_SHA256,
        )
        positive.append({"source_sha256": item["source_sha256"], "ordinal": item["ordinal"], "measurement": measurement})
    negative_source = selected[0]
    negative = adversarial.run_recovery_adversaries(
        Path(str(negative_source["path"])),
        expected_source_sha256=str(negative_source["source_sha256"]),
        channel_policy_authority_sha256=J0_PREVIEW_SHA256,
    )
    return positive, negative


def _validate(positive: list[dict[str, Any]], negative: Mapping[str, Any]) -> None:
    if len(positive) != 7 or len({item["source_sha256"] for item in positive}) != 7:
        raise Generation5RecoveryValidationError("Positive denominator is invalid.")
    for item in positive:
        measurement = item["measurement"]
        if (
            measurement.get("status") != "passing"
            or measurement.get("reason_codes") != []
            or abs(int(measurement.get("output_sample_error") or 0)) > 1
            or measurement["recipe_reference_decode"]["pcm_sha256"] != measurement["production_wav"]["pcm_sha256"]
        ):
            raise Generation5RecoveryValidationError("Fresh positive holdout failed.")
    if (
        negative.get("seed") != adversarial.RECOVERY_HOLDOUT_SEED
        or negative.get("case_count") != 11
        or negative.get("expected_reason_contract") != adversarial.EXPECTED_REASON_CONTRACT
        or negative.get("expected_reason_contract_sha256") != _canonical_hash(adversarial.EXPECTED_REASON_CONTRACT)
        or negative.get("all_expected_rejections_observed") is not True
    ):
        raise Generation5RecoveryValidationError("Recovery negative denominator failed.")
    for case in negative.get("cases") or []:
        if case.get("status") != "rejected" or case.get("expected_reason_observed") is not True or case.get("expected_reason") not in adversarial.EXPECTED_REASON_CONTRACT.values():
            raise Generation5RecoveryValidationError("Recovery negative reason failed.")


def preview_generation5_recovery_validation(
    *,
    r0_preview: Mapping[str, Any] | None = None,
    j0_preview: Mapping[str, Any] | None = None,
    positive_results: list[dict[str, Any]] | None = None,
    negative_result: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if r0_preview is None or j0_preview is None:
        parent_r0, parent_j0 = _parents()
    else:
        parent_r0, parent_j0 = dict(r0_preview), dict(j0_preview)
    if parent_r0.get("content_sha256") != R0_PREVIEW_SHA256 or parent_j0.get("content_sha256") != J0_PREVIEW_SHA256 or parent_j0.get("status") != "accepted_for_exact_r1_r2_only":
        raise Generation5RecoveryValidationError("R2 parent authority is invalid.")
    if positive_results is None or negative_result is None:
        positive, negative = _execute_once(parent_r0)
    else:
        positive, negative = list(positive_results), dict(negative_result)
    _validate(positive, negative)
    public_negative = {key: value for key, value in negative.items() if key not in {"private_fixture_hashes", "private_case_measurements"}}
    actions = {
        "submit_to_independent_j2": True,
        "enumerate_evaluation_candidates": False,
        "access_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {"positive_holdout_measurements": positive, "recovery_negative": negative}
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_recovery_j2",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "r0_preview_sha256": R0_PREVIEW_SHA256,
        "j0_preview_sha256": J0_PREVIEW_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "positive_holdout_count": len(positive),
        "positive_holdout_pass_count": sum(item["measurement"]["status"] == "passing" for item in positive),
        "positive_results_sha256": _canonical_hash(positive),
        "recovery_negative": public_negative,
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": False,
        "contains_private_membership": True,
        "did_execute_exact_one_pass_r2": True,
        "did_enumerate_evaluation_candidates": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in preview.items() if key not in {"private_evidence", "repository_authority"}}
    value["schema_version"] = RECEIPT_SCHEMA; value["contains_private_membership"] = False
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute(); run = root / f"generation5-recovery-validation-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_recovery_validation(reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    preview = dict(reviewed_preview); core = {key: value for key, value in preview.items() if key != "content_sha256"}
    _parents()
    if preview.get("content_sha256") != expected_content_sha256 or _canonical_hash(core) != expected_content_sha256 or preview.get("repository_authority") != _repository_authority() or preview.get("positive_holdout_pass_count") != 7 or preview.get("recovery_negative", {}).get("all_expected_rejections_observed") is not True:
        raise Generation5RecoveryValidationError("Reviewed R2 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_recovery_validation(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_recovery_validation(expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"]); require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"]); receipt = _read_json(paths["receipt"]); preview = manifest.get("preview")
    if not isinstance(preview, Mapping): raise Generation5RecoveryValidationError("R2 preview is missing.")
    preview = dict(preview); core = {key: value for key, value in preview.items() if key != "content_sha256"}; _parents()
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if preview.get("content_sha256") != expected_content_sha256 or _canonical_hash(core) != expected_content_sha256 or manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5RecoveryValidationError("R2 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
