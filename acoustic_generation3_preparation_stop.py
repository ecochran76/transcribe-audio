"""Record and replay the Generation-3 preparation terminal STOP."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import acoustic_generation3_preparation as preparation
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-preparation-stop-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation3-preparation-stop-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-preparation-stop-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-preparation-stop-replay.v1"
EXPECTED_PREPARATION_AUTHORITY = (
    "1bbee1bfbcc2648cefbb35393acfbebbfcad490a24352b0ac59a31c46796eb73"
)
EXPECTED_FAILED_RUN_ID = "audio-run-a566c9ed4d0a3262c493da9d"
EXPECTED_FAILED_SOURCE_SHA256 = (
    "affe7cee894110a12714fbd1c0d03247286f33522a6570cbc84fdb782588ad1a"
)
OBSERVED_SOURCE_DURATION_SECONDS = 3558.342104
OBSERVED_DECODED_DURATION_SECONDS = 3468.565313
EXPECTED_REASON_CODE = "p1_decoded_duration_drift_exceeds_frozen_tolerance"
EXPECTED_EXCEPTION_CLASS = "AudioDerivativeError"
EXPECTED_EXCEPTION_MESSAGE = "Decoded duration drift exceeds the frozen recipe tolerance."
POST_STOP_ACTIONS = (
    "retry_preparation", "measure_conditions", "freeze_evaluation_windows", "construct_exact_trial_child",
    "load_or_run_models", "score_evaluation_trials", "calculate_evaluation_metrics",
    "make_model_or_method_selection", "mutate_profiles_or_references",
    "enable_default_integration", "run_historical_reprocessing",
)
DEFAULT_RUNTIME_ROOT = preparation.DEFAULT_RUNTIME_ROOT
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation3PreparationStopError(ValueError):
    """Raised when the terminal preparation evidence cannot fail closed."""


def _canonical_hash(value: Any) -> str:
    return preparation._canonical_hash(value)


def _read_object(path: Path, root: Path) -> dict[str, Any]:
    require_private_file(path, root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3PreparationStopError("Terminal evidence is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3PreparationStopError("Terminal evidence must be an object.")
    return value


def _git(args: list[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=not binary,
    )
    if result.returncode:
        raise Generation3PreparationStopError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation3PreparationStopError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])).split() != ["0", "0"]:
        raise Generation3PreparationStopError("Repository must be upstream-even.")
    commit = str(_git(["log", "-1", "--format=%H", "--", Path(__file__).name]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation3PreparationStopError("Terminal module is not committed.")
    return {
        "commit": commit,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }


def _paths(runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / "evaluation-preparation" / (
        f"generation3-preparation-{EXPECTED_PREPARATION_AUTHORITY[:24]}"
    )
    stop = root / "terminal-stops" / (
        f"generation3-preparation-stop-{EXPECTED_PREPARATION_AUTHORITY[:24]}"
    )
    return {
        "root": root, "run": run, "authority": run / "authority.json",
        "application": run / "application.json", "receipt": run / "receipt.json",
        "stop": stop, "manifest": stop / "private-manifest.json",
        "stop_receipt": stop / "stop-receipt.json",
    }


def _validate_parent(paths: Mapping[str, Path]) -> dict[str, Any]:
    authority = _read_object(paths["authority"], paths["root"])
    if _canonical_hash(authority) != EXPECTED_PREPARATION_AUTHORITY:
        raise Generation3PreparationStopError("Preparation authority drifted.")
    repository = authority.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation3PreparationStopError("Parent repository authority is missing.")
    commit = str(repository.get("commit") or "")
    modules = repository.get("module_sha256")
    if not isinstance(modules, Mapping) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation3PreparationStopError("Parent repository authority is invalid.")
    if _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != "":
        raise Generation3PreparationStopError("Parent commit is not an ancestor.")
    for name, expected in modules.items():
        body = _git(["show", f"{commit}:{name}"], binary=True)
        if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != expected:
            raise Generation3PreparationStopError("Parent module authority drifted.")
    if paths["application"].exists() or paths["receipt"].exists():
        raise Generation3PreparationStopError("A successful preparation result already exists.")
    return authority


def _partial_state(paths: Mapping[str, Path]) -> dict[str, Any]:
    p1_runs = paths["run"] / "p1" / "runs"
    p2_runs = paths["run"] / "p2" / "runs"
    if not p1_runs.is_dir() or p1_runs.is_symlink() or not p2_runs.is_dir() or p2_runs.is_symlink():
        raise Generation3PreparationStopError("Partial run roots are invalid.")
    complete_p1 = {
        "dry-run.json", "recipe.json", "derived-audio.json", "apply-receipt.json",
        "audio-quality.json", "manifest.json", "replay-active-receipt.json",
    }
    p1 = sorted(item for item in p1_runs.iterdir() if item.is_dir() and not item.is_symlink())
    p2 = sorted(item for item in p2_runs.iterdir() if item.is_dir() and not item.is_symlink())
    if len(p1) != 7 or len(p2) != 6:
        raise Generation3PreparationStopError("Partial unit counts drifted.")
    completed = []
    failed = []
    for run in p1:
        names = {item.name for item in run.iterdir()}
        if names == complete_p1:
            manifest = _read_object(run / "manifest.json", paths["root"])
            applied = _read_object(run / "apply-receipt.json", paths["root"])
            replayed = _read_object(run / "replay-active-receipt.json", paths["root"])
            if (
                manifest.get("status") != "active"
                or manifest.get("eligible_for_identity") is not False
                or manifest.get("identity_eligibility_reason")
                != "usable_speech_not_assessed_until_p2"
                or applied.get("status") != "applied"
                or applied.get("source_unchanged") is not True
                or replayed.get("status") != "verified_active"
                or replayed.get("active") is not True
                or replayed.get("source_unchanged") is not True
            ):
                raise Generation3PreparationStopError("P1 success evidence drifted.")
            completed.append(run)
        elif names == {"dry-run.json"}:
            failed.append(run)
        else:
            raise Generation3PreparationStopError("P1 partial state drifted.")
    expected_p2 = {"dry-run.json", "comparison.json", "apply.json", "replay-active.json"}
    if len(completed) != 6 or len(failed) != 1 or failed[0].name != EXPECTED_FAILED_RUN_ID:
        raise Generation3PreparationStopError("Failed P1 boundary drifted.")
    for run in p2:
        if {item.name for item in run.iterdir()} != expected_p2:
            raise Generation3PreparationStopError("P2 partial state drifted.")
        comparison = _read_object(run / "comparison.json", paths["root"])
        applied = _read_object(run / "apply.json", paths["root"])
        replayed = _read_object(run / "replay-active.json", paths["root"])
        methods = comparison.get("method_results")
        if (
            comparison.get("status") != "success"
            or applied.get("status") != "success"
            or replayed.get("status") != "success"
            or not isinstance(methods, list) or len(methods) != 5
            or any(not isinstance(row, Mapping) or row.get("status") != "success" for row in methods)
            or {str(row.get("method_id") or "") for row in methods} != set(preparation.conditions.METHOD_IDS)
        ):
            raise Generation3PreparationStopError("P2 success evidence drifted.")
    failed_plan = _read_object(failed[0] / "dry-run.json", paths["root"])
    source = failed_plan.get("source")
    recipe = failed_plan.get("recipe")
    if not isinstance(source, Mapping) or not isinstance(recipe, Mapping):
        raise Generation3PreparationStopError("Failed P1 plan is invalid.")
    source_path = Path(str(source.get("path") or ""))
    tolerance = ((recipe.get("parameters") or {}).get("duration_tolerance_seconds"))
    if (
        source.get("sha256") != EXPECTED_FAILED_SOURCE_SHA256
        or not source_path.is_file() or source_path.is_symlink()
        or sha256_file(source_path) != EXPECTED_FAILED_SOURCE_SHA256
        or (source.get("probe") or {}).get("duration_seconds") != OBSERVED_SOURCE_DURATION_SECONDS
        or tolerance != 0.05
    ):
        raise Generation3PreparationStopError("Failed source binding drifted.")
    inventory = []
    for base in (paths["run"] / "p1", paths["run"] / "p2"):
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            require_private_file(path, paths["root"])
            inventory.append({
                "relative_path": path.relative_to(paths["run"]).as_posix(),
                "sha256": sha256_file(path), "bytes": path.stat().st_size,
            })
    drift = round(abs(OBSERVED_SOURCE_DURATION_SECONDS - OBSERVED_DECODED_DURATION_SECONDS), 6)
    downstream_names = {
        "evaluation-windows", "exact-trials", "evaluation-scores",
        "evaluation-metrics", "evaluation-decisions",
    }
    existing_downstream = sorted(
        name for name in downstream_names if (paths["root"] / name).exists()
    )
    if existing_downstream:
        raise Generation3PreparationStopError("Downstream evaluation state already exists.")
    return {
        "attempted_unit_count": 7, "completed_p1_unit_count": 6,
        "completed_p2_unit_count": 6, "completed_p2_method_count": 30,
        "failed_run_id": EXPECTED_FAILED_RUN_ID,
        "failed_source_sha256": EXPECTED_FAILED_SOURCE_SHA256,
        "source_duration_seconds": OBSERVED_SOURCE_DURATION_SECONDS,
        "decoded_duration_seconds": OBSERVED_DECODED_DURATION_SECONDS,
        "duration_drift_seconds": drift, "duration_tolerance_seconds": tolerance,
        "exception_class": EXPECTED_EXCEPTION_CLASS,
        "exception_message": EXPECTED_EXCEPTION_MESSAGE,
        "explicit_absences": {
            "failed_p1_outputs": True, "seventh_p2_unit": True,
            "preparation_application": True, "preparation_receipt": True,
            "condition_freeze": True, "evaluation_windows": True,
            "exact_trials": True, "scores": True, "metrics": True,
            "terminal_decision": True,
        },
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": _canonical_hash(inventory),
    }


def preview_generation3_preparation_stop(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    _validate_parent(paths)
    state = _partial_state(paths)
    observation = {
        "provenance": "operator_reviewed_runtime_observation",
        "source_duration_seconds": state["source_duration_seconds"],
        "decoded_duration_seconds": state["decoded_duration_seconds"],
        "duration_drift_seconds": state["duration_drift_seconds"],
        "duration_tolerance_seconds": state["duration_tolerance_seconds"],
        "exception_class": state["exception_class"],
        "exception_message": state["exception_message"],
        "did_recompute_audio": False,
    }
    actions = {key: False for key in POST_STOP_ACTIONS}
    actions["record_terminal_stop"] = True
    core = {
        "schema_version": PREVIEW_SCHEMA, "status": "terminal_stop_required",
        "reason_code": EXPECTED_REASON_CODE,
        "preparation_authority_sha256": EXPECTED_PREPARATION_AUTHORITY,
        "attempted_unit_count": state["attempted_unit_count"],
        "completed_p1_unit_count": state["completed_p1_unit_count"],
        "completed_p2_unit_count": state["completed_p2_unit_count"],
        "completed_p2_method_count": state["completed_p2_method_count"],
        "failure_observation": observation,
        "failure_observation_sha256": _canonical_hash(observation),
        "artifact_inventory_sha256": state["artifact_inventory_sha256"],
        "repository_authority": _repository_authority(),
        "authorized_actions": actions,
        "contains_private_evaluation": True,
        "contains_reviewed_failure_observation": True,
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_biometric_scores": False, "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {**core, "preview_id": f"generation3-preparation-stop-{digest[:24]}", "content_sha256": digest}


def portable_stop_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "transcribe-audio.generation3-preparation-stop-portable.v1",
        "status": preview["status"], "reason_code": preview["reason_code"],
        "preview_content_sha256": preview["content_sha256"],
        "preparation_authority_sha256": preview["preparation_authority_sha256"],
        "failure_observation_sha256": preview["failure_observation_sha256"],
        "artifact_inventory_sha256": preview["artifact_inventory_sha256"],
        "attempted_unit_count": preview["attempted_unit_count"],
        "completed_p1_unit_count": preview["completed_p1_unit_count"],
        "completed_p2_unit_count": preview["completed_p2_unit_count"],
        "completed_p2_method_count": preview["completed_p2_method_count"],
        "authorized_actions": dict(preview["authorized_actions"]),
        "contains_private_evaluation": False, "contains_paths": False,
        "contains_private_membership": False, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False, "will_perform_external_write": False,
    }


def apply_generation3_preparation_stop(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation3_preparation_stop(runtime_root=runtime_root)
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation3PreparationStopError("Reviewed STOP preview is stale.")
    paths = _paths(runtime_root)
    if paths["manifest"].exists() or paths["stop_receipt"].exists():
        return replay_generation3_preparation_stop(runtime_root=runtime_root)
    state = _partial_state(paths)
    core = {
        "schema_version": MANIFEST_SCHEMA, "status": "terminal_stop",
        "reason_code": EXPECTED_REASON_CODE, "preview": preview,
        "private_failure_evidence": state,
        "repository_authority": preview["repository_authority"],
        "authorized_actions_after_stop": {key: False for key in POST_STOP_ACTIONS},
        "contains_private_evaluation": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False, "will_perform_external_write": False,
    }
    ensure_private_tree(paths["root"], paths["stop"])
    write_immutable_private_json(paths["manifest"], core)
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "terminal_stop",
        "reason_code": EXPECTED_REASON_CODE,
        "preparation_authority_sha256": EXPECTED_PREPARATION_AUTHORITY,
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "attempted_unit_count": 7, "completed_p1_unit_count": 6,
        "completed_p2_unit_count": 6, "completed_p2_method_count": 30,
        "failure_observation_sha256": preview["failure_observation_sha256"],
        "artifact_inventory_sha256": state["artifact_inventory_sha256"],
        "action_vector": {key: False for key in POST_STOP_ACTIONS},
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_biometric_scores": False, "contains_embeddings_or_vectors": False,
        "mode": "0600", "will_perform_external_write": False,
    }
    write_immutable_private_json(paths["stop_receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation3_preparation_stop(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    paths = _paths(runtime_root)
    preview = preview_generation3_preparation_stop(runtime_root=runtime_root)
    state = _partial_state(paths)
    manifest = _read_object(paths["manifest"], paths["root"])
    receipt = _read_object(paths["stop_receipt"], paths["root"])
    expected_manifest = {
        "schema_version": MANIFEST_SCHEMA, "status": "terminal_stop",
        "reason_code": EXPECTED_REASON_CODE, "preview": preview,
        "private_failure_evidence": state,
        "repository_authority": preview["repository_authority"],
        "authorized_actions_after_stop": {key: False for key in POST_STOP_ACTIONS},
        "contains_private_evaluation": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False, "will_perform_external_write": False,
    }
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": "terminal_stop",
        "reason_code": EXPECTED_REASON_CODE,
        "preparation_authority_sha256": EXPECTED_PREPARATION_AUTHORITY,
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "attempted_unit_count": 7, "completed_p1_unit_count": 6,
        "completed_p2_unit_count": 6, "completed_p2_method_count": 30,
        "failure_observation_sha256": preview["failure_observation_sha256"],
        "artifact_inventory_sha256": state["artifact_inventory_sha256"],
        "action_vector": {key: False for key in POST_STOP_ACTIONS},
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_biometric_scores": False, "contains_embeddings_or_vectors": False,
        "mode": "0600", "will_perform_external_write": False,
    }
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation3PreparationStopError("Terminal STOP evidence drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA,
            "idempotent_replay": True, "replay_mode": "full_body_without_audio_execution"}
