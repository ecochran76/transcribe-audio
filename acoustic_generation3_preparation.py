"""Prediction-blind Generation-3 P1/P2 preparation and condition freeze."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation3_evaluation as evaluation
import acoustic_successor_conditions as conditions
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-preparation-preview.v1"
AUTHORITY_SCHEMA = "transcribe-audio.generation3-preparation-authority.v1"
APPLICATION_SCHEMA = "transcribe-audio.generation3-preparation-application.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-preparation-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-preparation-replay.v1"
DEFAULT_RUNTIME_ROOT = evaluation.DEFAULT_RUNTIME_ROOT
MODULE_NAMES = (
    "acoustic_generation3_preparation.py",
    "acoustic_generation3_evaluation.py",
    "acoustic_audio_derivatives.py",
    "acoustic_speech_preparation.py",
    "acoustic_successor_conditions.py",
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation3PreparationError(ValueError):
    """Raised when prediction-blind preparation or conditions fail integrity."""


def _canonical_hash(value: Any) -> str:
    return evaluation._canonical_hash(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3PreparationError("Preparation JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3PreparationError("Preparation JSON must be an object.")
    return value


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise Generation3PreparationError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    behind, ahead = (
        int(item) for item in _git(
            ["rev-list", "--left-right", "--count", "@{upstream}...HEAD"]
        ).split()
    )
    root = Path(__file__).resolve().parent
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": {name: sha256_file(root / name) for name in MODULE_NAMES},
        "clean": _git(["status", "--porcelain"]) == "",
        "upstream_ahead": ahead, "upstream_behind": behind,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Generation3PreparationError("Preparation repository authority is invalid.")
    modules = value.get("module_sha256")
    commit = str(value.get("commit") or "")
    if (
        set(value) != {"commit", "module_sha256", "clean", "upstream_ahead", "upstream_behind"}
        or not COMMIT_RE.fullmatch(commit)
        or not isinstance(modules, Mapping) or set(modules) != set(MODULE_NAMES)
        or any(not SHA256_RE.fullmatch(str(item)) for item in modules.values())
        or value.get("clean") is not True or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3PreparationError("Preparation repository authority drifted.")
    root = Path(__file__).resolve().parent
    for name, digest in modules.items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{name}"], cwd=root,
            check=False, capture_output=True,
        )
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != digest or sha256_file(root / name) != digest:
            raise Generation3PreparationError("Preparation module authority drifted.")
    current = _repository_authority()
    if current["clean"] is not True or current["upstream_ahead"] != 0 or current["upstream_behind"] != 0:
        raise Generation3PreparationError("Current repository is not clean and exact.")
    return dict(value)


def _reveal_context(runtime_root: Path) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    paths = sorted(root.glob("evaluation-reveals/*/reveal-authority.json"))
    if len(paths) != 1:
        raise Generation3PreparationError("Exactly one reveal authority is required.")
    require_private_file(paths[0], root)
    authority = _read_object(paths[0])
    authority_sha = _canonical_hash(authority)
    try:
        replay = evaluation.replay_generation3_reveal_and_preflight(
            authority_sha, runtime_root=root
        )
    except (evaluation.Generation3EvaluationError, ValueError) as exc:
        raise Generation3PreparationError("Reveal preflight did not replay.") from exc
    if (
        replay.get("status") != "preflight_pass_prediction_blind_preparation_authorized"
        or replay.get("action_vector", {}).get("run_prediction_blind_p1_p2") is not True
        or replay.get("action_vector", {}).get("freeze_evaluation_windows") is not False
        or replay.get("action_vector", {}).get("load_or_run_models") is not False
    ):
        raise Generation3PreparationError("Reveal does not authorize P1/P2 only.")
    preview = authority.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation3PreparationError("Reveal preview is unavailable.")
    return {
        "authority_sha256": authority_sha,
        "preflight_sha256": replay["preflight_sha256"],
        "gold_manifest_sha256": replay["gold_manifest_sha256"],
        "cohort_manifest_sha256": preview["cohort_manifest_sha256"],
        "membership_sha256": preview["membership_sha256"],
        "receipt": replay,
    }


def _cohort_units(runtime_root: Path, reveal: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = runtime_root.expanduser().absolute()
    paths = sorted(root.glob("cohort-authorities/*/private-manifest.json"))
    if len(paths) != 1:
        raise Generation3PreparationError("Exactly one cohort authority is required.")
    require_private_file(paths[0], root)
    manifest = _read_object(paths[0])
    preview = manifest.get("preview")
    private = manifest.get("private_inputs")
    safe = preview.get("membership", {}).get("conversations") if isinstance(preview, Mapping) else None
    private_units = private.get("conversations") if isinstance(private, Mapping) else None
    if not isinstance(safe, list) or not isinstance(private_units, list) or len(safe) != 7:
        raise Generation3PreparationError("Cohort preparation membership is incomplete.")
    if (
        sha256_file(paths[0]) != reveal["cohort_manifest_sha256"]
        or preview.get("membership_sha256") != reveal["membership_sha256"]
    ):
        raise Generation3PreparationError("Cohort does not match reveal authority.")
    private_by_id = {str(item["conversation_input_id"]): item for item in private_units}
    units = []
    for item in safe:
        private_item = private_by_id.get(str(item["conversation_input_id"]))
        if not isinstance(private_item, Mapping):
            raise Generation3PreparationError("Private cohort source is unavailable.")
        units.append({
            "conversation_input_id": item["conversation_input_id"],
            "recording_id": item["recording_id"],
            "conversation_id": item["conversation_id"],
            "source_sha256": item["source_sha256"],
            "source_path": private_item["source_path"],
            "split": "evaluation",
        })
    if len({item["recording_id"] for item in units}) != 7 or len({item["conversation_id"] for item in units}) != 7:
        raise Generation3PreparationError("Cohort identities are not seven by seven.")
    return sorted(units, key=lambda item: str(item["recording_id"]))


def _preview_core(
    reveal: Mapping[str, Any], units: Sequence[Mapping[str, Any]], repository: Mapping[str, Any]
) -> dict[str, Any]:
    safe_units = [
        {key: item[key] for key in ("recording_id", "conversation_id", "source_sha256", "split")}
        for item in units
    ]
    return {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_prediction_blind_p1_p2",
        "reveal_authority_sha256": reveal["authority_sha256"],
        "preflight_sha256": reveal["preflight_sha256"],
        "unit_count": len(units),
        "safe_unit_set_sha256": _canonical_hash(safe_units),
        "private_units": [dict(item) for item in units],
        "method_ids": list(conditions.METHOD_IDS),
        "condition_fields": list(conditions.CONDITION_FIELDS),
        "condition_minimum_observed_values": 2,
        "condition_missing_recordings_allowed": 0,
        "repository_authority": dict(repository),
        "action_vector": {
            "freeze_preparation_authority": False,
            "run_prediction_blind_p1_p2": False,
            "measure_conditions": False,
            "freeze_evaluation_windows": False,
            "record_terminal_stop": False,
            "construct_exact_trial_child": False,
            "load_or_run_models": False,
            "score_evaluation_trials": False,
            "calculate_evaluation_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "did_read_audio": False, "did_prepare_audio": False,
        "did_load_or_run_models": False, "did_score_trials": False,
        "contains_paths": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False,
    }


def preview_generation3_preparation(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    reveal = _reveal_context(runtime_root)
    units = _cohort_units(runtime_root, reveal)
    core = _preview_core(reveal, units, _repository_authority())
    content_sha = _canonical_hash(core)
    return {**core, "preview_id": f"generation3-preparation-preview-{content_sha[:24]}", "content_sha256": content_sha}


def portable_preparation_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "transcribe-audio.generation3-preparation-portable.v1",
        "status": preview["status"], "preview_content_sha256": preview["content_sha256"],
        "reveal_authority_sha256": preview["reveal_authority_sha256"],
        "preflight_sha256": preview["preflight_sha256"],
        "safe_unit_set_sha256": preview["safe_unit_set_sha256"],
        "unit_count": preview["unit_count"], "method_count": len(preview["method_ids"]),
        "condition_field_count": len(preview["condition_fields"]),
        "action_vector": dict(preview["action_vector"]),
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_biometric_scores": False, "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False,
    }


def _paths(runtime_root: Path, authority_sha: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    directory = root / "evaluation-preparation" / f"generation3-preparation-{authority_sha[:24]}"
    return {
        "root": root, "directory": directory,
        "authority": directory / "authority.json", "application": directory / "application.json",
        "receipt": directory / "receipt.json", "p1": directory / "p1", "p2": directory / "p2",
    }


def _authority_body(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_SCHEMA, "preview": dict(preview),
        "repository_authority": dict(preview["repository_authority"]),
        "audio_read_permitted_only_after_this_file_exists": True,
        "did_read_audio": False, "did_prepare_audio": False,
        "did_load_or_run_models": False,
    }


def _write_authority(preview: Mapping[str, Any], runtime_root: Path) -> tuple[dict[str, Path], str]:
    body = _authority_body(preview)
    authority_sha = _canonical_hash(body)
    paths = _paths(runtime_root, authority_sha)
    ensure_private_tree(paths["root"], paths["directory"])
    if paths["authority"].exists():
        require_private_file(paths["authority"], paths["root"])
        if _read_object(paths["authority"]) != body:
            raise Generation3PreparationError("Preparation authority conflicts.")
    else:
        write_immutable_private_json(paths["authority"], body)
    return paths, authority_sha


def _receipt(preview: Mapping[str, Any], authority_sha: str, application_sha: str, application: Mapping[str, Any]) -> dict[str, Any]:
    actions = dict(preview["action_vector"])
    actions["freeze_preparation_authority"] = True
    actions["run_prediction_blind_p1_p2"] = True
    actions["measure_conditions"] = True
    eligible = application["condition_coverage"]["terminal_selection_eligible"] is True
    if eligible:
        actions["freeze_evaluation_windows"] = True
        status = "conditions_pass_window_freeze_authorized"
    else:
        actions["record_terminal_stop"] = True
        status = "conditions_blocked_terminal_stop_authorized"
    return {
        "schema_version": RECEIPT_SCHEMA, "status": status,
        "reason_codes": list(application["condition_coverage"]["blockers"]),
        "preparation_authority_sha256": authority_sha,
        "application_sha256": application_sha,
        "unit_count": application["unit_count"], "method_attempt_count": application["method_attempt_count"],
        "method_success_count": application["method_success_count"],
        "condition_coverage_sha256": _canonical_hash(application["condition_coverage"]),
        "action_vector": actions,
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_biometric_scores": False, "contains_embeddings_or_vectors": False,
        "mode": "0600",
    }


def _method_hashes(comparison: Mapping[str, Any]) -> dict[str, str]:
    method_results = comparison.get("method_results")
    if (
        not isinstance(method_results, list)
        or len(method_results) != len(conditions.METHOD_IDS)
        or any(
            not isinstance(result, Mapping) or result.get("status") != "success"
            for result in method_results
        )
    ):
        raise Generation3PreparationError("P2 method results are incomplete.")
    hashes = {
        str(result.get("method_id") or ""): conditions._canonical_hash(result)
        for result in method_results
    }
    if len(hashes) != len(method_results) or set(hashes) != set(conditions.METHOD_IDS):
        raise Generation3PreparationError("P2 method-result inventory drifted.")
    return hashes


def apply_generation3_preparation(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation3_preparation(runtime_root=runtime_root)
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_preview_content_sha256 or any(preview["action_vector"].values()):
        raise Generation3PreparationError("Reviewed preparation preview is stale.")
    _validate_repository_authority(preview["repository_authority"])
    paths, authority_sha = _write_authority(preview, runtime_root)
    if paths["application"].exists():
        return replay_generation3_preparation(authority_sha, runtime_root=runtime_root)
    execution_preview = {"corpus": {"content_sha256": preview["reveal_authority_sha256"]}}
    execution_paths = {"p1": paths["p1"], "p2": paths["p2"]}
    units = []
    for item in preview["private_units"]:
        try:
            units.append(conditions._execute_unit(item, execution_preview, execution_paths))
        except (conditions.SuccessorConditionError, ValueError) as exc:
            raise Generation3PreparationError("Prediction-blind P1/P2 failed closed.") from exc
    coverage = conditions._aggregate_conditions(units)
    application = {
        "schema_version": APPLICATION_SCHEMA, "status": "success",
        "preparation_authority_sha256": authority_sha,
        "unit_count": len(units), "method_attempt_count": len(units) * len(conditions.METHOD_IDS),
        "method_success_count": len(units) * len(conditions.METHOD_IDS),
        "units": units, "condition_coverage": coverage,
        "did_run_prediction_blind_p1_p2": True, "did_measure_conditions": True,
        "did_use_gold_for_condition_measurement": False,
        "did_load_or_run_models": False, "did_score_trials": False,
        "contains_paths": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False,
    }
    write_immutable_private_json(paths["application"], application)
    receipt = _receipt(preview, authority_sha, sha256_file(paths["application"]), application)
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "private_application_path": str(paths["application"]), "private_receipt_path": str(paths["receipt"]), "idempotent_replay": False}


def replay_generation3_preparation(
    preparation_authority_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(str(preparation_authority_sha256)):
        raise Generation3PreparationError("Preparation authority hash is invalid.")
    paths = _paths(runtime_root, preparation_authority_sha256)
    require_private_file(paths["authority"], paths["root"])
    authority = _read_object(paths["authority"])
    repository = _validate_repository_authority(authority.get("repository_authority"))
    reveal = _reveal_context(runtime_root)
    units = _cohort_units(runtime_root, reveal)
    core = _preview_core(reveal, units, repository)
    content_sha = _canonical_hash(core)
    preview = {**core, "preview_id": f"generation3-preparation-preview-{content_sha[:24]}", "content_sha256": content_sha}
    if _canonical_hash(authority) != preparation_authority_sha256 or authority != _authority_body(preview):
        raise Generation3PreparationError("Preparation authority drifted.")
    require_private_file(paths["application"], paths["root"])
    application = _read_object(paths["application"])
    stored_units = application.get("units")
    if not isinstance(stored_units, list) or len(stored_units) != 7:
        raise Generation3PreparationError("Preparation units are incomplete.")
    expected_safe = {(item["recording_id"], item["conversation_id"], item["source_sha256"]) for item in preview["private_units"]}
    actual_safe = {(item.get("recording_id"), item.get("conversation_id"), item.get("source_sha256")) for item in stored_units}
    source_by_recording = {
        str(item["recording_id"]): item for item in preview["private_units"]
    }
    for item in stored_units:
        source = source_by_recording.get(str(item.get("recording_id") or ""))
        if not isinstance(source, Mapping):
            raise Generation3PreparationError("Preparation source binding drifted.")
        if item.get("split") != "evaluation":
            raise Generation3PreparationError("Preparation split binding drifted.")
        source_path = Path(str(source["source_path"]))
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or sha256_file(source_path) != source["source_sha256"]
        ):
            raise Generation3PreparationError("Frozen preparation source bytes drifted.")
        artifact_bindings = (
            ("p1_manifest_path", "p1_manifest_sha256"),
            ("p1_replay_path", "p1_replay_sha256"),
            ("p2_comparison_path", "p2_comparison_sha256"),
            ("p2_replay_path", "p2_replay_sha256"),
        )
        for key, digest_key in artifact_bindings:
            path = Path(str(item.get(key) or ""))
            require_private_file(path, paths["directory"])
            if sha256_file(path) != item.get(digest_key):
                raise Generation3PreparationError("Preparation artifact hash drifted.")
        if set(item.get("method_result_sha256") or {}) != set(conditions.METHOD_IDS):
            raise Generation3PreparationError("Preparation method coverage drifted.")
        comparison = _read_object(Path(str(item["p2_comparison_path"])))
        if _method_hashes(comparison) != item["method_result_sha256"]:
            raise Generation3PreparationError("P2 method-result binding drifted.")
    coverage = conditions._aggregate_conditions(stored_units)
    expected_application = {
        **application,
        "condition_coverage": coverage,
    }
    expected_application_keys = {
        "schema_version", "status", "preparation_authority_sha256",
        "unit_count", "method_attempt_count", "method_success_count", "units",
        "condition_coverage", "did_run_prediction_blind_p1_p2",
        "did_measure_conditions", "did_use_gold_for_condition_measurement",
        "did_load_or_run_models", "did_score_trials", "contains_paths",
        "contains_raw_audio", "contains_transcript_text",
        "contains_biometric_scores", "contains_embeddings_or_vectors",
    }
    if (
        application != expected_application or actual_safe != expected_safe
        or set(application) != expected_application_keys
        or application.get("schema_version") != APPLICATION_SCHEMA
        or application.get("status") != "success"
        or application.get("preparation_authority_sha256") != preparation_authority_sha256
        or application.get("unit_count") != 7
        or application.get("method_attempt_count") != 35
        or application.get("method_success_count") != 35
        or application.get("did_run_prediction_blind_p1_p2") is not True
        or application.get("did_measure_conditions") is not True
        or application.get("did_use_gold_for_condition_measurement") is not False
        or application.get("did_load_or_run_models") is not False
        or application.get("did_score_trials") is not False
        or application.get("contains_paths") is not True
        or application.get("contains_raw_audio") is not False
        or application.get("contains_transcript_text") is not False
        or application.get("contains_biometric_scores") is not False
        or application.get("contains_embeddings_or_vectors") is not False
    ):
        raise Generation3PreparationError("Preparation application drifted.")
    require_private_file(paths["receipt"], paths["root"])
    receipt = _read_object(paths["receipt"])
    expected_receipt = _receipt(preview, preparation_authority_sha256, sha256_file(paths["application"]), application)
    if receipt != expected_receipt:
        raise Generation3PreparationError("Preparation receipt drifted.")
    return {**receipt, "private_application_path": str(paths["application"]), "private_receipt_path": str(paths["receipt"]), "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True, "replay_mode": "structural_without_audio_or_preparation_execution"}
