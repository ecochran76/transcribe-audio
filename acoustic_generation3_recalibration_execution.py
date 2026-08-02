"""Self-bound Generation-3 successor recalibration scoring and threshold freeze."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import acoustic_generation3_recalibration as recalibration
import acoustic_verification as verification
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-recalibration-execution-preview.v1"
AUTHORITY_SCHEMA = "transcribe-audio.generation3-recalibration-execution-authority.v1"
SCORE_MATRIX_SCHEMA = "transcribe-audio.generation3-recalibration-score-matrix.v1"
SCORE_RECEIPT_SCHEMA = "transcribe-audio.generation3-recalibration-score-receipt.v1"
THRESHOLD_APPLICATION_SCHEMA = (
    "transcribe-audio.generation3-recalibration-threshold-application.v1"
)
THRESHOLD_RECEIPT_SCHEMA = (
    "transcribe-audio.generation3-recalibration-threshold-receipt.v1"
)
REPLAY_SCHEMA = "transcribe-audio.generation3-recalibration-execution-replay.v1"
EXECUTOR_MODULE = "acoustic_generation3_recalibration_execution.py"
DEFAULT_RUNTIME_ROOT = recalibration.DEFAULT_RUNTIME_ROOT
DEFAULT_CALIBRATION_ROOT = recalibration.DEFAULT_CALIBRATION_ROOT
DEFAULT_P3_RUNTIME_ROOT = recalibration.DEFAULT_P3_RUNTIME_ROOT
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation3RecalibrationExecutionError(ValueError):
    """Raised when successor scoring or threshold freeze cannot remain exact."""


def _canonical_hash(value: Any) -> str:
    return recalibration._canonical_hash(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3RecalibrationExecutionError(
            "Generation-3 recalibration execution JSON is unreadable."
        ) from exc
    if not isinstance(value, dict):
        raise Generation3RecalibrationExecutionError(
            "Generation-3 recalibration execution JSON must be an object."
        )
    return value


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise Generation3RecalibrationExecutionError(
            "Execution repository authority is unavailable."
        )
    return completed.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    upstream = _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    behind, ahead = (int(item) for item in upstream.split())
    module_path = Path(__file__).resolve()
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_name": EXECUTOR_MODULE,
        "module_sha256": sha256_file(module_path),
        "clean": _git(["status", "--porcelain"]) == "",
        "upstream_ahead": ahead,
        "upstream_behind": behind,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Generation3RecalibrationExecutionError(
            "Execution repository authority is invalid."
        )
    commit = str(value.get("commit") or "")
    digest = str(value.get("module_sha256") or "")
    if (
        set(value)
        != {
            "commit",
            "module_name",
            "module_sha256",
            "clean",
            "upstream_ahead",
            "upstream_behind",
        }
        or not COMMIT_RE.fullmatch(commit)
        or value.get("module_name") != EXECUTOR_MODULE
        or not SHA256_RE.fullmatch(digest)
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3RecalibrationExecutionError(
            "Execution repository authority drifted."
        )
    blob = subprocess.run(
        ["git", "show", f"{commit}:{EXECUTOR_MODULE}"],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
    )
    current = _repository_authority()
    if (
        blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest() != digest
        or sha256_file(Path(__file__).resolve()) != digest
        or current["clean"] is not True
        or current["upstream_ahead"] != 0
        or current["upstream_behind"] != 0
    ):
        raise Generation3RecalibrationExecutionError(
            "Execution module or current repository drifted."
        )
    return dict(value)


def _authority_manifest_path(runtime_root: Path) -> Path:
    path = recalibration._existing_manifest(runtime_root)
    if path is None:
        raise Generation3RecalibrationExecutionError(
            "Frozen Generation-3 recalibration authority is unavailable."
        )
    return path


def _authority_context(
    *,
    runtime_root: Path,
    authority_manifest_path: Optional[Path] = None,
    calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = recalibration.DEFAULT_CORPUS_MANIFESTS,
) -> dict[str, Any]:
    path = (authority_manifest_path or _authority_manifest_path(runtime_root))
    path = path.expanduser().absolute()
    try:
        replay = recalibration.replay_generation3_recalibration_authority(
            path,
            runtime_root=runtime_root,
            calibration_root=calibration_root,
            p3_runtime_root=p3_runtime_root,
            corpus_manifest_paths=corpus_manifest_paths,
        )
    except (recalibration.Generation3RecalibrationError, ValueError) as exc:
        raise Generation3RecalibrationExecutionError(
            "Frozen Generation-3 recalibration authority did not replay."
        ) from exc
    manifest = _read_object(path)
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or replay.get("idempotent_replay") is not True
        or replay.get("action_vector", {}).get("run_calibration_models") is not True
        or replay.get("action_vector", {}).get(
            "freeze_thresholds_and_temperatures"
        )
        is not False
    ):
        raise Generation3RecalibrationExecutionError(
            "Frozen recalibration authority does not authorize exact scoring."
        )
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "authority_id": manifest["authority_id"],
        "authority_content_sha256": manifest["content_sha256"],
        "preview": dict(preview),
    }


def _execution_paths(runtime_root: Path, execution_id: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    directory = root / "recalibration-executions" / execution_id
    return {
        "root": root,
        "directory": directory,
        "authority": directory / "execution-authority.json",
        "score_matrix": directory / "score-matrix.json",
        "score_receipt": directory / "score-receipt.json",
        "threshold_application": directory / "threshold-application.json",
        "threshold_receipt": directory / "threshold-receipt.json",
    }


def _preview_core(context: Mapping[str, Any], repository: Mapping[str, Any]) -> dict[str, Any]:
    frozen = context["preview"]
    return {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_self_bound_calibration_scoring",
        "recalibration_authority_id_sha256": _canonical_hash(context["authority_id"]),
        "recalibration_manifest_sha256": context["manifest_sha256"],
        "recalibration_content_sha256": context["authority_content_sha256"],
        "repository_authority": dict(repository),
        "calibration_membership_sha256": _canonical_hash(
            frozen["historical_calibration"]["calibration_dimensions"]
        ),
        "window_selection_sha256": frozen["historical_calibration"][
            "window_selection_sha256"
        ],
        "preparation_sha256": frozen["historical_calibration"]["preparation_sha256"],
        "profile_set_sha256": frozen["active_profile_authority"]["profile_set_sha256"],
        "model_asset_set_sha256": frozen["active_profile_authority"][
            "model_asset_set_sha256"
        ],
        "window_count": frozen["historical_calibration"]["window_count"],
        "profile_count": frozen["active_profile_authority"]["profile_count"],
        "candidate_count": frozen["active_profile_authority"]["candidate_count"],
        "method_count": len(frozen["historical_calibration"]["score_methods"]),
        "unit_count": frozen["unit_count"],
        "expected_trial_count": (
            frozen["unit_count"] * frozen["expected_trials_per_unit"]
        ),
        "expected_trials_per_unit": frozen["expected_trials_per_unit"],
        "expected_genuine_trials_per_unit": frozen[
            "expected_genuine_trials_per_unit"
        ],
        "expected_impostor_trials_per_unit": frozen[
            "expected_impostor_trials_per_unit"
        ],
        "expected_open_set_trials_per_unit": frozen[
            "expected_open_set_trials_per_unit"
        ],
        "abstention_margin_is_zero": frozen["abstention_margin"] == 0.0,
        "action_vector": {
            "run_calibration_models": False,
            "persist_private_score_matrix": False,
            "freeze_thresholds_and_temperatures": False,
            "build_pre_reveal_envelope": False,
            "reveal_evaluation": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "will_perform_external_write": False,
    }


def preview_generation3_recalibration_execution(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT, **authority_inputs: Any
) -> dict[str, Any]:
    """Replay frozen authority and preview self-bound scoring without model load."""
    context = _authority_context(runtime_root=runtime_root, **authority_inputs)
    core = _preview_core(context, _repository_authority())
    content_sha = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-recalibration-execution-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }


def _authority_body(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_SCHEMA,
        "preview": dict(preview),
        "repository_authority": dict(preview["repository_authority"]),
        "model_load_permitted_only_after_this_file_exists": True,
        "did_load_model": False,
        "did_read_generation3_gold_or_audio": False,
    }


def _write_or_replay_execution_authority(
    preview: Mapping[str, Any], *, runtime_root: Path
) -> tuple[dict[str, Path], str]:
    body = _authority_body(preview)
    authority_sha = _canonical_hash(body)
    execution_id = f"generation3-recalibration-execution-{authority_sha[:24]}"
    paths = _execution_paths(runtime_root, execution_id)
    ensure_private_tree(paths["root"], paths["directory"])
    if paths["authority"].exists():
        require_private_file(paths["authority"], paths["root"])
        if _read_object(paths["authority"]) != body:
            raise Generation3RecalibrationExecutionError(
                "Execution authority conflicts with the reviewed preview."
            )
    else:
        write_immutable_private_json(paths["authority"], body)
    if _canonical_hash(_read_object(paths["authority"])) != authority_sha:
        raise Generation3RecalibrationExecutionError(
            "Execution authority identity changed before model load."
        )
    return paths, authority_sha


def _private_historical_context(
    *, calibration_root: Path, corpus_manifest_paths: Sequence[Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        _, private, _, _ = recalibration._historical_context(
            calibration_root=calibration_root,
            corpus_manifest_paths=corpus_manifest_paths,
        )
    except (recalibration.Generation3RecalibrationError, ValueError) as exc:
        raise Generation3RecalibrationExecutionError(
            "Private historical calibration context did not replay."
        ) from exc
    selection = private.get("selection")
    preparation = private.get("preparation")
    if not isinstance(selection, Mapping) or not isinstance(preparation, Mapping):
        raise Generation3RecalibrationExecutionError(
            "Private historical calibration stages are unavailable."
        )
    return dict(selection), dict(preparation)


def _expected_inventory(preview: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = {
        str(item["profile_id"]): dict(item) for item in preview["profiles"]
    }
    units = {
        (str(item["candidate_id"]), str(item["method_id"])): dict(item)
        for item in preview["units"]
    }
    if len(profiles) != recalibration.EXPECTED_PROFILE_COUNT or len(units) != 9:
        raise Generation3RecalibrationExecutionError(
            "Frozen profile or unit inventory changed."
        )
    return profiles, units


def _validate_unit_denominators(
    trials: Sequence[Mapping[str, Any]], preview: Mapping[str, Any]
) -> None:
    expected = (
        preview["expected_trials_per_unit"],
        preview["expected_genuine_trials_per_unit"],
        preview["expected_impostor_trials_per_unit"],
        preview["expected_open_set_trials_per_unit"],
    )
    observed_units = set()
    for candidate_id in recalibration.CANDIDATE_IDS:
        for method_id in recalibration.METHOD_IDS:
            unit = [
                item
                for item in trials
                if item["candidate_id"] == candidate_id
                and item["method_id"] == method_id
            ]
            observed = (
                len(unit),
                sum(item["expected_match"] is True for item in unit),
                sum(item["expected_match"] is False for item in unit),
                sum(item["open_set_probe"] is True for item in unit),
            )
            if observed != expected:
                raise Generation3RecalibrationExecutionError(
                    "Calibration unit denominators are incomplete."
                )
            observed_units.add((candidate_id, method_id))
    if len(observed_units) != 9:
        raise Generation3RecalibrationExecutionError(
            "Calibration unit coverage is incomplete."
        )


def _score_receipt(
    *, preview: Mapping[str, Any], authority_sha: str, matrix_sha: str
) -> dict[str, Any]:
    actions = dict(preview["action_vector"])
    actions["run_calibration_models"] = True
    actions["persist_private_score_matrix"] = True
    actions["freeze_thresholds_and_temperatures"] = True
    return {
        "schema_version": SCORE_RECEIPT_SCHEMA,
        "status": "calibration_scores_frozen_threshold_freeze_authorized",
        "execution_authority_sha256": authority_sha,
        "score_matrix_sha256": matrix_sha,
        "logical_trial_count": preview["expected_trial_count"],
        "unit_count": preview["unit_count"],
        "trials_per_unit": preview["expected_trials_per_unit"],
        "genuine_trials_per_unit": preview["expected_genuine_trials_per_unit"],
        "impostor_trials_per_unit": preview["expected_impostor_trials_per_unit"],
        "open_set_trials_per_unit": preview["expected_open_set_trials_per_unit"],
        "action_vector": actions,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "mode": "0600",
    }


def apply_generation3_recalibration_scores(
    reviewed_preview: Mapping[str, Any],
    *,
    expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = recalibration.DEFAULT_CORPUS_MANIFESTS,
    adapters: Optional[Mapping[str, verification.VerificationAdapter]] = None,
    test_mode: bool = False,
    authority_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist self authority, then execute the exact 396 private trials."""
    if adapters is not None and not test_mode:
        raise Generation3RecalibrationExecutionError(
            "Custom recalibration adapters are test-only."
        )
    preview = preview_generation3_recalibration_execution(
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
    ):
        raise Generation3RecalibrationExecutionError(
            "Reviewed execution preview is stale."
        )
    _validate_repository_authority(preview["repository_authority"])
    paths, authority_sha = _write_or_replay_execution_authority(
        preview, runtime_root=runtime_root
    )
    if paths["score_matrix"].exists():
        return replay_generation3_recalibration_scores(
            authority_sha,
            runtime_root=runtime_root,
            calibration_root=calibration_root,
            p3_runtime_root=p3_runtime_root,
            corpus_manifest_paths=corpus_manifest_paths,
            authority_manifest_path=authority_manifest_path,
        )
    context = _authority_context(
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    frozen = context["preview"]
    profiles, _ = _expected_inventory(frozen)
    selection, preparation = _private_historical_context(
        calibration_root=calibration_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    windows = selection.get("windows")
    if not isinstance(windows, list) or len(windows) != recalibration.EXPECTED_WINDOW_COUNT:
        raise Generation3RecalibrationExecutionError(
            "Exact historical calibration windows are unavailable."
        )
    selected_adapters = dict(adapters or verification.adapter_registry())
    expected_models = {
        str(profile["candidate_id"]): str(profile["model_revision"])
        for profile in profiles.values()
    }
    if set(selected_adapters) != set(expected_models) or any(
        selected_adapters[key].revision_sha != expected_models[key]
        for key in expected_models
    ):
        raise Generation3RecalibrationExecutionError(
            "Successor calibration model inventory drifted."
        )
    profiles_by_model = {
        candidate_id: sorted(
            [p for p in profiles.values() if p["candidate_id"] == candidate_id],
            key=lambda item: str(item["profile_id"]),
        )
        for candidate_id in sorted(expected_models)
    }
    known_subjects = {str(item["person_ref_id"]) for item in profiles.values()}
    trials: list[dict[str, Any]] = []
    for window in windows:
        for method_id in recalibration.METHOD_IDS:
            try:
                samples = verification._calibration_pcm_window(
                    preparation, window, method_id
                )
            except verification.AcousticVerificationError as exc:
                raise Generation3RecalibrationExecutionError(
                    "Frozen calibration PCM window did not replay."
                ) from exc
            for candidate_id in recalibration.CANDIDATE_IDS:
                adapter = selected_adapters[candidate_id]
                for profile in profiles_by_model[candidate_id]:
                    try:
                        scored = verification.score_profile(
                            str(profile["profile_id"]),
                            adapter=adapter,
                            probe_samples=samples,
                            sample_rate=16_000,
                            runtime_root=calibration_root,
                            p3_runtime_root=p3_runtime_root,
                        )
                    except verification.AcousticVerificationError as exc:
                        raise Generation3RecalibrationExecutionError(
                            "Successor calibration scoring failed closed."
                        ) from exc
                    score = scored.get("score")
                    if (
                        scored.get("status") != "success"
                        or isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        or not -1.0 <= float(score) <= 1.0
                    ):
                        raise Generation3RecalibrationExecutionError(
                            "Successor calibration score is invalid."
                        )
                    expected_match = (
                        window["subject_id"] == profile["person_ref_id"]
                    )
                    identity = {
                        "execution_authority_sha256": authority_sha,
                        "window_id": window["window_id"],
                        "method_id": method_id,
                        "profile_id": profile["profile_id"],
                        "score_trial_id": scored["trial_id"],
                    }
                    trials.append(
                        {
                            "trial_id": "generation3-calibration-trial-"
                            + _canonical_hash(identity)[:24],
                            "status": "success",
                            "reason_code": None,
                            "window_id": window["window_id"],
                            "recording_id": window["recording_id"],
                            "conversation_id": window["conversation_id"],
                            "probe_subject_id": window["subject_id"],
                            "profile_person_ref_id": profile["person_ref_id"],
                            "expected_match": expected_match,
                            "open_set_probe": window["subject_id"] not in known_subjects,
                            "method_id": method_id,
                            "profile_id": profile["profile_id"],
                            "descendant_id": profile["descendant_id"],
                            "candidate_id": candidate_id,
                            "model_revision": adapter.revision_sha,
                            "probe_sha256": scored["probe_sha256"],
                            "score_trial_id": scored["trial_id"],
                            "score": float(score),
                            "conditions": dict(window["conditions"]),
                            "p4_state_verified_before_and_after": True,
                            "p3_eligibility_verified_before_and_after": True,
                            "contains_raw_biometric_values": False,
                        }
                    )
    trials.sort(key=lambda item: item["trial_id"])
    if (
        len(trials) != preview["expected_trial_count"]
        or len({item["trial_id"] for item in trials}) != len(trials)
    ):
        raise Generation3RecalibrationExecutionError(
            "Successor calibration trial coverage is invalid."
        )
    _validate_unit_denominators(trials, preview)
    matrix = {
        "schema_version": SCORE_MATRIX_SCHEMA,
        "status": "success",
        "reason_code": None,
        "execution_authority_sha256": authority_sha,
        "recalibration_manifest_sha256": context["manifest_sha256"],
        "preparation_sha256": frozen["historical_calibration"]["preparation_sha256"],
        "window_selection_sha256": frozen["historical_calibration"][
            "window_selection_sha256"
        ],
        "logical_trial_count": len(trials),
        "trials": trials,
        "did_run_biometrics": True,
        "did_select_thresholds": False,
        "did_read_generation3_gold_or_audio": False,
        "did_mutate_profiles_or_references": False,
        "contains_biometric_scores": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    write_immutable_private_json(paths["score_matrix"], matrix)
    matrix_sha = sha256_file(paths["score_matrix"])
    receipt = _score_receipt(
        preview=preview, authority_sha=authority_sha, matrix_sha=matrix_sha
    )
    write_immutable_private_json(paths["score_receipt"], receipt)
    return {
        **receipt,
        "private_score_matrix_path": str(paths["score_matrix"]),
        "private_score_receipt_path": str(paths["score_receipt"]),
        "idempotent_replay": False,
    }


def _replay_execution_authority(
    execution_authority_sha256: str, *, runtime_root: Path, context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Path]]:
    if not SHA256_RE.fullmatch(str(execution_authority_sha256)):
        raise Generation3RecalibrationExecutionError(
            "Execution authority hash is invalid."
        )
    execution_id = (
        f"generation3-recalibration-execution-{execution_authority_sha256[:24]}"
    )
    paths = _execution_paths(runtime_root, execution_id)
    require_private_file(paths["authority"], paths["root"])
    authority = _read_object(paths["authority"])
    if (
        _canonical_hash(authority) != execution_authority_sha256
        or authority.get("schema_version") != AUTHORITY_SCHEMA
        or authority.get("did_load_model") is not False
        or authority.get("did_read_generation3_gold_or_audio") is not False
    ):
        raise Generation3RecalibrationExecutionError(
            "Execution authority did not replay."
        )
    preview = authority.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation3RecalibrationExecutionError(
            "Execution authority preview is unavailable."
        )
    repository = _validate_repository_authority(authority.get("repository_authority"))
    expected = _preview_core(context, repository)
    content_sha = _canonical_hash(expected)
    expected_preview = {
        **expected,
        "preview_id": f"generation3-recalibration-execution-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }
    if dict(preview) != expected_preview or authority != _authority_body(expected_preview):
        raise Generation3RecalibrationExecutionError(
            "Execution authority bindings drifted."
        )
    return authority, paths


def _validate_score_matrix(
    matrix: Mapping[str, Any],
    *,
    authority_sha: str,
    context: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    frozen = context["preview"]
    profiles, _ = _expected_inventory(frozen)
    windows = {
        str(item["window_id"]): dict(item) for item in selection["windows"]
    }
    known_subjects = {str(item["person_ref_id"]) for item in profiles.values()}
    expected = {
        (window_id, method_id, profile_id)
        for window_id in windows
        for method_id in recalibration.METHOD_IDS
        for profile_id in profiles
    }
    trials = matrix.get("trials")
    if not isinstance(trials, list):
        raise Generation3RecalibrationExecutionError("Score trials are unavailable.")
    actual = set()
    probe_groups: dict[tuple[str, str], set[str]] = {}
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise Generation3RecalibrationExecutionError("Score trial is invalid.")
        window = windows.get(str(trial.get("window_id") or ""))
        profile = profiles.get(str(trial.get("profile_id") or ""))
        score = trial.get("score")
        if (
            window is None
            or profile is None
            or trial.get("status") != "success"
            or trial.get("reason_code") is not None
            or trial.get("candidate_id") != profile["candidate_id"]
            or trial.get("model_revision") != profile["model_revision"]
            or trial.get("descendant_id") != profile["descendant_id"]
            or trial.get("probe_subject_id") != window["subject_id"]
            or trial.get("profile_person_ref_id") != profile["person_ref_id"]
            or trial.get("expected_match")
            is not (window["subject_id"] == profile["person_ref_id"])
            or trial.get("open_set_probe")
            is not (window["subject_id"] not in known_subjects)
            or trial.get("method_id") not in recalibration.METHOD_IDS
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1.0 <= float(score) <= 1.0
            or trial.get("conditions") != window["conditions"]
            or trial.get("p4_state_verified_before_and_after") is not True
            or trial.get("p3_eligibility_verified_before_and_after") is not True
            or trial.get("contains_raw_biometric_values") is not False
        ):
            raise Generation3RecalibrationExecutionError(
                "Score trial binding changed."
            )
        score_identity = {
            "profile_id": profile["profile_id"],
            "descendant_id": profile["descendant_id"],
            "artifact_sha256": profile["artifact_sha256"],
            "candidate_id": profile["candidate_id"],
            "model_revision": profile["model_revision"],
            "probe_sha256": trial.get("probe_sha256"),
            "score": score,
        }
        expected_score_id = "verification-trial-" + verification.canonical_artifact_hash(
            score_identity
        )[:24]
        identity = {
            "execution_authority_sha256": authority_sha,
            "window_id": window["window_id"],
            "method_id": trial["method_id"],
            "profile_id": profile["profile_id"],
            "score_trial_id": expected_score_id,
        }
        if (
            trial.get("score_trial_id") != expected_score_id
            or trial.get("trial_id")
            != "generation3-calibration-trial-" + _canonical_hash(identity)[:24]
        ):
            raise Generation3RecalibrationExecutionError(
                "Score trial identity changed."
            )
        key = (str(window["window_id"]), str(trial["method_id"]), str(profile["profile_id"]))
        actual.add(key)
        probe_groups.setdefault(
            (str(window["window_id"]), str(trial["method_id"])), set()
        ).add(str(trial["probe_sha256"]))
    if (
        matrix.get("schema_version") != SCORE_MATRIX_SCHEMA
        or matrix.get("status") != "success"
        or matrix.get("execution_authority_sha256") != authority_sha
        or matrix.get("recalibration_manifest_sha256") != context["manifest_sha256"]
        or matrix.get("preparation_sha256")
        != frozen["historical_calibration"]["preparation_sha256"]
        or matrix.get("window_selection_sha256")
        != frozen["historical_calibration"]["window_selection_sha256"]
        or actual != expected
        or len(actual) != len(trials)
        or matrix.get("logical_trial_count") != len(trials)
        or any(len(values) != 1 for values in probe_groups.values())
        or matrix.get("did_run_biometrics") is not True
        or matrix.get("did_select_thresholds") is not False
        or matrix.get("did_read_generation3_gold_or_audio") is not False
        or matrix.get("did_mutate_profiles_or_references") is not False
        or matrix.get("contains_biometric_scores") is not True
    ):
        raise Generation3RecalibrationExecutionError(
            "Generation-3 score matrix replay is invalid."
        )
    _validate_unit_denominators(trials, frozen)


def replay_generation3_recalibration_scores(
    execution_authority_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = recalibration.DEFAULT_CORPUS_MANIFESTS,
    authority_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Structurally replay private scores without audio or model execution."""
    context = _authority_context(
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    authority, paths = _replay_execution_authority(
        execution_authority_sha256, runtime_root=runtime_root, context=context
    )
    selection, _ = _private_historical_context(
        calibration_root=calibration_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    require_private_file(paths["score_matrix"], paths["root"])
    matrix = _read_object(paths["score_matrix"])
    _validate_score_matrix(
        matrix,
        authority_sha=execution_authority_sha256,
        context=context,
        selection=selection,
    )
    matrix_sha = sha256_file(paths["score_matrix"])
    require_private_file(paths["score_receipt"], paths["root"])
    receipt = _read_object(paths["score_receipt"])
    expected = _score_receipt(
        preview=authority["preview"],
        authority_sha=execution_authority_sha256,
        matrix_sha=matrix_sha,
    )
    if receipt != expected:
        raise Generation3RecalibrationExecutionError("Score receipt drifted.")
    return {
        **receipt,
        "private_score_matrix_path": str(paths["score_matrix"]),
        "private_score_receipt_path": str(paths["score_receipt"]),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
        "score_replay_mode": "structural_without_audio_or_model_execution",
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return False


def _threshold_results(
    frozen: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    policy = {
        **dict(frozen["historical_calibration"]["threshold_policy"]),
        "condition_slices": frozen["historical_calibration"]["metric_policy"][
            "condition_slices"
        ],
    }
    results = []
    for candidate_id in recalibration.CANDIDATE_IDS:
        for method_id in recalibration.METHOD_IDS:
            trials = [
                item
                for item in matrix["trials"]
                if item["candidate_id"] == candidate_id
                and item["method_id"] == method_id
            ]
            try:
                result = verification._freeze_threshold_unit(
                    candidate_id, method_id, trials, policy
                )
            except verification.AcousticVerificationError as exc:
                raise Generation3RecalibrationExecutionError(
                    "Threshold selection failed closed."
                ) from exc
            if (
                result.get("status") != "success"
                or result.get("reason_code") is not None
                or result.get("metrics", {}).get("missing_denominator_status")
                != "success"
                or result.get("metrics", {}).get("trial_count")
                != frozen["expected_trials_per_unit"]
                or result.get("metrics", {}).get("genuine_trial_count")
                != frozen["expected_genuine_trials_per_unit"]
                or result.get("metrics", {}).get("impostor_trial_count")
                != frozen["expected_impostor_trials_per_unit"]
                or result.get("open_set_rejection", {}).get("probe_count")
                != frozen["expected_open_set_trials_per_unit"] // 2
                or result.get("candidate_margin", {}).get("count")
                != frozen["historical_calibration"]["window_count"]
                or not _finite_tree(result)
            ):
                raise Generation3RecalibrationExecutionError(
                    "Threshold unit is incomplete, nonfinite, or fallback."
                )
            results.append(result)
    if len(results) != 9:
        raise Generation3RecalibrationExecutionError(
            "Exactly nine threshold units are required."
        )
    return results


def _threshold_receipt(
    *, score_receipt: Mapping[str, Any], application_sha: str, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    actions = dict(score_receipt["action_vector"])
    actions["build_pre_reveal_envelope"] = True
    return {
        "schema_version": THRESHOLD_RECEIPT_SCHEMA,
        "status": "nine_thresholds_frozen_pre_reveal_envelope_authorized",
        "execution_authority_sha256": score_receipt["execution_authority_sha256"],
        "score_matrix_sha256": score_receipt["score_matrix_sha256"],
        "threshold_application_sha256": application_sha,
        "threshold_set_sha256": _canonical_hash(list(results)),
        "threshold_unit_count": 9,
        "abstention_margin_is_zero": True,
        "action_vector": actions,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_frozen_threshold_values": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "mode": "0600",
    }


def freeze_generation3_recalibration_thresholds(
    execution_authority_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = recalibration.DEFAULT_CORPUS_MANIFESTS,
    authority_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze nine deterministic pairs from persisted scores only."""
    score_receipt = replay_generation3_recalibration_scores(
        execution_authority_sha256,
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    context = _authority_context(
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    _, paths = _replay_execution_authority(
        execution_authority_sha256, runtime_root=runtime_root, context=context
    )
    matrix = _read_object(paths["score_matrix"])
    results = _threshold_results(context["preview"], matrix)
    application = {
        "schema_version": THRESHOLD_APPLICATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "execution_authority_sha256": execution_authority_sha256,
        "score_matrix_sha256": score_receipt["score_matrix_sha256"],
        "threshold_unit_count": 9,
        "thresholds": results,
        "selection_objective": context["preview"]["historical_calibration"][
            "selection_objective"
        ],
        "abstention_margin": 0.0,
        "did_recompute_from_persisted_scores": True,
        "did_read_generation3_gold_or_audio": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "contains_biometric_scores": True,
        "contains_frozen_thresholds": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if paths["threshold_application"].exists():
        require_private_file(paths["threshold_application"], paths["root"])
        if _read_object(paths["threshold_application"]) != application:
            raise Generation3RecalibrationExecutionError(
                "Threshold application conflicts."
            )
    else:
        write_immutable_private_json(paths["threshold_application"], application)
    application_sha = sha256_file(paths["threshold_application"])
    receipt = _threshold_receipt(
        score_receipt=score_receipt,
        application_sha=application_sha,
        results=results,
    )
    if paths["threshold_receipt"].exists():
        require_private_file(paths["threshold_receipt"], paths["root"])
        if _read_object(paths["threshold_receipt"]) != receipt:
            raise Generation3RecalibrationExecutionError("Threshold receipt conflicts.")
    else:
        write_immutable_private_json(paths["threshold_receipt"], receipt)
    return {
        **receipt,
        "private_threshold_application_path": str(paths["threshold_application"]),
        "private_threshold_receipt_path": str(paths["threshold_receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_recalibration_thresholds(
    execution_authority_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = recalibration.DEFAULT_CORPUS_MANIFESTS,
    authority_manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Recompute and replay all threshold results without audio or models."""
    score_receipt = replay_generation3_recalibration_scores(
        execution_authority_sha256,
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    context = _authority_context(
        runtime_root=runtime_root,
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
        authority_manifest_path=authority_manifest_path,
    )
    _, paths = _replay_execution_authority(
        execution_authority_sha256, runtime_root=runtime_root, context=context
    )
    require_private_file(paths["threshold_application"], paths["root"])
    application = _read_object(paths["threshold_application"])
    results = _threshold_results(context["preview"], _read_object(paths["score_matrix"]))
    expected_application = {
        "schema_version": THRESHOLD_APPLICATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "execution_authority_sha256": execution_authority_sha256,
        "score_matrix_sha256": score_receipt["score_matrix_sha256"],
        "threshold_unit_count": 9,
        "thresholds": results,
        "selection_objective": context["preview"]["historical_calibration"][
            "selection_objective"
        ],
        "abstention_margin": 0.0,
        "did_recompute_from_persisted_scores": True,
        "did_read_generation3_gold_or_audio": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "contains_biometric_scores": True,
        "contains_frozen_thresholds": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if application != expected_application:
        raise Generation3RecalibrationExecutionError("Threshold application drifted.")
    application_sha = sha256_file(paths["threshold_application"])
    require_private_file(paths["threshold_receipt"], paths["root"])
    receipt = _read_object(paths["threshold_receipt"])
    expected_receipt = _threshold_receipt(
        score_receipt=score_receipt,
        application_sha=application_sha,
        results=results,
    )
    if receipt != expected_receipt:
        raise Generation3RecalibrationExecutionError("Threshold receipt drifted.")
    return {
        **receipt,
        "private_threshold_application_path": str(paths["threshold_application"]),
        "private_threshold_receipt_path": str(paths["threshold_receipt"]),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
        "threshold_replay_mode": "recomputed_from_persisted_scores_without_audio",
    }
