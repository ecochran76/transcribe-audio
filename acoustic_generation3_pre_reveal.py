"""Generation-3 independently reviewed pre-reveal envelope."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_audio_derivatives as derivatives
import acoustic_generation3_recalibration_execution as execution
import acoustic_speech_preparation as preparation
import acoustic_successor_conditions as conditions
import acoustic_verification as verification
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-pre-reveal-preview.v1"
PORTABLE_SCHEMA = "transcribe-audio.generation3-pre-reveal-portable.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation3-pre-reveal-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-pre-reveal-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-pre-reveal-replay.v1"
DEFAULT_RUNTIME_ROOT = execution.DEFAULT_RUNTIME_ROOT
DEFAULT_TERMINAL_POLICY = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/generation-3-terminal-decision-policy.json"
)
MODULE_NAMES = (
    "acoustic_generation3_pre_reveal.py",
    "acoustic_generation3_recalibration_execution.py",
    "acoustic_generation3_recalibration.py",
    "acoustic_generation3_gold.py",
    "acoustic_generation3_authority.py",
    "acoustic_verification.py",
    "acoustic_audio_derivatives.py",
    "acoustic_speech_preparation.py",
    "acoustic_successor_conditions.py",
)
CONDITION_DIMENSIONS = tuple(conditions.CONDITION_FIELDS)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation3PreRevealError(ValueError):
    """Raised when the Generation-3 envelope cannot stay sealed and exact."""


def _canonical_hash(value: Any) -> str:
    return execution._canonical_hash(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3PreRevealError("Pre-reveal JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3PreRevealError("Pre-reveal JSON must be an object.")
    return value


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise Generation3PreRevealError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    behind, ahead = (
        int(item)
        for item in _git(
            ["rev-list", "--left-right", "--count", "@{upstream}...HEAD"]
        ).split()
    )
    root = Path(__file__).resolve().parent
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": {
            name: sha256_file(root / name) for name in MODULE_NAMES
        },
        "terminal_policy_sha256": sha256_file(DEFAULT_TERMINAL_POLICY),
        "clean": _git(["status", "--porcelain"]) == "",
        "upstream_ahead": ahead,
        "upstream_behind": behind,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Generation3PreRevealError("Repository authority is invalid.")
    commit = str(value.get("commit") or "")
    modules = value.get("module_sha256")
    if (
        set(value)
        != {
            "commit", "module_sha256", "terminal_policy_sha256", "clean",
            "upstream_ahead", "upstream_behind",
        }
        or not COMMIT_RE.fullmatch(commit)
        or not isinstance(modules, Mapping)
        or set(modules) != set(MODULE_NAMES)
        or any(not SHA256_RE.fullmatch(str(item)) for item in modules.values())
        or not SHA256_RE.fullmatch(str(value.get("terminal_policy_sha256") or ""))
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3PreRevealError("Repository authority drifted.")
    root = Path(__file__).resolve().parent
    for name, digest in modules.items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{name}"], cwd=root,
            check=False, capture_output=True,
        )
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != digest
            or sha256_file(root / name) != digest
        ):
            raise Generation3PreRevealError("Repository module authority drifted.")
    current = _repository_authority()
    if (
        current["clean"] is not True
        or current["upstream_ahead"] != 0
        or current["upstream_behind"] != 0
        or sha256_file(DEFAULT_TERMINAL_POLICY) != value["terminal_policy_sha256"]
    ):
        raise Generation3PreRevealError("Current repository is not clean and exact.")
    return dict(value)


def _single(root: Path, pattern: str, label: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise Generation3PreRevealError(f"Exactly one {label} is required.")
    require_private_file(paths[0], root)
    return paths[0]


def _terminal_policy() -> tuple[dict[str, Any], str]:
    path = DEFAULT_TERMINAL_POLICY.resolve(strict=True)
    policy = _read_object(path)
    expected_minimum = {
        "genuine_trials_per_model_method_unit": 20,
        "impostor_trials_per_model_method_unit": 100,
        "open_set_trials_per_model_method_unit": 20,
        "evaluation_recordings": 7,
        "evaluation_conversations": 7,
        "known_subjects": 5,
        "enrolled_subjects": 2,
        "enrolled_subject_conversations_each": 2,
        "independent_same_person_subject_session_pairs": 4,
        "all_declared_condition_slices_reported": True,
        "minimum_observed_values_per_condition": 2,
        "missing_condition_recordings": 0,
    }
    if (
        policy.get("schema_version")
        != "transcribe-audio.verification-generation-3-terminal-decision-policy.v1"
        or policy.get("precedence") != ["stop", "reject", "select", "refine"]
        or policy.get("minimum_evidence") != expected_minimum
        or policy.get("policy_changes_after_evaluation_unseal")
        != "forbidden_for_this_evaluation_generation"
        or policy.get("exact_trial_child_required_before_model_or_score_execution")
        is not True
        or policy.get("exact_trial_child_may_change_parent_policy") is not False
        or policy.get("condition_failure")
        != "global_stop_before_exact_trial_or_model_execution"
        or policy.get("incomplete_cartesian_or_failed_or_blocked_cell")
        != "global_stop"
        or policy.get("nonfinite_score_or_required_metric") != "global_stop"
        or policy.get("default_integration_authorized") is not False
        or policy.get("historical_reprocessing_authorized") is not False
    ):
        raise Generation3PreRevealError("Terminal decision policy is invalid.")
    return policy, sha256_file(path)


def _population_authority(gold_preview: Mapping[str, Any]) -> dict[str, Any]:
    counts = gold_preview.get("enrolled_conversation_counts")
    if not isinstance(counts, Mapping):
        raise Generation3PreRevealError("Enrolled population authority is missing.")
    values = sorted(int(item) for item in counts.values())
    pair_count = sum(value * (value - 1) // 2 for value in values)
    if (
        len(values) != 2
        or min(values) < 2
        or pair_count < 4
        or gold_preview.get("known_subject_count", 0) < 5
        or gold_preview.get("gold_label_count") != 28
    ):
        raise Generation3PreRevealError("Pre-reveal population gate failed.")
    return {
        "enrolled_subject_count": len(values),
        "enrolled_conversation_counts_sha256": _canonical_hash(values),
        "minimum_enrolled_conversations_per_subject": min(values),
        "independent_same_person_subject_session_pair_count": pair_count,
        "known_subject_count": gold_preview["known_subject_count"],
        "gold_label_count": gold_preview["gold_label_count"],
        "pair_formula": "sum_n_choose_2_over_enrolled_subject_conversation_counts",
        "gate_status": "pass",
    }


def _frozen_context(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    execution_authority_path = _single(
        root, "recalibration-executions/*/execution-authority.json",
        "recalibration execution authority",
    )
    execution_authority = _read_object(execution_authority_path)
    execution_sha = _canonical_hash(execution_authority)
    threshold = execution.replay_generation3_recalibration_thresholds(
        execution_sha, runtime_root=root
    )
    threshold_path = Path(threshold["private_threshold_application_path"])
    require_private_file(threshold_path, root)
    threshold_application = _read_object(threshold_path)
    recalibration_path = _single(
        root, "recalibration-authorities/*/private-manifest.json",
        "recalibration authority",
    )
    recalibration_manifest = _read_object(recalibration_path)
    recalibration_preview = recalibration_manifest.get("preview")
    cohort_path = _single(
        root, "cohort-authorities/*/private-manifest.json", "cohort authority"
    )
    gold_path = _single(
        root, "gold-authorities/*/private-manifest.json", "gold authority"
    )
    cohort_manifest = _read_object(cohort_path)
    gold_manifest = _read_object(gold_path)
    gold_preview = gold_manifest.get("preview")
    if (
        not isinstance(recalibration_preview, Mapping)
        or not isinstance(gold_preview, Mapping)
        or threshold.get("idempotent_replay") is not True
        or threshold.get("threshold_unit_count") != 9
        or threshold.get("abstention_margin_is_zero") is not True
        or threshold.get("action_vector", {}).get("build_pre_reveal_envelope")
        is not True
        or threshold.get("action_vector", {}).get("reveal_evaluation") is not False
        or threshold_application.get("threshold_unit_count") != 9
        or threshold_application.get("abstention_margin") != 0.0
        or threshold_application.get("did_read_generation3_gold_or_audio") is not False
        or recalibration_preview.get("generation3_cohort_authority", {}).get(
            "manifest_sha256"
        )
        != sha256_file(cohort_path)
        or recalibration_preview.get("generation3_gold_commitment", {}).get(
            "manifest_sha256"
        )
        != sha256_file(gold_path)
        or gold_preview.get("membership_sha256")
        != cohort_manifest.get("preview", {}).get("membership_sha256")
    ):
        raise Generation3PreRevealError("Frozen Generation-3 authorities drifted.")
    return {
        "root": str(root),
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "cohort": dict(cohort_manifest["preview"]),
        "gold_manifest_sha256": sha256_file(gold_path),
        "gold": dict(gold_preview),
        "recalibration_manifest_sha256": sha256_file(recalibration_path),
        "recalibration": dict(recalibration_preview),
        "execution_authority_sha256": execution_sha,
        "score_matrix_sha256": threshold["score_matrix_sha256"],
        "threshold_application_sha256": threshold[
            "threshold_application_sha256"
        ],
        "threshold_set_sha256": threshold["threshold_set_sha256"],
        "thresholds": list(threshold_application["thresholds"]),
    }


def _candidate_matrix(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = context["recalibration"]["profiles"]
    profile_by_candidate: dict[str, list[str]] = {}
    for profile in profiles:
        profile_by_candidate.setdefault(str(profile["candidate_id"]), []).append(
            str(profile["profile_id"])
        )
    results = []
    expected_units = {
        (candidate_id, method_id)
        for candidate_id in execution.recalibration.CANDIDATE_IDS
        for method_id in execution.recalibration.METHOD_IDS
    }
    for item in context["thresholds"]:
        threshold = item.get("threshold")
        temperature = item.get("temperature")
        if (
            item.get("status") != "success"
            or item.get("reason_code") is not None
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise Generation3PreRevealError("A threshold unit is not successful.")
        candidate_id = str(item["candidate_id"])
        results.append(
            {
                "candidate_id": candidate_id,
                "method_id": item["method_id"],
                "profile_ids": sorted(profile_by_candidate.get(candidate_id, [])),
                "threshold": item["threshold"],
                "temperature": item["temperature"],
                "abstention_margin": 0.0,
            }
        )
    observed_units = {
        (str(item["candidate_id"]), str(item["method_id"])) for item in results
    }
    if (
        len(results) != 9
        or observed_units != expected_units
        or len(observed_units) != len(results)
        or any(len(item["profile_ids"]) != 2 for item in results)
    ):
        raise Generation3PreRevealError("Candidate matrix is incomplete.")
    return sorted(results, key=lambda item: (item["candidate_id"], item["method_id"]))


def _evaluate(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    context = _frozen_context(runtime_root=runtime_root)
    policy, policy_sha = _terminal_policy()
    population = _population_authority(context["gold"])
    matrix = _candidate_matrix(context)
    cohort = context["cohort"]
    recalibration = context["recalibration"]
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "reason_code": None,
        "authority_generation": 3,
        "intended_split": "evaluation",
        "cohort_authority": {
            "manifest_sha256": context["cohort_manifest_sha256"],
            "membership_sha256": cohort["membership_sha256"],
            "conversation_count": cohort["membership"]["conversation_count"],
            "speaker_label_count": cohort["membership"]["speaker_label_count"],
        },
        "gold_authority": {
            "manifest_sha256": context["gold_manifest_sha256"],
            "receipt_sha256": recalibration["generation3_gold_commitment"][
                "receipt_sha256"
            ],
            "gold_body_sha256": recalibration["generation3_gold_commitment"][
                "gold_body_sha256"
            ],
            "membership_sha256": context["gold"]["membership_sha256"],
            "gold_label_count": context["gold"]["gold_label_count"],
            "outcome_counts": dict(context["gold"]["outcome_counts"]),
        },
        "population_authority": population,
        "recalibration_authority": {
            "manifest_sha256": context["recalibration_manifest_sha256"],
            "execution_authority_sha256": context["execution_authority_sha256"],
            "score_matrix_sha256": context["score_matrix_sha256"],
            "threshold_application_sha256": context[
                "threshold_application_sha256"
            ],
            "threshold_set_sha256": context["threshold_set_sha256"],
            "calibration_membership_sha256": _canonical_hash(
                recalibration["historical_calibration"]["calibration_dimensions"]
            ),
            "profile_set_sha256": recalibration["active_profile_authority"][
                "profile_set_sha256"
            ],
            "model_asset_set_sha256": recalibration["active_profile_authority"][
                "model_asset_set_sha256"
            ],
            "model_assets": dict(
                recalibration["active_profile_authority"]["model_assets"]
            ),
            "unit_count": 9,
        },
        "profiles": list(recalibration["profiles"]),
        "candidate_matrix": matrix,
        "window_policy": dict(cohort["window_policy"]),
        "preparation_contract": {
            "p1_module_sha256": sha256_file(Path(derivatives.__file__).resolve()),
            "p2_module_sha256": sha256_file(Path(preparation.__file__).resolve()),
            "condition_module_sha256": sha256_file(Path(conditions.__file__).resolve()),
            "preparation_methods": list(conditions.METHOD_IDS),
            "score_methods": list(execution.recalibration.METHOD_IDS),
            "no_fallback_method": True,
        },
        "condition_policy": {
            "dimensions": list(CONDITION_DIMENSIONS),
            "implementation_functions": ["_conditions", "_aggregate_conditions"],
            "measurement_algorithms": {
                "channel": "source_probe_channels_equals_2_else_mono",
                "device": "first_explicit_source_or_probe_device_id_or_model_else_unavailable",
                "noise": "silero_merged_speech_region_energy_snr_db_frozen_bands",
                "telephone_bandwidth": "source_sample_rate_at_most_16000_hz",
                "usable_duration_band": "sum_merged_silero_speech_regions_frozen_bands",
            },
            "measurement_module_sha256": sha256_file(
                Path(conditions.__file__).resolve()
            ),
            "minimum_observed_values_per_dimension": 2,
            "missing_recordings_allowed": 0,
            "measurement_after_prediction_blind_p1_p2": True,
            "measurement_before_window_freeze_and_exact_trials": True,
            "gold_may_not_change_measurement": True,
        },
        "trial_construction_policy": {
            "same_frozen_window_set_for_every_candidate_unit": True,
            "trial_score": "raw_cosine_against_fixed_successor_centroid",
            "same_person_class": "frozen_gold_subject_matches_profile_person_ref",
            "different_person_class": "frozen_gold_subject_differs_from_profile_person_ref",
            "open_set_class": "frozen_gold_subject_has_no_frozen_profile",
            "mixed_or_unknown_gold": "excluded_before_scoring",
            "no_model_output_may_change_membership": True,
        },
        "exact_trial_child_policy": {
            "required_before_model_or_score_execution": True,
            "must_bind_parent_content_sha256": True,
            "must_bind_window_manifest_and_every_trial_id": True,
            "must_cover_every_candidate_matrix_unit": True,
            "must_freeze_per_unit_class_denominators": {
                "genuine": 20, "impostor": 100, "open_set": 20,
            },
            "may_change_parent_policy_threshold_margin_or_candidate": False,
            "missing_or_incomplete_child_action": "global_stop_before_model_execution",
        },
        "score_aggregation_policy": {
            "threshold_input": "raw_cosine_score",
            "temperature_input": "frozen_successor_temperature",
            "profile_aggregation": "fixed_successor_centroid_only",
            "same_timestamp_bounds_across_score_methods": True,
            "ties_abstain_before_tie_break": True,
            "no_normalization_change": True,
        },
        "evaluation_metric_policy": {
            "trial_metrics": dict(
                recalibration["historical_calibration"]["metric_policy"]
            ),
            "thresholds_temperatures_and_zero_margin_are_frozen": True,
            "attempt_accounting": "attempted_success_failed_blocked_reported_separately",
            "all_declared_condition_slices_reported": True,
            "conversation_clustered_non_independent": True,
        },
        "terminal_decision_policy_sha256": policy_sha,
        "terminal_decision_policy": policy,
        "terminal_resolution_policy": {
            "unit_precedence": ["stop", "reject", "select", "refine"],
            "global_integrity_or_minimum_evidence_failure": "stop",
            "any_unit_stop": "global_stop_before_candidate_reduction",
            "evaluation_may_not_change_policy_threshold_margin_or_candidate": True,
        },
        "action_vector": {
            "build_pre_reveal_envelope": False,
            "reveal_evaluation": False,
            "run_denominator_preflight": False,
            "prepare_evaluation_audio": False,
            "measure_conditions": False,
            "freeze_evaluation_windows": False,
            "construct_exact_trial_child": False,
            "load_or_run_models": False,
            "score_evaluation_trials": False,
            "calculate_evaluation_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "production_apply_authorized": False,
        "requires_independent_review": True,
        "requires_clean_pushed_commit": True,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
        "will_perform_external_write": False,
    }
    if verification._contains_forbidden_private_key(core):
        raise Generation3PreRevealError("Envelope contains forbidden private data.")
    content_sha = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-pre-reveal-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }


def preview_generation3_pre_reveal(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    """Build the complete envelope without reveal, audio, or model execution."""
    return _evaluate(runtime_root=runtime_root)


def portable_pre_reveal_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PORTABLE_SCHEMA,
        "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "cohort_manifest_sha256": preview["cohort_authority"]["manifest_sha256"],
        "gold_manifest_sha256": preview["gold_authority"]["manifest_sha256"],
        "membership_sha256": preview["cohort_authority"]["membership_sha256"],
        "score_matrix_sha256": preview["recalibration_authority"][
            "score_matrix_sha256"
        ],
        "threshold_application_sha256": preview["recalibration_authority"][
            "threshold_application_sha256"
        ],
        "threshold_set_sha256": preview["recalibration_authority"][
            "threshold_set_sha256"
        ],
        "terminal_decision_policy_sha256": preview[
            "terminal_decision_policy_sha256"
        ],
        "conversation_count": preview["cohort_authority"]["conversation_count"],
        "gold_label_count": preview["gold_authority"]["gold_label_count"],
        "known_subject_count": preview["population_authority"]["known_subject_count"],
        "independent_same_person_subject_session_pair_count": preview[
            "population_authority"
        ]["independent_same_person_subject_session_pair_count"],
        "profile_count": len(preview["profiles"]),
        "candidate_unit_count": len(preview["candidate_matrix"]),
        "condition_dimension_count": len(preview["condition_policy"]["dimensions"]),
        "abstention_margin_is_zero": all(
            item["abstention_margin"] == 0.0
            for item in preview["candidate_matrix"]
        ),
        "action_vector": dict(preview["action_vector"]),
        "contains_profile_or_subject_ids": False,
        "contains_threshold_or_temperature_values": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False,
    }


def _paths(runtime_root: Path, authority_id: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    directory = root / "pre-reveal-authorities" / authority_id
    return {
        "root": root, "directory": directory,
        "manifest": directory / "private-manifest.json",
        "receipt": directory / "receipt.json",
    }


def _existing_manifest(runtime_root: Path) -> Path | None:
    paths = sorted(
        runtime_root.expanduser().absolute().glob(
            "pre-reveal-authorities/*/private-manifest.json"
        )
    )
    if len(paths) > 1:
        raise Generation3PreRevealError("Multiple pre-reveal authorities exist.")
    return paths[0] if paths else None


def _manifest_core(
    preview: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "preview": dict(preview),
        "portable_projection": portable_pre_reveal_projection(preview),
        "repository_authority": dict(repository),
        "exact_trial_child_required_before_model_or_score_execution": True,
    }


def _receipt(
    preview: Mapping[str, Any], authority_id: str, manifest_sha: str
) -> dict[str, Any]:
    portable = portable_pre_reveal_projection(preview)
    actions = dict(portable["action_vector"])
    actions["build_pre_reveal_envelope"] = True
    actions["reveal_evaluation"] = True
    return {
        **portable,
        "schema_version": RECEIPT_SCHEMA,
        "status": "pre_reveal_envelope_frozen_reveal_authorized",
        "authority_id": authority_id,
        "manifest_sha256": manifest_sha,
        "action_vector": actions,
        "mode": "0600",
    }


def apply_generation3_pre_reveal(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Freeze an independently reviewed envelope; do not reveal evaluation."""
    preview = _evaluate(runtime_root=runtime_root)
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
        or preview["status"] != "ready_for_independent_review"
        or any(preview["action_vector"].values())
    ):
        raise Generation3PreRevealError("Reviewed pre-reveal preview is stale.")
    existing = _existing_manifest(runtime_root)
    if existing is not None:
        return replay_generation3_pre_reveal(existing, runtime_root=runtime_root)
    repository = _repository_authority()
    if (
        repository["clean"] is not True
        or repository["upstream_ahead"] != 0
        or repository["upstream_behind"] != 0
    ):
        raise Generation3PreRevealError(
            "Pre-reveal apply requires a clean upstream-even repository."
        )
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    authority_id = f"generation3-pre-reveal-{content_sha[:24]}"
    paths = _paths(runtime_root, authority_id)
    ensure_private_tree(paths["root"], paths["directory"])
    manifest = {**core, "authority_id": authority_id, "content_sha256": content_sha}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, authority_id, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_pre_reveal(
    manifest_path: Path, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay the full envelope without reveal, audio, models, or scores."""
    root = runtime_root.expanduser().absolute()
    path = manifest_path.expanduser().absolute()
    require_private_file(path, root)
    manifest = _read_object(path)
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    preview = _evaluate(runtime_root=root)
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    authority_id = f"generation3-pre-reveal-{content_sha[:24]}"
    expected = {**core, "authority_id": authority_id, "content_sha256": content_sha}
    if manifest != expected or path != _paths(root, authority_id)["manifest"]:
        raise Generation3PreRevealError("Pre-reveal manifest drifted.")
    receipt_path = _paths(root, authority_id)["receipt"]
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = _receipt(preview, authority_id, sha256_file(path))
    if receipt != expected_receipt:
        raise Generation3PreRevealError("Pre-reveal receipt drifted.")
    return {
        **receipt,
        "private_manifest_path": str(path),
        "private_receipt_path": str(receipt_path),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
        "full_body_match": True,
    }
