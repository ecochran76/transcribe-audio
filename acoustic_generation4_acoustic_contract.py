"""Plan 0052 G1B calibration-only acoustic evidence contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


CONTRACT_SCHEMA = "transcribe-audio.generation4-acoustic-contract.v1"
FACTOR_CONTRACT_SCHEMA = "transcribe-audio.generation4-acoustic-factor.v1"
EVIDENCE_SCHEMA = "transcribe-audio.generation4-acoustic-evidence.v1"
SELECTION_POLICY_SCHEMA = (
    "transcribe-audio.generation4-calibration-factor-selection.v1"
)
DENOMINATOR_SCHEMA = "transcribe-audio.generation4-denominator-proof.v1"
EXACT_TRIAL_SCHEMA = "transcribe-audio.generation4-exact-trial.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-exact-trial-replay.v1"
CONTRACT_REPLAY_SCHEMA = "transcribe-audio.generation4-acoustic-contract-replay.v1"

G0_PREVIEW_SHA256 = (
    "aa179741e735247e87cc6143c6526669670734c8c562ed166160eb0c6d605010"
)
G0_MANIFEST_SHA256 = (
    "ad9e26b59502508c8810e11648d519d99860579aea1ca731445459b196836d22"
)
CANDIDATE_IDS = (
    "speechbrain_ecapa_tdnn",
    "wespeaker_campplus",
    "wespeaker_resnet34",
)
METHOD_IDS = ("no_enhancement", "deepfilternet", "rnnoise")
CONDITION_DIMENSIONS = (
    "channel",
    "device",
    "noise",
    "telephone_bandwidth",
    "usable_duration_band",
)
CALIBRATION_SELECTION_OBJECTIVE = (
    "minimum_brier_score",
    "minimum_expected_calibration_error_5_equal_width_bins",
    "minimum_balanced_error_rate",
    "minimum_absolute_far_minus_frr",
    "highest_threshold",
    "lowest_temperature",
)
PROHIBITED_ACTIONS = (
    "read_generation4_gold",
    "read_generation4_holdout",
    "load_or_run_models",
    "run_g2_policy_freeze",
    "run_g3_blind_baseline",
    "run_g4_augmented_predictions",
    "reveal_gold",
    "run_g5_scoring",
    "mutate_profiles_or_references",
    "enable_default_integration",
    "run_historical_reprocessing",
)

_SHA256_RE = re.compile(r"[a-f0-9]{64}")
_CALIBRATION_COUNTS = (44, 9, 35, 13, 22)
_G2_DENOMINATOR_MINIMA = {
    "genuine_trials_per_unit": 20,
    "known_impostor_trials_per_unit": 100,
    "open_set_trials_per_unit": 20,
}


class Generation4AcousticContractError(ValueError):
    """Raised when a G1B contract cannot remain calibration-only and exact."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Generation4AcousticContractError("Calibration ranking value is invalid.")
    result = float(value)
    if not math.isfinite(result):
        raise Generation4AcousticContractError("Calibration ranking value is nonfinite.")
    return result


def _reject_generation4_payloads(value: Any) -> None:
    allowed_guards = {
        "did_read_generation4_gold",
        "did_read_generation4_holdout",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if (
                key not in allowed_guards
                and any(token in key for token in ("gold", "holdout"))
            ):
                raise Generation4AcousticContractError(
                    "Generation-4 gold or holdout payload is forbidden."
                )
            _reject_generation4_payloads(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_generation4_payloads(child)


def _validated_units(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = packet.get("thresholds")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Generation4AcousticContractError("Calibration threshold units are missing.")
    expected = {(candidate, method) for candidate in CANDIDATE_IDS for method in METHOD_IDS}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise Generation4AcousticContractError("Calibration threshold unit is invalid.")
        unit = dict(raw)
        key = (str(unit.get("candidate_id") or ""), str(unit.get("method_id") or ""))
        metrics = unit.get("metrics")
        rejection = unit.get("open_set_rejection")
        margin = unit.get("candidate_margin")
        if (
            key not in expected
            or key in observed
            or unit.get("status") != "success"
            or unit.get("reason_code") is not None
            or not isinstance(metrics, Mapping)
            or not isinstance(rejection, Mapping)
            or not isinstance(margin, Mapping)
            or (
                metrics.get("trial_count"),
                metrics.get("genuine_trial_count"),
                metrics.get("impostor_trial_count"),
                rejection.get("probe_count"),
                margin.get("count"),
            )
            != _CALIBRATION_COUNTS
            or metrics.get("missing_denominator_status") != "success"
        ):
            raise Generation4AcousticContractError(
                "Calibration threshold inventory or denominators are incomplete."
            )
        observed[key] = unit
    if set(observed) != expected or len(observed) != 9:
        raise Generation4AcousticContractError(
            "Exactly nine frozen calibration threshold units are required."
        )
    return [observed[key] for key in sorted(observed)]


def _factor_contract_hash(unit: Mapping[str, Any], threshold_set_sha256: str) -> str:
    return _canonical_hash(
        {
            "schema_version": FACTOR_CONTRACT_SCHEMA,
            "candidate_id": unit["candidate_id"],
            "method_id": unit["method_id"],
            "threshold_set_sha256": threshold_set_sha256,
        }
    )


def _selection_rank(
    unit: Mapping[str, Any], threshold_set_sha256: str
) -> tuple[Any, ...]:
    metrics = unit["metrics"]
    rejection = unit["open_set_rejection"]
    margin = unit["candidate_margin"]
    factor_hash = _factor_contract_hash(unit, threshold_set_sha256)
    return (
        _finite_number(metrics["brier_score"]),
        _finite_number(metrics["expected_calibration_error_5_bins"]),
        _finite_number(metrics["balanced_error_rate"]),
        abs(
            _finite_number(metrics["false_acceptance_rate"])
            - _finite_number(metrics["false_rejection_rate"])
        ),
        -_finite_number(unit["threshold"]),
        _finite_number(unit["temperature"]),
        -_finite_number(rejection["rejection_rate"]),
        -_finite_number(margin["minimum"]),
        factor_hash,
    )


def _condition_taxonomy() -> dict[str, Any]:
    return {
        "schema_version": "transcribe-audio.generation4-condition-taxonomy.v1",
        "required_dimension_count": 5,
        "dimensions": {
            "channel": {"allowed_values": ["source_mono", "source_stereo"]},
            "device": {
                "representation": "observed_value_sha256",
                "unavailable_values_forbidden": True,
            },
            "noise": {
                "allowed_values": ["high_noise", "moderate_noise", "low_noise"]
            },
            "telephone_bandwidth": {
                "allowed_values": [
                    "telephone_bandwidth_candidate",
                    "not_telephone_band_limited_by_source_rate",
                ]
            },
            "usable_duration_band": {
                "allowed_values": [
                    "under_5_minutes",
                    "5_to_under_15_minutes",
                    "15_minutes_or_more",
                ]
            },
        },
        "minimum_observed_values_per_dimension": 2,
        "maximum_missing_assignments_per_dimension": 0,
    }


def _denominator_contract() -> dict[str, Any]:
    return {
        "schema_version": DENOMINATOR_SCHEMA,
        "full_matrix_unit_count": 9,
        "minimum_genuine_trials_per_unit": 20,
        "minimum_known_impostor_trials_per_unit": 100,
        "minimum_open_set_trials_per_unit": 20,
        "minimum_trial_count_per_unit": 140,
        "minimum_total_trial_count": 1260,
        "categories_are_disjoint": True,
        "g2_must_derive_and_freeze_exact_counts_from_cohort": True,
        "partial_failures_remain_in_denominator": True,
        "proof_requires_exact_child_set_replay": True,
    }


def validate_g2_exact_denominator_counts(
    exact_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cohort-derived G2 exact counts against the G1B minima."""
    counts: dict[str, int] = {}
    for field, minimum in _G2_DENOMINATOR_MINIMA.items():
        value = exact_counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise Generation4AcousticContractError(
                "G2 exact denominator is below the frozen G1B minimum."
            )
        counts[field] = value
    if set(exact_counts) != set(_G2_DENOMINATOR_MINIMA):
        raise Generation4AcousticContractError(
            "G2 exact denominator fields do not match the frozen contract."
        )
    per_unit = sum(counts.values())
    return {
        "schema_version": "transcribe-audio.generation4-g2-exact-denominators.v1",
        "status": "g2_exact_denominators_meet_g1b_minima",
        **counts,
        "exact_trial_count_per_unit": per_unit,
        "full_matrix_unit_count": 9,
        "exact_total_trial_count": per_unit * 9,
        "categories_are_disjoint": True,
        "partial_failures_remain_in_denominator": True,
        "exact_counts_frozen_by_g2": True,
    }


def _exact_trial_contract() -> dict[str, Any]:
    return {
        "schema_version": REPLAY_SCHEMA,
        "child_schema_version": EXACT_TRIAL_SCHEMA,
        "run_count": 1,
        "children_frozen_before_model_load": True,
        "children_bound_by_sha256": [
            "factor_contract_sha256",
            "probe_contract_sha256",
            "reference_contract_sha256",
            "trial_category_sha256",
            "condition_assignment_sha256",
            "policy_envelope_sha256",
        ],
        "replay_requires_identical_child_set_sha256": True,
        "replay_requires_zero_missing_children": True,
        "replay_requires_zero_duplicate_children": True,
        "replay_requires_zero_extra_children": True,
        "gold_reveal_may_label_but_not_replace_children": True,
        "reference_repair_attempts_per_model_phase": 1,
        "evaluation_evidence_may_become_training_evidence": False,
    }


def build_generation4_acoustic_contract(
    calibration_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the G1B design receipt from existing calibration evidence only."""
    packet = dict(calibration_packet)
    _reject_generation4_payloads(packet)
    if (
        packet.get("split") != "calibration"
        or packet.get("threshold_unit_count") != 9
        or packet.get("selection_objective") != list(CALIBRATION_SELECTION_OBJECTIVE)
        or packet.get("did_read_generation4_gold") is not False
        or packet.get("did_read_generation4_holdout") is not False
        or packet.get("did_load_or_run_models") is not False
        or not all(
            _SHA256_RE.fullmatch(str(packet.get(field) or ""))
            for field in (
                "threshold_application_sha256",
                "threshold_set_sha256",
                "score_matrix_sha256",
            )
        )
    ):
        raise Generation4AcousticContractError(
            "Factor selection requires exact calibration-only authority."
        )
    units = _validated_units(packet)
    threshold_set_sha256 = str(packet["threshold_set_sha256"])
    selected = min(
        units, key=lambda unit: _selection_rank(unit, threshold_set_sha256)
    )
    factor_hash = _factor_contract_hash(selected, threshold_set_sha256)
    unit_set_hash = _canonical_hash(
        sorted(
            _factor_contract_hash(unit, threshold_set_sha256)
            for unit in units
        )
    )
    evidence_schema = {
        "schema_version": EVIDENCE_SCHEMA,
        "required_fields": [
            "schema_version",
            "factor_contract_sha256",
            "calibration_lineage_sha256",
            "probe_contract_sha256",
            "reference_contract_sha256",
            "condition_assignment_sha256",
            "outcome",
            "reason_code",
        ],
        "factor_is_separately_visible": True,
        "allowed_outcomes": ["supports", "conflicts", "unavailable"],
        "missing_or_unusable_is_neutral": True,
        "requires_factor_contract_sha256": True,
        "requires_calibration_lineage_sha256": True,
        "requires_probe_and_reference_contract_sha256": True,
        "forbids_hidden_fusion": True,
        "forbids_raw_similarity_or_threshold_values": True,
    }
    selection_policy = {
        "schema_version": SELECTION_POLICY_SCHEMA,
        "basis": "frozen_calibration_only",
        "eligible_unit_count": 9,
        "primary_objective": "minimum_calibration_brier_loss",
        "tie_break_precedence": [
            "minimum_expected_calibration_error",
            "minimum_balanced_error",
            "minimum_false_accept_reject_gap",
            "maximum_frozen_calibration_threshold",
            "minimum_frozen_calibration_temperature",
            "maximum_open_set_rejection",
            "maximum_minimum_candidate_margin",
            "minimum_opaque_factor_contract_sha256",
        ],
        "generation4_gold_or_holdout_may_select_factor": False,
        "selected_factor_count": 1,
    }
    conditions = _condition_taxonomy()
    denominators = _denominator_contract()
    exact_trials = _exact_trial_contract()
    contract_hashes = {
        "acoustic_evidence_schema_sha256": _canonical_hash(evidence_schema),
        "selection_policy_sha256": _canonical_hash(selection_policy),
        "condition_taxonomy_sha256": _canonical_hash(conditions),
        "denominator_proof_sha256": _canonical_hash(denominators),
        "exact_trial_replay_sha256": _canonical_hash(exact_trials),
    }
    returned_evidence_sha256 = _canonical_hash(
        {
            "calibration_authority_sha256": packet["threshold_application_sha256"],
            "calibration_threshold_set_sha256": packet["threshold_set_sha256"],
            "calibration_matrix_sha256": packet["score_matrix_sha256"],
            "selected_factor_contract_sha256": factor_hash,
            "full_matrix_unit_set_sha256": unit_set_hash,
            "contract_hashes": contract_hashes,
        }
    )
    delegation_receipt = {
        "status": "spawned",
        "lane": "G1B",
        "runtime_handle": "/root/g1b_acoustic_contract",
        "terminal_status": "completed",
        "returned_evidence_sha256": returned_evidence_sha256,
        "primary_reconciliation": "pending_j1",
    }
    actions = {action: False for action in PROHIBITED_ACTIONS}
    actions["run_j1_design_reconciliation"] = True
    core = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "g1b_acoustic_contract_complete",
        "g0_preview_sha256": G0_PREVIEW_SHA256,
        "g0_manifest_sha256": G0_MANIFEST_SHA256,
        "calibration_authority_sha256": packet["threshold_application_sha256"],
        "calibration_threshold_set_sha256": packet["threshold_set_sha256"],
        "calibration_matrix_sha256": packet["score_matrix_sha256"],
        "selected_factor_count": 1,
        "selected_factor_contract_sha256": factor_hash,
        "full_matrix_unit_count": 9,
        "full_matrix_unit_set_sha256": unit_set_hash,
        "acoustic_evidence_schema": evidence_schema,
        "selection_policy": selection_policy,
        "condition_taxonomy": conditions,
        "denominator_proof_contract": denominators,
        "exact_trial_replay_contract": exact_trials,
        "contract_hashes": contract_hashes,
        "returned_evidence_sha256": returned_evidence_sha256,
        "delegation_receipt": delegation_receipt,
        "action_vector": actions,
        "contains_candidate_or_method_ids": False,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_frozen_threshold_values": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "did_read_generation4_gold": False,
        "did_read_generation4_holdout": False,
        "did_load_or_run_models": False,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def replay_generation4_acoustic_contract(
    calibration_packet: Mapping[str, Any], *, expected_content_sha256: str
) -> dict[str, Any]:
    """Rebuild the pure G1B contract and require exact content identity."""
    contract = build_generation4_acoustic_contract(calibration_packet)
    if contract["content_sha256"] != expected_content_sha256:
        raise Generation4AcousticContractError("Generation-4 acoustic contract drifted.")
    return {
        **contract,
        "replay_schema_version": CONTRACT_REPLAY_SCHEMA,
        "idempotent_replay": True,
    }
