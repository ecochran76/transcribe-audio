from __future__ import annotations

import hashlib
import json

import pytest

import acoustic_generation4_acoustic_contract as contract


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def calibration_packet() -> dict:
    units = []
    for candidate_index, candidate_id in enumerate(contract.CANDIDATE_IDS):
        for method_index, method_id in enumerate(contract.METHOD_IDS):
            rank = candidate_index * len(contract.METHOD_IDS) + method_index
            units.append(
                {
                    "candidate_id": candidate_id,
                    "method_id": method_id,
                    "status": "success",
                    "reason_code": None,
                    "threshold": 0.25 + rank / 100,
                    "temperature": 0.1,
                    "metrics": {
                        "missing_denominator_status": "success",
                        "trial_count": 44,
                        "genuine_trial_count": 9,
                        "impostor_trial_count": 35,
                        "brier_score": 0.01 + rank / 100,
                        "expected_calibration_error_5_bins": 0.02,
                        "balanced_error_rate": 0.03,
                        "false_acceptance_rate": 0.04,
                        "false_rejection_rate": 0.05,
                    },
                    "open_set_rejection": {
                        "status": "descriptive",
                        "probe_count": 13,
                        "rejection_rate": 0.75,
                    },
                    "candidate_margin": {
                        "status": "descriptive",
                        "count": 22,
                        "minimum": 0.05,
                        "mean": 0.1,
                        "maximum": 0.2,
                    },
                }
            )
    return {
        "split": "calibration",
        "threshold_application_sha256": SHA_A,
        "threshold_set_sha256": SHA_B,
        "score_matrix_sha256": SHA_C,
        "threshold_unit_count": 9,
        "selection_objective": list(contract.CALIBRATION_SELECTION_OBJECTIVE),
        "thresholds": units,
        "did_read_generation4_gold": False,
        "did_read_generation4_holdout": False,
        "did_load_or_run_models": False,
    }


def test_complete_calibration_selects_one_opaque_factor_and_freezes_contracts():
    receipt = contract.build_generation4_acoustic_contract(calibration_packet())

    assert receipt["status"] == "g1b_acoustic_contract_complete"
    assert receipt["selected_factor_count"] == 1
    assert len(receipt["selected_factor_contract_sha256"]) == 64
    assert receipt["full_matrix_unit_count"] == 9
    assert receipt["denominator_proof_contract"]["minimum_trial_count_per_unit"] == 140
    assert receipt["denominator_proof_contract"]["minimum_total_trial_count"] == 1260
    assert receipt["condition_taxonomy"]["required_dimension_count"] == 5
    assert receipt["exact_trial_replay_contract"]["run_count"] == 1
    assert receipt["delegation_receipt"] == {
        "status": "spawned",
        "lane": "G1B",
        "runtime_handle": "/root/g1b_acoustic_contract",
        "terminal_status": "completed",
        "returned_evidence_sha256": receipt["returned_evidence_sha256"],
        "primary_reconciliation": "pending_j1",
    }
    assert receipt["action_vector"]["run_j1_design_reconciliation"] is True
    assert all(
        receipt["action_vector"][action] is False
        for action in contract.PROHIBITED_ACTIONS
    )


def test_exact_contract_replay_requires_the_frozen_content_hash():
    packet = calibration_packet()
    frozen = contract.build_generation4_acoustic_contract(packet)

    replay = contract.replay_generation4_acoustic_contract(
        packet, expected_content_sha256=frozen["content_sha256"]
    )

    assert replay["content_sha256"] == frozen["content_sha256"]
    assert replay["replay_schema_version"] == contract.CONTRACT_REPLAY_SCHEMA
    assert replay["idempotent_replay"] is True

    with pytest.raises(contract.Generation4AcousticContractError, match="drifted"):
        contract.replay_generation4_acoustic_contract(
            packet, expected_content_sha256="f" * 64
        )


def test_factor_selection_rejects_any_generation4_gold_or_holdout_payload():
    packet = calibration_packet()
    packet["generation4_gold"] = {"speaker_labels": ["must-not-be-read"]}

    with pytest.raises(contract.Generation4AcousticContractError, match="forbidden"):
        contract.build_generation4_acoustic_contract(packet)


def test_calibration_quality_beats_opaque_lexicographic_tie_break():
    packet = calibration_packet()
    lexicographically_first = packet["thresholds"][0]
    quality_winner = packet["thresholds"][-1]
    for unit in packet["thresholds"]:
        unit["metrics"]["brier_score"] = 0.9
    quality_winner["metrics"]["brier_score"] = 0.001

    receipt = contract.build_generation4_acoustic_contract(packet)
    expected = hashlib.sha256(
        json.dumps(
            {
                "schema_version": contract.FACTOR_CONTRACT_SCHEMA,
                "candidate_id": quality_winner["candidate_id"],
                "method_id": quality_winner["method_id"],
                "threshold_set_sha256": packet["threshold_set_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    lexicographic_hash = hashlib.sha256(
        json.dumps(
            {
                "schema_version": contract.FACTOR_CONTRACT_SCHEMA,
                "candidate_id": lexicographically_first["candidate_id"],
                "method_id": lexicographically_first["method_id"],
                "threshold_set_sha256": packet["threshold_set_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert receipt["selected_factor_contract_sha256"] == expected
    assert receipt["selected_factor_contract_sha256"] != lexicographic_hash
    assert receipt["selection_policy"]["primary_objective"] == (
        "minimum_calibration_brier_loss"
    )
    assert receipt["selection_policy"]["tie_break_precedence"][-1] == (
        "minimum_opaque_factor_contract_sha256"
    )


@pytest.mark.parametrize("defect", ["missing_unit", "nonfinite_quality", "gold_read"])
def test_untrusted_calibration_packet_fails_closed(defect: str):
    packet = calibration_packet()
    if defect == "missing_unit":
        packet["thresholds"].pop()
    elif defect == "nonfinite_quality":
        packet["thresholds"][0]["metrics"]["brier_score"] = float("nan")
    else:
        packet["did_read_generation4_gold"] = True

    with pytest.raises(contract.Generation4AcousticContractError):
        contract.build_generation4_acoustic_contract(packet)


def test_portable_receipt_exposes_no_factor_identity_or_private_evidence():
    packet = calibration_packet()
    receipt = contract.build_generation4_acoustic_contract(packet)
    portable = json.dumps(receipt, sort_keys=True)

    for candidate_id in contract.CANDIDATE_IDS:
        assert candidate_id not in portable
    for method_id in contract.METHOD_IDS:
        assert method_id not in portable
    assert "must-not-be-read" not in portable
    assert "0.25" not in portable
    assert receipt["contains_candidate_or_method_ids"] is False
    assert receipt["contains_profile_or_subject_ids"] is False
    assert receipt["contains_paths"] is False
    assert receipt["contains_biometric_scores"] is False
    assert receipt["contains_frozen_threshold_values"] is False
    assert receipt["contains_embeddings_or_vectors"] is False
    assert receipt["contains_raw_audio"] is False
    assert receipt["contains_transcript_text"] is False


def test_g2_exact_denominators_may_exceed_but_never_fall_below_minima():
    proof = contract.validate_g2_exact_denominator_counts(
        {
            "genuine_trials_per_unit": 21,
            "known_impostor_trials_per_unit": 101,
            "open_set_trials_per_unit": 20,
        }
    )

    assert proof["status"] == "g2_exact_denominators_meet_g1b_minima"
    assert proof["exact_trial_count_per_unit"] == 142

    with pytest.raises(contract.Generation4AcousticContractError, match="minimum"):
        contract.validate_g2_exact_denominator_counts(
            {
                "genuine_trials_per_unit": 19,
                "known_impostor_trials_per_unit": 101,
                "open_set_trials_per_unit": 20,
            }
        )
