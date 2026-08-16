from __future__ import annotations

import pytest

from acoustic_audio_derivatives import canonical_artifact_hash
import speaker_identity_plan0071_d3_measurement as d3


def _content(value: dict) -> dict:
    return {**value, "content_sha256": canonical_artifact_hash(value)}


def _authority() -> dict:
    return _content(
        {
            "schema_version": d3.REVIEW_AUTHORITY_SCHEMA,
            "cases": [
                {"speaker_ref": f"document-{index // 3 + 1}::{chr(65 + index % 3)}"}
                for index in range(18)
            ],
            "people": [
                {"person_id": "person-1", "display_name": "Person One"},
            ],
            "human_decision_count": 0,
            "model_predictions_visible": False,
            "mutation_effect_counts": dict(d3.MUTATION_EFFECT_COUNTS),
        }
    )


def _resolution() -> dict:
    return _content(
        {
            "schema_version": d3.PREDICTION_RESOLUTION_SCHEMA,
            "recordings": [
                {
                    "speaker_slots": [
                        {
                            "speaker_ref": (
                                f"document-{recording_index + 1}::{chr(65 + speaker_index)}"
                            )
                        }
                        for speaker_index in range(3)
                    ]
                }
                for recording_index in range(6)
            ],
            "contains_gold": False,
            "mutation_effect_counts": dict(d3.MUTATION_EFFECT_COUNTS),
        }
    )


def test_normalize_human_gold_accepts_one_exact_complete_export() -> None:
    authority = _authority()
    resolution = _resolution()
    submission = {
        "schema_version": d3.DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": [
            {
                "speaker_ref": item["speaker_ref"],
                "decision": "canonical_person" if index == 0 else "not_listed",
                "person_id": "person-1" if index == 0 else None,
                "note": "" if index == 0 else "A literal reviewer note",
            }
            for index, item in enumerate(authority["cases"])
        ],
    }

    gold = d3.normalize_human_gold(
        submission,
        authority=authority,
        resolution=resolution,
    )

    assert gold["status"] == "complete_literal_human_gold"
    assert gold["decision_count"] == 18
    assert gold["decision_type_counts"] == {
        "canonical_person": 1,
        "not_listed": 17,
    }
    assert gold["decisions"] == submission["decisions"]
    assert all(value == 0 for value in gold["mutation_effect_counts"].values())


def test_normalize_human_gold_rejects_unallowlisted_or_reordered_rows() -> None:
    authority = _authority()
    resolution = _resolution()
    decisions = [
        {
            "speaker_ref": item["speaker_ref"],
            "decision": "not_listed",
            "person_id": None,
            "note": "",
        }
        for item in authority["cases"]
    ]
    decisions[0] = {
        "speaker_ref": decisions[1]["speaker_ref"],
        "decision": "canonical_person",
        "person_id": "not-allowlisted",
        "note": "",
    }
    submission = {
        "schema_version": d3.DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": decisions,
    }

    with pytest.raises(d3.Plan0071D3MeasurementError):
        d3.normalize_human_gold(
            submission,
            authority=authority,
            resolution=resolution,
        )


def _abstaining_slot(speaker_ref: str) -> dict:
    return {
        "speaker_ref": speaker_ref,
        "acoustic": {
            "disposition": "abstain",
            "candidate_person_id": None,
            "reason_code": "no_bound_profile_threshold_pass",
        },
        "context": {
            "disposition": "unavailable",
            "candidate_person_id": None,
            "reason_code": "speaker_missing_from_context_evaluation",
            "candidates": [],
        },
        "combined": {
            "disposition": "abstain",
            "candidate_person_id": None,
            "reason_code": "no_joined_candidate",
        },
        "residual_policy": {
            "disposition": "abstain",
            "candidate_person_id": None,
            "reason_code": "no_joined_candidate",
        },
    }


def _passing_resolution(authority: dict) -> dict:
    slots = [_abstaining_slot(item["speaker_ref"]) for item in authority["cases"]]
    acoustic = {
        "disposition": "candidate",
        "candidate_person_id": "person-1",
        "reason_code": "multi_model_acoustic_support",
        "probe_sha256": "a" * 64,
        "supporting_model_count": 2,
    }
    context = {
        "disposition": "candidate",
        "candidate_person_id": "person-1",
        "reason_code": "context_candidate",
        "candidates": [
            {
                "status": "candidate_match",
                "prepared_person_id": "person-1",
                "transcript_clue_ids": ["clue-1"],
                "provenance_source_ids": ["source-1"],
            }
        ],
    }
    slots[0] = {
        "speaker_ref": slots[0]["speaker_ref"],
        "acoustic": dict(acoustic),
        "context": dict(context),
        "combined": {
            "disposition": "candidate",
            "candidate_person_id": "person-1",
            "reason_code": "pillar_agreement",
        },
        "residual_policy": {
            "disposition": "candidate",
            "candidate_person_id": "person-1",
            "reason_code": "pillar_agreement",
        },
    }
    slots[1] = {
        "speaker_ref": slots[1]["speaker_ref"],
        "acoustic": dict(acoustic),
        "context": {**context, "disposition": "review", "candidate_person_id": None},
        "combined": {
            "disposition": "review",
            "candidate_person_id": None,
            "reason_code": "acoustic_only_support",
        },
        "residual_policy": {
            "disposition": "candidate",
            "candidate_person_id": "person-1",
            "reason_code": "two_known_plus_one_independently_supported_residual",
        },
    }
    return _content(
        {
            "schema_version": d3.PREDICTION_RESOLUTION_SCHEMA,
            "recordings": [
                {"speaker_slots": slots[index : index + 3]}
                for index in range(0, 18, 3)
            ],
            "contains_gold": False,
            "mutation_effect_counts": dict(d3.MUTATION_EFFECT_COUNTS),
        }
    )


def test_measure_d3_opens_fresh_evaluation_only_for_nonvacuous_safe_pass() -> None:
    authority = _authority()
    resolution = _passing_resolution(authority)
    submission = {
        "schema_version": d3.DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": [
            {
                "speaker_ref": item["speaker_ref"],
                "decision": "canonical_person" if index < 2 else "not_listed",
                "person_id": "person-1" if index < 2 else None,
                "note": "",
            }
            for index, item in enumerate(authority["cases"])
        ],
    }
    gold = d3.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )

    measurement = d3.measure_d3(
        authority=authority,
        resolution=resolution,
        gold=gold,
    )

    assert measurement["terminal_decision"] == "advance_to_fresh_evaluation"
    assert measurement["fresh_evaluation_allowed"] is True
    assert measurement["condition_metrics"]["combined"][
        "pillar_agreement_correct_count"
    ] == 1
    assert measurement["condition_metrics"]["residual_policy"][
        "residual_rule_correct_count"
    ] == 1
    assert measurement["acceptance_gate"]["failed_checks"] == []


def test_measure_d3_closes_residual_population_when_no_residual_candidate_exists() -> None:
    authority = _authority()
    slots = [_abstaining_slot(item["speaker_ref"]) for item in authority["cases"]]
    for index in range(7):
        slots[index]["acoustic"] = {
            "disposition": "candidate",
            "candidate_person_id": "person-1",
            "reason_code": "multi_model_acoustic_support",
            "probe_sha256": "a" * 64,
            "supporting_model_count": 2,
        }
        slots[index]["combined"] = {
            "disposition": "review",
            "candidate_person_id": None,
            "reason_code": "acoustic_only_support",
        }
        slots[index]["residual_policy"] = {
            "disposition": "review",
            "candidate_person_id": None,
            "reason_code": "acoustic_only_support",
        }
    resolution = _content(
        {
            "schema_version": d3.PREDICTION_RESOLUTION_SCHEMA,
            "recordings": [
                {"speaker_slots": slots[index : index + 3]}
                for index in range(0, 18, 3)
            ],
            "contains_gold": False,
            "mutation_effect_counts": dict(d3.MUTATION_EFFECT_COUNTS),
        }
    )
    submission = {
        "schema_version": d3.DECISION_SCHEMA,
        "authority_content_sha256": authority["content_sha256"],
        "decisions": [
            {
                "speaker_ref": item["speaker_ref"],
                "decision": (
                    "canonical_person"
                    if index < 6
                    else "unresolved"
                    if index == 6
                    else "not_listed"
                ),
                "person_id": "person-1" if index < 6 else None,
                "note": "",
            }
            for index, item in enumerate(authority["cases"])
        ],
    }
    gold = d3.normalize_human_gold(
        submission, authority=authority, resolution=resolution
    )

    measurement = d3.measure_d3(
        authority=authority,
        resolution=resolution,
        gold=gold,
    )

    assert measurement["terminal_decision"] == "residual_population_infeasible"
    assert measurement["fresh_evaluation_allowed"] is False
    acoustic = measurement["condition_metrics"]["acoustic"]
    assert acoustic["correct_candidate_count"] == 6
    assert acoustic["wrong_candidate_count"] == 0
    assert acoustic["unverifiable_candidate_count"] == 1
    assert measurement["condition_metrics"]["combined"]["candidate_count"] == 0
    assert measurement["condition_metrics"]["residual_policy"][
        "residual_rule_candidate_count"
    ] == 0
    assert measurement["acceptance_gate"]["failed_checks"] == [
        "pillar_agreement_correct_acceptance_observed",
        "residual_correct_acceptance_observed",
    ]
