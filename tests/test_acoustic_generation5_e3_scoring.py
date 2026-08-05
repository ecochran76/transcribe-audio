from __future__ import annotations

import copy

import pytest

import acoustic_generation5_e3_scoring as e3


def fixture() -> tuple[list[str], dict, dict, dict]:
    refs = [f"Speaker {index}" for index in range(22)]
    gold = {ref: {"private_identity_display": f"Person {index}",
                  "enrolled_subject_id": "enrolled" if index < 4 else ""}
            for index, ref in enumerate(refs)}
    context = {"predictions": []}
    augmented = {"predictions": []}
    for index, ref in enumerate(refs):
        identity = f"Person {index}"
        context["predictions"].append({
            "speaker_ref": ref,
            "identity_or_alias": "wrong" if index in {0, 1} else identity,
            "confidence_band": "medium", "disposition": "review" if index in {0, 1} else "assign",
            "rationale": "fixture",
        })
        augmented["predictions"].append({
            "speaker_ref": ref, "identity_or_alias": identity,
            "confidence_band": "high", "disposition": "assign", "rationale": "fixture",
        })
    return refs, gold, context, augmented


def test_score_prediction_pair_computes_complete_paired_metrics() -> None:
    refs, gold, context, augmented = fixture()
    score = e3.score_prediction_pair(
        expected_refs=refs, gold_by_ref=gold,
        context_predictions=context, augmented_predictions=augmented,
    )
    assert score["speaker_count"] == 22
    assert score["enrolled_speaker_count"] == 4
    assert score["context_only"]["correct_assignment"] == 20
    assert score["voice_augmented"]["correct_assignment"] == 22
    assert score["paired"]["corrected_baseline_error_count"] == 2
    assert score["paired"]["safe_review_resolution_count"] == 2
    assert score["paired"]["introduced_error_count"] == 0


def test_high_confidence_wrong_is_counted() -> None:
    refs, gold, context, augmented = fixture()
    augmented["predictions"][0]["identity_or_alias"] = "Wrong person"
    score = e3.score_prediction_pair(
        expected_refs=refs, gold_by_ref=gold,
        context_predictions=context, augmented_predictions=augmented,
    )
    assert score["voice_augmented"]["high_confidence_wrong"] == 1
    assert score["voice_augmented"]["wrong_assignment"] == 1


def test_alias_canonicalization_matches_doctor_title() -> None:
    refs, gold, context, augmented = fixture()
    gold[refs[0]]["private_identity_display"] = "Jeffrey Dikis"
    augmented["predictions"][0]["identity_or_alias"] = "Dr. Jeffrey Dikis"
    context["predictions"][0] = copy.deepcopy(augmented["predictions"][0])
    score = e3.score_prediction_pair(
        expected_refs=refs, gold_by_ref=gold,
        context_predictions=context, augmented_predictions=augmented,
    )
    assert score["voice_augmented"]["correct_assignment"] == 22


def test_incomplete_denominator_fails_closed() -> None:
    refs, gold, context, augmented = fixture()
    with pytest.raises(e3.Generation5E3Error, match="denominator"):
        e3.score_prediction_pair(
            expected_refs=refs, gold_by_ref=gold,
            context_predictions=context,
            augmented_predictions={"predictions": augmented["predictions"][:-1]},
        )

