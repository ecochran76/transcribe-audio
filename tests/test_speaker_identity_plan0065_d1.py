from __future__ import annotations

import speaker_identity_plan0065_d1 as d1


def _calibration():
    application = {
        "score_matrix_sha256": "a" * 64,
        "thresholds": [
            {
                "candidate_id": "model-a",
                "method_id": "no_enhancement",
                "threshold": 0.4,
                "metrics": {
                    "false_accept_count": 0,
                    "false_reject_count": 0,
                    "genuine_trial_count": 3,
                    "impostor_trial_count": 3,
                },
            },
            {
                "candidate_id": "model-b",
                "method_id": "no_enhancement",
                "threshold": 0.5,
                "metrics": {
                    "false_accept_count": 0,
                    "false_reject_count": 0,
                    "genuine_trial_count": 3,
                    "impostor_trial_count": 3,
                },
            },
            {
                "candidate_id": "model-unusable",
                "method_id": "no_enhancement",
                "threshold": 1.0,
                "metrics": {
                    "false_accept_count": 0,
                    "false_reject_count": 3,
                    "genuine_trial_count": 3,
                    "impostor_trial_count": 3,
                },
            },
        ],
    }
    trials = []
    for model, scores in (
        ("model-a", (0.7, 0.75, 0.8)),
        ("model-b", (0.68, 0.72, 0.76)),
    ):
        trials.extend(
            {
                "candidate_id": model,
                "method_id": "no_enhancement",
                "expected_match": True,
                "score": score,
            }
            for score in scores
        )
    return application, {"trials": trials}


def _slot(score_a: float, score_b: float, *, overlap: float = 0.0):
    return {
        "speaker_ref": "0123456789abcdefabcd::A",
        "status": "candidate",
        "candidate_person_id": "person-a",
        "candidate_acoustic_subject_id": "subject-a",
        "probe_duration_seconds": 20.0,
        "probe_audit": {
            "other_speaker_overlap_seconds": overlap,
            "source_hash_matches": True,
            "probe_hash_matches": True,
        },
        "model_rows": [
            {
                "candidate_id": "model-a",
                "top_canonical_person_id": "person-a",
                "top_acoustic_subject_id": "subject-a",
                "top_score": score_a,
                "threshold": 0.4,
                "threshold_pass": True,
                "binding_eligible": True,
            },
            {
                "candidate_id": "model-b",
                "top_canonical_person_id": "person-a",
                "top_acoustic_subject_id": "subject-a",
                "top_score": score_b,
                "threshold": 0.5,
                "threshold_pass": True,
                "binding_eligible": True,
            },
        ],
    }


def test_policy_floor_is_derived_from_frozen_genuine_calibration_support():
    application, matrix = _calibration()

    policy = d1.build_acoustic_safety_policy(
        threshold_application=application,
        score_matrix=matrix,
        threshold_application_file_sha256="b" * 64,
        score_matrix_file_sha256="a" * 64,
    )

    assert policy["safety_ratio"] == {"numerator": 2, "denominator": 3}
    assert policy["model_floors"] == [
        {
            "candidate_id": "model-a",
            "threshold": 0.4,
            "minimum_calibration_genuine_score": 0.7,
            "minimum_safe_score": 0.6,
        },
        {
            "candidate_id": "model-b",
            "threshold": 0.5,
            "minimum_calibration_genuine_score": 0.68,
            "minimum_safe_score": 0.62,
        },
    ]
    assert policy["excluded_models"] == [
        {
            "candidate_id": "model-unusable",
            "reason_code": "calibration_did_not_accept_genuine_trials",
        }
    ]


def test_boundary_impostor_is_reviewed_while_buffered_candidate_is_retained():
    application, matrix = _calibration()
    policy = d1.build_acoustic_safety_policy(
        threshold_application=application,
        score_matrix=matrix,
        threshold_application_file_sha256="b" * 64,
        score_matrix_file_sha256="a" * 64,
    )

    boundary = d1.apply_acoustic_safety_policy(_slot(0.59, 0.61), policy)
    retained = d1.apply_acoustic_safety_policy(_slot(0.61, 0.63), policy)

    assert boundary["status"] == "review"
    assert boundary["reason_code"] == "calibration_boundary_guard"
    assert boundary["candidate_person_id"] is None
    assert retained["status"] == "candidate"
    assert retained["reason_code"] == "multi_model_calibration_buffer_support"


def test_mixed_speaker_overlap_fails_purity_before_scores_are_considered():
    application, matrix = _calibration()
    policy = d1.build_acoustic_safety_policy(
        threshold_application=application,
        score_matrix=matrix,
        threshold_application_file_sha256="b" * 64,
        score_matrix_file_sha256="a" * 64,
    )

    result = d1.apply_acoustic_safety_policy(
        _slot(0.9, 0.9, overlap=0.25),
        policy,
    )

    assert result["status"] == "review"
    assert result["reason_code"] == "diarization_overlap_guard"
    assert result["candidate_person_id"] is None


def test_background_abstention_and_cross_model_disagreement_do_not_promote():
    application, matrix = _calibration()
    policy = d1.build_acoustic_safety_policy(
        threshold_application=application,
        score_matrix=matrix,
        threshold_application_file_sha256="b" * 64,
        score_matrix_file_sha256="a" * 64,
    )
    background = _slot(0.0, 0.0)
    background.update(
        status="abstain",
        reason_code="no_bound_profile_threshold_pass",
        candidate_person_id=None,
        candidate_acoustic_subject_id=None,
        model_rows=[],
    )
    disagreement = _slot(0.9, 0.9)
    disagreement["model_rows"][1]["top_canonical_person_id"] = "person-b"

    background_result = d1.apply_acoustic_safety_policy(background, policy)
    disagreement_result = d1.apply_acoustic_safety_policy(disagreement, policy)

    assert background_result["status"] == "abstain"
    assert background_result["candidate_person_id"] is None
    assert disagreement_result["status"] == "review"
    assert disagreement_result["reason_code"] == "calibration_boundary_guard"
    assert disagreement_result["candidate_person_id"] is None


def test_development_gate_requires_zero_wrong_and_retains_ten_correct():
    rows = [
        {"gold": "correct", "before": "candidate", "after": "candidate"}
        for _ in range(10)
    ]
    rows.extend(
        [
            {"gold": "correct", "before": "candidate", "after": "review"},
            {"gold": "wrong", "before": "candidate", "after": "review"},
        ]
    )

    gate = d1.evaluate_development_gate(rows)

    assert gate["original_correct_candidate_count"] == 11
    assert gate["retained_correct_candidate_count"] == 10
    assert gate["retained_wrong_candidate_count"] == 0
    assert gate["passed"] is True
