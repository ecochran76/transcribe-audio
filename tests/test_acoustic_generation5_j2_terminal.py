from __future__ import annotations

import acoustic_generation5_j2_terminal as j2


def metrics() -> dict:
    return {
        "context_only": {"assignment_correctness": 0.0, "candidate_recall": 0.0},
        "voice_augmented": {"assignment_correctness": 6 / 22, "candidate_recall": 6 / 22,
                            "high_confidence_wrong": 0},
        "paired": {"corrected_baseline_error_count": 6, "safe_review_resolution_count": 0,
                   "introduced_error_count": 0},
    }


def test_terminal_precedence_stops_on_gate_failure() -> None:
    assert j2.terminal_decision(metrics(), all_gates_pass=False) == "stop"


def test_terminal_precedence_rejects_high_confidence_wrong() -> None:
    value = metrics()
    value["voice_augmented"]["high_confidence_wrong"] = 1
    assert j2.terminal_decision(value, all_gates_pass=True) == "reject_acoustic_factor"


def test_terminal_precedence_rejects_regression() -> None:
    value = metrics()
    value["context_only"]["candidate_recall"] = 0.5
    assert j2.terminal_decision(value, all_gates_pass=True) == "reject_acoustic_factor"


def test_terminal_precedence_advances_accepted_result() -> None:
    assert j2.terminal_decision(metrics(), all_gates_pass=True) == "advance_to_limited_pilot_plan"


def test_terminal_precedence_keeps_shadow_without_improvement() -> None:
    value = metrics()
    value["paired"]["corrected_baseline_error_count"] = 0
    assert j2.terminal_decision(value, all_gates_pass=True) == "keep_shadow_and_refine"


def test_live_preview_binds_independent_pass() -> None:
    repository = {"commit": "a" * 40, "module_sha256": "b" * 64,
                  "clean": True, "upstream_ahead": 0, "upstream_behind": 0}
    preview = j2.preview_generation5_j2_terminal(repository_authority=repository)
    assert preview["status"] == "independent_j2_pass"
    assert preview["terminal_decision"] == "advance_to_limited_pilot_plan"
    assert preview["findings"]["unique_trial_count"] == 396
    assert preview["action_vector"]["open_limited_pilot_plan"] is True
    assert preview["action_vector"]["mutate_profiles_or_references"] is False

