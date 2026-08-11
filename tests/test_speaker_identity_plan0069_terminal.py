from __future__ import annotations

import pytest

import speaker_identity_plan0069_terminal as terminal


def _authorities() -> tuple[dict, dict]:
    a0 = {
        "content_sha256": "a" * 64,
        "case_count": 6,
        "original_recording_filename_count": 6,
    }
    a2 = {
        "content_sha256": "b" * 64,
        "measurement": {
            "status": "context_candidate_recovered",
            "passed": True,
            "correct_prepared_candidate_count": 5,
            "wrong_prepared_candidate_count": 0,
            "abstained_slot_count": 17,
            "incomplete_candidate_provenance_count": 0,
            "unavailable_case_count": 0,
            "validation_failure_count": 0,
        },
        "execution_counts": {
            "retained_output_replays": 6,
            "primary_model_turns": 0,
            "fallback_model_turns": 0,
            "retries": 0,
            "model_reference_repairs": 0,
            "fresh_retrievals": 0,
            "fresh_evaluations": 0,
        },
        "normalized_group_count": 10,
        "expanded_utterance_assignment_count": 28,
        "retained_output_change_count": 0,
        "source_store_index_change_count": 0,
        "effect_counts": dict(terminal.EFFECT_COUNTS),
    }
    return a0, a2


def test_terminal_closes_pass_with_six_filenames_and_zero_effects() -> None:
    a0, a2 = _authorities()

    result = terminal.build_terminal(
        a0_manifest=a0,
        a2_manifest=a2,
        source_commit="c" * 40,
    )

    assert result["status"] == "plan0069_closed_pass"
    assert result["decision"] == "pass"
    assert result["original_recording_filename_count"] == 6
    assert result["normalized_group_count"] == 10
    assert result["effect_counts"] == terminal.EFFECT_COUNTS


def test_terminal_rejects_wrong_or_unavailable_candidate() -> None:
    a0, a2 = _authorities()
    a2["measurement"] = {
        **a2["measurement"],
        "wrong_prepared_candidate_count": 1,
        "unavailable_case_count": 1,
    }

    with pytest.raises(terminal.Plan0069TerminalError, match="measurement differs"):
        terminal.build_terminal(
            a0_manifest=a0,
            a2_manifest=a2,
            source_commit="c" * 40,
        )
