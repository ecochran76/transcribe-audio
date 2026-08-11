from __future__ import annotations

import speaker_identity_plan0068_a0 as a0
import speaker_identity_plan0068_terminal as terminal


def test_terminal_withholds_three_invalid_retained_outputs_without_effects() -> None:
    value = terminal.build_terminal(
        a0_manifest={"case_count": 6, "content_sha256": "a" * 64},
        a2_manifest={
            "content_sha256": "b" * 64,
            "measurement": {
                "passed": False,
                "correct_prepared_candidate_count": 1,
                "wrong_prepared_candidate_count": 0,
                "validation_failure_count": 3,
            },
            "execution_counts": {
                "retained_output_replays": 6,
                "primary_model_turns": 0,
                "fallback_model_turns": 0,
                "retries": 0,
                "reference_repairs": 0,
                "fresh_retrievals": 0,
                "fresh_evaluations": 0,
            },
            "effect_counts": dict(a0.EFFECT_COUNTS),
        },
        source_commit="c" * 40,
    )

    assert value["status"] == "plan0068_closed_withhold"
    assert value["reason_code"] == "retained_output_schema_compliance_failed"
    assert value["validated_case_count"] == 3
    assert value["validation_failure_count"] == 3
    assert value["original_recording_filename_count"] == 6
    assert value["retained_output_change_count"] == 0
    assert value["effect_counts"] == a0.EFFECT_COUNTS
