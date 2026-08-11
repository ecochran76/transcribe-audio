from __future__ import annotations

import speaker_identity_plan0066_a0 as a0
import speaker_identity_plan0066_terminal as terminal


def test_terminal_withholds_on_reference_validation_failures() -> None:
    value = terminal.build_terminal(
        a0_receipt={
            "status": "a0_frozen_zero_effect",
            "activation_content_sha256": "a" * 64,
        },
        a1_receipt={
            "status": "a1_passed_zero_source_mutation",
            "manifest_content_sha256": "b" * 64,
            "reviewed_person_count": 6,
        },
        a2_receipt={
            "manifest_content_sha256": "c" * 64,
            "measurement": {
                "passed": False,
                "validation_failure_count": 6,
                "correct_prepared_candidate_count": 0,
                "wrong_prepared_candidate_count": 0,
            },
            "execution_counts": {
                "primary_model_turns": 6,
                "fallback_model_turns": 0,
                "retries": 0,
            },
            "source_store_index_change_count": 0,
        },
    )

    assert value["status"] == "plan0066_closed_withhold"
    assert value["reason_code"] == "evidence_reference_compliance_failed"
    assert value["effect_counts"] == a0.EFFECT_COUNTS
    assert value["joined_or_residual_gate_opened"] is False
