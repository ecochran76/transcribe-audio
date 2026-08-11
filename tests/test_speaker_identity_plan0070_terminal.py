from __future__ import annotations

import speaker_identity_plan0070_d0 as d0
import speaker_identity_plan0070_terminal as terminal


def test_terminal_withholds_after_two_zero_effect_d0_attempts() -> None:
    value = terminal.build_terminal(source_commit="a" * 40)

    assert value["status"] == "plan0070_closed_withhold"
    assert value["reason_code"] == "d0_authority_harness_shape_mismatch"
    assert value["d0_attempt_count"] == 2
    assert value["d0_artifact_written"] is False
    assert value["d3_counterfactual"] == d0.EXPECTED_D3_START
    assert value["supplemental_development_opened"] is False
    assert value["fresh_evaluation_opened"] is False
    assert value["effect_counts"] == d0.EFFECT_COUNTS
