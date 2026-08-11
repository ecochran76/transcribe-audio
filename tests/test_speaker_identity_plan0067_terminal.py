from __future__ import annotations

import speaker_identity_plan0067_a0 as a0
import speaker_identity_plan0067_terminal as terminal


def test_terminal_withholds_after_three_zero_effect_a0_attempts() -> None:
    value = terminal.build_terminal(source_commit="a" * 40)

    assert value["status"] == "plan0067_closed_withhold"
    assert value["reason_code"] == "a0_legacy_artifact_mode_contract_mismatch"
    assert value["a0_attempt_count"] == 3
    assert value["a0_artifact_written"] is False
    assert value["product_contract_changed"] is False
    assert value["retained_output_replay_count"] == 0
    assert value["effect_counts"] == a0.EFFECT_COUNTS
