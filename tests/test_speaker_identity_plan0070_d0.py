from __future__ import annotations

import speaker_identity_plan0070_d0 as d0


def test_expected_d3_start_is_nonvacuous_and_residual_open() -> None:
    assert d0.EXPECTED_D3_START == {
        "recording_count": 12,
        "speaker_slot_count": 39,
        "correct_acoustic_candidate_count": 10,
        "wrong_acoustic_candidate_count": 0,
        "correct_context_candidate_count": 5,
        "wrong_context_candidate_count": 1,
        "correct_pillar_agreement_count": 5,
        "wrong_combined_candidate_count": 0,
        "residual_acceptance_count": 0,
    }
    assert d0.EXPECTED_D3_START["correct_pillar_agreement_count"] > 0
    assert d0.EXPECTED_D3_START["residual_acceptance_count"] == 0


def test_zero_effect_contract_is_literal() -> None:
    assert d0._all_zero(d0.EFFECT_COUNTS)
    assert set(d0.EFFECT_COUNTS) == {
        "biometric_writes",
        "external_writes",
        "graphiti_writes",
        "identity_writes",
        "knowledge_writes",
        "model_turns",
        "provider_writes",
        "source_transcript_writes",
        "speaker_assignment_writes",
        "stored_transcript_writes",
        "transcript_index_writes",
    }
