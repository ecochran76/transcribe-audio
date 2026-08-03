from __future__ import annotations

from acoustic_generation4_context_contract import build_generation4_context_contract


def test_contract_freezes_paired_predictions_and_visible_acoustic_evidence() -> None:
    contract = build_generation4_context_contract()

    assert contract["status"] == "g1c_context_contract_complete"
    assert contract["prediction_families"] == [
        "context_only",
        "context_plus_separately_visible_acoustic",
    ]
    assert contract["acoustic_evidence_policy"] == {
        "visibility": "separate_cited_factor",
        "missing_evidence_effect": "neutral",
        "may_remove_context_candidate": False,
        "may_hide_conflict": False,
        "opaque_fusion_score_allowed": False,
    }
    assert contract["action_vector"]["submit_g1c_to_j1"] is True
    assert contract["action_vector"]["send_model_turn"] is False
    assert contract["action_vector"]["reveal_gold"] is False
    assert contract["action_vector"]["apply_assignments"] is False
    assert contract["contains_private_membership"] is False
    assert contract["contains_transcript_text"] is False


def test_contract_freezes_temporal_candidate_and_output_comparison_rules() -> None:
    contract = build_generation4_context_contract()

    assert contract["temporal_policy"]["as_of_field"] == "recording_start"
    assert contract["temporal_policy"]["post_as_of_evidence"] == "excluded"
    assert contract["candidate_policy"]["context_only_pool"] == "context_candidates"
    assert contract["candidate_policy"]["augmented_pool"] == (
        "stable_union_context_first_then_acoustic"
    )
    assert contract["candidate_policy"]["measurements"] == [
        "context_candidate_recall",
        "union_candidate_recall",
        "assignment_correctness",
    ]
    assert contract["comparison_policy"]["same_prompt_hash"] is True
    assert contract["comparison_policy"]["same_rubric_hash"] is True
    assert contract["comparison_policy"]["predictions_before_gold_reveal"] is True
    assert len(contract["prompt_sha256"]) == 64
    assert len(contract["rubric_sha256"]) == 64
    assert contract["output_schema"]["allowed_statuses"] == [
        "candidate_match",
        "unlisted",
        "unresolved",
        "conflicting",
    ]
