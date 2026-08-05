from copy import deepcopy

import acoustic_generation5_source_j1_acceptance as j1


def _proposal():
    cases = []
    for ordinal in range(1, 13):
        cases.append({
            "case_id": f"g5s-case-{ordinal:020x}", "enumerated_ordinal": ordinal,
            "source_sha256": f"{ordinal:064x}", "transcript_sha256": f"{ordinal + 20:064x}",
            "conversation_id": f"conversation-{ordinal}", "recording_id": f"recording-{ordinal}",
            "speaker_gold": [{"person_id": f"person-{ordinal}", "enrolled_subject_id": ""}], "overlap_codes": [],
        })
    selected = cases[:7]
    population = {"passing": True}
    return {"private_gold": {"all_cases": cases, "selected_cases": selected,
                              "selected_case_ids": [case["case_id"] for case in selected]},
            "selected_case_ids_sha256": "a" * 64, "selected_source_set_sha256": "b" * 64,
            "selected_transcript_set_sha256": "c" * 64, "population_result": population}


def test_preview_contains_private_gold_but_keeps_models_and_reveal_false():
    preview = j1.preview_generation5_source_j1_acceptance(
        gold_proposal=_proposal(), repository_authority={"commit": "test"}
    )
    assert preview["contains_private_gold"] is True
    assert preview["action_vector"]["freeze_selected_cohort_and_private_gold"] is True
    assert preview["action_vector"]["run_models_or_predictions"] is False
    assert preview["action_vector"]["reveal_gold_to_workers"] is False


def test_preview_rejects_missing_selected_case():
    proposal = _proposal()
    proposal = deepcopy(proposal)
    proposal["private_gold"]["selected_cases"].pop()
    try:
        j1.preview_generation5_source_j1_acceptance(
            gold_proposal=proposal, repository_authority={"commit": "test"}
        )
    except j1.Generation5SourceJ1Error as exc:
        assert "Selected private gold" in str(exc)
    else:
        raise AssertionError("incomplete selected gold was accepted")
