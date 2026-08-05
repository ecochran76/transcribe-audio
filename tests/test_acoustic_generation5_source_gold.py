from copy import deepcopy

import acoustic_generation5_source_gold as gold


def _case(ordinal, people):
    source = f"{ordinal:064x}"
    return {
        "case_id": f"g5s-case-{ordinal:020x}", "enumerated_ordinal": ordinal,
        "source_sha256": source, "transcript_sha256": f"{ordinal + 20:064x}",
        "conversation_id": f"conversation-{ordinal}", "recording_id": f"recording-{ordinal}",
        "speaker_gold": [{"person_id": person, "enrolled_subject_id": subject}
                         for person, subject in people], "overlap_codes": [],
    }


def test_parse_normalizes_known_aliases():
    text = "A = Dr. Jeffrey Dikis\nB = Dr. Dikis Nurse\nC = Alexendra Hoen"
    answers = gold.parse_review_answers(text, ["A", "B", "C"])
    assert answers == {"A": "Jeffrey Dikis", "B": "Dr. Dikis' Nurse", "C": "Alexandra Hoen"}


def test_select_requires_first_two_and_uses_first_passing_combination():
    cases = []
    for ordinal in range(1, 13):
        people = [(f"person-{ordinal}", "")]
        if ordinal in {1, 2, 3, 4, 5, 6, 7}:
            people.append(("eric", "enrolled-eric"))
        if ordinal in {1, 2}:
            people.append(("chris", "enrolled-chris"))
        cases.append(_case(ordinal, people))
    selected, result, checked = gold.select_first_passing(cases, {case["source_sha256"] for case in cases})
    assert checked == 1
    assert [case["enumerated_ordinal"] for case in selected] == [1, 2, 3, 4, 5, 6, 7]
    assert result["passing"] is True


def test_population_rejects_duplicate_source():
    cases = [_case(ordinal, [(f"person-{ordinal}", "")]) for ordinal in range(1, 8)]
    cases[1] = deepcopy(cases[1])
    cases[1]["source_sha256"] = cases[0]["source_sha256"]
    result = gold.evaluate_population(cases, {case["source_sha256"] for case in cases})
    assert result["gates"]["zero_overlap"] is False
