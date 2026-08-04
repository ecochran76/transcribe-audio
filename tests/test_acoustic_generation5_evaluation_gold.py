import hashlib
from pathlib import Path

import pytest

import acoustic_generation5_evaluation_gold as gold


REPO = {
    "commit": "1" * 40,
    "module_sha256": hashlib.sha256(b"module").hexdigest(),
    "clean": True,
    "upstream_ahead": 0,
    "upstream_behind": 0,
}


def _case(ordinal, people):
    source = hashlib.sha256(f"source-{ordinal}".encode()).hexdigest()
    transcript = hashlib.sha256(f"transcript-{ordinal}".encode()).hexdigest()
    case_id = "g5-case-" + gold._canonical_hash({"source": source, "transcript": transcript})[:20]
    return {
        "case_id": case_id,
        "enumerated_ordinal": ordinal,
        "source_sha256": source,
        "transcript_sha256": transcript,
        "conversation_id": gold._conversation_id(case_id),
        "recording_id": gold._recording_id(source),
        "speaker_gold": [
            {
                "speaker_label": chr(65 + index),
                "person_id": person,
                "enrolled_subject_id": enrolled,
            }
            for index, (person, enrolled) in enumerate(people)
        ],
        "overlap_codes": [],
    }


def test_answer_parser_requires_complete_stable_labels():
    refs = ["Candidate 1 / Speaker A", "Candidate 1 / Speaker B"]
    assert gold.parse_review_answers(
        "Candidate 1 / Speaker A = Eric\nCandidate 1 / Speaker B = Chris", expected_refs=refs
    ) == {refs[0]: "Eric", refs[1]: "Chris"}
    with pytest.raises(gold.Generation5EvaluationGoldError, match="incomplete"):
        gold.parse_review_answers("Candidate 1 / Speaker A = Eric", expected_refs=refs)
    with pytest.raises(gold.Generation5EvaluationGoldError, match="stable identity"):
        gold.parse_review_answers(
            "Candidate 1 / Speaker A = Eric\nCandidate 1 / Speaker B = UNKNOWN", expected_refs=refs
        )


def test_selects_lexicographically_first_passing_exact_seven():
    cases = [
        _case(1, [("enrolled-a", "subject-a"), ("person-3", "")]),
        _case(2, [("enrolled-a", "subject-a"), ("person-4", "")]),
        _case(3, [("person-3", ""), ("person-5", "")]),
        _case(4, [("person-4", ""), ("person-6", "")]),
        _case(5, [("person-5", ""), ("person-7", "")]),
        _case(6, [("person-6", ""), ("person-8", "")]),
        _case(7, [("enrolled-b", "subject-b"), ("person-7", "")]),
        _case(8, [("enrolled-b", "subject-b"), ("person-8", "")]),
    ]
    expected = {case["source_sha256"] for case in cases}
    selected, population, checked = gold.select_first_passing_seven(cases, expected_sources=expected)
    assert [case["enumerated_ordinal"] for case in selected] == [1, 2, 3, 4, 5, 7, 8]
    assert checked == 3
    assert population["passing"] is True
    assert population["conversation_count"] == 7


def test_generated_proposal_recomputes_against_exact_e1_membership(monkeypatch):
    identities = [
        [("Eric", "subject-a"), ("Person 3", "")],
        [("Eric", "subject-a"), ("Person 4", "")],
        [("Person 3", ""), ("Person 5", "")],
        [("Person 4", ""), ("Person 6", "")],
        [("Person 5", ""), ("Person 7", "")],
        [("Person 6", ""), ("Person 8", "")],
        [("Chris", "subject-b"), ("Person 7", "")],
        [("Chris", "subject-b"), ("Person 8", "")],
    ]
    membership = []
    cards = []
    answer_lines = []
    for ordinal, people in enumerate(identities, start=1):
        source = hashlib.sha256(f"source-{ordinal}".encode()).hexdigest()
        transcript = hashlib.sha256(f"transcript-{ordinal}".encode()).hexdigest()
        case_id = "g5-case-" + gold._canonical_hash({"source": source, "transcript": transcript})[:20]
        membership.append(
            {
                "enumerated_ordinal": ordinal,
                "source_sha256": source,
                "transcript_sha256": transcript,
            }
        )
        for index, (identity, _) in enumerate(people):
            label = chr(65 + index)
            reference = f"Candidate {ordinal} / Speaker {label}"
            cards.append(
                {
                    "case_id": case_id,
                    "speaker_ref": reference,
                    "speaker_label": label,
                    "source_sha256": source,
                    "transcript_sha256": transcript,
                }
            )
            answer_lines.append(f"{reference} = {identity}")
    authority = {
        "content_sha256": gold.E1_PREVIEW_SHA256,
        "speaker_label_count": len(cards),
        "private_evidence": {"candidate_membership": membership, "cards": cards},
    }
    enrolled = {"eric": "subject-a", "chris": "subject-b"}
    preview = gold.preview_generation5_evaluation_gold(
        "\n".join(answer_lines),
        e1_preview=authority,
        enrolled_identity_map=enrolled,
        repository_authority=REPO,
    )
    monkeypatch.setattr(gold, "_e1_preview", lambda: authority)
    monkeypatch.setattr(gold, "_enrolled_identity_map", lambda: enrolled)
    assert gold._validated_proposal(preview, require_current_repository=False) == preview
    assert preview["population_feasible"] is True
    assert preview["combinations_checked"] == 3


def test_private_proposal_apply_and_replay(tmp_path, monkeypatch):
    preview = {
        "schema_version": gold.PREVIEW_SCHEMA,
        "status": "ready_for_independent_j3_review",
        "repository_authority": REPO,
        "answer_set_sha256": "a" * 64,
        "reviewed_candidate_count": 7,
        "reviewed_speaker_label_count": 14,
        "combination_size": 7,
        "combinations_checked": 1,
        "population_feasible": True,
        "population_result": {"passing": True},
        "selected_case_ids_sha256": "b" * 64,
        "selected_source_set_sha256": "c" * 64,
        "selected_transcript_set_sha256": "d" * 64,
        "action_vector": {"submit_exact_cohort_and_gold_feasibility_to_j3": True},
        "private_gold": {"all_cases": [], "selected_cases": [], "selected_case_ids": []},
        "did_freeze_cohort_or_gold": False,
        "did_load_or_run_models": False,
        "did_reveal_gold_to_workers": False,
    }
    preview["content_sha256"] = gold._canonical_hash(preview)
    monkeypatch.setattr(
        gold,
        "_validated_proposal",
        lambda value, require_current_repository: dict(value),
    )
    monkeypatch.setattr(
        gold,
        "_git",
        lambda arguments, binary=False: b"module" if arguments[0] == "show" else "",
    )
    applied = gold.apply_generation5_evaluation_gold(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path,
    )
    replayed = gold.replay_generation5_evaluation_gold(preview["content_sha256"], runtime_root=tmp_path)
    assert applied["idempotent_replay"] is False
    assert replayed["idempotent_replay"] is True
    assert gold._paths(tmp_path, preview["content_sha256"])["manifest"].stat().st_mode & 0o777 == 0o600
