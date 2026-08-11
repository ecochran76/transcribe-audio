from __future__ import annotations

import pytest

import speaker_identity_plan0065_d2 as d2


def _factor(direction="neutral", evidence_ids=None):
    return {
        "factor": "time_window_alignment",
        "direction": direction,
        "strength": "weak",
        "evidence_ids": list(evidence_ids or []),
    }


def _readout():
    return {
        "calendar_association": {
            "status": "ambiguous",
            "factors": [_factor(), _factor("support", ["evidence-1"])],
        },
        "person_links": [],
        "speaker_assignments": [
            {
                "speaker_labels": ["A"],
                "status": "unresolved",
                "factors": [_factor()],
            }
        ],
    }


def test_uncited_neutral_factors_are_dropped_without_changing_supported_factors():
    normalized, audit = d2.neutralize_uncited_factors(_readout())

    assert normalized["calendar_association"]["factors"] == [
        _factor("support", ["evidence-1"])
    ]
    assert normalized["speaker_assignments"][0]["factors"] == []
    assert audit["neutralized_factor_count"] == 2


def test_uncited_support_or_contradiction_remains_a_validation_failure():
    readout = _readout()
    readout["calendar_association"]["factors"] = [_factor("support")]

    with pytest.raises(d2.Plan0065D2Error, match="Uncited non-neutral"):
        d2.neutralize_uncited_factors(readout)


def test_exhaustive_discovery_is_clue_only_and_cites_every_prepared_utterance():
    packet = {
        "speakers": [
            {
                "speaker_label": "A",
                "utterance_clues": [
                    {"utterance_id": "utterance-1"},
                    {"utterance_id": "utterance-2"},
                ],
            }
        ]
    }

    readout = d2.build_exhaustive_clue_only_discovery(packet)

    assert readout["speaker_clues"] == [
        {
            "speaker_label": "A",
            "transcript_clue_ids": ["utterance-1", "utterance-2"],
            "calendar_clue_ids": [],
            "observations": [],
            "person_hints": [],
            "retrieval_terms": [],
        }
    ]
    assert readout["policy"]["identify_people_in_this_pass"] is False


def test_context_gate_is_nonvacuous_and_rejects_wrong_or_incomplete_candidates():
    passing = [
        {
            "disposition": "candidate",
            "gold": "correct",
            "candidate_provenance_complete": True,
        }
    ] + [
        {
            "disposition": "abstain",
            "gold": "other",
            "candidate_provenance_complete": True,
        }
        for _ in range(38)
    ]

    assert d2.evaluate_context_gate(passing)["passed"] is True
    assert d2.evaluate_context_gate(
        [{**passing[0], "gold": "wrong"}, *passing[1:]]
    )["passed"] is False
    assert d2.evaluate_context_gate(
        [{**passing[0], "candidate_provenance_complete": False}, *passing[1:]]
    )["passed"] is False
