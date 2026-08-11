from __future__ import annotations

import speaker_identity_plan0071_d1 as d1


def _prediction(disposition: str, person_id: str | None, reason: str) -> dict[str, object]:
    return {
        "disposition": disposition,
        "candidate_person_id": person_id,
        "reason_code": reason,
    }


def test_measurement_counts_wrong_context_candidate_as_safely_unaccepted() -> None:
    recordings = [
        {
            "speaker_slots": [
                {
                    "speaker_ref": "doc::A",
                    "acoustic": _prediction("abstain", None, "none"),
                    "context": _prediction("candidate", "wrong", "context"),
                    "combined": _prediction("review", None, "context_only_support"),
                    "residual_policy": _prediction(
                        "review", None, "context_only_support"
                    ),
                },
                {
                    "speaker_ref": "doc::B",
                    "acoustic": _prediction("candidate", "right", "acoustic"),
                    "context": _prediction("candidate", "right", "context"),
                    "combined": _prediction(
                        "candidate", "right", "pillar_agreement"
                    ),
                    "residual_policy": _prediction(
                        "candidate", "right", "pillar_agreement"
                    ),
                },
            ]
        }
    ]
    gold = {
        "decisions": [
            {"speaker_ref": "doc::A", "decision": "not_listed", "person_id": None},
            {
                "speaker_ref": "doc::B",
                "decision": "canonical_person",
                "person_id": "right",
            },
        ]
    }

    measured = d1.measure_resolutions(recordings, gold)

    assert measured["condition_counts"]["context"]["wrong_candidate_count"] == 1
    assert measured["condition_counts"]["combined"]["wrong_candidate_count"] == 0
    assert measured["wrong_context_candidate_safely_unaccepted_count"] == 1
    assert measured["correct_pillar_agreement_count"] == 1
    assert measured["actual_residual_acceptance_count"] == 0


def test_filename_map_requires_one_original_filename_per_recording() -> None:
    review = {
        "cases": [
            {
                "document_id": f"doc-{index}",
                "recording_filename": f"original-{index}.m4a",
            }
            for index in range(12)
        ]
    }

    filenames = d1._filename_map(review)

    assert len(filenames) == 12
    assert filenames["doc-0"] == "original-0.m4a"
