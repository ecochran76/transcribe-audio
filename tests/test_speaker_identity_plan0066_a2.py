from __future__ import annotations

import speaker_identity_plan0066_a2 as a2


def test_a2_packet_retains_prior_provenance_and_adds_roster() -> None:
    prior = {
        "provenance_sources": [{"source_id": "old-evidence", "snippet": "old"}],
        "source_contexts": [{"source_id": "old-scope"}],
        "retrieval": {"allowlists": {"person_ids": []}},
    }
    current = {
        "people": [
            {"person_id": f"person-{index}", "display_name": f"Person {index}"}
            for index in range(6)
        ],
        "provenance_sources": [{"source_id": "new-evidence"}],
        "source_contexts": [
            {"source_id": "reviewed", "relationship_scope": "reviewed_identity"}
        ],
    }

    packet = a2.build_a2_packet(current, prior)

    assert packet["provenance_sources"] == prior["provenance_sources"]
    assert len(packet["people"]) == 6
    assert packet["retrieval"]["allowlists"]["person_ids"] == [
        f"person-{index}" for index in range(6)
    ]


def test_measurement_requires_nonvacuous_correct_and_zero_wrong() -> None:
    cases = [
        {
            "status": "model_readout_validated",
            "document_id": "272cfe27e462506228a4",
            "validated_readout": {
                "speaker_assignments": [
                    {
                        "speaker_labels": ["B"],
                        "status": "candidate_match",
                        "person_id": "person-1",
                        "provenance_source_ids": ["source-1"],
                        "factors": [{"factor": "verified_identifier_match"}],
                    }
                ]
            },
        }
    ]
    gold = {
        "decisions": [
            {
                "speaker_ref": "272cfe27e462506228a4::B",
                "decision": "canonical_person",
                "person_id": "person-1",
            }
        ]
    }

    result = a2.measure_cases(cases, gold)

    assert result["passed"] is True
    assert result["correct_prepared_candidate_count"] == 1
    assert result["wrong_prepared_candidate_count"] == 0
