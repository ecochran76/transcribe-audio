from __future__ import annotations

from copy import deepcopy

import speaker_identity_plan0064_development_replay as development


def _transition():
    return {
        "canonical_people": [
            {
                "person_id": "person-1",
                "primary_name": "Person One",
                "external_identities": [
                    {"kind": "email", "value": "one@example.test"}
                ],
            },
            {
                "person_id": "person-2",
                "primary_name": "Person Two",
                "external_identities": [],
            },
            {
                "person_id": "person-3",
                "primary_name": "Person Three",
                "external_identities": [],
            },
        ],
        "slot_bindings": [
            {
                "slot_id": f"document-1::SPEAKER_{index}",
                "person_id": f"person-{index}",
            }
            for index in range(1, 4)
        ],
    }


def test_context_projection_deduplicates_affinities_and_preserves_conflict():
    outcomes = [
        {
            "speaker_ref": "SPEAKER_1",
            "source_speaker_label": "SPEAKER_1",
            "context_status": "unlisted",
            "suggestions": [
                {"name": "Person One", "email": "one@example.test"},
                {"name": "Person One", "email": "ONE@example.test"},
            ],
            "context_evidence_ids": ["clue-1", "provider-1"],
        },
        {
            "speaker_ref": "SPEAKER_2",
            "source_speaker_label": "SPEAKER_2",
            "context_status": "conflicting",
            "suggestions": [{"name": "Person Two", "email": ""}],
            "context_evidence_ids": ["clue-2", "provider-2"],
        },
        {
            "speaker_ref": "SPEAKER_3",
            "source_speaker_label": "SPEAKER_3",
            "context_status": "unresolved",
            "suggestions": [{"name": "Unknown", "email": ""}],
            "context_evidence_ids": ["clue-3", "provider-3"],
        },
    ]
    p3 = {
        "results": [
            {
                "document_id": "document-1",
                "join": {
                    "context_bundle": {
                        "lineage": [
                            {
                                "evidence_id": f"clue-{index}",
                                "source_type": "transcript_clue",
                            }
                            for index in range(1, 4)
                        ]
                        + [
                            {
                                "evidence_id": f"provider-{index}",
                                "source_type": "provider_evidence",
                            }
                            for index in range(1, 4)
                        ],
                        "source_failures": [
                            ["optional-provider", "out_of_scope_provider_result", False]
                        ],
                    },
                    "review_outcomes": outcomes,
                },
            }
        ]
    }
    case = development.project_context_cases(p3, _transition())[0]
    assert case["provider_failures"] == []
    assert case["speaker_slots"][0]["disposition"] == "candidate"
    assert case["speaker_slots"][0]["candidate_person_id"] == "person-1"
    proposal = case["speaker_slots"][0]["candidates"][0]
    assert proposal["transcript_clue_ids"] == ["clue-1"]
    assert proposal["provenance_source_ids"] == ["provider-1"]
    assert case["speaker_slots"][1]["reason_code"] == "material_context_conflict"
    assert case["speaker_slots"][1]["candidate_person_id"] is None
    assert case["speaker_slots"][2]["disposition"] == "abstain"


def _condition(person_id: str | None, reason: str):
    return {
        "disposition": "candidate" if person_id else "review",
        "reason_code": reason,
        "candidate_person_id": person_id,
    }


def test_development_gate_requires_correct_nonvacuous_residual_outcome():
    slots = []
    for index in range(1, 4):
        person_id = f"person-{index}"
        combined = (
            _condition(person_id, "pillar_agreement")
            if index < 3
            else _condition(None, "context_only_support")
        )
        residual = (
            deepcopy(combined)
            if index < 3
            else _condition(
                person_id,
                "two_known_plus_one_independently_supported_residual",
            )
        )
        slots.append(
            {
                "speaker_ref": f"document-1::SPEAKER_{index}",
                "combined": combined,
                "residual_policy": residual,
            }
        )
    gate = development.build_development_gate(
        [{"document_id": "document-1", "speaker_slots": slots}], _transition()
    )
    assert gate["combined_correct_count"] == 2
    assert gate["residual_correct_count"] == 1
    assert gate["high_support_wrong_count"] == 0
    assert gate["quality_gate_passed"] is True
    assert not any(gate["action_counts"].values())

    wrong = deepcopy(slots)
    wrong[2]["residual_policy"]["candidate_person_id"] = "person-1"
    failed = development.build_development_gate(
        [{"document_id": "document-1", "speaker_slots": wrong}], _transition()
    )
    assert failed["high_support_wrong_count"] == 1
    assert failed["quality_gate_passed"] is False
