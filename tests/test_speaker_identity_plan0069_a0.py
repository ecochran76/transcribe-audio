from __future__ import annotations

import pytest

import speaker_identity_plan0069_a0 as a0


def test_grouped_inventory_binds_exact_plural_object() -> None:
    grouped = {
        "utterance_ids": ["utterance-2", "utterance-7"],
        "status": "candidate_match",
        "person_id": "person-1",
        "rationale": "same speaker",
    }
    readout = {
        "speaker_assignments": [
            {"utterance_assignments": [grouped, {"utterance_id": "utterance-9"}]}
        ]
    }

    inventory = a0.grouped_assignment_inventory(readout)

    assert inventory == [
        {
            "path": "speaker_assignments[0].utterance_assignments[0]",
            "utterance_ids": ["utterance-2", "utterance-7"],
            "grouped_object_sha256": a0._hash(grouped),
        }
    ]
    assert grouped["utterance_ids"] == ["utterance-2", "utterance-7"]


@pytest.mark.parametrize(
    "grouped,reason",
    [
        ({"utterance_ids": []}, "Ambiguous"),
        ({"utterance_ids": ["utterance-2", "utterance-2"]}, "Ambiguous"),
        ({"utterance_ids": [""]}, "Ambiguous"),
        (
            {"utterance_id": "utterance-2", "utterance_ids": ["utterance-2"]},
            "Mixed singular/plural",
        ),
    ],
)
def test_grouped_inventory_rejects_ambiguous_shapes(
    grouped: dict[str, object], reason: str
) -> None:
    readout = {
        "speaker_assignments": [{"utterance_assignments": [grouped]}]
    }

    with pytest.raises(a0.Plan0069A0Error, match=reason):
        a0.grouped_assignment_inventory(readout)
