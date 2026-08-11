from __future__ import annotations

import pytest

import speaker_identity_plan0069_a2 as a2


def test_normalization_must_equal_frozen_inventory() -> None:
    inventory = [
        {
            "path": "speaker_assignments[0].utterance_assignments[0]",
            "utterance_ids": ["utterance-1", "utterance-3"],
            "grouped_object_sha256": "a" * 64,
        }
    ]
    normalization = {
        "changes": [
            {
                "path": "speaker_assignments[0].utterance_assignments[0]",
                "utterance_ids": ["utterance-1", "utterance-3"],
                "expanded_count": 2,
            }
        ],
        "normalized_group_count": 1,
        "expanded_utterance_assignment_count": 2,
    }

    a2.assert_normalization_matches_inventory(inventory, normalization)


def test_normalization_rejects_path_or_order_drift() -> None:
    inventory = [
        {
            "path": "speaker_assignments[0].utterance_assignments[0]",
            "utterance_ids": ["utterance-1", "utterance-3"],
            "grouped_object_sha256": "a" * 64,
        }
    ]
    normalization = {
        "changes": [
            {
                "path": "speaker_assignments[0].utterance_assignments[0]",
                "utterance_ids": ["utterance-3", "utterance-1"],
                "expanded_count": 2,
            }
        ],
        "normalized_group_count": 1,
        "expanded_utterance_assignment_count": 2,
    }

    with pytest.raises(a2.Plan0069A2Error, match="differ from A0 inventory"):
        a2.assert_normalization_matches_inventory(inventory, normalization)
