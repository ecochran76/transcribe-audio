from __future__ import annotations

import pytest

import speaker_identity_plan0068_a2 as a2


def test_repair_packet_adds_only_exact_calendar_catalog() -> None:
    packet = {"evaluation_id": "evaluation-1", "people": []}
    catalog = [
        {
            "evidence_id": "calendar-title-aabbcc",
            "evidence_type": "title",
            "event_id": "event-1",
            "identity_use": "candidate_only",
        }
    ]

    repaired = a2.repair_packet(packet, catalog)

    assert packet == {"evaluation_id": "evaluation-1", "people": []}
    assert repaired == {**packet, "calendar_evidence": catalog}


def test_repair_packet_rejects_existing_or_non_candidate_calendar_catalog() -> None:
    with pytest.raises(a2.Plan0068A2Error, match="already contains"):
        a2.repair_packet({"calendar_evidence": []}, [])

    with pytest.raises(a2.Plan0068A2Error, match="catalog is invalid"):
        a2.repair_packet(
            {"evaluation_id": "evaluation-1"},
            [
                {
                    "evidence_id": "calendar-title-aabbcc",
                    "identity_use": "speaker_binding",
                }
            ],
        )
