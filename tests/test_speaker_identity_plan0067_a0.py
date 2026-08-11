from __future__ import annotations

import json

import pytest

import speaker_identity_plan0067_a0 as a0


def _case_values() -> dict[str, object]:
    packet = {"evaluation_id": "evaluation-1"}
    readout = {
        "evaluation_id": "evaluation-1",
        "calendar_association": {
            "factors": [
                {
                    "factor": "title_alignment",
                    "evidence_ids": ["calendar-title-aabbcc"],
                }
            ]
        },
    }
    return {
        "document_id": "document-1",
        "a1_case": {
            "packet": packet,
            "original_recording_filename": "Original recording.transcript.json",
        },
        "prepared": {
            "document_id": "document-1",
            "run_id": "run-1",
            "packet": packet,
            "packet_sha256": a0._hash(packet),
            "original_recording_filename": "Original recording.transcript.json",
        },
        "failed_case": {
            "document_id": "document-1",
            "reason": (
                "calendar_association factor references unprepared evidence: "
                "['calendar-title-aabbcc']."
            ),
        },
        "status": {
            "run_id": "run-1",
            "completed": True,
            "output_text": json.dumps(readout),
            "codex_thread_id": "thread-1",
            "codex_turn_id": "turn-1",
        },
        "calendar_evidence": [
            {
                "evidence_id": "calendar-title-aabbcc",
                "evidence_type": "title",
                "event_id": "event-1",
                "identity_use": "candidate_only",
            }
        ],
    }


def test_case_binding_preserves_filename_and_proves_rejected_id_is_catalogued() -> None:
    binding = a0.build_case_binding(**_case_values())

    assert binding["original_recording_filename"] == "Original recording.transcript.json"
    assert binding["plan0066_rejected_calendar_evidence_ids"] == [
        "calendar-title-aabbcc"
    ]
    assert binding["retained_calendar_evidence_ids"] == ["calendar-title-aabbcc"]
    assert binding["output_text_sha256"]


def test_case_binding_rejects_calendar_id_absent_from_host_catalog() -> None:
    values = _case_values()
    values["calendar_evidence"] = []

    with pytest.raises(a0.Plan0067A0Error, match="outside the host catalog"):
        a0.build_case_binding(**values)
