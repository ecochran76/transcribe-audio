from __future__ import annotations

from pathlib import Path

import pytest

import speaker_identity_plan0066_a1 as a1


def _prepared(people: list[dict]) -> dict:
    return {
        "packet": {"people": people},
        "will_send_prompt": False,
        "retrieval": {
            "source_transcript_sha256": "a" * 64,
            "preparation_transcript_sha256": "b" * 64,
            "source_was_derived": True,
            "source_failures": [],
        },
        "transcript_artifact": {"path": "/private/original-recording.m4a.transcript.json"},
        "run_id": "run-1",
    }


def test_case_receipt_keeps_original_filename_and_exact_reviewed_roster() -> None:
    roster = {
        "00000000-0000-4000-8000-000000000001": "First Person",
        "00000000-0000-4000-8000-000000000002": "Second Person",
    }
    prepared = _prepared(
        [
            {"person_id": person_id, "display_name": display_name}
            for person_id, display_name in roster.items()
        ]
    )

    receipt = a1.build_case_receipt(
        document_id="doc-1",
        discovery_run_id="discovery-1",
        prepared=prepared,
        expected_roster=roster,
    )

    assert receipt["original_recording_filename"] == Path(
        prepared["transcript_artifact"]["path"]
    ).name
    assert receipt["reviewed_person_count"] == 2
    assert receipt["model_turn_count"] == 0


def test_case_receipt_rejects_incomplete_roster() -> None:
    with pytest.raises(a1.Plan0066A1Error, match="exact reviewed roster"):
        a1.build_case_receipt(
            document_id="doc-1",
            discovery_run_id="discovery-1",
            prepared=_prepared([]),
            expected_roster={
                "00000000-0000-4000-8000-000000000001": "First Person"
            },
        )
