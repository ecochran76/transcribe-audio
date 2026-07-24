from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import conversation_processing


def test_append_evaluation_creates_conversation_owned_processing_sidecar(tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting Transcript.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "03d687b2-122a-4a27-8118-f0b4789f48d3",
                "recording_id": "dbbec02f-91d8-48bf-b578-fd94fcb3290c",
            }
        ),
        encoding="utf-8",
    )

    record = conversation_processing.append_evaluation(
        transcript_path,
        {"phase": "clue_discovery", "status": "prepared"},
    )

    sidecar_path = tmp_path / "meeting Transcript.processing.json"
    assert sidecar_path.exists()
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted == record
    assert persisted["schema_version"] == "transcribe-audio.conversation-processing.v1"
    assert persisted["conversation_id"] == "03d687b2-122a-4a27-8118-f0b4789f48d3"
    assert persisted["recording_ids"] == ["dbbec02f-91d8-48bf-b578-fd94fcb3290c"]
    assert str(UUID(persisted["current_evaluation_id"])) == persisted["current_evaluation_id"]
    assert persisted["evaluations"] == [
        {
            "evaluation_id": persisted["current_evaluation_id"],
            "phase": "clue_discovery",
            "status": "prepared",
        }
    ]


def test_append_evaluation_preserves_history_and_advances_current_pointer(tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting Transcript.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "03d687b2-122a-4a27-8118-f0b4789f48d3",
                "recording_id": "dbbec02f-91d8-48bf-b578-fd94fcb3290c",
            }
        ),
        encoding="utf-8",
    )
    first_id = "82965f00-17ef-480d-b5fb-133a1b177413"
    second_id = "d25affbc-df39-4fb1-bfcb-44a7dd29b844"

    conversation_processing.append_evaluation(
        transcript_path,
        {"evaluation_id": first_id, "phase": "clue_discovery", "status": "captured"},
    )
    record = conversation_processing.append_evaluation(
        transcript_path,
        {"evaluation_id": second_id, "phase": "identity_evaluation", "status": "prepared"},
    )

    assert [item["evaluation_id"] for item in record["evaluations"]] == [first_id, second_id]
    assert record["current_evaluation_id"] == second_id


def test_append_evaluation_lazily_backfills_existing_transcript_ids(tmp_path: Path) -> None:
    transcript_path = tmp_path / "legacy Transcript.transcript.json"
    transcript_path.write_text(
        json.dumps({"schema_version": 1, "utterances": []}),
        encoding="utf-8",
    )

    payload = conversation_processing.append_evaluation(
        transcript_path,
        {"phase": "prepared"},
    )
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    assert str(UUID(transcript["conversation_id"])) == transcript["conversation_id"]
    assert str(UUID(transcript["recording_id"])) == transcript["recording_id"]
    assert payload["conversation_id"] == transcript["conversation_id"]


def test_review_decisions_are_attributable_and_append_only(tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting Transcript.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "03d687b2-122a-4a27-8118-f0b4789f48d3",
                "recording_id": "dbbec02f-91d8-48bf-b578-fd94fcb3290c",
            }
        ),
        encoding="utf-8",
    )
    evaluation_id = "4bb608cf-cf81-4c42-b00f-07fb25a9f41d"
    conversation_processing.append_evaluation(
        transcript_path,
        {
            "evaluation_id": evaluation_id,
            "proposals": [{"proposal_id": "proposal-a"}],
        },
    )

    first = conversation_processing.append_review_decision(
        transcript_path,
        evaluation_id=evaluation_id,
        proposal_id="proposal-a",
        action="defer",
        reviewer="Eric Cochran",
        method="individual",
        note="Need to replay the audio.",
    )
    first_decision = first["review_decisions"][0]
    second = conversation_processing.append_review_decision(
        transcript_path,
        evaluation_id=evaluation_id,
        proposal_id="proposal-a",
        action="confirm",
        reviewer="Eric Cochran",
        method="individual",
        supersedes_decision_id=first_decision["decision_id"],
        reviewer_asserted_identity={
            "name": "Alice Example",
            "email": "alice@example.com",
        },
    )

    assert len(second["review_decisions"]) == 2
    assert second["review_decisions"][1]["supersedes_decision_id"] == first_decision["decision_id"]
    assert second["review_decisions"][1]["reviewer_asserted_identity"]["name"] == "Alice Example"
    assert "confidence" not in second["review_decisions"][1]["reviewer_asserted_identity"]
