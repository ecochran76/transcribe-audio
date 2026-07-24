from __future__ import annotations

import json
from pathlib import Path

import speaker_preprocessing_workflow


def _transcript(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "conversation_id": "1934616f-3cb5-4f70-b037-679d863d4847",
                "recording_id": "2dbdfe40-71e6-4597-ab77-5d56613d0e65",
                "transcript_title": "Proposal review",
                "utterances": [
                    {
                        "speaker": "Speaker A",
                        "start": 0,
                        "end": 4,
                        "text": "I sent it from alice@example.com.",
                    }
                ],
                "event": {
                    "id": "event-1",
                    "summary": "Proposal review",
                    "attendees": [{"email": "alice@example.com"}],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_clue_discovery_creates_reviewed_unsent_app_packet(tmp_path: Path) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")

    prepared = speaker_preprocessing_workflow.prepare_clue_discovery(
        transcript_path,
        document_id="doc-1",
        state_root=tmp_path / "state",
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    assert prepared["phase"] == "clue_discovery"
    assert prepared["will_send_prompt"] is False
    assert Path(prepared["packet_path"]).exists()
    assert prepared["packet"]["task"] == "speaker_clue_discovery"
    assert prepared["route"]["provider"] == "codex-app-server"
    assert prepared["route"]["model"] == "gpt-5.6-sol"


def test_prepare_identity_evaluation_requires_validated_discovery_first(tmp_path: Path) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    discovery = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-1"],
                "observations": ["The speaker names an email."],
                "person_hints": [{"name": "Alice", "email": "alice@example.com"}],
                "retrieval_terms": ["alice@example.com"],
            }
        ],
        "conversation_clues": [],
        "warnings": [],
    }

    prepared = speaker_preprocessing_workflow.prepare_identity_evaluation(
        transcript_path,
        document_id="doc-1",
        state_root=tmp_path / "state",
        discovery_readout=discovery,
        person_records=[
            {
                "contact_id": "contact-alice",
                "label": "Alice",
                "email": "alice@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            }
        ],
        provenance_sources=[],
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    assert prepared["phase"] == "identity_evaluation"
    assert prepared["will_send_prompt"] is False
    assert prepared["packet"]["task"] == "speaker_identity_evaluation"
    assert prepared["packet"]["discovery_readout"] == discovery


def test_persist_identity_evaluation_keeps_factors_scores_and_review_gate(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    discovery = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [],
        "conversation_clues": [],
        "warnings": [],
    }
    packet = speaker_preprocessing_workflow.build_identity_evaluation_packet(
        transcript=json.loads(transcript_path.read_text(encoding="utf-8")),
        discovery_readout=discovery,
        person_records=[
            {
                "contact_id": "contact-alice",
                "label": "Alice",
                "email": "alice@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            }
        ],
    )
    person_id = packet["people"][0]["person_id"]
    readout = {
        "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
        "evaluation_id": packet["evaluation_id"],
        "calendar_association": {
            "status": "matched",
            "factors": [
                {
                    "factor": "event_title_topic_alignment",
                    "direction": "support",
                    "strength": "strong",
                    "evidence_ids": ["event-1", "utterance-1"],
                }
            ],
        },
        "person_links": [],
        "speaker_assignments": [
            {
                "speaker_labels": ["Speaker A"],
                "status": "candidate_match",
                "person_id": person_id,
                "factors": [
                    {
                        "factor": "direct_self_identification",
                        "direction": "support",
                        "strength": "decisive",
                        "evidence_ids": ["utterance-1"],
                    },
                    {
                        "factor": "verified_identifier_match",
                        "direction": "support",
                        "strength": "moderate",
                        "evidence_ids": ["contact-alice"],
                    },
                ],
                "utterance_assignments": [],
                "review_flags": [],
            }
        ],
        "warnings": [],
    }

    record = speaker_preprocessing_workflow.persist_identity_evaluation(
        transcript_path,
        packet=packet,
        readout=readout,
        run_references={"clue_discovery_run_id": "run-1", "identity_evaluation_run_id": "run-2"},
    )

    evaluation = record["evaluations"][0]
    assert evaluation["proposals"][0]["confidence"]["band"] == "very_high"
    assert evaluation["proposals"][0]["proposal_id"]
    assert evaluation["review_state"]["pending_count"] == 1
    assert evaluation["safe_bulk_confirm_ready"] is True
    assert evaluation["rubric_versions"]["speaker_identity"] == "speaker-identity.v1"

    confirmed = speaker_preprocessing_workflow.confirm_ready_proposals(
        transcript_path,
        evaluation_id=evaluation["evaluation_id"],
        reviewer="Eric Cochran",
    )

    assert confirmed["confirmed_proposal_ids"] == [
        evaluation["proposals"][0]["proposal_id"]
    ]
    assert confirmed["record"]["review_decisions"][0]["decision_method"] == (
        "conversation_bulk_ready"
    )
