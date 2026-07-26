from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_prepare_reference_repair_names_invalid_clue_ids_and_exact_allowlist(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    discovery = speaker_preprocessing_workflow.prepare_clue_discovery(
        transcript_path,
        document_id="doc-1",
        state_root=tmp_path / "state",
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )
    rejected_readout = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-99"],
                "observations": ["The speaker names an email."],
                "person_hints": [],
                "retrieval_terms": ["alice@example.com"],
            }
        ],
        "conversation_clues": [],
        "warnings": [],
    }

    repair = speaker_preprocessing_workflow.prepare_reference_repair(
        phase="clue_discovery",
        document_id="doc-1",
        document_title="Proposal review",
        state_root=tmp_path / "state",
        original_run_id=discovery["run_id"],
        original_packet=discovery["packet"],
        rejected_readout=rejected_readout,
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    assert repair["phase"] == "clue_discovery_reference_repair"
    assert repair["repair_packet"]["invalid_reference_fields"] == [
        {
            "path": "speaker_clues[0].transcript_clue_ids",
            "invalid_ids": ["utterance-99"],
            "allowed_ids": ["utterance-1"],
        }
    ]
    assert repair["repair_packet"]["rejected_readout"] == rejected_readout
    assert repair["repair_packet"]["original_run_id"] == discovery["run_id"]
    assert Path(repair["input_packet_path"]).exists()
    assert repair["will_send_prompt"] is False


def test_prepare_reference_repair_names_invalid_provenance_source_ids(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    packet = speaker_preprocessing_workflow.build_identity_evaluation_packet(
        transcript=json.loads(transcript_path.read_text(encoding="utf-8")),
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
        person_records=[],
        provenance_sources=[
            {
                "source_id": "mail-allowed",
                "source_type": "gws_mail_message",
                "label": "Allowed message",
                "snippet": "Prepared evidence.",
            }
        ],
    )
    rejected_readout = {
        "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
        "evaluation_id": packet["evaluation_id"],
        "calendar_association": {
            "status": "ambiguous",
            "factors": [
                {
                    "factor": "event_title_topic_alignment",
                    "direction": "support",
                    "strength": "weak",
                    "evidence_ids": ["utterance-1"],
                }
            ],
        },
        "person_links": [],
        "speaker_assignments": [
            {
                "speaker_labels": ["Speaker A"],
                "status": "unresolved",
                "person_id": "",
                "transcript_clue_ids": ["utterance-1"],
                "provenance_source_ids": ["mail-invented"],
                "factors": [
                    {
                        "factor": "topic_role_alignment",
                        "direction": "support",
                        "strength": "weak",
                        "evidence_ids": ["utterance-1"],
                    }
                ],
                "utterance_assignments": [],
                "review_flags": [],
            }
        ],
        "warnings": [],
    }

    repair = speaker_preprocessing_workflow.prepare_reference_repair(
        phase="identity_evaluation",
        document_id="doc-1",
        document_title="Proposal review",
        state_root=tmp_path / "state",
        original_run_id="identity-run-1",
        original_packet=packet,
        rejected_readout=rejected_readout,
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    assert repair["repair_packet"]["invalid_reference_fields"] == [
        {
            "path": "speaker_assignments[0].provenance_source_ids",
            "invalid_ids": ["mail-invented"],
            "allowed_ids": ["mail-allowed"],
        }
    ]


def test_prepare_reference_repair_names_invalid_utterance_evidence_id(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    packet = speaker_preprocessing_workflow.build_identity_evaluation_packet(
        transcript=json.loads(transcript_path.read_text(encoding="utf-8")),
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
    )
    rejected_readout = {
        "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
        "evaluation_id": packet["evaluation_id"],
        "calendar_association": {
            "status": "ambiguous",
            "factors": [
                {
                    "factor": "event_title_topic_alignment",
                    "direction": "support",
                    "strength": "weak",
                    "evidence_ids": ["utterance-1"],
                }
            ],
        },
        "person_links": [],
        "speaker_assignments": [
            {
                "speaker_labels": ["Speaker A"],
                "status": "unresolved",
                "person_id": "",
                "transcript_clue_ids": ["utterance-1"],
                "provenance_source_ids": [],
                "factors": [
                    {
                        "factor": "topic_role_alignment",
                        "direction": "support",
                        "strength": "weak",
                        "evidence_ids": ["utterance-1"],
                    }
                ],
                "utterance_assignments": [
                    {"utterance_id": "utterance-99", "person_id": ""}
                ],
                "review_flags": [],
            }
        ],
        "warnings": [],
    }

    repair = speaker_preprocessing_workflow.prepare_reference_repair(
        phase="identity_evaluation",
        document_id="doc-1",
        document_title="Proposal review",
        state_root=tmp_path / "state",
        original_run_id="identity-run-1",
        original_packet=packet,
        rejected_readout=rejected_readout,
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    assert repair["repair_packet"]["invalid_reference_fields"] == [
        {
            "path": (
                "speaker_assignments[0].utterance_assignments[0].utterance_id"
            ),
            "invalid_ids": ["utterance-99"],
            "allowed_ids": ["utterance-1"],
        }
    ]


def test_prepare_reference_repair_rejects_valid_first_pass_output(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    discovery = speaker_preprocessing_workflow.prepare_clue_discovery(
        transcript_path,
        document_id="doc-1",
        state_root=tmp_path / "state",
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )
    valid_readout = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-1"],
                "observations": [],
                "person_hints": [],
                "retrieval_terms": [],
            }
        ],
        "conversation_clues": [],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="does not require reference repair"):
        speaker_preprocessing_workflow.prepare_reference_repair(
            phase="clue_discovery",
            document_id="doc-1",
            document_title="Proposal review",
            state_root=tmp_path / "state",
            original_run_id=discovery["run_id"],
            original_packet=discovery["packet"],
            rejected_readout=valid_readout,
            route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
        )


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
