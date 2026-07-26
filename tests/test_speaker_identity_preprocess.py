from __future__ import annotations

import json
from pathlib import Path

import pytest

import speaker_identity_preprocess
from context_sources import GwsProvenanceConfig, OdolloProvenanceConfig
from routing_artifacts import ProvenanceSource


def _discovery_transcript() -> dict:
    return {
        "conversation_id": "1934616f-3cb5-4f70-b037-679d863d4847",
        "recording_id": "2dbdfe40-71e6-4597-ab77-5d56613d0e65",
        "transcript_title": "Proposal review",
        "utterances": [
            {
                "speaker": "Speaker A",
                "start": 0,
                "end": 4,
                "text": "I sent the revision from alice@example.com.",
            },
            {
                "speaker": "Speaker B",
                "start": 4,
                "end": 8,
                "text": "Thanks Alice. Example Co needs it Friday.",
            },
        ],
        "event": {
            "summary": "Proposal review",
            "id": "event-1",
            "attendees": [
                {"displayName": "Alice Example", "email": "alice@example.com"},
                {"displayName": "Bob Buyer", "email": "bob@example.com"},
            ],
        },
    }


def test_build_clue_discovery_packet_has_no_retrieved_people_or_sources() -> None:
    packet = speaker_identity_preprocess.build_clue_discovery_packet(
        transcript=_discovery_transcript(),
        source_contexts=[
            {
                "source_id": "gws-personal",
                "owner": {"type": "person", "id": "operator", "label": "Local operator"},
                "relationship_scope": "personal",
                "account_label": "Personal Google Workspace",
                "evidence_capabilities": ["calendar", "people", "gmail"],
                "authoritative_identifiers": ["email"],
            }
        ],
    )

    assert packet["schema_version"] == "transcribe-audio.speaker-clue-discovery-packet.v1"
    assert packet["task"] == "speaker_clue_discovery"
    assert packet["conversation"]["conversation_id"] == _discovery_transcript()["conversation_id"]
    assert packet["conversation"]["recording_ids"] == [_discovery_transcript()["recording_id"]]
    assert packet["speakers"][0]["utterance_clues"][0]["utterance_id"] == "utterance-1"
    assert packet["calendar_context"]["attendees"][0]["email"] == "alice@example.com"
    assert packet["source_contexts"][0]["relationship_scope"] == "personal"
    assert "contact_candidates" not in packet
    assert "provenance_sources" not in packet


def test_validate_clue_discovery_readout_requires_prepared_citations() -> None:
    packet = speaker_identity_preprocess.build_clue_discovery_packet(
        transcript=_discovery_transcript(),
    )
    readout = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-1"],
                "observations": ["The speaker claims the Alice email address."],
                "person_hints": [
                    {"name": "Alice Example", "email": "alice@example.com", "organization": ""}
                ],
                "retrieval_terms": ["alice@example.com", "Example Co"],
            }
        ],
        "conversation_clues": [
            {
                "transcript_clue_ids": ["utterance-2"],
                "observation": "The conversation concerns Example Co.",
                "retrieval_terms": ["Example Co"],
            }
        ],
        "warnings": [],
    }

    validated = speaker_identity_preprocess.validate_clue_discovery_readout(packet, readout)

    assert validated["valid"] is True
    assert validated["readout"]["speaker_clues"][0]["retrieval_terms"] == [
        "alice@example.com",
        "Example Co",
    ]

    readout["speaker_clues"][0]["transcript_clue_ids"] = ["utterance-99"]
    with pytest.raises(ValueError, match="unprepared transcript clues"):
        speaker_identity_preprocess.validate_clue_discovery_readout(packet, readout)


def test_group_person_candidates_merges_duplicate_people_but_preserves_source_records() -> None:
    people = speaker_identity_preprocess.group_person_candidates(
        [
            {
                "contact_id": "gws-alice",
                "label": "Alice Example",
                "email": "Alice@Example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            },
            {
                "contact_id": "odoo-alice",
                "label": "Alice E.",
                "email": "alice@example.com",
                "source_id": "odoo-soylei",
                "source_type": "odollo_contact",
            },
        ]
    )

    assert len(people) == 1
    assert people[0]["person_id"].startswith("person-")
    assert people[0]["emails"] == ["alice@example.com"]
    assert {item["source_id"] for item in people[0]["source_records"]} == {
        "gws-personal",
        "odoo-soylei",
    }
    assert {item["record_id"] for item in people[0]["source_records"]} == {
        "gws-alice",
        "odoo-alice",
    }


def test_build_identity_evaluation_packet_contains_only_host_retrieved_evidence() -> None:
    transcript = _discovery_transcript()
    discovery = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [
            {
                "speaker_label": "Speaker A",
                "transcript_clue_ids": ["utterance-1"],
                "observations": ["The speaker claims the Alice email address."],
                "person_hints": [{"name": "Alice Example", "email": "alice@example.com"}],
                "retrieval_terms": ["alice@example.com"],
            }
        ],
        "conversation_clues": [],
        "warnings": [],
    }
    packet = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=transcript,
        discovery_readout=discovery,
        person_records=[
            {
                "contact_id": "gws-alice",
                "label": "Alice Example",
                "email": "alice@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            }
        ],
        source_contexts=[
            {
                "source_id": "gws-personal",
                "owner": {"type": "person", "id": "operator", "label": "Operator"},
                "relationship_scope": "personal",
                "account_label": "Personal Google Workspace",
                "evidence_capabilities": ["calendar", "people", "gmail"],
                "authoritative_identifiers": ["email"],
            }
        ],
        provenance_sources=[
            {
                "source_type": "gws_mail_message",
                "source_id": "mail-1",
                "label": "Proposal thread",
                "snippet": "Alice sent the proposal.",
                "metadata": {"profile": "gws-personal", "email": "alice@example.com"},
            }
        ],
    )

    assert packet["schema_version"] == "transcribe-audio.speaker-identity-evaluation-packet.v1"
    assert packet["discovery_readout"] == discovery
    assert packet["people"][0]["emails"] == ["alice@example.com"]
    assert packet["provenance_sources"][0]["source_id"] == "mail-1"
    assert set(packet["rubrics"]) == {
        "calendar_association",
        "person_link",
        "speaker_identity",
    }
    assert packet["policy"]["model_must_not_emit_numeric_confidence"] is True


def test_score_evidence_factors_deduplicates_correlated_sources_and_adds_band() -> None:
    factors = [
        {
            "factor": "calendar_attendee_topic_alignment",
            "direction": "support",
            "strength": "strong",
            "independence_key": "calendar:event-1",
            "evidence_ids": ["event-1", "utterance-2"],
        },
        {
            "factor": "duplicate_calendar_copy",
            "direction": "support",
            "strength": "strong",
            "independence_key": "calendar:event-1",
            "evidence_ids": ["event-1-copy"],
        },
        {
            "factor": "direct_email_self_reference",
            "direction": "support",
            "strength": "decisive",
            "independence_key": "transcript:utterance-1",
            "evidence_ids": ["utterance-1"],
        },
    ]

    score = speaker_identity_preprocess.score_evidence_factors("speaker_identity", factors)

    assert score["rubric_version"] == "speaker-identity.v2"
    assert score["numeric"] == 100
    assert score["band"] == "very_high"
    assert score["counted_independence_keys"] == [
        "calendar:event-1",
        "transcript:utterance-1",
    ]


def test_calibrate_speaker_identity_confidence_caps_material_identity_risk() -> None:
    calibrated = speaker_identity_preprocess.calibrate_speaker_identity_confidence(
        {
            "status": "candidate_match",
            "review_flags": ["human_confirmation_required", "first_name_only"],
            "factors": [],
        },
        {
            "rubric": "speaker_identity",
            "rubric_version": "speaker-identity.v2",
            "numeric": 100,
            "band": "very_high",
            "band_label": "Very High",
        },
    )

    assert calibrated["numeric"] == 59
    assert calibrated["band"] == "medium"
    assert calibrated["band_label"] == "Medium"
    assert calibrated["uncapped_numeric"] == 100
    assert calibrated["uncapped_band"] == "very_high"
    assert calibrated["calibration"] == {
        "version": "speaker-confidence-calibration.v1",
        "applied": True,
        "cap_numeric": 59,
        "reasons": ["material_flag:unsafe_first_name_only"],
    }


@pytest.mark.parametrize(
    ("assignment", "expected_reason"),
    [
        (
            {"status": "unresolved", "review_flags": [], "factors": []},
            "assignment_status:unresolved",
        ),
        (
            {"status": "unlisted", "review_flags": [], "factors": []},
            "unlisted_without_prepared_person",
        ),
        (
            {
                "status": "candidate_match",
                "review_flags": [],
                "factors": [
                    {
                        "factor": "speaker_mixing_or_contradiction",
                        "direction": "contradict",
                        "strength": "strong",
                    }
                ],
            },
            "strong_speaker_mixing_contradiction",
        ),
    ],
)
def test_calibrate_speaker_identity_confidence_caps_structural_risk(
    assignment: dict,
    expected_reason: str,
) -> None:
    calibrated = speaker_identity_preprocess.calibrate_speaker_identity_confidence(
        assignment,
        {"numeric": 85, "band": "very_high", "band_label": "Very High"},
    )

    assert calibrated["numeric"] == 59
    assert expected_reason in calibrated["calibration"]["reasons"]


def test_calibrate_speaker_identity_confidence_keeps_advisory_flag_score() -> None:
    calibrated = speaker_identity_preprocess.calibrate_speaker_identity_confidence(
        {
            "status": "candidate_match",
            "review_flags": [
                "human_confirmation_required",
                "spoken_name_variant",
            ],
            "factors": [],
        },
        {"numeric": 85, "band": "very_high", "band_label": "Very High"},
    )

    assert calibrated["numeric"] == 85
    assert calibrated["band"] == "very_high"
    assert calibrated["calibration"]["applied"] is False
    assert calibrated["calibration"]["reasons"] == []


def test_validate_identity_evaluation_supports_grouped_and_utterance_assignments() -> None:
    transcript = _discovery_transcript()
    discovery = {
        "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
        "speaker_clues": [],
        "conversation_clues": [],
        "warnings": [],
    }
    packet = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=transcript,
        discovery_readout=discovery,
        person_records=[
            {
                "contact_id": "gws-alice",
                "label": "Alice Example",
                "email": "alice@example.com",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            }
        ],
        source_contexts=[
            {
                "source_id": "gws-personal",
                "owner": {"type": "person", "id": "operator", "label": "Operator"},
                "relationship_scope": "personal",
                "account_label": "Personal Google Workspace",
                "evidence_capabilities": ["calendar", "people", "gmail"],
                "authoritative_identifiers": ["email"],
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
                    "evidence_ids": ["event-1", "utterance-2"],
                    "rationale": "The event and discussion are both a proposal review.",
                },
                {
                    "factor": "time_window_alignment",
                    "direction": "neutral",
                    "strength": "weak",
                    "evidence_ids": [
                        _discovery_transcript()["recording_id"],
                        packet["calendar_context"]["attendees"][0]["id"],
                        "gws-personal",
                    ],
                    "rationale": "The recording identity is prepared, but timestamps are absent.",
                }
            ],
        },
        "person_links": [],
        "speaker_assignments": [
            {
                "speaker_labels": ["Speaker A", "Speaker B"],
                "status": "candidate_match",
                "person_id": person_id,
                "suggested_person": {},
                "transcript_clue_ids": ["utterance-1", "utterance-2"],
                "provenance_source_ids": ["gws-personal"],
                "factors": [
                    {
                        "factor": "direct_self_identification",
                        "direction": "support",
                        "strength": "decisive",
                        "evidence_ids": [
                            "utterance-1",
                            person_id,
                            "alice@example.com",
                        ],
                        "rationale": "The speaker states the candidate email.",
                    }
                ],
                "utterance_assignments": [
                    {"utterance_id": "utterance-1", "person_id": person_id},
                    {"utterance_id": "utterance-2", "person_id": "", "status": "unresolved"},
                ],
                "rationale": "The diarization labels may represent one person.",
                "review_flags": ["possible_diarization_split"],
            }
        ],
        "warnings": [],
    }

    validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
        packet,
        readout,
    )

    assignment = validated["readout"]["speaker_assignments"][0]
    assert assignment["confidence"]["numeric"] == 70
    assert assignment["confidence"]["band"] == "high"
    assert validated["readout"]["calendar_association"]["confidence"]["numeric"] == 50
    assert validated["requires_human_confirmation"] is True
    assert validated["safe_bulk_confirm_ready"] is False

    readout["speaker_assignments"][0]["review_flags"] = ["mixed_speaker_label"]
    calibrated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
        packet,
        readout,
    )
    calibrated_confidence = (
        calibrated["readout"]["speaker_assignments"][0]["confidence"]
    )
    assert calibrated_confidence["numeric"] == 59
    assert calibrated_confidence["uncapped_numeric"] == 70
    assert calibrated_confidence["calibration"]["reasons"] == [
        "material_flag:mixed_speaker_label"
    ]


def test_validate_identity_evaluation_rejects_invented_evidence() -> None:
    packet = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=_discovery_transcript(),
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
    )
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
                    "evidence_ids": ["invented-source"],
                }
            ],
        },
        "person_links": [],
        "speaker_assignments": [],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="unprepared evidence"):
        speaker_identity_preprocess.validate_and_score_identity_evaluation(packet, readout)


def test_two_phase_prompts_keep_discovery_separate_from_identity_evaluation() -> None:
    discovery_packet = speaker_identity_preprocess.build_clue_discovery_packet(
        transcript=_discovery_transcript(),
    )
    discovery_prompt = speaker_identity_preprocess.build_clue_discovery_prompt(
        discovery_packet
    )

    assert "Do not identify the speakers in this pass" in discovery_prompt
    assert "transcribe-audio.speaker-clue-discovery-readout.v1" in discovery_prompt
    assert json.dumps(discovery_packet, sort_keys=True, ensure_ascii=False) in discovery_prompt

    evaluation_packet = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=_discovery_transcript(),
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
    )
    evaluation_prompt = speaker_identity_preprocess.build_identity_evaluation_prompt(
        evaluation_packet
    )

    assert "Do not emit numeric confidence" in evaluation_prompt
    assert "candidate_match|unlisted|unresolved|conflicting" in evaluation_prompt
    assert "transcribe-audio.speaker-identity-evaluation-readout.v1" in evaluation_prompt


def test_collect_speaker_provenance_uses_discovered_person_hints(monkeypatch) -> None:
    seen_participants = []

    def fake_gws(transcript, readout, *, config):
        seen_participants.extend(readout["participants"])
        return []

    monkeypatch.setattr(speaker_identity_preprocess, "collect_gws_provenance", fake_gws)

    speaker_identity_preprocess.collect_speaker_provenance(
        {"event": {"attendees": []}},
        {"contact_candidates": []},
        discovery_readout={
            "speaker_clues": [
                {
                    "person_hints": [
                        {
                            "name": "Contextual Casey",
                            "email": "casey@example.com",
                            "organization": "Example Co",
                        }
                    ],
                    "retrieval_terms": ["casey@example.com", "Example Co"],
                }
            ],
            "conversation_clues": [
                {"retrieval_terms": ["proposal codename"]}
            ],
        },
        gws_configs=[GwsProvenanceConfig(enabled=True, profile_label="work")],
    )

    assert {
        "name": "Contextual Casey",
        "email": "casey@example.com",
        "organization": "Example Co",
    } in seen_participants


def test_collect_configured_identity_evidence_returns_attributed_person_records(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        speaker_identity_preprocess.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {
            "gws": ["gws-config"],
            "odollo": [],
            "source_contexts": [{"source_id": "gws-personal", "relationship_scope": "personal"}],
            "warnings": ["one config warning"],
        },
    )
    monkeypatch.setattr(
        speaker_identity_preprocess,
        "collect_speaker_provenance",
        lambda *args, **kwargs: {
            "sources": [
                {
                    "source_type": "gws_contact",
                    "source_id": "contact-alice",
                    "label": "Alice Example",
                    "metadata": {
                        "profile": "gws-personal",
                        "email": "alice@example.com",
                    },
                },
                {
                    "source_type": "gws_mail_message",
                    "source_id": "mail-1",
                    "label": "Proposal",
                    "snippet": "Alice sent it.",
                    "metadata": {"profile": "gws-personal"},
                },
            ],
            "warnings": [],
        },
    )

    result = speaker_identity_preprocess.collect_configured_identity_evidence(
        transcript=_discovery_transcript(),
        identity_bundle={"contact_candidates": []},
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
        },
    )

    assert result["person_records"][0]["contact_id"] == "contact-alice"
    assert result["person_records"][0]["source_id"] == "gws-personal"
    assert result["provenance_sources"][1]["source_id"] == "mail-1"
    assert result["source_contexts"][0]["source_id"] == "gws-personal"
    assert result["warnings"] == ["one config warning"]


def test_person_link_assessment_can_propose_contextual_cross_source_group() -> None:
    packet = speaker_identity_preprocess.build_identity_evaluation_packet(
        transcript=_discovery_transcript(),
        discovery_readout={
            "schema_version": "transcribe-audio.speaker-clue-discovery-readout.v1",
            "speaker_clues": [],
            "conversation_clues": [],
            "warnings": [],
        },
        person_records=[
            {
                "contact_id": "gws-casey",
                "label": "Casey Jones",
                "source_id": "gws-personal",
                "source_type": "gws_contact",
            },
            {
                "contact_id": "odoo-casey",
                "label": "C. Jones",
                "source_id": "odoo-company",
                "source_type": "odollo_contact",
            },
        ],
    )
    person_ids = [item["person_id"] for item in packet["people"]]
    readout = {
        "schema_version": "transcribe-audio.speaker-identity-evaluation-readout.v1",
        "evaluation_id": packet["evaluation_id"],
        "calendar_association": {"status": "ambiguous", "factors": []},
        "person_links": [
            {
                "person_ids": person_ids,
                "status": "same_person",
                "factors": [
                    {
                        "factor": "name_and_organization_alignment",
                        "direction": "support",
                        "strength": "strong",
                        "evidence_ids": ["gws-casey"],
                    },
                    {
                        "factor": "cross_source_relationship_context",
                        "direction": "support",
                        "strength": "moderate",
                        "evidence_ids": ["odoo-casey"],
                    },
                ],
            }
        ],
        "speaker_assignments": [],
        "warnings": [],
    }

    validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
        packet,
        readout,
    )

    assert validated["readout"]["person_links"][0]["confidence"]["numeric"] == 80
    assert validated["person_group_proposals"][0]["person_ids"] == person_ids
    assert validated["person_group_proposals"][0]["status"] == "ready_to_group"


def test_build_speaker_clue_packet_is_speaker_specific_and_attendee_first() -> None:
    transcript = {
        "source_media_path": "/private/meeting.m4a",
        "utterances": [
            {
                "speaker": "Speaker A",
                "start": 0,
                "end": 4,
                "text": "I sent the revised proposal from alice@example.com.",
            },
            {
                "speaker": "Speaker B",
                "start": 4,
                "end": 8,
                "text": "Thanks Alice. I will review it for Example Co.",
            },
        ],
        "event": {
            "summary": "Proposal review",
            "attendees": [
                {"displayName": "Alice Example", "email": "alice@example.com"},
                {"displayName": "Bob Buyer", "email": "bob@example.com"},
            ],
        },
    }
    identity_bundle = {
        "source_document_id": "doc-1",
        "contact_candidates": [
            {
                "contact_id": "contact-bob",
                "label": "Bob Buyer",
                "email": "bob@example.com",
                "source_profile": "work",
                "confidence": 0.8,
            },
            {
                "contact_id": "contact-alice",
                "label": "Alice Example",
                "email": "alice@example.com",
                "source_profile": "work",
                "confidence": 0.8,
            },
        ],
    }
    provenance_sources = [
        {
            "source_type": "gws_mail_message",
            "source_id": "mail-1",
            "label": "Proposal thread",
            "snippet": "Alice sent the revised proposal.",
            "metadata": {"profile": "work", "email": "alice@example.com"},
        }
    ]

    packet = speaker_identity_preprocess.build_speaker_clue_packet(
        conversation_key="conversation-1",
        transcript=transcript,
        identity_bundle=identity_bundle,
        provenance_sources=provenance_sources,
    )

    assert packet["schema_version"] == "transcribe-audio.speaker-clue-packet.v1"
    assert [item["email"] for item in packet["calendar_attendees"]] == [
        "alice@example.com",
        "bob@example.com",
    ]
    assert [item["contact_id"] for item in packet["contact_candidates"]] == [
        "contact-alice",
        "contact-bob",
    ]
    assert packet["speakers"][0]["utterance_clues"][0]["text"].startswith("I sent")
    assert packet["speakers"][1]["utterance_clues"][0]["text"].startswith("Thanks Alice")
    assert packet["provenance_sources"][0]["source_id"] == "mail-1"
    assert packet["policy"]["requires_human_review"] is True
    assert packet["policy"]["will_apply_assignments"] is False
    assert "transcript_text" not in packet
    assert "source_media_path" not in packet["conversation"]


def test_validate_speaker_identity_readout_accepts_only_prepared_references() -> None:
    packet = {
        "speakers": [
            {
                "speaker_label": "Speaker A",
                "utterance_clues": [{"utterance_id": "utterance-1", "text": "I sent it."}],
            }
        ],
        "contact_candidates": [{"contact_id": "contact-alice"}],
        "provenance_sources": [{"source_id": "mail-1"}],
    }
    readout = {
        "schema_version": "transcribe-audio.speaker-identity-readout.v1",
        "speakers": [
            {
                "speaker_label": "Speaker A",
                "status": "proposed",
                "candidate_id": "contact-alice",
                "confidence": 0.91,
                "transcript_clue_ids": ["utterance-1"],
                "provenance_source_ids": ["mail-1"],
                "alternatives": [],
                "review_flags": [],
            }
        ],
        "warnings": [],
    }

    validated = speaker_identity_preprocess.validate_speaker_identity_readout(packet, readout)

    assert validated["valid"] is True
    assert validated["requires_human_review"] is True
    assert validated["will_apply_assignments"] is False
    assert validated["readout"]["speakers"][0]["candidate_id"] == "contact-alice"


def test_validate_speaker_identity_readout_rejects_invented_references() -> None:
    packet = {
        "speakers": [{"speaker_label": "Speaker A", "utterance_clues": []}],
        "contact_candidates": [{"contact_id": "contact-alice"}],
        "provenance_sources": [],
    }
    readout = {
        "schema_version": "transcribe-audio.speaker-identity-readout.v1",
        "speakers": [
            {
                "speaker_label": "Speaker A",
                "candidate_id": "invented-contact",
                "confidence": 0.8,
                "transcript_clue_ids": [],
                "provenance_source_ids": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="unprepared candidate_id"):
        speaker_identity_preprocess.validate_speaker_identity_readout(packet, readout)


def test_build_speaker_identity_prompt_requires_cited_json_only_output() -> None:
    packet = {
        "schema_version": "transcribe-audio.speaker-clue-packet.v1",
        "speakers": [{"speaker_label": "Speaker A", "utterance_clues": []}],
        "contact_candidates": [],
        "provenance_sources": [],
    }

    prompt = speaker_identity_preprocess.build_speaker_identity_prompt(packet)

    assert "Do not summarize the conversation" in prompt
    assert "Every identity proposal must cite" in prompt
    assert "transcribe-audio.speaker-identity-readout.v1" in prompt
    assert json.dumps(packet, sort_keys=True, ensure_ascii=False) in prompt


def test_collect_speaker_provenance_queries_each_read_only_profile(monkeypatch) -> None:
    calls = []

    def fake_gws(transcript, readout, *, config):
        calls.append(("gws", config.profile_label, readout["participants"]))
        return [
            ProvenanceSource(
                source_type="gws_mail_message",
                source_id="mail-1",
                label="Proposal thread",
                snippet="Alice sent the proposal.",
                metadata={"profile": config.profile_label},
            )
        ]

    def fake_odollo(transcript, readout, *, config):
        calls.append(("odollo", config.profiles[0], readout["participants"]))
        return [
            ProvenanceSource(
                source_type="odollo_lead",
                source_id="lead-1",
                label="Alice Example | Proposal",
                metadata={"profile": config.profiles[0]},
            )
        ]

    monkeypatch.setattr(speaker_identity_preprocess, "collect_gws_provenance", fake_gws)
    monkeypatch.setattr(speaker_identity_preprocess, "collect_odollo_provenance", fake_odollo)

    result = speaker_identity_preprocess.collect_speaker_provenance(
        {"event": {"attendees": [{"displayName": "Alice Example", "email": "alice@example.com"}]}},
        {"contact_candidates": []},
        gws_configs=[GwsProvenanceConfig(enabled=True, profile_label="work")],
        odollo_configs=[OdolloProvenanceConfig(enabled=True, profiles=("soylei-prod",))],
    )

    assert [source["source_type"] for source in result["sources"]] == [
        "gws_mail_message",
        "odollo_lead",
    ]
    assert result["warnings"] == []
    assert calls[0][2][0]["email"] == "alice@example.com"
    assert calls[1][2][0]["email"] == "alice@example.com"


def test_build_configured_packet_loads_shared_provenance_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        speaker_identity_preprocess.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {
            "gws": ["gws-config"],
            "odollo": ["odollo-config"],
            "source_contexts": [
                {
                    "source_id": "gws-work",
                    "owner": {"type": "person", "id": "operator", "label": "Local operator"},
                    "relationship_scope": "personal",
                    "account_label": "Personal workspace",
                    "evidence_capabilities": ["gmail", "people"],
                    "authoritative_identifiers": ["email"],
                }
            ],
            "warnings": [],
        },
    )

    def fake_collect(transcript, identity_bundle, *, gws_configs, odollo_configs):
        assert gws_configs == ["gws-config"]
        assert odollo_configs == ["odollo-config"]
        return {
            "sources": [
                {
                    "source_type": "gws_mail_message",
                    "source_id": "mail-1",
                    "label": "Proposal",
                    "snippet": "Alice sent it.",
                    "metadata": {"profile": "work"},
                }
            ],
            "warnings": ["one bounded source warning"],
        }

    monkeypatch.setattr(speaker_identity_preprocess, "collect_speaker_provenance", fake_collect)

    packet = speaker_identity_preprocess.build_configured_speaker_clue_packet(
        conversation_key="conversation-1",
        transcript={"utterances": [{"speaker": "Speaker A", "text": "Hello"}]},
        identity_bundle={"source_document_id": "doc-1", "contact_candidates": []},
        provenance_path=tmp_path / "provenance.config.json",
    )

    assert packet["provenance_sources"][0]["source_id"] == "mail-1"
    assert packet["source_contexts"][0]["source_id"] == "gws-work"
    assert packet["collection_warnings"] == ["one bounded source warning"]


def test_build_configured_packet_promotes_odollo_lead_as_review_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        speaker_identity_preprocess.provenance_config,
        "speaker_preprocessing_source_configs_from_provenance",
        lambda **kwargs: {"gws": [], "odollo": [], "source_contexts": [], "warnings": []},
    )
    monkeypatch.setattr(
        speaker_identity_preprocess,
        "collect_speaker_provenance",
        lambda *args, **kwargs: {
            "sources": [
                {
                    "source_type": "odollo_lead",
                    "source_id": "lead-1",
                    "label": "Alice Example | Proposal",
                    "metadata": {
                        "profile": "soylei-prod",
                        "email": "alice@example.com",
                    },
                }
            ],
            "warnings": [],
        },
    )

    packet = speaker_identity_preprocess.build_configured_speaker_clue_packet(
        conversation_key="conversation-1",
        transcript={
            "event": {"attendees": [{"email": "alice@example.com"}]},
            "utterances": [{"speaker": "Speaker A", "text": "Hello"}],
        },
        identity_bundle={"contact_candidates": []},
    )

    assert packet["contact_candidates"][0]["contact_id"] == "lead-1"
    assert packet["contact_candidates"][0]["email"] == "alice@example.com"
    assert packet["contact_candidates"][0]["source_profile"] == "soylei-prod"
