from __future__ import annotations

import json
from pathlib import Path

import pytest

import speaker_preprocessing_workflow
from conversation_identity_retrieval import (
    IdentityCandidate,
    PreparedIdentityEvidenceBundle,
    RankedEvidence,
)
from conversation_knowledge_evidence import (
    EvidenceBundleItem,
    EvidenceBundleRecord,
    EvidenceScope,
    EvidenceSnapshotRecord,
    RetrievalRequestRecord,
)


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


def _retrieval_bundle() -> PreparedIdentityEvidenceBundle:
    request = RetrievalRequestRecord(
        request_id="cdb0e51b-bd0e-44ea-8822-d69e879b2934",
        conversation_id="1934616f-3cb5-4f70-b037-679d863d4847",
        recording_ids=("2dbdfe40-71e6-4597-ab77-5d56613d0e65",),
        speaker_labels=("Speaker A",),
        clue_ids=("utterance-1",),
        conversation_at="2026-07-25T15:00:00Z",
        as_of="2026-07-25T16:00:00Z",
        prepared_person_ids=("a1a26512-4236-4e95-b3bc-e2d1d89d44a4",),
        scopes=(EvidenceScope("gws-personal", "ecochran76@gmail.com", ""),),
        capabilities=("contacts", "mail"),
        budgets={"max_records": 2, "max_characters": 1000},
        freshness_policy="current_only",
        hindsight_policy="exclude",
        retrieval_version="conversation-identity-retrieval.v1",
        ranking_version="conversation-identity-ranking.v1",
        requesting_workflow="speaker_identity_evaluation",
        run_id="run-retrieval-1",
        created_at="2026-07-25T16:00:00Z",
    )
    included = EvidenceSnapshotRecord(
        evidence_id="27b46671-5ef7-4993-ad59-66c6850b0753",
        source_record_id="mail-1",
        source_profile_id="gws-personal",
        provider_kind="gws",
        account_id="ecochran76@gmail.com",
        tenant_id="",
        source_type="gws_mail_message",
        capability="mail",
        snippet="Alice discusses the proposal with Eric.",
        structured_metadata={"label": "Proposal thread"},
        source_event_at="2026-07-25T14:00:00Z",
        observed_at="2026-07-25T14:00:00Z",
        retrieved_at="2026-07-25T16:00:00Z",
        temporal_class="contemporaneous",
        source_uri="gws://mail/mail-1",
        content_hash="a" * 64,
        independence_group_id="gws:thread-1",
        freshness_state="current",
    )
    excluded = EvidenceSnapshotRecord(
        evidence_id="c424ef6d-2e0f-4ace-b013-28b3d4c4564a",
        source_record_id="mail-2",
        source_profile_id="gws-personal",
        provider_kind="gws",
        account_id="ecochran76@gmail.com",
        tenant_id="",
        source_type="gws_mail_message",
        capability="mail",
        snippet="This excluded content must not reach the model.",
        structured_metadata={"label": "Later thread"},
        source_event_at="2026-07-25T15:30:00Z",
        observed_at="2026-07-25T15:30:00Z",
        retrieved_at="2026-07-25T16:00:00Z",
        temporal_class="contemporaneous",
        source_uri="gws://mail/mail-2",
        content_hash="b" * 64,
        independence_group_id="gws:thread-2",
        freshness_state="current",
    )
    bundle_record = EvidenceBundleRecord.create(
        bundle_id="7cd9f1ea-2b66-4287-b968-d6fa9b95a6b5",
        request_id=request.request_id,
        status="partial",
        items=(
            EvidenceBundleItem(
                evidence_id=included.evidence_id,
                disposition="included",
                reason_code="ranked_support",
                rank=1,
                score=0.9,
                metadata={},
            ),
            EvidenceBundleItem(
                evidence_id=excluded.evidence_id,
                disposition="excluded",
                reason_code="per_source_budget",
                rank=0,
                score=0.2,
                metadata={},
            ),
        ),
        candidate_person_ids=request.prepared_person_ids,
        warnings=("provider_partial_failure",),
        source_failures=({"source_profile_id": "odoo-main", "reason": "unavailable"},),
        allowlists={
            "person_ids": list(request.prepared_person_ids),
            "evidence_ids": [included.evidence_id],
            "speaker_labels": ["Speaker A"],
            "clue_ids": ["utterance-1"],
        },
        created_at=request.created_at,
    )
    return PreparedIdentityEvidenceBundle(
        request=request,
        persisted_bundle=bundle_record,
        calendar_candidates=(),
        transcript_clues=(),
        people=(
            IdentityCandidate(
                person_id=request.prepared_person_ids[0],
                source_record_ids=("contact-alice",),
                source_profile_ids=("gws-personal",),
                match_reasons=("calendar_attendee_email",),
                exact_identities=("email:alice@example.com",),
                display_name="Alice Example",
            ),
        ),
        relationships=(),
        evidence=(
            RankedEvidence(
                snapshot=included,
                score=0.9,
                direction="support",
                features={"exact_identifier": 1.0},
                disposition="included",
                reason_code="ranked_support",
            ),
            RankedEvidence(
                snapshot=excluded,
                score=0.2,
                direction="neutral",
                features={},
                disposition="excluded",
                reason_code="per_source_budget",
            ),
        ),
        warnings=bundle_record.warnings,
        source_failures=tuple(bundle_record.source_failures),
    )


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
    assert evaluation["rubric_versions"]["speaker_identity"] == "speaker-identity.v2"

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


def test_retrieval_bundle_drives_identity_packet_and_preserves_review_gate(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    bundle = _retrieval_bundle()
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
        retrieval_bundle=bundle,
        route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
    )

    packet = prepared["packet"]
    person_id = bundle.people[0].person_id
    included_id = bundle.evidence[0].snapshot.evidence_id
    excluded_id = bundle.evidence[1].snapshot.evidence_id
    assert packet["people"][0]["person_id"] == person_id
    assert packet["people"][0]["display_name"] == "Alice Example"
    assert [item["source_id"] for item in packet["provenance_sources"]] == [
        included_id
    ]
    assert packet["provenance_sources"][0]["freshness_state"] == "current"
    assert packet["provenance_sources"][0]["temporal_class"] == "contemporaneous"
    assert packet["provenance_sources"][0]["inclusion_reason"] == "ranked_support"
    assert packet["retrieval"]["status"] == "partial"
    assert packet["retrieval"]["warnings"] == ["provider_partial_failure"]
    assert packet["retrieval"]["evidence"][1]["evidence_id"] == excluded_id
    assert packet["retrieval"]["evidence"][1]["reason_code"] == "per_source_budget"
    assert "This excluded content" not in json.dumps(packet)
    assert packet["policy"]["requires_human_confirmation"] is True
    assert packet["policy"]["will_apply_assignments"] is False

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
                "transcript_clue_ids": ["utterance-1"],
                "provenance_source_ids": [included_id],
                    "factors": [
                        {
                            "factor": "verified_identifier_match",
                            "direction": "support",
                            "strength": "strong",
                            "evidence_ids": [included_id, "utterance-1"],
                        }
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
        run_references={"retrieval_bundle_id": bundle.persisted_bundle.bundle_id},
    )

    evaluation = record["evaluations"][0]
    assert evaluation["retrieval"]["bundle_id"] == bundle.persisted_bundle.bundle_id
    assert evaluation["evidence_snapshots"][0]["freshness_state"] == "current"
    assert evaluation["evidence_snapshots"][0]["inclusion_reason"] == "ranked_support"
    assert evaluation["proposals"][0]["factors"][0]["evidence_ids"] == [
        included_id,
        "utterance-1",
    ]
    assert evaluation["review_state"] == {
        "pending_count": 1,
        "requires_human_confirmation": True,
        "will_apply_assignments": False,
    }


def test_retrieval_bundle_rejects_parallel_legacy_evidence_authority(
    tmp_path: Path,
) -> None:
    transcript_path = _transcript(tmp_path / "meeting Transcript.transcript.json")
    with pytest.raises(ValueError, match="cannot be combined"):
        speaker_preprocessing_workflow.prepare_identity_evaluation(
            transcript_path,
            document_id="doc-1",
            state_root=tmp_path / "state",
            discovery_readout={
                "schema_version": (
                    "transcribe-audio.speaker-clue-discovery-readout.v1"
                ),
                "speaker_clues": [],
                "conversation_clues": [],
                "warnings": [],
            },
            retrieval_bundle=_retrieval_bundle(),
            provenance_sources=[{"source_id": "legacy-mail"}],
            route={"provider": "codex-app-server", "model": "gpt-5.6-sol"},
        )
