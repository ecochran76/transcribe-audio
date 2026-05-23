from __future__ import annotations

import json
from pathlib import Path

import participant_identity
from routing_artifacts import ProvenanceSource


def transcript_with_attendees() -> dict:
    return {
        "transcript_text": "Speaker A [0.00s - 1.00s]: Hello.",
        "utterances": [{"speaker": "Speaker A", "text": "Hello."}],
        "event": {
            "summary": "Alice Example review",
            "participants": ["Alice Example <alice@example.com>"],
            "matching_calendars": [
                {
                    "calendar_summary": "Work",
                    "event_summary": "Alice Example review",
                    "attendees": [{"displayName": "Bob Buyer", "email": "bob@example.com"}],
                }
            ],
        },
    }


def test_extract_calendar_attendees_normalizes_primary_and_matching_calendar() -> None:
    attendees = participant_identity.extract_calendar_attendees(transcript_with_attendees())

    assert [item["email"] for item in attendees] == ["alice@example.com", "bob@example.com"]
    assert attendees[0]["source"] == "primary_event.participants"
    assert attendees[1]["calendar_summary"] == "Work"


def test_identity_bundle_uses_configured_contact_provenance(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / participant_identity.CONTACT_SOURCE_CONFIG_NAME).write_text(
        json.dumps(
            {
                "gws": {"profiles": [{"label": "work", "surfaces": ["contacts"], "limit": 3}]},
                "odollo": {"profiles": [{"label": "soylei-prod", "limit": 3}]},
            }
        ),
        encoding="utf-8",
    )

    def fake_gws(query_terms, *, config):
        assert "alice@example.com" in query_terms
        return [
            ProvenanceSource(
                source_type="gws_contact",
                source_id="people/c1",
                label="Alice Example",
                snippet="Alice Example; alice@example.com",
                metadata={"profile": "work", "email": "alice@example.com"},
            )
        ]

    def fake_odollo(transcript, readout, *, config):
        return [
            ProvenanceSource(
                source_type="odollo_contact",
                source_id="partner-7",
                label="Alice Example | Example Co",
                uri="odoo://soylei-prod/res.partner/7",
                snippet="Alice Example; alice@example.com; Example Co",
                metadata={"profile": "soylei-prod", "email": "alice@example.com", "record_id": 7},
            )
        ]

    monkeypatch.setattr(participant_identity, "collect_gws_contact_provenance", fake_gws)
    monkeypatch.setattr(participant_identity, "collect_odollo_provenance", fake_odollo)

    bundle = participant_identity.build_participant_identity_bundle(
        conversation_key="conversation-1",
        source_document_id="doc-1",
        transcript=transcript_with_attendees(),
        transcript_text="",
        readout_participants=[],
        local_contacts=[],
        assignments={},
        state_root=state_root,
    )

    assert bundle["schema_version"] == participant_identity.IDENTITY_BUNDLE_SCHEMA_VERSION
    assert {profile["source"] for profile in bundle["source_profiles"]} == {"gws", "odollo"}
    assert {candidate["source_type"] for candidate in bundle["contact_candidates"]} == {
        "gws_contact",
        "odollo_contact",
    }
    assert bundle["contact_candidates"][0]["confidence"] == 0.95
    assert bundle["speakers"][0]["speaker_label"] == "Speaker A"
    assert bundle["speakers"][0]["review_required"] is True
    assert bundle["review_status"] == "needs_review"


def test_identity_bundle_records_operator_assignment_as_reviewed() -> None:
    bundle = participant_identity.build_participant_identity_bundle(
        conversation_key="conversation-1",
        source_document_id="doc-1",
        transcript=transcript_with_attendees(),
        transcript_text="",
        readout_participants=[],
        local_contacts=[],
        assignments={
            "Speaker A": {
                "speaker_label": "Speaker A",
                "status": "confirmed",
                "contact_id": "contact-alice",
                "contact_label": "Alice Example",
                "confidence": 1.0,
                "evidence": [{"source": "operator_review"}],
                "updated_at": "2026-05-23T00:00:00Z",
            }
        },
        state_root=None,
    )

    assert bundle["review_status"] == "reviewed"
    assert bundle["operator_decisions"][0]["contact_label"] == "Alice Example"
    assert bundle["unresolved_ambiguities"] == []


def test_identity_query_terms_excludes_anonymous_speaker_labels() -> None:
    terms = participant_identity.identity_query_terms(
        calendar_attendees=[
            {
                "label": "Eric Cochran",
                "name": "Eric Cochran",
                "email": "ecochran76@gmail.com",
            }
        ],
        readout_participants=[],
        speaker_labels=["A", "B", "Speaker C", "SPEAKER_00", "Actual Customer"],
    )

    assert "ecochran76@gmail.com" in terms
    assert "Eric Cochran" in terms
    assert "Actual Customer" in terms
    assert "A" not in terms
    assert "B" not in terms
    assert "Speaker C" not in terms
    assert "SPEAKER_00" not in terms


def test_ranked_contact_candidates_preserves_source_profile_representation() -> None:
    candidates = [
        {
            "contact_id": f"gws-{index}",
            "label": f"Google Candidate {index}",
            "source": "gws_contact",
            "source_type": "gws_contact",
            "source_profile": "default",
            "confidence": 0.9,
        }
        for index in range(30)
    ]
    candidates.extend(
        [
            {
                "contact_id": f"odollo-{index}",
                "label": f"Odollo Candidate {index}",
                "source": "odollo_contact",
                "source_type": "odollo_contact",
                "source_profile": "soylei-prod",
                "confidence": 0.4,
            }
            for index in range(2)
        ]
    )

    ranked = participant_identity.ranked_contact_candidates(candidates, limit=20, per_source_profile=2)

    assert len(ranked) == 20
    assert sum(1 for item in ranked if item["source_type"] == "odollo_contact") == 2
