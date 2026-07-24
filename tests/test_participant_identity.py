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
        assert transcript["event"]["participants"] == []
        assert "alice@example.com" in readout["participants"]
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
    assert len(bundle["contact_candidates"]) == 1
    assert {source["source_type"] for source in bundle["contact_candidates"][0]["merged_sources"]} == {
        "gws_contact",
        "odollo_contact",
    }
    assert bundle["contact_candidates"][0]["confidence"] == 0.95
    assert bundle["speakers"][0]["speaker_label"] == "Speaker A"
    assert bundle["speakers"][0]["review_required"] is True
    assert bundle["review_status"] == "needs_review"


def test_identity_bundle_uses_shared_provenance_config(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "provenance.config.json").write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.provenance-config.v1",
                "active_profile": "default",
                "profiles": {
                    "default": {
                        "source_ids": ["gws-work", "odollo-saber"],
                        "workflows": {
                            "participant_identity": {
                                "source_ids": ["gws-work", "odollo-saber"],
                            }
                        },
                    }
                },
                "sources": {
                    "gws-work": {
                        "kind": "gws",
                        "enabled": True,
                        "label": "Work gws",
                        "config_dir": "~/.config/gws-work",
                        "people": {"surfaces": ["contacts"], "limit": 2, "query_limit": 3},
                        "read_only": True,
                    },
                    "odollo-saber": {
                        "kind": "odollo",
                        "enabled": True,
                        "label": "SABER Odoo",
                        "tenant_profile": "saber-prod",
                        "command": ["odollo"],
                        "limits": {"contacts": 2},
                        "read_only": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_gws(query_terms, *, config):
        assert config.profile_label == "Work gws"
        assert config.people_page_size == 2
        return []

    def fake_odollo(transcript, readout, *, config):
        assert config.profiles == ("saber-prod",)
        assert config.command == ("odollo",)
        return [
            ProvenanceSource(
                source_type="odollo_contact",
                source_id="partner-9",
                label="Alice Example | SABER",
                snippet="Alice Example; alice@example.com; SABER",
                metadata={"profile": "saber-prod", "email": "alice@example.com"},
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

    assert bundle["source_profiles"] == [
        {"source": "gws", "profile": "Work gws", "surfaces": ["contacts"], "read_only": True},
        {"source": "odollo", "profile": "saber-prod", "models": ["res.partner"], "read_only": True},
    ]
    assert bundle["contact_candidates"][0]["source_profile"] == "saber-prod"


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
            },
            {
                "label": "Alice Example",
                "name": "Alice Example",
                "email": "alice@example.com",
            },
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
    assert terms[:2] == ["ecochran76@gmail.com", "alice@example.com"]


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


def test_ranked_contact_candidates_dedupes_same_email_across_sources() -> None:
    candidates = [
        {
            "contact_id": "gws-alice",
            "label": "Alice Example",
            "email": "alice@example.com",
            "source": "gws_contact",
            "source_type": "gws_contact",
            "source_profile": "work",
            "confidence": 0.95,
            "evidence": [{"kind": "gws"}],
        },
        {
            "contact_id": "odollo-alice",
            "label": "Alice Example",
            "email": "alice@example.com",
            "source": "odollo_contact",
            "source_type": "odollo_contact",
            "source_profile": "soylei-prod",
            "confidence": 0.85,
            "evidence": [{"kind": "odollo"}],
        },
    ]

    ranked = participant_identity.ranked_contact_candidates(candidates, limit=20, per_source_profile=2)

    assert len(ranked) == 1
    assert ranked[0]["contact_id"] == "gws-alice"
    assert ranked[0]["dedupe_key"] == "email:alice@example.com"
    assert ranked[0]["source_count"] == 2
    assert {source["source_type"] for source in ranked[0]["merged_sources"]} == {"gws_contact", "odollo_contact"}


def test_ranked_contact_candidates_merges_configured_contact_aliases() -> None:
    candidates = [
        {
            "contact_id": "gws-eric",
            "label": "eric@saberchemical.com",
            "email": "eric@saberchemical.com",
            "source": "gws_contact",
            "source_type": "gws_contact",
            "source_profile": "work",
            "confidence": 0.95,
        },
        {
            "contact_id": "odollo-eric",
            "label": "Eric C.",
            "email": "eco@example.com",
            "source": "odollo_contact",
            "source_type": "odollo_contact",
            "source_profile": "saber-prod",
            "confidence": 0.85,
        },
    ]
    aliases = participant_identity.normalize_contact_aliases(
        [
            {
                "id": "operator-eric",
                "label": "Eric Cochran",
                "primary_email": "eric@saberchemical.com",
                "emails": ["eric@saberchemical.com", "eco@example.com", "ecochran76@gmail.com"],
                "email_patterns": ["ecochran76*@gmail.com"],
            }
        ]
    )

    ranked = participant_identity.ranked_contact_candidates(candidates, aliases=aliases)

    assert len(ranked) == 1
    assert ranked[0]["label"] == "Eric Cochran"
    assert ranked[0]["email"] == "eric@saberchemical.com"
    assert ranked[0]["dedupe_key"] == "alias:operator-eric"
    assert ranked[0]["source_count"] == 2

    plus_ranked = participant_identity.ranked_contact_candidates(
        [
            {
                "contact_id": "gws-eric-plus",
                "label": "ecochran76+14@gmail.com",
                "email": "ecochran76+14@gmail.com",
                "source": "gws_other_contact",
                "source_type": "gws_other_contact",
                "source_profile": "work",
                "confidence": 0.7,
            }
        ],
        aliases=aliases,
    )

    assert plus_ranked[0]["label"] == "Eric Cochran"
    assert plus_ranked[0]["dedupe_key"] == "alias:operator-eric"


def test_ranked_contact_candidates_merges_strong_same_person_names() -> None:
    ranked = participant_identity.ranked_contact_candidates(
        [
            {
                "contact_id": "operator-baker",
                "label": "Baker Kuehl",
                "email": "baker@saberchemical.com",
                "source": "operator_participant_hint",
                "source_type": "operator_participant_hint",
                "source_profile": "user_config",
                "confidence": 0.9,
            },
            {
                "contact_id": "gws-baker",
                "label": "Baker Kuehl",
                "email": "bwkuehl@iastate.edu",
                "source": "gws_contact",
                "source_type": "gws_contact",
                "source_profile": "work",
                "confidence": 0.78,
            },
        ],
    )

    assert len(ranked) == 1
    assert ranked[0]["label"] == "Baker Kuehl"
    assert ranked[0]["email"] == "baker@saberchemical.com"
    assert ranked[0]["source_count"] == 2
    assert {source["email"] for source in ranked[0]["merged_sources"]} == {
        "baker@saberchemical.com",
        "bwkuehl@iastate.edu",
    }
    assert "name:baker kuehl" in ranked[0]["merge_keys"]


def test_ranked_contact_candidates_merges_inverted_names() -> None:
    ranked = participant_identity.ranked_contact_candidates(
        [
            {
                "contact_id": "operator-sean",
                "label": "Sean Solberg",
                "source": "operator_participant_hint",
                "source_type": "operator_participant_hint",
                "source_profile": "user_config",
                "confidence": 0.9,
            },
            {
                "contact_id": "gws-sean",
                "label": "Solberg, Sean",
                "email": "ssolberg@fredlaw.com",
                "source": "gws_other_contact",
                "source_type": "gws_other_contact",
                "source_profile": "work",
                "confidence": 0.78,
            },
        ],
    )

    assert len(ranked) == 1
    assert ranked[0]["label"] == "Sean Solberg"
    assert ranked[0]["email"] == "ssolberg@fredlaw.com"
    assert ranked[0]["source_count"] == 2
    assert "name:sean solberg" in ranked[0]["merge_keys"]


def test_ranked_contact_candidates_keeps_weak_single_token_names_apart() -> None:
    ranked = participant_identity.ranked_contact_candidates(
        [
            {
                "contact_id": "operator-michael",
                "label": "Michael",
                "source": "operator_participant_hint",
                "source_type": "operator_participant_hint",
                "source_profile": "user_config",
                "confidence": 0.9,
            },
            {
                "contact_id": "gws-michael",
                "label": "Michael",
                "email": "michael@example.com",
                "source": "gws_other_contact",
                "source_type": "gws_other_contact",
                "source_profile": "work",
                "confidence": 0.78,
            },
        ],
    )

    assert len(ranked) == 2
    assert all(candidate["source_count"] == 1 for candidate in ranked)


def test_identity_bundle_uses_operator_participant_hints_from_user_config(tmp_path: Path, monkeypatch) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "provenance.config.json").write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.provenance-config.v1",
                "active_profile": "default",
                "contacts": {
                    "participant_hints": [
                        {
                            "match": {"source_document_id": "doc-1"},
                            "participants": [{"name": "Sean Solberg"}],
                        }
                    ]
                },
                "profiles": {
                    "default": {
                        "workflows": {
                            "participant_identity": {"source_ids": ["gws-work"]},
                        }
                    }
                },
                "sources": {
                    "gws-work": {
                        "kind": "gws",
                        "enabled": True,
                        "label": "Work gws",
                        "people": {"surfaces": ["contacts"], "limit": 2},
                        "read_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_gws(query_terms, *, config):
        assert "Sean Solberg" in query_terms
        return [
            ProvenanceSource(
                source_type="gws_contact",
                source_id="people/sean",
                label="Sean Solberg",
                snippet="Sean Solberg; sean@example.com",
                metadata={"profile": "work", "email": "sean@example.com"},
            )
        ]

    monkeypatch.setattr(participant_identity, "collect_gws_contact_provenance", fake_gws)

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

    assert bundle["operator_participant_hints"][0]["label"] == "Sean Solberg"
    assert bundle["contact_candidates"][0]["label"] == "Sean Solberg"
