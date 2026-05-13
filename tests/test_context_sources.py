from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import context_sources
import route_transcript
from context_sources import (
    GraphitiProvenanceConfig,
    GwsProvenanceConfig,
    OdolloProvenanceConfig,
    build_drive_query,
    build_graphiti_query,
    build_odollo_terms,
    collect_graphiti_provenance,
    collect_gws_provenance,
    collect_odollo_provenance,
    filter_provenance_sources,
)


def sample_transcript() -> dict:
    return {
        "event": {
            "summary": "Soylei and Tempo Chemical Technical discussion",
            "matching_calendars": [
                {
                    "calendar_id": "primary",
                    "event_id": "event-1",
                    "calendar_summary": "primary",
                    "event_summary": "Soylei and Tempo Chemical Technical discussion",
                }
            ],
        }
    }


def sample_readout() -> dict:
    return {
        "matter_candidates": [
            {
                "label": "SoyLei Tempo Chemical technical collaboration",
                "confidence": 0.95,
                "evidence": "Calendar and transcript evidence.",
            }
        ]
    }


def test_build_drive_query_uses_event_and_matter_terms() -> None:
    query = build_drive_query(sample_transcript(), sample_readout())

    assert "trashed=false" in query
    assert "Soylei" in query
    assert "Tempo" in query
    assert "fullText contains" not in query
    assert " and " in query


def test_collect_gws_provenance_converts_calendar_and_drive(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, text, capture_output, timeout, check, env):
        commands.append(command)
        if command[1:4] == ["calendar", "events", "get"]:
            payload = {
                "id": "event-1",
                "summary": "Soylei and Tempo Chemical Technical discussion",
                "htmlLink": "https://calendar.example/event-1",
                "attendees": [{"email": "paul@tempochem.com"}],
            }
        else:
            payload = {
                "files": [
                    {
                        "id": "file-1",
                        "name": "Tempo Chemical NDA",
                        "mimeType": "application/vnd.google-apps.document",
                        "webViewLink": "https://drive.example/file-1",
                        "modifiedTime": "2026-05-10T00:00:00Z",
                    }
                ]
            }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(context_sources.subprocess, "run", fake_run)

    sources = collect_gws_provenance(
        sample_transcript(),
        sample_readout(),
        config=GwsProvenanceConfig(enabled=True, drive_page_size=3),
    )

    assert [source.source_type for source in sources] == [
        "gws_calendar_event_detail",
        "gws_docs_file",
    ]
    assert sources[0].source_id == "event-1"
    assert sources[1].source_id == "file-1"
    assert commands[0][:4] == ["gws", "calendar", "events", "get"]
    assert commands[1][:4] == ["gws", "drive", "files", "list"]


def test_route_transcript_includes_gws_sources(monkeypatch, tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    output_dir = tmp_path / "routes"
    transcript_path.write_text(json.dumps(sample_transcript()), encoding="utf-8")
    readout_path.write_text(json.dumps(sample_readout()), encoding="utf-8")

    monkeypatch.setattr(
        route_transcript,
        "collect_gws_provenance",
        lambda transcript, readout, *, config: [
            context_sources.ProvenanceSource(
                source_type="gws_drive_file",
                source_id="file-1",
                label="Tempo Chemical NDA",
            )
        ],
    )

    route_path, review_path = route_transcript.generate_route(
        route_transcript.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                "--output-dir",
                str(output_dir),
                "--gws-provenance",
            ]
        )
    )

    payload = json.loads(route_path.read_text(encoding="utf-8"))

    assert review_path is None
    assert "file-1" in payload["selected_candidate"]["provenance_source_ids"]
    assert any(
        source["source_type"] == "gws_drive_file" for source in payload["provenance_pack"]["sources"]
    )


def test_collect_graphiti_provenance_converts_discovery_payload(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, text, capture_output, timeout, check):
        commands.append(command)
        payload = {
            "facts": [
                {
                    "uuid": "fact-1",
                    "name": "RELATES_TO",
                    "fact_preview": "SoyLei Tempo Chemical matter relates to APEX work.",
                    "episodes": ["episode-1"],
                }
            ],
            "nodes": [{"uuid": "node-1", "name": "SoyLei Tempo Chemical matter", "summary_preview": "Matter node"}],
            "episodes": [
                {
                    "uuid": "episode-1",
                    "name": "SoyLei Tempo note",
                    "content_preview": "Reviewed note",
                    "source_description": "curated test fixture",
                }
            ],
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(context_sources.subprocess, "run", fake_run)

    sources = collect_graphiti_provenance(
        sample_transcript(),
        sample_readout(),
        config=GraphitiProvenanceConfig(enabled=True, group_ids=("matter_memory",), command="graphiti-runtime"),
    )

    assert [source.source_type for source in sources] == [
        "graphiti_fact",
        "graphiti_node",
        "graphiti_episode",
    ]
    assert sources[0].metadata["group_id"] == "matter_memory"
    assert sources[0].metadata["episodes"] == ["episode-1"]
    assert sources[1].metadata["candidate_label"] == "SoyLei Tempo Chemical matter"
    assert commands[0][:4] == ["graphiti-runtime", "discover", "--group-id", "matter_memory"]


def test_build_graphiti_query_uses_compact_context_not_raw_text() -> None:
    query = build_graphiti_query(
        {
            **sample_transcript(),
            "text": "raw transcript body should not be copied into the graph query",
            "event": {
                **sample_transcript()["event"],
                "participants": ["eric@soylei.com", "paul@tempochem.com"],
            },
        },
        sample_readout(),
    )

    assert "Soylei and Tempo Chemical Technical discussion" in query
    assert "SoyLei Tempo Chemical technical collaboration" in query
    assert "soylei" in query.lower()
    assert "raw transcript body" not in query


def test_build_odollo_terms_uses_participants_not_raw_transcript() -> None:
    query_terms = build_odollo_terms(
        {
            **sample_transcript(),
            "text": "raw transcript body should not be copied into an Odoo query",
            "event": {
                **sample_transcript()["event"],
                "participants": ["eric@soylei.com", "paul@tempochem.com"],
            },
        },
        sample_readout(),
    )

    assert "Soylei" in query_terms
    assert "Tempo" in query_terms
    assert "soylei" in [term.lower() for term in query_terms]
    assert "raw" not in [term.lower() for term in query_terms]


def test_filter_provenance_sources_excludes_weak_non_calendar_sources() -> None:
    sources = [
        context_sources.ProvenanceSource(
            source_type="gws_calendar_overlap",
            source_id="event-1",
            label="Shared calendar",
        ),
        context_sources.ProvenanceSource(
            source_type="odollo_contact",
            source_id="contact-1",
            label="Paul Tempo | Tempo Chemical",
        ),
        context_sources.ProvenanceSource(
            source_type="odollo_contact",
            source_id="contact-2",
            label="Bill Leightner | ACS Technical Products",
        ),
    ]

    retained, excluded, warnings = filter_provenance_sources(
        sources,
        transcript=sample_transcript(),
        readout=sample_readout(),
        min_score=2,
    )

    assert [source.source_id for source in retained] == ["event-1", "contact-1"]
    assert [source.source_id for source in excluded] == ["contact-2"]
    assert retained[1].metadata["quality_status"] == "included"
    assert "odollo_contact_identity" in retained[1].metadata["quality_reason"]
    assert excluded[0].metadata["quality_status"] == "excluded_low_quality"
    assert warnings == ["Excluded 1 provenance source(s) below quality threshold 2."]


def test_filter_provenance_sources_ignores_retrieval_control_metadata() -> None:
    sources = [
        context_sources.ProvenanceSource(
            source_type="graphiti_fact",
            source_id="fact-1",
            label="HAS_PLAN_LOCATION",
            snippet="Repository planning surface.",
            metadata={
                "query": "Soylei Tempo Chemical paul tempochem asphalt",
                "candidate_label": "Repository planning surface",
            },
        ),
        context_sources.ProvenanceSource(
            source_type="gws_docs_file",
            source_id="file-1",
            label="Tempo Chemical NDA",
            snippet="Tempo Chemical NDA",
            metadata={"query": "Soylei Tempo Chemical paul tempochem asphalt"},
        ),
    ]

    retained, excluded, _ = filter_provenance_sources(
        sources,
        transcript=sample_transcript(),
        readout=sample_readout(),
        min_score=2,
    )

    assert [source.source_id for source in retained] == ["file-1"]
    assert [source.source_id for source in excluded] == ["fact-1"]
    assert excluded[0].metadata["quality_matched_terms"] == []
    assert "drive_file_identity" in retained[0].metadata["quality_reason"]


def test_route_transcript_includes_graphiti_sources_and_candidates(monkeypatch, tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    output_dir = tmp_path / "routes"
    transcript_path.write_text(json.dumps(sample_transcript()), encoding="utf-8")
    readout_path.write_text(json.dumps({"matter_candidates": []}), encoding="utf-8")

    monkeypatch.setattr(route_transcript, "collect_gws_provenance", lambda transcript, readout, *, config: [])
    monkeypatch.setattr(
        route_transcript,
        "collect_graphiti_provenance",
        lambda transcript, readout, *, config: [
            context_sources.ProvenanceSource(
                source_type="graphiti_node",
                source_id="node-1",
                label="SoyLei Tempo Chemical matter",
                snippet="Graphiti advisory matter node",
                metadata={
                    "group_id": "matter_memory",
                    "candidate_label": "SoyLei Tempo Chemical matter",
                    "candidate_confidence": 0.56,
                },
            )
        ],
    )

    route_path, review_path = route_transcript.generate_route(
        route_transcript.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                "--output-dir",
                str(output_dir),
                "--graphiti-provenance",
            ]
        )
    )

    payload = json.loads(route_path.read_text(encoding="utf-8"))

    assert review_path is not None
    assert payload["selected_candidate"]["source"] == "graphiti_node"
    assert payload["selected_candidate"]["label"] == "SoyLei Tempo Chemical matter"
    assert any(source["source_type"] == "graphiti_node" for source in payload["provenance_pack"]["sources"])


def test_collect_odollo_provenance_converts_contacts_and_log_notes(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, text, capture_output, timeout, check, cwd):
        commands.append(command)
        if "res.partner" in command:
            payload = [
                {
                    "id": 42,
                    "name": "Paul Tempo",
                    "email": "paul@tempochem.com",
                    "company_name": "Tempo Chemical",
                    "parent_id": [7, "Tempo Chemical"],
                }
            ]
        else:
            payload = [
                {
                    "id": 99,
                    "subject": "Tempo follow-up",
                    "body": "<p>Private note body should not be emitted as snippet.</p>",
                    "model": "crm.lead",
                    "res_id": 123,
                    "date": "2026-05-10 12:00:00",
                    "author_id": [1, "Eric Cochran"],
                }
            ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(context_sources.subprocess, "run", fake_run)

    sources = collect_odollo_provenance(
        sample_transcript(),
        sample_readout(),
        config=OdolloProvenanceConfig(
            enabled=True,
            profiles=("soylei-prod",),
            command=("odollo",),
            limit=2,
        ),
    )

    assert [source.source_type for source in sources] == ["odollo_contact", "odollo_log_note"]
    assert sources[0].metadata["profile"] == "soylei-prod"
    assert sources[0].metadata["model"] == "res.partner"
    assert sources[1].metadata["related_model"] == "crm.lead"
    assert "Private note body" not in sources[1].snippet
    assert "--profile" in commands[0]
    assert "soylei-prod" in commands[0]


def test_route_transcript_includes_odollo_sources(monkeypatch, tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    output_dir = tmp_path / "routes"
    transcript_path.write_text(json.dumps(sample_transcript()), encoding="utf-8")
    readout_path.write_text(json.dumps(sample_readout()), encoding="utf-8")

    monkeypatch.setattr(route_transcript, "collect_gws_provenance", lambda transcript, readout, *, config: [])
    monkeypatch.setattr(route_transcript, "collect_graphiti_provenance", lambda transcript, readout, *, config: [])
    monkeypatch.setattr(
        route_transcript,
        "collect_odollo_provenance",
        lambda transcript, readout, *, config: [
            context_sources.ProvenanceSource(
                source_type="odollo_contact",
                source_id="odoo-contact-42",
                label="Paul Tempo | Tempo Chemical",
            )
        ],
    )

    route_path, review_path = route_transcript.generate_route(
        route_transcript.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                "--output-dir",
                str(output_dir),
                "--odollo-provenance",
            ]
        )
    )

    payload = json.loads(route_path.read_text(encoding="utf-8"))

    assert review_path is None
    assert "odoo-contact-42" in payload["selected_candidate"]["provenance_source_ids"]
    assert any(source["source_type"] == "odollo_contact" for source in payload["provenance_pack"]["sources"])


def test_route_transcript_records_excluded_low_quality_sources(monkeypatch, tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    output_dir = tmp_path / "routes"
    transcript_path.write_text(json.dumps(sample_transcript()), encoding="utf-8")
    readout_path.write_text(json.dumps(sample_readout()), encoding="utf-8")

    monkeypatch.setattr(route_transcript, "collect_gws_provenance", lambda transcript, readout, *, config: [])
    monkeypatch.setattr(route_transcript, "collect_graphiti_provenance", lambda transcript, readout, *, config: [])
    monkeypatch.setattr(
        route_transcript,
        "collect_odollo_provenance",
        lambda transcript, readout, *, config: [
            context_sources.ProvenanceSource(
                source_type="odollo_contact",
                source_id="odoo-contact-42",
                label="Paul Tempo | Tempo Chemical",
            ),
            context_sources.ProvenanceSource(
                source_type="odollo_contact",
                source_id="odoo-contact-99",
                label="Bill Leightner | ACS Technical Products",
            ),
        ],
    )

    route_path, _ = route_transcript.generate_route(
        route_transcript.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                "--output-dir",
                str(output_dir),
                "--odollo-provenance",
            ]
        )
    )

    payload = json.loads(route_path.read_text(encoding="utf-8"))

    assert "odoo-contact-42" in payload["selected_candidate"]["provenance_source_ids"]
    assert "odoo-contact-99" not in payload["selected_candidate"]["provenance_source_ids"]
    assert payload["provenance_pack"]["excluded_sources"][0]["source_id"] == "odoo-contact-99"
    assert payload["warnings"] == ["Excluded 1 provenance source(s) below quality threshold 2."]
