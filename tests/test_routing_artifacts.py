from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import route_transcript
from routing_artifacts import build_route_decision, provenance_from_transcript


def sample_transcript() -> dict:
    return {
        "event": {
            "summary": "Soylei and Tempo Chemical Technical discussion",
            "participants": ["eric.cochran@soylei.com", "paul@tempochem.com"],
            "matching_calendars": [
                {
                    "calendar_id": "primary",
                    "calendar_summary": "Eric - SoyLei",
                    "event_id": "event-1",
                    "event_summary": "Soylei and Tempo Chemical Technical discussion",
                    "coverage": 0.93,
                    "access_role": "owner",
                }
            ],
        }
    }


def sample_readout(confidence: float = 0.95) -> dict:
    return {
        "title": "SoyLei Tempo Readout",
        "matter_candidates": [
            {
                "label": "SoyLei / Tempo Chemical technical collaboration",
                "confidence": confidence,
                "evidence": "Calendar title and transcript discuss samples and NDA/MTA.",
            },
            {
                "label": "Concrete sealer evaluation",
                "confidence": 0.63,
                "evidence": "1133 concrete sealer sample was discussed.",
            },
        ],
    }


def test_provenance_pack_extracts_matching_calendars() -> None:
    pack = provenance_from_transcript(sample_transcript())
    payload = pack.to_dict()

    assert payload["schema_version"] == 1
    assert [source["source_type"] for source in payload["sources"]] == [
        "calendar_event",
        "gws_calendar_overlap",
    ]
    assert payload["sources"][1]["label"] == "Eric - SoyLei"
    assert payload["sources"][1]["source_id"] == "event-1"


def test_route_decision_selects_high_confidence_candidate(tmp_path: Path) -> None:
    decision = build_route_decision(
        transcript_path=tmp_path / "meeting.transcript.json",
        readout_path=tmp_path / "meeting.readout.json",
        transcript=sample_transcript(),
        readout=sample_readout(),
        confidence_threshold=0.8,
    )
    payload = decision.to_dict()

    assert payload["status"] == "selected"
    assert payload["review_required"] is False
    assert payload["selected_candidate"]["label"] == "SoyLei / Tempo Chemical technical collaboration"
    assert payload["rejected_candidates"][0]["rejected_reason"] == "lower confidence than selected candidate"
    assert payload["selected_candidate"]["provenance_source_ids"]
    assert len(payload["selected_candidate"]["provenance_source_ids"]) == 2


def test_route_decision_requires_review_for_low_confidence(tmp_path: Path) -> None:
    decision = build_route_decision(
        transcript_path=tmp_path / "meeting.transcript.json",
        readout_path=tmp_path / "meeting.readout.json",
        transcript=sample_transcript(),
        readout=sample_readout(confidence=0.72),
        confidence_threshold=0.8,
    )
    payload = decision.to_dict()

    assert payload["status"] == "review_required"
    assert payload["review_required"] is True
    assert payload["fallback"] == "local_review_queue"


def test_route_transcript_writes_route_and_review_queue(tmp_path: Path) -> None:
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path = tmp_path / "meeting.readout.json"
    output_dir = tmp_path / "routes"
    review_dir = tmp_path / "review"
    transcript_path.write_text(json.dumps(sample_transcript()), encoding="utf-8")
    readout_path.write_text(json.dumps(sample_readout(confidence=0.6)), encoding="utf-8")

    route_path, review_path = route_transcript.generate_route(
        route_transcript.parse_args(
            [
                str(transcript_path),
                str(readout_path),
                "--output-dir",
                str(output_dir),
                "--review-queue-dir",
                str(review_dir),
            ]
        )
    )

    assert route_path.exists()
    assert review_path is not None
    assert review_path.exists()
    route_payload = json.loads(route_path.read_text(encoding="utf-8"))
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert route_payload["review_required"] is True
    assert review_payload["route_decision_path"] == str(route_path)
