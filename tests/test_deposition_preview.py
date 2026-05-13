from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deposition_preview
from deposition_artifacts import DepositPreview, MemoryHarvestCandidate


def sample_readout() -> dict:
    return {
        "title": "Contextual Readout",
        "summary": "Private transcript details stay in the readout artifact.",
        "memory_candidates": [
            {
                "text": "Tempo Chemical is evaluating SoyLei technical samples.",
                "kind": "matter_fact",
                "evidence": "contextual readout and cited sources",
            }
        ],
        "contextualization": {
            "warnings": ["Excluded 1 provenance source(s) below quality threshold 2."],
            "excluded_source_count": 1,
            "supporting_context_sources": [
                {"source_type": "odollo_contact", "source_id": "contact-1", "label": "Tempo Chemical"},
                {"source_type": "gws_calendar_overlap", "source_id": "event-1", "label": "SoyLei calendar"},
            ]
        },
    }


def sample_route() -> dict:
    return {
        "status": "selected",
        "review_required": False,
        "selected_candidate": {
            "label": "SoyLei Tempo matter",
            "target_kind": "matter",
            "target_id": "matter-1",
            "confidence": 0.95,
            "source": "readout.matter_candidates",
            "provenance_source_ids": ["contact-1", "event-1"],
        },
    }


def test_memory_harvest_candidates_use_structured_readout_fields(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.contextual.readout.json"
    candidates = deposition_preview.build_memory_candidates(
        {
            **sample_readout(),
            "transcript_text": "raw transcript text must not be harvested",
        },
        readout_path=readout_path,
        graphiti_group="transcribe_audio_main",
    )

    assert len(candidates) == 1
    assert candidates[0].text == "Tempo Chemical is evaluating SoyLei technical samples."
    assert candidates[0].target_group_id == "transcribe_audio_main"
    assert candidates[0].source_ids == ["contact-1", "event-1"]
    assert "raw transcript" not in candidates[0].to_dict()["text"]


def test_deposition_preview_schema_round_trips() -> None:
    preview = DepositPreview(
        source_readout_path="/tmp/meeting.contextual.readout.json",
        memory_candidates=[
            MemoryHarvestCandidate(
                text="Reviewed memory",
                kind="matter_fact",
                evidence="structured readout",
            )
        ],
    )
    payload = preview.to_dict()

    assert payload["schema_version"] == 1
    assert payload["review_required"] is True
    assert payload["warnings"] == []
    assert payload["memory_candidates"][0]["status"] == "preview"
    assert payload["memory_candidates"][0]["candidate_id"]


def test_generate_deposition_preview_writes_no_apply_plan(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.contextual.readout.json"
    route_path = tmp_path / "meeting.route.json"
    transcript_path = tmp_path / "meeting.transcript.json"
    output_dir = tmp_path / "preview"
    readout_path.write_text(json.dumps(sample_readout()), encoding="utf-8")
    route_path.write_text(json.dumps(sample_route()), encoding="utf-8")
    transcript_path.write_text(json.dumps({"transcript_text": "private raw transcript"}), encoding="utf-8")

    output_path = deposition_preview.generate_deposition_preview(
        deposition_preview.parse_args(
            [
                str(readout_path),
                "--route",
                str(route_path),
                "--transcript",
                str(transcript_path),
                "--output-dir",
                str(output_dir),
                "--local-root",
                str(tmp_path / "deposits"),
                "--drive-folder-id",
                "drive-folder-1",
                "--drive-profile",
                "gws-default",
                "--odoo-profile",
                "soylei-prod",
                "--odoo-model",
                "crm.lead",
                "--odoo-record-id",
                "123",
            ]
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "meeting.deposit-preview.json"
    assert payload["review_required"] is False
    assert payload["selected_candidate"]["label"] == "SoyLei Tempo matter"
    assert [action["target_kind"] for action in payload["actions"]] == [
        "local_filesystem",
        "google_drive",
        "odoo_record",
    ]
    assert all(action["status"] == "preview" for action in payload["actions"])
    assert all(action["metadata"]["writes_enabled"] is False for action in payload["actions"])
    assert str(transcript_path.resolve()) not in payload["actions"][0]["source_paths"]
    assert payload["memory_candidates"][0]["source_ids"] == ["contact-1", "event-1"]
    assert payload["warnings"] == [
        "Excluded 1 provenance source(s) below quality threshold 2.",
        "1 provenance source(s) were excluded before deposition preview.",
    ]


def test_include_transcript_only_affects_deposition_actions(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.readout.json"
    route_path = tmp_path / "meeting.route.json"
    transcript_path = tmp_path / "meeting.transcript.json"
    readout_path.write_text(json.dumps(sample_readout()), encoding="utf-8")
    route_path.write_text(json.dumps(sample_route()), encoding="utf-8")
    transcript_path.write_text(json.dumps({"transcript_text": "private raw transcript"}), encoding="utf-8")

    output_path = deposition_preview.generate_deposition_preview(
        deposition_preview.parse_args(
            [
                str(readout_path),
                "--route",
                str(route_path),
                "--transcript",
                str(transcript_path),
                "--local-root",
                str(tmp_path / "deposits"),
                "--include-transcript",
            ]
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert str(transcript_path.resolve()) in payload["actions"][0]["source_paths"]
    assert "private raw transcript" not in json.dumps(payload["memory_candidates"])
