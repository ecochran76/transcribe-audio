from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import evaluate_provenance_calibration as calibration
from transcribe_common import TranscriptionError


def sample_manifest() -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "test-p04-calibration",
        "review_status": "accepted",
        "cases": [
            {
                "case_id": "tempo-route",
                "transcript": {
                    "event": {
                        "summary": "Soylei and Tempo Chemical Technical discussion",
                        "participants": ["paul@tempochem.example"],
                    }
                },
                "readout": {
                    "title": "Tempo Chemical collaboration readout",
                    "topics": ["samples", "NDA", "MTA"],
                    "matter_candidates": [{"label": "SoyLei Tempo Chemical collaboration"}],
                },
                "decisions": [
                    {
                        "decision_id": "calendar-include",
                        "expected": "include",
                        "source": {
                            "source_type": "gws_calendar_overlap",
                            "source_id": "event-1",
                            "label": "Shared calendar",
                        },
                    },
                    {
                        "decision_id": "drive-include",
                        "expected": "include",
                        "source": {
                            "source_type": "gws_docs_file",
                            "source_id": "file-1",
                            "label": "Tempo Chemical NDA",
                            "snippet": "Tempo Chemical NDA and MTA notes",
                        },
                    },
                    {
                        "decision_id": "graphiti-exclude",
                        "expected": "exclude",
                        "source": {
                            "source_type": "graphiti_fact",
                            "source_id": "fact-1",
                            "label": "HAS_PLAN_LOCATION",
                            "snippet": "Repository planning surface.",
                        },
                    },
                    {
                        "decision_id": "odollo-include",
                        "expected": "include",
                        "source": {
                            "source_type": "odollo_contact",
                            "source_id": "contact-1",
                            "label": "Paul Tempo | Tempo Chemical",
                        },
                    },
                ],
            }
        ],
    }


def test_evaluate_manifest_reports_passes_and_source_families() -> None:
    results = calibration.evaluate_manifest(sample_manifest(), min_score=2)
    report = calibration.build_report(
        manifest_paths=[Path("manifest.json")],
        results=results,
        min_score=2,
        include_passed=True,
    )

    assert report["quality_profile"]["profile_id"] == "p04-source-quality-v1"
    assert report["totals"]["decisions"] == 4
    assert report["totals"].get("false_positive", 0) == 0
    assert report["source_families"]["calendar"]["expected_include"] == 1
    assert report["source_families"]["drive_docs"]["expected_include"] == 1
    assert report["source_families"]["graphiti"]["expected_exclude"] == 1
    assert report["source_families"]["odollo"]["expected_include"] == 1


def test_run_writes_sanitized_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(sample_manifest()), encoding="utf-8")

    report = calibration.run(
        calibration.parse_args(
            [
                str(manifest_path),
                "--output",
                str(report_path),
                "--fail-on-mismatch",
                "--require-decision-count",
                "4",
                "--require-source-families",
                "4",
            ]
        )
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["totals"]["decisions"] == 4
    assert payload["mismatches"] == []
    assert "Tempo Chemical NDA and MTA notes" not in report_path.read_text(encoding="utf-8")


def test_manifest_rejects_raw_transcript_fields() -> None:
    manifest = sample_manifest()
    manifest["cases"][0]["transcript"]["transcript_text"] = "raw body"

    try:
        calibration.evaluate_manifest(manifest, min_score=2)
    except TranscriptionError as exc:
        assert "forbidden raw transcript field" in str(exc)
    else:
        raise AssertionError("Expected TranscriptionError")
