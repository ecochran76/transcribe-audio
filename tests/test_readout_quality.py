from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import check_readout_quality


def write_readout(path: Path, *, summary: str = "A substantive summary. " * 20) -> Path:
    source = path.with_suffix(".transcript.json")
    source.write_text(json.dumps({"transcript_text": "hello"}), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifact_path": str(source),
                "provider": {"name": "test"},
                "generated_at": "2026-05-17T00:00:00Z",
                "title": "Quality Readout",
                "summary": summary,
                "participants": [{"name": "A"}],
                "topics": ["topic"],
                "action_items": [{"task": "follow up"}],
                "matter_candidates": [{"label": "matter"}],
                "memory_candidates": [{"text": "memory", "kind": "fact"}],
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text("# Quality Readout\n", encoding="utf-8")
    return path


def test_quality_report_passes_valid_readout(tmp_path: Path) -> None:
    readout = write_readout(tmp_path / "meeting.readout.json")

    report = check_readout_quality.quality_report([readout], min_summary_chars=200)

    assert report["status"] == "pass"
    assert report["checked_count"] == 1
    assert report["pass_count"] == 1
    assert report["checks"][0]["metrics"]["memory_candidates_count"] == 1


def test_quality_report_fails_missing_markdown(tmp_path: Path) -> None:
    readout = write_readout(tmp_path / "meeting.readout.json")
    readout.with_suffix(".md").unlink()

    report = check_readout_quality.quality_report([readout], min_summary_chars=200)

    assert report["status"] == "fail"
    assert report["fail_count"] == 1
    assert any(finding["code"] == "missing_markdown" for finding in report["checks"][0]["findings"])


def test_manifest_paths_are_checked(tmp_path: Path) -> None:
    readout = write_readout(tmp_path / "meeting.readout.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"materialized": [{"readout_json": str(readout)}]}), encoding="utf-8")

    assert check_readout_quality.main(["--manifest", str(manifest), "--format", "text"]) == 0
