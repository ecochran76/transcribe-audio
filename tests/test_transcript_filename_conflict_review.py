from __future__ import annotations

import json
from pathlib import Path

import transcript_filename_conflict_review as review


def review_export(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"review_count": 2},
                "items": [
                    {
                        "artifact_path": "/tmp/noisy.transcript.json",
                        "clean_base_name": "Clean Meeting",
                        "reason": "target already exists: /tmp/Clean Meeting Transcript.docx",
                        "event": {"summary": "Clean Meeting", "start": "2026-05-16T10:00:00-05:00"},
                        "source_media_path": "/tmp/noisy.m4a",
                        "working_media_path": "/tmp/noisy.m4a",
                        "operations": [{"role": "output:txt", "old_path": "/tmp/noisy.txt", "new_path": "/tmp/clean.txt"}],
                        "target_conflicts": [
                            {
                                "role": "output:docx",
                                "old_path": "/tmp/noisy.docx",
                                "target_path": "/tmp/clean.docx",
                                "content_equivalent": False,
                                "old_sha256": "old",
                                "target_sha256": "target",
                                "diff_summary": {
                                    "classification": "high_overlap_needs_review",
                                    "body_similarity_ratio": 0.97,
                                    "changed_line_spans": 3,
                                },
                            }
                        ],
                    },
                    {
                        "artifact_path": "/tmp/distinct.transcript.json",
                        "clean_base_name": "Distinct Meeting",
                        "reason": "target already exists: /tmp/Distinct Transcript.txt",
                        "event": {"summary": "Distinct Meeting"},
                        "target_conflicts": [
                            {
                                "role": "output:txt",
                                "old_path": "/tmp/old.txt",
                                "target_path": "/tmp/target.txt",
                                "diff_summary": {
                                    "classification": "distinct_content_preserve_both",
                                    "body_similarity_ratio": 0.2,
                                    "changed_line_spans": 20,
                                },
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_review_template_initializes_pending_decisions(tmp_path: Path) -> None:
    source = tmp_path / "review.json"
    review_export(source)

    template = review.build_review_template(source, review.load_json(source))

    assert template["schema_version"] == 1
    assert template["summary"]["item_count"] == 2
    assert template["summary"]["by_classification"]["high_overlap_needs_review"] == 1
    assert template["summary"]["by_recommended_decision"]["needs_investigation"] == 1
    assert template["items"][0]["decision"] == "pending"
    assert template["items"][0]["recommended_decision"] == "needs_investigation"
    assert template["items"][1]["recommended_decision"] == "preserve_both"


def test_main_writes_json_and_markdown(tmp_path: Path, capsys) -> None:
    source = tmp_path / "review.json"
    output_json = tmp_path / "out" / "review-template.json"
    output_md = tmp_path / "out" / "review-template.md"
    review_export(source)

    assert review.main([str(source), "--json-output", str(output_json), "--markdown-output", str(output_md)]) == 0

    stdout = capsys.readouterr().out
    assert "Items: 2" in stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert payload["items"][0]["conflicts"][0]["diff_summary"]["body_similarity_ratio"] == 0.97
    assert "Transcript Filename Conflict Review" in markdown
    assert "high_overlap_needs_review" in markdown
