from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def write_apply_template(path: Path, *, decision: str = "quarantine_old") -> tuple[Path, Path, Path]:
    old_conflict = path.parent / "old.docx"
    target_conflict = path.parent / "target.docx"
    artifact = path.parent / "old.transcript.json"
    old_txt = path.parent / "old.txt"
    new_txt = path.parent / "new.txt"
    old_conflict.write_text("old docx", encoding="utf-8")
    target_conflict.write_text("target docx", encoding="utf-8")
    old_txt.write_text("old txt", encoding="utf-8")
    artifact.write_text(json.dumps({"output_paths": {"txt": str(old_txt), "docx": str(old_conflict)}}), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {
                        "id": "item-1",
                        "decision": decision,
                        "clean_base_name": "Clean Meeting",
                        "artifact_path": str(artifact),
                        "conflicts": [
                            {
                                "role": "output:docx",
                                "old_path": str(old_conflict),
                                "target_path": str(target_conflict),
                            }
                        ],
                        "planned_non_conflicting_operations": [
                            {"role": "output:txt", "old_path": str(old_txt), "new_path": str(new_txt)}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return old_conflict, target_conflict, old_txt


def test_apply_review_item_dry_run_does_not_move_files(tmp_path: Path) -> None:
    template_path = tmp_path / "review.json"
    old_conflict, _target_conflict, old_txt = write_apply_template(template_path)
    item = json.loads(template_path.read_text(encoding="utf-8"))["items"][0]

    result = review.apply_review_item(item, quarantine_dir=tmp_path / "quarantine", apply=False)

    assert result["status"] == "planned"
    assert result["dry_run"] is True
    assert old_conflict.exists()
    assert old_txt.exists()
    assert result["quarantined"][0]["quarantine_path"]


def test_apply_review_item_preserve_both_is_noop(tmp_path: Path) -> None:
    template_path = tmp_path / "review.json"
    old_conflict, _target_conflict, _old_txt = write_apply_template(template_path, decision="preserve_both")
    item = json.loads(template_path.read_text(encoding="utf-8"))["items"][0]

    result = review.apply_review_item(item, quarantine_dir=tmp_path / "quarantine", apply=True)

    assert result["status"] == "recorded_noop"
    assert old_conflict.exists()


def test_apply_review_item_quarantines_and_moves(tmp_path: Path) -> None:
    template_path = tmp_path / "review.json"
    old_conflict, _target_conflict, old_txt = write_apply_template(template_path)
    item = json.loads(template_path.read_text(encoding="utf-8"))["items"][0]

    result = review.apply_review_item(item, quarantine_dir=tmp_path / "quarantine", apply=True)

    assert result["status"] == "applied"
    assert not old_conflict.exists()
    assert not old_txt.exists()
    assert Path(result["quarantined"][0]["quarantine_path"]).exists()
    assert (tmp_path / "new.txt").exists()


def test_apply_review_template_requires_approval_for_apply(tmp_path: Path) -> None:
    template_path = tmp_path / "review.json"
    write_apply_template(template_path)
    args = review.parse_args(["--apply-review", str(template_path), "--apply"])

    with pytest.raises(ValueError, match="approval-token"):
        review.apply_review_template(args)


def test_apply_review_template_writes_audit_output(tmp_path: Path) -> None:
    template_path = tmp_path / "review.json"
    audit_path = tmp_path / "audit.json"
    write_apply_template(template_path, decision="preserve_both")
    args = review.parse_args(["--apply-review", str(template_path), "--audit-output", str(audit_path)])

    result = review.apply_review_template(args)

    assert result["audit_path"] == str(audit_path.resolve())
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["summary"]["by_status"]["recorded_noop"] == 1
    assert audit["results"][0]["decision"] == "preserve_both"
