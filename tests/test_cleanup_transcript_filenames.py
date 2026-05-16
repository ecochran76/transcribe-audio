from __future__ import annotations

import json
from pathlib import Path

from cleanup_transcript_filenames import (
    apply_plan,
    build_review_payload,
    content_difference_summary,
    content_equivalent,
    plan_artifact_cleanup,
    resolve_identical_conflict_plan,
    rewrite_state_file,
    write_review_file,
)


def write_artifact(root: Path) -> Path:
    duplicate_base = (
        "2026-04-10 10-00 M&Q - UT Knoxville NSF CAREER Proposal Debrief - Dustin Gilmer "
        "2026-04-10 10-00 M&Q - UT Knoxville NSF CAREER Proposal Debrief - Dustin Gilmer "
        "My recording 88"
    )
    clean_base = "2026-04-10 10-00 M&Q - UT Knoxville NSF CAREER Proposal Debrief - Dustin Gilmer My recording 88"
    media_path = root / f"{duplicate_base}.m4a"
    media_path.write_bytes(b"audio")
    docx_path = root / f"{duplicate_base} Transcript.docx"
    txt_path = root / f"{duplicate_base} Transcript.txt"
    artifact_path = root / f"{duplicate_base} Transcript.transcript.json"
    docx_path.write_bytes(b"docx")
    txt_path.write_text("transcript", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "source_media_path": str(root / f"{clean_base}.m4a"),
        "working_media_path": str(media_path),
        "backend": "assembly",
        "duration_seconds": 10.0,
        "recording_start": "2026-04-10T10:00:00-05:00",
        "recording_end": "2026-04-10T10:00:10-05:00",
        "transcript_window_start_seconds": 0.0,
        "transcript_window_end_seconds": 10.0,
        "utterance_count": 0,
        "transcript_text": "",
        "utterances": [],
        "event": {
            "start": "2026-04-10T10:00:00-05:00",
            "summary": "M&Q - UT Knoxville | NSF CAREER Proposal Debrief - Dustin Gilmer",
        },
        "output_paths": {
            "docx": str(docx_path),
            "txt": str(txt_path),
            "artifact": str(artifact_path),
        },
        "transcript_title": "AssemblyAI Transcript",
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def test_plan_artifact_cleanup_uses_event_canonical_name(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)

    plan = plan_artifact_cleanup(artifact_path)

    assert plan.skipped is False
    assert plan.clean_base_name == (
        "2026-04-10 10-00 M&Q - UT Knoxville NSF CAREER Proposal Debrief - Dustin Gilmer My recording 88"
    )
    assert {operation.role for operation in plan.operations} == {
        "media",
        "output:artifact",
        "output:docx",
        "output:txt",
    }


def test_apply_plan_rewrites_artifact_paths_and_state(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    plan = plan_artifact_cleanup(artifact_path)
    old_media_path = next(operation.old_path for operation in plan.operations if operation.role == "media")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "jobs": {
                    "downloads": {
                        "processed": {
                            str(old_media_path): {
                                "status": "success",
                                "artifact_paths": [str(artifact_path)],
                            }
                        },
                        "candidates": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    apply_plan(plan)
    rewrite_state_file(state_path, plan.replacements)

    new_artifact = Path(plan.replacements[str(artifact_path)])
    payload = json.loads(new_artifact.read_text(encoding="utf-8"))
    assert new_artifact.exists()
    assert Path(payload["working_media_path"]).name == (
        "2026-04-10 10-00 M&Q - UT Knoxville NSF CAREER Proposal Debrief - Dustin Gilmer My recording 88.m4a"
    )
    assert payload["output_paths"]["artifact"] == str(new_artifact)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    processed = state["jobs"]["downloads"]["processed"]
    assert str(old_media_path) not in processed
    assert payload["working_media_path"] in processed
    assert processed[payload["working_media_path"]]["artifact_paths"] == [str(new_artifact)]


def test_plan_skips_media_target_that_still_has_cleanup_noise(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    noisy_media = tmp_path / (
        "2026-05-13 13-00 Kiddie training and 1 other(s) "
        "2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129.m4a"
    )
    noisy_media.write_bytes(b"audio")
    payload["source_media_path"] = str(noisy_media)
    payload["working_media_path"] = str(noisy_media)
    payload["event"] = {
        "start": "2026-05-13T14:00:00-05:00",
        "summary": "Meet with Eric (Arjun B)",
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plan = plan_artifact_cleanup(artifact_path)

    assert not any(operation.role == "media" for operation in plan.operations)
    assert plan.reason == "canonical media target still contains cleanup noise"


def test_review_payload_describes_skipped_conflicts(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    plan = plan_artifact_cleanup(artifact_path)
    plan.skipped = True
    plan.reason = f"target already exists: {tmp_path / 'target.docx'}"

    payload = build_review_payload([plan])

    assert payload["schema_version"] == 1
    assert payload["summary"]["review_count"] == 1
    assert payload["items"][0]["artifact_path"] == str(artifact_path)
    assert payload["items"][0]["suggested_action"] == "compare duplicate artifact/media contents before merge or deletion"
    assert payload["items"][0]["event"]["summary"] == "M&Q - UT Knoxville | NSF CAREER Proposal Debrief - Dustin Gilmer"


def test_write_review_file_creates_parent_directory(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    plan = plan_artifact_cleanup(artifact_path)
    plan.skipped = True
    plan.reason = "shared media has conflicting canonical targets"
    review_path = tmp_path / "reviews" / "cleanup-review.json"

    written_path = write_review_file(review_path, [plan])

    assert written_path == review_path
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["items"][0]["suggested_action"] == "choose the canonical media title for shared overlapping calendar artifacts"


def test_content_equivalent_compares_artifact_transcript_text(tmp_path: Path) -> None:
    old_path = tmp_path / "old.transcript.json"
    new_path = tmp_path / "new.transcript.json"
    old_path.write_text(json.dumps({"transcript_text": "same text", "output_paths": {"artifact": str(old_path)}}), encoding="utf-8")
    new_path.write_text(
        json.dumps({"transcript_text": "same text", "output_paths": {"artifact": str(new_path)}, "source_media_path": "different"}),
        encoding="utf-8",
    )

    assert content_equivalent(old_path, new_path, "output:artifact") is True


def test_content_difference_summary_classifies_metadata_only_candidate(tmp_path: Path) -> None:
    old_path = tmp_path / "old.txt"
    target_path = tmp_path / "target.txt"
    old_path.write_text("Source: noisy name\nUtterance 1: Same body\nUtterance 2: Still same\n", encoding="utf-8")
    target_path.write_text("Source: clean name\nUtterance 1: Same body\nUtterance 2: Still same\n", encoding="utf-8")

    summary = content_difference_summary(old_path, target_path, "output:txt")

    assert summary["classification"] == "metadata_or_format_only_candidate"
    assert summary["body_similarity_ratio"] == 1.0
    assert summary["line_similarity_ratio"] < 1.0


def test_review_payload_can_include_diff_summary(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    Path(artifact_payload["output_paths"]["txt"]).write_text("shared line\nold detail", encoding="utf-8")
    plan = plan_artifact_cleanup(artifact_path)
    clean_txt = tmp_path / f"{plan.clean_base_name} Transcript.txt"
    clean_txt.write_text("shared line\nnew detail", encoding="utf-8")
    plan = plan_artifact_cleanup(artifact_path)

    payload = build_review_payload([plan], include_diff_summary=True)

    txt_conflict = next(
        conflict for conflict in payload["items"][0]["target_conflicts"] if conflict["role"] == "output:txt"
    )
    assert txt_conflict["content_equivalent"] is False
    assert txt_conflict["diff_summary"]["role"] == "output:txt"
    assert txt_conflict["diff_summary"]["classification"] == "partial_overlap_distinct_content"


def test_resolve_identical_conflict_quarantines_redundant_output(tmp_path: Path) -> None:
    artifact_path = write_artifact(tmp_path)
    plan = plan_artifact_cleanup(artifact_path)
    clean_artifact = tmp_path / f"{plan.clean_base_name} Transcript.transcript.json"
    clean_artifact.write_text(artifact_path.read_text(encoding="utf-8"), encoding="utf-8")
    plan = plan_artifact_cleanup(artifact_path)

    result = resolve_identical_conflict_plan(plan, quarantine_dir=tmp_path / "quarantine")

    assert result is not None
    assert result["resolved_identical_conflict"] is True
    assert not artifact_path.exists()
    assert clean_artifact.exists()
    assert Path(result["quarantined"][0]["quarantine_path"]).exists()
