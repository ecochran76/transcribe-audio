from __future__ import annotations

import json
from pathlib import Path

from cleanup_transcript_filenames import apply_plan, plan_artifact_cleanup, rewrite_state_file


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
