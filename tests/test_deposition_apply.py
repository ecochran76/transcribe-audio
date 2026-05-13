from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deposition_apply
from transcribe_common import TranscriptionError


def write_preview(path: Path, *, target_root: Path, source_paths: list[Path], review_required: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_readout_path": str(source_paths[0]),
                "selected_candidate": {
                    "label": "SoyLei Tempo Matter",
                    "target_id": "matter-1",
                },
                "review_required": review_required,
                "actions": [
                    {
                        "action_type": "copy_artifacts",
                        "target_kind": "local_filesystem",
                        "target_id": str(target_root),
                        "status": "preview",
                        "source_paths": [str(path) for path in source_paths],
                        "metadata": {"writes_enabled": False},
                    },
                    {
                        "action_type": "upload_artifacts",
                        "target_kind": "google_drive",
                        "target_id": "drive-folder-1",
                        "status": "preview",
                        "source_paths": [str(source_paths[0])],
                        "metadata": {"writes_enabled": False},
                    },
                ],
                "memory_candidates": [],
            }
        ),
        encoding="utf-8",
    )


def test_apply_preview_copies_local_files_and_skips_nonlocal(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.contextual.readout.json"
    route_path = tmp_path / "meeting.route.json"
    readout_path.write_text('{"summary":"readout"}\n', encoding="utf-8")
    route_path.write_text('{"status":"selected"}\n', encoding="utf-8")
    target_root = tmp_path / "deposits"
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, target_root=target_root, source_paths=[readout_path, route_path])

    result_path = deposition_apply.apply_preview(deposition_apply.parse_args([str(preview_path)]))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    local_action = payload["actions"][0]
    drive_action = payload["actions"][1]

    assert result_path.name == "meeting.deposit-apply.json"
    assert local_action["target_kind"] == "local_filesystem"
    assert local_action["status"] == "applied"
    assert len(local_action["files"]) == 2
    assert all(Path(item["destination_path"]).exists() for item in local_action["files"])
    assert drive_action["status"] == "skipped"
    assert drive_action["metadata"]["writes_enabled"] is False


def test_apply_preview_is_idempotent_for_same_hash(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.readout.json"
    readout_path.write_text('{"summary":"readout"}\n', encoding="utf-8")
    target_root = tmp_path / "deposits"
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, target_root=target_root, source_paths=[readout_path])

    first_result = deposition_apply.apply_preview(deposition_apply.parse_args([str(preview_path)]))
    first_payload = json.loads(first_result.read_text(encoding="utf-8"))
    second_result = deposition_apply.apply_preview(deposition_apply.parse_args([str(preview_path)]))
    second_payload = json.loads(second_result.read_text(encoding="utf-8"))

    assert first_payload["actions"][0]["files"][0]["status"] == "copied"
    assert second_payload["actions"][0]["files"][0]["status"] == "skipped_existing_same_hash"


def test_apply_preview_versions_conflicting_destinations(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.readout.json"
    readout_path.write_text('{"summary":"readout"}\n', encoding="utf-8")
    target_root = tmp_path / "deposits"
    target_dir = target_root / "soylei-tempo-matter"
    target_dir.mkdir(parents=True)
    destination = target_dir / readout_path.name
    destination.write_text("different content\n", encoding="utf-8")
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, target_root=target_root, source_paths=[readout_path])

    result_path = deposition_apply.apply_preview(deposition_apply.parse_args([str(preview_path)]))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    copied_file = payload["actions"][0]["files"][0]

    assert copied_file["status"] == "copied_versioned"
    assert copied_file["destination_path"].endswith("meeting.v2.readout.json")
    assert Path(copied_file["destination_path"]).exists()


def test_apply_preview_refuses_review_required_without_override(tmp_path: Path) -> None:
    readout_path = tmp_path / "meeting.readout.json"
    readout_path.write_text('{"summary":"readout"}\n', encoding="utf-8")
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, target_root=tmp_path / "deposits", source_paths=[readout_path], review_required=True)

    with pytest.raises(TranscriptionError, match="requires review"):
        deposition_apply.apply_preview(deposition_apply.parse_args([str(preview_path)]))

    result_path = deposition_apply.apply_preview(
        deposition_apply.parse_args([str(preview_path), "--allow-review-required"])
    )
    assert result_path.exists()
