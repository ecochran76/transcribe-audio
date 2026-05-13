from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_harvest_apply
from transcribe_common import TranscriptionError


def write_preview(path: Path, *, review_required: bool = False, warnings: list[str] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_readout_path": "/tmp/meeting.contextual.readout.json",
                "source_route_path": "/tmp/meeting.route.json",
                "selected_candidate": {
                    "label": "SoyLei Tempo matter",
                    "target_kind": "matter",
                    "target_id": "matter-1",
                    "confidence": 0.95,
                },
                "review_required": review_required,
                "warnings": warnings or [],
                "memory_candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "status": "preview",
                        "target_group_id": "transcribe_audio_main",
                        "kind": "matter_fact",
                        "text": "Tempo Chemical is evaluating SoyLei technical samples.",
                        "evidence": "structured readout",
                        "source_readout_path": "/tmp/meeting.contextual.readout.json",
                        "source_ids": ["event-1"],
                    },
                    {
                        "candidate_id": "candidate-2",
                        "status": "preview",
                        "target_group_id": "transcribe_audio_main",
                        "kind": "matter_fact",
                        "text": "Tempo Chemical requested small samples.",
                        "evidence": "structured readout",
                        "source_readout_path": "/tmp/meeting.contextual.readout.json",
                        "source_ids": ["event-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_memory_harvest_preview_plans_without_writing_body_file(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args([str(preview_path), "--output-dir", str(tmp_path / "out")])
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]

    assert result_path.name == "meeting.memory-harvest-apply.json"
    assert payload["mode"] == "preview"
    assert [candidate["status"] for candidate in payload["candidates"]] == ["planned", "planned"]
    assert "<BODY_FILE_CREATED_ON_APPLY>" in candidate["graphiti_command"]
    assert candidate["duplicate_check"]["status"] == "planned"


def test_memory_harvest_init_review_writes_pending_decisions(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args([str(preview_path), "--init-review", "--output-dir", str(tmp_path / "out")])
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path.name == "meeting.memory-harvest-review.json"
    assert payload["source"] == "transcribe-audio.memory_harvest_review.v1"
    assert [item["candidate_id"] for item in payload["candidates"]] == ["candidate-1", "candidate-2"]
    assert {item["decision"] for item in payload["candidates"]} == {"pending"}


def test_memory_harvest_refuses_review_required_without_override(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, review_required=True)

    with pytest.raises(TranscriptionError, match="requires review"):
        memory_harvest_apply.apply_preview(memory_harvest_apply.parse_args([str(preview_path)]))

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args([str(preview_path), "--allow-review-required"])
    )
    assert result_path.exists()


def test_memory_harvest_refuses_warnings_without_override(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path, warnings=["Excluded 1 source."])

    with pytest.raises(TranscriptionError, match="carries warnings"):
        memory_harvest_apply.apply_preview(memory_harvest_apply.parse_args([str(preview_path)]))

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args([str(preview_path), "--allow-warnings"])
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["warnings"] == ["Excluded 1 source."]


def test_memory_harvest_apply_requires_approval_token(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)

    with pytest.raises(TranscriptionError, match="approval-token"):
        memory_harvest_apply.apply_preview(memory_harvest_apply.parse_args([str(preview_path), "--apply"]))


def test_memory_harvest_apply_runs_graphiti_command(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        if command[1] == "discover":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "command": "discover",
                        "status": {"status": "ok"},
                        "episode_count": 0,
                        "fact_count": 0,
                        "node_count": 0,
                        "episodes": [],
                    }
                ),
                stderr="",
            )
        body_path = Path(command[command.index("--body-file") + 1])
        body = json.loads(body_path.read_text(encoding="utf-8"))
        assert body["source"] == "transcribe-audio.memory_harvest.v1"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "command": "benchmark-write",
                    "group_id": "transcribe_audio_main",
                    "status": "completed",
                    "job_id": "job-1",
                    "episode_uuid": "episode-1",
                    "verified": True,
                }
            ),
            stderr="",
        )

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args(
            [
                str(preview_path),
                "--apply",
                "--approval-token",
                memory_harvest_apply.APPROVAL_TOKEN,
            ]
        ),
        runner=fake_runner,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert calls
    assert payload["mode"] == "apply"
    assert payload["candidates"][0]["status"] == "applied"
    assert payload["candidates"][0]["graphiti_result"]["episode_uuid"] == "episode-1"
    assert payload["candidates"][0]["duplicate_check"]["exact_duplicate"] is False


def test_memory_harvest_review_file_applies_only_approved_candidates(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    review_path = tmp_path / "meeting.memory-harvest-review.json"
    write_preview(preview_path)
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_preview_path": str(preview_path),
                "candidates": [
                    {"candidate_id": "candidate-1", "decision": "approved", "reason": "durable matter context"},
                    {"candidate_id": "candidate-2", "decision": "rejected", "reason": "too low signal"},
                ],
            }
        ),
        encoding="utf-8",
    )
    benchmark_calls = []

    def fake_runner(command, **kwargs):
        if command[1] == "discover":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"status": {"status": "ok"}, "episodes": [], "episode_count": 0}),
                stderr="",
            )
        benchmark_calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"command": "benchmark-write", "episode_uuid": "episode-1"}),
            stderr="",
        )

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args(
            [
                str(preview_path),
                "--review-file",
                str(review_path),
                "--apply",
                "--approval-token",
                memory_harvest_apply.APPROVAL_TOKEN,
            ]
        ),
        runner=fake_runner,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(benchmark_calls) == 1
    assert [item["status"] for item in payload["candidates"]] == ["applied", "rejected"]
    assert payload["candidates"][0]["review_decision"] == "approved"
    assert payload["candidates"][1]["review_reason"] == "too low signal"


def test_memory_harvest_duplicate_check_skips_exact_candidate_replay(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)
    benchmark_calls = []

    def fake_runner(command, **kwargs):
        if command[1] == "discover":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "status": {"status": "ok"},
                        "episode_count": 1,
                        "episodes": [
                            {
                                "uuid": "episode-1",
                                "name": "transcribe-audio memory harvest candidate-1",
                                "source_description": "transcribe-audio memory harvest candidate candidate-1",
                                "content_preview": '{"candidate_id": "candidate-1"}',
                            }
                        ],
                    }
                ),
                stderr="",
            )
        benchmark_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({}), stderr="")

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args(
            [
                str(preview_path),
                "--candidate-id",
                "candidate-1",
                "--apply",
                "--approval-token",
                memory_harvest_apply.APPROVAL_TOKEN,
            ]
        ),
        runner=fake_runner,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert benchmark_calls == []
    assert payload["candidates"][0]["status"] == "duplicate_skipped"
    assert payload["candidates"][0]["duplicate_check"]["exact_duplicate"] is True


def test_memory_harvest_duplicate_check_failure_blocks_write(tmp_path: Path) -> None:
    preview_path = tmp_path / "meeting.deposit-preview.json"
    write_preview(preview_path)
    benchmark_calls = []

    def fake_runner(command, **kwargs):
        if command[1] == "discover":
            return subprocess.CompletedProcess(
                command,
                2,
                stdout=json.dumps({"status": {"status": "error"}, "error": "runtime unavailable"}),
                stderr="",
            )
        benchmark_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({}), stderr="")

    result_path = memory_harvest_apply.apply_preview(
        memory_harvest_apply.parse_args(
            [
                str(preview_path),
                "--candidate-id",
                "candidate-1",
                "--apply",
                "--approval-token",
                memory_harvest_apply.APPROVAL_TOKEN,
            ]
        ),
        runner=fake_runner,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert benchmark_calls == []
    assert payload["candidates"][0]["status"] == "duplicate_check_failed"
