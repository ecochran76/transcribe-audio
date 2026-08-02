from __future__ import annotations

import math
import struct
import threading
import time
import wave
from pathlib import Path

import pytest

import acoustic_audio_derivatives as shared
import acoustic_speech_preparation as speech
import acoustic_training_preparation as training


def _write_wav(path: Path, duration_seconds: float) -> Path:
    sample_rate = 16_000
    frames = bytearray()
    for index in range(int(duration_seconds * sample_rate)):
        value = int(0.2 * 32_767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(bytes(frames))
    return path


def _request(operation: str, **arguments: object) -> dict[str, object]:
    return {"operation": operation, "arguments": arguments}


def test_isolated_worker_does_not_change_concurrent_shared_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_source = _write_wav(tmp_path / "training.wav", 1.0)
    shared_source = _write_wav(tmp_path / "shared.wav", 1.0)
    ready = tmp_path / "worker-ready"
    monkeypatch.setenv("TRANSCRIBE_AUDIO_TRAINING_WORKER_TEST_DELAY", "0.75")
    monkeypatch.setenv(
        "TRANSCRIBE_AUDIO_TRAINING_WORKER_TEST_READY_PATH", str(ready)
    )
    result: dict[str, object] = {}

    def run_training() -> None:
        result.update(
            training.dry_run(
                training_source, runtime_root=tmp_path / "training-runtime"
            )
        )

    thread = threading.Thread(target=run_training)
    thread.start()
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    shared_plan = shared.dry_run(
        shared_source, runtime_root=tmp_path / "shared-runtime"
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result["recipe"]["parameters"]["duration_tolerance_seconds"] == 0.1
    assert shared_plan["recipe"]["parameters"]["duration_tolerance_seconds"] == 0.05
    assert shared.DURATION_TOLERANCE_SECONDS == 0.05


def test_training_boundary_accepts_point_zero_eight_and_rejects_point_eleven(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _write_wav(tmp_path / "source.wav", 1.0)
    accepted_root = tmp_path / "accepted"
    accepted_plan = training._execute_worker(
        _request("p1_dry_run", source_audio=source, runtime_root=accepted_root)
    )
    monkeypatch.setattr(
        shared,
        "_decode",
        lambda source_path, output_path, recipe: _write_wav(output_path, 1.08),
    )
    accepted = training._execute_worker(
        _request(
            "p1_apply",
            source_audio=source,
            runtime_root=accepted_root,
            approval_token=shared.APPLY_TOKEN,
        )
    )
    assert accepted["manifest"]["derived_audio"]["timestamp_map"][0][
        "duration_tolerance_seconds"
    ] == 0.1
    assert training.replay_derivative(
        accepted_plan["run_id"], runtime_root=accepted_root
    )["active"] is True

    rejected_root = tmp_path / "rejected"
    training._execute_worker(
        _request("p1_dry_run", source_audio=source, runtime_root=rejected_root)
    )
    monkeypatch.setattr(
        shared,
        "_decode",
        lambda source_path, output_path, recipe: _write_wav(output_path, 1.11),
    )
    with pytest.raises(shared.AudioDerivativeError, match="duration drift"):
        training._execute_worker(
            _request(
                "p1_apply",
                source_audio=source,
                runtime_root=rejected_root,
                approval_token=shared.APPLY_TOKEN,
            )
        )
    run_dir = rejected_root / "runs" / training._execute_worker(
        _request("p1_dry_run", source_audio=source, runtime_root=rejected_root)
    )["run_id"]
    assert not (run_dir / "manifest.json").exists()
    assert shared.DURATION_TOLERANCE_SECONDS == 0.05


def test_existing_shared_point_zero_five_authority_still_replays(
    tmp_path: Path,
) -> None:
    source = _write_wav(tmp_path / "source.wav", 1.0)
    runtime = tmp_path / "shared"
    plan = shared.dry_run(source, runtime_root=runtime)
    shared.apply_derivative(
        source, runtime_root=runtime, approval_token=shared.APPLY_TOKEN
    )
    training.dry_run(source, runtime_root=tmp_path / "training")
    assert shared.replay_derivative(plan["run_id"], runtime_root=runtime)[
        "active"
    ] is True


def test_training_p2_dry_apply_and_replay_resolve_point_one_p1(
    tmp_path: Path,
) -> None:
    source = _write_wav(tmp_path / "source.wav", 1.0)
    p1_root = tmp_path / "p1"
    p2_root = tmp_path / "p2"
    p1_plan = training.dry_run(source, runtime_root=p1_root)
    training.apply_derivative(
        source, runtime_root=p1_root, approval_token=shared.APPLY_TOKEN
    )
    readiness = {
        method_id: (
            speech.readiness_matrix()[method_id]
            if method_id == "no_enhancement"
            else {
                "status": "blocked",
                "reason_code": "synthetic_test_blocked",
                "reason": "synthetic test does not execute this method",
            }
        )
        for method_id in speech.METHOD_IDS
    }
    plan = training.speech_dry_run(
        p1_plan["run_id"],
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        test_mode=True,
    )
    applied = training.apply_comparison(
        p1_plan["run_id"],
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        test_mode=True,
    )
    replay = training.replay_comparison(plan["run_id"], runtime_root=p2_root)
    assert applied["comparison"]["denominators"] == {
        "methods": 5,
        "attempted": 1,
        "success": 1,
        "failure": 0,
        "blocked": 4,
    }
    assert replay["active"] is True
