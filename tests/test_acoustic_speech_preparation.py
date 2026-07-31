from __future__ import annotations

import json
import math
import stat
import struct
import wave
from pathlib import Path

import pytest

from acoustic_audio_derivatives import (
    APPLY_TOKEN as P1_APPLY_TOKEN,
    apply_derivative,
    dry_run as p1_dry_run,
    resolve_active_derivative,
)
from acoustic_speech_preparation import (
    APPLY_TOKEN,
    METHOD_IDS,
    ROLLBACK_TOKEN,
    FakePreparationAdapter,
    SpeechPreparationError,
    apply_comparison,
    dry_run,
    readiness_matrix,
    replay_comparison,
    rollback_comparison,
)


def write_wav(path: Path, *, duration_seconds: float = 1.25) -> Path:
    sample_rate = 16_000
    frames = bytearray()
    for index in range(int(duration_seconds * sample_rate)):
        value = int(
            0.2 * 32_767 * math.sin(2 * math.pi * 440 * index / sample_rate)
        )
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(bytes(frames))
    return path


def make_p1(tmp_path: Path) -> tuple[Path, Path, str]:
    source = write_wav(tmp_path / "source.wav")
    p1_root = tmp_path / "p1"
    plan = p1_dry_run(source, runtime_root=p1_root)
    apply_derivative(
        source,
        runtime_root=p1_root,
        approval_token=P1_APPLY_TOKEN,
    )
    return source, p1_root, plan["run_id"]


def synthetic_readiness() -> dict[str, dict[str, object]]:
    matrix = readiness_matrix()
    matrix["silero_vad"] = {
        "status": "success",
        "reason_code": None,
        "code_revision": "fake-silero-v1",
        "asset_sha256": "1" * 64,
        "acquisition_manifest_sha256": "2" * 64,
        "authorization": "synthetic_test_only",
        "reason": None,
    }
    return matrix


def method_readiness(method_id: str) -> dict[str, dict[str, object]]:
    matrix = readiness_matrix()
    matrix[method_id] = {
        "status": "success",
        "reason_code": None,
        "code_revision": f"fake-{method_id}-v1",
        "asset_sha256": "1" * 64,
        "acquisition_manifest_sha256": "2" * 64,
        "authorization": "synthetic_test_only",
        "reason": None,
    }
    return matrix


def fake_vad_result(*, invalid_regions: bool = False) -> dict[str, object]:
    return {
        "status": "success",
        "reason_code": None,
        "output_artifact_id": None,
        "output_sha256": None,
        "output_path": None,
        "timestamp_map": None,
        "speech_regions": [
            {
                "start_seconds": 0.8 if invalid_regions else 0.1,
                "end_seconds": 1.0 if invalid_regions else 0.4,
                "probability": 0.9,
            },
            {
                "start_seconds": 0.3 if invalid_regions else 0.6,
                "end_seconds": 0.7 if invalid_regions else 1.0,
                "probability": 0.8,
            },
        ],
        "overlap_regions": [],
        "speaker_change_regions": [],
        "quality_delta": None,
        "model_revisions": {"silero_vad": "fake-silero-v1"},
        "warnings": [],
        "abstention_reasons": [],
    }


def test_readiness_uses_normalized_status_and_explicit_reason_codes() -> None:
    matrix = readiness_matrix()

    assert tuple(matrix) == METHOD_IDS
    assert matrix["no_enhancement"]["status"] == "success"
    assert matrix["silero_vad"]["status"] == "blocked"
    assert matrix["silero_vad"]["reason_code"] == "not_acquired"
    assert matrix["deepfilternet"]["status"] == "blocked"
    assert matrix["rnnoise"]["status"] == "blocked"
    assert matrix["pyannote_community_1"]["status"] == "blocked"
    assert matrix["pyannote_community_1"]["reason_code"] == "human_gate"


def test_no_enhancement_lifecycle_is_private_replayable_and_non_destructive(
    tmp_path: Path,
) -> None:
    source, p1_root, p1_run_id = make_p1(tmp_path)
    source_before = source.read_bytes()
    source_mode = stat.S_IMODE(source.stat().st_mode)
    p2_root = tmp_path / "p2"
    plan = dry_run(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
    )
    assert plan["status"] == "success"
    assert plan["will_process_audio"] is False
    assert plan["will_read_calibration_or_evaluation"] is False

    with pytest.raises(SpeechPreparationError, match="requires token"):
        apply_comparison(
            p1_run_id,
            approval_token="",
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
        )
    applied = apply_comparison(
        p1_run_id,
        approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
    )
    comparison = applied["comparison"]
    assert applied["lifecycle_state"] == "applied"
    results = {result["method_id"]: result for result in comparison["method_results"]}
    p1_source = resolve_active_derivative(p1_run_id, runtime_root=p1_root)
    assert results["no_enhancement"]["status"] == "success"
    assert results["no_enhancement"]["output_sha256"] == p1_source["artifact_sha256"]
    assert results["no_enhancement"]["timestamp_map"] == p1_source["derived_audio"]["timestamp_map"]
    assert all(results[name]["status"] == "blocked" for name in METHOD_IDS[1:])
    assert comparison["denominators"] == {
        "methods": 5,
        "attempted": 1,
        "success": 1,
        "failure": 0,
        "blocked": 4,
    }
    assert comparison["eligible_for_identity"] is False
    assert comparison["status"] == "blocked"
    assert comparison["reason_code"] == "required_real_comparisons_not_run"
    assert source.read_bytes() == source_before
    assert stat.S_IMODE(source.stat().st_mode) == source_mode

    repeated = apply_comparison(
        p1_run_id,
        approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
    )
    assert repeated["idempotent_replay"] is True
    assert repeated["lifecycle_state"] == "applied"
    assert repeated["comparison"] == comparison
    active = replay_comparison(plan["run_id"], runtime_root=p2_root)
    assert active["status"] == "success"
    assert active["lifecycle_state"] == "verified_active"
    assert active["active"] is True

    with pytest.raises(SpeechPreparationError, match="requires token"):
        rollback_comparison(
            plan["run_id"], approval_token="", runtime_root=p2_root
        )
    rollback = rollback_comparison(
        plan["run_id"],
        approval_token=f"{ROLLBACK_TOKEN}:{plan['run_id']}",
        runtime_root=p2_root,
    )
    assert rollback["status"] == "success"
    assert rollback["lifecycle_state"] == "rolled_back"
    assert rollback["eligible_for_use"] is False
    assert (
        rollback_comparison(
            plan["run_id"],
            approval_token=f"{ROLLBACK_TOKEN}:{plan['run_id']}",
            runtime_root=p2_root,
        )
        == rollback
    )
    inactive = replay_comparison(plan["run_id"], runtime_root=p2_root)
    assert inactive["active"] is False
    assert inactive["lifecycle_state"] == "verified_rolled_back"
    with pytest.raises(SpeechPreparationError, match="cannot be reactivated"):
        apply_comparison(
            p1_run_id,
            approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
        )
    assert resolve_active_derivative(p1_run_id, runtime_root=p1_root)

    for path in p2_root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_fake_adapter_is_deterministic_and_segment_bounded(tmp_path: Path) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2"
    readiness = synthetic_readiness()
    adapter = FakePreparationAdapter("silero_vad", fake_vad_result())
    plan = dry_run(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        adapters={"silero_vad": adapter},
        test_mode=True,
    )
    applied = apply_comparison(
        p1_run_id,
        approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        adapters={"silero_vad": adapter},
        test_mode=True,
    )
    results = {
        result["method_id"]: result
        for result in applied["comparison"]["method_results"]
    }
    assert results["silero_vad"]["status"] == "success"
    assert results["silero_vad"]["speech_regions"][0]["start_seconds"] == 0.1
    assert replay_comparison(plan["run_id"], runtime_root=p2_root)["active"] is True


def test_invalid_or_private_fake_result_fails_before_comparison_write(
    tmp_path: Path,
) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2"
    readiness = synthetic_readiness()
    invalid_adapter = FakePreparationAdapter(
        "silero_vad", fake_vad_result(invalid_regions=True)
    )
    plan = dry_run(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        adapters={"silero_vad": invalid_adapter},
        test_mode=True,
    )
    with pytest.raises(SpeechPreparationError):
        apply_comparison(
            p1_run_id,
            approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            readiness=readiness,
            adapters={"silero_vad": invalid_adapter},
            test_mode=True,
        )
    assert not (p2_root / "runs" / plan["run_id"] / "comparison.json").exists()

    private_adapter = FakePreparationAdapter(
        "silero_vad", {**fake_vad_result(), "waveform": [0.1]}
    )
    with pytest.raises(SpeechPreparationError, match="forbidden"):
        dry_run(
            p1_run_id,
            p1_runtime_root=p1_root,
            runtime_root=tmp_path / "private-p2",
            readiness=readiness,
            adapters={"silero_vad": private_adapter},
            test_mode=True,
        )


def test_tamper_blocks_replay_and_rollback(tmp_path: Path) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2"
    plan = dry_run(
        p1_run_id, p1_runtime_root=p1_root, runtime_root=p2_root
    )
    applied = apply_comparison(
        p1_run_id,
        approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
    )
    comparison_path = Path(applied["comparison_path"])
    comparison_path.write_bytes(comparison_path.read_bytes() + b"tamper")

    with pytest.raises(SpeechPreparationError):
        replay_comparison(plan["run_id"], runtime_root=p2_root)
    with pytest.raises(SpeechPreparationError):
        rollback_comparison(
            plan["run_id"],
            approval_token=f"{ROLLBACK_TOKEN}:{plan['run_id']}",
            runtime_root=p2_root,
        )


def test_apply_receipt_contract_tamper_blocks_replay(tmp_path: Path) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2"
    plan = dry_run(
        p1_run_id, p1_runtime_root=p1_root, runtime_root=p2_root
    )
    applied = apply_comparison(
        p1_run_id,
        approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
    )
    apply_path = Path(applied["comparison_path"]).with_name("apply.json")
    receipt = json.loads(apply_path.read_text(encoding="utf-8"))
    receipt["lifecycle_state"] = "forged"
    apply_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SpeechPreparationError, match="apply receipt binding"):
        replay_comparison(plan["run_id"], runtime_root=p2_root)
    with pytest.raises(SpeechPreparationError, match="apply receipt binding"):
        apply_comparison(
            p1_run_id,
            approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
        )


@pytest.mark.parametrize(
    ("method_id", "result"),
    [
        (
            "silero_vad",
            {
                **fake_vad_result(),
                "speech_regions": [
                    {"start_seconds": float("nan"), "end_seconds": 0.5}
                ],
            },
        ),
        (
            "deepfilternet",
            {
                **fake_vad_result(),
                "speech_regions": None,
                "output_path": None,
                "output_sha256": None,
                "timestamp_map": None,
            },
        ),
        (
            "pyannote_community_1",
            {
                **fake_vad_result(),
                "speech_regions": None,
                "overlap_regions": None,
                "speaker_change_regions": None,
            },
        ),
    ],
)
def test_method_specific_success_evidence_fails_closed(
    tmp_path: Path, method_id: str, result: dict[str, object]
) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2"
    readiness = method_readiness(method_id)
    adapter = FakePreparationAdapter(method_id, result)
    plan = dry_run(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        readiness=readiness,
        adapters={method_id: adapter},
        test_mode=True,
    )
    with pytest.raises(SpeechPreparationError):
        apply_comparison(
            p1_run_id,
            approval_token=f"{APPLY_TOKEN}:{plan['run_id']}",
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            readiness=readiness,
            adapters={method_id: adapter},
            test_mode=True,
        )


def test_forged_or_private_readiness_is_rejected_before_dry_run(
    tmp_path: Path,
) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    forged = readiness_matrix()
    forged["silero_vad"] = {
        "status": "success",
        "reason_code": None,
        "code_revision": "forged",
        "asset_sha256": None,
        "acquisition_manifest_sha256": None,
        "authorization": "synthetic_test_only",
    }
    with pytest.raises(SpeechPreparationError, match="asset_sha256"):
        dry_run(
            p1_run_id,
            p1_runtime_root=p1_root,
            runtime_root=tmp_path / "forged",
            readiness=forged,
            test_mode=True,
        )

    private = method_readiness("silero_vad")
    private["silero_vad"]["access_token"] = "secret"
    with pytest.raises(SpeechPreparationError, match="forbidden"):
        dry_run(
            p1_run_id,
            p1_runtime_root=p1_root,
            runtime_root=tmp_path / "private",
            readiness=private,
            test_mode=True,
        )
