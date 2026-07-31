from __future__ import annotations

import math
import os
import stat
import struct
import wave
from pathlib import Path

import pytest

from acoustic_audio_derivatives import (
    APPLY_TOKEN,
    ROLLBACK_TOKEN,
    AudioDerivativeError,
    apply_derivative,
    dry_run,
    replay_derivative,
    rollback_derivative,
)


def write_wav(
    path: Path,
    *,
    duration_seconds: float = 1.25,
    sample_rate: int = 16_000,
    channels: int = 1,
    amplitude: float = 0.25,
    silence: bool = False,
    clipped: bool = False,
) -> Path:
    frame_count = max(1, int(duration_seconds * sample_rate))
    frames = bytearray()
    for index in range(frame_count):
        if silence:
            value = 0
        elif clipped:
            value = 32_767 if index % 2 == 0 else -32_768
        else:
            value = int(
                amplitude
                * 32_767
                * math.sin(2 * math.pi * 440 * index / sample_rate)
            )
        for channel in range(channels):
            selected = value if channel == 0 else int(value * 0.5)
            frames.extend(struct.pack("<h", selected))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(bytes(frames))
    return path


def source_state(path: Path) -> tuple[bytes, int, int]:
    metadata = path.stat()
    return path.read_bytes(), stat.S_IMODE(metadata.st_mode), metadata.st_mtime_ns


def test_dry_run_is_private_and_does_not_modify_source(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    source.chmod(0o744)
    before = source_state(source)

    result = dry_run(source, runtime_root=tmp_path / "runtime")

    assert result["will_execute_decoder"] is False
    assert result["will_modify_source"] is False
    assert result["recipe"]["operation"] == "baseline_decode"
    assert result["recipe"]["parameters"]["target_sample_rate"] == 16_000
    assert result["recipe"]["model_revisions"] == {}
    assert source_state(source) == before
    receipt = Path(result["dry_run_path"])
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert stat.S_IMODE(receipt.parent.stat().st_mode) == 0o700


def test_apply_requires_matching_dry_run_and_explicit_token(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"

    with pytest.raises(AudioDerivativeError, match="approval token"):
        apply_derivative(source, runtime_root=runtime, approval_token="")
    with pytest.raises(AudioDerivativeError, match="persisted dry run"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)


def test_apply_rejects_broadened_dry_run_receipt(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    Path(plan["dry_run_path"]).chmod(0o644)

    with pytest.raises(AudioDerivativeError, match="0600 regular file"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)


def test_apply_replay_and_rollback_are_content_addressed_and_non_destructive(
    tmp_path: Path,
) -> None:
    source = write_wav(tmp_path / "source.wav")
    source.chmod(0o754)
    before = source_state(source)
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)

    applied = apply_derivative(
        source,
        runtime_root=runtime,
        approval_token=APPLY_TOKEN,
    )
    manifest = applied["manifest"]
    artifact = Path(manifest["artifact_path"])
    derived = manifest["derived_audio"]
    quality = manifest["audio_quality"]

    assert applied["status"] == "applied"
    assert applied["source_unchanged"] is True
    assert artifact.name == f"{derived['output_sha256']}.wav"
    assert artifact.parent.name == derived["output_sha256"][:2]
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert stat.S_IMODE(artifact.parent.stat().st_mode) == 0o700
    assert derived["source_sha256"] == plan["source"]["sha256"]
    timestamp_map = derived["timestamp_map"]
    assert len(timestamp_map) == 1
    assert timestamp_map[0]["source_start_seconds"] == 0.0
    assert timestamp_map[0]["source_end_seconds"] == derived["source_duration_seconds"]
    assert timestamp_map[0]["output_start_seconds"] == 0.0
    assert timestamp_map[0]["output_end_seconds"] == derived["output_duration_seconds"]
    assert timestamp_map[0]["mapping"] == "affine_full_recording"
    assert timestamp_map[0]["output_frame_count"] == quality["metrics"]["frame_count"]
    assert derived["recipe"]["operation"] == "identity_copy"
    assert manifest["will_run_vad"] is False
    assert manifest["will_run_enhancement"] is False
    assert manifest["will_run_models"] is False
    assert manifest["will_perform_external_write"] is False
    assert source_state(source) == before

    repeated = apply_derivative(
        source,
        runtime_root=runtime,
        approval_token=APPLY_TOKEN,
    )
    assert repeated["idempotent_replay"] is True
    assert repeated["manifest"] == manifest

    replay = replay_derivative(plan["run_id"], runtime_root=runtime)
    assert replay["status"] == "verified_active"
    assert replay["active"] is True
    assert replay["artifact_sha256"] == derived["output_sha256"]

    with pytest.raises(AudioDerivativeError, match="approval token"):
        rollback_derivative(plan["run_id"], runtime_root=runtime, approval_token="")
    rollback = rollback_derivative(
        plan["run_id"],
        runtime_root=runtime,
        approval_token=ROLLBACK_TOKEN,
    )
    assert rollback["eligible_for_use"] is False
    assert rollback["artifact_retained"] is True
    assert artifact.is_file()
    assert source_state(source) == before
    repeated_rollback = rollback_derivative(
        plan["run_id"], runtime_root=runtime, approval_token=ROLLBACK_TOKEN
    )
    assert repeated_rollback == rollback

    rolled_back_replay = replay_derivative(plan["run_id"], runtime_root=runtime)
    assert rolled_back_replay["status"] == "verified_rolled_back"
    assert rolled_back_replay["active"] is False
    with pytest.raises(AudioDerivativeError, match="cannot be reactivated"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)


def test_multichannel_input_fails_closed_without_a_derivative(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "stereo.wav", channels=2)
    runtime = tmp_path / "runtime"

    with pytest.raises(AudioDerivativeError, match="mono source"):
        dry_run(source, runtime_root=runtime)
    assert not (runtime / "artifacts").exists()


@pytest.mark.parametrize(
    ("silence", "clipped", "warning", "abstention"),
    [
        (True, False, "predominantly_digital_silence", "no_usable_signal"),
        (False, True, "clipping_detected", "excessive_clipping"),
    ],
)
def test_quality_fail_closed_warnings(
    tmp_path: Path,
    silence: bool,
    clipped: bool,
    warning: str,
    abstention: str,
) -> None:
    source = write_wav(
        tmp_path / f"quality-{warning}.wav",
        silence=silence,
        clipped=clipped,
    )
    runtime = tmp_path / "runtime"
    dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source,
        runtime_root=runtime,
        approval_token=APPLY_TOKEN,
    )
    quality = applied["manifest"]["audio_quality"]

    assert warning in quality["warnings"]
    assert abstention in quality["abstention_reasons"]
    assert quality["usable_speech_seconds"] is None
    assert "usable_speech_not_assessed_until_p2" in quality["abstention_reasons"]
    if silence:
        assert quality["metrics"]["digital_silence_sample_count"] == quality["metrics"]["sample_count"]
        assert quality["metrics"]["digital_silence_fraction"] == 1.0
    if clipped:
        assert quality["metrics"]["clipped_sample_count"] == quality["metrics"]["sample_count"]
        assert quality["metrics"]["clipped_sample_fraction"] == 1.0


@pytest.mark.parametrize("content", [b"", b"not an audio file"])
def test_empty_and_corrupt_sources_fail_closed(
    tmp_path: Path,
    content: bytes,
) -> None:
    source = tmp_path / "invalid.bin"
    source.write_bytes(content)

    with pytest.raises(AudioDerivativeError):
        dry_run(source, runtime_root=tmp_path / "runtime")


def test_replay_detects_derived_artifact_tampering(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source,
        runtime_root=runtime,
        approval_token=APPLY_TOKEN,
    )
    artifact = Path(applied["manifest"]["artifact_path"])
    artifact.write_bytes(artifact.read_bytes() + b"tamper")

    with pytest.raises(AudioDerivativeError, match="hash mismatch"):
        replay_derivative(plan["run_id"], runtime_root=runtime)


def test_source_binding_and_source_or_runtime_symlinks_fail_closed(
    tmp_path: Path,
) -> None:
    source = write_wav(tmp_path / "source.wav")
    source_link = tmp_path / "source-link.wav"
    source_link.symlink_to(source)

    with pytest.raises(AudioDerivativeError, match="expected SHA-256"):
        dry_run(
            source,
            runtime_root=tmp_path / "runtime",
            expected_source_sha256="0" * 64,
        )
    with pytest.raises(AudioDerivativeError, match="must not be a symlink"):
        dry_run(source_link, runtime_root=tmp_path / "runtime")

    real_runtime = tmp_path / "real-runtime"
    real_runtime.mkdir()
    linked_runtime = tmp_path / "linked-runtime"
    linked_runtime.symlink_to(real_runtime, target_is_directory=True)
    with pytest.raises(AudioDerivativeError, match="symlink"):
        dry_run(source, runtime_root=linked_runtime)


def test_source_change_after_dry_run_is_rejected_without_derivative(
    tmp_path: Path,
) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    dry_run(source, runtime_root=runtime)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(AudioDerivativeError, match="matching persisted dry run"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)
    assert not (runtime / "artifacts").exists()


def test_content_address_conflict_is_not_overwritten(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    before = source_state(source)
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    output_hash = plan["source"]["sha256"]
    artifact_dir = runtime / "artifacts" / output_hash[:2]
    artifact_dir.mkdir(parents=True, mode=0o700)
    artifact = artifact_dir / f"{output_hash}.wav"
    artifact.write_bytes(b"conflict")
    artifact.chmod(0o600)

    with pytest.raises(AudioDerivativeError, match="Content-addressed artifact conflict"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)
    assert artifact.read_bytes() == b"conflict"
    assert source_state(source) == before


def test_equal_hash_preexisting_artifact_with_broad_mode_fails_closed(
    tmp_path: Path,
) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    output_hash = plan["source"]["sha256"]
    artifact_dir = runtime / "artifacts" / output_hash[:2]
    artifact_dir.mkdir(parents=True, mode=0o700)
    artifact = artifact_dir / f"{output_hash}.wav"
    artifact.write_bytes(source.read_bytes())
    artifact.chmod(0o644)

    with pytest.raises(AudioDerivativeError, match="0600 regular file"):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o644


def test_private_modes_survive_permissive_umask(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    previous_umask = os.umask(0)
    try:
        dry_run(source, runtime_root=runtime)
        applied = apply_derivative(
            source, runtime_root=runtime, approval_token=APPLY_TOKEN
        )
    finally:
        os.umask(previous_umask)

    for path in runtime.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected
    assert applied["manifest"]["eligible_for_identity"] is False


@pytest.mark.parametrize(
    "manifest_key",
    [
        "dry_run_path",
        "derived_audio_path",
        "audio_quality_path",
        "recipe_path",
    ],
)
def test_replay_rejects_individual_evidence_tamper(
    tmp_path: Path, manifest_key: str
) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source, runtime_root=runtime, approval_token=APPLY_TOKEN
    )
    evidence = Path(applied["manifest"][manifest_key])
    evidence.write_bytes(evidence.read_bytes() + b"tamper")

    with pytest.raises(AudioDerivativeError):
        replay_derivative(plan["run_id"], runtime_root=runtime)


def test_replay_rejects_manifest_and_apply_receipt_tamper(tmp_path: Path) -> None:
    for target_name in ("manifest.json", "apply-receipt.json"):
        case = tmp_path / target_name.removesuffix(".json")
        case.mkdir()
        source = write_wav(case / "source.wav")
        runtime = case / "runtime"
        plan = dry_run(source, runtime_root=runtime)
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)
        target = runtime / "runs" / plan["run_id"] / target_name
        target.write_bytes(target.read_bytes() + b"tamper")

        with pytest.raises(AudioDerivativeError):
            replay_derivative(plan["run_id"], runtime_root=runtime)


def test_repeated_apply_replays_evidence_before_idempotent_return(
    tmp_path: Path,
) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source, runtime_root=runtime, approval_token=APPLY_TOKEN
    )
    quality_path = Path(applied["manifest"]["audio_quality_path"])
    quality_path.write_bytes(quality_path.read_bytes() + b"tamper")

    with pytest.raises(AudioDerivativeError):
        apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)


def test_replay_rejects_broadened_artifact_permissions(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source, runtime_root=runtime, approval_token=APPLY_TOKEN
    )
    artifact = Path(applied["manifest"]["artifact_path"])
    artifact.chmod(0o644)

    with pytest.raises(AudioDerivativeError, match="0600 regular file"):
        replay_derivative(plan["run_id"], runtime_root=runtime)


def test_rollback_rejects_corrupted_active_evidence(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav")
    runtime = tmp_path / "runtime"
    plan = dry_run(source, runtime_root=runtime)
    applied = apply_derivative(
        source, runtime_root=runtime, approval_token=APPLY_TOKEN
    )
    quality_path = Path(applied["manifest"]["audio_quality_path"])
    quality_path.write_bytes(quality_path.read_bytes() + b"tamper")

    with pytest.raises(AudioDerivativeError):
        rollback_derivative(
            plan["run_id"], runtime_root=runtime, approval_token=ROLLBACK_TOKEN
        )
    assert not (
        runtime / "runs" / plan["run_id"] / "rollback-receipt.json"
    ).exists()
