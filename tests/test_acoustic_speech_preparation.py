from __future__ import annotations

import json
import math
import stat
import struct
import wave
from pathlib import Path

import pytest
import acoustic_speech_preparation as speech_preparation

from acoustic_audio_derivatives import (
    APPLY_TOKEN as P1_APPLY_TOKEN,
    apply_derivative,
    dry_run as p1_dry_run,
    resolve_active_derivative,
    sha256_file,
)


@pytest.fixture(autouse=True)
def isolate_live_acquisition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unit tests must not depend on or execute workstation model assets."""
    root = tmp_path / "unacquired-model-runtime"
    monkeypatch.setattr(speech_preparation, "DEFAULT_OPEN_ACQUISITION_ROOT", root)
    monkeypatch.setattr(
        speech_preparation,
        "DEFAULT_OPEN_ACQUISITION_MANIFEST",
        root / "acquisition-manifest.json",
    )
    pyannote_root = tmp_path / "unacquired-pyannote-runtime"
    monkeypatch.setattr(
        speech_preparation, "DEFAULT_PYANNOTE_ACQUISITION_ROOT", pyannote_root
    )
    monkeypatch.setattr(
        speech_preparation,
        "DEFAULT_PYANNOTE_ACQUISITION_MANIFEST",
        pyannote_root / "acquisition-manifest.json",
    )
from acoustic_speech_preparation import (
    APPLY_TOKEN,
    METHOD_IDS,
    ROLLBACK_TOKEN,
    FakePreparationAdapter,
    SpeechPreparationError,
    apply_comparison,
    dry_run_open_candidate_acquisition,
    dry_run,
    readiness_matrix,
    replay_comparison,
    replay_open_candidate_acquisition,
    resolve_comparison_lineage_receipt,
    rollback_comparison,
    _activity_regions,
    _bounded_turns,
    _speaker_change_regions,
    _pending_downstream_measurements,
    _validate_method_result,
    _verified_pyannote_acquisition,
    _write_pcm16_mono,
)


def test_activity_regions_merge_union_and_measure_overlap() -> None:
    turns = [
        (0.0, 1.0, "speaker-a"),
        (0.5, 1.5, "speaker-b"),
        (2.0, 2.5, "speaker-a"),
    ]

    assert _activity_regions(turns, minimum_count=1) == [
        {"start_seconds": 0.0, "end_seconds": 1.5},
        {"start_seconds": 2.0, "end_seconds": 2.5},
    ]
    assert _activity_regions(turns, minimum_count=2) == [
        {"start_seconds": 0.5, "end_seconds": 1.0},
    ]


def test_speaker_change_regions_are_bounded_and_non_overlapping() -> None:
    turns = [
        (0.0, 0.7, "speaker-a"),
        (0.7, 1.2, "speaker-b"),
        (0.7005, 1.0, "speaker-a"),
        (1.9995, 2.1, "speaker-b"),
        (2.0, 2.2, "speaker-a"),
    ]

    assert _speaker_change_regions(turns, 2.0) == [
        {"start_seconds": 0.7, "end_seconds": 0.701},
        {"start_seconds": 1.9995, "end_seconds": 2.0},
    ]


def test_provider_turns_are_clipped_to_authoritative_duration() -> None:
    assert _bounded_turns(
        [(-0.1, 0.2, "a"), (1.9, 2.1, "b"), (2.1, 2.2, "c")], 2.0
    ) == [(0.0, 0.2, "a"), (1.9, 2.0, "b")]


def test_downstream_reasons_distinguish_completed_preparation() -> None:
    pending = _pending_downstream_measurements(True)
    assert pending["transcription"]["reason_code"] == (
        "not_run_downstream_measurements"
    )
    assert pending["diarization"]["reason_code"] == (
        "not_run_downstream_measurements"
    )
    assert pending["verification"]["reason_code"] == "not_run_dependency_p3_p4"

    blocked = _pending_downstream_measurements(False)
    assert blocked["transcription"]["reason_code"] == (
        "not_run_dependency_real_methods"
    )


def test_pyannote_acquisition_requires_complete_private_hash_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "pyannote-acquisition"
    snapshot = root / "community-1"
    names = {
        ".gitattributes", "README.md", "config.yaml", "diarization.gif",
        "embedding/README.md", "embedding/pytorch_model.bin",
        "plda/README.md", "plda/plda.npz", "plda/xvec_transform.npz",
        "segmentation/pytorch_model.bin",
    }
    artifacts = {}
    for name in sorted(names):
        path = snapshot / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        path.chmod(0o600)
        artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    for directory in [root, snapshot, snapshot / "embedding", snapshot / "plda", snapshot / "segmentation"]:
        directory.chmod(0o700)
    manifest_path = root / "acquisition-manifest.json"
    manifest = {
        "schema_version": "transcribe-audio.pyannote-community-1-acquisition-manifest.v1",
        "repo_id": "pyannote/speaker-diarization-community-1",
        "revision_sha": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "package_distribution": "pyannote-audio",
        "package_version": "4.0.4",
        "authorization_basis": "operator_blanket_2026-07-31",
        "gated_access_verified": True,
        "contact_information_sharing_authorized": True,
        "snapshot_dir": str(snapshot),
        "artifacts": artifacts,
        "created_at": "2026-07-31T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    monkeypatch.setattr(speech_preparation, "DEFAULT_PYANNOTE_ACQUISITION_ROOT", root)
    monkeypatch.setattr(
        speech_preparation, "DEFAULT_PYANNOTE_ACQUISITION_MANIFEST", manifest_path
    )

    verified = _verified_pyannote_acquisition()
    assert verified is not None
    assert verified["revision_sha"] == manifest["revision_sha"]
    assert len(verified["asset_sha256"]) == 64
    (snapshot / "config.yaml").write_bytes(b"tampered")
    with pytest.raises(SpeechPreparationError, match="hash mismatch"):
        _verified_pyannote_acquisition()


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


def test_open_candidate_acquisition_dry_run_is_immutable_and_no_download(
    tmp_path: Path,
) -> None:
    root = tmp_path / "p2c"
    before = readiness_matrix()
    plan = dry_run_open_candidate_acquisition(runtime_root=root)
    assert plan["status"] == "success"
    assert plan["reason_code"] is None
    assert plan["authorization_basis"] == "operator_blanket_2026-07-31"
    assert "required_approval_token" not in plan
    assert plan["spec"]["authorization_scope"] == (
        "download_install_build_open_candidates_only"
    )
    assert [item["candidate_id"] for item in plan["spec"]["candidates"]] == [
        "silero_vad",
        "deepfilternet",
        "rnnoise",
    ]
    assert plan["host"]["deepfilterlib_cp312_wheel_available"] is False
    for field in (
        "will_download",
        "will_install",
        "will_build",
        "will_read_audio",
        "will_accept_terms",
        "will_share_contact_information",
        "will_perform_external_write",
    ):
        assert plan[field] is False
    assert readiness_matrix() == before
    replay = replay_open_candidate_acquisition(
        plan["run_id"],
        expected_dry_run_sha256=plan["dry_run_sha256"],
        runtime_root=root,
    )
    assert replay["dry_run_sha256"] == plan["dry_run_sha256"]
    assert replay["authorization_basis"] == plan["authorization_basis"]
    for path in root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_open_candidate_acquisition_replay_rejects_spec_drift(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p2/open-candidate-acquisition-plan.json"
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    root = tmp_path / "p2c"
    plan = dry_run_open_candidate_acquisition(
        runtime_root=root, spec_path=spec
    )
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["candidates"][0]["revision"] = "changed"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SpeechPreparationError, match="spec drifted"):
        replay_open_candidate_acquisition(
            plan["run_id"],
            expected_dry_run_sha256=plan["dry_run_sha256"],
            runtime_root=root,
        )


@pytest.mark.parametrize("tamper", ["created_at", "serialization"])
def test_open_candidate_acquisition_replay_requires_reviewed_plan_hash(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / "p2c"
    plan = dry_run_open_candidate_acquisition(runtime_root=root)
    path = Path(plan["dry_run_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "created_at":
        payload["created_at"] = "2099-01-01T00:00:00Z"
        replacement = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        replacement = json.dumps(payload, indent=4, sort_keys=False)
    path.write_text(replacement + "\n", encoding="utf-8")
    with pytest.raises(SpeechPreparationError, match="dry-run hash mismatch"):
        replay_open_candidate_acquisition(
            plan["run_id"],
            expected_dry_run_sha256=plan["dry_run_sha256"],
            runtime_root=root,
        )


def test_open_candidate_acquisition_rejects_gated_candidate_injection(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p2/open-candidate-acquisition-plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["candidates"].append(
        {**payload["candidates"][0], "candidate_id": "pyannote_community_1"}
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SpeechPreparationError, match="incomplete or unordered"):
        dry_run_open_candidate_acquisition(
            runtime_root=tmp_path / "p2c", spec_path=spec
        )


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
    assert matrix["silero_vad"]["reason_code"] == "asset_hash_unbound"
    assert matrix["deepfilternet"]["status"] == "blocked"
    assert matrix["rnnoise"]["status"] == "blocked"
    assert matrix["pyannote_community_1"]["status"] == "blocked"
    assert matrix["pyannote_community_1"]["reason_code"] == "provider_auth_required"


def test_enhanced_output_is_content_addressed_and_rehashed(tmp_path: Path) -> None:
    import torch

    _, p1_root, p1_run_id = make_p1(tmp_path)
    source = resolve_active_derivative(p1_run_id, runtime_root=p1_root)
    runtime_root = tmp_path / "private-p2"
    runtime_root.mkdir(mode=0o700)
    output_path, output_sha = _write_pcm16_mono(
        runtime_root / "outputs/deepfilternet-source.wav",
        torch.zeros((1, 20_000), dtype=torch.float32),
        16_000,
    )
    duration = float(source["derived_audio"]["output_duration_seconds"])
    result = {
        "method_id": "deepfilternet",
        "status": "success",
        "reason_code": None,
        "attempted": True,
        "denominator": 1,
        "readiness": {"authorization": "verified_acquisition"},
        "output_artifact_id": "p2-deepfilternet-" + output_sha[:24],
        "output_sha256": output_sha,
        "output_path": str(output_path),
        "timestamp_map": [{
            "source_start_seconds": 0.0,
            "source_end_seconds": duration,
            "output_start_seconds": 0.0,
            "output_end_seconds": duration,
        }],
        "speech_regions": None,
        "overlap_regions": [],
        "speaker_change_regions": [],
        "warnings": [],
        "abstention_reasons": [],
    }
    assert output_path.stem == output_sha
    assert _validate_method_result(
        result, source, runtime_root=runtime_root
    )["output_sha256"] == output_sha
    output_path.write_bytes(output_path.read_bytes() + b"tamper")
    with pytest.raises(SpeechPreparationError, match="SHA-256 binding"):
        _validate_method_result(result, source, runtime_root=runtime_root)


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

    applied = apply_comparison(
        p1_run_id,
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
    lineage = resolve_comparison_lineage_receipt(
        plan["run_id"],
        method_id="no_enhancement",
        replay_receipt_sha256=sha256_file(Path(active["replay_path"])),
        runtime_root=p2_root,
    )
    assert lineage["validation_status"] == "verified_active_metadata_receipt"
    assert lineage["will_read_audio"] is False
    with pytest.raises(SpeechPreparationError, match="not successful"):
        resolve_comparison_lineage_receipt(
            plan["run_id"],
            method_id="silero_vad",
            replay_receipt_sha256=lineage["replay_receipt_sha256"],
            runtime_root=p2_root,
        )

    rollback = rollback_comparison(
        plan["run_id"],
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
    with pytest.raises(SpeechPreparationError, match="not lineage eligible"):
        resolve_comparison_lineage_receipt(
            plan["run_id"],
            method_id="no_enhancement",
            replay_receipt_sha256=lineage["replay_receipt_sha256"],
            runtime_root=p2_root,
        )
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


def test_calibration_preparation_requires_and_replays_exact_split_authority(
    tmp_path: Path,
) -> None:
    _, p1_root, p1_run_id = make_p1(tmp_path)
    p2_root = tmp_path / "p2-calibration"
    authority_sha = "a" * 64
    with pytest.raises(SpeechPreparationError, match="exact split authority"):
        dry_run(
            p1_run_id,
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            intended_split="calibration",
        )
    with pytest.raises(SpeechPreparationError, match="cannot open"):
        dry_run(
            p1_run_id,
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            intended_split="evaluation",
            split_access_authority_sha256=authority_sha,
        )

    plan = dry_run(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        intended_split="calibration",
        split_access_authority_sha256=authority_sha,
    )
    assert plan["intended_split"] == "calibration"
    assert plan["split_access_authority_sha256"] == authority_sha
    assert plan["will_read_calibration_or_evaluation"] is True
    applied = apply_comparison(
        p1_run_id,
        p1_runtime_root=p1_root,
        runtime_root=p2_root,
        intended_split="calibration",
        split_access_authority_sha256=authority_sha,
    )
    assert applied["intended_split"] == "calibration"
    assert applied["split_access_authority_sha256"] == authority_sha
    assert applied["did_read_calibration_or_evaluation"] is True
    replayed = replay_comparison(plan["run_id"], runtime_root=p2_root)
    assert replayed["intended_split"] == "calibration"
    assert replayed["split_access_authority_sha256"] == authority_sha
    assert replayed["did_read_calibration_or_evaluation"] is True
    lineage = resolve_comparison_lineage_receipt(
        plan["run_id"],
        method_id="no_enhancement",
        replay_receipt_sha256=sha256_file(Path(replayed["replay_path"])),
        runtime_root=p2_root,
    )
    assert lineage["intended_split"] == "calibration"
    assert lineage["split_access_authority_sha256"] == authority_sha


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
