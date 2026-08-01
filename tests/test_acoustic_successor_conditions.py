from __future__ import annotations

import hashlib
import json
import math
import stat
import struct
import wave
from pathlib import Path

import pytest

import acoustic_successor_conditions as conditions


def _private_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sandbox = next(
        parent for parent in path.parents if parent.name.startswith("test_")
    )
    current = path.parent
    while True:
        current.chmod(0o700)
        if current == sandbox:
            break
        current = current.parent
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    return path


def _wav(path: Path, *, noisy: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    rate = 16_000
    frequency = 200 + int(path.stem.split("-")[-1])
    samples = []
    for index in range(rate * 10):
        second = index / rate
        speech = 1.0 <= second < 7.0
        amplitude = 9_000 if speech else (6_000 if noisy else 500)
        samples.append(round(amplitude * math.sin(2 * math.pi * frequency * second)))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    path.chmod(0o600)
    return path


def _corpus(tmp_path: Path) -> tuple[Path, list[Path]]:
    sources = []
    recordings = []
    splits = ["development"] * 3 + ["calibration"] * 2 + ["evaluation"] * 2
    for index, split in enumerate(splits, 1):
        source = _wav(tmp_path / "sources" / f"source-{index}.wav", noisy=index % 2 == 0)
        sources.append(source)
        recordings.append(
            {
                "recording_id": f"recording-{index}",
                "conversation_id": f"conversation-{index}",
                "split": split,
                "source_blob": {
                    "blob_id": f"blob-{index}",
                    "stored_path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "bytes": source.stat().st_size,
                    "mode": 0o600,
                },
                "operator_gold": {
                    "gold_id": f"gold-{index}",
                    "speaker_truth": [],
                    "same_person_label_groups": [],
                },
            }
        )
    core = {
        "schema_version": conditions.CORPUS_SCHEMA,
        "recordings": recordings,
        "denominators": {
            "recordings": 7,
            "split_recordings": {
                "development": 3,
                "calibration": 2,
                "evaluation": 2,
            },
        },
        "prediction_visibility": "excluded",
        "promotion_eligible": False,
    }
    digest = conditions._canonical_hash(core)
    manifest = {
        **core,
        "corpus_id": f"acoustic-corpus-{digest[:24]}",
        "content_sha256": digest,
        "runtime_readback_at_freeze": {},
        "frozen_at": "2026-08-01T00:00:00Z",
    }
    path = _private_json(tmp_path / "corpus" / "manifest.json", manifest)
    return path, sources


def _authorities(monkeypatch: pytest.MonkeyPatch) -> dict:
    authority = {
        "commit": "a" * 40,
        "clean": True,
        "module_sha256": hashlib.sha256(
            Path(conditions.__file__).read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(conditions, "_repository_authority", lambda: dict(authority))
    readiness = {
        method: {"status": "success", "asset_sha256": "b" * 64}
        for method in conditions.METHOD_IDS
    }
    monkeypatch.setattr(
        conditions.speech_preparation,
        "readiness_matrix",
        lambda: json.loads(json.dumps(readiness)),
    )
    return authority


def _mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def p1_dry(source_path: Path, **kwargs) -> dict:
        index = int(source_path.stem.split("-")[-1])
        return {"run_id": f"audio-run-{index:024d}"}

    def p1_apply(source_path: Path, **kwargs) -> dict:
        index = int(source_path.stem.split("-")[-1])
        run_id = f"audio-run-{index:024d}"
        root = Path(kwargs["runtime_root"])
        manifest_path = root / "runs" / run_id / "manifest.json"
        artifact_path = root / "artifacts" / f"derived-{index}.wav"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.parent.chmod(0o700)
        artifact_path.write_bytes(source_path.read_bytes())
        artifact_path.chmod(0o600)
        sample_rate = 16_000 if index % 2 == 0 else 48_000
        channels = 2 if index == 7 else 1
        manifest = {
            "run_id": run_id,
            "artifact_path": str(artifact_path),
            "source": {
                "probe": {
                    "codec_name": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "channel_layout": "stereo" if channels == 2 else "mono",
                    "format_name": "wav",
                }
            },
            "derived_audio": {"output_duration_seconds": 10.0},
        }
        _private_json(manifest_path, manifest)
        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }

    def p1_replay(run_id: str, **kwargs) -> dict:
        path = Path(kwargs["runtime_root"]) / "runs" / run_id / "replay.json"
        _private_json(path, {"run_id": run_id, "active": True})
        return {"replay_receipt_path": str(path)}

    def p2_dry(p1_run_id: str, **kwargs) -> dict:
        index = int(p1_run_id.split("-")[-1])
        return {"run_id": f"speech-prep-{index:024d}"}

    def p2_apply(p1_run_id: str, **kwargs) -> dict:
        index = int(p1_run_id.split("-")[-1])
        run_id = f"speech-prep-{index:024d}"
        root = Path(kwargs["runtime_root"])
        source = Path(kwargs["p1_runtime_root"]) / "runs" / p1_run_id / "manifest.json"
        p1_manifest = json.loads(source.read_text())
        output_path = p1_manifest["artifact_path"]
        methods = []
        for method in conditions.METHOD_IDS:
            methods.append(
                {
                    "method_id": method,
                    "status": "success",
                    "output_path": output_path,
                    "output_sha256": hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
                    "speech_regions": (
                        [{"start_seconds": 1.0, "end_seconds": 7.0}]
                        if method == "silero_vad"
                        else []
                    ),
                }
            )
        comparison = {"run_id": run_id, "method_results": methods}
        path = root / "runs" / run_id / "comparison.json"
        _private_json(path, comparison)
        return {"comparison": comparison, "comparison_path": str(path)}

    def p2_replay(run_id: str, **kwargs) -> dict:
        path = Path(kwargs["runtime_root"]) / "runs" / run_id / "replay.json"
        _private_json(path, {"run_id": run_id, "active": True})
        return {"replay_path": str(path)}

    monkeypatch.setattr(conditions.audio_derivatives, "dry_run", p1_dry)
    monkeypatch.setattr(conditions.audio_derivatives, "apply_derivative", p1_apply)
    monkeypatch.setattr(conditions.audio_derivatives, "replay_derivative", p1_replay)
    monkeypatch.setattr(conditions.speech_preparation, "dry_run", p2_dry)
    monkeypatch.setattr(conditions.speech_preparation, "apply_comparison", p2_apply)
    monkeypatch.setattr(conditions.speech_preparation, "replay_comparison", p2_replay)


def test_preview_is_deterministic_exact_and_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, _sources = _corpus(tmp_path)
    _authorities(monkeypatch)

    first = conditions.preview_condition_campaign(corpus_path)
    second = conditions.preview_condition_campaign(corpus_path)

    assert first == second
    assert first["denominators"] == {
        "recordings": 7,
        "methods_per_recording": 5,
        "method_attempts": 35,
    }
    assert first["will_process_audio"] is False
    assert first["will_run_models"] is False
    assert first["condition_policy"]["encoding_profile_is_not_device_evidence"] is True


def test_preview_recounts_recording_splits_instead_of_trusting_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, _sources = _corpus(tmp_path)
    _authorities(monkeypatch)
    manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    for recording in manifest["recordings"]:
        recording["split"] = "development"
    core = conditions._corpus_core(manifest)
    digest = conditions._canonical_hash(core)
    manifest["content_sha256"] = digest
    manifest["corpus_id"] = f"acoustic-corpus-{digest[:24]}"
    _private_json(corpus_path, manifest)

    with pytest.raises(conditions.SuccessorConditionError, match="corpus authority"):
        conditions.preview_condition_campaign(corpus_path)


def test_apply_replay_is_private_idempotent_and_blocks_missing_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, _sources = _corpus(tmp_path)
    _authorities(monkeypatch)
    _mock_pipeline(monkeypatch)
    preview = conditions.preview_condition_campaign(corpus_path)
    runtime = tmp_path / "runtime"

    with pytest.raises(conditions.SuccessorConditionError, match="preview hash"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256="f" * 64,
            runtime_root=runtime,
        )
    assert not runtime.exists()

    receipt = conditions.apply_condition_campaign(
        corpus_path,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
    )
    replay = conditions.replay_condition_campaign(
        Path(receipt["manifest_path"]), corpus_manifest_path=corpus_path
    )
    repeated = conditions.apply_condition_campaign(
        corpus_path,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
    )

    assert receipt["denominators"]["p1_successes"] == 7
    assert receipt["denominators"]["p2_method_successes"] == 35
    assert receipt["condition_coverage"]["fields"]["channel"]["status"] == "pass"
    assert receipt["condition_coverage"]["fields"]["device"] == {
        "observed_values": [],
        "observed_value_count": 0,
        "missing_recordings": 7,
        "status": "blocked",
    }
    assert receipt["condition_coverage"]["terminal_selection_eligible"] is False
    assert replay["full_body_match"] is True
    assert repeated["idempotent"] is True
    for path in Path(runtime).rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_source_and_lineage_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, sources = _corpus(tmp_path)
    _authorities(monkeypatch)
    _mock_pipeline(monkeypatch)
    preview = conditions.preview_condition_campaign(corpus_path)
    original = sources[0].read_bytes()
    sources[0].write_bytes(original + b"drift")
    with pytest.raises(conditions.SuccessorConditionError, match="source binding drifted"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "runtime",
        )
    sources[0].write_bytes(original)

    receipt = conditions.apply_condition_campaign(
        corpus_path,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "runtime",
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    comparison = Path(manifest["units"][0]["p2_comparison_path"])
    comparison.write_bytes(comparison.read_bytes() + b" ")
    with pytest.raises(conditions.SuccessorConditionError, match="lineage artifact drifted"):
        conditions.replay_condition_campaign(
            Path(receipt["manifest_path"]), corpus_manifest_path=corpus_path
        )


def test_execution_failure_is_private_and_prevents_unreviewed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, _sources = _corpus(tmp_path)
    _authorities(monkeypatch)
    _mock_pipeline(monkeypatch)
    preview = conditions.preview_condition_campaign(corpus_path)

    def fail_comparison(*args, **kwargs):
        raise RuntimeError("bounded fixture failure")

    monkeypatch.setattr(
        conditions.speech_preparation, "apply_comparison", fail_comparison
    )
    runtime = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="bounded fixture failure"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=runtime,
        )
    failure_path = (
        runtime / "runs" / preview["plan_id"] / "failure-receipt.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["retry_requires_new_review"] is True
    assert failure["did_run_biometrics"] is False
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o600
    with pytest.raises(conditions.SuccessorConditionError, match="prior condition"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=runtime,
        )


def test_receipt_write_failure_is_finalized_and_blocks_partial_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, _sources = _corpus(tmp_path)
    _authorities(monkeypatch)
    _mock_pipeline(monkeypatch)
    preview = conditions.preview_condition_campaign(corpus_path)
    runtime = tmp_path / "runtime"
    original_write = conditions.write_immutable_private_json

    def fail_receipt(path: Path, payload: dict, **kwargs) -> None:
        if Path(path).name == "apply-receipt.json":
            raise OSError("injected receipt finalization failure")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(conditions, "write_immutable_private_json", fail_receipt)
    with pytest.raises(OSError, match="receipt finalization"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=runtime,
        )

    run = runtime / "runs" / preview["plan_id"]
    failure = json.loads((run / "failure-receipt.json").read_text())
    assert failure["execution_phase"] == "receipt_write"
    assert failure["completed_recordings"] == 7
    assert (run / "condition-manifest.json").exists()
    assert not (run / "apply-receipt.json").exists()
    with pytest.raises(conditions.SuccessorConditionError, match="prior condition"):
        conditions.apply_condition_campaign(
            corpus_path,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=runtime,
        )
