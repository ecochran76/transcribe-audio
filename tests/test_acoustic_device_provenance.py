from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

import acoustic_device_provenance as device
import acoustic_successor_conditions as conditions


def _private_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.parent
    while current != current.parent and current.name != "test_device_authority":
        current.chmod(0o700)
        current = current.parent
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "test_device_authority"
    root.mkdir(mode=0o700)
    splits = ["development"] * 3 + ["calibration"] * 2 + ["evaluation"] * 2
    recordings = []
    sources = []
    for position, split in enumerate(splits, 1):
        source = root / "sources" / f"source-{position}.m4a"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.parent.chmod(0o700)
        source.write_bytes(f"source-{position}".encode())
        source.chmod(0o600)
        sources.append(source)
        transcript = _private_json(
            root / "transcripts" / f"transcript-{position}.json",
            {
                "schema_version": "fixture.v1",
                "recording_id": f"recording-{position}",
                "conversation_id": f"conversation-{position}",
                "recording_start": f"2026-01-{position:02d}T12:00:00Z",
                "recording_end": f"2026-01-{position:02d}T12:30:00Z",
                "transcript_title": f"Private case {position}",
                "source_media_path": f"/missing/source-{position}.m4a",
                "transcript_text": "private fixture body",
                "utterances": [],
            },
        )
        recordings.append(
            {
                "recording_id": f"recording-{position}",
                "conversation_id": f"conversation-{position}",
                "document_id": f"document-{position}",
                "chronological_rank": position,
                "split": split,
                "source_blob": {
                    "blob_id": f"blob-{position}",
                    "stored_path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "bytes": source.stat().st_size,
                    "mode": 0o600,
                },
                "transcript_lineage": {
                    "current_artifact_path": str(transcript),
                    "current_artifact_sha256": hashlib.sha256(
                        transcript.read_bytes()
                    ).hexdigest(),
                    "reviewed_artifact_sha256": hashlib.sha256(
                        transcript.read_bytes()
                    ).hexdigest(),
                },
                "operator_gold": {},
            }
        )
    corpus_core = {
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
    corpus_digest = conditions._canonical_hash(corpus_core)
    corpus_path = _private_json(
        root / "corpus" / "manifest.json",
        {
            **corpus_core,
            "corpus_id": f"acoustic-corpus-{corpus_digest[:24]}",
            "content_sha256": corpus_digest,
            "runtime_readback_at_freeze": {},
            "frozen_at": "2026-08-01T00:00:00Z",
        },
    )
    passed_field = {
        "observed_values": ["a", "b"],
        "observed_value_count": 2,
        "missing_recordings": 0,
        "status": "pass",
    }
    coverage = {
        "fields": {
            "channel": dict(passed_field),
            "device": {
                "observed_values": [],
                "observed_value_count": 0,
                "missing_recordings": 7,
                "status": "blocked",
            },
            "noise": dict(passed_field),
            "telephone_bandwidth": dict(passed_field),
            "usable_duration_band": dict(passed_field),
        },
        "terminal_selection_eligible": False,
        "blockers": ["device_condition_coverage_below_policy"],
    }
    condition_core = {
        "schema_version": conditions.MANIFEST_SCHEMA,
        "status": "complete",
        "plan_id": "successor-conditions-fixture",
        "plan_content_sha256": "1" * 64,
        "corpus": {
            "corpus_id": f"acoustic-corpus-{corpus_digest[:24]}",
            "content_sha256": corpus_digest,
            "manifest_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "manifest_path": str(corpus_path),
        },
        "repository_authority": {
            "commit": "b" * 40,
            "clean": True,
            "module_sha256": hashlib.sha256(
                Path(conditions.__file__).read_bytes()
            ).hexdigest(),
        },
        "module_authority": {
            "condition_sha256": hashlib.sha256(
                Path(conditions.__file__).read_bytes()
            ).hexdigest(),
            "p1_sha256": hashlib.sha256(
                Path(device.audio_derivatives.__file__).read_bytes()
            ).hexdigest(),
            "p2_sha256": hashlib.sha256(
                Path(device.speech_preparation.__file__).read_bytes()
            ).hexdigest(),
        },
        "readiness_sha256": "2" * 64,
        "denominators": {
            "recordings": 7,
            "methods_per_recording": 5,
            "method_attempts": 35,
            "p1_successes": 7,
            "p2_method_successes": 35,
        },
        "units": [
            {
                "recording_id": record["recording_id"],
                "conversation_id": record["conversation_id"],
                "split": record["split"],
                "source_sha256": record["source_blob"]["sha256"],
            }
            for record in recordings
        ],
        "condition_coverage": coverage,
        "did_process_audio": True,
        "did_run_p1_p2": True,
        "did_run_biometrics": False,
    }
    condition_digest = conditions._canonical_hash(condition_core)
    condition_path = _private_json(
        root / "condition" / "runs" / "fixture" / "condition-manifest.json",
        {
            **condition_core,
            "content_sha256": condition_digest,
            "applied_at": "2026-08-01T00:00:00Z",
        },
    )
    _private_json(
        condition_path.parent / "apply-receipt.json",
        {
            "schema_version": conditions.RECEIPT_SCHEMA,
            "plan_id": condition_core["plan_id"],
            "manifest_path": str(condition_path),
            "manifest_sha256": hashlib.sha256(condition_path.read_bytes()).hexdigest(),
            "content_sha256": condition_digest,
            "denominators": condition_core["denominators"],
            "condition_coverage": coverage,
            "mode": "0600",
            "will_perform_external_write": False,
        },
    )
    return corpus_path, condition_path, sources


def _authorities(monkeypatch: pytest.MonkeyPatch) -> dict:
    authority = {
        "commit": "a" * 40,
        "clean": True,
        "module_sha256": hashlib.sha256(Path(device.__file__).read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(device, "_repository_authority", lambda: dict(authority))
    monkeypatch.setattr(device, "_validate_closed_commit", lambda commit: None)
    monkeypatch.setattr(
        device,
        "_module_sha256_at_commit",
        lambda commit: authority["module_sha256"],
    )
    return authority


def _freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, str, list[Path]]:
    corpus, condition, sources = _fixture(tmp_path)
    _authorities(monkeypatch)
    runtime = tmp_path / "runtime"
    preview = device.preview_device_campaign(corpus, condition)
    device.apply_device_campaign(
        corpus,
        condition,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
    )
    return corpus, condition, runtime, preview["campaign_id"], sources


def _record_all(
    campaign_id: str,
    corpus: Path,
    condition: Path,
    runtime: Path,
    labels: list[str | None],
) -> None:
    for label in labels:
        device.open_next_device_case(
            campaign_id,
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )
        device.record_device_provenance(
            campaign_id,
            physical_device_label=label or "",
            attested_by="fixture-operator",
            status="observed" if label else "unavailable",
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )


def test_preview_is_exact_deterministic_and_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, _sources = _fixture(tmp_path)
    _authorities(monkeypatch)

    first = device.preview_device_campaign(corpus, condition)
    second = device.preview_device_campaign(corpus, condition)

    assert first == second
    assert len(first["cases"]) == 7
    assert first["denominators"]["split_recordings"] == {
        "development": 3,
        "calibration": 2,
        "evaluation": 2,
    }
    assert first["will_assert_device_fact"] is False
    assert first["will_run_models"] is False
    assert not (tmp_path / "runtime").exists()


def test_frozen_campaign_replays_after_clean_descendant_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, sources = _fixture(tmp_path)
    authority = _authorities(monkeypatch)
    frozen_authority = dict(authority)
    runtime = tmp_path / "runtime"
    preview = device.preview_device_campaign(corpus, condition)
    device.apply_device_campaign(
        corpus,
        condition,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=runtime,
    )

    authority["commit"] = "c" * 40
    replay = device.replay_device_campaign(
        preview["campaign_id"],
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )

    assert replay["full_body_match"] is True
    manifest = json.loads(
        (runtime / "campaigns" / preview["campaign_id"] / "manifest.json").read_text()
    )
    assert manifest["repository_authority"] == frozen_authority
    assert all(source.is_file() for source in sources)


def test_apply_after_descendant_commit_reuses_frozen_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, _sources = _fixture(tmp_path)
    authority = _authorities(monkeypatch)
    runtime = tmp_path / "runtime"
    frozen_preview = device.preview_device_campaign(corpus, condition)
    device.apply_device_campaign(
        corpus,
        condition,
        expected_content_sha256=frozen_preview["content_sha256"],
        runtime_root=runtime,
    )

    authority["commit"] = "c" * 40
    descendant_preview = device.preview_device_campaign(corpus, condition)
    assert descendant_preview["campaign_id"] != frozen_preview["campaign_id"]
    replay = device.apply_device_campaign(
        corpus,
        condition,
        expected_content_sha256=descendant_preview["content_sha256"],
        runtime_root=runtime,
    )

    assert replay["campaign_id"] == frozen_preview["campaign_id"]
    assert replay["full_body_match"] is True
    assert [path.name for path in (runtime / "campaigns").iterdir()] == [
        frozen_preview["campaign_id"]
    ]


def test_apply_rejects_two_campaigns_for_same_predecessor_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, _sources = _fixture(tmp_path)
    authority = _authorities(monkeypatch)
    runtime = tmp_path / "runtime"
    first = device.preview_device_campaign(corpus, condition)
    device.apply_device_campaign(
        corpus,
        condition,
        expected_content_sha256=first["content_sha256"],
        runtime_root=runtime,
    )

    authority["commit"] = "c" * 40
    second = device.preview_device_campaign(corpus, condition)
    second_paths = device._paths(runtime, second["campaign_id"])
    device.ensure_private_tree(second_paths["root"], second_paths["campaign"])
    second_manifest = {
        **second,
        "schema_version": device.CAMPAIGN_SCHEMA,
        "status": "open",
        "applied_at": "2026-08-01T12:00:00Z",
    }
    device.write_immutable_private_json(second_paths["manifest"], second_manifest)
    device.write_immutable_private_json(
        second_paths["receipt"],
        {
            "schema_version": device.CAMPAIGN_RECEIPT_SCHEMA,
            "campaign_id": second["campaign_id"],
            "content_sha256": second["content_sha256"],
            "manifest_path": str(second_paths["manifest"]),
            "manifest_sha256": device.sha256_file(second_paths["manifest"]),
            "recordings": 7,
            "mode": "0600",
            "contains_private_operator_context": False,
            "contains_device_labels": False,
            "will_perform_external_write": False,
        },
    )

    with pytest.raises(device.DeviceProvenanceError, match="Multiple device campaigns"):
        device.apply_device_campaign(
            corpus,
            condition,
            expected_content_sha256=second["content_sha256"],
            runtime_root=runtime,
        )


@pytest.mark.parametrize("drift", ["dirty", "historical_module", "non_ancestor"])
def test_frozen_campaign_descendant_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    corpus, condition, runtime, campaign_id, _sources = _freeze(
        tmp_path, monkeypatch
    )
    current = {
        "commit": "c" * 40,
        "clean": drift != "dirty",
        "module_sha256": "d" * 64,
    }
    monkeypatch.setattr(device, "_repository_authority", lambda: dict(current))
    if drift == "historical_module":
        monkeypatch.setattr(device, "_module_sha256_at_commit", lambda commit: "0" * 64)
    elif drift == "non_ancestor":
        monkeypatch.setattr(
            device,
            "_validate_closed_commit",
            lambda commit: (_ for _ in ()).throw(
                device.DeviceProvenanceError("not an ancestor")
            ),
        )

    with pytest.raises(device.DeviceProvenanceError):
        device.replay_device_campaign(
            campaign_id,
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )


def test_freeze_cursor_attestation_and_replay_are_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, runtime, campaign_id, _sources = _freeze(
        tmp_path, monkeypatch
    )

    first = device.open_next_device_case(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    reopened = device.open_next_device_case(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert first["packet"]["position"] == 1
    assert reopened["idempotent_reopen"] is True
    observed = device.record_device_provenance(
        campaign_id,
        physical_device_label="Operator Phone A",
        attested_by="fixture-operator",
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert observed["status"] == "observed"
    assert observed["contains_device_label"] is False
    second = device.open_next_device_case(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert second["packet"]["position"] == 2
    device.record_device_provenance(
        campaign_id,
        attested_by="fixture-operator",
        status="unavailable",
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    replay = device.replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert replay["opened_cases"] == 2
    assert replay["initial_records"] == 2
    assert replay["full_body_match"] is True
    for path in runtime.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (
            0o700 if path.is_dir() else 0o600
        )


def test_composite_requires_seven_direct_and_two_distinct_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, runtime, campaign_id, _sources = _freeze(
        tmp_path, monkeypatch
    )
    _record_all(
        campaign_id,
        corpus,
        condition,
        runtime,
        ["Phone A", "Phone B", "Phone A", "Phone B", "Phone A", "Phone B", "Phone A"],
    )
    preview = device.preview_composite_condition_authority(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert preview["direct_observed_attestation_count"] == 7
    assert preview["condition_coverage"]["fields"]["device"][
        "observed_value_count"
    ] == 2
    assert preview["condition_coverage"]["terminal_selection_eligible"] is True
    receipt = device.apply_composite_condition_authority(
        campaign_id,
        expected_content_sha256=preview["content_sha256"],
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    replay = device.replay_composite_condition_authority(
        Path(receipt["manifest_path"]),
        campaign_id=campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert replay["full_body_match"] is True


def test_unknown_and_one_device_remain_blocked_then_correction_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, runtime, campaign_id, _sources = _freeze(
        tmp_path, monkeypatch
    )
    _record_all(
        campaign_id,
        corpus,
        condition,
        runtime,
        ["Phone A", "Phone A", "Phone A", "Phone A", "Phone A", "Phone A", None],
    )
    blocked = device.preview_composite_condition_authority(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert blocked["condition_coverage"]["fields"]["device"]["status"] == "blocked"
    with pytest.raises(device.DeviceProvenanceError, match="remains blocked"):
        device.apply_composite_condition_authority(
            campaign_id,
            expected_content_sha256=blocked["content_sha256"],
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )
    manifest = json.loads(
        (
            runtime / "campaigns" / campaign_id / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    target = manifest["cases"][-1]["recording_id"]
    correction = device.correct_device_provenance(
        campaign_id,
        target,
        physical_device_label="Phone B",
        attested_by="fixture-operator",
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert correction["status"] == "observed"
    replay = device.replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert replay["initial_records"] == 7
    assert replay["corrections"] == 1
    passing = device.preview_composite_condition_authority(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    assert passing["condition_coverage"]["terminal_selection_eligible"] is True


def test_source_and_record_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, runtime, campaign_id, sources = _freeze(tmp_path, monkeypatch)
    source_body = sources[0].read_bytes()
    sources[0].write_bytes(source_body + b"drift")
    with pytest.raises(device.DeviceProvenanceError, match="source binding drifted"):
        device.replay_device_campaign(
            campaign_id,
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )
    sources[0].write_bytes(source_body)
    device.open_next_device_case(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    result = device.record_device_provenance(
        campaign_id,
        physical_device_label="Phone A",
        attested_by="fixture-operator",
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    record_path = Path(result["record_path"])
    record = json.loads(record_path.read_text())
    record["evidence_basis"] = "inferred_from_filename"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    record_path.chmod(0o600)
    with pytest.raises(device.DeviceProvenanceError, match="record history"):
        device.replay_device_campaign(
            campaign_id,
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )


def test_replay_rejects_rehashed_record_with_forged_cursor_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, condition, runtime, campaign_id, _sources = _freeze(
        tmp_path, monkeypatch
    )
    device.open_next_device_case(
        campaign_id,
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    result = device.record_device_provenance(
        campaign_id,
        physical_device_label="Phone A",
        attested_by="fixture-operator",
        corpus_manifest_path=corpus,
        condition_manifest_path=condition,
        runtime_root=runtime,
    )
    record_path = Path(result["record_path"])
    record = json.loads(record_path.read_text())
    record["open_receipt_sha256"] = "f" * 64
    core = dict(record)
    core.pop("record_sha256")
    record["record_sha256"] = device._canonical_hash(core)
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    record_path.chmod(0o600)

    with pytest.raises(device.DeviceProvenanceError, match="authoritative open"):
        device.replay_device_campaign(
            campaign_id,
            corpus_manifest_path=corpus,
            condition_manifest_path=condition,
            runtime_root=runtime,
        )
