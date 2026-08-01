from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import acoustic_source_device_metadata as metadata
import acoustic_generation2_authority as generation2
from acoustic_audio_derivatives import sha256_file


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    campaign_id = "device-provenance-fixture"
    campaign = root / "campaigns" / campaign_id
    campaign.mkdir(parents=True, mode=0o700)
    (root / "campaigns").chmod(0o700)
    campaign.chmod(0o700)
    cases = []
    source_paths = {}
    for position in range(1, 8):
        source = tmp_path / "sources" / f"source-{position}.m4a"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{position}".encode())
        recording_id = f"recording-{position}"
        cases.append(
            {
                "position": position,
                "recording_id": recording_id,
                "source_sha256": sha256_file(source),
                "source_bytes": source.stat().st_size,
            }
        )
        source_paths[recording_id] = source
    condition = root / "conditions" / "runs" / "fixture" / "condition-manifest.json"
    condition.parent.mkdir(parents=True)
    for directory in (root / "conditions", root / "conditions" / "runs", condition.parent):
        directory.chmod(0o700)
    condition_body = {
        "condition_coverage": {
            "fields": {
                field: {
                    "observed_values": [] if field == "device" else ["a", "b"],
                    "observed_value_count": 0 if field == "device" else 2,
                    "missing_recordings": 7 if field == "device" else 0,
                    "status": "blocked" if field == "device" else "pass",
                }
                for field in metadata.device.conditions.CONDITION_FIELDS
            },
            "terminal_selection_eligible": False,
            "blockers": ["device_condition_coverage_below_policy"],
        }
    }
    condition.write_text(json.dumps(condition_body), encoding="utf-8")
    condition.chmod(0o600)
    manifest = campaign / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": cases,
                "condition_authority": {
                    "manifest_path": str(condition),
                    "manifest_sha256": sha256_file(condition),
                    "content_sha256": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    monkeypatch.setattr(
        metadata.device,
        "replay_device_campaign",
        lambda *args, **kwargs: {
            "manifest_sha256": sha256_file(manifest),
            "records_state_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        metadata,
        "_extract",
        lambda path: (
            "13.10",
            "SM-S908U" if path.stem in {"source-1", "source-3", "source-5", "source-6", "source-7"} else "",
        ),
    )
    return root, campaign_id, source_paths, manifest


def _fixture_condition(manifest: Path) -> Path:
    body = json.loads(manifest.read_text(encoding="utf-8"))
    return Path(body["condition_authority"]["manifest_path"])


def test_preview_accepts_only_exact_sources_and_preserves_absence(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    preview = metadata.preview_source_device_metadata(
        campaign_id,
        source_paths,
        corpus_manifest_path=tmp_path / "corpus.json",
        condition_manifest_path=tmp_path / "condition.json",
        runtime_root=root,
    )
    assert preview["observed_count"] == 5
    assert preview["unavailable_count"] == 2
    assert {item["position"] for item in preview["results"] if item["status"] == "unavailable"} == {2, 4}
    assert {item["evidence_basis"] for item in preview["results"] if item["status"] == "observed"} == {
        "source_embedded_manufacturer_hardware_model"
    }
    assert len({item["device_id"] for item in preview["results"] if item["device_id"]}) == 1


def test_preview_rejects_source_drift(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    source_paths["recording-1"].write_bytes(b"drift")
    with pytest.raises(metadata.SourceDeviceMetadataError, match="frozen case"):
        metadata.preview_source_device_metadata(
            campaign_id,
            source_paths,
            corpus_manifest_path=tmp_path / "corpus.json",
            condition_manifest_path=tmp_path / "condition.json",
            runtime_root=root,
        )


def test_apply_replay_and_portable_receipt_privacy(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    kwargs = {
        "corpus_manifest_path": tmp_path / "corpus.json",
        "condition_manifest_path": tmp_path / "condition.json",
        "runtime_root": root,
    }
    preview = metadata.preview_source_device_metadata(campaign_id, source_paths, **kwargs)
    applied = metadata.apply_source_device_metadata(
        campaign_id,
        source_paths,
        expected_content_sha256=preview["content_sha256"],
        **kwargs,
    )
    receipt = json.loads(Path(applied["receipt_path"]).read_text(encoding="utf-8"))
    serialized = json.dumps(receipt)
    assert "SM-S908U" not in serialized
    assert str(next(iter(source_paths.values()))) not in serialized
    assert receipt["observed_count"] == 5
    assert receipt["unavailable_count"] == 2
    assert set(receipt) == {
        "schema_version", "authority_id", "campaign_id", "content_sha256",
        "manifest_sha256", "recordings", "observed_count", "unavailable_count",
        "device_ids", "mode", "contains_source_paths", "contains_device_labels",
        "will_perform_external_write",
    }
    replay = metadata.replay_source_device_metadata(
        Path(applied["manifest_path"]),
        campaign_id=campaign_id,
        source_paths=source_paths,
        **kwargs,
    )
    assert replay["full_body_match"] is True
    assert metadata.apply_source_device_metadata(
        campaign_id,
        source_paths,
        expected_content_sha256=preview["content_sha256"],
        **kwargs,
    )["idempotent"] is True


def test_apply_rejects_second_authority_for_same_campaign(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    kwargs = {
        "corpus_manifest_path": tmp_path / "corpus.json",
        "condition_manifest_path": tmp_path / "condition.json",
        "runtime_root": root,
    }
    preview = metadata.preview_source_device_metadata(campaign_id, source_paths, **kwargs)
    metadata.apply_source_device_metadata(
        campaign_id,
        source_paths,
        expected_content_sha256=preview["content_sha256"],
        **kwargs,
    )
    monkeypatch.setattr(
        metadata,
        "_extract",
        lambda path: (
            "13.10",
            "DIFFERENT"
            if path.stem == "source-1"
            else "SM-S908U"
            if path.stem in {"source-3", "source-5", "source-6", "source-7"}
            else "",
        ),
    )
    changed = metadata.preview_source_device_metadata(campaign_id, source_paths, **kwargs)
    with pytest.raises(metadata.SourceDeviceMetadataError, match="different source metadata"):
        metadata.apply_source_device_metadata(
            campaign_id,
            source_paths,
            expected_content_sha256=changed["content_sha256"],
            **kwargs,
        )


def test_preview_rejects_detached_or_duplicate_campaign_manifest(tmp_path, monkeypatch):
    root, campaign_id, source_paths, manifest = _fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][-1] = dict(payload["cases"][-2])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o600)
    with pytest.raises(metadata.SourceDeviceMetadataError, match="exact frozen cohort"):
        metadata.preview_source_device_metadata(
            campaign_id,
            source_paths,
            corpus_manifest_path=tmp_path / "corpus.json",
            condition_manifest_path=tmp_path / "condition.json",
            runtime_root=root,
        )


def test_preview_ignores_detached_loaded_campaign_body(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        metadata.device,
        "_load_campaign",
        lambda paths: {"cases": [{"position": 99, "recording_id": "forged"}] * 7},
    )
    preview = metadata.preview_source_device_metadata(
        campaign_id,
        source_paths,
        corpus_manifest_path=tmp_path / "corpus.json",
        condition_manifest_path=tmp_path / "condition.json",
        runtime_root=root,
    )
    assert [item["position"] for item in preview["results"]] == list(range(1, 8))


def test_preview_rejects_source_swap_during_extraction(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)

    def swapping_extract(path):
        if path.stem == "source-1":
            path.write_bytes(b"swapped-after-prehash")
        return "13.10", "SM-S908U"

    monkeypatch.setattr(metadata, "_extract", swapping_extract)
    with pytest.raises(metadata.SourceDeviceMetadataError, match="changed during"):
        metadata.preview_source_device_metadata(
            campaign_id,
            source_paths,
            corpus_manifest_path=tmp_path / "corpus.json",
            condition_manifest_path=tmp_path / "condition.json",
            runtime_root=root,
        )


def test_preview_rejects_result_distribution_drift(tmp_path, monkeypatch):
    root, campaign_id, source_paths, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(metadata, "_extract", lambda path: ("13.10", "SM-S908U"))
    with pytest.raises(metadata.SourceDeviceMetadataError, match="distribution"):
        metadata.preview_source_device_metadata(
            campaign_id,
            source_paths,
            corpus_manifest_path=tmp_path / "corpus.json",
            condition_manifest_path=tmp_path / "condition.json",
            runtime_root=root,
        )


def test_extractor_rejects_non_allowlisted_output(tmp_path, monkeypatch):
    source = tmp_path / "source.m4a"
    source.write_bytes(b"source")

    class Result:
        returncode = 0
        stdout = "13.10"

    calls = iter(
        [
            Result(),
            type("Payload", (), {"returncode": 0, "stdout": json.dumps([{
                "SourceFile": str(source.resolve()),
                "Samsung:SamsungModel": "SM-S908U",
                "QuickTime:HandlerDescription": "SoundHandle",
            }])})(),
        ]
    )
    monkeypatch.setattr(metadata.subprocess, "run", lambda *args, **kwargs: next(calls))
    with pytest.raises(metadata.SourceDeviceMetadataError, match="allowlist"):
        metadata._extract(source.resolve())


def _webcam_facts(manifest: Path) -> dict[str, str]:
    cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
    return {
        cases[1]["recording_id"]: "Fixture Webcam",
        cases[3]["recording_id"]: "Fixture Webcam",
    }


def _webcam_device_id() -> str:
    return metadata.device._device_id("fixture webcam")


def test_sparse_operator_preview_is_exactly_cases_two_and_four(tmp_path, monkeypatch):
    root, campaign_id, _, manifest = _fixture(tmp_path, monkeypatch)
    preview = metadata.preview_sparse_operator_device(
        campaign_id,
        _webcam_facts(manifest),
        attested_by="operator-confirmed-2026-08-01",
        expected_device_id=_webcam_device_id(),
        corpus_manifest_path=tmp_path / "corpus.json",
        condition_manifest_path=tmp_path / "condition.json",
        runtime_root=root,
    )
    assert [item["position"] for item in preview["results"]] == [2, 4]
    assert {item["evidence_basis"] for item in preview["results"]} == {
        "direct_operator_knowledge"
    }
    assert len({item["device_id"] for item in preview["results"]}) == 1


def test_sparse_operator_rejects_wrong_case_or_device(tmp_path, monkeypatch):
    root, campaign_id, _, manifest = _fixture(tmp_path, monkeypatch)
    facts = _webcam_facts(manifest)
    first = next(iter(facts))
    facts[first] = "Inferred Device"
    with pytest.raises(metadata.SourceDeviceMetadataError, match="reviewed authority"):
        metadata.preview_sparse_operator_device(
            campaign_id,
            facts,
            attested_by="operator",
            expected_device_id=_webcam_device_id(),
            corpus_manifest_path=tmp_path / "corpus.json",
            condition_manifest_path=tmp_path / "condition.json",
            runtime_root=root,
        )


def test_sparse_operator_apply_replay_and_receipt_privacy(tmp_path, monkeypatch):
    root, campaign_id, _, manifest = _fixture(tmp_path, monkeypatch)
    facts = _webcam_facts(manifest)
    kwargs = {
        "attested_by": "operator-confirmed-2026-08-01",
        "expected_device_id": _webcam_device_id(),
        "corpus_manifest_path": tmp_path / "corpus.json",
        "condition_manifest_path": tmp_path / "condition.json",
        "runtime_root": root,
    }
    preview = metadata.preview_sparse_operator_device(campaign_id, facts, **kwargs)
    applied = metadata.apply_sparse_operator_device(
        campaign_id,
        facts,
        expected_content_sha256=preview["content_sha256"],
        **kwargs,
    )
    receipt = json.loads(Path(applied["receipt_path"]).read_text(encoding="utf-8"))
    serialized = json.dumps(receipt)
    assert "Fixture Webcam" not in serialized
    assert "operator-confirmed" not in serialized
    assert set(receipt) == {
        "schema_version", "authority_id", "campaign_id", "content_sha256",
        "manifest_sha256", "observed_count", "positions", "device_ids", "mode",
        "contains_device_labels", "contains_operator_identifier",
        "will_perform_external_write",
    }
    replay = metadata.replay_sparse_operator_device(
        Path(applied["manifest_path"]),
        campaign_id=campaign_id,
        facts=facts,
        **kwargs,
    )
    assert replay["full_body_match"] is True
    assert metadata.apply_sparse_operator_device(
        campaign_id,
        facts,
        expected_content_sha256=preview["content_sha256"],
        **kwargs,
    )["idempotent"] is True


def test_augmented_composite_merges_five_source_and_two_operator_facts(
    tmp_path, monkeypatch
):
    root, campaign_id, source_paths, campaign_manifest = _fixture(tmp_path, monkeypatch)
    condition = _fixture_condition(campaign_manifest)
    corpus = tmp_path / "corpus.json"
    common = {
        "corpus_manifest_path": corpus,
        "condition_manifest_path": condition,
        "runtime_root": root,
    }
    source_preview = metadata.preview_source_device_metadata(
        campaign_id, source_paths, **common
    )
    source_applied = metadata.apply_source_device_metadata(
        campaign_id,
        source_paths,
        expected_content_sha256=source_preview["content_sha256"],
        **common,
    )
    facts = _webcam_facts(campaign_manifest)
    operator_common = {**common, "attested_by": "operator-confirmed-2026-08-01"}
    operator_common["expected_device_id"] = _webcam_device_id()
    operator_preview = metadata.preview_sparse_operator_device(
        campaign_id, facts, **operator_common
    )
    operator_applied = metadata.apply_sparse_operator_device(
        campaign_id,
        facts,
        expected_content_sha256=operator_preview["content_sha256"],
        **operator_common,
    )
    composite_kwargs = {
        **common,
        "source_metadata_manifest_path": Path(source_applied["manifest_path"]),
        "source_paths": source_paths,
        "operator_manifest_path": Path(operator_applied["manifest_path"]),
        "operator_facts": facts,
        "attested_by": "operator-confirmed-2026-08-01",
        "operator_expected_device_id": _webcam_device_id(),
    }
    preview = metadata.preview_augmented_composite(campaign_id, **composite_kwargs)
    assert preview["authoritative_device_evidence_count"] == 7
    assert preview["direct_operator_observed_count"] == 2
    assert preview["source_metadata_observed_count"] == 5
    assert preview["condition_coverage"]["terminal_selection_eligible"] is True
    assert preview["condition_coverage"]["fields"]["device"] == {
        "observed_values": sorted(
            {item["device_id"] for item in preview["evidence"]}
        ),
        "observed_value_count": 2,
        "missing_recordings": 0,
        "status": "pass",
    }
    assert [item["evidence_basis"] for item in preview["evidence"]] == [
        "source_embedded_manufacturer_hardware_model",
        "direct_operator_knowledge",
        "source_embedded_manufacturer_hardware_model",
        "direct_operator_knowledge",
        "source_embedded_manufacturer_hardware_model",
        "source_embedded_manufacturer_hardware_model",
        "source_embedded_manufacturer_hardware_model",
    ]
    applied = metadata.apply_augmented_composite(
        campaign_id,
        expected_content_sha256=preview["content_sha256"],
        **composite_kwargs,
    )
    receipt = json.loads(Path(applied["receipt_path"]).read_text(encoding="utf-8"))
    assert "Fixture Webcam" not in json.dumps(receipt)
    assert "SM-S908U" not in json.dumps(receipt)
    assert "operator-confirmed" not in json.dumps(receipt)
    assert set(receipt) == {
        "schema_version", "composite_id", "content_sha256", "manifest_sha256",
        "condition_coverage", "authoritative_device_evidence_count",
        "direct_operator_observed_count", "source_metadata_observed_count", "mode",
        "contains_device_labels", "contains_operator_identifier",
        "will_perform_external_write",
    }
    replay = metadata.replay_augmented_composite(
        Path(applied["manifest_path"]), campaign_id=campaign_id, **composite_kwargs
    )
    assert replay["full_body_match"] is True
    campaign_body = json.loads(campaign_manifest.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        generation2,
        "EXPECTED_CONDITION_MANIFEST_SHA256",
        campaign_body["condition_authority"]["manifest_sha256"],
    )
    monkeypatch.setattr(
        generation2,
        "EXPECTED_CONDITION_CONTENT_SHA256",
        campaign_body["condition_authority"]["content_sha256"],
    )
    binding = generation2._composite_binding(
        json.loads(Path(applied["manifest_path"]).read_text(encoding="utf-8")),
        replay,
    )
    assert binding["authoritative_device_evidence_count"] == 7
    assert binding["direct_operator_observed_count"] == 2
    assert binding["source_metadata_observed_count"] == 5
    stored_manifest = json.loads(
        Path(applied["manifest_path"]).read_text(encoding="utf-8")
    )
    for mutation in ("empty", "duplicate", "wrong_basis", "device_mismatch"):
        forged = copy.deepcopy(stored_manifest)
        if mutation == "empty":
            forged["evidence"] = []
        elif mutation == "duplicate":
            forged["evidence"][1] = copy.deepcopy(forged["evidence"][0])
        elif mutation == "wrong_basis":
            forged["evidence"][0]["evidence_basis"] = "direct_operator_knowledge"
        else:
            forged["evidence"][0]["device_id"] = "physical-device-forged"
        core = {
            key: value
            for key, value in forged.items()
            if key not in {"applied_at", "composite_id", "content_sha256", "status"}
        }
        core["schema_version"] = metadata.AUGMENTED_COMPOSITE_PLAN_SCHEMA
        digest = generation2._canonical_hash(core)
        forged["content_sha256"] = digest
        forged["composite_id"] = f"augmented-composite-{digest[:24]}"
        forged_replay = {
            **replay,
            "content_sha256": digest,
            "composite_id": forged["composite_id"],
        }
        with pytest.raises(generation2.Generation2AuthorityError):
            generation2._composite_binding(forged, forged_replay)
    assert metadata.apply_augmented_composite(
        campaign_id,
        expected_content_sha256=preview["content_sha256"],
        **composite_kwargs,
    )["idempotent"] is True
