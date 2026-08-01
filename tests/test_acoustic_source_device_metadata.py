from __future__ import annotations

import json
from pathlib import Path

import pytest

import acoustic_source_device_metadata as metadata
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
    manifest = campaign / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
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
