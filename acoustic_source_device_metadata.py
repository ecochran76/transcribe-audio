"""Immutable source-embedded physical-device metadata for Plan 0037."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import acoustic_device_provenance as device
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PLAN_SCHEMA = "transcribe-audio.acoustic-source-device-metadata-plan.v1"
MANIFEST_SCHEMA = "transcribe-audio.acoustic-source-device-metadata-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.acoustic-source-device-metadata-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.acoustic-source-device-metadata-replay.v1"
ALLOWLISTED_TAG = "Samsung:SamsungModel"
EXPECTED_RECORDINGS = 7


class SourceDeviceMetadataError(ValueError):
    """Raised when source-embedded device metadata is not trustworthy."""


def _extract(path: Path) -> tuple[str, str]:
    version = subprocess.run(
        ["exiftool", "-ver"], check=False, capture_output=True, text=True
    )
    result = subprocess.run(
        ["exiftool", "-j", "-G1", "-SamsungModel", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if version.returncode != 0 or result.returncode != 0:
        raise SourceDeviceMetadataError("Device metadata extractor failed.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SourceDeviceMetadataError("Device metadata extractor returned invalid JSON.") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise SourceDeviceMetadataError("Device metadata extractor returned an invalid body.")
    unexpected = set(payload[0]) - {"SourceFile", ALLOWLISTED_TAG}
    if unexpected or payload[0].get("SourceFile") != str(path):
        raise SourceDeviceMetadataError("Device metadata extractor exceeded its allowlist.")
    value = str(payload[0].get(ALLOWLISTED_TAG) or "").strip()
    return version.stdout.strip(), value


def _authority_paths(root: Path, campaign_id: str, authority_id: str = "") -> dict[str, Path]:
    campaign = device._paths(root, campaign_id)["campaign"]
    base = campaign / "source-device-metadata"
    selected = base / authority_id if authority_id else base
    return {
        "root": device._paths(root, campaign_id)["root"],
        "base": base,
        "authority": selected,
        "manifest": selected / "manifest.json",
        "receipt": selected / "apply-receipt.json",
    }


def preview_source_device_metadata(
    campaign_id: str,
    source_paths: Mapping[str, Path],
    *,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Build a deterministic no-write exact-seven source metadata plan."""
    replay = device.replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    campaign_paths = device._paths(root, campaign_id)
    require_private_file(campaign_paths["manifest"], campaign_paths["root"])
    manifest_bytes = campaign_paths["manifest"].read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise SourceDeviceMetadataError("Frozen campaign manifest is invalid JSON.") from exc
    if not isinstance(manifest, dict):
        raise SourceDeviceMetadataError("Frozen campaign manifest body is invalid.")
    cases = list(manifest.get("cases") or [])
    recording_ids = [str(case.get("recording_id") or "") for case in cases]
    positions = [case.get("position") for case in cases]
    expected_ids = set(recording_ids)
    if (
        manifest_digest != replay.get("manifest_sha256")
        or len(cases) != EXPECTED_RECORDINGS
        or len(expected_ids) != EXPECTED_RECORDINGS
        or positions != list(range(1, EXPECTED_RECORDINGS + 1))
        or set(source_paths) != expected_ids
    ):
        raise SourceDeviceMetadataError("Source mapping is not the exact frozen cohort.")
    extractor_version = ""
    results: list[dict[str, Any]] = []
    for case in cases:
        recording_id = str(case["recording_id"])
        path = Path(source_paths[recording_id]).expanduser().resolve(strict=True)
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != case.get("source_sha256") or size != case.get("source_bytes"):
            raise SourceDeviceMetadataError("Recovered source binding does not match the frozen case.")
        current_version, label = _extract(path)
        if sha256_file(path) != digest or path.stat().st_size != size:
            raise SourceDeviceMetadataError(
                "Recovered source changed during metadata extraction."
            )
        if extractor_version and current_version != extractor_version:
            raise SourceDeviceMetadataError("Device metadata extractor version changed mid-preview.")
        extractor_version = current_version
        if label:
            normalized = device._normalize_label(label)
            status = "observed"
            evidence_basis = "source_embedded_manufacturer_hardware_model"
            device_id = device._device_id(normalized)
            absence_reason = ""
        else:
            normalized = ""
            status = "unavailable"
            evidence_basis = "allowlisted_hardware_model_absent"
            device_id = ""
            absence_reason = "hardware_model_tag_absent"
        results.append(
            {
                "position": int(case["position"]),
                "recording_id": recording_id,
                "source_path": str(path),
                "source_sha256": digest,
                "source_bytes": size,
                "status": status,
                "metadata_tag": ALLOWLISTED_TAG,
                "physical_device_label": label,
                "normalized_device_label": normalized,
                "device_id": device_id,
                "evidence_basis": evidence_basis,
                "absence_reason": absence_reason,
            }
        )
    observed_count = sum(item["status"] == "observed" for item in results)
    unavailable_count = sum(item["status"] == "unavailable" for item in results)
    if (observed_count, unavailable_count) != (5, 2):
        raise SourceDeviceMetadataError(
            "Source metadata result distribution differs from the reviewed production inventory."
        )
    core = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": replay["manifest_sha256"],
        "campaign_records_state_sha256": replay["records_state_sha256"],
        "extractor": {
            "name": "exiftool",
            "version": extractor_version,
            "arguments": ["-j", "-G1", "-SamsungModel"],
        },
        "allowlisted_tags": [ALLOWLISTED_TAG],
        "results": results,
        "observed_count": observed_count,
        "unavailable_count": unavailable_count,
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_reveal_evaluation": False,
        "will_perform_external_write": False,
    }
    digest = device._canonical_hash(core)
    return {
        **core,
        "authority_id": f"source-device-metadata-{digest[:24]}",
        "content_sha256": digest,
    }


def _portable_receipt(preview: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "authority_id": preview["authority_id"],
        "campaign_id": preview["campaign_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "recordings": EXPECTED_RECORDINGS,
        "observed_count": preview["observed_count"],
        "unavailable_count": preview["unavailable_count"],
        "device_ids": sorted(
            {item["device_id"] for item in preview["results"] if item["device_id"]}
        ),
        "mode": "0600",
        "contains_source_paths": False,
        "contains_device_labels": False,
        "will_perform_external_write": False,
    }


def apply_source_device_metadata(
    campaign_id: str,
    source_paths: Mapping[str, Path],
    *,
    expected_content_sha256: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze the reviewed exact-seven source metadata authority."""
    preview = preview_source_device_metadata(
        campaign_id,
        source_paths,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    if preview["content_sha256"] != expected_content_sha256:
        raise SourceDeviceMetadataError("Reviewed source metadata hash is stale.")
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    paths = _authority_paths(root, campaign_id, preview["authority_id"])
    existing_manifests = sorted(paths["base"].glob("*/manifest.json"))
    if existing_manifests and existing_manifests != [paths["manifest"]]:
        raise SourceDeviceMetadataError(
            "A different source metadata authority already binds this campaign."
        )
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_source_device_metadata(
            paths["manifest"],
            campaign_id=campaign_id,
            source_paths=source_paths,
            corpus_manifest_path=corpus_manifest_path,
            condition_manifest_path=condition_manifest_path,
            runtime_root=runtime_root,
        )
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise SourceDeviceMetadataError("Partial source metadata authority exists.")
    ensure_private_tree(paths["root"], paths["authority"])
    manifest = {
        **preview,
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "applied_at": device._utc_now(),
    }
    write_immutable_private_json(paths["manifest"], manifest, volatile_fields=("applied_at",))
    receipt = _portable_receipt(preview, paths["manifest"])
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent": False,
    }


def replay_source_device_metadata(
    manifest_path: Path,
    *,
    campaign_id: str,
    source_paths: Mapping[str, Path],
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay source bytes, metadata extraction, manifest, and receipt exactly."""
    preview = preview_source_device_metadata(
        campaign_id,
        source_paths,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    selected = manifest_path.expanduser().resolve(strict=True)
    require_private_file(selected, device._paths(root, campaign_id)["root"])
    manifest = read_private_object(selected)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    if comparable != {**preview, "schema_version": MANIFEST_SCHEMA, "status": "complete"}:
        raise SourceDeviceMetadataError("Source metadata full-body replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    require_private_file(receipt_path, device._paths(root, campaign_id)["root"])
    expected_receipt = _portable_receipt(preview, selected)
    if read_private_object(receipt_path) != expected_receipt:
        raise SourceDeviceMetadataError("Source metadata receipt replay mismatch.")
    return {
        "schema_version": REPLAY_SCHEMA,
        "authority_id": preview["authority_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "observed_count": preview["observed_count"],
        "unavailable_count": preview["unavailable_count"],
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }
