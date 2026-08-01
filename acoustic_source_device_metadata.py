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
OPERATOR_PLAN_SCHEMA = "transcribe-audio.acoustic-sparse-device-operator-plan.v1"
OPERATOR_MANIFEST_SCHEMA = "transcribe-audio.acoustic-sparse-device-operator-manifest.v1"
OPERATOR_RECEIPT_SCHEMA = "transcribe-audio.acoustic-sparse-device-operator-receipt.v1"
OPERATOR_REPLAY_SCHEMA = "transcribe-audio.acoustic-sparse-device-operator-replay.v1"
AUGMENTED_COMPOSITE_PLAN_SCHEMA = "transcribe-audio.acoustic-augmented-composite-plan.v1"
AUGMENTED_COMPOSITE_MANIFEST_SCHEMA = "transcribe-audio.acoustic-augmented-composite-manifest.v1"
AUGMENTED_COMPOSITE_RECEIPT_SCHEMA = "transcribe-audio.acoustic-augmented-composite-receipt.v1"
AUGMENTED_COMPOSITE_REPLAY_SCHEMA = "transcribe-audio.acoustic-augmented-composite-replay.v1"
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


def _portable_receipt(
    preview: Mapping[str, Any], manifest_path: Path, manifest_sha256: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "authority_id": preview["authority_id"],
        "campaign_id": preview["campaign_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256 or sha256_file(manifest_path),
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
    private_root = device._paths(root, campaign_id)["root"]
    manifest, selected_sha256 = _private_snapshot(selected, private_root)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    if comparable != {**preview, "schema_version": MANIFEST_SCHEMA, "status": "complete"}:
        raise SourceDeviceMetadataError("Source metadata full-body replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    receipt, _ = _private_snapshot(receipt_path, private_root)
    expected_receipt = _portable_receipt(preview, selected, selected_sha256)
    if receipt != expected_receipt:
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


def _campaign_snapshot(
    campaign_id: str,
    replay: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = device._paths(root, campaign_id)
    require_private_file(paths["manifest"], paths["root"])
    body = paths["manifest"].read_bytes()
    if hashlib.sha256(body).hexdigest() != replay.get("manifest_sha256"):
        raise SourceDeviceMetadataError("Frozen campaign manifest detached from replay.")
    try:
        manifest = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SourceDeviceMetadataError("Frozen campaign manifest is invalid JSON.") from exc
    if not isinstance(manifest, dict):
        raise SourceDeviceMetadataError("Frozen campaign manifest body is invalid.")
    return manifest, paths


def _private_snapshot(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    selected = path.expanduser().resolve(strict=True)
    require_private_file(selected, root)
    body = selected.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SourceDeviceMetadataError("Private authority is invalid JSON.") from exc
    if not isinstance(value, dict):
        raise SourceDeviceMetadataError("Private authority body is invalid.")
    return value, digest


def _operator_paths(root: Path, campaign_id: str, authority_id: str = "") -> dict[str, Path]:
    campaign_paths = device._paths(root, campaign_id)
    base = campaign_paths["campaign"] / "sparse-operator-device"
    selected = base / authority_id if authority_id else base
    return {
        "root": campaign_paths["root"],
        "base": base,
        "authority": selected,
        "manifest": selected / "manifest.json",
        "receipt": selected / "apply-receipt.json",
    }


def preview_sparse_operator_device(
    campaign_id: str,
    facts: Mapping[str, str],
    *,
    attested_by: str,
    expected_device_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Build a no-write authority for the operator-confirmed cases 2 and 4."""
    if not str(attested_by).strip():
        raise SourceDeviceMetadataError("Sparse operator authority requires an attestor.")
    replay = device.replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    manifest, _ = _campaign_snapshot(campaign_id, replay, root)
    cases = list(manifest.get("cases") or [])
    if (
        len(cases) != EXPECTED_RECORDINGS
        or [case.get("position") for case in cases] != list(range(1, 8))
        or len({str(case.get("recording_id") or "") for case in cases}) != 7
    ):
        raise SourceDeviceMetadataError("Sparse operator campaign cohort is invalid.")
    selected = [cases[1], cases[3]]
    selected_ids = {str(case["recording_id"]) for case in selected}
    if set(facts) != selected_ids:
        raise SourceDeviceMetadataError("Sparse operator facts must cover exactly cases 2 and 4.")
    results = []
    for case in selected:
        recording_id = str(case["recording_id"])
        label = " ".join(str(facts[recording_id]).split())
        normalized = device._normalize_label(label)
        device_id = device._device_id(normalized)
        if device_id != expected_device_id:
            raise SourceDeviceMetadataError("Sparse operator device differs from reviewed authority.")
        results.append(
            {
                "position": int(case["position"]),
                "recording_id": recording_id,
                "source_sha256": case["source_sha256"],
                "physical_device_label": label,
                "normalized_device_label": normalized,
                "device_id": device_id,
                "evidence_basis": "direct_operator_knowledge",
                "attested_by": str(attested_by).strip(),
            }
        )
    core = {
        "schema_version": OPERATOR_PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": replay["manifest_sha256"],
        "campaign_records_state_sha256": replay["records_state_sha256"],
        "results": results,
        "observed_count": 2,
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_reveal_evaluation": False,
        "will_perform_external_write": False,
    }
    digest = device._canonical_hash(core)
    return {
        **core,
        "authority_id": f"sparse-operator-device-{digest[:24]}",
        "content_sha256": digest,
    }


def _operator_receipt(
    preview: Mapping[str, Any], manifest_path: Path, manifest_sha256: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": OPERATOR_RECEIPT_SCHEMA,
        "authority_id": preview["authority_id"],
        "campaign_id": preview["campaign_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256 or sha256_file(manifest_path),
        "observed_count": 2,
        "positions": [2, 4],
        "device_ids": sorted({item["device_id"] for item in preview["results"]}),
        "mode": "0600",
        "contains_device_labels": False,
        "contains_operator_identifier": False,
        "will_perform_external_write": False,
    }


def apply_sparse_operator_device(
    campaign_id: str,
    facts: Mapping[str, str],
    *,
    attested_by: str,
    expected_device_id: str,
    expected_content_sha256: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze exactly the two operator-confirmed webcam facts."""
    preview = preview_sparse_operator_device(
        campaign_id,
        facts,
        attested_by=attested_by,
        expected_device_id=expected_device_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    if preview["content_sha256"] != expected_content_sha256:
        raise SourceDeviceMetadataError("Reviewed sparse operator hash is stale.")
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    paths = _operator_paths(root, campaign_id, preview["authority_id"])
    existing = sorted(paths["base"].glob("*/manifest.json"))
    if existing and existing != [paths["manifest"]]:
        raise SourceDeviceMetadataError("A different sparse operator authority already exists.")
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_sparse_operator_device(
            paths["manifest"],
            campaign_id=campaign_id,
            facts=facts,
            attested_by=attested_by,
            expected_device_id=expected_device_id,
            corpus_manifest_path=corpus_manifest_path,
            condition_manifest_path=condition_manifest_path,
            runtime_root=runtime_root,
        )
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise SourceDeviceMetadataError("Partial sparse operator authority exists.")
    ensure_private_tree(paths["root"], paths["authority"])
    manifest = {
        **preview,
        "schema_version": OPERATOR_MANIFEST_SCHEMA,
        "status": "complete",
        "applied_at": device._utc_now(),
    }
    write_immutable_private_json(paths["manifest"], manifest, volatile_fields=("applied_at",))
    receipt = _operator_receipt(preview, paths["manifest"])
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent": False,
    }


def replay_sparse_operator_device(
    manifest_path: Path,
    *,
    campaign_id: str,
    facts: Mapping[str, str],
    attested_by: str,
    expected_device_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay the exact sparse operator facts and sanitized receipt."""
    preview = preview_sparse_operator_device(
        campaign_id,
        facts,
        attested_by=attested_by,
        expected_device_id=expected_device_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    selected = manifest_path.expanduser().resolve(strict=True)
    private_root = device._paths(root, campaign_id)["root"]
    manifest, selected_sha256 = _private_snapshot(selected, private_root)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    if comparable != {**preview, "schema_version": OPERATOR_MANIFEST_SCHEMA, "status": "complete"}:
        raise SourceDeviceMetadataError("Sparse operator full-body replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    receipt, _ = _private_snapshot(receipt_path, private_root)
    expected_receipt = _operator_receipt(preview, selected, selected_sha256)
    if receipt != expected_receipt:
        raise SourceDeviceMetadataError("Sparse operator receipt replay mismatch.")
    return {
        "schema_version": OPERATOR_REPLAY_SCHEMA,
        "authority_id": preview["authority_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "observed_count": 2,
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }


def _bound_private_object(path: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    value, digest = _private_snapshot(path, root)
    if digest != expected_sha256:
        raise SourceDeviceMetadataError("Private predecessor authority hash drifted.")
    return value


def _augmented_paths(root: Path, campaign_id: str, composite_id: str = "") -> dict[str, Path]:
    campaign_paths = device._paths(root, campaign_id)
    base = campaign_paths["campaign"] / "augmented-composite"
    selected = base / composite_id if composite_id else base
    return {
        "root": campaign_paths["root"],
        "base": base,
        "composite": selected,
        "manifest": selected / "manifest.json",
        "receipt": selected / "apply-receipt.json",
    }


def preview_augmented_composite(
    campaign_id: str,
    *,
    source_metadata_manifest_path: Path,
    source_paths: Mapping[str, Path],
    operator_manifest_path: Path,
    operator_facts: Mapping[str, str],
    attested_by: str,
    operator_expected_device_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Merge five source facts and two direct operator facts without rewriting either."""
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    source_replay = replay_source_device_metadata(
        source_metadata_manifest_path,
        campaign_id=campaign_id,
        source_paths=source_paths,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    operator_replay = replay_sparse_operator_device(
        operator_manifest_path,
        campaign_id=campaign_id,
        facts=operator_facts,
        attested_by=attested_by,
        expected_device_id=operator_expected_device_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    campaign_replay = device.replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    campaign, campaign_paths = _campaign_snapshot(campaign_id, campaign_replay, root)
    source_manifest = _bound_private_object(
        source_metadata_manifest_path,
        campaign_paths["root"],
        source_replay["manifest_sha256"],
    )
    operator_manifest = _bound_private_object(
        operator_manifest_path,
        campaign_paths["root"],
        operator_replay["manifest_sha256"],
    )
    source_by_id = {
        str(item["recording_id"]): item for item in source_manifest.get("results") or []
    }
    operator_by_id = {
        str(item["recording_id"]): item for item in operator_manifest.get("results") or []
    }
    cases = list(campaign.get("cases") or [])
    if len(source_by_id) != 7 or len(operator_by_id) != 2 or len(cases) != 7:
        raise SourceDeviceMetadataError("Augmented composite predecessor coverage is invalid.")
    evidence: list[dict[str, Any]] = []
    for case in cases:
        recording_id = str(case["recording_id"])
        source_item = source_by_id.get(recording_id)
        operator_item = operator_by_id.get(recording_id)
        if source_item is None:
            raise SourceDeviceMetadataError("Source metadata cohort is incomplete.")
        if operator_item is not None:
            if source_item.get("status") != "unavailable":
                raise SourceDeviceMetadataError("Operator evidence may only fill metadata absence.")
            selected = operator_item
            basis = "direct_operator_knowledge"
            evidence_sha256 = device._canonical_hash(operator_item)
        elif source_item.get("status") == "observed":
            selected = source_item
            basis = "source_embedded_manufacturer_hardware_model"
            evidence_sha256 = device._canonical_hash(source_item)
        else:
            raise SourceDeviceMetadataError("Augmented composite still has missing device evidence.")
        if selected.get("source_sha256") != case.get("source_sha256"):
            raise SourceDeviceMetadataError("Augmented composite source binding drifted.")
        evidence.append(
            {
                "position": int(case["position"]),
                "recording_id": recording_id,
                "source_sha256": case["source_sha256"],
                "device_id": selected["device_id"],
                "evidence_basis": basis,
                "evidence_sha256": evidence_sha256,
            }
        )
    condition_path = Path(str(campaign["condition_authority"]["manifest_path"]))
    if condition_path.resolve() != condition_manifest_path.expanduser().resolve():
        raise SourceDeviceMetadataError("Augmented composite condition path drifted.")
    condition = _bound_private_object(
        condition_path,
        condition_path.parents[2],
        str(campaign["condition_authority"]["manifest_sha256"]),
    )
    observed = sorted({item["device_id"] for item in evidence})
    device_field = {
        "observed_values": observed,
        "observed_value_count": len(observed),
        "missing_recordings": 0,
        "status": "pass" if len(observed) >= 2 else "blocked",
    }
    fields = dict(condition["condition_coverage"]["fields"])
    fields["device"] = device_field
    blockers = [
        f"{field}_condition_coverage_below_policy"
        for field in device.conditions.CONDITION_FIELDS
        if fields[field]["status"] != "pass"
    ]
    coverage = {
        "fields": fields,
        "terminal_selection_eligible": not blockers,
        "blockers": blockers,
    }
    core = {
        "schema_version": AUGMENTED_COMPOSITE_PLAN_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": campaign_replay["manifest_sha256"],
        "campaign_records_state_sha256": campaign_replay["records_state_sha256"],
        "source_metadata_authority_id": source_replay["authority_id"],
        "source_metadata_manifest_sha256": source_replay["manifest_sha256"],
        "operator_authority_id": operator_replay["authority_id"],
        "operator_manifest_sha256": operator_replay["manifest_sha256"],
        "condition_manifest_sha256": campaign["condition_authority"]["manifest_sha256"],
        "condition_content_sha256": campaign["condition_authority"]["content_sha256"],
        "recordings": 7,
        "authoritative_device_evidence_count": 7,
        "direct_operator_observed_count": 2,
        "source_metadata_observed_count": 5,
        "evidence": evidence,
        "condition_coverage": coverage,
        "overlay_policy": {
            "only_device_may_change": True,
            "operator_fills_only_source_metadata_absence": True,
            "minimum_distinct_devices": 2,
            "encoding_profile_is_not_device_evidence": True,
        },
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_reveal_evaluation": False,
        "will_perform_external_write": False,
    }
    digest = device._canonical_hash(core)
    return {
        **core,
        "composite_id": f"augmented-composite-{digest[:24]}",
        "content_sha256": digest,
    }


def _augmented_receipt(
    preview: Mapping[str, Any], manifest_path: Path, manifest_sha256: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": AUGMENTED_COMPOSITE_RECEIPT_SCHEMA,
        "composite_id": preview["composite_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256 or sha256_file(manifest_path),
        "condition_coverage": preview["condition_coverage"],
        "authoritative_device_evidence_count": 7,
        "direct_operator_observed_count": 2,
        "source_metadata_observed_count": 5,
        "mode": "0600",
        "contains_device_labels": False,
        "contains_operator_identifier": False,
        "will_perform_external_write": False,
    }


def apply_augmented_composite(
    campaign_id: str,
    *,
    expected_content_sha256: str,
    source_metadata_manifest_path: Path,
    source_paths: Mapping[str, Path],
    operator_manifest_path: Path,
    operator_facts: Mapping[str, str],
    attested_by: str,
    operator_expected_device_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze a passing seven-record augmented composite authority."""
    kwargs = {
        "source_metadata_manifest_path": source_metadata_manifest_path,
        "source_paths": source_paths,
        "operator_manifest_path": operator_manifest_path,
        "operator_facts": operator_facts,
        "attested_by": attested_by,
        "operator_expected_device_id": operator_expected_device_id,
        "corpus_manifest_path": corpus_manifest_path,
        "condition_manifest_path": condition_manifest_path,
        "runtime_root": runtime_root,
    }
    preview = preview_augmented_composite(campaign_id, **kwargs)
    if preview["content_sha256"] != expected_content_sha256:
        raise SourceDeviceMetadataError("Reviewed augmented composite hash is stale.")
    if preview["condition_coverage"]["terminal_selection_eligible"] is not True:
        raise SourceDeviceMetadataError("Augmented composite remains blocked.")
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    paths = _augmented_paths(root, campaign_id, preview["composite_id"])
    existing = sorted(paths["base"].glob("*/manifest.json"))
    if existing and existing != [paths["manifest"]]:
        raise SourceDeviceMetadataError("A different augmented composite already exists.")
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_augmented_composite(paths["manifest"], campaign_id=campaign_id, **kwargs)
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise SourceDeviceMetadataError("Partial augmented composite exists.")
    ensure_private_tree(paths["root"], paths["composite"])
    manifest = {
        **preview,
        "schema_version": AUGMENTED_COMPOSITE_MANIFEST_SCHEMA,
        "status": "complete",
        "applied_at": device._utc_now(),
    }
    write_immutable_private_json(paths["manifest"], manifest, volatile_fields=("applied_at",))
    receipt = _augmented_receipt(preview, paths["manifest"])
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent": False,
    }


def replay_augmented_composite(
    manifest_path: Path,
    *,
    campaign_id: str,
    source_metadata_manifest_path: Path,
    source_paths: Mapping[str, Path],
    operator_manifest_path: Path,
    operator_facts: Mapping[str, str],
    attested_by: str,
    operator_expected_device_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay the passing augmented composite and sanitized receipt."""
    preview = preview_augmented_composite(
        campaign_id,
        source_metadata_manifest_path=source_metadata_manifest_path,
        source_paths=source_paths,
        operator_manifest_path=operator_manifest_path,
        operator_facts=operator_facts,
        attested_by=attested_by,
        operator_expected_device_id=operator_expected_device_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    root = runtime_root or device.DEFAULT_RUNTIME_ROOT
    selected = manifest_path.expanduser().resolve(strict=True)
    private_root = device._paths(root, campaign_id)["root"]
    manifest, selected_sha256 = _private_snapshot(selected, private_root)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    if comparable != {
        **preview,
        "schema_version": AUGMENTED_COMPOSITE_MANIFEST_SCHEMA,
        "status": "complete",
    }:
        raise SourceDeviceMetadataError("Augmented composite full-body replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    receipt, _ = _private_snapshot(receipt_path, private_root)
    expected_receipt = _augmented_receipt(preview, selected, selected_sha256)
    if receipt != expected_receipt:
        raise SourceDeviceMetadataError("Augmented composite receipt replay mismatch.")
    return {
        "schema_version": AUGMENTED_COMPOSITE_REPLAY_SCHEMA,
        "composite_id": preview["composite_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "condition_coverage": preview["condition_coverage"],
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }
