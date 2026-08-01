"""Append-only physical capture-device provenance for Plan 0037."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

import acoustic_audio_derivatives as audio_derivatives
import acoustic_speech_preparation as speech_preparation
import acoustic_successor_conditions as conditions
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PLAN_SCHEMA = "transcribe-audio.acoustic-device-provenance-plan.v1"
CAMPAIGN_SCHEMA = "transcribe-audio.acoustic-device-provenance-campaign.v1"
CAMPAIGN_RECEIPT_SCHEMA = (
    "transcribe-audio.acoustic-device-provenance-campaign-receipt.v1"
)
OPEN_SCHEMA = "transcribe-audio.acoustic-device-provenance-open.v1"
RECORD_SCHEMA = "transcribe-audio.acoustic-device-provenance-record.v1"
REPLAY_SCHEMA = "transcribe-audio.acoustic-device-provenance-replay.v1"
COMPOSITE_SCHEMA = "transcribe-audio.acoustic-composite-condition-plan.v1"
COMPOSITE_MANIFEST_SCHEMA = (
    "transcribe-audio.acoustic-composite-condition-manifest.v1"
)
COMPOSITE_RECEIPT_SCHEMA = (
    "transcribe-audio.acoustic-composite-condition-receipt.v1"
)
EXPECTED_RECORDINGS = 7
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/device-provenance"
)
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class DeviceProvenanceError(ValueError):
    """Raised when physical-device provenance cannot be trusted."""


def _canonical_hash(value: Any) -> str:
    return conditions._canonical_hash(value)


def _utc_now() -> str:
    return conditions._utc_now()


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise DeviceProvenanceError("Repository authority is unavailable.")
    return {
        "commit": commit.stdout.strip(),
        "clean": not bool(status.stdout.strip()),
        "module_sha256": sha256_file(Path(__file__).resolve()),
    }


def _validate_repository_authority(authority: Mapping[str, Any]) -> None:
    current = _repository_authority()
    if (
        dict(authority) != current
        or current["clean"] is not True
        or not COMMIT_RE.fullmatch(str(current["commit"]))
    ):
        raise DeviceProvenanceError("Repository authority is stale or dirty.")


def _validate_closed_commit(commit: str) -> None:
    if not COMMIT_RE.fullmatch(commit):
        raise DeviceProvenanceError("Closed condition commit is invalid.")
    root = Path(__file__).resolve().parent
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DeviceProvenanceError("Closed condition commit is not an ancestor.")


def _load_closed_condition(
    manifest_path: Path,
    *,
    corpus: Mapping[str, Any],
    corpus_manifest_path: Path,
    corpus_manifest_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    selected = manifest_path.expanduser().resolve(strict=True)
    root = selected.parents[2]
    require_private_file(selected, root)
    manifest = read_private_object(selected)
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "applied_at"}
    }
    receipt_path = selected.parent / "apply-receipt.json"
    require_private_file(receipt_path, root)
    receipt = read_private_object(receipt_path)
    corpus_authority = manifest.get("corpus") or {}
    units = manifest.get("units") or []
    by_recording = {
        str(record.get("recording_id") or ""): record
        for record in corpus.get("recordings") or []
        if isinstance(record, Mapping)
    }
    if (
        manifest.get("schema_version") != conditions.MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("content_sha256") != _canonical_hash(core)
        or len(units) != EXPECTED_RECORDINGS
        or (manifest.get("denominators") or {}).get("p1_successes")
        != EXPECTED_RECORDINGS
        or (manifest.get("denominators") or {}).get("p2_method_successes") != 35
        or corpus_authority.get("corpus_id") != corpus.get("corpus_id")
        or corpus_authority.get("content_sha256") != corpus.get("content_sha256")
        or corpus_authority.get("manifest_sha256") != corpus_manifest_sha256
        or Path(str(corpus_authority.get("manifest_path") or "")).resolve()
        != corpus_manifest_path.resolve()
    ):
        raise DeviceProvenanceError("Closed condition manifest is invalid.")
    condition_coverage = manifest.get("condition_coverage") or {}
    fields = condition_coverage.get("fields") or {}
    if (
        condition_coverage.get("terminal_selection_eligible") is not False
        or condition_coverage.get("blockers")
        != ["device_condition_coverage_below_policy"]
        or (fields.get("device") or {}).get("observed_value_count") != 0
        or (fields.get("device") or {}).get("missing_recordings")
        != EXPECTED_RECORDINGS
        or any(
            (fields.get(field) or {}).get("status") != "pass"
            for field in conditions.CONDITION_FIELDS
            if field != "device"
        )
    ):
        raise DeviceProvenanceError("Closed condition blocker is not exact.")
    seen: set[str] = set()
    for unit in units:
        recording_id = str(unit.get("recording_id") or "")
        record = by_recording.get(recording_id)
        if (
            record is None
            or recording_id in seen
            or unit.get("conversation_id") != record.get("conversation_id")
            or unit.get("split") != record.get("split")
            or unit.get("source_sha256")
            != (record.get("source_blob") or {}).get("sha256")
        ):
            raise DeviceProvenanceError("Closed condition unit binding is invalid.")
        seen.add(recording_id)
    expected_receipt = {
        "schema_version": conditions.RECEIPT_SCHEMA,
        "plan_id": manifest["plan_id"],
        "manifest_path": str(selected),
        "manifest_sha256": sha256_file(selected),
        "content_sha256": manifest["content_sha256"],
        "denominators": manifest["denominators"],
        "condition_coverage": manifest["condition_coverage"],
        "mode": "0600",
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise DeviceProvenanceError("Closed condition receipt is invalid.")
    modules = manifest.get("module_authority") or {}
    repository = manifest.get("repository_authority") or {}
    if (
        modules.get("condition_sha256")
        != sha256_file(Path(conditions.__file__).resolve())
        or modules.get("p1_sha256")
        != sha256_file(Path(audio_derivatives.__file__).resolve())
        or modules.get("p2_sha256")
        != sha256_file(Path(speech_preparation.__file__).resolve())
        or repository.get("module_sha256") != modules.get("condition_sha256")
    ):
        raise DeviceProvenanceError("Closed condition module authority drifted.")
    _validate_closed_commit(str(repository.get("commit") or ""))
    return manifest, sha256_file(selected), receipt, sha256_file(receipt_path)


def _private_case(
    record: Mapping[str, Any], position: int
) -> dict[str, Any]:
    source = record.get("source_blob") or {}
    source_path = Path(str(source.get("stored_path") or "")).resolve(strict=True)
    require_private_file(source_path, source_path.parent)
    if (
        sha256_file(source_path) != source.get("sha256")
        or source_path.stat().st_size != source.get("bytes")
    ):
        raise DeviceProvenanceError("Device case source binding drifted.")
    lineage = record.get("transcript_lineage") or {}
    artifact = Path(str(lineage.get("current_artifact_path") or "")).resolve(
        strict=True
    )
    if (
        sha256_file(artifact) != lineage.get("current_artifact_sha256")
        or lineage.get("current_artifact_sha256")
        != lineage.get("reviewed_artifact_sha256")
    ):
        raise DeviceProvenanceError("Device case transcript lineage drifted.")
    transcript = json.loads(artifact.read_text(encoding="utf-8"))
    original_path = Path(str(transcript.get("source_media_path") or ""))
    return {
        "position": position,
        "recording_id": str(record.get("recording_id") or ""),
        "conversation_id": str(record.get("conversation_id") or ""),
        "document_id": str(record.get("document_id") or ""),
        "split": str(record.get("split") or ""),
        "chronological_rank": int(record.get("chronological_rank") or 0),
        "source_sha256": str(source.get("sha256") or ""),
        "source_bytes": int(source.get("bytes") or 0),
        "transcript_sha256": str(lineage.get("current_artifact_sha256") or ""),
        "private_operator_context": {
            "recording_start": transcript.get("recording_start"),
            "recording_end": transcript.get("recording_end"),
            "transcript_title": str(transcript.get("transcript_title") or ""),
            "original_source_parent": str(original_path.parent),
            "original_source_name": original_path.name,
            "original_source_exists": original_path.is_file(),
        },
    }


def preview_device_campaign(
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
) -> dict[str, Any]:
    """Build the exact no-write seven-case device provenance plan."""
    corpus, corpus_manifest_sha256 = conditions._load_corpus(corpus_manifest_path)
    condition, condition_manifest_sha256, receipt, receipt_sha256 = (
        _load_closed_condition(
            condition_manifest_path,
            corpus=corpus,
            corpus_manifest_path=corpus_manifest_path.expanduser().resolve(),
            corpus_manifest_sha256=corpus_manifest_sha256,
        )
    )
    cases = [
        _private_case(record, position)
        for position, record in enumerate(corpus["recordings"], 1)
    ]
    expected_splits = {"development": 3, "calibration": 2, "evaluation": 2}
    if (
        len(cases) != EXPECTED_RECORDINGS
        or len({case["recording_id"] for case in cases}) != EXPECTED_RECORDINGS
        or len({case["conversation_id"] for case in cases}) != EXPECTED_RECORDINGS
        or len({case["source_sha256"] for case in cases}) != EXPECTED_RECORDINGS
        or dict(Counter(case["split"] for case in cases)) != expected_splits
        or any(not case["document_id"] for case in cases)
    ):
        raise DeviceProvenanceError("Device campaign cases are not exact.")
    core = {
        "schema_version": PLAN_SCHEMA,
        "corpus_authority": {
            "corpus_id": corpus["corpus_id"],
            "content_sha256": corpus["content_sha256"],
            "manifest_path": str(corpus_manifest_path.expanduser().resolve()),
            "manifest_sha256": corpus_manifest_sha256,
        },
        "condition_authority": {
            "plan_id": condition["plan_id"],
            "content_sha256": condition["content_sha256"],
            "manifest_path": str(condition_manifest_path.expanduser().resolve()),
            "manifest_sha256": condition_manifest_sha256,
            "receipt_sha256": receipt_sha256,
            "receipt_schema_version": receipt["schema_version"],
        },
        "repository_authority": _repository_authority(),
        "cases": cases,
        "denominators": {
            "recordings": EXPECTED_RECORDINGS,
            "required_direct_attestations": EXPECTED_RECORDINGS,
            "minimum_distinct_devices": 2,
            "split_recordings": expected_splits,
        },
        "policy": {
            "accepted_observed_basis": "direct_operator_knowledge",
            "unavailable_basis": "operator_unknown",
            "device_label_equality": "casefold_whitespace_normalized",
            "inference_is_prohibited": True,
            "encoding_profile_is_not_device_evidence": True,
            "one_case_at_a_time": True,
            "append_only_corrections": True,
        },
        "will_open_case": False,
        "will_assert_device_fact": False,
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_use_transcript_text": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {
        **core,
        "campaign_id": f"device-provenance-{digest[:24]}",
        "content_sha256": digest,
    }


def _paths(root: Path, campaign_id: str) -> dict[str, Path]:
    selected = root.expanduser().absolute()
    campaign = selected / "campaigns" / campaign_id
    return {
        "root": selected,
        "campaign": campaign,
        "manifest": campaign / "manifest.json",
        "receipt": campaign / "apply-receipt.json",
        "opens": campaign / "opens",
        "records": campaign / "records",
        "composite": campaign / "composite",
    }


def apply_device_campaign(
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    *,
    expected_content_sha256: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze the reviewed campaign without opening a case or asserting a fact."""
    preview = preview_device_campaign(corpus_manifest_path, condition_manifest_path)
    if preview["content_sha256"] != expected_content_sha256:
        raise DeviceProvenanceError("Reviewed device campaign hash is stale.")
    _validate_repository_authority(preview["repository_authority"])
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, preview["campaign_id"])
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_device_campaign(
            preview["campaign_id"],
            corpus_manifest_path=corpus_manifest_path,
            condition_manifest_path=condition_manifest_path,
            runtime_root=runtime_root,
        )
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise DeviceProvenanceError("Partial device campaign finalization exists.")
    ensure_private_tree(paths["root"], paths["campaign"])
    manifest = {
        **preview,
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "open",
        "applied_at": _utc_now(),
    }
    write_immutable_private_json(
        paths["manifest"], manifest, volatile_fields=("applied_at",)
    )
    receipt = {
        "schema_version": CAMPAIGN_RECEIPT_SCHEMA,
        "campaign_id": preview["campaign_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recordings": EXPECTED_RECORDINGS,
        "mode": "0600",
        "contains_private_operator_context": False,
        "contains_device_labels": False,
        "will_perform_external_write": False,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "receipt_path": str(paths["receipt"]), "idempotent": False}


def _load_campaign(paths: Mapping[str, Path]) -> dict[str, Any]:
    require_private_file(paths["manifest"], paths["root"])
    return read_private_object(paths["manifest"])


def _record_files(paths: Mapping[str, Path]) -> list[Path]:
    if not paths["records"].is_dir():
        return []
    return sorted(paths["records"].glob("*.json"))


def _normalize_label(label: str) -> str:
    return " ".join(str(label).split()).casefold()


def _device_id(normalized_label: str) -> str:
    return "physical-device-" + _canonical_hash(
        {"kind": "physical_capture_device", "normalized_label": normalized_label}
    )[:24]


def _records(
    paths: Mapping[str, Path], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cases = manifest.get("cases") or []
    case_by_recording = {
        str(case.get("recording_id") or ""): case
        for case in cases
        if isinstance(case, Mapping)
    }
    records: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    initial_count = 0
    previous = ""
    expected_keys = {
        "schema_version",
        "campaign_id",
        "campaign_manifest_sha256",
        "action",
        "sequence",
        "case_position",
        "recording_id",
        "source_sha256",
        "status",
        "physical_device_label",
        "normalized_device_label",
        "device_id",
        "evidence_basis",
        "attested_by",
        "open_receipt_sha256",
        "supersedes_record_sha256",
        "previous_record_sha256",
        "recorded_at",
        "record_sha256",
    }
    manifest_sha256 = sha256_file(paths["manifest"])
    for sequence, path in enumerate(_record_files(paths), 1):
        require_private_file(path, paths["root"])
        record = read_private_object(path)
        core = dict(record)
        record_sha256 = str(core.pop("record_sha256", ""))
        recording_id = str(record.get("recording_id") or "")
        case = case_by_recording.get(recording_id)
        status = record.get("status")
        normalized = str(record.get("normalized_device_label") or "")
        if (
            set(record) != expected_keys
            or record.get("schema_version") != RECORD_SCHEMA
            or record.get("campaign_id") != manifest.get("campaign_id")
            or record.get("campaign_manifest_sha256") != manifest_sha256
            or int(record.get("sequence") or 0) != sequence
            or record_sha256 != _canonical_hash(core)
            or record.get("previous_record_sha256") != previous
            or case is None
            or record.get("source_sha256") != case.get("source_sha256")
            or status not in {"observed", "unavailable"}
            or not str(record.get("attested_by") or "").strip()
            or not SHA256_RE.fullmatch(str(record.get("open_receipt_sha256") or ""))
        ):
            raise DeviceProvenanceError("Device provenance record history is invalid.")
        if status == "observed":
            if (
                record.get("evidence_basis") != "direct_operator_knowledge"
                or not normalized
                or normalized
                != _normalize_label(str(record.get("physical_device_label") or ""))
                or record.get("device_id") != _device_id(normalized)
            ):
                raise DeviceProvenanceError("Observed device provenance is invalid.")
        elif (
            record.get("evidence_basis") != "operator_unknown"
            or record.get("physical_device_label")
            or normalized
            or record.get("device_id")
        ):
            raise DeviceProvenanceError("Unavailable device provenance is invalid.")
        action = record.get("action")
        if action == "attest":
            initial_count += 1
            if (
                initial_count > EXPECTED_RECORDINGS
                or int(record.get("case_position") or 0) != initial_count
                or case.get("position") != initial_count
                or recording_id in latest
                or record.get("supersedes_record_sha256")
            ):
                raise DeviceProvenanceError("Initial device attestation order is invalid.")
        elif action == "correct":
            prior = latest.get(recording_id)
            if (
                initial_count != EXPECTED_RECORDINGS
                or prior is None
                or int(record.get("case_position") or 0) != case.get("position")
                or record.get("supersedes_record_sha256")
                != prior.get("record_sha256")
            ):
                raise DeviceProvenanceError("Device correction authority is invalid.")
        else:
            raise DeviceProvenanceError("Device provenance action is invalid.")
        latest[recording_id] = record
        records.append(record)
        previous = record_sha256
    return records


def _latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["recording_id"])] = record
    return latest


def _records_state(records: list[dict[str, Any]]) -> str:
    return _canonical_hash([record["record_sha256"] for record in records])


def _open_receipts(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not paths["opens"].is_dir():
        return []
    files = sorted(paths["opens"].glob("*.json"))
    cases = manifest.get("cases") or []
    previous = ""
    receipts: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "campaign_id",
        "campaign_manifest_sha256",
        "position",
        "recording_id",
        "source_sha256",
        "records_state_sha256",
        "previous_open_receipt_sha256",
        "status",
        "will_read_private_operator_context",
        "will_infer_device",
        "will_perform_external_write",
        "opened_at",
        "open_receipt_sha256",
    }
    manifest_sha256 = sha256_file(paths["manifest"])
    for position, path in enumerate(files, 1):
        require_private_file(path, paths["root"])
        receipt = read_private_object(path)
        core = dict(receipt)
        receipt_sha = str(core.pop("open_receipt_sha256", ""))
        case = cases[position - 1] if position <= len(cases) else {}
        prior_records = [
            record
            for record in records
            if record.get("action") == "attest"
            and int(record.get("case_position") or 0) < position
        ]
        if (
            set(receipt) != expected_keys
            or receipt.get("schema_version") != OPEN_SCHEMA
            or receipt.get("campaign_id") != manifest.get("campaign_id")
            or receipt.get("campaign_manifest_sha256") != manifest_sha256
            or int(receipt.get("position") or 0) != position
            or receipt.get("recording_id") != case.get("recording_id")
            or receipt.get("source_sha256") != case.get("source_sha256")
            or receipt.get("records_state_sha256") != _records_state(prior_records)
            or receipt.get("previous_open_receipt_sha256") != previous
            or receipt.get("status") != "open"
            or receipt.get("will_read_private_operator_context") is not True
            or receipt.get("will_infer_device") is not False
            or receipt.get("will_perform_external_write") is not False
            or receipt_sha != _canonical_hash(core)
        ):
            raise DeviceProvenanceError("Device case cursor history is invalid.")
        receipts.append(receipt)
        previous = receipt_sha
    return receipts


def replay_device_campaign(
    campaign_id: str,
    *,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Read-only replay of campaign, cursor, and append-only provenance."""
    preview = preview_device_campaign(corpus_manifest_path, condition_manifest_path)
    if campaign_id != preview["campaign_id"]:
        raise DeviceProvenanceError("Device campaign identity is stale.")
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    manifest = _load_campaign(paths)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    expected = {**preview, "schema_version": CAMPAIGN_SCHEMA, "status": "open"}
    if comparable != expected:
        raise DeviceProvenanceError("Device campaign full-body replay mismatch.")
    require_private_file(paths["receipt"], paths["root"])
    receipt = read_private_object(paths["receipt"])
    expected_receipt = {
        "schema_version": CAMPAIGN_RECEIPT_SCHEMA,
        "campaign_id": campaign_id,
        "content_sha256": preview["content_sha256"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recordings": EXPECTED_RECORDINGS,
        "mode": "0600",
        "contains_private_operator_context": False,
        "contains_device_labels": False,
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise DeviceProvenanceError("Device campaign receipt replay mismatch.")
    records = _records(paths, manifest)
    opens = _open_receipts(paths, manifest, records)
    initial = [record for record in records if record["action"] == "attest"]
    if len(opens) not in {len(initial), min(len(initial) + 1, EXPECTED_RECORDINGS)}:
        raise DeviceProvenanceError("Device cursor and attestation progress diverged.")
    for record in records:
        position = int(record["case_position"])
        if (
            position > len(opens)
            or record["open_receipt_sha256"]
            != opens[position - 1]["open_receipt_sha256"]
        ):
            raise DeviceProvenanceError(
                "Device record is not bound to its authoritative open receipt."
            )
    return {
        "schema_version": REPLAY_SCHEMA,
        "campaign_id": campaign_id,
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "recordings": EXPECTED_RECORDINGS,
        "opened_cases": len(opens),
        "initial_records": len(initial),
        "corrections": len(records) - len(initial),
        "records_state_sha256": _records_state(records),
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }


def open_next_device_case(
    campaign_id: str,
    *,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Open or idempotently reopen exactly the next factual device case."""
    replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    manifest = _load_campaign(paths)
    records = _records(paths, manifest)
    opens = _open_receipts(paths, manifest, records)
    initial = [record for record in records if record["action"] == "attest"]
    if len(opens) == len(initial) + 1:
        receipt = opens[-1]
        case = manifest["cases"][int(receipt["position"]) - 1]
        return {
            "open_receipt": receipt,
            "packet": _operator_packet(case),
            "idempotent_reopen": True,
        }
    if len(initial) >= EXPECTED_RECORDINGS:
        raise DeviceProvenanceError("Device provenance campaign is complete.")
    position = len(initial) + 1
    case = manifest["cases"][position - 1]
    core = {
        "schema_version": OPEN_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": sha256_file(paths["manifest"]),
        "position": position,
        "recording_id": case["recording_id"],
        "source_sha256": case["source_sha256"],
        "records_state_sha256": _records_state(initial),
        "previous_open_receipt_sha256": (
            str(opens[-1]["open_receipt_sha256"]) if opens else ""
        ),
        "status": "open",
        "will_read_private_operator_context": True,
        "will_infer_device": False,
        "will_perform_external_write": False,
        "opened_at": _utc_now(),
    }
    receipt = {**core, "open_receipt_sha256": _canonical_hash(core)}
    ensure_private_tree(paths["root"], paths["opens"])
    path = paths["opens"] / f"{position:04d}.json"
    write_immutable_private_json(path, receipt, volatile_fields=("opened_at",))
    return {
        "open_receipt": {**receipt, "receipt_path": str(path)},
        "packet": _operator_packet(case),
        "idempotent_reopen": False,
    }


def _operator_packet(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "transcribe-audio.acoustic-device-operator-packet.v1",
        "position": case["position"],
        "recording_id": case["recording_id"],
        "chronological_rank": case["chronological_rank"],
        "context": dict(case["private_operator_context"]),
        "question": "Which physical device directly captured this recording?",
        "accepted_basis": "direct_operator_knowledge",
        "unknown_is_allowed": True,
        "do_not_infer_from_context": True,
        "private_runtime_artifact": True,
        "will_perform_external_write": False,
    }


def _write_record(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    action: str,
    case: Mapping[str, Any],
    open_receipt_sha256: str,
    physical_device_label: str,
    attested_by: str,
    status: str,
    supersedes_record_sha256: str = "",
) -> dict[str, Any]:
    normalized = _normalize_label(physical_device_label) if status == "observed" else ""
    if (
        status == "observed"
        and (not normalized or len(normalized) > 200)
    ):
        raise DeviceProvenanceError("A concise physical device label is required.")
    if status not in {"observed", "unavailable"}:
        raise DeviceProvenanceError("Device provenance status is invalid.")
    if not str(attested_by).strip():
        raise DeviceProvenanceError("Device attestation requires an operator.")
    sequence = len(records) + 1
    core = {
        "schema_version": RECORD_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": sha256_file(paths["manifest"]),
        "action": action,
        "sequence": sequence,
        "case_position": case["position"],
        "recording_id": case["recording_id"],
        "source_sha256": case["source_sha256"],
        "status": status,
        "physical_device_label": physical_device_label.strip() if status == "observed" else "",
        "normalized_device_label": normalized,
        "device_id": _device_id(normalized) if status == "observed" else "",
        "evidence_basis": (
            "direct_operator_knowledge" if status == "observed" else "operator_unknown"
        ),
        "attested_by": str(attested_by).strip(),
        "open_receipt_sha256": open_receipt_sha256,
        "supersedes_record_sha256": supersedes_record_sha256,
        "previous_record_sha256": (
            str(records[-1]["record_sha256"]) if records else ""
        ),
        "recorded_at": _utc_now(),
    }
    record = {**core, "record_sha256": _canonical_hash(core)}
    ensure_private_tree(paths["root"], paths["records"])
    path = paths["records"] / f"{sequence:04d}.json"
    write_immutable_private_json(path, record, volatile_fields=("recorded_at",))
    return {
        "schema_version": RECORD_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "recording_id": case["recording_id"],
        "case_position": case["position"],
        "status": status,
        "device_id": record["device_id"],
        "evidence_basis": record["evidence_basis"],
        "record_sha256": record["record_sha256"],
        "record_path": str(path),
        "contains_device_label": False,
        "will_perform_external_write": False,
    }


def record_device_provenance(
    campaign_id: str,
    *,
    physical_device_label: str = "",
    attested_by: str,
    status: str = "observed",
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Append one initial direct attestation or explicit unknown result."""
    replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    manifest = _load_campaign(paths)
    records = _records(paths, manifest)
    opens = _open_receipts(paths, manifest, records)
    initial = [record for record in records if record["action"] == "attest"]
    if len(opens) != len(initial) + 1:
        raise DeviceProvenanceError("No outstanding device case is open.")
    receipt = opens[-1]
    case = manifest["cases"][len(initial)]
    if receipt["recording_id"] != case["recording_id"]:
        raise DeviceProvenanceError("Outstanding device case binding drifted.")
    return _write_record(
        paths,
        manifest,
        records,
        action="attest",
        case=case,
        open_receipt_sha256=receipt["open_receipt_sha256"],
        physical_device_label=physical_device_label,
        attested_by=attested_by,
        status=status,
    )


def correct_device_provenance(
    campaign_id: str,
    recording_id: str,
    *,
    physical_device_label: str = "",
    attested_by: str,
    status: str = "observed",
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Append a correction after the exact initial seven-case pass completes."""
    replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    manifest = _load_campaign(paths)
    records = _records(paths, manifest)
    initial = [record for record in records if record["action"] == "attest"]
    if len(initial) != EXPECTED_RECORDINGS:
        raise DeviceProvenanceError("Corrections require all seven initial cases.")
    cases = {
        str(case["recording_id"]): case for case in manifest.get("cases") or []
    }
    case = cases.get(recording_id)
    latest = _latest_records(records).get(recording_id)
    if case is None or latest is None:
        raise DeviceProvenanceError("Correction recording is not authoritative.")
    opens = _open_receipts(paths, manifest, records)
    open_receipt = opens[int(case["position"]) - 1]
    return _write_record(
        paths,
        manifest,
        records,
        action="correct",
        case=case,
        open_receipt_sha256=open_receipt["open_receipt_sha256"],
        physical_device_label=physical_device_label,
        attested_by=attested_by,
        status=status,
        supersedes_record_sha256=latest["record_sha256"],
    )


def preview_composite_condition_authority(
    campaign_id: str,
    *,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Overlay only replayed direct device facts onto closed measured conditions."""
    replay = replay_device_campaign(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    manifest = _load_campaign(paths)
    records = _records(paths, manifest)
    latest = _latest_records(records)
    condition = read_private_object(Path(manifest["condition_authority"]["manifest_path"]))
    observed = sorted(
        {
            str(record["device_id"])
            for record in latest.values()
            if record["status"] == "observed"
        }
    )
    missing = EXPECTED_RECORDINGS - sum(
        1 for record in latest.values() if record["status"] == "observed"
    )
    device = {
        "observed_values": observed,
        "observed_value_count": len(observed),
        "missing_recordings": missing,
        "status": "pass" if len(observed) >= 2 and missing == 0 else "blocked",
    }
    fields = dict(condition["condition_coverage"]["fields"])
    fields["device"] = device
    blockers = [
        f"{field}_condition_coverage_below_policy"
        for field in conditions.CONDITION_FIELDS
        if fields[field]["status"] != "pass"
    ]
    coverage = {
        "fields": fields,
        "terminal_selection_eligible": not blockers,
        "blockers": blockers,
    }
    core = {
        "schema_version": COMPOSITE_SCHEMA,
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": replay["manifest_sha256"],
        "campaign_records_state_sha256": replay["records_state_sha256"],
        "condition_manifest_path": manifest["condition_authority"]["manifest_path"],
        "condition_manifest_sha256": manifest["condition_authority"]["manifest_sha256"],
        "condition_content_sha256": manifest["condition_authority"]["content_sha256"],
        "recordings": EXPECTED_RECORDINGS,
        "latest_attestation_count": len(latest),
        "direct_observed_attestation_count": EXPECTED_RECORDINGS - missing,
        "device_record_sha256": {
            recording_id: record["record_sha256"]
            for recording_id, record in sorted(latest.items())
        },
        "condition_coverage": coverage,
        "overlay_policy": {
            "only_device_may_change": True,
            "requires_seven_direct_observed": True,
            "minimum_distinct_devices": 2,
            "encoding_profile_is_not_device_evidence": True,
        },
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_reveal_evaluation": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {
        **core,
        "composite_id": f"composite-conditions-{digest[:24]}",
        "content_sha256": digest,
    }


def apply_composite_condition_authority(
    campaign_id: str,
    *,
    expected_content_sha256: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze a passing composite authority; blocked previews never write."""
    preview = preview_composite_condition_authority(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    if preview["content_sha256"] != expected_content_sha256:
        raise DeviceProvenanceError("Reviewed composite condition hash is stale.")
    if preview["condition_coverage"]["terminal_selection_eligible"] is not True:
        raise DeviceProvenanceError("Composite condition authority remains blocked.")
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    composite_dir = paths["composite"] / preview["composite_id"]
    manifest_path = composite_dir / "manifest.json"
    receipt_path = composite_dir / "apply-receipt.json"
    if manifest_path.exists() and receipt_path.exists():
        return replay_composite_condition_authority(
            manifest_path,
            campaign_id=campaign_id,
            corpus_manifest_path=corpus_manifest_path,
            condition_manifest_path=condition_manifest_path,
            runtime_root=runtime_root,
        )
    if manifest_path.exists() or receipt_path.exists():
        raise DeviceProvenanceError("Partial composite finalization exists.")
    ensure_private_tree(paths["root"], composite_dir)
    manifest = {
        **preview,
        "schema_version": COMPOSITE_MANIFEST_SCHEMA,
        "status": "complete",
        "applied_at": _utc_now(),
    }
    write_immutable_private_json(
        manifest_path, manifest, volatile_fields=("applied_at",)
    )
    receipt = {
        "schema_version": COMPOSITE_RECEIPT_SCHEMA,
        "composite_id": preview["composite_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "condition_coverage": preview["condition_coverage"],
        "mode": "0600",
        "contains_device_labels": False,
        "will_perform_external_write": False,
    }
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path), "idempotent": False}


def replay_composite_condition_authority(
    manifest_path: Path,
    *,
    campaign_id: str,
    corpus_manifest_path: Path,
    condition_manifest_path: Path,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Read-only exact replay of a passing composite condition authority."""
    preview = preview_composite_condition_authority(
        campaign_id,
        corpus_manifest_path=corpus_manifest_path,
        condition_manifest_path=condition_manifest_path,
        runtime_root=runtime_root,
    )
    selected = manifest_path.expanduser().resolve(strict=True)
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, campaign_id)
    require_private_file(selected, paths["root"])
    manifest = read_private_object(selected)
    comparable = dict(manifest)
    comparable.pop("applied_at", None)
    expected = {
        **preview,
        "schema_version": COMPOSITE_MANIFEST_SCHEMA,
        "status": "complete",
    }
    if comparable != expected:
        raise DeviceProvenanceError("Composite condition replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    require_private_file(receipt_path, paths["root"])
    receipt = read_private_object(receipt_path)
    expected_receipt = {
        "schema_version": COMPOSITE_RECEIPT_SCHEMA,
        "composite_id": preview["composite_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_path": str(selected),
        "manifest_sha256": sha256_file(selected),
        "condition_coverage": preview["condition_coverage"],
        "mode": "0600",
        "contains_device_labels": False,
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise DeviceProvenanceError("Composite condition receipt mismatch.")
    return {
        "schema_version": COMPOSITE_RECEIPT_SCHEMA,
        "composite_id": preview["composite_id"],
        "content_sha256": preview["content_sha256"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "condition_coverage": preview["condition_coverage"],
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }
