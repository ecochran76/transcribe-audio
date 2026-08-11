"""Replay six exact Plan 0066 outputs under the repaired Plan 0068 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_plan0066_a2 as plan0066_a2
import speaker_identity_plan0068_a0 as a0
import speaker_identity_preprocess
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


CASE_SCHEMA = "transcribe-audio.plan0068-a2-case.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0068-a2-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0068-a2-receipt.v1"
DEFAULT_RUNTIME_ROOT = a0.DEFAULT_RUNTIME_ROOT
EFFECT_COUNTS = dict(a0.EFFECT_COUNTS)


class Plan0068A2Error(ValueError):
    """Raised when exact retained-output replay or measurement drifts."""


def _hash(value: Any) -> str:
    return a0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        a0._validate_content(value, label)
    except a0.Plan0068A0Error as exc:
        raise Plan0068A2Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0068A2Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def repair_packet(
    packet: Mapping[str, Any], calendar_evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Add only the frozen explicit calendar catalog to a retained packet."""

    if "calendar_evidence" in packet:
        raise Plan0068A2Error("Retained packet already contains calendar_evidence.")
    if not calendar_evidence or any(
        not isinstance(item, Mapping)
        or not str(item.get("evidence_id") or "").startswith("calendar-")
        or item.get("identity_use") != "candidate_only"
        for item in calendar_evidence
    ):
        raise Plan0068A2Error("Frozen calendar evidence catalog is invalid.")
    repaired = copy.deepcopy(dict(packet))
    repaired["calendar_evidence"] = copy.deepcopy(calendar_evidence)
    changed = {
        key
        for key in set(packet) | set(repaired)
        if packet.get(key) != repaired.get(key)
    }
    if changed != {"calendar_evidence"}:
        raise Plan0068A2Error(f"Repaired packet changed unexpected fields: {sorted(changed)}.")
    return repaired


def _source_authority() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0068A2Error("A2 requires clean, upstream-even source authority.")
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    return {
        "commit": head,
        "upstream_commit": upstream,
        "speaker_identity_preprocess_sha256": hashlib.sha256(
            (root / "speaker_identity_preprocess.py").read_bytes()
        ).hexdigest(),
        "replay_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _case_from_retained(
    case_authority: Mapping[str, Any],
) -> dict[str, Any]:
    document_id = str(case_authority["document_id"])
    prepared_path = Path(
        str(case_authority["legacy_inputs"]["prepared_case"]["path"])
    )
    status_path = Path(
        str(case_authority["legacy_inputs"]["status_artifact"]["path"])
    )
    prepared = read_private_object(prepared_path)
    status = read_private_object(status_path)
    if sha256_file(prepared_path) != case_authority["legacy_inputs"]["prepared_case"]["file_sha256"]:
        raise Plan0068A2Error(f"Prepared case drifted: {document_id}.")
    if sha256_file(status_path) != case_authority["legacy_inputs"]["status_artifact"]["file_sha256"]:
        raise Plan0068A2Error(f"Retained status drifted: {document_id}.")
    output_text = str(status.get("output_text") or "")
    output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    if output_sha != case_authority["output_text_sha256"]:
        raise Plan0068A2Error(f"Retained output text drifted: {document_id}.")
    repaired = repair_packet(
        prepared["packet"],
        copy.deepcopy(case_authority["calendar_evidence"]),
    )
    if _hash(repaired["calendar_evidence"]) != case_authority["calendar_evidence_sha256"]:
        raise Plan0068A2Error(f"Calendar catalog drifted: {document_id}.")
    base = {
        "schema_version": CASE_SCHEMA,
        "document_id": document_id,
        "run_id": str(case_authority["run_id"]),
        "original_recording_filename": str(
            case_authority["original_recording_filename"]
        ),
        "retained_packet_sha256": str(prepared["packet_sha256"]),
        "repaired_packet_sha256": _hash(repaired),
        "calendar_evidence_sha256": case_authority["calendar_evidence_sha256"],
        "packet_delta_keys": ["calendar_evidence"],
        "retained_output_text_sha256": output_sha,
        "codex_thread_id": str(case_authority["codex_thread_id"]),
        "codex_turn_id": str(case_authority["codex_turn_id"]),
        "retained_output_changed": False,
        "primary_model_turn_count": 0,
        "fallback_model_turn_count": 0,
        "retry_count": 0,
        "reference_repair_count": 0,
        "fresh_retrieval_count": 0,
        "effect_counts": dict(EFFECT_COUNTS),
    }
    try:
        readout = app_intelligence_ledger.extract_json_object(output_text)
        validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
            repaired, readout
        )
        return _content(
            {
                **base,
                "status": "model_readout_validated",
                "validated_readout": validated["readout"],
            }
        )
    except Exception as exc:
        return _content(
            {
                **base,
                "status": "validation_failed",
                "reason": str(exc),
            }
        )


def execute_a2(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    run_root = root / "a2"
    receipt_path = run_root / "receipt.json"
    if receipt_path.exists():
        return replay_a2(runtime_root=root)
    a0_receipt = a0.replay_activation(runtime_root=root)
    a0_manifest = read_private_object(Path(a0_receipt["manifest_path"]))
    source_authority = _source_authority()
    case_root = run_root / "cases"
    ensure_private_tree(root, run_root, case_root)
    cases: list[dict[str, Any]] = []
    for case_authority in a0_manifest["cases"]:
        case = _case_from_retained(case_authority)
        case_path = case_root / f"{case['document_id']}.json"
        write_immutable_private_json(case_path, case)
        cases.append(case)
    cases.sort(key=lambda item: str(item["document_id"]))
    gold_binding = a0_manifest["artifact_bindings"]["human_gold"]
    gold_path = Path(str(gold_binding["path"]))
    if sha256_file(gold_path) != gold_binding["file_sha256"]:
        raise Plan0068A2Error("Frozen human gold drifted.")
    gold = read_private_object(gold_path)
    measurement = plan0066_a2.measure_cases(cases, gold)
    manifest = _content(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": measurement["status"],
            "a0_activation_content_sha256": a0_manifest["content_sha256"],
            "source_authority": source_authority,
            "selected_document_ids": [case["document_id"] for case in cases],
            "original_recording_filenames": [
                case["original_recording_filename"] for case in cases
            ],
            "case_content_sha256s": [case["content_sha256"] for case in cases],
            "measurement": measurement,
            "execution_counts": {
                "retained_output_replays": len(cases),
                "primary_model_turns": 0,
                "fallback_model_turns": 0,
                "retries": 0,
                "reference_repairs": 0,
                "fresh_retrievals": 0,
                "fresh_evaluations": 0,
            },
            "retained_output_change_count": 0,
            "source_store_index_change_count": 0,
            "will_apply_assignments": False,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    manifest_path = run_root / "private-manifest.json"
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": manifest["status"],
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "measurement": measurement,
            "execution_counts": manifest["execution_counts"],
            "original_recording_filename_count": len(
                manifest["original_recording_filenames"]
            ),
            "retained_output_change_count": 0,
            "source_store_index_change_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


def replay_a2(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    a0.replay_activation(runtime_root=root)
    manifest_path = root / "a2/private-manifest.json"
    receipt_path = root / "a2/receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    _validate_content(manifest, "Plan 0068 A2 manifest")
    _validate_content(receipt, "Plan 0068 A2 receipt")
    if receipt.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise Plan0068A2Error("A2 receipt lost its content binding.")
    if receipt.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise Plan0068A2Error("A2 receipt lost its file binding.")
    for document_id, content_sha in zip(
        manifest["selected_document_ids"], manifest["case_content_sha256s"]
    ):
        case = read_private_object(root / "a2/cases" / f"{document_id}.json")
        _validate_content(case, f"Plan 0068 A2 case {document_id}")
        if case["content_sha256"] != content_sha:
            raise Plan0068A2Error(f"A2 case binding drifted: {document_id}.")
    if manifest.get("execution_counts", {}).get("primary_model_turns") != 0:
        raise Plan0068A2Error("A2 model-turn budget drifted.")
    if manifest.get("effect_counts") != EFFECT_COUNTS:
        raise Plan0068A2Error("A2 effect budget drifted.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


if __name__ == "__main__":
    print(json.dumps(execute_a2(), indent=2, sort_keys=True))
