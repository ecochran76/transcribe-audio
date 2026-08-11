"""Normalize and replay six exact retained identity-evaluation outputs."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_plan0066_a2 as plan0066_a2
import speaker_identity_plan0068_a2 as plan0068_a2
import speaker_identity_plan0069_a0 as a0
import speaker_identity_preprocess
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


CASE_SCHEMA = "transcribe-audio.plan0069-a2-case.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0069-a2-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0069-a2-receipt.v1"
DEFAULT_RUNTIME_ROOT = a0.DEFAULT_RUNTIME_ROOT
EFFECT_COUNTS = dict(a0.EFFECT_COUNTS)


class Plan0069A2Error(ValueError):
    """Raised when grouped normalization or retained-output replay drifts."""


def _hash(value: Any) -> str:
    return a0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        a0._validate_content(value, label)
    except a0.Plan0069A0Error as exc:
        raise Plan0069A2Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0069A2Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def _source_authority() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0069A2Error("A2 requires clean, upstream-even source authority.")
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    return {
        "commit": head,
        "upstream_commit": upstream,
        "speaker_identity_preprocess_sha256": hashlib.sha256(
            (root / "speaker_identity_preprocess.py").read_bytes()
        ).hexdigest(),
        "replay_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def assert_normalization_matches_inventory(
    inventory: list[Mapping[str, Any]],
    normalization: Mapping[str, Any],
) -> None:
    """Require normalized paths and IDs to equal the frozen A0 inventory."""

    expected = [
        {
            "path": str(item["path"]),
            "utterance_ids": list(item["utterance_ids"]),
            "expanded_count": len(item["utterance_ids"]),
        }
        for item in inventory
    ]
    if normalization.get("changes") != expected:
        raise Plan0069A2Error("Normalization changes differ from A0 inventory.")
    if normalization.get("normalized_group_count") != len(expected):
        raise Plan0069A2Error("Normalized group count differs from A0 inventory.")
    if normalization.get("expanded_utterance_assignment_count") != sum(
        item["expanded_count"] for item in expected
    ):
        raise Plan0069A2Error("Expanded assignment count differs from A0 inventory.")


def _case_from_retained(
    authority: Mapping[str, Any],
    plan0068_authority: Mapping[str, Any],
) -> dict[str, Any]:
    document_id = str(authority["document_id"])
    if document_id != plan0068_authority.get("document_id"):
        raise Plan0069A2Error(f"Plan 0068 case authority mismatch: {document_id}.")
    prepared_path = Path(str(authority["prepared_case"]["path"]))
    status_path = Path(str(authority["status_artifact"]["path"]))
    prepared = read_private_object(prepared_path)
    status = read_private_object(status_path)
    if sha256_file(prepared_path) != authority["prepared_case"]["file_sha256"]:
        raise Plan0069A2Error(f"Prepared case drifted: {document_id}.")
    if sha256_file(status_path) != authority["status_artifact"]["file_sha256"]:
        raise Plan0069A2Error(f"Retained status drifted: {document_id}.")
    output_text = str(status.get("output_text") or "")
    output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    if output_sha != authority["retained_output_text_sha256"]:
        raise Plan0069A2Error(f"Retained output text drifted: {document_id}.")
    packet = plan0068_a2.repair_packet(
        prepared["packet"],
        copy.deepcopy(plan0068_authority["calendar_evidence"]),
    )
    readout = app_intelligence_ledger.extract_json_object(output_text)
    observed_inventory = a0.grouped_assignment_inventory(readout)
    if observed_inventory != authority["grouped_assignments"]:
        raise Plan0069A2Error(f"Grouped object inventory drifted: {document_id}.")
    normalization = speaker_identity_preprocess.normalize_grouped_utterance_assignments(
        packet, readout
    )
    assert_normalization_matches_inventory(
        list(authority["grouped_assignments"]), normalization
    )
    validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
        packet, normalization["readout"]
    )
    return _content(
        {
            "schema_version": CASE_SCHEMA,
            "status": "normalized_readout_validated",
            "document_id": document_id,
            "original_recording_filename": authority["original_recording_filename"],
            "retained_output_text_sha256": output_sha,
            "retained_output_changed": False,
            "normalized_readout_sha256": _hash(normalization["readout"]),
            "normalized_group_count": normalization["normalized_group_count"],
            "expanded_utterance_assignment_count": normalization[
                "expanded_utterance_assignment_count"
            ],
            "normalization_changes": normalization["changes"],
            "validated_readout": validated["readout"],
            "primary_model_turn_count": 0,
            "fallback_model_turn_count": 0,
            "retry_count": 0,
            "model_reference_repair_count": 0,
            "fresh_retrieval_count": 0,
            "fresh_evaluation_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
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
    plan0068_a0_path = Path(
        str(a0_manifest["artifact_bindings"]["plan0068_a0_manifest"]["path"])
    )
    plan0068_a0_manifest = read_private_object(plan0068_a0_path)
    plan0068_cases = {
        str(case["document_id"]): case for case in plan0068_a0_manifest["cases"]
    }
    source_authority = _source_authority()
    case_root = run_root / "cases"
    ensure_private_tree(root, case_root)
    cases = [
        _case_from_retained(authority, plan0068_cases[str(authority["document_id"])])
        for authority in a0_manifest["cases"]
    ]
    cases.sort(key=lambda item: str(item["document_id"]))
    gold_path = Path(
        str(plan0068_a0_manifest["artifact_bindings"]["human_gold"]["path"])
    )
    if sha256_file(gold_path) != plan0068_a0_manifest["artifact_bindings"]["human_gold"]["file_sha256"]:
        raise Plan0069A2Error("Frozen human gold drifted.")
    gold = read_private_object(gold_path)
    measurement_cases = [
        {
            **case,
            "status": "model_readout_validated",
        }
        for case in cases
    ]
    measurement = plan0066_a2.measure_cases(measurement_cases, gold)
    for case in cases:
        write_immutable_private_json(
            case_root / f"{case['document_id']}.json", case
        )
    normalized_groups = sum(case["normalized_group_count"] for case in cases)
    expanded_assignments = sum(
        case["expanded_utterance_assignment_count"] for case in cases
    )
    if normalized_groups != 10 or expanded_assignments != 28:
        raise Plan0069A2Error("A2 normalization totals drifted.")
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
            "normalized_group_count": normalized_groups,
            "expanded_utterance_assignment_count": expanded_assignments,
            "execution_counts": {
                "retained_output_replays": len(cases),
                "primary_model_turns": 0,
                "fallback_model_turns": 0,
                "retries": 0,
                "model_reference_repairs": 0,
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
            "normalized_group_count": normalized_groups,
            "expanded_utterance_assignment_count": expanded_assignments,
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
    _validate_content(manifest, "Plan 0069 A2 manifest")
    _validate_content(receipt, "Plan 0069 A2 receipt")
    if receipt.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise Plan0069A2Error("A2 receipt lost its content binding.")
    if receipt.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise Plan0069A2Error("A2 receipt lost its file binding.")
    for document_id, content_sha in zip(
        manifest["selected_document_ids"],
        manifest["case_content_sha256s"],
        strict=True,
    ):
        case = read_private_object(root / "a2/cases" / f"{document_id}.json")
        _validate_content(case, f"Plan 0069 A2 case {document_id}")
        if case["content_sha256"] != content_sha:
            raise Plan0069A2Error(f"A2 case binding drifted: {document_id}.")
    if (
        manifest.get("normalized_group_count") != 10
        or manifest.get("expanded_utterance_assignment_count") != 28
        or manifest.get("execution_counts", {}).get("primary_model_turns") != 0
        or manifest.get("retained_output_change_count") != 0
        or manifest.get("effect_counts") != EFFECT_COUNTS
    ):
        raise Plan0069A2Error("A2 frozen bounds drifted.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


if __name__ == "__main__":
    print(json.dumps(execute_a2(), indent=2, sort_keys=True))
