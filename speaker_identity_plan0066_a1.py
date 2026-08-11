"""Plan 0066 A1 private preparation and zero-source-mutation gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0066_a0 as a0
import speaker_preprocessing_workflow
import transcript_api
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0066-a1-case.v1"
MANIFEST_SCHEMA_VERSION = "transcribe-audio.plan0066-a1-manifest.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0066-a1-receipt.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0066")
DEFAULT_SOURCE_STATE_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_STORE_ROOT = Path("~/.transcripts")
SELECTED_DEVELOPMENT_CASES = (
    ("272cfe27e462506228a4", "20260810T011119Z-speaker-preprocessing-b2e98747"),
    ("413f68d5a8723309e8f8", "20260810T010511Z-speaker-preprocessing-0115dabc"),
    ("51272a57a52b0f74abe6", "20260810T011544Z-speaker-preprocessing-003bf17d"),
    ("694518476107a0285763", "20260810T012746Z-speaker-preprocessing-51e32e0f"),
    ("76110321e52a0f513f8f", "20260810T014556Z-speaker-preprocessing-c5858ff4"),
    ("cbc6b89668613d6b83d6", "20260810T012247Z-speaker-preprocessing-675a4da4"),
)
EFFECT_COUNTS = dict(a0.EFFECT_COUNTS)


class Plan0066A1Error(ValueError):
    """Raised when private preparation fails its roster or integrity gate."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _hash(result)
    return result


def build_case_receipt(
    *,
    document_id: str,
    discovery_run_id: str,
    prepared: Mapping[str, Any],
    expected_roster: Mapping[str, str],
) -> dict[str, Any]:
    """Validate and bind one prepared case while retaining its private packet."""

    packet = prepared.get("packet")
    if not isinstance(packet, Mapping):
        raise Plan0066A1Error("Prepared case is missing its identity packet.")
    people = packet.get("people") or []
    person_map = {
        str(item.get("person_id") or ""): str(item.get("display_name") or "")
        for item in people
        if isinstance(item, Mapping)
    }
    if person_map != dict(expected_roster):
        raise Plan0066A1Error("Prepared case does not contain the exact reviewed roster.")
    if prepared.get("will_send_prompt") is not False:
        raise Plan0066A1Error("A1 preparation attempted to send a model prompt.")
    retrieval = prepared.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise Plan0066A1Error("Prepared case is missing retrieval lineage.")
    if (
        not retrieval.get("source_transcript_sha256")
        or not retrieval.get("preparation_transcript_sha256")
    ):
        raise Plan0066A1Error("Prepared case is missing transcript snapshot lineage.")
    transcript_artifact = prepared.get("transcript_artifact")
    transcript_artifact = (
        transcript_artifact if isinstance(transcript_artifact, Mapping) else {}
    )
    original_path = Path(str(transcript_artifact.get("path") or ""))
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared_with_complete_reviewed_roster",
            "document_id": document_id,
            "discovery_run_id": discovery_run_id,
            "selection_class": "development_gold_known_roster_independent",
            "original_recording_filename": original_path.name,
            "source_transcript_sha256": retrieval["source_transcript_sha256"],
            "preparation_transcript_sha256": retrieval[
                "preparation_transcript_sha256"
            ],
            "source_was_derived": bool(retrieval.get("source_was_derived")),
            "packet_sha256": _hash(packet),
            "packet": dict(packet),
            "prepared_run_id": str(prepared.get("run_id") or ""),
            "input_packet_path": str(prepared.get("input_packet_path") or ""),
            "prompt_packet_path": str(prepared.get("packet_path") or ""),
            "provider_failure_count": len(retrieval.get("source_failures") or []),
            "reviewed_person_count": len(person_map),
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _live_bindings_from_a0(manifest: Mapping[str, Any], *, store_root: Path) -> list[dict[str, Any]]:
    cases = [
        {"document_id": str(item.get("document_id") or "")}
        for item in manifest.get("document_bindings") or []
    ]
    return a0._document_bindings(cases, store_root=store_root)


def execute_a1(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    run_root = root / "a1"
    case_root = run_root / "cases"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    if receipt_path.exists():
        return replay_a1(runtime_root=root, store_root=store_root)
    ensure_private_tree(root, case_root)
    a0_receipt = a0.freeze_activation(
        runtime_root=root,
        store_root=store_root,
    )
    a0_manifest = read_private_object(Path(a0_receipt["manifest_path"]))
    expected_roster = {
        str(item["person_id"]): str(item["primary_name"])
        for item in a0_manifest["reviewed_roster"]
    }
    case_receipts: list[dict[str, Any]] = []
    for document_id, discovery_run_id in SELECTED_DEVELOPMENT_CASES:
        case_path = case_root / f"{document_id}.json"
        if case_path.exists():
            case_receipts.append(read_private_object(case_path))
            continue
        discovery_readout = speaker_preprocessing_workflow.captured_run_json(
            state_root=source_state_root.expanduser().resolve(),
            run_id=discovery_run_id,
        )
        prepared = transcript_api.prepare_selected_speaker_identity_evaluation(
            document_id,
            state_root=run_root / "preparation-state",
            store_root=store_root,
            discovery_readout=discovery_readout,
            source_config_state_root=source_state_root.expanduser().resolve(),
        )
        case_receipt = build_case_receipt(
            document_id=document_id,
            discovery_run_id=discovery_run_id,
            prepared=prepared,
            expected_roster=expected_roster,
        )
        write_immutable_private_json(case_path, case_receipt)
        case_receipts.append(case_receipt)

    current_bindings = _live_bindings_from_a0(a0_manifest, store_root=store_root)
    if current_bindings != a0_manifest["document_bindings"]:
        raise Plan0066A1Error("A1 changed a source, stored artifact, or index row.")
    case_receipts.sort(key=lambda item: str(item["document_id"]))
    manifest = _content(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "a1_roster_and_source_integrity_gate_passed",
            "a0_activation_content_sha256": a0_manifest["content_sha256"],
            "selection_policy": (
                "six_gold_known_development_cases_selected_before_execution; "
                "complete_reviewed_roster_independent_of_gold"
            ),
            "selected_document_ids": [
                document_id for document_id, _ in SELECTED_DEVELOPMENT_CASES
            ],
            "case_content_sha256s": [
                item["content_sha256"] for item in case_receipts
            ],
            "case_packet_sha256s": [item["packet_sha256"] for item in case_receipts],
            "reviewed_roster_sha256": a0_manifest["reviewed_roster_sha256"],
            "reviewed_person_count": len(expected_roster),
            "source_binding_set_sha256_before": a0_manifest[
                "document_binding_set_sha256"
            ],
            "source_binding_set_sha256_after": _hash(current_bindings),
            "source_store_index_change_count": 0,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "a1_passed_zero_source_mutation",
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "prepared_case_count": len(case_receipts),
            "reviewed_person_count": len(expected_roster),
            "source_store_index_change_count": 0,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


def replay_a1(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    run_root = root / "a1"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("content_sha256") != _hash(core):
        raise Plan0066A1Error("A1 manifest content drifted.")
    if (
        receipt.get("manifest_content_sha256") != manifest["content_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise Plan0066A1Error("A1 receipt lost its manifest binding.")
    a0_manifest_path = next(root.glob("a0-*/private-manifest.json"), None)
    if a0_manifest_path is None:
        raise Plan0066A1Error("A1 replay cannot find its A0 authority.")
    a0_manifest = read_private_object(a0_manifest_path)
    current_bindings = _live_bindings_from_a0(a0_manifest, store_root=store_root)
    if current_bindings != a0_manifest["document_bindings"]:
        raise Plan0066A1Error("A1 replay detected source/store/index drift.")
    for document_id in manifest["selected_document_ids"]:
        case = read_private_object(run_root / "cases" / f"{document_id}.json")
        case_core = {key: value for key, value in case.items() if key != "content_sha256"}
        if case.get("content_sha256") != _hash(case_core):
            raise Plan0066A1Error(f"A1 case content drifted: {document_id}.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


if __name__ == "__main__":
    print(json.dumps(execute_a1(), indent=2, sort_keys=True))
