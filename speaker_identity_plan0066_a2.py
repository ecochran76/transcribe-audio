"""Plan 0066 A2 bounded primary-only development inference and measurement."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_plan0066_a1 as a1
import speaker_identity_preprocess
import transcript_api
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)
from speaker_evaluation_baseline import LocalSpeakerCaseRunner


CASE_SCHEMA = "transcribe-audio.plan0066-a2-case.v1"
MANIFEST_SCHEMA = "transcribe-audio.plan0066-a2-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0066-a2-receipt.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0066")
DEFAULT_SOURCE_STATE_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_STORE_ROOT = Path("~/.transcripts")
DEFAULT_GOLD_PATH = Path(
    "~/.local/state/transcribe-audio/plan-0064/"
    "p4-submission-6df988b11c152b78f9da59ab/submitted-decisions.json"
)
PRIOR_IDENTITY_RUNS = {
    "272cfe27e462506228a4": "20260810T011308Z-speaker-preprocessing-bf5832cb",
    "413f68d5a8723309e8f8": "20260810T010732Z-speaker-preprocessing-f399b013",
    "51272a57a52b0f74abe6": "20260810T011821Z-speaker-preprocessing-4a109043",
    "694518476107a0285763": "20260810T013021Z-speaker-preprocessing-22dd535a",
    "76110321e52a0f513f8f": "20260810T014653Z-speaker-preprocessing-49929300",
    "cbc6b89668613d6b83d6": "20260810T012432Z-speaker-preprocessing-0a85d868",
}


class Plan0066A2Error(ValueError):
    """Raised when A2 authority, execution, or measurement drifts."""


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


def build_a2_packet(
    a1_packet: Mapping[str, Any],
    prior_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain prior provenance exactly while adding the A1 reviewed roster."""

    packet = copy.deepcopy(dict(a1_packet))
    prior_sources = copy.deepcopy(list(prior_packet.get("provenance_sources") or []))
    packet["provenance_sources"] = prior_sources
    prior_retrieval = copy.deepcopy(dict(prior_packet.get("retrieval") or {}))
    allowlists = prior_retrieval.get("allowlists")
    allowlists = dict(allowlists) if isinstance(allowlists, Mapping) else {}
    allowlists["person_ids"] = sorted(
        str(item.get("person_id") or "")
        for item in packet.get("people") or []
        if isinstance(item, Mapping) and item.get("person_id")
    )
    prior_retrieval["allowlists"] = allowlists
    packet["retrieval"] = prior_retrieval
    contexts = [
        copy.deepcopy(item)
        for item in prior_packet.get("source_contexts") or []
        if isinstance(item, Mapping)
    ]
    contexts.extend(
        copy.deepcopy(item)
        for item in a1_packet.get("source_contexts") or []
        if isinstance(item, Mapping)
        and str(item.get("relationship_scope") or "") == "reviewed_identity"
    )
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in contexts:
        deduplicated[_hash(item)] = dict(item)
    packet["source_contexts"] = [deduplicated[key] for key in sorted(deduplicated)]
    if _hash(packet["provenance_sources"]) != _hash(prior_sources):
        raise Plan0066A2Error("A2 changed prior provenance evidence.")
    if len(packet.get("people") or []) != 6:
        raise Plan0066A2Error("A2 requires the complete six-person roster.")
    return packet


def measure_cases(
    cases: list[Mapping[str, Any]],
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    predictions: dict[str, Mapping[str, Any]] = {}
    unavailable = 0
    validation_failures = 0
    incomplete_provenance = 0
    for case in cases:
        if case.get("status") != "model_readout_validated":
            unavailable += 1
            validation_failures += case.get("status") == "validation_failed"
            continue
        readout = case.get("validated_readout") or {}
        for assignment in readout.get("speaker_assignments") or []:
            if not isinstance(assignment, Mapping):
                continue
            if assignment.get("status") == "candidate_match" and (
                not assignment.get("provenance_source_ids")
                or not assignment.get("factors")
            ):
                incomplete_provenance += 1
            for label in assignment.get("speaker_labels") or []:
                predictions[f"{case['document_id']}::{label}"] = assignment
    correct = wrong = abstained = 0
    selected = {document_id for document_id, _ in a1.SELECTED_DEVELOPMENT_CASES}
    for decision in gold.get("decisions") or []:
        speaker_ref = str(decision.get("speaker_ref") or "")
        if speaker_ref.split("::", 1)[0] not in selected:
            continue
        prediction = predictions.get(speaker_ref, {})
        if prediction.get("status") != "candidate_match":
            abstained += 1
            continue
        expected = (
            str(decision.get("person_id") or "")
            if decision.get("decision") == "canonical_person"
            else ""
        )
        if expected and prediction.get("person_id") == expected:
            correct += 1
        else:
            wrong += 1
    passed = (
        correct >= 1
        and wrong == 0
        and incomplete_provenance == 0
        and validation_failures == 0
        and unavailable == 0
    )
    return {
        "status": (
            "context_candidate_recovered" if passed else "context_candidate_not_recovered"
        ),
        "passed": passed,
        "correct_prepared_candidate_count": correct,
        "wrong_prepared_candidate_count": wrong,
        "abstained_slot_count": abstained,
        "incomplete_candidate_provenance_count": incomplete_provenance,
        "unavailable_case_count": unavailable,
        "validation_failure_count": validation_failures,
    }


def _prior_packet(state_root: Path, document_id: str) -> dict[str, Any]:
    run_id = PRIOR_IDENTITY_RUNS[document_id]
    path = (
        app_intelligence_ledger.run_dir(state_root, run_id)
        / "artifacts/speaker-preprocessing/identity_evaluation.input.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_cases(run_root: Path, source_state_root: Path) -> list[dict[str, Any]]:
    prepared_root = run_root / "prepared"
    ensure_private_tree(run_root, prepared_root)
    first_document_id = a1.SELECTED_DEVELOPMENT_CASES[0][0]
    first_a1_case = read_private_object(
        run_root.parent / "a1/cases" / f"{first_document_id}.json"
    )
    frozen_prompt = json.loads(
        Path(first_a1_case["prompt_packet_path"]).read_text(encoding="utf-8")
    )
    route = dict(frozen_prompt.get("route") or {})
    if route.get("provider") != "codex-app-server":
        raise Plan0066A2Error("A2 primary route is not codex-app-server.")
    prepared_cases: list[dict[str, Any]] = []
    for document_id, _ in a1.SELECTED_DEVELOPMENT_CASES:
        prepared_path = prepared_root / f"{document_id}.json"
        if prepared_path.exists():
            prepared_cases.append(read_private_object(prepared_path))
            continue
        a1_case = read_private_object(run_root.parent / "a1/cases" / f"{document_id}.json")
        packet = build_a2_packet(
            a1_case["packet"],
            _prior_packet(source_state_root, document_id),
        )
        created = app_intelligence_ledger.create_run(
            state_root=run_root / "state",
            workflow="speaker_preprocessing",
            purpose="Plan 0066 bounded development identity evaluation.",
            document_id=document_id,
            provider="codex-app-server",
        )
        run_id = created["run"]["run_id"]
        prompt = speaker_identity_preprocess.build_identity_evaluation_prompt(packet)
        prompt_packet = app_intelligence_ledger.prepare_model_turn_packet(
            state_root=run_root / "state",
            run_id=run_id,
            task="speaker_identity_evaluation",
            route=route,
            document={"id": document_id, "title": "Private development case"},
            prompt_text=prompt,
            approval_token=app_intelligence_ledger.MODEL_TURN_PREFLIGHT_TOKEN,
        )
        value = _content(
            {
                "document_id": document_id,
                "run_id": run_id,
                "prompt_packet": prompt_packet["packet"],
                "packet": packet,
                "packet_sha256": _hash(packet),
                "prior_provenance_sha256": _hash(packet["provenance_sources"]),
                "original_recording_filename": a1_case[
                    "original_recording_filename"
                ],
                "primary_turn_budget": 1,
                "fallback_turn_budget": 0,
                "retry_budget": 0,
            }
        )
        write_immutable_private_json(prepared_path, value)
        prepared_cases.append(value)
    return prepared_cases


def execute_a2(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    run_root = root / "a2"
    case_root = run_root / "cases"
    ensure_private_tree(root, case_root)
    receipt_path = run_root / "receipt.json"
    if receipt_path.exists():
        return replay_a2(runtime_root=root, store_root=store_root)
    a1.replay_a1(runtime_root=root, store_root=store_root)
    prepared_cases = _prepare_cases(
        run_root,
        source_state_root.expanduser().resolve(),
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=run_root / "state",
        quiet=True,
        static_dir=None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runner = LocalSpeakerCaseRunner(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        timeout_seconds=timeout_seconds,
    )
    case_receipts: list[dict[str, Any]] = []
    try:
        for prepared in prepared_cases:
            document_id = prepared["document_id"]
            case_path = case_root / f"{document_id}.json"
            if case_path.exists():
                case_receipts.append(read_private_object(case_path))
                continue
            try:
                status = runner._execute_prepared(prepared)
                readout = app_intelligence_ledger.extract_json_object(
                    str(status.get("output_text") or "")
                )
                validated = speaker_identity_preprocess.validate_and_score_identity_evaluation(
                    prepared["packet"], readout
                )
                case = _content(
                    {
                        "schema_version": CASE_SCHEMA,
                        "status": "model_readout_validated",
                        "document_id": document_id,
                        "run_id": prepared["run_id"],
                        "packet_sha256": prepared["packet_sha256"],
                        "prior_provenance_sha256": prepared[
                            "prior_provenance_sha256"
                        ],
                        "original_recording_filename": prepared[
                            "original_recording_filename"
                        ],
                        "validated_readout": validated["readout"],
                        "primary_model_turn_count": 1,
                        "fallback_model_turn_count": 0,
                        "retry_count": 0,
                    }
                )
            except Exception as exc:
                case = _content(
                    {
                        "schema_version": CASE_SCHEMA,
                        "status": "validation_failed",
                        "document_id": document_id,
                        "run_id": prepared["run_id"],
                        "packet_sha256": prepared["packet_sha256"],
                        "prior_provenance_sha256": prepared[
                            "prior_provenance_sha256"
                        ],
                        "original_recording_filename": prepared[
                            "original_recording_filename"
                        ],
                        "reason": str(exc),
                        "primary_model_turn_count": 1,
                        "fallback_model_turn_count": 0,
                        "retry_count": 0,
                    }
                )
            write_immutable_private_json(case_path, case)
            case_receipts.append(case)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    gold = json.loads(DEFAULT_GOLD_PATH.expanduser().read_text(encoding="utf-8"))
    measurement = measure_cases(case_receipts, gold)
    a1.replay_a1(runtime_root=root, store_root=store_root)
    manifest = _content(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": measurement["status"],
            "a1_manifest_content_sha256": read_private_object(
                root / "a1/private-manifest.json"
            )["content_sha256"],
            "selected_document_ids": [item[0] for item in a1.SELECTED_DEVELOPMENT_CASES],
            "case_content_sha256s": [item["content_sha256"] for item in case_receipts],
            "measurement": measurement,
            "execution_counts": {
                "primary_model_turns": sum(
                    int(item["primary_model_turn_count"]) for item in case_receipts
                ),
                "fallback_model_turns": 0,
                "retries": 0,
            },
            "source_store_index_change_count": 0,
            "will_apply_assignments": False,
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
            "source_store_index_change_count": 0,
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


def replay_a2(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT, store_root: Path = DEFAULT_STORE_ROOT
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    manifest_path = root / "a2/private-manifest.json"
    receipt_path = root / "a2/receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    core = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("content_sha256") != _hash(core):
        raise Plan0066A2Error("A2 manifest content drifted.")
    if receipt.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise Plan0066A2Error("A2 receipt lost its file binding.")
    a1.replay_a1(runtime_root=root, store_root=store_root)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


if __name__ == "__main__":
    print(json.dumps(execute_a2(), indent=2, sort_keys=True))
