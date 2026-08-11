"""Freeze exact authority for Plan 0069 grouped-assignment normalization."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_plan0068_a0 as plan0068_a0
import speaker_identity_plan0068_a2 as plan0068_a2
import speaker_identity_plan0068_terminal as plan0068_terminal
import speaker_identity_preprocess
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0069-a0-activation.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0069-a0-receipt.v1"
PLAN_ACTIVATION_COMMIT = "278566c"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0069")
DEFAULT_PLAN0068_ROOT = plan0068_a0.DEFAULT_RUNTIME_ROOT
EFFECT_COUNTS = dict(plan0068_a0.EFFECT_COUNTS)
EXPECTED_GROUP_COUNTS = {
    "51272a57a52b0f74abe6": (7, 25),
    "694518476107a0285763": (2, 2),
    "76110321e52a0f513f8f": (1, 1),
}


class Plan0069A0Error(ValueError):
    """Raised when Plan 0068 authority or the grouped inventory drifts."""


def _hash(value: Any) -> str:
    return plan0068_a0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return plan0068_a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        plan0068_a0._validate_content(value, label)
    except plan0068_a0.Plan0068A0Error as exc:
        raise Plan0069A0Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0069A0Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise Plan0069A0Error(f"Expected one {label}; found {len(paths)}.")
    return paths[0]


def grouped_assignment_inventory(readout: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe unambiguous plural assignment objects without changing them."""

    inventory: list[dict[str, Any]] = []
    assignments = readout.get("speaker_assignments") or []
    if not isinstance(assignments, list):
        raise Plan0069A0Error("speaker_assignments must be a list.")
    for assignment_index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            raise Plan0069A0Error("speaker_assignments contains a non-object.")
        utterances = assignment.get("utterance_assignments") or []
        if not isinstance(utterances, list):
            raise Plan0069A0Error("utterance_assignments must be a list.")
        for utterance_index, utterance in enumerate(utterances):
            if not isinstance(utterance, Mapping):
                raise Plan0069A0Error("utterance_assignments contains a non-object.")
            if "utterance_ids" not in utterance:
                continue
            path = (
                f"speaker_assignments[{assignment_index}]."
                f"utterance_assignments[{utterance_index}]"
            )
            if "utterance_id" in utterance:
                raise Plan0069A0Error(f"Mixed singular/plural assignment at {path}.")
            utterance_ids = utterance.get("utterance_ids")
            if (
                not isinstance(utterance_ids, list)
                or not utterance_ids
                or any(not isinstance(value, str) or not value.strip() for value in utterance_ids)
                or len(set(utterance_ids)) != len(utterance_ids)
            ):
                raise Plan0069A0Error(f"Ambiguous grouped assignment at {path}.")
            inventory.append(
                {
                    "path": path,
                    "utterance_ids": list(utterance_ids),
                    "grouped_object_sha256": _hash(dict(utterance)),
                }
            )
    return inventory


def _repository_authority() -> dict[str, Any]:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0069A0Error("A0 requires clean, upstream-even source authority.")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PLAN_ACTIVATION_COMMIT, head],
        check=False,
    ).returncode:
        raise Plan0069A0Error("Plan 0069 activation commit is not in history.")
    source_bindings = []
    for relative in (
        "speaker_identity_preprocess.py",
        "speaker_identity_plan0068_a0.py",
        "speaker_identity_plan0068_a2.py",
        "speaker_identity_plan0069_a0.py",
    ):
        path = root / relative
        source_bindings.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "last_commit": _git("log", "-1", "--format=%H", "--", relative),
            }
        )
    return {
        "plan_activation_commit": PLAN_ACTIVATION_COMMIT,
        "freeze_commit": head,
        "upstream_commit": upstream,
        "source_bindings": source_bindings,
    }


def build_activation_manifest(
    *, plan0068_root: Path = DEFAULT_PLAN0068_ROOT
) -> dict[str, Any]:
    root = plan0068_root.expanduser().resolve()
    a0_receipt = plan0068_a0.replay_activation(runtime_root=root)
    a2_receipt = plan0068_a2.replay_a2(runtime_root=root)
    terminal_receipt = plan0068_terminal.replay_terminal(runtime_root=root)
    a0_path = Path(str(a0_receipt["manifest_path"]))
    a2_path = Path(str(a2_receipt["manifest_path"]))
    terminal_path = Path(str(terminal_receipt["terminal_path"]))
    a0_manifest = read_private_object(a0_path)
    a2_manifest = read_private_object(a2_path)
    terminal = read_private_object(terminal_path)
    for value, label in (
        (a0_manifest, "Plan 0068 A0 manifest"),
        (a2_manifest, "Plan 0068 A2 manifest"),
        (terminal, "Plan 0068 terminal"),
    ):
        _validate_content(value, label)
    if terminal.get("status") != "plan0068_closed_withhold":
        raise Plan0069A0Error("Plan 0068 terminal disposition drifted.")

    cases: list[dict[str, Any]] = []
    for authority in a0_manifest["cases"]:
        document_id = str(authority["document_id"])
        prepared_path = Path(str(authority["legacy_inputs"]["prepared_case"]["path"]))
        status_path = Path(str(authority["legacy_inputs"]["status_artifact"]["path"]))
        prepared = read_private_object(prepared_path)
        status = read_private_object(status_path)
        output_text = str(status.get("output_text") or "")
        output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        if output_sha != authority["output_text_sha256"]:
            raise Plan0069A0Error(f"Retained output drifted: {document_id}.")
        readout = app_intelligence_ledger.extract_json_object(output_text)
        inventory = grouped_assignment_inventory(readout)
        repaired_packet = plan0068_a2.repair_packet(
            prepared["packet"], list(authority["calendar_evidence"])
        )
        _, evidence_ids, _, _ = speaker_identity_preprocess._prepared_identity_references(
            repaired_packet
        )
        grouped_ids = [
            value for item in inventory for value in item["utterance_ids"]
        ]
        if any(
            value not in evidence_ids or not value.startswith("utterance-")
            for value in grouped_ids
        ):
            raise Plan0069A0Error(
                f"Grouped assignment references unprepared evidence: {document_id}."
            )
        expected = EXPECTED_GROUP_COUNTS.get(document_id, (0, 0))
        if (len(inventory), len(grouped_ids)) != expected:
            raise Plan0069A0Error(f"Grouped inventory drifted: {document_id}.")
        cases.append(
            {
                "document_id": document_id,
                "original_recording_filename": authority["original_recording_filename"],
                "retained_output_text_sha256": output_sha,
                "prepared_case": plan0068_a0.legacy_input_binding(prepared_path, root.parent),
                "status_artifact": plan0068_a0.legacy_input_binding(status_path, root.parent),
                "grouped_assignments": inventory,
                "grouped_object_count": len(inventory),
                "expanded_utterance_assignment_count": len(grouped_ids),
            }
        )
    cases.sort(key=lambda item: item["document_id"])
    if len(cases) != 6:
        raise Plan0069A0Error("A0 requires six exact cases.")
    grouped_count = sum(case["grouped_object_count"] for case in cases)
    expanded_count = sum(
        case["expanded_utterance_assignment_count"] for case in cases
    )
    if grouped_count != 10 or expanded_count != 28:
        raise Plan0069A0Error("A0 grouped inventory totals drifted.")
    artifact_bindings = {
        "plan0068_a0_manifest": plan0068_a0.legacy_input_binding(a0_path, root),
        "plan0068_a0_receipt": plan0068_a0.legacy_input_binding(a0_path.parent / "receipt.json", root),
        "plan0068_a2_manifest": plan0068_a0.legacy_input_binding(a2_path, root),
        "plan0068_a2_receipt": plan0068_a0.legacy_input_binding(a2_path.parent / "receipt.json", root),
        "plan0068_terminal": plan0068_a0.legacy_input_binding(terminal_path, root),
    }
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "a0_grouped_assignment_authority_frozen_zero_effect",
            "repository_authority": _repository_authority(),
            "plan0068_a0_content_sha256": a0_manifest["content_sha256"],
            "plan0068_a2_content_sha256": a2_manifest["content_sha256"],
            "plan0068_terminal_content_sha256": terminal["content_sha256"],
            "artifact_bindings": artifact_bindings,
            "cases": cases,
            "case_count": 6,
            "original_recording_filename_count": 6,
            "grouped_object_count": grouped_count,
            "expanded_utterance_assignment_count": expanded_count,
            "model_turn_count": 0,
            "fresh_retrieval_count": 0,
            "fresh_evaluation_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _validate_binding(binding: Mapping[str, Any]) -> None:
    try:
        plan0068_a0._validate_binding(binding)
    except plan0068_a0.Plan0068A0Error as exc:
        raise Plan0069A0Error(str(exc)) from exc


def replay_activation(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    manifest_path = _one(list(root.glob("a0-*/private-manifest.json")), "Plan 0069 A0 manifest")
    receipt_path = manifest_path.parent / "receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    _validate_content(manifest, "Plan 0069 A0 manifest")
    _validate_content(receipt, "Plan 0069 A0 receipt")
    if receipt.get("activation_content_sha256") != manifest["content_sha256"]:
        raise Plan0069A0Error("A0 receipt lost its content binding.")
    if receipt.get("activation_file_sha256") != sha256_file(manifest_path):
        raise Plan0069A0Error("A0 receipt lost its file binding.")
    for binding in manifest["artifact_bindings"].values():
        _validate_binding(binding)
    for case in manifest["cases"]:
        _validate_binding(case["prepared_case"])
        _validate_binding(case["status_artifact"])
    if (
        manifest.get("grouped_object_count") != 10
        or manifest.get("expanded_utterance_assignment_count") != 28
        or manifest.get("effect_counts") != EFFECT_COUNTS
    ):
        raise Plan0069A0Error("A0 frozen bounds drifted.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


def freeze_activation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0068_root: Path = DEFAULT_PLAN0068_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    if list(root.glob("a0-*/receipt.json")):
        return replay_activation(runtime_root=root)
    manifest = build_activation_manifest(plan0068_root=plan0068_root)
    run_root = root / f"a0-{manifest['content_sha256'][:24]}"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    ensure_private_tree(root, run_root)
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "a0_frozen_zero_effect",
            "activation_content_sha256": manifest["content_sha256"],
            "activation_file_sha256": sha256_file(manifest_path),
            "case_count": 6,
            "original_recording_filename_count": 6,
            "grouped_object_count": 10,
            "expanded_utterance_assignment_count": 28,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


if __name__ == "__main__":
    print(json.dumps(freeze_activation(), indent=2, sort_keys=True))
