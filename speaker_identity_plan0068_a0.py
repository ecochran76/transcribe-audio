"""Freeze Plan 0068 authority with mode-preserving legacy input bindings."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import app_intelligence_ledger
import speaker_identity_plan0066_a2 as plan0066_a2
import speaker_identity_plan0067_a0 as prior_a0
import speaker_identity_plan0067_terminal as plan0067_terminal
import speaker_identity_preprocess
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0068-a0-activation.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0068-a0-receipt.v1"
PLAN_ACTIVATION_COMMIT = "981efe1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0068")
DEFAULT_PLAN0066_ROOT = prior_a0.DEFAULT_PLAN0066_ROOT
DEFAULT_PLAN0067_ROOT = prior_a0.DEFAULT_RUNTIME_ROOT
DEFAULT_SOURCE_STATE_ROOT = prior_a0.DEFAULT_SOURCE_STATE_ROOT
DEFAULT_GOLD_PATH = prior_a0.DEFAULT_GOLD_PATH
EFFECT_COUNTS = dict(prior_a0.EFFECT_COUNTS)


class Plan0068A0Error(ValueError):
    """Raised when inherited replay authority is incomplete or drifts."""


def _hash(value: Any) -> str:
    return prior_a0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return prior_a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        prior_a0._validate_content(value, label)
    except prior_a0.Plan0067A0Error as exc:
        raise Plan0068A0Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0068A0Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def legacy_input_binding(path: Path, root: Path) -> dict[str, Any]:
    """Bind an inherited file by shape, root, hash, and observed mode without chmod."""

    expanded = path.expanduser().absolute()
    if expanded.is_symlink() or not expanded.is_file():
        raise Plan0068A0Error(f"Legacy input is not a regular non-symlinked file: {expanded}.")
    resolved = expanded.resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise Plan0068A0Error(f"Legacy input is outside its authority root: {resolved}.") from exc
    return {
        "path": str(resolved),
        "file_sha256": sha256_file(resolved),
        "observed_mode": f"{resolved.stat().st_mode & 0o777:04o}",
        "mode_was_changed": False,
    }


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise Plan0068A0Error(f"Expected one {label}; found {len(paths)}.")
    return paths[0]


def _repository_authority() -> dict[str, Any]:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0068A0Error("A0 requires a clean, upstream-even repository.")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PLAN_ACTIVATION_COMMIT, head],
        check=False,
    ).returncode:
        raise Plan0068A0Error("Plan 0068 activation commit is not in history.")
    source_bindings = []
    for relative in (
        "speaker_identity_preprocess.py",
        "speaker_identity_plan0066_a2.py",
        "speaker_identity_plan0067_a0.py",
        "speaker_identity_plan0068_a0.py",
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
    *,
    plan0066_root: Path = DEFAULT_PLAN0066_ROOT,
    plan0067_root: Path = DEFAULT_PLAN0067_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> dict[str, Any]:
    source_root = plan0066_root.expanduser().resolve()
    state_root = source_state_root.expanduser().resolve()
    terminal_path = _one(list(source_root.glob("terminal-*/terminal.json")), "Plan 0066 terminal")
    a0_manifest_path = _one(list(source_root.glob("a0-*/private-manifest.json")), "Plan 0066 A0 manifest")
    plan67_terminal = plan0067_terminal.replay_terminal(runtime_root=plan0067_root)
    terminal = read_private_object(terminal_path)
    plan66_a0 = read_private_object(a0_manifest_path)
    _validate_content(terminal, "Plan 0066 terminal")
    _validate_content(plan66_a0, "Plan 0066 A0 manifest")
    if terminal.get("status") != "plan0066_closed_withhold":
        raise Plan0068A0Error("Plan 0066 terminal disposition drifted.")

    gold_resolved = gold_path.expanduser().resolve()
    gold = read_private_object(gold_resolved)
    if gold.get("authority_content_sha256") != "6df988b11c152b78f9da59ab6d2324516082196d70d0340ecba2298051582f67":
        raise Plan0068A0Error("Human-gold authority drifted.")

    document_bindings = {
        str(item.get("document_id") or ""): item
        for item in plan66_a0.get("document_bindings") or []
        if isinstance(item, Mapping)
    }
    cases: list[dict[str, Any]] = []
    for prepared_path in sorted((source_root / "a2/prepared").glob("*.json")):
        document_id = prepared_path.stem
        a1_path = source_root / "a1/cases" / f"{document_id}.json"
        failed_path = source_root / "a2/cases" / f"{document_id}.json"
        prepared = read_private_object(prepared_path)
        a1_case = read_private_object(a1_path)
        failed_case = read_private_object(failed_path)
        document_binding = document_bindings.get(document_id)
        if not document_binding:
            raise Plan0068A0Error(f"Missing source binding: {document_id}.")
        transcript_path = Path(str(document_binding.get("stored_path") or ""))
        transcript_binding = legacy_input_binding(
            transcript_path,
            Path("~/.transcripts/artifacts"),
        )
        if transcript_binding["file_sha256"] != document_binding.get("stored_sha256"):
            raise Plan0068A0Error(f"Stored transcript hash drifted: {document_id}.")
        transcript = json.loads(Path(transcript_binding["path"]).read_text(encoding="utf-8"))
        discovery_packet = speaker_identity_preprocess.build_clue_discovery_packet(
            transcript=transcript,
            source_contexts=a1_case.get("packet", {}).get("source_contexts") or (),
        )
        speaker_identity_preprocess.validate_clue_discovery_readout(
            discovery_packet,
            a1_case.get("packet", {}).get("discovery_readout") or {},
        )

        status_path = _one(
            list(
                (
                    source_root
                    / "a2/state/app-intelligence-runs"
                    / str(prepared.get("run_id") or "")
                    / "artifacts/model-turn-readouts"
                ).glob("*.status.json")
            ),
            f"retained status for {document_id}",
        )
        status = read_private_object(status_path)
        prior_run_id = plan0066_a2.PRIOR_IDENTITY_RUNS[document_id]
        prior_path = (
            app_intelligence_ledger.run_dir(state_root, prior_run_id)
            / "artifacts/speaker-preprocessing/identity_evaluation.input.json"
        )
        prior_packet = json.loads(prior_path.read_text(encoding="utf-8"))
        expected_packet = plan0066_a2.build_a2_packet(a1_case["packet"], prior_packet)
        try:
            case = prior_a0.build_case_binding(
                document_id=document_id,
                a1_case=a1_case,
                prepared=prepared,
                expected_packet=expected_packet,
                failed_case=failed_case,
                status=status,
                calendar_evidence=discovery_packet["calendar_evidence"],
            )
        except prior_a0.Plan0067A0Error as exc:
            raise Plan0068A0Error(str(exc)) from exc
        case["legacy_inputs"] = {
            "a1_case": legacy_input_binding(a1_path, source_root),
            "prepared_case": legacy_input_binding(prepared_path, source_root),
            "failed_case": legacy_input_binding(failed_path, source_root),
            "status_artifact": legacy_input_binding(status_path, source_root),
            "prior_packet_artifact": legacy_input_binding(prior_path, state_root),
            "transcript_artifact": transcript_binding,
        }
        cases.append(case)

    rejected_count = sum(
        len(case["plan0066_rejected_calendar_evidence_ids"]) for case in cases
    )
    if len(cases) != 6 or rejected_count != 7:
        raise Plan0068A0Error("A0 requires six cases and seven rejected calendar IDs.")
    artifact_bindings = {
        "plan0066_terminal": legacy_input_binding(terminal_path, source_root),
        "plan0066_a0_manifest": legacy_input_binding(a0_manifest_path, source_root),
        "plan0066_a0_receipt": legacy_input_binding(a0_manifest_path.parent / "receipt.json", source_root),
        "plan0066_a1_manifest": legacy_input_binding(source_root / "a1/private-manifest.json", source_root),
        "plan0066_a1_receipt": legacy_input_binding(source_root / "a1/receipt.json", source_root),
        "plan0066_a2_manifest": legacy_input_binding(source_root / "a2/private-manifest.json", source_root),
        "plan0066_a2_receipt": legacy_input_binding(source_root / "a2/receipt.json", source_root),
        "plan0067_terminal": legacy_input_binding(Path(plan67_terminal["terminal_path"]), plan0067_root),
        "human_gold": legacy_input_binding(gold_resolved, gold_resolved.parent),
    }
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "a0_legacy_authority_frozen_zero_effect",
            "repository_authority": _repository_authority(),
            "plan0066_terminal_content_sha256": terminal["content_sha256"],
            "plan0067_terminal_content_sha256": plan67_terminal["content_sha256"],
            "human_gold_authority_content_sha256": gold["authority_content_sha256"],
            "artifact_bindings": artifact_bindings,
            "cases": cases,
            "case_count": 6,
            "rejected_calendar_evidence_id_count": 7,
            "original_recording_filename_count": sum(
                bool(case["original_recording_filename"]) for case in cases
            ),
            "legacy_input_mode_change_count": 0,
            "model_turn_count": 0,
            "reference_repair_count": 0,
            "fresh_retrieval_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _validate_binding(binding: Mapping[str, Any]) -> None:
    path = Path(str(binding["path"])).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or sha256_file(path) != binding["file_sha256"]:
        raise Plan0068A0Error(f"Frozen legacy input drifted: {path}.")
    mode = f"{path.stat().st_mode & 0o777:04o}"
    if mode != binding["observed_mode"] or binding.get("mode_was_changed") is not False:
        raise Plan0068A0Error(f"Frozen legacy input mode drifted: {path}.")


def replay_activation(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    manifest_path = _one(list(root.glob("a0-*/private-manifest.json")), "Plan 0068 A0 manifest")
    receipt_path = manifest_path.parent / "receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    _validate_content(manifest, "Plan 0068 A0 manifest")
    _validate_content(receipt, "Plan 0068 A0 receipt")
    if receipt.get("activation_content_sha256") != manifest["content_sha256"]:
        raise Plan0068A0Error("A0 receipt lost its content binding.")
    if receipt.get("activation_file_sha256") != sha256_file(manifest_path):
        raise Plan0068A0Error("A0 receipt lost its file binding.")
    for binding in manifest["artifact_bindings"].values():
        _validate_binding(binding)
    for case in manifest["cases"]:
        for binding in case["legacy_inputs"].values():
            _validate_binding(binding)
    if manifest.get("effect_counts") != EFFECT_COUNTS:
        raise Plan0068A0Error("A0 effect budget drifted.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


def freeze_activation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0066_root: Path = DEFAULT_PLAN0066_ROOT,
    plan0067_root: Path = DEFAULT_PLAN0067_ROOT,
    source_state_root: Path = DEFAULT_SOURCE_STATE_ROOT,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    if list(root.glob("a0-*/receipt.json")):
        return replay_activation(runtime_root=root)
    manifest = build_activation_manifest(
        plan0066_root=plan0066_root,
        plan0067_root=plan0067_root,
        source_state_root=source_state_root,
        gold_path=gold_path,
    )
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
            "rejected_calendar_evidence_id_count": 7,
            "original_recording_filename_count": 6,
            "legacy_input_mode_change_count": 0,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


if __name__ == "__main__":
    print(json.dumps(freeze_activation(), indent=2, sort_keys=True))
