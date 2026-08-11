"""Freeze Plan 0070 joined/residual development authority."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import speaker_identity_plan0065_d0 as plan0065_d0
import speaker_identity_plan0065_d1 as plan0065_d1
import speaker_identity_plan0065_d2 as plan0065_d2
import speaker_identity_plan0069_a0 as plan0069_a0
import speaker_identity_plan0069_a2 as plan0069_a2
import speaker_identity_plan0069_terminal as plan0069_terminal
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    sha256_file,
    write_immutable_private_json,
)


SCHEMA_VERSION = "transcribe-audio.plan0070-d0-activation.v1"
RECEIPT_SCHEMA_VERSION = "transcribe-audio.plan0070-d0-receipt.v1"
PLAN_ACTIVATION_COMMIT = "305e8e0"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0070")
DEFAULT_PLAN0065_ROOT = plan0065_d0.DEFAULT_RUNTIME_ROOT
DEFAULT_PLAN0069_ROOT = plan0069_a0.DEFAULT_RUNTIME_ROOT
EFFECT_COUNTS = dict(plan0069_a0.EFFECT_COUNTS)
EXPECTED_D3_START = {
    "recording_count": 12,
    "speaker_slot_count": 39,
    "correct_acoustic_candidate_count": 10,
    "wrong_acoustic_candidate_count": 0,
    "correct_context_candidate_count": 5,
    "wrong_context_candidate_count": 1,
    "correct_pillar_agreement_count": 5,
    "wrong_combined_candidate_count": 0,
    "residual_acceptance_count": 0,
}


class Plan0070D0Error(ValueError):
    """Raised when inherited joined/residual authority is incomplete or drifts."""


def _hash(value: Any) -> str:
    return plan0069_a0._hash(value)


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    return plan0069_a0._content(value)


def _validate_content(value: Mapping[str, Any], label: str) -> None:
    try:
        plan0069_a0._validate_content(value, label)
    except plan0069_a0.Plan0069A0Error as exc:
        raise Plan0070D0Error(str(exc)) from exc


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise Plan0070D0Error(result.stderr.strip() or "Git authority read failed.")
    return result.stdout.strip()


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise Plan0070D0Error(f"Expected one {label}; found {len(paths)}.")
    return paths[0]


def _binding(path: Path, root: Path) -> dict[str, Any]:
    try:
        return plan0069_a0.plan0068_a0.legacy_input_binding(path, root)
    except plan0069_a0.plan0068_a0.Plan0068A0Error as exc:
        raise Plan0070D0Error(str(exc)) from exc


def _all_zero(values: Mapping[str, Any]) -> bool:
    return all(not int(value or 0) for value in values.values())


def _repository_authority() -> dict[str, Any]:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream or _git("status", "--porcelain=v1"):
        raise Plan0070D0Error("D0 requires clean, upstream-even source authority.")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PLAN_ACTIVATION_COMMIT, head],
        check=False,
    ).returncode:
        raise Plan0070D0Error("Plan 0070 activation commit is not in history.")
    source_bindings = []
    for relative in (
        "speaker_identity_plan0064_p2.py",
        "speaker_identity_plan0064_p3.py",
        "speaker_identity_plan0065_d1.py",
        "speaker_identity_plan0069_a2.py",
        "speaker_identity_plan0070_d0.py",
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
    plan0065_root: Path = DEFAULT_PLAN0065_ROOT,
    plan0069_root: Path = DEFAULT_PLAN0069_ROOT,
) -> dict[str, Any]:
    p65 = plan0065_root.expanduser().resolve()
    p69 = plan0069_root.expanduser().resolve()
    authority_root = p65.parent
    p65_d0_path = _one(list(p65.glob("d0-*/private-manifest.json")), "Plan 0065 D0 manifest")
    p65_d1_policy_path = _one(list(p65.glob("d1-*/policy.json")), "Plan 0065 D1 policy")
    p65_d1_evidence_path = p65_d1_policy_path.with_name("private-development-evidence.json")
    p65_d1_receipt_path = p65_d1_policy_path.with_name("receipt.json")
    p65_d2_receipt_path = _one(list(p65.glob("d2-execution-*/receipt.json")), "Plan 0065 D2 receipt")
    p65_d2_cases_root = p65_d2_receipt_path.parent / "cases"
    p69_a0_path = _one(list(p69.glob("a0-*/private-manifest.json")), "Plan 0069 A0 manifest")
    p69_a2_path = p69 / "a2/private-manifest.json"
    p69_terminal_path = _one(list(p69.glob("terminal-*/terminal.json")), "Plan 0069 terminal")
    plan64_paths = plan0065_d0._plan0064_paths(plan0065_d0.DEFAULT_PLAN0064_ROOT)
    p64_p0_path = plan64_paths["p0"] / "private-manifest.json"
    p64_p1_path = plan64_paths["p1"] / "private-acoustic-evidence.json"
    p64_p1_receipt_path = plan64_paths["p1"] / "receipt.json"
    p64_gold_path = plan64_paths["measurement"] / "human-gold.json"

    p65_d1_policy = read_private_object(p65_d1_policy_path)
    plan0065_d1.replay_d1(
        policy_content_sha256=str(p65_d1_policy.get("content_sha256") or ""),
        runtime_root=p65,
    )
    plan0069_terminal.replay_terminal(runtime_root=p69)

    p65_d0 = read_private_object(p65_d0_path)
    p65_d1_evidence = read_private_object(p65_d1_evidence_path)
    p65_d2_receipt = read_private_object(p65_d2_receipt_path)
    p69_a0 = read_private_object(p69_a0_path)
    p69_a2 = read_private_object(p69_a2_path)
    p69_terminal = read_private_object(p69_terminal_path)
    p64_p1 = read_private_object(p64_p1_path)
    p64_gold = read_private_object(p64_gold_path)
    for value, label in (
        (p65_d0, "Plan 0065 D0 manifest"),
        (p65_d1_policy, "Plan 0065 D1 policy"),
        (p65_d1_evidence, "Plan 0065 D1 evidence"),
        (p65_d2_receipt, "Plan 0065 D2 receipt"),
        (p69_a0, "Plan 0069 A0 manifest"),
        (p69_a2, "Plan 0069 A2 manifest"),
        (p69_terminal, "Plan 0069 terminal"),
        (p64_p1, "Plan 0064 P1 evidence"),
        (p64_gold, "Plan 0064 human gold"),
    ):
        _validate_content(value, label)
    if p69_terminal.get("status") != "plan0069_closed_pass":
        raise Plan0070D0Error("Plan 0069 terminal is not PASS.")
    if p65_d1_evidence.get("development_gate", {}).get("passed") is not True:
        raise Plan0070D0Error("Plan 0065 D1 acoustic gate is not PASS.")
    if not _all_zero(p65_d1_evidence.get("action_counts") or {}):
        raise Plan0070D0Error("Plan 0065 D1 action counts are nonzero.")
    if p69_a2.get("effect_counts") != EFFECT_COUNTS:
        raise Plan0070D0Error("Plan 0069 effect counts drifted.")

    d2_case_paths = sorted(p65_d2_cases_root.glob("*.json"))
    p69_case_paths = sorted((p69 / "a2/cases").glob("*.json"))
    if len(d2_case_paths) != 12 or len(p69_case_paths) != 6:
        raise Plan0070D0Error("Inherited context case denominator drifted.")
    original_filenames = list(p69_a2.get("original_recording_filenames") or [])
    if len(original_filenames) != 6 or any(not name for name in original_filenames):
        raise Plan0070D0Error("Plan 0069 original filename authority is incomplete.")
    exposure = p65_d0["plan0064_authority"]["exposure_set"]
    if (
        exposure.get("full_recordings") != 12
        or exposure.get("decision_rows") != 39
        or len(p64_p1.get("recordings") or []) != 12
        or len(p64_gold.get("decisions") or []) != 39
    ):
        raise Plan0070D0Error("Development exposure denominator drifted.")

    artifact_bindings = {
        "plan0065_d0_manifest": _binding(p65_d0_path, authority_root),
        "plan0065_d0_receipt": _binding(p65_d0_path.with_name("receipt.json"), authority_root),
        "plan0065_d1_policy": _binding(p65_d1_policy_path, authority_root),
        "plan0065_d1_evidence": _binding(p65_d1_evidence_path, authority_root),
        "plan0065_d1_receipt": _binding(p65_d1_receipt_path, authority_root),
        "plan0065_d2_receipt": _binding(p65_d2_receipt_path, authority_root),
        "plan0069_a0_manifest": _binding(p69_a0_path, authority_root),
        "plan0069_a2_manifest": _binding(p69_a2_path, authority_root),
        "plan0069_terminal": _binding(p69_terminal_path, authority_root),
        "plan0064_p0_manifest": _binding(p64_p0_path, authority_root),
        "plan0064_p1_evidence": _binding(p64_p1_path, authority_root),
        "plan0064_p1_receipt": _binding(p64_p1_receipt_path, authority_root),
        "plan0064_human_gold": _binding(p64_gold_path, authority_root),
    }
    return _content(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "d0_joined_residual_authority_frozen_zero_effect",
            "repository_authority": _repository_authority(),
            "artifact_bindings": artifact_bindings,
            "plan0065_d2_case_bindings": [
                _binding(path, authority_root) for path in d2_case_paths
            ],
            "plan0069_case_bindings": [
                _binding(path, authority_root) for path in p69_case_paths
            ],
            "exposure_set_content_sha256": exposure["content_sha256"],
            "exposure_counts": {
                key: exposure[key]
                for key in (
                    "document_ids",
                    "full_recordings",
                    "recording_hashes",
                    "probe_hashes",
                    "source_windows",
                    "review_clips",
                    "decision_rows",
                )
            },
            "original_recording_filenames": original_filenames,
            "original_recording_filename_count": 6,
            "expected_d3_start": dict(EXPECTED_D3_START),
            "model_turn_count": 0,
            "fresh_retrieval_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )


def _validate_binding(binding: Mapping[str, Any]) -> None:
    try:
        plan0069_a0.plan0068_a0._validate_binding(binding)
    except plan0069_a0.plan0068_a0.Plan0068A0Error as exc:
        raise Plan0070D0Error(str(exc)) from exc


def replay_activation(*, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    manifest_path = _one(list(root.glob("d0-*/private-manifest.json")), "Plan 0070 D0 manifest")
    receipt_path = manifest_path.parent / "receipt.json"
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    _validate_content(manifest, "Plan 0070 D0 manifest")
    _validate_content(receipt, "Plan 0070 D0 receipt")
    if receipt.get("activation_content_sha256") != manifest["content_sha256"]:
        raise Plan0070D0Error("D0 receipt lost its content binding.")
    if receipt.get("activation_file_sha256") != sha256_file(manifest_path):
        raise Plan0070D0Error("D0 receipt lost its file binding.")
    for binding in manifest["artifact_bindings"].values():
        _validate_binding(binding)
    for key in ("plan0065_d2_case_bindings", "plan0069_case_bindings"):
        for binding in manifest[key]:
            _validate_binding(binding)
    if (
        manifest.get("expected_d3_start") != EXPECTED_D3_START
        or manifest.get("original_recording_filename_count") != 6
        or manifest.get("effect_counts") != EFFECT_COUNTS
    ):
        raise Plan0070D0Error("D0 frozen contract drifted.")
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": True}


def freeze_activation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    plan0065_root: Path = DEFAULT_PLAN0065_ROOT,
    plan0069_root: Path = DEFAULT_PLAN0069_ROOT,
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    if list(root.glob("d0-*/receipt.json")):
        return replay_activation(runtime_root=root)
    manifest = build_activation_manifest(
        plan0065_root=plan0065_root,
        plan0069_root=plan0069_root,
    )
    run_root = root / f"d0-{manifest['content_sha256'][:24]}"
    manifest_path = run_root / "private-manifest.json"
    receipt_path = run_root / "receipt.json"
    ensure_private_tree(root, run_root)
    write_immutable_private_json(manifest_path, manifest)
    receipt = _content(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "d0_frozen_zero_effect",
            "activation_content_sha256": manifest["content_sha256"],
            "activation_file_sha256": sha256_file(manifest_path),
            "recording_count": 12,
            "speaker_slot_count": 39,
            "original_recording_filename_count": 6,
            "model_turn_count": 0,
            "effect_counts": dict(EFFECT_COUNTS),
        }
    )
    write_immutable_private_json(receipt_path, receipt)
    return {**receipt, "manifest_path": str(manifest_path), "idempotent_replay": False}


if __name__ == "__main__":
    print(json.dumps(freeze_activation(), indent=2, sort_keys=True))
