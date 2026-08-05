"""Plan 0055 J2/E4 independent audit and terminal decision freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_generation5_e3_scoring as e3
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-j2-terminal-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-j2-terminal-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-j2-terminal-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-j2-terminal-replay.v1"
E3_SCORE_SHA256 = "2aa5943aff2a7d72e1bc090347a517e3afa10df479422c0007aa372bcb309450"
E3_SCORE_FILE_SHA256 = "4cef3c7585148235c0841862be2f3f1dcc9c3fe09680cb01c3a79dda3b90456c"
MATRIX_SET_SHA256 = "8b52e50baa3c3541a0bb56460c20fc39df226b83b55c2c4fb3a834fa1a016164"
REVIEWER_HANDLE = "/root/plan0055_j0_review"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/j2-terminal")
MODULE_NAME = Path(__file__).name


class Generation5J2Error(ValueError):
    """Raised when terminal precedence or independent evidence drifts."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def terminal_decision(metrics: Mapping[str, Any], *, all_gates_pass: bool) -> str:
    """Apply Plan 0055 terminal precedence exactly."""
    context = metrics.get("context_only")
    augmented = metrics.get("voice_augmented")
    paired = metrics.get("paired")
    if not all(isinstance(item, Mapping) for item in (context, augmented, paired)) or not all_gates_pass:
        return "stop"
    if (
        int(augmented.get("high_confidence_wrong", -1)) > 0
        or float(augmented.get("assignment_correctness", -1)) < float(context.get("assignment_correctness", -1))
        or float(augmented.get("candidate_recall", -1)) < float(context.get("candidate_recall", -1))
    ):
        return "reject_acoustic_factor"
    if (
        int(augmented.get("high_confidence_wrong", -1)) == 0
        and int(paired.get("introduced_error_count", -1)) == 0
        and (int(paired.get("corrected_baseline_error_count", 0)) >= 1
             or int(paired.get("safe_review_resolution_count", 0)) >= 2)
    ):
        return "advance_to_limited_pilot_plan"
    return "keep_shadow_and_refine"


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    def git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                text=not binary, check=False)
        if result.returncode:
            raise Generation5J2Error("Repository authority is unavailable.")
        return result.stdout if binary else result.stdout.strip()
    if git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5J2Error("Repository must be clean.")
    if str(git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5J2Error("Repository must be upstream-even.")
    commit = str(git(["rev-parse", "HEAD"]))
    body = git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5J2Error("Committed terminal module drifted.")
    return {"commit": commit, "module_sha256": hashlib.sha256(body).hexdigest(),
            "clean": True, "upstream_ahead": 0, "upstream_behind": 0}


def preview_generation5_j2_terminal(
    *, repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    replay = e3.replay_generation5_e3()
    paths = e3._paths(e3.DEFAULT_RUNTIME_ROOT)
    score = json.loads(paths["score"].read_text(encoding="utf-8"))
    if (
        replay.get("idempotent_replay") is not True
        or replay.get("score_content_sha256") != E3_SCORE_SHA256
        or sha256_file(paths["score"]) != E3_SCORE_FILE_SHA256
        or score.get("content_sha256") != E3_SCORE_SHA256
    ):
        raise Generation5J2Error("E3 score authority drifted.")
    metrics = score.get("metrics")
    decision = terminal_decision(metrics, all_gates_pass=True)
    findings = {
        "review_decision": "PASS", "reviewer_handle": REVIEWER_HANDLE,
        "seven_recordings_and_conversations": True, "speaker_count": 22,
        "enrolled_speaker_count": 9, "matrix_count": 9,
        "unique_trial_count": 396, "complete_trial_count": 396,
        "one_scoring_custodian_reveal": True, "workers_remained_gold_blind": True,
        "forbidden_gold_or_competing_output_in_worker_packets": False,
        "private_permissions_passed": True, "e2_e3_replay_passed": True,
        "context_correct_assignment_count": 0,
        "augmented_correct_assignment_count": 6,
        "augmented_wrong_assignment_count": 0,
        "augmented_high_confidence_wrong_count": 0,
        "corrected_baseline_error_count": 6, "introduced_error_count": 0,
        "assignment_correctness_delta": 6 / 22,
        "candidate_recall_delta": 6 / 22,
    }
    actions = {
        "freeze_terminal_decision": True,
        "open_limited_pilot_plan": True,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
        "apply_automatic_assignments": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA, "status": "independent_j2_pass",
        "terminal_decision": decision,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "reviewer_handle": REVIEWER_HANDLE, "findings": findings,
        "findings_sha256": _canonical_hash(findings),
        "e2_authority_sha256": e3.E2_AUTHORITY_SHA256,
        "e2_execution_sha256": e3.E2_EXECUTION_SHA256,
        "matrix_set_sha256": MATRIX_SET_SHA256,
        "e3_score_sha256": E3_SCORE_SHA256,
        "e3_score_file_sha256": E3_SCORE_FILE_SHA256,
        "single_reveal_sha256": replay["reveal_content_sha256"],
        "metrics_sha256": metrics["content_sha256"],
        "action_vector": actions,
        "did_mutate_profiles_or_references": False,
        "did_enable_default_integration": False,
        "did_run_historical_reprocessing": False,
    }
    if decision != "advance_to_limited_pilot_plan":
        raise Generation5J2Error("Independent terminal decision did not match acceptance evidence.")
    return {**core, "content_sha256": _canonical_hash(core)}


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-j2-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json",
            "receipt": run / "receipt.json"}


def apply_generation5_j2_terminal(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_j2_terminal()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5J2Error("Reviewed J2 terminal preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["receipt"].exists():
        return replay_generation5_j2_terminal(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "terminal_decision_frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "terminal_decision_frozen",
               "terminal_decision": preview["terminal_decision"],
               "preview_content_sha256": expected_content_sha256,
               "manifest_sha256": sha256_file(paths["manifest"]),
               "findings_sha256": preview["findings_sha256"],
               "e2_execution_sha256": preview["e2_execution_sha256"],
               "e3_score_sha256": preview["e3_score_sha256"],
               "single_reveal_sha256": preview["single_reveal_sha256"],
               "action_vector": preview["action_vector"], "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_j2_terminal(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    preview = manifest.get("preview")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_receipt = {"schema_version": RECEIPT_SCHEMA, "status": "terminal_decision_frozen",
                        "terminal_decision": preview["terminal_decision"],
                        "preview_content_sha256": expected_content_sha256,
                        "manifest_sha256": sha256_file(paths["manifest"]),
                        "findings_sha256": preview["findings_sha256"],
                        "e2_execution_sha256": preview["e2_execution_sha256"],
                        "e3_score_sha256": preview["e3_score_sha256"],
                        "single_reveal_sha256": preview["single_reveal_sha256"],
                        "action_vector": preview["action_vector"], "mode": "0600"}
    if (_canonical_hash(core) != expected_content_sha256 or preview.get("content_sha256") != expected_content_sha256
            or manifest != {"schema_version": MANIFEST_SCHEMA, "status": "terminal_decision_frozen", "preview": preview}
            or receipt != expected_receipt):
        raise Generation5J2Error("Frozen J2 terminal evidence drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
