"""Plan 0052 immutable early terminal decision after frozen G3 preparation failure."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import acoustic_audio_derivatives as p1
import acoustic_generation4_cohort as cohort
import acoustic_generation4_freeze as g2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation4-terminal-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation4-terminal-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-terminal-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-terminal-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052/g6")
MODULE_NAME = Path(__file__).name
G2_PREVIEW_SHA256 = (
    "6d6e86094c809c34c45694c311063c06570020348eccd6f65a420535167e3d41"
)
G2_MANIFEST_SHA256 = (
    "20cb311ebf436ffdd382c4715de2987d854d1d5b5c56974739d1c4f94c96ae61"
)
G3_EXECUTION_G2_PREVIEW_SHA256 = (
    "cc3668c01d7f731ddc340c6bf39dd4983a4bb63b26da9314a6a1da14e14198ee"
)
G3_EXECUTION_G2_MANIFEST_SHA256 = (
    "ccd46ba08725ccccaaabc35e41bd1f56db8cae44e4afaa294150951de2e70a41"
)
FAILED_SOURCE_SHA256 = (
    "843ad4d9effde8b7"  # Exact full value is resolved from frozen private membership.
)
P1_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052/g3/p1")
P2_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0052/g3/p2")


class Generation4TerminalError(ValueError):
    """Raised when the Generation-4 terminal stop cannot replay exactly."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation4TerminalError("Private terminal authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation4TerminalError("Private terminal authority must be an object.")
    return value


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    parity = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "@{upstream}...HEAD"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if status.returncode or status.stdout or parity.returncode or parity.stdout.split() != ["0", "0"]:
        raise Generation4TerminalError("Repository must be clean and upstream-even.")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", MODULE_NAME],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    blob = subprocess.run(
        ["git", "show", f"{commit}:{MODULE_NAME}"],
        cwd=root, capture_output=True, check=False,
    )
    module_sha256 = hashlib.sha256(blob.stdout).hexdigest()
    if (
        not commit
        or ancestor.returncode
        or blob.returncode
        or module_sha256 != sha256_file(Path(__file__).resolve())
    ):
        raise Generation4TerminalError("Committed module authority drifted.")
    return {
        "commit": commit,
        "module_sha256": module_sha256,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _g2_preview() -> dict[str, Any]:
    paths = g2._paths(g2.DEFAULT_RUNTIME_ROOT, G2_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        sha256_file(paths["manifest"]) != G2_MANIFEST_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G2_PREVIEW_SHA256
        or preview.get("status") != "immutable_pre_model_authority"
        or preview.get("action_vector", {}).get("run_g3_blind_preparation_and_context_baseline") is not True
        or preview.get("did_reveal_gold_to_prediction_workers") is not False
        or preview.get("did_load_or_run_models") is not False
    ):
        raise Generation4TerminalError("Frozen G2 authority drifted.")
    return dict(preview)


def _g3_execution_g2_preview(current: Mapping[str, Any]) -> dict[str, Any]:
    paths = g2._paths(g2.DEFAULT_RUNTIME_ROOT, G3_EXECUTION_G2_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if (
        sha256_file(paths["manifest"]) != G3_EXECUTION_G2_MANIFEST_SHA256
        or not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G3_EXECUTION_G2_PREVIEW_SHA256
    ):
        raise Generation4TerminalError("G3 execution authority drifted.")
    old_semantics = {
        key: value
        for key, value in preview.items()
        if key not in {"repository_authority", "content_sha256"}
    }
    current_semantics = {
        key: value
        for key, value in current.items()
        if key not in {"repository_authority", "content_sha256"}
    }
    if old_semantics != current_semantics:
        raise Generation4TerminalError("Renewed G2 semantics differ from G3 execution authority.")
    return dict(preview)


def _failed_member(g2_preview: Mapping[str, Any]) -> dict[str, Any]:
    members = g2_preview.get("private_evidence", {}).get("cohort_membership")
    if not isinstance(members, list) or len(members) != 7:
        raise Generation4TerminalError("Frozen cohort membership is unavailable.")
    selected = [
        dict(item)
        for item in members
        if isinstance(item, Mapping)
        and str(item.get("source_sha256") or "").startswith(FAILED_SOURCE_SHA256)
    ]
    if len(selected) != 1:
        raise Generation4TerminalError("Failed source is not uniquely frozen in cohort.")
    g1a_paths = cohort._paths(cohort.DEFAULT_RUNTIME_ROOT, g2.G1A_PREVIEW_SHA256)
    g1a = _read_json(g1a_paths["manifest"])["preview"]
    row = next(
        (
            dict(item)
            for item in g1a["private_evidence"]["exact_transcript_rows"]
            if item.get("source_sha256") == selected[0]["source_sha256"]
        ),
        None,
    )
    if row is None:
        raise Generation4TerminalError("Failed source path lineage is unavailable.")
    return {**selected[0], "source_path": row["source_path"]}


def _failure_evidence() -> dict[str, Any]:
    frozen = _g2_preview()
    execution_g2 = _g3_execution_g2_preview(frozen)
    member = _failed_member(frozen)
    source = Path(str(member["source_path"]))
    plan, paths = p1._build_plan(
        source,
        runtime_root=P1_RUNTIME_ROOT,
        source_blob_id="g4-source-" + str(member["source_sha256"])[:24],
        expected_source_sha256=str(member["source_sha256"]),
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=str(execution_g2["content_sha256"]),
    )
    require_private_file(paths["dry_run"], paths["root"])
    if paths["manifest"].exists() or paths["apply_receipt"].exists():
        raise Generation4TerminalError("Failed P1 case unexpectedly became active.")
    with tempfile.TemporaryDirectory(prefix="generation4-p1-replay-") as directory:
        output = Path(directory) / "decoded.wav"
        p1._decode(source, output, plan["recipe"])
        metrics = p1._pcm_metrics(output)
    source_duration = float(plan["source"]["probe"]["duration_seconds"])
    decoded_duration = float(metrics["duration_seconds"])
    drift = abs(decoded_duration - source_duration)
    tolerance = float(plan["recipe"]["parameters"]["duration_tolerance_seconds"])
    if drift <= tolerance:
        raise Generation4TerminalError("Frozen P1 duration failure no longer reproduces.")
    completed = []
    root = P2_RUNTIME_ROOT.expanduser().absolute()
    for comparison_path in sorted(root.glob("runs/speech-prep-*/comparison.json")):
        require_private_file(comparison_path, root)
        comparison = _read_json(comparison_path)
        methods = comparison.get("method_results")
        if (
            comparison.get("status") == "success"
            and isinstance(methods, list)
            and len(methods) == 5
            and all(isinstance(item, Mapping) and item.get("status") == "success" for item in methods)
        ):
            completed.append(
                {
                    "run_id": comparison["run_id"],
                    "comparison_sha256": sha256_file(comparison_path),
                    "method_count": 5,
                }
            )
    if len(completed) != 3:
        raise Generation4TerminalError("Completed G3 preparation prefix drifted.")
    return {
        "g2_preview_sha256": G2_PREVIEW_SHA256,
        "g2_manifest_sha256": G2_MANIFEST_SHA256,
        "g3_execution_g2_preview_sha256": G3_EXECUTION_G2_PREVIEW_SHA256,
        "g3_execution_g2_manifest_sha256": G3_EXECUTION_G2_MANIFEST_SHA256,
        "failed_source_sha256": member["source_sha256"],
        "failed_source_path": str(source),
        "failed_p1_run_id": plan["run_id"],
        "failed_p1_dry_run_sha256": sha256_file(paths["dry_run"]),
        "source_duration_seconds": source_duration,
        "decoded_duration_seconds": decoded_duration,
        "duration_drift_seconds": drift,
        "frozen_tolerance_seconds": tolerance,
        "completed_p1_p2_case_count": 3,
        "failed_case_count": 1,
        "not_attempted_after_stop_count": 3,
        "completed_p2": completed,
    }


def preview_generation4_terminal(
    *, repository_authority: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    evidence = _failure_evidence()
    actions = {
        "reveal_gold": False,
        "send_context_or_augmented_prediction_turn": False,
        "load_or_run_biometric_models": False,
        "run_predictions": False,
        "score": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
        "create_successor_plan_without_fresh_authority": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "terminal_decision",
        "terminal_decision": "stop",
        "terminal_stage": "G3_blind_preparation",
        "reason_code": "p1_duration_drift_exceeds_frozen_tolerance",
        "policy_precedence": 1,
        "policy_basis": "safety_invalid_preparation_and_exhausted_frozen_attempt",
        "g2_preview_sha256": G2_PREVIEW_SHA256,
        "g2_manifest_sha256": G2_MANIFEST_SHA256,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "failure_evidence_sha256": _canonical_hash(evidence),
        "completed_p1_p2_case_count": evidence["completed_p1_p2_case_count"],
        "failed_case_count": evidence["failed_case_count"],
        "not_attempted_after_stop_count": evidence["not_attempted_after_stop_count"],
        "duration_drift_seconds": evidence["duration_drift_seconds"],
        "frozen_tolerance_seconds": evidence["frozen_tolerance_seconds"],
        "action_vector": actions,
        "private_evidence": evidence,
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_private_gold": False,
        "did_reveal_gold_to_prediction_workers": False,
        "did_send_prediction_turn": False,
        "did_load_or_run_biometric_models": False,
        "did_score": False,
        "did_mutate": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in preview.items()
        if key not in {"private_evidence", "repository_authority"}
    }
    result["schema_version"] = RECEIPT_SCHEMA
    result["contains_paths"] = False
    result["contains_private_membership"] = False
    return result


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation4-terminal-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation4_terminal(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation4_terminal()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation4TerminalError("Reviewed terminal preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation4_terminal(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation4_terminal(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    preview = preview_generation4_terminal()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation4TerminalError("Frozen terminal decision drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation4TerminalError("Frozen terminal body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
