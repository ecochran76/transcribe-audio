"""Generation-3 reveal authority and structural denominator preflight."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation3_pre_reveal as pre_reveal
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-reveal-preview.v1"
AUTHORITY_SCHEMA = "transcribe-audio.generation3-reveal-authority.v1"
PREFLIGHT_SCHEMA = "transcribe-audio.generation3-denominator-preflight.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-reveal-preflight-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-reveal-preflight-replay.v1"
DEFAULT_RUNTIME_ROOT = pre_reveal.DEFAULT_RUNTIME_ROOT
MODULE_NAMES = (
    "acoustic_generation3_evaluation.py",
    "acoustic_generation3_pre_reveal.py",
    "acoustic_generation3_gold.py",
    "acoustic_generation3_authority.py",
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation3EvaluationError(ValueError):
    """Raised when reveal or denominator preflight cannot fail closed."""


def _canonical_hash(value: Any) -> str:
    return pre_reveal._canonical_hash(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation3EvaluationError("Evaluation JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3EvaluationError("Evaluation JSON must be an object.")
    return value


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise Generation3EvaluationError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    behind, ahead = (
        int(item)
        for item in _git(
            ["rev-list", "--left-right", "--count", "@{upstream}...HEAD"]
        ).split()
    )
    root = Path(__file__).resolve().parent
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": {name: sha256_file(root / name) for name in MODULE_NAMES},
        "clean": _git(["status", "--porcelain"]) == "",
        "upstream_ahead": ahead,
        "upstream_behind": behind,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Generation3EvaluationError("Reveal repository authority is invalid.")
    modules = value.get("module_sha256")
    commit = str(value.get("commit") or "")
    if (
        set(value)
        != {"commit", "module_sha256", "clean", "upstream_ahead", "upstream_behind"}
        or not COMMIT_RE.fullmatch(commit)
        or not isinstance(modules, Mapping)
        or set(modules) != set(MODULE_NAMES)
        or any(not SHA256_RE.fullmatch(str(item)) for item in modules.values())
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3EvaluationError("Reveal repository authority drifted.")
    root = Path(__file__).resolve().parent
    for name, digest in modules.items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{name}"], cwd=root,
            check=False, capture_output=True,
        )
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != digest
            or sha256_file(root / name) != digest
        ):
            raise Generation3EvaluationError("Reveal module authority drifted.")
    current = _repository_authority()
    if (
        current["clean"] is not True
        or current["upstream_ahead"] != 0
        or current["upstream_behind"] != 0
    ):
        raise Generation3EvaluationError("Current repository is not clean and exact.")
    return dict(value)


def _parent_context(runtime_root: Path) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    paths = sorted(root.glob("pre-reveal-authorities/*/private-manifest.json"))
    if len(paths) != 1:
        raise Generation3EvaluationError("Exactly one pre-reveal parent is required.")
    require_private_file(paths[0], root)
    manifest = _read_object(paths[0])
    preview = manifest.get("preview")
    repository = manifest.get("repository_authority")
    if not isinstance(repository, Mapping) or not isinstance(preview, Mapping):
        raise Generation3EvaluationError("Pre-reveal parent is incomplete.")
    try:
        validated_repository = pre_reveal._validate_repository_authority(repository)
    except (pre_reveal.Generation3PreRevealError, ValueError) as exc:
        raise Generation3EvaluationError("Pre-reveal parent repository drifted.") from exc
    core = pre_reveal._manifest_core(preview, validated_repository)
    content_sha = _canonical_hash(core)
    authority_id = f"generation3-pre-reveal-{content_sha[:24]}"
    expected_manifest = {
        **core, "authority_id": authority_id, "content_sha256": content_sha,
    }
    receipt_path = paths[0].parent / "receipt.json"
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = pre_reveal._receipt(
        preview, authority_id, sha256_file(paths[0])
    )
    if (
        manifest != expected_manifest
        or paths[0] != pre_reveal._paths(root, authority_id)["manifest"]
        or receipt != expected_receipt
        or receipt.get("action_vector", {}).get("reveal_evaluation") is not True
        or receipt.get("action_vector", {}).get("run_denominator_preflight") is not False
        or receipt.get("action_vector", {}).get("prepare_evaluation_audio") is not False
        or receipt.get("action_vector", {}).get("load_or_run_models") is not False
    ):
        raise Generation3EvaluationError("Parent does not authorize reveal only.")
    return {
        "manifest_path": str(paths[0]),
        "manifest_sha256": sha256_file(paths[0]),
        "authority_id": authority_id,
        "content_sha256": manifest["content_sha256"],
        "preview": dict(preview),
    }


def _preview_core(parent: Mapping[str, Any], repository: Mapping[str, Any]) -> dict[str, Any]:
    preview = parent["preview"]
    return {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_to_freeze_reveal_before_private_gold_read",
        "parent_authority_id_sha256": _canonical_hash(parent["authority_id"]),
        "parent_manifest_sha256": parent["manifest_sha256"],
        "parent_content_sha256": parent["content_sha256"],
        "cohort_manifest_sha256": preview["cohort_authority"]["manifest_sha256"],
        "gold_manifest_sha256": preview["gold_authority"]["manifest_sha256"],
        "membership_sha256": preview["cohort_authority"]["membership_sha256"],
        "profile_set_sha256": preview["recalibration_authority"]["profile_set_sha256"],
        "candidate_unit_count": len(preview["candidate_matrix"]),
        "evaluation_conversation_count": preview["cohort_authority"]["conversation_count"],
        "gold_label_count": preview["gold_authority"]["gold_label_count"],
        "maximum_windows_per_speaker_per_conversation": preview["window_policy"][
            "maximum_windows_per_speaker_per_conversation"
        ],
        "minimum_evidence_policy": dict(
            preview["terminal_decision_policy"]["minimum_evidence"]
        ),
        "repository_authority": dict(repository),
        "action_vector": {
            "freeze_reveal_authority": False,
            "reveal_private_gold": False,
            "run_denominator_preflight": False,
            "run_prediction_blind_p1_p2": False,
            "record_terminal_stop": False,
            "prepare_evaluation_audio": False,
            "measure_conditions": False,
            "freeze_evaluation_windows": False,
            "construct_exact_trial_child": False,
            "load_or_run_models": False,
            "score_evaluation_trials": False,
            "calculate_evaluation_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "did_read_private_gold": False,
        "did_read_audio": False,
        "did_load_or_run_models": False,
        "contains_private_gold": False,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False,
    }


def preview_generation3_reveal(
    *, runtime_root: Path = DEFAULT_RUNTIME_ROOT
) -> dict[str, Any]:
    """Preview reveal authority without reading the private gold body."""
    parent = _parent_context(runtime_root)
    core = _preview_core(parent, _repository_authority())
    content_sha = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-reveal-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }


def _paths(runtime_root: Path, reveal_id: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    directory = root / "evaluation-reveals" / reveal_id
    return {
        "root": root, "directory": directory,
        "authority": directory / "reveal-authority.json",
        "preflight": directory / "denominator-preflight.json",
        "receipt": directory / "receipt.json",
    }


def _authority_body(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUTHORITY_SCHEMA,
        "preview": dict(preview),
        "repository_authority": dict(preview["repository_authority"]),
        "private_gold_read_permitted_only_after_this_file_exists": True,
        "did_read_private_gold": False,
        "did_read_audio": False,
        "did_load_or_run_models": False,
    }


def _write_or_replay_authority(
    preview: Mapping[str, Any], *, runtime_root: Path
) -> tuple[dict[str, Path], str]:
    body = _authority_body(preview)
    authority_sha = _canonical_hash(body)
    reveal_id = f"generation3-reveal-{authority_sha[:24]}"
    paths = _paths(runtime_root, reveal_id)
    ensure_private_tree(paths["root"], paths["directory"])
    if paths["authority"].exists():
        require_private_file(paths["authority"], paths["root"])
        if _read_object(paths["authority"]) != body:
            raise Generation3EvaluationError("Reveal authority conflicts.")
    else:
        write_immutable_private_json(paths["authority"], body)
    if _canonical_hash(_read_object(paths["authority"])) != authority_sha:
        raise Generation3EvaluationError("Reveal authority identity changed.")
    return paths, authority_sha


def _gold_preview(parent: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    paths = sorted(root.glob("gold-authorities/*/private-manifest.json"))
    if len(paths) != 1:
        raise Generation3EvaluationError("Exactly one gold authority is required.")
    require_private_file(paths[0], root)
    manifest = _read_object(paths[0])
    preview = manifest.get("preview")
    expected = parent["preview"]["gold_authority"]
    if (
        not isinstance(preview, Mapping)
        or sha256_file(paths[0]) != expected["manifest_sha256"]
        or preview.get("membership_sha256")
        != parent["preview"]["cohort_authority"]["membership_sha256"]
        or preview.get("gold_label_count") != expected["gold_label_count"]
        or not isinstance(preview.get("gold"), list)
    ):
        raise Generation3EvaluationError("Private gold authority drifted.")
    return dict(preview)


def _denominator_preflight(
    parent: Mapping[str, Any], gold: Mapping[str, Any], authority_sha: str
) -> dict[str, Any]:
    preview = parent["preview"]
    profiles = preview["profiles"]
    profile_subjects = {str(item["person_ref_id"]) for item in profiles}
    if len(profile_subjects) != 2:
        raise Generation3EvaluationError("Exactly two profile subjects are required.")
    enrolled = [item for item in gold["gold"] if item.get("outcome") == "enrolled"]
    open_set = [item for item in gold["gold"] if item.get("outcome") == "open_set"]
    excluded = [
        item for item in gold["gold"] if item.get("outcome") in {"mixed", "unknown"}
    ]
    if (
        len(enrolled) != 10
        or len(open_set) != 10
        or len(excluded) != 8
        or {str(item.get("subject_id")) for item in enrolled} != profile_subjects
        or any(str(item.get("subject_id")) in profile_subjects for item in open_set)
    ):
        raise Generation3EvaluationError("Revealed gold population drifted.")
    cap = int(preview["window_policy"]["maximum_windows_per_speaker_per_conversation"])
    minimum = preview["terminal_decision_policy"]["minimum_evidence"]
    maxima = {
        "genuine": len(enrolled) * cap,
        "impostor": len(enrolled) * cap,
        "open_set": len(open_set) * cap * len(profile_subjects),
    }
    required = {
        "genuine": minimum["genuine_trials_per_model_method_unit"],
        "impostor": minimum["impostor_trials_per_model_method_unit"],
        "open_set": minimum["open_set_trials_per_model_method_unit"],
    }
    status = "pass" if all(maxima[key] >= required[key] for key in required) else "stop"
    units = [
        {
            "candidate_id": item["candidate_id"],
            "method_id": item["method_id"],
            "maximum_genuine_trials": maxima["genuine"],
            "maximum_impostor_trials": maxima["impostor"],
            "maximum_open_set_trials": maxima["open_set"],
            "required_genuine_trials": required["genuine"],
            "required_impostor_trials": required["impostor"],
            "required_open_set_trials": required["open_set"],
            "status": status,
        }
        for item in preview["candidate_matrix"]
    ]
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": status,
        "reason_code": None if status == "pass" else "trial_class_denominator_below_policy",
        "reveal_authority_sha256": authority_sha,
        "parent_manifest_sha256": parent["manifest_sha256"],
        "membership_sha256": preview["cohort_authority"]["membership_sha256"],
        "gold_manifest_sha256": preview["gold_authority"]["manifest_sha256"],
        "gold_label_count": len(gold["gold"]),
        "enrolled_label_instance_count": len(enrolled),
        "open_set_label_instance_count": len(open_set),
        "excluded_label_instance_count": len(excluded),
        "profile_subject_count": len(profile_subjects),
        "maximum_windows_per_label_conversation": cap,
        "unit_count": len(units),
        "units": units,
        "calculation": {
            "genuine": "enrolled_label_instances_times_window_cap",
            "impostor": "enrolled_label_instances_times_other_profile_times_window_cap",
            "open_set": "open_set_label_instances_times_profile_count_times_window_cap",
        },
        "did_reveal_private_gold": True,
        "did_read_audio": False,
        "did_prepare_audio": False,
        "did_load_or_run_models": False,
        "did_score_trials": False,
        "contains_private_gold": True,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
    }


def _receipt(
    preview: Mapping[str, Any], authority_sha: str, preflight_sha: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    actions = dict(preview["action_vector"])
    actions["freeze_reveal_authority"] = True
    actions["reveal_private_gold"] = True
    actions["run_denominator_preflight"] = True
    if preflight["status"] == "pass":
        actions["run_prediction_blind_p1_p2"] = True
        status = "preflight_pass_prediction_blind_preparation_authorized"
    else:
        actions["record_terminal_stop"] = True
        status = "preflight_stop_terminal_stop_authorized"
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "reason_code": preflight["reason_code"],
        "reveal_authority_sha256": authority_sha,
        "preflight_sha256": preflight_sha,
        "parent_manifest_sha256": preview["parent_manifest_sha256"],
        "gold_manifest_sha256": preview["gold_manifest_sha256"],
        "unit_count": preflight["unit_count"],
        "enrolled_label_instance_count": preflight["enrolled_label_instance_count"],
        "open_set_label_instance_count": preflight["open_set_label_instance_count"],
        "excluded_label_instance_count": preflight["excluded_label_instance_count"],
        "action_vector": actions,
        "contains_private_gold": False,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "mode": "0600",
    }


def apply_generation3_reveal_and_preflight(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Freeze reveal authority, then read gold and run structural preflight."""
    preview = preview_generation3_reveal(runtime_root=runtime_root)
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
        or any(preview["action_vector"].values())
        or preview["did_read_private_gold"] is not False
    ):
        raise Generation3EvaluationError("Reviewed reveal preview is stale.")
    _validate_repository_authority(preview["repository_authority"])
    paths, authority_sha = _write_or_replay_authority(
        preview, runtime_root=runtime_root
    )
    if paths["preflight"].exists():
        return replay_generation3_reveal_and_preflight(
            authority_sha, runtime_root=runtime_root
        )
    parent = _parent_context(runtime_root)
    gold = _gold_preview(parent, runtime_root)
    preflight = _denominator_preflight(parent, gold, authority_sha)
    write_immutable_private_json(paths["preflight"], preflight)
    receipt = _receipt(
        preview, authority_sha, sha256_file(paths["preflight"]), preflight
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_reveal_authority_path": str(paths["authority"]),
        "private_preflight_path": str(paths["preflight"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_reveal_and_preflight(
    reveal_authority_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay revealed gold and preflight structurally without audio or models."""
    if not SHA256_RE.fullmatch(str(reveal_authority_sha256)):
        raise Generation3EvaluationError("Reveal authority hash is invalid.")
    reveal_id = f"generation3-reveal-{reveal_authority_sha256[:24]}"
    paths = _paths(runtime_root, reveal_id)
    require_private_file(paths["authority"], paths["root"])
    authority = _read_object(paths["authority"])
    repository = _validate_repository_authority(authority.get("repository_authority"))
    parent = _parent_context(runtime_root)
    core = _preview_core(parent, repository)
    content_sha = _canonical_hash(core)
    preview = {
        **core,
        "preview_id": f"generation3-reveal-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }
    if (
        _canonical_hash(authority) != reveal_authority_sha256
        or authority != _authority_body(preview)
    ):
        raise Generation3EvaluationError("Reveal authority drifted.")
    gold = _gold_preview(parent, runtime_root)
    expected_preflight = _denominator_preflight(parent, gold, reveal_authority_sha256)
    require_private_file(paths["preflight"], paths["root"])
    preflight = _read_object(paths["preflight"])
    if preflight != expected_preflight:
        raise Generation3EvaluationError("Denominator preflight drifted.")
    require_private_file(paths["receipt"], paths["root"])
    receipt = _read_object(paths["receipt"])
    expected_receipt = _receipt(
        preview, reveal_authority_sha256, sha256_file(paths["preflight"]), preflight
    )
    if receipt != expected_receipt:
        raise Generation3EvaluationError("Reveal receipt drifted.")
    return {
        **receipt,
        "private_reveal_authority_path": str(paths["authority"]),
        "private_preflight_path": str(paths["preflight"]),
        "private_receipt_path": str(paths["receipt"]),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
        "replay_mode": "structural_without_audio_or_model_execution",
    }
