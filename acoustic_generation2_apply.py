"""Immutable apply/replay seam for the reviewed generation-2 pre-reveal body."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import acoustic_generation2_authority as generation2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


MANIFEST_SCHEMA = "transcribe-audio.verification-generation-2-pre-reveal-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.verification-generation-2-pre-reveal-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.verification-generation-2-pre-reveal-apply-replay.v1"
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-2-pre-reveal"
)
SHA256_RE = generation2.SHA256_RE
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation2ApplyError(ValueError):
    """Raised when the reviewed generation-2 authority cannot be frozen."""


def _canonical_hash(value: Any) -> str:
    return generation2._canonical_hash(value)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Generation2ApplyError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    status = _git(["status", "--porcelain=v1", "--untracked-files=normal"])
    upstream = _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    if status or upstream.split() != ["0", "0"]:
        raise Generation2ApplyError("Repository must be clean and upstream-even.")
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "generation2_module_sha256": sha256_file(Path(generation2.__file__).resolve()),
        "apply_module_sha256": sha256_file(Path(__file__).resolve()),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _validate_repository_authority(frozen: Mapping[str, Any]) -> None:
    expected_keys = {
        "commit",
        "generation2_module_sha256",
        "apply_module_sha256",
        "clean",
        "upstream_ahead",
        "upstream_behind",
    }
    current = _repository_authority()
    frozen_commit = str(frozen.get("commit") or "")
    if (
        set(frozen) != expected_keys
        or frozen.get("clean") is not True
        or frozen.get("upstream_ahead") != 0
        or frozen.get("upstream_behind") != 0
        or not COMMIT_RE.fullmatch(frozen_commit)
        or _git(["merge-base", "--is-ancestor", frozen_commit, current["commit"]])
    ):
        raise Generation2ApplyError("Frozen repository authority is invalid.")
    root = Path(__file__).resolve().parent
    for relpath, key, current_path in (
        ("acoustic_generation2_authority.py", "generation2_module_sha256", Path(generation2.__file__).resolve()),
        ("acoustic_generation2_apply.py", "apply_module_sha256", Path(__file__).resolve()),
    ):
        blob = subprocess.run(
            ["git", "show", f"{frozen_commit}:{relpath}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if (
            blob.returncode != 0
            or hashlib.sha256(blob.stdout).hexdigest() != frozen.get(key)
            or sha256_file(current_path) != frozen.get(key)
        ):
            raise Generation2ApplyError("Frozen generation-2 module authority drifted.")


def _paths(root: Path, authority_id: str = "") -> dict[str, Path]:
    selected_root = root.expanduser().absolute()
    selected = selected_root / "authorities" / authority_id if authority_id else selected_root / "authorities"
    return {
        "root": selected_root,
        "base": selected_root / "authorities",
        "authority": selected,
        "manifest": selected / "manifest.json",
        "receipt": selected / "apply-receipt.json",
    }


def _authority_core(
    preview: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "applied_pre_reveal",
        "preview": dict(preview),
        "repository_authority": dict(repository),
        "authorized_actions": {
            "reveal_evaluation": True,
            "prepare_evaluation_audio": True,
            "freeze_evaluation_windows": True,
            "run_models": False,
            "score_trials": False,
            "calculate_terminal_metrics": False,
            "make_terminal_decision": False,
        },
        "exact_trial_child_required_before_model_or_score_execution": True,
        "contains_private_evaluation": False,
        "contains_raw_audio": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _existing_authority_manifests(root: Path) -> list[Path]:
    paths = _paths(root)
    if not paths["base"].exists():
        return []
    if not paths["base"].is_dir() or paths["base"].is_symlink():
        raise Generation2ApplyError("Generation-2 authority root is invalid.")
    manifests = []
    for child in sorted(paths["base"].iterdir()):
        if not child.is_dir() or child.is_symlink():
            raise Generation2ApplyError("Unknown generation-2 authority entry exists.")
        entries = {item.name for item in child.iterdir()}
        if entries != {"manifest.json", "apply-receipt.json"}:
            raise Generation2ApplyError("Partial or unknown generation-2 authority exists.")
        manifests.append(child / "manifest.json")
    return manifests


def _receipt(
    preview: Mapping[str, Any],
    authority_id: str,
    content_sha256: str,
    manifest_path: Path,
    manifest_sha256: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "authority_id": authority_id,
        "authority_content_sha256": content_sha256,
        "preview_id": preview["preview_id"],
        "preview_content_sha256": preview["content_sha256"],
        "manifest_sha256": manifest_sha256 or sha256_file(manifest_path),
        "evaluation_reveal_authorized": True,
        "model_execution_authorized": False,
        "trial_scoring_authorized": False,
        "contains_private_evaluation": False,
        "contains_device_labels": False,
        "mode": "0600",
        "will_perform_external_write": False,
    }


def apply_generation2_pre_reveal(
    stored_preview: Mapping[str, Any],
    *,
    expected_preview_content_sha256: str,
    preview_inputs: Mapping[str, Any],
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Freeze the independently reviewed preview without revealing evaluation."""
    replay = generation2.replay_generation2_pre_reveal_preview(
        stored_preview, **dict(preview_inputs)
    )
    if (
        replay["content_sha256"] != expected_preview_content_sha256
        or stored_preview.get("content_sha256") != expected_preview_content_sha256
        or stored_preview.get("status") != "ready_for_independent_review"
        or stored_preview.get("production_apply_authorized") is not False
        or stored_preview.get("will_run_models") is not False
        or stored_preview.get("will_score_trials") is not False
        or stored_preview.get("will_perform_external_write") is not False
    ):
        raise Generation2ApplyError("Reviewed generation-2 preview is stale or unsafe.")
    repository = _repository_authority()
    core = _authority_core(stored_preview, repository)
    content_sha256 = _canonical_hash(core)
    authority_id = f"generation-2-pre-reveal-authority-{content_sha256[:24]}"
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, authority_id)
    existing = _existing_authority_manifests(runtime_root or DEFAULT_RUNTIME_ROOT)
    if len(existing) > 1:
        raise Generation2ApplyError("Multiple generation-2 pre-reveal authorities exist.")
    if existing:
        return replay_generation2_pre_reveal(
            existing[0], preview_inputs=preview_inputs, runtime_root=runtime_root
        )
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_generation2_pre_reveal(
            paths["manifest"], preview_inputs=preview_inputs, runtime_root=runtime_root
        )
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise Generation2ApplyError("Partial generation-2 authority exists.")
    ensure_private_tree(paths["root"], paths["authority"])
    write_immutable_private_json(paths["manifest"], {**core, "authority_id": authority_id, "content_sha256": content_sha256})
    receipt = _receipt(stored_preview, authority_id, content_sha256, paths["manifest"])
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "receipt_path": str(paths["receipt"]), "idempotent": False}


def replay_generation2_pre_reveal(
    manifest_path: Path,
    *,
    preview_inputs: Mapping[str, Any],
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay the exact applied authority without reveal, models, scores, or writes."""
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    selected = manifest_path.expanduser().resolve(strict=True)
    manifest, manifest_sha256 = _private_snapshot(selected, root)
    repository = manifest.get("repository_authority")
    preview = manifest.get("preview")
    if not isinstance(repository, Mapping) or not isinstance(preview, Mapping):
        raise Generation2ApplyError("Generation-2 authority body is incomplete.")
    _validate_repository_authority(repository)
    generation2.replay_generation2_pre_reveal_preview(preview, **dict(preview_inputs))
    core = _authority_core(preview, repository)
    content_sha256 = _canonical_hash(core)
    authority_id = f"generation-2-pre-reveal-authority-{content_sha256[:24]}"
    expected_manifest = {
        **core,
        "authority_id": authority_id,
        "content_sha256": content_sha256,
    }
    if (
        manifest != expected_manifest
        or selected.parent.name != authority_id
        or selected != _paths(root, authority_id)["manifest"]
    ):
        raise Generation2ApplyError("Generation-2 authority full-body replay mismatch.")
    receipt_path = selected.parent / "apply-receipt.json"
    receipt, _ = _private_snapshot(receipt_path, root)
    expected_receipt = _receipt(
        preview, authority_id, content_sha256, selected, manifest_sha256
    )
    if receipt != expected_receipt:
        raise Generation2ApplyError("Generation-2 authority receipt mismatch.")
    return {
        "schema_version": REPLAY_SCHEMA,
        "authority_id": authority_id,
        "content_sha256": content_sha256,
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "preview_id": preview["preview_id"],
        "preview_content_sha256": preview["content_sha256"],
        "evaluation_reveal_authorized": True,
        "model_execution_authorized": False,
        "trial_scoring_authorized": False,
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }


def _private_snapshot(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    selected = path.expanduser().resolve(strict=True)
    require_private_file(selected, root)
    body = selected.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise Generation2ApplyError("Private generation-2 authority is invalid JSON.") from exc
    if not isinstance(value, dict):
        raise Generation2ApplyError("Private generation-2 authority body is invalid.")
    return value, digest
