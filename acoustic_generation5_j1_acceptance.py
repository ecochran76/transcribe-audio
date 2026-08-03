"""Freeze the independent Plan 0053 J1 acceptance as G2-only authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_development as g1
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-j1-acceptance-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-j1-acceptance-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-j1-acceptance-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-j1-acceptance-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0053/j1")
G1_PREVIEW_SHA256 = "3e66a93feb5a826680025135c6b60e0e541ed724a639146c1df126f403242919"
G1_MANIFEST_SHA256 = "4d5ad0f08ba14ca20bd489530cc8919ab0d2defa9bb0dfaecfba5a761a303f0f"
G1_CONTRACT_SHA256 = "2b3c988ffedebb8a0070499cc779795bea8bd44236b1234128e18859a6d8b7e9"
REVIEWER_HANDLE = "/root/g5_j1_rereview"
MODULE_NAME = Path(__file__).name


class Generation5J1AcceptanceError(ValueError):
    """Raised when the independent J1 acceptance cannot be bound exactly."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5J1AcceptanceError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5J1AcceptanceError("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5J1AcceptanceError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5J1AcceptanceError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5J1AcceptanceError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation5J1AcceptanceError("Repository commit is invalid.")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5J1AcceptanceError("Committed J1 module drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _g1_authority() -> dict[str, Any]:
    paths = g1._paths(g1.DEFAULT_RUNTIME_ROOT, G1_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    require_private_file(paths["receipt"], paths["root"].expanduser().absolute())
    if sha256_file(paths["manifest"]) != G1_MANIFEST_SHA256:
        raise Generation5J1AcceptanceError("G1 manifest drifted.")
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G1_PREVIEW_SHA256
        or preview.get("contract_sha256") != G1_CONTRACT_SHA256
        or preview.get("did_measure_holdout") is not False
        or receipt.get("content_sha256") != G1_PREVIEW_SHA256
        or receipt.get("manifest_sha256") != G1_MANIFEST_SHA256
    ):
        raise Generation5J1AcceptanceError("G1 authority drifted.")
    return dict(preview)


def preview_generation5_j1_acceptance(
    *,
    g1_preview: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = dict(g1_preview or _g1_authority())
    if (
        parent.get("content_sha256") != G1_PREVIEW_SHA256
        or parent.get("contract_sha256") != G1_CONTRACT_SHA256
        or parent.get("status") != "ready_for_independent_j1_review"
        or parent.get("did_measure_holdout") is not False
    ):
        raise Generation5J1AcceptanceError("Reviewed G1 packet is invalid.")
    findings = {
        "source_derivation_and_holdout_isolation": True,
        "causal_diagnosis_supported": True,
        "aac_priming_and_discard_handling_supported": True,
        "one_sample_resampling_bound_supported": True,
        "exact_missing_interval_boundary_rejected": True,
        "non_circular_content_oracle_supported": True,
        "compressed_packet_removal_rejected": True,
        "wrong_stream_runs_through_validator": True,
        "all_eleven_development_adversaries_rejected": True,
        "tool_and_construction_evidence_bound": True,
        "case_fitted_constant_used": False,
        "portable_privacy_passed": True,
        "independent_replay_passed": True,
        "holdout_touched_during_review": False,
    }
    actions = {
        "run_g2_positive_holdout_once": True,
        "instantiate_g2_heldout_negative_family_once": True,
        "enumerate_generation5_candidates": False,
        "reveal_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "accepted_for_g2_only",
        "review_decision": "PASS",
        "reviewer_handle": REVIEWER_HANDLE,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "g1_preview_sha256": G1_PREVIEW_SHA256,
        "g1_manifest_sha256": G1_MANIFEST_SHA256,
        "contract_sha256": G1_CONTRACT_SHA256,
        "findings": findings,
        "findings_sha256": _canonical_hash(findings),
        "action_vector": actions,
        "did_measure_holdout": False,
        "did_instantiate_heldout_negative_family": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in preview.items() if key != "repository_authority"}
    value["schema_version"] = RECEIPT_SCHEMA
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-j1-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation5_j1_acceptance(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_j1_acceptance()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5J1AcceptanceError("Reviewed J1 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_j1_acceptance(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        **_portable(preview),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "mode": "0600",
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_j1_acceptance(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_j1_acceptance()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation5J1AcceptanceError("J1 authority drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {
        **_portable(preview),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "mode": "0600",
    }
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5J1AcceptanceError("J1 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
