"""Record and replay the Plan 0053 independent J2 terminal STOP."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation5_holdout as g2
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-j2-stop-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-j2-stop-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-j2-stop-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-j2-stop-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0053/j2-stop")
G2_PREVIEW_SHA256 = "fc0d0dca9eec248df2e8dccdd262a3062ca2e5deca03aaa99799e4253e647c83"
G2_MANIFEST_SHA256 = "c59771c6d6054be0384956beaa7a7908dca1fe6cbdffb93bf8fe2d00469c69f2"
REVIEWER_HANDLE = "/root/g5_j2_audit"
MODULE_NAME = Path(__file__).name


class Generation5J2StopError(ValueError):
    """Raised when the J2 STOP cannot be recorded or replayed exactly."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5J2StopError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5J2StopError("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5J2StopError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5J2StopError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5J2StopError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation5J2StopError("Repository commit is invalid.")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != sha256_file(Path(__file__).resolve()):
        raise Generation5J2StopError("Committed terminal module drifted.")
    return {
        "commit": commit,
        "module_sha256": hashlib.sha256(body).hexdigest(),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _g2_preview() -> dict[str, Any]:
    paths = g2._paths(g2.DEFAULT_RUNTIME_ROOT, G2_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    require_private_file(paths["receipt"], paths["root"].expanduser().absolute())
    if sha256_file(paths["manifest"]) != G2_MANIFEST_SHA256:
        raise Generation5J2StopError("G2 manifest drifted.")
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != G2_PREVIEW_SHA256
        or preview.get("positive_holdout_count") != 7
        or preview.get("positive_holdout_pass_count") != 7
        or (preview.get("heldout_adversarial") or {}).get("case_count") != 11
        or receipt.get("manifest_sha256") != G2_MANIFEST_SHA256
    ):
        raise Generation5J2StopError("G2 authority drifted.")
    return dict(preview)


def preview_generation5_j2_stop(
    *,
    g2_preview: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parent = dict(g2_preview or _g2_preview())
    if parent.get("content_sha256") != G2_PREVIEW_SHA256:
        raise Generation5J2StopError("Reviewed G2 packet drifted.")
    finding = {
        "reason_code": "heldout_expected_reason_is_circular",
        "fault_case": "corrupt_source_tail",
        "positive_holdout_reproduced": 7,
        "fixed_negative_cases_reproduced": 10,
        "required_negative_cases": 11,
        "invalid_expression": "expected_reason_was_observed_reason",
        "same_holdout_rework_authorized": False,
    }
    actions = {
        "retry_g2": False,
        "enumerate_generation5_candidates": False,
        "access_gold": False,
        "run_predictions_or_models": False,
        "score": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "terminal_stop",
        "terminal_decision": "stop",
        "terminal_stage": "J2_independent_validation_audit",
        "reviewer_handle": REVIEWER_HANDLE,
        "repository_authority": dict(repository_authority or _repository_authority()),
        "g2_preview_sha256": G2_PREVIEW_SHA256,
        "g2_manifest_sha256": G2_MANIFEST_SHA256,
        "finding": finding,
        "finding_sha256": _canonical_hash(finding),
        "action_vector": actions,
        "did_enumerate_generation5_candidates": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
        "did_run_predictions": False,
        "did_score": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in preview.items() if key != "repository_authority"}
    value["schema_version"] = RECEIPT_SCHEMA
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-j2-stop-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation5_j2_stop(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_j2_stop()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5J2StopError("Reviewed J2 STOP preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_j2_stop(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_j2_stop(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5J2StopError("J2 STOP preview is missing.")
    preview = dict(preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    repository = preview.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation5J2StopError("Recorded repository authority is missing.")
    commit = str(repository.get("commit") or "")
    module_sha256 = str(repository.get("module_sha256") or "")
    body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if re.fullmatch(r"[a-f0-9]{40}", commit) else b""
    _g2_preview()
    if (
        preview.get("content_sha256") != expected_content_sha256
        or _canonical_hash(core) != expected_content_sha256
        or not isinstance(body, bytes)
        or hashlib.sha256(body).hexdigest() != module_sha256
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5J2StopError("J2 STOP authority drifted.")
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"
    }
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5J2StopError("J2 STOP body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
