"""Plan 0055 S0 source-expanded evaluation authority without content decode."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import acoustic_generation5_recovery_authority as r0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-source-expansion-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-source-expansion-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-source-expansion-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-source-expansion-replay.v1"
PLAN_PATH = Path("docs/dev/plans/0055-2026-08-04-generation-5-source-expanded-blind-evaluation.md")
DEFAULT_PRIOR_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0055/s0")
DEFAULT_ARCHIVE_ROOT = Path("/mnt/c/Users/ecoch/Documents/Sound Recordings/Transcribed")
DEFAULT_ZOOM_SOURCE = Path(
    "/mnt/bastion-mnt/data2024/syncthing/Cloud/Google/Video/Zoom/"
    "2021-10-25 15.05.15 Eric Cochran's Personal Meeting Room 4671813693/audio_only.m4a"
)
DEFAULT_ARCHIVE_REQUIRED = Path(
    "/mnt/c/Users/ecoch/Documents/Sound Recordings/Transcribed/"
    "2025-09-25 Agritalk Radio Show Chis and Eric My recording 31.m4a"
)
ZOOM_SHA256 = "06ff1b6b21736d3bb47c2d2789f30c5ae0e9c9998788f93d72cc54ce46840b12"
ARCHIVE_REQUIRED_SHA256 = "cc0cd45469d3de0d9e336dbdd4abba2458bd555916328f06115008aed1ff913b"
MAX_ADDITIONAL = 10
MINIMUM_DURATION_SECONDS = 60.0
MODULE_NAME = Path(__file__).name
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation5SourceExpansionError(ValueError):
    """Raised when Plan 0055 S0 cannot remain deterministic and sealed."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5SourceExpansionError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5SourceExpansionError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5SourceExpansionError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    module_body = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    plan_body = _git(["show", f"{commit}:{PLAN_PATH.as_posix()}"], binary=True)
    if not isinstance(module_body, bytes) or not isinstance(plan_body, bytes):
        raise Generation5SourceExpansionError("Committed authority is unavailable.")
    module_hash = hashlib.sha256(module_body).hexdigest()
    plan_hash = hashlib.sha256(plan_body).hexdigest()
    if module_hash != sha256_file(Path(__file__).resolve()) or plan_hash != sha256_file(PLAN_PATH):
        raise Generation5SourceExpansionError("Committed authority drifted.")
    return {
        "commit": commit,
        "module_sha256": module_hash,
        "plan_sha256": plan_hash,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _exclusion_union(prior_root: Path) -> dict[str, Any]:
    root = prior_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Generation5SourceExpansionError("Prior evidence root is invalid.")
    hashes: set[str] = set()
    evidence_files: list[str] = []
    parse_modes: dict[str, int] = {}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or (relative.parts and relative.parts[0] == "plan-0055"):
            continue
        found, mode = r0._evidence_hashes(path)  # Reuse the accepted strict JSON/hash parser.
        hashes.update(found)
        evidence_files.append(sha256_file(path))
        parse_modes[mode] = parse_modes.get(mode, 0) + 1
    if not hashes or not evidence_files:
        raise Generation5SourceExpansionError("Prior exclusion evidence is empty.")
    return {
        "hashes": hashes,
        "json_file_count": len(evidence_files),
        "json_file_set_sha256": _canonical_hash(sorted(evidence_files)),
        "excluded_hash_count": len(hashes),
        "excluded_hash_set_sha256": _canonical_hash(sorted(hashes)),
        "parse_mode_counts": parse_modes,
    }


def _probe(path: Path, ffprobe_path: str) -> dict[str, Any]:
    try:
        value = r0._probe(path, ffprobe_path)
    except r0.Generation5RecoveryAuthorityError as exc:
        raise Generation5SourceExpansionError(str(exc)) from exc
    if value["duration_seconds"] < MINIMUM_DURATION_SECONDS:
        raise Generation5SourceExpansionError("duration_below_minimum")
    return value


def _required_row(
    path: Path, expected_hash: str, role: str, exclusions: set[str], ffprobe_path: str,
    probe: Callable[[Path, str], dict[str, Any]],
) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    if not absolute.is_file() or absolute.is_symlink():
        raise Generation5SourceExpansionError(f"{role}_missing")
    digest = sha256_file(absolute)
    if digest != expected_hash:
        raise Generation5SourceExpansionError(f"{role}_hash_drift")
    if digest in exclusions:
        raise Generation5SourceExpansionError(f"{role}_prior_evidence_overlap")
    return {"role": role, "path": str(absolute), "source_sha256": digest, "probe": probe(absolute, ffprobe_path)}


def _additional_rows(
    archive_root: Path, required_hashes: set[str], exclusions: set[str], ffprobe_path: str,
    probe: Callable[[Path, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = archive_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Generation5SourceExpansionError("Archive root is invalid.")
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    candidates = sorted(
        path for path in root.rglob("*")
        if path.suffix.lower() in {".m4a", ".mp4", ".wav"} and path.is_file() and not path.is_symlink()
    )
    for path in candidates:
        digest = sha256_file(path)
        if digest in required_hashes:
            counts["required_source"] = counts.get("required_source", 0) + 1
            continue
        if digest in exclusions:
            counts["prior_evidence_overlap"] = counts.get("prior_evidence_overlap", 0) + 1
            continue
        try:
            media_probe = probe(path, ffprobe_path)
        except Generation5SourceExpansionError as exc:
            reason = str(exc)
            counts[reason] = counts.get(reason, 0) + 1
            continue
        selected.append({
            "role": "additional_candidate",
            "archive_relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "source_sha256": digest,
            "probe": media_probe,
        })
        if len(selected) == MAX_ADDITIONAL:
            break
    if len(selected) < 5:
        raise Generation5SourceExpansionError("insufficient_additional_candidates")
    return selected, counts


def preview_generation5_source_expansion(
    *, zoom_source: Path = DEFAULT_ZOOM_SOURCE,
    archive_required: Path = DEFAULT_ARCHIVE_REQUIRED,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
    ffprobe_path: str | None = None,
    probe: Callable[[Path, str], dict[str, Any]] = _probe,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ffprobe = ffprobe_path or shutil.which("ffprobe")
    if not ffprobe:
        raise Generation5SourceExpansionError("ffprobe_unavailable")
    exclusion = _exclusion_union(prior_root)
    required = [
        _required_row(zoom_source, ZOOM_SHA256, "required_zoom", exclusion["hashes"], ffprobe, probe),
        _required_row(archive_required, ARCHIVE_REQUIRED_SHA256, "required_archive", exclusion["hashes"], ffprobe, probe),
    ]
    additional, rejection_counts = _additional_rows(
        archive_root, {row["source_sha256"] for row in required}, exclusion["hashes"], ffprobe, probe,
    )
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j0_review",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "required_source_count": 2,
        "additional_candidate_count": len(additional),
        "candidate_count": len(required) + len(additional),
        "exclusion_summary": {key: value for key, value in exclusion.items() if key != "hashes"},
        "rejection_reason_counts": rejection_counts,
        "required_source_set_sha256": _canonical_hash(sorted(row["source_sha256"] for row in required)),
        "additional_source_set_sha256": _canonical_hash(sorted(row["source_sha256"] for row in additional)),
        "ordered_candidate_set_sha256": _canonical_hash([row["source_sha256"] for row in [*required, *additional]]),
        "private_evidence": {"required_sources": required, "additional_candidates": additional},
        "action_vector": {
            "submit_exact_source_authority_to_j0": True,
            "copy_required_zoom_to_private_runtime": False,
            "decode_audio": False,
            "transcribe_or_diarize": False,
            "access_or_freeze_gold": False,
            "run_models_or_predictions": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_paths": True,
        "contains_private_membership": True,
        "contains_identity_names_or_aliases": False,
        "did_decode_audio": False,
        "did_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "required_source_count": preview["required_source_count"],
        "additional_candidate_count": preview["additional_candidate_count"],
        "candidate_count": preview["candidate_count"],
        "exclusion_summary": preview["exclusion_summary"],
        "rejection_reason_counts": preview["rejection_reason_counts"],
        "required_source_set_sha256": preview["required_source_set_sha256"],
        "additional_source_set_sha256": preview["additional_source_set_sha256"],
        "ordered_candidate_set_sha256": preview["ordered_candidate_set_sha256"],
        "action_vector": preview["action_vector"],
        "did_decode_audio": False,
        "did_run_models": False,
    }


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-source-expansion-{content_sha256[:24]}"
    return {
        "root": root, "run": run, "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json", "zoom_copy": run / "required-zoom-audio.m4a",
    }


def apply_generation5_source_expansion(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = dict(reviewed_preview)
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    if preview.get("content_sha256") != expected_content_sha256 or _canonical_hash(core) != expected_content_sha256:
        raise Generation5SourceExpansionError("Reviewed S0 preview is stale.")
    if preview.get("repository_authority") != _repository_authority():
        raise Generation5SourceExpansionError("Reviewed repository authority is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_source_expansion(expected_content_sha256, runtime_root=runtime_root)
    private = preview.get("private_evidence")
    required = private.get("required_sources") if isinstance(private, Mapping) else None
    if not isinstance(required, list) or len(required) != 2:
        raise Generation5SourceExpansionError("Required source authority is incomplete.")
    zoom = Path(str(required[0].get("path") or ""))
    if sha256_file(zoom) != ZOOM_SHA256:
        raise Generation5SourceExpansionError("Required Zoom source drifted.")
    ensure_private_tree(paths["root"], paths["run"])
    shutil.copy2(zoom, paths["zoom_copy"])
    if sha256_file(paths["zoom_copy"]) != ZOOM_SHA256:
        raise Generation5SourceExpansionError("Private Zoom copy drifted.")
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]),
        "private_zoom_copy_sha256": ZOOM_SHA256, "mode": "0600",
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_source_expansion(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root, expected_content_sha256)
    for key in ("manifest", "receipt", "zoom_copy"):
        require_private_file(paths[key], paths["root"])
    manifest = json.loads(paths["manifest"].read_text())
    receipt = json.loads(paths["receipt"].read_text())
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping):
        raise Generation5SourceExpansionError("Recorded S0 preview is missing.")
    core = {key: value for key, value in preview.items() if key != "content_sha256"}
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {
        **_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]),
        "private_zoom_copy_sha256": ZOOM_SHA256, "mode": "0600",
    }
    repository = preview.get("repository_authority")
    repository = dict(repository) if isinstance(repository, Mapping) else {}
    commit = str(repository.get("commit") or "")
    module_hash = str(repository.get("module_sha256") or "")
    module = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True) if SHA256_RE.fullmatch(module_hash) else b""
    if (
        _canonical_hash(core) != expected_content_sha256
        or preview.get("content_sha256") != expected_content_sha256
        or manifest != expected_manifest or receipt != expected_receipt
        or sha256_file(paths["zoom_copy"]) != ZOOM_SHA256
        or not isinstance(module, bytes) or hashlib.sha256(module).hexdigest() != repository.get("module_sha256")
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != ""
    ):
        raise Generation5SourceExpansionError("S0 authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
