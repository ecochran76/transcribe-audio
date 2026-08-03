"""Generation-4 read-only media qualification authority."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation4-media-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation4-media-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation4-media-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation4-media-replay.v1"
PORTABLE_SCHEMA = "transcribe-audio.generation4-media-portable.v1"
DEFAULT_SOURCE_ROOT = Path("/mnt/c/Users/ecoch/Documents/Sound Recordings")
DEFAULT_PRIOR_ROOT = Path("~/.local/state/transcribe-audio/plan-0037")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0037/generation-4")
MODULE_NAME = "acoustic_generation4_media.py"
MAX_CANDIDATES = 12
MIN_QUALIFIED = 7
MIN_DURATION_SECONDS = 60.0
MAX_DURATION_DRIFT_SECONDS = 0.05
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
POST_QUALIFICATION_ACTIONS = (
    "build_generation4_cohort_preview", "freeze_generation4_cohort", "freeze_generation4_gold",
    "prepare_audio", "load_or_run_models", "construct_exact_trials",
    "score_trials", "calculate_metrics", "make_terminal_selection",
    "mutate_profiles_or_references", "enable_default_integration",
    "run_historical_reprocessing",
)


class Generation4MediaError(ValueError):
    """Raised when Generation-4 media qualification cannot fail closed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=not binary,
    )
    if result.returncode:
        raise Generation4MediaError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation4MediaError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])).split() != ["0", "0"]:
        raise Generation4MediaError("Repository must be upstream-even.")
    commit = str(_git(["log", "-1", "--format=%H", "--", MODULE_NAME]))
    if not COMMIT_RE.fullmatch(commit):
        raise Generation4MediaError("Qualification module is not committed.")
    digest = sha256_file(Path(__file__).resolve())
    blob = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(blob, bytes) or hashlib.sha256(blob).hexdigest() != digest:
        raise Generation4MediaError("Qualification module authority drifted.")
    return {
        "commit": commit, "module_name": MODULE_NAME, "module_sha256": digest,
        "clean": True, "upstream_ahead": 0, "upstream_behind": 0,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    current = _repository_authority()
    if (
        not isinstance(value, Mapping) or set(value) != set(current)
        or value.get("module_name") != MODULE_NAME
        or not COMMIT_RE.fullmatch(str(value.get("commit") or ""))
        or not SHA256_RE.fullmatch(str(value.get("module_sha256") or ""))
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0 or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", str(value["commit"]), "HEAD"]) != ""
        or value != current
    ):
        raise Generation4MediaError("Frozen repository authority drifted.")
    return dict(value)


def _tool(path_or_name: str) -> tuple[str, str]:
    selected = shutil.which(path_or_name)
    if not selected:
        raise Generation4MediaError(f"Required media tool is unavailable: {path_or_name}")
    resolved = str(Path(selected).resolve(strict=True))
    result = subprocess.run([resolved, "-version"], check=False, capture_output=True, text=True)
    if result.returncode:
        raise Generation4MediaError("Media tool revision is unavailable.")
    return resolved, result.stdout


def _all_hashes(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found.update(_all_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_all_hashes(child))
    elif isinstance(value, str) and SHA256_RE.fullmatch(value):
        found.add(value)
    return found


def _prior_hashes(prior_root: Path) -> tuple[set[str], str, int]:
    root = prior_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Generation4MediaError("Prior Plan-0037 evidence root is invalid.")
    hashes: set[str] = set()
    files = 0
    for path in sorted(root.rglob("*.json")):
        if not path.is_file() or path.is_symlink() or "generation-4" in path.parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Generation4MediaError("Prior evidence JSON is unreadable.") from exc
        hashes.update(_all_hashes(value))
        files += 1
    return hashes, _canonical_hash(sorted(hashes)), files


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise Generation4MediaError("probe_failed")
    try:
        body = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Generation4MediaError("probe_invalid_json") from exc
    streams = [item for item in body.get("streams", []) if item.get("codec_type") == "audio"]
    if len(streams) != 1:
        raise Generation4MediaError("audio_stream_count_not_one")
    stream = streams[0]
    channels = int(stream.get("channels") or 0)
    duration = float(stream.get("duration") or (body.get("format") or {}).get("duration") or 0)
    if channels not in {1, 2}:
        raise Generation4MediaError("unsupported_channel_count")
    if int(stream.get("sample_rate") or 0) <= 0 or duration <= 0:
        raise Generation4MediaError("invalid_probe_measurement")
    return {
        "audio_stream_count": 1, "codec_name": stream.get("codec_name"),
        "channels": channels, "sample_rate": int(stream["sample_rate"]),
        "duration_seconds": duration,
    }


def _decoded_duration(path: Path, ffmpeg: str) -> float:
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
         "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
        check=False, capture_output=True, text=True, timeout=300,
    )
    if result.returncode:
        raise Generation4MediaError("decode_failed")
    values = [line.split("=", 1)[1] for line in result.stdout.splitlines()
              if line.startswith("out_time_us=")]
    if not values:
        raise Generation4MediaError("decode_duration_missing")
    duration = int(values[-1]) / 1_000_000
    if duration <= 0:
        raise Generation4MediaError("decode_duration_invalid")
    return duration


def _qualify_one(
    path: Path, *, source_root: Path, prior_hashes: set[str], seen: set[str],
    ffmpeg: str, ffprobe: str,
) -> dict[str, Any]:
    source = path.expanduser().absolute()
    result: dict[str, Any] = {"path": str(source), "status": "rejected", "reason_code": "unknown"}
    if source.parent != source_root.expanduser().absolute() or not source.is_file() or source.is_symlink():
        result["reason_code"] = "not_top_level_regular_file"
        return result
    digest = sha256_file(source)
    result.update({"source_sha256": digest, "source_bytes": source.stat().st_size})
    if digest in seen:
        result["reason_code"] = "duplicate_candidate_bytes"
        return result
    seen.add(digest)
    if digest in prior_hashes:
        result["reason_code"] = "prior_plan0037_overlap"
        return result
    try:
        probe = _probe(source, ffprobe)
        result["probe"] = probe
        if probe["duration_seconds"] < MIN_DURATION_SECONDS:
            result["reason_code"] = "duration_below_minimum"
            return result
        decoded = _decoded_duration(source, ffmpeg)
        drift = abs(decoded - probe["duration_seconds"])
        result.update({"decoded_duration_seconds": decoded, "duration_drift_seconds": drift})
        if drift > MAX_DURATION_DRIFT_SECONDS:
            result["reason_code"] = "decoded_duration_drift_exceeds_policy"
            return result
    except (Generation4MediaError, OSError, subprocess.TimeoutExpired) as exc:
        result["reason_code"] = str(exc) or type(exc).__name__
        return result
    result.update({"status": "qualified", "reason_code": "qualified"})
    return result


def preview_generation4_media(
    candidates: Sequence[Path], *, source_root: Path = DEFAULT_SOURCE_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
) -> dict[str, Any]:
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise Generation4MediaError("Candidate count is outside the frozen bound.")
    if len({str(Path(item).expanduser().absolute()) for item in candidates}) != len(candidates):
        raise Generation4MediaError("Candidate paths must be unique.")
    ffmpeg, ffmpeg_revision = _tool("ffmpeg")
    ffprobe, ffprobe_revision = _tool("ffprobe")
    prior, prior_set_sha, prior_file_count = _prior_hashes(prior_root)
    seen: set[str] = set()
    results = [
        _qualify_one(Path(item), source_root=source_root, prior_hashes=prior,
                     seen=seen, ffmpeg=ffmpeg, ffprobe=ffprobe)
        for item in candidates
    ]
    qualified = [item for item in results if item["status"] == "qualified"]
    reason_counts: dict[str, int] = {}
    for item in results:
        reason_counts[item["reason_code"]] = reason_counts.get(item["reason_code"], 0) + 1
    eligible = len(qualified) >= MIN_QUALIFIED
    actions = {key: False for key in POST_QUALIFICATION_ACTIONS}
    actions["freeze_media_qualification"] = False
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_to_freeze" if eligible else "insufficient_qualified_media",
        "policy": {
            "maximum_candidates": MAX_CANDIDATES, "minimum_qualified": MIN_QUALIFIED,
            "minimum_duration_seconds": MIN_DURATION_SECONDS,
            "maximum_duration_drift_seconds": MAX_DURATION_DRIFT_SECONDS,
            "top_level_only": True, "retain_decoded_audio": False,
        },
        "tool_authority": {
            "ffmpeg_path": ffmpeg, "ffmpeg_revision_sha256": hashlib.sha256(ffmpeg_revision.encode()).hexdigest(),
            "ffprobe_path": ffprobe, "ffprobe_revision_sha256": hashlib.sha256(ffprobe_revision.encode()).hexdigest(),
        },
        "prior_evidence": {"json_file_count": prior_file_count, "hash_set_sha256": prior_set_sha},
        "candidate_count": len(results), "qualified_count": len(qualified),
        "rejected_count": len(results) - len(qualified), "reason_counts": reason_counts,
        "qualified_set_sha256": _canonical_hash(sorted(item["source_sha256"] for item in qualified)),
        "private_results": results, "repository_authority": _repository_authority(),
        "action_vector": actions, "contains_paths": True, "contains_private_membership": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_biometric_scores": False,
        "did_load_or_run_models": False, "did_retain_decoded_audio": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {**core, "preview_id": f"generation4-media-preview-{digest[:24]}", "content_sha256": digest}


def portable_media_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PORTABLE_SCHEMA, "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "candidate_count": preview["candidate_count"], "qualified_count": preview["qualified_count"],
        "rejected_count": preview["rejected_count"], "reason_counts": dict(preview["reason_counts"]),
        "qualified_set_sha256": preview["qualified_set_sha256"],
        "prior_hash_set_sha256": preview["prior_evidence"]["hash_set_sha256"],
        "action_vector": dict(preview["action_vector"]),
        "contains_paths": False, "contains_private_membership": False,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_biometric_scores": False,
        "did_load_or_run_models": False, "did_retain_decoded_audio": False,
        "will_perform_external_write": False,
    }


def _paths(runtime_root: Path, content_sha: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / "media-qualifications" / f"generation4-media-{content_sha[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def _receipt(preview: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    portable = portable_media_projection(preview)
    actions = dict(portable["action_vector"])
    actions["freeze_media_qualification"] = True
    actions["build_generation4_cohort_preview"] = preview["qualified_count"] >= MIN_QUALIFIED
    return {**portable, "schema_version": RECEIPT_SCHEMA,
            "status": "qualified_pool_frozen" if actions["build_generation4_cohort_preview"] else "insufficient_pool_frozen",
            "manifest_sha256": manifest_sha, "action_vector": actions, "mode": "0600"}


def apply_generation4_media(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    candidates: Sequence[Path], source_root: Path = DEFAULT_SOURCE_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation4_media(candidates, source_root=source_root, prior_root=prior_root)
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation4MediaError("Reviewed media preview is stale.")
    paths = _paths(runtime_root, preview["content_sha256"])
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation4_media(candidates, source_root=source_root, prior_root=prior_root,
                                        runtime_root=runtime_root, expected_content_sha256=expected_content_sha256)
    core = {
        "schema_version": MANIFEST_SCHEMA, "status": "frozen",
        "preview": preview, "repository_authority": preview["repository_authority"],
        "contains_paths": True, "contains_private_membership": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], core)
    receipt = _receipt(preview, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation4_media(
    candidates: Sequence[Path], *, expected_content_sha256: str,
    source_root: Path = DEFAULT_SOURCE_ROOT, prior_root: Path = DEFAULT_PRIOR_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation4_media(candidates, source_root=source_root, prior_root=prior_root)
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation4MediaError("Frozen media preview drifted.")
    _validate_repository_authority(preview["repository_authority"])
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview,
        "repository_authority": preview["repository_authority"],
        "contains_paths": True, "contains_private_membership": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    expected_receipt = _receipt(preview, sha256_file(paths["manifest"]))
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation4MediaError("Media qualification authority drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA,
            "replay_mode": "full_body_with_source_redecode_no_retained_audio",
            "idempotent_replay": True}


def replay_generation4_media_authority(
    expected_content_sha256: str, *, source_root: Path = DEFAULT_SOURCE_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Recover the exact private candidate list and replay the frozen authority."""
    if not SHA256_RE.fullmatch(expected_content_sha256):
        raise Generation4MediaError("Qualification authority hash is invalid.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        results = manifest["preview"]["private_results"]
        candidates = [Path(str(item["path"])) for item in results]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Generation4MediaError("Private candidate authority is unreadable.") from exc
    if not candidates or len(candidates) != manifest["preview"].get("candidate_count"):
        raise Generation4MediaError("Private candidate authority is incomplete.")
    return replay_generation4_media(
        candidates, expected_content_sha256=expected_content_sha256,
        source_root=source_root, prior_root=prior_root, runtime_root=runtime_root,
    )
