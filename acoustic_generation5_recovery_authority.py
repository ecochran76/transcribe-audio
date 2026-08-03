"""Plan 0054 R0 fresh recovery-holdout proposal without audio decoding."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_content_preservation as preservation
import acoustic_content_preservation_adversarial as adversarial
import acoustic_generation5_j2_stop as plan0053_stop
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-recovery-authority-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-recovery-authority-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-recovery-authority-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-recovery-authority-replay.v1"
DEFAULT_SOURCE_ROOT = Path("/mnt/c/Users/ecoch/Documents/Sound Recordings")
DEFAULT_PRIOR_ROOT = Path("~/.local/state/transcribe-audio")
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0054/r0")
PLAN0053_STOP_PREVIEW_SHA256 = "058a36ebebc5a9b743f6db6d856ea5cc2e0c0e123bc8e936bbd1e8597cf5fb3e"
PLAN0053_STOP_MANIFEST_SHA256 = "4033452c29812527209b478b5218b9e773e76726deeffc42a004183b78a994a5"
PLAN0053_FILE_SHA256 = "4ff5b5673bdefb7b61025691ad89c1f79b008887e07382c83588a6250c297073"
CONTRACT_SHA256 = "2b3c988ffedebb8a0070499cc779795bea8bd44236b1234128e18859a6d8b7e9"
MINIMUM_DURATION_SECONDS = 60.0
REQUIRED_MEMBERS = 8
MODULES = (
    "acoustic_content_preservation.py",
    "acoustic_content_preservation_adversarial.py",
    "acoustic_generation5_recovery_authority.py",
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")


class Generation5RecoveryAuthorityError(ValueError):
    """Raised when R0 recovery authority cannot remain deterministic and sealed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5RecoveryAuthorityError("JSON authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5RecoveryAuthorityError("JSON authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5RecoveryAuthorityError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5RecoveryAuthorityError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5RecoveryAuthorityError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise Generation5RecoveryAuthorityError("Repository commit is invalid.")
    hashes = {}
    for name in MODULES:
        body = _git(["show", f"{commit}:{name}"], binary=True)
        if not isinstance(body, bytes):
            raise Generation5RecoveryAuthorityError("Committed module is unavailable.")
        digest = hashlib.sha256(body).hexdigest()
        if digest != sha256_file(Path(__file__).resolve().parent / name):
            raise Generation5RecoveryAuthorityError("Committed module drifted.")
        hashes[name] = digest
    return {
        "commit": commit,
        "module_sha256": hashes,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _plan0053_terminal() -> dict[str, Any]:
    paths = plan0053_stop._paths(
        plan0053_stop.DEFAULT_RUNTIME_ROOT, PLAN0053_STOP_PREVIEW_SHA256
    )
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    if sha256_file(paths["manifest"]) != PLAN0053_STOP_MANIFEST_SHA256:
        raise Generation5RecoveryAuthorityError("Plan 0053 terminal manifest drifted.")
    preview = _read_json(paths["manifest"]).get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("content_sha256") != PLAN0053_STOP_PREVIEW_SHA256
        or preview.get("terminal_decision") != "stop"
        or any((preview.get("action_vector") or {}).values())
    ):
        raise Generation5RecoveryAuthorityError("Plan 0053 terminal authority drifted.")
    return dict(preview)


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


def _read_json_sequence(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Generation5RecoveryAuthorityError("Prior evidence JSON is unreadable.") from exc
    decoder = json.JSONDecoder()
    values: list[Any] = []
    offset = 0
    try:
        while offset < len(text):
            while offset < len(text) and text[offset].isspace():
                offset += 1
            if offset >= len(text):
                break
            value, offset = decoder.raw_decode(text, offset)
            values.append(value)
    except json.JSONDecodeError as exc:
        raise Generation5RecoveryAuthorityError("Prior evidence JSON is unreadable.") from exc
    if not values:
        raise Generation5RecoveryAuthorityError("Prior evidence JSON is empty.")
    return values


def _evidence_hashes(path: Path) -> tuple[set[str], str]:
    try:
        values = _read_json_sequence(path)
    except Generation5RecoveryAuthorityError:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise Generation5RecoveryAuthorityError("Prior evidence is unreadable.") from exc
        tokens = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", text))
        return tokens, "raw_sha256_token_fallback"
    return _all_hashes(values), "structured_json"


def _exclusion_union(prior_root: Path) -> dict[str, Any]:
    root = prior_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Generation5RecoveryAuthorityError("Prior evidence root is invalid.")
    hashes: set[str] = set()
    file_hashes: list[str] = []
    parse_mode_counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise Generation5RecoveryAuthorityError("Prior evidence escaped its root.") from exc
        if not path.is_file() or path.is_symlink() or (relative.parts and relative.parts[0] == "plan-0054"):
            continue
        found, parse_mode = _evidence_hashes(path)
        hashes.update(found)
        parse_mode_counts[parse_mode] = parse_mode_counts.get(parse_mode, 0) + 1
        file_hashes.append(sha256_file(path))
    if not hashes or not file_hashes:
        raise Generation5RecoveryAuthorityError("Prior exclusion evidence is empty.")
    return {
        "hashes": hashes,
        "json_file_count": len(file_hashes),
        "json_file_set_sha256": _canonical_hash(sorted(file_hashes)),
        "excluded_hash_count": len(hashes),
        "excluded_hash_set_sha256": _canonical_hash(sorted(hashes)),
        "parse_mode_counts": parse_mode_counts,
    }


def _recording_start(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    normalized = parsed.astimezone(timezone.utc)
    return value.strip(), normalized.isoformat().replace("+00:00", "Z")


def _sidecars(source_root: Path) -> dict[Path, list[dict[str, Any]]]:
    resolved_root = source_root.resolve(strict=True)
    found: dict[Path, list[dict[str, Any]]] = {}
    for path in sorted(source_root.glob("*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or value.get("schema_version") not in {1, 2}:
            continue
        start = _recording_start(value.get("recording_start"))
        if start is None:
            continue
        matched_sources: dict[Path, list[str]] = {}
        for field in ("source_media_path", "working_media_path"):
            raw_source = str(value.get(field) or "").strip()
            if not raw_source:
                continue
            candidate = Path(raw_source).expanduser()
            try:
                resolved_source = candidate.resolve(strict=True)
                relative = resolved_source.relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            if len(relative.parts) == 1 and resolved_source.suffix.lower() == ".m4a":
                matched_sources.setdefault(resolved_source, []).append(field)
        if not matched_sources:
            continue
        transcript_sha256 = sha256_file(path)
        raw_field_hashes = {
            field: hashlib.sha256(str(value.get(field) or "").encode("utf-8")).hexdigest()
            for field in ("source_media_path", "working_media_path")
        }
        ambiguous_field_targets = len(matched_sources) > 1
        for resolved_source, matched_fields in matched_sources.items():
            found.setdefault(resolved_source, []).append(
                {
                    "transcript_path": str(path.resolve()),
                    "transcript_sha256": transcript_sha256,
                    "matched_source_fields": sorted(matched_fields),
                    "raw_source_field_sha256": raw_field_hashes,
                    "ambiguous_field_targets": ambiguous_field_targets,
                    "recording_start_original": start[0],
                    "recording_start_utc": start[1],
                }
            )
    return found


def _tool(name: str) -> tuple[str, str]:
    selected = shutil.which(name)
    if not selected:
        raise Generation5RecoveryAuthorityError("Required probe tool is unavailable.")
    path = str(Path(selected).resolve(strict=True))
    result = subprocess.run([path, "-version"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise Generation5RecoveryAuthorityError("Probe tool identity is unavailable.")
    return path, result.stdout.splitlines()[0]


def _probe(path: Path, ffprobe_path: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe_path, "-v", "error", "-select_streams", "a",
            "-show_entries", "format=duration:stream=index,codec_type,codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if result.returncode:
        raise Generation5RecoveryAuthorityError("probe_failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Generation5RecoveryAuthorityError("probe_invalid_json") from exc
    streams = [
        item for item in value.get("streams") or []
        if isinstance(item, Mapping) and item.get("codec_type") == "audio"
    ]
    if len(streams) != 1:
        raise Generation5RecoveryAuthorityError("audio_stream_count_not_one")
    stream = streams[0]
    try:
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
        duration = float((value.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise Generation5RecoveryAuthorityError("probe_dimensions_invalid") from exc
    if stream.get("codec_name") != "aac":
        raise Generation5RecoveryAuthorityError("unsupported_codec")
    if channels not in {1, 2} or sample_rate <= 0:
        raise Generation5RecoveryAuthorityError("probe_dimensions_invalid")
    if duration < MINIMUM_DURATION_SECONDS:
        raise Generation5RecoveryAuthorityError("duration_below_minimum")
    return {
        "codec_name": "aac",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
    }


def _inventory(source_root: Path, prior_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    root = source_root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise Generation5RecoveryAuthorityError("Source root is invalid.")
    exclusions = _exclusion_union(prior_root)
    sidecars = _sidecars(root)
    ffprobe_path, ffprobe_revision = _tool("ffprobe")
    seen: set[str] = set()
    rows = []
    for path in sorted(root.glob("*.m4a")):
        source = path.absolute()
        row: dict[str, Any] = {"path": str(source), "status": "rejected", "reason_code": "unknown"}
        if not source.is_file() or source.is_symlink() or source.parent != root:
            row["reason_code"] = "not_top_level_regular_file"
            rows.append(row)
            continue
        digest = sha256_file(source)
        row["source_sha256"] = digest
        if digest in seen:
            row["reason_code"] = "duplicate_candidate_bytes"
            rows.append(row)
            continue
        seen.add(digest)
        if digest in exclusions["hashes"]:
            row["reason_code"] = "prior_evidence_overlap"
            rows.append(row)
            continue
        matches = sidecars.get(source.resolve(), [])
        if len(matches) != 1 or any(match.get("ambiguous_field_targets") is True for match in matches):
            row["reason_code"] = "sidecar_missing" if not matches else "sidecar_ambiguous"
            rows.append(row)
            continue
        row.update(matches[0])
        try:
            row["probe"] = _probe(source, ffprobe_path)
        except Generation5RecoveryAuthorityError as exc:
            row["reason_code"] = str(exc)
            rows.append(row)
            continue
        row.update({"status": "eligible", "reason_code": "eligible"})
        rows.append(row)
    eligible = sorted(
        (row for row in rows if row["status"] == "eligible"),
        key=lambda row: (
            row["recording_start_utc"], row["source_sha256"], row["transcript_sha256"]
        ),
    )
    ordered = eligible + [row for row in rows if row["status"] != "eligible"]
    public_exclusions = {key: value for key, value in exclusions.items() if key != "hashes"}
    tools = {"ffprobe_path": ffprobe_path, "ffprobe_revision": ffprobe_revision}
    return ordered, public_exclusions, tools


def preview_generation5_recovery_authority(
    *,
    terminal_preview: Mapping[str, Any] | None = None,
    inventory: list[dict[str, Any]] | None = None,
    exclusion_summary: Mapping[str, Any] | None = None,
    tool_identity: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    terminal = dict(terminal_preview or _plan0053_terminal())
    if terminal.get("content_sha256") != PLAN0053_STOP_PREVIEW_SHA256 or terminal.get("terminal_decision") != "stop":
        raise Generation5RecoveryAuthorityError("Plan 0053 terminal input is invalid.")
    plan0053_path = Path(__file__).resolve().parent / "docs/dev/plans/0053-2026-08-03-generation-5-duration-validation-and-blind-evaluation.md"
    if sha256_file(plan0053_path) != PLAN0053_FILE_SHA256:
        raise Generation5RecoveryAuthorityError("Plan 0053 file authority drifted.")
    if preservation.contract()["content_sha256"] != CONTRACT_SHA256:
        raise Generation5RecoveryAuthorityError("Content-preservation contract drifted.")
    if inventory is None:
        rows, exclusions, tools = _inventory(DEFAULT_SOURCE_ROOT, DEFAULT_PRIOR_ROOT)
    else:
        rows = list(inventory)
        exclusions = dict(exclusion_summary or {})
        tools = dict(tool_identity or {})
    eligible = [row for row in rows if row.get("status") == "eligible"]
    if len(eligible) < REQUIRED_MEMBERS:
        raise Generation5RecoveryAuthorityError("Fresh recovery membership is insufficient.")
    selected = [dict(row) for row in eligible[:REQUIRED_MEMBERS]]
    for index, row in enumerate(selected):
        row["role"] = "recovery_negative_source" if index == 0 else "positive_holdout"
        row["ordinal"] = index + 1
    hashes = [str(row.get("source_sha256") or "") for row in selected]
    if len(set(hashes)) != REQUIRED_MEMBERS or any(not SHA256_RE.fullmatch(value) for value in hashes):
        raise Generation5RecoveryAuthorityError("Fresh recovery membership overlaps or is invalid.")
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason_code") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_contract = dict(adversarial.EXPECTED_REASON_CONTRACT)
    private = {"inventory": rows, "selected_membership": selected}
    actions = {
        "submit_exact_membership_to_j0": True,
        "decode_recovery_negative_source": False,
        "decode_positive_holdout": False,
        "enumerate_evaluation_candidates": False,
        "access_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j0_review",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "plan0053_stop_preview_sha256": PLAN0053_STOP_PREVIEW_SHA256,
        "plan0053_stop_manifest_sha256": PLAN0053_STOP_MANIFEST_SHA256,
        "plan0053_file_sha256": PLAN0053_FILE_SHA256,
        "content_preservation_contract_sha256": CONTRACT_SHA256,
        "source_root": str(DEFAULT_SOURCE_ROOT),
        "selection_contract": {
            "top_level_only": True,
            "extension": ".m4a",
            "sidecar_schema_versions": [1, 2],
            "sidecar_source_fields": ["source_media_path", "working_media_path"],
            "recording_start": "offset_aware_rfc3339_required_no_fallback",
            "metadata_probe": "one_aac_stream_one_or_two_channels_positive_rate",
            "minimum_duration_seconds": MINIMUM_DURATION_SECONDS,
            "order": ["recording_start_utc", "source_sha256", "transcript_sha256"],
            "row_1_role": "recovery_negative_source",
            "rows_2_through_8_role": "positive_holdout",
        },
        "exclusion_summary": exclusions,
        "tool_identity": tools,
        "inventory_count": len(rows),
        "eligible_count": len(eligible),
        "rejection_reason_counts": reason_counts,
        "selected_count": len(selected),
        "selected_membership_sha256": _canonical_hash(selected),
        "selected_source_set_sha256": _canonical_hash(sorted(hashes)),
        "negative_source_sha256": hashes[0],
        "positive_holdout_set_sha256": _canonical_hash(sorted(hashes[1:])),
        "recovery_negative_seed": adversarial.RECOVERY_HOLDOUT_SEED,
        "recovery_segment_start_seconds": adversarial._segment_start_seconds(
            hashes[0], adversarial.RECOVERY_HOLDOUT_SEED
        ),
        "expected_reason_contract": reason_contract,
        "expected_reason_contract_sha256": _canonical_hash(reason_contract),
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": True,
        "contains_private_membership": True,
        "did_decode_audio": False,
        "did_inspect_transcript_utterances": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: item for key, item in preview.items()
        if key not in {"private_evidence", "repository_authority", "negative_source_sha256", "source_root"}
    }
    value["schema_version"] = RECEIPT_SCHEMA
    value["contains_paths"] = False
    value["contains_private_membership"] = False
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-recovery-{content_sha256[:24]}"
    return {"root": root, "run": run, "manifest": run / "private-manifest.json", "receipt": run / "receipt.json"}


def apply_generation5_recovery_authority(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_recovery_authority()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5RecoveryAuthorityError("Reviewed R0 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_recovery_authority(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_recovery_authority(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_recovery_authority()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation5RecoveryAuthorityError("R0 authority drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5RecoveryAuthorityError("R0 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
