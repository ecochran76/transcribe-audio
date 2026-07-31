"""Immutable, content-addressed audio derivatives for Plan 0037 P1.

This module owns only baseline decoding, timestamp identity, signal-quality
measurement, and private operation receipts.  VAD, enhancement, diarization,
and speaker models belong to later Plan 0037 packets.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from acoustic_identity_contracts import (
    AUDIO_QUALITY_SCHEMA,
    DERIVED_AUDIO_SCHEMA,
    validate_artifact,
)


DRY_RUN_SCHEMA = "transcribe-audio.audio-derivative-dry-run.v1"
RUN_MANIFEST_SCHEMA = "transcribe-audio.audio-derivative-run.v1"
APPLY_RECEIPT_SCHEMA = "transcribe-audio.audio-derivative-apply-receipt.v1"
REPLAY_RECEIPT_SCHEMA = "transcribe-audio.audio-derivative-replay-receipt.v1"
ROLLBACK_RECEIPT_SCHEMA = "transcribe-audio.audio-derivative-rollback-receipt.v1"
APPLY_TOKEN = "APPLY_AUDIO_DERIVATIVE"
ROLLBACK_TOKEN = "ROLLBACK_AUDIO_DERIVATIVE"
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/audio-derivatives"
)
RUN_ID_RE = re.compile(r"audio-run-[a-f0-9]{24}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
SOURCE_BLOB_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_FORMAT = "s16"
TARGET_CODEC = "pcm_s16le"
SILENCE_THRESHOLD = 0  # exact-zero digital silence in signed 16-bit PCM
CLIP_THRESHOLD = 32_760
DURATION_TOLERANCE_SECONDS = 0.05
TOOL_TIMEOUT_SECONDS = 180


class AudioDerivativeError(ValueError):
    """Raised when P1 cannot preserve its immutable audio contract."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AudioDerivativeError(f"Artifact must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioDerivativeError(f"Artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AudioDerivativeError(f"Artifact must contain an object: {path}")
    return value


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.absolute().parts[1:]:
        current /= part
        if current.is_symlink():
            raise AudioDerivativeError(f"Private runtime path must not contain symlinks: {current}")


def _ensure_private_tree(root: Path, leaf: Path) -> None:
    root = root.absolute()
    leaf = leaf.absolute()
    _reject_symlink_components(root)
    if root.is_symlink():
        raise AudioDerivativeError(f"Private runtime root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir() or root.is_symlink():
        raise AudioDerivativeError(f"Private runtime root is unsafe: {root}")
    os.chmod(root, 0o700)
    relative = leaf.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise AudioDerivativeError(f"Private runtime path must not be a symlink: {current}")
        current.mkdir(exist_ok=True, mode=0o700)
        if not current.is_dir():
            raise AudioDerivativeError(f"Private runtime path is not a directory: {current}")
        os.chmod(current, 0o700)


def _require_private_file(path: Path, root: Path) -> None:
    absolute_root = root.expanduser().absolute()
    absolute_path = path.expanduser().absolute()
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise AudioDerivativeError(
            f"Private artifact escapes its runtime root: {absolute_path}"
        ) from exc
    if not relative.parts:
        raise AudioDerivativeError("Private artifact path is invalid.")
    _reject_symlink_components(absolute_path)
    if not absolute_path.is_file() or stat.S_IMODE(absolute_path.stat().st_mode) != 0o600:
        raise AudioDerivativeError(f"Private artifact must be a 0600 regular file: {absolute_path}")
    current = absolute_root
    directories = [current]
    for part in relative.parts[:-1]:
        current /= part
        directories.append(current)
    for directory in directories:
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or stat.S_IMODE(directory.stat().st_mode) != 0o700
        ):
            raise AudioDerivativeError(
                f"Private artifact directory must be 0700: {directory}"
            )


def _write_private_json(path: Path, payload: dict[str, Any]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise AudioDerivativeError(f"Immutable artifact already exists: {path}") from exc
        Path(temporary_name).unlink()
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise
    return path


def _write_immutable_json(
    path: Path,
    payload: dict[str, Any],
    *,
    volatile_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    if path.is_symlink():
        raise AudioDerivativeError(f"Immutable artifact must not be a symlink: {path}")
    if path.exists():
        existing = _read_object(path)
        left = dict(existing)
        right = dict(payload)
        for field_name in volatile_fields:
            left.pop(field_name, None)
            right.pop(field_name, None)
        if left != right:
            raise AudioDerivativeError(f"Immutable artifact conflict: {path}")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise AudioDerivativeError(f"Private artifact mode is not 0600: {path}")
        return existing
    _write_private_json(path, payload)
    return payload


def _tool(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise AudioDerivativeError(f"Required local tool is unavailable: {binary}")
    return str(Path(path).resolve())


def _run(command: list[str], *, timeout_seconds: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip().splitlines()[-1:]
            detail = detail[0] if detail else ""
        raise AudioDerivativeError(
            f"Audio tool failed: {Path(command[0]).name}"
            + (f": {detail}" if detail else "")
        ) from exc


def _tool_version(binary_path: str) -> str:
    result = _run([binary_path, "-version"], timeout_seconds=30)
    version_output = result.stdout.strip()
    if not version_output:
        raise AudioDerivativeError(f"Could not read tool version: {binary_path}")
    return version_output


def _probe_audio(source_path: Path, ffprobe_path: str) -> dict[str, Any]:
    result = _run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration",
            "-of",
            "json",
            str(source_path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioDerivativeError("ffprobe returned invalid JSON.") from exc
    streams = [
        stream
        for stream in payload.get("streams") or []
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if not streams:
        raise AudioDerivativeError("Source has no decodable audio stream.")
    if len(streams) != 1:
        raise AudioDerivativeError("Source must contain exactly one audio stream.")
    stream = streams[0]
    duration_text = stream.get("duration") or (payload.get("format") or {}).get(
        "duration"
    )
    try:
        duration = float(duration_text)
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
    except (TypeError, ValueError) as exc:
        raise AudioDerivativeError("Source audio metadata is invalid.") from exc
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise AudioDerivativeError("Source audio is empty or has invalid dimensions.")
    if channels != 1:
        raise AudioDerivativeError("P1 supports mono source audio only.")
    return {
        "audio_stream_count": len(streams),
        "selected_audio_stream_index": int(stream.get("index") or 0),
        "codec_name": str(stream.get("codec_name") or ""),
        "format_name": str((payload.get("format") or {}).get("format_name") or ""),
        "sample_rate": sample_rate,
        "channels": channels,
        "channel_layout": str(stream.get("channel_layout") or "unknown"),
        "duration_seconds": duration,
    }


def _source_identity(source_audio: Path, ffprobe_path: str) -> dict[str, Any]:
    if source_audio.expanduser().is_symlink():
        raise AudioDerivativeError("Source audio must not be a symlink.")
    try:
        source_path = source_audio.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AudioDerivativeError(f"Source audio is unavailable: {source_audio}") from exc
    if not source_path.is_file():
        raise AudioDerivativeError("Source audio must be a regular file.")
    source_stat = source_path.stat()
    if source_stat.st_size <= 0:
        raise AudioDerivativeError("Source audio is empty.")
    identity = {
        "path": str(source_path),
        "sha256": _sha256_file(source_path),
        "bytes": source_stat.st_size,
        "mode": stat.S_IMODE(source_stat.st_mode),
        "mtime_ns": source_stat.st_mtime_ns,
        "probe": _probe_audio(source_path, ffprobe_path),
    }
    _source_unchanged(identity)
    return identity


def _recipe(ffmpeg_path: str, ffprobe_path: str) -> dict[str, Any]:
    decode_arguments = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        "{source_path}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        TARGET_CODEC,
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "{output_path}",
    ]
    base = {
        "operation": "baseline_decode",
        "decoder_path": ffmpeg_path,
        "probe_path": ffprobe_path,
        "decoder_revision": _tool_version(ffmpeg_path),
        "probe_revision": _tool_version(ffprobe_path),
        "decode_arguments": decode_arguments,
        "parameters": {
            "audio_stream_policy": "exactly_one_audio_stream",
            "channel_policy": "mono_only_no_mixdown",
            "target_channels": TARGET_CHANNELS,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "target_sample_format": TARGET_SAMPLE_FORMAT,
            "target_codec": TARGET_CODEC,
            "metadata_policy": "drop",
            "bitexact_flags": True,
            "subprocess_timeout_seconds": TOOL_TIMEOUT_SECONDS,
            "duration_tolerance_seconds": DURATION_TOLERANCE_SECONDS,
            "sample_normalization_denominator": 32768,
            "clip_threshold_signed_pcm": CLIP_THRESHOLD,
            "digital_silence_threshold_signed_pcm": SILENCE_THRESHOLD,
        },
        "model_revisions": {},
    }
    return {**base, "revision": f"audio-recipe-{_canonical_hash(base)[:24]}"}


def _effective_recipe(requested: Mapping[str, Any], *, identity_copy: bool) -> dict[str, Any]:
    if not identity_copy:
        return dict(requested)
    base = {key: value for key, value in requested.items() if key != "revision"}
    base["operation"] = "identity_copy"
    base["requested_operation"] = requested["operation"]
    base["requested_recipe_revision"] = requested["revision"]
    return {**base, "revision": f"audio-recipe-{_canonical_hash(base)[:24]}"}


def _runtime_paths(runtime_root: Path, run_id: str) -> dict[str, Path]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise AudioDerivativeError("Audio derivative run ID is invalid.")
    root = runtime_root.expanduser().absolute()
    run_dir = root / "runs" / run_id
    return {
        "root": root,
        "run_dir": run_dir,
        "dry_run": run_dir / "dry-run.json",
        "recipe": run_dir / "recipe.json",
        "derived": run_dir / "derived-audio.json",
        "quality": run_dir / "audio-quality.json",
        "manifest": run_dir / "manifest.json",
        "apply_receipt": run_dir / "apply-receipt.json",
        "replay_receipt_active": run_dir / "replay-active-receipt.json",
        "replay_receipt_rolled_back": run_dir / "replay-rolled-back-receipt.json",
        "rollback_receipt": run_dir / "rollback-receipt.json",
    }


def _build_plan(
    source_audio: Path,
    *,
    runtime_root: Optional[Path] = None,
    source_blob_id: Optional[str] = None,
    expected_source_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    ffmpeg_path = _tool("ffmpeg")
    ffprobe_path = _tool("ffprobe")
    source = _source_identity(source_audio, ffprobe_path)
    if expected_source_sha256 is not None and (
        not SHA256_RE.fullmatch(expected_source_sha256)
        or expected_source_sha256 != source["sha256"]
    ):
        raise AudioDerivativeError("Source audio does not match the expected SHA-256.")
    bound_blob_id = source_blob_id or f"source-{source['sha256'][:24]}"
    if not SOURCE_BLOB_ID_RE.fullmatch(bound_blob_id):
        raise AudioDerivativeError("Source blob ID is invalid.")
    source["source_blob_id"] = bound_blob_id
    recipe = _recipe(ffmpeg_path, ffprobe_path)
    run_identity = {
        "source_blob_id": bound_blob_id,
        "source_sha256": source["sha256"],
        "recipe_revision": recipe["revision"],
    }
    run_id = f"audio-run-{_canonical_hash(run_identity)[:24]}"
    paths = _runtime_paths((runtime_root or DEFAULT_RUNTIME_ROOT).expanduser(), run_id)
    plan = {
        "schema_version": DRY_RUN_SCHEMA,
        "run_id": run_id,
        "source": source,
        "recipe": recipe,
        "runtime_root": str(paths["root"]),
        "will_execute_decoder": False,
        "will_write_derived_audio": False,
        "will_modify_source": False,
        "will_run_vad": False,
        "will_run_enhancement": False,
        "will_run_models": False,
        "will_perform_external_write": False,
        "created_at": _utc_now(),
    }
    return plan, paths


def dry_run(
    source_audio: Path,
    *,
    runtime_root: Optional[Path] = None,
    source_blob_id: Optional[str] = None,
    expected_source_sha256: Optional[str] = None,
) -> dict[str, Any]:
    plan, paths = _build_plan(
        source_audio,
        runtime_root=runtime_root,
        source_blob_id=source_blob_id,
        expected_source_sha256=expected_source_sha256,
    )
    _ensure_private_tree(paths["root"], paths["run_dir"])
    stored = _write_immutable_json(
        paths["dry_run"], plan, volatile_fields=("created_at",)
    )
    return {**stored, "dry_run_path": str(paths["dry_run"])}


def _decode(source_path: Path, output_path: Path, recipe: Mapping[str, Any]) -> None:
    parameters = recipe["parameters"]
    arguments = [
        str(source_path)
        if value == "{source_path}"
        else str(output_path)
        if value == "{output_path}"
        else str(value)
        for value in recipe["decode_arguments"]
    ]
    _run(
        [str(recipe["decoder_path"]), *arguments],
        timeout_seconds=int(parameters["subprocess_timeout_seconds"]),
    )


def _pcm_metrics(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if channels != 1 or sample_width != 2 or sample_rate != TARGET_SAMPLE_RATE:
                raise AudioDerivativeError("Derived WAV dimensions violate the recipe.")
            samples = array.array("h")
            remaining = frame_count
            while remaining:
                frames = audio.readframes(min(remaining, 65_536))
                if not frames:
                    break
                chunk = array.array("h")
                chunk.frombytes(frames)
                if sys.byteorder != "little":
                    chunk.byteswap()
                samples.extend(chunk)
                remaining -= len(chunk)
    except (wave.Error, OSError) as exc:
        raise AudioDerivativeError("Derived audio is not valid PCM WAV.") from exc
    if not samples or frame_count <= 0:
        raise AudioDerivativeError("Derived audio contains no PCM frames.")
    count = len(samples)
    total = sum(int(value) for value in samples)
    squares = sum(int(value) * int(value) for value in samples)
    peak = max(abs(int(value)) for value in samples)
    rms = math.sqrt(squares / count)
    scale = 32_768.0
    clipped_count = sum(
        1 for value in samples if abs(int(value)) >= CLIP_THRESHOLD
    )
    digital_silence_count = sum(1 for value in samples if int(value) == 0)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "sample_count": count,
        "sample_normalization_denominator": 32_768,
        "duration_seconds": frame_count / sample_rate,
        "peak_amplitude": peak / scale,
        "peak_dbfs": 20 * math.log10(peak / scale) if peak else None,
        "rms_amplitude": rms / scale,
        "rms_dbfs": 20 * math.log10(rms / scale) if rms else None,
        "dc_offset": (total / count) / scale,
        "clip_threshold_signed_pcm": CLIP_THRESHOLD,
        "clipped_sample_count": clipped_count,
        "clipped_sample_fraction": clipped_count / count,
        "digital_silence_threshold_signed_pcm": SILENCE_THRESHOLD,
        "digital_silence_sample_count": digital_silence_count,
        "digital_silence_fraction": digital_silence_count / count,
    }


def _quality_payload(
    *,
    artifact_id: str,
    source_probe: Mapping[str, Any],
    metrics: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    abstention: list[str] = ["usable_speech_not_assessed_until_p2"]
    if metrics["duration_seconds"] < 1.0:
        warnings.append("short_audio")
        abstention.append("insufficient_duration_for_identity")
    if metrics["digital_silence_fraction"] >= 0.95:
        warnings.append("predominantly_digital_silence")
        abstention.append("no_usable_signal")
    if metrics["clipped_sample_fraction"] >= 0.001:
        warnings.append("clipping_detected")
        abstention.append("excessive_clipping")
    payload = {
        "schema_version": AUDIO_QUALITY_SCHEMA,
        "assessment_id": f"quality-{artifact_id.removeprefix('derived-')}",
        "audio_artifact_id": artifact_id,
        "usable_speech_seconds": None,
        "metrics": metrics,
        "warnings": warnings,
        "abstention_reasons": abstention,
        "created_at": created_at,
    }
    return validate_artifact("audio_quality", payload)


def _source_unchanged(before: Mapping[str, Any]) -> None:
    path = Path(str(before["path"]))
    current = path.stat()
    if (
        current.st_size != int(before["bytes"])
        or stat.S_IMODE(current.st_mode) != int(before["mode"])
        or current.st_mtime_ns != int(before["mtime_ns"])
        or _sha256_file(path) != before["sha256"]
    ):
        raise AudioDerivativeError("Source audio changed during derivative processing.")


def apply_derivative(
    source_audio: Path,
    *,
    runtime_root: Optional[Path] = None,
    approval_token: str,
    source_blob_id: Optional[str] = None,
    expected_source_sha256: Optional[str] = None,
) -> dict[str, Any]:
    if approval_token != APPLY_TOKEN:
        raise AudioDerivativeError(f"Apply requires approval token {APPLY_TOKEN}.")
    plan, paths = _build_plan(
        source_audio,
        runtime_root=runtime_root,
        source_blob_id=source_blob_id,
        expected_source_sha256=expected_source_sha256,
    )
    _ensure_private_tree(paths["root"], paths["run_dir"])
    if not paths["dry_run"].is_file():
        raise AudioDerivativeError("Apply requires the matching persisted dry run.")
    _require_private_file(paths["dry_run"], paths["root"])
    persisted_plan = _read_object(paths["dry_run"])
    current_plan = dict(plan)
    persisted_comparison = dict(persisted_plan)
    current_plan.pop("created_at", None)
    persisted_comparison.pop("created_at", None)
    if persisted_comparison != current_plan:
        raise AudioDerivativeError("Source or recipe changed after the dry run.")
    if paths["rollback_receipt"].exists():
        raise AudioDerivativeError("A rolled-back run cannot be reactivated.")
    if paths["manifest"].exists():
        if not paths["apply_receipt"].is_file():
            raise AudioDerivativeError("Applied run is missing its immutable receipt.")
        replay = replay_derivative(plan["run_id"], runtime_root=paths["root"])
        if not replay["active"]:
            raise AudioDerivativeError("A rolled-back run cannot be reactivated.")
        return {
            **_read_object(paths["apply_receipt"]),
            "manifest": _read_object(paths["manifest"]),
            "idempotent_replay": True,
        }

    temporary_dir = paths["root"] / "tmp"
    _ensure_private_tree(paths["root"], temporary_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{plan['run_id']}.", suffix=".wav", dir=temporary_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        live_recipe = _recipe(
            str(plan["recipe"]["decoder_path"]),
            str(plan["recipe"]["probe_path"]),
        )
        if live_recipe != plan["recipe"]:
            raise AudioDerivativeError("Audio tool or recipe changed before decode.")
        _decode(Path(plan["source"]["path"]), temporary_path, plan["recipe"])
        os.chmod(temporary_path, 0o600)
        _source_unchanged(plan["source"])
        output_sha256 = _sha256_file(temporary_path)
        metrics = _pcm_metrics(temporary_path)
        output_duration = float(metrics["duration_seconds"])
        source_duration = float(plan["source"]["probe"]["duration_seconds"])
        if output_duration <= 0 or source_duration <= 0:
            raise AudioDerivativeError("Measured audio duration is invalid.")
        duration_drift = abs(output_duration - source_duration)
        duration_tolerance = float(
            plan["recipe"]["parameters"]["duration_tolerance_seconds"]
        )
        if duration_drift > duration_tolerance:
            raise AudioDerivativeError(
                "Decoded duration drift exceeds the frozen recipe tolerance."
            )
        artifact_id = f"derived-{output_sha256}"
        artifact_dir = paths["root"] / "artifacts" / output_sha256[:2]
        _ensure_private_tree(paths["root"], artifact_dir)
        artifact_path = artifact_dir / f"{output_sha256}.wav"
        if artifact_path.is_symlink():
            raise AudioDerivativeError(
                f"Content-addressed artifact must not be a symlink: {artifact_path}"
            )
        if artifact_path.exists():
            _require_private_file(artifact_path, paths["root"])
            if _sha256_file(artifact_path) != output_sha256:
                raise AudioDerivativeError(
                    f"Content-addressed artifact conflict: {artifact_path}"
                )
            temporary_path.unlink()
        else:
            try:
                os.link(temporary_path, artifact_path)
            except FileExistsError as exc:
                raise AudioDerivativeError(
                    f"Content-addressed artifact appeared concurrently: {artifact_path}"
                ) from exc
            temporary_path.unlink()

        effective_recipe = _effective_recipe(
            plan["recipe"], identity_copy=output_sha256 == plan["source"]["sha256"]
        )

        created_at = _utc_now()
        derived = validate_artifact(
            "derived_audio",
            {
                "schema_version": DERIVED_AUDIO_SCHEMA,
                "artifact_id": artifact_id,
                "source_blob_id": plan["source"]["source_blob_id"],
                "source_sha256": plan["source"]["sha256"],
                "output_sha256": output_sha256,
                "source_duration_seconds": source_duration,
                "output_duration_seconds": output_duration,
                "recipe": effective_recipe,
                "timestamp_map": [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": source_duration,
                        "output_start_seconds": 0.0,
                        "output_end_seconds": output_duration,
                        "mapping": "affine_full_recording",
                        "source_sample_rate": plan["source"]["probe"]["sample_rate"],
                        "output_sample_rate": metrics["sample_rate"],
                        "output_frame_count": metrics["frame_count"],
                        "duration_tolerance_seconds": duration_tolerance,
                        "measured_duration_drift_seconds": duration_drift,
                    }
                ],
                "created_at": created_at,
            },
        )
        quality = _quality_payload(
            artifact_id=artifact_id,
            source_probe=plan["source"]["probe"],
            metrics=metrics,
            created_at=created_at,
        )
        stored_derived = _write_immutable_json(paths["derived"], derived)
        stored_quality = _write_immutable_json(paths["quality"], quality)
        stored_recipe = _write_immutable_json(paths["recipe"], effective_recipe)
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": plan["run_id"],
            "status": "active",
            "source": plan["source"],
            "recipe": plan["recipe"],
            "effective_recipe": stored_recipe,
            "recipe_path": str(paths["recipe"]),
            "recipe_sha256": _sha256_file(paths["recipe"]),
            "dry_run_path": str(paths["dry_run"]),
            "dry_run_sha256": _sha256_file(paths["dry_run"]),
            "artifact_path": str(artifact_path),
            "derived_audio_path": str(paths["derived"]),
            "derived_audio_sha256": _sha256_file(paths["derived"]),
            "audio_quality_path": str(paths["quality"]),
            "audio_quality_sha256": _sha256_file(paths["quality"]),
            "derived_audio": stored_derived,
            "audio_quality": stored_quality,
            "eligible_for_identity": False,
            "identity_eligibility_reason": "usable_speech_not_assessed_until_p2",
            "will_run_vad": False,
            "will_run_enhancement": False,
            "will_run_models": False,
            "will_modify_source": False,
            "will_perform_external_write": False,
            "created_at": created_at,
        }
        stored_manifest = _write_immutable_json(paths["manifest"], manifest)
        receipt = {
            "schema_version": APPLY_RECEIPT_SCHEMA,
            "run_id": plan["run_id"],
            "status": "applied",
            "manifest_path": str(paths["manifest"]),
            "manifest_sha256": _sha256_file(paths["manifest"]),
            "artifact_path": str(artifact_path),
            "output_sha256": output_sha256,
            "source_unchanged": True,
            "will_perform_external_write": False,
            "applied_at": created_at,
        }
        stored_receipt = _write_immutable_json(paths["apply_receipt"], receipt)
        return {
            **stored_receipt,
            "manifest": stored_manifest,
            "idempotent_replay": False,
        }
    finally:
        try:
            temporary_path.unlink()
        except OSError:
            pass


def replay_derivative(
    run_id: str,
    *,
    runtime_root: Optional[Path] = None,
    include_validated_manifest: bool = False,
) -> dict[str, Any]:
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser()
    paths = _runtime_paths(root, run_id)
    for evidence_path in (
        paths["dry_run"],
        paths["recipe"],
        paths["derived"],
        paths["quality"],
        paths["manifest"],
        paths["apply_receipt"],
    ):
        _require_private_file(evidence_path, paths["root"])
    manifest = _read_object(paths["manifest"])
    if manifest.get("run_id") != run_id:
        raise AudioDerivativeError("Audio derivative manifest run binding mismatch.")
    apply_receipt = _read_object(paths["apply_receipt"])
    if (
        apply_receipt.get("run_id") != run_id
        or apply_receipt.get("manifest_sha256") != _sha256_file(paths["manifest"])
    ):
        raise AudioDerivativeError("Audio derivative apply receipt binding mismatch.")
    source = manifest.get("source") or {}
    _source_unchanged(source)
    artifact_path = Path(str(manifest.get("artifact_path") or ""))
    derived_path = Path(str(manifest.get("derived_audio_path") or ""))
    quality_path = Path(str(manifest.get("audio_quality_path") or ""))
    recipe_path = Path(str(manifest.get("recipe_path") or ""))
    _require_private_file(artifact_path, paths["root"])
    if (
        _sha256_file(paths["dry_run"]) != manifest.get("dry_run_sha256")
        or _sha256_file(artifact_path)
        != (manifest.get("derived_audio") or {}).get("output_sha256")
        or _sha256_file(derived_path) != manifest.get("derived_audio_sha256")
        or _sha256_file(quality_path) != manifest.get("audio_quality_sha256")
        or _sha256_file(recipe_path) != manifest.get("recipe_sha256")
    ):
        raise AudioDerivativeError("Audio derivative replay hash mismatch.")
    stored_derived = validate_artifact("derived_audio", _read_object(derived_path))
    stored_quality = validate_artifact("audio_quality", _read_object(quality_path))
    stored_recipe = _read_object(recipe_path)
    if (
        stored_derived != manifest.get("derived_audio")
        or stored_quality != manifest.get("audio_quality")
        or stored_recipe != manifest.get("effective_recipe")
        or stored_derived.get("recipe") != stored_recipe
    ):
        raise AudioDerivativeError("Audio derivative manifest evidence mismatch.")
    timestamp_map = stored_derived.get("timestamp_map") or []
    if len(timestamp_map) != 1:
        raise AudioDerivativeError("P1 requires one full-coverage timestamp map.")
    mapping = timestamp_map[0]
    tolerance = float(mapping.get("duration_tolerance_seconds") or -1)
    if (
        mapping.get("source_start_seconds") != 0.0
        or mapping.get("output_start_seconds") != 0.0
        or mapping.get("source_end_seconds")
        != stored_derived.get("source_duration_seconds")
        or mapping.get("output_end_seconds")
        != stored_derived.get("output_duration_seconds")
        or tolerance != DURATION_TOLERANCE_SECONDS
        or float(mapping.get("measured_duration_drift_seconds") or 0) > tolerance
    ):
        raise AudioDerivativeError("P1 timestamp identity proof is incomplete.")
    metrics = _pcm_metrics(artifact_path)
    if metrics != stored_quality.get("metrics"):
        raise AudioDerivativeError("Audio derivative quality replay mismatch.")
    active = not paths["rollback_receipt"].exists()
    if not active:
        _require_private_file(paths["rollback_receipt"], paths["root"])
        rollback_receipt = _read_object(paths["rollback_receipt"])
        if (
            rollback_receipt.get("run_id") != run_id
            or rollback_receipt.get("manifest_sha256")
            != _sha256_file(paths["manifest"])
            or rollback_receipt.get("eligible_for_use") is not False
        ):
            raise AudioDerivativeError("Audio derivative rollback binding mismatch.")
    replay_path = (
        paths["replay_receipt_active"]
        if active
        else paths["replay_receipt_rolled_back"]
    )
    receipt = {
        "schema_version": REPLAY_RECEIPT_SCHEMA,
        "run_id": run_id,
        "status": "verified_active" if active else "verified_rolled_back",
        "active": active,
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "artifact_sha256": _sha256_file(artifact_path),
        "source_unchanged": True,
        "will_perform_external_write": False,
        "replayed_at": _utc_now(),
    }
    stored = _write_immutable_json(
        replay_path, receipt, volatile_fields=("replayed_at",)
    )
    result = {**stored, "replay_receipt_path": str(replay_path)}
    if include_validated_manifest:
        result["validated_manifest"] = manifest
    return result


def derivative_is_active(
    run_id: str, *, runtime_root: Optional[Path] = None
) -> bool:
    """Resolve authoritative eligibility only after a complete replay."""
    return bool(replay_derivative(run_id, runtime_root=runtime_root)["active"])


def resolve_active_derivative(
    run_id: str, *, runtime_root: Optional[Path] = None
) -> dict[str, Any]:
    """Return a fully replay-validated P1 input reference for host consumers."""
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    replay = replay_derivative(
        run_id, runtime_root=root, include_validated_manifest=True
    )
    if replay["active"] is not True:
        raise AudioDerivativeError("A rolled-back derivative is not consumable.")
    paths = _runtime_paths(root, run_id)
    manifest = replay["validated_manifest"]
    return {
        "run_id": run_id,
        "runtime_root": str(paths["root"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": replay["manifest_sha256"],
        "source_blob_id": manifest["source"]["source_blob_id"],
        "source_sha256": manifest["source"]["sha256"],
        "artifact_path": manifest["artifact_path"],
        "artifact_sha256": manifest["derived_audio"]["output_sha256"],
        "derived_audio": manifest["derived_audio"],
        "audio_quality": manifest["audio_quality"],
        "effective_recipe": manifest["effective_recipe"],
        "recipe_sha256": manifest["recipe_sha256"],
        "derived_audio_sha256": manifest["derived_audio_sha256"],
        "audio_quality_sha256": manifest["audio_quality_sha256"],
    }


def canonical_artifact_hash(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by private runtime identities."""
    return _canonical_hash(value)


def sha256_file(path: Path) -> str:
    """Hash a file for a private evidence binding."""
    return _sha256_file(path)


def utc_now() -> str:
    """Return the canonical UTC audit timestamp."""
    return _utc_now()


def ensure_private_tree(root: Path, leaf: Path) -> None:
    """Create or validate a 0700 private directory tree."""
    _ensure_private_tree(root, leaf)


def require_private_file(path: Path, root: Path) -> None:
    """Require a contained, non-symlinked 0600 private file."""
    _require_private_file(path, root)


def read_private_object(path: Path) -> dict[str, Any]:
    """Read a regular JSON evidence object."""
    return _read_object(path)


def write_immutable_private_json(
    path: Path,
    payload: dict[str, Any],
    *,
    volatile_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Write or replay a no-clobber 0600 JSON evidence object."""
    return _write_immutable_json(
        path, payload, volatile_fields=volatile_fields
    )


def rollback_derivative(
    run_id: str,
    *,
    runtime_root: Optional[Path] = None,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != ROLLBACK_TOKEN:
        raise AudioDerivativeError(
            f"Rollback requires approval token {ROLLBACK_TOKEN}."
        )
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser()
    paths = _runtime_paths(root, run_id)
    replay = replay_derivative(run_id, runtime_root=root)
    if paths["rollback_receipt"].exists() and replay["active"]:
        raise AudioDerivativeError("Rollback state resolver is inconsistent.")
    manifest = _read_object(paths["manifest"])
    _source_unchanged(manifest.get("source") or {})
    artifact_path = Path(str(manifest.get("artifact_path") or ""))
    if not artifact_path.is_file():
        raise AudioDerivativeError("Derived artifact is unavailable for rollback audit.")
    receipt = {
        "schema_version": ROLLBACK_RECEIPT_SCHEMA,
        "run_id": run_id,
        "status": "rolled_back",
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "artifact_path": str(artifact_path),
        "artifact_retained": True,
        "source_retained": True,
        "eligible_for_use": False,
        "will_perform_external_write": False,
        "rolled_back_at": _utc_now(),
    }
    stored = _write_immutable_json(
        paths["rollback_receipt"], receipt, volatile_fields=("rolled_back_at",)
    )
    return {**stored, "rollback_receipt_path": str(paths["rollback_receipt"])}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 0037 P1 audio derivatives")
    parser.add_argument("--runtime-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("source_audio", type=Path)
    dry_parser.add_argument("--source-blob-id")
    dry_parser.add_argument("--expected-source-sha256")
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("source_audio", type=Path)
    apply_parser.add_argument("--approval-token", default="")
    apply_parser.add_argument("--source-blob-id")
    apply_parser.add_argument("--expected-source-sha256")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("run_id")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("run_id")
    rollback_parser.add_argument("--approval-token", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "dry-run":
        result = dry_run(
            args.source_audio,
            runtime_root=args.runtime_root,
            source_blob_id=args.source_blob_id,
            expected_source_sha256=args.expected_source_sha256,
        )
    elif args.command == "apply":
        result = apply_derivative(
            args.source_audio,
            runtime_root=args.runtime_root,
            approval_token=args.approval_token,
            source_blob_id=args.source_blob_id,
            expected_source_sha256=args.expected_source_sha256,
        )
    elif args.command == "replay":
        result = replay_derivative(args.run_id, runtime_root=args.runtime_root)
    else:
        result = rollback_derivative(
            args.run_id,
            runtime_root=args.runtime_root,
            approval_token=args.approval_token,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
