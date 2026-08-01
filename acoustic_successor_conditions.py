"""Measure replayable P1/P2 conditions for the Plan 0037 successor corpus."""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import acoustic_audio_derivatives as audio_derivatives
import acoustic_speech_preparation as speech_preparation
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PLAN_SCHEMA = "transcribe-audio.acoustic-successor-condition-plan.v1"
MANIFEST_SCHEMA = "transcribe-audio.acoustic-successor-condition-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.acoustic-successor-condition-receipt.v1"
FAILURE_SCHEMA = "transcribe-audio.acoustic-successor-condition-failure.v1"
REPLAY_SCHEMA = "transcribe-audio.acoustic-successor-condition-replay.v1"
CORPUS_SCHEMA = "transcribe-audio.acoustic-evaluation-successor-corpus.v1"
EXPECTED_RECORDINGS = 7
METHOD_IDS = (
    "no_enhancement",
    "silero_vad",
    "deepfilternet",
    "rnnoise",
    "pyannote_community_1",
)
CONDITION_FIELDS = (
    "channel",
    "device",
    "noise",
    "telephone_bandwidth",
    "usable_duration_band",
)
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/successor-conditions"
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class SuccessorConditionError(ValueError):
    """Raised when successor condition evidence cannot be trusted."""


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
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _repository_authority() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise SuccessorConditionError("Repository authority is unavailable.")
    return {
        "commit": commit.stdout.strip(),
        "clean": not bool(status.stdout.strip()),
        "module_sha256": sha256_file(Path(__file__).resolve()),
    }


def _validate_repository_authority(authority: Mapping[str, Any]) -> None:
    current = _repository_authority()
    if (
        dict(authority) != current
        or current["clean"] is not True
        or not COMMIT_RE.fullmatch(str(current["commit"]))
    ):
        raise SuccessorConditionError("Repository authority is stale or dirty.")


def _corpus_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "corpus_id",
            "content_sha256",
            "runtime_readback_at_freeze",
            "frozen_at",
        }
    }


def _load_corpus(path: Path) -> tuple[dict[str, Any], str]:
    selected = path.expanduser().resolve(strict=True)
    require_private_file(selected, selected.parent)
    manifest = read_private_object(selected)
    digest = _canonical_hash(_corpus_core(manifest))
    recordings = manifest.get("recordings")
    denominators = manifest.get("denominators") or {}
    split_counts = denominators.get("split_recordings")
    actual_split_counts = (
        dict(Counter(recording.get("split") for recording in recordings))
        if isinstance(recordings, list)
        and all(isinstance(recording, Mapping) for recording in recordings)
        else {}
    )
    expected_split_counts = {
        "development": 3,
        "calibration": 2,
        "evaluation": 2,
    }
    if (
        manifest.get("schema_version") != CORPUS_SCHEMA
        or manifest.get("content_sha256") != digest
        or manifest.get("corpus_id") != f"acoustic-corpus-{digest[:24]}"
        or not isinstance(recordings, list)
        or len(recordings) != EXPECTED_RECORDINGS
        or denominators.get("recordings") != EXPECTED_RECORDINGS
        or split_counts != expected_split_counts
        or actual_split_counts != expected_split_counts
        or manifest.get("prediction_visibility") != "excluded"
        or manifest.get("promotion_eligible") is not False
    ):
        raise SuccessorConditionError("Successor corpus authority is invalid.")
    return manifest, sha256_file(selected)


def _source_unit(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record.get("source_blob")
    if not isinstance(source, Mapping):
        raise SuccessorConditionError("Successor source binding is missing.")
    path = Path(str(source.get("stored_path") or "")).expanduser().resolve(strict=True)
    require_private_file(path, path.parent)
    digest = sha256_file(path)
    if (
        digest != source.get("sha256")
        or path.stat().st_size != source.get("bytes")
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise SuccessorConditionError("Successor source binding drifted.")
    return {
        "recording_id": str(record.get("recording_id") or ""),
        "conversation_id": str(record.get("conversation_id") or ""),
        "split": str(record.get("split") or ""),
        "source_blob_id": str(source.get("blob_id") or ""),
        "source_path": str(path),
        "source_sha256": digest,
        "source_bytes": path.stat().st_size,
    }


def _readiness_authority() -> tuple[dict[str, Any], str]:
    readiness = speech_preparation.readiness_matrix()
    if set(readiness) != set(METHOD_IDS):
        raise SuccessorConditionError("P2 readiness method set is incomplete.")
    if any((readiness.get(method) or {}).get("status") != "success" for method in METHOD_IDS):
        raise SuccessorConditionError("P2 readiness is not fully successful.")
    return readiness, _canonical_hash(readiness)


def preview_condition_campaign(corpus_manifest_path: Path) -> dict[str, Any]:
    """Build an exact no-write/no-model seven-record condition plan."""
    corpus, manifest_sha256 = _load_corpus(corpus_manifest_path)
    readiness, readiness_sha256 = _readiness_authority()
    units = [_source_unit(item) for item in corpus["recordings"]]
    if (
        len({item["recording_id"] for item in units}) != EXPECTED_RECORDINGS
        or len({item["conversation_id"] for item in units}) != EXPECTED_RECORDINGS
        or len({item["source_sha256"] for item in units}) != EXPECTED_RECORDINGS
        or any(item["split"] not in {"development", "calibration", "evaluation"} for item in units)
    ):
        raise SuccessorConditionError("Successor condition units are not disjoint.")
    repository = _repository_authority()
    core = {
        "schema_version": PLAN_SCHEMA,
        "corpus": {
            "corpus_id": corpus["corpus_id"],
            "content_sha256": corpus["content_sha256"],
            "manifest_sha256": manifest_sha256,
            "manifest_path": str(corpus_manifest_path.expanduser().resolve()),
        },
        "repository_authority": repository,
        "module_authority": {
            "p1_sha256": sha256_file(Path(audio_derivatives.__file__).resolve()),
            "p2_sha256": sha256_file(Path(speech_preparation.__file__).resolve()),
            "condition_sha256": sha256_file(Path(__file__).resolve()),
        },
        "readiness_sha256": readiness_sha256,
        "readiness": readiness,
        "units": units,
        "denominators": {
            "recordings": EXPECTED_RECORDINGS,
            "methods_per_recording": len(METHOD_IDS),
            "method_attempts": EXPECTED_RECORDINGS * len(METHOD_IDS),
        },
        "condition_policy": {
            "fields": list(CONDITION_FIELDS),
            "minimum_observed_values_per_field": 2,
            "device_requires_explicit_source_metadata": True,
            "encoding_profile_is_not_device_evidence": True,
            "telephone_candidate_max_source_rate_hz": 16_000,
            "noise_snr_db_bands": {
                "high_noise": "snr_db_below_10",
                "moderate_noise": "snr_db_10_to_below_20",
                "low_noise": "snr_db_20_or_more",
            },
            "usable_speech_bands_seconds": [300, 900],
        },
        "will_process_audio": False,
        "will_run_models": False,
        "will_run_biometrics": False,
        "will_read_private_corpus_gold_authority": True,
        "will_use_gold_for_condition_measurement": False,
        "will_perform_external_write": False,
    }
    digest = _canonical_hash(core)
    return {
        **core,
        "plan_id": f"successor-conditions-{digest[:24]}",
        "content_sha256": digest,
    }


def _merged_regions(
    regions: Iterable[Mapping[str, Any]], duration: float
) -> list[tuple[float, float]]:
    values = sorted(
        (
            max(0.0, float(item.get("start_seconds") or 0.0)),
            min(duration, float(item.get("end_seconds") or 0.0)),
        )
        for item in regions
    )
    merged: list[tuple[float, float]] = []
    for start, end in values:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _energy(path: Path, regions: list[tuple[float, float]]) -> dict[str, Any]:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16_000
        ):
            raise SuccessorConditionError("P1 output is not mono PCM16 at 16 kHz.")
        rate = stream.getframerate()
        frame_count = stream.getnframes()
        total_energy = 0
        total_samples = 0
        while True:
            raw = stream.readframes(65_536)
            if not raw:
                break
            values = array.array("h")
            values.frombytes(raw)
            total_energy += sum(int(value) * int(value) for value in values)
            total_samples += len(values)
        speech_energy = 0
        speech_samples = 0
        for start, end in regions:
            first = min(frame_count, max(0, round(start * rate)))
            last = min(frame_count, max(first, round(end * rate)))
            stream.setpos(first)
            remaining = last - first
            while remaining:
                raw = stream.readframes(min(remaining, 65_536))
                if not raw:
                    break
                values = array.array("h")
                values.frombytes(raw)
                speech_energy += sum(int(value) * int(value) for value in values)
                speech_samples += len(values)
                remaining -= len(values)
    background_samples = total_samples - speech_samples
    background_energy = max(0, total_energy - speech_energy)
    speech_rms = math.sqrt(speech_energy / speech_samples) if speech_samples else 0.0
    background_rms = (
        math.sqrt(background_energy / background_samples)
        if background_samples
        else 0.0
    )
    snr_db = (
        20.0 * math.log10(speech_rms / background_rms)
        if speech_rms > 0 and background_rms > 0
        else None
    )
    return {
        "speech_samples": speech_samples,
        "background_samples": background_samples,
        "speech_rms": speech_rms / 32768.0,
        "background_rms": background_rms / 32768.0,
        "snr_db": snr_db,
    }


def _usable_band(seconds: float) -> str:
    if seconds < 300:
        return "under_5_minutes"
    if seconds < 900:
        return "5_to_under_15_minutes"
    return "15_minutes_or_more"


def _noise_band(snr_db: Optional[float]) -> str:
    if snr_db is None or not math.isfinite(snr_db):
        return "unavailable"
    if snr_db < 10:
        return "high_noise"
    if snr_db < 20:
        return "moderate_noise"
    return "low_noise"


def _encoding_profile(probe: Mapping[str, Any]) -> str:
    projection = {
        "codec": probe.get("codec_name"),
        "sample_rate": probe.get("sample_rate"),
        "channels": probe.get("channels"),
        "layout": probe.get("channel_layout"),
        "format": probe.get("format_name"),
    }
    return f"encoding-profile-{_canonical_hash(projection)[:16]}"


def _conditions(
    p1_manifest: Mapping[str, Any], p2_comparison: Mapping[str, Any]
) -> dict[str, Any]:
    source = p1_manifest.get("source") or {}
    probe = source.get("probe") or {}
    derived = p1_manifest.get("derived_audio") or {}
    methods = {
        str(item.get("method_id") or ""): item
        for item in p2_comparison.get("method_results") or []
        if isinstance(item, Mapping)
    }
    silero = methods.get("silero_vad") or {}
    duration = float(derived.get("output_duration_seconds") or 0.0)
    regions = _merged_regions(silero.get("speech_regions") or [], duration)
    usable_seconds = sum(end - start for start, end in regions)
    energy = _energy(Path(str(p1_manifest.get("artifact_path") or "")), regions)
    explicit_device = str(
        source.get("device_id")
        or source.get("device_model")
        or probe.get("device_id")
        or probe.get("device_model")
        or ""
    ).strip()
    sample_rate = int(probe.get("sample_rate") or 0)
    return {
        "channel": (
            "source_stereo" if int(probe.get("channels") or 0) == 2 else "source_mono"
        ),
        "device": explicit_device or "unavailable_not_reported",
        "device_observed": bool(explicit_device),
        "encoding_profile_proxy": _encoding_profile(probe),
        "telephone_bandwidth": (
            "telephone_bandwidth_candidate"
            if sample_rate <= 16_000
            else "not_telephone_band_limited_by_source_rate"
        ),
        "usable_duration_band": _usable_band(usable_seconds),
        "usable_speech_seconds": usable_seconds,
        "noise": _noise_band(energy["snr_db"]),
        "noise_measurement": energy,
    }


def _aggregate_conditions(units: list[dict[str, Any]]) -> dict[str, Any]:
    observed: dict[str, set[str]] = defaultdict(set)
    missing: dict[str, int] = defaultdict(int)
    for unit in units:
        values = unit["conditions"]
        for field in CONDITION_FIELDS:
            value = str(values.get(field) or "")
            unavailable = value.startswith("unavailable") or (
                field == "device" and values.get("device_observed") is not True
            )
            if unavailable:
                missing[field] += 1
            else:
                observed[field].add(value)
    coverage = {
        field: {
            "observed_values": sorted(observed[field]),
            "observed_value_count": len(observed[field]),
            "missing_recordings": missing[field],
            "status": (
                "pass"
                if len(observed[field]) >= 2 and missing[field] == 0
                else "blocked"
            ),
        }
        for field in CONDITION_FIELDS
    }
    blockers = [
        f"{field}_condition_coverage_below_policy"
        for field in CONDITION_FIELDS
        if coverage[field]["status"] != "pass"
    ]
    return {
        "fields": coverage,
        "terminal_selection_eligible": not blockers,
        "blockers": blockers,
    }


def _runtime_paths(root: Path, plan_id: str) -> dict[str, Path]:
    base = root.expanduser().absolute()
    run = base / "runs" / plan_id
    return {
        "root": base,
        "run": run,
        "manifest": run / "condition-manifest.json",
        "receipt": run / "apply-receipt.json",
        "failure": run / "failure-receipt.json",
        "p1": run / "p1",
        "p2": run / "p2",
    }


def _execute_unit(
    unit: Mapping[str, Any], preview: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    source_path = Path(str(unit["source_path"]))
    source_blob_id = "source-" + str(unit["source_sha256"])[:24]
    corpus_authority = str(preview["corpus"]["content_sha256"])
    p1_plan = audio_derivatives.dry_run(
        source_path,
        runtime_root=paths["p1"],
        source_blob_id=source_blob_id,
        expected_source_sha256=unit["source_sha256"],
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=corpus_authority,
    )
    p1_apply = audio_derivatives.apply_derivative(
        source_path,
        runtime_root=paths["p1"],
        approval_token=audio_derivatives.APPLY_TOKEN,
        source_blob_id=source_blob_id,
        expected_source_sha256=unit["source_sha256"],
        channel_policy="stereo_average_to_mono",
        channel_policy_authority_sha256=corpus_authority,
    )
    p1_replay = audio_derivatives.replay_derivative(
        p1_plan["run_id"], runtime_root=paths["p1"]
    )
    split_authority = None if unit["split"] == "development" else corpus_authority
    p2_plan = speech_preparation.dry_run(
        p1_plan["run_id"],
        p1_runtime_root=paths["p1"],
        runtime_root=paths["p2"],
        intended_split=unit["split"],
        split_access_authority_sha256=split_authority,
    )
    p2_apply = speech_preparation.apply_comparison(
        p1_plan["run_id"],
        p1_runtime_root=paths["p1"],
        runtime_root=paths["p2"],
        intended_split=unit["split"],
        split_access_authority_sha256=split_authority,
    )
    p2_replay = speech_preparation.replay_comparison(
        p2_plan["run_id"], runtime_root=paths["p2"]
    )
    comparison = p2_apply["comparison"]
    method_results = comparison.get("method_results") or []
    if (
        len(method_results) != len(METHOD_IDS)
        or {item.get("method_id") for item in method_results} != set(METHOD_IDS)
        or any(item.get("status") != "success" for item in method_results)
    ):
        raise SuccessorConditionError("P2 condition method matrix is incomplete.")
    return {
        "recording_id": unit["recording_id"],
        "conversation_id": unit["conversation_id"],
        "split": unit["split"],
        "source_sha256": unit["source_sha256"],
        "p1_run_id": p1_plan["run_id"],
        "p1_manifest_path": p1_apply["manifest_path"],
        "p1_manifest_sha256": p1_apply["manifest_sha256"],
        "p1_replay_sha256": sha256_file(Path(p1_replay["replay_receipt_path"])),
        "p1_replay_path": p1_replay["replay_receipt_path"],
        "p2_run_id": p2_plan["run_id"],
        "p2_comparison_path": p2_apply["comparison_path"],
        "p2_comparison_sha256": sha256_file(Path(p2_apply["comparison_path"])),
        "p2_replay_sha256": sha256_file(Path(p2_replay["replay_path"])),
        "p2_replay_path": p2_replay["replay_path"],
        "method_result_sha256": {
            item["method_id"]: _canonical_hash(item) for item in method_results
        },
        "conditions": _conditions(p1_apply["manifest"], comparison),
    }


def apply_condition_campaign(
    corpus_manifest_path: Path,
    *,
    expected_content_sha256: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Run exact seven-by-five P1/P2 condition measurement once."""
    preview = preview_condition_campaign(corpus_manifest_path)
    if preview["content_sha256"] != expected_content_sha256:
        raise SuccessorConditionError("Reviewed condition preview hash is stale.")
    _validate_repository_authority(preview["repository_authority"])
    paths = _runtime_paths(runtime_root or DEFAULT_RUNTIME_ROOT, preview["plan_id"])
    if paths["failure"].exists():
        raise SuccessorConditionError(
            "A prior condition execution failure prevents an unreviewed retry."
        )
    if paths["manifest"].exists() and paths["receipt"].exists():
        return replay_condition_campaign(
            paths["manifest"], corpus_manifest_path=corpus_manifest_path
        )
    if paths["manifest"].exists() or paths["receipt"].exists():
        raise SuccessorConditionError(
            "A partial prior condition finalization prevents an unreviewed retry."
        )
    units = []
    execution_phase = "unit_execution"
    try:
        for unit in preview["units"]:
            units.append(_execute_unit(unit, preview, paths))
        execution_phase = "aggregation"
        aggregate = _aggregate_conditions(units)
        core = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "plan_id": preview["plan_id"],
            "plan_content_sha256": preview["content_sha256"],
            "corpus": preview["corpus"],
            "repository_authority": preview["repository_authority"],
            "module_authority": preview["module_authority"],
            "readiness_sha256": preview["readiness_sha256"],
            "denominators": {
                **preview["denominators"],
                "p1_successes": len(units),
                "p2_method_successes": sum(
                    len(unit["method_result_sha256"]) for unit in units
                ),
            },
            "units": units,
            "condition_coverage": aggregate,
            "did_process_audio": True,
            "did_run_p1_p2": True,
            "did_run_biometrics": False,
            "did_read_private_corpus_gold_authority": True,
            "did_use_gold_for_condition_measurement": False,
            "did_perform_external_write": False,
            "contains_raw_audio": False,
            "contains_transcript_text": False,
            "contains_names_or_emails": False,
            "contains_embeddings_or_vectors": False,
        }
        content_sha256 = _canonical_hash(core)
        manifest = {
            **core,
            "content_sha256": content_sha256,
            "applied_at": _utc_now(),
        }
        ensure_private_tree(paths["root"], paths["run"])
        execution_phase = "manifest_write"
        write_immutable_private_json(
            paths["manifest"], manifest, volatile_fields=("applied_at",)
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "plan_id": preview["plan_id"],
            "manifest_path": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "content_sha256": content_sha256,
            "denominators": manifest["denominators"],
            "condition_coverage": aggregate,
            "mode": "0600",
            "will_perform_external_write": False,
        }
        execution_phase = "receipt_write"
        write_immutable_private_json(paths["receipt"], receipt)
    except Exception as exc:
        ensure_private_tree(paths["root"], paths["run"])
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed",
            "plan_id": preview["plan_id"],
            "plan_content_sha256": preview["content_sha256"],
            "completed_recordings": len(units),
            "attempted_recordings": min(len(units) + 1, EXPECTED_RECORDINGS),
            "execution_phase": execution_phase,
            "failure_type": type(exc).__name__,
            "reason": str(exc),
            "retry_requires_new_review": True,
            "did_run_biometrics": False,
            "did_perform_external_write": False,
            "failed_at": _utc_now(),
        }
        write_immutable_private_json(
            paths["failure"], failure, volatile_fields=("failed_at",)
        )
        raise
    return {**receipt, "receipt_path": str(paths["receipt"]), "idempotent": False}


def replay_condition_campaign(
    manifest_path: Path,
    *,
    corpus_manifest_path: Path,
) -> dict[str, Any]:
    """Read-only full-body replay of condition and P1/P2 lineage evidence."""
    selected = manifest_path.expanduser().resolve(strict=True)
    root = selected.parents[2]
    require_private_file(selected, root)
    manifest = read_private_object(selected)
    preview = preview_condition_campaign(corpus_manifest_path)
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"content_sha256", "applied_at"}
    }
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("plan_id") != preview["plan_id"]
        or manifest.get("plan_content_sha256") != preview["content_sha256"]
        or manifest.get("content_sha256") != _canonical_hash(core)
        or len(manifest.get("units") or []) != EXPECTED_RECORDINGS
    ):
        raise SuccessorConditionError("Condition manifest replay is invalid.")
    for unit in manifest["units"]:
        for path_key, sha_key in (
            ("p1_manifest_path", "p1_manifest_sha256"),
            ("p2_comparison_path", "p2_comparison_sha256"),
        ):
            artifact = Path(str(unit.get(path_key) or "")).resolve(strict=True)
            require_private_file(artifact, root)
            if sha256_file(artifact) != unit.get(sha_key):
                raise SuccessorConditionError("Condition lineage artifact drifted.")
        for path_key, sha_key in (
            ("p1_replay_path", "p1_replay_sha256"),
            ("p2_replay_path", "p2_replay_sha256"),
        ):
            artifact = Path(str(unit.get(path_key) or "")).resolve(strict=True)
            require_private_file(artifact, root)
            if sha256_file(artifact) != unit.get(sha_key):
                raise SuccessorConditionError("Condition replay lineage drifted.")
        p1_manifest = read_private_object(Path(unit["p1_manifest_path"]))
        p2_comparison = read_private_object(Path(unit["p2_comparison_path"]))
        methods = p2_comparison.get("method_results") or []
        if {
            str(item.get("method_id") or ""): _canonical_hash(item)
            for item in methods
            if isinstance(item, Mapping)
        } != unit.get("method_result_sha256"):
            raise SuccessorConditionError("Condition method result lineage drifted.")
        for method in methods:
            output = Path(str(method.get("output_path") or "")).resolve(strict=True)
            require_private_file(output, root)
            if sha256_file(output) != method.get("output_sha256"):
                raise SuccessorConditionError("Condition method output drifted.")
        if _conditions(p1_manifest, p2_comparison) != unit.get("conditions"):
            raise SuccessorConditionError("Condition measurement replay drifted.")
    if _aggregate_conditions(manifest["units"]) != manifest.get("condition_coverage"):
        raise SuccessorConditionError("Condition coverage replay drifted.")
    receipt_path = selected.parent / "apply-receipt.json"
    require_private_file(receipt_path, root)
    receipt = read_private_object(receipt_path)
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "plan_id": preview["plan_id"],
        "manifest_path": str(selected),
        "manifest_sha256": sha256_file(selected),
        "content_sha256": manifest["content_sha256"],
        "denominators": manifest["denominators"],
        "condition_coverage": manifest["condition_coverage"],
        "mode": "0600",
        "will_perform_external_write": False,
    }
    if receipt != expected_receipt:
        raise SuccessorConditionError("Condition receipt replay is invalid.")
    return {
        "schema_version": REPLAY_SCHEMA,
        "plan_id": preview["plan_id"],
        "manifest_sha256": expected_receipt["manifest_sha256"],
        "content_sha256": manifest["content_sha256"],
        "condition_coverage": manifest["condition_coverage"],
        "full_body_match": True,
        "idempotent": True,
        "will_perform_external_write": False,
    }
