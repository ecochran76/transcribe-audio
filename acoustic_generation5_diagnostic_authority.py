"""Plan 0053 G0 sealed duration-diagnostic development and holdout authority."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation3_preparation_stop as generation3
import acoustic_generation4_cohort as cohort
import acoustic_generation4_freeze as generation4_freeze
import acoustic_generation4_terminal as generation4_terminal
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-diagnostic-authority-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-diagnostic-authority-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-diagnostic-authority-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-diagnostic-authority-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0053/g0")
MODULE_NAME = Path(__file__).name
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
PLAN0052_G2_PREVIEW_SHA256 = generation4_terminal.G2_PREVIEW_SHA256
PLAN0052_G2_MANIFEST_SHA256 = generation4_terminal.G2_MANIFEST_SHA256
PLAN0052_TERMINAL_PREVIEW_SHA256 = (
    "2f7f228189072dfb90344c916c2e104d0d4836ea613cd0f081f7e9109e33fc17"
)
PLAN0052_TERMINAL_MANIFEST_SHA256 = (
    "7600629721bfcedcf3e6a1f708164fe4600441f8241874873a473e00080fe702"
)
PLAN0051_QUALIFIED_SET_SHA256 = cohort.QUALIFIED_SET_SHA256
DEVELOPMENT_CONTROL_COUNT = 3
HOLDOUT_COUNT = 7


class Generation5DiagnosticAuthorityError(ValueError):
    """Raised when G0 diagnostic authority cannot remain exact and sealed."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5DiagnosticAuthorityError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5DiagnosticAuthorityError("Private authority must be an object.")
    return value


def _git(args: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        raise Generation5DiagnosticAuthorityError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5DiagnosticAuthorityError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5DiagnosticAuthorityError("Repository must be upstream-even.")
    commit = str(_git(["log", "-1", "--format=%H", "--", MODULE_NAME]))
    if not COMMIT_RE.fullmatch(commit) or _git(["merge-base", "--is-ancestor", commit, "HEAD"]) != "":
        raise Generation5DiagnosticAuthorityError("Module commit is not an ancestor.")
    blob = _git(["show", f"{commit}:{MODULE_NAME}"], binary=True)
    if not isinstance(blob, bytes):
        raise Generation5DiagnosticAuthorityError("Committed module body is unavailable.")
    module_sha256 = hashlib.sha256(blob).hexdigest()
    if module_sha256 != sha256_file(Path(__file__).resolve()):
        raise Generation5DiagnosticAuthorityError("Committed module body drifted.")
    return {
        "commit": commit,
        "module_name": MODULE_NAME,
        "module_sha256": module_sha256,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _plan0052_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    g2_paths = generation4_freeze._paths(
        generation4_freeze.DEFAULT_RUNTIME_ROOT, PLAN0052_G2_PREVIEW_SHA256
    )
    terminal_paths = generation4_terminal._paths(
        generation4_terminal.DEFAULT_RUNTIME_ROOT, PLAN0052_TERMINAL_PREVIEW_SHA256
    )
    for paths in (g2_paths, terminal_paths):
        require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    g2_manifest = _read_json(g2_paths["manifest"])
    terminal_manifest = _read_json(terminal_paths["manifest"])
    g2_preview = g2_manifest.get("preview")
    terminal_preview = terminal_manifest.get("preview")
    if (
        sha256_file(g2_paths["manifest"]) != PLAN0052_G2_MANIFEST_SHA256
        or not isinstance(g2_preview, Mapping)
        or g2_preview.get("content_sha256") != PLAN0052_G2_PREVIEW_SHA256
        or sha256_file(terminal_paths["manifest"]) != PLAN0052_TERMINAL_MANIFEST_SHA256
        or not isinstance(terminal_preview, Mapping)
        or terminal_preview.get("content_sha256") != PLAN0052_TERMINAL_PREVIEW_SHA256
        or terminal_preview.get("terminal_decision") != "stop"
    ):
        raise Generation5DiagnosticAuthorityError("Plan 0052 authority drifted.")
    return dict(g2_preview), dict(terminal_preview)


def _generation3_failure() -> dict[str, Any]:
    paths = generation3._paths()
    generation3._validate_parent(paths)
    state = generation3._partial_state(paths)
    failed = paths["run"] / "p1" / "runs" / generation3.EXPECTED_FAILED_RUN_ID / "dry-run.json"
    plan = _read_json(failed)
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise Generation5DiagnosticAuthorityError("Generation-3 failure source is missing.")
    return _private_member(
        source_sha256=str(source.get("sha256") or ""),
        path=str(source.get("path") or ""),
        role="known_failure",
        origin="generation3_terminal_stop",
        prior_reason_code=generation3.EXPECTED_REASON_CODE,
        prior_drift_seconds=state["duration_drift_seconds"],
    )


def _generation4_failure(
    g2_preview: Mapping[str, Any], terminal_preview: Mapping[str, Any]
) -> dict[str, Any]:
    member = generation4_terminal._failed_member(g2_preview)
    terminal = terminal_preview.get("private_evidence")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("failed_source_sha256") != member.get("source_sha256")
        or terminal.get("duration_drift_seconds") != 0.17397950000031415
    ):
        raise Generation5DiagnosticAuthorityError("Generation-4 failure evidence drifted.")
    return _private_member(
        source_sha256=str(member.get("source_sha256") or ""),
        path=str(member.get("source_path") or ""),
        role="known_failure",
        origin="generation4_terminal_stop",
        prior_reason_code="p1_duration_drift_exceeds_frozen_tolerance",
        prior_drift_seconds=terminal["duration_drift_seconds"],
    )


def _private_member(
    *, source_sha256: str, path: str, role: str, origin: str,
    prior_reason_code: str | None = None, prior_drift_seconds: float | None = None,
) -> dict[str, Any]:
    source_path = Path(path)
    if (
        not SHA256_RE.fullmatch(source_sha256)
        or not source_path.is_file()
        or source_path.is_symlink()
        or sha256_file(source_path) != source_sha256
    ):
        raise Generation5DiagnosticAuthorityError("Diagnostic source binding drifted.")
    return {
        "source_sha256": source_sha256,
        "path": str(source_path.resolve()),
        "role": role,
        "authority_origin": origin,
        "prior_reason_code": prior_reason_code,
        "prior_drift_seconds": prior_drift_seconds,
    }


def _plan0051_split() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qualified = cohort._media_membership()
    ordered = sorted(qualified, key=lambda item: str(item.get("source_sha256") or ""))
    if (
        len(ordered) != DEVELOPMENT_CONTROL_COUNT + HOLDOUT_COUNT
        or _canonical_hash([str(item.get("source_sha256") or "") for item in ordered])
        != PLAN0051_QUALIFIED_SET_SHA256
    ):
        raise Generation5DiagnosticAuthorityError("Plan 0051 qualified set drifted.")
    members = [
        _private_member(
            source_sha256=str(item.get("source_sha256") or ""),
            path=str(item.get("path") or ""),
            role="healthy_control" if index < DEVELOPMENT_CONTROL_COUNT else "sealed_holdout",
            origin="plan0051_qualified_media",
        )
        for index, item in enumerate(ordered)
    ]
    return members[:DEVELOPMENT_CONTROL_COUNT], members[DEVELOPMENT_CONTROL_COUNT:]


def _measurement_contract() -> dict[str, Any]:
    return {
        "hypotheses": [
            "container_or_stream_timeline_differs_from_decodable_sample_extent",
            "start_time_edit_list_codec_delay_or_discard_padding",
            "packet_timestamp_gap_or_discontinuity",
            "resampling_rounding_or_filter_delay",
            "corrupt_missing_or_undecodable_packets",
            "wrong_stream_or_early_decoder_termination",
        ],
        "metadata_fields": [
            "format.duration", "format.start_time", "stream.duration",
            "stream.duration_ts", "stream.start_time", "stream.start_pts",
            "stream.time_base", "stream.codec_name", "stream.sample_rate",
            "stream.channels", "stream.initial_padding", "stream.trailing_padding",
        ],
        "packet_fields": [
            "pts", "pts_time", "dts", "dts_time", "duration", "duration_time",
            "flags", "side_data_list",
        ],
        "decoded_fields": [
            "input_sample_count", "input_last_pts", "output_frame_count",
            "output_sample_rate", "decode_error_count", "resampler_bound_samples",
            "canonical_pcm_fingerprint", "aligned_content_fingerprint",
        ],
        "reference_paths": [
            "source_packet_timestamp_accounting",
            "source_decode_to_null_frame_sample_count",
            "produced_wav_frame_count",
            "aligned_content_or_canonical_pcm_fingerprint",
        ],
        "negative_variant_families": {
            "development": {
                "seed": "generation5-duration-development-v1",
                "severities": [
                    "derived_bound_plus_one_sample", "twenty_milliseconds",
                    "two_hundred_fifty_milliseconds", "one_second",
                ],
            },
            "holdout": {
                "seed": "generation5-duration-holdout-v1",
                "severities": [
                    "derived_bound_plus_one_sample", "fifty_milliseconds",
                    "five_hundred_milliseconds", "two_seconds",
                ],
            },
            "fault_classes": [
                "tail_truncation", "packet_removal", "corrupt_tail",
                "timestamp_discontinuity", "wrong_stream",
            ],
        },
        "development_only_before_j1": True,
        "holdout_measurement_before_j1": False,
        "threshold_may_use_observed_maximum": False,
        "threshold_may_name_a_source": False,
        "container_duration_is_decision_authority": False,
    }


def preview_generation5_diagnostic_authority(
    *, repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    g2_preview, terminal_preview = _plan0052_authority()
    controls, holdout = _plan0051_split()
    failures = [
        _generation3_failure(),
        _generation4_failure(g2_preview, terminal_preview),
    ]
    development = failures + controls
    all_members = development + holdout
    hashes = [str(item["source_sha256"]) for item in all_members]
    if len(hashes) != 12 or len(set(hashes)) != 12:
        raise Generation5DiagnosticAuthorityError("Diagnostic membership overlaps.")
    actions = {
        "run_g1_development_diagnosis": True,
        "measure_holdout": False,
        "enumerate_generation5_candidates": False,
        "reveal_gold": False,
        "run_predictions": False,
        "load_or_run_biometric_models": False,
        "score": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {"development": development, "holdout": holdout}
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "sealed_diagnostic_membership",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "plan0051_qualified_set_sha256": PLAN0051_QUALIFIED_SET_SHA256,
        "plan0052_g2_preview_sha256": PLAN0052_G2_PREVIEW_SHA256,
        "plan0052_terminal_preview_sha256": PLAN0052_TERMINAL_PREVIEW_SHA256,
        "generation3_failure_source_sha256": generation3.EXPECTED_FAILED_SOURCE_SHA256,
        "development_count": len(development),
        "known_failure_count": len(failures),
        "healthy_control_count": len(controls),
        "holdout_count": len(holdout),
        "development_set_sha256": _canonical_hash(sorted(item["source_sha256"] for item in development)),
        "holdout_set_sha256": _canonical_hash(sorted(item["source_sha256"] for item in holdout)),
        "diagnostic_set_sha256": _canonical_hash(sorted(hashes)),
        "measurement_contract": _measurement_contract(),
        "measurement_contract_sha256": _canonical_hash(_measurement_contract()),
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": True,
        "contains_private_membership": True,
        "did_decode_audio": False,
        "did_measure_holdout": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
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
    run = root / f"generation5-diagnostic-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation5_diagnostic_authority(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_diagnostic_authority()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5DiagnosticAuthorityError("Reviewed diagnostic preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_diagnostic_authority(expected_content_sha256, runtime_root=runtime_root)
    ensure_private_tree(paths["root"], paths["run"])
    manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_generation5_diagnostic_authority(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_diagnostic_authority()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation5DiagnosticAuthorityError("Diagnostic authority drifted.")
    paths = _paths(runtime_root, expected_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = _read_json(paths["manifest"])
    receipt = _read_json(paths["receipt"])
    expected_manifest = {"schema_version": MANIFEST_SCHEMA, "status": "frozen", "preview": preview}
    expected_receipt = {**_portable(preview), "manifest_sha256": sha256_file(paths["manifest"]), "mode": "0600"}
    if manifest != expected_manifest or receipt != expected_receipt:
        raise Generation5DiagnosticAuthorityError("Diagnostic body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
