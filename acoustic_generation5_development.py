"""Plan 0053 G1 development diagnosis and proposed validation contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_content_preservation as preservation
import acoustic_content_preservation_adversarial as adversarial
import acoustic_generation5_diagnostic_authority as g0
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation5-development-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation5-development-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation5-development-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation5-development-replay.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0053/g1")
G0_PREVIEW_SHA256 = "5f765a67810bc4cb58c9c3a8d78aaa25aba4a67650e4fd242d456b9a54d55096"
G0_MANIFEST_SHA256 = "4c85505c18b12d6acf939d1cbe2dfa1f5d1e37de03fba40c99fd1b86d37dd818"
BOUND_MODULES = (
    "acoustic_content_preservation.py",
    "acoustic_content_preservation_adversarial.py",
    "acoustic_generation5_development.py",
)
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation5DevelopmentError(ValueError):
    """Raised when the G1 diagnosis or contract cannot be frozen exactly."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Generation5DevelopmentError("Private authority is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation5DevelopmentError("Private authority must be an object.")
    return value


def _git(arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=not binary, check=False,
    )
    if result.returncode:
        raise Generation5DevelopmentError("Repository authority is unavailable.")
    return result.stdout if binary else result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    if _git(["status", "--porcelain=v1", "--untracked-files=normal"]):
        raise Generation5DevelopmentError("Repository must be clean.")
    if str(_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])).split() != ["0", "0"]:
        raise Generation5DevelopmentError("Repository must be upstream-even.")
    commit = str(_git(["rev-parse", "HEAD"]))
    if not COMMIT_RE.fullmatch(commit):
        raise Generation5DevelopmentError("Repository commit is invalid.")
    module_hashes: dict[str, str] = {}
    for name in BOUND_MODULES:
        body = _git(["show", f"{commit}:{name}"], binary=True)
        if not isinstance(body, bytes):
            raise Generation5DevelopmentError("Committed module body is unavailable.")
        digest = hashlib.sha256(body).hexdigest()
        if digest != sha256_file(Path(__file__).resolve().parent / name):
            raise Generation5DevelopmentError("Committed module body drifted.")
        module_hashes[name] = digest
    return {
        "commit": commit,
        "module_sha256": module_hashes,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _g0_preview() -> dict[str, Any]:
    paths = g0._paths(g0.DEFAULT_RUNTIME_ROOT, G0_PREVIEW_SHA256)
    require_private_file(paths["manifest"], paths["root"].expanduser().absolute())
    if sha256_file(paths["manifest"]) != G0_MANIFEST_SHA256:
        raise Generation5DevelopmentError("G0 manifest drifted.")
    manifest = _read_json(paths["manifest"])
    preview = manifest.get("preview")
    if not isinstance(preview, Mapping) or preview.get("content_sha256") != G0_PREVIEW_SHA256:
        raise Generation5DevelopmentError("G0 preview drifted.")
    if preview.get("did_measure_holdout") is not False:
        raise Generation5DevelopmentError("G0 holdout isolation drifted.")
    return dict(preview)


def _collect_measurements(g0_preview: Mapping[str, Any]) -> list[dict[str, Any]]:
    private = g0_preview.get("private_evidence")
    members = private.get("development") if isinstance(private, Mapping) else None
    if not isinstance(members, list) or len(members) != 5:
        raise Generation5DevelopmentError("G0 development membership is invalid.")
    results = []
    for member in members:
        if not isinstance(member, Mapping):
            raise Generation5DevelopmentError("G0 development member is invalid.")
        measurement = preservation.measure(
            Path(str(member.get("path") or "")),
            expected_source_sha256=str(member.get("source_sha256") or ""),
            channel_policy_authority_sha256=G0_PREVIEW_SHA256,
        )
        results.append(
            {
                "authority_origin": member.get("authority_origin"),
                "role": member.get("role"),
                "source_sha256": member.get("source_sha256"),
                "measurement": measurement,
            }
        )
    return results


def _collect_adversaries(g0_preview: Mapping[str, Any]) -> dict[str, Any]:
    members = g0_preview["private_evidence"]["development"]
    controls = sorted(
        (member for member in members if member.get("role") == "healthy_control"),
        key=lambda member: str(member.get("source_sha256") or ""),
    )
    if len(controls) != 3:
        raise Generation5DevelopmentError("Development controls are invalid.")
    source = controls[0]
    return adversarial.run_development_adversaries(
        Path(str(source["path"])),
        expected_source_sha256=str(source["source_sha256"]),
        channel_policy_authority_sha256=G0_PREVIEW_SHA256,
    )


def _validate_development(results: list[dict[str, Any]]) -> None:
    if len(results) != 5:
        raise Generation5DevelopmentError("Development result count is invalid.")
    origins = [str(item.get("authority_origin") or "") for item in results]
    if origins.count("generation3_terminal_stop") != 1 or origins.count("generation4_terminal_stop") != 1 or origins.count("plan0051_qualified_media") != 3:
        raise Generation5DevelopmentError("Development result origins drifted.")
    for item in results:
        measurement = item.get("measurement")
        if not isinstance(measurement, Mapping):
            raise Generation5DevelopmentError("Development measurement is invalid.")
        origin = item["authority_origin"]
        if origin == "generation3_terminal_stop":
            if measurement.get("status") != "rejected" or "timeline_discontinuity" not in (measurement.get("reason_codes") or []):
                raise Generation5DevelopmentError("Generation-3 diagnosis is unsupported.")
        elif measurement.get("status") != "passing" or measurement.get("reason_codes") != []:
            raise Generation5DevelopmentError("Expected complete development content did not pass.")
        if measurement.get("output_sample_error") != 0:
            raise Generation5DevelopmentError("Development resampling arithmetic drifted.")
        if (measurement.get("recipe_reference_decode") or {}).get("pcm_sha256") != (measurement.get("production_wav") or {}).get("pcm_sha256"):
            raise Generation5DevelopmentError("Development content fingerprint drifted.")


def preview_generation5_development(
    *,
    g0_preview: Mapping[str, Any] | None = None,
    measurements: list[dict[str, Any]] | None = None,
    adversarial_result: Mapping[str, Any] | None = None,
    repository_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = dict(g0_preview or _g0_preview())
    results = list(measurements if measurements is not None else _collect_measurements(authority))
    negative = dict(adversarial_result or _collect_adversaries(authority))
    _validate_development(results)
    if negative.get("case_count") != 11 or negative.get("all_expected_rejections_observed") is not True:
        raise Generation5DevelopmentError("Development adversarial family did not pass.")
    public_negative = {
        key: value for key, value in negative.items()
        if key not in {"private_fixture_hashes", "private_case_measurements"}
    }
    diagnosis = {
        "container_duration_is_not_decodable_sample_authority": True,
        "generation3_cause": "packet_timestamp_discontinuity_collapsed_by_decode",
        "generation4_cause": "continuous_container_clock_cadence_differs_from_aac_sample_extent",
        "generation4_content_loss_observed": False,
        "comparison_authority": "packet_and_decode_sample_extent_plus_exact_pcm_content",
        "maximum_resampler_error_samples": preservation.MAX_RESAMPLER_ERROR_SAMPLES,
        "timeline_discontinuity_policy": "reject",
        "case_fitted_constant_used": False,
    }
    actions = {
        "submit_to_j1": True,
        "measure_positive_holdout": False,
        "instantiate_heldout_negative_family": False,
        "enumerate_generation5_candidates": False,
        "reveal_gold": False,
        "run_predictions_or_models": False,
        "mutate_profiles_or_references": False,
        "enable_default_integration": False,
        "run_historical_reprocessing": False,
    }
    private = {
        "development_measurements": results,
        "development_adversarial": negative,
    }
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_j1_review",
        "repository_authority": dict(repository_authority or _repository_authority()),
        "g0_preview_sha256": G0_PREVIEW_SHA256,
        "g0_manifest_sha256": G0_MANIFEST_SHA256,
        "contract": preservation.contract(),
        "contract_sha256": preservation.contract()["content_sha256"],
        "diagnosis": diagnosis,
        "development_count": len(results),
        "development_pass_count": sum(item["measurement"]["status"] == "passing" for item in results),
        "development_reject_count": sum(item["measurement"]["status"] == "rejected" for item in results),
        "development_results_sha256": _canonical_hash(results),
        "development_adversarial": public_negative,
        "action_vector": actions,
        "private_evidence": private,
        "contains_paths": False,
        "contains_private_membership": True,
        "did_measure_holdout": False,
        "did_instantiate_heldout_negative_family": False,
        "did_access_gold": False,
        "did_load_or_run_models": False,
    }
    return {**core, "content_sha256": _canonical_hash(core)}


def _portable(preview: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: item for key, item in preview.items()
        if key not in {"private_evidence", "repository_authority"}
    }
    value["schema_version"] = RECEIPT_SCHEMA
    value["contains_private_membership"] = False
    return value


def _paths(runtime_root: Path, content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"generation5-development-{content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def apply_generation5_development(
    reviewed_preview: Mapping[str, Any], *, expected_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_development()
    if dict(reviewed_preview) != preview or preview["content_sha256"] != expected_content_sha256:
        raise Generation5DevelopmentError("Reviewed G1 preview is stale.")
    paths = _paths(runtime_root, expected_content_sha256)
    if paths["manifest"].exists() or paths["receipt"].exists():
        return replay_generation5_development(expected_content_sha256, runtime_root=runtime_root)
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


def replay_generation5_development(
    expected_content_sha256: str, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    preview = preview_generation5_development()
    if preview["content_sha256"] != expected_content_sha256:
        raise Generation5DevelopmentError("G1 authority drifted.")
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
        raise Generation5DevelopmentError("G1 body or receipt drifted.")
    return {**receipt, "replay_schema_version": REPLAY_SCHEMA, "idempotent_replay": True}
