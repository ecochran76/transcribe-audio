"""Fail-closed verification-model acquisition evidence for Plan 0037 P4.

This module does not read audio, execute models, or authorize biometric
enrollment.  It records and replays the exact acquisition proposal that later
P4 work may apply under the operator's bounded acquisition grant.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)


ACQUISITION_PLAN_SCHEMA = (
    "transcribe-audio.verification-model-acquisition-dry-run.v1"
)
ACQUISITION_SPEC_SCHEMA = (
    "transcribe-audio.verification-model-acquisition-plan.v1"
)
AUTHORIZATION_BASIS = "operator_blanket_2026-07-31"
AUTHORIZATION_SCOPE = (
    "plan_0037_model_acquisition_install_and_development_processing_only"
)
DEFAULT_ACQUISITION_SPEC = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration/acquisitions"
)
RUN_ID_RE = re.compile(r"acquire-verification-[a-f0-9]{24}")
SHA256_RE = re.compile(r"[a-f0-9]{64}")
REVISION_RE = re.compile(r"[a-f0-9]{40}")

EXPECTED_CANDIDATES = {
    "speechbrain_ecapa_tdnn": {
        "repo_id": "speechbrain/spkrec-ecapa-voxceleb",
        "revision_sha": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "artifact_paths": (
            "classifier.ckpt",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "hyperparams.yaml",
            "label_encoder.txt",
        ),
    },
    "wespeaker_campplus": {
        "repo_id": "Wespeaker/wespeaker-voxceleb-campplus",
        "revision_sha": "acf623ad8ca746e50baa432255cf8fc57c669c45",
        "artifact_paths": ("voxceleb_CAM++.onnx", "config.yaml"),
    },
    "wespeaker_resnet34": {
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet34",
        "revision_sha": "ff1ac5bca8ef11e90662b879aa923979e0bd277b",
        "artifact_paths": ("voxceleb_resnet34.onnx", "config.yaml"),
    },
}
EXPECTED_EXCLUSIONS = {
    "private_audio_read",
    "real_reference_registration",
    "embedding_materialization",
    "development_trial_execution",
    "calibration_read",
    "evaluation_read",
    "plan_0036_prediction_read",
    "external_write",
}
SIDE_EFFECT_FIELDS = (
    "will_download",
    "will_install",
    "will_build",
    "will_read_audio",
    "will_materialize_embeddings",
    "will_register_references",
    "will_run_trials",
    "will_perform_external_write",
)


class AcousticVerificationError(ValueError):
    """Raised when P4 verification acquisition evidence fails closed."""


def _load_spec(path: Path) -> dict[str, Any]:
    selected = path.expanduser().absolute()
    if selected.is_symlink() or not selected.is_file():
        raise AcousticVerificationError("Verification acquisition spec is unavailable.")
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError(
            "Verification acquisition spec is unreadable."
        ) from exc
    return _validate_spec(value)


def _validate_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcousticVerificationError("Verification acquisition spec is invalid.")
    if value.get("schema_version") != ACQUISITION_SPEC_SCHEMA:
        raise AcousticVerificationError("Verification acquisition schema is invalid.")
    if value.get("authorization_basis") != AUTHORIZATION_BASIS:
        raise AcousticVerificationError("Verification acquisition authority is invalid.")
    if value.get("authorization_scope") != AUTHORIZATION_SCOPE:
        raise AcousticVerificationError("Verification acquisition scope is invalid.")
    if value.get("real_biometric_enrollment_authorized") is not False:
        raise AcousticVerificationError(
            "Verification acquisition cannot carry real enrollment authority."
        )
    authorities = value.get("source_authorities")
    bindings = value.get("wespeaker_authority_bindings")
    if (
        not isinstance(authorities, dict)
        or authorities.get("wespeaker_code")
        != "https://github.com/wenet-e2e/wespeaker/archive/dfa741957e5c11f477623b6e583d67d0af25ee88.tar.gz"
        or authorities.get("wespeaker_models")
        != "https://raw.githubusercontent.com/wenet-e2e/wespeaker/dfa741957e5c11f477623b6e583d67d0af25ee88/docs/pretrained.md"
        or bindings
        != {
            "code_archive_size_bytes": 343032,
            "code_archive_sha256": "160b268ea4020b7c9ee52ceb79ad7a8e663d7e21ec763cbd72aef00290f140ec",
            "pretrained_terms_size_bytes": 13382,
            "pretrained_terms_sha256": "34a46fc9faeb6a5c8204c840a06b1ca94fc5ac3c52d5460f5bd6c1bf9aa701cf",
            "checkpoint_terms": "CC-BY-4.0",
        }
    ):
        raise AcousticVerificationError(
            "WeSpeaker code and terms authority binding is invalid."
        )
    candidates = value.get("candidates")
    if (
        not isinstance(candidates, list)
        or [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
        != list(EXPECTED_CANDIDATES)
    ):
        raise AcousticVerificationError(
            "Verification candidate inventory is incomplete or unordered."
        )
    for candidate in candidates:
        expected = EXPECTED_CANDIDATES[candidate["candidate_id"]]
        runtime = candidate.get("runtime")
        if not isinstance(runtime, dict):
            raise AcousticVerificationError("Verification runtime descriptor is invalid.")
        if candidate["candidate_id"] == "speechbrain_ecapa_tdnn" and (
            runtime.get("distribution") != "speechbrain"
            or runtime.get("version") != "1.1.0"
            or runtime.get("artifact")
            != "speechbrain-1.1.0-py3-none-any.whl"
            or not str(runtime.get("url", "")).startswith(
                "https://files.pythonhosted.org/"
            )
            or runtime.get("size_bytes") != 2_278_632
            or runtime.get("sha256")
            != "0f1bc7d5c5ce07b9ed752a9d931a4858180f825f4d079b44035a0aed645f4dd2"
        ):
            raise AcousticVerificationError("SpeechBrain runtime pin is invalid.")
        model = candidate.get("model")
        if not isinstance(model, dict):
            raise AcousticVerificationError("Verification model descriptor is invalid.")
        if (
            model.get("repo_id") != expected["repo_id"]
            or model.get("revision_sha") != expected["revision_sha"]
            or not REVISION_RE.fullmatch(str(model.get("revision_sha", "")))
            or model.get("gated") is not False
            or model.get("missing_upstream_sha256_policy")
            != "compute_and_bind_before_load"
        ):
            raise AcousticVerificationError("Verification model pin is invalid.")
        artifacts = model.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or tuple(
                item.get("path") for item in artifacts if isinstance(item, dict)
            )
            != expected["artifact_paths"]
        ):
            raise AcousticVerificationError(
                "Verification artifact inventory is incomplete or unordered."
            )
        for artifact in artifacts:
            digest = artifact.get("sha256")
            if (
                not isinstance(artifact.get("size_bytes"), int)
                or artifact["size_bytes"] <= 0
                or (digest is not None and not SHA256_RE.fullmatch(str(digest)))
            ):
                raise AcousticVerificationError(
                    "Verification artifact binding is invalid."
                )
    if set(value.get("exclusions", [])) != EXPECTED_EXCLUSIONS:
        raise AcousticVerificationError("Verification acquisition exclusions are invalid.")
    deferred = value.get("deferred_candidates")
    if (
        not isinstance(deferred, list)
        or len(deferred) != 1
        or deferred[0].get("candidate_id") != "nvidia_titanet"
        or deferred[0].get("status") != "deferred_not_acquired"
    ):
        raise AcousticVerificationError("Deferred candidate policy is invalid.")
    return value


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "installed_distributions": {
            name: _package_version(name)
            for name in ("speechbrain", "onnxruntime", "torch", "torchaudio")
        },
    }


def _paths(root: Path, run_id: str) -> dict[str, Path]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise AcousticVerificationError("Verification acquisition run ID is invalid.")
    selected = root.expanduser().absolute()
    run_dir = selected / "acquisition-plans" / run_id
    return {
        "root": selected,
        "run_dir": run_dir,
        "dry_run": run_dir / "dry-run.json",
    }


def dry_run_model_acquisition(
    *,
    runtime_root: Optional[Path] = None,
    spec_path: Path = DEFAULT_ACQUISITION_SPEC,
) -> dict[str, Any]:
    """Persist an immutable, side-effect-free P4 acquisition proposal."""
    selected_spec = spec_path.expanduser().absolute()
    spec = _load_spec(selected_spec)
    spec_sha = sha256_file(selected_spec)
    host = _host_snapshot()
    identity = {
        "spec_path": str(selected_spec),
        "spec_sha256": spec_sha,
        "host": host,
        "authorization_basis": AUTHORIZATION_BASIS,
    }
    run_id = "acquire-verification-" + canonical_artifact_hash(identity)[:24]
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, run_id)
    plan = {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "authorization_basis": AUTHORIZATION_BASIS,
        "spec_path": str(selected_spec),
        "spec_sha256": spec_sha,
        "spec": spec,
        "host": host,
        "runtime_root": str(paths["root"]),
        **{field: False for field in SIDE_EFFECT_FIELDS},
        "created_at": utc_now(),
    }
    ensure_private_tree(paths["root"], paths["run_dir"])
    stored = write_immutable_private_json(
        paths["dry_run"], plan, volatile_fields=("created_at",)
    )
    return {
        **stored,
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": sha256_file(paths["dry_run"]),
    }


def replay_model_acquisition(
    run_id: str,
    *,
    expected_dry_run_sha256: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay and validate a reviewed P4 acquisition proposal."""
    paths = _paths(runtime_root or DEFAULT_RUNTIME_ROOT, run_id)
    require_private_file(paths["dry_run"], paths["root"])
    if (
        not SHA256_RE.fullmatch(expected_dry_run_sha256)
        or sha256_file(paths["dry_run"]) != expected_dry_run_sha256
    ):
        raise AcousticVerificationError(
            "Verification acquisition dry-run hash mismatch."
        )
    plan = read_private_object(paths["dry_run"])
    required_keys = {
        "schema_version", "run_id", "status", "reason_code",
        "authorization_basis", "spec_path", "spec_sha256", "spec", "host",
        "runtime_root", "created_at", *SIDE_EFFECT_FIELDS,
    }
    if (
        set(plan) != required_keys
        or plan.get("schema_version") != ACQUISITION_PLAN_SCHEMA
        or plan.get("run_id") != run_id
        or plan.get("status") != "success"
        or plan.get("reason_code") is not None
        or plan.get("authorization_basis") != AUTHORIZATION_BASIS
        or plan.get("runtime_root") != str(paths["root"])
        or any(plan.get(field) is not False for field in SIDE_EFFECT_FIELDS)
    ):
        raise AcousticVerificationError("Verification acquisition plan is invalid.")
    spec_path = Path(str(plan["spec_path"]))
    try:
        current_spec = _load_spec(spec_path)
    except AcousticVerificationError as exc:
        raise AcousticVerificationError(
            "Verification acquisition spec drifted."
        ) from exc
    if (
        sha256_file(spec_path) != plan.get("spec_sha256")
        or current_spec != plan.get("spec")
    ):
        raise AcousticVerificationError("Verification acquisition spec drifted.")
    identity = {
        "spec_path": str(spec_path),
        "spec_sha256": plan["spec_sha256"],
        "host": plan["host"],
        "authorization_basis": plan["authorization_basis"],
    }
    if run_id != "acquire-verification-" + canonical_artifact_hash(identity)[:24]:
        raise AcousticVerificationError(
            "Verification acquisition identity is invalid."
        )
    return {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "authorization_basis": AUTHORIZATION_BASIS,
        "spec_sha256": plan["spec_sha256"],
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": expected_dry_run_sha256,
        **{field: False for field in SIDE_EFFECT_FIELDS},
        "replayed_at": utc_now(),
    }
