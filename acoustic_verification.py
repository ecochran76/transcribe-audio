"""Fail-closed verification-model acquisition evidence for Plan 0037 P4.

This module does not read audio, execute models, or authorize biometric
enrollment.  It records and replays the exact acquisition proposal that later
P4 work may apply under the operator's bounded acquisition grant.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import acoustic_audio_derivatives as audio_derivatives
import acoustic_speech_preparation as speech_preparation
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)
from acoustic_biometric_references import (
    BiometricReferenceError,
    INVALIDATION_SCHEMA,
    MATERIALIZATION_SCHEMA,
    PROMOTION_SCHEMA,
    acknowledge_descendant_invalidation,
    acknowledge_descendant_promotion,
    descendant_is_eligible,
    register_descendant,
    request_descendant_invalidation,
    resolve_eligible_reference,
    source_set_sha256,
)
from acoustic_speech_preparation import (
    METHOD_IDS,
    SpeechPreparationError,
    resolve_comparison_lineage_receipt,
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
DEFAULT_MODEL_SNAPSHOT_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration/"
    "acquisitions/snapshots/p4a-20260731-v1/models"
)
EXPECTED_ACQUISITION_MANIFEST_SHA256 = (
    "6470ecc8591fd8a40f8d788ba9a3edddc37a508cc54d47800037ab594b957ebe"
)
EXPECTED_ACQUISITION_SPEC_SHA256 = (
    "c6cc78b265eed77b5b52637765dc3cde07a74e99b1ef7fde6328a15ae1345c1c"
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


class VerificationAdapter(Protocol):
    """Small host-owned boundary shared by real and deterministic adapters."""

    candidate_id: str
    revision_sha: str
    embedding_dimension: int

    @property
    def model_loaded(self) -> bool: ...

    def embed(
        self, samples: Sequence[float], *, sample_rate: int
    ) -> tuple[float, ...]: ...


def _validated_waveform(
    samples: Sequence[float], *, sample_rate: int
) -> tuple[float, ...]:
    if sample_rate != 16_000:
        raise AcousticVerificationError("Verification audio must use 16000 Hz.")
    values = tuple(float(value) for value in samples)
    if not values:
        raise AcousticVerificationError("Verification audio is empty.")
    if len(values) < 8_000:
        raise AcousticVerificationError(
            "Verification audio is shorter than the 0.5 second minimum."
        )
    if any(not math.isfinite(value) for value in values):
        raise AcousticVerificationError("Verification audio must be finite.")
    if any(abs(value) > 1.0 for value in values):
        raise AcousticVerificationError("Verification audio must be normalized PCM.")
    return values


def _normalized_embedding(values: Sequence[float]) -> tuple[float, ...]:
    selected = tuple(float(value) for value in values)
    if not selected or any(not math.isfinite(value) for value in selected):
        raise AcousticVerificationError("Verification embedding must be finite.")
    norm = math.sqrt(sum(value * value for value in selected))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise AcousticVerificationError("Verification embedding has zero norm.")
    return tuple(value / norm for value in selected)


def _adapter_embedding(
    adapter: VerificationAdapter,
    samples: Sequence[float],
    *,
    sample_rate: int,
) -> tuple[float, ...]:
    try:
        return adapter.embed(samples, sample_rate=sample_rate)
    except AcousticVerificationError:
        raise
    except Exception as exc:
        raise AcousticVerificationError("Verification adapter failed closed.") from exc


def _contains_forbidden_private_key(value: Any) -> bool:
    forbidden = {
        "embedding", "embeddings", "vector", "vectors", "tensor", "tensors",
        "transcript", "transcript_text", "email", "name", "audio_path",
        "waveform", "samples",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_private_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_private_key(item) for item in value)
    return False


def cosine_score(
    enrollment: Sequence[float], probe: Sequence[float]
) -> float:
    """Score two finite, same-dimension embeddings without mutating either."""
    left = tuple(float(value) for value in enrollment)
    right = tuple(float(value) for value in probe)
    if not left or len(left) != len(right):
        raise AcousticVerificationError("Verification embedding dimensions differ.")
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise AcousticVerificationError("Verification embeddings must be finite.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise AcousticVerificationError("Verification embedding has zero norm.")
    score = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    if not math.isfinite(score):
        raise AcousticVerificationError("Verification score must be finite.")
    return max(-1.0, min(1.0, score))


class FakeVerificationAdapter:
    """Deterministic synthetic adapter; never part of the production registry."""

    revision_sha = "synthetic-test-adapter-v1"
    embedding_dimension = 8

    def __init__(self, *, candidate_id: str) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", candidate_id):
            raise AcousticVerificationError("Synthetic candidate ID is invalid.")
        self.candidate_id = candidate_id

    @property
    def model_loaded(self) -> bool:
        return False

    def embed(
        self, samples: Sequence[float], *, sample_rate: int
    ) -> tuple[float, ...]:
        values = _validated_waveform(samples, sample_rate=sample_rate)
        count = len(values)
        mean = sum(values) / count
        absolute_mean = sum(abs(value) for value in values) / count
        rms = math.sqrt(sum(value * value for value in values) / count)
        maximum = max(values)
        minimum = min(values)
        zero_crossings = sum(
            (left < 0.0 <= right) or (right < 0.0 <= left)
            for left, right in zip(values, values[1:])
        ) / max(1, count - 1)
        first_half = sum(values[: count // 2]) / max(1, count // 2)
        second_half = sum(values[count // 2 :]) / max(1, count - count // 2)
        return _normalized_embedding(
            (
                mean,
                absolute_mean,
                rms,
                maximum,
                minimum,
                zero_crossings,
                first_half,
                second_half,
            )
        )


def _verified_model_artifacts(
    snapshot_root: Path, candidate_id: str
) -> dict[str, dict[str, Any]]:
    """Replay the exact P4A manifest and hash every file before model load."""
    root = snapshot_root.expanduser().absolute()
    manifest_path = root / "acquisition-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AcousticVerificationError("P4A acquisition manifest is unavailable.")
    require_private_file(manifest_path, root)
    if sha256_file(manifest_path) != EXPECTED_ACQUISITION_MANIFEST_SHA256:
        raise AcousticVerificationError("P4A acquisition manifest hash mismatch.")
    manifest = read_private_object(manifest_path)
    if (
        manifest.get("schema_version")
        != "transcribe-audio.verification-model-acquisition-manifest.v1"
        or manifest.get("authorization_basis") != AUTHORIZATION_BASIS
        or manifest.get("authorization_scope") != AUTHORIZATION_SCOPE
        or manifest.get("spec_sha256") != EXPECTED_ACQUISITION_SPEC_SHA256
        or manifest.get("snapshot_root") != str(root)
        or manifest.get("real_biometric_enrollment_authorized") is not False
        or manifest.get("audio_read") is not False
        or manifest.get("embedding_materialized") is not False
        or manifest.get("trial_executed") is not False
    ):
        raise AcousticVerificationError("P4A acquisition manifest is invalid.")
    artifacts = manifest.get("artifacts")
    records = artifacts.get(candidate_id) if isinstance(artifacts, dict) else None
    expected_paths = EXPECTED_CANDIDATES.get(candidate_id, {}).get("artifact_paths")
    if (
        not isinstance(records, dict)
        or set(records) != set(expected_paths or ())
    ):
        raise AcousticVerificationError("Candidate acquisition inventory is invalid.")
    model_root = root / "models" / candidate_id
    for name in expected_paths or ():
        record = records[name]
        if not isinstance(record, dict):
            raise AcousticVerificationError("Candidate artifact record is invalid.")
        path = Path(str(record.get("path", ""))).expanduser().absolute()
        if (
            path != model_root / name
            or not isinstance(record.get("size_bytes"), int)
            or not SHA256_RE.fullmatch(str(record.get("sha256", "")))
        ):
            raise AcousticVerificationError("Candidate artifact binding is invalid.")
        require_private_file(path, root)
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise AcousticVerificationError("Candidate artifact hash mismatch.")
    installed = manifest.get("installed_distributions")
    required_distribution = (
        ("speechbrain", "1.1.0")
        if candidate_id == "speechbrain_ecapa_tdnn"
        else ("onnxruntime", "1.24.4")
    )
    if (
        not isinstance(installed, dict)
        or installed.get(required_distribution[0]) != required_distribution[1]
        or _package_version(required_distribution[0]) != required_distribution[1]
    ):
        raise AcousticVerificationError("Candidate runtime version is invalid.")
    return records


class _SpeechBrainEcapaAdapter:
    candidate_id = "speechbrain_ecapa_tdnn"
    revision_sha = EXPECTED_CANDIDATES[candidate_id]["revision_sha"]
    embedding_dimension = 192

    def __init__(self, snapshot_root: Path) -> None:
        self.snapshot_root = snapshot_root.expanduser().absolute()
        self.model_dir = self.snapshot_root / "models" / self.candidate_id
        self._model: Any = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            _verified_model_artifacts(self.snapshot_root, self.candidate_id)
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self._model = EncoderClassifier.from_hparams(
                    source=str(self.model_dir),
                    savedir=str(self.model_dir),
                    overrides={"pretrained_path": str(self.model_dir)},
                    run_opts={"device": "cpu"},
                )
            except Exception as exc:
                raise AcousticVerificationError(
                    "SpeechBrain model could not be loaded."
                ) from exc
        return self._model

    def embed(
        self, samples: Sequence[float], *, sample_rate: int
    ) -> tuple[float, ...]:
        values = _validated_waveform(samples, sample_rate=sample_rate)
        try:
            import torch

            waveform = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                output = self._load().encode_batch(waveform).reshape(-1)
            result = output.detach().cpu().tolist()
        except AcousticVerificationError:
            raise
        except Exception as exc:
            raise AcousticVerificationError("SpeechBrain inference failed.") from exc
        if len(result) != self.embedding_dimension:
            raise AcousticVerificationError("SpeechBrain embedding dimension changed.")
        return _normalized_embedding(result)


class _WeSpeakerOnnxAdapter:
    def __init__(
        self,
        *,
        candidate_id: str,
        model_dir: Path,
        model_name: str,
        embedding_dimension: int,
    ) -> None:
        self.candidate_id = candidate_id
        self.revision_sha = EXPECTED_CANDIDATES[candidate_id]["revision_sha"]
        self.embedding_dimension = embedding_dimension
        self.snapshot_root = model_dir.expanduser().absolute().parent.parent
        self.model_path = model_dir.expanduser().absolute() / model_name
        self._model: Any = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            _verified_model_artifacts(self.snapshot_root, self.candidate_id)
            try:
                import onnxruntime

                self._model = onnxruntime.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                raise AcousticVerificationError(
                    "WeSpeaker model could not be loaded."
                ) from exc
        return self._model

    def embed(
        self, samples: Sequence[float], *, sample_rate: int
    ) -> tuple[float, ...]:
        values = _validated_waveform(samples, sample_rate=sample_rate)
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi

            waveform = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            features = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_frequency=16_000,
            )
            features = features - features.mean(dim=0, keepdim=True)
            output = self._load().run(
                None, {"feats": features.unsqueeze(0).cpu().numpy()}
            )[0]
            result = output.reshape(-1).tolist()
        except AcousticVerificationError:
            raise
        except Exception as exc:
            raise AcousticVerificationError("WeSpeaker inference failed.") from exc
        if len(result) != self.embedding_dimension:
            raise AcousticVerificationError("WeSpeaker embedding dimension changed.")
        return _normalized_embedding(result)


def adapter_registry(
    *, snapshot_root: Path = DEFAULT_MODEL_SNAPSHOT_ROOT
) -> dict[str, VerificationAdapter]:
    """Return the exact lazy P4 adapter inventory without loading any model."""
    models_root = snapshot_root.expanduser().absolute()
    snapshot = models_root.parent
    return {
        "speechbrain_ecapa_tdnn": _SpeechBrainEcapaAdapter(
            snapshot
        ),
        "wespeaker_campplus": _WeSpeakerOnnxAdapter(
            candidate_id="wespeaker_campplus",
            model_dir=models_root / "wespeaker_campplus",
            model_name="voxceleb_CAM++.onnx",
            embedding_dimension=512,
        ),
        "wespeaker_resnet34": _WeSpeakerOnnxAdapter(
            candidate_id="wespeaker_resnet34",
            model_dir=models_root / "wespeaker_resnet34",
            model_name="voxceleb_resnet34.onnx",
            embedding_dimension=256,
        ),
    }


PROFILE_SCHEMA = "transcribe-audio.biometric-profile.v1"
PROFILE_MANIFEST_SCHEMA = "transcribe-audio.biometric-profile-manifest.v1"
PROFILE_LIFECYCLE_SCHEMA = "transcribe-audio.biometric-profile-lifecycle.v1"
TRIAL_SCHEMA = "transcribe-audio.verification-trial.v1"
ENROLLMENT_PREVIEW_SCHEMA = (
    "transcribe-audio.biometric-enrollment-preview.v1"
)
ENROLLMENT_APPLY_AUTHORITY_SCHEMA = (
    "transcribe-audio.biometric-enrollment-apply-authority.v1"
)
ENROLLMENT_APPLICATION_SCHEMA = (
    "transcribe-audio.biometric-enrollment-application.v1"
)
DEVELOPMENT_TRIAL_AUTHORITY_SCHEMA = (
    "transcribe-audio.verification-development-trial-authority.v1"
)
DEVELOPMENT_TRIAL_APPLICATION_SCHEMA = (
    "transcribe-audio.verification-development-trial-application.v1"
)
CALIBRATION_APPLY_AUTHORITY_SCHEMA = (
    "transcribe-audio.verification-calibration-apply-authority.v1"
)
CALIBRATION_APPLICATION_SCHEMA = (
    "transcribe-audio.verification-calibration-application.v1"
)
CALIBRATION_SPLIT_REVEAL_SCHEMA = (
    "transcribe-audio.verification-calibration-split-reveal.v1"
)
CALIBRATION_PREPARATION_SCHEMA = (
    "transcribe-audio.verification-calibration-preparation.v1"
)
CALIBRATION_WINDOW_SELECTION_SCHEMA = (
    "transcribe-audio.verification-calibration-window-selection.v1"
)
CALIBRATION_SCORE_MATRIX_SCHEMA = (
    "transcribe-audio.verification-calibration-score-matrix.v1"
)
EVALUATION_APPLY_AUTHORITY_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-apply-authority.v1"
)
EVALUATION_SPLIT_REVEAL_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-split-reveal.v1"
)
EVALUATION_PREPARATION_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-preparation.v1"
)
EVALUATION_WINDOW_SELECTION_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-window-selection.v1"
)
EVALUATION_SCORE_MATRIX_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-score-matrix.v1"
)
EVALUATION_APPLICATION_SCHEMA = (
    "transcribe-audio.verification-terminal-evaluation-application.v1"
)
REAL_ENROLLMENT_AUTHORIZATION_BASIS = "operator_blanket_proceed_2026-07-31"
REAL_ENROLLMENT_AUTHORIZER_REF_ID = "operator-standing-20260731"
_OPAQUE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
DEFAULT_SPLIT_ACCESS_POLICY = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/split-access-policy.json"
)
DEFAULT_PARENT_CORPUS_MANIFEST = Path(
    "~/.local/state/transcribe-audio/plan-0037/corpora/"
    "acoustic-corpus-1f93d1405f82676420571e1b/manifest.json"
)
EXPECTED_SPLIT_ACCESS_POLICY_SHA256 = (
    "41808c1b654b20ea8b395f65757db0ffc9f1a79862b31a6a2770268be1083467"
)
EXPECTED_PARENT_CORPUS_MANIFEST_SHA256 = (
    "73f0e04aab0274ddfeaa7f6b1567ecb135eebc0a0d6e5818cb3bd2ee5535dabf"
)
EXPECTED_DEVELOPMENT_RECORD_SET_SHA256 = (
    "1326728c26dafcda41d3883ca25569dba928ee52d5e481348fac1773e4547f5e"
)
EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256 = (
    "b767b9b1e5167c1b13a01fb0e1c4add4dd1323e983e1df8708e5e9dcd379436c"
)
EXPECTED_CALIBRATION_RECORD_SET_SHA256 = (
    "23480f0cbc0a73555a77d94301ca7f135e932250225abe544267c5c0d36ea543"
)
EXPECTED_CALIBRATION_CONVERSATION_SET_SHA256 = (
    "44d844aff33d3ff27986a3102a320db2f7fafcbdcd5a196b4c2909af535cdbc6"
)
EXPECTED_EVALUATION_RECORD_SET_SHA256 = (
    "1ae7dde3b9b627859dd2885ea66be1693ebf1e4c8e9f9a946ed6c20a41000ec0"
)
EXPECTED_EVALUATION_CONVERSATION_SET_SHA256 = (
    "bc2b39a1e1d6dec4665dea98f087a45dd4ecd4e6dc9240b87620ce9d7b58704b"
)
DEFAULT_TERMINAL_DECISION_POLICY = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/terminal-decision-policy.json"
)
EXPECTED_TERMINAL_DECISION_POLICY_SHA256 = (
    "98eadfd2a3a55a77d873ff0f3efbf7f2e75e296915d89777c7243a9b7ff373d8"
)
EXPECTED_EVALUATION_SPLIT_REVEAL_SHA256 = (
    "99c28df0d50610523845684878cdeea05428451f3bc63af855011a6b40efa0d9"
)
OBSERVED_INCOMPATIBLE_EVALUATION_P2_MODULE_SHA256 = (
    "96946fcc39cbc77928bd2df5f3944b93fec6359cbcb741859d34b0a26f6e1f22"
)
CALIBRATION_SCORE_METHOD_IDS = (
    "no_enhancement",
    "deepfilternet",
    "rnnoise",
)
SUPERSEDED_CALIBRATION_AUTHORITY_SHA256 = (
    "d5df3aaa0dd61704a42af71bf04beeaae26721401e722024572b419127feb5b3"
)
EXPECTED_P2_OPEN_ACQUISITION_MANIFEST_SHA256 = (
    "fc28406a6c2a8a84763a238940d0cec29a414e1d7952d74d69c9f597fdbe1d13"
)
EXPECTED_P2_PYANNOTE_ACQUISITION_MANIFEST_SHA256 = (
    "b3fd1614b3f233fa0b2e0bece0dfd88aaa9063e6f864b5298a7cf86effdaca10"
)
ENROLLMENT_CANDIDATE_PROPOSAL_SCHEMA = (
    "transcribe-audio.biometric-enrollment-candidate-proposal.v2"
)
DEFAULT_SPEAKER_EVALUATION_CAMPAIGN_ROOT = Path(
    "~/.local/state/transcribe-audio/speaker-evaluation-campaigns"
)
DEFAULT_APP_INTELLIGENCE_RUNS_ROOT = Path(
    "~/.local/state/transcribe-audio/app-intelligence-runs"
)
DEFAULT_REVIEWED_CLUE_CONTINUITY_AUTHORITY = Path(__file__).parent / (
    "docs/dev/fixtures/plan-0037-p4/reviewed-clue-continuity-authority.json"
)
EXPECTED_REVIEWED_CLUE_CONTINUITY_AUTHORITY_SHA256 = (
    "4c952608568edea918265f0851e89f4abfec2f41ac3faf590aaca20cb10da868"
)
DEFAULT_DEVELOPMENT_COMPARISON_RECEIPT = Path(
    "~/.local/state/transcribe-audio/plan-0037/speech-preparation/"
    "development-comparison-20260731-v5/development-comparison.json"
)
EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256 = (
    "0b3c68a31cbf7bc7f80d5302a52c8c7630414ca198cef78223b63baedbfd0ac3"
)
_PREVIEW_REASON_CODES = {
    "p3_reference_store_unavailable",
    "no_requested_people",
    "no_replay_eligible_real_p3_generation",
}
_SOURCE_BINDING_KEYS = {
    "reference_id", "source_sha256", "recording_id", "conversation_id",
    "speaker_label_id", "session_id", "start_seconds", "end_seconds",
    "segment_sha256", "quality_evidence_sha256", "lineage_authority",
    "lineage_replay_receipt_sha256",
}
_UNIT_KEYS = {
    "person_ref_id", "p3_profile_id", "p3_generation_id",
    "p3_generation_sha256", "p3_source_set_sha256", "p3_approval_id",
    "p3_approval_sha256", "source_segments",
}
_PREVIEW_KEYS = {
    "schema_version", "status", "reason_codes", "intended_split",
    "requested_person_ref_ids", "p3_store_present", "enrollment_units",
    "models", "acquisition_manifest_sha256", "acquisition_spec_sha256",
    "split_access_policy_sha256", "parent_corpus_manifest_sha256",
    "development_record_set_sha256", "development_conversation_set_sha256",
    "exact_apply_authority_required", "real_biometric_enrollment_authorized",
    "will_read_audio", "will_materialize_embeddings",
    "will_register_references", "will_run_trials",
    "will_perform_external_write", "contains_raw_biometric_values",
}


def _development_split_authority(
    split_policy_path: Path, parent_corpus_manifest_path: Path
) -> dict[str, Any]:
    policy_path = split_policy_path.expanduser().absolute()
    parent_path = parent_corpus_manifest_path.expanduser().absolute()
    if policy_path.is_symlink() or not policy_path.is_file():
        raise AcousticVerificationError("Split access policy is unavailable.")
    if sha256_file(policy_path) != EXPECTED_SPLIT_ACCESS_POLICY_SHA256:
        raise AcousticVerificationError("Split access policy hash is invalid.")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError("Split access policy is unreadable.") from exc
    require_private_file(parent_path, parent_path.parent)
    if sha256_file(parent_path) != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256:
        raise AcousticVerificationError("Parent corpus manifest hash is invalid.")
    parent = read_private_object(parent_path)
    if not isinstance(policy, Mapping) or not isinstance(parent, Mapping):
        raise AcousticVerificationError("Split authority shape is invalid.")
    development = policy.get("splits", {}).get("development")
    recordings = parent.get("recordings") if isinstance(parent, Mapping) else None
    if (
        policy.get("schema_version")
        != "transcribe-audio.verification-split-access-policy.v1"
        or policy.get("parent_corpus_manifest_sha256")
        != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256
        or not isinstance(development, Mapping)
        or development.get("authorization_state")
        != "authorized_by_operator_blanket_2026-07-31"
        or development.get("record_set_sha256")
        != EXPECTED_DEVELOPMENT_RECORD_SET_SHA256
        or development.get("conversation_set_sha256")
        != EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256
        or not isinstance(recordings, list)
    ):
        raise AcousticVerificationError("Development split authority is invalid.")
    selected = [
        item for item in recordings
        if isinstance(item, Mapping) and item.get("split") == "development"
    ]
    recording_ids = [str(item.get("recording_id", "")) for item in selected]
    conversation_ids = [str(item.get("conversation_id", "")) for item in selected]
    if (
        len(selected) != development.get("recording_count")
        or len(set(recording_ids)) != len(recording_ids)
        or len(set(conversation_ids)) != development.get("conversation_count")
        or canonical_artifact_hash(selected)
        != EXPECTED_DEVELOPMENT_RECORD_SET_SHA256
        or canonical_artifact_hash(sorted(conversation_ids))
        != EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256
    ):
        raise AcousticVerificationError("Development split membership drifted.")
    return {
        "recording_conversation_pairs": set(zip(recording_ids, conversation_ids)),
        "development_records": selected,
        "source_campaign": parent.get("source_campaign"),
        "split_access_policy_sha256": EXPECTED_SPLIT_ACCESS_POLICY_SHA256,
        "parent_corpus_manifest_sha256": EXPECTED_PARENT_CORPUS_MANIFEST_SHA256,
        "development_record_set_sha256": EXPECTED_DEVELOPMENT_RECORD_SET_SHA256,
        "development_conversation_set_sha256": (
            EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256
        ),
    }


def _validate_enrollment_preview(
    value: Any, *, split_authority: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PREVIEW_KEYS:
        raise AcousticVerificationError("Enrollment preview shape is invalid.")
    expected_models = [
        {"candidate_id": key, "revision_sha": candidate["revision_sha"]}
        for key, candidate in EXPECTED_CANDIDATES.items()
    ]
    if (
        value.get("schema_version") != ENROLLMENT_PREVIEW_SCHEMA
        or value.get("intended_split") != "development"
        or value.get("models") != expected_models
        or value.get("acquisition_manifest_sha256")
        != EXPECTED_ACQUISITION_MANIFEST_SHA256
        or value.get("acquisition_spec_sha256")
        != EXPECTED_ACQUISITION_SPEC_SHA256
        or any(value.get(key) != split_authority[key] for key in (
            "split_access_policy_sha256", "parent_corpus_manifest_sha256",
            "development_record_set_sha256",
            "development_conversation_set_sha256",
        ))
        or value.get("exact_apply_authority_required") is not True
        or value.get("real_biometric_enrollment_authorized") is not False
        or value.get("contains_raw_biometric_values") is not False
        or any(value.get(field) is not False for field in SIDE_EFFECT_FIELDS[3:])
    ):
        raise AcousticVerificationError("Enrollment preview authority is invalid.")
    requested = value.get("requested_person_ref_ids")
    reasons = value.get("reason_codes")
    units = value.get("enrollment_units")
    if (
        not isinstance(requested, list)
        or any(not isinstance(item, str) for item in requested)
        or requested != sorted(requested)
        or len(set(requested)) != len(requested)
        or any(not _OPAQUE_ID_RE.fullmatch(item) for item in requested)
        or not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or reasons != sorted(reasons)
        or len(set(reasons)) != len(reasons)
        or any(reason not in _PREVIEW_REASON_CODES for reason in reasons)
        or not isinstance(units, list)
        or not isinstance(value.get("p3_store_present"), bool)
    ):
        raise AcousticVerificationError("Enrollment preview status inputs are invalid.")
    if value.get("status") == "ready_for_review":
        if reasons or not requested or len(units) != len(requested) or not value["p3_store_present"]:
            raise AcousticVerificationError("Ready enrollment preview is inconsistent.")
    elif value.get("status") == "blocked":
        if not reasons or units:
            raise AcousticVerificationError("Blocked enrollment preview is inconsistent.")
    else:
        raise AcousticVerificationError("Enrollment preview status is invalid.")
    expected_reasons: set[str] = set()
    if not requested:
        expected_reasons.add("no_requested_people")
    if value["p3_store_present"] is False:
        expected_reasons.add("p3_reference_store_unavailable")
    if requested and value.get("status") == "blocked":
        expected_reasons.add("no_replay_eligible_real_p3_generation")
    if reasons != sorted(expected_reasons):
        raise AcousticVerificationError(
            "Enrollment preview reasons do not match its facts."
        )
    unit_people: list[str] = []
    allowed_pairs = split_authority["recording_conversation_pairs"]
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != _UNIT_KEYS:
            raise AcousticVerificationError("Enrollment unit shape is invalid.")
        for field in ("person_ref_id", "p3_profile_id", "p3_generation_id", "p3_approval_id"):
            if not _OPAQUE_ID_RE.fullmatch(str(unit.get(field, ""))):
                raise AcousticVerificationError("Enrollment unit ID is invalid.")
        for field in ("p3_generation_sha256", "p3_source_set_sha256", "p3_approval_sha256"):
            if not SHA256_RE.fullmatch(str(unit.get(field, ""))):
                raise AcousticVerificationError("Enrollment unit hash is invalid.")
        unit_people.append(str(unit["person_ref_id"]))
        segments = unit.get("source_segments")
        if not isinstance(segments, list) or not segments:
            raise AcousticVerificationError("Enrollment unit sources are invalid.")
        for segment in segments:
            if not isinstance(segment, Mapping) or set(segment) != _SOURCE_BINDING_KEYS:
                raise AcousticVerificationError("Enrollment source shape is invalid.")
            for field in ("reference_id", "recording_id", "conversation_id", "speaker_label_id", "session_id"):
                if not _OPAQUE_ID_RE.fullmatch(str(segment.get(field, ""))):
                    raise AcousticVerificationError("Enrollment source ID is invalid.")
            for field in ("source_sha256", "segment_sha256", "quality_evidence_sha256", "lineage_replay_receipt_sha256"):
                if not SHA256_RE.fullmatch(str(segment.get(field, ""))):
                    raise AcousticVerificationError("Enrollment source hash is invalid.")
            try:
                if isinstance(segment["start_seconds"], bool) or isinstance(
                    segment["end_seconds"], bool
                ):
                    raise TypeError
                start = float(segment["start_seconds"])
                end = float(segment["end_seconds"])
            except (TypeError, ValueError) as exc:
                raise AcousticVerificationError("Enrollment source bounds are invalid.") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise AcousticVerificationError("Enrollment source bounds are invalid.")
            if segment.get("lineage_authority") not in {
                "p1_audio_derivative_replay", "p2_speech_preparation_replay"
            }:
                raise AcousticVerificationError("Enrollment lineage authority is invalid.")
            if (segment["recording_id"], segment["conversation_id"]) not in allowed_pairs:
                raise AcousticVerificationError(
                    "Enrollment source is outside the development split."
                )
    if value.get("status") == "ready_for_review" and sorted(unit_people) != requested:
        raise AcousticVerificationError("Enrollment unit people do not match request.")
    return value


def _enrollment_source_binding(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise AcousticVerificationError("P3 enrollment source is invalid.")
    if source.get("fixture_authority") is not None:
        raise AcousticVerificationError(
            "Synthetic references cannot enter real enrollment preview."
        )
    lineage = source.get("lineage")
    if not isinstance(lineage, Mapping):
        raise AcousticVerificationError(
            "Real enrollment preview requires production P1/P2 lineage."
        )
    required = {
        "reference_id",
        "source_sha256",
        "recording_id",
        "conversation_id",
        "speaker_label_id",
        "session_id",
        "start_seconds",
        "end_seconds",
        "source_key",
        "quality_evidence",
    }
    if not required.issubset(source):
        raise AcousticVerificationError("P3 enrollment source binding is incomplete.")
    quality = source.get("quality_evidence")
    if not isinstance(quality, Mapping):
        raise AcousticVerificationError("P3 enrollment quality binding is invalid.")
    for field in ("source_sha256", "source_key"):
        if not SHA256_RE.fullmatch(str(source.get(field, ""))):
            raise AcousticVerificationError("P3 enrollment source hash is invalid.")
    quality_sha = str(quality.get("sha256", ""))
    if not SHA256_RE.fullmatch(quality_sha):
        raise AcousticVerificationError("P3 enrollment quality hash is invalid.")
    for field in (
        "reference_id",
        "recording_id",
        "conversation_id",
        "speaker_label_id",
        "session_id",
    ):
        if not _OPAQUE_ID_RE.fullmatch(str(source.get(field, ""))):
            raise AcousticVerificationError("P3 enrollment source ID is invalid.")
    return {
        "reference_id": source["reference_id"],
        "source_sha256": source["source_sha256"],
        "recording_id": source["recording_id"],
        "conversation_id": source["conversation_id"],
        "speaker_label_id": source["speaker_label_id"],
        "session_id": source["session_id"],
        "start_seconds": source["start_seconds"],
        "end_seconds": source["end_seconds"],
        "segment_sha256": source["source_key"],
        "quality_evidence_sha256": quality_sha,
        "lineage_authority": lineage.get("authority"),
        "lineage_replay_receipt_sha256": lineage.get("replay_receipt_sha256"),
    }


def build_real_enrollment_preview(
    requested_person_ref_ids: Sequence[str],
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    intended_split: str = "development",
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Persist a private no-audio preview; never authorize or apply enrollment."""
    if intended_split != "development":
        raise AcousticVerificationError(
            "Real enrollment preview is limited to the authorized development split."
        )
    if isinstance(requested_person_ref_ids, (str, bytes)):
        raise AcousticVerificationError(
            "Enrollment person references must be a sequence of opaque IDs."
        )
    requested = [str(value) for value in requested_person_ref_ids]
    if any(not _OPAQUE_ID_RE.fullmatch(value) for value in requested):
        raise AcousticVerificationError("Enrollment person references must be opaque.")
    if len(set(requested)) != len(requested):
        raise AcousticVerificationError("Enrollment person references must be unique.")
    requested.sort()
    split_authority = _development_split_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    p3_root = p3_runtime_root.expanduser().absolute()
    p3_store_present = (p3_root / "references.sqlite3").is_file()
    blockers: list[dict[str, str]] = []
    units: list[dict[str, Any]] = []
    if not p3_store_present:
        blockers.append({"reason_code": "p3_reference_store_unavailable"})
    if not requested:
        blockers.append({"reason_code": "no_requested_people"})
    for person_ref_id in requested:
        try:
            resolved = resolve_eligible_reference(
                person_ref_id, runtime_root=p3_root
            )
            if (
                resolved.get("person_ref_id") != person_ref_id
                or resolved.get("materialization_contract")
                != "stage_then_register_then_promote"
            ):
                raise AcousticVerificationError(
                    "P3 enrollment resolution binding is invalid."
                )
            reference = resolved.get("reference")
            if not isinstance(reference, Mapping):
                raise AcousticVerificationError("P3 reference manifest is invalid.")
            if reference.get("synthetic_test_only") is not False:
                raise AcousticVerificationError(
                    "Synthetic references cannot enter real enrollment preview."
                )
            approval = reference.get("approval")
            sources = reference.get("sources")
            if not isinstance(approval, Mapping) or not isinstance(sources, list):
                raise AcousticVerificationError(
                    "P3 approval or source bindings are unavailable."
                )
            approval_id = str(approval.get("approval_id", ""))
            if not _OPAQUE_ID_RE.fullmatch(approval_id):
                raise AcousticVerificationError("P3 enrollment approval is invalid.")
            source_segments = [
                _enrollment_source_binding(source) for source in sources
            ]
            if any(
                (segment["recording_id"], segment["conversation_id"])
                not in split_authority["recording_conversation_pairs"]
                for segment in source_segments
            ):
                raise AcousticVerificationError(
                    "Enrollment source is outside the development split."
                )
            units.append(
                {
                    "person_ref_id": person_ref_id,
                    "p3_profile_id": resolved["profile_id"],
                    "p3_generation_id": resolved["generation_id"],
                    "p3_generation_sha256": resolved["generation_sha256"],
                    "p3_source_set_sha256": reference["source_set_sha256"],
                    "p3_approval_id": approval_id,
                    "p3_approval_sha256": canonical_artifact_hash(dict(approval)),
                    "source_segments": source_segments,
                }
            )
        except (AcousticVerificationError, BiometricReferenceError):
            blockers.append(
                {
                    "person_ref_id": person_ref_id,
                    "reason_code": "no_replay_eligible_real_p3_generation",
                }
            )
    if blockers:
        units = []
    models = [
        {
            "candidate_id": candidate_id,
            "revision_sha": candidate["revision_sha"],
        }
        for candidate_id, candidate in EXPECTED_CANDIDATES.items()
    ]
    preview = {
        "schema_version": ENROLLMENT_PREVIEW_SCHEMA,
        "status": "ready_for_review" if units else "blocked",
        "reason_codes": sorted(
            {blocker["reason_code"] for blocker in blockers}
        ),
        "intended_split": intended_split,
        "requested_person_ref_ids": requested,
        "p3_store_present": p3_store_present,
        "enrollment_units": units,
        "models": models,
        "acquisition_manifest_sha256": EXPECTED_ACQUISITION_MANIFEST_SHA256,
        "acquisition_spec_sha256": EXPECTED_ACQUISITION_SPEC_SHA256,
        "split_access_policy_sha256": split_authority[
            "split_access_policy_sha256"
        ],
        "parent_corpus_manifest_sha256": split_authority[
            "parent_corpus_manifest_sha256"
        ],
        "development_record_set_sha256": split_authority[
            "development_record_set_sha256"
        ],
        "development_conversation_set_sha256": split_authority[
            "development_conversation_set_sha256"
        ],
        "exact_apply_authority_required": True,
        "real_biometric_enrollment_authorized": False,
        "will_read_audio": False,
        "will_materialize_embeddings": False,
        "will_register_references": False,
        "will_run_trials": False,
        "will_perform_external_write": False,
        "contains_raw_biometric_values": False,
    }
    _validate_enrollment_preview(preview, split_authority=split_authority)
    root = runtime_root.expanduser().absolute()
    preview_sha = canonical_artifact_hash(preview)
    path = root / "enrollment-previews" / f"{preview_sha}.json"
    ensure_private_tree(root, path.parent)
    write_immutable_private_json(path, preview)
    return {
        **preview,
        "preview_sha256": preview_sha,
        "private_preview_path": str(path),
    }


def replay_real_enrollment_preview(
    preview_sha256: str,
    *,
    runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Replay one exact private P4C preview without resolving or opening media."""
    if not SHA256_RE.fullmatch(str(preview_sha256)):
        raise AcousticVerificationError("Enrollment preview hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "enrollment-previews" / f"{preview_sha256}.json"
    require_private_file(path, root)
    preview = read_private_object(path)
    split_authority = _development_split_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    if canonical_artifact_hash(preview) != preview_sha256:
        raise AcousticVerificationError("Enrollment preview replay is invalid.")
    _validate_enrollment_preview(preview, split_authority=split_authority)
    return {
        **preview,
        "preview_sha256": preview_sha256,
        "private_preview_path": str(path),
    }


def _real_enrollment_apply_payload(
    *,
    candidate_proposal: Mapping[str, Any],
    enrollment_preview: Mapping[str, Any],
    candidate_proposal_sha256: str,
    enrollment_preview_sha256: str,
    authorized_at: str,
) -> dict[str, Any]:
    """Build one exact P4C authority from replayed proposal and preview facts."""
    if (
        candidate_proposal.get("status") != "ready_for_operator_review"
        or candidate_proposal.get("reason_codes") != []
        or enrollment_preview.get("status") != "ready_for_review"
        or enrollment_preview.get("reason_codes") != []
    ):
        raise AcousticVerificationError(
            "Real enrollment apply requires ready proposal and preview receipts."
        )
    candidates = candidate_proposal.get("candidates")
    units = enrollment_preview.get("enrollment_units")
    models = enrollment_preview.get("models")
    if not isinstance(candidates, list) or not isinstance(units, list) or not units:
        raise AcousticVerificationError("Real enrollment apply units are unavailable.")
    if not isinstance(models, list) or not models:
        raise AcousticVerificationError("Real enrollment apply models are unavailable.")
    candidate_by_person = {
        str(candidate.get("person_ref_id") or ""): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    if len(candidate_by_person) != len(candidates):
        raise AcousticVerificationError("Real enrollment candidates are invalid.")
    for unit in units:
        if not isinstance(unit, Mapping):
            raise AcousticVerificationError("Real enrollment unit is invalid.")
        person_ref_id = str(unit.get("person_ref_id") or "")
        candidate = candidate_by_person.get(person_ref_id)
        if candidate is None:
            raise AcousticVerificationError(
                "Real enrollment preview is outside the candidate proposal."
            )
        proposed_sources = candidate.get("proposed_sources")
        if not isinstance(proposed_sources, list):
            raise AcousticVerificationError("Candidate proposal sources are invalid.")
        expected_segments = []
        for source in proposed_sources:
            if not isinstance(source, Mapping):
                raise AcousticVerificationError(
                    "Candidate proposal source is invalid."
                )
            lineage = source.get("lineage")
            quality = source.get("quality_evidence")
            if not isinstance(lineage, Mapping) or not isinstance(quality, Mapping):
                raise AcousticVerificationError(
                    "Candidate proposal source evidence is invalid."
                )
            expected_segments.append(
                {
                    "reference_id": source.get("reference_id"),
                    "source_sha256": source.get("source_sha256"),
                    "recording_id": source.get("recording_id"),
                    "conversation_id": source.get("conversation_id"),
                    "speaker_label_id": source.get("speaker_label_id"),
                    "session_id": source.get("session_id"),
                    "start_seconds": source.get("start_seconds"),
                    "end_seconds": source.get("end_seconds"),
                    "quality_evidence_sha256": quality.get("sha256"),
                    "lineage_authority": lineage.get("authority"),
                    "lineage_replay_receipt_sha256": lineage.get(
                        "replay_receipt_sha256"
                    ),
                }
            )
        preview_segments = unit.get("source_segments")
        if not isinstance(preview_segments, list):
            raise AcousticVerificationError("Preview source segments are invalid.")
        comparable_preview_segments = (
            [
                {key: segment.get(key) for key in expected_segments[0]}
                for segment in preview_segments
                if isinstance(segment, Mapping)
            ]
            if expected_segments
            else []
        )
        if (
            candidate.get("proposed_p3_profile_id") != unit.get("p3_profile_id")
            or candidate.get("proposed_source_set_sha256")
            != unit.get("p3_source_set_sha256")
            or len(comparable_preview_segments) != len(preview_segments)
            or expected_segments != comparable_preview_segments
        ):
            raise AcousticVerificationError(
                "Real enrollment proposal and preview bindings differ."
            )
    if set(candidate_by_person) != {
        str(unit.get("person_ref_id") or "") for unit in units
    }:
        raise AcousticVerificationError(
            "Real enrollment proposal and preview people differ."
        )
    authority = {
        "schema_version": ENROLLMENT_APPLY_AUTHORITY_SCHEMA,
        "status": "authorized",
        "reason_code": None,
        "authorization_basis": REAL_ENROLLMENT_AUTHORIZATION_BASIS,
        "authorized_by_ref_id": REAL_ENROLLMENT_AUTHORIZER_REF_ID,
        "authorized_at": authorized_at,
        "intended_split": "development",
        "candidate_proposal_sha256": candidate_proposal_sha256,
        "enrollment_preview_sha256": enrollment_preview_sha256,
        "enrollment_units": [dict(unit) for unit in units],
        "models": [dict(model) for model in models],
        "preparation_methods": [
            {
                "method_id": "no_enhancement",
                "development_comparison_receipt_sha256": (
                    EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256
                ),
            }
        ],
        "authorization_scope": (
            "exact_p3_generations_and_source_segments_for_p4c_"
            "development_enrollment"
        ),
        "real_biometric_enrollment_authorized": True,
        "will_read_audio": True,
        "will_materialize_embeddings": True,
        "will_register_references": False,
        "will_register_p4_descendants": True,
        "will_run_trials": False,
        "will_read_calibration_or_evaluation": False,
        "will_perform_external_write": False,
        "contains_raw_biometric_values": False,
    }
    if _contains_forbidden_private_key(authority):
        raise AcousticVerificationError(
            "Real enrollment authority contains forbidden private data."
        )
    return authority


def _validate_real_enrollment_apply_authority(
    value: Any,
    *,
    candidate_proposal: Mapping[str, Any],
    enrollment_preview: Mapping[str, Any],
    candidate_proposal_sha256: str,
    enrollment_preview_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AcousticVerificationError("Real enrollment authority is invalid.")
    authorized_at = str(value.get("authorized_at") or "")
    expected = _real_enrollment_apply_payload(
        candidate_proposal=candidate_proposal,
        enrollment_preview=enrollment_preview,
        candidate_proposal_sha256=candidate_proposal_sha256,
        enrollment_preview_sha256=enrollment_preview_sha256,
        authorized_at=authorized_at,
    )
    if dict(value) != expected or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        authorized_at,
    ):
        raise AcousticVerificationError("Real enrollment authority is invalid.")
    return dict(value)


def build_real_enrollment_apply_authority(
    candidate_proposal_sha256: str,
    enrollment_preview_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Persist the exact P4C apply scope authorized by the standing grant."""
    proposal = replay_real_enrollment_candidate_proposal(
        candidate_proposal_sha256,
        runtime_root=runtime_root,
    )
    preview = replay_real_enrollment_preview(
        enrollment_preview_sha256,
        runtime_root=runtime_root,
    )
    for unit in preview.get("enrollment_units") or []:
        resolved = resolve_eligible_reference(
            str(unit.get("person_ref_id") or ""), runtime_root=p3_runtime_root
        )
        if (
            resolved.get("profile_id") != unit.get("p3_profile_id")
            or resolved.get("generation_id") != unit.get("p3_generation_id")
            or resolved.get("generation_sha256")
            != unit.get("p3_generation_sha256")
        ):
            raise AcousticVerificationError(
                "P3 generation changed before enrollment authorization."
            )
    root = runtime_root.expanduser().absolute()
    authority_dir = root / "enrollment-authorities"
    ensure_private_tree(root, authority_dir)
    existing_authorities: list[tuple[Path, dict[str, Any], str]] = []
    for existing_path in sorted(authority_dir.glob("*.json")):
        require_private_file(existing_path, root)
        existing = read_private_object(existing_path)
        if (
            existing.get("candidate_proposal_sha256") != candidate_proposal_sha256
            or existing.get("enrollment_preview_sha256")
            != enrollment_preview_sha256
        ):
            continue
        _validate_real_enrollment_apply_authority(
            existing,
            candidate_proposal=proposal,
            enrollment_preview=preview,
            candidate_proposal_sha256=candidate_proposal_sha256,
            enrollment_preview_sha256=enrollment_preview_sha256,
        )
        existing_sha256 = canonical_artifact_hash(existing)
        if existing_path.name != f"{existing_sha256}.json":
            raise AcousticVerificationError(
                "Existing real enrollment authority path is invalid."
            )
        existing_authorities.append((existing_path, existing, existing_sha256))
    if len(existing_authorities) > 1:
        raise AcousticVerificationError(
            "Multiple real enrollment authorities exist for one exact scope."
        )
    if existing_authorities:
        path, authority, authority_sha256 = existing_authorities[0]
        return {
            **authority,
            "authority_sha256": authority_sha256,
            "private_authority_path": str(path),
        }
    authority = _real_enrollment_apply_payload(
        candidate_proposal=proposal,
        enrollment_preview=preview,
        candidate_proposal_sha256=candidate_proposal_sha256,
        enrollment_preview_sha256=enrollment_preview_sha256,
        authorized_at=utc_now(),
    )
    authority_sha256 = canonical_artifact_hash(authority)
    path = authority_dir / f"{authority_sha256}.json"
    write_immutable_private_json(path, authority)
    return {
        **authority,
        "authority_sha256": authority_sha256,
        "private_authority_path": str(path),
    }


def replay_real_enrollment_apply_authority(
    authority_sha256: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Replay one exact P4C authority without opening audio or running models."""
    if not SHA256_RE.fullmatch(str(authority_sha256)):
        raise AcousticVerificationError("Real enrollment authority hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "enrollment-authorities" / f"{authority_sha256}.json"
    require_private_file(path, root)
    authority = read_private_object(path)
    proposal_sha256 = str(authority.get("candidate_proposal_sha256") or "")
    preview_sha256 = str(authority.get("enrollment_preview_sha256") or "")
    proposal = replay_real_enrollment_candidate_proposal(
        proposal_sha256, runtime_root=root
    )
    preview = replay_real_enrollment_preview(preview_sha256, runtime_root=root)
    _validate_real_enrollment_apply_authority(
        authority,
        candidate_proposal=proposal,
        enrollment_preview=preview,
        candidate_proposal_sha256=proposal_sha256,
        enrollment_preview_sha256=preview_sha256,
    )
    if canonical_artifact_hash(authority) != authority_sha256:
        raise AcousticVerificationError("Real enrollment authority replay is invalid.")
    return {
        **authority,
        "authority_sha256": authority_sha256,
        "private_authority_path": str(path),
    }


def _subtract_intervals(
    intervals: Sequence[tuple[float, float]],
    blocked: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in intervals:
        parts = [(start, end)]
        for blocked_start, blocked_end in blocked:
            retained: list[tuple[float, float]] = []
            for part_start, part_end in parts:
                if blocked_end <= part_start or blocked_start >= part_end:
                    retained.append((part_start, part_end))
                    continue
                if part_start < blocked_start:
                    retained.append((part_start, blocked_start))
                if blocked_end < part_end:
                    retained.append((blocked_end, part_end))
            parts = retained
        result.extend(parts)
    return result


def _candidate_windows(
    utterances: Any,
    *,
    speaker_label: Any,
    speech_regions: Any,
    blocked_regions: Any,
) -> list[tuple[float, float]]:
    if not isinstance(utterances, list) or not isinstance(speaker_label, str):
        raise AcousticVerificationError("Candidate transcript metadata is invalid.")
    if not isinstance(speech_regions, list) or not isinstance(blocked_regions, list):
        raise AcousticVerificationError("Candidate preparation regions are invalid.")
    speech = [
        (float(item["start_seconds"]), float(item["end_seconds"]))
        for item in speech_regions
        if isinstance(item, Mapping)
    ]
    blocked = [
        (float(item["start_seconds"]), float(item["end_seconds"]))
        for item in blocked_regions
        if isinstance(item, Mapping)
    ]
    if not speech or any(
        not math.isfinite(value) or start < 0 or end <= start
        for start, end in (*speech, *blocked)
        for value in (start, end)
    ):
        raise AcousticVerificationError("Candidate preparation regions are invalid.")
    spans: list[tuple[float, float]] = []
    for utterance in utterances:
        if not isinstance(utterance, Mapping) or utterance.get("speaker") != speaker_label:
            continue
        try:
            if isinstance(utterance.get("start"), bool) or isinstance(
                utterance.get("end"), bool
            ):
                raise TypeError
            start = float(utterance["start"]) / 1000.0
            end = float(utterance["end"]) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            raise AcousticVerificationError(
                "Candidate utterance timestamps are invalid."
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise AcousticVerificationError("Candidate utterance timestamps are invalid.")
        intersections = [
            (max(start, speech_start), min(end, speech_end))
            for speech_start, speech_end in speech
            if max(start, speech_start) < min(end, speech_end)
        ]
        for clean_start, clean_end in _subtract_intervals(intersections, blocked):
            if clean_end - clean_start >= 0.75:
                spans.append((clean_start, min(clean_end, clean_start + 15.0)))
    selected: list[tuple[float, float]] = []
    for start, end in sorted(spans, key=lambda item: (-(item[1] - item[0]), item[0])):
        if any(start < other_end and end > other_start for other_start, other_end in selected):
            continue
        selected.append((round(start, 6), round(end, 6)))
        if len(selected) == 3:
            break
    return selected


def _cap_candidate_sources_per_session(
    sources: Sequence[Mapping[str, Any]],
    *,
    maximum_windows: int = 3,
) -> list[dict[str, Any]]:
    """Deduplicate and cap clean windows after all labels map to one person."""
    by_session: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        session_id = str(source.get("session_id") or "")
        if not session_id:
            raise AcousticVerificationError("Candidate source session is invalid.")
        by_session.setdefault(session_id, []).append(dict(source))
    retained: list[dict[str, Any]] = []
    for session_id in sorted(by_session):
        selected: list[dict[str, Any]] = []
        seen_reference_ids: set[str] = set()
        ordered = sorted(
            by_session[session_id],
            key=lambda item: (
                -(float(item["end_seconds"]) - float(item["start_seconds"])),
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                str(item["reference_id"]),
            ),
        )
        for source in ordered:
            reference_id = str(source.get("reference_id") or "")
            start = float(source["start_seconds"])
            end = float(source["end_seconds"])
            if reference_id in seen_reference_ids:
                continue
            if any(
                start < float(other["end_seconds"])
                and end > float(other["start_seconds"])
                for other in selected
            ):
                continue
            seen_reference_ids.add(reference_id)
            selected.append(source)
            if len(selected) == maximum_windows:
                break
        if len(selected) > maximum_windows:
            raise AcousticVerificationError("Candidate session window cap failed.")
        retained.extend(selected)
    return sorted(retained, key=lambda item: str(item["reference_id"]))


def _bounded_existing_file(path: Path, root: Path) -> Path:
    """Resolve an existing regular file without trusting an escaping path."""
    try:
        resolved_root = root.expanduser().resolve(strict=True)
        resolved = path.expanduser().resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise AcousticVerificationError("Continuity witness path is invalid.") from exc
    if not resolved.is_file():
        raise AcousticVerificationError("Continuity witness is not a regular file.")
    return resolved


def _reviewed_clue_authority(
    path: Path,
    *,
    source_campaign: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    selected = path.expanduser().absolute()
    if sha256_file(selected) != EXPECTED_REVIEWED_CLUE_CONTINUITY_AUTHORITY_SHA256:
        raise AcousticVerificationError("Reviewed clue authority hash is invalid.")
    try:
        authority = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError(
            "Reviewed clue authority is unreadable."
        ) from exc
    authority_hashes = source_campaign.get("authority_hashes")
    entries = authority.get("entries")
    if (
        authority.get("schema_version")
        != "transcribe-audio.reviewed-clue-continuity-authority.v1"
        or authority.get("campaign_id") != source_campaign.get("campaign_id")
        or not isinstance(authority_hashes, Mapping)
        or authority.get("campaign_manifest_sha256")
        != authority_hashes.get("campaign_manifest_sha256")
        or authority.get("gold_index_sha256")
        != authority_hashes.get("gold_index_sha256")
        or authority.get("contains_transcript_text") is not False
        or authority.get("will_read_audio") is not False
        or authority.get("will_authorize_biometric_enrollment") is not False
        or not isinstance(entries, list)
    ):
        raise AcousticVerificationError("Reviewed clue authority is invalid.")
    by_document: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AcousticVerificationError("Reviewed clue authority entry is invalid.")
        document_id = str(entry.get("document_id") or "")
        hashes = [
            entry.get(key)
            for key in (
                "blind_prediction_sha256",
                "clue_packet_sha256",
                "events_jsonl_sha256",
                "prompt_packet_sha256",
                "prompt_text_sha256",
                "reviewed_artifact_sha256",
                "run_json_sha256",
                "status_sha256",
            )
        ]
        if (
            not document_id
            or document_id in by_document
            or any(not SHA256_RE.fullmatch(str(value or "")) for value in hashes)
        ):
            raise AcousticVerificationError("Reviewed clue authority entry is invalid.")
        by_document[document_id] = entry
    return by_document


def _authority_bound_file(
    root: Path,
    relative_path: Any,
    expected_sha256: Any,
) -> Path:
    if not isinstance(relative_path, str) or not SHA256_RE.fullmatch(
        str(expected_sha256 or "")
    ):
        raise AcousticVerificationError("Continuity authority file binding is invalid.")
    path = _bounded_existing_file(root / relative_path, root)
    if sha256_file(path) != expected_sha256:
        raise AcousticVerificationError("Continuity authority file hash drifted.")
    return path


def _reviewed_clue_continuity(
    *,
    record: Mapping[str, Any],
    transcript: Mapping[str, Any],
    reviewed_artifact_sha256: str,
    campaign_id: str,
    authority_entry: Mapping[str, Any],
    campaign_root: Path,
    app_intelligence_runs_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Resolve a committed reviewed-hash witness and exact current clue matches."""
    document_id = str(record.get("document_id") or "")
    recording_id = str(record.get("recording_id") or "")
    conversation_id = str(record.get("conversation_id") or "")
    clue_run_id = str(authority_entry.get("clue_discovery_run_id") or "")
    if (
        not document_id
        or not campaign_id
        or authority_entry.get("document_id") != document_id
        or authority_entry.get("recording_id") != recording_id
        or authority_entry.get("conversation_id") != conversation_id
        or authority_entry.get("reviewed_artifact_sha256")
        != reviewed_artifact_sha256
        or not clue_run_id
    ):
        return None
    selected_campaign_root = campaign_root.expanduser().absolute()
    campaign_dir = selected_campaign_root / campaign_id
    selected_runs_root = app_intelligence_runs_root.expanduser().absolute()
    try:
        prediction_path = _authority_bound_file(
            campaign_dir,
            authority_entry.get("blind_prediction_relative_path"),
            authority_entry.get("blind_prediction_sha256"),
        )
        run_root = _bounded_existing_file(
            selected_runs_root / clue_run_id / "run.json", selected_runs_root
        ).parent
        run_path = _authority_bound_file(
            run_root, "run.json", authority_entry.get("run_json_sha256")
        )
        events_path = _authority_bound_file(
            run_root, "events.jsonl", authority_entry.get("events_jsonl_sha256")
        )
        packet_path = _authority_bound_file(
            run_root,
            authority_entry.get("clue_packet_relative_path"),
            authority_entry.get("clue_packet_sha256"),
        )
        prompt_packet_path = _authority_bound_file(
            run_root,
            authority_entry.get("prompt_packet_relative_path"),
            authority_entry.get("prompt_packet_sha256"),
        )
        prompt_text_path = _authority_bound_file(
            run_root,
            authority_entry.get("prompt_text_relative_path"),
            authority_entry.get("prompt_text_sha256"),
        )
        status_path = _authority_bound_file(
            run_root,
            authority_entry.get("status_relative_path"),
            authority_entry.get("status_sha256"),
        )
    except AcousticVerificationError:
        return None
    prediction = read_private_object(prediction_path)
    run = read_private_object(run_path)
    packet = read_private_object(packet_path)
    prompt_packet = read_private_object(prompt_packet_path)
    status = read_private_object(status_path)
    try:
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if (
        prediction.get("schema_version")
        != "transcribe-audio.speaker-evaluation-blind-prediction.v1"
        or prediction.get("campaign_id") != campaign_id
        or prediction.get("document_id") != document_id
        or prediction.get("artifact_sha256") != reviewed_artifact_sha256
        or prediction.get("prediction_visibility") != "blind"
        or prediction.get("gold_content_included") is not False
        or prediction.get("will_read_gold_records") is not False
        or prediction.get("will_perform_external_write") is not False
        or not isinstance(prediction.get("run_references"), Mapping)
        or prediction["run_references"].get("clue_discovery_run_id") != clue_run_id
        or run.get("schema_version") != "transcribe-audio.app-intelligence-run.v1"
        or run.get("run_id") != clue_run_id
        or run.get("document_id") != document_id
        or run.get("workflow") != "speaker_preprocessing"
        or status.get("schema_version")
        != "transcribe-audio.app-intelligence-model-turn-status.v1"
        or status.get("run_id") != clue_run_id
        or status.get("status") != "completed"
        or status.get("completed") is not True
        or status.get("will_execute_structured_decision") is not False
        or not any(
            isinstance(event, Mapping)
            and event.get("run_id") == clue_run_id
            and event.get("event_type") == "model_turn_status_captured"
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("completed") is True
            and event["payload"].get("status") == "completed"
            for event in events
        )
        or prompt_packet.get("run_id") != clue_run_id
        or prompt_packet.get("task") != "speaker_clue_discovery"
        or not isinstance(prompt_packet.get("document"), Mapping)
        or prompt_packet["document"].get("id") != document_id
        or prompt_packet.get("prompt_text") != prompt_text_path.read_text(encoding="utf-8")
    ):
        return None
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        return None
    conversation = packet.get("conversation")
    speakers = packet.get("speakers")
    packet_json = json.dumps(packet, sort_keys=True, ensure_ascii=False)
    if (
        packet.get("schema_version")
        != "transcribe-audio.speaker-clue-discovery-packet.v1"
        or packet.get("task") != "speaker_clue_discovery"
        or not isinstance(conversation, Mapping)
        or conversation.get("conversation_id") != conversation_id
        or recording_id not in (conversation.get("recording_ids") or [])
        or not isinstance(speakers, list)
        or packet_json not in str(prompt_packet.get("prompt_text") or "")
    ):
        return None
    matched: list[dict[str, Any]] = []
    seen_clue_ids: set[str] = set()
    seen_speaker_labels: set[str] = set()
    for speaker in speakers:
        if not isinstance(speaker, Mapping):
            return None
        speaker_label = speaker.get("speaker_label")
        clues = speaker.get("utterance_clues")
        if (
            not isinstance(speaker_label, str)
            or speaker_label in seen_speaker_labels
            or not isinstance(clues, list)
        ):
            return None
        seen_speaker_labels.add(speaker_label)
        for clue in clues:
            if not isinstance(clue, Mapping):
                return None
            clue_id = str(clue.get("utterance_id") or "")
            match = re.fullmatch(r"utterance-([1-9][0-9]*)", clue_id)
            if match is None or clue_id in seen_clue_ids:
                return None
            seen_clue_ids.add(clue_id)
            ordinal = int(match.group(1))
            if ordinal > len(utterances) or not isinstance(
                utterances[ordinal - 1], Mapping
            ):
                return None
            current = utterances[ordinal - 1]
            current_text = str(current.get("text") or "").strip()[:1_200]
            if (
                current.get("speaker") != speaker_label
                or current.get("start") != clue.get("start")
                or current.get("end") != clue.get("end")
                or current_text != str(clue.get("text") or "")
            ):
                return None
            matched.append(dict(current))
    if not matched:
        return None
    clue_projection = [
        {
            "speaker": item.get("speaker"),
            "start": item.get("start"),
            "end": item.get("end"),
            "text_sha256": hashlib.sha256(
                str(item.get("text") or "").strip()[:1_200].encode("utf-8")
            ).hexdigest(),
        }
        for item in matched
    ]
    return matched, {
        "mode": "committed_reviewed_clue_authority",
        "reviewed_artifact_sha256": reviewed_artifact_sha256,
        "authority_sha256": EXPECTED_REVIEWED_CLUE_CONTINUITY_AUTHORITY_SHA256,
        "prediction_sha256": sha256_file(prediction_path),
        "clue_packet_sha256": sha256_file(packet_path),
        "clue_projection_sha256": canonical_artifact_hash(clue_projection),
        "matched_clue_count": len(matched),
    }


def _candidate_proposal_payload(
    *,
    split_policy_path: Path,
    parent_corpus_manifest_path: Path,
    development_comparison_receipt_path: Path,
    reviewed_clue_continuity_authority_path: Path,
    campaign_root: Path,
    app_intelligence_runs_root: Path,
) -> dict[str, Any]:
    split_authority = _development_split_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    development_records = split_authority["development_records"]
    source_campaign = split_authority.get("source_campaign")
    campaign_id = (
        str(source_campaign.get("campaign_id") or "")
        if isinstance(source_campaign, Mapping)
        else ""
    )
    continuity_authority: Optional[dict[str, Mapping[str, Any]]] = None
    record_by_id = {str(item["recording_id"]): item for item in development_records}
    joined_path = development_comparison_receipt_path.expanduser().absolute()
    require_private_file(joined_path, joined_path.parent)
    if sha256_file(joined_path) != EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256:
        raise AcousticVerificationError(
            "Development comparison receipt hash is invalid."
        )
    joined = read_private_object(joined_path)
    selected_recordings = joined.get("selected_recordings")
    if (
        joined.get("schema_version")
        != "transcribe-audio.speech-preparation-development-comparison.v2"
        or joined.get("status") != "success"
        or joined.get("corpus_manifest_sha256")
        != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256
        or joined.get("will_run_biometrics") is not False
        or joined.get("will_read_calibration_or_evaluation") is not False
        or joined.get("will_perform_external_write") is not False
        or not isinstance(selected_recordings, list)
        or not selected_recordings
    ):
        raise AcousticVerificationError("Development comparison receipt is invalid.")
    subject_sources: dict[str, list[dict[str, Any]]] = {}
    subject_evidence: dict[str, list[dict[str, Any]]] = {}
    lineage_exclusions: list[dict[str, str]] = []
    person_rows_considered = 0
    for selected in selected_recordings:
        if not isinstance(selected, Mapping) or selected.get("split") != "development":
            raise AcousticVerificationError("Selected preparation recording is invalid.")
        recording_id = str(selected.get("recording_id", ""))
        record = record_by_id.get(recording_id)
        if record is None:
            raise AcousticVerificationError(
                "Selected preparation recording is outside development."
            )
        comparison_path = Path(str(selected.get("comparison_path", ""))).expanduser().absolute()
        require_private_file(comparison_path, comparison_path.parent)
        if sha256_file(comparison_path) != selected.get("comparison_sha256"):
            raise AcousticVerificationError("Preparation comparison hash is invalid.")
        comparison = read_private_object(comparison_path)
        pyannote = next(
            (
                item for item in comparison.get("method_results") or []
                if isinstance(item, Mapping)
                and item.get("method_id") == "pyannote_community_1"
            ),
            None,
        )
        if pyannote is None or pyannote.get("status") != "success":
            raise AcousticVerificationError(
                "Pyannote preparation evidence is unavailable."
            )
        try:
            lineage = resolve_comparison_lineage_receipt(
                str(selected.get("p2_run_id", "")),
                method_id="no_enhancement",
                replay_receipt_sha256=str(selected.get("replay_sha256", "")),
                runtime_root=comparison_path.parents[2],
            )
        except SpeechPreparationError as exc:
            raise AcousticVerificationError(
                "No-enhancement preparation lineage is invalid."
            ) from exc
        transcript_lineage = record.get("transcript_lineage")
        if not isinstance(transcript_lineage, Mapping):
            raise AcousticVerificationError("Transcript lineage is unavailable.")
        transcript_path = Path(
            str(transcript_lineage.get("current_artifact_path", ""))
        ).expanduser().absolute()
        require_private_file(transcript_path, transcript_path.parent)
        transcript_sha = sha256_file(transcript_path)
        if transcript_sha != transcript_lineage.get("current_artifact_sha256"):
            raise AcousticVerificationError("Transcript artifact hash is invalid.")
        reviewed_artifact_sha256 = str(
            transcript_lineage.get("reviewed_artifact_sha256") or ""
        )
        transcript = read_private_object(transcript_path)
        if (
            transcript.get("schema_version") != 2
            or transcript.get("recording_id") != recording_id
            or transcript.get("conversation_id") != record.get("conversation_id")
            or not isinstance(transcript.get("utterances"), list)
        ):
            raise AcousticVerificationError("Transcript metadata binding is invalid.")
        continuity_evidence: dict[str, Any]
        candidate_utterances = transcript["utterances"]
        if transcript_sha == reviewed_artifact_sha256:
            continuity_evidence = {
                "mode": "exact_reviewed_artifact",
                "reviewed_artifact_sha256": reviewed_artifact_sha256,
                "current_artifact_sha256": transcript_sha,
                "matched_clue_count": len(candidate_utterances),
            }
        else:
            if not isinstance(source_campaign, Mapping):
                continuity = None
            else:
                if continuity_authority is None:
                    continuity_authority = _reviewed_clue_authority(
                        reviewed_clue_continuity_authority_path,
                        source_campaign=source_campaign,
                    )
                authority_entry = continuity_authority.get(
                    str(record.get("document_id") or "")
                )
                continuity = (
                    _reviewed_clue_continuity(
                        record=record,
                        transcript=transcript,
                        reviewed_artifact_sha256=reviewed_artifact_sha256,
                        campaign_id=campaign_id,
                        authority_entry=authority_entry,
                        campaign_root=campaign_root,
                        app_intelligence_runs_root=app_intelligence_runs_root,
                    )
                    if authority_entry is not None
                    else None
                )
            if continuity is None:
                lineage_exclusions.append(
                    {
                        "recording_id": recording_id,
                        "reason": "reviewed_artifact_continuity_unavailable",
                        "reviewed_artifact_sha256": reviewed_artifact_sha256,
                        "current_artifact_sha256": transcript_sha,
                    }
                )
                continue
            candidate_utterances, continuity_evidence = continuity
            continuity_evidence["current_artifact_sha256"] = transcript_sha
        gold = record.get("operator_gold")
        if not isinstance(gold, Mapping) or not isinstance(gold.get("speaker_truth"), list):
            raise AcousticVerificationError("Operator gold metadata is invalid.")
        conditions = record.get("conditions")
        if not isinstance(conditions, Mapping):
            raise AcousticVerificationError("Recording conditions are invalid.")
        blocked_regions = [
            *(pyannote.get("overlap_regions") or []),
            *(pyannote.get("speaker_change_regions") or []),
        ]
        for truth in gold["speaker_truth"]:
            if (
                not isinstance(truth, Mapping)
                or truth.get("outcome") != "person"
                or not isinstance(truth.get("subject_id"), str)
            ):
                continue
            person_ref_id = str(truth["subject_id"])
            if not _OPAQUE_ID_RE.fullmatch(person_ref_id):
                raise AcousticVerificationError("Gold subject ID is not opaque.")
            person_rows_considered += 1
            windows = _candidate_windows(
                candidate_utterances,
                speaker_label=truth.get("speaker_label"),
                speech_regions=pyannote.get("speech_regions"),
                blocked_regions=blocked_regions,
            )
            if not windows:
                continue
            speaker_label_id = "speaker-label-" + canonical_artifact_hash(
                {"recording_id": recording_id, "speaker_label": truth.get("speaker_label")}
            )[:20]
            sources = subject_sources.setdefault(person_ref_id, [])
            for start, end in windows:
                reference_id = "reference-" + canonical_artifact_hash(
                    {
                        "person_ref_id": person_ref_id,
                        "recording_id": recording_id,
                        "start_seconds": start,
                        "end_seconds": end,
                    }
                )[:24]
                acoustic_conditions = [
                    f"{key}:{conditions[key]}" for key in sorted(conditions)
                ]
                source = {
                    "reference_id": reference_id,
                    "source_blob_id": lineage["source_blob_id"],
                    "source_sha256": lineage["source_sha256"],
                    "recording_id": recording_id,
                    "conversation_id": record["conversation_id"],
                    "speaker_label_id": speaker_label_id,
                    "session_id": record["conversation_id"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "source_duration_seconds": lineage["source_duration_seconds"],
                    "quality_evidence": {
                        "evidence_id": "quality-" + lineage["audio_quality_sha256"][:24],
                        "sha256": lineage["audio_quality_sha256"],
                    },
                    "device_class": str(conditions.get("device") or "unspecified"),
                    "acoustic_conditions": acoustic_conditions,
                    "lineage": lineage,
                }
                sources.append(source)
            subject_evidence.setdefault(person_ref_id, []).append(
                {
                    "recording_id": recording_id,
                    "conversation_id": record["conversation_id"],
                    "operator_gold_sha256": canonical_artifact_hash(dict(gold)),
                    "selected_truth_sha256": canonical_artifact_hash(dict(truth)),
                    "transcript_artifact_sha256": transcript_sha,
                    "reviewed_clue_continuity": continuity_evidence,
                    "p2_comparison_sha256": selected["comparison_sha256"],
                    "pyannote_result_sha256": canonical_artifact_hash(dict(pyannote)),
                    "selected_reference_ids": [source["reference_id"] for source in sources
                                               if source["recording_id"] == recording_id],
                }
            )
    candidates: list[dict[str, Any]] = []
    for person_ref_id, sources in sorted(subject_sources.items()):
        normalized_sources = _cap_candidate_sources_per_session(sources)
        sessions = sorted({source["session_id"] for source in normalized_sources})
        if len(sessions) < 2:
            continue
        if any(
            sum(1 for source in normalized_sources if source["session_id"] == session_id)
            > 3
            for session_id in sessions
        ):
            raise AcousticVerificationError("Candidate session window cap failed.")
        selected_reference_ids = {
            str(source["reference_id"]) for source in normalized_sources
        }
        selection_evidence = []
        for evidence in subject_evidence[person_ref_id]:
            retained_ids = sorted(
                set(evidence["selected_reference_ids"]) & selected_reference_ids
            )
            if retained_ids:
                selection_evidence.append(
                    {**evidence, "selected_reference_ids": retained_ids}
                )
        source_hash = source_set_sha256(normalized_sources, test_mode=False)
        total_seconds = round(
            sum(
                float(source["end_seconds"]) - float(source["start_seconds"])
                for source in normalized_sources
            ),
            6,
        )
        candidates.append(
            {
                "person_ref_id": person_ref_id,
                "proposed_p3_profile_id": "reference-profile-" + canonical_artifact_hash(
                    {"person_ref_id": person_ref_id, "source_set_sha256": source_hash}
                )[:24],
                "proposed_source_set_sha256": source_hash,
                "session_count": len(sessions),
                "window_count": len(normalized_sources),
                "total_selected_seconds": total_seconds,
                "proposed_sources": normalized_sources,
                "selection_evidence": sorted(
                    selection_evidence, key=lambda item: item["recording_id"]
                ),
                "operator_decision_required": True,
            }
        )
    reason_codes = []
    if lineage_exclusions:
        reason_codes.append("reviewed_artifact_lineage_drift")
    if not candidates:
        reason_codes.append("no_multi_session_clean_candidates")
    proposal = {
        "schema_version": ENROLLMENT_CANDIDATE_PROPOSAL_SCHEMA,
        "status": "ready_for_operator_review" if candidates else "blocked",
        "reason_codes": reason_codes,
        "intended_split": "development",
        "proposal_only": True,
        "biometric_enrollment_authorized": False,
        "exact_apply_manifest_required": True,
        "split_access_policy_sha256": split_authority["split_access_policy_sha256"],
        "parent_corpus_manifest_sha256": split_authority[
            "parent_corpus_manifest_sha256"
        ],
        "development_record_set_sha256": split_authority[
            "development_record_set_sha256"
        ],
        "development_conversation_set_sha256": split_authority[
            "development_conversation_set_sha256"
        ],
        "development_comparison_receipt_sha256": (
            EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256
        ),
        "selection_contract": {
            "source_scope": "three_recording_p2_development_slice",
            "speaker_identity_source": "frozen_operator_gold_person_rows",
            "timestamp_source": (
                "exact_reviewed_artifact_or_committed_reviewed_clue_authority"
            ),
            "speech_filter": "p2_pyannote_community_1_speech_regions",
            "excluded_regions": ["overlap", "speaker_change"],
            "minimum_window_seconds": 0.75,
            "maximum_window_seconds": 15.0,
            "maximum_windows_per_session": 3,
            "minimum_sessions_per_candidate": 2,
            "p3_lineage_method": "no_enhancement",
        },
        "denominators": {
            "selected_development_recordings": len(selected_recordings),
            "reviewed_artifact_lineage_exclusions": len(lineage_exclusions),
            "eligible_reviewed_artifact_recordings": (
                len(selected_recordings) - len(lineage_exclusions)
            ),
            "person_rows_considered": person_rows_considered,
            "candidate_people": len(candidates),
            "candidate_sessions": sum(item["session_count"] for item in candidates),
            "candidate_windows": sum(item["window_count"] for item in candidates),
        },
        "lineage_exclusions": sorted(
            lineage_exclusions, key=lambda item: item["recording_id"]
        ),
        "candidates": candidates,
        "will_read_audio": False,
        "will_materialize_embeddings": False,
        "will_register_references": False,
        "will_run_trials": False,
        "will_perform_external_write": False,
        "contains_transcript_text": False,
        "contains_raw_biometric_values": False,
    }
    if _contains_forbidden_private_key(proposal):
        raise AcousticVerificationError("Enrollment proposal contains forbidden data.")
    return proposal


def build_real_enrollment_candidate_proposal(
    *,
    runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
    development_comparison_receipt_path: Path = DEFAULT_DEVELOPMENT_COMPARISON_RECEIPT,
    reviewed_clue_continuity_authority_path: Path = (
        DEFAULT_REVIEWED_CLUE_CONTINUITY_AUTHORITY
    ),
    campaign_root: Path = DEFAULT_SPEAKER_EVALUATION_CAMPAIGN_ROOT,
    app_intelligence_runs_root: Path = DEFAULT_APP_INTELLIGENCE_RUNS_ROOT,
) -> dict[str, Any]:
    """Persist a private metadata-only candidate packet for operator review."""
    proposal = _candidate_proposal_payload(
        split_policy_path=split_policy_path,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
        development_comparison_receipt_path=development_comparison_receipt_path,
        reviewed_clue_continuity_authority_path=(
            reviewed_clue_continuity_authority_path
        ),
        campaign_root=campaign_root,
        app_intelligence_runs_root=app_intelligence_runs_root,
    )
    root = runtime_root.expanduser().absolute()
    proposal_sha = canonical_artifact_hash(proposal)
    path = root / "enrollment-proposals" / f"{proposal_sha}.json"
    ensure_private_tree(root, path.parent)
    write_immutable_private_json(path, proposal)
    return {
        **proposal,
        "proposal_sha256": proposal_sha,
        "private_proposal_path": str(path),
    }


def replay_real_enrollment_candidate_proposal(
    proposal_sha256: str,
    *,
    runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
    development_comparison_receipt_path: Path = DEFAULT_DEVELOPMENT_COMPARISON_RECEIPT,
    reviewed_clue_continuity_authority_path: Path = (
        DEFAULT_REVIEWED_CLUE_CONTINUITY_AUTHORITY
    ),
    campaign_root: Path = DEFAULT_SPEAKER_EVALUATION_CAMPAIGN_ROOT,
    app_intelligence_runs_root: Path = DEFAULT_APP_INTELLIGENCE_RUNS_ROOT,
) -> dict[str, Any]:
    """Recompute all metadata selection semantics and replay one exact proposal."""
    if not SHA256_RE.fullmatch(str(proposal_sha256)):
        raise AcousticVerificationError("Enrollment proposal hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "enrollment-proposals" / f"{proposal_sha256}.json"
    require_private_file(path, root)
    stored = read_private_object(path)
    expected = _candidate_proposal_payload(
        split_policy_path=split_policy_path,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
        development_comparison_receipt_path=development_comparison_receipt_path,
        reviewed_clue_continuity_authority_path=(
            reviewed_clue_continuity_authority_path
        ),
        campaign_root=campaign_root,
        app_intelligence_runs_root=app_intelligence_runs_root,
    )
    if canonical_artifact_hash(stored) != proposal_sha256 or stored != expected:
        raise AcousticVerificationError("Enrollment proposal replay is invalid.")
    return {
        **stored,
        "proposal_sha256": proposal_sha256,
        "private_proposal_path": str(path),
    }


def _profile_database(root: Path) -> sqlite3.Connection:
    selected = root.expanduser().absolute()
    ensure_private_tree(selected, selected / "profiles")
    ensure_private_tree(selected, selected / "authority")
    path = selected / "profiles.sqlite3"
    connection = sqlite3.connect(path)
    os.chmod(path, 0o600)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY,
            descendant_id TEXT NOT NULL UNIQUE,
            person_ref_id TEXT NOT NULL,
            p3_profile_id TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            generation_sha256 TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            preprocessing_json TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            vector_dimension INTEGER NOT NULL,
            window_count INTEGER NOT NULL,
            session_count INTEGER NOT NULL,
            dispersion REAL NOT NULL,
            lifecycle_state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            invalidation_receipt_sha256 TEXT,
            tombstone_path TEXT,
            replacement_profile_id TEXT,
            state_receipt_sha256 TEXT,
            profile_manifest_path TEXT,
            profile_manifest_sha256 TEXT
        )
        """
    )
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(profiles)")
    }
    if "invalidation_receipt_sha256" not in columns:
        connection.execute(
            "ALTER TABLE profiles ADD COLUMN invalidation_receipt_sha256 TEXT"
        )
    if "tombstone_path" not in columns:
        connection.execute("ALTER TABLE profiles ADD COLUMN tombstone_path TEXT")
    if "replacement_profile_id" not in columns:
        connection.execute(
            "ALTER TABLE profiles ADD COLUMN replacement_profile_id TEXT"
        )
    if "state_receipt_sha256" not in columns:
        connection.execute(
            "ALTER TABLE profiles ADD COLUMN state_receipt_sha256 TEXT"
        )
    if "profile_manifest_path" not in columns:
        connection.execute(
            "ALTER TABLE profiles ADD COLUMN profile_manifest_path TEXT"
        )
    if "profile_manifest_sha256" not in columns:
        connection.execute(
            "ALTER TABLE profiles ADD COLUMN profile_manifest_sha256 TEXT"
        )
    connection.commit()
    return connection


def _window_hash(samples: Sequence[float]) -> str:
    values = _validated_waveform(samples, sample_rate=16_000)
    payload = struct.pack(f"<{len(values)}f", *values)
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _write_private_blob(path: Path, payload: bytes, root: Path) -> str:
    ensure_private_tree(root, path.parent)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        require_private_file(path, root)
        if sha256_file(path) != digest:
            raise AcousticVerificationError("Private profile artifact conflicts.")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise AcousticVerificationError(
                "Private profile artifact already exists."
            ) from exc
        Path(temporary_name).unlink()
    finally:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
    return digest


def _authority_anchor(root: Path, receipt: dict[str, Any]) -> Path:
    authority_root = root / "authority"
    ensure_private_tree(root, authority_root)
    digest = canonical_artifact_hash(receipt)
    path = authority_root / f"{digest}.json"
    write_immutable_private_json(path, receipt)
    return path


def _lifecycle_receipt(
    *,
    profile_id: str,
    descendant_id: str,
    artifact_sha256: str,
    from_state: Optional[str],
    to_state: str,
    reason: str,
    previous_receipt_sha256: Optional[str],
    replacement_profile_id: Optional[str] = None,
    profile_manifest_sha256: str,
    transitioned_at: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_LIFECYCLE_SCHEMA,
        "profile_id": profile_id,
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha256,
        "profile_manifest_sha256": profile_manifest_sha256,
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason,
        "replacement_profile_id": replacement_profile_id,
        "previous_receipt_sha256": previous_receipt_sha256,
        "transitioned_at": transitioned_at or utc_now(),
        "will_perform_external_write": False,
    }


def _require_current_lifecycle_receipt(
    root: Path, row: Mapping[str, Any]
) -> dict[str, Any]:
    digest = str(row["state_receipt_sha256"] or "")
    if not SHA256_RE.fullmatch(digest):
        raise AcousticVerificationError("Profile lifecycle receipt is missing.")
    path = root / "authority" / f"{digest}.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    if (
        canonical_artifact_hash(receipt) != digest
        or receipt.get("schema_version") != PROFILE_LIFECYCLE_SCHEMA
        or receipt.get("profile_id") != row["profile_id"]
        or receipt.get("descendant_id") != row["descendant_id"]
        or receipt.get("artifact_sha256") != row["artifact_sha256"]
        or receipt.get("profile_manifest_sha256")
        != row["profile_manifest_sha256"]
        or receipt.get("to_state") != row["lifecycle_state"]
        or receipt.get("replacement_profile_id") != row["replacement_profile_id"]
        or receipt.get("will_perform_external_write") is not False
    ):
        raise AcousticVerificationError("Profile lifecycle receipt binding is invalid.")
    return receipt


def _public_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA,
        "profile_id": row["profile_id"],
        "descendant_id": row["descendant_id"],
        "person_ref_id": row["person_ref_id"],
        "p3_profile_id": row["p3_profile_id"],
        "generation_id": row["generation_id"],
        "generation_sha256": row["generation_sha256"],
        "candidate_id": row["candidate_id"],
        "model_revision": row["model_revision"],
        "artifact_sha256": row["artifact_sha256"],
        "profile_manifest_sha256": row["profile_manifest_sha256"],
        "private_artifact_path": (
            row["artifact_path"] if row["lifecycle_state"] != "deleted" else None
        ),
        "window_count": row["window_count"],
        "session_count": row["session_count"],
        "dispersion": row["dispersion"],
        "lifecycle_state": row["lifecycle_state"],
        "calibration_eligible": row["lifecycle_state"] == "active",
        "replacement_profile_id": row["replacement_profile_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _authorized_real_windows(
    sources: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
) -> list[dict[str, Any]]:
    """Open exact hash-bound P2 PCM only after real enrollment authorization."""
    if method_id not in METHOD_IDS:
        raise AcousticVerificationError("Real audio preparation method is invalid.")
    windows: list[dict[str, Any]] = []
    for source in sources:
        lineage = source.get("lineage") if isinstance(source, Mapping) else None
        if not isinstance(lineage, Mapping):
            raise AcousticVerificationError("Real enrollment lineage is unavailable.")
        try:
            resolved_lineage = resolve_comparison_lineage_receipt(
                str(lineage.get("run_id") or ""),
                method_id=method_id,
                replay_receipt_sha256=str(
                    lineage.get("replay_receipt_sha256") or ""
                ),
                runtime_root=Path(str(lineage.get("runtime_root") or "")),
            )
        except SpeechPreparationError as exc:
            raise AcousticVerificationError(
                "Real enrollment preparation lineage is invalid."
            ) from exc
        if (
            lineage.get("method_id") != "no_enhancement"
            or resolved_lineage.get("method_id") != method_id
            or any(
                resolved_lineage.get(key) != lineage.get(key)
                for key in (
                "run_id",
                "replay_receipt_sha256",
                "comparison_path",
                "comparison_sha256",
                "source_blob_id",
                "source_sha256",
                "audio_quality_sha256",
            )
            )
        ):
            raise AcousticVerificationError(
                "Real enrollment preparation lineage binding changed."
            )
        comparison_path = Path(str(lineage.get("comparison_path") or ""))
        require_private_file(comparison_path, comparison_path.parent)
        if sha256_file(comparison_path) != lineage.get("comparison_sha256"):
            raise AcousticVerificationError(
                "Real enrollment preparation comparison drifted."
            )
        comparison = read_private_object(comparison_path)
        method = next(
            (
                item
                for item in comparison.get("method_results") or []
                if isinstance(item, Mapping) and item.get("method_id") == method_id
            ),
            None,
        )
        if (
            method is None
            or method.get("status") != "success"
            or canonical_artifact_hash(dict(method))
            != resolved_lineage.get("method_result_sha256")
        ):
            raise AcousticVerificationError(
                "Real enrollment preparation result is invalid."
            )
        audio_path = Path(str(method.get("output_path") or ""))
        require_private_file(audio_path, audio_path.parent)
        if sha256_file(audio_path) != method.get("output_sha256"):
            raise AcousticVerificationError("Real enrollment PCM artifact drifted.")
        try:
            with wave.open(str(audio_path), "rb") as reader:
                if (
                    reader.getnchannels() != 1
                    or reader.getsampwidth() != 2
                    or reader.getframerate() != 16_000
                    or reader.getcomptype() != "NONE"
                ):
                    raise AcousticVerificationError(
                        "Real enrollment PCM contract is invalid."
                    )
                start_frame = round(float(source.get("start_seconds")) * 16_000)
                end_frame = round(float(source.get("end_seconds")) * 16_000)
                if (
                    start_frame < 0
                    or end_frame <= start_frame
                    or end_frame > reader.getnframes()
                ):
                    raise AcousticVerificationError(
                        "Real enrollment window is outside the PCM artifact."
                    )
                reader.setpos(start_frame)
                payload = reader.readframes(end_frame - start_frame)
        except (EOFError, OSError, wave.Error, TypeError, ValueError) as exc:
            raise AcousticVerificationError(
                "Real enrollment PCM window is unreadable."
            ) from exc
        sample_count = end_frame - start_frame
        if len(payload) != sample_count * 2:
            raise AcousticVerificationError("Real enrollment PCM window is truncated.")
        samples = tuple(
            value / 32768.0
            for value in struct.unpack(f"<{sample_count}h", payload)
        )
        windows.append(
            {"session_id": str(source.get("session_id") or ""), "samples": samples}
        )
    return windows


def _real_enrollment_application_payload(
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    profiles: Sequence[Mapping[str, Any]],
    applied_at: str,
) -> dict[str, Any]:
    receipt = {
        "schema_version": ENROLLMENT_APPLICATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha256,
        "candidate_proposal_sha256": authority["candidate_proposal_sha256"],
        "enrollment_preview_sha256": authority["enrollment_preview_sha256"],
        "intended_split": "development",
        "profile_count": len(profiles),
        "profiles": [dict(profile) for profile in profiles],
        "will_read_audio": True,
        "did_read_audio": True,
        "will_materialize_embeddings": True,
        "did_materialize_embeddings": True,
        "will_register_p4_descendants": True,
        "did_register_p4_descendants": True,
        "will_run_trials": False,
        "did_run_trials": False,
        "will_read_calibration_or_evaluation": False,
        "did_read_calibration_or_evaluation": False,
        "will_perform_external_write": False,
        "did_perform_external_write": False,
        "contains_raw_biometric_values": False,
        "applied_at": applied_at,
    }
    if _contains_forbidden_private_key(receipt):
        raise AcousticVerificationError(
            "Real enrollment receipt contains forbidden private data."
        )
    return receipt


def apply_real_enrollment(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    adapters: Optional[Mapping[str, VerificationAdapter]] = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Apply one exact P4C authority and activate model-specific profiles."""
    if adapters is not None and not test_mode:
        raise AcousticVerificationError(
            "Custom enrollment adapters are limited to deterministic tests."
        )
    authority = replay_real_enrollment_apply_authority(
        authority_sha256, runtime_root=runtime_root
    )
    if (
        authority.get("real_biometric_enrollment_authorized") is not True
        or authority.get("will_read_audio") is not True
        or authority.get("will_materialize_embeddings") is not True
        or authority.get("will_run_trials") is not False
        or authority.get("will_read_calibration_or_evaluation") is not False
    ):
        raise AcousticVerificationError("Real enrollment apply authority is invalid.")
    proposal = replay_real_enrollment_candidate_proposal(
        str(authority["candidate_proposal_sha256"]), runtime_root=runtime_root
    )
    candidates = {
        str(candidate.get("person_ref_id") or ""): candidate
        for candidate in proposal.get("candidates") or []
        if isinstance(candidate, Mapping)
    }
    selected_adapters = dict(adapters or adapter_registry())
    expected_models = {
        str(model.get("candidate_id") or ""): str(model.get("revision_sha") or "")
        for model in authority.get("models") or []
        if isinstance(model, Mapping)
    }
    if set(selected_adapters) != set(expected_models) or any(
        selected_adapters[candidate_id].revision_sha != revision
        for candidate_id, revision in expected_models.items()
    ):
        raise AcousticVerificationError("Real enrollment model inventory drifted.")
    method_ids = [
        str(method.get("method_id") or "")
        for method in authority.get("preparation_methods") or []
        if isinstance(method, Mapping)
    ]
    if method_ids != ["no_enhancement"]:
        raise AcousticVerificationError("Real enrollment preparation scope drifted.")
    profiles: list[dict[str, Any]] = []
    for unit in authority.get("enrollment_units") or []:
        if not isinstance(unit, Mapping):
            raise AcousticVerificationError("Real enrollment unit is invalid.")
        person_ref_id = str(unit.get("person_ref_id") or "")
        candidate = candidates.get(person_ref_id)
        if candidate is None:
            raise AcousticVerificationError("Real enrollment candidate disappeared.")
        resolved = resolve_eligible_reference(
            person_ref_id, runtime_root=p3_runtime_root
        )
        if (
            resolved.get("profile_id") != unit.get("p3_profile_id")
            or resolved.get("generation_id") != unit.get("p3_generation_id")
            or resolved.get("generation_sha256")
            != unit.get("p3_generation_sha256")
        ):
            raise AcousticVerificationError(
                "P3 generation changed before real enrollment apply."
            )
        sources = candidate.get("proposed_sources")
        if not isinstance(sources, list) or not sources:
            raise AcousticVerificationError("Real enrollment sources are unavailable.")
        windows = _authorized_real_windows(sources, method_id="no_enhancement")
        for candidate_id in sorted(expected_models):
            profile = _materialize_profile_core(
                resolved=resolved,
                adapter=selected_adapters[candidate_id],
                windows=windows,
                preprocessing={
                    "method_id": "no_enhancement",
                    "revision": EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256,
                },
                runtime_root=runtime_root,
                p3_runtime_root=p3_runtime_root,
            )
            profiles.append(profile)
    applied_at = utc_now()
    receipt = _real_enrollment_application_payload(
        authority=authority,
        authority_sha256=authority_sha256,
        profiles=profiles,
        applied_at=applied_at,
    )
    receipt_identity = {
        key: value for key, value in receipt.items() if key != "applied_at"
    }
    receipt_sha256 = canonical_artifact_hash(receipt_identity)
    root = runtime_root.expanduser().absolute()
    path = root / "enrollment-applications" / f"{receipt_sha256}.json"
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(
        path, receipt, volatile_fields=("applied_at",)
    )
    return {
        **stored,
        "application_sha256": receipt_sha256,
        "private_application_path": str(path),
    }


def replay_real_enrollment_application(
    application_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Replay a real enrollment receipt and every active P3/P4 binding."""
    if not SHA256_RE.fullmatch(str(application_sha256)):
        raise AcousticVerificationError("Enrollment application hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "enrollment-applications" / f"{application_sha256}.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    identity = {key: value for key, value in receipt.items() if key != "applied_at"}
    if canonical_artifact_hash(identity) != application_sha256:
        raise AcousticVerificationError("Enrollment application replay is invalid.")
    authority = replay_real_enrollment_apply_authority(
        str(receipt.get("authority_sha256") or ""), runtime_root=root
    )
    profiles = receipt.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != receipt.get("profile_count"):
        raise AcousticVerificationError("Enrollment application profiles are invalid.")
    expected_coverage = [
        (
            str(unit.get("person_ref_id") or ""),
            str(model.get("candidate_id") or ""),
            str(model.get("revision_sha") or ""),
            "no_enhancement",
            EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256,
            str(unit.get("p3_profile_id") or ""),
            str(unit.get("p3_generation_id") or ""),
            str(unit.get("p3_generation_sha256") or ""),
        )
        for unit in authority.get("enrollment_units") or []
        if isinstance(unit, Mapping)
        for model in sorted(
            authority.get("models") or [],
            key=lambda item: str(item.get("candidate_id") or "")
            if isinstance(item, Mapping)
            else "",
        )
        if isinstance(model, Mapping)
    ]
    actual_coverage: list[tuple[str, ...]] = []
    canonical_profiles: list[dict[str, Any]] = []
    for expected in profiles:
        if not isinstance(expected, Mapping):
            raise AcousticVerificationError("Enrollment application profile is invalid.")
        current = replay_profile(str(expected.get("profile_id") or ""), runtime_root=root)
        comparable_current = {key: current.get(key) for key in expected}
        if comparable_current != dict(expected) or not descendant_is_eligible(
            str(expected.get("descendant_id") or ""), runtime_root=p3_runtime_root
        ):
            raise AcousticVerificationError(
                "Enrollment application profile binding changed."
            )
        with _profile_database(root) as connection:
            profile_row = connection.execute(
                "SELECT preprocessing_json FROM profiles WHERE profile_id = ?",
                (str(expected.get("profile_id") or ""),),
            ).fetchone()
        if profile_row is None:
            raise AcousticVerificationError(
                "Enrollment application preprocessing is invalid."
            )
        try:
            preprocessing = json.loads(str(profile_row["preprocessing_json"]))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AcousticVerificationError(
                "Enrollment application preprocessing is invalid."
            ) from exc
        if not isinstance(preprocessing, Mapping):
            raise AcousticVerificationError(
                "Enrollment application preprocessing is invalid."
            )
        actual_coverage.append(
            (
                str(expected.get("person_ref_id") or ""),
                str(expected.get("candidate_id") or ""),
                str(expected.get("model_revision") or ""),
                str(preprocessing.get("method_id") or ""),
                str(preprocessing.get("revision") or ""),
                str(expected.get("p3_profile_id") or ""),
                str(expected.get("generation_id") or ""),
                str(expected.get("generation_sha256") or ""),
            )
        )
        canonical_profiles.append(dict(expected))
    if actual_coverage != expected_coverage:
        raise AcousticVerificationError(
            "Enrollment application profile coverage is invalid."
        )
    applied_at = str(receipt.get("applied_at") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", applied_at):
        raise AcousticVerificationError("Enrollment application time is invalid.")
    expected_receipt = _real_enrollment_application_payload(
        authority=authority,
        authority_sha256=str(receipt.get("authority_sha256") or ""),
        profiles=canonical_profiles,
        applied_at=applied_at,
    )
    if receipt != expected_receipt:
        raise AcousticVerificationError("Enrollment application semantics are invalid.")
    return {
        **receipt,
        "application_sha256": application_sha256,
        "private_application_path": str(path),
    }


def _development_trial_source_binding(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise AcousticVerificationError("Development trial source is invalid.")
    lineage = source.get("lineage")
    quality = source.get("quality_evidence")
    if not isinstance(lineage, Mapping) or not isinstance(quality, Mapping):
        raise AcousticVerificationError(
            "Development trial source evidence is invalid."
        )
    required = {
        "reference_id", "source_sha256", "recording_id", "conversation_id",
        "speaker_label_id", "session_id", "start_seconds", "end_seconds",
    }
    if not required.issubset(source):
        raise AcousticVerificationError(
            "Development trial source binding is incomplete."
        )
    return {
        "reference_id": source["reference_id"],
        "source_sha256": source["source_sha256"],
        "recording_id": source["recording_id"],
        "conversation_id": source["conversation_id"],
        "speaker_label_id": source["speaker_label_id"],
        "session_id": source["session_id"],
        "start_seconds": source["start_seconds"],
        "end_seconds": source["end_seconds"],
        "quality_evidence_sha256": quality.get("sha256"),
        "lineage_authority": lineage.get("authority"),
        "lineage_replay_receipt_sha256": lineage.get("replay_receipt_sha256"),
        "lineage_comparison_sha256": lineage.get("comparison_sha256"),
        "proposal_source_sha256": canonical_artifact_hash(dict(source)),
    }


def _development_trial_source_units(
    proposal: Mapping[str, Any], split_authority: Mapping[str, Any]
) -> list[dict[str, Any]]:
    allowed_pairs = split_authority["recording_conversation_pairs"]
    units: list[dict[str, Any]] = []
    for candidate in proposal.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            raise AcousticVerificationError("Development trial candidate is invalid.")
        person_ref_id = str(candidate.get("person_ref_id") or "")
        for source in candidate.get("proposed_sources") or []:
            if not isinstance(source, Mapping):
                raise AcousticVerificationError(
                    "Development trial source is invalid."
                )
            pair = (
                str(source.get("recording_id") or ""),
                str(source.get("conversation_id") or ""),
            )
            if pair not in allowed_pairs:
                raise AcousticVerificationError(
                    "Development trial source is outside the authorized split."
                )
            units.append(
                {
                    "person_ref_id": person_ref_id,
                    "source": _development_trial_source_binding(source),
                }
            )
    if not units:
        raise AcousticVerificationError("Development trial sources are unavailable.")
    return units


def _development_method_inventory(
    proposal: Mapping[str, Any], source_units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source_lookup = {
        (str(candidate.get("person_ref_id") or ""), str(source.get("reference_id") or "")): source
        for candidate in proposal.get("candidates") or []
        if isinstance(candidate, Mapping)
        for source in candidate.get("proposed_sources") or []
        if isinstance(source, Mapping)
    }
    representative_by_recording: dict[str, Mapping[str, Any]] = {}
    for unit in source_units:
        binding = unit.get("source") if isinstance(unit, Mapping) else None
        source = source_lookup.get(
            (
                str(unit.get("person_ref_id") or ""),
                str(binding.get("reference_id") or "")
                if isinstance(binding, Mapping)
                else "",
            )
        )
        if source is None:
            raise AcousticVerificationError(
                "Development trial source inventory changed."
            )
        representative_by_recording.setdefault(
            str(source.get("recording_id") or ""), source
        )
    inventory: list[dict[str, Any]] = []
    for recording_id, source in sorted(representative_by_recording.items()):
        lineage = source.get("lineage")
        if not isinstance(lineage, Mapping):
            raise AcousticVerificationError(
                "Development trial lineage is unavailable."
            )
        for method_id in METHOD_IDS:
            try:
                resolved = resolve_comparison_lineage_receipt(
                    str(lineage.get("run_id") or ""),
                    method_id=method_id,
                    replay_receipt_sha256=str(
                        lineage.get("replay_receipt_sha256") or ""
                    ),
                    runtime_root=Path(str(lineage.get("runtime_root") or "")),
                )
            except SpeechPreparationError as exc:
                raise AcousticVerificationError(
                    "Development trial method lineage is invalid."
                ) from exc
            comparison_path = Path(str(resolved.get("comparison_path") or ""))
            require_private_file(comparison_path, comparison_path.parent)
            if sha256_file(comparison_path) != resolved.get("comparison_sha256"):
                raise AcousticVerificationError(
                    "Development trial comparison drifted."
                )
            comparison = read_private_object(comparison_path)
            method = next(
                (
                    item
                    for item in comparison.get("method_results") or []
                    if isinstance(item, Mapping)
                    and item.get("method_id") == method_id
                ),
                None,
            )
            if (
                method is None
                or method.get("status") != "success"
                or canonical_artifact_hash(dict(method))
                != resolved.get("method_result_sha256")
                or not SHA256_RE.fullmatch(str(method.get("output_sha256") or ""))
            ):
                raise AcousticVerificationError(
                    "Development trial method result is invalid."
                )
            inventory.append(
                {
                    "recording_id": recording_id,
                    "conversation_id": source["conversation_id"],
                    "run_id": resolved["run_id"],
                    "replay_receipt_sha256": resolved[
                        "replay_receipt_sha256"
                    ],
                    "comparison_sha256": resolved["comparison_sha256"],
                    "method_id": method_id,
                    "method_result_sha256": resolved["method_result_sha256"],
                    "output_sha256": method["output_sha256"],
                    "output_equivalence_class_sha256": method[
                        "output_sha256"
                    ],
                    "pcm_contract": {
                        "channels": 1,
                        "sample_rate_hz": 16_000,
                        "sample_width_bytes": 2,
                        "compression": "NONE",
                    },
                }
            )
    return inventory


def _development_trial_authority_payload(
    *,
    application: Mapping[str, Any],
    application_sha256: str,
    proposal: Mapping[str, Any],
    split_authority: Mapping[str, Any],
    authorized_at: str,
) -> dict[str, Any]:
    profiles = application.get("profiles")
    if (
        application.get("status") != "success"
        or application.get("intended_split") != "development"
        or application.get("did_run_trials") is not False
        or application.get("did_read_calibration_or_evaluation") is not False
        or not isinstance(profiles, list)
        or not profiles
    ):
        raise AcousticVerificationError(
            "Development trial application dependency is invalid."
        )
    source_units = _development_trial_source_units(proposal, split_authority)
    method_inventory = _development_method_inventory(proposal, source_units)
    profile_people_by_model: dict[str, set[str]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise AcousticVerificationError("Development trial profile is invalid.")
        profile_people_by_model.setdefault(
            str(profile.get("candidate_id") or ""), set()
        ).add(str(profile.get("person_ref_id") or ""))
    if not profile_people_by_model or any(
        len(people) != 2 for people in profile_people_by_model.values()
    ):
        raise AcousticVerificationError(
            "Development trials require two people for every model."
        )
    logical_trials = len(source_units) * len(METHOD_IDS) * len(profiles)
    unique_probe_waveforms = sum(
        len(
            {
                item["output_sha256"]
                for item in method_inventory
                if item["recording_id"] == unit["source"]["recording_id"]
            }
        )
        for unit in source_units
    )
    authority = {
        "schema_version": DEVELOPMENT_TRIAL_AUTHORITY_SCHEMA,
        "status": "authorized",
        "reason_code": None,
        "authorization_basis": REAL_ENROLLMENT_AUTHORIZATION_BASIS,
        "authorized_by_ref_id": REAL_ENROLLMENT_AUTHORIZER_REF_ID,
        "authorized_at": authorized_at,
        "intended_split": "development",
        "enrollment_application_sha256": application_sha256,
        "candidate_proposal_sha256": application["candidate_proposal_sha256"],
        "enrollment_preview_sha256": application["enrollment_preview_sha256"],
        "split_access_policy_sha256": split_authority[
            "split_access_policy_sha256"
        ],
        "parent_corpus_manifest_sha256": split_authority[
            "parent_corpus_manifest_sha256"
        ],
        "development_record_set_sha256": split_authority[
            "development_record_set_sha256"
        ],
        "development_conversation_set_sha256": split_authority[
            "development_conversation_set_sha256"
        ],
        "development_comparison_receipt_sha256": (
            EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256
        ),
        "source_units": source_units,
        "preparation_methods": list(METHOD_IDS),
        "method_inventory": method_inventory,
        "profiles": [dict(profile) for profile in profiles],
        "expected_coverage": {
            "logical_trials": logical_trials,
            "genuine_trials": logical_trials // 2,
            "impostor_trials": logical_trials // 2,
            "unique_probe_waveforms": unique_probe_waveforms,
            "unique_waveform_model_profile_combinations": (
                unique_probe_waveforms * len(profiles)
            ),
        },
        "evidence_class": "development_resubstitution_diagnostic",
        "held_out": False,
        "enrollment_probe_overlap": True,
        "permits_generalization_claim": False,
        "permits_accuracy_far_frr_eer_claim": False,
        "permits_threshold_or_model_selection": False,
        "will_read_audio": True,
        "will_run_trials": True,
        "will_select_threshold": False,
        "will_read_calibration_or_evaluation": False,
        "will_perform_external_write": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if _contains_forbidden_private_key(authority):
        raise AcousticVerificationError(
            "Development trial authority contains forbidden private data."
        )
    return authority


def _validate_development_trial_authority(
    value: Any,
    *,
    application: Mapping[str, Any],
    application_sha256: str,
    proposal: Mapping[str, Any],
    split_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AcousticVerificationError("Development trial authority is invalid.")
    authorized_at = str(value.get("authorized_at") or "")
    expected = _development_trial_authority_payload(
        application=application,
        application_sha256=application_sha256,
        proposal=proposal,
        split_authority=split_authority,
        authorized_at=authorized_at,
    )
    if dict(value) != expected or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", authorized_at
    ):
        raise AcousticVerificationError("Development trial authority is invalid.")
    return dict(value)


def build_development_trial_authority(
    enrollment_application_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Persist one exact development-only trial authority without opening audio."""
    root = runtime_root.expanduser().absolute()
    application = replay_real_enrollment_application(
        enrollment_application_sha256,
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
    )
    proposal = replay_real_enrollment_candidate_proposal(
        str(application["candidate_proposal_sha256"]), runtime_root=root
    )
    split_authority = _development_split_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    authority_dir = root / "development-trial-authorities"
    ensure_private_tree(root, authority_dir)
    matches: list[tuple[Path, dict[str, Any], str]] = []
    for path in sorted(authority_dir.glob("*.json")):
        require_private_file(path, root)
        value = read_private_object(path)
        if value.get("enrollment_application_sha256") != enrollment_application_sha256:
            continue
        _validate_development_trial_authority(
            value,
            application=application,
            application_sha256=enrollment_application_sha256,
            proposal=proposal,
            split_authority=split_authority,
        )
        value_sha = canonical_artifact_hash(value)
        if path.name != f"{value_sha}.json":
            raise AcousticVerificationError(
                "Development trial authority path is invalid."
            )
        matches.append((path, value, value_sha))
    if len(matches) > 1:
        raise AcousticVerificationError(
            "Multiple development trial authorities exist for one application."
        )
    if matches:
        path, authority, authority_sha = matches[0]
    else:
        authority = _development_trial_authority_payload(
            application=application,
            application_sha256=enrollment_application_sha256,
            proposal=proposal,
            split_authority=split_authority,
            authorized_at=utc_now(),
        )
        authority_sha = canonical_artifact_hash(authority)
        path = authority_dir / f"{authority_sha}.json"
        write_immutable_private_json(path, authority)
    return {
        **authority,
        "authority_sha256": authority_sha,
        "private_authority_path": str(path),
    }


def replay_development_trial_authority(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Replay an exact P4D development authority with sealed later splits."""
    if not SHA256_RE.fullmatch(str(authority_sha256)):
        raise AcousticVerificationError("Development trial authority hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "development-trial-authorities" / f"{authority_sha256}.json"
    require_private_file(path, root)
    authority = read_private_object(path)
    application_sha = str(authority.get("enrollment_application_sha256") or "")
    application = replay_real_enrollment_application(
        application_sha, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    proposal = replay_real_enrollment_candidate_proposal(
        str(application["candidate_proposal_sha256"]), runtime_root=root
    )
    split_authority = _development_split_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    _validate_development_trial_authority(
        authority,
        application=application,
        application_sha256=application_sha,
        proposal=proposal,
        split_authority=split_authority,
    )
    if canonical_artifact_hash(authority) != authority_sha256:
        raise AcousticVerificationError("Development trial authority replay is invalid.")
    return {
        **authority,
        "authority_sha256": authority_sha256,
        "private_authority_path": str(path),
    }


def _development_trial_application_payload(
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    trials: Sequence[Mapping[str, Any]],
    applied_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENT_TRIAL_APPLICATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha256,
        "intended_split": "development",
        "enrollment_application_sha256": authority[
            "enrollment_application_sha256"
        ],
        "candidate_proposal_sha256": authority["candidate_proposal_sha256"],
        "development_comparison_receipt_sha256": authority[
            "development_comparison_receipt_sha256"
        ],
        "evidence_class": "development_resubstitution_diagnostic",
        "held_out": False,
        "enrollment_probe_overlap": True,
        "permits_generalization_claim": False,
        "permits_accuracy_far_frr_eer_claim": False,
        "permits_threshold_or_model_selection": False,
        "denominators": {
            "attempted": len(trials),
            "success": len(trials),
            "failure": 0,
            "blocked": 0,
        },
        "trials": [dict(trial) for trial in trials],
        "did_read_audio": True,
        "did_run_trials": True,
        "did_select_threshold": False,
        "scores_recomputed_during_replay": False,
        "score_replay_scope": "structural_identity_and_authority_only",
        "did_read_calibration_or_evaluation": False,
        "did_mutate_profiles_or_references": False,
        "did_perform_external_write": False,
        "contains_biometric_scores": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
        "applied_at": applied_at,
    }


def apply_development_trials(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    adapters: Optional[Mapping[str, VerificationAdapter]] = None,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Run the exact P4D development matrix without reading later splits."""
    if adapters is not None and not test_mode:
        raise AcousticVerificationError(
            "Custom trial adapters are limited to deterministic tests."
        )
    root = runtime_root.expanduser().absolute()
    authority = replay_development_trial_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    application_sha = str(authority["enrollment_application_sha256"])
    application = replay_real_enrollment_application(
        application_sha, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    proposal = replay_real_enrollment_candidate_proposal(
        str(authority["candidate_proposal_sha256"]), runtime_root=root
    )
    application_dir = root / "development-trial-applications"
    if application_dir.exists():
        matches = []
        for existing_path in sorted(application_dir.glob("*.json")):
            require_private_file(existing_path, root)
            existing = read_private_object(existing_path)
            if existing.get("authority_sha256") != authority_sha256:
                continue
            matches.append(
                replay_development_trial_application(
                    existing_path.stem,
                    runtime_root=root,
                    p3_runtime_root=p3_runtime_root,
                )
            )
        if len(matches) > 1:
            raise AcousticVerificationError(
                "Multiple development trial applications exist for one authority."
            )
        if matches:
            return matches[0]
    source_lookup = {
        (str(candidate.get("person_ref_id") or ""), str(source.get("reference_id") or "")): source
        for candidate in proposal.get("candidates") or []
        if isinstance(candidate, Mapping)
        for source in candidate.get("proposed_sources") or []
        if isinstance(source, Mapping)
    }
    selected_adapters = dict(adapters or adapter_registry())
    expected_models = {
        str(profile.get("candidate_id") or ""): str(profile.get("model_revision") or "")
        for profile in authority.get("profiles") or []
        if isinstance(profile, Mapping)
    }
    if set(selected_adapters) != set(expected_models) or any(
        selected_adapters[candidate_id].revision_sha != revision
        for candidate_id, revision in expected_models.items()
    ):
        raise AcousticVerificationError("Development trial model inventory drifted.")
    profiles_by_model = {
        candidate_id: [
            dict(profile)
            for profile in authority.get("profiles") or []
            if isinstance(profile, Mapping)
            and profile.get("candidate_id") == candidate_id
        ]
        for candidate_id in sorted(expected_models)
    }
    if any(
        len(model_profiles) != 2
        or len({profile["person_ref_id"] for profile in model_profiles}) != 2
        for model_profiles in profiles_by_model.values()
    ):
        raise AcousticVerificationError(
            "Development trials require two distinct profiles per model."
        )
    trials: list[dict[str, Any]] = []
    for unit in authority.get("source_units") or []:
        source_binding = unit.get("source") if isinstance(unit, Mapping) else None
        if not isinstance(source_binding, Mapping):
            raise AcousticVerificationError("Development trial source unit is invalid.")
        probe_person_ref_id = str(unit.get("person_ref_id") or "")
        source = source_lookup.get(
            (probe_person_ref_id, str(source_binding.get("reference_id") or ""))
        )
        if (
            source is None
            or _development_trial_source_binding(source) != dict(source_binding)
        ):
            raise AcousticVerificationError("Development trial source binding changed.")
        for method_id in authority.get("preparation_methods") or []:
            windows = _authorized_real_windows([source], method_id=str(method_id))
            if len(windows) != 1:
                raise AcousticVerificationError("Development trial window is invalid.")
            for candidate_id in sorted(expected_models):
                adapter = selected_adapters[candidate_id]
                for profile in profiles_by_model[candidate_id]:
                    scored = score_profile(
                        str(profile["profile_id"]),
                        adapter=adapter,
                        probe_samples=windows[0]["samples"],
                        sample_rate=16_000,
                        runtime_root=root,
                        p3_runtime_root=p3_runtime_root,
                    )
                    identity = {
                        "authority_sha256": authority_sha256,
                        "reference_id": source_binding["reference_id"],
                        "method_id": method_id,
                        "profile_id": profile["profile_id"],
                        "score_trial_id": scored["trial_id"],
                    }
                    trials.append(
                        {
                            "trial_id": "development-trial-"
                            + canonical_artifact_hash(identity)[:24],
                            "status": "success",
                            "reason_code": None,
                            "reference_id": source_binding["reference_id"],
                            "recording_id": source_binding["recording_id"],
                            "conversation_id": source_binding["conversation_id"],
                            "probe_person_ref_id": probe_person_ref_id,
                            "profile_person_ref_id": profile["person_ref_id"],
                            "expected_match": (
                                probe_person_ref_id == profile["person_ref_id"]
                            ),
                            "method_id": method_id,
                            "profile_id": profile["profile_id"],
                            "descendant_id": profile["descendant_id"],
                            "candidate_id": candidate_id,
                            "model_revision": adapter.revision_sha,
                            "probe_sha256": scored["probe_sha256"],
                            "score_trial_id": scored["trial_id"],
                            "score": scored["score"],
                            "p4_state_verified_before_and_after": True,
                            "p3_eligibility_verified_before_and_after": True,
                            "contains_raw_biometric_values": False,
                        }
                    )
    expected_coverage = authority.get("expected_coverage")
    if not isinstance(expected_coverage, Mapping):
        raise AcousticVerificationError("Development trial coverage is invalid.")
    genuine_count = sum(trial["expected_match"] is True for trial in trials)
    impostor_count = sum(trial["expected_match"] is False for trial in trials)
    unique_combinations = {
        (
            trial["probe_sha256"],
            trial["candidate_id"],
            trial["profile_id"],
        )
        for trial in trials
    }
    if (
        len(trials) != expected_coverage.get("logical_trials")
        or genuine_count != expected_coverage.get("genuine_trials")
        or impostor_count != expected_coverage.get("impostor_trials")
        or len(unique_combinations)
        != expected_coverage.get("unique_waveform_model_profile_combinations")
    ):
        raise AcousticVerificationError(
            "Development trial observed coverage is invalid."
        )
    applied_at = utc_now()
    receipt = _development_trial_application_payload(
        authority=authority,
        authority_sha256=authority_sha256,
        trials=trials,
        applied_at=applied_at,
    )
    receipt_sha = canonical_artifact_hash(
        {key: value for key, value in receipt.items() if key != "applied_at"}
    )
    path = root / "development-trial-applications" / f"{receipt_sha}.json"
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("applied_at",))
    return {
        **stored,
        "application_sha256": receipt_sha,
        "private_application_path": str(path),
    }


def replay_development_trial_application(
    application_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Replay P4D structure and live eligibility without recomputing scores."""
    if not SHA256_RE.fullmatch(str(application_sha256)):
        raise AcousticVerificationError(
            "Development trial application hash is invalid."
        )
    root = runtime_root.expanduser().absolute()
    path = root / "development-trial-applications" / f"{application_sha256}.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    identity = {key: value for key, value in receipt.items() if key != "applied_at"}
    if canonical_artifact_hash(identity) != application_sha256:
        raise AcousticVerificationError(
            "Development trial application replay is invalid."
        )
    authority = replay_development_trial_authority(
        str(receipt.get("authority_sha256") or ""),
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
    )
    profiles = authority.get("profiles")
    if not isinstance(profiles, list):
        raise AcousticVerificationError("Development trial profiles are invalid.")
    profile_by_id: dict[str, dict[str, Any]] = {}
    profiles_by_model: dict[str, list[dict[str, Any]]] = {}
    for expected_profile in profiles:
        if not isinstance(expected_profile, Mapping):
            raise AcousticVerificationError(
                "Development trial profile is invalid."
            )
        profile = dict(expected_profile)
        current = replay_profile(str(profile.get("profile_id") or ""), runtime_root=root)
        if (
            {key: current.get(key) for key in profile} != profile
            or not descendant_is_eligible(
                str(profile.get("descendant_id") or ""),
                runtime_root=p3_runtime_root,
            )
        ):
            raise AcousticVerificationError(
                "Development trial profile binding changed."
            )
        profile_by_id[str(profile["profile_id"])] = profile
        profiles_by_model.setdefault(str(profile["candidate_id"]), []).append(profile)
    source_units = authority.get("source_units")
    methods = authority.get("preparation_methods")
    if not isinstance(source_units, list) or not isinstance(methods, list):
        raise AcousticVerificationError("Development trial coverage is invalid.")
    expected_coverage = [
        (
            str(unit["source"]["reference_id"]),
            str(method_id),
            str(candidate_id),
            str(profile["profile_id"]),
        )
        for unit in source_units
        if isinstance(unit, Mapping) and isinstance(unit.get("source"), Mapping)
        for method_id in methods
        for candidate_id in sorted(profiles_by_model)
        for profile in profiles_by_model[candidate_id]
    ]
    trials = receipt.get("trials")
    if not isinstance(trials, list):
        raise AcousticVerificationError("Development trial results are invalid.")
    source_by_reference = {
        str(unit["source"]["reference_id"]): unit
        for unit in source_units
        if isinstance(unit, Mapping) and isinstance(unit.get("source"), Mapping)
    }
    actual_coverage: list[tuple[str, str, str, str]] = []
    canonical_trials: list[dict[str, Any]] = []
    expected_trial_keys = {
        "trial_id", "status", "reason_code", "reference_id", "recording_id",
        "conversation_id", "probe_person_ref_id", "profile_person_ref_id",
        "expected_match", "method_id", "profile_id", "descendant_id",
        "candidate_id", "model_revision", "probe_sha256", "score_trial_id",
        "score", "p4_state_verified_before_and_after",
        "p3_eligibility_verified_before_and_after",
        "contains_raw_biometric_values",
    }
    for trial_value in trials:
        if not isinstance(trial_value, Mapping) or set(trial_value) != expected_trial_keys:
            raise AcousticVerificationError("Development trial result shape is invalid.")
        trial = dict(trial_value)
        source_unit = source_by_reference.get(str(trial.get("reference_id") or ""))
        profile = profile_by_id.get(str(trial.get("profile_id") or ""))
        source = source_unit.get("source") if source_unit else None
        score = trial.get("score")
        if (
            source_unit is None
            or profile is None
            or not isinstance(source, Mapping)
            or trial.get("status") != "success"
            or trial.get("reason_code") is not None
            or trial.get("recording_id") != source.get("recording_id")
            or trial.get("conversation_id") != source.get("conversation_id")
            or trial.get("probe_person_ref_id") != source_unit.get("person_ref_id")
            or trial.get("profile_person_ref_id") != profile.get("person_ref_id")
            or trial.get("expected_match")
            is not (
                source_unit.get("person_ref_id") == profile.get("person_ref_id")
            )
            or trial.get("candidate_id") != profile.get("candidate_id")
            or trial.get("model_revision") != profile.get("model_revision")
            or trial.get("descendant_id") != profile.get("descendant_id")
            or trial.get("method_id") not in methods
            or not SHA256_RE.fullmatch(str(trial.get("probe_sha256") or ""))
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1.0 <= float(score) <= 1.0
            or trial.get("p4_state_verified_before_and_after") is not True
            or trial.get("p3_eligibility_verified_before_and_after") is not True
            or trial.get("contains_raw_biometric_values") is not False
        ):
            raise AcousticVerificationError("Development trial result is invalid.")
        score_identity = {
            "profile_id": profile["profile_id"],
            "descendant_id": profile["descendant_id"],
            "artifact_sha256": profile["artifact_sha256"],
            "candidate_id": profile["candidate_id"],
            "model_revision": profile["model_revision"],
            "probe_sha256": trial["probe_sha256"],
            "score": score,
        }
        expected_score_trial_id = (
            "verification-trial-" + canonical_artifact_hash(score_identity)[:24]
        )
        wrapper_identity = {
            "authority_sha256": authority["authority_sha256"],
            "reference_id": trial["reference_id"],
            "method_id": trial["method_id"],
            "profile_id": trial["profile_id"],
            "score_trial_id": expected_score_trial_id,
        }
        if (
            trial.get("score_trial_id") != expected_score_trial_id
            or trial.get("trial_id")
            != "development-trial-" + canonical_artifact_hash(wrapper_identity)[:24]
        ):
            raise AcousticVerificationError(
                "Development trial identity is invalid."
            )
        actual_coverage.append(
            (
                str(trial["reference_id"]),
                str(trial["method_id"]),
                str(trial["candidate_id"]),
                str(trial["profile_id"]),
            )
        )
        canonical_trials.append(trial)
    if actual_coverage != expected_coverage:
        raise AcousticVerificationError("Development trial coverage is invalid.")
    coverage = authority.get("expected_coverage")
    unique_combinations = {
        (trial["probe_sha256"], trial["candidate_id"], trial["profile_id"])
        for trial in canonical_trials
    }
    if (
        not isinstance(coverage, Mapping)
        or len(canonical_trials) != coverage.get("logical_trials")
        or sum(trial["expected_match"] is True for trial in canonical_trials)
        != coverage.get("genuine_trials")
        or sum(trial["expected_match"] is False for trial in canonical_trials)
        != coverage.get("impostor_trials")
        or len(unique_combinations)
        != coverage.get("unique_waveform_model_profile_combinations")
    ):
        raise AcousticVerificationError(
            "Development trial denominators are invalid."
        )
    applied_at = str(receipt.get("applied_at") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", applied_at):
        raise AcousticVerificationError("Development trial application time is invalid.")
    expected_receipt = _development_trial_application_payload(
        authority=authority,
        authority_sha256=str(receipt.get("authority_sha256") or ""),
        trials=canonical_trials,
        applied_at=applied_at,
    )
    if receipt != expected_receipt:
        raise AcousticVerificationError(
            "Development trial application semantics are invalid."
        )
    return {
        **receipt,
        "application_sha256": application_sha256,
        "private_application_path": str(path),
    }


def _calibration_split_metadata_authority(
    split_policy_path: Path, parent_corpus_manifest_path: Path
) -> dict[str, Any]:
    """Validate calibration scope hashes without revealing calibration records."""
    policy_path = split_policy_path.expanduser().absolute()
    parent_path = parent_corpus_manifest_path.expanduser().absolute()
    if policy_path.is_symlink() or not policy_path.is_file():
        raise AcousticVerificationError("Split access policy is unavailable.")
    if sha256_file(policy_path) != EXPECTED_SPLIT_ACCESS_POLICY_SHA256:
        raise AcousticVerificationError("Split access policy hash is invalid.")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError("Split access policy is unreadable.") from exc
    require_private_file(parent_path, parent_path.parent)
    if sha256_file(parent_path) != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256:
        raise AcousticVerificationError("Parent corpus manifest hash is invalid.")
    calibration = policy.get("splits", {}).get("calibration")
    evaluation = policy.get("splits", {}).get("evaluation")
    if (
        policy.get("schema_version")
        != "transcribe-audio.verification-split-access-policy.v1"
        or policy.get("parent_corpus_manifest_sha256")
        != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256
        or not isinstance(calibration, Mapping)
        or calibration.get("authorization_state")
        != "not_authorized_pending_exact_calibration_apply"
        or calibration.get("recording_count") != 3
        or calibration.get("conversation_count") != 3
        or calibration.get("record_set_sha256")
        != EXPECTED_CALIBRATION_RECORD_SET_SHA256
        or calibration.get("conversation_set_sha256")
        != EXPECTED_CALIBRATION_CONVERSATION_SET_SHA256
        or not isinstance(evaluation, Mapping)
        or evaluation.get("authorization_state")
        != "not_authorized_pending_exact_terminal_evaluation_apply"
    ):
        raise AcousticVerificationError("Calibration split metadata is invalid.")
    return {
        "split_access_policy_sha256": EXPECTED_SPLIT_ACCESS_POLICY_SHA256,
        "parent_corpus_manifest_sha256": EXPECTED_PARENT_CORPUS_MANIFEST_SHA256,
        "calibration_record_set_sha256": EXPECTED_CALIBRATION_RECORD_SET_SHA256,
        "calibration_conversation_set_sha256": (
            EXPECTED_CALIBRATION_CONVERSATION_SET_SHA256
        ),
        "calibration_recording_count": 3,
        "calibration_conversation_count": 3,
    }


def _calibration_apply_authority_payload(
    *,
    development_application: Mapping[str, Any],
    development_application_sha256: str,
    development_authority: Mapping[str, Any],
    split_metadata: Mapping[str, Any],
    authorized_at: str,
) -> dict[str, Any]:
    if (
        development_application.get("status") != "success"
        or development_application.get("intended_split") != "development"
        or development_application.get("did_run_trials") is not True
        or development_application.get("did_select_threshold") is not False
        or development_application.get("did_read_calibration_or_evaluation")
        is not False
        or development_application.get("contains_biometric_scores") is not True
    ):
        raise AcousticVerificationError(
            "Calibration authority requires a verified development receipt."
        )
    profiles = development_authority.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 6:
        raise AcousticVerificationError(
            "Calibration authority profile inventory is invalid."
        )
    threshold_policy = {
        "unit": "candidate_model_preparation_path",
        "labels": ["genuine", "impostor"],
        "threshold_candidates": (
            "sorted_unique_score_midpoints_plus_negative_one_and_positive_one"
        ),
        "temperature_candidates": [0.01, 0.025, 0.05, 0.1, 0.2],
        "selection_order": [
            "minimum_brier_score",
            "minimum_expected_calibration_error_5_equal_width_bins",
            "minimum_balanced_error_rate",
            "minimum_absolute_far_minus_frr",
            "highest_threshold",
            "lowest_temperature",
        ],
        "minimum_genuine_trials_per_unit": 3,
        "minimum_impostor_trials_per_unit": 3,
        "score_range": [-1.0, 1.0],
        "probability_mapping": "sigmoid((score-threshold)/temperature)",
        "eer_is_diagnostic_only": True,
        "evaluation_may_not_change_policy": True,
    }
    metric_policy = {
        "classification_rule": "accept_when_score_greater_than_or_equal_to_threshold",
        "false_acceptance_rate": "false_accepts_divided_by_impostor_trials",
        "false_rejection_rate": "false_rejects_divided_by_genuine_trials",
        "balanced_error_rate": "mean(false_acceptance_rate,false_rejection_rate)",
        "eer_diagnostic": (
            "candidate_threshold_minimizing_absolute_far_minus_frr_then_"
            "balanced_error_rate_then_highest_threshold"
        ),
        "brier_score": "mean((mapped_probability-binary_label)^2)",
        "expected_calibration_error": (
            "sum(bin_count/total*abs(mean_probability-bin_positive_rate))_"
            "over_5_equal_width_bins"
        ),
        "candidate_margin": "highest_profile_score_minus_second_highest_profile_score",
        "open_set_rejection": (
            "open_set_probe_rejected_when_all_profile_scores_below_their_"
            "model_method_thresholds"
        ),
        "abstention": (
            "not_run_without_separately_precommitted_abstention_margin"
        ),
        "condition_slices": [
            "channel", "device", "noise", "overlap",
            "telephone_bandwidth", "usable_duration_band",
        ],
        "missing_denominator": "status_not_run_and_numeric_value_null",
        "results_are_descriptive": True,
        "conversation_clustered_non_independent": True,
        "permits_generalization_claim": False,
    }
    aggregation_policy = {
        "score_unit": "one_clean_window_against_one_fixed_profile",
        "profile_aggregation": "fixed_enrollment_centroid_only",
        "score_normalization": "none",
        "threshold_input": "raw_cosine_score",
        "same_timestamp_bounds_across_score_methods": True,
        "method_output_duplicates_are_one_equivalence_class": True,
        "calibration_may_not_change_profiles_features_or_window_rules": True,
    }
    p1_module_path = Path(audio_derivatives.__file__).resolve()
    p2_module_path = Path(speech_preparation.__file__).resolve()
    open_manifest_path = (
        speech_preparation.DEFAULT_OPEN_ACQUISITION_MANIFEST.expanduser().absolute()
    )
    pyannote_manifest_path = (
        speech_preparation.DEFAULT_PYANNOTE_ACQUISITION_MANIFEST.expanduser().absolute()
    )
    require_private_file(open_manifest_path, open_manifest_path.parent)
    require_private_file(pyannote_manifest_path, pyannote_manifest_path.parent)
    if (
        sha256_file(open_manifest_path)
        != EXPECTED_P2_OPEN_ACQUISITION_MANIFEST_SHA256
        or sha256_file(pyannote_manifest_path)
        != EXPECTED_P2_PYANNOTE_ACQUISITION_MANIFEST_SHA256
    ):
        raise AcousticVerificationError(
            "Calibration preparation acquisition authority drifted."
        )
    authority = {
        "schema_version": CALIBRATION_APPLY_AUTHORITY_SCHEMA,
        "status": "authorized",
        "reason_code": None,
        "authorization_basis": REAL_ENROLLMENT_AUTHORIZATION_BASIS,
        "authorized_by_ref_id": REAL_ENROLLMENT_AUTHORIZER_REF_ID,
        "authorized_at": authorized_at,
        "authority_generation": 2,
        "supersedes_authority_sha256": (
            SUPERSEDED_CALIBRATION_AUTHORITY_SHA256
        ),
        "supersession_reason": (
            "stereo_source_requires_explicit_channel_policy"
        ),
        "prior_generation_did_not_run_p2_or_biometric_scoring": True,
        "intended_split": "calibration",
        "development_application_sha256": development_application_sha256,
        "development_authority_sha256": development_application[
            "authority_sha256"
        ],
        "enrollment_application_sha256": development_authority[
            "enrollment_application_sha256"
        ],
        **dict(split_metadata),
        "preparation_methods": list(METHOD_IDS),
        "score_methods": list(CALIBRATION_SCORE_METHOD_IDS),
        "window_selection_methods": ["silero_vad", "pyannote_community_1"],
        "profiles": [dict(profile) for profile in profiles],
        "window_policy": {
            "minimum_seconds": 0.75,
            "maximum_seconds": 15.0,
            "maximum_windows_per_speaker_per_conversation": 3,
            "exclude_mixed_or_unknown_gold": True,
            "exclude_overlap_and_speaker_change_regions": True,
            "preserve_original_timestamps": True,
            "allowed_outcome": "person_with_opaque_subject_id",
            "pre_score_exclusion_reasons": [
                "mixed_gold", "unknown_gold", "missing_subject_id",
                "no_speech_intersection", "overlap_or_change_region",
                "shorter_than_minimum", "duplicate_or_overlapping_window",
            ],
        },
        "aggregation_policy": aggregation_policy,
        "threshold_policy": threshold_policy,
        "metric_policy": metric_policy,
        "preparation_contract": {
            "p1_module_sha256": sha256_file(p1_module_path),
            "p2_module_sha256": sha256_file(p2_module_path),
            "p2_open_acquisition_manifest_sha256": (
                EXPECTED_P2_OPEN_ACQUISITION_MANIFEST_SHA256
            ),
            "p2_pyannote_acquisition_manifest_sha256": (
                EXPECTED_P2_PYANNOTE_ACQUISITION_MANIFEST_SHA256
            ),
            "pcm_channels": 1,
            "pcm_sample_rate_hz": 16_000,
            "pcm_sample_width_bytes": 2,
            "pcm_compression": "NONE",
            "channel_policy": {
                "allowed_source_channels": [1, 2],
                "mono": "identity",
                "stereo": (
                    "arithmetic_average_0.5_left_plus_0.5_right"
                ),
                "output_channels": 1,
                "authority_binding": "this_calibration_authority_sha256",
                "no_silent_fallback": True,
            },
            "no_fallback_method": True,
        },
        "will_prepare_calibration_audio": True,
        "will_read_calibration_gold": True,
        "will_run_calibration_trials": True,
        "will_select_and_freeze_thresholds": True,
        "will_read_evaluation": False,
        "will_perform_external_write": False,
        "will_mutate_profiles_or_references": False,
        "will_enable_default_integration": False,
        "will_make_terminal_model_or_method_selection": False,
        "contains_biometric_scores": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if _contains_forbidden_private_key(authority):
        raise AcousticVerificationError(
            "Calibration authority contains forbidden private data."
        )
    return authority


def _validate_calibration_apply_authority(
    value: Any,
    *,
    development_application: Mapping[str, Any],
    development_application_sha256: str,
    development_authority: Mapping[str, Any],
    split_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AcousticVerificationError("Calibration authority is invalid.")
    authorized_at = str(value.get("authorized_at") or "")
    expected = _calibration_apply_authority_payload(
        development_application=development_application,
        development_application_sha256=development_application_sha256,
        development_authority=development_authority,
        split_metadata=split_metadata,
        authorized_at=authorized_at,
    )
    if dict(value) != expected or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", authorized_at
    ):
        raise AcousticVerificationError("Calibration authority is invalid.")
    return dict(value)


def build_calibration_apply_authority(
    development_application_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Create the exact P4D2 authority before revealing calibration records."""
    root = runtime_root.expanduser().absolute()
    development = replay_development_trial_application(
        development_application_sha256,
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
    )
    development_authority = replay_development_trial_authority(
        str(development["authority_sha256"]),
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
    )
    split_metadata = _calibration_split_metadata_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    authority_dir = root / "calibration-authorities"
    ensure_private_tree(root, authority_dir)
    matches: list[tuple[Path, dict[str, Any], str]] = []
    for path in sorted(authority_dir.glob("*.json")):
        require_private_file(path, root)
        value = read_private_object(path)
        if value.get("development_application_sha256") != development_application_sha256:
            continue
        _validate_calibration_apply_authority(
            value,
            development_application=development,
            development_application_sha256=development_application_sha256,
            development_authority=development_authority,
            split_metadata=split_metadata,
        )
        value_sha = canonical_artifact_hash(value)
        if path.name != f"{value_sha}.json":
            raise AcousticVerificationError("Calibration authority path is invalid.")
        matches.append((path, value, value_sha))
    if len(matches) > 1:
        raise AcousticVerificationError(
            "Multiple calibration authorities exist for one development receipt."
        )
    if matches:
        path, authority, authority_sha = matches[0]
    else:
        authority = _calibration_apply_authority_payload(
            development_application=development,
            development_application_sha256=development_application_sha256,
            development_authority=development_authority,
            split_metadata=split_metadata,
            authorized_at=utc_now(),
        )
        authority_sha = canonical_artifact_hash(authority)
        path = authority_dir / f"{authority_sha}.json"
        write_immutable_private_json(path, authority)
    return {
        **authority,
        "authority_sha256": authority_sha,
        "private_authority_path": str(path),
    }


def replay_calibration_apply_authority(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Replay P4D2 authority while evaluation remains sealed."""
    if not SHA256_RE.fullmatch(str(authority_sha256)):
        raise AcousticVerificationError("Calibration authority hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "calibration-authorities" / f"{authority_sha256}.json"
    require_private_file(path, root)
    authority = read_private_object(path)
    development_sha = str(authority.get("development_application_sha256") or "")
    development = replay_development_trial_application(
        development_sha, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    development_authority = replay_development_trial_authority(
        str(development["authority_sha256"]),
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
    )
    split_metadata = _calibration_split_metadata_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    _validate_calibration_apply_authority(
        authority,
        development_application=development,
        development_application_sha256=development_sha,
        development_authority=development_authority,
        split_metadata=split_metadata,
    )
    if canonical_artifact_hash(authority) != authority_sha256:
        raise AcousticVerificationError("Calibration authority replay is invalid.")
    return {
        **authority,
        "authority_sha256": authority_sha256,
        "private_authority_path": str(path),
    }


def _calibration_records_after_authority(
    authority: Mapping[str, Any], *, parent_corpus_manifest_path: Path
) -> list[dict[str, Any]]:
    if authority.get("intended_split") != "calibration":
        raise AcousticVerificationError("Calibration split authority is invalid.")
    parent_path = parent_corpus_manifest_path.expanduser().absolute()
    require_private_file(parent_path, parent_path.parent)
    if sha256_file(parent_path) != authority.get("parent_corpus_manifest_sha256"):
        raise AcousticVerificationError("Calibration parent manifest drifted.")
    parent = read_private_object(parent_path)
    recordings = parent.get("recordings")
    if not isinstance(recordings, list):
        raise AcousticVerificationError("Calibration parent records are invalid.")
    by_split: dict[str, list[Mapping[str, Any]]] = {
        split: [
            record
            for record in recordings
            if isinstance(record, Mapping) and record.get("split") == split
        ]
        for split in ("development", "calibration", "evaluation")
    }
    selected = by_split["calibration"]
    if (
        len(selected) != authority.get("calibration_recording_count")
        or canonical_artifact_hash(selected)
        != authority.get("calibration_record_set_sha256")
        or canonical_artifact_hash(
            sorted(str(record.get("conversation_id") or "") for record in selected)
        )
        != authority.get("calibration_conversation_set_sha256")
    ):
        raise AcousticVerificationError("Calibration split membership drifted.")
    for key in ("recording_id", "conversation_id"):
        split_sets = [
            {str(record.get(key) or "") for record in by_split[split]}
            for split in ("development", "calibration", "evaluation")
        ]
        if any(
            split_sets[left] & split_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise AcousticVerificationError(
                f"Calibration {key} overlaps another split."
            )
    source_sets = [
        {
            str((record.get("source_blob") or {}).get("sha256") or "")
            for record in by_split[split]
            if isinstance(record.get("source_blob"), Mapping)
        }
        for split in ("development", "calibration", "evaluation")
    ]
    if any(
        source_sets[left] & source_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AcousticVerificationError(
            "Calibration source content overlaps another split."
        )
    validated: list[dict[str, Any]] = []
    for record_value in selected:
        record = dict(record_value)
        source = record.get("source_blob")
        lineage = record.get("transcript_lineage")
        gold = record.get("operator_gold")
        if (
            not isinstance(source, Mapping)
            or not isinstance(lineage, Mapping)
            or not isinstance(gold, Mapping)
            or not isinstance(gold.get("speaker_truth"), list)
        ):
            raise AcousticVerificationError("Calibration record evidence is invalid.")
        source_path = Path(str(source.get("stored_path") or ""))
        transcript_path = Path(str(lineage.get("current_artifact_path") or ""))
        require_private_file(source_path, source_path.parent)
        require_private_file(transcript_path, transcript_path.parent)
        if (
            sha256_file(source_path) != source.get("sha256")
            or source_path.stat().st_size != source.get("bytes")
            or sha256_file(transcript_path)
            != lineage.get("current_artifact_sha256")
        ):
            raise AcousticVerificationError("Calibration source evidence drifted.")
        validated.append(record)
    return validated


def reveal_calibration_split(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Reveal only the exact authorized calibration metadata and opaque gold."""
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256,
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    records = _calibration_records_after_authority(
        authority, parent_corpus_manifest_path=parent_corpus_manifest_path
    )
    public_records = []
    for record in records:
        source = record["source_blob"]
        lineage = record["transcript_lineage"]
        gold = record["operator_gold"]
        public_records.append(
            {
                "recording_id": record["recording_id"],
                "conversation_id": record["conversation_id"],
                "source_blob_id": source["blob_id"],
                "source_sha256": source["sha256"],
                "source_bytes": source["bytes"],
                "transcript_artifact_sha256": lineage[
                    "current_artifact_sha256"
                ],
                "gold_id": gold["gold_id"],
                "speaker_truth": [dict(item) for item in gold["speaker_truth"]],
                "conditions": dict(record.get("conditions") or {}),
            }
        )
    receipt = {
        "schema_version": CALIBRATION_SPLIT_REVEAL_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha256,
        "intended_split": "calibration",
        "record_set_sha256": authority["calibration_record_set_sha256"],
        "conversation_set_sha256": authority[
            "calibration_conversation_set_sha256"
        ],
        "record_count": len(public_records),
        "conversation_count": len(
            {record["conversation_id"] for record in public_records}
        ),
        "records": public_records,
        "development_disjoint": True,
        "evaluation_disjoint": True,
        "source_content_disjoint": True,
        "contains_opaque_gold_labels": True,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "will_read_evaluation": False,
        "will_perform_external_write": False,
        "revealed_at": utc_now(),
    }
    receipt_sha = canonical_artifact_hash(
        {key: value for key, value in receipt.items() if key != "revealed_at"}
    )
    path = root / "calibration-stages" / authority_sha256 / "split-reveal.json"
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(
        path, receipt, volatile_fields=("revealed_at",)
    )
    return {
        **stored,
        "split_reveal_sha256": receipt_sha,
        "private_split_reveal_path": str(path),
    }


def prepare_calibration_split(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Run exact P1/P2 preparation for all authorized calibration records."""
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256,
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    reveal = reveal_calibration_split(
        authority_sha256,
        runtime_root=root,
        p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    path = root / "calibration-stages" / authority_sha256 / "preparation.json"
    if path.exists():
        require_private_file(path, root)
        existing = read_private_object(path)
        identity = {
            key: value for key, value in existing.items() if key != "prepared_at"
        }
        existing_sha = canonical_artifact_hash(identity)
        if (
            existing.get("authority_sha256") != authority_sha256
            or existing.get("split_reveal_sha256")
            != reveal["split_reveal_sha256"]
            or existing.get("status") != "success"
        ):
            raise AcousticVerificationError(
                "Calibration preparation receipt conflicts."
            )
        return {
            **existing,
            "preparation_sha256": existing_sha,
            "private_preparation_path": str(path),
        }
    records = _calibration_records_after_authority(
        authority, parent_corpus_manifest_path=parent_corpus_manifest_path
    )
    p1_root = root / "calibration-preparation" / authority_sha256 / "p1"
    p2_root = root / "calibration-preparation" / authority_sha256 / "p2"
    units: list[dict[str, Any]] = []
    for record in records:
        source = record["source_blob"]
        source_path = Path(str(source["stored_path"]))
        source_sha = str(source["sha256"])
        source_blob_id = "source-" + source_sha[:24]
        p1_plan = audio_derivatives.dry_run(
            source_path,
            runtime_root=p1_root,
            source_blob_id=source_blob_id,
            expected_source_sha256=source_sha,
            channel_policy="stereo_average_to_mono",
            channel_policy_authority_sha256=authority_sha256,
        )
        p1_applied = audio_derivatives.apply_derivative(
            source_path,
            runtime_root=p1_root,
            approval_token=audio_derivatives.APPLY_TOKEN,
            source_blob_id=source_blob_id,
            expected_source_sha256=source_sha,
            channel_policy="stereo_average_to_mono",
            channel_policy_authority_sha256=authority_sha256,
        )
        p1_replay = audio_derivatives.replay_derivative(
            p1_plan["run_id"], runtime_root=p1_root
        )
        p2_plan = speech_preparation.dry_run(
            p1_plan["run_id"],
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            intended_split="calibration",
            split_access_authority_sha256=authority_sha256,
        )
        p2_applied = speech_preparation.apply_comparison(
            p1_plan["run_id"],
            p1_runtime_root=p1_root,
            runtime_root=p2_root,
            intended_split="calibration",
            split_access_authority_sha256=authority_sha256,
        )
        p2_replay = speech_preparation.replay_comparison(
            p2_plan["run_id"], runtime_root=p2_root
        )
        comparison = p2_applied["comparison"]
        methods = []
        for method in comparison.get("method_results") or []:
            if not isinstance(method, Mapping) or method.get("status") != "success":
                raise AcousticVerificationError(
                    "Calibration preparation method did not succeed."
                )
            output_path = Path(str(method.get("output_path") or ""))
            require_private_file(output_path, p2_root if output_path.is_relative_to(p2_root) else output_path.parent)
            if sha256_file(output_path) != method.get("output_sha256"):
                raise AcousticVerificationError(
                    "Calibration preparation output drifted."
                )
            methods.append(
                {
                    "method_id": method["method_id"],
                    "method_result_sha256": canonical_artifact_hash(dict(method)),
                    "output_path": str(output_path),
                    "output_sha256": method["output_sha256"],
                    "output_equivalence_class_sha256": method["output_sha256"],
                    "speech_region_count": len(method.get("speech_regions") or []),
                    "overlap_region_count": len(method.get("overlap_regions") or []),
                    "speaker_change_region_count": len(
                        method.get("speaker_change_regions") or []
                    ),
                }
            )
        units.append(
            {
                "recording_id": record["recording_id"],
                "conversation_id": record["conversation_id"],
                "source_sha256": source_sha,
                "p1_run_id": p1_plan["run_id"],
                "p1_manifest_sha256": p1_applied["manifest_sha256"],
                "p1_replay_receipt_sha256": sha256_file(
                    Path(str(p1_replay["replay_receipt_path"]))
                ),
                "p2_run_id": p2_plan["run_id"],
                "p2_comparison_path": p2_applied["comparison_path"],
                "p2_comparison_sha256": sha256_file(
                    Path(str(p2_applied["comparison_path"]))
                ),
                "p2_replay_receipt_sha256": sha256_file(
                    Path(str(p2_replay["replay_path"]))
                ),
                "methods": methods,
            }
        )
    receipt = {
        "schema_version": CALIBRATION_PREPARATION_SCHEMA,
        "status": "success",
        "reason_code": None,
        "authority_sha256": authority_sha256,
        "split_reveal_sha256": reveal["split_reveal_sha256"],
        "intended_split": "calibration",
        "record_count": len(units),
        "method_attempts": len(units) * len(METHOD_IDS),
        "method_successes": sum(len(unit["methods"]) for unit in units),
        "units": units,
        "did_read_calibration_audio": True,
        "did_run_p1_p2": True,
        "did_read_evaluation": False,
        "did_run_biometrics": False,
        "did_perform_external_write": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "prepared_at": utc_now(),
    }
    receipt_sha = canonical_artifact_hash(
        {key: value for key, value in receipt.items() if key != "prepared_at"}
    )
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(
        path, receipt, volatile_fields=("prepared_at",)
    )
    return {
        **stored,
        "preparation_sha256": receipt_sha,
        "private_preparation_path": str(path),
    }


def _calibration_stage_identity(value: Mapping[str, Any], volatile: str) -> str:
    return canonical_artifact_hash(
        {key: item for key, item in value.items() if key != volatile}
    )


def select_calibration_windows(
    authority_sha256: str,
    *,
    runtime_root: Path,
    p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Freeze opaque calibration windows before any biometric score exists."""
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    preparation = prepare_calibration_split(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    path = root / "calibration-stages" / authority_sha256 / "window-selection.json"
    if path.exists():
        require_private_file(path, root)
        existing = read_private_object(path)
        existing_sha = _calibration_stage_identity(existing, "selected_at")
        if (
            existing.get("schema_version") != CALIBRATION_WINDOW_SELECTION_SCHEMA
            or existing.get("status") != "success"
            or existing.get("authority_sha256") != authority_sha256
            or existing.get("preparation_sha256") != preparation["preparation_sha256"]
            or existing.get("did_run_biometrics") is not False
            or existing.get("did_read_evaluation") is not False
        ):
            raise AcousticVerificationError("Calibration window selection conflicts.")
        return {**existing, "window_selection_sha256": existing_sha,
                "private_window_selection_path": str(path)}
    records = _calibration_records_after_authority(
        authority, parent_corpus_manifest_path=parent_corpus_manifest_path
    )
    unit_by_recording = {
        str(unit["recording_id"]): unit for unit in preparation["units"]
    }
    windows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        recording_id = str(record["recording_id"])
        unit = unit_by_recording.get(recording_id)
        if not isinstance(unit, Mapping):
            raise AcousticVerificationError("Calibration preparation coverage changed.")
        comparison_path = Path(str(unit["p2_comparison_path"]))
        require_private_file(comparison_path, comparison_path.parent)
        if sha256_file(comparison_path) != unit["p2_comparison_sha256"]:
            raise AcousticVerificationError("Calibration comparison drifted.")
        comparison = read_private_object(comparison_path)
        pyannote = next(
            (item for item in comparison.get("method_results") or []
             if isinstance(item, Mapping)
             and item.get("method_id") == "pyannote_community_1"), None,
        )
        py_binding = next(
            (item for item in unit["methods"]
             if item.get("method_id") == "pyannote_community_1"), None,
        )
        if (
            not isinstance(pyannote, Mapping)
            or not isinstance(py_binding, Mapping)
            or pyannote.get("status") != "success"
            or canonical_artifact_hash(dict(pyannote))
            != py_binding.get("method_result_sha256")
        ):
            raise AcousticVerificationError("Calibration Pyannote binding changed.")
        transcript_path = Path(str(record["transcript_lineage"]["current_artifact_path"]))
        require_private_file(transcript_path, transcript_path.parent)
        if sha256_file(transcript_path) != record["transcript_lineage"]["current_artifact_sha256"]:
            raise AcousticVerificationError("Calibration transcript artifact drifted.")
        transcript = read_private_object(transcript_path)
        blocked = [
            *list(pyannote.get("overlap_regions") or []),
            *list(pyannote.get("speaker_change_regions") or []),
        ]
        p1_replay = audio_derivatives.replay_derivative(
            str(unit["p1_run_id"]),
            runtime_root=(
                root / "calibration-preparation" / authority_sha256 / "p1"
            ),
            include_validated_manifest=True,
        )
        source_channels = int(
            p1_replay["validated_manifest"]["source"]["probe"]["channels"]
        )
        if source_channels not in (1, 2):
            raise AcousticVerificationError(
                "Calibration source channel binding changed."
            )
        conditions = dict(record.get("conditions") or {})
        conditions["channel"] = f"source_{source_channels}_channel"
        conditions["overlap"] = "overlap_regions_excluded"
        for truth in record["operator_gold"]["speaker_truth"]:
            outcome = str(truth.get("outcome") or "")
            speaker_ref = str(truth.get("speaker_label") or "")
            subject_id = truth.get("subject_id")
            if outcome != "person" or not isinstance(subject_id, str):
                exclusions.append({
                    "recording_id": recording_id,
                    "conversation_id": record["conversation_id"],
                    "speaker_ref": speaker_ref,
                    "reason_code": "mixed_gold" if outcome == "mixed" else "unknown_gold",
                })
                continue
            spans = _candidate_windows(
                transcript.get("utterances"), speaker_label=speaker_ref,
                speech_regions=pyannote.get("speech_regions"),
                blocked_regions=blocked,
            )
            if not spans:
                exclusions.append({
                    "recording_id": recording_id,
                    "conversation_id": record["conversation_id"],
                    "speaker_ref": speaker_ref,
                    "reason_code": "no_speech_intersection",
                })
            for start, end in spans:
                identity = {
                    "authority_sha256": authority_sha256,
                    "recording_id": recording_id,
                    "speaker_ref": speaker_ref,
                    "subject_id": subject_id,
                    "start_seconds": start,
                    "end_seconds": end,
                }
                duration = end - start
                window_conditions = dict(conditions)
                window_conditions["usable_duration_band"] = (
                    "0.75_to_under_3_seconds" if duration < 3
                    else "3_to_under_8_seconds" if duration < 8
                    else "8_to_15_seconds"
                )
                windows.append({
                    "window_id": "calibration-window-"
                    + canonical_artifact_hash(identity)[:24],
                    "recording_id": recording_id,
                    "conversation_id": record["conversation_id"],
                    "speaker_ref": speaker_ref,
                    "subject_id": subject_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "conditions": window_conditions,
                    "source_sha256": record["source_blob"]["sha256"],
                    "transcript_artifact_sha256": record["transcript_lineage"]["current_artifact_sha256"],
                    "pyannote_method_result_sha256": py_binding["method_result_sha256"],
                })
    windows.sort(key=lambda item: item["window_id"])
    exclusions.sort(key=lambda item: (item["recording_id"], item["speaker_ref"]))
    if not windows or len({item["window_id"] for item in windows}) != len(windows):
        raise AcousticVerificationError("Calibration window coverage is invalid.")
    receipt = {
        "schema_version": CALIBRATION_WINDOW_SELECTION_SCHEMA,
        "status": "success", "reason_code": None,
        "authority_sha256": authority_sha256,
        "split_reveal_sha256": preparation["split_reveal_sha256"],
        "preparation_sha256": preparation["preparation_sha256"],
        "intended_split": "calibration",
        "selection_method": "operator_gold_intersect_pyannote_speech_minus_overlap_and_change",
        "maximum_windows_per_speaker_per_conversation": 3,
        "window_count": len(windows),
        "included_speaker_count": len({(w["recording_id"], w["speaker_ref"]) for w in windows}),
        "excluded_speaker_count": len(exclusions),
        "windows": windows, "exclusions": exclusions,
        "did_read_calibration_gold": True, "did_run_biometrics": False,
        "did_read_evaluation": False, "did_perform_external_write": False,
        "contains_opaque_gold_labels": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_embeddings_or_vectors": False,
        "selected_at": utc_now(),
    }
    if _contains_forbidden_private_key(receipt):
        raise AcousticVerificationError("Calibration selection contains forbidden data.")
    receipt_sha = _calibration_stage_identity(receipt, "selected_at")
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("selected_at",))
    return {**stored, "window_selection_sha256": receipt_sha,
            "private_window_selection_path": str(path)}


def _calibration_pcm_window(
    preparation: Mapping[str, Any], window: Mapping[str, Any], method_id: str,
) -> tuple[float, ...]:
    unit = next((item for item in preparation["units"]
                 if item["recording_id"] == window["recording_id"]), None)
    if not isinstance(unit, Mapping):
        raise AcousticVerificationError("Calibration PCM unit is unavailable.")
    comparison_path = Path(str(unit["p2_comparison_path"]))
    require_private_file(comparison_path, comparison_path.parent)
    if sha256_file(comparison_path) != unit["p2_comparison_sha256"]:
        raise AcousticVerificationError("Calibration comparison drifted.")
    comparison = read_private_object(comparison_path)
    method = next((item for item in comparison.get("method_results") or []
                   if isinstance(item, Mapping) and item.get("method_id") == method_id), None)
    binding = next((item for item in unit["methods"] if item["method_id"] == method_id), None)
    if (
        not isinstance(method, Mapping) or not isinstance(binding, Mapping)
        or method.get("status") != "success"
        or canonical_artifact_hash(dict(method)) != binding.get("method_result_sha256")
    ):
        raise AcousticVerificationError("Calibration score-method binding changed.")
    pcm_path = Path(str(method["output_path"]))
    require_private_file(pcm_path, pcm_path.parent)
    if sha256_file(pcm_path) != method["output_sha256"]:
        raise AcousticVerificationError("Calibration PCM drifted.")
    try:
        with wave.open(str(pcm_path), "rb") as reader:
            if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, 16_000, "NONE"):
                raise AcousticVerificationError("Calibration PCM contract is invalid.")
            start = round(float(window["start_seconds"]) * 16_000)
            end = round(float(window["end_seconds"]) * 16_000)
            if start < 0 or end <= start or end > reader.getnframes():
                raise AcousticVerificationError("Calibration window is outside PCM.")
            reader.setpos(start)
            payload = reader.readframes(end - start)
    except (EOFError, OSError, wave.Error, TypeError, ValueError) as exc:
        raise AcousticVerificationError("Calibration PCM is unreadable.") from exc
    if len(payload) != (end - start) * 2:
        raise AcousticVerificationError("Calibration PCM window is truncated.")
    return tuple(value / 32768.0 for value in struct.unpack(f"<{end-start}h", payload))


def apply_calibration_scores(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    adapters: Optional[Mapping[str, VerificationAdapter]] = None,
    test_mode: bool = False,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Run the frozen calibration score matrix without selecting thresholds."""
    if adapters is not None and not test_mode:
        raise AcousticVerificationError("Custom calibration adapters are test-only.")
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    preparation = prepare_calibration_split(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    selection = select_calibration_windows(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    path = root / "calibration-stages" / authority_sha256 / "score-matrix.json"
    if path.exists():
        require_private_file(path, root)
        existing = read_private_object(path)
        existing_sha = _calibration_stage_identity(existing, "scored_at")
        if (existing.get("status") != "success" or existing.get("authority_sha256") != authority_sha256
                or existing.get("window_selection_sha256") != selection["window_selection_sha256"]):
            raise AcousticVerificationError("Calibration score matrix conflicts.")
        return {**existing, "score_matrix_sha256": existing_sha,
                "private_score_matrix_path": str(path)}
    selected_adapters = dict(adapters or adapter_registry())
    expected_models = {str(p["candidate_id"]): str(p["model_revision"])
                       for p in authority["profiles"]}
    if set(selected_adapters) != set(expected_models) or any(
        selected_adapters[key].revision_sha != revision
        for key, revision in expected_models.items()
    ):
        raise AcousticVerificationError("Calibration model inventory drifted.")
    profiles_by_model = {candidate_id: [dict(p) for p in authority["profiles"]
                                        if p["candidate_id"] == candidate_id]
                         for candidate_id in sorted(expected_models)}
    known_profile_subjects = {str(p["person_ref_id"]) for p in authority["profiles"]}
    trials: list[dict[str, Any]] = []
    for window in selection["windows"]:
        for method_id in authority["score_methods"]:
            samples = _calibration_pcm_window(preparation, window, method_id)
            for candidate_id in sorted(expected_models):
                adapter = selected_adapters[candidate_id]
                for profile in profiles_by_model[candidate_id]:
                    scored = score_profile(
                        str(profile["profile_id"]), adapter=adapter,
                        probe_samples=samples, sample_rate=16_000,
                        runtime_root=root, p3_runtime_root=p3_runtime_root,
                    )
                    expected_match = window["subject_id"] == profile["person_ref_id"]
                    identity = {"authority_sha256": authority_sha256,
                                "window_id": window["window_id"], "method_id": method_id,
                                "profile_id": profile["profile_id"],
                                "score_trial_id": scored["trial_id"]}
                    trials.append({
                        "trial_id": "calibration-trial-" + canonical_artifact_hash(identity)[:24],
                        "status": "success", "reason_code": None,
                        "window_id": window["window_id"],
                        "recording_id": window["recording_id"],
                        "conversation_id": window["conversation_id"],
                        "probe_subject_id": window["subject_id"],
                        "profile_person_ref_id": profile["person_ref_id"],
                        "expected_match": expected_match,
                        "open_set_probe": window["subject_id"] not in known_profile_subjects,
                        "method_id": method_id, "profile_id": profile["profile_id"],
                        "descendant_id": profile["descendant_id"],
                        "candidate_id": candidate_id, "model_revision": adapter.revision_sha,
                        "probe_sha256": scored["probe_sha256"],
                        "score_trial_id": scored["trial_id"], "score": scored["score"],
                        "conditions": dict(window["conditions"]),
                        "p4_state_verified_before_and_after": True,
                        "p3_eligibility_verified_before_and_after": True,
                        "contains_raw_biometric_values": False,
                    })
    trials.sort(key=lambda item: item["trial_id"])
    expected_count = len(selection["windows"]) * len(authority["score_methods"]) * 6
    if (len(trials) != expected_count
            or any(not math.isfinite(float(t["score"])) or not -1 <= float(t["score"]) <= 1 for t in trials)
            or len({t["trial_id"] for t in trials}) != len(trials)):
        raise AcousticVerificationError("Calibration score coverage is invalid.")
    receipt = {
        "schema_version": CALIBRATION_SCORE_MATRIX_SCHEMA,
        "status": "success", "reason_code": None,
        "authority_sha256": authority_sha256,
        "preparation_sha256": preparation["preparation_sha256"],
        "window_selection_sha256": selection["window_selection_sha256"],
        "intended_split": "calibration", "logical_trial_count": len(trials),
        "genuine_trial_count": sum(t["expected_match"] for t in trials),
        "impostor_trial_count": sum(not t["expected_match"] for t in trials),
        "open_set_trial_count": sum(t["open_set_probe"] for t in trials),
        "trials": trials, "did_run_biometrics": True,
        "did_select_threshold": False, "did_read_evaluation": False,
        "did_perform_external_write": False, "contains_biometric_scores": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False, "contains_raw_biometric_values": False,
        "scored_at": utc_now(),
    }
    receipt_sha = _calibration_stage_identity(receipt, "scored_at")
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("scored_at",))
    return {**stored, "score_matrix_sha256": receipt_sha,
            "private_score_matrix_path": str(path)}


def replay_calibration_score_matrix(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Structurally replay persisted calibration scores without model execution."""
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    selection = select_calibration_windows(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    path = root / "calibration-stages" / authority_sha256 / "score-matrix.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    receipt_sha = _calibration_stage_identity(receipt, "scored_at")
    profiles = {str(item["profile_id"]): dict(item) for item in authority["profiles"]}
    windows = {str(item["window_id"]): dict(item) for item in selection["windows"]}
    for profile in profiles.values():
        current = replay_profile(str(profile["profile_id"]), runtime_root=root)
        if ({key: current.get(key) for key in profile} != profile
                or not descendant_is_eligible(
                    str(profile["descendant_id"]), runtime_root=p3_runtime_root
                )):
            raise AcousticVerificationError(
                "Calibration profile eligibility changed."
            )
    expected = {
        (window_id, method_id, profile_id)
        for window_id in windows
        for method_id in authority["score_methods"]
        for profile_id in profiles
    }
    actual: set[tuple[str, str, str]] = set()
    for trial in receipt.get("trials") or []:
        if not isinstance(trial, Mapping):
            raise AcousticVerificationError("Calibration trial is invalid.")
        window = windows.get(str(trial.get("window_id") or ""))
        profile = profiles.get(str(trial.get("profile_id") or ""))
        score = trial.get("score")
        if (
            window is None or profile is None
            or trial.get("status") != "success"
            or trial.get("candidate_id") != profile["candidate_id"]
            or trial.get("model_revision") != profile["model_revision"]
            or trial.get("descendant_id") != profile["descendant_id"]
            or trial.get("probe_subject_id") != window["subject_id"]
            or trial.get("profile_person_ref_id") != profile["person_ref_id"]
            or trial.get("expected_match")
            is not (window["subject_id"] == profile["person_ref_id"])
            or trial.get("open_set_probe")
            is not (window["subject_id"] not in {p["person_ref_id"] for p in profiles.values()})
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score)) or not -1 <= float(score) <= 1
            or trial.get("conditions") != window["conditions"]
            or trial.get("p4_state_verified_before_and_after") is not True
            or trial.get("p3_eligibility_verified_before_and_after") is not True
        ):
            raise AcousticVerificationError("Calibration trial binding changed.")
        score_identity = {
            "profile_id": profile["profile_id"],
            "descendant_id": profile["descendant_id"],
            "artifact_sha256": profile["artifact_sha256"],
            "candidate_id": profile["candidate_id"],
            "model_revision": profile["model_revision"],
            "probe_sha256": trial.get("probe_sha256"),
            "score": score,
        }
        if trial.get("score_trial_id") != (
            "verification-trial-" + canonical_artifact_hash(score_identity)[:24]
        ):
            raise AcousticVerificationError(
                "Calibration score-trial identity changed."
            )
        identity = {"authority_sha256": authority_sha256,
                    "window_id": window["window_id"], "method_id": trial["method_id"],
                    "profile_id": profile["profile_id"],
                    "score_trial_id": trial["score_trial_id"]}
        if trial.get("trial_id") != "calibration-trial-" + canonical_artifact_hash(identity)[:24]:
            raise AcousticVerificationError("Calibration trial identity changed.")
        actual.add((str(trial["window_id"]), str(trial["method_id"]), str(trial["profile_id"])))
    trials = receipt.get("trials")
    probe_groups: dict[tuple[str, str], set[str]] = {}
    for trial in trials or []:
        probe_groups.setdefault(
            (str(trial["window_id"]), str(trial["method_id"])), set()
        ).add(str(trial["probe_sha256"]))
    if (
        receipt.get("schema_version") != CALIBRATION_SCORE_MATRIX_SCHEMA
        or receipt.get("status") != "success"
        or receipt.get("authority_sha256") != authority_sha256
        or receipt.get("window_selection_sha256") != selection["window_selection_sha256"]
        or actual != expected or not isinstance(trials, list) or len(actual) != len(trials)
        or receipt.get("logical_trial_count") != len(trials)
        or receipt.get("genuine_trial_count") != sum(t["expected_match"] for t in trials)
        or receipt.get("impostor_trial_count") != sum(not t["expected_match"] for t in trials)
        or receipt.get("open_set_trial_count") != sum(t["open_set_probe"] for t in trials)
        or any(len(hashes) != 1 for hashes in probe_groups.values())
        or receipt.get("did_select_threshold") is not False
        or receipt.get("did_read_evaluation") is not False
        or receipt.get("contains_biometric_scores") is not True
    ):
        raise AcousticVerificationError("Calibration score matrix replay is invalid.")
    return {**receipt, "score_matrix_sha256": receipt_sha,
            "private_score_matrix_path": str(path),
            "score_replay_mode": "structural_without_audio_or_model_execution"}


def _sigmoid_probability(score: float, threshold: float, temperature: float) -> float:
    value = max(-60.0, min(60.0, (score - threshold) / temperature))
    return 1.0 / (1.0 + math.exp(-value))


def _classification_metrics(
    trials: Sequence[Mapping[str, Any]], *, threshold: float, temperature: float,
) -> dict[str, Any]:
    genuine = [item for item in trials if item["expected_match"] is True]
    impostor = [item for item in trials if item["expected_match"] is False]
    false_rejects = sum(float(item["score"]) < threshold for item in genuine)
    false_accepts = sum(float(item["score"]) >= threshold for item in impostor)
    far = false_accepts / len(impostor) if impostor else None
    frr = false_rejects / len(genuine) if genuine else None
    ber = (far + frr) / 2 if far is not None and frr is not None else None
    probabilities = [
        _sigmoid_probability(float(item["score"]), threshold, temperature)
        for item in trials
    ]
    labels = [1.0 if item["expected_match"] else 0.0 for item in trials]
    brier = (
        sum((probability - label) ** 2 for probability, label in zip(probabilities, labels))
        / len(trials) if trials else None
    )
    ece = None
    if trials:
        total = len(trials)
        ece_value = 0.0
        for bin_index in range(5):
            lower, upper = bin_index / 5, (bin_index + 1) / 5
            indexes = [index for index, probability in enumerate(probabilities)
                       if lower <= probability < upper or (bin_index == 4 and probability == 1.0)]
            if indexes:
                mean_probability = sum(probabilities[index] for index in indexes) / len(indexes)
                positive_rate = sum(labels[index] for index in indexes) / len(indexes)
                ece_value += len(indexes) / total * abs(mean_probability - positive_rate)
        ece = ece_value
    return {
        "trial_count": len(trials), "genuine_trial_count": len(genuine),
        "impostor_trial_count": len(impostor), "false_accept_count": false_accepts,
        "false_reject_count": false_rejects, "false_acceptance_rate": far,
        "false_rejection_rate": frr, "balanced_error_rate": ber,
        "brier_score": brier, "expected_calibration_error_5_bins": ece,
        "missing_denominator_status": (
            "not_run" if far is None or frr is None else "success"
        ),
    }


def _freeze_threshold_unit(
    candidate_id: str, method_id: str, trials: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    genuine_count = sum(item["expected_match"] is True for item in trials)
    impostor_count = sum(item["expected_match"] is False for item in trials)
    if (genuine_count < int(policy["minimum_genuine_trials_per_unit"])
            or impostor_count < int(policy["minimum_impostor_trials_per_unit"])):
        return {
            "candidate_id": candidate_id, "method_id": method_id,
            "status": "not_run", "reason_code": "insufficient_class_denominator",
            "threshold": None, "temperature": None, "metrics": None,
            "condition_slices": [], "candidate_margin": None,
            "open_set_rejection": None,
        }
    scores = sorted({float(item["score"]) for item in trials})
    thresholds = sorted({-1.0, 1.0, *[(left + right) / 2
                                     for left, right in zip(scores, scores[1:])]})
    candidates = []
    for threshold in thresholds:
        for temperature in policy["temperature_candidates"]:
            metrics = _classification_metrics(
                trials, threshold=threshold, temperature=float(temperature)
            )
            candidates.append((
                (metrics["brier_score"], metrics["expected_calibration_error_5_bins"],
                 metrics["balanced_error_rate"],
                 abs(metrics["false_acceptance_rate"] - metrics["false_rejection_rate"]),
                 -threshold, float(temperature)),
                threshold, float(temperature), metrics,
            ))
    _, threshold, temperature, metrics = min(candidates, key=lambda item: item[0])
    eer_threshold, eer_metrics = min(
        ((candidate_threshold,
          _classification_metrics(trials, threshold=candidate_threshold, temperature=temperature))
         for candidate_threshold in thresholds),
        key=lambda item: (
            abs(item[1]["false_acceptance_rate"] - item[1]["false_rejection_rate"]),
            item[1]["balanced_error_rate"], -item[0],
        ),
    )
    slices = []
    for dimension in policy.get("condition_slices", []):
        values = sorted({str(item.get("conditions", {}).get(dimension, "missing")) for item in trials})
        for value in values:
            subset = [item for item in trials
                      if str(item.get("conditions", {}).get(dimension, "missing")) == value]
            slices.append({"dimension": dimension, "value": value,
                           "metrics": _classification_metrics(
                               subset, threshold=threshold, temperature=temperature)})
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in trials:
        grouped.setdefault((str(item["window_id"]), str(item["conversation_id"])), []).append(item)
    margins = []
    open_set_groups = []
    for items in grouped.values():
        ordered = sorted((float(item["score"]) for item in items), reverse=True)
        if len(ordered) != 2:
            raise AcousticVerificationError("Calibration candidate margin coverage changed.")
        margins.append(ordered[0] - ordered[1])
        if items[0]["open_set_probe"] is True:
            open_set_groups.append(all(float(item["score"]) < threshold for item in items))
    return {
        "candidate_id": candidate_id, "method_id": method_id,
        "status": "success", "reason_code": None,
        "threshold": threshold, "temperature": temperature,
        "threshold_candidate_count": len(thresholds), "metrics": metrics,
        "eer_diagnostic": {"threshold": eer_threshold,
                           "false_acceptance_rate": eer_metrics["false_acceptance_rate"],
                           "false_rejection_rate": eer_metrics["false_rejection_rate"],
                           "estimated_equal_error_rate": (eer_metrics["false_acceptance_rate"] + eer_metrics["false_rejection_rate"]) / 2},
        "condition_slices": slices,
        "candidate_margin": {"status": "descriptive", "count": len(margins),
                             "minimum": min(margins), "mean": sum(margins) / len(margins),
                             "maximum": max(margins)},
        "open_set_rejection": {
            "status": "descriptive", "probe_count": len(open_set_groups),
            "rejected_count": sum(open_set_groups),
            "rejection_rate": sum(open_set_groups) / len(open_set_groups) if open_set_groups else None,
        },
    }


def _calibration_threshold_results(
    authority: Mapping[str, Any], score_matrix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    policy = {**dict(authority["threshold_policy"]),
              "condition_slices": authority["metric_policy"]["condition_slices"]}
    results = []
    for candidate_id in sorted({str(item["candidate_id"]) for item in authority["profiles"]}):
        for method_id in authority["score_methods"]:
            trials = [item for item in score_matrix["trials"]
                      if item["candidate_id"] == candidate_id and item["method_id"] == method_id]
            results.append(_freeze_threshold_unit(candidate_id, method_id, trials, policy))
    return results


def apply_calibration_thresholds(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Deterministically freeze calibration-only thresholds and diagnostics."""
    root = runtime_root.expanduser().absolute()
    authority = replay_calibration_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    scores = replay_calibration_score_matrix(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    results = _calibration_threshold_results(authority, scores)
    if len(results) != 9 or any(item["status"] != "success" for item in results):
        raise AcousticVerificationError("Calibration threshold coverage is incomplete.")
    receipt = {
        "schema_version": CALIBRATION_APPLICATION_SCHEMA,
        "status": "success", "reason_code": None,
        "authority_sha256": authority_sha256,
        "score_matrix_sha256": scores["score_matrix_sha256"],
        "intended_split": "calibration", "threshold_unit_count": len(results),
        "thresholds": results,
        "selection_objective": list(authority["threshold_policy"]["selection_order"]),
        "abstention_status": "not_run_without_separately_precommitted_abstention_margin",
        "results_are_descriptive": True,
        "conversation_clustered_non_independent": True,
        "permits_generalization_claim": False,
        "did_read_calibration_gold": True, "did_run_calibration_trials": True,
        "did_select_and_freeze_thresholds": True, "did_read_evaluation": False,
        "did_mutate_profiles_or_references": False,
        "did_make_terminal_model_or_method_selection": False,
        "did_enable_default_integration": False, "did_perform_external_write": False,
        "contains_biometric_scores": True, "contains_frozen_thresholds": True,
        "contains_raw_audio": False, "contains_transcript_text": False,
        "contains_names_or_emails": False, "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False, "applied_at": utc_now(),
    }
    receipt_sha = _calibration_stage_identity(receipt, "applied_at")
    path = root / "calibration-applications" / f"{receipt_sha}.json"
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("applied_at",))
    return {**stored, "application_sha256": receipt_sha,
            "private_application_path": str(path)}


def replay_calibration_thresholds(
    application_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Recompute thresholds and metrics from persisted scores, never audio."""
    if not SHA256_RE.fullmatch(str(application_sha256)):
        raise AcousticVerificationError("Calibration application hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "calibration-applications" / f"{application_sha256}.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    if _calibration_stage_identity(receipt, "applied_at") != application_sha256:
        raise AcousticVerificationError("Calibration application identity changed.")
    authority_sha256 = str(receipt.get("authority_sha256") or "")
    authority = replay_calibration_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    scores = replay_calibration_score_matrix(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    recomputed = _calibration_threshold_results(authority, scores)
    if (
        receipt.get("schema_version") != CALIBRATION_APPLICATION_SCHEMA
        or receipt.get("status") != "success"
        or receipt.get("score_matrix_sha256") != scores["score_matrix_sha256"]
        or receipt.get("thresholds") != recomputed
        or receipt.get("threshold_unit_count") != 9
        or receipt.get("did_read_evaluation") is not False
        or receipt.get("did_mutate_profiles_or_references") is not False
        or receipt.get("did_make_terminal_model_or_method_selection") is not False
        or receipt.get("contains_biometric_scores") is not True
        or receipt.get("contains_frozen_thresholds") is not True
    ):
        raise AcousticVerificationError("Calibration threshold replay is invalid.")
    return {**receipt, "application_sha256": application_sha256,
            "private_application_path": str(path),
            "threshold_replay_mode": "recomputed_from_persisted_scores_without_audio"}


def _evaluation_split_metadata_authority(
    split_policy_path: Path, parent_corpus_manifest_path: Path,
) -> dict[str, Any]:
    """Validate evaluation scope metadata without revealing evaluation rows."""
    policy_path = split_policy_path.expanduser().absolute()
    parent_path = parent_corpus_manifest_path.expanduser().absolute()
    if policy_path.is_symlink() or not policy_path.is_file():
        raise AcousticVerificationError("Split access policy is unavailable.")
    if sha256_file(policy_path) != EXPECTED_SPLIT_ACCESS_POLICY_SHA256:
        raise AcousticVerificationError("Split access policy hash is invalid.")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError("Split access policy is unreadable.") from exc
    require_private_file(parent_path, parent_path.parent)
    if sha256_file(parent_path) != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256:
        raise AcousticVerificationError("Parent corpus manifest hash is invalid.")
    evaluation = policy.get("splits", {}).get("evaluation")
    if (
        policy.get("schema_version")
        != "transcribe-audio.verification-split-access-policy.v1"
        or policy.get("parent_corpus_manifest_sha256")
        != EXPECTED_PARENT_CORPUS_MANIFEST_SHA256
        or not isinstance(evaluation, Mapping)
        or evaluation.get("authorization_state")
        != "not_authorized_pending_exact_terminal_evaluation_apply"
        or evaluation.get("recording_count") != 5
        or evaluation.get("conversation_count") != 5
        or evaluation.get("record_set_sha256")
        != EXPECTED_EVALUATION_RECORD_SET_SHA256
        or evaluation.get("conversation_set_sha256")
        != EXPECTED_EVALUATION_CONVERSATION_SET_SHA256
    ):
        raise AcousticVerificationError("Evaluation split metadata is invalid.")
    return {
        "split_access_policy_sha256": EXPECTED_SPLIT_ACCESS_POLICY_SHA256,
        "parent_corpus_manifest_sha256": EXPECTED_PARENT_CORPUS_MANIFEST_SHA256,
        "evaluation_record_set_sha256": EXPECTED_EVALUATION_RECORD_SET_SHA256,
        "evaluation_conversation_set_sha256": EXPECTED_EVALUATION_CONVERSATION_SET_SHA256,
        "evaluation_recording_count": 5,
        "evaluation_conversation_count": 5,
    }


def _terminal_decision_policy(path: Path) -> dict[str, Any]:
    policy_path = path.expanduser().absolute()
    if policy_path.is_symlink() or not policy_path.is_file():
        raise AcousticVerificationError("Terminal decision policy is unavailable.")
    if sha256_file(policy_path) != EXPECTED_TERMINAL_DECISION_POLICY_SHA256:
        raise AcousticVerificationError("Terminal decision policy hash is invalid.")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcousticVerificationError("Terminal decision policy is unreadable.") from exc
    if (
        policy.get("schema_version")
        != "transcribe-audio.verification-terminal-decision-policy.v1"
        or policy.get("precedence") != ["stop", "reject", "select", "refine"]
        or policy.get("policy_changes_after_evaluation_unseal")
        != "forbidden_for_this_evaluation_generation"
    ):
        raise AcousticVerificationError("Terminal decision policy is invalid.")
    return policy


def _evaluation_apply_authority_payload(
    *, calibration_application: Mapping[str, Any],
    calibration_application_sha256: str,
    calibration_authority: Mapping[str, Any],
    split_metadata: Mapping[str, Any], terminal_policy: Mapping[str, Any],
    authorized_at: str,
) -> dict[str, Any]:
    if (
        calibration_application.get("status") != "success"
        or calibration_application.get("intended_split") != "calibration"
        or calibration_application.get("did_select_and_freeze_thresholds") is not True
        or calibration_application.get("did_read_evaluation") is not False
        or calibration_application.get("threshold_unit_count") != 9
        or calibration_application.get("permits_generalization_claim") is not False
    ):
        raise AcousticVerificationError(
            "Evaluation authority requires verified calibration evidence."
        )
    frozen_thresholds = [
        {
            "candidate_id": item["candidate_id"],
            "method_id": item["method_id"],
            "threshold": item["threshold"],
            "temperature": item["temperature"],
            "calibration_status": item["status"],
        }
        for item in calibration_application["thresholds"]
    ]
    p1_path = Path(audio_derivatives.__file__).resolve()
    p2_path = Path(speech_preparation.__file__).resolve()
    authority = {
        "schema_version": EVALUATION_APPLY_AUTHORITY_SCHEMA,
        "status": "authorized", "reason_code": None,
        "authority_generation": 1,
        "canonicalization": "json_sort_keys_compact_utf8",
        "authorization_basis": REAL_ENROLLMENT_AUTHORIZATION_BASIS,
        "authorized_by_ref_id": REAL_ENROLLMENT_AUTHORIZER_REF_ID,
        "authorized_at": authorized_at, "intended_split": "evaluation",
        "calibration_application_sha256": calibration_application_sha256,
        "calibration_authority_sha256": calibration_application["authority_sha256"],
        "calibration_score_matrix_sha256": calibration_application["score_matrix_sha256"],
        "development_application_sha256": calibration_authority["development_application_sha256"],
        "development_authority_sha256": calibration_authority["development_authority_sha256"],
        "enrollment_application_sha256": calibration_authority["enrollment_application_sha256"],
        **dict(split_metadata),
        "terminal_decision_policy_sha256": EXPECTED_TERMINAL_DECISION_POLICY_SHA256,
        "terminal_decision_policy": dict(terminal_policy),
        "profiles": [dict(item) for item in calibration_authority["profiles"]],
        "preparation_methods": list(calibration_authority["preparation_methods"]),
        "score_methods": list(calibration_authority["score_methods"]),
        "frozen_thresholds": frozen_thresholds,
        "fixed_abstention_margins": [
            {"candidate_id": item["candidate_id"], "method_id": item["method_id"],
             "margin": 0.0, "derivation": "fixed_before_evaluation_not_data_selected"}
            for item in frozen_thresholds
        ],
        "preparation_contract": {
            "p1_module_sha256": sha256_file(p1_path),
            "p2_module_sha256": sha256_file(p2_path),
            "p2_open_acquisition_manifest_sha256": EXPECTED_P2_OPEN_ACQUISITION_MANIFEST_SHA256,
            "p2_pyannote_acquisition_manifest_sha256": EXPECTED_P2_PYANNOTE_ACQUISITION_MANIFEST_SHA256,
            "pcm_channels": 1, "pcm_sample_rate_hz": 16_000,
            "pcm_sample_width_bytes": 2, "pcm_compression": "NONE",
            "channel_policy": {
                **dict(calibration_authority["preparation_contract"]["channel_policy"]),
                "authority_binding": "terminal_evaluation_authority_sha256",
            },
            "no_fallback_method": True,
        },
        "window_policy": dict(calibration_authority["window_policy"]),
        "score_aggregation_policy": {
            "trial_score": "raw_cosine_against_fixed_enrollment_centroid",
            "window_assignment": (
                "highest_profile_score_if_at_or_above_frozen_threshold_and_"
                "uniquely_top_with_fixed_zero_margin_otherwise_abstain"
            ),
            "known_person_abstention_denominator": "known_person_probe_windows_only",
            "label_resolution": (
                "at_least_one_nonabstained_window_and_all_nonabstained_windows_"
                "resolve_to_same_profile_person_otherwise_unresolved"
            ),
            "ties_abstain_before_tie_break": True,
            "same_timestamp_bounds_across_score_methods": True,
            "no_score_or_threshold_normalization_change": True,
        },
        "label_grouping_policy": {
            "unit": "unordered_distinct_speaker_label_pair_within_conversation",
            "eligible_gold": "person_outcome_with_opaque_subject_id_only",
            "true_pair": "unordered_pair_expanded_from_operator_gold_same_person_label_groups",
            "predicted_pair": "both_labels_accept_the_same_profile_person_ref_id",
            "mixed_or_unknown_gold": "excluded_before_scoring",
            "precision": "true_predicted_pairs_divided_by_predicted_pairs",
            "recall": "true_predicted_pairs_divided_by_true_pairs",
            "missing_denominator": "status_not_run_and_numeric_value_null",
            "missing_precision_or_recall_denominator": "global_minimum_evidence_stop",
        },
        "evaluation_metric_policy": {
            "trial_metrics": dict(calibration_authority["metric_policy"]),
            "thresholds_and_temperatures_are_frozen": True,
            "abstention_rate": "abstained_known_person_windows_divided_by_known_person_windows",
            "label_unresolved_rate": "unresolved_eligible_labels_divided_by_eligible_labels",
            "candidate_recall": (
                "known_person_windows_with_correct_profile_top_scoring_divided_by_"
                "known_person_windows"
            ),
            "candidate_margin": "top_profile_score_minus_second_highest_profile_score",
            "attempt_accounting": "attempted_success_failed_blocked_reported_separately",
            "eer_diagnostic": (
                "frozen_candidate_threshold_and_tie_method_diagnostic_only_"
                "cannot_change_operational_threshold"
            ),
            "enhancement_safety_metrics": [
                "false_acceptance_rate", "open_set_rejection_rate",
                "label_group_precision",
            ],
            "enhancement_has_no_safety_metric_regression": (
                "far_not_higher_and_open_set_rejection_and_label_group_precision_"
                "not_lower_than_same_model_no_enhancement_with_zero_tolerance"
            ),
            "byte_identical_method_outputs": (
                "identical_metrics_and_one_acoustic_evidence_equivalence_class"
            ),
            "all_declared_condition_slices_reported": True,
            "conversation_clustered_non_independent": True,
        },
        "minimum_evidence_policy": {
            **dict(terminal_policy["minimum_evidence"]),
            "applies_per_model_method_unit": True,
            "missing_aggregate_decision_denominator": "global_stop",
            "missing_slice_denominator": "null_not_run",
            "incomplete_cartesian_or_failed_or_blocked_cell": "global_stop",
            "nonfinite_score_or_required_metric": "global_stop",
        },
        "terminal_resolution_policy": {
            "unit_precedence": ["stop", "reject", "select", "refine"],
            "global_integrity_or_minimum_evidence_failure": "stop",
            "any_terminal_policy_stop_if_condition_or_any_unit_stop": "global_stop_before_candidate_reduction",
            "global_candidate_resolution": (
                "select_best_selectable_else_refine_best_refinable_else_reject_when_all_units_reject"
            ),
            "winner_tie_break": [
                "lower_false_acceptance_rate", "lower_abstention_rate",
                "lower_false_rejection_rate", "simpler_runtime",
                "lexicographic_candidate_id",
            ],
            "simpler_runtime_order": ["no_enhancement", "rnnoise", "deepfilternet"],
            "simpler_model_runtime_order": [
                "wespeaker_resnet34", "wespeaker_campplus",
                "speechbrain_ecapa_tdnn",
            ],
            "runtime_cross_product_order": "method_rank_then_model_rank",
            "evaluation_may_not_change_policy_or_threshold": True,
        },
        "will_prepare_evaluation_audio": True, "will_read_evaluation_gold": True,
        "will_run_evaluation_trials": True, "will_make_terminal_decision": True,
        "will_select_or_change_thresholds": False,
        "will_change_temperatures_features_or_window_rules": False,
        "will_mutate_profiles_or_references": False,
        "will_enable_default_integration": False,
        "will_automatically_confirm_identity": False,
        "will_run_historical_reprocessing": False,
        "will_create_or_send_prompts": False,
        "will_change_terminal_policy_after_reveal": False,
        "will_perform_external_write": False,
        "contains_biometric_scores": False, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
    }
    if _contains_forbidden_private_key(authority):
        raise AcousticVerificationError("Evaluation authority contains forbidden data.")
    return authority


def build_evaluation_apply_authority(
    calibration_application_sha256: str, *, runtime_root: Path,
    p3_runtime_root: Path, split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
    terminal_policy_path: Path = DEFAULT_TERMINAL_DECISION_POLICY,
) -> dict[str, Any]:
    """Create the exact terminal-evaluation authority before split reveal."""
    root = runtime_root.expanduser().absolute()
    calibration = replay_calibration_thresholds(
        calibration_application_sha256, runtime_root=root,
        p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    calibration_authority = replay_calibration_apply_authority(
        str(calibration["authority_sha256"]), runtime_root=root,
        p3_runtime_root=p3_runtime_root,
        split_policy_path=split_policy_path,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    split_metadata = _evaluation_split_metadata_authority(
        split_policy_path, parent_corpus_manifest_path
    )
    terminal_policy = _terminal_decision_policy(terminal_policy_path)
    directory = root / "evaluation-authorities"
    ensure_private_tree(root, directory)
    matches = []
    for path in sorted(directory.glob("*.json")):
        require_private_file(path, root)
        value = read_private_object(path)
        if value.get("calibration_application_sha256") != calibration_application_sha256:
            continue
        expected = _evaluation_apply_authority_payload(
            calibration_application=calibration,
            calibration_application_sha256=calibration_application_sha256,
            calibration_authority=calibration_authority,
            split_metadata=split_metadata, terminal_policy=terminal_policy,
            authorized_at=str(value.get("authorized_at") or ""),
        )
        if value != expected or canonical_artifact_hash(value) != path.stem:
            raise AcousticVerificationError("Evaluation authority is invalid.")
        matches.append((path, value))
    if len(matches) > 1:
        raise AcousticVerificationError("Multiple evaluation authorities exist.")
    if matches:
        path, authority = matches[0]
    else:
        authority = _evaluation_apply_authority_payload(
            calibration_application=calibration,
            calibration_application_sha256=calibration_application_sha256,
            calibration_authority=calibration_authority,
            split_metadata=split_metadata, terminal_policy=terminal_policy,
            authorized_at=utc_now(),
        )
        path = directory / f"{canonical_artifact_hash(authority)}.json"
        write_immutable_private_json(path, authority)
    return {**authority, "authority_sha256": path.stem,
            "private_authority_path": str(path)}


def replay_evaluation_apply_authority(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    split_policy_path: Path = DEFAULT_SPLIT_ACCESS_POLICY,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
    terminal_policy_path: Path = DEFAULT_TERMINAL_DECISION_POLICY,
) -> dict[str, Any]:
    """Replay terminal-evaluation authority without opening evaluation rows."""
    if not SHA256_RE.fullmatch(str(authority_sha256)):
        raise AcousticVerificationError("Evaluation authority hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "evaluation-authorities" / f"{authority_sha256}.json"
    require_private_file(path, root)
    authority = read_private_object(path)
    calibration_sha = str(authority.get("calibration_application_sha256") or "")
    calibration = replay_calibration_thresholds(
        calibration_sha, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    calibration_authority = replay_calibration_apply_authority(
        str(calibration["authority_sha256"]), runtime_root=root,
        p3_runtime_root=p3_runtime_root, split_policy_path=split_policy_path,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    expected = _evaluation_apply_authority_payload(
        calibration_application=calibration,
        calibration_application_sha256=calibration_sha,
        calibration_authority=calibration_authority,
        split_metadata=_evaluation_split_metadata_authority(
            split_policy_path, parent_corpus_manifest_path
        ),
        terminal_policy=_terminal_decision_policy(terminal_policy_path),
        authorized_at=str(authority.get("authorized_at") or ""),
    )
    if authority != expected or canonical_artifact_hash(authority) != authority_sha256:
        raise AcousticVerificationError("Evaluation authority replay is invalid.")
    return {**authority, "authority_sha256": authority_sha256,
            "private_authority_path": str(path)}


def _evaluation_records_after_authority(
    authority: Mapping[str, Any], *, parent_corpus_manifest_path: Path,
) -> list[dict[str, Any]]:
    if authority.get("intended_split") != "evaluation":
        raise AcousticVerificationError("Evaluation split authority is invalid.")
    parent_path = parent_corpus_manifest_path.expanduser().absolute()
    require_private_file(parent_path, parent_path.parent)
    if sha256_file(parent_path) != authority.get("parent_corpus_manifest_sha256"):
        raise AcousticVerificationError("Evaluation parent manifest drifted.")
    parent = read_private_object(parent_path)
    recordings = parent.get("recordings")
    if not isinstance(recordings, list):
        raise AcousticVerificationError("Evaluation parent records are invalid.")
    by_split = {
        split: [record for record in recordings
                if isinstance(record, Mapping) and record.get("split") == split]
        for split in ("development", "calibration", "evaluation")
    }
    selected = by_split["evaluation"]
    if (
        len(selected) != authority.get("evaluation_recording_count")
        or canonical_artifact_hash(selected) != authority.get("evaluation_record_set_sha256")
        or canonical_artifact_hash(sorted(str(item.get("conversation_id") or "") for item in selected))
        != authority.get("evaluation_conversation_set_sha256")
    ):
        raise AcousticVerificationError("Evaluation split membership drifted.")
    for key in ("recording_id", "conversation_id"):
        sets = [{str(item.get(key) or "") for item in by_split[split]}
                for split in ("development", "calibration", "evaluation")]
        if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise AcousticVerificationError(f"Evaluation {key} overlaps another split.")
    source_sets = [
        {str((item.get("source_blob") or {}).get("sha256") or "")
         for item in by_split[split] if isinstance(item.get("source_blob"), Mapping)}
        for split in ("development", "calibration", "evaluation")
    ]
    if any(source_sets[left] & source_sets[right]
           for left in range(3) for right in range(left + 1, 3)):
        raise AcousticVerificationError("Evaluation source content overlaps another split.")
    validated = []
    for value in selected:
        record = dict(value)
        source = record.get("source_blob")
        lineage = record.get("transcript_lineage")
        gold = record.get("operator_gold")
        if (not isinstance(source, Mapping) or not isinstance(lineage, Mapping)
                or not isinstance(gold, Mapping)
                or not isinstance(gold.get("speaker_truth"), list)
                or not isinstance(gold.get("same_person_label_groups"), list)):
            raise AcousticVerificationError("Evaluation record evidence is invalid.")
        source_path = Path(str(source.get("stored_path") or ""))
        transcript_path = Path(str(lineage.get("current_artifact_path") or ""))
        require_private_file(source_path, source_path.parent)
        require_private_file(transcript_path, transcript_path.parent)
        if (sha256_file(source_path) != source.get("sha256")
                or source_path.stat().st_size != source.get("bytes")
                or sha256_file(transcript_path) != lineage.get("current_artifact_sha256")):
            raise AcousticVerificationError("Evaluation source evidence drifted.")
        validated.append(record)
    return validated


def reveal_evaluation_split(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Reveal exact evaluation metadata and opaque gold after authority."""
    root = runtime_root.expanduser().absolute()
    authority = replay_evaluation_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    records = _evaluation_records_after_authority(
        authority, parent_corpus_manifest_path=parent_corpus_manifest_path
    )
    public_records = []
    for record in records:
        source, lineage, gold = (
            record["source_blob"], record["transcript_lineage"], record["operator_gold"]
        )
        public_records.append({
            "recording_id": record["recording_id"],
            "conversation_id": record["conversation_id"],
            "source_blob_id": source["blob_id"], "source_sha256": source["sha256"],
            "source_bytes": source["bytes"],
            "transcript_artifact_sha256": lineage["current_artifact_sha256"],
            "gold_id": gold["gold_id"],
            "speaker_truth": [dict(item) for item in gold["speaker_truth"]],
            "same_person_label_groups": [
                dict(item) if isinstance(item, Mapping) else list(item)
                for item in gold["same_person_label_groups"]
            ],
            "conditions": dict(record.get("conditions") or {}),
        })
    receipt = {
        "schema_version": EVALUATION_SPLIT_REVEAL_SCHEMA,
        "status": "success", "reason_code": None,
        "authority_sha256": authority_sha256, "intended_split": "evaluation",
        "record_set_sha256": authority["evaluation_record_set_sha256"],
        "conversation_set_sha256": authority["evaluation_conversation_set_sha256"],
        "record_count": len(public_records),
        "conversation_count": len({item["conversation_id"] for item in public_records}),
        "records": public_records, "development_disjoint": True,
        "calibration_disjoint": True, "source_content_disjoint": True,
        "contains_opaque_gold_labels": True, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "will_perform_external_write": False, "revealed_at": utc_now(),
    }
    receipt_sha = _calibration_stage_identity(receipt, "revealed_at")
    path = root / "evaluation-stages" / authority_sha256 / "split-reveal.json"
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("revealed_at",))
    return {**stored, "split_reveal_sha256": receipt_sha,
            "private_split_reveal_path": str(path)}


def prepare_evaluation_split(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
    parent_corpus_manifest_path: Path = DEFAULT_PARENT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Run exact P1/P2 preparation for all authorized evaluation records."""
    root = runtime_root.expanduser().absolute()
    authority = replay_evaluation_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    reveal = reveal_evaluation_split(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root,
        parent_corpus_manifest_path=parent_corpus_manifest_path,
    )
    path = root / "evaluation-stages" / authority_sha256 / "preparation.json"
    if path.exists():
        require_private_file(path, root)
        existing = read_private_object(path)
        existing_sha = _calibration_stage_identity(existing, "prepared_at")
        if (existing.get("status") != "success"
                or existing.get("authority_sha256") != authority_sha256
                or existing.get("split_reveal_sha256") != reveal["split_reveal_sha256"]):
            raise AcousticVerificationError("Evaluation preparation conflicts.")
        return {**existing, "preparation_sha256": existing_sha,
                "private_preparation_path": str(path)}
    records = _evaluation_records_after_authority(
        authority, parent_corpus_manifest_path=parent_corpus_manifest_path
    )
    p1_root = root / "evaluation-preparation" / authority_sha256 / "p1"
    p2_root = root / "evaluation-preparation" / authority_sha256 / "p2"
    units = []
    for record in records:
        source = record["source_blob"]
        source_path = Path(str(source["stored_path"]))
        source_sha = str(source["sha256"])
        source_blob_id = "source-" + source_sha[:24]
        p1_plan = audio_derivatives.dry_run(
            source_path, runtime_root=p1_root, source_blob_id=source_blob_id,
            expected_source_sha256=source_sha,
            channel_policy="stereo_average_to_mono",
            channel_policy_authority_sha256=authority_sha256,
        )
        p1_applied = audio_derivatives.apply_derivative(
            source_path, runtime_root=p1_root,
            approval_token=audio_derivatives.APPLY_TOKEN,
            source_blob_id=source_blob_id, expected_source_sha256=source_sha,
            channel_policy="stereo_average_to_mono",
            channel_policy_authority_sha256=authority_sha256,
        )
        p1_replay = audio_derivatives.replay_derivative(
            p1_plan["run_id"], runtime_root=p1_root
        )
        p2_plan = speech_preparation.dry_run(
            p1_plan["run_id"], p1_runtime_root=p1_root, runtime_root=p2_root,
            intended_split="evaluation",
            split_access_authority_sha256=authority_sha256,
        )
        p2_applied = speech_preparation.apply_comparison(
            p1_plan["run_id"], p1_runtime_root=p1_root, runtime_root=p2_root,
            intended_split="evaluation",
            split_access_authority_sha256=authority_sha256,
        )
        p2_replay = speech_preparation.replay_comparison(
            p2_plan["run_id"], runtime_root=p2_root
        )
        methods = []
        for method in p2_applied["comparison"].get("method_results") or []:
            if not isinstance(method, Mapping) or method.get("status") != "success":
                raise AcousticVerificationError("Evaluation preparation method failed.")
            output_path = Path(str(method.get("output_path") or ""))
            require_private_file(output_path, output_path.parent)
            if sha256_file(output_path) != method.get("output_sha256"):
                raise AcousticVerificationError("Evaluation preparation output drifted.")
            methods.append({
                "method_id": method["method_id"],
                "method_result_sha256": canonical_artifact_hash(dict(method)),
                "output_path": str(output_path),
                "output_sha256": method["output_sha256"],
                "output_equivalence_class_sha256": method["output_sha256"],
                "speech_region_count": len(method.get("speech_regions") or []),
                "overlap_region_count": len(method.get("overlap_regions") or []),
                "speaker_change_region_count": len(method.get("speaker_change_regions") or []),
            })
        units.append({
            "recording_id": record["recording_id"],
            "conversation_id": record["conversation_id"],
            "source_sha256": source_sha, "p1_run_id": p1_plan["run_id"],
            "p1_manifest_sha256": p1_applied["manifest_sha256"],
            "p1_replay_receipt_sha256": sha256_file(Path(str(p1_replay["replay_receipt_path"]))),
            "p2_run_id": p2_plan["run_id"],
            "p2_comparison_path": p2_applied["comparison_path"],
            "p2_comparison_sha256": sha256_file(Path(str(p2_applied["comparison_path"]))),
            "p2_replay_receipt_sha256": sha256_file(Path(str(p2_replay["replay_path"]))),
            "methods": methods,
        })
    receipt = {
        "schema_version": EVALUATION_PREPARATION_SCHEMA,
        "status": "success", "reason_code": None,
        "authority_sha256": authority_sha256,
        "split_reveal_sha256": reveal["split_reveal_sha256"],
        "intended_split": "evaluation", "record_count": len(units),
        "method_attempts": len(units) * len(METHOD_IDS),
        "method_successes": sum(len(unit["methods"]) for unit in units),
        "units": units, "did_read_evaluation_audio": True,
        "did_run_p1_p2": True, "did_run_biometrics": False,
        "did_select_or_change_thresholds": False,
        "did_perform_external_write": False, "contains_raw_audio": False,
        "contains_transcript_text": False, "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False, "prepared_at": utc_now(),
    }
    receipt_sha = _calibration_stage_identity(receipt, "prepared_at")
    ensure_private_tree(root, path.parent)
    stored = write_immutable_private_json(path, receipt, volatile_fields=("prepared_at",))
    return {**stored, "preparation_sha256": receipt_sha,
            "private_preparation_path": str(path)}


def _evaluation_terminal_stop_payload(
    *, authority: Mapping[str, Any], authority_sha256: str, stopped_at: str,
) -> dict[str, Any]:
    expected_p2 = str(authority["preparation_contract"]["p2_module_sha256"])
    return {
        "schema_version": EVALUATION_APPLICATION_SCHEMA,
        "status": "stopped",
        "terminal_decision": "stop",
        "reason_codes": [
            "split_or_hash_integrity_failure",
            "required_candidate_or_preparation_path_not_run",
            "authority_bound_p2_lacks_evaluation_split_seam",
        ],
        "authority_sha256": authority_sha256,
        "split_reveal_sha256": EXPECTED_EVALUATION_SPLIT_REVEAL_SHA256,
        "terminal_decision_policy_sha256": authority[
            "terminal_decision_policy_sha256"
        ],
        "expected_authority_bound_p2_module_sha256": expected_p2,
        "observed_incompatible_attempt_p2_module_sha256": (
            OBSERVED_INCOMPATIBLE_EVALUATION_P2_MODULE_SHA256
        ),
        "restored_current_p2_module_sha256": expected_p2,
        "integrity_failure": (
            "evaluation_split_was_revealed_before_the_required_p2_evaluation_"
            "split_seam_was_present_in_the_authority_bound_module"
        ),
        "preparation_receipt_count": 0,
        "window_selection_receipt_count": 0,
        "score_matrix_receipt_count": 0,
        "logical_trial_count": 0,
        "audio_preparation_execution_count": 0,
        "model_execution_count": 0,
        "threshold_or_temperature_change_count": 0,
        "did_read_evaluation_split_metadata_and_opaque_gold": True,
        "did_read_evaluation_audio": False,
        "did_run_p1_p2": False,
        "did_select_windows": False,
        "did_run_biometrics": False,
        "did_compute_evaluation_metrics": False,
        "did_change_thresholds_temperatures_profiles_or_policy": False,
        "did_make_model_or_method_selection": False,
        "did_enable_default_integration": False,
        "did_perform_external_write": False,
        "evaluation_generation_state": (
            "terminally_stopped_revealed_not_reusable_for_terminal_selection"
        ),
        "required_follow_up": "new_sealed_evaluation_cohort_and_authority_generation",
        "current_split_future_use": "nonblind_diagnostic_or_refinement_only",
        "contains_biometric_scores": False,
        "contains_frozen_thresholds": False,
        "contains_opaque_gold_labels": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_names_or_emails": False,
        "contains_embeddings_or_vectors": False,
        "contains_raw_biometric_values": False,
        "stopped_at": stopped_at,
    }


def record_evaluation_terminal_stop(
    authority_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
) -> dict[str, Any]:
    """Persist the fail-closed terminal decision without reopening evaluation."""
    root = runtime_root.expanduser().absolute()
    authority = replay_evaluation_apply_authority(
        authority_sha256, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    split_path = root / "evaluation-stages" / authority_sha256 / "split-reveal.json"
    require_private_file(split_path, root)
    for forbidden_stage in ("preparation.json", "window-selection.json", "score-matrix.json"):
        if (split_path.parent / forbidden_stage).exists():
            raise AcousticVerificationError(
                "Evaluation stop evidence conflicts with an executed later stage."
            )
    if sha256_file(Path(speech_preparation.__file__).resolve()) != authority[
        "preparation_contract"
    ]["p2_module_sha256"]:
        raise AcousticVerificationError(
            "Evaluation stop requires the authority-bound P2 module restored."
        )
    directory = root / "evaluation-applications"
    ensure_private_tree(root, directory)
    matches = []
    for path in sorted(directory.glob("*.json")):
        require_private_file(path, root)
        value = read_private_object(path)
        if value.get("authority_sha256") != authority_sha256:
            continue
        expected = _evaluation_terminal_stop_payload(
            authority=authority, authority_sha256=authority_sha256,
            stopped_at=str(value.get("stopped_at") or ""),
        )
        if value != expected or _calibration_stage_identity(value, "stopped_at") != path.stem:
            raise AcousticVerificationError("Evaluation stop receipt is invalid.")
        matches.append((path, value))
    if len(matches) > 1:
        raise AcousticVerificationError("Multiple evaluation terminal decisions exist.")
    if matches:
        path, receipt = matches[0]
    else:
        receipt = _evaluation_terminal_stop_payload(
            authority=authority, authority_sha256=authority_sha256,
            stopped_at=utc_now(),
        )
        receipt_sha = _calibration_stage_identity(receipt, "stopped_at")
        path = directory / f"{receipt_sha}.json"
        write_immutable_private_json(path, receipt, volatile_fields=("stopped_at",))
    return {**receipt, "application_sha256": path.stem,
            "private_application_path": str(path)}


def replay_evaluation_terminal_stop(
    application_sha256: str, *, runtime_root: Path, p3_runtime_root: Path,
) -> dict[str, Any]:
    """Replay the stop without reading the split receipt body or evaluation."""
    if not SHA256_RE.fullmatch(str(application_sha256)):
        raise AcousticVerificationError("Evaluation stop hash is invalid.")
    root = runtime_root.expanduser().absolute()
    path = root / "evaluation-applications" / f"{application_sha256}.json"
    require_private_file(path, root)
    receipt = read_private_object(path)
    authority_sha = str(receipt.get("authority_sha256") or "")
    authority = replay_evaluation_apply_authority(
        authority_sha, runtime_root=root, p3_runtime_root=p3_runtime_root
    )
    split_path = root / "evaluation-stages" / authority_sha / "split-reveal.json"
    require_private_file(split_path, root)
    expected = _evaluation_terminal_stop_payload(
        authority=authority, authority_sha256=authority_sha,
        stopped_at=str(receipt.get("stopped_at") or ""),
    )
    if (receipt != expected
            or _calibration_stage_identity(receipt, "stopped_at") != application_sha256):
        raise AcousticVerificationError("Evaluation stop replay is invalid.")
    return {**receipt, "application_sha256": application_sha256,
            "private_application_path": str(path),
            "replay_mode": "metadata_only_without_evaluation_or_split_body_read"}


def materialize_profile(
    person_ref_id: str,
    *,
    adapter: VerificationAdapter,
    windows: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Materialize one synthetic-only P4B profile through the shared core."""
    if (
        not isinstance(preprocessing, Mapping)
        or set(preprocessing) != {"method_id", "revision"}
        or preprocessing.get("method_id") != "synthetic_raw"
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}",
            str(preprocessing.get("revision", "")),
        )
    ):
        raise AcousticVerificationError(
            "P4B preprocessing must use the exact synthetic schema."
        )
    if _contains_forbidden_private_key(preprocessing):
        raise AcousticVerificationError(
            "Profile preprocessing contains forbidden private metadata."
        )
    resolved = resolve_eligible_reference(
        person_ref_id, runtime_root=p3_runtime_root
    )
    if resolved.get("materialization_contract") != "stage_then_register_then_promote":
        raise AcousticVerificationError("P3 materialization contract is invalid.")
    reference = resolved.get("reference")
    sources = reference.get("sources") if isinstance(reference, Mapping) else None
    if not isinstance(sources, list) or not sources:
        raise AcousticVerificationError(
            "P4B materialization requires synthetic fixture authority."
        )
    for source in sources:
        fixture = source.get("fixture_authority") if isinstance(source, Mapping) else None
        if (
            not isinstance(fixture, Mapping)
            or fixture.get("schema_version")
            != "transcribe-audio.synthetic-reference-fixture.v1"
            or fixture.get("source_sha256") != source.get("source_sha256")
            or source.get("device_class") != "synthetic-fixture"
        ):
            raise AcousticVerificationError(
                "P4B materialization requires synthetic fixture authority."
            )
    return _materialize_profile_core(
        resolved=resolved,
        adapter=adapter,
        windows=windows,
        preprocessing=preprocessing,
        runtime_root=runtime_root,
        p3_runtime_root=p3_runtime_root,
    )


def _materialize_profile_core(
    *,
    resolved: Mapping[str, Any],
    adapter: VerificationAdapter,
    windows: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Materialize after the caller has proved synthetic or exact real authority."""
    if not windows:
        raise AcousticVerificationError("Profile materialization requires windows.")
    if (
        not isinstance(preprocessing, Mapping)
        or set(preprocessing) != {"method_id", "revision"}
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}",
            str(preprocessing.get("method_id", "")),
        )
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}",
            str(preprocessing.get("revision", "")),
        )
    ):
        raise AcousticVerificationError("Profile preprocessing schema is invalid.")
    if _contains_forbidden_private_key(preprocessing):
        raise AcousticVerificationError(
            "Profile preprocessing contains forbidden private metadata."
        )
    if resolved.get("materialization_contract") != "stage_then_register_then_promote":
        raise AcousticVerificationError("P3 materialization contract is invalid.")
    vectors: list[tuple[float, ...]] = []
    window_hashes: list[str] = []
    session_ids: list[str] = []
    for window in windows:
        if set(window) != {"session_id", "samples"}:
            raise AcousticVerificationError("Profile window shape is invalid.")
        session_id = str(window.get("session_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", session_id):
            raise AcousticVerificationError("Profile session ID must be opaque.")
        samples = window.get("samples")
        if not isinstance(samples, Sequence):
            raise AcousticVerificationError("Profile window samples are invalid.")
        vectors.append(
            _adapter_embedding(adapter, samples, sample_rate=16_000)
        )
        window_hashes.append(_window_hash(samples))
        session_ids.append(session_id)
    dimensions = {len(vector) for vector in vectors}
    if dimensions != {adapter.embedding_dimension}:
        raise AcousticVerificationError("Profile vector dimensions are inconsistent.")
    centroid = _normalized_embedding(
        [sum(vector[index] for vector in vectors) / len(vectors)
         for index in range(adapter.embedding_dimension)]
    )
    dispersion = sum(1.0 - cosine_score(centroid, vector) for vector in vectors) / len(vectors)
    identity = {
        "person_ref_id": resolved["person_ref_id"],
        "p3_profile_id": resolved["profile_id"],
        "generation_id": resolved["generation_id"],
        "generation_sha256": resolved["generation_sha256"],
        "candidate_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "preprocessing": dict(preprocessing),
        "window_sha256s": window_hashes,
        "session_ids": sorted(set(session_ids)),
    }
    identity_sha = canonical_artifact_hash(identity)
    profile_id = "verification-profile-" + identity_sha[:24]
    descendant_id = "verification-descendant-" + identity_sha[:24]
    root = runtime_root.expanduser().absolute()
    profile_dir = root / "profiles" / profile_id
    artifact_path = profile_dir / "aggregate.f32le"
    artifact_sha = _write_private_blob(
        artifact_path,
        struct.pack(f"<{len(centroid)}f", *centroid),
        root,
    )
    profile_manifest = {
        "schema_version": PROFILE_MANIFEST_SCHEMA,
        "profile_id": profile_id,
        "descendant_id": descendant_id,
        "person_ref_id": resolved["person_ref_id"],
        "p3_profile_id": resolved["profile_id"],
        "generation_id": resolved["generation_id"],
        "generation_sha256": resolved["generation_sha256"],
        "candidate_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "preprocessing": dict(preprocessing),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "vector_dimension": len(centroid),
        "window_count": len(vectors),
        "session_count": len(set(session_ids)),
        "dispersion": dispersion,
        "window_sha256s": window_hashes,
        "session_ids": sorted(set(session_ids)),
        "contains_raw_biometric_values": False,
    }
    profile_manifest_path = profile_dir / "manifest.json"
    write_immutable_private_json(profile_manifest_path, profile_manifest)
    profile_manifest_sha = sha256_file(profile_manifest_path)
    created_at = utc_now()
    with _profile_database(root) as connection:
        existing = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if existing is None:
            staged_lifecycle = _lifecycle_receipt(
                profile_id=profile_id,
                descendant_id=descendant_id,
                artifact_sha256=artifact_sha,
                profile_manifest_sha256=profile_manifest_sha,
                from_state=None,
                to_state="staged",
                reason="materialization_staged",
                previous_receipt_sha256=None,
                transitioned_at=created_at,
            )
            _authority_anchor(root, staged_lifecycle)
            staged_lifecycle_sha = canonical_artifact_hash(staged_lifecycle)
            connection.execute(
                """
                INSERT INTO profiles
                (profile_id, descendant_id, person_ref_id, p3_profile_id,
                 generation_id, generation_sha256, candidate_id, model_revision,
                 preprocessing_json, artifact_path, artifact_sha256,
                 vector_dimension, window_count, session_count, dispersion,
                 lifecycle_state, created_at, updated_at, state_receipt_sha256,
                 profile_manifest_path, profile_manifest_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id, descendant_id, resolved["person_ref_id"],
                    resolved["profile_id"], resolved["generation_id"],
                    resolved["generation_sha256"], adapter.candidate_id,
                    adapter.revision_sha,
                    json.dumps(dict(preprocessing), sort_keys=True, separators=(",", ":")),
                    str(artifact_path), artifact_sha, len(centroid), len(vectors),
                    len(set(session_ids)), dispersion, "staged", created_at, created_at,
                    staged_lifecycle_sha,
                    str(profile_manifest_path), profile_manifest_sha,
                ),
            )
        elif (
            existing["artifact_sha256"] != artifact_sha
            or existing["profile_manifest_sha256"] != profile_manifest_sha
            or existing["generation_sha256"] != resolved["generation_sha256"]
            or existing["lifecycle_state"] not in {"staged", "active"}
        ):
            raise AcousticVerificationError("Profile materialization conflicts.")
        elif existing["lifecycle_state"] == "active":
            if not descendant_is_eligible(
                descendant_id, runtime_root=p3_runtime_root
            ):
                raise AcousticVerificationError(
                    "Existing profile descendant is no longer eligible."
                )
            replay_profile(profile_id, runtime_root=root)
            return _public_profile(existing)
    staging_identity = {
        "profile_id": profile_id,
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha,
        "generation_sha256": resolved["generation_sha256"],
        "candidate_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "window_count": len(vectors),
        "session_count": len(set(session_ids)),
        "preprocessing_sha256": canonical_artifact_hash(dict(preprocessing)),
    }
    materialization = {
        "schema_version": MATERIALIZATION_SCHEMA,
        "status": "staged",
        "profile_id": resolved["profile_id"],
        "generation_id": resolved["generation_id"],
        "generation_sha256": resolved["generation_sha256"],
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha,
        "staging_ref_sha256": canonical_artifact_hash(staging_identity),
        "eligible_for_use": False,
        "will_perform_external_write": False,
        "created_at": created_at,
    }
    materialization_path = _authority_anchor(root, materialization)
    materialization_sha = canonical_artifact_hash(materialization)
    registration = register_descendant(
        resolved["profile_id"], resolved["generation_id"], descendant_id,
        artifact_sha,
        materialization_receipt=materialization,
        authority_receipt_path=materialization_path,
        p4_authority_root=root / "authority",
        approval_token=(
            f"REGISTER_BIOMETRIC_DESCENDANT:{resolved['generation_id']}:"
            f"{descendant_id}:{artifact_sha}:{materialization_sha}"
        ),
        runtime_root=p3_runtime_root,
    )
    promoted_at = utc_now()
    promotion = {
        "schema_version": PROMOTION_SCHEMA,
        "status": "promoted",
        "descendant_id": descendant_id,
        "artifact_sha256": artifact_sha,
        "materialization_receipt_sha256": registration[
            "materialization_receipt_sha256"
        ],
        "eligible_for_use": True,
        "will_perform_external_write": False,
        "promoted_at": promoted_at,
    }
    promotion_path = _authority_anchor(root, promotion)
    acknowledge_descendant_promotion(
        descendant_id,
        promotion,
        authority_receipt_path=promotion_path,
        p4_authority_root=root / "authority",
        approval_token=registration["required_promotion_token"],
        runtime_root=p3_runtime_root,
    )
    if not descendant_is_eligible(descendant_id, runtime_root=p3_runtime_root):
        raise AcousticVerificationError("P3 descendant promotion did not become eligible.")
    with _profile_database(root) as connection:
        staged = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if staged is None:
            raise AcousticVerificationError("Staged profile disappeared.")
        active_lifecycle = _lifecycle_receipt(
            profile_id=profile_id,
            descendant_id=descendant_id,
            artifact_sha256=artifact_sha,
            profile_manifest_sha256=profile_manifest_sha,
            from_state="staged",
            to_state="active",
            reason="p3_descendant_promoted",
            previous_receipt_sha256=staged["state_receipt_sha256"],
            transitioned_at=promoted_at,
        )
        _authority_anchor(root, active_lifecycle)
        active_lifecycle_sha = canonical_artifact_hash(active_lifecycle)
        connection.execute(
            "UPDATE profiles SET lifecycle_state = 'active', updated_at = ?, "
            "state_receipt_sha256 = ? "
            "WHERE profile_id = ? AND lifecycle_state = 'staged'",
            (promoted_at, active_lifecycle_sha, profile_id),
        )
        row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    if row is None or row["lifecycle_state"] != "active":
        raise AcousticVerificationError("Profile promotion failed closed.")
    return _public_profile(row)


def score_profile(
    profile_id: str,
    *,
    adapter: VerificationAdapter,
    probe_samples: Sequence[float],
    sample_rate: int,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Score one probe with P4 and P3 eligibility checked on both sides."""
    root = runtime_root.expanduser().absolute()
    replayed = replay_profile(profile_id, runtime_root=root)
    if replayed["lifecycle_state"] != "active":
        raise AcousticVerificationError("Verification profile is not active.")
    with _profile_database(root) as connection:
        row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    if row is None or row["lifecycle_state"] != "active":
        raise AcousticVerificationError("Verification profile is not active.")
    if (
        row["candidate_id"] != adapter.candidate_id
        or row["model_revision"] != adapter.revision_sha
    ):
        raise AcousticVerificationError("Verification adapter binding changed.")
    descendant_id = str(row["descendant_id"])
    if not descendant_is_eligible(descendant_id, runtime_root=p3_runtime_root):
        raise AcousticVerificationError("P3 descendant is not eligible.")
    artifact_path = Path(str(row["artifact_path"]))
    require_private_file(artifact_path, root)
    if sha256_file(artifact_path) != row["artifact_sha256"]:
        raise AcousticVerificationError("Private profile artifact hash mismatch.")
    payload = artifact_path.read_bytes()
    expected_size = int(row["vector_dimension"]) * 4
    if len(payload) != expected_size:
        raise AcousticVerificationError("Private profile artifact size changed.")
    centroid = struct.unpack(f"<{row['vector_dimension']}f", payload)
    probe = _adapter_embedding(adapter, probe_samples, sample_rate=sample_rate)
    score = cosine_score(centroid, probe)
    current = replay_profile(profile_id, runtime_root=root)
    if (
        current["lifecycle_state"] != "active"
        or not descendant_is_eligible(descendant_id, runtime_root=p3_runtime_root)
    ):
        raise AcousticVerificationError(
            "Verification profile eligibility changed during scoring."
        )
    trial_identity = {
        "profile_id": profile_id,
        "descendant_id": descendant_id,
        "artifact_sha256": row["artifact_sha256"],
        "candidate_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "probe_sha256": _window_hash(probe_samples),
        "score": score,
    }
    return {
        "schema_version": TRIAL_SCHEMA,
        "trial_id": "verification-trial-" + canonical_artifact_hash(trial_identity)[:24],
        "status": "success",
        "reason_code": None,
        "profile_id": profile_id,
        "descendant_id": descendant_id,
        "candidate_id": adapter.candidate_id,
        "model_revision": adapter.revision_sha,
        "probe_sha256": trial_identity["probe_sha256"],
        "score": score,
        "p4_state_verified_before_and_after": True,
        "p3_eligibility_verified_before_and_after": True,
        "will_perform_external_write": False,
    }


def replay_profile(
    profile_id: str, *, runtime_root: Path
) -> dict[str, Any]:
    """Validate P4 metadata plus private-byte presence for one profile."""
    root = runtime_root.expanduser().absolute()
    with _profile_database(root) as connection:
        row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    if row is None:
        raise AcousticVerificationError("Verification profile does not exist.")
    _require_current_lifecycle_receipt(root, row)
    manifest_path = Path(str(row["profile_manifest_path"] or ""))
    if row["lifecycle_state"] != "deleted":
        require_private_file(manifest_path, root)
        if sha256_file(manifest_path) != row["profile_manifest_sha256"]:
            raise AcousticVerificationError("Profile manifest hash mismatch.")
        manifest = read_private_object(manifest_path)
        expected_manifest_fields = {
            "profile_id": row["profile_id"],
            "descendant_id": row["descendant_id"],
            "person_ref_id": row["person_ref_id"],
            "p3_profile_id": row["p3_profile_id"],
            "generation_id": row["generation_id"],
            "generation_sha256": row["generation_sha256"],
            "candidate_id": row["candidate_id"],
            "model_revision": row["model_revision"],
            "preprocessing": json.loads(row["preprocessing_json"]),
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "vector_dimension": row["vector_dimension"],
            "window_count": row["window_count"],
            "session_count": row["session_count"],
            "dispersion": row["dispersion"],
        }
        if (
            manifest.get("schema_version") != PROFILE_MANIFEST_SCHEMA
            or manifest.get("contains_raw_biometric_values") is not False
            or any(manifest.get(key) != value for key, value in expected_manifest_fields.items())
        ):
            raise AcousticVerificationError("Profile manifest binding is invalid.")
    artifact_path = Path(str(row["artifact_path"]))
    if row["lifecycle_state"] == "deleted":
        if artifact_path.exists():
            raise AcousticVerificationError("Deleted profile retains private bytes.")
        tombstone_path = Path(str(row["tombstone_path"] or ""))
        require_private_file(tombstone_path, root)
        tombstone = read_private_object(tombstone_path)
        if (
            tombstone.get("profile_id") != profile_id
            or tombstone.get("prior_artifact_sha256") != row["artifact_sha256"]
            or tombstone.get("prior_profile_manifest_sha256")
            != row["profile_manifest_sha256"]
        ):
            raise AcousticVerificationError("Profile tombstone binding is invalid.")
        private_bytes_present = False
    else:
        require_private_file(artifact_path, root)
        if sha256_file(artifact_path) != row["artifact_sha256"]:
            raise AcousticVerificationError("Private profile artifact hash mismatch.")
        private_bytes_present = True
    return {
        **_public_profile(row),
        "private_bytes_present": private_bytes_present,
        "tombstone_path": row["tombstone_path"],
        "replayed_at": utc_now(),
    }


def _disable_profile(
    profile_id: str,
    *,
    target_state: str,
    operator_reason: str,
    replacement_profile_id: Optional[str] = None,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    if target_state not in {"withdrawn", "superseded"}:
        raise AcousticVerificationError("Profile transition is invalid.")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", operator_reason):
        raise AcousticVerificationError("Profile transition reason is invalid.")
    root = runtime_root.expanduser().absolute()
    proposed_transitioned_at = utc_now()
    with _profile_database(root) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise AcousticVerificationError("Verification profile does not exist.")
        if target_state == "superseded":
            replacement = connection.execute(
                "SELECT * FROM profiles WHERE profile_id = ?",
                (replacement_profile_id,),
            ).fetchone()
            if (
                replacement is None
                or replacement["profile_id"] == profile_id
                or replacement["lifecycle_state"] != "active"
                or replacement["person_ref_id"] != row["person_ref_id"]
            ):
                raise AcousticVerificationError(
                    "Profile replacement must be active for the same person."
                )
        elif replacement_profile_id is not None:
            raise AcousticVerificationError(
                "Only supersession can bind a replacement profile."
            )
        if row["lifecycle_state"] == "active":
            lifecycle = _lifecycle_receipt(
                profile_id=profile_id,
                descendant_id=row["descendant_id"],
                artifact_sha256=row["artifact_sha256"],
                profile_manifest_sha256=row["profile_manifest_sha256"],
                from_state="active",
                to_state=target_state,
                reason=operator_reason,
                previous_receipt_sha256=row["state_receipt_sha256"],
                replacement_profile_id=replacement_profile_id,
                transitioned_at=proposed_transitioned_at,
            )
            _authority_anchor(root, lifecycle)
            lifecycle_sha = canonical_artifact_hash(lifecycle)
            connection.execute(
                "UPDATE profiles SET lifecycle_state = ?, updated_at = ?, "
                "replacement_profile_id = ?, state_receipt_sha256 = ? "
                "WHERE profile_id = ? AND lifecycle_state = 'active'",
                (
                    target_state,
                    proposed_transitioned_at,
                    replacement_profile_id,
                    lifecycle_sha,
                    profile_id,
                ),
            )
        elif row["lifecycle_state"] != target_state:
            raise AcousticVerificationError("Profile lifecycle transition conflicts.")
        elif row["replacement_profile_id"] != replacement_profile_id:
            raise AcousticVerificationError("Profile replacement binding conflicts.")
        current_row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if current_row is None:
            raise AcousticVerificationError("Profile transition disappeared.")
        lifecycle = _require_current_lifecycle_receipt(root, current_row)
        if (
            lifecycle.get("reason") != operator_reason
            or lifecycle.get("to_state") != target_state
        ):
            raise AcousticVerificationError("Profile transition receipt conflicts.")
        row = current_row
    invalidation_reason = f"p4_profile_{target_state}"
    requested = request_descendant_invalidation(
        str(row["descendant_id"]),
        reason=invalidation_reason,
        approval_token=(
            f"INVALIDATE_BIOMETRIC_DESCENDANT:{row['descendant_id']}:"
            f"{row['artifact_sha256']}:{invalidation_reason}"
        ),
        runtime_root=p3_runtime_root,
    )
    requested_at = str(requested.get("requested_at", ""))
    if not requested_at.endswith("Z"):
        raise AcousticVerificationError("P3 invalidation request time is invalid.")
    evidence_sha = canonical_artifact_hash(
        {
            "profile_lifecycle_receipt_sha256": row["state_receipt_sha256"],
            "invalidation_requested_at": requested_at,
        }
    )
    invalidation = {
        "schema_version": INVALIDATION_SCHEMA,
        "status": "invalidated",
        "descendant_id": row["descendant_id"],
        "artifact_sha256": row["artifact_sha256"],
        "reason": invalidation_reason,
        "evidence_sha256": evidence_sha,
        "will_perform_external_write": False,
        "acknowledged_at": requested_at,
    }
    authority_path = _authority_anchor(root, invalidation)
    acknowledge_descendant_invalidation(
        str(row["descendant_id"]),
        invalidation,
        authority_receipt_path=authority_path,
        p4_authority_root=root / "authority",
        approval_token=requested["required_acknowledgment_token"],
        runtime_root=p3_runtime_root,
    )
    if descendant_is_eligible(str(row["descendant_id"]), runtime_root=p3_runtime_root):
        raise AcousticVerificationError("P3 descendant invalidation failed closed.")
    invalidation_sha = canonical_artifact_hash(invalidation)
    if (
        row["invalidation_receipt_sha256"] is not None
        and row["invalidation_receipt_sha256"] != invalidation_sha
    ):
        raise AcousticVerificationError("Profile invalidation receipt conflicts.")
    with _profile_database(root) as connection:
        connection.execute(
            "UPDATE profiles SET invalidation_receipt_sha256 = ?, updated_at = ? "
            "WHERE profile_id = ? AND lifecycle_state = ?",
            (invalidation_sha, requested_at, profile_id, target_state),
        )
        current = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    if current is None:
        raise AcousticVerificationError("Profile transition disappeared.")
    return _public_profile(current)


def withdraw_profile(
    profile_id: str,
    *,
    reason: str,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    return _disable_profile(
        profile_id,
        target_state="withdrawn",
        operator_reason=reason,
        runtime_root=runtime_root,
        p3_runtime_root=p3_runtime_root,
    )


def supersede_profile(
    profile_id: str,
    *,
    replacement_profile_id: str,
    reason: str,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    return _disable_profile(
        profile_id,
        target_state="superseded",
        operator_reason=reason,
        replacement_profile_id=replacement_profile_id,
        runtime_root=runtime_root,
        p3_runtime_root=p3_runtime_root,
    )


def delete_profile(
    profile_id: str,
    *,
    reason: str,
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Delete private profile bytes after acknowledged descendant invalidation."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", reason):
        raise AcousticVerificationError("Profile deletion reason is invalid.")
    root = runtime_root.expanduser().absolute()
    with _profile_database(root) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise AcousticVerificationError("Verification profile does not exist.")
        if row["lifecycle_state"] == "deleted":
            connection.execute("COMMIT")
            return replay_profile(profile_id, runtime_root=root)
        if row["lifecycle_state"] not in {"withdrawn", "superseded"}:
            raise AcousticVerificationError(
                "Profile must be withdrawn or superseded before deletion."
            )
        if row["invalidation_receipt_sha256"] is None:
            raise AcousticVerificationError(
                "Profile deletion requires invalidation acknowledgment."
            )
        if descendant_is_eligible(
            str(row["descendant_id"]), runtime_root=p3_runtime_root
        ):
            raise AcousticVerificationError(
                "Profile descendant remains eligible during deletion."
            )
        artifact_path = Path(str(row["artifact_path"]))
        require_private_file(artifact_path, root)
        if sha256_file(artifact_path) != row["artifact_sha256"]:
            raise AcousticVerificationError("Private profile artifact hash mismatch.")
        deleted_at = utc_now()
        deleted_lifecycle = _lifecycle_receipt(
            profile_id=profile_id,
            descendant_id=row["descendant_id"],
            artifact_sha256=row["artifact_sha256"],
            profile_manifest_sha256=row["profile_manifest_sha256"],
            from_state=row["lifecycle_state"],
            to_state="deleted",
            reason=reason,
            previous_receipt_sha256=row["state_receipt_sha256"],
            replacement_profile_id=row["replacement_profile_id"],
            transitioned_at=deleted_at,
        )
        _authority_anchor(root, deleted_lifecycle)
        deleted_lifecycle_sha = canonical_artifact_hash(deleted_lifecycle)
        tombstone = {
            "schema_version": "transcribe-audio.biometric-profile-tombstone.v1",
            "profile_id": profile_id,
            "descendant_id": row["descendant_id"],
            "person_ref_id": row["person_ref_id"],
            "generation_id": row["generation_id"],
            "candidate_id": row["candidate_id"],
            "prior_artifact_sha256": row["artifact_sha256"],
            "prior_profile_manifest_sha256": row["profile_manifest_sha256"],
            "invalidation_receipt_sha256": row[
                "invalidation_receipt_sha256"
            ],
            "replacement_profile_id": row["replacement_profile_id"],
            "state_receipt_sha256": deleted_lifecycle_sha,
            "reason": reason,
            "deleted_at": deleted_at,
            "private_bytes_retained": False,
        }
        tombstone_path = root / "profiles" / profile_id / "tombstone.json"
        write_immutable_private_json(tombstone_path, tombstone)
        artifact_path.unlink()
        manifest_path = Path(str(row["profile_manifest_path"]))
        require_private_file(manifest_path, root)
        manifest_path.unlink()
        connection.execute(
            "UPDATE profiles SET lifecycle_state = 'deleted', updated_at = ?, "
            "tombstone_path = ?, state_receipt_sha256 = ? WHERE profile_id = ?",
            (deleted_at, str(tombstone_path), deleted_lifecycle_sha, profile_id),
        )
        connection.execute("COMMIT")
    return {
        **replay_profile(profile_id, runtime_root=root),
        "tombstone_path": str(tombstone_path),
    }
