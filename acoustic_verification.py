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
    if method_id != "no_enhancement":
        raise AcousticVerificationError(
            "P4C enrollment is limited to no-enhancement preparation."
        )
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
        if any(
            resolved_lineage.get(key) != lineage.get(key)
            for key in (
                "run_id",
                "method_id",
                "replay_receipt_sha256",
                "comparison_path",
                "comparison_sha256",
                "method_result_sha256",
                "source_blob_id",
                "source_sha256",
                "audio_quality_sha256",
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
            != lineage.get("method_result_sha256")
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
