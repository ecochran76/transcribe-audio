"""Fail-closed verification-model acquisition evidence for Plan 0037 P4.

This module does not read audio, execute models, or authorize biometric
enrollment.  It records and replays the exact acquisition proposal that later
P4 work may apply under the operator's bounded acquisition grant.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import struct
import tempfile
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
    INVALIDATION_SCHEMA,
    MATERIALIZATION_SCHEMA,
    PROMOTION_SCHEMA,
    acknowledge_descendant_invalidation,
    acknowledge_descendant_promotion,
    descendant_is_eligible,
    register_descendant,
    request_descendant_invalidation,
    resolve_eligible_reference,
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


def materialize_profile(
    person_ref_id: str,
    *,
    adapter: VerificationAdapter,
    windows: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, Any],
    runtime_root: Path,
    p3_runtime_root: Path,
) -> dict[str, Any]:
    """Materialize, register, and promote one private model-specific profile."""
    if not windows:
        raise AcousticVerificationError("Profile materialization requires windows.")
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
            return replay_profile(profile_id, runtime_root=root)
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
