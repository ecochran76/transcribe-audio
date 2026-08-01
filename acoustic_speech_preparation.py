"""Fail-closed Plan 0037 P2 speech-preparation lifecycle.

The host seam owns readiness, private receipts, timing evidence, and lifecycle
state. Provider implementations remain internal adapters and unavailable
providers remain explicit rather than silently falling back.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import shutil
import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol

from acoustic_audio_derivatives import (
    DEFAULT_RUNTIME_ROOT as P1_DEFAULT_RUNTIME_ROOT,
    AudioDerivativeError,
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    resolve_active_derivative,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)


DRY_RUN_SCHEMA = "transcribe-audio.speech-preparation-dry-run.v1"
COMPARISON_SCHEMA = "transcribe-audio.speech-preparation-comparison.v1"
APPLY_RECEIPT_SCHEMA = "transcribe-audio.speech-preparation-apply.v1"
REPLAY_SCHEMA = "transcribe-audio.speech-preparation-replay.v1"
ROLLBACK_SCHEMA = "transcribe-audio.speech-preparation-rollback.v1"
LINEAGE_SCHEMA = "transcribe-audio.speech-preparation-lineage.v1"
ACQUISITION_SPEC_SCHEMA = "transcribe-audio.open-candidate-acquisition-spec.v1"
ACQUISITION_PLAN_SCHEMA = "transcribe-audio.open-candidate-acquisition-plan.v1"
# Backward-compatible names for callers with stored command templates. P2 no
# longer validates or requires either value under the operator's standing grant.
APPLY_TOKEN = "APPLY_SPEECH_PREPARATION"
ROLLBACK_TOKEN = "ROLLBACK_SPEECH_PREPARATION"
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/speech-preparation"
)
DEFAULT_OPEN_ACQUISITION_ROOT = (
    DEFAULT_RUNTIME_ROOT / "acquisitions/open-candidates-20260731"
)
DEFAULT_OPEN_ACQUISITION_MANIFEST = (
    DEFAULT_OPEN_ACQUISITION_ROOT / "acquisition-manifest-v2.json"
)
DEFAULT_PYANNOTE_ACQUISITION_ROOT = (
    DEFAULT_RUNTIME_ROOT / "acquisitions/pyannote-community-1-20260731"
)
DEFAULT_PYANNOTE_ACQUISITION_MANIFEST = (
    DEFAULT_PYANNOTE_ACQUISITION_ROOT / "acquisition-manifest.json"
)
METHOD_IDS = (
    "no_enhancement",
    "silero_vad",
    "deepfilternet",
    "rnnoise",
    "pyannote_community_1",
)
EXPECTED_PACKAGES = {
    "silero_vad": ("silero-vad", "6.2.1"),
    "deepfilternet": ("deepfilternet", "0.5.6"),
    "pyannote_community_1": ("pyannote-audio", "4.0.4"),
}
DEFAULT_ACQUISITION_SPEC = (
    Path(__file__).parent
    / "docs/dev/fixtures/plan-0037-p2/open-candidate-acquisition-plan.json"
)
OPEN_CANDIDATE_IDS = ("silero_vad", "deepfilternet", "rnnoise")


class SpeechPreparationError(ValueError):
    """Raised when P2 evidence cannot be made complete and replayable."""


def _private_require(path: Path, root: Path) -> None:
    try:
        require_private_file(path, root)
    except AudioDerivativeError as exc:
        raise SpeechPreparationError(str(exc)) from exc


def _private_read(path: Path) -> dict[str, Any]:
    try:
        return read_private_object(path)
    except AudioDerivativeError as exc:
        raise SpeechPreparationError(str(exc)) from exc


def _private_write(
    path: Path,
    payload: dict[str, Any],
    *,
    volatile_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        return write_immutable_private_json(
            path, payload, volatile_fields=volatile_fields
        )
    except AudioDerivativeError as exc:
        raise SpeechPreparationError(str(exc)) from exc


class PreparationAdapter(Protocol):
    """Stable host seam; provider-native values must not escape it."""

    method_id: str

    def descriptor(self) -> dict[str, Any]: ...

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]: ...


class NoEnhancementAdapter:
    method_id = "no_enhancement"

    def descriptor(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "adapter_revision": "host-no-enhancement-v1",
            "operation": "identity_reference",
            "executes_audio": False,
            "executes_model": False,
        }

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        derived = source["derived_audio"]
        return {
            "status": "success",
            "reason_code": None,
            "output_artifact_id": derived["artifact_id"],
            "output_sha256": derived["output_sha256"],
            "output_path": source["artifact_path"],
            "timestamp_map": derived["timestamp_map"],
            "speech_regions": None,
            "overlap_regions": None,
            "speaker_change_regions": None,
            "quality_delta": {
                "operation": "identity_reference",
                "changed": False,
            },
            "model_revisions": {},
            "warnings": [],
            "abstention_reasons": [
                "speech_regions_not_assessed_by_no_enhancement"
            ],
        }


def _read_pcm16_mono(path: Path, expected_rate: int = 16_000) -> tuple[Any, int]:
    import torch

    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != expected_rate
        ):
            raise SpeechPreparationError(
                f"P2 requires mono PCM16 audio at {expected_rate} Hz."
            )
        frames = audio.readframes(audio.getnframes())
    values = struct.unpack(f"<{len(frames) // 2}h", frames)
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0) / 32768.0, expected_rate


def _write_pcm16_mono(path: Path, audio: Any, sample_rate: int) -> tuple[Path, str]:
    import torch

    output = audio.detach().cpu().reshape(-1).clamp(-1.0, 1.0)
    samples = (output * 32767.0).round().to(torch.int16).tolist()
    frame_bytes = struct.pack(f"<{len(samples)}h", *samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(frame_bytes)
    content = buffer.getvalue()
    output_sha = hashlib.sha256(content).hexdigest()
    method_id = path.stem.split("-", 1)[0]
    output_path = path.parent / method_id / output_sha[:2] / f"{output_sha}.wav"
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for parent in (output_path.parent, output_path.parent.parent, path.parent):
        parent.chmod(0o700)
    if output_path.exists():
        if output_path.read_bytes() != content:
            raise SpeechPreparationError("P2 content-addressed output conflict.")
        return output_path, output_sha
    output_path.write_bytes(content)
    output_path.chmod(0o600)
    return output_path, output_sha


def _full_timestamp_map(source: Mapping[str, Any]) -> list[dict[str, float]]:
    duration = float(source["derived_audio"]["output_duration_seconds"])
    return [{
        "source_start_seconds": 0.0,
        "source_end_seconds": duration,
        "output_start_seconds": 0.0,
        "output_end_seconds": duration,
    }]


class SileroVadAdapter:
    method_id = "silero_vad"

    def __init__(self, *, model_sha256: str) -> None:
        self.model_sha256 = model_sha256

    def descriptor(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "adapter_revision": "host-silero-vad-v2",
            "operation": "speech_region_detection",
            "executes_audio": True,
            "executes_model": True,
            "model_sha256": self.model_sha256,
        }

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        from silero_vad import get_speech_timestamps, load_silero_vad

        audio, sample_rate = _read_pcm16_mono(Path(str(source["artifact_path"])))
        model = load_silero_vad(onnx=True, opset_version=16)
        timestamps = get_speech_timestamps(
            audio.reshape(-1), model, sampling_rate=sample_rate, return_seconds=True
        )
        regions = [
            {
                "start_seconds": float(item["start"]),
                "end_seconds": float(item["end"]),
            }
            for item in timestamps
        ]
        status = "success" if regions else "blocked"
        return {
            "status": status,
            "reason_code": None if regions else "abstained_no_speech",
            "output_artifact_id": source["derived_audio"]["artifact_id"],
            "output_sha256": source["artifact_sha256"],
            "output_path": source["artifact_path"],
            "timestamp_map": source["derived_audio"]["timestamp_map"],
            "speech_regions": regions,
            "overlap_regions": [],
            "speaker_change_regions": [],
            "quality_delta": None,
            "model_revisions": {
                "silero_vad": EXPECTED_PACKAGES["silero_vad"][1],
                "model_sha256": self.model_sha256,
            },
            "warnings": [],
            "abstention_reasons": [] if regions else ["no_speech_detected"],
        }


def _install_deepfilternet_torchaudio_compatibility() -> None:
    """Supply the type-only module removed by recent torchaudio releases."""
    try:
        import torchaudio.backend.common  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = type("AudioMetaData", (), {})
        backend = types.ModuleType("torchaudio.backend")
        backend.common = common
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common


class DeepFilterNetAdapter:
    method_id = "deepfilternet"

    def __init__(self, *, model_dir: Path, model_sha256: str, output_root: Path) -> None:
        self.model_dir = model_dir
        self.model_sha256 = model_sha256
        self.output_root = output_root

    def descriptor(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "adapter_revision": "host-deepfilternet-v2",
            "operation": "speech_enhancement",
            "executes_audio": True,
            "executes_model": True,
            "model_sha256": self.model_sha256,
        }

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from torchaudio.functional import resample

        _install_deepfilternet_torchaudio_compatibility()
        from df.enhance import enhance, init_df

        audio, sample_rate = _read_pcm16_mono(Path(str(source["artifact_path"])))
        model, state, _ = init_df(str(self.model_dir), log_file=None)
        prepared = resample(audio, sample_rate, 48_000).contiguous()
        chunk_samples = 60 * 48_000
        enhanced_chunks = []
        for offset in range(0, prepared.shape[-1], chunk_samples):
            chunk = prepared[..., offset : offset + chunk_samples].contiguous()
            if hasattr(state, "reset"):
                state.reset()
            enhanced_chunks.append(enhance(model, state, chunk).contiguous())
        enhanced = torch.cat(enhanced_chunks, dim=-1)[..., : prepared.shape[-1]]
        output = resample(enhanced, 48_000, sample_rate)
        output = output[..., : audio.shape[-1]]
        if output.shape[-1] < audio.shape[-1]:
            output = torch.nn.functional.pad(output, (0, audio.shape[-1] - output.shape[-1]))
        output_path = self.output_root / (
            f"deepfilternet-{str(source['artifact_sha256'])[:24]}.wav"
        )
        output_path, output_sha = _write_pcm16_mono(output_path, output, sample_rate)
        return {
            "status": "success",
            "reason_code": None,
            "output_artifact_id": "p2-deepfilternet-" + output_sha[:24],
            "output_sha256": output_sha,
            "output_path": str(output_path),
            "timestamp_map": _full_timestamp_map(source),
            "speech_regions": None,
            "overlap_regions": [],
            "speaker_change_regions": [],
            "quality_delta": {"status": "not_assessed_in_adapter"},
            "model_revisions": {
                "deepfilternet": EXPECTED_PACKAGES["deepfilternet"][1],
                "model_sha256": self.model_sha256,
            },
            "warnings": [],
            "abstention_reasons": [],
        }


class RNNoiseAdapter:
    method_id = "rnnoise"

    def __init__(self, *, library_path: Path, library_sha256: str, output_root: Path) -> None:
        self.library_path = library_path
        self.library_sha256 = library_sha256
        self.output_root = output_root

    def descriptor(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "adapter_revision": "host-rnnoise-v2",
            "operation": "noise_suppression",
            "executes_audio": True,
            "executes_model": True,
            "library_sha256": self.library_sha256,
        }

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from torchaudio.functional import resample

        audio, sample_rate = _read_pcm16_mono(Path(str(source["artifact_path"])))
        prepared = resample(audio, sample_rate, 48_000).reshape(-1)
        library = ctypes.CDLL(str(self.library_path))
        library.rnnoise_create.restype = ctypes.c_void_p
        library.rnnoise_create.argtypes = [ctypes.c_void_p]
        library.rnnoise_process_frame.restype = ctypes.c_float
        library.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        state = library.rnnoise_create(None)
        if not state:
            raise SpeechPreparationError("RNNoise state initialization failed.")
        chunks: list[Any] = []
        try:
            for offset in range(0, prepared.numel(), 480):
                frame = prepared[offset : offset + 480]
                valid = frame.numel()
                if valid < 480:
                    frame = torch.nn.functional.pad(frame, (0, 480 - valid))
                values = (frame * 32768.0).tolist()
                input_frame = (ctypes.c_float * 480)(*values)
                output_frame = (ctypes.c_float * 480)()
                library.rnnoise_process_frame(state, output_frame, input_frame)
                chunks.append(torch.tensor(list(output_frame)[:valid]) / 32768.0)
        finally:
            library.rnnoise_destroy(state)
        denoised = torch.cat(chunks).unsqueeze(0)
        output = resample(denoised, 48_000, sample_rate)[..., : audio.shape[-1]]
        if output.shape[-1] < audio.shape[-1]:
            output = torch.nn.functional.pad(output, (0, audio.shape[-1] - output.shape[-1]))
        output_path = self.output_root / f"rnnoise-{str(source['artifact_sha256'])[:24]}.wav"
        output_path, output_sha = _write_pcm16_mono(output_path, output, sample_rate)
        return {
            "status": "success",
            "reason_code": None,
            "output_artifact_id": "p2-rnnoise-" + output_sha[:24],
            "output_sha256": output_sha,
            "output_path": str(output_path),
            "timestamp_map": _full_timestamp_map(source),
            "speech_regions": None,
            "overlap_regions": [],
            "speaker_change_regions": [],
            "quality_delta": {"status": "not_assessed_in_adapter"},
            "model_revisions": {
                "rnnoise": "v0.2",
                "library_sha256": self.library_sha256,
            },
            "warnings": [],
            "abstention_reasons": [],
        }


def _activity_regions(
    turns: list[tuple[float, float, str]], *, minimum_count: int
) -> list[dict[str, float]]:
    events: dict[float, int] = {}
    for start, end, _ in turns:
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
    regions: list[dict[str, float]] = []
    active = 0
    previous: Optional[float] = None
    for position in sorted(events):
        if previous is not None and position > previous and active >= minimum_count:
            if regions and abs(regions[-1]["end_seconds"] - previous) < 1e-9:
                regions[-1]["end_seconds"] = position
            else:
                regions.append({
                    "start_seconds": previous,
                    "end_seconds": position,
                })
        active += events[position]
        previous = position
    return regions


def _bounded_turns(
    turns: list[tuple[float, float, str]], duration: float
) -> list[tuple[float, float, str]]:
    bounded: list[tuple[float, float, str]] = []
    for start, end, label in turns:
        clipped_start = max(0.0, min(duration, start))
        clipped_end = max(0.0, min(duration, end))
        if clipped_end > clipped_start:
            bounded.append((clipped_start, clipped_end, label))
    return bounded


def _speaker_change_regions(
    turns: list[tuple[float, float, str]], duration: float
) -> list[dict[str, float]]:
    changes: list[dict[str, float]] = []
    ordered = sorted(turns, key=lambda item: (item[0], item[1], item[2]))
    previous_label: Optional[str] = None
    for start, _, label in ordered:
        if previous_label is not None and label != previous_label and start < duration:
            end = min(duration, start + 0.001)
            if end > start and (not changes or start >= changes[-1]["end_seconds"]):
                changes.append({
                    "start_seconds": start,
                    "end_seconds": end,
                })
        previous_label = label
    return changes


class PyannoteCommunity1Adapter:
    method_id = "pyannote_community_1"

    def __init__(
        self, *, snapshot_dir: Path, revision_sha: str, asset_sha256: str
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.revision_sha = revision_sha
        self.asset_sha256 = asset_sha256

    def descriptor(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "adapter_revision": "host-pyannote-community-1-v1",
            "operation": "diarization_preparation",
            "executes_audio": True,
            "executes_model": True,
            "revision_sha": self.revision_sha,
            "asset_sha256": self.asset_sha256,
        }

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        import torch
        from pyannote.audio import Pipeline

        audio, sample_rate = _read_pcm16_mono(Path(str(source["artifact_path"])))
        pipeline = Pipeline.from_pretrained(self.snapshot_dir)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        output = pipeline({"waveform": audio, "sample_rate": sample_rate})
        annotation = output.speaker_diarization
        exclusive = output.exclusive_speaker_diarization
        turns = [
            (float(segment.start), float(segment.end), str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        exclusive_turns = [
            (float(segment.start), float(segment.end), str(label))
            for segment, _, label in exclusive.itertracks(yield_label=True)
        ]
        duration = float(source["derived_audio"]["output_duration_seconds"])
        bounded_turns = _bounded_turns(turns, duration)
        bounded_exclusive_turns = _bounded_turns(exclusive_turns, duration)
        speech_regions = _activity_regions(bounded_turns, minimum_count=1)
        overlap_regions = _activity_regions(bounded_turns, minimum_count=2)
        changes = _speaker_change_regions(bounded_exclusive_turns, duration)
        status = "success" if speech_regions else "blocked"
        return {
            "status": status,
            "reason_code": None if speech_regions else "abstained_no_speech",
            "output_artifact_id": source["derived_audio"]["artifact_id"],
            "output_sha256": source["artifact_sha256"],
            "output_path": source["artifact_path"],
            "timestamp_map": source["derived_audio"]["timestamp_map"],
            "speech_regions": speech_regions,
            "overlap_regions": overlap_regions,
            "speaker_change_regions": changes,
            "quality_delta": None,
            "model_revisions": {
                "pyannote_audio": EXPECTED_PACKAGES[self.method_id][1],
                "community_1_revision": self.revision_sha,
                "asset_sha256": self.asset_sha256,
            },
            "warnings": ["in_memory_pcm_decode"],
            "abstention_reasons": [] if speech_regions else ["no_speech_detected"],
        }

class FakePreparationAdapter:
    """Deterministic test adapter; never selected by the production registry."""

    def __init__(self, method_id: str, result: Mapping[str, Any]) -> None:
        if method_id not in METHOD_IDS:
            raise SpeechPreparationError("Fake adapter method is not a P2 method.")
        self.method_id = method_id
        self._result = dict(result)

    def prepare(self, source: Mapping[str, Any]) -> dict[str, Any]:
        del source
        return dict(self._result)

    def descriptor(self) -> dict[str, Any]:
        forbidden = sorted(_forbidden_result_keys(self._result))
        if forbidden:
            raise SpeechPreparationError(
                "Fake adapter result contains forbidden fields: "
                + ", ".join(forbidden)
            )
        return {
            "method_id": self.method_id,
            "adapter_revision": (
                "synthetic-fake-" + canonical_artifact_hash(self._result)[:24]
            ),
            "operation": "synthetic_test_only",
            "executes_audio": False,
            "executes_model": False,
        }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verified_open_acquisition() -> Optional[dict[str, Any]]:
    path = DEFAULT_OPEN_ACQUISITION_MANIFEST.expanduser().absolute()
    root = DEFAULT_OPEN_ACQUISITION_ROOT.expanduser().absolute()
    if not path.is_file():
        return None
    _private_require(path, root)
    manifest = _private_read(path)
    if (
        manifest.get("schema_version")
        != "transcribe-audio.open-candidate-acquisition-manifest.v1"
        or set(manifest) != {
            "schema_version", "authorization_basis", "source_plan_run_id",
            "source_plan_sha256", "supersedes_manifest_sha256", "artifacts",
            "candidates", "created_at",
        }
        or manifest.get("authorization_basis") != "operator_blanket_2026-07-31"
        or manifest.get("source_plan_run_id")
        != "acquire-open-585ef49febe61caf5a3d99b1"
        or manifest.get("source_plan_sha256")
        != "d4b2a4c800b10cd8604b4e2f73ac553a097652f0bc1271ff27def5628c9ac836"
        or not re.fullmatch(
            r"[a-f0-9]{64}", str(manifest.get("supersedes_manifest_sha256"))
        )
        or not isinstance(manifest.get("artifacts"), Mapping)
        or not isinstance(manifest.get("candidates"), Mapping)
    ):
        raise SpeechPreparationError("Open-candidate acquisition manifest is invalid.")
    artifacts = manifest["artifacts"]
    for artifact_id, value in artifacts.items():
        if not isinstance(value, Mapping) or set(value) != {
            "path", "sha256", "size_bytes"
        }:
            raise SpeechPreparationError("Acquired artifact entry is invalid.")
        artifact_path = Path(str(value["path"])).expanduser().absolute()
        if (
            not re.fullmatch(r"[a-f0-9]{64}", str(value["sha256"]))
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != value["size_bytes"]
            or sha256_file(artifact_path) != value["sha256"]
        ):
            raise SpeechPreparationError(
                f"Acquired artifact {artifact_id} failed hash validation."
            )
    required_assets = {
        "silero_vad": {"silero-vad-wheel", "silero-vad-onnx-op16"},
        "deepfilternet": {
            "deepfilternet-wheel", "deepfilterlib-sdist",
            "deepfilternet3-model-archive", "deepfilternet3-checkpoint",
            "deepfilternet3-config",
        },
        "rnnoise": {"rnnoise-release-source", "rnnoise-library"},
    }
    for candidate_id, asset_ids in required_assets.items():
        candidate = manifest["candidates"].get(candidate_id)
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"revision", "asset_ids"}
            or set(candidate["asset_ids"]) != asset_ids
            or not all(asset_id in artifacts for asset_id in asset_ids)
        ):
            raise SpeechPreparationError(
                f"Acquired candidate {candidate_id} is incomplete."
            )
    expected_revisions = {
        "silero_vad": "6.2.1",
        "deepfilternet": "0.5.6",
        "rnnoise": "v0.2",
    }
    if any(
        manifest["candidates"][candidate_id]["revision"] != revision
        for candidate_id, revision in expected_revisions.items()
    ):
        raise SpeechPreparationError("Acquired candidate revision is invalid.")
    return {
        **manifest,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
    }


def _verified_pyannote_acquisition() -> Optional[dict[str, Any]]:
    path = DEFAULT_PYANNOTE_ACQUISITION_MANIFEST.expanduser().absolute()
    root = DEFAULT_PYANNOTE_ACQUISITION_ROOT.expanduser().absolute()
    if not path.is_file():
        return None
    _private_require(path, root)
    manifest = _private_read(path)
    expected_keys = {
        "schema_version", "repo_id", "revision_sha", "package_distribution",
        "package_version", "authorization_basis", "gated_access_verified",
        "contact_information_sharing_authorized", "snapshot_dir", "artifacts",
        "created_at",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version")
        != "transcribe-audio.pyannote-community-1-acquisition-manifest.v1"
        or manifest.get("repo_id")
        != "pyannote/speaker-diarization-community-1"
        or manifest.get("revision_sha")
        != "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
        or manifest.get("package_distribution") != "pyannote-audio"
        or manifest.get("package_version") != "4.0.4"
        or manifest.get("authorization_basis") != "operator_blanket_2026-07-31"
        or manifest.get("gated_access_verified") is not True
        or manifest.get("contact_information_sharing_authorized") is not True
    ):
        raise SpeechPreparationError("Community-1 acquisition manifest is invalid.")
    snapshot_dir = Path(str(manifest["snapshot_dir"])).expanduser().absolute()
    if snapshot_dir != root / "community-1" or snapshot_dir.is_symlink():
        raise SpeechPreparationError("Community-1 snapshot path is invalid.")
    expected_artifacts = {
        ".gitattributes", "README.md", "config.yaml", "diarization.gif",
        "embedding/README.md", "embedding/pytorch_model.bin", "plda/README.md",
        "plda/plda.npz", "plda/xvec_transform.npz",
        "segmentation/pytorch_model.bin",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise SpeechPreparationError("Community-1 artifact inventory is incomplete.")
    for name, value in artifacts.items():
        artifact_path = snapshot_dir / name
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256", "size_bytes"}
            or Path(str(value["path"])).expanduser().absolute() != artifact_path
            or not re.fullmatch(r"[a-f0-9]{64}", str(value["sha256"]))
        ):
            raise SpeechPreparationError("Community-1 artifact entry is invalid.")
        _private_require(artifact_path, root)
        if (
            artifact_path.stat().st_size != value["size_bytes"]
            or sha256_file(artifact_path) != value["sha256"]
        ):
            raise SpeechPreparationError("Community-1 artifact hash mismatch.")
    return {
        **manifest,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "asset_sha256": canonical_artifact_hash([
            {"name": name, "sha256": artifacts[name]["sha256"]}
            for name in sorted(artifacts)
        ]),
    }


def readiness_matrix() -> dict[str, dict[str, Any]]:
    """Return live code/asset/authorization state without loading a model."""
    matrix: dict[str, dict[str, Any]] = {
        "no_enhancement": {
            "status": "success",
            "reason_code": None,
            "code_revision": "host-no-enhancement-v1",
            "asset_sha256": None,
            "acquisition_manifest_sha256": None,
            "authorization": "not_required",
            "reason": None,
        }
    }
    acquisition = _verified_open_acquisition()
    for method_id in ("silero_vad", "deepfilternet"):
        distribution, expected = EXPECTED_PACKAGES[method_id]
        installed = _package_version(distribution)
        code_present = installed == expected
        candidate = acquisition["candidates"][method_id] if acquisition else None
        asset_sha = canonical_artifact_hash([
            {"artifact_id": asset_id, "sha256": acquisition["artifacts"][asset_id]["sha256"]}
            for asset_id in sorted(candidate["asset_ids"])
        ]) if candidate else None
        ready = code_present and acquisition is not None
        matrix[method_id] = {
            "status": "success" if ready else "blocked",
            "reason_code": None if ready else (
                "asset_hash_unbound" if code_present else "not_acquired"
            ),
            "code_revision": installed,
            "expected_code_revision": expected,
            "asset_sha256": asset_sha,
            "acquisition_manifest_sha256": (
                acquisition["manifest_sha256"] if acquisition else None
            ),
            "authorization": "verified_acquisition" if ready else "open_license_reviewed",
            "authorization_basis": "operator_blanket_2026-07-31" if ready else None,
            "reason": None if ready else (
                "installed code still requires a verified acquisition manifest and asset hashes"
                if code_present
                else f"requires pinned {distribution}=={expected}"
            ),
        }
    rn_candidate = acquisition["candidates"]["rnnoise"] if acquisition else None
    rn_asset_sha = canonical_artifact_hash([
        {"artifact_id": asset_id, "sha256": acquisition["artifacts"][asset_id]["sha256"]}
        for asset_id in sorted(rn_candidate["asset_ids"])
    ]) if rn_candidate else None
    matrix["rnnoise"] = {
        "status": "success" if acquisition else "blocked",
        "reason_code": None if acquisition else "not_acquired",
        "code_revision": "v0.2" if acquisition else None,
        "asset_sha256": rn_asset_sha,
        "acquisition_manifest_sha256": acquisition["manifest_sha256"] if acquisition else None,
        "authorization": "verified_acquisition" if acquisition else "open_license_reviewed",
        "authorization_basis": "operator_blanket_2026-07-31" if acquisition else None,
        "reason": None if acquisition else "requires pinned RNNoise v0.2 library and model hashes",
    }
    pyannote_version = _package_version(EXPECTED_PACKAGES["pyannote_community_1"][0])
    pyannote_acquisition = _verified_pyannote_acquisition()
    pyannote_ready = (
        pyannote_version == EXPECTED_PACKAGES["pyannote_community_1"][1]
        and pyannote_acquisition is not None
    )
    matrix["pyannote_community_1"] = {
        "status": "success" if pyannote_ready else "blocked",
        "reason_code": None if pyannote_ready else "provider_auth_required",
        "code_revision": pyannote_version,
        "expected_code_revision": EXPECTED_PACKAGES["pyannote_community_1"][1],
        "asset_sha256": (
            pyannote_acquisition["asset_sha256"] if pyannote_acquisition else None
        ),
        "acquisition_manifest_sha256": (
            pyannote_acquisition["manifest_sha256"]
            if pyannote_acquisition else None
        ),
        "authorization": (
            "verified_acquisition" if pyannote_ready
            else "operator_blanket_2026-07-31"
        ),
        "authorization_basis": "operator_blanket_2026-07-31",
        "reason": None if pyannote_ready else (
            "Community-1 terms/contact sharing are authorized, but the local "
            "Hugging Face client has no authenticated identity or access token"
        ),
    }
    return matrix


def _validate_open_candidate_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeechPreparationError("Open-candidate acquisition spec is invalid.")
    spec = dict(value)
    if set(spec) != {
        "schema_version",
        "reviewed_at",
        "authorization_scope",
        "excludes",
        "candidates",
        "hash_policy",
    } or spec["schema_version"] != ACQUISITION_SPEC_SCHEMA:
        raise SpeechPreparationError("Open-candidate acquisition spec shape is invalid.")
    if spec["authorization_scope"] != "download_install_build_open_candidates_only":
        raise SpeechPreparationError("Open-candidate authorization scope is invalid.")
    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", str(spec.get("reviewed_at"))):
        raise SpeechPreparationError("Open-candidate review date is invalid.")
    required_exclusions = {
        "pyannote_terms_acceptance",
        "contact_information_sharing",
        "private_audio_read",
        "development_cohort_apply",
        "biometric_enrollment_or_scoring",
    }
    if set(spec.get("excludes") or []) != required_exclusions:
        raise SpeechPreparationError("Open-candidate exclusions are incomplete.")
    candidates = spec.get("candidates")
    if not isinstance(candidates, list) or tuple(
        candidate.get("candidate_id")
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ) != OPEN_CANDIDATE_IDS:
        raise SpeechPreparationError("Open-candidate inventory is incomplete or unordered.")
    for candidate in candidates:
        required = {
            "candidate_id",
            "license_expression",
            "revision",
            "source_commit",
            "install_strategy",
            "python_312_compatibility",
            "artifacts",
            "terms_sources",
        }
        allowed = required | {"signed_tag_object"}
        if set(candidate) - allowed or not required.issubset(candidate):
            raise SpeechPreparationError("Open-candidate entry shape is invalid.")
        if any(
            not isinstance(candidate[field], str) or not candidate[field]
            for field in (
                "license_expression",
                "revision",
                "install_strategy",
                "python_312_compatibility",
            )
        ):
            raise SpeechPreparationError("Open-candidate metadata is incomplete.")
        if not re.fullmatch(r"[a-f0-9]{40}", str(candidate["source_commit"])):
            raise SpeechPreparationError("Open-candidate source commit is invalid.")
        if "signed_tag_object" in candidate and not re.fullmatch(
            r"[a-f0-9]{40}", str(candidate["signed_tag_object"])
        ):
            raise SpeechPreparationError("Open-candidate tag object is invalid.")
        if not isinstance(candidate["terms_sources"], list) or not all(
            isinstance(url, str) and url.startswith("https://")
            for url in candidate["terms_sources"]
        ):
            raise SpeechPreparationError("Open-candidate terms sources are invalid.")
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise SpeechPreparationError("Open-candidate artifacts are missing.")
        for artifact in artifacts:
            required_artifact = {
                "artifact_id",
                "filename",
                "size_bytes",
                "sha256",
                "url",
            }
            if (
                not isinstance(artifact, Mapping)
                or set(artifact) - (required_artifact | {"git_blob_sha1"})
                or not required_artifact.issubset(artifact)
                or not isinstance(artifact["size_bytes"], int)
                or artifact["size_bytes"] <= 0
                or not str(artifact["url"]).startswith("https://")
                or (
                    artifact["sha256"] is not None
                    and not re.fullmatch(r"[a-f0-9]{64}", str(artifact["sha256"]))
                )
                or (
                    "git_blob_sha1" in artifact
                    and not re.fullmatch(
                        r"[a-f0-9]{40}", str(artifact["git_blob_sha1"])
                    )
                )
            ):
                raise SpeechPreparationError("Open-candidate artifact is invalid.")
    artifact_ids = {
        candidate["candidate_id"]: tuple(
            artifact["artifact_id"] for artifact in candidate["artifacts"]
        )
        for candidate in candidates
    }
    if artifact_ids != {
        "silero_vad": ("silero-vad-wheel",),
        "deepfilternet": (
            "deepfilternet-wheel",
            "deepfilterlib-sdist",
            "deepfilternet3-model",
        ),
        "rnnoise": ("rnnoise-release-source",),
    }:
        raise SpeechPreparationError("Open-candidate artifact inventory is invalid.")
    missing_sha_ids = {
        artifact["artifact_id"]
        for candidate in candidates
        for artifact in candidate["artifacts"]
        if artifact["sha256"] is None
    }
    if missing_sha_ids != {"deepfilternet3-model", "rnnoise-release-source"}:
        raise SpeechPreparationError("Open-candidate pre-download hash coverage is invalid.")
    expected_hash_policy = {
        "known_sha256_must_match_before_install": True,
        "missing_official_sha256_must_be_computed_after_download": True,
        "all_downloads_must_be_content_addressed_before_build_or_use": True,
        "hash_mismatch_fails_closed": True,
    }
    if spec.get("hash_policy") != expected_hash_policy:
        raise SpeechPreparationError("Open-candidate hash policy is invalid.")
    return spec


def _acquisition_paths(root: Path, run_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"acquire-open-[a-f0-9]{24}", run_id):
        raise SpeechPreparationError("Open-candidate acquisition run ID is invalid.")
    selected = root.expanduser().absolute()
    run_dir = selected / "acquisition-plans" / run_id
    return {
        "root": selected,
        "run_dir": run_dir,
        "dry_run": run_dir / "dry-run.json",
    }


def _acquisition_host_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "installed_distributions": {
            method_id: _package_version(distribution)
            for method_id, (distribution, _) in EXPECTED_PACKAGES.items()
            if method_id in {"silero_vad", "deepfilternet"}
        },
        "executables": {
            name: str(Path(path).resolve()) if (path := shutil.which(name)) else None
            for name in (
                "cargo",
                "rustc",
                "cmake",
                "autoconf",
                "automake",
                "libtool",
                "pkg-config",
                "deepFilter",
                "rnnoise_demo",
            )
        },
        "deepfilterlib_cp312_wheel_available": False,
    }


def dry_run_open_candidate_acquisition(
    *,
    runtime_root: Optional[Path] = None,
    spec_path: Path = DEFAULT_ACQUISITION_SPEC,
) -> dict[str, Any]:
    """Persist immutable acquisition evidence under the standing operator grant."""
    selected_spec = spec_path.expanduser().absolute()
    if selected_spec.is_symlink() or not selected_spec.is_file():
        raise SpeechPreparationError("Open-candidate acquisition spec is unavailable.")
    try:
        spec = _validate_open_candidate_spec(
            json.loads(selected_spec.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SpeechPreparationError("Open-candidate acquisition spec is unreadable.") from exc
    spec_sha = sha256_file(selected_spec)
    host = _acquisition_host_snapshot()
    identity = {
        "spec_path": str(selected_spec),
        "spec_sha256": spec_sha,
        "host": host,
        "authorization_basis": "operator_blanket_2026-07-31",
    }
    run_id = "acquire-open-" + canonical_artifact_hash(identity)[:24]
    paths = _acquisition_paths(runtime_root or DEFAULT_RUNTIME_ROOT, run_id)
    plan = {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "authorization_basis": "operator_blanket_2026-07-31",
        "spec_path": str(selected_spec),
        "spec_sha256": spec_sha,
        "spec": spec,
        "host": host,
        "runtime_root": str(paths["root"]),
        "will_download": False,
        "will_install": False,
        "will_build": False,
        "will_read_audio": False,
        "will_accept_terms": False,
        "will_share_contact_information": False,
        "will_perform_external_write": False,
        "created_at": utc_now(),
    }
    ensure_private_tree(paths["root"], paths["run_dir"])
    stored = _private_write(paths["dry_run"], plan, volatile_fields=("created_at",))
    plan_sha = sha256_file(paths["dry_run"])
    return {
        **stored,
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": plan_sha,
    }


def replay_open_candidate_acquisition(
    run_id: str,
    *,
    expected_dry_run_sha256: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay standing-grant P2C evidence without executing side effects."""
    paths = _acquisition_paths(runtime_root or DEFAULT_RUNTIME_ROOT, run_id)
    _private_require(paths["dry_run"], paths["root"])
    if (
        not re.fullmatch(r"[a-f0-9]{64}", expected_dry_run_sha256)
        or sha256_file(paths["dry_run"]) != expected_dry_run_sha256
    ):
        raise SpeechPreparationError(
            "Open-candidate acquisition dry-run hash mismatch."
        )
    plan = _private_read(paths["dry_run"])
    if (
        set(plan)
        != {
            "schema_version", "run_id", "status", "reason_code", "spec_path",
            "authorization_basis", "spec_sha256", "spec", "host", "runtime_root", "will_download",
            "will_install", "will_build", "will_read_audio", "will_accept_terms",
            "will_share_contact_information", "will_perform_external_write",
            "created_at",
        }
        or plan["schema_version"] != ACQUISITION_PLAN_SCHEMA
        or plan["run_id"] != run_id
        or plan["status"] != "success"
        or plan["reason_code"] is not None
        or plan["authorization_basis"] != "operator_blanket_2026-07-31"
        or plan["runtime_root"] != str(paths["root"])
        or any(
            plan[field] is not False
            for field in (
                "will_download", "will_install", "will_build", "will_read_audio",
                "will_accept_terms", "will_share_contact_information",
                "will_perform_external_write",
            )
        )
    ):
        raise SpeechPreparationError("Open-candidate acquisition plan is invalid.")
    spec_path = Path(str(plan["spec_path"]))
    if (
        spec_path.is_symlink()
        or not spec_path.is_file()
        or sha256_file(spec_path) != plan["spec_sha256"]
        or _validate_open_candidate_spec(plan["spec"])
        != json.loads(spec_path.read_text(encoding="utf-8"))
    ):
        raise SpeechPreparationError("Open-candidate acquisition spec drifted.")
    identity = {
        "spec_path": str(spec_path),
        "spec_sha256": plan["spec_sha256"],
        "host": plan["host"],
        "authorization_basis": plan["authorization_basis"],
    }
    if run_id != "acquire-open-" + canonical_artifact_hash(identity)[:24]:
        raise SpeechPreparationError("Open-candidate acquisition identity is invalid.")
    plan_sha = expected_dry_run_sha256
    return {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "authorization_basis": "operator_blanket_2026-07-31",
        "spec_sha256": plan["spec_sha256"],
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": plan_sha,
        "will_download": False,
        "will_install": False,
        "will_build": False,
        "will_read_audio": False,
        "will_perform_external_write": False,
        "replayed_at": utc_now(),
    }


def _paths(runtime_root: Path, run_id: str) -> dict[str, Path]:
    if not run_id.startswith("speech-prep-") or len(run_id) != 36:
        raise SpeechPreparationError("Speech-preparation run ID is invalid.")
    root = runtime_root.expanduser().absolute()
    run_dir = root / "runs" / run_id
    return {
        "root": root,
        "run_dir": run_dir,
        "dry_run": run_dir / "dry-run.json",
        "comparison": run_dir / "comparison.json",
        "apply_receipt": run_dir / "apply.json",
        "replay_active": run_dir / "replay-active.json",
        "replay_rolled_back": run_dir / "replay-rolled-back.json",
        "rollback": run_dir / "rollback.json",
    }


def _active_p1_source(run_id: str, runtime_root: Path) -> dict[str, Any]:
    try:
        return resolve_active_derivative(run_id, runtime_root=runtime_root)
    except AudioDerivativeError as exc:
        raise SpeechPreparationError("P2 requires a replayable P1 derivative.") from exc


def _adapter_registry(
    adapters: Optional[Mapping[str, PreparationAdapter]], *, test_mode: bool,
    runtime_root: Path,
) -> dict[str, PreparationAdapter]:
    registry: dict[str, PreparationAdapter] = {
        "no_enhancement": NoEnhancementAdapter()
    }
    if not test_mode:
        acquisition = _verified_open_acquisition()
        if acquisition is not None:
            artifacts = acquisition["artifacts"]
            registry.update({
                "silero_vad": SileroVadAdapter(
                    model_sha256=artifacts["silero-vad-onnx-op16"]["sha256"]
                ),
                "deepfilternet": DeepFilterNetAdapter(
                    model_dir=Path(artifacts["deepfilternet3-config"]["path"]).parent,
                    model_sha256=artifacts["deepfilternet3-checkpoint"]["sha256"],
                    output_root=runtime_root / "outputs",
                ),
                "rnnoise": RNNoiseAdapter(
                    library_path=Path(artifacts["rnnoise-library"]["path"]),
                    library_sha256=artifacts["rnnoise-library"]["sha256"],
                    output_root=runtime_root / "outputs",
                ),
            })
        pyannote_acquisition = _verified_pyannote_acquisition()
        if pyannote_acquisition is not None:
            registry["pyannote_community_1"] = PyannoteCommunity1Adapter(
                snapshot_dir=Path(pyannote_acquisition["snapshot_dir"]),
                revision_sha=pyannote_acquisition["revision_sha"],
                asset_sha256=pyannote_acquisition["asset_sha256"],
            )
    if not adapters:
        return registry
    if not test_mode:
        raise SpeechPreparationError(
            "Adapter overrides are allowed only in explicit synthetic test mode."
        )
    if "no_enhancement" in adapters:
        raise SpeechPreparationError(
            "The host-owned no-enhancement adapter cannot be overridden."
        )
    if set(adapters) - set(METHOD_IDS):
        raise SpeechPreparationError("Adapter registry contains unknown methods.")
    if any(not isinstance(adapter, FakePreparationAdapter) for adapter in adapters.values()):
        raise SpeechPreparationError(
            "Synthetic adapter overrides must use FakePreparationAdapter."
        )
    registry.update(adapters)
    return registry


def _build_plan(
    p1_run_id: str,
    *,
    p1_runtime_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    readiness: Optional[Mapping[str, Mapping[str, Any]]] = None,
    adapters: Optional[Mapping[str, PreparationAdapter]] = None,
    test_mode: bool = False,
    intended_split: str = "development",
    split_access_authority_sha256: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    if intended_split == "development":
        if split_access_authority_sha256 is not None:
            raise SpeechPreparationError(
                "Development preparation cannot bind a later-split authority."
            )
        split_fields: dict[str, Any] = {}
    elif intended_split in {"calibration", "evaluation"}:
        if not re.fullmatch(r"[a-f0-9]{64}", str(split_access_authority_sha256 or "")):
            raise SpeechPreparationError(
                "Later-split preparation requires an exact split authority."
            )
        split_fields = {
            "intended_split": intended_split,
            "split_access_authority_sha256": split_access_authority_sha256,
        }
    else:
        raise SpeechPreparationError(
            "Speech preparation cannot open the requested split."
        )
    p1_root = (p1_runtime_root or P1_DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    source = _active_p1_source(p1_run_id, p1_root)
    if readiness is not None and not test_mode:
        raise SpeechPreparationError(
            "Readiness overrides are allowed only in explicit synthetic test mode."
        )
    live_readiness = {
        method_id: dict(value)
        for method_id, value in (readiness or readiness_matrix()).items()
    }
    if tuple(live_readiness) != METHOD_IDS:
        raise SpeechPreparationError("Readiness must cover every P2 method in order.")
    _validate_readiness(live_readiness, test_mode=test_mode)
    selected_runtime_root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    adapter_registry = _adapter_registry(
        adapters, test_mode=test_mode, runtime_root=selected_runtime_root
    )
    adapter_descriptors = {
        method_id: adapter_registry[method_id].descriptor()
        if method_id in adapter_registry
        else None
        for method_id in METHOD_IDS
    }
    identity = {
        "p1_manifest_sha256": source["manifest_sha256"],
        "readiness": live_readiness,
        "method_ids": list(METHOD_IDS),
        "synthetic_test_mode": test_mode,
        "adapter_descriptors": adapter_descriptors,
        **split_fields,
    }
    run_id = f"speech-prep-{canonical_artifact_hash(identity)[:24]}"
    paths = _paths(selected_runtime_root, run_id)
    plan = {
        "schema_version": DRY_RUN_SCHEMA,
        "status": "success",
        "reason_code": None,
        "lifecycle_state": "planned",
        "run_id": run_id,
        "p1_source": source,
        "readiness": live_readiness,
        "method_ids": list(METHOD_IDS),
        "synthetic_test_mode": test_mode,
        "adapter_descriptors": adapter_descriptors,
        "runtime_root": str(paths["root"]),
        "will_process_audio": any(
            bool(descriptor and descriptor["executes_audio"])
            for descriptor in adapter_descriptors.values()
        ),
        "will_run_models": any(
            bool(descriptor and descriptor["executes_model"])
            for descriptor in adapter_descriptors.values()
        ),
        "will_modify_source": False,
        "will_read_calibration_or_evaluation": False,
        "will_run_biometrics": False,
        "will_perform_external_write": False,
        "created_at": utc_now(),
        **split_fields,
    }
    if intended_split in {"calibration", "evaluation"}:
        plan["will_read_calibration_or_evaluation"] = True
    return plan, paths


def dry_run(
    p1_run_id: str,
    *,
    p1_runtime_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    readiness: Optional[Mapping[str, Mapping[str, Any]]] = None,
    adapters: Optional[Mapping[str, PreparationAdapter]] = None,
    test_mode: bool = False,
    intended_split: str = "development",
    split_access_authority_sha256: Optional[str] = None,
) -> dict[str, Any]:
    plan, paths = _build_plan(
        p1_run_id,
        p1_runtime_root=p1_runtime_root,
        runtime_root=runtime_root,
        readiness=readiness,
        adapters=adapters,
        test_mode=test_mode,
        intended_split=intended_split,
        split_access_authority_sha256=split_access_authority_sha256,
    )
    ensure_private_tree(paths["root"], paths["run_dir"])
    stored = _private_write(
        paths["dry_run"], plan, volatile_fields=("created_at",)
    )
    return {**stored, "dry_run_path": str(paths["dry_run"])}


def _not_run_result(method_id: str, readiness: Mapping[str, Any]) -> dict[str, Any]:
    status = "blocked"
    reason_code = str(readiness.get("reason_code") or "not_run_dependency")
    return {
        "method_id": method_id,
        "status": status,
        "reason_code": reason_code,
        "attempted": False,
        "denominator": 0,
        "readiness": dict(readiness),
        "output_artifact_id": None,
        "output_sha256": None,
        "timestamp_map": None,
        "speech_regions": None,
        "overlap_regions": None,
        "speaker_change_regions": None,
        "resource_usage": None,
        "warnings": [],
        "abstention_reasons": [str(readiness.get("reason") or reason_code)],
    }


def _forbidden_result_keys(value: Any) -> set[str]:
    forbidden_families = (
        "waveform",
        "embedding",
        "audio_bytes",
        "tensor",
        "credential",
        "token",
        "person_name",
        "transcript",
    )
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if any(family in normalized for family in forbidden_families):
                found.add(str(key))
            found.update(_forbidden_result_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_result_keys(child))
    return found


def _validate_readiness(
    matrix: Mapping[str, Mapping[str, Any]], *, test_mode: bool
) -> None:
    forbidden = sorted(_forbidden_result_keys(matrix))
    if forbidden:
        raise SpeechPreparationError(
            "P2 readiness contains forbidden private/provider fields: "
            + ", ".join(forbidden)
        )
    for method_id in METHOD_IDS:
        value = matrix.get(method_id)
        if not isinstance(value, Mapping):
            raise SpeechPreparationError("P2 readiness entry must be an object.")
        status = value.get("status")
        reason_code = value.get("reason_code")
        if status not in {"success", "failure", "blocked"}:
            raise SpeechPreparationError("P2 readiness status is invalid.")
        if status == "success" and reason_code is not None:
            raise SpeechPreparationError(
                "Successful P2 readiness must not carry a reason code."
            )
        if status != "success" and not isinstance(reason_code, str):
            raise SpeechPreparationError(
                "Non-success P2 readiness requires a reason code."
            )
        if method_id == "no_enhancement":
            if status != "success":
                raise SpeechPreparationError("No-enhancement must remain ready.")
            continue
        if status == "success":
            if test_mode:
                if value.get("authorization") != "synthetic_test_only":
                    raise SpeechPreparationError(
                        "Synthetic readiness requires synthetic-test authorization."
                    )
            elif value.get("authorization") != "verified_acquisition":
                raise SpeechPreparationError(
                    "Real readiness requires a verified acquisition."
                )
            for field_name in ("asset_sha256", "acquisition_manifest_sha256"):
                if not re.fullmatch(r"[a-f0-9]{64}", str(value.get(field_name) or "")):
                    raise SpeechPreparationError(
                        f"Successful P2 readiness requires {field_name}."
                    )


def _validate_regions(regions: Any, duration: float, field_name: str) -> None:
    if regions is None:
        return
    if not isinstance(regions, list):
        raise SpeechPreparationError(f"{field_name} must be null or a list.")
    prior_end = 0.0
    for region in regions:
        if not isinstance(region, Mapping):
            raise SpeechPreparationError(f"{field_name} entries must be objects.")
        try:
            start = float(region["start_seconds"])
            end = float(region["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SpeechPreparationError(f"{field_name} entry is incomplete.") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < prior_end
            or start < 0
            or end <= start
            or end > duration
        ):
            raise SpeechPreparationError(
                f"{field_name} must be monotonic, non-overlapping, and bounded."
            )
        prior_end = end


def _validate_full_timestamp_map(value: Any, duration: float) -> None:
    if not isinstance(value, list) or len(value) != 1:
        raise SpeechPreparationError(
            "Enhanced output requires one full-coverage timestamp map."
        )
    mapping = value[0]
    if not isinstance(mapping, Mapping):
        raise SpeechPreparationError("Enhanced timestamp map must be an object.")
    try:
        bounds = [
            float(mapping["source_start_seconds"]),
            float(mapping["source_end_seconds"]),
            float(mapping["output_start_seconds"]),
            float(mapping["output_end_seconds"]),
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise SpeechPreparationError("Enhanced timestamp map is incomplete.") from exc
    if (
        not all(math.isfinite(bound) for bound in bounds)
        or bounds[0] != 0.0
        or bounds[1] != duration
        or bounds[2] != 0.0
        or bounds[3] != duration
    ):
        raise SpeechPreparationError(
            "Enhanced timestamp map must cover the complete P1 timebase."
        )


def _validate_method_result(
    result: Mapping[str, Any], source: Mapping[str, Any],
    *, runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    forbidden = sorted(_forbidden_result_keys(result))
    if forbidden:
        raise SpeechPreparationError(
            "P2 result contains forbidden private/provider fields: "
            + ", ".join(forbidden)
        )
    status = result.get("status")
    reason_code = result.get("reason_code")
    if status not in {"success", "failure", "blocked"}:
        raise SpeechPreparationError("P2 result status must be success, failure, or blocked.")
    if status == "success" and reason_code is not None:
        raise SpeechPreparationError("Successful P2 result must not carry a reason code.")
    if status != "success" and not isinstance(reason_code, str):
        raise SpeechPreparationError("Non-success P2 result requires a reason code.")
    attempted = result.get("attempted")
    denominator = result.get("denominator")
    if attempted is not (denominator == 1) or denominator not in {0, 1}:
        raise SpeechPreparationError("P2 attempted flag and denominator are inconsistent.")
    duration = float(source["derived_audio"]["output_duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise SpeechPreparationError("P2 source duration is invalid.")
    for field_name in (
        "speech_regions",
        "overlap_regions",
        "speaker_change_regions",
    ):
        _validate_regions(result.get(field_name), duration, field_name)
    if result.get("method_id") == "no_enhancement" and status == "success":
        if (
            result.get("output_sha256") != source["artifact_sha256"]
            or result.get("timestamp_map")
            != source["derived_audio"]["timestamp_map"]
        ):
            raise SpeechPreparationError(
                "No-enhancement must reuse the replay-verified P1 artifact and map."
            )
    method_id = result.get("method_id")
    if status == "success" and method_id in {"deepfilternet", "rnnoise"}:
        if (
            not re.fullmatch(r"[a-f0-9]{64}", str(result.get("output_sha256") or ""))
            or not isinstance(result.get("output_path"), str)
            or not result.get("output_path")
        ):
            raise SpeechPreparationError(
                "Successful enhancement requires a hashed output reference."
            )
        _validate_full_timestamp_map(result.get("timestamp_map"), duration)
        readiness = result.get("readiness")
        if (
            isinstance(readiness, Mapping)
            and readiness.get("authorization") == "verified_acquisition"
        ):
            if runtime_root is None:
                raise SpeechPreparationError(
                    "Enhanced output validation requires its private runtime root."
                )
            output_path = Path(str(result["output_path"])).expanduser().absolute()
            _private_require(output_path, runtime_root.expanduser().absolute())
            output_sha = str(result["output_sha256"])
            if (
                output_path.stem != output_sha
                or sha256_file(output_path) != output_sha
            ):
                raise SpeechPreparationError(
                    "Enhanced output content-address or SHA-256 binding mismatch."
                )
    if status == "success" and method_id == "silero_vad":
        if not isinstance(result.get("speech_regions"), list) or not result.get(
            "speech_regions"
        ):
            raise SpeechPreparationError(
                "Successful Silero VAD requires measured speech regions."
            )
    if status == "success" and method_id == "pyannote_community_1":
        if (
            not isinstance(result.get("speech_regions"), list)
            or not result.get("speech_regions")
            or not isinstance(result.get("overlap_regions"), list)
            or not isinstance(result.get("speaker_change_regions"), list)
        ):
            raise SpeechPreparationError(
                "Successful diarization requires speech, overlap, and change evidence."
            )
    return dict(result)


def _pending_downstream_measurements(
    all_methods_succeeded: bool,
) -> dict[str, dict[str, Any]]:
    preparation_reason = (
        "not_run_downstream_measurements"
        if all_methods_succeeded
        else "not_run_dependency_real_methods"
    )
    return {
        "transcription": {
            "status": "blocked",
            "reason_code": preparation_reason,
            "denominator": 0,
        },
        "diarization": {
            "status": "blocked",
            "reason_code": preparation_reason,
            "denominator": 0,
        },
        "verification": {
            "status": "blocked",
            "reason_code": "not_run_dependency_p3_p4",
            "denominator": 0,
        },
    }


def apply_comparison(
    p1_run_id: str,
    *,
    approval_token: Optional[str] = None,
    p1_runtime_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    readiness: Optional[Mapping[str, Mapping[str, Any]]] = None,
    adapters: Optional[Mapping[str, PreparationAdapter]] = None,
    test_mode: bool = False,
    intended_split: str = "development",
    split_access_authority_sha256: Optional[str] = None,
) -> dict[str, Any]:
    plan, paths = _build_plan(
        p1_run_id,
        p1_runtime_root=p1_runtime_root,
        runtime_root=runtime_root,
        readiness=readiness,
        adapters=adapters,
        test_mode=test_mode,
        intended_split=intended_split,
        split_access_authority_sha256=split_access_authority_sha256,
    )
    del approval_token
    ensure_private_tree(paths["root"], paths["run_dir"])
    if not paths["dry_run"].is_file():
        raise SpeechPreparationError("Apply requires the matching P2 dry run.")
    _private_require(paths["dry_run"], paths["root"])
    persisted = _private_read(paths["dry_run"])
    left = dict(persisted)
    right = dict(plan)
    left.pop("created_at", None)
    right.pop("created_at", None)
    if left != right:
        raise SpeechPreparationError("P1 source or P2 readiness changed after dry run.")
    if paths["rollback"].exists():
        raise SpeechPreparationError("A rolled-back P2 run cannot be reactivated.")
    if paths["comparison"].exists():
        replay = replay_comparison(plan["run_id"], runtime_root=paths["root"])
        return {
            **_private_read(paths["apply_receipt"]),
            "comparison": _private_read(paths["comparison"]),
            "comparison_path": str(paths["comparison"]),
            "idempotent_replay": True,
            "replay": replay,
        }

    adapter_registry = _adapter_registry(
        adapters, test_mode=test_mode, runtime_root=paths["root"]
    )
    method_results: list[dict[str, Any]] = []
    for method_id in METHOD_IDS:
        method_readiness = plan["readiness"][method_id]
        adapter = adapter_registry.get(method_id)
        if adapter is None or method_readiness.get("status") != "success":
            method_results.append(_not_run_result(method_id, method_readiness))
            continue
        if adapter.method_id != method_id:
            raise SpeechPreparationError("Adapter method binding mismatch.")
        prepared = adapter.prepare(plan["p1_source"])
        method_results.append(
            _validate_method_result(
                {
                    **prepared,
                    "method_id": method_id,
                    "attempted": True,
                    "denominator": 1,
                    "readiness": dict(method_readiness),
                    "resource_usage": None,
                },
                plan["p1_source"],
                runtime_root=paths["root"],
            )
        )
    method_results = [
        _validate_method_result(
            result, plan["p1_source"], runtime_root=paths["root"]
        )
        for result in method_results
    ]
    all_methods_succeeded = all(
        result["status"] == "success" for result in method_results
    )
    created_at = utc_now()
    comparison = {
        "schema_version": COMPARISON_SCHEMA,
        "run_id": plan["run_id"],
        "status": "success" if all_methods_succeeded else "blocked",
        "reason_code": (
            None if all_methods_succeeded else "required_real_comparisons_not_run"
        ),
        "lifecycle_state": "active",
        "p1_source": plan["p1_source"],
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": sha256_file(paths["dry_run"]),
        "method_results": method_results,
        "denominators": {
            "methods": len(METHOD_IDS),
            "attempted": sum(bool(result["attempted"]) for result in method_results),
            "success": sum(result["status"] == "success" for result in method_results),
            "failure": sum(result["status"] == "failure" for result in method_results),
            "blocked": sum(result["status"] == "blocked" for result in method_results),
        },
        "downstream_measurements": _pending_downstream_measurements(
            all_methods_succeeded
        ),
        "selection_decision": {
            "status": "blocked",
            "reason_code": (
                "not_run_downstream_measurements"
                if all_methods_succeeded
                else "required_real_comparisons_not_run"
            ),
            "selected_method": None,
        },
        "privacy_mode": "private_operation",
        "eligible_for_identity": False,
        "will_modify_source": False,
        "will_read_calibration_or_evaluation": False,
        "will_run_biometrics": False,
        "will_perform_external_write": False,
        "created_at": created_at,
    }
    if intended_split in {"calibration", "evaluation"}:
        comparison.update(
            {
                "intended_split": intended_split,
                "split_access_authority_sha256": split_access_authority_sha256,
            }
        )
        comparison["will_read_calibration_or_evaluation"] = True
    stored = _private_write(paths["comparison"], comparison)
    apply_receipt = {
        "schema_version": APPLY_RECEIPT_SCHEMA,
        "run_id": plan["run_id"],
        "status": "success",
        "reason_code": None,
        "lifecycle_state": "applied",
        "comparison_path": str(paths["comparison"]),
        "comparison_sha256": sha256_file(paths["comparison"]),
        "p1_manifest_sha256": plan["p1_source"]["manifest_sha256"],
        "will_perform_external_write": False,
        "applied_at": created_at,
    }
    if intended_split in {"calibration", "evaluation"}:
        apply_receipt.update(
            {
                "intended_split": intended_split,
                "split_access_authority_sha256": split_access_authority_sha256,
                "did_read_calibration_or_evaluation": True,
            }
        )
    stored_apply = _private_write(paths["apply_receipt"], apply_receipt)
    return {
        **stored_apply,
        "comparison": stored,
        "comparison_path": str(paths["comparison"]),
        "idempotent_replay": False,
    }


def replay_comparison(
    run_id: str, *, runtime_root: Optional[Path] = None
) -> dict[str, Any]:
    paths = _paths((runtime_root or DEFAULT_RUNTIME_ROOT).expanduser(), run_id)
    for path in (paths["dry_run"], paths["comparison"], paths["apply_receipt"]):
        _private_require(path, paths["root"])
    comparison = _private_read(paths["comparison"])
    apply_receipt = _private_read(paths["apply_receipt"])
    if comparison.get("run_id") != run_id:
        raise SpeechPreparationError("P2 comparison run binding mismatch.")
    if sha256_file(paths["dry_run"]) != comparison.get("dry_run_sha256"):
        raise SpeechPreparationError("P2 dry-run hash mismatch.")
    expected_apply_keys = {
        "schema_version",
        "run_id",
        "status",
        "reason_code",
        "lifecycle_state",
        "comparison_path",
        "comparison_sha256",
        "p1_manifest_sha256",
        "will_perform_external_write",
        "applied_at",
    }
    later_split = comparison.get("intended_split")
    later_split_mode = later_split in {"calibration", "evaluation"}
    if later_split_mode:
        expected_apply_keys.update(
            {
                "intended_split",
                "split_access_authority_sha256",
                "did_read_calibration_or_evaluation",
            }
        )
    source = comparison.get("p1_source") or {}
    if (
        set(apply_receipt) != expected_apply_keys
        or apply_receipt.get("schema_version") != APPLY_RECEIPT_SCHEMA
        or apply_receipt.get("run_id") != run_id
        or apply_receipt.get("status") != "success"
        or apply_receipt.get("reason_code") is not None
        or apply_receipt.get("lifecycle_state") != "applied"
        or apply_receipt.get("comparison_path") != str(paths["comparison"])
        or apply_receipt.get("comparison_sha256")
        != sha256_file(paths["comparison"])
        or apply_receipt.get("p1_manifest_sha256")
        != source.get("manifest_sha256")
        or apply_receipt.get("will_perform_external_write") is not False
        or apply_receipt.get("applied_at") != comparison.get("created_at")
        or (
            later_split_mode
            and (
                not re.fullmatch(
                    r"[a-f0-9]{64}",
                    str(comparison.get("split_access_authority_sha256") or ""),
                )
                or comparison.get("will_read_calibration_or_evaluation") is not True
                or apply_receipt.get("intended_split") != later_split
                or apply_receipt.get("split_access_authority_sha256")
                != comparison.get("split_access_authority_sha256")
                or apply_receipt.get("did_read_calibration_or_evaluation") is not True
            )
        )
        or (
            not later_split_mode
            and (
                comparison.get("intended_split") is not None
                or comparison.get("split_access_authority_sha256") is not None
                or comparison.get("will_read_calibration_or_evaluation") is not False
            )
        )
    ):
        raise SpeechPreparationError("P2 apply receipt binding mismatch.")
    current = _active_p1_source(
        str(source.get("run_id") or ""), Path(str(source.get("runtime_root") or ""))
    )
    if current != source:
        raise SpeechPreparationError("P1 evidence changed during P2 replay.")
    method_results = comparison.get("method_results")
    if not isinstance(method_results, list) or [
        result.get("method_id") for result in method_results if isinstance(result, Mapping)
    ] != list(METHOD_IDS):
        raise SpeechPreparationError("P2 comparison method coverage is invalid.")
    validated_results = [
        _validate_method_result(result, source, runtime_root=paths["root"])
        for result in method_results
    ]
    denominators = comparison.get("denominators") or {}
    if (
        denominators.get("methods") != len(METHOD_IDS)
        or denominators.get("attempted")
        != sum(bool(result["attempted"]) for result in validated_results)
        or denominators.get("success")
        != sum(result["status"] == "success" for result in validated_results)
        or denominators.get("failure")
        != sum(result["status"] == "failure" for result in validated_results)
        or denominators.get("blocked")
        != sum(result["status"] == "blocked" for result in validated_results)
    ):
        raise SpeechPreparationError("P2 comparison denominators are inconsistent.")
    active = not paths["rollback"].exists()
    if not active:
        _private_require(paths["rollback"], paths["root"])
        rollback = _private_read(paths["rollback"])
        if (
            rollback.get("run_id") != run_id
            or rollback.get("comparison_sha256") != sha256_file(paths["comparison"])
            or rollback.get("eligible_for_use") is not False
        ):
            raise SpeechPreparationError("P2 rollback binding mismatch.")
    replay_path = paths["replay_active"] if active else paths["replay_rolled_back"]
    receipt = {
        "schema_version": REPLAY_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "lifecycle_state": "verified_active" if active else "verified_rolled_back",
        "active": active,
        "comparison_path": str(paths["comparison"]),
        "comparison_sha256": sha256_file(paths["comparison"]),
        "p1_manifest_sha256": source["manifest_sha256"],
        "will_perform_external_write": False,
        "replayed_at": utc_now(),
    }
    if later_split_mode:
        receipt.update(
            {
                "intended_split": later_split,
                "split_access_authority_sha256": comparison[
                    "split_access_authority_sha256"
                ],
                "did_read_calibration_or_evaluation": True,
            }
        )
    stored = _private_write(
        replay_path, receipt, volatile_fields=("replayed_at",)
    )
    return {**stored, "replay_path": str(replay_path)}


def resolve_comparison_lineage_receipt(
    run_id: str,
    *,
    method_id: str,
    replay_receipt_sha256: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate prior P2 replay metadata without reopening P1/audio bytes."""
    if method_id not in METHOD_IDS or not re.fullmatch(
        r"[a-f0-9]{64}", replay_receipt_sha256
    ):
        raise SpeechPreparationError("P2 lineage request is invalid.")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    paths = _paths(root, run_id)
    for path in (
        paths["comparison"],
        paths["apply_receipt"],
        paths["replay_active"],
    ):
        _private_require(path, paths["root"])
    if paths["rollback"].exists():
        raise SpeechPreparationError("A rolled-back P2 run is not lineage eligible.")
    if sha256_file(paths["replay_active"]) != replay_receipt_sha256:
        raise SpeechPreparationError("P2 lineage replay receipt hash mismatch.")
    replay = _private_read(paths["replay_active"])
    comparison = _private_read(paths["comparison"])
    apply_receipt = _private_read(paths["apply_receipt"])
    comparison_sha = sha256_file(paths["comparison"])
    if (
        replay.get("schema_version") != REPLAY_SCHEMA
        or replay.get("run_id") != run_id
        or replay.get("status") != "success"
        or replay.get("lifecycle_state") != "verified_active"
        or replay.get("active") is not True
        or replay.get("comparison_path") != str(paths["comparison"])
        or replay.get("comparison_sha256") != comparison_sha
        or replay.get("will_perform_external_write") is not False
        or apply_receipt.get("schema_version") != APPLY_RECEIPT_SCHEMA
        or apply_receipt.get("run_id") != run_id
        or apply_receipt.get("status") != "success"
        or apply_receipt.get("lifecycle_state") != "applied"
        or apply_receipt.get("comparison_sha256") != comparison_sha
        or comparison.get("run_id") != run_id
    ):
        raise SpeechPreparationError("P2 lineage receipt binding mismatch.")
    result = next(
        (
            item
            for item in comparison.get("method_results") or []
            if isinstance(item, Mapping) and item.get("method_id") == method_id
        ),
        None,
    )
    if result is None or result.get("status") != "success":
        raise SpeechPreparationError("P2 lineage method is not successful.")
    source = comparison.get("p1_source") or {}
    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "authority": "p2_speech_preparation_replay",
        "run_id": run_id,
        "runtime_root": str(paths["root"]),
        "method_id": method_id,
        "replay_receipt_path": str(paths["replay_active"]),
        "replay_receipt_sha256": replay_receipt_sha256,
        "comparison_path": str(paths["comparison"]),
        "comparison_sha256": comparison_sha,
        "method_result_sha256": canonical_artifact_hash(result),
        "source_blob_id": source.get("source_blob_id"),
        "source_sha256": source.get("source_sha256"),
        "source_duration_seconds": (source.get("derived_audio") or {}).get(
            "source_duration_seconds"
        ),
        "audio_quality_sha256": source.get("audio_quality_sha256"),
        "validation_status": "verified_active_metadata_receipt",
        "will_read_audio": False,
    }
    if comparison.get("intended_split") in {"calibration", "evaluation"}:
        lineage.update(
            {
                "intended_split": comparison["intended_split"],
                "split_access_authority_sha256": comparison[
                    "split_access_authority_sha256"
                ],
            }
        )
    return lineage


def rollback_comparison(
    run_id: str,
    *,
    approval_token: Optional[str] = None,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    del approval_token
    paths = _paths((runtime_root or DEFAULT_RUNTIME_ROOT).expanduser(), run_id)
    replay_comparison(run_id, runtime_root=paths["root"])
    receipt = {
        "schema_version": ROLLBACK_SCHEMA,
        "run_id": run_id,
        "status": "success",
        "reason_code": None,
        "lifecycle_state": "rolled_back",
        "comparison_path": str(paths["comparison"]),
        "comparison_sha256": sha256_file(paths["comparison"]),
        "eligible_for_use": False,
        "evidence_retained": True,
        "p1_source_retained": True,
        "will_perform_external_write": False,
        "rolled_back_at": utc_now(),
    }
    stored = _private_write(
        paths["rollback"], receipt, volatile_fields=("rolled_back_at",)
    )
    return {**stored, "rollback_path": str(paths["rollback"])}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan 0037 P2 speech preparation")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--p1-runtime-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("p1_run_id")
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("p1_run_id")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("run_id")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("run_id")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "dry-run":
        result = dry_run(
            args.p1_run_id,
            p1_runtime_root=args.p1_runtime_root,
            runtime_root=args.runtime_root,
        )
    elif args.command == "apply":
        result = apply_comparison(
            args.p1_run_id,
            p1_runtime_root=args.p1_runtime_root,
            runtime_root=args.runtime_root,
        )
    elif args.command == "replay":
        result = replay_comparison(args.run_id, runtime_root=args.runtime_root)
    else:
        result = rollback_comparison(
            args.run_id,
            runtime_root=args.runtime_root,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
