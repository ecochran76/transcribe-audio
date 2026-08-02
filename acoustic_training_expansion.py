"""Private exact-five intake authority for additional acoustic training data."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.acoustic-training-intake-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.acoustic-training-intake-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.acoustic-training-intake-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.acoustic-training-intake-replay.v1"
DEFAULT_SOURCE_ROOT = Path("~/Documents/Sound Recordings")
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/training-expansion"
)
DEFAULT_CORPUS_MANIFESTS = (
    Path(
        "~/.local/state/transcribe-audio/plan-0037/corpora/"
        "acoustic-corpus-1f93d1405f82676420571e1b/manifest.json"
    ),
    Path(
        "~/.local/state/transcribe-audio/plan-0037/corpora/"
        "acoustic-corpus-e81ea546dea777fa40e9d1c9/manifest.json"
    ),
    Path(
        "~/.local/state/transcribe-audio/plan-0037/corpora/"
        "acoustic-corpus-4a2b13e7bdc201f694af2f43/manifest.json"
    ),
)
EXPECTED_CORPORA = {
    "acoustic-corpus-1f93d1405f82676420571e1b": {
        "content_sha256": (
            "1f93d1405f82676420571e1b88b892ab5d7d8dc4a8b14232d7c77685e6aae5ec"
        ),
        "manifest_sha256": (
            "73f0e04aab0274ddfeaa7f6b1567ecb135eebc0a0d6e5818cb3bd2ee5535dabf"
        ),
    },
    "acoustic-corpus-e81ea546dea777fa40e9d1c9": {
        "content_sha256": (
            "e81ea546dea777fa40e9d1c9ce3b2bce28d8660e8884ce0b92725bd34074d627"
        ),
        "manifest_sha256": (
            "bec631d8ad277a41801a359fdfbe79200fc85b1ca074cca63f052dad9a4e939a"
        ),
    },
    "acoustic-corpus-4a2b13e7bdc201f694af2f43": {
        "content_sha256": (
            "4a2b13e7bdc201f694af2f43d4ab845749eeeb3ea06c7a97a40164cab40b83fe"
        ),
        "manifest_sha256": (
            "4b77479d25d7b248cc62d500ed84c1604f105848da25ecef53661c5d9ea05a30"
        ),
    },
}
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")


class TrainingExpansionError(ValueError):
    """Raised when additional training intake cannot remain exact and private."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingExpansionError("Training intake JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise TrainingExpansionError("Training intake JSON must be an object.")
    return value


def _regular_input(path: Path, root: Path) -> Path:
    selected = path.expanduser().absolute()
    selected_root = root.expanduser().absolute()
    if selected_root.is_symlink() or not selected_root.is_dir():
        raise TrainingExpansionError("Training source root is invalid.")
    try:
        relative = selected.relative_to(selected_root)
    except ValueError as exc:
        raise TrainingExpansionError("Training input escapes the selected root.") from exc
    cursor = selected_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise TrainingExpansionError("Training input must not use symlinks.")
    try:
        resolved_root = selected_root.resolve(strict=True)
        resolved = selected.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise TrainingExpansionError("Training input escapes the selected root.") from exc
    if not resolved.is_file():
        raise TrainingExpansionError("Training input must be a regular non-symlink file.")
    return resolved


def _recorded_path(value: Any, root: Path, *, description: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise TrainingExpansionError(f"Training transcript {description} is invalid.")
    selected = path.absolute()
    selected_root = root.expanduser().resolve(strict=True)
    try:
        selected.relative_to(selected_root)
    except ValueError as exc:
        raise TrainingExpansionError(
            f"Training transcript {description} escapes the selected root."
        ) from exc
    return selected


def _corpus_authority(
    manifest_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], set[str]]:
    if len(manifest_paths) != len(EXPECTED_CORPORA):
        raise TrainingExpansionError("All frozen prior corpora are required.")
    authorities = []
    prior_sources: set[str] = set()
    seen_ids: set[str] = set()
    for raw_path in manifest_paths:
        path = raw_path.expanduser().absolute()
        if path.is_symlink() or not path.is_file():
            raise TrainingExpansionError("Frozen corpus manifest is unavailable.")
        manifest = _read_object(path)
        corpus_id = str(manifest.get("corpus_id") or "")
        expected = EXPECTED_CORPORA.get(corpus_id)
        recordings = manifest.get("recordings")
        if (
            expected is None
            or corpus_id in seen_ids
            or manifest.get("content_sha256") != expected["content_sha256"]
            or sha256_file(path) != expected["manifest_sha256"]
            or path.name != "manifest.json"
            or path.parent.name != corpus_id
            or not isinstance(recordings, list)
        ):
            raise TrainingExpansionError("Frozen corpus authority drifted.")
        seen_ids.add(corpus_id)
        for recording in recordings:
            source = recording.get("source_blob") if isinstance(recording, Mapping) else None
            digest = str(source.get("sha256") or "") if isinstance(source, Mapping) else ""
            if not SHA256_RE.fullmatch(digest):
                raise TrainingExpansionError("Frozen corpus source binding is invalid.")
            prior_sources.add(digest)
        authorities.append({
            "corpus_id": corpus_id,
            "content_sha256": manifest["content_sha256"],
            "manifest_sha256": expected["manifest_sha256"],
            "recording_count": len(recordings),
        })
    if seen_ids != set(EXPECTED_CORPORA):
        raise TrainingExpansionError("Frozen corpus authority set is incomplete.")
    return sorted(authorities, key=lambda item: item["corpus_id"]), prior_sources


def _conversation_input(
    value: Mapping[str, Any], *, source_root: Path, prior_sources: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(value) != {"source_path", "transcript_path"}:
        raise TrainingExpansionError("Training conversation input shape is invalid.")
    source_path = _regular_input(Path(str(value["source_path"])), source_root)
    transcript_path = _regular_input(Path(str(value["transcript_path"])), source_root)
    if source_path.suffix.lower() not in {".m4a", ".mp3", ".wav", ".mp4", ".webm"}:
        raise TrainingExpansionError("Training source format is unsupported.")
    if not transcript_path.name.endswith("Transcript.transcript.json"):
        raise TrainingExpansionError("Training transcript path is not canonical.")
    source_sha = sha256_file(source_path)
    transcript_sha = sha256_file(transcript_path)
    if source_sha in prior_sources:
        raise TrainingExpansionError("Training source overlaps a frozen corpus.")
    transcript = _read_object(transcript_path)
    utterances = transcript.get("utterances")
    duration = transcript.get("duration_seconds")
    working_path = _recorded_path(
        transcript.get("working_media_path"), source_root,
        description="working-media path",
    )
    source_media_path = _recorded_path(
        transcript.get("source_media_path"), source_root,
        description="source-media path",
    )
    output_paths = transcript.get("output_paths")
    artifact_path = _recorded_path(
        output_paths.get("artifact") if isinstance(output_paths, Mapping) else None,
        source_root,
        description="artifact path",
    )
    canonical_transcript_path = source_path.with_name(
        f"{source_path.stem} Transcript.transcript.json"
    )
    if (
        transcript.get("schema_version") != 1
        or working_path != source_path
        or source_media_path.parent != source_path.parent
        or artifact_path != transcript_path
        or transcript_path != canonical_transcript_path
        or not isinstance(duration, (int, float))
        or not 1.0 <= float(duration)
        or not isinstance(utterances, list)
        or not utterances
        or transcript.get("utterance_count") != len(utterances)
    ):
        raise TrainingExpansionError("Training transcript/source binding is invalid.")
    label_milliseconds: dict[str, int] = {}
    label_order: list[str] = []
    previous_start = -1
    for utterance in utterances:
        if not isinstance(utterance, Mapping):
            raise TrainingExpansionError("Training utterance is invalid.")
        label = str(utterance.get("speaker") or "")
        start = utterance.get("start")
        end = utterance.get("end")
        if (
            not LABEL_RE.fullmatch(label)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or start < previous_start
            or end > int(float(duration) * 1000) + 1000
        ):
            raise TrainingExpansionError("Training utterance timing is invalid.")
        previous_start = start
        if label not in label_milliseconds:
            label_order.append(label)
        label_milliseconds[label] = label_milliseconds.get(label, 0) + end - start
    labels = [
        {
            "speaker_label_id": "diarized-label-" + hashlib.sha256(
                f"{source_sha}\0{ordinal}".encode("utf-8")
            ).hexdigest()[:20],
            "utterance_milliseconds": label_milliseconds[label],
        }
        for ordinal, label in enumerate(label_order)
    ]
    safe_core = {
        "source_sha256": source_sha,
        "source_bytes": source_path.stat().st_size,
        "transcript_sha256": transcript_sha,
        "transcript_bytes": transcript_path.stat().st_size,
        "duration_seconds": float(duration),
        "utterance_count": len(utterances),
        "labels": labels,
    }
    input_id = "training-conversation-" + _canonical_hash(safe_core)[:24]
    return {
        "conversation_input_id": input_id,
        **safe_core,
    }, {
        "conversation_input_id": input_id,
        "source_path": str(source_path),
        "transcript_path": str(transcript_path),
        "diarized_label_bindings": [
            {
                "speaker_label": label,
                "speaker_label_id": "diarized-label-" + hashlib.sha256(
                    f"{source_sha}\0{ordinal}".encode("utf-8")
                ).hexdigest()[:20],
            }
            for ordinal, label in enumerate(label_order)
        ],
    }


def _evaluate(
    conversations: Sequence[Mapping[str, Any]], *, source_root: Path,
    corpus_manifest_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= len(conversations) <= 5:
        raise TrainingExpansionError("Training intake requires between one and five conversations.")
    corpus_authorities, prior_sources = _corpus_authority(corpus_manifest_paths)
    safe_units = []
    private_units = []
    for value in conversations:
        if not isinstance(value, Mapping):
            raise TrainingExpansionError("Training conversation must be an object.")
        safe, private = _conversation_input(
            value, source_root=source_root, prior_sources=prior_sources
        )
        safe_units.append(safe)
        private_units.append(private)
    source_hashes = [unit["source_sha256"] for unit in safe_units]
    transcript_hashes = [unit["transcript_sha256"] for unit in safe_units]
    input_ids = [unit["conversation_input_id"] for unit in safe_units]
    if (
        len(set(source_hashes)) != len(source_hashes)
        or len(set(transcript_hashes)) != len(transcript_hashes)
        or len(set(input_ids)) != len(input_ids)
    ):
        raise TrainingExpansionError("Training intake contains duplicate conversations.")
    ordered = sorted(
        zip(safe_units, private_units),
        key=lambda pair: pair[0]["source_sha256"],
    )
    safe_units = [pair[0] for pair in ordered]
    private_units = [pair[1] for pair in ordered]
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "reason_codes": [],
        "conversation_count": len(safe_units),
        "maximum_conversation_count": 5,
        "prior_corpus_authorities": corpus_authorities,
        "prior_corpus_overlap_count": 0,
        "conversations": safe_units,
        "speaker_label_count": sum(len(unit["labels"]) for unit in safe_units),
        "total_duration_seconds": sum(unit["duration_seconds"] for unit in safe_units),
        "identity_confirmation_required": True,
        "training_sufficiency_policy": {
            "minimum_confirmed_people": 2,
            "minimum_independent_sessions_per_person": 2,
            "minimum_eligible_windows_per_person": 6,
            "maximum_windows_per_person_per_conversation": 3,
        },
        "will_read_audio": False,
        "will_process_audio": False,
        "will_run_biometrics": False,
        "will_register_references": False,
        "will_infer_identity": False,
        "will_perform_external_write": False,
        "contains_paths": False,
        "contains_names_or_emails": False,
        "contains_transcript_text": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
    }
    content_sha256 = _canonical_hash(core)
    preview = {
        **core,
        "preview_id": f"training-intake-{content_sha256[:24]}",
        "content_sha256": content_sha256,
    }
    return preview, {"conversations": private_units}


def preview_training_intake(
    conversations: Sequence[Mapping[str, Any]], *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
) -> dict[str, Any]:
    """Return a deterministic path-free preview for up to five conversations."""
    preview, _ = _evaluate(
        conversations, source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    return preview


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise TrainingExpansionError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    status = _git(["status", "--porcelain=v1", "--untracked-files=normal"])
    upstream = _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    if status or upstream.split() != ["0", "0"]:
        raise TrainingExpansionError("Repository must be clean and upstream-even.")
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "commit", "module_sha256", "clean", "upstream_ahead", "upstream_behind"
    }:
        raise TrainingExpansionError("Frozen repository authority is invalid.")
    commit = str(value.get("commit") or "")
    current = _repository_authority()
    if (
        not COMMIT_RE.fullmatch(commit)
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, current["commit"]])
    ):
        raise TrainingExpansionError("Frozen repository authority is invalid.")
    blob = subprocess.run(
        ["git", "show", f"{commit}:acoustic_training_expansion.py"],
        cwd=Path(__file__).resolve().parent, check=False, capture_output=True,
    )
    if (
        blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest() != value.get("module_sha256")
        or sha256_file(Path(__file__).resolve()) != value.get("module_sha256")
    ):
        raise TrainingExpansionError("Training intake module authority drifted.")
    return dict(value)


def _paths(root: Path, intake_id: str = "") -> dict[str, Path]:
    selected_root = root.expanduser().absolute()
    base = selected_root / "intakes"
    intake = base / intake_id if intake_id else base
    return {
        "root": selected_root,
        "base": base,
        "intake": intake,
        "manifest": intake / "private-manifest.json",
        "receipt": intake / "receipt.json",
    }


def _existing_manifests(root: Path) -> list[Path]:
    paths = _paths(root)
    if not paths["base"].exists():
        return []
    if not paths["base"].is_dir() or paths["base"].is_symlink():
        raise TrainingExpansionError("Training intake root is invalid.")
    manifests = []
    for child in sorted(paths["base"].iterdir()):
        if not child.is_dir() or child.is_symlink():
            raise TrainingExpansionError("Unknown training intake entry exists.")
        if {item.name for item in child.iterdir()} != {
            "private-manifest.json", "receipt.json"
        }:
            raise TrainingExpansionError("Partial or unknown training intake exists.")
        manifests.append(child / "private-manifest.json")
    return manifests


def _manifest_core(
    preview: Mapping[str, Any], private: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "applied",
        "preview": dict(preview),
        "private_inputs": dict(private),
        "repository_authority": dict(repository),
        "authorized_actions": {
            "prepare_audio": True,
            "build_speaker_review_packets": True,
            "infer_or_confirm_identity": False,
            "register_references": False,
            "materialize_profiles": False,
            "run_evaluation": False,
        },
        "contains_private_paths": True,
        "contains_transcript_text": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _receipt(
    preview: Mapping[str, Any], intake_id: str, content_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "applied",
        "intake_id": intake_id,
        "authority_content_sha256": content_sha256,
        "manifest_sha256": manifest_sha256,
        "preview_id": preview["preview_id"],
        "preview_content_sha256": preview["content_sha256"],
        "conversation_count": preview["conversation_count"],
        "speaker_label_count": preview["speaker_label_count"],
        "prior_corpus_overlap_count": 0,
        "audio_preparation_authorized": True,
        "speaker_review_packet_authorized": True,
        "identity_confirmation_authorized": False,
        "reference_registration_authorized": False,
        "profile_materialization_authorized": False,
        "evaluation_execution_authorized": False,
        "contains_paths": False,
        "contains_names_or_emails": False,
        "contains_transcript_text": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "mode": "0600",
        "will_perform_external_write": False,
    }


def apply_training_intake(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    conversations: Sequence[Mapping[str, Any]],
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Apply the reviewed exact intake without processing audio or identity."""
    preview, private = _evaluate(
        conversations, source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
        or preview["status"] != "ready_for_independent_review"
    ):
        raise TrainingExpansionError("Reviewed training intake preview is stale.")
    repository = _repository_authority()
    core = _manifest_core(preview, private, repository)
    content_sha256 = _canonical_hash(core)
    intake_id = f"training-intake-authority-{content_sha256[:24]}"
    root = runtime_root or DEFAULT_RUNTIME_ROOT
    existing = _existing_manifests(root)
    if len(existing) > 1:
        raise TrainingExpansionError("Multiple training intake authorities exist.")
    if existing:
        return replay_training_intake(
            existing[0], conversations=conversations, source_root=source_root,
            corpus_manifest_paths=corpus_manifest_paths, runtime_root=root,
        )
    paths = _paths(root, intake_id)
    ensure_private_tree(paths["root"], paths["intake"])
    manifest = {**core, "intake_id": intake_id, "content_sha256": content_sha256}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, intake_id, content_sha256, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent": False,
    }


def replay_training_intake(
    manifest_path: Path, *, conversations: Sequence[Mapping[str, Any]],
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Replay the exact private intake and portable receipt full-body."""
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    selected = manifest_path.expanduser().resolve(strict=True)
    require_private_file(selected, root)
    manifest = _read_object(selected)
    preview, private = _evaluate(
        conversations, source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    core = _manifest_core(preview, private, repository)
    content_sha256 = _canonical_hash(core)
    intake_id = f"training-intake-authority-{content_sha256[:24]}"
    expected_manifest = {**core, "intake_id": intake_id, "content_sha256": content_sha256}
    if (
        manifest != expected_manifest
        or selected != _paths(root, intake_id)["manifest"]
        or _existing_manifests(root) != [selected]
    ):
        raise TrainingExpansionError("Training intake manifest replay mismatch.")
    receipt_path = selected.parent / "receipt.json"
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = _receipt(
        preview, intake_id, content_sha256, sha256_file(selected)
    )
    if receipt != expected_receipt:
        raise TrainingExpansionError("Training intake receipt replay mismatch.")
    return {
        "schema_version": REPLAY_SCHEMA,
        **expected_receipt,
        "full_body_match": True,
        "idempotent": True,
    }
