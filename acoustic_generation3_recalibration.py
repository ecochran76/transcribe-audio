"""Generation-3 successor recalibration pre-score authority."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import acoustic_generation3_authority as cohort
import acoustic_generation3_gold as gold
import acoustic_training_expansion as training
import acoustic_verification as verification
from acoustic_audio_derivatives import (
    AudioDerivativeError,
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-recalibration-preview.v1"
PORTABLE_SCHEMA = "transcribe-audio.generation3-recalibration-portable.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation3-recalibration-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-recalibration-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-recalibration-replay.v1"
DEFAULT_RUNTIME_ROOT = cohort.DEFAULT_RUNTIME_ROOT
DEFAULT_CALIBRATION_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/verification-calibration"
)
DEFAULT_P3_RUNTIME_ROOT = cohort.DEFAULT_P3_RUNTIME_ROOT
DEFAULT_CORPUS_MANIFESTS = cohort.DEFAULT_CORPUS_MANIFESTS
HISTORICAL_CALIBRATION_AUTHORITY_SHA256 = (
    "0fe6009bef2adfc9c48d87eea7d4ac15c00734ec45376ba3dbba45952e42fae5"
)
HISTORICAL_CALIBRATION_APPLICATION_SHA256 = (
    "c00df454c799e5afa3993dec01c4f021e9236ced109b9bfcd6a44685a3f6a05b"
)
EXPECTED_WINDOW_COUNT = 22
EXPECTED_PROFILE_COUNT = 6
EXPECTED_SUBJECT_COUNT = 2
EXPECTED_UNIT_COUNT = 9
EXPECTED_TRIALS_PER_UNIT = 44
EXPECTED_GENUINE_PER_UNIT = 9
EXPECTED_IMPOSTOR_PER_UNIT = 35
EXPECTED_OPEN_SET_PER_UNIT = 26
METHOD_IDS = verification.CALIBRATION_SCORE_METHOD_IDS
CANDIDATE_IDS = tuple(sorted(verification.EXPECTED_CANDIDATES))
DIMENSIONS = (
    "source_sha256",
    "recording_identity_sha256",
    "conversation_identity_sha256",
    "derivative_identity_sha256",
)
SHA256_RE = re.compile(r"[a-f0-9]{64}")
COMMIT_RE = re.compile(r"[a-f0-9]{40}")


class Generation3RecalibrationError(ValueError):
    """Raised when successor recalibration cannot remain pre-score and exact."""


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
        raise Generation3RecalibrationError(
            "Generation-3 recalibration JSON is unreadable."
        ) from exc
    if not isinstance(value, dict):
        raise Generation3RecalibrationError(
            "Generation-3 recalibration JSON must be an object."
        )
    return value


def _empty_dimensions() -> dict[str, set[str]]:
    return {key: set() for key in DIMENSIONS}


def _dimension_authority(values: Mapping[str, set[str]]) -> dict[str, Any]:
    return {
        key: {
            "count": len(values[key]),
            "set_sha256": _canonical_hash(sorted(values[key])),
        }
        for key in DIMENSIONS
    }


def _evaluation_semantic_identity(
    record: Mapping[str, Any], *, transcripts_root: Path | None = None
) -> tuple[tuple[str, str, str], dict[str, str] | None]:
    source = record.get("source_blob")
    lineage = record.get("transcript_lineage")
    if not isinstance(source, Mapping):
        raise Generation3RecalibrationError("Evaluation corpus lineage is invalid.")
    source_sha = str(source.get("sha256") or "")
    recording_id = str(record.get("recording_id") or "")
    conversation_id = str(record.get("conversation_id") or "")
    key = (source_sha, recording_id, conversation_id)
    if (
        not SHA256_RE.fullmatch(source_sha)
        or not recording_id
        or not conversation_id
    ):
        raise Generation3RecalibrationError("Evaluation corpus identity is invalid.")
    if not isinstance(lineage, Mapping):
        return key, None
    transcript_path = Path(str(lineage.get("current_artifact_path") or ""))
    expected_sha = str(lineage.get("current_artifact_sha256") or "")
    root = (transcripts_root or (Path.home() / ".transcripts")).absolute()
    try:
        require_private_file(transcript_path, root)
    except AudioDerivativeError as exc:
        raise Generation3RecalibrationError(
            "Evaluation transcript lineage is not private."
        ) from exc
    if (
        not SHA256_RE.fullmatch(expected_sha)
        or sha256_file(transcript_path) != expected_sha
    ):
        raise Generation3RecalibrationError(
            "Evaluation transcript lineage drifted."
        )
    transcript = _read_object(transcript_path)
    if (
        str(transcript.get("recording_id") or "") != recording_id
        or str(transcript.get("conversation_id") or "") != conversation_id
    ):
        raise Generation3RecalibrationError(
            "Evaluation transcript semantic identity drifted."
        )
    return key, cohort._transcript_identities(transcript)


def _require_semantic_coverage(
    expected: set[tuple[str, str, str]],
    observed: set[tuple[str, str, str]],
) -> None:
    if expected != observed:
        raise Generation3RecalibrationError(
            "Every prior evaluation source requires validated semantic lineage."
        )


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise Generation3RecalibrationError("Repository authority is unavailable.")
    return completed.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    module_names = (
        "acoustic_generation3_recalibration.py",
        "acoustic_generation3_gold.py",
        "acoustic_generation3_authority.py",
        "acoustic_verification.py",
        "acoustic_speech_preparation.py",
        "acoustic_audio_derivatives.py",
        "acoustic_training_expansion.py",
        "acoustic_biometric_references.py",
    )
    status = _git(["status", "--porcelain"])
    upstream = _git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"])
    behind, ahead = (int(value) for value in upstream.split())
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in module_names
        },
        "clean": status == "",
        "upstream_ahead": ahead,
        "upstream_behind": behind,
    }


def _require_current_repository_even() -> None:
    current_upstream = _git(
        ["rev-list", "--left-right", "--count", "@{upstream}...HEAD"]
    )
    current_behind, current_ahead = (
        int(item) for item in current_upstream.split()
    )
    if (
        _git(["status", "--porcelain"])
        or current_ahead != 0
        or current_behind != 0
    ):
        raise Generation3RecalibrationError(
            "Current repository is not clean and upstream-even."
        )


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    expected_names = {
        "acoustic_generation3_recalibration.py",
        "acoustic_generation3_gold.py",
        "acoustic_generation3_authority.py",
        "acoustic_verification.py",
        "acoustic_speech_preparation.py",
        "acoustic_audio_derivatives.py",
        "acoustic_training_expansion.py",
        "acoustic_biometric_references.py",
    }
    if not isinstance(value, Mapping):
        raise Generation3RecalibrationError("Repository authority is invalid.")
    commit = str(value.get("commit") or "")
    modules = value.get("module_sha256")
    if (
        set(value) != {
            "commit",
            "module_sha256",
            "clean",
            "upstream_ahead",
            "upstream_behind",
        }
        or not COMMIT_RE.fullmatch(commit)
        or not isinstance(modules, Mapping)
        or set(modules) != expected_names
        or any(not SHA256_RE.fullmatch(str(item)) for item in modules.values())
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3RecalibrationError("Repository authority drifted.")
    for name, digest in modules.items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{name}"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
        )
        if (
            blob.returncode != 0
            or hashlib.sha256(blob.stdout).hexdigest() != digest
            or sha256_file(Path(__file__).resolve().parent / name) != digest
        ):
            raise Generation3RecalibrationError("Repository module authority drifted.")
    _require_current_repository_even()
    return dict(value)


def _historical_context(
    *, calibration_root: Path, corpus_manifest_paths: Sequence[Path]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, set[str]]]:
    root = calibration_root.expanduser().absolute()
    authority_path = root / "calibration-authorities" / (
        HISTORICAL_CALIBRATION_AUTHORITY_SHA256 + ".json"
    )
    application_path = root / "calibration-applications" / (
        HISTORICAL_CALIBRATION_APPLICATION_SHA256 + ".json"
    )
    require_private_file(authority_path, root)
    require_private_file(application_path, root)
    authority = _read_object(authority_path)
    application = _read_object(application_path)
    if (
        verification.canonical_artifact_hash(authority)
        != HISTORICAL_CALIBRATION_AUTHORITY_SHA256
        or verification._calibration_stage_identity(application, "applied_at")
        != HISTORICAL_CALIBRATION_APPLICATION_SHA256
        or application.get("authority_sha256")
        != HISTORICAL_CALIBRATION_AUTHORITY_SHA256
        or application.get("status") != "success"
        or application.get("threshold_unit_count") != EXPECTED_UNIT_COUNT
        or application.get("did_read_evaluation") is not False
        or application.get("did_select_and_freeze_thresholds") is not True
    ):
        raise Generation3RecalibrationError(
            "Historical calibration authority is invalid."
        )
    try:
        selection = verification._replay_historical_calibration_window_selection(
            authority,
            HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
            runtime_root=root,
            parent_corpus_manifest_path=Path(corpus_manifest_paths[0]).expanduser(),
        )
    except verification.AcousticVerificationError as exc:
        raise Generation3RecalibrationError(
            "Historical calibration membership replay failed."
        ) from exc
    preparation_path = root / "calibration-stages" / (
        HISTORICAL_CALIBRATION_AUTHORITY_SHA256
    ) / "preparation.json"
    require_private_file(preparation_path, root)
    historical_preparation = _read_object(preparation_path)
    if (
        selection.get("window_count") != EXPECTED_WINDOW_COUNT
        or selection.get("did_run_biometrics") is not False
        or selection.get("did_read_evaluation") is not False
        or historical_preparation.get("status") != "success"
        or historical_preparation.get("did_run_biometrics") is not False
        or historical_preparation.get("did_read_evaluation") is not False
    ):
        raise Generation3RecalibrationError(
            "Historical calibration stages are invalid."
        )
    records = verification._calibration_records_after_authority(
        authority,
        parent_corpus_manifest_path=Path(corpus_manifest_paths[0]).expanduser(),
    )
    records_by_id = {str(item["recording_id"]): item for item in records}
    calibration_dimensions = _empty_dimensions()
    for item in selection["windows"]:
        record = records_by_id.get(str(item["recording_id"]))
        if not isinstance(record, Mapping):
            raise Generation3RecalibrationError(
                "Historical calibration semantic lineage is invalid."
            )
        transcript_path = Path(
            str(record.get("transcript_lineage", {}).get("current_artifact_path") or "")
        )
        identities = cohort._transcript_identities(_read_object(transcript_path))
        calibration_dimensions["source_sha256"].add(str(item["source_sha256"]))
        for key in DIMENSIONS[1:]:
            calibration_dimensions[key].add(identities[key])
    evaluation_dimensions = _empty_dimensions()
    evaluation_keys: set[tuple[str, str, str]] = set()
    semantic_keys: set[tuple[str, str, str]] = set()
    corpus_bindings = []
    for raw_path in corpus_manifest_paths:
        path = Path(raw_path).expanduser().absolute()
        require_private_file(path, path.parent.parent)
        manifest = _read_object(path)
        content_sha = str(manifest.get("content_sha256") or "")
        if not SHA256_RE.fullmatch(content_sha):
            raise Generation3RecalibrationError("Corpus authority is invalid.")
        corpus_bindings.append(
            {
                "corpus_id": str(manifest.get("corpus_id") or ""),
                "content_sha256": content_sha,
                "manifest_sha256": sha256_file(path),
            }
        )
        for record in manifest.get("recordings") or []:
            if not isinstance(record, Mapping) or record.get("split") != "evaluation":
                continue
            key, identities = _evaluation_semantic_identity(record)
            source_sha, recording_id, conversation_id = key
            evaluation_keys.add(key)
            evaluation_dimensions["source_sha256"].add(source_sha)
            if identities is not None:
                for key in DIMENSIONS[1:]:
                    evaluation_dimensions[key].add(identities[key])
                semantic_keys.add((source_sha, recording_id, conversation_id))
    _require_semantic_coverage(evaluation_keys, semantic_keys)
    if any(calibration_dimensions[key] & evaluation_dimensions[key] for key in DIMENSIONS):
        raise Generation3RecalibrationError(
            "Historical calibration overlaps a prior evaluation generation."
        )
    safe = {
        "authority_sha256": HISTORICAL_CALIBRATION_AUTHORITY_SHA256,
        "application_sha256": HISTORICAL_CALIBRATION_APPLICATION_SHA256,
        "authority_file_sha256": sha256_file(authority_path),
        "application_file_sha256": sha256_file(application_path),
        "window_selection_sha256": selection["window_selection_sha256"],
        "preparation_sha256": verification._calibration_stage_identity(
            historical_preparation, "prepared_at"
        ),
        "window_count": selection["window_count"],
        "calibration_dimensions": _dimension_authority(calibration_dimensions),
        "prior_evaluation_dimensions": _dimension_authority(evaluation_dimensions),
        "corpora": sorted(corpus_bindings, key=lambda item: item["corpus_id"]),
        "score_methods": list(authority["score_methods"]),
        "threshold_policy": dict(authority["threshold_policy"]),
        "metric_policy": dict(authority["metric_policy"]),
        "selection_objective": list(application["selection_objective"]),
    }
    private = {
        "authority": authority,
        "selection": selection,
        "preparation": historical_preparation,
    }
    return safe, private, calibration_dimensions, evaluation_dimensions


def _training_dimensions() -> tuple[dict[str, Any], dict[str, set[str]]]:
    try:
        paths = training._existing_manifests(cohort.DEFAULT_TRAINING_RUNTIME_ROOT)
    except training.TrainingExpansionError as exc:
        raise Generation3RecalibrationError(str(exc)) from exc
    if len(paths) != 1:
        raise Generation3RecalibrationError("Active training authority is unavailable.")
    path = paths[0]
    manifest = _read_object(path)
    safe_units = manifest.get("preview", {}).get("conversations")
    private_units = manifest.get("private_inputs", {}).get("conversations")
    if not isinstance(safe_units, list) or not isinstance(private_units, list):
        raise Generation3RecalibrationError("Active training lineage is invalid.")
    if not private_units:
        raise Generation3RecalibrationError("Active training lineage is empty.")
    active_sources = {
        str(item.get("source_sha256") or "")
        for item in safe_units
        if isinstance(item, Mapping)
    }
    source_root = Path(str(private_units[0].get("source_path") or "")).parent
    try:
        authority, dimensions = cohort._active_training_dimensions(
            active_sources=active_sources,
            source_root=source_root,
            training_runtime_root=cohort.DEFAULT_TRAINING_RUNTIME_ROOT,
            corpus_manifest_paths=DEFAULT_CORPUS_MANIFESTS,
        )
    except cohort.Generation3AuthorityError as exc:
        raise Generation3RecalibrationError(
            "Active training authority replay failed."
        ) from exc
    selected = {key: set(dimensions[key]) for key in DIMENSIONS}
    return {
        **authority,
        "dimensions": _dimension_authority(selected),
    }, selected


def _generation3_context() -> tuple[dict[str, Any], dict[str, set[str]]]:
    authority_root = DEFAULT_RUNTIME_ROOT.expanduser() / "cohort-authorities"
    manifests = sorted(authority_root.glob("*/private-manifest.json"))
    if len(manifests) != 1:
        raise Generation3RecalibrationError("Generation-3 cohort authority is unavailable.")
    path = manifests[0]
    require_private_file(path, DEFAULT_RUNTIME_ROOT.expanduser().absolute())
    manifest = _read_object(path)
    membership = manifest.get("preview", {}).get("membership", {}).get("conversations")
    if not isinstance(membership, list) or len(membership) != 7:
        raise Generation3RecalibrationError("Generation-3 cohort membership is invalid.")
    dimensions = _empty_dimensions()
    for item in membership:
        dimensions["source_sha256"].add(str(item.get("source_sha256") or ""))
        for key in DIMENSIONS[1:]:
            dimensions[key].add(str(item.get(key) or ""))
    return {
        "authority_id": str(manifest.get("authority_id") or ""),
        "manifest_sha256": sha256_file(path),
        "membership_sha256": str(manifest.get("preview", {}).get("membership_sha256") or ""),
        "dimensions": _dimension_authority(dimensions),
    }, dimensions


def _gold_receipt() -> dict[str, Any]:
    root = DEFAULT_RUNTIME_ROOT.expanduser().absolute()
    receipts = sorted((root / "gold-authorities").glob("*/receipt.json"))
    if len(receipts) != 1:
        raise Generation3RecalibrationError("Generation-3 gold receipt is unavailable.")
    path = receipts[0]
    require_private_file(path, root)
    receipt = _read_object(path)
    manifest_path = path.parent / "private-manifest.json"
    require_private_file(manifest_path, root)
    actions = receipt.get("action_vector")
    if (
        receipt.get("schema_version") != gold.RECEIPT_SCHEMA
        or receipt.get("status") != "applied_gold_frozen_evaluation_not_revealed"
        or receipt.get("gold_label_count") != 28
        or not isinstance(actions, Mapping)
        or actions.get("freeze_gold") is not True
        or actions.get("build_successor_recalibration_authority") is not True
        or actions.get("reveal_evaluation") is not False
        or sha256_file(manifest_path) != receipt.get("manifest_sha256")
        or any(
            receipt.get(key) is not False
            for key in (
                "contains_names",
                "contains_paths",
                "contains_private_gold",
                "contains_source_membership",
                "contains_subject_ids",
                "contains_transcript_text",
                "contains_raw_audio",
                "contains_embeddings_or_vectors",
                "contains_biometric_scores",
            )
        )
    ):
        raise Generation3RecalibrationError("Generation-3 gold receipt is invalid.")
    return {
        "gold_id": str(receipt.get("gold_id") or ""),
        "receipt_sha256": sha256_file(path),
        "manifest_sha256": str(receipt.get("manifest_sha256") or ""),
        "gold_body_sha256": str(receipt.get("gold_body_sha256") or ""),
        "membership_sha256": str(receipt.get("membership_sha256") or ""),
        "gold_label_count": receipt["gold_label_count"],
    }


def _active_profiles(
    *, calibration_root: Path, p3_runtime_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = calibration_root.expanduser().absolute()
    database = root / "profiles.sqlite3"
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM profiles WHERE lifecycle_state = 'active' "
            "ORDER BY candidate_id, person_ref_id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise Generation3RecalibrationError("Active profile inventory is unavailable.") from exc
    finally:
        if "connection" in locals():
            connection.close()
    if (
        len(rows) != EXPECTED_PROFILE_COUNT
        or {str(row["candidate_id"]) for row in rows} != set(CANDIDATE_IDS)
        or len({str(row["person_ref_id"]) for row in rows}) != EXPECTED_SUBJECT_COUNT
        or any(
            sum(row["candidate_id"] == candidate_id for row in rows) != 2
            for candidate_id in CANDIDATE_IDS
        )
    ):
        raise Generation3RecalibrationError("Exactly six active successor profiles are required.")
    profiles = []
    try:
        adapters = verification.adapter_registry()
    except verification.AcousticVerificationError as exc:
        raise Generation3RecalibrationError("Model adapter inventory is invalid.") from exc
    if set(adapters) != set(CANDIDATE_IDS) or any(
        adapter.model_loaded for adapter in adapters.values()
    ):
        raise Generation3RecalibrationError("Model adapter inventory is invalid.")
    for row in rows:
        profile_id = str(row["profile_id"])
        try:
            replay = verification.replay_profile(profile_id, runtime_root=root)
            eligible = verification.descendant_is_eligible(
                str(row["descendant_id"]), runtime_root=p3_runtime_root.expanduser()
            )
        except verification.AcousticVerificationError as exc:
            raise Generation3RecalibrationError("Active profile replay failed.") from exc
        adapter = adapters.get(str(row["candidate_id"]))
        if (
            replay.get("lifecycle_state") != "active"
            or eligible is not True
            or adapter is None
            or adapter.revision_sha != row["model_revision"]
        ):
            raise Generation3RecalibrationError("Active profile eligibility is invalid.")
        profiles.append(
            {
                "profile_id": profile_id,
                "descendant_id": str(row["descendant_id"]),
                "person_ref_id": str(row["person_ref_id"]),
                "p3_profile_id": str(row["p3_profile_id"]),
                "generation_id": str(row["generation_id"]),
                "generation_sha256": str(row["generation_sha256"]),
                "candidate_id": str(row["candidate_id"]),
                "model_revision": str(row["model_revision"]),
                "preprocessing": json.loads(str(row["preprocessing_json"])),
                "artifact_sha256": str(row["artifact_sha256"]),
                "profile_manifest_sha256": str(row["profile_manifest_sha256"]),
                "state_receipt_sha256": str(row["state_receipt_sha256"]),
                "vector_dimension": int(row["vector_dimension"]),
                "window_count": int(row["window_count"]),
                "session_count": int(row["session_count"]),
            }
        )
    model_assets = {}
    for candidate_id in CANDIDATE_IDS:
        try:
            records = verification._verified_model_artifacts(
                verification.DEFAULT_MODEL_SNAPSHOT_ROOT.expanduser().parent,
                candidate_id,
            )
        except verification.AcousticVerificationError as exc:
            raise Generation3RecalibrationError("Model asset replay failed.") from exc
        model_assets[candidate_id] = {
            name: {
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
            for name, record in sorted(records.items())
        }
    return profiles, {
        "profile_count": len(profiles),
        "subject_count": len({item["person_ref_id"] for item in profiles}),
        "candidate_count": len(CANDIDATE_IDS),
        "profile_set_sha256": _canonical_hash(profiles),
        "model_assets": model_assets,
        "model_asset_set_sha256": _canonical_hash(model_assets),
    }


def _evaluate(
    *, calibration_root: Path = DEFAULT_CALIBRATION_ROOT,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
) -> dict[str, Any]:
    selected_corpora = {
        Path(path).expanduser().absolute() for path in corpus_manifest_paths
    }
    expected_corpora = {
        Path(path).expanduser().absolute() for path in DEFAULT_CORPUS_MANIFESTS
    }
    if selected_corpora != expected_corpora or len(corpus_manifest_paths) != len(
        DEFAULT_CORPUS_MANIFESTS
    ):
        raise Generation3RecalibrationError(
            "The exact prior evaluation corpus inventory is required."
        )
    historical, private_historical, calibration_dimensions, _ = _historical_context(
        calibration_root=calibration_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    training_authority, training_dimensions = _training_dimensions()
    cohort_authority, generation3_dimensions = _generation3_context()
    gold_commitment = _gold_receipt()
    if (
        gold_commitment["membership_sha256"]
        != cohort_authority["membership_sha256"]
    ):
        raise Generation3RecalibrationError(
            "Generation-3 gold and cohort membership differ."
        )
    overlaps = {
        "training": {
            key: len(calibration_dimensions[key] & training_dimensions[key])
            for key in DIMENSIONS
        },
        "generation3": {
            key: len(calibration_dimensions[key] & generation3_dimensions[key])
            for key in DIMENSIONS
        },
    }
    if any(value for group in overlaps.values() for value in group.values()):
        raise Generation3RecalibrationError(
            "Calibration membership overlaps training or Generation-3 evaluation."
        )
    profiles, profile_authority = _active_profiles(
        calibration_root=calibration_root,
        p3_runtime_root=p3_runtime_root,
    )
    profile_subjects = {str(item["person_ref_id"]) for item in profiles}
    expected_pairs = {
        (candidate_id, subject_id)
        for candidate_id in CANDIDATE_IDS
        for subject_id in profile_subjects
    }
    actual_pairs = {
        (str(item["candidate_id"]), str(item["person_ref_id"]))
        for item in profiles
    }
    subject_lineage: dict[str, set[tuple[str, str, str]]] = {}
    for item in profiles:
        subject_lineage.setdefault(str(item["person_ref_id"]), set()).add(
            (
                str(item["p3_profile_id"]),
                str(item["generation_id"]),
                str(item["generation_sha256"]),
            )
        )
    if (
        len(profiles) != EXPECTED_PROFILE_COUNT
        or len(profile_subjects) != EXPECTED_SUBJECT_COUNT
        or actual_pairs != expected_pairs
        or any(len(values) != 1 for values in subject_lineage.values())
        or profile_authority.get("profile_count") != len(profiles)
        or profile_authority.get("subject_count") != len(profile_subjects)
        or profile_authority.get("candidate_count") != len(CANDIDATE_IDS)
        or profile_authority.get("profile_set_sha256") != _canonical_hash(profiles)
    ):
        raise Generation3RecalibrationError(
            "Active successor profile Cartesian lineage is invalid."
        )
    windows = private_historical.get("selection", {}).get("windows")
    if (
        not isinstance(windows, list)
        or len(windows) != EXPECTED_WINDOW_COUNT
        or any(
            not isinstance(item, Mapping)
            or not str(item.get("window_id") or "")
            or not str(item.get("subject_id") or "")
            for item in windows
        )
    ):
        raise Generation3RecalibrationError(
            "Historical calibration denominator membership is invalid."
        )
    genuine_windows = sum(
        str(item.get("subject_id") or "") in profile_subjects
        for item in windows
        if isinstance(item, Mapping)
    )
    open_set_windows = sum(
        str(item.get("subject_id") or "") not in profile_subjects
        for item in windows
        if isinstance(item, Mapping)
    )
    derived_denominators = {
        "trials_per_unit": len(windows) * EXPECTED_SUBJECT_COUNT,
        "genuine_trials_per_unit": genuine_windows,
        "impostor_trials_per_unit": (
            len(windows) * EXPECTED_SUBJECT_COUNT - genuine_windows
        ),
        "open_set_trials_per_unit": open_set_windows * EXPECTED_SUBJECT_COUNT,
    }
    if derived_denominators != {
        "trials_per_unit": EXPECTED_TRIALS_PER_UNIT,
        "genuine_trials_per_unit": EXPECTED_GENUINE_PER_UNIT,
        "impostor_trials_per_unit": EXPECTED_IMPOSTOR_PER_UNIT,
        "open_set_trials_per_unit": EXPECTED_OPEN_SET_PER_UNIT,
    }:
        raise Generation3RecalibrationError(
            "Historical successor calibration denominators changed."
        )
    if historical["score_methods"] != list(METHOD_IDS):
        raise Generation3RecalibrationError("Calibration method inventory drifted.")
    units = [
        {"candidate_id": candidate_id, "method_id": method_id}
        for candidate_id in CANDIDATE_IDS
        for method_id in METHOD_IDS
    ]
    if len(units) != EXPECTED_UNIT_COUNT:
        raise Generation3RecalibrationError("Recalibration unit inventory is invalid.")
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "historical_calibration": historical,
        "active_profile_authority": profile_authority,
        "profiles": profiles,
        "training_authority": training_authority,
        "generation3_cohort_authority": cohort_authority,
        "generation3_gold_commitment": gold_commitment,
        "disjointness_overlap_counts": overlaps,
        "units": units,
        "unit_count": len(units),
        "expected_trials_per_unit": derived_denominators["trials_per_unit"],
        "expected_genuine_trials_per_unit": derived_denominators[
            "genuine_trials_per_unit"
        ],
        "expected_impostor_trials_per_unit": derived_denominators[
            "impostor_trials_per_unit"
        ],
        "expected_open_set_trials_per_unit": derived_denominators[
            "open_set_trials_per_unit"
        ],
        "denominator_derivation_sha256": _canonical_hash(
            {
                "window_subject_set_sha256": _canonical_hash(
                    sorted(str(item.get("subject_id") or "") for item in windows)
                ),
                "profile_subject_set_sha256": _canonical_hash(
                    sorted(profile_subjects)
                ),
                **derived_denominators,
            }
        ),
        "scoring_rule": "raw_cosine_successor_centroid_to_probe_embedding",
        "aggregation_rule": "one_trial_per_profile_window_method_candidate",
        "abstention_margin": 0.0,
        "did_read_generation3_gold": False,
        "did_read_generation3_audio": False,
        "did_load_or_run_models": False,
        "did_score_trials": False,
        "did_select_thresholds": False,
        "action_vector": {
            "freeze_recalibration_authority": False,
            "run_calibration_models": False,
            "freeze_thresholds_and_temperatures": False,
            "build_pre_reveal_envelope": False,
            "reveal_evaluation": False,
            "prepare_evaluation_audio": False,
            "freeze_evaluation_windows": False,
            "construct_exact_trial_child": False,
            "score_evaluation_trials": False,
            "calculate_evaluation_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "contains_private_membership": True,
        "contains_profile_or_subject_ids": True,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }
    content_sha = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-recalibration-preview-{content_sha[:24]}",
        "content_sha256": content_sha,
    }


def preview_generation3_recalibration(**kwargs: Any) -> dict[str, Any]:
    """Build the exact pre-score successor recalibration preview."""
    return _evaluate(**kwargs)


def portable_recalibration_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PORTABLE_SCHEMA,
        "status": preview["status"],
        "preview_content_sha256": preview["content_sha256"],
        "historical_calibration_authority_sha256": preview["historical_calibration"][
            "authority_sha256"
        ],
        "calibration_membership_sha256": _canonical_hash(
            preview["historical_calibration"]["calibration_dimensions"]
        ),
        "window_selection_sha256": preview["historical_calibration"][
            "window_selection_sha256"
        ],
        "profile_set_sha256": preview["active_profile_authority"][
            "profile_set_sha256"
        ],
        "model_asset_set_sha256": preview["active_profile_authority"][
            "model_asset_set_sha256"
        ],
        "gold_receipt_sha256": preview["generation3_gold_commitment"][
            "receipt_sha256"
        ],
        "profile_count": preview["active_profile_authority"]["profile_count"],
        "subject_count": preview["active_profile_authority"]["subject_count"],
        "candidate_count": preview["active_profile_authority"]["candidate_count"],
        "method_count": len(METHOD_IDS),
        "unit_count": preview["unit_count"],
        "window_count": preview["historical_calibration"]["window_count"],
        "expected_trials_per_unit": preview["expected_trials_per_unit"],
        "expected_genuine_trials_per_unit": preview[
            "expected_genuine_trials_per_unit"
        ],
        "expected_impostor_trials_per_unit": preview[
            "expected_impostor_trials_per_unit"
        ],
        "expected_open_set_trials_per_unit": preview[
            "expected_open_set_trials_per_unit"
        ],
        "disjointness_overlap_counts": preview["disjointness_overlap_counts"],
        "abstention_margin_sha256": _canonical_hash(preview["abstention_margin"]),
        "abstention_margin_is_zero": preview["abstention_margin"] == 0.0,
        "action_vector": dict(preview["action_vector"]),
        "contains_private_membership": False,
        "contains_profile_or_subject_ids": False,
        "contains_paths": False,
        "contains_raw_audio": False,
        "contains_transcript_text": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _manifest_core(
    preview: Mapping[str, Any], repository: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "preview": dict(preview),
        "portable_projection": portable_recalibration_projection(preview),
        "repository_authority": dict(repository),
    }


def _paths(runtime_root: Path, authority_id: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    authority = root / "recalibration-authorities" / authority_id
    return {
        "root": root,
        "authority": authority,
        "manifest": authority / "private-manifest.json",
        "receipt": authority / "receipt.json",
    }


def _existing_manifest(runtime_root: Path) -> Path | None:
    root = runtime_root.expanduser().absolute()
    paths = sorted((root / "recalibration-authorities").glob("*/private-manifest.json"))
    if len(paths) > 1:
        raise Generation3RecalibrationError("Multiple recalibration authorities exist.")
    return paths[0] if paths else None


def _receipt(
    preview: Mapping[str, Any], authority_id: str, manifest_sha256: str
) -> dict[str, Any]:
    portable = portable_recalibration_projection(preview)
    actions = dict(portable["action_vector"])
    actions["freeze_recalibration_authority"] = True
    actions["run_calibration_models"] = True
    return {
        **portable,
        "schema_version": RECEIPT_SCHEMA,
        "status": "applied_recalibration_authority_scores_not_run",
        "authority_id": authority_id,
        "manifest_sha256": manifest_sha256,
        "action_vector": actions,
        "mode": "0600",
    }


def apply_generation3_recalibration_authority(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT, **preview_inputs: Any,
) -> dict[str, Any]:
    """Freeze pre-score recalibration inputs without loading a model."""
    preview = _evaluate(**preview_inputs)
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
    ):
        raise Generation3RecalibrationError("Reviewed recalibration preview is stale.")
    existing = _existing_manifest(runtime_root)
    if existing is not None:
        return replay_generation3_recalibration_authority(
            existing, runtime_root=runtime_root, **preview_inputs
        )
    repository = _repository_authority()
    if (
        repository["clean"] is not True
        or repository["upstream_ahead"] != 0
        or repository["upstream_behind"] != 0
    ):
        raise Generation3RecalibrationError(
            "Recalibration apply requires a clean upstream-even repository."
        )
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    authority_id = f"generation3-recalibration-{content_sha[:24]}"
    paths = _paths(runtime_root, authority_id)
    ensure_private_tree(paths["root"], paths["authority"])
    manifest = {**core, "authority_id": authority_id, "content_sha256": content_sha}
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = _receipt(preview, authority_id, sha256_file(paths["manifest"]))
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_recalibration_authority(
    manifest_path: Path, *, runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    **preview_inputs: Any,
) -> dict[str, Any]:
    """Replay exact recalibration inputs while evaluation remains sealed."""
    root = runtime_root.expanduser().absolute()
    path = manifest_path.expanduser().absolute()
    require_private_file(path, root)
    manifest = _read_object(path)
    preview = _evaluate(**preview_inputs)
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    core = _manifest_core(preview, repository)
    content_sha = _canonical_hash(core)
    authority_id = f"generation3-recalibration-{content_sha[:24]}"
    expected = {**core, "authority_id": authority_id, "content_sha256": content_sha}
    if manifest != expected or path != _paths(root, authority_id)["manifest"]:
        raise Generation3RecalibrationError("Recalibration manifest drifted.")
    receipt_path = _paths(root, authority_id)["receipt"]
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = _receipt(preview, authority_id, sha256_file(path))
    if receipt != expected_receipt:
        raise Generation3RecalibrationError("Recalibration receipt drifted.")
    return {
        **receipt,
        "private_manifest_path": str(path),
        "private_receipt_path": str(receipt_path),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
    }
