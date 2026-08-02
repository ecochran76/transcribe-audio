"""Private Generation-3 cohort intake and disjointness authority."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import acoustic_biometric_references as references
import acoustic_audio_derivatives as derivatives
import acoustic_training_expansion as training
from acoustic_audio_derivatives import (
    ensure_private_tree,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)


PREVIEW_SCHEMA = "transcribe-audio.generation3-cohort-preview.v1"
MANIFEST_SCHEMA = "transcribe-audio.generation3-cohort-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.generation3-cohort-receipt.v1"
REPLAY_SCHEMA = "transcribe-audio.generation3-cohort-replay.v1"
DEFAULT_SOURCE_ROOT = Path("~/Documents/Sound Recordings")
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/generation-3"
)
DEFAULT_P3_RUNTIME_ROOT = references.DEFAULT_RUNTIME_ROOT
DEFAULT_TRAINING_RUNTIME_ROOT = training.DEFAULT_RUNTIME_ROOT
DEFAULT_CORPUS_MANIFESTS = training.DEFAULT_CORPUS_MANIFESTS
EXPECTED_CONVERSATION_COUNT = 7
EXPECTED_ACTIVE_REFERENCE_COUNT = 2
MAXIMUM_WINDOWS_PER_SPEAKER_PER_CONVERSATION = 12
SHA256_RE = training.SHA256_RE
COMMIT_RE = training.COMMIT_RE
LINEAGE_DIMENSIONS = (
    "source_sha256",
    "recording_identity_sha256",
    "conversation_identity_sha256",
    "derivative_identity_sha256",
)


class Generation3AuthorityError(ValueError):
    """Raised when Generation-3 cohort authority cannot remain exact."""


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
        raise Generation3AuthorityError("Generation-3 JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise Generation3AuthorityError("Generation-3 JSON must be an object.")
    return value


def _transcript_identities(transcript: Mapping[str, Any]) -> dict[str, str]:
    """Derive path-free identities that survive a media container re-encode."""
    utterances = transcript.get("utterances")
    duration = transcript.get("duration_seconds")
    if (
        transcript.get("schema_version") not in {1, 2}
        or not isinstance(duration, (int, float))
        or not isinstance(utterances, list)
        or not utterances
    ):
        raise Generation3AuthorityError(
            "Transcript identity evidence is unavailable."
        )
    normalized_utterances = []
    for utterance in utterances:
        if not isinstance(utterance, Mapping):
            raise Generation3AuthorityError(
                "Transcript identity evidence is unavailable."
            )
        normalized_utterances.append(
            {
                "speaker": utterance.get("speaker"),
                "start": utterance.get("start"),
                "end": utterance.get("end"),
                "text": utterance.get("text"),
            }
        )
    derivative_identity = _canonical_hash(
        {
            "duration_seconds": float(duration),
            "utterances": normalized_utterances,
        }
    )
    recording_start = transcript.get("recording_start")
    recording_end = transcript.get("recording_end")
    declared_recording_id = str(transcript.get("recording_id") or "")
    if recording_start and recording_end:
        recording_evidence = {
            "recording_start": recording_start,
            "recording_end": recording_end,
            "duration_seconds": float(duration),
            "transcript_window_start_seconds": transcript.get(
                "transcript_window_start_seconds"
            ),
            "transcript_window_end_seconds": transcript.get(
                "transcript_window_end_seconds"
            ),
        }
    elif declared_recording_id:
        recording_evidence = {"declared_recording_id": declared_recording_id}
    else:
        raise Generation3AuthorityError(
            "Recording identity evidence is unavailable."
        )
    recording_identity = _canonical_hash(recording_evidence)
    event = transcript.get("event")
    if isinstance(event, Mapping) and event:
        conversation_evidence = {"event": dict(event)}
    elif transcript.get("conversation_id"):
        conversation_evidence = {
            "declared_conversation_id": str(transcript["conversation_id"])
        }
    else:
        raise Generation3AuthorityError(
            "Conversation identity evidence is unavailable."
        )
    conversation_identity = _canonical_hash(conversation_evidence)
    return {
        "recording_identity_sha256": recording_identity,
        "conversation_identity_sha256": conversation_identity,
        "derivative_identity_sha256": derivative_identity,
    }


def _empty_lineage_dimensions() -> dict[str, set[str]]:
    return {key: set() for key in LINEAGE_DIMENSIONS}


def _merge_lineage_dimensions(
    target: dict[str, set[str]], source: Mapping[str, set[str]]
) -> None:
    for key in LINEAGE_DIMENSIONS:
        target[key].update(source[key])


def _prior_corpus_dimensions(
    manifest_paths: Sequence[Path], expected_sources: set[str]
) -> dict[str, set[str]]:
    dimensions = _empty_lineage_dimensions()
    seen_transcripts: set[str] = set()
    sources_with_transcript_lineage: set[str] = set()
    for raw_path in manifest_paths:
        manifest = _read_object(raw_path.expanduser().absolute())
        for recording in manifest.get("recordings") or []:
            if not isinstance(recording, Mapping):
                raise Generation3AuthorityError(
                    "Frozen corpus recording lineage is invalid."
                )
            source = recording.get("source_blob")
            source_sha = (
                str(source.get("sha256") or "")
                if isinstance(source, Mapping)
                else ""
            )
            recording_id = str(recording.get("recording_id") or "")
            conversation_id = str(recording.get("conversation_id") or "")
            lineage = recording.get("transcript_lineage")
            if (
                source_sha not in expected_sources
                or not recording_id
                or not conversation_id
            ):
                raise Generation3AuthorityError(
                    "Frozen corpus dimensional lineage is invalid."
                )
            dimensions["source_sha256"].add(source_sha)
            if not isinstance(lineage, Mapping):
                continue
            current_sha = str(lineage.get("current_artifact_sha256") or "")
            reviewed_sha = str(lineage.get("reviewed_artifact_sha256") or "")
            transcript_path = Path(
                str(lineage.get("current_artifact_path") or "")
            ).expanduser().absolute()
            if (
                not SHA256_RE.fullmatch(current_sha)
                or not SHA256_RE.fullmatch(reviewed_sha)
                or transcript_path.is_symlink()
                or not transcript_path.is_file()
                or sha256_file(transcript_path) != current_sha
            ):
                raise Generation3AuthorityError(
                    "Frozen corpus transcript lineage drifted."
                )
            dimensions["derivative_identity_sha256"].update(
                {current_sha, reviewed_sha}
            )
            sources_with_transcript_lineage.add(source_sha)
            if current_sha not in seen_transcripts:
                identities = _transcript_identities(_read_object(transcript_path))
                for key, digest in identities.items():
                    dimensions[key].add(digest)
                seen_transcripts.add(current_sha)
    if (
        dimensions["source_sha256"] != expected_sources
        or sources_with_transcript_lineage != expected_sources
    ):
        raise Generation3AuthorityError(
            "Frozen corpus dimensional lineage is incomplete."
        )
    return dimensions


def _active_reference_authority(
    p3_runtime_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    root = p3_runtime_root.expanduser().absolute()
    database = root / "references.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise Generation3AuthorityError("Active P3 reference database is unavailable.")
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        profile_rows = connection.execute(
            """
            SELECT profile_id, person_ref_id, head_generation_id
            FROM profiles WHERE status = 'active' ORDER BY profile_id
            """
        ).fetchall()
        if len(profile_rows) != EXPECTED_ACTIVE_REFERENCE_COUNT:
            raise Generation3AuthorityError(
                "Generation-3 requires exactly two active P3 references."
            )
        authorities: list[dict[str, Any]] = []
        dimensions = _empty_lineage_dimensions()
        active_recording_ids: set[str] = set()
        active_conversation_ids: set[str] = set()
        for profile in profile_rows:
            profile_id = str(profile["profile_id"])
            replay = references.replay_reference(profile_id, runtime_root=root)
            if (
                replay.get("status") != "success"
                or replay.get("eligible_for_materialization") is not True
                or replay.get("head_generation_id")
                != profile["head_generation_id"]
            ):
                raise Generation3AuthorityError(
                    "Active P3 reference replay is ineligible."
                )
            generation = connection.execute(
                """
                SELECT generation_id, manifest_json, manifest_sha256
                FROM generations
                WHERE profile_id = ? AND status = 'active'
                  AND eligible_for_materialization = 1
                """,
                (profile_id,),
            ).fetchone()
            if generation is None:
                raise Generation3AuthorityError(
                    "Active P3 generation binding is unavailable."
                )
            try:
                manifest = json.loads(str(generation["manifest_json"]))
            except json.JSONDecodeError as exc:
                raise Generation3AuthorityError(
                    "Active P3 generation manifest is unreadable."
                ) from exc
            sources = manifest.get("sources") if isinstance(manifest, dict) else None
            if (
                not isinstance(sources, list)
                or not sources
                or references._hash(manifest) != generation["manifest_sha256"]
                or generation["generation_id"] != profile["head_generation_id"]
                or manifest.get("profile_id") != profile_id
                or manifest.get("person_ref_id") != profile["person_ref_id"]
                or manifest.get("eligible_for_materialization") is not True
                or manifest.get("status") != "active"
            ):
                raise Generation3AuthorityError(
                    "Active P3 generation manifest drifted."
                )
            source_hashes: set[str] = set()
            for source in sources:
                digest = (
                    str(source.get("source_sha256") or "")
                    if isinstance(source, Mapping)
                    else ""
                )
                if not SHA256_RE.fullmatch(digest):
                    raise Generation3AuthorityError(
                        "Active P3 source lineage is invalid."
                    )
                source_hashes.add(digest)
                recording_id = str(source.get("recording_id") or "")
                conversation_id = str(source.get("conversation_id") or "")
                lineage = source.get("lineage")
                comparison_sha = (
                    str(lineage.get("comparison_sha256") or "")
                    if isinstance(lineage, Mapping)
                    else ""
                )
                if (
                    not recording_id
                    or not conversation_id
                    or not SHA256_RE.fullmatch(comparison_sha)
                ):
                    raise Generation3AuthorityError(
                        "Active P3 dimensional lineage is invalid."
                    )
                active_recording_ids.add(recording_id)
                active_conversation_ids.add(conversation_id)
                dimensions["derivative_identity_sha256"].add(comparison_sha)
            if not source_hashes:
                raise Generation3AuthorityError(
                    "Active P3 source lineage is empty."
                )
            dimensions["source_sha256"].update(source_hashes)
            authorities.append(
                {
                    "profile_id": profile_id,
                    "person_ref_id": str(profile["person_ref_id"]),
                    "generation_id": str(generation["generation_id"]),
                    "generation_manifest_sha256": str(
                        generation["manifest_sha256"]
                    ),
                    "source_set_sha256": str(manifest.get("source_set_sha256") or ""),
                    "source_count": len(source_hashes),
                }
            )
        for authority in authorities:
            authority["active_recording_id_set_sha256"] = _canonical_hash(
                sorted(active_recording_ids)
            )
            authority["active_conversation_id_set_sha256"] = _canonical_hash(
                sorted(active_conversation_ids)
            )
        return authorities, dimensions
    except sqlite3.Error as exc:
        raise Generation3AuthorityError(
            "Active P3 reference database is unreadable."
        ) from exc
    finally:
        if "connection" in locals():
            connection.close()


def _active_training_dimensions(
    *, active_sources: set[str], source_root: Path,
    training_runtime_root: Path, corpus_manifest_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    try:
        manifests = training._existing_manifests(training_runtime_root)
    except training.TrainingExpansionError as exc:
        raise Generation3AuthorityError(str(exc)) from exc
    if len(manifests) != 1:
        raise Generation3AuthorityError(
            "Exactly one active training intake authority is required."
        )
    path = manifests[0]
    manifest = _read_object(path)
    private = manifest.get("private_inputs")
    conversations = (
        private.get("conversations") if isinstance(private, Mapping) else None
    )
    if not isinstance(conversations, list):
        raise Generation3AuthorityError(
            "Active training intake private lineage is unavailable."
        )
    replay_inputs = []
    for conversation in conversations:
        if not isinstance(conversation, Mapping):
            raise Generation3AuthorityError(
                "Active training intake private lineage is invalid."
            )
        replay_inputs.append(
            {
                "source_path": conversation.get("source_path"),
                "transcript_path": conversation.get("transcript_path"),
            }
        )
    try:
        expected_preview, expected_private = training._evaluate(
            replay_inputs,
            source_root=source_root,
            corpus_manifest_paths=corpus_manifest_paths,
        )
    except training.TrainingExpansionError as exc:
        raise Generation3AuthorityError(
            "Active training intake replay failed."
        ) from exc
    repository = manifest.get("repository_authority")
    if not isinstance(repository, Mapping):
        raise Generation3AuthorityError(
            "Active training intake repository authority is invalid."
        )
    commit = str(repository.get("commit") or "")
    module_sha = str(repository.get("module_sha256") or "")
    if (
        set(repository)
        != {"commit", "module_sha256", "clean", "upstream_ahead", "upstream_behind"}
        or not COMMIT_RE.fullmatch(commit)
        or not SHA256_RE.fullmatch(module_sha)
        or repository.get("clean") is not True
        or repository.get("upstream_ahead") != 0
        or repository.get("upstream_behind") != 0
        or training._git(["merge-base", "--is-ancestor", commit, "HEAD"])
    ):
        raise Generation3AuthorityError(
            "Active training intake repository authority drifted."
        )
    blob = subprocess.run(
        ["git", "show", f"{commit}:acoustic_training_expansion.py"],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
    )
    if (
        blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest() != module_sha
        or sha256_file(Path(training.__file__).resolve()) != module_sha
    ):
        raise Generation3AuthorityError(
            "Active training intake module authority drifted."
        )
    core = training._manifest_core(
        expected_preview, expected_private, repository
    )
    content_sha = training._canonical_hash(core)
    intake_id = f"training-intake-authority-{content_sha[:24]}"
    expected_manifest = {
        **core,
        "intake_id": intake_id,
        "content_sha256": content_sha,
    }
    receipt_path = path.parent / "receipt.json"
    require_private_file(path, training_runtime_root.expanduser().absolute())
    require_private_file(
        receipt_path, training_runtime_root.expanduser().absolute()
    )
    expected_receipt = training._receipt(
        expected_preview,
        intake_id,
        content_sha,
        sha256_file(path),
    )
    if (
        manifest != expected_manifest
        or path.parent.name != intake_id
        or _read_object(receipt_path) != expected_receipt
    ):
        raise Generation3AuthorityError(
            "Active training intake replay failed."
        )
    preview = expected_preview
    safe_conversations = (
        preview.get("conversations") if isinstance(preview, Mapping) else None
    )
    if not isinstance(safe_conversations, list):
        raise Generation3AuthorityError(
            "Active training intake preview lineage is unavailable."
        )
    intake_sources = {
        str(item.get("source_sha256") or "")
        for item in safe_conversations
        if isinstance(item, Mapping)
    }
    if intake_sources != active_sources:
        raise Generation3AuthorityError(
            "Active P3 and training intake source lineage differ."
        )
    dimensions = _empty_lineage_dimensions()
    dimensions["source_sha256"].update(intake_sources)
    for conversation in replay_inputs:
        transcript_path = Path(str(conversation["transcript_path"])).expanduser()
        transcript_sha = sha256_file(transcript_path)
        dimensions["derivative_identity_sha256"].add(transcript_sha)
        identities = _transcript_identities(_read_object(transcript_path))
        for key, digest in identities.items():
            dimensions[key].add(digest)
    return {
        "intake_id": intake_id,
        "content_sha256": content_sha,
        "manifest_sha256": sha256_file(path),
        "conversation_count": len(replay_inputs),
    }, dimensions


def _source_lineage_authority(
    *, corpus_manifest_paths: Sequence[Path], p3_runtime_root: Path,
    source_root: Path, training_runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    try:
        corpus_authorities, corpus_sources = training._corpus_authority(
            corpus_manifest_paths
        )
    except training.TrainingExpansionError as exc:
        raise Generation3AuthorityError(str(exc)) from exc
    corpus_dimensions = _prior_corpus_dimensions(
        corpus_manifest_paths, corpus_sources
    )
    references_authority, reference_dimensions = _active_reference_authority(
        p3_runtime_root
    )
    active_sources = reference_dimensions["source_sha256"]
    training_authority, training_dimensions = _active_training_dimensions(
        active_sources=active_sources,
        source_root=source_root,
        training_runtime_root=training_runtime_root,
        corpus_manifest_paths=corpus_manifest_paths,
    )
    excluded = _empty_lineage_dimensions()
    _merge_lineage_dimensions(excluded, corpus_dimensions)
    _merge_lineage_dimensions(excluded, reference_dimensions)
    _merge_lineage_dimensions(excluded, training_dimensions)
    dimension_authority = {
        key: {
            "count": len(excluded[key]),
            "set_sha256": _canonical_hash(sorted(excluded[key])),
        }
        for key in LINEAGE_DIMENSIONS
    }
    authority = {
        "prior_corpus_authorities": corpus_authorities,
        "active_reference_authorities": references_authority,
        "active_training_authority": training_authority,
        "active_reference_authority_sha256": _canonical_hash(
            references_authority
        ),
        "prior_corpus_source_count": len(corpus_sources),
        "active_reference_source_count": len(active_sources),
        "excluded_source_count": len(excluded["source_sha256"]),
        "excluded_source_set_sha256": _canonical_hash(
            sorted(excluded["source_sha256"])
        ),
        "dimensional_lineage": dimension_authority,
    }
    authority["content_sha256"] = _canonical_hash(authority)
    return authority, excluded


def _evaluate(
    conversations: Sequence[Mapping[str, Any]], *, source_root: Path,
    corpus_manifest_paths: Sequence[Path], p3_runtime_root: Path,
    training_runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(conversations) != EXPECTED_CONVERSATION_COUNT:
        raise Generation3AuthorityError(
            "Generation-3 cohort requires exactly seven conversations."
        )
    lineage, excluded_sources = _source_lineage_authority(
        corpus_manifest_paths=corpus_manifest_paths,
        p3_runtime_root=p3_runtime_root,
        source_root=source_root,
        training_runtime_root=training_runtime_root,
    )
    safe_units: list[dict[str, Any]] = []
    private_units: list[dict[str, Any]] = []
    for value in conversations:
        if not isinstance(value, Mapping):
            raise Generation3AuthorityError(
                "Generation-3 conversation input must be an object."
            )
        try:
            safe, private = training._conversation_input(
                value,
                source_root=source_root,
                prior_sources=excluded_sources["source_sha256"],
            )
        except training.TrainingExpansionError as exc:
            raise Generation3AuthorityError(str(exc)) from exc
        source_sha = str(safe["source_sha256"])
        transcript = _read_object(Path(private["transcript_path"]))
        identities = _transcript_identities(transcript)
        safe_units.append(
            {
                **safe,
                **identities,
                "recording_id": (
                    "generation3-recording-"
                    + identities["recording_identity_sha256"][:24]
                ),
                "conversation_id": (
                    "generation3-conversation-"
                    + identities["conversation_identity_sha256"][:24]
                ),
                "derivative_id": (
                    "generation3-derivative-"
                    + identities["derivative_identity_sha256"][:24]
                ),
            }
        )
        private_units.append(private)
    candidate_dimensions = {
        key: {str(unit[key]) for unit in safe_units}
        for key in LINEAGE_DIMENSIONS
    }
    overlap_counts = {
        key: len(candidate_dimensions[key] & excluded_sources[key])
        for key in LINEAGE_DIMENSIONS
    }
    if any(
        len(candidate_dimensions[key]) != EXPECTED_CONVERSATION_COUNT
        for key in LINEAGE_DIMENSIONS
    ) or any(overlap_counts.values()):
        raise Generation3AuthorityError(
            "Generation-3 cohort membership is duplicate or non-disjoint."
        )
    ordered = sorted(
        zip(safe_units, private_units), key=lambda pair: pair[0]["source_sha256"]
    )
    safe_units = [pair[0] for pair in ordered]
    private_units = [pair[1] for pair in ordered]
    membership = {
        "conversations": safe_units,
        "conversation_count": EXPECTED_CONVERSATION_COUNT,
        "speaker_label_count": sum(len(unit["labels"]) for unit in safe_units),
        "total_duration_seconds": sum(
            float(unit["duration_seconds"]) for unit in safe_units
        ),
    }
    membership_sha256 = _canonical_hash(membership)
    core = {
        "schema_version": PREVIEW_SCHEMA,
        "status": "ready_for_independent_review",
        "reason_codes": [],
        "membership": membership,
        "membership_sha256": membership_sha256,
        "source_lineage_authority": lineage,
        "source_lineage_authority_sha256": lineage["content_sha256"],
        "source_overlap_count": 0,
        "dimensional_overlap_counts": overlap_counts,
        "gold_status": "not_frozen_operator_confirmation_required",
        "window_policy": {
            "minimum_seconds": 0.75,
            "maximum_seconds": 15.0,
            "maximum_windows_per_speaker_per_conversation": (
                MAXIMUM_WINDOWS_PER_SPEAKER_PER_CONVERSATION
            ),
            "exclude_overlap_and_speaker_change_regions": True,
            "exclude_mixed_or_unknown_gold": True,
            "preserve_original_timestamps": True,
            "same_frozen_window_set_for_every_candidate_unit": True,
        },
        "minimum_evidence_policy": {
            "evaluation_recordings": 7,
            "evaluation_conversations": 7,
            "minimum_gold_subjects": 5,
            "minimum_recurrent_enrolled_subjects": 2,
            "minimum_enrolled_conversations_per_subject": 2,
            "minimum_independent_same_person_subject_session_pairs": 4,
            "genuine_trials_per_model_method_unit": 20,
            "impostor_trials_per_model_method_unit": 100,
            "open_set_trials_per_model_method_unit": 20,
        },
        "action_vector": {
            "freeze_cohort_membership": False,
            "freeze_gold": False,
            "reveal_evaluation": False,
            "prepare_audio": False,
            "freeze_windows": False,
            "construct_exact_trial_child": False,
            "load_or_run_models": False,
            "score_trials": False,
            "calculate_metrics": False,
            "make_terminal_decision": False,
            "mutate_profiles_or_references": False,
            "enable_default_integration": False,
            "run_historical_reprocessing": False,
        },
        "privacy": {
            "contains_private_source_membership": True,
            "contains_paths": False,
            "contains_names_or_emails": False,
            "contains_subject_ids": True,
            "contains_gold_bodies": False,
            "contains_transcript_text": False,
            "contains_raw_audio": False,
            "contains_embeddings_or_vectors": False,
            "contains_biometric_scores": False,
        },
        "will_perform_external_write": False,
    }
    content_sha256 = _canonical_hash(core)
    return {
        **core,
        "preview_id": f"generation3-cohort-preview-{content_sha256[:24]}",
        "content_sha256": content_sha256,
    }, {"conversations": private_units}


def preview_generation3_cohort(
    conversations: Sequence[Mapping[str, Any]], *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    training_runtime_root: Path = DEFAULT_TRAINING_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Return a deterministic private preview without opening gold or audio."""
    preview, _ = _evaluate(
        conversations,
        source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
        p3_runtime_root=p3_runtime_root,
        training_runtime_root=training_runtime_root,
    )
    return preview


def portable_cohort_projection(preview: Mapping[str, Any]) -> dict[str, Any]:
    """Return the aggregate-only projection allowed outside private runtime."""
    membership = preview.get("membership")
    policy = preview.get("minimum_evidence_policy")
    actions = preview.get("action_vector")
    if (
        preview.get("schema_version") != PREVIEW_SCHEMA
        or preview.get("status") != "ready_for_independent_review"
        or not isinstance(membership, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(actions, Mapping)
    ):
        raise Generation3AuthorityError("Generation-3 preview is invalid.")
    return {
        "schema_version": "transcribe-audio.generation3-cohort-portable.v1",
        "status": preview["status"],
        "reason_codes": list(preview.get("reason_codes") or []),
        "preview_content_sha256": preview["content_sha256"],
        "membership_sha256": preview["membership_sha256"],
        "source_lineage_authority_sha256": preview[
            "source_lineage_authority_sha256"
        ],
        "conversation_count": membership["conversation_count"],
        "speaker_label_count": membership["speaker_label_count"],
        "minimum_evidence_policy_sha256": _canonical_hash(dict(policy)),
        "window_policy_sha256": _canonical_hash(dict(preview["window_policy"])),
        "action_vector": dict(actions),
        "contains_paths": False,
        "contains_names_or_emails": False,
        "contains_subject_ids": False,
        "contains_gold_bodies": False,
        "contains_source_membership": False,
        "contains_private_lineage": False,
        "contains_transcript_text": False,
        "contains_raw_audio": False,
        "contains_embeddings_or_vectors": False,
        "contains_biometric_scores": False,
        "will_perform_external_write": False,
    }


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent,
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise Generation3AuthorityError("Repository authority is unavailable.")
    return result.stdout.strip()


def _repository_authority() -> dict[str, Any]:
    status = _git(["status", "--porcelain=v1", "--untracked-files=normal"])
    upstream = _git(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
    if status or upstream.split() != ["0", "0"]:
        raise Generation3AuthorityError("Repository must be clean and upstream-even.")
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "training_dependency_sha256": sha256_file(Path(training.__file__).resolve()),
        "p3_dependency_sha256": sha256_file(Path(references.__file__).resolve()),
        "private_io_dependency_sha256": sha256_file(
            Path(derivatives.__file__).resolve()
        ),
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def _validate_repository_authority(value: Any) -> dict[str, Any]:
    required = {
        "commit", "module_sha256", "training_dependency_sha256",
        "p3_dependency_sha256", "private_io_dependency_sha256", "clean",
        "upstream_ahead", "upstream_behind",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Generation3AuthorityError("Frozen repository authority is invalid.")
    commit = str(value.get("commit") or "")
    current = _repository_authority()
    if (
        not COMMIT_RE.fullmatch(commit)
        or value.get("clean") is not True
        or value.get("upstream_ahead") != 0
        or value.get("upstream_behind") != 0
        or _git(["merge-base", "--is-ancestor", commit, current["commit"]])
    ):
        raise Generation3AuthorityError("Frozen repository authority is invalid.")
    for filename, field in (
        ("acoustic_generation3_authority.py", "module_sha256"),
        ("acoustic_training_expansion.py", "training_dependency_sha256"),
        ("acoustic_biometric_references.py", "p3_dependency_sha256"),
        ("acoustic_audio_derivatives.py", "private_io_dependency_sha256"),
    ):
        blob = subprocess.run(
            ["git", "show", f"{commit}:{filename}"],
            cwd=Path(__file__).resolve().parent, check=False, capture_output=True,
        )
        if (
            blob.returncode != 0
            or hashlib.sha256(blob.stdout).hexdigest() != value[field]
        ):
            raise Generation3AuthorityError("Frozen repository module drifted.")
    if any(
        current[field] != value[field]
        for field in required
        if field.endswith("sha256")
    ):
        raise Generation3AuthorityError("Current repository dependency drifted.")
    return dict(value)


def _paths(root: Path, authority_id: str = "") -> dict[str, Path]:
    selected = root.expanduser().absolute()
    base = selected / "cohort-authorities"
    authority = base / authority_id if authority_id else base
    return {
        "root": selected,
        "base": base,
        "authority": authority,
        "manifest": authority / "private-manifest.json",
        "receipt": authority / "receipt.json",
    }


def _existing_manifest(root: Path) -> Optional[Path]:
    base = _paths(root)["base"]
    if not base.exists():
        return None
    if not base.is_dir() or base.is_symlink():
        raise Generation3AuthorityError("Generation-3 authority root is invalid.")
    children = sorted(base.iterdir())
    if len(children) > 1:
        raise Generation3AuthorityError(
            "Multiple Generation-3 cohort authorities exist."
        )
    if not children:
        return None
    child = children[0]
    if (
        not child.is_dir()
        or child.is_symlink()
        or {item.name for item in child.iterdir()}
        != {"private-manifest.json", "receipt.json"}
    ):
        raise Generation3AuthorityError("Partial Generation-3 authority exists.")
    return child / "private-manifest.json"


def _receipt(preview: Mapping[str, Any], authority_id: str,
             manifest_sha256: str) -> dict[str, Any]:
    portable = portable_cohort_projection(preview)
    applied_actions = dict(portable["action_vector"])
    applied_actions["freeze_cohort_membership"] = True
    applied_actions["build_private_gold_review_packet"] = True
    return {
        **portable,
        "schema_version": RECEIPT_SCHEMA,
        "status": "applied_membership_only_gold_not_frozen",
        "authority_id": authority_id,
        "manifest_sha256": manifest_sha256,
        "action_vector": applied_actions,
        "mode": "0600",
    }


def _applied_actions(preview: Mapping[str, Any]) -> dict[str, bool]:
    actions = dict(preview["action_vector"])
    actions["freeze_cohort_membership"] = True
    actions["build_private_gold_review_packet"] = True
    return actions


def apply_generation3_cohort(
    reviewed_preview: Mapping[str, Any], *, expected_preview_content_sha256: str,
    conversations: Sequence[Mapping[str, Any]],
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    training_runtime_root: Path = DEFAULT_TRAINING_RUNTIME_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Freeze exact private membership without gold, preparation, or models."""
    preview, private = _evaluate(
        conversations, source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
        p3_runtime_root=p3_runtime_root,
        training_runtime_root=training_runtime_root,
    )
    if (
        dict(reviewed_preview) != preview
        or preview["content_sha256"] != expected_preview_content_sha256
    ):
        raise Generation3AuthorityError("Reviewed Generation-3 preview is stale.")
    existing = _existing_manifest(runtime_root)
    if existing is not None:
        return replay_generation3_cohort(
            existing, conversations=conversations, source_root=source_root,
            corpus_manifest_paths=corpus_manifest_paths,
            p3_runtime_root=p3_runtime_root,
            training_runtime_root=training_runtime_root,
            runtime_root=runtime_root,
        )
    repository = _repository_authority()
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "applied_membership_only_gold_not_frozen",
        "preview": preview,
        "private_inputs": private,
        "repository_authority": repository,
        "authorized_actions": _applied_actions(preview),
        "contains_private_paths": True,
        "contains_source_membership": True,
        "contains_gold_bodies": False,
        "will_perform_external_write": False,
    }
    content_sha256 = _canonical_hash(core)
    authority_id = f"generation3-cohort-{content_sha256[:24]}"
    paths = _paths(runtime_root, authority_id)
    ensure_private_tree(paths["root"], paths["authority"])
    manifest = {**core, "authority_id": authority_id, "content_sha256": content_sha256}
    write_immutable_private_json(paths["manifest"], manifest)
    manifest_sha256 = sha256_file(paths["manifest"])
    receipt = _receipt(preview, authority_id, manifest_sha256)
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_manifest_path": str(paths["manifest"]),
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_generation3_cohort(
    manifest_path: Path, *, conversations: Sequence[Mapping[str, Any]],
    source_root: Path = DEFAULT_SOURCE_ROOT,
    corpus_manifest_paths: Sequence[Path] = DEFAULT_CORPUS_MANIFESTS,
    p3_runtime_root: Path = DEFAULT_P3_RUNTIME_ROOT,
    training_runtime_root: Path = DEFAULT_TRAINING_RUNTIME_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Replay private membership and all exclusion/repository authorities."""
    root = runtime_root.expanduser().absolute()
    path = manifest_path.expanduser().absolute()
    require_private_file(path, root)
    manifest = _read_object(path)
    preview, private = _evaluate(
        conversations, source_root=source_root,
        corpus_manifest_paths=corpus_manifest_paths,
        p3_runtime_root=p3_runtime_root,
        training_runtime_root=training_runtime_root,
    )
    repository = _validate_repository_authority(manifest.get("repository_authority"))
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "applied_membership_only_gold_not_frozen",
        "preview": preview,
        "private_inputs": private,
        "repository_authority": repository,
        "authorized_actions": _applied_actions(preview),
        "contains_private_paths": True,
        "contains_source_membership": True,
        "contains_gold_bodies": False,
        "will_perform_external_write": False,
    }
    content_sha256 = _canonical_hash(core)
    authority_id = f"generation3-cohort-{content_sha256[:24]}"
    expected = {**core, "authority_id": authority_id, "content_sha256": content_sha256}
    if manifest != expected or path != _paths(root, authority_id)["manifest"]:
        raise Generation3AuthorityError("Generation-3 private manifest drifted.")
    receipt_path = _paths(root, authority_id)["receipt"]
    require_private_file(receipt_path, root)
    receipt = _read_object(receipt_path)
    expected_receipt = _receipt(preview, authority_id, sha256_file(path))
    if receipt != expected_receipt:
        raise Generation3AuthorityError("Generation-3 portable receipt drifted.")
    return {
        **receipt,
        "private_manifest_path": str(path),
        "private_receipt_path": str(receipt_path),
        "replay_schema_version": REPLAY_SCHEMA,
        "idempotent_replay": True,
    }
