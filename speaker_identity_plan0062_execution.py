"""Freeze the Plan 0062 contextual/acoustic cohort join for human review."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_context_join import join_contextual_identity
from speaker_identity_orchestration import AcousticEvidenceBundle, negative_action_vector


MANIFEST_SCHEMA = "transcribe-audio.plan0062-contextual-join-manifest.v1"
RECEIPT_SCHEMA = "transcribe-audio.plan0062-contextual-join-receipt.v1"
EXPECTED_DOCUMENTS = (
    "8232481d6076282d7a8e",
    "47ea79857aa1ac2d1d79",
    "92d2cd3ed6fc6c1275ca",
)
EXPECTED_SPEAKER_COUNTS = {
    "8232481d6076282d7a8e": 4,
    "47ea79857aa1ac2d1d79": 3,
    "92d2cd3ed6fc6c1275ca": 3,
}
EXPECTED_CONDITIONS = frozenset({"context_only", "acoustic_only", "combined"})


class Plan0062ExecutionError(ValueError):
    """Raised when the bounded Plan 0062 cohort or receipt drifts."""


@dataclass(frozen=True)
class ContextualJoinCase:
    document_id: str
    transcript_sha256: str
    identity_packet: Mapping[str, Any]
    identity_readout: Mapping[str, Any]
    acoustic_bundle: AcousticEvidenceBundle
    speaker_ref_bindings: Mapping[str, str]
    acoustic_subject_person_bindings: Mapping[str, str]
    evaluated_at: str
    run_references: Mapping[str, str]


def _fail(message: str) -> None:
    raise Plan0062ExecutionError(message)


def _paths(runtime_root: Path, activation_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p3-contextual-join-{activation_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def build_contextual_join_manifest(
    cases: Sequence[ContextualJoinCase],
    *,
    activation_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the exact private three-conversation join without applying it."""

    if len(activation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in activation_sha256
    ):
        _fail("Plan 0062 activation SHA-256 is invalid.")
    case_by_document = {case.document_id: case for case in cases}
    if len(case_by_document) != len(cases) or tuple(case_by_document) != EXPECTED_DOCUMENTS:
        _fail("Plan 0062 cohort order or membership drifted.")

    results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    proposal_counts: Counter[str] = Counter()
    suggestion_count = 0
    suggested_speaker_count = 0
    for case in cases:
        result = join_contextual_identity(
            document_id=case.document_id,
            transcript_sha256=case.transcript_sha256,
            identity_packet=case.identity_packet,
            identity_readout=case.identity_readout,
            acoustic_bundle=case.acoustic_bundle,
            speaker_ref_bindings=case.speaker_ref_bindings,
            acoustic_subject_person_bindings=case.acoustic_subject_person_bindings,
            evaluated_at=case.evaluated_at,
        )
        expected_speakers = EXPECTED_SPEAKER_COUNTS[case.document_id]
        if (
            len(result.review_outcomes) != expected_speakers
            or len(result.candidate_snapshots) != expected_speakers
            or len(result.evaluations) != expected_speakers * 3
        ):
            _fail("Plan 0062 per-conversation denominator drifted.")
        by_speaker: dict[str, set[str]] = {}
        for evaluation in result.evaluations:
            by_speaker.setdefault(evaluation.speaker_ref, set()).add(
                evaluation.condition
            )
            proposal_counts[f"{evaluation.condition}:{evaluation.outcome}"] += 1
            if evaluation.abstention_reason:
                reason_counts[evaluation.abstention_reason] += 1
        if any(conditions != EXPECTED_CONDITIONS for conditions in by_speaker.values()):
            _fail("Plan 0062 condition coverage drifted.")
        for outcome in result.review_outcomes:
            suggestion_count += len(outcome.suggestions)
            suggested_speaker_count += bool(outcome.suggestions)
        results.append(
            {
                "document_id": case.document_id,
                "speaker_count": expected_speakers,
                "identity_packet_sha256": canonical_artifact_hash(
                    dict(case.identity_packet)
                ),
                "identity_readout_sha256": canonical_artifact_hash(
                    dict(case.identity_readout)
                ),
                "speaker_ref_binding_sha256": canonical_artifact_hash(
                    dict(case.speaker_ref_bindings)
                ),
                "acoustic_subject_person_binding_sha256": canonical_artifact_hash(
                    dict(case.acoustic_subject_person_bindings)
                ),
                "run_references": dict(case.run_references),
                "join": result.to_dict(),
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "joined_pending_human_review",
        "activation_sha256": activation_sha256,
        "created_at": created_at,
        "recording_count": len(results),
        "speaker_count": sum(item["speaker_count"] for item in results),
        "evaluation_count": sum(
            len(item["join"]["evaluations"]) for item in results
        ),
        "suggestion_count": suggestion_count,
        "suggested_speaker_count": suggested_speaker_count,
        "outcome_counts": dict(sorted(proposal_counts.items())),
        "abstention_reason_counts": dict(sorted(reason_counts.items())),
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    if (
        manifest["recording_count"] != 3
        or manifest["speaker_count"] != 10
        or manifest["evaluation_count"] != 30
        or any(manifest["negative_actions"].values())
    ):
        _fail("Plan 0062 joined denominator or negative-action boundary drifted.")
    return manifest


def freeze_contextual_join_manifest(
    cases: Sequence[ContextualJoinCase],
    *,
    activation_sha256: str,
    created_at: str,
    runtime_root: Path,
) -> dict[str, Any]:
    """Write or exactly replay the private Plan 0062 P3 manifest and receipt."""

    paths = _paths(runtime_root, activation_sha256)
    if paths["receipt"].exists():
        return replay_contextual_join_manifest(
            activation_sha256=activation_sha256,
            runtime_root=runtime_root,
        )
    if paths["run"].exists():
        _fail("Plan 0062 P3 directory exists without a terminal receipt.")
    manifest = build_contextual_join_manifest(
        cases,
        activation_sha256=activation_sha256,
        created_at=created_at,
    )
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": manifest["status"],
        "activation_sha256": activation_sha256,
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": manifest["recording_count"],
        "speaker_count": manifest["speaker_count"],
        "evaluation_count": manifest["evaluation_count"],
        "suggestion_count": manifest["suggestion_count"],
        "suggested_speaker_count": manifest["suggested_speaker_count"],
        "outcome_counts": manifest["outcome_counts"],
        "abstention_reason_counts": manifest["abstention_reason_counts"],
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_contextual_join_manifest(
    *, activation_sha256: str, runtime_root: Path
) -> dict[str, Any]:
    """Verify and return the already-frozen Plan 0062 P3 receipt."""

    paths = _paths(runtime_root, activation_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "joined_pending_human_review"
        or manifest.get("activation_sha256") != activation_sha256
        or int(manifest.get("recording_count") or 0) != 3
        or int(manifest.get("speaker_count") or 0) != 10
        or int(manifest.get("evaluation_count") or 0) != 30
        or any((manifest.get("negative_actions") or {}).values())
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("live_mutation_count") != 0
        or receipt.get("negative_actions_preserved") is not True
    ):
        _fail("Plan 0062 P3 replay binding is invalid.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }
