from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import acoustic_plan0057
import provenance_config
from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
from speaker_identity_evidence_execution import (
    _acoustic_review_for_case,
    _contact_rows,
    _context_for_case,
    _model_versions,
    _source_case,
    _stable_id,
    adapt_acoustic_review,
    normalize_explicit_provider_scopes,
)
from speaker_identity_orchestration import (
    AcousticEvidenceBundle,
    AcousticSpeakerEvidence,
    CanonicalCandidate,
    CanonicalCandidateSnapshot,
    ContextEvidenceBundle,
    EvidenceLineage,
    EvidenceScope,
    IdentityCaseEvaluation,
    IdentityEvidenceFactor,
    IdentityOrchestrationError,
    _fail,
    _quick_check,
    _sqlite_backup,
    confidence_cap,
    negative_action_vector,
    replay_activation,
    replay_shadow_store,
    validate_bundle_bindings,
)


ACTIVATION_VERSION = "transcribe-audio.plan0060-activation.v1"
P2A_MANIFEST_VERSION = "transcribe-audio.plan0060-p2a-acoustic-manifest.v1"
P2A_RECEIPT_VERSION = "transcribe-audio.plan0060-p2a-acoustic-receipt.v1"
P2B_MANIFEST_VERSION = "transcribe-audio.plan0060-p2b-context-manifest.v1"
P2B_RECEIPT_VERSION = "transcribe-audio.plan0060-p2b-context-receipt.v1"
P3_MANIFEST_VERSION = "transcribe-audio.plan0060-p3-blinded-join-manifest.v1"
P3_RECEIPT_VERSION = "transcribe-audio.plan0060-p3-blinded-join-receipt.v1"
JOIN_POLICY_VERSION = "plan0060-conservative-shadow-join-v1"
EXPECTED_RECORDINGS = 3
EXPECTED_SPEAKERS = 10


def _lane_paths(runtime_root: Path, activation_sha256: str, lane: str) -> dict[str, Path]:
    if lane not in {"p2a-acoustic", "p2b-context"}:
        _fail("invalid_plan0060_lane", "Plan 0060 lane is not recognized.")
    root = runtime_root.expanduser().absolute()
    run = root / f"{lane}-{activation_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "sources": run / "sources",
        "context_root": run / "context-shadow",
        "context_database": run / "context-shadow" / "transcripts.sqlite3",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _activation_paths(runtime_root: Path, activation_sha256: str) -> tuple[Path, Path]:
    run = runtime_root.expanduser().absolute() / f"activation-{activation_sha256[:24]}"
    return run / "private-manifest.json", run / "receipt.json"


def replay_plan0060_activation(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    root = runtime_root.expanduser().absolute()
    manifest_path, receipt_path = _activation_paths(root, activation_sha256)
    require_private_file(manifest_path, root)
    require_private_file(receipt_path, root)
    manifest = read_private_object(manifest_path)
    receipt = read_private_object(receipt_path)
    if (
        manifest.get("schema_version") != ACTIVATION_VERSION
        or manifest.get("status") != "activated_pre_implementation"
        or canonical_artifact_hash(manifest) != activation_sha256
        or receipt.get("content_sha256") != activation_sha256
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or int(receipt.get("recording_count") or 0) != EXPECTED_RECORDINGS
        or int(receipt.get("speaker_ref_count") or 0) != EXPECTED_SPEAKERS
        or receipt.get("inherited_replay_verified") is not True
        or receipt.get("negative_actions_preserved") is not True
        or any((manifest.get("negative_actions") or {}).values())
    ):
        _fail("plan0060_activation_invalid", "Plan 0060 activation binding is invalid.")
    return {
        "manifest": manifest,
        "receipt": receipt,
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "idempotent_replay": True,
    }


def _shared_inputs(
    *,
    runtime_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    plan0060 = replay_plan0060_activation(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
    )
    activation = replay_activation(
        plan0059_activation_sha256,
        runtime_root=plan0059_runtime_root,
    )
    p1 = replay_shadow_store(
        activation_content_sha256=plan0059_activation_sha256,
        runtime_root=plan0059_runtime_root,
    )
    inherited = plan0060["manifest"].get("inherited") or {}
    if (
        inherited.get("plan0059_activation_content_sha256")
        != plan0059_activation_sha256
        or inherited.get("plan0059_p1_content_sha256") != p1.get("content_sha256")
        or activation.get("recording_count") != EXPECTED_RECORDINGS
        or activation.get("speaker_ref_count") != EXPECTED_SPEAKERS
    ):
        _fail("plan0059_inheritance_drift", "Inherited Plan 0059 evidence drifted.")

    p1_manifest_path = Path(str(p1["manifest_path"]))
    require_private_file(p1_manifest_path, plan0059_runtime_root)
    p1_manifest = read_private_object(p1_manifest_path)
    active_shadow_root = Path(str(p1["active_shadow_root"]))
    active_database = active_shadow_root / "transcripts.sqlite3"
    if _quick_check(active_database) != "ok":
        _fail("plan0059_shadow_integrity", "Inherited P1 shadow failed quick_check.")

    activation_manifest_path = (
        plan0059_runtime_root.expanduser().absolute()
        / f"activation-{plan0059_activation_sha256[:24]}"
        / "private-manifest.json"
    )
    require_private_file(activation_manifest_path, plan0059_runtime_root)
    activation_manifest = read_private_object(activation_manifest_path)
    cohort = (activation_manifest.get("cohort") or {}).get("members") or []
    cases = [
        _source_case(active_database, str(item.get("document_id") or ""))
        for item in cohort
    ]
    if len(cases) != EXPECTED_RECORDINGS or sum(
        len(case["speaker_refs"]) for case in cases
    ) != EXPECTED_SPEAKERS:
        _fail("plan0060_denominator_drift", "Plan 0060 inherited denominator drifted.")

    authority_root = plan0057_authority_manifest.parents[2]
    require_private_file(plan0057_authority_manifest, authority_root)
    prior = read_private_object(plan0057_authority_manifest)
    authority = prior.get("preview") if isinstance(prior.get("preview"), Mapping) else {}
    if (
        set(authority.get("allowlisted_subject_ids") or [])
        != set(acoustic_plan0057.ALLOWLISTED_SUBJECT_IDS)
        or (authority.get("local_runtime") or {}).get("network_required") is not False
        or sha256_file(plan0057_authority_manifest)
        != (plan0060["manifest"].get("acoustic") or {}).get(
            "authority_manifest_sha256"
        )
    ):
        _fail("plan0060_acoustic_authority_invalid", "Acoustic authority drifted.")

    identity_state = acoustic_plan0057._current_identity_state()
    if identity_state.get("snapshot_sha256") != (
        plan0060["manifest"].get("live") or {}
    ).get("identity_state_sha256"):
        _fail("plan0060_identity_state_drift", "Identity/profile/reference state drifted.")
    return {
        "plan0060": plan0060,
        "activation": activation,
        "p1": p1,
        "p1_manifest": p1_manifest,
        "active_shadow_root": active_shadow_root,
        "active_database": active_database,
        "cases": cases,
        "authority": authority,
        "identity_state": identity_state,
    }


def execute_acoustic_lane(
    *,
    runtime_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2a-acoustic")
    if paths["receipt"].exists():
        return replay_acoustic_lane(
            runtime_root=runtime_root,
            activation_sha256=activation_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_plan0060_p2a", "P2A directory exists without a receipt.")
    shared = _shared_inputs(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
        plan0059_runtime_root=plan0059_runtime_root,
        plan0059_activation_sha256=plan0059_activation_sha256,
        plan0057_authority_manifest=plan0057_authority_manifest,
    )
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["sources"])
    created_at = str(shared["plan0060"]["manifest"]["activated_at"])
    identity_before = shared["identity_state"]
    results = []
    for ordinal, case in enumerate(shared["cases"], start=1):
        source_root = paths["sources"] / (
            f"{ordinal:02d}-{hashlib.sha256(case['document_id'].encode()).hexdigest()[:16]}"
        )
        ensure_private_tree(paths["root"], source_root)
        review = _acoustic_review_for_case(
            case,
            authority=shared["authority"],
            source_root=source_root,
            identity_state=identity_before,
            created_at=created_at,
        )
        bundle = adapt_acoustic_review(
            review,
            conversation_id=case["conversation_id"],
            recording_id=case["recording_id"],
            document_id=case["document_id"],
            transcript_sha256=case["transcript_sha256"],
            model_versions=_model_versions(shared["authority"]),
            created_at=created_at,
        )
        results.append(
            {
                "document_id": case["document_id"],
                "recording_id": case["recording_id"],
                "speaker_ref_count": len(case["speaker_refs"]),
                "execution_content_sha256": review["execution_content_sha256"],
                "bundle_id": bundle.bundle_id,
                "bundle": asdict(bundle),
            }
        )
    identity_after = acoustic_plan0057._current_identity_state()
    if identity_after != identity_before:
        _fail("plan0060_identity_state_mutation", "P2A changed identity state.")
    manifest = {
        "schema_version": P2A_MANIFEST_VERSION,
        "status": "acoustic_lane_complete",
        "activation_sha256": activation_sha256,
        "created_at": created_at,
        "recording_count": len(results),
        "speaker_ref_count": sum(item["speaker_ref_count"] for item in results),
        "identity_state_before": identity_before,
        "identity_state_after": identity_after,
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    if (
        manifest["recording_count"] != EXPECTED_RECORDINGS
        or manifest["speaker_ref_count"] != EXPECTED_SPEAKERS
    ):
        _fail("plan0060_p2a_incomplete", "P2A denominator is incomplete.")
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P2A_RECEIPT_VERSION,
        "status": "acoustic_lane_complete",
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "bundle_count": EXPECTED_RECORDINGS,
        "identity_state_unchanged": True,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def replay_acoustic_lane(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2a-acoustic")
    manifest, receipt = _read_lane(paths)
    if (
        manifest.get("schema_version") != P2A_MANIFEST_VERSION
        or manifest.get("activation_sha256") != activation_sha256
        or manifest.get("recording_count") != EXPECTED_RECORDINGS
        or manifest.get("speaker_ref_count") != EXPECTED_SPEAKERS
        or receipt.get("bundle_count") != EXPECTED_RECORDINGS
        or manifest.get("identity_state_before") != manifest.get("identity_state_after")
    ):
        _fail("plan0060_p2a_replay_invalid", "P2A receipt binding is invalid.")
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": True}


def execute_context_lane(
    *,
    runtime_root: Path,
    state_root: Path,
    activation_sha256: str,
    plan0059_runtime_root: Path,
    plan0059_activation_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2b-context")
    if paths["receipt"].exists():
        return replay_context_lane(
            runtime_root=runtime_root,
            activation_sha256=activation_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_plan0060_p2b", "P2B directory exists without a receipt.")
    shared = _shared_inputs(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
        plan0059_runtime_root=plan0059_runtime_root,
        plan0059_activation_sha256=plan0059_activation_sha256,
        plan0057_authority_manifest=plan0057_authority_manifest,
    )
    ensure_private_tree(paths["root"], paths["run"])
    ensure_private_tree(paths["root"], paths["context_root"])
    _sqlite_backup(shared["active_database"], paths["context_database"])
    if _quick_check(paths["context_database"]) != "ok":
        _fail("plan0060_context_shadow_integrity", "P2B copy failed quick_check.")
    contacts = _contact_rows(paths["context_database"])
    resolved = normalize_explicit_provider_scopes(
        provenance_config.speaker_preprocessing_source_configs_from_provenance(
            state_root=state_root
        )
    )
    expected_scope = shared["plan0060"]["manifest"].get("context") or {}
    scope_hash = hashlib.sha256(
        (
            json.dumps(
                resolved.get("retrieval_sources") or [],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if (
        len(resolved.get("retrieval_sources") or [])
        != int(expected_scope.get("provider_scope_count") or -1)
        or scope_hash != expected_scope.get("provider_scope_sha256")
    ):
        _fail("plan0060_provider_scope_drift", "P2B provider scopes drifted.")

    p1_manifest = shared["p1_manifest"]
    reconciliation = p1_manifest.get("reconciliation_preview") or {}
    watermark = str((p1_manifest.get("active_shadow") or {}).get("table_counts_sha256") or "")
    created_at = str(shared["plan0060"]["manifest"]["activated_at"])
    run_id = _stable_id("plan0060-p2b", activation_sha256)
    results = []
    for case in shared["cases"]:
        context_bundle, candidate_snapshot, private_context = _context_for_case(
            case,
            active_shadow_root=paths["context_root"],
            resolved=resolved,
            contacts=contacts,
            reconciliation=reconciliation,
            projection_watermark=watermark,
            created_at=created_at,
            run_id=run_id,
        )
        results.append(
            {
                "document_id": case["document_id"],
                "recording_id": case["recording_id"],
                "speaker_ref_count": len(case["speaker_refs"]),
                "context_bundle_id": context_bundle.bundle_id,
                "context_bundle": asdict(context_bundle),
                "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                "candidate_snapshot": asdict(candidate_snapshot),
                "private_context": private_context,
            }
        )
    identity_after = acoustic_plan0057._current_identity_state()
    if identity_after != shared["identity_state"]:
        _fail("plan0060_identity_state_mutation", "P2B changed identity state.")
    manifest = {
        "schema_version": P2B_MANIFEST_VERSION,
        "status": "context_lane_complete",
        "activation_sha256": activation_sha256,
        "created_at": created_at,
        "run_id": run_id,
        "recording_count": len(results),
        "speaker_ref_count": sum(item["speaker_ref_count"] for item in results),
        "provider_scope_count": len(resolved.get("retrieval_sources") or []),
        "provider_scope_sha256": scope_hash,
        "context_shadow_sha256": sha256_file(paths["context_database"]),
        "context_shadow_quick_check": _quick_check(paths["context_database"]),
        "identity_state_before": shared["identity_state"],
        "identity_state_after": identity_after,
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    if (
        manifest["recording_count"] != EXPECTED_RECORDINGS
        or manifest["speaker_ref_count"] != EXPECTED_SPEAKERS
    ):
        _fail("plan0060_p2b_incomplete", "P2B denominator is incomplete.")
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P2B_RECEIPT_VERSION,
        "status": "context_lane_complete",
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "bundle_count": EXPECTED_RECORDINGS,
        "candidate_snapshot_count": EXPECTED_RECORDINGS,
        "provider_scope_count": manifest["provider_scope_count"],
        "provider_failure_count": sum(
            int(item["private_context"]["provider_failure_count"]) for item in results
        ),
        "included_evidence_count": sum(
            int(item["private_context"]["included_evidence_count"]) for item in results
        ),
        "identity_state_unchanged": True,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def replay_context_lane(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _lane_paths(runtime_root, activation_sha256, "p2b-context")
    manifest, receipt = _read_lane(paths)
    if (
        manifest.get("schema_version") != P2B_MANIFEST_VERSION
        or manifest.get("activation_sha256") != activation_sha256
        or manifest.get("recording_count") != EXPECTED_RECORDINGS
        or manifest.get("speaker_ref_count") != EXPECTED_SPEAKERS
        or receipt.get("bundle_count") != EXPECTED_RECORDINGS
        or receipt.get("candidate_snapshot_count") != EXPECTED_RECORDINGS
        or manifest.get("context_shadow_sha256") != sha256_file(paths["context_database"])
        or manifest.get("context_shadow_quick_check") != "ok"
        or _quick_check(paths["context_database"]) != "ok"
        or manifest.get("identity_state_before") != manifest.get("identity_state_after")
    ):
        _fail("plan0060_p2b_replay_invalid", "P2B receipt binding is invalid.")
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": True}


def _read_lane(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if (
        receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("negative_actions_preserved") is not True
        or receipt.get("live_mutation_count") != 0
        or any((manifest.get("negative_actions") or {}).values())
    ):
        _fail("plan0060_lane_replay_invalid", "Plan 0060 lane receipt is invalid.")
    return manifest, receipt


def _join_paths(runtime_root: Path, activation_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p3-blinded-join-{activation_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _lineage(raw: Mapping[str, Any]) -> EvidenceLineage:
    return EvidenceLineage(
        evidence_id=str(raw["evidence_id"]),
        source_record_id=str(raw["source_record_id"]),
        independence_group=str(raw["independence_group"]),
        source_type=str(raw["source_type"]),
        source_event_at=str(raw["source_event_at"]),
        observed_at=str(raw["observed_at"]),
        retrieved_at=str(raw["retrieved_at"]),
        content_sha256=str(raw["content_sha256"]),
        derived_from_evidence_ids=tuple(raw.get("derived_from_evidence_ids") or ()),
        proposed_by_current_evaluation=bool(
            raw.get("proposed_by_current_evaluation", False)
        ),
    )


def _acoustic_bundle(raw: Mapping[str, Any]) -> AcousticEvidenceBundle:
    return AcousticEvidenceBundle(
        conversation_id=str(raw["conversation_id"]),
        recording_id=str(raw["recording_id"]),
        document_id=str(raw["document_id"]),
        speaker_refs=tuple(raw["speaker_refs"]),
        source_media_sha256=str(raw["source_media_sha256"]),
        transcript_sha256=str(raw["transcript_sha256"]),
        execution_sha256=str(raw["execution_sha256"]),
        identity_state_sha256=str(raw["identity_state_sha256"]),
        model_versions=tuple(tuple(item) for item in raw["model_versions"]),
        created_at=str(raw["created_at"]),
        evidence=tuple(
            AcousticSpeakerEvidence(
                speaker_ref=str(item["speaker_ref"]),
                disposition=str(item["disposition"]),
                acoustic_subject_id=item.get("acoustic_subject_id"),
                score=float(item["score"]),
                confidence_band=str(item["confidence_band"]),
                supporting_unit_count=int(item["supporting_unit_count"]),
                opposing_unit_count=int(item["opposing_unit_count"]),
                insufficient_unit_count=int(item["insufficient_unit_count"]),
                evidence_ids=tuple(item["evidence_ids"]),
            )
            for item in raw["evidence"]
        ),
        lineage=tuple(_lineage(item) for item in raw["lineage"]),
        negative_actions=dict(raw["negative_actions"]),
    )


def _context_bundle(raw: Mapping[str, Any]) -> ContextEvidenceBundle:
    return ContextEvidenceBundle(
        conversation_id=str(raw["conversation_id"]),
        recording_id=str(raw["recording_id"]),
        document_id=str(raw["document_id"]),
        speaker_refs=tuple(raw["speaker_refs"]),
        transcript_sha256=str(raw["transcript_sha256"]),
        scopes=tuple(
            EvidenceScope(
                source_type=str(item["source_type"]),
                source_profile=str(item["source_profile"]),
                account_id=str(item["account_id"]),
                tenant_id=str(item["tenant_id"]),
                capabilities=tuple(item["capabilities"]),
                as_of=str(item["as_of"]),
                max_records=int(item["max_records"]),
                max_characters=int(item["max_characters"]),
                max_per_source=int(item["max_per_source"]),
                max_provider_calls=int(item["max_provider_calls"]),
                max_relationship_hops=int(item["max_relationship_hops"]),
            )
            for item in raw["scopes"]
        ),
        retrieval_version=str(raw["retrieval_version"]),
        ranking_version=str(raw["ranking_version"]),
        policy_version=str(raw["policy_version"]),
        included_evidence_ids=tuple(raw["included_evidence_ids"]),
        excluded_evidence=tuple(tuple(item) for item in raw["excluded_evidence"]),
        warnings=tuple(raw["warnings"]),
        source_failures=tuple(tuple(item) for item in raw["source_failures"]),
        lineage=tuple(_lineage(item) for item in raw["lineage"]),
        negative_actions=dict(raw["negative_actions"]),
    )


def _candidate_snapshot(raw: Mapping[str, Any]) -> CanonicalCandidateSnapshot:
    return CanonicalCandidateSnapshot(
        conversation_id=str(raw["conversation_id"]),
        document_id=str(raw["document_id"]),
        as_of=str(raw["as_of"]),
        schema_version=str(raw["schema_version"]),
        projection_watermark=str(raw["projection_watermark"]),
        candidates=tuple(
            CanonicalCandidate(
                person_id=str(item["person_id"]),
                source_record_ids=tuple(item["source_record_ids"]),
                evidence_ids=tuple(item["evidence_ids"]),
                score=float(item["score"]),
                accepted_relationship_evidence_ids=tuple(
                    item.get("accepted_relationship_evidence_ids") or ()
                ),
            )
            for item in raw["candidates"]
        ),
        lineage=tuple(_lineage(item) for item in raw["lineage"]),
        negative_actions=dict(raw["negative_actions"]),
    )


def _factor(
    *, factor_type: str, score: float, evidence_ids: Iterable[str], lineage: Iterable[EvidenceLineage]
) -> IdentityEvidenceFactor:
    selected_ids = tuple(dict.fromkeys(str(value) for value in evidence_ids))
    groups_by_id = {item.evidence_id: item.independence_group for item in lineage}
    groups = tuple(
        dict.fromkeys(groups_by_id[value] for value in selected_ids if value in groups_by_id)
    )
    if not selected_ids or not groups:
        _fail("plan0060_factor_lineage_missing", "Evaluation factor lacks lineage.")
    return IdentityEvidenceFactor(
        factor_type=factor_type,
        score=score,
        evidence_ids=selected_ids,
        independence_groups=groups,
    )


def _evaluation_from_dict(raw: Mapping[str, Any]) -> IdentityCaseEvaluation:
    return IdentityCaseEvaluation(
        evaluation_id=str(raw["evaluation_id"]),
        conversation_id=str(raw["conversation_id"]),
        recording_id=str(raw["recording_id"]),
        document_id=str(raw["document_id"]),
        speaker_ref=str(raw["speaker_ref"]),
        condition=str(raw["condition"]),
        acoustic_bundle_id=raw.get("acoustic_bundle_id"),
        context_bundle_id=raw.get("context_bundle_id"),
        candidate_snapshot_id=str(raw["candidate_snapshot_id"]),
        candidate_person_ids=tuple(raw["candidate_person_ids"]),
        factors=tuple(
            IdentityEvidenceFactor(
                factor_type=str(item["factor_type"]),
                score=float(item["score"]),
                evidence_ids=tuple(item["evidence_ids"]),
                independence_groups=tuple(item["independence_groups"]),
            )
            for item in raw["factors"]
        ),
        outcome=str(raw["outcome"]),
        proposed_person_id=raw.get("proposed_person_id"),
        alternative_person_ids=tuple(raw["alternative_person_ids"]),
        contradiction_evidence_ids=tuple(raw["contradiction_evidence_ids"]),
        base_confidence=float(raw["base_confidence"]),
        capped_confidence=float(raw["capped_confidence"]),
        confidence_cap_reasons=tuple(raw["confidence_cap_reasons"]),
        abstention_reason=raw.get("abstention_reason"),
        source_failures=tuple(tuple(item) for item in raw["source_failures"]),
        policy_version=str(raw["policy_version"]),
        evaluated_at=str(raw["evaluated_at"]),
        negative_actions=dict(raw["negative_actions"]),
    )


def execute_blinded_join(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _join_paths(runtime_root, activation_sha256)
    if paths["receipt"].exists():
        return replay_blinded_join(
            runtime_root=runtime_root,
            activation_sha256=activation_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_plan0060_p3", "P3 directory exists without a receipt.")
    activation = replay_plan0060_activation(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
    )
    acoustic_receipt = replay_acoustic_lane(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
    )
    context_receipt = replay_context_lane(
        runtime_root=runtime_root,
        activation_sha256=activation_sha256,
    )
    acoustic_manifest = read_private_object(Path(acoustic_receipt["manifest_path"]))
    context_manifest = read_private_object(Path(context_receipt["manifest_path"]))
    acoustic_by_document = {
        str(item["document_id"]): item for item in acoustic_manifest["results"]
    }
    context_by_document = {
        str(item["document_id"]): item for item in context_manifest["results"]
    }
    if acoustic_by_document.keys() != context_by_document.keys():
        _fail("plan0060_p3_document_mismatch", "P2 lane documents differ.")

    evaluated_at = str(activation["manifest"]["activated_at"])
    evaluations = []
    for document_id in acoustic_by_document:
        acoustic = _acoustic_bundle(acoustic_by_document[document_id]["bundle"])
        context = _context_bundle(context_by_document[document_id]["context_bundle"])
        candidates = _candidate_snapshot(
            context_by_document[document_id]["candidate_snapshot"]
        )
        validate_bundle_bindings(acoustic, context, candidates)
        candidate_ids = tuple(item.person_id for item in candidates.candidates)
        candidate_score = max((item.score for item in candidates.candidates), default=0.0)
        context_evidence_ids = tuple(
            dict.fromkeys(
                [
                    *context.included_evidence_ids,
                    *(
                        evidence_id
                        for candidate in candidates.candidates
                        for evidence_id in candidate.evidence_ids
                    ),
                ]
            )
        )
        context_lineage = (*context.lineage, *candidates.lineage)
        for acoustic_row in acoustic.evidence:
            acoustic_factor = _factor(
                factor_type="acoustic",
                score=acoustic_row.score,
                evidence_ids=acoustic_row.evidence_ids,
                lineage=acoustic.lineage,
            )
            context_factor = _factor(
                factor_type="context",
                score=candidate_score,
                evidence_ids=context_evidence_ids,
                lineage=context_lineage,
            )
            for condition in ("context_only", "acoustic_only", "combined"):
                factors = (
                    (context_factor,)
                    if condition == "context_only"
                    else (acoustic_factor,)
                    if condition == "acoustic_only"
                    else (acoustic_factor, context_factor)
                )
                base_confidence = candidate_score if condition != "acoustic_only" else 0.0
                cap_reasons = (
                    ("partial_provider_failure",)
                    if condition != "acoustic_only" and context.source_failures
                    else ()
                )
                capped_confidence, normalized_reasons = confidence_cap(
                    base_confidence, cap_reasons
                )
                abstention_reason = (
                    "acoustic_subject_not_mapped_to_canonical_person"
                    if condition == "acoustic_only"
                    else "context_candidates_not_speaker_specific"
                    if condition == "context_only"
                    else "pillar_identity_link_missing"
                )
                evaluation = IdentityCaseEvaluation(
                    evaluation_id=_stable_id(
                        "evaluation",
                        activation_sha256,
                        document_id,
                        acoustic_row.speaker_ref,
                        condition,
                    ),
                    conversation_id=acoustic.conversation_id,
                    recording_id=acoustic.recording_id,
                    document_id=document_id,
                    speaker_ref=acoustic_row.speaker_ref,
                    condition=condition,
                    acoustic_bundle_id=(
                        acoustic.bundle_id if condition != "context_only" else None
                    ),
                    context_bundle_id=(
                        context.bundle_id if condition != "acoustic_only" else None
                    ),
                    candidate_snapshot_id=candidates.snapshot_id,
                    candidate_person_ids=candidate_ids,
                    factors=factors,
                    outcome="abstained",
                    proposed_person_id=None,
                    alternative_person_ids=candidate_ids,
                    contradiction_evidence_ids=(),
                    base_confidence=base_confidence,
                    capped_confidence=capped_confidence,
                    confidence_cap_reasons=normalized_reasons,
                    abstention_reason=abstention_reason,
                    source_failures=(
                        context.source_failures if condition != "acoustic_only" else ()
                    ),
                    policy_version=JOIN_POLICY_VERSION,
                    evaluated_at=evaluated_at,
                    negative_actions=negative_action_vector(),
                )
                evaluations.append(asdict(evaluation))

    condition_counts = {
        condition: sum(item["condition"] == condition for item in evaluations)
        for condition in ("context_only", "acoustic_only", "combined")
    }
    if len(evaluations) != 30 or set(condition_counts.values()) != {10}:
        _fail("plan0060_p3_incomplete", "P3 blinded denominator is incomplete.")
    manifest = {
        "schema_version": P3_MANIFEST_VERSION,
        "status": "blinded_join_complete_gold_sealed",
        "activation_sha256": activation_sha256,
        "acoustic_receipt_content_sha256": acoustic_receipt["content_sha256"],
        "context_receipt_content_sha256": context_receipt["content_sha256"],
        "policy_version": JOIN_POLICY_VERSION,
        "evaluated_at": evaluated_at,
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "evaluation_count": len(evaluations),
        "condition_counts": condition_counts,
        "human_gold_read": False,
        "human_decision_count": 0,
        "evaluations": evaluations,
        "negative_actions": negative_action_vector(),
    }
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P3_RECEIPT_VERSION,
        "status": "blinded_join_complete_gold_sealed",
        "content_sha256": canonical_artifact_hash(manifest),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": EXPECTED_RECORDINGS,
        "speaker_ref_count": EXPECTED_SPEAKERS,
        "evaluation_count": 30,
        "context_only_count": 10,
        "acoustic_only_count": 10,
        "combined_count": 10,
        "proposal_count": 0,
        "abstention_count": 30,
        "human_gold_read": False,
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": False}


def replay_blinded_join(
    *, runtime_root: Path, activation_sha256: str
) -> dict[str, Any]:
    paths = _join_paths(runtime_root, activation_sha256)
    manifest, receipt = _read_lane(paths)
    evaluations = manifest.get("evaluations") or []
    for raw in evaluations:
        _evaluation_from_dict(raw)
    if (
        manifest.get("schema_version") != P3_MANIFEST_VERSION
        or manifest.get("activation_sha256") != activation_sha256
        or len(evaluations) != 30
        or manifest.get("condition_counts")
        != {"context_only": 10, "acoustic_only": 10, "combined": 10}
        or manifest.get("human_gold_read") is not False
        or manifest.get("human_decision_count") != 0
        or receipt.get("human_gold_read") is not False
    ):
        _fail("plan0060_p3_replay_invalid", "P3 blinded receipt is invalid.")
    return {**receipt, "manifest_path": str(paths["manifest"]), "idempotent_replay": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute Plan 0060 independent evidence lanes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("execute-acoustic", "execute-context"):
        lane = subparsers.add_parser(command)
        lane.add_argument("--runtime-root", type=Path, required=True)
        lane.add_argument("--activation-sha256", required=True)
        lane.add_argument("--plan0059-runtime-root", type=Path, required=True)
        lane.add_argument("--plan0059-activation-sha256", required=True)
        lane.add_argument("--plan0057-authority-manifest", type=Path, required=True)
        if command == "execute-context":
            lane.add_argument("--state-root", type=Path, required=True)
    for command in ("replay-acoustic", "replay-context"):
        replay = subparsers.add_parser(command)
        replay.add_argument("--runtime-root", type=Path, required=True)
        replay.add_argument("--activation-sha256", required=True)
    for command in ("execute-join", "replay-join"):
        join = subparsers.add_parser(command)
        join.add_argument("--runtime-root", type=Path, required=True)
        join.add_argument("--activation-sha256", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "execute-acoustic":
            result = execute_acoustic_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
                plan0059_runtime_root=args.plan0059_runtime_root,
                plan0059_activation_sha256=args.plan0059_activation_sha256,
                plan0057_authority_manifest=args.plan0057_authority_manifest,
            )
        elif args.command == "execute-context":
            result = execute_context_lane(
                runtime_root=args.runtime_root,
                state_root=args.state_root,
                activation_sha256=args.activation_sha256,
                plan0059_runtime_root=args.plan0059_runtime_root,
                plan0059_activation_sha256=args.plan0059_activation_sha256,
                plan0057_authority_manifest=args.plan0057_authority_manifest,
            )
        elif args.command == "replay-acoustic":
            result = replay_acoustic_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
        elif args.command == "replay-context":
            result = replay_context_lane(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
        elif args.command == "execute-join":
            result = execute_blinded_join(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
        else:
            result = replay_blinded_join(
                runtime_root=args.runtime_root,
                activation_sha256=args.activation_sha256,
            )
    except IdentityOrchestrationError as exc:
        print(json.dumps({"status": "error", "reason_code": exc.reason_code, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
