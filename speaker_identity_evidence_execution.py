from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

import acoustic_plan0056_execution as acoustic_execution
import acoustic_plan0056_runner as acoustic_runner
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
from conversation_identity_policy import build_identity_evidence_policy
from conversation_identity_retrieval import prepare_identity_evidence
from speaker_identity_orchestration import (
    AcousticEvidenceBundle,
    AcousticSpeakerEvidence,
    CanonicalCandidate,
    CanonicalCandidateSnapshot,
    ContextEvidenceBundle,
    EvidenceLineage,
    EvidenceScope,
    IdentityOrchestrationError,
    _document_artifact,
    _fail,
    _quick_check,
    _readonly_connection,
    _sqlite_backup,
    negative_action_vector,
    replay_activation,
    replay_shadow_store,
    validate_bundle_bindings,
)


P2_MANIFEST_VERSION = "transcribe-audio.plan0059-evidence-manifest.v1"
P2_RECEIPT_VERSION = "transcribe-audio.plan0059-evidence-receipt.v1"
REFINE_AUDIT_VERSION = "transcribe-audio.plan0059-refine-audit.v1"
ACOUSTIC_ADAPTER_VERSION = "plan0059-acoustic-shadow-adapter-v1"
CONTEXT_ADAPTER_VERSION = "plan0059-context-canonical-adapter-v1"
CONTEXT_POLICY_VERSION = "plan0059-bounded-context-policy-v1"
EXPECTED_PROVIDER_BUDGETS = {
    "max_records": 20,
    "max_characters": 12_000,
    "max_per_source": 5,
    "max_provider_calls": 4,
    "max_relationship_hops": 1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{uuid5(NAMESPACE_URL, value)}"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _p2_paths(runtime_root: Path, activation_content_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"p2-evidence-{activation_content_sha256[:24]}"
    return {
        "root": root,
        "run": run,
        "sources": run / "sources",
        "context_root": run / "context-shadow",
        "context_database": run / "context-shadow" / "transcripts.sqlite3",
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
    }


def _model_versions(authority: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    runtime = authority.get("local_runtime") or {}
    return (
        ("adapter", ACOUSTIC_ADAPTER_VERSION),
        ("runtime_sha256", str(runtime.get("runtime_sha256") or "")),
        (
            "diarization_model_sha256",
            str((runtime.get("diarization_model") or {}).get("file_set_sha256") or ""),
        ),
        (
            "transcription_model_sha256",
            str((runtime.get("transcription_model") or {}).get("file_set_sha256") or ""),
        ),
        ("threshold_units_sha256", _json_hash(authority.get("threshold_units") or [])),
    )


def adapt_acoustic_review(
    review: Mapping[str, Any],
    *,
    conversation_id: str,
    recording_id: str,
    document_id: str,
    transcript_sha256: str,
    model_versions: tuple[tuple[str, str], ...],
    created_at: str,
) -> AcousticEvidenceBundle:
    """Normalize validated source-bound acoustic review evidence into P0."""

    rows = review.get("rows")
    if review.get("status") != "complete_pending_human_review" or not isinstance(rows, list):
        _fail("acoustic_review_incomplete", "Acoustic review evidence is incomplete.")
    speaker_refs = tuple(str(item.get("speaker_ref") or "") for item in rows)
    evidence_rows: list[AcousticSpeakerEvidence] = []
    lineage: list[EvidenceLineage] = []
    for raw in rows:
        speaker_ref = str(raw.get("speaker_ref") or "")
        supporting = int(raw.get("supporting_unit_count") or 0)
        opposing = int(raw.get("opposing_unit_count") or 0)
        insufficient = max(0, 9 - supporting - opposing)
        evidence_id = _stable_id(
            "evidence-acoustic",
            document_id,
            speaker_ref,
            review.get("execution_content_sha256"),
        )
        evidence_rows.append(
            AcousticSpeakerEvidence(
                speaker_ref=speaker_ref,
                disposition=str(raw.get("disposition") or ""),
                acoustic_subject_id=raw.get("subject_id"),
                score=min(1.0, max(0.0, supporting / 9.0)),
                confidence_band=str(raw.get("confidence_band") or ""),
                supporting_unit_count=supporting,
                opposing_unit_count=opposing,
                insufficient_unit_count=insufficient,
                evidence_ids=(evidence_id,),
            )
        )
        lineage.append(
            EvidenceLineage(
                evidence_id=evidence_id,
                source_record_id=_stable_id("recording-speaker", recording_id, speaker_ref),
                independence_group="acoustic-model-consensus",
                source_type="acoustic_shadow_review",
                source_event_at=created_at,
                observed_at=created_at,
                retrieved_at=created_at,
                content_sha256=_json_hash(dict(raw)),
            )
        )
    return AcousticEvidenceBundle(
        conversation_id=conversation_id,
        recording_id=recording_id,
        document_id=document_id,
        speaker_refs=speaker_refs,
        source_media_sha256=str(review.get("source_media_sha256") or ""),
        transcript_sha256=transcript_sha256,
        execution_sha256=str(review.get("execution_content_sha256") or ""),
        identity_state_sha256=str(review.get("identity_state_sha256") or ""),
        model_versions=model_versions,
        created_at=created_at,
        evidence=tuple(evidence_rows),
        lineage=tuple(lineage),
        negative_actions=negative_action_vector(),
    )


def normalize_explicit_provider_scopes(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Fill provider account/tenant bindings only from operator-authored context."""

    contexts = {
        str(item.get("source_id") or ""): item
        for item in resolved.get("source_contexts") or []
        if isinstance(item, Mapping)
    }
    normalized = dict(resolved)
    retrieval_sources: list[dict[str, Any]] = []
    for raw in resolved.get("retrieval_sources") or []:
        source = dict(raw)
        source_id = str(source.get("source_id") or "")
        context = contexts.get(source_id) or {}
        owner = context.get("owner") if isinstance(context.get("owner"), Mapping) else {}
        account_id = str(source.get("account_id") or owner.get("id") or "")
        tenant_id = str(source.get("tenant_id") or context.get("relationship_scope") or "")
        if not account_id or not tenant_id:
            _fail(
                "provider_scope_incomplete",
                f"Provider source {source_id!r} lacks an operator-authored account or tenant binding.",
            )
        source["account_id"] = account_id
        source["tenant_id"] = tenant_id
        capabilities = [
            str(value) for value in source.get("evidence_capabilities") or []
        ]
        identity_capability = "people" if source.get("provider_kind") == "gws" else "contacts"
        if identity_capability not in capabilities:
            _fail(
                "provider_capability_missing",
                f"Provider source {source_id!r} lacks its identity lookup capability.",
            )
        source["evidence_capabilities"] = [identity_capability]
        retrieval_sources.append(source)
    normalized["retrieval_sources"] = retrieval_sources
    return normalized


def transcript_speaker_timeline(
    utterances: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Convert stored millisecond offsets to bounded acoustic seconds."""

    label_order: list[str] = []
    timeline: list[dict[str, Any]] = []
    for utterance in utterances:
        label = str(utterance.get("speaker") or "").strip()
        if not label:
            continue
        if label not in label_order:
            label_order.append(label)
        timeline.append(
            {
                "speaker": label,
                "start": float(utterance.get("start") or 0.0) / 1000.0,
                "end": float(utterance.get("end") or 0.0) / 1000.0,
            }
        )
    return timeline, tuple(label_order)


def normalize_provider_lineage(item: Any) -> EvidenceLineage:
    """Keep provider-native record handles private behind stable host IDs."""

    snapshot = item.snapshot
    return EvidenceLineage(
        evidence_id=snapshot.evidence_id,
        source_record_id=_stable_id(
            "provider-record",
            snapshot.source_profile_id,
            snapshot.provider_kind,
            snapshot.source_record_id or snapshot.evidence_id,
        ),
        independence_group=snapshot.independence_group_id,
        source_type=snapshot.source_type,
        source_event_at=snapshot.source_event_at,
        observed_at=snapshot.observed_at,
        retrieved_at=snapshot.retrieved_at,
        content_sha256=snapshot.content_hash,
    )


def _source_case(database_path: Path, document_id: str) -> dict[str, Any]:
    transcript_path, transcript_sha256 = _document_artifact(database_path, document_id)
    with _readonly_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT d.json_payload, d.generated_at, b.original_path, b.stored_path,
                   b.sha256 AS source_media_sha256
            FROM documents d
            JOIN document_blobs db ON db.document_id=d.id
            JOIN blobs b ON b.id=db.blob_id AND b.role='source_recording'
            WHERE d.id=?
            """,
            (document_id,),
        ).fetchone()
        knowledge = connection.execute(
            """
            SELECT kc.id AS conversation_id, kr.id AS recording_id
            FROM knowledge_recordings kr
            JOIN knowledge_conversations kc ON kc.id=kr.conversation_id
            WHERE kr.transcript_document_id=?
            """,
            (document_id,),
        ).fetchone()
    if row is None or knowledge is None:
        _fail("p2_case_missing", "A frozen document lacks source or knowledge bindings.")
    payload = json.loads(str(row["json_payload"]))
    if not isinstance(payload, dict):
        _fail("invalid_transcript_payload", "A frozen transcript payload is invalid.")
    media_candidates = [
        str(payload.get("source_media_path") or ""),
        str(row["original_path"] or ""),
        str(row["stored_path"] or ""),
    ]
    media_path = next(
        (Path(value).expanduser().resolve() for value in media_candidates if value and Path(value).expanduser().is_file()),
        None,
    )
    if media_path is None or sha256_file(media_path) != str(row["source_media_sha256"]):
        _fail("source_media_unavailable", "Frozen source media is unavailable or hash-mismatched.")
    utterances = payload.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        _fail("speaker_denominator_missing", "Frozen transcript utterances are missing.")
    timeline, label_order = transcript_speaker_timeline(
        [item for item in utterances if isinstance(item, Mapping)]
    )
    return {
        "document_id": document_id,
        "conversation_id": str(knowledge["conversation_id"]),
        "recording_id": str(knowledge["recording_id"]),
        "conversation_key": str(payload.get("conversation_id") or knowledge["conversation_id"]),
        "transcript_path": transcript_path,
        "transcript_sha256": transcript_sha256,
        "source_media_path": media_path,
        "source_media_sha256": str(row["source_media_sha256"]),
        "generated_at": str(row["generated_at"]),
        "payload": payload,
        "timeline": timeline,
        "speaker_labels": label_order,
        "speaker_refs": tuple(f"SPEAKER_{index}" for index in range(1, len(label_order) + 1)),
    }


def _acoustic_review_for_case(
    case: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    source_root: Path,
    identity_state: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    run = source_root / "acoustic"
    paths = {
        "root": source_root.parent.parent,
        "run": run,
        "pcm": run / "source-pcm.wav",
        "diarization": run / "transcript-timeline.json",
        "clips": run / "clips",
        "transcripts": run / "transcripts",
        "p1": run / "preparation-p1",
        "p2": run / "preparation-p2",
        "matrices": run / "matrices",
        "proposals": run / "proposals.json",
    }
    ensure_private_tree(paths["root"], paths["run"])
    acoustic_runner._decode_private_pcm(Path(case["source_media_path"]), paths["pcm"])
    selected = acoustic_runner.select_review_segments(
        case["timeline"],
        minimum_turn_seconds=authority["review_clip_policy"]["minimum_turn_seconds"],
        maximum_turn_seconds=authority["review_clip_policy"]["maximum_turn_seconds"],
        maximum_turns_per_speaker=authority["review_clip_policy"]["maximum_turns_per_speaker"],
        target_seconds_per_speaker=authority["review_clip_policy"]["target_seconds_per_speaker"],
        minimum_usable_seconds_per_speaker=authority["review_clip_policy"]["minimum_usable_seconds_per_speaker"],
    )
    write_immutable_private_json(paths["diarization"], {"timeline": case["timeline"], "selected": selected})
    bindings = []
    for speaker_ref, segments in selected.items():
        clip = acoustic_runner._write_speaker_clip(paths["pcm"], paths["clips"] / f"{speaker_ref}.wav", segments)
        bindings.append({"speaker_ref": speaker_ref, **clip})
    snapshots = acoustic_execution.DEFAULT_WHISPER_CACHE_ROOT.expanduser().absolute() / "snapshots"
    model_snapshots = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(model_snapshots) != 1:
        _fail("ambiguous_transcription_model", "The local transcription model snapshot is ambiguous.")
    transcripts = acoustic_runner._transcribe_clips(bindings, model_snapshot=model_snapshots[0])
    ensure_private_tree(paths["root"], paths["transcripts"])
    write_immutable_private_json(paths["transcripts"] / "speaker-transcripts.json", {"rows": transcripts})
    scoring_preview = {
        "content_sha256": str(authority.get("content_sha256") or ""),
        "p0_authority": {"allowlisted_subject_ids": authority["allowlisted_subject_ids"]},
        "threshold_units": authority["threshold_units"],
    }
    matrices = acoustic_runner._score_matrices(bindings, preview=scoring_preview, paths=paths)
    proposals = acoustic_execution.proposals_from_matrices(
        matrices,
        expected_speaker_refs=list(selected),
        allowlisted_subject_ids=authority["allowlisted_subject_ids"],
    )
    by_ref = {str(item["speaker_ref"]): dict(item) for item in proposals["proposals"]}
    rows = []
    for speaker_ref in case["speaker_refs"]:
        if speaker_ref in by_ref:
            rows.append(by_ref[speaker_ref])
        else:
            rows.append(
                {
                    "speaker_ref": speaker_ref,
                    "disposition": "abstain",
                    "subject_id": None,
                    "confidence_band": "none",
                    "supporting_unit_count": 0,
                    "supporting_candidate_family_count": 0,
                    "opposing_unit_count": 0,
                    "rationale": "Insufficient bounded source speech for acoustic evaluation.",
                }
            )
    core = {
        "status": "complete_pending_human_review",
        "document_id": case["document_id"],
        "conversation_key": case["conversation_key"],
        "source_media_sha256": case["source_media_sha256"],
        "execution_content_sha256": _json_hash(
            {
                "document_id": case["document_id"],
                "matrix_hashes": [item["content_sha256"] for item in matrices],
                "adapter": ACOUSTIC_ADAPTER_VERSION,
            }
        ),
        "identity_state_sha256": str(identity_state["snapshot_sha256"]),
        "speaker_count": len(rows),
        "rows": rows,
        "created_at": created_at,
    }
    write_immutable_private_json(paths["proposals"], core)
    return core


def _contact_rows(database_path: Path) -> list[dict[str, Any]]:
    with _readonly_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT id, label, email, external_ref FROM contacts ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def _case_identity_text(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    values: list[str] = []
    for field_name in ("summary", "description", "organizer"):
        value = event.get(field_name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Mapping):
            values.extend(str(value.get(key) or "") for key in ("name", "email"))
    for collection_name in ("participants", "attendees"):
        for item in event.get(collection_name) or []:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, Mapping):
                values.extend(str(item.get(key) or "") for key in ("name", "email", "display_name"))
    return "\n".join(values).casefold()


def _prepared_terms(payload: Mapping[str, Any]) -> tuple[str, ...]:
    event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    values: list[str] = []
    for collection_name in ("participants", "attendees"):
        for item in event.get(collection_name) or []:
            if isinstance(item, str):
                values.append(item.strip())
            elif isinstance(item, Mapping):
                values.extend(
                    str(item.get(key) or "").strip()
                    for key in ("name", "email", "display_name")
                )
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return tuple(result[:12])


def _context_for_case(
    case: Mapping[str, Any],
    *,
    active_shadow_root: Path,
    resolved: Mapping[str, Any],
    contacts: Sequence[Mapping[str, Any]],
    reconciliation: Mapping[str, Any],
    projection_watermark: str,
    created_at: str,
    run_id: str,
) -> tuple[ContextEvidenceBundle, CanonicalCandidateSnapshot, dict[str, Any]]:
    preview_by_contact = {
        str(item.get("contact_id") or ""): item
        for item in reconciliation.get("candidates") or []
        if isinstance(item, Mapping)
    }
    private_candidates = []
    person_ids = []
    for contact in contacts:
        preview = preview_by_contact.get(str(contact.get("id") or ""))
        if not preview:
            _fail("candidate_reconciliation_missing", "P1 candidate reconciliation is incomplete.")
        preview_person_id = str(preview.get("candidate_person_id") or "")
        person_id = str(uuid5(NAMESPACE_URL, f"plan0059-canonical-candidate:{preview_person_id}"))
        person_ids.append(person_id)
        private_candidates.append(
            {
                "person_id": person_id,
                "preview_person_id": preview_person_id,
                "contact_id": str(contact.get("id") or ""),
                "label": str(contact.get("label") or ""),
                "email": str(contact.get("email") or "").casefold(),
                "status": "separate_review_only",
            }
        )

    request_id = str(uuid5(NAMESPACE_URL, f"plan0059-context:{run_id}:{case['document_id']}"))
    policy_build = build_identity_evidence_policy(
        resolved,
        requested_at=created_at,
        request_id=request_id,
        run_id=run_id,
        environment=os.environ,
        prepared_query_terms=_prepared_terms(case["payload"]),
        max_records=EXPECTED_PROVIDER_BUDGETS["max_records"],
        max_characters=EXPECTED_PROVIDER_BUDGETS["max_characters"],
        max_per_source=EXPECTED_PROVIDER_BUDGETS["max_per_source"],
        max_provider_calls=EXPECTED_PROVIDER_BUDGETS["max_provider_calls"],
    )
    policy = replace(
        policy_build.policy,
        prepared_person_ids=tuple(person_ids),
        max_relationship_hops=EXPECTED_PROVIDER_BUDGETS["max_relationship_hops"],
    )
    prepared = prepare_identity_evidence(
        str(case["conversation_id"]),
        speaker_labels=tuple(case["speaker_refs"]),
        policy=policy,
        root=active_shadow_root,
    )
    scopes = tuple(
        EvidenceScope(
            source_type=str(raw.get("provider_kind") or ""),
            source_profile=str(raw.get("source_profile_id") or ""),
            account_id=str(raw.get("account_id") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            capabilities=tuple(str(value) for value in raw.get("evidence_capabilities") or []),
            as_of=prepared.request.as_of,
            max_records=EXPECTED_PROVIDER_BUDGETS["max_records"],
            max_characters=EXPECTED_PROVIDER_BUDGETS["max_characters"],
            max_per_source=EXPECTED_PROVIDER_BUDGETS["max_per_source"],
            max_provider_calls=EXPECTED_PROVIDER_BUDGETS["max_provider_calls"],
            max_relationship_hops=EXPECTED_PROVIDER_BUDGETS["max_relationship_hops"],
        )
        for raw in resolved.get("retrieval_sources") or []
    )
    if not scopes:
        _fail("provider_scope_empty", "No explicit read-only context source is configured.")

    ranked_by_id = {item.snapshot.evidence_id: item for item in prepared.evidence}
    context_lineage = tuple(normalize_provider_lineage(item) for item in prepared.evidence)
    included = tuple(
        item.snapshot.evidence_id
        for item in prepared.evidence
        if item.disposition == "included"
    )
    excluded = tuple(
        (item.snapshot.evidence_id, item.reason_code)
        for item in prepared.evidence
        if item.disposition != "included"
    )
    source_failures = tuple(
        (
            str(item.get("adapter_id") or "provider"),
            str(item.get("reason_code") or "provider_failure"),
            False,
        )
        for item in prepared.source_failures
    )
    context_bundle = ContextEvidenceBundle(
        conversation_id=str(case["conversation_id"]),
        recording_id=str(case["recording_id"]),
        document_id=str(case["document_id"]),
        speaker_refs=tuple(case["speaker_refs"]),
        transcript_sha256=str(case["transcript_sha256"]),
        scopes=scopes,
        retrieval_version=prepared.request.retrieval_version,
        ranking_version=prepared.request.ranking_version,
        policy_version=CONTEXT_POLICY_VERSION,
        included_evidence_ids=included,
        excluded_evidence=excluded,
        warnings=tuple(sorted(set((*policy_build.warnings, *prepared.warnings)))),
        source_failures=source_failures,
        lineage=context_lineage,
        negative_actions=negative_action_vector(),
    )

    identity_text = _case_identity_text(case["payload"])
    candidate_rows: list[CanonicalCandidate] = []
    candidate_lineage: list[EvidenceLineage] = []
    for private in private_candidates:
        local_evidence_id = _stable_id(
            "evidence-local-contact", private["contact_id"], projection_watermark
        )
        candidate_lineage.append(
            EvidenceLineage(
                evidence_id=local_evidence_id,
                source_record_id=_stable_id("source-local-contact", private["contact_id"]),
                independence_group="local-contact-shadow",
                source_type="local_contact_shadow",
                source_event_at=created_at,
                observed_at=created_at,
                retrieved_at=created_at,
                content_sha256=_json_hash(private),
            )
        )
        exact_context_match = bool(
            (private["email"] and private["email"] in identity_text)
            or (private["label"] and private["label"].casefold() in identity_text)
        )
        matched_provider_ids = []
        for evidence_id, ranked in ranked_by_id.items():
            searchable = json.dumps(
                {
                    "snippet": ranked.snapshot.snippet,
                    "metadata": ranked.snapshot.structured_metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).casefold()
            if private["email"] and private["email"] in searchable:
                matched_provider_ids.append(evidence_id)
        evidence_ids = tuple([local_evidence_id, *sorted(set(matched_provider_ids))])
        score = 0.95 if exact_context_match else (0.80 if matched_provider_ids else 0.20)
        candidate_rows.append(
            CanonicalCandidate(
                person_id=private["person_id"],
                source_record_ids=(
                    _stable_id("source-local-contact", private["contact_id"]),
                ),
                evidence_ids=evidence_ids,
                score=score,
            )
        )
    candidate_snapshot = CanonicalCandidateSnapshot(
        conversation_id=str(case["conversation_id"]),
        document_id=str(case["document_id"]),
        as_of=prepared.request.as_of,
        schema_version="plan0059-canonical-person-shadow-v1",
        projection_watermark=projection_watermark,
        candidates=tuple(candidate_rows),
        lineage=tuple(candidate_lineage),
        negative_actions=negative_action_vector(),
    )
    return context_bundle, candidate_snapshot, {
        "private_candidates": private_candidates,
        "request_id": prepared.request.request_id,
        "request_sha256": canonical_artifact_hash(asdict(prepared.request)),
        "persisted_bundle_id": prepared.persisted_bundle.bundle_id,
        "persisted_bundle_sha256": prepared.persisted_bundle.content_hash,
        "provider_adapter_count": len(policy.provider_adapters),
        "provider_failure_count": len(source_failures),
        "included_evidence_count": len(included),
        "excluded_evidence_count": len(excluded),
        "calendar_candidate_count": len(prepared.calendar_candidates),
        "candidate_count": len(candidate_rows),
    }


def execute_evidence_adapters(
    *,
    runtime_root: Path,
    state_root: Path,
    activation_content_sha256: str,
    plan0057_authority_manifest: Path,
) -> dict[str, Any]:
    activation = replay_activation(activation_content_sha256, runtime_root=runtime_root)
    p1 = replay_shadow_store(
        activation_content_sha256=activation_content_sha256,
        runtime_root=runtime_root,
    )
    paths = _p2_paths(runtime_root, activation_content_sha256)
    if paths["receipt"].exists():
        return replay_evidence_adapters(
            runtime_root=runtime_root,
            activation_content_sha256=activation_content_sha256,
        )
    if paths["run"].exists():
        _fail("incomplete_p2_run", "P2 run directory exists without a terminal receipt.")
    ensure_private_tree(paths["root"], paths["run"])

    require_private_file(plan0057_authority_manifest, plan0057_authority_manifest.parent.parent)
    prior = read_private_object(plan0057_authority_manifest)
    authority = prior.get("preview") if isinstance(prior.get("preview"), Mapping) else {}
    if (
        set(authority.get("allowlisted_subject_ids") or [])
        != set(acoustic_plan0057.ALLOWLISTED_SUBJECT_IDS)
        or authority.get("local_runtime", {}).get("network_required") is not False
    ):
        _fail("acoustic_authority_invalid", "Closed acoustic authority is not safe for P2A.")

    p1_manifest_path = Path(str(p1["manifest_path"]))
    require_private_file(p1_manifest_path, paths["root"])
    p1_manifest = read_private_object(p1_manifest_path)
    active_shadow_root = Path(str(p1["active_shadow_root"]))
    active_database = active_shadow_root / "transcripts.sqlite3"
    _sqlite_backup(active_database, paths["context_database"])
    if _quick_check(paths["context_database"]) != "ok":
        _fail("context_shadow_integrity", "P2 context shadow failed quick_check.")
    contacts = _contact_rows(paths["context_database"])
    reconciliation = p1_manifest.get("reconciliation_preview") or {}
    projection_watermark = str(
        (p1_manifest.get("active_shadow") or {}).get("table_counts_sha256") or ""
    )
    activation_manifest_path = (
        paths["root"]
        / f"activation-{activation_content_sha256[:24]}"
        / "private-manifest.json"
    )
    require_private_file(activation_manifest_path, paths["root"])
    activation_manifest = read_private_object(activation_manifest_path)
    cohort = (activation_manifest.get("cohort") or {}).get("members") or []
    cases = [
        _source_case(paths["context_database"], str(item.get("document_id") or ""))
        for item in cohort
    ]
    if sum(len(case["speaker_refs"]) for case in cases) != int(activation.get("speaker_ref_count") or -1):
        _fail("speaker_denominator_drift", "P2 speaker denominator differs from activation.")

    resolved = provenance_config.speaker_preprocessing_source_configs_from_provenance(
        state_root=state_root
    )
    resolved = normalize_explicit_provider_scopes(resolved)
    created_at = _utc_now()
    run_id = _stable_id("plan0059-p2", activation_content_sha256)
    identity_before = acoustic_plan0057._current_identity_state()
    results = []
    for ordinal, case in enumerate(cases, start=1):
        source_root = paths["sources"] / f"{ordinal:02d}-{hashlib.sha256(case['document_id'].encode()).hexdigest()[:16]}"
        ensure_private_tree(paths["root"], source_root)
        review = _acoustic_review_for_case(
            case,
            authority=authority,
            source_root=source_root,
            identity_state=identity_before,
            created_at=created_at,
        )
        acoustic_bundle = adapt_acoustic_review(
            review,
            conversation_id=case["conversation_id"],
            recording_id=case["recording_id"],
            document_id=case["document_id"],
            transcript_sha256=case["transcript_sha256"],
            model_versions=_model_versions(authority),
            created_at=created_at,
        )
        context_bundle, candidate_snapshot, private_context = _context_for_case(
            case,
            active_shadow_root=paths["context_root"],
            resolved=resolved,
            contacts=contacts,
            reconciliation=reconciliation,
            projection_watermark=projection_watermark,
            created_at=created_at,
            run_id=run_id,
        )
        validate_bundle_bindings(acoustic_bundle, context_bundle, candidate_snapshot)
        results.append(
            {
                "document_id": case["document_id"],
                "conversation_id": case["conversation_id"],
                "recording_id": case["recording_id"],
                "speaker_ref_count": len(case["speaker_refs"]),
                "acoustic_bundle": asdict(acoustic_bundle),
                "acoustic_bundle_id": acoustic_bundle.bundle_id,
                "context_bundle": asdict(context_bundle),
                "context_bundle_id": context_bundle.bundle_id,
                "candidate_snapshot": asdict(candidate_snapshot),
                "candidate_snapshot_id": candidate_snapshot.snapshot_id,
                "private_context": private_context,
            }
        )

    identity_after = acoustic_plan0057._current_identity_state()
    if identity_after != identity_before:
        _fail("identity_state_mutation", "Identity or acoustic profile state changed during P2.")
    provider_failures = sum(
        item["private_context"]["provider_failure_count"] for item in results
    )
    included_evidence = sum(
        item["private_context"]["included_evidence_count"] for item in results
    )
    manifest = {
        "schema_version": P2_MANIFEST_VERSION,
        "status": "evidence_adapters_complete",
        "activation_content_sha256": activation_content_sha256,
        "p1_content_sha256": p1["content_sha256"],
        "created_at": created_at,
        "run_id": run_id,
        "acoustic_authority_sha256": sha256_file(plan0057_authority_manifest),
        "provider_source_context_sha256": canonical_artifact_hash(resolved.get("source_contexts") or []),
        "provider_scope_count": len(resolved.get("retrieval_sources") or []),
        "context_shadow_sha256": sha256_file(paths["context_database"]),
        "context_shadow_quick_check": _quick_check(paths["context_database"]),
        "identity_state_before": identity_before,
        "identity_state_after": identity_after,
        "results": results,
        "negative_actions": negative_action_vector(),
    }
    content_sha256 = canonical_artifact_hash(manifest)
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": P2_RECEIPT_VERSION,
        "status": "evidence_adapters_complete",
        "activation_content_sha256": activation_content_sha256,
        "content_sha256": content_sha256,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "recording_count": len(results),
        "speaker_ref_count": sum(item["speaker_ref_count"] for item in results),
        "acoustic_bundle_count": len(results),
        "context_bundle_count": len(results),
        "candidate_snapshot_count": len(results),
        "canonical_candidate_count": sum(
            item["private_context"]["candidate_count"] for item in results
        ),
        "provider_scope_count": manifest["provider_scope_count"],
        "provider_failure_count": provider_failures,
        "included_context_evidence_count": included_evidence,
        "identity_state_unchanged": True,
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


def replay_evidence_adapters(
    *, runtime_root: Path, activation_content_sha256: str
) -> dict[str, Any]:
    paths = _p2_paths(runtime_root, activation_content_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    results = manifest.get("results") or []
    if (
        manifest.get("activation_content_sha256") != activation_content_sha256
        or receipt.get("content_sha256") != canonical_artifact_hash(manifest)
        or receipt.get("manifest_sha256") != sha256_file(paths["manifest"])
        or receipt.get("recording_count") != len(results)
        or receipt.get("speaker_ref_count")
        != sum(int(item.get("speaker_ref_count") or 0) for item in results)
        or receipt.get("identity_state_unchanged") is not True
        or receipt.get("negative_actions_preserved") is not True
        or manifest.get("context_shadow_quick_check") != "ok"
        or _quick_check(paths["context_database"]) != "ok"
    ):
        _fail("p2_replay_invalid", "P2 evidence receipt binding is invalid.")
    return {
        **receipt,
        "manifest_path": str(paths["manifest"]),
        "receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def _systemd_service_state(unit: str) -> dict[str, Any]:
    process = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "NRestarts",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in process.stdout.splitlines():
        key, _, value = line.partition("=")
        values[key] = value
    return {
        "unit": unit,
        "active_state": values.get("ActiveState", ""),
        "sub_state": values.get("SubState", ""),
        "restarts": int(values.get("NRestarts") or 0),
    }


def freeze_refine_audit(
    *,
    runtime_root: Path,
    live_store_root: Path,
    activation_content_sha256: str,
    expected_identity_state_sha256: str,
    audited_at: str,
) -> dict[str, Any]:
    """Freeze the governed refine result after the bounded P2 attempts."""

    activation = replay_activation(activation_content_sha256, runtime_root=runtime_root)
    p1 = replay_shadow_store(
        activation_content_sha256=activation_content_sha256,
        runtime_root=runtime_root,
    )
    root = runtime_root.expanduser().absolute()
    run = root / f"terminal-refine-{activation_content_sha256[:24]}"
    manifest_path = run / "private-manifest.json"
    receipt_path = run / "receipt.json"
    if receipt_path.exists():
        require_private_file(manifest_path, root)
        require_private_file(receipt_path, root)
        manifest = read_private_object(manifest_path)
        receipt = read_private_object(receipt_path)
        if (
            receipt.get("content_sha256") != canonical_artifact_hash(manifest)
            or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        ):
            _fail("refine_audit_replay_invalid", "Terminal refine audit binding is invalid.")
        return {**receipt, "idempotent_replay": True}

    failure_roots = (
        root / "p2-failed-attempt-1-utterance-ms",
        root / "p2a-complete-p2b-failed-attempt-1-person-id",
        root / "p2b-failed-attempt-2-provider-record-id",
    )
    if any(not path.is_dir() for path in failure_roots):
        _fail("refine_attempt_missing", "A bounded P2 attempt archive is missing.")
    proposal_paths = [
        path
        for failure_root in failure_roots
        for path in failure_root.glob("sources/*/acoustic/proposals.json")
    ]
    execution_hashes = {
        str(read_private_object(path).get("execution_content_sha256") or "")
        for path in proposal_paths
    }
    if len(proposal_paths) != 2 or len(execution_hashes) != 1:
        _fail("acoustic_recompute_drift", "Successful bounded acoustic recomputations diverged.")
    latest_proposal = read_private_object(proposal_paths[-1])
    if int(latest_proposal.get("speaker_count") or 0) != 4:
        _fail("acoustic_partial_denominator_invalid", "Partial acoustic denominator is invalid.")

    context_database = failure_roots[-1] / "context-shadow" / "transcripts.sqlite3"
    if _quick_check(context_database) != "ok":
        _fail("failed_context_integrity", "Failed-attempt context shadow is not intact.")
    with _readonly_connection(context_database) as connection:
        context_counts = {
            "requests": int(connection.execute("SELECT COUNT(*) FROM knowledge_retrieval_requests").fetchone()[0]),
            "snapshots": int(connection.execute("SELECT COUNT(*) FROM knowledge_evidence_snapshots").fetchone()[0]),
            "bundles": int(connection.execute("SELECT COUNT(*) FROM knowledge_evidence_bundles").fetchone()[0]),
            "bundle_items": int(connection.execute("SELECT COUNT(*) FROM knowledge_evidence_bundle_items").fetchone()[0]),
            "people": int(connection.execute("SELECT COUNT(*) FROM knowledge_people").fetchone()[0]),
            "source_records": int(connection.execute("SELECT COUNT(*) FROM knowledge_source_records").fetchone()[0]),
            "external_identities": int(connection.execute("SELECT COUNT(*) FROM knowledge_external_identities").fetchone()[0]),
        }

    live_database = live_store_root.expanduser().resolve() / "transcripts.sqlite3"
    with _readonly_connection(live_database) as connection:
        live_counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("documents", "contacts", "speaker_assignments")
        }
        live_status = {
            "quick_check": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
            "knowledge_status": (
                "absent"
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_store_state'"
                ).fetchone()
                is None
                else "present"
            ),
        }
    expected_live_counts = {
        "documents": 466,
        "contacts": 2,
        "speaker_assignments": 3,
    }
    if live_counts != expected_live_counts or live_status != {
        "quick_check": "ok",
        "knowledge_status": "absent",
    }:
        _fail("live_state_drift", "Live transcript state changed during Plan 0059.")
    identity_state = acoustic_plan0057._current_identity_state()
    if identity_state.get("snapshot_sha256") != expected_identity_state_sha256:
        _fail("identity_state_drift", "Live identity/profile/reference state changed.")
    services = tuple(
        _systemd_service_state(unit)
        for unit in ("transcribe-watch.service", "transcripts.service")
    )
    if any(
        item["active_state"] != "active"
        or item["sub_state"] != "running"
        or item["restarts"] != 0
        for item in services
    ):
        _fail("runtime_service_drift", "Transcript runtime continuity changed.")

    unsafe_directories = []
    unsafe_files = []
    for path in root.rglob("*"):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir() and mode != 0o700:
            unsafe_directories.append(str(path))
        elif path.is_file() and mode != 0o600:
            unsafe_files.append(str(path))
    if unsafe_directories or unsafe_files:
        _fail("private_mode_drift", "Plan 0059 private artifacts have unsafe modes.")

    manifest = {
        "schema_version": REFINE_AUDIT_VERSION,
        "status": "closed_refine",
        "audited_at": audited_at,
        "activation_content_sha256": activation_content_sha256,
        "p1_content_sha256": p1["content_sha256"],
        "terminal_decision": "refine",
        "completed_units": ["A0", "P0", "P1"],
        "partial_units": {
            "P2A": {
                "recordings_with_replayable_output": 1,
                "required_recordings": activation["recording_count"],
                "speaker_refs_with_output": 4,
                "required_speaker_refs": activation["speaker_ref_count"],
                "deterministic_recomputation_count": 2,
                "execution_content_sha256": next(iter(execution_hashes)),
            },
            "P2B": {
                **context_counts,
                "required_context_bundles": activation["recording_count"],
                "frozen_plan0059_context_bundles": 0,
            },
        },
        "not_run_units": ["P3", "P4", "P5", "P6-comparative-audit"],
        "refine_reasons": [
            "p2a_attempt_1_transcript_millisecond_offset_mismatch",
            "p2b_attempt_1_review_candidate_handle_not_uuid",
            "p2b_attempt_2_provider_source_record_not_host_opaque",
            "p2b_work_unit_attempt_limit_reached",
        ],
        "live_state": {"counts": live_counts, **live_status},
        "identity_state_sha256": identity_state["snapshot_sha256"],
        "services": list(services),
        "private_modes": {"unsafe_directory_count": 0, "unsafe_file_count": 0},
        "live_mutation_count": 0,
        "negative_actions": negative_action_vector(),
    }
    content_sha256 = canonical_artifact_hash(manifest)
    ensure_private_tree(root, run)
    write_immutable_private_json(manifest_path, manifest)
    receipt = {
        "schema_version": REFINE_AUDIT_VERSION,
        "status": "closed_refine",
        "content_sha256": content_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "completed_unit_count": 3,
        "partial_unit_count": 2,
        "not_run_unit_count": 4,
        "live_mutation_count": 0,
        "identity_state_unchanged": True,
        "runtime_continuity_preserved": True,
        "private_modes_preserved": True,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(receipt_path, receipt)
    return {
        **receipt,
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "idempotent_replay": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute Plan 0059 private evidence adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--runtime-root", type=Path, required=True)
    execute.add_argument("--state-root", type=Path, required=True)
    execute.add_argument("--activation-content-sha256", required=True)
    execute.add_argument("--plan0057-authority-manifest", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--runtime-root", type=Path, required=True)
    replay.add_argument("--activation-content-sha256", required=True)
    refine = subparsers.add_parser("freeze-refine-audit")
    refine.add_argument("--runtime-root", type=Path, required=True)
    refine.add_argument("--live-store-root", type=Path, required=True)
    refine.add_argument("--activation-content-sha256", required=True)
    refine.add_argument("--expected-identity-state-sha256", required=True)
    refine.add_argument("--audited-at", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "execute":
            result = execute_evidence_adapters(
                runtime_root=args.runtime_root,
                state_root=args.state_root,
                activation_content_sha256=args.activation_content_sha256,
                plan0057_authority_manifest=args.plan0057_authority_manifest,
            )
        elif args.command == "replay":
            result = replay_evidence_adapters(
                runtime_root=args.runtime_root,
                activation_content_sha256=args.activation_content_sha256,
            )
        else:
            result = freeze_refine_audit(
                runtime_root=args.runtime_root,
                live_store_root=args.live_store_root,
                activation_content_sha256=args.activation_content_sha256,
                expected_identity_state_sha256=args.expected_identity_state_sha256,
                audited_at=args.audited_at,
            )
    except IdentityOrchestrationError as exc:
        print(json.dumps({"status": "error", "reason_code": exc.reason_code, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
