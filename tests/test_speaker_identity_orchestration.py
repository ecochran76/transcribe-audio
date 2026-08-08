from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from acoustic_audio_derivatives import write_immutable_private_json
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
    ShadowIdentityDecision,
    TransitionReceipt,
    confidence_cap,
    freeze_activation,
    negative_action_vector,
    replay_activation,
    validate_bundle_bindings,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = "2026-08-08T14:35:32Z"


def lineage(evidence_id: str, group: str, *, current: bool = False) -> EvidenceLineage:
    return EvidenceLineage(
        evidence_id=evidence_id,
        source_record_id=f"source-{evidence_id}",
        independence_group=group,
        source_type="local-store",
        source_event_at=NOW,
        observed_at=NOW,
        retrieved_at=NOW,
        content_sha256=HASH_A,
        proposed_by_current_evaluation=current,
    )


def acoustic() -> AcousticEvidenceBundle:
    return AcousticEvidenceBundle(
        conversation_id="conversation-001",
        recording_id="recording-001",
        document_id="document-001",
        speaker_refs=("SPEAKER_1",),
        source_media_sha256=HASH_A,
        transcript_sha256=HASH_B,
        execution_sha256=HASH_C,
        identity_state_sha256=HASH_A,
        model_versions=(("model-a", "version-1"),),
        created_at=NOW,
        evidence=(
            AcousticSpeakerEvidence(
                speaker_ref="SPEAKER_1",
                disposition="review",
                acoustic_subject_id="subject-001",
                score=0.8,
                confidence_band="medium",
                supporting_unit_count=5,
                opposing_unit_count=1,
                insufficient_unit_count=3,
                evidence_ids=("evidence-acoustic",),
            ),
        ),
        lineage=(lineage("evidence-acoustic", "group-acoustic"),),
        negative_actions=negative_action_vector(),
    )


def scope() -> EvidenceScope:
    return EvidenceScope(
        source_type="local-store",
        source_profile="profile-default",
        account_id="not-applicable",
        tenant_id="not-applicable",
        capabilities=("identity", "relationship"),
        as_of=NOW,
        max_records=20,
        max_characters=12000,
        max_per_source=5,
        max_provider_calls=0,
        max_relationship_hops=1,
    )


def context(*, document_id: str = "document-001", current: bool = False) -> ContextEvidenceBundle:
    return ContextEvidenceBundle(
        conversation_id="conversation-001",
        recording_id="recording-001",
        document_id=document_id,
        speaker_refs=("SPEAKER_1",),
        transcript_sha256=HASH_B,
        scopes=(scope(),),
        retrieval_version="retrieval-v1",
        ranking_version="ranking-v1",
        policy_version="policy-v1",
        included_evidence_ids=("evidence-context",),
        excluded_evidence=(("evidence-excluded", "outside-as-of"),),
        warnings=(),
        source_failures=(("source-optional", "unavailable", False),),
        lineage=(lineage("evidence-context", "group-context", current=current),),
        negative_actions=negative_action_vector(),
    )


def candidates() -> CanonicalCandidateSnapshot:
    return CanonicalCandidateSnapshot(
        conversation_id="conversation-001",
        document_id="document-001",
        as_of=NOW,
        schema_version="knowledge-v3",
        projection_watermark=HASH_C,
        candidates=(
            CanonicalCandidate(
                person_id="person-001",
                source_record_ids=("source-person-001",),
                evidence_ids=("evidence-person",),
                score=0.7,
            ),
        ),
        lineage=(lineage("evidence-person", "group-person"),),
        negative_actions=negative_action_vector(),
    )


def test_contracts_bind_and_are_content_addressed() -> None:
    acoustic_bundle = acoustic()
    context_bundle = context()
    snapshot = candidates()
    validate_bundle_bindings(acoustic_bundle, context_bundle, snapshot)
    assert acoustic_bundle.bundle_id.startswith("bundle-")
    assert context_bundle.bundle_id.startswith("bundle-")
    assert snapshot.snapshot_id.startswith("snapshot-")


def test_binding_and_current_run_circularity_fail_closed() -> None:
    with pytest.raises(IdentityOrchestrationError, match="same frozen case") as mismatch:
        validate_bundle_bindings(acoustic(), context(document_id="document-999"), candidates())
    assert mismatch.value.reason_code == "binding_mismatch"
    with pytest.raises(IdentityOrchestrationError) as circular:
        context(current=True)
    assert circular.value.reason_code == "circular_current_run_support"


def test_negative_action_and_state_transition_guards() -> None:
    vector = negative_action_vector()
    vector["write_graphiti"] = True
    with pytest.raises(IdentityOrchestrationError) as forbidden:
        AcousticEvidenceBundle(**{**acoustic().__dict__, "negative_actions": vector})
    assert forbidden.value.reason_code == "forbidden_mutation"
    with pytest.raises(IdentityOrchestrationError) as invalid:
        TransitionReceipt(
            evaluation_id="evaluation-001",
            actor="host-policy",
            transitioned_at=NOW,
            prior_state="pending",
            next_state="accepted",
            input_hashes=(HASH_A,),
            policy_version="policy-v1",
            reason_code="invalid-shortcut",
            negative_actions=negative_action_vector(),
        )
    assert invalid.value.reason_code == "invalid_state_transition"


def test_confidence_caps_are_reason_coded_and_conservative() -> None:
    capped, reasons = confidence_cap(0.95, ["partial_provider_failure", "material_contradiction"])
    assert capped == 0.49
    assert reasons == ("material_contradiction", "partial_provider_failure")


def test_evaluation_and_shadow_decision_are_immutable_and_non_applying() -> None:
    evaluation = IdentityCaseEvaluation(
        evaluation_id="evaluation-001",
        conversation_id="conversation-001",
        recording_id="recording-001",
        document_id="document-001",
        speaker_ref="SPEAKER_1",
        condition="combined",
        acoustic_bundle_id="bundle-acoustic-001",
        context_bundle_id="bundle-context-001",
        candidate_snapshot_id="snapshot-candidates-001",
        candidate_person_ids=("person-001", "person-002"),
        factors=(
            IdentityEvidenceFactor(
                factor_type="acoustic",
                score=0.8,
                evidence_ids=("evidence-acoustic",),
                independence_groups=("group-acoustic",),
            ),
            IdentityEvidenceFactor(
                factor_type="contradiction",
                score=-0.4,
                evidence_ids=("evidence-conflict",),
                independence_groups=("group-context",),
            ),
        ),
        outcome="proposed",
        proposed_person_id="person-001",
        alternative_person_ids=("person-002",),
        contradiction_evidence_ids=("evidence-conflict",),
        base_confidence=0.9,
        capped_confidence=0.49,
        confidence_cap_reasons=("material_contradiction",),
        abstention_reason=None,
        source_failures=(),
        policy_version="policy-v1",
        evaluated_at=NOW,
        negative_actions=negative_action_vector(),
    )
    decision = ShadowIdentityDecision(
        decision_id="decision-001",
        evaluation_id=evaluation.evaluation_id,
        speaker_ref=evaluation.speaker_ref,
        outcome="unresolved",
        selected_person_id=None,
        reviewer="operator-review",
        decided_at=NOW,
        evaluation_sha256=evaluation.content_sha256,
        reason_code="insufficient-independent-evidence",
        negative_actions=negative_action_vector(),
    )
    assert len(evaluation.content_sha256) == 64
    assert len(decision.content_sha256) == 64
    with pytest.raises(IdentityOrchestrationError) as required_failure:
        IdentityCaseEvaluation(
            **{
                **evaluation.__dict__,
                "source_failures": (("source-required", "unavailable", True),),
            }
        )
    assert required_failure.value.reason_code == "required_failure_proposed"


def _make_store(root: Path) -> tuple[list[str], str]:
    root.mkdir(mode=0o700)
    database = root / "transcripts.sqlite3"
    con = sqlite3.connect(database)
    con.executescript(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, kind TEXT, artifact_sha256 TEXT,
            generated_at TEXT, json_payload TEXT
        );
        CREATE TABLE contacts (id TEXT PRIMARY KEY);
        CREATE TABLE speaker_assignments (id TEXT PRIMARY KEY);
        CREATE TABLE blobs (
            id TEXT PRIMARY KEY, role TEXT, sha256 TEXT, bytes INTEGER
        );
        CREATE TABLE document_blobs (document_id TEXT, blob_id TEXT);
        """
    )
    ids = ["document-001", "document-002"]
    rows = []
    for index, document_id in enumerate(ids, 1):
        artifact = f"{index}" * 64
        media = f"{index + 2}" * 64
        generated = f"2026-08-0{index}T10:00:00Z"
        speakers = [{"speaker": "SPEAKER_1"}, {"speaker": "SPEAKER_2"}]
        con.execute(
            "INSERT INTO documents VALUES (?, 'transcript', ?, ?, ?)",
            (document_id, artifact, generated, json.dumps({"utterances": speakers})),
        )
        blob_id = f"blob-{index:03d}"
        con.execute("INSERT INTO blobs VALUES (?, 'source_recording', ?, 100)", (blob_id, media))
        con.execute("INSERT INTO document_blobs VALUES (?, ?)", (document_id, blob_id))
        rows.append(
            {
                "document_id": document_id,
                "artifact_sha256": artifact,
                "generated_at": generated,
                "speaker_count": 2,
                "source_media_sha256": media,
                "source_media_bytes": 100,
            }
        )
    con.commit()
    con.close()
    database.chmod(0o600)
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"
    import hashlib

    return ids, hashlib.sha256(serialized.encode()).hexdigest()


def test_activation_receipt_is_private_bound_and_replayable(tmp_path: Path) -> None:
    store = tmp_path / "store"
    ids, membership = _make_store(store)
    prior_root = tmp_path / "prior"
    prior_run = prior_root / "run"
    prior_root.mkdir(mode=0o700)
    prior_run.mkdir(mode=0o700)
    prior_manifest = prior_run / "private-manifest.json"
    write_immutable_private_json(
        prior_manifest,
        {"preview": {"private_evidence": {"cohort": [{"source_media_sha256": "f" * 64}]}}},
    )
    runtime = tmp_path / "runtime"
    receipt = freeze_activation(
        store_root=store,
        prior_plan0057_manifest=prior_manifest,
        runtime_root=runtime,
        cohort_document_ids=ids,
        expected_membership_sha256=membership,
        repository_head=HASH_A,
        branch="plan-0037-campaign",
        activated_at=NOW,
        service_active_state="active",
        service_sub_state="running",
        service_restarts=0,
    )
    replayed = replay_activation(receipt["content_sha256"], runtime_root=runtime)
    assert receipt["recording_count"] == 2
    assert receipt["speaker_ref_count"] == 4
    assert replayed["idempotent_replay"] is True
    assert Path(receipt["manifest_path"]).stat().st_mode & 0o777 == 0o600
    assert Path(receipt["manifest_path"]).parent.stat().st_mode & 0o777 == 0o700
    assert runtime.stat().st_mode & 0o777 == 0o700
