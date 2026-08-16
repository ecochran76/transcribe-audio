from __future__ import annotations

import json
from pathlib import Path

import pytest

from identity_learning_contracts import (
    ContractError,
    contract_catalog,
    validate_adapter_exchange,
    validate_artifact,
)


def test_catalog_freezes_plan0072_a0_contract_families() -> None:
    catalog = contract_catalog()

    assert catalog["catalog_version"] == (
        "transcribe-audio.identity-learning-contract-catalog.v1"
    )
    assert catalog["contract_versions"] == {
        "domain": "transcribe-audio.identity-learning-domain.v1",
        "correction": "transcribe-audio.identity-learning-correction.v1",
        "privacy": "transcribe-audio.identity-learning-privacy.v1",
        "threat_model": "transcribe-audio.identity-learning-threat-model.v1",
        "api": "transcribe-audio.identity-learning-api.v1",
        "adapter": "transcribe-audio.identity-learning-adapter.v1",
        "supervisor": "transcribe-audio.identity-learning-supervisor.v1",
    }
    assert set(catalog["artifacts"]) == {
        "source_observation",
        "person",
        "external_identity",
        "source_record",
        "person_alias",
        "role_assertion",
        "relationship_assertion",
        "conversation_association_candidate",
        "participant_hypothesis",
        "speaker_identity_proposal",
        "speaker_review_decision",
        "voice_sample",
        "voice_profile_version",
        "correction_event",
        "transcript_correction_proposal",
        "normalized_transcript_generation",
        "terminology_entry",
        "processing_run",
        "identity_review_queue_item",
        "provider_adapter_request",
        "provider_adapter_result",
        "identity_review_submission",
        "effect_preview",
    }
    assert catalog["effect_policy"] == {
        "provider_writes": "prohibited",
        "live_store_migration": "prohibited_in_a0",
        "historical_processing": "prohibited_in_a0",
        "biometric_collection": "prohibited_in_a0",
        "dashboard_publication": "prohibited_in_a0",
    }


def test_identity_review_queue_requires_original_filename_and_safe_lineage() -> None:
    item = {
        "schema_version": "transcribe-audio.identity-review-queue-item.v1",
        "queue_item_id": "queue-1",
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "original_recording_filename": "2026-01-07 Example Recording.m4a",
        "source_artifact_sha256": "a" * 64,
        "source_media_sha256": "b" * 64,
        "processing_run_id": "run-1",
        "model_versions": {"identity": "identity.test.v1"},
        "rubric_versions": {"calendar": "calendar.test.v1"},
        "profile_versions": [],
        "calendar_candidates": [],
        "participant_hypotheses": [],
        "speakers": [],
        "review_state": "unreviewed",
        "decision_history": [],
        "effect_preview_ref": "preview-1",
        "projection_version": "queue-projection.v1",
        "created_at": "2026-08-16T19:00:00Z",
    }

    assert validate_artifact("identity_review_queue_item", item) == item

    item["original_recording_filename"] = "/private/recordings/example.m4a"
    with pytest.raises(ContractError, match="filename"):
        validate_artifact("identity_review_queue_item", item)

    item["original_recording_filename"] = "example.m4a"
    item["speakers"] = [{"raw_path": "/private/sample.wav"}]
    with pytest.raises(ContractError, match="forbidden private fields"):
        validate_artifact("identity_review_queue_item", item)


def test_provider_adapter_exchange_is_bounded_read_only_and_scope_bound() -> None:
    scope = {
        "provider_kind": "calendar",
        "profile_id": "profile-redacted",
        "account_id": "account-redacted",
        "tenant_id": "tenant-redacted",
        "capabilities": ["event_metadata_read"],
    }
    request = {
        "schema_version": "transcribe-audio.provider-adapter-request.v1",
        "request_id": "request-1",
        "processing_run_id": "run-1",
        "conversation_id": "conversation-1",
        "adapter_id": "calendar-adapter.v1",
        "capability": "event_metadata_read",
        "source_scope": scope,
        "as_of": "2026-01-07T16:00:00Z",
        "query": {"window_start": "2026-01-07T15:00:00Z"},
        "budgets": {
            "max_records": 10,
            "max_characters": 4000,
            "max_calls": 1,
            "max_latency_ms": 5000,
        },
        "mode": "read_only",
        "idempotency_key": "adapter-request-1",
        "created_at": "2026-08-16T19:00:00Z",
    }
    result = {
        "schema_version": "transcribe-audio.provider-adapter-result.v1",
        "result_id": "result-1",
        "request_id": "request-1",
        "processing_run_id": "run-1",
        "source_scope": scope,
        "status": "partial",
        "observations": [],
        "warnings": ["provider_timeout_after_first_page"],
        "failure": {"kind": "transient_timeout", "retryable": True},
        "retrieved_at": "2026-08-16T19:00:01Z",
        "provider_write_count": 0,
        "consumed_budget": {"records": 0, "characters": 0, "calls": 1},
    }

    assert validate_adapter_exchange(request, result) == (request, result)

    request["mode"] = "apply"
    with pytest.raises(ContractError, match="read_only"):
        validate_artifact("provider_adapter_request", request)
    request["mode"] = "read_only"

    result["provider_write_count"] = 1
    with pytest.raises(ContractError, match="provider writes"):
        validate_artifact("provider_adapter_result", result)
    result["provider_write_count"] = 0

    result["source_scope"] = {**scope, "tenant_id": "another-tenant"}
    with pytest.raises(ContractError, match="scope"):
        validate_adapter_exchange(request, result)


def test_review_submission_is_stale_safe_and_effects_are_preview_only() -> None:
    submission = {
        "schema_version": "transcribe-audio.identity-review-submission.v1",
        "submission_id": "submission-1",
        "queue_item_id": "queue-1",
        "conversation_id": "conversation-1",
        "proposal_id": "speaker-proposal-1",
        "action": "choose_existing_person",
        "expected_projection_version": "queue-projection.v7",
        "decision_payload": {"person_id": "person-1"},
        "comment": "Redacted fixture decision.",
        "idempotency_key": "review-submission-1",
        "reviewer": "operator-redacted",
        "decided_at": "2026-08-16T19:10:00Z",
    }
    preview = {
        "schema_version": "transcribe-audio.effect-preview.v1",
        "preview_id": "preview-1",
        "queue_item_id": "queue-1",
        "submission_id": "submission-1",
        "expected_projection_version": "queue-projection.v7",
        "effect_mode": "preview_only",
        "proposed_effects": ["append_speaker_review_decision"],
        "invalidations": [],
        "profile_rebuilds": [],
        "provider_write_count": 0,
        "raw_deletion_count": 0,
        "warnings": [],
        "created_at": "2026-08-16T19:10:00Z",
    }

    assert validate_artifact("identity_review_submission", submission) == submission
    assert validate_artifact("effect_preview", preview) == preview

    submission["expected_projection_version"] = ""
    with pytest.raises(ContractError, match="stale-write"):
        validate_artifact("identity_review_submission", submission)
    submission["expected_projection_version"] = "queue-projection.v7"

    preview["effect_mode"] = "apply"
    with pytest.raises(ContractError, match="preview_only"):
        validate_artifact("effect_preview", preview)


def test_biometric_custody_requires_reviewed_person_binding_and_private_refs() -> None:
    sample = {
        "schema_version": "transcribe-audio.voice-sample.v1",
        "sample_id": "sample-1",
        "recording_id": "recording-1",
        "conversation_id": "conversation-1",
        "speaker_ref": "speaker-A",
        "start_ms": 1000,
        "end_ms": 4500,
        "source_media_sha256": "a" * 64,
        "sample_sha256": "b" * 64,
        "quality": {"eligible": True, "reason_codes": []},
        "preparation_lineage": {"recipe_version": "sample.test.v1"},
        "review_authority_id": "review-1",
        "consent_authority": "operator-policy.v1",
        "person_id": "person-1",
        "review_state": "reviewed",
        "exclusion_state": "included",
        "private_audio_ref": {"object_id": "private-sample-1", "sha256": "b" * 64},
        "created_at": "2026-08-16T19:20:00Z",
    }
    profile = {
        "schema_version": "transcribe-audio.voice-profile-version.v1",
        "profile_version_id": "profile-1-v1",
        "person_id": "person-1",
        "profile_family": "default",
        "predecessor_profile_version_id": None,
        "sample_allowlist": [
            {
                "sample_id": "sample-1",
                "review_authority_id": "review-1",
                "consent_authority": "operator-policy.v1",
            }
        ],
        "evaluation_id": "evaluation-1",
        "status": "pending",
        "active_interval": None,
        "private_profile_ref": {"object_id": "private-profile-1", "sha256": "c" * 64},
        "created_at": "2026-08-16T19:20:00Z",
    }

    assert validate_artifact("voice_sample", sample) == sample
    assert validate_artifact("voice_profile_version", profile) == profile

    sample["review_state"] = "unreviewed"
    with pytest.raises(ContractError, match="person-bound"):
        validate_artifact("voice_sample", sample)

    profile["private_profile_ref"]["embedding_vector"] = [0.1, 0.2]
    with pytest.raises(ContractError, match="inline biometric"):
        validate_artifact("voice_profile_version", profile)


def test_correction_and_supervisor_contracts_bound_requeue_and_live_effects() -> None:
    correction = {
        "schema_version": "transcribe-audio.transcript-correction-proposal.v1",
        "proposal_id": "correction-1",
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "raw_transcript_sha256": "a" * 64,
        "raw_span": {"start": 14, "end": 18, "text_sha256": "b" * 64},
        "replacement_text": "SESO",
        "correction_kind": "asr_confusion",
        "terminology_entry_id": "term-seso-1",
        "scope": {"type": "domain", "id": "chemistry"},
        "evidence_ids": ["observation-1"],
        "review_state": "proposed",
        "correction_pass": "pre_identity",
        "processing_version": "identity-learning.test.v1",
        "cascade_count": 0,
        "created_at": "2026-08-16T19:30:00Z",
    }
    run = {
        "schema_version": "transcribe-audio.processing-run.v1",
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "recording_id": "recording-1",
        "original_recording_filename": "2026-01-07 Example Recording.m4a",
        "source_artifact_sha256": "c" * 64,
        "source_media_sha256": "d" * 64,
        "operation_mode": "contract_fixture",
        "policy_version": "identity-learning.test.v1",
        "as_of": "2026-01-07T16:00:00Z",
        "capabilities": [],
        "budgets": {"provider_calls": 0, "model_calls": 0},
        "model_versions": {},
        "rubric_versions": {},
        "profile_versions": [],
        "stage": "bind_conversation",
        "state": "complete",
        "transcript_correction_passes": 0,
        "identity_cascade_count": 0,
        "provider_retry_count": 0,
        "input_ids": ["fixture-input-1"],
        "output_ids": ["fixture-output-1"],
        "failures": [],
        "effect_counts": {
            "provider_writes": 0,
            "accepted_identity": 0,
            "profile_enrollment": 0,
            "live_store_migration": 0,
        },
        "created_at": "2026-08-16T19:30:00Z",
    }

    assert validate_artifact("transcript_correction_proposal", correction) == correction
    assert validate_artifact("processing_run", run) == run

    correction["cascade_count"] = 2
    with pytest.raises(ContractError, match="one-cascade"):
        validate_artifact("transcript_correction_proposal", correction)

    run["effect_counts"]["accepted_identity"] = 1
    with pytest.raises(ContractError, match="contract_fixture"):
        validate_artifact("processing_run", run)


def test_catalog_freezes_supervisor_limits_and_threat_controls() -> None:
    catalog = contract_catalog()

    assert catalog["supervisor"]["stages"] == [
        "bind_conversation",
        "pre_identity_correction",
        "calendar_candidate_generation",
        "participant_and_evidence_collection",
        "speaker_and_relationship_proposals",
        "post_identity_correction",
        "queue_projection",
        "complete",
    ]
    assert catalog["supervisor"]["limits"] == {
        "max_provider_retries": 1,
        "max_transcript_correction_passes": 2,
        "max_transcript_identity_cascades": 1,
        "max_model_reference_repairs_per_phase": 1,
        "expensive_enrichment_backlog_threshold": 500,
    }
    assert set(catalog["threat_controls"]) >= {
        "authenticated_private_route",
        "tenant_account_capability_scope",
        "provider_write_prohibition",
        "raw_path_non_disclosure",
        "biometric_private_storage",
        "review_before_enrollment",
        "stale_write_rejection",
        "source_disjoint_evaluation",
        "self_training_prohibition",
        "correction_cascade_limit",
        "deletion_invalidation_and_tombstone",
        "evidence_independence_groups",
        "hindsight_labeling",
    }


def test_every_frozen_artifact_has_a_versioned_nontrivial_schema() -> None:
    artifacts = contract_catalog()["artifacts"]

    assert all(spec["schema_version"].endswith(".v1") for spec in artifacts.values())
    assert all(len(spec["required_fields"]) >= 5 for spec in artifacts.values())
    assert all(spec["privacy_class"] for spec in artifacts.values())


def test_redacted_a0_fixtures_match_contract_and_resolve_privacy() -> None:
    fixture_root = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "dev"
        / "fixtures"
        / "plan-0072-a0"
    )
    freeze = json.loads((fixture_root / "contract-freeze.json").read_text())
    examples = json.loads((fixture_root / "redacted-artifacts.json").read_text())
    threats = json.loads((fixture_root / "threat-control-matrix.json").read_text())
    catalog = contract_catalog()

    assert freeze == {
        "catalog_version": catalog["catalog_version"],
        "contract_versions": catalog["contract_versions"],
        "artifact_schemas": {
            kind: spec["schema_version"]
            for kind, spec in catalog["artifacts"].items()
        },
        "effect_policy": catalog["effect_policy"],
        "supervisor": catalog["supervisor"],
        "threat_controls": catalog["threat_controls"],
    }
    assert set(examples["artifacts"]) == {
        "identity_review_queue_item",
        "provider_adapter_request",
        "provider_adapter_result",
        "identity_review_submission",
        "effect_preview",
        "transcript_correction_proposal",
        "processing_run",
        "voice_sample",
        "voice_profile_version",
    }
    for kind, payload in examples["artifacts"].items():
        assert validate_artifact(kind, payload) == payload
    assert all(
        row["control"] in catalog["threat_controls"]
        for row in threats["threats"]
    )
    assert all(row["disposition"] == "accepted_for_a0" for row in threats["threats"])
    assert threats["unresolved_privacy_decisions"] == []
