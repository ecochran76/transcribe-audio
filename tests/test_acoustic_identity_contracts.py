from __future__ import annotations

import json
from pathlib import Path

import pytest

from acoustic_identity_contracts import (
    ACOUSTIC_EVIDENCE_SCHEMA,
    BIOMETRIC_PROFILE_SCHEMA,
    DERIVED_AUDIO_SCHEMA,
    REPROCESSING_MANIFEST_SCHEMA,
    VERIFICATION_TRIAL_SCHEMA,
    ContractError,
    contract_catalog,
    validate_artifact,
)


def test_catalog_freezes_six_versioned_artifact_families() -> None:
    catalog = contract_catalog()

    assert catalog["catalog_version"] == (
        "transcribe-audio.acoustic-contract-catalog.v1"
    )
    assert set(catalog["artifacts"]) == {
        "derived_audio",
        "audio_quality",
        "biometric_profile",
        "verification_trial_manifest",
        "acoustic_evidence_bundle",
        "reprocessing_manifest",
    }
    assert all(
        artifact["schema_version"].endswith(".v1")
        for artifact in catalog["artifacts"].values()
    )


def test_portable_bundle_rejects_nested_raw_embedding() -> None:
    payload = {
        "schema_version": ACOUSTIC_EVIDENCE_SCHEMA,
        "bundle_id": "bundle-1",
        "source_blob_id": "blob-1",
        "source_sha256": "a" * 64,
        "recipe_revision": "recipe.v1",
        "model_revisions": {"fake": "fake.v1"},
        "candidate_evidence": [
            {
                "prepared_candidate_id": "person-1",
                "evidence_id": "evidence-1",
                "metadata": {"embedding_vector": [0.1, 0.2]},
            }
        ],
        "same_person_label_evidence": [],
        "quality_summary": {},
        "abstention_reasons": [],
        "warnings": [],
        "created_at": "2026-07-31T00:00:00Z",
    }

    with pytest.raises(ContractError, match="forbidden biometric fields"):
        validate_artifact("acoustic_evidence_bundle", payload)


def test_biometric_profile_requires_confirmed_provenance_and_private_ref() -> None:
    payload = {
        "schema_version": BIOMETRIC_PROFILE_SCHEMA,
        "profile_id": "profile-1",
        "person_id": "person-1",
        "status": "active",
        "eligible_for_scoring": True,
        "confirmed_sources": [
            {
                "recording_id": "recording-1",
                "speaker_label": "A",
                "operator_confirmation_id": "gold-1",
            }
        ],
        "embedding_model_revision": "fake.v1",
        "preprocessing_revision": "recipe.v1",
        "private_embedding_ref": {
            "path": "profiles/profile-1/embedding.bin",
            "sha256": "b" * 64,
        },
        "session_diversity": {},
        "dispersion": {},
        "audit": {},
    }

    assert validate_artifact("biometric_profile", payload) == payload
    payload["private_embedding_ref"] = {
        "sha256": "b" * 64,
        "vector": [0.1],
    }
    with pytest.raises(ContractError, match="not inline"):
        validate_artifact("biometric_profile", payload)

    payload["status"] = "deleted"
    payload["eligible_for_scoring"] = False
    with pytest.raises(ContractError, match="must not retain"):
        validate_artifact("biometric_profile", payload)
    payload["private_embedding_ref"] = None
    assert validate_artifact("biometric_profile", payload) == payload


def test_derived_audio_rejects_nonmonotonic_timestamp_map() -> None:
    payload = {
        "schema_version": DERIVED_AUDIO_SCHEMA,
        "artifact_id": "derived-1",
        "source_blob_id": "blob-1",
        "source_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "source_duration_seconds": 10.0,
        "output_duration_seconds": 8.0,
        "recipe": {
            "revision": "recipe.v1",
            "operation": "vad_windows",
            "decoder_revision": "ffmpeg.test",
            "parameters": {},
            "model_revisions": {},
        },
        "timestamp_map": [
            {
                "source_start_seconds": 2.0,
                "source_end_seconds": 5.0,
                "output_start_seconds": 0.0,
                "output_end_seconds": 3.0,
            },
            {
                "source_start_seconds": 4.0,
                "source_end_seconds": 7.0,
                "output_start_seconds": 3.0,
                "output_end_seconds": 6.0,
            },
        ],
        "created_at": "2026-07-31T00:00:00Z",
    }

    with pytest.raises(ContractError, match="monotonic"):
        validate_artifact("derived_audio", payload)


def test_verification_manifest_rejects_conversation_split_crossing() -> None:
    payload = {
        "schema_version": VERIFICATION_TRIAL_SCHEMA,
        "manifest_id": "trials-1",
        "corpus_id": "corpus-1",
        "conversation_split_policy": "conversation_id_disjoint",
        "recipe_revision": "recipe.v1",
        "model_revisions": {"fake": "fake.v1"},
        "threshold_policy_revision": "thresholds.v1",
        "trials": [
            {
                "split": "development",
                "truth": "same_person",
                "test_conversation_id": "conversation-a",
                "enrollment_conversation_ids": ["conversation-b"],
            },
            {
                "split": "evaluation",
                "truth": "different_person",
                "test_conversation_id": "conversation-c",
                "enrollment_conversation_ids": ["conversation-a"],
            },
        ],
        "denominators": {"trials": 2},
        "prediction_visibility": "excluded",
        "created_at": "2026-07-31T00:00:00Z",
    }

    with pytest.raises(ContractError, match="cross splits"):
        validate_artifact("verification_trial_manifest", payload)


def test_apply_reprocessing_requires_explicit_approval() -> None:
    payload = {
        "schema_version": REPROCESSING_MANIFEST_SCHEMA,
        "manifest_id": "manifest-1",
        "mode": "apply",
        "status": "planned",
        "items": [{"recording_id": "recording-1", "source_sha256": "a" * 64}],
        "source_lineage": [{"source_blob_id": "blob-1", "sha256": "a" * 64}],
        "idempotency_key": "idempotency-1",
        "output_policy": {
            "overwrite_original_audio": False,
            "overwrite_original_transcript": False,
        },
        "dry_run_predecessor": {"manifest_id": "dry-run-1", "sha256": "b" * 64},
        "resumable_checkpoint": {"checkpoint_id": "checkpoint-1"},
        "rollback": {"rollback_manifest_id": "rollback-1"},
        "recipe_revision": "recipe.v1",
        "model_revisions": {},
        "approval": {"status": "pending", "approval_id": ""},
        "created_at": "2026-07-31T00:00:00Z",
    }

    with pytest.raises(ContractError, match="approval bound"):
        validate_artifact("reprocessing_manifest", payload)

    payload["approval"] = {
        "status": "approved",
        "approval_id": "approval-1",
        "approved_manifest_id": "dry-run-1",
        "approved_manifest_sha256": "b" * 64,
    }
    assert validate_artifact("reprocessing_manifest", payload) == payload


def test_redacted_catalog_and_license_inventory_are_complete() -> None:
    fixture_root = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "dev"
        / "fixtures"
        / "plan-0037-p0"
    )
    redacted = json.loads(
        (fixture_root / "contract-catalog.json").read_text(encoding="utf-8")
    )
    assert {
        name: {
            "privacy_class": artifact["privacy_class"],
            "schema_version": artifact["schema_version"],
        }
        for name, artifact in contract_catalog()["artifacts"].items()
    } == redacted["artifacts"]

    inventory = json.loads(
        (fixture_root / "model-license-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = inventory["candidates"]
    assert len(candidates) >= 7
    required = {
        "candidate_id",
        "role",
        "code_license",
        "code_source",
        "checkpoint_candidates",
        "checkpoint_terms",
        "checkpoint_terms_source",
        "acquisition_status",
        "revision",
        "sha256",
        "promotion_blocker",
    }
    assert all(required <= candidate.keys() for candidate in candidates)
    assert all(candidate["acquisition_status"] != "acquired" for candidate in candidates)
    assert all(candidate["promotion_blocker"] for candidate in candidates)
