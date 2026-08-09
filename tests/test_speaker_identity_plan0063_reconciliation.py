from __future__ import annotations

from copy import deepcopy

import pytest

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0062_human_comparison as plan0062
import speaker_identity_plan0063_reconciliation as reconciliation


HASH_A = "a" * 64


def _with_hash(core: dict) -> dict:
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _sources() -> tuple[dict, dict]:
    decisions = [
        {
            "slot_id": "recording-1::SPEAKER_1",
            "decision_type": "new_person",
            "label": "Alex Example",
            "selected_token": "new_person:token-1",
        },
        {
            "slot_id": "recording-2::SPEAKER_1",
            "decision_type": "contextual_unlisted_suggestion",
            "label": "Alex Example | Example Co | alex@example.com",
            "suggestion": {
                "name": "Alex Example",
                "email": "alex@example.com",
                "organization": "Example Co",
            },
            "selected_token": "suggested-token-2",
        },
        {
            "slot_id": "recording-1::SPEAKER_2",
            "decision_type": "contextual_unlisted_suggestion",
            "label": "Eric Example | Example Co | eric@example.com",
            "suggestion": {
                "name": "Eric Example",
                "email": "eric@example.com",
                "organization": "Example Co",
            },
            "selected_token": "suggested-token-3",
        },
        {
            "slot_id": "recording-2::SPEAKER_2",
            "decision_type": "linked_enrolled_context_identity",
            "label": "Eric Example | Example Co | eric@example.com",
            "suggestion": {
                "name": "Eric Example",
                "email": "eric@example.com",
                "organization": "Example Co",
            },
            "selected_token": "linked-token-4",
            "acoustic_subject_id": "subject-eric",
            "acoustic_bundle_id": "bundle-eric",
            "acoustic_bundle_sha256": HASH_A,
        },
        {
            "slot_id": "recording-3::SPEAKER_1",
            "decision_type": "contextual_role_placeholder",
            "label": "Unknown staff member",
            "suggestion": {
                "name": "Unknown staff member",
                "email": "",
                "organization": "",
            },
            "selected_token": "suggested-role",
        },
        {
            "slot_id": "recording-3::SPEAKER_2",
            "decision_type": "new_person",
            "label": "Jordan Example",
            "selected_token": "new_person:token-6",
        },
    ]
    submission = _with_hash(
        {
            "schema_version": plan0062.DECISION_SCHEMA,
            "status": "human_gold_frozen_pending_comparison",
            "p3_content_sha256": "b" * 64,
            "p4_content_sha256": "c" * 64,
            "enrolled_binding_content_sha256": "placeholder",
            "decision_count": len(decisions),
            "decisions": decisions,
            "submission_source": "client_export",
            "human_observations": [],
            "negative_actions": plan0062.negative_action_vector(),
        }
    )
    binding = _with_hash(
        {
            "schema_version": plan0062.BINDING_SCHEMA,
            "status": "private_enrolled_option_bindings_ready",
            "p4_content_sha256": "c" * 64,
            "binding_count": 1,
            "bindings": [
                {
                    "slot_id": "recording-2::SPEAKER_2",
                    "token": "enrolled-token",
                    "acoustic_subject_id": "subject-eric",
                    "acoustic_bundle_id": "bundle-eric",
                    "acoustic_bundle_sha256": HASH_A,
                }
            ],
            "negative_actions": plan0062.negative_action_vector(),
        }
    )
    submission_core = {
        key: value for key, value in submission.items() if key != "content_sha256"
    }
    submission_core["enrolled_binding_content_sha256"] = binding["content_sha256"]
    submission = _with_hash(submission_core)
    return submission, binding


def test_reconciliation_keeps_merge_evidence_and_voice_binding_review_only() -> None:
    submission, binding = _sources()
    manifest = reconciliation.build_reconciliation_manifest(
        submission, binding, activation_content_sha256=HASH_A
    )

    assert manifest["metrics"] == {
        "speaker_slot_count": 6,
        "named_slot_count": 5,
        "role_placeholder_count": 1,
        "slot_person_identity_count": 5,
        "person_proposal_count": 3,
        "merge_proposal_count": 2,
        "exact_external_identity_merge_proposal_count": 1,
        "name_only_merge_proposal_count": 1,
        "existing_voice_binding_proposal_count": 1,
        "new_enrollment_candidate_count": 2,
    }
    assert {item["basis"] for item in manifest["merge_proposals"]} == {
        "exact_external_identity",
        "name_only",
    }
    assert all(
        item["decision"] == "pending" and item["authoritative_merge"] is False
        for item in manifest["merge_proposals"]
    )
    name_only = next(
        item
        for item in manifest["person_proposals"]
        if item["basis"] == "name_only"
    )
    assert len(set(name_only["member_slot_person_ids"])) == 2
    assert manifest["role_placeholders"][0]["creates_person"] is False
    assert manifest["voice_binding_proposals"][0]["binding_applied"] is False
    assert manifest["live_mutation_count"] == 0
    assert not any(manifest["negative_actions"].values())


def test_reconciliation_is_deterministic_and_excludes_enrolled_group_from_new() -> None:
    submission, binding = _sources()
    first = reconciliation.build_reconciliation_manifest(
        submission,
        binding,
        activation_content_sha256=HASH_A,
        repository_authority={"commit": "test-commit"},
    )
    second = reconciliation.build_reconciliation_manifest(
        deepcopy(submission),
        deepcopy(binding),
        activation_content_sha256=HASH_A,
        repository_authority={"commit": "test-commit"},
    )
    enrolled_person = first["voice_binding_proposals"][0]["proposed_person_id"]

    assert first == second
    assert first["repository_authority"] == {"commit": "test-commit"}
    assert enrolled_person not in {
        item["proposed_person_id"] for item in first["enrollment_candidates"]
    }


def test_unselected_enrolled_option_does_not_become_a_voice_binding() -> None:
    submission, binding = _sources()
    binding_core = {
        key: deepcopy(value)
        for key, value in binding.items()
        if key != "content_sha256"
    }
    binding_core["bindings"].append(
        {
            "slot_id": "recording-1::SPEAKER_1",
            "token": "unselected-enrolled-token",
            "acoustic_subject_id": "subject-eric",
            "acoustic_bundle_id": "unselected-bundle",
            "acoustic_bundle_sha256": "d" * 64,
        }
    )
    binding_core["binding_count"] = 2
    binding = _with_hash(binding_core)
    submission_core = {
        key: deepcopy(value)
        for key, value in submission.items()
        if key != "content_sha256"
    }
    submission_core["enrolled_binding_content_sha256"] = binding["content_sha256"]
    submission = _with_hash(submission_core)

    manifest = reconciliation.build_reconciliation_manifest(
        submission, binding, activation_content_sha256=HASH_A
    )

    assert manifest["metrics"]["existing_voice_binding_proposal_count"] == 1
    assert manifest["voice_binding_proposals"][0]["slot_id"] == (
        "recording-2::SPEAKER_2"
    )


def test_reconciliation_rejects_source_or_voice_binding_drift() -> None:
    submission, binding = _sources()
    tampered = deepcopy(submission)
    tampered["decisions"][0]["label"] = "Changed"
    with pytest.raises(
        reconciliation.Plan0063ReconciliationError,
        match="human-gold source",
    ):
        reconciliation.build_reconciliation_manifest(
            tampered, binding, activation_content_sha256=HASH_A
        )

    mismatched = deepcopy(binding)
    mismatched["bindings"][0]["acoustic_subject_id"] = "different-subject"
    mismatched = _with_hash(
        {key: value for key, value in mismatched.items() if key != "content_sha256"}
    )
    with pytest.raises(
        reconciliation.Plan0063ReconciliationError,
        match="source binding",
    ):
        reconciliation.build_reconciliation_manifest(
            submission, mismatched, activation_content_sha256=HASH_A
        )


def test_private_reconciliation_replay_checks_hashes_and_modes(tmp_path) -> None:
    submission, binding = _sources()
    manifest = reconciliation.build_reconciliation_manifest(
        submission, binding, activation_content_sha256=HASH_A
    )
    paths = reconciliation._reconciliation_paths(
        tmp_path, manifest["content_sha256"]
    )
    ensure_private_tree(paths["root"], paths["run"])
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": reconciliation.RECONCILIATION_RECEIPT_SCHEMA,
        "status": "private_reconciliation_frozen_pending_review",
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": sha256_file(paths["manifest"]),
        "metrics": manifest["metrics"],
        "live_mutation_count": 0,
        "negative_actions_preserved": True,
    }
    write_immutable_private_json(paths["receipt"], receipt)

    replay = reconciliation.replay_reconciliation(
        content_sha256=manifest["content_sha256"], runtime_root=tmp_path
    )

    assert replay["idempotent_replay"] is True
    assert paths["run"].stat().st_mode & 0o777 == 0o700
    assert paths["manifest"].stat().st_mode & 0o777 == 0o600
    assert paths["receipt"].stat().st_mode & 0o777 == 0o600
