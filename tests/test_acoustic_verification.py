from __future__ import annotations

import json
import hashlib
import sqlite3
import stat
import struct
import wave
from pathlib import Path

import pytest
import acoustic_verification as verification

from acoustic_verification import (
    AcousticVerificationError,
    FakeVerificationAdapter,
    adapter_registry,
    apply_real_enrollment,
    apply_development_trials,
    build_real_enrollment_apply_authority,
    build_development_trial_authority,
    build_real_enrollment_candidate_proposal,
    build_real_enrollment_preview,
    cosine_score,
    dry_run_model_acquisition,
    replay_model_acquisition,
    replay_real_enrollment_apply_authority,
    replay_development_trial_application,
    replay_real_enrollment_application,
    replay_real_enrollment_candidate_proposal,
    replay_real_enrollment_preview,
    materialize_profile,
    delete_profile,
    replay_profile,
    score_profile,
    supersede_profile,
    acknowledge_parent_reference_supersession,
    withdraw_profile,
)


def real_enrollment_authority_inputs() -> tuple[dict, dict, dict]:
    resolved = production_reference()
    source = resolved["reference"]["sources"][0]
    proposal = {
        "status": "ready_for_operator_review",
        "reason_codes": [],
        "candidates": [
            {
                "person_ref_id": resolved["person_ref_id"],
                "proposed_p3_profile_id": resolved["profile_id"],
                "proposed_source_set_sha256": resolved["reference"][
                    "source_set_sha256"
                ],
                "proposed_sources": [source],
            }
        ],
    }
    preview = {
        "status": "ready_for_review",
        "reason_codes": [],
        "enrollment_units": [
            {
                "person_ref_id": resolved["person_ref_id"],
                "p3_profile_id": resolved["profile_id"],
                "p3_generation_id": resolved["generation_id"],
                "p3_generation_sha256": resolved["generation_sha256"],
                "p3_source_set_sha256": resolved["reference"][
                    "source_set_sha256"
                ],
                "p3_approval_id": resolved["reference"]["approval"]["approval_id"],
                "p3_approval_sha256": verification.canonical_artifact_hash(
                    resolved["reference"]["approval"]
                ),
                "source_segments": [verification._enrollment_source_binding(source)],
            }
        ],
        "models": [
            {"candidate_id": "synthetic-verifier", "revision_sha": "1" * 40}
        ],
    }
    return proposal, preview, resolved


def production_reference() -> dict:
    approval = {
        "schema_version": "transcribe-audio.biometric-reference-approval.v1",
        "approval_id": "biometric-approval-001",
        "reviewer_ref_id": "reviewer-ref-001",
        "reviewed_at": "2026-07-31T12:00:00Z",
        "purpose": "biometric_reference_create",
        "scope": {"profile_id": "reference-profile-001"},
    }
    return {
        "profile_id": "reference-profile-001",
        "person_ref_id": "person-ref-001",
        "generation_id": "reference-generation-001",
        "generation_sha256": "a" * 64,
        "materialization_contract": "stage_then_register_then_promote",
        "reference": {
            "synthetic_test_only": False,
            "source_set_sha256": "b" * 64,
            "approval": approval,
            "sources": [
                {
                    "reference_id": "reference-segment-001",
                    "source_sha256": "c" * 64,
                    "recording_id": "recording-001",
                    "conversation_id": "conversation-001",
                    "speaker_label_id": "speaker-label-001",
                    "session_id": "session-001",
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                    "source_key": "d" * 64,
                    "quality_evidence": {"sha256": "e" * 64},
                    "lineage": {
                        "authority": "p2_speech_preparation_replay",
                        "replay_receipt_sha256": "f" * 64,
                    },
                }
            ],
        },
    }


def split_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    parent = tmp_path / "parent-corpus.json"
    recordings = [
        {
            "recording_id": "recording-001",
            "conversation_id": "conversation-001",
            "split": "development",
        },
        {
            "recording_id": "recording-002",
            "conversation_id": "conversation-002",
            "split": "calibration",
        },
    ]
    parent.write_text(json.dumps({"recordings": recordings}), encoding="utf-8")
    parent.chmod(0o600)
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    record_set_sha = verification.canonical_artifact_hash(recordings[:1])
    conversation_set_sha = verification.canonical_artifact_hash(
        ["conversation-001"]
    )
    policy = {
        "schema_version": "transcribe-audio.verification-split-access-policy.v1",
        "parent_corpus_manifest_sha256": parent_sha,
        "splits": {
            "development": {
                "recording_count": 1,
                "conversation_count": 1,
                "record_set_sha256": record_set_sha,
                "conversation_set_sha256": conversation_set_sha,
                "authorization_state": "authorized_by_operator_blanket_2026-07-31",
            }
        },
    }
    policy_path = tmp_path / "split-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(
        verification, "EXPECTED_SPLIT_ACCESS_POLICY_SHA256",
        hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        verification, "EXPECTED_PARENT_CORPUS_MANIFEST_SHA256", parent_sha
    )
    monkeypatch.setattr(
        verification, "EXPECTED_DEVELOPMENT_RECORD_SET_SHA256", record_set_sha
    )
    monkeypatch.setattr(
        verification,
        "EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256",
        conversation_set_sha,
    )
    return policy_path, parent


def candidate_proposal_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    same_subject: bool = True,
    multiple_labels_same_subject: bool = False,
    reviewed_matches_current: bool = True,
    include_source_campaign: bool = False,
) -> tuple[Path, Path, Path]:
    records = []
    selected = []
    lineage_by_run = {}
    for index in (1, 2):
        recording_id = f"recording-00{index}"
        conversation_id = f"conversation-00{index}"
        run_id = f"speech-prep-test-{index:03d}"
        transcript_path = tmp_path / "transcripts" / f"{recording_id}.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.parent.chmod(0o700)
        transcript = {
            "schema_version": 2,
            "recording_id": recording_id,
            "conversation_id": conversation_id,
            "utterances": [
                {"speaker": "A", "start": 1000, "end": 3000},
                {"speaker": "A", "start": 4000, "end": 6000},
                {"speaker": "A", "start": 7000, "end": 9000},
                *(
                    [
                        {"speaker": "B", "start": 1500, "end": 3500},
                        {"speaker": "B", "start": 4500, "end": 6500},
                        {"speaker": "B", "start": 9250, "end": 10000},
                    ]
                    if multiple_labels_same_subject
                    else []
                ),
            ],
        }
        transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
        transcript_path.chmod(0o600)
        transcript_sha = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
        subject = "subject-review-001" if same_subject else f"subject-review-00{index}"
        record = {
            "recording_id": recording_id,
            "conversation_id": conversation_id,
            "split": "development",
            "conditions": {
                "channel": "mono",
                "device": "synthetic-metadata",
                "noise": "low",
                "overlap": "none",
                "telephone_bandwidth": "false",
                "usable_duration_band": "short",
            },
            "transcript_lineage": {
                "current_artifact_path": str(transcript_path),
                "current_artifact_sha256": transcript_sha,
                "reviewed_artifact_sha256": (
                    transcript_sha if reviewed_matches_current else "0" * 64
                ),
            },
            "operator_gold": {
                "gold_id": f"gold-00{index}",
                "prediction_visibility": "sealed",
                "review_method": "synthetic-test",
                "reviewed_at": "2026-07-31T12:00:00Z",
                "same_person_label_groups": [],
                "speaker_truth": [
                    {
                        "outcome": "person",
                        "speaker_label": "A",
                        "subject_id": subject,
                    },
                    *(
                        [
                            {
                                "outcome": "person",
                                "speaker_label": "B",
                                "subject_id": subject,
                            }
                        ]
                        if multiple_labels_same_subject
                        else []
                    ),
                ],
            },
        }
        records.append(record)
        comparison_path = (
            tmp_path / "p2" / f"unit-{index}" / "runs" / run_id / "comparison.json"
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.parent.chmod(0o700)
        comparison = {
            "method_results": [
                {
                    "method_id": "pyannote_community_1",
                    "status": "success",
                    "speech_regions": [{"start_seconds": 0.0, "end_seconds": 10.0}],
                    "overlap_regions": [],
                    "speaker_change_regions": [],
                }
            ]
        }
        comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
        comparison_path.chmod(0o600)
        comparison_sha = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        replay_sha = ("a" if index == 1 else "b") * 64
        selected.append(
            {
                "recording_id": recording_id,
                "split": "development",
                "comparison_path": str(comparison_path),
                "comparison_sha256": comparison_sha,
                "p2_run_id": run_id,
                "replay_sha256": replay_sha,
            }
        )
        lineage_by_run[run_id] = {
            "schema_version": "transcribe-audio.speech-preparation-lineage.v1",
            "authority": "p2_speech_preparation_replay",
            "run_id": run_id,
            "runtime_root": str(comparison_path.parents[2]),
            "method_id": "no_enhancement",
            "replay_receipt_path": str(comparison_path.parents[2] / "replay.json"),
            "replay_receipt_sha256": replay_sha,
            "comparison_path": str(comparison_path),
            "comparison_sha256": comparison_sha,
            "method_result_sha256": ("c" if index == 1 else "d") * 64,
            "source_blob_id": f"source-blob-00{index}",
            "source_sha256": ("e" if index == 1 else "f") * 64,
            "source_duration_seconds": 20.0,
            "audio_quality_sha256": ("1" if index == 1 else "2") * 64,
            "validation_status": "verified_active_metadata_receipt",
            "will_read_audio": False,
        }
    parent = tmp_path / "candidate-parent.json"
    parent_payload = {"recordings": records}
    if include_source_campaign:
        parent_payload["source_campaign"] = {
            "campaign_id": "campaign-test-continuity-0001",
            "authority_hashes": {
                "campaign_manifest_sha256": "3" * 64,
                "gold_index_sha256": "4" * 64,
            },
        }
    parent.write_text(json.dumps(parent_payload), encoding="utf-8")
    parent.chmod(0o600)
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    record_set_sha = verification.canonical_artifact_hash(records)
    conversation_set_sha = verification.canonical_artifact_hash(
        sorted(record["conversation_id"] for record in records)
    )
    policy = {
        "schema_version": "transcribe-audio.verification-split-access-policy.v1",
        "parent_corpus_manifest_sha256": parent_sha,
        "splits": {
            "development": {
                "recording_count": 2,
                "conversation_count": 2,
                "record_set_sha256": record_set_sha,
                "conversation_set_sha256": conversation_set_sha,
                "authorization_state": "authorized_by_operator_blanket_2026-07-31",
            }
        },
    }
    policy_path = tmp_path / "candidate-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    joined = {
        "schema_version": "transcribe-audio.speech-preparation-development-comparison.v2",
        "status": "success",
        "corpus_manifest_sha256": parent_sha,
        "selected_recordings": selected,
        "will_run_biometrics": False,
        "will_read_calibration_or_evaluation": False,
        "will_perform_external_write": False,
    }
    joined_path = tmp_path / "joined-development.json"
    joined_path.write_text(json.dumps(joined), encoding="utf-8")
    joined_path.chmod(0o600)
    monkeypatch.setattr(
        verification,
        "EXPECTED_SPLIT_ACCESS_POLICY_SHA256",
        hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        verification, "EXPECTED_PARENT_CORPUS_MANIFEST_SHA256", parent_sha
    )
    monkeypatch.setattr(
        verification, "EXPECTED_DEVELOPMENT_RECORD_SET_SHA256", record_set_sha
    )
    monkeypatch.setattr(
        verification,
        "EXPECTED_DEVELOPMENT_CONVERSATION_SET_SHA256",
        conversation_set_sha,
    )
    monkeypatch.setattr(
        verification,
        "EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256",
        hashlib.sha256(joined_path.read_bytes()).hexdigest(),
    )

    def fake_lineage(run_id: str, **kwargs) -> dict:
        expected = lineage_by_run[run_id]
        assert kwargs["method_id"] == "no_enhancement"
        assert kwargs["replay_receipt_sha256"] == expected["replay_receipt_sha256"]
        return dict(expected)

    monkeypatch.setattr(verification, "resolve_comparison_lineage_receipt", fake_lineage)
    monkeypatch.setattr(
        "acoustic_biometric_references.resolve_comparison_lineage_receipt",
        fake_lineage,
    )
    return policy_path, parent, joined_path


def test_real_enrollment_preview_persists_truthful_no_store_blocker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    root = tmp_path / "p4"
    preview = build_real_enrollment_preview(
        [], runtime_root=root, p3_runtime_root=tmp_path / "missing-p3",
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )

    assert preview["status"] == "blocked"
    assert preview["reason_codes"] == [
        "no_requested_people",
        "p3_reference_store_unavailable",
    ]
    assert preview["enrollment_units"] == []
    assert preview["real_biometric_enrollment_authorized"] is False
    assert replay_real_enrollment_preview(
        preview["preview_sha256"], runtime_root=root,
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    ) == preview
    assert stat.S_IMODE(Path(preview["private_preview_path"]).stat().st_mode) == 0o600


def test_real_enrollment_preview_binds_exact_production_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    p3_root = tmp_path / "p3"
    p3_root.mkdir()
    (p3_root / "references.sqlite3").touch()
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: production_reference(),
    )

    preview = build_real_enrollment_preview(
        ["person-ref-001"], runtime_root=tmp_path / "p4", p3_runtime_root=p3_root,
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )

    assert preview["status"] == "ready_for_review"
    assert preview["reason_codes"] == []
    assert [item["candidate_id"] for item in preview["models"]] == [
        "speechbrain_ecapa_tdnn",
        "wespeaker_campplus",
        "wespeaker_resnet34",
    ]
    unit = preview["enrollment_units"][0]
    assert unit["p3_generation_sha256"] == "a" * 64
    assert unit["p3_source_set_sha256"] == "b" * 64
    assert unit["source_segments"][0]["segment_sha256"] == "d" * 64
    assert unit["source_segments"][0]["lineage_replay_receipt_sha256"] == "f" * 64
    assert preview["will_read_audio"] is False
    assert preview["will_materialize_embeddings"] is False
    assert preview["acquisition_manifest_sha256"] == (
        "6470ecc8591fd8a40f8d788ba9a3edddc37a508cc54d47800037ab594b957ebe"
    )


def test_real_enrollment_preview_rejects_synthetic_and_non_development_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    p3_root = tmp_path / "p3"
    p3_root.mkdir()
    (p3_root / "references.sqlite3").touch()
    synthetic = production_reference()
    synthetic["reference"]["synthetic_test_only"] = True
    synthetic["reference"]["sources"][0]["fixture_authority"] = {}
    monkeypatch.setattr(
        verification, "resolve_eligible_reference", lambda *args, **kwargs: synthetic
    )

    preview = build_real_enrollment_preview(
        ["person-ref-001"], runtime_root=tmp_path / "p4", p3_runtime_root=p3_root,
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )
    assert preview["status"] == "blocked"
    assert preview["reason_codes"] == ["no_replay_eligible_real_p3_generation"]
    with pytest.raises(AcousticVerificationError, match="development split"):
        build_real_enrollment_preview(
            [],
            runtime_root=tmp_path / "p4",
            p3_runtime_root=p3_root,
            intended_split="calibration",
            split_policy_path=policy,
            parent_corpus_manifest_path=parent,
        )


def test_real_enrollment_preview_rejects_non_development_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    p3_root = tmp_path / "p3"
    p3_root.mkdir()
    (p3_root / "references.sqlite3").touch()
    reference = production_reference()
    reference["reference"]["sources"][0]["recording_id"] = "recording-002"
    reference["reference"]["sources"][0]["conversation_id"] = "conversation-002"
    monkeypatch.setattr(
        verification, "resolve_eligible_reference", lambda *args, **kwargs: reference
    )
    preview = build_real_enrollment_preview(
        ["person-ref-001"], runtime_root=tmp_path / "p4", p3_runtime_root=p3_root,
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )
    assert preview["status"] == "blocked"
    assert preview["reason_codes"] == ["no_replay_eligible_real_p3_generation"]


@pytest.mark.parametrize("mutation", ["split", "models", "status", "source"])
def test_real_enrollment_preview_replay_rejects_forged_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    p3_root = tmp_path / "p3"
    p3_root.mkdir()
    (p3_root / "references.sqlite3").touch()
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: production_reference(),
    )
    root = tmp_path / "p4"
    preview = build_real_enrollment_preview(
        ["person-ref-001"], runtime_root=root, p3_runtime_root=p3_root,
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )
    forged = {
        key: value for key, value in preview.items()
        if key not in {"preview_sha256", "private_preview_path"}
    }
    if mutation == "split":
        forged["intended_split"] = "calibration"
    elif mutation == "models":
        forged["models"] = forged["models"][:2]
    elif mutation == "status":
        forged["status"] = "blocked"
    else:
        forged["enrollment_units"][0]["source_segments"][0][
            "recording_id"
        ] = "recording-002"
    forged_sha = verification.canonical_artifact_hash(forged)
    forged_path = root / "enrollment-previews" / f"{forged_sha}.json"
    verification.write_immutable_private_json(forged_path, forged)
    with pytest.raises(AcousticVerificationError, match="[Ee]nrollment"):
        replay_real_enrollment_preview(
            forged_sha, runtime_root=root, split_policy_path=policy,
            parent_corpus_manifest_path=parent,
        )


def test_real_enrollment_preview_replay_rejects_forged_reason_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent = split_authority(monkeypatch, tmp_path)
    root = tmp_path / "p4"
    preview = build_real_enrollment_preview(
        [], runtime_root=root, p3_runtime_root=tmp_path / "missing-p3",
        split_policy_path=policy, parent_corpus_manifest_path=parent,
    )
    forged = {
        key: value for key, value in preview.items()
        if key not in {"preview_sha256", "private_preview_path"}
    }
    forged["reason_codes"] = ["no_replay_eligible_real_p3_generation"]
    forged_sha = verification.canonical_artifact_hash(forged)
    verification.write_immutable_private_json(
        root / "enrollment-previews" / f"{forged_sha}.json", forged
    )
    with pytest.raises(AcousticVerificationError, match="reasons"):
        replay_real_enrollment_preview(
            forged_sha, runtime_root=root, split_policy_path=policy,
            parent_corpus_manifest_path=parent,
        )


def test_real_enrollment_apply_authority_is_exact_private_and_replayable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal, preview, resolved = real_enrollment_authority_inputs()
    proposal_sha = "2" * 64
    preview_sha = "3" * 64
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_candidate_proposal",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_preview",
        lambda *args, **kwargs: preview,
    )
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: resolved,
    )
    root = tmp_path / "p4"
    authority = build_real_enrollment_apply_authority(
        proposal_sha,
        preview_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )

    assert authority["status"] == "authorized"
    assert authority["candidate_proposal_sha256"] == proposal_sha
    assert authority["enrollment_preview_sha256"] == preview_sha
    assert authority["real_biometric_enrollment_authorized"] is True
    assert authority["will_read_audio"] is True
    assert authority["will_materialize_embeddings"] is True
    assert authority["will_run_trials"] is False
    assert authority["will_read_calibration_or_evaluation"] is False
    assert authority["contains_raw_biometric_values"] is False
    assert replay_real_enrollment_apply_authority(
        authority["authority_sha256"], runtime_root=root
    ) == authority
    repeated = build_real_enrollment_apply_authority(
        proposal_sha,
        preview_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert repeated == authority
    path = Path(authority["private_authority_path"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["will_run_trials"] = True
    path.write_text(json.dumps(forged), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(AcousticVerificationError, match="authority"):
        replay_real_enrollment_apply_authority(
            authority["authority_sha256"], runtime_root=root
        )


def test_real_enrollment_apply_authority_rejects_proposal_preview_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal, preview, resolved = real_enrollment_authority_inputs()
    preview["enrollment_units"][0]["p3_source_set_sha256"] = "9" * 64
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_candidate_proposal",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_preview",
        lambda *args, **kwargs: preview,
    )
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: resolved,
    )

    with pytest.raises(AcousticVerificationError, match="bindings differ"):
        build_real_enrollment_apply_authority(
            "2" * 64,
            "3" * 64,
            runtime_root=tmp_path / "p4",
            p3_runtime_root=tmp_path / "p3",
        )


def test_candidate_enrollment_proposal_is_exact_private_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(monkeypatch, tmp_path)
    root = tmp_path / "p4"
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    )

    assert proposal["status"] == "ready_for_operator_review"
    assert proposal["biometric_enrollment_authorized"] is False
    assert proposal["proposal_only"] is True
    assert proposal["denominators"] == {
        "selected_development_recordings": 2,
        "reviewed_artifact_lineage_exclusions": 0,
        "eligible_reviewed_artifact_recordings": 2,
        "person_rows_considered": 2,
        "candidate_people": 1,
        "candidate_sessions": 2,
        "candidate_windows": 6,
    }
    candidate = proposal["candidates"][0]
    assert candidate["person_ref_id"] == "subject-review-001"
    assert candidate["session_count"] == 2
    assert candidate["window_count"] == 6
    assert candidate["operator_decision_required"] is True
    assert all(
        source["lineage"]["will_read_audio"] is False
        for source in candidate["proposed_sources"]
    )
    assert replay_real_enrollment_candidate_proposal(
        proposal["proposal_sha256"],
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    ) == proposal
    assert stat.S_IMODE(Path(proposal["private_proposal_path"]).stat().st_mode) == 0o600


def test_candidate_enrollment_proposal_caps_multiple_labels_per_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(
        monkeypatch, tmp_path, multiple_labels_same_subject=True
    )
    root = tmp_path / "p4"
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    )
    candidate = proposal["candidates"][0]
    assert candidate["window_count"] == 6
    for session_id in {item["session_id"] for item in candidate["proposed_sources"]}:
        session_sources = [
            item for item in candidate["proposed_sources"]
            if item["session_id"] == session_id
        ]
        assert len(session_sources) == 3
        assert all(
            left["end_seconds"] <= right["start_seconds"]
            or right["end_seconds"] <= left["start_seconds"]
            for index, left in enumerate(session_sources)
            for right in session_sources[index + 1:]
        )
    assert replay_real_enrollment_candidate_proposal(
        proposal["proposal_sha256"],
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    ) == proposal


def test_candidate_enrollment_proposal_blocks_reviewed_artifact_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(
        monkeypatch, tmp_path, reviewed_matches_current=False
    )
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=tmp_path / "p4",
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    )
    assert proposal["status"] == "blocked"
    assert proposal["reason_codes"] == [
        "reviewed_artifact_lineage_drift",
        "no_multi_session_clean_candidates",
    ]
    assert proposal["denominators"]["reviewed_artifact_lineage_exclusions"] == 2
    assert proposal["denominators"]["eligible_reviewed_artifact_recordings"] == 0
    assert proposal["candidates"] == []


def test_candidate_enrollment_proposal_recovers_reviewed_clue_continuity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(
        monkeypatch,
        tmp_path,
        reviewed_matches_current=False,
        include_source_campaign=True,
    )
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    campaign_id = parent_payload["source_campaign"]["campaign_id"]
    campaign_root = tmp_path / "campaigns"
    runs_root = tmp_path / "runs"
    campaign_root.mkdir(mode=0o700)
    runs_root.mkdir(mode=0o700)
    packet_paths = []
    prediction_paths = []
    events_paths = []
    authority_entries = []
    for index, record in enumerate(parent_payload["recordings"], start=1):
        document_id = f"document-00{index}"
        record["document_id"] = document_id
        transcript_path = Path(record["transcript_lineage"]["current_artifact_path"])
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        run_id = f"20260731T12000{index}Z-speaker-preprocessing-test{index}"
        packet = {
            "schema_version": "transcribe-audio.speaker-clue-discovery-packet.v1",
            "task": "speaker_clue_discovery",
            "conversation": {
                "conversation_id": record["conversation_id"],
                "recording_ids": [record["recording_id"]],
                "title": "Synthetic",
            },
            "speakers": [
                {
                    "speaker_label": "A",
                    "utterance_clues": [
                        {
                            "utterance_id": f"utterance-{ordinal}",
                            "start": utterance["start"],
                            "end": utterance["end"],
                            "text": str(utterance.get("text") or "")[:1_200],
                        }
                        for ordinal, utterance in enumerate(
                            transcript["utterances"], start=1
                        )
                    ],
                }
            ],
        }
        packet_path = (
            runs_root
            / run_id
            / "artifacts/speaker-preprocessing/clue_discovery.input.json"
        )
        packet_path.parent.mkdir(parents=True, mode=0o700)
        for ancestor in (
            packet_path.parent,
            packet_path.parent.parent,
            packet_path.parent.parent.parent,
        ):
            ancestor.chmod(0o700)
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
        packet_path.chmod(0o600)
        packet_paths.append(packet_path)
        run_root = runs_root / run_id
        run_path = run_root / "run.json"
        run_path.write_text(
            json.dumps(
                {
                    "schema_version": "transcribe-audio.app-intelligence-run.v1",
                    "run_id": run_id,
                    "document_id": document_id,
                    "workflow": "speaker_preprocessing",
                }
            ),
            encoding="utf-8",
        )
        events_path = run_root / "events.jsonl"
        events_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "event_type": "model_turn_status_captured",
                    "payload": {"completed": True, "status": "completed"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        events_paths.append(events_path)
        prompt_text = (
            "Inspect this clue packet:\n"
            + json.dumps(packet, sort_keys=True, ensure_ascii=False)
        )
        prompt_text_path = run_root / "prompt.txt"
        prompt_text_path.write_text(prompt_text, encoding="utf-8")
        prompt_packet_path = run_root / "prompt.json"
        prompt_packet_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "task": "speaker_clue_discovery",
                    "document": {"id": document_id},
                    "prompt_text": prompt_text,
                }
            ),
            encoding="utf-8",
        )
        status_path = run_root / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "transcribe-audio.app-intelligence-model-turn-status.v1"
                    ),
                    "run_id": run_id,
                    "status": "completed",
                    "completed": True,
                    "will_execute_structured_decision": False,
                }
            ),
            encoding="utf-8",
        )
        for witness_path in (
            run_path,
            events_path,
            prompt_text_path,
            prompt_packet_path,
            status_path,
        ):
            witness_path.chmod(0o600)
        prediction = {
            "schema_version": "transcribe-audio.speaker-evaluation-blind-prediction.v1",
            "campaign_id": campaign_id,
            "baseline_id": "baseline-test-0001",
            "document_id": document_id,
            "artifact_sha256": record["transcript_lineage"]["reviewed_artifact_sha256"],
            "captured_at": f"2026-07-31T12:00:0{index}Z",
            "prediction_visibility": "blind",
            "gold_content_included": False,
            "will_read_gold_records": False,
            "will_perform_external_write": False,
            "run_references": {"clue_discovery_run_id": run_id},
        }
        prediction_path = (
            campaign_root
            / campaign_id
            / "baselines/baseline-test-0001/predictions"
            / document_id
            / f"prediction-{index}.json"
        )
        prediction_path.parent.mkdir(parents=True, mode=0o700)
        current = prediction_path.parent
        while current != campaign_root:
            current.chmod(0o700)
            current = current.parent
        prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
        prediction_path.chmod(0o600)
        prediction_paths.append(prediction_path)
        authority_entries.append(
            {
                "document_id": document_id,
                "recording_id": record["recording_id"],
                "conversation_id": record["conversation_id"],
                "reviewed_artifact_sha256": record["transcript_lineage"][
                    "reviewed_artifact_sha256"
                ],
                "blind_prediction_relative_path": str(
                    prediction_path.relative_to(campaign_root / campaign_id)
                ),
                "blind_prediction_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
                "clue_discovery_run_id": run_id,
                "clue_packet_relative_path": str(packet_path.relative_to(run_root)),
                "clue_packet_sha256": hashlib.sha256(
                    packet_path.read_bytes()
                ).hexdigest(),
                "run_json_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "events_jsonl_sha256": hashlib.sha256(
                    events_path.read_bytes()
                ).hexdigest(),
                "prompt_packet_relative_path": str(
                    prompt_packet_path.relative_to(run_root)
                ),
                "prompt_packet_sha256": hashlib.sha256(
                    prompt_packet_path.read_bytes()
                ).hexdigest(),
                "prompt_text_relative_path": str(
                    prompt_text_path.relative_to(run_root)
                ),
                "prompt_text_sha256": hashlib.sha256(
                    prompt_text_path.read_bytes()
                ).hexdigest(),
                "status_relative_path": str(status_path.relative_to(run_root)),
                "status_sha256": hashlib.sha256(
                    status_path.read_bytes()
                ).hexdigest(),
            }
        )
    authority_path = tmp_path / "reviewed-clue-authority.json"
    authority_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "transcribe-audio.reviewed-clue-continuity-authority.v1"
                ),
                "campaign_id": campaign_id,
                "campaign_manifest_sha256": "3" * 64,
                "gold_index_sha256": "4" * 64,
                "contains_transcript_text": False,
                "will_read_audio": False,
                "will_authorize_biometric_enrollment": False,
                "entries": authority_entries,
            }
        ),
        encoding="utf-8",
    )
    authority_path.chmod(0o600)
    monkeypatch.setattr(
        verification,
        "EXPECTED_REVIEWED_CLUE_CONTINUITY_AUTHORITY_SHA256",
        hashlib.sha256(authority_path.read_bytes()).hexdigest(),
    )
    parent.write_text(json.dumps(parent_payload), encoding="utf-8")
    parent.chmod(0o600)
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    records = parent_payload["recordings"]
    monkeypatch.setattr(verification, "EXPECTED_PARENT_CORPUS_MANIFEST_SHA256", parent_sha)
    monkeypatch.setattr(
        verification,
        "EXPECTED_DEVELOPMENT_RECORD_SET_SHA256",
        verification.canonical_artifact_hash(records),
    )
    policy_payload = json.loads(policy.read_text(encoding="utf-8"))
    policy_payload["parent_corpus_manifest_sha256"] = parent_sha
    policy_payload["splits"]["development"]["record_set_sha256"] = (
        verification.canonical_artifact_hash(records)
    )
    policy.write_text(json.dumps(policy_payload), encoding="utf-8")
    monkeypatch.setattr(
        verification,
        "EXPECTED_SPLIT_ACCESS_POLICY_SHA256",
        hashlib.sha256(policy.read_bytes()).hexdigest(),
    )
    joined_payload = json.loads(joined.read_text(encoding="utf-8"))
    joined_payload["corpus_manifest_sha256"] = parent_sha
    joined.write_text(json.dumps(joined_payload), encoding="utf-8")
    joined.chmod(0o600)
    monkeypatch.setattr(
        verification,
        "EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256",
        hashlib.sha256(joined.read_bytes()).hexdigest(),
    )

    root = tmp_path / "p4"
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
        campaign_root=campaign_root,
        app_intelligence_runs_root=runs_root,
        reviewed_clue_continuity_authority_path=authority_path,
    )
    assert proposal["status"] == "ready_for_operator_review"
    assert proposal["reason_codes"] == []
    assert proposal["denominators"]["reviewed_artifact_lineage_exclusions"] == 0
    assert proposal["denominators"]["eligible_reviewed_artifact_recordings"] == 2
    assert proposal["denominators"]["candidate_people"] == 1
    assert {
        evidence["reviewed_clue_continuity"]["mode"]
        for evidence in proposal["candidates"][0]["selection_evidence"]
    } == {"committed_reviewed_clue_authority"}
    assert replay_real_enrollment_candidate_proposal(
        proposal["proposal_sha256"],
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
        campaign_root=campaign_root,
        app_intelligence_runs_root=runs_root,
        reviewed_clue_continuity_authority_path=authority_path,
    ) == proposal

    for case, witness_path in (
        ("packet-sha-mismatch", packet_paths[0]),
        ("prediction-drift", prediction_paths[0]),
        ("run-ledger-drift", events_paths[0]),
    ):
        original = witness_path.read_bytes()
        witness_path.write_bytes(original + b" ")
        witness_path.chmod(0o600)
        blocked = build_real_enrollment_candidate_proposal(
            runtime_root=tmp_path / case,
            split_policy_path=policy,
            parent_corpus_manifest_path=parent,
            development_comparison_receipt_path=joined,
            campaign_root=campaign_root,
            app_intelligence_runs_root=runs_root,
            reviewed_clue_continuity_authority_path=authority_path,
        )
        assert blocked["status"] == "blocked"
        assert blocked["reason_codes"] == [
            "reviewed_artifact_lineage_drift",
            "no_multi_session_clean_candidates",
        ]
        assert blocked["denominators"]["reviewed_artifact_lineage_exclusions"] == 1
        assert blocked["candidates"] == []
        witness_path.write_bytes(original)
        witness_path.chmod(0o600)

    packet = json.loads(packet_paths[0].read_text(encoding="utf-8"))
    packet["speakers"][0]["utterance_clues"][0]["start"] += 1
    packet_paths[0].write_text(json.dumps(packet), encoding="utf-8")
    packet_paths[0].chmod(0o600)
    with pytest.raises(AcousticVerificationError, match="proposal replay"):
        replay_real_enrollment_candidate_proposal(
            proposal["proposal_sha256"],
            runtime_root=root,
            split_policy_path=policy,
            parent_corpus_manifest_path=parent,
            development_comparison_receipt_path=joined,
            campaign_root=campaign_root,
            app_intelligence_runs_root=runs_root,
            reviewed_clue_continuity_authority_path=authority_path,
        )


def test_candidate_enrollment_proposal_blocks_single_session_people(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(
        monkeypatch, tmp_path, same_subject=False
    )
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=tmp_path / "p4",
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    )
    assert proposal["status"] == "blocked"
    assert proposal["reason_codes"] == ["no_multi_session_clean_candidates"]
    assert proposal["candidates"] == []


def test_candidate_enrollment_proposal_replay_rejects_source_or_receipt_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy, parent, joined = candidate_proposal_authority(monkeypatch, tmp_path)
    root = tmp_path / "p4"
    proposal = build_real_enrollment_candidate_proposal(
        runtime_root=root,
        split_policy_path=policy,
        parent_corpus_manifest_path=parent,
        development_comparison_receipt_path=joined,
    )
    forged = {
        key: value for key, value in proposal.items()
        if key not in {"proposal_sha256", "private_proposal_path"}
    }
    forged["biometric_enrollment_authorized"] = True
    forged_sha = verification.canonical_artifact_hash(forged)
    verification.write_immutable_private_json(
        root / "enrollment-proposals" / f"{forged_sha}.json", forged
    )
    with pytest.raises(AcousticVerificationError, match="proposal replay"):
        replay_real_enrollment_candidate_proposal(
            forged_sha,
            runtime_root=root,
            split_policy_path=policy,
            parent_corpus_manifest_path=parent,
            development_comparison_receipt_path=joined,
        )

    transcript_path = next((tmp_path / "transcripts").glob("*.json"))
    transcript_path.write_text(
        transcript_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    transcript_path.chmod(0o600)
    with pytest.raises(AcousticVerificationError, match="Transcript artifact hash"):
        replay_real_enrollment_candidate_proposal(
            proposal["proposal_sha256"],
            runtime_root=root,
            split_policy_path=policy,
            parent_corpus_manifest_path=parent,
            development_comparison_receipt_path=joined,
        )


def test_model_acquisition_dry_run_is_immutable_and_side_effect_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "p4-acquisition"

    plan = dry_run_model_acquisition(runtime_root=root)

    assert plan["status"] == "success"
    assert plan["reason_code"] is None
    assert plan["authorization_basis"] == "operator_blanket_2026-07-31"
    assert plan["spec"]["authorization_scope"] == (
        "plan_0037_model_acquisition_install_and_development_processing_only"
    )
    assert plan["spec"]["real_biometric_enrollment_authorized"] is False
    assert [item["candidate_id"] for item in plan["spec"]["candidates"]] == [
        "speechbrain_ecapa_tdnn",
        "wespeaker_campplus",
        "wespeaker_resnet34",
    ]
    for field in (
        "will_download",
        "will_install",
        "will_build",
        "will_read_audio",
        "will_materialize_embeddings",
        "will_register_references",
        "will_run_trials",
        "will_perform_external_write",
    ):
        assert plan[field] is False

    replay = replay_model_acquisition(
        plan["run_id"],
        expected_dry_run_sha256=plan["dry_run_sha256"],
        runtime_root=root,
    )
    assert replay["dry_run_sha256"] == plan["dry_run_sha256"]
    assert replay["spec_sha256"] == plan["spec_sha256"]
    for path in root.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_model_acquisition_replay_rejects_spec_drift(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    root = tmp_path / "p4-acquisition"
    plan = dry_run_model_acquisition(runtime_root=root, spec_path=spec)

    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["candidates"][0]["model"]["revision_sha"] = "0" * 40
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="spec drifted"):
        replay_model_acquisition(
            plan["run_id"],
            expected_dry_run_sha256=plan["dry_run_sha256"],
            runtime_root=root,
        )


def test_model_acquisition_rejects_real_enrollment_authority(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["real_biometric_enrollment_authorized"] = True
    spec = tmp_path / "acquisition.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="enrollment authority"):
        dry_run_model_acquisition(
            runtime_root=tmp_path / "p4-acquisition", spec_path=spec
        )


def test_model_acquisition_rejects_mutable_terms_authority(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_authorities"]["wespeaker_models"] = (
        "https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md"
    )
    spec = tmp_path / "acquisition.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AcousticVerificationError, match="authority binding"):
        dry_run_model_acquisition(
            runtime_root=tmp_path / "p4-acquisition", spec_path=spec
        )


def test_fake_adapter_is_deterministic_normalized_and_separates_trials() -> None:
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    same = [0.25, -0.25] * 8_000
    different = [0.75] * 8_000 + [-0.75] * 8_000

    first = adapter.embed(same, sample_rate=16_000)
    second = adapter.embed(same, sample_rate=16_000)
    other = adapter.embed(different, sample_rate=16_000)

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert cosine_score(first, second) == pytest.approx(1.0)
    assert cosine_score(first, other) < 0.95


@pytest.mark.parametrize(
    ("samples", "message"),
    [([], "empty"), ([0.0] * 100, "shorter"), ([float("nan")] * 16_000, "finite")],
)
def test_fake_adapter_rejects_invalid_audio(samples: list[float], message: str) -> None:
    with pytest.raises(AcousticVerificationError, match=message):
        FakeVerificationAdapter(candidate_id="synthetic_verifier").embed(
            samples, sample_rate=16_000
        )


def test_real_adapter_registry_is_lazy_and_exactly_pinned(tmp_path: Path) -> None:
    registry = adapter_registry(snapshot_root=tmp_path / "not-loaded")

    assert list(registry) == [
        "speechbrain_ecapa_tdnn",
        "wespeaker_campplus",
        "wespeaker_resnet34",
    ]
    assert [adapter.embedding_dimension for adapter in registry.values()] == [
        192,
        512,
        256,
    ]
    assert all(adapter.model_loaded is False for adapter in registry.values())


def test_model_load_preflight_replays_manifest_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "snapshot"
    model_root = root / "models" / "wespeaker_campplus"
    model_root.mkdir(parents=True, mode=0o700)
    for directory in (root, root / "models", model_root):
        directory.chmod(0o700)
    records = {}
    for name in verification.EXPECTED_CANDIDATES[
        "wespeaker_campplus"
    ]["artifact_paths"]:
        path = model_root / name
        path.write_bytes(name.encode())
        path.chmod(0o600)
        records[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    spec_sha = "d" * 64
    manifest = {
        "schema_version": "transcribe-audio.verification-model-acquisition-manifest.v1",
        "authorization_basis": verification.AUTHORIZATION_BASIS,
        "authorization_scope": verification.AUTHORIZATION_SCOPE,
        "spec_sha256": spec_sha,
        "snapshot_root": str(root),
        "real_biometric_enrollment_authorized": False,
        "audio_read": False,
        "embedding_materialized": False,
        "trial_executed": False,
        "installed_distributions": {"onnxruntime": "1.24.4"},
        "artifacts": {"wespeaker_campplus": records},
    }
    manifest_path = root / "acquisition-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_path.chmod(0o600)
    monkeypatch.setattr(
        verification, "EXPECTED_ACQUISITION_SPEC_SHA256", spec_sha
    )
    monkeypatch.setattr(
        verification,
        "EXPECTED_ACQUISITION_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    verified = verification._verified_model_artifacts(
        root, "wespeaker_campplus"
    )
    assert set(verified) == {"voxceleb_CAM++.onnx", "config.yaml"}
    (model_root / "config.yaml").write_bytes(b"tampered")
    with pytest.raises(AcousticVerificationError, match="artifact hash mismatch"):
        verification._verified_model_artifacts(root, "wespeaker_campplus")


def test_cosine_score_rejects_nonfinite_or_mismatched_embeddings() -> None:
    with pytest.raises(AcousticVerificationError, match="dimensions"):
        cosine_score((1.0, 0.0), (1.0,))
    with pytest.raises(AcousticVerificationError, match="finite"):
        cosine_score((1.0, 0.0), (float("inf"), 0.0))


def synthetic_reference() -> dict[str, object]:
    return {
        "profile_id": "reference-profile-001",
        "person_ref_id": "person-ref-001",
        "generation_id": "reference-generation-001",
        "generation_sha256": "1" * 64,
        "reference": {
            "schema_version": "synthetic-reference",
            "sources": [
                {
                    "source_sha256": "a" * 64,
                    "device_class": "synthetic-fixture",
                    "fixture_authority": {
                        "schema_version": "transcribe-audio.synthetic-reference-fixture.v1",
                        "source_sha256": "a" * 64,
                    },
                }
            ],
        },
        "materialization_contract": "stage_then_register_then_promote",
    }


def install_fake_p3(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda person_ref_id, **kwargs: (
            calls.append("resolve") or synthetic_reference()
        ),
    )

    def register(*args, **kwargs):
        calls.append("register")
        receipt = kwargs["materialization_receipt"]
        return {
            "descendant_id": args[2],
            "artifact_sha256": args[3],
            "materialization_receipt_sha256": verification.canonical_artifact_hash(
                receipt
            ),
            "required_promotion_token": "promotion-token",
            "state": "staged",
        }

    monkeypatch.setattr(verification, "register_descendant", register)
    monkeypatch.setattr(
        verification,
        "acknowledge_descendant_promotion",
        lambda *args, **kwargs: calls.append("promote")
        or {"status": "promoted"},
    )
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda *args, **kwargs: calls.append("eligible") or True,
    )
    return calls


def test_real_enrollment_apply_materializes_and_replays_exact_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = install_fake_p3(monkeypatch)
    proposal, preview, resolved = real_enrollment_authority_inputs()
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    authority_sha = "4" * 64
    authority = {
        "candidate_proposal_sha256": "2" * 64,
        "enrollment_preview_sha256": "3" * 64,
        "enrollment_units": preview["enrollment_units"],
        "models": [
            {
                "candidate_id": adapter.candidate_id,
                "revision_sha": adapter.revision_sha,
            }
        ],
        "preparation_methods": [
            {
                "method_id": "no_enhancement",
                "development_comparison_receipt_sha256": (
                    verification.EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256
                ),
            }
        ],
        "real_biometric_enrollment_authorized": True,
        "will_read_audio": True,
        "will_materialize_embeddings": True,
        "will_run_trials": False,
        "will_read_calibration_or_evaluation": False,
    }
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_apply_authority",
        lambda *args, **kwargs: authority,
    )
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_candidate_proposal",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: resolved,
    )
    monkeypatch.setattr(
        verification,
        "_authorized_real_windows",
        lambda *args, **kwargs: [
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
    )
    root = tmp_path / "p4"
    application = apply_real_enrollment(
        authority_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
        adapters={adapter.candidate_id: adapter},
        test_mode=True,
    )

    assert application["status"] == "success"
    assert application["profile_count"] == 1
    assert application["did_read_audio"] is True
    assert application["did_materialize_embeddings"] is True
    assert application["did_register_p4_descendants"] is True
    assert application["did_run_trials"] is False
    assert application["contains_raw_biometric_values"] is False
    assert calls == ["register", "promote", "eligible"]
    assert replay_real_enrollment_application(
        application["application_sha256"],
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    ) == application
    for field, forged_value in (
        ("candidate_proposal_sha256", "a" * 64),
        ("enrollment_preview_sha256", "b" * 64),
        ("intended_split", "evaluation"),
        ("did_run_trials", True),
        ("did_read_calibration_or_evaluation", True),
        ("did_perform_external_write", True),
        ("contains_raw_biometric_values", True),
    ):
        forged = {
            key: value
            for key, value in application.items()
            if key not in {"application_sha256", "private_application_path"}
        }
        forged[field] = forged_value
        forged_sha = verification.canonical_artifact_hash(
            {key: value for key, value in forged.items() if key != "applied_at"}
        )
        forged_path = root / "enrollment-applications" / f"{forged_sha}.json"
        verification.write_immutable_private_json(
            forged_path, forged, volatile_fields=("applied_at",)
        )
        with pytest.raises(AcousticVerificationError, match="semantics"):
            replay_real_enrollment_application(
                forged_sha,
                runtime_root=root,
                p3_runtime_root=tmp_path / "p3",
            )
    repeated = apply_real_enrollment(
        authority_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
        adapters={adapter.candidate_id: adapter},
        test_mode=True,
    )
    assert repeated == application
    assert stat.S_IMODE(
        Path(application["private_application_path"]).stat().st_mode
    ) == 0o600


def test_real_enrollment_apply_rejects_unreviewed_adapter_override(
    tmp_path: Path,
) -> None:
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    with pytest.raises(AcousticVerificationError, match="deterministic tests"):
        apply_real_enrollment(
            "4" * 64,
            runtime_root=tmp_path / "p4",
            p3_runtime_root=tmp_path / "p3",
            adapters={adapter.candidate_id: adapter},
        )


def test_development_trials_are_exact_resubstitution_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal, _, _ = real_enrollment_authority_inputs()
    source = proposal["candidates"][0]["proposed_sources"][0]
    second_person = "subject-second-person-001"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    profiles = []
    for index, person_ref_id in enumerate(
        (proposal["candidates"][0]["person_ref_id"], second_person), start=1
    ):
        profiles.append(
            {
                "profile_id": f"verification-profile-{index:024d}",
                "descendant_id": f"verification-descendant-{index:024d}",
                "person_ref_id": person_ref_id,
                "p3_profile_id": f"reference-profile-{index:024d}",
                "generation_id": f"refgen-{index:024d}",
                "generation_sha256": str(index) * 64,
                "candidate_id": adapter.candidate_id,
                "model_revision": adapter.revision_sha,
                "artifact_sha256": chr(96 + index) * 64,
                "profile_manifest_sha256": chr(98 + index) * 64,
                "private_artifact_path": str(tmp_path / f"profile-{index}.f32le"),
                "window_count": 1,
                "session_count": 1,
                "dispersion": 0.0,
                "lifecycle_state": "active",
                "calibration_eligible": True,
                "replacement_profile_id": None,
                "created_at": "2026-07-31T12:00:00Z",
                "updated_at": "2026-07-31T12:00:00Z",
            }
        )
    application = {
        "status": "success",
        "intended_split": "development",
        "did_run_trials": False,
        "did_read_calibration_or_evaluation": False,
        "candidate_proposal_sha256": "2" * 64,
        "enrollment_preview_sha256": "3" * 64,
        "profiles": profiles,
    }
    source_binding = verification._development_trial_source_binding(source)
    authority_sha = "4" * 64
    authority = {
        "authority_sha256": authority_sha,
        "enrollment_application_sha256": "5" * 64,
        "candidate_proposal_sha256": "2" * 64,
        "enrollment_preview_sha256": "3" * 64,
        "development_comparison_receipt_sha256": (
            verification.EXPECTED_DEVELOPMENT_COMPARISON_RECEIPT_SHA256
        ),
        "source_units": [
            {
                "person_ref_id": proposal["candidates"][0]["person_ref_id"],
                "source": source_binding,
            }
        ],
        "preparation_methods": list(verification.METHOD_IDS),
        "profiles": profiles,
        "expected_coverage": {
            "logical_trials": 10,
            "genuine_trials": 5,
            "impostor_trials": 5,
            "unique_probe_waveforms": 5,
            "unique_waveform_model_profile_combinations": 10,
        },
    }
    monkeypatch.setattr(
        verification,
        "replay_development_trial_authority",
        lambda *args, **kwargs: authority,
    )
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_application",
        lambda *args, **kwargs: application,
    )
    monkeypatch.setattr(
        verification,
        "replay_real_enrollment_candidate_proposal",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        verification,
        "_authorized_real_windows",
        lambda sources, method_id: [
            {
                "session_id": source["session_id"],
                "samples": [
                    float(verification.METHOD_IDS.index(method_id) + 1) / 10.0,
                    -0.5,
                ]
                * 4_000,
            }
        ],
    )
    monkeypatch.setattr(
        verification,
        "replay_profile",
        lambda profile_id, **kwargs: {
            **next(profile for profile in profiles if profile["profile_id"] == profile_id),
            "private_bytes_present": True,
            "tombstone_path": None,
            "replayed_at": "2026-07-31T12:00:01Z",
        },
    )
    monkeypatch.setattr(verification, "descendant_is_eligible", lambda *args, **kwargs: True)
    score_calls: list[str] = []

    def fake_score(profile_id: str, *, probe_samples: list[float], **kwargs: object) -> dict:
        profile = next(profile for profile in profiles if profile["profile_id"] == profile_id)
        probe_sha = verification._window_hash(probe_samples)
        score = 0.75 if profile["person_ref_id"] == proposal["candidates"][0]["person_ref_id"] else -0.25
        identity = {
            "profile_id": profile_id,
            "descendant_id": profile["descendant_id"],
            "artifact_sha256": profile["artifact_sha256"],
            "candidate_id": profile["candidate_id"],
            "model_revision": profile["model_revision"],
            "probe_sha256": probe_sha,
            "score": score,
        }
        score_calls.append(profile_id)
        return {
            "trial_id": "verification-trial-"
            + verification.canonical_artifact_hash(identity)[:24],
            "probe_sha256": probe_sha,
            "score": score,
        }

    monkeypatch.setattr(verification, "score_profile", fake_score)
    root = tmp_path / "p4"
    result = apply_development_trials(
        authority_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
        adapters={adapter.candidate_id: adapter},
        test_mode=True,
    )

    assert result["denominators"] == {
        "attempted": 10, "success": 10, "failure": 0, "blocked": 0
    }
    assert result["evidence_class"] == "development_resubstitution_diagnostic"
    assert result["held_out"] is False
    assert result["enrollment_probe_overlap"] is True
    assert result["contains_biometric_scores"] is True
    assert result["did_select_threshold"] is False
    assert replay_development_trial_application(
        result["application_sha256"],
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    ) == result
    repeated = apply_development_trials(
        authority_sha,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
        adapters={adapter.candidate_id: adapter},
        test_mode=True,
    )
    assert repeated == result
    assert len(score_calls) == 10

    forged = {
        key: value
        for key, value in result.items()
        if key not in {"application_sha256", "private_application_path"}
    }
    forged["held_out"] = True
    forged_sha = verification.canonical_artifact_hash(
        {key: value for key, value in forged.items() if key != "applied_at"}
    )
    verification.write_immutable_private_json(
        root / "development-trial-applications" / f"{forged_sha}.json",
        forged,
        volatile_fields=("applied_at",),
    )
    with pytest.raises(AcousticVerificationError, match="semantics"):
        replay_development_trial_application(
            forged_sha, runtime_root=root, p3_runtime_root=tmp_path / "p3"
        )


def test_authorized_real_windows_validate_lineage_and_exact_pcm_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "p2"
    root.mkdir(mode=0o700)
    audio_path = root / "source.wav"
    values = [1_000, -1_000] * 16_000
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    audio_path.chmod(0o600)
    method = {
        "method_id": "no_enhancement",
        "status": "success",
        "output_path": str(audio_path),
        "output_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(
        json.dumps({"method_results": [method]}), encoding="utf-8"
    )
    comparison_path.chmod(0o600)
    lineage = {
        "run_id": "speech-prep-test-001",
        "method_id": "no_enhancement",
        "replay_receipt_sha256": "1" * 64,
        "comparison_path": str(comparison_path),
        "comparison_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "method_result_sha256": verification.canonical_artifact_hash(method),
        "source_blob_id": "source-001",
        "source_sha256": "2" * 64,
        "audio_quality_sha256": "3" * 64,
        "runtime_root": str(root),
    }
    monkeypatch.setattr(
        verification,
        "resolve_comparison_lineage_receipt",
        lambda *args, **kwargs: dict(lineage),
    )
    source = {
        "session_id": "session-001",
        "start_seconds": 0.5,
        "end_seconds": 1.5,
        "lineage": lineage,
    }

    windows = verification._authorized_real_windows(
        [source], method_id="no_enhancement"
    )
    assert len(windows) == 1
    assert windows[0]["session_id"] == "session-001"
    assert len(windows[0]["samples"]) == 16_000
    assert max(abs(value) for value in windows[0]["samples"]) < 1.0

    audio_path.write_bytes(audio_path.read_bytes() + b"drift")
    audio_path.chmod(0o600)
    with pytest.raises(AcousticVerificationError, match="PCM artifact drifted"):
        verification._authorized_real_windows(
            [source], method_id="no_enhancement"
        )


def test_synthetic_profile_stages_registers_promotes_and_scores_privately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = install_fake_p3(monkeypatch)
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    windows = [
        {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000},
        {"session_id": "session-002", "samples": [0.2, -0.2] * 8_000},
    ]

    profile = materialize_profile(
        "person-ref-001",
        adapter=adapter,
        windows=windows,
        preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )

    assert profile["lifecycle_state"] == "active"
    assert profile["window_count"] == 2
    assert profile["session_count"] == 2
    assert calls == ["resolve", "register", "promote", "eligible"]
    assert "embedding" not in json.dumps(profile).lower()
    blob = Path(profile["private_artifact_path"])
    assert stat.S_IMODE(blob.stat().st_mode) == 0o600

    trial = score_profile(
        profile["profile_id"],
        adapter=adapter,
        probe_samples=[0.25, -0.25] * 8_000,
        sample_rate=16_000,
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert trial["status"] == "success"
    assert trial["score"] > 0.99
    assert "embedding" not in json.dumps(trial).lower()
    assert calls[-2:] == ["eligible", "eligible"]


def test_staged_profile_resumes_after_parent_promotion_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    eligibility_checks = iter((False, True))
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda *args, **kwargs: next(eligibility_checks),
    )
    root = tmp_path / "profiles"
    kwargs = {
        "person_ref_id": "person-ref-001",
        "adapter": FakeVerificationAdapter(candidate_id="synthetic_verifier"),
        "windows": [
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        "preprocessing": {"method_id": "synthetic_raw", "revision": "v1"},
        "runtime_root": root,
        "p3_runtime_root": tmp_path / "p3",
    }
    with pytest.raises(
        AcousticVerificationError,
        match="P3 descendant promotion did not become eligible",
    ):
        materialize_profile(**kwargs)

    resumed = materialize_profile(**kwargs)
    assert resumed["lifecycle_state"] == "active"
    assert replay_profile(
        resumed["profile_id"], runtime_root=root
    )["lifecycle_state"] == "active"


def test_score_fails_if_p3_eligibility_changes_during_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    profile = materialize_profile(
        "person-ref-001",
        adapter=adapter,
        windows=[
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    eligibility = iter((True, False))
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda *args, **kwargs: next(eligibility),
    )

    with pytest.raises(AcousticVerificationError, match="eligibility changed"):
        score_profile(
            profile["profile_id"],
            adapter=adapter,
            probe_samples=[0.25, -0.25] * 8_000,
            sample_rate=16_000,
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )


def test_withdraw_then_delete_disables_scoring_and_removes_private_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    eligible = {"value": True}
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda *args, **kwargs: eligible["value"],
    )
    def request_invalidation(descendant_id, **kwargs):
        eligible["value"] = False
        return {
            "descendant_id": descendant_id,
            "artifact_sha256": "unused",
            "state": "invalidation_pending",
            "requested_at": "2026-07-31T23:10:00Z",
            "required_acknowledgment_token": "invalidation-token",
        }
    monkeypatch.setattr(
        verification,
        "request_descendant_invalidation",
        request_invalidation,
    )
    monkeypatch.setattr(
        verification,
        "acknowledge_descendant_invalidation",
        lambda *args, **kwargs: {"status": "invalidated"},
    )
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    profile = materialize_profile(
        "person-ref-001",
        adapter=adapter,
        windows=[
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    blob = Path(profile["private_artifact_path"])

    withdrawn = withdraw_profile(
        profile["profile_id"],
        reason="operator_withdrawal",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert withdrawn["lifecycle_state"] == "withdrawn"
    repeated_withdrawal = withdraw_profile(
        profile["profile_id"],
        reason="operator_withdrawal",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert repeated_withdrawal["lifecycle_state"] == "withdrawn"
    with pytest.raises(AcousticVerificationError, match="not active"):
        score_profile(
            profile["profile_id"],
            adapter=adapter,
            probe_samples=[0.25, -0.25] * 8_000,
            sample_rate=16_000,
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )

    deleted = delete_profile(
        profile["profile_id"],
        reason="retention_expired",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert deleted["lifecycle_state"] == "deleted"
    assert not blob.exists()
    tombstone = json.loads(Path(deleted["tombstone_path"]).read_text())
    assert tombstone["prior_artifact_sha256"] == profile["artifact_sha256"]
    assert "private_artifact_path" not in tombstone
    assert "embedding" not in json.dumps(tombstone).lower()
    replayed = replay_profile(profile["profile_id"], runtime_root=root)
    assert replayed["lifecycle_state"] == "deleted"
    assert replayed["private_bytes_present"] is False


def test_supersede_requires_active_same_person_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    eligibility: dict[str, bool] = {}
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda descendant_id, **kwargs: eligibility.get(descendant_id, True),
    )

    def request_invalidation(descendant_id, **kwargs):
        eligibility[descendant_id] = False
        return {
            "state": "invalidation_pending",
            "requested_at": "2026-07-31T23:10:00Z",
            "required_acknowledgment_token": "invalidation-token",
        }

    monkeypatch.setattr(
        verification, "request_descendant_invalidation", request_invalidation
    )
    monkeypatch.setattr(
        verification,
        "acknowledge_descendant_invalidation",
        lambda *args, **kwargs: {"status": "invalidated"},
    )
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")

    def create(revision: str, amplitude: float) -> dict:
        return materialize_profile(
            "person-ref-001",
            adapter=adapter,
            windows=[
                {
                    "session_id": f"session-{revision}",
                    "samples": [amplitude, -amplitude] * 8_000,
                }
            ],
            preprocessing={"method_id": "synthetic_raw", "revision": revision},
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )

    original = create("one", 0.25)
    replacement = create("two", 0.2)
    superseded = supersede_profile(
        original["profile_id"],
        replacement_profile_id=replacement["profile_id"],
        reason="model_profile_refresh",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )

    assert superseded["lifecycle_state"] == "superseded"
    assert superseded["replacement_profile_id"] == replacement["profile_id"]
    repeated = supersede_profile(
        original["profile_id"],
        replacement_profile_id=replacement["profile_id"],
        reason="model_profile_refresh",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert repeated["lifecycle_state"] == "superseded"
    assert replay_profile(
        replacement["profile_id"], runtime_root=root
    )["lifecycle_state"] == "active"


def test_parent_reference_supersession_can_precede_profile_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    eligibility: dict[str, bool] = {}
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda descendant_id, **kwargs: eligibility.get(descendant_id, True),
    )

    def request_invalidation(descendant_id, **kwargs):
        assert kwargs["reason"] == "reference_superseded"
        eligibility[descendant_id] = False
        return {
            "state": "invalidation_pending",
            "requested_at": "2026-08-02T12:00:00Z",
            "required_acknowledgment_token": "parent-invalidation-token",
        }

    monkeypatch.setattr(
        verification, "request_descendant_invalidation", request_invalidation
    )
    monkeypatch.setattr(
        verification,
        "acknowledge_descendant_invalidation",
        lambda *args, **kwargs: {"status": "invalidated"},
    )
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")

    original = materialize_profile(
        "person-ref-001",
        adapter=adapter,
        windows=[{"session_id": "session-one", "samples": [0.25, -0.25] * 8_000}],
        preprocessing={"method_id": "synthetic_raw", "revision": "one"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    acknowledged = acknowledge_parent_reference_supersession(
        original["profile_id"],
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert acknowledged["lifecycle_state"] == "active"
    repeated_acknowledgment = acknowledge_parent_reference_supersession(
        original["profile_id"],
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert repeated_acknowledgment["lifecycle_state"] == "active"

    replacement = materialize_profile(
        "person-ref-001",
        adapter=adapter,
        windows=[{"session_id": "session-two", "samples": [0.2, -0.2] * 8_000}],
        preprocessing={"method_id": "synthetic_raw", "revision": "two"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    superseded = supersede_profile(
        original["profile_id"],
        replacement_profile_id=replacement["profile_id"],
        reason="model_profile_refresh",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert superseded["lifecycle_state"] == "superseded"
    assert superseded["replacement_profile_id"] == replacement["profile_id"]


def test_profile_materialization_replays_and_lifecycle_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    kwargs = {
        "adapter": adapter,
        "windows": [
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        "preprocessing": {"method_id": "synthetic_raw", "revision": "v1"},
        "runtime_root": root,
        "p3_runtime_root": tmp_path / "p3",
    }
    first = materialize_profile("person-ref-001", **kwargs)
    replayed = materialize_profile("person-ref-001", **kwargs)
    assert replayed["profile_id"] == first["profile_id"]

    with sqlite3.connect(root / "profiles.sqlite3") as connection:
        connection.execute(
            "UPDATE profiles SET lifecycle_state = 'withdrawn' WHERE profile_id = ?",
            (first["profile_id"],),
        )
    with pytest.raises(AcousticVerificationError, match="lifecycle receipt binding"):
        replay_profile(first["profile_id"], runtime_root=root)


def test_withdraw_recovers_after_p3_acknowledgment_interruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    eligible = {"value": True}
    monkeypatch.setattr(
        verification,
        "descendant_is_eligible",
        lambda *args, **kwargs: eligible["value"],
    )

    def request(descendant_id, **kwargs):
        eligible["value"] = False
        return {
            "descendant_id": descendant_id,
            "state": "invalidation_pending",
            "requested_at": "2026-07-31T23:10:00Z",
            "required_acknowledgment_token": "invalidation-token",
        }

    acknowledgments = {"count": 0}

    def acknowledge(*args, **kwargs):
        acknowledgments["count"] += 1
        if acknowledgments["count"] == 1:
            raise OSError("synthetic interruption")
        return {"status": "invalidated"}

    monkeypatch.setattr(verification, "request_descendant_invalidation", request)
    monkeypatch.setattr(
        verification, "acknowledge_descendant_invalidation", acknowledge
    )
    root = tmp_path / "profiles"
    profile = materialize_profile(
        "person-ref-001",
        adapter=FakeVerificationAdapter(candidate_id="synthetic_verifier"),
        windows=[
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    with pytest.raises(OSError, match="synthetic interruption"):
        withdraw_profile(
            profile["profile_id"],
            reason="operator_withdrawal",
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )
    assert replay_profile(
        profile["profile_id"], runtime_root=root
    )["lifecycle_state"] == "withdrawn"
    recovered = withdraw_profile(
        profile["profile_id"],
        reason="operator_withdrawal",
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert recovered["lifecycle_state"] == "withdrawn"


def test_profile_materialization_rejects_private_metadata_and_adapter_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_p3(monkeypatch)
    root = tmp_path / "profiles"
    adapter = FakeVerificationAdapter(candidate_id="synthetic_verifier")
    with pytest.raises(AcousticVerificationError, match="window shape"):
        materialize_profile(
            "person-ref-001",
            adapter=adapter,
            windows=[
                {
                    "session_id": "session-001",
                    "samples": [0.25, -0.25] * 8_000,
                    "transcript_text": "must not persist",
                }
            ],
            preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )

    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: {
            **synthetic_reference(),
            "reference": {"sources": [{"device_class": "real"}]},
        },
    )
    with pytest.raises(AcousticVerificationError, match="synthetic fixture authority"):
        materialize_profile(
            "person-ref-001",
            adapter=adapter,
            windows=[
                {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
            ],
            preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )

    monkeypatch.setattr(
        verification,
        "resolve_eligible_reference",
        lambda *args, **kwargs: synthetic_reference(),
    )

    class FailingAdapter(FakeVerificationAdapter):
        def embed(self, samples, *, sample_rate):
            raise MemoryError("synthetic OOM")

    with pytest.raises(AcousticVerificationError, match="adapter failed"):
        materialize_profile(
            "person-ref-001",
            adapter=FailingAdapter(candidate_id="synthetic_failure"),
            windows=[
                {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
            ],
            preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
            runtime_root=root,
            p3_runtime_root=tmp_path / "p3",
        )


def test_unavailable_real_adapter_and_silence_fail_closed(tmp_path: Path) -> None:
    sine = [0.1] * 16_000
    adapter = adapter_registry(snapshot_root=tmp_path / "missing")[
        "wespeaker_campplus"
    ]
    with pytest.raises(AcousticVerificationError, match="manifest is unavailable"):
        adapter.embed(sine, sample_rate=16_000)
    with pytest.raises(AcousticVerificationError, match="zero norm"):
        FakeVerificationAdapter(candidate_id="synthetic_verifier").embed(
            [0.0] * 16_000, sample_rate=16_000
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("generation_sha256", "f" * 64),
        ("model_revision", "tampered-revision"),
        ("preprocessing_json", '{"method_id":"changed"}'),
        ("window_count", 99),
        ("dispersion", 0.99),
        ("artifact_path", "/tmp/redirected-private-vector"),
    ],
)
def test_profile_manifest_detects_metadata_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    install_fake_p3(monkeypatch)
    root = tmp_path / "profiles"
    profile = materialize_profile(
        "person-ref-001",
        adapter=FakeVerificationAdapter(candidate_id="synthetic_verifier"),
        windows=[
            {"session_id": "session-001", "samples": [0.25, -0.25] * 8_000}
        ],
        preprocessing={"method_id": "synthetic_raw", "revision": "v1"},
        runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    with sqlite3.connect(root / "profiles.sqlite3") as connection:
        connection.execute(
            f"UPDATE profiles SET {column} = ? WHERE profile_id = ?",
            (value, profile["profile_id"]),
        )
    with pytest.raises(AcousticVerificationError, match="manifest binding"):
        replay_profile(profile["profile_id"], runtime_root=root)


def _synthetic_calibration_trials() -> list[dict]:
    trials = []
    for index, (genuine_score, impostor_score) in enumerate(
        [(0.90, 0.10), (0.85, 0.15), (0.80, 0.20)]
    ):
        for profile_id, expected_match, score in (
            ("profile-genuine", True, genuine_score),
            ("profile-impostor", False, impostor_score),
        ):
            trials.append(
                {
                    "window_id": f"window-{index}",
                    "conversation_id": f"conversation-{index}",
                    "profile_id": profile_id,
                    "expected_match": expected_match,
                    "open_set_probe": False,
                    "score": score,
                    "conditions": {
                        "channel": "source_1_channel",
                        "device": "unassessed_until_p1",
                        "noise": "unassessed_until_p2",
                        "overlap": "overlap_regions_excluded",
                        "telephone_bandwidth": "unassessed_until_p1",
                        "usable_duration_band": "3_to_under_8_seconds",
                    },
                }
            )
    return trials


def test_calibration_threshold_freeze_is_deterministic_and_precommitted() -> None:
    policy = {
        "minimum_genuine_trials_per_unit": 3,
        "minimum_impostor_trials_per_unit": 3,
        "temperature_candidates": [0.01, 0.05],
        "condition_slices": [
            "channel", "device", "noise", "overlap",
            "telephone_bandwidth", "usable_duration_band",
        ],
    }
    first = verification._freeze_threshold_unit(
        "model-a", "no_enhancement", _synthetic_calibration_trials(), policy
    )
    second = verification._freeze_threshold_unit(
        "model-a", "no_enhancement", _synthetic_calibration_trials(), policy
    )
    assert first == second
    assert first["status"] == "success"
    assert first["metrics"]["balanced_error_rate"] == 0.0
    assert first["metrics"]["missing_denominator_status"] == "success"
    assert len(first["condition_slices"]) == 6
    assert first["candidate_margin"]["count"] == 3
    assert first["open_set_rejection"]["probe_count"] == 0


def test_calibration_threshold_freeze_fails_closed_on_missing_class() -> None:
    trials = [
        {**item, "expected_match": False}
        for item in _synthetic_calibration_trials()
    ]
    result = verification._freeze_threshold_unit(
        "model-a",
        "rnnoise",
        trials,
        {
            "minimum_genuine_trials_per_unit": 3,
            "minimum_impostor_trials_per_unit": 3,
            "temperature_candidates": [0.01],
            "condition_slices": [],
        },
    )
    assert result == {
        "candidate_id": "model-a",
        "method_id": "rnnoise",
        "status": "not_run",
        "reason_code": "insufficient_class_denominator",
        "threshold": None,
        "temperature": None,
        "metrics": None,
        "condition_slices": [],
        "candidate_margin": None,
        "open_set_rejection": None,
    }


def test_calibration_stage_hash_ignores_only_the_declared_timestamp() -> None:
    baseline = {"status": "success", "value": 3, "applied_at": "first"}
    replayed = {**baseline, "applied_at": "second"}
    forged = {**baseline, "value": 4}
    assert verification._calibration_stage_identity(
        baseline, "applied_at"
    ) == verification._calibration_stage_identity(replayed, "applied_at")
    assert verification._calibration_stage_identity(
        baseline, "applied_at"
    ) != verification._calibration_stage_identity(forged, "applied_at")


def _evaluation_authority_inputs() -> tuple[dict, dict, dict, dict]:
    methods = ["no_enhancement", "deepfilternet", "rnnoise"]
    models = [
        "speechbrain_ecapa_tdnn", "wespeaker_campplus", "wespeaker_resnet34"
    ]
    thresholds = [
        {
            "candidate_id": model,
            "method_id": method,
            "threshold": 0.5,
            "temperature": 0.05,
            "status": "success",
        }
        for model in models
        for method in methods
    ]
    profiles = [
        {
            "profile_id": f"profile-{model}-{person}",
            "person_ref_id": f"person-{person}",
            "candidate_id": model,
            "model_revision": str(index + 1) * 40,
            "profile_manifest_sha256": str(index + 2) * 64,
            "descendant_id": f"descendant-{model}-{person}",
        }
        for index, model in enumerate(models)
        for person in ("a", "b")
    ]
    calibration = {
        "status": "success",
        "intended_split": "calibration",
        "did_select_and_freeze_thresholds": True,
        "did_read_evaluation": False,
        "threshold_unit_count": 9,
        "permits_generalization_claim": False,
        "authority_sha256": "a" * 64,
        "score_matrix_sha256": "b" * 64,
        "thresholds": thresholds,
    }
    calibration_authority = {
        "development_application_sha256": "c" * 64,
        "development_authority_sha256": "d" * 64,
        "enrollment_application_sha256": "e" * 64,
        "profiles": profiles,
        "preparation_methods": list(verification.METHOD_IDS),
        "score_methods": methods,
        "preparation_contract": {
            "channel_policy": {
                "allowed_source_channels": [1, 2],
                "mono": "identity",
                "stereo": "arithmetic_average_0.5_left_plus_0.5_right",
                "output_channels": 1,
                "authority_binding": "this_calibration_authority_sha256",
                "no_silent_fallback": True,
            }
        },
        "window_policy": {
            "minimum_seconds": 0.75,
            "maximum_seconds": 15.0,
            "maximum_windows_per_speaker_per_conversation": 3,
        },
        "metric_policy": {
            "condition_slices": [
                "channel", "device", "noise", "overlap",
                "telephone_bandwidth", "usable_duration_band",
            ]
        },
    }
    split = {
        "split_access_policy_sha256": "f" * 64,
        "parent_corpus_manifest_sha256": "1" * 64,
        "evaluation_record_set_sha256": "2" * 64,
        "evaluation_conversation_set_sha256": "3" * 64,
        "evaluation_recording_count": 5,
        "evaluation_conversation_count": 5,
    }
    terminal = {
        "schema_version": "transcribe-audio.verification-terminal-decision-policy.v1",
        "precedence": ["stop", "reject", "select", "refine"],
        "minimum_evidence": {
            "same_person_trials": 20,
            "different_person_trials": 100,
            "open_set_trials": 20,
            "eligible_evaluation_recordings": 5,
            "all_declared_condition_slices_reported": True,
        },
    }
    return calibration, calibration_authority, split, terminal


def test_evaluation_authority_is_pre_reveal_idempotent_and_semantic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calibration, calibration_authority, split, terminal = (
        _evaluation_authority_inputs()
    )
    monkeypatch.setattr(
        verification, "replay_calibration_thresholds",
        lambda *args, **kwargs: calibration,
    )
    monkeypatch.setattr(
        verification, "replay_calibration_apply_authority",
        lambda *args, **kwargs: calibration_authority,
    )
    monkeypatch.setattr(
        verification, "_evaluation_split_metadata_authority",
        lambda *args, **kwargs: split,
    )
    monkeypatch.setattr(
        verification, "_terminal_decision_policy",
        lambda *args, **kwargs: terminal,
    )
    monkeypatch.setattr(
        verification,
        "_evaluation_records_after_authority",
        lambda *args, **kwargs: pytest.fail(
            "evaluation rows opened before reveal"
        ),
    )
    root = tmp_path / "runtime"
    first = verification.build_evaluation_apply_authority(
        "9" * 64, runtime_root=root, p3_runtime_root=tmp_path / "p3"
    )
    second = verification.build_evaluation_apply_authority(
        "9" * 64, runtime_root=root, p3_runtime_root=tmp_path / "p3"
    )
    replayed = verification.replay_evaluation_apply_authority(
        first["authority_sha256"], runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert first["authority_sha256"] == second["authority_sha256"]
    assert replayed["authority_sha256"] == first["authority_sha256"]
    assert Path(first["private_authority_path"]).stat().st_mode & 0o777 == 0o600
    assert first["preparation_contract"]["channel_policy"][
        "authority_binding"
    ] == "terminal_evaluation_authority_sha256"
    resolution = first["terminal_resolution_policy"]
    assert resolution[
        "any_terminal_policy_stop_if_condition_or_any_unit_stop"
    ] == "global_stop_before_candidate_reduction"
    assert resolution["runtime_cross_product_order"] == (
        "method_rank_then_model_rank"
    )
    assert all(item["margin"] == 0.0 for item in first["fixed_abstention_margins"])

    forged = {
        key: value
        for key, value in first.items()
        if key not in {"authority_sha256", "private_authority_path"}
    }
    forged["will_read_evaluation_gold"] = False
    forged_sha = verification.canonical_artifact_hash(forged)
    forged_path = root / "evaluation-authorities" / f"{forged_sha}.json"
    forged_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    forged_path.chmod(0o600)
    with pytest.raises(AcousticVerificationError, match="replay is invalid"):
        verification.replay_evaluation_apply_authority(
            forged_sha, runtime_root=root, p3_runtime_root=tmp_path / "p3"
        )


def test_evaluation_terminal_stop_is_idempotent_without_split_body_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority_sha = "8" * 64
    authority = {
        "preparation_contract": {
            "p2_module_sha256": verification.sha256_file(
                Path(verification.speech_preparation.__file__).resolve()
            )
        },
        "terminal_decision_policy_sha256": "7" * 64,
    }
    monkeypatch.setattr(
        verification,
        "replay_evaluation_apply_authority",
        lambda *args, **kwargs: authority,
    )
    root = tmp_path / "runtime"
    split_path = (
        root / "evaluation-stages" / authority_sha / "split-reveal.json"
    )
    verification.ensure_private_tree(root, split_path.parent)
    split_path.write_text("body must remain unopened\n", encoding="utf-8")
    split_path.chmod(0o600)
    original_reader = verification.read_private_object

    def guarded_reader(path: Path) -> dict:
        assert path.name != "split-reveal.json"
        return original_reader(path)

    monkeypatch.setattr(verification, "read_private_object", guarded_reader)
    first = verification.record_evaluation_terminal_stop(
        authority_sha, runtime_root=root, p3_runtime_root=tmp_path / "p3"
    )
    second = verification.record_evaluation_terminal_stop(
        authority_sha, runtime_root=root, p3_runtime_root=tmp_path / "p3"
    )
    replayed = verification.replay_evaluation_terminal_stop(
        first["application_sha256"], runtime_root=root,
        p3_runtime_root=tmp_path / "p3",
    )
    assert first["application_sha256"] == second["application_sha256"]
    assert replayed["terminal_decision"] == "stop"
    assert replayed["logical_trial_count"] == 0
    assert replayed["replay_mode"] == (
        "metadata_only_without_evaluation_or_split_body_read"
    )
