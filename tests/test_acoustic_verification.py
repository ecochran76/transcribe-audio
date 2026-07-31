from __future__ import annotations

import json
import hashlib
import sqlite3
import stat
from pathlib import Path

import pytest
import acoustic_verification as verification

from acoustic_verification import (
    AcousticVerificationError,
    FakeVerificationAdapter,
    adapter_registry,
    build_real_enrollment_candidate_proposal,
    build_real_enrollment_preview,
    cosine_score,
    dry_run_model_acquisition,
    replay_model_acquisition,
    replay_real_enrollment_candidate_proposal,
    replay_real_enrollment_preview,
    materialize_profile,
    delete_profile,
    replay_profile,
    score_profile,
    supersede_profile,
    withdraw_profile,
)


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
    parent.write_text(json.dumps({"recordings": records}), encoding="utf-8")
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
