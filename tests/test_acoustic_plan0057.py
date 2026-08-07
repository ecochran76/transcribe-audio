from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acoustic_plan0057 as plan


def cohort() -> list[dict]:
    return [
        {
            "ordinal": ordinal,
            "document_id": f"document-{ordinal}",
            "conversation_key": f"conversation-{ordinal}",
            "transcript_path": f"/private/transcript-{ordinal}.json",
            "transcript_sha256": f"{ordinal}" * 64,
            "source_media_path": f"/private/source-{ordinal}.m4a",
            "source_media_sha256": f"{ordinal + 3}" * 64,
            "recording_start": f"2026-08-0{5 if ordinal == 1 else 6}T{14 + ordinal:02d}:00:00-05:00",
            "context_id": f"context-{min(ordinal, 2)}",
            "duration_seconds": 120.0 + ordinal,
            "probe": {
                "codec_name": "aac",
                "duration_seconds": 120.0 + ordinal,
                "sample_rate": 48000,
                "channels": 1,
            },
        }
        for ordinal in range(1, 4)
    ]


def profiles() -> tuple[list[dict], dict]:
    rows = []
    for candidate in plan.CANDIDATE_IDS:
        for subject in sorted(plan.ALLOWLISTED_SUBJECT_IDS):
            rows.append(
                {
                    "candidate_id": candidate,
                    "person_ref_id": subject,
                    "profile_id": f"{candidate}-{subject}",
                }
            )
    return rows, {
        "profile_count": 6,
        "subject_count": 2,
        "candidate_count": 3,
        "profile_set_sha256": "a" * 64,
    }


def thresholds() -> list[dict]:
    return [
        {
            "candidate_id": candidate,
            "method_id": method,
            "threshold": 0.5,
            "temperature": 1.0,
        }
        for candidate in plan.CANDIDATE_IDS
        for method in plan.METHOD_IDS
    ]


def preview() -> dict:
    return plan.preview_authority(
        cohort=cohort(),
        prior_hashes={"f" * 64},
        prior_json_hashes=["e" * 64],
        profile_inventory=profiles(),
        identity_state_snapshot={"snapshot_sha256": "b" * 64},
        repository_authority={
            "commit": "c" * 40,
            "module_sha256": "d" * 64,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
        local_runtime={
            "runtime_sha256": "e" * 64,
            "network_required": False,
            "diarization_model_local": True,
            "transcription_model_local": True,
            "compute_device": "cuda",
        },
        threshold_units=thresholds(),
    )


def test_preview_freezes_exact_fresh_batch_and_negative_actions() -> None:
    authority = preview()

    assert authority["source_count"] == 3
    assert authority["context_count"] == 2
    assert authority["allowlisted_subject_ids"] == sorted(
        plan.ALLOWLISTED_SUBJECT_IDS
    )
    assert authority["threshold_unit_count"] == 9
    assert authority["identity_state_before"]["snapshot_sha256"] == "b" * 64
    assert authority["action_vector"]["run_local_models"] is False
    assert authority["action_vector"]["apply_speaker_assignments"] is False
    assert authority["contains_display_names"] is False


@pytest.mark.parametrize(
    "mutation",
    ["overlap", "old", "one_context", "wrong_subjects", "network"],
)
def test_preview_rejects_unfresh_unsafe_or_incomplete_authority(
    mutation: str,
) -> None:
    current_cohort = cohort()
    current_profiles = profiles()
    prior_hashes = {"f" * 64}
    runtime = {
        "runtime_sha256": "e" * 64,
        "network_required": False,
        "diarization_model_local": True,
        "transcription_model_local": True,
        "compute_device": "cuda",
    }
    if mutation == "overlap":
        prior_hashes.add(current_cohort[0]["source_media_sha256"])
    elif mutation == "old":
        current_cohort[0]["recording_start"] = plan.PLAN0056_SOURCE_START
    elif mutation == "one_context":
        for item in current_cohort:
            item["context_id"] = "one-context"
    elif mutation == "wrong_subjects":
        current_profiles[0][0]["person_ref_id"] = "subject-not-enrolled"
    elif mutation == "network":
        runtime["network_required"] = True

    with pytest.raises(plan.Plan0057Error):
        plan.preview_authority(
            cohort=current_cohort,
            prior_hashes=prior_hashes,
            prior_json_hashes=["e" * 64],
            profile_inventory=current_profiles,
            identity_state_snapshot={"snapshot_sha256": "b" * 64},
            repository_authority={
                "commit": "c" * 40,
                "module_sha256": "d" * 64,
                "clean": True,
                "upstream_ahead": 0,
                "upstream_behind": 0,
            },
            local_runtime=runtime,
            threshold_units=thresholds(),
        )


def test_freeze_and_replay_are_private_and_source_bound(tmp_path: Path) -> None:
    authority = preview()
    runtime_root = tmp_path / "runtime"

    receipt = plan.freeze_authority(
        authority,
        expected_content_sha256=authority["content_sha256"],
        runtime_root=runtime_root,
        verify_live_files=False,
    )
    replay = plan.replay_authority(
        authority["content_sha256"],
        runtime_root=runtime_root,
        verify_live_files=False,
    )

    assert receipt["status"] == "frozen_pre_model_authority"
    assert replay["idempotent_replay"] is True
    assert Path(receipt["manifest_path"]).stat().st_mode & 0o777 == 0o600


def test_execution_authority_enables_only_local_shadow_actions(
    tmp_path: Path,
) -> None:
    p0 = preview()
    execution = plan.preview_execution_authority(
        p0_authority=p0,
        repository_authority=p0["repository_authority"],
    )

    assert execution["action_vector"]["run_local_models"] is True
    assert execution["action_vector"]["publish_read_only_evidence"] is True
    assert execution["action_vector"]["read_human_gold"] is False
    assert execution["action_vector"]["apply_speaker_assignments"] is False
    assert execution["action_vector"]["create_or_mutate_identities"] is False

    receipt = plan.freeze_execution_authority(
        execution,
        expected_content_sha256=execution["content_sha256"],
        runtime_root=tmp_path / "runtime",
    )
    replay = plan.replay_execution_authority(
        execution["content_sha256"],
        runtime_root=tmp_path / "runtime",
    )
    assert receipt["status"] == "frozen_before_local_execution"
    assert replay["idempotent_replay"] is True


def test_build_execution_manifest_requires_complete_source_and_speaker_yield() -> None:
    authority = preview()
    results = []
    for source in authority["private_evidence"]["cohort"]:
        results.append(
            {
                "document_id": source["document_id"],
                "conversation_key": source["conversation_key"],
                "source_media_sha256": source["source_media_sha256"],
                "entered": True,
                "eligible_speaker_count": 2,
                "covered_speaker_count": 2,
                "stop_reason": None,
                "proposals": [
                    {
                        "speaker_ref": "SPEAKER_1",
                        "disposition": "review",
                        "subject_id": "subject-df34bc192c07bd86566fff12",
                        "confidence_band": "low",
                        "supporting_unit_count": 1,
                        "supporting_candidate_family_count": 1,
                        "opposing_unit_count": 0,
                        "rationale": "Frozen consensus evidence.",
                    },
                    {
                        "speaker_ref": "SPEAKER_2",
                        "disposition": "abstain",
                        "subject_id": None,
                        "confidence_band": "none",
                        "supporting_unit_count": 0,
                        "supporting_candidate_family_count": 0,
                        "opposing_unit_count": 0,
                        "rationale": "No threshold support.",
                    },
                ],
                "artifact_hashes": {"matrices": "a" * 64},
            }
        )
    manifest = plan.build_execution_manifest(
        authority=authority,
        source_results=results,
        identity_state_before={"snapshot_sha256": "b" * 64},
        identity_state_after={"snapshot_sha256": "b" * 64},
    )

    assert manifest["eligible_recording_count"] == 3
    assert manifest["entered_recording_count"] == 3
    assert manifest["eligible_speaker_count"] == 6
    assert manifest["covered_speaker_count"] == 6
    assert manifest["stop_reasons"] == []
    assert manifest["applied_assignments"] is False

    results[0]["covered_speaker_count"] = 1
    with pytest.raises(plan.Plan0057Error):
        plan.build_execution_manifest(
            authority=authority,
            source_results=results,
            identity_state_before={"snapshot_sha256": "b" * 64},
            identity_state_after={"snapshot_sha256": "b" * 64},
        )
