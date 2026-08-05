from __future__ import annotations

import pytest
from pathlib import Path

import acoustic_plan0056_review as review


CHRIS = "subject-7c24e8f41409c6f517291fe7"
ERIC = "subject-df34bc192c07bd86566fff12"


def execution_manifest() -> dict:
    return {
        "authority_content_sha256": review.EXECUTION_AUTHORITY_SHA256,
        "identity_state_unchanged": True,
        "read_pilot_outcome_gold": False,
        "applied_assignments": False,
        "artifacts": {
            "proposals": {
                "content_sha256": review.PROPOSAL_CONTENT_SHA256,
                "allowlisted_subject_ids": [CHRIS, ERIC],
                "proposals": [
                    {
                        "speaker_ref": "SPEAKER_1",
                        "disposition": "review",
                        "subject_id": CHRIS,
                        "confidence_band": "low",
                    },
                    {
                        "speaker_ref": "SPEAKER_2",
                        "disposition": "assign",
                        "subject_id": ERIC,
                        "confidence_band": "medium",
                    },
                ],
            }
        },
    }


def test_review_preview_resolves_display_names_without_creating_identities() -> None:
    preview = review.preview_plan0056_review(
        "Speaker 1 = Neither enrolled person\nSpeaker 2 = Eric Cochran",
        execution_manifest=execution_manifest(),
        repository_authority={
            "commit": "a" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
    )

    assert preview["decisions"] == [
        {
            "speaker_ref": "SPEAKER_1",
            "actual_identity": "neither_enrolled",
            "proposal_decision": "reject",
            "proposed_subject_id": CHRIS,
        },
        {
            "speaker_ref": "SPEAKER_2",
            "actual_identity": ERIC,
            "proposal_decision": "confirm",
            "proposed_subject_id": ERIC,
        },
    ]
    assert preview["action_vector"]["apply_speaker_assignments"] is False
    assert preview["action_vector"]["create_or_mutate_identities"] is False
    assert preview["review_complete"] is True


def test_review_preview_rejects_unanswered_or_forking_identity_text() -> None:
    for value in ("UNANSWERED", "Eric Cochran, PhD"):
        with pytest.raises(review.Plan0056ReviewError):
            review.preview_plan0056_review(
                f"Speaker 1 = {value}\nSpeaker 2 = Eric Cochran",
                execution_manifest=execution_manifest(),
                repository_authority={
                    "commit": "a" * 40,
                    "clean": True,
                    "upstream_ahead": 0,
                    "upstream_behind": 0,
                },
            )


def test_complete_review_freezes_and_replays_without_applying_assignments(
    tmp_path: Path,
) -> None:
    preview = review.preview_plan0056_review(
        "Speaker 1 = UNKNOWN\nSpeaker 2 = Eric Cochran",
        execution_manifest=execution_manifest(),
        repository_authority={
            "commit": "a" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
    )

    receipt = review.freeze_plan0056_review(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "review",
    )
    replay = review.replay_plan0056_review(
        preview["content_sha256"], runtime_root=tmp_path / "review"
    )

    assert receipt["decision_count"] == 2
    assert receipt["applied_assignments"] is False
    assert replay["idempotent_replay"] is True
    assert ((tmp_path / "review").stat().st_mode & 0o777) == 0o700
