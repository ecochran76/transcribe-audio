from __future__ import annotations

from pathlib import Path

import acoustic_plan0056_audit as audit


CHRIS = "subject-7c24e8f41409c6f517291fe7"
ERIC = "subject-df34bc192c07bd86566fff12"


def execution_manifest() -> dict:
    return {
        "identity_state_unchanged": True,
        "read_pilot_outcome_gold": False,
        "applied_assignments": False,
        "artifacts": {
            "proposals": {
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


def review_preview() -> dict:
    return {
        "review_complete": True,
        "decision_count": 2,
        "decisions": [
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
        ],
    }


def test_independent_audit_recomputes_complete_pilot_denominators() -> None:
    result = audit.recompute_plan0056_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(),
    )

    assert result["metrics"] == {
        "speaker_count": 2,
        "enrolled_speaker_count": 1,
        "proposal_count": 2,
        "proposal_confirmed_count": 1,
        "proposal_rejected_count": 1,
        "assign_disposition_count": 1,
        "correct_assignment_count": 1,
        "wrong_assignment_count": 0,
        "high_confidence_wrong_count": 0,
        "review_count": 1,
        "abstention_count": 0,
        "enrolled_correct_assignment_count": 1,
        "enrolled_recall": 1.0,
        "proposal_precision": 0.5,
        "identity_creation_count": 0,
        "profile_or_reference_mutation_count": 0,
    }
    assert result["terminal_decision"] == "plan_next_bounded_integration_milestone"
    assert result["independent_guard_recomputed"] is True


def test_independent_audit_refines_after_wrong_assignment() -> None:
    human_review = review_preview()
    human_review["decisions"][1] = {
        **human_review["decisions"][1],
        "actual_identity": CHRIS,
        "proposal_decision": "reject",
    }

    result = audit.recompute_plan0056_audit(
        execution_manifest=execution_manifest(),
        review_preview=human_review,
    )

    assert result["metrics"]["wrong_assignment_count"] == 1
    assert result["terminal_decision"] == "refine"


def test_independent_terminal_audit_freezes_and_replays(tmp_path: Path) -> None:
    preview = audit.preview_plan0056_terminal_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(),
        repository_authority={
            "commit": "a" * 40,
            "clean": True,
            "upstream_ahead": 0,
            "upstream_behind": 0,
        },
    )

    receipt = audit.freeze_plan0056_terminal_audit(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "audit",
    )
    replay = audit.replay_plan0056_terminal_audit(
        preview["content_sha256"], runtime_root=tmp_path / "audit"
    )

    assert receipt["terminal_decision"] == "plan_next_bounded_integration_milestone"
    assert receipt["applied_assignments"] is False
    assert replay["idempotent_replay"] is True
