from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acoustic_plan0057_audit as audit
import acoustic_plan0057_review as review
from tests.test_acoustic_plan0057_review import (
    complete_answers,
    execution_manifest,
    repository_authority,
)


def review_preview(*, second_identity: str | None = None) -> dict:
    return review.preview_plan0057_review(
        complete_answers(second_identity=second_identity),
        execution_manifest=execution_manifest(),
        repository_authority=repository_authority(),
    )


def test_independent_audit_recomputes_complete_perfect_denominator() -> None:
    result = audit.recompute_plan0057_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(),
        current_identity_state={"snapshot_sha256": "a" * 64},
    )
    metrics = result["metrics"]

    assert metrics["eligible_recording_count"] == 3
    assert metrics["entered_recording_count"] == 3
    assert metrics["eligible_speaker_count"] == 15
    assert metrics["covered_speaker_count"] == 15
    assert metrics["human_review_decision_count"] == 15
    assert metrics["proposal_count"] == 2
    assert metrics["proposal_confirmed_count"] == 2
    assert metrics["abstention_count"] == 13
    assert metrics["correct_abstention_count"] == 13
    assert metrics["wrong_proposal_disposition_count"] == 0
    assert metrics["high_confidence_wrong_count"] == 0
    assert metrics["enrolled_recall"] == 1.0
    assert metrics["proposal_precision"] == 1.0
    assert metrics["review_burden"] == 1.0
    assert result["terminal_decision"] == "plan_next_bounded_milestone"


def test_independent_audit_refines_on_medium_wrong_or_unknown() -> None:
    wrong = audit.recompute_plan0057_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(second_identity=review.CHRIS_SUBJECT_ID),
        current_identity_state={"snapshot_sha256": "a" * 64},
    )
    unresolved = audit.recompute_plan0057_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(second_identity="unknown"),
        current_identity_state={"snapshot_sha256": "a" * 64},
    )

    assert wrong["metrics"]["wrong_proposal_disposition_count"] == 1
    assert wrong["metrics"]["enrolled_recall"] == 0.5
    assert wrong["terminal_decision"] == "refine"
    assert unresolved["metrics"]["unknown_identity_count"] == 1
    assert unresolved["terminal_decision"] == "refine"


def test_independent_audit_stops_on_high_confidence_wrong() -> None:
    manifest = execution_manifest()
    proposal = manifest["source_results"][1]["proposals"][0]
    proposal["confidence_band"] = "high"
    proposal["supporting_units"] = [
        [candidate, method]
        for candidate in audit.execution.CANDIDATE_IDS
        for method in audit.execution.METHOD_IDS
    ]
    proposal["supporting_unit_count"] = 9
    preview = review.preview_plan0057_review(
        complete_answers(second_identity=review.CHRIS_SUBJECT_ID),
        execution_manifest=manifest,
        repository_authority=repository_authority(),
    )

    result = audit.recompute_plan0057_audit(
        execution_manifest=manifest,
        review_preview=preview,
        current_identity_state={"snapshot_sha256": "a" * 64},
    )

    assert result["metrics"]["high_confidence_wrong_count"] == 1
    assert result["terminal_decision"] == "stop"


def test_independent_audit_rejects_identity_state_drift() -> None:
    with pytest.raises(audit.Plan0057AuditError):
        audit.recompute_plan0057_audit(
            execution_manifest=execution_manifest(),
            review_preview=review_preview(),
            current_identity_state={"snapshot_sha256": "d" * 64},
        )


def test_terminal_audit_freeze_and_replay_are_private(tmp_path: Path) -> None:
    preview = audit.preview_plan0057_terminal_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(),
        current_identity_state={"snapshot_sha256": "a" * 64},
        repository_authority=repository_authority(),
    )
    receipt = audit.freeze_plan0057_terminal_audit(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "audit",
    )
    replay = audit.replay_plan0057_terminal_audit(
        preview["content_sha256"],
        runtime_root=tmp_path / "audit",
    )

    assert receipt["terminal_decision"] == "plan_next_bounded_milestone"
    assert receipt["applied_assignments"] is False
    assert replay["idempotent_replay"] is True
    assert Path(tmp_path / "audit").stat().st_mode & 0o777 == 0o700


def test_terminal_audit_freeze_rejects_rehashed_mutation_vector(
    tmp_path: Path,
) -> None:
    preview = audit.preview_plan0057_terminal_audit(
        execution_manifest=execution_manifest(),
        review_preview=review_preview(),
        current_identity_state={"snapshot_sha256": "a" * 64},
        repository_authority=repository_authority(),
    )
    preview["action_vector"]["apply_speaker_assignments"] = True
    preview["content_sha256"] = audit.canonical_hash(
        {key: value for key, value in preview.items() if key != "content_sha256"}
    )

    with pytest.raises(audit.Plan0057AuditError):
        audit.freeze_plan0057_terminal_audit(
            preview,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "audit",
        )
