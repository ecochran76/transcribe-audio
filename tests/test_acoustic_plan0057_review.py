from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import acoustic_plan0057 as execution
import acoustic_plan0057_review as review


def execution_manifest() -> dict:
    source_results = []
    speaker_counts = (3, 6, 6)
    proposal_ordinal = 0
    for source_ordinal, speaker_count in enumerate(speaker_counts, start=1):
        proposals = []
        for speaker_ordinal in range(1, speaker_count + 1):
            proposal_ordinal += 1
            assign = proposal_ordinal in {1, 4}
            supporting_units = [
                [candidate, method]
                for candidate in execution.CANDIDATE_IDS
                for method in execution.METHOD_IDS
            ][:7] if assign else []
            proposals.append(
                {
                    "speaker_ref": f"SPEAKER_{speaker_ordinal}",
                    "disposition": "assign" if assign else "abstain",
                    "subject_id": review.ERIC_SUBJECT_ID if assign else None,
                    "confidence_band": "medium" if assign else "none",
                    "supporting_unit_count": 7 if assign else 0,
                    "supporting_candidate_family_count": 3 if assign else 0,
                    "opposing_unit_count": 0,
                    "supporting_units": supporting_units,
                    "opposing_units": [],
                    "rationale": "Frozen consensus evidence.",
                }
            )
        source_results.append(
            {
                "document_id": f"document-{source_ordinal}",
                "conversation_key": f"conversation-{source_ordinal}",
                "source_media_sha256": str(source_ordinal) * 64,
                "entered": True,
                "eligible_speaker_count": speaker_count,
                "covered_speaker_count": speaker_count,
                "stop_reason": None,
                "proposals": proposals,
                "review_rows": [],
                "artifact_hashes": {},
            }
        )
    return {
        "schema_version": execution.EXECUTION_SCHEMA,
        "status": "complete_pending_human_review",
        "execution_authority_content_sha256": review.EXECUTION_AUTHORITY_SHA256,
        "content_sha256": review.EXECUTION_CONTENT_SHA256,
        "source_results": source_results,
        "eligible_recording_count": 3,
        "entered_recording_count": 3,
        "eligible_speaker_count": 15,
        "covered_speaker_count": 15,
        "stop_reasons": [],
        "identity_state_before": {"snapshot_sha256": "a" * 64},
        "identity_state_after": {"snapshot_sha256": "a" * 64},
        "identity_state_unchanged": True,
        "read_human_gold": False,
        "applied_assignments": False,
        "created_or_mutated_identities": False,
        "mutated_profiles_or_references": False,
        "wrote_external_provider": False,
        "enabled_default_integration": False,
        "ran_historical_reprocessing": False,
    }


def complete_answers(*, second_identity: str | None = None) -> str:
    lines = []
    ordinal = 0
    for source_ordinal, speaker_count in enumerate((3, 6, 6), start=1):
        for speaker_ordinal in range(1, speaker_count + 1):
            ordinal += 1
            value = (
                review.ERIC_SUBJECT_ID
                if ordinal in {1, 4}
                else "neither_enrolled"
            )
            if ordinal == 4 and second_identity is not None:
                value = second_identity
            lines.append(
                f"document-{source_ordinal}::SPEAKER_{speaker_ordinal} = {value}"
            )
    return "\n".join(lines)


def repository_authority() -> dict:
    return {
        "commit": "b" * 40,
        "module_sha256": "c" * 64,
        "clean": True,
        "upstream_ahead": 0,
        "upstream_behind": 0,
    }


def test_review_preview_requires_and_binds_all_fifteen_decisions() -> None:
    preview = review.preview_plan0057_review(
        complete_answers(),
        execution_manifest=execution_manifest(),
        repository_authority=repository_authority(),
    )

    assert preview["recording_count"] == 3
    assert preview["speaker_count"] == 15
    assert preview["decision_count"] == 15
    assert preview["review_complete"] is True
    assert sum(item["proposal_decision"] == "confirm" for item in preview["decisions"]) == 2
    assert sum(
        item["proposal_decision"] == "confirm_abstention"
        for item in preview["decisions"]
    ) == 13
    assert preview["action_vector"]["apply_speaker_assignments"] is False
    assert preview["action_vector"]["create_or_mutate_identities"] is False


@pytest.mark.parametrize(
    "answer_text",
    [
        "\n".join(complete_answers().splitlines()[:-1]),
        complete_answers().replace("document-1::SPEAKER_1", "unknown-card", 1),
        complete_answers().replace(review.ERIC_SUBJECT_ID, "Chief Eric", 1),
    ],
)
def test_review_parser_rejects_partial_unknown_or_inexact_answers(
    answer_text: str,
) -> None:
    with pytest.raises(review.Plan0057ReviewError):
        review.preview_plan0057_review(
            answer_text,
            execution_manifest=execution_manifest(),
            repository_authority=repository_authority(),
        )


def test_review_accepts_private_non_enrolled_display_label_only(tmp_path: Path) -> None:
    answer_text = complete_answers().replace(
        "document-1::SPEAKER_2 = neither_enrolled",
        "document-1::SPEAKER_2 = Neither enrolled person (Private label)",
    )
    preview = review.preview_plan0057_review(
        answer_text,
        execution_manifest=execution_manifest(),
        repository_authority=repository_authority(),
    )
    labeled = next(
        item for item in preview["decisions"] if item["review_display_label"]
    )
    assert labeled["actual_identity"] == "neither_enrolled"
    assert labeled["review_display_label"] == "Private label"
    receipt = review.freeze_plan0057_review(
        preview,
        expected_content_sha256=preview["content_sha256"],
        runtime_root=tmp_path / "review",
    )
    replay = review.replay_plan0057_review(
        preview["content_sha256"],
        runtime_root=tmp_path / "review",
    )

    assert receipt["decision_count"] == 15
    assert receipt["applied_assignments"] is False
    assert replay["idempotent_replay"] is True
    assert Path(tmp_path / "review").stat().st_mode & 0o777 == 0o700


def test_review_rejects_execution_with_mutation_or_incomplete_denominator() -> None:
    manifest = execution_manifest()
    manifest["applied_assignments"] = True
    with pytest.raises(review.Plan0057ReviewError):
        review.preview_plan0057_review(
            complete_answers(),
            execution_manifest=manifest,
            repository_authority=repository_authority(),
        )

    manifest = execution_manifest()
    manifest["source_results"][0]["proposals"][0]["supporting_unit_count"] = 8
    with pytest.raises(review.Plan0057ReviewError):
        review.preview_plan0057_review(
            complete_answers(),
            execution_manifest=manifest,
            repository_authority=repository_authority(),
        )


def test_review_freeze_rejects_rehashed_mutation_vector(tmp_path: Path) -> None:
    preview = review.preview_plan0057_review(
        complete_answers(),
        execution_manifest=execution_manifest(),
        repository_authority=repository_authority(),
    )
    preview["action_vector"]["apply_speaker_assignments"] = True
    preview["content_sha256"] = review.canonical_hash(
        {key: value for key, value in preview.items() if key != "content_sha256"}
    )

    with pytest.raises(review.Plan0057ReviewError):
        review.freeze_plan0057_review(
            preview,
            expected_content_sha256=preview["content_sha256"],
            runtime_root=tmp_path / "review",
        )
