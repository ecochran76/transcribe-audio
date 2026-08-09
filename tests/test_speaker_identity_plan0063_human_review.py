from __future__ import annotations

import re

import pytest

import speaker_identity_plan0063_human_review as review


def _inputs():
    people = [
        {
            "proposed_person_id": f"provisional-person-{index:024x}",
            "member_names": [f"Person {index}"],
            "member_slot_ids": [f"doc{index}::SPEAKER_1"],
        }
        for index in range(1, 7)
    ]
    people[4]["member_slot_ids"] = [
        "47ea79857aa1ac2d1d79::SPEAKER_2",
        "47ea79857aa1ac2d1d79::SPEAKER_3",
    ]
    merges = []
    for index in range(3):
        merges.append(
            {
                "merge_proposal_id": f"person-merge-{index + 1:024x}",
                "proposed_person_id": people[index]["proposed_person_id"],
                "basis": "name_only",
                "member_slot_ids": [f"doc{index}::SPEAKER_1", f"doc{index}::SPEAKER_2"],
                "decision": "pending",
            }
        )
    reconciled = {
        "content_sha256": review.RECONCILIATION_SHA256,
        "status": "pending_human_grouping_and_binding_review",
        "negative_actions": review.NEGATIVE_ACTIONS,
        "person_proposals": people,
        "merge_proposals": merges,
        "voice_binding_proposals": [
            {
                "binding_proposal_id": f"voice-person-binding-{1:024x}",
                "proposed_person_id": people[0]["proposed_person_id"],
                "acoustic_subject_id": "subject-example",
                "slot_id": "doc0::SPEAKER_1",
                "decision": "pending",
            }
        ],
    }
    proposals = []
    clip_hashes = {}
    reference_number = 1
    for person_index in range(5):
        count = 6 if person_index else 2
        windows = []
        for _ in range(count):
            reference_id = f"review-window-{reference_number:024x}"
            clip_hashes[reference_id] = f"{reference_number:064x}"
            windows.append(
                {
                    "reference_id": reference_id,
                    "slot_id": f"doc{person_index}::SPEAKER_1",
                    "speaker_label_id": "SPEAKER_1",
                    "start_seconds": float(reference_number),
                    "end_seconds": float(reference_number + 5),
                    "source_sha256": f"{person_index + 1:064x}",
                    "future_holdout_excluded": True,
                    "data_split": "development_training_candidate",
                }
            )
            reference_number += 1
        proposals.append(
            {
                "proposed_person_id": people[person_index]["proposed_person_id"],
                "member_slot_ids": people[person_index]["member_slot_ids"],
                "device_metadata_status": "unverified",
                "status": "source_feasible_pending_human_review",
                "enrollment_authorized": False,
                "source_windows": windows,
            }
        )
    feasibility = {
        "content_sha256": review.FEASIBILITY_SHA256,
        "status": "source_feasibility_ready_pending_human_review",
        "reconciliation_content_sha256": review.RECONCILIATION_SHA256,
        "negative_actions": review.NEGATIVE_ACTIONS,
        "person_source_proposals": proposals,
    }
    return reconciled, feasibility, clip_hashes


def _manifest():
    reconciled, feasibility, clip_hashes = _inputs()
    return review.build_review_manifest(
        reconciled,
        feasibility,
        clip_sha256_by_reference=clip_hashes,
        repository_authority={"commit": "example"},
    )


def test_review_manifest_preserves_exact_blank_denominator_and_calendar_gap():
    manifest = _manifest()

    assert manifest["decision_count"] == 30
    assert len(manifest["merge_reviews"]) == 3
    assert len(manifest["binding_reviews"]) == 1
    assert sum(len(item["windows"]) for item in manifest["source_reviews"]) == 26
    assert manifest["calendar_candidate_observation"]["candidate_only"] is True
    assert manifest["calendar_candidate_observation"]["speaker_assignment_proven"] is False
    assert all(item["selected"] is None for item in manifest["merge_reviews"])
    assert not any(manifest["negative_actions"].values())


def test_review_html_has_direct_audio_working_export_controls_and_no_apply_path():
    body = review.render_review_html(_manifest())

    assert body.count("<audio controls") == 26
    assert 'id="build"' in body
    assert 'id="copy"' in body
    assert "addEventListener('click',build)" in body
    assert "navigator.clipboard.writeText" in body
    assert "document.execCommand('copy')" in body
    assert "fetch(" not in body
    assert 'method="post"' not in body.casefold()
    assert re.search(r"clips/review-window-[a-f0-9]{24}\.wav", body)


def test_complete_answer_block_round_trips_and_incomplete_or_changed_fails():
    manifest = _manifest()
    decisions = [
        *manifest["merge_reviews"],
        *manifest["binding_reviews"],
        *[
            window
            for person in manifest["source_reviews"]
            for window in person["windows"]
        ],
    ]
    lines = [
        f"PLAN0063_SCHEMA={review.SUBMISSION_SCHEMA}",
        f"PLAN0063_P2_CONTENT_SHA256={manifest['reconciliation_content_sha256']}",
        f"PLAN0063_P3_CONTENT_SHA256={manifest['feasibility_content_sha256']}",
        f"PLAN0063_P4_CONTENT_SHA256={manifest['content_sha256']}",
        *[f"{item['decision_key']}={item['choices'][0]}" for item in decisions],
    ]

    parsed = review.parse_review_submission("\n".join(lines), manifest)
    assert len(parsed["decisions"]) == 30
    assert parsed["live_mutation_count"] == 0

    with pytest.raises(review.Plan0063HumanReviewError):
        review.parse_review_submission("\n".join(lines[:-1]), manifest)
    with pytest.raises(review.Plan0063HumanReviewError):
        review.parse_review_submission(
            "\n".join(lines).replace("SOURCE::", "SOURCE::changed-", 1), manifest
        )


def test_review_manifest_rejects_holdout_reuse_or_missing_clip_binding():
    reconciled, feasibility, clip_hashes = _inputs()
    feasibility["person_source_proposals"][0]["source_windows"][0][
        "future_holdout_excluded"
    ] = False
    with pytest.raises(review.Plan0063HumanReviewError):
        review.build_review_manifest(
            reconciled,
            feasibility,
            clip_sha256_by_reference=clip_hashes,
            repository_authority={},
        )

    reconciled, feasibility, clip_hashes = _inputs()
    clip_hashes.pop(next(iter(clip_hashes)))
    with pytest.raises(review.Plan0063HumanReviewError):
        review.build_review_manifest(
            reconciled,
            feasibility,
            clip_sha256_by_reference=clip_hashes,
            repository_authority={},
        )
