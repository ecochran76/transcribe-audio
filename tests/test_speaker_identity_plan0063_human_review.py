from __future__ import annotations

import hashlib
import json
import re
import stat

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
    people[0]["member_names"] = ["Michael Forrester"]
    people[0]["member_slot_ids"] = sorted(review.ABSENT_PARTICIPANT_SLOTS)
    people[4]["member_slot_ids"] = [
        "47ea79857aa1ac2d1d79::SPEAKER_2",
        "47ea79857aa1ac2d1d79::SPEAKER_3",
    ]
    people[4]["member_names"] = ["Dr. Stefl"]
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


def _comparison_audio(reconciled):
    slots = [
        slot
        for proposal in reconciled["merge_proposals"]
        for slot in proposal["member_slot_ids"]
    ]
    return {
        slot: {
            "recording_ordinal": index,
            "speaker_ref": f"SPEAKER_{index}",
            "clip_url": (
                f"comparison-clips/recording-{index:02d}/SPEAKER_{index}.wav"
            ),
            "clip_sha256": f"{index + 100:064x}",
        }
        for index, slot in enumerate(slots, 1)
    }


def _manifest():
    reconciled, feasibility, clip_hashes = _inputs()
    return review.build_review_manifest(
        reconciled,
        feasibility,
        clip_sha256_by_reference=clip_hashes,
        comparison_audio_by_slot=_comparison_audio(reconciled),
        repository_authority={"commit": "example"},
    )


def test_review_manifest_preserves_denominator_and_no_calendar_correction():
    manifest = _manifest()

    assert manifest["decision_count"] == 30
    assert len(manifest["merge_reviews"]) == 3
    assert all(len(item["comparison_audio"]) == 2 for item in manifest["merge_reviews"])
    assert len(manifest["binding_reviews"]) == 1
    assert sum(len(item["windows"]) for item in manifest["source_reviews"]) == 26
    correction = manifest["recording_context_correction"]
    assert correction["calendar_status"] == "operator_confirmed_no_calendar_event"
    assert correction["calendar_evidence_available"] is False
    assert correction["calendar_candidate_claim_withdrawn"] is True
    assert correction["identified_display_label"] == "Dr. Stefl"
    assert correction["identity_authority"] == "operator_listening_review"
    assert correction["absent_participant_display_label"] == "Michael Forrester"
    assert all(
        not slot.startswith(f"{review.NO_CALENDAR_DOCUMENT_ID}::")
        for slot in correction["absent_participant_member_slot_ids"]
    )
    assert manifest["supersedes_review_content_sha256"] == (
        review.SUPERSEDED_REVIEW_SHA256
    )
    assert all(item["selected"] is None for item in manifest["merge_reviews"])
    assert not any(manifest["negative_actions"].values())


def test_review_html_has_direct_audio_working_export_controls_and_no_apply_path():
    body = review.render_review_html(_manifest())

    assert body.count("<audio controls") == 32
    assert body.count("Open this WAV directly") == 32
    assert 'id="build"' in body
    assert 'id="copy"' in body
    assert "addEventListener('click',build)" in body
    assert "navigator.clipboard.writeText" in body
    assert "document.execCommand('copy')" in body
    assert "Separate recording-context correction — no answer required" in body
    assert "This notice is not attached to the questions below" in body
    assert "Listen to both labeled voice samples" in body
    assert "Voice sample 1: Recording" in body
    assert "identified by operator listening review, not calendar evidence" in body
    assert "Michael Forrester</strong> is not present" in body
    assert "rows.join('\\n')" in body
    assert "rows.join('\n')" not in body
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
            comparison_audio_by_slot=_comparison_audio(reconciled),
            repository_authority={},
        )


def test_review_manifest_rejects_absent_participant_in_no_calendar_recording():
    reconciled, feasibility, clip_hashes = _inputs()
    michael = next(
        person
        for person in reconciled["person_proposals"]
        if person["member_names"] == ["Michael Forrester"]
    )
    michael["member_slot_ids"] = [
        f"{review.NO_CALENDAR_DOCUMENT_ID}::SPEAKER_1"
    ]

    with pytest.raises(
        review.Plan0063HumanReviewError,
        match="no-calendar context is incomplete",
    ):
        review.build_review_manifest(
            reconciled,
            feasibility,
            clip_sha256_by_reference=clip_hashes,
            comparison_audio_by_slot=_comparison_audio(reconciled),
            repository_authority={},
        )

    reconciled, feasibility, clip_hashes = _inputs()
    clip_hashes.pop(next(iter(clip_hashes)))
    with pytest.raises(review.Plan0063HumanReviewError):
        review.build_review_manifest(
            reconciled,
            feasibility,
            clip_sha256_by_reference=clip_hashes,
            comparison_audio_by_slot=_comparison_audio(reconciled),
            repository_authority={},
        )


def test_review_manifest_rejects_missing_grouping_comparison_audio():
    reconciled, feasibility, clip_hashes = _inputs()
    comparison_audio = _comparison_audio(reconciled)
    comparison_audio.pop(next(iter(comparison_audio)))

    with pytest.raises(
        review.Plan0063HumanReviewError,
        match="missing comparison audio",
    ):
        review.build_review_manifest(
            reconciled,
            feasibility,
            clip_sha256_by_reference=clip_hashes,
            comparison_audio_by_slot=comparison_audio,
            repository_authority={},
        )


def test_comparison_audio_copy_makes_every_directory_private(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    preview_root = source_root / "preview"
    target_root = tmp_path / "target"
    for path in (source_root, preview_root, target_root):
        path.mkdir(mode=0o700)
    cards = []
    clips = []
    slots = set()
    for index in range(1, 7):
        slot = f"doc{index}::SPEAKER_{index}"
        slots.add(slot)
        relative = f"media/recording-{index:02d}/SPEAKER_{index}.wav"
        audio = preview_root / relative
        review.ensure_private_tree(source_root, audio.parent)
        audio.write_bytes(f"audio-{index}".encode())
        audio.chmod(0o600)
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        cards.append(
            {
                "slot_id": slot,
                "recording_ordinal": index,
                "speaker_ref": f"SPEAKER_{index}",
                "audio_path": relative,
            }
        )
        clips.append(
            {"slot_id": slot, "relative_path": relative, "sha256": digest}
        )
    manifest = {
        "schema_version": "transcribe-audio.plan0062-human-review-manifest.v1",
        "status": "awaiting_literal_human_review",
        "packet": {
            "content_sha256": review.PLAN0062_REVIEW_CONTENT_SHA256,
            "cards": cards,
        },
        "audio_clips": clips,
    }
    manifest_path = source_root / "private-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(review, "PLAN0062_REVIEW_MANIFEST_SHA256", manifest_hash)
    receipt_path = source_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "content_sha256": review.PLAN0062_REVIEW_CONTENT_SHA256,
                "manifest_sha256": manifest_hash,
                "audio_clip_count": 10,
                "live_mutation_count": 0,
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)

    copied = review._copy_comparison_audio(
        source_root=source_root,
        target_root=target_root,
        required_slots=slots,
    )

    assert len(copied) == 6
    directories = [path for path in target_root.rglob("*") if path.is_dir()]
    assert directories
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)
