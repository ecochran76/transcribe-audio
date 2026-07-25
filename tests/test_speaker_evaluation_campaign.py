from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import transcript_store
from speaker_evaluation_campaign import (
    apply_campaign,
    capture_blind_prediction,
    freeze_gold_batch,
    main,
    preview_campaign,
    record_gold_review,
    reveal_blind_baseline_comparison,
    review_case_packet,
    start_blind_baseline,
)


def write_transcript(
    path: Path,
    *,
    recording_start: str,
    utterances: list[dict[str, object]],
) -> Path:
    payload = {
        "schema_version": 2,
        "transcript_title": path.stem,
        "backend": "test",
        "recording_start": recording_start,
        "recording_end": recording_start,
        "duration_seconds": 60,
        "transcript_text": "\n".join(
            transcript_store.formatted_utterance_text(utterance)
            for utterance in utterances
        ),
        "utterance_count": len(utterances),
        "utterances": utterances,
        "event": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_preview_orders_oldest_first_without_creating_campaign_state(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    newer = write_transcript(
        tmp_path / "newer.transcript.json",
        recording_start="2024-01-02T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening question with enough context. " * 5},
            {"speaker": "B", "start": 1, "end": 2, "text": "Detailed response with identifying context. " * 5},
        ],
    )
    older = write_transcript(
        tmp_path / "older.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Earlier opening question with context. " * 5},
            {"speaker": "B", "start": 1, "end": 2, "text": "Earlier detailed response with context. " * 5},
        ],
    )
    newer_result = transcript_store.ingest_artifact(newer, root=store_root)
    older_result = transcript_store.ingest_artifact(older, root=store_root)

    manifest = preview_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
    )

    assert manifest["schema_version"] == "transcribe-audio.speaker-evaluation-campaign-manifest.v1"
    assert [item["document_id"] for item in manifest["items"]] == [
        older_result.id,
        newer_result.id,
    ]
    assert manifest["items"][0]["chronological_rank"] == 1
    assert manifest["items"][0]["disposition"] == "needs_operator_classification"
    assert manifest["cursor"]["document_id"] == older_result.id
    assert manifest["summary"]["total_rows"] == 2
    assert not runtime_root.exists()


def test_preview_keeps_incomplete_transcripts_counted_but_out_of_review_batch(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    short = write_transcript(
        tmp_path / "short.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Accidental recording."},
        ],
    )
    result = transcript_store.ingest_artifact(short, root=store_root)

    manifest = preview_campaign(store_root=store_root, batch_size=10)

    item = manifest["items"][0]
    assert item["document_id"] == result.id
    assert item["disposition"] == "incomplete"
    assert item["disposition_reason"] == "one_or_zero_utterances"
    assert manifest["summary"]["disposition_counts"] == {"incomplete": 1}
    assert manifest["cursor"]["document_id"] == ""


def test_preview_marks_missing_source_and_stored_copy_unavailable(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    artifact = write_transcript(
        tmp_path / "missing.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    result = transcript_store.ingest_artifact(artifact, root=store_root)
    artifact.unlink()
    Path(result.stored_path).unlink()

    manifest = preview_campaign(store_root=store_root, batch_size=10)

    item = manifest["items"][0]
    assert item["disposition"] == "artifact_unavailable"
    assert (
        item["disposition_reason"]
        == "no_accessible_source_or_stored_artifact"
    )
    assert item["artifact"]["selected_location"] == "unavailable"
    assert manifest["cursor"]["document_id"] == ""


def test_preview_clusters_exact_normalized_duplicates_without_spending_two_slots(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    utterances = [
        {"speaker": "A", "start": 0, "end": 1, "text": "Opening context. " * 12},
        {"speaker": "B", "start": 1, "end": 2, "text": "Substantive answer. " * 12},
    ]
    first = write_transcript(
        tmp_path / "first.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=utterances,
    )
    duplicate = write_transcript(
        tmp_path / "duplicate.transcript.json",
        recording_start="2024-01-01T10:05:00Z",
        utterances=utterances,
    )
    first_result = transcript_store.ingest_artifact(first, root=store_root)
    duplicate_result = transcript_store.ingest_artifact(duplicate, root=store_root)

    manifest = preview_campaign(store_root=store_root, batch_size=1)

    first_item, duplicate_item = manifest["items"]
    assert first_item["document_id"] == first_result.id
    assert first_item["disposition"] == "needs_operator_classification"
    assert duplicate_item["document_id"] == duplicate_result.id
    assert duplicate_item["disposition"] == "duplicate_member"
    assert duplicate_item["duplicate_of_document_id"] == first_result.id
    assert first_item["duplicate_cluster_id"] == duplicate_item["duplicate_cluster_id"]
    assert manifest["summary"]["duplicate_cluster_count"] == 1


def test_preview_reserves_gold_review_and_blind_holdout_candidates(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    for index, text in enumerate(("first topic", "second topic"), start=1):
        artifact = write_transcript(
            tmp_path / f"conversation-{index}.transcript.json",
            recording_start=f"2024-01-0{index}T10:00:00Z",
            utterances=[
                {"speaker": "A", "start": 0, "end": 1, "text": f"{text} opening. " * 12},
                {"speaker": "B", "start": 1, "end": 2, "text": f"{text} answer. " * 12},
            ],
        )
        transcript_store.ingest_artifact(artifact, root=store_root)

    manifest = preview_campaign(store_root=store_root, batch_size=1)

    assert [item["candidate_role"] for item in manifest["items"]] == [
        "gold_review_candidate",
        "blind_holdout_candidate",
    ]
    assert manifest["summary"]["gold_review_candidate_count"] == 1
    assert manifest["summary"]["blind_holdout_candidate_count"] == 1


def test_preview_cli_emits_json_without_applying_state(
    tmp_path: Path,
    capsys,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    transcript_store.ingest_artifact(artifact, root=store_root)

    exit_code = main(
        [
            "preview",
            "--store-root",
            str(store_root),
            "--runtime-root",
            str(runtime_root),
            "--batch-size",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "preview"
    assert payload["summary"]["total_rows"] == 1
    assert not runtime_root.exists()


def test_preview_records_reproducibility_fingerprints_and_live_model_route(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    transcript_store.ingest_artifact(artifact, root=store_root)

    first = preview_campaign(store_root=store_root, batch_size=1)
    second = preview_campaign(store_root=store_root, batch_size=1)

    assert first["manifest_id"] == second["manifest_id"]
    assert first["algorithm"]["commit"]
    assert isinstance(first["algorithm"]["dirty_tree"], bool)
    assert first["model_route"]["provider"] == "codex-app-server"
    assert first["model_route"]["model"] == "gpt-5.6-sol"
    assert first["rubric_versions"] == {
        "calendar_association": "calendar-association.v1",
        "person_link": "person-link.v1",
        "speaker_identity": "speaker-identity.v1",
    }
    assert len(first["provenance_config_fingerprint"]) == 64


def test_apply_campaign_requires_approval_and_writes_private_manifest(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    transcript_store.ingest_artifact(artifact, root=store_root)

    with pytest.raises(ValueError, match="approval token"):
        apply_campaign(
            store_root=store_root,
            runtime_root=runtime_root,
            batch_size=1,
            approval_token="wrong",
        )
    applied = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )

    manifest_path = Path(applied["manifest_path"])
    assert manifest_path.exists()
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.parent.stat().st_mode) == 0o700
    assert applied["will_execute_app_intelligence"] is False
    assert applied["will_perform_external_write"] is False


def test_review_packet_is_private_clue_surface_without_gold(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "My name is Alice. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Hello Alice. " * 20},
        ],
    )
    result = transcript_store.ingest_artifact(artifact, root=store_root)
    applied = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )

    packet = review_case_packet(
        applied["campaign_id"],
        result.id,
        store_root=store_root,
        runtime_root=runtime_root,
    )

    assert packet["document_id"] == result.id
    assert packet["speaker_labels"] == ["A", "B"]
    assert packet["utterances"][0]["text"].startswith("My name is Alice")
    assert packet["will_read_gold_records"] is False
    assert "speaker_outcomes" not in packet
    assert "people" not in packet
    assert packet["will_execute_app_intelligence"] is False


def test_gold_review_is_append_only_and_freezes_complete_batch(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    result = transcript_store.ingest_artifact(artifact, root=store_root)
    applied = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )
    review = {
        "disposition": "eligible_known",
        "calendar_association": "correct",
        "people": [
            {
                "person_ground_truth_id": "person-alice",
                "name": "Alice Example",
                "email": "alice@example.com",
            }
        ],
        "speaker_outcomes": [
            {
                "speaker_label": "A",
                "outcome": "person",
                "person_ground_truth_id": "person-alice",
            },
            {
                "speaker_label": "B",
                "outcome": "unknown_to_reviewer",
                "person_ground_truth_id": "",
            },
        ],
        "same_person_label_groups": [],
        "reviewer": "Eric Cochran",
        "review_method": "transcript_and_calendar",
        "notes": "",
    }

    first = record_gold_review(
        applied["campaign_id"],
        result.id,
        review,
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
    )
    second = record_gold_review(
        applied["campaign_id"],
        result.id,
        {**review, "notes": "Corrected note."},
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
        supersedes_gold_id=first["gold_id"],
    )
    frozen = freeze_gold_batch(
        applied["campaign_id"],
        runtime_root=runtime_root,
        approval_token="FREEZE_SPEAKER_EVALUATION_GOLD_BATCH",
    )

    assert first["gold_id"] != second["gold_id"]
    assert Path(first["gold_path"]).exists()
    assert Path(second["gold_path"]).exists()
    assert second["supersedes_gold_id"] == first["gold_id"]
    assert frozen["status"] == "gold_batch_frozen"
    assert frozen["gold_case_count"] == 1
    assert frozen["gold_ids"] == [second["gold_id"]]
    assert Path(frozen["freeze_path"]).exists()


def test_blind_baseline_starts_from_freeze_without_exposing_gold(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    result = transcript_store.ingest_artifact(artifact, root=store_root)
    applied = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )
    review = {
        "disposition": "eligible_known",
        "calendar_association": "correct",
        "people": [
            {
                "person_ground_truth_id": "person-alice",
                "name": "Alice Example",
                "email": "alice@example.com",
            }
        ],
        "speaker_outcomes": [
            {
                "speaker_label": "A",
                "outcome": "person",
                "person_ground_truth_id": "person-alice",
            },
            {
                "speaker_label": "B",
                "outcome": "unknown_to_reviewer",
                "person_ground_truth_id": "",
            },
        ],
        "same_person_label_groups": [],
        "reviewer": "Eric Cochran",
        "review_method": "transcript_and_calendar",
        "notes": "",
    }
    gold = record_gold_review(
        applied["campaign_id"],
        result.id,
        review,
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
    )
    frozen = freeze_gold_batch(
        applied["campaign_id"],
        runtime_root=runtime_root,
        approval_token="FREEZE_SPEAKER_EVALUATION_GOLD_BATCH",
    )

    baseline = start_blind_baseline(
        applied["campaign_id"],
        freeze_id=frozen["freeze_id"],
        runtime_root=runtime_root,
        approval_token="START_SPEAKER_EVALUATION_BLIND_BASELINE",
    )

    serialized = json.dumps(baseline, sort_keys=True)
    assert baseline["status"] == "awaiting_predictions"
    assert baseline["document_ids"] == [result.id]
    assert baseline["prediction_visibility"] == "blind"
    assert baseline["will_read_gold_records"] is False
    assert baseline["captured_prediction_count"] == 0
    assert gold["gold_id"] not in serialized
    assert "Alice Example" not in serialized
    assert stat.S_IMODE(Path(baseline["baseline_path"]).stat().st_mode) == 0o600

    refinement = start_blind_baseline(
        applied["campaign_id"],
        freeze_id=frozen["freeze_id"],
        runtime_root=runtime_root,
        approval_token="START_SPEAKER_EVALUATION_BLIND_BASELINE",
        run_kind="refinement",
        parent_baseline_id=baseline["baseline_id"],
        hypothesis="speaker-specific citation locality",
        evidence_mode="fresh_retrieval_comparison",
    )
    assert refinement["run_kind"] == "refinement"
    assert refinement["parent_baseline_id"] == baseline["baseline_id"]
    assert refinement["hypothesis"] == "speaker-specific citation locality"
    assert refinement["evidence_mode"] == "fresh_retrieval_comparison"


def test_blind_prediction_capture_completes_batch_without_revealing_gold(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    artifact = write_transcript(
        tmp_path / "conversation.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Opening. " * 20},
            {"speaker": "B", "start": 1, "end": 2, "text": "Response. " * 20},
        ],
    )
    result = transcript_store.ingest_artifact(artifact, root=store_root)
    applied = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )
    review = {
        "disposition": "eligible_known",
        "calendar_association": "correct",
        "people": [
            {
                "person_ground_truth_id": "person-alice",
                "name": "Alice Example",
                "email": "alice@example.com",
            }
        ],
        "speaker_outcomes": [
            {
                "speaker_label": "A",
                "outcome": "person",
                "person_ground_truth_id": "person-alice",
            },
            {
                "speaker_label": "B",
                "outcome": "unknown_to_reviewer",
                "person_ground_truth_id": "",
            },
        ],
        "same_person_label_groups": [],
        "reviewer": "Eric Cochran",
        "review_method": "transcript_and_calendar",
        "notes": "",
    }
    gold = record_gold_review(
        applied["campaign_id"],
        result.id,
        review,
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
    )
    frozen = freeze_gold_batch(
        applied["campaign_id"],
        runtime_root=runtime_root,
        approval_token="FREEZE_SPEAKER_EVALUATION_GOLD_BATCH",
    )
    baseline = start_blind_baseline(
        applied["campaign_id"],
        freeze_id=frozen["freeze_id"],
        runtime_root=runtime_root,
        approval_token="START_SPEAKER_EVALUATION_BLIND_BASELINE",
    )

    captured = capture_blind_prediction(
        applied["campaign_id"],
        baseline_id=baseline["baseline_id"],
        document_id=result.id,
        artifact_sha256=baseline["cases"][0]["artifact_sha256"],
        prediction={
            "evaluation_id": "evaluation-1",
            "calendar_association": {
                "status": "matched",
                "confidence": {"numeric": 90, "band": "very_high"},
            },
            "people": [
                {
                    "person_id": "candidate-alice",
                    "label": "Alice Example",
                    "email": "alice@example.com",
                }
            ],
            "proposals": [
                {
                    "speaker_labels": ["A"],
                    "status": "candidate_match",
                    "person_id": "candidate-alice",
                    "confidence": {"numeric": 90, "band": "very_high"},
                }
            ],
        },
        run_references={
            "clue_discovery_run_id": "run-discovery",
            "identity_evaluation_run_id": "run-evaluation",
        },
        runtime_root=runtime_root,
        approval_token="CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION",
    )

    serialized = json.dumps(captured, sort_keys=True)
    assert captured["status"] == "predictions_complete"
    assert captured["captured_prediction_count"] == 1
    assert captured["will_read_gold_records"] is False
    assert gold["gold_id"] not in serialized
    assert Path(captured["prediction_path"]).exists()
    assert stat.S_IMODE(Path(captured["prediction_path"]).stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="already has a captured prediction"):
        capture_blind_prediction(
            applied["campaign_id"],
            baseline_id=baseline["baseline_id"],
            document_id=result.id,
            artifact_sha256=baseline["cases"][0]["artifact_sha256"],
            prediction={"evaluation_id": "evaluation-2"},
            runtime_root=runtime_root,
            approval_token="CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION",
        )

    comparison = reveal_blind_baseline_comparison(
        applied["campaign_id"],
        baseline_id=baseline["baseline_id"],
        runtime_root=runtime_root,
        approval_token="REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON",
    )

    assert comparison["status"] == "comparison_complete"
    assert comparison["predictions_captured_before_gold_reveal"] is True
    assert comparison["metrics"]["calendar_association"] == {
        "cases": 1,
        "exact": 1,
        "high_or_very_high_wrong": 0,
    }
    assert comparison["metrics"]["speaker_identity"] == {
        "known_person_labels": 1,
        "top_proposal_correct": 1,
        "correct_person_present": 1,
        "high_or_very_high_wrong": 0,
    }
    assert comparison["cases"][0]["gold_revealed_at"] >= captured[
        "predictions_completed_at"
    ]
    assert Path(comparison["comparison_path"]).exists()
