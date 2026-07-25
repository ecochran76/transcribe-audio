from __future__ import annotations

import json
from pathlib import Path

import transcript_store
from speaker_evaluation_campaign import main, preview_campaign


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
