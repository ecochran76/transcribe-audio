from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import speaker_evaluation_campaign as campaign
import transcript_store
from speaker_evaluation_campaign import (
    APPLY_SUCCESSOR_CAMPAIGN_TOKEN,
    APPLY_SUCCESSOR_IDENTITY_PROJECTION_TOKEN,
    apply_campaign,
    apply_successor_campaign,
    apply_successor_identity_projection,
    capture_blind_prediction,
    freeze_gold_batch,
    main,
    open_successor_review_case,
    preview_campaign,
    preview_successor_campaign,
    preview_successor_identity_projection,
    replay_successor_campaign,
    replay_speaker_confidence_calibration,
    record_gold_review,
    record_refinement_decision,
    reveal_blind_baseline_comparison,
    review_case_packet,
    start_blind_baseline,
)


def write_transcript(
    path: Path,
    *,
    recording_start: str,
    utterances: list[dict[str, object]],
    conversation_id: str = "",
    recording_id: str = "",
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
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if recording_id:
        payload["recording_id"] = recording_id
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def attach_source_blob(
    store_root: Path,
    document_id: str,
    source_path: Path,
) -> dict:
    source_path.write_bytes(b"RIFF" + document_id.encode("utf-8"))
    blob = transcript_store.prepare_blob(
        store_root,
        str(source_path),
        role="source_recording",
    )
    with transcript_store.connect(store_root) as con:
        transcript_store.init_db(con)
        transcript_store.upsert_document_blob(
            con,
            document_id,
            blob,
            now="2026-07-31T00:00:00Z",
        )
        con.commit()
    return blob


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
        "speaker_identity": "speaker-identity.v2",
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


def successor_campaign_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, str, list[str]]:
    monkeypatch.setattr(
        campaign,
        "_repository_state",
        lambda: {"commit": "a" * 40, "dirty_tree": False},
    )
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    corpus_root = tmp_path / "corpora"
    document_ids: list[str] = []
    blobs: list[dict] = []
    for index in range(1, 9):
        artifact = write_transcript(
            tmp_path / f"conversation-{index}.transcript.json",
            recording_start=f"2024-01-0{index}T10:00:00Z",
            conversation_id=f"00000000-0000-4000-8000-{index:012d}",
            recording_id=f"10000000-0000-4000-8000-{index:012d}",
            utterances=[
                {
                    "speaker": "A",
                    "start": 0,
                    "end": 1,
                    "text": f"Opening question {index} with enough context. " * 8,
                },
                {
                    "speaker": "B",
                    "start": 1,
                    "end": 2,
                    "text": f"Detailed response {index} with identifying context. " * 8,
                },
            ],
        )
        result = transcript_store.ingest_artifact(artifact, root=store_root)
        document_ids.append(result.id)
        blobs.append(
            attach_source_blob(
                store_root,
                result.id,
                tmp_path / f"source-{index}.wav",
            )
        )
    parent = apply_campaign(
        store_root=store_root,
        runtime_root=runtime_root,
        batch_size=1,
        approval_token="APPLY_SPEAKER_EVALUATION_CAMPAIGN_MANIFEST",
    )
    parent_dir = runtime_root / parent["campaign_id"]
    gold_dir = parent_dir / "gold"
    gold_dir.mkdir(mode=0o700)
    gold_index = gold_dir / "index.json"
    gold_index.write_text(
        json.dumps(
            {
                "schema_version": (
                    "transcribe-audio.speaker-evaluation-gold-index.v1"
                ),
                "records": [
                    {
                        "gold_id": "prior-gold",
                        "document_id": document_ids[0],
                        "chronological_rank": 1,
                        "disposition": "eligible_known",
                        "reviewed_at": "2026-07-31T00:00:00Z",
                        "supersedes_gold_id": "",
                        "path": "private-prior-gold.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gold_index.chmod(0o600)
    corpus_dir = corpus_root / ("acoustic-corpus-" + "b" * 24)
    corpus_dir.mkdir(parents=True, mode=0o700)
    corpus_manifest = corpus_dir / "manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "corpus_id": corpus_dir.name,
                "recordings": [
                    {
                        "document_id": document_ids[0],
                        "conversation_id": "00000000-0000-4000-8000-000000000001",
                        "recording_id": "10000000-0000-4000-8000-000000000001",
                        "source_blob": {"sha256": blobs[0]["sha256"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_manifest.chmod(0o600)
    identity_preview = preview_successor_identity_projection(
        parent["campaign_id"],
        store_root=store_root,
        runtime_root=runtime_root,
    )
    identity = apply_successor_identity_projection(
        parent["campaign_id"],
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token=APPLY_SUCCESSOR_IDENTITY_PROJECTION_TOKEN,
        expected_content_sha256=identity_preview["content_sha256"],
        expected_projection_id=identity_preview["projection_id"],
    )
    return (
        store_root,
        runtime_root,
        corpus_root,
        Path(identity["projection_path"]),
        parent["campaign_id"],
        document_ids,
    )


def test_successor_campaign_freezes_full_body_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root, runtime_root, corpus_root, identity_path, parent_id, document_ids = (
        successor_campaign_fixture(tmp_path, monkeypatch)
    )
    outside_artifact = write_transcript(
        tmp_path / "outside-parent.transcript.json",
        recording_start="2024-02-01T10:00:00Z",
        conversation_id="20000000-0000-4000-8000-000000000001",
        recording_id="30000000-0000-4000-8000-000000000001",
        utterances=[
            {
                "speaker": "A",
                "start": 0,
                "end": 1,
                "text": "Outside parent opening with enough context. " * 8,
            },
            {
                "speaker": "B",
                "start": 1,
                "end": 2,
                "text": "Outside parent response with enough context. " * 8,
            },
        ],
    )
    outside = transcript_store.ingest_artifact(outside_artifact, root=store_root)
    attach_source_blob(store_root, outside.id, tmp_path / "outside-parent.wav")

    real_connect = campaign.connect

    class GuardedConnection:
        def __init__(self, connection: object) -> None:
            self.connection = connection

        def __enter__(self) -> GuardedConnection:
            return self

        def __exit__(self, *args: object) -> None:
            self.connection.close()

        def execute(self, sql: str, parameters: object = ()) -> object:
            lowered = sql.lower()
            assert "json_payload" not in lowered
            assert "text_content" not in lowered
            return self.connection.execute(sql, parameters)

    monkeypatch.setattr(
        campaign,
        "connect",
        lambda root: GuardedConnection(real_connect(root)),
    )
    monkeypatch.setattr(
        campaign,
        "preview_campaign",
        lambda **_kwargs: pytest.fail("selector opened legacy full-store preview"),
    )
    monkeypatch.setattr(
        campaign.transcript_artifact_access,
        "read_resolved_transcript",
        lambda *_args, **_kwargs: pytest.fail("selector read transcript artifact"),
    )
    with pytest.raises(ValueError, match="exactly 7"):
        preview_successor_campaign(
            parent_id,
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
            identity_projection_path=identity_path,
            batch_size=6,
        )
    preview = preview_successor_campaign(
        parent_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
        identity_projection_path=identity_path,
        batch_size=7,
    )
    assert [item["document_id"] for item in preview["items"]] == document_ids[1:]
    assert outside.id not in {item["document_id"] for item in preview["items"]}
    assert preview["summary"]["durable_disjoint_candidate_pool"] == 7
    assert preview["selection_policy"]["availability_conditioned_census"] is True
    assert len(preview["selection_authority"]["selector_module_sha256"]) == 64
    assert len(preview["selection_authority"]["metadata_projection_sha256"]) == 64
    serialized = json.dumps(preview)
    assert "Opening question" not in serialized
    assert "source_stored_path" not in serialized

    with pytest.raises(ValueError, match="exact approval token"):
        apply_successor_campaign(
            parent_id,
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
            identity_projection_path=identity_path,
            batch_size=7,
            approval_token="wrong",
            expected_content_sha256=preview["content_sha256"],
            expected_manifest_id=preview["manifest_id"],
        )
    corpus_manifest_path = next(corpus_root.glob("*/manifest.json"))
    original_corpus_manifest = corpus_manifest_path.read_text(encoding="utf-8")
    drifted_corpus_manifest = json.loads(original_corpus_manifest)
    drifted_corpus_manifest["review_note"] = "authority drift"
    corpus_manifest_path.write_text(
        json.dumps(drifted_corpus_manifest),
        encoding="utf-8",
    )
    corpus_manifest_path.chmod(0o600)
    with pytest.raises(ValueError, match="preview drifted"):
        apply_successor_campaign(
            parent_id,
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
            identity_projection_path=identity_path,
            batch_size=7,
            approval_token=APPLY_SUCCESSOR_CAMPAIGN_TOKEN,
            expected_content_sha256=preview["content_sha256"],
            expected_manifest_id=preview["manifest_id"],
        )
    corpus_manifest_path.write_text(
        original_corpus_manifest,
        encoding="utf-8",
    )
    corpus_manifest_path.chmod(0o600)
    applied = apply_successor_campaign(
        parent_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
        identity_projection_path=identity_path,
        batch_size=7,
        approval_token=APPLY_SUCCESSOR_CAMPAIGN_TOKEN,
        expected_content_sha256=preview["content_sha256"],
        expected_manifest_id=preview["manifest_id"],
    )
    replayed = replay_successor_campaign(
        applied["campaign_id"],
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
    )
    repeated = apply_successor_campaign(
        parent_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
        identity_projection_path=identity_path,
        batch_size=7,
        approval_token=APPLY_SUCCESSOR_CAMPAIGN_TOKEN,
        expected_content_sha256=preview["content_sha256"],
        expected_manifest_id=preview["manifest_id"],
    )
    manifest_path = Path(applied["manifest_path"])
    assert replayed["full_body_equal"] is True
    assert replayed["manifest_sha256"] == applied["manifest_sha256"]
    assert repeated["manifest_sha256"] == applied["manifest_sha256"]
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.parent.stat().st_mode) == 0o700

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["summary"]["selected_recordings"] = 99
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(ValueError, match="full-body replay mismatch"):
        replay_successor_campaign(
            applied["campaign_id"],
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
        )


def test_successor_review_cursor_enforces_one_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root, runtime_root, corpus_root, identity_path, parent_id, document_ids = (
        successor_campaign_fixture(tmp_path, monkeypatch)
    )
    preview = preview_successor_campaign(
        parent_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
        identity_projection_path=identity_path,
        batch_size=7,
    )
    applied = apply_successor_campaign(
        parent_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
        identity_projection_path=identity_path,
        batch_size=7,
        approval_token=APPLY_SUCCESSOR_CAMPAIGN_TOKEN,
        expected_content_sha256=preview["content_sha256"],
        expected_manifest_id=preview["manifest_id"],
    )
    child_id = applied["campaign_id"]
    with pytest.raises(ValueError, match="cursor"):
        review_case_packet(
            child_id,
            document_ids[1],
            store_root=store_root,
            runtime_root=runtime_root,
        )
    with pytest.raises(ValueError, match="manifest order"):
        open_successor_review_case(
            child_id,
            document_id=document_ids[2],
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
        )
    opened = open_successor_review_case(
        child_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
    )
    assert opened["packet"]["document_id"] == document_ids[1]
    assert opened["idempotent_reopen"] is False
    reopened = open_successor_review_case(
        child_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
    )
    assert reopened["idempotent_reopen"] is True
    open_path = Path(opened["open_receipt"]["receipt_path"])
    original_open_receipt = open_path.read_text(encoding="utf-8")
    tampered_open_receipt = json.loads(original_open_receipt)
    tampered_open_receipt["will_read_gold_body"] = True
    open_path.write_text(json.dumps(tampered_open_receipt), encoding="utf-8")
    open_path.chmod(0o600)
    with pytest.raises(ValueError, match="history binding"):
        open_successor_review_case(
            child_id,
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
        )
    open_path.write_text(original_open_receipt, encoding="utf-8")
    open_path.chmod(0o600)

    gold_index_path = runtime_root / child_id / "gold" / "index.json"
    gold_index_path.parent.mkdir(mode=0o700)
    empty_gold_index = {
        "schema_version": "transcribe-audio.speaker-evaluation-gold-index.v1",
        "records": [],
    }
    tampered_gold_index = {
        **empty_gold_index,
        "records": [
            {
                "gold_id": "unattributed",
                "document_id": "outside-successor-campaign",
            }
        ],
    }
    gold_index_path.write_text(json.dumps(tampered_gold_index), encoding="utf-8")
    gold_index_path.chmod(0o600)
    with pytest.raises(ValueError, match="gold index changed"):
        open_successor_review_case(
            child_id,
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
        )
    gold_index_path.write_text(json.dumps(empty_gold_index), encoding="utf-8")
    gold_index_path.chmod(0o600)
    with pytest.raises(ValueError, match="outstanding"):
        open_successor_review_case(
            child_id,
            document_id=document_ids[2],
            store_root=store_root,
            runtime_root=runtime_root,
            corpus_root=corpus_root,
        )
    with pytest.raises(ValueError, match="cursor"):
        record_gold_review(
            child_id,
            document_ids[2],
            {
                "disposition": "eligible_known",
                "reviewer": "operator",
                "review_method": "operator_test",
            },
            store_root=store_root,
            runtime_root=runtime_root,
            approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
        )
    first_gold = record_gold_review(
        child_id,
        document_ids[1],
        {
            "disposition": "eligible_known",
            "calendar_association": "uncertain",
            "people": [
                {"person_ground_truth_id": "person-a"},
                {"person_ground_truth_id": "person-b"},
            ],
            "speaker_outcomes": [
                {
                    "speaker_label": "A",
                    "outcome": "person",
                    "person_ground_truth_id": "person-a",
                },
                {
                    "speaker_label": "B",
                    "outcome": "person",
                    "person_ground_truth_id": "person-b",
                },
            ],
            "same_person_label_groups": [],
            "reviewer": "operator",
            "review_method": "operator_test",
        },
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
    )
    assert first_gold["prediction_visibility"] == "excluded"
    second = open_successor_review_case(
        child_id,
        store_root=store_root,
        runtime_root=runtime_root,
        corpus_root=corpus_root,
    )
    assert second["packet"]["document_id"] == document_ids[2]
    assert second["open_receipt"]["position"] == 2
    second_open_path = Path(second["open_receipt"]["receipt_path"])
    assert stat.S_IMODE(second_open_path.stat().st_mode) == 0o600


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


def test_holdout_uses_reserved_documents_and_reveals_only_post_prediction_gold(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    runtime_root = tmp_path / "campaigns"
    first_artifact = write_transcript(
        tmp_path / "first.transcript.json",
        recording_start="2024-01-01T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "First opening. " * 20},
        ],
    )
    holdout_artifact = write_transcript(
        tmp_path / "holdout.transcript.json",
        recording_start="2024-01-02T10:00:00Z",
        utterances=[
            {"speaker": "A", "start": 0, "end": 1, "text": "Holdout opening. " * 20},
            {"speaker": "A", "start": 1, "end": 2, "text": "Holdout response. " * 20},
        ],
    )
    first = transcript_store.ingest_artifact(first_artifact, root=store_root)
    holdout = transcript_store.ingest_artifact(holdout_artifact, root=store_root)
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
            }
        ],
        "same_person_label_groups": [],
        "reviewer": "Eric Cochran",
        "review_method": "transcript_and_calendar",
        "notes": "",
    }
    record_gold_review(
        applied["campaign_id"],
        first.id,
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
        run_kind="holdout",
    )

    assert baseline["document_ids"] == [holdout.id]
    assert baseline["gold_content_included"] is False
    captured = capture_blind_prediction(
        applied["campaign_id"],
        baseline_id=baseline["baseline_id"],
        document_id=holdout.id,
        artifact_sha256=baseline["cases"][0]["artifact_sha256"],
        prediction={"evaluation_id": "holdout-evaluation"},
        runtime_root=runtime_root,
        approval_token="CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION",
    )
    with pytest.raises(ValueError, match="reviewed after predictions completed"):
        reveal_blind_baseline_comparison(
            applied["campaign_id"],
            baseline_id=baseline["baseline_id"],
            runtime_root=runtime_root,
            approval_token="REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON",
        )

    gold = record_gold_review(
        applied["campaign_id"],
        holdout.id,
        {
            **review,
            "disposition": "duplicate_member",
            "people": [],
            "speaker_outcomes": [],
        },
        store_root=store_root,
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_GOLD",
    )
    comparison = reveal_blind_baseline_comparison(
        applied["campaign_id"],
        baseline_id=baseline["baseline_id"],
        runtime_root=runtime_root,
        approval_token="REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON",
    )

    assert comparison["status"] == "comparison_complete"
    assert comparison["cases"][0]["gold_id"] == gold["gold_id"]
    assert comparison["cases"][0]["evaluation_excluded"] is True
    assert comparison["metrics"]["calendar_association"]["cases"] == 0
    assert comparison["cases"][0]["prediction_captured_at"] <= gold["reviewed_at"]
    assert captured["predictions_completed_at"] <= gold["reviewed_at"]

    replay = start_blind_baseline(
        applied["campaign_id"],
        freeze_id=frozen["freeze_id"],
        runtime_root=runtime_root,
        approval_token="START_SPEAKER_EVALUATION_BLIND_BASELINE",
        run_kind="holdout",
        hypothesis="reference repair on reviewed holdout",
        evidence_mode="fresh_retrieval_comparison",
    )
    capture_blind_prediction(
        applied["campaign_id"],
        baseline_id=replay["baseline_id"],
        document_id=holdout.id,
        artifact_sha256=replay["cases"][0]["artifact_sha256"],
        prediction={"evaluation_id": "holdout-replay-evaluation"},
        runtime_root=runtime_root,
        approval_token="CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION",
    )
    replay_comparison = reveal_blind_baseline_comparison(
        applied["campaign_id"],
        baseline_id=replay["baseline_id"],
        runtime_root=runtime_root,
        approval_token="REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON",
        allow_reviewed_holdout_replay=True,
    )

    assert replay_comparison["status"] == "comparison_complete"
    assert replay_comparison["comparison_mode"] == "reviewed_holdout_replay"
    assert replay_comparison["prior_holdout_baseline_id"] == baseline["baseline_id"]
    assert replay_comparison["predictions_captured_before_gold_reveal"] is False


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

    refinement = start_blind_baseline(
        applied["campaign_id"],
        freeze_id=frozen["freeze_id"],
        runtime_root=runtime_root,
        approval_token="START_SPEAKER_EVALUATION_BLIND_BASELINE",
        run_kind="refinement",
        parent_baseline_id=baseline["baseline_id"],
        hypothesis="clarify citation locality",
        evidence_mode="fresh_retrieval_comparison",
    )
    capture_blind_prediction(
        applied["campaign_id"],
        baseline_id=refinement["baseline_id"],
        document_id=result.id,
        artifact_sha256=refinement["cases"][0]["artifact_sha256"],
        prediction={
            "evaluation_id": "evaluation-refinement",
            "calendar_association": {
                "status": "matched",
                "confidence": {"numeric": 90, "band": "very_high"},
            },
            "people": [],
            "proposals": [],
        },
        runtime_root=runtime_root,
        approval_token="CAPTURE_SPEAKER_EVALUATION_BLIND_PREDICTION",
    )
    reveal_blind_baseline_comparison(
        applied["campaign_id"],
        baseline_id=refinement["baseline_id"],
        runtime_root=runtime_root,
        approval_token="REVEAL_SPEAKER_EVALUATION_GOLD_COMPARISON",
    )
    decision = record_refinement_decision(
        applied["campaign_id"],
        baseline_id=refinement["baseline_id"],
        decision="rejected",
        target_failure_class="transcript_clue_discovery",
        rationale="Target failures were unchanged and a regression appeared.",
        runtime_root=runtime_root,
        approval_token="RECORD_SPEAKER_EVALUATION_REFINEMENT_DECISION",
    )
    assert decision["decision"] == "rejected"
    assert decision["parent_baseline_id"] == baseline["baseline_id"]
    assert Path(decision["decision_path"]).exists()


def test_confidence_calibration_replay_writes_redacted_private_receipt(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "campaigns"
    campaign_id = "campaign-" + ("a" * 20)
    baseline_id = "baseline-00000000-0000-4000-8000-000000000001"
    baseline_dir = runtime_root / campaign_id / "baselines" / baseline_id
    prediction_path = baseline_dir / "predictions" / "doc-1" / "prediction.json"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text(
        json.dumps(
            {
                "prediction": {
                    "proposals": [
                        {
                            "speaker_labels": ["A"],
                            "status": "conflicting",
                            "review_flags": [],
                            "factors": [],
                            "confidence": {
                                "numeric": 100,
                                "band": "very_high",
                                "band_label": "Very High",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "baseline.json").write_text(
        json.dumps(
            {
                "baseline_id": baseline_id,
                "campaign_id": campaign_id,
                "status": "refinement_rejected",
                "cases": [
                    {
                        "document_id": "doc-1",
                        "prediction_path": str(prediction_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "comparison.json").write_text(
        json.dumps(
            {
                "baseline_id": baseline_id,
                "campaign_id": campaign_id,
                "status": "comparison_complete",
                "metrics": {
                    "validation": {
                        "predictions": 1,
                        "completed": 1,
                        "model_output_rejected": 0,
                    }
                },
                "cases": [
                    {
                        "document_id": "doc-1",
                        "speaker_labels": [
                            {
                                "speaker_label": "A",
                                "gold_outcome": "person",
                                "top_proposal_correct": False,
                                "top_confidence_band": "very_high",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = replay_speaker_confidence_calibration(
        campaign_id,
        baseline_ids=[baseline_id],
        runtime_root=runtime_root,
        approval_token="REPLAY_SPEAKER_CONFIDENCE_CALIBRATION",
    )

    assert receipt["metrics"]["before_high_or_very_high_wrong"] == 1
    assert receipt["metrics"]["after_high_or_very_high_wrong"] == 0
    assert receipt["metrics"]["top_proposal_correct"] == 0
    assert receipt["cases"][0]["calibration_reasons"] == [
        "assignment_status:conflicting"
    ]
    receipt_path = Path(receipt["receipt_path"])
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert "prediction" not in json.dumps(receipt["cases"])
