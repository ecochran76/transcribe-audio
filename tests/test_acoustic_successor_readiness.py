from __future__ import annotations

import json
import stat
from pathlib import Path

import transcript_store
from acoustic_successor_readiness import (
    assess_successor_readiness,
    build_readiness_receipt,
)


def _candidate(
    index: int, *, split: str = "evaluation", subject: str = "person-a"
) -> dict:
    return {
        "document_id": f"document-{index}",
        "recording_id": f"recording-{index}",
        "conversation_id": f"conversation-{index}",
        "split": split,
        "source_blob": {"sha256": f"{index:064x}", "stored_path": "/private/omitted"},
        "operator_gold": {
            "speaker_truth": [
                {"speaker_label": "A", "outcome": "known", "subject_id": subject}
            ]
        },
    }


def test_readiness_excludes_every_prior_identity_and_leaks_no_bodies() -> None:
    prior = _candidate(1)
    candidates = [prior, _candidate(2, subject="person-b")]
    receipt = build_readiness_receipt(
        candidates,
        [{"recordings": [prior]}],
        campaign_id="campaign-" + "a" * 20,
        campaign_authority_hashes={"manifest": "b" * 64},
        prior_manifest_hashes=["c" * 64],
    )
    assert receipt["counts"]["latest_eligible_candidates"] == 2
    assert receipt["counts"]["fully_disjoint_candidates"] == 1
    assert receipt["counts"]["overlap_by_identity"] == {
        "conversation": 1,
        "document": 1,
        "recording": 1,
        "source": 1,
    }
    assert receipt["contains_gold_body"] is False
    assert receipt["contains_source_paths"] is False
    serialized = json.dumps(receipt)
    assert "person-a" not in serialized
    assert "person-b" not in serialized
    assert "/private/omitted" not in serialized


def test_readiness_reports_zero_disjoint_candidates_as_blocked() -> None:
    prior = _candidate(1)
    receipt = build_readiness_receipt(
        [prior],
        [{"recordings": [prior]}],
        campaign_id="campaign-" + "a" * 20,
        campaign_authority_hashes={},
        prior_manifest_hashes=["c" * 64],
    )
    assert receipt["status"] == "blocked"
    assert receipt["counts"]["fully_disjoint_candidates"] == 0
    assert "no_fully_disjoint_operator_confirmed_candidates" in receipt["blockers"]
    assert receipt["will_read_audio"] is False
    assert receipt["will_run_models"] is False
    assert receipt["will_reveal_split"] is False


def test_readiness_identity_is_deterministic_except_timestamp() -> None:
    prior = _candidate(1)
    arguments = dict(
        candidates=[prior],
        prior_manifests=[{"recordings": [prior]}],
        campaign_id="campaign-" + "a" * 20,
        campaign_authority_hashes={},
        prior_manifest_hashes=["c" * 64],
    )
    first = build_readiness_receipt(**arguments)
    second = build_readiness_receipt(**arguments)
    assert first["readiness_id"] == second["readiness_id"]
    first.pop("assessed_at")
    second.pop("assessed_at")
    assert first == second


def test_live_assessment_never_opens_source_blob(
    tmp_path: Path, monkeypatch,
) -> None:
    campaign_id = "campaign-" + "a" * 20
    campaign_dir = tmp_path / "campaigns" / campaign_id
    gold_dir = campaign_dir / "gold"
    gold_dir.mkdir(parents=True)
    document_id = "document-1"
    artifact_sha = "a" * 64
    source_sha = "b" * 64
    (campaign_dir / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "items": [
                    {
                        "document_id": document_id,
                        "chronological_rank": 1,
                        "artifact_sha256": artifact_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (gold_dir / "index.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "document_id": document_id,
                        "chronological_rank": 1,
                        "disposition": "eligible_known",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus_root = tmp_path / "corpora"
    corpus_dir = corpus_root / ("acoustic-corpus-" + "c" * 24)
    corpus_dir.mkdir(parents=True)
    corpus_manifest = corpus_dir / "manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "document_id": document_id,
                        "recording_id": "recording-1",
                        "conversation_id": "conversation-1",
                        "source_blob": {"sha256": source_sha},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus_manifest.chmod(0o600)
    corpus_root.chmod(0o700)
    corpus_dir.chmod(0o700)

    store_root = tmp_path / "store"
    forbidden_blob = tmp_path / "raw.wav"
    forbidden_blob.write_bytes(b"must not be opened")
    connection = transcript_store.connect(store_root)
    transcript_store.init_db(connection)
    timestamp = "2026-07-31T00:00:00Z"
    try:
        connection.execute(
            """
            INSERT INTO documents (
                id, kind, title, source_path, stored_path, artifact_sha256,
                generated_at, text_content, json_payload, metadata_json,
                embedding_json, created_at, updated_at
            ) VALUES (?, 'transcript', '', '', '', ?, ?, '', '{}', '{}', '{}', ?, ?)
            """,
            (document_id, artifact_sha, timestamp, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO blobs (
                id, role, original_path, stored_path, sha256, mime_type, bytes,
                created_at, updated_at
            ) VALUES ('blob-1', 'source_recording', '', ?, ?, 'audio/wav', 18, ?, ?)
            """,
            (str(forbidden_blob), source_sha, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO document_blobs (document_id, blob_id, role, created_at)
            VALUES (?, 'blob-1', 'source_recording', ?)
            """,
            (document_id, timestamp),
        )
        connection.commit()
    finally:
        connection.close()

    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == forbidden_blob:
            raise AssertionError("source blob was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = assess_successor_readiness(
        campaign_id,
        campaign_root=tmp_path / "campaigns",
        store_root=store_root,
        corpus_root=corpus_root,
        output_root=tmp_path / "output",
    )
    receipt_path = Path(result["receipt_path"])
    assert result["counts"]["fully_disjoint_candidates"] == 0
    assert result["source_hash_evidence"] == (
        "transcript_store_metadata_not_blob_rehashed"
    )
    assert result["will_read_audio"] is False
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(receipt_path.parent.stat().st_mode) == 0o700
