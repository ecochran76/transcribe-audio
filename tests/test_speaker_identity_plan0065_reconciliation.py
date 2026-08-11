from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import speaker_identity_plan0065_reconciliation as reconciliation


def _legacy_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _mutated_payload(legacy: dict) -> dict:
    return {
        **legacy,
        "schema_version": 2,
        "conversation_id": "10000000-0000-4000-8000-000000000001",
        "recording_id": "20000000-0000-4000-8000-000000000002",
    }


def test_reconstructs_only_the_identity_container_backfill() -> None:
    legacy = {
        "schema_version": 1,
        "transcript_text": "hello",
        "utterances": [{"speaker": "A", "text": "hello"}],
    }
    expected = _legacy_bytes(legacy)
    mutated = _legacy_bytes(_mutated_payload(legacy))

    restored = reconciliation.reconstruct_legacy_transcript_bytes(
        mutated,
        expected_sha256=hashlib.sha256(expected).hexdigest(),
    )

    assert restored == expected


def test_reconstruction_rejects_any_additional_semantic_drift() -> None:
    legacy = {"schema_version": 1, "transcript_text": "hello"}
    expected = _legacy_bytes(legacy)
    mutated = _mutated_payload({**legacy, "transcript_text": "changed"})

    with pytest.raises(reconciliation.Plan0065ReconciliationError):
        reconciliation.reconstruct_legacy_transcript_bytes(
            _legacy_bytes(mutated),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
        )


def test_reconciles_source_stored_and_database_with_private_backups(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema_version": 1,
        "transcript_text": "hello",
        "utterances": [{"speaker": "A", "text": "hello"}],
    }
    expected = _legacy_bytes(legacy)
    expected_sha = hashlib.sha256(expected).hexdigest()
    mutated_payload = _mutated_payload(legacy)
    mutated = _legacy_bytes(mutated_payload)
    mutated_sha = hashlib.sha256(mutated).hexdigest()
    source = tmp_path / "source.transcript.json"
    stored = tmp_path / "stored.transcript.json"
    source.write_bytes(mutated)
    stored.write_bytes(mutated)
    database = tmp_path / "transcripts.sqlite3"
    with sqlite3.connect(database) as con:
        con.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                json_payload TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "document-1",
                str(source),
                str(stored),
                mutated_sha,
                json.dumps(mutated_payload, separators=(",", ":"), sort_keys=True),
                "{}",
                "before",
            ),
        )
    backup_dir = tmp_path / "private-backup"

    result = reconciliation.reconcile_targets(
        targets=(
            reconciliation.RestorationTarget(
                document_id="document-1",
                expected_sha256=expected_sha,
                stored_path=stored,
            ),
        ),
        database_path=database,
        backup_dir=backup_dir,
        restored_at="2026-08-11T16:00:00Z",
    )

    assert source.read_bytes() == expected
    assert stored.read_bytes() == expected
    assert result["restored_document_count"] == 1
    assert result["restored_artifact_copy_count"] == 2
    assert len(list(backup_dir.glob("*.transcript.json"))) == 2
    with sqlite3.connect(database) as con:
        row = con.execute(
            "SELECT artifact_sha256, json_payload, updated_at FROM documents"
        ).fetchone()
    assert row == (
        expected_sha,
        json.dumps(legacy, separators=(",", ":"), sort_keys=True),
        "2026-08-11T16:00:00Z",
    )
