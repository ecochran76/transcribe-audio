from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import speaker_identity_plan0071_d2_predictions_attempt2 as attempt2


def test_private_store_paths_are_bounded_under_expected_roots(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    transcript = source_root / "source.transcript.json"
    media = source_root / "source.m4a"
    transcript.write_text(json.dumps({"utterances": []}), encoding="utf-8")
    media.write_bytes(b"media")
    database = source_root / "transcripts.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                stored_path TEXT NOT NULL
            );
            CREATE TABLE blobs (sha256 TEXT PRIMARY KEY, stored_path TEXT NOT NULL);
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?)",
            ("document-1", str(transcript), str(transcript)),
        )
        connection.execute(
            "INSERT INTO blobs VALUES (?, ?)", ("media-hash", str(media))
        )
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    destination = run / "store" / "transcripts.sqlite3"
    selected = [
        {
            "document_id": "document-1",
            "transcript_artifact": {"path": str(transcript)},
            "source_media_artifact": {"path": str(media)},
            "transcript_sha256": attempt2.sha256_file(transcript),
            "source_media_sha256": "media-hash",
        }
    ]

    attempt2._prepare_private_store(database, destination, selected)

    with sqlite3.connect(destination) as connection:
        row = connection.execute(
            "SELECT source_path, stored_path FROM documents WHERE id = ?",
            ("document-1",),
        ).fetchone()
        blob_path = Path(
            connection.execute(
                "SELECT stored_path FROM blobs WHERE sha256 = ?", ("media-hash",)
            ).fetchone()[0]
        )
    source_path, stored_path = map(Path, row)
    assert source_path != stored_path
    assert source_path.is_relative_to(destination.parent / "private-source")
    assert stored_path.is_relative_to(destination.parent / "artifacts")
    assert blob_path.is_relative_to(destination.parent / "blobs")
    source_path.write_text("private normalization", encoding="utf-8")
    assert stored_path.read_bytes() == transcript.read_bytes()
    assert json.loads(transcript.read_text(encoding="utf-8")) == {"utterances": []}


def test_attempt2_binds_exact_fail_safe_predecessor() -> None:
    assert attempt2.PRIOR_RECEIPT_CONTENT_SHA256 == (
        "94458b21dceabab024f7deed59544d1d0c696bbbddb2b7d94dfa05b6a61ca217"
    )
    assert attempt2.PRIOR_MANIFEST_CONTENT_SHA256 == (
        "020fe3d077cb545a6500ca2c9fca4e759a159d7911a3d2c58e7b9ae342304ed0"
    )
    assert all(value == 0 for value in attempt2.MUTATION_EFFECT_COUNTS.values())
