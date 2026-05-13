from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import legacy_media_link
import transcript_store


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def legacy_payload(source_path: Path) -> dict:
    return {
        "transcript_title": "Client Call",
        "backend": "legacy-import",
        "recording_start": "2026-05-06T13:15:00-05:00",
        "duration_seconds": 0,
        "transcript_text": "Client call transcript.",
        "legacy_import": {
            "source_path": str(source_path),
            "source_sha256": "abc123",
            "needs_enrichment": True,
        },
    }


def test_plan_media_links_matches_from_media_index(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    source_transcript = tmp_path / "Client Call Transcript.docx"
    artifact = write_json(tmp_path / "client.transcript.json", legacy_payload(source_transcript))
    media = tmp_path / "Client Call.m4a"
    media.write_bytes(b"audio")
    media_index = tmp_path / "media-index.txt"
    media_index.write_text(str(media), encoding="utf-8")
    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")

    planned = legacy_media_link.plan_media_links(root=store_root, media_index_files=[media_index])

    assert len(planned) == 1
    assert planned[0]["status"] == "link"
    assert planned[0]["media_path"] == str(media.resolve())


def test_apply_media_links_updates_artifact_and_registers_blob(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    source_transcript = tmp_path / "Client Call Transcript.docx"
    artifact = write_json(tmp_path / "client.transcript.json", legacy_payload(source_transcript))
    media = tmp_path / "Client Call.wav"
    media.write_bytes(b"audio")
    media_index = tmp_path / "media-index.txt"
    media_index.write_text(str(media), encoding="utf-8")
    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")
    planned = legacy_media_link.plan_media_links(root=store_root, media_index_files=[media_index])

    results = legacy_media_link.apply_media_links(planned, root=store_root, embedding_provider="debug-hash")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert results[0]["result"] == "updated"
    assert payload["source_media_path"] == str(media.resolve())
    with sqlite3.connect(store_root / "transcripts.sqlite3") as con:
        blob_count = con.execute("SELECT COUNT(*) FROM document_blobs").fetchone()[0]
    assert blob_count == 1


def test_cli_media_link_dry_run_reports_status(tmp_path: Path, capsys) -> None:
    store_root = tmp_path / "store"
    source_transcript = tmp_path / "Client Call Transcript.docx"
    artifact = write_json(tmp_path / "client.transcript.json", legacy_payload(source_transcript))
    transcript_store.ingest_artifact(artifact, root=store_root, embedding_provider="debug-hash")

    assert legacy_media_link.main(["--store-dir", str(store_root)]) == 0

    stdout = capsys.readouterr().out
    payload = json.loads(stdout.split(legacy_media_link.LEGACY_MEDIA_LINK_JSON_STDOUT_PREFIX)[0])
    assert payload["dry_run"] is True
    assert payload["by_status"] == {"no_match": 1}
