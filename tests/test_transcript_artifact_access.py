from __future__ import annotations

import json
from pathlib import Path

import pytest

import transcript_artifact_access
import transcript_store


def write_transcript(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transcript_title": "Archived interview",
                "transcript_text": "Speaker A: Hello.\nSpeaker B: Welcome.",
                "utterances": [
                    {"speaker": "Speaker A", "start": 0, "end": 1, "text": "Hello."},
                    {"speaker": "Speaker B", "start": 1, "end": 2, "text": "Welcome."},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def document_row(store_root: Path, document_id: str) -> dict[str, object]:
    with transcript_store.connect(store_root) as con:
        row = con.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def test_resolver_uses_hash_verified_stored_copy_when_source_is_missing(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = write_transcript(tmp_path / "archived.transcript.json")
    result = transcript_store.ingest_artifact(
        source,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    source.unlink()

    resolved = transcript_artifact_access.resolve_transcript_artifact(
        document_row(store_root, result.id),
        store_root=store_root,
    )

    assert resolved.location == "stored"
    assert resolved.path == Path(result.stored_path).resolve()
    assert resolved.actual_sha256 == resolved.expected_sha256


def test_resolver_rejects_a_tampered_stored_copy(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    source = write_transcript(tmp_path / "archived.transcript.json")
    result = transcript_store.ingest_artifact(
        source,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    source.unlink()
    Path(result.stored_path).write_text('{"tampered": true}', encoding="utf-8")

    with pytest.raises(
        transcript_artifact_access.TranscriptArtifactAccessError,
        match="hash does not match",
    ):
        transcript_artifact_access.resolve_transcript_artifact(
            document_row(store_root, result.id),
            store_root=store_root,
        )


def test_resolver_rejects_a_stored_path_outside_the_store(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    outside = write_transcript(tmp_path / "outside.transcript.json")
    document = {
        "id": "doc-outside",
        "kind": "transcript",
        "source_path": str(tmp_path / "missing.transcript.json"),
        "stored_path": str(outside),
        "artifact_sha256": transcript_store.sha256_file(outside),
    }

    with pytest.raises(
        transcript_artifact_access.TranscriptArtifactAccessError,
        match="outside the transcript store",
    ):
        transcript_artifact_access.resolve_transcript_artifact(
            document,
            store_root=store_root,
        )


def test_identity_backfill_on_source_synchronizes_stored_copy_and_index(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = write_transcript(tmp_path / "current.transcript.json")
    result = transcript_store.ingest_artifact(
        source,
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload, resolved = (
        transcript_artifact_access.ensure_resolved_transcript_identity(
            document_row(store_root, result.id),
            store_root=store_root,
        )
    )

    stored_path = Path(result.stored_path)
    stored_payload = json.loads(stored_path.read_text(encoding="utf-8"))
    indexed = document_row(store_root, result.id)
    indexed_payload = json.loads(str(indexed["json_payload"]))
    assert resolved.location == "source"
    assert payload["conversation_id"] == stored_payload["conversation_id"]
    assert payload["recording_id"] == stored_payload["recording_id"]
    assert indexed_payload["conversation_id"] == payload["conversation_id"]
    assert indexed_payload["recording_id"] == payload["recording_id"]
    assert indexed["artifact_sha256"] == transcript_store.sha256_file(source)
    assert indexed["artifact_sha256"] == transcript_store.sha256_file(stored_path)
