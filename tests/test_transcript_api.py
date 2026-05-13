from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transcript_api
import transcript_store


def write_transcript_artifact(tmp_path: Path) -> Path:
    media_path = tmp_path / "meeting.m4a"
    media_path.write_bytes(b"0123456789abcdef")
    artifact_path = tmp_path / "meeting.transcript.json"
    artifact_path.write_text(
        json.dumps(
            {
                "transcript_title": "Weekly Product Sync",
                "transcript_text": "Speaker A [0.00s - 1.00s]: We discussed Tempo Chemical samples.",
                "source_media_path": str(media_path),
                "working_media_path": str(media_path),
                "backend": "test",
                "duration_seconds": 16,
                "utterances": [
                    {
                        "speaker": "Speaker A",
                        "start": 0,
                        "end": 1000,
                        "text": "We discussed Tempo Chemical samples.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def test_ingest_registers_media_blob_for_api(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    payload = transcript_api.get_document(result.id, root=store_root)

    assert payload["title"] == "Weekly Product Sync"
    assert payload["media_blob"]["id"]
    assert payload["media_blob"]["playback_url"].startswith("/api/blobs/")
    assert payload["blobs"][0]["bytes"] == 16
    assert Path(payload["metadata"]["media_blob"]["id"])


def test_library_and_search_use_store_documents(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    library = transcript_api.list_documents(root=store_root, limit=10)
    search = transcript_store.search_store(
        "Tempo Chemical",
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    assert library["total"] == 1
    assert library["items"][0]["media_blob"]["id"]
    assert search[0]["title"] == "Weekly Product Sync"


def test_blob_route_supports_range_reads(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    result = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    blob_id = transcript_api.get_document(result.id, root=store_root)["blobs"][0]["id"]
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(f"http://{host}:{port}/api/blobs/{blob_id}", headers={"Range": "bytes=2-5"})
        response = urlopen(request, timeout=5)
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 2-5/16"
        assert response.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()
