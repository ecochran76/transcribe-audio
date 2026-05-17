from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
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
                "legacy_import": {
                    "needs_enrichment": True,
                    "source_path": str(artifact_path),
                    "source_sha256": "test-transcript-sha",
                },
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


def test_static_frontend_serves_index_and_assets(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text('<div id="root"></div><script src="/assets/app.js"></script>', encoding="utf-8")
    assets = static_root / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ok')", encoding="utf-8")
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
        static_dir=static_root,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        index = urlopen(f"http://{host}:{port}/", timeout=5)
        asset = urlopen(f"http://{host}:{port}/assets/app.js", timeout=5)
        fallback = urlopen(f"http://{host}:{port}/library/deep-link", timeout=5)
        assert index.status == 200
        assert b'<div id="root">' in index.read()
        assert asset.headers["Content-Type"].startswith("text/javascript") or asset.headers["Content-Type"].startswith(
            "application/javascript"
        )
        assert b"console.log" in asset.read()
        assert b'<div id="root">' in fallback.read()
        try:
            urlopen(f"http://{host}:{port}/api/missing", timeout=5)
        except HTTPError as exc:
            assert exc.code == 404
            assert json.loads(exc.read())["error"] == "Not found"
        else:
            raise AssertionError("Unknown API routes must not fall through to the SPA")
    finally:
        server.shutdown()
        server.server_close()


def test_review_queue_summary_reads_local_state(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    route_path = tmp_path / "meeting.route.json"
    route_path.write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "label": "SoyLei Tempo Chemical matter",
                    "target_kind": "matter",
                    "confidence": 0.62,
                }
            }
        ),
        encoding="utf-8",
    )
    review_dir = state_root / "review-queue"
    review_dir.mkdir(parents=True)
    (review_dir / "route-a.route-review.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T21:00:00Z",
                "reason": "Route confidence below threshold.",
                "route_decision_path": str(route_path),
                "selected_label": "SoyLei Tempo Chemical matter",
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "route-b.route-review.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T21:01:00Z",
                "reason": "Route confidence below threshold.",
                "route_decision_path": str(tmp_path / "missing.route.json"),
                "selected_label": "Missing route",
            }
        ),
        encoding="utf-8",
    )
    conflict_dir = state_root / "filename-conflict-reviews"
    conflict_dir.mkdir()
    (conflict_dir / "filename-conflict-review-20260516-153723.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T20:37:23Z",
                "items": [
                    {"id": "one", "decision": "keep_target"},
                    {"id": "two", "decision": "pending"},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = transcript_api.review_queue_summary(state_root=state_root, store_root=store_root, limit=20)

    route_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "route_reviews")
    conflict_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "filename_conflicts")
    summary_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "first_pass_summaries")
    assert route_bucket["count"] == 1
    assert route_bucket["stale_count"] == 1
    assert conflict_bucket["count"] == 1
    assert summary_bucket["label"] == "First-pass summaries"
    assert conflict_bucket["decisions"] == {"keep_target": 1, "pending": 1}
    assert {item["status"] for item in payload["items"]} == {"pending", "stale_reference"}


def test_prepare_first_pass_summary_endpoint_writes_dry_run_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=http://127.0.0.1:18095/v1",
                "OPENAI_API_KEY=test-key",
                "AURACALL_BATCH_URL=http://127.0.0.1:18095/v1/response-batches",
            ]
        ),
        encoding="utf-8",
    )
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=state_root,
        batch_env_file=env_file,
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/prepare",
            data=json.dumps({"limit": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urlopen(request, timeout=5)
        payload = json.loads(response.read())
        manifest = json.loads(Path(payload["manifest"]).read_text(encoding="utf-8"))

        assert response.status == 201
        assert payload["action"] == "prepare_first_pass_summary_batch"
        assert payload["bucket"] == "first_pass_summaries"
        assert payload["request_count"] == 1
        assert payload["dry_run"] is True
        assert payload["batch_id"] is None
        assert payload["workflow"] == "transcribe-audio-first-pass-summary"
        assert payload["artifact_file"] == "first_pass_readout.json"
        assert manifest["batch"] is None
        assert manifest["request_count"] == 1
        assert manifest["batch_payload"]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"
    finally:
        server.shutdown()
        server.server_close()
