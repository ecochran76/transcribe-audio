from __future__ import annotations

import json
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_intelligence_ledger
import transcript_api
import transcript_store


class FakeAuraCallHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self.__class__.requests.append(payload)
        self.write_json(
            {
                "id": "batch_test",
                "status": "queued",
                "request_count": len(payload.get("requests") or []),
            },
            HTTPStatus.CREATED,
        )

    def do_GET(self) -> None:
        if self.path.endswith("/response-batches/batch_test"):
            self.write_json(
                {
                    "id": "batch_test",
                    "status": "running",
                    "jobs": [{"index": 0, "status": "running"}],
                }
            )
            return
        self.write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
    app_intelligence_ledger.create_run(
        state_root=state_root,
        workflow="contextual_reread",
        purpose="Review app-server decision.",
        document_id="doc-abc",
        run_id="human-review-run",
    )
    run_payload = app_intelligence_ledger.response_for_run(state_root=state_root, run_id="human-review-run")["run"]
    decision_dir = state_root / "app-intelligence-runs" / "human-review-run" / "artifacts" / "structured-decisions"
    decision_dir.mkdir(parents=True)
    decision_path = decision_dir / "decision-1.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
                "decision_id": "decision-1",
                "run_id": "human-review-run",
                "valid": True,
                "decision": {
                    "action": "ask_for_human_review",
                    "rationale": "Operator should review ambiguous context.",
                    "confidence": 0.7,
                    "review_flags": ["ambiguous_context"],
                },
            }
        ),
        encoding="utf-8",
    )
    run_payload["decisions"] = [
        {
            "decision_id": "decision-1",
            "valid": True,
            "action": "ask_for_human_review",
            "status": "validated",
            "artifact_path": str(decision_path),
            "will_execute_host_action": False,
            "created_at": "2026-05-16T21:02:00Z",
        }
    ]
    (state_root / "app-intelligence-runs" / "human-review-run" / "run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    payload = transcript_api.review_queue_summary(state_root=state_root, store_root=store_root, limit=20)

    route_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "route_reviews")
    app_review_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "app_intelligence_human_review")
    conflict_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "filename_conflicts")
    summary_bucket = next(bucket for bucket in payload["buckets"] if bucket["id"] == "first_pass_summaries")
    assert route_bucket["count"] == 1
    assert route_bucket["stale_count"] == 1
    assert app_review_bucket["count"] == 1
    assert app_review_bucket["pending_apply_count"] == 1
    assert conflict_bucket["count"] == 1
    assert summary_bucket["label"] == "First-pass summaries"
    assert conflict_bucket["decisions"] == {"keep_target": 1, "pending": 1}
    assert {item["status"] for item in payload["items"]} == {"pending", "stale_reference", "pending_apply"}
    assert any(item["bucket"] == "app_intelligence_human_review" for item in payload["items"])


def test_codex_app_server_provider_readiness(monkeypatch) -> None:
    def fake_run(args, check=False, capture_output=True, text=True, timeout=10):
        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 10
        stdout = "codex-cli 0.131.0\n"
        if args[1:] == ["app-server", "--help"]:
            stdout = "Usage: codex app-server [OPTIONS]\n      --listen <URL>\n      --ws-auth <MODE>\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(transcript_api.shutil, "which", lambda name: "/usr/local/bin/codex")
    monkeypatch.setattr(transcript_api.subprocess, "run", fake_run)

    provider = transcript_api.codex_app_server_readiness()

    assert provider["id"] == "codex-app-server"
    assert provider["status"] == "ready"
    assert provider["ready"] is True
    assert provider["recommended_transport"] == "stdio"
    assert provider["capabilities"]["persistent_sessions"] is True
    assert provider["capabilities"]["remote_transport"] is True
    assert provider["capabilities"]["websocket_auth"] is True


def test_intelligence_providers_endpoint_includes_app_server(tmp_path: Path, monkeypatch) -> None:
    def fake_run(args, check=False, capture_output=True, text=True, timeout=10):
        stdout = "codex-cli 0.131.0\n"
        if args[1:] == ["app-server", "--help"]:
            stdout = "Usage: codex app-server [OPTIONS]\n      --listen <URL>\n      --ws-auth <MODE>\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(transcript_api.shutil, "which", lambda name: "/usr/local/bin/codex")
    monkeypatch.setattr(transcript_api.subprocess, "run", fake_run)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/providers", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    providers = {provider["id"]: provider for provider in payload["providers"]}
    assert payload["default_supervisor"] == "codex-app-server"
    assert providers["codex-app-server"]["status"] == "ready"
    assert providers["codex-app-server"]["control_plane"] == "codex-app-server"


def test_intelligence_config_endpoint_returns_task_routing(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert payload["schema_version"] == "transcribe-audio.intelligence-config.v1"
    assert payload["tasks"]["first_pass_summary"]["provider"] == "openai-compatible"
    assert payload["tasks"]["app_supervisor"]["provider"] == "codex-app-server"


def test_intelligence_config_preview_and_apply_endpoints(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "intelligence.config.json"
    monkeypatch.setenv("TRANSCRIPTS_INTELLIGENCE_CONFIG", str(config_path))
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        preview_request = Request(
            f"http://{host}:{port}/api/intelligence/config/preview",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec", "model": "gpt-preview"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preview = json.loads(urlopen(preview_request, timeout=5).read())
        assert preview["will_write"] is False
        assert not config_path.exists()

        blocked_request = Request(
            f"http://{host}:{port}/api/intelligence/config/apply",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("apply without approval token must fail")

        apply_request = Request(
            f"http://{host}:{port}/api/intelligence/config/apply",
            data=json.dumps(
                {
                    "task": "first_pass_summary",
                    "update": {"provider": "codex-exec", "model": "gpt-applied"},
                    "approval_token": "APPLY_INTELLIGENCE_CONFIG_UPDATE",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        applied_response = urlopen(apply_request, timeout=5)
        applied = json.loads(applied_response.read())
        config = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/config", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert applied_response.status == 202
    assert applied["applied"] is True
    assert config["tasks"]["first_pass_summary"]["provider"] == "codex-exec"
    assert config["tasks"]["first_pass_summary"]["model"] == "gpt-applied"


def test_app_intelligence_run_prepare_and_read_endpoints(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "context-reread",
                    "purpose": "Prepare a supervised contextual reread.",
                    "document_id": "doc_123",
                    "run_id": "test-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        prepared_response = urlopen(request, timeout=5)
        prepared = json.loads(prepared_response.read())
        listing = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs", timeout=5).read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/test-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert prepared_response.status == 201
    assert prepared["run"]["run_id"] == "test-run"
    assert prepared["run"]["phase"] == "prepared"
    assert prepared["run"]["policy"]["host_owns_control_flow"] is True
    assert listing["total"] == 1
    assert shown["run"]["document_id"] == "doc_123"
    assert shown["events"][0]["event_type"] == "run_prepared"


def test_app_intelligence_session_start_preflight_endpoint_does_not_start_work(tmp_path: Path) -> None:
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin=sys.executable,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "app-supervisor",
                    "purpose": "Prepare supervised work.",
                    "run_id": "preflight-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()

        preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/preflight-run/session-start-preflight",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        preflight_response = urlopen(preflight_request, timeout=5)
        preflight = json.loads(preflight_response.read())

        append_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/preflight-run/session-start-preflight",
            data=json.dumps(
                {
                    "approval_token": "APPEND_SESSION_START_PREFLIGHT_EVENT",
                    "append_event": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        append_response = urlopen(append_request, timeout=5)
        appended = json.loads(append_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/preflight-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert preflight_response.status == 200
    assert preflight["will_start_session"] is False
    assert preflight["checks"]["phase_prepared"] is True
    assert preflight["checks"]["approval_token_shape"] is True
    assert preflight["checks"]["provider_ready"] is False
    assert "provider_ready" in preflight["blocking_checks"]
    assert append_response.status == 202
    assert appended["event"]["event_type"] == "session_start_preflight"
    assert shown["events"][-1]["event_type"] == "session_start_preflight"


def test_app_intelligence_session_start_endpoint_starts_daemon_without_model_turn(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        transcript_api,
        "codex_app_server_readiness",
        lambda codex_bin="codex": {"ready": True, "status": "ready"},
    )

    def fake_run_codex_command(args: list[str], *, timeout: int = 30) -> dict:
        commands.append(args)
        if args[-1] == "start":
            return {"args": args, "ok": True, "returncode": 0, "stdout": "started\n", "stderr": ""}
        return {"args": args, "ok": True, "returncode": 0, "stdout": '{"running": true}\n', "stderr": ""}

    monkeypatch.setattr(transcript_api, "run_codex_command", fake_run_codex_command)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin="/usr/local/bin/codex",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "app-supervisor",
                    "purpose": "Prepare supervised work.",
                    "run_id": "session-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()

        start_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/session-run/session-start",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION", "transport": "stdio"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start_response = urlopen(start_request, timeout=5)
        started = json.loads(start_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/session-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert start_response.status == 202
    assert started["ok"] is True
    assert started["will_start_model_turn"] is False
    assert commands == [
        ["/usr/local/bin/codex", "app-server", "daemon", "start"],
        ["/usr/local/bin/codex", "app-server", "daemon", "version"],
    ]
    assert shown["run"]["phase"] == "session_started"
    assert shown["run"]["state"]["active_codex_thread_id"] is None
    assert [event["event_type"] for event in shown["events"][-2:]] == [
        "app_server_session_start_requested",
        "app_server_session_started",
    ]


def test_app_intelligence_model_turn_preflight_endpoint_writes_prompt_packet(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []
    store_root = tmp_path / "store"
    ingest = transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )

    monkeypatch.setattr(
        transcript_api,
        "codex_app_server_readiness",
        lambda codex_bin="codex": {"ready": True, "status": "ready"},
    )

    def fake_run_codex_command(args: list[str], *, timeout: int = 30) -> dict:
        commands.append(args)
        return {"args": args, "ok": True, "returncode": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(transcript_api, "run_codex_command", fake_run_codex_command)

    def fake_start_model_turn(**kwargs: object) -> dict:
        assert kwargs["codex_bin"] == "/usr/local/bin/codex"
        assert "Tempo Chemical samples" in str(kwargs["prompt_text"])
        assert kwargs["existing_thread_id"] == ""
        return {
            "ok": True,
            "thread_id": "thread_test",
            "turn_id": "turn_test",
            "thread_start_response": {"thread": {"id": "thread_test"}},
            "turn_start_response": {"turn": {"id": "turn_test"}},
            "events": [{"method": "turn/started", "params": {"turn": {"id": "turn_test"}}}],
        }

    monkeypatch.setattr(transcript_api.codex_app_server_client, "start_model_turn", fake_start_model_turn)

    def fake_inspect_model_turn(**kwargs: object) -> dict:
        assert kwargs["codex_bin"] == "/usr/local/bin/codex"
        assert kwargs["thread_id"] == "thread_test"
        assert kwargs["turn_id"] == "turn_test"
        return {
            "ok": True,
            "thread_id": "thread_test",
            "turn_id": "turn_test",
            "status": "completed",
            "completed": True,
            "output_text": json.dumps(
                {
                    "action": "ask_for_human_review",
                    "rationale": "Tempo readout requires operator review.",
                    "confidence": 0.8,
                    "review_flags": ["operator_review"],
                    "recommended_next_prompt": "Validate context sources.",
                }
            ),
            "thread_read_response": {"thread": {"id": "thread_test"}},
            "turns_list_response": {"data": [{"id": "turn_test", "status": "completed"}]},
            "items_list_response": {"data": []},
            "events": [{"method": "turn/completed", "params": {"turn": {"id": "turn_test"}}}],
        }

    monkeypatch.setattr(transcript_api.codex_app_server_client, "inspect_model_turn", fake_inspect_model_turn)

    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        state_root=tmp_path / "state",
        quiet=True,
        codex_bin="/usr/local/bin/codex",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        prepare_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/prepare",
            data=json.dumps(
                {
                    "workflow": "contextual_reread",
                    "purpose": "Prepare supervised work.",
                    "document_id": ingest.id,
                    "run_id": "packet-run",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(prepare_request, timeout=5).read()
        start_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/session-start",
            data=json.dumps({"approval_token": "START_APP_SERVER_SESSION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(start_request, timeout=5).read()

        packet_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/model-turn-preflight",
            data=json.dumps(
                {
                    "approval_token": "PREPARE_MODEL_TURN_PREFLIGHT",
                    "task": "contextual_reread",
                    "document_id": ingest.id,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        packet_response = urlopen(packet_request, timeout=5)
        packet = json.loads(packet_response.read())
        review = json.loads(
            urlopen(
                f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}",
                timeout=5,
            ).read()
        )
        send_preflight_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}/send-preflight",
            data=json.dumps({"approval_token": "SEND_APP_SERVER_MODEL_TURN"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        send_preflight_response = urlopen(send_preflight_request, timeout=5)
        send_preflight = json.loads(send_preflight_response.read())
        send_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/prompt-packets/{packet['packet']['packet_id']}/send",
            data=json.dumps({"approval_token": "SEND_APP_SERVER_MODEL_TURN"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        send_response = urlopen(send_request, timeout=5)
        send = json.loads(send_response.read())
        status_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/turn-status",
            data=json.dumps({"approval_token": "CAPTURE_MODEL_TURN_STATUS"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status_response = urlopen(status_request, timeout=5)
        status = json.loads(status_response.read())
        decision_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/structured-decision/validate",
            data=json.dumps({"approval_token": "VALIDATE_STRUCTURED_DECISION"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        decision_response = urlopen(decision_request, timeout=5)
        decision = json.loads(decision_response.read())
        apply_request = Request(
            f"http://{host}:{port}/api/intelligence/runs/packet-run/structured-decisions/{decision['decision_id']}/apply",
            data=json.dumps({"approval_token": "APPLY_STRUCTURED_DECISION", "reviewer": "api-test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        apply_response = urlopen(apply_request, timeout=5)
        applied = json.loads(apply_response.read())
        shown = json.loads(urlopen(f"http://{host}:{port}/api/intelligence/runs/packet-run", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()

    assert packet_response.status == 201
    assert packet["will_send_prompt"] is False
    assert Path(packet["packet_path"]).exists()
    assert Path(packet["prompt_path"]).exists()
    assert packet["packet"]["document"]["id"] == ingest.id
    assert packet["packet"]["route"]["task"] == "contextual_reread"
    assert "Tempo Chemical samples" in Path(packet["prompt_path"]).read_text(encoding="utf-8")
    assert review["will_send_prompt"] is False
    assert review["future_required_approval_token_for_send"] == "SEND_APP_SERVER_MODEL_TURN"
    assert "Tempo Chemical samples" in review["prompt_text"]
    assert send_preflight_response.status == 200
    assert send_preflight["ok"] is True
    assert send_preflight["will_send_prompt"] is False
    assert send_preflight["will_write_event"] is False
    assert send_preflight["prompt_char_count"] > 0
    assert send_response.status == 202
    assert send["ok"] is True
    assert send["codex_thread_id"] == "thread_test"
    assert send["codex_turn_id"] == "turn_test"
    assert send["will_execute_downstream_action"] is False
    assert shown["run"]["prompt_packets"][0]["sent"] is True
    assert status_response.status == 202
    assert status["completed"] is True
    assert status["will_execute_structured_decision"] is False
    assert status["codex_thread_id"] == "thread_test"
    assert status["codex_turn_id"] == "turn_test"
    assert Path(status["artifact_path"]).exists()
    assert decision_response.status == 202
    assert decision["valid"] is True
    assert decision["decision"]["action"] == "ask_for_human_review"
    assert decision["will_execute_host_action"] is False
    assert apply_response.status == 202
    assert applied["decision_action"] == "ask_for_human_review"
    assert applied["applied_ledger_state"] is True
    assert applied["will_execute_external_action"] is False
    assert applied["will_execute_write_bearing_action"] is False
    assert shown["run"]["phase"] == "human_review_requested"
    assert shown["run"]["status"] == "needs_human_review"
    assert shown["run"]["state"]["active_codex_thread_id"] == "thread_test"
    assert shown["run"]["decisions"][0]["status"] == "applied"
    assert shown["events"][-1]["event_type"] == "structured_decision_applied"
    assert shown["codex_events_count"] == 2


def test_batch_status_counts_prefers_provider_aggregate_counts() -> None:
    assert transcript_api.batch_status_counts(
        {
            "counts": {"total": 5, "in_progress": 5, "completed": 0},
            "jobs": [{"status": "in_progress"} for _ in range(5)],
        }
    ) == {"total": 5, "in_progress": 5, "completed": 0}


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
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
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
        assert manifest["dispatch_team"] == "transcribe-audio-chatgpt-pro-pool"
        assert manifest["model"] == "gpt-5.2-pro"
        assert manifest["batch_payload"]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"
        assert manifest["batch_payload"]["dispatch"]["team"] == "transcribe-audio-chatgpt-pro-pool"
    finally:
        server.shutdown()
        server.server_close()


def test_first_pass_summary_submit_and_status_use_prepared_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    state_root = tmp_path / "state"
    env_file = tmp_path / "auracall.env"
    transcript_store.ingest_artifact(
        write_transcript_artifact(tmp_path),
        root=store_root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    FakeAuraCallHandler.requests = []
    provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeAuraCallHandler)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    host, provider_port = provider.server_address
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_BASE_URL=http://{host}:{provider_port}/v1",
                "OPENAI_API_KEY=test-key",
                f"AURACALL_BATCH_URL=http://{host}:{provider_port}/v1/response-batches",
                "AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool",
                "AURACALL_DISPATCH_MODEL=gpt-5.2-pro",
            ]
        ),
        encoding="utf-8",
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
        prepare_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/prepare",
            data=json.dumps({"limit": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        prepared = json.loads(urlopen(prepare_request, timeout=5).read())
        manifest_path = Path(prepared["manifest"])

        blocked_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/submit",
            data=json.dumps({"manifest": prepared["manifest"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(blocked_request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 400
            assert "approval_token" in json.loads(exc.read())["error"]
        else:
            raise AssertionError("Submit without approval token must fail")

        submit_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/submit",
            data=json.dumps(
                {
                    "manifest": prepared["manifest"],
                    "approval_token": "SUBMIT_FIRST_PASS_SUMMARY_BATCH",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        submitted_response = urlopen(submit_request, timeout=5)
        submitted = json.loads(submitted_response.read())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert submitted_response.status == 202
        assert submitted["status"] == "submitted"
        assert submitted["batch_id"] == "batch_test"
        assert submitted["dry_run"] is False
        assert manifest["batch"]["id"] == "batch_test"
        assert manifest["dry_run"] is False
        assert FakeAuraCallHandler.requests[0]["dispatch"]["team"] == "transcribe-audio-chatgpt-pro-pool"
        assert FakeAuraCallHandler.requests[0]["metadata"]["workflow"] == "transcribe-audio-first-pass-summary"

        status_request = Request(
            f"http://{host}:{port}/api/review-queue/first-pass-summaries/status",
            data=json.dumps({"manifest": prepared["manifest"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status_payload = json.loads(urlopen(status_request, timeout=5).read())

        assert status_payload["status"] == "running"
        assert status_payload["batch_id"] == "batch_test"
        assert status_payload["batch_counts"] == {"running": 1}
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["last_status"]["status"] == "running"
    finally:
        server.shutdown()
        server.server_close()
        provider.shutdown()
        provider.server_close()
