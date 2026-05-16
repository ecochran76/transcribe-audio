#!/usr/bin/env python3
"""
Read-only local HTTP API for the transcript review console.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from transcript_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_STORE_DIR,
    TranscriptStoreError,
    connect,
    context_for_document,
    db_path,
    init_db,
    legacy_enrichment_queue,
    parse_object_json,
    search_store,
    store_dir,
)

DEFAULT_API_PORT = 18876
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")


def document_summary(row: sqlite3.Row) -> dict[str, Any]:
    metadata = parse_object_json(row["metadata_json"])
    media_blob = metadata.get("media_blob") if isinstance(metadata.get("media_blob"), dict) else {}
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "source_path": row["source_path"],
        "stored_path": row["stored_path"],
        "generated_at": row["generated_at"],
        "updated_at": row["updated_at"],
        "embedding_provider": row["embedding_provider"],
        "embedding_model": row["embedding_model"],
        "metadata": metadata,
        "media_blob": media_blob,
    }


def list_documents(
    *,
    root: Optional[Path] = None,
    kind: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        where = ""
        params: list[Any] = []
        if kind:
            where = "WHERE kind = ?"
            params.append(kind)
        total = int(con.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0])
        rows = con.execute(
            f"""
            SELECT * FROM documents
            {where}
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
    return {
        "items": [document_summary(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


def get_document(document_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        blobs = con.execute(
            """
            SELECT blobs.id, document_blobs.role, blobs.mime_type, blobs.bytes, blobs.sha256
            FROM document_blobs
            JOIN blobs ON document_blobs.blob_id = blobs.id
            WHERE document_blobs.document_id = ?
            ORDER BY document_blobs.role, blobs.id
            """,
            (document_id,),
        ).fetchall()
    payload = parse_object_json(row["json_payload"])
    summary = document_summary(row)
    summary.update(
        {
            "json_payload": payload,
            "text_content": row["text_content"],
            "blobs": [
                {
                    "id": blob["id"],
                    "role": blob["role"],
                    "mime_type": blob["mime_type"],
                    "bytes": blob["bytes"],
                    "sha256": blob["sha256"],
                    "playback_url": f"/api/blobs/{blob['id']}",
                    "download_url": f"/api/blobs/{blob['id']}?download=1",
                }
                for blob in blobs
            ],
        }
    )
    return summary


def get_blob(blob_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM blobs WHERE id = ?", (blob_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No blob found with id {blob_id}")
    path = Path(row["stored_path"])
    if not path.exists() or not path.is_file():
        raise TranscriptStoreError(f"Blob file is missing for id {blob_id}")
    return {
        "id": row["id"],
        "role": row["role"],
        "path": path,
        "mime_type": row["mime_type"] or "application/octet-stream",
        "bytes": int(row["bytes"]),
        "sha256": row["sha256"],
    }


def parse_int(value: str, default: int, *, minimum: int = 0, maximum: int = 500) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key) or []
    return values[0] if values else default


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def review_status(count: int, *, stale_count: int = 0, pending_count: int = 0) -> str:
    if count <= 0 and pending_count <= 0:
        return "clear"
    if pending_count > 0:
        return "pending"
    if stale_count >= count:
        return "stale"
    if stale_count > 0:
        return "mixed"
    return "pending"


def filename_conflict_summary(state_root: Path) -> dict[str, Any]:
    review_dir = state_root / "filename-conflict-reviews"
    review_files = sorted(review_dir.glob("filename-conflict-review-*.json"), key=lambda path: path.stat().st_mtime)
    review_files = [path for path in review_files if "audit" not in path.name]
    if not review_files:
        return {
            "id": "filename_conflicts",
            "label": "Filename conflicts",
            "count": 0,
            "status": "clear",
            "detail": "No filename-conflict review file found.",
            "path": "",
            "decisions": {},
            "total_count": 0,
        }
    path = review_files[-1]
    payload = read_json_file(path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    decisions: dict[str, int] = {}
    pending_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "pending")
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision in {"pending", "needs_investigation"}:
            pending_count += 1
    decision_text = ", ".join(f"{key}: {value}" for key, value in sorted(decisions.items())) or "no items"
    return {
        "id": "filename_conflicts",
        "label": "Filename conflicts",
        "count": pending_count,
        "status": "pending" if pending_count else "clear",
        "detail": f"Latest review has {decision_text}.",
        "path": str(path),
        "decisions": decisions,
        "total_count": len(items),
        "updated_at": payload.get("updated_at") or payload.get("created_at") or "",
    }


def route_review_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    review_dir = state_root / "review-queue"
    paths = sorted(review_dir.glob("*.route-review.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in paths[:limit]:
        payload = read_json_file(path)
        route_path = Path(str(payload.get("route_decision_path") or "")).expanduser()
        route_exists = bool(str(route_path)) and route_path.exists()
        route_payload = read_json_file(route_path) if route_exists else {}
        selected = route_payload.get("selected_candidate") if isinstance(route_payload.get("selected_candidate"), dict) else {}
        item = {
            "id": path.stem.removesuffix(".route-review"),
            "bucket": "route_reviews",
            "type": "route_review",
            "label": payload.get("selected_label") or selected.get("label") or "Unselected route",
            "reason": payload.get("reason") or "",
            "created_at": payload.get("created_at") or "",
            "review_path": str(path),
            "route_decision_path": str(route_path) if str(route_path) else "",
            "route_decision_exists": route_exists,
            "status": "pending" if route_exists else "stale_reference",
            "confidence": selected.get("confidence"),
            "target_kind": selected.get("target_kind") or "",
        }
        items.append(item)
    return items


def review_queue_summary(*, state_root: Optional[Path] = None, store_root: Optional[Path] = None, limit: int = 50) -> dict[str, Any]:
    runtime_state_root = (state_root or DEFAULT_STATE_DIR).expanduser()
    route_items = route_review_items(runtime_state_root, limit=limit)
    stale_count = sum(1 for item in route_items if not item["route_decision_exists"])
    actionable_count = len(route_items) - stale_count
    filename_bucket = filename_conflict_summary(runtime_state_root)
    legacy_payload = legacy_enrichment_queue(root=store_root, pending_only=True)
    legacy_count = int(legacy_payload.get("selected_count") or 0)
    route_bucket = {
        "id": "route_reviews",
        "label": "Route reviews",
        "count": actionable_count,
        "status": review_status(len(route_items), stale_count=stale_count),
        "detail": f"{actionable_count} actionable, {stale_count} stale local references.",
        "total_count": len(route_items),
        "stale_count": stale_count,
    }
    legacy_bucket = {
        "id": "first_pass_summaries",
        "label": "First-pass summaries",
        "count": legacy_count,
        "status": "pending" if legacy_count else "clear",
        "detail": "Stored transcripts waiting for first-pass summaries.",
        "duplicate_count": legacy_payload.get("duplicate_count") or 0,
        "sample_items": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "generated_at": item.get("generated_at"),
                "has_media_blob": item.get("has_media_blob"),
            }
            for item in legacy_payload.get("items", [])[:5]
            if isinstance(item, dict)
        ],
    }
    buckets = [
        route_bucket,
        filename_bucket,
        legacy_bucket,
        {
            "id": "memory_harvest",
            "label": "Memory harvest",
            "count": 0,
            "status": "gated",
            "detail": "Requires explicit review file approval before live Graphiti writes.",
        },
        {
            "id": "speaker_ids",
            "label": "Speaker IDs",
            "count": 0,
            "status": "planned",
            "detail": "Contact dedupe and speaker assignment tables are planned in P09.",
        },
    ]
    return {
        "state_dir": str(runtime_state_root),
        "store_dir": str(store_dir(store_root)),
        "limit": limit,
        "buckets": buckets,
        "items": route_items,
        "total_open": sum(int(bucket.get("count") or 0) for bucket in buckets),
    }


def parse_range_header(header: str, size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes="):
        return None
    value = header.removeprefix("bytes=").split(",", 1)[0].strip()
    if not value or "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    if start_text == "":
        suffix = parse_int(end_text, 0, minimum=0, maximum=size)
        if suffix <= 0:
            return None
        return max(size - suffix, 0), size - 1
    start = parse_int(start_text, 0, minimum=0, maximum=max(size - 1, 0))
    end = parse_int(end_text, size - 1, minimum=start, maximum=max(size - 1, 0)) if end_text else size - 1
    if start > end:
        return None
    return start, end


class TranscriptApiHandler(BaseHTTPRequestHandler):
    server_version = "TranscriptApi/0.1"

    @property
    def store_root(self) -> Path:
        return self.server.store_root  # type: ignore[attr-defined]

    @property
    def embedding_provider(self) -> str:
        return self.server.embedding_provider  # type: ignore[attr-defined]

    @property
    def embedding_model(self) -> str:
        return self.server.embedding_model  # type: ignore[attr-defined]

    @property
    def state_root(self) -> Path:
        return self.server.state_root  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self.write_json({"status": "ok", "store_dir": str(self.store_root), "db_path": str(db_path(self.store_root))})
                return
            if parsed.path == "/api/library":
                self.write_json(
                    list_documents(
                        root=self.store_root,
                        kind=first(params, "kind"),
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                        offset=parse_int(first(params, "offset"), 0, minimum=0, maximum=100000),
                    )
                )
                return
            if parsed.path == "/api/review-queue":
                self.write_json(
                    review_queue_summary(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                    )
                )
                return
            if parsed.path == "/api/search":
                query = first(params, "q") or first(params, "query")
                if not query:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required query parameter: q")
                    return
                self.write_json(
                    {
                        "query": query,
                        "results": search_store(
                            query,
                            root=self.store_root,
                            kind=first(params, "kind"),
                            limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        ),
                    }
                )
                return
            if parsed.path.startswith("/api/documents/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_json(get_document(parts[2], root=self.store_root))
                    return
                if len(parts) == 4 and parts[3] == "context":
                    chunk_text = first(params, "chunk_index")
                    self.write_json(
                        context_for_document(
                            parts[2],
                            root=self.store_root,
                            chunk_index=int(chunk_text) if chunk_text else None,
                            context_chunks=parse_int(first(params, "context_chunks"), 1, minimum=0, maximum=10),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        )
                    )
                    return
            if parsed.path.startswith("/api/blobs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_blob(parts[2], download=first(params, "download") in {"1", "true", "yes"})
                    return
            if parsed.path.startswith("/api/"):
                self.write_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if self.write_static(parsed.path):
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except TranscriptStoreError as exc:
            self.write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, status: HTTPStatus, message: str) -> None:
        self.write_json({"error": message, "status": status.value}, status=status)

    def write_static(self, request_path: str) -> bool:
        static_dir = self.server.static_dir  # type: ignore[attr-defined]
        if static_dir is None:
            return False
        static_root = Path(static_dir)
        if not static_root.exists() or not static_root.is_dir():
            return False
        relative = unquote(request_path).lstrip("/")
        target = static_root / relative if relative else static_root / "index.html"
        if not target.exists() or target.is_dir():
            target = static_root / "index.html"
        try:
            resolved = target.resolve()
            resolved.relative_to(static_root.resolve())
        except ValueError:
            self.write_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return True
        if not resolved.exists() or not resolved.is_file():
            return False
        body = resolved.read_bytes()
        mime_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-store" if resolved.name == "index.html" else "public, max-age=3600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def write_blob(self, blob_id: str, *, download: bool = False) -> None:
        blob = get_blob(blob_id, root=self.store_root)
        size = int(blob["bytes"])
        file_range = parse_range_header(self.headers.get("Range", ""), size)
        status = HTTPStatus.PARTIAL_CONTENT if file_range else HTTPStatus.OK
        start, end = file_range if file_range else (0, max(size - 1, 0))
        length = max(end - start + 1, 0)

        self.send_response(status)
        self.send_header("Content-Type", str(blob["mime_type"]))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if file_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{blob_id}"')
        self.end_headers()

        with Path(blob["path"]).open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class TranscriptApiServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store_root: Path,
        embedding_provider: str,
        embedding_model: str,
        state_root: Path = DEFAULT_STATE_DIR,
        quiet: bool = False,
        static_dir: Optional[Path] = DEFAULT_STATIC_DIR,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store_root = store_root
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.state_root = state_root.expanduser()
        self.quiet = quiet
        self.static_dir = static_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local transcript review API.")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def serve(args: argparse.Namespace) -> None:
    root = store_dir(args.store_dir)
    with connect(root) as con:
        init_db(con)
    server = TranscriptApiServer(
        (args.host, args.port),
        TranscriptApiHandler,
        store_root=root,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        state_root=args.state_dir,
        quiet=bool(args.quiet),
        static_dir=None if args.no_static else args.static_dir,
    )
    print(f"Serving transcript API on http://{args.host}:{args.port} using {db_path(root)}")
    server.serve_forever()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        serve(args)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
