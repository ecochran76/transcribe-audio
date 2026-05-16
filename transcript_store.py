#!/usr/bin/env python3
"""
User-scoped transcript/readout store with lexical and semantic search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_STORE_DIR = Path("~/.transcripts")
DEFAULT_DB_NAME = "transcripts.sqlite3"
STORE_JSON_STDOUT_PREFIX = "TRANSCRIPT_STORE_JSON="
SEARCH_JSON_STDOUT_PREFIX = "TRANSCRIPT_SEARCH_JSON="
BACKFILL_JSON_STDOUT_PREFIX = "TRANSCRIPT_BACKFILL_JSON="
CONTEXT_JSON_STDOUT_PREFIX = "TRANSCRIPT_CONTEXT_JSON="
LEGACY_ENRICHMENT_QUEUE_JSON_STDOUT_PREFIX = "TRANSCRIPT_LEGACY_ENRICHMENT_QUEUE_JSON="
FIRST_PASS_SUMMARY_QUEUE_JSON_STDOUT_PREFIX = "TRANSCRIPT_FIRST_PASS_SUMMARY_QUEUE_JSON="
EMBEDDING_DIM = 128
EMBEDDING_MAX_CHARS = 1500
EMBEDDING_CHUNK_OVERLAP_CHARS = 200
DEFAULT_EMBEDDING_PROVIDER = "ollama"
DEFAULT_EMBEDDING_MODEL = "ollama/nomic-embed-text"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BACKFILL_PATTERNS = ("*.transcript.json", "*.readout.json", "*.contextual.readout.json")
DEFAULT_BACKFILL_EXCLUDE_PATTERNS = ("*/transcripts-store-*/*", "*/store/artifacts/*")


class TranscriptStoreError(RuntimeError):
    pass


def utcish_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def store_dir(path: Optional[Path] = None) -> Path:
    return (path or DEFAULT_STORE_DIR).expanduser()


def db_path(path: Optional[Path] = None) -> Path:
    root = store_dir(path)
    return root / DEFAULT_DB_NAME


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text.lower())


def debug_hash_embedding_for_text(text: str, *, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    summed = [0.0] * dimensions
    for vector in vectors:
        if len(vector) != dimensions:
            raise TranscriptStoreError("Embedding chunks returned inconsistent vector dimensions.")
        for index, value in enumerate(vector):
            summed[index] += value
    return normalize_vector([value / len(vectors) for value in summed])


def chunk_text_for_embedding(
    text: str,
    *,
    max_chars: int = EMBEDDING_MAX_CHARS,
    overlap_chars: int = EMBEDDING_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped] if stripped else [""]

    chunks: list[str] = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            candidates = [
                stripped.rfind("\n\n", start, end),
                stripped.rfind(". ", start, end),
                stripped.rfind(" ", start, end),
            ]
            boundary = max(candidates)
            if boundary > start + (max_chars // 2):
                end = boundary + 1
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks or [stripped[:max_chars]]


def chunk_text_spans_for_embedding(
    text: str,
    *,
    max_chars: int = EMBEDDING_MAX_CHARS,
    overlap_chars: int = EMBEDDING_CHUNK_OVERLAP_CHARS,
) -> list[tuple[str, int, int]]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [(stripped, 0, len(stripped))] if stripped else [("", 0, 0)]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            candidates = [
                stripped.rfind("\n\n", start, end),
                stripped.rfind(". ", start, end),
                stripped.rfind(" ", start, end),
            ]
            boundary = max(candidates)
            if boundary > start + (max_chars // 2):
                end = boundary + 1
        raw_chunk = stripped[start:end]
        leading = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing = len(raw_chunk.rstrip())
        chunk = raw_chunk.strip()
        if chunk:
            chunks.append((chunk, start + leading, start + trailing))
        if end >= length:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks or [(stripped[:max_chars], 0, min(max_chars, len(stripped)))]


def ollama_model_name(model_name: str) -> str:
    return model_name.removeprefix("ollama/")


def format_embedding_input(text: str, *, provider: str, model_name: str, purpose: str) -> str:
    if provider != "ollama":
        return text
    model = ollama_model_name(model_name).lower()
    if model.startswith("nomic-embed-text"):
        prefix = "search_query: " if purpose == "query" else "search_document: "
        if text.startswith(("search_query: ", "search_document: ")):
            return text
        return f"{prefix}{text}"
    return text


def ollama_embedding_for_text(
    text: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 60.0,
) -> list[float]:
    request = Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=json.dumps({"model": ollama_model_name(model_name), "input": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise TranscriptStoreError(f"Ollama embedding failed ({exc.code}): {detail or exc.reason}") from exc
    except (URLError, TimeoutError) as exc:
        raise TranscriptStoreError(f"Ollama embedding endpoint unavailable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptStoreError("Ollama embedding response was not valid JSON.") from exc

    vector: Any = payload.get("embedding")
    if not isinstance(vector, list) and isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
        vector = payload["embeddings"][0]
    if not isinstance(vector, list) or not vector:
        raise TranscriptStoreError("Ollama embedding response did not include an embedding vector.")
    return normalize_vector([float(value) for value in vector])


def openai_compatible_embedding_for_text(
    text: str,
    *,
    model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    base_url: str = "",
    api_key: str = "",
    timeout_seconds: float = 60.0,
) -> list[float]:
    import os

    resolved_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
    if not resolved_api_key:
        raise TranscriptStoreError("OpenAI-compatible embeddings require OPENAI_API_KEY.")
    request = Request(
        f"{resolved_base_url}/embeddings",
        data=json.dumps({"model": model_name, "input": text}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {resolved_api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise TranscriptStoreError(f"OpenAI-compatible embedding failed ({exc.code}): {detail or exc.reason}") from exc
    except (URLError, TimeoutError) as exc:
        raise TranscriptStoreError(f"OpenAI-compatible embedding endpoint unavailable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptStoreError("OpenAI-compatible embedding response was not valid JSON.") from exc
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise TranscriptStoreError("OpenAI-compatible embedding response did not include data.")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or not vector:
        raise TranscriptStoreError("OpenAI-compatible embedding response did not include an embedding vector.")
    return normalize_vector([float(value) for value in vector])


def embedding_for_text(
    text: str,
    *,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    purpose: str = "document",
) -> list[float]:
    if provider == "sentence-transformers":
        raise TranscriptStoreError("sentence-transformers provider was removed; use ollama or openai-compatible.")
    if provider in {"hash", "debug-hash"}:
        return debug_hash_embedding_for_text(text)

    chunks = chunk_text_for_embedding(text) if purpose == "document" else [text]
    vectors: list[list[float]] = []
    for chunk in chunks:
        formatted = format_embedding_input(chunk, provider=provider, model_name=model_name, purpose=purpose)
        if provider == "ollama":
            vectors.append(ollama_embedding_for_text(formatted, model_name=model_name))
        elif provider == "openai-compatible":
            vectors.append(openai_compatible_embedding_for_text(formatted, model_name=model_name))
        else:
            raise TranscriptStoreError(f"Unsupported embedding provider: {provider}")
    return average_vectors(vectors)


def utterance_seconds(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return numeric / 1000.0 if numeric > 100.0 else numeric


def formatted_utterance_text(utterance: dict[str, Any]) -> str:
    speaker = utterance.get("speaker") or "Speaker"
    start = utterance_seconds(utterance.get("start"))
    end = utterance_seconds(utterance.get("end"))
    text = normalize_string(utterance.get("text"))
    return f"{speaker} [{start:0.2f}s - {end:0.2f}s]: {text}"


def transcript_utterance_spans(payload: dict[str, Any], text_content: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    utterances = payload.get("utterances") if isinstance(payload.get("utterances"), list) else []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        formatted = formatted_utterance_text(utterance)
        start_char = text_content.find(formatted, cursor)
        end_char = start_char + len(formatted) if start_char >= 0 else -1
        if start_char < 0:
            raw_text = normalize_string(utterance.get("text"))
            start_char = text_content.find(raw_text, cursor) if raw_text else -1
            end_char = start_char + len(raw_text) if start_char >= 0 else -1
        if start_char < 0:
            continue
        cursor = max(cursor, end_char)
        spans.append(
            {
                "start_char": start_char,
                "end_char": end_char,
                "speaker": normalize_string(utterance.get("speaker")) or "Speaker",
                "start_seconds": utterance_seconds(utterance.get("start")),
                "end_seconds": utterance_seconds(utterance.get("end")),
            }
        )
    return spans


def chunk_metadata(
    kind: str,
    payload: dict[str, Any],
    text_content: str,
    start_char: int,
    end_char: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"char_start": start_char, "char_end": end_char}
    if kind != "transcript":
        return metadata

    overlapping = [
        span
        for span in transcript_utterance_spans(payload, text_content)
        if span["end_char"] > start_char and span["start_char"] < end_char
    ]
    if not overlapping:
        return metadata

    speakers = sorted({span["speaker"] for span in overlapping if span.get("speaker")})
    metadata.update(
        {
            "start_seconds": min(span["start_seconds"] for span in overlapping),
            "end_seconds": max(span["end_seconds"] for span in overlapping),
            "speakers": speakers,
            "utterance_count": len(overlapping),
        }
    )
    return metadata


def embeddings_for_text_chunks(
    text: str,
    *,
    kind: str = "",
    payload: Optional[dict[str, Any]] = None,
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[tuple[int, str, list[float], dict[str, Any]]]:
    chunks = chunk_text_spans_for_embedding(text)
    return [
        (
            index,
            chunk,
            embedding_for_text(chunk, provider=provider, model_name=model_name, purpose="document"),
            chunk_metadata(kind, payload or {}, text, start, end),
        )
        for index, (chunk, start, end) in enumerate(chunks)
    ]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptStoreError(f"Failed to read artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptStoreError(f"Artifact {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptStoreError(f"Artifact {path} must contain a JSON object.")
    return payload


def artifact_kind(path: Path, payload: dict[str, Any]) -> str:
    name = path.name
    if name.endswith(".transcript.json") or "transcript_text" in payload:
        return "transcript"
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    contextualization = payload.get("contextualization") if isinstance(payload.get("contextualization"), dict) else {}
    if name.endswith(".contextual.readout.json") or provider.get("mode") == "contextual_reread" or contextualization:
        return "contextual_readout"
    if name.endswith(".readout.json") or "summary" in payload:
        return "readout"
    return "artifact"


def readout_text(payload: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("title", "summary"):
        values.append(normalize_string(payload.get(key)))
    for key in ("topics", "key_decisions", "unresolved_questions", "risks", "next_steps"):
        value = payload.get(key)
        if isinstance(value, list):
            values.extend(normalize_string(item) for item in value)
    for key in ("participants", "action_items", "matter_candidates", "memory_candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    values.extend(normalize_string(part) for part in item.values())
    contextualization = payload.get("contextualization")
    if isinstance(contextualization, dict):
        for source in contextualization.get("supporting_context_sources") or []:
            if isinstance(source, dict):
                values.extend(
                    normalize_string(source.get(key))
                    for key in ("source_type", "source_id", "label", "snippet", "uri")
                )
    return "\n".join(value for value in values if value)


def transcript_text(payload: dict[str, Any]) -> str:
    values = [
        normalize_string(payload.get("transcript_title")),
        normalize_string(payload.get("transcript_text")),
    ]
    event = payload.get("event")
    if isinstance(event, dict):
        values.extend(
            normalize_string(event.get(key))
            for key in ("summary", "description", "provider", "calendar_id")
        )
        values.extend(normalize_string(item) for item in event.get("participants") or [])
        for item in event.get("matching_calendars") or []:
            if isinstance(item, dict):
                values.extend(
                    normalize_string(item.get(key))
                    for key in ("calendar_summary", "event_summary", "calendar_id", "event_id")
                )
    return "\n".join(value for value in values if value)


def artifact_text(kind: str, payload: dict[str, Any]) -> str:
    if kind == "transcript":
        return transcript_text(payload)
    if kind in {"readout", "contextual_readout"}:
        return readout_text(payload)
    return json_dumps(payload)


def artifact_title(kind: str, path: Path, payload: dict[str, Any]) -> str:
    if kind == "transcript":
        return normalize_string(payload.get("transcript_title")) or path.stem
    return normalize_string(payload.get("title")) or path.stem


def artifact_time(payload: dict[str, Any]) -> str:
    for key in ("generated_at", "recording_start", "created_at"):
        value = normalize_string(payload.get(key))
        if value:
            return value
    return ""


def metadata_for_artifact(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"kind": kind}
    if kind == "transcript":
        metadata.update(
            {
                "backend": payload.get("backend"),
                "duration_seconds": payload.get("duration_seconds"),
                "recording_start": payload.get("recording_start"),
                "recording_end": payload.get("recording_end"),
                "event": payload.get("event"),
                "output_paths": payload.get("output_paths"),
            }
        )
    else:
        metadata.update(
            {
                "provider": payload.get("provider"),
                "source_artifact_path": payload.get("source_artifact_path"),
                "contextualization": payload.get("contextualization"),
            }
        )
    return metadata


def artifact_store_path(root: Path, doc_id: str, source_path: Path) -> Path:
    return root / "artifacts" / doc_id[:2] / f"{doc_id}-{source_path.name}"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    root = store_dir(path)
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path(root))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            text_content TEXT NOT NULL,
            json_payload TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_provider TEXT NOT NULL DEFAULT 'ollama',
            embedding_model TEXT NOT NULL DEFAULT 'ollama/nomic-embed-text',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_kind_source ON documents(kind, source_path);
        CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind);
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            embedding_json TEXT NOT NULL,
            embedding_provider TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(document_id, chunk_index, embedding_provider, embedding_model)
        );
        CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks(embedding_provider, embedding_model);
        CREATE TABLE IF NOT EXISTS blobs (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            original_path TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_blobs_sha256 ON blobs(sha256);
        CREATE TABLE IF NOT EXISTS document_blobs (
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            blob_id TEXT NOT NULL REFERENCES blobs(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(document_id, blob_id, role)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id UNINDEXED,
            title,
            text_content
        );
        """
    )
    existing_columns = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in con.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "embedding_provider" not in existing_columns:
        con.execute("ALTER TABLE documents ADD COLUMN embedding_provider TEXT NOT NULL DEFAULT 'hash'")
    if "embedding_model" not in existing_columns:
        con.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'local-token-hash-v1'")
    chunk_columns = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in con.execute("PRAGMA table_info(document_chunks)").fetchall()
    }
    if "metadata_json" not in chunk_columns:
        con.execute("ALTER TABLE document_chunks ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
    con.commit()


def media_path_for_payload(payload: dict[str, Any]) -> str:
    return normalize_string(payload.get("working_media_path")) or normalize_string(payload.get("source_media_path"))


def blob_store_path(root: Path, blob_id: str, source_path: Path) -> Path:
    return root / "blobs" / blob_id[:2] / f"{blob_id}{source_path.suffix}"


def prepare_blob(root: Path, source_path_text: str, *, role: str = "source_recording") -> dict[str, Any]:
    if not source_path_text:
        return {}
    source_path = Path(source_path_text).expanduser()
    if not source_path.exists() or not source_path.is_file():
        return {}
    resolved = source_path.resolve()
    artifact_hash = sha256_file(resolved)
    blob_id = stable_id(role, artifact_hash)
    stored_path = blob_store_path(root, blob_id, resolved)
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    if not stored_path.exists() or sha256_file(stored_path) != artifact_hash:
        shutil.copy2(resolved, stored_path)
    return {
        "id": blob_id,
        "role": role,
        "original_path": str(resolved),
        "stored_path": str(stored_path),
        "sha256": artifact_hash,
        "mime_type": mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
        "bytes": stored_path.stat().st_size,
    }


def upsert_document_blob(con: sqlite3.Connection, document_id: str, blob: dict[str, Any], *, now: str) -> None:
    if not blob:
        return
    con.execute(
        """
        INSERT INTO blobs (id, role, original_path, stored_path, sha256, mime_type, bytes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            role=excluded.role,
            original_path=excluded.original_path,
            stored_path=excluded.stored_path,
            sha256=excluded.sha256,
            mime_type=excluded.mime_type,
            bytes=excluded.bytes,
            updated_at=excluded.updated_at
        """,
        (
            blob["id"],
            blob["role"],
            blob["original_path"],
            blob["stored_path"],
            blob["sha256"],
            blob["mime_type"],
            int(blob["bytes"]),
            now,
            now,
        ),
    )
    con.execute(
        """
        INSERT OR IGNORE INTO document_blobs (document_id, blob_id, role, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (document_id, blob["id"], blob["role"], now),
    )


def rebuild_fts_for_document(con: sqlite3.Connection, doc_id: str, title: str, text_content: str) -> None:
    con.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
    con.execute("INSERT INTO documents_fts(doc_id, title, text_content) VALUES (?, ?, ?)", (doc_id, title, text_content))


def rebuild_chunks_for_document(
    con: sqlite3.Connection,
    doc_id: str,
    chunks: list[tuple[int, str, list[float], dict[str, Any]]],
    *,
    embedding_provider: str,
    embedding_model: str,
    now: str,
) -> None:
    con.execute(
        "DELETE FROM document_chunks WHERE document_id = ? AND embedding_provider = ? AND embedding_model = ?",
        (doc_id, embedding_provider, embedding_model),
    )
    for chunk_index, text_content, embedding, metadata in chunks:
        chunk_id = stable_id(doc_id, str(chunk_index), embedding_provider, embedding_model)
        con.execute(
            """
            INSERT INTO document_chunks (
                id, document_id, chunk_index, text_content, metadata_json, embedding_json,
                embedding_provider, embedding_model, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                doc_id,
                chunk_index,
                text_content,
                json_dumps(metadata),
                json_dumps(embedding),
                embedding_provider,
                embedding_model,
                now,
                now,
            ),
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class IngestResult:
    id: str
    kind: str
    title: str
    source_path: str
    stored_path: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "source_path": self.source_path,
            "stored_path": self.stored_path,
            "status": self.status,
        }


def iter_backfill_candidates(
    roots: Iterable[Path],
    *,
    recursive: bool = False,
    patterns: Iterable[str] = DEFAULT_BACKFILL_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_BACKFILL_EXCLUDE_PATTERNS,
    modified_within_days: Optional[float] = None,
) -> list[Path]:
    import fnmatch
    import time

    normalized_patterns = tuple(patterns) or DEFAULT_BACKFILL_PATTERNS
    normalized_excludes = tuple(exclude_patterns)
    min_mtime = None
    if modified_within_days is not None:
        min_mtime = time.time() - (modified_within_days * 86400.0)

    candidates: dict[str, Path] = {}
    for root in roots:
        expanded = root.expanduser()
        if expanded.is_file():
            paths = [expanded]
        elif expanded.is_dir():
            iterator = expanded.rglob("*") if recursive else expanded.glob("*")
            paths = [path for path in iterator if path.is_file()]
        else:
            continue
        for path in paths:
            path_text = str(path)
            if any(fnmatch.fnmatch(path_text, pattern) for pattern in normalized_excludes):
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in normalized_patterns):
                continue
            try:
                if min_mtime is not None and path.stat().st_mtime < min_mtime:
                    continue
                resolved = path.resolve()
            except OSError:
                continue
            candidates[str(resolved)] = resolved
    return [candidates[key] for key in sorted(candidates)]


def existing_artifact_status(
    path: Path,
    *,
    root: Optional[Path] = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    source_path = path.expanduser().resolve()
    payload = load_json(source_path)
    raw = source_path.read_bytes()
    artifact_hash = sha256_bytes(raw)
    kind = artifact_kind(source_path, payload)
    title = artifact_title(kind, source_path, payload)

    with connect(root) as con:
        init_db(con)
        row = con.execute(
            """
            SELECT id, artifact_sha256, embedding_provider, embedding_model
            FROM documents
            WHERE kind = ? AND source_path = ?
            """,
            (kind, str(source_path)),
        ).fetchone()
        chunk_count = 0
        metadata_chunk_count = 0
        if row:
            chunk_count = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM document_chunks
                    WHERE document_id = ? AND embedding_provider = ? AND embedding_model = ?
                    """,
                    (row["id"], embedding_provider, embedding_model),
                ).fetchone()[0]
            )
            metadata_chunk_count = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM document_chunks
                    WHERE document_id = ? AND embedding_provider = ? AND embedding_model = ?
                      AND metadata_json != '{}'
                    """,
                    (row["id"], embedding_provider, embedding_model),
                ).fetchone()[0]
            )

    if not row:
        status = "insert"
    elif (
        row["artifact_sha256"] == artifact_hash
        and row["embedding_provider"] == embedding_provider
        and row["embedding_model"] == embedding_model
        and chunk_count > 0
        and (kind != "transcript" or metadata_chunk_count > 0)
    ):
        status = "skip"
    else:
        status = "update"

    return {
        "kind": kind,
        "title": title,
        "source_path": str(source_path),
        "artifact_sha256": artifact_hash,
        "status": status,
    }


def summarize_backfill_status(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for item in items:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        by_status[item["status"]] = by_status.get(item["status"], 0) + 1
    return {"by_kind": by_kind, "by_status": by_status}


def plan_backfill(
    roots: Iterable[Path],
    *,
    root: Optional[Path] = None,
    recursive: bool = False,
    patterns: Iterable[str] = DEFAULT_BACKFILL_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_BACKFILL_EXCLUDE_PATTERNS,
    modified_within_days: Optional[float] = None,
    kinds: Optional[set[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    candidates = iter_backfill_candidates(
        roots,
        recursive=recursive,
        patterns=patterns,
        exclude_patterns=exclude_patterns,
        modified_within_days=modified_within_days,
    )
    planned: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            status = existing_artifact_status(
                candidate,
                root=root,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
        except (OSError, TranscriptStoreError) as exc:
            status = {
                "kind": "artifact",
                "title": candidate.name,
                "source_path": str(candidate),
                "artifact_sha256": "",
                "status": "error",
                "error": str(exc),
            }
        if kinds and status["kind"] not in kinds:
            continue
        if force and status["status"] == "skip":
            status["status"] = "update"
        planned.append(status)
        if limit is not None and len(planned) >= limit:
            break
    return planned


def ingest_artifact(
    path: Path,
    *,
    root: Optional[Path] = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> IngestResult:
    store_root = store_dir(root)
    source_path = path.expanduser().resolve()
    payload = load_json(source_path)
    raw = source_path.read_bytes()
    artifact_hash = sha256_bytes(raw)
    kind = artifact_kind(source_path, payload)
    title = artifact_title(kind, source_path, payload)
    text_content = artifact_text(kind, payload)
    generated_at = artifact_time(payload)
    doc_id = stable_id(kind, str(source_path), artifact_hash)
    stored_path = artifact_store_path(store_root, doc_id, source_path)
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    if not stored_path.exists() or sha256_bytes(stored_path.read_bytes()) != artifact_hash:
        shutil.copy2(source_path, stored_path)
    metadata = metadata_for_artifact(kind, payload)
    media_blob = prepare_blob(store_root, media_path_for_payload(payload)) if kind == "transcript" else {}
    if media_blob:
        metadata["media_blob"] = {
            "id": media_blob["id"],
            "role": media_blob["role"],
            "mime_type": media_blob["mime_type"],
            "bytes": media_blob["bytes"],
            "playback_url": f"/api/blobs/{media_blob['id']}",
            "download_url": f"/api/blobs/{media_blob['id']}?download=1",
        }
    now = utcish_now()
    chunk_embeddings = embeddings_for_text_chunks(
        text_content,
        kind=kind,
        payload=payload,
        provider=embedding_provider,
        model_name=embedding_model,
    )
    embedding = average_vectors([vector for _, _, vector, _ in chunk_embeddings])

    with connect(store_root) as con:
        init_db(con)
        existing = con.execute(
            "SELECT id FROM documents WHERE kind = ? AND source_path = ?",
            (kind, str(source_path)),
        ).fetchone()
        status = "updated" if existing else "inserted"
        if existing:
            doc_id = str(existing["id"])
        con.execute(
            """
            INSERT INTO documents (
                id, kind, title, source_path, stored_path, artifact_sha256, generated_at,
                text_content, json_payload, metadata_json, embedding_json, embedding_provider,
                embedding_model, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                source_path=excluded.source_path,
                stored_path=excluded.stored_path,
                artifact_sha256=excluded.artifact_sha256,
                generated_at=excluded.generated_at,
                text_content=excluded.text_content,
                json_payload=excluded.json_payload,
                metadata_json=excluded.metadata_json,
                embedding_json=excluded.embedding_json,
                embedding_provider=excluded.embedding_provider,
                embedding_model=excluded.embedding_model,
                updated_at=excluded.updated_at
            """,
            (
                doc_id,
                kind,
                title,
                str(source_path),
                str(stored_path),
                artifact_hash,
                generated_at,
                text_content,
                json_dumps(payload),
                json_dumps(metadata),
                json_dumps(embedding),
                embedding_provider,
                embedding_model,
                now,
                now,
            ),
        )
        rebuild_fts_for_document(con, doc_id, title, text_content)
        rebuild_chunks_for_document(
            con,
            doc_id,
            chunk_embeddings,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            now=now,
        )
        upsert_document_blob(con, doc_id, media_blob, now=now)
        con.commit()
    return IngestResult(doc_id, kind, title, str(source_path), str(stored_path), status)


def fts_query(query: str) -> str:
    terms = [term for term in tokens(query) if len(term) >= 2]
    return " OR ".join(f"{term}*" for term in terms) or query


def snippet(text: str, query: str, *, length: int = 240) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in tokens(query)]
    positions = [pos for pos in positions if pos >= 0]
    start = max(min(positions) - 60, 0) if positions else 0
    value = re.sub(r"\s+", " ", text[start : start + length]).strip()
    return value


def lexical_scores(con: sqlite3.Connection, query: str, kind: str = "", limit: int = 50) -> dict[str, float]:
    match = fts_query(query)
    where = "documents_fts MATCH ?"
    params: list[Any] = [match]
    if kind:
        where += " AND documents.kind = ?"
        params.append(kind)
    try:
        rows = con.execute(
            f"""
            SELECT documents.id, bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents ON documents_fts.doc_id = documents.id
            WHERE {where}
            ORDER BY rank
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    scores: dict[str, float] = {}
    for row in rows:
        # bm25 is lower-is-better and often negative in SQLite FTS5.
        scores[str(row["id"])] = 1.0 / (1.0 + max(float(row["rank"]), 0.0))
    return scores


def semantic_scores(
    con: sqlite3.Connection,
    query: str,
    kind: str = "",
    *,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, float]:
    query_embedding = embedding_for_text(
        query,
        provider=embedding_provider,
        model_name=embedding_model,
        purpose="query",
    )
    sql = "SELECT id, embedding_json FROM documents WHERE embedding_provider = ? AND embedding_model = ?"
    params: list[Any] = [embedding_provider, embedding_model]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    rows = con.execute(sql, params).fetchall()
    scores: dict[str, float] = {}
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"])
        except json.JSONDecodeError:
            continue
        if isinstance(embedding, list):
            scores[str(row["id"])] = cosine(query_embedding, [float(item) for item in embedding])
    return scores


def chunk_semantic_matches(
    con: sqlite3.Connection,
    query: str,
    kind: str = "",
    *,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, dict[str, Any]]:
    query_embedding = embedding_for_text(
        query,
        provider=embedding_provider,
        model_name=embedding_model,
        purpose="query",
    )
    sql = """
        SELECT
            document_chunks.document_id,
            document_chunks.chunk_index,
            document_chunks.text_content,
            document_chunks.metadata_json,
            document_chunks.embedding_json
        FROM document_chunks
        JOIN documents ON document_chunks.document_id = documents.id
        WHERE document_chunks.embedding_provider = ?
          AND document_chunks.embedding_model = ?
    """
    params: list[Any] = [embedding_provider, embedding_model]
    if kind:
        sql += " AND documents.kind = ?"
        params.append(kind)

    matches: dict[str, dict[str, Any]] = {}
    for row in con.execute(sql, params).fetchall():
        try:
            embedding = json.loads(row["embedding_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(embedding, list):
            continue
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        score = cosine(query_embedding, [float(item) for item in embedding])
        doc_id = str(row["document_id"])
        current = matches.get(doc_id)
        if current is None or score > current["score"]:
            matches[doc_id] = {
                "score": score,
                "chunk_index": int(row["chunk_index"]),
                "text": str(row["text_content"]),
                "metadata": metadata,
            }
    return matches


def search_store(
    query: str,
    *,
    root: Optional[Path] = None,
    kind: str = "",
    limit: int = 10,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    with connect(root) as con:
        init_db(con)
        lexical = lexical_scores(con, query, kind=kind, limit=max(limit * 5, 20))
        semantic = semantic_scores(
            con,
            query,
            kind=kind,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        chunk_matches = chunk_semantic_matches(
            con,
            query,
            kind=kind,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        ids = set(lexical) | set(semantic) | set(chunk_matches)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = con.execute(f"SELECT * FROM documents WHERE id IN ({placeholders})", tuple(ids)).fetchall()
    results = []
    for row in rows:
        lexical_score = lexical.get(str(row["id"]), 0.0)
        semantic_score = semantic.get(str(row["id"]), 0.0)
        chunk_match = chunk_matches.get(str(row["id"]))
        chunk_score = float(chunk_match["score"]) if chunk_match else 0.0
        combined = (0.55 * lexical_score) + (0.20 * max(semantic_score, 0.0)) + (0.25 * max(chunk_score, 0.0))
        best_chunk = None
        if chunk_match:
            chunk_metadata_value = chunk_match.get("metadata") if isinstance(chunk_match.get("metadata"), dict) else {}
            best_chunk = {
                "chunk_index": chunk_match["chunk_index"],
                "semantic_score": chunk_score,
                "snippet": snippet(chunk_match["text"], query),
                "metadata": chunk_metadata_value,
            }
            for key in ("start_seconds", "end_seconds", "speakers", "utterance_count"):
                if key in chunk_metadata_value:
                    best_chunk[key] = chunk_metadata_value[key]
        results.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "source_path": row["source_path"],
                "stored_path": row["stored_path"],
                "generated_at": row["generated_at"],
                "score": combined,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "chunk_semantic_score": chunk_score,
                "embedding_provider": row["embedding_provider"],
                "embedding_model": row["embedding_model"],
                "best_chunk": best_chunk,
                "snippet": snippet(row["text_content"], query),
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def parse_object_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def chunk_payload(row: sqlite3.Row) -> dict[str, Any]:
    metadata = parse_object_json(row["metadata_json"])
    payload = {
        "chunk_index": int(row["chunk_index"]),
        "text": str(row["text_content"]),
        "metadata": metadata,
    }
    for key in ("start_seconds", "end_seconds", "speakers", "utterance_count", "char_start", "char_end"):
        if key in metadata:
            payload[key] = metadata[key]
    return payload


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes:02d}:{remainder:05.2f}"


def context_for_document(
    document_id: str,
    *,
    root: Optional[Path] = None,
    chunk_index: Optional[int] = None,
    context_chunks: int = 1,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    normalized_context_chunks = max(context_chunks, 0)
    with connect(root) as con:
        init_db(con)
        document = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if document is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")

        params = (document_id, embedding_provider, embedding_model)
        if chunk_index is None:
            selected = con.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = ? AND embedding_provider = ? AND embedding_model = ?
                ORDER BY
                    CASE WHEN metadata_json LIKE '%start_seconds%' THEN 0 ELSE 1 END,
                    chunk_index
                LIMIT 1
                """,
                params,
            ).fetchone()
        else:
            selected = con.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = ? AND embedding_provider = ? AND embedding_model = ? AND chunk_index = ?
                """,
                (*params, chunk_index),
            ).fetchone()
        if selected is None:
            raise TranscriptStoreError(
                "No chunk found for document "
                f"{document_id} with provider={embedding_provider} model={embedding_model}"
            )

        selected_index = int(selected["chunk_index"])
        nearby_rows = con.execute(
            """
            SELECT * FROM document_chunks
            WHERE document_id = ?
              AND embedding_provider = ?
              AND embedding_model = ?
              AND chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index
            """,
            (
                document_id,
                embedding_provider,
                embedding_model,
                selected_index - normalized_context_chunks,
                selected_index + normalized_context_chunks,
            ),
        ).fetchall()

    json_payload = parse_object_json(document["json_payload"])
    metadata = parse_object_json(document["metadata_json"])
    chunk = chunk_payload(selected)
    media_path = normalize_string(json_payload.get("working_media_path")) or normalize_string(
        json_payload.get("source_media_path")
    )
    start_seconds = chunk.get("start_seconds")
    end_seconds = chunk.get("end_seconds")
    seek_hint = ""
    if media_path and start_seconds is not None:
        seek_hint = f"ffplay -ss {float(start_seconds):0.2f} {shell_quote(media_path)}"

    return {
        "document": {
            "id": document["id"],
            "kind": document["kind"],
            "title": document["title"],
            "source_path": document["source_path"],
            "stored_path": document["stored_path"],
            "generated_at": document["generated_at"],
            "metadata": metadata,
        },
        "chunk": chunk,
        "context_chunks": [chunk_payload(row) for row in nearby_rows],
        "media": {
            "path": media_path,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "start_timestamp": format_seconds(start_seconds),
            "end_timestamp": format_seconds(end_seconds),
            "seek_hint": seek_hint,
        },
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }


def legacy_enrichment_queue(
    *,
    root: Optional[Path] = None,
    limit: Optional[int] = None,
    pending_only: bool = True,
    provider: str = "openai-compatible",
    model: str = "",
    store_readouts: bool = True,
    dedupe: bool = True,
) -> dict[str, Any]:
    runtime_root = store_dir(root)
    with connect(runtime_root) as con:
        init_db(con)
        transcript_rows = con.execute(
            """
            SELECT documents.*,
                   CASE WHEN document_blobs.blob_id IS NULL THEN 0 ELSE 1 END AS has_media_blob
            FROM documents
            LEFT JOIN document_blobs
              ON document_blobs.document_id = documents.id
             AND document_blobs.role = 'source_recording'
            WHERE documents.kind = 'transcript'
            ORDER BY documents.generated_at, documents.title
            """
        ).fetchall()
        readout_rows = con.execute(
            """
            SELECT kind, source_path, json_payload, metadata_json
            FROM documents
            WHERE kind IN ('readout', 'contextual_readout')
            """
        ).fetchall()

    readout_counts: dict[str, int] = {}
    contextual_counts: dict[str, int] = {}
    for row in readout_rows:
        metadata = parse_object_json(row["metadata_json"])
        payload = parse_object_json(row["json_payload"])
        source_artifact_path = normalize_string(metadata.get("source_artifact_path")) or normalize_string(
            payload.get("source_artifact_path")
        )
        if not source_artifact_path:
            continue
        counts = contextual_counts if row["kind"] == "contextual_readout" else readout_counts
        counts[source_artifact_path] = counts.get(source_artifact_path, 0) + 1

    items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    duplicate_count = 0
    for row in transcript_rows:
        payload = parse_object_json(row["json_payload"])
        legacy = payload.get("legacy_import") if isinstance(payload.get("legacy_import"), dict) else {}
        if legacy.get("needs_enrichment") is not True:
            continue
        source_path = normalize_string(row["source_path"])
        readout_count = readout_counts.get(source_path, 0)
        contextual_readout_count = contextual_counts.get(source_path, 0)
        pending = readout_count == 0
        if pending_only and not pending:
            continue
        dedupe_key_parts = [
            normalize_string(legacy.get("source_sha256")),
            "title:" + " ".join(tokens(str(row["title"]))),
        ]
        dedupe_keys = [value for value in dedupe_key_parts if value and value != "title:"]
        if dedupe:
            matched_key = next((key for key in dedupe_keys if key in seen_keys), "")
            if matched_key:
                duplicate_count += 1
                continue
            seen_keys.update(dedupe_keys)
        command = [
            "python",
            "summarize_transcript.py",
            source_path,
            "--provider",
            provider,
        ]
        if model:
            command.extend(["--model", model])
        if store_readouts:
            command.append("--store")
        item = {
            "id": row["id"],
            "title": row["title"],
            "generated_at": row["generated_at"],
            "source_path": source_path,
            "stored_path": row["stored_path"],
            "legacy_source_path": normalize_string(legacy.get("source_path")),
            "legacy_source_sha256": normalize_string(legacy.get("source_sha256")),
            "source_media_path": normalize_string(payload.get("source_media_path")),
            "has_media_blob": bool(row["has_media_blob"]),
            "readout_count": readout_count,
            "contextual_readout_count": contextual_readout_count,
            "pending_first_pass_readout": pending,
            "command": command,
        }
        items.append(item)
        if limit is not None and len(items) >= limit:
            break

    return {
        "store_dir": str(runtime_root),
        "pending_only": pending_only,
        "dedupe": dedupe,
        "duplicate_count": duplicate_count,
        "selected_count": len(items),
        "items": items,
    }


def format_context_text(payload: dict[str, Any]) -> str:
    document = payload["document"]
    chunk = payload["chunk"]
    media = payload["media"]
    lines = [
        f"{document['title']} ({document['kind']})",
        f"Document: {document['id']}",
        f"Artifact: {document['source_path']}",
    ]
    if media.get("path"):
        lines.append(f"Media: {media['path']}")
    if media.get("start_seconds") is not None:
        end = f" - {media['end_timestamp']}" if media.get("end_timestamp") else ""
        lines.append(f"Time: {media['start_timestamp']}{end}")
    if chunk.get("speakers"):
        lines.append(f"Speakers: {', '.join(str(item) for item in chunk['speakers'])}")
    if media.get("seek_hint"):
        lines.append(f"Media seek: {media['seek_hint']}")
    lines.append("")
    for item in payload["context_chunks"]:
        marker = ">" if item["chunk_index"] == chunk["chunk_index"] else " "
        time_range = ""
        if item.get("start_seconds") is not None:
            time_range = f" [{format_seconds(item.get('start_seconds'))}"
            if item.get("end_seconds") is not None:
                time_range += f" - {format_seconds(item.get('end_seconds'))}"
            time_range += "]"
        lines.append(f"{marker} chunk {item['chunk_index']}{time_range}")
        lines.append(item["text"])
        lines.append("")
    return "\n".join(lines).rstrip()


def shell_command(argv: list[str]) -> str:
    return " ".join(shell_quote(item) for item in argv)


def format_legacy_enrichment_queue_text(payload: dict[str, Any]) -> str:
    lines = [
        f"First-pass summary queue: {payload['selected_count']} item(s)",
        f"Store: {payload['store_dir']}",
    ]
    if payload.get("duplicate_count"):
        lines.append(f"Deduped queue entries skipped: {payload['duplicate_count']}")
    lines.append("")
    for index, item in enumerate(payload["items"], start=1):
        media = "blob" if item.get("has_media_blob") else "no blob"
        lines.append(f"{index}. {item['title']} [{media}]")
        lines.append(f"   document: {item['id']}")
        lines.append(f"   artifact: {item['source_path']}")
        if item.get("legacy_source_path"):
            lines.append(f"   import source: {item['legacy_source_path']}")
        lines.append(f"   command: {shell_command(item['command'])}")
    return "\n".join(lines).rstrip()


def search_context_payload(
    *,
    query: str,
    results: list[dict[str, Any]],
    selected_rank: int,
    selected_result: dict[str, Any],
    context_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": query,
        "result_count": len(results),
        "selected_rank": selected_rank,
        "selected_result": selected_result,
        "context": context_payload,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the user-scoped ~/.transcripts store.")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR, help="Store root directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the store database.")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest transcript/readout JSON artifacts.")
    ingest_parser.add_argument("paths", nargs="+", type=Path, help="Artifact JSON paths to ingest.")
    ingest_parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "openai-compatible", "debug-hash", "hash"),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider. debug-hash/hash is an explicit test fallback, not the semantic path.",
    )
    ingest_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    search_parser = subparsers.add_parser("search", help="Search stored artifacts.")
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument("--kind", choices=("transcript", "readout", "contextual_readout", "artifact"), default="")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--context",
        action="store_true",
        help="Open the selected search result in the nearby-context view instead of printing search JSON.",
    )
    search_parser.add_argument("--context-rank", type=int, default=1, help="1-based search result rank to open.")
    search_parser.add_argument("--context-chunks", type=int, default=1, help="Chunks to include before and after.")
    search_parser.add_argument(
        "--context-format",
        choices=("text", "json", "compact-json"),
        default="text",
        help="Output format used with --context.",
    )
    search_parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "openai-compatible", "debug-hash", "hash"),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider used for semantic ranking.",
    )
    search_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    context_parser = subparsers.add_parser("context", help="Show nearby transcript context for a stored document chunk.")
    context_parser.add_argument("document_id", help="Document id from search or ingest output.")
    context_parser.add_argument("--chunk-index", type=int, help="Chunk index from a search result best_chunk.")
    context_parser.add_argument("--context-chunks", type=int, default=1, help="Chunks to include before and after.")
    context_parser.add_argument(
        "--format",
        choices=("text", "json", "compact-json"),
        default="text",
        help="Output format. compact-json emits pure single-line JSON without a sentinel.",
    )
    context_parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "openai-compatible", "debug-hash", "hash"),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider used when the document was chunked.",
    )
    context_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    legacy_queue_parsers = [
        subparsers.add_parser(
            "first-pass-summary-queue",
            help="List stored transcripts that still need first-pass readouts.",
        ),
        subparsers.add_parser(
            "legacy-enrichment-queue",
            help="Compatibility alias for first-pass-summary-queue.",
        ),
    ]
    for legacy_queue_parser in legacy_queue_parsers:
        legacy_queue_parser.add_argument("--limit", type=int, help="Limit queued item count.")
        legacy_queue_parser.add_argument(
            "--all",
            action="store_true",
            help="Include rows that already have at least one readout.",
        )
        legacy_queue_parser.add_argument(
            "--format",
            choices=("text", "json", "compact-json", "commands"),
            default="text",
            help="Queue output format. commands prints summarize_transcript.py commands only.",
        )
        legacy_queue_parser.add_argument(
            "--provider",
            default="openai-compatible",
            help="Provider to include in generated commands.",
        )
        legacy_queue_parser.add_argument("--model", default="", help="Optional model to include in generated commands.")
        legacy_queue_parser.add_argument(
            "--no-store",
            action="store_true",
            help="Do not include --store in generated readout commands.",
        )
        legacy_queue_parser.add_argument(
            "--no-dedupe",
            action="store_true",
            help="Do not collapse same-hash or same-title queue entries.",
        )

    backfill_parser = subparsers.add_parser("backfill", help="Discover and ingest artifact JSON files.")
    backfill_parser.add_argument("roots", nargs="+", type=Path, help="Files or directories to scan.")
    backfill_parser.add_argument("--recursive", action="store_true", help="Scan directories recursively.")
    backfill_parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob pattern to include. Can be repeated. Defaults to transcript/readout artifact patterns.",
    )
    backfill_parser.add_argument(
        "--exclude",
        action="append",
        dest="exclude_patterns",
        help="Path glob to exclude. Can be repeated. Defaults exclude pytest and copied store artifacts.",
    )
    backfill_parser.add_argument(
        "--modified-within-days",
        type=float,
        help="Only include files modified within this many days.",
    )
    backfill_parser.add_argument(
        "--kind",
        action="append",
        choices=("transcript", "readout", "contextual_readout", "artifact"),
        help="Only include this artifact kind. Can be repeated.",
    )
    backfill_parser.add_argument("--limit", type=int, help="Limit selected artifact count after filtering.")
    backfill_parser.add_argument("--dry-run", action="store_true", help="Only report selected paths and statuses.")
    backfill_parser.add_argument("--force", action="store_true", help="Re-ingest artifacts that are already current.")
    backfill_parser.add_argument(
        "--embedding-provider",
        choices=("ollama", "openai-compatible", "debug-hash", "hash"),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider used for ingested artifacts.",
    )
    backfill_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    root = args.store_dir.expanduser()
    try:
        if args.command == "init":
            with connect(root) as con:
                init_db(con)
            print(f"Initialized transcript store at {db_path(root)}")
            return 0
        if args.command == "ingest":
            results = [
                ingest_artifact(
                    path,
                    root=root,
                    embedding_provider=args.embedding_provider,
                    embedding_model=args.embedding_model,
                ).to_dict()
                for path in args.paths
            ]
            print(json.dumps({"store_dir": str(root), "results": results}, indent=2, ensure_ascii=False))
            print(f"{STORE_JSON_STDOUT_PREFIX}{db_path(root)}")
            return 0
        if args.command == "search":
            search_limit = max(args.limit, args.context_rank) if args.context else args.limit
            results = search_store(
                args.query,
                root=root,
                kind=args.kind,
                limit=search_limit,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            if args.context:
                if not results:
                    raise TranscriptStoreError(f"No search results for query: {args.query}")
                if args.context_rank < 1 or args.context_rank > len(results):
                    raise TranscriptStoreError(
                        f"--context-rank {args.context_rank} is outside the {len(results)} available results."
                    )
                result = results[args.context_rank - 1]
                best_chunk = result.get("best_chunk") if isinstance(result.get("best_chunk"), dict) else {}
                chunk_index = best_chunk.get("chunk_index")
                context_payload = context_for_document(
                    str(result["id"]),
                    root=root,
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                    context_chunks=args.context_chunks,
                    embedding_provider=args.embedding_provider,
                    embedding_model=args.embedding_model,
                )
                if args.context_format in {"json", "compact-json"}:
                    payload = search_context_payload(
                        query=args.query,
                        results=results,
                        selected_rank=args.context_rank,
                        selected_result=result,
                        context_payload=context_payload,
                    )
                    if args.context_format == "compact-json":
                        print(json_compact(payload))
                    else:
                        print(json.dumps(payload, indent=2, ensure_ascii=False))
                        print(f"{CONTEXT_JSON_STDOUT_PREFIX}{db_path(root)}")
                    return 0
                print(
                    f"Search hit {args.context_rank}/{len(results)}: "
                    f"{result['title']} score={float(result['score']):0.3f}"
                )
                print("")
                print(format_context_text(context_payload))
                print(f"{CONTEXT_JSON_STDOUT_PREFIX}{db_path(root)}")
                return 0
            print(json.dumps({"query": args.query, "results": results}, indent=2, ensure_ascii=False))
            print(f"{SEARCH_JSON_STDOUT_PREFIX}{db_path(root)}")
            return 0
        if args.command == "context":
            payload = context_for_document(
                args.document_id,
                root=root,
                chunk_index=args.chunk_index,
                context_chunks=args.context_chunks,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            if args.format == "json":
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            elif args.format == "compact-json":
                print(json_compact(payload))
                return 0
            else:
                print(format_context_text(payload))
            print(f"{CONTEXT_JSON_STDOUT_PREFIX}{db_path(root)}")
            return 0
        if args.command in {"first-pass-summary-queue", "legacy-enrichment-queue"}:
            stdout_prefix = (
                FIRST_PASS_SUMMARY_QUEUE_JSON_STDOUT_PREFIX
                if args.command == "first-pass-summary-queue"
                else LEGACY_ENRICHMENT_QUEUE_JSON_STDOUT_PREFIX
            )
            payload = legacy_enrichment_queue(
                root=root,
                limit=args.limit,
                pending_only=not args.all,
                provider=args.provider,
                model=args.model,
                store_readouts=not args.no_store,
                dedupe=not args.no_dedupe,
            )
            if args.format == "compact-json":
                print(json_compact(payload))
                return 0
            if args.format == "json":
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                print(f"{stdout_prefix}{db_path(root)}")
                return 0
            if args.format == "commands":
                for item in payload["items"]:
                    print(shell_command(item["command"]))
                return 0
            print(format_legacy_enrichment_queue_text(payload))
            print(f"{stdout_prefix}{db_path(root)}")
            return 0
        if args.command == "backfill":
            planned = plan_backfill(
                args.roots,
                root=root,
                recursive=args.recursive,
                patterns=args.patterns or DEFAULT_BACKFILL_PATTERNS,
                exclude_patterns=(*DEFAULT_BACKFILL_EXCLUDE_PATTERNS, *(args.exclude_patterns or [])),
                modified_within_days=args.modified_within_days,
                kinds=set(args.kind or []),
                limit=args.limit,
                force=args.force,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
            )
            items: list[dict[str, Any]]
            if args.dry_run:
                items = planned
            else:
                items = []
                for item in planned:
                    if item["status"] == "error":
                        items.append({**item, "result": "error"})
                        continue
                    if item["status"] == "skip" and not args.force:
                        items.append({**item, "result": "skipped"})
                        continue
                    result = ingest_artifact(
                        Path(item["source_path"]),
                        root=root,
                        embedding_provider=args.embedding_provider,
                        embedding_model=args.embedding_model,
                    ).to_dict()
                    items.append({**item, "result": result["status"], "stored_path": result["stored_path"]})
            payload = {
                "store_dir": str(root),
                "dry_run": bool(args.dry_run),
                "selected_count": len(planned),
                **summarize_backfill_status(planned),
                "items": items,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"{BACKFILL_JSON_STDOUT_PREFIX}{db_path(root)}")
            return 0
    except TranscriptStoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
