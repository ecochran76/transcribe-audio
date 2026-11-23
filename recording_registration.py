"""Authenticated exact-byte admissions to the local transcript authority."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import sqlite3
import tempfile
from http import HTTPStatus
from pathlib import Path

from speech_harvesting import _private_token
from transcript_store import TranscriptStoreError, connect, ingest_artifact, stable_id

ROUTE = "/api/recording-registrations"
MAX_TRANSCRIPT = 16 * 1024**2
MAX_RECORDING = 2 * 1024**3
RECORDING_EXTENSIONS = frozenset({
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".m4b", ".mkv", ".mov", ".mp3",
    ".mp4", ".ogg", ".opus", ".wav", ".webm", ".wma",
})


def load_token(filename: Path, host: str) -> str:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError("Recording registration requires a loopback-bound API.")
    token = _private_token(str(filename))
    if len(token) < 32:
        raise ValueError("Recording registration requires a private random token of at least 32 characters.")
    return token


def authorize(handler) -> bool:
    token = handler.server.recording_registration_token
    if not token:
        handler.write_error(HTTPStatus.NOT_FOUND, "Not found")
        return False
    authorization = handler.headers.get("Authorization", "").encode("latin-1")
    if not hmac.compare_digest(authorization, f"Bearer {token}".encode("ascii")):
        handler.write_error(HTTPStatus.FORBIDDEN, "Recording registration authorization required")
        return False
    return True


def copy_part(stream, path: Path, count: int, expected_hash: str) -> None:
    digest = hashlib.sha256()
    with path.open("xb") as output:
        while count:
            chunk = stream.read(min(count, 1024**2))
            if not chunk:
                raise ValueError("The request body ended before the declared length")
            digest.update(chunk)
            output.write(chunk)
            count -= len(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("The uploaded bytes do not match the declared SHA-256")


def submit(handler) -> None:
    if not authorize(handler):
        return
    headers = handler.headers
    if headers.get_content_type() != "application/octet-stream" or headers.get("Transfer-Encoding"):
        handler.write_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send a fixed-length application/octet-stream body")
        return
    try:
        lengths = [headers.get_all(name, []) for name in ("Content-Length", "X-Transcript-Bytes")]
        if any(len(values) != 1 or not re.fullmatch(r"[0-9]{1,10}", values[0]) for values in lengths):
            raise ValueError("Content-Length and X-Transcript-Bytes must each be a decimal byte count")
        total, transcript_bytes = (int(values[0]) for values in lengths)
        recording_bytes = total - transcript_bytes
        if not 0 < transcript_bytes <= MAX_TRANSCRIPT or not 0 < recording_bytes <= MAX_RECORDING:
            handler.write_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Transcript or recording lengths are outside limits")
            return
        transcript_hash = headers.get("X-Transcript-Sha256", "")
        recording_hash = headers.get("X-Recording-Sha256", "")
        if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in (transcript_hash, recording_hash)):
            raise ValueError("Both SHA-256 headers must use 64 lowercase hexadecimal digits")
        extension = headers.get("X-Recording-Extension", "").lower()
        if extension not in RECORDING_EXTENSIONS:
            raise ValueError("A supported recording extension is required")
        with tempfile.TemporaryDirectory(prefix=".recording-registration-", dir=handler.store_root) as directory:
            transcript = Path(directory) / "External Recording.transcript.json"
            recording = Path(directory) / f"original{extension}"
            handler.connection.settimeout(30)
            copy_part(handler.rfile, transcript, transcript_bytes, transcript_hash)
            copy_part(handler.rfile, recording, recording_bytes, recording_hash)
            result = ingest_artifact(
                transcript,
                root=handler.store_root,
                embedding_provider=handler.server.embedding_provider,
                embedding_model=handler.server.embedding_model,
                registered_recording_path=recording,
            )
    except TimeoutError:
        handler.write_error(HTTPStatus.REQUEST_TIMEOUT, "Recording upload timed out")
        return
    except (ValueError, TranscriptStoreError):
        handler.write_error(HTTPStatus.UNPROCESSABLE_ENTITY, "Recording or transcript registration did not validate")
        return
    except (OSError, sqlite3.Error):
        handler.write_error(HTTPStatus.SERVICE_UNAVAILABLE, "Recording registration is temporarily unavailable")
        return
    handler.write_json(
        {
            "native_document_id": result.id,
            "native_recording_id": stable_id("source_recording", recording_hash),
            "recording_sha256": recording_hash,
            "transcript_sha256": transcript_hash,
        },
        status=HTTPStatus.CREATED if result.status == "inserted" else HTTPStatus.OK,
    )


def download_transcript(handler, document_id: str) -> None:
    if not authorize(handler):
        return
    with connect(handler.store_root) as con:
        row = con.execute(
            "SELECT source_path, stored_path, artifact_sha256 FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not row or Path(row["source_path"]).parts[:1] != ("recording-registration",):
        handler.write_error(HTTPStatus.NOT_FOUND, "Registered transcript unavailable")
        return
    try:
        raw = Path(row["stored_path"]).read_bytes()
    except OSError:
        raw = b""
    if hashlib.sha256(raw).hexdigest() != row["artifact_sha256"]:
        handler.write_error(HTTPStatus.CONFLICT, "Registered transcript readback failed")
        return
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Disposition", 'attachment; filename="recorded.transcript.json"')
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)
