"""Append-only authored overlays on exact registered transcript utterances."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

from recording_registration import MAX_TRANSCRIPT, authorize
from recording_source_guard import guard_media
from transcript_store import connect

ROUTE = "/api/transcript-corrections"
HEX = re.compile(r"[a-f0-9]{64}")
DOC = re.compile(r"[a-f0-9]{20}")
FIELDS = {
    "id", "document_id", "transcript_sha256", "recording_sha256", "target",
    "previous_id", "text", "workspace_id", "correction_id", "revision",
    "actor_id", "authored_at", "reason",
}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def initialize(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS transcript_corrections (
            sequence INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
            document_id TEXT NOT NULL, target_key TEXT NOT NULL, payload TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS transcript_corrections_target
        ON transcript_corrections (document_id, target_key, sequence)
    """)


def verify_target(con, payload):
    registered = con.execute("""
        SELECT documents.*, blobs.sha256 AS recording_sha256 FROM documents
        JOIN document_blobs ON documents.id = document_blobs.document_id
        JOIN blobs ON blobs.id = document_blobs.blob_id
        WHERE documents.id = ? AND document_blobs.role = 'source_recording'
    """, (payload["document_id"],)).fetchall()
    if len(registered) != 1:
        return False
    row = registered[0]
    if (row["kind"] != "transcript" or row["artifact_sha256"] != payload["transcript_sha256"]
            or row["recording_sha256"] != payload["recording_sha256"]):
        return False
    return target_matches(row, digest(payload["target"]))


def target_matches(row, target_key):
    try:
        with Path(row["stored_path"]).open("rb") as stream:
            raw = stream.read(MAX_TRANSCRIPT + 1)
        if len(raw) > MAX_TRANSCRIPT or hashlib.sha256(raw).hexdigest() != row["artifact_sha256"]:
            return False
        sidecar = json.loads(raw)
    except (OSError, ValueError):
        return False
    if not isinstance(sidecar, dict):
        return False
    original, working = (sidecar.get(field) for field in ("source_media_path", "working_media_path"))
    if working and working != original:
        return False
    utterances = sidecar.get("utterances")
    if not isinstance(utterances, list):
        return False
    return sum(
        isinstance(turn, dict)
        and type(turn.get("start")) is int and type(turn.get("end")) is int
        and isinstance(turn.get("speaker"), str) and isinstance(turn.get("text"), str)
        and digest(dict(start_ms=turn["start"], end_ms=turn["end"], speaker=turn["speaker"],
                        machine_text=turn["text"])) == target_key
        for turn in utterances
    ) == 1


def validate(payload):
    if set(payload) != FIELDS:
        return False
    for field in ("id", "transcript_sha256", "recording_sha256"):
        if not isinstance(payload[field], str) or not HEX.fullmatch(payload[field]):
            return False
    prior = payload["previous_id"]
    if prior is not None and (not isinstance(prior, str) or not HEX.fullmatch(prior)):
        return False
    if not isinstance(payload["document_id"], str) or not DOC.fullmatch(payload["document_id"]):
        return False
    for field in ("workspace_id", "correction_id", "actor_id", "authored_at", "reason"):
        if not isinstance(payload[field], str) or not payload[field].strip() or len(payload[field]) > 4096:
            return False
    at = datetime.fromisoformat(payload["authored_at"].replace("Z", "+00:00"))
    target = payload["target"]
    return (
        at.utcoffset() is not None
        and type(payload["revision"]) is int and payload["revision"] > 0
        and isinstance(payload["text"], str) and len(payload["text"]) <= 32000
        and isinstance(target, dict) and set(target) == {"start_ms", "end_ms", "speaker", "machine_text"}
        and all(type(target[field]) is int for field in ("start_ms", "end_ms"))
        and 0 <= target["start_ms"] < target["end_ms"] < 2**53
        and isinstance(target["speaker"], str) and len(target["speaker"]) <= 512
        and isinstance(target["machine_text"], str) and len(target["machine_text"]) <= 32000
        and payload["id"] == digest([payload[key] for key in ("workspace_id", "correction_id", "revision")])
    )


def submit(handler):
    if not authorize(handler):
        return
    try:
        payload = handler.read_json_body()
        if not validate(payload):
            raise ValueError("Invalid exact correction")
    except (ValueError, TypeError):
        handler.write_error(HTTPStatus.BAD_REQUEST, "Invalid exact correction")
        return
    result = append(handler.store_root, payload)
    if result == "conflict":
        handler.write_error(HTTPStatus.CONFLICT, "The exact native target or prior correction changed")
        return
    handler.write_json(payload, status=HTTPStatus.OK if result == "replayed" else HTTPStatus.CREATED)


@guard_media
def append(root, payload):
    target_key = digest(payload["target"])
    with connect(root) as con:
        con.execute("BEGIN IMMEDIATE")
        initialize(con)
        if not verify_target(con, payload):
            return "conflict"
        old = con.execute("SELECT payload FROM transcript_corrections WHERE id = ?", (payload["id"],)).fetchone()
        if old:
            return "replayed" if json.loads(old["payload"]) == payload else "conflict"
        latest = con.execute("""
            SELECT id FROM transcript_corrections WHERE document_id = ? AND target_key = ?
            ORDER BY sequence DESC LIMIT 1
        """, (payload["document_id"], target_key)).fetchone()
        if (latest["id"] if latest else None) != payload["previous_id"]:
            return "conflict"
        con.execute("""
            INSERT INTO transcript_corrections (id, document_id, target_key, payload)
            VALUES (?, ?, ?, ?)
        """, (payload["id"], payload["document_id"], target_key, json.dumps(payload, ensure_ascii=False)))
        return "accepted"


def read(handler, params):
    if not authorize(handler):
        return
    document_id = (params.get("document_id") or [""])[0]
    target_key = (params.get("target_key") or [""])[0]
    if not DOC.fullmatch(document_id) or (target_key and not HEX.fullmatch(target_key)):
        handler.write_error(HTTPStatus.BAD_REQUEST, "A native document ID and optional target key are required")
        return
    with connect(handler.store_root) as con:
        verification = {}
        if target_key:
            document = con.execute(
                "SELECT stored_path, artifact_sha256 FROM documents WHERE id = ? AND kind = 'transcript'",
                (document_id,),
            ).fetchone()
            verification["target_found"] = bool(document and target_matches(document, target_key))
            rows = con.execute("""
                SELECT payload FROM transcript_corrections WHERE document_id = ? AND target_key = ?
                ORDER BY sequence
            """, (document_id, target_key))
        else:
            rows = con.execute("""
                SELECT payload FROM transcript_corrections WHERE document_id = ? ORDER BY sequence
            """, (document_id,))
        items = [json.loads(row["payload"]) for row in rows]
    handler.write_json({"items": items, **verification})
