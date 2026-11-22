"""Append-only Workbench range assignments against a registered transcript."""
import json
import os
import re
from contextlib import closing
from http import HTTPStatus

from contacts_receipts_client import ContactsLookupError, ContactsReceiptsLookup
from recording_registration import authorize
from transcript_store import connect

ROUTE = re.compile(r"/api/recording-registrations/([0-9a-f]{20})/speaker-assignments"
                   r"(?:/([A-Za-z0-9_-]{1,160}))?")
FIELDS = {"recording_sha256", "transcript_sha256", "start_ms", "end_ms", "revision",
          "previous_revision", "actor", "reason", "identity", "perspective"}


def apply(handler, connection, document_id, assignment_id, current):
    payload = handler.read_json_body()
    if set(payload) != FIELDS:
        raise ValueError("Expected an exact assignment projection.")
    start, end, revision, previous = (payload[key] for key in
                                      ("start_ms", "end_ms", "revision", "previous_revision"))
    identity = payload["identity"]
    if (
        any(type(value) is not int for value in (start, end, revision, previous))
        or not (0 <= start < end <= 2**53 - 1 and 0 <= previous < revision <= 2**53 - 1)
        or any(not isinstance(payload[key], str) or not payload[key].strip()
               for key in ("actor", "reason", "perspective"))
        or not isinstance(identity, dict) or set(identity) != {"id", "kind", "revision"}
        or any(not isinstance(payload[key], str) or not re.fullmatch(r"[0-9a-f]{64}", payload[key])
               for key in ("recording_sha256", "transcript_sha256"))
        or identity["kind"] != "entity" or not isinstance(identity["revision"], str)
        or not identity["revision"]
    ):
        raise ValueError("Invalid assignment interval, provenance or identity.")
    if current and current == payload:
        handler.write_json(current)
        return
    lookup = ContactsReceiptsLookup("Bearer " + os.environ.get("TRANSCRIBE_CONTACTS_ACCESS_TOKEN", ""))
    person = lookup.exact(identity["id"], identity["revision"])
    if person["qualification"] != "exact" or person["kind"] != "individual" or person["perspective"] != payload["perspective"]:
        raise ValueError("The contact is not an exact person in the selected perspective.")
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        document = connection.execute(
            "SELECT artifact_sha256 FROM documents WHERE id = ? AND kind = 'transcript'", (document_id,)
        ).fetchone()
        if not document or document[0] != payload["transcript_sha256"]:
            raise ValueError("Transcript hash does not match the native document.")
        blob = connection.execute(
            """SELECT 1 FROM document_blobs JOIN blobs ON document_blobs.blob_id = blobs.id
               WHERE document_id = ? AND role = 'source_recording' AND sha256 = ?""",
            (document_id, payload["recording_sha256"]),
        ).fetchone()
        if not blob:
            raise ValueError("Recording hash does not match the native document.")
        old = read(connection, document_id, assignment_id)
        if old == payload:
            handler.write_json(old)
            return
        if (old["revision"] if old else 0) != previous or (old and any(
            old[key] != payload[key] for key in ("start_ms", "end_ms", "recording_sha256", "transcript_sha256")
        )):
            handler.write_error(HTTPStatus.CONFLICT, "Assignment revision conflict.")
            return
        connection.execute("INSERT INTO workbench_speaker_projections VALUES (?, ?, ?, ?)",
                           (document_id, assignment_id, revision, json.dumps(payload, sort_keys=True)))
    handler.write_json(payload, status=HTTPStatus.CREATED)


def read(connection, document_id, assignment_id):
    row = connection.execute(
        """SELECT payload FROM workbench_speaker_projections
           WHERE document_id = ? AND assignment_id = ? ORDER BY revision DESC LIMIT 1""",
        (document_id, assignment_id),
    ).fetchone()
    return json.loads(row[0]) if row else None


def handle(handler, match, *, write=False):
    if not authorize(handler):
        return
    document_id, assignment_id = match.groups()
    if write and assignment_id is None:
        handler.write_error(HTTPStatus.METHOD_NOT_ALLOWED, "Choose an assignment identifier.")
        return
    try:
        with closing(connect(handler.store_root)) as connection:
            if write:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS workbench_speaker_projections (
                        document_id TEXT NOT NULL REFERENCES documents(id),
                        assignment_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
                        PRIMARY KEY (document_id, assignment_id, revision)
                    )"""
                )
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workbench_speaker_projections'"
            ).fetchone()
            if assignment_id is None:
                assignments = connection.execute(
                    """SELECT assignment_id, payload FROM workbench_speaker_projections AS p
                       WHERE document_id = ? AND revision = (
                           SELECT MAX(revision) FROM workbench_speaker_projections
                           WHERE document_id = p.document_id AND assignment_id = p.assignment_id
                       ) ORDER BY assignment_id""", (document_id,)
                ) if exists else ()
                handler.write_json({"items": [dict(id=row[0], **json.loads(row[1])) for row in assignments]})
                return
            current = read(connection, document_id, assignment_id) if exists else None
            if write:
                apply(handler, connection, document_id, assignment_id, current)
            elif current:
                handler.write_json(current)
            else:
                handler.write_error(HTTPStatus.NOT_FOUND, "No projected speaker assignment.")
    except ContactsLookupError as error:
        handler.write_error(error.status, str(error))
    except (ValueError, TypeError):
        handler.write_error(HTTPStatus.BAD_REQUEST, "Invalid exact speaker assignment.")
