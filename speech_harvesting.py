"""Optional owned meeting consumer of Speech's authenticated harvesting contract.

No diarizer, identity acceptance, biometric authority or private Speech database is
imported. Only the meeting's existing public source/diarization resolver is used.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from recording_source_guard import guard_media

MAX_MEDIA = 32_000_000
MAX_RESPONSE = 4_000_000
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("speech_invalid_request")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SpeechHarvestConsumer:
    def __init__(
        self,
        *,
        directory: Path,
        speech_url: str,
        token: str,
        source_token: str,
        owner: dict[str, str],
        resolve: Callable,
        allowed_origins: tuple[str, ...] = (),
        now: Callable = time.time,
        timeout: float = 5.0,
        observation_token: str | None = None,
    ):
        if set(owner) != {"tenant", "app", "account"} or not all(
            isinstance(x, str) and x.strip() and len(x) <= 256 for x in owner.values()
        ):
            raise ValueError("speech_owner_required")
        parsed = urlsplit(speech_url)
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not (
                parsed.scheme == "https"
                or (parsed.scheme == "http" and parsed.hostname == "127.0.0.1")
            )
            or not token
            or not source_token
            or timeout <= 0
            or timeout > 30
        ):
            raise ValueError("speech_secure_configuration_required")
        self.owner, self.resolve, self.now = dict(owner), resolve, now
        self.url, self.token, self.source_token = (
            speech_url.rstrip("/") + "/",
            token,
            source_token,
        )
        self.origins, self.timeout = tuple(allowed_origins), timeout
        self.observation_token = observation_token
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError("speech_invalid_storage")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise ValueError("speech_invalid_storage")
        self.directory.chmod(0o700)
        self._mutex = threading.RLock()
        custody_path = self.directory / "custody.sqlite"
        if custody_path.is_symlink():
            raise ValueError("speech_invalid_storage")
        self.custody = sqlite3.connect(custody_path, timeout=0, check_same_thread=False)
        custody_path.chmod(0o600)
        try:
            self.custody.execute("CREATE TABLE IF NOT EXISTS owner(id INTEGER)")
            self.custody.execute("BEGIN EXCLUSIVE")
        except sqlite3.Error:
            self.custody.close()
            raise ValueError("speech_storage_owned") from None
        filename = self.directory / "speech-harvesting.sqlite"
        try:
            if filename.is_symlink():
                raise ValueError("speech_invalid_storage")
            self.db = sqlite3.connect(filename, check_same_thread=False)
            filename.chmod(0o600)
            self.db.execute("PRAGMA secure_delete=ON")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,kind TEXT,intent TEXT,value TEXT)"
            )
            existing = self.db.execute(
                "SELECT value FROM settings WHERE key='owner'"
            ).fetchone()
            if existing and existing[0] != canonical(owner):
                self.db.close()
                raise ValueError("speech_owner_conflict")
            self.db.execute(
                "INSERT OR IGNORE INTO settings VALUES('owner',?)", (canonical(owner),)
            )
            for row in self.db.execute("SELECT id,value FROM operations").fetchall():
                value = json.loads(row[1])
                if value["state"] in {"submitting", "pending"}:
                    value["state"] = "unknown"
                    self.db.execute(
                        "UPDATE operations SET value=? WHERE id=?",
                        (canonical(value), row[0]),
                    )
            self.db.commit()
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            self.custody.close()
            raise
        self.closed = False

    def _source_hashes(self, value):
        values = [value]
        seen = set()
        hashes = set()
        while values:
            item = values.pop()
            sha = (item.get("source") or {}).get("sha256") or item.get("sourceSha256")
            if sha:
                hashes.add(sha)
            for field in ["harvestRequestId", "harvestOperationId"]:
                parent_id = item.get(field)
                if parent_id and parent_id not in seen:
                    seen.add(parent_id)
                    parent = self.db.execute(
                        "SELECT value FROM operations WHERE id=?", (parent_id,)
                    ).fetchone()
                    if parent:
                        values.append(json.loads(parent[0]))
        return hashes

    def _source_allowed(self, value):
        callback = getattr(self, "source_erasure_check", None)
        if callback:
            try:
                for sha in self._source_hashes(value):
                    callback(sha)
            except Exception as error:
                raise ValueError("speech_source_erased") from error
        if value.get("state") == "erased":
            raise ValueError("speech_source_erased")

    def erase_source_journal(self, sha256):
        """Original-owner callback after shared voice participants have purged."""
        with self._mutex, self.db:
            rows = self.db.execute("SELECT id,kind,value FROM operations").fetchall()
            ids = [
                (request_id, kind)
                for request_id, kind, raw in rows
                if sha256 in self._source_hashes(json.loads(raw))
            ]
            for request_id, kind in ids:
                self.db.execute(
                    "UPDATE operations SET value=? WHERE id=?",
                    (
                        canonical(
                            {
                                "requestId": request_id,
                                "state": "erased",
                                "kind": kind,
                                "sourceSha256": sha256,
                                "owner": self.owner,
                            }
                        ),
                        request_id,
                    ),
                )

    def close(self):
        with self._mutex:
            if not self.closed:
                self.closed = True
                self.db.close()
                self.custody.close()

    def check_origin(self, origin: str | None):
        if origin and origin not in self.origins:
            raise ValueError("speech_origin_forbidden")

    def _request(self, route: str, method="GET", body=None):
        data = None if body is None else canonical(body).encode()
        request = Request(
            self.url + route.lstrip("/"),
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            },
        )
        try:
            with build_opener(_NoRedirect).open(
                request, timeout=self.timeout
            ) as response:
                payload = response.read(MAX_RESPONSE + 1)
                if len(payload) > MAX_RESPONSE:
                    raise ValueError("speech_response_limit")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError("speech_invalid_response")
                return value
        except HTTPError as error:
            if error.code == 404:
                raise ValueError("speech_not_found") from None
            if error.code in {401, 403}:
                raise ValueError("speech_not_authorized") from None
            raise ValueError("speech_service_unavailable") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise ValueError("speech_service_unavailable") from None

    def _get(self, request_id: str):
        identifier(request_id)
        with self._mutex:
            if self.closed:
                raise ValueError("speech_closed")
            row = self.db.execute(
                "SELECT kind,intent,value FROM operations WHERE id=?", (request_id,)
            ).fetchone()
        if not row:
            raise ValueError("speech_operation_not_found")
        value = json.loads(row[2])
        self._source_allowed(value)
        return row[0], row[1], value

    def _save(self, value):
        with self._mutex, self.db:
            self._source_allowed(value)
            self.db.execute(
                "UPDATE operations SET value=? WHERE id=?",
                (canonical(value), value["requestId"]),
            )
        return value

    def _admit(self, request_id, kind, intent, value):
        identifier(request_id)
        with self._mutex, self.db:
            self._source_allowed(value)
            row = self.db.execute(
                "SELECT kind,intent,value FROM operations WHERE id=?", (request_id,)
            ).fetchone()
            if row:
                if row[0] != kind or row[1] != digest(intent):
                    raise ValueError("speech_request_conflict")
                return json.loads(row[2])
            count, size = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(length(value)),0) FROM operations"
            ).fetchone()
            if count >= 1024 or size + len(canonical(value).encode()) > 64_000_000:
                raise ValueError("speech_journal_limit")
            self.db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (request_id, kind, digest(intent), canonical(value)),
            )
        return None

    def prepare(self, document_id):
        identifier(document_id)
        value = self.resolve(document_id)
        if value.get("owner") != self.owner:
            raise ValueError("speech_source_forbidden")
        source, diarization = value["source"], value["diarization"]
        if (
            source.get("account") != self.owner["account"]
            or source.get("service") != "transcribe-audio"
            or not re.fullmatch("[a-f0-9]{64}", source.get("sha256", ""))
            or source.get("revision") != source["sha256"]
            or source.get("stream") != 0
            or source.get("channelPolicy") != "mix-to-mono"
        ):
            raise ValueError("speech_source_invalid")
        identifier(source.get("id"))
        if (
            diarization.get("units") != "ms"
            or not diarization.get("revision")
            or not isinstance(diarization.get("turns"), list)
            or not 0 < len(diarization["turns"]) <= 1000
            or len(diarization.get("words", [])) > 20000
        ):
            raise ValueError("speech_diarization_unavailable")
        if len(canonical(diarization).encode()) > 3_000_000:
            raise ValueError("speech_diarization_limit")
        return {
            "documentId": document_id,
            "source": source,
            "diarization": diarization,
            "diarizationSha256": digest(diarization),
            "owner": self.owner,
        }

    def context(self, document_id):
        prepared = self.prepare(document_id)
        with self._mutex:
            jobs = [
                (row[0], json.loads(row[1]))
                for row in self.db.execute(
                    "SELECT kind,value FROM operations ORDER BY rowid DESC LIMIT 256"
                )
            ]
        return {
            "configured": True,
            "owner": self.owner,
            "prepared": prepared,
            "jobs": [
                value
                for kind, value in jobs
                if kind == "harvest" and value["documentId"] == document_id
            ],
            "retentions": [
                value
                for kind, value in jobs
                if kind == "retain" and value["documentId"] == document_id
            ],
            "proposals": [
                value
                for kind, value in jobs
                if kind == "proposal" and value["documentId"] == document_id
            ],
            "foreignObservations": [
                value
                for kind, value in jobs
                if kind == "observation" and value["documentId"] == document_id
            ],
            "assignmentAuthority": "transcribe-audio",
            "sampleAuthority": "speech-receipts",
        }

    def start(self, submitted):
        request_id, document_id = (
            identifier(submitted.get("requestId")),
            identifier(submitted.get("documentId")),
        )
        try:
            kind, intent, prior = self._get(request_id)
            if kind != "harvest" or intent != digest(submitted):
                raise ValueError("speech_request_conflict")
            return prior
        except ValueError as error:
            if str(error) != "speech_operation_not_found":
                raise
        prepared = self.prepare(document_id)
        if (
            submitted.get("sourceSha256") != prepared["source"]["sha256"]
            or submitted.get("sourceRevision") != prepared["source"]["revision"]
            or submitted.get("diarizationSha256") != prepared["diarizationSha256"]
        ):
            raise ValueError("speech_source_changed")
        speech_id = "ta-harvest:" + digest([self.owner, request_id, document_id])[:48]
        source = {
            **prepared["source"],
            "kind": "application",
            "operationId": request_id,
        }
        value = {
            "requestId": request_id,
            "documentId": document_id,
            "speechRequestId": speech_id,
            "source": source,
            "diarization": prepared["diarization"],
            "state": "submitting",
            "sourceExpiresAt": self.now() + 600,
            "owner": self.owner,
        }
        prior = self._admit(request_id, "harvest", submitted, value)
        if prior:
            return prior
        try:
            result = self._request(
                "/v1/harvesting",
                "POST",
                {
                    "requestId": speech_id,
                    "owned": True,
                    "source": source,
                    "diarization": prepared["diarization"],
                },
            )
            value.update(state=result["state"], result=result)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)

    def status(self, request_id):
        kind, _, value = self._get(request_id)
        if kind == "observation":
            return value
        if kind == "retain":
            return self.reconcile_retention(request_id)
        if kind == "proposal":
            if value["state"] in {"pending", "unknown"}:
                try:
                    state = self._request(
                        "/v1/sample-associations/samples/"
                        + quote(value["sampleId"], safe="")
                    )
                    matches = [
                        item
                        for item in state.get("proposals", [])
                        if item.get("id") == value["speechRequestId"]
                        and item.get("sourceApp") == self.owner["app"]
                    ]
                    if len(matches) == 1:
                        value.update(state="proposed", result=matches[0])
                except ValueError as error:
                    value.update(state="unknown", error=str(error))
            return self._save(value)
        try:
            result = self._request(
                "/v1/harvesting/" + quote(value["speechRequestId"], safe="")
            )
            if result.get("requestId") != value["speechRequestId"]:
                raise ValueError("speech_invalid_response")
            value.update(state=result["state"], result=result)
            value.pop("error", None)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)

    def operation(self, request_id):
        return self._get(request_id)[2]

    @guard_media
    def export_source(self, request_id, authorization):
        if not isinstance(authorization, str) or not hmac.compare_digest(
            authorization, "Bearer " + self.source_token
        ):
            raise ValueError("speech_source_forbidden")
        kind, _, value = self._get(request_id)
        if (
            kind != "harvest"
            or value["owner"] != self.owner
            or value["sourceExpiresAt"] <= self.now()
            or value["state"] not in {"submitting", "queued", "processing", "unknown"}
        ):
            raise ValueError("speech_source_unavailable")
        current = self.prepare(value["documentId"])
        if current["source"] != {
            key: val
            for key, val in value["source"].items()
            if key not in {"kind", "operationId"}
        } or current["diarizationSha256"] != digest(value["diarization"]):
            raise ValueError("speech_source_changed")
        resolved = self.resolve(value["documentId"])
        media = Path(resolved["path"])
        if (
            media.is_symlink()
            or not media.is_file()
            or media.stat().st_size > MAX_MEDIA
        ):
            raise ValueError("speech_media_limit")
        with media.open("rb") as handle:
            audio = handle.read(MAX_MEDIA + 1)
        if (
            len(audio) > MAX_MEDIA
            or hashlib.sha256(audio).hexdigest() != value["source"]["sha256"]
        ):
            raise ValueError("speech_source_changed")
        return {
            "owner": self.owner,
            "source": value["source"],
            "audioBase64": base64.b64encode(audio).decode("ascii"),
        }

    def _capture_reference(self, value):
        current = self._request(
            "/v1/sample-associations/samples/" + quote(value["sampleId"], safe="")
        )
        value.update(
            sampleRevision=current["sample"]["revision"],
            associationRevision=current["revision"],
            person=current["person"]["reference"] if current.get("person") else None,
        )

    def _observation_authority(self, authorization):
        if (
            not self.observation_token
            or not isinstance(authorization, str)
            or not hmac.compare_digest(
                authorization, "Bearer " + self.observation_token
            )
        ):
            raise ValueError("speech_observation_forbidden")

    def observation_status(self, document_id, request_id, authorization):
        self._observation_authority(authorization)
        kind, _, value = self._get(request_id)
        if (
            kind != "observation"
            or value.get("documentId") != document_id
            or value.get("owner") != self.owner
        ):
            raise ValueError("speech_observation_forbidden")
        return value

    def references(self, authorization):
        self._observation_authority(authorization)
        with self._mutex:
            records = [
                (row[0], json.loads(row[1]))
                for row in self.db.execute(
                    "SELECT kind,value FROM operations ORDER BY rowid"
                )
            ]
        retained = [
            value
            for kind, value in records
            if kind == "retain"
            and value["state"] == "applied"
            and "sampleRevision" in value
        ]
        if len(retained) > 256:
            raise ValueError("speech_reference_limit")
        observations = [value for kind, value in records if kind == "observation"]
        current = self._request("/v1/speakers") if retained else {"speakers": []}
        snapshots = {
            sample["id"]: sample
            for speaker in current.get("speakers", [])
            for sample in speaker.get("samples", [])
        }
        result = []
        for value in retained:
            try:
                self._source_allowed(value)
            except ValueError:
                continue
            _, _, harvest = self._get(value["harvestRequestId"])
            sample = snapshots.get(value["sampleId"])
            association = (sample or {}).get("association") or {}
            seen = [
                item for item in observations if item["sampleId"] == value["sampleId"]
            ]
            latest = max(
                seen, key=lambda item: item["associationRevision"], default=None
            )
            conflicting = [
                item
                for item in seen
                if latest
                and item["associationRevision"] == latest["associationRevision"]
                and item["person"] != latest["person"]
            ]
            result.append(
                {
                    "documentId": value["documentId"],
                    "harvestOperationId": value["harvestRequestId"],
                    "sampleId": value["sampleId"],
                    "sampleRevision": sample["revision"]
                    if sample
                    else value["sampleRevision"],
                    "associationRevision": association.get("revision", 0)
                    if sample
                    else value["associationRevision"],
                    "person": None
                    if conflicting
                    else (
                        (association.get("person") or {}).get("reference")
                        if sample
                        else value["person"]
                    ),
                    "sourceId": harvest["source"]["id"],
                    "sourceService": "transcribe-audio",
                    "sourceAccount": self.owner["account"],
                    "sourceSha256": harvest["source"]["sha256"],
                    "sourceRevision": harvest["source"]["revision"],
                    "diarizationSha256": harvest["result"]["diarizationSha256"],
                    "actor": association.get("actor")
                    or {"tenant": self.owner["tenant"], "app": self.owner["app"]},
                    "evidence": association.get("evidence")
                    or "Explicitly retained shared Speech sample",
                    "observationId": latest["requestId"] if latest else None,
                    "state": "sample-unavailable"
                    if sample is None
                    else (
                        "association-pending"
                        if sample.get("associationPending")
                        else (
                            "conflict"
                            if conflicting
                            else (latest["state"] if latest else "retained")
                        )
                    ),
                    "conflictingObservationIds": [
                        item["requestId"]
                        for item in seen
                        if latest
                        and item["associationRevision"] == latest["associationRevision"]
                    ]
                    if conflicting
                    else [],
                    "meetingAssignmentChanged": False,
                }
            )
        result.sort(key=lambda item: item["sampleId"])
        return {"revision": digest(result), "references": result, "owner": self.owner}

    def observe(self, document_id, harvest_id, submitted, authorization):
        with self._mutex:
            return self._observe(document_id, harvest_id, submitted, authorization)

    def _observe(self, document_id, harvest_id, submitted, authorization):
        self._observation_authority(authorization)
        operation = identifier(submitted.get("requestId"))
        _, _, harvest = self._get(harvest_id)
        if harvest["documentId"] != document_id:
            raise ValueError("speech_source_forbidden")
        expected = {
            "sourceId": harvest["source"]["id"],
            "sourceSha256": harvest["source"]["sha256"],
            "sourceRevision": harvest["source"]["revision"],
            "diarizationSha256": harvest["result"]["diarizationSha256"],
        }
        if any(submitted.get(key) != value for key, value in expected.items()):
            raise ValueError("speech_source_changed")
        sample_id = identifier(submitted.get("sampleId"))
        with self._mutex:
            retained = [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT value FROM operations WHERE kind='retain'"
                )
            ]
            previous = [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT value FROM operations WHERE kind='observation'"
                )
            ]
        if not any(
            item.get("sampleId") == sample_id
            and item["harvestRequestId"] == harvest_id
            and item["state"] == "applied"
            for item in retained
        ):
            raise ValueError("speech_retention_unconfirmed")
        actor, person, evidence = (
            submitted.get("actor"),
            submitted.get("person"),
            submitted.get("evidence"),
        )
        if (
            not isinstance(actor, dict)
            or actor.get("tenant") != self.owner["tenant"]
            or not isinstance(actor.get("app"), str)
            or not ID.fullmatch(actor["app"])
            or not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 2000
        ):
            raise ValueError("speech_observation_invalid")
        if person is not None and (
            not isinstance(person, dict)
            or person.get("kind") != "entity"
            or not re.fullmatch(r"[a-f0-9-]{36}", str(person.get("id", "")), re.I)
            or not isinstance(person.get("revision"), str)
            or not 0 < len(person["revision"]) <= 128
        ):
            raise ValueError("speech_observation_invalid")
        if any(
            type(submitted.get(field)) is not int or submitted[field] < minimum
            for field, minimum in (("sampleRevision", 1), ("associationRevision", 0))
        ):
            raise ValueError("speech_observation_invalid")
        revision = submitted["associationRevision"]
        conflicts = [
            item
            for item in previous
            if item["sampleId"] == sample_id
            and (
                item["associationRevision"] > revision
                or (
                    item["associationRevision"] == revision and item["person"] != person
                )
            )
        ]
        value = {
            **submitted,
            "requestId": operation,
            "documentId": document_id,
            "harvestOperationId": harvest_id,
            "state": "conflict" if conflicts else "proposed",
            "meetingAssignmentChanged": False,
            "owner": self.owner,
        }
        prior = self._admit(
            operation,
            "observation",
            {**submitted, "documentId": document_id, "harvestOperationId": harvest_id},
            value,
        )
        return prior or value

    def search_people(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 120:
            raise ValueError("speech_invalid_request")
        return self._request(
            "/v1/sample-associations/search?q=" + quote(query, safe="")
        )

    def candidate(self, request_id, candidate_id):
        identifier(candidate_id)
        _, _, value = self._get(request_id)
        return self._request(
            "/v1/harvesting/"
            + quote(value["speechRequestId"], safe="")
            + "/candidates/"
            + quote(candidate_id, safe="")
        )

    def cancel(self, request_id):
        kind, _, value = self._get(request_id)
        if kind != "harvest":
            raise ValueError("speech_invalid_request")
        value["state"] = "cancelling"
        self._save(value)
        try:
            result = self._request(
                "/v1/harvesting/"
                + quote(value["speechRequestId"], safe="")
                + "/cancel",
                "POST",
            )
            value.update(state=result["state"], result=result)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)

    def retain(self, request_id, submitted):
        _, _, harvest = self._get(request_id)
        operation = identifier(submitted.get("requestId"))
        candidate = identifier(submitted.get("candidateId"))
        if submitted.get("reviewed") is not True:
            raise ValueError("speech_review_required")
        remote_id = "ta-retain:" + digest([self.owner, operation])[:48]
        value = {
            "requestId": operation,
            "documentId": harvest["documentId"],
            "harvestRequestId": request_id,
            "speechRequestId": remote_id,
            "state": "pending",
            "candidateId": candidate,
            "owner": self.owner,
        }
        prior = self._admit(
            operation, "retain", {**submitted, "harvestRequestId": request_id}, value
        )
        if prior:
            return self.reconcile_retention(operation)
        try:
            result = self._request(
                "/v1/speakers/harvest-retain",
                "POST",
                {
                    "requestId": remote_id,
                    "harvestId": harvest["speechRequestId"],
                    "candidateId": candidate,
                    "mode": "reviewed",
                    "name": submitted.get("name", "Meeting voice sample"),
                },
            )
            value.update(state="applied", result=result, sampleId=result["id"])
            self._capture_reference(value)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)

    def reconcile_retention(self, request_id):
        kind, _, value = self._get(request_id)
        if kind != "retain":
            raise ValueError("speech_invalid_request")
        if value["state"] == "applied":
            if "sampleRevision" not in value:
                self._capture_reference(value)
                self._save(value)
            return value
        try:
            result = self._request(
                "/v1/speakers/requests/" + quote(value["speechRequestId"], safe="")
            )
            if result.get("status") == "applied":
                value.update(state="applied", result=result, sampleId=result["id"])
                self._capture_reference(value)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)

    def propose(self, retention_id, submitted):
        retained = self.reconcile_retention(retention_id)
        if retained["state"] != "applied":
            raise ValueError("speech_retention_unconfirmed")
        operation = identifier(submitted.get("requestId"))
        target, evidence = submitted.get("target"), submitted.get("evidence")
        if (
            not isinstance(target, dict)
            or target.get("kind") != "entity"
            or not re.fullmatch(r"[a-f0-9-]{36}", str(target.get("id", "")), re.I)
            or not isinstance(target.get("revision"), str)
            or not target["revision"]
            or not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 2000
        ):
            raise ValueError("speech_proposal_invalid")
        remote_id = "ta-proposal:" + digest([self.owner, operation])[:48]
        value = {
            "requestId": operation,
            "documentId": retained["documentId"],
            "retentionRequestId": retention_id,
            "sampleId": retained["sampleId"],
            "speechRequestId": remote_id,
            "state": "pending",
            "owner": self.owner,
        }
        prior = self._admit(
            operation,
            "proposal",
            {**submitted, "retentionRequestId": retention_id},
            value,
        )
        if prior:
            return prior
        try:
            current = self._request(
                "/v1/sample-associations/samples/"
                + quote(retained["sampleId"], safe="")
            )
            result = self._request(
                "/v1/sample-associations/proposals",
                "POST",
                {
                    "requestId": remote_id,
                    "sampleId": retained["sampleId"],
                    "sampleRevision": current["sample"]["revision"],
                    "associationRevision": current["revision"],
                    "target": target,
                    "evidence": evidence,
                },
            )
            value.update(state="proposed", result=result)
        except ValueError as error:
            value.update(state="unknown", error=str(error))
        return self._save(value)


def normalize_diarization(payload: dict, *, document_revision: str) -> dict:
    """Normalize documented exporter units, never infer seconds from magnitude."""
    turns, words = [], []
    unit = payload.get("word_timestamp_units")
    if unit is None:
        unit = {"faster_whisper": "s", "assembly": "ms"}.get(payload.get("backend"))
    for utterance in payload.get("utterances") or []:
        label = str(
            utterance.get("speaker")
            if utterance.get("speaker") is not None
            else "unidentified"
        )
        start, end = utterance.get("start"), utterance.get("end")
        if (
            not label
            or len(label) > 128
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 3_600_000
                for x in (start, end)
            )
            or end <= start
        ):
            raise ValueError("speech_diarization_unavailable")
        turns.append({"speaker": label, "startMs": start, "endMs": end})
        for word in utterance.get("words") or []:
            if unit not in {"s", "ms"}:
                continue
            a, b = word.get("start"), word.get("end")
            if any(
                isinstance(x, bool)
                or not isinstance(x, (float, int))
                or x < 0
                or x != x
                or x == float("inf")
                for x in (a, b)
            ):
                raise ValueError("speech_diarization_unavailable")
            if unit == "s":
                a, b = round(a * 1000), round(b * 1000)
            if int(a) != a or int(b) != b or b <= a or b > 3_600_000:
                raise ValueError("speech_diarization_unavailable")
            words.append({"speaker": label, "startMs": int(a), "endMs": int(b)})
    if not 0 < len(turns) <= 1000 or len(words) > 20000:
        raise ValueError("speech_diarization_unavailable")
    body = {"turns": turns, "units": "ms"}
    if words:
        body["words"] = words
    return {**body, "revision": "ta-diarization:" + digest([document_revision, body])}


def read_consumer_config(filename: Path) -> dict:
    filename = Path(filename)
    metadata = filename.lstat()
    if (
        not filename.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size > 32_000
    ):
        raise ValueError("speech_private_config_required")
    value = json.loads(filename.read_text())
    if not isinstance(value, dict):
        raise ValueError("speech_invalid_configuration")
    return value


def _private_token(filename: str) -> str:
    path = Path(filename)
    metadata = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size > 8192
    ):
        raise ValueError("speech_private_token_required")
    token = path.read_text().strip()
    if not token or any(ord(x) < 33 or ord(x) > 126 for x in token):
        raise ValueError("speech_private_token_required")
    return token


def configured_consumer(filename: Path, resolve: Callable) -> SpeechHarvestConsumer:
    config = read_consumer_config(filename)
    return SpeechHarvestConsumer(
        directory=Path(config["state_directory"]),
        speech_url=config["speech_url"],
        token=_private_token(config["speech_token_file"]),
        source_token=_private_token(config["source_token_file"]),
        owner=config["owner"],
        resolve=resolve,
        allowed_origins=tuple(config["allowed_origins"]),
        observation_token=_private_token(config["observation_token_file"])
        if config.get("observation_token_file")
        else None,
    )
