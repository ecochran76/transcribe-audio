"""Explicit recording-owner erasure; disabled unless independently configured.

The owner covers registered recording blobs, explicitly owned original roots and
known playback copies. It never grants authority over arbitrary request paths.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
import threading
from pathlib import Path

import transcript_store
from speech_harvesting import canonical, digest, identifier

# Shared with supported streaming/playback paths. A fence waits for existing I/O;
# all later reads recheck the durable store tombstone while holding this lock.
from recording_source_guard import media_access


def source_available(root: Path, sha256: str):
    with transcript_store.connect(root) as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recording_source_erasures'"
        ).fetchone()
        if (
            exists
            and con.execute(
                "SELECT 1 FROM recording_source_erasures WHERE sha256=?", (sha256,)
            ).fetchone()
        ):
            raise transcript_store.TranscriptStoreError("Recording source erased")


class RecordingSourceOwner:
    def __init__(
        self,
        *,
        store_root,
        directory,
        owner,
        token,
        principals,
        original_roots,
        consumer=None,
    ):
        self.root, self.directory = Path(store_root), Path(directory)
        if not hasattr(os, "getuid"):
            raise ValueError("speech_owner_platform_unavailable")
        if (
            not self.root.is_absolute()
            or not self.directory.is_absolute()
            or set(owner) != {"tenant", "app", "account"}
            or not token
            or not principals
        ):
            raise ValueError("speech_owner_configuration_required")
        self.owner, self.token, self.principals = dict(owner), token, list(principals)
        self.consumer = consumer
        self.roots = []
        for submitted in [self.root, *map(Path, original_roots)]:
            if (
                not submitted.is_absolute()
                or submitted.is_symlink()
                or not submitted.is_dir()
                or submitted.stat().st_uid != os.getuid()
            ):
                raise ValueError("speech_owner_configuration_required")
            self.roots.append(submitted.resolve())
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise ValueError("speech_owner_configuration_required")
        self.directory.chmod(0o700)
        self._mutex = threading.RLock()
        filename = self.directory / "recording-erasure.sqlite"
        if filename.is_symlink():
            raise ValueError("speech_owner_configuration_required")
        custody = self.directory / "custody.sqlite"
        if custody.is_symlink():
            raise ValueError("speech_owner_configuration_required")
        self.custody = sqlite3.connect(custody, timeout=0, check_same_thread=False)
        custody.chmod(0o600)
        try:
            self.custody.execute("CREATE TABLE IF NOT EXISTS owner(id INTEGER)")
            self.custody.execute("BEGIN EXCLUSIVE")
        except sqlite3.Error:
            self.custody.close()
            raise ValueError("speech_owner_storage_owned") from None
        try:
            self.db = sqlite3.connect(filename, check_same_thread=False)
            filename.chmod(0o600)
            self.db.executescript(
                "PRAGMA secure_delete=ON;PRAGMA synchronous=FULL;CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY,value TEXT);CREATE TABLE IF NOT EXISTS jobs(principal TEXT,id TEXT,intent TEXT,value TEXT,PRIMARY KEY(principal,id));"
            )
            prior = self.db.execute("SELECT value FROM settings WHERE id=1").fetchone()
            binding = canonical(
                {
                    "owner": self.owner,
                    "roots": list(map(str, self.roots)),
                    "principals": self.principals,
                }
            )
            if prior and prior[0] != binding:
                self.db.close()
                self.custody.close()
                raise ValueError("speech_owner_configuration_changed")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES(1,?)", (binding,))
            self.db.commit()
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            self.custody.close()
            raise
        self.closed = False
        if self.consumer:
            self.consumer.source_store_root = self.root

    def close(self):
        with self._mutex:
            if not self.closed:
                self.closed = True
                self.db.close()
                self.custody.close()

    def _source(self, source):
        if (
            not isinstance(source, dict)
            or set(source) != {"service", "account", "id", "revision", "sha256"}
            or source.get("service") != "transcribe-audio"
            or source.get("account") != self.owner["account"]
            or source.get("revision") != source.get("sha256")
            or not isinstance(source.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
        ):
            raise ValueError("speech_owner_source_changed")
        identifier(source.get("id"))
        return source

    def _file(self, path, *, quarantined=False):
        # Reject symlink components, hardlinks and changed ownership, not just the
        # final filename. Only explicitly owned roots participate in deletion.
        path = Path(path)
        if not path.is_absolute() or not any(
            path.is_relative_to(root)
            for root in [
                *self.roots,
                *([self.directory.resolve()] if quarantined else []),
            ]
        ):
            raise ValueError("speech_owner_original_unavailable")
        current = path
        while current != current.parent:
            if current.is_symlink():
                raise ValueError("speech_owner_original_unavailable")
            current = current.parent
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size > 32_000_000
        ):
            raise ValueError("speech_owner_original_unavailable")
        with path.open("rb") as handle:
            data = handle.read(32_000_001)
        if len(data) > 32_000_000:
            raise ValueError("speech_owner_original_unavailable")
        return {
            "id": digest(str(path)),
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "inode": metadata.st_ino,
            "device": metadata.st_dev,
        }

    def _capture(self, source):
        with transcript_store.connect(self.root) as con:
            transcript_store.init_db(con)
            rows = [
                dict(row)
                for row in con.execute(
                    "SELECT b.*,d.document_id,x.json_payload AS document_payload FROM blobs b JOIN document_blobs d ON d.blob_id=b.id JOIN documents x ON x.id=d.document_id WHERE b.sha256=?",
                    (source["sha256"],),
                )
            ]
        if not any(
            "ta-recording-"
            + digest([self.owner["account"], row["document_id"], row["id"]])[:40]
            == source["id"]
            for row in rows
        ):
            raise ValueError("speech_owner_source_changed")
        with transcript_store.connect(self.root) as con:
            transcript_store.init_db(con)
            expected = {
                row["path"]: row["sha256"]
                for row in con.execute(
                    "SELECT path,sha256 FROM recording_original_copies WHERE source_sha256=?",
                    (source["sha256"],),
                )
            }
            for sha in set(expected.values()):
                aliases = con.execute(
                    "SELECT DISTINCT source_sha256 FROM recording_original_copies WHERE sha256=? AND source_sha256<>?",
                    (sha, source["sha256"]),
                ).fetchall()
                if aliases:
                    raise ValueError("speech_owner_shared_original_unavailable")
        paths = set(expected)
        canonical_copies = set()
        blob_ids = set()
        for row in rows:
            blob_ids.add(row["id"])
            payload = json.loads(row.get("document_payload") or "{}")
            for field in ["working_media_path", "source_media_path"]:
                if isinstance(payload.get(field), str) and payload[field]:
                    paths.add(str(Path(payload[field]).expanduser()))
            for key in ["stored_path", "original_path"]:
                if row[key]:
                    paths.add(row[key])
        for blob_id in blob_ids:
            for path in (self.root / "blobs" / blob_id[:2]).glob(blob_id + ".*"):
                paths.add(str(path))
                canonical_copies.add(str(path))
            for pattern in [f"{blob_id}-*.mp3", f".{blob_id}-*.mp3"]:
                paths.update(
                    str(p) for p in (self.root / "playback-cache").glob(pattern)
                )
        if len(paths) > 256:
            raise ValueError("speech_owner_inventory_limit")
        if sum(Path(path).lstat().st_size for path in paths) > 128_000_000:
            raise ValueError("speech_owner_inventory_limit")
        files = sorted([self._file(p) for p in paths], key=lambda value: value["id"])
        if sum(item["bytes"] for item in files) > 128_000_000:
            raise ValueError("speech_owner_inventory_limit")
        # Original and canonical copies must match the exact immutable recording.
        originals = (
            set(expected)
            | canonical_copies
            | {
                row[k]
                for row in rows
                for k in ["stored_path", "original_path"]
                if row[k]
            }
        )
        if any(
            item["path"] in originals
            and item["sha256"] != expected.get(item["path"], source["sha256"])
            for item in files
        ):
            raise ValueError("speech_owner_source_changed")
        return files, sorted(blob_ids)

    def _write(self, principal, request_id, value):
        self.db.execute(
            "UPDATE jobs SET value=? WHERE principal=? AND id=?",
            (canonical(value), principal, request_id),
        )
        self.db.commit()

    def request(self, action, submitted, authorization):
        if not isinstance(authorization, str) or not hmac.compare_digest(
            authorization, "Bearer " + self.token
        ):
            raise ValueError("speech_owner_forbidden")
        principal = submitted.get("principal")
        if (
            principal not in self.principals
            or principal.get("tenant") != self.owner["tenant"]
        ):
            raise ValueError("speech_owner_forbidden")
        key = canonical(principal)
        with self._mutex, media_access(self.root):
            if self.closed:
                raise ValueError("speech_owner_closed")
            if action == "inventory":
                source = self._source(submitted.get("source"))
                try:
                    source_available(self.root, source["sha256"])
                    files, _ = self._capture(source)
                    inventory = {
                        "available": True,
                        "fingerprint": digest(files),
                        "items": [
                            {
                                "id": item["id"],
                                "kind": "recording-owner-copy",
                                "bytes": item["bytes"],
                            }
                            for item in files
                        ],
                        "operations": [],
                        "mayDeleteOriginal": True,
                    }
                except (
                    ValueError,
                    OSError,
                    transcript_store.TranscriptStoreError,
                ) as error:
                    inventory = {
                        "available": False,
                        "fingerprint": digest([source, str(error)]),
                        "items": [],
                        "operations": [],
                        "mayDeleteOriginal": False,
                        "reason": "Original owner inventory is unavailable or source copies changed",
                    }
                return {"owner": self.owner, "source": source, "inventory": inventory}
            request_id = identifier(submitted.get("requestId"))
            row = self.db.execute(
                "SELECT intent,value FROM jobs WHERE principal=? AND id=?",
                (key, request_id),
            ).fetchone()
            if action == "status":
                if row:
                    prior = json.loads(row[1])
                    if (
                        prior.get("purgeStarted")
                        and prior["receipt"]["phase"] != "purged"
                    ):
                        return self.request(
                            "purge",
                            {
                                "principal": principal,
                                "requestId": request_id,
                                "source": prior["source"],
                                "expectedFingerprint": prior["expectedFingerprint"],
                            },
                            authorization,
                        )
                    if not prior["receipt"]["quiescent"]:
                        with transcript_store.connect(self.root) as con:
                            exists = con.execute(
                                "SELECT 1 FROM sqlite_master WHERE name='recording_source_erasures'"
                            ).fetchone()
                            fence = (
                                con.execute(
                                    "SELECT owner,request FROM recording_source_erasures WHERE sha256=?",
                                    (prior["source"]["sha256"],),
                                ).fetchone()
                                if exists
                                else None
                            )
                        if fence and tuple(fence) == (key, request_id):
                            prior["receipt"].update(status="applied", quiescent=True)
                            self._write(key, request_id, prior)
                    return {
                        "owner": self.owner,
                        "source": prior["source"],
                        "receipt": prior["receipt"],
                    }
                return {
                    "owner": self.owner,
                    "receipt": {"requestId": request_id, "status": "not_found"},
                }
            source = self._source(submitted.get("source"))
            intent = digest([source, submitted.get("expectedFingerprint")])
            if row:
                if row[0] != intent:
                    raise ValueError("speech_owner_request_conflict")
                value = json.loads(row[1])
            else:
                if action != "fence":
                    raise ValueError("speech_owner_fence_required")
                source_available(self.root, source["sha256"])
                files, blobs = self._capture(source)
                if digest(files) != submitted.get("expectedFingerprint"):
                    raise ValueError("speech_owner_preview_stale")
                if self.db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= 4096:
                    raise ValueError("speech_owner_journal_limit")
                value = {
                    "source": source,
                    "files": files,
                    "blobs": blobs,
                    "expectedFingerprint": submitted["expectedFingerprint"],
                    "receipt": {
                        "requestId": request_id,
                        "status": "unknown",
                        "phase": "fenced",
                        "quiescent": False,
                        "privateCleanupVerified": False,
                    },
                }
                self.db.execute(
                    "INSERT INTO jobs VALUES(?,?,?,?)",
                    (key, request_id, intent, canonical(value)),
                )
                self.db.commit()
            if action == "fence":
                if value["receipt"]["status"] != "applied":
                    with transcript_store.connect(self.root) as con:
                        con.execute(
                            "CREATE TABLE IF NOT EXISTS recording_source_erasures(sha256 TEXT PRIMARY KEY,owner TEXT,request TEXT)"
                        )
                        for sha in {
                            source["sha256"],
                            *(item["sha256"] for item in value["files"]),
                        }:
                            con.execute(
                                "INSERT OR IGNORE INTO recording_source_erasures VALUES(?,?,?)",
                                (sha, key, request_id),
                            )
                        existing = con.execute(
                            "SELECT owner,request FROM recording_source_erasures WHERE sha256=?",
                            (source["sha256"],),
                        ).fetchone()
                        if tuple(existing) != (key, request_id):
                            raise ValueError("speech_owner_source_already_fenced")
                    value["receipt"].update(status="applied", quiescent=True)
                    self._write(key, request_id, value)
            elif action == "purge":
                if not value["receipt"]["quiescent"]:
                    raise ValueError("speech_owner_fence_required")
                if value["receipt"]["phase"] != "purged":
                    value["purgeStarted"] = True
                    value["receipt"]["status"] = "unknown"
                    self._write(key, request_id, value)
                    quarantine = self.directory / "quarantine"
                    quarantine.mkdir(mode=0o700, exist_ok=True)
                    if (
                        quarantine.is_symlink()
                        or quarantine.stat().st_uid != os.getuid()
                    ):
                        raise ValueError("speech_owner_original_unavailable")
                    for item in value["files"]:
                        path = Path(item["path"])
                        staged = quarantine / digest([key, request_id, item["id"]])
                        if (
                            not staged.exists()
                            and not staged.is_symlink()
                            and (path.exists() or path.is_symlink())
                        ):
                            if self._file(path) != item:
                                raise ValueError("speech_owner_source_changed")
                            os.replace(path, staged)
                        if staged.exists() or staged.is_symlink():
                            actual = self._file(staged, quarantined=True)
                            if any(
                                actual[field] != item[field]
                                for field in ["sha256", "bytes", "inode", "device"]
                            ):
                                if not path.exists() and not path.is_symlink():
                                    os.link(staged, path, follow_symlinks=False)
                                    staged.unlink()
                                raise ValueError("speech_owner_source_changed")
                            staged.unlink()
                    if self.consumer:
                        self.consumer.erase_source_journal(source["sha256"])
                    with transcript_store.connect(self.root) as con:
                        con.execute(
                            "UPDATE blobs SET original_path='',stored_path='',bytes=0 WHERE sha256=?",
                            (source["sha256"],),
                        )
                    value.pop("files", None)
                    value.pop("blobs", None)
                    value["receipt"].update(
                        status="applied", phase="purged", privateCleanupVerified=True
                    )
                    self._write(key, request_id, value)
            else:
                raise ValueError("speech_owner_invalid_request")
            return {"owner": self.owner, "source": source, "receipt": value["receipt"]}
