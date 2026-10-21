"""Scoped process enrollment for explicitly reviewed, newly supplied samples.

This boundary never borrows campaign authority or discovers private recordings.
Portable JSON contains provenance, not biometric bytes, vectors or private paths.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import struct
import sys
import wave
from datetime import datetime, timezone
from typing import Any, Mapping

from acoustic_verification import adapter_registry, VerificationAdapter
from biometric_custody_ledger import BiometricCustodyLedger
from conversation_knowledge_store import ConversationKnowledgeStore
import transcript_store

SCHEMA = "transcribe-audio.speaker-enrollment-process.v1"
MAX_INPUT = 12_000_000
RECIPE = "pcm16-mono16k-normalized-centroid-v1"


class EnrollmentError(ValueError):
    pass


def _fail(code: str) -> None:
    raise EnrollmentError("enrollment_" + code)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value
    ):
        _fail("invalid_request")
    return value


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _private(path: Path) -> None:
    if (
        path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) & 0o077
    ):
        _fail("invalid_storage")


def _normalized(values: Any, dimension: int) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != dimension or any(not math.isfinite(v) for v in result):
        _fail("invalid_embedding")
    norm = math.sqrt(sum(v * v for v in result))
    if norm < 1e-12 or not math.isfinite(norm):
        _fail("invalid_embedding")
    return tuple(v / norm for v in result)


class EnrollmentBridge:
    """One explicit tenant root, one pinned adapter, immutable A3 custody.

    The adapter seam reuses the upstream VerificationAdapter protocol. The CLI
    accepts only the genuine pinned registry; tests may inject deterministic
    adapters directly without exposing a test-mode wire escape hatch.
    """

    def __init__(self, config: Mapping[str, Any], adapter: VerificationAdapter):
        self.config = dict(config)
        self.tenant = _id(config.get("tenant"))
        self.apps = tuple(_id(v) for v in config.get("apps", []))
        if not self.apps or not isinstance(config.get("authorities"), dict):
            _fail("invalid_configuration")
        self.adapter = adapter
        if adapter.candidate_id != config.get(
            "adapter_id"
        ) or adapter.revision_sha != config.get("model_revision"):
            _fail("model_drift")
        self.root = Path(str(config.get("directory", "")))
        if not self.root.is_absolute():
            _fail("invalid_storage")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _private(self.root)
        marker = self.root / "bridge-owner.json"
        owner = {"schema": SCHEMA, "tenant": self.tenant, "apps": sorted(self.apps)}
        if not marker.exists():
            if any(self.root.iterdir()):
                _fail("unregistered_storage")
            with marker.open("x", encoding="utf8") as handle:
                os.chmod(marker, 0o600)
                handle.write(_json(owner))
                handle.flush()
                os.fsync(handle.fileno())
        if marker.is_symlink() or json.loads(marker.read_text()) != owner:
            _fail("storage_scope_conflict")
        self.private_root = self.root / "private"
        self.private_root.mkdir(mode=0o700, exist_ok=True)
        _private(self.private_root)
        # This root is bridge-owned and created above; it cannot be a live or
        # existing campaign knowledge store silently selected by configuration.
        self.knowledge_root = self.root / "knowledge"
        if self.knowledge_root.is_symlink():
            _fail("invalid_storage")
        store = ConversationKnowledgeStore(self.knowledge_root)
        store.migrate(backup=False)
        self.custody = BiometricCustodyLedger(
            self.knowledge_root, private_root=self.private_root
        )
        journal = self.root / "bridge.sqlite"
        if journal.is_symlink():
            _fail("invalid_storage")
        self.db = sqlite3.connect(journal, timeout=5)
        self.db.row_factory = sqlite3.Row
        os.chmod(journal, 0o600)
        self.db.executescript("""PRAGMA secure_delete=ON;
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,app TEXT NOT NULL,digest TEXT NOT NULL,payload TEXT NOT NULL,cancel_requested INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS heads(person TEXT PRIMARY KEY,profile TEXT NOT NULL,job TEXT NOT NULL);
        """)

    def close(self) -> None:
        self.db.close()

    def _authorize(self, request: Mapping[str, Any]) -> str:
        if request.get("tenant") != self.tenant or request.get("app") not in self.apps:
            _fail("forbidden")
        return str(request["app"])

    @contextlib.contextmanager
    def _worker_lock(self):
        filename = self.root / "worker.lock"
        if filename.is_symlink():
            _fail("invalid_storage")
        fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                _fail("busy")
            yield
        finally:
            os.close(fd)

    def _persist(self, job: dict[str, Any]) -> None:
        self.db.execute("UPDATE jobs SET payload=? WHERE id=?", (_json(job), job["id"]))
        self.db.commit()

    def _recover_interrupted(self, job: dict[str, Any]) -> dict[str, Any]:
        candidate = job.get("candidate_profile_version_id")
        if candidate:
            try:
                state = self.custody.profile_state(candidate)
                with transcript_store.connect(self.knowledge_root) as con:
                    event = con.execute(
                        "SELECT id FROM knowledge_voice_profile_events WHERE idempotency_key=? AND profile_version_id=? AND action='activate'",
                        ("bridge-activate-" + job["id"], candidate),
                    ).fetchone()
                digest = job["candidate_profile_sha256"]
                obj = self.private_root / "objects" / ("profile-" + digest)
                if (
                    event
                    and state["status"] == "active"
                    and state["event_id"] == event[0]
                    and not obj.is_symlink()
                    and hashlib.sha256(obj.read_bytes()).hexdigest() == digest
                ):
                    self.db.execute("BEGIN IMMEDIATE")
                    head = self.db.execute(
                        "SELECT profile FROM heads WHERE person=?",
                        (job["person"]["id"],),
                    ).fetchone()
                    if (head[0] if head else None) not in (
                        job["predecessor_profile_version_id"],
                        candidate,
                    ):
                        self.db.rollback()
                        _fail("predecessor_conflict")
                    job.update(
                        state="completed",
                        profile_version_id=candidate,
                        profile_sha256=digest,
                    )
                    self.db.execute(
                        "INSERT OR REPLACE INTO heads(person,profile,job) VALUES(?,?,?)",
                        (job["person"]["id"], candidate, job["id"]),
                    )
                    self.db.execute(
                        "UPDATE jobs SET payload=? WHERE id=?", (_json(job), job["id"])
                    )
                    self.db.commit()
                    return job
            except Exception:
                self.db.rollback()
        job.update(state="unknown", error="enrollment_interrupted_inspect_custody")
        self._persist(job)
        return job

    def _status(self, app: str, job_id: str) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT * FROM jobs WHERE id=? AND app=?", (_id(job_id), app)
        ).fetchone()
        if row is None:
            _fail("not_found")
        job = json.loads(row["payload"])
        if job["state"] in {"running", "unknown"}:
            try:
                with self._worker_lock():
                    # Re-read under ownership: a prior worker may have completed
                    # after our first read, before it released its lock.
                    latest = self.db.execute(
                        "SELECT payload FROM jobs WHERE id=?", (job_id,)
                    ).fetchone()
                    job = json.loads(latest["payload"])
                    if job["state"] in {"running", "unknown"}:
                        job = self._recover_interrupted(job)
            except EnrollmentError as exc:
                if str(exc) != "enrollment_busy":
                    raise
        return job

    def _reference_blocked(self, sample_id: str, sha256: str) -> bool:
        exists = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_deletion_fences'"
        ).fetchone()
        return bool(
            exists
            and self.db.execute(
                "SELECT 1 FROM sample_deletion_fences WHERE sample=? OR (sha256<>'' AND sha256=?) LIMIT 1",
                (sample_id, sha256),
            ).fetchone()
        )

    def _samples(self, request: Mapping[str, Any]):
        samples = request.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 8:
            _fail("sample_limits")
        refs, decoded = [], []
        for sample in samples:
            if not isinstance(sample, dict) or not isinstance(
                sample.get("source"), dict
            ):
                _fail("invalid_sample")
            source = sample["source"]
            _id(sample.get("id"))
            _id(source.get("recording_id"))
            if not isinstance(sample.get("revision"), (str, int)) or isinstance(
                sample.get("revision"), bool
            ):
                _fail("invalid_sample")
            for digest in (sample.get("sha256"), source.get("sha256")):
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[a-f0-9]{64}", digest
                ):
                    _fail("invalid_sample")
            start, end = source.get("start_ms"), source.get("end_ms")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or not 500 <= end - start <= 30_000
            ):
                _fail("sample_limits")
            if self._reference_blocked(sample["id"], sample["sha256"]):
                _fail("reference_deleted")
            encoded = sample.get("data_base64")
            if not isinstance(encoded, str) or len(encoded) > 1_300_000:
                _fail("sample_limits")
            try:
                data = base64.b64decode(encoded, validate=True)
                if hashlib.sha256(data).hexdigest() != sample["sha256"]:
                    _fail("sample_hash_mismatch")
                with wave.open(io.BytesIO(data), "rb") as reader:
                    if (
                        reader.getnchannels(),
                        reader.getsampwidth(),
                        reader.getframerate(),
                        reader.getcomptype(),
                    ) != (1, 2, 16_000, "NONE"):
                        _fail("invalid_sample")
                    count = reader.getnframes()
                    if count != (end - start) * 16:
                        _fail("sample_duration_mismatch")
                    pcm = reader.readframes(count)
                    if len(pcm) != count * 2:
                        _fail("invalid_sample")
                values = tuple(v[0] / 32768 for v in struct.iter_unpack("<h", pcm))
            except EnrollmentError:
                raise
            except Exception:
                _fail("invalid_sample")
            refs.append(
                {
                    "id": sample["id"],
                    "revision": sample["revision"],
                    "sha256": sample["sha256"],
                    "source": {
                        k: source[k]
                        for k in ("recording_id", "sha256", "start_ms", "end_ms")
                    },
                }
            )
            decoded.append((sample["id"], data, values))
        if len({ref["id"] for ref in refs}) != len(refs):
            _fail("invalid_sample")
        return sorted(refs, key=lambda ref: ref["id"]), decoded

    def _authority(self, request: Mapping[str, Any]):
        person, authority = request.get("person"), request.get("authority")
        if not isinstance(person, dict) or not isinstance(authority, dict):
            _fail("authority_required")
        _id(person.get("id"))
        revision = person.get("revision")
        configured = self.config["authorities"].get(person["id"])
        if (
            not configured
            or not isinstance(revision, str)
            or not revision
            or len(revision) > 1000
            or authority.get("reviewed_identity") is not True
            or authority.get("retained") is not True
        ):
            _fail("authority_required")
        if configured != {
            "person_revision": revision,
            "review_id": authority.get("review_id"),
            "enrollment_id": authority.get("enrollment_id"),
        }:
            _fail("authority_drift")
        _id(authority.get("review_id"))
        _id(authority.get("enrollment_id"))
        return {"id": person["id"], "revision": revision}, dict(authority)

    def _cancelled(self, job_id: str) -> bool:
        return bool(
            self.db.execute(
                "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()[0]
        )

    def _enroll(self, request: Mapping[str, Any], app: str) -> dict[str, Any]:
        person, authority = self._authority(request)
        manifest, decoded = self._samples(request)
        job_id = _id(request.get("job_id"))
        digest = _hash({k: request[k] for k in sorted(request)})
        predecessor = request.get("expected_predecessor")
        if predecessor is not None:
            _id(predecessor)
        with self._worker_lock():
            old = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old:
                if old["app"] != app or old["digest"] != digest:
                    _fail("request_conflict")
                job = json.loads(old["payload"])
                if job["state"] == "running":
                    job = self._recover_interrupted(job)
                return job
            if self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= 256:
                _fail("quota_exceeded")
            if any(
                json.loads(row[0])["state"] in {"running", "unknown"}
                for row in self.db.execute("SELECT payload FROM jobs")
            ):
                _fail("busy_or_unknown")
            head = self.db.execute(
                "SELECT profile FROM heads WHERE person=?", (person["id"],)
            ).fetchone()
            if (head[0] if head else None) != predecessor:
                _fail("predecessor_conflict")
            retained = sum(
                p.stat().st_size
                for p in (self.private_root / "objects").iterdir()
                if p.is_file()
            )
            maximum = self.config.get("maximum_stored_bytes", 128_000_000)
            if (
                type(maximum) is not int
                or maximum < 1
                or retained + sum(len(data) for _, data, _ in decoded) + 65_536
                > maximum
            ):
                _fail("quota_exceeded")
            job = {
                "schema_version": SCHEMA,
                "id": job_id,
                "state": "running",
                "person": person,
                "adapter_id": self.adapter.candidate_id,
                "model_revision": self.adapter.revision_sha,
                "recipe_revision": RECIPE,
                "sample_manifest": manifest,
                "sample_manifest_sha256": _hash(manifest),
                "predecessor_profile_version_id": predecessor,
                "qualified": False,
            }
            self.db.execute(
                "INSERT INTO jobs(id,app,digest,payload) VALUES(?,?,?,?)",
                (job_id, app, digest, _json(job)),
            )
            self.db.commit()  # Exact job and authority digest precede private effects.
            try:
                activation_started = False
                vectors, sample_ids = [], []
                created = _now()
                for sample_id, data, values in decoded:
                    if self._cancelled(job_id):
                        job["state"] = "cancelled"
                        self._persist(job)
                        return job
                    ref = next(ref for ref in manifest if ref["id"] == sample_id)
                    obj = self.custody.store_private_object(
                        object_id="sample-" + ref["sha256"], payload=data
                    )
                    sample = self.custody.register_sample(
                        conversation_id="speech:" + ref["source"]["recording_id"],
                        recording_id=ref["source"]["recording_id"],
                        speaker_ref=person["id"],
                        start_ms=ref["source"]["start_ms"],
                        end_ms=ref["source"]["end_ms"],
                        source_media_sha256=ref["source"]["sha256"],
                        sample_sha256=ref["sha256"],
                        quality={
                            "eligible": True,
                            "reason_codes": ["operator_reviewed_reference"],
                        },
                        preparation_lineage={
                            "recipe_version": RECIPE,
                            "source_sample_id": sample_id,
                            "source_sample_revision": ref["revision"],
                            "person_revision": person["revision"],
                        },
                        review_state="reviewed",
                        person_id=person["id"],
                        review_authority_id=authority["review_id"],
                        consent_authority=authority["enrollment_id"],
                        private_object_id=obj.object_id,
                        private_object_sha256=obj.sha256,
                        created_at=created,
                    )
                    sample_ids.append(sample.sample_id)
                    vectors.append(
                        _normalized(
                            self.adapter.embed(values, sample_rate=16_000),
                            self.adapter.embedding_dimension,
                        )
                    )
                centroid = _normalized(
                    [
                        sum(v[i] for v in vectors) / len(vectors)
                        for i in range(self.adapter.embedding_dimension)
                    ],
                    self.adapter.embedding_dimension,
                )
                family = self.custody.register_profile_family(
                    person_id=person["id"],
                    family_key=self.adapter.candidate_id,
                    conditions={
                        "adapter_id": self.adapter.candidate_id,
                        "recipe_revision": RECIPE,
                    },
                    created_at=created,
                )
                payload = _json(
                    {
                        "schema": SCHEMA,
                        "adapter_id": self.adapter.candidate_id,
                        "model_revision": self.adapter.revision_sha,
                        "person": person,
                        "sample_manifest_sha256": job["sample_manifest_sha256"],
                        "centroid": centroid,
                    }
                ).encode()
                obj = self.custody.store_private_object(
                    object_id="profile-" + hashlib.sha256(payload).hexdigest(),
                    payload=payload,
                )
                version = self.custody.build_profile_version(
                    profile_family_id=family.profile_family_id,
                    sample_ids=tuple(sample_ids),
                    evaluation_id="structural-check-" + job_id,
                    model_revision=self.adapter.revision_sha,
                    recipe_revision=RECIPE,
                    private_object_id=obj.object_id,
                    private_object_sha256=obj.sha256,
                    created_at=created,
                    predecessor_profile_version_id=predecessor or "",
                )
                job.update(
                    candidate_profile_version_id=version.profile_version_id,
                    candidate_profile_sha256=obj.sha256,
                )
                self._persist(job)
                # Activation is a custody event, not calibrated identity acceptance.
                # The public selected head changes only with the terminal receipt.
                self.db.execute("BEGIN IMMEDIATE")
                if self._cancelled(job_id):
                    self.db.rollback()
                    job["state"] = "cancelled"
                    self._persist(job)
                    return job
                activation_started = True
                self.custody.record_profile_event(
                    profile_version_id=version.profile_version_id,
                    action="activate",
                    reason_code="structural_profile_check_only_not_calibrated",
                    authority_id=authority["enrollment_id"],
                    idempotency_key="bridge-activate-" + job_id,
                    created_at=created,
                )
                job.update(
                    state="completed",
                    profile_version_id=version.profile_version_id,
                    profile_sha256=obj.sha256,
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO heads(person,profile,job) VALUES(?,?,?)",
                    (person["id"], version.profile_version_id, job_id),
                )
                self.db.execute(
                    "UPDATE jobs SET payload=? WHERE id=?", (_json(job), job_id)
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                job.update(
                    state="unknown" if activation_started else "failed",
                    error="enrollment_promotion_unknown"
                    if activation_started
                    else "enrollment_build_failed",
                )
                self._persist(job)
            return job

    def request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            _fail("invalid_request")
        app = self._authorize(request)
        action = request.get("action")
        if action in {"deletion_preview", "deletion_apply", "deletion_status"}:
            from speaker_sample_deletion import SampleDeletionBridge

            return SampleDeletionBridge(self).request(request)
        if action == "readiness":
            try:
                from acoustic_verification import _verified_model_artifacts

                root = Path(str(self.config["model_snapshot_root"])).parent
                _verified_model_artifacts(root, self.adapter.candidate_id)
                available = True
            except Exception:
                available = False
            return {
                "schema_version": SCHEMA,
                "configured": True,
                "available": available,
                "adapter_id": self.adapter.candidate_id,
                "model_revision": self.adapter.revision_sha,
                "qualified": False,
            }
        if action == "enroll":
            return self._enroll(request, app)
        if action in {"status", "cancel"}:
            job_id = _id(request.get("job_id"))
            job = self._status(app, job_id)
            if action == "cancel" and job["state"] == "running":
                self.db.execute(
                    "UPDATE jobs SET cancel_requested=1 WHERE id=? AND app=?",
                    (job_id, app),
                )
                self.db.commit()
                return {**job, "cancellation": "waiting_for_quiescence"}
            return job
        if action == "profiles":
            profiles = []
            for row in self.db.execute(
                "SELECT jobs.payload FROM heads JOIN jobs ON jobs.id=heads.job WHERE jobs.app=?",
                (app,),
            ):
                job = json.loads(row[0])
                current = self.config["authorities"].get(job["person"]["id"], {})
                profiles.append(
                    {
                        **job,
                        "active": True,
                        "stale": current.get("person_revision")
                        != job["person"]["revision"],
                    }
                )
            return {"profiles": profiles}
        _fail("invalid_request")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid is not None:
        # Set the death signal before any custody/inference effect, then check
        # the parent again to close the fork-to-prctl orphaning race.
        if sys.platform != "linux" or args.parent_pid < 1:
            print(_json({"ok": False, "error": "enrollment_parent_guard_unavailable"}))
            return 1
        libc = ctypes.CDLL(None, use_errno=True)
        if (
            libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0
            or os.getppid() != args.parent_pid
        ):
            print(_json({"ok": False, "error": "enrollment_parent_guard_unavailable"}))
            return 1
    bridge = None
    try:
        if (
            args.config.is_symlink()
            or not args.config.is_file()
            or stat.S_IMODE(args.config.stat().st_mode) & 0o077
        ):
            _fail("invalid_configuration")
        config = json.loads(args.config.read_text())
        raw = sys.stdin.buffer.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            _fail("input_limit")
        request = json.loads(raw)
        if (
            not isinstance(request, dict)
            or request.get("tenant") != config.get("tenant")
            or request.get("app") not in config.get("apps", [])
        ):
            _fail("forbidden")
        adapter = adapter_registry(snapshot_root=Path(config["model_snapshot_root"]))[
            config["adapter_id"]
        ]
        bridge = EnrollmentBridge(config, adapter)
        # Optional native libraries must never contaminate protocol stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = bridge.request(request)
        print(_json({"ok": True, "result": result}))
        return 0
    except Exception as exc:
        code = (
            str(exc)
            if isinstance(exc, EnrollmentError)
            else "enrollment_request_failed"
        )
        print(_json({"ok": False, "error": code}))
        return 1
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
