"""Selected-sample deletion over the existing immutable A3 custody effect protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from biometric_custody_ledger import BiometricCustodyLedger
from speaker_enrollment_bridge import (
    EnrollmentBridge,
    EnrollmentError,
    _hash,
    _id,
    _json,
    _now,
)
import transcript_store


def _fail(code: str) -> None:
    raise EnrollmentError("deletion_" + code)


class SampleSetCustodyLedger(BiometricCustodyLedger):
    """One exact union, with unchanged A3 quarantine/events/tombstone semantics."""

    def __init__(self, root: Path, *, private_root: Path, sample_ids: tuple[str, ...]):
        super().__init__(root, private_root=private_root)
        self.sample_ids = tuple(sorted(set(sample_ids)))

    def _resolve_effect(self, *, mode: str, target_type: str, target_id: str):
        if target_type != "sample" or not target_id.startswith("sample-set:"):
            return super()._resolve_effect(
                mode=mode, target_type=target_type, target_id=target_id
            )
        if (
            mode != "delete"
            or target_id != "sample-set:" + _hash(self.sample_ids)
            or not self.sample_ids
        ):
            _fail("invalid_preview")
        parts = [
            super(SampleSetCustodyLedger, self)._resolve_effect(
                mode=mode, target_type="sample", target_id=sample_id
            )
            for sample_id in self.sample_ids
        ]
        refs = {
            (ref["object_id"], ref["sha256"]): ref
            for part in parts
            for ref in part["private_refs"]
        }
        # Earlier selected-sample deletions may already have erased a shared
        # invalidated profile. Omit only objects proven absent by its tombstone,
        # never a missing file whose cleanup cannot be established.
        with transcript_store.connect(self.root) as con:
            tombstones = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_biometric_deletion_tombstones"
                )
            ]
        for key, ref in list(refs.items()):
            obj = self.private_root / "objects" / ref["object_id"]
            if not obj.exists():
                known = [
                    row
                    for row in tombstones
                    if ref["sha256"] in json.loads(row["deleted_object_hashes_json"])
                ]
                if known and all(
                    not (
                        self.private_root / "quarantine" / row["id"] / ref["object_id"]
                    ).exists()
                    for row in known
                ):
                    del refs[key]
        return {
            "schema_version": "transcribe-audio.biometric-custody-sample-set-effect.v1",
            "mode": "delete",
            "target_type": "sample",
            "target_id": target_id,
            "sample_ids": list(self.sample_ids),
            "profile_version_ids": sorted(
                {item for part in parts for item in part["profile_version_ids"]}
            ),
            "cluster_version_ids": sorted(
                {item for part in parts for item in part["cluster_version_ids"]}
            ),
            "deleted_object_hashes": sorted({ref["sha256"] for ref in refs.values()}),
            "backup_disposition": "exclude_from_future_backups",
            "private_refs": [refs[key] for key in sorted(refs)],
        }


class SampleDeletionBridge:
    def __init__(self, enrollment: EnrollmentBridge):
        self.enrollment = enrollment
        enrollment.db.executescript("""CREATE TABLE IF NOT EXISTS sample_deletions(id TEXT PRIMARY KEY,app TEXT NOT NULL,digest TEXT NOT NULL,metadata TEXT NOT NULL,snapshot TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sample_deletion_fences(sample TEXT NOT NULL,sha256 TEXT NOT NULL,operation TEXT NOT NULL,app TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(sample,sha256));""")

    def _ids(self, request: Mapping[str, Any]) -> tuple[str, ...]:
        ids = request.get("sample_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 256:
            _fail("invalid_request")
        selected = tuple(sorted({_id(item) for item in ids}))
        if len(selected) != len(ids):
            _fail("invalid_request")
        return selected

    def _collect(self, ids: tuple[str, ...]):
        bridge = self.enrollment
        jobs = [
            json.loads(row[0]) for row in bridge.db.execute("SELECT payload FROM jobs")
        ]
        relevant = [
            job
            for job in jobs
            if any(ref["id"] in ids for ref in job["sample_manifest"])
        ]
        hashes = {
            ref["sha256"]
            for job in relevant
            for ref in job["sample_manifest"]
            if ref["id"] in ids
        }
        with transcript_store.connect(bridge.knowledge_root) as con:
            rows = [
                dict(row)
                for row in con.execute("SELECT * FROM knowledge_voice_samples")
            ]
        matched = [
            row
            for row in rows
            if json.loads(row["preparation_lineage_json"]).get("source_sample_id")
            in ids
            or row["sample_sha256"] in hashes
        ]
        upstream_ids = tuple(sorted(row["id"] for row in matched))
        ledger = SampleSetCustodyLedger(
            bridge.knowledge_root,
            private_root=bridge.private_root,
            sample_ids=upstream_ids,
        )
        preview = (
            ledger.preview_effect(
                mode="delete",
                target_type="sample",
                target_id="sample-set:" + _hash(upstream_ids),
            )
            if upstream_ids
            else None
        )
        resolved = (
            ledger._resolve_effect(
                mode="delete",
                target_type="sample",
                target_id="sample-set:" + _hash(upstream_ids),
            )
            if upstream_ids
            else {"private_refs": [], "profile_version_ids": []}
        )
        objects = {
            (ref["object_id"], ref["sha256"]): ref for ref in resolved["private_refs"]
        }
        # A job intent can precede registration of its copied bytes or profile.
        # Include only exact objects provably tied to those persisted manifests.
        for sha in hashes:
            candidate = bridge.private_root / "objects" / ("sample-" + sha)
            if candidate.exists():
                objects[(candidate.name, sha)] = {
                    "object_id": candidate.name,
                    "sha256": sha,
                }
        manifest_hashes = {job["sample_manifest_sha256"] for job in relevant}
        for candidate in (bridge.private_root / "objects").glob("profile-*"):
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.stat().st_size > 65_536
            ):
                _fail("invalid_storage")
            payload = json.loads(candidate.read_bytes())
            if payload.get("sample_manifest_sha256") in manifest_hashes:
                sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if candidate.name != "profile-" + sha:
                    _fail("object_drift")
                objects[(candidate.name, sha)] = {
                    "object_id": candidate.name,
                    "sha256": sha,
                }
        private_refs = [objects[key] for key in sorted(objects)]
        for ref in private_refs:
            candidate = bridge.private_root / "objects" / ref["object_id"]
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.stat().st_nlink != 1
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != ref["sha256"]
            ):
                _fail("object_drift")
        profiles = [
            {
                "profile_version_id": pid,
                **{
                    k: v
                    for k, v in ledger.profile_state(pid).items()
                    if k in {"status", "person_id", "adapter_id", "model_revision"}
                },
            }
            for pid in resolved["profile_version_ids"]
        ]
        operations = [
            {"id": job["id"], "state": job["state"]}
            for job in relevant
            if job["state"] in {"running", "unknown"}
        ]
        mappings = [
            {
                "id": sample_id,
                "sha256": sorted(
                    {
                        ref["sha256"]
                        for job in relevant
                        for ref in job["sample_manifest"]
                        if ref["id"] == sample_id
                    }
                ),
                "upstream_sample_ids": sorted(
                    row["id"]
                    for row in matched
                    if json.loads(row["preparation_lineage_json"]).get(
                        "source_sample_id"
                    )
                    == sample_id
                ),
            }
            for sample_id in ids
        ]
        public = {
            "sample_ids": list(ids),
            "samples": mappings,
            "profiles": profiles,
            "operations": operations,
            "deleted_object_hashes": sorted({ref["sha256"] for ref in private_refs}),
        }
        snapshot = {
            "public": public,
            "preview": preview,
            "private_refs": private_refs,
            "upstream_sample_ids": list(upstream_ids),
        }
        fingerprint = _hash(snapshot)
        authority = bridge.config.get("deletion_authority_id")
        return {
            **public,
            "fingerprint": fingerprint,
            "available": bool(authority),
            "reason": None if authority else "deletion_authority_unconfigured",
        }, snapshot

    def _row(self, app: str, operation: str):
        row = self.enrollment.db.execute(
            "SELECT * FROM sample_deletions WHERE id=? AND app=?", (_id(operation), app)
        ).fetchone()
        if row is None:
            return None
        return row

    def _persist(self, result: dict[str, Any]):
        self.enrollment.db.execute(
            "UPDATE sample_deletions SET metadata=? WHERE id=?",
            (_json(result), result["operation_id"]),
        )
        self.enrollment.db.commit()

    def _effect(self, operation: str):
        with transcript_store.connect(self.enrollment.knowledge_root) as con:
            return con.execute(
                "SELECT * FROM knowledge_biometric_effect_receipts WHERE idempotency_key=?",
                ("speech-sample-delete:" + operation,),
            ).fetchone()

    def _cleanup(
        self,
        snapshot: Mapping[str, Any],
        effect: Mapping[str, Any] | None,
        *,
        delete_extras: bool,
    ):
        root = self.enrollment.private_root
        expected = {ref["object_id"]: ref["sha256"] for ref in snapshot["private_refs"]}
        if effect:
            quarantine = root / "quarantine" / effect["id"]
            if quarantine.exists():
                if quarantine.is_symlink() or not quarantine.is_dir():
                    _fail("invalid_quarantine")
                for item in quarantine.iterdir():
                    if (
                        item.name not in expected
                        or item.is_symlink()
                        or not item.is_file()
                        or item.stat().st_nlink != 1
                        or hashlib.sha256(item.read_bytes()).hexdigest()
                        != expected[item.name]
                    ):
                        _fail("invalid_quarantine")
                    item.unlink()
                quarantine.rmdir()
        for name, sha in expected.items():
            obj = root / "objects" / name
            if obj.exists():
                if (
                    not delete_extras
                    or obj.is_symlink()
                    or not obj.is_file()
                    or obj.stat().st_nlink != 1
                    or hashlib.sha256(obj.read_bytes()).hexdigest() != sha
                ):
                    _fail("cleanup_unconfirmed")
                obj.unlink()
        if any((root / "objects" / name).exists() for name in expected):
            _fail("cleanup_unconfirmed")

    def _finish(
        self,
        result: dict[str, Any],
        snapshot: Mapping[str, Any],
        effect: Mapping[str, Any] | None,
    ):
        bridge = self.enrollment
        if snapshot["preview"] and not effect:
            result.update(status="unknown", error="deletion_effect_unconfirmed")
            self._persist(result)
            return result
        try:
            # The persisted approved intent scopes orphan-only cleanup to exact,
            # hash-verified private refs; registered effects still require their receipt.
            self._cleanup(snapshot, effect, delete_extras=True)
            result.update(
                status="applied",
                tombstone_id=effect["tombstone_id"] if effect else None,
                private_cleanup_verified=True,
            )
            bridge.db.execute("BEGIN IMMEDIATE")
            affected = set(
                snapshot["public"]["profiles"][i]["profile_version_id"]
                for i in range(len(snapshot["public"]["profiles"]))
            )
            for row in bridge.db.execute("SELECT person,profile FROM heads").fetchall():
                if row["profile"] in affected:
                    bridge.db.execute(
                        "DELETE FROM heads WHERE person=?", (row["person"],)
                    )
            bridge.db.execute(
                "UPDATE sample_deletion_fences SET state='deleted' WHERE operation=?",
                (result["operation_id"],),
            )
            bridge.db.execute(
                "UPDATE sample_deletions SET metadata=? WHERE id=?",
                (_json(result), result["operation_id"]),
            )
            bridge.db.commit()
        except Exception:
            bridge.db.rollback()
            result.update(status="unknown", error="deletion_cleanup_unconfirmed")
            self._persist(result)
        return result

    def request(self, request: Mapping[str, Any]):
        bridge = self.enrollment
        app = bridge._authorize(request)
        action = request.get("action")
        if action == "deletion_preview":
            return self._collect(self._ids(request))[0]
        operation = _id(request.get("operation_id"))
        row = self._row(app, operation)
        if action == "deletion_status":
            if row is None:
                return {"status": "not_found", "operation_id": operation}
            result = json.loads(row["metadata"])
            if result["status"] in {"pending", "unknown"}:
                try:
                    with bridge._worker_lock():
                        return self._finish(
                            result, json.loads(row["snapshot"]), self._effect(operation)
                        )
                except EnrollmentError:
                    return result
            return result
        if action != "deletion_apply" or request.get("approve") is not True:
            _fail("invalid_request")
        ids = self._ids(request)
        intent = _hash(
            {"sample_ids": ids, "fingerprint": request.get("fingerprint"), "app": app}
        )
        if row:
            if row["digest"] != intent:
                _fail("request_conflict")
            # Exact effects are reconciled, never applied twice after uncertainty.
            return self.request(
                {
                    "action": "deletion_status",
                    "operation_id": operation,
                    "tenant": request["tenant"],
                    "app": app,
                }
            )
        if bridge.db.execute(
            "SELECT id FROM sample_deletions WHERE id=?", (operation,)
        ).fetchone():
            _fail("request_conflict")
        authority = bridge.config.get("deletion_authority_id")
        if not isinstance(authority, str) or not authority:
            _fail("authority_unconfigured")
        with bridge._worker_lock():
            current, snapshot = self._collect(ids)
            if current["fingerprint"] != request.get("fingerprint"):
                _fail("preview_stale")
            if current["operations"]:
                _fail("operations_unsettled")
            if bridge.db.execute(
                "SELECT operation FROM sample_deletion_fences WHERE sample IN ("
                + ",".join("?" for _ in ids)
                + ") LIMIT 1",
                ids,
            ).fetchone():
                _fail("reference_locked")
            result = {
                "status": "pending",
                "operation_id": operation,
                "sample_ids": list(ids),
                "fingerprint": current["fingerprint"],
                "profiles": current["profiles"],
                "private_cleanup_verified": False,
            }
            bridge.db.execute("BEGIN IMMEDIATE")
            bridge.db.execute(
                "INSERT INTO sample_deletions VALUES(?,?,?,?,?)",
                (operation, app, intent, _json(result), _json(snapshot)),
            )
            for mapping in current["samples"]:
                for sha in mapping["sha256"] or [""]:
                    bridge.db.execute(
                        "INSERT INTO sample_deletion_fences VALUES(?,?,?,?,?)",
                        (mapping["id"], sha, operation, app, "pending"),
                    )
            bridge.db.commit()
            try:
                if snapshot["preview"]:
                    ledger = SampleSetCustodyLedger(
                        bridge.knowledge_root,
                        private_root=bridge.private_root,
                        sample_ids=tuple(snapshot["upstream_sample_ids"]),
                    )
                    ledger.apply_effect(
                        preview=snapshot["preview"],
                        authority_id=authority,
                        idempotency_key="speech-sample-delete:" + operation,
                        created_at=_now(),
                    )
                self._cleanup(snapshot, self._effect(operation), delete_extras=True)
                return self._finish(result, snapshot, self._effect(operation))
            except Exception:
                result.update(
                    status="unknown", error="deletion_effect_or_cleanup_unconfirmed"
                )
                self._persist(result)
                return result
