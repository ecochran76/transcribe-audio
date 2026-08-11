#!/usr/bin/env python3
"""Reconcile the bounded local transcript-identity backfill caused by Plan 0065 D2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from acoustic_audio_derivatives import (
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    sha256_file,
    write_immutable_private_json,
)
import speaker_identity_plan0065_d0 as d0
import speaker_identity_plan0065_d1 as d1
import speaker_identity_plan0065_d2 as d2


SCHEMA = "transcribe-audio.plan0065-d2-local-reconciliation.v1"
DEFAULT_RUNTIME_ROOT = d0.DEFAULT_RUNTIME_ROOT
DEFAULT_STORE_ROOT = Path("~/.transcripts")
D1_POLICY_SHA256 = d2.D1_POLICY_SHA256
D2_ACTIVATION_SHA256 = (
    "ef76ba3392ca28a27c695e547765cf03ef2ea062d0d8bc67292549d182009959"
)
D2_RECEIPT_SHA256 = (
    "8d65f6be10259cd54a8e1c8bb3112dcd7db4c9838ca70a89daadfda509e86ad7"
)


class Plan0065ReconciliationError(ValueError):
    """Raised when the bounded D2 transcript restoration cannot be proven exact."""


@dataclass(frozen=True)
class RestorationTarget:
    document_id: str
    expected_sha256: str
    stored_path: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_content(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": canonical_artifact_hash(core)}


def _validate_content(value: Mapping[str, Any], *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_artifact_hash(core):
        raise Plan0065ReconciliationError(f"{label} content hash drifted.")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _database_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _read_object_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Plan0065ReconciliationError(f"{label} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise Plan0065ReconciliationError(f"{label} is not a JSON object.")
    return payload


def reconstruct_legacy_transcript_bytes(
    current_bytes: bytes,
    *,
    expected_sha256: str,
) -> bytes:
    """Remove only D2's lazy identity fields and prove the frozen legacy bytes."""

    payload = _read_object_bytes(current_bytes, label="Current transcript artifact")
    if (
        payload.get("schema_version") != 2
        or not str(payload.get("conversation_id") or "").strip()
        or not str(payload.get("recording_id") or "").strip()
    ):
        raise Plan0065ReconciliationError(
            "Transcript does not contain the exact version-2 identity backfill shape."
        )
    restored = dict(payload)
    restored.pop("conversation_id")
    restored.pop("recording_id")
    restored["schema_version"] = 1
    restored_bytes = _json_bytes(restored)
    if _sha256_bytes(restored_bytes) != expected_sha256:
        raise Plan0065ReconciliationError(
            "Removing only the identity backfill does not reproduce frozen authority."
        )
    return restored_bytes


def _atomic_write_bytes(path: Path, value: bytes, *, mode: int) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        try:
            Path(temporary).unlink()
        except OSError:
            pass
        raise


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    path.chmod(0o700)


def _write_private_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        raise Plan0065ReconciliationError("A reconciliation backup already exists.")
    path.write_bytes(value)
    path.chmod(0o600)


def reconcile_targets(
    *,
    targets: Iterable[RestorationTarget],
    database_path: Path,
    backup_dir: Path,
    restored_at: str,
) -> dict[str, Any]:
    """Restore exact copies and synchronize their rows with rollback on failure."""

    target_rows = tuple(targets)
    if not target_rows or len({item.document_id for item in target_rows}) != len(
        target_rows
    ):
        raise Plan0065ReconciliationError(
            "Reconciliation targets must be non-empty and document-unique."
        )
    database = database_path.expanduser().resolve(strict=True)
    plans: list[dict[str, Any]] = []
    with sqlite3.connect(database) as con:
        con.row_factory = sqlite3.Row
        for target in target_rows:
            row = con.execute(
                """
                SELECT id, source_path, stored_path, artifact_sha256,
                       json_payload, metadata_json, updated_at
                FROM documents WHERE id = ?
                """,
                (target.document_id,),
            ).fetchone()
            if row is None:
                raise Plan0065ReconciliationError(
                    "A reconciliation target is absent from the transcript index."
                )
            stored = Path(str(row["stored_path"])).expanduser().resolve(strict=True)
            if stored != target.stored_path.expanduser().resolve(strict=True):
                raise Plan0065ReconciliationError(
                    "The frozen stored artifact path differs from the index."
                )
            current = stored.read_bytes()
            current_sha = _sha256_bytes(current)
            if current_sha != str(row["artifact_sha256"]):
                raise Plan0065ReconciliationError(
                    "The current stored transcript and index hash differ."
                )
            restored = reconstruct_legacy_transcript_bytes(
                current,
                expected_sha256=target.expected_sha256,
            )
            current_payload = _read_object_bytes(current, label="Indexed transcript")
            try:
                indexed_payload = json.loads(str(row["json_payload"]))
            except json.JSONDecodeError as exc:
                raise Plan0065ReconciliationError(
                    "The indexed transcript payload is invalid."
                ) from exc
            if indexed_payload != current_payload:
                raise Plan0065ReconciliationError(
                    "The indexed transcript payload differs from the artifact."
                )
            paths = [("stored", stored)]
            source_text = str(row["source_path"] or "").strip()
            if source_text:
                source = Path(source_text).expanduser()
                if source.is_file():
                    source = source.resolve(strict=True)
                    if source != stored:
                        if source.read_bytes() != current:
                            raise Plan0065ReconciliationError(
                                "The source and stored D2-mutated copies differ."
                            )
                        paths.append(("source", source))
            plans.append(
                {
                    "target": target,
                    "row": dict(row),
                    "current": current,
                    "current_sha256": current_sha,
                    "restored": restored,
                    "restored_payload": _read_object_bytes(
                        restored, label="Restored transcript"
                    ),
                    "paths": paths,
                }
            )

    _private_directory(backup_dir)
    backup_rows = []
    backup_files = []
    for plan in plans:
        backup_rows.append(plan["row"])
        for role, path in plan["paths"]:
            backup_path = backup_dir / (
                f"{plan['target'].document_id}-{role}-"
                f"{plan['current_sha256'][:24]}.transcript.json"
            )
            _write_private_bytes(backup_path, plan["current"])
            backup_files.append(
                {
                    "document_id": plan["target"].document_id,
                    "role": role,
                    "backup_path": str(backup_path),
                    "sha256": sha256_file(backup_path),
                }
            )
    row_backup_path = backup_dir / "database-rows-before.json"
    _write_private_bytes(row_backup_path, _json_bytes({"rows": backup_rows}))

    applied_paths: list[tuple[Path, bytes, int]] = []
    try:
        for plan in plans:
            for _role, path in plan["paths"]:
                mode = path.stat().st_mode & 0o777
                applied_paths.append((path, plan["current"], mode))
                _atomic_write_bytes(path, plan["restored"], mode=mode)
        with sqlite3.connect(database) as con:
            con.execute("BEGIN IMMEDIATE")
            for plan in plans:
                con.execute(
                    """
                    UPDATE documents
                    SET artifact_sha256 = ?, json_payload = ?, updated_at = ?
                    WHERE id = ? AND artifact_sha256 = ?
                    """,
                    (
                        plan["target"].expected_sha256,
                        _database_json(plan["restored_payload"]),
                        restored_at,
                        plan["target"].document_id,
                        plan["current_sha256"],
                    ),
                )
                if con.execute("SELECT changes()").fetchone()[0] != 1:
                    raise Plan0065ReconciliationError(
                        "A transcript index row changed during reconciliation."
                    )
            con.commit()
    except Exception:
        for path, original, mode in reversed(applied_paths):
            _atomic_write_bytes(path, original, mode=mode)
        with sqlite3.connect(database) as con:
            for plan in plans:
                row = plan["row"]
                con.execute(
                    """
                    UPDATE documents
                    SET artifact_sha256 = ?, json_payload = ?, metadata_json = ?,
                        updated_at = ? WHERE id = ?
                    """,
                    (
                        row["artifact_sha256"],
                        row["json_payload"],
                        row["metadata_json"],
                        row["updated_at"],
                        row["id"],
                    ),
                )
            con.commit()
        raise

    with sqlite3.connect(database) as con:
        con.row_factory = sqlite3.Row
        for plan in plans:
            expected = plan["target"].expected_sha256
            if any(sha256_file(path) != expected for _role, path in plan["paths"]):
                raise Plan0065ReconciliationError(
                    "A restored transcript copy failed exact hash verification."
                )
            row = con.execute(
                "SELECT artifact_sha256, json_payload, updated_at FROM documents WHERE id = ?",
                (plan["target"].document_id,),
            ).fetchone()
            if (
                row is None
                or row["artifact_sha256"] != expected
                or json.loads(row["json_payload"]) != plan["restored_payload"]
                or row["updated_at"] != restored_at
            ):
                raise Plan0065ReconciliationError(
                    "A reconciled transcript index row failed verification."
                )
    return {
        "restored_document_count": len(plans),
        "restored_artifact_copy_count": sum(len(plan["paths"]) for plan in plans),
        "restored_database_row_count": len(plans),
        "backup_row_file": str(row_backup_path),
        "backup_row_file_sha256": sha256_file(row_backup_path),
        "backup_files": backup_files,
        "targets": [
            {
                "document_id": plan["target"].document_id,
                "expected_sha256": plan["target"].expected_sha256,
                "stored_path": str(plan["target"].stored_path),
                "restored_paths": [str(path) for _role, path in plan["paths"]],
            }
            for plan in plans
        ],
    }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _paths(runtime_root: Path) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    run = root / f"d2-reconciliation-{D2_RECEIPT_SHA256[:24]}"
    return {
        "root": root,
        "run": run,
        "backup": run / "before",
        "receipt": run / "receipt.json",
    }


def _authorities(runtime_root: Path) -> tuple[dict[str, Any], ...]:
    plan64 = d0._plan0064_paths(d0.DEFAULT_PLAN0064_ROOT)
    p0_manifest = read_private_object(plan64["p0"] / "private-manifest.json")
    _validate_content(p0_manifest, label="Plan 0064 P0 manifest")
    if p0_manifest.get("content_sha256") != d0.P0_CONTENT_SHA256:
        raise Plan0065ReconciliationError("Plan 0064 P0 authority drifted.")
    d1_paths = d1._paths(runtime_root, D1_POLICY_SHA256)
    d1_evidence = read_private_object(d1_paths["evidence"])
    d1_receipt = read_private_object(d1_paths["receipt"])
    _validate_content(d1_evidence, label="Plan 0065 D1 evidence")
    _validate_content(d1_receipt, label="Plan 0065 D1 receipt")
    if d1_receipt.get("evidence_content_sha256") != d1_evidence.get(
        "content_sha256"
    ):
        raise Plan0065ReconciliationError("Plan 0065 D1 evidence binding drifted.")
    d2_paths = d2._execution_paths(runtime_root, D2_ACTIVATION_SHA256)
    d2_receipt = read_private_object(d2_paths["receipt"])
    _validate_content(d2_receipt, label="Plan 0065 D2 receipt")
    if d2_receipt.get("content_sha256") != D2_RECEIPT_SHA256:
        raise Plan0065ReconciliationError("Plan 0065 D2 authority drifted.")
    return p0_manifest, d1_evidence, d1_receipt, d2_receipt


def _discover_targets(
    p0_manifest: Mapping[str, Any],
    d1_evidence: Mapping[str, Any],
) -> tuple[RestorationTarget, ...]:
    exact_at_d1: dict[str, str] = {}
    for row in d1_evidence.get("development_rows") or []:
        document_id = str(row.get("speaker_ref") or "").split("::", 1)[0]
        audit = row.get("probe_audit") or {}
        expected = str(audit.get("transcript_expected_file_sha256") or "")
        if audit.get("transcript_hash_matches") is True and expected:
            prior = exact_at_d1.setdefault(document_id, expected)
            if prior != expected:
                raise Plan0065ReconciliationError(
                    "D1 has conflicting transcript authority for one document."
                )
    targets = []
    for item in p0_manifest.get("evaluation_cohort", {}).get("considered") or []:
        document_id = str(item.get("document_id") or "")
        if document_id not in exact_at_d1:
            continue
        artifact = item.get("transcript_artifact") or {}
        path = Path(str(artifact.get("path") or "")).expanduser()
        expected = exact_at_d1[document_id]
        if str(item.get("artifact_sha256") or "") != expected:
            raise Plan0065ReconciliationError(
                "P0 and D1 transcript hashes disagree for an exact D1 artifact."
            )
        if not path.is_file():
            raise Plan0065ReconciliationError(
                "An exact-at-D1 transcript artifact is unavailable."
            )
        if sha256_file(path) != expected:
            targets.append(
                RestorationTarget(
                    document_id=document_id,
                    expected_sha256=expected,
                    stored_path=path,
                )
            )
    if len(targets) != 3:
        raise Plan0065ReconciliationError(
            "The bounded D2 identity-backfill denominator is not exactly three."
        )
    return tuple(targets)


def execute_reconciliation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root)
    if paths["receipt"].exists():
        return replay_reconciliation(runtime_root=runtime_root, store_root=store_root)
    if paths["run"].exists():
        raise Plan0065ReconciliationError(
            "A partial Plan 0065 reconciliation directory already exists."
        )
    p0_manifest, d1_evidence, d1_receipt, d2_receipt = _authorities(runtime_root)
    targets = _discover_targets(p0_manifest, d1_evidence)
    ensure_private_tree(paths["root"], paths["run"])
    audit = reconcile_targets(
        targets=targets,
        database_path=store_root.expanduser() / "transcripts.sqlite3",
        backup_dir=paths["backup"],
        restored_at=_utc_now(),
    )
    receipt = _canonical_content(
        {
            "schema_version": SCHEMA,
            "status": "d2_local_identity_metadata_reconciled",
            "p0_content_sha256": p0_manifest["content_sha256"],
            "d1_evidence_content_sha256": d1_evidence["content_sha256"],
            "d1_receipt_content_sha256": d1_receipt["content_sha256"],
            "d2_receipt_content_sha256": d2_receipt["content_sha256"],
            "reconciliation": audit,
            "effect_accounting": {
                "d2_identity_container_mutation_document_count": audit[
                    "restored_document_count"
                ],
                "restored_local_artifact_copy_count": audit[
                    "restored_artifact_copy_count"
                ],
                "reconciled_transcript_index_row_count": audit[
                    "restored_database_row_count"
                ],
                "lasting_identity_container_mutation_count": 0,
                "speaker_assignments": 0,
                "new_enrollments": 0,
                "profile_mutations": 0,
                "reference_mutations": 0,
                "threshold_default_changes": 0,
                "knowledge_writes": 0,
                "graphiti_writes": 0,
                "provider_writes": 0,
                "external_writes": 0,
            },
        }
    )
    write_immutable_private_json(paths["receipt"], receipt)
    return {
        **receipt,
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": False,
    }


def replay_reconciliation(
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> dict[str, Any]:
    paths = _paths(runtime_root)
    require_private_file(paths["receipt"], paths["root"])
    receipt = read_private_object(paths["receipt"])
    _validate_content(receipt, label="Plan 0065 reconciliation receipt")
    p0_manifest, d1_evidence, d1_receipt, d2_receipt = _authorities(runtime_root)
    if (
        receipt.get("p0_content_sha256") != p0_manifest.get("content_sha256")
        or receipt.get("d1_evidence_content_sha256")
        != d1_evidence.get("content_sha256")
        or receipt.get("d1_receipt_content_sha256")
        != d1_receipt.get("content_sha256")
        or receipt.get("d2_receipt_content_sha256")
        != d2_receipt.get("content_sha256")
    ):
        raise Plan0065ReconciliationError(
            "Plan 0065 reconciliation authority binding drifted."
        )
    database = store_root.expanduser() / "transcripts.sqlite3"
    with sqlite3.connect(database) as con:
        con.row_factory = sqlite3.Row
        for target in receipt.get("reconciliation", {}).get("targets") or []:
            expected = str(target.get("expected_sha256") or "")
            for path_text in target.get("restored_paths") or []:
                if sha256_file(Path(path_text)) != expected:
                    raise Plan0065ReconciliationError(
                        "A reconciled transcript artifact drifted."
                    )
            row = con.execute(
                "SELECT artifact_sha256, json_payload FROM documents WHERE id = ?",
                (str(target.get("document_id") or ""),),
            ).fetchone()
            if row is None or row["artifact_sha256"] != expected:
                raise Plan0065ReconciliationError(
                    "A reconciled transcript index row drifted."
                )
            payload = json.loads(row["json_payload"])
            if (
                payload.get("schema_version") != 1
                or "conversation_id" in payload
                or "recording_id" in payload
            ):
                raise Plan0065ReconciliationError(
                    "A reconciled transcript index payload drifted."
                )
    for backup in receipt.get("reconciliation", {}).get("backup_files") or []:
        path = Path(str(backup.get("backup_path") or ""))
        require_private_file(path, paths["root"])
        if sha256_file(path) != backup.get("sha256"):
            raise Plan0065ReconciliationError("A reconciliation backup drifted.")
    row_backup = Path(
        str(receipt.get("reconciliation", {}).get("backup_row_file") or "")
    )
    require_private_file(row_backup, paths["root"])
    if sha256_file(row_backup) != receipt.get("reconciliation", {}).get(
        "backup_row_file_sha256"
    ):
        raise Plan0065ReconciliationError("The reconciliation row backup drifted.")
    return {
        **receipt,
        "private_receipt_path": str(paths["receipt"]),
        "idempotent_replay": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "replay"))
    args = parser.parse_args()
    result = (
        execute_reconciliation()
        if args.mode == "execute"
        else replay_reconciliation()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
