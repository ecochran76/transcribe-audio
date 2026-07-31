"""Private, reference-only biometric enrollment authority for Plan 0037 P3.

P3 records explicit biometric-purpose approvals and source segment references.
It never opens audio, computes embeddings, or creates scoring profiles. P4 may
materialize an eligible immutable generation and must register its descendant
so P3 lifecycle revocation remains authoritative.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from acoustic_audio_derivatives import (
    AudioDerivativeError,
    canonical_artifact_hash,
    ensure_private_tree,
    read_private_object,
    require_private_file,
    resolve_derivative_lineage_receipt,
    sha256_file,
    utc_now,
    write_immutable_private_json,
)
from acoustic_speech_preparation import (
    SpeechPreparationError,
    resolve_comparison_lineage_receipt,
)


REFERENCE_SCHEMA = "transcribe-audio.biometric-reference-profile.v1"
TOMBSTONE_SCHEMA = "transcribe-audio.biometric-reference-tombstone.v1"
APPROVAL_SCHEMA = "transcribe-audio.biometric-reference-approval.v1"
DRY_RUN_SCHEMA = "transcribe-audio.biometric-reference-dry-run.v1"
RECEIPT_SCHEMA = "transcribe-audio.biometric-reference-receipt.v1"
DESCENDANT_SCHEMA = "transcribe-audio.biometric-descendant-registration.v1"
MATERIALIZATION_SCHEMA = "transcribe-audio.biometric-materialization-staging.v1"
PROMOTION_SCHEMA = "transcribe-audio.biometric-materialization-promotion.v1"
INVALIDATION_SCHEMA = "transcribe-audio.biometric-descendant-invalidation.v1"
RECEIPT_STAGE_SCHEMA = "transcribe-audio.biometric-receipt-stage.v1"
SOURCE_MANIFEST_SCHEMA = "transcribe-audio.biometric-source-manifest.v1"
SYNTHETIC_FIXTURE_SCHEMA = "transcribe-audio.synthetic-reference-fixture.v1"
DEFAULT_RUNTIME_ROOT = Path(
    "~/.local/state/transcribe-audio/plan-0037/biometric-references"
)

ACTION_TOKEN_PREFIXES = {
    "create": "CREATE_BIOMETRIC_REFERENCE",
    "supersede": "SUPERSEDE_BIOMETRIC_REFERENCE",
    "withdraw": "WITHDRAW_BIOMETRIC_REFERENCE",
    "delete": "DELETE_BIOMETRIC_REFERENCE",
}
ACTIVE = "active"
INACTIVE_STATUSES = {"superseded", "withdrawn", "deleted"}
ALL_STATUSES = {ACTIVE, *INACTIVE_STATUSES}
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_FORBIDDEN_KEYS = (
    "audio_bytes",
    "contact",
    "email",
    "embedding",
    "model_output",
    "name",
    "raw_audio",
    "transcript",
    "vector",
    "waveform",
)
_SOURCE_KEYS = {
    "reference_id",
    "source_blob_id",
    "source_sha256",
    "recording_id",
    "conversation_id",
    "speaker_label_id",
    "session_id",
    "start_seconds",
    "end_seconds",
    "source_duration_seconds",
    "quality_evidence",
    "lineage",
    "device_class",
    "acoustic_conditions",
    "source_key",
    "fixture_authority",
}
_P1_LINEAGE_KEYS = {
    "schema_version",
    "authority",
    "run_id",
    "runtime_root",
    "replay_receipt_path",
    "replay_receipt_sha256",
    "manifest_path",
    "manifest_sha256",
    "source_blob_id",
    "source_sha256",
    "artifact_sha256",
    "source_duration_seconds",
    "audio_quality_sha256",
    "timestamp_map_sha256",
    "validation_status",
    "will_read_audio",
}
_P2_LINEAGE_KEYS = {
    "schema_version",
    "authority",
    "run_id",
    "runtime_root",
    "method_id",
    "replay_receipt_path",
    "replay_receipt_sha256",
    "comparison_path",
    "comparison_sha256",
    "method_result_sha256",
    "source_blob_id",
    "source_sha256",
    "source_duration_seconds",
    "audio_quality_sha256",
    "validation_status",
    "will_read_audio",
}
_REFERENCE_MANIFEST_KEYS = {
    "schema_version", "profile_id", "person_ref_id", "generation_id",
    "generation_sequence", "predecessor_generation_id", "status",
    "eligible_for_materialization", "sources", "source_set_sha256", "approval",
    "synthetic_test_only", "descendant_policy", "created_at",
}
_TOMBSTONE_MANIFEST_KEYS = {
    "schema_version", "profile_id", "person_ref_id", "generation_id",
    "generation_sequence", "predecessor_generation_id", "status",
    "eligible_for_materialization", "prior_manifest_sha256", "deleted_at",
}


class BiometricReferenceError(ValueError):
    """Raised when P3 reference authority cannot prove a safe transition."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _hash(value: Any) -> str:
    return canonical_artifact_hash(value)


def _require_opaque_id(value: Any, field: str) -> str:
    text = str(value or "")
    if not _OPAQUE_ID.fullmatch(text) or "@" in text:
        raise BiometricReferenceError(f"{field} must be an opaque identifier.")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise BiometricReferenceError(f"{field} must be a lowercase SHA-256.")
    return text


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if any(family in normalized for family in _FORBIDDEN_KEYS):
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _require_private_regular(path: Path, root: Path) -> None:
    try:
        require_private_file(path, root)
    except ValueError as exc:
        raise BiometricReferenceError(str(exc)) from exc
    if path.stat().st_nlink != 1:
        raise BiometricReferenceError("Private authority files must not be hard-linked.")


@contextmanager
def _private_umask() -> Iterable[None]:
    prior = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(prior)


def _paths(root: Path, run_id: Optional[str] = None) -> dict[str, Path]:
    selected = root.expanduser().absolute()
    result = {
        "root": selected,
        "database": selected / "references.sqlite3",
        "runs": selected / "runs",
        "receipts": selected / "receipts",
        "attempts": selected / "attempts",
        "staged_receipts": selected / "staged-receipts",
    }
    if run_id is not None:
        _require_opaque_id(run_id, "run_id")
        run_dir = selected / "runs" / run_id
        result.update(
            {
                "run_dir": run_dir,
                "dry_run": run_dir / "dry-run.json",
                "source_manifest": run_dir / "source-manifest.json",
            }
        )
    return result


def _write_content_addressed_receipt(
    root: Path, receipt: dict[str, Any], *, attempt: bool = False
) -> tuple[Path, str]:
    receipt_sha = _hash(receipt)
    paths = _paths(root)
    directory = paths["attempts"] if attempt else paths["receipts"]
    ensure_private_tree(paths["root"], directory)
    path = directory / f"{receipt_sha}.json"
    try:
        write_immutable_private_json(path, receipt)
    except ValueError as exc:
        raise BiometricReferenceError(str(exc)) from exc
    return path, receipt_sha


def _require_content_addressed_receipt(
    root: Path, receipt: Mapping[str, Any], expected_sha256: str
) -> Path:
    if _hash(receipt) != expected_sha256:
        raise BiometricReferenceError("Content-addressed receipt hash mismatch.")
    path = _paths(root)["receipts"] / f"{expected_sha256}.json"
    _require_private_regular(path, root)
    if read_private_object(path) != dict(receipt):
        raise BiometricReferenceError("Content-addressed receipt anchor mismatch.")
    return path


def _stage_content_addressed_receipt(
    root: Path, receipt: Mapping[str, Any]
) -> tuple[Path, str]:
    """Persist explicitly non-authoritative prepared evidence before DB commit."""
    receipt_sha = _hash(receipt)
    stage = {
        "schema_version": RECEIPT_STAGE_SCHEMA,
        "state": "prepared_not_committed",
        "receipt_sha256": receipt_sha,
        "receipt": dict(receipt),
    }
    paths = _paths(root)
    ensure_private_tree(paths["root"], paths["staged_receipts"])
    path = paths["staged_receipts"] / f"{receipt_sha}.json"
    try:
        write_immutable_private_json(path, stage)
    except ValueError as exc:
        raise BiometricReferenceError(str(exc)) from exc
    return path, receipt_sha


def _promote_staged_receipt(root: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    """Publish an authoritative anchor only after its database commit succeeds."""
    receipt_sha = _hash(receipt)
    stage_path = _paths(root)["staged_receipts"] / f"{receipt_sha}.json"
    _require_private_regular(stage_path, root)
    expected_stage = {
        "schema_version": RECEIPT_STAGE_SCHEMA,
        "state": "prepared_not_committed",
        "receipt_sha256": receipt_sha,
        "receipt": dict(receipt),
    }
    if read_private_object(stage_path) != expected_stage:
        raise BiometricReferenceError("Prepared receipt evidence is invalid.")
    return _write_content_addressed_receipt(root, dict(receipt))


def _recover_or_require_receipt(
    root: Path, receipt: Mapping[str, Any], expected_sha256: str
) -> Path:
    """Recover post-commit publication after an interrupted promotion."""
    try:
        return _require_content_addressed_receipt(root, receipt, expected_sha256)
    except (BiometricReferenceError, FileNotFoundError):
        path, promoted_sha = _promote_staged_receipt(root, receipt)
        if promoted_sha != expected_sha256:
            raise BiometricReferenceError("Recovered receipt hash mismatch.")
        return path


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY,
            person_ref_id TEXT NOT NULL,
            status TEXT NOT NULL,
            head_generation_id TEXT NOT NULL,
            head_version INTEGER NOT NULL,
            event_sequence INTEGER NOT NULL,
            descendant_count INTEGER NOT NULL,
            last_event_sha256 TEXT,
            created_at TEXT NOT NULL,
            deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS generations (
            generation_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            predecessor_generation_id TEXT,
            status TEXT NOT NULL,
            eligible_for_materialization INTEGER NOT NULL,
            manifest_json TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(profile_id, sequence),
            FOREIGN KEY(profile_id) REFERENCES profiles(profile_id)
        );
        CREATE TABLE IF NOT EXISTS person_heads (
            person_ref_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            version INTEGER NOT NULL,
            FOREIGN KEY(profile_id) REFERENCES profiles(profile_id)
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            action TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            previous_event_sha256 TEXT,
            payload_json TEXT NOT NULL,
            event_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(profile_id, sequence),
            FOREIGN KEY(profile_id) REFERENCES profiles(profile_id)
        );
        CREATE TABLE IF NOT EXISTS source_claims (
            source_key TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            person_ref_id TEXT NOT NULL,
            first_profile_id TEXT NOT NULL,
            first_generation_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS descendants (
            descendant_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            generation_sha256 TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            state TEXT NOT NULL,
            materialization_receipt_json TEXT NOT NULL,
            materialization_receipt_sha256 TEXT NOT NULL,
            materialization_authority_path TEXT NOT NULL,
            materialization_authority_sha256 TEXT NOT NULL,
            promotion_receipt_json TEXT,
            promotion_receipt_sha256 TEXT,
            promotion_authority_path TEXT,
            promotion_authority_sha256 TEXT,
            invalidated_at TEXT,
            invalidation_reason TEXT,
            invalidation_receipt_json TEXT,
            invalidation_receipt_sha256 TEXT,
            invalidation_authority_path TEXT,
            invalidation_authority_sha256 TEXT,
            registration_sha256 TEXT NOT NULL,
            FOREIGN KEY(profile_id) REFERENCES profiles(profile_id),
            FOREIGN KEY(generation_id) REFERENCES generations(generation_id)
        );
        CREATE TABLE IF NOT EXISTS idempotency (
            token_sha256 TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY,
            approval_sha256 TEXT NOT NULL,
            run_id TEXT NOT NULL
        );
        """
    )


@contextmanager
def _connection(root: Path, *, create: bool) -> Iterable[sqlite3.Connection]:
    paths = _paths(root)
    database = paths["database"]
    if create:
        ensure_private_tree(paths["root"], paths["root"])
        if database.is_symlink():
            raise BiometricReferenceError("Private authority database must not be a symlink.")
        if not database.exists():
            try:
                descriptor = os.open(
                    database,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        if database.is_symlink() or not database.is_file():
            raise BiometricReferenceError("Private authority database must be regular.")
        if database.exists():
            _require_private_regular(database, paths["root"])
        before_open = database.stat()
        with _private_umask():
            connection = sqlite3.connect(database, timeout=30, isolation_level=None)
        after_open = database.stat()
        if (before_open.st_dev, before_open.st_ino) != (
            after_open.st_dev,
            after_open.st_ino,
        ):
            connection.close()
            raise BiometricReferenceError("Private authority database changed during open.")
        os.chmod(database, 0o600)
    else:
        if not database.is_file():
            raise BiometricReferenceError("Biometric reference store does not exist.")
        _require_private_regular(database, paths["root"])
        before_open = database.stat()
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=30, isolation_level=None
        )
        after_open = database.stat()
        if (before_open.st_dev, before_open.st_ino) != (
            after_open.st_dev,
            after_open.st_ino,
        ):
            connection.close()
            raise BiometricReferenceError("Private authority database changed during open.")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    if create:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA secure_delete = ON")
        _initialize_schema(connection)
        os.chmod(database, 0o600)
    try:
        yield connection
    finally:
        connection.close()
        if create and database.exists():
            os.chmod(database, 0o600)


def _row_object(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def _rollback_if_active(connection: sqlite3.Connection) -> None:
    if connection.in_transaction:
        connection.execute("ROLLBACK")


def _require_external_authority_receipt(
    receipt: Mapping[str, Any], *, path: Path, authority_root: Path, p3_root: Path
) -> tuple[str, str]:
    selected_root = authority_root.expanduser().absolute()
    selected_path = path.expanduser().absolute()
    if (
        selected_root == p3_root
        or selected_root in p3_root.parents
        or p3_root in selected_root.parents
    ):
        raise BiometricReferenceError("P4 authority must be independent from P3 storage.")
    _require_private_regular(selected_path, selected_root)
    if read_private_object(selected_path) != dict(receipt):
        raise BiometricReferenceError("Independent P4 authority receipt mismatch.")
    receipt_sha = _hash(receipt)
    return str(selected_path), receipt_sha


def _require_stored_external_authority(
    receipt: Mapping[str, Any], *, stored_path: Any, stored_sha256: Any, p3_root: Path
) -> None:
    selected = Path(str(stored_path or "")).expanduser().absolute()
    if selected == p3_root or p3_root in selected.parents:
        raise BiometricReferenceError("Stored P4 authority is not independent.")
    _require_private_regular(selected, selected.parent)
    if (
        _require_sha256(stored_sha256, "P4 authority sha256") != _hash(receipt)
        or read_private_object(selected) != dict(receipt)
    ):
        raise BiometricReferenceError("Stored P4 authority receipt mismatch.")


def _state(
    connection: sqlite3.Connection, *, profile_id: str, person_ref_id: str
) -> dict[str, Any]:
    profile = connection.execute(
        "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    person_head = connection.execute(
        "SELECT * FROM person_heads WHERE person_ref_id = ?", (person_ref_id,)
    ).fetchone()
    return {"profile": _row_object(profile), "person_head": _row_object(person_head)}


def _state_without_store(profile_id: str, person_ref_id: str) -> dict[str, Any]:
    return {"profile": None, "person_head": None}


def _validated_sources(value: Any, *, test_mode: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise BiometricReferenceError("Reference generation requires source segments.")
    forbidden = sorted(_forbidden_keys(value))
    if forbidden:
        raise BiometricReferenceError(
            "Reference metadata contains forbidden fields: " + ", ".join(forbidden)
        )
    result: list[dict[str, Any]] = []
    reference_ids: set[str] = set()
    source_keys: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise BiometricReferenceError("Reference source must be an object.")
        source = dict(item)
        if not set(source).issubset(_SOURCE_KEYS):
            raise BiometricReferenceError("Reference source contains unknown fields.")
        reference_id = _require_opaque_id(source.get("reference_id"), "reference_id")
        if reference_id in reference_ids:
            raise BiometricReferenceError("Reference IDs must be unique.")
        reference_ids.add(reference_id)
        for field in (
            "source_blob_id",
            "recording_id",
            "conversation_id",
            "speaker_label_id",
            "session_id",
        ):
            _require_opaque_id(source.get(field), field)
        _require_sha256(source.get("source_sha256"), "source_sha256")
        try:
            if any(
                isinstance(source.get(field), bool)
                for field in (
                    "start_seconds",
                    "end_seconds",
                    "source_duration_seconds",
                )
            ):
                raise TypeError
            start = float(source["start_seconds"])
            end = float(source["end_seconds"])
            duration = float(source["source_duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BiometricReferenceError("Reference segment bounds are incomplete.") from exc
        if (
            not all(math.isfinite(number) for number in (start, end, duration))
            or start < 0
            or end <= start
            or duration <= 0
            or end > duration
        ):
            raise BiometricReferenceError("Reference segment bounds are invalid.")
        source["start_seconds"] = start
        source["end_seconds"] = end
        source["source_duration_seconds"] = duration
        quality = source.get("quality_evidence")
        if not isinstance(quality, Mapping) or set(quality) != {
            "evidence_id",
            "sha256",
        }:
            raise BiometricReferenceError("Reference source requires quality evidence.")
        _require_opaque_id(quality.get("evidence_id"), "quality evidence_id")
        _require_sha256(quality.get("sha256"), "quality evidence sha256")
        source["quality_evidence"] = dict(quality)
        device_class = str(source.get("device_class") or "")
        if not device_class or len(device_class) > 128 or "@" in device_class:
            raise BiometricReferenceError("Reference device class is invalid.")
        conditions = source.get("acoustic_conditions")
        if not isinstance(conditions, list) or any(
            not isinstance(condition, str)
            or not condition
            or len(condition) > 128
            or "@" in condition
            for condition in conditions
        ):
            raise BiometricReferenceError("Reference acoustic conditions are invalid.")
        lineage = source.get("lineage")
        fixture_authority = source.get("fixture_authority")
        if test_mode:
            expected_fixture = {
                "schema_version": SYNTHETIC_FIXTURE_SCHEMA,
                "fixture_id": reference_id,
                "source_sha256": source["source_sha256"],
                "source_duration_seconds": duration,
                "quality_evidence_sha256": source["quality_evidence"]["sha256"],
            }
            if lineage is not None or fixture_authority != expected_fixture:
                raise BiometricReferenceError(
                    "Synthetic sources require exact test-only fixture authority."
                )
            source["fixture_authority"] = expected_fixture
        elif fixture_authority is not None or lineage is None:
            raise BiometricReferenceError(
                "Production reference sources require replay-validated lineage."
            )
        if lineage is not None:
            if not isinstance(lineage, Mapping):
                raise BiometricReferenceError("Reference lineage must be an object.")
            authority = lineage.get("authority")
            try:
                if authority == "p1_audio_derivative_replay":
                    if "schema_version" in lineage:
                        if set(lineage) != _P1_LINEAGE_KEYS:
                            raise BiometricReferenceError(
                                "Reference P1 lineage shape is invalid."
                            )
                    elif set(lineage) != {
                        "authority",
                        "run_id",
                        "runtime_root",
                        "replay_receipt_sha256",
                    }:
                        raise BiometricReferenceError(
                            "Reference P1 lineage request is invalid."
                        )
                    resolved_lineage = resolve_derivative_lineage_receipt(
                        _require_opaque_id(lineage.get("run_id"), "lineage run_id"),
                        replay_receipt_sha256=_require_sha256(
                            lineage.get("replay_receipt_sha256"),
                            "lineage replay_receipt_sha256",
                        ),
                        runtime_root=Path(str(lineage.get("runtime_root") or "")),
                    )
                elif authority == "p2_speech_preparation_replay":
                    if "schema_version" in lineage:
                        if set(lineage) != _P2_LINEAGE_KEYS:
                            raise BiometricReferenceError(
                                "Reference P2 lineage shape is invalid."
                            )
                    elif set(lineage) != {
                        "authority",
                        "run_id",
                        "runtime_root",
                        "method_id",
                        "replay_receipt_sha256",
                    }:
                        raise BiometricReferenceError(
                            "Reference P2 lineage request is invalid."
                        )
                    resolved_lineage = resolve_comparison_lineage_receipt(
                        _require_opaque_id(lineage.get("run_id"), "lineage run_id"),
                        method_id=str(lineage.get("method_id") or ""),
                        replay_receipt_sha256=_require_sha256(
                            lineage.get("replay_receipt_sha256"),
                            "lineage replay_receipt_sha256",
                        ),
                        runtime_root=Path(str(lineage.get("runtime_root") or "")),
                    )
                else:
                    raise BiometricReferenceError(
                        "Reference lineage authority is unsupported."
                    )
            except (AudioDerivativeError, SpeechPreparationError) as exc:
                raise BiometricReferenceError(
                    "Reference lineage is not replay-validated."
                ) from exc
            if "schema_version" in lineage and dict(lineage) != resolved_lineage:
                raise BiometricReferenceError("Stored reference lineage drifted.")
            if (
                resolved_lineage["source_blob_id"] != source["source_blob_id"]
                or resolved_lineage["source_sha256"] != source["source_sha256"]
                or float(resolved_lineage["source_duration_seconds"]) != duration
                or resolved_lineage["audio_quality_sha256"]
                != source["quality_evidence"]["sha256"]
            ):
                raise BiometricReferenceError("Reference lineage source binding mismatch.")
            source["lineage"] = resolved_lineage
        source_key = _hash(
            {
                "source_sha256": source["source_sha256"],
                "start_seconds": start,
                "end_seconds": end,
            }
        )
        if source_key in source_keys:
            raise BiometricReferenceError("Reference segments must not be duplicated.")
        source_keys.add(source_key)
        source["source_key"] = source_key
        result.append(source)
    return sorted(result, key=lambda item: item["reference_id"])


def source_set_sha256(
    sources: list[dict[str, Any]], *, test_mode: bool = False
) -> str:
    """Return the canonical approval binding for validated reference sources."""
    return _hash(_validated_sources(sources, test_mode=test_mode))


def _validated_approval(
    value: Any,
    *,
    action: str,
    profile_id: str,
    person_ref_id: str,
    source_hash: Optional[str],
    expected_generation_id: Optional[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BiometricReferenceError("Biometric-purpose approval is required.")
    approval = dict(value)
    if set(approval) != {
        "schema_version",
        "approval_id",
        "reviewer_ref_id",
        "reviewed_at",
        "purpose",
        "scope",
    }:
        raise BiometricReferenceError("Biometric approval shape is invalid.")
    forbidden = sorted(_forbidden_keys(approval))
    if forbidden:
        raise BiometricReferenceError(
            "Approval contains forbidden fields: " + ", ".join(forbidden)
        )
    if approval.get("schema_version") != APPROVAL_SCHEMA:
        raise BiometricReferenceError("Biometric approval schema is invalid.")
    _require_opaque_id(approval.get("approval_id"), "approval_id")
    _require_opaque_id(approval.get("reviewer_ref_id"), "reviewer_ref_id")
    reviewed_at = str(approval.get("reviewed_at") or "")
    try:
        parsed_reviewed_at = datetime.fromisoformat(
            reviewed_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise BiometricReferenceError(
            "Biometric approval requires a canonical UTC review time."
        ) from exc
    canonical_reviewed_at = (
        parsed_reviewed_at.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    if parsed_reviewed_at.utcoffset() != timezone.utc.utcoffset(None) or (
        reviewed_at != canonical_reviewed_at
    ):
        raise BiometricReferenceError("Biometric approval requires a UTC review time.")
    if approval.get("purpose") != f"biometric_reference_{action}":
        raise BiometricReferenceError("Approval purpose does not authorize this action.")
    scope = approval.get("scope")
    if not isinstance(scope, Mapping) or set(scope) != {
        "profile_id",
        "person_ref_id",
        "source_set_sha256",
        "expected_generation_id",
    }:
        raise BiometricReferenceError("Biometric approval scope is missing.")
    expected_scope = {
        "profile_id": profile_id,
        "person_ref_id": person_ref_id,
        "source_set_sha256": source_hash,
        "expected_generation_id": expected_generation_id,
    }
    if dict(scope) != expected_scope:
        raise BiometricReferenceError("Biometric approval scope binding mismatch.")
    return approval


def _generation_id(
    *,
    profile_id: str,
    person_ref_id: str,
    source_hash: str,
    approval: Mapping[str, Any],
    predecessor: Optional[str],
) -> str:
    identity = {
        "profile_id": profile_id,
        "person_ref_id": person_ref_id,
        "source_set_sha256": source_hash,
        "approval": dict(approval),
        "predecessor_generation_id": predecessor,
    }
    return "refgen-" + _hash(identity)[:24]


def _read_plan(run_id: str, root: Path) -> tuple[dict[str, Any], Path, str]:
    paths = _paths(root, run_id)
    _require_private_regular(paths["dry_run"], paths["root"])
    try:
        plan = read_private_object(paths["dry_run"])
    except ValueError as exc:
        raise BiometricReferenceError(str(exc)) from exc
    plan_sha = sha256_file(paths["dry_run"])
    if plan.get("schema_version") != DRY_RUN_SCHEMA or plan.get("run_id") != run_id:
        raise BiometricReferenceError("Biometric reference dry run is invalid.")
    _validate_plan_contract(plan)
    expected_source_path = (
        str(paths["source_manifest"])
        if plan["action"] in {"create", "supersede"}
        else None
    )
    if plan.get("source_manifest_path") != expected_source_path:
        raise BiometricReferenceError("Biometric source manifest path binding is invalid.")
    return plan, paths["dry_run"], plan_sha


def required_approval_token(plan: Mapping[str, Any], dry_run_sha256: str) -> str:
    action = str(plan.get("action") or "")
    prefix = ACTION_TOKEN_PREFIXES.get(action)
    if prefix is None:
        raise BiometricReferenceError("Biometric reference action is invalid.")
    if action == "create":
        binding = str(plan["run_id"])
    elif action == "supersede":
        binding = f"{plan['expected_generation_id']}:{plan['run_id']}"
    elif action == "withdraw":
        binding = str(plan["expected_generation_id"])
    else:
        binding = str(plan["profile_id"])
    return f"{prefix}:{binding}:{dry_run_sha256}"


def _validate_plan_contract(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "action",
        "profile_id",
        "person_ref_id",
        "source_claims",
        "source_set_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "synthetic_test_only",
        "approval",
        "expected_state",
        "expected_generation_id",
        "target_generation_id",
        "will_read_audio",
        "will_run_model",
        "will_create_embedding",
        "will_perform_external_write",
        "run_id",
        "created_at",
    }
    if set(plan) != expected_keys:
        raise BiometricReferenceError("Biometric reference dry-run shape is invalid.")
    action = str(plan.get("action") or "")
    if action not in ACTION_TOKEN_PREFIXES:
        raise BiometricReferenceError("Biometric reference action is invalid.")
    profile_id = _require_opaque_id(plan.get("profile_id"), "profile_id")
    person_ref_id = _require_opaque_id(plan.get("person_ref_id"), "person_ref_id")
    expected_state = plan.get("expected_state")
    if not isinstance(expected_state, Mapping) or set(expected_state) != {
        "profile",
        "person_head",
    }:
        raise BiometricReferenceError("Biometric reference expected state is invalid.")
    expected_generation = plan.get("expected_generation_id")
    if expected_generation is not None:
        _require_opaque_id(expected_generation, "expected_generation_id")
    if action in {"create", "supersede"}:
        source_hash = _require_sha256(
            plan.get("source_set_sha256"), "source_set_sha256"
        )
        claims = plan.get("source_claims")
        if not isinstance(claims, list) or not claims:
            raise BiometricReferenceError("Biometric reference source claims are invalid.")
        for claim in claims:
            if not isinstance(claim, Mapping) or set(claim) != {
                "source_key",
                "source_sha256",
                "start_seconds",
                "end_seconds",
            }:
                raise BiometricReferenceError(
                    "Biometric reference source claim shape is invalid."
                )
            _require_sha256(claim["source_key"], "source_key")
            _require_sha256(claim["source_sha256"], "source_sha256")
            if any(
                isinstance(claim[field], bool)
                for field in ("start_seconds", "end_seconds")
            ) or not all(
                math.isfinite(float(claim[field]))
                for field in ("start_seconds", "end_seconds")
            ) or float(claim["end_seconds"]) <= float(claim["start_seconds"]):
                raise BiometricReferenceError(
                    "Biometric reference source claim bounds are invalid."
                )
        _require_sha256(plan.get("source_manifest_sha256"), "source_manifest_sha256")
        if not isinstance(plan.get("source_manifest_path"), str):
            raise BiometricReferenceError("Biometric source manifest path is invalid.")
        if not isinstance(plan.get("synthetic_test_only"), bool):
            raise BiometricReferenceError("Biometric source authority mode is invalid.")
    else:
        if (
            plan.get("source_claims") != []
            or plan.get("source_set_sha256") is not None
            or plan.get("source_manifest_path") is not None
            or plan.get("source_manifest_sha256") is not None
            or plan.get("synthetic_test_only") is not False
        ):
            raise BiometricReferenceError("Lifecycle-only dry run contains sources.")
        source_hash = None
    approval = _validated_approval(
        plan.get("approval"),
        action=action,
        profile_id=profile_id,
        person_ref_id=person_ref_id,
        source_hash=source_hash,
        expected_generation_id=expected_generation,
    )
    expected_target = (
        _generation_id(
            profile_id=profile_id,
            person_ref_id=person_ref_id,
            source_hash=str(source_hash),
            approval=approval,
            predecessor=(expected_generation if action == "supersede" else None),
        )
        if action in {"create", "supersede"}
        else expected_generation
    )
    if plan.get("target_generation_id") != expected_target:
        raise BiometricReferenceError("Biometric reference target binding is invalid.")
    if any(
        plan.get(flag) is not False
        for flag in (
            "will_read_audio",
            "will_run_model",
            "will_create_embedding",
            "will_perform_external_write",
        )
    ):
        raise BiometricReferenceError("Biometric reference dry run exceeds P3 scope.")
    identity = {
        key: value
        for key, value in plan.items()
        if key not in {"run_id", "created_at", "source_manifest_path"}
    }
    if plan.get("run_id") != "bio-ref-run-" + _hash(identity)[:24]:
        raise BiometricReferenceError("Biometric reference run identity is invalid.")


def dry_run(
    action: str,
    *,
    profile_id: str,
    approval: Mapping[str, Any],
    person_ref_id: Optional[str] = None,
    sources: Optional[list[dict[str, Any]]] = None,
    test_mode: bool = False,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist one immutable, no-audio P3 transition plan."""
    if action not in ACTION_TOKEN_PREFIXES:
        raise BiometricReferenceError("Biometric reference action is invalid.")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    selected_profile = _require_opaque_id(profile_id, "profile_id")
    database = _paths(root)["database"]
    descendant_states: list[str] = []
    if action == "create":
        selected_person = _require_opaque_id(person_ref_id, "person_ref_id")
        current = _state_without_store(selected_profile, selected_person)
        if database.exists():
            with _connection(root, create=False) as connection:
                current = _state(
                    connection,
                    profile_id=selected_profile,
                    person_ref_id=selected_person,
                )
    else:
        if not database.exists():
            raise BiometricReferenceError("Biometric reference store does not exist.")
        with _connection(root, create=False) as connection:
            profile = connection.execute(
                "SELECT * FROM profiles WHERE profile_id = ?", (selected_profile,)
            ).fetchone()
            if profile is None:
                raise BiometricReferenceError("Biometric reference profile does not exist.")
            selected_person = str(profile["person_ref_id"])
            current = _state(
                connection,
                profile_id=selected_profile,
                person_ref_id=selected_person,
            )
            descendant_states = [
                str(row["state"])
                for row in connection.execute(
                    "SELECT state FROM descendants WHERE profile_id = ?",
                    (selected_profile,),
                ).fetchall()
            ]
    profile_state = current["profile"]
    expected_generation = (
        str(profile_state["head_generation_id"]) if profile_state else None
    )
    if action == "create":
        if profile_state is not None:
            raise BiometricReferenceError("Biometric reference profile ID already exists.")
        validated_sources = _validated_sources(sources, test_mode=test_mode)
        source_hash = _hash(validated_sources)
        predecessor = None
    elif action == "supersede":
        if profile_state is None or profile_state["status"] != ACTIVE:
            raise BiometricReferenceError("Only an active reference can be superseded.")
        validated_sources = _validated_sources(sources, test_mode=test_mode)
        source_hash = _hash(validated_sources)
        predecessor = expected_generation
    else:
        if sources not in (None, []):
            raise BiometricReferenceError("Lifecycle-only actions do not accept sources.")
        if profile_state is None or profile_state["status"] not in (
            {ACTIVE} if action == "withdraw" else {ACTIVE, "withdrawn"}
        ):
            raise BiometricReferenceError(f"Reference cannot be {action}d from its state.")
        validated_sources = []
        source_hash = None
        predecessor = expected_generation
        test_mode = False
        if action == "delete" and profile_state["status"] == ACTIVE and any(
            state != "invalidated" for state in descendant_states
        ):
            raise BiometricReferenceError(
                "Reference with descendants must be withdrawn before deletion."
            )
        if action == "delete" and any(
            state == "invalidation_pending" for state in descendant_states
        ):
            raise BiometricReferenceError(
                "Descendant invalidation acknowledgments are required before deletion."
            )
    validated_approval = _validated_approval(
        approval,
        action=action,
        profile_id=selected_profile,
        person_ref_id=selected_person,
        source_hash=source_hash,
        expected_generation_id=expected_generation,
    )
    target_generation = (
        _generation_id(
            profile_id=selected_profile,
            person_ref_id=selected_person,
            source_hash=str(source_hash),
            approval=validated_approval,
            predecessor=predecessor,
        )
        if action in {"create", "supersede"}
        else expected_generation
    )
    source_manifest = (
        {
            "schema_version": SOURCE_MANIFEST_SCHEMA,
            "synthetic_test_only": test_mode,
            "sources": validated_sources,
        }
        if validated_sources
        else None
    )
    source_manifest_sha = _hash(source_manifest) if source_manifest else None
    identity = {
        "schema_version": DRY_RUN_SCHEMA,
        "action": action,
        "profile_id": selected_profile,
        "person_ref_id": selected_person,
        "source_claims": [
            {
                "source_key": source["source_key"],
                "source_sha256": source["source_sha256"],
                "start_seconds": source["start_seconds"],
                "end_seconds": source["end_seconds"],
            }
            for source in validated_sources
        ],
        "source_set_sha256": source_hash,
        "source_manifest_sha256": source_manifest_sha,
        "synthetic_test_only": test_mode,
        "approval": validated_approval,
        "expected_state": current,
        "expected_generation_id": expected_generation,
        "target_generation_id": target_generation,
        "will_read_audio": False,
        "will_run_model": False,
        "will_create_embedding": False,
        "will_perform_external_write": False,
    }
    run_id = "bio-ref-run-" + _hash(identity)[:24]
    paths = _paths(root, run_id)
    source_manifest_path = str(paths["source_manifest"]) if source_manifest else None
    plan = {
        **identity,
        "source_manifest_path": source_manifest_path,
        "run_id": run_id,
        "created_at": utc_now(),
    }
    ensure_private_tree(paths["root"], paths["run_dir"])
    try:
        if source_manifest is not None:
            write_immutable_private_json(paths["source_manifest"], source_manifest)
        stored = write_immutable_private_json(
            paths["dry_run"], plan, volatile_fields=("created_at",)
        )
    except ValueError as exc:
        raise BiometricReferenceError(str(exc)) from exc
    plan_sha = sha256_file(paths["dry_run"])
    return {
        **stored,
        "dry_run_path": str(paths["dry_run"]),
        "dry_run_sha256": plan_sha,
        "required_approval_token": required_approval_token(stored, plan_sha),
    }


def _assert_expected_state(
    connection: sqlite3.Connection, plan: Mapping[str, Any]
) -> None:
    current = _state(
        connection,
        profile_id=str(plan["profile_id"]),
        person_ref_id=str(plan["person_ref_id"]),
    )
    if current != plan.get("expected_state"):
        raise BiometricReferenceError("Biometric reference head changed after dry run.")


def _validated_apply_sources(
    plan: Mapping[str, Any], value: Optional[list[dict[str, Any]]], *, root: Path,
    test_mode: bool,
) -> list[dict[str, Any]]:
    if plan["action"] not in {"create", "supersede"}:
        if value not in (None, []):
            raise BiometricReferenceError("Lifecycle-only apply does not accept sources.")
        return []
    if test_mode is not plan["synthetic_test_only"]:
        raise BiometricReferenceError("Apply source authority mode differs from dry run.")
    manifest_path = Path(str(plan["source_manifest_path"]))
    _require_private_regular(manifest_path, root)
    stored_manifest = read_private_object(manifest_path)
    if (
        set(stored_manifest) != {"schema_version", "synthetic_test_only", "sources"}
        or stored_manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA
        or stored_manifest["synthetic_test_only"] is not test_mode
        or _hash(stored_manifest) != plan["source_manifest_sha256"]
    ):
        raise BiometricReferenceError("Immutable source manifest is invalid.")
    sources = _validated_sources(value, test_mode=test_mode)
    claims = [
        {
            "source_key": source["source_key"],
            "source_sha256": source["source_sha256"],
            "start_seconds": source["start_seconds"],
            "end_seconds": source["end_seconds"],
        }
        for source in sources
    ]
    if (
        stored_manifest["sources"] != sources
        or _hash(sources) != plan["source_set_sha256"]
        or claims != plan["source_claims"]
    ):
        raise BiometricReferenceError("Apply source set differs from the dry run.")
    return sources


def _check_source_claims(
    connection: sqlite3.Connection,
    *,
    sources: list[dict[str, Any]],
    person_ref_id: str,
) -> None:
    for source in sources:
        claim = connection.execute(
            """
            SELECT * FROM source_claims
            WHERE source_sha256 = ? AND start_seconds < ? AND end_seconds > ?
              AND person_ref_id != ?
            LIMIT 1
            """,
            (
                source["source_sha256"],
                source["end_seconds"],
                source["start_seconds"],
                person_ref_id,
            ),
        ).fetchone()
        if claim is not None:
            raise BiometricReferenceError(
                "A reference segment is already claimed by another person reference."
            )


def _record_source_claims(
    connection: sqlite3.Connection,
    *,
    sources: list[dict[str, Any]],
    person_ref_id: str,
    profile_id: str,
    generation_id: str,
) -> None:
    for source in sources:
        connection.execute(
            """
            INSERT OR IGNORE INTO source_claims
            (source_key, source_sha256, start_seconds, end_seconds,
             person_ref_id, first_profile_id, first_generation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source["source_key"],
                source["source_sha256"],
                source["start_seconds"],
                source["end_seconds"],
                person_ref_id,
                profile_id,
                generation_id,
            ),
        )


def _append_event(
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    action: str,
    generation_id: str,
    details: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    profile = connection.execute(
        "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    if profile is None:
        raise BiometricReferenceError("Biometric reference profile disappeared.")
    sequence = int(profile["event_sequence"]) + 1
    event = {
        "profile_id": profile_id,
        "sequence": sequence,
        "action": action,
        "generation_id": generation_id,
        "previous_event_sha256": profile["last_event_sha256"],
        "details": dict(details),
        "created_at": created_at,
    }
    event_sha = _hash(event)
    event_id = "refevent-" + event_sha[:24]
    connection.execute(
        """
        INSERT INTO events
        (event_id, profile_id, sequence, action, generation_id,
         previous_event_sha256, payload_json, event_sha256, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            profile_id,
            sequence,
            action,
            generation_id,
            profile["last_event_sha256"],
            _canonical_json(event),
            event_sha,
            created_at,
        ),
    )
    connection.execute(
        "UPDATE profiles SET event_sequence = ?, last_event_sha256 = ? WHERE profile_id = ?",
        (sequence, event_sha, profile_id),
    )
    return {"event_id": event_id, "event_sha256": event_sha, "sequence": sequence}


def _manifest(
    plan: Mapping[str, Any],
    sources: list[dict[str, Any]],
    *,
    sequence: int,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_SCHEMA,
        "profile_id": plan["profile_id"],
        "person_ref_id": plan["person_ref_id"],
        "generation_id": plan["target_generation_id"],
        "generation_sequence": sequence,
        "predecessor_generation_id": plan["expected_generation_id"],
        "status": ACTIVE,
        "eligible_for_materialization": True,
        "sources": sources,
        "source_set_sha256": plan["source_set_sha256"],
        "approval": plan["approval"],
        "synthetic_test_only": plan["synthetic_test_only"],
        "descendant_policy": "p4_registration_and_parent_lifecycle_required",
        "created_at": created_at,
    }


def _invalidate_descendants(
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    reason: str,
    created_at: str,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT descendant_id FROM descendants
        WHERE profile_id = ? AND state IN ('staged', 'eligible')
        """,
        (profile_id,),
    ).fetchall()
    identifiers = [str(row["descendant_id"]) for row in rows]
    connection.execute(
        """
        UPDATE descendants SET state = 'invalidation_pending',
        invalidated_at = ?, invalidation_reason = ?
        WHERE profile_id = ? AND state IN ('staged', 'eligible')
        """,
        (created_at, reason, profile_id),
    )
    return identifiers


def _apply_create_or_supersede(
    connection: sqlite3.Connection,
    plan: Mapping[str, Any],
    created_at: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    action = str(plan["action"])
    profile_id = str(plan["profile_id"])
    person_ref_id = str(plan["person_ref_id"])
    generation_id = str(plan["target_generation_id"])
    _check_source_claims(
        connection, sources=sources, person_ref_id=person_ref_id
    )
    invalidated: list[str] = []
    if action == "create":
        connection.execute(
            """
            INSERT INTO profiles
            (profile_id, person_ref_id, status, head_generation_id, head_version,
             event_sequence, descendant_count, last_event_sha256, created_at,
             deleted_at)
            VALUES (?, ?, ?, ?, 1, 0, 0, NULL, ?, NULL)
            """,
            (profile_id, person_ref_id, ACTIVE, generation_id, created_at),
        )
        prior_head = plan["expected_state"].get("person_head")
        if prior_head is None:
            connection.execute(
                "INSERT INTO person_heads VALUES (?, ?, ?, ?, 1)",
                (person_ref_id, profile_id, generation_id, ACTIVE),
            )
        else:
            updated = connection.execute(
                """
                UPDATE person_heads SET profile_id = ?, generation_id = ?,
                status = ?, version = version + 1
                WHERE person_ref_id = ? AND version = ? AND status != 'active'
                """,
                (
                    profile_id,
                    generation_id,
                    ACTIVE,
                    person_ref_id,
                    prior_head["version"],
                ),
            )
            if updated.rowcount != 1:
                raise BiometricReferenceError("Person reference head CAS failed.")
        sequence = 1
    else:
        profile = plan["expected_state"]["profile"]
        old_generation = str(profile["head_generation_id"])
        updated = connection.execute(
            """
            UPDATE profiles SET head_generation_id = ?, head_version = head_version + 1
            WHERE profile_id = ? AND head_generation_id = ? AND head_version = ?
              AND status = 'active'
            """,
            (generation_id, profile_id, old_generation, profile["head_version"]),
        )
        if updated.rowcount != 1:
            raise BiometricReferenceError("Biometric reference profile CAS failed.")
        connection.execute(
            """
            UPDATE generations SET status = 'superseded',
            eligible_for_materialization = 0
            WHERE generation_id = ? AND status = 'active'
            """,
            (old_generation,),
        )
        head = plan["expected_state"]["person_head"]
        updated = connection.execute(
            """
            UPDATE person_heads SET generation_id = ?, version = version + 1
            WHERE person_ref_id = ? AND profile_id = ? AND generation_id = ?
              AND version = ? AND status = 'active'
            """,
            (
                generation_id,
                person_ref_id,
                profile_id,
                old_generation,
                head["version"],
            ),
        )
        if updated.rowcount != 1:
            raise BiometricReferenceError("Person reference head CAS failed.")
        invalidated = _invalidate_descendants(
            connection,
            profile_id=profile_id,
            reason="reference_superseded",
            created_at=created_at,
        )
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS value FROM generations WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()["value"]
    manifest = _manifest(
        plan, sources, sequence=int(sequence), created_at=created_at
    )
    manifest_sha = _hash(manifest)
    connection.execute(
        """
        INSERT INTO generations
        (generation_id, profile_id, sequence, predecessor_generation_id,
         status, eligible_for_materialization, manifest_json,
         manifest_sha256, created_at)
        VALUES (?, ?, ?, ?, 'active', 1, ?, ?, ?)
        """,
        (
            generation_id,
            profile_id,
            sequence,
            plan["expected_generation_id"],
            _canonical_json(manifest),
            manifest_sha,
            created_at,
        ),
    )
    _record_source_claims(
        connection,
        sources=sources,
        person_ref_id=person_ref_id,
        profile_id=profile_id,
        generation_id=generation_id,
    )
    event = _append_event(
        connection,
        profile_id=profile_id,
        action=action,
        generation_id=generation_id,
        details={
            "manifest_sha256": manifest_sha,
            "approval_id": plan["approval"]["approval_id"],
            "source_set_sha256": plan["source_set_sha256"],
            "invalidated_descendant_ids": invalidated,
        },
        created_at=created_at,
    )
    return {
        "generation_id": generation_id,
        "generation_sha256": manifest_sha,
        "lifecycle_state": ACTIVE,
        "invalidated_descendant_ids": invalidated,
        **event,
    }


def _apply_withdraw(
    connection: sqlite3.Connection, plan: Mapping[str, Any], created_at: str
) -> dict[str, Any]:
    profile = plan["expected_state"]["profile"]
    head = plan["expected_state"]["person_head"]
    generation_id = str(profile["head_generation_id"])
    updated = connection.execute(
        """
        UPDATE profiles SET status = 'withdrawn', head_version = head_version + 1
        WHERE profile_id = ? AND status = 'active' AND head_generation_id = ?
          AND head_version = ?
        """,
        (plan["profile_id"], generation_id, profile["head_version"]),
    )
    if updated.rowcount != 1:
        raise BiometricReferenceError("Biometric reference withdrawal CAS failed.")
    connection.execute(
        """
        UPDATE generations SET status = 'withdrawn', eligible_for_materialization = 0
        WHERE generation_id = ? AND status = 'active'
        """,
        (generation_id,),
    )
    updated = connection.execute(
        """
        UPDATE person_heads SET status = 'withdrawn', version = version + 1
        WHERE person_ref_id = ? AND profile_id = ? AND generation_id = ?
          AND version = ? AND status = 'active'
        """,
        (
            plan["person_ref_id"],
            plan["profile_id"],
            generation_id,
            head["version"],
        ),
    )
    if updated.rowcount != 1:
        raise BiometricReferenceError("Person reference withdrawal CAS failed.")
    invalidated = _invalidate_descendants(
        connection,
        profile_id=str(plan["profile_id"]),
        reason="reference_withdrawn",
        created_at=created_at,
    )
    event = _append_event(
        connection,
        profile_id=str(plan["profile_id"]),
        action="withdraw",
        generation_id=generation_id,
        details={
            "approval_id": plan["approval"]["approval_id"],
            "invalidated_descendant_ids": invalidated,
        },
        created_at=created_at,
    )
    return {
        "generation_id": generation_id,
        "lifecycle_state": "withdrawn",
        "invalidated_descendant_ids": invalidated,
        **event,
    }


def _tombstone_manifest(
    row: sqlite3.Row, *, profile_id: str, person_ref_id: str, deleted_at: str
) -> dict[str, Any]:
    return {
        "schema_version": TOMBSTONE_SCHEMA,
        "profile_id": profile_id,
        "person_ref_id": person_ref_id,
        "generation_id": row["generation_id"],
        "generation_sequence": row["sequence"],
        "predecessor_generation_id": row["predecessor_generation_id"],
        "status": "deleted",
        "eligible_for_materialization": False,
        "prior_manifest_sha256": row["manifest_sha256"],
        "deleted_at": deleted_at,
    }


def _apply_delete(
    connection: sqlite3.Connection, plan: Mapping[str, Any], created_at: str
) -> dict[str, Any]:
    profile = plan["expected_state"]["profile"]
    head = plan["expected_state"]["person_head"]
    generation_id = str(profile["head_generation_id"])
    pending_descendants = connection.execute(
        """
        SELECT descendant_id FROM descendants
        WHERE profile_id = ? AND state != 'invalidated'
        """,
        (plan["profile_id"],),
    ).fetchall()
    if pending_descendants:
        raise BiometricReferenceError(
            "Deletion requires P4 descendant invalidation acknowledgments."
        )
    invalidated = _invalidate_descendants(
        connection,
        profile_id=str(plan["profile_id"]),
        reason="reference_deleted",
        created_at=created_at,
    )
    rows = connection.execute(
        "SELECT * FROM generations WHERE profile_id = ? ORDER BY sequence",
        (plan["profile_id"],),
    ).fetchall()
    tombstone_hashes: list[str] = []
    for row in rows:
        tombstone = _tombstone_manifest(
            row,
            profile_id=str(plan["profile_id"]),
            person_ref_id=str(plan["person_ref_id"]),
            deleted_at=created_at,
        )
        tombstone_sha = _hash(tombstone)
        tombstone_hashes.append(tombstone_sha)
        connection.execute(
            """
            UPDATE generations SET status = 'deleted', eligible_for_materialization = 0,
            manifest_json = ?, manifest_sha256 = ? WHERE generation_id = ?
            """,
            (_canonical_json(tombstone), tombstone_sha, row["generation_id"]),
        )
    updated = connection.execute(
        """
        UPDATE profiles SET status = 'deleted', head_version = head_version + 1,
        deleted_at = ? WHERE profile_id = ? AND head_generation_id = ?
          AND head_version = ? AND status IN ('active', 'withdrawn')
        """,
        (created_at, plan["profile_id"], generation_id, profile["head_version"]),
    )
    if updated.rowcount != 1:
        raise BiometricReferenceError("Biometric reference deletion CAS failed.")
    updated = connection.execute(
        """
        UPDATE person_heads SET status = 'deleted', version = version + 1
        WHERE person_ref_id = ? AND profile_id = ? AND generation_id = ?
          AND version = ? AND status IN ('active', 'withdrawn')
        """,
        (
            plan["person_ref_id"],
            plan["profile_id"],
            generation_id,
            head["version"],
        ),
    )
    if updated.rowcount != 1:
        raise BiometricReferenceError("Person reference deletion CAS failed.")
    event = _append_event(
        connection,
        profile_id=str(plan["profile_id"]),
        action="delete",
        generation_id=generation_id,
        details={
            "approval_id": plan["approval"]["approval_id"],
            "tombstone_sha256s": tombstone_hashes,
            "invalidated_descendant_ids": invalidated,
        },
        created_at=created_at,
    )
    return {
        "generation_id": generation_id,
        "lifecycle_state": "deleted",
        "tombstone_sha256s": tombstone_hashes,
        "invalidated_descendant_ids": invalidated,
        **event,
    }


def apply_change(
    run_id: str,
    *,
    approval_token: str,
    sources: Optional[list[dict[str, Any]]] = None,
    test_mode: bool = False,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Apply one persisted P3 plan under a SQLite transaction and exact CAS."""
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    plan, dry_path, dry_sha = _read_plan(run_id, root)
    required = required_approval_token(plan, dry_sha)
    if approval_token != required:
        raise BiometricReferenceError(f"Apply requires token {required}.")
    token_sha = _hash(approval_token)
    with _connection(root, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT * FROM idempotency WHERE token_sha256 = ?", (token_sha,)
            ).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id:
                    raise BiometricReferenceError("Approval token conflicts with another run.")
                if sources is not None:
                    _validated_apply_sources(
                        plan, sources, root=root, test_mode=test_mode
                    )
                receipt = json.loads(existing["receipt_json"])
                if _hash(receipt) != existing["receipt_sha256"]:
                    raise BiometricReferenceError("Idempotency receipt hash mismatch.")
                anchor = _recover_or_require_receipt(
                    root, receipt, str(existing["receipt_sha256"])
                )
                connection.execute("COMMIT")
                current = replay_reference(
                    str(receipt["profile_id"]), runtime_root=root
                )
                return {
                    **receipt,
                    "historical_lifecycle_state": receipt["lifecycle_state"],
                    "lifecycle_state": current["lifecycle_state"],
                    "current_status": current["status"],
                    "receipt_anchor_path": str(anchor),
                    "idempotent_replay": True,
                }
            _assert_expected_state(connection, plan)
            validated_sources = _validated_apply_sources(
                plan, sources, root=root, test_mode=test_mode
            )
            approval_sha = _hash(plan["approval"])
            consumed = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (plan["approval"]["approval_id"],),
            ).fetchone()
            if consumed is not None:
                if (
                    consumed["approval_sha256"] != approval_sha
                    or consumed["run_id"] != run_id
                ):
                    raise BiometricReferenceError(
                        "Biometric approval was already consumed by another scope."
                    )
            else:
                connection.execute(
                    "INSERT INTO approvals VALUES (?, ?, ?)",
                    (plan["approval"]["approval_id"], approval_sha, run_id),
                )
            created_at = utc_now()
            if plan["action"] in {"create", "supersede"}:
                result = _apply_create_or_supersede(
                    connection, plan, created_at, validated_sources
                )
            elif plan["action"] == "withdraw":
                result = _apply_withdraw(connection, plan, created_at)
            elif plan["action"] == "delete":
                result = _apply_delete(connection, plan, created_at)
            else:
                raise BiometricReferenceError("Biometric reference action is invalid.")
            receipt = {
                "schema_version": RECEIPT_SCHEMA,
                "run_id": run_id,
                "action": plan["action"],
                "status": "success",
                "reason_code": None,
                "profile_id": plan["profile_id"],
                "person_ref_id": plan["person_ref_id"],
                "dry_run_path": str(dry_path),
                "dry_run_sha256": dry_sha,
                "will_read_audio": False,
                "will_run_model": False,
                "will_create_embedding": False,
                "will_perform_external_write": False,
                "applied_at": created_at,
                **result,
            }
            receipt_sha = _hash(receipt)
            _, staged_sha = _stage_content_addressed_receipt(root, receipt)
            if staged_sha != receipt_sha:
                raise BiometricReferenceError("Immutable receipt anchor hash mismatch.")
            connection.execute(
                "INSERT INTO idempotency VALUES (?, ?, ?, ?)",
                (token_sha, run_id, _canonical_json(receipt), receipt_sha),
            )
            connection.execute("COMMIT")
            anchor, anchor_sha = _promote_staged_receipt(root, receipt)
            if anchor_sha != receipt_sha:
                raise BiometricReferenceError("Immutable receipt anchor hash mismatch.")
        except Exception as exc:
            _rollback_if_active(connection)
            attempt = {
                "schema_version": RECEIPT_SCHEMA,
                "run_id": run_id,
                "action": plan["action"],
                "status": "failure",
                "reason_code": exc.__class__.__name__,
                "profile_id": plan["profile_id"],
                "dry_run_sha256": dry_sha,
                "will_read_audio": False,
                "will_run_model": False,
                "will_create_embedding": False,
                "will_perform_external_write": False,
                "attempted_at": utc_now(),
            }
            _write_content_addressed_receipt(root, attempt, attempt=True)
            raise
    return {
        **receipt,
        "receipt_anchor_path": str(anchor),
        "idempotent_replay": False,
    }


def _validated_event_chain(
    connection: sqlite3.Connection, profile: sqlite3.Row
) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM events WHERE profile_id = ? ORDER BY sequence",
        (profile["profile_id"],),
    ).fetchall()
    prior: Optional[str] = None
    events: list[dict[str, Any]] = []
    for expected_sequence, row in enumerate(rows, start=1):
        payload = json.loads(row["payload_json"])
        if (
            row["sequence"] != expected_sequence
            or payload.get("sequence") != expected_sequence
            or payload.get("profile_id") != profile["profile_id"]
            or payload.get("action") != row["action"]
            or payload.get("generation_id") != row["generation_id"]
            or row["action"] not in ACTION_TOKEN_PREFIXES
            or not isinstance(payload.get("details"), Mapping)
            or payload.get("created_at") != row["created_at"]
            or payload.get("previous_event_sha256") != prior
            or row["previous_event_sha256"] != prior
            or _hash(payload) != row["event_sha256"]
        ):
            raise BiometricReferenceError("Biometric reference event chain is invalid.")
        prior = str(row["event_sha256"])
        events.append({**payload, "event_sha256": row["event_sha256"]})
    if (
        len(events) != profile["event_sequence"]
        or prior != profile["last_event_sha256"]
    ):
        raise BiometricReferenceError("Biometric reference event head is invalid.")
    return events


def _assert_lifecycle_rows(
    profile: sqlite3.Row,
    events: list[dict[str, Any]],
    generation_rows: list[sqlite3.Row],
) -> None:
    if not events or events[0]["action"] != "create":
        raise BiometricReferenceError("Biometric reference lifecycle must begin with create.")
    generation_ids = [str(row["generation_id"]) for row in generation_rows]
    expected: dict[str, str] = {}
    active_generation: Optional[str] = None
    profile_status = ACTIVE
    for index, event in enumerate(events):
        action = str(event["action"])
        generation_id = str(event["generation_id"])
        if action == "create":
            if index != 0 or generation_id not in generation_ids:
                raise BiometricReferenceError("Biometric reference create event is invalid.")
            active_generation = generation_id
            expected[generation_id] = ACTIVE
        elif action == "supersede":
            if active_generation is None or generation_id not in generation_ids:
                raise BiometricReferenceError("Biometric supersession event is invalid.")
            expected[active_generation] = "superseded"
            active_generation = generation_id
            expected[generation_id] = ACTIVE
        elif action == "withdraw":
            if generation_id != active_generation or profile_status != ACTIVE:
                raise BiometricReferenceError("Biometric withdrawal event is invalid.")
            expected[generation_id] = "withdrawn"
            profile_status = "withdrawn"
        elif action == "delete":
            if generation_id != active_generation or profile_status not in {
                ACTIVE,
                "withdrawn",
            }:
                raise BiometricReferenceError("Biometric deletion event is invalid.")
            expected = {identifier: "deleted" for identifier in generation_ids}
            profile_status = "deleted"
        else:
            raise BiometricReferenceError("Biometric lifecycle action is invalid.")
    if set(expected) != set(generation_ids) or profile["status"] != profile_status:
        raise BiometricReferenceError("Biometric reference lifecycle state is invalid.")
    for row in generation_rows:
        expected_status = expected[str(row["generation_id"])]
        if row["status"] != expected_status or bool(
            row["eligible_for_materialization"]
        ) != (expected_status == ACTIVE):
            raise BiometricReferenceError("Biometric generation lifecycle state is invalid.")
    if active_generation != profile["head_generation_id"]:
        raise BiometricReferenceError("Biometric reference lifecycle head is invalid.")


def _validated_generations(
    connection: sqlite3.Connection,
    profile: sqlite3.Row,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM generations WHERE profile_id = ? ORDER BY sequence",
        (profile["profile_id"],),
    ).fetchall()
    manifests: list[dict[str, Any]] = []
    predecessor: Optional[str] = None
    delete_events = [event for event in events if event["action"] == "delete"]
    if profile["status"] == "deleted" and len(delete_events) != 1:
        raise BiometricReferenceError("Biometric deletion evidence is invalid.")
    for expected_sequence, row in enumerate(rows, start=1):
        manifest = json.loads(row["manifest_json"])
        creation_events = [
            event
            for event in events
            if event["action"] in {"create", "supersede"}
            and event["generation_id"] == row["generation_id"]
        ]
        if len(creation_events) != 1:
            raise BiometricReferenceError(
                "Biometric generation creation evidence is invalid."
            )
        creation_event = creation_events[0]
        creation_sha = creation_event["details"].get("manifest_sha256")
        if (
            row["sequence"] != expected_sequence
            or manifest.get("generation_sequence") != expected_sequence
            or manifest.get("predecessor_generation_id") != predecessor
            or _hash(manifest) != row["manifest_sha256"]
            or manifest.get("profile_id") != profile["profile_id"]
            or manifest.get("person_ref_id") != profile["person_ref_id"]
            or manifest.get("generation_id") != row["generation_id"]
        ):
            raise BiometricReferenceError("Biometric reference generation chain is invalid.")
        if profile["status"] == "deleted":
            if (
                set(manifest) != _TOMBSTONE_MANIFEST_KEYS
                or
                manifest.get("schema_version") != TOMBSTONE_SCHEMA
                or manifest.get("status") != "deleted"
                or manifest.get("eligible_for_materialization") is not False
                or "sources" in manifest
                or "approval" in manifest
                or manifest.get("prior_manifest_sha256") != creation_sha
                or manifest.get("deleted_at") != profile["deleted_at"]
                or manifest.get("deleted_at") != delete_events[0]["created_at"]
            ):
                raise BiometricReferenceError("Deleted reference tombstone is invalid.")
        else:
            if (
                set(manifest) != _REFERENCE_MANIFEST_KEYS
                or manifest.get("schema_version") != REFERENCE_SCHEMA
                or not isinstance(manifest.get("synthetic_test_only"), bool)
            ):
                raise BiometricReferenceError("Biometric reference schema is invalid.")
            sources = _validated_sources(
                manifest.get("sources"),
                test_mode=manifest["synthetic_test_only"],
            )
            if _hash(sources) != manifest.get("source_set_sha256"):
                raise BiometricReferenceError("Reference source hash is invalid.")
            approval = _validated_approval(
                manifest.get("approval"),
                action=str(creation_event["action"]),
                profile_id=str(profile["profile_id"]),
                person_ref_id=str(profile["person_ref_id"]),
                source_hash=str(manifest["source_set_sha256"]),
                expected_generation_id=predecessor,
            )
            if (
                manifest.get("status") != ACTIVE
                or manifest.get("eligible_for_materialization") is not True
                or manifest.get("descendant_policy")
                != "p4_registration_and_parent_lifecycle_required"
                or manifest.get("created_at") != row["created_at"]
                or creation_sha != row["manifest_sha256"]
                or creation_event["details"].get("source_set_sha256")
                != manifest["source_set_sha256"]
                or creation_event["details"].get("approval_id")
                != approval["approval_id"]
                or _generation_id(
                    profile_id=str(profile["profile_id"]),
                    person_ref_id=str(profile["person_ref_id"]),
                    source_hash=str(manifest["source_set_sha256"]),
                    approval=approval,
                    predecessor=predecessor,
                )
                != row["generation_id"]
            ):
                raise BiometricReferenceError(
                    "Biometric generation creation binding is invalid."
                )
            for source in sources:
                claim = connection.execute(
                    "SELECT * FROM source_claims WHERE source_key = ?",
                    (source["source_key"],),
                ).fetchone()
                if (
                    claim is None
                    or claim["person_ref_id"] != profile["person_ref_id"]
                    or claim["source_sha256"] != source["source_sha256"]
                    or float(claim["start_seconds"]) != source["start_seconds"]
                    or float(claim["end_seconds"]) != source["end_seconds"]
                ):
                    raise BiometricReferenceError("Reference source claim is invalid.")
        predecessor = str(row["generation_id"])
        manifests.append(manifest)
    if not manifests or rows[-1]["generation_id"] != profile["head_generation_id"]:
        raise BiometricReferenceError("Biometric reference generation head is invalid.")
    if profile["status"] == "deleted" and [
        _hash(manifest) for manifest in manifests
    ] != delete_events[0]["details"].get("tombstone_sha256s"):
        raise BiometricReferenceError("Biometric tombstone history is invalid.")
    return manifests


def replay_reference(
    profile_id: str, *, runtime_root: Optional[Path] = None
) -> dict[str, Any]:
    """Validate full P3 history and return a metadata-only lifecycle receipt."""
    selected = _require_opaque_id(profile_id, "profile_id")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    with _connection(root, create=False) as connection:
        profile = connection.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (selected,)
        ).fetchone()
        if profile is None:
            raise BiometricReferenceError("Biometric reference profile does not exist.")
        events = _validated_event_chain(connection, profile)
        manifests = _validated_generations(connection, profile, events)
        head = connection.execute(
            "SELECT * FROM person_heads WHERE person_ref_id = ?",
            (profile["person_ref_id"],),
        ).fetchone()
        if head is None:
            raise BiometricReferenceError("Person reference head is inconsistent.")
        if profile["status"] == ACTIVE:
            if (
                head["profile_id"] != profile["profile_id"]
                or head["generation_id"] != profile["head_generation_id"]
                or head["status"] != ACTIVE
            ):
                raise BiometricReferenceError("Person reference head is inconsistent.")
        elif head["profile_id"] == profile["profile_id"]:
            if (
                head["generation_id"] != profile["head_generation_id"]
                or head["status"] != profile["status"]
            ):
                raise BiometricReferenceError("Person reference head is inconsistent.")
        elif head["status"] != ACTIVE:
            raise BiometricReferenceError("Replacement person reference head is invalid.")
        generation_rows = connection.execute(
            "SELECT * FROM generations WHERE profile_id = ? ORDER BY sequence",
            (selected,),
        ).fetchall()
        _assert_lifecycle_rows(profile, events, generation_rows)
        active_rows = [row for row in generation_rows if row["status"] == ACTIVE]
        eligible = profile["status"] == ACTIVE
        if eligible:
            if (
                len(active_rows) != 1
                or active_rows[0]["generation_id"] != profile["head_generation_id"]
                or active_rows[0]["eligible_for_materialization"] != 1
            ):
                raise BiometricReferenceError("Active reference eligibility is invalid.")
        elif any(row["eligible_for_materialization"] for row in generation_rows):
            raise BiometricReferenceError("Inactive reference remains materialization eligible.")
        descendants = connection.execute(
            "SELECT * FROM descendants WHERE profile_id = ?", (selected,)
        ).fetchall()
        if len(descendants) != profile["descendant_count"]:
            raise BiometricReferenceError("Descendant inventory head is invalid.")
        generation_by_id = {
            str(row["generation_id"]): row for row in generation_rows
        }
        for row in descendants:
            generation = generation_by_id.get(str(row["generation_id"]))
            if generation is None:
                raise BiometricReferenceError("Descendant generation binding is invalid.")
            registration = {
                "schema_version": DESCENDANT_SCHEMA,
                "profile_id": row["profile_id"],
                "generation_id": row["generation_id"],
                "generation_sha256": row["generation_sha256"],
                "descendant_id": row["descendant_id"],
                "artifact_sha256": row["artifact_sha256"],
                "materialization_receipt_sha256": row[
                    "materialization_receipt_sha256"
                ],
                "materialization_authority_path": row[
                    "materialization_authority_path"
                ],
                "materialization_authority_sha256": row[
                    "materialization_authority_sha256"
                ],
            }
            _require_opaque_id(row["descendant_id"], "descendant_id")
            _require_sha256(row["artifact_sha256"], "descendant artifact_sha256")
            if (
                row["profile_id"] != selected
                or (
                    profile["status"] != "deleted"
                    and row["generation_sha256"] != generation["manifest_sha256"]
                )
                or _hash(registration) != row["registration_sha256"]
                or row["state"]
                not in {"staged", "eligible", "invalidation_pending", "invalidated"}
            ):
                raise BiometricReferenceError("Descendant registration is invalid.")
            materialization = json.loads(row["materialization_receipt_json"])
            _validated_materialization_receipt(
                materialization,
                profile_id=selected,
                generation_id=str(row["generation_id"]),
                generation_sha256=str(row["generation_sha256"]),
                descendant_id=str(row["descendant_id"]),
                artifact_sha256=str(row["artifact_sha256"]),
            )
            if _hash(materialization) != row["materialization_receipt_sha256"]:
                raise BiometricReferenceError("P4 materialization receipt hash mismatch.")
            _require_stored_external_authority(
                materialization,
                stored_path=row["materialization_authority_path"],
                stored_sha256=row["materialization_authority_sha256"],
                p3_root=root,
            )
            _require_content_addressed_receipt(
                root, materialization, str(row["materialization_receipt_sha256"])
            )
            _require_content_addressed_receipt(
                root, registration, str(row["registration_sha256"])
            )
            if row["state"] == "staged":
                if any(
                    row[field] is not None
                    for field in (
                        "promotion_receipt_json",
                        "promotion_receipt_sha256",
                        "promotion_authority_path",
                        "promotion_authority_sha256",
                        "invalidated_at",
                        "invalidation_reason",
                        "invalidation_receipt_json",
                        "invalidation_receipt_sha256",
                    )
                ):
                    raise BiometricReferenceError("Staged descendant state is invalid.")
            else:
                if row["promotion_receipt_json"] is None:
                    raise BiometricReferenceError("Descendant promotion evidence is missing.")
                promotion = json.loads(row["promotion_receipt_json"])
                _validated_promotion_receipt(promotion, row=row)
                if _hash(promotion) != row["promotion_receipt_sha256"]:
                    raise BiometricReferenceError("P4 promotion receipt hash mismatch.")
                _require_stored_external_authority(
                    promotion,
                    stored_path=row["promotion_authority_path"],
                    stored_sha256=row["promotion_authority_sha256"],
                    p3_root=root,
                )
                _require_content_addressed_receipt(
                    root, promotion, str(row["promotion_receipt_sha256"])
                )
            if row["state"] in {"invalidation_pending", "invalidated"}:
                if row["invalidated_at"] is None or row["invalidation_reason"] is None:
                    raise BiometricReferenceError("Descendant invalidation request is invalid.")
            if row["state"] == "invalidated":
                if row["invalidation_receipt_json"] is None:
                    raise BiometricReferenceError(
                        "Descendant invalidation acknowledgment is missing."
                    )
                invalidation = json.loads(row["invalidation_receipt_json"])
                _validated_invalidation_receipt(invalidation, row=row)
                if _hash(invalidation) != row["invalidation_receipt_sha256"]:
                    raise BiometricReferenceError("P4 invalidation receipt hash mismatch.")
                _require_stored_external_authority(
                    invalidation,
                    stored_path=row["invalidation_authority_path"],
                    stored_sha256=row["invalidation_authority_sha256"],
                    p3_root=root,
                )
                _require_content_addressed_receipt(
                    root, invalidation, str(row["invalidation_receipt_sha256"])
                )
        pending_descendants = sum(
            row["state"] == "invalidation_pending" for row in descendants
        )
        if not eligible and any(
            row["state"] in {"staged", "eligible"} for row in descendants
        ):
            raise BiometricReferenceError("Inactive reference has eligible descendants.")
        receipt_rows = connection.execute("SELECT * FROM idempotency").fetchall()
        profile_receipts: list[dict[str, Any]] = []
        for receipt_row in receipt_rows:
            receipt = json.loads(receipt_row["receipt_json"])
            if receipt.get("profile_id") != selected:
                continue
            if (
                receipt.get("schema_version") != RECEIPT_SCHEMA
                or receipt.get("status") != "success"
                or _hash(receipt) != receipt_row["receipt_sha256"]
            ):
                raise BiometricReferenceError("Biometric apply receipt is invalid.")
            anchor = _paths(root)["receipts"] / f"{receipt_row['receipt_sha256']}.json"
            _require_private_regular(anchor, root)
            if read_private_object(anchor) != receipt:
                raise BiometricReferenceError("Immutable apply receipt anchor mismatch.")
            plan, _, plan_sha = _read_plan(str(receipt["run_id"]), root)
            matching_events = [
                event
                for event in events
                if event["event_sha256"] == receipt["event_sha256"]
            ]
            if (
                len(matching_events) != 1
                or plan_sha != receipt["dry_run_sha256"]
                or plan["action"] != receipt["action"]
                or plan["approval"]["approval_id"]
                != matching_events[0]["details"].get("approval_id")
                or receipt.get("generation_id")
                != matching_events[0]["generation_id"]
                or (
                    receipt["action"] in {"create", "supersede"}
                    and receipt.get("generation_sha256")
                    != matching_events[0]["details"].get("manifest_sha256")
                )
                or (
                    receipt["action"] == "delete"
                    and receipt.get("tombstone_sha256s")
                    != matching_events[0]["details"].get("tombstone_sha256s")
                )
            ):
                raise BiometricReferenceError("Apply receipt plan binding mismatch.")
            approval_row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (plan["approval"]["approval_id"],),
            ).fetchone()
            if (
                approval_row is None
                or approval_row["approval_sha256"] != _hash(plan["approval"])
                or approval_row["run_id"] != plan["run_id"]
            ):
                raise BiometricReferenceError("Biometric approval claim is invalid.")
            profile_receipts.append(receipt)
        if len(profile_receipts) != len(events):
            raise BiometricReferenceError("Apply receipt inventory is incomplete.")
        return {
            "schema_version": RECEIPT_SCHEMA,
            "profile_id": selected,
            "person_ref_id": profile["person_ref_id"],
            "status": "blocked" if pending_descendants else "success",
            "reason_code": (
                "descendant_invalidation_pending" if pending_descendants else None
            ),
            "lifecycle_state": f"verified_{profile['status']}",
            "eligible_for_materialization": eligible,
            "head_generation_id": profile["head_generation_id"],
            "head_manifest_sha256": _hash(manifests[-1]),
            "generation_count": len(manifests),
            "event_count": len(events),
            "apply_receipt_count": len(profile_receipts),
            "descendant_count": len(descendants),
            "eligible_descendant_count": sum(
                row["state"] == "eligible" for row in descendants
            ),
            "pending_descendant_count": pending_descendants,
            "will_read_audio": False,
            "will_run_model": False,
            "will_create_embedding": False,
            "will_perform_external_write": False,
            "replayed_at": utc_now(),
        }


def resolve_eligible_reference(
    person_ref_id: str, *, runtime_root: Optional[Path] = None
) -> dict[str, Any]:
    """Return a fully replay-validated restricted generation for P4."""
    selected = _require_opaque_id(person_ref_id, "person_ref_id")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    with _connection(root, create=False) as connection:
        connection.execute("BEGIN")
        try:
            head = connection.execute(
                "SELECT * FROM person_heads WHERE person_ref_id = ?", (selected,)
            ).fetchone()
            if head is None or head["status"] != ACTIVE:
                raise BiometricReferenceError(
                    "Person has no eligible biometric reference."
                )
            profile_id = str(head["profile_id"])
            replay = replay_reference(profile_id, runtime_root=root)
            if replay["eligible_for_materialization"] is not True:
                raise BiometricReferenceError(
                    "Biometric reference is not materialization eligible."
                )
            row = connection.execute(
                "SELECT * FROM generations WHERE generation_id = ?",
                (replay["head_generation_id"],),
            ).fetchone()
            if row is None or row["status"] != ACTIVE:
                raise BiometricReferenceError(
                    "Biometric reference head changed during resolve."
                )
            manifest = json.loads(row["manifest_json"])
            if _hash(manifest) != replay["head_manifest_sha256"]:
                raise BiometricReferenceError(
                    "Biometric reference changed during resolve."
                )
            connection.execute("COMMIT")
            return {
                "profile_id": profile_id,
                "person_ref_id": selected,
                "generation_id": row["generation_id"],
                "generation_sha256": row["manifest_sha256"],
                "reference": manifest,
                "materialization_contract": "stage_then_register_then_promote",
            }
        except Exception:
            _rollback_if_active(connection)
            raise


def _validated_materialization_receipt(
    value: Mapping[str, Any],
    *,
    profile_id: str,
    generation_id: str,
    generation_sha256: str,
    descendant_id: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    receipt = dict(value)
    if set(receipt) != {
        "schema_version",
        "status",
        "profile_id",
        "generation_id",
        "generation_sha256",
        "descendant_id",
        "artifact_sha256",
        "staging_ref_sha256",
        "eligible_for_use",
        "will_perform_external_write",
        "created_at",
    } or _forbidden_keys(receipt):
        raise BiometricReferenceError("P4 materialization receipt shape is invalid.")
    if (
        receipt["schema_version"] != MATERIALIZATION_SCHEMA
        or receipt["status"] != "staged"
        or receipt["profile_id"] != profile_id
        or receipt["generation_id"] != generation_id
        or receipt["generation_sha256"] != generation_sha256
        or receipt["descendant_id"] != descendant_id
        or receipt["artifact_sha256"] != artifact_sha256
        or receipt["eligible_for_use"] is not False
        or receipt["will_perform_external_write"] is not False
    ):
        raise BiometricReferenceError("P4 materialization receipt binding mismatch.")
    _require_sha256(receipt["staging_ref_sha256"], "staging_ref_sha256")
    _require_canonical_utc(receipt["created_at"], "materialization created_at")
    return receipt


def _require_canonical_utc(value: Any, field: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BiometricReferenceError(f"{field} must be canonical UTC.") from exc
    canonical = (
        parsed.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    if parsed.utcoffset() != timezone.utc.utcoffset(None) or text != canonical:
        raise BiometricReferenceError(f"{field} must be canonical UTC.")
    return text


def _validated_promotion_receipt(
    value: Mapping[str, Any], *, row: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    if set(receipt) != {
        "schema_version",
        "status",
        "descendant_id",
        "artifact_sha256",
        "materialization_receipt_sha256",
        "eligible_for_use",
        "will_perform_external_write",
        "promoted_at",
    } or _forbidden_keys(receipt):
        raise BiometricReferenceError("P4 promotion receipt shape is invalid.")
    if (
        receipt["schema_version"] != PROMOTION_SCHEMA
        or receipt["status"] != "promoted"
        or receipt["descendant_id"] != row["descendant_id"]
        or receipt["artifact_sha256"] != row["artifact_sha256"]
        or receipt["materialization_receipt_sha256"]
        != row["materialization_receipt_sha256"]
        or receipt["eligible_for_use"] is not True
        or receipt["will_perform_external_write"] is not False
    ):
        raise BiometricReferenceError("P4 promotion receipt binding mismatch.")
    _require_canonical_utc(receipt["promoted_at"], "promotion promoted_at")
    materialization = json.loads(str(row["materialization_receipt_json"]))
    if receipt["promoted_at"] < materialization["created_at"]:
        raise BiometricReferenceError("P4 promotion predates materialization.")
    return receipt


def _validated_invalidation_receipt(
    value: Mapping[str, Any], *, row: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = dict(value)
    if set(receipt) != {
        "schema_version",
        "status",
        "descendant_id",
        "artifact_sha256",
        "reason",
        "evidence_sha256",
        "will_perform_external_write",
        "acknowledged_at",
    } or _forbidden_keys(receipt):
        raise BiometricReferenceError("P4 invalidation receipt shape is invalid.")
    if (
        receipt["schema_version"] != INVALIDATION_SCHEMA
        or receipt["status"] not in {"invalidated", "deleted"}
        or receipt["descendant_id"] != row["descendant_id"]
        or receipt["artifact_sha256"] != row["artifact_sha256"]
        or receipt["reason"] != row["invalidation_reason"]
        or receipt["will_perform_external_write"] is not False
    ):
        raise BiometricReferenceError("P4 invalidation receipt binding mismatch.")
    _require_sha256(receipt["evidence_sha256"], "invalidation evidence_sha256")
    _require_canonical_utc(
        receipt["acknowledged_at"], "invalidation acknowledged_at"
    )
    if receipt["acknowledged_at"] < str(row["invalidated_at"]):
        raise BiometricReferenceError("P4 invalidation acknowledgment predates request.")
    return receipt


def register_descendant(
    profile_id: str,
    generation_id: str,
    descendant_id: str,
    artifact_sha256: str,
    *,
    materialization_receipt: Mapping[str, Any],
    authority_receipt_path: Path,
    p4_authority_root: Path,
    approval_token: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Register one P4 artifact so P3 lifecycle revocation reaches it."""
    selected_profile = _require_opaque_id(profile_id, "profile_id")
    selected_generation = _require_opaque_id(generation_id, "generation_id")
    selected_descendant = _require_opaque_id(descendant_id, "descendant_id")
    selected_sha = _require_sha256(artifact_sha256, "artifact_sha256")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    with _connection(root, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            generation = connection.execute(
                "SELECT * FROM generations WHERE generation_id = ? AND profile_id = ?",
                (selected_generation, selected_profile),
            ).fetchone()
            profile = connection.execute(
                "SELECT * FROM profiles WHERE profile_id = ?", (selected_profile,)
            ).fetchone()
            if generation is None or profile is None:
                raise BiometricReferenceError("Descendant parent does not exist.")
            existing = connection.execute(
                "SELECT * FROM descendants WHERE descendant_id = ?",
                (selected_descendant,),
            ).fetchone()
            bound_generation_sha = (
                str(existing["generation_sha256"])
                if existing is not None
                else str(generation["manifest_sha256"])
            )
            validated_materialization = _validated_materialization_receipt(
                materialization_receipt,
                profile_id=selected_profile,
                generation_id=selected_generation,
                generation_sha256=bound_generation_sha,
                descendant_id=selected_descendant,
                artifact_sha256=selected_sha,
            )
            materialization_sha = _hash(validated_materialization)
            authority_path, authority_sha = _require_external_authority_receipt(
                validated_materialization,
                path=authority_receipt_path,
                authority_root=p4_authority_root,
                p3_root=root,
            )
            required = (
                f"REGISTER_BIOMETRIC_DESCENDANT:{selected_generation}:"
                f"{selected_descendant}:{selected_sha}:{materialization_sha}"
            )
            if approval_token != required:
                raise BiometricReferenceError(f"Registration requires token {required}.")
            registration_identity = {
                "schema_version": DESCENDANT_SCHEMA,
                "profile_id": selected_profile,
                "generation_id": selected_generation,
                "generation_sha256": bound_generation_sha,
                "descendant_id": selected_descendant,
                "artifact_sha256": selected_sha,
                "materialization_receipt_sha256": materialization_sha,
                "materialization_authority_path": authority_path,
                "materialization_authority_sha256": authority_sha,
            }
            registration_sha = _hash(registration_identity)
            if existing is not None:
                if existing["registration_sha256"] != registration_sha:
                    raise BiometricReferenceError("Descendant registration conflicts.")
                if (
                    existing["materialization_authority_path"] != authority_path
                    or existing["materialization_authority_sha256"] != authority_sha
                ):
                    raise BiometricReferenceError("P4 materialization authority conflicts.")
                _recover_or_require_receipt(
                    root, validated_materialization, materialization_sha
                )
                _recover_or_require_receipt(
                    root, registration_identity, registration_sha
                )
                connection.execute("COMMIT")
                return {
                    **registration_identity,
                    "registered_at": existing["registered_at"],
                    "state": existing["state"],
                    "required_promotion_token": (
                        f"ACK_BIOMETRIC_DESCENDANT_PROMOTION:{selected_descendant}:"
                        f"{materialization_sha}"
                    ),
                    "idempotent_replay": True,
                }
            if (
                profile["status"] != ACTIVE
                or generation["status"] != ACTIVE
                or generation["eligible_for_materialization"] != 1
                or profile["head_generation_id"] != selected_generation
            ):
                raise BiometricReferenceError("Descendant parent is not eligible.")
            _stage_content_addressed_receipt(root, validated_materialization)
            _stage_content_addressed_receipt(root, registration_identity)
            registered_at = utc_now()
            connection.execute(
                """
                INSERT INTO descendants
                (descendant_id, profile_id, generation_id, generation_sha256,
                 artifact_sha256,
                 registered_at, state, materialization_receipt_json,
                 materialization_receipt_sha256, materialization_authority_path,
                 materialization_authority_sha256, promotion_receipt_json,
                 promotion_receipt_sha256, promotion_authority_path,
                 promotion_authority_sha256, invalidated_at, invalidation_reason,
                 invalidation_receipt_json, invalidation_receipt_sha256,
                 invalidation_authority_path, invalidation_authority_sha256,
                 registration_sha256)
                VALUES (?, ?, ?, ?, ?, ?, 'staged', ?, ?, ?, ?, NULL, NULL,
                        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
                """,
                (
                    selected_descendant,
                    selected_profile,
                    selected_generation,
                    generation["manifest_sha256"],
                    selected_sha,
                    registered_at,
                    _canonical_json(validated_materialization),
                    materialization_sha,
                    authority_path,
                    authority_sha,
                    registration_sha,
                ),
            )
            updated = connection.execute(
                """
                UPDATE profiles SET descendant_count = descendant_count + 1
                WHERE profile_id = ?
                """,
                (selected_profile,),
            )
            if updated.rowcount != 1:
                raise BiometricReferenceError("Descendant inventory update failed.")
            connection.execute("COMMIT")
            _promote_staged_receipt(root, validated_materialization)
            _promote_staged_receipt(root, registration_identity)
        except Exception:
            _rollback_if_active(connection)
            raise
    return {
        **registration_identity,
        "registered_at": registered_at,
        "state": "staged",
        "required_promotion_token": (
            f"ACK_BIOMETRIC_DESCENDANT_PROMOTION:{selected_descendant}:"
            f"{materialization_sha}"
        ),
        "idempotent_replay": False,
    }


def acknowledge_descendant_promotion(
    descendant_id: str,
    promotion_receipt: Mapping[str, Any],
    *,
    authority_receipt_path: Path,
    p4_authority_root: Path,
    approval_token: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    selected = _require_opaque_id(descendant_id, "descendant_id")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    with _connection(root, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM descendants WHERE descendant_id = ?", (selected,)
            ).fetchone()
            if row is None:
                raise BiometricReferenceError("Descendant registration does not exist.")
            required = (
                f"ACK_BIOMETRIC_DESCENDANT_PROMOTION:{selected}:"
                f"{row['materialization_receipt_sha256']}"
            )
            if approval_token != required:
                raise BiometricReferenceError(f"Promotion requires token {required}.")
            receipt = _validated_promotion_receipt(promotion_receipt, row=row)
            receipt_sha = _hash(receipt)
            authority_path, authority_sha = _require_external_authority_receipt(
                receipt,
                path=authority_receipt_path,
                authority_root=p4_authority_root,
                p3_root=root,
            )
            if row["promotion_receipt_sha256"] is not None:
                if row["promotion_receipt_sha256"] != receipt_sha:
                    raise BiometricReferenceError("P4 promotion receipt conflicts.")
                if (
                    row["promotion_authority_path"] != authority_path
                    or row["promotion_authority_sha256"] != authority_sha
                ):
                    raise BiometricReferenceError("P4 promotion authority conflicts.")
                _recover_or_require_receipt(root, receipt, receipt_sha)
                connection.execute("COMMIT")
                return {**receipt, "idempotent_replay": True}
            if row["state"] != "staged":
                raise BiometricReferenceError("Descendant is not awaiting promotion.")
            _stage_content_addressed_receipt(root, receipt)
            connection.execute(
                """
                UPDATE descendants SET state = 'eligible', promotion_receipt_json = ?,
                promotion_receipt_sha256 = ?, promotion_authority_path = ?,
                promotion_authority_sha256 = ?
                WHERE descendant_id = ? AND state = 'staged'
                """,
                (
                    _canonical_json(receipt), receipt_sha, authority_path,
                    authority_sha, selected,
                ),
            )
            connection.execute("COMMIT")
            _promote_staged_receipt(root, receipt)
        except Exception:
            _rollback_if_active(connection)
            raise
    return {**receipt, "idempotent_replay": False}


def acknowledge_descendant_invalidation(
    descendant_id: str,
    invalidation_receipt: Mapping[str, Any],
    *,
    authority_receipt_path: Path,
    p4_authority_root: Path,
    approval_token: str,
    runtime_root: Optional[Path] = None,
) -> dict[str, Any]:
    selected = _require_opaque_id(descendant_id, "descendant_id")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    with _connection(root, create=True) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM descendants WHERE descendant_id = ?", (selected,)
            ).fetchone()
            if row is None or row["invalidation_reason"] is None:
                raise BiometricReferenceError("Descendant has no invalidation request.")
            required = (
                f"ACK_BIOMETRIC_DESCENDANT_INVALIDATION:{selected}:"
                f"{row['artifact_sha256']}:{row['invalidation_reason']}"
            )
            if approval_token != required:
                raise BiometricReferenceError(f"Invalidation requires token {required}.")
            receipt = _validated_invalidation_receipt(
                invalidation_receipt, row=row
            )
            receipt_sha = _hash(receipt)
            authority_path, authority_sha = _require_external_authority_receipt(
                receipt,
                path=authority_receipt_path,
                authority_root=p4_authority_root,
                p3_root=root,
            )
            if row["invalidation_receipt_sha256"] is not None:
                if row["invalidation_receipt_sha256"] != receipt_sha:
                    raise BiometricReferenceError("P4 invalidation receipt conflicts.")
                if (
                    row["invalidation_authority_path"] != authority_path
                    or row["invalidation_authority_sha256"] != authority_sha
                ):
                    raise BiometricReferenceError("P4 invalidation authority conflicts.")
                _recover_or_require_receipt(root, receipt, receipt_sha)
                connection.execute("COMMIT")
                return {**receipt, "idempotent_replay": True}
            if row["state"] != "invalidation_pending":
                raise BiometricReferenceError("Descendant is not awaiting invalidation.")
            _stage_content_addressed_receipt(root, receipt)
            connection.execute(
                """
                UPDATE descendants SET state = 'invalidated',
                invalidation_receipt_json = ?, invalidation_receipt_sha256 = ?,
                invalidation_authority_path = ?, invalidation_authority_sha256 = ?
                WHERE descendant_id = ? AND state = 'invalidation_pending'
                """,
                (
                    _canonical_json(receipt), receipt_sha, authority_path,
                    authority_sha, selected,
                ),
            )
            connection.execute("COMMIT")
            _promote_staged_receipt(root, receipt)
        except Exception:
            _rollback_if_active(connection)
            raise
    return {**receipt, "idempotent_replay": False}


def descendant_is_eligible(
    descendant_id: str, *, runtime_root: Optional[Path] = None
) -> bool:
    selected = _require_opaque_id(descendant_id, "descendant_id")
    root = (runtime_root or DEFAULT_RUNTIME_ROOT).expanduser().absolute()
    if not _paths(root)["database"].exists():
        return False
    with _connection(root, create=False) as connection:
        connection.execute("BEGIN")
        try:
            initial = connection.execute(
                "SELECT profile_id FROM descendants WHERE descendant_id = ?",
                (selected,),
            ).fetchone()
            if initial is None:
                connection.execute("COMMIT")
                return False
            replay = replay_reference(str(initial["profile_id"]), runtime_root=root)
            row = connection.execute(
                """
                SELECT d.state, d.profile_id, p.status AS profile_status,
                       g.status AS generation_status,
                       g.eligible_for_materialization
                FROM descendants d
                JOIN profiles p ON p.profile_id = d.profile_id
                JOIN generations g ON g.generation_id = d.generation_id
                WHERE d.descendant_id = ?
                """,
                (selected,),
            ).fetchone()
            eligible = bool(
                row is not None
                and replay["status"] == "success"
                and row["state"] == "eligible"
                and row["profile_status"] == ACTIVE
                and row["generation_status"] == ACTIVE
                and row["eligible_for_materialization"] == 1
            )
            connection.execute("COMMIT")
            return eligible
        except Exception:
            _rollback_if_active(connection)
            raise
