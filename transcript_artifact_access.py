"""Safe transcript artifact resolution for user-scoped processing workflows."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid5

import conversation_processing
import transcript_store


_PREPARATION_ID_NAMESPACE = UUID("f8c325d2-fd13-50d9-a5f0-b6bb2688c6a3")


class TranscriptArtifactAccessError(ValueError):
    """Raised when a document has no safe, provenance-preserving artifact."""


@dataclass(frozen=True)
class ResolvedTranscriptArtifact:
    path: Path
    location: str
    expected_sha256: str
    actual_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "location": self.location,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


@dataclass(frozen=True)
class PrivateTranscriptIdentitySnapshot:
    path: Path
    source_path: Path
    source_transcript_sha256: str
    preparation_transcript_sha256: str
    source_was_derived: bool
    conversation_id: str
    recording_id: str


def _valid_uuid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(UUID(text))
    except ValueError:
        return ""


def materialize_private_transcript_identity_snapshot(
    transcript_path: Path,
    *,
    document_id: str,
    state_root: Path,
) -> PrivateTranscriptIdentitySnapshot:
    """Materialize a replayable schema-2 transcript without changing its source."""

    source_path = transcript_path.expanduser().resolve(strict=True)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        source = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise TranscriptArtifactAccessError(
            f"Selected transcript artifact is not valid JSON: {exc}"
        ) from exc
    if not isinstance(source, dict):
        raise TranscriptArtifactAccessError(
            "Selected transcript artifact must contain a JSON object."
        )
    payload = dict(source)
    conversation_id = _valid_uuid(payload.get("conversation_id"))
    recording_id = _valid_uuid(payload.get("recording_id"))
    source_was_derived = bool(
        int(payload.get("schema_version") or 1) < 2
        or not conversation_id
        or not recording_id
    )
    if not conversation_id:
        conversation_id = str(
            uuid5(
                _PREPARATION_ID_NAMESPACE,
                f"conversation\x1f{document_id}\x1f{source_sha256}",
            )
        )
    if not recording_id:
        legacy_import = payload.get("legacy_import")
        legacy_import = legacy_import if isinstance(legacy_import, dict) else {}
        recording_authority = str(
            payload.get("source_media_sha256")
            or payload.get("source_media_path")
            or legacy_import.get("source_sha256")
            or source_sha256
        )
        recording_id = str(
            uuid5(
                _PREPARATION_ID_NAMESPACE,
                f"recording\x1f{recording_authority}",
            )
        )
    payload["schema_version"] = max(int(payload.get("schema_version") or 1), 2)
    payload["conversation_id"] = conversation_id
    payload["recording_id"] = recording_id
    prepared_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    prepared_sha256 = hashlib.sha256(prepared_bytes).hexdigest()

    snapshot_root = (
        state_root.expanduser().resolve() / "speaker-preprocessing-snapshots"
    )
    snapshot_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot_root.chmod(0o700)
    document_hash = hashlib.sha256(str(document_id).encode("utf-8")).hexdigest()
    snapshot_path = snapshot_root / (
        f"{document_hash[:24]}-{source_sha256[:24]}.transcript.json"
    )
    if snapshot_path.exists():
        if snapshot_path.is_symlink() or snapshot_path.stat().st_nlink != 1:
            raise TranscriptArtifactAccessError(
                "Private transcript snapshot containment is invalid."
            )
        if snapshot_path.read_bytes() != prepared_bytes:
            raise TranscriptArtifactAccessError(
                "Private transcript snapshot content drifted."
            )
    else:
        descriptor = os.open(
            snapshot_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(prepared_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    snapshot_path.chmod(0o600)
    return PrivateTranscriptIdentitySnapshot(
        path=snapshot_path,
        source_path=source_path,
        source_transcript_sha256=source_sha256,
        preparation_transcript_sha256=prepared_sha256,
        source_was_derived=source_was_derived,
        conversation_id=conversation_id,
        recording_id=recording_id,
    )


def _is_transcript_path(path: Path) -> bool:
    return path.name.endswith(".transcript.json")


def _stored_path_under_root(path: Path, store_root: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise TranscriptArtifactAccessError(
            "Selected conversation does not have an accessible transcript artifact."
        ) from exc
    artifacts_root = (store_root.expanduser().resolve() / "artifacts")
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise TranscriptArtifactAccessError(
            "Recorded stored transcript path is outside the transcript store."
        ) from exc
    return resolved


def resolve_transcript_artifact(
    document: dict[str, Any],
    *,
    store_root: Optional[Path] = None,
) -> ResolvedTranscriptArtifact:
    """Resolve source first, then a store-bounded hash-verified copied artifact."""
    if str(document.get("kind") or "") != "transcript":
        raise TranscriptArtifactAccessError(
            "Selected document is not a transcript artifact."
        )
    expected_sha256 = str(document.get("artifact_sha256") or "")
    source_text = str(document.get("source_path") or "")
    if source_text:
        source_path = Path(source_text).expanduser()
        if source_path.is_file() and _is_transcript_path(source_path):
            resolved_source = source_path.resolve()
            return ResolvedTranscriptArtifact(
                path=resolved_source,
                location="source",
                expected_sha256=expected_sha256,
                actual_sha256=transcript_store.sha256_file(resolved_source),
            )

    stored_text = str(document.get("stored_path") or "")
    if not stored_text or not _is_transcript_path(Path(stored_text)):
        raise TranscriptArtifactAccessError(
            "Selected conversation does not have an accessible transcript artifact."
        )
    selected_store_root = transcript_store.store_dir(store_root)
    stored_path = _stored_path_under_root(Path(stored_text), selected_store_root)
    actual_sha256 = transcript_store.sha256_file(stored_path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise TranscriptArtifactAccessError(
            "Stored transcript artifact hash does not match the indexed artifact."
        )
    return ResolvedTranscriptArtifact(
        path=stored_path,
        location="stored",
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
    )


def ensure_resolved_transcript_identity(
    document: dict[str, Any],
    *,
    store_root: Optional[Path] = None,
) -> tuple[dict[str, Any], ResolvedTranscriptArtifact]:
    """Add durable IDs and synchronize the exact selected artifact with its row."""
    resolved = resolve_transcript_artifact(document, store_root=store_root)
    payload = conversation_processing.ensure_transcript_identity(resolved.path)
    synchronized = transcript_store.synchronize_document_artifact(
        str(document.get("id") or ""),
        artifact_path=resolved.path,
        location=resolved.location,
        root=store_root,
    )
    refreshed = ResolvedTranscriptArtifact(
        path=resolved.path,
        location=resolved.location,
        expected_sha256=str(synchronized["artifact_sha256"]),
        actual_sha256=str(synchronized["artifact_sha256"]),
    )
    return payload, refreshed


def read_resolved_transcript(
    resolved: ResolvedTranscriptArtifact,
) -> dict[str, Any]:
    try:
        payload = json.loads(resolved.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptArtifactAccessError(
            f"Selected transcript artifact is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TranscriptArtifactAccessError(
            "Selected transcript artifact must contain a JSON object."
        )
    return payload
