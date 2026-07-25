"""Safe transcript artifact resolution for user-scoped processing workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import conversation_processing
import transcript_store


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
