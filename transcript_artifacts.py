from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

ARTIFACT_SCHEMA_VERSION = 2


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone().isoformat()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def _valid_opaque_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(UUID(text))
    except ValueError:
        return ""


@dataclass
class TranscriptArtifact:
    source_media_path: str
    working_media_path: str
    backend: str
    duration_seconds: float
    recording_start: datetime
    recording_end: datetime
    transcript_window_start_seconds: float
    transcript_window_end_seconds: float
    utterance_count: int
    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    recording_id: str = field(default_factory=lambda: str(uuid4()))
    transcript_text: str = ""
    utterances: list[dict[str, Any]] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)
    event: Optional[dict[str, Any]] = None
    transcript_title: str = "Transcript"
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "conversation_id": self.conversation_id,
                "recording_id": self.recording_id,
                "source_media_path": self.source_media_path,
                "working_media_path": self.working_media_path,
                "backend": self.backend,
                "duration_seconds": self.duration_seconds,
                "recording_start": self.recording_start,
                "recording_end": self.recording_end,
                "transcript_window_start_seconds": self.transcript_window_start_seconds,
                "transcript_window_end_seconds": self.transcript_window_end_seconds,
                "utterance_count": self.utterance_count,
                "transcript_text": self.transcript_text,
                "utterances": self.utterances,
                "output_paths": self.output_paths,
                "event": self.event,
                "transcript_title": self.transcript_title,
            }
        )

    def write(self, path: Path, *, preserve_existing_recording_id: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if isinstance(existing, dict):
                self.conversation_id = _valid_opaque_id(existing.get("conversation_id")) or self.conversation_id
                if preserve_existing_recording_id:
                    self.recording_id = _valid_opaque_id(existing.get("recording_id")) or self.recording_id
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
