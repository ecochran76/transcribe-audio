from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transcript_artifacts import json_ready

DEPOSITION_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class DepositAction:
    action_type: str
    target_kind: str
    status: str = "preview"
    target_id: str = ""
    target_profile: str = ""
    source_paths: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "action_type": self.action_type,
                "target_kind": self.target_kind,
                "target_id": self.target_id,
                "target_profile": self.target_profile,
                "status": self.status,
                "source_paths": self.source_paths,
                "evidence": self.evidence,
                "metadata": self.metadata,
            }
        )


@dataclass
class MemoryHarvestCandidate:
    text: str
    kind: str
    evidence: str = ""
    target_group_id: str = "transcribe_audio_main"
    source_readout_path: str = ""
    source_ids: list[str] = field(default_factory=list)
    status: str = "preview"
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            self.candidate_id = stable_id(self.target_group_id, self.kind, self.text, self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "candidate_id": self.candidate_id,
                "status": self.status,
                "target_group_id": self.target_group_id,
                "kind": self.kind,
                "text": self.text,
                "evidence": self.evidence,
                "source_readout_path": self.source_readout_path,
                "source_ids": self.source_ids,
            }
        )


@dataclass
class DepositPreview:
    source_readout_path: str
    source_route_path: str = ""
    source_transcript_path: str = ""
    selected_candidate: dict[str, Any] = field(default_factory=dict)
    actions: list[DepositAction] = field(default_factory=list)
    memory_candidates: list[MemoryHarvestCandidate] = field(default_factory=list)
    review_required: bool = True
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = DEPOSITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "source_readout_path": self.source_readout_path,
                "source_route_path": self.source_route_path,
                "source_transcript_path": self.source_transcript_path,
                "selected_candidate": self.selected_candidate,
                "review_required": self.review_required,
                "warnings": self.warnings,
                "actions": [action.to_dict() for action in self.actions],
                "memory_candidates": [candidate.to_dict() for candidate in self.memory_candidates],
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


@dataclass
class AppliedDepositFile:
    source_path: str
    destination_path: str
    status: str
    sha256: str = ""
    bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "source_path": self.source_path,
                "destination_path": self.destination_path,
                "status": self.status,
                "sha256": self.sha256,
                "bytes": self.bytes,
            }
        )


@dataclass
class AppliedDepositAction:
    action_type: str
    target_kind: str
    status: str
    target_id: str = ""
    target_profile: str = ""
    files: list[AppliedDepositFile] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "action_type": self.action_type,
                "target_kind": self.target_kind,
                "target_id": self.target_id,
                "target_profile": self.target_profile,
                "status": self.status,
                "files": [file.to_dict() for file in self.files],
                "reason": self.reason,
                "metadata": self.metadata,
            }
        )


@dataclass
class DepositApplyResult:
    source_preview_path: str
    actions: list[AppliedDepositAction] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = DEPOSITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "source_preview_path": self.source_preview_path,
                "actions": [action.to_dict() for action in self.actions],
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


@dataclass
class AppliedMemoryHarvestCandidate:
    candidate_id: str
    target_group_id: str
    status: str
    kind: str = ""
    reason: str = ""
    review_decision: str = ""
    review_reason: str = ""
    source_readout_path: str = ""
    source_ids: list[str] = field(default_factory=list)
    duplicate_check: dict[str, Any] = field(default_factory=dict)
    graphiti_command: list[str] = field(default_factory=list)
    graphiti_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "candidate_id": self.candidate_id,
                "target_group_id": self.target_group_id,
                "status": self.status,
                "kind": self.kind,
                "reason": self.reason,
                "review_decision": self.review_decision,
                "review_reason": self.review_reason,
                "source_readout_path": self.source_readout_path,
                "source_ids": self.source_ids,
                "duplicate_check": self.duplicate_check,
                "graphiti_command": self.graphiti_command,
                "graphiti_result": self.graphiti_result,
            }
        )


@dataclass
class MemoryHarvestApplyResult:
    source_preview_path: str
    mode: str
    candidates: list[AppliedMemoryHarvestCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = DEPOSITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "source_preview_path": self.source_preview_path,
                "mode": self.mode,
                "warnings": self.warnings,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
