from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transcript_artifacts import json_ready

READOUT_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


@dataclass
class Readout:
    source_artifact_path: str
    provider: dict[str, Any]
    generated_at: str
    title: str
    summary: str
    participants: list[dict[str, Any]] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    matter_candidates: list[dict[str, Any]] = field(default_factory=list)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    contextualization: dict[str, Any] = field(default_factory=dict)
    schema_version: int = READOUT_SCHEMA_VERSION

    @classmethod
    def from_model_payload(
        cls,
        payload: dict[str, Any],
        *,
        source_artifact_path: Path,
        provider: dict[str, Any],
        contextualization: dict[str, Any] | None = None,
    ) -> "Readout":
        return cls(
            source_artifact_path=str(source_artifact_path),
            provider=provider,
            generated_at=utc_now_iso(),
            title=str(payload.get("title") or "Transcript Readout").strip(),
            summary=str(payload.get("summary") or "").strip(),
            participants=normalize_object_list(payload.get("participants")),
            topics=normalize_string_list(payload.get("topics")),
            action_items=normalize_object_list(payload.get("action_items")),
            unresolved_questions=normalize_string_list(payload.get("unresolved_questions")),
            matter_candidates=normalize_object_list(payload.get("matter_candidates")),
            memory_candidates=normalize_object_list(payload.get("memory_candidates")),
            key_decisions=normalize_string_list(payload.get("key_decisions")),
            risks=normalize_string_list(payload.get("risks")),
            next_steps=normalize_string_list(payload.get("next_steps")),
            contextualization=contextualization or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "source_artifact_path": self.source_artifact_path,
                "provider": self.provider,
                "generated_at": self.generated_at,
                "title": self.title,
                "summary": self.summary,
                "participants": self.participants,
                "topics": self.topics,
                "action_items": self.action_items,
                "unresolved_questions": self.unresolved_questions,
                "matter_candidates": self.matter_candidates,
                "memory_candidates": self.memory_candidates,
                "key_decisions": self.key_decisions,
                "risks": self.risks,
                "next_steps": self.next_steps,
                "contextualization": self.contextualization,
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")

    def to_markdown(self) -> str:
        sections = [f"# {self.title}", "", f"Generated: {self.generated_at}", ""]
        sections.extend(["## Summary", "", self.summary or "_No summary returned._", ""])
        sections.extend(markdown_list_section("Participants", self.participants, format_participant))
        sections.extend(markdown_list_section("Topics", self.topics, str))
        sections.extend(markdown_list_section("Key Decisions", self.key_decisions, str))
        sections.extend(markdown_list_section("Action Items", self.action_items, format_action_item))
        sections.extend(markdown_list_section("Unresolved Questions", self.unresolved_questions, str))
        sections.extend(markdown_list_section("Matter Candidates", self.matter_candidates, format_candidate))
        sections.extend(markdown_list_section("Memory Candidates", self.memory_candidates, format_memory_candidate))
        sections.extend(markdown_list_section("Risks", self.risks, str))
        sections.extend(markdown_list_section("Next Steps", self.next_steps, str))
        if self.contextualization:
            warnings = self.contextualization.get("warnings")
            if isinstance(warnings, list):
                sections.extend(markdown_list_section("Context Warnings", warnings, str))
            sources = self.contextualization.get("supporting_context_sources")
            if isinstance(sources, list):
                sections.extend(markdown_list_section("Supporting Context Sources", sources, format_supporting_source))
        return "\n".join(sections).rstrip() + "\n"

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


def markdown_list_section(title: str, values: list[Any], formatter) -> list[str]:
    lines = [f"## {title}", ""]
    if not values:
        lines.extend(["_None identified._", ""])
        return lines
    for value in values:
        lines.append(f"- {formatter(value)}")
    lines.append("")
    return lines


def format_participant(value: dict[str, Any]) -> str:
    name = str(value.get("name") or value.get("email") or "Unknown").strip()
    role = str(value.get("role") or "").strip()
    evidence = str(value.get("evidence") or "").strip()
    parts = [name]
    if role:
        parts.append(f"role: {role}")
    if evidence:
        parts.append(f"evidence: {evidence}")
    return " | ".join(parts)


def format_action_item(value: dict[str, Any]) -> str:
    task = str(value.get("task") or value.get("description") or "").strip()
    owner = str(value.get("owner") or "").strip()
    due = str(value.get("due") or "").strip()
    status = str(value.get("status") or "").strip()
    parts = [task or "Unspecified action"]
    if owner:
        parts.append(f"owner: {owner}")
    if due:
        parts.append(f"due: {due}")
    if status:
        parts.append(f"status: {status}")
    return " | ".join(parts)


def format_candidate(value: dict[str, Any]) -> str:
    label = str(value.get("label") or value.get("name") or "Unspecified matter").strip()
    confidence = value.get("confidence")
    evidence = str(value.get("evidence") or "").strip()
    parts = [label]
    if confidence is not None:
        parts.append(f"confidence: {confidence}")
    if evidence:
        parts.append(f"evidence: {evidence}")
    return " | ".join(parts)


def format_memory_candidate(value: dict[str, Any]) -> str:
    text = str(value.get("text") or value.get("memory") or "").strip()
    kind = str(value.get("kind") or "").strip()
    evidence = str(value.get("evidence") or "").strip()
    parts = [text or "Unspecified memory"]
    if kind:
        parts.append(f"kind: {kind}")
    if evidence:
        parts.append(f"evidence: {evidence}")
    return " | ".join(parts)


def format_supporting_source(value: dict[str, Any]) -> str:
    label = str(value.get("label") or value.get("source_id") or "Supporting source").strip()
    source_type = str(value.get("source_type") or "").strip()
    source_id = str(value.get("source_id") or "").strip()
    parts = [label]
    if source_type:
        parts.append(f"type: {source_type}")
    if source_id:
        parts.append(f"id: {source_id}")
    return " | ".join(parts)
