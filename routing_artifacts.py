from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from transcript_artifacts import json_ready

ROUTING_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_confidence(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def normalize_string(value: Any) -> str:
    return str(value or "").strip()


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (normalize_string(item) for item in value) if item]


def unique_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


@dataclass
class ProvenanceSource:
    source_type: str
    label: str
    source_id: str = ""
    uri: str = ""
    snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "source_type": self.source_type,
                "source_id": self.source_id,
                "label": self.label,
                "uri": self.uri,
                "snippet": self.snippet,
                "metadata": self.metadata,
            }
        )


@dataclass
class ContextProvenancePack:
    sources: list[ProvenanceSource] = field(default_factory=list)
    excluded_sources: list[ProvenanceSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = ROUTING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "sources": [source.to_dict() for source in self.sources],
                "excluded_sources": [source.to_dict() for source in self.excluded_sources],
                "warnings": self.warnings,
            }
        )


@dataclass
class RouteCandidate:
    label: str
    target_kind: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    source: str = "readout"
    target_id: str = ""
    provenance_source_ids: list[str] = field(default_factory=list)
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "label": self.label,
                "target_kind": self.target_kind,
                "target_id": self.target_id,
                "confidence": self.confidence,
                "evidence": self.evidence,
                "source": self.source,
                "provenance_source_ids": self.provenance_source_ids,
                "rejected_reason": self.rejected_reason,
            }
        )


@dataclass
class RouteDecision:
    source_transcript_path: str
    source_readout_path: str
    status: str
    selected_candidate: Optional[RouteCandidate]
    candidates: list[RouteCandidate]
    rejected_candidates: list[RouteCandidate]
    provenance_pack: ContextProvenancePack
    review_required: bool
    fallback: str
    confidence_threshold: float
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = ROUTING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "source_transcript_path": self.source_transcript_path,
                "source_readout_path": self.source_readout_path,
                "status": self.status,
                "selected_candidate": self.selected_candidate.to_dict() if self.selected_candidate else None,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "rejected_candidates": [candidate.to_dict() for candidate in self.rejected_candidates],
                "provenance_pack": self.provenance_pack.to_dict(),
                "review_required": self.review_required,
                "fallback": self.fallback,
                "confidence_threshold": self.confidence_threshold,
                "warnings": self.warnings,
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


@dataclass
class ReviewQueueItem:
    route_decision_path: str
    reason: str
    selected_label: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = ROUTING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "created_at": self.created_at,
                "route_decision_path": self.route_decision_path,
                "selected_label": self.selected_label,
                "reason": self.reason,
            }
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


def provenance_from_transcript(transcript: dict[str, Any]) -> ContextProvenancePack:
    sources: list[ProvenanceSource] = []
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    if event:
        event_label = normalize_string(event.get("summary")) or "Calendar event"
        sources.append(
            ProvenanceSource(
                source_type="calendar_event",
                source_id=normalize_string(event.get("id") or event.get("event_id")),
                label=event_label,
                snippet=event_label,
                metadata={
                    "start": event.get("start"),
                    "end": event.get("end"),
                    "participants": event.get("participants") or event.get("attendees") or [],
                    "provider": event.get("provider"),
                },
            )
        )
        for item in event.get("matching_calendars") or []:
            if not isinstance(item, dict):
                continue
            calendar_id = normalize_string(item.get("calendar_id"))
            event_summary = normalize_string(item.get("event_summary")) or event_label
            calendar_summary = normalize_string(item.get("calendar_summary")) or calendar_id or "calendar"
            source_id = normalize_string(item.get("event_id") or event_summary or calendar_id)
            sources.append(
                ProvenanceSource(
                    source_type="gws_calendar_overlap",
                    source_id=source_id,
                    label=calendar_summary,
                    snippet=event_summary,
                    metadata={
                        "calendar_id": calendar_id,
                        "calendar_summary": calendar_summary,
                        "event_summary": event_summary,
                        "event_start": item.get("event_start"),
                        "event_end": item.get("event_end"),
                        "overlap_seconds": item.get("overlap_seconds"),
                        "coverage": item.get("coverage"),
                        "access_role": item.get("access_role"),
                    },
                )
            )
    return ContextProvenancePack(sources=sources)


def candidates_from_readout(
    readout: dict[str, Any],
    provenance_pack: ContextProvenancePack,
    *,
    target_kind: str = "matter",
) -> list[RouteCandidate]:
    provenance_source_ids = unique_strings([source.source_id or source.label for source in provenance_pack.sources])
    candidates = []
    for item in readout.get("matter_candidates") or []:
        if not isinstance(item, dict):
            continue
        label = normalize_string(item.get("label") or item.get("name"))
        if not label:
            continue
        evidence = normalize_string(item.get("evidence"))
        confidence = normalize_confidence(item.get("confidence"), default=0.5)
        candidates.append(
            RouteCandidate(
                label=label,
                target_kind=target_kind,
                target_id=stable_id(target_kind, label),
                confidence=confidence,
                evidence=[evidence] if evidence else [],
                source="readout.matter_candidates",
                provenance_source_ids=provenance_source_ids,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)


def candidates_from_graphiti_sources(sources: list[ProvenanceSource], *, target_kind: str = "matter") -> list[RouteCandidate]:
    candidates = []
    for source in sources:
        if source.source_type != "graphiti_node":
            continue
        label = normalize_string(source.metadata.get("candidate_label")) or normalize_string(source.label)
        if not label:
            continue
        confidence = normalize_confidence(source.metadata.get("candidate_confidence"), default=0.45)
        candidates.append(
            RouteCandidate(
                label=label,
                target_kind=target_kind,
                target_id=stable_id(target_kind, "graphiti", source.metadata.get("group_id", ""), label),
                confidence=confidence,
                evidence=[source.snippet] if source.snippet else [],
                source=source.source_type,
                provenance_source_ids=[source.source_id or source.label],
            )
        )
    return candidates


def build_route_decision(
    *,
    transcript_path: Path,
    readout_path: Path,
    transcript: dict[str, Any],
    readout: dict[str, Any],
    confidence_threshold: float = 0.8,
    extra_provenance_sources: Optional[list[ProvenanceSource]] = None,
    excluded_provenance_sources: Optional[list[ProvenanceSource]] = None,
    provenance_warnings: Optional[list[str]] = None,
    extra_candidates: Optional[list[RouteCandidate]] = None,
) -> RouteDecision:
    provenance_pack = provenance_from_transcript(transcript)
    if extra_provenance_sources:
        provenance_pack.sources.extend(extra_provenance_sources)
    if excluded_provenance_sources:
        provenance_pack.excluded_sources.extend(excluded_provenance_sources)
    if provenance_warnings:
        provenance_pack.warnings.extend(provenance_warnings)
    candidates = candidates_from_readout(readout, provenance_pack)
    if extra_candidates:
        candidates.extend(extra_candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
    selected = candidates[0] if candidates else None
    rejected = [
        RouteCandidate(
            **{
                **candidate.__dict__,
                "rejected_reason": "lower confidence than selected candidate" if selected else "no selected candidate",
            }
        )
        for candidate in candidates[1:]
    ]
    review_required = selected is None or selected.confidence < confidence_threshold
    status = "review_required" if review_required else "selected"
    fallback = "local_review_queue" if review_required else ""
    return RouteDecision(
        source_transcript_path=str(transcript_path),
        source_readout_path=str(readout_path),
        status=status,
        selected_candidate=selected,
        candidates=candidates,
        rejected_candidates=rejected,
        provenance_pack=provenance_pack,
        review_required=review_required,
        fallback=fallback,
        confidence_threshold=confidence_threshold,
        warnings=provenance_pack.warnings,
    )
