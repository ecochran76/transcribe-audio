from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Optional

from routing_artifacts import ProvenanceSource, normalize_string, stable_id, unique_strings
from transcribe_common import TranscriptionError

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
QUALITY_CALENDAR_SOURCE_TYPES = {"calendar_event", "gws_calendar_overlap", "gws_calendar_event_detail"}
QUALITY_STOP_TERMS = {
    "about",
    "and",
    "calendar",
    "candidate",
    "collaboration",
    "cochran",
    "com",
    "company",
    "discussion",
    "eric",
    "event",
    "meeting",
    "matter",
    "product",
    "products",
    "request",
    "soylei",
    "technical",
    "transcript",
    "with",
}


@dataclass
class GwsProvenanceConfig:
    enabled: bool = False
    config_dir: Optional[Path] = None
    drive_query: str = ""
    drive_page_size: int = 5
    timeout: float = 30.0
    include_calendar_details: bool = True
    include_drive_search: bool = True


@dataclass
class GraphitiProvenanceConfig:
    enabled: bool = False
    group_ids: tuple[str, ...] = ("transcribe_audio_main",)
    command: str = str(Path.home() / ".local/bin/graphiti-runtime")
    timeout: float = 30.0
    max_facts: int = 5
    max_nodes: int = 5
    max_episodes: int = 5
    preview_chars: int = 240


@dataclass
class OdolloProvenanceConfig:
    enabled: bool = False
    profiles: tuple[str, ...] = ("soylei-prod", "saber-prod")
    command: tuple[str, ...] = (
        str(Path.home() / "workspace.local/odollo/.venv/bin/python"),
        "-m",
        "odollo.cli",
    )
    repo_root: Path = Path.home() / "workspace.local/odollo"
    config_path: Path = Path.home() / ".odollo/odollo.yml"
    timeout: float = 30.0
    limit: int = 5
    include_contacts: bool = True
    include_log_notes: bool = True


def gws_env(config: GwsProvenanceConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.config_dir:
        env["GOOGLE_WORKSPACE_CLI_CONFIG_DIR"] = str(config.config_dir.expanduser())
    return env


def parse_gws_json(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            values.append(json.loads(line))
        return values


def run_gws_json(command: list[str], *, config: GwsProvenanceConfig) -> Any:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=config.timeout,
            check=False,
            env=gws_env(config),
        )
    except FileNotFoundError as exc:
        raise TranscriptionError("gws provenance requires the `gws` CLI on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"gws provenance command timed out after {config.timeout:g} seconds.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise TranscriptionError(f"gws provenance command failed ({result.returncode}): {detail}")
    try:
        return parse_gws_json(result.stdout)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("gws provenance command did not return valid JSON.") from exc


def event_items(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    items = []
    for item in event.get("matching_calendars") or []:
        if not isinstance(item, dict):
            continue
        calendar_id = normalize_string(item.get("calendar_id"))
        event_id = normalize_string(item.get("event_id"))
        if calendar_id and event_id:
            items.append({"calendar_id": calendar_id, "event_id": event_id})
    return unique_event_items(items)


def unique_event_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for item in items:
        key = (item["calendar_id"], item["event_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def collect_gws_calendar_sources(transcript: dict[str, Any], *, config: GwsProvenanceConfig) -> list[ProvenanceSource]:
    sources = []
    for item in event_items(transcript):
        payload = run_gws_json(
            [
                "gws",
                "calendar",
                "events",
                "get",
                "--params",
                json.dumps({"calendarId": item["calendar_id"], "eventId": item["event_id"]}, separators=(",", ":")),
            ],
            config=config,
        )
        if not isinstance(payload, dict):
            continue
        summary = normalize_string(payload.get("summary")) or item["event_id"]
        attendees = payload.get("attendees") if isinstance(payload.get("attendees"), list) else []
        sources.append(
            ProvenanceSource(
                source_type="gws_calendar_event_detail",
                source_id=normalize_string(payload.get("id")) or item["event_id"],
                label=summary,
                uri=normalize_string(payload.get("htmlLink")),
                snippet=summary,
                metadata={
                    "calendar_id": item["calendar_id"],
                    "event_id": item["event_id"],
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "organizer": payload.get("organizer"),
                    "attendee_emails": [
                        normalize_string(attendee.get("email"))
                        for attendee in attendees
                        if isinstance(attendee, dict) and normalize_string(attendee.get("email"))
                    ],
                    "attachments": payload.get("attachments") or [],
                },
            )
        )
    return sources


def search_terms_from_readout(transcript: dict[str, Any], readout: dict[str, Any]) -> list[str]:
    values = []
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    values.append(normalize_string(event.get("summary")))
    for item in readout.get("matter_candidates") or []:
        if isinstance(item, dict):
            values.append(normalize_string(item.get("label")))
    text = " ".join(values)
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
    skip = {"with", "and", "the", "discussion", "technical", "meeting", "candidate"}
    return unique_strings([term for term in terms if term.lower() not in skip])[:6]


def participant_terms(transcript: dict[str, Any]) -> list[str]:
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    values = event.get("participants") or event.get("attendees") or []
    terms = []
    iterable = values if isinstance(values, list) else []
    for value in iterable:
        text = normalize_string(value)
        if "@" in text:
            local, domain = text.split("@", 1)
            terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", local))
            terms.extend(part for part in domain.split(".") if len(part) >= 3)
        else:
            terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text))
    return unique_strings(terms)[:8]


def build_graphiti_query(transcript: dict[str, Any], readout: dict[str, Any]) -> str:
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    values = [normalize_string(event.get("summary")), normalize_string(readout.get("title"))]
    for item in readout.get("matter_candidates") or []:
        if isinstance(item, dict):
            values.append(normalize_string(item.get("label")))
    terms = search_terms_from_readout(transcript, readout) + participant_terms(transcript)
    query = " ".join(unique_strings([value for value in values if value] + terms))
    return query or "meeting transcript matter routing"


def build_odollo_terms(transcript: dict[str, Any], readout: dict[str, Any]) -> list[str]:
    """Build compact Odoo lookup terms without copying raw transcript text."""
    terms = search_terms_from_readout(transcript, readout) + participant_terms(transcript)
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    for attendee in event.get("attendees") or []:
        if isinstance(attendee, dict):
            terms.append(normalize_string(attendee.get("email")))
            terms.append(normalize_string(attendee.get("displayName") or attendee.get("display_name")))
    compact_terms: list[str] = []
    for term in terms:
        if "@" in term:
            compact_terms.append(term)
            local, domain = term.split("@", 1)
            compact_terms.extend([local, *domain.split(".")])
        else:
            compact_terms.append(term)
    skip = {"calendar", "meeting", "discussion", "technical", "candidate", "collaboration"}
    return unique_strings(
        [
            term
            for term in (normalize_string(item) for item in compact_terms)
            if len(term) >= 3 and term.lower() not in skip
        ]
    )[:8]


def provenance_quality_terms(transcript: dict[str, Any], readout: dict[str, Any]) -> list[str]:
    """Build compact matching terms for source-quality checks without raw transcript text."""
    values: list[str] = []
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    values.append(normalize_string(event.get("summary")))
    values.extend(participant_terms(transcript))
    values.append(normalize_string(readout.get("title")))
    for item in readout.get("topics") or []:
        values.append(normalize_string(item))
    for item in readout.get("matter_candidates") or []:
        if isinstance(item, dict):
            values.append(normalize_string(item.get("label") or item.get("name")))
    tokens: list[str] = []
    for value in values:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value):
            lowered = token.lower()
            if lowered not in QUALITY_STOP_TERMS:
                tokens.append(lowered)
    return unique_strings(tokens)[:32]


def source_quality_text(source: ProvenanceSource) -> str:
    metadata = source.metadata if isinstance(source.metadata, dict) else {}
    source_type = source.source_type
    metadata_keys_by_type = {
        "gws_drive_file": {"mime_type", "owners"},
        "gws_docs_file": {"mime_type", "owners"},
        "graphiti_fact": {"candidate_label", "episodes"},
        "graphiti_node": {"candidate_label", "labels"},
        "graphiti_episode": {"source_description"},
        "odollo_contact": {"email", "company"},
        "odollo_log_note": {"related_model", "related_record_id"},
    }
    metadata_keys = metadata_keys_by_type.get(source_type, set())
    metadata_for_quality = {key: metadata.get(key) for key in metadata_keys if metadata.get(key)}
    values = [
        source.source_type,
        source.label,
        source.source_id,
        source.uri,
        source.snippet,
        json.dumps(metadata_for_quality, sort_keys=True, ensure_ascii=False, default=str),
    ]
    return " ".join(normalize_string(value).lower() for value in values if normalize_string(value))


def source_quality_profile(source: ProvenanceSource) -> str:
    if source.source_type in {"gws_drive_file", "gws_docs_file"}:
        return "drive_file_identity"
    if source.source_type == "odollo_contact":
        return "odollo_contact_identity"
    if source.source_type == "odollo_log_note":
        return "odollo_log_note_subject"
    if source.source_type.startswith("graphiti_"):
        return "graphiti_label_or_preview"
    return "generic_label_or_snippet"


def source_type_min_score(source: ProvenanceSource, default_min_score: int) -> int:
    if source.source_type in QUALITY_CALENDAR_SOURCE_TYPES:
        return 0
    if source.source_type in {"gws_drive_file", "gws_docs_file"}:
        return max(default_min_score, 2)
    if source.source_type in {"odollo_contact", "odollo_log_note"}:
        return max(default_min_score, 2)
    if source.source_type.startswith("graphiti_"):
        return max(default_min_score, 2)
    return max(default_min_score, 2)


def annotate_source_quality(
    source: ProvenanceSource,
    *,
    status: str,
    score: float,
    matched_terms: list[str],
    reason: str,
) -> ProvenanceSource:
    source.metadata = {
        **source.metadata,
        "quality_status": status,
        "quality_score": score,
        "quality_matched_terms": matched_terms,
        "quality_reason": reason,
    }
    return source


def quality_for_source(
    source: ProvenanceSource,
    *,
    terms: list[str],
    min_score: int,
) -> tuple[str, float, list[str], str]:
    if source.source_type in QUALITY_CALENDAR_SOURCE_TYPES:
        return ("included", 1.0, [], "calendar source")
    text = source_quality_text(source)
    matched = [term for term in terms if term and term in text]
    score = float(len(matched))
    required_score = source_type_min_score(source, min_score)
    profile = source_quality_profile(source)
    if len(matched) >= required_score:
        return ("included", score, matched, f"{profile} matched compact meeting/readout terms")
    return (
        "excluded_low_quality",
        score,
        matched,
        f"{profile} matched {len(matched)} compact terms; required {required_score}",
    )


def filter_provenance_sources(
    sources: list[ProvenanceSource],
    *,
    transcript: dict[str, Any],
    readout: dict[str, Any],
    min_score: int = 2,
    enabled: bool = True,
) -> tuple[list[ProvenanceSource], list[ProvenanceSource], list[str]]:
    if not enabled:
        return (
            [
                annotate_source_quality(
                    source,
                    status="included_unfiltered",
                    score=0.0,
                    matched_terms=[],
                    reason="quality filter disabled",
                )
                for source in sources
            ],
            [],
            [],
        )
    terms = provenance_quality_terms(transcript, readout)
    included: list[ProvenanceSource] = []
    excluded: list[ProvenanceSource] = []
    for source in sources:
        status, score, matched_terms, reason = quality_for_source(source, terms=terms, min_score=max(min_score, 0))
        annotated = annotate_source_quality(
            source,
            status=status,
            score=score,
            matched_terms=matched_terms,
            reason=reason,
        )
        if status == "included":
            included.append(annotated)
        else:
            excluded.append(annotated)
    warnings = []
    if excluded:
        warnings.append(
            f"Excluded {len(excluded)} provenance source(s) below quality threshold {max(min_score, 0)}."
        )
    return included, excluded, warnings


def odollo_or_domain(clauses: list[list[Any]]) -> list[Any]:
    if not clauses:
        return []
    if len(clauses) == 1:
        return clauses[0]
    return ["|"] * (len(clauses) - 1) + clauses


def odollo_contact_domain(terms: list[str]) -> list[Any]:
    clauses: list[list[Any]] = []
    for term in terms:
        for field in ("name", "email"):
            clauses.append([field, "ilike", term])
    return odollo_or_domain(clauses)


def odollo_log_note_domain(terms: list[str]) -> list[Any]:
    term_clauses: list[list[Any]] = []
    for term in terms:
        term_clauses.append(["subject", "ilike", term])
        term_clauses.append(["body", "ilike", term])
    if not term_clauses:
        return [["message_type", "=", "comment"]]
    return ["&", ["message_type", "=", "comment"], *odollo_or_domain(term_clauses)]


def parse_command(value: str | tuple[str, ...]) -> list[str]:
    if isinstance(value, tuple):
        return list(value)
    return shlex.split(value)


def run_odollo_search(
    *,
    profile: str,
    model: str,
    domain: list[Any],
    fields: list[str],
    config: OdolloProvenanceConfig,
) -> list[dict[str, Any]]:
    command = [
        *parse_command(config.command),
        "--config",
        str(config.config_path.expanduser()),
        "--profile",
        profile,
        "--timeout",
        str(int(config.timeout)),
        "odoo",
        "records",
        "search",
        "--model",
        model,
        "--domain",
        json.dumps(domain, separators=(",", ":")),
        "--fields",
        ",".join(fields),
        "--limit",
        str(config.limit),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=config.timeout,
            check=False,
            cwd=config.repo_root.expanduser(),
        )
    except FileNotFoundError as exc:
        raise TranscriptionError(f"Odollo provenance requires `{command[0]}`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"Odollo provenance command timed out after {config.timeout:g} seconds.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise TranscriptionError(f"Odollo provenance command failed for profile {profile} ({result.returncode}): {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("Odollo provenance command did not return valid JSON.") from exc
    if not isinstance(payload, list):
        raise TranscriptionError("Odollo provenance command returned a non-list payload.")
    return [item for item in payload if isinstance(item, dict)]


def strip_html(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def m2o_label(value: Any) -> str:
    if isinstance(value, list) and len(value) >= 2:
        return normalize_string(value[1])
    return normalize_string(value)


def odollo_contact_sources(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    terms: list[str],
) -> list[ProvenanceSource]:
    sources: list[ProvenanceSource] = []
    for row in rows:
        record_id = normalize_string(row.get("id"))
        name = normalize_string(row.get("name")) or "Odoo contact"
        email = normalize_string(row.get("email"))
        company = m2o_label(row.get("parent_id"))
        label = " | ".join(item for item in [name, company] if item)
        sources.append(
            ProvenanceSource(
                source_type="odollo_contact",
                source_id=stable_id("odollo_contact", profile, record_id or label),
                label=label,
                uri=f"odoo://{profile}/res.partner/{record_id}" if record_id else "",
                snippet="; ".join(item for item in [name, email, company] if item),
                metadata={
                    "profile": profile,
                    "model": "res.partner",
                    "record_id": row.get("id"),
                    "email": email,
                    "company": company,
                    "matched_terms": terms,
                },
            )
        )
    return sources


def odollo_log_note_sources(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    terms: list[str],
) -> list[ProvenanceSource]:
    sources: list[ProvenanceSource] = []
    for row in rows:
        message_id = normalize_string(row.get("id"))
        subject = strip_html(row.get("subject")) or "Odoo log note"
        model = normalize_string(row.get("model"))
        res_id = normalize_string(row.get("res_id"))
        label = f"{subject} ({model}/{res_id})" if model and res_id else subject
        sources.append(
            ProvenanceSource(
                source_type="odollo_log_note",
                source_id=stable_id("odollo_log_note", profile, message_id or label),
                label=label,
                uri=f"odoo://{profile}/mail.message/{message_id}" if message_id else "",
                snippet="; ".join(item for item in [subject, normalize_string(row.get("date"))] if item),
                metadata={
                    "profile": profile,
                    "model": "mail.message",
                    "message_id": row.get("id"),
                    "related_model": model,
                    "related_record_id": row.get("res_id"),
                    "author": m2o_label(row.get("author_id")),
                    "date": row.get("date"),
                    "matched_terms": terms,
                    "body_matched_but_not_stored": bool(row.get("body")),
                },
            )
        )
    return sources


def collect_odollo_provenance(
    transcript: dict[str, Any],
    readout: dict[str, Any],
    *,
    config: OdolloProvenanceConfig,
) -> list[ProvenanceSource]:
    if not config.enabled:
        return []
    terms = build_odollo_terms(transcript, readout)
    if not terms:
        return []
    sources: list[ProvenanceSource] = []
    for profile in config.profiles:
        profile_name = normalize_string(profile)
        if not profile_name:
            continue
        if config.include_contacts:
            contact_rows = run_odollo_search(
                profile=profile_name,
                model="res.partner",
                domain=odollo_contact_domain(terms),
                fields=["id", "name", "email", "parent_id"],
                config=config,
            )
            sources.extend(odollo_contact_sources(contact_rows, profile=profile_name, terms=terms))
        if config.include_log_notes:
            note_rows = run_odollo_search(
                profile=profile_name,
                model="mail.message",
                domain=odollo_log_note_domain(terms),
                fields=["id", "subject", "body", "model", "res_id", "date", "author_id"],
                config=config,
            )
            sources.extend(odollo_log_note_sources(note_rows, profile=profile_name, terms=terms))
    return sources


def run_graphiti_discover(
    query: str,
    *,
    group_id: str,
    config: GraphitiProvenanceConfig,
) -> dict[str, Any]:
    command = [
        config.command,
        "discover",
        "--group-id",
        group_id,
        "--max-facts",
        str(config.max_facts),
        "--max-nodes",
        str(config.max_nodes),
        "--max-episodes",
        str(config.max_episodes),
        "--preview-chars",
        str(config.preview_chars),
        query,
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=config.timeout, check=False)
    except FileNotFoundError as exc:
        raise TranscriptionError(f"Graphiti provenance requires `{config.command}`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"Graphiti provenance command timed out after {config.timeout:g} seconds.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise TranscriptionError(f"Graphiti provenance command failed ({result.returncode}): {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("Graphiti provenance command did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError("Graphiti provenance command returned a non-object payload.")
    return payload


def graphiti_confidence(kind: str, index: int) -> float:
    base = {"fact": 0.62, "node": 0.56, "episode": 0.5}.get(kind, 0.45)
    return max(0.35, base - (0.03 * index))


def collect_graphiti_sources_from_payload(
    payload: dict[str, Any],
    *,
    group_id: str,
    query: str,
) -> list[ProvenanceSource]:
    sources: list[ProvenanceSource] = []
    for index, item in enumerate(payload.get("facts") or []):
        if not isinstance(item, dict):
            continue
        source_id = normalize_string(item.get("uuid")) or stable_id("graphiti_fact", group_id, str(index))
        preview = normalize_string(item.get("fact_preview"))
        label = normalize_string(item.get("name")) or "Graphiti fact"
        sources.append(
            ProvenanceSource(
                source_type="graphiti_fact",
                source_id=source_id,
                label=label,
                snippet=preview,
                metadata={
                    "group_id": group_id,
                    "query": query,
                    "episodes": item.get("episodes") or [],
                    "candidate_label": preview or label,
                    "candidate_confidence": graphiti_confidence("fact", index),
                },
            )
        )
    for index, item in enumerate(payload.get("nodes") or []):
        if not isinstance(item, dict):
            continue
        source_id = normalize_string(item.get("uuid")) or stable_id("graphiti_node", group_id, str(index))
        label = normalize_string(item.get("name")) or "Graphiti node"
        preview = normalize_string(item.get("summary_preview"))
        sources.append(
            ProvenanceSource(
                source_type="graphiti_node",
                source_id=source_id,
                label=label,
                snippet=preview,
                metadata={
                    "group_id": group_id,
                    "query": query,
                    "labels": item.get("labels") or [],
                    "candidate_label": label,
                    "candidate_confidence": graphiti_confidence("node", index),
                },
            )
        )
    for index, item in enumerate(payload.get("episodes") or []):
        if not isinstance(item, dict):
            continue
        source_id = normalize_string(item.get("uuid")) or stable_id("graphiti_episode", group_id, str(index))
        label = normalize_string(item.get("name")) or "Graphiti episode"
        preview = normalize_string(item.get("content_preview"))
        sources.append(
            ProvenanceSource(
                source_type="graphiti_episode",
                source_id=source_id,
                label=label,
                snippet=preview,
                metadata={
                    "group_id": group_id,
                    "query": query,
                    "source": item.get("source"),
                    "source_description": item.get("source_description"),
                    "created_at": item.get("created_at"),
                    "candidate_label": label,
                    "candidate_confidence": graphiti_confidence("episode", index),
                },
            )
        )
    return sources


def collect_graphiti_provenance(
    transcript: dict[str, Any],
    readout: dict[str, Any],
    *,
    config: GraphitiProvenanceConfig,
) -> list[ProvenanceSource]:
    if not config.enabled:
        return []
    query = build_graphiti_query(transcript, readout)
    sources: list[ProvenanceSource] = []
    for group_id in config.group_ids:
        group = normalize_string(group_id)
        if not group:
            continue
        payload = run_graphiti_discover(query, group_id=group, config=config)
        sources.extend(collect_graphiti_sources_from_payload(payload, group_id=group, query=query))
    return sources


def escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def build_drive_query(transcript: dict[str, Any], readout: dict[str, Any], explicit_query: str = "") -> str:
    if explicit_query:
        return explicit_query
    terms = search_terms_from_readout(transcript, readout)
    if not terms:
        return "trashed=false"
    clauses = []
    for term in terms[:3]:
        safe = escape_drive_query_value(term)
        clauses.append(f"name contains '{safe}'")
    return "trashed=false and " + " and ".join(clauses)


def collect_gws_drive_sources(
    transcript: dict[str, Any],
    readout: dict[str, Any],
    *,
    config: GwsProvenanceConfig,
) -> list[ProvenanceSource]:
    query = build_drive_query(transcript, readout, config.drive_query)
    payload = run_gws_json(
        [
            "gws",
            "drive",
            "files",
            "list",
            "--params",
            json.dumps(
                {
                    "q": query,
                    "pageSize": config.drive_page_size,
                    "includeItemsFromAllDrives": True,
                    "supportsAllDrives": True,
                    "fields": (
                        "files(id,name,mimeType,webViewLink,modifiedTime,createdTime,"
                        "owners(displayName,emailAddress),driveId,parents)"
                    ),
                    "orderBy": "modifiedTime desc",
                },
                separators=(",", ":"),
            ),
        ],
        config=config,
    )
    files = payload.get("files") if isinstance(payload, dict) else []
    sources = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        file_id = normalize_string(item.get("id"))
        name = normalize_string(item.get("name")) or file_id or "Drive file"
        source_type = "gws_docs_file" if item.get("mimeType") == GOOGLE_DOC_MIME_TYPE else "gws_drive_file"
        sources.append(
            ProvenanceSource(
                source_type=source_type,
                source_id=file_id or stable_id("gws_drive_file", name),
                label=name,
                uri=normalize_string(item.get("webViewLink")),
                snippet=name,
                metadata={
                    "mime_type": item.get("mimeType"),
                    "modified_time": item.get("modifiedTime"),
                    "created_time": item.get("createdTime"),
                    "owners": item.get("owners") or [],
                    "drive_id": item.get("driveId"),
                    "parents": item.get("parents") or [],
                    "query": query,
                },
            )
        )
    return sources


def collect_gws_provenance(
    transcript: dict[str, Any],
    readout: dict[str, Any],
    *,
    config: GwsProvenanceConfig,
) -> list[ProvenanceSource]:
    if not config.enabled:
        return []
    sources: list[ProvenanceSource] = []
    if config.include_calendar_details:
        sources.extend(collect_gws_calendar_sources(transcript, config=config))
    if config.include_drive_search:
        sources.extend(collect_gws_drive_sources(transcript, readout, config=config))
    return sources
