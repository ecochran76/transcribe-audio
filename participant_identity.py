from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from context_sources import (
    GwsProvenanceConfig,
    OdolloProvenanceConfig,
    collect_gws_contact_provenance,
    collect_odollo_provenance,
)
from routing_artifacts import ProvenanceSource, normalize_string, stable_id, unique_strings
from transcript_store import utcish_now

IDENTITY_BUNDLE_SCHEMA_VERSION = "transcribe-audio.participant-identity-bundle.v1"
CONTACT_SOURCE_CONFIG_NAME = "contact-provenance.config.json"
EMAIL_RE = re.compile(r"(?P<email>[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+\-]{1,}")


def normalize_email(value: Any) -> str:
    text = normalize_string(value).lower()
    match = EMAIL_RE.search(text)
    return match.group("email").lower() if match else ""


def text_tokens(value: Any) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(normalize_string(value).lower()):
        if len(token) < 3:
            continue
        if token in {"com", "org", "net", "meeting", "speaker", "calendar", "event", "contact"}:
            continue
        tokens.add(token)
    return tokens


def compact_person(value: Any, *, source: str, event_summary: str = "", calendar_summary: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        email = normalize_email(
            value.get("email")
            or value.get("emailAddress")
            or value.get("value")
            or value.get("address")
            or value.get("mail")
        )
        name = (
            normalize_string(value.get("displayName"))
            or normalize_string(value.get("name"))
            or normalize_string(value.get("summary"))
            or normalize_string(value.get("label"))
        )
        raw_label = normalize_string(value.get("formatted")) or normalize_string(value)
    else:
        raw_label = normalize_string(value)
        email = normalize_email(raw_label)
        name = normalize_string(EMAIL_RE.sub("", raw_label).replace("<", "").replace(">", "").strip())
    label = name or email or raw_label
    return {
        "id": stable_id("participant-evidence", source, event_summary, calendar_summary, label, email),
        "label": label,
        "name": name,
        "email": email,
        "source": source,
        "event_summary": event_summary,
        "calendar_summary": calendar_summary,
    }


def add_unique_person(result: list[dict[str, Any]], person: dict[str, Any], seen: set[str]) -> None:
    key = person["email"] or f"{person['source']}:{person['label']}:{person.get('event_summary', '')}"
    if not person["label"] or key in seen:
        return
    result.append(person)
    seen.add(key)


def extract_calendar_attendees(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    event_summary = normalize_string(event.get("summary"))
    for field in ("participants", "attendees"):
        values = event.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            add_unique_person(
                result,
                compact_person(value, source=f"primary_event.{field}", event_summary=event_summary),
                seen,
            )
    matching = event.get("matching_calendars") if isinstance(event.get("matching_calendars"), list) else []
    for item in matching:
        if not isinstance(item, dict):
            continue
        item_event = normalize_string(item.get("event_summary"))
        calendar = normalize_string(item.get("calendar_summary"))
        for field in ("participants", "attendees", "attendee_emails"):
            values = item.get(field)
            if not isinstance(values, list):
                continue
            for value in values:
                add_unique_person(
                    result,
                    compact_person(value, source=f"matching_calendar.{field}", event_summary=item_event, calendar_summary=calendar),
                    seen,
                )
    return result


def normalize_readout_participants(participants: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in participants:
        add_unique_person(result, compact_person(value, source="readout_participants"), seen)
    return result


def speaker_labels_from_transcript(document: dict[str, Any] | None) -> list[str]:
    if not document:
        return []
    payload = document.get("json_payload") if isinstance(document.get("json_payload"), dict) else document
    labels: list[str] = []
    utterances = payload.get("utterances") if isinstance(payload.get("utterances"), list) else []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        label = normalize_string(utterance.get("speaker"))
        if label and label not in labels:
            labels.append(label)
    text = normalize_string(payload.get("transcript_text") or document.get("text_content"))
    for line in text.splitlines():
        match = re.match(r"^(.{1,64}?)\s+\[[^\]]+\]:", line)
        if not match:
            continue
        label = match.group(1).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def identity_query_terms(
    *,
    calendar_attendees: list[dict[str, Any]],
    readout_participants: list[dict[str, Any]],
    speaker_labels: list[str],
) -> list[str]:
    values: list[str] = []
    for person in [*calendar_attendees, *readout_participants]:
        for value in [person.get("email", ""), person.get("name", ""), person.get("label", "")]:
            text = normalize_string(value)
            if text.lower().startswith("speaker "):
                continue
            values.append(text)
    values.extend(label for label in speaker_labels if not label.lower().startswith("speaker "))
    return unique_strings([normalize_string(value) for value in values if normalize_string(value)])[:16]


def contact_source_config_path(state_root: Optional[Path]) -> Path:
    root = state_root.expanduser() if state_root else Path("~/.local/state/transcribe-audio").expanduser()
    return root / CONTACT_SOURCE_CONFIG_NAME


def load_contact_source_config(state_root: Optional[Path]) -> dict[str, Any]:
    path = contact_source_config_path(state_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def source_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_string(item) for item in value if normalize_string(item)]
    return []


def collect_configured_contact_sources(
    *,
    query_terms: list[str],
    transcript: dict[str, Any],
    state_root: Optional[Path],
) -> tuple[list[ProvenanceSource], list[dict[str, Any]], list[str]]:
    config = load_contact_source_config(state_root)
    if not config or not query_terms:
        return [], [], []
    sources: list[ProvenanceSource] = []
    profiles: list[dict[str, Any]] = []
    warnings: list[str] = []

    gws_config = config.get("gws") if isinstance(config.get("gws"), dict) else {}
    for profile in gws_config.get("profiles") if isinstance(gws_config.get("profiles"), list) else []:
        if not isinstance(profile, dict) or not bool_config(profile.get("enabled"), True):
            continue
        surfaces = set(source_list(profile.get("surfaces")) or ["contacts"])
        label = normalize_string(profile.get("label")) or normalize_string(profile.get("profile")) or "gws-default"
        gws = GwsProvenanceConfig(
            enabled=True,
            profile_label=label,
            config_dir=Path(profile["config_dir"]).expanduser() if profile.get("config_dir") else None,
            include_calendar_details=False,
            include_drive_search=False,
            include_people_contacts="contacts" in surfaces,
            include_other_contacts="other_contacts" in surfaces,
            include_directory_people="directory" in surfaces,
            people_page_size=int(profile.get("limit") or profile.get("page_size") or 5),
            people_query_limit=int(profile.get("query_limit") or 8),
            timeout=float(profile.get("timeout") or 30.0),
        )
        profiles.append({"source": "gws", "profile": label, "surfaces": sorted(surfaces), "read_only": True})
        try:
            sources.extend(collect_gws_contact_provenance(query_terms, config=gws))
        except Exception as exc:
            warnings.append(f"gws contact provenance failed for profile {label}: {exc}")

    odollo_config = config.get("odollo") if isinstance(config.get("odollo"), dict) else {}
    odollo_profiles = odollo_config.get("profiles") if isinstance(odollo_config.get("profiles"), list) else []
    for profile in odollo_profiles:
        if isinstance(profile, str):
            profile = {"label": profile}
        if not isinstance(profile, dict) or not bool_config(profile.get("enabled"), True):
            continue
        label = normalize_string(profile.get("label")) or normalize_string(profile.get("profile"))
        if not label:
            continue
        default_odollo = OdolloProvenanceConfig()
        odollo = OdolloProvenanceConfig(
            enabled=True,
            profiles=(label,),
            repo_root=Path(profile["repo_root"]).expanduser() if profile.get("repo_root") else default_odollo.repo_root,
            config_path=Path(profile["config_path"]).expanduser() if profile.get("config_path") else default_odollo.config_path,
            timeout=float(profile.get("timeout") or 30.0),
            limit=int(profile.get("limit") or 5),
            include_contacts=True,
            include_log_notes=False,
        )
        profiles.append({"source": "odollo", "profile": label, "models": ["res.partner"], "read_only": True})
        try:
            sources.extend(collect_odollo_provenance(transcript, {"participants": query_terms}, config=odollo))
        except Exception as exc:
            warnings.append(f"Odollo contact provenance failed for profile {label}: {exc}")

    seen: set[tuple[str, str]] = set()
    deduped: list[ProvenanceSource] = []
    for source in sources:
        if source.source_type not in {"gws_contact", "gws_other_contact", "gws_directory_person", "odollo_contact"}:
            continue
        key = (source.source_type, source.source_id)
        if key in seen:
            continue
        deduped.append(source)
        seen.add(key)
    return deduped, profiles, warnings


def provenance_candidate(source: ProvenanceSource, *, evidence_pool: list[dict[str, Any]]) -> dict[str, Any]:
    source_dict = source.to_dict()
    metadata = source_dict.get("metadata") if isinstance(source_dict.get("metadata"), dict) else {}
    email = normalize_email(metadata.get("email") or source.snippet)
    label = normalize_string(source.label) or email or "Contact candidate"
    source_text = " ".join([label, source.snippet, email, normalize_string(metadata.get("company"))])
    evidence: list[dict[str, Any]] = []
    best = 0.4
    for item in evidence_pool:
        item_email = normalize_email(item.get("email"))
        item_label = normalize_string(item.get("label"))
        item_tokens = text_tokens(item_label)
        source_tokens = text_tokens(source_text)
        if email and item_email and email == item_email:
            score = 0.95
            reason = "email_exact"
        elif email and item_email and email.split("@", 1)[-1] == item_email.split("@", 1)[-1]:
            score = 0.7
            reason = "email_domain"
        elif len(item_tokens & source_tokens) >= 2:
            score = 0.78
            reason = "name_token_overlap"
        elif item_tokens & source_tokens:
            score = 0.6
            reason = "partial_token_overlap"
        else:
            continue
        best = max(best, score)
        evidence.append(
            {
                "kind": reason,
                "participant_evidence_id": item.get("id", ""),
                "participant_label": item_label,
                "participant_email": item_email,
                "source": item.get("source", ""),
            }
        )
    if not evidence:
        evidence.append({"kind": "configured_provenance_query", "query": metadata.get("query", "")})
    return {
        "contact_id": stable_id("provenance-contact", source.source_type, source.source_id),
        "label": label,
        "email": email,
        "source": source.source_type,
        "source_type": source.source_type,
        "source_profile": normalize_string(metadata.get("profile")),
        "confidence": round(best, 3),
        "evidence": evidence,
        "provenance_source": {
            "source_id": source.source_id,
            "source_type": source.source_type,
            "label": label,
            "uri": source.uri,
            "snippet": source.snippet,
            "metadata": {
                "profile": metadata.get("profile"),
                "company": metadata.get("company"),
                "record_id": metadata.get("record_id"),
                "resource_name": metadata.get("resource_name"),
            },
        },
    }


def local_contact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source = normalize_string(metadata.get("source")) or "local_contact"
    return {
        "contact_id": row.get("contact_id") or row.get("id") or "",
        "label": row.get("label") or row.get("contact_label") or "",
        "email": row.get("email") or "",
        "source": source,
        "source_type": source,
        "source_profile": "transcripts.sqlite3",
        "confidence": float(row.get("confidence") or 0.75),
        "evidence": [{"kind": source, "source": "contacts"}],
    }


def assignment_decision(assignment: dict[str, Any] | None) -> dict[str, Any] | None:
    if not assignment:
        return None
    return {
        "speaker_label": assignment.get("speaker_label", ""),
        "status": assignment.get("status", ""),
        "contact_id": assignment.get("contact_id", ""),
        "contact_label": assignment.get("contact_label", ""),
        "confidence": assignment.get("confidence"),
        "evidence": assignment.get("evidence") or [],
        "updated_at": assignment.get("updated_at", ""),
    }


def build_participant_identity_bundle(
    *,
    conversation_key: str,
    source_document_id: str,
    transcript: dict[str, Any],
    transcript_text: str,
    readout_participants: list[Any],
    local_contacts: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    transcript_payload = {**transcript, "transcript_text": transcript.get("transcript_text") or transcript_text}
    speaker_labels = speaker_labels_from_transcript(transcript_payload)
    calendar_attendees = extract_calendar_attendees(transcript_payload)
    normalized_readout_participants = normalize_readout_participants(readout_participants)
    query_terms = identity_query_terms(
        calendar_attendees=calendar_attendees,
        readout_participants=normalized_readout_participants,
        speaker_labels=speaker_labels,
    )
    provenance_sources, source_profiles, warnings = collect_configured_contact_sources(
        query_terms=query_terms,
        transcript=transcript_payload,
        state_root=state_root,
    )
    evidence_pool = [*calendar_attendees, *normalized_readout_participants]
    provenance_candidates = [
        provenance_candidate(source, evidence_pool=evidence_pool)
        for source in provenance_sources
    ]
    local_candidates = [local_contact_candidate(row) for row in local_contacts]
    candidate_seen: set[str] = set()
    all_candidates = []
    for candidate in sorted([*provenance_candidates, *local_candidates], key=lambda item: item.get("confidence", 0), reverse=True):
        key = f"{candidate.get('source')}:{candidate.get('email') or candidate.get('label')}"
        if not key or key in candidate_seen:
            continue
        all_candidates.append(candidate)
        candidate_seen.add(key)

    speakers = []
    unresolved = []
    for label in speaker_labels:
        assignment = assignment_decision(assignments.get(label))
        status = assignment.get("status") if assignment else "pending"
        speaker_candidates = all_candidates[:8]
        review_required = status not in {"confirmed", "deferred"}
        if review_required:
            unresolved.append(
                {
                    "speaker_label": label,
                    "reason": "no_reviewed_assignment" if speaker_candidates else "no_contact_candidates",
                    "candidate_count": len(speaker_candidates),
                }
            )
        speakers.append(
            {
                "speaker_label": label,
                "status": status,
                "assignment": assignment,
                "candidates": speaker_candidates,
                "candidate_count": len(speaker_candidates),
                "review_required": review_required,
            }
        )

    if unresolved:
        warnings.append(f"{len(unresolved)} speaker identity decision(s) need review before deposition.")
    return {
        "schema_version": IDENTITY_BUNDLE_SCHEMA_VERSION,
        "generated_at": utcish_now(),
        "conversation_key": conversation_key,
        "source_document_id": source_document_id,
        "speaker_labels": speaker_labels,
        "calendar_attendees": calendar_attendees,
        "readout_participants": normalized_readout_participants,
        "query_terms": query_terms,
        "source_profiles": source_profiles,
        "contact_candidates": all_candidates[:20],
        "speakers": speakers,
        "operator_decisions": [assignment_decision(assignments[label]) for label in speaker_labels if label in assignments],
        "unresolved_ambiguities": unresolved,
        "warnings": unique_strings([normalize_string(item) for item in warnings if normalize_string(item)]),
        "review_status": "needs_review" if unresolved else "reviewed",
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
