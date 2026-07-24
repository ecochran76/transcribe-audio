from __future__ import annotations

import json
import re
import shlex
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

import provenance_config
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
CONTACT_ALIAS_CONFIG_NAME = "contact-aliases.config.json"
EMAIL_RE = re.compile(r"(?P<email>[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+\-]{1,}")
NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SPEAKER_WORD_RE = re.compile(r"^(?:speaker|spk|unknown|participant|person)(?:[\s_-]*[A-Za-z0-9]+)?$", re.IGNORECASE)
NAME_STOP_TOKENS = {
    "and",
    "attendee",
    "calendar",
    "candidate",
    "contact",
    "contacts",
    "corp",
    "corporation",
    "dr",
    "email",
    "event",
    "inc",
    "llc",
    "meeting",
    "mr",
    "mrs",
    "ms",
    "participant",
    "person",
    "speaker",
    "the",
}


def normalize_email(value: Any) -> str:
    text = normalize_string(value).lower()
    match = EMAIL_RE.search(text)
    return match.group("email").lower() if match else ""


def email_alias_keys(value: Any) -> set[str]:
    email = normalize_email(value)
    if not email or "@" not in email:
        return set()
    local, domain = email.split("@", 1)
    keys = {email}
    if domain in {"gmail.com", "googlemail.com"}:
        base = local.split("+", 1)[0].replace(".", "")
        keys.add(f"{base}@gmail.com")
        keys.add(f"{base}@googlemail.com")
    return keys


def text_tokens(value: Any) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(normalize_string(value).lower()):
        if len(token) < 3:
            continue
        if token in {"com", "org", "net", "meeting", "speaker", "calendar", "event", "contact"}:
            continue
        tokens.add(token)
    return tokens


def candidate_name_text(value: Any) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    text = EMAIL_RE.sub(" ", text)
    if "|" in text:
        text = text.split("|", 1)[0]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 2:
        text = f"{parts[1]} {parts[0]}"
    return normalize_string(text)


def person_name_tokens(value: Any) -> list[str]:
    text = candidate_name_text(value).lower()
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_token in NAME_TOKEN_RE.findall(text):
        token = raw_token.strip("'-").lower()
        if len(token) < 2 or token in NAME_STOP_TOKENS:
            continue
        if token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def strong_person_name_keys(value: Any) -> list[str]:
    tokens = person_name_tokens(value)
    if len(tokens) < 2:
        return []
    keys = {f"name:{' '.join(tokens)}", f"name_tokens:{' '.join(sorted(tokens))}"}
    if len(tokens) > 2:
        keys.add(f"name_first_last:{tokens[0]} {tokens[-1]}")
    return sorted(keys)


def is_anonymous_speaker_label(value: Any) -> bool:
    text = normalize_string(value).strip()
    if not text:
        return True
    if len(text) <= 2 and re.fullmatch(r"[A-Za-z0-9]+", text):
        return True
    return bool(SPEAKER_WORD_RE.fullmatch(text))


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
    people = [*calendar_attendees, *readout_participants]
    for person in people:
        email = normalize_email(person.get("email"))
        if email:
            values.append(email)
    for person in people:
        email = normalize_email(person.get("email"))
        for value in [person.get("name", ""), person.get("label", "")]:
            text = normalize_string(value)
            if not text or text == email or is_anonymous_speaker_label(text):
                continue
            if email or len(text_tokens(text)) >= 2:
                values.append(text)
    values.extend(label for label in speaker_labels if not is_anonymous_speaker_label(label))
    return unique_strings([normalize_string(value) for value in values if normalize_string(value)])[:16]


def contact_source_config_path(state_root: Optional[Path]) -> Path:
    root = state_root.expanduser() if state_root else Path("~/.local/state/transcribe-audio").expanduser()
    return root / CONTACT_SOURCE_CONFIG_NAME


def load_contact_source_config(state_root: Optional[Path]) -> dict[str, Any]:
    try:
        shared_config = provenance_config.contact_source_config_from_provenance(state_root=state_root)
    except ValueError:
        shared_config = {}
    if shared_config:
        return shared_config
    path = contact_source_config_path(state_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_contact_settings(state_root: Optional[Path]) -> dict[str, Any]:
    try:
        raw_provenance = provenance_config.read_config(state_root=state_root)
    except ValueError:
        raw_provenance = {}
    contacts = raw_provenance.get("contacts") if isinstance(raw_provenance.get("contacts"), dict) else {}
    if contacts:
        return contacts
    path = contact_alias_config_path(state_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {"canonical_aliases": payload} if isinstance(payload, list) else {}


def contact_alias_config_path(state_root: Optional[Path]) -> Path:
    root = state_root.expanduser() if state_root else Path("~/.local/state/transcribe-audio").expanduser()
    return root / CONTACT_ALIAS_CONFIG_NAME


def normalize_contact_aliases(raw_aliases: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_aliases, list):
        return []
    aliases: list[dict[str, Any]] = []
    for item in raw_aliases:
        if not isinstance(item, dict):
            continue
        label = normalize_string(item.get("label") or item.get("name"))
        emails = unique_strings(
            normalize_email(value)
            for value in item.get("emails", [])
            if normalize_email(value)
        ) if isinstance(item.get("emails"), list) else []
        email_patterns = unique_strings(
            normalize_string(value).lower()
            for value in item.get("email_patterns", [])
            if normalize_string(value)
        ) if isinstance(item.get("email_patterns"), list) else []
        primary_email = normalize_email(item.get("primary_email")) or (emails[0] if emails else "")
        names = unique_strings(
            normalize_string(value).lower()
            for value in item.get("names", [])
            if normalize_string(value)
        ) if isinstance(item.get("names"), list) else []
        if label:
            names = unique_strings([label.lower(), *names])
        name_keys = sorted({key for name in names for key in strong_person_name_keys(name)})
        if not label or not (emails or email_patterns or names):
            continue
        alias_id = normalize_string(item.get("id")) or stable_id("contact-alias", label.lower(), ",".join(emails), ",".join(names))
        aliases.append(
            {
                "id": alias_id,
                "label": label,
                "primary_email": primary_email,
                "emails": emails,
                "email_keys": sorted({key for email in emails for key in email_alias_keys(email)}),
                "email_patterns": email_patterns,
                "names": names,
                "name_keys": name_keys,
            }
        )
    return aliases


def load_contact_aliases(state_root: Optional[Path]) -> list[dict[str, Any]]:
    settings = load_contact_settings(state_root)
    raw_aliases = settings.get("canonical_aliases") or settings.get("aliases") or []
    return normalize_contact_aliases(raw_aliases)


def load_operator_participant_hints(
    *,
    state_root: Optional[Path],
    conversation_key: str,
    source_document_id: str,
) -> list[dict[str, Any]]:
    settings = load_contact_settings(state_root)
    raw_hints = settings.get("participant_hints")
    if not isinstance(raw_hints, list):
        return []
    hints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_hints:
        if not isinstance(item, dict):
            continue
        match = item.get("match") if isinstance(item.get("match"), dict) else {}
        match_source_id = normalize_string(match.get("source_document_id"))
        match_conversation = normalize_string(match.get("conversation_key"))
        if match_source_id and match_source_id != source_document_id:
            continue
        if match_conversation and match_conversation != conversation_key:
            continue
        if not match_source_id and not match_conversation:
            continue
        participants = item.get("participants") if isinstance(item.get("participants"), list) else []
        for participant in participants:
            person = compact_person(participant, source="operator_participant_hint")
            key = person.get("email") or person.get("label")
            if not key or key in seen:
                continue
            hints.append(person)
            seen.add(key)
    return hints


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def source_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_string(item) for item in value if normalize_string(item)]
    return []


def command_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, list):
        result = tuple(str(item) for item in value if str(item))
        return result or default
    if isinstance(value, str) and value.strip():
        return tuple(shlex.split(value))
    return default


def collect_configured_contact_sources(
    *,
    query_terms: list[str],
    transcript: dict[str, Any],
    state_root: Optional[Path],
    source_filters: Optional[set[str]] = None,
) -> tuple[list[ProvenanceSource], list[dict[str, Any]], list[str]]:
    config = load_contact_source_config(state_root)
    if not config or not query_terms:
        return [], [], []
    sources: list[ProvenanceSource] = []
    profiles: list[dict[str, Any]] = []
    warnings: list[str] = []
    filters = {normalize_string(value).lower() for value in (source_filters or set()) if normalize_string(value)}

    gws_config = config.get("gws") if isinstance(config.get("gws"), dict) else {}
    for profile in gws_config.get("profiles") if isinstance(gws_config.get("profiles"), list) else []:
        if not isinstance(profile, dict) or not bool_config(profile.get("enabled"), True):
            continue
        surfaces = set(source_list(profile.get("surfaces")) or ["contacts"])
        label = normalize_string(profile.get("label")) or normalize_string(profile.get("profile")) or "gws-default"
        if filters and "gws" not in filters and label.lower() not in filters:
            continue
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
        if filters and "odollo" not in filters and label.lower() not in filters:
            continue
        default_odollo = OdolloProvenanceConfig()
        odollo = OdolloProvenanceConfig(
            enabled=True,
            profiles=(label,),
            command=command_tuple(profile.get("command"), default_odollo.command),
            repo_root=Path(profile["repo_root"]).expanduser() if profile.get("repo_root") else default_odollo.repo_root,
            config_path=Path(profile["config_path"]).expanduser() if profile.get("config_path") else default_odollo.config_path,
            timeout=float(profile.get("timeout") or 30.0),
            limit=int(profile.get("limit") or 5),
            include_contacts=True,
            include_log_notes=False,
        )
        profiles.append({"source": "odollo", "profile": label, "models": ["res.partner"], "read_only": True})
        try:
            event = transcript.get("event") if isinstance(transcript.get("event"), dict) else {}
            query_transcript = {
                "event": {
                    "summary": event.get("summary") or "",
                    "participants": [],
                    "attendees": [],
                }
            }
            sources.extend(collect_odollo_provenance(query_transcript, {"participants": query_terms}, config=odollo))
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


def operator_hint_candidate(person: dict[str, Any]) -> dict[str, Any]:
    label = normalize_string(person.get("label") or person.get("name") or person.get("email"))
    email = normalize_email(person.get("email"))
    return {
        "contact_id": stable_id("operator-participant-contact", label.lower(), email),
        "label": label or email,
        "email": email,
        "source": "operator_participant_hint",
        "source_type": "operator_participant_hint",
        "source_profile": "user_config",
        "confidence": 0.9,
        "evidence": [
            {
                "kind": "operator_participant_hint",
                "participant_evidence_id": person.get("id", ""),
                "participant_label": label,
                "participant_email": email,
                "source": person.get("source", "operator_participant_hint"),
            }
        ],
    }


def candidate_dedupe_key(candidate: dict[str, Any]) -> str:
    existing = normalize_string(candidate.get("dedupe_key"))
    if existing:
        return existing
    split_merge_key = normalize_string(candidate.get("split_merge_key"))
    if split_merge_key:
        return f"split:{split_merge_key}"
    canonical_key = normalize_string(candidate.get("canonical_key"))
    if canonical_key:
        return f"alias:{canonical_key}"
    email = normalize_email(candidate.get("email") or candidate.get("label"))
    if email:
        return f"email:{email}"
    label = normalize_string(candidate.get("label")).lower()
    if label and not is_anonymous_speaker_label(label):
        return f"label:{label}"
    contact_id = normalize_string(candidate.get("contact_id"))
    if contact_id:
        return f"contact:{contact_id}"
    return f"{candidate.get('source_type') or candidate.get('source')}:{candidate.get('source_id') or ''}"


def candidate_merge_keys(candidate: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    split_merge_key = normalize_string(candidate.get("split_merge_key"))
    if split_merge_key:
        return [f"split:{split_merge_key}"]
    canonical_key = normalize_string(candidate.get("canonical_key"))
    if canonical_key:
        keys.add(f"alias:{canonical_key}")
    existing = normalize_string(candidate.get("dedupe_key"))
    if existing.startswith("alias:"):
        keys.add(existing)
    email = normalize_email(candidate.get("email") or candidate.get("label"))
    for key in email_alias_keys(email):
        keys.add(f"email:{key}")
    if existing.startswith("email:"):
        keys.add(existing)
    label = normalize_string(candidate.get("label") or candidate.get("contact_label"))
    for key in strong_person_name_keys(label):
        keys.add(key)
    return sorted(keys)


def candidate_fallback_group_key(candidate: dict[str, Any], index: int) -> str:
    key = candidate_dedupe_key(candidate)
    if key and not key.startswith("label:"):
        return key
    contact_id = normalize_string(candidate.get("contact_id"))
    if contact_id:
        return f"contact:{contact_id}"
    source_type = normalize_string(candidate.get("source_type") or candidate.get("source"))
    source_profile = normalize_string(candidate.get("source_profile"))
    return f"candidate:{source_type}:{source_profile}:{index}"


def contact_alias_for_candidate(candidate: dict[str, Any], aliases: list[dict[str, Any]]) -> dict[str, Any] | None:
    email = normalize_email(candidate.get("email") or candidate.get("label"))
    keys = email_alias_keys(email)
    label = normalize_string(candidate.get("label")).lower()
    name_keys = set(strong_person_name_keys(label))
    for alias in aliases:
        alias_keys = set(alias.get("email_keys") or alias.get("emails") or [])
        if email and (email in alias.get("emails", []) or keys & alias_keys):
            return alias
        for pattern in alias.get("email_patterns") or []:
            if email and fnmatch(email, pattern):
                return alias
        if label and label in alias.get("names", []):
            return alias
        if name_keys and name_keys & set(alias.get("name_keys") or []):
            return alias
    return None


def apply_contact_alias(candidate: dict[str, Any], alias: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    original = {
        "label": normalize_string(candidate.get("label")),
        "email": normalize_email(candidate.get("email") or candidate.get("label")),
        "contact_id": str(candidate.get("contact_id") or ""),
        "source_type": normalize_string(candidate.get("source_type") or candidate.get("source")),
        "source_profile": normalize_string(candidate.get("source_profile")),
    }
    result["canonical_key"] = str(alias.get("id") or "")
    result["dedupe_key"] = f"alias:{alias.get('id')}"
    result["label"] = str(alias.get("label") or result.get("label") or "")
    result["email"] = str(alias.get("primary_email") or result.get("email") or "")
    result["canonical_contact"] = {
        "id": alias.get("id"),
        "label": alias.get("label"),
        "primary_email": alias.get("primary_email"),
        "matched_original": original,
    }
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    result["evidence"] = [
        *evidence,
        {
            "kind": "contact_alias",
            "alias_id": alias.get("id"),
            "canonical_label": alias.get("label"),
            "original_label": original["label"],
            "original_email": original["email"],
        },
    ]
    return result


def apply_contact_aliases(candidates: list[dict[str, Any]], aliases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not aliases:
        return candidates
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        alias = contact_alias_for_candidate(candidate, aliases)
        result.append(apply_contact_alias(candidate, alias) if alias else candidate)
    return result


def contact_candidate_source_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    canonical = candidate.get("canonical_contact") if isinstance(candidate.get("canonical_contact"), dict) else {}
    original = canonical.get("matched_original") if isinstance(canonical.get("matched_original"), dict) else {}
    return {
        "contact_id": str(candidate.get("contact_id") or ""),
        "label": normalize_string(candidate.get("label")),
        "email": normalize_email(candidate.get("email") or candidate.get("label")),
        "original_label": normalize_string(original.get("label")),
        "original_email": normalize_email(original.get("email")),
        "source": normalize_string(candidate.get("source")),
        "source_type": normalize_string(candidate.get("source_type") or candidate.get("source")),
        "source_profile": normalize_string(candidate.get("source_profile")),
        "confidence": candidate.get("confidence"),
    }


def merged_contact_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    primary = dict(candidates[0])
    merged_sources = [contact_candidate_source_summary(candidate) for candidate in candidates]
    merge_keys = sorted({key for candidate in candidates for key in candidate_merge_keys(candidate)})
    merged_contact_ids = [
        str(candidate.get("contact_id") or "")
        for candidate in candidates
        if str(candidate.get("contact_id") or "")
    ]
    evidence: list[Any] = []
    seen_evidence: set[str] = set()
    for candidate in candidates:
        candidate_evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []
        for item in candidate_evidence:
            key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            if key in seen_evidence:
                continue
            evidence.append(item)
            seen_evidence.add(key)
    primary["dedupe_key"] = candidate_dedupe_key(primary)
    primary["merge_keys"] = merge_keys
    primary["merged_contact_ids"] = merged_contact_ids
    primary["merged_sources"] = merged_sources
    primary["source_count"] = len(merged_sources)
    if evidence:
        primary["evidence"] = evidence
    if len(candidates) > 1 and any(key.startswith("name") for key in merge_keys):
        primary["evidence"] = [
            *primary.get("evidence", []),
            {
                "kind": "deterministic_contact_name_merge",
                "merge_keys": [key for key in merge_keys if key.startswith("name")][:4],
                "source_count": len(candidates),
            },
        ]
    if not primary.get("email"):
        primary["email"] = next((source["email"] for source in merged_sources if source.get("email")), "")
    if not primary.get("label"):
        primary["label"] = next((source["label"] for source in merged_sources if source.get("label")), "")
    return primary


def candidate_group_key(candidate: dict[str, Any]) -> str:
    return f"{candidate.get('source_type') or candidate.get('source')}:{candidate.get('source_profile') or ''}"


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, str, str]:
    return (
        float(candidate.get("confidence") or 0.0),
        normalize_string(candidate.get("source_type") or candidate.get("source")),
        normalize_string(candidate.get("label")),
    )


def ranked_contact_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int = 20,
    per_source_profile: int = 3,
    aliases: Optional[list[dict[str, Any]]] = None,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    candidates = apply_contact_aliases(candidates, aliases or [])
    eligible: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=candidate_sort_key, reverse=True):
        if float(candidate.get("confidence") or 0.0) < min_confidence:
            continue
        eligible.append(candidate)
    parents = list(range(len(eligible)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    key_owner: dict[str, int] = {}
    for index, candidate in enumerate(eligible):
        keys = candidate_merge_keys(candidate) or [candidate_fallback_group_key(candidate, index)]
        for key in keys:
            owner = key_owner.get(key)
            if owner is None:
                key_owner[key] = index
            else:
                union(owner, index)

    dedupe_groups: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(eligible):
        dedupe_groups.setdefault(find(index), []).append(candidate)
    deduped = [
        merged_contact_candidate(group)
        for group in dedupe_groups.values()
        if group
    ]
    deduped = sorted(deduped, key=candidate_sort_key, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in deduped:
        groups.setdefault(candidate_group_key(candidate), []).append(candidate)
    for group in sorted(groups.values(), key=lambda values: candidate_sort_key(values[0]), reverse=True):
        for candidate in group[: max(0, per_source_profile)]:
            key = candidate_dedupe_key(candidate)
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= limit:
                return sorted(selected, key=candidate_sort_key, reverse=True)
    for candidate in deduped:
        key = candidate_dedupe_key(candidate)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return sorted(selected, key=candidate_sort_key, reverse=True)


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
    operator_participant_hints = load_operator_participant_hints(
        state_root=state_root,
        conversation_key=conversation_key,
        source_document_id=source_document_id,
    )
    query_terms = identity_query_terms(
        calendar_attendees=calendar_attendees,
        readout_participants=[*normalized_readout_participants, *operator_participant_hints],
        speaker_labels=speaker_labels,
    )
    provenance_sources, source_profiles, warnings = collect_configured_contact_sources(
        query_terms=query_terms,
        transcript=transcript_payload,
        state_root=state_root,
    )
    evidence_pool = [*calendar_attendees, *normalized_readout_participants, *operator_participant_hints]
    provenance_candidates = [
        provenance_candidate(source, evidence_pool=evidence_pool)
        for source in provenance_sources
    ]
    operator_candidates = [operator_hint_candidate(person) for person in operator_participant_hints]
    local_candidates = [local_contact_candidate(row) for row in local_contacts]
    contact_aliases = load_contact_aliases(state_root)
    all_candidates = ranked_contact_candidates(
        [*operator_candidates, *provenance_candidates, *local_candidates],
        limit=20,
        aliases=contact_aliases,
        min_confidence=0.55,
    )

    speakers = []
    unresolved = []
    for label in speaker_labels:
        assignment = assignment_decision(assignments.get(label))
        status = assignment.get("status") if assignment else "pending"
        speaker_candidates = all_candidates[:8]
        review_required = status not in {"confirmed", "deferred", "llm_readout"}
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
        "operator_participant_hints": operator_participant_hints,
        "query_terms": query_terms,
        "source_profiles": source_profiles,
        "contact_aliases": [
            {
                "id": alias["id"],
                "label": alias["label"],
                "email_count": len(alias.get("emails") or []),
                "name_count": len(alias.get("names") or []),
            }
            for alias in contact_aliases
        ],
        "contact_candidates": all_candidates,
        "speakers": speakers,
        "operator_decisions": [assignment_decision(assignments[label]) for label in speaker_labels if label in assignments],
        "unresolved_ambiguities": unresolved,
        "warnings": unique_strings([normalize_string(item) for item in warnings if normalize_string(item)]),
        "review_status": "needs_review" if unresolved else "reviewed",
        "will_execute_external_action": False,
        "will_perform_external_write": False,
    }
