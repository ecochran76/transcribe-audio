from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import provenance_config
import transcript_store
from context_sources import (
    GwsProvenanceConfig,
    OdolloProvenanceConfig,
    gws_people_profile,
    m2o_label,
    run_gws_json,
    run_odollo_search,
)
from participant_identity import extract_calendar_attendees, normalize_email, normalize_string


SCHEMA_VERSION = "transcribe-audio.calendar-attendee-contact-ingest.v1"
CONTACT_METADATA_SCHEMA = "transcribe-audio.calendar-attendee-contact.v2"
APPLY_TOKEN = "INGEST_CALENDAR_ATTENDEE_CONTACTS"
UNDO_TOKEN = "UNDO_CALENDAR_ATTENDEE_CONTACTS"
DEFAULT_STATE_ROOT = Path("~/.local/state/transcribe-audio").expanduser()
ROLE_LOCAL_PARTS = {
    "admin",
    "billing",
    "board",
    "careers",
    "contact",
    "customerservice",
    "events",
    "finance",
    "hello",
    "help",
    "hr",
    "info",
    "marketing",
    "no-reply",
    "noreply",
    "office",
    "operations",
    "orders",
    "reception",
    "sales",
    "service",
    "support",
    "team",
}


class CalendarContactIngestError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def _strings(values: Iterable[Any]) -> list[str]:
    normalized = sorted(
        {normalize_string(value) for value in values if normalize_string(value)},
        key=lambda value: (value.casefold(), value),
    )
    result: dict[str, str] = {}
    for value in normalized:
        result.setdefault(value.casefold(), value)
    return list(result.values())


def _email_values(person: dict[str, Any]) -> list[str]:
    values = person.get("emailAddresses") if isinstance(person.get("emailAddresses"), list) else []
    return _strings(
        normalize_email(item.get("value"))
        for item in values
        if isinstance(item, dict) and normalize_email(item.get("value"))
    )


def _person_name(person: dict[str, Any]) -> str:
    values = person.get("names") if isinstance(person.get("names"), list) else []
    for item in values:
        if not isinstance(item, dict):
            continue
        for field in ("displayName", "unstructuredName", "givenName"):
            value = normalize_string(item.get(field))
            if value:
                return value
    return ""


def _person_organizations(person: dict[str, Any]) -> list[str]:
    values = person.get("organizations") if isinstance(person.get("organizations"), list) else []
    return _strings(
        item.get("name") or item.get("title")
        for item in values
        if isinstance(item, dict)
    )


def _person_roles(person: dict[str, Any]) -> list[dict[str, Any]]:
    values = person.get("organizations") if isinstance(person.get("organizations"), list) else []
    roles: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        title = normalize_string(item.get("title"))
        if not title:
            continue
        role = {
            "title": title,
            "organization": normalize_string(item.get("name")),
            "department": normalize_string(item.get("department")),
            "current": item.get("current") is True,
        }
        if role not in roles:
            roles.append(role)
    return sorted(
        roles,
        key=lambda value: (
            value["title"].casefold(),
            value["organization"].casefold(),
            value["department"].casefold(),
        ),
    )


def _person_phones(person: dict[str, Any]) -> list[str]:
    values = person.get("phoneNumbers") if isinstance(person.get("phoneNumbers"), list) else []
    return _strings(
        item.get("value")
        for item in values
        if isinstance(item, dict)
    )


def _original_filename(payload: dict[str, Any], source_path: str) -> str:
    for value in (
        payload.get("original_recording_filename"),
        payload.get("source_recording_filename"),
        payload.get("original_filename"),
    ):
        text = normalize_string(value)
        if text:
            return Path(text).name
    media_path = normalize_string(payload.get("source_media_path") or payload.get("working_media_path"))
    return Path(media_path or source_path).name


def _recorded_at(payload: dict[str, Any], generated_at: str) -> str:
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    for value in (
        payload.get("recorded_at"),
        payload.get("recording_date"),
        event.get("start"),
        event.get("start_time"),
        generated_at,
    ):
        text = normalize_string(value)
        if text:
            return text
    return ""


def collect_attendee_candidates(root: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        rows = con.execute(
            """
            SELECT id, title, source_path, generated_at, json_payload
            FROM documents
            WHERE kind = 'transcript'
            ORDER BY generated_at, id
            """
        ).fetchall()
    for row in rows:
        payload = _object(str(row["json_payload"]))
        event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        for attendee in extract_calendar_attendees(payload):
            email = normalize_email(attendee.get("email"))
            if not email:
                continue
            item = candidates.setdefault(
                email,
                {
                    "email": email,
                    "names": Counter(),
                    "appearances": [],
                    "sources": set(),
                },
            )
            name = normalize_string(attendee.get("name"))
            if name and name.casefold() != email.casefold():
                item["names"][name] += 1
            item["sources"].add(normalize_string(attendee.get("source")) or "calendar_attendee")
            item["appearances"].append(
                {
                    "document_id": str(row["id"]),
                    "recording_title": normalize_string(row["title"]) or "Untitled recording",
                    "recording_filename": _original_filename(payload, str(row["source_path"])),
                    "recorded_at": _recorded_at(payload, str(row["generated_at"])),
                    "event_summary": normalize_string(attendee.get("event_summary") or event.get("summary")),
                    "calendar_summary": normalize_string(attendee.get("calendar_summary")),
                    "evidence_source": normalize_string(attendee.get("source")) or "calendar_attendee",
                }
            )
    for item in candidates.values():
        item["names"] = dict(sorted(item["names"].items(), key=lambda pair: (-pair[1], pair[0].casefold())))
        item["sources"] = sorted(item["sources"])
        unique_appearances = {_json(value): value for value in item["appearances"]}
        item["appearances"] = sorted(
            unique_appearances.values(),
            key=lambda value: (value["recorded_at"], value["document_id"]),
            reverse=True,
        )
    return dict(sorted(candidates.items()))


@dataclass
class EnrichmentResult:
    matches: dict[str, list[dict[str, Any]]]
    read_calls: int
    read_records: int
    warnings: list[str]


def _add_match(result: dict[str, list[dict[str, Any]]], email: str, match: dict[str, Any]) -> None:
    key = normalize_email(email)
    if not key:
        return
    signature = _json(match)
    existing = result.setdefault(key, [])
    if all(_json(value) != signature for value in existing):
        existing.append(match)


def collect_gws_matches(
    emails: set[str],
    *,
    config: dict[str, Any],
    runner: Callable[..., Any] = run_gws_json,
    max_pages: int = 10,
) -> EnrichmentResult:
    matches: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    read_calls = 0
    read_records = 0
    profiles = config.get("profiles") if isinstance(config.get("profiles"), list) else []
    for raw_profile in profiles:
        if not isinstance(raw_profile, dict) or raw_profile.get("enabled", True) is False:
            continue
        surfaces = set(raw_profile.get("surfaces") if isinstance(raw_profile.get("surfaces"), list) else ["contacts"])
        gws = GwsProvenanceConfig(
            enabled=True,
            profile_label=normalize_string(raw_profile.get("label") or raw_profile.get("profile")) or "default",
            config_dir=Path(str(raw_profile["config_dir"])).expanduser() if raw_profile.get("config_dir") else None,
            timeout=float(raw_profile.get("timeout") or 30.0),
        )
        for surface in ("contacts", "other_contacts"):
            if surface not in surfaces:
                continue
            token = ""
            for page_number in range(max_pages):
                params: dict[str, Any]
                if surface == "contacts":
                    command = ["gws", "people", "people", "connections", "list"]
                    params = {
                        "resourceName": "people/me",
                        "personFields": "names,emailAddresses,organizations,phoneNumbers,metadata",
                        "pageSize": 1000,
                    }
                    collection_key = "connections"
                    source_type = "gws_contact"
                else:
                    command = ["gws", "people", "otherContacts", "list"]
                    params = {
                        "readMask": "names,emailAddresses,phoneNumbers,metadata",
                        "pageSize": 1000,
                    }
                    collection_key = "otherContacts"
                    source_type = "gws_other_contact"
                if token:
                    params["pageToken"] = token
                try:
                    payload = runner([*command, "--params", _json(params), "--format", "json"], config=gws)
                except Exception as exc:
                    warnings.append(f"GWS {gws_people_profile(gws)} {surface} failed: {type(exc).__name__}")
                    break
                read_calls += 1
                people = payload.get(collection_key) if isinstance(payload, dict) else []
                if not isinstance(people, list):
                    people = []
                read_records += len(people)
                for person in people:
                    if not isinstance(person, dict):
                        continue
                    person_emails = _email_values(person)
                    exact = emails.intersection(person_emails)
                    if not exact:
                        continue
                    match = {
                        "provider": "gws",
                        "profile": gws_people_profile(gws),
                        "record_type": source_type,
                        "source_record_id": normalize_string(person.get("resourceName")),
                        "label": _person_name(person),
                        "organizations": _person_organizations(person),
                        "roles": _person_roles(person),
                        "phones": _person_phones(person),
                        "match_basis": "exact_email",
                    }
                    for email in exact:
                        _add_match(matches, email, match)
                token = normalize_string(payload.get("nextPageToken")) if isinstance(payload, dict) else ""
                if not token:
                    break
            else:
                warnings.append(f"GWS {gws_people_profile(gws)} {surface} reached the {max_pages}-page safety cap")
    return EnrichmentResult(matches, read_calls, read_records, warnings)


def collect_odollo_matches(
    emails: set[str],
    *,
    config: dict[str, Any],
    runner: Callable[..., list[dict[str, Any]]] = run_odollo_search,
) -> EnrichmentResult:
    matches: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    read_calls = 0
    read_records = 0
    profiles = config.get("profiles") if isinstance(config.get("profiles"), list) else []
    for raw_profile in profiles:
        if isinstance(raw_profile, str):
            raw_profile = {"label": raw_profile}
        if not isinstance(raw_profile, dict) or raw_profile.get("enabled", True) is False:
            continue
        profile = normalize_string(raw_profile.get("label") or raw_profile.get("profile"))
        if not profile:
            continue
        default = OdolloProvenanceConfig()
        command = raw_profile.get("command")
        if isinstance(command, list):
            parsed_command = tuple(str(value) for value in command if str(value))
        else:
            parsed_command = default.command
        odollo = OdolloProvenanceConfig(
            enabled=True,
            profiles=(profile,),
            command=parsed_command or default.command,
            repo_root=Path(str(raw_profile["repo_root"])).expanduser() if raw_profile.get("repo_root") else default.repo_root,
            config_path=Path(str(raw_profile["config_path"])).expanduser() if raw_profile.get("config_path") else default.config_path,
            timeout=float(raw_profile.get("timeout") or 30.0),
            limit=max(500, len(emails) * 3),
            include_contacts=True,
            include_leads=False,
            include_log_notes=False,
        )
        try:
            rows = runner(
                profile=profile,
                model="res.partner",
                domain=[["email", "in", sorted(emails)]],
                fields=["id", "name", "email", "parent_id"],
                config=replace(odollo, limit=max(500, len(emails) * 3)),
            )
        except Exception as exc:
            warnings.append(f"Odollo {profile} contacts failed: {type(exc).__name__}")
            continue
        read_calls += 1
        read_records += len(rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            email = normalize_email(row.get("email"))
            if email not in emails:
                continue
            _add_match(
                matches,
                email,
                {
                    "provider": "odollo",
                    "profile": profile,
                    "record_type": "odollo_contact",
                    "source_record_id": normalize_string(row.get("id")),
                    "label": normalize_string(row.get("name")),
                    "organizations": _strings([m2o_label(row.get("parent_id"))]),
                    "roles": [],
                    "phones": [],
                    "match_basis": "exact_email",
                },
            )
    return EnrichmentResult(matches, read_calls, read_records, warnings)


def collect_configured_matches(
    emails: set[str], *, state_root: Path
) -> EnrichmentResult:
    try:
        config = provenance_config.contact_source_config_from_provenance(state_root=state_root)
    except ValueError as exc:
        return EnrichmentResult({}, 0, 0, [f"Configured contact sources invalid: {type(exc).__name__}"])
    combined = EnrichmentResult({}, 0, 0, [])
    for result in (
        collect_gws_matches(emails, config=config.get("gws") or {}),
        collect_odollo_matches(emails, config=config.get("odollo") or {}),
    ):
        for email, values in result.matches.items():
            for value in values:
                _add_match(combined.matches, email, value)
        combined.read_calls += result.read_calls
        combined.read_records += result.read_records
        combined.warnings.extend(result.warnings)
    return combined


def _humanize_email(email: str) -> str:
    local = email.split("@", 1)[0]
    value = re.sub(r"[._+-]+", " ", local)
    return " ".join(part.capitalize() for part in value.split()) or email


def _role_address(email: str) -> bool:
    local = email.split("@", 1)[0].casefold()
    compact = re.sub(r"[^a-z0-9-]", "", local)
    return compact in ROLE_LOCAL_PARTS or compact.startswith("no-reply") or compact.startswith("noreply")


def _chosen_label(candidate: dict[str, Any], matches: list[dict[str, Any]]) -> str:
    provider_names = _strings(match.get("label") for match in matches)
    if len({name.casefold() for name in provider_names}) == 1 and provider_names:
        return provider_names[0]
    names = candidate.get("names") if isinstance(candidate.get("names"), dict) else {}
    if names:
        ranked = sorted(names.items(), key=lambda pair: (-int(pair[1]), pair[0].casefold()))
        if len(ranked) == 1 or int(ranked[0][1]) > int(ranked[1][1]):
            return ranked[0][0]
    return _humanize_email(candidate["email"])


def _contact_metadata(candidate: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    email = candidate["email"]
    names = candidate.get("names") if isinstance(candidate.get("names"), dict) else {}
    role_address = _role_address(email)
    provider_names = _strings(match.get("label") for match in matches)
    conflicting_names = len({name.casefold() for name in [*names, *provider_names]}) > 1
    appearances = candidate.get("appearances") if isinstance(candidate.get("appearances"), list) else []
    recording_ids = _strings(item.get("document_id") for item in appearances if isinstance(item, dict))
    organizations = _strings(
        organization
        for match in matches
        for organization in (match.get("organizations") if isinstance(match.get("organizations"), list) else [])
    )
    phones = _strings(
        phone
        for match in matches
        for phone in (match.get("phones") if isinstance(match.get("phones"), list) else [])
    )
    roles = sorted(
        {
            _json(role): role
            for match in matches
            for role in (
                match.get("roles") if isinstance(match.get("roles"), list) else []
            )
            if isinstance(role, dict) and normalize_string(role.get("title"))
        }.values(),
        key=lambda value: (
            normalize_string(value.get("title")).casefold(),
            normalize_string(value.get("organization")).casefold(),
            normalize_string(value.get("department")).casefold(),
        ),
    )
    corpus_payload = {
        "email": email,
        "names": names,
        "appearances": appearances,
        "sources": candidate.get("sources") or [],
    }
    return {
        "schema_version": CONTACT_METADATA_SCHEMA,
        "source": "calendar_attendee",
        "resolution_status": "review_required" if role_address or conflicting_names else "provisional",
        "contact_class": "shared_or_role_address" if role_address else "person_candidate",
        "identity_boundary": "exact_email_source_join_not_person_or_speaker_proof",
        "calendar_attendee": {
            "aliases": _strings([*names, *provider_names]),
            "name_observation_counts": names,
            "domain": email.split("@", 1)[1] if "@" in email else "",
            "occurrence_count": len(appearances),
            "recording_count": len(recording_ids),
            "appearances": appearances,
            "evidence_sources": candidate.get("sources") or [],
            "corpus_fingerprint": _hash(corpus_payload),
        },
        "enrichment": {
            "match_basis": "exact_email",
            "exact_match_count": len(matches),
            "source_records": sorted(matches, key=lambda value: (value["provider"], value["profile"], value["source_record_id"])),
            "organizations": organizations,
            "roles": roles,
            "phones": phones,
        },
    }


def build_ingest_plan(
    root: Optional[Path] = None,
    *,
    state_root: Path = DEFAULT_STATE_ROOT,
    enrich: bool = True,
    enrichment: Optional[EnrichmentResult] = None,
) -> dict[str, Any]:
    candidates = collect_attendee_candidates(root)
    found = enrichment or (collect_configured_matches(set(candidates), state_root=state_root) if enrich else EnrichmentResult({}, 0, 0, []))
    operations: list[dict[str, Any]] = []
    counts = Counter()
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        existing_rows = con.execute("SELECT * FROM contacts ORDER BY id").fetchall()
    by_email: dict[str, list[dict[str, Any]]] = {}
    for row in existing_rows:
        normalized = normalize_email(row["email"])
        if normalized:
            by_email.setdefault(normalized, []).append(dict(row))
    for email, candidate in candidates.items():
        rows = by_email.get(email, [])
        if len(rows) > 1:
            counts["conflicted"] += 1
            operations.append({"action": "conflict", "email": email, "reason": "multiple_existing_contacts_with_exact_email", "existing_ids": [row["id"] for row in rows]})
            continue
        matches = found.matches.get(email, [])
        generated = _contact_metadata(candidate, matches)
        label = _chosen_label(candidate, matches)
        if rows:
            before = rows[0]
            existing_metadata = _object(str(before["metadata_json"]))
            after_metadata = {**existing_metadata, **generated}
            after = {
                **before,
                "metadata_json": _json(after_metadata),
            }
            if str(before["metadata_json"]) == after["metadata_json"]:
                action = "unchanged"
            else:
                action = "update"
            counts["enriched" if action == "update" else "unchanged"] += 1
            operations.append({"action": action, "email": email, "before": before, "after": after})
        else:
            now = _now()
            after = {
                "id": transcript_store.stable_id("calendar-attendee-contact", email),
                "label": label,
                "email": email,
                "external_ref": "",
                "metadata_json": _json(generated),
                "created_at": now,
                "updated_at": now,
            }
            counts["inserted"] += 1
            operations.append({"action": "insert", "email": email, "before": None, "after": after})
    counts["excluded"] = 0
    accounted = sum(counts[key] for key in ("inserted", "enriched", "unchanged", "conflicted", "excluded"))
    if accounted != len(candidates):
        raise CalendarContactIngestError("Attendee contact accounting does not equal the unique-email corpus total.")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "preview",
        "corpus_fingerprint": _hash({email: candidate for email, candidate in candidates.items()}),
        "unique_attendee_email_count": len(candidates),
        "attendee_appearance_count": sum(len(candidate["appearances"]) for candidate in candidates.values()),
        "counts": dict(counts),
        "provider_read_call_count": found.read_calls,
        "provider_read_record_count": found.read_records,
        "provider_exact_match_email_count": len(found.matches),
        "provider_write_count": 0,
        "person_merge_count": 0,
        "speaker_assignment_apply_count": 0,
        "warnings": found.warnings,
        "operations": operations,
    }


def _receipt_path(state_root: Path) -> Path:
    directory = state_root.expanduser() / "contact-ingest"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"calendar-attendees-{stamp}-{os.getpid()}.json"


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def apply_ingest_plan(
    plan: dict[str, Any],
    *,
    root: Optional[Path] = None,
    state_root: Path = DEFAULT_STATE_ROOT,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != APPLY_TOKEN:
        raise CalendarContactIngestError(f"Apply requires approval token {APPLY_TOKEN}.")
    applied_at = _now()
    for operation in plan.get("operations") or []:
        if operation.get("action") == "update" and isinstance(operation.get("after"), dict):
            operation["after"]["updated_at"] = applied_at
    receipt_path = _receipt_path(state_root)
    prepared_receipt = {
        **plan,
        "mode": "apply_prepared",
        "applied_at": applied_at,
        "receipt_path": str(receipt_path),
    }
    _write_receipt(receipt_path, prepared_receipt)
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        con.execute("BEGIN IMMEDIATE")
        try:
            for operation in plan.get("operations") or []:
                action = operation.get("action")
                after = operation.get("after")
                if action == "insert" and isinstance(after, dict):
                    existing = con.execute(
                        "SELECT * FROM contacts WHERE id = ? OR LOWER(email) = ?",
                        (after["id"], after["email"]),
                    ).fetchall()
                    if existing:
                        raise CalendarContactIngestError(
                            f"Contact plan became stale before insert: {after['id']}."
                        )
                    con.execute(
                        """
                        INSERT INTO contacts (id, label, email, external_ref, metadata_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (after["id"], after["label"], after["email"], after["external_ref"], after["metadata_json"], after["created_at"], after["updated_at"]),
                    )
                elif action == "update" and isinstance(after, dict):
                    before = operation.get("before")
                    current = con.execute(
                        "SELECT * FROM contacts WHERE id = ?", (after["id"],)
                    ).fetchone()
                    if current is None or not isinstance(before, dict) or dict(current) != before:
                        raise CalendarContactIngestError(
                            f"Contact plan became stale before update: {after['id']}."
                        )
                    con.execute(
                        "UPDATE contacts SET metadata_json = ?, updated_at = ? WHERE id = ?",
                        (after["metadata_json"], applied_at, after["id"]),
                    )
            con.commit()
        except Exception as exc:
            con.rollback()
            _write_receipt(
                receipt_path,
                {
                    **prepared_receipt,
                    "mode": "apply_failed",
                    "failed_at": _now(),
                    "failure_type": type(exc).__name__,
                },
            )
            raise
    receipt = {**plan, "mode": "applied", "applied_at": applied_at}
    receipt["receipt_path"] = str(receipt_path)
    _write_receipt(receipt_path, receipt)
    return receipt


def undo_receipt(
    receipt_path: Path,
    *,
    root: Optional[Path] = None,
    state_root: Path = DEFAULT_STATE_ROOT,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != UNDO_TOKEN:
        raise CalendarContactIngestError(f"Undo requires approval token {UNDO_TOKEN}.")
    receipt = json.loads(receipt_path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("mode") != "applied":
        raise CalendarContactIngestError("Receipt is not an applied calendar-attendee contact ingest receipt.")
    restored = deleted = 0
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        con.execute("BEGIN IMMEDIATE")
        try:
            for operation in reversed(receipt.get("operations") or []):
                action = operation.get("action")
                after = operation.get("after") if isinstance(operation.get("after"), dict) else {}
                before = operation.get("before") if isinstance(operation.get("before"), dict) else None
                if action not in {"insert", "update"}:
                    continue
                current = con.execute("SELECT * FROM contacts WHERE id = ?", (after.get("id"),)).fetchone()
                if current is None:
                    raise CalendarContactIngestError(f"Contact {after.get('id')} changed after receipt; undo stopped.")
                if dict(current) != after:
                    raise CalendarContactIngestError(f"Contact {after.get('id')} changed after receipt; undo stopped.")
                if action == "insert":
                    con.execute("DELETE FROM contacts WHERE id = ?", (after["id"],))
                    deleted += 1
                elif before:
                    con.execute(
                        """
                        UPDATE contacts SET label = ?, email = ?, external_ref = ?, metadata_json = ?,
                            created_at = ?, updated_at = ? WHERE id = ?
                        """,
                        (before["label"], before["email"], before["external_ref"], before["metadata_json"], before["created_at"], before["updated_at"], before["id"]),
                    )
                    restored += 1
            con.commit()
        except Exception:
            con.rollback()
            raise
    undo = {
        "schema_version": SCHEMA_VERSION,
        "mode": "undone",
        "source_receipt_path": str(receipt_path.expanduser()),
        "undone_at": _now(),
        "deleted_insert_count": deleted,
        "restored_update_count": restored,
        "provider_write_count": 0,
        "person_merge_count": 0,
        "speaker_assignment_apply_count": 0,
    }
    path = _receipt_path(state_root)
    undo["receipt_path"] = str(path)
    _write_receipt(path, undo)
    return undo


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in (
        "schema_version", "mode", "unique_attendee_email_count", "attendee_appearance_count",
        "counts", "provider_read_call_count", "provider_read_record_count",
        "provider_exact_match_email_count", "provider_write_count", "person_merge_count",
        "speaker_assignment_apply_count", "receipt_path", "warnings",
        "deleted_insert_count", "restored_update_count",
    ) if key in payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest exact-email calendar attendee contact candidates.")
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--no-enrich", action="store_true", help="Skip configured read-only provider enrichment.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approval-token", default="")
    parser.add_argument("--undo-receipt", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.undo_receipt:
        payload = undo_receipt(
            args.undo_receipt,
            root=args.store_dir,
            state_root=args.state_root,
            approval_token=args.approval_token,
        )
    else:
        payload = build_ingest_plan(
            args.store_dir,
            state_root=args.state_root,
            enrich=not args.no_enrich,
        )
        if args.apply:
            payload = apply_ingest_plan(
                payload,
                root=args.store_dir,
                state_root=args.state_root,
                approval_token=args.approval_token,
            )
    print(json.dumps(summary(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
