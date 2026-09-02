"""Canonical people, organization, and cross-channel activity projections.

The module consumes the existing source-oriented People payload and produces a
privacy-bounded directory read model.  It never turns name overlap into an
accepted person link; unresolved groups retain independently addressable
members and source records.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "transcribe-audio.people-organization-activity-index.v4"
CHANNELS = ("transcript", "calendar", "email")
MAIL_HYPOTHESIS_KINDS = {"correspondence", "sent_mail", "thread_coparticipation"}
REVIEWED_ROLE_STATUSES = {"accepted", "reviewed"}
DIRECTORY_REVIEW_HYPOTHESIS_KINDS = {"affiliation", "contextual_role"}
HONORIFIC_PREFIXES = {
    "doctor",
    "dr",
    "miss",
    "mr",
    "mrs",
    "ms",
    "mx",
    "prof",
    "professor",
    "rev",
}


def _text(value: object) -> str:
    return str(value or "").strip()


def _array(value: object) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _name_candidate(value: object) -> tuple[str, str]:
    """Return a human display form and its completeness without changing evidence."""

    source = _text(value)
    if not source:
        return "", "incomplete"
    if "@" in source and " " not in source:
        return source, "identifier_only"

    display_name = " ".join(source.split())
    if display_name.count(",") == 1:
        family_name, given_names = (
            " ".join(part.split()) for part in display_name.split(",", maxsplit=1)
        )
        if family_name and given_names:
            display_name = f"{given_names} {family_name}"

    tokens = display_name.split()
    without_honorific = tokens
    if tokens and tokens[0].rstrip(".").casefold() in HONORIFIC_PREFIXES:
        without_honorific = tokens[1:]
        if len(without_honorific) >= 2:
            display_name = " ".join(without_honorific)

    completeness = "complete" if len(without_honorific) >= 2 else "incomplete"
    return display_name, completeness


def _person_name_presentation(
    primary_name: object,
    aliases: Sequence[object],
) -> dict[str, str]:
    display_name, completeness = _name_candidate(primary_name)
    if completeness != "complete":
        complete_aliases = sorted(
            (
                candidate
                for candidate in (_name_candidate(alias) for alias in aliases)
                if candidate[1] == "complete"
            ),
            key=lambda candidate: candidate[0].casefold(),
        )
        if complete_aliases:
            display_name, completeness = complete_aliases[0]
    return {
        "display_name": display_name,
        "sort_name": display_name.casefold(),
        "name_completeness": completeness,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(chr(31).join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _record_member(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": _text(record.get("person_id")),
        "record_kind": _text(record.get("identity_kind")) or "source_record",
        "status": _text(record.get("status")) or "unresolved",
        "primary_name": _text(record.get("primary_name")),
        "aliases": sorted({_text(value) for value in _array(record.get("aliases")) if _text(value)}),
        "source_records": [
            dict(value)
            for value in _array(record.get("source_records"))
            if isinstance(value, Mapping)
        ],
    }


def _activity(
    *,
    channel: str,
    member_id: str,
    occurrence: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    occurred_at = _text(
        occurrence.get("recorded_at")
        or occurrence.get("reviewed_at")
        or occurrence.get("occurred_at")
    )
    source_id = _text(
        occurrence.get("document_id")
        or occurrence.get("hypothesis_id")
        or occurrence.get("source_record_id")
    )
    activity_id = _stable_id(
        "activity",
        channel,
        source_id,
        occurred_at,
        _text(occurrence.get("speaker_ref")),
    )
    independence_group_id = _stable_id(
        "independence",
        channel,
        source_id,
        _text(occurrence.get("speaker_ref")),
    )
    return {
        "observation_id": activity_id,
        "channel": channel,
        "occurred_at": occurred_at,
        "evidence_status": "confirmed" if channel == "transcript" else "proposed",
        "participation_status": "confirmed" if channel == "transcript" else "candidate",
        "independence_group_ids": [independence_group_id],
        "observation_count": 1,
        "first_at": occurred_at,
        "last_at": occurred_at,
        "source_member_id": member_id,
        "source_record_id": source_id,
        "title": _text(
            occurrence.get("recording_title")
            or occurrence.get("event_summary")
            or occurrence.get("calendar_summary")
        ),
        "source_locator": {
            key: _text(occurrence.get(key))
            for key in ("document_id", "recording_filename", "speaker_ref")
            if _text(occurrence.get(key))
        },
    }


def _mail_activity(member_id: str, hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis_id = _text(hypothesis.get("hypothesis_id"))
    groups = sorted(
        {
            _text(value)
            for value in _array(hypothesis.get("evidence_independence_group_ids"))
            if _text(value)
        }
    )
    if not groups:
        groups = [_stable_id("mail-evidence", hypothesis_id)]
    first_at = _text(hypothesis.get("first_observed_at"))
    last_at = _text(hypothesis.get("last_observed_at")) or first_at
    review_state = _text(hypothesis.get("review_state") or hypothesis.get("status"))
    return {
        "observation_id": hypothesis_id or _stable_id("mail-activity", member_id, last_at),
        "channel": "email",
        "occurred_at": last_at,
        "first_at": first_at,
        "last_at": last_at,
        "evidence_status": "accepted" if review_state == "accepted" else "proposed",
        "participation_status": "observed" if review_state == "accepted" else "candidate",
        "independence_group_ids": groups,
        "observation_count": len(groups),
        "source_member_id": member_id,
        "source_record_id": hypothesis_id,
        "title": _text(hypothesis.get("basis")) or _text(hypothesis.get("hypothesis_kind")),
        "direction": _text(hypothesis.get("mail_direction")),
        "source_locator": {
            key: _text(hypothesis.get(key))
            for key in ("hypothesis_id", "source_content_sha256")
            if _text(hypothesis.get(key))
        },
    }


def _record_activities(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    member_id = _text(record.get("person_id"))
    activities: list[dict[str, Any]] = []
    for ordinal, occurrence in enumerate(_array(record.get("calendar_occurrences"))):
        if isinstance(occurrence, Mapping):
            activities.append(
                _activity(
                    channel="calendar",
                    member_id=member_id,
                    occurrence=occurrence,
                    ordinal=ordinal,
                )
            )
    for ordinal, occurrence in enumerate(_array(record.get("review_occurrences"))):
        if isinstance(occurrence, Mapping):
            activities.append(
                _activity(
                    channel="transcript",
                    member_id=member_id,
                    occurrence=occurrence,
                    ordinal=ordinal,
                )
            )
    for hypothesis in _array(record.get("relationship_hypotheses")):
        if (
            isinstance(hypothesis, Mapping)
            and _text(hypothesis.get("hypothesis_kind")) in MAIL_HYPOTHESIS_KINDS
        ):
            activities.append(_mail_activity(member_id, hypothesis))
    return activities


def _summary(activities: Sequence[Mapping[str, Any]], *, resolved: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for channel in CHANNELS:
        rows = [row for row in activities if row.get("channel") == channel]
        dates = sorted(
            {
                _text(value)
                for row in rows
                for value in (row.get("first_at"), row.get("last_at"), row.get("occurred_at"))
                if _text(value)
            }
        )
        confirmed_groups: set[str] = set()
        proposed_groups: set[str] = set()
        for row in rows:
            groups = {
                _text(value)
                for value in _array(row.get("independence_group_ids"))
                if _text(value)
            } or {_text(row.get("observation_id"))}
            confirmed = resolved and _text(row.get("evidence_status")) in {
                "accepted",
                "confirmed",
                "observed",
            }
            (confirmed_groups if confirmed else proposed_groups).update(groups)
        result[channel] = {
            "confirmed_count": len(confirmed_groups),
            "proposed_count": len(proposed_groups - confirmed_groups),
            "coverage_state": "partial" if rows else "not_queried",
            "first_at": dates[0] if dates else "",
            "last_at": dates[-1] if dates else "",
        }
    return result


def _directory_entity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    members = sorted((_record_member(record) for record in records), key=lambda item: item["record_id"])
    canonical = [record for record in records if record.get("identity_kind") == "canonical_person"]
    resolved = len(canonical) == 1
    accepted_person_id = _text(canonical[0].get("person_id")) if resolved else ""
    primary_name = _text(records[0].get("primary_name"))
    member_ids = [member["record_id"] for member in members]
    entity_id = accepted_person_id or _stable_id("unresolved", *member_ids)
    source_records = {
        _text(source.get("source_record_id")): source
        for member in members
        for source in member["source_records"]
        if _text(source.get("source_record_id"))
    }
    activities_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for activity in _record_activities(record):
            key = (_text(activity.get("channel")), _text(activity.get("observation_id")))
            existing = activities_by_id.get(key)
            if existing is None:
                activities_by_id[key] = activity
                continue
            existing_groups = {
                _text(value)
                for value in _array(existing.get("independence_group_ids"))
                if _text(value)
            }
            existing_groups.update(
                _text(value)
                for value in _array(activity.get("independence_group_ids"))
                if _text(value)
            )
            existing["independence_group_ids"] = sorted(existing_groups)
            existing["observation_count"] = len(existing_groups)
            if _text(activity.get("evidence_status")) in {"accepted", "confirmed"}:
                existing["evidence_status"] = _text(activity.get("evidence_status"))
    activities = list(activities_by_id.values())
    activities.sort(
        key=lambda item: (
            _text(item.get("occurred_at")),
            _text(item.get("channel")),
            _text(item.get("observation_id")),
        ),
        reverse=True,
    )
    summary = _summary(activities, resolved=resolved)
    last_interaction_at = max(
        (_text(value.get("last_at")) for value in summary.values()),
        default="",
    )
    review_leads = [
        dict(hypothesis)
        for record in records
        for collection_name in ("role_hypotheses", "relationship_hypotheses")
        for hypothesis in _array(record.get(collection_name))
        if isinstance(hypothesis, Mapping)
        and _text(hypothesis.get("hypothesis_kind"))
        in DIRECTORY_REVIEW_HYPOTHESIS_KINDS
    ]
    review_leads.sort(
        key=lambda item: (
            _text(item.get("hypothesis_kind")),
            _text(item.get("organization") or item.get("counterpart_label")).casefold(),
            _text(item.get("display_value")).casefold(),
            _text(item.get("hypothesis_id")),
        )
    )
    aliases = sorted(
        {
            alias
            for member in members
            for alias in member["aliases"]
            if alias and alias.casefold() != primary_name.casefold()
        }
    )
    return {
        "entity_id": entity_id,
        "person_id": entity_id,
        "entity_kind": "canonical_person" if resolved else "unresolved_group",
        "resolution_state": _text(canonical[0].get("status")) if resolved else "review_required",
        "accepted_person_id": accepted_person_id,
        "primary_name": primary_name,
        **_person_name_presentation(primary_name, aliases),
        "aliases": aliases,
        "members": members,
        "source_records": [source_records[key] for key in sorted(source_records)],
        "organizations": [],
        "review_leads": review_leads,
        "activities": activities,
        "activity_summary": summary,
        "last_interaction_at": last_interaction_at,
        "identity_health": {
            "member_count": len(members),
            "source_record_count": len(source_records),
            "conflict_count": 0,
            "review_lead_count": len(review_leads),
            "requires_review": not resolved,
        },
    }


def _json_value(value: object, fallback: object) -> object:
    if not isinstance(value, str):
        return value if value is not None else fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _authority_activity(row: Mapping[str, Any]) -> dict[str, Any]:
    occurred_at = _text(row.get("occurred_at"))
    return {
        "observation_id": _text(row.get("observation_id")),
        "channel": _text(row.get("channel")),
        "occurred_at": occurred_at,
        "first_at": occurred_at,
        "last_at": occurred_at,
        "evidence_status": _text(row.get("evidence_status")),
        "participation_status": _text(row.get("participation_status")),
        "independence_group_ids": [_text(row.get("independence_group_id"))],
        "observation_count": 1,
        "source_member_id": "",
        "source_record_id": _text(row.get("source_record_id")),
        "title": _text(
            (_json_value(row.get("metadata_json"), {}) or {}).get("title")
            if isinstance(_json_value(row.get("metadata_json"), {}), Mapping)
            else ""
        ),
        "direction": _text(row.get("direction")),
        "source_locator": _json_value(row.get("source_locator_json"), {}),
    }


def _role_appointment(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence_ids = _json_value(row.get("evidence_ids_json"), row.get("evidence_ids") or [])
    return {
        "role_id": _text(row.get("role_id")),
        "role_type": _text(row.get("role_type")),
        "status": _text(row.get("status")) or "proposed",
        "starts_at": _text(row.get("starts_at")),
        "ends_at": _text(row.get("ends_at")),
        "project_id": _text(row.get("project_id")),
        "matter_id": _text(row.get("matter_id")),
        "conversation_id": _text(row.get("conversation_id")),
        "evidence_ids": sorted(
            {_text(value) for value in _array(evidence_ids) if _text(value)}
        ),
    }


def _role_rank(role: Mapping[str, Any]) -> tuple[int, int, str, str, str]:
    return (
        1 if _text(role.get("status")) in REVIEWED_ROLE_STATUSES else 0,
        1 if not _text(role.get("ends_at")) else 0,
        _text(role.get("starts_at")),
        _text(role.get("role_type")).casefold(),
        _text(role.get("role_id")),
    )


def _finalize_affiliations(person: dict[str, Any]) -> None:
    affiliations: list[dict[str, Any]] = []
    for raw in person.get("organizations") or []:
        affiliation = dict(raw)
        roles = [
            dict(role)
            for role in _array(affiliation.get("roles"))
            if isinstance(role, Mapping) and _text(role.get("role_id"))
        ]
        roles.sort(key=_role_rank, reverse=True)
        affiliation["roles"] = roles
        affiliation["role_types"] = [
            role["role_type"] for role in roles if role.get("role_type")
        ]
        affiliation["role_count"] = len(roles)
        affiliation["evidence_ids"] = sorted(
            {
                evidence_id
                for role in roles
                for evidence_id in role.get("evidence_ids") or []
            }
        )
        affiliation["starts_at"] = min(
            (role["starts_at"] for role in roles if role.get("starts_at")),
            default="",
        )
        affiliation["ends_at"] = "" if any(
            not role.get("ends_at") for role in roles
        ) else max(
            (role["ends_at"] for role in roles if role.get("ends_at")),
            default="",
        )
        if roles:
            affiliation["status"] = roles[0]["status"]
        affiliations.append(affiliation)
    affiliations.sort(
        key=lambda item: (
            _role_rank((item.get("roles") or [{}])[0]),
            _text(item.get("primary_name")).casefold(),
            _text(item.get("affiliation_id")),
        ),
        reverse=True,
    )
    person["organizations"] = affiliations
    person["primary_affiliation"] = dict(affiliations[0]) if affiliations else None
    person["additional_organization_count"] = max(0, len(affiliations) - 1)
    person["identity_health"]["affiliation_count"] = len(affiliations)
    person["identity_health"]["role_count"] = sum(
        len(item.get("roles") or []) for item in affiliations
    )


def _merge_activities(
    target: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    by_key = {
        (_text(row.get("channel")), _text(row.get("observation_id"))): dict(row)
        for row in target["activities"]
    }
    for row in rows:
        key = (_text(row.get("channel")), _text(row.get("observation_id")))
        by_key[key] = dict(row)
    target["activities"] = sorted(
        by_key.values(),
        key=lambda row: (
            _text(row.get("occurred_at")),
            _text(row.get("channel")),
            _text(row.get("observation_id")),
        ),
        reverse=True,
    )
    target["activity_summary"] = _summary(
        target["activities"], resolved=bool(target.get("accepted_person_id") or target.get("accepted_organization_id"))
    )
    target["last_interaction_at"] = max(
        (
            _text(summary.get("last_at"))
            for summary in target["activity_summary"].values()
        ),
        default="",
    )


def build_directory_index(
    source_payload: Mapping[str, Any],
    *,
    authority_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic canonical/unresolved directory from People rows."""

    source_items = [
        item
        for item in _array(source_payload.get("items"))
        if isinstance(item, Mapping) and _text(item.get("person_id"))
    ]
    authority = authority_snapshot or {}
    canonical_by_id: dict[str, list[Mapping[str, Any]]] = {}
    unresolved_by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in source_items:
        if item.get("identity_kind") == "canonical_person" and not _text(
            item.get("merged_into_person_id")
        ):
            canonical_by_id[_text(item.get("person_id"))] = [item]
    authority_sources = authority.get("sources") or {}
    for item in source_items:
        if item.get("identity_kind") == "canonical_person":
            continue
        linked_person_ids = {
            _text(authority_sources[source_id].get("person_id"))
            for source_id in (
                _text(source.get("source_record_id"))
                for source in _array(item.get("source_records"))
                if isinstance(source, Mapping)
            )
            if source_id in authority_sources
            and isinstance(authority_sources[source_id], Mapping)
            and _text(authority_sources[source_id].get("person_id"))
        }
        if len(linked_person_ids) == 1:
            linked_person_id = next(iter(linked_person_ids))
            if linked_person_id in canonical_by_id:
                canonical_by_id[linked_person_id].append(item)
                continue
        key = _text(item.get("primary_name")).casefold() or _text(item.get("person_id"))
        unresolved_by_name[key].append(item)

    people = [
        _directory_entity(records)
        for _person_id, records in sorted(canonical_by_id.items())
    ]
    people.extend(_directory_entity(records) for _, records in sorted(unresolved_by_name.items()))
    entity_by_member_id = {
        member["record_id"]: person
        for person in people
        for member in person["members"]
    }
    organization_mentions: dict[str, list[tuple[str, str, list[dict[str, Any]]]]] = defaultdict(list)
    for record in source_items:
        record_id = _text(record.get("person_id"))
        for raw_name in _array(record.get("organizations")):
            name = _text(raw_name)
            if not name:
                continue
            organization_mentions[name.casefold()].append(
                (name, record_id, _record_activities(record))
            )

    organizations: list[dict[str, Any]] = []
    for normalized_name, mentions in sorted(organization_mentions.items()):
        primary_name = mentions[0][0]
        organization_id = _stable_id("organization", normalized_name)
        linked_entity_ids: set[str] = set()
        activities: list[dict[str, Any]] = []
        for _name, record_id, record_activities in mentions:
            person = entity_by_member_id.get(record_id)
            if person is None:
                continue
            linked_entity_ids.add(person["entity_id"])
            person["organizations"].append(
                {
                    "affiliation_id": _stable_id(
                        "affiliation", person["entity_id"], organization_id
                    ),
                    "organization_id": organization_id,
                    "primary_name": primary_name,
                    "status": "proposed",
                    "basis": "provider_organization_string",
                    "roles": [],
                }
            )
            activities.extend(record_activities)
        activities.sort(
            key=lambda item: (
                _text(item.get("occurred_at")),
                _text(item.get("observation_id")),
            ),
            reverse=True,
        )
        organizations.append(
            {
                "organization_id": organization_id,
                "entity_kind": "organization",
                "resolution_state": "proposed",
                "accepted_organization_id": "",
                "primary_name": primary_name,
                "aliases": sorted(
                    {
                        name
                        for name, _record_id, _activities in mentions
                        if name.casefold() != primary_name.casefold()
                    }
                ),
                "affiliated_person_ids": sorted(linked_entity_ids),
                "activities": activities,
                "activity_summary": _summary(activities, resolved=False),
                "last_interaction_at": max(
                    (_text(activity.get("occurred_at")) for activity in activities),
                    default="",
                ),
                "identity_health": {
                    "source_name_count": len(mentions),
                    "affiliation_count": len(mentions),
                    "requires_review": True,
                },
            }
        )
    organization_by_id = {
        organization["organization_id"]: organization
        for organization in organizations
    }
    for organization_id, raw in sorted(
        (authority.get("organizations") or {}).items()
    ):
        if not isinstance(raw, Mapping) or _text(
            raw.get("merged_into_organization_id")
        ):
            continue
        aliases = _json_value(raw.get("aliases_json"), [])
        accepted = {
            "organization_id": organization_id,
            "entity_kind": "organization",
            "resolution_state": _text(raw.get("status")) or "reviewed",
            "accepted_organization_id": organization_id,
            "primary_name": _text(raw.get("primary_name")),
            "aliases": sorted(_text(value) for value in _array(aliases) if _text(value)),
            "domains": _json_value(raw.get("domains_json"), []),
            "websites": _json_value(raw.get("websites_json"), []),
            "organization_type": _text(raw.get("organization_type")),
            "locations": _json_value(raw.get("locations_json"), []),
            "parent_organization_id": _text(raw.get("parent_organization_id")),
            "source_records": [],
            "affiliated_person_ids": [],
            "activities": [],
            "activity_summary": _summary([], resolved=True),
            "last_interaction_at": "",
            "identity_health": {
                "source_name_count": 0,
                "affiliation_count": 0,
                "requires_review": False,
            },
        }
        organization_by_id[organization_id] = accepted

    for raw in (authority.get("organization_sources") or {}).values():
        if not isinstance(raw, Mapping):
            continue
        organization = organization_by_id.get(_text(raw.get("organization_id")))
        if organization is not None:
            organization.setdefault("source_records", []).append(dict(raw))

    person_by_id = {
        person["accepted_person_id"]: person
        for person in people
        if person.get("accepted_person_id")
    }
    role_rows: dict[str, Mapping[str, Any]] = {
        _text(role.get("role_id")): role
        for role in (authority.get("roles") or {}).values()
        if isinstance(role, Mapping) and _text(role.get("role_id"))
    }
    for record in source_items:
        for role in _array(record.get("roles")):
            if isinstance(role, Mapping) and _text(role.get("role_id")):
                role_rows.setdefault(_text(role.get("role_id")), role)
    for role_id, role in sorted(role_rows.items()):
        person = person_by_id.get(_text(role.get("person_id")))
        if person is None:
            continue
        organization_id = _text(role.get("organization_id"))
        organization = organization_by_id.get(organization_id)
        if not organization:
            continue
        affiliation = next(
            (
                item
                for item in person["organizations"]
                if item.get("organization_id") == organization_id
            ),
            None,
        )
        if affiliation is None:
            affiliation = {
                "affiliation_id": _stable_id(
                    "affiliation", person["entity_id"], organization_id
                ),
                "organization_id": organization_id,
                "primary_name": organization["primary_name"],
                "status": "proposed",
                "basis": "identity_role_projection",
                "roles": [],
            }
            person["organizations"].append(affiliation)
        affiliation.setdefault("roles", []).append(
            _role_appointment({**role, "role_id": role_id})
        )
        affiliation["basis"] = "identity_role_projection"
        organization["affiliated_person_ids"] = sorted(
            {*organization.get("affiliated_person_ids", []), person["entity_id"]}
        )

    for relationship_id, relationship in sorted(
        (authority.get("relationships") or {}).items()
    ):
        if not isinstance(relationship, Mapping) or _text(
            relationship.get("relationship_type")
        ) != "AFFILIATED_WITH":
            continue
        if _text(relationship.get("status")) not in REVIEWED_ROLE_STATUSES:
            continue
        if not (
            relationship.get("subject_type") == "person"
            and relationship.get("object_type") == "organization"
        ):
            continue
        person = person_by_id.get(_text(relationship.get("subject_id")))
        organization_id = _text(relationship.get("object_id"))
        organization = organization_by_id.get(organization_id)
        if person is None or organization is None:
            continue
        affiliation = next(
            (
                item
                for item in person["organizations"]
                if item.get("organization_id") == organization_id
            ),
            None,
        )
        if affiliation is None:
            affiliation = {
                "affiliation_id": relationship_id,
                "organization_id": organization_id,
                "primary_name": organization["primary_name"],
                "status": _text(relationship.get("status")),
                "basis": "identity_relationship_projection",
                "roles": [],
            }
            person["organizations"].append(affiliation)
        else:
            affiliation.update(
                affiliation_id=relationship_id,
                status=_text(relationship.get("status")),
                basis="identity_relationship_projection",
            )
        organization["affiliated_person_ids"] = sorted(
            {*organization.get("affiliated_person_ids", []), person["entity_id"]}
        )

    for person in people:
        _finalize_affiliations(person)
    for organization in organization_by_id.values():
        organization["identity_health"]["affiliation_count"] = len(
            organization.get("affiliated_person_ids") or []
        )

    activities_by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in (authority.get("activities") or {}).values():
        if isinstance(raw, Mapping):
            activities_by_subject[
                (_text(raw.get("subject_type")), _text(raw.get("subject_id")))
            ].append(_authority_activity(raw))
    for person_id, person in person_by_id.items():
        _merge_activities(person, activities_by_subject.get(("person", person_id), []))
    for organization_id, organization in organization_by_id.items():
        _merge_activities(
            organization,
            activities_by_subject.get(("organization", organization_id), []),
        )

    for raw in (authority.get("activity_coverage") or {}).values():
        if not isinstance(raw, Mapping):
            continue
        subject = (
            person_by_id.get(_text(raw.get("subject_id")))
            if raw.get("subject_type") == "person"
            else organization_by_id.get(_text(raw.get("subject_id")))
        )
        channel = _text(raw.get("channel"))
        if subject is not None and channel in CHANNELS:
            subject["activity_summary"][channel]["coverage_state"] = _text(
                raw.get("coverage_state")
            )

    organizations = sorted(
        organization_by_id.values(),
        key=lambda item: (
            _text(item.get("last_interaction_at")),
            _text(item.get("primary_name")).casefold(),
            _text(item.get("organization_id")),
        ),
        reverse=True,
    )
    people.sort(
        key=lambda item: (
            _text(item.get("last_interaction_at")),
            _text(item.get("primary_name")).casefold(),
            _text(item.get("entity_id")),
        ),
        reverse=True,
    )
    source_record_count = sum(
        len(item.get("source_records") or [])
        for item in people
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "people": people,
        "organizations": organizations,
        "review_targets": {
            "people": sorted(
                [
                    {
                        "id": item["accepted_person_id"],
                        "label": item["display_name"],
                        "display_name": item["display_name"],
                        "sort_name": item["sort_name"],
                        "name_completeness": item["name_completeness"],
                        "primary_name": item["primary_name"],
                        "aliases": item.get("aliases") or [],
                    }
                    for item in people
                    if item.get("accepted_person_id")
                ],
                key=lambda item: (item["sort_name"], item["id"]),
            ),
            "organizations": [
                {
                    "id": item["accepted_organization_id"],
                    "label": item["primary_name"],
                }
                for item in organizations
                if item.get("accepted_organization_id")
            ],
        },
        "counts": {
            "people": sum(item["entity_kind"] == "canonical_person" for item in people),
            "unresolved_groups": sum(item["entity_kind"] == "unresolved_group" for item in people),
            "source_records": source_record_count,
            "organizations": len(organizations),
            "review_leads": sum(len(item["review_leads"]) for item in people),
        },
        "source_projection": {
            "schema_version": _text(source_payload.get("schema_version")),
            "total": len(source_items),
        },
    }
    result["semantic_hash"] = hashlib.sha256(_canonical_json(result).encode("utf-8")).hexdigest()
    return result
