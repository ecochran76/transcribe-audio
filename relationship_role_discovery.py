"""Deterministic shadow discovery for contact roles and relationships."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import transcript_store
from mail_evidence_normalization import NormalizedMailEvidence
from mail_relationship_discovery import (
    MailRelationshipDiscovery,
    discover_mail_relationship_hypotheses,
)


SCHEMA_VERSION = "transcribe-audio.relationship-role-discovery.v1"
MIN_RECURRING_INVITATIONS = 2


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _object(value: object) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _text(value: object) -> str:
    return str(value or "").strip()


def _source_ref(source: dict[str, Any]) -> dict[str, str]:
    return {
        "provider": _text(source.get("provider")),
        "profile": _text(source.get("profile")),
        "record_type": _text(source.get("record_type")),
        "source_record_id": _text(source.get("source_record_id")),
        "match_basis": _text(source.get("match_basis")),
    }


def _candidate_id(kind: str, *parts: str) -> str:
    return f"{kind}-{_hash([kind, *parts])[:24]}"


def discover_relationship_roles(
    root: Optional[Path] = None,
    *,
    minimum_recurring_invitations: int = MIN_RECURRING_INVITATIONS,
    mail_evidence: NormalizedMailEvidence | None = None,
    mail_account_address: str = "",
) -> dict[str, Any]:
    """Build review-only hypotheses from exact local contact observations."""
    if minimum_recurring_invitations < 2:
        raise ValueError("Recurring calendar discovery requires at least two invitations.")
    with transcript_store.connect(root) as con:
        transcript_store.init_db(con)
        rows = [
            dict(row)
            for row in con.execute(
                "SELECT * FROM contacts ORDER BY id"
            ).fetchall()
        ]

    contacts: dict[str, dict[str, Any]] = {}
    input_rows: list[dict[str, Any]] = []
    for row in rows:
        metadata = _object(row.get("metadata_json"))
        calendar = metadata.get("calendar_attendee")
        if not isinstance(calendar, dict):
            continue
        enrichment = metadata.get("enrichment")
        if not isinstance(enrichment, dict):
            enrichment = {}
        contact_id = _text(row.get("id"))
        contacts[contact_id] = {
            "contact_id": contact_id,
            "label": _text(row.get("label")) or "Unnamed contact",
            "email": _text(row.get("email")).casefold(),
            "contact_class": _text(metadata.get("contact_class")) or "person_candidate",
            "updated_at": _text(row.get("updated_at")),
            "appearances": [
                dict(item)
                for item in calendar.get("appearances") or []
                if isinstance(item, dict)
            ],
            "source_records": [
                dict(item)
                for item in enrichment.get("source_records") or []
                if isinstance(item, dict)
            ],
        }
        input_rows.append(
            {
                "contact_id": contact_id,
                "label": contacts[contact_id]["label"],
                "contact_class": contacts[contact_id]["contact_class"],
                "metadata": metadata,
            }
        )

    role_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    affiliation_groups: dict[tuple[str, str], dict[str, Any]] = {}
    excluded_shared = 0
    for contact in contacts.values():
        if contact["contact_class"] == "shared_or_role_address":
            excluded_shared += 1
            continue
        for source in contact["source_records"]:
            source_ref = _source_ref(source)
            if source_ref["match_basis"] != "exact_email":
                continue
            for raw_role in source.get("roles") or []:
                if not isinstance(raw_role, dict):
                    continue
                title = _text(raw_role.get("title"))
                if not title:
                    continue
                organization = _text(raw_role.get("organization"))
                department = _text(raw_role.get("department"))
                key = (contact["contact_id"], title.casefold(), organization.casefold(), department.casefold())
                group = role_groups.setdefault(
                    key,
                    {
                        "hypothesis_id": _candidate_id("role-hypothesis", *key),
                        "hypothesis_kind": "contextual_role",
                        "role_type": "professional_title",
                        "display_value": title,
                        "organization": organization,
                        "department": department,
                        "effective_time": "unknown",
                        "status": "proposed",
                        "basis": "Exact-email provider source declares this title.",
                        "why_not_accepted": "Provider-declared title may be stale or contextual and has not been reviewed.",
                        "source_records": [],
                    },
                )
                if source_ref not in group["source_records"]:
                    group["source_records"].append(source_ref)
            for organization in source.get("organizations") or []:
                organization = _text(organization)
                if not organization:
                    continue
                key = (contact["contact_id"], organization.casefold())
                group = affiliation_groups.setdefault(
                    key,
                    {
                        "hypothesis_id": _candidate_id("relationship-hypothesis", "affiliation", *key),
                        "hypothesis_kind": "affiliation",
                        "relationship_type": "AFFILIATED_WITH",
                        "directionality": "directional",
                        "counterpart_type": "organization",
                        "counterpart_id": f"organization-{_hash(organization.casefold())[:24]}",
                        "counterpart_label": organization,
                        "status": "proposed",
                        "basis": "Exact-email provider source declares organization membership.",
                        "why_not_accepted": "Organization membership does not prove employment, current status, or speaker identity.",
                        "observation_count": 0,
                        "source_records": [],
                    },
                )
                if source_ref not in group["source_records"]:
                    group["source_records"].append(source_ref)
                    group["observation_count"] += 1

    event_contacts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for contact in contacts.values():
        if contact["contact_class"] == "shared_or_role_address":
            continue
        for appearance in contact["appearances"]:
            document_id = _text(appearance.get("document_id"))
            if not document_id:
                continue
            event_contacts[document_id].setdefault(contact["contact_id"], appearance)

    pair_evidence: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for document_id, participants in sorted(event_contacts.items()):
        contact_ids = sorted(participants)
        for index, left_id in enumerate(contact_ids):
            for right_id in contact_ids[index + 1 :]:
                appearance = participants[left_id]
                pair_evidence[(left_id, right_id)].append(
                    {
                        "document_id": document_id,
                        "recording_filename": _text(appearance.get("recording_filename")),
                        "recorded_at": _text(appearance.get("recorded_at")),
                        "event_summary": _text(
                            appearance.get("event_summary")
                            or appearance.get("calendar_summary")
                        ),
                    }
                )

    pair_candidates: list[tuple[str, str, dict[str, Any]]] = []
    for (left_id, right_id), evidence in sorted(pair_evidence.items()):
        if len(evidence) < minimum_recurring_invitations:
            continue
        evidence.sort(key=lambda item: (item["recorded_at"], item["document_id"]), reverse=True)
        observed_dates = sorted(item["recorded_at"] for item in evidence if item["recorded_at"])
        candidate = {
            "hypothesis_id": _candidate_id("relationship-hypothesis", "calendar-co-invitation", left_id, right_id),
            "hypothesis_kind": "calendar_co_invitation",
            "relationship_type": "CALENDAR_CO_INVITED_WITH",
            "directionality": "symmetric",
            "status": "proposed",
            "basis": f"Both contacts were listed on {len(evidence)} recording-associated calendar events.",
            "why_not_accepted": "Calendar co-invitation does not prove presence, interaction, identity, or a personal or professional relationship.",
            "observation_count": len(evidence),
            "first_observed_at": observed_dates[0] if observed_dates else "",
            "last_observed_at": observed_dates[-1] if observed_dates else "",
            "evidence": evidence,
        }
        pair_candidates.append((left_id, right_id, candidate))

    by_contact_id: dict[str, dict[str, list[dict[str, Any]]]] = {
        contact_id: {"role_hypotheses": [], "relationship_hypotheses": []}
        for contact_id in contacts
    }
    for (contact_id, *_), role in sorted(role_groups.items()):
        by_contact_id[contact_id]["role_hypotheses"].append(role)
    for (contact_id, _), relationship in sorted(affiliation_groups.items()):
        by_contact_id[contact_id]["relationship_hypotheses"].append(relationship)
    for left_id, right_id, candidate in pair_candidates:
        left = {
            **candidate,
            "counterpart_type": "contact_candidate",
            "counterpart_id": f"contact:{right_id}",
            "counterpart_label": contacts[right_id]["label"],
        }
        right = {
            **candidate,
            "counterpart_type": "contact_candidate",
            "counterpart_id": f"contact:{left_id}",
            "counterpart_label": contacts[left_id]["label"],
        }
        by_contact_id[left_id]["relationship_hypotheses"].append(left)
        by_contact_id[right_id]["relationship_hypotheses"].append(right)

    mail_discovery: MailRelationshipDiscovery | None = None
    if mail_evidence is not None:
        if not _text(mail_account_address):
            raise ValueError("Mail discovery requires an explicit account address.")
        mail_discovery = discover_mail_relationship_hypotheses(
            mail_evidence.observations,
            mail_evidence.independence_groups,
            contacts=contacts,
            account_address=mail_account_address,
            input_watermark=mail_evidence.input_watermark,
        )
        for hypothesis in mail_discovery.hypotheses:
            subject_id = hypothesis["subject_contact_id"]
            if subject_id not in by_contact_id:
                raise ValueError("Mail hypothesis subject is outside Contacts.")
            if hypothesis["hypothesis_kind"] == "contextual_role":
                by_contact_id[subject_id]["role_hypotheses"].append(
                    {
                        **hypothesis,
                        "display_value": hypothesis["counterpart_label"],
                        "organization": "",
                        "department": "",
                        "effective_time": (
                            f"{hypothesis['first_observed_at']} to "
                            f"{hypothesis['last_observed_at']}"
                        ),
                        "evidence_source": "mail_metadata",
                    }
                )
                continue
            by_contact_id[subject_id]["relationship_hypotheses"].append(
                {
                    **hypothesis,
                    "mail_direction": (
                        "sent"
                        if hypothesis["hypothesis_kind"] == "sent_mail"
                        else "symmetric"
                    ),
                    "evidence_source": "mail_metadata",
                }
            )
            counterpart_id = hypothesis["counterpart_id"]
            if (
                hypothesis["counterpart_type"] == "contact_candidate"
                and counterpart_id in by_contact_id
            ):
                by_contact_id[counterpart_id]["relationship_hypotheses"].append(
                    {
                        **hypothesis,
                        "counterpart_id": subject_id,
                        "counterpart_label": contacts[subject_id]["label"],
                        "mail_direction": (
                            "received"
                            if hypothesis["hypothesis_kind"] == "sent_mail"
                            else "symmetric"
                        ),
                        "evidence_source": "mail_metadata",
                    }
                )
    for candidates in by_contact_id.values():
        candidates["role_hypotheses"].sort(
            key=lambda item: (item["display_value"].casefold(), item["organization"].casefold())
        )
        candidates["relationship_hypotheses"].sort(
            key=lambda item: (
                item["hypothesis_kind"],
                -int(item.get("observation_count") or 0),
                item["counterpart_label"].casefold(),
            )
        )

    contacts_with_candidates = sum(
        1
        for value in by_contact_id.values()
        if value["role_hypotheses"] or value["relationship_hypotheses"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority_mode": "shadow_hypotheses_only",
        "input_watermark": _hash(
            {
                "contacts": input_rows,
                "mail": mail_evidence.input_watermark if mail_evidence else "",
            }
        ),
        "built_at": max((_text(row.get("updated_at")) for row in rows), default=""),
        "minimum_recurring_invitations": minimum_recurring_invitations,
        "contact_count": len(contacts),
        "contacts_with_candidates": contacts_with_candidates,
        "excluded_shared_address_count": excluded_shared,
        "role_hypothesis_count": len(role_groups),
        "affiliation_hypothesis_count": len(affiliation_groups),
        "calendar_co_invitation_hypothesis_count": len(pair_candidates),
        "mail_hypothesis_count": (
            len(mail_discovery.hypotheses) if mail_discovery else 0
        ),
        "mail_hypothesis_counts": (
            {
                kind: sum(
                    1
                    for item in mail_discovery.hypotheses
                    if item["hypothesis_kind"] == kind
                )
                for kind in sorted(
                    {item["hypothesis_kind"] for item in mail_discovery.hypotheses}
                )
            }
            if mail_discovery
            else {}
        ),
        "mail_excluded_reason_counts": (
            mail_discovery.excluded_reason_counts if mail_discovery else {}
        ),
        "mail_input_watermark": (
            mail_discovery.input_watermark if mail_discovery else ""
        ),
        "mail_hypotheses": (
            list(mail_discovery.hypotheses) if mail_discovery else []
        ),
        "accepted_effect_count": 0,
        "provider_write_count": 0,
        "person_merge_count": 0,
        "speaker_assignment_apply_count": 0,
        "by_contact_id": by_contact_id,
    }
