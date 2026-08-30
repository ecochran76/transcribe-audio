from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from mail_evidence_normalization import normalize_mail_address
from mail_relationship_contracts import (
    THRESHOLDS,
    ZERO_EFFECTS,
    validate_mail_artifact,
)


@dataclass(frozen=True)
class MailRelationshipDiscovery:
    hypotheses: tuple[dict[str, Any], ...]
    excluded_reason_counts: dict[str, int]
    input_watermark: str


def _hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contact_indexes(
    contacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_email: dict[str, dict[str, str]] = {}
    for key, raw in contacts.items():
        contact_id = str(raw.get("contact_id") or key).strip()
        email = normalize_mail_address(raw.get("email"))
        contact = {
            "contact_id": contact_id,
            "label": str(raw.get("label") or "Unnamed contact").strip(),
            "email": email,
            "contact_class": str(
                raw.get("contact_class") or "person_candidate"
            ).strip(),
        }
        if not contact_id or contact_id in by_id or email in by_email:
            raise ValueError("Mail discovery contacts require unique IDs and emails.")
        by_id[contact_id] = contact
        by_email[email] = contact
    return by_id, by_email


def _contact_exclusion(contact_class: str) -> str | None:
    return {
        "shared_or_role_address": "excluded_shared_address",
        "shared_address": "excluded_shared_address",
        "role_address": "excluded_role_address",
        "mailing_list": "excluded_mailing_list",
        "automated_sender": "excluded_automated_sender",
    }.get(contact_class)


def _hypothesis(
    *,
    kind: str,
    relationship_type: str,
    directionality: str,
    subject: Mapping[str, str],
    counterpart_type: str,
    counterpart_id: str,
    counterpart_label: str,
    observations: Sequence[Mapping[str, Any]],
    group_threads: Mapping[str, str],
    basis: str,
    why_not_accepted: str,
    conflicts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    observation_ids = sorted(
        {str(item["observation_id"]) for item in observations}
    )
    group_ids = sorted(
        {str(item["independence_group_id"]) for item in observations}
    )
    thread_ids = {group_threads[group_id] for group_id in group_ids}
    observed = sorted(str(item["source_event_at"]) for item in observations)
    core = {
        "hypothesis_kind": kind,
        "relationship_type": relationship_type,
        "directionality": directionality,
        "subject_contact_id": subject["contact_id"],
        "counterpart_type": counterpart_type,
        "counterpart_id": counterpart_id,
        "counterpart_label": counterpart_label,
        "evidence_observation_ids": observation_ids,
        "evidence_independence_group_ids": group_ids,
        "observation_count": len(observation_ids),
        "independent_thread_count": len(thread_ids),
        "first_observed_at": observed[0],
        "last_observed_at": observed[-1],
        "status": "proposed",
        "basis": basis,
        "why_not_accepted": why_not_accepted,
        "conflicts": [dict(value) for value in conflicts],
        "effect_counts": dict(ZERO_EFFECTS),
    }
    artifact = {
        "schema_version": "transcribe-audio.mail-relationship-hypothesis.v1",
        "hypothesis_id": "mail-hypothesis-" + _hash(core)[:32],
        **core,
    }
    validate_mail_artifact("mail_relationship_hypothesis", artifact)
    return artifact


def discover_mail_relationship_hypotheses(
    observations: Sequence[Mapping[str, Any]],
    independence_groups: Sequence[Mapping[str, Any]],
    *,
    contacts: Mapping[str, Mapping[str, Any]],
    account_address: str,
    input_watermark: str,
) -> MailRelationshipDiscovery:
    """Derive review-only mail semantics from validated deterministic evidence."""
    account = normalize_mail_address(account_address)
    if not str(input_watermark or "").strip():
        raise ValueError("Mail discovery requires an input watermark.")
    by_id, by_email = _contact_indexes(contacts)
    groups = {
        str(item["group_id"]): validate_mail_artifact(
            "mail_independence_group", item
        )
        for item in independence_groups
    }
    if len(groups) != len(independence_groups):
        raise ValueError("Mail discovery independence group IDs must be unique.")
    group_threads = {
        group_id: str(item["independent_thread_key"])
        for group_id, item in groups.items()
    }
    members_by_group = {
        group_id: set(item["member_observation_ids"])
        for group_id, item in groups.items()
    }

    exclusions: Counter[str] = Counter()
    exclusion_keys: set[tuple[str, str]] = set()
    valid: list[dict[str, Any]] = []
    for raw in observations:
        item = validate_mail_artifact("mail_observation", raw)
        group_id = str(item["independence_group_id"])
        if (
            group_id not in groups
            or item["observation_id"] not in members_by_group[group_id]
        ):
            raise ValueError("Mail observation is not accounted for by its group.")
        reason = item.get("excluded_reason_code")
        if reason:
            exclusions[str(reason)] += 1
            continue
        valid.append(item)

    def contact_for(item: Mapping[str, Any], address: str) -> dict[str, str] | None:
        normalized = normalize_mail_address(address)
        mapped_id = str(
            item["contact_ids_by_address"].get(normalized) or ""
        ).strip()
        contact = by_email.get(normalized)
        if (
            contact is None
            or not mapped_id
            or mapped_id != contact["contact_id"]
            or by_id.get(mapped_id) != contact
        ):
            reason = "excluded_unresolved_contact"
        else:
            reason = _contact_exclusion(contact["contact_class"])
        if reason:
            key = (str(item["observation_id"]), normalized)
            if key not in exclusion_keys:
                exclusions[reason] += 1
                exclusion_keys.add(key)
            return None
        return contact

    sent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    coparticipants: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    roles: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    affiliations: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in valid:
        participants = item["participants"]
        sender = contact_for(item, participants["from"][0])
        recipients: list[dict[str, str]] = []
        for address in (*participants["to"], *participants["cc"]):
            contact = contact_for(item, address)
            if contact and contact not in recipients:
                recipients.append(contact)
        if sender and len(recipients) == 1 and not item["signature_observations"]:
            for recipient in recipients:
                if recipient["contact_id"] != sender["contact_id"]:
                    sent[(sender["contact_id"], recipient["contact_id"])].append(item)

        non_account = []
        for address in (
            *participants["from"],
            *participants["to"],
            *participants["cc"],
        ):
            if normalize_mail_address(address) == account:
                continue
            contact = contact_for(item, address)
            if contact and contact["contact_id"] not in {
                value["contact_id"] for value in non_account
            }:
                non_account.append(contact)
        for left, right in combinations(
            sorted(non_account, key=lambda value: value["contact_id"]), 2
        ):
            coparticipants[(left["contact_id"], right["contact_id"])].append(item)

        for signature in item["signature_observations"]:
            contact = contact_for(item, signature["address"])
            if not contact:
                continue
            title = str(signature["title"] or "").strip()
            organization = str(signature["organization"] or "").strip()
            department = str(signature["department"] or "").strip()
            if title:
                roles[
                    (
                        contact["contact_id"],
                        title.casefold(),
                        organization.casefold(),
                        department.casefold(),
                    )
                ].append(item)
            if organization:
                affiliations[
                    (contact["contact_id"], organization.casefold())
                ].append(item)

    hypotheses: list[dict[str, Any]] = []
    for (subject_id, counterpart_id), evidence in sorted(sent.items()):
        subject = by_id[subject_id]
        counterpart = by_id[counterpart_id]
        hypotheses.append(
            _hypothesis(
                kind="sent_mail",
                relationship_type="SENT_MAIL_TO",
                directionality="directional",
                subject=subject,
                counterpart_type="contact_candidate",
                counterpart_id=counterpart_id,
                counterpart_label=counterpart["label"],
                observations=evidence,
                group_threads=group_threads,
                basis="Bounded mail metadata records this sender and recipient.",
                why_not_accepted=(
                    "A message transmission does not establish a named personal "
                    "or professional relationship."
                ),
            )
        )

    unordered_pairs = {
        tuple(sorted((subject_id, counterpart_id)))
        for subject_id, counterpart_id in sent
        if (counterpart_id, subject_id) in sent
    }
    for left_id, right_id in sorted(unordered_pairs):
        evidence = sent[(left_id, right_id)] + sent[(right_id, left_id)]
        thread_count = len(
            {
                group_threads[str(item["independence_group_id"])]
                for item in evidence
            }
        )
        if thread_count < THRESHOLDS["min_correspondence_threads"]:
            continue
        hypotheses.append(
            _hypothesis(
                kind="correspondence",
                relationship_type="CORRESPONDED_WITH",
                directionality="symmetric",
                subject=by_id[left_id],
                counterpart_type="contact_candidate",
                counterpart_id=right_id,
                counterpart_label=by_id[right_id]["label"],
                observations=evidence,
                group_threads=group_threads,
                basis=(
                    f"Mail occurred in both directions across {thread_count} "
                    "independent threads."
                ),
                why_not_accepted=(
                    "Correspondence does not establish a named personal or "
                    "professional relationship."
                ),
            )
        )

    for (left_id, right_id), evidence in sorted(coparticipants.items()):
        thread_count = len(
            {
                group_threads[str(item["independence_group_id"])]
                for item in evidence
            }
        )
        if thread_count < THRESHOLDS["min_coparticipant_threads"]:
            continue
        hypotheses.append(
            _hypothesis(
                kind="thread_coparticipation",
                relationship_type="MAIL_THREAD_COPARTICIPANT_WITH",
                directionality="symmetric",
                subject=by_id[left_id],
                counterpart_type="contact_candidate",
                counterpart_id=right_id,
                counterpart_label=by_id[right_id]["label"],
                observations=evidence,
                group_threads=group_threads,
                basis=(
                    f"Both contacts participated in {thread_count} independent "
                    "mail threads."
                ),
                why_not_accepted=(
                    "Mail thread coparticipation does not prove interaction, "
                    "identity, or a personal or professional relationship."
                ),
            )
        )

    roles_by_contact: dict[str, list[tuple[tuple[str, str, str, str], list[dict[str, Any]]]]] = defaultdict(list)
    for key, evidence in roles.items():
        roles_by_contact[key[0]].append((key, evidence))
    for contact_id, contact_roles in sorted(roles_by_contact.items()):
        for key, evidence in sorted(contact_roles):
            _, title_key, organization_key, department_key = key
            signature = next(
                signature
                for item in evidence
                for signature in item["signature_observations"]
                if str(signature["title"]).strip().casefold() == title_key
                and str(signature["organization"]).strip().casefold()
                == organization_key
                and str(signature["department"]).strip().casefold()
                == department_key
            )
            conflicts = [
                {
                    "reason": "conflicting_structured_role",
                    "title": other_signature["title"],
                    "organization": other_signature["organization"],
                    "department": other_signature["department"],
                    "observed_at": other_signature["observed_at"],
                }
                for other_key, other_evidence in contact_roles
                if other_key != key
                for other_item in other_evidence
                for other_signature in other_item["signature_observations"]
            ]
            hypotheses.append(
                _hypothesis(
                    kind="contextual_role",
                    relationship_type="HAS_CONTEXTUAL_ROLE",
                    directionality="directional",
                    subject=by_id[contact_id],
                    counterpart_type="contextual_role",
                    counterpart_id="contextual-role-" + _hash(key)[:24],
                    counterpart_label=str(signature["title"]),
                    observations=evidence,
                    group_threads=group_threads,
                    basis="A structured mail signature declares this title.",
                    why_not_accepted=(
                        "A signature title may be stale or contextual and has "
                        "not been reviewed."
                    ),
                    conflicts=conflicts,
                )
            )

    for (contact_id, organization_key), evidence in sorted(affiliations.items()):
        organization = next(
            str(signature["organization"])
            for item in evidence
            for signature in item["signature_observations"]
            if str(signature["organization"]).strip().casefold()
            == organization_key
        )
        hypotheses.append(
            _hypothesis(
                kind="affiliation",
                relationship_type="AFFILIATED_WITH",
                directionality="directional",
                subject=by_id[contact_id],
                counterpart_type="organization",
                counterpart_id="organization-" + _hash(organization_key)[:24],
                counterpart_label=organization,
                observations=evidence,
                group_threads=group_threads,
                basis="A structured mail signature declares this organization.",
                why_not_accepted=(
                    "A signature organization does not prove employment, "
                    "current status, or speaker identity."
                ),
            )
        )

    hypotheses.sort(
        key=lambda item: (
            item["hypothesis_kind"],
            item["subject_contact_id"],
            item["counterpart_label"].casefold(),
            item["hypothesis_id"],
        )
    )
    return MailRelationshipDiscovery(
        hypotheses=tuple(hypotheses),
        excluded_reason_counts=dict(sorted(exclusions.items())),
        input_watermark=str(input_watermark),
    )
