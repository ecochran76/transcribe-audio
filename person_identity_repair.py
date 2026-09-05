"""Derived, stale-safe repair workflow for accepted person authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from identity_learning_ledger import IdentityLearningLedger


REPAIR_QUEUE_SCHEMA = "transcribe-audio.person-identity-repair-queue.v1"
REPAIR_SUBMISSION_SCHEMA = "transcribe-audio.person-identity-repair-submission.v1"
REPAIR_RECEIPT_SCHEMA = "transcribe-audio.person-identity-repair-receipt.v1"
ALLOWED_ACTIONS = {"correct_name", "merge_people", "mark_distinct"}
GENERIC_MAILBOX_LOCAL_PARTS = {
    "admin",
    "contact",
    "hello",
    "info",
    "office",
    "research",
    "sales",
    "service",
    "support",
    "team",
}


class PersonIdentityRepairError(ValueError):
    """Raised when a person repair cannot preserve the repair contract."""


class StalePersonIdentityRepair(PersonIdentityRepairError):
    """Raised when current accepted identity authority differs from the queue row."""


def _text(value: object) -> str:
    return str(value or "").strip()


def _array(value: object) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{_hash(list(parts))[:24]}"


def _organization_names(item: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            _text(value.get("primary_name"))
            for value in _array(item.get("organizations"))
            if isinstance(value, Mapping) and _text(value.get("primary_name"))
        },
        key=str.casefold,
    )


def _evidence_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    sources = [
        value
        for value in _array(item.get("source_records"))
        if isinstance(value, Mapping)
    ]
    providers = sorted(
        {_text(value.get("provider_kind")) for value in sources if _text(value.get("provider_kind"))}
    )
    labels = sorted(
        {_text(value.get("label")) for value in sources if _text(value.get("label"))},
        key=str.casefold,
    )
    return {
        "source_count": len(sources),
        "providers": providers,
        "labels": labels,
        "organization_names": _organization_names(item),
    }


def _has_generic_mailbox(item: Mapping[str, Any]) -> bool:
    for source in _array(item.get("source_records")):
        if not isinstance(source, Mapping):
            continue
        external_ref = _text(source.get("external_ref"))
        if "@" in external_ref and external_ref.split("@", maxsplit=1)[0].casefold() in GENERIC_MAILBOX_LOCAL_PARTS:
            return True
    return False


def _human_name_values(item: Mapping[str, Any]) -> list[str]:
    values = [
        item.get("display_name"),
        item.get("primary_name"),
        *_array(item.get("aliases")),
        *_array(item.get("person_name_candidates")),
    ]
    values.extend(
        value.get("label")
        for value in _array(item.get("source_records"))
        if isinstance(value, Mapping)
    )
    result: list[str] = []
    for value in values:
        name = _text(value)
        if not name or "@" in name or "·" in name:
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*\.?", name)
        if len(tokens) < 2:
            continue
        normalized = " ".join(tokens)
        if normalized.casefold() not in {item.casefold() for item in result}:
            result.append(normalized)
    return result


def _name_shape(value: str) -> tuple[str, str, tuple[str, ...], str] | None:
    tokens = [token.rstrip(".").casefold() for token in value.split()]
    if len(tokens) < 2:
        return None
    given = tokens[:-1]
    family = tokens[-1]
    if len(given) >= 2 and len(given[0]) == 1 and len(given[1]) > 1:
        return given[0], given[1], tuple(token[0] for token in given[2:]), family
    return "", given[0], tuple(token[0] for token in given[1:]), family


def _preferred_middle_shape(value: str) -> tuple[str, str, tuple[str, ...], str] | None:
    shape = _name_shape(value)
    return shape if shape and shape[0] else None


def _compatible_name_shapes(
    left: tuple[str, str, tuple[str, ...], str],
    right: tuple[str, str, tuple[str, ...], str],
) -> bool:
    left_leading, left_preferred, left_middle, left_family = left
    right_leading, right_preferred, right_middle, right_family = right
    if (left_preferred, left_family) != (right_preferred, right_family):
        return False
    if left_leading and right_leading and left_leading != right_leading:
        return False
    return all(
        left_initial == right_initial
        for left_initial, right_initial in zip(left_middle, right_middle)
    )


def _shared_organizations(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    left_names = {value.casefold(): value for value in _organization_names(left)}
    right_names = {value.casefold(): value for value in _organization_names(right)}
    return sorted(
        (left_names[key] for key in left_names.keys() & right_names.keys()),
        key=str.casefold,
    )


def _identity_item_id(item: Mapping[str, Any]) -> str:
    return _text(item.get("accepted_person_id")) or _text(item.get("person_id"))


def _repair_participant(
    item: Mapping[str, Any], people: Mapping[str, Any]
) -> dict[str, Any]:
    person_id = _identity_item_id(item)
    person = people.get(person_id)
    return {
        "person_id": person_id,
        "primary_name": _text(
            person.get("primary_name")
            if isinstance(person, Mapping)
            else item.get("primary_name")
        ),
        "display_name": _text(item.get("display_name")),
        "candidate_names": list(_array(item.get("person_name_candidates"))),
        "in_identity_ledger": person_id in people,
        "evidence": _evidence_summary(item),
    }


def _finding_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value) if key not in {"content_sha256"}}


def _finding_was_rejected(
    authority_snapshot: Mapping[str, Any], content_sha256: str
) -> bool:
    reconciliations = authority_snapshot.get("reconciliations") or {}
    if not isinstance(reconciliations, Mapping):
        return False
    for value in reconciliations.values():
        if not isinstance(value, Mapping) or _text(
            value.get("decision_status")
        ) != "rejected":
            continue
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            try:
                metadata = json.loads(_text(value.get("metadata_json")) or "{}")
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(metadata, Mapping) and _text(
            metadata.get("finding_content_sha256")
        ) == content_sha256:
            return True
    return False


def build_person_identity_repair_queue(
    directory_payload: Mapping[str, Any],
    authority_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive repair findings without changing accepted identity authority."""

    people = authority_snapshot.get("people") or {}
    identity_items: list[dict[str, Any]] = []
    accepted_items: list[dict[str, Any]] = []
    for raw in _array(directory_payload.get("items")):
        if not isinstance(raw, Mapping):
            continue
        item_id = _identity_item_id(raw)
        person_id = _text(raw.get("accepted_person_id"))
        person = people.get(person_id) if isinstance(people, Mapping) else None
        if not item_id or (
            person_id
            and isinstance(person, Mapping)
            and _text(person.get("merged_into_person_id"))
        ):
            continue
        identity_items.append(dict(raw))
        if person_id:
            accepted_items.append(dict(raw))

    findings: list[dict[str, Any]] = []
    by_display_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted_items:
        person_id = _text(item.get("accepted_person_id"))
        person = people.get(person_id) if isinstance(people, Mapping) else None
        current_name = _text(
            person.get("primary_name") if isinstance(person, Mapping) else item.get("primary_name")
        )
        display_name = _text(item.get("display_name")) or current_name
        candidates = []
        for value in _array(item.get("person_name_candidates")):
            candidate = _text(value)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        evidence = _evidence_summary(item)
        base = {
            "person_id": person_id,
            "display_name": display_name,
            "current_primary_name": current_name,
            "input_watermark": _text(
                person.get("input_watermark") if isinstance(person, Mapping) else ""
            ),
            "evidence": evidence,
        }
        generic_mailbox = _has_generic_mailbox(item)
        if isinstance(person, Mapping) and (generic_mailbox or (
            _text(item.get("name_completeness")) != "complete" and not candidates
        )):
            finding = {
                **base,
                "repair_kind": "identity_ambiguity",
                "reason": (
                    "Shared or role mailbox requires a person-versus-mailbox decision."
                    if generic_mailbox
                    else "No complete human name exists in retained source evidence."
                ),
                "suggested_primary_name": "",
                "candidate_names": candidates,
                "allowed_actions": [],
            }
            finding["repair_id"] = _stable_id(
                "person-repair", "identity_ambiguity", person_id, current_name
            )
            finding["content_sha256"] = _hash(_finding_content(finding))
            findings.append(finding)
        elif (
            isinstance(person, Mapping)
            and display_name
            and display_name.casefold() != current_name.casefold()
        ):
            finding = {
                **base,
                "repair_kind": "canonical_name",
                "reason": "A cleaner human name is present in retained source evidence.",
                "suggested_primary_name": display_name,
                "candidate_names": candidates,
                "allowed_actions": ["correct_name"],
            }
            finding["repair_id"] = _stable_id(
                "person-repair", "canonical_name", person_id, current_name, display_name
            )
            finding["content_sha256"] = _hash(_finding_content(finding))
            findings.append(finding)
        by_display_name[display_name.casefold()].append(item)

    for normalized_name, items in sorted(by_display_name.items()):
        if not normalized_name or len(items) < 2:
            continue
        person_ids = sorted(_text(item.get("accepted_person_id")) for item in items)
        participants = []
        for item in sorted(items, key=lambda value: _text(value.get("accepted_person_id"))):
            person_id = _text(item.get("accepted_person_id"))
            participants.append(_repair_participant(item, people))
        finding = {
            "repair_kind": "possible_duplicate",
            "display_name": _text(items[0].get("display_name")),
            "person_ids": person_ids,
            "participants": participants,
            "reason": "Equal display names are a review lead, not merge proof.",
            "allowed_actions": ["merge_people"],
        }
        finding["repair_id"] = _stable_id(
            "person-repair", "possible_duplicate", *person_ids
        )
        finding["content_sha256"] = _hash(_finding_content(finding))
        findings.append(finding)

    seen_variant_pairs: set[tuple[str, str]] = set()
    for left_index, left in enumerate(identity_items):
        for right in identity_items[left_index + 1 :]:
            if not (
                _text(left.get("accepted_person_id"))
                or _text(right.get("accepted_person_id"))
            ):
                continue
            left_id = _identity_item_id(left)
            right_id = _identity_item_id(right)
            pair = tuple(sorted((left_id, right_id)))
            if pair in seen_variant_pairs:
                continue
            left_names = _human_name_values(left)
            right_names = _human_name_values(right)
            shared_organizations = _shared_organizations(left, right)
            shared_preferred_middle = sorted(
                {
                    left_name
                    for left_name in left_names
                    if _preferred_middle_shape(left_name)
                    and any(
                        left_name.casefold() == right_name.casefold()
                        for right_name in right_names
                    )
                },
                key=str.casefold,
            )
            compatibility_basis = ""
            if shared_preferred_middle:
                compatibility_basis = "preferred_middle_alias_overlap"
            elif shared_organizations and any(
                left_name.casefold() != right_name.casefold()
                and (left_shape := _name_shape(left_name)) is not None
                and (right_shape := _name_shape(right_name)) is not None
                and _compatible_name_shapes(left_shape, right_shape)
                for left_name in left_names
                for right_name in right_names
            ):
                compatibility_basis = "compatible_initial_with_shared_organization"
            if not compatibility_basis:
                continue
            seen_variant_pairs.add(pair)
            participants = [
                _repair_participant(item, people)
                for item in sorted(
                    (left, right), key=lambda item: _identity_item_id(item)
                )
            ]
            finding = {
                "repair_kind": "name_variant_candidate",
                "display_name": " / ".join(
                    _text(item.get("display_name")) for item in (left, right)
                ),
                "person_ids": list(pair),
                "participants": participants,
                "compatibility_basis": compatibility_basis,
                "shared_name_variants": shared_preferred_middle,
                "shared_organizations": shared_organizations,
                "reason": (
                    "A preferred-middle-name form is shared across otherwise "
                    "separate identity records; review is required."
                ),
                "allowed_actions": ["merge_people", "mark_distinct"],
            }
            finding["repair_id"] = _stable_id(
                "person-repair", "name_variant_candidate", *pair
            )
            finding["content_sha256"] = _hash(_finding_content(finding))
            if not _finding_was_rejected(
                authority_snapshot, finding["content_sha256"]
            ):
                findings.append(finding)

    order = {
        "canonical_name": 0,
        "name_variant_candidate": 1,
        "identity_ambiguity": 2,
        "possible_duplicate": 3,
    }
    findings.sort(
        key=lambda item: (
            order.get(_text(item.get("repair_kind")), 99),
            _text(item.get("display_name")).casefold(),
            _text(item.get("repair_id")),
        )
    )
    counts = {
        "all": len(findings),
        "actionable": sum(bool(item.get("allowed_actions")) for item in findings),
        "canonical_name": sum(item["repair_kind"] == "canonical_name" for item in findings),
        "identity_ambiguity": sum(item["repair_kind"] == "identity_ambiguity" for item in findings),
        "possible_duplicate": sum(item["repair_kind"] == "possible_duplicate" for item in findings),
        "name_variant_candidate": sum(
            item["repair_kind"] == "name_variant_candidate" for item in findings
        ),
    }
    return {
        "schema_version": REPAIR_QUEUE_SCHEMA,
        "items": findings,
        "counts": counts,
        "mutation_count": 0,
        "default_sort": "repair_kind:asc,display_name:asc",
    }


def _event_repair_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    changes = payload.get("changes")
    if isinstance(changes, Mapping) and isinstance(changes.get("metadata"), Mapping):
        return changes["metadata"]
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _receipt(
    *, repair_id: str, action: str, event_id: str, status: str, idempotent_replay: bool
) -> dict[str, Any]:
    return {
        "schema_version": REPAIR_RECEIPT_SCHEMA,
        "repair_id": repair_id,
        "action": action,
        "event_id": event_id,
        "status": status,
        "idempotent_replay": idempotent_replay,
        "provider_write_count": 0,
    }


def record_person_identity_repair(
    root: Path | None,
    submission: Mapping[str, Any],
    current_queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one exact reviewed repair and rebuild the identity projection."""

    if submission.get("schema_version") != REPAIR_SUBMISSION_SCHEMA:
        raise PersonIdentityRepairError("Person identity repair schema is unsupported.")
    required = (
        "repair_id",
        "repair_kind",
        "action",
        "expected_content_sha256",
        "reviewer",
        "decided_at",
        "idempotency_key",
    )
    if any(not _text(submission.get(field)) for field in required):
        raise PersonIdentityRepairError("Person identity repair submission is incomplete.")
    action = _text(submission.get("action"))
    if action not in ALLOWED_ACTIONS:
        raise PersonIdentityRepairError("Person identity repair action is unsupported.")

    repair_id = _text(submission.get("repair_id"))
    idempotency_key = _text(submission.get("idempotency_key"))
    submission_hash = _hash(dict(submission))
    ledger = IdentityLearningLedger(root)
    for event in ledger.events(
        event_types=("person_corrected", "people_merged", "reconciliation_decided")
    ):
        metadata = _event_repair_metadata(event)
        if (
            _text(metadata.get("repair_id")) == repair_id
            and _text(metadata.get("repair_idempotency_key")) == idempotency_key
        ):
            if _text(metadata.get("submission_content_sha256")) != submission_hash:
                raise PersonIdentityRepairError(
                    "Repair idempotency key was reused with different content."
                )
            return _receipt(
                repair_id=repair_id,
                action=action,
                event_id=_text(event.get("id")),
                status="unchanged",
                idempotent_replay=True,
            )

    finding = next(
        (
            item
            for item in _array(current_queue.get("items"))
            if isinstance(item, Mapping) and _text(item.get("repair_id")) == repair_id
        ),
        None,
    )
    if finding is None or _text(finding.get("content_sha256")) != _text(
        submission.get("expected_content_sha256")
    ):
        raise StalePersonIdentityRepair(
            "Person identity repair is stale; reload Repairs before applying it."
        )
    if action not in _array(finding.get("allowed_actions")):
        raise PersonIdentityRepairError("Repair action is not allowed for this finding.")

    metadata = {
        "repair_id": repair_id,
        "repair_kind": _text(finding.get("repair_kind")),
        "repair_idempotency_key": idempotency_key,
        "submission_content_sha256": submission_hash,
        "finding_content_sha256": _text(finding.get("content_sha256")),
    }
    common = {
        "actor_id": _text(submission.get("reviewer")),
        "occurred_at": _text(submission.get("decided_at")),
        "idempotency_key": idempotency_key,
        "subject_type": "person_identity_repair",
        "subject_id": repair_id,
    }
    if action == "correct_name":
        person_id = _text(submission.get("person_id"))
        if person_id != _text(finding.get("person_id")):
            raise PersonIdentityRepairError("Repair person_id does not match the finding.")
        replacement = _text(submission.get("replacement_primary_name"))
        if replacement not in _array(finding.get("candidate_names")):
            raise PersonIdentityRepairError(
                "Replacement name must be one of the retained human-name candidates."
            )
        event_type = "person_corrected"
        payload = {
            "person_id": person_id,
            "changes": {"primary_name": replacement, "metadata": metadata},
        }
    elif action == "merge_people":
        person_ids = sorted(_text(value) for value in _array(finding.get("person_ids")))
        target_person_id = _text(submission.get("target_person_id"))
        if target_person_id not in person_ids:
            raise PersonIdentityRepairError("Merge target is not part of the finding.")
        events: list[dict[str, Any]] = []
        for participant in _array(finding.get("participants")):
            if not isinstance(participant, Mapping) or participant.get("in_identity_ledger"):
                continue
            participant_id = _text(participant.get("person_id"))
            events.append(
                {
                    "event_type": "person_created",
                    "payload": {
                        "person_id": participant_id,
                        "primary_name": _text(participant.get("primary_name"))
                        or _text(participant.get("display_name")),
                        "status": "reviewed",
                        "metadata": {
                            **metadata,
                            "adopted_from": "knowledge_current_person_profiles",
                        },
                    },
                    **{
                        **common,
                        "idempotency_key": f"{idempotency_key}:adopt:{participant_id}",
                    },
                }
            )
            for ordinal, alias in enumerate(_array(participant.get("candidate_names"))):
                alias_text = _text(alias)
                if alias_text and alias_text.casefold() != _text(participant.get("primary_name")).casefold():
                    events.append(
                        {
                            "event_type": "alias_added",
                            "payload": {"person_id": participant_id, "alias": alias_text},
                            **{
                                **common,
                                "idempotency_key": f"{idempotency_key}:adopt:{participant_id}:alias:{ordinal}",
                            },
                        }
                    )
        event_type = "people_merged"
        payload = {
            "target_person_id": target_person_id,
            "source_person_ids": [value for value in person_ids if value != target_person_id],
            "metadata": metadata,
        }
        events.append({"event_type": event_type, "payload": payload, **common})
    else:
        person_ids = sorted(
            _text(value) for value in _array(finding.get("person_ids")) if _text(value)
        )
        if len(person_ids) != 2:
            raise PersonIdentityRepairError(
                "Distinct-person review requires exactly two candidate people."
            )
        proposal_id = _stable_id(
            "person-reconciliation",
            repair_id,
            _text(finding.get("content_sha256")),
        )
        source_record_ids = sorted(
            {
                _text(source.get("source_record_id"))
                for participant in _array(finding.get("participants"))
                if isinstance(participant, Mapping)
                for source in _array(participant.get("evidence", {}).get("source_records"))
                if isinstance(source, Mapping) and _text(source.get("source_record_id"))
            }
        )
        events = [
            {
                "event_type": "reconciliation_proposed",
                "payload": {
                    "proposal_id": proposal_id,
                    "proposal_type": "person_name_variant",
                    "source_record_ids": source_record_ids,
                    "candidate_person_ids": person_ids,
                    "reason_codes": [
                        _text(finding.get("compatibility_basis"))
                        or "name_variant_candidate"
                    ],
                    "decision_status": "pending",
                    "metadata": metadata,
                },
                **{
                    **common,
                    "idempotency_key": f"{idempotency_key}:proposal",
                },
            },
            {
                "event_type": "reconciliation_decided",
                "payload": {
                    "proposal_id": proposal_id,
                    "decision_status": "rejected",
                    "decided_by": _text(submission.get("reviewer")),
                    "decided_at": _text(submission.get("decided_at")),
                    "metadata": metadata,
                },
                **common,
            },
        ]
    if action == "correct_name":
        events = [{"event_type": event_type, "payload": payload, **common}]
    receipts = ledger.append_events(
        events,
        rebuild=True,
    )
    receipt = receipts[-1]
    return _receipt(
        repair_id=repair_id,
        action=action,
        event_id=receipt.event_id,
        status=receipt.status,
        idempotent_replay=receipt.status == "unchanged",
    )
