"""Derived, stale-safe repair workflow for accepted organization authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from identity_learning_ledger import IdentityLearningLedger


REPAIR_QUEUE_SCHEMA = "transcribe-audio.organization-identity-repair-queue.v1"
REPAIR_SUBMISSION_SCHEMA = (
    "transcribe-audio.organization-identity-repair-submission.v1"
)
REPAIR_RECEIPT_SCHEMA = "transcribe-audio.organization-identity-repair-receipt.v1"
ALLOWED_ACTIONS = {
    "merge_organizations",
    "set_parent",
    "relate_organizations",
    "mark_distinct",
}
ALLOWED_RELATIONSHIP_TYPES = {
    "related_to",
    "predecessor_of",
    "successor_of",
}
UNIT_MARKERS = {
    "center",
    "college",
    "department",
    "division",
    "institute",
    "laboratory",
    "office",
    "program",
    "school",
}
FORMAL_INSTITUTION_SUFFIX_WORDS = {"and", "of", "science", "technology", "the"}


class OrganizationIdentityRepairError(ValueError):
    """Raised when an organization repair cannot preserve its contract."""


class StaleOrganizationIdentityRepair(OrganizationIdentityRepairError):
    """Raised when current organization authority differs from a queue row."""


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


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", _text(value).casefold()))


def _without_leading_the(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tokens[1:] if tokens[:1] == ("the",) else tokens


def _possible_alias(left: str, right: str) -> bool:
    left_tokens = _without_leading_the(_tokens(left))
    right_tokens = _without_leading_the(_tokens(right))
    if left_tokens == right_tokens:
        return True
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    remainder = longer[len(shorter) :]
    return (
        len(shorter) >= 3
        and shorter[-1:] == ("university",)
        and longer[: len(shorter)] == shorter
        and bool(remainder)
        and set(remainder) <= FORMAL_INSTITUTION_SUFFIX_WORDS
    )


def _unit_pair(left: str, right: str) -> tuple[int, int] | None:
    values = (_without_leading_the(_tokens(left)), _without_leading_the(_tokens(right)))
    for parent_index, child_index in ((0, 1), (1, 0)):
        parent = values[parent_index]
        child = values[child_index]
        if (
            len(child) > len(parent)
            and child[: len(parent)] == parent
            and child[len(parent)] in UNIT_MARKERS
        ):
            return parent_index, child_index
    return None


def _acronym(value: str) -> str:
    tokens = [token for token in _tokens(value) if token not in {"and", "of", "the"}]
    return "".join(token[0] for token in tokens if token)


def _acronym_related(left: str, right: str) -> bool:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    left_acronym = _acronym(left)
    right_acronym = _acronym(right)
    return (
        len(left_acronym) >= 3 and left_acronym in right_tokens
    ) or (
        len(right_acronym) >= 3 and right_acronym in left_tokens
    )


def _organization_participant(
    item: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    organization_id = _text(item.get("accepted_organization_id"))
    organization = authority.get(organization_id)
    return {
        "organization_id": organization_id,
        "primary_name": _text(
            organization.get("primary_name")
            if isinstance(organization, Mapping)
            else item.get("primary_name")
        ),
        "aliases": sorted(
            {_text(value) for value in _array(item.get("aliases")) if _text(value)},
            key=str.casefold,
        ),
        "organization_type": _text(
            organization.get("organization_type")
            if isinstance(organization, Mapping)
            else item.get("organization_type")
        ),
        "parent_organization_id": _text(
            organization.get("parent_organization_id")
            if isinstance(organization, Mapping)
            else item.get("parent_organization_id")
        ),
        "source_count": len(
            [
                value
                for value in _array(item.get("source_records"))
                if isinstance(value, Mapping)
            ]
        ),
    }


def _finding_content(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(value) if key != "content_sha256"}


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


def build_organization_identity_repair_queue(
    directory_payload: Mapping[str, Any],
    authority_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive organization reconciliation candidates without mutation."""

    authority = authority_snapshot.get("organizations") or {}
    if not isinstance(authority, Mapping):
        authority = {}
    items = [
        dict(value)
        for value in _array(directory_payload.get("items"))
        if isinstance(value, Mapping)
        and _text(value.get("accepted_organization_id"))
        and not _text(
            authority.get(_text(value.get("accepted_organization_id")), {}).get(
                "merged_into_organization_id"
            )
            if isinstance(
                authority.get(_text(value.get("accepted_organization_id"))), Mapping
            )
            else ""
        )
    ]
    findings: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            left_id = _text(left.get("accepted_organization_id"))
            right_id = _text(right.get("accepted_organization_id"))
            organization_ids = sorted((left_id, right_id))
            left_name = _text(left.get("primary_name"))
            right_name = _text(right.get("primary_name"))
            repair_kind = ""
            allowed_actions: list[str] = []
            suggested_parent_id = ""
            suggested_child_id = ""
            reason = ""
            if _possible_alias(left_name, right_name):
                repair_kind = "possible_alias"
                allowed_actions = ["merge_organizations", "mark_distinct"]
                reason = "The names may identify the same organization."
            elif unit_pair := _unit_pair(left_name, right_name):
                repair_kind = "unit_candidate"
                allowed_actions = ["set_parent", "relate_organizations", "mark_distinct"]
                pair_items = (left, right)
                suggested_parent_id = _text(
                    pair_items[unit_pair[0]].get("accepted_organization_id")
                )
                suggested_child_id = _text(
                    pair_items[unit_pair[1]].get("accepted_organization_id")
                )
                reason = "The longer name looks like a named organizational unit."
            elif _acronym_related(left_name, right_name):
                repair_kind = "related_candidate"
                allowed_actions = [
                    "merge_organizations",
                    "set_parent",
                    "relate_organizations",
                    "mark_distinct",
                ]
                reason = "An institutional acronym links the names but does not prove equivalence."
            if not repair_kind:
                continue
            participants = [
                _organization_participant(item, authority)
                for item in sorted(
                    (left, right),
                    key=lambda item: _text(item.get("accepted_organization_id")),
                )
            ]
            suggested_action = {
                "possible_alias": "merge_organizations",
                "unit_candidate": "set_parent",
                "related_candidate": "relate_organizations",
            }[repair_kind]
            suggested_target_id = ""
            if repair_kind == "possible_alias":
                suggested_target_id = _text(
                    min(
                        participants,
                        key=lambda participant: (
                            -int(participant.get("source_count") or 0),
                            len(_tokens(participant.get("primary_name"))),
                            _text(participant.get("primary_name")).casefold(),
                        ),
                    ).get("organization_id")
                )
            finding = {
                "repair_kind": repair_kind,
                "organization_ids": organization_ids,
                "participants": participants,
                "suggested_parent_id": suggested_parent_id,
                "suggested_child_id": suggested_child_id,
                "suggested_action": suggested_action,
                "suggested_target_organization_id": suggested_target_id,
                "reason": reason,
                "allowed_actions": allowed_actions,
                "mutation_count": 0,
            }
            finding["repair_id"] = _stable_id(
                "organization-repair", repair_kind, *organization_ids
            )
            finding["content_sha256"] = _hash(_finding_content(finding))
            if _finding_was_rejected(
                authority_snapshot, _text(finding.get("content_sha256"))
            ):
                continue
            findings.append(finding)

    order = {"possible_alias": 0, "unit_candidate": 1, "related_candidate": 2}
    findings.sort(
        key=lambda item: (
            order.get(_text(item.get("repair_kind")), 99),
            tuple(item.get("organization_ids") or []),
        )
    )
    counts = {
        "all": len(findings),
        "actionable": len(findings),
        "possible_alias": sum(
            item["repair_kind"] == "possible_alias" for item in findings
        ),
        "unit_candidate": sum(
            item["repair_kind"] == "unit_candidate" for item in findings
        ),
        "related_candidate": sum(
            item["repair_kind"] == "related_candidate" for item in findings
        ),
    }
    return {
        "schema_version": REPAIR_QUEUE_SCHEMA,
        "items": findings,
        "counts": counts,
        "mutation_count": 0,
        "default_sort": "repair_kind:asc,organization:asc",
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


def record_organization_identity_repair(
    root: Path | None,
    submission: Mapping[str, Any],
    current_queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one exact reviewed organization repair and rebuild projections."""

    if submission.get("schema_version") != REPAIR_SUBMISSION_SCHEMA:
        raise OrganizationIdentityRepairError(
            "Organization identity repair schema is unsupported."
        )
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
        raise OrganizationIdentityRepairError(
            "Organization identity repair submission is incomplete."
        )
    action = _text(submission.get("action"))
    if action not in ALLOWED_ACTIONS:
        raise OrganizationIdentityRepairError(
            "Organization identity repair action is unsupported."
        )

    repair_id = _text(submission.get("repair_id"))
    idempotency_key = _text(submission.get("idempotency_key"))
    submission_hash = _hash(dict(submission))
    ledger = IdentityLearningLedger(root)
    for event in ledger.events(
        event_types=(
            "organizations_merged",
            "organization_corrected",
            "relationship_asserted",
            "reconciliation_decided",
        )
    ):
        metadata = _event_repair_metadata(event)
        if (
            _text(metadata.get("repair_id")) == repair_id
            and _text(metadata.get("repair_idempotency_key")) == idempotency_key
        ):
            if _text(metadata.get("submission_content_sha256")) != submission_hash:
                raise OrganizationIdentityRepairError(
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
        raise StaleOrganizationIdentityRepair(
            "Organization identity repair is stale; reload Repairs before applying it."
        )
    if action not in _array(finding.get("allowed_actions")):
        raise OrganizationIdentityRepairError(
            "Repair action is not allowed for this finding."
        )

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
        "subject_type": "organization_identity_repair",
        "subject_id": repair_id,
    }
    organization_ids = sorted(
        _text(value)
        for value in _array(finding.get("organization_ids"))
        if _text(value)
    )
    if len(organization_ids) != 2:
        raise OrganizationIdentityRepairError(
            "Organization repair requires exactly two organizations."
        )
    events: list[dict[str, Any]] = []
    if action == "merge_organizations":
        target_id = _text(submission.get("target_organization_id"))
        if target_id not in organization_ids:
            raise OrganizationIdentityRepairError(
                "Merge target is not part of the finding."
            )
        source_id = next(value for value in organization_ids if value != target_id)
        participants = {
            _text(value.get("organization_id")): value
            for value in _array(finding.get("participants"))
            if isinstance(value, Mapping)
        }
        source = participants.get(source_id) or {}
        target = participants.get(target_id) or {}
        target_names = {
            _text(target.get("primary_name")).casefold(),
            *(_text(value).casefold() for value in _array(target.get("aliases"))),
        }
        aliases = [
            value
            for value in [
                _text(source.get("primary_name")),
                *(_text(alias) for alias in _array(source.get("aliases"))),
            ]
            if value and value.casefold() not in target_names
        ]
        for ordinal, alias in enumerate(dict.fromkeys(aliases)):
            events.append(
                {
                    "event_type": "organization_alias_added",
                    "payload": {"organization_id": target_id, "alias": alias},
                    **common,
                    "idempotency_key": f"{idempotency_key}:alias:{ordinal}",
                }
            )
        events.append(
            {
                "event_type": "organizations_merged",
                "payload": {
                    "target_organization_id": target_id,
                    "source_organization_ids": [source_id],
                    "metadata": metadata,
                },
                **common,
                "idempotency_key": idempotency_key,
            }
        )
    elif action == "set_parent":
        parent_id = _text(submission.get("parent_organization_id"))
        child_id = _text(submission.get("child_organization_id"))
        if sorted((parent_id, child_id)) != organization_ids or parent_id == child_id:
            raise OrganizationIdentityRepairError(
                "Parent and child organizations must match the finding."
            )
        if _text(finding.get("repair_kind")) == "unit_candidate" and (
            parent_id != _text(finding.get("suggested_parent_id"))
            or child_id != _text(finding.get("suggested_child_id"))
        ):
            raise OrganizationIdentityRepairError(
                "Parent and child organizations oppose the derived unit direction."
            )
        relationship_id = _stable_id(
            "organization-relationship", "unit_of", child_id, parent_id
        )
        events = [
            {
                "event_type": "organization_corrected",
                "payload": {
                    "organization_id": child_id,
                    "changes": {"parent_organization_id": parent_id},
                    "metadata": metadata,
                },
                **common,
                "idempotency_key": f"{idempotency_key}:parent",
            },
            {
                "event_type": "relationship_asserted",
                "payload": {
                    "relationship_id": relationship_id,
                    "relationship_type": "unit_of",
                    "subject_type": "organization",
                    "subject_id": child_id,
                    "object_type": "organization",
                    "object_id": parent_id,
                    "directionality": "directional",
                    "status": "reviewed",
                    "metadata": metadata,
                },
                **common,
                "idempotency_key": idempotency_key,
            },
        ]
    elif action == "relate_organizations":
        subject_id = _text(submission.get("subject_organization_id"))
        object_id = _text(submission.get("object_organization_id"))
        relationship_type = _text(submission.get("relationship_type"))
        if sorted((subject_id, object_id)) != organization_ids or subject_id == object_id:
            raise OrganizationIdentityRepairError(
                "Relationship endpoints must match the finding."
            )
        if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
            raise OrganizationIdentityRepairError(
                "Organization relationship type is unsupported."
            )
        relationship_id = _stable_id(
            "organization-relationship", relationship_type, subject_id, object_id
        )
        events = [
            {
                "event_type": "relationship_asserted",
                "payload": {
                    "relationship_id": relationship_id,
                    "relationship_type": relationship_type,
                    "subject_type": "organization",
                    "subject_id": subject_id,
                    "object_type": "organization",
                    "object_id": object_id,
                    "directionality": (
                        "symmetric" if relationship_type == "related_to" else "directional"
                    ),
                    "status": "reviewed",
                    "metadata": metadata,
                },
                **common,
                "idempotency_key": idempotency_key,
            }
        ]
    else:
        proposal_id = _stable_id(
            "organization-reconciliation",
            repair_id,
            _text(finding.get("content_sha256")),
        )
        reconciliation_metadata = {
            **metadata,
            "candidate_organization_ids": organization_ids,
        }
        events = [
            {
                "event_type": "reconciliation_proposed",
                "payload": {
                    "proposal_id": proposal_id,
                    "proposal_type": "organization_identity_candidate",
                    "source_record_ids": [],
                    "candidate_person_ids": [],
                    "reason_codes": [_text(finding.get("repair_kind"))],
                    "decision_status": "pending",
                    "metadata": reconciliation_metadata,
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
                    "metadata": reconciliation_metadata,
                },
                **common,
            },
        ]
    receipts = ledger.append_events(events, rebuild=True)
    receipt = receipts[-1]
    return _receipt(
        repair_id=repair_id,
        action=action,
        event_id=receipt.event_id,
        status=receipt.status,
        idempotent_replay=receipt.status == "unchanged",
    )
