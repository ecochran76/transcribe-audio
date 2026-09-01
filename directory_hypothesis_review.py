"""Explicit review authority for provider-derived organization and role leads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from identity_learning_ledger import IdentityLearningLedger
from relationship_role_discovery import discover_relationship_roles
import transcript_store


SUBMISSION_SCHEMA = "transcribe-audio.directory-hypothesis-review-submission.v1"
RECEIPT_SCHEMA = "transcribe-audio.directory-hypothesis-review-receipt.v1"
PROJECTION_SCHEMA = "transcribe-audio.directory-hypothesis-review-projection.v1"
ALLOWED_ACTIONS = {"accept", "reject", "defer"}
REVIEW_KINDS = {"affiliation", "contextual_role"}


class DirectoryHypothesisReviewError(ValueError):
    """Raised when a directory lead cannot be reviewed exactly."""


class StaleDirectoryHypothesisReview(DirectoryHypothesisReviewError):
    """Raised when a review no longer targets the current lead projection."""


def _text(value: object) -> str:
    return str(value or "").strip()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{_hash([prefix, *parts])[:24]}"


def _organization_id(name: str) -> str:
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:24]
    return f"organization:{digest}"


def _source_hypothesis(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in hypothesis.items()
        if key
        not in {
            "decision_history",
            "projection_version",
            "review_state",
            "source_content_sha256",
        }
    }


def _event_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return {}
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    changes = payload.get("changes")
    if isinstance(changes, Mapping) and isinstance(changes.get("metadata"), Mapping):
        return changes["metadata"]
    return {}


def _review_events(root: Path | None) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for event in IdentityLearningLedger(root).events(
        event_types=(
            "role_asserted",
            "role_corrected",
            "relationship_asserted",
            "relationship_corrected",
            "reconciliation_proposed",
        )
    ):
        metadata = _event_metadata(event)
        hypothesis_id = _text(metadata.get("source_hypothesis_id"))
        action = _text(metadata.get("review_action"))
        if not hypothesis_id or action not in ALLOWED_ACTIONS:
            continue
        events.append(
            {
                "event_id": event["id"],
                "hypothesis_id": hypothesis_id,
                "action": action,
                "reviewer": event["actor_id"],
                "decided_at": event["occurred_at"],
                "note": _text(metadata.get("review_note")),
                "source_content_sha256": _text(
                    metadata.get("source_content_sha256")
                ),
                "resulting_projection_version": _text(
                    metadata.get("resulting_projection_version")
                ),
                "idempotency_key": _text(metadata.get("review_idempotency_key")),
                "submission_content_sha256": _text(
                    metadata.get("submission_content_sha256")
                ),
                "person_id": _text(metadata.get("reviewed_person_id")),
                "organization_id": _text(
                    metadata.get("reviewed_organization_id")
                ),
                "effect_counts": dict(metadata.get("effect_counts") or {}),
            }
        )
    return tuple(events)


def project_directory_review_hypotheses(
    root: Path | None,
    discovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decorate current provider leads with immutable review state."""
    discovery = dict(discovery or discover_relationship_roles(root))
    history_by_id: dict[str, list[dict[str, Any]]] = {}
    for event in _review_events(root):
        history_by_id.setdefault(event["hypothesis_id"], []).append(event)

    by_contact_id: dict[str, dict[str, list[dict[str, Any]]]] = {}
    review_count = 0
    for contact_id, collections in discovery["by_contact_id"].items():
        projected = {"role_hypotheses": [], "relationship_hypotheses": []}
        for collection_name in projected:
            for raw in collections.get(collection_name) or []:
                hypothesis = dict(raw)
                if _text(hypothesis.get("hypothesis_kind")) not in REVIEW_KINDS:
                    projected[collection_name].append(hypothesis)
                    continue
                source = _source_hypothesis(hypothesis)
                source_hash = _hash(source)
                history = [
                    event
                    for event in history_by_id.get(
                        _text(hypothesis.get("hypothesis_id")), []
                    )
                    if event["source_content_sha256"] == source_hash
                ]
                current_action = history[-1]["action"] if history else ""
                projected[collection_name].append(
                    {
                        **source,
                        "source_content_sha256": source_hash,
                        "projection_version": str(len(history) + 1),
                        "review_state": {
                            "accept": "accepted",
                            "reject": "rejected",
                            "defer": "deferred",
                        }.get(current_action, "unreviewed"),
                        "decision_history": history,
                    }
                )
                review_count += 1
        by_contact_id[contact_id] = projected
    return {
        **{
            key: value
            for key, value in discovery.items()
            if key != "by_contact_id"
        },
        "schema_version": PROJECTION_SCHEMA,
        "by_contact_id": by_contact_id,
        "directory_review_hypothesis_count": review_count,
        "review_authority_mode": "explicit_human_review",
    }


def _target(
    submission: Mapping[str, Any],
    field: str,
) -> tuple[str, str]:
    raw = submission.get(field)
    if not isinstance(raw, Mapping):
        raise DirectoryHypothesisReviewError(
            f"Directory review accept requires an explicit {field.replace('_', ' ')}."
        )
    mode = _text(raw.get("mode"))
    if mode not in {"create", "existing"}:
        raise DirectoryHypothesisReviewError(
            f"Directory review {field.replace('_', ' ')} mode is unsupported."
        )
    target_id = _text(raw.get("id"))
    if mode == "existing" and not target_id:
        raise DirectoryHypothesisReviewError(
            f"Directory review existing {field.replace('_', ' ')} requires an id."
        )
    return mode, target_id


def _contact(root: Path | None, contact_id: str) -> dict[str, str]:
    with transcript_store.connect(root) as con:
        row = con.execute(
            "SELECT id, label, updated_at FROM contacts WHERE id = ?",
            (contact_id,),
        ).fetchone()
    if row is None:
        raise DirectoryHypothesisReviewError(
            "Directory review lead no longer has its exact local contact."
        )
    return {
        "id": str(row["id"]),
        "label": str(row["label"]),
        "updated_at": str(row["updated_at"]),
    }


def _accepted_person(root: Path | None, person_id: str) -> dict[str, Any] | None:
    with transcript_store.connect(root) as con:
        row = con.execute(
            """
            SELECT person_id, resolution_status, primary_name, aliases_json,
                   input_watermark
            FROM knowledge_current_person_profiles
            WHERE person_id = ?
            """,
            (person_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        aliases = json.loads(str(row["aliases_json"]))
    except json.JSONDecodeError:
        aliases = []
    return {
        "person_id": str(row["person_id"]),
        "status": str(row["resolution_status"]),
        "primary_name": str(row["primary_name"]),
        "aliases": [str(value) for value in aliases if _text(value)],
        "input_watermark": str(row["input_watermark"]),
    }


def _receipt(
    *,
    hypothesis_id: str,
    action: str,
    projection_version: str,
    event_id: str,
    idempotent_replay: bool,
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "action": action,
        "projection_version": projection_version,
        "event_id": event_id,
        "idempotent_replay": idempotent_replay,
        "accepted_person_effect_count": int(counts.get("person") or 0),
        "accepted_organization_effect_count": int(
            counts.get("organization") or 0
        ),
        "accepted_role_effect_count": int(counts.get("role") or 0),
        "accepted_relationship_effect_count": int(
            counts.get("relationship") or 0
        ),
        "provider_write_count": 0,
        "speaker_assignment_count": 0,
    }


def record_directory_hypothesis_review(
    root: Path | None,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    """Record one exact operator decision and rebuild accepted authority."""
    if submission.get("schema_version") != SUBMISSION_SCHEMA:
        raise DirectoryHypothesisReviewError(
            "Directory hypothesis review schema is unsupported."
        )
    required = (
        "hypothesis_id",
        "action",
        "expected_projection_version",
        "source_content_sha256",
        "reviewer",
        "decided_at",
        "idempotency_key",
    )
    if any(not _text(submission.get(field)) for field in required):
        raise DirectoryHypothesisReviewError(
            "Directory hypothesis review submission is incomplete."
        )
    action = _text(submission.get("action"))
    if action not in ALLOWED_ACTIONS:
        raise DirectoryHypothesisReviewError(
            "Directory hypothesis review action is unsupported."
        )

    projection = project_directory_review_hypotheses(root)
    leads = [
        hypothesis
        for collections in projection["by_contact_id"].values()
        for collection_name in ("role_hypotheses", "relationship_hypotheses")
        for hypothesis in collections[collection_name]
        if _text(hypothesis.get("hypothesis_kind")) in REVIEW_KINDS
    ]
    hypothesis_id = _text(submission.get("hypothesis_id"))
    hypothesis = next(
        (item for item in leads if item["hypothesis_id"] == hypothesis_id), None
    )
    if hypothesis is None:
        raise DirectoryHypothesisReviewError(
            "Directory hypothesis is not in the current provider projection."
        )
    if _text(submission.get("source_content_sha256")) != hypothesis[
        "source_content_sha256"
    ]:
        raise StaleDirectoryHypothesisReview(
            "Directory hypothesis source changed; reload Contacts before reviewing."
        )

    submission_hash = _hash(dict(submission))
    idempotency_key = _text(submission.get("idempotency_key"))
    replay = next(
        (
            event
            for event in _review_events(root)
            if event["hypothesis_id"] == hypothesis_id
            and event["idempotency_key"] == idempotency_key
        ),
        None,
    )
    if replay is not None:
        if replay["submission_content_sha256"] != submission_hash:
            raise DirectoryHypothesisReviewError(
                "Review idempotency key was reused with different content."
            )
        return _receipt(
            hypothesis_id=hypothesis_id,
            action=action,
            projection_version=replay["resulting_projection_version"],
            event_id=replay["event_id"],
            idempotent_replay=True,
            counts=replay["effect_counts"],
        )

    expected_version = _text(submission.get("expected_projection_version"))
    current_version = _text(hypothesis.get("projection_version"))
    if expected_version != current_version:
        raise StaleDirectoryHypothesisReview(
            "Directory hypothesis projection is stale: "
            f"expected {expected_version}, current {current_version}."
        )
    next_version = str(int(current_version) + 1)
    ledger = IdentityLearningLedger(root)
    snapshot = ledger.projection_snapshot()
    history = list(hypothesis.get("decision_history") or [])
    prior_accept = next(
        (event for event in reversed(history) if event["action"] == "accept"), None
    )
    contact_id = _text(hypothesis.get("subject_contact_id"))
    contact = _contact(root, contact_id)

    person_id = ""
    organization_id = ""
    events: list[dict[str, Any]] = []
    counts = {"person": 0, "organization": 0, "role": 0, "relationship": 0}
    common = {
        "actor_id": _text(submission.get("reviewer")),
        "occurred_at": _text(submission.get("decided_at")),
        "subject_type": "directory_hypothesis",
        "subject_id": hypothesis_id,
    }

    if action == "accept":
        person_mode, requested_person_id = _target(submission, "person_target")
        organization_mode, requested_organization_id = _target(
            submission, "organization_target"
        )
        person_id = (
            _stable_id("person-directory", contact_id)
            if person_mode == "create"
            else requested_person_id
        )
        organization_name = _text(
            hypothesis.get("organization") or hypothesis.get("counterpart_label")
        )
        if not organization_name:
            raise DirectoryHypothesisReviewError(
                "Directory review lead has no organization name."
            )
        organization_id = (
            _organization_id(organization_name)
            if organization_mode == "create"
            else requested_organization_id
        )
        accepted_person = (
            _accepted_person(root, person_id)
            if person_mode == "existing" and person_id not in snapshot["people"]
            else None
        )
        if (
            person_mode == "existing"
            and person_id not in snapshot["people"]
            and accepted_person is None
        ):
            raise DirectoryHypothesisReviewError(
                "Directory review selected an unknown canonical person."
            )
        if (
            organization_mode == "existing"
            and organization_id not in snapshot["organizations"]
        ):
            raise DirectoryHypothesisReviewError(
                "Directory review selected an unknown organization."
            )
        if prior_accept and (
            prior_accept["person_id"] != person_id
            or prior_accept["organization_id"] != organization_id
        ):
            raise DirectoryHypothesisReviewError(
                "Changing accepted review targets requires an explicit correction workflow."
            )
        if accepted_person is not None:
            events.append(
                {
                    **common,
                    "event_type": "person_created",
                    "idempotency_key": f"{idempotency_key}:person-adopted",
                    "payload": {
                        "person_id": person_id,
                        "primary_name": accepted_person["primary_name"],
                        "status": accepted_person["status"],
                        "metadata": {
                            "source_hypothesis_id": hypothesis_id,
                            "adopted_from": "knowledge_current_person_profiles",
                            "input_watermark": accepted_person["input_watermark"],
                        },
                    },
                }
            )
            for ordinal, alias in enumerate(accepted_person["aliases"]):
                events.append(
                    {
                        **common,
                        "event_type": "alias_added",
                        "idempotency_key": (
                            f"{idempotency_key}:person-alias:{ordinal}"
                        ),
                        "payload": {"person_id": person_id, "alias": alias},
                    }
                )
        elif person_id not in snapshot["people"]:
            events.append(
                {
                    **common,
                    "event_type": "person_created",
                    "idempotency_key": f"{idempotency_key}:person",
                    "payload": {
                        "person_id": person_id,
                        "primary_name": contact["label"],
                        "status": "reviewed",
                        "metadata": {"source_hypothesis_id": hypothesis_id},
                    },
                }
            )
            counts["person"] = 1
        source = snapshot["sources"].get(contact_id)
        if source and _text(source.get("person_id")) != person_id:
            raise DirectoryHypothesisReviewError(
                "Reviewed local contact is already linked to another person."
            )
        if source is None:
            events.append(
                {
                    **common,
                    "event_type": "source_record_observed",
                    "idempotency_key": f"{idempotency_key}:source",
                    "payload": {
                        "source_record_id": contact_id,
                        "person_id": person_id,
                        "source_profile_id": "transcribe-audio-local-contacts",
                        "provider_kind": "local",
                        "record_type": "local_contact",
                        "external_ref": f"local-contact:{contact_id}",
                        "label": contact["label"],
                        "observed_at": _text(submission.get("decided_at")),
                        "content_hash": _hash(contact),
                        "metadata": {"source_hypothesis_id": hypothesis_id},
                    },
                }
            )
        if organization_id not in snapshot["organizations"]:
            events.append(
                {
                    **common,
                    "event_type": "organization_created",
                    "idempotency_key": f"{idempotency_key}:organization",
                    "payload": {
                        "organization_id": organization_id,
                        "primary_name": organization_name,
                        "status": "reviewed",
                        "metadata": {"source_hypothesis_id": hypothesis_id},
                    },
                }
            )
            counts["organization"] = 1

    source_records = [
        _text(item.get("source_record_id"))
        for item in hypothesis.get("source_records") or []
        if isinstance(item, Mapping) and _text(item.get("source_record_id"))
    ]
    effect_metadata = {
        "source_hypothesis_id": hypothesis_id,
        "source_content_sha256": hypothesis["source_content_sha256"],
        "review_action": action,
        "review_note": _text(submission.get("note")),
        "review_idempotency_key": idempotency_key,
        "submission_content_sha256": submission_hash,
        "resulting_projection_version": next_version,
        "reviewed_person_id": person_id,
        "reviewed_organization_id": organization_id,
        "effect_counts": counts,
    }
    kind = _text(hypothesis.get("hypothesis_kind"))
    if action == "accept" and kind == "contextual_role":
        role_id = _stable_id("directory-role", hypothesis_id, person_id)
        title = _text(submission.get("role_title")) or _text(
            hypothesis.get("display_value")
        )
        if not title:
            raise DirectoryHypothesisReviewError(
                "Accepted contextual role requires a title."
            )
        counts["role"] = 1
        effect_metadata["effect_counts"] = counts
        if role_id in snapshot["roles"]:
            event_type = "role_corrected"
            payload = {
                "role_id": role_id,
                "changes": {
                    "role_type": title,
                    "organization_id": organization_id,
                    "status": "reviewed",
                    "evidence_ids": source_records,
                    "metadata": {
                        **effect_metadata,
                        "ontology_role_type": _text(hypothesis.get("role_type")),
                        "department": _text(hypothesis.get("department")),
                    },
                },
            }
        else:
            event_type = "role_asserted"
            payload = {
                "role_id": role_id,
                "person_id": person_id,
                "role_type": title,
                "organization_id": organization_id,
                "status": "reviewed",
                "evidence_ids": source_records,
                "metadata": {
                    **effect_metadata,
                    "ontology_role_type": _text(hypothesis.get("role_type")),
                    "department": _text(hypothesis.get("department")),
                },
            }
    elif action == "accept":
        relationship_id = _stable_id(
            "directory-affiliation", hypothesis_id, person_id, organization_id
        )
        counts["relationship"] = 1
        effect_metadata["effect_counts"] = counts
        if relationship_id in snapshot["relationships"]:
            event_type = "relationship_corrected"
            payload = {
                "relationship_id": relationship_id,
                "changes": {
                    "status": "reviewed",
                    "metadata": effect_metadata,
                },
            }
        else:
            event_type = "relationship_asserted"
            payload = {
                "relationship_id": relationship_id,
                "relationship_type": "AFFILIATED_WITH",
                "subject_type": "person",
                "subject_id": person_id,
                "object_type": "organization",
                "object_id": organization_id,
                "directionality": "directional",
                "status": "reviewed",
                "evidence_ids": source_records,
                "metadata": effect_metadata,
            }
    elif prior_accept:
        person_id = prior_accept["person_id"]
        organization_id = prior_accept["organization_id"]
        effect_metadata["reviewed_person_id"] = person_id
        effect_metadata["reviewed_organization_id"] = organization_id
        if kind == "contextual_role":
            event_type = "role_corrected"
            payload = {
                "role_id": _stable_id("directory-role", hypothesis_id, person_id),
                "changes": {
                    "status": "rejected" if action == "reject" else "proposed",
                    "metadata": effect_metadata,
                },
            }
        else:
            event_type = "relationship_corrected"
            payload = {
                "relationship_id": _stable_id(
                    "directory-affiliation",
                    hypothesis_id,
                    person_id,
                    organization_id,
                ),
                "changes": {
                    "status": "rejected" if action == "reject" else "proposed",
                    "metadata": effect_metadata,
                },
            }
    else:
        event_type = "reconciliation_proposed"
        payload = {
            "proposal_id": _stable_id(
                "directory-review", hypothesis_id, next_version
            ),
            "proposal_type": "directory_hypothesis_review",
            "source_record_ids": [hypothesis_id],
            "candidate_person_ids": [],
            "reason_codes": [f"operator_{action}"],
            "decision_status": action,
            "metadata": effect_metadata,
        }
    events.append(
        {
            **common,
            "event_type": event_type,
            "idempotency_key": f"{idempotency_key}:decision",
            "payload": payload,
        }
    )
    receipts = ledger.append_events(events, rebuild=True)
    return _receipt(
        hypothesis_id=hypothesis_id,
        action=action,
        projection_version=next_version,
        event_id=receipts[-1].event_id,
        idempotent_replay=all(receipt.status == "unchanged" for receipt in receipts),
        counts=counts,
    )
