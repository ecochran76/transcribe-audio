"""Hash-pinned mail hypothesis projection and append-only review decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import transcript_store
from identity_learning_ledger import IdentityLearningLedger


SOURCE_SCHEMA = "transcribe-audio.mail-hypothesis-projection-source.v1"
SUBMISSION_SCHEMA = "transcribe-audio.mail-hypothesis-review-submission.v1"
RECEIPT_SCHEMA = "transcribe-audio.mail-hypothesis-review-receipt.v1"
SOURCE_FILENAME = "mail-hypothesis-source.json"
ALLOWED_ACTIONS = {"accept", "reject", "defer"}


class MailHypothesisProjectionError(ValueError):
    """Raised when the configured immutable source cannot be trusted."""


class StaleMailHypothesisReview(MailHypothesisProjectionError):
    """Raised when a decision targets an obsolete hypothesis projection."""


@dataclass(frozen=True)
class MailHypothesisProjection:
    preview_id: str
    source_content_sha256: str
    hypotheses: tuple[dict[str, Any], ...]
    artifact_root: Path

    def public_source(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "preview_id": self.preview_id,
            "content_sha256": self.source_content_sha256,
            "hypothesis_count": len(self.hypotheses),
            "authority_mode": "explicit_human_review",
        }


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MailHypothesisProjectionError(
            f"Mail hypothesis artifact is unreadable: {path.name}."
        ) from exc
    if not isinstance(value, dict):
        raise MailHypothesisProjectionError(
            f"Mail hypothesis artifact is not an object: {path.name}."
        )
    return value


def _text(value: object) -> str:
    return str(value or "").strip()


def _integer(value: object, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MailHypothesisProjectionError(
            f"Mail hypothesis {field} must be an integer."
        ) from exc


def _aggregate_hash(aggregate: Mapping[str, Any]) -> str:
    return _hash({key: value for key, value in aggregate.items() if key != "content_sha256"})


def _load_artifact_root(
    artifact_root: Path,
    *,
    expected_preview_id: str = "",
    expected_content_sha256: str = "",
) -> MailHypothesisProjection:
    aggregate = _object(artifact_root / "aggregate-validation.json")
    preview_id = _text(aggregate.get("preview_id"))
    content_sha256 = _text(aggregate.get("content_sha256"))
    if aggregate.get("schema_version") != "transcribe-audio.plan0073-p5-execution-receipt.v1":
        raise MailHypothesisProjectionError("Mail hypothesis aggregate schema is unsupported.")
    if aggregate.get("status") != "complete":
        raise MailHypothesisProjectionError("Mail hypothesis aggregate is not complete.")
    if not preview_id or not content_sha256 or _aggregate_hash(aggregate) != content_sha256:
        raise MailHypothesisProjectionError(
            "Mail hypothesis aggregate content hash does not match."
        )
    if expected_preview_id and preview_id != expected_preview_id:
        raise MailHypothesisProjectionError(
            "Mail hypothesis preview ID does not match the configured source."
        )
    if expected_content_sha256 and content_sha256 != expected_content_sha256:
        raise MailHypothesisProjectionError(
            "Mail hypothesis aggregate hash does not match the configured source."
        )
    effects = aggregate.get("effects")
    if not isinstance(effects, Mapping) or any(
        _integer(value or 0, field="effect count") for value in effects.values()
    ):
        raise MailHypothesisProjectionError("Mail hypothesis source contains forbidden effects.")
    manifest = aggregate.get("artifacts")
    entries = manifest.get("hypotheses") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise MailHypothesisProjectionError("Mail hypothesis manifest is empty.")

    hypotheses: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MailHypothesisProjectionError("Mail hypothesis manifest entry is invalid.")
        name = _text(entry.get("name"))
        expected_hash = _text(entry.get("content_sha256"))
        if not name or Path(name).name != name or not expected_hash:
            raise MailHypothesisProjectionError("Mail hypothesis manifest entry is incomplete.")
        artifact = _object(artifact_root / "hypotheses" / name)
        if _hash(artifact) != expected_hash:
            raise MailHypothesisProjectionError(
                f"Mail hypothesis artifact hash does not match: {name}."
            )
        if artifact.get("schema_version") != "transcribe-audio.plan0073-p5-shadow-hypotheses.v1":
            raise MailHypothesisProjectionError("Mail hypothesis artifact schema is unsupported.")
        conversation_id = _text(artifact.get("conversation_id"))
        rows = artifact.get("hypotheses")
        if not isinstance(rows, list):
            raise MailHypothesisProjectionError("Mail hypothesis artifact rows are invalid.")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise MailHypothesisProjectionError("Mail hypothesis row is invalid.")
            row = dict(raw)
            hypothesis_id = _text(row.get("hypothesis_id"))
            if not hypothesis_id or hypothesis_id in seen_ids:
                raise MailHypothesisProjectionError(
                    "Mail hypothesis IDs must be non-empty and unique."
                )
            if row.get("status") != "proposed":
                raise MailHypothesisProjectionError(
                    "Mail hypothesis source must remain proposed-only."
                )
            if not all(
                _text(row.get(field))
                for field in ("relationship_type", "subject_contact_id", "counterpart_id")
            ):
                raise MailHypothesisProjectionError(
                    "Mail hypothesis relationship anchors are incomplete."
                )
            seen_ids.add(hypothesis_id)
            row["originating_conversation_id"] = conversation_id
            hypotheses.append(row)
    counts = aggregate.get("counts")
    if not isinstance(counts, Mapping):
        raise MailHypothesisProjectionError("Mail hypothesis aggregate counts are invalid.")
    expected_count = _integer(
        counts.get("hypotheses") or 0,
        field="aggregate hypothesis count",
    )
    if len(hypotheses) != expected_count:
        raise MailHypothesisProjectionError("Mail hypothesis count does not match the aggregate.")
    hypotheses.sort(
        key=lambda item: (
            _text(item.get("last_observed_at")),
            _text(item.get("hypothesis_id")),
        ),
        reverse=True,
    )
    return MailHypothesisProjection(
        preview_id=preview_id,
        source_content_sha256=content_sha256,
        hypotheses=tuple(hypotheses),
        artifact_root=artifact_root,
    )


def install_mail_hypothesis_source(root: Path | None, artifact_root: Path) -> dict[str, Any]:
    projection = _load_artifact_root(Path(artifact_root).expanduser().resolve())
    store_root = transcript_store.store_dir(root)
    locator = {
        "schema_version": SOURCE_SCHEMA,
        "artifact_root": str(projection.artifact_root),
        "preview_id": projection.preview_id,
        "content_sha256": projection.source_content_sha256,
        "hypothesis_count": len(projection.hypotheses),
    }
    path = store_root / SOURCE_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_canonical(locator) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {**projection.public_source(), "locator": str(path)}


def load_mail_hypothesis_projection(root: Path | None) -> MailHypothesisProjection:
    store_root = transcript_store.store_dir(root)
    locator_path = store_root / SOURCE_FILENAME
    if not locator_path.exists():
        raise FileNotFoundError(locator_path)
    locator = _object(locator_path)
    if locator.get("schema_version") != SOURCE_SCHEMA:
        raise MailHypothesisProjectionError("Mail hypothesis source locator schema is unsupported.")
    artifact_root = Path(_text(locator.get("artifact_root"))).expanduser()
    if not artifact_root.is_absolute():
        raise MailHypothesisProjectionError(
            "Mail hypothesis source locator requires an absolute path."
        )
    projection = _load_artifact_root(
        artifact_root,
        expected_preview_id=_text(locator.get("preview_id")),
        expected_content_sha256=_text(locator.get("content_sha256")),
    )
    locator_count = _integer(
        locator.get("hypothesis_count") or -1,
        field="locator hypothesis count",
    )
    if locator_count != len(projection.hypotheses):
        raise MailHypothesisProjectionError("Mail hypothesis locator count does not match.")
    return projection


def _review_events(root: Path | None) -> tuple[dict[str, Any], ...]:
    ledger = IdentityLearningLedger(root)
    rows = ledger.events(
        event_types=(
            "reconciliation_proposed",
            "relationship_asserted",
            "relationship_corrected",
        )
    )
    events: list[dict[str, Any]] = []
    for row in rows:
        payload = row["payload"]
        if row["event_type"] == "reconciliation_proposed":
            reconciliation_metadata = payload.get("metadata")
            metadata = (
                reconciliation_metadata
                if isinstance(reconciliation_metadata, Mapping)
                else {}
            )
            hypothesis_id = _text(metadata.get("source_hypothesis_id"))
            action = _text(metadata.get("review_action"))
        elif row["event_type"] == "relationship_asserted":
            relationship_metadata = payload.get("metadata")
            metadata = relationship_metadata if isinstance(relationship_metadata, Mapping) else {}
            hypothesis_id = _text(metadata.get("source_hypothesis_id"))
            action = _text(metadata.get("review_action")) if hypothesis_id else ""
        else:
            changes = payload.get("changes")
            relationship_metadata = (
                changes.get("metadata") if isinstance(changes, Mapping) else None
            )
            metadata = relationship_metadata if isinstance(relationship_metadata, Mapping) else {}
            hypothesis_id = _text(metadata.get("source_hypothesis_id"))
            action = _text(metadata.get("review_action")) if hypothesis_id else ""
        if hypothesis_id and action in ALLOWED_ACTIONS:
            events.append(
                {
                    "event_id": row["id"],
                    "idempotency_key": row["idempotency_key"],
                    "hypothesis_id": hypothesis_id,
                    "action": action,
                    "reviewer": row["actor_id"],
                    "decided_at": row["occurred_at"],
                    "note": _text(metadata.get("review_note")),
                    "source_content_sha256": _text(metadata.get("source_content_sha256")),
                    "resulting_projection_version": _text(
                        metadata.get("resulting_projection_version")
                    ),
                }
            )
    return tuple(events)


def reviewed_mail_hypotheses(
    root: Path | None,
    projection: MailHypothesisProjection,
) -> tuple[dict[str, Any], ...]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for event in _review_events(root):
        if event["source_content_sha256"] == projection.source_content_sha256:
            by_id.setdefault(event["hypothesis_id"], []).append(event)
    rows: list[dict[str, Any]] = []
    for hypothesis in projection.hypotheses:
        history = by_id.get(_text(hypothesis.get("hypothesis_id")), [])
        current = history[-1]["action"] if history else "unreviewed"
        review_state = {
            "accept": "accepted",
            "reject": "rejected",
            "defer": "deferred",
        }.get(current, "unreviewed")
        rows.append(
            {
                **hypothesis,
                "review_state": review_state,
                "projection_version": str(len(history) + 1),
                "decision_history": history,
                "source_content_sha256": projection.source_content_sha256,
            }
        )
    return tuple(rows)


def _relationship_id(hypothesis_id: str) -> str:
    return f"mail-relationship-{hashlib.sha256(hypothesis_id.encode('utf-8')).hexdigest()[:24]}"


def record_mail_hypothesis_review(
    root: Path | None,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    if submission.get("schema_version") != SUBMISSION_SCHEMA:
        raise MailHypothesisProjectionError("Mail hypothesis review schema is unsupported.")
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
        raise MailHypothesisProjectionError("Mail hypothesis review submission is incomplete.")
    action = _text(submission.get("action"))
    if action not in ALLOWED_ACTIONS:
        raise MailHypothesisProjectionError("Mail hypothesis review action is unsupported.")
    try:
        projection = load_mail_hypothesis_projection(root)
    except FileNotFoundError as exc:
        raise MailHypothesisProjectionError(
            "Mail hypothesis review source is not configured."
        ) from exc
    if _text(submission.get("source_content_sha256")) != projection.source_content_sha256:
        raise StaleMailHypothesisReview(
            "Mail hypothesis source changed; reload Contacts before reviewing."
        )
    hypothesis_id = _text(submission.get("hypothesis_id"))
    hypothesis = next(
        (row for row in projection.hypotheses if row["hypothesis_id"] == hypothesis_id),
        None,
    )
    if hypothesis is None:
        raise MailHypothesisProjectionError("Mail hypothesis is not in the configured source.")
    current_events = [
        event
        for event in _review_events(root)
        if event["hypothesis_id"] == hypothesis_id
        and event["source_content_sha256"] == projection.source_content_sha256
    ]
    idempotency_key = _text(submission.get("idempotency_key"))
    replay = next(
        (event for event in current_events if event["idempotency_key"] == idempotency_key),
        None,
    )
    if replay is not None:
        replay_content = {
            "action": replay["action"],
            "reviewer": replay["reviewer"],
            "decided_at": replay["decided_at"],
            "note": replay["note"],
            "source_content_sha256": replay["source_content_sha256"],
        }
        submitted_content = {
            "action": action,
            "reviewer": _text(submission.get("reviewer")),
            "decided_at": _text(submission.get("decided_at")),
            "note": _text(submission.get("note")),
            "source_content_sha256": _text(
                submission.get("source_content_sha256")
            ),
        }
        if replay_content != submitted_content:
            raise MailHypothesisProjectionError(
                "Review idempotency key was reused with different content."
            )
        return {
            "schema_version": RECEIPT_SCHEMA,
            "hypothesis_id": hypothesis_id,
            "action": action,
            "projection_version": replay["resulting_projection_version"],
            "event_id": replay["event_id"],
            "idempotent_replay": True,
            "accepted_relationship_effect_count": 1 if action == "accept" else 0,
            "provider_write_count": 0,
            "speaker_assignment_count": 0,
        }
    expected_version = _text(submission.get("expected_projection_version"))
    current_version = str(len(current_events) + 1)
    if expected_version != current_version:
        raise StaleMailHypothesisReview(
            "Mail hypothesis projection is stale: "
            f"expected {expected_version}, current {current_version}."
        )
    next_version = str(int(current_version) + 1)
    ledger = IdentityLearningLedger(root)
    common_metadata = {
        "source_hypothesis_id": hypothesis_id,
        "source_preview_id": projection.preview_id,
        "source_content_sha256": projection.source_content_sha256,
        "originating_conversation_id": _text(hypothesis.get("originating_conversation_id")),
        "accepted_at": _text(submission.get("decided_at")) if action == "accept" else "",
        "review_action": action,
        "review_note": _text(submission.get("note")),
        "resulting_projection_version": next_version,
    }
    relationship_payload = {
        "relationship_id": _relationship_id(hypothesis_id),
        "relationship_type": _text(hypothesis.get("relationship_type")),
        "subject_type": "person",
        "subject_id": _text(hypothesis.get("subject_contact_id")),
        "object_type": (
            "person"
            if hypothesis.get("counterpart_type") == "contact_candidate"
            else _text(hypothesis.get("counterpart_type"))
        ),
        "object_id": _text(hypothesis.get("counterpart_id")),
        "directionality": _text(hypothesis.get("directionality")) or "directional",
        "starts_at": _text(hypothesis.get("first_observed_at")),
        "status": "reviewed",
        "evidence_ids": [
            *[
                str(value)
                for value in hypothesis.get("evidence_observation_ids") or []
            ],
            *[
                str(value)
                for value in hypothesis.get("evidence_independence_group_ids") or []
            ],
        ],
        "metadata": common_metadata,
    }
    current_action = current_events[-1]["action"] if current_events else ""
    if action == "accept":
        event_type = "relationship_asserted"
        payload = relationship_payload
    elif current_action == "accept":
        event_type = "relationship_corrected"
        payload = {
            "relationship_id": _relationship_id(hypothesis_id),
            "changes": {
                "status": "rejected" if action == "reject" else "proposed",
                "metadata": common_metadata,
            },
        }
    else:
        event_type = "reconciliation_proposed"
        payload = {
            "proposal_id": f"mail-review-{_relationship_id(hypothesis_id)}-{next_version}",
            "proposal_type": "relationship_hypothesis_review",
            "source_record_ids": [hypothesis_id],
            "candidate_person_ids": [
                _text(hypothesis.get("subject_contact_id")),
                _text(hypothesis.get("counterpart_id")),
            ],
            "reason_codes": [f"operator_{action}"],
            "decision_status": action,
            "metadata": common_metadata,
        }
    receipt = ledger.append_event(
        event_type=event_type,
        payload=payload,
        actor_id=_text(submission.get("reviewer")),
        occurred_at=_text(submission.get("decided_at")),
        idempotency_key=idempotency_key,
        subject_type="mail_hypothesis",
        subject_id=hypothesis_id,
    )
    ledger.rebuild()
    return {
        "schema_version": RECEIPT_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "action": action,
        "projection_version": next_version,
        "event_id": receipt.event_id,
        "idempotent_replay": receipt.status == "unchanged",
        "accepted_relationship_effect_count": 1 if action == "accept" else 0,
        "provider_write_count": 0,
        "speaker_assignment_count": 0,
    }
