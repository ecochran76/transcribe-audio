"""Stale-safe local review projections for Plan 0072 Identity Review and People."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import transcript_store
from conversation_knowledge_store import ConversationKnowledgeStore, LATEST_SCHEMA_VERSION
from identity_learning_contracts import ARTIFACT_SCHEMAS, validate_artifact
from identity_learning_ledger import IdentityLearningLedger
from directory_hypothesis_review import (
    StaleDirectoryHypothesisReview,
    project_directory_review_hypotheses,
    record_directory_hypothesis_review,
)
from mail_hypothesis_review import (
    MailHypothesisProjectionError,
    StaleMailHypothesisReview,
    load_mail_hypothesis_projection,
    record_mail_hypothesis_review,
    reviewed_mail_hypotheses,
)
from relationship_role_discovery import discover_relationship_roles
from people_organization_activity import build_directory_index


OPERATOR_GOLD_SCHEMA = "transcribe-audio.speaker-evaluation-gold.v1"
LEGACY_OPERATOR_REVIEW_METHODS = {"transcript_and_calendar"}


class IdentityReviewWorkflowError(ValueError):
    """Raised when a review projection or decision cannot remain exact."""


class StaleReviewSubmission(IdentityReviewWorkflowError):
    """Raised when optimistic concurrency detects an obsolete queue item."""


class IdempotencyConflict(IdentityReviewWorkflowError):
    """Raised when an idempotency key is reused for different content."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise IdentityReviewWorkflowError("Stored identity-review artifact is not an object.")
    return decoded


def _array(value: str) -> list[Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise IdentityReviewWorkflowError("Stored People projection value is not an array.")
    return decoded


def _version(value: Any) -> int:
    text = str(value or "").strip()
    if not text.isdigit() or int(text) < 1:
        raise IdentityReviewWorkflowError("projection_version must be a positive integer string.")
    return int(text)


def _search_text(item: Mapping[str, Any]) -> str:
    values: list[str] = [
        str(item.get("queue_item_id") or ""),
        str(item.get("conversation_id") or ""),
        str(item.get("recording_id") or ""),
        str(item.get("original_recording_filename") or ""),
        str(item.get("review_state") or ""),
    ]
    for collection in (
        item.get("calendar_candidates"),
        item.get("participant_hypotheses"),
        item.get("speakers"),
    ):
        values.append(_json(collection or []))
    return " ".join(values).lower()


def _display_title(item: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    event = payload.get("event")
    event_title = str(event.get("summary") or "").strip() if isinstance(event, Mapping) else ""
    if not event_title:
        candidates = item.get("calendar_candidates") or []
        if candidates and isinstance(candidates[0], Mapping):
            event_title = str(candidates[0].get("label") or candidates[0].get("summary") or "").strip()
    if event_title:
        return re.sub(r"^\s*\d+\s*:\s*", "", event_title).strip() or event_title
    filename = str(item.get("original_recording_filename") or "Recording").strip()
    return Path(filename).stem or "Recording"


def _review_display_metadata(
    con: sqlite3.Connection, item: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive human-facing queue metadata without mutating review artifacts."""

    conversation_id = str(item.get("conversation_id") or "")
    recording_id = str(item.get("recording_id") or "")
    documents_exist = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
    ).fetchone()
    row = None
    if documents_exist is not None:
        row = con.execute(
            """
            SELECT id, generated_at, json_payload, metadata_json
            FROM documents
            WHERE kind = 'transcript'
              AND (
                json_extract(json_payload, '$.conversation_id') = ?
                OR json_extract(json_payload, '$.recording_id') = ?
              )
            ORDER BY
              CASE WHEN json_extract(json_payload, '$.conversation_id') = ? THEN 0 ELSE 1 END,
              COALESCE(NULLIF(generated_at, ''), updated_at) DESC
            LIMIT 1
            """,
            (conversation_id, recording_id, conversation_id),
        ).fetchone()
    payload = _object(str(row["json_payload"])) if row is not None else {}
    metadata = _object(str(row["metadata_json"])) if row is not None else {}
    raw_utterances = payload.get("utterances") or []
    utterances = [value for value in raw_utterances if isinstance(value, Mapping)]
    duration_ms = max(
        (int(value.get("end") or value.get("end_ms") or 0) for value in utterances),
        default=max(
            (
                int((speaker.get("audio") or {}).get("end_ms") or 0)
                for speaker in item.get("speakers") or []
                if isinstance(speaker, Mapping) and isinstance(speaker.get("audio"), Mapping)
            ),
            default=0,
        ),
    )
    media_blob = metadata.get("media_blob") if isinstance(metadata.get("media_blob"), Mapping) else {}
    media_url = str(media_blob.get("playback_url") or "")
    if not media_url:
        media_url = next(
            (
                str((speaker.get("audio") or {}).get("media_url") or "")
                for speaker in item.get("speakers") or []
                if isinstance(speaker, Mapping) and isinstance(speaker.get("audio"), Mapping)
            ),
            "",
        )
    diarization = []
    for raw_speaker in item.get("speakers") or []:
        if not isinstance(raw_speaker, Mapping):
            continue
        speaker_ref = str(raw_speaker.get("speaker_ref") or "")
        turns = [value for value in utterances if str(value.get("speaker") or "") == speaker_ref]
        samples = []
        for turn in turns[:3]:
            samples.append(
                {
                    "start_ms": int(turn.get("start") or turn.get("start_ms") or 0),
                    "end_ms": int(turn.get("end") or turn.get("end_ms") or 0),
                    "text": str(turn.get("text") or "").strip(),
                }
            )
        diarization.append(
            {
                "speaker_ref": speaker_ref,
                "utterance_count": len(turns),
                "talk_time_ms": sum(
                    max(
                        0,
                        int(turn.get("end") or turn.get("end_ms") or 0)
                        - int(turn.get("start") or turn.get("start_ms") or 0),
                    )
                    for turn in turns
                ),
                "sample_segments": samples,
            }
        )
    event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    return {
        "source_document_id": str(row["id"]) if row is not None else "",
        "title": _display_title(item, payload),
        "event_title": str(event.get("summary") or ""),
        "recorded_at": str(row["generated_at"] or "") if row is not None else str(item.get("created_at") or ""),
        "duration_ms": duration_ms,
        "utterance_count": len(utterances),
        "media_url": media_url,
        "diarization": diarization,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _reviewed_component_label(
    component: Mapping[str, Any], people: Mapping[str, Mapping[str, Any]]
) -> str:
    person_id = str(component.get("person_ground_truth_id") or "")
    if person_id:
        person = people.get(person_id) or {}
        return str(person.get("name") or person_id)
    return str(component.get("label") or component.get("type") or "Mixed audio")


def _operator_gold_display(
    gold: Mapping[str, Any], *, expected_speakers: set[str]
) -> dict[str, Any] | None:
    people = {
        str(person.get("person_ground_truth_id") or ""): person
        for person in gold.get("people") or []
        if isinstance(person, Mapping)
        and str(person.get("person_ground_truth_id") or "")
    }
    raw_outcomes = gold.get("speaker_outcomes") or []
    if not isinstance(raw_outcomes, list):
        return None
    outcome_labels = {
        str(outcome.get("speaker_label") or "")
        for outcome in raw_outcomes
        if isinstance(outcome, Mapping)
    }
    disposition = str(gold.get("disposition") or "")
    if disposition == "eligible_known" and outcome_labels != expected_speakers:
        return None
    if disposition != "eligible_known" and raw_outcomes:
        return None

    outcomes = []
    matched_count = 0
    for raw_outcome in raw_outcomes:
        if not isinstance(raw_outcome, Mapping):
            return None
        speaker_ref = str(raw_outcome.get("speaker_label") or "")
        outcome = str(raw_outcome.get("outcome") or "")
        person_id = str(raw_outcome.get("person_ground_truth_id") or "")
        person = people.get(person_id) or {}
        components = [
            {
                "type": str(component.get("type") or ""),
                "label": _reviewed_component_label(component, people),
                "person_ground_truth_id": str(
                    component.get("person_ground_truth_id") or ""
                ),
            }
            for component in raw_outcome.get("mixed_components") or []
            if isinstance(component, Mapping)
        ]
        if outcome == "person":
            if not person_id or not person:
                return None
            matched_count += 1
            outcome_label = str(person.get("name") or person_id)
        elif outcome == "mixed":
            outcome_label = "Mixed: " + " + ".join(
                component["label"] for component in components
            ) if components else "Mixed speaker label"
        elif outcome == "unknown_to_reviewer":
            outcome_label = "Unknown to reviewer"
        elif outcome == "insufficient_transcript":
            outcome_label = "Insufficient transcript"
        else:
            return None
        outcomes.append(
            {
                "speaker_ref": speaker_ref,
                "outcome": outcome,
                "label": outcome_label,
                "person_ground_truth_id": person_id,
                "mixed_components": components,
            }
        )

    return {
        "status": "reviewed",
        "source": "operator_gold",
        "campaign_id": str(gold.get("campaign_id") or ""),
        "gold_id": str(gold.get("gold_id") or ""),
        "reviewed_at": str(gold.get("reviewed_at") or ""),
        "reviewer": str(gold.get("reviewer") or ""),
        "review_method": str(gold.get("review_method") or ""),
        "disposition": disposition,
        "matched_speaker_count": matched_count,
        "reviewed_speaker_count": len(outcomes),
        "speaker_count": len(expected_speakers),
        "speaker_outcomes": outcomes,
    }


def _operator_gold_by_document(
    gold_root: Path,
    expected_speakers: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Read latest exact operator gold for queue documents without mutating it."""

    if not expected_speakers or not gold_root.is_dir():
        return {}
    candidates: dict[str, tuple[str, dict[str, Any]]] = {}
    for index_path in sorted(gold_root.glob("campaign-*/gold/index.json")):
        campaign_dir = index_path.parent.parent
        index = _read_json_object(index_path)
        records = index.get("records")
        if not isinstance(records, list):
            continue
        latest: dict[str, Mapping[str, Any]] = {}
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                continue
            document_id = str(raw_record.get("document_id") or "")
            if document_id in expected_speakers:
                latest[document_id] = raw_record
        for document_id, record in latest.items():
            raw_path = Path(str(record.get("path") or "")).expanduser()
            try:
                gold_path = raw_path.resolve(strict=True)
                gold_path.relative_to((campaign_dir / "gold").resolve(strict=True))
            except (OSError, ValueError):
                continue
            gold = _read_json_object(gold_path)
            review_method = str(gold.get("review_method") or "")
            operator_confirmed = review_method.startswith("operator_") or (
                review_method in LEGACY_OPERATOR_REVIEW_METHODS
                and bool(str(gold.get("reviewer") or "").strip())
            )
            if (
                gold.get("schema_version") != OPERATOR_GOLD_SCHEMA
                or gold.get("prediction_visibility") != "excluded"
                or not operator_confirmed
                or str(gold.get("campaign_id") or "") != campaign_dir.name
                or str(gold.get("document_id") or "") != document_id
                or str(gold.get("gold_id") or "")
                != str(record.get("gold_id") or "")
                or str(gold.get("reviewer") or "").strip() == ""
            ):
                continue
            display = _operator_gold_display(
                gold, expected_speakers=expected_speakers[document_id]
            )
            if display is None:
                continue
            reviewed_at = str(gold.get("reviewed_at") or "")
            if document_id not in candidates or reviewed_at > candidates[document_id][0]:
                candidates[document_id] = (reviewed_at, display)
    return {document_id: display for document_id, (_date, display) in candidates.items()}


def _operator_review_people(
    con: sqlite3.Connection, gold_root: Path
) -> list[dict[str, Any]]:
    """Project reviewed speaker labels as unlinked directory records."""

    documents_exist = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
    ).fetchone()
    if documents_exist is None:
        return []
    document_rows = con.execute(
        """
        SELECT id, generated_at, json_payload
        FROM documents
        WHERE kind = 'transcript'
        ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, id
        """
    ).fetchall()
    documents: dict[str, tuple[str, Mapping[str, Any], str]] = {}
    for row in document_rows:
        payload = _object(str(row["json_payload"]))
        for key in (
            str(payload.get("conversation_id") or ""),
            str(payload.get("recording_id") or ""),
        ):
            if key and key not in documents:
                documents[key] = (str(row["id"]), payload, str(row["generated_at"] or ""))

    expected_speakers: dict[str, set[str]] = {}
    display_by_document: dict[str, dict[str, str]] = {}
    queue_rows = con.execute(
        """
        SELECT artifact_json
        FROM knowledge_identity_review_queue
        ORDER BY queue_item_id
        """
    ).fetchall()
    for row in queue_rows:
        item = _object(str(row["artifact_json"]))
        document = documents.get(str(item.get("conversation_id") or "")) or documents.get(
            str(item.get("recording_id") or "")
        )
        if document is None:
            continue
        document_id, payload, generated_at = document
        expected_speakers[document_id] = {
            str(speaker.get("speaker_ref") or "")
            for speaker in item.get("speakers") or []
            if isinstance(speaker, Mapping) and str(speaker.get("speaker_ref") or "")
        }
        display_by_document[document_id] = {
            "recording_title": _display_title(item, payload),
            "recording_filename": str(item.get("original_recording_filename") or ""),
            "recorded_at": str(payload.get("recording_start") or generated_at),
        }

    grouped: dict[str, dict[str, Any]] = {}
    for document_id, review in _operator_gold_by_document(
        gold_root, expected_speakers
    ).items():
        display = display_by_document.get(document_id) or {}
        for outcome in review.get("speaker_outcomes") or []:
            if not isinstance(outcome, Mapping) or outcome.get("outcome") != "person":
                continue
            source_identity_id = str(outcome.get("person_ground_truth_id") or "")
            name = str(outcome.get("label") or source_identity_id)
            if not source_identity_id or not name:
                continue
            record = grouped.setdefault(
                source_identity_id,
                {
                    "person_id": f"review:{source_identity_id}",
                    "source_identity_id": source_identity_id,
                    "identity_kind": "reviewed_speaker",
                    "status": "reviewed",
                    "primary_name": name,
                    "aliases": [],
                    "merged_into_person_id": "",
                    "source_records": [],
                    "roles": [],
                    "relationships": [],
                    "review_occurrences": [],
                    "possible_related_records": [],
                    "input_watermark": "",
                    "built_at": "",
                },
            )
            occurrence = {
                **display,
                "speaker_ref": str(outcome.get("speaker_ref") or ""),
                "reviewed_at": str(review.get("reviewed_at") or ""),
                "campaign_id": str(review.get("campaign_id") or ""),
                "gold_id": str(review.get("gold_id") or ""),
            }
            occurrence["source_record_id"] = (
                f"{occurrence['gold_id']}:{occurrence['speaker_ref']}"
            )
            record["review_occurrences"].append(occurrence)
            record["source_records"].append(
                {
                    "source_record_id": occurrence["source_record_id"],
                    "provider_kind": "operator_review",
                    "record_type": "speaker_identity",
                    "label": f"{occurrence['recording_filename']} · Speaker {occurrence['speaker_ref']}",
                    "external_ref": "",
                    "resolution_status": "reviewed_label",
                }
            )
            record["built_at"] = max(record["built_at"], occurrence["reviewed_at"])

    people = []
    for record in grouped.values():
        occurrences = sorted(
            record["review_occurrences"],
            key=lambda value: (
                str(value.get("recorded_at") or ""),
                str(value.get("recording_filename") or ""),
                str(value.get("speaker_ref") or ""),
            ),
            reverse=True,
        )
        record["review_occurrences"] = occurrences
        record["speaker_review_count"] = len(occurrences)
        record["recording_count"] = len(
            {
                (value.get("gold_id"), value.get("recording_filename"))
                for value in occurrences
            }
        )
        record["input_watermark"] = _hash(occurrences)
        people.append(record)
    return people


class IdentityReviewWorkflow:
    """Project review work and record preview-only, stale-safe decisions."""

    def __init__(
        self, root: Path | None = None, *, gold_root: Path | None = None
    ) -> None:
        self.root = transcript_store.store_dir(root)
        self.gold_root = (
            gold_root
            or Path("~/.local/state/transcribe-audio/speaker-evaluation-campaigns")
        ).expanduser()
        status = ConversationKnowledgeStore(self.root).schema_status()
        if status.schema_version != LATEST_SCHEMA_VERSION or status.dirty:
            raise IdentityReviewWorkflowError(
                "Identity review requires clean conversation knowledge schema "
                f"version {LATEST_SCHEMA_VERSION}."
            )

    def project_queue_item(
        self,
        payload: Mapping[str, Any],
        *,
        priority: int = 0,
        impact_score: float = 0.0,
    ) -> str:
        """Insert or replace one derived queue item without changing source truth."""
        item = validate_artifact("identity_review_queue_item", payload)
        version = _version(item["projection_version"])
        if not isinstance(priority, int) or priority < 0 or priority > 100:
            raise IdentityReviewWorkflowError("Queue priority must be an integer from 0 to 100.")
        if not isinstance(impact_score, (int, float)) or not 0 <= float(impact_score) <= 1:
            raise IdentityReviewWorkflowError("Queue impact_score must be from 0 to 1.")
        content_hash = _hash(item)
        now = str(item["created_at"])
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT projection_version, content_hash FROM knowledge_identity_review_queue WHERE queue_item_id = ?",
                (item["queue_item_id"],),
            ).fetchone()
            if row is not None:
                current_version = int(row["projection_version"])
                if version < current_version:
                    raise StaleReviewSubmission(
                        f"Queue projection version {version} is older than current version {current_version}."
                    )
                if version == current_version and str(row["content_hash"]) != content_hash:
                    raise StaleReviewSubmission(
                        "Queue projection content changed without advancing projection_version."
                    )
            con.execute(
                """
                INSERT INTO knowledge_identity_review_queue (
                  queue_item_id, conversation_id, recording_id,
                  original_recording_filename, review_state, projection_version,
                  priority, impact_score, search_text, artifact_json, content_hash,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(queue_item_id) DO UPDATE SET
                  conversation_id = excluded.conversation_id,
                  recording_id = excluded.recording_id,
                  original_recording_filename = excluded.original_recording_filename,
                  review_state = excluded.review_state,
                  projection_version = excluded.projection_version,
                  priority = excluded.priority,
                  impact_score = excluded.impact_score,
                  search_text = excluded.search_text,
                  artifact_json = excluded.artifact_json,
                  content_hash = excluded.content_hash,
                  updated_at = excluded.updated_at
                """,
                (
                    item["queue_item_id"],
                    item["conversation_id"],
                    item["recording_id"],
                    item["original_recording_filename"],
                    item["review_state"],
                    version,
                    priority,
                    float(impact_score),
                    _search_text(item),
                    _json(item),
                    content_hash,
                    now,
                    now,
                ),
            )
            con.commit()
        return "unchanged" if row is not None and str(row["content_hash"]) == content_hash else "projected"

    def get_queue_item(self, queue_item_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT artifact_json, priority, impact_score FROM knowledge_identity_review_queue WHERE queue_item_id = ?",
                (queue_item_id,),
            ).fetchone()
        if row is None:
            raise IdentityReviewWorkflowError(f"Unknown identity review queue item: {queue_item_id}.")
        return {
            **_object(str(row["artifact_json"])),
            "priority": int(row["priority"]),
            "impact_score": float(row["impact_score"]),
        }

    def list_queue(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        state: str = "",
        query: str = "",
    ) -> dict[str, Any]:
        if limit < 1 or limit > 200 or offset < 0:
            raise IdentityReviewWorkflowError("Queue pagination is outside its bounds.")
        clauses: list[str] = []
        values: list[Any] = []
        if state:
            clauses.append("review_state = ?")
            values.append(state)
        if query.strip():
            clauses.append("search_text LIKE ?")
            values.append(f"%{query.strip().lower()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with transcript_store.connect(self.root) as con:
            total = int(
                con.execute(
                    f"SELECT COUNT(*) FROM knowledge_identity_review_queue {where}",
                    tuple(values),
                ).fetchone()[0]
            )
            rows = con.execute(
                f"""
                SELECT artifact_json, priority, impact_score
                FROM knowledge_identity_review_queue
                {where}
                ORDER BY priority DESC, impact_score DESC, created_at, queue_item_id
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
            items = [
                {
                    **_object(str(row["artifact_json"])),
                    "priority": int(row["priority"]),
                    "impact_score": float(row["impact_score"]),
                }
                for row in rows
            ]
            for item in items:
                item["display"] = _review_display_metadata(con, item)
            expected_speakers = {
                str(item["display"].get("source_document_id") or ""): {
                    str(speaker.get("speaker_ref") or "")
                    for speaker in item.get("speakers") or []
                    if isinstance(speaker, Mapping)
                    and str(speaker.get("speaker_ref") or "")
                }
                for item in items
                if str(item["display"].get("source_document_id") or "")
            }
            operator_gold = _operator_gold_by_document(
                self.gold_root, expected_speakers
            )
            for item in items:
                document_id = str(
                    item["display"].get("source_document_id") or ""
                )
                item["display"]["operator_review"] = operator_gold.get(
                    document_id
                )
        return {
            "schema_version": "transcribe-audio.identity-review-queue.v1",
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {"state": state, "query": query},
        }

    def preview_submission(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        submission = validate_artifact("identity_review_submission", payload)
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT artifact_json, projection_version FROM knowledge_identity_review_queue WHERE queue_item_id = ?",
                (submission["queue_item_id"],),
            ).fetchone()
        if row is None:
            raise IdentityReviewWorkflowError(
                f"Unknown identity review queue item: {submission['queue_item_id']}."
            )
        item = _object(str(row["artifact_json"]))
        current_version = str(row["projection_version"])
        expected_version = str(submission["expected_projection_version"])
        if submission["conversation_id"] != item["conversation_id"]:
            raise IdentityReviewWorkflowError("Submission conversation does not match its queue item.")
        if expected_version != current_version:
            raise StaleReviewSubmission(
                f"Submission expected projection version {expected_version}, but current version is {current_version}."
            )
        submission_hash = _hash(submission)
        action = str(submission["action"])
        decision = dict(submission["decision_payload"])
        proposed_effects = [
            {
                "effect_type": "speaker_identity_decision",
                "action": action,
                "proposal_id": submission["proposal_id"],
                "speaker_ref": decision.get("speaker_ref") or "",
                "person_id": decision.get("person_id") or "",
                "scope": "local_review_projection_only",
            }
        ]
        if action in {"correct_role", "correct_relationship", "merge_people", "split_person"}:
            proposed_effects.append(
                {
                    "effect_type": "people_projection_proposal",
                    "action": action,
                    "scope": "local_review_projection_only",
                }
            )
        preview = {
            "schema_version": ARTIFACT_SCHEMAS["effect_preview"],
            "preview_id": f"preview-{submission_hash[:24]}",
            "queue_item_id": submission["queue_item_id"],
            "submission_id": submission["submission_id"],
            "expected_projection_version": expected_version,
            "effect_mode": "preview_only",
            "proposed_effects": proposed_effects,
            "invalidations": [
                {"proposal_id": submission["proposal_id"], "reason": "reviewed_decision_supersedes_proposal"}
            ],
            "profile_rebuilds": [],
            "provider_write_count": 0,
            "raw_deletion_count": 0,
            "warnings": ["No identity, contact, relationship, profile, provider, or deletion effect is applied by A5."],
            "created_at": submission["decided_at"],
        }
        return validate_artifact("effect_preview", preview)

    def record_submission(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        submission = validate_artifact("identity_review_submission", payload)
        submission_hash = _hash(submission)
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                """
                SELECT submission_id, content_hash, result_projection_version
                FROM knowledge_identity_review_submissions
                WHERE idempotency_key = ?
                """,
                (submission["idempotency_key"],),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != submission_hash:
                    con.rollback()
                    raise IdempotencyConflict(
                        "Identity review idempotency key was reused for different content."
                    )
                preview_row = con.execute(
                    "SELECT artifact_json FROM knowledge_identity_review_effect_previews WHERE submission_id = ?",
                    (existing["submission_id"],),
                ).fetchone()
                con.commit()
                return {
                    "schema_version": "transcribe-audio.identity-review-decision-receipt.v1",
                    "submission_id": str(existing["submission_id"]),
                    "queue_item_id": submission["queue_item_id"],
                    "projection_version": str(existing["result_projection_version"]),
                    "effect_preview": _object(str(preview_row["artifact_json"])),
                    "idempotent_replay": True,
                    "accepted_identity_effect_count": 0,
                    "provider_write_count": 0,
                }
            queue_row = con.execute(
                "SELECT artifact_json, projection_version, priority, impact_score FROM knowledge_identity_review_queue WHERE queue_item_id = ?",
                (submission["queue_item_id"],),
            ).fetchone()
            if queue_row is None:
                con.rollback()
                raise IdentityReviewWorkflowError(
                    f"Unknown identity review queue item: {submission['queue_item_id']}."
                )
            item = _object(str(queue_row["artifact_json"]))
            current_version = str(queue_row["projection_version"])
            expected_version = str(submission["expected_projection_version"])
            if submission["conversation_id"] != item["conversation_id"]:
                con.rollback()
                raise IdentityReviewWorkflowError("Submission conversation does not match its queue item.")
            if expected_version != current_version:
                con.rollback()
                raise StaleReviewSubmission(
                    f"Submission expected projection version {expected_version}, but current version is {current_version}."
                )
            preview = self.preview_submission(submission)
            next_version = int(current_version) + 1
            con.execute(
                """
                INSERT INTO knowledge_identity_review_submissions (
                  submission_id, queue_item_id, conversation_id, proposal_id, action,
                  expected_projection_version, result_projection_version,
                  idempotency_key, artifact_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission["submission_id"], submission["queue_item_id"],
                    submission["conversation_id"], submission["proposal_id"],
                    submission["action"], int(expected_version), next_version,
                    submission["idempotency_key"], _json(submission), submission_hash,
                    submission["decided_at"],
                ),
            )
            con.execute(
                """
                INSERT INTO knowledge_identity_review_effect_previews (
                  preview_id, queue_item_id, submission_id, expected_projection_version,
                  artifact_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preview["preview_id"], preview["queue_item_id"],
                    preview["submission_id"], int(expected_version), _json(preview),
                    _hash(preview), preview["created_at"],
                ),
            )
            history = list(item.get("decision_history") or [])
            history.append(
                {
                    "submission_id": submission["submission_id"],
                    "proposal_id": submission["proposal_id"],
                    "action": submission["action"],
                    "reviewer": submission["reviewer"],
                    "decided_at": submission["decided_at"],
                    "comment": submission["comment"],
                }
            )
            item = {
                **item,
                "review_state": (
                    "unresolved"
                    if submission["action"] in {"unresolved", "defer"}
                    else "reviewed"
                ),
                "decision_history": history,
                "effect_preview_ref": preview["preview_id"],
                "projection_version": str(next_version),
            }
            validate_artifact("identity_review_queue_item", item)
            con.execute(
                """
                UPDATE knowledge_identity_review_queue
                SET review_state = ?, projection_version = ?, search_text = ?,
                    artifact_json = ?, content_hash = ?, updated_at = ?
                WHERE queue_item_id = ? AND projection_version = ?
                """,
                (
                    item["review_state"], next_version, _search_text(item), _json(item),
                    _hash(item), submission["decided_at"], item["queue_item_id"],
                    int(expected_version),
                ),
            )
            con.commit()
        return {
            "schema_version": "transcribe-audio.identity-review-decision-receipt.v1",
            "submission_id": submission["submission_id"],
            "queue_item_id": submission["queue_item_id"],
            "projection_version": str(next_version),
            "effect_preview": preview,
            "idempotent_replay": False,
            "accepted_identity_effect_count": 0,
            "provider_write_count": 0,
        }

    def list_people(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        status: str = "",
        kind: str = "",
    ) -> dict[str, Any]:
        if limit < 1 or limit > 500 or offset < 0:
            raise IdentityReviewWorkflowError("People pagination is outside its bounds.")
        allowed_kinds = {"", "canonical_person", "local_contact", "reviewed_speaker"}
        if kind not in allowed_kinds:
            raise IdentityReviewWorkflowError("Unsupported Contacts record type.")
        try:
            mail_projection = load_mail_hypothesis_projection(self.root)
            projected_mail = reviewed_mail_hypotheses(self.root, mail_projection)
            mail_source = mail_projection.public_source()
        except FileNotFoundError:
            projected_mail = ()
            mail_source = {"status": "not_configured"}
        except MailHypothesisProjectionError:
            projected_mail = ()
            mail_source = {
                "status": "invalid",
                "reason_code": "configured_mail_hypothesis_source_failed_validation",
            }
        graph_discovery = discover_relationship_roles(
            self.root,
            projected_mail_hypotheses=projected_mail,
            mail_source=mail_source,
        )
        graph_discovery = project_directory_review_hypotheses(
            self.root, graph_discovery
        )
        graph_by_contact = graph_discovery["by_contact_id"]
        with transcript_store.connect(self.root) as con:
            people = con.execute(
                """
                SELECT * FROM knowledge_identity_people_projection
                ORDER BY LOWER(primary_name), person_id
                """
            ).fetchall()
            items: list[dict[str, Any]] = []
            for person in people:
                person_id = str(person["person_id"])
                sources = con.execute(
                    "SELECT * FROM knowledge_identity_source_projection WHERE person_id = ? ORDER BY provider_kind, label, source_record_id",
                    (person_id,),
                ).fetchall()
                roles = con.execute(
                    "SELECT * FROM knowledge_identity_role_projection WHERE person_id = ? ORDER BY role_type, role_id",
                    (person_id,),
                ).fetchall()
                relationships = con.execute(
                    """
                    SELECT * FROM knowledge_identity_relationship_projection
                    WHERE (subject_type = 'person' AND subject_id = ?)
                       OR (object_type = 'person' AND object_id = ?)
                    ORDER BY relationship_type, relationship_id
                    """,
                    (person_id, person_id),
                ).fetchall()
                items.append(
                    {
                        "person_id": person_id,
                        "identity_kind": "canonical_person",
                        "status": str(person["status"]),
                        "primary_name": str(person["primary_name"]),
                        "aliases": _array(str(person["aliases_json"])),
                        "merged_into_person_id": str(person["merged_into_person_id"] or ""),
                        "source_records": [dict(row) for row in sources],
                        "roles": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in roles
                        ],
                        "relationships": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in relationships
                        ],
                        "role_hypotheses": [],
                        "relationship_hypotheses": [],
                        "review_occurrences": [],
                        "speaker_review_count": 0,
                        "recording_count": 0,
                        "possible_related_records": [],
                        "input_watermark": str(person["input_watermark"]),
                        "built_at": str(person["built_at"]),
                    }
                )

            existing_person_ids = {item["person_id"] for item in items}
            profiles = con.execute(
                """
                SELECT * FROM knowledge_current_person_profiles
                ORDER BY LOWER(primary_name), person_id
                """
            ).fetchall()
            for profile in profiles:
                person_id = str(profile["person_id"])
                if person_id in existing_person_ids:
                    continue
                source_rows = con.execute(
                    """
                    SELECT * FROM knowledge_source_records
                    WHERE person_id = ?
                    ORDER BY provider_kind, label, id
                    """,
                    (person_id,),
                ).fetchall()
                sources = []
                for source in source_rows:
                    metadata = _object(str(source["metadata_json"]))
                    sources.append(
                        {
                            "source_record_id": str(source["id"]),
                            "source_profile_id": str(source["source_profile_id"]),
                            "provider_kind": str(source["provider_kind"]),
                            "record_type": str(
                                metadata.get("record_type")
                                or source["relationship_scope"]
                                or "source_record"
                            ),
                            "external_ref": str(source["external_ref"]),
                            "label": str(source["label"]),
                            "resolution_status": str(profile["resolution_status"]),
                        }
                    )
                roles = con.execute(
                    "SELECT * FROM knowledge_identity_role_projection WHERE person_id = ? ORDER BY role_type, role_id",
                    (person_id,),
                ).fetchall()
                relationships = con.execute(
                    """
                    SELECT * FROM knowledge_identity_relationship_projection
                    WHERE (subject_type = 'person' AND subject_id = ?)
                       OR (object_type = 'person' AND object_id = ?)
                    ORDER BY relationship_type, relationship_id
                    """,
                    (person_id, person_id),
                ).fetchall()
                items.append(
                    {
                        "person_id": person_id,
                        "identity_kind": "canonical_person",
                        "status": str(profile["resolution_status"]),
                        "primary_name": str(profile["primary_name"]),
                        "aliases": _array(str(profile["aliases_json"])),
                        "merged_into_person_id": "",
                        "source_records": sources,
                        "roles": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in roles
                        ],
                        "relationships": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in relationships
                        ],
                        "role_hypotheses": [],
                        "relationship_hypotheses": [],
                        "review_occurrences": [],
                        "speaker_review_count": 0,
                        "recording_count": 0,
                        "possible_related_records": [],
                        "input_watermark": str(profile["input_watermark"]),
                        "built_at": str(profile["built_at"]),
                    }
                )

            contacts_exist = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'contacts'"
            ).fetchone()
            if contacts_exist is not None:
                contacts = con.execute(
                    "SELECT * FROM contacts ORDER BY LOWER(label), id"
                ).fetchall()
                for contact in contacts:
                    metadata = _object(str(contact["metadata_json"]))
                    email = str(contact["email"] or "")
                    external_ref = str(contact["external_ref"] or "")
                    contact_id = str(contact["id"])
                    calendar = (
                        metadata.get("calendar_attendee")
                        if isinstance(metadata.get("calendar_attendee"), dict)
                        else {}
                    )
                    enrichment = (
                        metadata.get("enrichment")
                        if isinstance(metadata.get("enrichment"), dict)
                        else {}
                    )
                    contact_status = str(metadata.get("resolution_status") or "provisional")
                    source_resolution_status = contact_status if calendar else "unlinked"
                    source_records = [
                        {
                            "source_record_id": contact_id,
                            "provider_kind": str(metadata.get("source") or "local"),
                            "record_type": "calendar_attendee_contact"
                            if calendar
                            else "local_contact",
                            "external_ref": email or external_ref,
                            "label": str(contact["label"]),
                            "resolution_status": source_resolution_status,
                        }
                    ]
                    for source in enrichment.get("source_records") or []:
                        if not isinstance(source, dict):
                            continue
                        source_records.append(
                            {
                                "source_record_id": str(
                                    source.get("source_record_id")
                                    or _hash(source)
                                ),
                                "provider_kind": str(source.get("provider") or "configured_source"),
                                "record_type": str(source.get("record_type") or "contact"),
                                "external_ref": email,
                                "label": str(source.get("label") or contact["label"]),
                                "resolution_status": "exact_email_observation",
                            }
                        )
                    contact_methods = []
                    if email:
                        contact_methods.append({"kind": "email", "value": email})
                    contact_methods.extend(
                        {"kind": "phone", "value": str(phone)}
                        for phone in enrichment.get("phones") or []
                        if str(phone).strip()
                    )
                    appearances = [
                        dict(value)
                        for value in calendar.get("appearances") or []
                        if isinstance(value, dict)
                    ]
                    person_anchor_id = f"contact:{contact_id}"
                    accepted_relationship_rows = con.execute(
                        """
                        SELECT * FROM knowledge_identity_relationship_projection
                        WHERE (subject_type = 'person' AND subject_id = ?)
                           OR (object_type = 'person' AND object_id = ?)
                        ORDER BY relationship_type, relationship_id
                        """,
                        (person_anchor_id, person_anchor_id),
                    ).fetchall()
                    graph_candidates = graph_by_contact.get(
                        contact_id,
                        {"role_hypotheses": [], "relationship_hypotheses": []},
                    )
                    items.append(
                        {
                            "person_id": f"contact:{contact_id}",
                            "source_identity_id": contact_id,
                            "identity_kind": "local_contact",
                            "status": contact_status,
                            "primary_name": str(contact["label"]),
                            "aliases": [
                                str(value)
                                for value in calendar.get("aliases") or []
                                if str(value).strip()
                                and str(value).strip().casefold()
                                != str(contact["label"]).strip().casefold()
                            ],
                            "merged_into_person_id": "",
                            "source_records": source_records,
                            "contact_methods": contact_methods,
                            "contact_class": str(metadata.get("contact_class") or "local_contact"),
                            "identity_boundary": str(metadata.get("identity_boundary") or ""),
                            "organizations": [
                                str(value)
                                for value in enrichment.get("organizations") or []
                                if str(value).strip()
                            ],
                            "calendar_occurrences": appearances,
                            "roles": [],
                            "relationships": [
                                {
                                    **dict(row),
                                    "evidence_ids": _array(str(row["evidence_ids_json"])),
                                }
                                for row in accepted_relationship_rows
                            ],
                            "role_hypotheses": graph_candidates["role_hypotheses"],
                            "relationship_hypotheses": graph_candidates[
                                "relationship_hypotheses"
                            ],
                            "review_occurrences": [],
                            "speaker_review_count": 0,
                            "recording_count": int(calendar.get("recording_count") or 0),
                            "attendee_occurrence_count": int(calendar.get("occurrence_count") or 0),
                            "possible_related_records": [],
                            "input_watermark": _hash(
                                {
                                    "id": contact_id,
                                    "label": str(contact["label"]),
                                    "email": email,
                                    "external_ref": external_ref,
                                    "metadata": metadata,
                                }
                            ),
                            "built_at": str(contact["updated_at"]),
                        }
                    )

            items.extend(_operator_review_people(con, self.gold_root))

        name_groups: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            normalized_name = str(item.get("primary_name") or "").strip().casefold()
            if normalized_name:
                name_groups.setdefault(normalized_name, []).append(item)
        for group in name_groups.values():
            if len(group) < 2:
                continue
            for item in group:
                item["possible_related_records"] = [
                    {
                        "person_id": other["person_id"],
                        "identity_kind": other["identity_kind"],
                        "primary_name": other["primary_name"],
                        "status": other["status"],
                        "reason_code": "exact_display_name_requires_review",
                    }
                    for other in group
                    if other is not item
                ]

        counts = {
            identity_kind: sum(
                1 for item in items if item["identity_kind"] == identity_kind
            )
            for identity_kind in ("canonical_person", "local_contact", "reviewed_speaker")
        }
        query_text = query.strip().casefold()
        filtered = [
            item
            for item in items
            if (not status or item["status"] == status)
            and (not kind or item["identity_kind"] == kind)
            and (
                not query_text
                or query_text
                in " ".join(
                    [
                        str(item.get("primary_name") or ""),
                        *[str(value) for value in item.get("aliases") or []],
                        _json(item.get("source_records") or []),
                        _json(item.get("review_occurrences") or []),
                        _json(item.get("role_hypotheses") or []),
                        _json(item.get("relationship_hypotheses") or []),
                    ]
                ).casefold()
            )
        ]
        kind_rank = {"canonical_person": 0, "local_contact": 1, "reviewed_speaker": 2}
        filtered.sort(
            key=lambda item: (
                str(item.get("primary_name") or "").casefold(),
                kind_rank.get(str(item.get("identity_kind") or ""), 9),
                str(item.get("person_id") or ""),
            )
        )
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return {
            "schema_version": "transcribe-audio.people-projection.v1",
            "items": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {"query": query, "status": status, "kind": kind},
            "counts": counts,
            "projection_sources": [
                "identity_ledger",
                "current_person_profiles",
                "local_contacts",
                "calendar_attendees",
                "configured_exact_email_sources",
                "shadow_role_relationship_discovery",
                "operator_gold",
            ],
            "authoritative_editing_surface": "explicit_relationship_review",
            "relationship_hop_limit": 2,
            "graph_discovery": {
                key: value
                for key, value in graph_discovery.items()
                if key not in {"by_contact_id", "mail_hypotheses"}
            },
        }

    def list_directory_index(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        query: str = "",
        view: str = "people",
    ) -> dict[str, Any]:
        """Return the canonical people/organization activity directory."""
        if limit < 1 or limit > 500 or offset < 0:
            raise IdentityReviewWorkflowError(
                "Directory pagination is outside its bounds."
            )
        if view not in {"people", "organizations", "unresolved"}:
            raise IdentityReviewWorkflowError("Unsupported directory view.")
        source = self.list_people(limit=500)
        items = list(source["items"])
        while len(items) < int(source["total"]):
            page = self.list_people(limit=500, offset=len(items))
            if not page["items"]:
                break
            items.extend(page["items"])
        source = {**source, "items": items, "limit": len(items), "offset": 0}
        index = build_directory_index(
            source,
            authority_snapshot=IdentityLearningLedger(self.root).projection_snapshot(),
        )
        candidates = (
            index["organizations"]
            if view == "organizations"
            else [
                item
                for item in index["people"]
                if view != "unresolved"
                or item["entity_kind"] == "unresolved_group"
            ]
        )
        query_text = query.strip().casefold()
        filtered = [
            item
            for item in candidates
            if not query_text
            or query_text
            in " ".join(
                [
                    _json(item.get("primary_name") or ""),
                    _json(item.get("aliases") or []),
                    _json(item.get("organizations") or []),
                    _json(item.get("source_records") or []),
                ]
            ).casefold()
        ]
        return {
            **index,
            "items": filtered[offset : offset + limit],
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "view": view,
            "query": query,
            "default_sort": "last_interaction_at:desc",
        }

    def record_mail_hypothesis_review(
        self,
        submission: Mapping[str, Any],
    ) -> dict[str, Any]:
        return record_mail_hypothesis_review(self.root, submission)

    def record_directory_hypothesis_review(
        self,
        submission: Mapping[str, Any],
    ) -> dict[str, Any]:
        return record_directory_hypothesis_review(self.root, submission)
