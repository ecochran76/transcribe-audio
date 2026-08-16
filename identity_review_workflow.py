"""Stale-safe local review projections for Plan 0072 Identity Review and People."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import transcript_store
from conversation_knowledge_store import ConversationKnowledgeStore
from identity_learning_contracts import ARTIFACT_SCHEMAS, validate_artifact


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


class IdentityReviewWorkflow:
    """Project review work and record preview-only, stale-safe decisions."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        status = ConversationKnowledgeStore(self.root).schema_status()
        if status.schema_version != 8 or status.dirty:
            raise IdentityReviewWorkflowError(
                "Identity review requires clean conversation knowledge schema version 8."
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
        return {
            "schema_version": "transcribe-audio.identity-review-queue.v1",
            "items": [
                {
                    **_object(str(row["artifact_json"])),
                    "priority": int(row["priority"]),
                    "impact_score": float(row["impact_score"]),
                }
                for row in rows
            ],
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
    ) -> dict[str, Any]:
        if limit < 1 or limit > 200 or offset < 0:
            raise IdentityReviewWorkflowError("People pagination is outside its bounds.")
        clauses: list[str] = []
        values: list[Any] = []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if query.strip():
            clauses.append("LOWER(primary_name || ' ' || aliases_json) LIKE ?")
            values.append(f"%{query.strip().lower()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with transcript_store.connect(self.root) as con:
            total = int(
                con.execute(
                    f"SELECT COUNT(*) FROM knowledge_identity_people_projection {where}",
                    tuple(values),
                ).fetchone()[0]
            )
            people = con.execute(
                f"""
                SELECT * FROM knowledge_identity_people_projection
                {where}
                ORDER BY LOWER(primary_name), person_id
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
            items = []
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
                        "status": str(person["status"]),
                        "primary_name": str(person["primary_name"]),
                        "aliases": _array(str(person["aliases_json"])),
                        "merged_into_person_id": str(person["merged_into_person_id"]),
                        "source_records": [dict(row) for row in sources],
                        "roles": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in roles
                        ],
                        "relationships": [
                            {**dict(row), "evidence_ids": _array(str(row["evidence_ids_json"]))}
                            for row in relationships
                        ],
                        "input_watermark": str(person["input_watermark"]),
                        "built_at": str(person["built_at"]),
                    }
                )
        return {
            "schema_version": "transcribe-audio.people-projection.v1",
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {"query": query, "status": status},
            "authoritative_editing_surface": "tables_and_explicit_forms",
            "relationship_hop_limit": 2,
        }
