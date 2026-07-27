from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import transcript_store
from conversation_knowledge_store import (
    ConversationKnowledgeStore,
    ObservationRecord,
)


PROJECTION_NAME = "reviewed-affinity-profiles"
PROJECTION_SCHEMA_VERSION = "transcribe-audio.reviewed-affinity.v1"
PROFILE_SCHEMA_VERSION = 3
_IDENTITY_NAMESPACE = UUID("c57a4299-f81a-4414-89ee-e1a18261b51a")
_SPLIT_FLAGS = {
    "possible_diarization_split",
    "split_speaker",
    "same_person_multiple_speakers",
}
_MIXED_FLAGS = {
    "mixed_speaker_label",
    "mixed_diarization_label",
    "possible_mixed_speaker",
}


@dataclass(frozen=True)
class ObservationAppendReceipt:
    status: str
    conversation_id: str
    observation_count: int
    observation_types: tuple[str, ...]


@dataclass(frozen=True)
class CurrentPersonProfile:
    person_id: str
    resolution_status: str
    primary_name: str
    aliases: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    input_watermark: str
    built_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AffinityProfile:
    affinity_id: str
    subject_type: str
    subject_id: str
    affinity_type: str
    object_type: str
    object_id: str
    normalized_value: str
    display_value: str
    support_count: int
    independent_interaction_count: int
    first_observed_at: str
    last_observed_at: str
    review_state: str
    observation_ids: tuple[str, ...]
    input_watermark: str
    built_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileBuildReceipt:
    status: str
    input_watermark: str
    observation_count: int
    person_profile_count: int
    affinity_profile_count: int


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value or "{}")
    return loaded if isinstance(loaded, dict) else {}


def _json_list(value: str) -> list[Any]:
    loaded = json.loads(value or "[]")
    return loaded if isinstance(loaded, list) else []


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _stable_uuid(*parts: str) -> str:
    return str(uuid5(_IDENTITY_NAMESPACE, "\x1f".join(parts)))


def _proposal_person_id(proposal: dict[str, Any]) -> str:
    direct = str(proposal.get("person_id") or "")
    if direct:
        return direct
    identity = proposal.get("identity")
    if isinstance(identity, dict):
        return str(identity.get("person_id") or "")
    suggested = proposal.get("suggested_person")
    if isinstance(suggested, dict):
        return str(suggested.get("person_id") or "")
    return ""


def _proposal_values(
    proposal: dict[str, Any],
    key: str,
) -> tuple[str, ...]:
    value = proposal.get(key)
    if isinstance(value, list):
        return tuple(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    text = str(value or "").strip()
    return (text,) if text else ()


class ConversationProfileProjector:
    """Append reviewed observations and rebuild replaceable current profiles."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = transcript_store.store_dir(root)
        self.store = ConversationKnowledgeStore(self.root)
        status = self.store.schema_status()
        if status.schema_version < PROFILE_SCHEMA_VERSION or status.dirty:
            raise RuntimeError(
                "Conversation profile schema version 3 is not initialized."
            )

    def append_reviewed_observations(
        self,
        conversation_id: str,
    ) -> ObservationAppendReceipt:
        """Project immutable review, diarization, and source-affinity outcomes."""
        history = self.store.load_processing_history(conversation_id)
        if history is None:
            raise ValueError("Conversation has no processing history.")
        snapshot = self.store.load_conversation_snapshot(conversation_id)
        conversation_time = (
            snapshot.conversation.starts_at if snapshot is not None else ""
        )
        proposals: dict[tuple[str, str], dict[str, Any]] = {}
        conversation_observations: list[ObservationRecord] = []
        for evaluation in history.evaluations:
            for proposal in evaluation.payload.get("proposals", []):
                if not isinstance(proposal, dict):
                    continue
                proposal_id = str(proposal.get("proposal_id") or "")
                if not proposal_id:
                    continue
                proposals[(evaluation.evaluation_id, proposal_id)] = proposal
                conversation_observations.extend(
                    self._diarization_observations(
                        conversation_id=conversation_id,
                        evaluation_id=evaluation.evaluation_id,
                        evaluated_at=(
                            evaluation.created_at
                            or str(
                                evaluation.payload.get("evaluated_at")
                                or evaluation.payload.get("created_at")
                                or ""
                            )
                            or conversation_time
                        ),
                        proposal=proposal,
                    )
                )
        superseded_ids = {
            decision.supersedes_decision_id
            for decision in history.review_decisions
            if decision.supersedes_decision_id
        }
        decisions = {
            item.decision_id: item for item in history.review_decisions
        }
        for decision in history.review_decisions:
            proposal = proposals.get(
                (decision.evaluation_id, decision.proposal_id),
                {},
            )
            person_id = _proposal_person_id(proposal)
            payload = {
                "decision_id": decision.decision_id,
                "evaluation_id": decision.evaluation_id,
                "proposal_id": decision.proposal_id,
                "action": decision.action,
                "reviewer": decision.reviewer,
                "method": decision.method,
                "note": decision.note,
                "person_id": person_id,
                "proposal": proposal,
            }
            review_state = (
                "superseded"
                if decision.decision_id in superseded_ids
                else decision.action
            )
            conversation_observations.append(
                self._observation(
                    observation_key=(
                        "review-action",
                        decision.decision_id,
                        decision.action,
                    ),
                    observation_type={
                        "confirm": "speaker_identity_confirmed",
                        "reject": "speaker_identity_rejected",
                        "defer": "speaker_identity_deferred",
                    }[decision.action],
                    subject_type="speaker_identity_proposal",
                    subject_id=decision.proposal_id,
                    source_type="review_decision",
                    source_id=decision.decision_id,
                    conversation_id=conversation_id,
                    observed_at=decision.decided_at,
                    review_state=review_state,
                    payload=payload,
                )
            )
            if decision.reviewer_asserted_identity:
                asserted_payload = {
                    **payload,
                    "reviewer_asserted_identity": dict(
                        decision.reviewer_asserted_identity
                    ),
                }
                conversation_observations.append(
                    self._observation(
                        observation_key=(
                            "reviewer-asserted",
                            decision.decision_id,
                        ),
                        observation_type="reviewer_asserted_identity",
                        subject_type="speaker_identity_proposal",
                        subject_id=decision.proposal_id,
                        source_type="review_decision",
                        source_id=decision.decision_id,
                        conversation_id=conversation_id,
                        observed_at=decision.decided_at,
                        review_state=review_state,
                        payload=asserted_payload,
                    )
                )
        for superseded_id in sorted(superseded_ids):
            prior = decisions.get(superseded_id)
            if prior is None:
                continue
            replacing = next(
                (
                    decision
                    for decision in history.review_decisions
                    if decision.supersedes_decision_id == superseded_id
                ),
                None,
            )
            observed_at = replacing.decided_at if replacing else prior.decided_at
            conversation_observations.append(
                self._observation(
                    observation_key=("superseded", superseded_id),
                    observation_type="review_decision_superseded",
                    subject_type="review_decision",
                    subject_id=superseded_id,
                    source_type="review_decision",
                    source_id=(
                        replacing.decision_id if replacing else superseded_id
                    ),
                    conversation_id=conversation_id,
                    observed_at=observed_at,
                    review_state="superseded",
                    payload={
                        "superseded_decision_id": superseded_id,
                        "replacement_decision_id": (
                            replacing.decision_id if replacing else ""
                        ),
                    },
                )
            )
        global_observations = self._source_affinity_observations()
        conversation_observations.extend(
            self._concept_affinity_observations(conversation_id)
        )
        conversation_tuple = tuple(
            sorted(
                conversation_observations,
                key=lambda item: (item.observed_at, item.observation_id),
            )
        )
        global_tuple = tuple(
            sorted(
                global_observations,
                key=lambda item: (item.observed_at, item.observation_id),
            )
        )
        conversation_receipt = self.store.save_observations(
            conversation_id,
            conversation_tuple,
        )
        global_receipt = self.store.save_observations("", global_tuple)
        statuses = {
            conversation_receipt.status,
            global_receipt.status,
        }
        status = (
            "inserted"
            if "inserted" in statuses
            else "updated"
            if "updated" in statuses
            else "unchanged"
        )
        all_observations = (*conversation_tuple, *global_tuple)
        return ObservationAppendReceipt(
            status=status,
            conversation_id=conversation_id,
            observation_count=len(all_observations),
            observation_types=tuple(
                sorted(
                    {
                        item.observation_type
                        for item in all_observations
                    }
                )
            ),
        )

    def rebuild(self) -> ProfileBuildReceipt:
        """Rebuild current profiles from immutable observations."""
        observations = self._all_observations()
        people, source_records = self._people_and_sources()
        watermark = _canonical_hash(
            {
                "projection_schema_version": PROJECTION_SCHEMA_VERSION,
                "observations": [
                    {
                        "id": item.observation_id,
                        "content_hash": item.content_hash,
                        "review_state": item.review_state,
                    }
                    for item in observations
                ],
                "people": people,
                "source_records": source_records,
            }
        )
        built_at = max(
            (item.observed_at for item in observations),
            default="",
        )
        person_profiles = self._build_person_profiles(
            people,
            source_records,
            observations,
            watermark=watermark,
            built_at=built_at,
        )
        affinity_profiles = self._build_affinity_profiles(
            observations,
            watermark=watermark,
            built_at=built_at,
        )
        existing_people = self.load_person_profiles()
        existing_affinities = self.load_affinity_profiles()
        if (
            existing_people == person_profiles
            and existing_affinities == affinity_profiles
            and self._projection_watermark() == watermark
        ):
            return ProfileBuildReceipt(
                status="unchanged",
                input_watermark=watermark,
                observation_count=len(observations),
                person_profile_count=len(person_profiles),
                affinity_profile_count=len(affinity_profiles),
            )
        status = (
            "updated"
            if existing_people
            or existing_affinities
            or self._projection_watermark()
            else "inserted"
        )
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute("DELETE FROM knowledge_affinity_profiles")
                con.execute("DELETE FROM knowledge_current_person_profiles")
                for profile in person_profiles:
                    con.execute(
                        """
                        INSERT INTO knowledge_current_person_profiles (
                            person_id, resolution_status, primary_name,
                            aliases_json, source_record_ids_json,
                            observation_ids_json, input_watermark,
                            metadata_json, built_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            profile.person_id,
                            profile.resolution_status,
                            profile.primary_name,
                            _json_dumps(list(profile.aliases)),
                            _json_dumps(list(profile.source_record_ids)),
                            _json_dumps(list(profile.observation_ids)),
                            profile.input_watermark,
                            _json_dumps(profile.metadata),
                            profile.built_at,
                        ),
                    )
                for profile in affinity_profiles:
                    con.execute(
                        """
                        INSERT INTO knowledge_affinity_profiles (
                            id, subject_type, subject_id, affinity_type,
                            object_type, object_id, normalized_value,
                            display_value, support_count,
                            independent_interaction_count,
                            first_observed_at, last_observed_at, review_state,
                            observation_ids_json, input_watermark,
                            metadata_json, built_at
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            profile.affinity_id,
                            profile.subject_type,
                            profile.subject_id,
                            profile.affinity_type,
                            profile.object_type,
                            profile.object_id,
                            profile.normalized_value,
                            profile.display_value,
                            profile.support_count,
                            profile.independent_interaction_count,
                            profile.first_observed_at,
                            profile.last_observed_at,
                            profile.review_state,
                            _json_dumps(list(profile.observation_ids)),
                            profile.input_watermark,
                            _json_dumps(profile.metadata),
                            profile.built_at,
                        ),
                    )
                con.execute(
                    """
                    INSERT INTO knowledge_projection_state (
                        projection_name, scope_type, scope_id, schema_version,
                        input_watermark, built_at, metadata_json
                    )
                    VALUES (?, 'global', 'all', ?, ?, ?, ?)
                    ON CONFLICT(
                        projection_name, scope_type, scope_id
                    ) DO UPDATE SET
                        schema_version = excluded.schema_version,
                        input_watermark = excluded.input_watermark,
                        built_at = excluded.built_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        PROJECTION_NAME,
                        PROJECTION_SCHEMA_VERSION,
                        watermark,
                        built_at,
                        _json_dumps(
                            {
                                "observation_count": len(observations),
                                "person_profile_count": len(person_profiles),
                                "affinity_profile_count": len(
                                    affinity_profiles
                                ),
                            }
                        ),
                    ),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return ProfileBuildReceipt(
            status=status,
            input_watermark=watermark,
            observation_count=len(observations),
            person_profile_count=len(person_profiles),
            affinity_profile_count=len(affinity_profiles),
        )

    def load_person_profiles(self) -> tuple[CurrentPersonProfile, ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_current_person_profiles
                ORDER BY primary_name, person_id
                """
            ).fetchall()
        return tuple(
            CurrentPersonProfile(
                person_id=str(row["person_id"]),
                resolution_status=str(row["resolution_status"]),
                primary_name=str(row["primary_name"]),
                aliases=tuple(
                    str(item)
                    for item in _json_list(str(row["aliases_json"]))
                ),
                source_record_ids=tuple(
                    str(item)
                    for item in _json_list(
                        str(row["source_record_ids_json"])
                    )
                ),
                observation_ids=tuple(
                    str(item)
                    for item in _json_list(
                        str(row["observation_ids_json"])
                    )
                ),
                input_watermark=str(row["input_watermark"]),
                built_at=str(row["built_at"]),
                metadata=_json_object(str(row["metadata_json"])),
            )
            for row in rows
        )

    def load_affinity_profiles(self) -> tuple[AffinityProfile, ...]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_affinity_profiles
                ORDER BY
                    subject_type, subject_id, affinity_type, object_type,
                    object_id, normalized_value, id
                """
            ).fetchall()
        return tuple(
            AffinityProfile(
                affinity_id=str(row["id"]),
                subject_type=str(row["subject_type"]),
                subject_id=str(row["subject_id"]),
                affinity_type=str(row["affinity_type"]),
                object_type=str(row["object_type"]),
                object_id=str(row["object_id"]),
                normalized_value=str(row["normalized_value"]),
                display_value=str(row["display_value"]),
                support_count=int(row["support_count"]),
                independent_interaction_count=int(
                    row["independent_interaction_count"]
                ),
                first_observed_at=str(row["first_observed_at"]),
                last_observed_at=str(row["last_observed_at"]),
                review_state=str(row["review_state"]),
                observation_ids=tuple(
                    str(item)
                    for item in _json_list(
                        str(row["observation_ids_json"])
                    )
                ),
                input_watermark=str(row["input_watermark"]),
                built_at=str(row["built_at"]),
                metadata=_json_object(str(row["metadata_json"])),
            )
            for row in rows
        )

    @staticmethod
    def _observation(
        *,
        observation_key: tuple[str, ...],
        observation_type: str,
        subject_type: str,
        subject_id: str,
        source_type: str,
        source_id: str,
        conversation_id: str,
        observed_at: str,
        review_state: str,
        payload: dict[str, Any],
    ) -> ObservationRecord:
        return ObservationRecord(
            observation_id=_stable_uuid(*observation_key),
            observation_type=observation_type,
            subject_type=subject_type,
            subject_id=subject_id,
            source_type=source_type,
            source_id=source_id,
            conversation_id=conversation_id,
            observed_at=observed_at,
            source_event_at=observed_at,
            review_state=review_state,
            payload=payload,
            content_hash=_canonical_hash(payload),
        )

    def _diarization_observations(
        self,
        *,
        conversation_id: str,
        evaluation_id: str,
        evaluated_at: str,
        proposal: dict[str, Any],
    ) -> list[ObservationRecord]:
        proposal_id = str(proposal.get("proposal_id") or "")
        labels = tuple(
            str(label)
            for label in proposal.get("speaker_labels", [])
            if str(label)
        )
        flags = {
            str(flag)
            for flag in proposal.get("review_flags", [])
            if str(flag)
        }
        person_ids = {
            str(item.get("person_id") or "")
            for item in proposal.get("utterance_assignments", [])
            if isinstance(item, dict)
            and str(item.get("person_id") or "")
        }
        observations: list[ObservationRecord] = []
        if len(labels) > 1 or flags & _SPLIT_FLAGS:
            payload = {
                "evaluation_id": evaluation_id,
                "proposal_id": proposal_id,
                "person_id": _proposal_person_id(proposal),
                "speaker_labels": list(labels),
                "review_flags": sorted(flags),
            }
            observations.append(
                self._observation(
                    observation_key=(
                        "diarization-split",
                        evaluation_id,
                        proposal_id,
                    ),
                    observation_type="diarization_split",
                    subject_type="speaker_identity_proposal",
                    subject_id=proposal_id,
                    source_type="evaluation",
                    source_id=evaluation_id,
                    conversation_id=conversation_id,
                    observed_at=evaluated_at,
                    review_state="unreviewed",
                    payload=payload,
                )
            )
        if len(person_ids) > 1 or flags & _MIXED_FLAGS:
            payload = {
                "evaluation_id": evaluation_id,
                "proposal_id": proposal_id,
                "speaker_labels": list(labels),
                "person_ids": sorted(person_ids),
                "review_flags": sorted(flags),
            }
            observations.append(
                self._observation(
                    observation_key=(
                        "mixed-speaker",
                        evaluation_id,
                        proposal_id,
                    ),
                    observation_type="mixed_speaker",
                    subject_type="speaker_identity_proposal",
                    subject_id=proposal_id,
                    source_type="evaluation",
                    source_id=evaluation_id,
                    conversation_id=conversation_id,
                    observed_at=evaluated_at,
                    review_state="unreviewed",
                    payload=payload,
                )
            )
        return observations

    def _source_affinity_observations(self) -> list[ObservationRecord]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT *
                FROM knowledge_source_records
                WHERE person_id IS NOT NULL AND person_id != ''
                ORDER BY person_id, id
                """
            ).fetchall()
        observations: list[ObservationRecord] = []
        for row in rows:
            payload = {
                "person_id": str(row["person_id"]),
                "source_record_id": str(row["id"]),
                "source_profile_id": str(row["source_profile_id"]),
                "provider_kind": str(row["provider_kind"]),
                "account_id": str(row["account_id"]),
                "tenant_id": str(row["tenant_id"]),
                "relationship_scope": str(row["relationship_scope"]),
                "label": str(row["label"]),
                "content_hash": str(row["content_hash"]),
                "independence_group_id": f"source-record:{row['id']}",
            }
            observations.append(
                self._observation(
                    observation_key=(
                        "source-affinity",
                        str(row["id"]),
                        str(row["content_hash"]),
                    ),
                    observation_type="source_affinity",
                    subject_type="person",
                    subject_id=str(row["person_id"]),
                    source_type="source_record",
                    source_id=str(row["id"]),
                    conversation_id="",
                    observed_at=str(row["observed_at"]),
                    review_state="observed",
                    payload=payload,
                )
            )
        return observations

    def _concept_affinity_observations(
        self,
        conversation_id: str,
    ) -> list[ObservationRecord]:
        with transcript_store.connect(self.root) as con:
            rows = con.execute(
                """
                SELECT mention.*, concept.concept_type,
                       concept.normalized_value, concept.display_value
                FROM knowledge_concept_mentions AS mention
                JOIN knowledge_concepts AS concept
                  ON concept.id = mention.concept_id
                WHERE mention.conversation_id = ?
                ORDER BY mention.observed_at, mention.id
                """,
                (conversation_id,),
            ).fetchall()
        observations: list[ObservationRecord] = []
        for row in rows:
            person_id = str(row["person_id"] or "")
            payload = {
                "person_id": person_id,
                "concept_id": str(row["concept_id"]),
                "concept_type": str(row["concept_type"]),
                "normalized_value": str(row["normalized_value"]),
                "display_value": str(row["display_value"]),
                "evidence_snapshot_id": str(
                    row["evidence_snapshot_id"]
                ),
                "independence_group_id": (
                    f"concept-mention:{row['id']}"
                ),
            }
            observations.append(
                self._observation(
                    observation_key=("concept-affinity", str(row["id"])),
                    observation_type="concept_affinity",
                    subject_type="person" if person_id else "conversation",
                    subject_id=person_id or conversation_id,
                    source_type="concept_mention",
                    source_id=str(row["id"]),
                    conversation_id=conversation_id,
                    observed_at=str(row["observed_at"]),
                    review_state="observed",
                    payload=payload,
                )
            )
        return observations

    def _all_observations(self) -> tuple[ObservationRecord, ...]:
        with transcript_store.connect(self.root) as con:
            conversation_ids = [
                str(row["conversation_id"])
                for row in con.execute(
                    """
                    SELECT DISTINCT conversation_id
                    FROM knowledge_observations
                    WHERE conversation_id IS NOT NULL
                    ORDER BY conversation_id
                    """
                ).fetchall()
            ]
        observations = list(self.store.load_observations(""))
        for conversation_id in conversation_ids:
            observations.extend(
                self.store.load_observations(conversation_id)
            )
        return tuple(
            sorted(
                observations,
                key=lambda item: (item.observed_at, item.observation_id),
            )
        )

    def _people_and_sources(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with transcript_store.connect(self.root) as con:
            people = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_people ORDER BY id"
                ).fetchall()
            ]
            sources = [
                dict(row)
                for row in con.execute(
                    """
                    SELECT *
                    FROM knowledge_source_records
                    ORDER BY person_id, id
                    """
                ).fetchall()
            ]
        return people, sources

    @staticmethod
    def _build_person_profiles(
        people: list[dict[str, Any]],
        source_records: list[dict[str, Any]],
        observations: tuple[ObservationRecord, ...],
        *,
        watermark: str,
        built_at: str,
    ) -> tuple[CurrentPersonProfile, ...]:
        profiles: list[CurrentPersonProfile] = []
        for person in people:
            person_id = str(person["id"])
            sources = [
                item
                for item in source_records
                if str(item.get("person_id") or "") == person_id
            ]
            related = [
                item
                for item in observations
                if item.subject_id == person_id
                or str(item.payload.get("person_id") or "") == person_id
            ]
            if not related:
                continue
            primary_name = str(person["primary_name"])
            aliases = tuple(
                sorted(
                    {
                        str(item.get("label") or "")
                        for item in sources
                        if str(item.get("label") or "")
                        and str(item.get("label") or "") != primary_name
                    },
                    key=str.casefold,
                )
            )
            profiles.append(
                CurrentPersonProfile(
                    person_id=person_id,
                    resolution_status=str(person["status"]),
                    primary_name=primary_name,
                    aliases=aliases,
                    source_record_ids=tuple(
                        sorted(str(item["id"]) for item in sources)
                    ),
                    observation_ids=tuple(
                        sorted(item.observation_id for item in related)
                    ),
                    input_watermark=watermark,
                    built_at=built_at,
                    metadata={
                        "source_profiles": sorted(
                            {
                                str(item["source_profile_id"])
                                for item in sources
                            }
                        )
                    },
                )
            )
        return tuple(
            sorted(
                profiles,
                key=lambda item: (
                    item.primary_name.casefold(),
                    item.person_id,
                ),
            )
        )

    @staticmethod
    def _build_affinity_profiles(
        observations: tuple[ObservationRecord, ...],
        *,
        watermark: str,
        built_at: str,
    ) -> tuple[AffinityProfile, ...]:
        buckets: dict[
            tuple[str, str, str, str, str, str],
            dict[str, Any],
        ] = {}

        def add(
            *,
            person_id: str,
            affinity_type: str,
            object_type: str,
            object_id: str,
            value: str,
            display_value: str,
            observation: ObservationRecord,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            if not person_id:
                return
            normalized = value.strip().casefold()
            key = (
                "person",
                person_id,
                affinity_type,
                object_type,
                object_id,
                normalized,
            )
            bucket = buckets.setdefault(
                key,
                {
                    "display_value": display_value or value,
                    "observations": [],
                    "independence": set(),
                    "metadata": {},
                },
            )
            bucket["observations"].append(observation)
            independence = str(
                observation.payload.get("independence_group_id")
                or observation.conversation_id
                or observation.source_id
            )
            bucket["independence"].add(independence)
            if metadata:
                for metadata_key, metadata_value in metadata.items():
                    existing = bucket["metadata"].setdefault(
                        metadata_key,
                        [],
                    )
                    values = (
                        metadata_value
                        if isinstance(metadata_value, list)
                        else [metadata_value]
                    )
                    for item in values:
                        if item not in existing:
                            existing.append(item)

        for observation in observations:
            payload = observation.payload
            person_id = str(payload.get("person_id") or "")
            if observation.observation_type == "source_affinity":
                add(
                    person_id=person_id,
                    affinity_type="source_relationship",
                    object_type="source_profile",
                    object_id=str(payload.get("source_profile_id") or ""),
                    value=str(payload.get("relationship_scope") or ""),
                    display_value=str(payload.get("source_profile_id") or ""),
                    observation=observation,
                    metadata={
                        "source_record_ids": [
                            str(payload.get("source_record_id") or "")
                        ],
                        "account_ids": [
                            str(payload.get("account_id") or "")
                        ],
                        "tenant_ids": [
                            str(payload.get("tenant_id") or "")
                        ],
                    },
                )
                continue
            if observation.observation_type == "concept_affinity":
                concept_type = str(payload.get("concept_type") or "topic")
                affinity_type = (
                    "terminology"
                    if concept_type in {"term", "terminology", "phrase"}
                    else concept_type
                )
                add(
                    person_id=person_id,
                    affinity_type=affinity_type,
                    object_type="concept",
                    object_id=str(payload.get("concept_id") or ""),
                    value=str(payload.get("normalized_value") or ""),
                    display_value=str(payload.get("display_value") or ""),
                    observation=observation,
                )
                continue
            if observation.observation_type != "speaker_identity_confirmed":
                continue
            proposal = payload.get("proposal")
            proposal = proposal if isinstance(proposal, dict) else {}
            add(
                person_id=person_id,
                affinity_type="interaction",
                object_type="conversation",
                object_id=observation.conversation_id,
                value="participant",
                display_value="Conversation participant",
                observation=observation,
            )
            identity = proposal.get("identity")
            identity = identity if isinstance(identity, dict) else {}
            organization = str(
                proposal.get("organization")
                or identity.get("organization")
                or ""
            ).strip()
            if organization:
                add(
                    person_id=person_id,
                    affinity_type="organization",
                    object_type="organization",
                    object_id="",
                    value=organization,
                    display_value=organization,
                    observation=observation,
                )
            for value in _proposal_values(proposal, "project") + _proposal_values(
                proposal,
                "projects",
            ):
                add(
                    person_id=person_id,
                    affinity_type="project",
                    object_type="project",
                    object_id="",
                    value=value,
                    display_value=value,
                    observation=observation,
                )
            for value in _proposal_values(proposal, "topics"):
                add(
                    person_id=person_id,
                    affinity_type="topic",
                    object_type="concept",
                    object_id="",
                    value=value,
                    display_value=value,
                    observation=observation,
                )
            for value in _proposal_values(proposal, "terms"):
                add(
                    person_id=person_id,
                    affinity_type="terminology",
                    object_type="concept",
                    object_id="",
                    value=value,
                    display_value=value,
                    observation=observation,
                )

        profiles: list[AffinityProfile] = []
        for key, bucket in buckets.items():
            (
                subject_type,
                subject_id,
                affinity_type,
                object_type,
                object_id,
                normalized_value,
            ) = key
            supporting = sorted(
                bucket["observations"],
                key=lambda item: (item.observed_at, item.observation_id),
            )
            review_state = (
                "reviewed"
                if any(
                    item.review_state in {"confirm", "confirmed", "reviewed"}
                    for item in supporting
                )
                else "observed"
            )
            profiles.append(
                AffinityProfile(
                    affinity_id=_stable_uuid(
                        "affinity",
                        *key,
                    ),
                    subject_type=subject_type,
                    subject_id=subject_id,
                    affinity_type=affinity_type,
                    object_type=object_type,
                    object_id=object_id,
                    normalized_value=normalized_value,
                    display_value=str(bucket["display_value"]),
                    support_count=len(supporting),
                    independent_interaction_count=len(
                        bucket["independence"]
                    ),
                    first_observed_at=(
                        supporting[0].observed_at if supporting else ""
                    ),
                    last_observed_at=(
                        supporting[-1].observed_at if supporting else ""
                    ),
                    review_state=review_state,
                    observation_ids=tuple(
                        item.observation_id for item in supporting
                    ),
                    input_watermark=watermark,
                    built_at=built_at,
                    metadata={
                        key: sorted(
                            str(item)
                            for item in values
                        )
                        for key, values in bucket["metadata"].items()
                    },
                )
            )
        return tuple(
            sorted(
                profiles,
                key=lambda item: (
                    item.subject_type,
                    item.subject_id,
                    item.affinity_type,
                    item.object_type,
                    item.object_id,
                    item.normalized_value,
                    item.affinity_id,
                ),
            )
        )

    def _projection_watermark(self) -> str:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                """
                SELECT input_watermark
                FROM knowledge_projection_state
                WHERE projection_name = ?
                  AND scope_type = 'global'
                  AND scope_id = 'all'
                """,
                (PROJECTION_NAME,),
            ).fetchone()
        return str(row["input_watermark"]) if row is not None else ""
