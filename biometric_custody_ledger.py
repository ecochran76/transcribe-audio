from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

import transcript_store
from identity_learning_contracts import ARTIFACT_SCHEMAS, validate_artifact


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{uuid5(NAMESPACE_URL, chr(31).join(parts))}"


def _text(value: object) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class PrivateObjectReceipt:
    object_id: str
    sha256: str
    size_bytes: int
    status: str


@dataclass(frozen=True)
class VoiceSampleReceipt:
    sample_id: str
    sample_sha256: str
    review_state: str
    exclusion_state: str
    status: str


@dataclass(frozen=True)
class ClusterVersionReceipt:
    cluster_version_id: str
    cluster_id: str
    membership_count: int
    status: str


@dataclass(frozen=True)
class ClusterRescoreReceipt:
    rescore_receipt_id: str
    cluster_version_id: str
    anchor_sample_id: str
    requeued_sample_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ProfileFamilyReceipt:
    profile_family_id: str
    person_id: str
    family_key: str
    status: str


@dataclass(frozen=True)
class ProfileVersionReceipt:
    profile_version_id: str
    profile_family_id: str
    sample_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ProfileEventReceipt:
    event_id: str
    profile_version_id: str
    action: str
    status: str


@dataclass(frozen=True)
class SampleEventReceipt:
    event_id: str
    sample_id: str
    event_type: str
    status: str


@dataclass(frozen=True)
class CustodyEffectReceipt:
    effect_id: str
    mode: str
    target_type: str
    target_id: str
    sample_event_ids: tuple[str, ...]
    profile_event_ids: tuple[str, ...]
    cluster_event_ids: tuple[str, ...]
    tombstone_id: str
    status: str


@dataclass(frozen=True)
class ProfileRebuildReceipt:
    rebuild_receipt_id: str
    profile_version_id: str
    source_object_sha256: str
    rebuilt_object_sha256: str
    byte_equal: bool
    status: str


class BiometricCustodyLedger:
    """Keep portable custody metadata separate from private biometric bytes."""

    def __init__(self, root: Path, *, private_root: Path) -> None:
        self.root = transcript_store.store_dir(root)
        self.private_root = private_root.expanduser().resolve()
        if not self.private_root.is_dir() or self.private_root.is_symlink():
            raise ValueError("Biometric private root must be an existing directory.")
        if stat.S_IMODE(self.private_root.stat().st_mode) & 0o077:
            raise ValueError(
                "Biometric private root must not be group/world accessible."
            )
        self.objects_root = self.private_root / "objects"
        self.objects_root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(self.objects_root, 0o700)
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT schema_version, dirty FROM knowledge_store_state "
                "WHERE singleton = 1"
            ).fetchone()
        if row is None or int(row["schema_version"]) < 6 or bool(row["dirty"]):
            raise RuntimeError("Biometric custody requires knowledge schema v6.")

    def store_private_object(
        self,
        *,
        object_id: str,
        payload: bytes,
    ) -> PrivateObjectReceipt:
        path = self._object_path(object_id)
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("Private biometric payload must be non-empty bytes.")
        sha256 = hashlib.sha256(payload).hexdigest()
        if path.exists():
            self._validate_private_object(object_id, sha256)
            return PrivateObjectReceipt(object_id, sha256, len(payload), "unchanged")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        self._validate_private_object(object_id, sha256)
        return PrivateObjectReceipt(object_id, sha256, len(payload), "inserted")

    def register_sample(
        self,
        *,
        conversation_id: str,
        recording_id: str,
        speaker_ref: str,
        start_ms: int,
        end_ms: int,
        source_media_sha256: str,
        sample_sha256: str,
        quality: Mapping[str, Any],
        preparation_lineage: Mapping[str, Any],
        review_state: str,
        private_object_id: str,
        private_object_sha256: str,
        created_at: str,
        person_id: str = "",
        review_authority_id: str = "",
        consent_authority: str = "",
    ) -> VoiceSampleReceipt:
        if review_state not in {"unreviewed", "reviewed", "rejected"}:
            raise ValueError("Voice sample review state is invalid.")
        if person_id and (
            review_state != "reviewed"
            or not _text(review_authority_id)
            or not _text(consent_authority)
        ):
            raise ValueError(
                "A person-bound sample requires reviewed identity and authority."
            )
        self._require_sha256(source_media_sha256, "source_media_sha256")
        self._require_sha256(sample_sha256, "sample_sha256")
        self._require_sha256(private_object_sha256, "private_object_sha256")
        if sample_sha256 != private_object_sha256:
            raise ValueError("Voice sample and private object hashes must match.")
        self._validate_private_object(private_object_id, private_object_sha256)
        core = {
            "conversation_id": _text(conversation_id),
            "recording_id": _text(recording_id),
            "speaker_ref": _text(speaker_ref),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "source_media_sha256": source_media_sha256,
            "sample_sha256": sample_sha256,
            "quality": dict(quality),
            "preparation_lineage": dict(preparation_lineage),
            "review_authority_id": _text(review_authority_id),
            "consent_authority": _text(consent_authority),
            "person_id": _text(person_id),
            "review_state": review_state,
            "exclusion_state": "included",
            "private_object_id": private_object_id,
            "private_object_sha256": private_object_sha256,
        }
        if not all(
            (
                core["conversation_id"],
                core["recording_id"],
                core["speaker_ref"],
                _text(created_at),
            )
        ):
            raise ValueError("Voice sample identity and created_at are required.")
        sample_id = _stable_id(
            "voice-sample",
            core["recording_id"],
            str(start_ms),
            str(end_ms),
            sample_sha256,
        )
        validate_artifact(
            "voice_sample",
            {
                "schema_version": ARTIFACT_SCHEMAS["voice_sample"],
                "sample_id": sample_id,
                "recording_id": core["recording_id"],
                "conversation_id": core["conversation_id"],
                "speaker_ref": core["speaker_ref"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "source_media_sha256": source_media_sha256,
                "sample_sha256": sample_sha256,
                "quality": dict(quality),
                "preparation_lineage": dict(preparation_lineage),
                "review_authority_id": core["review_authority_id"],
                "consent_authority": core["consent_authority"],
                "person_id": core["person_id"],
                "review_state": review_state,
                "exclusion_state": "included",
                "private_audio_ref": {
                    "object_id": private_object_id,
                    "sha256": private_object_sha256,
                },
                "created_at": _text(created_at),
            },
        )
        content_hash = _hash(core)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT id, content_hash FROM knowledge_voice_samples "
                "WHERE sample_sha256 = ?",
                (sample_sha256,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Voice sample hash already has different metadata."
                    )
                return VoiceSampleReceipt(
                    str(existing["id"]),
                    sample_sha256,
                    review_state,
                    "included",
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_voice_samples (
                    id, conversation_id, recording_id, speaker_ref,
                    start_ms, end_ms, source_media_sha256, sample_sha256,
                    quality_json, preparation_lineage_json,
                    review_authority_id, consent_authority, person_id,
                    review_state, exclusion_state, private_object_id,
                    private_object_sha256, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    core["conversation_id"],
                    core["recording_id"],
                    core["speaker_ref"],
                    start_ms,
                    end_ms,
                    source_media_sha256,
                    sample_sha256,
                    _json(dict(quality)),
                    _json(dict(preparation_lineage)),
                    core["review_authority_id"] or None,
                    core["consent_authority"] or None,
                    core["person_id"] or None,
                    review_state,
                    "included",
                    private_object_id,
                    private_object_sha256,
                    content_hash,
                    _text(created_at),
                ),
            )
            con.commit()
        return VoiceSampleReceipt(
            sample_id,
            sample_sha256,
            review_state,
            "included",
            "inserted",
        )

    def record_cluster_version(
        self,
        *,
        cluster_id: str,
        algorithm_version: str,
        memberships: tuple[Mapping[str, Any], ...],
        status: str,
        created_at: str,
        predecessor_version_id: str = "",
    ) -> ClusterVersionReceipt:
        if status not in {"candidate", "reviewed", "superseded", "deleted"}:
            raise ValueError("Anonymous cluster status is invalid.")
        if not all(map(_text, (cluster_id, algorithm_version, created_at))):
            raise ValueError("Anonymous cluster identity and version are required.")
        if not memberships:
            raise ValueError("Anonymous cluster requires soft memberships.")
        prepared: list[dict[str, Any]] = []
        seen_samples: set[str] = set()
        seen_ranks: set[int] = set()
        with transcript_store.connect(self.root) as con:
            if predecessor_version_id:
                predecessor = con.execute(
                    "SELECT cluster_id FROM knowledge_anonymous_cluster_versions "
                    "WHERE id = ?",
                    (predecessor_version_id,),
                ).fetchone()
                if predecessor is None or str(predecessor["cluster_id"]) != cluster_id:
                    raise ValueError("Anonymous cluster predecessor is invalid.")
            for membership in memberships:
                sample_id = _text(membership.get("sample_id"))
                rank = membership.get("rank")
                score = membership.get("score")
                state = _text(membership.get("membership_state"))
                evidence_ids = tuple(
                    dict.fromkeys(
                        _text(item) for item in membership.get("evidence_ids", ())
                    )
                )
                sample = con.execute(
                    "SELECT 1 FROM knowledge_voice_samples WHERE id = ?",
                    (sample_id,),
                ).fetchone()
                if sample is None:
                    raise ValueError("Anonymous cluster sample is unknown.")
                if (
                    sample_id in seen_samples
                    or not isinstance(rank, int)
                    or rank < 1
                    or rank in seen_ranks
                    or not isinstance(score, (int, float))
                    or not 0.0 <= float(score) <= 1.0
                    or state not in {
                        "candidate",
                        "confirmed",
                        "rejected",
                        "excluded",
                    }
                    or not evidence_ids
                    or any(not item for item in evidence_ids)
                ):
                    raise ValueError("Anonymous cluster membership is invalid.")
                seen_samples.add(sample_id)
                seen_ranks.add(rank)
                prepared.append(
                    {
                        "sample_id": sample_id,
                        "rank": rank,
                        "score": float(score),
                        "evidence_ids": evidence_ids,
                        "membership_state": state,
                    }
                )
        prepared.sort(key=lambda item: (item["rank"], item["sample_id"]))
        core = {
            "cluster_id": cluster_id,
            "predecessor_version_id": _text(predecessor_version_id),
            "algorithm_version": algorithm_version,
            "status": status,
            "memberships": prepared,
        }
        content_hash = _hash(core)
        version_id = _stable_id("anonymous-cluster-version", cluster_id, content_hash)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM knowledge_anonymous_cluster_versions "
                "WHERE id = ?",
                (version_id,),
            ).fetchone()
            if existing is not None:
                return ClusterVersionReceipt(
                    version_id, cluster_id, len(prepared), "unchanged"
                )
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute(
                    """
                    INSERT INTO knowledge_anonymous_cluster_versions (
                        id, cluster_id, predecessor_version_id,
                        algorithm_version, status, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        cluster_id,
                        _text(predecessor_version_id) or None,
                        algorithm_version,
                        status,
                        content_hash,
                        created_at,
                    ),
                )
                for membership in prepared:
                    membership_hash = _hash(
                        {"cluster_version_id": version_id, **membership}
                    )
                    con.execute(
                        """
                        INSERT INTO knowledge_anonymous_cluster_memberships (
                            id, cluster_version_id, sample_id, rank, score,
                            evidence_ids_json, membership_state, content_hash,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id(
                                "cluster-membership",
                                version_id,
                                membership["sample_id"],
                                str(membership["rank"]),
                            ),
                            version_id,
                            membership["sample_id"],
                            membership["rank"],
                            membership["score"],
                            _json(list(membership["evidence_ids"])),
                            membership["membership_state"],
                            membership_hash,
                            created_at,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return ClusterVersionReceipt(version_id, cluster_id, len(prepared), "inserted")

    def load_cluster_version(self, cluster_version_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT * FROM knowledge_anonymous_cluster_versions WHERE id = ?",
                (cluster_version_id,),
            ).fetchone()
            memberships = con.execute(
                "SELECT * FROM knowledge_anonymous_cluster_memberships "
                "WHERE cluster_version_id = ? ORDER BY rank, sample_id",
                (cluster_version_id,),
            ).fetchall()
        if row is None:
            raise ValueError("Anonymous cluster version is unknown.")
        result = dict(row)
        result["memberships"] = [
            {
                "sample_id": str(item["sample_id"]),
                "rank": int(item["rank"]),
                "score": float(item["score"]),
                "evidence_ids": json.loads(str(item["evidence_ids_json"])),
                "membership_state": str(item["membership_state"]),
                "effective_membership_state": (
                    "excluded"
                    if self.sample_state(str(item["sample_id"]))[
                        "exclusion_state"
                    ]
                    in {"excluded", "deleted"}
                    else str(item["membership_state"])
                ),
            }
            for item in memberships
        ]
        return result

    def record_cluster_rescore(
        self,
        *,
        cluster_version_id: str,
        anchor_sample_id: str,
        score_updates: tuple[Mapping[str, Any], ...],
        material_threshold: float,
        processing_version: str,
        created_at: str,
    ) -> ClusterRescoreReceipt:
        if (
            not all(
                map(
                    _text,
                    (
                        cluster_version_id,
                        anchor_sample_id,
                        processing_version,
                        created_at,
                    ),
                )
            )
            or not isinstance(material_threshold, (int, float))
            or not 0.0 < float(material_threshold) <= 1.0
            or not score_updates
        ):
            raise ValueError("Anonymous cluster rescore receipt is incomplete.")
        with transcript_store.connect(self.root) as con:
            cluster = con.execute(
                "SELECT status FROM knowledge_anonymous_cluster_versions "
                "WHERE id = ?",
                (cluster_version_id,),
            ).fetchone()
            anchor = con.execute(
                """
                SELECT sample.*, membership.membership_state
                FROM knowledge_anonymous_cluster_memberships membership
                JOIN knowledge_voice_samples sample
                  ON sample.id = membership.sample_id
                WHERE membership.cluster_version_id = ?
                  AND membership.sample_id = ?
                """,
                (cluster_version_id, anchor_sample_id),
            ).fetchone()
            memberships = {
                str(row["sample_id"]): row
                for row in con.execute(
                    """
                    SELECT membership.sample_id, membership.score,
                           sample.review_state, sample.person_id
                    FROM knowledge_anonymous_cluster_memberships membership
                    JOIN knowledge_voice_samples sample
                      ON sample.id = membership.sample_id
                    WHERE membership.cluster_version_id = ?
                    """,
                    (cluster_version_id,),
                ).fetchall()
            }
        if cluster is None or str(cluster["status"]) != "reviewed":
            raise ValueError("Cluster rescore requires a reviewed cluster version.")
        if (
            anchor is None
            or str(anchor["membership_state"]) != "confirmed"
            or str(anchor["review_state"]) != "reviewed"
            or not str(anchor["person_id"] or "")
        ):
            raise ValueError("Cluster rescore requires a confirmed reviewed anchor.")
        prepared: list[dict[str, Any]] = []
        seen: set[str] = set()
        for update in score_updates:
            sample_id = _text(update.get("sample_id"))
            old_score = update.get("old_score")
            new_score = update.get("new_score")
            membership = memberships.get(sample_id)
            if (
                not sample_id
                or sample_id == anchor_sample_id
                or sample_id in seen
                or membership is None
                or not isinstance(old_score, (int, float))
                or not isinstance(new_score, (int, float))
                or not 0.0 <= float(old_score) <= 1.0
                or not 0.0 <= float(new_score) <= 1.0
                or float(old_score) != float(membership["score"])
                or str(membership["review_state"]) != "unreviewed"
                or str(membership["person_id"] or "")
            ):
                raise ValueError(
                    "Cluster rescore updates must target related person-unbound "
                    "unreviewed memberships at their recorded score."
                )
            seen.add(sample_id)
            prepared.append(
                {
                    "sample_id": sample_id,
                    "old_score": float(old_score),
                    "new_score": float(new_score),
                    "material": abs(float(new_score) - float(old_score))
                    >= float(material_threshold),
                }
            )
        prepared.sort(key=lambda item: item["sample_id"])
        requeued = tuple(
            item["sample_id"] for item in prepared if item["material"]
        )
        core = {
            "cluster_version_id": cluster_version_id,
            "anchor_sample_id": anchor_sample_id,
            "processing_version": processing_version,
            "material_threshold": float(material_threshold),
            "updates": prepared,
            "requeued_sample_ids": list(requeued),
        }
        content_hash = _hash(core)
        receipt_id = _stable_id(
            "cluster-rescore-receipt",
            cluster_version_id,
            anchor_sample_id,
            processing_version,
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT id, content_hash, requeued_sample_ids_json "
                "FROM knowledge_cluster_rescore_receipts "
                "WHERE cluster_version_id = ? AND anchor_sample_id = ? "
                "AND processing_version = ?",
                (cluster_version_id, anchor_sample_id, processing_version),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Cluster rescore receipt idempotency drifted.")
                return ClusterRescoreReceipt(
                    str(existing["id"]),
                    cluster_version_id,
                    anchor_sample_id,
                    tuple(json.loads(str(existing["requeued_sample_ids_json"]))),
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_cluster_rescore_receipts (
                    id, cluster_version_id, anchor_sample_id,
                    processing_version, material_threshold, updates_json,
                    requeued_sample_ids_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    cluster_version_id,
                    anchor_sample_id,
                    processing_version,
                    float(material_threshold),
                    _json(prepared),
                    _json(list(requeued)),
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return ClusterRescoreReceipt(
            receipt_id,
            cluster_version_id,
            anchor_sample_id,
            requeued,
            "inserted",
        )

    def register_profile_family(
        self,
        *,
        person_id: str,
        family_key: str,
        conditions: Mapping[str, Any],
        created_at: str,
    ) -> ProfileFamilyReceipt:
        if not all(map(_text, (person_id, family_key, created_at))):
            raise ValueError("Voice profile family identity is incomplete.")
        core = {
            "person_id": _text(person_id),
            "family_key": _text(family_key),
            "conditions": dict(conditions),
        }
        content_hash = _hash(core)
        family_id = _stable_id("voice-profile-family", person_id, family_key)
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT id, content_hash FROM knowledge_voice_profile_families "
                "WHERE person_id = ? AND family_key = ?",
                (_text(person_id), _text(family_key)),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError(
                        "Voice profile family already has other conditions."
                    )
                return ProfileFamilyReceipt(
                    str(existing["id"]), person_id, family_key, "unchanged"
                )
            con.execute(
                """
                INSERT INTO knowledge_voice_profile_families (
                    id, person_id, family_key, conditions_json,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    family_id,
                    person_id,
                    family_key,
                    _json(dict(conditions)),
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return ProfileFamilyReceipt(family_id, person_id, family_key, "inserted")

    def build_profile_version(
        self,
        *,
        profile_family_id: str,
        sample_ids: tuple[str, ...],
        evaluation_id: str,
        model_revision: str,
        recipe_revision: str,
        private_object_id: str,
        private_object_sha256: str,
        created_at: str,
        predecessor_profile_version_id: str = "",
    ) -> ProfileVersionReceipt:
        sample_ids = tuple(sorted(dict.fromkeys(map(_text, sample_ids))))
        if not sample_ids or not all(
            map(
                _text,
                (
                    profile_family_id,
                    evaluation_id,
                    model_revision,
                    recipe_revision,
                    created_at,
                ),
            )
        ):
            raise ValueError("Voice profile version is incomplete.")
        self._require_sha256(private_object_sha256, "private_object_sha256")
        self._validate_private_object(private_object_id, private_object_sha256)
        with transcript_store.connect(self.root) as con:
            family = con.execute(
                "SELECT person_id FROM knowledge_voice_profile_families WHERE id = ?",
                (profile_family_id,),
            ).fetchone()
            if family is None:
                raise ValueError("Voice profile family is unknown.")
            person_id = str(family["person_id"])
            samples = con.execute(
                f"SELECT * FROM knowledge_voice_samples WHERE id IN "
                f"({','.join('?' for _ in sample_ids)}) ORDER BY id",
                sample_ids,
            ).fetchall()
            if predecessor_profile_version_id:
                predecessor = con.execute(
                    "SELECT profile_family_id FROM knowledge_voice_profile_versions "
                    "WHERE id = ?",
                    (predecessor_profile_version_id,),
                ).fetchone()
                if (
                    predecessor is None
                    or str(predecessor["profile_family_id"]) != profile_family_id
                ):
                    raise ValueError("Voice profile predecessor is invalid.")
        if len(samples) != len(sample_ids) or any(
            str(row["review_state"]) != "reviewed"
            or str(row["exclusion_state"]) != "included"
            or str(row["person_id"] or "") != person_id
            or not str(row["review_authority_id"] or "")
            or not str(row["consent_authority"] or "")
            or not bool(json.loads(str(row["quality_json"])).get("eligible"))
            for row in samples
        ):
            raise ValueError("Voice profiles require reviewed eligible samples.")
        allowlist = [
            {
                "sample_id": str(row["id"]),
                "review_authority_id": str(row["review_authority_id"]),
                "consent_authority": str(row["consent_authority"]),
            }
            for row in samples
        ]
        core = {
            "profile_family_id": profile_family_id,
            "person_id": person_id,
            "predecessor_profile_version_id": _text(
                predecessor_profile_version_id
            ),
            "sample_allowlist": allowlist,
            "evaluation_id": evaluation_id,
            "model_revision": model_revision,
            "recipe_revision": recipe_revision,
            "status": "pending",
            "private_object_id": private_object_id,
            "private_object_sha256": private_object_sha256,
        }
        content_hash = _hash(core)
        version_id = _stable_id(
            "voice-profile-version", profile_family_id, content_hash
        )
        validate_artifact(
            "voice_profile_version",
            {
                "schema_version": ARTIFACT_SCHEMAS["voice_profile_version"],
                "profile_version_id": version_id,
                "person_id": person_id,
                "profile_family": profile_family_id,
                "predecessor_profile_version_id": (
                    predecessor_profile_version_id or None
                ),
                "sample_allowlist": allowlist,
                "evaluation_id": evaluation_id,
                "status": "pending",
                "active_interval": None,
                "private_profile_ref": {
                    "object_id": private_object_id,
                    "sha256": private_object_sha256,
                },
                "created_at": created_at,
            },
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM knowledge_voice_profile_versions "
                "WHERE id = ?",
                (version_id,),
            ).fetchone()
            if existing is not None:
                return ProfileVersionReceipt(
                    version_id, profile_family_id, sample_ids, "unchanged"
                )
            con.execute(
                """
                INSERT INTO knowledge_voice_profile_versions (
                    id, profile_family_id, person_id,
                    predecessor_profile_version_id, sample_allowlist_json,
                    evaluation_id, model_revision, recipe_revision, status,
                    private_object_id, private_object_sha256,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    profile_family_id,
                    person_id,
                    predecessor_profile_version_id or None,
                    _json(allowlist),
                    evaluation_id,
                    model_revision,
                    recipe_revision,
                    "pending",
                    private_object_id,
                    private_object_sha256,
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return ProfileVersionReceipt(
            version_id, profile_family_id, sample_ids, "inserted"
        )

    def record_profile_event(
        self,
        *,
        profile_version_id: str,
        action: str,
        reason_code: str,
        authority_id: str,
        idempotency_key: str,
        created_at: str,
        supersedes_event_id: str = "",
    ) -> ProfileEventReceipt:
        actions = {
            "activate",
            "reject",
            "supersede",
            "invalidate",
            "rollback",
            "delete",
        }
        if action not in actions or not all(
            map(
                _text,
                (
                    profile_version_id,
                    reason_code,
                    authority_id,
                    idempotency_key,
                    created_at,
                ),
            )
        ):
            raise ValueError("Voice profile event is invalid.")
        core = {
            "profile_version_id": profile_version_id,
            "action": action,
            "reason_code": reason_code,
            "authority_id": authority_id,
            "idempotency_key": idempotency_key,
            "supersedes_event_id": _text(supersedes_event_id),
        }
        content_hash = _hash(core)
        event_id = _stable_id("voice-profile-event", idempotency_key)
        with transcript_store.connect(self.root) as con:
            profile = con.execute(
                "SELECT 1 FROM knowledge_voice_profile_versions WHERE id = ?",
                (profile_version_id,),
            ).fetchone()
            if profile is None:
                raise ValueError("Voice profile version is unknown.")
            existing = con.execute(
                "SELECT id, content_hash FROM knowledge_voice_profile_events "
                "WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_hash"]) != content_hash:
                    raise ValueError("Voice profile event idempotency drifted.")
                return ProfileEventReceipt(
                    str(existing["id"]), profile_version_id, action, "unchanged"
                )
            heads = con.execute(
                """
                SELECT event.id FROM knowledge_voice_profile_events event
                WHERE event.profile_version_id = ? AND NOT EXISTS (
                    SELECT 1 FROM knowledge_voice_profile_events successor
                    WHERE successor.supersedes_event_id = event.id
                )
                """,
                (profile_version_id,),
            ).fetchall()
            current_id = str(heads[0]["id"]) if len(heads) == 1 else ""
            if len(heads) > 1 or _text(supersedes_event_id) != current_id:
                raise ValueError("Voice profile event must supersede its current head.")
            con.execute(
                """
                INSERT INTO knowledge_voice_profile_events (
                    id, profile_version_id, action, reason_code, authority_id,
                    idempotency_key, supersedes_event_id, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    profile_version_id,
                    action,
                    reason_code,
                    authority_id,
                    idempotency_key,
                    supersedes_event_id or None,
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return ProfileEventReceipt(event_id, profile_version_id, action, "inserted")

    def profile_state(self, profile_version_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            profile = con.execute(
                "SELECT * FROM knowledge_voice_profile_versions WHERE id = ?",
                (profile_version_id,),
            ).fetchone()
            head = con.execute(
                """
                SELECT event.* FROM knowledge_voice_profile_events event
                WHERE event.profile_version_id = ? AND NOT EXISTS (
                    SELECT 1 FROM knowledge_voice_profile_events successor
                    WHERE successor.supersedes_event_id = event.id
                )
                """,
                (profile_version_id,),
            ).fetchone()
        if profile is None:
            raise ValueError("Voice profile version is unknown.")
        action_status = {
            "activate": "active",
            "reject": "rejected",
            "supersede": "superseded",
            "invalidate": "invalidated",
            "rollback": "active",
            "delete": "deleted",
        }
        return {
            "profile_version_id": profile_version_id,
            "profile_family_id": str(profile["profile_family_id"]),
            "person_id": str(profile["person_id"]),
            "sample_ids": [
                item["sample_id"]
                for item in json.loads(str(profile["sample_allowlist_json"]))
            ],
            "status": (
                action_status[str(head["action"])]
                if head is not None
                else str(profile["status"])
            ),
            "event_id": str(head["id"]) if head is not None else "",
        }

    def verify_profile_rebuild(
        self,
        *,
        profile_version_id: str,
        rebuilt_object_id: str,
        rebuilt_object_sha256: str,
        model_revision: str,
        recipe_revision: str,
        created_at: str,
    ) -> ProfileRebuildReceipt:
        self._require_sha256(rebuilt_object_sha256, "rebuilt_object_sha256")
        self._validate_private_object(rebuilt_object_id, rebuilt_object_sha256)
        with transcript_store.connect(self.root) as con:
            profile = con.execute(
                "SELECT * FROM knowledge_voice_profile_versions WHERE id = ?",
                (profile_version_id,),
            ).fetchone()
        if profile is None:
            raise ValueError("Voice profile version is unknown.")
        if (
            str(profile["model_revision"]) != _text(model_revision)
            or str(profile["recipe_revision"]) != _text(recipe_revision)
            or not _text(created_at)
        ):
            raise ValueError("Voice profile rebuild lineage does not match.")
        source_sha256 = str(profile["private_object_sha256"] or "")
        byte_equal = source_sha256 == rebuilt_object_sha256
        core = {
            "profile_version_id": profile_version_id,
            "source_object_sha256": source_sha256,
            "rebuilt_object_sha256": rebuilt_object_sha256,
            "model_revision": model_revision,
            "recipe_revision": recipe_revision,
            "byte_equal": byte_equal,
        }
        content_hash = _hash(core)
        receipt_id = _stable_id(
            "voice-profile-rebuild", profile_version_id, rebuilt_object_sha256
        )
        with transcript_store.connect(self.root) as con:
            existing = con.execute(
                "SELECT content_hash FROM knowledge_voice_profile_rebuild_receipts "
                "WHERE id = ?",
                (receipt_id,),
            ).fetchone()
            if existing is not None:
                return ProfileRebuildReceipt(
                    receipt_id,
                    profile_version_id,
                    source_sha256,
                    rebuilt_object_sha256,
                    byte_equal,
                    "unchanged",
                )
            con.execute(
                """
                INSERT INTO knowledge_voice_profile_rebuild_receipts (
                    id, profile_version_id, source_object_sha256,
                    rebuilt_object_sha256, model_revision, recipe_revision,
                    byte_equal, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    profile_version_id,
                    source_sha256,
                    rebuilt_object_sha256,
                    model_revision,
                    recipe_revision,
                    int(byte_equal),
                    content_hash,
                    created_at,
                ),
            )
            con.commit()
        return ProfileRebuildReceipt(
            receipt_id,
            profile_version_id,
            source_sha256,
            rebuilt_object_sha256,
            byte_equal,
            "inserted",
        )

    def record_sample_event(
        self,
        *,
        sample_id: str,
        event_type: str,
        actor_id: str,
        authority_id: str,
        idempotency_key: str,
        created_at: str,
        supersedes_event_id: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> SampleEventReceipt:
        with transcript_store.connect(self.root) as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                receipt = self._insert_sample_event(
                    con,
                    sample_id=sample_id,
                    event_type=event_type,
                    actor_id=actor_id,
                    authority_id=authority_id,
                    idempotency_key=idempotency_key,
                    created_at=created_at,
                    supersedes_event_id=supersedes_event_id,
                    payload=payload,
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
        return receipt

    def sample_state(self, sample_id: str) -> dict[str, Any]:
        sample = self.load_sample(sample_id)
        with transcript_store.connect(self.root) as con:
            heads = con.execute(
                """
                SELECT event.* FROM knowledge_voice_sample_events event
                WHERE event.sample_id = ? AND NOT EXISTS (
                    SELECT 1 FROM knowledge_voice_sample_events successor
                    WHERE successor.supersedes_event_id = event.id
                )
                """,
                (sample_id,),
            ).fetchall()
        if len(heads) > 1:
            raise RuntimeError("Voice sample event history has multiple heads.")
        head = heads[0] if heads else None
        state = str(sample["exclusion_state"])
        if head is not None:
            state = {
                "exclude": "excluded",
                "restore": "included",
                "delete": "deleted",
                "bind_person": state,
                "unbind_person": state,
            }[str(head["event_type"])]
        return {
            "sample_id": sample_id,
            "exclusion_state": state,
            "event_id": str(head["id"]) if head is not None else "",
        }

    def cluster_state(self, cluster_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            exists = con.execute(
                "SELECT 1 FROM knowledge_anonymous_cluster_versions "
                "WHERE cluster_id = ? LIMIT 1",
                (cluster_id,),
            ).fetchone()
            head = self._cluster_head(con, cluster_id)
        if exists is None:
            raise ValueError("Anonymous cluster is unknown.")
        status = "active"
        if head is not None:
            status = {
                "exclude": "excluded",
                "restore": "active",
                "delete": "deleted",
            }[str(head["action"])]
        return {
            "cluster_id": cluster_id,
            "status": status,
            "event_id": str(head["id"]) if head is not None else "",
        }

    def list_unclustered_samples(self) -> tuple[str, ...]:
        with transcript_store.connect(self.root) as con:
            sample_ids = [
                str(row["id"])
                for row in con.execute(
                    "SELECT id FROM knowledge_voice_samples ORDER BY id"
                ).fetchall()
            ]
            memberships = con.execute(
                """
                SELECT membership.sample_id, version.cluster_id,
                       membership.membership_state
                FROM knowledge_anonymous_cluster_memberships membership
                JOIN knowledge_anonymous_cluster_versions version
                  ON version.id = membership.cluster_version_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM knowledge_anonymous_cluster_versions successor
                    WHERE successor.predecessor_version_id = version.id
                )
                ORDER BY membership.sample_id
                """
            ).fetchall()
        currently_clustered = {
            str(row["sample_id"])
            for row in memberships
            if str(row["membership_state"]) in {"candidate", "confirmed"}
            and self.cluster_state(str(row["cluster_id"]))["status"] == "active"
        }
        return tuple(
            sample_id
            for sample_id in sample_ids
            if sample_id not in currently_clustered
            and self.sample_state(sample_id)["exclusion_state"] == "included"
        )

    def preview_effect(
        self,
        *,
        mode: str,
        target_type: str,
        target_id: str,
    ) -> dict[str, Any]:
        resolved = self._resolve_effect(
            mode=mode,
            target_type=target_type,
            target_id=target_id,
        )
        public = self._public_effect(resolved)
        return {**public, "preview_hash": _hash(public)}

    def apply_effect(
        self,
        *,
        preview: Mapping[str, Any],
        authority_id: str,
        idempotency_key: str,
        created_at: str,
    ) -> CustodyEffectReceipt:
        if not all(map(_text, (authority_id, idempotency_key, created_at))):
            raise ValueError("Custody effect authority and idempotency are required.")
        mode = _text(preview.get("mode"))
        target_type = _text(preview.get("target_type"))
        target_id = _text(preview.get("target_id"))
        preview_hash = _text(preview.get("preview_hash"))
        incoming_public = {
            key: value for key, value in preview.items() if key != "preview_hash"
        }
        if _hash(incoming_public) != preview_hash:
            raise ValueError("Biometric custody effect preview hash is invalid.")
        with transcript_store.connect(self.root) as con:
            existing_effect = con.execute(
                "SELECT * FROM knowledge_biometric_effect_receipts "
                "WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if existing_effect is not None:
            if (
                str(existing_effect["mode"]) != mode
                or str(existing_effect["target_type"]) != target_type
                or str(existing_effect["target_id"]) != target_id
                or str(existing_effect["preview_hash"]) != preview_hash
                or str(existing_effect["authority_id"]) != authority_id
            ):
                raise ValueError("Biometric custody effect idempotency drifted.")
            return CustodyEffectReceipt(
                effect_id=str(existing_effect["id"]),
                mode=mode,
                target_type=target_type,
                target_id=target_id,
                sample_event_ids=tuple(
                    json.loads(str(existing_effect["sample_event_ids_json"]))
                ),
                profile_event_ids=tuple(
                    json.loads(str(existing_effect["profile_event_ids_json"]))
                ),
                cluster_event_ids=tuple(
                    json.loads(str(existing_effect["cluster_event_ids_json"]))
                ),
                tombstone_id=str(existing_effect["tombstone_id"] or ""),
                status="unchanged",
            )
        resolved = self._resolve_effect(
            mode=mode,
            target_type=target_type,
            target_id=target_id,
        )
        public = self._public_effect(resolved)
        expected_hash = _hash(public)
        if (
            _text(preview.get("preview_hash")) != expected_hash
            or {key: preview.get(key) for key in public} != public
        ):
            raise ValueError("Biometric custody effect preview is stale.")
        effect_id = _stable_id("biometric-custody-effect", idempotency_key)
        sample_event_ids: list[str] = []
        profile_event_ids: list[str] = []
        cluster_event_ids: list[str] = []
        moved: list[tuple[Path, Path]] = []
        quarantine = self.private_root / "quarantine" / effect_id
        if mode == "delete" and resolved["private_refs"]:
            quarantine.mkdir(parents=True, mode=0o700, exist_ok=False)
            os.chmod(quarantine.parent, 0o700)
            os.chmod(quarantine, 0o700)
            try:
                for private_ref in resolved["private_refs"]:
                    source = self._validate_private_object(
                        private_ref["object_id"], private_ref["sha256"]
                    )
                    destination = quarantine / source.name
                    os.replace(source, destination)
                    moved.append((source, destination))
            except Exception:
                for source, destination in reversed(moved):
                    os.replace(destination, source)
                quarantine.rmdir()
                raise
        try:
            with transcript_store.connect(self.root) as con:
                con.execute("BEGIN IMMEDIATE")
                try:
                    for sample_id in resolved["sample_ids"]:
                        current = self._sample_head(con, sample_id)
                        receipt = self._insert_sample_event(
                            con,
                            sample_id=sample_id,
                            event_type="delete" if mode == "delete" else "exclude",
                            actor_id="custody_effect",
                            authority_id=authority_id,
                            idempotency_key=f"{idempotency_key}:sample:{sample_id}",
                            created_at=created_at,
                            supersedes_event_id=(
                                str(current["id"]) if current is not None else ""
                            ),
                            payload={
                                "effect_id": effect_id,
                                "target_type": target_type,
                                "target_id": target_id,
                            },
                        )
                        sample_event_ids.append(receipt.event_id)
                    for profile_id in resolved["profile_version_ids"]:
                        current = self._profile_head(con, profile_id)
                        action = (
                            "delete"
                            if mode == "delete" and target_type in {"profile", "person"}
                            else "invalidate"
                        )
                        receipt = self._insert_profile_event(
                            con,
                            profile_version_id=profile_id,
                            action=action,
                            reason_code=f"{mode}_{target_type}",
                            authority_id=authority_id,
                            idempotency_key=f"{idempotency_key}:profile:{profile_id}",
                            created_at=created_at,
                            supersedes_event_id=(
                                str(current["id"]) if current is not None else ""
                            ),
                        )
                        profile_event_ids.append(receipt.event_id)
                    if target_type == "cluster":
                        current = self._cluster_head(con, target_id)
                        cluster_event_id = self._insert_cluster_event(
                            con,
                            cluster_id=target_id,
                            action="delete" if mode == "delete" else "exclude",
                            authority_id=authority_id,
                            idempotency_key=f"{idempotency_key}:cluster:{target_id}",
                            created_at=created_at,
                            supersedes_event_id=(
                                str(current["id"]) if current is not None else ""
                            ),
                            payload={"effect_id": effect_id},
                        )
                        cluster_event_ids.append(cluster_event_id)
                    tombstone_id = ""
                    if mode == "delete":
                        tombstone_id = effect_id
                        tombstone_core = {
                            "target_type": target_type,
                            "target_id": target_id,
                            "preview_hash": expected_hash,
                            "deleted_object_hashes": resolved[
                                "deleted_object_hashes"
                            ],
                            "invalidated_ids": sorted(
                                resolved["profile_version_ids"]
                                + resolved["cluster_version_ids"]
                            ),
                            "backup_disposition": "exclude_from_future_backups",
                            "historical_backup_disposition": (
                                "expire_on_retention_schedule"
                            ),
                            "authority_id": authority_id,
                            "idempotency_key": idempotency_key,
                        }
                        con.execute(
                            """
                            INSERT INTO knowledge_biometric_deletion_tombstones (
                                id, target_type, target_id, preview_hash,
                                deleted_object_hashes_json, invalidated_ids_json,
                                backup_disposition,
                                historical_backup_disposition, authority_id,
                                idempotency_key, content_hash, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                tombstone_id,
                                target_type,
                                target_id,
                                expected_hash,
                                _json(resolved["deleted_object_hashes"]),
                                _json(tombstone_core["invalidated_ids"]),
                                tombstone_core["backup_disposition"],
                                tombstone_core[
                                    "historical_backup_disposition"
                                ],
                                authority_id,
                                idempotency_key,
                                _hash(tombstone_core),
                                created_at,
                            ),
                        )
                    effect_core = {
                        "mode": mode,
                        "target_type": target_type,
                        "target_id": target_id,
                        "preview_hash": expected_hash,
                        "sample_event_ids": sample_event_ids,
                        "profile_event_ids": profile_event_ids,
                        "cluster_event_ids": cluster_event_ids,
                        "tombstone_id": tombstone_id,
                        "authority_id": authority_id,
                        "idempotency_key": idempotency_key,
                    }
                    con.execute(
                        """
                        INSERT INTO knowledge_biometric_effect_receipts (
                            id, mode, target_type, target_id, preview_hash,
                            sample_event_ids_json, profile_event_ids_json,
                            cluster_event_ids_json, tombstone_id, authority_id,
                            idempotency_key, content_hash, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            effect_id,
                            mode,
                            target_type,
                            target_id,
                            expected_hash,
                            _json(sample_event_ids),
                            _json(profile_event_ids),
                            _json(cluster_event_ids),
                            tombstone_id or None,
                            authority_id,
                            idempotency_key,
                            _hash(effect_core),
                            created_at,
                        ),
                    )
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
        except Exception:
            for source, destination in reversed(moved):
                os.replace(destination, source)
            if quarantine.exists():
                quarantine.rmdir()
            raise
        for _, destination in moved:
            destination.unlink()
        if quarantine.exists():
            quarantine.rmdir()
            try:
                quarantine.parent.rmdir()
            except OSError:
                pass
        return CustodyEffectReceipt(
            effect_id=effect_id,
            mode=mode,
            target_type=target_type,
            target_id=target_id,
            sample_event_ids=tuple(sample_event_ids),
            profile_event_ids=tuple(profile_event_ids),
            cluster_event_ids=tuple(cluster_event_ids),
            tombstone_id=tombstone_id,
            status="applied",
        )

    def load_deletion_tombstone(self, tombstone_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT * FROM knowledge_biometric_deletion_tombstones WHERE id = ?",
                (tombstone_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Biometric deletion tombstone is unknown.")
        return {
            "id": str(row["id"]),
            "target_type": str(row["target_type"]),
            "target_id": str(row["target_id"]),
            "preview_hash": str(row["preview_hash"]),
            "deleted_object_hashes": json.loads(
                str(row["deleted_object_hashes_json"])
            ),
            "invalidated_ids": json.loads(str(row["invalidated_ids_json"])),
            "backup_disposition": str(row["backup_disposition"]),
            "historical_backup_disposition": str(
                row["historical_backup_disposition"]
            ),
            "authority_id": str(row["authority_id"]),
            "created_at": str(row["created_at"]),
        }

    def load_sample(self, sample_id: str) -> dict[str, Any]:
        with transcript_store.connect(self.root) as con:
            row = con.execute(
                "SELECT * FROM knowledge_voice_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Voice sample is unknown.")
        result = dict(row)
        result["quality"] = json.loads(str(row["quality_json"]))
        result["preparation_lineage"] = json.loads(
            str(row["preparation_lineage_json"])
        )
        result.pop("quality_json")
        result.pop("preparation_lineage_json")
        result.pop("private_object_id")
        return result

    def _resolve_effect(
        self,
        *,
        mode: str,
        target_type: str,
        target_id: str,
    ) -> dict[str, Any]:
        if mode not in {"exclude", "delete"}:
            raise ValueError("Biometric custody effect mode is invalid.")
        if target_type not in {"sample", "cluster", "profile", "recording", "person"}:
            raise ValueError("Biometric custody target type is invalid.")
        target_id = _text(target_id)
        if not target_id:
            raise ValueError("Biometric custody target ID is required.")
        with transcript_store.connect(self.root) as con:
            samples = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_voice_samples ORDER BY id"
                ).fetchall()
            ]
            profiles = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_voice_profile_versions ORDER BY id"
                ).fetchall()
            ]
            cluster_versions = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_anonymous_cluster_versions ORDER BY id"
                ).fetchall()
            ]
            memberships = [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM knowledge_anonymous_cluster_memberships "
                    "ORDER BY cluster_version_id, rank, sample_id"
                ).fetchall()
            ]
        if target_type == "sample":
            sample_ids = [row["id"] for row in samples if row["id"] == target_id]
        elif target_type == "recording":
            sample_ids = [
                row["id"] for row in samples if row["recording_id"] == target_id
            ]
        elif target_type == "person":
            sample_ids = [
                row["id"] for row in samples if row["person_id"] == target_id
            ]
        else:
            sample_ids = []
        profile_ids: list[str] = []
        if target_type == "profile":
            profile_ids = [row["id"] for row in profiles if row["id"] == target_id]
        else:
            for profile in profiles:
                allowlist_ids = {
                    item["sample_id"]
                    for item in json.loads(str(profile["sample_allowlist_json"]))
                }
                if (
                    allowlist_ids.intersection(sample_ids)
                    or target_type == "person"
                    and profile["person_id"] == target_id
                ):
                    profile_ids.append(str(profile["id"]))
        if target_type == "cluster":
            cluster_version_ids = [
                row["id"]
                for row in cluster_versions
                if row["cluster_id"] == target_id
            ]
        else:
            cluster_version_ids = sorted(
                {
                    str(row["cluster_version_id"])
                    for row in memberships
                    if row["sample_id"] in sample_ids
                }
            )
        exists = {
            "sample": bool(sample_ids),
            "recording": bool(sample_ids),
            "person": bool(sample_ids or profile_ids),
            "profile": bool(profile_ids),
            "cluster": bool(cluster_version_ids),
        }[target_type]
        if not exists:
            raise ValueError("Biometric custody effect target is unknown.")
        private_refs: list[dict[str, str]] = []
        if mode == "delete":
            private_refs.extend(
                {
                    "object_id": str(row["private_object_id"]),
                    "sha256": str(row["private_object_sha256"]),
                }
                for row in samples
                if row["id"] in sample_ids and row["private_object_id"]
            )
            private_refs.extend(
                {
                    "object_id": str(row["private_object_id"]),
                    "sha256": str(row["private_object_sha256"]),
                }
                for row in profiles
                if row["id"] in profile_ids and row["private_object_id"]
            )
        unique_refs = {
            (item["object_id"], item["sha256"]): item for item in private_refs
        }
        return {
            "schema_version": "transcribe-audio.biometric-custody-effect.v1",
            "mode": mode,
            "target_type": target_type,
            "target_id": target_id,
            "sample_ids": sorted(map(str, sample_ids)),
            "profile_version_ids": sorted(set(profile_ids)),
            "cluster_version_ids": sorted(set(cluster_version_ids)),
            "deleted_object_hashes": sorted(
                {item["sha256"] for item in unique_refs.values()}
            ),
            "backup_disposition": (
                "exclude_from_future_backups" if mode == "delete" else "unchanged"
            ),
            "private_refs": list(unique_refs.values()),
        }

    @staticmethod
    def _public_effect(resolved: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in resolved.items()
            if key != "private_refs"
        }

    @staticmethod
    def _sample_head(con: Any, sample_id: str) -> Any:
        rows = con.execute(
            """
            SELECT event.* FROM knowledge_voice_sample_events event
            WHERE event.sample_id = ? AND NOT EXISTS (
                SELECT 1 FROM knowledge_voice_sample_events successor
                WHERE successor.supersedes_event_id = event.id
            )
            """,
            (sample_id,),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("Voice sample event history has multiple heads.")
        return rows[0] if rows else None

    def _insert_sample_event(
        self,
        con: Any,
        *,
        sample_id: str,
        event_type: str,
        actor_id: str,
        authority_id: str,
        idempotency_key: str,
        created_at: str,
        supersedes_event_id: str,
        payload: Mapping[str, Any] | None,
    ) -> SampleEventReceipt:
        if event_type not in {
            "exclude",
            "restore",
            "bind_person",
            "unbind_person",
            "delete",
        } or not all(
            map(
                _text,
                (sample_id, actor_id, authority_id, idempotency_key, created_at),
            )
        ):
            raise ValueError("Voice sample event is invalid.")
        if con.execute(
            "SELECT 1 FROM knowledge_voice_samples WHERE id = ?", (sample_id,)
        ).fetchone() is None:
            raise ValueError("Voice sample is unknown.")
        core = {
            "sample_id": sample_id,
            "event_type": event_type,
            "payload": dict(payload or {}),
            "actor_id": actor_id,
            "authority_id": authority_id,
            "idempotency_key": idempotency_key,
            "supersedes_event_id": _text(supersedes_event_id),
        }
        content_hash = _hash(core)
        existing = con.execute(
            "SELECT id, content_hash FROM knowledge_voice_sample_events "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            if str(existing["content_hash"]) != content_hash:
                raise ValueError("Voice sample event idempotency drifted.")
            return SampleEventReceipt(
                str(existing["id"]), sample_id, event_type, "unchanged"
            )
        current = self._sample_head(con, sample_id)
        current_id = str(current["id"]) if current is not None else ""
        if _text(supersedes_event_id) != current_id:
            raise ValueError("Voice sample event must supersede its current head.")
        event_id = _stable_id("voice-sample-event", idempotency_key)
        con.execute(
            """
            INSERT INTO knowledge_voice_sample_events (
                id, sample_id, event_type, payload_json, actor_id,
                authority_id, idempotency_key, supersedes_event_id,
                content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                sample_id,
                event_type,
                _json(dict(payload or {})),
                actor_id,
                authority_id,
                idempotency_key,
                supersedes_event_id or None,
                content_hash,
                created_at,
            ),
        )
        return SampleEventReceipt(event_id, sample_id, event_type, "inserted")

    @staticmethod
    def _profile_head(con: Any, profile_version_id: str) -> Any:
        rows = con.execute(
            """
            SELECT event.* FROM knowledge_voice_profile_events event
            WHERE event.profile_version_id = ? AND NOT EXISTS (
                SELECT 1 FROM knowledge_voice_profile_events successor
                WHERE successor.supersedes_event_id = event.id
            )
            """,
            (profile_version_id,),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("Voice profile event history has multiple heads.")
        return rows[0] if rows else None

    def _insert_profile_event(
        self,
        con: Any,
        *,
        profile_version_id: str,
        action: str,
        reason_code: str,
        authority_id: str,
        idempotency_key: str,
        created_at: str,
        supersedes_event_id: str,
    ) -> ProfileEventReceipt:
        core = {
            "profile_version_id": profile_version_id,
            "action": action,
            "reason_code": reason_code,
            "authority_id": authority_id,
            "idempotency_key": idempotency_key,
            "supersedes_event_id": _text(supersedes_event_id),
        }
        content_hash = _hash(core)
        existing = con.execute(
            "SELECT id, content_hash FROM knowledge_voice_profile_events "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            if str(existing["content_hash"]) != content_hash:
                raise ValueError("Voice profile event idempotency drifted.")
            return ProfileEventReceipt(
                str(existing["id"]), profile_version_id, action, "unchanged"
            )
        current = self._profile_head(con, profile_version_id)
        current_id = str(current["id"]) if current is not None else ""
        if _text(supersedes_event_id) != current_id:
            raise ValueError("Voice profile event must supersede its current head.")
        event_id = _stable_id("voice-profile-event", idempotency_key)
        con.execute(
            """
            INSERT INTO knowledge_voice_profile_events (
                id, profile_version_id, action, reason_code, authority_id,
                idempotency_key, supersedes_event_id, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                profile_version_id,
                action,
                reason_code,
                authority_id,
                idempotency_key,
                supersedes_event_id or None,
                content_hash,
                created_at,
            ),
        )
        return ProfileEventReceipt(event_id, profile_version_id, action, "inserted")

    @staticmethod
    def _cluster_head(con: Any, cluster_id: str) -> Any:
        rows = con.execute(
            """
            SELECT event.* FROM knowledge_anonymous_cluster_events event
            WHERE event.cluster_id = ? AND NOT EXISTS (
                SELECT 1 FROM knowledge_anonymous_cluster_events successor
                WHERE successor.supersedes_event_id = event.id
            )
            """,
            (cluster_id,),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("Anonymous cluster event history has multiple heads.")
        return rows[0] if rows else None

    def _insert_cluster_event(
        self,
        con: Any,
        *,
        cluster_id: str,
        action: str,
        authority_id: str,
        idempotency_key: str,
        created_at: str,
        supersedes_event_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        core = {
            "cluster_id": cluster_id,
            "action": action,
            "payload": dict(payload),
            "authority_id": authority_id,
            "idempotency_key": idempotency_key,
            "supersedes_event_id": _text(supersedes_event_id),
        }
        content_hash = _hash(core)
        existing = con.execute(
            "SELECT id, content_hash FROM knowledge_anonymous_cluster_events "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            if str(existing["content_hash"]) != content_hash:
                raise ValueError("Anonymous cluster event idempotency drifted.")
            return str(existing["id"])
        current = self._cluster_head(con, cluster_id)
        current_id = str(current["id"]) if current is not None else ""
        if _text(supersedes_event_id) != current_id:
            raise ValueError("Anonymous cluster event must supersede its current head.")
        event_id = _stable_id("anonymous-cluster-event", idempotency_key)
        con.execute(
            """
            INSERT INTO knowledge_anonymous_cluster_events (
                id, cluster_id, action, payload_json, authority_id,
                idempotency_key, supersedes_event_id, content_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                cluster_id,
                action,
                _json(dict(payload)),
                authority_id,
                idempotency_key,
                supersedes_event_id or None,
                content_hash,
                created_at,
            ),
        )
        return event_id

    def _object_path(self, object_id: str) -> Path:
        object_id = _text(object_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", object_id):
            raise ValueError("Private object ID is invalid.")
        path = self.objects_root / object_id
        if path.parent.resolve() != self.objects_root.resolve():
            raise ValueError("Private object escaped the custody root.")
        return path

    def _validate_private_object(self, object_id: str, sha256: str) -> Path:
        path = self._object_path(object_id)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Private biometric object is unavailable.")
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("Private biometric object permissions are too broad.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha256:
            raise ValueError("Private biometric object hash drifted.")
        return path

    @staticmethod
    def _require_sha256(value: str, field_name: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{64}", _text(value)):
            raise ValueError(f"{field_name} must be a lowercase SHA-256.")
