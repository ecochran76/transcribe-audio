from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from conversation_knowledge_evidence import EvidenceSnapshotRecord
from mail_relationship_contracts import validate_mail_artifact


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True)
class NormalizedMailEvidence:
    observations: tuple[dict[str, Any], ...]
    independence_groups: tuple[dict[str, Any], ...]
    input_watermark: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_mail_address(value: object) -> str:
    address = str(value or "").strip().casefold()
    if not _EMAIL_RE.fullmatch(address):
        raise ValueError("Mail address must be an exact email address.")
    return address


def classify_account_direction(
    *,
    from_addresses: Sequence[str],
    to_addresses: Sequence[str],
    cc_addresses: Sequence[str],
    account_address: str,
) -> str:
    senders = tuple(normalize_mail_address(value) for value in from_addresses)
    recipients = tuple(
        normalize_mail_address(value) for value in (*to_addresses, *cc_addresses)
    )
    account = normalize_mail_address(account_address)
    if len(senders) != 1:
        return "unknown"
    sender_is_account = senders[0] == account
    recipient_has_account = account in recipients
    if sender_is_account and recipient_has_account:
        return "internal"
    if sender_is_account:
        return "outbound"
    if recipient_has_account:
        return "inbound"
    return "external"


def classify_mail_temporal(
    source_event_at: str,
    *,
    as_of: str,
) -> tuple[str, str | None]:
    _, source_event = _timestamp(source_event_at, field_name="source_event_at")
    _, parsed_as_of = _timestamp(as_of, field_name="as_of")
    if source_event > parsed_as_of:
        return "hindsight", "temporal_after_as_of"
    return "contemporaneous", None


def mail_independence_key(
    *,
    message_ref_hash: str,
    thread_ref_hash: str,
    source_event_at: str,
    participants: Sequence[str],
) -> str:
    message_key = str(message_ref_hash or "").strip()
    if message_key:
        return _hash({"version": "provider-message-ref.v1", "value": message_key})
    fallback = {
        "version": "conservative-mail-fallback.v1",
        "thread_ref_hash": str(thread_ref_hash or "").strip(),
        "source_event_at": _timestamp(
            source_event_at,
            field_name="source_event_at",
        )[0],
        "participants": sorted(
            {normalize_mail_address(value) for value in participants}
        ),
    }
    if not fallback["thread_ref_hash"]:
        raise ValueError("Mail independence fallback requires a thread key.")
    return _hash(fallback)


def _timestamp(value: object, *, field_name: str) -> tuple[str, datetime]:
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z"), normalized


def _expected_scope(snapshot: EvidenceSnapshotRecord) -> dict[str, Any]:
    metadata = snapshot.structured_metadata
    return {
        "provider_kind": snapshot.provider_kind,
        "profile_id": snapshot.source_profile_id,
        "account_id": snapshot.account_id,
        "tenant_id": snapshot.tenant_id,
        "namespace": str(metadata.get("namespace") or ""),
        "corpus_id": str(metadata.get("corpus_id") or ""),
        "capabilities": [snapshot.capability],
    }


def _observation(
    snapshot: EvidenceSnapshotRecord,
    *,
    query_receipt: Mapping[str, Any],
    group_id: str,
) -> dict[str, Any]:
    metadata = snapshot.structured_metadata
    source_event_at, _ = _timestamp(
        snapshot.source_event_at,
        field_name="source_event_at",
    )
    as_of, _ = _timestamp(query_receipt["as_of"], field_name="as_of")
    retrieved_at, _ = _timestamp(
        snapshot.retrieved_at,
        field_name="retrieved_at",
    )
    temporal_class, excluded_reason = classify_mail_temporal(
        source_event_at,
        as_of=as_of,
    )
    source_scope = _expected_scope(snapshot)
    if source_scope != query_receipt["source_scope"]:
        raise ValueError("Mail evidence snapshot scope does not match its receipt.")
    if snapshot.provider_kind != "mail_receipts" or snapshot.capability != (
        "mail_metadata_read"
    ):
        raise ValueError("Mail evidence snapshot capability is unsupported.")
    if snapshot.snippet:
        raise ValueError("Mail evidence normalization does not accept message text.")
    observation_core = {
        "query_receipt_id": str(query_receipt["receipt_id"]),
        "source_scope": source_scope,
        "capability": snapshot.capability,
        "source_ref": {
            "evidence_id": str(metadata.get("evidence_id") or ""),
            "record_ref": str(metadata.get("record_ref") or ""),
            "message_ref_hash": str(metadata.get("message_ref_hash") or ""),
            "thread_ref_hash": str(metadata.get("thread_ref_hash") or ""),
        },
        "source_event_at": source_event_at,
        "retrieved_at": retrieved_at,
        "as_of": as_of,
        "temporal_class": temporal_class,
        "participants": {
            "from": list(metadata.get("from_addresses") or []),
            "to": list(metadata.get("to_addresses") or []),
            "cc": list(metadata.get("cc_addresses") or []),
        },
        "account_direction": str(metadata.get("account_direction") or "unknown"),
        "contact_ids_by_address": dict(
            metadata.get("contact_ids_by_address") or {}
        ),
        "signature_observations": [
            dict(value) for value in metadata.get("signature_observations") or []
        ],
        "independence_group_id": group_id,
        "redaction": {"body_retained": False},
        "truncation": {
            "snippet_characters": int(
                snapshot.truncation.get("snippet_characters", 0)
            )
        },
        "excluded_reason_code": excluded_reason,
    }
    observation = {
        "schema_version": "transcribe-audio.mail-observation.v1",
        "observation_id": "mail-observation-" + _hash(observation_core)[:32],
        **observation_core,
    }
    validate_mail_artifact("mail_observation", observation)
    return observation


def normalize_mail_evidence(
    snapshots: Sequence[EvidenceSnapshotRecord],
    *,
    query_receipt: Mapping[str, Any],
) -> NormalizedMailEvidence:
    receipt = validate_mail_artifact("mail_query_receipt", query_receipt)
    result_hashes = [snapshot.content_hash for snapshot in snapshots]
    if sorted(result_hashes) != sorted(receipt["result_hashes"]):
        raise ValueError("Mail evidence snapshots do not match the query receipt.")

    grouped_snapshots: dict[str, list[EvidenceSnapshotRecord]] = defaultdict(list)
    for snapshot in snapshots:
        interaction_key = str(snapshot.independence_group_id or "").strip()
        if not interaction_key:
            raise ValueError("Mail evidence requires an independence key.")
        grouped_snapshots[interaction_key].append(snapshot)

    observations_by_id: dict[str, dict[str, Any]] = {}
    observation_sources: dict[str, str] = {}
    observation_groups: dict[str, list[str]] = defaultdict(list)
    group_threads: dict[str, set[str]] = defaultdict(set)
    group_ids = {
        key: "mail-interaction-" + _hash({"interaction_key": key})[:32]
        for key in grouped_snapshots
    }
    for interaction_key, members in grouped_snapshots.items():
        group_id = group_ids[interaction_key]
        for snapshot in members:
            observation = _observation(
                snapshot,
                query_receipt=receipt,
                group_id=group_id,
            )
            observation_id = observation["observation_id"]
            observations_by_id[observation_id] = observation
            observation_groups[group_id].append(observation_id)
            observation_sources[observation_id] = str(
                snapshot.structured_metadata.get("source_key_hash") or ""
            )
            group_threads[group_id].add(
                observation["source_ref"]["thread_ref_hash"]
            )

    independence_groups: list[dict[str, Any]] = []
    for group_id in sorted(observation_groups):
        members = sorted(set(observation_groups[group_id]))
        threads = group_threads[group_id]
        if len(threads) != 1:
            raise ValueError(
                "One mail interaction cannot span multiple independent threads."
            )
        duplicate_count = len(members) - 1
        group_core = {
            "group_id": group_id,
            "interaction_key_version": "mail-interaction-key.v1",
            "independent_thread_key": next(iter(threads)),
            "member_observation_ids": members,
            "duplicate_count": duplicate_count,
            "source_count": len(
                {observation_sources[observation_id] for observation_id in members}
            ),
            "reason_code": (
                "duplicate_interaction" if duplicate_count else None
            ),
        }
        group = {
            "schema_version": "transcribe-audio.mail-independence-group.v1",
            **group_core,
            "content_hash": _hash(group_core),
        }
        validate_mail_artifact("mail_independence_group", group)
        independence_groups.append(group)

    observations = tuple(
        sorted(
            observations_by_id.values(),
            key=lambda item: (item["source_event_at"], item["observation_id"]),
        )
    )
    groups = tuple(independence_groups)
    watermark = _hash(
        {
            "observations": observations,
            "independence_groups": groups,
        }
    )
    return NormalizedMailEvidence(
        observations=observations,
        independence_groups=groups,
        input_watermark=watermark,
    )
