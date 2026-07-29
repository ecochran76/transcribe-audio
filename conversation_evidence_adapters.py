from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5

from conversation_knowledge_evidence import (
    MAX_EVIDENCE_METADATA_CHARS,
    MAX_EVIDENCE_SNIPPET_CHARS,
    EvidenceSnapshotRecord,
)


_ADAPTER_NAMESPACE = UUID("1e8724ad-76e1-4d53-8d63-45248c9cd937")
_RAW_PROVIDER_BODY_KEYS = frozenset(
    {
        "body",
        "content",
        "full_body",
        "full_content",
        "full_text",
        "message_body",
        "raw",
        "raw_body",
        "raw_content",
    }
)
ADAPTER_FAILURE_REASON_CODES = frozenset(
    {
        "budget_exhausted",
        "provider_auth_failed",
        "provider_query_failed",
        "provider_response_invalid",
        "provider_unavailable",
        "unsupported_capability",
    }
)
ADAPTER_WARNING_CODES = frozenset(
    {
        "provider_characters_truncated",
        "provider_partial_result",
        "provider_records_truncated",
        "provider_scope_skipped",
        "provider_timestamp_missing",
    }
)
MAX_ADAPTER_FAILURE_DETAIL_CHARS = 500


@dataclass(frozen=True)
class AdapterSourceScope:
    source_profile_id: str
    provider_kind: str
    account_id: str
    tenant_id: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class BoundedProviderRecord:
    provider_record_id: str
    source_type: str
    capability: str
    snippet: str
    structured_metadata: dict[str, Any] = field(default_factory=dict)
    source_event_at: str = ""
    source_uri: str = ""
    source_record_id: str = ""
    independence_group_id: str = ""
    freshness_state: str = "current"
    expires_at: str = ""
    redaction: dict[str, Any] = field(default_factory=dict)
    truncation: dict[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_uuid(*parts: str) -> str:
    return str(uuid5(_ADAPTER_NAMESPACE, "\x1f".join(parts)))


def _timestamp(value: str, *, field_name: str) -> tuple[str, datetime]:
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")
    normalized = parsed.astimezone(timezone.utc)
    canonical = normalized.isoformat().replace("+00:00", "Z")
    return canonical, normalized


def _raw_body_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {
            str(key).strip().casefold()
            for key in value
            if str(key).strip().casefold() in _RAW_PROVIDER_BODY_KEYS
        }
        for nested in value.values():
            keys.update(_raw_body_keys(nested))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for nested in value:
            keys.update(_raw_body_keys(nested))
        return keys
    return set()


def _validate_control_metadata(value: dict[str, Any], *, field_name: str) -> None:
    if _raw_body_keys(value):
        raise ValueError(f"{field_name} cannot contain raw provider bodies.")
    if len(_canonical_json(value)) > MAX_EVIDENCE_METADATA_CHARS:
        raise ValueError(f"{field_name} exceeds the bounded character cap.")


def adapter_failure(
    *,
    adapter_id: str,
    scope: AdapterSourceScope,
    capability: str,
    reason_code: str,
    detail: str = "",
) -> dict[str, str]:
    if reason_code not in ADAPTER_FAILURE_REASON_CODES:
        raise ValueError("Unsupported adapter failure reason code.")
    if not adapter_id.strip():
        raise ValueError("adapter_id is required.")
    if capability not in scope.capabilities:
        raise ValueError("Failure capability is not in the source scope.")
    bounded_detail = str(detail or "").strip()
    if len(bounded_detail) > MAX_ADAPTER_FAILURE_DETAIL_CHARS:
        raise ValueError("Adapter failure detail exceeds its character cap.")
    return {
        "adapter_id": adapter_id,
        "source_profile_id": scope.source_profile_id,
        "provider_kind": scope.provider_kind,
        "account_id": scope.account_id,
        "tenant_id": scope.tenant_id,
        "capability": capability,
        "reason_code": reason_code,
        "detail": bounded_detail,
    }


def adapter_warning(code: str) -> str:
    if code not in ADAPTER_WARNING_CODES:
        raise ValueError("Unsupported adapter warning code.")
    return code


class EvidenceSnapshotNormalizer:
    def __init__(
        self,
        *,
        scope: AdapterSourceScope,
        allowed_source_types: tuple[str, ...],
        allowed_metadata_fields: tuple[str, ...],
    ) -> None:
        if not scope.source_profile_id.strip():
            raise ValueError("source_profile_id is required.")
        if not scope.provider_kind.strip():
            raise ValueError("provider_kind is required.")
        if not scope.capabilities:
            raise ValueError("At least one capability is required.")
        if not allowed_source_types:
            raise ValueError("At least one source type is required.")
        self.scope = scope
        self.allowed_source_types = frozenset(allowed_source_types)
        self.allowed_metadata_fields = frozenset(allowed_metadata_fields)

    def normalize(
        self,
        record: BoundedProviderRecord,
        *,
        as_of: str,
        retrieved_at: str,
    ) -> EvidenceSnapshotRecord:
        provider_record_id = record.provider_record_id.strip()
        if not provider_record_id or provider_record_id != record.provider_record_id:
            raise ValueError(
                "provider_record_id must be a non-empty stable provider identity."
            )
        if record.source_type not in self.allowed_source_types:
            raise ValueError("source_type is not allowlisted for this adapter.")
        if record.capability not in self.scope.capabilities:
            raise ValueError("capability is not allowlisted for this source scope.")
        metadata_fields = set(record.structured_metadata)
        if not metadata_fields <= self.allowed_metadata_fields:
            raise ValueError("structured metadata fields are not allowlisted.")
        if _raw_body_keys(record.structured_metadata):
            raise ValueError("structured metadata fields cannot contain raw bodies.")
        if len(record.snippet) > MAX_EVIDENCE_SNIPPET_CHARS:
            raise ValueError("snippet exceeds the bounded evidence character cap.")
        _validate_control_metadata(record.redaction, field_name="redaction")
        _validate_control_metadata(record.truncation, field_name="truncation")

        canonical_as_of, parsed_as_of = _timestamp(as_of, field_name="as_of")
        canonical_retrieved, parsed_retrieved = _timestamp(
            retrieved_at,
            field_name="retrieved_at",
        )
        source_event_at = ""
        parsed_source_event: datetime | None = None
        if record.source_event_at:
            source_event_at, parsed_source_event = _timestamp(
                record.source_event_at,
                field_name="source_event_at",
            )

        if parsed_source_event is None:
            temporal_class = "later_retrieved"
        elif parsed_source_event > parsed_as_of:
            temporal_class = "hindsight"
        elif parsed_retrieved > parsed_as_of:
            temporal_class = "later_retrieved"
        else:
            temporal_class = "contemporaneous"

        metadata = {
            **record.structured_metadata,
            "provider_record_id": provider_record_id,
        }
        if len(_canonical_json(metadata)) > MAX_EVIDENCE_METADATA_CHARS:
            raise ValueError(
                "structured metadata exceeds the bounded evidence character cap."
            )
        hash_payload = {
            "scope": {
                "source_profile_id": self.scope.source_profile_id,
                "provider_kind": self.scope.provider_kind,
                "account_id": self.scope.account_id,
                "tenant_id": self.scope.tenant_id,
            },
            "provider_record_id": provider_record_id,
            "source_type": record.source_type,
            "capability": record.capability,
            "snippet": record.snippet,
            "structured_metadata": metadata,
            "source_event_at": source_event_at,
            "source_uri": record.source_uri,
            "redaction": record.redaction,
            "truncation": record.truncation,
        }
        content_hash = _content_hash(hash_payload)
        evidence_id = _stable_uuid(
            "provider-evidence",
            self.scope.source_profile_id,
            self.scope.account_id,
            self.scope.tenant_id,
            provider_record_id,
            canonical_retrieved,
            content_hash,
        )
        independence_group_id = (
            record.independence_group_id
            or _stable_uuid(
                "provider-interaction",
                self.scope.source_profile_id,
                self.scope.account_id,
                self.scope.tenant_id,
                provider_record_id,
            )
        )
        return EvidenceSnapshotRecord(
            evidence_id=evidence_id,
            source_record_id=record.source_record_id,
            source_profile_id=self.scope.source_profile_id,
            provider_kind=self.scope.provider_kind,
            account_id=self.scope.account_id,
            tenant_id=self.scope.tenant_id,
            source_type=record.source_type,
            capability=record.capability,
            snippet=record.snippet,
            structured_metadata=metadata,
            source_event_at=source_event_at,
            observed_at=canonical_retrieved,
            retrieved_at=canonical_retrieved,
            temporal_class=temporal_class,
            source_uri=record.source_uri,
            content_hash=content_hash,
            independence_group_id=independence_group_id,
            freshness_state=record.freshness_state,
            expires_at=record.expires_at,
            redaction=dict(record.redaction),
            truncation=dict(record.truncation),
        )
