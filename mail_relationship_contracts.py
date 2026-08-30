from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = "transcribe-audio.mail-relationship-contract.v1"

ARTIFACT_SCHEMAS = {
    "mail_query_receipt": "transcribe-audio.mail-query-receipt.v1",
    "mail_observation": "transcribe-audio.mail-observation.v1",
    "mail_independence_group": "transcribe-audio.mail-independence-group.v1",
    "mail_relationship_hypothesis": (
        "transcribe-audio.mail-relationship-hypothesis.v1"
    ),
}

THRESHOLDS = {
    "min_correspondence_threads": 2,
    "min_coparticipant_threads": 2,
    "max_records": 250,
    "max_characters": 1,
    "max_calls": 4,
    "max_latency_ms": 30_000,
    "max_lookback_days": 365,
    "max_retries": 1,
    "max_pilot_conversations": 25,
    "max_page_size": 100,
    "max_pages": 10,
}

REASON_CODES = (
    "excluded_shared_address",
    "excluded_role_address",
    "excluded_mailing_list",
    "excluded_automated_sender",
    "excluded_unresolved_contact",
    "temporal_after_as_of",
    "provider_response_invalid",
    "duplicate_interaction",
    "budget_exhausted",
    "partial_source_failure",
    "namespace_scope_mismatch",
    "unsupported_capability",
    "opaque_reference_unavailable",
)

ZERO_EFFECTS = {
    "accepted_relationships": 0,
    "accepted_roles": 0,
    "provider_writes": 0,
    "person_merges": 0,
    "speaker_assignments": 0,
    "biometric_effects": 0,
    "graphiti_writes": 0,
}

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_RAW_BODY_KEYS = frozenset(
    {
        "body",
        "body_text",
        "body_html",
        "raw_body",
        "raw_content",
        "attachment_content",
        "quoted_reply",
    }
)


class MailRelationshipContractError(ValueError):
    """Raised when a Plan 0073 portable artifact violates its contract."""


def _timestamp(value: object, *, field_name: str) -> datetime:
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MailRelationshipContractError(
            f"{field_name} must be an ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MailRelationshipContractError(
            f"{field_name} must include a timezone."
        )
    return parsed.astimezone(timezone.utc)


def _raw_body_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        keys = {
            str(key).strip().casefold()
            for key in value
            if str(key).strip().casefold() in _RAW_BODY_KEYS
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


def _require_fields(
    payload: Mapping[str, Any], fields: tuple[str, ...], *, kind: str
) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise MailRelationshipContractError(
            f"{kind} is missing required fields: {', '.join(missing)}."
        )


def _validate_source_scope(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MailRelationshipContractError("source_scope must be an object.")
    scope = dict(value)
    required = (
        "provider_kind",
        "profile_id",
        "account_id",
        "tenant_id",
        "namespace",
        "corpus_id",
        "capabilities",
    )
    _require_fields(scope, required, kind="source_scope")
    if scope["provider_kind"] != "mail_receipts":
        raise MailRelationshipContractError(
            "Mail evidence requires provider_kind mail_receipts."
        )
    for field in required[:-1]:
        if not str(scope[field] or "").strip():
            raise MailRelationshipContractError(
                f"source_scope {field} must be explicit."
            )
    capabilities = scope["capabilities"]
    if not isinstance(capabilities, list) or capabilities != ["mail_metadata_read"]:
        raise MailRelationshipContractError(
            "Mail evidence scope must allow only mail_metadata_read."
        )
    return scope


def _validate_query_receipt(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        (
            "receipt_id",
            "request_hash",
            "source_scope",
            "capability",
            "query_mode",
            "exact_addresses",
            "as_of",
            "lookback_start",
            "budgets",
            "status",
            "counts",
            "warnings",
            "failures",
            "result_hashes",
            "created_at",
        ),
        kind="mail_query_receipt",
    )
    scope = _validate_source_scope(payload["source_scope"])
    if payload["capability"] not in scope["capabilities"]:
        raise MailRelationshipContractError("Query capability is outside source scope.")
    if payload["query_mode"] != "exact_email_only":
        raise MailRelationshipContractError("Mail queries must be exact_email_only.")
    addresses = payload["exact_addresses"]
    if (
        not isinstance(addresses, list)
        or not addresses
        or any(
            not isinstance(address, str)
            or address != address.strip().casefold()
            or not _EMAIL_RE.fullmatch(address)
            for address in addresses
        )
        or len(addresses) != len(set(addresses))
    ):
        raise MailRelationshipContractError(
            "Mail queries require unique normalized exact email addresses."
        )
    if not _SHA256_RE.fullmatch(str(payload["request_hash"] or "")):
        raise MailRelationshipContractError("request_hash must be a lowercase SHA-256.")
    result_hashes = payload["result_hashes"]
    if not isinstance(result_hashes, list) or any(
        not _SHA256_RE.fullmatch(str(value or "")) for value in result_hashes
    ):
        raise MailRelationshipContractError(
            "result_hashes must contain lowercase SHA-256 values."
        )
    as_of = _timestamp(payload["as_of"], field_name="as_of")
    lookback = _timestamp(payload["lookback_start"], field_name="lookback_start")
    _timestamp(payload["created_at"], field_name="created_at")
    if lookback > as_of or (as_of - lookback).days > THRESHOLDS["max_lookback_days"]:
        raise MailRelationshipContractError("Mail query lookback exceeds its bound.")
    budgets = payload["budgets"]
    budget_fields = {
        "max_records": "max_records",
        "max_characters": "max_characters",
        "max_calls": "max_calls",
        "max_latency_ms": "max_latency_ms",
        "max_pages": "max_pages",
    }
    if not isinstance(budgets, Mapping) or set(budgets) != set(budget_fields):
        raise MailRelationshipContractError("Mail query budgets are incomplete.")
    for field, threshold in budget_fields.items():
        value = budgets[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > THRESHOLDS[threshold]
        ):
            raise MailRelationshipContractError(
                f"Mail query budget {field} is outside its frozen bound."
            )
    if payload["status"] not in {"complete", "partial", "unavailable"}:
        raise MailRelationshipContractError("Mail query status is invalid.")
    counts = payload["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {
        "selected",
        "excluded",
        "truncated",
        "provider_writes",
    }:
        raise MailRelationshipContractError("Mail query counts are incomplete.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise MailRelationshipContractError("Mail query counts must be nonnegative.")
    if counts["provider_writes"] != 0:
        raise MailRelationshipContractError("Mail query receipts require zero writes.")
    if not isinstance(payload["warnings"], list) or not isinstance(
        payload["failures"], list
    ):
        raise MailRelationshipContractError(
            "Mail query warnings and failures must be lists."
        )
    if payload["status"] != "complete" and not payload["failures"]:
        raise MailRelationshipContractError(
            "Partial or unavailable mail queries require a failure."
        )


def _normalized_addresses(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise MailRelationshipContractError(f"{field_name} must be a list.")
    if any(
        not isinstance(address, str)
        or address != address.strip().casefold()
        or not _EMAIL_RE.fullmatch(address)
        for address in value
    ) or len(value) != len(set(value)):
        raise MailRelationshipContractError(
            f"{field_name} must contain unique normalized email addresses."
        )
    return value


def _validate_mail_observation(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        (
            "observation_id",
            "query_receipt_id",
            "source_scope",
            "capability",
            "source_ref",
            "source_event_at",
            "retrieved_at",
            "as_of",
            "temporal_class",
            "participants",
            "account_direction",
            "contact_ids_by_address",
            "signature_observations",
            "independence_group_id",
            "redaction",
            "truncation",
            "excluded_reason_code",
        ),
        kind="mail_observation",
    )
    for field in ("observation_id", "query_receipt_id", "independence_group_id"):
        if not str(payload[field] or "").strip():
            raise MailRelationshipContractError(f"{field} must be explicit.")
    scope = _validate_source_scope(payload["source_scope"])
    if payload["capability"] not in scope["capabilities"]:
        raise MailRelationshipContractError(
            "Observation capability is outside source scope."
        )
    source_ref = payload["source_ref"]
    if not isinstance(source_ref, Mapping) or set(source_ref) != {
        "evidence_id",
        "record_ref",
        "message_ref_hash",
        "thread_ref_hash",
    }:
        raise MailRelationshipContractError(
            "Mail observation source_ref is incomplete."
        )
    if not str(source_ref["evidence_id"] or "").strip() or not str(
        source_ref["record_ref"] or ""
    ).strip():
        raise MailRelationshipContractError(
            "Mail observation requires opaque evidence and record references."
        )
    for field in ("message_ref_hash", "thread_ref_hash"):
        if not _SHA256_RE.fullmatch(str(source_ref[field] or "")):
            raise MailRelationshipContractError(
                f"source_ref {field} must be a lowercase SHA-256."
            )
    source_event_at = _timestamp(
        payload["source_event_at"], field_name="source_event_at"
    )
    _timestamp(payload["retrieved_at"], field_name="retrieved_at")
    as_of = _timestamp(payload["as_of"], field_name="as_of")
    if payload["temporal_class"] not in {"contemporaneous", "hindsight"}:
        raise MailRelationshipContractError(
            "Mail observation temporal_class is invalid."
        )
    participants = payload["participants"]
    if not isinstance(participants, Mapping) or set(participants) != {
        "from",
        "to",
        "cc",
    }:
        raise MailRelationshipContractError(
            "Mail observation participants must contain from, to, and cc."
        )
    sender = _normalized_addresses(participants["from"], field_name="participants.from")
    recipients = _normalized_addresses(participants["to"], field_name="participants.to")
    copied = _normalized_addresses(participants["cc"], field_name="participants.cc")
    if len(sender) != 1 or not recipients:
        raise MailRelationshipContractError(
            "Mail observation requires one sender and at least one recipient."
        )
    all_addresses = set(sender + recipients + copied)
    direction = payload["account_direction"]
    if direction not in {"inbound", "outbound", "internal", "external", "unknown"}:
        raise MailRelationshipContractError(
            "Mail observation account_direction is invalid."
        )
    contact_ids = payload["contact_ids_by_address"]
    if not isinstance(contact_ids, Mapping) or any(
        address not in all_addresses or not str(contact_id or "").strip()
        for address, contact_id in contact_ids.items()
    ):
        raise MailRelationshipContractError(
            "Mail observation contact joins must use exact participant addresses."
        )
    signature_observations = payload["signature_observations"]
    if not isinstance(signature_observations, list):
        raise MailRelationshipContractError(
            "signature_observations must be a list."
        )
    for observation in signature_observations:
        if not isinstance(observation, Mapping) or set(observation) != {
            "address",
            "title",
            "organization",
            "department",
            "observed_at",
        }:
            raise MailRelationshipContractError(
                "Structured signature observation is invalid."
            )
        if observation["address"] not in all_addresses:
            raise MailRelationshipContractError(
                "Structured signature address is not an exact participant."
            )
        _timestamp(observation["observed_at"], field_name="signature observed_at")
        if not any(
            str(observation[field] or "").strip()
            for field in ("title", "organization", "department")
        ):
            raise MailRelationshipContractError(
                "Structured signature observation must contain a declared value."
            )
    if not isinstance(payload["redaction"], Mapping) or not isinstance(
        payload["truncation"], Mapping
    ):
        raise MailRelationshipContractError(
            "Mail observation redaction and truncation must be objects."
        )
    excluded_reason = payload["excluded_reason_code"]
    if excluded_reason is not None and excluded_reason not in REASON_CODES:
        raise MailRelationshipContractError(
            "Mail observation exclusion reason is unsupported."
        )
    if source_event_at > as_of:
        if (
            payload["temporal_class"] != "hindsight"
            or excluded_reason != "temporal_after_as_of"
        ):
            raise MailRelationshipContractError(
                "Mail observation after as_of must be excluded as hindsight."
            )
    elif payload["temporal_class"] != "contemporaneous":
        raise MailRelationshipContractError(
            "Mail observation at or before as_of must be contemporaneous."
        )


def _validate_independence_group(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        (
            "group_id",
            "interaction_key_version",
            "independent_thread_key",
            "member_observation_ids",
            "duplicate_count",
            "source_count",
            "reason_code",
            "content_hash",
        ),
        kind="mail_independence_group",
    )
    if not str(payload["group_id"] or "").strip():
        raise MailRelationshipContractError("group_id must be explicit.")
    if payload["interaction_key_version"] != "mail-interaction-key.v1":
        raise MailRelationshipContractError(
            "Mail independence interaction key version is invalid."
        )
    for field in ("independent_thread_key", "content_hash"):
        if not _SHA256_RE.fullmatch(str(payload[field] or "")):
            raise MailRelationshipContractError(
                f"{field} must be a lowercase SHA-256."
            )
    members = payload["member_observation_ids"]
    if (
        not isinstance(members, list)
        or not members
        or any(not str(member or "").strip() for member in members)
        or len(members) != len(set(members))
    ):
        raise MailRelationshipContractError(
            "Mail independence groups require unique observation members."
        )
    duplicate_count = payload["duplicate_count"]
    source_count = payload["source_count"]
    if (
        isinstance(duplicate_count, bool)
        or not isinstance(duplicate_count, int)
        or duplicate_count != len(members) - 1
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 1
        or source_count > len(members)
    ):
        raise MailRelationshipContractError(
            "Mail independence duplicate or source accounting is invalid."
        )
    expected_reason = "duplicate_interaction" if duplicate_count else None
    if payload["reason_code"] != expected_reason:
        raise MailRelationshipContractError(
            "Mail independence duplicate reason does not match its members."
        )


def _validate_relationship_hypothesis(payload: Mapping[str, Any]) -> None:
    _require_fields(
        payload,
        (
            "hypothesis_id",
            "hypothesis_kind",
            "relationship_type",
            "directionality",
            "subject_contact_id",
            "counterpart_type",
            "counterpart_id",
            "counterpart_label",
            "evidence_observation_ids",
            "evidence_independence_group_ids",
            "observation_count",
            "independent_thread_count",
            "first_observed_at",
            "last_observed_at",
            "status",
            "basis",
            "why_not_accepted",
            "conflicts",
            "effect_counts",
        ),
        kind="mail_relationship_hypothesis",
    )
    for field in (
        "hypothesis_id",
        "subject_contact_id",
        "counterpart_id",
        "counterpart_label",
        "basis",
        "why_not_accepted",
    ):
        if not str(payload[field] or "").strip():
            raise MailRelationshipContractError(f"{field} must be explicit.")
    kinds = {
        "sent_mail",
        "correspondence",
        "thread_coparticipation",
        "contextual_role",
        "affiliation",
    }
    relationship_types = {
        "SENT_MAIL_TO",
        "CORRESPONDED_WITH",
        "MAIL_THREAD_COPARTICIPANT_WITH",
        "HAS_CONTEXTUAL_ROLE",
        "AFFILIATED_WITH",
    }
    if payload["hypothesis_kind"] not in kinds:
        raise MailRelationshipContractError("Mail hypothesis kind is invalid.")
    if payload["relationship_type"] not in relationship_types:
        raise MailRelationshipContractError(
            "Mail hypothesis relationship type is invalid."
        )
    if payload["directionality"] not in {"directional", "symmetric"}:
        raise MailRelationshipContractError(
            "Mail hypothesis directionality is invalid."
        )
    expected_direction = (
        "symmetric"
        if payload["hypothesis_kind"]
        in {"correspondence", "thread_coparticipation"}
        else "directional"
    )
    if payload["directionality"] != expected_direction:
        raise MailRelationshipContractError(
            "Mail hypothesis directionality does not match its kind."
        )
    if payload["counterpart_type"] not in {
        "contact_candidate",
        "organization",
        "contextual_role",
    }:
        raise MailRelationshipContractError(
            "Mail hypothesis counterpart type is invalid."
        )
    observation_ids = payload["evidence_observation_ids"]
    group_ids = payload["evidence_independence_group_ids"]
    for value, field_name in (
        (observation_ids, "evidence_observation_ids"),
        (group_ids, "evidence_independence_group_ids"),
    ):
        if (
            not isinstance(value, list)
            or not value
            or any(not str(item or "").strip() for item in value)
            or len(value) != len(set(value))
        ):
            raise MailRelationshipContractError(
                f"{field_name} must contain unique evidence identities."
            )
    observation_count = payload["observation_count"]
    independent_threads = payload["independent_thread_count"]
    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or observation_count < len(observation_ids)
        or isinstance(independent_threads, bool)
        or not isinstance(independent_threads, int)
        or independent_threads < 1
        or independent_threads > observation_count
    ):
        raise MailRelationshipContractError(
            "Mail hypothesis evidence counts are invalid."
        )
    first = _timestamp(payload["first_observed_at"], field_name="first_observed_at")
    last = _timestamp(payload["last_observed_at"], field_name="last_observed_at")
    if first > last:
        raise MailRelationshipContractError(
            "Mail hypothesis observation interval is reversed."
        )
    if payload["status"] != "proposed":
        raise MailRelationshipContractError(
            "Mail hypotheses must remain proposed."
        )
    if not isinstance(payload["conflicts"], list):
        raise MailRelationshipContractError("Mail hypothesis conflicts must be a list.")
    if payload["effect_counts"] != ZERO_EFFECTS:
        raise MailRelationshipContractError(
            "Mail hypotheses must report zero accepted effects."
        )


def contract() -> dict[str, Any]:
    """Return the immutable Plan 0073 deterministic contract catalog."""
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifact_schemas": dict(ARTIFACT_SCHEMAS),
        "thresholds": dict(THRESHOLDS),
        "reason_codes": list(REASON_CODES),
        "effects": dict(ZERO_EFFECTS),
    }


def validate_mail_artifact(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one portable Plan 0073 artifact at the stable host seam."""
    if kind not in ARTIFACT_SCHEMAS:
        raise MailRelationshipContractError(f"Unknown mail artifact kind: {kind}.")
    if not isinstance(payload, Mapping):
        raise MailRelationshipContractError("Mail artifact must be an object.")
    if payload.get("schema_version") != ARTIFACT_SCHEMAS[kind]:
        raise MailRelationshipContractError(
            f"{kind} schema_version must be {ARTIFACT_SCHEMAS[kind]}."
        )
    forbidden = sorted(_raw_body_keys(payload))
    if forbidden:
        raise MailRelationshipContractError(
            "Mail artifact contains prohibited raw content fields: "
            + ", ".join(forbidden)
            + "."
        )
    if kind == "mail_query_receipt":
        _validate_query_receipt(payload)
    elif kind == "mail_observation":
        _validate_mail_observation(payload)
    elif kind == "mail_independence_group":
        _validate_independence_group(payload)
    elif kind == "mail_relationship_hypothesis":
        _validate_relationship_hypothesis(payload)
    return dict(payload)
