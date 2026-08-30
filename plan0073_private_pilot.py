from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from mail_evidence_normalization import normalize_mail_address
from mail_relationship_contracts import THRESHOLDS, ZERO_EFFECTS


REQUEST_SCHEMA_VERSION = "transcribe-audio.plan0073-p5-preview-request.v1"
PREVIEW_SCHEMA_VERSION = "transcribe-audio.plan0073-p5-preview.v1"
APPROVAL_SCHEMA_VERSION = "transcribe-audio.plan0073-p5-approval.v1"
RUNTIME_RELATIVE_ROOT = "plan-0073/private-pilots"


class Plan0073PrivatePilotError(ValueError):
    """Raised when a Plan 0073 private-pilot preview is not exact and safe."""


def _canonical_hash(value: object) -> str:
    body = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Plan0073PrivatePilotError(f"{label} shape is not exact.")


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"placeholder", "unknown", "tbd"}:
        raise Plan0073PrivatePilotError(f"{label} must be explicit.")
    return text


def _timestamp(value: object, label: str) -> str:
    text = _required_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Plan0073PrivatePilotError(f"{label} must be an ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Plan0073PrivatePilotError(f"{label} must include a timezone.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _email(value: object, label: str) -> str:
    try:
        return normalize_mail_address(value)
    except ValueError as exc:
        raise Plan0073PrivatePilotError(
            f"{label} must be a normalized exact email address."
        ) from exc


def _is_redacted_fixture(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_is_redacted_fixture(item) for item in value.values())
    if isinstance(value, list):
        return any(_is_redacted_fixture(item) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return "redacted" in lowered or lowered.endswith(".test")
    return False


def build_private_pilot_cohort(
    queue_payload: Mapping[str, Any], people_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Join queued conversations to exact, non-shared local contact emails."""

    if not isinstance(queue_payload, Mapping) or not isinstance(
        queue_payload.get("items"), list
    ):
        raise Plan0073PrivatePilotError("Identity review queue payload is invalid.")
    if not isinstance(people_payload, Mapping) or not isinstance(
        people_payload.get("items"), list
    ):
        raise Plan0073PrivatePilotError("Contacts payload is invalid.")
    queue_items = queue_payload["items"]
    if not queue_items or len(queue_items) > THRESHOLDS["max_pilot_conversations"]:
        raise Plan0073PrivatePilotError(
            "P5 requires between 1 and 25 already-queued conversations."
        )

    addresses_by_document: dict[str, set[str]] = {}
    for person in people_payload["items"]:
        if not isinstance(person, Mapping):
            continue
        if (
            person.get("identity_kind") != "local_contact"
            or person.get("contact_class") != "person_candidate"
        ):
            continue
        emails: set[str] = set()
        for method in person.get("contact_methods") or []:
            if not isinstance(method, Mapping) or method.get("kind") != "email":
                continue
            emails.add(_email(method.get("value"), "contact email"))
        if not emails:
            continue
        for occurrence in person.get("calendar_occurrences") or []:
            if not isinstance(occurrence, Mapping):
                continue
            document_id = str(occurrence.get("document_id") or "").strip()
            if document_id:
                addresses_by_document.setdefault(document_id, set()).update(emails)

    cohort: list[dict[str, Any]] = []
    for item in queue_items:
        if not isinstance(item, Mapping):
            raise Plan0073PrivatePilotError("Identity review queue item is invalid.")
        display = item.get("display")
        if not isinstance(display, Mapping):
            raise Plan0073PrivatePilotError("Identity review queue display is missing.")
        document_id = _required_text(
            display.get("source_document_id"), "source_document_id"
        )
        addresses = sorted(addresses_by_document.get(document_id, set()))
        if not addresses:
            raise Plan0073PrivatePilotError(
                "A queued conversation has no exact non-shared contact email."
            )
        if len(addresses) > 8:
            raise Plan0073PrivatePilotError(
                "A queued conversation exceeds 8 exact contact emails."
            )
        cohort.append(
            {
                "queue_item_id": _required_text(
                    item.get("queue_item_id"), "queue_item_id"
                ),
                "conversation_id": _required_text(
                    item.get("conversation_id"), "conversation_id"
                ),
                "as_of": _timestamp(display.get("recorded_at"), "recorded_at"),
                "exact_addresses": addresses,
            }
        )
    cohort.sort(key=lambda item: (item["as_of"], item["conversation_id"]))
    if sum(len(item["exact_addresses"]) for item in cohort) > 64:
        raise Plan0073PrivatePilotError("P5 cohort exceeds 64 exact address queries.")
    return cohort


def _normalized_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise Plan0073PrivatePilotError("Preview request must be an object.")
    _exact_keys(
        request,
        {"schema_version", "source_scope", "retrieval", "budgets", "cohort"},
        "Preview request",
    )
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise Plan0073PrivatePilotError("Preview request schema version is unsupported.")

    source = request.get("source_scope")
    if not isinstance(source, Mapping):
        raise Plan0073PrivatePilotError("source_scope must be an object.")
    _exact_keys(
        source,
        {
            "service_profile",
            "source_profile_id",
            "provider_kind",
            "account_id",
            "tenant_id",
            "account_address",
            "namespace",
            "corpus_id",
            "capabilities",
            "required_tools",
            "mailbox_mutation",
            "corpus_operation_execution",
        },
        "source_scope",
    )
    normalized_source = {
        "service_profile": _required_text(source.get("service_profile"), "service_profile"),
        "source_profile_id": _required_text(source.get("source_profile_id"), "source_profile_id"),
        "provider_kind": _required_text(source.get("provider_kind"), "provider_kind"),
        "account_id": _required_text(source.get("account_id"), "account_id"),
        "tenant_id": _required_text(source.get("tenant_id"), "tenant_id"),
        "account_address": _email(source.get("account_address"), "account_address"),
        "namespace": _required_text(source.get("namespace"), "namespace"),
        "corpus_id": _required_text(source.get("corpus_id"), "corpus_id"),
        "capabilities": list(source.get("capabilities") or []),
        "required_tools": list(source.get("required_tools") or []),
        "mailbox_mutation": source.get("mailbox_mutation"),
        "corpus_operation_execution": source.get("corpus_operation_execution"),
    }
    if normalized_source != dict(source):
        raise Plan0073PrivatePilotError("source_scope is not normalized.")
    if (
        normalized_source["service_profile"] != "operator-lite"
        or normalized_source["provider_kind"] != "mail_receipts"
        or normalized_source["capabilities"] != ["mail_metadata_read"]
        or normalized_source["required_tools"] != ["search_mail"]
        or normalized_source["mailbox_mutation"] is not False
        or normalized_source["corpus_operation_execution"] is not False
    ):
        raise Plan0073PrivatePilotError(
            "P5 requires the operator-lite, metadata-only Mail Receipts surface."
        )

    retrieval = request.get("retrieval")
    expected_retrieval = {
        "query_mode": "exact_email_only",
        "include_body": False,
        "include_subject": False,
        "include_attachments": False,
        "hindsight_policy": "exclude_after_conversation_as_of",
    }
    if not isinstance(retrieval, Mapping) or dict(retrieval) != expected_retrieval:
        raise Plan0073PrivatePilotError("P5 retrieval must remain exact and metadata-only.")

    budgets = request.get("budgets")
    expected_budgets = {
        "max_conversations": THRESHOLDS["max_pilot_conversations"],
        "max_exact_address_queries": 64,
        "max_records_per_query": 25,
        "max_characters_per_query": THRESHOLDS["max_characters"],
        "max_calls_per_query": THRESHOLDS["max_calls"],
        "max_pages_per_query": THRESHOLDS["max_pages"],
        "max_latency_ms_per_query": THRESHOLDS["max_latency_ms"],
        "max_lookback_days": THRESHOLDS["max_lookback_days"],
        "max_retries_per_pilot": THRESHOLDS["max_retries"],
    }
    if not isinstance(budgets, Mapping) or dict(budgets) != expected_budgets:
        raise Plan0073PrivatePilotError("P5 budgets must match the frozen pilot bounds.")

    cohort = request.get("cohort")
    if not isinstance(cohort, list) or not cohort:
        raise Plan0073PrivatePilotError("P5 cohort must be a non-empty list.")
    if len(cohort) > expected_budgets["max_conversations"]:
        raise Plan0073PrivatePilotError("P5 cohort exceeds 25 conversations.")
    normalized_cohort: list[dict[str, Any]] = []
    for item in cohort:
        if not isinstance(item, Mapping):
            raise Plan0073PrivatePilotError("P5 cohort entries must be objects.")
        _exact_keys(
            item,
            {"queue_item_id", "conversation_id", "as_of", "exact_addresses"},
            "P5 cohort entry",
        )
        addresses = item.get("exact_addresses")
        if not isinstance(addresses, list) or not addresses:
            raise Plan0073PrivatePilotError("Each conversation needs an exact email query.")
        normalized_addresses = sorted(
            {_email(value, "exact_addresses") for value in addresses}
        )
        if normalized_addresses != addresses or len(normalized_addresses) > 8:
            raise Plan0073PrivatePilotError(
                "Conversation addresses must be normalized, unique, sorted, and at most 8."
            )
        normalized_cohort.append(
            {
                "queue_item_id": _required_text(item.get("queue_item_id"), "queue_item_id"),
                "conversation_id": _required_text(item.get("conversation_id"), "conversation_id"),
                "as_of": _timestamp(item.get("as_of"), "as_of"),
                "exact_addresses": normalized_addresses,
            }
        )
    normalized_cohort.sort(key=lambda item: (item["as_of"], item["conversation_id"]))
    if len({item["conversation_id"] for item in normalized_cohort}) != len(normalized_cohort):
        raise Plan0073PrivatePilotError("P5 conversation IDs must be unique.")
    if len({item["queue_item_id"] for item in normalized_cohort}) != len(normalized_cohort):
        raise Plan0073PrivatePilotError("P5 queue item IDs must be unique.")
    if (
        sum(len(item["exact_addresses"]) for item in normalized_cohort)
        > expected_budgets["max_exact_address_queries"]
    ):
        raise Plan0073PrivatePilotError("P5 cohort exceeds 64 exact address queries.")

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_scope": normalized_source,
        "retrieval": expected_retrieval,
        "budgets": expected_budgets,
        "cohort": normalized_cohort,
    }


def build_private_pilot_preview(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return one immutable, zero-read P5 preview for explicit approval."""

    normalized = _normalized_request(request)
    request_sha256 = _canonical_hash(normalized)
    preview_id = "plan0073-p5-" + request_sha256[:32]
    query_count = sum(len(item["exact_addresses"]) for item in normalized["cohort"])
    budgets = normalized["budgets"]
    fixture_only = _is_redacted_fixture(normalized)
    query_plan = []
    for item in normalized["cohort"]:
        for address in item["exact_addresses"]:
            query = {
                "queue_item_id": item["queue_item_id"],
                "conversation_id": item["conversation_id"],
                "as_of": item["as_of"],
                "exact_address": address,
            }
            query_id = "mail-query-plan-" + _canonical_hash(query)[:32]
            query_plan.append(
                {
                    "query_id": query_id,
                    **query,
                    "query_mode": "exact_email_only",
                    "include_body": False,
                    "max_records": budgets["max_records_per_query"],
                    "max_characters": budgets["max_characters_per_query"],
                }
            )
    relative_root = f"{RUNTIME_RELATIVE_ROOT}/{preview_id}"
    core: dict[str, Any] = {
        "schema_version": PREVIEW_SCHEMA_VERSION,
        "preview_id": preview_id,
        "request_sha256": request_sha256,
        "status": (
            "redacted_fixture_only" if fixture_only else "awaiting_explicit_approval"
        ),
        "request": normalized,
        "query_plan": query_plan,
        "counts": {
            "conversations": len(normalized["cohort"]),
            "exact_address_queries": query_count,
            "maximum_provider_calls": (
                query_count * budgets["max_calls_per_query"]
            ),
            "maximum_records": (
                query_count * budgets["max_records_per_query"]
            ),
        },
        "runtime_write_surface": {
            "root_kind": "user_scoped_transcribe_audio_state",
            "relative_root": relative_root,
            "preview": f"{relative_root}/preview.json",
            "approval": f"{relative_root}/approval.json",
            "query_receipts": f"{relative_root}/query-receipts/",
            "normalized_evidence": f"{relative_root}/normalized/",
            "hypotheses": f"{relative_root}/hypotheses/",
            "aggregate_validation": f"{relative_root}/aggregate-validation.json",
        },
        "action_vector": {
            "owned_corpus_read": False,
            "mailbox_or_provider_call": False,
            "runtime_write": False,
            "schema_migration": False,
            "deployment": False,
            "accepted_graph_write": False,
            "person_merge": False,
            "speaker_or_profile_effect": False,
            "graphiti_write": False,
        },
        "effects": dict(ZERO_EFFECTS),
        "approval": (
            {
                "required": False,
                "reason": "redacted_fixture_cannot_authorize_private_read",
            }
            if fixture_only
            else {
                "required": True,
                "schema_version": APPROVAL_SCHEMA_VERSION,
                "exact_phrase": f"APPROVE_PLAN_0073_P5:{preview_id}",
                "authorizes": "one_private_shadow_pilot_apply",
                "does_not_authorize": [
                    "mailbox_mutation",
                    "provider_write",
                    "accepted_graph_decision",
                    "person_merge",
                    "speaker_or_biometric_effect",
                    "deployment",
                    "graphiti_write",
                    "plan0073_p6",
                ],
            }
        ),
    }
    core["content_sha256"] = _canonical_hash(core)
    return core


def validate_private_pilot_approval(
    preview: Mapping[str, Any], approval: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an exact approval without performing or persisting the pilot."""

    if not isinstance(preview, Mapping):
        raise Plan0073PrivatePilotError("P5 preview must be an object.")
    preview_content = deepcopy(dict(preview))
    preview_sha256 = str(preview_content.pop("content_sha256", ""))
    if _canonical_hash(preview_content) != preview_sha256:
        raise Plan0073PrivatePilotError("P5 preview drifted after review.")
    if (
        preview.get("schema_version") != PREVIEW_SCHEMA_VERSION
        or preview.get("status") != "awaiting_explicit_approval"
        or not isinstance(preview.get("approval"), Mapping)
    ):
        raise Plan0073PrivatePilotError("P5 preview is not awaiting approval.")
    if not isinstance(approval, Mapping):
        raise Plan0073PrivatePilotError("P5 approval must be an object.")
    _exact_keys(
        approval,
        {
            "schema_version",
            "preview_content_sha256",
            "exact_phrase",
            "decision",
            "approved_at",
        },
        "P5 approval",
    )
    normalized = {
        "schema_version": approval.get("schema_version"),
        "preview_content_sha256": str(approval.get("preview_content_sha256") or ""),
        "exact_phrase": str(approval.get("exact_phrase") or ""),
        "decision": str(approval.get("decision") or ""),
        "approved_at": _timestamp(approval.get("approved_at"), "approved_at"),
    }
    expected_phrase = str(preview["approval"].get("exact_phrase") or "")
    if (
        normalized["schema_version"] != APPROVAL_SCHEMA_VERSION
        or normalized["preview_content_sha256"] != preview_sha256
        or normalized["exact_phrase"] != expected_phrase
        or normalized["decision"] != "approve"
    ):
        raise Plan0073PrivatePilotError("P5 approval does not bind the exact preview.")
    if normalized != dict(approval):
        raise Plan0073PrivatePilotError("P5 approval is not normalized.")
    return normalized
