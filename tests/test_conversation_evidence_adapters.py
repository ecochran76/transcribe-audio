from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conversation_evidence_adapters
from conversation_knowledge_evidence import (
    MAX_EVIDENCE_METADATA_CHARS,
    MAX_EVIDENCE_SNIPPET_CHARS,
)


def _normalizer() -> conversation_evidence_adapters.EvidenceSnapshotNormalizer:
    return conversation_evidence_adapters.EvidenceSnapshotNormalizer(
        scope=conversation_evidence_adapters.AdapterSourceScope(
            source_profile_id="gws-default",
            provider_kind="gws",
            account_id="",
            tenant_id="",
            capabilities=("contacts", "mail"),
        ),
        allowed_source_types=("gws_contact", "gmail_message"),
        allowed_metadata_fields=(
            "display_name",
            "email",
            "thread_id",
        ),
    )


def test_normalize_creates_stable_scoped_later_retrieved_snapshot() -> None:
    normalizer = _normalizer()
    record = conversation_evidence_adapters.BoundedProviderRecord(
        provider_record_id="people/c123",
        source_type="gws_contact",
        capability="contacts",
        snippet="Ada Example <ada@example.com>",
        structured_metadata={
            "display_name": "Ada Example",
            "email": "ada@example.com",
        },
        source_uri="gws:people/c123",
    )

    first = normalizer.normalize(
        record,
        as_of="2026-07-26T14:00:00Z",
        retrieved_at="2026-07-29T12:00:00Z",
    )
    second = normalizer.normalize(
        record,
        as_of="2026-07-26T14:00:00Z",
        retrieved_at="2026-07-29T12:00:00Z",
    )

    assert first == second
    assert str(UUID(first.evidence_id)) == first.evidence_id
    assert first.source_profile_id == "gws-default"
    assert first.provider_kind == "gws"
    assert first.account_id == ""
    assert first.tenant_id == ""
    assert first.temporal_class == "later_retrieved"
    assert first.structured_metadata == {
        "display_name": "Ada Example",
        "email": "ada@example.com",
        "provider_record_id": "people/c123",
    }


def test_normalizer_rejects_missing_required_scope_identity() -> None:
    with pytest.raises(ValueError, match="source_profile_id"):
        conversation_evidence_adapters.EvidenceSnapshotNormalizer(
            scope=conversation_evidence_adapters.AdapterSourceScope(
                source_profile_id="",
                provider_kind="gws",
                account_id="",
                tenant_id="",
                capabilities=("contacts",),
            ),
            allowed_source_types=("gws_contact",),
            allowed_metadata_fields=("email",),
        )


@pytest.mark.parametrize(
    ("record", "message"),
    (
        (
            conversation_evidence_adapters.BoundedProviderRecord(
                provider_record_id="",
                source_type="gws_contact",
                capability="contacts",
                snippet="Ada Example",
            ),
            "provider_record_id",
        ),
        (
            conversation_evidence_adapters.BoundedProviderRecord(
                provider_record_id="people/c123",
                source_type="unknown_record",
                capability="contacts",
                snippet="Ada Example",
            ),
            "source_type",
        ),
        (
            conversation_evidence_adapters.BoundedProviderRecord(
                provider_record_id="people/c123",
                source_type="gws_contact",
                capability="drive",
                snippet="Ada Example",
            ),
            "capability",
        ),
        (
            conversation_evidence_adapters.BoundedProviderRecord(
                provider_record_id="people/c123",
                source_type="gws_contact",
                capability="contacts",
                snippet="Ada Example",
                structured_metadata={"body": "unbounded raw body"},
            ),
            "metadata fields",
        ),
        (
            conversation_evidence_adapters.BoundedProviderRecord(
                provider_record_id="people/c123",
                source_type="gws_contact",
                capability="contacts",
                snippet="Ada Example",
                structured_metadata={
                    "display_name": {"raw_body": "unbounded raw body"}
                },
            ),
            "raw bodies",
        ),
    ),
)
def test_normalize_rejects_records_outside_allowlists(
    record: conversation_evidence_adapters.BoundedProviderRecord,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _normalizer().normalize(
            record,
            as_of="2026-07-26T14:00:00Z",
            retrieved_at="2026-07-29T12:00:00Z",
        )


def test_failure_and_warning_schemas_are_fixed_and_bounded() -> None:
    scope = _normalizer().scope

    assert conversation_evidence_adapters.adapter_failure(
        adapter_id="gws-evidence",
        scope=scope,
        capability="mail",
        reason_code="provider_query_failed",
        detail="TimeoutError",
    ) == {
        "adapter_id": "gws-evidence",
        "source_profile_id": "gws-default",
        "provider_kind": "gws",
        "account_id": "",
        "tenant_id": "",
        "capability": "mail",
        "reason_code": "provider_query_failed",
        "detail": "TimeoutError",
    }
    assert (
        conversation_evidence_adapters.adapter_warning(
            "provider_partial_result"
        )
        == "provider_partial_result"
    )
    with pytest.raises(ValueError, match="failure reason"):
        conversation_evidence_adapters.adapter_failure(
            adapter_id="gws-evidence",
            scope=scope,
            capability="mail",
            reason_code="made_up",
        )
    with pytest.raises(ValueError, match="warning code"):
        conversation_evidence_adapters.adapter_warning("made_up")


@pytest.mark.parametrize(
    ("source_event_at", "retrieved_at", "expected"),
    (
        ("", "2026-07-29T12:00:00Z", "later_retrieved"),
        (
            "2026-07-25T12:00:00Z",
            "2026-07-29T12:00:00Z",
            "later_retrieved",
        ),
        (
            "2026-07-27T12:00:00Z",
            "2026-07-29T12:00:00Z",
            "hindsight",
        ),
        (
            "2026-07-25T12:00:00Z",
            "2026-07-26T13:00:00Z",
            "contemporaneous",
        ),
    ),
)
def test_normalize_assigns_temporal_class_from_event_and_retrieval_times(
    source_event_at: str,
    retrieved_at: str,
    expected: str,
) -> None:
    snapshot = _normalizer().normalize(
        conversation_evidence_adapters.BoundedProviderRecord(
            provider_record_id="messages/m123",
            source_type="gmail_message",
            capability="mail",
            snippet="Bounded message clue",
            source_event_at=source_event_at,
        ),
        as_of="2026-07-26T14:00:00Z",
        retrieved_at=retrieved_at,
    )

    assert snapshot.temporal_class == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("as_of", "not-a-timestamp"),
        ("as_of", "2026-07-26T14:00:00"),
        ("retrieved_at", "not-a-timestamp"),
        ("source_event_at", "not-a-timestamp"),
    ),
)
def test_normalize_rejects_invalid_or_timezone_naive_timestamps(
    field: str,
    value: str,
) -> None:
    kwargs = {
        "as_of": "2026-07-26T14:00:00Z",
        "retrieved_at": "2026-07-29T12:00:00Z",
    }
    record_kwargs = {
        "provider_record_id": "people/c123",
        "source_type": "gws_contact",
        "capability": "contacts",
        "snippet": "Ada Example",
        "source_event_at": "",
    }
    if field == "source_event_at":
        record_kwargs[field] = value
    else:
        kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        _normalizer().normalize(
            conversation_evidence_adapters.BoundedProviderRecord(**record_kwargs),
            **kwargs,
        )


@pytest.mark.parametrize(
    "record",
    (
        conversation_evidence_adapters.BoundedProviderRecord(
            provider_record_id="people/c123",
            source_type="gws_contact",
            capability="contacts",
            snippet="x" * (MAX_EVIDENCE_SNIPPET_CHARS + 1),
        ),
        conversation_evidence_adapters.BoundedProviderRecord(
            provider_record_id="people/c123",
            source_type="gws_contact",
            capability="contacts",
            snippet="Ada Example",
            structured_metadata={
                "display_name": "x" * (MAX_EVIDENCE_METADATA_CHARS + 1)
            },
        ),
    ),
)
def test_normalize_rejects_unbounded_provider_content(
    record: conversation_evidence_adapters.BoundedProviderRecord,
) -> None:
    with pytest.raises(ValueError, match="character cap"):
        _normalizer().normalize(
            record,
            as_of="2026-07-26T14:00:00Z",
            retrieved_at="2026-07-29T12:00:00Z",
        )

    with pytest.raises(ValueError, match="provider_kind"):
        conversation_evidence_adapters.EvidenceSnapshotNormalizer(
            scope=conversation_evidence_adapters.AdapterSourceScope(
                source_profile_id="gws-default",
                provider_kind="",
                account_id="",
                tenant_id="",
                capabilities=("contacts",),
            ),
            allowed_source_types=("gws_contact",),
            allowed_metadata_fields=("email",),
        )
