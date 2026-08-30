from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from conversation_evidence_adapters import AdapterSourceScope
from conversation_evidence_mail_receipts import (
    MailReceiptsAdapterConfig,
    MailReceiptsEvidenceAdapter,
    MailReceiptsPage,
    MailReceiptsReadError,
)
from conversation_identity_retrieval import ProviderRetrievalRequest
from conversation_knowledge_evidence import EvidenceScope
from mail_relationship_contracts import validate_mail_artifact


SCOPE = AdapterSourceScope(
    source_profile_id="mail-receipts-default",
    provider_kind="mail_receipts",
    account_id="account-redacted",
    tenant_id="tenant-redacted",
    capabilities=("mail_metadata_read",),
)
REQUEST_SCOPE = EvidenceScope(
    source_profile_id="mail-receipts-default",
    account_id="account-redacted",
    tenant_id="tenant-redacted",
)
PAGE_SCOPE = {
    "source_profile_id": "mail-receipts-default",
    "account_id": "account-redacted",
    "tenant_id": "tenant-redacted",
    "namespace": "namespace-redacted",
    "corpus_id": "corpus-redacted",
}


@dataclass
class FakeMailReceiptsReader:
    profile: Mapping[str, Any]
    pages: dict[tuple[str, str], MailReceiptsPage | Exception] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def service_profile(self) -> Mapping[str, Any]:
        return self.profile

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        self.calls.append(dict(kwargs))
        result = self.pages[(str(kwargs["address"]), str(kwargs["cursor"]))]
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class SequencedMailReceiptsReader:
    profile: Mapping[str, Any]
    results: list[MailReceiptsPage | Exception | object]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def service_profile(self) -> Mapping[str, Any]:
        return self.profile

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        self.calls.append(dict(kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]


def operator_lite_profile() -> dict[str, Any]:
    return {
        "profile": "operator-lite",
        "capabilities": [
            "search_mail",
            "selected_result_context_pack",
            "resolve_person",
            "get_person_neighborhood",
            "get_person_relationship_path",
        ],
        "mailbox_mutation": False,
        "corpus_operation_execution": False,
    }


def mail_page(**kwargs: Any) -> MailReceiptsPage:
    return MailReceiptsPage(source_scope=PAGE_SCOPE, **kwargs)


def request(
    *,
    query_terms: tuple[str, ...] = ("alex@example.test",),
    max_records: int = 25,
    max_characters: int = 1,
) -> ProviderRetrievalRequest:
    return ProviderRetrievalRequest(
        conversation_id="conversation-redacted-1",
        query_terms=query_terms,
        scopes=(REQUEST_SCOPE,),
        capabilities=("mail_metadata_read",),
        as_of="2026-01-07T16:00:00Z",
        max_records=max_records,
        max_characters=max_characters,
    )


def test_adapter_requires_the_operator_lite_read_only_profile() -> None:
    reader = FakeMailReceiptsReader(
        {
            "profile": "mailbox-operator",
            "capabilities": ["search_mail"],
            "mailbox_mutation": True,
            "corpus_operation_execution": True,
        }
    )

    with pytest.raises(ValueError, match="operator-lite read-only profile"):
        MailReceiptsEvidenceAdapter(
            config=MailReceiptsAdapterConfig(
                scope=SCOPE,
                namespace="namespace-redacted",
                corpus_id="corpus-redacted",
                account_address="account@example.test",
            ),
            reader=reader,
            retrieved_at="2026-01-07T16:01:00Z",
        )

    assert reader.calls == []


def test_adapter_normalizes_one_exact_mail_metadata_page_and_emits_a_receipt() -> None:
    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(
                    {
                        "evidence_id": "evidence-redacted-1",
                        "record_ref": "record-redacted-1",
                        "logical_message_ref": "logical-message-redacted-1",
                        "thread_ref": "thread-redacted-1",
                        "source_key": "source-redacted-1",
                        "sent_at": "2026-01-06T12:00:00Z",
                        "from": ["alex@example.test"],
                        "to": ["account@example.test"],
                        "cc": [],
                        "contact_ids_by_address": {
                            "alex@example.test": "contact-redacted-alex"
                        },
                        "signature": {
                            "address": "alex@example.test",
                            "title": "Program Director",
                            "organization": "Example Organization",
                            "department": "Programs",
                        },
                    },
                ),
                as_of="2026-01-07T16:00:00Z",
            )
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert result.failures == ()
    assert result.warnings == ()
    assert len(result.snapshots) == 1
    snapshot = result.snapshots[0]
    assert snapshot.provider_kind == "mail_receipts"
    assert snapshot.capability == "mail_metadata_read"
    assert snapshot.snippet == ""
    assert snapshot.source_event_at == "2026-01-06T12:00:00Z"
    assert snapshot.structured_metadata["message_ref_hash"]
    assert snapshot.structured_metadata["thread_ref_hash"]
    assert "logical_message_ref" not in snapshot.structured_metadata
    assert "thread_ref" not in snapshot.structured_metadata
    assert validate_mail_artifact(
        "mail_query_receipt", result.query_receipt
    ) == result.query_receipt
    assert reader.calls == [
        {
            "namespace": "namespace-redacted",
            "corpus_id": "corpus-redacted",
            "address": "alex@example.test",
            "as_of": "2026-01-07T16:00:00Z",
            "cursor": "",
            "page_size": 25,
            "include_body": False,
            "timeout_ms": 30_000,
        }
    ]


def test_adapter_excludes_records_older_than_the_frozen_365_day_lookback() -> None:
    def record(evidence_id: str, sent_at: str) -> dict[str, Any]:
        return {
            "evidence_id": evidence_id,
            "record_ref": f"record-{evidence_id}",
            "logical_message_ref": f"logical-{evidence_id}",
            "thread_ref": f"thread-{evidence_id}",
            "source_key": f"source-{evidence_id}",
            "sent_at": sent_at,
            "from": ["alex@example.test"],
            "to": ["account@example.test"],
            "cc": [],
            "contact_ids_by_address": {
                "alex@example.test": "contact-redacted-alex"
            },
            "signature": None,
        }

    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(
                    record("within-lookback", "2025-01-07T16:00:00Z"),
                    record("before-lookback", "2025-01-07T15:59:59Z"),
                ),
                as_of="2026-01-07T16:00:00Z",
            )
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert len(result.snapshots) == 1
    assert result.snapshots[0].structured_metadata["evidence_id"] == "within-lookback"
    assert result.query_receipt["counts"]["selected"] == 1
    assert result.query_receipt["counts"]["excluded"] == 1


def test_adapter_preserves_successful_evidence_when_another_exact_query_fails() -> None:
    record = {
        "evidence_id": "evidence-redacted-1",
        "record_ref": "record-redacted-1",
        "logical_message_ref": "logical-message-redacted-1",
        "thread_ref": "thread-redacted-1",
        "source_key": "source-redacted-1",
        "sent_at": "2026-01-06T12:00:00Z",
        "from": ["alex@example.test"],
        "to": ["account@example.test"],
        "cc": [],
        "contact_ids_by_address": {
            "alex@example.test": "contact-redacted-alex"
        },
        "signature": None,
    }
    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(record,), as_of="2026-01-07T16:00:00Z"
            ),
            ("blair@example.test", ""): MailReceiptsReadError(
                "provider_unavailable", "operator-lite read unavailable"
            ),
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(
        request(query_terms=("alex@example.test", "blair@example.test"))
    )

    assert len(result.snapshots) == 1
    assert result.failures[0]["reason_code"] == "provider_unavailable"
    assert result.warnings == ("provider_partial_result",)
    assert result.query_receipt["status"] == "partial"
    assert result.query_receipt["failures"] == [
        {
            "reason_code": "partial_source_failure",
            "detail": "provider_unavailable",
        }
    ]


def test_adapter_preserves_opaque_cursor_and_as_of_while_paging() -> None:
    def record(index: int) -> dict[str, Any]:
        return {
            "evidence_id": f"evidence-redacted-{index}",
            "record_ref": f"record-redacted-{index}",
            "logical_message_ref": f"logical-message-redacted-{index}",
            "thread_ref": f"thread-redacted-{index}",
            "source_key": "source-redacted-1",
            "sent_at": f"2026-01-0{index}T12:00:00Z",
            "from": ["alex@example.test"],
            "to": ["account@example.test"],
            "cc": [],
            "contact_ids_by_address": {
                "alex@example.test": "contact-redacted-alex"
            },
            "signature": None,
        }

    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(record(1),),
                next_cursor="opaque-cursor-1",
                as_of="2026-01-07T16:00:00Z",
            ),
            ("alex@example.test", "opaque-cursor-1"): mail_page(
                records=(record(2),),
                as_of="2026-01-07T16:00:00Z",
            ),
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request(max_records=2))

    assert len(result.snapshots) == 2
    assert [call["cursor"] for call in reader.calls] == ["", "opaque-cursor-1"]
    assert {call["as_of"] for call in reader.calls} == {
        "2026-01-07T16:00:00Z"
    }
    assert [call["page_size"] for call in reader.calls] == [2, 1]


def test_adapter_rejects_raw_content_as_a_visible_unavailable_result() -> None:
    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(
                    {
                        "evidence_id": "evidence-redacted-1",
                        "record_ref": "record-redacted-1",
                        "logical_message_ref": "logical-message-redacted-1",
                        "thread_ref": "thread-redacted-1",
                        "source_key": "source-redacted-1",
                        "sent_at": "2026-01-06T12:00:00Z",
                        "from": ["alex@example.test"],
                        "to": ["account@example.test"],
                        "cc": [],
                        "contact_ids_by_address": {},
                        "signature": None,
                        "body": "prohibited private content",
                    },
                ),
                as_of="2026-01-07T16:00:00Z",
            )
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert result.snapshots == ()
    assert result.failures[0]["reason_code"] == "provider_response_invalid"
    assert result.query_receipt["status"] == "unavailable"
    assert "private content" not in str(result.query_receipt)


def test_adapter_retries_one_transient_read_without_widening_the_query() -> None:
    page = mail_page(records=(), as_of="2026-01-07T16:00:00Z")
    reader = SequencedMailReceiptsReader(
        operator_lite_profile(),
        [
            MailReceiptsReadError(
                "provider_unavailable",
                "temporary operator-lite outage",
                retryable=True,
            ),
            page,
        ],
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert result.failures == ()
    assert len(reader.calls) == 2
    assert reader.calls[0] == reader.calls[1]
    assert result.query_receipt["status"] == "complete"


def test_adapter_reports_a_malformed_page_without_raising() -> None:
    reader = SequencedMailReceiptsReader(
        operator_lite_profile(),
        [{"records": []}],
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert result.snapshots == ()
    assert result.failures[0]["reason_code"] == "provider_response_invalid"
    assert result.query_receipt["status"] == "unavailable"


def test_adapter_rejects_a_page_from_a_different_namespace() -> None:
    wrong_scope = dict(PAGE_SCOPE)
    wrong_scope["namespace"] = "another-namespace"
    reader = SequencedMailReceiptsReader(
        operator_lite_profile(),
        [
            MailReceiptsPage(
                records=(),
                as_of="2026-01-07T16:00:00Z",
                source_scope=wrong_scope,
            )
        ],
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request())

    assert result.snapshots == ()
    assert result.failures[0]["reason_code"] == "provider_response_invalid"
    assert result.query_receipt["status"] == "unavailable"


def test_adapter_skips_a_nonmatching_tenant_scope_without_a_provider_call() -> None:
    reader = FakeMailReceiptsReader(operator_lite_profile())
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )
    mismatched = request()
    mismatched = ProviderRetrievalRequest(
        conversation_id=mismatched.conversation_id,
        query_terms=mismatched.query_terms,
        scopes=(
            EvidenceScope(
                source_profile_id="mail-receipts-default",
                account_id="account-redacted",
                tenant_id="another-tenant",
            ),
        ),
        capabilities=mismatched.capabilities,
        as_of=mismatched.as_of,
        max_records=mismatched.max_records,
        max_characters=mismatched.max_characters,
    )

    result = adapter.retrieve(mismatched)

    assert result.snapshots == ()
    assert result.warnings == ("provider_scope_skipped",)
    assert result.query_receipt == {}
    assert reader.calls == []


def test_adapter_replay_is_deterministic_for_the_same_bounded_page() -> None:
    page = mail_page(
        records=(
            {
                "evidence_id": "evidence-redacted-1",
                "record_ref": "record-redacted-1",
                "logical_message_ref": "logical-message-redacted-1",
                "thread_ref": "thread-redacted-1",
                "source_key": "source-redacted-1",
                "sent_at": "2026-01-06T12:00:00Z",
                "from": ["alex@example.test"],
                "to": ["account@example.test"],
                "cc": [],
                "contact_ids_by_address": {
                    "alex@example.test": "contact-redacted-alex"
                },
                "signature": None,
            },
        ),
        as_of="2026-01-07T16:00:00Z",
    )

    def replay() -> object:
        adapter = MailReceiptsEvidenceAdapter(
            config=MailReceiptsAdapterConfig(
                scope=SCOPE,
                namespace="namespace-redacted",
                corpus_id="corpus-redacted",
                account_address="account@example.test",
            ),
            reader=FakeMailReceiptsReader(
                operator_lite_profile(),
                {("alex@example.test", ""): page},
            ),
            retrieved_at="2026-01-07T16:01:00Z",
        )
        return adapter.retrieve(request())

    first = replay()
    second = replay()

    assert first == second


def test_adapter_marks_records_hidden_by_the_frozen_record_budget() -> None:
    def record(index: int) -> dict[str, Any]:
        return {
            "evidence_id": f"evidence-redacted-{index}",
            "record_ref": f"record-redacted-{index}",
            "logical_message_ref": f"logical-message-redacted-{index}",
            "thread_ref": f"thread-redacted-{index}",
            "source_key": "source-redacted-1",
            "sent_at": f"2026-01-0{index}T12:00:00Z",
            "from": ["alex@example.test"],
            "to": ["account@example.test"],
            "cc": [],
            "contact_ids_by_address": {},
            "signature": None,
        }

    reader = FakeMailReceiptsReader(
        operator_lite_profile(),
        {
            ("alex@example.test", ""): mail_page(
                records=(record(1), record(2)),
                as_of="2026-01-07T16:00:00Z",
            )
        },
    )
    adapter = MailReceiptsEvidenceAdapter(
        config=MailReceiptsAdapterConfig(
            scope=SCOPE,
            namespace="namespace-redacted",
            corpus_id="corpus-redacted",
            account_address="account@example.test",
        ),
        reader=reader,
        retrieved_at="2026-01-07T16:01:00Z",
    )

    result = adapter.retrieve(request(max_records=1))

    assert len(result.snapshots) == 1
    assert result.warnings == ("provider_records_truncated",)
    assert result.query_receipt["counts"]["truncated"] == 1
