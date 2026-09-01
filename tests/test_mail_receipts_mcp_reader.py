from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from conversation_evidence_mail_receipts import MailReceiptsReadError
from mail_receipts_mcp_reader import MailReceiptsMcpReader


@dataclass
class FakeMcpClient:
    calls: list[tuple[str, dict[str, Any], int]] = field(default_factory=list)

    def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_ms: int
    ) -> Mapping[str, Any]:
        self.calls.append((name, dict(arguments), timeout_ms))
        if name == "search_mail":
            return {
                "corpus_id": "owned-corpus",
                "namespace": "default",
                "page": {"has_more": False},
                "hits": [
                    {
                        "id": "chunk-1",
                        "kind": "chunk",
                        "record_ref": "provider-message-1",
                        "follow_up": {
                            "record_ref": "provider-message-1",
                            "thread_id": "provider-thread-1",
                            "corpus_id": "owned-corpus",
                            "namespace": "default",
                        },
                    }
                ],
            }
        assert name == "selected_result_context_pack"
        return {
            "corpus_id": "owned-corpus",
            "namespace": "default",
            "items": [
                {
                    "resolved": True,
                    "context": [
                        {
                            "corpus_id": "owned-corpus",
                            "namespace": "default",
                            "message_id": "provider-message-1",
                            "thread_id": "provider-thread-1",
                            "sender": "Alex Example <alex@example.test>",
                            "to": ["Operator <operator@example.test>"],
                            "cc": [],
                            "sent_at": "2026-01-06T12:00:00Z",
                            "body_text": None,
                            "source_refs": {"provider": "fixture"},
                        }
                    ],
                }
            ],
        }


def reader(client: FakeMcpClient) -> MailReceiptsMcpReader:
    return MailReceiptsMcpReader(
        client,
        source_profile_id="source-profile",
        account_id="operator@example.test",
        tenant_id="tenant-local",
        namespace="default",
        corpus_id="owned-corpus",
    )


def test_reader_uses_one_bounded_exact_actor_search_and_one_body_free_context_pack() -> None:
    client = FakeMcpClient()

    page = reader(client).search_exact_email(
        namespace="default",
        corpus_id="owned-corpus",
        address="alex@example.test",
        as_of="2026-01-07T16:00:00Z",
        cursor="",
        page_size=25,
        include_body=False,
        timeout_ms=30_000,
    )

    assert [call[0] for call in client.calls] == [
        "search_mail",
        "selected_result_context_pack",
    ]
    assert all(
        'before:2026-01-07T16:00:00Z' in call[1]["intent"]
        for call in client.calls[:1]
    )
    assert client.calls[-1][1]["include_body"] is False
    assert client.calls[-1][1]["before"] == 0
    assert client.calls[-1][1]["after"] == 0
    assert page.as_of == "2026-01-07T16:00:00Z"
    assert page.source_scope["corpus_id"] == "owned-corpus"
    assert len(page.records) == 1
    record = page.records[0]
    assert set(record) == {
        "evidence_id",
        "record_ref",
        "logical_message_ref",
        "thread_ref",
        "source_key",
        "sent_at",
        "from",
        "to",
        "cc",
        "contact_ids_by_address",
        "signature",
    }
    assert record["from"] == ["alex@example.test"]
    assert record["to"] == ["operator@example.test"]
    assert record["signature"] is None
    assert "provider-message-1" not in str(record)


def test_reader_rejects_scope_or_body_widening_before_mcp_calls() -> None:
    client = FakeMcpClient()

    with pytest.raises(MailReceiptsReadError, match="scope or bounds"):
        reader(client).search_exact_email(
            namespace="other",
            corpus_id="owned-corpus",
            address="alex@example.test",
            as_of="2026-01-07T16:00:00Z",
            cursor="",
            page_size=25,
            include_body=False,
            timeout_ms=30_000,
        )
    with pytest.raises(MailReceiptsReadError, match="scope or bounds"):
        reader(client).search_exact_email(
            namespace="default",
            corpus_id="owned-corpus",
            address="alex@example.test",
            as_of="2026-01-07T16:00:00Z",
            cursor="",
            page_size=25,
            include_body=True,
            timeout_ms=30_000,
        )

    assert client.calls == []


def test_reader_rejects_single_corpus_fast_path_for_archive_plus_live_scope() -> None:
    class IncompleteMergeClient(FakeMcpClient):
        def call_tool(
            self, name: str, arguments: Mapping[str, Any], *, timeout_ms: int
        ) -> Mapping[str, Any]:
            response = dict(super().call_tool(name, arguments, timeout_ms=timeout_ms))
            if name == "search_mail":
                response["merge_target"] = {
                    "merge_kind": "archive_plus_live",
                    "target_corpus_ids": ["archive-corpus", "owned-corpus"],
                }
                response["retrieval_index_validation"] = {
                    "workflow_action_effect": "duckdb-message-search-direct-participant-address"
                }
            return response

    client = IncompleteMergeClient()

    with pytest.raises(MailReceiptsReadError, match="omitted part of archive-plus-live"):
        reader(client).search_exact_email(
            namespace="default",
            corpus_id="owned-corpus",
            address="alex@example.test",
            as_of="2026-01-07T16:00:00Z",
            cursor="",
            page_size=25,
            include_body=False,
            timeout_ms=30_000,
        )

    assert [call[0] for call in client.calls] == ["search_mail"]


def test_reader_accepts_archive_context_from_requested_live_scope() -> None:
    class ArchiveMergeClient(FakeMcpClient):
        def call_tool(
            self, name: str, arguments: Mapping[str, Any], *, timeout_ms: int
        ) -> Mapping[str, Any]:
            response = dict(super().call_tool(name, arguments, timeout_ms=timeout_ms))
            if name == "search_mail":
                response["merge_target"] = {
                    "merge_kind": "archive_plus_live",
                    "target_corpus_ids": ["archive-corpus", "owned-corpus"],
                }
                response["retrieval_index_validation"] = {
                    "workflow_action_effect": "archive-plus-live-duckdb-message-search"
                }
                response["hits"] = [
                    {
                        **response["hits"][0],
                        "follow_up": {
                            **response["hits"][0]["follow_up"],
                            "corpus_id": "archive-corpus",
                        },
                    }
                ]
            else:
                response["items"] = [
                    {
                        "resolved": True,
                        "context": [
                            {
                                **response["items"][0]["context"][0],
                                "corpus_id": "archive-corpus",
                            }
                        ],
                    }
                ]
            return response

    client = ArchiveMergeClient()
    page = reader(client).search_exact_email(
        namespace="default",
        corpus_id="owned-corpus",
        address="alex@example.test",
        as_of="2026-01-07T16:00:00Z",
        cursor="",
        page_size=25,
        include_body=False,
        timeout_ms=30_000,
    )

    assert len(page.records) == 1
    assert client.calls[-1][1]["targets"] == [
        {
            "target_kind": "chunk",
            "hit_kind": "chunk",
            "namespace": "default",
            "corpus_id": "archive-corpus",
            "native_ids": {},
            "message_id": "provider-message-1",
            "thread_id": "provider-thread-1",
            "chunk_id": "chunk-1",
        }
    ]
