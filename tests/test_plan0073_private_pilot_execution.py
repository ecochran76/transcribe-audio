from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from acoustic_audio_derivatives import ensure_private_tree, write_immutable_private_json
from conversation_evidence_mail_receipts import MailReceiptsPage, MailReceiptsReadError
from mail_relationship_contracts import ZERO_EFFECTS
from plan0073_private_pilot import build_private_pilot_preview
from plan0073_private_pilot_execution import (
    Plan0073PrivatePilotExecutionError,
    build_private_pilot_contacts,
    execute_private_pilot,
    replay_private_pilot,
)


def approved_preview() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "docs/dev/fixtures/plan-0073-p5/preview-request.redacted.json"
    )
    request = json.loads(fixture.read_text(encoding="utf-8"))
    request["source_scope"].update(
        {
            "source_profile_id": "mail-receipts-source-profile-1",
            "account_id": "mail-account-1",
            "tenant_id": "mail-tenant-1",
            "account_address": "operator@private-mail.invalid",
            "namespace": "default",
            "corpus_id": "owned-mail-corpus-1",
        }
    )
    request["cohort"][0].update(
        {
            "queue_item_id": "queue-item-1",
            "conversation_id": "conversation-1",
            "exact_addresses": ["alex@private-mail.invalid"],
        }
    )
    preview = build_private_pilot_preview(request)
    approval = {
        "schema_version": "transcribe-audio.plan0073-p5-approval.v1",
        "preview_content_sha256": preview["content_sha256"],
        "exact_phrase": preview["approval"]["exact_phrase"],
        "decision": "approve",
        "approved_at": "2026-08-30T16:00:00Z",
    }
    return preview, approval


def approval_for(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "transcribe-audio.plan0073-p5-approval.v1",
        "preview_content_sha256": preview["content_sha256"],
        "exact_phrase": preview["approval"]["exact_phrase"],
        "decision": "approve",
        "approved_at": "2026-08-30T16:00:00Z",
    }


@dataclass
class TrackingReader:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def service_profile(self) -> Mapping[str, Any]:
        self.calls.append({"operation": "service_profile"})
        return {}

    def search_exact_email(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        raise AssertionError("An unapproved pilot must not read evidence.")


@dataclass
class OnePageReader:
    page: MailReceiptsPage
    calls: list[dict[str, Any]] = field(default_factory=list)

    def service_profile(self) -> Mapping[str, Any]:
        self.calls.append({"operation": "service_profile"})
        return {
            "profile": "operator-lite",
            "capabilities": ["search_mail"],
            "mailbox_mutation": False,
            "corpus_operation_execution": False,
        }

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        self.calls.append(dict(kwargs))
        return self.page


@dataclass
class RetryEveryAddressReader:
    page_scope: Mapping[str, str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)

    def service_profile(self) -> Mapping[str, Any]:
        return {
            "profile": "operator-lite",
            "capabilities": ["search_mail"],
            "mailbox_mutation": False,
            "corpus_operation_execution": False,
        }

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        self.calls.append(dict(kwargs))
        address = str(kwargs["address"])
        self.attempts[address] = self.attempts.get(address, 0) + 1
        if self.attempts[address] == 1:
            raise MailReceiptsReadError(
                "provider_unavailable",
                "synthetic transient timeout",
                retryable=True,
            )
        return MailReceiptsPage(
            records=(),
            as_of=str(kwargs["as_of"]),
            source_scope=self.page_scope,
        )


def test_execute_rejects_unbound_approval_before_reader_or_runtime_effects(
    tmp_path: Path,
) -> None:
    preview, _ = approved_preview()
    reader = TrackingReader()
    runtime_root = tmp_path / "private-runtime"

    with pytest.raises(
        Plan0073PrivatePilotExecutionError,
        match="approval",
    ):
        execute_private_pilot(
            preview,
            {},
            reader=reader,
            contacts={},
            runtime_root=runtime_root,
            executed_at="2026-08-30T16:01:00Z",
        )

    assert reader.calls == []
    assert not runtime_root.exists()


def test_contact_builder_uses_exact_local_emails_and_adds_the_mail_account() -> None:
    people_payload = {
        "items": [
            {
                "identity_kind": "local_contact",
                "person_id": "contact-alex",
                "primary_name": "Alex Example",
                "contact_class": "person_candidate",
                "contact_methods": [
                    {"kind": "email", "value": "alex@private-mail.invalid"}
                ],
            },
            {
                "identity_kind": "local_contact",
                "person_id": "contact-team",
                "primary_name": "Team",
                "contact_class": "shared_or_role_address",
                "contact_methods": [
                    {"kind": "email", "value": "team@private-mail.invalid"}
                ],
            },
            {
                "identity_kind": "provider_record",
                "person_id": "provider-only",
                "primary_name": "Provider only",
                "contact_class": "person_candidate",
                "contact_methods": [
                    {"kind": "email", "value": "provider@private-mail.invalid"}
                ],
            },
        ]
    }

    contacts = build_private_pilot_contacts(
        people_payload,
        account_address="operator@private-mail.invalid",
    )

    assert contacts["contact-alex"] == {
        "contact_id": "contact-alex",
        "label": "Alex Example",
        "email": "alex@private-mail.invalid",
        "contact_class": "person_candidate",
    }
    assert contacts["contact-team"]["contact_class"] == "shared_or_role_address"
    assert "provider-only" not in contacts
    account = next(
        value
        for value in contacts.values()
        if value["email"] == "operator@private-mail.invalid"
    )
    assert account["label"] == "Mail account"
    assert account["contact_class"] == "person_candidate"


def test_execute_rejects_profile_mismatch_before_runtime_or_corpus_read(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    reader = TrackingReader()
    runtime_root = tmp_path / "private-runtime"

    with pytest.raises(
        Plan0073PrivatePilotExecutionError,
        match="operator-lite",
    ):
        execute_private_pilot(
            preview,
            approval,
            reader=reader,
            contacts={},
            runtime_root=runtime_root,
            executed_at="2026-08-30T16:01:00Z",
        )

    assert reader.calls == [{"operation": "service_profile"}]
    assert not runtime_root.exists()


def test_execute_rejects_preapproval_time_before_reader_or_runtime_effects(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    reader = TrackingReader()
    runtime_root = tmp_path / "private-runtime"

    with pytest.raises(
        Plan0073PrivatePilotExecutionError,
        match="cannot predate",
    ):
        execute_private_pilot(
            preview,
            approval,
            reader=reader,
            contacts={},
            runtime_root=runtime_root,
            executed_at="2026-08-30T15:59:59Z",
        )

    assert reader.calls == []
    assert not runtime_root.exists()


def test_execute_rejects_invalid_contacts_before_reader_or_runtime_effects(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    reader = TrackingReader()
    runtime_root = tmp_path / "private-runtime"
    duplicate_contacts = {
        "contact-1": {
            "contact_id": "contact-1",
            "label": "First",
            "email": "alex@private-mail.invalid",
            "contact_class": "person_candidate",
        },
        "contact-2": {
            "contact_id": "contact-2",
            "label": "Second",
            "email": "alex@private-mail.invalid",
            "contact_class": "person_candidate",
        },
    }

    with pytest.raises(
        Plan0073PrivatePilotExecutionError,
        match="contact map",
    ):
        execute_private_pilot(
            preview,
            approval,
            reader=reader,
            contacts=duplicate_contacts,
            runtime_root=runtime_root,
            executed_at="2026-08-30T16:01:00Z",
        )

    assert reader.calls == []
    assert not runtime_root.exists()


def test_execute_one_approved_query_persists_private_zero_effect_receipts(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    page_scope = {
        "source_profile_id": "mail-receipts-source-profile-1",
        "account_id": "mail-account-1",
        "tenant_id": "mail-tenant-1",
        "namespace": "default",
        "corpus_id": "owned-mail-corpus-1",
    }
    reader = OnePageReader(
        MailReceiptsPage(
            records=(
                {
                    "evidence_id": "evidence-1",
                    "record_ref": "record-1",
                    "logical_message_ref": "logical-message-1",
                    "thread_ref": "thread-1",
                    "source_key": "source-1",
                    "sent_at": "2026-01-06T12:00:00Z",
                    "from": ["alex@private-mail.invalid"],
                    "to": ["operator@private-mail.invalid"],
                    "cc": [],
                    "contact_ids_by_address": {},
                    "signature": {
                        "address": "alex@private-mail.invalid",
                        "title": "Program Director",
                        "organization": "Example Organization",
                        "department": "Programs",
                    },
                },
            ),
            as_of="2026-01-07T16:00:00Z",
            source_scope=page_scope,
        )
    )
    contacts = {
        "contact-alex": {
            "contact_id": "contact-alex",
            "label": "Alex Example",
            "email": "alex@private-mail.invalid",
            "contact_class": "person_candidate",
        },
        "contact-account": {
            "contact_id": "contact-account",
            "label": "Mail account",
            "email": "operator@private-mail.invalid",
            "contact_class": "person_candidate",
        },
    }
    runtime_root = tmp_path / "private-runtime"

    receipt = execute_private_pilot(
        preview,
        approval,
        reader=reader,
        contacts=contacts,
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:01:00Z",
    )

    assert receipt["schema_version"] == (
        "transcribe-audio.plan0073-p5-execution-receipt.v1"
    )
    assert receipt["status"] == "complete"
    assert receipt["counts"] == {
        "planned_queries": 1,
        "accounted_queries": 1,
        "unavailable_queries": 0,
        "selected_records": 1,
        "observations": 1,
        "independence_groups": 1,
        "hypotheses": 2,
        "provider_writes": 0,
        "accepted_effects": 0,
    }
    assert receipt["effects"] == ZERO_EFFECTS
    assert receipt["action_vector"] == {
        "owned_corpus_read": True,
        "mailbox_or_provider_call": False,
        "runtime_write": True,
        "schema_migration": False,
        "deployment": False,
        "accepted_graph_write": False,
        "person_merge": False,
        "speaker_or_profile_effect": False,
        "graphiti_write": False,
    }
    assert reader.calls == [
        {"operation": "service_profile"},
        {
            "namespace": "default",
            "corpus_id": "owned-mail-corpus-1",
            "address": "alex@private-mail.invalid",
            "as_of": "2026-01-07T16:00:00Z",
            "cursor": "",
            "page_size": 25,
            "include_body": False,
            "timeout_ms": 30_000,
        },
    ]
    aggregate_path = Path(receipt["aggregate_path"])
    assert aggregate_path.is_file()
    assert aggregate_path.stat().st_mode & 0o777 == 0o600
    assert runtime_root.stat().st_mode & 0o777 == 0o700


def test_execute_accepts_the_exact_persisted_preview_only_checkpoint(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    runtime_root = tmp_path / "private-runtime"
    run = runtime_root / preview["runtime_write_surface"]["relative_root"]
    ensure_private_tree(runtime_root, run)
    write_immutable_private_json(run / "preview.json", preview)
    source = preview["request"]["source_scope"]
    reader = OnePageReader(
        MailReceiptsPage(
            records=(),
            as_of=preview["query_plan"][0]["as_of"],
            source_scope={
                "source_profile_id": source["source_profile_id"],
                "account_id": source["account_id"],
                "tenant_id": source["tenant_id"],
                "namespace": source["namespace"],
                "corpus_id": source["corpus_id"],
            },
        )
    )

    receipt = execute_private_pilot(
        preview,
        approval,
        reader=reader,
        contacts={},
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:01:00Z",
    )

    assert receipt["status"] == "complete"
    assert receipt["counts"]["accounted_queries"] == 1
    assert (run / "approval.json").is_file()
    assert replay_private_pilot(
        preview["preview_id"], runtime_root=runtime_root
    )["replay_equal"] is True


def test_execute_accounts_for_each_exact_query_without_double_counting_one_message(
    tmp_path: Path,
) -> None:
    preview, _ = approved_preview()
    request = preview["request"]
    request["cohort"][0]["exact_addresses"] = [
        "alex@private-mail.invalid",
        "sam@private-mail.invalid",
    ]
    preview = build_private_pilot_preview(request)
    page_scope = {
        "source_profile_id": "mail-receipts-source-profile-1",
        "account_id": "mail-account-1",
        "tenant_id": "mail-tenant-1",
        "namespace": "default",
        "corpus_id": "owned-mail-corpus-1",
    }
    record = {
        "evidence_id": "evidence-shared-1",
        "record_ref": "record-shared-1",
        "logical_message_ref": "logical-message-shared-1",
        "thread_ref": "thread-shared-1",
        "source_key": "source-shared-1",
        "sent_at": "2026-01-06T12:00:00Z",
        "from": ["alex@private-mail.invalid"],
        "to": [
            "operator@private-mail.invalid",
            "sam@private-mail.invalid",
        ],
        "cc": [],
        "contact_ids_by_address": {
            "alex@private-mail.invalid": "contact-alex",
            "operator@private-mail.invalid": "contact-account",
            "sam@private-mail.invalid": "contact-sam",
        },
        "signature": None,
    }
    reader = OnePageReader(
        MailReceiptsPage(
            records=(record,),
            as_of="2026-01-07T16:00:00Z",
            source_scope=page_scope,
        )
    )
    contacts = {
        "contact-alex": {
            "contact_id": "contact-alex",
            "label": "Alex Example",
            "email": "alex@private-mail.invalid",
            "contact_class": "person_candidate",
        },
        "contact-sam": {
            "contact_id": "contact-sam",
            "label": "Sam Example",
            "email": "sam@private-mail.invalid",
            "contact_class": "person_candidate",
        },
        "contact-account": {
            "contact_id": "contact-account",
            "label": "Mail account",
            "email": "operator@private-mail.invalid",
            "contact_class": "person_candidate",
        },
    }

    receipt = execute_private_pilot(
        preview,
        approval_for(preview),
        reader=reader,
        contacts=contacts,
        runtime_root=tmp_path / "private-runtime",
        executed_at="2026-08-30T16:01:00Z",
    )

    assert receipt["counts"]["planned_queries"] == 2
    assert receipt["counts"]["accounted_queries"] == 2
    assert receipt["counts"]["selected_records"] == 2
    assert receipt["counts"]["observations"] == 1
    assert receipt["counts"]["independence_groups"] == 1
    assert len(receipt["artifacts"]["query_receipts"]) == 2
    assert [call.get("address") for call in reader.calls[1:]] == [
        "alex@private-mail.invalid",
        "sam@private-mail.invalid",
    ]


def test_offline_replay_recomputes_hypotheses_without_a_reader(tmp_path: Path) -> None:
    preview, approval = approved_preview()
    page_scope = {
        "source_profile_id": "mail-receipts-source-profile-1",
        "account_id": "mail-account-1",
        "tenant_id": "mail-tenant-1",
        "namespace": "default",
        "corpus_id": "owned-mail-corpus-1",
    }
    reader = OnePageReader(
        MailReceiptsPage(
            records=(),
            as_of="2026-01-07T16:00:00Z",
            source_scope=page_scope,
        )
    )
    runtime_root = tmp_path / "private-runtime"
    executed = execute_private_pilot(
        preview,
        approval,
        reader=reader,
        contacts={
            "contact-alex": {
                "contact_id": "contact-alex",
                "label": "Alex Example",
                "email": "alex@private-mail.invalid",
                "contact_class": "person_candidate",
            }
        },
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:01:00Z",
    )

    replay = replay_private_pilot(
        preview["preview_id"],
        runtime_root=runtime_root,
    )

    assert replay["schema_version"] == (
        "transcribe-audio.plan0073-p5-offline-replay.v1"
    )
    assert replay["replay_equal"] is True
    assert replay["source_execution_sha256"] == executed["content_sha256"]
    assert replay["counts"] == executed["counts"]
    assert replay["effects"] == ZERO_EFFECTS


def test_execute_allows_only_one_transient_retry_across_the_whole_pilot(
    tmp_path: Path,
) -> None:
    preview, _ = approved_preview()
    request = preview["request"]
    request["cohort"][0]["exact_addresses"] = [
        "alex@private-mail.invalid",
        "sam@private-mail.invalid",
    ]
    preview = build_private_pilot_preview(request)
    reader = RetryEveryAddressReader(
        {
            "source_profile_id": "mail-receipts-source-profile-1",
            "account_id": "mail-account-1",
            "tenant_id": "mail-tenant-1",
            "namespace": "default",
            "corpus_id": "owned-mail-corpus-1",
        }
    )

    receipt = execute_private_pilot(
        preview,
        approval_for(preview),
        reader=reader,
        contacts={
            "contact-alex": {
                "contact_id": "contact-alex",
                "label": "Alex Example",
                "email": "alex@private-mail.invalid",
                "contact_class": "person_candidate",
            },
            "contact-sam": {
                "contact_id": "contact-sam",
                "label": "Sam Example",
                "email": "sam@private-mail.invalid",
                "contact_class": "person_candidate",
            },
        },
        runtime_root=tmp_path / "private-runtime",
        executed_at="2026-08-30T16:01:00Z",
    )

    assert len(reader.calls) == 3
    assert reader.attempts == {
        "alex@private-mail.invalid": 2,
        "sam@private-mail.invalid": 1,
    }
    assert receipt["status"] == "partial"
    assert receipt["counts"]["unavailable_queries"] == 1


def test_repeat_apply_replays_completed_receipt_without_another_read(
    tmp_path: Path,
) -> None:
    preview, approval = approved_preview()
    page_scope = {
        "source_profile_id": "mail-receipts-source-profile-1",
        "account_id": "mail-account-1",
        "tenant_id": "mail-tenant-1",
        "namespace": "default",
        "corpus_id": "owned-mail-corpus-1",
    }
    runtime_root = tmp_path / "private-runtime"
    first = execute_private_pilot(
        preview,
        approval,
        reader=OnePageReader(
            MailReceiptsPage(
                records=(),
                as_of="2026-01-07T16:00:00Z",
                source_scope=page_scope,
            )
        ),
        contacts={},
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:01:00Z",
    )
    no_read = TrackingReader()

    second = execute_private_pilot(
        preview,
        approval,
        reader=no_read,
        contacts={},
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:02:00Z",
    )

    assert no_read.calls == []
    assert second["idempotent"] is True
    assert second["content_sha256"] == first["content_sha256"]


def test_execute_accounts_for_the_current_25_conversation_57_query_shape(
    tmp_path: Path,
) -> None:
    preview, _ = approved_preview()
    request = preview["request"]
    request["cohort"] = []
    for index in range(25):
        address_count = 3 if index < 7 else 2
        request["cohort"].append(
            {
                "queue_item_id": f"queue-item-{index:02d}",
                "conversation_id": f"conversation-{index:02d}",
                "as_of": f"2026-01-{index + 1:02d}T16:00:00Z",
                "exact_addresses": [
                    f"person-{index:02d}-{address}@private-mail.invalid"
                    for address in range(address_count)
                ],
            }
        )
    preview = build_private_pilot_preview(request)
    reader = OnePageReader(
        MailReceiptsPage(
            records=(),
            source_scope={
                "source_profile_id": "mail-receipts-source-profile-1",
                "account_id": "mail-account-1",
                "tenant_id": "mail-tenant-1",
                "namespace": "default",
                "corpus_id": "owned-mail-corpus-1",
            },
        )
    )
    runtime_root = tmp_path / "private-runtime"

    receipt = execute_private_pilot(
        preview,
        approval_for(preview),
        reader=reader,
        contacts={},
        runtime_root=runtime_root,
        executed_at="2026-08-30T16:01:00Z",
    )

    assert preview["counts"]["conversations"] == 25
    assert preview["counts"]["exact_address_queries"] == 57
    assert receipt["counts"]["planned_queries"] == 57
    assert receipt["counts"]["accounted_queries"] == 57
    assert len(receipt["artifacts"]["query_receipts"]) == 57
    assert len(receipt["artifacts"]["normalized"]) == 25
    assert len(reader.calls) == 58
    assert replay_private_pilot(
        preview["preview_id"], runtime_root=runtime_root
    )["replay_equal"] is True
