from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from mail_relationship_contracts import ZERO_EFFECTS
from plan0073_private_pilot import (
    Plan0073PrivatePilotError,
    build_private_pilot_cohort,
    build_private_pilot_preview,
    validate_private_pilot_approval,
)


def private_pilot_request() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/dev/fixtures/plan-0073-p5/preview-request.redacted.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def approvable_private_pilot_request() -> dict:
    request = private_pilot_request()
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
    return request


def test_preview_is_deterministic_zero_read_authority() -> None:
    request = approvable_private_pilot_request()

    preview = build_private_pilot_preview(request)
    replay = build_private_pilot_preview(deepcopy(request))

    assert preview == replay
    assert preview["status"] == "awaiting_explicit_approval"
    assert preview["counts"] == {
        "conversations": 1,
        "exact_address_queries": 1,
        "maximum_provider_calls": 4,
        "maximum_records": 25,
    }
    assert preview["effects"] == ZERO_EFFECTS
    assert not any(preview["action_vector"].values())
    assert preview["approval"]["required"] is True
    assert preview["approval"]["exact_phrase"].endswith(
        preview["preview_id"]
    )


def test_approval_binds_the_exact_immutable_preview() -> None:
    preview = build_private_pilot_preview(approvable_private_pilot_request())
    approval = {
        "schema_version": "transcribe-audio.plan0073-p5-approval.v1",
        "preview_content_sha256": preview["content_sha256"],
        "exact_phrase": preview["approval"]["exact_phrase"],
        "decision": "approve",
        "approved_at": "2026-08-30T16:00:00Z",
    }

    assert validate_private_pilot_approval(preview, approval) == approval

    drifted = deepcopy(preview)
    drifted["counts"]["maximum_records"] += 1
    with pytest.raises(Plan0073PrivatePilotError, match="preview drifted"):
        validate_private_pilot_approval(drifted, approval)


def test_redacted_fixture_is_shape_only_and_cannot_be_approved() -> None:
    preview = build_private_pilot_preview(private_pilot_request())

    assert preview["status"] == "redacted_fixture_only"
    assert preview["approval"] == {
        "required": False,
        "reason": "redacted_fixture_cannot_authorize_private_read",
    }
    with pytest.raises(Plan0073PrivatePilotError, match="not awaiting approval"):
        validate_private_pilot_approval(preview, {})


def test_preview_rejects_non_exact_or_unsafe_mail_scope() -> None:
    invalid = private_pilot_request()
    invalid["cohort"][0]["exact_addresses"] = ["not-an-email"]

    with pytest.raises(Plan0073PrivatePilotError, match="email"):
        build_private_pilot_preview(invalid)

    body_request = private_pilot_request()
    body_request["retrieval"]["include_body"] = True
    with pytest.raises(Plan0073PrivatePilotError, match="metadata-only"):
        build_private_pilot_preview(body_request)

    mailbox_request = private_pilot_request()
    mailbox_request["source_scope"]["service_profile"] = "mailbox-operator"
    with pytest.raises(Plan0073PrivatePilotError, match="operator-lite"):
        build_private_pilot_preview(mailbox_request)

    widened = private_pilot_request()
    widened["budgets"]["max_calls_per_query"] = 5
    with pytest.raises(Plan0073PrivatePilotError, match="frozen pilot bounds"):
        build_private_pilot_preview(widened)


def test_preview_caps_conversations_and_exact_queries() -> None:
    oversized = private_pilot_request()
    template = oversized["cohort"][0]
    oversized["cohort"] = [
        {
            **template,
            "queue_item_id": f"queue-redacted-{index:02d}",
            "conversation_id": f"conversation-redacted-{index:02d}",
        }
        for index in range(26)
    ]
    with pytest.raises(Plan0073PrivatePilotError, match="exceeds 25"):
        build_private_pilot_preview(oversized)

    too_many_queries = private_pilot_request()
    too_many_queries["cohort"] = [
        {
            **template,
            "queue_item_id": f"queue-redacted-{index:02d}",
            "conversation_id": f"conversation-redacted-{index:02d}",
            "exact_addresses": [
                f"person{index:02d}-{address_index}@example.test"
                for address_index in range(8)
            ],
        }
        for index in range(9)
    ]
    with pytest.raises(Plan0073PrivatePilotError, match="exceeds 64"):
        build_private_pilot_preview(too_many_queries)


def test_preview_replay_is_independent_of_cohort_input_order() -> None:
    request = private_pilot_request()
    second = {
        **request["cohort"][0],
        "queue_item_id": "queue-redacted-2",
        "conversation_id": "conversation-redacted-2",
        "as_of": "2026-01-08T16:00:00Z",
        "exact_addresses": ["sam@example.test"],
    }
    request["cohort"].append(second)
    reversed_request = deepcopy(request)
    reversed_request["cohort"].reverse()

    assert build_private_pilot_preview(request) == build_private_pilot_preview(
        reversed_request
    )


def test_queue_cohort_uses_exact_contact_joins_and_excludes_shared_addresses() -> None:
    queue = {
        "items": [
            {
                "queue_item_id": "queue-redacted-2",
                "conversation_id": "conversation-redacted-2",
                "review_state": "unreviewed",
                "display": {
                    "source_document_id": "document-redacted-2",
                    "recorded_at": "2026-01-08T16:00:00Z",
                },
            },
            {
                "queue_item_id": "queue-redacted-1",
                "conversation_id": "conversation-redacted-1",
                "review_state": "unreviewed",
                "display": {
                    "source_document_id": "document-redacted-1",
                    "recorded_at": "2026-01-07T16:00:00Z",
                },
            },
        ]
    }
    people = {
        "items": [
            {
                "identity_kind": "local_contact",
                "contact_class": "person_candidate",
                "contact_methods": [{"kind": "email", "value": "alex@example.test"}],
                "calendar_occurrences": [{"document_id": "document-redacted-1"}],
            },
            {
                "identity_kind": "local_contact",
                "contact_class": "shared_or_role_address",
                "contact_methods": [{"kind": "email", "value": "team@example.test"}],
                "calendar_occurrences": [{"document_id": "document-redacted-1"}],
            },
            {
                "identity_kind": "local_contact",
                "contact_class": "person_candidate",
                "contact_methods": [{"kind": "email", "value": "sam@example.test"}],
                "calendar_occurrences": [{"document_id": "document-redacted-2"}],
            },
        ]
    }

    assert build_private_pilot_cohort(queue, people) == [
        {
            "queue_item_id": "queue-redacted-1",
            "conversation_id": "conversation-redacted-1",
            "as_of": "2026-01-07T16:00:00Z",
            "exact_addresses": ["alex@example.test"],
        },
        {
            "queue_item_id": "queue-redacted-2",
            "conversation_id": "conversation-redacted-2",
            "as_of": "2026-01-08T16:00:00Z",
            "exact_addresses": ["sam@example.test"],
        },
    ]
