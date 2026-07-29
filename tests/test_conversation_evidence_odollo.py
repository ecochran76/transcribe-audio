from __future__ import annotations

import json
import subprocess
from pathlib import Path

from conversation_evidence_adapters import AdapterSourceScope
from conversation_identity_retrieval import ProviderRetrievalRequest
from conversation_knowledge_evidence import EvidenceScope
from conversation_evidence_odollo import (
    OdolloAdapterConfig,
    OdolloEvidenceAdapter,
)


def _request(
    *,
    capabilities: tuple[str, ...] = ("contacts", "leads", "log_notes"),
    max_records: int = 10,
    max_characters: int = 2_000,
    tenant_id: str = "soylei-prod",
    query_terms: tuple[str, ...] = ("person@example.com", "Project Juniper"),
) -> ProviderRetrievalRequest:
    return ProviderRetrievalRequest(
        conversation_id="conversation-1",
        query_terms=query_terms,
        scopes=(
            EvidenceScope(
                source_profile_id="odollo-soylei",
                account_id="",
                tenant_id=tenant_id,
            ),
        ),
        capabilities=capabilities,
        as_of="2026-07-01T12:00:00Z",
        max_records=max_records,
        max_characters=max_characters,
    )


def _config() -> OdolloAdapterConfig:
    return OdolloAdapterConfig(
        scope=AdapterSourceScope(
            source_profile_id="odollo-soylei",
            provider_kind="odollo",
            account_id="",
            tenant_id="soylei-prod",
            capabilities=("contacts", "leads", "log_notes"),
        ),
        command=("/opt/odollo",),
        repo_root=Path("/srv/odollo"),
        config_path=Path("/run/private/odollo.yml"),
        timeout=7.0,
    )


def test_retrieve_normalizes_all_odollo_capabilities_with_explicit_tenant_scope() -> None:
    rows_by_model = {
        "res.partner": [
            {
                "id": 11,
                "name": "Alex Person",
                "email": "person@example.com",
                "parent_id": [7, "Juniper LLC"],
            }
        ],
        "crm.lead": [
            {
                "id": 22,
                "name": "Project Juniper",
                "contact_name": "Alex Person",
                "email_from": "person@example.com",
                "partner_id": [11, "Alex Person"],
                "partner_name": "Juniper LLC",
                "create_date": "2026-06-10T08:30:00Z",
            }
        ],
        "mail.message": [
            {
                "id": 33,
                "subject": "<p>Juniper follow-up</p>",
                "model": "crm.lead",
                "res_id": 22,
                "date": "2026-06-20T09:00:00Z",
                "author_id": [11, "Alex Person"],
            }
        ],
    }
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        model = command[command.index("--model") + 1]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(rows_by_model[model]),
            stderr="",
        )

    adapter = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    )

    result = adapter.retrieve(_request())

    assert not result.failures
    assert not result.warnings
    assert [item.capability for item in result.snapshots] == [
        "contacts",
        "leads",
        "log_notes",
    ]
    assert all(item.source_profile_id == "odollo-soylei" for item in result.snapshots)
    assert all(item.account_id == "" for item in result.snapshots)
    assert all(item.tenant_id == "soylei-prod" for item in result.snapshots)
    assert result.snapshots[0].temporal_class == "later_retrieved"
    assert result.snapshots[1].source_event_at == "2026-06-10T08:30:00Z"
    assert result.snapshots[1].temporal_class == "later_retrieved"
    assert result.snapshots[2].source_event_at == "2026-06-20T09:00:00Z"
    assert result.snapshots[2].snippet == "Juniper follow-up; Alex Person"
    assert all(item.structured_metadata.get("provider_record_id") for item in result.snapshots)
    assert [command[command.index("--model") + 1] for command in commands] == [
        "res.partner",
        "crm.lead",
        "mail.message",
    ]
    assert all("--config" in command and "--profile" in command for command in commands)
    assert "body" not in commands[2][commands[2].index("--fields") + 1].split(",")


def test_retrieve_preserves_success_when_one_capability_fails() -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        model = command[command.index("--model") + 1]
        if model == "crm.lead":
            return subprocess.CompletedProcess(
                command,
                3,
                stdout="",
                stderr="authentication failed: secret material omitted",
            )
        rows = (
            [{"id": 11, "name": "Alex Person", "email": "person@example.com"}]
            if model == "res.partner"
            else []
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(rows),
            stderr="",
        )

    result = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    ).retrieve(_request())

    assert [item.capability for item in result.snapshots] == ["contacts"]
    assert result.failures == (
        {
            "adapter_id": "odollo-evidence.v1",
            "source_profile_id": "odollo-soylei",
            "provider_kind": "odollo",
            "account_id": "",
            "tenant_id": "soylei-prod",
            "capability": "leads",
            "reason_code": "provider_auth_failed",
            "detail": "query exited with status 3",
        },
    )
    assert result.warnings == ("provider_partial_result",)


def test_retrieve_filters_exact_tenant_scope_and_requested_capability() -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    adapter = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    )

    skipped = adapter.retrieve(_request(tenant_id="saber-prod"))
    contacts_only = adapter.retrieve(_request(capabilities=("contacts",)))

    assert skipped.snapshots == ()
    assert skipped.failures == ()
    assert skipped.warnings == ("provider_scope_skipped",)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model") + 1] == "res.partner"
    assert commands[0][commands[0].index("--profile") + 1] == "soylei-prod"
    assert contacts_only == contacts_only.__class__()


def test_retrieve_enforces_shared_record_and_character_budgets_with_stable_ids() -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {
                        "id": 11,
                        "name": "Alex Person",
                        "email": "person@example.com",
                    },
                    {
                        "id": 12,
                        "name": "Second Person",
                        "email": "second@example.com",
                    },
                ]
            ),
            stderr="",
        )

    adapter = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    )
    request = _request(
        capabilities=("contacts", "leads"),
        max_records=1,
        max_characters=12,
    )

    first = adapter.retrieve(request)
    second = adapter.retrieve(request)

    assert len(first.snapshots) == 1
    assert first.snapshots[0].snippet == "Alex Person;"
    assert first.snapshots[0].truncation == {
        "snippet_original_characters": 31,
        "snippet_retained_characters": 12,
    }
    assert first.snapshots[0].evidence_id == second.snapshots[0].evidence_id
    assert first.snapshots[0].content_hash == second.snapshots[0].content_hash
    assert first.warnings == (
        "provider_records_truncated",
        "provider_characters_truncated",
    )
    assert all(command[command.index("--limit") + 1] == "1" for command in commands)
    assert all(command[command.index("--model") + 1] == "res.partner" for command in commands)


def test_retrieve_rejects_unexpected_raw_provider_body_without_persisting_it() -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {
                        "id": 33,
                        "subject": "Follow-up",
                        "model": "crm.lead",
                        "res_id": 22,
                        "date": "2026-06-20T09:00:00Z",
                        "body": "<p>raw private note</p>",
                    }
                ]
            ),
            stderr="",
        )

    result = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    ).retrieve(_request(capabilities=("log_notes",)))

    assert result.snapshots == ()
    assert result.warnings == ()
    assert result.failures[0]["reason_code"] == "provider_response_invalid"
    assert result.failures[0]["detail"] == (
        "query response contained a prohibited raw body"
    )
    assert "raw private note" not in json.dumps(result.failures)


def test_retrieve_labels_malformed_provider_timestamp_as_bounded_failure() -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {
                        "id": 22,
                        "name": "Project Juniper",
                        "create_date": "not-a-timestamp",
                    }
                ]
            ),
            stderr="",
        )

    result = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    ).retrieve(_request(capabilities=("leads",)))

    assert result.snapshots == ()
    assert result.failures[0]["reason_code"] == "provider_response_invalid"
    assert result.failures[0]["detail"] == (
        "source_event_at must be a valid ISO 8601 timestamp."
    )


def test_retrieve_does_not_enumerate_tenant_when_query_plan_has_no_terms() -> None:
    called = False

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    result = OdolloEvidenceAdapter(
        _config(),
        run_command=run,
        retrieved_at=lambda: "2026-07-29T14:00:00Z",
    ).retrieve(_request(capabilities=("contacts",), query_terms=()))

    assert called is False
    assert result.snapshots == ()
    assert result.failures[0]["reason_code"] == "provider_query_failed"
    assert result.failures[0]["detail"] == "query terms are required"
