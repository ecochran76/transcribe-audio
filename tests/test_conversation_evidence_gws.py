from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field

import pytest

from conversation_evidence_adapters import AdapterSourceScope
from conversation_evidence_gws import (
    GwsCliConfig,
    GwsCliReader,
    GwsEvidenceAdapter,
    GwsProviderPage,
    GwsProviderReadError,
)
from conversation_identity_retrieval import ProviderRetrievalRequest
from conversation_knowledge_evidence import EvidenceScope


SCOPE = AdapterSourceScope(
    source_profile_id="gws-default",
    provider_kind="gws",
    account_id="",
    tenant_id="",
    capabilities=("people", "gmail", "drive", "calendar"),
)
REQUEST_SCOPE = EvidenceScope(
    source_profile_id="gws-default",
    account_id="",
    tenant_id="",
)


@dataclass
class FakeGwsProvider:
    pages: dict[tuple[str, str], GwsProviderPage | Exception]
    calls: list[tuple[str, tuple[str, ...], str, int]] = field(default_factory=list)

    def fetch_page(
        self,
        *,
        capability: str,
        query_terms: tuple[str, ...],
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        self.calls.append((capability, query_terms, page_token, page_size))
        outcome = self.pages[(capability, page_token)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def request(
    *,
    scopes: tuple[EvidenceScope, ...] = (REQUEST_SCOPE,),
    capabilities: tuple[str, ...] = ("people",),
    query_terms: tuple[str, ...] = ("person@example.com", "orchard"),
    max_records: int = 10,
    max_characters: int = 1_000,
) -> ProviderRetrievalRequest:
    return ProviderRetrievalRequest(
        conversation_id="conversation-1",
        query_terms=query_terms,
        scopes=scopes,
        capabilities=capabilities,
        as_of="2024-01-01T12:00:00Z",
        max_records=max_records,
        max_characters=max_characters,
    )


def test_gws_adapter_normalizes_explicit_scope_and_temporal_evidence() -> None:
    provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "people/c123",
                        "source_type": "gws_contact",
                        "snippet": "Pat Person <person@example.com>",
                        "structured_metadata": {
                            "name": "Pat Person",
                            "email": "person@example.com",
                            "surface": "contacts",
                        },
                        "source_uri": "gws:people/c123",
                    },
                )
            ),
            ("gmail", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "message-m456",
                        "source_type": "gws_mail_message",
                        "snippet": "Orchard planning follow-up",
                        "structured_metadata": {
                            "subject": "Orchard planning",
                            "thread_id": "thread-t1",
                        },
                        "source_event_at": "2023-12-20T09:30:00-06:00",
                        "independence_group_id": "thread-t1",
                    },
                )
            ),
            ("drive", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "file-f789",
                        "source_type": "gws_docs_file",
                        "snippet": "Orchard plan",
                        "structured_metadata": {
                            "name": "Orchard plan",
                            "mime_type": "application/vnd.google-apps.document",
                        },
                        "source_event_at": "2023-12-18T10:00:00Z",
                    },
                )
            ),
            ("calendar", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "event-e123",
                        "source_type": "gws_calendar_event_detail",
                        "snippet": "Orchard review",
                        "structured_metadata": {
                            "calendar_id": "primary",
                            "attendee_emails": ["person@example.com"],
                        },
                        "source_event_at": "2024-01-01T10:00:00Z",
                    },
                )
            ),
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    retrieval_request = request(
        capabilities=("people", "gmail", "drive", "calendar")
    )
    result = adapter.retrieve(retrieval_request)
    replay = adapter.retrieve(retrieval_request)

    assert result.failures == ()
    assert result.warnings == ()
    assert len(result.snapshots) == 4
    contact, message, drive_file, calendar_event = result.snapshots
    assert contact.source_profile_id == "gws-default"
    assert contact.provider_kind == "gws"
    assert contact.account_id == ""
    assert contact.tenant_id == ""
    assert contact.capability == "people"
    assert contact.source_record_id == "people/c123"
    assert contact.temporal_class == "later_retrieved"
    assert contact.structured_metadata["provider_record_id"] == "people/c123"
    assert message.source_event_at == "2023-12-20T15:30:00Z"
    assert message.temporal_class == "later_retrieved"
    assert message.independence_group_id == "thread-t1"
    assert drive_file.capability == "drive"
    assert calendar_event.capability == "calendar"
    assert [item.evidence_id for item in replay.snapshots] == [
        item.evidence_id for item in result.snapshots
    ]
    assert [call[0] for call in provider.calls] == [
        "people",
        "gmail",
        "drive",
        "calendar",
        "people",
        "gmail",
        "drive",
        "calendar",
    ]


def test_gws_adapter_skips_nonmatching_scope_and_filters_capabilities() -> None:
    provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(records=()),
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )
    other_scope = EvidenceScope(
        source_profile_id="other-profile",
        account_id="",
        tenant_id="",
    )

    skipped = adapter.retrieve(request(scopes=(other_scope,)))
    filtered = adapter.retrieve(request(capabilities=("log_notes",)))

    assert skipped.snapshots == ()
    assert skipped.failures == ()
    assert skipped.warnings == ("provider_scope_skipped",)
    assert filtered.snapshots == ()
    assert filtered.failures == ()
    assert filtered.warnings == ()
    assert provider.calls == []


def test_gws_adapter_labels_empty_query_as_provider_failure() -> None:
    provider = FakeGwsProvider({})
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    result = adapter.retrieve(
        request(capabilities=("people", "gmail"), query_terms=())
    )

    assert result.snapshots == ()
    assert [item["capability"] for item in result.failures] == [
        "people",
        "gmail",
    ]
    assert {
        item["reason_code"] for item in result.failures
    } == {"provider_query_failed"}
    assert result.warnings == ()
    assert provider.calls == []


def test_gws_adapter_bounds_empty_pagination_and_invalid_failures() -> None:
    class AdvancingEmptyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_page(self, **_kwargs: object) -> GwsProviderPage:
            self.calls += 1
            return GwsProviderPage(
                records=(),
                next_page_token=f"page-{self.calls}",
            )

    empty_provider = AdvancingEmptyProvider()
    empty_result = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=empty_provider,
        retrieved_at="2026-07-29T16:00:00Z",
    ).retrieve(request(max_records=1))

    assert empty_provider.calls == 1
    assert [item["reason_code"] for item in empty_result.failures] == [
        "budget_exhausted"
    ]
    assert empty_result.warnings == ("provider_records_truncated",)

    invalid_provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(
                records=tuple(
                    {
                        "provider_record_id": f"people/{index}",
                        "source_type": "gws_contact",
                        "snippet": "invalid",
                        "raw_body": "private",
                    }
                    for index in range(10)
                )
            )
        }
    )
    invalid_result = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=invalid_provider,
        retrieved_at="2026-07-29T16:00:00Z",
    ).retrieve(request(max_records=2))

    assert len(invalid_result.failures) == 2
    assert invalid_result.warnings == ("provider_records_truncated",)


def test_gws_adapter_enforces_pagination_record_and_character_budgets() -> None:
    provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "people/1",
                        "source_type": "gws_contact",
                        "snippet": "first",
                    },
                ),
                next_page_token="page-2",
            ),
            ("people", "page-2"): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "people/2",
                        "source_type": "gws_contact",
                        "snippet": "second",
                    },
                    {
                        "provider_record_id": "people/3",
                        "source_type": "gws_contact",
                        "snippet": "third",
                    },
                ),
            ),
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    result = adapter.retrieve(request(max_records=2, max_characters=9))

    assert [item.source_record_id for item in result.snapshots] == ["people/1"]
    assert result.failures == ()
    assert result.warnings == ("provider_characters_truncated",)
    assert provider.calls == [
        ("people", ("person@example.com", "orchard"), "", 2),
        ("people", ("person@example.com", "orchard"), "page-2", 1),
    ]


def test_gws_adapter_stops_at_record_budget_when_more_pages_exist() -> None:
    provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "people/1",
                        "source_type": "gws_contact",
                        "snippet": "first",
                    },
                ),
                next_page_token="page-2",
            ),
            ("people", "page-2"): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "people/2",
                        "source_type": "gws_contact",
                        "snippet": "second",
                    },
                ),
                next_page_token="page-3",
            ),
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    result = adapter.retrieve(request(max_records=2))

    assert [item.source_record_id for item in result.snapshots] == [
        "people/1",
        "people/2",
    ]
    assert result.warnings == ("provider_records_truncated",)
    assert [call[2] for call in provider.calls] == ["", "page-2"]


def test_gws_adapter_rejects_raw_body_and_preserves_partial_success() -> None:
    provider = FakeGwsProvider(
        {
            ("gmail", ""): GwsProviderPage(
                records=(
                    {
                        "provider_record_id": "message-bad",
                        "source_type": "gws_gmail_message",
                        "snippet": "bounded excerpt",
                        "body": "raw provider body must never enter evidence",
                    },
                    {
                        "provider_record_id": "message-good",
                        "source_type": "gws_gmail_message",
                        "snippet": "safe excerpt",
                    },
                )
            )
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    result = adapter.retrieve(request(capabilities=("gmail",)))

    assert [item.source_record_id for item in result.snapshots] == ["message-good"]
    assert result.failures == (
        {
            "adapter_id": "gws-evidence-v1",
            "source_profile_id": "gws-default",
            "provider_kind": "gws",
            "account_id": "",
            "tenant_id": "",
            "capability": "gmail",
            "reason_code": "provider_response_invalid",
            "detail": "provider record contains forbidden raw body fields",
        },
    )
    assert result.warnings == ("provider_partial_result",)


def test_gws_adapter_freezes_provider_failure_without_fallback() -> None:
    provider = FakeGwsProvider(
        {
            ("people", ""): GwsProviderPage(records=()),
            ("drive", ""): GwsProviderReadError(
                "provider_unavailable",
                "gws executable unavailable",
            ),
        }
    )
    adapter = GwsEvidenceAdapter(
        scope=SCOPE,
        provider=provider,
        retrieved_at="2026-07-29T16:00:00Z",
    )

    result = adapter.retrieve(request(capabilities=("people", "drive")))

    assert result.snapshots == ()
    assert result.failures == (
        {
            "adapter_id": "gws-evidence-v1",
            "source_profile_id": "gws-default",
            "provider_kind": "gws",
            "account_id": "",
            "tenant_id": "",
            "capability": "drive",
            "reason_code": "provider_unavailable",
            "detail": "gws executable unavailable",
        },
    )
    assert result.warnings == ()
    assert [call[0] for call in provider.calls] == ["people", "drive"]


def completed(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def command_params(command: list[str]) -> dict[str, object]:
    return json.loads(command[command.index("--params") + 1])


def test_gws_cli_reader_uses_explicit_env_and_people_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    outcomes = iter(
        (
            completed(
                json.dumps(
                    {
                        "results": [
                            {
                                "person": {
                                    "resourceName": "people/c1",
                                    "names": [{"displayName": "Pat Person"}],
                                    "emailAddresses": [
                                        {"value": "person@example.com"}
                                    ],
                                    "organizations": [{"name": "Orchard Co"}],
                                }
                            }
                        ]
                    }
                )
            ),
            completed(json.dumps({"otherContacts": []})),
            completed(json.dumps({"people": []})),
        )
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return next(outcomes)

    monkeypatch.setattr("conversation_evidence_gws.subprocess.run", fake_run)
    reader = GwsCliReader(
        GwsCliConfig(
            config_dir="/state/gws",
            environment={"PATH": "/opt/gws/bin", "LANG": "C.UTF-8"},
            timeout=7.5,
        )
    )

    page = reader.fetch_page(
        capability="people",
        query_terms=("person@example.com",),
        page_token="",
        page_size=5,
    )

    assert page.records[0] == {
        "provider_record_id": "people/c1",
        "source_type": "gws_contact",
        "snippet": "Pat Person; person@example.com; Orchard Co",
        "structured_metadata": {
            "name": "Pat Person",
            "email": "person@example.com",
            "company": "Orchard Co",
            "surface": "contacts",
            "matched_terms": ["person@example.com"],
            "resource_name": "people/c1",
        },
        "source_uri": "gws://people/people/c1",
    }
    assert [call[0][1:4] for call in calls] == [
        ["people", "people", "searchContacts"],
        ["people", "otherContacts", "search"],
        ["people", "people", "searchDirectoryPeople"],
    ]
    for command, kwargs in calls:
        assert command[-2:] == ["--format", "json"]
        assert kwargs["timeout"] == 7.5
        assert kwargs["env"] == {
            "PATH": "/opt/gws/bin",
            "LANG": "C.UTF-8",
            "GOOGLE_WORKSPACE_CLI_CONFIG_DIR": "/state/gws",
        }


def test_gws_cli_reader_fetches_gmail_metadata_and_snippet_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    outcomes = iter(
        (
            completed(
                json.dumps(
                    {
                        "messages": [{"id": "m1", "threadId": "t1"}],
                        "nextPageToken": "gmail-next",
                    }
                )
            ),
            completed(
                json.dumps(
                    {
                        "id": "m1",
                        "threadId": "t1",
                        "internalDate": "1703500200000",
                        "snippet": "Orchard follow-up",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "pat@example.com"},
                                {"name": "To", "value": "eric@example.com"},
                                {"name": "Subject", "value": "Orchard"},
                                {
                                    "name": "Date",
                                    "value": "Mon, 25 Dec 2023 10:30:00 -0600",
                                },
                            ]
                        },
                    }
                )
            ),
        )
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return next(outcomes)

    monkeypatch.setattr("conversation_evidence_gws.subprocess.run", fake_run)
    reader = GwsCliReader(
        GwsCliConfig(
            config_dir="/state/gws",
            environment={"PATH": "/opt/gws/bin"},
        )
    )

    page = reader.fetch_page(
        capability="gmail",
        query_terms=("pat@example.com", "orchard"),
        page_token="gmail-current",
        page_size=3,
    )

    assert page.next_page_token == "gmail-next"
    assert page.records[0]["source_type"] == "gws_mail_message"
    assert page.records[0]["snippet"] == "Orchard follow-up"
    assert page.records[0]["source_event_at"] == "2023-12-25T16:30:00Z"
    list_params = command_params(calls[0])
    get_params = command_params(calls[1])
    assert list_params["pageToken"] == "gmail-current"
    assert list_params["maxResults"] == 3
    assert get_params["format"] == "metadata"
    assert get_params["metadataHeaders"] == ["From", "To", "Subject", "Date"]
    assert "body" not in json.dumps(get_params).lower()


def test_gws_cli_reader_uses_bounded_drive_and_calendar_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    outcomes = iter(
        (
            completed(
                json.dumps(
                    {
                        "files": [
                            {
                                "id": "f1",
                                "name": "Orchard plan",
                                "mimeType": "application/vnd.google-apps.document",
                                "webViewLink": "https://drive.example/f1",
                                "modifiedTime": "2023-12-28T12:00:00Z",
                            }
                        ],
                        "nextPageToken": "drive-next",
                    }
                )
            ),
            completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "e1",
                                "summary": "Orchard review",
                                "htmlLink": "https://calendar.example/e1",
                                "start": {"dateTime": "2024-01-01T10:00:00-06:00"},
                                "end": {"dateTime": "2024-01-01T11:00:00-06:00"},
                                "organizer": {"email": "eric@example.com"},
                                "attendees": [{"email": "pat@example.com"}],
                            }
                        ],
                        "nextPageToken": "calendar-next",
                    }
                )
            ),
        )
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return next(outcomes)

    monkeypatch.setattr("conversation_evidence_gws.subprocess.run", fake_run)
    reader = GwsCliReader(
        GwsCliConfig(
            config_dir="/state/gws",
            environment={"PATH": "/opt/gws/bin"},
            calendar_id="primary",
        )
    )

    drive_page = reader.fetch_page(
        capability="drive",
        query_terms=("orchard",),
        page_token="drive-current",
        page_size=2,
    )
    calendar_page = reader.fetch_page(
        capability="calendar",
        query_terms=("orchard",),
        page_token="calendar-current",
        page_size=2,
    )

    assert drive_page.next_page_token == "drive-next"
    assert drive_page.records[0]["source_type"] == "gws_docs_file"
    assert calendar_page.next_page_token == "calendar-next"
    assert calendar_page.records[0]["source_event_at"] == (
        "2024-01-01T10:00:00-06:00"
    )
    drive_params = command_params(calls[0])
    calendar_params = command_params(calls[1])
    assert drive_params["pageToken"] == "drive-current"
    assert "fullText contains" in str(drive_params["q"])
    assert "files(id,name,mimeType,webViewLink,modifiedTime" in str(
        drive_params["fields"]
    )
    assert calendar_params["calendarId"] == "primary"
    assert calendar_params["pageToken"] == "calendar-current"
    assert "description" not in str(calendar_params["fields"])


@pytest.mark.parametrize(
    ("outcome", "reason_code"),
    (
        (completed("", returncode=1, stderr="authentication token expired"), "provider_auth_failed"),
        (completed("", returncode=2, stderr="invalid query syntax"), "provider_query_failed"),
        (completed("not-json"), "provider_response_invalid"),
    ),
)
def test_gws_cli_reader_maps_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
    outcome: subprocess.CompletedProcess[str],
    reason_code: str,
) -> None:
    monkeypatch.setattr(
        "conversation_evidence_gws.subprocess.run",
        lambda *_args, **_kwargs: outcome,
    )
    reader = GwsCliReader(
        GwsCliConfig(
            config_dir="/state/gws",
            environment={"PATH": "/opt/gws/bin"},
        )
    )

    with pytest.raises(GwsProviderReadError) as caught:
        reader.fetch_page(
            capability="drive",
            query_terms=("orchard",),
            page_token="",
            page_size=2,
        )

    assert caught.value.reason_code == reason_code
    if outcome.returncode:
        assert str(outcome.stderr or outcome.stdout or "") not in caught.value.detail
        assert caught.value.detail == (
            f"gws command failed with status {outcome.returncode}"
        )
