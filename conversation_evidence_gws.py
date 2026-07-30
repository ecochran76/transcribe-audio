from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from conversation_evidence_adapters import (
    ADAPTER_FAILURE_REASON_CODES,
    MAX_ADAPTER_FAILURE_DETAIL_CHARS,
    AdapterSourceScope,
    BoundedProviderRecord,
    EvidenceSnapshotNormalizer,
    adapter_failure,
    adapter_warning,
)
from conversation_identity_retrieval import (
    ProviderRetrievalRequest,
    ProviderRetrievalResult,
)


GWS_CAPABILITIES = ("calendar", "drive", "gmail", "people")
GWS_SOURCE_TYPES = {
    "calendar": frozenset(
        {
            "gws_calendar_event_detail",
            "gws_calendar_overlap",
        }
    ),
    "drive": frozenset(
        {
            "gws_docs_file",
            "gws_drive_document",
            "gws_drive_file",
        }
    ),
    "gmail": frozenset({"gws_gmail_message", "gws_mail_message"}),
    "people": frozenset(
        {
            "gws_contact",
            "gws_directory_person",
            "gws_other_contact",
        }
    ),
}
GWS_METADATA_FIELDS = (
    "attendee_emails",
    "calendar_id",
    "cc_emails",
    "company",
    "content_scope",
    "created_time",
    "date",
    "drive_id",
    "email",
    "end",
    "event_id",
    "from",
    "from_email",
    "internal_date",
    "matched_emails",
    "matched_terms",
    "mime_type",
    "modified_time",
    "name",
    "organizer_email",
    "owners",
    "parents",
    "phone",
    "profile",
    "query",
    "resource_name",
    "start",
    "subject",
    "surface",
    "thread_id",
    "to",
    "to_emails",
)
_RECORD_FIELDS = frozenset(
    {
        "provider_record_id",
        "source_type",
        "snippet",
        "structured_metadata",
        "source_event_at",
        "source_uri",
        "source_record_id",
        "independence_group_id",
        "freshness_state",
        "expires_at",
        "redaction",
        "truncation",
    }
)
_RAW_BODY_FIELDS = frozenset(
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


@dataclass(frozen=True)
class GwsCliConfig:
    config_dir: str
    environment: Mapping[str, str]
    timeout: float = 20.0
    executable: str = "gws"
    user_id: str = "me"
    calendar_id: str = "primary"
    people_surfaces: tuple[str, ...] = (
        "contacts",
        "other_contacts",
        "directory",
    )

    def __post_init__(self) -> None:
        if not self.config_dir.strip():
            raise ValueError("GWS CLI config_dir must be explicit.")
        if not self.environment:
            raise ValueError("GWS CLI environment must be explicit.")
        if self.timeout <= 0:
            raise ValueError("GWS CLI timeout must be positive.")
        if not self.executable.strip():
            raise ValueError("GWS CLI executable is required.")
        if not self.people_surfaces:
            raise ValueError("At least one GWS CLI people surface is required.")
        if set(self.people_surfaces) - {
            "contacts",
            "other_contacts",
            "directory",
        }:
            raise ValueError("GWS CLI people surface is unsupported.")


@dataclass(frozen=True)
class GwsProviderPage:
    records: tuple[Mapping[str, Any], ...]
    next_page_token: str = ""


class GwsProviderReader(Protocol):
    def fetch_page(
        self,
        *,
        capability: str,
        query_terms: tuple[str, ...],
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage: ...


class GwsProviderReadError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        if reason_code not in ADAPTER_FAILURE_REASON_CODES:
            raise ValueError("Unsupported GWS provider failure reason code.")
        bounded_detail = str(detail or "")
        super().__init__(bounded_detail)
        self.reason_code = reason_code
        self.detail = bounded_detail


class GwsCliReader:
    def __init__(self, config: GwsCliConfig) -> None:
        self.config = config

    def fetch_page(
        self,
        *,
        capability: str,
        query_terms: tuple[str, ...],
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        if capability not in GWS_CAPABILITIES:
            raise GwsProviderReadError(
                "unsupported_capability",
                f"unsupported GWS capability: {capability}",
            )
        bounded_size = max(1, min(int(page_size), 100))
        terms = tuple(
            term.strip() for term in query_terms if str(term).strip()
        )
        if not terms:
            return GwsProviderPage(records=())
        if capability == "people":
            return self._fetch_people(
                terms,
                page_token=page_token,
                page_size=bounded_size,
            )
        if capability == "gmail":
            return self._fetch_gmail(
                terms,
                page_token=page_token,
                page_size=bounded_size,
            )
        if capability == "drive":
            return self._fetch_drive(
                terms,
                page_token=page_token,
                page_size=bounded_size,
            )
        return self._fetch_calendar(
            terms,
            page_token=page_token,
            page_size=bounded_size,
        )

    def _run(self, command: list[str], params: dict[str, Any]) -> Any:
        full_command = [
            self.config.executable,
            *command,
            "--params",
            json.dumps(params, separators=(",", ":"), ensure_ascii=False),
            "--format",
            "json",
        ]
        environment = {
            str(key): str(value)
            for key, value in self.config.environment.items()
        }
        environment["GOOGLE_WORKSPACE_CLI_CONFIG_DIR"] = str(
            Path(self.config.config_dir).expanduser()
        )
        try:
            result = subprocess.run(
                full_command,
                text=True,
                capture_output=True,
                timeout=self.config.timeout,
                check=False,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise GwsProviderReadError(
                "provider_unavailable",
                "gws executable unavailable",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GwsProviderReadError(
                "provider_unavailable",
                f"gws command timed out after {self.config.timeout:g} seconds",
            ) from exc
        if result.returncode != 0:
            diagnostic = str(result.stderr or result.stdout or "").strip()
            lowered = diagnostic.casefold()
            reason_code = (
                "provider_auth_failed"
                if any(
                    marker in lowered
                    for marker in (
                        "auth",
                        "credential",
                        "permission denied",
                        "token",
                        "unauthorized",
                    )
                )
                else "provider_query_failed"
            )
            raise GwsProviderReadError(
                reason_code,
                f"gws command failed with status {result.returncode}",
            )
        text = str(result.stdout or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GwsProviderReadError(
                "provider_response_invalid",
                "gws command did not return valid JSON",
            ) from exc

    def _fetch_people(
        self,
        terms: tuple[str, ...],
        *,
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        term_index, surface_index, provider_token = self._people_cursor(page_token)
        records: list[Mapping[str, Any]] = []
        surfaces = self.config.people_surfaces
        if term_index >= len(terms) or surface_index >= len(surfaces):
            raise GwsProviderReadError(
                "provider_response_invalid",
                "GWS people pagination token is outside the query bounds",
            )
        while term_index < len(terms) and len(records) < page_size:
            surface = surfaces[surface_index]
            command, params, source_type, result_field = self._people_command(
                surface,
                query=terms[term_index],
                page_size=page_size - len(records),
                page_token=provider_token,
            )
            payload = self._run(command, params)
            if not isinstance(payload, dict):
                raise GwsProviderReadError(
                    "provider_response_invalid",
                    "gws people response must be an object",
                )
            values = payload.get(result_field)
            people = values if isinstance(values, list) else []
            for value in people:
                person = (
                    value.get("person")
                    if result_field == "results" and isinstance(value, dict)
                    else value
                )
                if not isinstance(person, dict):
                    continue
                records.append(
                    self._people_record(
                        person,
                        source_type=source_type,
                        surface=surface,
                        query=terms[term_index],
                    )
                )
                if len(records) >= page_size:
                    break

            next_provider_token = str(payload.get("nextPageToken") or "")
            if next_provider_token:
                next_cursor = self._encode_people_cursor(
                    term_index,
                    surface_index,
                    next_provider_token,
                )
            else:
                next_term = term_index
                next_surface = surface_index + 1
                if next_surface >= len(surfaces):
                    next_term += 1
                    next_surface = 0
                next_cursor = (
                    self._encode_people_cursor(next_term, next_surface, "")
                    if next_term < len(terms)
                    else ""
                )
            if len(records) >= page_size:
                return GwsProviderPage(
                    records=tuple(records),
                    next_page_token=next_cursor,
                )
            if next_provider_token:
                return GwsProviderPage(
                    records=tuple(records),
                    next_page_token=next_cursor,
                )
            else:
                surface_index += 1
                provider_token = ""
                if surface_index >= len(surfaces):
                    term_index += 1
                    surface_index = 0
        return GwsProviderPage(records=tuple(records))

    def _people_command(
        self,
        surface: str,
        *,
        query: str,
        page_size: int,
        page_token: str,
    ) -> tuple[list[str], dict[str, Any], str, str]:
        if surface == "contacts":
            command = ["people", "people", "searchContacts"]
            params: dict[str, Any] = {
                "query": query,
                "readMask": "names,emailAddresses,organizations,metadata",
                "pageSize": min(page_size, 30),
            }
            source_type = "gws_contact"
            result_field = "results"
        elif surface == "other_contacts":
            command = ["people", "otherContacts", "search"]
            params = {
                "query": query,
                "readMask": "names,emailAddresses,phoneNumbers,metadata",
                "pageSize": min(page_size, 30),
            }
            source_type = "gws_other_contact"
            result_field = "otherContacts"
        else:
            command = ["people", "people", "searchDirectoryPeople"]
            params = {
                "query": query,
                "readMask": "names,emailAddresses,organizations,metadata",
                "pageSize": min(page_size, 100),
                "sources": [
                    "DIRECTORY_SOURCE_TYPE_DOMAIN_PROFILE",
                    "DIRECTORY_SOURCE_TYPE_DOMAIN_CONTACT",
                ],
            }
            source_type = "gws_directory_person"
            result_field = "people"
        if page_token:
            params["pageToken"] = page_token
        return command, params, source_type, result_field

    def _people_record(
        self,
        person: dict[str, Any],
        *,
        source_type: str,
        surface: str,
        query: str,
    ) -> Mapping[str, Any]:
        resource_name = str(person.get("resourceName") or "").strip()
        names = person.get("names") if isinstance(person.get("names"), list) else []
        emails = (
            person.get("emailAddresses")
            if isinstance(person.get("emailAddresses"), list)
            else []
        )
        organizations = (
            person.get("organizations")
            if isinstance(person.get("organizations"), list)
            else []
        )
        name = self._first_value(names, "displayName")
        email = self._first_value(emails, "value")
        company = self._first_value(organizations, "name")
        return {
            "provider_record_id": resource_name,
            "source_type": source_type,
            "snippet": "; ".join(
                value for value in (name, email, company) if value
            ),
            "structured_metadata": {
                "name": name,
                "email": email,
                "company": company,
                "surface": surface,
                "matched_terms": [query],
                "resource_name": resource_name,
            },
            "source_uri": f"gws://people/{resource_name}",
        }

    def _fetch_gmail(
        self,
        terms: tuple[str, ...],
        *,
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        params: dict[str, Any] = {
            "userId": self.config.user_id,
            "q": " OR ".join(f'"{term}"' for term in terms),
            "maxResults": min(page_size, 50),
        }
        if page_token:
            params["pageToken"] = page_token
        payload = self._run(["gmail", "users", "messages", "list"], params)
        if not isinstance(payload, dict):
            raise GwsProviderReadError(
                "provider_response_invalid",
                "gws Gmail list response must be an object",
            )
        listed = payload.get("messages")
        messages = listed if isinstance(listed, list) else []
        records: list[Mapping[str, Any]] = []
        for listed_message in messages[:page_size]:
            if not isinstance(listed_message, dict):
                continue
            message_id = str(listed_message.get("id") or "").strip()
            if not message_id:
                continue
            message = self._run(
                ["gmail", "users", "messages", "get"],
                {
                    "userId": self.config.user_id,
                    "id": message_id,
                    "format": "metadata",
                    "metadataHeaders": ["From", "To", "Subject", "Date"],
                },
            )
            if not isinstance(message, dict):
                raise GwsProviderReadError(
                    "provider_response_invalid",
                    "gws Gmail metadata response must be an object",
                )
            records.append(self._gmail_record(message, terms=terms))
        return GwsProviderPage(
            records=tuple(records),
            next_page_token=str(payload.get("nextPageToken") or ""),
        )

    def _gmail_record(
        self,
        message: dict[str, Any],
        *,
        terms: tuple[str, ...],
    ) -> Mapping[str, Any]:
        message_id = str(message.get("id") or "").strip()
        part = message.get("payload")
        headers_list = (
            part.get("headers")
            if isinstance(part, dict) and isinstance(part.get("headers"), list)
            else []
        )
        headers = {
            str(item.get("name") or "").strip().casefold(): str(
                item.get("value") or ""
            ).strip()
            for item in headers_list
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        event_at = self._email_date(headers.get("date", ""))
        return {
            "provider_record_id": message_id,
            "source_type": "gws_mail_message",
            "snippet": str(message.get("snippet") or "")[:600],
            "structured_metadata": {
                "subject": headers.get("subject", ""),
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "date": headers.get("date", ""),
                "thread_id": str(message.get("threadId") or ""),
                "internal_date": str(message.get("internalDate") or ""),
                "matched_terms": list(terms),
                "content_scope": "metadata_and_snippet_only",
            },
            "source_event_at": event_at,
            "source_uri": (
                f"https://mail.google.com/mail/u/0/#all/{message_id}"
            ),
            "independence_group_id": str(message.get("threadId") or ""),
        }

    def _fetch_drive(
        self,
        terms: tuple[str, ...],
        *,
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        query = " or ".join(
            f"fullText contains '{self._escape_drive(term)}'" for term in terms
        )
        params: dict[str, Any] = {
            "q": query,
            "pageSize": page_size,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "fields": (
                "nextPageToken,files(id,name,mimeType,webViewLink,"
                "modifiedTime,createdTime,owners(displayName,emailAddress),"
                "driveId,parents)"
            ),
            "orderBy": "modifiedTime desc",
        }
        if page_token:
            params["pageToken"] = page_token
        payload = self._run(["drive", "files", "list"], params)
        if not isinstance(payload, dict):
            raise GwsProviderReadError(
                "provider_response_invalid",
                "gws Drive list response must be an object",
            )
        values = payload.get("files")
        files = values if isinstance(values, list) else []
        records = tuple(
            self._drive_record(item, terms=terms)
            for item in files[:page_size]
            if isinstance(item, dict)
        )
        return GwsProviderPage(
            records=records,
            next_page_token=str(payload.get("nextPageToken") or ""),
        )

    def _drive_record(
        self,
        item: dict[str, Any],
        *,
        terms: tuple[str, ...],
    ) -> Mapping[str, Any]:
        file_id = str(item.get("id") or "").strip()
        mime_type = str(item.get("mimeType") or "")
        return {
            "provider_record_id": file_id,
            "source_type": (
                "gws_docs_file"
                if mime_type == "application/vnd.google-apps.document"
                else "gws_drive_file"
            ),
            "snippet": str(item.get("name") or file_id),
            "structured_metadata": {
                "name": str(item.get("name") or ""),
                "mime_type": mime_type,
                "modified_time": str(item.get("modifiedTime") or ""),
                "created_time": str(item.get("createdTime") or ""),
                "owners": item.get("owners") or [],
                "drive_id": str(item.get("driveId") or ""),
                "parents": item.get("parents") or [],
                "matched_terms": list(terms),
            },
            "source_event_at": str(item.get("modifiedTime") or ""),
            "source_uri": str(item.get("webViewLink") or ""),
        }

    def _fetch_calendar(
        self,
        terms: tuple[str, ...],
        *,
        page_token: str,
        page_size: int,
    ) -> GwsProviderPage:
        params: dict[str, Any] = {
            "calendarId": self.config.calendar_id,
            "q": " ".join(terms),
            "maxResults": page_size,
            "singleEvents": True,
            "orderBy": "startTime",
            "fields": (
                "nextPageToken,items(id,summary,htmlLink,start,end,"
                "organizer(email),attendees(email))"
            ),
        }
        if page_token:
            params["pageToken"] = page_token
        payload = self._run(["calendar", "events", "list"], params)
        if not isinstance(payload, dict):
            raise GwsProviderReadError(
                "provider_response_invalid",
                "gws Calendar list response must be an object",
            )
        values = payload.get("items")
        events = values if isinstance(values, list) else []
        records = tuple(
            self._calendar_record(item, terms=terms)
            for item in events[:page_size]
            if isinstance(item, dict)
        )
        return GwsProviderPage(
            records=records,
            next_page_token=str(payload.get("nextPageToken") or ""),
        )

    def _calendar_record(
        self,
        item: dict[str, Any],
        *,
        terms: tuple[str, ...],
    ) -> Mapping[str, Any]:
        event_id = str(item.get("id") or "").strip()
        start = item.get("start") if isinstance(item.get("start"), dict) else {}
        end = item.get("end") if isinstance(item.get("end"), dict) else {}
        organizer = (
            item.get("organizer")
            if isinstance(item.get("organizer"), dict)
            else {}
        )
        attendees = (
            item.get("attendees")
            if isinstance(item.get("attendees"), list)
            else []
        )
        return {
            "provider_record_id": event_id,
            "source_type": "gws_calendar_event_detail",
            "snippet": str(item.get("summary") or event_id),
            "structured_metadata": {
                "calendar_id": self.config.calendar_id,
                "event_id": event_id,
                "start": start,
                "end": end,
                "organizer_email": str(organizer.get("email") or ""),
                "attendee_emails": [
                    str(attendee.get("email") or "")
                    for attendee in attendees
                    if isinstance(attendee, dict)
                    and str(attendee.get("email") or "")
                ],
                "matched_terms": list(terms),
            },
            "source_event_at": str(start.get("dateTime") or ""),
            "source_uri": str(item.get("htmlLink") or ""),
        }

    @staticmethod
    def _first_value(values: list[Any], key: str) -> str:
        for value in values:
            if isinstance(value, dict) and str(value.get(key) or "").strip():
                return str(value.get(key)).strip()
        return ""

    @staticmethod
    def _email_date(value: str) -> str:
        if not value:
            return ""
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return ""
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return ""
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _escape_drive(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _encode_people_cursor(
        term_index: int,
        surface_index: int,
        provider_token: str,
    ) -> str:
        payload = json.dumps(
            [term_index, surface_index, provider_token],
            separators=(",", ":"),
        ).encode("utf-8")
        return "people:" + base64.urlsafe_b64encode(payload).decode("ascii")

    @staticmethod
    def _people_cursor(value: str) -> tuple[int, int, str]:
        if not value:
            return 0, 0, ""
        if not value.startswith("people:"):
            raise GwsProviderReadError(
                "provider_response_invalid",
                "invalid GWS people pagination token",
            )
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(value.removeprefix("people:")).decode(
                    "utf-8"
                )
            )
            term_index, surface_index, provider_token = payload
            if int(term_index) < 0 or int(surface_index) < 0:
                raise ValueError
            return int(term_index), int(surface_index), str(provider_token)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise GwsProviderReadError(
                "provider_response_invalid",
                "invalid GWS people pagination token",
            ) from exc


class GwsEvidenceAdapter:
    adapter_id = "gws-evidence-v1"

    def __init__(
        self,
        *,
        scope: AdapterSourceScope,
        provider: GwsProviderReader,
        retrieved_at: str,
    ) -> None:
        if scope.provider_kind != "gws":
            raise ValueError("GWS evidence adapter requires provider_kind='gws'.")
        unsupported = set(scope.capabilities) - set(GWS_CAPABILITIES)
        if unsupported:
            raise ValueError("GWS evidence scope contains unsupported capabilities.")
        self.scope = scope
        self.provider = provider
        self.retrieved_at = retrieved_at
        self.normalizer = EvidenceSnapshotNormalizer(
            scope=scope,
            allowed_source_types=tuple(
                sorted(
                    source_type
                    for capability in scope.capabilities
                    for source_type in GWS_SOURCE_TYPES[capability]
                )
            ),
            allowed_metadata_fields=GWS_METADATA_FIELDS,
        )

    def retrieve(
        self,
        request: ProviderRetrievalRequest,
    ) -> ProviderRetrievalResult:
        if not self._scope_is_requested(request):
            return ProviderRetrievalResult(
                warnings=(adapter_warning("provider_scope_skipped"),)
            )

        capabilities = tuple(
            capability
            for capability in request.capabilities
            if capability in self.scope.capabilities
            and capability in GWS_CAPABILITIES
        )
        if not capabilities:
            return ProviderRetrievalResult()
        if not any(str(term or "").strip() for term in request.query_terms):
            return ProviderRetrievalResult(
                failures=tuple(
                    adapter_failure(
                        adapter_id=self.adapter_id,
                        scope=self.scope,
                        capability=capability,
                        reason_code="provider_query_failed",
                        detail="query terms are required",
                    )
                    for capability in capabilities
                )
            )

        if request.max_records <= 0 or request.max_characters <= 0:
            return ProviderRetrievalResult(
                failures=(
                    adapter_failure(
                        adapter_id=self.adapter_id,
                        scope=self.scope,
                        capability=capabilities[0],
                        reason_code="budget_exhausted",
                        detail="retrieval request has no remaining evidence budget",
                    ),
                )
            )

        snapshots = []
        failures: list[dict[str, str]] = []
        warnings: list[str] = []
        characters = 0
        inspected_records = 0
        stop_for_budget = False

        for capability_index, capability in enumerate(capabilities):
            remaining_global_records = request.max_records - inspected_records
            if remaining_global_records <= 0:
                self._add_warning(warnings, "provider_records_truncated")
                break
            remaining_capabilities = len(capabilities) - capability_index
            capability_record_budget = max(
                1,
                remaining_global_records // remaining_capabilities,
            )
            capability_records_inspected = 0
            capability_truncated = False
            page_token = ""
            seen_page_tokens: set[str] = set()
            more_records_available = False
            page_attempts = 0
            max_page_attempts = max(1, min(capability_record_budget, 100))
            while (
                len(snapshots) < request.max_records
                and inspected_records < request.max_records
                and capability_records_inspected < capability_record_budget
            ):
                if page_attempts >= max_page_attempts:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.scope,
                            capability=capability,
                            reason_code="budget_exhausted",
                            detail="provider pagination page budget exhausted",
                        )
                    )
                    self._add_warning(warnings, "provider_records_truncated")
                    break
                page_attempts += 1
                remaining_records = min(
                    request.max_records - inspected_records,
                    capability_record_budget - capability_records_inspected,
                )
                try:
                    page = self.provider.fetch_page(
                        capability=capability,
                        query_terms=request.query_terms,
                        page_token=page_token,
                        page_size=remaining_records,
                    )
                except GwsProviderReadError as exc:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.scope,
                            capability=capability,
                            reason_code=exc.reason_code,
                            detail=exc.detail[:MAX_ADAPTER_FAILURE_DETAIL_CHARS],
                        )
                    )
                    break
                except Exception as exc:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.scope,
                            capability=capability,
                            reason_code="provider_query_failed",
                            detail=type(exc).__name__[
                                :MAX_ADAPTER_FAILURE_DETAIL_CHARS
                            ],
                        )
                    )
                    break

                if not isinstance(page, GwsProviderPage):
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.scope,
                            capability=capability,
                            reason_code="provider_response_invalid",
                            detail="provider page has an invalid shape",
                        )
                    )
                    break

                more_records_available = False
                for payload in page.records:
                    if inspected_records >= request.max_records:
                        self._add_warning(warnings, "provider_records_truncated")
                        more_records_available = True
                        stop_for_budget = True
                        break
                    if capability_records_inspected >= capability_record_budget:
                        self._add_warning(warnings, "provider_records_truncated")
                        more_records_available = True
                        capability_truncated = True
                        break
                    inspected_records += 1
                    capability_records_inspected += 1
                    try:
                        record = self._bounded_record(payload, capability=capability)
                        snapshot = self.normalizer.normalize(
                            record,
                            as_of=request.as_of,
                            retrieved_at=self.retrieved_at,
                        )
                    except (TypeError, ValueError) as exc:
                        failures.append(
                            adapter_failure(
                                adapter_id=self.adapter_id,
                                scope=self.scope,
                                capability=capability,
                                reason_code="provider_response_invalid",
                                detail=str(exc)[
                                    :MAX_ADAPTER_FAILURE_DETAIL_CHARS
                                ],
                            )
                        )
                        if inspected_records >= request.max_records:
                            self._add_warning(
                                warnings,
                                "provider_records_truncated",
                            )
                            stop_for_budget = True
                            break
                        continue

                    next_characters = characters + len(snapshot.snippet)
                    if next_characters > request.max_characters:
                        self._add_warning(
                            warnings,
                            "provider_characters_truncated",
                        )
                        stop_for_budget = True
                        break
                    snapshots.append(snapshot)
                    characters = next_characters

                if stop_for_budget:
                    break
                if capability_truncated:
                    break
                next_page_token = str(page.next_page_token or "").strip()
                if not next_page_token:
                    break
                more_records_available = True
                if next_page_token in seen_page_tokens:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.scope,
                            capability=capability,
                            reason_code="provider_response_invalid",
                            detail="provider repeated a pagination token",
                        )
                    )
                    break
                seen_page_tokens.add(next_page_token)
                page_token = next_page_token

            if (
                len(snapshots) >= request.max_records
                and more_records_available
                and not stop_for_budget
            ):
                self._add_warning(warnings, "provider_records_truncated")
                stop_for_budget = True
            elif (
                capability_records_inspected >= capability_record_budget
                and more_records_available
            ):
                self._add_warning(warnings, "provider_records_truncated")
            if stop_for_budget:
                break

        if failures and snapshots:
            self._add_warning(warnings, "provider_partial_result")
        return ProviderRetrievalResult(
            snapshots=tuple(snapshots),
            failures=tuple(failures),
            warnings=tuple(warnings),
        )

    def _scope_is_requested(self, request: ProviderRetrievalRequest) -> bool:
        return any(
            scope.source_profile_id == self.scope.source_profile_id
            and scope.account_id == self.scope.account_id
            and scope.tenant_id == self.scope.tenant_id
            for scope in request.scopes
        )

    def _bounded_record(
        self,
        payload: Mapping[str, Any],
        *,
        capability: str,
    ) -> BoundedProviderRecord:
        if not isinstance(payload, Mapping):
            raise TypeError("provider record must be an object")
        payload_keys = {str(key).strip().casefold() for key in payload}
        if payload_keys & _RAW_BODY_FIELDS:
            raise ValueError("provider record contains forbidden raw body fields")

        values = {key: payload.get(key) for key in _RECORD_FIELDS}
        source_type = str(values["source_type"] or "")
        if source_type not in GWS_SOURCE_TYPES[capability]:
            raise ValueError("provider record source_type does not match capability")
        provider_record_id = str(values["provider_record_id"] or "")
        metadata = values["structured_metadata"]
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("structured_metadata must be an object")

        return BoundedProviderRecord(
            provider_record_id=provider_record_id,
            source_type=source_type,
            capability=capability,
            snippet=str(values["snippet"] or ""),
            structured_metadata=dict(metadata),
            source_event_at=str(values["source_event_at"] or ""),
            source_uri=str(values["source_uri"] or ""),
            source_record_id=str(values["source_record_id"] or provider_record_id),
            independence_group_id=str(values["independence_group_id"] or ""),
            freshness_state=str(values["freshness_state"] or "current"),
            expires_at=str(values["expires_at"] or ""),
            redaction=self._object_field(values["redaction"], "redaction"),
            truncation=self._object_field(values["truncation"], "truncation"),
        )

    @staticmethod
    def _object_field(value: Any, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"{field_name} must be an object")
        return dict(value)

    @staticmethod
    def _add_warning(warnings: list[str], code: str) -> None:
        warning = adapter_warning(code)
        if warning not in warnings:
            warnings.append(warning)
