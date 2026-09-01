from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Callable

from conversation_evidence_adapters import (
    AdapterSourceScope,
    BoundedProviderRecord,
    EvidenceSnapshotNormalizer,
    adapter_failure,
    adapter_warning,
)
from conversation_evidence_fabric import (
    ProviderRetrievalRequest,
    ProviderRetrievalResult,
)


ODOLLO_ADAPTER_ID = "odollo-evidence.v1"
ODOLLO_SOURCE_TYPES = (
    "odollo_contact",
    "odollo_lead",
    "odollo_log_note",
)
ODOLLO_METADATA_FIELDS = (
    "author",
    "company",
    "contact_name",
    "email",
    "model",
    "name",
    "record_id",
    "related_model",
    "related_record_id",
    "subject",
)
_RAW_BODY_KEYS = frozenset(
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
class OdolloAdapterConfig:
    scope: AdapterSourceScope
    command: tuple[str, ...]
    repo_root: Path
    config_path: Path
    timeout: float = 30.0


@dataclass(frozen=True)
class _CapabilitySpec:
    model: str
    source_type: str
    fields: tuple[str, ...]
    search_fields: tuple[str, ...]


_CAPABILITY_SPECS = {
    "contacts": _CapabilitySpec(
        model="res.partner",
        source_type="odollo_contact",
        fields=("id", "name", "email", "parent_id"),
        search_fields=("name", "email"),
    ),
    "leads": _CapabilitySpec(
        model="crm.lead",
        source_type="odollo_lead",
        fields=(
            "id",
            "name",
            "contact_name",
            "email_from",
            "partner_id",
            "partner_name",
            "create_date",
        ),
        search_fields=("name", "contact_name", "email_from", "partner_name"),
    ),
    "log_notes": _CapabilitySpec(
        model="mail.message",
        source_type="odollo_log_note",
        fields=("id", "subject", "model", "res_id", "date", "author_id"),
        search_fields=("subject", "body"),
    ),
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _m2o_label(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _text(value[1])
    return _text(value)


def _strip_html(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", _text(value))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _odoo_utc_timestamp(value: Any) -> str:
    text = _text(value)
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?",
        text,
    ):
        return text.replace(" ", "T", 1) + "Z"
    return text


def _raw_body_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
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


def _or_domain(clauses: list[list[Any]]) -> list[Any]:
    if not clauses:
        return []
    if len(clauses) == 1:
        return clauses[0]
    return ["|"] * (len(clauses) - 1) + clauses


def _domain(spec: _CapabilitySpec, terms: tuple[str, ...]) -> list[Any]:
    clauses = [
        [field, "ilike", term]
        for term in terms
        if _text(term)
        for field in spec.search_fields
    ]
    search = _or_domain(clauses)
    if spec.model == "mail.message":
        if not search:
            return [["message_type", "=", "comment"]]
        return ["&", ["message_type", "=", "comment"], *search]
    return search


def _matching_scope(
    request: ProviderRetrievalRequest,
    scope: AdapterSourceScope,
) -> bool:
    return any(
        item.source_profile_id == scope.source_profile_id
        and item.account_id == scope.account_id
        and item.tenant_id == scope.tenant_id
        for item in request.scopes
    )


class OdolloEvidenceAdapter:
    adapter_id = ODOLLO_ADAPTER_ID

    def __init__(
        self,
        config: OdolloAdapterConfig,
        *,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        retrieved_at: Callable[[], str] = _utc_now,
    ) -> None:
        if config.scope.provider_kind != "odollo":
            raise ValueError("Odollo adapter requires provider_kind='odollo'.")
        if not config.scope.tenant_id.strip():
            raise ValueError("Odollo adapter requires an explicit tenant_id.")
        if not config.command or not _text(config.command[0]):
            raise ValueError("Odollo adapter requires a configured command.")
        if config.timeout <= 0:
            raise ValueError("Odollo adapter timeout must be positive.")
        unsupported = set(config.scope.capabilities) - set(_CAPABILITY_SPECS)
        if unsupported:
            raise ValueError("Odollo adapter scope contains unsupported capabilities.")
        self.config = config
        self._run_command = run_command
        self._retrieved_at = retrieved_at
        self._normalizer = EvidenceSnapshotNormalizer(
            scope=config.scope,
            allowed_source_types=ODOLLO_SOURCE_TYPES,
            allowed_metadata_fields=ODOLLO_METADATA_FIELDS,
        )

    def retrieve(
        self,
        request: ProviderRetrievalRequest,
    ) -> ProviderRetrievalResult:
        if not _matching_scope(request, self.config.scope):
            return ProviderRetrievalResult(
                warnings=(adapter_warning("provider_scope_skipped"),),
            )

        capabilities = [
            capability
            for capability in self.config.scope.capabilities
            if capability in request.capabilities
        ]
        if capabilities and not any(_text(term) for term in request.query_terms):
            return ProviderRetrievalResult(
                failures=tuple(
                    self._failure(
                        capability,
                        "provider_query_failed",
                        "query terms are required",
                    )
                    for capability in capabilities
                ),
            )
        snapshots = []
        failures: list[dict[str, str]] = []
        warnings: list[str] = []
        remaining_records = max(0, request.max_records)
        remaining_characters = max(0, request.max_characters)
        retrieved_at = self._retrieved_at()

        for capability in capabilities:
            if remaining_records <= 0:
                warnings.append(adapter_warning("provider_records_truncated"))
                break
            if remaining_characters <= 0:
                warnings.append(adapter_warning("provider_characters_truncated"))
                break
            spec = _CAPABILITY_SPECS[capability]
            rows, failure = self._search(
                capability=capability,
                spec=spec,
                terms=request.query_terms,
                limit=remaining_records,
            )
            if failure is not None:
                failures.append(failure)
                continue
            if len(rows) > remaining_records:
                rows = rows[:remaining_records]
                warnings.append(adapter_warning("provider_records_truncated"))

            capability_failed = False
            for row in rows:
                if _raw_body_keys(row):
                    failures.append(
                        self._failure(
                            capability,
                            "provider_response_invalid",
                            "query response contained a prohibited raw body",
                        )
                    )
                    capability_failed = True
                    break
                try:
                    record = self._bounded_record(
                        capability=capability,
                        spec=spec,
                        row=row,
                    )
                except ValueError as exc:
                    failures.append(
                        self._failure(
                            capability,
                            "provider_response_invalid",
                            str(exc),
                        )
                    )
                    capability_failed = True
                    break

                if len(record.snippet) > remaining_characters:
                    retained = record.snippet[:remaining_characters]
                    record = replace(
                        record,
                        snippet=retained,
                        truncation={
                            "snippet_original_characters": len(record.snippet),
                            "snippet_retained_characters": len(retained),
                        },
                    )
                    warnings.append(
                        adapter_warning("provider_characters_truncated")
                    )
                try:
                    snapshot = self._normalizer.normalize(
                        record,
                        as_of=request.as_of,
                        retrieved_at=retrieved_at,
                    )
                except ValueError as exc:
                    failures.append(
                        self._failure(
                            capability,
                            "provider_response_invalid",
                            str(exc),
                        )
                    )
                    capability_failed = True
                    break
                snapshots.append(snapshot)
                remaining_records -= 1
                remaining_characters -= len(record.snippet)
                if remaining_records <= 0 or remaining_characters <= 0:
                    break
            if capability_failed:
                continue

        if failures and snapshots:
            warnings.append(adapter_warning("provider_partial_result"))
        return ProviderRetrievalResult(
            snapshots=tuple(snapshots),
            failures=tuple(failures),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _search(
        self,
        *,
        capability: str,
        spec: _CapabilitySpec,
        terms: tuple[str, ...],
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
        command = [
            *self._command_prefix(),
            "--config",
            str(self.config.config_path.expanduser()),
            "--profile",
            self.config.scope.tenant_id,
            "--timeout",
            str(int(self.config.timeout)),
            "odoo",
            "records",
            "search",
            "--model",
            spec.model,
            "--domain",
            json.dumps(_domain(spec, terms), separators=(",", ":")),
            "--fields",
            ",".join(spec.fields),
            "--limit",
            str(limit),
        ]
        try:
            result = self._run_command(
                command,
                text=True,
                capture_output=True,
                timeout=self.config.timeout,
                check=False,
                cwd=self.config.repo_root.expanduser(),
            )
        except FileNotFoundError:
            return [], self._failure(
                capability,
                "provider_unavailable",
                "configured executable was not found",
            )
        except subprocess.TimeoutExpired:
            return [], self._failure(
                capability,
                "provider_unavailable",
                f"query exceeded {self.config.timeout:g} seconds",
            )
        if result.returncode != 0:
            diagnostic = f"{result.stderr or ''} {result.stdout or ''}".casefold()
            reason = (
                "provider_auth_failed"
                if any(token in diagnostic for token in ("auth", "credential", "unauthorized"))
                else "provider_query_failed"
            )
            return [], self._failure(
                capability,
                reason,
                f"query exited with status {result.returncode}",
            )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return [], self._failure(
                capability,
                "provider_response_invalid",
                "query response was not valid JSON",
            )
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            return [], self._failure(
                capability,
                "provider_response_invalid",
                "query response was not a record list",
            )
        return payload, None

    def _command_prefix(self) -> list[str]:
        prefix = list(self.config.command)
        if prefix[0].startswith("~"):
            prefix[0] = str(Path(prefix[0]).expanduser())
        return prefix

    def _failure(
        self,
        capability: str,
        reason_code: str,
        detail: str,
    ) -> dict[str, str]:
        return adapter_failure(
            adapter_id=self.adapter_id,
            scope=self.config.scope,
            capability=capability,
            reason_code=reason_code,
            detail=detail,
        )

    def _bounded_record(
        self,
        *,
        capability: str,
        spec: _CapabilitySpec,
        row: dict[str, Any],
    ) -> BoundedProviderRecord:
        record_id = _text(row.get("id"))
        if not record_id:
            raise ValueError("query response record did not include an id")
        provider_record_id = f"{spec.model}:{record_id}"
        source_uri = (
            f"odoo://{self.config.scope.tenant_id}/{spec.model}/{record_id}"
        )
        if capability == "contacts":
            name = _text(row.get("name"))
            email = _text(row.get("email"))
            company = _m2o_label(row.get("parent_id"))
            metadata = {
                "model": spec.model,
                "record_id": record_id,
                "name": name,
                "email": email,
                "company": company,
            }
            snippet = "; ".join(item for item in (name, email, company) if item)
            source_event_at = ""
        elif capability == "leads":
            name = _text(row.get("name"))
            contact_name = _text(row.get("contact_name"))
            email = _text(row.get("email_from"))
            company = _m2o_label(row.get("partner_id")) or _text(
                row.get("partner_name")
            )
            metadata = {
                "model": spec.model,
                "record_id": record_id,
                "name": name,
                "contact_name": contact_name,
                "email": email,
                "company": company,
            }
            snippet = "; ".join(
                item for item in (contact_name, email, name, company) if item
            )
            source_event_at = _odoo_utc_timestamp(row.get("create_date"))
        else:
            subject = _strip_html(row.get("subject")) or "Odoo log note"
            author = _m2o_label(row.get("author_id"))
            related_model = _text(row.get("model"))
            related_record_id = _text(row.get("res_id"))
            metadata = {
                "model": spec.model,
                "record_id": record_id,
                "subject": subject,
                "related_model": related_model,
                "related_record_id": related_record_id,
                "author": author,
            }
            snippet = "; ".join(item for item in (subject, author) if item)
            source_event_at = _odoo_utc_timestamp(row.get("date"))
        return BoundedProviderRecord(
            provider_record_id=provider_record_id,
            source_record_id=provider_record_id,
            source_type=spec.source_type,
            capability=capability,
            snippet=snippet,
            structured_metadata=metadata,
            source_event_at=source_event_at,
            source_uri=source_uri,
        )
