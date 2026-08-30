from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol

from conversation_evidence_adapters import (
    ADAPTER_FAILURE_REASON_CODES,
    AdapterSourceScope,
    BoundedProviderRecord,
    EvidenceSnapshotNormalizer,
    adapter_failure,
)
from conversation_identity_retrieval import (
    ProviderRetrievalRequest,
    ProviderRetrievalResult,
)
from mail_evidence_normalization import (
    classify_account_direction,
    normalize_mail_address,
)
from mail_relationship_contracts import validate_mail_artifact


MAIL_RECEIPTS_CAPABILITIES = ("mail_metadata_read",)
REQUIRED_OPERATOR_LITE_TOOLS = frozenset({"search_mail"})
MAIL_RECEIPTS_SOURCE_TYPES = ("mail_receipts_message_metadata",)
MAIL_RECEIPTS_METADATA_FIELDS = (
    "evidence_id",
    "record_ref",
    "message_ref_hash",
    "thread_ref_hash",
    "source_key_hash",
    "from_addresses",
    "to_addresses",
    "cc_addresses",
    "account_direction",
    "contact_ids_by_address",
    "signature_observations",
    "namespace",
    "corpus_id",
    "query_address",
)


@dataclass(frozen=True)
class MailReceiptsPage:
    records: tuple[Mapping[str, Any], ...]
    next_cursor: str = ""
    as_of: str = ""
    source_scope: Mapping[str, str] = field(default_factory=dict)


class MailReceiptsReader(Protocol):
    def service_profile(self) -> Mapping[str, Any]: ...

    def search_exact_email(
        self,
        *,
        namespace: str,
        corpus_id: str,
        address: str,
        as_of: str,
        cursor: str,
        page_size: int,
        include_body: bool,
    ) -> MailReceiptsPage: ...


class MailReceiptsReadError(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        detail: str = "",
        *,
        retryable: bool = False,
    ) -> None:
        if reason_code not in ADAPTER_FAILURE_REASON_CODES:
            raise ValueError("Unsupported Mail Receipts failure reason code.")
        bounded_detail = str(detail or "").strip()[:500]
        super().__init__(bounded_detail)
        self.reason_code = reason_code
        self.detail = bounded_detail
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class MailReceiptsAdapterConfig:
    scope: AdapterSourceScope
    namespace: str
    corpus_id: str
    account_address: str


@dataclass(frozen=True)
class MailReceiptsRetrievalResult(ProviderRetrievalResult):
    query_receipt: dict[str, Any] = field(default_factory=dict)


class MailReceiptsEvidenceAdapter:
    adapter_id = "mail-receipts-evidence-v1"

    def __init__(
        self,
        *,
        config: MailReceiptsAdapterConfig,
        reader: MailReceiptsReader,
        retrieved_at: str,
    ) -> None:
        if config.scope.provider_kind != "mail_receipts":
            raise ValueError(
                "Mail Receipts adapter requires provider_kind='mail_receipts'."
            )
        if config.scope.capabilities != MAIL_RECEIPTS_CAPABILITIES:
            raise ValueError(
                "Mail Receipts adapter supports only mail_metadata_read."
            )
        if (
            not config.namespace.strip()
            or not config.corpus_id.strip()
            or not self._normalized_address(config.account_address)
            or config.account_address != config.account_address.strip().casefold()
        ):
            raise ValueError(
                "Mail Receipts adapter requires explicit namespace, corpus_id, "
                "and normalized account_address."
            )
        profile = reader.service_profile()
        capabilities = profile.get("capabilities")
        capability_set = (
            {str(value) for value in capabilities}
            if isinstance(capabilities, list)
            else set()
        )
        if (
            profile.get("profile") != "operator-lite"
            or not REQUIRED_OPERATOR_LITE_TOOLS <= capability_set
            or profile.get("mailbox_mutation") is not False
            or profile.get("corpus_operation_execution") is not False
        ):
            raise ValueError(
                "Mail Receipts adapter requires an operator-lite read-only profile."
            )
        self.config = config
        self.reader = reader
        self.retrieved_at = retrieved_at
        self.normalizer = EvidenceSnapshotNormalizer(
            scope=config.scope,
            allowed_source_types=MAIL_RECEIPTS_SOURCE_TYPES,
            allowed_metadata_fields=MAIL_RECEIPTS_METADATA_FIELDS,
        )

    def retrieve(
        self, request: ProviderRetrievalRequest
    ) -> MailReceiptsRetrievalResult:
        addresses = self._query_addresses(request.query_terms)
        if not self._scope_is_requested(request):
            return MailReceiptsRetrievalResult(
                warnings=("provider_scope_skipped",),
                query_receipt={},
            )
        if "mail_metadata_read" not in request.capabilities:
            return MailReceiptsRetrievalResult(query_receipt={})
        if not addresses:
            raise ValueError(
                "Mail Receipts retrieval requires normalized exact email terms."
            )
        if request.max_records < 1 or request.max_records > 250:
            raise ValueError("Mail Receipts max_records is outside its frozen bound.")
        if request.max_characters < 1 or request.max_characters > 1:
            raise ValueError(
                "Mail Receipts metadata retrieval does not permit message text."
            )

        snapshots = []
        failures: list[dict[str, str]] = []
        warnings: list[str] = []
        provider_calls = 0
        retries = 0
        truncated_count = 0
        for address in addresses:
            cursor = ""
            seen_cursors: set[str] = set()
            page_attempts = 0
            while True:
                remaining = request.max_records - len(snapshots)
                if remaining <= 0:
                    break
                if provider_calls >= 4 or page_attempts >= 10:
                    truncated_count += 1
                    if "provider_records_truncated" not in warnings:
                        warnings.append("provider_records_truncated")
                    break
                try:
                    provider_calls += 1
                    page_attempts += 1
                    page = self.reader.search_exact_email(
                        namespace=self.config.namespace,
                        corpus_id=self.config.corpus_id,
                        address=address,
                        as_of=request.as_of,
                        cursor=cursor,
                        page_size=min(100, remaining),
                        include_body=False,
                    )
                except MailReceiptsReadError as exc:
                    if exc.retryable and retries < 1 and provider_calls < 4:
                        retries += 1
                        continue
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.config.scope,
                            capability="mail_metadata_read",
                            reason_code=exc.reason_code,
                            detail=exc.detail,
                        )
                    )
                    break
                if not isinstance(page, MailReceiptsPage):
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.config.scope,
                            capability="mail_metadata_read",
                            reason_code="provider_response_invalid",
                            detail="Mail Receipts returned an invalid page.",
                        )
                    )
                    break
                if not self._page_scope_matches(page.source_scope):
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.config.scope,
                            capability="mail_metadata_read",
                            reason_code="provider_response_invalid",
                            detail=(
                                "Mail Receipts page scope did not match the "
                                "authorized namespace, corpus, account, and tenant."
                            ),
                        )
                    )
                    break
                if page.as_of and page.as_of != request.as_of:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.config.scope,
                            capability="mail_metadata_read",
                            reason_code="provider_response_invalid",
                            detail="Mail Receipts page as_of drifted from the request.",
                        )
                    )
                    break
                invalid_record = False
                for record_index, payload in enumerate(page.records):
                    try:
                        record = self._bounded_record(
                            payload,
                            query_address=address,
                        )
                        snapshots.append(
                            self.normalizer.normalize(
                                record,
                                as_of=request.as_of,
                                retrieved_at=self.retrieved_at,
                            )
                        )
                    except (TypeError, ValueError):
                        failures.append(
                            adapter_failure(
                                adapter_id=self.adapter_id,
                                scope=self.config.scope,
                                capability="mail_metadata_read",
                                reason_code="provider_response_invalid",
                                detail=(
                                    "Mail Receipts returned a record outside the "
                                    "closed metadata contract."
                                ),
                            )
                        )
                        invalid_record = True
                        break
                    if len(snapshots) >= request.max_records:
                        hidden_on_page = len(page.records) - record_index - 1
                        truncated_count += hidden_on_page
                        if hidden_on_page or page.next_cursor:
                            if "provider_records_truncated" not in warnings:
                                warnings.append("provider_records_truncated")
                        break
                if invalid_record:
                    break
                next_cursor = str(page.next_cursor or "")
                if not next_cursor:
                    break
                if next_cursor in seen_cursors:
                    failures.append(
                        adapter_failure(
                            adapter_id=self.adapter_id,
                            scope=self.config.scope,
                            capability="mail_metadata_read",
                            reason_code="provider_response_invalid",
                            detail="Mail Receipts repeated an opaque cursor.",
                        )
                    )
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            if provider_calls >= 4 or len(snapshots) >= request.max_records:
                break

        if failures and snapshots:
            warnings.append("provider_partial_result")
        receipt = self._query_receipt(
            request,
            addresses,
            snapshots,
            failures=failures,
            warnings=warnings,
            provider_calls=provider_calls,
            truncated_count=truncated_count,
        )
        validate_mail_artifact("mail_query_receipt", receipt)
        return MailReceiptsRetrievalResult(
            snapshots=tuple(snapshots),
            failures=tuple(failures),
            warnings=tuple(warnings),
            query_receipt=receipt,
        )

    def _bounded_record(
        self, payload: Mapping[str, Any], *, query_address: str
    ) -> BoundedProviderRecord:
        if not isinstance(payload, Mapping):
            raise ValueError("Mail Receipts evidence record must be an object.")
        required = {
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
        if set(payload) != required:
            raise ValueError("Mail Receipts evidence record shape is invalid.")
        senders = self._addresses(payload["from"], field_name="from")
        recipients = self._addresses(payload["to"], field_name="to")
        copied = self._addresses(payload["cc"], field_name="cc")
        if len(senders) != 1 or not recipients:
            raise ValueError(
                "Mail Receipts evidence requires one sender and a recipient."
            )
        message_ref = str(payload["logical_message_ref"] or "").strip()
        thread_ref = str(payload["thread_ref"] or "").strip()
        source_key = str(payload["source_key"] or "").strip()
        evidence_id = str(payload["evidence_id"] or "").strip()
        record_ref = str(payload["record_ref"] or "").strip()
        if not all((message_ref, thread_ref, source_key, evidence_id, record_ref)):
            raise ValueError("Mail Receipts opaque evidence references are required.")
        contacts = payload["contact_ids_by_address"]
        participant_addresses = set(senders + recipients + copied)
        if not isinstance(contacts, Mapping) or any(
            address not in participant_addresses or not str(contact_id or "").strip()
            for address, contact_id in contacts.items()
        ):
            raise ValueError(
                "Mail Receipts contact joins require exact participant addresses."
            )
        signature = payload["signature"]
        signature_observations: list[dict[str, str]] = []
        if signature is not None:
            if not isinstance(signature, Mapping) or set(signature) != {
                "address",
                "title",
                "organization",
                "department",
            }:
                raise ValueError("Mail Receipts structured signature is invalid.")
            address = str(signature["address"] or "").strip().casefold()
            if address not in participant_addresses:
                raise ValueError(
                    "Mail Receipts structured signature is not an exact participant."
                )
            signature_observations.append(
                {
                    "address": address,
                    "title": str(signature["title"] or "").strip(),
                    "organization": str(signature["organization"] or "").strip(),
                    "department": str(signature["department"] or "").strip(),
                    "observed_at": str(payload["sent_at"] or "").strip(),
                }
            )
        direction = classify_account_direction(
            from_addresses=senders,
            to_addresses=recipients,
            cc_addresses=copied,
            account_address=self.config.account_address,
        )
        message_ref_hash = self._hash(message_ref)
        thread_ref_hash = self._hash(thread_ref)
        return BoundedProviderRecord(
            provider_record_id=evidence_id,
            source_type="mail_receipts_message_metadata",
            capability="mail_metadata_read",
            snippet="",
            structured_metadata={
                "evidence_id": evidence_id,
                "record_ref": record_ref,
                "message_ref_hash": message_ref_hash,
                "thread_ref_hash": thread_ref_hash,
                "source_key_hash": self._hash(source_key),
                "from_addresses": senders,
                "to_addresses": recipients,
                "cc_addresses": copied,
                "account_direction": direction,
                "contact_ids_by_address": dict(contacts),
                "signature_observations": signature_observations,
                "namespace": self.config.namespace,
                "corpus_id": self.config.corpus_id,
                "query_address": query_address,
            },
            source_event_at=str(payload["sent_at"] or "").strip(),
            source_record_id=record_ref,
            independence_group_id=self._hash("interaction:" + message_ref),
            redaction={
                "body_retained": False,
                "subject_retained": False,
                "provider_ids_hashed": True,
            },
            truncation={"snippet_characters": 0},
        )

    def _query_receipt(
        self,
        request: ProviderRetrievalRequest,
        addresses: tuple[str, ...],
        snapshots: list[Any],
        *,
        failures: list[dict[str, str]],
        warnings: list[str],
        provider_calls: int,
        truncated_count: int,
    ) -> dict[str, Any]:
        request_core = {
            "conversation_id": request.conversation_id,
            "source_profile_id": self.config.scope.source_profile_id,
            "namespace": self.config.namespace,
            "corpus_id": self.config.corpus_id,
            "addresses": list(addresses),
            "as_of": request.as_of,
            "max_records": request.max_records,
        }
        request_hash = self._hash_json(request_core)
        as_of = datetime.fromisoformat(request.as_of.replace("Z", "+00:00"))
        lookback_start = (as_of - timedelta(days=365)).astimezone(
            timezone.utc
        ).isoformat().replace("+00:00", "Z")
        status = "complete"
        if failures:
            status = "partial" if snapshots else "unavailable"
        receipt_failures = (
            [
                {
                    "reason_code": "partial_source_failure",
                    "detail": failure["reason_code"],
                }
                for failure in failures
            ]
            if failures
            else []
        )
        return {
            "schema_version": "transcribe-audio.mail-query-receipt.v1",
            "receipt_id": "mail-query-" + request_hash[:32],
            "request_hash": request_hash,
            "source_scope": {
                "provider_kind": "mail_receipts",
                "profile_id": self.config.scope.source_profile_id,
                "account_id": self.config.scope.account_id,
                "tenant_id": self.config.scope.tenant_id,
                "namespace": self.config.namespace,
                "corpus_id": self.config.corpus_id,
                "capabilities": ["mail_metadata_read"],
            },
            "capability": "mail_metadata_read",
            "query_mode": "exact_email_only",
            "exact_addresses": list(addresses),
            "as_of": request.as_of,
            "lookback_start": lookback_start,
            "budgets": {
                "max_records": request.max_records,
                "max_characters": request.max_characters,
                "max_calls": 4,
                "max_latency_ms": 30_000,
                "max_pages": 10,
            },
            "status": status,
            "counts": {
                "selected": len(snapshots),
                "excluded": 0,
                "truncated": truncated_count,
                "provider_writes": 0,
            },
            "warnings": list(warnings),
            "failures": receipt_failures,
            "result_hashes": [snapshot.content_hash for snapshot in snapshots],
            "created_at": self.retrieved_at,
        }

    def _scope_is_requested(self, request: ProviderRetrievalRequest) -> bool:
        return any(
            scope.source_profile_id == self.config.scope.source_profile_id
            and scope.account_id == self.config.scope.account_id
            and scope.tenant_id == self.config.scope.tenant_id
            for scope in request.scopes
        )

    def _page_scope_matches(self, value: Mapping[str, str]) -> bool:
        expected = {
            "source_profile_id": self.config.scope.source_profile_id,
            "account_id": self.config.scope.account_id,
            "tenant_id": self.config.scope.tenant_id,
            "namespace": self.config.namespace,
            "corpus_id": self.config.corpus_id,
        }
        return isinstance(value, Mapping) and dict(value) == expected

    @classmethod
    def _query_addresses(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        addresses = tuple(dict.fromkeys(str(value).strip().casefold() for value in values))
        if any(not cls._normalized_address(value) for value in addresses):
            return ()
        return addresses

    @classmethod
    def _addresses(cls, value: object, *, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"Mail Receipts {field_name} must be a list.")
        addresses = [str(item).strip().casefold() for item in value]
        if (
            any(not cls._normalized_address(item) for item in addresses)
            or len(addresses) != len(set(addresses))
        ):
            raise ValueError(
                f"Mail Receipts {field_name} addresses must be normalized and unique."
            )
        return addresses

    @staticmethod
    def _normalized_address(value: str) -> bool:
        try:
            return normalize_mail_address(value) == value
        except ValueError:
            return False

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_json(value: object) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
