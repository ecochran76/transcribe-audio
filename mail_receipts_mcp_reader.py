from __future__ import annotations

import hashlib
import json
import selectors
import subprocess
from datetime import datetime, timezone
from email.utils import getaddresses
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from conversation_evidence_mail_receipts import (
    MailReceiptsPage,
    MailReceiptsReadError,
)
from mail_evidence_normalization import normalize_mail_address


class McpToolClient(Protocol):
    def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_ms: int
    ) -> Mapping[str, Any]: ...


class JsonLineMcpClient:
    """Small persistent JSON-lines MCP client for the installed stdio shim."""

    def __init__(self, command: Sequence[str]) -> None:
        self.command = tuple(str(value) for value in command)
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1

    def __enter__(self) -> "JsonLineMcpClient":
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "transcribe-audio-plan0073",
                    "version": "1",
                },
            },
            timeout_ms=30_000,
        )
        self._notify("notifications/initialized", {})
        return self

    def __exit__(self, *_args: object) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, timeout_ms: int
    ) -> Mapping[str, Any]:
        response = self._request(
            "tools/call",
            {"name": name, "arguments": dict(arguments)},
            timeout_ms=timeout_ms,
        )
        result = response.get("result")
        if not isinstance(result, Mapping) or result.get("isError") is True:
            raise MailReceiptsReadError(
                "provider_query_failed",
                "Mail Receipts MCP tool returned an error.",
            )
        structured = result.get("structuredContent")
        if isinstance(structured, Mapping):
            return structured
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, Mapping) or item.get("type") != "text":
                    continue
                try:
                    parsed = json.loads(str(item.get("text") or ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping):
                    return parsed
        raise MailReceiptsReadError(
            "provider_response_invalid",
            "Mail Receipts MCP response did not contain structured content.",
        )

    def _request(
        self, method: str, params: Mapping[str, Any], *, timeout_ms: int
    ) -> Mapping[str, Any]:
        process = self._running_process()
        request_id = self._next_id
        self._next_id += 1
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params),
            }
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                events = selector.select(timeout=max(0.001, timeout_ms / 1000))
                if not events:
                    raise MailReceiptsReadError(
                        "provider_unavailable",
                        "Mail Receipts MCP response timed out.",
                        retryable=True,
                    )
                line = process.stdout.readline()
                if not line:
                    raise MailReceiptsReadError(
                        "provider_unavailable",
                        "Mail Receipts MCP transport closed.",
                        retryable=True,
                    )
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(response, Mapping) or response.get("id") != request_id:
                    continue
                if response.get("error") is not None:
                    raise MailReceiptsReadError(
                        "provider_query_failed",
                        "Mail Receipts MCP request failed.",
                    )
                return response
        finally:
            selector.close()

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _write(self, payload: Mapping[str, Any]) -> None:
        process = self._running_process()
        assert process.stdin is not None
        process.stdin.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        process.stdin.flush()

    def _running_process(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None or process.poll() is not None:
            raise MailReceiptsReadError(
                "provider_unavailable",
                "Mail Receipts MCP transport is not running.",
                retryable=True,
            )
        return process


class MailReceiptsMcpReader:
    """Adapt public operator-lite search/context tools to Plan 0073 metadata."""

    def __init__(
        self,
        client: McpToolClient,
        *,
        source_profile_id: str,
        account_id: str,
        tenant_id: str,
        namespace: str,
        corpus_id: str,
    ) -> None:
        self.client = client
        self.source_scope = {
            "source_profile_id": source_profile_id,
            "account_id": account_id,
            "tenant_id": tenant_id,
            "namespace": namespace,
            "corpus_id": corpus_id,
        }

    def service_profile(self) -> Mapping[str, Any]:
        return {
            "profile": "operator-lite",
            "capabilities": ["search_mail", "selected_result_context_pack"],
            "mailbox_mutation": False,
            "corpus_operation_execution": False,
        }

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
        timeout_ms: int,
    ) -> MailReceiptsPage:
        if (
            namespace != self.source_scope["namespace"]
            or corpus_id != self.source_scope["corpus_id"]
            or cursor not in {"", "mail-receipts-mcp-reader:truncated"}
            or include_body
            or page_size < 1
            or page_size > 100
        ):
            raise MailReceiptsReadError(
                "provider_response_invalid",
                "Plan 0073 Mail Receipts reader scope or bounds drifted.",
            )
        normalized_address = normalize_mail_address(address)
        normalized_as_of = self._timestamp(as_of)
        if cursor:
            return MailReceiptsPage(
                records=(),
                as_of=normalized_as_of,
                source_scope=self.source_scope,
            )

        targets: list[dict[str, Any]] = []
        target_keys: set[str] = set()
        search_truncated = False
        for field_name in ("from", "to", "cc"):
            query = (
                f'{field_name}:{json.dumps(normalized_address)} '
                "rank:lexical group_by:message"
            )
            response = self.client.call_tool(
                "search_mail",
                {
                    "corpus_id": corpus_id,
                    "namespace": namespace,
                    "intent": query,
                    "page_size": page_size,
                    "retrieval_mode": "lexical",
                    "result_mode": "occurrence",
                    "rerank": False,
                    "include_summary": False,
                    "include_annotations": False,
                    "include_direct_search": False,
                    "include_logical_message_occurrence_preview": False,
                    "explain": False,
                    "group_by": "message",
                    "persist_workflow_snapshot": False,
                },
                timeout_ms=timeout_ms,
            )
            if response.get("corpus_id") != corpus_id or response.get("namespace") != namespace:
                raise MailReceiptsReadError(
                    "provider_response_invalid",
                    "Mail Receipts search response scope drifted.",
                )
            self._validate_merge_coverage(response)
            page = response.get("page")
            if isinstance(page, Mapping) and page.get("has_more") is True:
                search_truncated = True
            for hit in response.get("hits") or []:
                target = self._target(hit, corpus_id=corpus_id, namespace=namespace)
                if target is None:
                    continue
                key = self._hash_json(target)
                if key not in target_keys:
                    target_keys.add(key)
                    targets.append(target)

        if not targets:
            return MailReceiptsPage(
                records=(),
                as_of=normalized_as_of,
                source_scope=self.source_scope,
            )
        context = self.client.call_tool(
            "selected_result_context_pack",
            {
                "corpus_id": corpus_id,
                "namespace": namespace,
                "targets": targets,
                "before": 0,
                "after": 0,
                "include_body": False,
                "body_max_chars": 1,
            },
            timeout_ms=timeout_ms,
        )
        if context.get("corpus_id") != corpus_id or context.get("namespace") != namespace:
            raise MailReceiptsReadError(
                "provider_response_invalid",
                "Mail Receipts context response scope drifted.",
            )
        records_by_id: dict[str, dict[str, Any]] = {}
        for item in context.get("items") or []:
            if not isinstance(item, Mapping) or item.get("resolved") is not True:
                continue
            for message in item.get("context") or []:
                record = self._record(
                    message,
                    address=normalized_address,
                    as_of=normalized_as_of,
                    corpus_id=corpus_id,
                    namespace=namespace,
                )
                if record is not None:
                    records_by_id[record["evidence_id"]] = record
        records = sorted(
            records_by_id.values(),
            key=lambda value: (value["sent_at"], value["evidence_id"]),
            reverse=True,
        )
        truncated = search_truncated or len(records) > page_size
        return MailReceiptsPage(
            records=tuple(records[:page_size]),
            next_cursor=("mail-receipts-mcp-reader:truncated" if truncated else ""),
            as_of=normalized_as_of,
            source_scope=self.source_scope,
        )

    @staticmethod
    def _validate_merge_coverage(response: Mapping[str, Any]) -> None:
        merge_target = response.get("merge_target")
        if not isinstance(merge_target, Mapping):
            return
        target_corpus_ids = merge_target.get("target_corpus_ids")
        if not isinstance(target_corpus_ids, list) or len(target_corpus_ids) <= 1:
            return
        validation = response.get("retrieval_index_validation")
        effect = (
            str(validation.get("workflow_action_effect") or "")
            if isinstance(validation, Mapping)
            else ""
        )
        if effect == "duckdb-message-search-direct-participant-address":
            raise MailReceiptsReadError(
                "provider_response_invalid",
                "Mail Receipts exact-address search omitted part of archive-plus-live scope.",
            )

    @staticmethod
    def _target(
        hit: object, *, corpus_id: str, namespace: str
    ) -> dict[str, Any] | None:
        if not isinstance(hit, Mapping):
            return None
        kind = str(hit.get("kind") or "chunk")
        if kind not in {"message", "attachment", "thread", "chunk", "logical_message"}:
            return None
        follow_up = hit.get("follow_up")
        if not isinstance(follow_up, Mapping):
            return None
        record_ref = str(follow_up.get("record_ref") or hit.get("record_ref") or "")
        target: dict[str, Any] = {
            "target_kind": kind,
            "hit_kind": kind,
            "namespace": namespace,
            "corpus_id": corpus_id,
            "native_ids": {},
        }
        if record_ref:
            target["message_id"] = record_ref
        if follow_up.get("thread_id"):
            target["thread_id"] = str(follow_up["thread_id"])
        if follow_up.get("logical_message_id"):
            target["logical_message_id"] = str(follow_up["logical_message_id"])
        if kind == "chunk":
            target["chunk_id"] = str(hit.get("id") or "")
        if not any(target.get(key) for key in ("message_id", "thread_id", "chunk_id", "logical_message_id")):
            return None
        return target

    @classmethod
    def _record(
        cls,
        message: object,
        *,
        address: str,
        as_of: str,
        corpus_id: str,
        namespace: str,
    ) -> dict[str, Any] | None:
        if not isinstance(message, Mapping):
            return None
        if message.get("corpus_id") != corpus_id or message.get("namespace") != namespace:
            return None
        senders = cls._addresses([message.get("sender")])
        recipients = cls._addresses(message.get("to"))
        copied = cls._addresses(message.get("cc"))
        if len(senders) != 1 or not recipients:
            return None
        if address not in set(senders + recipients + copied):
            return None
        sent_at = cls._timestamp(message.get("sent_at") or message.get("received_at"))
        if sent_at > as_of:
            return None
        message_id = str(message.get("message_id") or "").strip()
        if not message_id:
            return None
        logical = str(message.get("logical_message_id") or message_id)
        thread = str(message.get("thread_id") or logical)
        source_refs = message.get("source_refs")
        source_identity = {
            "namespace": namespace,
            "corpus_id": corpus_id,
            "message_id": message_id,
            "source_refs": source_refs if isinstance(source_refs, Mapping) else {},
        }
        digest = cls._hash_json(source_identity)
        return {
            "evidence_id": "mail-evidence-" + digest,
            "record_ref": "mail-record-" + digest,
            "logical_message_ref": "logical-" + hashlib.sha256(logical.encode()).hexdigest(),
            "thread_ref": "thread-" + hashlib.sha256(thread.encode()).hexdigest(),
            "source_key": "source-" + digest,
            "sent_at": sent_at,
            "from": senders,
            "to": recipients,
            "cc": copied,
            "contact_ids_by_address": {},
            "signature": None,
        }

    @staticmethod
    def _addresses(value: object) -> list[str]:
        raw = value if isinstance(value, list) else []
        parsed = []
        for _name, address in getaddresses([str(item or "") for item in raw]):
            try:
                parsed.append(normalize_mail_address(address))
            except ValueError:
                continue
        return sorted(set(parsed))

    @staticmethod
    def _timestamp(value: object) -> str:
        try:
            parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        except ValueError as exc:
            raise MailReceiptsReadError(
                "provider_response_invalid",
                "Mail Receipts metadata timestamp is invalid.",
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise MailReceiptsReadError(
                "provider_response_invalid",
                "Mail Receipts metadata timestamp lacks a timezone.",
            )
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _hash_json(value: object) -> str:
        payload = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def installed_operator_lite_command() -> tuple[str, ...]:
    executable = Path("/home/ecochran76/.local/bin/mail-receipts")
    socket = Path(
        "/home/ecochran76/.local/share/mail-receipts/storage/.runtime/"
        "mail-receipts-mcp-backend.sock"
    )
    return (
        str(executable),
        "mcp-server",
        "--profile",
        "operator-lite",
        "--backend-socket",
        str(socket),
        "--namespace",
        "default",
        "--allow-namespace",
        "default",
        "--auth-subject",
        "codex-user",
        "--auth-mechanism",
        "user-scoped-local",
        "--auth-role",
        "operator-lite",
        "--require-auth",
        "--idle-timeout-seconds",
        "60",
    )
