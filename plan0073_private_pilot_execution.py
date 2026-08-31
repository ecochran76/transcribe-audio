from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)
from conversation_evidence_adapters import AdapterSourceScope
from conversation_evidence_mail_receipts import (
    MailReceiptsAdapterConfig,
    MailReceiptsEvidenceAdapter,
    MailReceiptsPage,
    MailReceiptsReader,
    MailReceiptsReadError,
)
from conversation_identity_retrieval import ProviderRetrievalRequest
from conversation_knowledge_evidence import EvidenceScope
from mail_evidence_normalization import (
    NormalizedMailEvidence,
    normalize_mail_address,
    normalize_mail_evidence,
)
from mail_relationship_contracts import ZERO_EFFECTS, validate_mail_artifact
from mail_relationship_discovery import discover_mail_relationship_hypotheses
from plan0073_private_pilot import (
    Plan0073PrivatePilotError,
    validate_private_pilot_approval,
)


class Plan0073PrivatePilotExecutionError(RuntimeError):
    """Raised when the exact approved P5 pilot cannot execute safely."""


EXECUTION_RECEIPT_SCHEMA = "transcribe-audio.plan0073-p5-execution-receipt.v1"
OFFLINE_REPLAY_SCHEMA = "transcribe-audio.plan0073-p5-offline-replay.v1"
_PREVIEW_ID_RE = re.compile(r"plan0073-p5-[a-f0-9]{32}")


def _hash(value: object) -> str:
    body = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _timestamp(value: object, label: str) -> tuple[str, datetime]:
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Plan0073PrivatePilotExecutionError(
            f"{label} must be an ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Plan0073PrivatePilotExecutionError(
            f"{label} must include a timezone."
        )
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z"), normalized


def _run_path(preview: Mapping[str, Any], runtime_root: Path) -> tuple[Path, Path]:
    root = runtime_root.expanduser().absolute()
    relative = Path(
        str((preview.get("runtime_write_surface") or {}).get("relative_root") or "")
    )
    expected = Path("plan-0073/private-pilots") / str(preview.get("preview_id") or "")
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise Plan0073PrivatePilotExecutionError(
            "P5 runtime write surface does not match the approved private root."
        )
    return root, root / relative


def build_private_pilot_contacts(
    people_payload: Mapping[str, Any],
    *,
    account_address: str,
) -> dict[str, dict[str, str]]:
    """Build the exact local Contacts projection used by P5 discovery."""

    if not isinstance(people_payload, Mapping) or not isinstance(
        people_payload.get("items"), list
    ):
        raise Plan0073PrivatePilotExecutionError(
            "P5 Contacts payload is invalid."
        )
    contacts: dict[str, dict[str, str]] = {}
    emails: set[str] = set()
    for raw in people_payload["items"]:
        if not isinstance(raw, Mapping) or raw.get("identity_kind") != "local_contact":
            continue
        methods = raw.get("contact_methods") or []
        exact_emails = sorted(
            {
                normalize_mail_address(method.get("value"))
                for method in methods
                if isinstance(method, Mapping) and method.get("kind") == "email"
            }
        )
        if not exact_emails:
            continue
        if len(exact_emails) != 1:
            raise Plan0073PrivatePilotExecutionError(
                "P5 requires one exact email per local contact projection."
            )
        contact_id = str(raw.get("person_id") or "").strip()
        if not contact_id or contact_id in contacts or exact_emails[0] in emails:
            raise Plan0073PrivatePilotExecutionError(
                "P5 local Contacts require unique person IDs and emails."
            )
        email = exact_emails[0]
        contacts[contact_id] = {
            "contact_id": contact_id,
            "label": str(raw.get("primary_name") or "Unnamed contact").strip(),
            "email": email,
            "contact_class": str(
                raw.get("contact_class") or "person_candidate"
            ).strip(),
        }
        emails.add(email)
    account = normalize_mail_address(account_address)
    if account not in emails:
        contact_id = "mail-account-contact-" + _hash(account)[:24]
        if contact_id in contacts:
            raise Plan0073PrivatePilotExecutionError(
                "P5 mail account contact ID collides with a local contact."
            )
        contacts[contact_id] = {
            "contact_id": contact_id,
            "label": "Mail account",
            "email": account,
            "contact_class": "person_candidate",
        }
    return {key: contacts[key] for key in sorted(contacts)}


def _merge_normalized(
    values: Sequence[NormalizedMailEvidence],
) -> NormalizedMailEvidence:
    observations_by_source: dict[str, dict[str, Any]] = {}
    group_inputs: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        for raw in value.observations:
            observation = validate_mail_artifact("mail_observation", raw)
            source_identity = dict(observation)
            source_identity.pop("observation_id", None)
            source_identity.pop("query_receipt_id", None)
            key = _hash(source_identity)
            existing = observations_by_source.get(key)
            if (
                existing is None
                or observation["observation_id"] < existing["observation_id"]
            ):
                observations_by_source[key] = observation
        for raw in value.independence_groups:
            group = validate_mail_artifact("mail_independence_group", raw)
            group_inputs.setdefault(str(group["group_id"]), []).append(group)

    observations = tuple(
        sorted(
            observations_by_source.values(),
            key=lambda item: (item["source_event_at"], item["observation_id"]),
        )
    )
    members_by_group: dict[str, list[str]] = {}
    for observation in observations:
        members_by_group.setdefault(
            str(observation["independence_group_id"]), []
        ).append(str(observation["observation_id"]))
    groups: list[dict[str, Any]] = []
    for group_id in sorted(members_by_group):
        sources = group_inputs.get(group_id, [])
        if not sources:
            raise Plan0073PrivatePilotExecutionError(
                "A merged mail observation has no independence receipt."
            )
        thread_keys = {str(item["independent_thread_key"]) for item in sources}
        versions = {str(item["interaction_key_version"]) for item in sources}
        if len(thread_keys) != 1 or versions != {"mail-interaction-key.v1"}:
            raise Plan0073PrivatePilotExecutionError(
                "Mail independence receipts conflict across exact queries."
            )
        members = sorted(set(members_by_group[group_id]))
        duplicate_count = len(members) - 1
        group_core = {
            "group_id": group_id,
            "interaction_key_version": "mail-interaction-key.v1",
            "independent_thread_key": next(iter(thread_keys)),
            "member_observation_ids": members,
            "duplicate_count": duplicate_count,
            "source_count": max(int(item["source_count"]) for item in sources),
            "reason_code": "duplicate_interaction" if duplicate_count else None,
        }
        group = {
            "schema_version": "transcribe-audio.mail-independence-group.v1",
            **group_core,
            "content_hash": _hash(group_core),
        }
        validate_mail_artifact("mail_independence_group", group)
        groups.append(group)
    independence_groups = tuple(groups)
    return NormalizedMailEvidence(
        observations=observations,
        independence_groups=independence_groups,
        input_watermark=_hash(
            {
                "observations": observations,
                "independence_groups": independence_groups,
            }
        ),
    )


class _PilotRetryBudgetReader:
    def __init__(self, reader: MailReceiptsReader, *, max_retries: int) -> None:
        self.reader = reader
        self.max_retries = max_retries
        self.retryable_failures = 0

    def service_profile(self) -> Mapping[str, Any]:
        return self.reader.service_profile()

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        try:
            return self.reader.search_exact_email(**kwargs)
        except MailReceiptsReadError as exc:
            if not exc.retryable:
                raise
            if self.retryable_failures >= self.max_retries:
                raise MailReceiptsReadError(
                    exc.reason_code,
                    exc.detail,
                    retryable=False,
                ) from exc
            self.retryable_failures += 1
            raise


class _ContactJoiningReader:
    def __init__(
        self,
        reader: MailReceiptsReader,
        contacts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.reader = reader
        self.contact_ids_by_email = {
            normalize_mail_address(contact.get("email")): str(
                contact.get("contact_id") or contact_id
            )
            for contact_id, contact in contacts.items()
        }

    def service_profile(self) -> Mapping[str, Any]:
        return self.reader.service_profile()

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        page = self.reader.search_exact_email(**kwargs)
        if not isinstance(page, MailReceiptsPage):
            return page
        records: list[Mapping[str, Any]] = []
        for raw in page.records:
            if not isinstance(raw, Mapping):
                records.append(raw)
                continue
            joined = dict(raw)
            addresses: list[str] = []
            try:
                for field_name in ("from", "to", "cc"):
                    value = raw.get(field_name)
                    if not isinstance(value, list):
                        raise ValueError
                    addresses.extend(normalize_mail_address(item) for item in value)
            except ValueError:
                addresses = []
            joined["contact_ids_by_address"] = {
                address: self.contact_ids_by_email[address]
                for address in sorted(set(addresses))
                if address in self.contact_ids_by_email
            }
            records.append(joined)
        return MailReceiptsPage(
            records=tuple(records),
            next_cursor=page.next_cursor,
            as_of=page.as_of,
            source_scope=page.source_scope,
        )


def _with_account_contact(
    contacts: Mapping[str, Mapping[str, Any]],
    account_address: str,
) -> dict[str, dict[str, Any]]:
    effective = {str(key): dict(value) for key, value in contacts.items()}
    account = normalize_mail_address(account_address)
    known = {
        normalize_mail_address(value.get("email")) for value in effective.values()
    }
    if account not in known:
        contact_id = "mail-account-contact-" + _hash(account)[:24]
        if contact_id in effective:
            raise Plan0073PrivatePilotExecutionError(
                "P5 mail account contact ID collides with a local contact."
            )
        effective[contact_id] = {
            "contact_id": contact_id,
            "label": "Mail account",
            "email": account,
            "contact_class": "person_candidate",
        }
    return {key: effective[key] for key in sorted(effective)}


def execute_private_pilot(
    preview: Mapping[str, Any],
    approval: Mapping[str, Any],
    *,
    reader: MailReceiptsReader,
    contacts: Mapping[str, Mapping[str, Any]],
    runtime_root: Path,
    executed_at: str,
) -> dict[str, Any]:
    """Execute one approval-bound P5 pilot through an injected read seam."""

    try:
        normalized_approval = validate_private_pilot_approval(preview, approval)
    except Plan0073PrivatePilotError as exc:
        raise Plan0073PrivatePilotExecutionError(
            "P5 approval is absent, invalid, or no longer binds the preview."
        ) from exc

    query_plan = preview.get("query_plan")
    if not isinstance(query_plan, list) or not query_plan:
        raise Plan0073PrivatePilotExecutionError("P5 query plan is missing.")
    source = (preview.get("request") or {}).get("source_scope") or {}
    budgets = (preview.get("request") or {}).get("budgets") or {}
    if not isinstance(source, Mapping) or not isinstance(budgets, Mapping):
        raise Plan0073PrivatePilotExecutionError("P5 preview request is incomplete.")
    root, run = _run_path(preview, runtime_root)
    aggregate_path = run / "aggregate-validation.json"
    if aggregate_path.exists():
        replay_private_pilot(str(preview["preview_id"]), runtime_root=root)
        require_private_file(aggregate_path, root)
        existing = read_private_object(aggregate_path)
        return {
            **existing,
            "aggregate_path": str(aggregate_path),
            "idempotent": True,
        }
    if run.exists():
        ensure_private_tree(root, run)
        preview_path = run / "preview.json"
        children = sorted(run.iterdir(), key=lambda path: path.name)
        if children != [preview_path]:
            raise Plan0073PrivatePilotExecutionError(
                "P5 has an incomplete private run and will not repeat its corpus read."
            )
        require_private_file(preview_path, root)
        if read_private_object(preview_path) != dict(preview):
            raise Plan0073PrivatePilotExecutionError(
                "P5 persisted preview no longer matches the approved preview."
            )
    normalized_executed_at, execution_time = _timestamp(executed_at, "executed_at")
    _, approval_time = _timestamp(normalized_approval["approved_at"], "approved_at")
    if execution_time < approval_time:
        raise Plan0073PrivatePilotExecutionError(
            "P5 execution cannot predate its exact approval."
        )
    try:
        effective_contacts = _with_account_contact(
            contacts,
            str(source.get("account_address") or ""),
        )
        discover_mail_relationship_hypotheses(
            [],
            [],
            contacts=effective_contacts,
            account_address=str(source.get("account_address") or ""),
            input_watermark="plan0073-p5-contact-preflight",
        )
    except ValueError as exc:
        raise Plan0073PrivatePilotExecutionError(
            "P5 contact map failed deterministic preflight."
        ) from exc

    query_receipts_dir = run / "query-receipts"
    normalized_dir = run / "normalized"
    hypotheses_dir = run / "hypotheses"

    scope = AdapterSourceScope(
        source_profile_id=str(source["source_profile_id"]),
        provider_kind=str(source["provider_kind"]),
        account_id=str(source["account_id"]),
        tenant_id=str(source["tenant_id"]),
        capabilities=tuple(str(value) for value in source["capabilities"]),
    )
    joined_reader = _ContactJoiningReader(reader, effective_contacts)
    bounded_reader = _PilotRetryBudgetReader(
        joined_reader,
        max_retries=int(budgets["max_retries_per_pilot"]),
    )
    try:
        adapter = MailReceiptsEvidenceAdapter(
            config=MailReceiptsAdapterConfig(
                scope=scope,
                namespace=str(source["namespace"]),
                corpus_id=str(source["corpus_id"]),
                account_address=str(source["account_address"]),
                max_latency_ms=int(budgets["max_latency_ms_per_query"]),
            ),
            reader=bounded_reader,
            retrieved_at=normalized_executed_at,
        )
    except ValueError as exc:
        raise Plan0073PrivatePilotExecutionError(str(exc)) from exc
    for directory in (run, query_receipts_dir, normalized_dir, hypotheses_dir):
        ensure_private_tree(root, directory)
    contacts_path = normalized_dir / "contacts.json"
    write_immutable_private_json(run / "preview.json", dict(preview))
    write_immutable_private_json(run / "approval.json", normalized_approval)
    write_immutable_private_json(
        contacts_path,
        {"contacts": effective_contacts},
    )
    by_conversation: dict[
        str, list[tuple[Mapping[str, Any], Any, NormalizedMailEvidence]]
    ] = {}
    query_receipt_artifacts: list[dict[str, str]] = []
    selected_records = 0
    unavailable_queries = 0
    query_statuses: list[str] = []
    for query in query_plan:
        retrieved = adapter.retrieve(
            ProviderRetrievalRequest(
                conversation_id=str(query["conversation_id"]),
                query_terms=(str(query["exact_address"]),),
                scopes=(
                    EvidenceScope(
                        source_profile_id=scope.source_profile_id,
                        account_id=scope.account_id,
                        tenant_id=scope.tenant_id,
                    ),
                ),
                capabilities=scope.capabilities,
                as_of=str(query["as_of"]),
                max_records=int(budgets["max_records_per_query"]),
                max_characters=int(budgets["max_characters_per_query"]),
            )
        )
        normalized = normalize_mail_evidence(
            retrieved.snapshots,
            query_receipt=retrieved.query_receipt,
        )
        conversation_id = str(query["conversation_id"])
        by_conversation.setdefault(conversation_id, []).append(
            (query, retrieved, normalized)
        )
        query_receipt_path = query_receipts_dir / f"{query['query_id']}.json"
        write_immutable_private_json(query_receipt_path, retrieved.query_receipt)
        query_receipt_artifacts.append(
            {
                "name": query_receipt_path.name,
                "content_sha256": _hash(retrieved.query_receipt),
            }
        )
        selected_records += len(retrieved.snapshots)
        query_status = str(retrieved.query_receipt.get("status") or "unavailable")
        query_statuses.append(query_status)
        unavailable_queries += int(query_status == "unavailable")

    normalized_artifacts: list[dict[str, str]] = []
    hypotheses_artifacts: list[dict[str, str]] = []
    observation_count = 0
    independence_group_count = 0
    hypothesis_count = 0
    for conversation_id in sorted(by_conversation):
        parts = by_conversation[conversation_id]
        merged = _merge_normalized([part[2] for part in parts])
        discovery = discover_mail_relationship_hypotheses(
            merged.observations,
            merged.independence_groups,
            contacts=effective_contacts,
            account_address=str(source["account_address"]),
            input_watermark=merged.input_watermark,
        )
        conversation_key = _hash(conversation_id)[:24]
        normalized_path = normalized_dir / f"{conversation_key}.json"
        hypotheses_path = hypotheses_dir / f"{conversation_key}.json"
        normalized_artifact = {
            "schema_version": "transcribe-audio.plan0073-p5-normalized-input.v1",
            "conversation_id": conversation_id,
            "observations": list(merged.observations),
            "independence_groups": list(merged.independence_groups),
            "input_watermark": merged.input_watermark,
        }
        hypotheses_artifact = {
            "schema_version": "transcribe-audio.plan0073-p5-shadow-hypotheses.v1",
            "conversation_id": conversation_id,
            "input_watermark": discovery.input_watermark,
            "hypotheses": list(discovery.hypotheses),
            "excluded_reason_counts": discovery.excluded_reason_counts,
            "effects": dict(ZERO_EFFECTS),
        }
        write_immutable_private_json(normalized_path, normalized_artifact)
        write_immutable_private_json(hypotheses_path, hypotheses_artifact)
        normalized_artifacts.append(
            {"name": normalized_path.name, "content_sha256": _hash(normalized_artifact)}
        )
        hypotheses_artifacts.append(
            {"name": hypotheses_path.name, "content_sha256": _hash(hypotheses_artifact)}
        )
        observation_count += len(merged.observations)
        independence_group_count += len(merged.independence_groups)
        hypothesis_count += len(discovery.hypotheses)

    if all(status == "unavailable" for status in query_statuses):
        status = "unavailable"
    elif any(status != "complete" for status in query_statuses):
        status = "partial"
    else:
        status = "complete"
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "preview_id": str(preview["preview_id"]),
        "preview_content_sha256": str(preview["content_sha256"]),
        "approval_sha256": _hash(normalized_approval),
        "executed_at": normalized_executed_at,
        "status": status,
        "counts": {
            "planned_queries": len(query_plan),
            "accounted_queries": len(query_statuses),
            "unavailable_queries": unavailable_queries,
            "selected_records": selected_records,
            "observations": observation_count,
            "independence_groups": independence_group_count,
            "hypotheses": hypothesis_count,
            "provider_writes": 0,
            "accepted_effects": 0,
        },
        "artifacts": {
            "query_receipts": query_receipt_artifacts,
            "normalized": normalized_artifacts,
            "hypotheses": hypotheses_artifacts,
            "contacts_sha256": _hash({"contacts": effective_contacts}),
        },
        "action_vector": {
            "owned_corpus_read": True,
            "mailbox_or_provider_call": False,
            "runtime_write": True,
            "schema_migration": False,
            "deployment": False,
            "accepted_graph_write": False,
            "person_merge": False,
            "speaker_or_profile_effect": False,
            "graphiti_write": False,
        },
        "effects": dict(ZERO_EFFECTS),
    }
    receipt["content_sha256"] = _hash(receipt)
    write_immutable_private_json(aggregate_path, receipt)
    return {**receipt, "aggregate_path": str(aggregate_path), "idempotent": False}


def _private_artifact(
    directory: Path,
    descriptor: Mapping[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    name = str(descriptor.get("name") or "")
    if not name or Path(name).name != name:
        raise Plan0073PrivatePilotExecutionError(
            "P5 aggregate contains an unsafe artifact name."
        )
    path = directory / name
    require_private_file(path, runtime_root)
    value = read_private_object(path)
    if _hash(value) != str(descriptor.get("content_sha256") or ""):
        raise Plan0073PrivatePilotExecutionError(
            "P5 private replay artifact drifted after execution."
        )
    return value


def replay_private_pilot(
    preview_id: str,
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Replay approved P5 normalized inputs without a provider or corpus read."""

    if not _PREVIEW_ID_RE.fullmatch(str(preview_id or "")):
        raise Plan0073PrivatePilotExecutionError("P5 preview ID is invalid.")
    root = runtime_root.expanduser().absolute()
    run = root / "plan-0073/private-pilots" / preview_id
    required = {
        "preview": run / "preview.json",
        "approval": run / "approval.json",
        "aggregate": run / "aggregate-validation.json",
        "contacts": run / "normalized/contacts.json",
    }
    for path in required.values():
        require_private_file(path, root)
    preview = read_private_object(required["preview"])
    approval = read_private_object(required["approval"])
    aggregate = read_private_object(required["aggregate"])
    contacts_artifact = read_private_object(required["contacts"])
    try:
        validate_private_pilot_approval(preview, approval)
    except Plan0073PrivatePilotError as exc:
        raise Plan0073PrivatePilotExecutionError(
            "P5 replay authority no longer binds its preview."
        ) from exc
    if preview.get("preview_id") != preview_id:
        raise Plan0073PrivatePilotExecutionError("P5 replay preview ID drifted.")
    aggregate_core = dict(aggregate)
    aggregate_sha256 = str(aggregate_core.pop("content_sha256", ""))
    if (
        aggregate.get("schema_version") != EXECUTION_RECEIPT_SCHEMA
        or _hash(aggregate_core) != aggregate_sha256
        or aggregate.get("preview_content_sha256") != preview.get("content_sha256")
        or aggregate.get("effects") != ZERO_EFFECTS
    ):
        raise Plan0073PrivatePilotExecutionError(
            "P5 aggregate validation receipt drifted."
        )
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Plan0073PrivatePilotExecutionError("P5 aggregate artifacts are missing.")
    if _hash(contacts_artifact) != str(artifacts.get("contacts_sha256") or ""):
        raise Plan0073PrivatePilotExecutionError("P5 replay contacts drifted.")
    contacts = contacts_artifact.get("contacts")
    if not isinstance(contacts, Mapping):
        raise Plan0073PrivatePilotExecutionError("P5 replay contacts are invalid.")
    source = (preview.get("request") or {}).get("source_scope") or {}
    if not isinstance(source, Mapping):
        raise Plan0073PrivatePilotExecutionError("P5 replay source scope is invalid.")

    normalized_descriptors = artifacts.get("normalized")
    hypothesis_descriptors = artifacts.get("hypotheses")
    query_descriptors = artifacts.get("query_receipts")
    if not all(
        isinstance(value, list)
        for value in (
            normalized_descriptors,
            hypothesis_descriptors,
            query_descriptors,
        )
    ):
        raise Plan0073PrivatePilotExecutionError(
            "P5 replay artifact inventories are invalid."
        )
    hypotheses_by_name = {
        str(item.get("name") or ""): item for item in hypothesis_descriptors
    }
    observations = 0
    groups = 0
    hypotheses = 0
    for descriptor in normalized_descriptors:
        normalized = _private_artifact(
            run / "normalized", descriptor, runtime_root=root
        )
        name = str(descriptor.get("name") or "")
        expected_descriptor = hypotheses_by_name.get(name)
        if expected_descriptor is None:
            raise Plan0073PrivatePilotExecutionError(
                "P5 replay hypothesis inventory is incomplete."
            )
        expected = _private_artifact(
            run / "hypotheses", expected_descriptor, runtime_root=root
        )
        replayed = discover_mail_relationship_hypotheses(
            normalized.get("observations") or [],
            normalized.get("independence_groups") or [],
            contacts=contacts,
            account_address=str(source.get("account_address") or ""),
            input_watermark=str(normalized.get("input_watermark") or ""),
        )
        rebuilt = {
            "schema_version": "transcribe-audio.plan0073-p5-shadow-hypotheses.v1",
            "conversation_id": str(normalized.get("conversation_id") or ""),
            "input_watermark": replayed.input_watermark,
            "hypotheses": list(replayed.hypotheses),
            "excluded_reason_counts": replayed.excluded_reason_counts,
            "effects": dict(ZERO_EFFECTS),
        }
        if rebuilt != expected:
            raise Plan0073PrivatePilotExecutionError(
                "P5 offline hypothesis replay is not equal."
            )
        observations += len(normalized.get("observations") or [])
        groups += len(normalized.get("independence_groups") or [])
        hypotheses += len(replayed.hypotheses)
    if set(hypotheses_by_name) != {
        str(item.get("name") or "") for item in normalized_descriptors
    }:
        raise Plan0073PrivatePilotExecutionError(
            "P5 replay hypothesis inventory has extra artifacts."
        )

    selected_records = 0
    unavailable_queries = 0
    for descriptor in query_descriptors:
        receipt = _private_artifact(
            run / "query-receipts", descriptor, runtime_root=root
        )
        validate_mail_artifact("mail_query_receipt", receipt)
        selected_records += int((receipt.get("counts") or {}).get("selected", 0))
        unavailable_queries += int(receipt.get("status") == "unavailable")
    counts = {
        "planned_queries": len(query_descriptors),
        "accounted_queries": len(query_descriptors),
        "unavailable_queries": unavailable_queries,
        "selected_records": selected_records,
        "observations": observations,
        "independence_groups": groups,
        "hypotheses": hypotheses,
        "provider_writes": 0,
        "accepted_effects": 0,
    }
    if counts != aggregate.get("counts"):
        raise Plan0073PrivatePilotExecutionError(
            "P5 offline replay aggregate counts are not equal."
        )
    return {
        "schema_version": OFFLINE_REPLAY_SCHEMA,
        "preview_id": preview_id,
        "source_execution_sha256": aggregate_sha256,
        "replay_equal": True,
        "counts": counts,
        "effects": dict(ZERO_EFFECTS),
    }
