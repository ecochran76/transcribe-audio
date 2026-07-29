from __future__ import annotations

import os
import re
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import provenance_config
from conversation_evidence_adapters import AdapterSourceScope
from conversation_evidence_gws import (
    GWS_CAPABILITIES,
    GwsCliConfig,
    GwsCliReader,
    GwsEvidenceAdapter,
)
from conversation_evidence_odollo import (
    OdolloAdapterConfig,
    OdolloEvidenceAdapter,
)
from conversation_identity_retrieval import IdentityEvidencePolicy
from conversation_identity_retrieval import (
    PreparedIdentityEvidenceBundle,
    prepare_identity_evidence,
)
from conversation_knowledge_projection import (
    APPLY_APPROVAL_TOKEN,
    ConversationKnowledgeProjector,
    ReconciliationReceipt,
)
from conversation_knowledge_evidence import EvidenceScope


ODOLLO_CAPABILITIES = ("contacts", "leads", "log_notes")
LEGACY_ROLLBACK_APPROVAL_TOKEN = "USE_LEGACY_SPEAKER_EVIDENCE"
LEGACY_ROLLBACK_WARNING = (
    "Legacy speaker evidence rollback is active by explicit operator action."
)


@dataclass(frozen=True)
class IdentityEvidencePolicyBuild:
    policy: IdentityEvidencePolicy
    source_contexts: tuple[dict[str, Any], ...]
    retrieval_sources: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PreparedTranscriptIdentityEvidence:
    bundle: PreparedIdentityEvidenceBundle
    policy_build: IdentityEvidencePolicyBuild
    projection_receipt: ReconciliationReceipt
    shadow_root: Path
    retrieval_receipt_path: Path
    retrieval_receipt_sha256: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: object) -> str:
    return str(value or "").strip()


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
        + ("\n" if pretty else "")
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_exclusive_private_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    return tuple(
        value
        for value in values
        if value and not (value in seen or seen.add(value))
    )


def _gws_people_surfaces(config: Any) -> tuple[str, ...]:
    surfaces: list[str] = []
    if config.include_people_contacts:
        surfaces.append("contacts")
    if config.include_other_contacts:
        surfaces.append("other_contacts")
    if config.include_directory_people:
        surfaces.append("directory")
    return tuple(surfaces) or ("contacts",)


def _enabled_gws_capabilities(config: Any) -> set[str]:
    enabled: set[str] = set()
    if config.include_calendar_details:
        enabled.add("calendar")
    if config.include_drive_search:
        enabled.add("drive")
    if config.include_gmail_search:
        enabled.add("gmail")
    if (
        config.include_people_contacts
        or config.include_other_contacts
        or config.include_directory_people
    ):
        enabled.add("people")
    return enabled


def _enabled_odollo_capabilities(config: Any) -> set[str]:
    enabled: set[str] = set()
    if config.include_contacts:
        enabled.add("contacts")
    if config.include_leads:
        enabled.add("leads")
    if config.include_log_notes:
        enabled.add("log_notes")
    return enabled


def build_identity_evidence_policy(
    resolved: Mapping[str, Any],
    *,
    requested_at: str,
    request_id: str = "",
    run_id: str = "",
    environment: Mapping[str, str] | None = None,
    prepared_query_terms: tuple[str, ...] = (),
    max_records: int = 20,
    max_characters: int = 12_000,
    max_per_source: int = 5,
    max_provider_calls: int = 4,
) -> IdentityEvidencePolicyBuild:
    """Build an exact, explicit provider policy from validated provenance."""
    if not _text(requested_at):
        raise ValueError("Identity evidence requested_at is required.")
    runtime_environment = dict(environment if environment is not None else os.environ)
    if not runtime_environment:
        raise ValueError("Identity evidence provider environment must be explicit.")

    gws_configs = list(resolved.get("gws") or [])
    odollo_configs = list(resolved.get("odollo") or [])
    gws_index = 0
    odollo_index = 0
    scopes: list[EvidenceScope] = []
    capabilities: list[str] = []
    adapters: list[object] = []
    retrieval_sources: list[dict[str, Any]] = []

    for raw_source in resolved.get("retrieval_sources") or []:
        if not isinstance(raw_source, dict):
            raise TypeError("Retrieval source entries must be objects.")
        source_id = _text(raw_source.get("source_id"))
        source_profile_id = _text(raw_source.get("source_profile_id"))
        provider_kind = _text(raw_source.get("provider_kind"))
        account_id = _text(raw_source.get("account_id"))
        tenant_id = _text(raw_source.get("tenant_id"))
        source_capabilities = _unique(
            [_text(value) for value in raw_source.get("evidence_capabilities") or []]
        )
        if not source_id or source_profile_id != source_id:
            raise ValueError(
                "Retrieval source_profile_id must equal its explicit source_id."
            )
        if not source_capabilities:
            raise ValueError("Retrieval source capabilities must be explicit.")

        scope = AdapterSourceScope(
            source_profile_id=source_profile_id,
            provider_kind=provider_kind,
            account_id=account_id,
            tenant_id=tenant_id,
            capabilities=source_capabilities,
        )
        if provider_kind == "gws":
            if gws_index >= len(gws_configs):
                raise ValueError("GWS retrieval source has no resolved provider config.")
            config = gws_configs[gws_index]
            gws_index += 1
            unsupported = set(source_capabilities) - set(GWS_CAPABILITIES)
            disabled = set(source_capabilities) - _enabled_gws_capabilities(config)
            if unsupported or disabled:
                raise ValueError(
                    "GWS retrieval capabilities are unsupported or not enabled."
                )
            if config.config_dir is None:
                raise ValueError("GWS retrieval requires an explicit config_dir.")
            provider = GwsCliReader(
                GwsCliConfig(
                    config_dir=str(config.config_dir),
                    environment=runtime_environment,
                    timeout=float(config.timeout),
                    people_surfaces=_gws_people_surfaces(config),
                )
            )
            adapters.append(
                GwsEvidenceAdapter(
                    scope=scope,
                    provider=provider,
                    retrieved_at=requested_at,
                )
            )
        elif provider_kind == "odollo":
            if odollo_index >= len(odollo_configs):
                raise ValueError(
                    "Odollo retrieval source has no resolved provider config."
                )
            config = odollo_configs[odollo_index]
            odollo_index += 1
            unsupported = set(source_capabilities) - set(ODOLLO_CAPABILITIES)
            disabled = set(source_capabilities) - _enabled_odollo_capabilities(config)
            if unsupported or disabled:
                raise ValueError(
                    "Odollo retrieval capabilities are unsupported or not enabled."
                )
            if tuple(config.profiles) != (tenant_id,):
                raise ValueError(
                    "Odollo retrieval tenant must match its explicit provider profile."
                )
            adapters.append(
                OdolloEvidenceAdapter(
                    OdolloAdapterConfig(
                        scope=scope,
                        command=tuple(config.command),
                        repo_root=config.repo_root,
                        config_path=config.config_path,
                        timeout=float(config.timeout),
                    ),
                    retrieved_at=lambda value=requested_at: value,
                )
            )
        else:
            raise ValueError("Identity retrieval provider_kind is unsupported.")

        scopes.append(
            EvidenceScope(
                source_profile_id=source_profile_id,
                account_id=account_id,
                tenant_id=tenant_id,
            )
        )
        capabilities.extend(source_capabilities)
        retrieval_sources.append(dict(raw_source))

    if gws_index != len(gws_configs) or odollo_index != len(odollo_configs):
        raise ValueError("Resolved provider configs and retrieval sources diverge.")

    policy = IdentityEvidencePolicy(
        scopes=tuple(scopes),
        capabilities=_unique(capabilities),
        prepared_query_terms=_unique(
            [_text(value) for value in prepared_query_terms]
        ),
        provider_adapters=tuple(adapters),
        hindsight_policy="allow_later_retrieved",
        request_id=request_id,
        run_id=run_id,
        requested_at=requested_at,
        max_records=max_records,
        max_characters=max_characters,
        max_per_source=max_per_source,
        max_provider_calls=max_provider_calls,
    )
    return IdentityEvidencePolicyBuild(
        policy=policy,
        source_contexts=tuple(
            dict(value)
            for value in resolved.get("source_contexts") or []
            if isinstance(value, dict)
        ),
        retrieval_sources=tuple(retrieval_sources),
        warnings=tuple(_text(value) for value in resolved.get("warnings") or []),
    )


def discovery_retrieval_inputs(
    discovery_readout: Mapping[str, Any],
    *,
    utterance_ids: tuple[str, ...],
    default_speaker_labels: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Translate prepared utterance-N citations to durable projected IDs."""
    cited: list[str] = []
    labels: list[str] = list(default_speaker_labels)
    for collection_name in (
        "speaker_clues",
        "conversation_clues",
        "speaker_group_hints",
        "mixed_speaker_hints",
    ):
        collection = discovery_readout.get(collection_name)
        for item in collection if isinstance(collection, list) else []:
            if not isinstance(item, dict):
                continue
            label = _text(item.get("speaker_label"))
            if label:
                labels.append(label)
            group_labels = item.get("speaker_labels")
            if isinstance(group_labels, list):
                labels.extend(_text(value) for value in group_labels)
            clue_ids = item.get("transcript_clue_ids")
            if isinstance(clue_ids, list):
                cited.extend(_text(value) for value in clue_ids)

    durable_ids: list[str] = []
    available = set(utterance_ids)
    for clue_id in _unique(cited):
        if clue_id in available:
            durable_ids.append(clue_id)
            continue
        match = re.fullmatch(r"utterance-(\d+)", clue_id)
        if not match:
            raise ValueError("Discovery readout contains an unsupported clue identity.")
        ordinal = int(match.group(1)) - 1
        if ordinal < 0 or ordinal >= len(utterance_ids):
            raise ValueError("Discovery readout clue identity is outside the transcript.")
        durable_ids.append(utterance_ids[ordinal])
    return _unique(labels), _unique(durable_ids)


def discovery_provider_terms(
    discovery_readout: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return bounded-query inputs explicitly proposed by Clue Discovery."""
    values: list[str] = []
    for collection_name in (
        "speaker_clues",
        "conversation_clues",
        "speaker_group_hints",
        "mixed_speaker_hints",
    ):
        collection = discovery_readout.get(collection_name)
        for item in collection if isinstance(collection, list) else []:
            if not isinstance(item, dict):
                continue
            person_hints = item.get("person_hints")
            for hint in person_hints if isinstance(person_hints, list) else []:
                if not isinstance(hint, dict):
                    continue
                values.extend(
                    _text(hint.get(field))
                    for field in ("email", "name", "organization")
                )
            retrieval_terms = item.get("retrieval_terms")
            if isinstance(retrieval_terms, list):
                values.extend(_text(value) for value in retrieval_terms)
    return _unique(values)


def prepare_transcript_identity_evidence(
    transcript_path: Path,
    discovery_readout: Mapping[str, Any],
    *,
    state_root: Path,
    provenance_path: Path | None = None,
    provenance_profile: str | None = None,
    environment: Mapping[str, str] | None = None,
    resolved: Mapping[str, Any] | None = None,
    run_id: str = "",
    requested_at: str = "",
) -> PreparedTranscriptIdentityEvidence:
    """Prepare the default immutable retrieval bundle in a private shadow store."""
    effective_requested_at = _text(requested_at) or _utc_now()
    shadow_root = state_root.expanduser().resolve() / "conversation-identity-shadow"
    shadow_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    shadow_root.chmod(0o700)

    projector = ConversationKnowledgeProjector(shadow_root)
    plan = projector.preview(transcript_path, document_id="")
    projection_receipt = projector.apply(
        plan,
        approval_token=APPLY_APPROVAL_TOKEN,
    )
    utterance_ids = tuple(
        item.utterance_id for item in plan.conversation_snapshot.utterances
    )
    default_labels = _unique(
        [
            _text(item.speaker_label)
            for item in plan.conversation_snapshot.utterances
        ]
    )
    speaker_labels, clue_ids = discovery_retrieval_inputs(
        discovery_readout,
        utterance_ids=utterance_ids,
        default_speaker_labels=default_labels,
    )
    if resolved is None:
        resolved = (
            provenance_config.speaker_preprocessing_source_configs_from_provenance(
                path=provenance_path,
                state_root=state_root,
                profile=provenance_profile,
            )
        )
    policy_build = build_identity_evidence_policy(
        resolved,
        requested_at=effective_requested_at,
        request_id=str(uuid4()),
        run_id=run_id,
        environment=environment,
        prepared_query_terms=discovery_provider_terms(discovery_readout),
    )
    bundle = prepare_identity_evidence(
        plan.processing_history.conversation_id,
        speaker_labels=speaker_labels,
        clue_ids=clue_ids,
        policy=policy_build.policy,
        root=projector.root,
    )
    request_payload = {
        "request_id": bundle.request.request_id,
        "conversation_id": bundle.request.conversation_id,
        "recording_ids": list(bundle.request.recording_ids),
        "speaker_labels": list(bundle.request.speaker_labels),
        "clue_ids": list(bundle.request.clue_ids),
        "conversation_at": bundle.request.conversation_at,
        "as_of": bundle.request.as_of,
        "prepared_person_ids": list(bundle.request.prepared_person_ids),
        "scopes": [
            {
                "source_profile_id": scope.source_profile_id,
                "account_id": scope.account_id,
                "tenant_id": scope.tenant_id,
            }
            for scope in bundle.request.scopes
        ],
        "capabilities": list(bundle.request.capabilities),
        "budgets": bundle.request.budgets,
        "freshness_policy": bundle.request.freshness_policy,
        "hindsight_policy": bundle.request.hindsight_policy,
        "retrieval_version": bundle.request.retrieval_version,
        "ranking_version": bundle.request.ranking_version,
        "requesting_workflow": bundle.request.requesting_workflow,
        "run_id": bundle.request.run_id,
        "created_at": bundle.request.created_at,
    }
    query_plan_payload = {
        "conversation_id": bundle.request.conversation_id,
        "speaker_labels": list(bundle.request.speaker_labels),
        "clue_ids": list(bundle.request.clue_ids),
        "as_of": bundle.request.as_of,
        "scopes": request_payload["scopes"],
        "capabilities": request_payload["capabilities"],
        "prepared_person_ids": request_payload["prepared_person_ids"],
        "query_terms": bundle.request.budgets.get("query_terms") or [],
        "max_records": bundle.request.budgets.get("max_records"),
        "max_characters": bundle.request.budgets.get("max_characters"),
        "max_provider_calls": bundle.request.budgets.get("max_provider_calls"),
        "freshness_policy": bundle.request.freshness_policy,
        "hindsight_policy": bundle.request.hindsight_policy,
    }
    retrieval_receipt_payload = {
        "schema_version": "transcribe-audio.identity-retrieval-receipt.v1",
        "authority_mode": "sidecar_shadow",
        "request_id": bundle.request.request_id,
        "request_sha256": _sha256(request_payload),
        "query_plan_sha256": _sha256(query_plan_payload),
        "bundle_id": bundle.persisted_bundle.bundle_id,
        "bundle_sha256": bundle.persisted_bundle.content_hash,
        "bundle_status": bundle.persisted_bundle.status,
        "source_failures": [
            dict(value) for value in bundle.source_failures
        ],
        "warnings": list(bundle.warnings),
        "evidence_controls": [
            {
                "evidence_id": item.snapshot.evidence_id,
                "source_profile_id": item.snapshot.source_profile_id,
                "capability": item.snapshot.capability,
                "temporal_class": item.snapshot.temporal_class,
                "freshness_state": item.snapshot.freshness_state,
                "independence_group_id": (
                    item.snapshot.independence_group_id
                ),
                "disposition": item.disposition,
                "reason_code": item.reason_code,
            }
            for item in bundle.evidence
        ],
        "projection_receipt_path": projection_receipt.receipt_path,
        "recorded_at": effective_requested_at,
        "will_perform_external_write": False,
    }
    retrieval_receipt_dir = projector.root / "identity-retrieval-receipts"
    retrieval_receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    retrieval_receipt_dir.chmod(0o700)
    retrieval_receipt_path = (
        retrieval_receipt_dir / f"{bundle.request.request_id}.json"
    )
    retrieval_receipt_bytes = _canonical_bytes(
        retrieval_receipt_payload,
        pretty=True,
    )
    _write_exclusive_private_bytes(
        retrieval_receipt_path,
        retrieval_receipt_bytes,
    )
    return PreparedTranscriptIdentityEvidence(
        bundle=bundle,
        policy_build=policy_build,
        projection_receipt=projection_receipt,
        shadow_root=shadow_root,
        retrieval_receipt_path=retrieval_receipt_path,
        retrieval_receipt_sha256=hashlib.sha256(
            retrieval_receipt_bytes
        ).hexdigest(),
    )


def record_legacy_rollback(
    *,
    state_root: Path,
    document_id: str,
    operator: str,
    approval_token: str,
) -> dict[str, str]:
    """Record one explicit, warning-bearing legacy evidence rollback action."""
    if approval_token != LEGACY_ROLLBACK_APPROVAL_TOKEN:
        raise ValueError("Legacy evidence rollback requires its approval token.")
    if not _text(operator):
        raise ValueError("Legacy evidence rollback requires an operator.")
    receipt_dir = (
        state_root.expanduser().resolve() / "plan-0030" / "legacy-rollbacks"
    )
    receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_dir.chmod(0o700)
    receipt_id = str(uuid4())
    receipt_path = receipt_dir / f"{receipt_id}.json"
    payload = {
        "schema_version": "transcribe-audio.legacy-evidence-rollback.v1",
        "receipt_id": receipt_id,
        "document_id": _text(document_id),
        "operator": _text(operator),
        "warning": LEGACY_ROLLBACK_WARNING,
        "recorded_at": _utc_now(),
        "will_perform_external_write": False,
    }
    encoded = _canonical_bytes(payload, pretty=True)
    _write_exclusive_private_bytes(receipt_path, encoded)
    return {
        "receipt_id": receipt_id,
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "warning": LEGACY_ROLLBACK_WARNING,
    }
