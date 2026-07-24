#!/usr/bin/env python3
"""
User-scoped provenance source configuration.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_CONFIG_PATH = DEFAULT_STATE_DIR / "provenance.config.json"
ENV_CONFIG_PATH = "TRANSCRIPTS_PROVENANCE_CONFIG"
SCHEMA_VERSION = "transcribe-audio.provenance-config.v1"
APPLY_APPROVAL_TOKEN = "APPLY_PROVENANCE_CONFIG_UPDATE"
SAMPLE_CONFIG_PATH = Path(__file__).resolve().parent / "provenance.config.json.sample"

SOURCE_KIND_GOG = "gog"
SOURCE_KIND_GWS = "gws"
SOURCE_KIND_MSGCLI = "msgcli"
SOURCE_KIND_ODOLLO = "odollo"
SOURCE_KIND_ICAL = "ical_calendar"
REGISTERED_SOURCE_KINDS = {
    SOURCE_KIND_GOG,
    SOURCE_KIND_GWS,
    SOURCE_KIND_MSGCLI,
    SOURCE_KIND_ODOLLO,
    SOURCE_KIND_ICAL,
}
REDACTED = "[redacted]"
DEFAULT_SENSITIVE_FIELDS = {
    "api_key",
    "password",
    "secret",
    "token",
    "url",
    "url_ref",
}


@dataclass(frozen=True)
class CalendarMetadataSettings:
    config_path: str
    profile: str
    primary_calendar_id: str = "primary"
    provider_configs: list[Any] = field(default_factory=list)
    provenance_calendar_ids: list[str] = field(default_factory=list)
    provenance_ical_urls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "provider_configs": [
                {
                    "name": getattr(config, "name", ""),
                    "account": getattr(config, "account", None),
                    "client": getattr(config, "client", None),
                    "config_dir": str(getattr(config, "config_dir", "") or ""),
                }
                for config in self.provider_configs
            ],
            "provenance_ical_urls": [redact_ical_spec(value) for value in self.provenance_ical_urls],
        }


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def config_path(path: Optional[Path] = None, *, state_root: Optional[Path] = None) -> Path:
    if path:
        return path.expanduser()
    env_value = os.getenv(ENV_CONFIG_PATH)
    if env_value:
        return Path(env_value).expanduser()
    root = state_root.expanduser() if state_root else DEFAULT_STATE_DIR.expanduser()
    return root / DEFAULT_CONFIG_PATH.name


def read_config(path: Optional[Path] = None, *, state_root: Optional[Path] = None) -> dict[str, Any]:
    target = config_path(path, state_root=state_root)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Provenance config {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Provenance config {target} must contain a JSON object.")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def write_sample_config(path: Optional[Path] = None, *, state_root: Optional[Path] = None, force: bool = False) -> Path:
    target = config_path(path, state_root=state_root)
    if target.exists() and not force:
        raise ValueError(f"Provenance config already exists: {target}")
    payload = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(payload)
    atomic_write_json(target, payload)
    return target


def base_config(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": raw.get("schema_version") or SCHEMA_VERSION,
        "active_profile": str(raw.get("active_profile") or "default"),
        "profiles": copy.deepcopy(raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}),
        "sources": copy.deepcopy(raw.get("sources") if isinstance(raw.get("sources"), dict) else {}),
        "contacts": copy.deepcopy(raw.get("contacts") if isinstance(raw.get("contacts"), dict) else {}),
        "mutation_policy": copy.deepcopy(
            raw.get("mutation_policy") if isinstance(raw.get("mutation_policy"), dict) else {}
        ),
    }


def active_profile_name(raw: dict[str, Any], profile: Optional[str] = None) -> str:
    return str(profile or raw.get("active_profile") or "default")


def profile_payload(raw: dict[str, Any], profile: Optional[str] = None) -> dict[str, Any]:
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    name = active_profile_name(raw, profile)
    payload = profiles.get(name) if isinstance(profiles.get(name), dict) else {}
    return payload


def workflow_payload(raw: dict[str, Any], workflow: str, profile: Optional[str] = None) -> dict[str, Any]:
    profile_config = profile_payload(raw, profile)
    workflows = profile_config.get("workflows") if isinstance(profile_config.get("workflows"), dict) else {}
    payload = workflows.get(workflow) if isinstance(workflows.get(workflow), dict) else {}
    return payload


def enabled_source(raw: dict[str, Any], source_id: str) -> Optional[dict[str, Any]]:
    sources = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    source = sources.get(source_id)
    if not isinstance(source, dict) or source.get("enabled", True) is False:
        return None
    return source


def profile_source_ids(raw: dict[str, Any], profile: Optional[str] = None) -> list[str]:
    profile_config = profile_payload(raw, profile)
    values = profile_config.get("source_ids")
    return unique_strings(values if isinstance(values, list) else [])


def workflow_source_ids(raw: dict[str, Any], workflow: str, profile: Optional[str] = None) -> list[str]:
    workflow_config = workflow_payload(raw, workflow, profile)
    values = workflow_config.get("source_ids")
    if isinstance(values, list):
        return unique_strings(values)
    if workflow == "calendar_metadata":
        result: list[str] = []
        primary = workflow_config.get("primary") if isinstance(workflow_config.get("primary"), dict) else {}
        if primary.get("source_id"):
            result.append(str(primary["source_id"]))
        provenance_sources = workflow_config.get("provenance_sources")
        if isinstance(provenance_sources, list):
            for item in provenance_sources:
                if isinstance(item, dict) and item.get("source_id"):
                    result.append(str(item["source_id"]))
        return unique_strings(result)
    return profile_source_ids(raw, profile)


def source_capabilities(source: dict[str, Any]) -> set[str]:
    values = source.get("capabilities")
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values if str(item)}


def _context_source_configs_for_ids(
    raw: dict[str, Any],
    source_ids: Iterable[str],
) -> dict[str, Any]:
    from context_sources import GwsProvenanceConfig, OdolloProvenanceConfig

    gws_configs: list[GwsProvenanceConfig] = []
    odollo_configs: list[OdolloProvenanceConfig] = []
    msgcli_configs: list[dict[str, Any]] = []
    for source_id in source_ids:
        source = enabled_source(raw, source_id)
        if not source:
            continue
        kind = str(source.get("kind") or "")
        capabilities = source_capabilities(source)
        if kind == SOURCE_KIND_GWS:
            people = source.get("people") if isinstance(source.get("people"), dict) else {}
            gmail = source.get("gmail") if isinstance(source.get("gmail"), dict) else {}
            surfaces = set(people.get("surfaces") if isinstance(people.get("surfaces"), list) else [])
            gws_configs.append(
                GwsProvenanceConfig(
                    enabled=True,
                    profile_label=str(source.get("label") or source_id),
                    config_dir=Path(str(source["config_dir"])).expanduser() if source.get("config_dir") else None,
                    drive_page_size=int((source.get("drive") or {}).get("page_size") or 5)
                    if isinstance(source.get("drive"), dict)
                    else 5,
                    people_page_size=int(people.get("limit") or 5),
                    people_query_limit=int(people.get("query_limit") or 8),
                    timeout=float(source.get("timeout") or 30.0),
                    include_calendar_details="calendar" in capabilities,
                    include_drive_search="drive" in capabilities,
                    include_gmail_search="gmail" in capabilities,
                    gmail_page_size=int(gmail.get("page_size") or 5),
                    include_people_contacts="contacts" in surfaces or "people" in capabilities,
                    include_other_contacts="other_contacts" in surfaces,
                    include_directory_people="directory" in surfaces,
                )
            )
        elif kind == SOURCE_KIND_ODOLLO:
            limits = source.get("limits") if isinstance(source.get("limits"), dict) else {}
            command = source.get("command") if isinstance(source.get("command"), list) else []
            default_odollo = OdolloProvenanceConfig()
            odollo_configs.append(
                OdolloProvenanceConfig(
                    enabled=True,
                    profiles=(str(source.get("tenant_profile") or source.get("profile") or source_id),),
                    command=tuple(str(item) for item in command) or default_odollo.command,
                    repo_root=Path(str(source["repo_root"])).expanduser()
                    if source.get("repo_root")
                    else default_odollo.repo_root,
                    config_path=Path(str(source["config_path"])).expanduser()
                    if source.get("config_path")
                    else default_odollo.config_path,
                    timeout=float(source.get("timeout") or 30.0),
                    limit=int(limits.get("contacts") or limits.get("log_notes") or source.get("limit") or 5),
                    include_contacts="res.partner" in source.get("models", ["res.partner"])
                    if isinstance(source.get("models", ["res.partner"]), list)
                    else True,
                    include_leads="crm.lead" in source.get("models", []),
                    include_log_notes="mail.message" in source.get("models", []),
                )
            )
        elif kind == SOURCE_KIND_MSGCLI:
            msgcli_configs.append(redacted_config({"sources": {source_id: source}})["sources"][source_id])
    return {
        "gws": gws_configs,
        "odollo": odollo_configs,
        "msgcli": msgcli_configs,
        "warnings": [],
    }


def _validated_source_context(value: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, dict):
        return {}, ["Source Context is required"]
    errors: list[str] = []
    owner = value.get("owner") if isinstance(value.get("owner"), dict) else {}
    owner_type = str(owner.get("type") or "")
    if owner_type not in {"person", "organization"}:
        errors.append("owner.type must be person or organization")
    for field_name in ("id", "label"):
        if not str(owner.get(field_name) or "").strip():
            errors.append(f"owner.{field_name} is required")
    relationship_scope = str(value.get("relationship_scope") or "").strip()
    if not relationship_scope:
        errors.append("relationship_scope is required")
    account_label = str(value.get("account_label") or "").strip()
    if not account_label:
        errors.append("account_label is required")
    evidence_capabilities = unique_strings(
        value.get("evidence_capabilities")
        if isinstance(value.get("evidence_capabilities"), list)
        else []
    )
    if not evidence_capabilities:
        errors.append("evidence_capabilities must not be empty")
    authoritative_identifiers = unique_strings(
        value.get("authoritative_identifiers")
        if isinstance(value.get("authoritative_identifiers"), list)
        else []
    )
    if not isinstance(value.get("authoritative_identifiers"), list):
        errors.append("authoritative_identifiers must be a list")
    if errors:
        return {}, errors
    return {
        "owner": {
            "type": owner_type,
            "id": str(owner["id"]).strip(),
            "label": str(owner["label"]).strip(),
        },
        "relationship_scope": relationship_scope,
        "account_label": account_label,
        "evidence_capabilities": evidence_capabilities,
        "authoritative_identifiers": authoritative_identifiers,
    }, []


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Provenance config must be a JSON object.")
    errors: list[str] = []
    warnings: list[str] = []
    schema = raw.get("schema_version")
    if schema not in (None, SCHEMA_VERSION):
        errors.append(f"Unsupported schema_version: {schema}")
    sources = raw.get("sources") if isinstance(raw.get("sources"), dict) else {}
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    for source_id, source in sources.items():
        if not isinstance(source, dict):
            errors.append(f"sources.{source_id} must be an object.")
            continue
        kind = str(source.get("kind") or "")
        enabled = source.get("enabled", True) is not False
        if not kind:
            errors.append(f"sources.{source_id}.kind is required.")
        elif kind not in REGISTERED_SOURCE_KINDS and enabled and not source.get("planned"):
            errors.append(f"sources.{source_id}.kind '{kind}' has no registered adapter.")
        if source.get("read_only", True) is not True:
            warnings.append(f"sources.{source_id} is not marked read_only.")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            errors.append(f"profiles.{profile_id} must be an object.")
            continue
        for source_id in unique_strings(profile.get("source_ids") if isinstance(profile.get("source_ids"), list) else []):
            if source_id not in sources:
                errors.append(f"profiles.{profile_id} references unknown source_id {source_id}.")
        workflows = profile.get("workflows") if isinstance(profile.get("workflows"), dict) else {}
        for workflow_name in workflows:
            for source_id in workflow_source_ids(raw, workflow_name, profile_id):
                if source_id not in sources:
                    errors.append(
                        f"profiles.{profile_id}.workflows.{workflow_name} references unknown source_id {source_id}."
                    )
    if errors:
        raise ValueError("; ".join(errors))
    return {"valid": True, "warnings": warnings}


def redact_value(key: str, value: Any, sensitive_fields: set[str]) -> Any:
    lowered = key.lower()
    if lowered in sensitive_fields or any(term in lowered for term in ("secret", "token", "password", "api_key")):
        return REDACTED if value not in (None, "") else value
    if isinstance(value, dict):
        nested_sensitive = set(sensitive_fields)
        extra = value.get("sensitive_fields")
        if isinstance(extra, list):
            nested_sensitive.update(str(item).lower() for item in extra)
        return {nested_key: redact_value(nested_key, nested_value, nested_sensitive) for nested_key, nested_value in value.items()}
    if isinstance(value, list):
        return [redact_value(key, item, sensitive_fields) for item in value]
    return value


def redacted_config(raw: dict[str, Any]) -> dict[str, Any]:
    payload = base_config(raw)
    sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
    for source_id, source in list(sources.items()):
        if not isinstance(source, dict):
            continue
        sensitive = set(DEFAULT_SENSITIVE_FIELDS)
        extra = source.get("sensitive_fields")
        if isinstance(extra, list):
            sensitive.update(str(item).lower() for item in extra)
        sources[source_id] = {key: redact_value(key, value, sensitive) for key, value in source.items()}
    return payload


def redact_ical_spec(value: str) -> str:
    text = str(value or "")
    if "=http" in text:
        label, _, _url = text.partition("=")
        return f"{label}={REDACTED}"
    if text.startswith("http"):
        return REDACTED
    return text


def resolve_secret_ref(value: Any) -> tuple[str, Optional[str]]:
    text = str(value or "").strip()
    if not text:
        return "", None
    if text.startswith("env:"):
        name = text[4:]
        resolved = os.getenv(name, "")
        if not resolved:
            return "", f"Environment variable {name} is not set."
        return resolved, None
    return "", f"Unsupported secret ref: {text}"


def maybe_resolve_url(source_id: str, source: dict[str, Any]) -> tuple[str, Optional[str]]:
    url = str(source.get("url") or "").strip()
    if url:
        return url, None
    resolved, warning = resolve_secret_ref(source.get("url_ref"))
    if warning:
        return "", f"iCalendar source {source_id}: {warning}"
    return resolved, None


def _calendar_provider_config_for_source(source: dict[str, Any]) -> Any:
    from transcribe_common import CalendarProviderConfig

    kind = str(source.get("kind") or "")
    if kind == SOURCE_KIND_GOG:
        return CalendarProviderConfig(
            name="gog",
            account=str(source.get("account") or "") or None,
            client=str(source.get("client") or "") or None,
        )
    if kind == SOURCE_KIND_GWS:
        config_dir = Path(str(source["config_dir"])).expanduser() if source.get("config_dir") else None
        return CalendarProviderConfig(name="gws", config_dir=config_dir)
    return None


def resolve_calendar_metadata_settings(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> CalendarMetadataSettings:
    target = config_path(path, state_root=state_root)
    raw = read_config(target)
    if not raw:
        return CalendarMetadataSettings(config_path=str(target), profile=str(profile or "default"))
    validate_config(raw)
    profile_name = active_profile_name(raw, profile)
    workflow = workflow_payload(raw, "calendar_metadata", profile_name)
    if workflow.get("enabled", True) is False:
        return CalendarMetadataSettings(config_path=str(target), profile=profile_name)
    primary = workflow.get("primary") if isinstance(workflow.get("primary"), dict) else {}
    primary_calendar_id = str(primary.get("calendar_id") or "primary")
    provider_configs: list[Any] = []
    provider_source_ids: list[str] = []
    warnings: list[str] = []
    provenance_calendar_ids: list[str] = []
    provenance_ical_urls: list[str] = []

    for source_id in workflow_source_ids(raw, "calendar_metadata", profile_name):
        source = enabled_source(raw, source_id)
        if not source:
            continue
        kind = str(source.get("kind") or "")
        if kind in {SOURCE_KIND_GOG, SOURCE_KIND_GWS}:
            provider_config = _calendar_provider_config_for_source(source)
            if provider_config and getattr(provider_config, "name", "") not in [getattr(item, "name", "") for item in provider_configs]:
                provider_configs.append(provider_config)
            provider_source_ids.append(source_id)
        elif kind == SOURCE_KIND_ICAL:
            url, warning = maybe_resolve_url(source_id, source)
            if warning:
                warnings.append(warning)
            if url:
                label = str(source.get("label") or source_id)
                provenance_ical_urls.append(f"{label}={url}")

    provenance_sources = workflow.get("provenance_sources")
    if isinstance(provenance_sources, list):
        for item in provenance_sources:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            source = enabled_source(raw, source_id) if source_id else None
            if not source:
                continue
            if str(source.get("kind") or "") in {SOURCE_KIND_GOG, SOURCE_KIND_GWS}:
                ids = item.get("calendar_ids")
                if isinstance(ids, list):
                    provenance_calendar_ids.extend(str(value) for value in ids if str(value))
                source_calendar_ids = (source.get("calendar") or {}).get("calendar_ids") if isinstance(source.get("calendar"), dict) else []
                if isinstance(source_calendar_ids, list):
                    provenance_calendar_ids.extend(str(value) for value in source_calendar_ids if str(value))

    return CalendarMetadataSettings(
        config_path=str(target),
        profile=profile_name,
        primary_calendar_id=primary_calendar_id,
        provider_configs=provider_configs,
        provenance_calendar_ids=unique_strings(provenance_calendar_ids),
        provenance_ical_urls=unique_strings(provenance_ical_urls),
        warnings=warnings,
        source_ids=unique_strings([*provider_source_ids, *workflow_source_ids(raw, "calendar_metadata", profile_name)]),
    )


def apply_calendar_settings_to_args(args: argparse.Namespace) -> CalendarMetadataSettings:
    settings = resolve_calendar_metadata_settings(
        path=getattr(args, "provenance_config", None),
        profile=getattr(args, "provenance_profile", None),
    )
    if settings.primary_calendar_id and getattr(args, "calendar_id", "primary") == "primary":
        args.calendar_id = settings.primary_calendar_id
    configured_ids = list(settings.provenance_calendar_ids)
    explicit_ids = list(getattr(args, "calendar_provenance_calendar_ids", None) or [])
    args.calendar_provenance_calendar_ids = unique_strings([*configured_ids, *explicit_ids])
    configured_ical = list(settings.provenance_ical_urls)
    explicit_ical = list(getattr(args, "calendar_provenance_ical_urls", None) or [])
    args.calendar_provenance_ical_urls = unique_strings([*configured_ical, *explicit_ical])
    args.provenance_config_source = settings.config_path
    args.provenance_config_profile = settings.profile
    args.provenance_config_warnings = settings.warnings
    return settings


def configured_provider_configs_or_fallback(args: argparse.Namespace, fallback_provider_configs: list[Any]) -> list[Any]:
    settings = apply_calendar_settings_to_args(args)
    if getattr(args, "calendar_providers", None):
        return fallback_provider_configs
    return settings.provider_configs or fallback_provider_configs


def contact_source_config_from_provenance(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    raw = read_config(path, state_root=state_root)
    if not raw:
        return {}
    validate_config(raw)
    profile_name = active_profile_name(raw, profile)
    workflow = workflow_payload(raw, "participant_identity", profile_name)
    if workflow.get("enabled", True) is False:
        return {}
    source_ids = workflow_source_ids(raw, "participant_identity", profile_name)
    gws_profiles: list[dict[str, Any]] = []
    odollo_profiles: list[dict[str, Any]] = []
    for source_id in source_ids:
        source = enabled_source(raw, source_id)
        if not source:
            continue
        kind = str(source.get("kind") or "")
        if kind == SOURCE_KIND_GWS:
            people = source.get("people") if isinstance(source.get("people"), dict) else {}
            gws_profiles.append(
                {
                    "label": str(source.get("label") or source_id),
                    "config_dir": str(source.get("config_dir") or ""),
                    "surfaces": people.get("surfaces") if isinstance(people.get("surfaces"), list) else ["contacts"],
                    "limit": int(people.get("limit") or 5),
                    "query_limit": int(people.get("query_limit") or 8),
                    "timeout": float(source.get("timeout") or people.get("timeout") or 30.0),
                }
            )
        elif kind == SOURCE_KIND_ODOLLO:
            limits = source.get("limits") if isinstance(source.get("limits"), dict) else {}
            command = source.get("command") if isinstance(source.get("command"), list) else None
            odollo_profiles.append(
                {
                    "label": str(source.get("tenant_profile") or source.get("profile") or source_id),
                    "display_label": str(source.get("label") or source_id),
                    "repo_root": str(source.get("repo_root") or ""),
                    "config_path": str(source.get("config_path") or ""),
                    "command": command,
                    "limit": int(limits.get("contacts") or source.get("limit") or 5),
                    "timeout": float(source.get("timeout") or 30.0),
                }
            )
    result: dict[str, Any] = {}
    if gws_profiles:
        result["gws"] = {"profiles": gws_profiles}
    if odollo_profiles:
        result["odollo"] = {"profiles": odollo_profiles}
    return result


def context_source_configs_from_provenance(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    raw = read_config(path, state_root=state_root)
    if not raw:
        return {"gws": [], "odollo": [], "msgcli": [], "warnings": []}
    validate_config(raw)
    profile_name = active_profile_name(raw, profile)
    workflow = workflow_payload(raw, "context_workbench", profile_name)
    if workflow.get("enabled", True) is False:
        return {"gws": [], "odollo": [], "msgcli": [], "warnings": []}

    return _context_source_configs_for_ids(
        raw,
        workflow_source_ids(raw, "context_workbench", profile_name),
    )


def speaker_preprocessing_source_configs_from_provenance(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve only sources with explicit semantic Source Context."""
    raw = read_config(path, state_root=state_root)
    if not raw:
        return {"gws": [], "odollo": [], "msgcli": [], "source_contexts": [], "warnings": []}
    validate_config(raw)
    profile_name = active_profile_name(raw, profile)
    workflow = workflow_payload(raw, "context_workbench", profile_name)
    if workflow.get("enabled", True) is False:
        return {"gws": [], "odollo": [], "msgcli": [], "source_contexts": [], "warnings": []}

    eligible_ids: list[str] = []
    source_contexts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source_id in workflow_source_ids(raw, "context_workbench", profile_name):
        source = enabled_source(raw, source_id)
        if not source:
            continue
        kind = str(source.get("kind") or "")
        if kind not in {SOURCE_KIND_GWS, SOURCE_KIND_ODOLLO}:
            continue
        source_context_value = source.get("source_context")
        if not isinstance(source_context_value, dict):
            warnings.append(f"Speaker preprocessing excluded source {source_id}: missing Source Context.")
            continue
        source_context, context_errors = _validated_source_context(source_context_value)
        if context_errors:
            warnings.append(
                f"Speaker preprocessing excluded source {source_id}: "
                f"invalid Source Context ({'; '.join(context_errors)})."
            )
            continue
        eligible_ids.append(source_id)
        source_contexts.append({"source_id": source_id, **source_context})

    configs = _context_source_configs_for_ids(raw, eligible_ids)
    return {
        **configs,
        "source_contexts": source_contexts,
        "warnings": warnings,
    }


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def preview_config_update(
    *,
    update: dict[str, Any],
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise ValueError("Provenance config update must be an object.")
    target = config_path(path, state_root=state_root)
    before_raw = base_config(read_config(target))
    after_raw = deep_merge(before_raw, update)
    after_raw["schema_version"] = after_raw.get("schema_version") or SCHEMA_VERSION
    validate_config(after_raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "preview_provenance_config_update",
        "config_path": str(target),
        "before": redacted_config(before_raw),
        "after": redacted_config(after_raw),
        "requires_approval_token": APPLY_APPROVAL_TOKEN,
        "will_write": False,
    }


def apply_config_update(
    *,
    update: dict[str, Any],
    approval_token: str,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    if approval_token != APPLY_APPROVAL_TOKEN:
        raise ValueError(f"Apply requires approval_token={APPLY_APPROVAL_TOKEN}.")
    target = config_path(path, state_root=state_root)
    before_raw = base_config(read_config(target))
    after_raw = deep_merge(before_raw, update)
    after_raw["schema_version"] = after_raw.get("schema_version") or SCHEMA_VERSION
    validate_config(after_raw)
    atomic_write_json(target, after_raw)
    audit_dir = after_raw.get("mutation_policy", {}).get("audit_dir") if isinstance(after_raw.get("mutation_policy"), dict) else ""
    if audit_dir:
        audit_root = Path(str(audit_dir)).expanduser()
        audit_path = audit_root / f"{safe_timestamp()}-provenance-config-apply.json"
        atomic_write_json(
            audit_path,
            {
                "schema_version": "transcribe-audio.provenance-config-apply.v1",
                "config_path": str(target),
                "before": redacted_config(before_raw),
                "after": redacted_config(after_raw),
            },
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "apply_provenance_config_update",
        "config_path": str(target),
        "before": redacted_config(before_raw),
        "after": redacted_config(after_raw),
        "will_write": True,
        "applied": True,
    }


def safe_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def doctor(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    target = config_path(path, state_root=state_root)
    raw = read_config(target)
    warnings: list[str] = []
    errors: list[str] = []
    if not raw:
        return {
            "schema_version": SCHEMA_VERSION,
            "config_path": str(target),
            "status": "missing",
            "valid": False,
            "errors": [f"Config file does not exist: {target}"],
            "warnings": [],
        }
    try:
        validation = validate_config(raw)
        warnings.extend(validation.get("warnings", []))
    except ValueError as exc:
        errors.append(str(exc))
    profile_name = active_profile_name(raw, profile)
    for source_id in workflow_source_ids(raw, "calendar_metadata", profile_name):
        source = enabled_source(raw, source_id)
        if not source or str(source.get("kind") or "") != SOURCE_KIND_ICAL:
            continue
        _url, warning = maybe_resolve_url(source_id, source)
        if warning:
            errors.append(warning)
    status = "ok" if not errors else "error"
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(target),
        "profile": profile_name,
        "status": status,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "redacted_config": redacted_config(raw),
        "calendar_metadata": resolve_calendar_metadata_settings(path=target, profile=profile_name).to_dict()
        if not errors
        else {},
    }


def all_config(
    *,
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    target = config_path(path, state_root=state_root)
    raw = read_config(target)
    profile_name = active_profile_name(raw, profile)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(target),
        "exists": bool(raw),
        "profile": profile_name,
        "config": redacted_config(raw) if raw else {},
        "calendar_metadata": resolve_calendar_metadata_settings(path=target, profile=profile_name).to_dict()
        if raw
        else {},
        "contact_source_config": redact_value(
            "contact_source_config",
            contact_source_config_from_provenance(path=target, profile=profile_name),
            set(DEFAULT_SENSITIVE_FIELDS),
        ),
    }
    return payload


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provenance-config", type=Path, help="Provenance config JSON. Defaults to user state.")
    parser.add_argument("--provenance-profile", help="Named provenance profile to resolve.")


def parse_json_arg(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON update: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Update JSON must be an object.")
    return payload


def update_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.update_file:
        payload = json.loads(args.update_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Update file must contain a JSON object.")
        return payload
    if args.update_json:
        return parse_json_arg(args.update_json)
    raise ValueError("Provide --update-json or --update-file.")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and mutate user-scoped provenance configuration.")
    parser.add_argument("--config", type=Path, help="Config path. Defaults to TRANSCRIPTS_PROVENANCE_CONFIG or user state.")
    parser.add_argument("--profile", help="Profile to resolve.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="Show redacted config and resolved settings.")
    init_sample = subparsers.add_parser("init-sample", help="Write the redacted sample config to the target path.")
    init_sample.add_argument("--force", action="store_true", help="Overwrite an existing config.")
    subparsers.add_parser("doctor", help="Validate config and required local secret refs.")
    for command_name in ("preview-update", "apply-update"):
        update_parser = subparsers.add_parser(command_name, help=f"{command_name.replace('-', ' ').title()}.")
        update_parser.add_argument("--update-json", help="JSON object to deep-merge into the config.")
        update_parser.add_argument("--update-file", type=Path, help="Path to a JSON object to deep-merge into the config.")
        if command_name == "apply-update":
            update_parser.add_argument("--approval-token", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "show":
            payload = all_config(path=args.config, profile=args.profile)
        elif args.command == "doctor":
            payload = doctor(path=args.config, profile=args.profile)
        elif args.command == "init-sample":
            path = write_sample_config(args.config, force=args.force)
            payload = {"schema_version": SCHEMA_VERSION, "path": str(path), "created": True}
        elif args.command == "preview-update":
            payload = preview_config_update(update=update_payload_from_args(args), path=args.config)
        elif args.command == "apply-update":
            payload = apply_config_update(
                update=update_payload_from_args(args),
                approval_token=args.approval_token,
                path=args.config,
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
