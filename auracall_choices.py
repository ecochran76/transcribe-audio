#!/usr/bin/env python3
"""
Redacted AuraCall agent-choice readiness for transcript workflows.
"""
from __future__ import annotations

from typing import Any

import requests


DEFAULT_CHOICES_TIMEOUT_SECONDS = 5.0


def resolve_agent_id(model: str) -> str | None:
    value = (model or "").strip()
    if value.startswith("agent:"):
        agent_id = value.removeprefix("agent:").strip()
        return agent_id or None
    return None


def resolve_choices_url(base_url: str, env: dict[str, str]) -> str:
    configured = (env.get("AURACALL_AGENT_CHOICES_URL") or env.get("AURACALL_CHOICES_URL") or "").strip()
    if configured:
        return configured.rstrip("/")
    value = (base_url or "").rstrip("/")
    if value.endswith("/v1"):
        return f"{value}/config/agent-choices"
    return f"{value}/v1/config/agent-choices"


def build_readiness(
    *,
    env: dict[str, str],
    base_url: str,
    api_key: str | None,
    model: str,
    dispatch_team: str | None,
    choices_payload: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    selected_agent_id = resolve_agent_id(model)
    choices_url = resolve_choices_url(base_url, env) if base_url else ""
    agents = _object_list(choices_payload, "agents")
    teams = _object_list(choices_payload, "teams")
    bindings = _object_list(choices_payload, "bindings")
    validations = _object_list(_object(choices_payload, "validation"), "agents")
    agent = _find_by_id(agents, selected_agent_id) if selected_agent_id else None
    team = _find_by_id(teams, dispatch_team) if dispatch_team else None
    binding = _find_by_key(bindings, "bindingKey", str(agent.get("bindingKey") or "")) if agent else None
    validation = _find_by_key(validations, "agentId", selected_agent_id or "") if selected_agent_id else None
    member_summaries = _team_member_summaries(team, bindings)
    warnings: list[str] = []

    if error:
        warnings.append("AuraCall agent choices were not readable.")
    if dispatch_team and choices_payload is not None and not teams:
        warnings.append("AuraCall agent choices did not include teams; dispatch-pool membership cannot be proven.")
    if dispatch_team and choices_payload is not None and not team:
        warnings.append(f"AuraCall dispatch team {dispatch_team} was not found in agent choices.")
    if selected_agent_id and choices_payload is not None and not agent:
        warnings.append(f"AuraCall agent {selected_agent_id} was not found in agent choices.")
    if agent and not (binding or {}).get("ready"):
        warnings.append(f"AuraCall agent {selected_agent_id} does not have a ready browser binding.")
    if validation and validation.get("valid") is False:
        warnings.append(f"AuraCall agent {selected_agent_id} failed agent-choice validation.")
    if dispatch_team and team and any(not member.get("ready") for member in member_summaries):
        warnings.append(f"AuraCall dispatch team {dispatch_team} has one or more not-ready members.")

    selected_agent = None
    if selected_agent_id:
        selected_agent = {
            "id": selected_agent_id,
            "exists": agent is not None,
            "valid": None if validation is None else bool(validation.get("valid")),
            "ready": bool(agent and binding and binding.get("ready") and (validation is None or validation.get("valid") is not False)),
            "runtimeProfileId": _string_or_none(agent.get("runtimeProfileId")) if agent else None,
            "browserProfileId": _string_or_none(agent.get("browserProfileId")) if agent else None,
            "bindingKey": _string_or_none(agent.get("bindingKey")) if agent else None,
            "projectBinding": _redacted_project_binding(_object(agent, "projectBinding")) if agent else None,
        }

    dispatch_summary = None
    if dispatch_team:
        dispatch_summary = {
            "id": dispatch_team,
            "exists": team is not None,
            "type": _string_or_none(team.get("type")) if team else None,
            "agentIds": [str(value) for value in (team.get("agentIds") or [])] if team else [],
            "members": member_summaries,
            "ready": bool(team and member_summaries and all(member.get("ready") for member in member_summaries)),
        }

    ok_subjects: list[bool] = []
    if selected_agent is not None:
        ok_subjects.append(bool(selected_agent["ready"]))
    if dispatch_summary is not None:
        ok_subjects.append(bool(dispatch_summary["ready"]))

    return {
        "schema_version": "transcribe-audio.auracall-choices-readiness.v1",
        "source": {
            "choices_url": choices_url,
            "base_url_configured": bool(base_url),
            "api_key_configured": bool(api_key),
            "fetched": choices_payload is not None,
        },
        "selected_model": model,
        "selected_agent_id": selected_agent_id,
        "dispatch_team": dispatch_team,
        "selected_agent": selected_agent,
        "dispatch": dispatch_summary,
        "counts": {
            "agents": len(agents),
            "teams": len(teams),
            "bindings": len(bindings),
        },
        "links": {
            "agent_choices": "/v1/config/agent-choices",
            "response_batches": "/v1/response-batches",
            "response_batch_status": "/v1/response-batches/{batch_id}",
        },
        "ok": bool(ok_subjects and all(ok_subjects) and not warnings),
        "warnings": warnings,
        "error": error,
    }


def read_choices_readiness(
    *,
    env: dict[str, str],
    base_url: str,
    api_key: str | None,
    model: str,
    dispatch_team: str | None,
    timeout: float = DEFAULT_CHOICES_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    choices_url = resolve_choices_url(base_url, env)
    try:
        response = requests.get(
            choices_url,
            headers={"authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )
        if response.status_code >= 400:
            return build_readiness(
                env=env,
                base_url=base_url,
                api_key=api_key,
                model=model,
                dispatch_team=dispatch_team,
                choices_payload=None,
                error=f"HTTP {response.status_code}",
            )
        payload = response.json()
        if not isinstance(payload, dict):
            return build_readiness(
                env=env,
                base_url=base_url,
                api_key=api_key,
                model=model,
                dispatch_team=dispatch_team,
                choices_payload=None,
                error="AuraCall agent choices response was not a JSON object.",
            )
        return build_readiness(
            env=env,
            base_url=base_url,
            api_key=api_key,
            model=model,
            dispatch_team=dispatch_team,
            choices_payload=payload,
        )
    except (requests.RequestException, ValueError) as exc:
        return build_readiness(
            env=env,
            base_url=base_url,
            api_key=api_key,
            model=model,
            dispatch_team=dispatch_team,
            choices_payload=None,
            error=str(exc),
        )


def _team_member_summaries(team: dict[str, Any] | None, bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not team:
        return []
    members = team.get("members") if isinstance(team.get("members"), list) else []
    summaries: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, dict):
            continue
        binding = _member_binding(member, bindings)
        summaries.append(
            {
                "agentId": _string_or_none(member.get("agentId")),
                "exists": bool(member.get("exists")),
                "ready": bool(member.get("exists") and binding and binding.get("ready")),
                "runtimeProfileId": _string_or_none(member.get("runtimeProfileId")),
                "browserProfileId": _string_or_none(member.get("browserProfileId")),
                "bindingKey": _string_or_none(member.get("bindingKey")),
            }
        )
    return summaries


def _member_binding(member: dict[str, Any], bindings: list[dict[str, Any]]) -> dict[str, Any] | None:
    binding_key = str(member.get("bindingKey") or "")
    if binding_key:
        return _find_by_key(bindings, "bindingKey", binding_key)
    runtime_profile_id = _string_or_none(member.get("runtimeProfileId"))
    browser_profile_id = _string_or_none(member.get("browserProfileId"))
    service = _string_or_none(member.get("service") or member.get("defaultService"))
    for binding in bindings:
        if runtime_profile_id and binding.get("runtimeProfileId") != runtime_profile_id:
            continue
        if browser_profile_id and binding.get("browserProfileId") != browser_profile_id:
            continue
        if service and binding.get("service") != service:
            continue
        return binding
    return None


def _object(value: dict[str, Any] | None, key: str) -> dict[str, Any]:
    child = value.get(key) if isinstance(value, dict) else None
    return child if isinstance(child, dict) else {}


def _object_list(value: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    child = value.get(key) if isinstance(value, dict) else None
    return [item for item in child if isinstance(item, dict)] if isinstance(child, list) else []


def _find_by_id(items: list[dict[str, Any]], item_id: str | None) -> dict[str, Any] | None:
    if not item_id:
        return None
    return _find_by_key(items, "id", item_id)


def _find_by_key(items: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    if not value:
        return None
    for item in items:
        if item.get(key) == value:
            return item
    return None


def _redacted_project_binding(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": _string_or_none(value.get("mode")),
        "source": _string_or_none(value.get("source")),
        "id": _string_or_none(value.get("id")),
        "providerProjectId": _string_or_none(value.get("providerProjectId")),
        "label": _string_or_none(value.get("label")),
    }


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
