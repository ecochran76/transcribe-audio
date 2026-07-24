#!/usr/bin/env python3
"""
Central task-based intelligence provider configuration.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_CONFIG_PATH = DEFAULT_STATE_DIR / "intelligence.config.json"
ENV_CONFIG_PATH = "TRANSCRIPTS_INTELLIGENCE_CONFIG"
SCHEMA_VERSION = "transcribe-audio.intelligence-config.v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CODEX_APP_MODEL = "gpt-5.6-sol"
APPLY_APPROVAL_TOKEN = "APPLY_INTELLIGENCE_CONFIG_UPDATE"
PROFILE_ROUTE_FIELDS = {"provider", "model", "base_url", "timeout", "temperature"}
PROFILE_FIELDS = {"label", "description", *PROFILE_ROUTE_FIELDS}

TASK_FIRST_PASS_SUMMARY = "first_pass_summary"
TASK_CONTEXTUAL_REREAD = "contextual_reread"
TASK_CONTEXT_SOURCE_RANKING = "context_source_ranking"
TASK_ROUTE_SELECTION = "route_selection"
TASK_SPEAKER_DISAMBIGUATION = "speaker_disambiguation"
TASK_MEMORY_HARVEST_REVIEW = "memory_harvest_review"
TASK_EMBEDDING = "embedding"
TASK_APP_SUPERVISOR = "app_supervisor"

TASK_IDS = (
    TASK_FIRST_PASS_SUMMARY,
    TASK_CONTEXTUAL_REREAD,
    TASK_CONTEXT_SOURCE_RANKING,
    TASK_ROUTE_SELECTION,
    TASK_SPEAKER_DISAMBIGUATION,
    TASK_MEMORY_HARVEST_REVIEW,
    TASK_EMBEDDING,
    TASK_APP_SUPERVISOR,
)

DEFAULT_TASKS: dict[str, dict[str, Any]] = {
    TASK_FIRST_PASS_SUMMARY: {
        "provider": "openai-compatible",
        "model": DEFAULT_OPENAI_MODEL,
        "timeout": 120.0,
        "temperature": 0.1,
        "fallbacks": ["codex-exec"],
        "requires_ledger": False,
        "human_review": "on_warning",
    },
    TASK_CONTEXTUAL_REREAD: {
        "provider": "openai-compatible",
        "model": DEFAULT_OPENAI_MODEL,
        "timeout": 120.0,
        "temperature": 0.1,
        "fallbacks": ["codex-exec"],
        "requires_ledger": False,
        "human_review": "on_warning",
    },
    TASK_CONTEXT_SOURCE_RANKING: {
        "provider": "codex-app-server",
        "model": "",
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["openai-compatible", "codex-exec"],
        "requires_ledger": True,
        "human_review": "on_low_confidence",
    },
    TASK_ROUTE_SELECTION: {
        "provider": "codex-app-server",
        "model": "",
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["openai-compatible"],
        "requires_ledger": True,
        "human_review": "on_low_confidence",
    },
    TASK_SPEAKER_DISAMBIGUATION: {
        "provider": "codex-app-server",
        "model": DEFAULT_CODEX_APP_MODEL,
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["openai-compatible"],
        "requires_ledger": True,
        "human_review": "on_low_confidence",
    },
    TASK_MEMORY_HARVEST_REVIEW: {
        "provider": "codex-app-server",
        "model": "",
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["openai-compatible"],
        "requires_ledger": True,
        "human_review": "required",
    },
    TASK_EMBEDDING: {
        "provider": "ollama",
        "model": "ollama/nomic-embed-text",
        "timeout": 60.0,
        "temperature": 0.0,
        "fallbacks": [],
        "requires_ledger": False,
        "human_review": "never",
    },
    TASK_APP_SUPERVISOR: {
        "provider": "codex-app-server",
        "model": DEFAULT_CODEX_APP_MODEL,
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["codex-exec"],
        "requires_ledger": True,
        "human_review": "phase_policy",
    },
}

DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "openai_readout": {
        "label": "OpenAI readout",
        "description": "General transcript summarization and contextual readout profile.",
        "provider": "openai-compatible",
        "model": DEFAULT_OPENAI_MODEL,
        "base_url": "",
        "timeout": 120.0,
        "temperature": 0.1,
    },
    "codex_supervisor": {
        "label": "Codex supervisor",
        "description": "Ledger-backed App Intelligence supervisor profile.",
        "provider": "codex-app-server",
        "model": DEFAULT_CODEX_APP_MODEL,
        "base_url": "",
        "timeout": 120.0,
        "temperature": 0.0,
    },
    "local_embedding": {
        "label": "Local embeddings",
        "description": "Local semantic search embedding profile.",
        "provider": "ollama",
        "model": "ollama/nomic-embed-text",
        "base_url": "",
        "timeout": 60.0,
        "temperature": 0.0,
    },
}

DEFAULT_TASK_PROFILES: dict[str, str] = {
    TASK_FIRST_PASS_SUMMARY: "openai_readout",
    TASK_CONTEXTUAL_REREAD: "openai_readout",
    TASK_CONTEXT_SOURCE_RANKING: "codex_supervisor",
    TASK_ROUTE_SELECTION: "codex_supervisor",
    TASK_SPEAKER_DISAMBIGUATION: "codex_supervisor",
    TASK_MEMORY_HARVEST_REVIEW: "codex_supervisor",
    TASK_EMBEDDING: "local_embedding",
    TASK_APP_SUPERVISOR: "codex_supervisor",
}


@dataclass(frozen=True)
class IntelligenceTaskConfig:
    task: str
    provider: str
    model: str = ""
    profile: str = ""
    base_url: str = ""
    timeout: float = 120.0
    temperature: float = 0.0
    fallbacks: list[str] = field(default_factory=list)
    requires_ledger: bool = False
    human_review: str = "on_warning"
    source: str = "defaults"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_path(path: Optional[Path] = None) -> Path:
    if path:
        return path.expanduser()
    env_value = os.getenv(ENV_CONFIG_PATH)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_CONFIG_PATH.expanduser()


def read_config(path: Optional[Path] = None) -> dict[str, Any]:
    target = config_path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Intelligence config {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Intelligence config {target} must contain a JSON object.")
    return payload


def write_sample_config(path: Optional[Path] = None) -> Path:
    target = config_path(path)
    task_policy = {
        task: {key: value for key, value in config.items() if key not in PROFILE_ROUTE_FIELDS}
        for task, config in DEFAULT_TASKS.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profiles": DEFAULT_PROFILES,
        "task_profiles": DEFAULT_TASK_PROFILES,
        "tasks": task_policy,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _base_config(raw: dict[str, Any]) -> dict[str, Any]:
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    task_profiles = raw.get("task_profiles") if isinstance(raw.get("task_profiles"), dict) else {}
    tasks = raw.get("tasks") if isinstance(raw.get("tasks"), dict) else {}
    return {
        "schema_version": raw.get("schema_version") or SCHEMA_VERSION,
        "profiles": {**copy.deepcopy(DEFAULT_PROFILES), **copy.deepcopy(profiles)},
        "task_profiles": {**copy.deepcopy(DEFAULT_TASK_PROFILES), **copy.deepcopy(task_profiles)},
        "tasks": copy.deepcopy(tasks),
    }


def _task_payload(raw: dict[str, Any], task: str) -> dict[str, Any]:
    tasks = raw.get("tasks") if isinstance(raw.get("tasks"), dict) else {}
    payload = tasks.get(task) if isinstance(tasks.get(task), dict) else {}
    return payload


def _profile_payload(raw: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    payload = profiles.get(profile_id) if isinstance(profiles.get(profile_id), dict) else {}
    if payload:
        return payload
    default = DEFAULT_PROFILES.get(profile_id)
    return copy.deepcopy(default) if isinstance(default, dict) else {}


def _task_profile(raw: dict[str, Any], task: str) -> str:
    task_profiles = raw.get("task_profiles") if isinstance(raw.get("task_profiles"), dict) else {}
    value = task_profiles.get(task)
    if isinstance(value, str) and value:
        return value
    return DEFAULT_TASK_PROFILES.get(task, "")


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _validate_task_update(task: str, update: dict[str, Any]) -> dict[str, Any]:
    if task not in TASK_IDS:
        raise ValueError(f"Unknown intelligence task: {task}")
    if not isinstance(update, dict):
        raise ValueError("Task update must be an object.")
    allowed = {"profile", "provider", "model", "base_url", "timeout", "temperature", "fallbacks", "requires_ledger", "human_review"}
    unknown = sorted(set(update) - allowed)
    if unknown:
        raise ValueError(f"Unknown task config field(s): {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    for key, value in update.items():
        if key in {"profile", "provider", "model", "base_url", "human_review"}:
            normalized[key] = str(value or "")
        elif key in {"timeout", "temperature"}:
            normalized[key] = _coerce_float(value, float(DEFAULT_TASKS[task].get(key) or 0.0))
        elif key == "requires_ledger":
            normalized[key] = _coerce_bool(value, bool(DEFAULT_TASKS[task].get(key)))
        elif key == "fallbacks":
            if not isinstance(value, list):
                raise ValueError("fallbacks must be a list.")
            normalized[key] = [str(item) for item in value if str(item)]
    if "provider" in normalized and not normalized["provider"]:
        raise ValueError("provider cannot be empty.")
    if "profile" in normalized and not normalized["profile"]:
        raise ValueError("profile cannot be empty.")
    return normalized


def _validate_profile_update(profile_id: str, update: dict[str, Any]) -> dict[str, Any]:
    if not profile_id:
        raise ValueError("Missing required profile id.")
    if not isinstance(update, dict):
        raise ValueError("Profile update must be an object.")
    unknown = sorted(set(update) - PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown profile config field(s): {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    for key, value in update.items():
        if key in {"label", "description", "provider", "model", "base_url"}:
            normalized[key] = str(value or "")
        elif key in {"timeout", "temperature"}:
            default = _profile_payload({}, profile_id).get(key, 0.0)
            normalized[key] = _coerce_float(value, float(default or 0.0))
    if "provider" in normalized and not normalized["provider"]:
        raise ValueError("profile provider cannot be empty.")
    return normalized


def preview_config_update(
    *,
    task: str = "",
    update: Optional[dict[str, Any]] = None,
    profile_id: str = "",
    profile_update: Optional[dict[str, Any]] = None,
    delete_profile: bool = False,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    target = config_path(path)
    raw = read_config(target)
    before = _base_config(raw)
    after = copy.deepcopy(before)
    rollback: dict[str, Any] = {}
    normalized_update: dict[str, Any] = {}
    normalized_profile_update: dict[str, Any] = {}
    if delete_profile:
        if not profile_id:
            raise ValueError("Missing required profile id.")
        if profile_id in DEFAULT_PROFILES:
            raise ValueError("Default intelligence profiles cannot be deleted.")
        if profile_id not in before.get("profiles", {}):
            raise ValueError(f"Unknown intelligence profile: {profile_id}")
        referenced_tasks = [
            task_id
            for task_id, assigned_profile in after.get("task_profiles", {}).items()
            if assigned_profile == profile_id
        ]
        if referenced_tasks:
            raise ValueError(f"Cannot delete profile {profile_id}; it is assigned to: {', '.join(sorted(referenced_tasks))}")
        profiles = after.setdefault("profiles", {})
        profiles.pop(profile_id, None)
        rollback["profile"] = {
            "profile_id": profile_id,
            "previous_profile_config": copy.deepcopy(before.get("profiles", {}).get(profile_id, {})),
            "delete_profile": False,
            "restore_deleted_profile": True,
        }
    elif profile_id or profile_update:
        normalized_profile_update = _validate_profile_update(profile_id, profile_update or {})
        profiles = after.setdefault("profiles", {})
        current_profile = profiles.get(profile_id) if isinstance(profiles.get(profile_id), dict) else {}
        profiles[profile_id] = {**current_profile, **normalized_profile_update}
        rollback["profile"] = {
            "profile_id": profile_id,
            "previous_profile_config": copy.deepcopy(before.get("profiles", {}).get(profile_id, {})),
            "delete_profile": profile_id not in before.get("profiles", {}),
        }
    if task or update:
        normalized_update = _validate_task_update(task, update or {})
        tasks = after.setdefault("tasks", {})
        task_profiles = after.setdefault("task_profiles", {})
        task_payload = tasks.get(task) if isinstance(tasks.get(task), dict) else {}
        next_task_payload = {**task_payload}
        selected_profile = normalized_update.pop("profile", "")
        if selected_profile:
            if selected_profile not in after.get("profiles", {}) and selected_profile not in DEFAULT_PROFILES:
                raise ValueError(f"Unknown intelligence profile: {selected_profile}")
            task_profiles[task] = selected_profile
            for field in PROFILE_ROUTE_FIELDS:
                next_task_payload.pop(field, None)
        next_task_payload.update(normalized_update)
        tasks[task] = next_task_payload
        rollback.update(
            {
                "task": task,
                "previous_task_config": copy.deepcopy(task_payload),
                "delete_task": task not in before.get("tasks", {}),
            }
        )
        rollback["task_update"] = {
            "task": task,
            "previous_task_config": copy.deepcopy(task_payload),
            "previous_profile": before.get("task_profiles", {}).get(task),
            "delete_task": task not in before.get("tasks", {}),
        }
    if not rollback:
        raise ValueError("No intelligence config update was provided.")
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "preview_intelligence_config_update",
        "config_path": str(target),
        "task": task,
        "profile_id": profile_id,
        "update": normalized_update,
        "profile_update": normalized_profile_update,
        "delete_profile": delete_profile,
        "before": before,
        "after": after,
        "resolved_before": resolve_task_config(task, path=target).to_dict() if task else None,
        "resolved_after": resolve_task_config_from_raw(task, after).to_dict() if task else None,
        "rollback": rollback,
        "requires_approval_token": APPLY_APPROVAL_TOKEN,
        "will_write": False,
    }


def apply_config_update(
    *,
    task: str = "",
    update: Optional[dict[str, Any]] = None,
    profile_id: str = "",
    profile_update: Optional[dict[str, Any]] = None,
    delete_profile: bool = False,
    approval_token: str,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    if approval_token != APPLY_APPROVAL_TOKEN:
        raise ValueError(f"Apply requires approval_token={APPLY_APPROVAL_TOKEN}.")
    preview = preview_config_update(
        task=task,
        update=update or {},
        profile_id=profile_id,
        profile_update=profile_update or {},
        delete_profile=delete_profile,
        path=path,
    )
    target = Path(preview["config_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(preview["after"], indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    resolved = resolve_task_config(task, path=target) if task else None
    return {
        **preview,
        "action": "apply_intelligence_config_update",
        "will_write": True,
        "applied": True,
        "resolved_after": resolved.to_dict() if resolved else None,
    }


def resolve_task_config_from_raw(task: str, raw: dict[str, Any]) -> IntelligenceTaskConfig:
    if task not in TASK_IDS:
        raise ValueError(f"Unknown intelligence task: {task}")
    defaults = DEFAULT_TASKS[task]
    profile_id = _task_profile(raw, task)
    profile_payload = {
        key: value
        for key, value in _profile_payload(raw, profile_id).items()
        if key in PROFILE_ROUTE_FIELDS and value not in (None, "")
    }
    task_payload = _task_payload(raw, task)
    task_config = {**defaults, **profile_payload, **task_payload}
    fallbacks = task_config.get("fallbacks") if isinstance(task_config.get("fallbacks"), list) else []
    return IntelligenceTaskConfig(
        task=task,
        provider=str(task_config.get("provider") or defaults["provider"]),
        model=str(task_config.get("model") or ""),
        profile=profile_id,
        base_url=str(task_config.get("base_url") or ""),
        timeout=_coerce_float(task_config.get("timeout"), float(defaults.get("timeout") or 120.0)),
        temperature=_coerce_float(task_config.get("temperature"), float(defaults.get("temperature") or 0.0)),
        fallbacks=[str(item) for item in fallbacks if str(item)],
        requires_ledger=_coerce_bool(task_config.get("requires_ledger"), bool(defaults.get("requires_ledger"))),
        human_review=str(task_config.get("human_review") or defaults.get("human_review") or "on_warning"),
        source="defaults",
    )


def resolve_task_config(
    task: str,
    *,
    path: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> IntelligenceTaskConfig:
    if task not in TASK_IDS:
        raise ValueError(f"Unknown intelligence task: {task}")
    raw = read_config(path)
    defaults = DEFAULT_TASKS[task]
    profile_id = _task_profile(raw, task)
    profile_payload = {
        key: value
        for key, value in _profile_payload(raw, profile_id).items()
        if key in PROFILE_ROUTE_FIELDS and value not in (None, "")
    }
    task_config = {**defaults, **profile_payload, **_task_payload(raw, task)}
    has_file_task = bool(_task_payload(raw, task))
    has_file_profile = profile_id in (raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {})
    has_file_assignment = task in (raw.get("task_profiles") if isinstance(raw.get("task_profiles"), dict) else {})
    source = str(config_path(path)) if (has_file_task or has_file_profile or has_file_assignment) else "defaults"
    env_prefix = f"TRANSCRIPTS_INTELLIGENCE_{task.upper()}_"
    env_overrides = {
        "provider": os.getenv(env_prefix + "PROVIDER"),
        "model": os.getenv(env_prefix + "MODEL"),
        "base_url": os.getenv(env_prefix + "BASE_URL"),
        "timeout": os.getenv(env_prefix + "TIMEOUT"),
        "temperature": os.getenv(env_prefix + "TEMPERATURE"),
    }
    for key, value in env_overrides.items():
        if value not in (None, ""):
            task_config[key] = value
            source = "environment"
    for key, value in (overrides or {}).items():
        if value not in (None, ""):
            task_config[key] = value
            source = "override"

    fallbacks = task_config.get("fallbacks") if isinstance(task_config.get("fallbacks"), list) else []
    return IntelligenceTaskConfig(
        task=task,
        provider=str(task_config.get("provider") or defaults["provider"]),
        model=str(task_config.get("model") or ""),
        profile=profile_id,
        base_url=str(task_config.get("base_url") or ""),
        timeout=_coerce_float(task_config.get("timeout"), float(defaults.get("timeout") or 120.0)),
        temperature=_coerce_float(task_config.get("temperature"), float(defaults.get("temperature") or 0.0)),
        fallbacks=[str(item) for item in fallbacks if str(item)],
        requires_ledger=_coerce_bool(task_config.get("requires_ledger"), bool(defaults.get("requires_ledger"))),
        human_review=str(task_config.get("human_review") or defaults.get("human_review") or "on_warning"),
        source=source,
    )


def all_task_configs(*, path: Optional[Path] = None) -> dict[str, Any]:
    raw = read_config(path)
    base = _base_config(raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path(path)),
        "profiles": base["profiles"],
        "default_profile_ids": sorted(DEFAULT_PROFILES),
        "task_profiles": base["task_profiles"],
        "tasks": {task: resolve_task_config(task, path=path).to_dict() for task in TASK_IDS},
    }


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--intelligence-config", type=Path, help="Task routing config JSON. Defaults to user state.")


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "timeout": getattr(args, "timeout", None),
        "temperature": getattr(args, "temperature", None),
    }


def apply_task_config(args: argparse.Namespace, task: str) -> IntelligenceTaskConfig:
    resolved = resolve_task_config(
        task,
        path=getattr(args, "intelligence_config", None),
        overrides=overrides_from_args(args),
    )
    args.provider = resolved.provider
    args.model = resolved.model
    args.base_url = resolved.base_url or getattr(args, "base_url", None)
    args.timeout = resolved.timeout
    args.temperature = resolved.temperature
    args.intelligence_task = task
    args.intelligence_config_source = resolved.source
    return resolved


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect transcript intelligence task configuration.")
    parser.add_argument("--config", type=Path, help="Config path. Defaults to TRANSCRIPTS_INTELLIGENCE_CONFIG or user state.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    show = subparsers.add_parser("show", help="Show all resolved task configs.")
    show.add_argument("--task", choices=TASK_IDS)
    subparsers.add_parser("init-sample", help="Write a sample config with default task routing.")
    update = subparsers.add_parser("preview-update", help="Preview a task config update.")
    update.add_argument("--task", choices=TASK_IDS, required=True)
    update.add_argument("--provider")
    update.add_argument("--model")
    update.add_argument("--base-url")
    update.add_argument("--timeout", type=float)
    update.add_argument("--temperature", type=float)
    update.add_argument("--fallback", action="append", dest="fallbacks")
    update.add_argument("--requires-ledger", choices=("true", "false"))
    update.add_argument("--human-review")

    apply_update = subparsers.add_parser("apply-update", help="Apply a task config update.")
    apply_update.add_argument("--task", choices=TASK_IDS, required=True)
    apply_update.add_argument("--approval-token", required=True)
    apply_update.add_argument("--provider")
    apply_update.add_argument("--model")
    apply_update.add_argument("--base-url")
    apply_update.add_argument("--timeout", type=float)
    apply_update.add_argument("--temperature", type=float)
    apply_update.add_argument("--fallback", action="append", dest="fallbacks")
    apply_update.add_argument("--requires-ledger", choices=("true", "false"))
    apply_update.add_argument("--human-review")
    return parser.parse_args(argv)


def update_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("provider", "model", "base_url", "timeout", "temperature", "fallbacks", "human_review"):
        value = getattr(args, key, None)
        if value not in (None, ""):
            payload[key] = value
    if getattr(args, "requires_ledger", None) is not None:
        payload["requires_ledger"] = args.requires_ledger == "true"
    return payload


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "init-sample":
            path = write_sample_config(args.config)
            payload: dict[str, Any] = {"path": str(path), "created": True}
        elif args.command == "show":
            if args.task:
                payload = resolve_task_config(args.task, path=args.config).to_dict()
            else:
                payload = all_task_configs(path=args.config)
        elif args.command == "preview-update":
            payload = preview_config_update(task=args.task, update=update_from_args(args), path=args.config)
        elif args.command == "apply-update":
            payload = apply_config_update(
                task=args.task,
                update=update_from_args(args),
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
