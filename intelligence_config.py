#!/usr/bin/env python3
"""
Central task-based intelligence provider configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from transcribe_common import DEFAULT_OPENAI_MODEL

DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_CONFIG_PATH = DEFAULT_STATE_DIR / "intelligence.config.json"
ENV_CONFIG_PATH = "TRANSCRIPTS_INTELLIGENCE_CONFIG"
SCHEMA_VERSION = "transcribe-audio.intelligence-config.v1"

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
        "model": "",
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
        "model": "",
        "timeout": 120.0,
        "temperature": 0.0,
        "fallbacks": ["codex-exec"],
        "requires_ledger": True,
        "human_review": "phase_policy",
    },
}


@dataclass(frozen=True)
class IntelligenceTaskConfig:
    task: str
    provider: str
    model: str = ""
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
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tasks": DEFAULT_TASKS,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _task_payload(raw: dict[str, Any], task: str) -> dict[str, Any]:
    tasks = raw.get("tasks") if isinstance(raw.get("tasks"), dict) else {}
    payload = tasks.get(task) if isinstance(tasks.get(task), dict) else {}
    return payload


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
    task_config = {**defaults, **_task_payload(raw, task)}
    source = str(config_path(path)) if _task_payload(raw, task) else "defaults"
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
        base_url=str(task_config.get("base_url") or ""),
        timeout=_coerce_float(task_config.get("timeout"), float(defaults.get("timeout") or 120.0)),
        temperature=_coerce_float(task_config.get("temperature"), float(defaults.get("temperature") or 0.0)),
        fallbacks=[str(item) for item in fallbacks if str(item)],
        requires_ledger=_coerce_bool(task_config.get("requires_ledger"), bool(defaults.get("requires_ledger"))),
        human_review=str(task_config.get("human_review") or defaults.get("human_review") or "on_warning"),
        source=source,
    )


def all_task_configs(*, path: Optional[Path] = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path(path)),
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
    return parser.parse_args(argv)


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
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
