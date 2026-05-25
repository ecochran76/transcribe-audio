#!/usr/bin/env python3
"""
User-scoped workflow automation policy.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_CONFIG_PATH = DEFAULT_STATE_DIR / "automation.config.json"
ENV_CONFIG_PATH = "TRANSCRIPTS_AUTOMATION_CONFIG"
SCHEMA_VERSION = "transcribe-audio.automation-config.v1"
APPLY_APPROVAL_TOKEN = "APPLY_AUTOMATION_CONFIG_UPDATE"

MODE_MANUAL = "manual"
MODE_ONE_CLICK = "one_click"
MODE_AUTOMATIC = "automatic"
MODE_CHOICES = (MODE_MANUAL, MODE_ONE_CLICK, MODE_AUTOMATIC)

STAGE_INGEST_AUDIO = "ingest_audio"
STAGE_TRANSCRIBE_AUDIO = "transcribe_audio"
STAGE_INITIAL_SUMMARY = "initial_summary"
STAGE_SPEAKER_IDENTITY = "speaker_identity"
STAGE_CONTEXT_COLLECTION = "context_collection"
STAGE_FINAL_READOUT = "final_readout"

STAGE_IDS = (
    STAGE_INGEST_AUDIO,
    STAGE_TRANSCRIBE_AUDIO,
    STAGE_INITIAL_SUMMARY,
    STAGE_SPEAKER_IDENTITY,
    STAGE_CONTEXT_COLLECTION,
    STAGE_FINAL_READOUT,
)

STAGE_LABELS = {
    STAGE_INGEST_AUDIO: "Ingest audio",
    STAGE_TRANSCRIBE_AUDIO: "Transcribe audio",
    STAGE_INITIAL_SUMMARY: "Initial summary",
    STAGE_SPEAKER_IDENTITY: "Speaker identity",
    STAGE_CONTEXT_COLLECTION: "Context collection",
    STAGE_FINAL_READOUT: "Final readout",
}

STAGE_CAPABILITIES = {
    STAGE_INGEST_AUDIO: {"manual_available": True, "one_click_available": False, "automatic_available": False},
    STAGE_TRANSCRIBE_AUDIO: {"manual_available": True, "one_click_available": False, "automatic_available": False},
    STAGE_INITIAL_SUMMARY: {"manual_available": True, "one_click_available": True, "automatic_available": False},
    STAGE_SPEAKER_IDENTITY: {"manual_available": True, "one_click_available": False, "automatic_available": False},
    STAGE_CONTEXT_COLLECTION: {"manual_available": True, "one_click_available": False, "automatic_available": False},
    STAGE_FINAL_READOUT: {"manual_available": True, "one_click_available": False, "automatic_available": False},
}

DEFAULT_STAGE = {
    "enabled": False,
    "mode": MODE_MANUAL,
    "requires_review": True,
    "notes": "",
}


def default_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": "default",
        "stages": {stage: copy.deepcopy(DEFAULT_STAGE) for stage in STAGE_IDS},
    }


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
        raise ValueError(f"Automation config {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Automation config {target} must contain a JSON object.")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _normalize_stage(stage_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if stage_id not in STAGE_IDS:
        raise ValueError(f"Unknown automation stage: {stage_id}")
    if not isinstance(payload, dict):
        raise ValueError(f"Automation stage {stage_id} must be an object.")
    allowed = {"enabled", "mode", "requires_review", "notes"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown automation stage field(s) for {stage_id}: {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    if "enabled" in payload:
        normalized["enabled"] = _coerce_bool(payload.get("enabled"), False)
    if "mode" in payload:
        mode = str(payload.get("mode") or MODE_MANUAL)
        if mode not in MODE_CHOICES:
            raise ValueError(f"Automation stage {stage_id} mode must be one of: {', '.join(MODE_CHOICES)}")
        normalized["mode"] = mode
    if "requires_review" in payload:
        normalized["requires_review"] = _coerce_bool(payload.get("requires_review"), True)
    if "notes" in payload:
        normalized["notes"] = str(payload.get("notes") or "")
    return normalized


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Automation config must be a JSON object.")
    schema = raw.get("schema_version")
    if schema not in (None, SCHEMA_VERSION):
        raise ValueError(f"Unsupported automation schema_version: {schema}")
    stages = raw.get("stages") if isinstance(raw.get("stages"), dict) else {}
    for stage_id, payload in stages.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Automation stage {stage_id} must be an object.")
        _normalize_stage(str(stage_id), payload)
    return {"valid": True, "warnings": []}


def merged_config(raw: dict[str, Any]) -> dict[str, Any]:
    validate_config(raw)
    merged = default_config()
    if raw.get("profile"):
        merged["profile"] = str(raw.get("profile") or "default")
    stages = raw.get("stages") if isinstance(raw.get("stages"), dict) else {}
    for stage_id in STAGE_IDS:
        payload = stages.get(stage_id) if isinstance(stages.get(stage_id), dict) else {}
        merged["stages"][stage_id] = {
            **copy.deepcopy(DEFAULT_STAGE),
            **_normalize_stage(stage_id, payload),
            "label": STAGE_LABELS[stage_id],
            "capabilities": copy.deepcopy(STAGE_CAPABILITIES[stage_id]),
        }
    return merged


def _validate_update(update: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise ValueError("Automation config update must be an object.")
    allowed = {"profile", "stages"}
    unknown = sorted(set(update) - allowed)
    if unknown:
        raise ValueError(f"Unknown automation config field(s): {', '.join(unknown)}")
    normalized: dict[str, Any] = {}
    if "profile" in update:
        normalized["profile"] = str(update.get("profile") or "default")
    if "stages" in update:
        stages = update.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("Automation config update stages must be an object.")
        normalized_stages: dict[str, Any] = {}
        for stage_id, payload in stages.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Automation stage {stage_id} must be an object.")
            normalized_stages[str(stage_id)] = _normalize_stage(str(stage_id), payload)
        normalized["stages"] = normalized_stages
    return normalized


def _apply_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    after = copy.deepcopy(base)
    if "profile" in update:
        after["profile"] = update["profile"]
    if "stages" in update:
        stages = after.setdefault("stages", {})
        for stage_id, stage_update in update["stages"].items():
            current = stages.get(stage_id) if isinstance(stages.get(stage_id), dict) else {}
            clean_current = {key: current[key] for key in DEFAULT_STAGE if key in current}
            stages[stage_id] = {**copy.deepcopy(DEFAULT_STAGE), **clean_current, **stage_update}
    return after


def storable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": str(config.get("profile") or "default"),
        "stages": {
            stage_id: {
                "enabled": bool(stage.get("enabled")),
                "mode": str(stage.get("mode") or MODE_MANUAL),
                "requires_review": stage.get("requires_review") is not False,
                "notes": str(stage.get("notes") or ""),
            }
            for stage_id, stage in (config.get("stages") or {}).items()
            if stage_id in STAGE_IDS and isinstance(stage, dict)
        },
    }


def preview_config_update(
    *,
    update: dict[str, Any],
    path: Optional[Path] = None,
    state_root: Optional[Path] = None,
) -> dict[str, Any]:
    target = config_path(path, state_root=state_root)
    raw = read_config(target)
    before = merged_config(raw)
    normalized_update = _validate_update(update)
    after = merged_config(_apply_update(storable_config(before), normalized_update))
    rollback = {
        "previous_config": before,
        "delete_config": not target.exists(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "preview_automation_config_update",
        "config_path": str(target),
        "exists": target.exists(),
        "update": normalized_update,
        "before": before,
        "after": after,
        "rollback": rollback,
        "requires_approval_token": APPLY_APPROVAL_TOKEN,
        "will_write": False,
        "will_execute_workflow_stage": False,
        "will_execute_external_action": False,
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
    preview = preview_config_update(update=update, path=path, state_root=state_root)
    target = Path(preview["config_path"])
    stored = storable_config(preview["after"])
    atomic_write_json(target, stored)
    return {
        **preview,
        "action": "apply_automation_config_update",
        "after": merged_config(stored),
        "will_write": True,
        "applied": True,
    }


def all_config(*, path: Optional[Path] = None, state_root: Optional[Path] = None) -> dict[str, Any]:
    target = config_path(path, state_root=state_root)
    raw = read_config(target)
    config = merged_config(raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(target),
        "exists": target.exists(),
        "profile": config["profile"],
        "stage_order": list(STAGE_IDS),
        "mode_choices": list(MODE_CHOICES),
        "stages": config["stages"],
        "will_execute_workflow_stage": False,
        "will_execute_external_action": False,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect transcript workflow automation configuration.")
    parser.add_argument("--config", type=Path, help="Config path. Defaults to TRANSCRIPTS_AUTOMATION_CONFIG or user state.")
    parser.add_argument("--state-dir", type=Path, help="Runtime state root when --config is not supplied.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="Show resolved automation policy.")
    preview = subparsers.add_parser("preview-update", help="Preview one stage automation update.")
    preview.add_argument("--stage", choices=STAGE_IDS, required=True)
    preview.add_argument("--enabled", choices=("true", "false"))
    preview.add_argument("--mode", choices=MODE_CHOICES)
    preview.add_argument("--requires-review", choices=("true", "false"))
    preview.add_argument("--notes")
    apply_update = subparsers.add_parser("apply-update", help="Apply one stage automation update.")
    apply_update.add_argument("--stage", choices=STAGE_IDS, required=True)
    apply_update.add_argument("--approval-token", required=True)
    apply_update.add_argument("--enabled", choices=("true", "false"))
    apply_update.add_argument("--mode", choices=MODE_CHOICES)
    apply_update.add_argument("--requires-review", choices=("true", "false"))
    apply_update.add_argument("--notes")
    return parser.parse_args(argv)


def update_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.enabled is not None:
        payload["enabled"] = args.enabled == "true"
    if args.mode:
        payload["mode"] = args.mode
    if args.requires_review is not None:
        payload["requires_review"] = args.requires_review == "true"
    if args.notes is not None:
        payload["notes"] = args.notes
    return {"stages": {args.stage: payload}}


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "show":
            payload = all_config(path=args.config, state_root=args.state_dir)
        elif args.command == "preview-update":
            payload = preview_config_update(update=update_from_args(args), path=args.config, state_root=args.state_dir)
        elif args.command == "apply-update":
            payload = apply_config_update(
                update=update_from_args(args),
                approval_token=args.approval_token,
                path=args.config,
                state_root=args.state_dir,
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"automation_config: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
