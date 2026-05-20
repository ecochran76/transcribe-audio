#!/usr/bin/env python3
"""
User-scoped App Intelligence run ledger for supervised transcript workflows.
"""
from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
RUNS_DIR_NAME = "app-intelligence-runs"
SCHEMA_VERSION = "transcribe-audio.app-intelligence-run.v1"
SESSION_START_APPROVAL_TOKEN = "START_APP_SERVER_SESSION"
SESSION_START_PREFLIGHT_EVENT_TOKEN = "APPEND_SESSION_START_PREFLIGHT_EVENT"
MODEL_TURN_PREFLIGHT_TOKEN = "PREPARE_MODEL_TURN_PREFLIGHT"
MODEL_TURN_SEND_TOKEN = "SEND_APP_SERVER_MODEL_TURN"
MODEL_TURN_STATUS_TOKEN = "CAPTURE_MODEL_TURN_STATUS"

DEFAULT_ALLOWED_ACTIONS = [
    "inspect_context",
    "prepare_prompt",
    "send_model_turn",
    "start_app_server_session",
    "record_codex_event",
    "record_structured_decision",
    "run_eval",
    "fork_thread",
    "rollback_thread",
    "ask_for_human_review",
    "stop",
]

DEFAULT_EVAL_POLICY = {
    "required_before_write": True,
    "required_before_external_apply": True,
    "authoritative_checks": [],
}

DEFAULT_APPROVAL_POLICY = {
    "network": "review_required",
    "external_write": "review_required",
    "repo_write": "phase_scoped",
    "destructive": "forbidden",
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def runs_root(state_root: Optional[Path] = None) -> Path:
    return (state_root or DEFAULT_STATE_DIR).expanduser() / RUNS_DIR_NAME


def validate_run_id(run_id: str) -> str:
    if not run_id or not RUN_ID_RE.match(run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, colon, and hyphen.")
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be a single safe path segment.")
    return run_id


def new_run_id(workflow: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", workflow.strip().lower()).strip("-") or "run"
    return f"{stamp}-{slug}-{uuid.uuid4().hex[:8]}"


def run_dir(state_root: Optional[Path], run_id: str) -> Path:
    safe_id = validate_run_id(run_id)
    root = runs_root(state_root).resolve()
    path = (root / safe_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("run_id resolves outside the App Intelligence run directory.") from exc
    return path


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def append_codex_event(*, state_root: Optional[Path] = None, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = run_dir(state_root, run_id)
    event = {
        "schema_version": "transcribe-audio.app-intelligence-codex-event.v1",
        "run_id": run_id,
        "captured_at": utc_now(),
        "payload": payload,
    }
    append_jsonl(path / "codex_events.jsonl", event)
    return event


def create_run(
    *,
    state_root: Optional[Path] = None,
    workflow: str,
    purpose: str,
    document_id: str = "",
    provider: str = "codex-app-server",
    created_by: str = "operator",
    allowed_actions: Optional[list[str]] = None,
    run_id: str = "",
) -> dict[str, Any]:
    run_id = validate_run_id(run_id) if run_id else new_run_id(workflow)
    path = run_dir(state_root, run_id)
    if path.exists():
        raise FileExistsError(f"App Intelligence run already exists: {run_id}")
    now = utc_now()
    actions = allowed_actions or DEFAULT_ALLOWED_ACTIONS
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workflow": workflow,
        "purpose": purpose,
        "document_id": document_id,
        "provider": provider,
        "status": "prepared",
        "phase": "prepared",
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
        "state": {
            "current_branch": "main",
            "branches": {
                "main": {
                    "parent_branch": None,
                    "codex_thread_id": None,
                    "latest_turn_id": None,
                    "status": "prepared",
                }
            },
            "active_codex_thread_id": None,
            "latest_turn_id": None,
        },
        "rng": {
            "branch_sampler": secrets.randbits(32),
            "eval_subset_selector": secrets.randbits(32),
            "prompt_variant_selector": secrets.randbits(32),
        },
        "policy": {
            "host_owns_control_flow": True,
            "structured_decisions_required": True,
            "allowed_actions": actions,
            "approval_policy": DEFAULT_APPROVAL_POLICY,
            "eval_policy": DEFAULT_EVAL_POLICY,
            "remote_transport": "forbidden_without_auth_review",
        },
        "artifacts": {
            "run_json": "run.json",
            "events_jsonl": "events.jsonl",
            "codex_events_jsonl": "codex_events.jsonl",
            "branches_dir": "branches",
            "artifacts_dir": "artifacts",
            "diffs_dir": "diffs",
        },
        "decisions": [],
        "final": None,
    }
    for child in ["branches", "artifacts", "diffs"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    write_json(path / "run.json", ledger)
    append_event(state_root=state_root, run_id=run_id, event_type="run_prepared", payload={"workflow": workflow})
    return response_for_run(state_root=state_root, run_id=run_id)


def append_event(*, state_root: Optional[Path] = None, run_id: str, event_type: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    path = run_dir(state_root, run_id)
    event = {
        "schema_version": "transcribe-audio.app-intelligence-event.v1",
        "run_id": run_id,
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "created_at": utc_now(),
        "payload": payload or {},
    }
    append_jsonl(path / "events.jsonl", event)
    return event


def update_run_json(*, state_root: Optional[Path] = None, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = run_dir(state_root, run_id)
    ledger_path = path / "run.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"App Intelligence run not found: {run_id}")
    ledger = read_json(ledger_path)
    ledger.update(updates)
    ledger["updated_at"] = utc_now()
    write_json(ledger_path, ledger)
    return ledger


def read_events(path: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


def run_summary(path: Path) -> dict[str, Any]:
    ledger = read_json(path / "run.json")
    return {
        "run_id": ledger.get("run_id") or path.name,
        "workflow": ledger.get("workflow") or "",
        "purpose": ledger.get("purpose") or "",
        "document_id": ledger.get("document_id") or "",
        "provider": ledger.get("provider") or "",
        "status": ledger.get("status") or "unknown",
        "phase": ledger.get("phase") or "",
        "created_at": ledger.get("created_at") or "",
        "updated_at": ledger.get("updated_at") or "",
        "path": str(path),
    }


def list_runs(*, state_root: Optional[Path] = None, limit: int = 50) -> dict[str, Any]:
    root = runs_root(state_root)
    paths = [path for path in root.glob("*/run.json") if path.is_file()] if root.exists() else []
    dirs = sorted((path.parent for path in paths), key=lambda path: path.stat().st_mtime, reverse=True)
    return {
        "state_dir": str((state_root or DEFAULT_STATE_DIR).expanduser()),
        "runs_dir": str(root),
        "limit": limit,
        "items": [run_summary(path) for path in dirs[:limit]],
        "total": len(dirs),
    }


def response_for_run(*, state_root: Optional[Path] = None, run_id: str, event_limit: int = 50) -> dict[str, Any]:
    path = run_dir(state_root, run_id)
    ledger_path = path / "run.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"App Intelligence run not found: {run_id}")
    ledger = read_json(ledger_path)
    return {
        "run": ledger,
        "path": str(path),
        "events": read_events(path / "events.jsonl", limit=event_limit),
        "codex_events_count": sum(1 for _ in (path / "codex_events.jsonl").open(encoding="utf-8")) if (path / "codex_events.jsonl").exists() else 0,
    }


def session_start_preflight(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    provider_ready: bool,
    provider_status: str = "",
    approval_token: str = "",
    append_event_log: bool = False,
) -> dict[str, Any]:
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=5)
    run = shown["run"]
    policy = run.get("policy") if isinstance(run.get("policy"), dict) else {}
    allowed_actions = policy.get("allowed_actions") if isinstance(policy.get("allowed_actions"), list) else []
    token_shape_ok = approval_token in {"", SESSION_START_APPROVAL_TOKEN, SESSION_START_PREFLIGHT_EVENT_TOKEN}
    checks = {
        "run_exists": True,
        "phase_prepared": run.get("phase") == "prepared",
        "provider_is_codex_app_server": run.get("provider") == "codex-app-server",
        "provider_ready": bool(provider_ready),
        "start_action_allowed": "start_app_server_session" in allowed_actions,
        "host_owns_control_flow": policy.get("host_owns_control_flow") is True,
        "structured_decisions_required": policy.get("structured_decisions_required") is True,
        "approval_token_shape": token_shape_ok,
    }
    blocking = [name for name, ok in checks.items() if not ok]
    event = None
    if append_event_log:
        if approval_token != SESSION_START_PREFLIGHT_EVENT_TOKEN:
            raise ValueError(f"Appending a preflight event requires approval_token={SESSION_START_PREFLIGHT_EVENT_TOKEN}.")
        event = append_event(
            state_root=state_root,
            run_id=run_id,
            event_type="session_start_preflight",
            payload={
                "provider_status": provider_status,
                "checks": checks,
                "would_start_session": False,
                "future_required_approval_token": SESSION_START_APPROVAL_TOKEN,
            },
        )
    return {
        "schema_version": "transcribe-audio.app-intelligence-session-start-preflight.v1",
        "action": "session_start_preflight",
        "run_id": run_id,
        "provider": run.get("provider") or "",
        "provider_status": provider_status,
        "checks": checks,
        "blocking_checks": blocking,
        "ok": not blocking,
        "will_start_session": False,
        "will_write_event": bool(append_event_log),
        "required_approval_token_for_event": SESSION_START_PREFLIGHT_EVENT_TOKEN,
        "future_required_approval_token_for_session_start": SESSION_START_APPROVAL_TOKEN,
        "event": event,
        "run": run_summary(run_dir(state_root, run_id)),
    }


def record_session_start_requested(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    transport: str,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != SESSION_START_APPROVAL_TOKEN:
        raise ValueError(f"Starting an app-server session requires approval_token={SESSION_START_APPROVAL_TOKEN}.")
    if transport not in {"stdio", "unix"}:
        raise ValueError("App-server session transport must be stdio or unix.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    if run.get("phase") != "prepared":
        raise ValueError("App-server session start requires a prepared ledger.")
    return append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="app_server_session_start_requested",
        payload={
            "transport": transport,
            "approval_token": SESSION_START_APPROVAL_TOKEN,
            "will_start_model_turn": False,
        },
    )


def record_session_start_failed(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    transport: str,
    error: str,
) -> dict[str, Any]:
    return append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="app_server_session_start_failed",
        payload={
            "transport": transport,
            "error": error,
            "started_model_turn": False,
        },
    )


def mark_session_started(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    transport: str,
    codex_bin: str,
    start_result: dict[str, Any],
    version_result: dict[str, Any],
) -> dict[str, Any]:
    if transport not in {"stdio", "unix"}:
        raise ValueError("App-server session transport must be stdio or unix.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    branches = state.get("branches") if isinstance(state.get("branches"), dict) else {}
    main_branch = branches.get("main") if isinstance(branches.get("main"), dict) else {}
    now = utc_now()
    main_branch = {**main_branch, "status": "session_started"}
    branches = {**branches, "main": main_branch}
    state = {
        **state,
        "branches": branches,
        "active_codex_thread_id": None,
        "latest_turn_id": None,
        "app_server": {
            "transport": transport,
            "codex_bin": codex_bin,
            "started_at": now,
            "start_result": start_result,
            "version_result": version_result,
            "model_turn_started": False,
        },
    }
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={
            "status": "running",
            "phase": "session_started",
            "state": state,
        },
    )
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="app_server_session_started",
        payload={
            "transport": transport,
            "codex_bin": codex_bin,
            "started_model_turn": False,
            "version": version_result,
        },
    )
    return {
        "run": ledger,
        "event": event,
        "path": str(run_dir(state_root, run_id)),
    }


def prepare_model_turn_packet(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    task: str,
    route: dict[str, Any],
    document: dict[str, Any],
    prompt_text: str,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != MODEL_TURN_PREFLIGHT_TOKEN:
        raise ValueError(f"Model-turn preflight requires approval_token={MODEL_TURN_PREFLIGHT_TOKEN}.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    if run.get("phase") != "session_started":
        raise ValueError("Model-turn preflight requires a session_started ledger.")
    policy = run.get("policy") if isinstance(run.get("policy"), dict) else {}
    allowed_actions = policy.get("allowed_actions") if isinstance(policy.get("allowed_actions"), list) else []
    if "prepare_prompt" not in allowed_actions:
        raise ValueError("Ledger policy does not allow prepare_prompt.")

    path = run_dir(state_root, run_id)
    packet_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-model-turn-{uuid.uuid4().hex[:8]}"
    packet_dir = path / "artifacts" / "prompt-packets"
    packet_json = packet_dir / f"{packet_id}.json"
    packet_text = packet_dir / f"{packet_id}.prompt.txt"
    packet = {
        "schema_version": "transcribe-audio.app-intelligence-model-turn-preflight.v1",
        "packet_id": packet_id,
        "run_id": run_id,
        "task": task,
        "route": route,
        "document": document,
        "prompt_path": str(packet_text),
        "review_required": True,
        "will_send_prompt": False,
        "future_required_approval_token_for_send": MODEL_TURN_SEND_TOKEN,
        "created_at": utc_now(),
    }
    write_json(packet_json, {**packet, "prompt_text": prompt_text})
    packet_text.parent.mkdir(parents=True, exist_ok=True)
    packet_text.write_text(prompt_text, encoding="utf-8")

    prompt_packets = run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []
    prompt_summary = {
        "packet_id": packet_id,
        "task": task,
        "document_id": document.get("id") or "",
        "packet_path": str(packet_json),
        "prompt_path": str(packet_text),
        "created_at": packet["created_at"],
        "sent": False,
    }
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={"prompt_packets": [*prompt_packets, prompt_summary]},
    )
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="model_turn_preflight_prepared",
        payload={
            "packet_id": packet_id,
            "task": task,
            "document_id": document.get("id") or "",
            "packet_path": str(packet_json),
            "prompt_path": str(packet_text),
            "will_send_prompt": False,
            "future_required_approval_token": MODEL_TURN_SEND_TOKEN,
        },
    )
    return {
        "schema_version": packet["schema_version"],
        "action": "prepare_model_turn_preflight",
        "ok": True,
        "will_send_prompt": False,
        "packet": packet,
        "packet_path": str(packet_json),
        "prompt_path": str(packet_text),
        "event": event,
        "run": ledger,
    }


def read_model_turn_packet(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    packet_id: str,
) -> dict[str, Any]:
    if not packet_id or "/" in packet_id or "\\" in packet_id or ".." in packet_id:
        raise ValueError("packet_id must be a single safe path segment.")
    path = run_dir(state_root, run_id)
    packet_dir = (path / "artifacts" / "prompt-packets").resolve()
    packet_path = (packet_dir / f"{packet_id}.json").resolve()
    try:
        packet_path.relative_to(packet_dir)
    except ValueError as exc:
        raise ValueError("packet_id resolves outside the prompt-packets directory.") from exc
    if not packet_path.exists():
        raise FileNotFoundError(f"Prompt packet not found: {packet_id}")
    packet_payload = read_json(packet_path)
    prompt_path = Path(str(packet_payload.get("prompt_path") or ""))
    prompt_text = ""
    if prompt_path.exists():
        prompt_resolved = prompt_path.resolve()
        try:
            prompt_resolved.relative_to(packet_dir)
        except ValueError as exc:
            raise ValueError("prompt_path resolves outside the prompt-packets directory.") from exc
        prompt_text = prompt_resolved.read_text(encoding="utf-8")
    return {
        "schema_version": "transcribe-audio.app-intelligence-model-turn-packet-review.v1",
        "action": "review_model_turn_packet",
        "run_id": run_id,
        "packet_id": packet_id,
        "packet_path": str(packet_path),
        "prompt_path": str(prompt_path),
        "packet": packet_payload,
        "prompt_text": prompt_text,
        "will_send_prompt": False,
        "future_required_approval_token_for_send": MODEL_TURN_SEND_TOKEN,
    }


def model_turn_send_preflight(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    packet_id: str,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != MODEL_TURN_SEND_TOKEN:
        raise ValueError(f"Model-turn send preflight requires approval_token={MODEL_TURN_SEND_TOKEN}.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=5)
    run = shown["run"]
    packet_review = read_model_turn_packet(state_root=state_root, run_id=run_id, packet_id=packet_id)
    policy = run.get("policy") if isinstance(run.get("policy"), dict) else {}
    allowed_actions = policy.get("allowed_actions") if isinstance(policy.get("allowed_actions"), list) else []
    packet = packet_review.get("packet") if isinstance(packet_review.get("packet"), dict) else {}
    matching_summary = next(
        (
            summary for summary in run.get("prompt_packets", [])
            if isinstance(summary, dict) and summary.get("packet_id") == packet_id
        ),
        {},
    ) if isinstance(run.get("prompt_packets"), list) else {}
    checks = {
        "run_exists": True,
        "phase_session_started": run.get("phase") == "session_started",
        "provider_is_codex_app_server": run.get("provider") == "codex-app-server",
        "send_action_allowed": "send_model_turn" in allowed_actions,
        "host_owns_control_flow": policy.get("host_owns_control_flow") is True,
        "structured_decisions_required": policy.get("structured_decisions_required") is True,
        "packet_exists": bool(packet),
        "packet_matches_run": packet.get("run_id") == run_id,
        "packet_review_required": packet.get("review_required") is True,
        "packet_not_sent": matching_summary.get("sent") is False,
        "prompt_text_present": bool(str(packet_review.get("prompt_text") or "").strip()),
    }
    blocking = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "transcribe-audio.app-intelligence-model-turn-send-preflight.v1",
        "action": "model_turn_send_preflight",
        "run_id": run_id,
        "packet_id": packet_id,
        "ok": not blocking,
        "checks": checks,
        "blocking_checks": blocking,
        "will_send_prompt": False,
        "will_write_event": False,
        "required_approval_token_checked": MODEL_TURN_SEND_TOKEN,
        "future_action": "send_model_turn",
        "packet": packet,
        "packet_path": packet_review.get("packet_path") or "",
        "prompt_path": packet_review.get("prompt_path") or "",
        "prompt_char_count": len(str(packet_review.get("prompt_text") or "")),
        "run": run_summary(run_dir(state_root, run_id)),
    }


def record_model_turn_started(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    packet_id: str,
    thread_id: str,
    turn_id: str,
    app_server_result: dict[str, Any],
) -> dict[str, Any]:
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    branches = state.get("branches") if isinstance(state.get("branches"), dict) else {}
    current_branch = str(state.get("current_branch") or "main")
    branch_state = branches.get(current_branch) if isinstance(branches.get(current_branch), dict) else {}
    branch_state = {
        **branch_state,
        "codex_thread_id": thread_id,
        "latest_turn_id": turn_id,
        "status": "model_turn_started",
    }
    state = {
        **state,
        "branches": {**branches, current_branch: branch_state},
        "active_codex_thread_id": thread_id,
        "latest_turn_id": turn_id,
        "app_server": {
            **(state.get("app_server") if isinstance(state.get("app_server"), dict) else {}),
            "model_turn_started": True,
            "latest_packet_id": packet_id,
            "latest_turn_id": turn_id,
        },
    }
    prompt_packets = run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []
    updated_packets: list[dict[str, Any]] = []
    for packet in prompt_packets:
        if not isinstance(packet, dict):
            continue
        if packet.get("packet_id") == packet_id:
            updated_packets.append(
                {
                    **packet,
                    "sent": True,
                    "sent_at": utc_now(),
                    "codex_thread_id": thread_id,
                    "codex_turn_id": turn_id,
                }
            )
        else:
            updated_packets.append(packet)
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={
            "status": "running",
            "phase": "model_turn_started",
            "state": state,
            "prompt_packets": updated_packets,
        },
    )
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="model_turn_started",
        payload={
            "packet_id": packet_id,
            "codex_thread_id": thread_id,
            "codex_turn_id": turn_id,
            "will_execute_downstream_action": False,
            "app_server_result": app_server_result,
        },
    )
    return {"run": ledger, "event": event, "path": str(run_dir(state_root, run_id))}


def record_model_turn_failed(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    packet_id: str,
    error: str,
) -> dict[str, Any]:
    return append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="model_turn_send_failed",
        payload={
            "packet_id": packet_id,
            "error": error,
            "started_downstream_action": False,
        },
    )


def record_model_turn_status(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    thread_id: str,
    turn_id: str,
    status_payload: dict[str, Any],
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != MODEL_TURN_STATUS_TOKEN:
        raise ValueError(f"Capturing model-turn status requires approval_token={MODEL_TURN_STATUS_TOKEN}.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    if state.get("active_codex_thread_id") != thread_id or state.get("latest_turn_id") != turn_id:
        raise ValueError("Requested Codex thread/turn does not match the active run state.")

    path = run_dir(state_root, run_id)
    artifact_dir = path / "artifacts" / "model-turn-readouts"
    artifact_path = artifact_dir / f"{turn_id}.status.json"
    output_text = str(status_payload.get("output_text") or "")
    payload = {
        "schema_version": "transcribe-audio.app-intelligence-model-turn-status.v1",
        "run_id": run_id,
        "codex_thread_id": thread_id,
        "codex_turn_id": turn_id,
        "captured_at": utc_now(),
        "status": status_payload.get("status") or "",
        "completed": bool(status_payload.get("completed")),
        "output_text": output_text,
        "raw": status_payload,
        "will_execute_structured_decision": False,
    }
    write_json(artifact_path, payload)

    app_server_state = state.get("app_server") if isinstance(state.get("app_server"), dict) else {}
    state = {
        **state,
        "app_server": {
            **app_server_state,
            "latest_turn_status": payload["status"],
            "latest_turn_completed": payload["completed"],
            "latest_turn_status_artifact": str(artifact_path),
        },
    }
    phase = "model_turn_completed" if payload["completed"] else str(run.get("phase") or "model_turn_started")
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={
            "phase": phase,
            "state": state,
            "latest_model_turn_status": {
                "codex_thread_id": thread_id,
                "codex_turn_id": turn_id,
                "status": payload["status"],
                "completed": payload["completed"],
                "artifact_path": str(artifact_path),
                "output_char_count": len(output_text),
                "captured_at": payload["captured_at"],
            },
        },
    )
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="model_turn_status_captured",
        payload={
            "codex_thread_id": thread_id,
            "codex_turn_id": turn_id,
            "status": payload["status"],
            "completed": payload["completed"],
            "artifact_path": str(artifact_path),
            "output_char_count": len(output_text),
            "will_execute_structured_decision": False,
        },
    )
    return {
        "schema_version": payload["schema_version"],
        "action": "capture_model_turn_status",
        "ok": True,
        "run": ledger,
        "event": event,
        "artifact_path": str(artifact_path),
        "status": payload["status"],
        "completed": payload["completed"],
        "output_text": output_text,
        "will_execute_structured_decision": False,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local App Intelligence run ledgers.")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a prepared run ledger.")
    create_parser.add_argument("--workflow", required=True)
    create_parser.add_argument("--purpose", required=True)
    create_parser.add_argument("--document-id", default="")
    create_parser.add_argument("--provider", default="codex-app-server")
    create_parser.add_argument("--created-by", default="operator")
    create_parser.add_argument("--run-id", default="")

    list_parser = subparsers.add_parser("list", help="List run ledgers.")
    list_parser.add_argument("--limit", type=int, default=50)

    show_parser = subparsers.add_parser("show", help="Show one run ledger.")
    show_parser.add_argument("run_id")
    show_parser.add_argument("--event-limit", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "create":
            payload = create_run(
                state_root=args.state_dir,
                workflow=args.workflow,
                purpose=args.purpose,
                document_id=args.document_id,
                provider=args.provider,
                created_by=args.created_by,
                run_id=args.run_id,
            )
        elif args.command == "list":
            payload = list_runs(state_root=args.state_dir, limit=args.limit)
        elif args.command == "show":
            payload = response_for_run(state_root=args.state_dir, run_id=args.run_id, event_limit=args.event_limit)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
