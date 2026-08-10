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
STRUCTURED_DECISION_VALIDATE_TOKEN = "VALIDATE_STRUCTURED_DECISION"
STRUCTURED_DECISION_APPLY_TOKEN = "APPLY_STRUCTURED_DECISION"
HUMAN_REVIEW_DECISION_TOKEN = "RECORD_HUMAN_REVIEW_DECISION"
FORK_BRANCHES_PREFLIGHT_TOKEN = "PREVIEW_FORK_BRANCHES"
ROLLBACK_PREFLIGHT_TOKEN = "PREVIEW_ROLLBACK"

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
STRUCTURED_DECISION_ACTIONS = {
    "continue_current_branch",
    "fork_branches",
    "rollback",
    "stop",
    "ask_for_human_review",
}
LEDGER_ONLY_STRUCTURED_DECISION_ACTIONS = {"continue_current_branch", "stop", "ask_for_human_review"}
HUMAN_REVIEW_DECISION_ACTIONS = {"annotate", "resolve", "reopen"}


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


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Captured turn output does not contain a JSON object.") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Captured turn output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Captured turn output JSON must be an object.")
    return payload


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
    if run.get("phase") not in {"prepared", "session_started"}:
        raise ValueError(
            "Model-turn preflight requires a prepared or session_started ledger."
        )
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
    packet_json.chmod(0o600)
    packet_text.parent.mkdir(parents=True, exist_ok=True)
    packet_text.write_text(prompt_text, encoding="utf-8")
    packet_text.chmod(0o600)

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


def validate_structured_decision_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    action = payload.get("action")
    if action not in STRUCTURED_DECISION_ACTIONS:
        errors.append(f"action must be one of {sorted(STRUCTURED_DECISION_ACTIONS)}")
    if not isinstance(payload.get("rationale"), str) or not payload.get("rationale", "").strip():
        errors.append("rationale must be a non-empty string")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be a number between 0 and 1")
    review_flags = payload.get("review_flags")
    if not isinstance(review_flags, list) or not all(isinstance(item, str) for item in review_flags):
        errors.append("review_flags must be an array of strings")
    next_prompt = payload.get("recommended_next_prompt", "")
    if next_prompt is not None and not isinstance(next_prompt, str):
        errors.append("recommended_next_prompt must be a string or null")
    if payload.get("action") == "fork_branches":
        branch_count = payload.get("branch_count")
        if not isinstance(branch_count, int) or isinstance(branch_count, bool) or not 1 <= branch_count <= 5:
            errors.append("branch_count must be an integer from 1 to 5 when action is fork_branches")
        experiments = payload.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            errors.append("experiments must be a non-empty array when action is fork_branches")
    return not errors, errors


def validate_latest_structured_decision(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    approval_token: str,
) -> dict[str, Any]:
    if approval_token != STRUCTURED_DECISION_VALIDATE_TOKEN:
        raise ValueError(f"Validating structured decisions requires approval_token={STRUCTURED_DECISION_VALIDATE_TOKEN}.")
    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    status = run.get("latest_model_turn_status") if isinstance(run.get("latest_model_turn_status"), dict) else {}
    artifact_path = Path(str(status.get("artifact_path") or ""))
    path = run_dir(state_root, run_id)
    if not artifact_path.exists():
        raise FileNotFoundError("No captured model-turn status artifact exists for this run.")
    resolved = artifact_path.resolve()
    try:
        resolved.relative_to(path.resolve())
    except ValueError as exc:
        raise ValueError("Captured status artifact resolves outside the run directory.") from exc
    status_payload = read_json(resolved)
    output_text = str(status_payload.get("output_text") or "")
    parsed: dict[str, Any] = {}
    parse_error = ""
    try:
        parsed = extract_json_object(output_text)
    except ValueError as exc:
        parse_error = str(exc)
    valid, errors = validate_structured_decision_payload(parsed) if parsed else (False, [parse_error])
    decision_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-decision-{uuid.uuid4().hex[:8]}"
    validation = {
        "schema_version": "transcribe-audio.app-intelligence-structured-decision-validation.v1",
        "decision_id": decision_id,
        "run_id": run_id,
        "codex_thread_id": status.get("codex_thread_id") or "",
        "codex_turn_id": status.get("codex_turn_id") or "",
        "source_status_artifact": str(resolved),
        "validated_at": utc_now(),
        "valid": valid,
        "errors": errors,
        "decision": parsed,
        "allowed_actions": sorted(STRUCTURED_DECISION_ACTIONS),
        "will_execute_host_action": False,
    }
    artifact_dir = path / "artifacts" / "structured-decisions"
    validation_path = artifact_dir / f"{decision_id}.json"
    write_json(validation_path, validation)

    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    decision_summary = {
        "decision_id": decision_id,
        "codex_thread_id": validation["codex_thread_id"],
        "codex_turn_id": validation["codex_turn_id"],
        "valid": valid,
        "action": parsed.get("action") or "",
        "status": "validated" if valid else "rejected",
        "artifact_path": str(validation_path),
        "will_execute_host_action": False,
        "created_at": validation["validated_at"],
    }
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={"decisions": [*decisions, decision_summary]},
    )
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="structured_decision_validated",
        payload={
            "decision_id": decision_id,
            "valid": valid,
            "action": parsed.get("action") or "",
            "artifact_path": str(validation_path),
            "error_count": len(errors),
            "will_execute_host_action": False,
        },
    )
    return {
        "schema_version": validation["schema_version"],
        "action": "validate_structured_decision",
        "ok": True,
        "valid": valid,
        "errors": errors,
        "decision": parsed,
        "decision_id": decision_id,
        "artifact_path": str(validation_path),
        "will_execute_host_action": False,
        "run": ledger,
        "event": event,
    }


def apply_validated_structured_decision(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    decision_id: str,
    approval_token: str,
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    if approval_token != STRUCTURED_DECISION_APPLY_TOKEN:
        raise ValueError(f"Applying structured decisions requires approval_token={STRUCTURED_DECISION_APPLY_TOKEN}.")
    if not decision_id:
        raise ValueError("decision_id is required.")

    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    path = run_dir(state_root, run_id)
    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    index = next((idx for idx, item in enumerate(decisions) if item.get("decision_id") == decision_id), -1)
    if index < 0:
        raise ValueError(f"Decision not found for run: {decision_id}")
    summary = decisions[index]
    if summary.get("status") != "validated" or not summary.get("valid"):
        raise ValueError("Only validated structured decisions can be applied.")
    if summary.get("apply_result"):
        raise ValueError("Structured decision has already been applied.")

    validation_path = Path(str(summary.get("artifact_path") or ""))
    resolved = validation_path.resolve()
    try:
        resolved.relative_to(path.resolve())
    except ValueError as exc:
        raise ValueError("Decision artifact resolves outside the run directory.") from exc
    if not resolved.exists():
        raise FileNotFoundError("Decision validation artifact is missing.")
    validation = read_json(resolved)
    decision = validation.get("decision") if isinstance(validation.get("decision"), dict) else {}
    action = str(decision.get("action") or summary.get("action") or "")
    if action not in LEDGER_ONLY_STRUCTURED_DECISION_ACTIONS:
        raise ValueError(
            "This apply endpoint only records ledger-only continue_current_branch, stop, "
            "and ask_for_human_review decisions; "
            f"action {action!r} is not enabled."
        )

    applied_at = utc_now()
    if action == "stop":
        status = "stopped"
        phase = "stopped"
    elif action == "ask_for_human_review":
        status = "needs_human_review"
        phase = "human_review_requested"
    else:
        status = "running"
        phase = "current_branch_continued"
    current_branch = str((run.get("state") if isinstance(run.get("state"), dict) else {}).get("current_branch") or "main")
    apply_artifact = {
        "schema_version": "transcribe-audio.app-intelligence-structured-decision-apply.v1",
        "run_id": run_id,
        "decision_id": decision_id,
        "action": action,
        "applied_at": applied_at,
        "reviewer": reviewer or "operator",
        "note": note,
        "source_validation_artifact": str(resolved),
        "decision": decision,
        "current_branch": current_branch,
        "applied_ledger_state": True,
        "will_execute_external_action": False,
        "will_execute_downstream_action": False,
        "will_execute_write_bearing_action": False,
        "will_fork_or_rollback": False,
    }
    apply_path = path / "artifacts" / "structured-decisions" / f"{decision_id}.apply.json"
    write_json(apply_path, apply_artifact)

    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="structured_decision_applied",
        payload={
            "decision_id": decision_id,
            "action": action,
            "artifact_path": str(apply_path),
            "applied_ledger_state": True,
            "current_branch": current_branch,
            "will_execute_external_action": False,
            "will_execute_downstream_action": False,
            "will_execute_write_bearing_action": False,
        },
    )
    updated_summary = {
        **summary,
        "status": "applied",
        "applied_at": applied_at,
        "apply_event_id": event["event_id"],
        "apply_result": {
            "action": action,
            "artifact_path": str(apply_path),
            "applied_ledger_state": True,
            "current_branch": current_branch,
            "will_execute_external_action": False,
            "will_execute_downstream_action": False,
            "will_execute_write_bearing_action": False,
        },
    }
    updates = {
        "status": status,
        "phase": phase,
        "decisions": [*decisions[:index], updated_summary, *decisions[index + 1 :]],
    }
    if action == "continue_current_branch":
        updates["latest_continuation"] = {
            "decision_id": decision_id,
            "action": action,
            "current_branch": current_branch,
            "applied_at": applied_at,
            "ledger_only": True,
        }
    else:
        updates["final"] = {
            "decision_id": decision_id,
            "action": action,
            "status": status,
            "applied_at": applied_at,
            "ledger_only": True,
        }
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates=updates,
    )
    return {
        "schema_version": apply_artifact["schema_version"],
        "action": "apply_structured_decision",
        "ok": True,
        "decision_id": decision_id,
        "decision_action": action,
        "artifact_path": str(apply_path),
        "applied_ledger_state": True,
        "will_execute_external_action": False,
        "will_execute_downstream_action": False,
        "will_execute_write_bearing_action": False,
        "will_fork_or_rollback": False,
        "run": ledger,
        "event": event,
    }


def preflight_fork_branches(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    decision_id: str,
    approval_token: str,
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    if approval_token != FORK_BRANCHES_PREFLIGHT_TOKEN:
        raise ValueError(f"Fork branch preflight requires approval_token={FORK_BRANCHES_PREFLIGHT_TOKEN}.")
    if not decision_id:
        raise ValueError("decision_id is required.")

    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    path = run_dir(state_root, run_id)
    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    summary = next((item for item in decisions if item.get("decision_id") == decision_id), None)
    if not isinstance(summary, dict):
        raise ValueError(f"Decision not found for run: {decision_id}")
    if summary.get("status") != "validated" or not summary.get("valid"):
        raise ValueError("Only validated structured decisions can be preflighted.")
    if summary.get("action") != "fork_branches":
        raise ValueError("Fork preflight requires a fork_branches decision.")

    validation_path = Path(str(summary.get("artifact_path") or ""))
    resolved = validation_path.resolve()
    try:
        resolved.relative_to(path.resolve())
    except ValueError as exc:
        raise ValueError("Decision artifact resolves outside the run directory.") from exc
    if not resolved.exists():
        raise FileNotFoundError("Decision validation artifact is missing.")
    validation = read_json(resolved)
    decision = validation.get("decision") if isinstance(validation.get("decision"), dict) else {}
    branch_count = int(decision.get("branch_count") or 0)
    experiments = decision.get("experiments") if isinstance(decision.get("experiments"), list) else []
    if branch_count < 1 or not experiments:
        raise ValueError("Fork decision is missing branch_count or experiments.")

    current_branch = str((run.get("state") if isinstance(run.get("state"), dict) else {}).get("current_branch") or "main")
    planned_at = utc_now()
    planned_branches = []
    for index in range(branch_count):
        experiment = str(experiments[index] if index < len(experiments) else f"experiment-{index + 1}")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", experiment.strip().lower()).strip("-") or f"branch-{index + 1}"
        branch_id = f"{current_branch}-fork-{index + 1}-{slug[:32]}"
        planned_branches.append(
            {
                "branch_id": branch_id,
                "parent_branch": current_branch,
                "experiment": experiment,
                "planned_status": "preview_only",
                "will_create_codex_thread": False,
                "will_modify_ledger_branch_state": False,
                "will_run_provider": False,
            }
        )

    preflight = {
        "schema_version": "transcribe-audio.app-intelligence-fork-branches-preflight.v1",
        "run_id": run_id,
        "decision_id": decision_id,
        "created_at": planned_at,
        "reviewer": reviewer or "operator",
        "note": note,
        "source_validation_artifact": str(resolved),
        "current_branch": current_branch,
        "requested_branch_count": branch_count,
        "planned_branches": planned_branches,
        "will_create_thread": False,
        "will_modify_branches": False,
        "will_run_provider": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }
    artifact_path = path / "artifacts" / "structured-decisions" / f"{decision_id}.fork-preflight.json"
    write_json(artifact_path, preflight)
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="fork_branches_preflight",
        payload={
            "decision_id": decision_id,
            "artifact_path": str(artifact_path),
            "planned_branch_count": len(planned_branches),
            "will_create_thread": False,
            "will_modify_branches": False,
            "will_run_provider": False,
        },
    )
    return {
        **preflight,
        "action": "preflight_fork_branches",
        "ok": True,
        "artifact_path": str(artifact_path),
        "event": event,
    }


def preflight_rollback(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    decision_id: str,
    approval_token: str,
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    if approval_token != ROLLBACK_PREFLIGHT_TOKEN:
        raise ValueError(f"Rollback preflight requires approval_token={ROLLBACK_PREFLIGHT_TOKEN}.")
    if not decision_id:
        raise ValueError("decision_id is required.")

    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    path = run_dir(state_root, run_id)
    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    summary = next((item for item in decisions if item.get("decision_id") == decision_id), None)
    if not isinstance(summary, dict):
        raise ValueError(f"Decision not found for run: {decision_id}")
    if summary.get("status") != "validated" or not summary.get("valid"):
        raise ValueError("Only validated structured decisions can be preflighted.")
    if summary.get("action") != "rollback":
        raise ValueError("Rollback preflight requires a rollback decision.")

    validation_path = Path(str(summary.get("artifact_path") or ""))
    resolved = validation_path.resolve()
    try:
        resolved.relative_to(path.resolve())
    except ValueError as exc:
        raise ValueError("Decision artifact resolves outside the run directory.") from exc
    if not resolved.exists():
        raise FileNotFoundError("Decision validation artifact is missing.")
    validation = read_json(resolved)
    decision = validation.get("decision") if isinstance(validation.get("decision"), dict) else {}
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    current_branch = str(state.get("current_branch") or "main")
    target_branch = str(decision.get("target_branch") or current_branch)
    target_event_id = str(decision.get("target_event_id") or "")
    target_turn_id = str(decision.get("target_turn_id") or "")
    warnings = []
    if not target_event_id and not target_turn_id:
        warnings.append("Rollback target is advisory only; no target_event_id or target_turn_id was supplied.")

    preflight = {
        "schema_version": "transcribe-audio.app-intelligence-rollback-preflight.v1",
        "run_id": run_id,
        "decision_id": decision_id,
        "created_at": utc_now(),
        "reviewer": reviewer or "operator",
        "note": note,
        "source_validation_artifact": str(resolved),
        "current_branch": current_branch,
        "target_branch": target_branch,
        "target_event_id": target_event_id,
        "target_turn_id": target_turn_id,
        "planned_status": "preview_only",
        "decision": decision,
        "warnings": warnings,
        "will_modify_branches": False,
        "will_revert_artifacts": False,
        "will_create_thread": False,
        "will_run_provider": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }
    artifact_path = path / "artifacts" / "structured-decisions" / f"{decision_id}.rollback-preflight.json"
    write_json(artifact_path, preflight)
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="rollback_preflight",
        payload={
            "decision_id": decision_id,
            "artifact_path": str(artifact_path),
            "current_branch": current_branch,
            "target_branch": target_branch,
            "target_event_id": target_event_id,
            "target_turn_id": target_turn_id,
            "will_modify_branches": False,
            "will_revert_artifacts": False,
            "will_run_provider": False,
        },
    )
    return {
        **preflight,
        "action": "preflight_rollback",
        "ok": True,
        "artifact_path": str(artifact_path),
        "event": event,
    }


def record_human_review_decision(
    *,
    state_root: Optional[Path] = None,
    run_id: str,
    decision_id: str,
    review_action: str,
    approval_token: str,
    reviewer: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    if approval_token != HUMAN_REVIEW_DECISION_TOKEN:
        raise ValueError(f"Recording human-review decisions requires approval_token={HUMAN_REVIEW_DECISION_TOKEN}.")
    if review_action not in HUMAN_REVIEW_DECISION_ACTIONS:
        raise ValueError(f"review_action must be one of {sorted(HUMAN_REVIEW_DECISION_ACTIONS)}.")
    if not note.strip() and review_action in {"resolve", "reopen"}:
        raise ValueError("A note is required when resolving or reopening a human-review decision.")

    shown = response_for_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown["run"]
    decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
    index = next((idx for idx, item in enumerate(decisions) if item.get("decision_id") == decision_id), -1)
    if index < 0:
        raise ValueError(f"Decision not found for run: {decision_id}")
    summary = decisions[index]
    if summary.get("action") != "ask_for_human_review":
        raise ValueError("Only ask_for_human_review decisions can be recorded through this endpoint.")
    if summary.get("status") not in {"validated", "applied"} or not summary.get("valid"):
        raise ValueError("Only validated or ledger-applied human-review decisions can be recorded.")
    if review_action in {"resolve", "reopen"} and summary.get("status") != "applied":
        raise ValueError("Resolve and reopen require a ledger-applied human-review decision.")

    recorded_at = utc_now()
    previous_state = summary.get("human_review") if isinstance(summary.get("human_review"), dict) else {}
    previous_notes = previous_state.get("notes") if isinstance(previous_state.get("notes"), list) else []
    state_status = str(previous_state.get("status") or "open")
    if review_action == "resolve":
        state_status = "resolved"
    elif review_action == "reopen":
        state_status = "open"
    note_record = {
        "action": review_action,
        "note": note,
        "reviewer": reviewer or "operator",
        "recorded_at": recorded_at,
    }
    updated_summary = {
        **summary,
        "human_review": {
            **previous_state,
            "status": state_status,
            "updated_at": recorded_at,
            "updated_by": reviewer or "operator",
            "notes": [*previous_notes, note_record],
        },
    }
    event = append_event(
        state_root=state_root,
        run_id=run_id,
        event_type="human_review_decision_recorded",
        payload={
            "decision_id": decision_id,
            "review_action": review_action,
            "human_review_status": state_status,
            "reviewer": reviewer or "operator",
            "note": note,
            "will_execute_external_action": False,
            "will_execute_downstream_action": False,
            "will_execute_write_bearing_action": False,
            "will_fork_or_rollback": False,
        },
    )
    final = run.get("final") if isinstance(run.get("final"), dict) else None
    if final and final.get("decision_id") == decision_id:
        final = {**final, "human_review_status": state_status, "human_review_updated_at": recorded_at}
    ledger = update_run_json(
        state_root=state_root,
        run_id=run_id,
        updates={
            "decisions": [*decisions[:index], updated_summary, *decisions[index + 1 :]],
            "final": final,
        },
    )
    return {
        "schema_version": "transcribe-audio.app-intelligence-human-review-decision.v1",
        "action": "record_human_review_decision",
        "ok": True,
        "run_id": run_id,
        "decision_id": decision_id,
        "review_action": review_action,
        "human_review_status": state_status,
        "reviewer": reviewer or "operator",
        "note": note,
        "will_execute_external_action": False,
        "will_execute_downstream_action": False,
        "will_execute_write_bearing_action": False,
        "will_fork_or_rollback": False,
        "run": ledger,
        "event": event,
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
