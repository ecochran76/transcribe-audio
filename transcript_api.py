#!/usr/bin/env python3
"""
Read-only local HTTP API for the transcript review console.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import intelligence_config
import codex_app_server_client
from app_intelligence_ledger import (
    append_codex_event as append_app_intelligence_codex_event,
    apply_validated_structured_decision as apply_app_intelligence_structured_decision,
    create_run as create_app_intelligence_run,
    list_runs as list_app_intelligence_runs,
    mark_session_started as mark_app_intelligence_session_started,
    model_turn_send_preflight as preflight_app_intelligence_model_turn_send,
    preflight_fork_branches as preflight_app_intelligence_fork_branches,
    preflight_rollback as preflight_app_intelligence_rollback,
    prepare_model_turn_packet as prepare_app_intelligence_model_turn_packet,
    read_model_turn_packet as read_app_intelligence_model_turn_packet,
    record_human_review_decision as record_app_intelligence_human_review_decision,
    record_model_turn_failed as record_app_intelligence_model_turn_failed,
    record_model_turn_started as record_app_intelligence_model_turn_started,
    record_model_turn_status as record_app_intelligence_model_turn_status,
    record_session_start_failed as record_app_intelligence_session_start_failed,
    record_session_start_requested as record_app_intelligence_session_start_requested,
    response_for_run as get_app_intelligence_run,
    session_start_preflight as preflight_app_intelligence_session_start,
    validate_latest_structured_decision as validate_app_intelligence_structured_decision,
)
from transcript_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_STORE_DIR,
    TranscriptStoreError,
    connect,
    context_for_document,
    db_path,
    init_db,
    legacy_enrichment_queue,
    parse_object_json,
    search_store,
    store_dir,
)
from transcribe_common import TranscriptionError

DEFAULT_API_PORT = 18876
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
DEFAULT_STATE_DIR = Path("~/.local/state/transcribe-audio")
DEFAULT_BATCH_ENV_FILE = Path("~/.local/state/transcribe-audio/auracall-transcripts.env")
DEFAULT_CODEX_BIN = "codex"
MAX_READINESS_OUTPUT_CHARS = 2000
MAX_APP_ARTIFACT_BYTES = 512 * 1024
APP_SMOKE_RUN_PREFIX = "smoke-replay-manifest"
APP_BROWSER_SMOKE_DIRNAME = "browser-smokes"
APP_SMOKE_JOB_DIRNAME = "smoke-jobs"
APP_SMOKE_JOB_TOKEN = "RUN_APP_SMOKE_JOB"
APP_SMOKE_CLEANUP_TOKEN = "CLEANUP_APP_SMOKE_ARTIFACTS"
APP_SMOKE_JOB_TIMEOUT_SECONDS = 180
MAX_SMOKE_JOB_TAIL_CHARS = 20000
MAX_SMOKE_EVIDENCE_BYTES = 5 * 1024 * 1024


def document_summary(row: sqlite3.Row) -> dict[str, Any]:
    metadata = parse_object_json(row["metadata_json"])
    media_blob = metadata.get("media_blob") if isinstance(metadata.get("media_blob"), dict) else {}
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "source_path": row["source_path"],
        "stored_path": row["stored_path"],
        "generated_at": row["generated_at"],
        "updated_at": row["updated_at"],
        "embedding_provider": row["embedding_provider"],
        "embedding_model": row["embedding_model"],
        "metadata": metadata,
        "media_blob": media_blob,
    }


def list_documents(
    *,
    root: Optional[Path] = None,
    kind: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        where = ""
        params: list[Any] = []
        if kind:
            where = "WHERE kind = ?"
            params.append(kind)
        total = int(con.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0])
        rows = con.execute(
            f"""
            SELECT * FROM documents
            {where}
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
    return {
        "items": [document_summary(row) for row in rows],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


def get_document(document_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        blobs = con.execute(
            """
            SELECT blobs.id, document_blobs.role, blobs.mime_type, blobs.bytes, blobs.sha256
            FROM document_blobs
            JOIN blobs ON document_blobs.blob_id = blobs.id
            WHERE document_blobs.document_id = ?
            ORDER BY document_blobs.role, blobs.id
            """,
            (document_id,),
        ).fetchall()
    payload = parse_object_json(row["json_payload"])
    summary = document_summary(row)
    summary.update(
        {
            "json_payload": payload,
            "text_content": row["text_content"],
            "blobs": [
                {
                    "id": blob["id"],
                    "role": blob["role"],
                    "mime_type": blob["mime_type"],
                    "bytes": blob["bytes"],
                    "sha256": blob["sha256"],
                    "playback_url": f"/api/blobs/{blob['id']}",
                    "download_url": f"/api/blobs/{blob['id']}?download=1",
                }
                for blob in blobs
            ],
        }
    )
    return summary


def get_related_documents(document_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No document found with id {document_id}")
        payload = parse_object_json(row["json_payload"])
        metadata = parse_object_json(row["metadata_json"])
        source_artifact_path = str(metadata.get("source_artifact_path") or payload.get("source_artifact_path") or "")
        source_row = None
        if source_artifact_path:
            source_row = con.execute("SELECT * FROM documents WHERE source_path = ?", (source_artifact_path,)).fetchone()
        derived_rows = con.execute(
            """
            SELECT * FROM documents
            WHERE (
                json_extract(metadata_json, '$.source_artifact_path') = ?
                OR json_extract(json_payload, '$.source_artifact_path') = ?
            )
            AND id != ?
            ORDER BY COALESCE(NULLIF(generated_at, ''), updated_at) DESC, updated_at DESC
            """,
            (row["source_path"], row["source_path"], document_id),
        ).fetchall()
    return {
        "document": document_summary(row),
        "source_artifact_path": source_artifact_path,
        "source_document": document_summary(source_row) if source_row else None,
        "derived_documents": [document_summary(derived_row) for derived_row in derived_rows],
    }


def get_blob(blob_id: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    with connect(root) as con:
        init_db(con)
        row = con.execute("SELECT * FROM blobs WHERE id = ?", (blob_id,)).fetchone()
        if row is None:
            raise TranscriptStoreError(f"No blob found with id {blob_id}")
    path = Path(row["stored_path"])
    if not path.exists() or not path.is_file():
        raise TranscriptStoreError(f"Blob file is missing for id {blob_id}")
    return {
        "id": row["id"],
        "role": row["role"],
        "path": path,
        "mime_type": row["mime_type"] or "application/octet-stream",
        "bytes": int(row["bytes"]),
        "sha256": row["sha256"],
    }


def parse_int(value: str, default: int, *, minimum: int = 0, maximum: int = 500) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def first(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key) or []
    return values[0] if values else default


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def app_intelligence_smoke_status(*, state_root: Path, limit: int = 5) -> dict[str, Any]:
    state_root = state_root.expanduser()
    runs_dir = state_root / "app-intelligence-runs"
    browser_smoke_dir = state_root / APP_BROWSER_SMOKE_DIRNAME
    run_dirs = sorted(
        [
            path
            for path in runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(APP_SMOKE_RUN_PREFIX)
        ] if runs_dir.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    reports = sorted(
        [path for path in browser_smoke_dir.glob("*.json") if path.is_file()] if browser_smoke_dir.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    report_items: list[dict[str, Any]] = []
    for path in reports[:limit]:
        payload = read_json_file(path)
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        screenshot_path = Path(str(payload.get("screenshot_path") or ""))
        report_items.append(
            {
                "path": str(path),
                "status": str(payload.get("status") or "unknown"),
                "run_id": str(payload.get("run_id") or ""),
                "created_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "screenshot_path": str(screenshot_path) if str(screenshot_path) else "",
                "screenshot_exists": bool(str(screenshot_path)) and screenshot_path.exists(),
                "missing_checks": payload.get("missing_checks") if isinstance(payload.get("missing_checks"), list) else [],
                "checks": checks,
            }
        )

    run_items: list[dict[str, Any]] = []
    for path in run_dirs[:limit]:
        ledger = read_json_file(path / "run.json")
        run_items.append(
            {
                "run_id": str(ledger.get("run_id") or path.name),
                "workflow": str(ledger.get("workflow") or ""),
                "status": str(ledger.get("status") or ""),
                "phase": str(ledger.get("phase") or ""),
                "path": str(path),
                "updated_at": str(ledger.get("updated_at") or ""),
            }
        )

    return {
        "schema_version": "transcribe-audio.app-smoke-status.v1",
        "state_root": str(state_root),
        "run_prefix": APP_SMOKE_RUN_PREFIX,
        "runs_dir": str(runs_dir),
        "browser_smoke_dir": str(browser_smoke_dir),
        "latest_report": report_items[0] if report_items else None,
        "reports": report_items,
        "report_count": len(reports),
        "runs": run_items,
        "run_count": len(run_dirs),
        "will_read_artifact_content": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def smoke_jobs_dir(state_root: Path) -> Path:
    return state_root.expanduser() / APP_SMOKE_JOB_DIRNAME


def app_smoke_job_path(state_root: Path, job_id: str) -> Path:
    if not job_id or "/" in job_id or "\\" in job_id or job_id.startswith("."):
        raise ValueError("Invalid smoke job id.")
    return smoke_jobs_dir(state_root) / f"{job_id}.json"


def app_smoke_job_command(
    *,
    job_type: str,
    state_root: Path,
    base_url: str,
    cleanup: bool = True,
    apply_cleanup: bool = False,
) -> tuple[list[str], int, str]:
    repo_root = Path(__file__).resolve().parent
    state_root = state_root.expanduser()
    if job_type == "api_replay_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_app_replay_manifest.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "browser_replay_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_app_replay_manifest_ui.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "first_pass_resume_ui_smoke":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "smoke_first_pass_batch_resume_ui.py"),
            "--base-url",
            base_url.rstrip("/"),
            "--state-root",
            str(state_root),
        ]
        if cleanup:
            command.append("--cleanup")
        return command, APP_SMOKE_JOB_TIMEOUT_SECONDS, APP_SMOKE_JOB_TOKEN
    if job_type == "cleanup_smokes":
        command = [
            sys.executable,
            str(repo_root / "scripts" / "cleanup_app_smokes.py"),
            "--state-root",
            str(state_root),
            "--format",
            "json",
        ]
        if apply_cleanup:
            command.append("--apply")
        return command, 60, APP_SMOKE_CLEANUP_TOKEN if apply_cleanup else APP_SMOKE_JOB_TOKEN
    raise ValueError(f"Unsupported smoke job type: {job_type}")


def write_app_smoke_job(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def cleanup_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_smoke_cleanup_summary(text: str) -> dict[str, Any] | None:
    prefix = "APP_SMOKE_CLEANUP_JSON="
    for line in str(text or "").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "schema_version": str(payload.get("schema_version") or "transcribe-audio.app-smoke-cleanup.v1"),
            "apply": bool(payload.get("apply")),
            "matched_run_count": cleanup_count(payload.get("matched_run_count")),
            "kept_run_count": cleanup_count(payload.get("kept_run_count")),
            "delete_run_count": cleanup_count(payload.get("delete_run_count")),
            "matched_evidence_count": cleanup_count(payload.get("matched_evidence_count")),
            "keep_evidence": cleanup_count(payload.get("keep_evidence")),
            "evidence_days": cleanup_count(payload.get("evidence_days")),
            "delete_evidence_count": cleanup_count(payload.get("delete_evidence_count")),
        }
    return None


def parse_smoke_evidence_summary(text: str) -> dict[str, Any] | None:
    prefixes = [
        "APP_REPLAY_MANIFEST_UI_SMOKE_JSON=",
        "FIRST_PASS_RESUME_UI_SMOKE_JSON=",
    ]
    for line in str(text or "").splitlines():
        prefix = next((item for item in prefixes if line.startswith(item)), "")
        if not prefix:
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        report_path = str(payload.get("report_path") or "")
        screenshot_path = str(payload.get("screenshot_path") or "")
        return {
            "schema_version": str(payload.get("schema_version") or "transcribe-audio.smoke-evidence.v1"),
            "stdout_prefix": prefix.removesuffix("="),
            "status": str(payload.get("status") or "unknown"),
            "report_path": report_path,
            "screenshot_path": screenshot_path,
            "report_url": f"/api/intelligence/smoke-evidence?path={quote(report_path, safe='')}" if report_path else "",
            "screenshot_url": f"/api/intelligence/smoke-evidence?path={quote(screenshot_path, safe='')}" if screenshot_path else "",
            "check_count": len(checks),
            "failed_check_count": len([value for value in checks.values() if value is not True]),
        }
    return None


def summarize_smoke_job(path: Path) -> dict[str, Any]:
    payload = read_json_file(path)
    if not payload:
        return {}
    stdout_path = Path(str(payload.get("stdout_path") or ""))
    stderr_path = Path(str(payload.get("stderr_path") or ""))
    summary = {
        "job_id": str(payload.get("job_id") or path.stem),
        "job_type": str(payload.get("job_type") or ""),
        "status": str(payload.get("status") or "unknown"),
        "created_at": str(payload.get("created_at") or ""),
        "started_at": str(payload.get("started_at") or ""),
        "finished_at": str(payload.get("finished_at") or ""),
        "returncode": payload.get("returncode"),
        "path": str(path),
        "stdout_path": str(stdout_path) if str(stdout_path) else "",
        "stderr_path": str(stderr_path) if str(stderr_path) else "",
        "stdout_exists": bool(str(stdout_path)) and stdout_path.exists(),
        "stderr_exists": bool(str(stderr_path)) and stderr_path.exists(),
        "will_execute_write_bearing_action": bool(payload.get("will_execute_write_bearing_action")),
        "will_execute_external_action": bool(payload.get("will_execute_external_action")),
        "will_read_artifact_content": False,
    }
    cleanup_summary = parse_smoke_cleanup_summary(str(payload.get("stdout_tail") or ""))
    if cleanup_summary is not None:
        summary["cleanup_summary"] = cleanup_summary
    evidence_summary = parse_smoke_evidence_summary(str(payload.get("stdout_tail") or ""))
    if evidence_summary is not None:
        summary["evidence_summary"] = evidence_summary
    return summary


def resolve_smoke_evidence_path(*, state_root: Path, path_value: str) -> Path:
    if not path_value:
        raise ValueError("Missing smoke evidence path.")
    root = (state_root.expanduser() / APP_BROWSER_SMOKE_DIRNAME).resolve()
    candidate = Path(path_value).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError("Smoke evidence path is outside the browser-smokes directory.")
    if resolved.suffix.lower() not in {".json", ".png"}:
        raise ValueError("Smoke evidence must be a JSON report or PNG screenshot.")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    if resolved.stat().st_size > MAX_SMOKE_EVIDENCE_BYTES:
        raise ValueError("Smoke evidence file is too large to serve.")
    return resolved


def resolve_smoke_job_output_path(*, state_root: Path, job: dict[str, Any], stream: str) -> Path:
    if stream not in {"stdout", "stderr"}:
        raise ValueError("stream must be stdout or stderr.")
    root = smoke_jobs_dir(state_root).resolve()
    candidate = Path(str(job.get(f"{stream}_path") or "")).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileNotFoundError(str(candidate)) from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError("Smoke job output path is outside the smoke job directory.")
    return resolved


def read_app_smoke_job_tail(*, state_root: Path, job_id: str, stream: str = "stderr", chars: int = 4000) -> dict[str, Any]:
    job_path = app_smoke_job_path(state_root, job_id)
    job = read_json_file(job_path)
    if not job:
        raise FileNotFoundError(str(job_path))
    output_path = resolve_smoke_job_output_path(state_root=state_root, job=job, stream=stream)
    if not output_path.exists() or not output_path.is_file():
        text = ""
    else:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    limit = max(1, min(chars, MAX_SMOKE_JOB_TAIL_CHARS))
    return {
        "schema_version": "transcribe-audio.app-smoke-job-tail.v1",
        "job": summarize_smoke_job(job_path),
        "job_id": str(job.get("job_id") or job_path.stem),
        "stream": stream,
        "path": str(output_path),
        "exists": output_path.exists(),
        "bytes": output_path.stat().st_size if output_path.exists() else 0,
        "char_count": len(text),
        "tail_chars": limit,
        "tail": text[-limit:],
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
        "will_read_arbitrary_file": False,
    }


def list_app_smoke_jobs(*, state_root: Path, limit: int = 10) -> dict[str, Any]:
    root = smoke_jobs_dir(state_root)
    paths = sorted(
        [path for path in root.glob("*.json") if path.is_file()] if root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        "schema_version": "transcribe-audio.app-smoke-jobs.v1",
        "job_dir": str(root),
        "items": [summarize_smoke_job(path) for path in paths[:limit]],
        "total": len(paths),
        "available_job_types": [
            "api_replay_smoke",
            "browser_replay_smoke",
            "first_pass_resume_ui_smoke",
            "cleanup_smokes",
        ],
        "will_read_artifact_content": False,
    }


def run_app_smoke_job(job_path: Path) -> None:
    payload = read_json_file(job_path)
    if not payload:
        return
    command = payload.get("command") if isinstance(payload.get("command"), list) else []
    timeout = int(payload.get("timeout_seconds") or APP_SMOKE_JOB_TIMEOUT_SECONDS)
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload.update({"status": "running", "started_at": started_at})
    write_app_smoke_job(job_path, payload)
    stdout_path = Path(str(payload.get("stdout_path") or ""))
    stderr_path = Path(str(payload.get("stderr_path") or ""))
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        payload.update(
            {
                "status": "succeeded" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:],
            }
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = subprocess_text(exc.stdout)
        stderr_text = subprocess_text(exc.stderr)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        payload.update(
            {
                "status": "failed",
                "returncode": None,
                "error": f"Timed out after {timeout} seconds.",
                "stdout_tail": stdout_text[-2000:],
                "stderr_tail": stderr_text[-2000:],
            }
        )
    except OSError as exc:
        payload.update({"status": "failed", "returncode": None, "error": str(exc)})
    payload["finished_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_app_smoke_job(job_path, payload)


def enqueue_app_smoke_job(
    *,
    state_root: Path,
    job_type: str,
    approval_token: str,
    base_url: str,
    cleanup: bool = True,
    apply_cleanup: bool = False,
    start_thread: bool = True,
) -> dict[str, Any]:
    command, timeout, required_token = app_smoke_job_command(
        job_type=job_type,
        state_root=state_root,
        base_url=base_url,
        cleanup=cleanup,
        apply_cleanup=apply_cleanup,
    )
    if approval_token != required_token:
        raise ValueError(f"{job_type} requires approval_token={required_token}.")
    root = smoke_jobs_dir(state_root)
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"{job_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    job_path = root / f"{job_id}.json"
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "transcribe-audio.app-smoke-job.v1",
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "created_at": created_at,
        "base_url": base_url.rstrip("/"),
        "command": command,
        "timeout_seconds": timeout,
        "stdout_path": str(root / f"{job_id}.stdout.txt"),
        "stderr_path": str(root / f"{job_id}.stderr.txt"),
        "cleanup": cleanup,
        "apply_cleanup": apply_cleanup,
        "will_execute_external_action": job_type in {"browser_replay_smoke", "first_pass_resume_ui_smoke"},
        "will_execute_write_bearing_action": job_type == "cleanup_smokes" and apply_cleanup,
        "will_read_artifact_content": False,
    }
    write_app_smoke_job(job_path, payload)
    if start_thread:
        threading.Thread(target=run_app_smoke_job, args=(job_path,), daemon=True).start()
    return {
        "schema_version": "transcribe-audio.app-smoke-job-enqueue.v1",
        "job": summarize_smoke_job(job_path),
        "required_approval_token_checked": required_token,
        "will_start_background_job": bool(start_thread),
        "will_execute_arbitrary_shell": False,
    }


def read_app_intelligence_events_file(run_path: Path) -> list[dict[str, Any]]:
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def collect_app_intelligence_artifact_paths(run: dict[str, Any], run_path: Path) -> set[Path]:
    candidates: list[str] = []
    latest_status = run.get("latest_model_turn_status") if isinstance(run.get("latest_model_turn_status"), dict) else {}
    candidates.append(str(latest_status.get("artifact_path") or ""))
    for packet in run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []:
        if isinstance(packet, dict):
            candidates.append(str(packet.get("packet_path") or ""))
            candidates.append(str(packet.get("prompt_path") or ""))
    for decision in run.get("decisions") if isinstance(run.get("decisions"), list) else []:
        if not isinstance(decision, dict):
            continue
        candidates.append(str(decision.get("artifact_path") or ""))
        apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
        candidates.append(str(apply_result.get("artifact_path") or ""))
    for event in read_app_intelligence_events_file(run_path):
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidates.append(str(payload.get("artifact_path") or ""))
        candidates.append(str(payload.get("packet_path") or ""))
        candidates.append(str(payload.get("prompt_path") or ""))

    root = run_path.resolve()
    allowed: set[Path] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = run_path / path
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        allowed.add(resolved)
    return allowed


def app_intelligence_replay_manifest(*, state_root: Path, run_id: str) -> dict[str, Any]:
    shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    run_path = Path(str(shown.get("path") or ""))
    allowed_paths = collect_app_intelligence_artifact_paths(run, run_path)
    events = read_app_intelligence_events_file(run_path)
    seen: set[Path] = set()
    artifacts: list[dict[str, Any]] = []

    def resolve_registered(path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = run_path / path
        resolved = path.resolve()
        if resolved not in allowed_paths:
            return None
        return resolved

    def add_artifact(
        path_value: str,
        *,
        artifact_role: str,
        label: str,
        source: str,
        created_at: str = "",
        event_id: str = "",
        event_type: str = "",
        packet_id: str = "",
        decision_id: str = "",
        codex_turn_id: str = "",
    ) -> None:
        resolved = resolve_registered(path_value)
        if resolved is None or resolved in seen:
            return
        seen.add(resolved)
        exists = resolved.exists() and resolved.is_file()
        suffix = resolved.suffix.lower()
        artifacts.append(
            {
                "artifact_id": f"artifact-{len(artifacts) + 1:03d}",
                "artifact_role": artifact_role,
                "label": label,
                "path": str(resolved),
                "relative_path": str(resolved.relative_to(run_path.resolve())),
                "artifact_type": "json" if suffix == ".json" else "text",
                "exists": exists,
                "bytes": resolved.stat().st_size if exists else None,
                "source": source,
                "created_at": created_at,
                "event_id": event_id,
                "event_type": event_type,
                "packet_id": packet_id,
                "decision_id": decision_id,
                "codex_turn_id": codex_turn_id,
                "can_read_via_artifact_endpoint": exists,
            }
        )

    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        event_type = str(event.get("event_type") or "")
        event_id = str(event.get("event_id") or "")
        created_at = str(event.get("created_at") or "")
        packet_id = str(payload.get("packet_id") or "")
        decision_id = str(payload.get("decision_id") or "")
        codex_turn_id = str(payload.get("codex_turn_id") or "")
        if payload.get("packet_path"):
            add_artifact(
                str(payload.get("packet_path") or ""),
                artifact_role="prompt_packet_json",
                label=f"Prompt packet {packet_id or len(artifacts) + 1}",
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
            )
        if payload.get("prompt_path"):
            add_artifact(
                str(payload.get("prompt_path") or ""),
                artifact_role="prompt_text",
                label=f"Prompt text {packet_id or len(artifacts) + 1}",
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
            )
        artifact_path = str(payload.get("artifact_path") or "")
        if artifact_path:
            if event_type == "model_turn_status_captured":
                role = "model_turn_status"
                label = f"Turn status {codex_turn_id or len(artifacts) + 1}"
            elif event_type == "structured_decision_validated":
                role = "structured_decision_validation"
                label = f"Decision validation {decision_id or len(artifacts) + 1}"
            elif event_type == "structured_decision_applied":
                role = "structured_decision_apply"
                label = f"Decision apply {decision_id or len(artifacts) + 1}"
            elif "preflight" in event_type:
                role = "preflight"
                label = event_type.replace("_", " ").title()
            else:
                role = "event_artifact"
                label = event_type.replace("_", " ").title() or "Event artifact"
            add_artifact(
                artifact_path,
                artifact_role=role,
                label=label,
                source="event_log",
                created_at=created_at,
                event_id=event_id,
                event_type=event_type,
                packet_id=packet_id,
                decision_id=decision_id,
                codex_turn_id=codex_turn_id,
            )

    for packet in run.get("prompt_packets") if isinstance(run.get("prompt_packets"), list) else []:
        if not isinstance(packet, dict):
            continue
        packet_id = str(packet.get("packet_id") or "")
        add_artifact(
            str(packet.get("packet_path") or ""),
            artifact_role="prompt_packet_json",
            label=f"Prompt packet {packet_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(packet.get("created_at") or ""),
            packet_id=packet_id,
        )
        add_artifact(
            str(packet.get("prompt_path") or ""),
            artifact_role="prompt_text",
            label=f"Prompt text {packet_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(packet.get("created_at") or ""),
            packet_id=packet_id,
        )
    latest_status = run.get("latest_model_turn_status") if isinstance(run.get("latest_model_turn_status"), dict) else {}
    add_artifact(
        str(latest_status.get("artifact_path") or ""),
        artifact_role="model_turn_status",
        label=f"Turn status {latest_status.get('codex_turn_id') or len(artifacts) + 1}",
        source="run_ledger",
        created_at=str(latest_status.get("captured_at") or ""),
        codex_turn_id=str(latest_status.get("codex_turn_id") or ""),
    )
    for decision in run.get("decisions") if isinstance(run.get("decisions"), list) else []:
        if not isinstance(decision, dict):
            continue
        decision_id = str(decision.get("decision_id") or "")
        add_artifact(
            str(decision.get("artifact_path") or ""),
            artifact_role="structured_decision_validation",
            label=f"Decision validation {decision_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(decision.get("created_at") or ""),
            decision_id=decision_id,
        )
        apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
        add_artifact(
            str(apply_result.get("artifact_path") or ""),
            artifact_role="structured_decision_apply",
            label=f"Decision apply {decision_id or len(artifacts) + 1}",
            source="run_ledger",
            created_at=str(decision.get("applied_at") or ""),
            decision_id=decision_id,
        )

    return {
        "schema_version": "transcribe-audio.app-intelligence-replay-manifest.v1",
        "run_id": run_id,
        "run_path": str(run_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
        "will_read_artifact_content": False,
    }


def read_app_intelligence_artifact(*, state_root: Path, run_id: str, artifact_path: str) -> dict[str, Any]:
    if not artifact_path:
        raise ValueError("Missing required query parameter: path")
    shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    run_path = Path(str(shown.get("path") or ""))
    requested = Path(artifact_path).expanduser()
    if not requested.is_absolute():
        requested = run_path / requested
    resolved = requested.resolve()
    run_root = run_path.resolve()
    try:
        relative_path = resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("Artifact path resolves outside the App Intelligence run directory.") from exc
    if resolved not in collect_app_intelligence_artifact_paths(run, run_path):
        raise ValueError("Artifact path is not registered in this App Intelligence run ledger or event log.")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("App Intelligence artifact file is missing.")
    size = resolved.stat().st_size
    if size > MAX_APP_ARTIFACT_BYTES:
        raise ValueError(f"Artifact is too large to display through this endpoint: {size} bytes.")
    raw = resolved.read_text(encoding="utf-8")
    parsed_json: Any = None
    artifact_type = "text"
    try:
        parsed_json = json.loads(raw)
        artifact_type = "json"
    except json.JSONDecodeError:
        parsed_json = None
    return {
        "schema_version": "transcribe-audio.app-intelligence-artifact-read.v1",
        "run_id": run_id,
        "path": str(resolved),
        "relative_path": str(relative_path),
        "artifact_type": artifact_type,
        "bytes": size,
        "json": parsed_json,
        "text": raw,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def compact_document_for_prompt(document: dict[str, Any], *, max_text_chars: int = 12000) -> dict[str, Any]:
    text = str(document.get("text_content") or "")
    return {
        "id": document.get("id") or "",
        "kind": document.get("kind") or "",
        "title": document.get("title") or "",
        "source_path": document.get("source_path") or "",
        "generated_at": document.get("generated_at") or "",
        "metadata": document.get("metadata") if isinstance(document.get("metadata"), dict) else {},
        "text_excerpt": text[:max_text_chars],
        "text_truncated": len(text) > max_text_chars,
        "text_chars": len(text),
    }


def model_turn_prompt_text(*, task: str, route: dict[str, Any], document: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the Transcripts App Intelligence worker.",
            "Do not perform external writes. Return structured analysis only.",
            "The host application owns routing, approvals, memory writes, repository writes, and final application.",
            "",
            f"Task: {task}",
            f"Provider route: {json.dumps(route, sort_keys=True, ensure_ascii=False)}",
            "",
            "Document:",
            f"- id: {document.get('id')}",
            f"- kind: {document.get('kind')}",
            f"- title: {document.get('title')}",
            f"- generated_at: {document.get('generated_at')}",
            f"- text_truncated: {document.get('text_truncated')}",
            "",
            "Transcript or readout text:",
            str(document.get("text_excerpt") or ""),
            "",
            "Return JSON with: summary, important_entities, candidate_context_sources, risks, recommended_next_actions, and review_flags.",
        ]
    )


def write_json_file(path: Path, payload: dict[str, Any]) -> Path:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _run_readiness_command(args: list[str], *, timeout: int = 10, probes: Optional[list[str]] = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[:MAX_READINESS_OUTPUT_CHARS],
        }
    stdout = completed.stdout[:MAX_READINESS_OUTPUT_CHARS]
    stderr = completed.stderr[:MAX_READINESS_OUTPUT_CHARS]
    full_text = f"{completed.stdout}\n{completed.stderr}"
    return {
        "args": args,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": len(completed.stdout) > MAX_READINESS_OUTPUT_CHARS,
        "stderr_truncated": len(completed.stderr) > MAX_READINESS_OUTPUT_CHARS,
        "probes": {probe: probe in full_text for probe in probes or []},
    }


def run_codex_command(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "args": args,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc)[:MAX_READINESS_OUTPUT_CHARS],
        }
    return {
        "args": args,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:MAX_READINESS_OUTPUT_CHARS],
        "stderr": completed.stderr[:MAX_READINESS_OUTPUT_CHARS],
        "stdout_truncated": len(completed.stdout) > MAX_READINESS_OUTPUT_CHARS,
        "stderr_truncated": len(completed.stderr) > MAX_READINESS_OUTPUT_CHARS,
    }


def codex_app_server_readiness(*, codex_bin: str = DEFAULT_CODEX_BIN) -> dict[str, Any]:
    resolved = shutil.which(codex_bin) if not Path(codex_bin).is_absolute() else codex_bin
    if not resolved:
        return {
            "id": "codex-app-server",
            "label": "Codex app-server",
            "status": "unavailable",
            "ready": False,
            "control_plane": "codex-app-server",
            "capabilities": {
                "persistent_sessions": True,
                "branching": True,
                "rollback": True,
                "streamed_events": True,
                "structured_decisions": True,
                "remote_transport": False,
            },
            "transports": ["stdio"],
            "checks": {
                "which": {
                    "args": ["which", codex_bin],
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "codex executable not found",
                }
            },
            "notes": [
                "Use codex app-server for supervised long-lived intelligence runs.",
                "Do not expose websocket transport without explicit auth and network-boundary review.",
            ],
        }

    version = _run_readiness_command([resolved, "--version"])
    app_server_help = _run_readiness_command(
        [resolved, "app-server", "--help"],
        probes=["--listen <URL>", "--ws-auth"],
    )
    schema_help = _run_readiness_command([resolved, "app-server", "generate-json-schema", "--help"])
    ts_help = _run_readiness_command([resolved, "app-server", "generate-ts", "--help"])
    help_probes = app_server_help.get("probes") if isinstance(app_server_help.get("probes"), dict) else {}
    ready = all(check.get("ok") for check in [version, app_server_help, schema_help, ts_help])
    return {
        "id": "codex-app-server",
        "label": "Codex app-server",
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "binary": resolved,
        "control_plane": "codex-app-server",
        "version": str(version.get("stdout") or "").strip(),
        "capabilities": {
            "persistent_sessions": True,
            "branching": True,
            "rollback": True,
            "streamed_events": True,
            "structured_decisions": True,
            "schema_generation": bool(schema_help.get("ok")),
            "typescript_generation": bool(ts_help.get("ok")),
            "remote_transport": bool(help_probes.get("--listen <URL>")),
            "websocket_auth": bool(help_probes.get("--ws-auth")),
        },
        "transports": ["stdio", "unix"] + (["websocket"] if help_probes.get("--listen <URL>") else []),
        "recommended_transport": "stdio",
        "checks": {
            "version": version,
            "app_server_help": app_server_help,
            "generate_json_schema_help": schema_help,
            "generate_ts_help": ts_help,
        },
        "notes": [
            "Use codex app-server for supervised long-lived intelligence runs with a host-owned ledger.",
            "Use codex exec for stateless leaf jobs and CI-style analysis.",
            "Do not expose websocket transport without explicit auth and network-boundary review.",
        ],
    }


def intelligence_provider_registry(*, codex_bin: str = DEFAULT_CODEX_BIN) -> dict[str, Any]:
    providers = [
        {
            "id": "openai-compatible",
            "label": "OpenAI-compatible API",
            "status": "configured-by-env",
            "capabilities": ["summarize", "contextual_reread", "classify", "extract"],
            "control_plane": "direct-http",
            "notes": ["Uses OPENAI_API_KEY and optional OPENAI_BASE_URL at execution time."],
        },
        {
            "id": "auracall",
            "label": "AuraCall",
            "status": "configured-by-runtime-env",
            "capabilities": ["summarize", "contextual_reread", "browser_backed_batch"],
            "control_plane": "openai-compatible-or-response-batch",
            "notes": ["Batch actions remain manifest-scoped and approval-gated."],
        },
        {
            "id": "codex-exec",
            "label": "codex exec",
            "status": "configured-by-cli",
            "capabilities": ["summarize", "contextual_reread", "leaf_analysis"],
            "control_plane": "process",
            "notes": ["Best for bounded one-shot jobs, not durable supervised sessions."],
        },
        codex_app_server_readiness(codex_bin=codex_bin),
        {
            "id": "openclaw",
            "label": "OpenClaw transcripts agent",
            "status": "configured-by-openclaw-runtime",
            "capabilities": ["notification", "routing_review", "agentic_context"],
            "control_plane": "openclaw",
            "notes": ["Use repo-local OpenClaw contracts and runtime checks before writes."],
        },
        {
            "id": "graphiti",
            "label": "Graphiti",
            "status": "advisory-memory",
            "capabilities": ["memory_lookup", "matter_candidates", "reviewed_memory_harvest"],
            "control_plane": "mcp",
            "notes": ["Graphiti-derived claims are advisory until verified against source artifacts."],
        },
        {
            "id": "local-embedder",
            "label": "Local embedder",
            "status": "configured-by-store",
            "capabilities": ["embed", "semantic_search"],
            "control_plane": "local-provider",
            "notes": ["Current store search uses configured embedding provider/model metadata."],
        },
    ]
    return {
        "providers": providers,
        "default_supervisor": "codex-app-server",
        "policy": {
            "host_owns_control_flow": True,
            "structured_decisions_required": True,
            "ledger_required_for_write_bearing_runs": True,
            "remote_transport_requires_auth_review": True,
        },
    }


def review_status(count: int, *, stale_count: int = 0, pending_count: int = 0) -> str:
    if count <= 0 and pending_count <= 0:
        return "clear"
    if pending_count > 0:
        return "pending"
    if stale_count >= count:
        return "stale"
    if stale_count > 0:
        return "mixed"
    return "pending"


def filename_conflict_summary(state_root: Path) -> dict[str, Any]:
    review_dir = state_root / "filename-conflict-reviews"
    review_files = sorted(review_dir.glob("filename-conflict-review-*.json"), key=lambda path: path.stat().st_mtime)
    review_files = [path for path in review_files if "audit" not in path.name]
    if not review_files:
        return {
            "id": "filename_conflicts",
            "label": "Filename conflicts",
            "count": 0,
            "status": "clear",
            "detail": "No filename-conflict review file found.",
            "path": "",
            "decisions": {},
            "total_count": 0,
        }
    path = review_files[-1]
    payload = read_json_file(path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    decisions: dict[str, int] = {}
    pending_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "pending")
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision in {"pending", "needs_investigation"}:
            pending_count += 1
    decision_text = ", ".join(f"{key}: {value}" for key, value in sorted(decisions.items())) or "no items"
    return {
        "id": "filename_conflicts",
        "label": "Filename conflicts",
        "count": pending_count,
        "status": "pending" if pending_count else "clear",
        "detail": f"Latest review has {decision_text}.",
        "path": str(path),
        "decisions": decisions,
        "total_count": len(items),
        "updated_at": payload.get("updated_at") or payload.get("created_at") or "",
    }


def route_review_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    review_dir = state_root / "review-queue"
    paths = sorted(review_dir.glob("*.route-review.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    items: list[dict[str, Any]] = []
    for path in paths[:limit]:
        payload = read_json_file(path)
        route_path = Path(str(payload.get("route_decision_path") or "")).expanduser()
        route_exists = bool(str(route_path)) and route_path.exists()
        route_payload = read_json_file(route_path) if route_exists else {}
        selected = route_payload.get("selected_candidate") if isinstance(route_payload.get("selected_candidate"), dict) else {}
        item = {
            "id": path.stem.removesuffix(".route-review"),
            "bucket": "route_reviews",
            "type": "route_review",
            "label": payload.get("selected_label") or selected.get("label") or "Unselected route",
            "reason": payload.get("reason") or "",
            "created_at": payload.get("created_at") or "",
            "review_path": str(path),
            "route_decision_path": str(route_path) if str(route_path) else "",
            "route_decision_exists": route_exists,
            "status": "pending" if route_exists else "stale_reference",
            "confidence": selected.get("confidence"),
            "target_kind": selected.get("target_kind") or "",
        }
        items.append(item)
    return items


def app_intelligence_human_review_items(state_root: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    runs = list_app_intelligence_runs(state_root=state_root, limit=500)
    items: list[dict[str, Any]] = []
    for summary in runs.get("items", []):
        if not isinstance(summary, dict):
            continue
        run_id = str(summary.get("run_id") or "")
        if not run_id:
            continue
        try:
            shown = get_app_intelligence_run(state_root=state_root, run_id=run_id, event_limit=1)
        except (FileNotFoundError, ValueError):
            continue
        run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
        decisions = run.get("decisions") if isinstance(run.get("decisions"), list) else []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            if decision.get("action") != "ask_for_human_review":
                continue
            status = str(decision.get("status") or "")
            if status not in {"validated", "applied"}:
                continue
            human_review = decision.get("human_review") if isinstance(decision.get("human_review"), dict) else {}
            human_review_status = str(human_review.get("status") or "open")
            apply_result = decision.get("apply_result") if isinstance(decision.get("apply_result"), dict) else {}
            item_status = "resolved" if human_review_status == "resolved" else "needs_human_review" if status == "applied" else "pending_apply"
            items.append(
                {
                    "id": f"{run_id}:{decision.get('decision_id') or ''}",
                    "bucket": "app_intelligence_human_review",
                    "type": "app_intelligence_human_review",
                    "label": run.get("purpose") or run.get("workflow") or run_id,
                    "reason": "App Intelligence requested human review.",
                    "created_at": decision.get("applied_at") or decision.get("created_at") or run.get("updated_at") or "",
                    "run_id": run_id,
                    "document_id": run.get("document_id") or "",
                    "workflow": run.get("workflow") or "",
                    "decision_id": decision.get("decision_id") or "",
                    "decision_status": status,
                    "human_review_status": human_review_status,
                    "human_review_note_count": len(human_review.get("notes") if isinstance(human_review.get("notes"), list) else []),
                    "status": item_status,
                    "review_path": str(shown.get("path") or ""),
                    "artifact_path": apply_result.get("artifact_path") or decision.get("artifact_path") or "",
                    "confidence": None,
                    "target_kind": "app_intelligence_run",
                }
            )
    return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)[:limit]


def review_queue_summary(*, state_root: Optional[Path] = None, store_root: Optional[Path] = None, limit: int = 50) -> dict[str, Any]:
    runtime_state_root = (state_root or DEFAULT_STATE_DIR).expanduser()
    route_items = route_review_items(runtime_state_root, limit=limit)
    app_human_review_items = app_intelligence_human_review_items(runtime_state_root, limit=limit)
    stale_count = sum(1 for item in route_items if not item["route_decision_exists"])
    actionable_count = len(route_items) - stale_count
    filename_bucket = filename_conflict_summary(runtime_state_root)
    legacy_payload = legacy_enrichment_queue(root=store_root, pending_only=True)
    legacy_count = int(legacy_payload.get("selected_count") or 0)
    route_bucket = {
        "id": "route_reviews",
        "label": "Route reviews",
        "count": actionable_count,
        "status": review_status(len(route_items), stale_count=stale_count),
        "detail": f"{actionable_count} actionable, {stale_count} stale local references.",
        "total_count": len(route_items),
        "stale_count": stale_count,
    }
    legacy_bucket = {
        "id": "first_pass_summaries",
        "label": "First-pass summaries",
        "count": legacy_count,
        "status": "pending" if legacy_count else "clear",
        "detail": "Stored transcripts waiting for first-pass summaries.",
        "duplicate_count": legacy_payload.get("duplicate_count") or 0,
        "sample_items": [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "generated_at": item.get("generated_at"),
                "has_media_blob": item.get("has_media_blob"),
            }
            for item in legacy_payload.get("items", [])[:5]
            if isinstance(item, dict)
        ],
    }
    app_human_review_bucket = {
        "id": "app_intelligence_human_review",
        "label": "App Intelligence review",
        "count": sum(1 for item in app_human_review_items if item.get("status") != "resolved"),
        "status": "pending" if any(item.get("status") != "resolved" for item in app_human_review_items) else "clear",
        "detail": f"{sum(1 for item in app_human_review_items if item.get('status') != 'resolved')} App Intelligence human-review decisions need operator attention.",
        "pending_apply_count": sum(1 for item in app_human_review_items if item.get("status") == "pending_apply"),
        "needs_review_count": sum(1 for item in app_human_review_items if item.get("status") == "needs_human_review"),
    }
    buckets = [
        route_bucket,
        app_human_review_bucket,
        filename_bucket,
        legacy_bucket,
        {
            "id": "memory_harvest",
            "label": "Memory harvest",
            "count": 0,
            "status": "gated",
            "detail": "Requires explicit review file approval before live Graphiti writes.",
        },
        {
            "id": "speaker_ids",
            "label": "Speaker IDs",
            "count": 0,
            "status": "planned",
            "detail": "Contact dedupe and speaker assignment tables are planned in P09.",
        },
    ]
    return {
        "state_dir": str(runtime_state_root),
        "store_dir": str(store_dir(store_root)),
        "limit": limit,
        "buckets": buckets,
        "items": [*app_human_review_items, *route_items][:limit],
        "total_open": sum(int(bucket.get("count") or 0) for bucket in buckets),
    }


def default_prepare_manifest_path(state_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return state_root.expanduser() / "first-pass-summary-batches" / f"first-pass-summary-prepare-{stamp}.json"


def resolve_batch_manifest_path(state_root: Path, path_text: str) -> Path:
    if not path_text:
        raise ValueError("Missing required manifest path.")
    root = (state_root.expanduser() / "first-pass-summary-batches").resolve()
    path = Path(path_text).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest path is outside the first-pass summary batch directory.") from exc
    if not path.exists() or not path.is_file():
        raise TranscriptStoreError(f"Manifest not found: {path}")
    return path


def batch_status_counts(batch_status: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw_counts = batch_status.get("counts")
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            if isinstance(value, int):
                counts[str(key)] = value
        if counts:
            return counts
    for job in batch_status.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def batch_action_response(
    *,
    action: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    status: str,
    batch_status: Optional[dict[str, Any]] = None,
    materialized: Optional[list[dict[str, Any]]] = None,
    materialization_errors: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    batch = manifest.get("batch") if isinstance(manifest.get("batch"), dict) else {}
    batch_payload = manifest.get("batch_payload") if isinstance(manifest.get("batch_payload"), dict) else {}
    requests = batch_payload.get("requests") if isinstance(batch_payload.get("requests"), list) else []
    first_request = requests[0] if requests and isinstance(requests[0], dict) else {}
    output_contract = (
        (first_request.get("metadata") or {}).get("outputContract")
        if isinstance(first_request.get("metadata"), dict)
        else {}
    )
    return {
        "action": action,
        "bucket": "first_pass_summaries",
        "status": status,
        "manifest": str(manifest_path),
        "request_count": int(manifest.get("request_count") or 0),
        "dry_run": bool(manifest.get("dry_run")),
        "batch_id": batch.get("id") if batch else None,
        "workflow": (batch_payload.get("metadata") or {}).get("workflow") if isinstance(batch_payload.get("metadata"), dict) else "",
        "artifact_file": output_contract.get("artifactFileName") if isinstance(output_contract, dict) else "",
        "batch_status": batch_status,
        "batch_counts": batch_status_counts(batch_status or {}),
        "materialized": materialized or [],
        "materialization_errors": materialization_errors or [],
    }


def summarize_first_pass_batch_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    batch_payload = payload.get("batch_payload") if isinstance(payload.get("batch_payload"), dict) else {}
    last_status = payload.get("last_status") if isinstance(payload.get("last_status"), dict) else {}
    materialized = payload.get("materialized") if isinstance(payload.get("materialized"), list) else []
    materialization_errors = payload.get("materialization_errors") if isinstance(payload.get("materialization_errors"), list) else []
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except OSError:
        mtime = ""
    status = str(last_status.get("status") or ("submitted" if batch else "prepared"))
    return {
        "schema_version": "transcribe-audio.first-pass-summary-batch-manifest-summary.v1",
        "manifest": str(path),
        "updated_at": mtime,
        "status": status,
        "request_count": int(payload.get("request_count") or 0),
        "dry_run": bool(payload.get("dry_run")),
        "batch_id": str(batch.get("id") or ""),
        "workflow": (batch_payload.get("metadata") or {}).get("workflow") if isinstance(batch_payload.get("metadata"), dict) else "",
        "batch_counts": batch_status_counts(last_status),
        "materialized_count": len(materialized),
        "materialization_error_count": len(materialization_errors),
        "will_read_request_payloads": False,
        "will_read_transcript_content": False,
    }


def list_first_pass_summary_batch_manifests(*, state_root: Path, limit: int = 10) -> dict[str, Any]:
    root = state_root.expanduser() / "first-pass-summary-batches"
    paths = sorted(
        [path for path in root.glob("*.json") if path.is_file()] if root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items = [summarize_first_pass_batch_manifest(path, read_json_file(path)) for path in paths[:limit]]
    return {
        "schema_version": "transcribe-audio.first-pass-summary-batch-manifests.v1",
        "manifest_dir": str(root),
        "items": items,
        "total": len(paths),
        "limit": limit,
        "will_read_request_payloads": False,
        "will_read_transcript_content": False,
        "will_execute_external_action": False,
        "will_execute_write_bearing_action": False,
    }


def prepare_first_pass_summary_batch(
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    limit: int = 5,
    store: bool = True,
    model: str = "",
    manifest: Optional[Path] = None,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    manifest_path = manifest or default_prepare_manifest_path(state_root)
    args = argparse.Namespace(
        command="prepare",
        env_file=env_file,
        base_url=None,
        api_key=None,
        store_dir=store_root,
        limit=limit,
        model=model or None,
        dispatch_team=None,
        all=False,
        no_dedupe=False,
        manifest=manifest_path,
        store=store,
        max_concurrent_runs=auracall_legacy_enrichment_batch.DEFAULT_MAX_CONCURRENT_RUNS,
        max_browser_interactions_per_minute=auracall_legacy_enrichment_batch.DEFAULT_MAX_BROWSER_INTERACTIONS_PER_MINUTE,
    )
    exit_code = auracall_legacy_enrichment_batch.enqueue(args, force_dry_run=True)
    if exit_code != 0:
        raise TranscriptStoreError(f"First-pass summary prepare failed with exit code {exit_code}")
    payload = read_json_file(manifest_path)
    return batch_action_response(
        action="prepare_first_pass_summary_batch",
        manifest_path=manifest_path.expanduser(),
        manifest=payload,
        status="prepared",
    )


def submit_first_pass_summary_batch(
    *,
    state_root: Path,
    env_file: Path,
    manifest: str,
    approval_token: str,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    if approval_token != "SUBMIT_FIRST_PASS_SUMMARY_BATCH":
        raise ValueError("Submit requires approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH.")
    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    if payload.get("batch"):
        return batch_action_response(
            action="submit_first_pass_summary_batch",
            manifest_path=manifest_path,
            manifest=payload,
            status="already_submitted",
        )
    batch_payload = payload.get("batch_payload")
    if not isinstance(batch_payload, dict):
        raise ValueError("Manifest does not include a prepared batch payload.")
    if int(payload.get("request_count") or 0) <= 0:
        raise ValueError("Manifest has no prepared requests to submit.")
    args = argparse.Namespace(env_file=env_file, base_url=None, api_key=None)
    env = auracall_legacy_enrichment_batch.runtime_env(args)
    payload["batch"] = auracall_legacy_enrichment_batch.post_batch(
        str(payload["batch_url"]),
        auracall_legacy_enrichment_batch.resolve_api_key(args, env),
        batch_payload,
    )
    payload["dry_run"] = False
    payload["submitted_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_json_file(manifest_path, payload)
    return batch_action_response(
        action="submit_first_pass_summary_batch",
        manifest_path=manifest_path,
        manifest=payload,
        status="submitted",
    )


def first_pass_summary_batch_status(
    *,
    state_root: Path,
    store_root: Path,
    env_file: Path,
    manifest: str,
    materialize: bool = False,
) -> dict[str, Any]:
    from scripts import auracall_legacy_enrichment_batch

    manifest_path = resolve_batch_manifest_path(state_root, manifest)
    payload = read_json_file(manifest_path)
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    batch_id = str(batch.get("id") or "")
    if not batch_id:
        return batch_action_response(
            action="first_pass_summary_batch_status",
            manifest_path=manifest_path,
            manifest=payload,
            status="prepared",
        )
    args = argparse.Namespace(env_file=env_file, base_url=None, api_key=None, store=bool(payload.get("store")), store_dir=store_root, output_dir=None)
    env = auracall_legacy_enrichment_batch.runtime_env(args)
    batch_url = f"{str(payload['batch_url']).rstrip('/')}/{batch_id}"
    batch_status = auracall_legacy_enrichment_batch.read_json_url(
        batch_url,
        auracall_legacy_enrichment_batch.resolve_api_key(args, env),
    )
    materialized: list[dict[str, Any]] = []
    materialization_errors: list[dict[str, Any]] = []
    if materialize:
        materialized, materialization_errors = auracall_legacy_enrichment_batch.materialize_completed(args, payload, batch_status)
    payload["last_status"] = batch_status
    if materialized:
        payload["materialized"] = materialized
    if materialization_errors:
        payload["materialization_errors"] = materialization_errors
    write_json_file(manifest_path, payload)
    return batch_action_response(
        action="first_pass_summary_batch_status",
        manifest_path=manifest_path,
        manifest=payload,
        status=str(batch_status.get("status") or "unknown"),
        batch_status=batch_status,
        materialized=materialized,
        materialization_errors=materialization_errors,
    )


def parse_range_header(header: str, size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes="):
        return None
    value = header.removeprefix("bytes=").split(",", 1)[0].strip()
    if not value or "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    if start_text == "":
        suffix = parse_int(end_text, 0, minimum=0, maximum=size)
        if suffix <= 0:
            return None
        return max(size - suffix, 0), size - 1
    start = parse_int(start_text, 0, minimum=0, maximum=max(size - 1, 0))
    end = parse_int(end_text, size - 1, minimum=start, maximum=max(size - 1, 0)) if end_text else size - 1
    if start > end:
        return None
    return start, end


class TranscriptApiHandler(BaseHTTPRequestHandler):
    server_version = "TranscriptApi/0.1"

    @property
    def store_root(self) -> Path:
        return self.server.store_root  # type: ignore[attr-defined]

    @property
    def embedding_provider(self) -> str:
        return self.server.embedding_provider  # type: ignore[attr-defined]

    @property
    def embedding_model(self) -> str:
        return self.server.embedding_model  # type: ignore[attr-defined]

    @property
    def state_root(self) -> Path:
        return self.server.state_root  # type: ignore[attr-defined]

    def local_base_url(self) -> str:
        port = self.server.server_address[1]  # type: ignore[attr-defined]
        return f"http://127.0.0.1:{port}"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self.write_json({"status": "ok", "store_dir": str(self.store_root), "db_path": str(db_path(self.store_root))})
                return
            if parsed.path == "/api/library":
                self.write_json(
                    list_documents(
                        root=self.store_root,
                        kind=first(params, "kind"),
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                        offset=parse_int(first(params, "offset"), 0, minimum=0, maximum=100000),
                    )
                )
                return
            if parsed.path == "/api/review-queue":
                self.write_json(
                    review_queue_summary(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                    )
                )
                return
            if parsed.path == "/api/intelligence/providers":
                self.write_json(intelligence_provider_registry(codex_bin=self.server.codex_bin))  # type: ignore[attr-defined]
                return
            if parsed.path == "/api/intelligence/smokes":
                self.write_json(
                    app_intelligence_smoke_status(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 5, minimum=1, maximum=50),
                    )
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/manifests":
                self.write_json(
                    list_first_pass_summary_batch_manifests(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                    )
                )
                return
            if parsed.path == "/api/intelligence/smoke-jobs":
                self.write_json(
                    list_app_smoke_jobs(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                    )
                )
                return
            if parsed.path == "/api/intelligence/smoke-evidence":
                self.write_smoke_evidence(first(params, "path"))
                return
            if parsed.path.startswith("/api/intelligence/smoke-jobs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[4] == "tail":
                    self.write_json(
                        read_app_smoke_job_tail(
                            state_root=self.state_root,
                            job_id=parts[3],
                            stream=first(params, "stream", "stderr"),
                            chars=parse_int(
                                first(params, "chars"),
                                4000,
                                minimum=1,
                                maximum=MAX_SMOKE_JOB_TAIL_CHARS,
                            ),
                        )
                    )
                    return
            if parsed.path == "/api/intelligence/config":
                self.write_json(intelligence_config.all_task_configs())
                return
            if parsed.path == "/api/intelligence/runs":
                self.write_json(
                    list_app_intelligence_runs(
                        state_root=self.state_root,
                        limit=parse_int(first(params, "limit"), 50, minimum=1, maximum=200),
                    )
                )
                return
            if parsed.path.startswith("/api/intelligence/runs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 6 and parts[4] == "prompt-packets":
                    self.write_json(
                        read_app_intelligence_model_turn_packet(
                            state_root=self.state_root,
                            run_id=parts[3],
                            packet_id=parts[5],
                        )
                    )
                    return
                if len(parts) == 5 and parts[4] == "replay-manifest":
                    self.write_json(
                        app_intelligence_replay_manifest(
                            state_root=self.state_root,
                            run_id=parts[3],
                        )
                    )
                    return
                if len(parts) == 5 and parts[4] == "artifacts":
                    self.write_json(
                        read_app_intelligence_artifact(
                            state_root=self.state_root,
                            run_id=parts[3],
                            artifact_path=first(params, "path"),
                        )
                    )
                    return
                if len(parts) == 4:
                    self.write_json(
                        get_app_intelligence_run(
                            state_root=self.state_root,
                            run_id=parts[3],
                            event_limit=parse_int(first(params, "event_limit"), 50, minimum=0, maximum=500),
                        )
                    )
                    return
            if parsed.path == "/api/search":
                query = first(params, "q") or first(params, "query")
                if not query:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required query parameter: q")
                    return
                self.write_json(
                    {
                        "query": query,
                        "results": search_store(
                            query,
                            root=self.store_root,
                            kind=first(params, "kind"),
                            limit=parse_int(first(params, "limit"), 10, minimum=1, maximum=100),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        ),
                    }
                )
                return
            if parsed.path.startswith("/api/documents/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_json(get_document(parts[2], root=self.store_root))
                    return
                if len(parts) == 4 and parts[3] == "related":
                    self.write_json(get_related_documents(parts[2], root=self.store_root))
                    return
                if len(parts) == 4 and parts[3] == "context":
                    chunk_text = first(params, "chunk_index")
                    self.write_json(
                        context_for_document(
                            parts[2],
                            root=self.store_root,
                            chunk_index=int(chunk_text) if chunk_text else None,
                            context_chunks=parse_int(first(params, "context_chunks"), 1, minimum=0, maximum=10),
                            embedding_provider=self.embedding_provider,
                            embedding_model=self.embedding_model,
                        )
                    )
                    return
            if parsed.path.startswith("/api/blobs/"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    self.write_blob(parts[2], download=first(params, "download") in {"1", "true", "yes"})
                    return
            if parsed.path.startswith("/api/"):
                self.write_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if self.write_static(parsed.path):
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except TranscriptStoreError as exc:
            self.write_error(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError as exc:
            self.write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/review-queue/first-pass-summaries/prepare":
                body = self.read_json_body()
                limit = parse_int(str(body.get("limit") or ""), 5, minimum=1, maximum=50)
                store = bool(body.get("store", True))
                model = str(body.get("model") or "")
                self.write_json(
                    prepare_first_pass_summary_batch(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        limit=limit,
                        store=store,
                        model=model,
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/submit":
                body = self.read_json_body()
                self.write_json(
                    submit_first_pass_summary_batch(
                        state_root=self.state_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        manifest=str(body.get("manifest") or ""),
                        approval_token=str(body.get("approval_token") or ""),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/review-queue/first-pass-summaries/status":
                body = self.read_json_body()
                self.write_json(
                    first_pass_summary_batch_status(
                        state_root=self.state_root,
                        store_root=self.store_root,
                        env_file=self.server.batch_env_file,  # type: ignore[attr-defined]
                        manifest=str(body.get("manifest") or ""),
                        materialize=bool(body.get("materialize", False)),
                    )
                )
                return
            if parsed.path == "/api/intelligence/config/preview":
                body = self.read_json_body()
                self.write_json(
                    intelligence_config.preview_config_update(
                        task=str(body.get("task") or ""),
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                    )
                )
                return
            if parsed.path == "/api/intelligence/config/apply":
                body = self.read_json_body()
                self.write_json(
                    intelligence_config.apply_config_update(
                        task=str(body.get("task") or ""),
                        update=body.get("update") if isinstance(body.get("update"), dict) else {},
                        approval_token=str(body.get("approval_token") or ""),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/intelligence/smoke-jobs":
                body = self.read_json_body()
                self.write_json(
                    enqueue_app_smoke_job(
                        state_root=self.state_root,
                        job_type=str(body.get("job_type") or ""),
                        approval_token=str(body.get("approval_token") or ""),
                        base_url=str(body.get("base_url") or self.local_base_url()),
                        cleanup=bool(body.get("cleanup", True)),
                        apply_cleanup=bool(body.get("apply_cleanup", False)),
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/intelligence/runs/prepare":
                body = self.read_json_body()
                workflow = str(body.get("workflow") or "").strip()
                purpose = str(body.get("purpose") or "").strip()
                if not workflow:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required field: workflow")
                    return
                if not purpose:
                    self.write_error(HTTPStatus.BAD_REQUEST, "Missing required field: purpose")
                    return
                self.write_json(
                    # Explicit API provider wins; otherwise use task routing.
                    create_app_intelligence_run(
                        state_root=self.state_root,
                        workflow=workflow,
                        purpose=purpose,
                        document_id=str(body.get("document_id") or ""),
                        provider=str(
                            body.get("provider")
                            or intelligence_config.resolve_task_config(
                                str(body.get("task") or intelligence_config.TASK_APP_SUPERVISOR)
                            ).provider
                        ),
                        created_by=str(body.get("created_by") or "operator"),
                        run_id=str(body.get("run_id") or ""),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/send-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "prompt-packets":
                    body = self.read_json_body()
                    preflight = preflight_app_intelligence_model_turn_send(
                        state_root=self.state_root,
                        run_id=parts[3],
                        packet_id=parts[5],
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    self.write_json(preflight, status=HTTPStatus.OK if preflight.get("ok") else HTTPStatus.CONFLICT)
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/send"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "prompt-packets":
                    body = self.read_json_body()
                    run_id = parts[3]
                    packet_id = parts[5]
                    preflight = preflight_app_intelligence_model_turn_send(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    if not preflight.get("ok"):
                        self.write_json(preflight, status=HTTPStatus.CONFLICT)
                        return
                    packet_review = read_app_intelligence_model_turn_packet(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                    )
                    current = get_app_intelligence_run(state_root=self.state_root, run_id=run_id, event_limit=1)
                    current_run = current.get("run") if isinstance(current.get("run"), dict) else {}
                    current_state = current_run.get("state") if isinstance(current_run.get("state"), dict) else {}
                    route = packet_review.get("packet", {}).get("route") if isinstance(packet_review.get("packet"), dict) else {}
                    model = str(route.get("model") or "") if isinstance(route, dict) else ""
                    try:
                        app_server_result = codex_app_server_client.start_model_turn(
                            codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                            cwd=Path(__file__).resolve().parent,
                            prompt_text=str(packet_review.get("prompt_text") or ""),
                            model=model,
                            existing_thread_id=str(current_state.get("active_codex_thread_id") or ""),
                            timeout_seconds=float(body.get("timeout_seconds") or 30),
                        )
                    except Exception as exc:
                        event = record_app_intelligence_model_turn_failed(
                            state_root=self.state_root,
                            run_id=run_id,
                            packet_id=packet_id,
                            error=str(exc),
                        )
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-model-turn-send.v1",
                                "action": "send_model_turn",
                                "ok": False,
                                "preflight": preflight,
                                "event": event,
                                "will_execute_downstream_action": False,
                                "error": str(exc),
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    for codex_event in app_server_result.get("events", []):
                        if isinstance(codex_event, dict):
                            append_app_intelligence_codex_event(state_root=self.state_root, run_id=run_id, payload=codex_event)
                    started = record_app_intelligence_model_turn_started(
                        state_root=self.state_root,
                        run_id=run_id,
                        packet_id=packet_id,
                        thread_id=str(app_server_result.get("thread_id") or ""),
                        turn_id=str(app_server_result.get("turn_id") or ""),
                        app_server_result={
                            "thread_start_response": app_server_result.get("thread_start_response") or {},
                            "turn_start_response": app_server_result.get("turn_start_response") or {},
                            "captured_event_count": len(app_server_result.get("events") or []),
                        },
                    )
                    self.write_json(
                        {
                            "schema_version": "transcribe-audio.app-intelligence-model-turn-send.v1",
                            "action": "send_model_turn",
                            "ok": True,
                            "packet_id": packet_id,
                            "codex_thread_id": app_server_result.get("thread_id") or "",
                            "codex_turn_id": app_server_result.get("turn_id") or "",
                            "captured_event_count": len(app_server_result.get("events") or []),
                            "preflight": preflight,
                            "will_execute_downstream_action": False,
                            **started,
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/turn-status"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    run_id = parts[3]
                    current = get_app_intelligence_run(state_root=self.state_root, run_id=run_id, event_limit=1)
                    run = current.get("run") if isinstance(current.get("run"), dict) else {}
                    state = run.get("state") if isinstance(run.get("state"), dict) else {}
                    thread_id = str(body.get("thread_id") or state.get("active_codex_thread_id") or "")
                    turn_id = str(body.get("turn_id") or state.get("latest_turn_id") or "")
                    if not thread_id or not turn_id:
                        self.write_error(HTTPStatus.BAD_REQUEST, "No active Codex thread/turn is recorded for this run.")
                        return
                    try:
                        status_result = codex_app_server_client.inspect_model_turn(
                            codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                            thread_id=thread_id,
                            turn_id=turn_id,
                            timeout_seconds=float(body.get("timeout_seconds") or 30),
                        )
                    except Exception as exc:
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-model-turn-status.v1",
                                "action": "capture_model_turn_status",
                                "ok": False,
                                "codex_thread_id": thread_id,
                                "codex_turn_id": turn_id,
                                "will_execute_structured_decision": False,
                                "error": str(exc),
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    for codex_event in status_result.get("events", []):
                        if isinstance(codex_event, dict):
                            append_app_intelligence_codex_event(state_root=self.state_root, run_id=run_id, payload=codex_event)
                    captured = record_app_intelligence_model_turn_status(
                        state_root=self.state_root,
                        run_id=run_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status_payload=status_result,
                        approval_token=str(body.get("approval_token") or ""),
                    )
                    self.write_json(
                        {
                            **captured,
                            "codex_thread_id": thread_id,
                            "codex_turn_id": turn_id,
                            "captured_event_count": len(status_result.get("events") or []),
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/structured-decision/validate"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 6:
                    body = self.read_json_body()
                    self.write_json(
                        validate_app_intelligence_structured_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/apply"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        apply_app_intelligence_structured_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/human-review"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        record_app_intelligence_human_review_decision(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            review_action=str(body.get("review_action") or ""),
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/fork-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        preflight_app_intelligence_fork_branches(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/rollback-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 7 and parts[4] == "structured-decisions":
                    body = self.read_json_body()
                    self.write_json(
                        preflight_app_intelligence_rollback(
                            state_root=self.state_root,
                            run_id=parts[3],
                            decision_id=parts[5],
                            approval_token=str(body.get("approval_token") or ""),
                            reviewer=str(body.get("reviewer") or "operator"),
                            note=str(body.get("note") or ""),
                        ),
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/session-start-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    provider = codex_app_server_readiness(codex_bin=self.server.codex_bin)  # type: ignore[attr-defined]
                    append_event_log = bool(body.get("append_event", False))
                    self.write_json(
                        preflight_app_intelligence_session_start(
                            state_root=self.state_root,
                            run_id=parts[3],
                            provider_ready=bool(provider.get("ready")),
                            provider_status=str(provider.get("status") or ""),
                            approval_token=str(body.get("approval_token") or ""),
                            append_event_log=append_event_log,
                        ),
                        status=HTTPStatus.ACCEPTED if append_event_log else HTTPStatus.OK,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/session-start"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    transport = str(body.get("transport") or "stdio")
                    approval_token = str(body.get("approval_token") or "")
                    provider = codex_app_server_readiness(codex_bin=self.server.codex_bin)  # type: ignore[attr-defined]
                    preflight = preflight_app_intelligence_session_start(
                        state_root=self.state_root,
                        run_id=parts[3],
                        provider_ready=bool(provider.get("ready")),
                        provider_status=str(provider.get("status") or ""),
                        approval_token=approval_token,
                    )
                    if not preflight.get("ok"):
                        self.write_json(preflight, status=HTTPStatus.CONFLICT)
                        return
                    record_app_intelligence_session_start_requested(
                        state_root=self.state_root,
                        run_id=parts[3],
                        transport=transport,
                        approval_token=approval_token,
                    )
                    start_result = run_codex_command(
                        [self.server.codex_bin, "app-server", "daemon", "start"],  # type: ignore[attr-defined]
                        timeout=30,
                    )
                    if not start_result.get("ok"):
                        event = record_app_intelligence_session_start_failed(
                            state_root=self.state_root,
                            run_id=parts[3],
                            transport=transport,
                            error=str(start_result.get("stderr") or start_result.get("stdout") or "unknown start failure"),
                        )
                        self.write_json(
                            {
                                "schema_version": "transcribe-audio.app-intelligence-session-start.v1",
                                "action": "start_app_server_session",
                                "ok": False,
                                "will_start_model_turn": False,
                                "preflight": preflight,
                                "start_result": start_result,
                                "event": event,
                            },
                            status=HTTPStatus.BAD_GATEWAY,
                        )
                        return
                    version_result = run_codex_command(
                        [self.server.codex_bin, "app-server", "daemon", "version"],  # type: ignore[attr-defined]
                        timeout=15,
                    )
                    started = mark_app_intelligence_session_started(
                        state_root=self.state_root,
                        run_id=parts[3],
                        transport=transport,
                        codex_bin=str(self.server.codex_bin),  # type: ignore[attr-defined]
                        start_result=start_result,
                        version_result=version_result,
                    )
                    self.write_json(
                        {
                            "schema_version": "transcribe-audio.app-intelligence-session-start.v1",
                            "action": "start_app_server_session",
                            "ok": True,
                            "will_start_model_turn": False,
                            "transport": transport,
                            "preflight": preflight,
                            "start_result": start_result,
                            "version_result": version_result,
                            **started,
                        },
                        status=HTTPStatus.ACCEPTED,
                    )
                    return
            if parsed.path.startswith("/api/intelligence/runs/") and parsed.path.endswith("/model-turn-preflight"):
                parts = [unquote(part) for part in parsed.path.split("/") if part]
                if len(parts) == 5:
                    body = self.read_json_body()
                    shown = get_app_intelligence_run(state_root=self.state_root, run_id=parts[3], event_limit=1)
                    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
                    task = str(body.get("task") or run.get("workflow") or intelligence_config.TASK_APP_SUPERVISOR)
                    document_id = str(body.get("document_id") or run.get("document_id") or "")
                    if not document_id:
                        self.write_error(HTTPStatus.BAD_REQUEST, "Missing required document_id for model-turn preflight")
                        return
                    document = compact_document_for_prompt(get_document(document_id, root=self.store_root))
                    route = intelligence_config.resolve_task_config(task).to_dict()
                    prompt_text = model_turn_prompt_text(task=task, route=route, document=document)
                    self.write_json(
                        prepare_app_intelligence_model_turn_packet(
                            state_root=self.state_root,
                            run_id=parts[3],
                            task=task,
                            route=route,
                            document=document,
                            prompt_text=prompt_text,
                            approval_token=str(body.get("approval_token") or ""),
                        ),
                        status=HTTPStatus.CREATED,
                    )
                    return
            if parsed.path.startswith("/api/"):
                self.write_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except (TranscriptStoreError, TranscriptionError, OSError, json.JSONDecodeError) as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))

    def read_json_body(self) -> dict[str, Any]:
        length = parse_int(self.headers.get("Content-Length", "0"), 0, minimum=0, maximum=1024 * 1024)
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request JSON body must be an object.")
        return payload

    def write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, status: HTTPStatus, message: str) -> None:
        self.write_json({"error": message, "status": status.value}, status=status)

    def write_smoke_evidence(self, path_value: str) -> None:
        path = resolve_smoke_evidence_path(state_root=self.state_root, path_value=path_value)
        body = path.read_bytes()
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_static(self, request_path: str) -> bool:
        static_dir = self.server.static_dir  # type: ignore[attr-defined]
        if static_dir is None:
            return False
        static_root = Path(static_dir)
        if not static_root.exists() or not static_root.is_dir():
            return False
        relative = unquote(request_path).lstrip("/")
        target = static_root / relative if relative else static_root / "index.html"
        if not target.exists() or target.is_dir():
            target = static_root / "index.html"
        try:
            resolved = target.resolve()
            resolved.relative_to(static_root.resolve())
        except ValueError:
            self.write_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return True
        if not resolved.exists() or not resolved.is_file():
            return False
        body = resolved.read_bytes()
        mime_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-store" if resolved.name == "index.html" else "public, max-age=3600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def write_blob(self, blob_id: str, *, download: bool = False) -> None:
        blob = get_blob(blob_id, root=self.store_root)
        size = int(blob["bytes"])
        file_range = parse_range_header(self.headers.get("Range", ""), size)
        status = HTTPStatus.PARTIAL_CONTENT if file_range else HTTPStatus.OK
        start, end = file_range if file_range else (0, max(size - 1, 0))
        length = max(end - start + 1, 0)

        self.send_response(status)
        self.send_header("Content-Type", str(blob["mime_type"]))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if file_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{blob_id}"')
        self.end_headers()

        with Path(blob["path"]).open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class TranscriptApiServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store_root: Path,
        embedding_provider: str,
        embedding_model: str,
        state_root: Path = DEFAULT_STATE_DIR,
        batch_env_file: Path = DEFAULT_BATCH_ENV_FILE,
        codex_bin: str = DEFAULT_CODEX_BIN,
        quiet: bool = False,
        static_dir: Optional[Path] = DEFAULT_STATIC_DIR,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store_root = store_root
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.state_root = state_root.expanduser()
        self.batch_env_file = batch_env_file.expanduser()
        self.codex_bin = codex_bin
        self.quiet = quiet
        self.static_dir = static_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local transcript review API.")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--batch-env-file", type=Path, default=DEFAULT_BATCH_ENV_FILE)
    parser.add_argument("--codex-bin", default=os.environ.get("TRANSCRIPTS_CODEX_BIN", DEFAULT_CODEX_BIN))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def serve(args: argparse.Namespace) -> None:
    root = store_dir(args.store_dir)
    with connect(root) as con:
        init_db(con)
    server = TranscriptApiServer(
        (args.host, args.port),
        TranscriptApiHandler,
        store_root=root,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        state_root=args.state_dir,
        batch_env_file=args.batch_env_file,
        codex_bin=args.codex_bin,
        quiet=bool(args.quiet),
        static_dir=None if args.no_static else args.static_dir,
    )
    print(f"Serving transcript API on http://{args.host}:{args.port} using {db_path(root)}")
    server.serve_forever()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        serve(args)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
