#!/usr/bin/env python3
"""
Watch one or more directories for newly finished audio/video files and hand them off to one
of the transcription CLIs once the files appear stable.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from transcript_store import TranscriptStoreError, ingest_artifact

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = SCRIPT_DIR / ".openclaw" / "watch_transcriptions_state.json"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "watch_transcriptions.json"
DEFAULT_SETTLE_SECONDS = 90.0
DEFAULT_SCAN_INTERVAL = 30.0
DEFAULT_FAILURE_RETRY_SECONDS = 900.0
DEFAULT_MIN_AGE_SECONDS = 15.0
DEFAULT_HEARTBEAT_SECONDS = 600.0
DEFAULT_NO_PROGRESS_RESTART_SECONDS = 1800.0

AUDIO_VIDEO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4b",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


class WatcherError(RuntimeError):
    """Raised for watcher-level configuration and execution errors."""


@dataclass
class WatchJob:
    name: str
    watch_dir: Path
    glob: str
    backends: list[str]
    recursive: bool
    settle_seconds: float
    min_age_seconds: float
    scan_interval: float
    failure_retry_seconds: float
    cli_args: dict[str, list[str]]
    notify_on_success: bool
    notify_on_failure: bool
    slack_channel: Optional[str]
    readout_enabled: bool = False
    readout_args: Optional[list[str]] = None
    store_enabled: bool = False
    store_dir: Optional[Path] = None
    store_embedding_provider: str = "ollama"
    store_embedding_model: str = "ollama/nomic-embed-text"
    enabled: bool = True


@dataclass
class CandidateSnapshot:
    size: int
    mtime: float
    seen_at: float


@dataclass
class ProcessedRecord:
    status: str
    completed_at: float
    size: int
    mtime: float
    fingerprint: str
    command: list[str]
    returncode: int
    backend: Optional[str] = None
    attempted_backends: Optional[list[str]] = None
    artifact_paths: Optional[list[str]] = None
    readout_paths: Optional[list[str]] = None
    store_paths: Optional[list[str]] = None
    next_retry_after: Optional[float] = None
    stderr: Optional[str] = None


@dataclass
class JobState:
    processed: dict[str, ProcessedRecord]
    candidates: dict[str, CandidateSnapshot]


@dataclass
class CommandResult:
    backend: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


ARTIFACT_STDOUT_PREFIX = "TRANSCRIPT_ARTIFACT_JSON="
READOUT_STDOUT_PREFIX = "READOUT_JSON="


@dataclass
class ScanStats:
    processed_attempts: int = 0
    candidate_count: int = 0
    success_count: int = 0
    failure_count: int = 0


def probe_media_readiness(media_path: Path) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration:stream=codec_type",
                str(media_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "ffprobe not found on PATH"

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return False, detail or f"ffprobe exited with {completed.returncode}"

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, f"invalid ffprobe JSON: {exc}"

    streams = payload.get("streams") or []
    if not any((stream or {}).get("codec_type") in {"audio", "video"} for stream in streams):
        return False, "no audio/video streams detected"

    raw_duration = ((payload.get("format") or {}).get("duration"))
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0.0:
        return False, "duration is missing or zero"

    return True, ""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch directories for stable audio/video files and invoke one of the transcription "
            "CLIs when a file stops changing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to watcher JSON config (default: %(default)s).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to watcher state JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        help="Override per-job scan interval in seconds.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        help="Override per-job stability wait in seconds.",
    )
    parser.add_argument(
        "--failure-retry-seconds",
        type=float,
        help="Override per-job failure retry delay in seconds.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Scan once, process any currently stable files, then exit.",
    )
    parser.add_argument(
        "--job",
        dest="job_names",
        action="append",
        help="Run only the named job(s). Can be provided multiple times.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print scan decisions and skipped-file reasons.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Emit a watcher heartbeat log at most this often while idle (default: %(default)s).",
    )
    parser.add_argument(
        "--no-progress-restart-seconds",
        type=float,
        default=DEFAULT_NO_PROGRESS_RESTART_SECONDS,
        help="Exit nonzero when queued candidates make no processing progress for this long so systemd can restart the watcher (default: %(default)s).",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def expand_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def normalize_cli_args(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, str):
        return shlex.split(raw_value)
    raise WatcherError("cli_args must be a list of strings or a shell-style string.")


def parse_calendar_cli_args(raw_job: dict[str, Any]) -> list[str]:
    raw_calendar = raw_job.get("calendar")
    if raw_calendar is None:
        return []
    if raw_calendar is False:
        return []
    if raw_calendar is True:
        return ["--use-calendar"]
    if not isinstance(raw_calendar, dict):
        raise WatcherError("calendar must be a boolean or object when provided.")
    if raw_calendar.get("enabled", True) is False:
        return []

    cli_args = ["--use-calendar"]
    providers = raw_calendar.get("providers")
    if providers is not None:
        if isinstance(providers, list):
            provider_value = ",".join(str(item) for item in providers)
        else:
            provider_value = str(providers)
        if provider_value.strip():
            cli_args.extend(["--calendar-providers", provider_value])

    calendar_id = raw_calendar.get("calendar_id")
    if calendar_id:
        cli_args.extend(["--calendar-id", str(calendar_id)])
    window = raw_calendar.get("window_hours")
    if window is not None:
        cli_args.extend(["--calendar-window", str(window)])

    gog_config = raw_calendar.get("gog") or {}
    if gog_config and not isinstance(gog_config, dict):
        raise WatcherError("calendar.gog must be an object when provided.")
    if isinstance(gog_config, dict):
        if gog_config.get("account"):
            cli_args.extend(["--calendar-gog-account", str(gog_config["account"])])
        if gog_config.get("client"):
            cli_args.extend(["--calendar-gog-client", str(gog_config["client"])])

    gws_config = raw_calendar.get("gws") or {}
    if gws_config and not isinstance(gws_config, dict):
        raise WatcherError("calendar.gws must be an object when provided.")
    if isinstance(gws_config, dict) and gws_config.get("config_dir"):
        cli_args.extend(["--calendar-gws-config-dir", str(gws_config["config_dir"])])

    return cli_args


def parse_readout_args(raw_job: dict[str, Any]) -> tuple[bool, list[str]]:
    raw_readout = raw_job.get("readout")
    if raw_readout is None or raw_readout is False:
        return False, []
    if raw_readout is True:
        return True, []
    if not isinstance(raw_readout, dict):
        raise WatcherError("readout must be a boolean or object when provided.")
    if raw_readout.get("enabled", True) is False:
        return False, []

    cli_args: list[str] = []
    provider = raw_readout.get("provider")
    if provider:
        cli_args.extend(["--provider", str(provider)])
    model = raw_readout.get("model")
    if model:
        cli_args.extend(["--model", str(model)])
    base_url = raw_readout.get("base_url")
    if base_url:
        cli_args.extend(["--base-url", str(base_url)])
    api_key_file = raw_readout.get("api_key_file")
    if api_key_file:
        cli_args.extend(["--api-key-file", str(api_key_file)])
    output_dir = raw_readout.get("output_dir")
    if output_dir:
        cli_args.extend(["--output-dir", str(output_dir)])
    timeout = raw_readout.get("timeout")
    if timeout is not None:
        cli_args.extend(["--timeout", str(timeout)])
    temperature = raw_readout.get("temperature")
    if temperature is not None:
        cli_args.extend(["--temperature", str(temperature)])
    extra_args = normalize_cli_args(raw_readout.get("cli_args"))
    cli_args.extend(extra_args)
    return True, cli_args


def parse_store_config(raw_job: dict[str, Any]) -> tuple[bool, Optional[Path], str, str]:
    raw_store = raw_job.get("store")
    if raw_store is None or raw_store is False:
        return False, None, "ollama", "ollama/nomic-embed-text"
    if raw_store is True:
        return True, None, "ollama", "ollama/nomic-embed-text"
    if not isinstance(raw_store, dict):
        raise WatcherError("store must be a boolean or object when provided.")
    if raw_store.get("enabled", True) is False:
        return False, None, "ollama", "ollama/nomic-embed-text"

    store_dir = expand_path(str(raw_store["store_dir"])) if raw_store.get("store_dir") else None
    embedding_provider = str(raw_store.get("embedding_provider") or "ollama")
    embedding_model = str(raw_store.get("embedding_model") or "ollama/nomic-embed-text")
    return True, store_dir, embedding_provider, embedding_model


def parse_backends(raw_job: dict[str, Any]) -> list[str]:
    raw_backends = raw_job.get("backends")
    if raw_backends is None:
        raw_backend = str(raw_job.get("backend") or "assembly").strip().lower()
        backends = [raw_backend]
    elif isinstance(raw_backends, list) and raw_backends:
        backends = [str(item).strip().lower() for item in raw_backends]
    else:
        raise WatcherError("backends must be a non-empty array when provided.")

    normalized: list[str] = []
    for backend in backends:
        if backend not in {"assembly", "faster_whisper"}:
            raise WatcherError(
                f"Unsupported backend '{backend}'. Use 'assembly' or 'faster_whisper'."
            )
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def parse_cli_args_by_backend(raw_job: dict[str, Any], backends: list[str]) -> dict[str, list[str]]:
    shared_args = [*parse_calendar_cli_args(raw_job), *normalize_cli_args(raw_job.get("cli_args"))]
    backend_specific = raw_job.get("cli_args_by_backend") or {}
    if backend_specific and not isinstance(backend_specific, dict):
        raise WatcherError("cli_args_by_backend must be an object keyed by backend name.")

    cli_args: dict[str, list[str]] = {}
    for backend in backends:
        specific_args = normalize_cli_args(backend_specific.get(backend)) if isinstance(backend_specific, dict) else []
        cli_args[backend] = [*shared_args, *specific_args]
    return cli_args


def load_jobs(config_path: Path, args: argparse.Namespace) -> list[WatchJob]:
    if not config_path.exists():
        raise WatcherError(
            f"Watcher config not found at {config_path}. Create it from the documented example first."
        )

    payload = load_json(config_path)
    if not isinstance(payload, dict):
        raise WatcherError("Watcher config must be a JSON object.")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise WatcherError("Watcher config must contain a non-empty 'jobs' array.")

    selected_names = set(args.job_names or [])
    jobs: list[WatchJob] = []
    for index, raw_job in enumerate(raw_jobs, start=1):
        if not isinstance(raw_job, dict):
            raise WatcherError(f"Job #{index} is not a JSON object.")

        name = str(raw_job.get("name") or f"job-{index}")
        if selected_names and name not in selected_names:
            continue

        watch_dir_raw = raw_job.get("watch_dir")
        if not watch_dir_raw:
            raise WatcherError(f"Job '{name}' is missing 'watch_dir'.")

        backends = parse_backends(raw_job)
        cli_args = parse_cli_args_by_backend(raw_job, backends)
        readout_enabled, readout_args = parse_readout_args(raw_job)
        store_enabled, store_dir, store_embedding_provider, store_embedding_model = parse_store_config(raw_job)

        job = WatchJob(
            name=name,
            watch_dir=expand_path(str(watch_dir_raw)),
            glob=str(raw_job.get("glob") or "*"),
            backends=backends,
            recursive=bool(raw_job.get("recursive", False)),
            settle_seconds=float(args.settle_seconds if args.settle_seconds is not None else raw_job.get("settle_seconds", DEFAULT_SETTLE_SECONDS)),
            min_age_seconds=float(raw_job.get("min_age_seconds", DEFAULT_MIN_AGE_SECONDS)),
            scan_interval=float(args.scan_interval if args.scan_interval is not None else raw_job.get("scan_interval", DEFAULT_SCAN_INTERVAL)),
            failure_retry_seconds=float(args.failure_retry_seconds if args.failure_retry_seconds is not None else raw_job.get("failure_retry_seconds", DEFAULT_FAILURE_RETRY_SECONDS)),
            cli_args=cli_args,
            notify_on_success=bool(raw_job.get("notify_on_success", True)),
            notify_on_failure=bool(raw_job.get("notify_on_failure", True)),
            slack_channel=str(raw_job.get("slack_channel")).strip() if raw_job.get("slack_channel") else None,
            readout_enabled=readout_enabled,
            readout_args=readout_args,
            store_enabled=store_enabled,
            store_dir=store_dir,
            store_embedding_provider=store_embedding_provider,
            store_embedding_model=store_embedding_model,
            enabled=bool(raw_job.get("enabled", True)),
        )
        jobs.append(job)

    if selected_names:
        found = {job.name for job in jobs}
        missing = sorted(selected_names - found)
        if missing:
            raise WatcherError(f"Requested job(s) not found in config: {', '.join(missing)}")

    jobs = [job for job in jobs if job.enabled]
    if not jobs:
        raise WatcherError("No enabled jobs remain after filtering.")
    return jobs


def load_state(state_path: Path, jobs: list[WatchJob]) -> dict[str, JobState]:
    if state_path.exists():
        try:
            payload = load_json(state_path)
        except Exception as exc:
            raise WatcherError(f"Failed to read watcher state {state_path}: {exc}") from exc
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    raw_jobs = payload.get("jobs") if isinstance(payload.get("jobs"), dict) else {}
    state: dict[str, JobState] = {}
    for job in jobs:
        raw_job_state = raw_jobs.get(job.name, {}) if isinstance(raw_jobs, dict) else {}
        raw_processed = raw_job_state.get("processed", {}) if isinstance(raw_job_state, dict) else {}
        raw_candidates = raw_job_state.get("candidates", {}) if isinstance(raw_job_state, dict) else {}

        processed: dict[str, ProcessedRecord] = {}
        if isinstance(raw_processed, dict):
            for key, value in raw_processed.items():
                if not isinstance(value, dict):
                    continue
                try:
                    attempted_backends = value.get("attempted_backends")
                    artifact_paths = value.get("artifact_paths")
                    readout_paths = value.get("readout_paths")
                    store_paths = value.get("store_paths")
                    processed[key] = ProcessedRecord(
                        status=str(value.get("status") or "unknown"),
                        completed_at=float(value.get("completed_at") or 0.0),
                        size=int(value.get("size") or 0),
                        mtime=float(value.get("mtime") or 0.0),
                        fingerprint=str(value.get("fingerprint") or ""),
                        command=[str(item) for item in value.get("command") or []],
                        returncode=int(value.get("returncode") or 0),
                        backend=str(value.get("backend")) if value.get("backend") else None,
                        attempted_backends=[str(item) for item in attempted_backends] if isinstance(attempted_backends, list) else None,
                        artifact_paths=[str(item) for item in artifact_paths] if isinstance(artifact_paths, list) else None,
                        readout_paths=[str(item) for item in readout_paths] if isinstance(readout_paths, list) else None,
                        store_paths=[str(item) for item in store_paths] if isinstance(store_paths, list) else None,
                        next_retry_after=(
                            float(value["next_retry_after"]) if value.get("next_retry_after") is not None else None
                        ),
                        stderr=str(value.get("stderr")) if value.get("stderr") is not None else None,
                    )
                except Exception:
                    continue

        candidates: dict[str, CandidateSnapshot] = {}
        if isinstance(raw_candidates, dict):
            for key, value in raw_candidates.items():
                if not isinstance(value, dict):
                    continue
                try:
                    candidates[key] = CandidateSnapshot(
                        size=int(value.get("size") or 0),
                        mtime=float(value.get("mtime") or 0.0),
                        seen_at=float(value.get("seen_at") or 0.0),
                    )
                except Exception:
                    continue

        state[job.name] = JobState(processed=processed, candidates=candidates)
    return state


def save_state(state_path: Path, state: dict[str, JobState]) -> None:
    payload = {"jobs": {}}
    for job_name, job_state in state.items():
        payload["jobs"][job_name] = {
            "processed": {
                key: {
                    "status": record.status,
                    "completed_at": record.completed_at,
                    "size": record.size,
                    "mtime": record.mtime,
                    "fingerprint": record.fingerprint,
                    "command": record.command,
                    "returncode": record.returncode,
                    "backend": record.backend,
                    "attempted_backends": record.attempted_backends,
                    "artifact_paths": record.artifact_paths,
                    "readout_paths": record.readout_paths,
                    "store_paths": record.store_paths,
                    "next_retry_after": record.next_retry_after,
                    "stderr": record.stderr,
                }
                for key, record in job_state.processed.items()
            },
            "candidates": {
                key: {
                    "size": snapshot.size,
                    "mtime": snapshot.mtime,
                    "seen_at": snapshot.seen_at,
                }
                for key, snapshot in job_state.candidates.items()
            },
        }
    dump_json(state_path, payload)


def build_backend_command(job: WatchJob, backend: str, media_path: Path) -> list[str]:
    script_name = "assembly_transcribe.py" if backend == "assembly" else "faster_whisper_transcribe.py"
    cli_args = job.cli_args.get(backend, [])
    return [sys.executable, str(SCRIPT_DIR / script_name), str(media_path), *cli_args]


def file_key(media_path: Path) -> str:
    return str(media_path)


def fingerprint_for(path: Path, size: int, mtime: float) -> str:
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii", errors="ignore"))
    digest.update(b"\0")
    digest.update(f"{mtime:.6f}".encode("ascii", errors="ignore"))
    return digest.hexdigest()


def iter_candidates(job: WatchJob) -> list[Path]:
    if not job.watch_dir.exists():
        return []
    iterator = job.watch_dir.rglob("*") if job.recursive else job.watch_dir.glob("*")
    candidates: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_VIDEO_EXTENSIONS:
            continue
        if not fnmatch.fnmatch(path.name.lower(), job.glob.lower()):
            continue
        candidates.append(path.resolve())
    candidates.sort(
        key=lambda item: (item.stat().st_mtime, str(item).lower()),
        reverse=True,
    )
    return candidates


def should_retry(record: ProcessedRecord, now: float, fingerprint: str) -> bool:
    if record.fingerprint != fingerprint:
        return True
    if record.status == "success":
        return False
    if record.next_retry_after is None:
        return True
    return now >= record.next_retry_after


def shorten(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_artifact_paths(stdout: str) -> list[str]:
    paths: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith(ARTIFACT_STDOUT_PREFIX):
            continue
        path = line[len(ARTIFACT_STDOUT_PREFIX) :].strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def extract_readout_paths(stdout: str) -> list[str]:
    paths: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith(READOUT_STDOUT_PREFIX):
            continue
        path = line[len(READOUT_STDOUT_PREFIX) :].strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def send_slack_notification(channel: str, message_text: str) -> None:
    openclaw_executable = shutil.which("openclaw")
    if not openclaw_executable:
        print("Warning: openclaw not found on PATH; skipping Slack notification.", file=sys.stderr, flush=True)
        return
    subprocess.run(
        [
            openclaw_executable,
            "message",
            "send",
            "--channel",
            "slack",
            "--to",
            channel,
            "--message",
            message_text,
        ],
        cwd=str(SCRIPT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )


def notify_success(job: WatchJob, media_path: Path, result: CommandResult, fallback_used: bool) -> None:
    if not job.notify_on_success or not job.slack_channel:
        return
    fallback_note = " (fallback after AssemblyAI failure)" if fallback_used else ""
    message_text = (
        f":white_check_mark: Transcription completed{fallback_note}\n"
        f"job: `{job.name}`\n"
        f"backend: `{result.backend}`\n"
        f"file: `{media_path}`"
    )
    send_slack_notification(job.slack_channel, message_text)


def notify_failure(job: WatchJob, media_path: Path, results: list[CommandResult], next_retry_after: float) -> None:
    if not job.notify_on_failure or not job.slack_channel:
        return
    backend_list = ", ".join(result.backend for result in results)
    last_result = results[-1]
    detail = shorten(last_result.stderr or last_result.stdout or "No error output captured.", limit=700)
    retry_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(next_retry_after))
    message_text = (
        f":x: Transcription failed after trying `{backend_list}`\n"
        f"job: `{job.name}`\n"
        f"file: `{media_path}`\n"
        f"retry_after: `{retry_text}`\n"
        f"detail: ```{detail}```"
    )
    send_slack_notification(job.slack_channel, message_text)


def run_backend(job: WatchJob, backend: str, media_path: Path) -> CommandResult:
    command = build_backend_command(job, backend, media_path)
    print(f"[{job.name}] Transcribing {media_path} via {backend}...", flush=True)
    completed = subprocess.run(command, cwd=str(SCRIPT_DIR), capture_output=True, text=True)

    if completed.stdout:
        stdout = completed.stdout.rstrip()
        if stdout:
            print(stdout, flush=True)
    if completed.stderr:
        stderr = completed.stderr.rstrip()
        if stderr:
            print(stderr, file=sys.stderr, flush=True)

    return CommandResult(
        backend=backend,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def build_readout_command(job: WatchJob, artifact_path: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "summarize_transcript.py"),
        artifact_path,
        *(job.readout_args or []),
    ]


def run_readouts(job: WatchJob, artifact_paths: list[str]) -> list[str]:
    if not job.readout_enabled or not artifact_paths:
        return []

    readout_paths: list[str] = []
    for artifact_path in artifact_paths:
        command = build_readout_command(job, artifact_path)
        print(f"[{job.name}] Generating readout for {artifact_path}...", flush=True)
        completed = subprocess.run(command, cwd=str(SCRIPT_DIR), capture_output=True, text=True)

        if completed.stdout:
            stdout = completed.stdout.rstrip()
            if stdout:
                print(stdout, flush=True)
                readout_paths.extend(path for path in extract_readout_paths(stdout) if path not in readout_paths)
        if completed.stderr:
            stderr = completed.stderr.rstrip()
            if stderr:
                print(stderr, file=sys.stderr, flush=True)

        if completed.returncode != 0:
            print(
                f"[{job.name}] Warning: readout generation failed for {artifact_path} "
                f"(exit {completed.returncode}); transcription remains successful.",
                file=sys.stderr,
                flush=True,
            )

    return readout_paths


def ingest_store_artifacts(job: WatchJob, artifact_paths: list[str]) -> list[str]:
    if not job.store_enabled:
        return []

    store_paths: list[str] = []
    for artifact_path in artifact_paths:
        try:
            result = ingest_artifact(
                Path(artifact_path),
                root=job.store_dir,
                embedding_provider=job.store_embedding_provider,
                embedding_model=job.store_embedding_model,
            )
        except (OSError, TranscriptStoreError) as exc:
            print(
                f"[{job.name}] Warning: store ingest failed for {artifact_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"[{job.name}] Stored {result.kind} artifact in transcript store: {result.stored_path}",
            flush=True,
        )
        if result.stored_path not in store_paths:
            store_paths.append(result.stored_path)
    return store_paths


def process_file(job: WatchJob, media_path: Path, now: float, job_state: JobState) -> bool:
    stats = media_path.stat()
    size = int(stats.st_size)
    mtime = float(stats.st_mtime)
    fingerprint = fingerprint_for(media_path, size, mtime)

    results: list[CommandResult] = []
    success_result: Optional[CommandResult] = None
    for backend in job.backends:
        result = run_backend(job, backend, media_path)
        results.append(result)
        if result.returncode == 0:
            success_result = result
            break
        if backend != job.backends[-1]:
            print(
                f"[{job.name}] Backend {backend} failed for {media_path.name}; trying fallback backend.",
                file=sys.stderr,
                flush=True,
            )

    if success_result is not None:
        fallback_used = len(results) > 1 and success_result.backend != job.backends[0]
        artifact_paths = extract_artifact_paths(success_result.stdout)
        readout_paths = run_readouts(job, artifact_paths)
        store_paths = ingest_store_artifacts(job, [*artifact_paths, *readout_paths])
        print(f"[{job.name}] Completed {media_path.name} via {success_result.backend}", flush=True)
        job_state.processed[file_key(media_path)] = ProcessedRecord(
            status="success",
            completed_at=now,
            size=size,
            mtime=mtime,
            fingerprint=fingerprint,
            command=success_result.command,
            returncode=success_result.returncode,
            backend=success_result.backend,
            attempted_backends=[result.backend for result in results],
            artifact_paths=artifact_paths,
            readout_paths=readout_paths,
            store_paths=store_paths,
            next_retry_after=None,
            stderr=None,
        )
        notify_success(job, media_path, success_result, fallback_used=fallback_used)
    else:
        final_result = results[-1]
        next_retry_after = now + job.failure_retry_seconds
        print(
            f"[{job.name}] Failed {media_path.name} after trying {', '.join(result.backend for result in results)} "
            f"(exit {final_result.returncode}); will retry after "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_retry_after))}",
            file=sys.stderr,
            flush=True,
        )
        job_state.processed[file_key(media_path)] = ProcessedRecord(
            status="failed",
            completed_at=now,
            size=size,
            mtime=mtime,
            fingerprint=fingerprint,
            command=final_result.command,
            returncode=final_result.returncode,
            backend=final_result.backend,
            attempted_backends=[result.backend for result in results],
            next_retry_after=next_retry_after,
            stderr=(final_result.stderr or final_result.stdout or "").strip() or None,
        )
        notify_failure(job, media_path, results, next_retry_after)

    job_state.candidates.pop(file_key(media_path), None)
    return True


def scan_job(job: WatchJob, job_state: JobState, *, verbose: bool) -> tuple[bool, ScanStats]:
    now = time.time()
    changed = False
    live_keys: set[str] = set()
    stats = ScanStats()

    for media_path in iter_candidates(job):
        key = file_key(media_path)
        live_keys.add(key)
        stats.candidate_count += 1
        try:
            file_stats = media_path.stat()
        except FileNotFoundError:
            continue

        size = int(file_stats.st_size)
        mtime = float(file_stats.st_mtime)
        age = max(0.0, now - mtime)
        fingerprint = fingerprint_for(media_path, size, mtime)
        processed = job_state.processed.get(key)

        if processed and not should_retry(processed, now, fingerprint):
            if verbose:
                print(f"[{job.name}] Skipping already-processed file {media_path}", flush=True)
            job_state.candidates.pop(key, None)
            continue

        if age < job.min_age_seconds:
            if verbose:
                print(f"[{job.name}] Waiting for minimum age on {media_path} ({age:.1f}s)", flush=True)
            job_state.candidates[key] = CandidateSnapshot(size=size, mtime=mtime, seen_at=now)
            changed = True
            continue

        snapshot = job_state.candidates.get(key)
        if snapshot is None or snapshot.size != size or snapshot.mtime != mtime:
            job_state.candidates[key] = CandidateSnapshot(size=size, mtime=mtime, seen_at=now)
            changed = True
            if verbose:
                print(f"[{job.name}] Tracking candidate {media_path}; waiting for stability", flush=True)
            continue

        stable_for = now - snapshot.seen_at
        if stable_for < job.settle_seconds:
            if verbose:
                print(f"[{job.name}] Candidate {media_path} stable for {stable_for:.1f}s; waiting", flush=True)
            continue

        ready, detail = probe_media_readiness(media_path)
        if not ready:
            if verbose:
                print(
                    f"[{job.name}] Candidate {media_path} is not ready for transcription yet ({detail}); waiting",
                    flush=True,
                )
            continue

        stats.processed_attempts += 1
        process_file(job, media_path, now, job_state)
        latest = job_state.processed.get(key)
        if latest and latest.status == "success":
            stats.success_count += 1
        elif latest:
            stats.failure_count += 1
        changed = True

    stale_candidates = [key for key in job_state.candidates if key not in live_keys]
    for key in stale_candidates:
        job_state.candidates.pop(key, None)
        changed = True

    return changed, stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        jobs = load_jobs(args.config.expanduser().resolve(), args)
        state_path = args.state_file.expanduser().resolve()
        state = load_state(state_path, jobs)
    except WatcherError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(jobs)} watcher job(s).", flush=True)
    for job in jobs:
        print(
            f" - {job.name}: {job.watch_dir} backends={','.join(job.backends)} glob={job.glob} "
            f"settle={job.settle_seconds:.0f}s scan={job.scan_interval:.0f}s",
            flush=True,
        )

    last_progress_at = time.time()
    last_heartbeat_at = 0.0
    while True:
        any_changed = False
        shortest_sleep: Optional[float] = None
        total_candidates = 0
        total_processed_attempts = 0
        total_successes = 0
        total_failures = 0
        for job in jobs:
            job_state = state.setdefault(job.name, JobState(processed={}, candidates={}))
            changed, stats = scan_job(job, job_state, verbose=args.verbose)
            if changed:
                any_changed = True
            total_candidates += stats.candidate_count
            total_processed_attempts += stats.processed_attempts
            total_successes += stats.success_count
            total_failures += stats.failure_count
            if shortest_sleep is None:
                shortest_sleep = job.scan_interval
            else:
                shortest_sleep = min(shortest_sleep, job.scan_interval)

        now = time.time()
        if total_processed_attempts > 0 or any_changed:
            last_progress_at = now

        if (
            args.heartbeat_seconds > 0
            and (last_heartbeat_at == 0.0 or now - last_heartbeat_at >= args.heartbeat_seconds)
        ):
            print(
                "Watcher heartbeat: "
                f"candidates={total_candidates} attempted={total_processed_attempts} "
                f"successes={total_successes} failures={total_failures}",
                flush=True,
            )
            last_heartbeat_at = now

        if (
            not args.run_once
            and args.no_progress_restart_seconds > 0
            and total_candidates > 0
            and now - last_progress_at >= args.no_progress_restart_seconds
        ):
            print(
                "Error: watcher made no progress while candidates remained queued; exiting so systemd can restart it.",
                file=sys.stderr,
                flush=True,
            )
            if any_changed:
                save_state(state_path, state)
            return 1

        if any_changed:
            save_state(state_path, state)

        if args.run_once:
            if not any_changed:
                save_state(state_path, state)
            return 0

        sleep_seconds = shortest_sleep if shortest_sleep is not None else DEFAULT_SCAN_INTERVAL
        time.sleep(max(1.0, sleep_seconds))


if __name__ == "__main__":
    sys.exit(main())
