#!/usr/bin/env python3
"""
Submit legacy transcript readout work to AuraCall response batches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summarize_transcript import (  # noqa: E402
    build_readout_prompt,
    load_transcript_artifact,
    parse_model_json_object,
    readout_system_prompt,
    write_readout_from_payload,
)
from transcript_store import (  # noqa: E402
    TranscriptStoreError,
    legacy_enrichment_queue,
    store_dir,
)
from transcribe_common import TranscriptionError, extract_response_detail  # noqa: E402

DEFAULT_MODEL = "agent:pro-extended-chatgpt-soylei-transcripts"
DEFAULT_CLIENT_ENV = Path("~/.local/state/transcribe-audio/auracall-transcripts.env")
DEFAULT_MANIFEST_DIR = Path("~/.local/state/transcribe-audio/auracall-batches")
DEFAULT_MAX_CONCURRENT_RUNS = 2
DEFAULT_MAX_BROWSER_INTERACTIONS_PER_MINUTE = 8
MANIFEST_JSON_STDOUT_PREFIX = "AURACALL_BATCH_MANIFEST="


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue or materialize legacy transcript readouts via AuraCall.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_CLIENT_ENV, help="AuraCall client .env file.")
    parser.add_argument("--base-url", help="AuraCall/OpenAI-compatible base URL. Defaults to env.")
    parser.add_argument("--api-key", help="AuraCall API key. Defaults to env.")
    parser.add_argument("--store-dir", type=Path, default=store_dir(), help="Transcript store root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Submit pending legacy transcript readouts as one AuraCall batch.")
    enqueue.add_argument("--limit", type=int, help="Limit queued item count.")
    enqueue.add_argument("--model", default=None, help=f"AuraCall model. Defaults to {DEFAULT_MODEL}.")
    enqueue.add_argument("--all", action="store_true", help="Include legacy rows that already have a readout.")
    enqueue.add_argument("--no-dedupe", action="store_true", help="Do not collapse same-hash or same-title queue entries.")
    enqueue.add_argument("--dry-run", action="store_true", help="Build and write a manifest without submitting.")
    enqueue.add_argument("--manifest", type=Path, help="Manifest path. Defaults under ~/.local/state/transcribe-audio.")
    enqueue.add_argument("--store", action="store_true", help="Ingest readouts during later materialization.")
    enqueue.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_RUNS,
        help="Per-batch AuraCall concurrency limit.",
    )
    enqueue.add_argument(
        "--max-browser-interactions-per-minute",
        type=int,
        default=DEFAULT_MAX_BROWSER_INTERACTIONS_PER_MINUTE,
        help="Per-batch AuraCall browser interaction rate limit.",
    )

    status = subparsers.add_parser("status", help="Read AuraCall batch status from a manifest.")
    status.add_argument("manifest", type=Path)
    status.add_argument("--materialize", action="store_true", help="Write completed readouts beside transcripts.")
    status.add_argument("--output-dir", type=Path, help="Directory for materialized readouts.")
    status.add_argument("--store", action="store_true", help="Ingest materialized readouts into the transcript store.")
    return parser.parse_args(argv)


def read_env_file(path: Path) -> dict[str, str]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {}
    values: dict[str, str] = {}
    for line in expanded.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def runtime_env(args: argparse.Namespace) -> dict[str, str]:
    values = read_env_file(args.env_file)
    return {**os.environ, **values}


def resolve_base_url(args: argparse.Namespace, env: dict[str, str]) -> str:
    value = args.base_url or env.get("OPENAI_BASE_URL") or env.get("AURACALL_BASE_URL")
    if not value:
        raise TranscriptionError("AuraCall base URL is required. Set OPENAI_BASE_URL or pass --base-url.")
    return value.rstrip("/")


def resolve_batch_url(args: argparse.Namespace, env: dict[str, str]) -> str:
    if env.get("AURACALL_BATCH_URL"):
        return env["AURACALL_BATCH_URL"].rstrip("/")
    return f"{resolve_base_url(args, env)}/response-batches"


def resolve_api_key(args: argparse.Namespace, env: dict[str, str]) -> str:
    value = args.api_key or env.get("OPENAI_API_KEY") or env.get("AURACALL_API_KEY")
    if not value:
        raise TranscriptionError("AuraCall API key is required. Set OPENAI_API_KEY or pass --api-key.")
    return value


def resolve_model(args: argparse.Namespace, env: dict[str, str]) -> str:
    return args.model or env.get("AURACALL_MODEL") or DEFAULT_MODEL


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def auracall_readout_system_prompt() -> str:
    return (
        readout_system_prompt()
        + " For AuraCall browser-backed batch runs, write the structured JSON readout into a ChatGPT "
        "REPL/workspace file named legacy_readout.json and surface it as a downloadable artifact in the final "
        "assistant response. The final response must expose the actual artifact/download link or attachment card; "
        "a text-only readiness note is not sufficient. Never reply with a future-tense status such as "
        "\"I'll create the file\". Use the workspace/REPL file mechanism before answering. Do not compress the "
        "readout for chat-message length; preserve substantive detail in the JSON file."
    )


def auracall_readout_prompt(artifact: dict[str, Any]) -> str:
    return (
        "Create a ChatGPT REPL/workspace file named legacy_readout.json. The file must contain exactly one valid "
        "JSON object using the requested readout schema. Surface legacy_readout.json as a downloadable artifact. "
        "Do not put the full JSON object in the assistant message. Do not describe what you will do. Do the file "
        "creation first. Your final response must include the actual downloadable legacy_readout.json "
        "artifact/link or attachment card, such as a sandbox:/.../legacy_readout.json link when that is the "
        "workspace file URL shape; do not reply with only a text readiness note. Preserve substantive detail "
        "instead of summarizing away important context.\n\n"
        + build_readout_prompt(artifact)
    )


def create_request(item: dict[str, Any], model: str) -> dict[str, Any]:
    source_path = Path(str(item["source_path"])).expanduser().resolve()
    artifact = load_transcript_artifact(source_path)
    return {
        "model": model,
        "input": [
            {"role": "system", "content": auracall_readout_system_prompt()},
            {"role": "user", "content": auracall_readout_prompt(artifact)},
        ],
        "metadata": {
            "workflow": "transcribe-audio-legacy-enrichment",
            "outputContract": {
                "mode": "chatgpt_workspace_artifact",
                "artifactFileName": "legacy_readout.json",
                "mimeType": "application/json",
                "fallback": "none",
            },
            "transcriptDocumentId": item.get("id"),
            "transcriptTitle": item.get("title"),
            "sourcePath": str(source_path),
        },
        "auracall": {
            "agent": model.removeprefix("agent:") if model.startswith("agent:") else None,
            "service": "chatgpt",
            "runtimeProfile": "wsl-chrome-3",
            "transport": "browser",
        },
    }


def default_manifest_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_MANIFEST_DIR.expanduser() / f"legacy-enrichment-{stamp}.json"


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def post_batch(batch_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        batch_url,
        json=payload,
        headers={"authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise TranscriptionError(f"AuraCall batch enqueue failed ({response.status_code}): {extract_response_detail(response)}")
    data = response.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise TranscriptionError("AuraCall batch enqueue did not return a batch id.")
    return data


def read_json_url(url: str, api_key: str) -> dict[str, Any]:
    response = requests.get(url, headers={"authorization": f"Bearer {api_key}"}, timeout=60)
    if response.status_code >= 400:
        raise TranscriptionError(f"AuraCall read failed ({response.status_code}): {extract_response_detail(response)}")
    data = response.json()
    if not isinstance(data, dict):
        raise TranscriptionError("AuraCall read did not return a JSON object.")
    return data


def response_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts).strip()


def artifact_payload_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    path_text = (
        item.get("path")
        or metadata.get("path")
        or metadata.get("localPath")
        or metadata.get("uri")
        or metadata.get("download_path")
        or item.get("uri")
    )
    if path_text:
        path = Path(str(path_text)).expanduser()
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")

    for key in ("text", "content", "json", "data"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    return ""


def response_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text = response_text(payload)
    if text:
        try:
            return parse_model_json_object(text)
        except TranscriptionError:
            pass

    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "artifact":
            continue
        artifact_type = str(item.get("artifact_type") or item.get("artifactType") or item.get("kind") or "").lower()
        mime_type = str(item.get("mime_type") or item.get("mimeType") or "").lower()
        title = str(item.get("title") or item.get("filename") or item.get("name") or "")
        uri = str(item.get("uri") or "")
        json_named = title.lower().endswith(".json") or uri.lower().endswith(".json")
        if artifact_type and artifact_type not in {"file", "json", "generated", "download"} and not json_named:
            continue
        if mime_type and "json" not in mime_type and not json_named:
            continue
        content = artifact_payload_text(item)
        if content.strip():
            return parse_model_json_object(content)

    raise TranscriptionError("AuraCall response did not include parseable readout JSON text or artifact output.")


def enqueue(args: argparse.Namespace) -> int:
    env = runtime_env(args)
    model = resolve_model(args, env)
    queue = legacy_enrichment_queue(
        root=args.store_dir,
        limit=args.limit,
        pending_only=not args.all,
        provider="openai-compatible",
        model=model,
        store_readouts=args.store,
        dedupe=not args.no_dedupe,
    )
    requests_payload = [create_request(item, model) for item in queue["items"]]
    batch_payload = {
        "metadata": {
            "workflow": "transcribe-audio-legacy-enrichment",
            "createdAt": utc_now_iso(),
            "model": model,
            "storeDir": queue["store_dir"],
            "selectedCount": queue["selected_count"],
            "duplicateCount": queue["duplicate_count"],
        },
        "limits": {
            "maxConcurrentRuns": args.max_concurrent_runs,
            "maxBrowserInteractionsPerMinute": args.max_browser_interactions_per_minute,
        },
        "requests": requests_payload,
    }
    manifest = {
        "object": "transcribe_audio_auracall_batch_manifest",
        "created_at": utc_now_iso(),
        "model": model,
        "dry_run": bool(args.dry_run),
        "store": bool(args.store),
        "batch_url": resolve_batch_url(args, env),
        "response_base_url": resolve_base_url(args, env),
        "queue": queue,
        "request_count": len(requests_payload),
        "batch_payload": batch_payload if args.dry_run else None,
        "batch": None,
    }
    if requests_payload and not args.dry_run:
        manifest["batch"] = post_batch(manifest["batch_url"], resolve_api_key(args, env), batch_payload)
    manifest_path = write_manifest(args.manifest or default_manifest_path(), manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "request_count": len(requests_payload),
                "batch_id": (manifest["batch"] or {}).get("id"),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    print(f"{MANIFEST_JSON_STDOUT_PREFIX}{manifest_path}")
    return 0


def materialize_completed(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    batch_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = runtime_env(args)
    api_key = resolve_api_key(args, env)
    base_url = str(manifest["response_base_url"]).rstrip("/")
    jobs_by_index = {
        int(job["index"]): job
        for job in batch_status.get("jobs") or []
        if isinstance(job, dict) and isinstance(job.get("index"), int)
    }
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate((manifest.get("queue") or {}).get("items") or []):
        job = jobs_by_index.get(index)
        if not job or job.get("status") != "completed" or not job.get("responseId"):
            continue
        try:
            response = read_json_url(f"{base_url}/responses/{job['responseId']}", api_key)
            model_payload = response_model_payload(response)
            json_path, markdown_path = write_readout_from_payload(
                Path(str(item["source_path"])),
                model_payload,
                provider={
                    "name": "auracall-response-batch",
                    "model": manifest.get("model"),
                    "batch_id": (manifest.get("batch") or {}).get("id"),
                    "response_id": job["responseId"],
                },
                output_dir=args.output_dir,
                store=args.store,
                store_dir=args.store_dir,
            )
            results.append(
                {
                    "index": index,
                    "response_id": job["responseId"],
                    "source_path": item["source_path"],
                    "readout_json": str(json_path),
                    "readout_markdown": str(markdown_path),
                }
            )
        except (TranscriptionError, TranscriptStoreError, OSError, json.JSONDecodeError) as exc:
            errors.append(
                {
                    "index": index,
                    "response_id": job["responseId"],
                    "source_path": item.get("source_path"),
                    "error": str(exc),
                }
            )
    return results, errors


def status(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_id = ((manifest.get("batch") or {}).get("id") or "").strip()
    if not batch_id:
        raise TranscriptionError(f"Manifest has no submitted AuraCall batch id: {manifest_path}")
    env = runtime_env(args)
    batch_url = f"{str(manifest['batch_url']).rstrip('/')}/{batch_id}"
    batch_status = read_json_url(batch_url, resolve_api_key(args, env))
    materialized: list[dict[str, Any]] = []
    materialization_errors: list[dict[str, Any]] = []
    if args.materialize:
        materialized, materialization_errors = materialize_completed(args, manifest, batch_status)
    manifest["last_status"] = batch_status
    if materialized:
        manifest["materialized"] = materialized
    if materialization_errors:
        manifest["materialization_errors"] = materialization_errors
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "status": batch_status,
                "materialized": materialized,
                "materialization_errors": materialization_errors,
            },
            indent=2,
        )
    )
    print(f"{MANIFEST_JSON_STDOUT_PREFIX}{manifest_path}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "enqueue":
            return enqueue(args)
        if args.command == "status":
            return status(args)
    except (TranscriptionError, TranscriptStoreError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
