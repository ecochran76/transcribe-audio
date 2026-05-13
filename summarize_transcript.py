#!/usr/bin/env python3
"""
Generate structured AI readouts from transcript artifact sidecars.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

from readout_artifacts import Readout
from transcript_store import TranscriptStoreError, ingest_artifact
from transcribe_common import (
    DEFAULT_OPENAI_MODEL,
    TranscriptionError,
    extract_response_detail,
    load_json_config,
    resolve_config_candidates,
    resolve_openai_key,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
READOUT_JSON_STDOUT_PREFIX = "READOUT_JSON="
READOUT_USER_INSTRUCTION = (
    "Return ONLY one valid JSON object. Do not include markdown, prose, prefaces, explanations, or code fences. "
    "The first character of your response must be '{' and the last character must be '}'. "
    "Use this exact top-level shape: "
    '{"title": string, "summary": string, "participants": [{"name": string, "role": string, "evidence": string}], '
    '"topics": [string], "key_decisions": [string], '
    '"action_items": [{"task": string, "owner": string, "due": string, "status": string}], '
    '"unresolved_questions": [string], '
    '"matter_candidates": [{"label": string, "confidence": number, "evidence": string}], '
    '"memory_candidates": [{"text": string, "kind": string, "evidence": string}], '
    '"risks": [string], "next_steps": [string]}.'
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured readout JSON/Markdown from a *.transcript.json artifact."
    )
    parser.add_argument("artifact", type=Path, help="Path to a *.transcript.json artifact.")
    parser.add_argument("--output-dir", type=Path, help="Directory for readout outputs. Defaults beside the artifact.")
    parser.add_argument(
        "--provider",
        choices=("openai-compatible", "codex-exec", "auracall", "openclaw"),
        default="openai-compatible",
        help="Readout intelligence provider. openai-compatible and codex-exec are implemented.",
    )
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="Model name for the readout provider.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL or https://api.openai.com/v1.",
    )
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to JSON file containing openai_api_key and optionally openai_base_url.",
    )
    parser.add_argument("--openai-api-key", dest="openai_api_key", help="OpenAI-compatible API key.")
    parser.add_argument("--openai-api-key-stdin", action="store_true", help="Read API key from stdin.")
    parser.add_argument("--openai-api-key-prompt", action="store_true", help="Prompt for API key interactively.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Provider request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Provider sampling temperature.")
    parser.add_argument("--store", action="store_true", help="Ingest the generated readout into ~/.transcripts.")
    parser.add_argument("--store-dir", type=Path, help="Override transcript store directory.")
    return parser.parse_args(argv)


def load_transcript_artifact(path: Path) -> dict[str, Any]:
    try:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise TranscriptionError(f"Failed to read transcript artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"Transcript artifact {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TranscriptionError("Transcript artifact must be a JSON object.")
    return payload


def resolve_openai_base_url(args: argparse.Namespace) -> tuple[str, str]:
    if args.base_url:
        return args.base_url.rstrip("/"), "--base-url"

    for candidate in resolve_config_candidates(Path(args.api_key_file).expanduser()):
        if not candidate.exists():
            continue
        try:
            payload = load_json_config(candidate)
        except Exception:
            continue
        for field in ("openai_base_url", "openai_api_base", "openai_compatible_base_url"):
            value = payload.get(field)
            if value:
                return str(value).rstrip("/"), str(candidate)
        break

    env_value = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if env_value:
        return env_value.rstrip("/"), "OPENAI_BASE_URL"
    return DEFAULT_OPENAI_BASE_URL, "default"


def build_readout_prompt(artifact: dict[str, Any]) -> str:
    event = artifact.get("event") or {}
    if isinstance(event, dict):
        matching_calendars = event.get("matching_calendars") or []
        primary_event_summary = event.get("summary")
        primary_event_participants = event.get("participants") or event.get("attendees") or []
    else:
        matching_calendars = []
        primary_event_summary = None
        primary_event_participants = []
    payload = {
        "instructions": READOUT_USER_INSTRUCTION,
        "transcript_title": artifact.get("transcript_title"),
        "backend": artifact.get("backend"),
        "duration_seconds": artifact.get("duration_seconds"),
        "recording_start": artifact.get("recording_start"),
        "recording_end": artifact.get("recording_end"),
        "event": event,
        "calendar_context": {
            "primary_event_summary": primary_event_summary,
            "primary_event_participants": primary_event_participants,
            "matching_calendars": matching_calendars,
            "calendar_context_guidance": (
                "Use matching_calendars to infer meeting domain, likely repository, and matter candidates. "
                "Calendar names are context evidence, not proof; cite them in evidence fields when used."
            ),
        },
        "prior_readout": artifact.get("prior_readout") or {},
        "route_decision": artifact.get("route_decision") or {},
        "supporting_context": artifact.get("supporting_context") or {},
        "transcript_text": artifact.get("transcript_text") or "",
        "utterances": artifact.get("utterances") or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def readout_system_prompt() -> str:
    return (
        "You produce structured meeting transcript readouts for downstream routing and review. "
        "Return ONLY valid JSON. Do not wrap it in markdown. Use this exact top-level shape: "
        "{"
        '"title": string, '
        '"summary": string, '
        '"participants": [{"name": string, "role": string, "evidence": string}], '
        '"topics": [string], '
        '"key_decisions": [string], '
        '"action_items": [{"task": string, "owner": string, "due": string, "status": string}], '
        '"unresolved_questions": [string], '
        '"matter_candidates": [{"label": string, "confidence": number, "evidence": string}], '
        '"memory_candidates": [{"text": string, "kind": string, "evidence": string}], '
        '"risks": [string], '
        '"next_steps": [string]'
        "}. Use empty arrays when nothing is identified. Keep confidence between 0 and 1. "
        "Do not invent facts; mark uncertain items with low confidence and evidence. "
        "If calendar_context.matching_calendars is present, use calendar summaries and event summaries as "
        "context clues for participants, topics, meeting type, matter_candidates, and memory_candidates. "
        "When calendar context influences an item, mention the calendar or event source in that item's evidence. "
        "If supporting_context is present, use it as cited context for a contextual reread. Mention source labels "
        "or source_type/source_id values in evidence fields when they influence participants, matter candidates, "
        "memory candidates, risks, or next steps."
    )


def parse_model_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if not content:
        raise TranscriptionError("OpenAI-compatible readout returned an empty response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None:
        raise TranscriptionError("OpenAI-compatible readout JSON must be an object.")

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise TranscriptionError("OpenAI-compatible readout JSON must be an object.")
    raise TranscriptionError("OpenAI-compatible readout did not return valid JSON.")


def validate_readout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not str(payload.get("summary") or "").strip() and not any(
        payload.get(key)
        for key in (
            "participants",
            "topics",
            "key_decisions",
            "action_items",
            "unresolved_questions",
            "matter_candidates",
            "memory_candidates",
            "risks",
            "next_steps",
        )
    ):
        raise TranscriptionError("OpenAI-compatible readout returned JSON without readout content.")
    return payload


def call_openai_compatible(
    artifact: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    temperature: float,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()
    session.headers.update({"authorization": f"Bearer {api_key}"})
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": readout_system_prompt()},
            {"role": "user", "content": build_readout_prompt(artifact)},
        ],
    }
    try:
        response = session.post(endpoint, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise TranscriptionError(f"OpenAI-compatible readout request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = extract_response_detail(response)
        raise TranscriptionError(f"OpenAI-compatible readout failed ({response.status_code}): {detail}")

    data = response.json()
    content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    return validate_readout_payload(parse_model_json_object(content))


def call_codex_exec(
    artifact: dict[str, Any],
    *,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="transcribe-readout-codex-", suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    command = [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "--cd",
        str(Path.cwd()),
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--output-last-message",
        str(output_path),
    ]
    if model and model != DEFAULT_OPENAI_MODEL:
        command.extend(["--model", model])
    command.append("-")

    prompt = "\n\n".join(
        [
            readout_system_prompt(),
            READOUT_USER_INSTRUCTION,
            "Do not run shell commands or inspect files. Use only the transcript artifact JSON provided below.",
            build_readout_prompt(artifact),
        ]
    )
    try:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        content = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
    except FileNotFoundError as exc:
        raise TranscriptionError("codex-exec readouts require the `codex` CLI on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"codex-exec readout timed out after {timeout:g} seconds.") from exc
    finally:
        try:
            output_path.unlink()
        except OSError:
            pass

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise TranscriptionError(f"codex-exec readout failed ({result.returncode}): {detail}")
    if not content:
        content = (result.stdout or "").strip()
    if not content:
        raise TranscriptionError("codex-exec readout returned an empty response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("codex-exec readout did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise TranscriptionError("codex-exec readout JSON must be an object.")
    return parsed


def output_base_path(artifact_path: Path, output_dir: Optional[Path]) -> Path:
    base_name = artifact_path.name
    if base_name.endswith(".transcript.json"):
        base_name = base_name[: -len(".transcript.json")]
    else:
        base_name = artifact_path.stem
    directory = output_dir.expanduser() if output_dir else artifact_path.parent
    return directory / base_name


def write_readout_from_payload(
    artifact_path: Path,
    model_payload: dict[str, Any],
    *,
    provider: dict[str, Any],
    output_dir: Optional[Path] = None,
    store: bool = False,
    store_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    artifact_path = artifact_path.expanduser().resolve()
    readout = Readout.from_model_payload(
        validate_readout_payload(model_payload),
        source_artifact_path=artifact_path,
        provider=provider,
    )
    base_path = output_base_path(artifact_path, output_dir)
    json_path = base_path.with_suffix(".readout.json")
    markdown_path = base_path.with_suffix(".readout.md")
    readout.write_json(json_path)
    readout.write_markdown(markdown_path)
    if store:
        ingest_artifact(json_path, root=store_dir)
    return json_path, markdown_path


def generate_readout(args: argparse.Namespace) -> tuple[Path, Path]:
    artifact_path = args.artifact.expanduser().resolve()
    artifact = load_transcript_artifact(artifact_path)
    if args.provider not in {"openai-compatible", "codex-exec"}:
        raise TranscriptionError(
            f"Provider '{args.provider}' is defined as a seam but is not implemented yet. "
            "Use openai-compatible for unattended readout generation."
        )

    provider_info = {
        "name": args.provider,
        "model": args.model,
    }
    if args.provider == "openai-compatible":
        api_key, api_key_source = resolve_openai_key(args)
        if not api_key:
            raise TranscriptionError(
                "openai-compatible readouts require an API key. Provide --openai-api-key, "
                "set OPENAI_API_KEY, or add openai_api_key to api_keys.json."
            )
        base_url, base_url_source = resolve_openai_base_url(args)
        provider_info.update(
            {
                "base_url": base_url,
                "api_key_source": api_key_source,
                "base_url_source": base_url_source,
            }
        )
        model_payload = call_openai_compatible(
            artifact,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            timeout=args.timeout,
            temperature=args.temperature,
        )
    else:
        provider_info["execution"] = "codex exec --sandbox read-only --ask-for-approval never"
        model_payload = call_codex_exec(
            artifact,
            model=args.model,
            timeout=args.timeout,
        )
    return write_readout_from_payload(
        artifact_path,
        model_payload,
        provider=provider_info,
        output_dir=args.output_dir,
        store=args.store,
        store_dir=args.store_dir,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        json_path, markdown_path = generate_readout(args)
    except (TranscriptionError, TranscriptStoreError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Writing readout JSON to {json_path}...")
    print(f"Writing readout Markdown to {markdown_path}...")
    print(f"{READOUT_JSON_STDOUT_PREFIX}{json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
