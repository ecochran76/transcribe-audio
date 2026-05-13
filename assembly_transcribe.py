#!/usr/bin/env python3
"""
AssemblyAI transcription CLI for producing DOCX, TXT, or subtitle outputs with optional diarization.

Quick start:
1. python -m venv .venv
2. source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
3. pip install -r requirements.txt
4. export ASSEMBLYAI_API_KEY=your_key  (or copy api_keys.json.sample to api_keys.json)
5. python assembly_transcribe.py demo.wav --text-output

API keys are read in this order:
1. The `--api-key` argument.
2. The `ASSEMBLYAI_API_KEY` environment variable.
3. The `assemblyai_api_key` field inside a JSON file (default `api_keys.json`).

Language defaults to English (`en_us`) unless overridden via `assemblyai_language_code` in the same JSON file
(for example, `pt` for Portuguese).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from transcribe_common import (
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_OPENAI_MODEL,
    SCRIPT_DIR,
    TranscriptionError,
    build_calendar_provider_configs_from_args,
    build_calendar_service,
    expand_audio_inputs,
    extract_response_detail,
    load_json_config,
    process_transcription_outputs,
    resolve_config_candidates,
    resolve_language_settings,
)

DEFAULT_BASE_URL = "https://api.assemblyai.com"
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
CLI_DESCRIPTION = textwrap.dedent(
    """
    Upload one or more audio/video files to AssemblyAI, wait for completion, and export DOCX, TXT,
    or SRT outputs. Glob patterns are expanded by the script, and optional Google Calendar and
    ffmpeg integrations can enrich or embed the transcript.
    """
)
CLI_EPILOG = textwrap.dedent(
    """Examples:
      python assembly_transcribe.py meeting.m4a
      python assembly_transcribe.py meeting.m4a --text-output --output-dir transcripts
      python assembly_transcribe.py webinar.mp4 --srt-output
      python assembly_transcribe.py board_call.wav --use-calendar --embed-subtitles
    """
)


def raise_for_status_with_details(response: requests.Response, *, context: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = extract_response_detail(response)
        hint = ""
        if response.status_code == 400 and detail:
            lowered_detail = detail.lower()
            if not any(term in lowered_detail for term in ("balance", "top up", "quota", "billing", "payment")):
                hint = " (Hint: try `--no-speaker-labels` and/or a different `--model`.)"
        if detail:
            raise TranscriptionError(f"{context} failed ({response.status_code}): {detail}{hint}") from exc
        raise TranscriptionError(f"{context} failed ({response.status_code}).{hint}") from exc


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "audio_inputs",
        nargs="+",
        help="Audio/video file(s) or glob patterns to transcribe (quote globs on Windows shells).",
    )

    api_key_group = parser.add_mutually_exclusive_group()
    api_key_group.add_argument(
        "--api-key",
        dest="api_key",
        help="AssemblyAI API key. Overrides environment variables and api_keys.json.",
    )
    api_key_group.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read AssemblyAI API key from stdin (first line). Useful to avoid saving keys to disk.",
    )
    api_key_group.add_argument(
        "--api-key-prompt",
        action="store_true",
        help="Prompt for AssemblyAI API key interactively (input hidden).",
    )
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to JSON file containing `assemblyai_api_key` (default: %(default)s). Copy api_keys.json.sample to create one.",
    )
    parser.add_argument(
        "--print-key-sources",
        action="store_true",
        help="Print where API keys were loaded from (does not print the keys).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for transcript outputs. Defaults to the audio file's directory.",
    )
    parser.add_argument(
        "--model",
        default="universal",
        help="AssemblyAI speech model (default: %(default)s).",
    )
    parser.add_argument(
        "--language",
        dest="language",
        help="Transcription language (overrides config). Examples: en_us (default), pt, en_uk, or auto for detection.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between polling attempts (default: %(default)s).",
    )
    parser.add_argument(
        "--text-output",
        action="store_true",
        help="Also emit a plain-text transcript alongside the DOCX file.",
    )
    parser.add_argument(
        "--srt-output",
        action="store_true",
        help="Emit an SRT subtitle file instead of a DOCX transcript.",
    )
    parser.add_argument(
        "--docx-output",
        action="store_true",
        help="Also emit a DOCX transcript when --srt-output is set.",
    )
    parser.add_argument(
        "--embed-subtitles",
        action="store_true",
        help="Embed generated subtitles into the source media using ffmpeg.",
    )
    parser.add_argument(
        "--translate-to",
        dest="translate_to",
        help="Translate transcript and subtitles to a target language (e.g. en). Requires an OpenAI key.",
    )

    openai_key_group = parser.add_mutually_exclusive_group()
    openai_key_group.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        help="OpenAI API key (used for --translate-to). Overrides OPENAI_API_KEY and api_keys.json.",
    )
    openai_key_group.add_argument(
        "--openai-api-key-stdin",
        action="store_true",
        help="Read OpenAI API key from stdin (first line).",
    )
    openai_key_group.add_argument(
        "--openai-api-key-prompt",
        action="store_true",
        help="Prompt for OpenAI API key interactively (input hidden).",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model for translation (default: %(default)s).",
    )
    parser.add_argument(
        "--use-calendar",
        action="store_true",
        help="Match the audio file to a Google Calendar event and include metadata.",
    )
    parser.add_argument(
        "--calendar-id",
        default="primary",
        help="Google Calendar ID to query when --use-calendar is set (default: %(default)s).",
    )
    parser.add_argument(
        "--calendar-credentials",
        type=Path,
        default=SCRIPT_DIR / "credentials.json",
        help="Path to Google OAuth client credentials when using --use-calendar (default: %(default)s). "
        "If this file contains legacy tokens, the script will look for client secrets nearby.",
    )
    parser.add_argument(
        "--calendar-client-secrets",
        type=Path,
        default=SCRIPT_DIR / "client_secrets.json",
        help="Optional explicit path to Google client secrets (used when --calendar-credentials contains tokens).",
    )
    parser.add_argument(
        "--calendar-token",
        type=Path,
        default=SCRIPT_DIR / "token.json",
        help="Path to store Google OAuth tokens when using --use-calendar (default: %(default)s).",
    )
    parser.add_argument(
        "--calendar-window",
        type=float,
        default=24.0,
        help="Hours on either side of the file timestamp to search for matching events (default: %(default)s).",
    )
    parser.add_argument(
        "--calendar-providers",
        help="Comma-separated calendar provider order. Supported: gog,gws,google-api. Default: gog,gws,google-api.",
    )
    parser.add_argument(
        "--calendar-gog-account",
        help="Account email passed to gog as --account when the gog calendar provider is used.",
    )
    parser.add_argument(
        "--calendar-gog-client",
        help="OAuth client name passed to gog as --client when the gog calendar provider is used.",
    )
    parser.add_argument(
        "--calendar-gws-config-dir",
        type=Path,
        help="Config directory passed to gws through GOOGLE_WORKSPACE_CLI_CONFIG_DIR.",
    )
    speaker_group = parser.add_mutually_exclusive_group()
    speaker_group.add_argument(
        "--speaker-labels",
        dest="speaker_labels",
        action="store_true",
        help="Enable speaker diarization (default).",
    )
    speaker_group.add_argument(
        "--no-speaker-labels",
        dest="speaker_labels",
        action="store_false",
        help="Disable speaker diarization.",
    )
    parser.set_defaults(speaker_labels=True)
    return parser.parse_args(argv)


def resolve_api_key(args: argparse.Namespace) -> tuple[str, str]:
    if args.api_key:
        return args.api_key, "--api-key"

    if getattr(args, "api_key_prompt", False):
        if not sys.stdin.isatty():
            raise TranscriptionError("--api-key-prompt requires an interactive terminal (stdin is not a TTY).")
        entered = getpass.getpass("AssemblyAI API key: ").strip()
        if entered:
            return entered, "--api-key-prompt"

    if getattr(args, "api_key_stdin", False):
        entered = sys.stdin.readline().strip() if sys.stdin.isatty() else sys.stdin.read().strip()
        if entered:
            return entered, "--api-key-stdin"

    seen: set[Path] = set()
    for candidate in resolve_config_candidates(Path(args.api_key_file).expanduser()):
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            payload = load_json_config(candidate)
        except json.JSONDecodeError as exc:
            print(f"Warning: skipping invalid JSON in {candidate} ({exc}).", file=sys.stderr)
            continue
        except TranscriptionError as exc:
            print(f"Warning: skipping {candidate} ({exc}).", file=sys.stderr)
            continue

        for key_field in ("assemblyai_api_key", "assembly_ai_api_key"):
            api_key = payload.get(key_field)
            if api_key:
                return api_key, str(candidate)

    for env_name in ("ASSEMBLYAI_API_KEY", "ASSEMBLY_API_KEY"):
        env_key = os.getenv(env_name)
        if env_key:
            return env_key, env_name

    raise TranscriptionError(
        "AssemblyAI API key not found. Provide --api-key, set ASSEMBLYAI_API_KEY or ASSEMBLY_API_KEY, "
        f"or store it in {args.api_key_file} (or alongside the script) under 'assemblyai_api_key'."
    )


def stream_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def upload_audio(session: requests.Session, audio_path: Path) -> str:
    response = session.post(f"{DEFAULT_BASE_URL}/v2/upload", data=stream_file(audio_path))
    raise_for_status_with_details(response, context="Upload")
    data = response.json()
    if "upload_url" not in data:
        raise TranscriptionError(f"Upload response missing 'upload_url': {data}")
    return data["upload_url"]


def request_transcription(
    session: requests.Session,
    audio_url: str,
    *,
    model: str,
    speaker_labels: bool,
    language_code: Optional[str],
    language_detection: bool,
) -> str:
    payload = {
        "audio_url": audio_url,
        "speech_model": model,
        "speaker_labels": speaker_labels,
    }
    if language_detection:
        payload["language_detection"] = True
    else:
        payload["language_code"] = language_code or DEFAULT_LANGUAGE_CODE
    response = session.post(f"{DEFAULT_BASE_URL}/v2/transcript", json=payload)
    raise_for_status_with_details(response, context="Transcription request")
    data = response.json()
    if "id" not in data:
        raise TranscriptionError(f"Transcript response missing 'id': {data}")
    return data["id"]


def poll_transcript(
    session: requests.Session,
    transcript_id: str,
    *,
    poll_interval: float,
) -> dict:
    status_endpoint = f"{DEFAULT_BASE_URL}/v2/transcript/{transcript_id}"
    while True:
        response = session.get(status_endpoint)
        raise_for_status_with_details(response, context="Polling transcript")
        payload = response.json()

        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise TranscriptionError(payload.get("error", "Unknown AssemblyAI error"))

        eta = payload.get("eta")
        if eta is not None:
            print(f"Waiting... status={status}, eta={eta}s", flush=True)
        else:
            print(f"Waiting... status={status}", flush=True)
        time.sleep(poll_interval)


def process_audio_file(
    audio_path: Path,
    args: argparse.Namespace,
    api_key: str,
    language_settings: tuple[Optional[str], bool],
    calendar_service,
) -> bool:
    session = requests.Session()
    session.headers.update({"authorization": api_key, "user-agent": "assembly-transcribe-cli"})

    print(f"Uploading {audio_path} to AssemblyAI...", flush=True)
    try:
        audio_url = upload_audio(session, audio_path)
    except (requests.RequestException, TranscriptionError) as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return False

    print("Starting transcription job...", flush=True)
    try:
        language_code, language_detection = language_settings
        transcript_id = request_transcription(
            session,
            audio_url,
            model=args.model,
            speaker_labels=args.speaker_labels,
            language_code=language_code,
            language_detection=language_detection,
        )
    except (requests.RequestException, TranscriptionError) as exc:
        print(f"Transcription request failed: {exc}", file=sys.stderr)
        return False

    print(f"Polling transcript {transcript_id}...", flush=True)
    try:
        transcript_payload = poll_transcript(
            session,
            transcript_id,
            poll_interval=args.poll_interval,
        )
    except (requests.RequestException, TranscriptionError) as exc:
        print(f"Polling failed: {exc}", file=sys.stderr)
        return False

    utterances = transcript_payload.get("utterances") or []
    if not utterances:
        print("AssemblyAI response did not contain diarized utterances; falling back to plain text.")
        text = transcript_payload.get("text") or ""
        utterances = [{"speaker": "Speaker", "start": 0, "end": 0, "text": text}]

    return process_transcription_outputs(
        audio_path,
        utterances,
        transcript_payload.get("audio_duration"),
        args,
        calendar_service,
        docx_title="AssemblyAI Transcript",
        backend_name="assembly",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    audio_paths = expand_audio_inputs(args.audio_inputs)
    if not audio_paths:
        print("Error: no audio files matched the provided inputs.", file=sys.stderr)
        return 1

    try:
        api_key, api_key_source = resolve_api_key(args)
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.print_key_sources:
        print(f"AssemblyAI API key source: {api_key_source}", file=sys.stderr, flush=True)

    language_settings = resolve_language_settings(args)

    calendar_service = None
    if args.use_calendar:
        try:
            calendar_service = build_calendar_service(
                args.calendar_credentials,
                args.calendar_token,
                fallback_client_path=args.calendar_client_secrets,
                provider_configs=build_calendar_provider_configs_from_args(args),
            )
        except Exception as exc:
            print(
                f"Warning: calendar service initialization failed ({exc}); continuing without calendar metadata.",
                file=sys.stderr,
            )
            calendar_service = None

    any_failures = False
    for path in audio_paths:
        success = process_audio_file(path, args, api_key, language_settings, calendar_service)
        if not success:
            any_failures = True

    return 0 if not any_failures else 1


if __name__ == "__main__":
    sys.exit(main())
