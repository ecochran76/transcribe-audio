#!/usr/bin/env python3
"""
Command-line helper for AssemblyAI transcription with speaker diarization.

Key features:
* Streams large audio uploads using the AssemblyAI upload endpoint.
* Polls the transcript job until completion with progress feedback.
* Emits a DOCX transcript (and optional plain-text transcript).

API keys are read from, in order of precedence:
1. The `--api-key` argument.
2. The `ASSEMBLYAI_API_KEY` environment variable.
3. The `assemblyai_api_key` field inside a JSON file (default `api_keys.json`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import requests
from docx import Document
from docx.shared import Pt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

DEFAULT_BASE_URL = "https://api.assemblyai.com"
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
SCRIPT_DIR = Path(__file__).resolve().parent
WILDCARD_PATTERN = re.compile(r"[*?\[\]]")


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an error payload."""


def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_file_creation_time(path: Path) -> datetime:
    stats = path.stat()
    if hasattr(stats, "st_birthtime"):
        return datetime.fromtimestamp(stats.st_birthtime, tz=timezone.utc)
    if os.name == "nt":
        return datetime.fromtimestamp(stats.st_ctime, tz=timezone.utc)
    return datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)


def load_client_config(credentials_path: Path) -> dict[str, Any]:
    with credentials_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "installed" in config:
        return config
    if "web" in config:
        # Convert web client credentials into installed-style config accepted by InstalledAppFlow.
        web_config = config["web"]
        # Installed flow expects redirect URIs with localhost and urn scheme.
        installed_config = {
            "installed": {
                "client_id": web_config.get("client_id"),
                "project_id": web_config.get("project_id"),
                "auth_uri": web_config.get("auth_uri"),
                "token_uri": web_config.get("token_uri"),
                "auth_provider_x509_cert_url": web_config.get("auth_provider_x509_cert_url"),
                "client_secret": web_config.get("client_secret"),
                "redirect_uris": web_config.get(
                    "redirect_uris",
                    ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
                ),
            }
        }
        return installed_config
    raise AssemblyAIError(
        f"Unsupported Google client secrets structure in {credentials_path}. Expected 'installed' or 'web'."
    )


def build_calendar_service(credentials_path: Path, token_path: Path, fallback_client_path: Optional[Path] = None):
    client_config: Optional[dict[str, Any]] = None
    chosen_config_path: Optional[Path] = None

    candidate_paths = [credentials_path]
    if fallback_client_path:
        candidate_paths.append(fallback_client_path)
    candidate_paths.extend(
        [
            credentials_path.with_name("client_secrets.json"),
            Path("client_secrets.json"),
            SCRIPT_DIR / "credentials.json",
            SCRIPT_DIR / "client_secrets.json",
        ]
    )

    unique_candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate in candidate_paths:
        if not candidate:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if not candidate.exists():
            continue
        try:
            client_config = load_client_config(candidate)
            chosen_config_path = candidate
            break
        except AssemblyAIError:
            continue

    if not client_config or not chosen_config_path:
        raise AssemblyAIError(
            "Google client secrets not found or invalid. Provide --calendar-credentials pointing to a "
            "client secret JSON downloaded from Google Cloud Console."
        )

    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), CALENDAR_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(client_config, CALENDAR_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def parse_event_datetime(event_time: dict) -> Optional[datetime]:
    if not event_time:
        return None
    raw = event_time.get("dateTime") or event_time.get("date")
    if not raw:
        return None
    if len(raw) == 10:
        raw = f"{raw}T00:00:00+00:00"
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if "T" in raw and "+" not in raw[10:] and "-" not in raw[10:]:
        raw = f"{raw}+00:00"
    return datetime.fromisoformat(raw)


def format_event_datetime(dt_value: Optional[datetime]) -> Optional[str]:
    if not dt_value:
        return None
    return dt_value.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def extract_event_metadata(event: dict) -> dict[str, Any]:
    start_dt = parse_event_datetime(event.get("start", {}))
    end_dt = parse_event_datetime(event.get("end", {}))
    attendees = []
    for attendee in event.get("attendees", []):
        if attendee.get("responseStatus") == "declined":
            continue
        name = attendee.get("displayName")
        email = attendee.get("email")
        if name and email:
            attendees.append(f"{name} <{email}>")
        elif name:
            attendees.append(name)
        elif email:
            attendees.append(email)
    return {
        "summary": event.get("summary") or "Untitled Event",
        "start": start_dt,
        "end": end_dt,
        "location": event.get("location"),
        "attendees": attendees,
        "hangoutLink": event.get("hangoutLink"),
    }


def find_closest_event(service, calendar_id: str, target_time: datetime, window_hours: float) -> Optional[dict]:
    time_min = to_rfc3339(target_time - timedelta(hours=window_hours))
    time_max = to_rfc3339(target_time + timedelta(hours=window_hours))
    result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            maxResults=50,
            orderBy="startTime",
        )
        .execute()
    )
    events = result.get("items", [])
    if not events:
        return None

    best_event: Optional[dict] = None
    best_score: tuple[int, float] = (1, float("inf"))

    for event in events:
        event_start = parse_event_datetime(event.get("start", {}))
        event_end = parse_event_datetime(event.get("end", {}))
        if not event_start:
            continue

        contains_flag = 1
        distance = float("inf")

        if event_end and event_start <= target_time <= event_end:
            contains_flag = 0
            distance = 0.0
        else:
            start_delta = abs((event_start - target_time).total_seconds())
            if event_end:
                end_delta = abs((event_end - target_time).total_seconds())
                distance = min(start_delta, end_delta)
            else:
                distance = start_delta

        score = (contains_flag, distance)
        if score < best_score:
            best_score = score
            best_event = event

    return best_event


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def unique_path(base_path: Path) -> Path:
    counter = 1
    candidate = base_path
    while candidate.exists():
        candidate = base_path.with_name(f"{base_path.stem} ({counter}){base_path.suffix}")
        counter += 1
    return candidate


def rename_audio_with_event(audio_path: Path, event_info: dict[str, Any], created_at: datetime) -> tuple[Path, str]:
    timestamp_str = created_at.astimezone().strftime("%Y-%m-%d %H-%M")
    event_summary = sanitize_filename_part(event_info.get("summary", "Untitled Event"))
    original_base = sanitize_filename_part(audio_path.stem)
    parts = [timestamp_str, event_summary, original_base]
    base_name = " ".join(part for part in parts if part)
    target_path = audio_path.with_name(f"{base_name}{audio_path.suffix}")
    target_path = unique_path(target_path)
    if target_path != audio_path:
        audio_path = audio_path.rename(target_path)
    return audio_path, base_name


def expand_audio_inputs(inputs: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        expanded = Path(raw).expanduser()
        matches: list[str] = []
        if WILDCARD_PATTERN.search(str(raw)):
            matches = glob.glob(str(expanded), recursive=True)
        else:
            if expanded.exists():
                matches = [str(expanded)]
            else:
                matches = glob.glob(str(expanded), recursive=True)

        if not matches:
            print(f"Warning: no files matched '{raw}'", file=sys.stderr)
            continue

        for match in matches:
            match_path = Path(match).expanduser().resolve()
            if not match_path.is_file():
                print(f"Skipping non-file match: {match_path}", file=sys.stderr)
                continue
            if match_path in seen:
                continue
            seen.add(match_path)
            resolved.append(match_path)

    return resolved

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_inputs",
        nargs="+",
        help="Audio file(s) or glob patterns to transcribe.",
    )

    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="AssemblyAI API key. Overrides env vars and config files.",
    )
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to JSON file containing `assemblyai_api_key` (default: %(default)s).",
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


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key

    env_key = os.getenv("ASSEMBLYAI_API_KEY")
    if env_key:
        return env_key

    config_path = Path(args.api_key_file)
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssemblyAIError(f"Invalid JSON in {config_path}: {exc}") from exc

        for candidate in ("assemblyai_api_key", "assembly_ai_api_key"):
            api_key = payload.get(candidate)
            if api_key:
                return api_key

    raise AssemblyAIError(
        "AssemblyAI API key not found. Provide --api-key, set ASSEMBLYAI_API_KEY, "
        f"or store it in {args.api_key_file} under 'assemblyai_api_key'."
    )


def stream_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[bytes, None, None]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def upload_audio(session: requests.Session, audio_path: Path) -> str:
    upload_endpoint = f"{DEFAULT_BASE_URL}/v2/upload"
    response = session.post(upload_endpoint, data=stream_file(audio_path))
    response.raise_for_status()
    data = response.json()
    if "upload_url" not in data:
        raise AssemblyAIError(f"Upload response missing 'upload_url': {data}")
    return data["upload_url"]


def request_transcription(
    session: requests.Session,
    audio_url: str,
    *,
    model: str,
    speaker_labels: bool,
) -> str:
    transcript_endpoint = f"{DEFAULT_BASE_URL}/v2/transcript"
    payload = {
        "audio_url": audio_url,
        "speech_model": model,
        "speaker_labels": speaker_labels,
    }
    response = session.post(transcript_endpoint, json=payload)
    response.raise_for_status()
    data = response.json()
    if "id" not in data:
        raise AssemblyAIError(f"Transcript response missing 'id': {data}")
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
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise AssemblyAIError(payload.get("error", "Unknown AssemblyAI error"))

        ETA = payload.get("eta")
        if ETA is not None:
            print(f"Waiting... status={status}, eta={ETA}s", flush=True)
        else:
            print(f"Waiting... status={status}", flush=True)
        time.sleep(poll_interval)


def format_utterance(utterance: dict) -> str:
    speaker = utterance.get("speaker", "Speaker")
    start = utterance.get("start", 0) / 1000
    end = utterance.get("end", 0) / 1000
    text = (utterance.get("text") or "").strip()
    return f"{speaker} [{start:0.2f}s - {end:0.2f}s]: {text}"


def add_event_metadata_to_doc(document: Document, event_info: dict[str, Any]) -> None:
    document.add_heading("Event Details", level=2)
    document.add_paragraph(f"Event: {event_info['summary']}")

    start_str = format_event_datetime(event_info.get("start"))
    end_str = format_event_datetime(event_info.get("end"))
    if start_str:
        document.add_paragraph(f"Start: {start_str}")
    if end_str:
        document.add_paragraph(f"End: {end_str}")
    if event_info.get("location"):
        document.add_paragraph(f"Location: {event_info['location']}")
    if event_info.get("hangoutLink"):
        document.add_paragraph(f"Meeting Link: {event_info['hangoutLink']}")

    attendees = event_info.get("attendees") or []
    if attendees:
        document.add_paragraph("Attendees:")
        for attendee in attendees:
            document.add_paragraph(attendee, style="List Bullet")
    document.add_paragraph("")


def write_docx(
    utterances: list[dict],
    output_path: Path,
    *,
    event_info: Optional[dict[str, Any]] = None,
) -> None:
    document = Document()
    document.add_heading("AssemblyAI Transcript", level=1)
    if event_info:
        add_event_metadata_to_doc(document, event_info)
    for utterance in utterances:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(format_utterance(utterance))
        run.font.size = Pt(11)
    document.save(output_path)


def write_text(
    utterances: list[dict],
    output_path: Path,
    *,
    event_info: Optional[dict[str, Any]] = None,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        if event_info:
            handle.write(f"Event: {event_info['summary']}\n")
            start_str = format_event_datetime(event_info.get("start"))
            end_str = format_event_datetime(event_info.get("end"))
            if start_str:
                handle.write(f"Start: {start_str}\n")
            if end_str:
                handle.write(f"End: {end_str}\n")
            if event_info.get("location"):
                handle.write(f"Location: {event_info['location']}\n")
            if event_info.get("hangoutLink"):
                handle.write(f"Meeting Link: {event_info['hangoutLink']}\n")
            attendees = event_info.get("attendees") or []
            if attendees:
                handle.write("Attendees:\n")
                for attendee in attendees:
                    handle.write(f" - {attendee}\n")
            handle.write("\n")
        for utterance in utterances:
            handle.write(format_utterance(utterance))
            handle.write("\n")


def ensure_output_dir(path: Optional[Path], audio_path: Path) -> Path:
    if path is None:
        return audio_path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def process_audio_file(
    audio_path: Path,
    args: argparse.Namespace,
    api_key: str,
    calendar_service,
) -> bool:
    working_path = audio_path
    event_info: Optional[dict[str, Any]] = None
    base_name_override: Optional[str] = None

    if args.use_calendar:
        if calendar_service is None:
            print("Warning: calendar service unavailable; skipping event lookup.", file=sys.stderr)
        else:
            try:
                created_at = get_file_creation_time(working_path)
                event = find_closest_event(
                    calendar_service,
                    args.calendar_id,
                    created_at,
                    args.calendar_window,
                )
                if event:
                    event_info = extract_event_metadata(event)
                    try:
                        working_path, base_name_override = rename_audio_with_event(working_path, event_info, created_at)
                        print(
                            f"Matched calendar event '{event_info['summary']}' and renamed file to {working_path.name}"
                        )
                    except OSError as exc:
                        print(
                            f"Warning: failed to rename {working_path.name} ({exc}); continuing without rename.",
                            file=sys.stderr,
                        )
                else:
                    print("No calendar event found near the file timestamp; continuing without rename.")
            except Exception as exc:
                print(
                    f"Warning: calendar lookup failed ({exc}); continuing without calendar metadata.",
                    file=sys.stderr,
                )

    output_dir = ensure_output_dir(args.output_dir, working_path)
    base_name = base_name_override or working_path.stem
    docx_path = output_dir / f"{base_name} Transcript.docx"
    text_path = output_dir / f"{base_name} Transcript.txt"

    session = requests.Session()
    session.headers.update({"authorization": api_key, "user-agent": "assembly-transcribe-cli"})

    print(f"Uploading {working_path} to AssemblyAI...")
    try:
        audio_url = upload_audio(session, working_path)
    except (requests.RequestException, AssemblyAIError) as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return False

    print("Starting transcription job...")
    try:
        transcript_id = request_transcription(
            session,
            audio_url,
            model=args.model,
            speaker_labels=args.speaker_labels,
        )
    except (requests.RequestException, AssemblyAIError) as exc:
        print(f"Transcription request failed: {exc}", file=sys.stderr)
        return False

    print(f"Polling transcript {transcript_id}...")
    try:
        transcript_payload = poll_transcript(
            session,
            transcript_id,
            poll_interval=args.poll_interval,
        )
    except (requests.RequestException, AssemblyAIError) as exc:
        print(f"Polling failed: {exc}", file=sys.stderr)
        return False

    utterances = transcript_payload.get("utterances") or []
    if not utterances:
        print("AssemblyAI response did not contain diarized utterances; falling back to plain text.")
        text = transcript_payload.get("text") or ""
        utterances = [{"speaker": "Speaker", "start": 0, "end": 0, "text": text}]

    print(f"Writing DOCX transcript to {docx_path}...")
    write_docx(utterances, docx_path, event_info=event_info)

    if args.text_output:
        print(f"Writing plain-text transcript to {text_path}...")
        write_text(utterances, text_path, event_info=event_info)

    print("Completed successfully.")
    return True


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    audio_paths = expand_audio_inputs(args.audio_inputs)

    if not audio_paths:
        print("Error: no audio files matched the provided inputs.", file=sys.stderr)
        return 1

    try:
        api_key = resolve_api_key(args)
    except AssemblyAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    calendar_service = None
    if args.use_calendar:
        try:
            calendar_service = build_calendar_service(
                args.calendar_credentials,
                args.calendar_token,
                fallback_client_path=args.calendar_client_secrets,
            )
        except Exception as exc:
            print(
                f"Warning: calendar service initialization failed ({exc}); continuing without calendar metadata.",
                file=sys.stderr,
            )
            calendar_service = None

    any_failures = False
    for path in audio_paths:
        success = process_audio_file(path, args, api_key, calendar_service)
        if not success:
            any_failures = True

    return 0 if not any_failures else 1


if __name__ == "__main__":
    sys.exit(main())
