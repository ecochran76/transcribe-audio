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
import json
import os
import sys
import time
from pathlib import Path
from typing import Generator, Iterable, Optional

import requests
from docx import Document
from docx.shared import Pt

DEFAULT_BASE_URL = "https://api.assemblyai.com"
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an error payload."""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", help="Path to the audio file to transcribe.")

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


def write_docx(utterances: list[dict], output_path: Path) -> None:
    document = Document()
    document.add_heading("AssemblyAI Transcript", level=1)
    for utterance in utterances:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(format_utterance(utterance))
        run.font.size = Pt(11)
    document.save(output_path)


def write_text(utterances: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for utterance in utterances:
            handle.write(format_utterance(utterance))
            handle.write("\n")


def ensure_output_dir(path: Optional[Path], audio_path: Path) -> Path:
    if path is None:
        return audio_path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    audio_path = Path(args.audio_file).expanduser().resolve()

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    try:
        api_key = resolve_api_key(args)
    except AssemblyAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = ensure_output_dir(args.output_dir, audio_path)
    base_name = audio_path.stem
    docx_path = output_dir / f"{base_name}_transcript.docx"
    text_path = output_dir / f"{base_name}_transcript.txt"

    session = requests.Session()
    session.headers.update({"authorization": api_key, "user-agent": "assembly-transcribe-cli"})

    print(f"Uploading {audio_path} to AssemblyAI...")
    try:
        audio_url = upload_audio(session, audio_path)
    except (requests.RequestException, AssemblyAIError) as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1

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
        return 1

    print(f"Polling transcript {transcript_id}...")
    try:
        transcript_payload = poll_transcript(
            session,
            transcript_id,
            poll_interval=args.poll_interval,
        )
    except (requests.RequestException, AssemblyAIError) as exc:
        print(f"Polling failed: {exc}", file=sys.stderr)
        return 1

    utterances = transcript_payload.get("utterances") or []
    if not utterances:
        print("AssemblyAI response did not contain diarized utterances; falling back to plain text.")
        text = transcript_payload.get("text") or ""
        utterances = [{"speaker": "Speaker", "start": 0, "end": 0, "text": text}]

    print(f"Writing DOCX transcript to {docx_path}...")
    write_docx(utterances, docx_path)

    if args.text_output:
        print(f"Writing plain-text transcript to {text_path}...")
        write_text(utterances, text_path)

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
