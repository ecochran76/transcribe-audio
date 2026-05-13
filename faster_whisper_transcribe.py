#!/usr/bin/env python3
"""
Local faster-whisper transcription CLI that reuses the AssemblyAI export workflow.

Quick start:
1. python -m venv .venv
2. source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
3. pip install -r requirements.txt
4. python faster_whisper_transcribe.py demo.wav --text-output --no-speaker-labels

The same `api_keys.json` file can still provide:
- `assemblyai_language_code` for the default language selection
- `openai_api_key` for optional `--translate-to`
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from transcribe_common import (
    DEFAULT_OPENAI_MODEL,
    SCRIPT_DIR,
    TranscriptionError,
    build_calendar_provider_configs_from_args,
    build_calendar_service,
    expand_audio_inputs,
    load_json_config,
    process_transcription_outputs,
    resolve_config_candidates,
    resolve_language_settings,
)

CLI_DESCRIPTION = textwrap.dedent(
    """
    Transcribe one or more audio/video files locally with faster-whisper, then export DOCX, TXT,
    or SRT outputs. Glob patterns are expanded by the script, and optional Google Calendar and
    ffmpeg integrations can enrich or embed the transcript.
    """
)
CLI_EPILOG = textwrap.dedent(
    """Examples:
      python faster_whisper_transcribe.py meeting.m4a
      python faster_whisper_transcribe.py meeting.m4a --text-output --output-dir transcripts
      python faster_whisper_transcribe.py webinar.mp4 --srt-output
      python faster_whisper_transcribe.py board_call.wav --use-calendar --embed-subtitles
    """
)


@dataclass
class FasterWhisperRuntime:
    transcriber: Any
    device: str
    compute_type: str
    batched: bool


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
    parser.add_argument(
        "--api-key-file",
        default="api_keys.json",
        help="Path to JSON file containing optional language and OpenAI translation settings (default: %(default)s).",
    )
    parser.add_argument(
        "--print-key-sources",
        action="store_true",
        help="Print where optional translation keys were loaded from (does not print the keys).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for transcript outputs. Defaults to the audio file's directory.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name or local CTranslate2 model directory (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: %(default)s). `auto` prefers CUDA when available.",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type (default: %(default)s). Common values: float16, int8_float16, int8.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Override CPU thread count when running on CPU.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        help="Directory for downloaded faster-whisper model weights.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for speaker diarization. Overrides environment variables, api_keys.json, and the cached Hugging Face token.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched inference (default: %(default)s).",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batched inference and use WhisperModel directly.",
    )
    parser.add_argument(
        "--language",
        dest="language",
        help="Transcription language (overrides config). AssemblyAI-style values such as en_us, en_uk, pt, or auto are accepted.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.0,
        help="Compatibility no-op kept for parity with assembly_transcribe.py.",
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
        help="Run local speaker diarization and preserve speaker-aware transcript formatting (default).",
    )
    speaker_group.add_argument(
        "--no-speaker-labels",
        dest="speaker_labels",
        action="store_false",
        help="Disable speaker-label oriented formatting.",
    )
    parser.set_defaults(speaker_labels=True)
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers hint for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers hint for diarization.",
    )

    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        help="Enable voice activity detection filtering (default).",
    )
    vad_group.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="Disable voice activity detection filtering.",
    )
    parser.set_defaults(vad_filter=True)

    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_true",
        help="Condition decoding on previously generated text (default).",
    )
    condition_group.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable conditioning on previously generated text.",
    )
    parser.set_defaults(condition_on_previous_text=True)

    return parser.parse_args(argv)


def detect_cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def choose_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if detect_cuda_available() else "cpu"


def choose_compute_type(device: str, requested_compute_type: str) -> str:
    if device == "cpu":
        if requested_compute_type == "float16":
            return "default"
        if requested_compute_type == "int8_float16":
            return "int8"
    return requested_compute_type


def normalize_whisper_language(language_code: Optional[str], language_detection: bool) -> Optional[str]:
    if language_detection or not language_code:
        return None
    normalized = language_code.strip().lower()
    for separator in ("_", "-"):
        if separator in normalized:
            normalized = normalized.split(separator, 1)[0]
            break
    return normalized or None


def resolve_hf_token(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    if getattr(args, "hf_token", None):
        return args.hf_token, "--hf-token"

    for env_name in (
        "HF_TOKEN",
        "HUGGING_FACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ):
        env_value = os.getenv(env_name)
        if env_value:
            return env_value, env_name

    for candidate in resolve_config_candidates(Path(args.api_key_file).expanduser()):
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

        for key_field in ("huggingface_token", "hf_token", "hugging_face_token"):
            token = payload.get(key_field)
            if token:
                return token, str(candidate)
        break

    cached_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if cached_token_path.exists():
        token = cached_token_path.read_text(encoding="utf-8").strip()
        if token:
            return token, str(cached_token_path)

    return None, None


def build_runtime(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    cpu_threads: Optional[int],
    download_root: Optional[Path],
    batched: bool,
) -> FasterWhisperRuntime:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt` inside the project environment."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }
    if cpu_threads is not None:
        model_kwargs["cpu_threads"] = cpu_threads
    if download_root is not None:
        model_kwargs["download_root"] = str(download_root.expanduser())

    model = WhisperModel(model_name, **model_kwargs)
    transcriber = BatchedInferencePipeline(model=model) if batched else model
    return FasterWhisperRuntime(
        transcriber=transcriber,
        device=device,
        compute_type=compute_type,
        batched=batched,
    )


def load_runtime(args: argparse.Namespace) -> FasterWhisperRuntime:
    requested_device = choose_device(args.device)
    compute_type = choose_compute_type(requested_device, args.compute_type)
    batched = not args.no_batch

    try:
        return build_runtime(
            args.model,
            device=requested_device,
            compute_type=compute_type,
            cpu_threads=args.cpu_threads,
            download_root=args.download_root,
            batched=batched,
        )
    except Exception as exc:
        if args.device == "auto" and requested_device == "cuda":
            fallback_device = "cpu"
            fallback_compute_type = choose_compute_type(fallback_device, args.compute_type)
            print(
                f"Warning: failed to initialize faster-whisper on CUDA ({exc}); falling back to CPU.",
                file=sys.stderr,
            )
            return build_runtime(
                args.model,
                device=fallback_device,
                compute_type=fallback_compute_type,
                cpu_threads=args.cpu_threads,
                download_root=args.download_root,
                batched=batched,
            )
        raise


def maybe_move_diarization_pipeline_to_gpu(pipeline, *, use_cuda: bool) -> None:
    if not use_cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass


def speaker_at_time(timeline: list[dict[str, Any]], timestamp: float) -> Optional[str]:
    best_speaker: Optional[str] = None
    best_distance = float("inf")
    for item in timeline:
        if item["start"] <= timestamp <= item["end"]:
            return item["speaker"]
        distance = min(
            abs(item["start"] - timestamp),
            abs(item["end"] - timestamp),
        )
        if distance < best_distance:
            best_distance = distance
            best_speaker = item["speaker"]
    return best_speaker


def load_audio_for_diarization(audio_path: Path) -> tuple[Any, int]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise TranscriptionError(
            "numpy and torch are required for local speaker diarization. "
            "Install the project dependencies again."
        ) from exc

    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError(
            "ffmpeg is required for local speaker diarization audio decoding, but it was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
        detail = f" ffmpeg error: {stderr}" if stderr else ""
        raise TranscriptionError(
            f"Failed to decode audio for speaker diarization.{detail}"
        ) from exc

    waveform = np.frombuffer(completed.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise TranscriptionError("Decoded audio for speaker diarization was empty.")

    return torch.from_numpy(waveform.copy()).unsqueeze(0), 16000


def run_diarization(
    audio_path: Path,
    utterances: list[dict[str, Any]],
    *,
    hf_token: str,
    use_cuda: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> list[dict[str, Any]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*torchcodec is not installed correctly.*",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError as exc:
        raise TranscriptionError(
            "pyannote.audio is required for local speaker diarization. "
            "Install the project dependencies again."
        ) from exc

    try:
        load_signature = inspect.signature(PyannotePipeline.from_pretrained)
        token_parameter = "token" if "token" in load_signature.parameters else "use_auth_token"
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **{token_parameter: hf_token},
        )
    except Exception as exc:
        raise TranscriptionError(
            "Failed to load pyannote speaker diarization. Ensure your Hugging Face token is valid and you have "
            "accepted the model agreements for https://hf.co/pyannote/speaker-diarization-3.1 and, on newer "
            "pyannote releases, https://hf.co/pyannote/speaker-diarization-community-1. "
            f"Original error: {exc}"
        ) from exc

    maybe_move_diarization_pipeline_to_gpu(pipeline, use_cuda=use_cuda)

    waveform, sample_rate = load_audio_for_diarization(audio_path)
    diarize_kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers
    diarize_result = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        **diarize_kwargs,
    )

    annotation = (
        diarize_result.speaker_diarization
        if hasattr(diarize_result, "speaker_diarization")
        else diarize_result
    )

    timeline = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    if not timeline:
        return utterances

    all_words: list[dict[str, Any]] = []
    for utterance in utterances:
        all_words.extend(utterance.get("words") or [])

    diarized_utterances: list[dict[str, Any]]
    if all_words:
        for word in all_words:
            mid = (word["start"] + word["end"]) / 2
            word["speaker"] = speaker_at_time(timeline, mid)

        diarized_utterances = []
        current_words: list[dict[str, Any]] = []
        current_speaker: Optional[str] = None

        def flush_group() -> None:
            if not current_words:
                return
            text = "".join(word["word"] for word in current_words).strip()
            diarized_utterances.append(
                {
                    "speaker": current_speaker,
                    "start": int(round(current_words[0]["start"] * 1000)),
                    "end": int(round(current_words[-1]["end"] * 1000)),
                    "text": text,
                    "words": list(current_words),
                }
            )

        for word in all_words:
            speaker = word.get("speaker")
            if current_words and speaker != current_speaker:
                flush_group()
                current_words = []
            current_speaker = speaker
            current_words.append(word)
        flush_group()
    else:
        diarized_utterances = []
        for utterance in utterances:
            diarized_utterance = dict(utterance)
            mid = ((utterance.get("start") or 0) + (utterance.get("end") or 0)) / 2000
            diarized_utterance["speaker"] = speaker_at_time(timeline, mid)
            diarized_utterances.append(diarized_utterance)

    speaker_map: dict[str, str] = {}
    for utterance in diarized_utterances:
        raw_speaker = utterance.get("speaker")
        if raw_speaker and raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"SPEAKER_{len(speaker_map) + 1}"
        if raw_speaker:
            utterance["speaker"] = speaker_map[raw_speaker]

    return diarized_utterances or utterances


def transcribe_audio(
    audio_path: Path,
    runtime: FasterWhisperRuntime,
    args: argparse.Namespace,
    whisper_language: Optional[str],
) -> tuple[list[dict[str, Any]], Optional[float]]:
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": args.beam_size,
        "vad_filter": args.vad_filter,
        "condition_on_previous_text": args.condition_on_previous_text,
        "word_timestamps": args.speaker_labels,
    }
    if whisper_language:
        transcribe_kwargs["language"] = whisper_language
    if runtime.batched:
        transcribe_kwargs["batch_size"] = args.batch_size

    segments, info = runtime.transcriber.transcribe(str(audio_path), **transcribe_kwargs)
    segment_list = list(segments)

    utterances: list[dict[str, Any]] = []
    for segment in segment_list:
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        utterances.append(
            {
                "speaker": "Speaker",
                "start": int(round(float(getattr(segment, "start", 0.0)) * 1000)),
                "end": int(round(float(getattr(segment, "end", 0.0)) * 1000)),
                "text": text,
            }
        )
        words = []
        for word in getattr(segment, "words", []) or []:
            if getattr(word, "start", None) is None or getattr(word, "end", None) is None:
                continue
            word_entry = {
                "start": float(word.start),
                "end": float(word.end),
                "word": str(word.word),
            }
            if getattr(word, "probability", None) is not None:
                word_entry["probability"] = float(word.probability)
            words.append(word_entry)
        if words:
            utterances[-1]["words"] = words

    if not utterances:
        combined_text = " ".join(
            (getattr(segment, "text", "") or "").strip() for segment in segment_list if (getattr(segment, "text", "") or "").strip()
        ).strip()
        utterances = [{"speaker": "Speaker", "start": 0, "end": 0, "text": combined_text}]

    reported_duration = getattr(info, "duration", None)
    return utterances, reported_duration


def process_audio_file(
    audio_path: Path,
    args: argparse.Namespace,
    runtime: FasterWhisperRuntime,
    whisper_language: Optional[str],
    hf_token: Optional[str],
    calendar_service,
) -> bool:
    print(f"Transcribing {audio_path} with faster-whisper...", flush=True)
    try:
        utterances, reported_duration = transcribe_audio(audio_path, runtime, args, whisper_language)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return False

    if args.speaker_labels:
        if not hf_token:
            print(
                "Diarization failed: no Hugging Face token found. Provide --hf-token, set HF_TOKEN, "
                "set HUGGING_FACE_TOKEN, add `huggingface_token` to api_keys.json, or log in via huggingface-cli.",
                file=sys.stderr,
            )
            return False
        print("Running local speaker diarization...", flush=True)
        try:
            utterances = run_diarization(
                audio_path,
                utterances,
                hf_token=hf_token,
                use_cuda=runtime.device == "cuda",
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            speaker_count = len(
                {
                    (utterance.get("speaker") or "").strip()
                    for utterance in utterances
                    if (utterance.get("speaker") or "").strip()
                }
            )
            if speaker_count:
                print(f"Identified {speaker_count} speaker(s).", flush=True)
        except TranscriptionError as exc:
            print(f"Diarization failed: {exc}", file=sys.stderr)
            return False

    return process_transcription_outputs(
        audio_path,
        utterances,
        reported_duration,
        args,
        calendar_service,
        docx_title="faster-whisper Transcript",
        backend_name="faster_whisper",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    audio_paths = expand_audio_inputs(args.audio_inputs)
    if not audio_paths:
        print("Error: no audio files matched the provided inputs.", file=sys.stderr)
        return 1

    language_code, language_detection = resolve_language_settings(args)
    whisper_language = normalize_whisper_language(language_code, language_detection)
    hf_token = None
    hf_token_source = None
    if args.speaker_labels:
        hf_token, hf_token_source = resolve_hf_token(args)
        if args.print_key_sources and hf_token_source:
            print(f"Hugging Face token source: {hf_token_source}", file=sys.stderr, flush=True)
        if not hf_token:
            print(
                "Error: speaker diarization requires a Hugging Face token. Provide --hf-token, set HF_TOKEN, "
                "set HUGGING_FACE_TOKEN, add `huggingface_token` to api_keys.json, or log in via huggingface-cli.",
                file=sys.stderr,
            )
            return 1

    try:
        runtime = load_runtime(args)
    except Exception as exc:
        print(f"Error: failed to initialize faster-whisper ({exc}).", file=sys.stderr)
        return 1

    mode = f"batched batch_size={args.batch_size}" if runtime.batched else "standard"
    print(
        f"Loaded faster-whisper model '{args.model}' on {runtime.device}/{runtime.compute_type} [{mode}].",
        flush=True,
    )

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
        success = process_audio_file(path, args, runtime, whisper_language, hf_token, calendar_service)
        if not success:
            any_failures = True

    return 0 if not any_failures else 1


if __name__ == "__main__":
    sys.exit(main())
