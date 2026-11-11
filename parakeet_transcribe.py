#!/usr/bin/env python3
"""CUDA Parakeet transcription followed by mandatory CUDA speaker diarization.

NeMo runs in an isolated interpreter and exits before Pyannote loads. A shared
file lock serializes fallback jobs without evicting other GPU services.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def diarized_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Assign by greatest temporal overlap; preserve gaps as unknown."""
    if not words or not turns:
        raise ValueError(
            "No timed speech or diarization turns; refusing unlabeled output"
        )
    utterances: list[dict] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if end < start or start < 0:
            raise ValueError("Invalid word timestamps")
        overlaps = [
            (max(0, min(end, t["end"]) - max(start, t["start"])), t["speaker"])
            for t in turns
        ]
        overlap, speaker = max(overlaps, key=lambda item: item[0])
        speaker = speaker if overlap > 0 else "UNKNOWN"
        if (
            not utterances
            or utterances[-1]["speaker"] != speaker
            or start * 1000 - utterances[-1]["end"] > 1500
        ):
            utterances.append(
                {
                    "speaker": speaker,
                    "start": round(start * 1000),
                    "end": round(end * 1000),
                    "text": "",
                    "words": [],
                }
            )
        u = utterances[-1]
        u["end"] = round(end * 1000)
        u["text"] = (u["text"] + " " + word["word"]).strip()
        u["words"].append({**word, "speaker": speaker})
    if all(u["speaker"] == "UNKNOWN" for u in utterances):
        raise ValueError("Diarization did not overlap recognized speech")
    return utterances


def asr_worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    if os.getppid() != args.parent_pid:
        raise RuntimeError("ASR owner changed")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Cannot install ASR parent-death guard")
    if os.getppid() != args.parent_pid:
        raise RuntimeError("ASR owner changed")
    config = json.loads(args.config.read_text())
    model_path = Path(config["model_path"]).expanduser()
    with model_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != config["model_sha256"]:
            raise ValueError("Parakeet model checksum mismatch")
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet requires CUDA")
    torch.set_num_threads(4)
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cuda")
    model.eval()
    words = []
    with sf.SoundFile(args.audio) as source:
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError("Expected mono 16kHz WAV")
        total = len(source)
        # Padded windows retain words at chunk boundaries; midpoint ownership
        # removes duplicate words from the overlap.
        for base in range(0, total, 45 * 16000):
            left = max(0, base - 2 * 16000)
            right = min(total, base + 47 * 16000)
            source.seek(left)
            audio = source.read(right - left, dtype="float32")
            with torch.inference_mode():
                hypothesis = model.transcribe([audio], batch_size=1, timestamps=True)[0]
            for item in hypothesis.timestamp.get("word", []):
                start = left / 16000 + float(item["start"])
                end = left / 16000 + float(item["end"])
                midpoint = (start + end) / 2
                if base / 16000 <= midpoint < min(total, base + 45 * 16000) / 16000:
                    words.append({"word": item["word"], "start": start, "end": end})
            audio.fill(0)
    if not words:
        raise ValueError("Parakeet returned no timed words")
    args.output.write_text(json.dumps({"words": words, "duration": total / 16000}))
    return 0


def transcribe(audio: Path, config_path: Path) -> tuple[list[dict], float]:
    config = json.loads(config_path.read_text())
    lock = Path(config["lock_path"]).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU fallback busy; retry later")
                time.sleep(1)
        with tempfile.TemporaryDirectory(prefix="parakeet-diarized-") as work:
            wav, result = Path(work) / "audio.wav", Path(work) / "asr.json"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(audio),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                check=True,
                timeout=3600,
            )
            subprocess.run(
                [
                    str(Path(config["asr_python"]).expanduser()),
                    str(Path(__file__).resolve()),
                    "--asr-worker",
                    str(config_path),
                    str(wav),
                    str(result),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                check=True,
                timeout=7200,
            )
            asr = json.loads(result.read_text())
            import torch
            from pyannote.audio import Pipeline

            if not torch.cuda.is_available():
                raise RuntimeError("Diarization requires CUDA")
            torch.set_num_threads(4)
            token = os.environ.get(config.get("hf_token_env", "HUGGING_FACE_TOKEN"))
            pipeline = Pipeline.from_pretrained(
                config["diarization_model"],
                revision=config["diarization_revision"],
                token=token,
            )
            if pipeline is None:
                raise RuntimeError("Diarization model unavailable")
            pipeline.to(torch.device("cuda"))
            # Decode explicitly: avoids optional torchcodec dependencies.
            import soundfile as sf

            waveform, rate = sf.read(wav, dtype="float32")
            output = pipeline(
                {
                    "waveform": torch.from_numpy(waveform).unsqueeze(0),
                    "sample_rate": rate,
                }
            )
            annotation = getattr(output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = getattr(output, "speaker_diarization", output)
            turns = [
                {"start": float(t.start), "end": float(t.end), "speaker": str(s)}
                for t, _, s in annotation.itertracks(yield_label=True)
            ]
            return diarized_words(asr["words"], turns), asr["duration"]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--asr-worker":
        return asr_worker(argv[1:])
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--gpu-config", required=True, type=Path)
    options, remaining = extra.parse_known_args(argv)
    import provenance_config
    from assembly_transcribe import parse_args
    from transcribe_common import (
        build_calendar_provider_configs_from_args,
        build_calendar_service,
        expand_audio_inputs,
        process_transcription_outputs,
    )

    args = parse_args(remaining)
    if not args.speaker_labels:
        raise ValueError("Parakeet fallback requires speaker diarization")
    args.model = "nvidia/parakeet-tdt-0.6b-v3+pyannote/community-1"
    paths = expand_audio_inputs(args.audio_inputs)
    if not paths:
        raise ValueError("No matching media inputs")
    calendar = None
    if args.use_calendar:
        configs = provenance_config.configured_provider_configs_or_fallback(
            args, build_calendar_provider_configs_from_args(args)
        )
        calendar = build_calendar_service(
            args.calendar_credentials,
            args.calendar_token,
            fallback_client_path=args.calendar_client_secrets,
            provider_configs=configs,
        )
    success = True
    for path in paths:
        try:
            utterances, duration = transcribe(
                path, options.gpu_config.expanduser().resolve()
            )
            success = (
                process_transcription_outputs(
                    path,
                    utterances,
                    duration,
                    args,
                    calendar,
                    docx_title="Parakeet Diarized Transcript",
                    backend_name="parakeet",
                )
                and success
            )
        except Exception as exc:  # noqa: BLE001 - isolate a failed backend job
            print(f"Parakeet diarized transcription failed: {exc}", file=sys.stderr)
            success = False
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
