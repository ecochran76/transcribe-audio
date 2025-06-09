#!/usr/bin/env python3
"""transcribe_audio.py

CLI utility and module for transcribing audio with either **openai‑whisper**
or **faster‑whisper**. The backend is chosen at runtime via the
``method = whisper | faster-whisper`` key in *transcription_config.ini* (or the
``--method`` flag).

Why the rewrite?
----------------
* The original hard‑coded Whisper path now shares logic with faster‑whisper.
* Unified preprocessing (FFmpeg→WAV) that both back‑ends consume.
* Keeps every previous command‑line flag so existing scripts continue to work.
* Leaves diarisation, speaker hints, etc. untouched and optional.
"""
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Tuple, List

import pkg_resources
import soundfile as sf
import torch
from mutagen import File
from pydub import AudioSegment
from tqdm import tqdm

# ─────────────────────── silence noisy libs ──────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch")
# for noisy in ("pyannote", "speechbrain", "huggingface_hub", "urllib3"):
#     logging.getLogger(noisy).setLevel(logging.ERROR)

# ───────────────────────────── logging ───────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

file_handler = logging.FileHandler("transcription.log", encoding="utf-8", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler, file_handler])
logger = logging.getLogger(__name__)

# ───────────────────── dependency helpers ────────────────────────────
REQUIRED_PACKAGES = {
    "openai-whisper": "openai-whisper",
    "faster-whisper": "faster-whisper",
    "assemblyai": "assemblyai",  # placeholder – for future backend
    "soundfile": "soundfile",
    "tqdm": "tqdm",
    "torch": "torch",
    "pyannote.audio": "pyannote-audio",
    "pydub": "pydub",
    "mutagen": "mutagen",
}


def _check_and_install_deps() -> None:
    missing: List[str] = []
    for pkg, install_name in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            missing.append(install_name)

    if missing:
        logger.info("Missing deps: %s – installing…", ", ".join(missing))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except subprocess.CalledProcessError as exc:
            logger.error("pip install failed: %s", exc)
            sys.exit(1)


# ───────────────────────────── config ────────────────────────────────
CONFIG_FILE = Path(__file__).with_name("transcription_config.ini")
CONFIG = configparser.ConfigParser()

_DEFAULT_TRANSCRIPTION: dict[str, str] = {
    "method": "whisper",  # whisper | faster-whisper
    "model": "base",
    "speakers": "true",
    "temp_dir": "",
    "transcript_output_format": "txt",
}


def _initialise_config() -> None:
    CONFIG["DEFAULT"] = {
        "method": "whisper",
        "assemblyai_api_key": "",
        "hf_token": "",  # diarisation
    }
    CONFIG["Transcription"] = _DEFAULT_TRANSCRIPTION.copy()
    CONFIG.write(CONFIG_FILE.open("w", encoding="utf-8"))
    logger.info("Created default configuration at %s", CONFIG_FILE)


def load_config() -> None:
    if not CONFIG_FILE.exists():
        _initialise_config()
    CONFIG.read(CONFIG_FILE)
    CONFIG.setdefault("Transcription", _DEFAULT_TRANSCRIPTION)
    CONFIG.setdefault("DEFAULT", {
        "method": "whisper",
        "assemblyai_api_key": "",
        "hf_token": "",
    })


# ─────────────────── helper: Hugging‑Face token ──────────────────────

def _get_hf_token() -> str:
    token = CONFIG["DEFAULT"].get("hf_token", "").strip()
    if token:
        return token
    token = input("Enter your Hugging Face access token (for diarisation): ").strip()
    CONFIG["DEFAULT"]["hf_token"] = token
    CONFIG.write(CONFIG_FILE.open("w", encoding="utf-8"))
    return token


# ─────────────────── helper: timestamp from audio ────────────────────

def _audio_timestamp(path: Path) -> str:
    try:
        audio = File(path)
        if audio and "date" in audio.tags:
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y"):
                try:
                    return time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(audio.tags["date"][0], fmt))
                except ValueError:
                    continue
    except Exception:
        pass  # best‑effort only
    ts = time.localtime(path.stat().st_mtime)
    return time.strftime("%Y-%m-%d %H:%M:%S", ts)


# ───────────────────────── back‑end loaders ─────────────────────────

def _setup_whisper(device: str, model_name: str):  # noqa: ANN001
    try:
        import whisper  # lazy import
    except ImportError:
        logger.error("openai‑whisper not installed – run pip install openai-whisper")
        sys.exit(1)
    logger.info("Loading Whisper %s on %s", model_name, device)
    start = time.time()
    try:
        model = whisper.load_model(model_name, device=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        model = whisper.load_model(model_name, device=device)
    logger.info("Whisper ready in %.1fs", time.time() - start)
    return model


def _setup_faster_whisper(device: str, model_name: str):  # noqa: ANN001
    try:
        from faster_whisper import WhisperModel  # lazy import
    except ImportError:
        logger.error("faster‑whisper not installed – run pip install faster-whisper")
        sys.exit(1)
    compute_type = "int8_float16" if device == "cuda" else "int8"
    logger.info("Loading faster‑whisper %s on %s (%s)", model_name, device, compute_type)
    start = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.info("faster‑whisper ready in %.1fs", time.time() - start)
    return model


# ─────────── helper: ensure backend gets a real WAV file ─────────────

def _prepare_audio(audio_file: Path, temp_dir: str | None) -> Tuple[Path, Path | None]:
    tmp_wav: Path | None = None
    process_file = audio_file
    try:
        sf.SoundFile(str(audio_file))  # sanity check
    except Exception:
        logger.info("Converting %s → WAV for backend", audio_file.name)
        tmp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="transcribe_"))
        tmp_wav = tmp_dir / f"{audio_file.stem}.wav"
        AudioSegment.from_file(audio_file).export(tmp_wav, format="wav")
        process_file = tmp_wav
    return process_file, tmp_wav


# ─────────── per‑backend transcription wrappers ─────────────────────

def _transcribe_with_whisper(model, wav_path: Path, with_words: bool = True):  # noqa: ANN001
    import whisper  # type: ignore
    result = model.transcribe(str(wav_path), verbose=False, word_timestamps=with_words)
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]


def _transcribe_with_faster_whisper(model, wav_path: Path):  # noqa: ANN001
    segments_generator, _info = model.transcribe(str(wav_path), word_timestamps=True)
    segments_out: List[dict] = []
    for segment in tqdm(segments_generator, desc="Decoding"):
        segments_out.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
    return segments_out


# ───────────────────────── main API ──────────────────────────────────

def transcribe(
    audio_file: Path,
    *,
    method: str,
    model: str,
    with_speakers: bool,
    temp_dir: str | None,
) -> Tuple[str, List[dict]]:  # text, segments
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wav_path, tmp_wav = _prepare_audio(audio_file, temp_dir)

    if method == "whisper":
        backend = _setup_whisper(device, model)
        segments = _transcribe_with_whisper(backend, wav_path)
    elif method == "faster-whisper":
        backend = _setup_faster_whisper(device, model)
        segments = _transcribe_with_faster_whisper(backend, wav_path)
    else:
        logger.error("Unsupported method: %s", method)
        sys.exit(1)

    if with_speakers:
        try:
            from pyannote.audio import Pipeline  # lazy import
            diar_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=_get_hf_token()
            )
            for seg in segments:
                from pyannote.core import Segment as _PySegment
                segment_obj = _PySegment(seg["start"], seg["end"])
                speakers = {
                    speaker for _, _, speaker in diar_pipeline(str(wav_path)).crop(segment_obj).itertracks(yield_label=True)  # type: ignore[arg-type]
                }
                seg["speaker"] = "/".join(sorted(speakers)) or "Unknown"
        except Exception as exc:
            logger.error("Diarisation failed: %s", exc)

    if tmp_wav:
        tmp_wav.unlink(missing_ok=True)

    ts_header = f"Date and Time: {_audio_timestamp(audio_file)}\n\n"
    text_body = "\n".join(
        f"[{s['start']:.2f}s - {s['end']:.2f}s] "
        f"{'Speaker ' + s.get('speaker', 'Unknown') + ': ' if with_speakers else ''}{s['text']}"
        for s in segments
    )
    return ts_header + text_body, segments


# ───────────────────────── CLI entrypoint ────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper or faster‑whisper",
        epilog="Example: python transcribe_audio.py audio.mp3 --method faster-whisper --speakers --model base -o out.txt",
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("-o", "--output", nargs="?", const=True, default=None)
    parser.add_argument("--method", choices=["whisper", "faster-whisper"], help="Override backend from config")
    parser.add_argument("--model", help="Model size (tiny, base, small, …)")
    parser.add_argument("--speakers", action="store_true", help="Run speaker diarisation")
    parser.add_argument("--temp-dir", help="Scratch directory for temp WAVs")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    _check_and_install_deps()
    load_config()

    tr_cfg = CONFIG["Transcription"]
    method = args.method or tr_cfg.get("method", "whisper")
    model = args.model or tr_cfg.get("model", "base")
    speakers = args.speakers or tr_cfg.getboolean("speakers", fallback=False)
    temp_dir = args.temp_dir or tr_cfg.get("temp_dir", "").strip() or None

    text, segments = transcribe(
        args.audio_file,
        method=method,
        model=model,
        with_speakers=speakers,
        temp_dir=temp_dir,
    )

    if args.output is not None:
        if args.output is True:
            out_path = args.audio_file.with_suffix(" Transcript.json" if args.json else " Transcript.txt")
        else:
            out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.json:
            json.dump(segments, out_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
        else:
            out_path.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", out_path)
    else:
        print("\n".join(text.splitlines()[:15]))


if __name__ == "__main__":
    _cli()
