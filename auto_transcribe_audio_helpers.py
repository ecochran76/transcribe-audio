#!/usr/bin/env python3
"""Helpers for *auto_transcribe_audio.py*

This module contains the per‑file pipeline and the batch helper.  It keeps
all naming, renaming and verification logic in **one** place so we do not
repeat mistakes across entry‑points.

Behaviour
---------
* `rename_from_context = true` → the **base name** returned by the
  summariser is applied consistently to **all** artefacts (audio copy,
  transcript, docx summary, context JSON).
* The verification step checks the *post‑rename* paths – so the original
  audio is deleted only when *all* outputs really exist in their final
  location.
* Works with the current summariser which emits
  ``<title>-context.docx`` / ``<title>-context.json``.
"""

from __future__ import annotations

import configparser
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

__all__ = [
    "process_audio_file",
    "process_existing_files",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Run *cmd* and return the CompletedProcess **always** logging stderr."""
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stderr:
        logger.error("%s stderr:\n%s", Path(cmd[0]).stem, proc.stderr.strip())
    return proc


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_audio_file(
    audio_file: Path,
    cfg: configparser.ConfigParser,
    output_folder: Path | None,
    move_after: bool,
    *,
    source_file: Path,
) -> None:
    """Transcribe → summarise → rename → (optionally) move a single file.

    Robust against *any* summary naming convention:
    • Summariser may emit either ``<base>-context.docx`` or just
      ``<base>.docx`` plus ``<base>-context.json``.
    • We locate the newest *docx* and *json* created after the summariser
      started and derive *renamed_base* from whichever file includes
      "-context".
    """

    tr_cfg = cfg["Transcription"]
    sum_cfg = cfg["Summarization"]

    out_dir: Path = output_folder or source_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 1. Transcription ---------------------------
    transcript_ext = tr_cfg.get("transcript_output_format", "txt").split(";")[0].strip()
    transcript_out = out_dir / f"{source_file.stem}.{transcript_ext}"

    transcribe_cmd = [
        sys.executable,
        str(Path(__file__).with_name("transcribe_audio.py")),
        str(audio_file),
        "--method", tr_cfg.get("method", "whisper").split(";")[0].strip(),
        "--model", tr_cfg.get("model", "base").split(";")[0].strip(),
        "-o", str(transcript_out),
    ]
    if tr_cfg.get("temp_dir", "").strip():
        transcribe_cmd += ["--temp-dir", tr_cfg["temp_dir"].strip()]
    if tr_cfg.getboolean("speakers", False):
        transcribe_cmd.append("--speakers")
    if transcript_ext.lower() == "json":
        transcribe_cmd.append("--json")

    logger.info("Transcribing %s", audio_file.name)
    _run(transcribe_cmd)
    if not transcript_out.exists() or transcript_out.stat().st_size < 32:
        logger.warning("Transcript missing/empty for %s – aborting", audio_file.name)
        return

    # -------------------- 2. Summarisation ---------------------------
    summary_ext = sum_cfg.get("summary_output_format", "docx").split(";")[0].strip()
    start_ts = time.time()

    summarize_cmd = [
        sys.executable,
        str(Path(__file__).with_name("summarize_transcript.py")),
        str(transcript_out),
        "--model", sum_cfg.get("model", "gpt-4o-mini").split(";")[0].strip(),
        "--api-key-file", sum_cfg.get("api_key_file", "api_keys.json"),
        "--output-format", summary_ext,
    ]
    if sum_cfg.get("speaker_hints", "").strip():
        summarize_cmd += ["--speaker-hints", sum_cfg["speaker_hints"].strip()]
    if sum_cfg.getboolean("rename_from_context", False):
        summarize_cmd.append("--rename-from-context")

    _run(summarize_cmd)

    # -------------------- 3. Locate summary & context ----------------
    # Look for newest docx and json created since *start_ts*
    recent_files = [p for p in out_dir.iterdir() if p.stat().st_mtime >= start_ts - 1]
    recent_docx  = max((p for p in recent_files if p.suffix.lower() == f".{summary_ext}"),
                       default=None, key=lambda p: p.stat().st_mtime)
    recent_json  = max((p for p in recent_files if p.suffix.lower() == ".json"),
                       default=None, key=lambda p: p.stat().st_mtime)

    if recent_docx is None or recent_json is None:
        logger.warning("Summary/context files missing – leaving original audio in place")
        return

    if "-context" in recent_docx.stem:
        renamed_base = recent_docx.stem.replace("-context", "")
    elif "-context" in recent_json.stem:
        renamed_base = recent_json.stem.replace("-context", "")
    else:
        renamed_base = recent_docx.stem  # best effort

    summary_out = recent_docx
    context_json = recent_json

    # -------------------- 4. Rename transcript -----------------------
    if sum_cfg.getboolean("rename_from_context", False):
        new_tx = transcript_out.with_name(f"{renamed_base}{transcript_out.suffix}")
        if not new_tx.exists():
            transcript_out.rename(new_tx)
        transcript_out = new_tx

    # -------------------- 5. Move/copy originals ---------------------
    audio_dest: Path | None = None
    if move_after and output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
        audio_dest = output_folder / f"{renamed_base}{source_file.suffix}"
        shutil.copy2(source_file, audio_dest)
        for f in (transcript_out, summary_out, context_json):
            dest = output_folder / f.name
            if f.parent != output_folder:
                shutil.move(str(f), dest)
        transcript_out = output_folder / transcript_out.name
        summary_out = output_folder / summary_out.name
        context_json = output_folder / context_json.name

    # -------------------- 6. Verify & clean --------------------------------
    wanted = [transcript_out, summary_out, context_json]
    if audio_dest is not None:
        wanted.append(audio_dest)

    missing = [p.name for p in wanted if not p.exists()]
    if not missing:
        try:
            source_file.unlink(missing_ok=True)
            logger.info("Cleanup: deleted original %s", source_file.name)
        except Exception as exc:
            logger.warning("Could not delete original %s: %s", source_file.name, exc)
    else:
        logger.warning("Outputs missing – left original in place: %s", ", ".join(missing))


# ---------------------------------------------------------------------------
# Batch helper remains unchanged below
# ---------------------------------------------------------------------------

def process_existing_files(
    input_folder: Path,
    regex_pattern: str | None,
    output_folder: Path | None,
    move_after: bool,
    cfg: configparser.ConfigParser,
) -> None:
    """Process media files already in *input_folder*."""

    valid_exts = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".webp",
    }
    regex = re.compile(regex_pattern) if regex_pattern else None

    count = 0
    for file in input_folder.iterdir():
        if not file.is_file() or file.suffix.lower() not in valid_exts:
            continue
        if regex and not regex.search(file.name):
            continue
        logger.info("Existing file: %s", file.name)
        process_audio_file(file, cfg, output_folder, move_after, source_file=file)
        count += 1
    logger.info("Processed %d existing file(s) in folder scan", count)
