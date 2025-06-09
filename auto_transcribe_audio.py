#!/usr/bin/env python3
"""auto_transcribe_audio.py

High‑level features
-------------------
* **Config‑driven defaults** – if not supplied on the CLI, the watcher will
  monitor `[Downloads]` and emit results in
  `[Documents]/Sound Recordings/transcribed` (Windows‑style known folders).
* **Special‑folder tokens** – strings like `[Downloads]` or
  `[Documents]/Subdir` are expanded to the current user’s known locations on
  Windows and sensible fall‑backs (`~/Downloads`, `~/Documents`, …) on other
  platforms.
* **Single‑shot mode** – if every positional argument resolves to a file (or
  glob) the script processes the files once and exits.  Otherwise it starts a
  Watchdog observer.

Usage examples
--------------
Single file::

    python auto_transcribe_audio.py "~/Downloads/meeting.m4a"

All M4As in Downloads (no watcher)::

    python auto_transcribe_audio.py "[Downloads]/*.m4a" --move

Continuous folder watch (uses config defaults)::

    python auto_transcribe_audio.py --process-existing --move
"""
from __future__ import annotations

import argparse
import configparser
import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, List

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ──────────────────────────────  logging  ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────  known‑folder helpers  ───────────────────────
_KNOWN_FOLDER_GUIDS: Dict[str, str] = {
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "music":     "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "pictures":  "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "videos":    "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}

class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]

def _guid_from_str(guid_str: str) -> _GUID:
    u = uuid.UUID(guid_str)
    g = _GUID()
    ctypes.memmove(ctypes.byref(g), u.bytes_le, 16)
    return g

def _win_known_folder(guid_str: str) -> Path:
    shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(_GUID), ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    path_ptr = ctypes.c_wchar_p()
    if shell32.SHGetKnownFolderPath(_guid_from_str(guid_str), 0, None, ctypes.byref(path_ptr)):
        raise OSError("Failed to retrieve known folder")
    return Path(path_ptr.value)

def _fallback(folder_name: str) -> Path:
    """Return ~/FolderName as a cross‑platform fallback."""
    return Path.home() / folder_name.capitalize()

def get_special_folder(name: str) -> Path:
    name_lc = name.lower()
    if os.name == "nt" and name_lc in _KNOWN_FOLDER_GUIDS:
        try:
            return _win_known_folder(_KNOWN_FOLDER_GUIDS[name_lc])
        except Exception as exc:
            logger.warning("Failed to get Windows known folder '%s': %s", name, exc)
    return _fallback(name_lc)

_SPECIAL_PATTERN = re.compile(r"^\[(downloads|documents|videos|music|pictures)\](.*)$", re.IGNORECASE)

def resolve_path(token: str) -> Path:
    """Expand special tokens like "[Downloads]/sub" into actual paths."""
    m = _SPECIAL_PATTERN.match(token)
    if m:
        base = get_special_folder(m.group(1))
        rest = m.group(2).lstrip("/\\")
        return base if not rest else base / rest
    return Path(token).expanduser()

# ──────────────────────────────  globals  ──────────────────────────────
process_lock = Lock()
processed_files: set[Path] = set()

# ──────────────────────────────  handler  ──────────────────────────────
class AudioFileHandler(FileSystemEventHandler):
    """Watchdog handler that defers heavy work to helper functions."""

    valid_audio_ext = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
    valid_video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".webp"}

    def __init__(self, regex: str | None, out_dir: Path | None, move: bool, cfg: configparser.ConfigParser):
        super().__init__()
        self.regex = re.compile(regex) if regex else None
        self.out_dir = out_dir
        self.move = move
        self.cfg = cfg
        self.stability = int(cfg["Watcher"].get("stability_seconds", 30))
        self.temp_patterns = [r"\.tmp$", r"~syncthing~.*", r"\.part$", r"\.crdownload$", r"\.download$"]
        self.pending: dict[Path, float] = {}
        self.stop_event = Event()
        Thread(target=self._scan_pending, daemon=True).start()

    # ────────────────────  stability / validation helpers  ────────────────────
    def _is_valid(self, f: Path) -> bool:
        if f.suffix.lower() not in (self.valid_audio_ext | self.valid_video_ext):
            return False
        if any(re.search(p, f.name, re.IGNORECASE) for p in self.temp_patterns):
            return False
        try:
            size0 = f.stat().st_size
            time.sleep(self.stability)
            return size0 == f.stat().st_size and size0 > 0
        except OSError:
            return False

    def _scan_pending(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()
            for fp, ts in list(self.pending.items()):
                if now - ts > 600 and self._is_valid(fp):
                    self._handle(fp)
                    self.pending.pop(fp, None)
            time.sleep(60)

    # ──────────────────────────────  core  ──────────────────────────────
    def _handle(self, f: Path):
        with process_lock:
            if f in processed_files:
                return
            processed_files.add(f)
        if not self._is_valid(f):
            return
        if self.regex and not self.regex.search(f.name):
            return
        from auto_transcribe_audio_helpers import process_audio_file  # local helper
        logger.info("Processing %s", f)
        if f.suffix.lower() in self.valid_video_ext:
            tmp_wav = f.with_suffix("_temp.wav")
            cmd = [
                "ffmpeg", "-y", "-nostdin", "-i", str(f),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", str(tmp_wav),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                process_audio_file(tmp_wav, self.cfg, self.out_dir, self.move, source_file=f)
            finally:
                tmp_wav.unlink(missing_ok=True)
        else:
            process_audio_file(f, self.cfg, self.out_dir, self.move, source_file=f)

    # ────────────────────  watchdog event overrides  ────────────────────
    def on_created(self, event):
        if not event.is_directory:
            self.pending[Path(event.src_path)] = time.time()

    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path) in self.pending:
            self.pending[Path(event.src_path)] = time.time()

    def on_moved(self, event):
        if not event.is_directory:
            dest = Path(event.dest_path)
            if self._is_valid(dest):
                self._handle(dest)
            else:
                self.pending[dest] = time.time()

    def stop(self):
        self.stop_event.set()

# ──────────────────────────  one‑shot processing  ─────────────────────────

def process_files_once(files: List[Path], cfg: configparser.ConfigParser, out_dir: Path | None, move: bool):
    """Process a finite list of files and exit."""
    from auto_transcribe_audio_helpers import process_audio_file  # local helper
    for f in files:
        logger.info("Single‑shot: processing %s", f)
        process_audio_file(f, cfg, out_dir, move, source_file=f)
    logger.info("Processed %d file(s) in single‑shot mode.", len(files))

# ───────────────────────────────  main  ────────────────────────────────

def main():  # pylint: disable=too-many-branches,too-many-locals
    p = argparse.ArgumentParser("Transcribe & summarise audio/video")
    p.add_argument("paths", nargs="*", help="Files, globs or directory token [Downloads] …")
    p.add_argument("--regex")
    p.add_argument("--output-folder")
    p.add_argument("--move", action="store_true", help="Copy/move outputs and delete source when complete")
    p.add_argument("--process-existing", action="store_true", help="Process files already present in watch folder before starting observer")
    p.add_argument("--config", default="transcription_config.ini", help="INI file with Transcription/Summarization/Watcher sections")
    args = p.parse_args()

    # ─────────────  configuration  ─────────────
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # Ensure required sections exist
    for section in ("Watcher", "Transcription", "Summarization"):
        if section not in cfg:
            cfg[section] = {}

    watcher_cfg = cfg["Watcher"]
    watcher_cfg.setdefault("input_folder", "[Downloads]")
    watcher_cfg.setdefault("output_folder", "[Documents]/Sound Recordings/transcribed")
    watcher_cfg.setdefault("stability_seconds", "30")
    watcher_cfg.setdefault("move_default", "false")

    # Resolve output folder
    out_dir: Path | None = None
    if args.move or watcher_cfg.getboolean("move_default", fallback=False):
        out_dir = resolve_path(args.output_folder or watcher_cfg["output_folder"])
        out_dir.mkdir(parents=True, exist_ok=True)

    move_after = args.move or watcher_cfg.getboolean("move_default", fallback=False)

    # Expand positional arguments into paths/globs (strip quotes first)
    tokens = args.paths or [watcher_cfg["input_folder"]]
    expanded: List[Path] = []
    for tok in tokens:
        tok = tok.strip("'\"")  # remove leading/trailing single/double quotes
        if any(ch in tok for ch in "*?"):
            base = resolve_path(Path(tok).parent.as_posix())
            expanded.extend(base.glob(Path(tok).name))
        else:
            expanded.append(resolve_path(tok))

    # Determine mode: single‑shot if every expanded item is a file
    if expanded and all(fp.is_file() for fp in expanded):
        process_files_once(expanded, cfg, out_dir, move_after)
        return

    watch_dir = expanded[0]
    if not watch_dir.exists():
        logger.error("Watch directory '%s' does not exist", watch_dir)
        sys.exit(1)
    if not watch_dir.is_dir():
        logger.error("'%s' is not a directory (did you forget quotes around glob?)", watch_dir)
        sys.exit(1)

    # Optionally process existing files before starting observer
    if args.process_existing:
        from auto_transcribe_audio_helpers import process_existing_files
        process_existing_files(watch_dir, args.regex, out_dir, move_after, cfg)

    # ─────  start watchdog observer  ─────
    handler = AudioFileHandler(args.regex, out_dir, move_after, cfg)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    logger.info("Watching %s … (Ctrl‑C to quit)", watch_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping observer…")
        handler.stop()
        observer.stop()
    observer.join()


if __name__ == "__main__":  # pragma: no cover
    main()
