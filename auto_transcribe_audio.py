#!/usr/bin/env python3
import os
import sys
import time
import re
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import configparser
from threading import Thread, Event

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Globals for thread-safe file processing
process_lock = Lock()
processed_files = set()

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, regex_pattern: str, output_folder: Path, move_after: bool, config):
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.output_folder = output_folder
        self.move_after = move_after
        self.config = config
        self.valid_audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
        self.valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        self.temp_patterns = [
            r"\.tmp$", r"~syncthing~.*", r"\.part$", r"\.crdownload$", r"\.download$"
        ]
        self.stability_seconds = int(config['Watcher'].get('stability_seconds', 30))
        self.pending_files = {}  # file_path: last_mod_time
        self.stop_event = Event()
        self.check_thread = Thread(target=self.check_stable_files, daemon=True)
        self.check_thread.start()

    def check_stable_files(self):
        while not self.stop_event.is_set():
            time.sleep(60)
            current_time = time.time()
            files_to_process = []
            for file_path, last_mod_time in list(self.pending_files.items()):
                if current_time - last_mod_time > 600:
                    if self.is_valid_file(file_path):
                        files_to_process.append(file_path)
                    else:
                        logging.info(f"File {file_path} still not valid; continuing to monitor.")
            for file_path in files_to_process:
                self.process_file(file_path)
                del self.pending_files[file_path]
            time.sleep(60)

    def is_valid_file(self, file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix not in self.valid_audio_extensions and suffix not in self.valid_video_extensions:
            return False
        file_name = file_path.name
        if any(re.search(pattern, file_name, re.IGNORECASE) for pattern in self.temp_patterns):
            return False
        try:
            initial_size = file_path.stat().st_size
            time.sleep(self.stability_seconds)
            final_size = file_path.stat().st_size
            return initial_size == final_size and initial_size > 0
        except OSError:
            return False

    def process_file(self, file_path: Path):
        with process_lock:
            if file_path in processed_files:
                return
            processed_files.add(file_path)
        self._process_file_inner(file_path)

    def _process_file_inner(self, file_path: Path):
        if not self.is_valid_file(file_path):
            logging.info(f"Skipping {file_path}: not a valid file or still being written.")
            return
        if self.regex and not self.regex.search(file_path.name):
            logging.info(f"File {file_path} does not match regex; skipping.")
            return
        logging.info(f"Processing file: {file_path}")
        if file_path.suffix.lower() in self.valid_video_extensions:
            audio_file = file_path.parent / f"{file_path.stem}_temp.wav"
            extract_cmd = [
                'ffmpeg', '-y', '-nostdin', '-i', str(file_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', str(audio_file)
            ]
            try:
                subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
                logging.info(f"Extracted audio from {file_path} to {audio_file}")
                process_audio_file(audio_file, self.config, self.output_folder, self.move_after, source_file=file_path)
                if audio_file.exists():
                    audio_file.unlink()
                    logging.info(f"Deleted temporary audio file: {audio_file}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to extract audio from {file_path}: {e.stderr}")
            except Exception as e:
                logging.error(f"Unexpected error processing {file_path}: {e}")
        else:
            process_audio_file(file_path, self.config, self.output_folder, self.move_after, source_file=file_path)

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        logging.info(f"Detected new file: {file_path}")
        self.pending_files[file_path] = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path in self.pending_files:
            self.pending_files[file_path] = time.time()

    def on_moved(self, event):
        if event.is_directory:
            return
        file_path = Path(event.dest_path)
        logging.info(f"Detected renamed file: {file_path}")
        if self.is_valid_file(file_path):
            self.process_file(file_path)
        else:
            self.pending_files[file_path] = time.time()

    def stop(self):
        self.stop_event.set()
        self.check_thread.join()

def process_audio_file(audio_file: Path, config: configparser.ConfigParser, output_folder: Path, move_after: bool, source_file: Path):
    """
    Transcribe the audio file and generate a summary, saving both to the output folder (if provided)
    or the input folder. Move audio, transcript, summary, and context files if --move is set.
    Ensure source file is not deleted if processing fails.

    Args:
        audio_file (Path): Path to the audio file to transcribe (may be temporary for videos).
        config (configparser.ConfigParser): Configuration settings.
        output_folder (Path): Destination folder for output files.
        move_after (bool): Whether to move files to output folder.
        source_file (Path): Original audio or video file (not deleted until success).
    """
    transcript_ext = config['Transcription'].get('transcript_output_format', 'txt').split(';')[0].strip()
    if transcript_ext != 'txt':
        logger.warning("Transcript output format is not 'txt'. Summarization may fail if not text.")
    transcript_filename = f"{source_file.stem}.{transcript_ext}"
    transcript_path = output_folder / transcript_filename if output_folder else source_file.parent / transcript_filename

    transcribe_script = Path(__file__).parent / "transcribe_audio.py"
    if not transcribe_script.exists():
        logger.error(f"Transcription script {transcribe_script} not found.")
        return

    transcribe_cmd = [
        sys.executable,
        str(transcribe_script),
        str(audio_file),
        "--method", config['Transcription'].get('method', 'whisper').split(';')[0].strip(),
        "--model", config['Transcription'].get('model', 'base').split(';')[0].strip(),
        "-o", str(transcript_path),
    ]
    temp_dir = config['Transcription'].get('temp_dir', '').split(';')[0].strip()
    if temp_dir:
        transcribe_cmd.extend(["--temp-dir", temp_dir])
    if config['Transcription'].getboolean('speakers', False):
        transcribe_cmd.append("--speakers")
    if transcript_ext == 'json':
        transcribe_cmd.append("--json")

    logger.info(f"Transcribing {audio_file} to {transcript_path}")
    try:
        result = subprocess.run(transcribe_cmd, capture_output=True, text=True)
        logger.info(f"Transcription finished for {audio_file}.\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors during transcription of {audio_file}:\n{result.stderr}")
        if not transcript_path.exists():
            logger.error(f"Transcription failed: {transcript_path} was not created.")
            return
        # Check if transcript is empty or only contains timestamp
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read().strip()
        if not transcript_content or (transcript_content.startswith("Date and Time:") and len(transcript_content.splitlines()) <= 2):
            logger.warning(f"Transcript {transcript_path} is empty or contains only timestamp. Skipping summarization.")
            return
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_file}: {e}")
        return

    summarize_script = Path(__file__).parent / "summarize_transcript.py"
    if not summarize_script.exists():
        logger.error(f"Summarization script {summarize_script} not found.")
        return

    summary_ext = config['Summarization'].get('summary_output_format', 'docx').split(';')[0].strip()
    rename_from_context = config['Summarization'].getboolean('rename_from_context', False)
    summary_filename = f"{source_file.stem} Summary.{summary_ext}"
    summary_path = output_folder / summary_filename if output_folder else source_file.parent / summary_filename

    summarize_cmd = [
        sys.executable,
        str(summarize_script),
        str(transcript_path),
        "--model", config['Summarization'].get('model', 'gpt-4o-mini').split(';')[0].strip(),
        "--api-key-file", config['Summarization'].get('api_key_file', 'api_keys.json').split(';')[0].strip(),
        "--output-format", summary_ext,
        "--output-file", str(summary_path),
    ]
    speaker_hints = config['Summarization'].get('speaker_hints', '').split(';')[0].strip()
    if speaker_hints:
        summarize_cmd.extend(["--speaker-hints", speaker_hints])
    if rename_from_context:
        summarize_cmd.append("--rename-from-context")

    logger.info(f"Generating summary for {transcript_path} to {summary_path}")
    try:
        result = subprocess.run(summarize_cmd, capture_output=True, text=True)
        logger.info(f"Summarization finished for {transcript_path}.\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors during summarization of {transcript_path}:\n{result.stderr}")

        renamed_base = None
        if rename_from_context:
            output_dir = output_folder if output_folder else source_file.parent
            for line in result.stdout.splitlines():
                if "Summary saved to" in line:
                    summary_path = Path(line.split("Summary saved to ")[1].strip())
                    renamed_base = summary_path.stem
                    break
            if not renamed_base:
                context_files = list(output_dir.glob("*-context.json"))
                if context_files:
                    latest_context = max(context_files, key=os.path.getmtime)
                    renamed_base = latest_context.stem.replace("-context", "")
                    summary_path = output_dir / f"{renamed_base}.{summary_ext}"
                    logger.info(f"Determined renamed base from context file: {renamed_base}")
            if not renamed_base:
                logger.warning("Could not determine renamed base name; using default.")
                renamed_base = f"{source_file.stem} Summary"
            context_filename = f"{renamed_base}-context.json"
            context_path = output_dir / context_filename
        else:
            context_filename = f"{source_file.stem}-context.json"
            context_path = output_dir / context_filename

        # Only move files if all outputs are successfully created
        if move_after and output_folder:
            try:
                # Copy source file instead of moving to preserve original
                audio_dest = output_folder / (f"{renamed_base}{source_file.suffix}" if rename_from_context else source_file.name)
                shutil.copy2(str(source_file), str(audio_dest))
                logger.info(f"Copied source file {source_file} to {audio_dest}")

                if transcript_path.exists():
                    transcript_dest = output_folder / (f"{renamed_base}.{transcript_ext}" if rename_from_context else transcript_filename)
                    shutil.move(str(transcript_path), str(transcript_dest))
                    logger.info(f"Moved transcript {transcript_path} to {transcript_dest}")
                    transcript_path = transcript_dest

                if summary_path.exists() and summary_path.parent != output_folder:
                    summary_dest = output_folder / summary_path.name
                    shutil.move(str(summary_path), str(summary_dest))
                    logger.info(f"Moved summary {summary_path} to {summary_dest}")
                    summary_path = summary_dest

                if context_path.exists() and context_path.parent != output_folder:
                    context_dest = output_folder / context_path.name
                    shutil.move(str(context_path), str(context_dest))
                    logger.info(f"Moved context {context_path} to {context_dest}")

                # Verify all files exist before deleting original
                if all(p.exists() for p in [audio_dest, transcript_path, summary_path, context_path]):
                    source_file.unlink()
                    logger.info(f"Deleted original source file {source_file} after successful processing.")
                else:
                    logger.warning(f"Not deleting {source_file}: not all output files were created successfully.")
            except Exception as e:
                logger.error(f"Failed to move/copy files to {output_folder}: {e}")
                logger.warning(f"Preserving original source file {source_file} due to error.")

    except Exception as e:
        logger.error(f"Failed to summarize {transcript_path}: {e}")
        logger.warning(f"Preserving original source file {source_file} due to summarization failure.")

def process_existing_files(input_folder: Path, regex_pattern: str, output_folder: Path, move_after: bool, config: configparser.ConfigParser):
    logger.info("Processing existing files in folder.")
    regex_compiled = re.compile(regex_pattern) if regex_pattern else None
    valid_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".mp4", ".avi", ".mov", ".mkv"]
    for file in input_folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            if regex_compiled and not regex_compiled.search(file.name):
                continue
            logger.info(f"Processing existing file: {file}")
            handler = AudioFileHandler(regex_pattern, output_folder, move_after, config)
            handler.process_file(file)

def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder for audio and video files, extract audio from videos, transcribe them, and generate summaries."
    )
    parser.add_argument("input_folder", nargs="?", default=".", help="Folder to watch for audio and video files")
    parser.add_argument("--regex", type=str, help="Regex pattern to match filenames")
    parser.add_argument("--output-folder", type=str, help="Folder for transcripts and summaries")
    parser.add_argument("--move", action="store_true", help="Move audio, transcript, and summary files to output folder after processing")
    parser.add_argument("--process-existing", action="store_true", help="Process existing files at startup")
    parser.add_argument("--config", type=str, default="auto_transcribe_config.ini", help="Path to config file")
    args = parser.parse_args()

    input_folder = Path(args.input_folder).resolve()
    if not input_folder.exists():
        logger.error(f"Input folder {input_folder} does not exist.")
        sys.exit(1)

    output_folder = Path(args.output_folder).resolve() if args.output_folder else None
    if output_folder and not output_folder.exists():
        try:
            output_folder.mkdir(parents=True)
            logger.info(f"Created output folder {output_folder}")
        except Exception as e:
            logger.error(f"Failed to create output folder {output_folder}: {e}")
            sys.exit(1)

    if args.move and not output_folder:
        logger.error("The --move option requires an --output-folder.")
        sys.exit(1)

    config = configparser.ConfigParser()
    if not config.read(args.config):
        logger.warning(f"Config file {args.config} not found. Creating with defaults.")
        config['Transcription'] = {
            'method': 'whisper',
            'speakers': 'True',
            'model': 'base',
            'temp_dir': '',
            'transcript_output_format': 'txt'
        }
        config['Summarization'] = {
            'model': 'gpt-4o-mini',
            'api_key_file': 'api_keys.json',
            'summary_output_format': 'docx',
            'speaker_hints': ''
        }
        with open(args.config, 'w') as configfile:
            config.write(configfile)
        logger.info(f"Created default config file at {args.config}")

    if 'Transcription' not in config or 'Summarization' not in config:
        logger.error(f"Config file {args.config} missing required sections.")
        sys.exit(1)

    if args.process_existing:
        process_existing_files(input_folder, args.regex, output_folder, args.move, config)

    event_handler = AudioFileHandler(args.regex, output_folder, args.move, config)
    observer = Observer()
    observer.schedule(event_handler, str(input_folder), recursive=False)
    observer.start()
    logger.info(f"Watching folder: {input_folder} for new audio and video files.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        event_handler.stop()
        logger.info("Stopping folder watcher.")
    observer.join()

if __name__ == "__main__":
    main()