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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def process_audio_file(audio_file: Path, config: configparser.ConfigParser, output_folder: Path, move_after: bool):
    """
    Transcribe the audio file and generate a summary, saving both to the output folder (if provided)
    or the input folder. Move audio, transcript, summary, and context files if --move is set.
    """
    # Strip comments and whitespace from config values
    transcript_ext = config['Transcription'].get('transcript_output_format', 'txt').split(';')[0].strip()
    if transcript_ext != 'txt':
        logger.warning("Transcript output format is not 'txt'. Summarization may fail if not text.")
    transcript_filename = f"{audio_file.stem}.{transcript_ext}"
    transcript_path = output_folder / transcript_filename if output_folder else audio_file.parent / transcript_filename

    # Check for transcribe_audio.py
    transcribe_script = Path(__file__).parent / "transcribe_audio.py"
    if not transcribe_script.exists():
        logger.error(f"Transcription script {transcribe_script} not found.")
        return

    # Build transcription command
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

    # Run transcription
    logger.info(f"Transcribing {audio_file} to {transcript_path}")
    try:
        result = subprocess.run(transcribe_cmd, capture_output=True, text=True)
        logger.info(f"Transcription finished for {audio_file}.\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors during transcription of {audio_file}:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_file}: {e}")
        return

    # Check for summarize_transcript.py
    summarize_script = Path(__file__).parent / "summarize_transcript.py"
    if not summarize_script.exists():
        logger.error(f"Summarization script {summarize_script} not found.")
        return

    # Build summarization command
    summary_ext = config['Summarization'].get('summary_output_format', 'docx').split(';')[0].strip()
    rename_from_context = config['Summarization'].getboolean('rename_from_context', False)
    summary_filename = f"{audio_file.stem} Summary.{summary_ext}"  # Temporary name
    summary_path = output_folder / summary_filename if output_folder else audio_file.parent / summary_filename

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

    # Run summarization and capture output to determine renamed files
    logger.info(f"Generating summary for {transcript_path} to {summary_path}")
    try:
        result = subprocess.run(summarize_cmd, capture_output=True, text=True)
        logger.info(f"Summarization finished for {transcript_path}.\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors during summarization of {transcript_path}:\n{result.stderr}")

        # Determine the actual renamed base name if --rename-from-context is used
        renamed_base = None
        if rename_from_context:
            output_dir = output_folder if output_folder else audio_file.parent
            # Method 1: Parse stdout to find the saved summary file path
            for line in result.stdout.splitlines():
                if "Summary saved to" in line:
                    summary_path = Path(line.split("Summary saved to ")[1].strip())
                    renamed_base = summary_path.stem  # e.g., "20250306-164134-Collaborative-Efforts..."
                    break
            # Method 2: Fallback to checking for context file in output directory
            if not renamed_base:
                context_files = list(output_dir.glob("*-context.json"))
                if context_files:
                    # Use the most recently modified context file
                    latest_context = max(context_files, key=os.path.getmtime)
                    renamed_base = latest_context.stem.replace("-context", "")
                    summary_path = output_dir / f"{renamed_base}.{summary_ext}"
                    logger.info(f"Determined renamed base from context file: {renamed_base}")
            if not renamed_base:
                logger.warning("Could not determine renamed base name; using default.")
                renamed_base = f"{audio_file.stem} Summary"
            context_filename = f"{renamed_base}-context.json"
            context_path = output_dir / context_filename
        else:
            context_filename = f"{audio_file.stem}-context.json"
            context_path = output_dir / context_filename

    except Exception as e:
        logger.error(f"Failed to summarize {transcript_path}: {e}")
        return

    # Move audio, transcript, summary, and context if --move is set
    if move_after and output_folder:
        try:
            # Move audio file with renamed base if applicable
            audio_dest = output_folder / (f"{renamed_base}{audio_file.suffix}" if rename_from_context else audio_file.name)
            shutil.move(str(audio_file), str(audio_dest))
            logger.info(f"Moved audio file {audio_file} to {audio_dest}")

            # Move transcript file, renaming if --rename-from-context
            if transcript_path.exists():
                transcript_dest = output_folder / (f"{renamed_base}.{transcript_ext}" if rename_from_context else transcript_filename)
                shutil.move(str(transcript_path), str(transcript_dest))
                logger.info(f"Moved transcript {transcript_path} to {transcript_dest}")
                transcript_path = transcript_dest

            # Move summary file (already renamed by summarize_transcript.py)
            if summary_path.exists() and summary_path.parent != output_folder:
                summary_dest = output_folder / summary_path.name
                shutil.move(str(summary_path), str(summary_dest))
                logger.info(f"Moved summary {summary_path} to {summary_dest}")
                summary_path = summary_dest

            # Move context file (already renamed by summarize_transcript.py)
            if context_path.exists() and context_path.parent != output_folder:
                context_dest = output_folder / context_path.name
                shutil.move(str(context_path), str(context_dest))
                logger.info(f"Moved context {context_path} to {context_dest}")

        except Exception as e:
            logger.error(f"Failed to move files to {output_folder}: {e}")

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, regex_pattern: str, output_folder: Path, move_after: bool, config: configparser.ConfigParser):
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.output_folder = output_folder
        self.move_after = move_after
        self.config = config

    def process(self, file_path: Path):
        # Wait briefly to ensure file is fully written
        time.sleep(1)
        if not file_path.is_file():
            return
        if self.regex and not self.regex.search(file_path.name):
            logger.info(f"File {file_path} does not match regex; skipping.")
            return

        valid_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
        if file_path.suffix.lower() not in valid_extensions:
            logger.info(f"File {file_path} is not a recognized audio file; skipping.")
            return

        process_audio_file(file_path, self.config, self.output_folder, self.move_after)

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        logger.info(f"Detected new file: {file_path}")
        self.process(file_path)

def process_existing_files(input_folder: Path, regex_pattern: str, output_folder: Path, move_after: bool, config: configparser.ConfigParser):
    logger.info("Processing existing files in folder.")
    regex_compiled = re.compile(regex_pattern) if regex_pattern else None
    valid_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
    for file in input_folder.iterdir():
        if file.is_file():
            if regex_compiled and not regex_compiled.search(file.name):
                continue
            if file.suffix.lower() not in valid_extensions:
                continue
            logger.info(f"Processing existing file: {file}")
            process_audio_file(file, config, output_folder, move_after)

def main():
    parser = argparse.ArgumentParser(
        description="Watch a folder for audio files, transcribe them, and generate summaries."
    )
    parser.add_argument("input_folder", nargs="?", default=".", help="Folder to watch for audio files")
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

    # Load configuration
    config = configparser.ConfigParser()
    if not config.read(args.config):
        logger.warning(f"Config file {args.config} not found. Creating with defaults.")
        config['Transcription'] = {
            'method': 'whisper',
            'speakers': 'True',
            'model': 'base',
            'temp_dir': '',  # Empty means None
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
    logger.info(f"Watching folder: {input_folder} for new audio files.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping folder watcher.")
    observer.join()

if __name__ == "__main__":
    main()