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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def transcribe_file(audio_file: Path, output_folder: Path, move_after: bool):
    """
    Call the existing transcribe_audio.py script for the given audio file.
    The transcript file is saved either in the output folder (if provided)
    or in the same folder as the audio file.
    """
    transcript_filename = audio_file.stem + ".txt"
    if output_folder:
        transcript_path = output_folder / transcript_filename
    else:
        transcript_path = audio_file.parent / transcript_filename

    # Assume transcribe_audio.py is located in the same directory as this script.
    script_path = Path(__file__).parent / "transcribe_audio.py"
    if not script_path.exists():
        logger.error(f"Transcription script {script_path} not found.")
        return

    command = [sys.executable, str(script_path), str(audio_file), "-o", str(transcript_path)]
    logger.info(f"Transcribing {audio_file} to {transcript_path}")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info(f"Transcription finished for {audio_file}.\n{result.stdout}")
        if result.stderr:
            logger.error(f"Errors during transcription of {audio_file}:\n{result.stderr}")
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_file}: {e}")
        return

    # If --move is set and an output folder is provided, move the audio file.
    if move_after and output_folder:
        try:
            dest_path = output_folder / audio_file.name
            shutil.move(str(audio_file), str(dest_path))
            logger.info(f"Moved audio file {audio_file} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to move {audio_file} to {output_folder}: {e}")

class AudioFileHandler(FileSystemEventHandler):
    def __init__(self, regex_pattern: str, output_folder: Path, move_after: bool):
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.output_folder = output_folder
        self.move_after = move_after

    def process(self, file_path: Path):
        # Wait briefly to ensure the file is completely written.
        time.sleep(1)
        if not file_path.is_file():
            return
        if self.regex and not self.regex.search(file_path.name):
            logger.info(f"File {file_path} does not match regex; skipping.")
            return

        # Optionally check common audio extensions.
        valid_extensions = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
        if file_path.suffix.lower() not in valid_extensions:
            logger.info(f"File {file_path} is not a recognized audio file; skipping.")
            return

        transcribe_file(file_path, self.output_folder, self.move_after)

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        logger.info(f"Detected new file: {file_path}")
        self.process(file_path)

def process_existing_files(input_folder: Path, regex_pattern: str, output_folder: Path, move_after: bool):
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
            transcribe_file(file, output_folder, move_after)

def main():
    parser = argparse.ArgumentParser(
        description="Watch an input folder for new audio files and transcribe them using transcribe_audio.py"
    )
    parser.add_argument("input_folder", nargs="?", default=".", help="Folder to watch for audio files")
    parser.add_argument("--regex", type=str, help="Regex pattern to match filenames")
    parser.add_argument("--output-folder", type=str, help="Folder where transcripts (and optionally audio files) will be saved")
    parser.add_argument("--move", action="store_true", help="Move audio file to output folder after transcription (requires --output-folder)")
    parser.add_argument("--process-existing", action="store_true", help="Process files already in the folder at startup")
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
        logger.error("The --move option requires an --output-folder to be specified.")
        sys.exit(1)

    # Process files already in the folder if requested.
    if args.process_existing:
        process_existing_files(input_folder, args.regex, output_folder, args.move)

    # Set up the folder observer.
    event_handler = AudioFileHandler(args.regex, output_folder, args.move)
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
