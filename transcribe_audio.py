#!/usr/bin/env python3
# ./transcribe_audio.py audio.mp3 --method whisper --speakers --model small -o output.txt
import os
import sys
import subprocess
import configparser
import pkg_resources
from pathlib import Path
import argparse
import json
import logging
import torch
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
from pyannote.core import Segment

# Define required packages with their exact install names
REQUIRED_PACKAGES = {
    "openai-whisper": "openai-whisper",
    "assemblyai": "assemblyai",
    "soundfile": "soundfile",
    "tqdm": "tqdm",
    "torch": "torch",
    "pyannote.audio": "pyannote-audio",
    "pydub": "pydub",
}

# Configuration file setup
CONFIG_FILE = Path(__file__).parent / "transcription_config.ini"
CONFIG = configparser.ConfigParser()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """
    Check for required packages and install them if missing.

    Returns:
        None; exits with status 1 if installation fails.
    """
    missing = []
    installed = []
    for pkg, install_name in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.get_distribution(pkg)
            installed.append(pkg)
        except pkg_resources.DistributionNotFound:
            missing.append(install_name)
    
    if missing:
        logger.info(f"Installed packages: {', '.join(installed)}")
        logger.info(f"Missing dependencies: {', '.join(missing)}. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            logger.info("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        logger.info("All dependencies are already installed.")

def load_or_create_config():
    """
    Load existing config or create a new one with default values.

    Returns:
        None; exits with status 1 if config file is malformed.
    """
    try:
        if CONFIG_FILE.exists():
            CONFIG.read(CONFIG_FILE)
            # Validate config structure
            if 'DEFAULT' not in CONFIG or 'whisper' not in CONFIG:
                raise configparser.Error("Config file missing required sections.")
        else:
            CONFIG['DEFAULT'] = {'method': 'whisper', 'assemblyai_api_key': '', 'hf_token': ''}
            CONFIG['whisper'] = {'model': 'base', 'temp_dir': str(Path.cwd())}
            save_config()
            logger.info(f"Created new configuration file at {CONFIG_FILE}")
    except configparser.Error as e:
        logger.error(f"Failed to parse config file {CONFIG_FILE}: {e}")
        sys.exit(1)

def save_config():
    """Save the current configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as configfile:
            CONFIG.write(configfile)
    except Exception as e:
        logger.error(f"Failed to save config file {CONFIG_FILE}: {e}")

def setup_whisper(device, model_name):
    """
    Bootstrap Whisper installation and configuration.

    Args:
        device (str): Device to use ('cuda' or 'cpu').
        model_name (str): Whisper model size (e.g., 'base', 'small').

    Returns:
        whisper.Model: Loaded Whisper model instance.
    """
    try:
        import whisper
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        logger.info(f"Loading Whisper model: {model_name} on {device_name}")
        try:
            model = whisper.load_model(model_name, device=device, weights_only=True)
        except TypeError:
            logger.warning("This version of openai-whisper does not support 'weights_only'. Loading without it.")
            model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper is set up successfully on {device_name}.")
        return model
    except ImportError as e:
        logger.error(f"Required module not installed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error setting up Whisper: {e}")
        logger.error("Ensure you have sufficient disk space and compatible hardware.")
        sys.exit(1)

def setup_assemblyai():
    """
    Guide user to set up AssemblyAI and store API key.

    Returns:
        bool: True if setup is successful.
    """
    api_key = CONFIG['DEFAULT'].get('assemblyai_api_key', '')
    if not api_key:
        logger.info("AssemblyAI requires an API key.")
        logger.info("1. Go to https://www.assemblyai.com/ and sign up for a free account.")
        logger.info("2. Get your API key from the dashboard.")
        api_key = input("Enter your AssemblyAI API key: ").strip()
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key
            logger.info("Testing API key...")
            test_transcriber = aai.Transcriber()
            test_transcriber.transcribe("https://example.com/test.wav")
            CONFIG['DEFAULT']['assemblyai_api_key'] = api_key
            save_config()
            logger.info("API key validated and saved.")
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            logger.error("Please try again with a valid key.")
            sys.exit(1)
    else:
        import assemblyai as aai
        aai.settings.api_key = api_key
    return True

def get_hf_token():
    """
    Get Hugging Face token from config or user input with detailed guidance.

    Returns:
        str: Hugging Face access token.
    """
    hf_token = CONFIG['DEFAULT'].get('hf_token', '')
    if not hf_token:
        logger.info("\nPyannote-audio requires a Hugging Face access token and model permissions.")
        logger.info("Follow these steps to set it up:")
        logger.info("1. Go to https://huggingface.co/ and log in or sign up for an account.")
        logger.info("2. Create an access token:")
        logger.info("   - Visit https://huggingface.co/settings/tokens")
        logger.info("   - Click 'New token', name it (e.g., 'pyannote-access'), and select 'read' scope.")
        logger.info("   - Copy the generated token (starts with 'hf_').")
        logger.info("3. Accept model conditions:")
        logger.info("   - Visit https://huggingface.co/pyannote/segmentation-3.0")
        logger.info("   - Review and accept the conditions to access this model.")
        logger.info("   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
        logger.info("   - Review and accept the conditions for this model as well.")
        logger.info("   - You may need to agree to share contact info for pyannote's userbase study.")
        logger.info("Note: These models are gated; acceptance is required even with a valid token.")
        hf_token = input("Enter your Hugging Face access token: ").strip()
        CONFIG['DEFAULT']['hf_token'] = hf_token
        save_config()
        logger.info("Hugging Face token saved.")
    return hf_token

def get_audio_duration(audio_file):
    """
    Get audio duration in seconds using soundfile, fall back to pydub if needed.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        float: Duration in seconds, or None if calculation fails.
    """
    try:
        with sf.SoundFile(audio_file) as f:
            return len(f) / f.samplerate
    except Exception as e:
        logger.debug(f"Could not open {audio_file} with soundfile: {e}")
        logger.info("Attempting conversion to .wav with pydub...")
        try:
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # pydub returns milliseconds
            return duration
        except Exception as e:
            logger.error(f"Failed to convert {audio_file} with pydub: {e}")
            logger.error("Ensure FFmpeg is installed and in PATH.")
            return None

def transcribe_with_whisper(audio_file, model, device, with_speaker_ids=False, temp_dir=None):
    """
    Transcribe audio using Whisper with progress, timestamps, and optional speaker IDs.

    Args:
        audio_file (str): Path to the audio file.
        model (whisper.Model): Loaded Whisper model instance.
        device (str): Device to use ('cuda' or 'cpu').
        with_speaker_ids (bool): Whether to perform speaker diarization.
        temp_dir (str, optional): Directory for temporary files.

    Returns:
        str: Formatted transcription text, or None if failed.
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file {audio_file} not found.")
        return None
    
    temp_dir = temp_dir or CONFIG['whisper'].get('temp_dir', str(Path.cwd()))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Try opening with soundfile, convert to wav if it fails
        temp_wav = None
        try:
            with sf.SoundFile(audio_file):
                pass
            process_file = str(audio_file)
        except Exception as e:
            logger.debug(f"Soundfile failed to open {audio_file}: {e}")
            logger.info("Converting to .wav with pydub for processing...")
            audio = AudioSegment.from_file(audio_file)
            temp_wav = Path(temp_dir) / f"{Path(audio_file).stem}.wav"
            audio.export(temp_wav, format="wav")
            process_file = str(temp_wav)
            logger.info(f"Converted to {process_file}")

        # Log audio file properties
        try:
            with sf.SoundFile(process_file) as f:
                logger.info(f"Audio properties: sample_rate={f.samplerate}, channels={f.channels}, frames={len(f)}")
        except Exception as e:
            logger.debug(f"Could not read audio properties: {e}")

        duration = get_audio_duration(process_file)
        if duration:
            logger.info(f"Audio duration: {duration:.2f} seconds")
        
        with tqdm(total=100, desc="Transcribing audio", unit="%") as pbar:
            result = model.transcribe(process_file, verbose=False, word_timestamps=True)
            pbar.update(100)

        output = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            output.append({"start": start, "end": end, "text": text})

        if with_speaker_ids:
            try:
                from pyannote.audio import Pipeline
                hf_token = get_hf_token()
                logger.info(f"Loading speaker diarization pipeline with token: {hf_token[:6]}...")
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
                if pipeline is None:
                    raise ValueError("Failed to load speaker diarization pipeline.")
                
                pipeline.to(torch.device(device))
                with tqdm(total=100, desc="Diarizing speakers", unit="%") as pbar:
                    diarization = pipeline(process_file)
                    pbar.update(100)
                
                if diarization is None:
                    logger.warning("Diarization returned no results. Speaker IDs will not be added.")
                else:
                    logger.info(f"Diarization result: {diarization}")
                    for segment in output:
                        start, end = segment["start"], segment["end"]
                        segment_obj = Segment(start, end)
                        cropped = diarization.crop(segment_obj)
                        speakers = set()
                        if cropped is not None:
                            for seg, _, speaker in cropped.itertracks(yield_label=True):
                                speakers.add(speaker)
                        segment["speaker"] = "Unknown" if not speakers else "/".join(sorted(speakers))
            except ImportError:
                logger.error("Speaker diarization requires 'pyannote-audio'. Install it and provide a Hugging Face token.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Speaker diarization encountered an issue: {e}")
                logger.info("Continuing with transcription only, without speaker IDs.")

        # Clean up temporary wav file if created
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
            logger.info(f"Cleaned up temporary file: {temp_wav}")

        # Format output as text or JSON
        text_output = "\n".join(
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] "
            f"{'Speaker ' + seg.get('speaker', 'Unknown') + ': ' if 'speaker' in seg else ''}"
            f"{seg['text']}"
            for seg in output
        )
        return text_output, output

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return None, None

def transcribe_with_assemblyai(audio_file):
    """
    Transcribe audio using AssemblyAI.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: Transcription text, or None if failed.
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file {audio_file} not found.")
        return None
    try:
        import assemblyai as aai
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"AssemblyAI error: {transcript.error}")
            return None
        return transcript.text
    except Exception as e:
        logger.error(f"AssemblyAI transcription failed: {e}")
        return None

def save_to_file(text, output_file, json_output=False, json_data=None):
    """
    Save transcription text or JSON to a file.

    Args:
        text (str): Transcription text to save.
        output_file (str): Path to the output file.
        json_output (bool): Whether to save as JSON instead of text.
        json_data (list): List of segment dictionaries for JSON output.
    """
    try:
        if json_output and json_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
        logger.info(f"Transcription saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save transcription to {output_file}: {e}")

def main():
    """Main script logic with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper or AssemblyAI.",
        epilog="Example: python transcribe_audio.py audio.mp3 --method whisper --speakers --model small -o output.txt"
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument(
        "--method",
        choices=["whisper", "assemblyai"],
        default=None,
        help="Transcription method (default: from config, typically 'whisper')"
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="Enable speaker identification (Whisper only, requires pyannote-audio)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper model size (e.g., 'base', 'small', 'medium'); defaults to config 'base'"
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for temporary files; defaults to config or current directory"
    )
    parser.add_argument(
        "-o", "--output",
        nargs="?",
        const=True,
        default=None,
        help="Output transcription to specified file (e.g., output.txt); if no file specified, uses audio file basename with .txt"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output transcription as JSON instead of plain text"
    )
    args = parser.parse_args()

    check_and_install_dependencies()
    load_or_create_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model or CONFIG['whisper'].get('model', 'base')
    temp_dir = args.temp_dir or CONFIG['whisper'].get('temp_dir', str(Path.cwd()))

    method = args.method or CONFIG['DEFAULT'].get('method', 'whisper')

    if method == "whisper":
        model = setup_whisper(device, model_name)
        logger.info(f"Transcribing {args.audio_file} with Whisper...")
        text, json_data = transcribe_with_whisper(
            args.audio_file, model, device, with_speaker_ids=args.speakers, temp_dir=temp_dir
        )
    elif method == "assemblyai":
        setup_assemblyai()
        logger.info(f"Transcribing {args.audio_file} with AssemblyAI...")
        text = transcribe_with_assemblyai(args.audio_file)
        json_data = None  # AssemblyAI doesn't support JSON output in this script
    else:
        logger.error(f"Unknown method '{method}'. Use 'whisper' or 'assemblyai'.")
        sys.exit(1)

    if text:
        transcript_lines = text.splitlines()
        logger.info("\nTranscription Result (first 10 lines):")
        for line in transcript_lines[:10]:
            print(line)  # Keep console output simple for user
        
        if args.output:
            output_file = Path(args.audio_file).stem + (".json" if args.json else ".txt") if args.output is True else args.output
            save_to_file(text, output_file, json_output=args.json, json_data=json_data)
    else:
        logger.error("Transcription failed.")

if __name__ == "__main__":
    main()