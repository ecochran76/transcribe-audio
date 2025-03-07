#!/usr/bin/env python3

import os
import sys
import subprocess
import configparser
import pkg_resources
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment

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


def check_and_install_dependencies():
    """Check for required packages and install them if missing."""
    missing = []
    for pkg, install_name in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            missing.append(install_name)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print("All dependencies are already installed.")


def load_or_create_config():
    """Load existing config or create a new one."""
    if CONFIG_FILE.exists():
        CONFIG.read(CONFIG_FILE)
    else:
        CONFIG['DEFAULT'] = {'method': 'whisper', 'assemblyai_api_key': '', 'hf_token': ''}
        CONFIG['whisper'] = {'model': 'base'}
        save_config()
        print(f"Created new configuration file at {CONFIG_FILE}")


def save_config():
    """Save the current configuration to file."""
    with open(CONFIG_FILE, 'w') as configfile:
        CONFIG.write(configfile)


def setup_whisper():
    """Bootstrap Whisper installation and configuration."""
    try:
        import whisper
        model_name = CONFIG['whisper'].get('model', 'base')
        # Device selection: use CUDA if available, otherwise CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        print(f"Loading Whisper model: {model_name} on {device_name}")
        try:
            model = whisper.load_model(model_name, device=device, weights_only=True)
        except TypeError:
            print("Warning: This version of openai-whisper does not support 'weights_only'. Loading without it.")
            try:
                model = whisper.load_model(model_name, device=device)
            except TypeError:
                model = whisper.load_model(model_name, device=device)
        print(f"Whisper is set up successfully on {device_name}.")
        return model, device
    except ImportError as e:
        print(f"Required module not installed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up Whisper: {e}")
        print("Ensure you have sufficient disk space and compatible hardware.")
        sys.exit(1)


def setup_assemblyai():
    """Guide user to set up AssemblyAI and store API key."""
    api_key = CONFIG['DEFAULT'].get('assemblyai_api_key', '')
    if not api_key:
        print("AssemblyAI requires an API key.")
        print("1. Go to https://www.assemblyai.com/ and sign up for a free account.")
        print("2. Get your API key from the dashboard.")
        api_key = input("Enter your AssemblyAI API key: ").strip()
        try:
            import assemblyai as aai
            aai.settings.api_key = api_key
            print("Testing API key...")
            test_transcriber = aai.Transcriber()
            test_transcriber.transcribe("https://example.com/test.wav")
            CONFIG['DEFAULT']['assemblyai_api_key'] = api_key
            save_config()
            print("API key validated and saved.")
        except Exception as e:
            print(f"Failed to validate API key: {e}")
            print("Please try again with a valid key.")
            sys.exit(1)
    else:
        import assemblyai as aai
        aai.settings.api_key = api_key
    return True


def get_hf_token():
    """Get Hugging Face token from config or user input."""
    hf_token = CONFIG['DEFAULT'].get('hf_token', '')
    if not hf_token:
        print("Pyannote-audio requires a Hugging Face token for speaker diarization.")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'read' access.")
        hf_token = input("Enter your Hugging Face token: ").strip()
        CONFIG['DEFAULT']['hf_token'] = hf_token
        save_config()
        print("Hugging Face token saved.")
    return hf_token


def get_audio_duration(audio_file):
    """Get audio duration in seconds using soundfile, fall back to pydub if needed."""
    try:
        with sf.SoundFile(audio_file) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Could not open {audio_file} with soundfile: {e}")
        print("Attempting conversion to .wav with pydub...")
        try:
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # pydub returns milliseconds
            return duration
        except Exception as e:
            print(f"Failed to convert {audio_file} with pydub: {e}")
            print("Ensure FFmpeg is installed and in PATH.")
            return None


def transcribe_with_whisper(audio_file, model, device, with_speaker_ids=False):
    """Transcribe audio using Whisper with progress, timestamps, and optional speaker IDs."""
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None

    try:
        # Try opening with soundfile; convert to wav if it fails.
        temp_wav = None
        try:
            with sf.SoundFile(audio_file):
                pass  # File can be processed as-is.
            process_file = str(audio_file)
        except Exception as e:
            print(f"Soundfile failed to open {audio_file}: {e}")
            print("Converting to .wav with pydub for processing...")
            audio = AudioSegment.from_file(audio_file)
            temp_wav = Path(audio_file).with_suffix('.wav')
            audio.export(temp_wav, format="wav")
            process_file = str(temp_wav)
            print(f"Converted to {process_file}")

        # Use the processed file for duration calculation.
        duration = get_audio_duration(process_file)
        if duration:
            print(f"Audio duration: {duration:.2f} seconds")

        with tqdm(total=100, desc="Transcribing", unit="%") as pbar:
            result = model.transcribe(process_file, verbose=False, word_timestamps=True)
            pbar.update(100)

        output = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            output.append(f"[{start:.2f}s - {end:.2f}s] {text}")

        if with_speaker_ids:
            try:
                from pyannote.audio import Pipeline
                from pyannote.core import Segment
                hf_token = get_hf_token()
                print(f"Loading speaker diarization pipeline with token: {hf_token[:6]}...")
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
                if pipeline is None:
                    raise ValueError("Failed to load speaker diarization pipeline.")

                # Convert device string to a torch.device instance.
                device_obj = torch.device(device)
                pipeline.to(device_obj)

                with tqdm(total=100, desc="Diarizing speakers", unit="%") as pbar:
                    diarization = pipeline(process_file)
                    pbar.update(100)

                if diarization is None:
                    print("Warning: Diarization returned no results. Speaker IDs will not be added.")
                else:
                    # Update each segment with speaker information.
                    for i, segment in enumerate(result["segments"]):
                        start, end = segment["start"], segment["end"]
                        segment_obj = Segment(start, end)
                        cropped = diarization.crop(segment_obj)
                        speakers = set()
                        if cropped is not None:
                            for seg, _, speaker in cropped.itertracks(yield_label=True):
                                speakers.add(speaker)
                        speaker_label = "Unknown" if not speakers else "/".join(sorted(speakers))
                        output[i] = f"[{start:.2f}s - {end:.2f}s] Speaker {speaker_label}: {segment['text'].strip()}"
            except ImportError:
                print("Speaker diarization requires 'pyannote-audio'. Install it and provide a Hugging Face token.")
                sys.exit(1)
            except Exception as e:
                print(f"Speaker diarization encountered an issue: {e}")
                print("Continuing with transcription only, without speaker IDs.")

        # Clean up temporary wav file if created.
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
            print(f"Cleaned up temporary file: {temp_wav}")

        return "\n".join(output)
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return None


def transcribe_with_assemblyai(audio_file):
    """Transcribe audio using AssemblyAI."""
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None
    try:
        import assemblyai as aai
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)
        if transcript.status == aai.TranscriptStatus.error:
            print(f"AssemblyAI error: {transcript.error}")
            return None
        return transcript.text
    except Exception as e:
        print(f"AssemblyAI transcription failed: {e}")
        return None


def save_to_file(text, output_file):
    """Save transcription text to a file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Transcription saved to {output_file}")
    except Exception as e:
        print(f"Failed to save transcription to {output_file}: {e}")


def main():
    """Main script logic."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Whisper or AssemblyAI.",
        epilog="Example: python transcribe_audio.py audio.mp3 --method whisper --speakers -o output.txt"
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
        "-o", "--output",
        nargs="?",
        const=True,
        default=None,
        help="Output transcription to specified file (e.g., output.txt); if no file specified, uses audio file basename with .txt"
    )
    args = parser.parse_args()

    check_and_install_dependencies()
    load_or_create_config()

    method = args.method if args.method else CONFIG['DEFAULT'].get('method', 'whisper')

    if method == "whisper":
        model, device = setup_whisper()
        print(f"Transcribing {args.audio_file} with Whisper...")
        text = transcribe_with_whisper(args.audio_file, model, device, with_speaker_ids=args.speakers)
    elif method == "assemblyai":
        setup_assemblyai()
        print(f"Transcribing {args.audio_file} with AssemblyAI...")
        text = transcribe_with_assemblyai(args.audio_file)
    else:
        print(f"Unknown method '{method}'. Use 'whisper' or 'assemblyai'.")
        sys.exit(1)

    if text:
        transcript_lines = text.splitlines()
        print("\nTranscription Result (first 10 lines):")
        print("\n".join(transcript_lines[:10]))

        if args.output:
            output_file = Path(args.audio_file).stem + ".txt" if args.output is True else args.output
            save_to_file(text, output_file)
    else:
        print("Transcription failed.")


if __name__ == "__main__":
    main()
