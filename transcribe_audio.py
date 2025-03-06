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

# Define required packages with their exact install names
REQUIRED_PACKAGES = {
    "openai-whisper": "openai-whisper",
    "assemblyai": "assemblyai",
    "soundfile": "soundfile",
    "tqdm": "tqdm",
    "torch-directml": "torch-directml",  # Added for AMD GPU support
    # "pyannote.audio": "pyannote-audio"  # Uncomment for speaker diarization
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
        CONFIG['DEFAULT'] = {'method': 'whisper', 'assemblyai_api_key': ''}
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
        
        # Device selection logic
        if torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
            device_name = torch.cuda.get_device_name(0)
        else:
            try:
                import torch_directml
                dml_device = torch_directml.device()  # AMD GPU via DirectML
                device = dml_device  # Pass the DirectML device object directly
                device_name = "DirectML (AMD GPU)"  # Simplified name for logging
            except ImportError:
                device = "cpu"
                device_name = "CPU"
        
        print(f"Loading Whisper model: {model_name} on {device_name}")
        # Load model with weights_only=True to address FutureWarning
        model = whisper.load_model(model_name, device=device, weights_only=True)
        print(f"Whisper is set up successfully on {device_name}.")
        return model
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
            test_transcriber.transcribe("https://example.com/test.wav")  # Dummy test
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

def get_audio_duration(audio_file):
    """Get audio duration in seconds using soundfile."""
    try:
        with sf.SoundFile(audio_file) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Could not determine audio duration: {e}")
        return None

def transcribe_with_whisper(audio_file, model, with_speaker_ids=False):
    """Transcribe audio using Whisper with progress, timestamps, and optional speaker IDs."""
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None
    
    try:
        duration = get_audio_duration(audio_file)
        if duration:
            print(f"Audio duration: {duration:.2f} seconds")
        
        with tqdm(total=100, desc="Transcribing", unit="%") as pbar:
            result = model.transcribe(audio_file, verbose=False, word_timestamps=True)
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
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="YOUR_HF_TOKEN")
                pipeline.to(model.device)
                diarization = pipeline(audio_file)
                
                for i, segment in enumerate(result["segments"]):
                    start, end = segment["start"], segment["end"]
                    speakers = set()
                    for turn, _, speaker in diarization.crop((start, end)):
                        speakers.add(speaker)
                    speaker_label = "Unknown" if not speakers else "/".join(sorted(speakers))
                    output[i] = f"[{start:.2f}s - {end:.2f}s] Speaker {speaker_label}: {segment['text'].strip()}"
            except ImportError:
                print("Speaker diarization requires 'pyannote-audio'. Install it and provide a Hugging Face token.")
            except Exception as e:
                print(f"Speaker diarization failed: {e}")

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
        help="Output transcription to specified file (e.g., output.txt)"
    )
    args = parser.parse_args()

    check_and_install_dependencies()
    load_or_create_config()

    method = args.method if args.method else CONFIG['DEFAULT'].get('method', 'whisper')

    if method == "whisper":
        model = setup_whisper()
        print(f"Transcribing {args.audio_file} with Whisper...")
        text = transcribe_with_whisper(args.audio_file, model, with_speaker_ids=args.speakers)
    elif method == "assemblyai":
        setup_assemblyai()
        print(f"Transcribing {args.audio_file} with AssemblyAI...")
        text = transcribe_with_assemblyai(args.audio_file)
    else:
        print(f"Unknown method '{method}'. Use 'whisper' or 'assemblyai'.")
        sys.exit(1)

    if text:
        print("\nTranscription Result:")
        print(text)
        if args.output:
            save_to_file(text, args.output)
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    main()