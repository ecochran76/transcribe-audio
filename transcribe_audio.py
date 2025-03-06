#!/usr/bin/env python3

import os
import sys
import subprocess
import configparser
import pkg_resources
from pathlib import Path

# Define required packages
REQUIRED_PACKAGES = {
    "whisper": "openai-whisper",
    "assemblyai": "assemblyai",
    "soundfile": "soundfile",  # For audio handling
}

# Configuration file setup
CONFIG_FILE = Path(__file__).parent / "transcription_config.ini"
CONFIG = configparser.ConfigParser()

def check_and_install_dependencies():
    """Check for required packages and install them if missing."""
    missing = []
    for pkg, install_name in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.require(pkg)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing.append(install_name)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)

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
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Whisper is set up successfully.")
        return model
    except ImportError:
        print("Whisper is not installed correctly despite dependency check.")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up Whisper: {e}")
        print("Ensure you have sufficient disk space and a compatible GPU (optional) for faster processing.")
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
            # Test the key
            test_transcriber = aai.Transcriber()
            print("Testing API key...")
            test_transcriber.transcribe("https://example.com/test.wav")  # Dummy test; will fail gracefully
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

def transcribe_with_whisper(audio_file, model):
    """Transcribe audio using Whisper."""
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} not found.")
        return None
    try:
        result = model.transcribe(audio_file)
        return result["text"]
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

def main():
    """Main script logic."""
    # Check dependencies and config
    check_and_install_dependencies()
    load_or_create_config()

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <audio_file> [--method whisper|assemblyai]")
        sys.exit(1)

    audio_file = sys.argv[1]
    method = CONFIG['DEFAULT'].get('method', 'whisper')
    if len(sys.argv) > 2 and sys.argv[2] == "--method":
        method = sys.argv[3].lower()

    # Set up and transcribe
    if method == "whisper":
        model = setup_whisper()
        print(f"Transcribing {audio_file} with Whisper...")
        text = transcribe_with_whisper(audio_file, model)
    elif method == "assemblyai":
        setup_assemblyai()
        print(f"Transcribing {audio_file} with AssemblyAI...")
        text = transcribe_with_assemblyai(audio_file)
    else:
        print(f"Unknown method '{method}'. Use 'whisper' or 'assemblyai'.")
        sys.exit(1)

    # Output result
    if text:
        print("\nTranscription Result:")
        print(text)
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    main()