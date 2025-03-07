#!/usr/bin/env python3

import os
import sys
import subprocess
import venv
import shutil
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / "transcribe_env"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# Define required dependencies
REQUIRED_PACKAGES = [
    "openai-whisper>=20231117",
    "assemblyai>=0.33.0",
    "soundfile>=0.12.1",
    "tqdm>=4.67.1",
    "torch>=2.0.0",
    "pyannote.audio>=3.1.1",  # For speaker diarization
    "pydub>=0.25.1",
]

def check_ffmpeg():
    """Check if FFmpeg is installed and available in PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg is already installed and available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg is not found. Please install it manually:")
        print("1. Download from https://ffmpeg.org/download.html or https://www.gyan.dev/ffmpeg/builds/")
        print("2. Extract and add the 'bin' folder (e.g., C:\\ffmpeg\\bin) to your system PATH.")
        print("3. Restart your terminal or system to apply PATH changes.")
        sys.exit(1)

def create_virtualenv():
    """Create a virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}...")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)
    else:
        print(f"Virtual environment already exists at {VENV_DIR}.")

def get_pip_executable():
    """Get the path to the pip executable in the virtual environment."""
    if sys.platform == "win32":
        pip_exe = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip_exe = VENV_DIR / "bin" / "pip"
    return pip_exe

def install_dependencies():
    """Install required Python packages into the virtual environment."""
    pip_exe = get_pip_executable()
    if not pip_exe.exists():
        print(f"pip not found in virtual environment at {pip_exe}. Re-creating environment...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        create_virtualenv()
        pip_exe = get_pip_executable()

    # Write requirements.txt
    with open(REQUIREMENTS_FILE, "w") as f:
        f.write("\n".join(REQUIREMENTS_PACKAGES) + "\n")
    
    print("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([str(pip_exe), "install", "-r", str(REQUIREMENTS_FILE)])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def main():
    """Set up the environment for the transcription script."""
    print("Setting up environment for audio transcription...")
    
    # Check FFmpeg first
    check_ffmpeg()
    
    # Create virtual environment
    create_virtualenv()
    
    # Install dependencies
    install_dependencies()
    
    # Instructions for activation
    if sys.platform == "win32":
        activate_cmd = str(VENV_DIR / "Scripts" / "activate")
    else:
        activate_cmd = f"source {VENV_DIR / 'bin' / 'activate'}"
    
    print("\nEnvironment setup complete!")
    print("To activate the virtual environment, run:")
    print(f"  {activate_cmd}")
    print("Then, run the transcription script with:")
    print("  python transcribe_audio.py --help")

if __name__ == "__main__":
    main()
    