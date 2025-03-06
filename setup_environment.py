#!/usr/bin/env python3

import subprocess
import sys

def check_python_version():
    """Ensure Python version is 3.7 or higher."""
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version {sys.version} is compatible.")

def install_requirements():
    """Install dependencies from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def main():
    print("Setting up Python environment for transcription script...")
    check_python_version()
    install_requirements()
    print("Environment setup complete. Run 'python transcribe_audio.py --help' for usage.")

if __name__ == "__main__":
    main()