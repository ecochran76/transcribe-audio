#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import socket
from pathlib import Path
import venv

BASE_DIR = Path(__file__).parent.resolve()
HOSTNAME = socket.gethostname().lower().replace(" ", "_")
VENV_DIR = BASE_DIR / f"transcribe_env_{HOSTNAME}"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✔ FFmpeg is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✘ FFmpeg not found. Install it and add to PATH.")
        sys.exit(1)

def create_virtualenv():
    if VENV_DIR.exists():
        print(f"Virtual environment already exists at {VENV_DIR}")
    else:
        print(f"Creating virtual environment at {VENV_DIR}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(VENV_DIR)

def get_python_executable():
    return VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python.exe"

def get_pip_executable():
    return VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "pip"

def verify_pip(pip_path):
    try:
        subprocess.run([str(pip_path), "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def reinstall_venv():
    print("Removing corrupted virtual environment...")
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    create_virtualenv()

def install_dependencies():
    pip = get_pip_executable()
    if not verify_pip(pip):
        print("✘ pip appears broken or missing in venv. Recreating environment.")
        reinstall_venv()
        pip = get_pip_executable()
        if not verify_pip(pip):
            print("✘ pip is still broken after rebuild. Aborting.")
            sys.exit(1)

    print(f"Installing packages from {REQUIREMENTS_FILE}...")
    try:
        subprocess.check_call([str(pip), "install", "-r", str(REQUIREMENTS_FILE)])
        print("✔ Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"✘ pip install failed: {e}")
        sys.exit(1)

def main():
    print(f"Setting up environment (venv: {VENV_DIR.name})")
    check_ffmpeg()
    create_virtualenv()
    install_dependencies()

    activate = VENV_DIR / ("Scripts/activate" if os.name == "nt" else "bin/activate")
    print("\nEnvironment setup complete.")
    print("To activate the environment, run:")
    print(f"  {activate}")

if __name__ == "__main__":
    main()
