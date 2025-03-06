#!/usr/bin/env python3

import subprocess
import sys
import platform
import pkg_resources

def check_python_version():
    """Ensure Python version is 3.7 or higher."""
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} is compatible.")

def check_gpu_capability():
    """Check for NVIDIA GPU and CUDA availability, return status."""
    print("Checking GPU capability...")
    
    # Step 1: Check if NVIDIA GPU is present via nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            print("NVIDIA GPU detected:")
            print(result.stdout.splitlines()[0])  # Show GPU info
        else:
            print("No NVIDIA GPU detected or nvidia-smi not found.")
            return False, None
    except FileNotFoundError:
        print("nvidia-smi command not found. NVIDIA drivers may not be installed.")
        return False, None

    # Step 2: Check if PyTorch is installed and CUDA-capable
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
            print(f"Current CUDA version: {torch.version.cuda}")
            return True, torch.version.cuda
        else:
            print("CUDA is not available in PyTorch, but GPU detected.")
            return True, None  # GPU present, but PyTorch lacks CUDA
    except ImportError:
        print("PyTorch not yet installed (will be installed later).")
        return True, None  # GPU present, no PyTorch yet

def print_gpu_install_guidance():
    """Guide user to install NVIDIA drivers."""
    print("\nTo enable GPU acceleration:")
    print("1. Ensure you have an NVIDIA GPU.")
    print("2. Install NVIDIA drivers:")
    print("   - Visit https://www.nvidia.com/Download/index.aspx")
    print("   - Select your GPU model and OS (Windows), download, and install.")
    print("3. Verify with 'nvidia-smi' in a Command Prompt after installation.")
    print("   - Expected output: A table listing your GPU(s).")

def print_cuda_install_guidance(cuda_version=None):
    """Guide user to install CUDA Toolkit and configure PyTorch."""
    print("\nTo enable CUDA for GPU acceleration:")
    print("1. Install CUDA Toolkit:")
    print("   - Visit https://developer.nvidia.com/cuda-downloads")
    print("   - Select your OS (Windows), architecture (x86_64), and version (e.g., 11.8 or 12.1).")
    print("   - Recommended: CUDA 11.8 or 12.1 for latest PyTorch compatibility.")
    print("   - Download and install (e.g., cuda_11.8.0_522.06_windows.exe).")
    print("2. (Optional) Install cuDNN for better performance:")
    print("   - Visit https://developer.nvidia.com/cudnn (requires free NVIDIA account)")
    print("   - Download a version matching your CUDA (e.g., cuDNN 8.9 for CUDA 11.8)")
    print("   - Extract and copy to CUDA install directory (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8)")
    if cuda_version:
        print(f"3. Reinstall PyTorch with CUDA support (current PyTorch is CPU-only):")
    else:
        print("3. Install PyTorch with CUDA support after CUDA is installed:")
    print("   - For CUDA 11.8: pip install torch --extra-index-url https://download.pytorch.org/whl/cu118")
    print("   - For CUDA 12.1: pip install torch --extra-index-url https://download.pytorch.org/whl/cu121")
    print("4. Verify CUDA with Python:")
    print("   - Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("   - Expected output: True")

def install_requirements():
    """Install dependencies from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def reinstall_pytorch_with_cuda():
    """Offer to reinstall PyTorch with CUDA support."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nPyTorch is installed but lacks CUDA support.")
            response = input("Would you like to reinstall PyTorch with CUDA support? (y/n): ").strip().lower()
            if response == 'y':
                cuda_version = input("Enter CUDA version (11.8 or 12.1 recommended): ").strip()
                if cuda_version not in ["11.8", "12.1"]:
                    print(f"Unsupported CUDA version '{cuda_version}'. Defaulting to 11.8.")
                    cuda_version = "11.8"
                print(f"Uninstalling existing PyTorch...")
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "-y"])
                print(f"Installing PyTorch with CUDA {cuda_version} support...")
                url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--extra-index-url", url])
                print("PyTorch reinstalled with CUDA support.")
                import torch  # Re-import to verify
                if torch.cuda.is_available():
                    print(f"Success! CUDA is now available: {torch.cuda.get_device_name(0)}")
                else:
                    print("CUDA still not available. Ensure CUDA Toolkit matches the version selected.")
            else:
                print("Keeping CPU-only PyTorch.")
        else:
            print("PyTorch already has CUDA support.")
    except ImportError:
        print("PyTorch not installed yet; will install via requirements.txt.")

def main():
    print("Setting up Python environment for transcription script...")
    check_python_version()
    
    # Check GPU and CUDA status
    has_gpu, cuda_version = check_gpu_capability()
    
    if has_gpu:
        if cuda_version:
            print("GPU environment looks good!")
        else:
            print_cuda_install_guidance(cuda_version)
            reinstall_pytorch_with_cuda()
    else:
        print_gpu_install_guidance()
        print_cuda_install_guidance()
        print("Continuing without GPU support. You can add CUDA later for faster transcription.")

    # Install or verify requirements
    install_requirements()
    print("Environment setup complete. Run 'python transcribe_audio.py --help' for usage.")

if __name__ == "__main__":
    main()