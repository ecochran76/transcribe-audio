# Audio Transcription with Whisper and Pyannote

This project provides a Python script (`transcribe_audio.py`) to transcribe audio files using OpenAI's Whisper model and optionally perform speaker diarization with `pyannote.audio`. It supports multiple audio formats via `pydub` and offers both text and JSON output options.

## Features
- **Transcription**: Uses Whisper for high-quality speech-to-text conversion.
- **Speaker Diarization**: Identifies speakers in audio using `pyannote.audio` (optional).
- **Format Support**: Handles various audio formats (e.g., `.m4a`, `.mp3`) by converting to `.wav` if needed.
- **Configurable**: Supports command-line arguments and a config file for customization.
- **Output Options**: Outputs transcriptions as plain text or JSON with timestamps and speaker labels.

## Requirements
- **Python**: 3.8 or higher
- **FFmpeg**: Required for audio format conversion with `pydub`.
- **Hugging Face Account**: For `pyannote.audio` model access.

## Setup

### 1. Clone or Download the Repository
Clone this repository or download the files (`transcribe_audio.py`, `setup_environment.py`, etc.) to your local machine.

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Run the Setup Script
The `setup_environment.py` script creates a virtual environment and installs dependencies.

```bash
python setup_environment.py
```

#### What It Does:
- Creates a virtual environment (`transcribe_env`).
- Checks for FFmpeg and prompts for installation if missing.
- Installs Python packages: `openai-whisper`, `pyannote.audio`, `pydub`, `soundfile`, `tqdm`, `torch`, `assemblyai`.

#### FFmpeg Installation (if prompted):
- **Windows**:
  1. Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (e.g., `ffmpeg-release-essentials.zip`).
  2. Extract to `C:\ffmpeg`.
  3. Add `C:\ffmpeg\bin` to your system PATH (Control Panel > System > Advanced > Environment Variables).
  4. Restart your terminal.
- **Linux/macOS**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  # macOS (via Homebrew)
  brew install ffmpeg
  ```

### 3. Activate the Virtual Environment
- **Windows**:
  ```bash
  transcribe_env\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  source transcribe_env/bin/activate
  ```

### 4. Configure Hugging Face Access (for Speaker Diarization)
`pyannote.audio` requires a Hugging Face token and model permissions:
1. **Create an Account**: Sign up at [huggingface.co](https://huggingface.co/).
2. **Generate a Token**:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Click "New token," name it (e.g., `pyannote-access`), select "read" scope, and copy the token (starts with `hf_`).
3. **Accept Model Conditions**:
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the conditions.
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the conditions.
   - You may need to agree to share contact info for pyannote’s userbase study.
The script will prompt you to enter this token on first run and save it to `transcription_config.ini`.

## Usage

Run `transcribe_audio.py` with the following options:

```bash
python transcribe_audio.py <audio_file> [options]
```

### Options
- `--method`: Transcription method (`whisper` or `assemblyai`; default: config or `whisper`).
- `--speakers`: Enable speaker diarization (Whisper only).
- `--model`: Whisper model size (e.g., `base`, `small`, `medium`; default: config or `base`).
- `--temp-dir`: Directory for temporary files (default: config or current directory).
- `-o/--output`: Output file (e.g., `output.txt`); if no file specified, uses `<audio_file>.txt` or `.json`.
- `--json`: Output as JSON instead of plain text.
- `--help`: Show help message.

### Examples
1. **Basic Transcription (Text Output)**:
   ```bash
   python transcribe_audio.py "path/to/audio.m4a" -o output.txt
   ```
2. **Transcription with Speaker IDs**:
   ```bash
   python transcribe_audio.py "path/to/audio.m4a" --speakers -o output.txt
   ```
3. **JSON Output with Speaker IDs**:
   ```bash
   python transcribe_audio.py "path/to/audio.m4a" --speakers --json -o output.json
   ```
4. **Custom Model and Temp Directory**:
   ```bash
   python transcribe_audio.py "path/to/audio.m4a" --model small --temp-dir "C:\Temp" -o output.txt
   ```

### Output
- **Text**: Lines with `[start-end] Speaker <label>: <text>` or `[start-end] <text>` (no speakers).
- **JSON**: List of dictionaries with `start`, `end`, `text`, and optional `speaker` fields.

## Configuration
The script uses `transcription_config.ini` for persistent settings:
- **DEFAULT**: `method`, `assemblyai_api_key`, `hf_token`.
- **whisper**: `model`, `temp_dir`.

Edit this file manually or let the script configure it on first run.

## Troubleshooting
- **FFmpeg Not Found**: Ensure FFmpeg is installed and in your PATH (`ffmpeg -version` should work).
- **Hugging Face Token Errors**: Verify token validity and model permissions at Hugging Face.
- **Diarization Fails**: Check audio quality (e.g., silence, noise) or token permissions.
- **Warnings**: Update `openai-whisper` (`pip install --upgrade openai-whisper`) to resolve `weights_only` warnings.

## Dependencies
- `openai-whisper`: Speech-to-text transcription.
- `pyannote.audio`: Speaker diarization (version 3.1+).
- `pydub`: Audio format conversion (requires FFmpeg).
- `soundfile`: Audio file reading.
- `tqdm`: Progress bars.
- `torch`: PyTorch for model execution.
- `assemblyai`: Alternative transcription service.

## Contributing
Feel free to submit issues or pull requests to improve the script. Suggestions for alternative diarization methods (e.g., SpeechBrain) or additional features are welcome!

## License
This project is unlicensed (public domain). Use it freely, but ensure compliance with the licenses of dependencies (e.g., MIT for `pyannote.audio`).
