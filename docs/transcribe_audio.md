# transcribe_audio.py

## Overview

`transcribe_audio.py` is a Python script that converts audio files into text transcripts using either OpenAI’s Whisper or AssemblyAI transcription models. It supports optional speaker diarization (identifying who spoke when) using `pyannote-audio`, and can output transcripts in plain text or JSON format. The script prepends each transcript with a "Date and Time:" line derived from the audio file’s metadata or filesystem modified time, enhancing compatibility with downstream tools like `summarize_transcript.py`.

This script is ideal for transcribing meetings, interviews, voicemails, or any audio where accurate text conversion and speaker identification are needed.

## Usage

Basic usage with Whisper:
```bash
python transcribe_audio.py audio.mp3 --method whisper --model small -o output.txt
```

With speaker identification:
```bash
python transcribe_audio.py audio.mp3 --method whisper --speakers --model base -o output.txt
```

Using AssemblyAI:
```bash
python transcribe_audio.py audio.mp3 --method assemblyai -o output.txt
```

JSON output:
```bash
python transcribe_audio.py audio.mp3 --method whisper --json -o output.json
```

## Arguments

| Argument      | Type    | Description                                                                                   | Default          | Required |
|---------------|---------|-----------------------------------------------------------------------------------------------|------------------|----------|
| `audio_file`  | str     | Path to the audio file to transcribe.                                                         | None             | Yes      |
| `--method`    | str     | Transcription method: `whisper` or `assemblyai`.                                              | Config-based     | No       |
| `--speakers`  | flag    | Enable speaker identification (Whisper only, requires `pyannote-audio` and Hugging Face token).| False            | No       |
| `--model`     | str     | Whisper model size (e.g., `base`, `small`, `medium`, `large`).                                | Config-based (`base`) | No  |
| `--temp-dir`  | str     | Directory for temporary files (e.g., WAV conversions). Auto-generated if unspecified.          | None             | No       |
| `-o/--output` | str     | Output file path. If unspecified, defaults to `<audio_file>.txt` or `<audio_file>.json`.      | None             | No       |
| `--json`      | flag    | Output transcription as JSON instead of plain text.                                           | False            | No       |

## Configuration

### `transcription_config.ini`
Created automatically if missing:
```ini
[DEFAULT]
method = whisper
assemblyai_api_key = 
hf_token = 

[whisper]
model = base
```
- **`method`**: Default transcription method (`whisper` or `assemblyai`).
- **`assemblyai_api_key`**: Required for AssemblyAI; prompts for input if missing.
- **`hf_token`**: Hugging Face token for `pyannote-audio` speaker diarization; prompts if missing.
- **`model`**: Default Whisper model size.

The script saves updates (e.g., API keys) to this file after user input.

## Output Format

### Plain Text (Default)
When `--json` is not specified:
```
Date and Time: 2025-03-03 14:30:00

[0.00s - 5.23s] Speaker SPEAKER_00: Hello, this is a test.
[5.23s - 10.45s] Speaker SPEAKER_01: Great to hear you!
```
- "Date and Time:" reflects audio metadata (e.g., ID3 `date` tag) or filesystem modified time.
- Speaker labels appear only if `--speakers` is enabled.

### JSON (with `--json`)
```
[
  {
    "start": 0.00,
    "end": 5.23,
    "speaker": "SPEAKER_00",
    "text": "Hello, this is a test."
  },
  {
    "start": 5.23,
    "end": 10.45,
    "speaker": "SPEAKER_01",
    "text": "Great to hear you!"
  }
]
```
- Timestamp not included in JSON; use text output for compatibility with `--rename-from-context` in `summarize_transcript.py`.

## Dependencies

- **Required**: 
  - `openai-whisper` (Whisper transcription)
  - `assemblyai` (AssemblyAI transcription)
  - `soundfile` (audio file handling)
  - `tqdm` (progress bars)
  - `torch` (Whisper backend)
  - `pydub` (audio conversion)
  - `mutagen` (audio metadata extraction)
- **Optional**:
  - `pyannote-audio` (speaker diarization, requires Hugging Face token)
- Install via:
  ```bash
  pip install openai-whisper assemblyai soundfile tqdm torch pydub mutagen pyannote-audio
  ```
- **External**: FFmpeg (for audio conversion with `pydub`):
  - Windows: `choco install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

## Workflow

1. **Input**: Provide an audio file (e.g., `.mp3`, `.wav`, `.flac`, `.m4a`, `.aac`, `.ogg`).
2. **Processing**:
   - Extracts timestamp from audio metadata or filesystem.
   - Converts audio to WAV if needed (using `pydub`).
   - Transcribes using Whisper or AssemblyAI.
   - Adds speaker labels if `--speakers` is set (Whisper only).
3. **Output**:
   - Saves transcript with "Date and Time:" prefix in text or JSON format.

## Key Features

- **Multi-Model Support**: Choose between Whisper (local) or AssemblyAI (cloud-based) transcription.
- **Speaker Diarization**: Identifies speakers with `pyannote-audio` when `--speakers` is enabled.
- **Timestamp Embedding**: Prepends transcripts with a "Date and Time:" line for downstream renaming.
- **Progress Tracking**: Displays transcription and diarization progress with `tqdm`.
- **Flexible Formats**: Outputs as text or JSON, compatible with `summarize_transcript.py`.

## Troubleshooting

- **FFmpeg Not Found**: Install FFmpeg and ensure it’s in your PATH. Check with `ffmpeg -version`.
- **No Speaker Labels**: Verify `--speakers` is set, `pyannote-audio` is installed, and a valid Hugging Face token is provided (prompted if missing).
- **Transcription Fails**: Check audio file format compatibility and ensure sufficient disk space for temporary files.
- **Invalid Timestamp**: Ensure audio metadata is readable by `mutagen`; otherwise, it falls back to modified time (logged).
- **AssemblyAI Errors**: Provide a valid API key in `transcription_config.ini` or via prompt.

## Example Workflow

1. Transcribe an audio file with speakers:
   ```bash
   python transcribe_audio.py meeting.mp3 --method whisper --speakers --model small -o meeting.txt
   ```
   Output: `meeting.txt`:
   ```
   Date and Time: 2025-03-03 14:30:00

   [0.00s - 5.23s] Speaker SPEAKER_00: Welcome to the meeting.
   [5.23s - 10.45s] Speaker SPEAKER_01: Thanks for joining.
   ```

2. Use with `summarize_transcript.py`:
   ```bash
   python summarize_transcript.py meeting.txt --model gpt-4o-mini --rename-from-context
   ```
   Output: `20250303-143000-Meeting-Summary.md`

## Notes

- **Performance**: Larger Whisper models (e.g., `large`) improve accuracy but require more memory and time.
- **Temporary Files**: Managed in `--temp-dir` or an auto-generated directory; cleaned up after processing.
- **Logging**: Detailed logs are written to `transcription.log` and the console for debugging.

## See Also

- [summarize_transcript.py](summarize_transcript.md)
- [auto_transcribe_audio.py](auto_transcribe_audio.md)
