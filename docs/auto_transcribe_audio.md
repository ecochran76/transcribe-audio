# auto_transcribe_audio.py

## Overview

`auto_transcribe_audio.py` is a Python script that automates the transcription and summarization of audio files by monitoring a specified folder. It integrates `transcribe_audio.py` and `summarize_transcript.py` to process new audio files as they appear, optionally moving the resulting files (audio, transcript, summary, and context) to a designated output folder. The script supports configurable settings via an INI file, including the option to rename files based on context-derived timestamps and titles.

This script is ideal for workflows requiring continuous processing of audio recordings, such as meeting archives, podcast production, or voicemail management.

## Usage

Basic folder watching:
```bash
python auto_transcribe_audio.py /path/to/audio_folder
```

With output folder and file moving:
```bash
python auto_transcribe_audio.py /path/to/audio_folder --output-folder /path/to/output --move
```

Process existing files at startup:
```bash
python auto_transcribe_audio.py /path/to/audio_folder --output-folder /path/to/output --move --process-existing
```

With regex filtering:
```bash
python auto_transcribe_audio.py /path/to/audio_folder --regex "meeting.*\.mp3" --output-folder /path/to/output --move
```

## Arguments

| Argument            | Type    | Description                                                                                   | Default          | Required |
|---------------------|---------|-----------------------------------------------------------------------------------------------|------------------|----------|
| `input_folder`      | str     | Folder to watch for audio files.                                                              | `.` (current dir)| No       |
| `--regex`           | str     | Regex pattern to filter audio filenames (e.g., `.*\.mp3`).                                    | None             | No       |
| `--output-folder`   | str     | Directory to save transcripts, summaries, and context files.                                  | None             | No       |
| `--move`            | flag    | Move audio, transcript, summary, and context files to `--output-folder` after processing.     | False            | No       |
| `--process-existing`| flag    | Process existing audio files in `input_folder` at startup.                                    | False            | No       |
| `--config`          | str     | Path to configuration INI file.                                                               | `auto_transcribe_config.ini` | No |

- **Note**: `--move` requires `--output-folder` to be specified.

## Configuration

### `auto_transcribe_config.ini`
Created automatically if missing:
```ini
[Transcription]
method = whisper
speakers = False
model = base
temp_dir = 
transcript_output_format = txt

[Summarization]
model = gpt-4o-mini
api_key_file = api_keys.json
summary_output_format = docx
speaker_hints = 
rename_from_context = False
```
- **[Transcription]**:
  - `method`: `whisper` or `assemblyai`.
  - `speakers`: Enable speaker diarization (`True`/`False`).
  - `model`: Whisper model size (e.g., `base`, `small`).
  - `temp_dir`: Temporary file directory (optional).
  - `transcript_output_format`: `txt` or `json`.
- **[Summarization]**:
  - `model`: LLM model (e.g., `gpt-4o-mini`, `grok-v2`).
  - `api_key_file`: Path to API keys JSON.
  - `summary_output_format`: `json`, `markdown`, `html`, `docx`, `pdf`.
  - `speaker_hints`: Optional speaker identity hints.
  - `rename_from_context`: Rename files using timestamp and title (`True`/`False`).

Edit this file to customize transcription and summarization settings.

## Output Format

The script generates:
1. **Transcript**: Via `transcribe_audio.py` (text or JSON).
2. **Summary**: Via `summarize_transcript.py` (configurable format).
3. **Context**: JSON file with metadata and summary.

### With `--rename-from-context` (in `output_folder`)
- Audio: `20250303-143000-Collaborative-Meeting.mp3`
- Transcript: `20250303-143000-Collaborative-Meeting.txt`
- Summary: `20250303-143000-Collaborative-Meeting.docx`
- Context: `20250303-143000-Collaborative-Meeting-context.json`

### Without `--rename-from-context`
- Audio: `original_audio.mp3`
- Transcript: `original_audio.txt`
- Summary: `original_audio Summary.docx`
- Context: `original_audio-context.json`

## Dependencies

- **Required**: All dependencies from `transcribe_audio.py` and `summarize_transcript.py`, plus:
  - `watchdog` (for folder monitoring)
- Install via:
  ```bash
  pip install watchdog openai-whisper assemblyai soundfile tqdm torch pydub mutagen pyannote-audio openai colorlog markdown xhtml2pdf
  ```
- **External**:
  - FFmpeg (for audio conversion): Install via `choco install ffmpeg` (Windows), `sudo apt-get install ffmpeg` (Linux), or `brew install ffmpeg` (macOS).
  - Pandoc (for DOCX output): Install via `choco install pandoc` (Windows), `sudo apt-get install pandoc` (Linux), or `brew install pandoc` (macOS).

## Workflow

1. **Start Monitoring**:
   - Launches a watcher on `input_folder`.
   - Processes existing files if `--process-existing` is set.
2. **Detect New Audio**:
   - Matches files against `--regex` (if provided) and supported extensions (`.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`).
3. **Transcription**:
   - Calls `transcribe_audio.py` with configured settings.
4. **Summarization**:
   - Calls `summarize_transcript.py` with configured settings.
5. **Move Files** (if `--move`):
   - Relocates audio, transcript, summary, and context files to `--output-folder`, renaming them if `rename_from_context` is enabled.

The script runs until interrupted (Ctrl+C).

## Key Features

- **Folder Watching**: Continuously monitors `input_folder` for new audio files using `watchdog`.
- **Automation**: Chains transcription and summarization without manual intervention.
- **File Moving**: Optionally moves all output files to a specified folder, with consistent naming when `rename_from_context` is enabled.
- **Configurability**: Customizes transcription and summarization via `auto_transcribe_config.ini`.

## Troubleshooting

- **No Files Processed**: Ensure `input_folder` contains supported audio files (`.mp3`, etc.) and matches `--regex` if set.
- **Move Fails**: Verify `--output-folder` exists and is writable. Check logs for errors.
- **Renaming Issues**: Confirm `rename_from_context = True` in config and the transcript has a "Date and Time:" line. Check `summarize_transcript.py` logs for title generation.
- **Script Stops**: Look for exceptions in logs (e.g., API key errors, missing dependencies). Restart with `--process-existing` to catch up.
- **Dependencies Missing**: Run `pip install -r requirements.txt` with all listed packages.

## Example Workflow

1. Configure `auto_transcribe_config.ini`:
   ```ini
   [Summarization]
   rename_from_context = True
   summary_output_format = markdown
   ```

2. Start the watcher:
   ```bash
   python auto_transcribe_audio.py ./audio --output-folder ./output --move --process-existing
   ```

3. Add an audio file (e.g., `meeting.mp3`):
   - Output in `./output`:
     - `20250303-143000-Meeting-Discussion.mp3`
     - `20250303-143000-Meeting-Discussion.txt`
     - `20250303-143000-Meeting-Discussion.md`
     - `20250303-143000-Meeting-Discussion-context.json`

## Notes

- **Performance**: Depends on transcription and summarization settings; larger models increase processing time.
- **Interruption**: Use Ctrl+C to stop gracefully; restart resumes monitoring.
- **Logging**: Console output tracks progress; check logs for detailed debugging.

## See Also

- [transcribe_audio.py](transcribe_audio.md)
- [summarize_transcript.py](summarize_transcript.md)
