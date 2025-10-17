# AssemblyAI Transcription Toolkit

This repository now focuses exclusively on a lightweight AssemblyAI-powered transcription workflow. Supply an audio file, and the helper script uploads it to AssemblyAI, polls until the transcript is ready, and emits a diarized DOCX transcript (with optional plain text).

The former local Whisper/LLM automation has been archived under the `legacy-whisper-pipeline` Git tag for future reference.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ASSEMBLYAI_API_KEY=your_api_key  # or store it in api_keys.json
python assembly_transcribe.py path/to/audio.wav --text-output
# (Optional) Include calendar metadata:
# python assembly_transcribe.py meeting.mp3 --use-calendar --text-output
```

- The script supports most common audio/video formats accepted by AssemblyAI.
- Outputs default to the same directory as the source file (override with `--output-dir`).

## Configuration & API Keys

The CLI discovers the AssemblyAI key in this priority order:

1. `--api-key` command-line argument.
2. `ASSEMBLYAI_API_KEY` environment variable.
3. `assemblyai_api_key` entry inside `api_keys.json` (default path is the repo root).

Keep `api_keys.json` out of version control—copy `api_keys.json.sample` or use environment variables for local development.

## Usage

```bash
python assembly_transcribe.py meeting.mp3 \
  --model universal \
  --poll-interval 2 \
  --output-dir transcripts \
  --text-output
```

Key options:

- `--model`: AssemblyAI speech model (defaults to `universal`).
- `--speaker-labels` / `--no-speaker-labels`: toggle diarization (enabled by default).
- `--text-output`: also writes a `.txt` transcript alongside the DOCX export.
- `--poll-interval`: adjust polling cadence to balance speed and API quota usage.
- `--use-calendar`: match the file timestamp to a Google Calendar event, rename artifacts, and embed event metadata.

## Calendar-Aware Renaming (Optional)

The CLI can rename audio and transcript outputs to align with a matching Google Calendar event and prepend the transcript with event details and attendees. The pattern used is `YYYY-mm-dd HH-MM <event name> <original base>` (colons are replaced with dashes for cross-platform compatibility).

1. In the [Google Cloud Console](https://console.cloud.google.com/), create an OAuth client ID (Desktop or Web app) and download the secrets file (commonly `credentials.json` or `client_secrets.json`). Place it alongside `assembly_transcribe.py`; the script searches this directory even when invoked from elsewhere.
2. Run the CLI with `--use-calendar`. The first invocation opens a browser window for Google authorization and stores a token in `token.json` (also next to the script) for reuse.
3. Optional flags:
   - `--calendar-id` to query a non-primary calendar.
   - `--calendar-credentials` / `--calendar-token` to override file locations.
   - `--calendar-client-secrets` to point at a specific secrets file if you keep multiple credentials around.
   - `--calendar-window` to expand or shrink the search window (hours either side of the file timestamp).

Example:

```bash
python assembly_transcribe.py ~/Downloads/board_meeting.mp4 \
  --use-calendar \
  --text-output
```

If no event is found within the search window, the script continues without renaming or event metadata.

## Development Notes

- Python 3.9+ is recommended (the CLI relies on postponed annotations).
- Dependencies are kept lightweight (`requests`, `python-docx`, and the Google Calendar client libraries); install with `pip install -r requirements.txt`.
- Contributions should target AssemblyAI-specific enhancements such as richer formatting, metadata decoration, or integration hooks for downstream tooling.

## Legacy Pipeline

If you need the original Whisper + local summarizer automation, checkout the `legacy-whisper-pipeline` tag:

```bash
git fetch origin --tags
git checkout legacy-whisper-pipeline
```

That tag preserves the full diarization + LLM summarization workflow before this repository refocus.
