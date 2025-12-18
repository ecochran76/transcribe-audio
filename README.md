# AssemblyAI Transcription Toolkit

This repository contains a lightweight CLI for sending audio or video files to AssemblyAI, waiting for the transcript to finish, and exporting the results as DOCX, plain text, or subtitles. Optional Google Calendar integration can rename files and embed event metadata, and ffmpeg support enables subtitle embedding.

The former Whisper-based automation lives in the `legacy-whisper-pipeline` tag.

## Prerequisites

- **Python 3.9+** – required for postponed annotations and the Google client libraries.
- **ffmpeg (optional)** – required for `--embed-subtitles` and for burning subtitles.
- **AssemblyAI account** – required to generate an API key.

## Install Dependencies

```bash
git clone https://github.com/ecochran76/transcribe-audio.git
cd transcribe-audio
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `requests`, `python-docx`, and the Google Calendar client libraries—everything the CLI needs at runtime.

## Get an AssemblyAI API Key

1. Sign in or create an account at [AssemblyAI](https://www.assemblyai.com/).
2. Navigate to the dashboard’s API section and copy your **live** API key (the CLI does not use temporary tokens).
3. Keep the key private; it grants access to your AssemblyAI quota and billing.

## Configure Secrets

The CLI checks for credentials in this order:

1. `--api-key` command-line flag.
   - Use `--api-key-prompt` (hidden input) or `--api-key-stdin` to avoid storing keys on disk.
2. `ASSEMBLYAI_API_KEY` environment variable.
3. `assemblyai_api_key` entry inside `api_keys.json` (searches the current working directory first, then alongside `assembly_transcribe.py`).

To store the key in a file, copy the sample template and fill it in:

```bash
cp api_keys.json.sample api_keys.json
# edit api_keys.json and add your AssemblyAI (and optional OpenAI/Grok) keys
```

Optional language configuration:

- `assemblyai_language_code` defaults to `en_us` (English).
- Set it to `pt` for Portuguese transcription.

`api_keys.json` is already ignored by git. Use whichever option—flag, env var, or file—works best for your workflow.

## Run the CLI

Once dependencies and credentials are in place, invoke the script with one or more audio/video files. Run `python assembly_transcribe.py --help` to see every option.

### Common commands

```bash
# 1. Basic DOCX transcript in-place
python assembly_transcribe.py meeting.m4a

# 2. DOCX + plain-text output in a custom folder
python assembly_transcribe.py meeting.m4a --text-output --output-dir transcripts

# 3. Standalone subtitles (SRT)
python assembly_transcribe.py webinar.mp4 --srt-output

# 4. Embed subtitles back into the media (requires ffmpeg)
python assembly_transcribe.py webinar.mp4 --embed-subtitles

# 5. Add Google Calendar metadata and diarization
python assembly_transcribe.py board_call.wav --use-calendar --speaker-labels

# 6. Batch processing and glob expansion
python assembly_transcribe.py "~/Downloads/*.mp3" --model universal-2
python assembly_transcribe.py "C:\\Calls\\*.m4a" --text-output
```

Key options:

- `--model`: Choose the AssemblyAI model (`universal` by default).
- `--language`: Set the transcription language (e.g. `pt` for Portuguese, or `auto` for detection).
- `--text-output`: Write a `.txt` transcript alongside the DOCX.
- `--srt-output`: Produce subtitles instead of DOCX (use `--docx-output` to emit both).
- `--docx-output`: Also emit DOCX when `--srt-output` is set.
- `--embed-subtitles`: Embed subtitles as a subtitle track in the source media (creates `<original> subtitled.<ext>`).
- `--translate-to`: Translate transcript + subtitles to a target language (e.g. `en`). Requires an OpenAI key (`OPENAI_API_KEY` or `openai_api_key` in `api_keys.json`).
- `--poll-interval`: Control how often the script checks AssemblyAI for job status.
- `--speaker-labels` / `--no-speaker-labels`: Toggle diarization.
- `--output-dir`: Override where outputs are saved.
- `--use-calendar`: Look up a nearby Google Calendar event and rename/add metadata.

## Subtitle Outputs

`--srt-output` generates sentence-length cues. When only one speaker is detected, speaker labels are omitted automatically. `--embed-subtitles` muxes subtitles into MP4/M4V/MOV (text subtitles) or MKV (SRT subtitles) as a selectable track. You can enable both flags to keep the `.srt` file while also embedding it.

### Embed an existing SRT (soft subtitles)

If you already have an `.srt` (for example, a manually repaired translation), you can mux it into a copy of an MP4 without re-encoding video/audio:

```bash
ffmpeg -y \
  -i Petrobras.mp4 \
  -sub_charenc UTF-8 -i "outputs_pt/Petrobras Transcript ENG.srt" \
  -map 0:v -map "0:a?" -map 1:0 \
  -c:v copy -c:a copy -c:s mov_text \
  -metadata:s:s:0 language=eng -metadata:s:s:0 title="English" \
  -movflags +faststart \
  "Petrobras subtitled ENG.mp4"
```

### Burn subtitles (hard subtitles)

Burning subtitles re-encodes the video (the text becomes part of the pixels):

```bash
ffmpeg -y \
  -i Petrobras.mp4 \
  -vf "subtitles=outputs_pt/Petrobras\\ Transcript\\ ENG.srt:charenc=UTF-8" \
  -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p \
  -c:a copy \
  -movflags +faststart \
  "Petrobras burned ENG.mp4"
```

## Calendar-Aware Renaming (Optional)

1. Create a Google OAuth client (Desktop or Web app) in the [Google Cloud Console](https://console.cloud.google.com/) and download the secrets file (e.g., `credentials.json`).
2. Place `credentials.json` or `client_secrets.json` next to `assembly_transcribe.py`.
3. Invoke the CLI with `--use-calendar`; the first run launches a browser window to authorize access and writes `token.json` for reuse.
4. Additional flags:
   - `--calendar-id` for non-primary calendars.
   - `--calendar-credentials`, `--calendar-token`, and `--calendar-client-secrets` to override file locations.
   - `--calendar-window` (hours on either side of the file timestamp) to tighten or widen the search.

If an event is found, outputs are renamed to `YYYY-mm-dd HH-MM <event name> <original base>` (colons replaced with dashes) and the transcript is annotated with event metadata. If no event matches, transcription continues without renaming.

## Development Notes

- Keep sample secrets in `api_keys.json.sample`; real keys belong in the ignored `api_keys.json`.
- Manual smoke tests are encouraged: run the CLI on a short clip after changes to confirm DOCX/TXT/SRT outputs.
- For automated tests, create `tests/` powered by `pytest` and mock AssemblyAI calls with `responses` or similar.

## Legacy Pipeline

Need the old Whisper + local summarizer workflow? Fetch the archived tag:

```bash
git fetch origin --tags
git checkout legacy-whisper-pipeline
```

That tag preserves the diarization + LLM summarization tooling that previously lived in this repository.
