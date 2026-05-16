# AssemblyAI Transcription Toolkit

This repository contains two lightweight transcription CLIs and a directory watcher:

- `assembly_transcribe.py` uploads audio or video to AssemblyAI, waits for the transcript, and exports DOCX, plain text, or subtitles.
- `faster_whisper_transcribe.py` runs the same export/calendar workflow locally with `faster-whisper`, which is a good fit for an NVIDIA GPU such as an RTX 5080.
- `watch_transcriptions.py` scans one or more directories, waits for files to stop changing, and then invokes either transcription backend automatically.

Every successful transcription now also writes a machine-readable `*.transcript.json` sidecar next to the human-facing transcript outputs. The sidecar records the source media path, final media path after calendar rename, backend, recording window, event metadata when available, emitted output paths, transcript window, transcript text, and structured utterances. Calendar metadata includes `matching_calendars` so downstream readouts can see which accessible calendars overlapped the recording. The watcher stores these artifact paths in `.openclaw/watch_transcriptions_state.json` for downstream summarization, routing, and deposition.

Optional Google Calendar integration can rename files and embed event metadata. The shared calendar lookup uses an explicit provider order, defaulting to local `gog`, then `gws`, then the bundled Google OAuth flow as fallback. ffmpeg support enables subtitle embedding.

The former Whisper-based automation lives in the `legacy-whisper-pipeline` tag.

## Prerequisites

- **Python 3.9+** – required for postponed annotations and the Google client libraries.
- **ffmpeg (optional)** – required for `--embed-subtitles` and for burning subtitles.
- **AssemblyAI account** – required to generate an API key.
- **CUDA libraries (local GPU path only)** – required when running `faster_whisper_transcribe.py` on NVIDIA hardware.

## Install Dependencies

```bash
git clone https://github.com/ecochran76/transcribe-audio.git
cd transcribe-audio
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `requests`, `python-docx`, the Google Calendar client libraries, `faster-whisper`, and `pyannote.audio` for local speaker diarization.

If you plan to use `faster_whisper_transcribe.py` on Linux with NVIDIA CUDA userspace libraries managed by Python packages, install the CUDA 12/cuDNN 9 runtime packages before running the script:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
export LD_LIBRARY_PATH="$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')"
```

## Get an AssemblyAI API Key

1. Sign in or create an account at [AssemblyAI](https://www.assemblyai.com/).
2. Navigate to the dashboard’s API section and copy your **live** API key (the CLI does not use temporary tokens).
3. Keep the key private; it grants access to your AssemblyAI quota and billing.

## Configure Secrets

`assembly_transcribe.py` checks for AssemblyAI credentials in this order:

1. `--api-key` command-line flag.
   - Use `--api-key-prompt` (hidden input) or `--api-key-stdin` to avoid storing keys on disk.
2. `assemblyai_api_key` entry inside `api_keys.json` (searches alongside `assembly_transcribe.py` first, then the current working directory).
3. `ASSEMBLYAI_API_KEY` environment variable.

Tip: add `--print-key-sources` to see which source was used (without printing the keys).

To store the key in a file, copy the sample template and fill it in:

```bash
cp api_keys.json.sample api_keys.json
# edit api_keys.json and add your AssemblyAI (and optional OpenAI/Grok) keys
```

Optional language configuration:

- `assemblyai_language_code` defaults to `en_us` (English).
- Set it to `pt` for Portuguese transcription.

`faster_whisper_transcribe.py` reuses the same `api_keys.json` for:

- `assemblyai_language_code` as the default transcription language selection.
- `huggingface_token` for speaker diarization.
- `openai_api_key` when `--translate-to` is used.
- `openai_base_url` when using an OpenAI-compatible API for translation or readout generation.

Local diarization also requires:

- a Hugging Face token (`--hf-token`, `HF_TOKEN`, `HUGGING_FACE_TOKEN`, cached `~/.cache/huggingface/token`, or `huggingface_token` in `api_keys.json`)
- acceptance of the model agreements for `pyannote/speaker-diarization-3.1`
- on newer `pyannote.audio` releases, acceptance of `pyannote/speaker-diarization-community-1` may also be required

`api_keys.json` is already ignored by git. Use whichever option—flag, env var, or file—works best for your workflow.

## Run the CLI

Once dependencies and credentials are in place, invoke either script with one or more audio/video files. Run `python assembly_transcribe.py --help` or `python faster_whisper_transcribe.py --help` to see every option.

### Common commands

```bash
# 1. Basic DOCX transcript in-place
python assembly_transcribe.py meeting.m4a

# 2. Basic local GPU transcript in-place (skip diarization)
python faster_whisper_transcribe.py meeting.m4a --no-speaker-labels

# 3. DOCX + plain-text output in a custom folder
python assembly_transcribe.py meeting.m4a --text-output --output-dir transcripts

# 4. DOCX + plain-text output with faster-whisper
python faster_whisper_transcribe.py meeting.m4a --text-output --output-dir transcripts --no-speaker-labels

# 5. Standalone subtitles (SRT)
python assembly_transcribe.py webinar.mp4 --srt-output

# 6. Embed subtitles back into the media (requires ffmpeg)
python assembly_transcribe.py webinar.mp4 --embed-subtitles

# 7. Add Google Calendar metadata and diarization
python assembly_transcribe.py board_call.wav --use-calendar --speaker-labels

# 8. Local batch processing and diarization
python faster_whisper_transcribe.py board_call.wav --hf-token "$HUGGING_FACE_TOKEN" --speaker-labels

# 9. Local batch processing and glob expansion
python faster_whisper_transcribe.py "~/Downloads/*.mp3" --model large-v3 --compute-type float16 --no-speaker-labels

# 10. AssemblyAI batch processing and glob expansion
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
- `--calendar-providers`: Comma-separated provider order. Supported values are `gog`, `gws`, and `google-api`; default is `gog,gws,google-api`.
- `--calendar-gog-account` / `--calendar-gog-client`: Pass explicit tenant/client selectors through to `gog`.
- `--calendar-gws-config-dir`: Set `GOOGLE_WORKSPACE_CLI_CONFIG_DIR` for `gws`.

Output contract:

- DOCX/TXT/SRT files remain the human-facing transcript outputs.
- `*.transcript.json` is the durable automation handoff for future readout, routing, and deposition stages.
- `*.readout.json` and `*.readout.md` are the AI-generated summary/readout outputs produced by `summarize_transcript.py`.
- Backend CLIs print `TRANSCRIPT_ARTIFACT_JSON=<path>` for each sidecar so service wrappers can record artifact paths without guessing filenames.

### Clean historical calendar filenames

If older calendar-mode runs created duplicated date/title prefixes, use
`cleanup_transcript_filenames.py` from the virtual environment. It defaults to a
dry-run and refuses to apply while the watcher is active unless it is allowed to
stop and restart the service.

```bash
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --apply --manage-service --refresh-store
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --export-review ~/.local/state/transcribe-audio/filename-cleanup-review.json --include-diff-summary
.venv/bin/python transcript_filename_conflict_review.py ~/.local/state/transcribe-audio/filename-cleanup-review.json
.venv/bin/python transcript_filename_conflict_review.py --investigate-review ~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review.json
.venv/bin/python transcript_filename_conflict_review.py --apply-review ~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review.json --audit-output ~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review-audit.json
.venv/bin/python transcript_filename_conflict_review.py --apply-review ~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review.json --apply --approval-token APPLY_FILENAME_CONFLICT_REVIEW --manage-service --refresh-store
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --resolve-reviewed-conflicts
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --apply --manage-service --refresh-store --resolve-reviewed-conflicts
.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --apply --manage-service --refresh-store --resolve-identical-conflicts
```

The cleanup derives canonical names from each transcript sidecar's calendar
event, renames only non-conflicting files, rewrites sidecar path fields, updates
watcher state, and can refresh `~/.transcripts` rows for changed sidecars.
Use `--export-review` when dry-run reports skipped conflicts; the review file
lists the event metadata, proposed operations, existing targets, and suggested
manual action for each ambiguous artifact. Add `--include-diff-summary` to
include privacy-conscious line counts, similarity ratios, and conflict
classifications without embedding transcript excerpts. Use
`transcript_filename_conflict_review.py` to turn that export into a local
operator decision template and Markdown report with explicit choices:
`preserve_both`, `quarantine_old`, `keep_target`, or `needs_investigation`. Use
`--investigate-review` to generate a private local report with bounded diff
snippets for pending items; do not commit those generated reports because they
can contain transcript text. Use
`--apply-review` as a dry-run over an edited decision template; live mutation
requires `--apply --approval-token APPLY_FILENAME_CONFLICT_REVIEW`. Only
`quarantine_old` moves files; `preserve_both` and `keep_target` are recorded
no-op decisions, while `pending` and `needs_investigation` are skipped. Use
`--resolve-reviewed-conflicts` first as a dry-run to report conflicts classified
as `metadata_or_format_only_candidate`; in apply mode it quarantines the old
conflict files, moves non-conflicting outputs, and rewrites sidecar pointers.
Use
`--resolve-identical-conflicts` only after review; it compares transcript JSON
text, TXT text, and DOCX paragraph text, then moves redundant old conflict files
to `~/.local/state/transcribe-audio/filename-cleanup-quarantine/` instead of
deleting them.

Local faster-whisper options:

- `--device`: `auto`, `cuda`, or `cpu` (`auto` prefers CUDA).
- `--compute-type`: CTranslate2 precision/quantization (`float16` is a good RTX default).
- `--batch-size` / `--no-batch`: Control batched inference.
- `--vad-filter` / `--no-vad-filter`: Toggle silence filtering before transcription.
- `--hf-token`: Provide the Hugging Face token used to load the pyannote diarization pipeline.
- `--min-speakers` / `--max-speakers`: Hint the expected speaker count to the diarization model.

Note: `faster_whisper_transcribe.py` now performs local speaker diarization via `pyannote.audio` when `--speaker-labels` is enabled. If you want the fastest local transcript without diarization overhead, use `--no-speaker-labels`.

## Automatic Watcher Service

`watch_transcriptions.py` is the missing glue for the mobile-recording workflow: it scans a directory, waits for files to become stable, remembers what it has already processed, and then launches either `assembly_transcribe.py` or `faster_whisper_transcribe.py` with the CLI flags you choose.

Typical flow:

- SyncThing drops a half-written `My Recording*.m4a` into `~/Downloads`
- the watcher notices the file but does nothing while size/mtime are still moving
- once the file has been unchanged for the configured settle window, the watcher transcribes it
- `--use-calendar` still works, so the final outputs can be renamed around the matching meeting

### Configure the watcher

Start from the sample config:

```bash
cp watch_transcriptions.json.sample watch_transcriptions.json
```

Example job:

```json
{
  "name": "downloads-assembly",
  "watch_dir": "~/Downloads",
  "glob": "My Recording*.m4a",
  "backend": "assembly",
  "settle_seconds": 120,
  "scan_interval": 30,
  "calendar": {
    "enabled": true,
    "providers": ["gog", "gws", "google-api"],
    "calendar_id": "primary",
    "gog": {
      "account": "me@example.com",
      "client": "work"
    },
    "gws": {
      "config_dir": "~/.config/gws-work"
    }
  },
  "readout": {
    "enabled": true,
    "provider": "openai-compatible",
    "model": "gpt-4o-mini"
  },
  "store": {
    "enabled": true,
    "store_dir": "~/.transcripts",
    "embedding_provider": "ollama",
    "embedding_model": "ollama/nomic-embed-text"
  },
  "cli_args": ["--text-output"]
}
```

Important fields:

- `watch_dir`: directory to scan
- `glob`: filename pattern matched against the basename, for example `My Recording*.m4a`
- `backend`: `assembly` or `faster_whisper`
- `settle_seconds`: how long the file must remain unchanged before processing
- `scan_interval`: how often the watcher scans
- `failure_retry_seconds`: how long to wait before retrying a failed file
- `cli_args`: flags forwarded to the chosen transcription script
- `readout`: optional post-processing config; when enabled, the watcher runs `summarize_transcript.py` for each emitted transcript artifact
- `store`: optional transcript-store config; when enabled, the watcher ingests emitted transcript artifacts and generated readouts into `~/.transcripts`
- `enabled`: optional boolean to turn jobs on/off without deleting them

## Intelligence Readouts

Generate a structured readout from an existing transcript artifact:

```bash
python summarize_transcript.py "meeting Transcript.transcript.json" \
  --provider openai-compatible \
  --model gpt-4o-mini
```

The OpenAI-compatible provider resolves credentials from `--openai-api-key`, `OPENAI_API_KEY`, or `openai_api_key` in `api_keys.json`. It resolves the API base URL from `--base-url`, `OPENAI_BASE_URL`, or `openai_base_url` in `api_keys.json`.

Readouts include:

- summary, participants, topics, key decisions, action items, unresolved questions
- matter candidates and memory candidates for later routing/review stages
- risks and next steps

When the transcript artifact has `event.matching_calendars`, `summarize_transcript.py` passes it in a dedicated `calendar_context` block. The readout prompt tells the provider to treat calendar names and overlapping event summaries as evidence for meeting type, likely matter candidates, and memory candidates without treating them as proof.

The `openai-compatible` and `codex-exec` providers are implemented. The `codex-exec` provider runs `codex exec` with read-only sandboxing and no approvals, captures the final message, and validates it as the same readout JSON shape. Provider seams remain for `auracall` and `openclaw`. Watcher readout failures are logged and do not mark transcription itself as failed.

## Matter Routing Dry Run

Create an auditable route decision from an existing transcript artifact and readout:

```bash
python route_transcript.py "meeting Transcript.transcript.json" "meeting Transcript.readout.json"
```

This writes a `*.route.json` decision containing route candidates, confidence, evidence, rejected alternatives, fallback behavior, and a `provenance_pack`. Current provenance comes from calendar metadata already stored in the transcript artifact, including `event.matching_calendars`. Low-confidence decisions write a local review queue item under `~/.local/state/transcribe-audio/review-queue/` unless `--no-review-queue` is passed.

Add live read-only Google Workspace provenance with `gws`:

```bash
python route_transcript.py "meeting Transcript.transcript.json" "meeting Transcript.readout.json" \
  --gws-provenance
```

The `gws` adapter can refresh Calendar event details and search Drive file metadata into the same `provenance_pack`. Generated Drive queries default to precise filename intersections to avoid noisy full-text matches; pass `--gws-drive-query` when you want a broader or hand-tuned Drive search. Use `--gws-config-dir` for a non-default `gws` tenant/profile.

Add read-only Graphiti/OpenClaw advisory provenance:

```bash
python route_transcript.py "meeting Transcript.transcript.json" "meeting Transcript.readout.json" \
  --graphiti-provenance \
  --graphiti-group transcribe_audio_main \
  --graphiti-group openclaw_ec_main
```

The Graphiti adapter queries compact calendar/readout terms, not raw transcript text. Graphiti facts and episodes are stored as provenance; Graphiti nodes may add low-confidence advisory route candidates. These candidates are evidence for review, not deposition authority, and can be disabled with `--no-graphiti-candidates`.

Add read-only Odollo/Odoo provenance from configured contacts and log notes:

```bash
python route_transcript.py "meeting Transcript.transcript.json" "meeting Transcript.readout.json" \
  --odollo-provenance
```

The Odollo adapter queries compact meeting, attendee, and readout terms against the configured production profiles `soylei-prod` and `saber-prod` by default. It adds `odollo_contact` and `odollo_log_note` sources to the route provenance pack, using Odoo record identifiers and subjects/dates without storing raw log-note bodies. Use repeated `--odollo-profile` flags to narrow or change tenants.

Non-calendar provenance is quality-filtered before it can support a selected route or contextual reread. The filter scores source-specific identity fields against compact meeting/readout/participant terms: Drive file names/snippets, Odollo contact identity, Odollo note subjects, and Graphiti labels/previews. It ignores retrieval-control metadata such as Graphiti queries and Odollo matched-term lists. Weak sources are written to `provenance_pack.excluded_sources` with `quality_status`, `quality_score`, and `quality_reason`. Route warnings record exclusion counts and are copied into contextual reread metadata and deposition previews. Use `--provenance-quality-threshold` to tune the required match count or `--no-provenance-quality-filter` for diagnostic runs only.

Generate an upgraded contextual reread from the selected route:

```bash
python contextual_reread.py \
  "meeting Transcript.transcript.json" \
  "meeting Transcript.readout.json" \
  "meeting Transcript.route.json" \
  --provider codex-exec
```

This writes `*.contextual.readout.json` and `*.contextual.readout.md`. The reread prompt includes the prior readout, selected route candidate, and a compact supporting-context packet from cited route provenance sources. The output JSON records the exact supporting source list under `contextualization.supporting_context_sources`.

## Deposition And Memory-Harvest Preview

Create a no-write preview plan from a contextual readout:

```bash
python deposition_preview.py \
  "meeting Transcript.contextual.readout.json" \
  --route "meeting Transcript.route.json" \
  --local-root ~/Transcripts/Reviewed \
  --graphiti-group transcribe_audio_main
```

The preview writes `*.deposit-preview.json`. It can describe proposed local, Google Drive, and Odoo deposition actions, but every action is `status: preview` and records `writes_enabled: false`. Memory harvest candidates are taken only from structured `memory_candidates` fields in the readout; raw transcript text is not harvested.

Apply only the local filesystem actions from a reviewed preview:

```bash
python deposition_apply.py "meeting Transcript.deposit-preview.json"
```

This writes `*.deposit-apply.json`. Only `local_filesystem` actions are applied; Google Drive, Odoo, and Graphiti-related actions are skipped. Existing destination files with the same hash are skipped idempotently, while conflicting filenames are versioned.

Preview reviewed Graphiti memory-harvest candidates without writing:

```bash
python memory_harvest_apply.py "meeting Transcript.deposit-preview.json"
```

This writes `*.memory-harvest-apply.json` with planned `graphiti-runtime benchmark-write` commands. It refuses previews that require review or carry warnings unless the operator passes `--allow-review-required` or `--allow-warnings` after review. No temporary memory body files are written in dry-run mode.

Create an operator review template:

```bash
python memory_harvest_apply.py "meeting Transcript.deposit-preview.json" --init-review
```

This writes `*.memory-harvest-review.json` with one entry per structured memory candidate. Set each candidate `decision` to `approved`, `rejected`, or `pending`; rejected and pending candidates are carried into the apply audit as non-written statuses.

Apply approved memory candidates to Graphiti only with an explicit approval token:

```bash
python memory_harvest_apply.py "meeting Transcript.deposit-preview.json" \
  --review-file "meeting Transcript.memory-harvest-review.json" \
  --apply \
  --approval-token APPROVE_GRAPHITI_MEMORY_HARVEST
```

The apply path writes only approved structured `memory_candidates` from the readout-derived preview. It does not harvest raw transcript text. Use repeated `--candidate-id` flags to narrow the reviewed subset further. By default, each approved candidate gets a Graphiti discovery duplicate preflight before write; exact same-candidate replays are skipped as `duplicate_skipped`, and failed duplicate checks stop the candidate as `duplicate_check_failed`. The Graphiti runtime still performs its own queue/write/readback verification; the apply result records compact review, duplicate-check, and write status fields without storing raw Graphiti stdout/stderr.

## User-Scoped Transcript Store

Initialize and search the local store:

```bash
python transcript_store.py init
python transcript_store.py ingest "meeting Transcript.transcript.json" "meeting Transcript.readout.json"
python transcript_store.py search "Tempo Chemical concrete sealer"
python transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context
python transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json
python transcript_store.py context "<document-id-from-search>" --chunk-index 5
```

The store lives under `~/.transcripts`: `transcripts.sqlite3` holds metadata, text, FTS indexes, document vectors, and chunk vectors, while copied JSON artifacts live under `~/.transcripts/artifacts/`. Search combines SQLite FTS5 lexical results with local Ollama embeddings by default (`ollama/nomic-embed-text`), chunks long documents before embedding, and returns a `best_chunk` snippet/score for precise segment hits. Transcript chunks also carry character offsets, utterance time ranges, speaker lists, and utterance counts when the source artifact has structured utterances. Use `search --context` to open the top hit directly, or `context` with a search result `id` and `best_chunk.chunk_index` to print nearby transcript chunks plus a media seek hint when the artifact records `working_media_path` or `source_media_path`. `context --format compact-json` and `search --context --context-format compact-json` emit pure single-line JSON for `jq` and other machine consumers. `openai-compatible` embeddings are also supported with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`; `debug-hash` is reserved for tests and offline debugging, not production semantic search.

Use the compact packet recipe helper when you want the next downstream commands without manually copying paths:

```bash
python transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json \
  | python scripts/context_packet_recipe.py --store --with-provenance
```

The helper is non-mutating. It reads the compact context packet, extracts `context.document.source_path`, and prints shell commands for `summarize_transcript.py`, `route_transcript.py`, and `contextual_reread.py`. Pass `--readout` and `--route` when those artifact paths already exist.

To preview or explicitly execute the same steps, use `context_packet_apply.py`. Preview is the default; nothing runs unless `--apply` is present:

```bash
python transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json \
  | python scripts/context_packet_apply.py --store --with-provenance

python transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json \
  | python scripts/context_packet_apply.py --apply --store --with-provenance --provider-timeout 600 --timeout 900
```

When executing, the helper captures `READOUT_JSON=...` and `ROUTE_DECISION_JSON=...` from stdout and passes those generated artifact paths into the later steps. Passing `--readout` or `--route` skips that generation step and uses the existing artifact.

Child commands default to the repo-local `.venv/bin/python` when present so they inherit installed runtime dependencies. Use `--python` to override the child interpreter. `--timeout` guards each child process, while `--provider-timeout` is passed through to `summarize_transcript.py` and `contextual_reread.py` for slower LLM providers such as `codex-exec`.

Completed `--apply` runs write a manifest under `~/.local/state/transcribe-audio/context-packet-runs/` by default. The manifest records the transcript path, selected store document/chunk, generated readout/route/contextual-readout paths, and sanitized step metadata without raw stdout/stderr. Use `--manifest-dir` to override the runtime location or `--no-manifest` to suppress the manifest.

Inspect recent manifests without browsing the runtime directory:

```bash
python scripts/context_packet_apply.py --list-manifests
python scripts/context_packet_apply.py --list-manifests --format json --limit 20
```

Generated readouts can be ingested automatically with `--store`, and watcher jobs can set `"store": {"enabled": true}` to ingest emitted transcript sidecars plus generated readouts after each successful transcription:

```bash
python summarize_transcript.py "meeting Transcript.transcript.json" --provider codex-exec --store
python contextual_reread.py "meeting Transcript.transcript.json" "meeting Transcript.readout.json" "meeting Transcript.route.json" --provider codex-exec --store
```

Set `TRANSCRIPTS_STORE=true` for transcription runs to ingest emitted transcript sidecars into `~/.transcripts`. Override the runtime home with `--store-dir` on readout commands or `TRANSCRIPTS_STORE_DIR` for transcription.

Backfill existing artifact files deterministically:

```bash
# Review what would be selected and whether each file would insert, update, or skip.
python transcript_store.py backfill ~/Downloads --modified-within-days 14 --dry-run

# Apply the same backfill. Already-current artifacts are skipped without re-embedding.
python transcript_store.py backfill ~/Downloads --modified-within-days 14
```

Useful filters include `--kind transcript`, `--kind readout`, repeated `--pattern`, `--recursive`, `--limit`, `--exclude`, and `--force` when you intentionally want to re-ingest current artifacts. Default excludes skip copied store internals such as `*/store/artifacts/*` and `*/transcripts-store-*/*`; add `--exclude '*/pytest-of-*/*'` for broad `/tmp` scans.

Import older transcript outputs that predate `*.transcript.json` sidecars:

```bash
# Dry-run by default: discovers legacy TXT/DOCX transcript outputs.
python legacy_transcript_import.py ~/HistoricalTranscripts --recursive --media-root ~/HistoricalRecordings

# Apply: synthesize sidecars under ~/.transcripts/legacy-artifacts and ingest them.
python legacy_transcript_import.py ~/HistoricalTranscripts --recursive --media-root ~/HistoricalRecordings --apply
```

For mounted folders where Python directory walks are slow, generate media paths with `find` and pass a newline-delimited index:

```bash
find ~/HistoricalRecordings -type f \( -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.wav' \) -print > /tmp/media-index.txt
python legacy_transcript_import.py ~/HistoricalTranscripts --recursive --media-index-file /tmp/media-index.txt --apply
```

The legacy importer supports `*Transcript.txt` and `*Transcript.docx` by default. It extracts transcript text, attempts to match nearby source recordings by basename, writes synthesized private sidecars only under `~/.transcripts/legacy-artifacts/`, marks them with `legacy_import.needs_enrichment=true`, and then sends them through the normal store ingestion path. Use `--embedding-provider ollama` for production semantic indexing, or `--embedding-provider debug-hash` only for tests/offline validation.

Duplicate protection is enabled by default. The importer skips candidates with the same source transcript hash or the same normalized title as an existing stored transcript, and also skips duplicate hashes/titles within the same batch. Use `--no-dedupe` only for diagnostic runs where you intentionally want duplicate rows. Use `--no-media-match` when importing from large mounted Shared Drives and you want transcript indexing first, with recording/blob matching deferred to a later targeted pass.

List legacy rows that still need first-pass readouts:

```bash
python transcript_store.py legacy-enrichment-queue --format text
python transcript_store.py legacy-enrichment-queue --format commands --provider openai-compatible --limit 10
```

The queue de-dupes same-hash or same-title legacy rows by default so batch enrichment does not waste provider calls. Use `--no-dedupe` only when you are auditing duplicate rows.

For AuraCall-backed burst processing, use the project-bound `Transcripts` agent client env:

```bash
python scripts/auracall_legacy_enrichment_batch.py \
  --env-file ~/.local/state/transcribe-audio/auracall-transcripts.env \
  enqueue --limit 25 --store

python scripts/auracall_legacy_enrichment_batch.py \
  --env-file ~/.local/state/transcribe-audio/auracall-transcripts.env \
  status ~/.local/state/transcribe-audio/auracall-batches/<manifest>.json --materialize --store
```

The enqueue command submits all selected readout requests to AuraCall in one response batch. AuraCall owns browser concurrency and interaction rate limits; this repo keeps the transcript payloads complete and later materializes completed responses into `*.readout.json` and `*.readout.md`.

Link already-imported legacy transcripts to recordings later, using an explicit media index instead of rescanning mounted drives:

```bash
find ~/HistoricalRecordings -type f \( -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.wav' \) -print > /tmp/media-index.txt
python legacy_media_link.py --media-index-file /tmp/media-index.txt
python legacy_media_link.py --media-index-file /tmp/media-index.txt --apply
```

`legacy_media_link.py` updates only matched legacy sidecars, re-ingests them through the store, and registers copied blobs under `~/.transcripts/blobs/`.

## Transcript Review API

Start the local read API for the planned React review console:

```bash
python transcript_api.py --store-dir ~/.transcripts --host 127.0.0.1 --port 18876
```

The API exposes `/api/library`, `/api/search`, `/api/documents/<id>`, `/api/documents/<id>/context`, and range-capable `/api/blobs/<blob_id>` routes over the user-scoped store. When `frontend/dist/` exists, the same server serves the built React console at `/`. Transcript ingestion registers existing source recordings as copied blobs under `~/.transcripts/blobs/` and links them to documents through SQLite pointers, so the UI can play recordings without streaming arbitrary original filesystem paths. See `docs/dev/transcript-review-api.md` for the endpoint contract.

### Run the watcher

```bash
# One scan pass, useful for testing
python watch_transcriptions.py --run-once --verbose

# Continuous service loop
python watch_transcriptions.py --verbose
```

The watcher stores its memory in `.openclaw/watch_transcriptions_state.json` so it can avoid reprocessing the same finished file every time it scans.

### Recommended config for the current Downloads workflow

For mobile meeting recordings synced into `~/Downloads`, start with:

- `watch_dir`: `~/Downloads`
- `glob`: `My Recording*.m4a`
- `backend`: `assembly`
- `settle_seconds`: `120`
- `calendar`: `{ "enabled": true, "providers": ["gog", "gws", "google-api"] }`
- `cli_args`: `["--text-output"]`

That is intentionally conservative. Two minutes of stability is usually better than accidentally transcribing the broken partial sync artifact.

### Running it as a user service (systemd)

Create `~/.config/systemd/user/transcribe-watch.service`:

```ini
[Unit]
Description=Watch Downloads for stable meeting recordings and transcribe them
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/workspace.local/transcribe-audio
ExecStart=%h/workspace.local/transcribe-audio/.venv/bin/python %h/workspace.local/transcribe-audio/watch_transcriptions.py
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
```

Then enable it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now transcribe-watch.service
journalctl --user -u transcribe-watch.service -f
```

If you prefer cron or another scheduler, `--run-once` also works for a periodic scan model, but a long-running user service is cleaner and catches files sooner.

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
   - `--calendar-providers gog,gws,google-api` to choose provider order.
   - `--calendar-gog-account` and `--calendar-gog-client` for `gog` tenant selection.
   - `--calendar-gws-config-dir` for a non-default `gws` config directory.

Lookup order is:

1. `gog calendar events` if `gog` is installed on `PATH`.
2. `gws calendar events list` if `gws` is installed on `PATH`.
3. Direct Google Calendar API access via `credentials.json` / `token.json`.

If an event is found, outputs are renamed to `YYYY-mm-dd HH-MM <event name> <original base>` (colons replaced with dashes) and the transcript is annotated with event metadata, including participants when available. If no provider returns a matching event, transcription continues without renaming.

When accessible calendars contain overlapping events for the same recording window, the transcript sidecar stores `event.matching_calendars`. Each entry includes the calendar ID, calendar summary, access role when available, event ID/summary, event start/end, overlap seconds, and coverage. This is intended for readout/routing context; it does not change the primary event selected for filenames.

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
