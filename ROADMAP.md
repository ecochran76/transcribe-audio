# Roadmap

`ROADMAP.md` is the master plan for this repo. Bounded execution plans live under `docs/dev/plans/`; turn-by-turn history lives in `RUNBOOK.md`.

## P01 | Normalize Transcript Artifacts

State: CLOSED

Current State: Transcription outputs include `*.transcript.json` sidecars with transcript text, structured utterances, output paths, backend, timing, and optional event metadata. Watcher state records sidecar paths for successful runs. A temp-location short-recording watcher smoke validated TXT/DOCX/sidecar output and artifact path capture.

Plans:

- `docs/dev/plans/0001-2026-05-04-normalize-transcript-artifacts.md`

Definition of Done:

- Both speech backends emit sidecar JSON artifacts.
- Output path generation returns structured metadata.
- Watcher state records artifact paths.
- Tests cover artifact serialization and current output behavior.

## P02 | Calendar Provider Configuration

State: CLOSED

Current State: Calendar lookup supports explicit provider order, `gog` account/client selection, `gws` config-dir selection, lazy built-in Google API fallback, and `matching_calendars` context for overlapping events found on accessible calendars. CLI flags and watcher `calendar` config expose these fields. A temp-location watcher `--run-once` smoke validated structured calendar config expansion and `gog` provider lookup.

Plans:

- `docs/dev/plans/0002-2026-05-04-calendar-provider-config.md`

Definition of Done:

- Calendar providers are configured in ordered policy.
- `gog` supports account/client selection.
- `gws` supports environment/config-dir selection.
- Built-in Google Calendar API remains fallback.
- Verbose logs show provider choice and provider failures.

## P03 | Intelligence Readouts

State: CLOSED

Current State: Readout JSON/Markdown schemas exist and `summarize_transcript.py` can generate structured readouts from transcript sidecars through an OpenAI-compatible API or `codex exec`. Watcher jobs can enable readout post-processing behind config, and readout failures do not mark transcription as failed. Readout prompts include calendar overlap context from `event.matching_calendars` plus redundant user-payload JSON-only instructions for browser-backed providers that do not reliably honor system messages. Real AuraCall and `codex-exec` provider smokes generated valid readout JSON/Markdown from the SoyLei/Tempo transcript.

Plans:

- `docs/dev/plans/0003-2026-05-04-intelligence-readouts.md`

Definition of Done:

- Readout JSON and Markdown schemas exist.
- At least one OpenAI-compatible intelligence provider works unattended.
- Optional `../auracall` and OpenClaw provider seams are defined.
- Watcher can run readout generation behind config.

## P04 | Matter Routing And Contextual Reread

State: OPEN

Current State: The first dry-run routing slice exists. `routing_artifacts.py` defines the route/provenance/review schemas and `route_transcript.py` emits auditable `*.route.json` decisions from existing transcript/readout artifacts. Calendar overlap metadata from `event.matching_calendars` is represented as provenance, and low-confidence decisions can write a local review queue item. `context_sources.py` adds explicit read-only adapters for live `gws` Calendar/Drive metadata provenance, Graphiti/OpenClaw advisory provenance, and Odollo/Odoo contact and log-note provenance across selectable profiles. Non-calendar provenance is now quality-filtered with source-type-specific profiles for Drive file identity, Odollo contact identity, Odollo note subjects, and Graphiti labels/previews before it can support selected routes or contextual rereads; weak sources are retained under `provenance_pack.excluded_sources` with warnings. `contextual_reread.py` generates upgraded readouts from transcript, prior readout, route decision, and cited supporting provenance sources, and carries route warnings into contextualization metadata. `scripts/context_packet_apply.py` has completed a real reviewed apply over the SoyLei/Tempo transcript/readout pair, producing route and contextual-readout artifacts plus a sanitized run manifest. Graphiti facts and episodes are recorded as evidence, while Graphiti nodes may add low-confidence advisory route candidates. Deeper Drive/Docs content fetch and calibrated source-specific scoring remain future work.

Plans:

- `docs/dev/plans/0004-2026-05-04-matter-routing-contextual-reread.md`

Definition of Done:

- Route decision schema records candidates, confidence, evidence, and rejected alternatives.
- Low-confidence routes are queued for review.
- `gws` provenance adapter can add Calendar/Drive context without external writes.
- Graphiti/OpenClaw lookup adapter can propose candidate matters.
- Odollo/Odoo provenance adapter can add contact and log-note context without external writes.
- Contextual reread uses supporting context and produces an upgraded readout.

## P05 | Deposition And Memory Harvest

State: OPEN

Current State: Transcripts remain in the watched folder unless explicitly deposited. `deposition_preview.py` defines the no-write deposition and memory-harvest preview contract over contextual readouts. Preview actions can describe local filesystem, Google Drive, and Odoo targets, but they are explicitly `status=preview` with `writes_enabled=false`. A live preview over the context-packet-generated SoyLei/Tempo contextual readout produced one local-filesystem copy action and six Graphiti memory-harvest candidates without enabling writes. `deposition_apply.py` can apply only local filesystem preview actions with idempotent same-hash skips and versioned conflict handling. `memory_harvest_apply.py` previews reviewed Graphiti writes by default, can generate per-candidate review templates with `--init-review`, limits review-file applies to approved candidates, records rejected/pending candidates in the audit, performs duplicate preflight checks, and requires `--apply --approval-token APPROVE_GRAPHITI_MEMORY_HARVEST` for live memory writes. One reviewed SoyLei/Tempo relationship-context candidate has been written to Graphiti and read back from `transcribe_audio_main`. Memory harvest candidates are extracted only from structured readout `memory_candidates`; raw transcript text is excluded. Route-level provenance filtering now keeps weak sources out of contextual rereads before deposition preview, and preview JSON carries contextual warnings. Drive/Odoo apply paths and calibrated multi-candidate memory-approval operations remain future work.

Plans:

- `docs/dev/plans/0005-2026-05-04-deposition-memory-harvest.md`

Definition of Done:

- Local filesystem deposition works.
- Google Drive deposition is implemented through `gog` or `gws`.
- Odoo deposition has a defined target model before unattended writes are enabled.
- Graphiti/OpenClaw memory candidates are harvested from reviewed readout fields.

## P06 | Service Reliability And Observability

State: OPEN

Current State: The watcher runs under systemd and has heartbeat logging, but recent `ffprobe` PATH failure showed readiness failures need stronger visibility.

Plans:

- `docs/dev/plans/0006-2026-05-04-service-reliability-observability.md`

Definition of Done:

- Service environment checks fail loudly for missing dependencies.
- Heartbeats include blocked-reason summaries when candidates are queued.
- Runbook commands document service health checks and recovery.
- Tests cover readiness failure classification.

## P07 | OpenClaw Transcripts Agent

State: CLOSED

Current State: Portable OpenClaw workspace Markdown files exist for the
`transcripts` agent under `openclaw/agents/transcripts/workspace/`. The
dry-run-first installer can copy those files to
`~/.openclaw/workspace-transcripts`, create the agent, set identity, and apply
the exact Slack channel-peer binding when given a resolved channel id. The live
agent is installed and bound to Slack account `default`, private channel
`oc-transcripts`, conversation id `C0B3WDRN38Q`. A live Slack smoke routed to
`transcripts` and returned `TRANSCRIPTS_BINDING_SMOKE_OK`.

Plans:

- `docs/dev/plans/0007-2026-05-11-openclaw-transcripts-agent.md`

Definition of Done:

- Portable agent workspace files are stored in this repo.
- Install routine creates or updates the OpenClaw `transcripts` agent
  idempotently.
- The agent is bound only to Slack account `default` and private channel
  `oc-transcripts`.
- Live install is verified with OpenClaw agent, channel, and route-binding
  status checks.

## P08 | User-Scoped Transcript Store And Search

State: CLOSED

Current State: CLOSED. `transcript_store.py` creates a user-scoped `~/.transcripts` runtime home with `transcripts.sqlite3` plus copied JSON artifacts under `~/.transcripts/artifacts/`. The store can ingest transcript artifacts, first-pass readouts, and contextual readouts. Search combines SQLite FTS5 lexical matching with provider-backed document and chunk embeddings; the default is local Ollama `ollama/nomic-embed-text` with long-document chunking and document/query prefixes, with `openai-compatible` support and an explicit `debug-hash` fallback for tests. Search results include `best_chunk` segment snippets/scores plus transcript chunk metadata for character offsets, utterance time ranges, speakers, and utterance counts. `transcript_store.py search --context` opens the selected search hit directly; `transcript_store.py context` can also open a specific document/chunk and print nearby transcript chunks plus media timestamp guidance when the source artifact includes media paths. Compact JSON modes are available for both direct context and search-to-context output. `scripts/context_packet_recipe.py` turns those packets into explicit downstream summarize/route/reread commands, and `scripts/context_packet_apply.py` previews or explicitly executes those commands only when `--apply` is present. Executed apply runs write sanitized manifests under `~/.local/state/transcribe-audio/context-packet-runs/` unless disabled; `--list-manifests` lists recent runs. Readout CLIs can ingest generated readouts with `--store`; transcription can opt in with `TRANSCRIPTS_STORE=true`. Watcher jobs can also enable a `store` block so successful transcript artifacts and generated readouts are ingested automatically. `transcript_store.py backfill` provides deterministic dry-run/apply enumeration with skip/update/insert/error reporting and safe excludes for copied store internals. The live user store currently contains 9 recent transcript artifacts, 3 readouts, and 1 contextual readout with Ollama/Nomic vectors, 369 chunk rows, and 247 timestamped transcript chunks.

Plans:

- `docs/dev/plans/0008-2026-05-11-transcript-store-search.md`

Definition of Done:

- User-scoped store initializes without secrets in the repo.
- Transcript/readout/contextual-readout artifacts are ingested and copied into the store.
- Lexical and semantic search return ranked JSON results.
- Watcher and service flows can opt into automatic ingestion.

Closeout Notes:

- P08 definition of done is satisfied by the implemented store, ingestion, ranked lexical/semantic search, and watcher/service opt-in.
- Context navigation, compact JSON handoff, downstream recipe/apply helpers, and apply-run manifests were completed as operator polish inside the lane.
- Future UI/operator polish should be tracked separately rather than reopening the core store/search lane.

## P09 | React Vite Review Console

State: OPEN

Current State: OPEN. A bounded product plan defines the React + Vite operator console. The first `frontend/` Vite shell exists with a sticky navbar, animated left filter pane, central library/review viewport, and right inspector pane inspired by the `buffer-cli` layout. The shell reads `/api/health`, `/api/library`, and `/api/review-queue` through a Vite dev proxy to `transcript_api.py`, falls back to redacted fixture rows when the API is offline, and surfaces live review buckets from user-scoped runtime state. `transcript_api.py` provides the first local read API for library listing, search, document detail/context, registered blob playback/download, and read-only review-queue aggregation over route-review files, filename-conflict decisions, and legacy enrichment queue counts. `review_queue_maintenance.py` provides a reviewed archive path for stale local route-review files whose referenced route decisions no longer exist. Transcript ingestion copies existing source recordings into `~/.transcripts/blobs/`, records blob pointers in SQLite, and exposes range-capable `/api/blobs/<blob_id>` routes so the UI does not stream arbitrary original filesystem paths. `legacy_transcript_import.py` can synthesize private sidecars for older TXT/DOCX transcript outputs under `~/.transcripts/legacy-artifacts/`, de-dupe by source hash and normalized title, and mark imports for re-enrichment. Historical imports have inserted 70 deduped legacy transcript rows; targeted Shared Drive media linking added 16 more recording blobs, leaving 60 transcript documents blob-linked and 10 imported legacy rows still unmatched. `transcript_store.py legacy-enrichment-queue` lists de-duped pending first-pass readouts and emits runnable summary commands; the live queue currently reports 29 pending enrichment items after subsequent readout work. Runtime blobs, share tokens, tenant credentials, and live workflow state remain in `~/.transcripts` or `~/.local/state/transcribe-audio/`, not tracked repo files.

Plans:

- `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md`

Definition of Done:

- React + Vite frontend shell exists with navbar, animated left pane, central viewport, and right inspector pane.
- Backend read APIs expose library/search/readout/contact/provenance state without leaking secrets.
- Source recordings are playable from stored blobs through DB pointers with seek/range support.
- Login and scoped artifact sharing reuse the `previews` design contract.
- Human review workflows cover speaker/contact assignment, context acquisition, deposition, and memory harvest.
