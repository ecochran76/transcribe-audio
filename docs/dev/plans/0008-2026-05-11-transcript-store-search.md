# Plan 0008 | User-Scoped Transcript Store And Search

State: CLOSED

Lane: P08

## Scope

Store transcript artifacts, first-pass readouts, and context-enriched readouts in a user-scoped database under `~/.transcripts` and make them searchable.

## Non-Goals

- No repo-local storage of raw transcripts or private readouts.
- No repo-local model weights, secrets, or private transcript artifacts.
- No Drive/Odoo/Graphiti writes from the store.

## Current State

Closed. `transcript_store.py` initializes `~/.transcripts/transcripts.sqlite3`, copies ingested JSON artifacts into `~/.transcripts/artifacts/`, and indexes transcript, readout, and contextual-readout content. Search combines SQLite FTS5 lexical matching with provider-backed semantic embeddings. The default path uses local Ollama `ollama/nomic-embed-text`, chunks long documents before embedding, stores chunk vectors with transcript timestamp/speaker metadata when available, and applies model-specific document/query prefixes; `openai-compatible` is available for API-backed embeddings, and `debug-hash` is an explicit test/offline fallback. `transcript_store.py search --context` opens a selected search hit directly; `transcript_store.py context` opens a stored document/chunk and prints nearby chunks plus media timestamp guidance. Both context paths support compact pure-JSON output for machine consumers. `scripts/context_packet_recipe.py` prints downstream summarize/route/reread commands from compact context packets, while `scripts/context_packet_apply.py` previews those steps by default and executes them only with `--apply`. Executed apply runs write sanitized manifests under `~/.local/state/transcribe-audio/context-packet-runs/` by default and `--list-manifests` provides a recent-run inspection surface. `summarize_transcript.py` and `contextual_reread.py` can ingest generated readouts with `--store`; transcription artifact generation can opt in with `TRANSCRIPTS_STORE=true`.

## Work Items

- Done: define user-scoped runtime home and SQLite schema.
- Done: ingest transcript/readout/contextual-readout JSON artifacts.
- Done: copy source JSON artifacts into the runtime store.
- Done: implement lexical FTS search.
- Done: implement provider-backed semantic ranking with local Ollama default.
- Done: store embedding provider/model metadata with each ingested document.
- Done: apply `nomic-embed-text` document/query prefixes for ingest and search.
- Done: chunk long documents before embedding to avoid provider context overflow.
- Done: add watcher config support for automatic store ingestion.
- Done: backfill recent Downloads transcript artifacts plus recent readout artifacts into `/home/ecochran76/.transcripts`.
- Done: add `transcript_store.py backfill` with dry-run/apply counts, kind/pattern/time filters, limit, force, and current-artifact skips.
- Done: add chunk-level storage/retrieval with `best_chunk` snippets and chunk semantic scores.
- Done: add transcript chunk metadata for character offsets, utterance timestamp ranges, speaker spans, and utterance counts.
- Done: add `transcript_store.py context` to jump from a `best_chunk` hit to nearby transcript context and media timestamp guidance.
- Done: add `transcript_store.py search --context` so the top hit can open without manually copying document id and chunk index.
- Done: add compact JSON output for direct context and search-to-context payloads.
- Done: add a non-mutating compact context recipe helper for downstream summarize/route/reread commands.
- Done: add a preview-first apply helper for executing compact context downstream steps with explicit `--apply`.
- Done: add executed-run manifests for context-packet apply runs.
- Done: add a manifest listing command for recent context-packet apply runs.

## Acceptance Criteria

- Store initialization creates only user-scoped runtime files.
- Ingest preserves source and stored artifact paths.
- Search returns ranked JSON results with snippets and source paths.
- Context opens a stored document/chunk and returns nearby chunk text plus timestamp/media guidance when available.
- Tests cover ingest, lexical search, semantic ranking, and CLI output.

## Validation

- Unit tests over temp stores.
- Manual smoke ingesting the SoyLei/Tempo transcript, readout, and contextual readout.
- Closure review on 2026-05-12 found every P08 acceptance criterion satisfied.
- Remaining UI/operator polish should be tracked as a separate lane if needed.
