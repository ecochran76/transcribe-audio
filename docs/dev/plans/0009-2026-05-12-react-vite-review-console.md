# Plan 0009 | React Vite Review Console

State: OPEN

Lane: P09

## Scope

Create a React + Vite operator console for transcript search, recording playback, contact/speaker review, context gathering, provenance management, intelligence-provider management, and deposition/memory-harvest review.

The console should reuse these proven patterns:

- `../previews`: single-operator login guard, session/artifact sharing, revocable share-link semantics, and feedback/approval boundaries.
- `../buffer-cli`: sticky top navbar, animated collapsible left pane, central table/viewport, animated right inspector pane, readiness/account menu, and status-dense review UI.

## Non-Goals

- No raw private transcript fixtures in the repo.
- No tenant secrets, OAuth tokens, API keys, audio blobs, share tokens, or live runtime state in tracked files.
- No unattended external writes from the UI until the backend apply contracts exist and expose preview/apply boundaries.
- No multi-user account database in the first slice; use a single-operator guard plus scoped share links first.
- No replacement of existing CLIs; the first UI should orchestrate and inspect the same artifacts/contracts they already produce.

## Current State

The repo already has transcript artifacts, first-pass readouts, contextual readouts, route decisions, deposition previews, memory-harvest review/apply artifacts, and a user-scoped SQLite/vector transcript store under `~/.transcripts`.

The first UI shell now exists under `frontend/`. It provides the navbar,
animated pane layout, central library/review viewport, and right inspector
surface. It is read-only and currently wires `/api/health` plus `/api/library`
through the Vite dev proxy, with redacted fallback rows when the API is offline.

The remaining UI layer should make the workflow operational:

1. Search or pick a recording/transcript.
2. Play the recording and inspect transcript/readouts.
3. Deduplicate contacts and map speakers to contacts.
4. Gather provenance context from Google Workspace, msgcli, Odollo, Graphiti, and local store sources.
5. Generate or inspect contextual readouts.
6. Review deposition and memory-harvest candidates.
7. Share selected artifacts for human review without exposing the full operator surface.

## Information Architecture

### Navbar

Navbar items should map to operator jobs, not implementation modules:

- `Library`: search and browse recordings, transcripts, summaries, contextual readouts, and stored artifacts.
- `Review Queue`: items needing human approval, including low-confidence routes, pending memory candidates, contact/speaker conflicts, and failed duplicate/provenance checks.
- `Context Runs`: context-gathering pipelines, provenance packs, reread status, deterministic recurring-meeting recipes, and run manifests.
- `Contacts`: deduplicated people/organization records, speaker aliases, email/calendar identities, Odoo contacts, msgcli identities, and merge history.
- `Provenance`: connected source profiles and search surfaces for GWS, msgcli, Odollo, Graphiti, local files, and future Drive/Docs targets.
- `Intelligence`: provider registry, task-to-provider routing, readiness, model config, cost/latency notes, and provider-specific smoke results.
- `Depositions`: local filesystem, Drive, Odoo, and Graphiti memory-harvest preview/apply history.
- `Settings`: runtime profile, auth/share-link controls, storage paths, service health, watcher status, and retention.

The topbar should also include global search, current runtime profile, readiness indicators, and account/share controls borrowed from `buffer-cli` and `previews`.

### Left Pane

The left pane is the workflow navigator and filter surface. It should be collapsible and animated.

Per navbar section, it should show:

- `Library`: saved filters, date ranges, meeting/calendar filters, kind filters (`recording`, `transcript`, `summary`, `contextual readout`), processing status, and semantic-search controls.
- `Review Queue`: queue buckets (`Needs route review`, `Needs speaker IDs`, `Needs context approval`, `Needs memory review`, `Failed preflight`), priority filters, and SLA/age filters.
- `Context Runs`: deterministic recipes, recurring meeting profiles, active/failed/completed runs, and provider/source filters.
- `Contacts`: duplicate clusters, unassigned speakers, source-system filters, and merge queues.
- `Provenance`: tenant/source tree with GWS profiles, calendars, Gmail, Drive, msgcli accounts, Odollo tenants, and Graphiti groups.
- `Intelligence`: provider list grouped by capability (`summarize`, `route`, `reread`, `classify`, `embed`, `memory harvest`), readiness, and preferred/default routing.
- `Depositions`: target types, preview/apply status, warnings, and external write gates.

### Central Viewport

The central viewport is the primary work surface. It should be table-first where review throughput matters and document/player-first where reading matters.

Core views:

- Search results table with transcript/readout/contextual-readout rows, best chunk, semantic score, calendar context, contact confidence, processing state, and warnings.
- Recording detail with waveform/timeline placeholder, audio player, transcript segments, speaker lanes, timestamp seeking, and playback speeds such as `0.75x`, `1x`, `1.25x`, `1.5x`, and `2x`.
- Summary/readout comparison view with initial summary beside contextualized readout and a visible provenance delta.
- Context-run timeline showing acquisition steps, source hits, excluded weak sources, warnings, and deterministic recurring-meeting recipe status.
- Review table for pending route, contact, deposition, and memory candidates with batch actions.
- Contacts table with dedupe clusters, aliases, source identities, and speaker assignment status.

The center should own selection. Selecting a row, chunk, candidate, contact, or provider opens details/actions in the right pane.

### Right Pane

The right pane is the inspector/action panel. It should be collapsible, animated, and resizable.

It should show:

- Recording inspector: source blob metadata, original filename, media duration, calendar match, share/download links, derived artifact links, and storage pointer.
- Transcript inspector: selected chunk text, utterance timing, speakers, contact candidates, confidence, and quick speaker assignment.
- Contact inspector: dedupe evidence, linked calendar/Gmail/Odoo/msgcli identities, merge/split actions, and audit history.
- Context inspector: provenance source packet, included/excluded status, quality score, reason, source tenant/profile, and fetch/run action buttons.
- Intelligence inspector: provider readiness, selected model/agent, task routing, last smoke, failure detail, and config surface.
- Deposition/memory inspector: preview action details, warnings, review decisions, duplicate-check output, apply status, and explicit gated write controls.
- Share inspector: create/list/revoke scoped links, select read-only vs feedback-capable, and expose copyable links without logging raw tokens.

## Data And Backend Contracts

### Runtime Home

Use the existing user-scoped runtime split:

- `~/.transcripts/transcripts.sqlite3`: metadata, text indexes, embeddings, contact tables, speaker aliases, blob pointers, and workflow state.
- `~/.transcripts/blobs/`: ingested source recordings and derived binary artifacts, addressed by content hash or stable blob id.
- `~/.transcripts/artifacts/`: copied JSON/Markdown artifacts already used by the store.
- `~/.local/state/transcribe-audio/`: operator run manifests, review queues, share-link records, provider readiness caches, and apply logs.

Blob storage should be content-addressed or id-addressed with DB pointers. The UI should never rely on original Downloads paths as the durable source of truth after ingestion.

### API Shape

The first backend should be local and boring:

- Provide read APIs over the existing SQLite store and artifact files.
- Add explicit blob routes for playback/download with range-request support for audio seeking.
- Add share routes modeled after `previews`: scoped bearer links, hashed tokens at rest, expiration, revocation, read-only vs feedback-capable mode.
- Expose preview/apply endpoints only where a CLI already has the same explicit gate.
- Keep tenant/provider credentials in ignored runtime config and environment variables.

### Contacts And Speaker Identification

Add first-class tables for:

- `contacts`: canonical deduped person/org records.
- `contact_identities`: email, calendar attendee, Odoo partner id, msgcli handle, Slack/user ids, phone, and source-system aliases.
- `speaker_assignments`: transcript speaker label to contact mapping with confidence, reviewer, evidence, and timestamps.
- `contact_merge_events`: reversible audit trail for merges/splits.

Speaker identification should be part of review workflow, not a hidden post-processing side effect.

### Context Gathering

Context gathering lives between first-pass summary and contextual readout.

It should support:

- Manual runs from the UI.
- Automatic runs based on config and filters.
- Deterministic recipes for recurring meetings or known matter patterns.
- Source profiles for GWS Calendar/Gmail/Drive, msgcli, Odollo tenants, Graphiti groups, and local transcript-store retrieval.
- Provenance packs that record included/excluded source decisions before reread.

Automatic progression must remain configurable by confidence, meeting pattern, source availability, and warning state. Low-confidence or warning-bearing runs land in `Review Queue`.

### Provenance Management

Provenance providers are source profiles, not global toggles:

- `gws`: multitenant Google Workspace profiles for calendars, Gmail, Drive search, and later Docs/Sheets.
- `msgcli`: message/contact/search profiles.
- `odollo`: multiple Odoo tenant profiles for contacts and log notes.
- `graphiti`: memory groups with sensitivity/retrieval policy.
- `local`: transcript-store lexical/semantic retrieval.

The UI should show readiness, last smoke, accessible scopes, and which workflows may use each profile.

### Intelligence Management

Intelligence providers should be managed by capability:

- OpenAI-compatible API using `OPENAI_API_KEY`/base URL.
- AuraCall MCP/OpenAI-compatible endpoints.
- `codex exec`.
- OpenClaw agent calls.
- Graphiti memory lookup/write workflows.
- Local embedders for semantic search.

Different workflow stages may use different providers. The UI should expose task routing such as:

- summary generation provider;
- context-source ranking provider;
- contextual reread provider;
- speaker/contact disambiguation provider;
- deposition/memory-candidate reviewer provider;
- embedding provider.

## Frontend Layout Contract

Use a React + Vite app under `frontend/` with:

- A sticky dark topbar inspired by `../buffer-cli/frontend/src/App.jsx`.
- CSS variables and a distinct transcript-console visual system, not a generic template.
- Animated left and right panes using CSS grid width transitions.
- Table viewport with sortable columns and status chips.
- Detail inspector actions kept to the right pane.
- No hard-coded private data in fixtures; use redacted local development seed data only.

## Implementation Slices

1. Done: product plan and route contract.
2. In progress: backend read API for store/library/search plus audio blob route and read-only review queue aggregation. New ingests register media blobs; older stored transcripts need a migration/backfill pass to populate blob links. Legacy TXT/DOCX transcript outputs can be synthesized into private sidecars with `legacy_transcript_import.py` and marked for enrichment; live historical imports inserted 70 deduped legacy transcript rows. The first Sound Recordings import matched 44 source recordings, while targeted SoyLei Shared Drive media linking later added 16 matched blobs from an explicit `find` index. `transcript_store.py legacy-enrichment-queue` now exposes the de-duped first-pass readout queue for those legacy rows, `/api/review-queue` summarizes local route-review files, filename-conflict decisions, and legacy enrichment counts, and `review_queue_maintenance.py` archives stale route-review files only after explicit approval.
3. Done: React + Vite shell with navbar, animated panes, library table, live review queue cards, and inspector wired to read API.
4. Login guard and share-link model borrowed from `previews`.
5. Contact/speaker review tables and merge audit.
6. Context-run and provenance-management surfaces.
7. Intelligence-management surfaces and provider readiness smokes.
8. Deposition/memory-harvest review UI over existing review/apply artifacts.

## Acceptance Criteria

- The plan is wired into `ROADMAP.md`.
- Frontend responsibilities are separated from CLI/provider implementation.
- Layout defines navbar, left pane, central viewport, and right pane before scaffolding.
- Runtime, tenant, secrets, blobs, and share-token state stay outside tracked repo files.
- Audio playback is backed by stored blobs with DB pointers, not original transient paths.
- Human review is preserved for warning-bearing, low-confidence, sensitive, or external-write workflows.

## Validation

- Documentation review against repo policies.
- After scaffold: `npm`/Vite build and lint/type checks if configured.
- Backend API tests for store search, blob range reads, contact merge audit, and share-link auth.
- UI smoke with redacted fixtures only.
- Live local smoke against `~/.transcripts` only after read-only routes are implemented.

## API Contract

The initial local read API contract is documented in `docs/dev/transcript-review-api.md`.
