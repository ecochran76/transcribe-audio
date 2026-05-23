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
surface. It wires `/api/health`, `/api/library`, `/api/review-queue`, and
first-pass summary batch actions through the Vite dev proxy, with redacted
fallback rows when the API is offline.

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

Current implementation note: the Library kind filters are real scoped controls
with active `aria-pressed` state and row counts. The center pane also shows an
operator test-status strip with API state, rows in scope, active filter/search
state, latest smoke status, and a suggested next testing action.

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
- `codex app-server` for supervised App Intelligence runs with persistent sessions, branch/fork/rollback control, streamed events, and structured decision turns under a host-owned ledger.
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

Task-level provider selection is centralized in `intelligence_config.py`. It resolves from built-in defaults, optional user-scoped config at `~/.local/state/transcribe-audio/intelligence.config.json`, `TRANSCRIPTS_INTELLIGENCE_CONFIG`, per-task environment overrides, and explicit CLI/API overrides. Existing first-pass summary and contextual reread CLIs call this library before invoking providers, and the API exposes the resolved routing at `/api/intelligence/config`.

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
2. In progress: backend read API for store/library/search plus audio blob route, read-only review queue aggregation, and manifest-scoped first-pass summary prepare/submit/status actions. New ingests register media blobs; older stored transcripts need a migration/backfill pass to populate blob links. Older TXT/DOCX transcript outputs can be synthesized into private sidecars with `legacy_transcript_import.py` and marked for first-pass summary preparation; live historical imports inserted 70 deduped transcript rows. The first Sound Recordings import matched 44 source recordings, while targeted SoyLei Shared Drive media linking later added 16 matched blobs from an explicit `find` index. `transcript_store.py first-pass-summary-queue` now exposes the de-duped first-pass readout queue for stored transcripts, `/api/review-queue` summarizes local route-review files, App Intelligence `ask_for_human_review` decisions, filename-conflict decisions, and first-pass summary counts, `review_queue_maintenance.py` archives stale route-review files only after explicit approval, and the first-pass queue is currently clear after AuraCall dispatch-pool materialization.
3. Done: React + Vite shell with navbar, animated panes, library table, live review queue cards, and inspector wired to read API.
4. Login guard and share-link model borrowed from `previews`.
5. Contact/speaker review tables and merge audit.
6. Context-run and provenance-management surfaces.
7. In progress: Intelligence-management surfaces now show provider readiness,
   resolved task routing, reviewed config preview/apply, provider detail
   affordances, smoke status, queued smoke jobs, prepared App Intelligence run ledgers, and a
   selected-run inspector for events, policy, paths, plus a non-starting
   session-start preflight and control-plane daemon start that record ledger
   events before any model turn. Initial model-turn preflight now prepares reviewed prompt
   packet artifacts without sending prompts, and the inspector can read the
   packet JSON/text plus run a send-token preflight before any send action
   exists. The first gated send action starts one Codex app-server turn and
   records thread/turn ids plus raw event capture without executing downstream
   structured-decision actions. Turn-status capture can then store completion
   and output artifacts, and structured-decision validation can accept or
   reject the JSON shape without executing the decision. Ledger-only apply can
   record validated `continue_current_branch`, `stop`, and
   `ask_for_human_review` decisions. `continue_current_branch` leaves the run
   open with a `latest_continuation` record, and `ask_for_human_review`
   decisions are surfaced in the Review Queue for operator attention. The
   Review Queue can now record local-only annotations, resolutions, and
   reopens for App Intelligence human-review decisions, and shows structured
   request/status/count/materialization details for first-pass summary batch
   prepare, submit, and status responses, plus a read-only recent manifest list
   for resuming status checks after reload, covered by a browser smoke that selects a saved manifest and polls prepared status.
   Fork preflight can preview validated `fork_branches` decisions without
   creating threads, modifying branch state, or running provider work.
   Rollback preflight can preview validated `rollback` decisions without
   modifying branch state, reverting artifacts, creating threads, or running
   provider work.
8. Deposition/memory-harvest review UI over existing review/apply artifacts.

### App Intelligence Control Plane

`codex app-server` is the preferred control plane for long-lived or write-bearing intelligence workflows that need deterministic supervision. The transcript app remains the host: it owns run state, allowed actions, approval policy, eval gates, replay logs, and final apply/rollback decisions. App-server sessions are stochastic workers inside that harness, not the source of workflow authority.

The first backend surfaces are readiness reporting through `/api/intelligence/providers`, smoke-status metadata through `/api/intelligence/smokes`, queued smoke jobs through `/api/intelligence/smoke-jobs`, safe smoke evidence links through `/api/intelligence/smoke-evidence`, and prepared run-ledger management through `/api/intelligence/runs`. Readiness checks the local Codex binary and app-server protocol surfaces without starting sessions. Smoke status reports latest browser-smoke report paths, screenshot existence, check booleans, and disposable smoke run summaries without reading screenshot bytes or artifact contents. Smoke jobs execute only allowlisted API/browser/reload-resume smoke and cleanup commands, require explicit approval tokens, persist job metadata/stdout/stderr paths under `~/.local/state/transcribe-audio/smoke-jobs/`, and never accept arbitrary shell input. The Smoke Status card polls every 2 seconds while any loaded smoke job is queued or running, can read bounded stdout/stderr tails for one job without arbitrary file reads, shows redacted cleanup retention counts on completed cleanup jobs, visually distinguishes write-bearing cleanup apply jobs from read-only jobs with an inline legend, shows friendly smoke-job timing, groups recent jobs by action type, surfaces loaded-vs-total counts, links known browser-smoke reports/screenshots through a browser-smokes-confined endpoint, surfaces failed smoke jobs in a dedicated alert band ahead of successful history, and requires a typed `CLEANUP_APP_SMOKE_ARTIFACTS` confirmation before cleanup apply. Prepared ledgers persist under `~/.local/state/transcribe-audio/app-intelligence-runs/` with `run.json`, `events.jsonl`, `codex_events.jsonl`, branch placeholders, host-owned policy, structured-decision requirements, approval policy, eval policy, and RNG seeds. The React Intelligence panel can prepare these ledgers for the selected task and document, queue smoke jobs, select an existing ledger, inspect smoke status, inspect events/policy/paths/decision history/replay manifest/browser smoke/smoke cleanup/prompt-status artifacts/preflight artifacts/registered artifacts, run a non-starting session-start preflight, optionally append a `session_start_preflight` event with a separate preflight-event token, start only the managed Codex app-server control-plane daemon with `approval_token=START_APP_SERVER_SESSION`, prepare an initial prompt packet from a selected document plus resolved task route with `approval_token=PREPARE_MODEL_TURN_PREFLIGHT`, inspect the packet JSON/text through the read-only prompt-packet review endpoint, run a non-sending `SEND_APP_SERVER_MODEL_TURN` preflight against that packet, send the reviewed packet to one Codex app-server turn only when the same token is supplied, capture turn completion/output with `approval_token=CAPTURE_MODEL_TURN_STATUS`, validate captured JSON as a structured decision with `approval_token=VALIDATE_STRUCTURED_DECISION`, apply validated `continue_current_branch`, `stop`, or `ask_for_human_review` decisions as ledger-only records with `approval_token=APPLY_STRUCTURED_DECISION`, annotate/resolve/reopen human-review decisions with `approval_token=RECORD_HUMAN_REVIEW_DECISION`, preview `fork_branches` with `approval_token=PREVIEW_FORK_BRANCHES`, and preview `rollback` with `approval_token=PREVIEW_ROLLBACK`. Replay manifests expose ordered metadata only. Registered artifact reads require the artifact path to resolve inside the run directory and match the run ledger or event log. This preserves the boundary that no provider turn, Codex thread, branch, rollback, or write-bearing phase starts from ledger preparation, smoke-status inspection, smoke-job listing, smoke-job tail reads, smoke-evidence link reads, inspection, preflight, control-plane start, prompt-packet preparation, prompt-packet inspection, registered artifact read, send-preflight, fork preflight, or rollback preflight alone. The current send/status/validation/apply/review/fork-preflight/rollback-preflight path captures thread/turn ids, app-server events, output artifacts, accepted/rejected decision metadata, no-op apply records including non-terminal continuation records, local human-review records, fork preview artifacts, rollback preview artifacts, read-only decision history, replay-manifest artifact selection, live replay smoke, browser-assisted replay smoke, reload-resume UI smoke, smoke cleanup, prompt-status artifact selection, event-log preflight artifact selection, registered artifact content, smoke job records, and smoke-status metadata but does not execute fork, rollback, memory write, route apply, repository write, external network write, or deposition action. This establishes the replay boundary before enabling branch, rollback, or write-bearing phases from the UI.

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
