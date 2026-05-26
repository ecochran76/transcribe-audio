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

Current State: The first dry-run routing slice exists. `routing_artifacts.py` defines the route/provenance/review schemas and `route_transcript.py` emits auditable `*.route.json` decisions from existing transcript/readout artifacts. Calendar overlap metadata from `event.matching_calendars` is represented as provenance, and low-confidence decisions can write a local review queue item. `context_sources.py` adds explicit read-only adapters for live `gws` Calendar/Drive metadata provenance, Graphiti/OpenClaw advisory provenance, and Odollo/Odoo contact and log-note provenance across selectable profiles. Non-calendar provenance is now quality-filtered with source-type-specific profiles for Drive file identity, Odollo contact identity, Odollo note subjects, and Graphiti labels/previews before it can support selected routes or contextual rereads; weak sources are retained under `provenance_pack.excluded_sources` with warnings. `contextual_reread.py` generates upgraded readouts from transcript, prior readout, route decision, and cited supporting provenance sources, and carries route warnings into contextualization metadata. `scripts/context_packet_apply.py` has completed a real reviewed apply over the SoyLei/Tempo transcript/readout pair, producing route and contextual-readout artifacts plus a sanitized run manifest. Graphiti facts and episodes are recorded as evidence, while Graphiti nodes may add low-confidence advisory route candidates. Plan 0011 closed the first source-quality calibration profile, `p04-source-quality-v1`: a reviewed local corpus evaluated 12 source decisions across Calendar, Drive/Docs, Graphiti, and Odollo with zero false positives and zero false negatives, and route/contextual artifacts now record the active profile. Deeper Drive/Docs content fetch remains future work after calibrated metadata-level scoring.

Plans:

- `docs/dev/plans/0004-2026-05-04-matter-routing-contextual-reread.md`
- `docs/dev/plans/0011-2026-05-23-p04-provenance-calibration.md`

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

State: CLOSED

Current State: The watcher runs under systemd with heartbeat logging, no-progress restart behavior, startup readiness checks, and a `--check` doctor path. Readiness checks fail loudly for missing `ffprobe`, missing watch directories, missing backend scripts, and missing readout scripts before the service loop starts; `--check --check-json` exposes the same diagnostics for automation. Candidate state records `blocked_kind`, `blocked_reason`, and `blocked_since`, backend failures record `failure_kind` and `failure_reason`, and heartbeat logs summarize queued work as `blocked=kind=count` so active services explain minimum-age waits, settling, incomplete media, missing tools, media probe failures, and retry backoff.

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

Current State: OPEN. A bounded product plan defines the React + Vite operator console. The first `frontend/` Vite shell exists with a sticky navbar, animated left filter pane, central library/review viewport, and right inspector pane inspired by the `buffer-cli` layout. The shell reads `/api/health`, `/api/library`, `/api/conversations`, and `/api/review-queue` through a Vite dev proxy to `transcript_api.py`, falls back to redacted fixture rows when the API is offline, surfaces live review buckets from user-scoped runtime state, provides functional Library kind filters with active state and counts, disables planned navigation instead of reusing unwired views, exposes attractive SVG pane collapse controls, supports mouse/keyboard pane resizing, shows selected artifact readouts as human-readable summaries instead of raw JSON, follows readout-to-source-transcript audio links through `/api/documents/<id>/related` when the readout itself has no blob, keeps raw context JSON behind a developer-labelled link, opens a modal conversation workspace for raw audio, re-transcription, raw summary, context workbench, speaker/contact identity, and final readout stages, and keeps API/row-scope/filter diagnostics inside the Library diagnostics disclosure instead of routine Settings chrome. The Intelligence panel now lists provider readiness/details, smoke status, queued smoke jobs with short polling, read-only stdout/stderr tails, browser-smoke report/screenshot links, failed-job alert band, smoke-job filter toggles, redacted cleanup retention counts, write-bearing/read-only smoke-job badges with an inline legend, friendly smoke-job timing, action-type grouping, loaded-vs-total smoke-job counts, and typed cleanup-apply gating, resolved task routing, reviewed config preview/apply, prepared App Intelligence run ledgers, a read-only selected-run inspector for ledger events/policy/paths/decision history/replay manifest/browser smoke/smoke cleanup/prompt-status artifacts/preflight artifacts/registered artifacts, non-starting session-start preflight, control-plane daemon start for prepared ledgers, reviewed prompt-packet preparation before any model turn, read-only prompt-packet JSON/text inspection, non-sending send-token preflight, gated model-turn send that records thread/turn ids without downstream actions, turn-status capture that stores completion/output artifacts, structured-decision validation that records accepted/rejected metadata without executing host actions, ledger-only apply for validated `continue_current_branch`, `stop`, and `ask_for_human_review` decisions, fork preflight for validated `fork_branches` decisions, and rollback preflight for validated `rollback` decisions. The Review Queue can also annotate, resolve, or reopen App Intelligence human-review items as local ledger records, show first-pass summary batch request/status/count/materialization details after prepare, submit, or status checks, list recent first-pass batch manifests so status checks can resume after reload, and smoke that reload-resume path through the Review Queue UI. `transcript_api.py` now provides the local backend for library listing, search, document detail/context/related-document lookup, registered blob playback/download, read-only review-queue aggregation including App Intelligence human-review decisions, intelligence-provider readiness, App Intelligence smoke status and allowlisted smoke-job queueing including first-pass resume UI smoke and browser-smoke evidence serving, resolved intelligence task config, prepared App Intelligence run ledgers, App Intelligence session-start preflight/control-plane start/model-turn preflight/prompt-packet review/replay manifest/browser smoke/smoke cleanup/registered artifact read/send-preflight/model-turn send/turn-status capture/structured-decision validation/ledger-only apply/human-review annotation/fork preflight/rollback preflight, and manifest-scoped first-pass summary batch prepare/submit/status operations. `intelligence_config.py` centralizes named intelligence profiles plus component profile selections for first-pass summary, contextual reread, ranking/routing, speaker disambiguation, memory review, embedding, and app-supervisor workflows, while retaining compatible task-level overrides. Submit requires `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`, status can materialize completed readouts back into the store, and batch manifests are constrained to the user-scoped state directory. `review_queue_maintenance.py` provides a reviewed archive path for stale local route-review files whose referenced route decisions no longer exist. Transcript ingestion copies existing source recordings into `~/.transcripts/blobs/`, records blob pointers in SQLite, and exposes range-capable `/api/blobs/<blob_id>` routes so the UI does not stream arbitrary original filesystem paths. `legacy_transcript_import.py` can synthesize private sidecars for older TXT/DOCX transcript outputs under `~/.transcripts/legacy-artifacts/`, de-dupe by source hash and normalized title, and mark imports for re-enrichment. Historical imports, media linking, and AuraCall-backed first-pass generation have populated the live store with 240 stored documents: 164 transcripts, 74 readouts, and 2 contextual readouts, with 122 blobs, 144 document-blob links, and 6560 chunk rows. `transcript_store.py first-pass-summary-queue` now reports no pending first-pass summary items. Runtime blobs, share tokens, tenant credentials, intelligence config, App Intelligence run ledgers, smoke job records, and live workflow state remain in `~/.transcripts` or `~/.local/state/transcribe-audio/`, not tracked repo files.

The Library now calls `/api/conversations` with active query/kind filters to group transcript, first-pass readout, and contextual readout artifacts into server-backed conversation rows with workflow progress icons, explicit loading/empty/error table states, a `Load more` control for paginated results, URL-addressable query/filter/selection/workspace-tab state, a copy-workspace-link affordance for dogfooding feedback, and no separate artifact rows, and the conversation workspace loads `/api/conversations/<document_id>` so transcript, summary, final readout, artifact membership, participants, and media are available in one payload. Library columns are operator-resizable, and the media column exposes a healthy `Play` action when source audio is available instead of raw blob-status text. The inspector shows metadata, audio, and a short conversation summary without falling back to misleading raw transcript text. The browser-reviewed UI density pass tightened topbar navigation, pane/card spacing, Library/inspector headings, workflow tabs, and conversation workspace chrome; the share fallback is a non-blocking selectable URL field rather than a JavaScript prompt. `scripts/smoke_library_deeplink_share_ui.py` now makes the Library deep-link/share checks repeatable with `agent-browser` and persisted JSON/screenshot evidence. The conversation workspace now uses the full viewport with selectable Transcript, First-pass summary, Context workbench, Speakers, and Final readout views; the transcript view loads source transcripts for readouts, parses text into speaker turns when structured utterances are unavailable, and renders a scrollable color-coded transcript frame. The re-transcription workflow includes dry-run preflight and reviewed queue-manifest actions, but queueing still does not start a speech backend or write transcript outputs.

Plan 0010/M1 is closed as the first dogfoodable conversation review loop. The workspace now exposes selected-conversation first-pass summary prepare/submit/status actions, durable speaker/contact review backed by SQLite contact and assignment tables, context workbench provenance with included/excluded sources and warnings, no-write deposition/memory preview queueing, and Review Queue rows that link back to the relevant conversation workflow tab. Plan 0012/M2 is also closed: `participant_identity.py` builds a participant identity bundle from calendar attendee evidence, readout participants, local reviewed contacts, configured user-scoped `gws` People/Contacts provenance, configured Odollo contact provenance, and operator decisions. The identity bundle is exposed through the conversation API, context-workbench previews, selected first-pass summary request artifacts, contextual reread artifacts, and AuraCall/Extended Pro ChatGPT prompt payloads. The React workspace shows calendar evidence, source-profile chips, contact candidates, manual contact entry, and final-preview blocking while identity/context warnings remain. `scripts/smoke_conversation_review_loop_ui.py` validates the M2 path against local runtime state with persisted JSON/screenshot evidence. Plan 0013 is closed: `provenance_config.py` now centralizes user-scoped `gog`, `gws`, `msgcli`, `odollo`, and iCalendar source definitions for CLI calendar lookup, watcher jobs, participant identity, route/context provenance, transcript API endpoints, and the React Provenance tab. Plan 0014 is closed: the contact-search workbench is cache-first by default, already-fetched contacts select instantly without backend calls, selected contacts stay visible while searching, explicit configured-source refresh populates user-scoped cache/job records, recency/frequency relationship affinity ranks broad searches with visible reasons, merge/split decisions are reviewed and persisted locally, and App Intelligence uses the same local batch contracts as the operator. Plan 0015 is closed: selected-conversation initial summary now has a one-click reviewed run action, `automation_config.py` centralizes user-scoped workflow-stage automation policy, automation preview/apply APIs never run stages, and the React Settings tab exposes account/runtime status, intelligence route summaries, and automation toggles with every stage disabled/manual by default. Plan 0016 is closed as the design authority for turning Settings into a full configuration workbench before implementation code changes. Plan 0017 is closed: Settings is now a config workbench with Account, Intelligence, Automation, Provenance, Safety, and Evidence sections; browser proof shows local edits stay local until Preview/Apply. Plan 0018 is closed: the root Library landing page now has workflow-only primary nav, account-chip settings/admin access, Library-scoped search and kind controls, collapsed filters, and desktop/mobile browser proof that the main Library work surface appears above the fold. Plan 0019 is closed: the First-pass summary tab now presents one workflow-prep card with a single primary next action, while prepare/submit/check remain under advanced controls and the conversation review smoke guards the one-click surface. Plan 0020 is closed: intelligence settings now define named provider/model profiles once, components select those profiles, profile-only preview/apply is supported without task-route writes, and low-action Settings status/config facts are compact or hidden behind disclosures. Plan 0021 is closed: Settings no longer renders Library/test-status diagnostics, API preview/offline status blocks, latest-smoke text, or no-op staged-edit chrome when there is nothing to act on. P09 remains open for broader console productization, share/auth surfaces, richer deposition controls, and eventual external apply contracts.

Plans:

- `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md`
- `docs/dev/plans/0010-2026-05-23-dogfoodable-conversation-review-loop.md`
- `docs/dev/plans/0012-2026-05-23-speaker-deanonymization-context-workbench.md`
- `docs/dev/plans/0013-2026-05-24-user-scoped-provenance-config.md`
- `docs/dev/plans/0014-2026-05-24-contact-search-workbench.md`
- `docs/dev/plans/0015-2026-05-25-one-click-initial-summary-automation-settings.md`
- `docs/dev/plans/0016-2026-05-25-config-panel-design-path.md`
- `docs/dev/plans/0017-2026-05-25-settings-config-workbench.md`
- `docs/dev/plans/0018-2026-05-25-landing-page-navigation-redesign.md`
- `docs/dev/plans/0019-2026-05-25-one-click-summary-workflow-prep-polish.md`
- `docs/dev/plans/0020-2026-05-25-intelligence-profile-settings-redesign.md`
- `docs/dev/plans/0021-2026-05-25-settings-screen-chrome-cleanup.md`

Milestone Focus:

- Closed milestone M1 is the dogfoodable conversation review loop: a single
  selected conversation can move through source-audio verification, transcript
  review, scoped first-pass summary actions, speaker/contact review, context
  workbench, contextual readout inspection, and deposition/memory preview review
  without unattended external writes.
- P09 remains the UI/backend lane, but M1 deliberately binds P04 contextual
  routing, P05 deposition/memory preview, and P06 service visibility into the
  operator workflow instead of continuing isolated UI increments.
- Closed milestone M2 is speaker deanonymization and the participant-aware
  context workbench: configured user-scoped `gws` People/Contacts and Odollo
  tenant contacts provide contact provenance, calendar invite attendees and
  matching-calendar participants provide deterministic matching evidence,
  operator input can resolve missing identities, and the resulting
  participant/context bundle feeds high-powered readout providers before
  deposition work.
- Next P09/P05 work should dogfood the configured identity sources over more
  recordings, tune contact-source quality, and keep external deposition apply
  gated until identity/context warnings have a reviewed resolution path.
- Plan 0013 moved provenance configuration into the shared user-scoped profile.
  Next P09 work can focus on workflow ergonomics and deeper source-specific
  readiness controls rather than inventing parallel config stores.
- Plan 0014 closed the conversation-scoped contact search workbench: cached
  search remains instant, provider refresh is explicit, relationship-affinity
  ranking uses local/calendar/contact history, reviewed merge/split policy is
  durable, and operator/App Intelligence actions share the same local batch
  contracts.
- Plan 0020 closed the intelligence settings correction: profile definitions
  are the provider/model source of truth, components select profiles, and
  status/config paths stay compact or hidden unless the operator opens them.
- Plan 0021 closed the Settings chrome cleanup: Library diagnostics, API
  preview/offline labels, latest-smoke text, and no-op staged-edit bars are not
  rendered on Settings when they are not actionable.
- Plan 0015 closed summary-workflow ergonomics and automation settings:
  first-pass summary has one primary reviewed action, while production
  automation toggles stay user-scoped and disabled/manual until each stage is
  validated.
- Plan 0016 closed the design-first contract for the configuration panel:
  aesthetics, ergonomics, provenance/intelligence/automation sectioning,
  staged preview/apply behavior, and required `agent-browser` inspection must
  be settled before implementation.
- Plan 0017 closed the Settings config workbench implementation: Account,
  Intelligence, Automation, Provenance, Safety, and Evidence are in one Settings
  surface, and `agent-browser` proved routine local edits do not call backend
  endpoints before explicit Preview/Apply.
- Plan 0018 closed the root Library landing-page redesign: workflow
  destinations stay primary, search is now a Library/workbench control, filters
  are collapsed by default, and settings/admin surfaces live behind the
  upper-right account avatar chip.
- Plan 0019 closed the selected-conversation summary-stage polish: first-pass
  summary prep now has one primary next action, and lower-level
  prepare/submit/check operations are advanced controls rather than peer
  workflow actions.

Definition of Done:

- React + Vite frontend shell exists with navbar, animated left pane, central viewport, and right inspector pane.
- Backend read APIs expose library/search/readout/contact/provenance state without leaking secrets.
- Source recordings are playable from stored blobs through DB pointers with seek/range support.
- Login and scoped artifact sharing reuse the `previews` design contract.
- Human review workflows cover speaker/contact assignment, context acquisition, deposition, and memory harvest.
- Intelligence management includes `codex app-server` as the supervised App Intelligence control plane for persistent, branchable, replayable agent runs, while `codex exec` remains the stateless leaf-job path.
