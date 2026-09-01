# Roadmap

`ROADMAP.md` is the master plan for this repo. Bounded execution plans live under `docs/dev/plans/`; turn-by-turn history lives in `RUNBOOK.md`.

## Product north star

[VISION.md](VISION.md) defines the product outcome that this roadmap exists to
deliver: trustworthy contextual readouts that feed a private, growing body of
conversation knowledge for future transcripts and authorized agents.

Every new or revised roadmap lane and bounded plan must name the vision
outcomes it advances, its current and target maturity levels, its measurable
effect, and its evidence gate. Closing an infrastructure or provider-readiness
lane does not by itself establish end-to-end progress unless the downstream
contextualization or knowledge-reuse outcome is also measured.

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

Current State: The watcher runs under systemd with heartbeat logging, no-progress restart behavior, startup readiness checks, and a `--check` doctor path. Readiness checks fail loudly for missing shared dependencies, configured backend scripts, configured readout scripts, or when every watch directory is unavailable. One temporarily unavailable watch root is job-local: readiness reports a warning, healthy jobs continue, and heartbeat logs include `blocked=unavailable_watch_dir=1`. `--check --check-json` exposes the same diagnostics for automation. Candidate state records `blocked_kind`, `blocked_reason`, and `blocked_since`, backend failures record `failure_kind` and `failure_reason`, successful runs that continue without calendar metadata record `warning_kind` and `warning_reason`, and heartbeat logs summarize blocked work as `blocked=kind=count`. The Voice Recordings job now uses `/mnt/d/SyncThing/Voice Recordings`; recursive scans exclude Syncthing's `.stversions` archive so incomplete historical versions do not create permanent false backlog.

Plans:

- `docs/dev/plans/0006-2026-05-04-service-reliability-observability.md`
- `docs/dev/plans/0023-2026-07-20-watcher-mount-resilience-calendar-recovery.md`
- `docs/dev/plans/0024-2026-07-20-voice-recordings-d-drive-cutover-catchup.md`

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

Plan 0010/M1 is closed as the first dogfoodable conversation review loop. The workspace now exposes selected-conversation first-pass summary prepare/submit/status actions, durable speaker/contact review backed by SQLite contact and assignment tables, context workbench provenance with included/excluded sources and warnings, no-write deposition/memory preview queueing, and Review Queue rows that link back to the relevant conversation workflow tab. Plan 0012/M2 is also closed: `participant_identity.py` builds a participant identity bundle from calendar attendee evidence, readout participants, local reviewed contacts, configured user-scoped `gws` People/Contacts provenance, configured Odollo contact provenance, and operator decisions. The identity bundle is exposed through the conversation API, context-workbench previews, selected first-pass summary request artifacts, contextual reread artifacts, and AuraCall/Extended Pro ChatGPT prompt payloads. The React workspace shows calendar evidence, source-profile chips, contact candidates, manual contact entry, and final-preview blocking while identity/context warnings remain. `scripts/smoke_conversation_review_loop_ui.py` validates the M2 path against local runtime state with persisted JSON/screenshot evidence. Plan 0013 is closed: `provenance_config.py` now centralizes user-scoped `gog`, `gws`, `msgcli`, `odollo`, and iCalendar source definitions for CLI calendar lookup, watcher jobs, participant identity, route/context provenance, transcript API endpoints, and the React Provenance tab. Plan 0014 is closed: the contact-search workbench is cache-first by default, already-fetched contacts select instantly without backend calls, selected contacts stay visible while searching, explicit configured-source refresh populates user-scoped cache/job records, recency/frequency relationship affinity ranks broad searches with visible reasons, merge/split decisions are reviewed and persisted locally, and App Intelligence uses the same local batch contracts as the operator. Plan 0015 is closed: selected-conversation initial summary now has a one-click reviewed run action, `automation_config.py` centralizes user-scoped workflow-stage automation policy, automation preview/apply APIs never run stages, and the React Settings tab exposes account/runtime status, intelligence route summaries, and automation toggles with every stage disabled/manual by default. Plan 0016 is closed as the design authority for turning Settings into a full configuration workbench before implementation code changes. Plan 0017 is closed: Settings is now a config workbench with Account, Intelligence, Automation, Provenance, Safety, and Validation sections; browser proof shows local edits stay local until Preview/Apply. Plan 0018 is closed: the root Library landing page now has workflow-only primary nav, account-chip settings/admin access, Library-scoped search and kind controls, collapsed filters, and desktop/mobile browser proof that the main Library work surface appears above the fold. Plan 0019 is closed: the First-pass summary tab now presents one workflow-prep card with a single primary next action, while prepare/submit/check remain under advanced controls and the conversation review smoke guards the one-click surface. Plan 0020 is closed: intelligence settings now define named provider/model profiles once, components select those profiles, profile-only preview/apply is supported without task-route writes, and low-action Settings status/config facts are compact or hidden behind disclosures. Plan 0021 is closed: Settings no longer renders Library/test-status diagnostics, API preview/offline status blocks, latest-smoke text, or no-op staged-edit chrome when there is nothing to act on. Plan 0022 is closed: Settings now uses a dedicated full-width configuration workspace with transcript panes hidden, settings-only status, responsive section navigation, unclipped intelligence route tables, and validation/dogfood evidence copy in place of user-visible smoke terminology. AuraCall-backed first-pass summary preparation now reads the AuraCall agent choices contract, prefers stable `AURACALL_AGENT_ID` for single-agent runs, preserves `AURACALL_DISPATCH_TEAM` for dispatch-pool routing, writes redacted `auracall_readiness` into prepare/enqueue manifests, and exposes the same readiness plus redacted agent selector options through `/api/intelligence/config` without leaking API secrets. Settings > Intelligence now renders an AuraCall agent selector with a selected-agent settings description whenever the profile is AuraCall-backed or already uses an `agent:<id>` model. P09 remains open for broader console productization, share/auth surfaces, richer deposition controls, and eventual external apply contracts.

Plan 0072 A6-R1 is delivered as the bounded P09 contacts refinement. The
previously empty People surface is now a compact searchable Contacts directory
containing already-local canonical people, local contacts, and validated prior
speaker-review labels while keeping those record classes separate and
read-only. Provider refresh, contact/person reconciliation, speaker-assignment
apply, and other reviewed-learning effects remain outside this slice.

Plan 0072 A6-R2 is delivered as a bounded contact-ingestion refinement. It gathers
exact calendar-attendee emails across the current recording corpus, enriches
them through read-only exact-email queries to configured GWS and Odollo
sources, and writes replayable local contact candidates with recording
provenance. The target is Level 2 manual/shadow ingestion: shared or ambiguous
addresses stay review-required, and provider writes, person merges, speaker
assignments, background scheduling, and Level 3 automation remain gated.

Plan 0072 A6-R3 is delivered as bounded relationship-and-role discovery. The
Contacts read model now derives deterministic Level 2 shadow hypotheses for
provider-declared contextual roles, provider-declared organization
affiliations, and recurring calendar co-invitation. The live corpus currently
surfaces 3 role, 59 affiliation, and 279 recurring co-invitation hypotheses
across 103 contacts. Every lead is review-required; accepted graph events,
ontology changes, provider writes, person merges, speaker assignment, and
automatic identity conclusions remain gated.

Plan 0073 is the closed deterministic-mail successor. P0 is closed with four
versioned portable-artifact contracts, closed JSON Schemas, exact thresholds
and reason codes, and seven synthetic/redacted scenario fixtures. P1 is closed
with an injected operator-lite adapter that enforces exact-address queries,
scope echo, bounded cursor pagination, one transient retry, deterministic
receipts, and privacy-safe partial failures. P2 is closed with validated pure
normalization, cross-source deduplication, temporal exclusion, independence
grouping, conflict preservation, stable ordering, and replay-equal watermarks.
P3 is closed with deterministic proposed-only mail transmission,
correspondence, recurring coparticipation, contextual-role, and affiliation
hypotheses, exact exclusions, conflict visibility, and zero accepted effects.
P4 is closed with bounded `/api/people` summaries and a dense Contacts review
table whose hypotheses default to reverse-time order, expose visible SVG sort
controls, support pointer/keyboard column resizing, and expand inline into
privacy-safe evidence detail on desktop and mobile. The user's 2026-08-31
session-wide Mail Receipts read grant satisfied P5's private-read gate without
another approval ceremony. Plan 0073's implementation is closed; its earlier
measured no-advance decision is superseded by corrected corpus-serving
evidence:
its zero-read preview and literal approval contract is implemented, and the
current 25-item local review cohort resolves deterministically to 57 exact,
non-shared contact-email queries. Its approval-bound execution seam now proves
the full 25/57 shape against synthetic metadata, including one pilot-wide
retry, duplicate control, immutable private receipts, and provider-free offline
replay. The installed registration fixes namespace `default` and profile
`operator-lite`. Mail Receipts Plan 232 now returns four complete authenticated
public source/account/tenant/corpus selector candidates with zero blocked
reasons. The operator selected the documented default, and immutable preview
`plan0073-p5-139eea68bfb7e6929e4e22115458e35e` binds the exact source scope,
historical cutoffs, frozen budgets, and 25-conversation/57-query cohort with
content SHA-256
`00de9e575524e17ab4c853ec7f9594f875ebd3e26068df6adc0688d2bf152156`.
Mail Receipts Plan 236 installed checkpoint
`e5fb9b8dd2722e55941715cbc331a9a45544630c` now serves the retained pre-2024
and 2024 partitions as one 359,693-message archive member, hydrates aggregate
selected-result context, normalizes retained timestamps to UTC, and applies
historical bounds before ranking/page limiting on the two-member
archive-plus-live DuckDB path. Transcribe Audio now sends one bounded exact
participant query per frozen address. The unchanged cohort accounted for all
57 queries in a full repaired-source P5 rerun, selected 966 records, retained
951 normalized observations and independence groups, and produced 120
proposed-only hypotheses: 70 sent-mail, 47 thread-coparticipation, and 3
correspondence hypotheses. Provider-free replay was equal and all
provider/mailbox/graph/speaker effects remained zero. The prior zero-record P6
measurement is preserved as superseded evidence of the repaired serving
defects, not current corpus coverage. The installed `/api/people` surface still
uses the earlier calendar/provider shadow projection and does not consume these
private P5 artifacts, so a bounded consumer/evaluation successor is required
before the recovered mail hypotheses appear in live Contacts. The mail lane
remains a Level 2 review-only shadow capability. The plan extends the
existing bounded evidence adapters and Plan 0072 shadow graph projection with
exact-email, directional, thread-aware observations from the installed
user-scoped Mail Receipts operator-lite workbench. Its target remains Level 2
review-only discovery: the current session grant covers bounded read-only
corpus access, message bodies stay out of the initial contract,
duplicate source copies count once, historical queries obey `as_of`, and
accepted graph facts, speaker effects, provider writes, and Graphiti projection
remain separately gated.

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
- `docs/dev/plans/0022-2026-05-25-settings-layout-refactor.md`
- `docs/dev/plans/0025-2026-07-21-app-intelligence-speaker-preprocessing.md`
- `docs/dev/plans/0026-2026-07-24-oldest-forward-speaker-identity-test-campaign.md`
- `docs/dev/plans/0027-2026-07-25-speaker-output-reference-repair.md`
- `docs/dev/plans/0028-2026-07-25-speaker-confidence-calibration.md`
- `docs/dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md`
- `docs/dev/plans/0030-2026-07-26-provider-adapters-and-blind-retrieval-evaluation.md`
- `docs/dev/plans/0035-2026-07-30-blind-combined-speaker-outcome-measurement.md`
- `docs/dev/plans/0036-2026-07-30-literal-fts-blind-speaker-rerun.md`
- `docs/dev/plans/0059-2026-08-08-speaker-identity-foundation-shadow-orchestration.md`
- `docs/dev/plans/0060-2026-08-08-complete-speaker-identity-shadow-join.md`
- `docs/dev/plans/0061-2026-08-08-plan-0060-human-gold-comparison.md`
- `docs/dev/plans/0062-2026-08-08-reconnect-contextual-speaker-identity-join.md`
- `docs/dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md`
- `docs/dev/plans/0073-2026-08-30-deterministic-mail-evidence-for-relationship-role-discovery.md`

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
- Closed milestone M3 is the P09/P10 speaker-identity foundation and shadow
  join in Plan 0059. It freezes the acoustic, context, canonical-person, and
  decision contracts; rehearses schema migration and reconciliation on a
  private live-database copy; and compares context-only, acoustic-only, and
  combined evidence in the selected-conversation review path. Note 0055 is the
  architecture authority. Plan 0059 is closed with terminal `refine`: A0/P0/P1
  completed, while bounded P2 execution exposed and hardened transcript-time,
  candidate-UUID, and provider-lineage adapter boundaries without completing
  the 3-recording/10-speaker denominator. Plan 0060 is the closed
  `review_ready` successor: independent P2A and P2B receipts cover all 3
  recordings/10 speakers, P3 froze 30 blinded abstentions, P4 sealed 10 empty
  decision slots with apply disabled, and P6 proved unchanged live state.
  Closed Plan 0061 supplied the literal 10/10 human-gold successor: 3
  canonical-person and 7 `not_listed` decisions yielded candidate recall 3/10,
  zero known-person recall for all three conditions, 7/7 appropriate
  abstentions per condition, zero wrong proposals, and terminal `refine` with
  no live mutation. Active Plan 0062 is the bounded integration successor.
  P1-P2 reused the Plan 0025 clue/calendar/retrieval workflow on all ten recent
  speaker slots, preserved seven named unlisted suggestions, and added the
  explicit contextual/canonical/acoustic join. P3 is frozen and replayed; one enrolled
  acoustic subject remains intentionally unbound to any prepared canonical
  person until human review. P4 is published with direct audio and actual
  contextual suggestions. Biometric enrollment remains a later, separate
  mutation authority. Plan 0059 authorizes no live
  schema migration, watcher enqueueing, assignment or relationship apply,
  provider write-back, profile learning, or automatic identity.
- Open milestone M4 is Plan 0072's correction-first identity-learning
  product path. Note 0058 freezes the grilled decisions: baseline read-only
  contact reconciliation; hierarchical multi-role relationships; scoped
  terminology and non-destructive transcript correction; private indefinite
  person-unbound sample retention with soft voice clustering; separately
  visible evidence pillars and measured likelihood; and distinct Authelia-
  protected Identity Review and People tabs that always show actual original
  recording filenames. Launch is staged as no-effect shadow, reviewed local
  learning, then policy-qualified automation. The first bounded private A6
  shadow is now active: 25 oldest eligible historical conversations were
  processed in an isolated schema-v8 snapshot and the seven-day
  stabilized-arrival window runs through `2026-08-23T23:33:20Z`. Live schema
  migration, biometric collection/enrollment, provider calls, local identity
  effects, publication, deployment, and background scheduling remain
  unstarted. A0 is
  closed at the non-live contract, privacy/threat-model, schema, review,
  adapter, supervisor, and redacted-fixture freeze. A1-A5 are now closed on
  synthetic/redacted stores: schema v8 adds the rebuildable review queue,
  append-only submissions/effect previews, stale-safe APIs, and separate
  original-filename-bearing Identity Review and People views. The explicit A6
  private-shadow checkpoint was granted for exactly this historical cohort and
  arrival window. All 25 historical cases are terminal and privately queued
  with zero provider calls or accepted effects; A6 remains open pending the
  full arrival window, terminal scorecard, and replay. Live migration and
  provider access remain gated.
- Closed Plan 0025 adds the missing post-transcription reasoning stage:
  a ledger-backed Codex app-server clue-discovery pass produces host-validated
  retrieval terms, bounded GWS/Odollo provenance feeds a separate identity
  evaluation pass, and host-owned rubrics derive auditable confidence.
  Split/merged diarization findings, inferred cross-source person grouping,
  durable conversation processing history, and lightweight-but-mandatory
  identity confirmation are in scope; full-conversation interpretation remains
  deferred.
- Open Plan 0026 turns that workflow into an oldest-forward calibration
  campaign. It freezes private operator-reviewed ground truth separately from
  blind model predictions, advances in chronological batches, preserves an
  untouched next-batch holdout, measures calendar/identity/diarization/evidence
  outcomes separately, and accepts only hypothesis-specific refinements that
  pass the accumulated gold regression set. C1-C6 are complete, including the
  first reviewed gold batch, immutable baseline, explicitly rejected/reverted
  refinement, and once-scored chronological holdout. The holdout passed host
  validation for only 2/10 predictions and retained three High/Very High wrong
  speaker proposals, so C7 is paused at chronological rank 24 rather than
  spending another review batch. Plan 0027's bounded reference repair was
  rejected as a complete identity-quality repair, while Plan 0028's confidence
  calibration was accepted. Plan 0026 remains paused while Plan 0030 completes
  the bounded provider-adapter and blind accumulated-context evaluation slice.
- Closed Plan 0027 preserves rejected App Intelligence output and permits at
  most one host-mediated corrective turn containing the invalid fields and
  exact prepared-reference allowlists. It does not weaken validation, remap
  invented IDs, retrieve new evidence, or broaden identity reasoning. Its
  repair improved validation yield but was rejected for promotion after
  High/Very High wrong identity proposals increased in both comparison
  cohorts.
- Closed Plan 0028 is the successor confidence-calibration gate. Plan 0027's
  repair materially improved validation yield but exposed unsafe
  High/Very High wrong identity proposals. Plan 0028 preserved those
  predictions and added a host-owned, reason-coded confidence cap for
  unresolved, conflicting, mixed, or materially unverified identities before
  automatic confirmation can be considered. Its immutable replay preserved
  17/53 top-person correctness while reducing High/Very High wrong proposals
  from 12 to 0; a future unseen chronological holdout remains required before
  automatic confirmation.
- Closed Plan 0029 established the durable conversation-knowledge storage and
  retrieval campaign. It evolved the existing user-scoped SQLite transcript
  store through sidecar-authoritative shadow projection, immutable
  observations and evidence bundles, temporal and tenant-aware hybrid
  retrieval, reproducible person/topic/relationship profiles, and a
  chronological comparison before either storage-authority cutover or
  automatic confirmation. The architecture authority is
  `docs/conversation-knowledge-storage-and-retrieval.md`, with the storage
  decision recorded in ADR 0002. C1 is complete in source with versioned
  additive schema migration, private backup, rollback, and deep
  conversation/person/processing-history interfaces. C2 is complete with
  hash-bound read-only preview, approval-gated idempotent shadow apply,
  immutable reconciliation receipts, legacy contact/assignment projection,
  and sidecar round-trip export. A private isolated preview reconciled all 3
  current Voice Recordings processing sidecars without migrating the live
  store. C3 is complete with bounded evidence snapshots, explicit
  source/account/tenant/capability/time isolation, exact and FTS5 lookup,
  bounded embedding search, typed concepts and mentions, immutable retrieval
  requests, content-hashed evidence bundles, and reason-coded
  inclusion/exclusion. Private live-database-copy migration and rollback
  rehearsals passed without changing the live store. C4 is complete with
  immutable reviewed and diarization outcomes, versioned source-affinity
  observations, deterministic current person and interaction/organization/
  project/topic/terminology/source profiles, same-name ambiguity preservation,
  supporting observation IDs, and build watermarks. Delete/rebuild and private
  live-source rehearsals pass without observation mutation or live migration.
  C5 is complete with an explicit host-owned `prepare_identity_evidence(...)`
  policy, exact-first candidate resolution, bounded lexical/semantic/
  relationship retrieval, support and contradiction features, independence
  and total packet budgets, scope-safe multi-database person grouping,
  immutable request/bundle replay, calendar/prepared fallback, and labeled
  partial-provider failure. C6 integrates immutable bundles into the existing
  exact-reference, factor-scoring, confidence, and human-review path. C7 froze
  ten unseen chronological cases. Closed Plan 0030 added production GWS/Odollo
  adapters, made scoped immutable bundles the default selected caller, gated
  legacy collection behind an operator receipt, and proved deterministic
  private shadow replay/restore/rollback without changing authority.
- Plan 0030 closed `refine` at J2 before gold or prediction. GWS authorization
  remained revoked and the repaired Odollo persistence path could not receive
  a forbidden third live attempt, so zero included provider snapshots were
  proven. The frozen cohort remains unconsumed with all predictions
  `not_started` and gold unread. Any successor must restore provider readiness,
  authorize a fresh bounded attempt packet, and prove included provider yield
  before provenance or combined measurement. Sidecars remain authoritative,
  the live database remains schema v0, and automatic confirmation remains
  disabled.
- `docs/dev/plans/0031-2026-07-29-provider-yield-retry.md` closed `refine`.
  Its GWS metadata probe succeeded, but its prior smoke target had zero
  deterministic query terms, so all adapters correctly refused retrieval
  without executing a provider query.
- `docs/dev/plans/0032-2026-07-29-target-qualified-provider-yield-retry.md`
  closed `pass`: its six-term immutable request included four normalized
  Odollo snapshots across both configured tenants, proving the general
  provider-yield prerequisite. GWS remained explicitly unavailable inside the
  service because the installed user-systemd PATH omitted its executable.
- `docs/dev/plans/0033-2026-07-29-gws-service-path-repair.md` is the bounded
  installed-runtime repair and closed `refine`. The PATH drop-in succeeded and
  GWS executed, but its first high-yield capability consumed the complete
  twenty-record adapter budget; all twenty snapshots were outside the
  historical scope and later GWS capabilities were starved.
- `docs/dev/plans/0034-2026-07-29-gws-capability-budget-fairness.md` closed
  `pass`. The test-first adapter-local repair preserves the public interface
  and global caps while retaining budget access for later configured
  capabilities. The final served immutable request included two normalized
  `gws-default` People controls and four Odollo controls. The installed service
  PATH repair is active, the 143-test joined suite passes, and the frozen
  cohort and authority states remain untouched. Any blind prediction or human
  gold-review campaign still requires separate explicit authorization.
- Open Plan 0035 is the authorized outcome-measurement successor. It binds the
  exact ten-case conversation-evaluation freeze to the existing blind
  App Intelligence runner, captures the current default combined path before
  any review, then requires independent post-prediction gold before reveal.
  Its current maturity is `2 — Shadow`; completion will provide the evidence
  to accept or reject advancement toward `3 — Operational`, not declare that
  level from provider or runtime readiness alone.
- Plan 0035 closed `refine` after four predictions. Two model-derived
  hyphenated query terms reached FTS5 without literal quoting and were
  interpreted as column expressions. The sole unchanged retry recovered the
  first failure; the second exhausted the bound. Six cases remain unstarted,
  gold remains absent, and no prediction was revealed.
- Open Plan 0036 owns one shared FTS literalization, an explicit superseding
  baseline for the same unseen cohort, and continuation through the
  independent-review and outcome-scoring gates. It permits no prompt,
  retrieval-policy, confidence, or candidate change. The pushed repair is
  served and its linked replacement baseline completed all ten blind
  predictions with zero infrastructure retries. Independent operator gold
  reached five of ten current reviews and is checkpointed at chronological
  rank 30. The operator paused further review while Plan 0037 develops
  acoustic preprocessing and biometric speaker evidence. Prediction bodies
  remain sealed until all ten reviews exist. Resumption must record whether
  the changed review method requires a successor evaluation.
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
- Plan 0022 closed the Settings layout refactor: Settings now uses a dedicated
  configuration workspace instead of the transcript three-pane shell, with
  responsive section navigation, unclipped intelligence route tables, and
  validation/dogfood evidence copy in place of user-visible smoke terminology.
- AuraCall choices integration now gives the Settings/Intelligence surface a
  runtime-backed agent selector with redacted settings descriptions for
  selected transcript agents or dispatch-pool teams while keeping transcript
  prompts, readout schemas, ledgers, and materialization in this repo.
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

## P10 | Acoustic Processing And Biometric Speaker Identity

State: OPEN — Plans 0058 and 0060 are closed; Plan 0059 closed `refine`

Current State: Plan 0037 is closed unsuccessfully with terminal `STOP` through
Plan 0048. P0 through P3 remain closed through Plans 0038 through 0041. P4
created six real profiles and nine held-out calibration thresholds, but the
successor evaluation revealed zero overlap between five evaluation subjects
and the two frozen profile subjects. Every candidate unit therefore has zero
possible genuine and impostor trials against the frozen 20/100 minima. P5
integration and P6 historical reprocessing are `not_run`; nothing is promoted.

Plan 0049 is a separate bounded training-data expansion over at most five
novel `Documents/Sound Recordings` conversations. It may add reviewed private
references and successor profiles, but it cannot rewrite the Plan 0048 STOP or
claim terminal evaluation sufficiency. Exact intake plus P1/P2 preparation are
complete for five conversations, with 25/25 required P2 method attempts
successful. The operator confirmed the private 14-label/40-clip packet. Two
successor reference generations now contain 10 windows across four sessions
and 15 windows across five sessions, respectively. Six successor profiles are
active across all three pinned models and their six predecessors are
superseded. The exact two-person/two-session/six-window training sufficiency
contract passes. Plan 0048's terminal evaluation STOP remains unchanged.

Plan 0050 opens a separate Generation-3 campaign. It requires a new sealed
cohort disjoint from both revealed evaluation generations and all Plan 0049
training sources, binds enrolled gold directly to the active P3 subject IDs,
recalibrates the six successor profiles before reveal, and adds the missing
exact-trial child and positive evaluation path. No Generation-3 reveal, model
score, metric, or terminal decision has run yet. Its exact cohort authority is
implemented and the current seven-conversation/28-label preview passes source,
recording, conversation, and derivative disjointness. Membership is now frozen
under an independently audited, clean pushed, live-replayed private
authority. Recalibration, reveal, preparation, and scoring remain not run. The
separate exact-gold preview/apply/replay authority is independently audited and pushed
at `43fcced`: it requires all 28 frozen labels, binds enrolled outcomes to
active P3 lineage, enforces explicit identity-token evidence, and emits an
aggregate-only portable projection. The exact gold packet passed independent
no-write reproduction, was frozen under
`generation3-gold-5f60fa794c40c8fa5a2c5cb0`, and replayed idempotently. Its
aggregate receipt authorizes only successor recalibration.
Evaluation reveal, preparation, windows, models, scores, metrics, and terminal
decision remain not run.

The successor recalibration pre-score authority is now implemented and passed
independent re-audit. Its live no-write preview binds the exact historical
22-window membership, the complete six-profile/two-subject/three-candidate
Cartesian inventory, all nine candidate-method units, and derived per-unit
44/9/35/26 total/genuine/impostor/open-set denominators. Four-dimensional
overlap with active training and Generation-3 is zero. The clean pushed
implementation is frozen under
`generation3-recalibration-99fcabf628404df4940f2be0` and replays
idempotently. It authorizes only calibration-model execution; no calibration
score has run in that pre-score authority and evaluation reveal remains
locked. A separate self-bound executor is pushed at `7d5b535`; it persisted
execution authority before model load, completed and structurally replayed the
exact 396-trial successor score matrix, and deterministically froze all nine
threshold/temperature pairs. Every unit has exact 44/9/35/26
total/genuine/impostor/open-set denominators, the abstention margin remains
zero, and the aggregate threshold receipt authorizes only pre-reveal-envelope
construction. Generation-3 reveal, preparation, windows, exact trials,
evaluation scores/metrics/decision, profile mutation, default integration, and
historical reprocessing remain not run.

The required independently audited pre-reveal envelope is also frozen. Clean
commit `da4acdc` reproduced preview hash
`c9db91fb9ed2d69055893ded7a9f987f641b3962364bef6de66f88061f968797`
without changing the runtime. Applied authority
`generation3-pre-reveal-2dac320b6577456bd38a281b` binds the exact cohort,
gold, population, six profiles, score matrix, nine threshold/temperature
pairs, five condition dimensions and algorithms, 12-window cap, preparation,
exact-trial, metric, minimum-evidence, and terminal policies. It authorizes
only the separate reveal step; no Generation-3 acoustic execution has begun.

The separately self-bound reveal authority is now frozen before private gold
access. Exact structural preflight revealed 10 enrolled, 10 open-set, and 8
excluded labels and passed every unit with conservative maxima 120 genuine,
120 known-impostor, and 240 open-set against 20/100/20. It authorizes only
prediction-blind P1/P2; all condition, window, trial, model, score, metric,
decision, mutation, integration, and reprocessing actions remain false.

Generation 3 then executed its single authorized prediction-blind preparation
attempt. Six P1/P2 units completed with 30/30 P2 method cells, but the seventh
P1 decode lost 89.776791 seconds relative to the frozen source duration,
exceeding the immutable 0.05-second tolerance. The independently audited
terminal packet at commit `944e554` replays full-body and records global
`STOP`. Windows, exact trials, evaluation models, scores, metrics, selection,
default integration, and historical reprocessing are `not_run`. A future
attempt requires a new evaluation generation and authority, not a Generation 3
repair or retry.

Plan 0051 opened the next bounded step under the durable speaker-identity
roadmap in
`docs/dev/notes/0051-2026-08-02-speaker-identity-product-roadmap.md`. It
qualifies a fresh, explicit Generation-4 media pool before cohort or gold
freeze, including full decoded-duration validation that would have rejected
the malformed Generation-3 source. It does not run biometrics or authorize
speaker identity.

Plan 0051 is now closed. Its pushed qualification authority froze 10 healthy,
unique, prior-disjoint recordings from 12 explicit candidates; two short files
were excluded. Full source re-decode replay passes with no retained audio.
Only a separate cohort-preview action is authorized. The next bounded slice
must determine conversation identity, enrolled-speaker coverage, and private
gold feasibility before freezing any Generation-4 cohort.

Plan 0052 is closed with immutable terminal `STOP`. Private review and the one
bounded supplemental pool produced a valid seven-recording cohort with nine
people, 17 same-person session pairs, both enrolled people represented twice,
complete gold, zero overlap, and exact original-plus-supplemental authority.
The first independent J1 review found caller-injected source authority; the one
allowed rework pinned both manifests, exact set hashes, disjoint union, and
per-member origin, after which J1 signed acceptance. G2 froze the exact cohort,
private-gold commitment, selected acoustic factor, nine-unit contract,
contextual prompt/rubric, metrics, negative actions, and terminal policy.

G3 completed all five P2 methods on three cases. The fourth P1 decode drifted
`0.1739795` seconds from its frozen source duration, exceeding the immutable
`0.05`-second tolerance. Post-freeze substitution and recipe relaxation were
forbidden, so the remaining three cases were not attempted and the first
terminal-policy rule recorded `STOP`. Gold was never revealed to prediction
workers; contextual predictions, biometric execution, acoustic scores,
profile mutation, integration, and historical reprocessing are `not_run`.
Acoustic and combined voice-context maturity therefore remain at their prior
levels. Any successor evaluation requires fresh authority rather than a Plan
0052 repair or retry.

Plan 0053 opens that fresh authority in
`docs/dev/plans/0053-2026-08-03-generation-5-duration-validation-and-blind-evaluation.md`.
It first separates known duration failures and healthy controls from a sealed
diagnostic holdout, then requires a sample-preservation rule justified by
audio timing and decode semantics rather than a larger case-fitted tolerance.
Only after independent holdout acceptance may it enumerate a fresh,
diagnostic-disjoint evaluation cohort and run the complete blind context-only
versus separately voice-augmented comparison. The lane remains open until one
immutable Generation-5 terminal decision; a decoder fix, passing holdout,
cohort freeze, or model score is not closure.

Plan 0053 is now closed with terminal `STOP` at J2. Its J1-accepted
sample-preservation rule passed all seven positive holdouts and ten fixed
negative cases, but the corrupt-tail negative copied its observed exception
reason into the expected field. That circular assertion invalidated the 11/11
negative denominator. Candidate enumeration, gold, models, predictions,
mutation, integration, and reprocessing never ran. Plan 0054 opens a fresh
successor with a predeclared corrupt-tail reason, a newly selected disjoint
positive holdout, and the same end-to-end paired evaluation milestone in
`docs/dev/plans/0054-2026-08-03-generation-5-fresh-holdout-recovery-and-blind-evaluation.md`.
Plan 0054 recovered the duration-validation evidence: an independently
accepted fresh run passed 7/7 positive recordings and 11/11 predeclared
negative controls. E1 then enumerated 12 further fresh recordings, rejected
one with no usable speech, and materialized 29 private listening cards across
the remaining 11. The operator completed all 29 identities, but exhaustive
selection of all 330 seven-recording combinations proved population
infeasibility: only one enrolled person had the required two-recording
coverage. Plan 0054 is closed at an immutable E1 stop. No evaluation model or
prediction ran, and no cohort or gold was frozen. A successor may reopen the
paired-evaluation milestone only after freezing a newly authorized,
prior-disjoint source expansion that can supply the missing second recording;
it must not hand-substitute into Plan 0054's revealed candidate pool.
Plan 0055 is that successor. It predeclares two hash-bound,
operator-identified, prior-disjoint recordings as required population strata,
including one Zoom audio artifact accessed through the existing
bastion-mounted SyncThing route. It will freeze a bounded archive expansion
before transcription, require private listening confirmation and complete
gold, and then run the original seven-recording context-only versus
voice-augmented comparison. Plan 0055 S0 is independently accepted with two
required plus ten additional prior-disjoint recordings, and S1 has produced a
private 40-card listening surface from source-hash-bound diarized transcripts.
The 40-card review completed with 39 operator labels plus one transcript-
context-derived Mark Mba-Wright identity. Independent J1 accepted and froze
the first passing cohort, Required A/B plus Candidates 3–7. E2 completed a
22-speaker paired comparison with nine full acoustic matrices and 396 trials.
After exactly one scoring-custodian reveal, context-only produced 0/22 correct
assignments while voice augmentation produced 6/22, including 6/9 enrolled
appearances, with no wrong or high-confidence wrong augmented assignment.
Independent J2 returned PASS and Plan 0055 is closed at terminal decision
`advance_to_limited_pilot_plan`. The next P10 slice is a new bounded pilot plan;
the decision does not authorize automatic assignment, profile learning,
production integration, or historical reprocessing.
Canonical closure:
`docs/dev/plans/0055-2026-08-04-generation-5-source-expanded-blind-evaluation.md`.

Plan 0056 is closed in
`docs/dev/plans/0056-2026-08-05-enrolled-only-acoustic-pilot-identity-guard.md`.
Its two-speaker shadow run produced one human-confirmed correct enrolled
assignment disposition and one human-rejected enrolled-person proposal, with
zero wrong or high-confidence wrong assignments. Independent audit reported
enrolled recall `1.0`, proposal precision `0.5`, and zero identity creation or
profile/reference mutation. Frozen before, after, and current identity state
are identical. The terminal decision is
`plan_next_bounded_integration_milestone`; it does not authorize automatic
assignment, profile learning, production integration, provider write-back, or
historical reprocessing. The broader canonical-person, provider-contact, role,
evidence-backed relationship-graph, App Intelligence inference, and bounded
multi-hop retrieval contract is memorialized in
`docs/dev/notes/0052-2026-08-05-contact-role-relationship-sequencing.md` and
remains deferred to the P09 conversation-knowledge path rather than becoming a
prerequisite for this enrolled-only non-mutating pilot. A separate plan is
required before the next P10 integration step begins.

Plan 0057 closed that separate bounded integration step in
`docs/dev/plans/0057-2026-08-06-enrolled-only-acoustic-shadow-review-integration.md`.
It attached validated, non-authoritative acoustic subject-ID evidence to the
ordinary transcript identity-review payload and measured one exact
three-recording fresh batch across at least two meeting contexts. Complete
private review found two enrolled Eric speakers and 13 neither-enrolled
speakers. Both proposals and all 13 abstentions were correct; enrolled recall
and proposal precision were `1.0`, with zero wrong/high-confidence-wrong
dispositions and unchanged identity state. The terminal decision is
`plan_next_bounded_milestone`. Missing decision-entry controls and operator-
reported unreliable audio playback remain review-surface follow-up findings;
they did not change the complete denominator. All speaker assignments,
identity/contact/relationship records, profiles, references, provider writes,
default integration, and historical reprocessing remained false. A separate
plan is required before any next P10 milestone.

Plan 0058 is the bounded review-surface reliability successor in
`docs/dev/plans/0058-2026-08-07-review-surface-reliability.md`. Fresh browser
diagnosis found that valid, byte-identical WAVs can receive intermittent public
ingress 502 responses when the generated page eagerly preloads all 15 media
files; Chromium reports those responses as format errors and disables the
affected controls. The plan adds strict per-card decisions, exact importer
export, on-demand audio, a direct-file fallback, and a non-sensitive 15-card
public-browser proof. Plan 0058 is now closed `complete`: the synthetic surface
rendered 15/15 decision groups and fallbacks, refused incomplete export,
round-tripped a complete 15-line block through the existing strict parser, and
loaded, sought, and played all 15 media files serially through public ingress
with 15 HTTP 200 Media responses, zero failures, and zero 502 responses. It did
not run a fresh acoustic cohort, modify Previews, or authorize any identity,
assignment, profile, provider, integration, or historical mutation. A separate
bounded plan is required for any next P10 milestone.

Plan 0059 was the cross-lane successor recorded in
`docs/dev/plans/0059-2026-08-08-speaker-identity-foundation-shadow-orchestration.md`.
It completed A0/P0/P1 but closed `refine` after bounded P2 adapter attempts
reached their limit. One recording/four speakers produced deterministic
acoustic evidence twice; the full 3-recording/10-speaker denominator and P3+
join/review/comparison work were not run. A new bounded successor plan is
required before further execution.
Plan 0060 is that closed `review_ready` successor in
`docs/dev/plans/0060-2026-08-08-complete-speaker-identity-shadow-join.md`.
It kept the exact inherited cohort and private-copy boundary, completed the
hardened acoustic and context lanes independently, froze 30 blinded
abstentions, and prepared a sealed 10-speaker human-review packet with no
preselection or apply path. Its terminal audit preserved exact live counters,
identity state, service continuity, privacy modes, and zero mutations. P5 is
not started because it requires 10/10 literal human decisions.
It moved the canonical-person store, bounded context collection, acoustic
evidence, and review join toward one Level 2 selected-conversation shadow
workflow. It deliberately stops before live schema migration, background
watcher integration, assignment or relationship apply, profile learning,
provider write-back, or automatic confirmation. The durable pillar contract is
`docs/dev/notes/0055-2026-08-08-speaker-identity-pillar-integration-architecture.md`.
Its A0 checkpoint froze the current runtime, exact three-recording/ten-speaker
chronological cohort, provider and acoustic permissions, private boundaries,
human gates, and all-false live mutation contract before implementation.

Plan 0061 is the closed human-gold/comparison successor in
`docs/dev/plans/0061-2026-08-08-plan-0060-human-gold-comparison.md`. It leaves
Plan 0060 immutable, reuses its exact sealed 3-recording/10-speaker/30-condition
packet, and completed strict 10/10 human decision capture plus independent
three-condition measurement. It closed `refine`: candidate recall was 3/10,
all three conditions abstained on all ten slots, and the two visible people
came from a compatibility snapshot rather than calendar or speaker-specific
context. Direct-audio Previews review worked remotely and all forbidden
mutations remained zero.

Plan 0062 is the closed `advance` repair in
`docs/dev/plans/0062-2026-08-08-reconnect-contextual-speaker-identity-join.md`.
It preserves Plan 0036's seal and the Plan 0060/0061 receipts while reconnecting
the existing two-phase contextual identity workflow to the newer canonical and
acoustic contracts on the exact three recent conversations. P1-P2 completed
with 3 recordings, 10 speakers, 30 joined evaluations, and seven unlisted
suggestion records. P3 and P4 completed with zero live mutations. Literal P5
review identified nine of ten slots, confirmed or corrected five contextual
suggestions, linked one contextual identity to an enrolled voice subject, and
retained one role-only unresolved slot. A later operator correction establishes
that the Dr. Stefl recording had no calendar appointment: the name came from
listening review, and Michael Forrester was not present. The three frozen
conditions made no proposals because
no reviewed names were in the canonical candidate authority, but also made no
wrong proposals. P5 closed `advance` with one existing-voice binding candidate
and eight enrollment-candidate appearances across six distinct reviewed names.
Assignment apply, canonical-person creation, calendar-title repair, candidate
deduplication, and biometric enrollment require a separate later plan and
authority.

Plan 0063 is the closed P09/P10 reviewed-learning successor in
`docs/dev/plans/0063-2026-08-09-reviewed-speaker-canonicalization-enrollment.md`.
Its non-applying P1-P4 implementation now turns the exact Plan 0062 human gold
into six deterministic provisional-person proposals, three literal merge
choices, one preserved enrolled-voice/context binding, and five new enrollment
candidates backed by 26 replay-valid source windows. Available calendar title,
description, attendee, and secondary-calendar evidence is now independently
citable and fail-closed; the Dr. Stefl case is an explicit no-calendar negative
control, not an uncaptured calendar-source case. The original combined
direct-WAV review at `bf53f4bf...`, published through authenticated Previews
session `cb529a51053b`, is superseded because its notice claimed calendar
provenance the recording did not have. Corrected v2 review `486dce68...` is
published as Previews session `b18f14803bd7`, artifact `e7ffdb4c90a5`; browser
proof covers all 26 players, blank/incomplete export refusal, a working 30-row
hash-bound export, clipboard fallback, and zero POST requests. Operator review
then found its merge decisions unanswerable: the compared slots had no paired
audio, and the Michael cross-recording card appeared attached to the separate
Dr. Stefl correction. That v2 submission is now superseded. The v3 replacement
requires two labeled, hash-bound Plan 0062 audio samples inside every grouping
card, six comparison clips total, direct-WAV fallbacks, and an explicitly
separate no-answer-required Recording 2 notice. The repaired v3 content is
`d782cd7d...`, published as authenticated Previews session `4ac17bd09f2f`,
artifact `93bf4884a21e`; strict replay verifies 32 clips, 30 blank decisions,
private modes, and zero mutations, while remote ingress serves 32/32 unique
WAVs with HTTP 200. This review-surface repair advances the Level 2
replayable human canonicalization target without changing the 30-decision
denominator or authorizing application. The deterministic post-review transition plus
conversation-knowledge and biometric private-copy rehearsal are now
implemented: reviewed merges, slot bindings, voice/context agreement, source
choices, governed store apply, reference generation, three-model profile
materialization, lifecycle rollback, table reconciliation, exact byte restore,
and unchanged-live checks are covered by six transition tests, 105 combined
acoustic/rehearsal tests, and the then-current 984-test full suite. A read-only integration
check validates all 26 frozen P3 windows across five proposed people and three
exact P1 lineages against the production schema and audio bounds. The exact
one-shot production-mode rehearsal now replays transition `75166646...` and
receipt `7fe33287...` after all 30 operator decisions, with 23 included and
three excluded source windows, one provider-backed full-name correction, exact
private-copy rollback, and zero live mutations. The A1 request and
literal-authorization contract is also implemented: it requires a
production-mode rehearsal, binds current live-store and clean committed-code
snapshots, exports one exact five-line approval block, rejects any drift, and
freezes authority without applying it. The A1-gated one-shot live driver is now
also implemented: it quiesces only the two transcript services, takes exact
byte backups, uses the governed knowledge/reference/profile write paths,
reconciles the authorized counts, restores services, and becomes terminal on
either success or proven exact rollback. Four disposable-store driver tests,
114 combined acoustic/rehearsal/authority tests, and the 993-test full suite
pass. The rehearsal and terminal driver now share the exact governed knowledge
and biometric apply helpers, eliminating the remaining indirect-coverage gap;
10 focused tests and the unchanged 993-test full suite pass after that
consolidation. The completed A0 work remained non-applying until the operator's
explicit `okay go` authorized the separate exact A1 transition. Terminal
receipt `259ea605...` records one completed apply,
zero rollback, zero unauthorized effects, six canonical people, nine slot
bindings, one voice/person binding, five references, fifteen profiles, and
twenty-three enrollment sources. Both transcript services returned to
active/running with zero restarts, and terminal replay is idempotent.

Plan 0064 is the closed fail-safe P09/P10 implementation campaign in
`docs/dev/plans/0064-2026-08-09-context-assisted-automatic-speaker-recognition.md`.
Its durable architecture decision is Note 0056. It makes the newly installed
voice/person learning state part of the existing contextual workflow: new
speaker slots receive independent acoustic and context evidence, active voice
subjects resolve through reviewed canonical-person bindings, and a
conversation-level solver may assign a remaining speaker only when one
independently supported candidate remains without contradiction. The plan then
measures that behavior on source-disjoint chronological recordings before
enabling one policy-qualified local acceptance band. Accepted observations
enrich canonical/contact provenance for later retrieval; external Google,
Odollo, or other provider writes remain zero while the plan prepares a
field-owned provider-enrichment proposal contract.

Plan 0064 P0 is terminal on private content
`f24722166f5f147ee6b26b13bba87d1f12ab60530c3ca0add3d8687046c5675a`:
the complete seven-reference/twenty-one-profile inventory and twelve-recording
oldest-forward source-disjoint cohort replay idempotently with zero identity,
knowledge, provider, or external effects. P1 dynamic acoustic evidence
`b6a87465...` and receipt `d0e2441a...` cover all 39 slots and 819 trials. P2
receipt `50a7f4fd...` preserves the same denominator with four completed
context workflows, eight reason-coded unavailable cases, and an exact replay.
The last four cases are unavailable because both configured model routes are
externally exhausted. All effect counters remain zero. P3 conversation-level
resolution is terminal at receipt `b630d12d...`: acoustic-only has 12
candidates, while combined and residual-policy remain fully review/abstain
because context produced no canonical candidate. P4 review v2 receipt
`22bbfd4e...` binds all 39 private direct-audio decisions, the 12-source
original-recording filename set `97eeb3ed...`, and is browser verified on
desktop and mobile. Each review card exposes only the hash-bound source
basename; the strict decision export remains unchanged. The operator supplied
all 39 literal human-gold decisions. An exact filename-only authority bridge
preserved every decision while rebinding the submission to review v2.
Source-disjoint measurement produced 12 acoustic candidates: 11 correct and
one high-support wrong; context, combined, and residual-policy produced zero
candidates. The exact three-conversation reviewed development replay
produced three correct pillar-agreement acceptances, zero high-support wrong
identities, and zero acceptances through the specific residual rule, so its
non-vacuous gate also failed. Terminal P4 content `f178f418...` records
`withhold_p5`; Plan 0064 is closed without reaching its product acceptance or
target maturity movement. P5/P6 remain withheld and all action counters remain
zero. A successor must use this gold only as development/hindsight evidence
and validate any correction on a fresh source-disjoint cohort.

Plan 0065 is the closed, no-apply successor in
`docs/dev/plans/0065-2026-08-11-speaker-identity-recovery-fresh-validation.md`.
Its acoustic development correction retained 10 of 11 prior correct candidates
while demoting the prior wrong candidate, but D2 produced zero correct prepared
context candidates across 39 terminal slot dispositions. The mandatory
non-vacuous context gate failed `context_recovery_failed`, so D3 and the fresh
E0-E3 cohort never opened and no filename-bearing review publication was
created. Terminal `e73e2ebc...` closes `withhold`. A bounded reconciliation
restored three lazily identity-backfilled transcript containers across five
copies and three index rows to their frozen Plan 0064 bytes; zero lasting local
container mutation and zero identity, knowledge, Graphiti, provider-write, or
external effect remain. The exposed Plan 0064 development sources remain
permanently excluded from any future unseen evaluation.

Plan 0066 is the closed, no-apply recovery in
`docs/dev/plans/0066-2026-08-11-nonmutating-context-candidate-recovery.md`.
It resolved both independently measured Plan 0065 preparation blockers:
retrieval mode now leaves legacy transcript containers and index rows exact,
and host retrieval projects the complete six-person reviewed roster with
reviewed primary names into each private shadow packet. Its one bounded
development pass consumed six primary turns with zero fallback and zero retry.
All six readouts cited calendar evidence IDs outside the frozen second-pass
validator authority, so strict validation rejected them before persistence. A
later audit proved all seven rejected citation occurrences were valid
host-prepared first-pass calendar IDs omitted from the second-pass evidence
catalog; they were not invented by the model. Terminal
`c5a843f8...` closes `withhold` with reason
`evidence_reference_compliance_failed`: zero correct, zero wrong, six
validation failures, and no joined/residual or fresh evaluation gate. All
source transcripts, live index rows, identity/biometric state, accepted
knowledge, Graphiti, provider records, and external systems remained
non-mutating.

Plan 0067 is the closed, zero-effect reconciliation attempt in
`docs/dev/plans/0067-2026-08-11-calendar-evidence-contract-reconciliation.md`.
Its three bounded A0 attempts failed closed before artifact creation because
the freeze imposed new-artifact `0600` mode policy on inherited read-only
transcript and model-status files and initially byte-equated A1 with the
documented A2 provenance transformation. Terminal `8c11ed7c...` closes
`withhold`; the product contract and retained outputs were untouched and every
effect count is zero.

Plan 0068 is the closed, zero-model-turn successor in
`docs/dev/plans/0068-2026-08-11-legacy-bound-calendar-evidence-replay.md`.
It froze six exact inherited cases with all six original recording filenames,
repaired the first-pass/second-pass calendar catalog boundary, preserved
calendar evidence as candidate-only, and replayed all six outputs without a
model call or output edit. Three outputs validated and three independently
failed on empty retained `utterance_id` assignments. Measurement was one
correct, zero wrong, 21 abstained, and three validation failures. Terminal
`07d6bda4...` closes `withhold` with reason
`retained_output_schema_compliance_failed`. The product contract is repaired,
but Level 2 context-candidate acceptance did not pass; no joined/residual,
fresh evaluation, live acceptance, knowledge, Graphiti, provider-write, or
external gate opened.

Plan 0069 is the closed, zero-model-turn successor in
`docs/dev/plans/0069-2026-08-11-grouped-utterance-assignment-normalization.md`.
It refines the Plan 0068 failure diagnosis: the three rejected outputs contain
10 substantive grouped `utterance_ids` objects covering 28 prepared
utterances, not empty semantic assignments. The bounded packet may only expand
each unambiguous grouped object into otherwise-identical singular
`utterance_id` objects. It retains all six original recording filenames and
keeps model, retrieval, fresh-evaluation, live-apply, knowledge, Graphiti,
provider-write, and external effect budgets at zero.
Its exact normalization validated all six retained outputs while preserving
all six original recording filenames. Frozen human-gold measurement reports
five correct candidates, zero wrong, 17 abstained, zero incomplete provenance,
zero unavailable, and zero validation failures. Terminal `2d53d01f...` closes
PASS; no joined/residual, fresh-evaluation, live-apply, knowledge, Graphiti,
provider-write, or external gate opened.

Plan 0070 is the closed fail-safe shadow successor in
`docs/dev/plans/0070-2026-08-11-joined-residual-and-fresh-shadow-validation.md`.
Its verified starting counterfactual combines Plan 0065's corrected acoustic
policy with Plan 0069 context evidence across 12 reviewed recordings/39 slots:
five correct pillar agreements, zero combined wrong identities, and no actual
residual acceptance. Both bounded D0 attempts failed before artifact creation
because the authority harness first expected a nonexistent source constant and
then treated list-valued exposure fields as integer counts. Terminal
`35cf5ae9...` closes `withhold`; no supplemental development or fresh
evaluation opened, and live assignment, knowledge, Graphiti, provider-write,
and external effects remained zero.

Plan 0071 is the closed corrected shadow successor in
`docs/dev/plans/0071-2026-08-11-corrected-joined-residual-shadow-validation.md`.
It inherits the unchanged 12-recording/39-slot counterfactual and the exact
Plan 0070 terminal. Its D0 reads the frozen Plan 0065 policy hash from the
artifact and validates exact lengths for all seven list-valued exposure fields
before writing anything. It must reproduce the joined gate and prove an actual
residual acceptance on a bounded supplemental development cohort before one
fresh source-disjoint evaluation may open. Original recording filenames remain
mandatory; live assignment, knowledge, Graphiti, provider-write, and external
effects remain closed.
Corrected D0 passed on its first attempt: activation `27b011dc...` binds all
seven exact exposure collections, 12 inherited Plan 0065 cases, six Plan 0069
cases, six original recording filenames, and the Plan 0070 terminal with zero
model turns and zero effects. D1 is complete: resolution `a235ddc...` and
measurement `32b75982...` preserve all 12 actual `.m4a` filenames and
reproduce acoustic 10/0, context 5/1, combined 5/0, and residual 5/0 candidate
correctness. The context-only wrong proposal remains unaccepted and no
residual acceptance exists, so D2 supplemental development is active while
fresh evaluation stays closed.
D2 cohort manifest `ea37ea38...` freezes six next-oldest source-disjoint
three-speaker recordings/18 slots with six actual original `.m4a` filenames.
Its two bounded prediction attempts are exhausted. The final receipt
`8de26c83...`, manifest `71fc5685...`, and resolution `bf1876e0...` replay an
acoustic result of seven review candidates/11 abstentions, 18 unavailable
context outcomes, and therefore seven joined/residual review candidates/11
abstentions with zero residual acceptances. Human gold remained unread and all
effect counts stayed zero. Authenticated private review session
`6b97d7a8da32` now contains 18 direct-audio decisions, displays the original
filename on every card, hides predictions, and disables incomplete export.
D3 accepted all 18 literal decisions under exact review authority and froze
human gold `8150c029...`, measurement `14bec969...`, and terminal
`c4cd3679...`. Acoustic measured six correct, zero wrong, one unverifiable,
and 11 abstained; context was unavailable for all 18 slots. Combined and
residual-policy each had zero candidates, seven review outcomes, 11
abstentions, and zero wrong identities. Because neither a correct pillar
agreement nor a correct actual residual acceptance existed, Plan 0071 closes
`residual_population_infeasible`. Fresh evaluation never opened; E0-E3 remain
not run, and live assignment, knowledge, Graphiti, provider-write, and external
effects remain closed.

Plan 0072 is the open correction-first P09/P10 productization campaign in
`docs/dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md`.
Its accepted detailed design authority is
`docs/dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md`.
It replaces neither the conversation knowledge store nor the governed acoustic
stores. It defines one append-only architecture for event association,
suspected participants, source-scoped contact gathering, canonical people,
hierarchical temporal roles and relationships, scoped transcript corrections,
speaker proposals, soft anonymous voice clusters, samples, profile families,
score history, processing ledgers, and rebuildable review projections. The
Authelia-protected Identity Review and People tabs must show actual original
recording filenames, event alternatives, suspected attendees, independent
contextual/acoustic evidence, playable samples, and reversible controls.
Identification is explicitly not authentication. Unreviewed proposals cannot
become accepted people, relationships, speaker assignments, or named-profile
enrollment authority. Evidence Strength is not probability; empirical
likelihood and future automatic acceptance have explicit source-disjoint
denominators. The first authorized private A6 historical cohort has completed
in an isolated copied store; the seven-day arrival observation remains active.
Provider retrieval, live migration, biometric collection, dashboard
publication, deployment, and accepted identity/profile effects have not
begun. A0 is closed with versioned domain, correction, privacy, threat, review,
adapter, and supervisor contracts, redacted fixtures, and deterministic tests.
A1 is also closed: knowledge schema v4 now provides immutable ontology and
correction events, hashed external identities, rebuildable people, source,
role, relationship, and reconciliation projections, merge/split/reversal,
exact scoped deduplication, and fail-closed baseline reconciliation proved only
on synthetic disposable stores. No live migration, private directory read,
provider effect, biometric effect, or publication occurred. A2 is closed on
redacted/synthetic stores: knowledge schema v5 adds reviewed scope-aware
terminology and version-pinned hint bundles, immutable raw and normalized
transcript generations, superseding span decisions, dual-layer search/reindex
lineage, transcript-only semantic maps, and the two-pass/one-cascade manual-stop
boundary. It produced no private, live, provider, biometric, publication, or
  deployment effects. A3 is closed on synthetic temporary stores: knowledge
  schema v6 adds private sample custody, soft reversible clusters, reviewed
  profile families and versions, material-only cluster rescore receipts,
  deterministic rebuild proof, previewed exclusions and every initial deletion
  scope, transactional private-byte removal, and minimal tombstones. It
  produced no real-voice, private-corpus, live, provider, enrollment,
  publication, deployment, or external effects. A4 is closed on synthetic and
  redacted inputs: knowledge schema v7 adds exact supervisor runs and stages,
  source-scoped adapter receipts, cumulative budgets, bounded transient retry,
  partial-failure isolation, nonbinding calendar/purpose/participant
  hypotheses, separate evidence pillars, conservative duplicate/contradiction
  caps, and source-disjoint calibration history. It made zero provider calls
  or writes and produced no accepted effects. A5 is closed on a disposable
  redacted store: knowledge schema v8, queue/People projections, stale-safe and
  idempotent review APIs, exact zero-effect previews, and separate desktop/
  mobile Identity Review and People views all passed. The A6 Level 1 readiness
  slice now adds deterministic oldest-forward selection, a distinct seven-day
  artifact-stabilization window, immutable private activation/arrival/case
  receipts, persisted A4 supervisor verification, optional A5 queue
  projection, exact zero-effect enforcement, and a replayable aggregate
  scorecard on redacted fixtures. The explicit private checkpoint then
  activated campaign `identity-shadow-36ce4c7eb1467aea97e071b4`. Its 25
  historical cases reached terminal A4 state and projected 25 private queue
  records with 108 speaker labels, 25 calendar candidates, and 55 participant
  hypotheses. All cases remain partial because provider access stayed closed;
  there were zero provider calls, review submissions, or accepted effects.
  Real-queue loopback desktop/mobile and range-audio usability passed. A6
  remains below its terminal Level 2 gate because the arrival window runs
  through `2026-08-23T23:33:20Z`, authenticated-route usability is unproven,
  and the terminal scorecard/replay have not run.

Historical execution: P4A model/code/terms acquisition and P4B offline
adapters plus synthetic private profile lifecycle closed before the initial P4C
exact real-enrollment preview reported that no canonical real P3 reference
store existed. A private no-audio candidate
proposal recovers two
raw-file hash drifts through a committed metadata-only authority that binds
the frozen campaign, blind-prediction, completed-run, prompt, status, and clue
packet hashes and matches the clues exactly against current speaker, ordinal,
timestamp, and bounded-text fields. Two multi-session opaque
candidates were resolved through exact production P3 approvals and a
content-addressed P4 apply authority under the operator's blanket proceed
instruction. Six active real profiles now cover both candidates across all
three pinned models using no-enhancement enrollment windows. P4D development
resubstitution diagnostics are complete across all five preparation paths:
450 logical scores, 225 genuine/225 impostor, and 270 unique
waveform/model/profile combinations. They are explicitly non-held-out and do
not support accuracy, threshold, or model-selection claims. P4D2 held-out
calibration is complete: 22 pre-score-frozen windows produced 396 trials and
nine model-by-method thresholds with descriptive error, calibration, margin,
open-set, and condition-slice evidence. P4E generation 1 was revealed under an
independently reviewed authority but terminally stopped before audio/model
execution because the bound P2 module lacked an evaluation-split seam. The
revealed cohort cannot be reused for terminal selection. Plan 0043 added and
replay-tested the evaluation seam before any successor authority freeze. Its
exact-seven successor campaign is now fully operator-reviewed, and the new
conversation/source-disjoint corpus is frozen and replayed with a deterministic
3/2/2 development/calibration/evaluation split. It has 10 known subjects, 3
recurrent subjects, and 23 feasible same-person session pairs. The successor
condition campaign completed exact 7 P1 and 35 P2 method successes with replay
and independent audit. Channel, noise, telephone-bandwidth, and usable-duration
coverage passed. Plan 0044 therefore closed with terminal `STOP`; generation-2
biometric scoring and selection remain `not_run`. Plan 0047 subsequently
recovered all seven exact frozen sources and froze five allowlisted
manufacturer hardware-model facts in a separate replayed authority. Cases 2
and 4 remain unavailable, and the five observed rows represent only one
distinct device, so the original device gate still failed without inference.
Plan 0047 subsequently closed that gate with two operator-confirmed webcam rows
and five manufacturer-metadata rows, yielding seven authoritative rows, two
distinct opaque devices, and zero missing recordings. Plan 0046 is now closed:
it replays the archived calibration authority across the reviewed
evaluation-split P2 seam, freezes an independently audited generation-2
pre-reveal authority, and authorizes the later evaluation reveal. The apply did
not reveal evaluation, prepare audio, freeze windows, run models, score trials,
calculate metrics, or decide. Model execution and scoring remain blocked until
the exact post-window trial child exists and replays.
Plan 0048 then executed the authorized prediction-excluded reveal. Its two
evaluation recordings contain five opaque subjects, none represented by the
two frozen profile subjects. All nine candidate-by-method units therefore have
zero possible genuine and impostor trials against required minima of 20 and
100. Immutable run
`generation-2-evaluation-stop-5945db0810a482bbbe80db74` records terminal
`STOP` and replays full-body. Audio preparation, window freeze, exact-trial
child construction, models, scores, metrics, P5 integration, and P6 historical
reprocessing are `not_run`. No operational accuracy claim, default integration,
or model/method selection exists. Plan 0036 remains sealed and paused at five
of ten current gold reviews.

Plans:

- `docs/dev/plans/0037-2026-07-31-audio-enhancement-biometric-speaker-identity.md`
- `docs/dev/plans/0038-2026-07-31-plan-0037-p0-contract-evaluation-freeze.md`
- `docs/dev/plans/0039-2026-07-31-plan-0037-p1-audio-derivatives-quality.md`
- `docs/dev/plans/0040-2026-07-31-plan-0037-p2-speech-preparation-comparison.md`
- `docs/dev/plans/0041-2026-07-31-plan-0037-p3-biometric-reference-library.md`
- `docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`
- `docs/dev/plans/0043-2026-07-31-plan-0037-p4e2-successor-evaluation.md`
- `docs/dev/plans/0044-2026-08-01-plan-0037-p4e2-condition-measurement.md`
- `docs/dev/plans/0045-2026-08-01-plan-0037-p4e2-device-provenance-refinement.md`
- `docs/dev/plans/0046-2026-08-01-plan-0037-p4e2-generation-2-authority.md`
- `docs/dev/plans/0047-2026-08-01-plan-0037-source-device-metadata.md`
- `docs/dev/plans/0048-2026-08-01-plan-0037-generation-2-evaluation-execution.md`
- `docs/dev/plans/0049-2026-08-02-additional-acoustic-training-conversations.md`
- `docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`
- `docs/dev/plans/0057-2026-08-06-enrolled-only-acoustic-shadow-review-integration.md`

Research:

- `docs/dev/notes/2026-07-31-acoustic-processing-and-speaker-verification-research.md`

Vision Outcome:

- Preserve immutable source audio while deriving reproducible speech-cleanup
  artifacts.
- Use calibrated biometric speaker evidence to improve correct identity,
  abstention, and same-person diarization-label grouping.
- Reprocess historical audio before returning to context-assisted speaker
  identity and full contextual readouts.

Definition of Done:

- Versioned voice activity, enhancement, quality, and diarization preparation
  produce timestamp-aligned derived artifacts without changing originals.
- A private, provenance-backed biometric library supports reviewed enrollment,
  supersession, withdrawal, and deletion.
- At least two purpose-built speaker-verification models are compared and
  calibrated on conversation-separated local evidence.
- App Intelligence receives bounded acoustic evidence rather than raw audio or
  embeddings.
- Historical reprocessing is dry-run-first, approval-gated, resumable,
  idempotent, and independently auditable.
- An unseen evaluation records a terminal decision before the acoustic path
  becomes default or automatic speaker confirmation is reconsidered.
