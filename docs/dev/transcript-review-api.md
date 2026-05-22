# Transcript Review API

`transcript_api.py` is the local API for the planned React + Vite review console.

It serves only the configured user-scoped transcript store. It does not read arbitrary filesystem paths from request parameters, and blob playback is limited to blob ids registered in `~/.transcripts/transcripts.sqlite3`.

## Start

```bash
python transcript_api.py --store-dir ~/.transcripts --host 127.0.0.1 --port 18876
```

Development and test runs can use `--embedding-provider debug-hash` when search should avoid a live embedder. The default port is pinned to `18876` for cooper ingress as `transcripts.localhost` and `transcripts.ecochran.dyndns.org`.

The API reads operator workflow state from `--state-dir`, defaulting to `~/.local/state/transcribe-audio`. That state remains user-scoped runtime data, not tracked repo content.

When `frontend/dist/` exists, the same server also serves the built React console at `/`; API routes remain under `/api`.

## Endpoints

- `GET /api/health`: service and store path.
- `GET /api/library?kind=transcript&limit=50&offset=0`: paged stored document list.
- `GET /api/review-queue?limit=50`: read-only review queue aggregation over local route-review files, App Intelligence human-review decisions, filename-conflict reviews, and first-pass summary queue counts.
- `GET /api/intelligence/providers`: local provider registry and readiness checks for intelligence surfaces, including `codex-app-server` as the preferred supervised App Intelligence control plane.
- `GET /api/intelligence/smokes`: latest App Intelligence smoke run and browser-smoke evidence metadata. It reports paths, status, and check booleans only; it does not read screenshot bytes or artifact contents.
- `GET /api/intelligence/smoke-jobs?limit=10`: list recent allowlisted smoke jobs and stdout/stderr artifact paths without reading artifact contents.
- `GET /api/intelligence/smoke-jobs/<job_id>/tail?stream=stderr&chars=4000`: read a bounded stdout or stderr tail for one smoke job. The stream must be `stdout` or `stderr`, the job id is path-segment validated, and the output path must remain inside the smoke-job directory.
- `POST /api/intelligence/smoke-jobs`: queue an allowlisted smoke job. Supported `job_type` values are `api_replay_smoke`, `browser_replay_smoke`, and `cleanup_smokes`. API/browser smoke jobs require `approval_token=RUN_APP_SMOKE_JOB`; cleanup with `apply_cleanup=true` requires `approval_token=CLEANUP_APP_SMOKE_ARTIFACTS`. The endpoint writes a job record under `~/.local/state/transcribe-audio/smoke-jobs/`, starts a background worker, and never accepts arbitrary shell input. The React Smoke Status card polls this endpoint every 2 seconds while any loaded job is `queued` or `running`, cleanup apply requires typing `CLEANUP_APP_SMOKE_ARTIFACTS`, completed cleanup jobs expose a redacted retention summary with matched/kept/delete counts, write-bearing jobs are visually distinguished from read-only jobs with an inline legend, recent rows show queued/finished/runtime timing, loaded jobs are grouped by action type, and the UI shows loaded-vs-total job counts.
- `GET /api/intelligence/config`: resolved task-level intelligence routing from defaults, optional user config, environment, and runtime overrides.
- `POST /api/intelligence/config/preview`: validate and preview a task routing update without writing.
- `POST /api/intelligence/config/apply`: apply a validated task routing update to the user-scoped config. Requires `approval_token=APPLY_INTELLIGENCE_CONFIG_UPDATE`.
- `GET /api/intelligence/runs?limit=50`: list prepared App Intelligence run ledgers under the user-scoped state directory.
- `GET /api/intelligence/runs/<run_id>`: read one App Intelligence run ledger, structured decision history, prompt/status artifact paths, and recent append-only events. The console derives preflight artifact choices from event payloads with recorded artifact paths.
- `GET /api/intelligence/runs/<run_id>/replay-manifest`: read an ordered manifest of registered prompt, status, decision, and preflight artifacts for replay UI display. The manifest reports metadata only; it does not read artifact contents or execute actions.
- `GET /api/intelligence/runs/<run_id>/artifacts?path=<path>`: read a registered run artifact for inspector display. The path must resolve inside that run directory and match an artifact path recorded in the run ledger or event log; this does not allow arbitrary filesystem reads.
- `POST /api/intelligence/runs/prepare`: create a prepared App Intelligence run ledger without starting app-server sessions or provider work.
- `POST /api/intelligence/runs/<run_id>/session-start-preflight`: validate session-start prerequisites without starting app-server sessions or provider work. A dry run can pass `approval_token=START_APP_SERVER_SESSION` to validate token shape. Passing `append_event=true` records only a `session_start_preflight` event and requires `approval_token=APPEND_SESSION_START_PREFLIGHT_EVENT`.
- `POST /api/intelligence/runs/<run_id>/session-start`: start the managed Codex app-server control-plane daemon for a prepared ledger. Requires `approval_token=START_APP_SERVER_SESSION`, restricts `transport` to `stdio` or `unix`, writes host ledger events before and after daemon start, and does not start a model turn or create a Codex thread.
- `POST /api/intelligence/runs/<run_id>/model-turn-preflight`: prepare a reviewed initial prompt packet from a selected stored document and resolved task route. Requires `approval_token=PREPARE_MODEL_TURN_PREFLIGHT`, writes packet JSON and prompt text under the run artifacts directory, appends a ledger event, and does not send a prompt.
- `GET /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>`: read one prompt-packet JSON artifact and its prompt text for operator review. It returns `will_send_prompt=false` and surfaces the future `SEND_APP_SERVER_MODEL_TURN` token without using it.
- `POST /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>/send-preflight`: validate that a reviewed packet is eligible for a future model turn send. Requires `approval_token=SEND_APP_SERVER_MODEL_TURN`, returns check results, and still returns `will_send_prompt=false` and `will_write_event=false`.
- `POST /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>/send`: send a reviewed prompt packet to Codex app-server. Requires `approval_token=SEND_APP_SERVER_MODEL_TURN`, starts or reuses a Codex thread, starts one turn, records thread/turn ids and captured app-server events, and returns `will_execute_downstream_action=false`.
- `POST /api/intelligence/runs/<run_id>/turn-status`: capture the latest recorded Codex turn status and output. Requires `approval_token=CAPTURE_MODEL_TURN_STATUS`, writes a status artifact under the run directory, records captured app-server events, and returns `will_execute_structured_decision=false`.
- `POST /api/intelligence/runs/<run_id>/structured-decision/validate`: parse and validate the latest captured turn output as a host-owned structured decision. Requires `approval_token=VALIDATE_STRUCTURED_DECISION`, writes a validation artifact, appends accepted/rejected decision metadata, and returns `will_execute_host_action=false`.
- `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/apply`: record a validated ledger-only structured decision. Requires `approval_token=APPLY_STRUCTURED_DECISION`, currently accepts `continue_current_branch`, `stop`, and `ask_for_human_review`, writes an apply artifact, updates the local run ledger, and returns `will_execute_external_action=false`, `will_execute_write_bearing_action=false`, and `will_fork_or_rollback=false`. `continue_current_branch` leaves the run open as `current_branch_continued`.
- `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/human-review`: annotate, resolve, or reopen a human-review decision. Requires `approval_token=RECORD_HUMAN_REVIEW_DECISION`, writes only to the local run ledger, appends a `human_review_decision_recorded` event, and returns no external action, write-bearing action, fork, or rollback.
- `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/fork-preflight`: preview a validated `fork_branches` decision. Requires `approval_token=PREVIEW_FORK_BRANCHES`, writes a reviewable preflight artifact/event, and returns `will_create_thread=false`, `will_modify_branches=false`, and `will_run_provider=false`.
- `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/rollback-preflight`: preview a validated `rollback` decision. Requires `approval_token=PREVIEW_ROLLBACK`, writes a reviewable preflight artifact/event, and returns `will_modify_branches=false`, `will_revert_artifacts=false`, `will_create_thread=false`, and `will_run_provider=false`.

Replay-manifest smoke:

```bash
python scripts/smoke_app_replay_manifest.py --cleanup
python scripts/smoke_app_replay_manifest_ui.py --cleanup
python scripts/smoke_first_pass_batch_resume_ui.py --cleanup
python scripts/cleanup_app_smokes.py
```

This creates a disposable user-scoped run, verifies `/replay-manifest`, opens
one registered artifact through `/artifacts?path=...`, checks the no-write
flags, and deletes the run when `--cleanup` is supplied. The UI smoke also
drives the React Intelligence panel with `agent-browser` and stores JSON plus
screenshot evidence under `~/.local/state/transcribe-audio/browser-smokes/`.
The first-pass resume UI smoke creates a disposable prepared manifest, selects
it from the Review Queue after page load, and checks prepared status without
submitting provider work.
The cleanup command is dry-run by default; pass `--apply` only after reviewing
the reported run and evidence paths.

- `POST /api/review-queue/first-pass-summaries/prepare`: create a dry-run first-pass summary batch manifest without submitting provider work.
- `GET /api/review-queue/first-pass-summaries/manifests?limit=10`: list recent first-pass summary batch manifests as redacted summaries for resuming status checks after reload. It reports request counts, batch id/status, provider counts, materialized count, and materialization error count without reading request payloads or transcript content.
- `POST /api/review-queue/first-pass-summaries/submit`: submit an existing prepared manifest. Requires `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`.
- `POST /api/review-queue/first-pass-summaries/status`: poll a submitted manifest and optionally materialize completed readouts with `materialize=true`. The Review Queue renders returned request counts, batch id/status, provider counts, materialized count, and materialization error count.
- `GET /api/search?q=<query>&kind=transcript&limit=10`: lexical/semantic search over stored artifacts.
- `GET /api/documents/<document_id>`: document detail, JSON payload, text content, metadata, and linked blobs.
- `GET /api/documents/<document_id>/context?chunk_index=5&context_chunks=1`: nearby transcript/readout context from stored chunks.
- `GET /api/blobs/<blob_id>`: registered blob playback/download endpoint with `Range` support.
- `GET /api/blobs/<blob_id>?download=1`: same blob as an attachment.

## Blob Contract

Transcript ingestion copies existing source recordings into:

```text
~/.transcripts/blobs/<prefix>/<blob-id>.<ext>
```

The SQLite store records:

- `blobs`: blob id, original path, stored path, hash, MIME type, byte size.
- `document_blobs`: document-to-blob links and roles such as `source_recording`.
- document metadata `media_blob`: compact frontend-facing playback/download URLs.

The UI should play recordings through `/api/blobs/<blob_id>` rather than using original `~/Downloads` paths. This keeps playback stable after the source file is moved or deleted and prevents arbitrary path streaming.

## Review Queue Contract

`/api/review-queue` returns:

- `buckets`: summary cards for route reviews, App Intelligence human-review decisions, filename conflicts, first-pass summaries, memory harvest, and speaker ID work.
- `items`: mixed review rows. Route rows come from `~/.local/state/transcribe-audio/review-queue/`; App Intelligence rows come from `~/.local/state/transcribe-audio/app-intelligence-runs/`.
- `status=needs_human_review`: an App Intelligence `ask_for_human_review` decision has been ledger-applied and needs operator attention.
- `status=resolved`: an App Intelligence human-review item has been resolved locally and does not count as open.
- `route_decision_exists`: whether a route-review item still points at a readable route-decision artifact.
- `status=stale_reference`: a local review item exists, but its referenced route decision is gone, commonly from earlier pytest/temp runs.

The queue aggregation endpoint is read-only and intentionally reports stale references instead of deleting or hiding them.

Use `review_queue_maintenance.py` for reviewed cleanup of stale local
route-review files. It is dry-run by default and requires
`--apply --approval-token ARCHIVE_STALE_ROUTE_REVIEWS` before moving files to
`~/.local/state/transcribe-audio/review-queue-archive/<run-id>/`.

First-pass summary preparation writes a dry-run manifest under
`~/.local/state/transcribe-audio/first-pass-summary-batches/`, returns the
manifest path and request count, and leaves `batch=null`. It does not submit
provider work. Submit and status actions are manifest-scoped: the API refuses
manifest paths outside that directory, submit requires an explicit approval
token, and status can materialize completed provider results back into the
store when requested.

After materialization, run `scripts/check_readout_quality.py --manifest <path>`
before scaling batch size. The check is non-mutating and reports only structural
quality metadata: schema version, paired Markdown presence, source-artifact link
existence, summary length, and counts for participants, topics, action items,
matter candidates, and memory candidates.

## Security Boundary

This API is currently local and exposes manifest-scoped first-pass summary batch actions. Operator login and scoped share links are planned for a later P09 slice and should follow the `previews` model: single-operator guard for operator routes and revocable token-hash-backed share links for scoped reviewer access.

Do not expose this service publicly without an auth layer.

## Intelligence Provider Registry

`/api/intelligence/providers` is read-only. It reports provider capabilities and readiness metadata for the operator UI without launching long-running agent sessions or touching provider secrets. `/api/intelligence/config` reports the resolved task routing used by routines that call `intelligence_config.py`.

The registry treats `codex-app-server` as the default supervised App Intelligence surface because it supports persistent sessions, branching, rollback, streamed events, and structured decision turns under a host-owned ledger. The current readiness check is intentionally narrow: it verifies the configured `codex` binary, `codex --version`, `codex app-server --help`, and protocol generation help surfaces.

If `codex` is not on the service `PATH`, configure it with `--codex-bin /absolute/path/to/codex` or `TRANSCRIPTS_CODEX_BIN=/absolute/path/to/codex`.

Use `codex exec` for stateless leaf jobs. Use `codex app-server` only for workflows that need durable thread state, replayable event streams, schema-validated decisions, or branch/rollback control. WebSocket transport must remain disabled for public or non-loopback exposure until an explicit auth and network-boundary review is completed.

## Intelligence Task Config

The central intelligence library is `intelligence_config.py`. It resolves task-level provider choices from:

1. Built-in defaults.
2. `~/.local/state/transcribe-audio/intelligence.config.json`, or `TRANSCRIPTS_INTELLIGENCE_CONFIG`.
3. Per-task environment variables such as `TRANSCRIPTS_INTELLIGENCE_FIRST_PASS_SUMMARY_PROVIDER`.
4. Explicit CLI or API request overrides.

Current task ids are `first_pass_summary`, `contextual_reread`, `context_source_ranking`, `route_selection`, `speaker_disambiguation`, `memory_harvest_review`, `embedding`, and `app_supervisor`.

Config updates use the same preview/apply pattern as other write-bearing operator flows. Preview accepts a task id plus an `update` object with allowed fields `provider`, `model`, `base_url`, `timeout`, `temperature`, `fallbacks`, `requires_ledger`, and `human_review`; it returns before/after config, resolved task values, rollback metadata, and does not write. Apply writes only to the user-scoped config path and requires `approval_token=APPLY_INTELLIGENCE_CONFIG_UPDATE`.

## App Intelligence Run Ledgers

App Intelligence run ledgers live under:

```text
~/.local/state/transcribe-audio/app-intelligence-runs/<run_id>/
```

Each prepared run has:

- `run.json`: schema version, workflow, phase, provider, host-owned policy, current branch, Codex thread placeholders, RNG seed ledger, artifact registry, and final decision slot.
- `events.jsonl`: append-only host event log.
- `codex_events.jsonl`: reserved append-only capture of future app-server streamed events.
- `branches/`, `artifacts/`, and `diffs/`: reserved user-scoped run artifacts.

The prepare endpoint only creates this ledger. It does not spawn `codex app-server`, create Codex threads, fork branches, run model turns, or perform external writes. The session-start preflight endpoint validates provider readiness, prepared ledger phase, allowed actions, host-owned policy, structured-decision policy, and approval-token shape. It never starts a session. Its optional event-append mode records only that the preflight was checked and uses a separate event token from the session-start approval token. The session-start endpoint starts only the managed app-server control-plane daemon and records daemon/version metadata in the ledger; `active_codex_thread_id` remains empty and `will_start_model_turn=false`. The model-turn preflight endpoint creates reviewable prompt artifacts only and surfaces the future `SEND_APP_SERVER_MODEL_TURN` approval token without using it. The prompt-packet review endpoint reads only existing packet JSON/text from the run artifacts directory so the operator can inspect the exact prompt before any send path exists. The send-preflight endpoint requires the future send token and validates ledger phase, policy, packet/run match, review requirement, unsent state, and prompt text presence, but still sends no prompt and writes no event. The send endpoint uses the same token, starts or reuses a Codex thread through the local app-server proxy, starts one turn, records the Codex thread id, turn id, packet sent state, and captured app-server events, and returns before any downstream apply/write path exists. The turn-status endpoint reads the recorded thread/turn, captures app-server status and output into `artifacts/model-turn-readouts/<turn_id>.status.json`, and marks completed turns without executing model output. The structured-decision validator accepts only a JSON object with `action`, `rationale`, `confidence`, and `review_flags`; allowed actions are `continue_current_branch`, `fork_branches`, `rollback`, `stop`, and `ask_for_human_review`. Validation writes an artifact and ledger decision record but still does not execute the action. The first apply endpoint is ledger-only: it accepts validated `continue_current_branch`, `stop`, and `ask_for_human_review` decisions, writes `artifacts/structured-decisions/<decision_id>.apply.json`, updates run phase/status, and records `structured_decision_applied` without forking, rolling back, writing memory, applying routes, touching repositories, or calling external systems. `continue_current_branch` records `latest_continuation` and leaves the run open. Human-review actions can annotate, resolve, or reopen an `ask_for_human_review` decision with `RECORD_HUMAN_REVIEW_DECISION`; they update only the local run ledger and review queue visibility. Fork preflight can preview `fork_branches` and write `artifacts/structured-decisions/<decision_id>.fork-preflight.json`, but it does not create Codex threads, modify branch state, or run providers. Rollback preflight can preview `rollback` and write `artifacts/structured-decisions/<decision_id>.rollback-preflight.json`, but it does not modify branch state, revert artifacts, create threads, or run providers. Future app-server phases must add separate approval-gated apply endpoints before the host executes any fork, rollback, write, network, memory, routing, or deposition action.
