# Runbook

`RUNBOOK.md` is the dated execution log for this repo. Use it to record policy adoption, roadmap changes, implementation slices, validation evidence, and operational incidents that should survive chat history.

## Turn 1 | 2026-05-04

Summary: Adopted repo-local policy and planning surfaces for the expanded transcription platform scope.

Changes:

- Added `docs/dev/policies/` with operations-platform-oriented local policy.
- Added `ROADMAP.md` with six lanes.
- Added bounded plan files under `docs/dev/plans/`.
- Reclassified `docs/platform-expansion-plan.md` as background architecture notes rather than planning authority.
- Wired plans: `0001-2026-05-04-normalize-transcript-artifacts.md`, `0002-2026-05-04-calendar-provider-config.md`, `0003-2026-05-04-intelligence-readouts.md`, `0004-2026-05-04-matter-routing-contextual-reread.md`, `0005-2026-05-04-deposition-memory-harvest.md`, `0006-2026-05-04-service-reliability-observability.md`.
- Added Graphiti repo memory guidance for group `transcribe_audio_main`.

Policy Decision:

- Deterministic selector first pass recommended `standalone-library` from current repo shape.
- Maintainer scope requires operations-platform policy because the target system includes tenant-aware calendar access, runtime state, intelligence providers, matter routing, deposition, Graphiti/OpenClaw memory, and service reliability.

Validation:

- Ran `select_policy.py` against the local policy library.
- Ran `audit_planning_contract.py`; initial audit reported missing `ROADMAP.md`, `RUNBOOK.md`, and `docs/dev/plans/`.
- Re-ran `audit_planning_contract.py` after adoption; final result passed with `ok: true`.

Next:

- Start P01 by implementing transcript artifact sidecars.

## Turn 2 | 2026-05-04

Summary: Bootstrapped Graphiti repo memory for `transcribe-audio`.

Changes:

- Added concrete Graphiti discovery guidance to `AGENTS.md`.
- Added repo memory group guidance to `docs/dev/policies/0005-memory-and-context-routing.md`.
- Seeded Graphiti group `transcribe_audio_main` from curated repo authorities only.

Seeded Episodes:

- `9c987621-2252-45b4-a9ea-c98f9d6aff17`: `transcribe-audio: bootstrap: policy roadmap runbook`.
- `dbaa9e6b-1e6f-4929-91f7-9d5547a2923c`: `transcribe-audio: roadmap lanes and next implementation slices`.
- `d9a28757-ca32-4382-b8fb-fd18e3d689f0`: `transcribe-audio: notable event: systemd ffprobe path stall`.
- `6b698756-bcdb-4111-a873-5200ab6e7940`: `transcribe-audio: memory and context routing contract`.

Validation:

- `graphiti-runtime status` reported healthy runtime, HTTP, and FalkorDB.
- `graphiti-runtime discover --group-id transcribe_audio_main ...` returned 4 episodes and 10 facts.
- `graphiti-runtime queue` showed `transcribe_audio_main` queue size 0 after writes completed.

Next:

- Use `graphiti-discovery` against `transcribe_audio_main` before future non-trivial planning, debugging, architecture, routing, memory, or handoff work.

## Turn 3 | 2026-05-04

Summary: Started P01 by adding transcript artifact sidecars.

Changes:

- Added `transcript_artifacts.py` with `TranscriptArtifact` JSON serialization.
- Updated shared transcript output processing to write `*.transcript.json` sidecars.
- Sidecars include transcript text and selected structured utterances for downstream automation.
- Backend CLIs now emit `TRANSCRIPT_ARTIFACT_JSON=<path>` lines through the shared output path.
- Watcher state now preserves `artifact_paths` for successful processed records.
- Added focused tests under `tests/test_transcript_artifacts.py`.
- Documented the sidecar output contract in `README.md`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed with 3 tests.
- `python -m py_compile transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py` passed.
- Graphiti event `ca8c403c-73b2-47ca-ae40-3b777d81c605` recorded the P01 artifact sidecar implementation and smoke search returned 5 facts.

Next:

- Run a manual short-recording smoke with `--text-output --use-calendar` when a suitable non-sensitive clip is available.
- Continue P01 by running manual smoke and deciding whether P01 can close or needs richer artifact schema fields.

## Turn 4 | 2026-05-04

Summary: Started P02 by making calendar provider selection explicit and tenant-aware.

Changes:

- Added `CalendarProviderConfig` and explicit provider ordering.
- Changed default calendar lookup order to `gog`, then `gws`, then built-in `google-api` fallback.
- Made Google API fallback lazy so OAuth is not triggered unless that provider is reached.
- Added CLI flags for `--calendar-providers`, `--calendar-gog-account`, `--calendar-gog-client`, and `--calendar-gws-config-dir`.
- Added watcher `calendar` config expansion into backend CLI args.
- Documented the provider config in `README.md` and `watch_transcriptions.json.sample`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed with 7 tests.
- `python -m py_compile transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py` passed.
- Graphiti event `350f35aa-1570-42e1-b84a-3abde59256ed` recorded the P02 calendar provider implementation.

Next:

- Run a manual watcher `--run-once` calendar lookup test against a non-sensitive short clip.
- Continue P03 after P01/P02 manual smoke gates are handled or intentionally deferred.

## Turn 5 | 2026-05-04

Summary: Closed P01 and P02 with a temp-location watcher smoke.

Smoke Setup:

- Copied `/home/ecochran76/Downloads/You Have Been Banned.mp3` to `/tmp/transcribe-watch-smoke.TFGX7X/watch/`.
- Created temp watcher config at `/tmp/transcribe-watch-smoke.TFGX7X/watch_config.json`.
- Used backend `faster_whisper` with `tiny.en`, CPU `int8`, `--text-output`, and `--no-speaker-labels`.
- Enabled structured watcher calendar config with providers `gog,gws` to avoid touching Google OAuth fallback during smoke.
- Used temp state file `/tmp/transcribe-watch-smoke.TFGX7X/state.json`.

Validation:

- First `watch_transcriptions.py --run-once` pass tracked the candidate.
- Second `watch_transcriptions.py --run-once` pass processed the stable file successfully.
- Calendar lookup tried `gog` and returned 8 events.
- The temp media was renamed from matched event metadata.
- Outputs were written under `/tmp/transcribe-watch-smoke.TFGX7X/out/`.
- Watcher state recorded the artifact path and backend success.
- Transcript text: `Speaker [0.00s - 1.74s]: You have been banned.`

Evidence:

- Artifact path: `/tmp/transcribe-watch-smoke.TFGX7X/out/2026-05-04 19-30 choir concert You Have Been Banned Transcript.transcript.json`.
- State command recorded `--use-calendar --calendar-providers gog,gws --calendar-id primary --calendar-window 24`.
- P01 and P02 are now closed in `ROADMAP.md` and their plan files.

Notes:

- Because the file was copied into `/tmp` with a fresh mtime, calendar matching used the temp copy timestamp and selected a nearby event. This was sufficient for provider/watcher validation, not a semantic event-match quality test.

Next:

- Start P03 by defining readout schemas and the first OpenAI-compatible intelligence provider seam.

## Turn 6 | 2026-05-04

Summary: Started P03 by implementing structured intelligence readouts.

Changes:

- Added `readout_artifacts.py` with readout JSON schema and Markdown rendering.
- Added `summarize_transcript.py` to read transcript sidecars and call an OpenAI-compatible chat completions API.
- Added provider seams for `openai-compatible`, `codex-exec`, `auracall`, and `openclaw`; only `openai-compatible` is implemented in this slice.
- Added watcher `readout` config and post-processing after successful transcription.
- Watcher state now preserves `readout_paths` when readout generation succeeds.
- Readout failures are logged as warnings and do not mark transcription as failed.
- Documented readout usage in `README.md`, `api_keys.json.sample`, and `watch_transcriptions.json.sample`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 12 tests.
- `python -m py_compile readout_artifacts.py summarize_transcript.py transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py tests/test_readouts.py` passed.
- Manual local OpenAI-compatible smoke generated readout JSON and Markdown from the temp transcript artifact at `/tmp/transcribe-watch-smoke.TFGX7X/out/2026-05-04 19-30 choir concert You Have Been Banned Transcript.transcript.json`.
- Graphiti event `a7dd576c-a25d-426b-8754-f24c8178030e` recorded the P03 readout implementation.

Evidence:

- Readout JSON: `/tmp/transcribe-watch-smoke.TFGX7X/readouts/2026-05-04 19-30 choir concert You Have Been Banned Transcript.readout.json`.
- Readout Markdown: `/tmp/transcribe-watch-smoke.TFGX7X/readouts/2026-05-04 19-30 choir concert You Have Been Banned Transcript.readout.md`.
- Smoke summary: `The short clip says the listener has been banned.`

Next:

- Smoke P03 against a real configured provider/key, then close P03 or move to P04 routing schemas if local-compatible validation is accepted.

## Turn 7 | 2026-05-08

Summary: Repaired recent eventless transcripts and fixed service calendar provider PATH.

Cause:

- The watcher config requested calendar lookup, but the long-running service process used older provider behavior and then the systemd user PATH did not include `~/.local/bin` or `~/.cargo/bin`.
- As a result, child transcription processes could not find `gog` or `gws` and fell through to the built-in Google API provider, which failed because OAuth client secrets were not configured.

Changes:

- Updated `watch_transcriptions.json` to use structured `calendar` config with provider order `gog,gws,google-api`.
- Updated `~/.config/systemd/user/transcribe-watch.service` and `.openclaw/transcribe-watch.service` PATH to include `%h/.local/bin` and `%h/.cargo/bin`.
- Added `repair_calendar_metadata.py` for dry-run/apply backfills using each artifact's recorded recording window.
- Restarted `transcribe-watch.service` after `systemctl --user daemon-reload`.

Backfill:

- Repaired 9 recent eventless transcript artifacts: recordings 115 through 123.
- Regenerated TXT/DOCX transcript outputs with event details and participants.
- Renamed transcript artifacts and media files to calendar-based names.
- Updated `.openclaw/watch_transcriptions_state.json` so renamed recordings are not reprocessed.

Validation:

- `gog status` showed an authenticated `ecochran76@gmail.com` account.
- Repair dry run matched 8 artifacts before apply; follow-up repair matched recording 123 created during the service restart window.
- Recent sidecar scan reported `missing_recent_event_count=0`.
- Sample repaired transcript `2026-05-08 14-00 Timothy Clark My recording 122 Transcript.txt` includes event details and participants.
- `python -m py_compile repair_calendar_metadata.py watch_transcriptions.py assembly_transcribe.py faster_whisper_transcribe.py transcribe_common.py` passed.
- Planning audit passed with `ok: true`.

Notes:

- `My recording 123 (1).m4a` remains a live watcher candidate but is currently incomplete/corrupt (`moov atom not found`) and still changing. The watcher is correctly waiting rather than processing it.

Next:

- After `My recording 123 (1).m4a` finishes syncing, confirm the next successful service transcript logs `Calendar lookup: trying provider gog...`.

## Turn 8 | 2026-05-08

Summary: Confirmed the live service now uses `gog` calendar mode.

Validation:

- `My recording 123 (1).m4a` finished syncing and was processed by `transcribe-watch.service`.
- The journal showed `Calendar lookup: trying provider gog...` followed by `Calendar lookup: provider gog returned 7 event(s).`
- The service matched event `SIP WMA Hamburg` and renamed media/output files to `2026-05-08 15-00 SIP WMA Hamburg My recording 123 (1) ...`.
- The repaired transcript text includes event details, location, and participants.
- Recent sidecar scan still reports `missing_recent_event_count=0`.

Next:

- Leave the service running and monitor the next normal mobile recording only if another calendar miss appears.

## Turn 9 | 2026-05-08

Summary: Added accessible-calendar overlap context to calendar event metadata.

Changes:

- Extended calendar lookup to list accessible calendars for the active provider and query each calendar for overlapping events.
- Added `event.matching_calendars` to transcript sidecar metadata.
- Kept primary event selection and filename behavior stable; matching-calendar context is for downstream readout/routing.
- Updated `repair_calendar_metadata.py` so existing event sidecars can be refreshed with `matching_calendars` without renaming files again.
- Refreshed recent repaired transcripts to include `matching_calendars`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 17 tests.
- `python -m py_compile transcribe_common.py repair_calendar_metadata.py tests/test_transcript_artifacts.py` passed.
- Recent event sidecar scan reported `recent_event_sidecars_without_matching_calendars=0`.
- Sample `SIP WMA Hamburg` sidecar has `matching_calendars` entries for overlapping accessible calendars.
- `transcribe-watch.service` was restarted and is active.

Next:

- Use `event.matching_calendars` in the P03/P04 readout and routing prompts so overlapping calendar context helps identify the meeting matter.

## Turn 10 | 2026-05-08

Summary: Fed matching-calendar overlap context into intelligence readout prompts.

Changes:

- Added a dedicated `calendar_context` prompt block in `summarize_transcript.py`.
- Included the primary event summary, primary event participants, and `event.matching_calendars` in readout requests.
- Updated system guidance so calendar names and overlapping event summaries can inform meeting type, participant context, matter candidates, and memory candidates as evidence rather than proof.
- Documented the prompt contract in `README.md`, `ROADMAP.md`, and the P03 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_readouts.py -q` passed with 7 tests after adding prompt coverage.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.
- A local OpenAI-compatible smoke generated readout JSON/Markdown from the repaired `SIP WMA Hamburg` artifact and confirmed `matching_calendars` reached matter-candidate evidence.
- `transcribe-watch.service` remained active.

Next:

- Decide whether to close P03 after a live-provider readout smoke or move directly into P04 routing with the current local-compatible validation.

## Turn 11 | 2026-05-08

Summary: Verified recent transcripts have calendar-context metadata.

Action:

- Ran `repair_calendar_metadata.py` against `/mnt/c/Users/ecoch/Downloads/*.transcript.json` for the last 8 days with `gog,gws`.
- Applied the repair command with `--no-rename-media`; it was a no-op because all recent event sidecars already had `event.matching_calendars`.

Validation:

- Recent scan found 10 transcript artifacts, 10 with event metadata, and 0 missing `matching_calendars`.
- Calendar-context counts ranged from 1 to 5 matching calendar entries per recent transcript.
- `transcribe-watch.service` remained active.

Next:

- Run a real-provider readout smoke on one of these updated artifacts, then decide whether to close P03 or move directly into P04 routing.

## Turn 12 | 2026-05-10

Summary: Validated AuraCall as an OpenAI-compatible chat provider.

Action:

- Loaded API parameters from `/home/ecochran76/.auracall/api.env` without printing secrets.
- Sent a minimal `/v1/chat/completions` request to the AuraCall local endpoint using model `agent:instant-chatgpt-ecochran76`.

Validation:

- AuraCall returned HTTP 200 in 7.239 seconds.
- The response content matched the requested smoke phrase: `auracall smoke ok`.
- The response included a completion ID; usage accounting was not returned.

Next:

- Run `summarize_transcript.py` against one updated transcript artifact using the AuraCall OpenAI-compatible endpoint and confirm readout JSON/Markdown generation.

## Turn 13 | 2026-05-10

Summary: Tried AuraCall SoyLei ChatGPT agents for the P03 readout smoke.

Action:

- Selected `agent:pro-extended-chatgpt-soylei` from `/home/ecochran76/.auracall/config.json`.
- Ran `summarize_transcript.py` against the SoyLei/Tempo transcript artifact using the AuraCall OpenAI-compatible endpoint and model `agent:pro-extended-chatgpt-soylei`.
- Probed nearby SoyLei ChatGPT agent modes to separate profile readiness from pro/extended selector behavior.

Validation:

- `agent:pro-extended-chatgpt-soylei` returned HTTP 200 but an empty assistant message; no readout JSON/Markdown was written.
- AuraCall response metadata reported runner failure: `Unable to find the Thinking time dropdown menu.`
- `agent:instant-chatgpt-soylei` succeeded with non-empty content in 24.713 seconds.
- `agent:pro-standard-chatgpt-soylei` timed out after 109.183 seconds.
- `agent:thinking-extended-chatgpt-soylei` timed out after 109.175 seconds.

Next:

- Repair or adjust AuraCall's ChatGPT pro/thinking selector handling for the SoyLei profile, or run the P03 readout smoke with `agent:instant-chatgpt-soylei` as a temporary provider.

## Turn 14 | 2026-05-10

Summary: Retried AuraCall SoyLei pro-extended ChatGPT.

Action:

- Retried a direct one-line OpenAI-compatible smoke using `agent:pro-extended-chatgpt-soylei`.
- Kept the scope to a short smoke rather than resubmitting the full transcript, because the prior full readout failed with an empty assistant message.

Validation:

- The retry timed out after 163.74 seconds without a usable assistant response.
- `transcribe-watch.service` remained active.

Next:

- Fix AuraCall's SoyLei ChatGPT pro/thinking selector automation before using pro-extended for readouts, or temporarily use `agent:instant-chatgpt-soylei` to complete the P03 readout validation.

## Turn 15 | 2026-05-10

Summary: Passed AuraCall SoyLei pro-extended P03 readout smoke and closed P03.

Action:

- Retried a direct one-line smoke using `agent:pro-extended-chatgpt-soylei`; it succeeded with non-empty content in 15.898 seconds.
- Retried the SoyLei/Tempo transcript readout through `summarize_transcript.py`.
- Hardened `summarize_transcript.py` by duplicating the JSON-only response contract inside the user payload, because the browser-backed AuraCall path ignored the system-message-only contract on the prior attempt.

Validation:

- The hardened AuraCall readout generated valid JSON and Markdown under `/tmp/transcribe-readout-auracall-soylei-proextended-hardened.Q2cus3/`.
- Generated readout provider metadata records model `agent:pro-extended-chatgpt-soylei` and base URL `http://127.0.0.1:18095/v1`.
- The readout contains 4 participants, 15 topics, 9 action items, 5 matter candidates, and 5 memory candidates.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 19 tests.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.
- P03 is now closed in `ROADMAP.md` and `docs/dev/plans/0003-2026-05-04-intelligence-readouts.md`.

Next:

- Start P04 by defining the route decision schema and review queue for matter routing/contextual rereads.

## Turn 16 | 2026-05-10

Summary: Implemented and validated `codex-exec` readouts.

Action:

- Verified `codex exec` can return JSON through `--output-last-message` under read-only sandboxing with approval policy `never`.
- Added `codex-exec` support to `summarize_transcript.py`.
- `codex-exec` runs `codex --ask-for-approval never exec --sandbox read-only --ephemeral --output-last-message ...`, feeds the readout prompt over stdin, and validates the final message as JSON.
- Added unit coverage for command construction and JSON parsing.
- Ran the SoyLei/Tempo transcript readout through `codex-exec` with model `gpt-5.5`.

Validation:

- The direct `codex exec` smoke returned `{"ok":true,"provider":"codex-exec"}`.
- The SoyLei/Tempo `codex-exec` smoke generated valid readout JSON and Markdown under `/tmp/transcribe-readout-codex-exec-soylei.klslWj/`.
- Generated readout provider metadata records provider `codex-exec`, model `gpt-5.5`, and read-only/no-approval execution.
- The readout contains 4 participants, 13 topics, 8 action items, 4 matter candidates, and 5 memory candidates.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 20 tests.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.

Next:

- Start P04 by defining the route decision schema and review queue for matter routing/contextual rereads.

## Turn 17 | 2026-05-11

Summary: Started P04 with dry-run route decisions and local review queue support.

Action:

- Added `routing_artifacts.py` with `ContextProvenancePack`, `ProvenanceSource`, `RouteCandidate`, `RouteDecision`, and `ReviewQueueItem`.
- Added `route_transcript.py`, a dry-run CLI that reads existing transcript/readout artifacts and writes `*.route.json`.
- Current provenance extraction uses transcript calendar metadata, including `event.matching_calendars`, as the first `gws`-shaped context source.
- Current route candidates come from structured readout `matter_candidates`.
- Low-confidence route decisions write a local review queue item unless `--no-review-queue` is passed.
- Documented the dry-run route command in `README.md`.
- Moved P04 from PLANNED to OPEN in `ROADMAP.md` and `docs/dev/plans/0004-2026-05-04-matter-routing-contextual-reread.md`.

Validation:

- Added `tests/test_routing_artifacts.py` for provenance extraction, route selection, low-confidence review behavior, and CLI output.
- Manual route dry-run against the SoyLei/Tempo transcript and `codex-exec` readout selected `SoyLei Tempo Chemical technical collaboration` at confidence `0.95`.
- The manual route output had 4 route candidates, 3 rejected alternatives, and 4 provenance sources: one primary calendar event plus three calendar-overlap records.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py -q` passed with 24 tests.
- `python -m py_compile routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_routing_artifacts.py` passed.

Next:

- Add the live `gws` provenance adapter for Drive/Docs/Calendar context packs, then add Graphiti/OpenClaw candidate lookup as an advisory source.

## Turn 18 | 2026-05-11

Summary: Added live read-only `gws` provenance for P04 route decisions.

Action:

- Added `context_sources.py` with a `GwsProvenanceConfig` and read-only `gws` collection helpers.
- `route_transcript.py --gws-provenance` now collects live Calendar event details and Drive metadata search results into the route decision `provenance_pack`.
- Added `--gws-config-dir`, `--gws-drive-query`, `--gws-drive-page-size`, `--gws-timeout`, `--no-gws-calendar-details`, and `--no-gws-drive` flags.
- Default generated Drive queries now use precise filename-term intersections; operators can pass `--gws-drive-query` for broader full-text searches.
- Documented the live `gws` provenance mode in `README.md`.

Validation:

- Added `tests/test_context_sources.py` for generated Drive query behavior, `gws` response conversion, and `route_transcript.py` integration.
- Live read-only `gws` smoke against the SoyLei/Tempo transcript/readout selected `SoyLei Tempo Chemical technical collaboration` and produced 7 provenance sources: one stored calendar event, three calendar overlaps, and three live `gws_calendar_event_detail` records.
- No noisy Drive full-text hits were included with the default generated query after tightening to filename intersections.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 27 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py` passed.

Next:

- Add Graphiti/OpenClaw candidate lookup as an advisory routing source and keep it source-cited alongside `gws` provenance.

## Turn 19 | 2026-05-11

Summary: Started P07 with portable OpenClaw workspace files for the `transcripts` agent.

Action:

- Read OpenClaw docs for agent config, `openclaw agents`, channel routing, Slack channel behavior, and workspace Markdown templates.
- Confirmed the live OpenClaw Slack setup has a configured `default` account and a separate `soylei` account; the `transcripts` agent must target `slack/default`.
- Added portable workspace Markdown files under `openclaw/agents/transcripts/workspace/`.
- Added `openclaw/agents/transcripts/INSTALL.md` documenting the safe install flow and exact Slack channel-peer binding shape.
- Added `scripts/install_openclaw_transcripts_agent.py`, a dry-run-first installer scaffold that copies Markdown files to `~/.openclaw/workspace-transcripts` and runs safe agent creation/identity commands only with `--apply`.
- Opened P07 in `ROADMAP.md` and added `docs/dev/plans/0007-2026-05-11-openclaw-transcripts-agent.md`.

Validation:

- Graphiti runtime doctor reported healthy before the planning slice.
- Graphiti discovery against `transcribe_audio_main` returned existing repo planning and policy context, but no prior `transcripts` agent install routine.
- `openclaw agents list --json` showed no existing `transcripts` agent.
- `openclaw channels status --json` showed Slack account `default` is configured and running.
- `openclaw directory peers list --channel slack --query oc-transcripts --limit 20 --json` returned no matching peer, so the private channel still needs to be created or made visible to the Slack app.
- `python -m py_compile scripts/install_openclaw_transcripts_agent.py` passed.
- `scripts/install_openclaw_transcripts_agent.py` dry-run showed the expected Markdown copy targets, safe `openclaw agents add` command, identity command, and route-binding patch shape.
- Planning audit passed with `ok: true`; open lanes are P04, P06, and P07.

Next:

- Resolve or create private Slack channel `oc-transcripts`, obtain its Slack conversation id, then extend/apply the installer to add the exact `slack/default` channel-peer route binding.

## Turn 20 | 2026-05-11

Summary: Installed and live-verified the OpenClaw `transcripts` agent on Slack.

Action:

- Consulted the existing `gpod` OpenClaw agent for the channel-binding workflow; it confirmed the safe pattern of exact Slack conversation id, one route binding, channel allowlist, and post-restart validation.
- Created private Slack channel `oc-transcripts` on the default Slack tenant.
- Resolved Slack conversation id `C0B3WDRN38Q`.
- Invited the OpenClaw bot to the private channel.
- Applied the portable `transcripts` workspace files to `~/.openclaw/workspace-transcripts`.
- Created OpenClaw agent `transcripts` and set identity.
- Added exactly one route binding for `transcripts`: `slack accountId=default peer=channel:C0B3WDRN38Q`.
- Added Slack per-channel config for `C0B3WDRN38Q` with `enabled: true`, `requireMention: false`, and user allowlist `UEGM25PMG`.
- Updated the installer so `--slack-channel-id` can apply the binding idempotently and so identity setup no longer depends on indirect `IDENTITY.md` resolution.
- Closed P07 in `ROADMAP.md` and `docs/dev/plans/0007-2026-05-11-openclaw-transcripts-agent.md`.

Validation:

- `openclaw config validate` passed after binding.
- `openclaw gateway status --deep --require-rpc` passed after restart with read probe `ok` and capability `admin-capable`.
- `openclaw agents list --bindings --json` shows `transcripts` with one binding: `slack accountId=default peer=channel:C0B3WDRN38Q`.
- Slack API confirmed user and OpenClaw bot membership in `oc-transcripts`.
- A live Slack smoke message in `oc-transcripts` routed to `transcripts`; the bot replied `TRANSCRIPTS_BINDING_SMOKE_OK`.
- `python -m py_compile scripts/install_openclaw_transcripts_agent.py` passed.

Notes:

- OpenClaw directory search did not list the new private channel immediately, but Slack API, OpenClaw session state, and the live routed Slack response verified the channel and route.
- OpenClaw session created: `agent:transcripts:slack:channel:c0b3wdrn38q`.

Next:

- Continue P04 by adding Graphiti/OpenClaw advisory candidate lookup for route decisions, now using `transcripts` as the Slack operational surface for review and routing work.

## Turn 21 | 2026-05-11

Summary: Added read-only Graphiti/OpenClaw advisory provenance for P04 route decisions.

Action:

- Added `GraphitiProvenanceConfig` and read-only `graphiti-runtime discover` helpers in `context_sources.py`.
- `route_transcript.py --graphiti-provenance` now queries compact calendar/readout terms rather than raw transcript text.
- Added `--graphiti-group`, `--graphiti-command`, `--graphiti-timeout`, `--graphiti-max-facts`, `--graphiti-max-nodes`, `--graphiti-max-episodes`, and `--no-graphiti-candidates`.
- Graphiti facts and episodes are stored as provenance evidence only.
- Graphiti nodes can add low-confidence advisory `RouteCandidate` entries; they do not override high-confidence readout candidates.
- Documented Graphiti routing usage in `README.md`.
- Updated P04 roadmap and plan text to mark Graphiti/OpenClaw advisory lookup implemented.

Validation:

- `graphiti-runtime doctor` reported healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing advisory planning facts and confirmed the repo memory policy, but no prior route adapter implementation.
- Added tests for Graphiti query construction, discovery-payload conversion, and route integration.
- `.venv/bin/python -m pytest tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 10 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py tests/test_context_sources.py tests/test_routing_artifacts.py` passed.
- Live read-only Graphiti route smoke on the SoyLei/Tempo transcript/readout selected the same readout candidate, `SoyLei Tempo Chemical technical collaboration`, at confidence `0.95`.
- Live smoke output included 27 Graphiti provenance sources and 10 node-based advisory Graphiti candidates; the route remained `status=selected` and `review_required=false`.

Next:

- Add local candidate index support or move into contextual reread source fetching for the selected route.

## Turn 22 | 2026-05-11

Summary: Added read-only Odollo/Odoo provenance for P04 route decisions.

Action:

- Added `OdolloProvenanceConfig` and `route_transcript.py --odollo-provenance`.
- Default Odollo profiles are the two configured production tenants: `soylei-prod` and `saber-prod`.
- Added repeated `--odollo-profile` selectors plus command, repo, config, timeout, limit, contact, and log-note flags.
- Odollo contact and log-note searches use compact meeting/readout/attendee terms, not raw transcript text.
- Route provenance can now include `odollo_contact` and `odollo_log_note` sources.
- Odoo log-note bodies may be searched for matches but are not stored in route provenance snippets or metadata.
- Updated README, roadmap, and P04 plan text to mark Odollo provenance as implemented and evidence-only.

Validation:

- Graphiti runtime doctor was healthy before the routing change.
- Graphiti discovery against `transcribe_audio_main` returned existing repo routing context and no prior Odollo route adapter implementation.
- Odollo config inspection found the production profiles `soylei-prod` and `saber-prod`; the repository also has dev/test profiles that are not used by default.
- Added tests for Odollo compact query terms, Odollo source conversion, and route integration.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 33 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py tests/test_context_sources.py tests/test_routing_artifacts.py` passed.
- Live Odollo doctor check reported SoyLei production Odoo readiness `ready`.
- Live Odollo doctor check reported Saber production Odoo readiness `degraded` because Amazon product fields/views are missing, but the contact/log-note read path used by this adapter worked.
- Live read-only Odollo route smoke on the SoyLei/Tempo transcript/readout selected `SoyLei Tempo Chemical technical collaboration`, had `review_required=false`, and added 12 Odollo provenance sources across `soylei-prod` and `saber-prod`.

Next:

- Move into contextual reread source fetching for selected routes, using calendar, gws, Graphiti, and Odollo provenance packs as cited inputs.

## Turn 23 | 2026-05-11

Summary: Added contextual reread generation for selected P04 routes.

Action:

- Added `contextual_reread.py` to generate upgraded readouts from transcript, prior readout, and route decision artifacts.
- Contextual rereads reuse the existing `openai-compatible` and `codex-exec` provider paths.
- Supporting context is built from the selected candidate's cited provenance sources plus calendar context unless `--all-provenance` is passed.
- `Readout` JSON now records `contextualization.supporting_context_sources`.
- Readout Markdown now renders supporting context sources.
- The readout prompt now accepts `prior_readout`, `route_decision`, and `supporting_context` blocks and instructs providers to cite source labels or ids in evidence fields.
- Updated README, roadmap, and P04 plan text.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned older P04 planning context; repo files remained the authority for this slice.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 37 tests.
- `python -m py_compile context_sources.py contextual_reread.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- Full `codex-exec` contextual reread smoke on the SoyLei/Tempo transcript/readout/route succeeded.
- Smoke output: `/tmp/transcribe-contextual-reread-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Smoke output included 10 supporting context sources: 4 `odollo_contact`, 3 `odollo_log_note`, and 3 `gws_calendar_overlap` sources.

Next:

- Add deeper Google Drive/Docs content fetch for selected route sources or move to P05 deposition/memory harvest preview contracts.

## Turn 24 | 2026-05-11

Summary: Started P05 with a no-write deposition and memory-harvest preview contract.

Action:

- Added `deposition_artifacts.py` with `DepositAction`, `MemoryHarvestCandidate`, and `DepositPreview`.
- Added `deposition_preview.py` to generate `*.deposit-preview.json` from readout/contextual-readout artifacts.
- Preview actions can describe local filesystem, Google Drive, and Odoo targets but always use `status=preview` and `writes_enabled=false`.
- Memory harvest candidates are extracted only from structured readout `memory_candidates`.
- Transcript artifact paths are excluded from deposition action source paths unless `--include-transcript` is explicitly passed.
- Raw transcript text is never harvested into memory candidates.
- Moved P05 to OPEN in roadmap and plan docs.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` confirmed P05 was still a planned lane and returned repo memory policy facts.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py -q` passed with 41 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- No-write preview smoke over the SoyLei/Tempo contextual readout wrote `/tmp/transcribe-deposition-preview-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json`.
- Smoke output contained 3 preview actions: local filesystem, Google Drive, and Odoo record.
- Smoke output contained 7 memory candidates, all with `status=preview`.
- Smoke output did not include the transcript artifact path in deposition action source paths by default.

Next:

- Add the first apply path for local filesystem deposition, keeping Drive/Odoo/Graphiti as preview-only until their write contracts are explicitly selected.

## Turn 25 | 2026-05-11

Summary: Added local filesystem apply for reviewed deposition previews.

Action:

- Added `DepositApplyResult`, `AppliedDepositAction`, and `AppliedDepositFile` result schemas.
- Added `deposition_apply.py` to consume `*.deposit-preview.json` artifacts.
- Local apply handles only `local_filesystem` actions.
- Google Drive, Odoo, and other non-local actions are skipped with `writes_enabled=false`.
- Apply refuses previews with `review_required=true` unless `--allow-review-required` is passed.
- Local copies are idempotent: same destination hash is skipped, conflicting filenames are versioned.
- Updated README, roadmap, and P05 plan text.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing P05 planning and memory-policy facts, but no newer local apply contract.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py -q` passed with 45 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- Local apply smoke over the SoyLei/Tempo preview wrote `/tmp/transcribe-deposition-apply-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-apply.json`.
- First local apply copied the preview's readout and route artifacts into the preview local target.
- Second local apply skipped both local files by same-hash idempotency.
- Google Drive and Odoo preview actions remained skipped during apply.

Next:

- Add a reviewed Graphiti memory-harvest apply path with duplicate preflight, keeping raw transcript text excluded and Drive/Odoo still preview-only.

## Turn 26 | 2026-05-11

Summary: Added user-scoped transcript/readout store and search.

Action:

- Added `transcript_store.py` with a SQLite runtime store under `~/.transcripts`.
- Store artifacts are copied under `~/.transcripts/artifacts/`.
- The database indexes transcript artifacts, first-pass readouts, and contextual readouts.
- Added SQLite FTS5 lexical search.
- Added deterministic local token-hash embeddings for semantic-style ranking without an external provider dependency.
- Added `summarize_transcript.py --store` and `contextual_reread.py --store`.
- Added transcription opt-in with `TRANSCRIPTS_STORE=true` and optional `TRANSCRIPTS_STORE_DIR`.
- Added P08 roadmap and plan docs for the user-scoped store/search lane.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing artifact/readout facts but no prior store implementation.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 49 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_store.py` passed.
- Temp-store smoke ingested the SoyLei/Tempo transcript, first-pass readout, and contextual readout into `/tmp/transcripts-store-smoke`.
- Temp-store search for `Tempo Chemical concrete sealer` returned 3 results: readout, contextual readout, and transcript.
- Initialized the actual user-scoped store at `/home/ecochran76/.transcripts/transcripts.sqlite3`.
- Ingested the same three validated SoyLei/Tempo artifacts into `/home/ecochran76/.transcripts`.
- Real-store search for `Tempo Chemical concrete sealer` returned 3 results.

Next:

- Add watcher config support for automatic store ingestion and then backfill recent artifacts from Downloads into `/home/ecochran76/.transcripts`.

## Turn 27 | 2026-05-11

Summary: Replaced token-hash semantics with provider-backed embeddings.

Action:

- Inspected adjacent `../imcli` and `../ragmail` embedding patterns.
- Updated `transcript_store.py` to default to local Ollama embeddings with `ollama/nomic-embed-text`.
- Added `openai-compatible` embedding support with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- Kept `debug-hash` and `hash` only as explicit test/offline fallbacks.
- Stored embedding provider/model metadata on documents and filtered semantic search to matching provider/model rows.
- Added `nomic-embed-text` `search_document:` and `search_query:` formatting for ingest and search.
- Added long-document chunking with averaged document vectors so full transcripts do not overflow the embedding provider context.
- Updated README, roadmap, and P08 plan docs to reflect the real embedder path.

Validation:

- Graphiti runtime doctor was healthy.
- Graphiti discovery against `transcribe_audio_main` returned older artifact/readout facts but no store/embedder implementation.
- `ollama list` showed `nomic-embed-text:latest` available locally.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 7 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 52 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_store.py` passed.
- Initial live Ollama smoke failed on the full transcript with `the input length exceeds the context length`; chunking fixed that failure.
- Temp-store smoke ingested the SoyLei/Tempo transcript, first-pass readout, and contextual readout into `/tmp/transcripts-store-ollama-smoke` with Ollama/Nomic vectors.
- Temp-store search for `Tempo Chemical concrete sealer` returned 3 results with `embedding_provider=ollama` and `embedding_model=ollama/nomic-embed-text`, ranking the contextual readout first.
- Re-ingested the same three validated SoyLei/Tempo artifacts into `/home/ecochran76/.transcripts` with Ollama/Nomic vectors.
- Real-store search for `Tempo Chemical concrete sealer` returned 3 Ollama/Nomic-backed results, ranking the contextual readout first.

Next:

- Backfill recent Downloads artifacts into `/home/ecochran76/.transcripts`, then add watcher config support for automatic store ingestion.

## Turn 28 | 2026-05-11

Summary: Added watcher store ingestion and backfilled recent artifacts.

Action:

- Added a watcher job `store` config block with `enabled`, `store_dir`, `embedding_provider`, and `embedding_model`.
- The watcher now ingests successful transcript artifacts and generated readouts into the transcript store after transcription/readout processing.
- Watcher state now preserves `store_paths` for auditability.
- Updated `watch_transcriptions.json.sample`, README, roadmap, and P08 plan docs.
- Enabled store ingestion in the live `watch_transcriptions.json` job for `downloads-mobile-recordings`.
- Tightened embedding chunk size to avoid local Ollama/Nomic context overflow during larger recent-artifact backfills.
- Backfilled 9 recent Downloads transcript artifacts into `/home/ecochran76/.transcripts`.
- Backfilled 4 recent readout/contextual-readout artifacts from temp smoke locations into `/home/ecochran76/.transcripts`.
- Restarted `transcribe-watch.service` so the new watcher store config is active.

Validation:

- Graphiti runtime doctor was healthy.
- Graphiti discovery against `transcribe_audio_main` confirmed watcher artifact-path facts but no prior store-ingestion implementation.
- Watcher config parsed with `store_enabled=True`, `store_dir=/home/ecochran76/.transcripts`, `embedding_provider=ollama`, and `embedding_model=ollama/nomic-embed-text`.
- `/home/ecochran76/.transcripts/transcripts.sqlite3` now has Ollama/Nomic rows for 9 transcripts, 3 readouts, and 1 contextual readout.
- Store search for `SIP WMA Hamburg` returned Ollama/Nomic-backed results from the recent Downloads backfill.
- `systemctl --user restart transcribe-watch.service` succeeded.
- `systemctl --user status transcribe-watch.service --no-pager` showed the service active with PID 78937 and loaded job `downloads-mobile-recordings`.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 55 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.

Next:

- Add a store backfill command that can enumerate artifact paths, dry-run counts, and skip/ingest deterministically without shell one-liners.

## Turn 29 | 2026-05-11

Summary: Added deterministic transcript-store backfill command.

Action:

- Added `transcript_store.py backfill` for files or directories.
- Backfill discovers `*.transcript.json`, `*.readout.json`, and `*.contextual.readout.json` by default.
- Added `--dry-run`, `--modified-within-days`, repeated `--kind`, repeated `--pattern`, `--recursive`, `--limit`, and `--force`.
- Backfill reports selected counts by kind and status.
- Current artifacts with matching source path, artifact hash, embedding provider, and embedding model are skipped without re-embedding unless `--force` is used.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was degraded: MCP HTTP was down, while FalkorDB and local inspector were healthy. Work continued from repo authority.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 10 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live dry-run over `/mnt/c/Users/ecoch/Downloads --modified-within-days 14` selected 9 transcript artifacts and reported all 9 as `skip`.
- Live apply over the same Downloads path selected 9 transcript artifacts and skipped all 9 without re-embedding.

Next:

- Add chunk-level storage/retrieval so long transcripts can return precise segment hits instead of only averaged document-level semantic scores.

## Turn 30 | 2026-05-11

Summary: Added chunk-level semantic retrieval.

Action:

- Added `document_chunks` table with per-document chunk text, vector JSON, provider, and model metadata.
- Ingest now embeds chunks and stores both averaged document vectors and per-chunk vectors.
- Search now scores document-level semantic matches and best chunk-level semantic matches.
- Search results include `chunk_semantic_score` and `best_chunk` with chunk index, score, and snippet.
- Backfill status now treats documents without matching chunk rows as `update`, so older rows can be migrated by re-running backfill.
- Backfill now reports invalid matching files as `error` instead of aborting the whole scan.
- Added default excludes for copied store internals and support for additive `--exclude` patterns.
- Cleaned the live store after an overly broad recursive `/tmp` apply ingested pytest fixture artifacts; removed 54 unintended rows and restored the live store to the intended 13 documents.
- Rebackfilled recent Downloads transcripts to populate chunk rows.
- Rebackfilled known recent `transcribe-*` readout/contextual artifacts to populate chunk rows.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no chunk-level store implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 15 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live Downloads dry-run selected 9 transcripts as `update` before chunk migration.
- Live Downloads apply updated 9 transcripts and populated chunk rows.
- Live store now has 13 documents and 369 chunk rows: 9 transcripts, 3 readouts, and 1 contextual readout.
- Search for `SIP WMA Hamburg` returned `best_chunk` snippets and chunk semantic scores.
- Broad `/tmp` dry-run with `--exclude '*/pytest-of-*/*'` selected only 5 non-pytest transcribe/readout candidates; 4 were already current and 1 reviewed-preview duplicate would insert if applied.

Next:

- Add transcript-aware chunk metadata for utterance timestamp ranges and speaker spans so `best_chunk` can point back to recording time, not just text.

## Turn 31 | 2026-05-12

Summary: Added transcript timestamp and speaker metadata to store chunks.

Action:

- Added chunk character offsets during text chunking.
- Added `metadata_json` to `document_chunks`.
- Transcript chunk metadata now includes `char_start`, `char_end`, `start_seconds`, `end_seconds`, `speakers`, and `utterance_count` when structured utterances are available.
- `best_chunk` search results now surface timestamp range, speaker list, utterance count, and full metadata.
- Backfill status now marks transcript rows as `update` when matching chunks exist but lack timestamp/speaker metadata.
- Updated README, roadmap, and P08 plan docs.
- Rebackfilled 9 recent Downloads transcripts to populate timestamp/speaker metadata in live chunk rows.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no transcript chunk metadata implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 16 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live Downloads dry-run selected 9 transcripts as `update` before metadata migration.
- Live Downloads apply updated 9 transcripts.
- Live store now has 13 documents, 369 chunk rows, and 247 transcript chunks with non-empty metadata.
- Live search for `Hamburg sample` returned `best_chunk` with `start_seconds`, `end_seconds`, `speakers`, and `utterance_count`.
- Follow-up live Downloads dry-run reported all 9 transcripts as `skip`.

Next:

- Add a user-facing `open`/`context` command that can take a search result document/chunk and show nearby transcript context or media timestamp instructions.

## Turn 32 | 2026-05-12

Summary: Added transcript-store context view for search hits.

Action:

- Added `transcript_store.py context <document-id>` with optional `--chunk-index` and `--context-chunks`.
- The context view prints document metadata, artifact path, media path, timestamp range, speaker list, an `ffplay -ss` seek hint, and nearby chunk text.
- Added JSON-format support through `--format json` and a `TRANSCRIPT_CONTEXT_JSON=` sentinel for CLI automation.
- Added tests for timestamp/media context extraction and CLI output.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no prior context command implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 18 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 66 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live search for `Hamburg sample --kind transcript` returned document `1711b25666b79b3142d1` with `best_chunk.chunk_index=5`.
- Live context smoke for document `1711b25666b79b3142d1` chunk `5` printed chunks 4-6, timestamp `08:02.96 - 10:04.86`, speakers `A, B`, and a media seek hint for the original `.m4a`.

Next:

- Add a convenience flow that pipes a search result directly into `context`, so operators do not have to copy the document id and chunk index manually.

## Turn 33 | 2026-05-12

Summary: Added direct search-to-context shortcut.

Action:

- Added `transcript_store.py search --context`.
- Added `--context-rank` to choose a 1-based search hit and `--context-chunks` to control the nearby transcript window.
- Kept normal `search` output unchanged unless `--context` is passed.
- Added CLI test coverage for opening the best search chunk directly.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts, but no prior search-to-context implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 19 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 67 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-chunks 1` opened document `1711b25666b79b3142d1`, chunk `5`, timestamp `08:02.96 - 10:04.86`, and printed the media seek hint.

Next:

- Add compact JSON output for machine consumers of the context view, including selected search metadata when `search --context` is used.

## Turn 34 | 2026-05-12

Summary: Added compact JSON context output.

Action:

- Added `context --format compact-json` for pure single-line JSON without a sentinel.
- Added `search --context --context-format compact-json`.
- Search-to-context compact output includes `query`, `result_count`, `selected_rank`, `selected_result`, and the full `context` payload.
- Added test coverage for both compact JSON entrypoints.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts, but no prior compact context JSON implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 21 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 69 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | jq ...` parsed cleanly and returned document `1711b25666b79b3142d1`, chunk `5`, timestamp `08:02.96`, and the media seek hint.
- Live smoke `transcript_store.py context 1711b25666b79b3142d1 --chunk-index 5 --format compact-json | jq ...` parsed cleanly and returned the expected document, chunk, and timestamp.

Next:

- Add a small CLI recipe for piping compact context JSON into downstream routing/readout tools.

## Turn 35 | 2026-05-12

Summary: Added compact context downstream recipe helper.

Action:

- Added `scripts/context_packet_recipe.py`.
- The helper reads direct `context --format compact-json` packets or `search --context --context-format compact-json` packets from a file or stdin.
- The helper validates `context.document.source_path` and prints non-mutating shell commands for `summarize_transcript.py`, `route_transcript.py`, and `contextual_reread.py`.
- Added `--readout`, `--route`, `--provider`, `--model`, `--store`, and `--with-provenance` options.
- Added tests for stdin packets and explicit readout/route paths.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned existing readout/routing facts but no prior compact context recipe helper.
- `.venv/bin/python -m pytest tests/test_context_packet_recipe.py -q` passed with 2 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py -q` passed with 71 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live pipe smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | scripts/context_packet_recipe.py - --store --with-provenance` printed a valid downstream recipe for document `1711b25666b79b3142d1`, chunk `5`.

Next:

- Add an apply-style helper that can optionally execute the recipe steps while preserving preview/apply boundaries.

## Turn 36 | 2026-05-12

Summary: Added preview-first context packet apply helper.

Action:

- Added `scripts/context_packet_apply.py`.
- The helper reads compact context packets from stdin or a file.
- Preview is the default; downstream commands execute only when `--apply` is present.
- Execution runs first-pass readout when `--readout` is absent, routing when `--route` is absent, then contextual reread with the resolved artifact paths.
- The helper captures `READOUT_JSON=...`, `ROUTE_DECISION_JSON=...`, and `CONTEXTUAL_READOUT_JSON=...` stdout sentinels.
- Added tests for preview mode, existing-path skips, and fake-runner execution.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Re-read runtime/tenant, memory/routing, and git/validation policies before implementing the apply boundary.
- Graphiti runtime doctor was healthy and discovery returned existing readout/routing facts but no prior preview/apply helper.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py tests/test_context_packet_recipe.py -q` passed with 5 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 74 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live preview smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | scripts/context_packet_apply.py - --store --with-provenance` printed a preview plan for document `1711b25666b79b3142d1`, chunk `5`, without executing downstream commands.

Next:

- Add an artifact manifest for executed context-packet apply runs so generated readout, route, and contextual-readout paths are captured in one durable JSON record.

## Turn 37 | 2026-05-12

Summary: Added executed-run manifests for context packet apply.

Action:

- Updated `scripts/context_packet_apply.py` to write a manifest after successful `--apply` runs.
- Default manifest directory is `~/.local/state/transcribe-audio/context-packet-runs/`, keeping live operator state out of the repo.
- Added `--manifest-dir` and `--no-manifest`.
- Manifest schema captures transcript path, query, selected store document/chunk, generated readout/route/contextual-readout paths, and sanitized step metadata.
- Manifest intentionally omits raw context chunks and command stdout/stderr.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Re-read runtime/tenant, memory/routing, and git/validation policies before adding runtime manifests.
- Graphiti runtime doctor was healthy and discovery returned older artifact-path facts but no prior context-packet apply manifest implementation.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py -q` passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 74 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Temp live `--apply` smoke used fake downstream scripts from a temp directory, wrote a manifest under the temp manifest dir, captured fake readout/route/contextual-readout paths, and confirmed the manifest step records omit raw `stdout`.

Next:

- Add a manifest listing command so operators can inspect recent context-packet apply runs without browsing the runtime directory manually.

## Turn 38 | 2026-05-12

Summary: Added manifest listing for context packet apply runs.

Action:

- Added `scripts/context_packet_apply.py --list-manifests`.
- Added `--limit` and JSON/text output support for manifest lists.
- Listing reads sanitized manifest summaries from `--manifest-dir` without reading raw context chunks or command stdout/stderr.
- Added test coverage for recent-manifest listing.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts but no prior context-packet manifest listing implementation.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py -q` passed with 4 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 75 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Temp manifest-list smoke read one demo manifest through `context_packet_apply.py --list-manifests --format json` and returned count `1`, run id `demo`, and the contextual readout path.

Next:

- Review P08 definition-of-done and decide whether to close the transcript store/search lane or keep it open for UI/operator polish.

## Turn 39 | 2026-05-12

Summary: Closed P08 transcript store/search lane.

Action:

- Reviewed P08 definition of done against current implementation and runbook validation.
- Closed P08 in `ROADMAP.md`.
- Closed `docs/dev/plans/0008-2026-05-11-transcript-store-search.md`.
- Recorded that future UI/operator polish should be tracked separately rather than holding the core store/search lane open.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded and returned only older roadmap/source facts.
- P08 definition of done is satisfied: user-scoped store initializes without repo secrets, transcript/readout/contextual-readout artifacts ingest and copy into the store, lexical/semantic search returns ranked JSON, and watcher/service flows can opt into automatic ingestion.
- Latest focused suite evidence remains Turn 38: 75 tests passed plus py_compile.

Next:

- Move back to P04/P05 integration work: run one real reviewed context-packet apply on a known transcript/readout pair, then feed the generated contextual readout into deposition preview.

## Turn 40 | 2026-05-12

Summary: Ran a real P04/P05 context-packet apply through deposition preview.

Action:

- Selected the stored SoyLei/Tempo transcript/readout pair from `~/.transcripts`.
- Generated a compact context packet from `transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json --context-chunks 1`.
- Fixed `scripts/context_packet_apply.py` child interpreter selection so downstream commands default to repo `.venv/bin/python` when present instead of the parent `sys.executable`.
- Added `--python` for explicit child interpreter override.
- Added `--provider-timeout` so `codex-exec` provider calls can have a longer request timeout than the default 120 seconds.
- Re-ran the reviewed context-packet apply with read-only `gws`, Graphiti, and Odollo provenance and an existing readout.
- Fed the generated contextual readout and route into `deposition_preview.py` with no writes enabled.
- Updated README, ROADMAP, and P04/P05 plan docs with the integration evidence and remaining provenance-quality risk.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- Initial apply exposed two integration issues: child Python lacked `requests`, and `codex-exec` contextual reread hit its default 120-second timeout.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py` passed with 4 tests after the wrapper fixes.
- Full `.venv/bin/python -m pytest` passed with 75 tests.
- Successful apply manifest: `/home/ecochran76/.local/state/transcribe-audio/context-packet-runs/2026-05-12T16-04-03Z-930f0fc94f9abe19d050.json`.
- Generated route: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.route.json`.
- Generated contextual readout: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Deposition preview: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json`.
- Route selected `SoyLei Tempo Chemical technical collaboration` with confidence `0.95` and `review_required=false`.
- Deposition preview produced one local-filesystem copy action with `writes_enabled=false` and six Graphiti memory-harvest candidates.
- Integration risk: some retrieved supporting provenance was broad/noisy, so source-quality filtering should precede unattended deposition or live memory harvest.

Next:

- Add provenance-source quality filtering and route/readout warnings so irrelevant Graphiti/Odollo context cannot silently flow into contextual rereads or memory-harvest previews.

## Turn 41 | 2026-05-12

Summary: Added provenance-source quality filtering for routing and contextual rereads.

Action:

- Added compact provenance quality terms derived from calendar summary, participants, readout title/topics, and matter candidates without using raw transcript text.
- Added `filter_provenance_sources()` to retain calendar sources by default and require non-calendar sources to match enough meeting-specific terms.
- Ignored retrieval-control metadata such as Graphiti query strings, Odollo matched-term lists, and quality annotations during source scoring.
- Added `--provenance-quality-threshold` and `--no-provenance-quality-filter` to `route_transcript.py`.
- Extended route provenance packs with `excluded_sources` and `warnings`.
- Added quality metadata to retained/excluded provenance sources: `quality_status`, `quality_score`, `quality_matched_terms`, and `quality_reason`.
- Propagated route warnings and excluded-source counts into contextual reread support packets and readout contextualization metadata.
- Rendered contextual warning sections in readout Markdown.
- Updated README, ROADMAP, and P04/P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_readouts.py` passed with 23 tests.
- `.venv/bin/python -m py_compile context_sources.py route_transcript.py routing_artifacts.py contextual_reread.py readout_artifacts.py tests/test_context_sources.py tests/test_readouts.py` passed.
- Live route smoke over the SoyLei/Tempo transcript/readout with `--gws-provenance --graphiti-provenance --odollo-provenance` wrote `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.route.json`.
- The live route retained 7 calendar-derived sources and excluded 35 weak Graphiti/Odollo sources below threshold 2.
- The live route warning is `Excluded 35 provenance source(s) below quality threshold 2.`
- A direct contextual support build from the live route carried `excluded_source_count=35` and the same warning forward.

Next:

- Add source-type-specific provenance scoring so true Odollo/Drive/Graphiti hits can be retained on stronger evidence than generic term overlap, then rerun a contextual reread/deposition preview smoke with the warning surface visible.

## Turn 42 | 2026-05-12

Summary: Added source-type-specific provenance scoring and deposition preview warnings.

Action:

- Tightened provenance quality scoring to use source-type-specific identity fields.
- Drive sources now score on file name/snippet plus limited file identity metadata, not the Drive query.
- Odollo contacts now score on contact label/snippet/email/company, not the search matched-term list.
- Odollo log notes now score on note label/snippet and related record identifiers, not author/date/body or matched-term metadata.
- Graphiti sources now score on labels/previews and limited source descriptions, not the original discovery query.
- Added quality reason profiles such as `drive_file_identity`, `odollo_contact_identity`, `odollo_log_note_subject`, and `graphiti_label_or_preview`.
- Added deposition preview `warnings` so route/contextual warnings remain visible at the no-write deposition review point.
- Updated README, ROADMAP, and P04/P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- `.venv/bin/python -m pytest tests/test_context_sources.py` passed with 12 tests after the scoring change.
- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_deposition_preview.py` passed with 16 tests after adding preview warnings.
- `.venv/bin/python -m py_compile context_sources.py deposition_artifacts.py deposition_preview.py tests/test_context_sources.py tests/test_deposition_preview.py` passed.
- Live SoyLei/Tempo route smoke retained 7 calendar-derived sources and excluded 35 weak Graphiti/Odollo sources with source-profile quality reasons.
- Live contextual reread with `codex-exec --timeout 600` regenerated `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Contextual readout metadata carries `excluded_source_count=35`, 7 supporting context sources, and warning `Excluded 35 provenance source(s) below quality threshold 2.`
- Contextual Markdown renders `## Context Warnings`.
- Regenerated deposition preview `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json` now includes preview warnings.

Next:

- Add a reviewed memory-harvest approval/apply contract for Graphiti candidates, keeping live writes disabled until a preview artifact is explicitly approved.

## Turn 43 | 2026-05-12

Summary: Added reviewed Graphiti memory-harvest apply contract.

Action:

- Added `memory_harvest_apply.py`.
- Default mode is a dry-run preview that writes `*.memory-harvest-apply.json`.
- Live Graphiti writes require both `--apply` and `--approval-token APPROVE_GRAPHITI_MEMORY_HARVEST`.
- The CLI refuses `review_required` previews unless `--allow-review-required` is passed after review.
- The CLI refuses warning-bearing previews unless `--allow-warnings` is passed after review.
- Added repeated `--candidate-id` filtering for reviewed subsets.
- Apply command bodies are built only from structured deposition-preview `memory_candidates`; raw transcript text is not read or harvested.
- Dry-run mode does not write temporary memory body files.
- Added `AppliedMemoryHarvestCandidate` and `MemoryHarvestApplyResult` schemas.
- Updated README, ROADMAP, and P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded and returned older P05 policy/planning facts.
- `.venv/bin/python -m pytest tests/test_memory_harvest_apply.py` passed with 5 tests.
- `.venv/bin/python -m py_compile memory_harvest_apply.py deposition_artifacts.py tests/test_memory_harvest_apply.py` passed.
- Live dry-run refusal over the SoyLei/Tempo deposit preview failed as intended because the preview carries warnings.
- Live dry-run with `--allow-warnings` wrote `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-apply.json`.
- The live dry-run result has `mode=preview`, six planned candidates, warnings carried forward, and planned `graphiti-runtime benchmark-write` commands.
- No live Graphiti memory write was executed in this slice.

Next:

- Review the six planned SoyLei/Tempo memory candidates and, if acceptable, run one approved single-candidate Graphiti apply with queue/readback verification.

## Turn 44 | 2026-05-12

Summary: Applied one reviewed SoyLei/Tempo memory candidate to Graphiti.

Action:

- Reviewed the six structured memory candidates from the SoyLei/Tempo deposition preview.
- Selected only candidate `3a80941071fe9036` for the first live smoke because it captures durable relationship/matter context without pricing details.
- Ran a duplicate-oriented Graphiti discovery preflight for the Tempo/SoyLei matter query; it returned only existing repo/project memories and no existing Tempo/SoyLei matter episode.
- Applied the selected candidate with `memory_harvest_apply.py --apply --approval-token APPROVE_GRAPHITI_MEMORY_HARVEST --allow-warnings`.
- Wrote the live audit artifact to `/home/ecochran76/.local/state/transcribe-audio/memory-harvest-runs/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-apply.json`.

Validation:

- Graphiti `doctor` was degraded only because Inspector API was down; MCP HTTP and FalkorDB were healthy.
- `graphiti-runtime queue` was empty/unlocked before the apply.
- The live apply returned `episode_uuid=a39add00-d8d7-4d76-a1f9-855802ffb680` in group `transcribe_audio_main`.
- Post-apply `graphiti-runtime queue` returned empty/unlocked with the transcribe-audio worker running.
- Post-apply `graphiti-runtime discover --group-id transcribe_audio_main` found the new episode plus extracted facts and nodes for Tempo Chemical, SoyLei SIP 1132 concentrate, and SoyLei 1119 emulsion.

Next:

- Add an operator review workflow that can approve or reject individual memory candidates from a preview artifact, then batch-apply only accepted candidates with duplicate checks and per-candidate audit status.

## Turn 45 | 2026-05-12

Summary: Added per-candidate memory-harvest review files and duplicate audit status.

Action:

- Added `memory_harvest_apply.py --init-review` to create `*.memory-harvest-review.json` templates from deposition previews.
- Added `--review-file` support so only `approved` candidates are eligible for live Graphiti writes.
- Recorded `rejected`, `pending`, and missing-review candidates as non-written candidate statuses in the apply audit.
- Added default per-candidate Graphiti discovery duplicate preflights during `--apply`.
- Exact same-candidate replays are skipped with status `duplicate_skipped`; possible duplicate metadata is retained under each candidate's `duplicate_check`.
- Failed duplicate preflights stop that candidate with `duplicate_check_failed` before any write attempt.
- Extended `AppliedMemoryHarvestCandidate` with review decision/reason and duplicate-check metadata.
- Updated README, ROADMAP, and the P05 plan.

Validation:

- Graphiti `doctor` was degraded only because Inspector API was down; MCP HTTP and FalkorDB were healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main` found the prior live SoyLei/Tempo memory episode, confirming the duplicate-check surface is visible to Graphiti reads.
- `.venv/bin/python -m pytest tests/test_memory_harvest_apply.py -q` passed with 9 tests.
- `.venv/bin/python -m py_compile memory_harvest_apply.py deposition_artifacts.py tests/test_memory_harvest_apply.py` passed.
- `.venv/bin/python -m pytest -q` passed with 87 tests.
- Live no-write `--init-review` over the SoyLei/Tempo deposition preview wrote `/home/ecochran76/.local/state/transcribe-audio/memory-harvest-runs/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-review.json`.
- The live review template contains six candidates, all initialized to `pending`.

Next:

- Review the generated SoyLei/Tempo memory-harvest review file, approve/reject candidates, then run a batch apply for approved non-duplicate candidates.

## Turn 46 | 2026-05-12

Summary: Planned the React + Vite transcript review console.

Action:

- Inspected repo planning, runtime/tenant, architecture, and memory-routing policies before changing the roadmap.
- Ran Graphiti discovery for existing frontend/review-console context in `transcribe_audio_main`.
- Reviewed `../previews` access-control and review-sharing docs for single-operator login, scoped share links, and feedback semantics.
- Reviewed `../buffer-cli/frontend` for the sticky navbar, animated left pane, central table viewport, and animated right inspector pane pattern.
- Added `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md`.
- Added P09 to `ROADMAP.md`.

Validation:

- Graphiti `doctor` was healthy.
- Graphiti discovery returned existing repo/project memory and no conflicting frontend plan.
- The P09 plan includes the required navbar, left pane, central viewport, and right pane responsibilities.
- The plan keeps blobs, share tokens, tenant credentials, and live runtime state under `~/.transcripts` or `~/.local/state/transcribe-audio/`.

Next:

- Start P09 slice 1 by adding the backend read API contract and minimal local API service for library/search/detail plus blob range playback before scaffolding the React shell.

## Turn 47 | 2026-05-12

Summary: Added the first local transcript review API and blob playback contract.

Action:

- Added blob storage tables to the transcript store: `blobs` and `document_blobs`.
- Updated transcript ingestion to copy existing source recordings into `~/.transcripts/blobs/` when the transcript artifact points to an available source/working media file.
- Added compact `media_blob` metadata with playback/download URLs for frontend use.
- Added `transcript_api.py`, a read-only local HTTP API over the user-scoped store.
- Implemented `/api/health`, `/api/library`, `/api/search`, `/api/documents/<id>`, `/api/documents/<id>/context`, and range-capable `/api/blobs/<blob_id>`.
- Added `docs/dev/transcript-review-api.md` and updated README, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with 3 tests.
- `.venv/bin/python -m py_compile transcript_api.py transcript_store.py tests/test_transcript_api.py` passed.
- `.venv/bin/python -m pytest -q` passed with 90 tests.
- Read-only live-store smoke through `transcript_api.list_documents(root=~/.transcripts, limit=5)` returned 14 total documents and 5 listed documents.
- The live-store smoke found no media blobs among the first 5 existing documents because older rows predate blob registration; re-backfill/update is needed to populate blob links for already-ingested transcripts.

Next:

- Add a safe store migration/backfill path that updates existing transcript rows with blob pointers, then run a dry-run/apply over recent transcripts before scaffolding the React shell.

## Turn 48 | 2026-05-12

Summary: Added dry-run-first legacy transcript import for historical transcripts without sidecars.

Action:

- Added `legacy_transcript_import.py`.
- The importer discovers legacy `*Transcript.txt` and `*Transcript.docx` outputs.
- Dry-run is the default; `--apply` is required to write synthesized sidecars and ingest them.
- Synthesized sidecars are written under `~/.transcripts/legacy-artifacts/`, not the repo or source folders.
- Legacy sidecars use the normal transcript artifact shape, set `backend=legacy-import`, preserve the source TXT/DOCX path under `output_paths`, and mark `legacy_import.needs_enrichment=true`.
- The importer attempts to match nearby source recordings by basename and passes matched media paths into the normal blob registration path.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 27 tests.
- `.venv/bin/python -m py_compile legacy_transcript_import.py transcript_api.py transcript_store.py tests/test_legacy_transcript_import.py` passed.
- `.venv/bin/python -m pytest -q` passed with 93 tests.
- A bounded `find` check over `~/Downloads` showed it is a symlink to `/mnt/c/Users/ecoch/Downloads` and found zero top-level legacy transcript candidates matching the default patterns.
- A Python dry-run against `~/Downloads` was killed after slow mount behavior; no writes were made.

Next:

- Point `legacy_transcript_import.py` at the real historical transcript root with `--recursive` and run a dry-run inventory, then apply the import with production embeddings and run the context/readout enrichment pipeline over rows marked `legacy_import.needs_enrichment=true`.

## Turn 49 | 2026-05-12

Summary: Imported historical transcript DOCX/TXT outputs from Downloads and Sound Recordings.

Action:

- Counted 45 legacy transcript candidates and 262 possible media files under `~/Downloads` and `/mnt/h/My Drive/Documents/Sound Recordings`.
- Added `--media-index-file` to `legacy_transcript_import.py` after direct Python media-root walks proved too slow on mounted folders.
- Generated transcript and media indexes with `find`, then ran a dry-run inventory from exact candidate paths.
- Saved dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-import-dry-run-2026-05-12.json`.
- Applied the import with `--embedding-provider ollama --embedding-model ollama/nomic-embed-text`.
- Saved apply output to `/home/ecochran76/.local/state/transcribe-audio/legacy-import-apply-2026-05-12.json`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Dry-run found 45 candidates, all `convert`, with 44 media matches.
- The only unmatched media item was `2025-07-09 Lululemon Summary and Transcript.docx`.
- Apply inserted 45 legacy transcript sidecars under `~/.transcripts/legacy-artifacts/`.
- Store verification showed 54 total transcript documents using Ollama embeddings, 45 documents marked `legacy_import.needs_enrichment=true`, 44 legacy documents linked to blobs, and 36 total blobs.
- Read-only API smoke listed 54 transcript documents.
- `transcript_store.py search "Scott Roberts" --kind transcript --limit 3` returned legacy transcript hits from the imported set.
- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 28 tests.
- `.venv/bin/python -m pytest -q` passed with 94 tests.

Next:

- Add an enrichment queue/list command for documents marked `legacy_import.needs_enrichment=true`, then run first-pass summaries and calendar/context enrichment in small batches.

## Turn 50 | 2026-05-12

Summary: Searched SoyLei Shared Drives and imported deduped additional legacy transcripts.

Action:

- Confirmed the prior `Sound Recordings` search was recursive and included subfolders such as `Transcribed/`.
- Searched these additional roots:
  - `/mnt/h/Shared drives/SoyLei Officers`
  - `/mnt/h/Shared drives/SoyLei Core Team`
  - `/mnt/h/.shortcut-targets-by-id/0B1xe-E5-InccUThWQ0QyY0Z0Tms/Documents/Corkboard/Clients/SoyLei`
- Found 65 additional transcript candidates in those SoyLei Shared Drive/shortcut roots.
- Added de-dupe behavior to `legacy_transcript_import.py` using existing source transcript hashes and normalized titles, plus within-batch duplicate detection.
- Added `--no-dedupe` for diagnostics and `--no-media-match` for fast mounted-drive transcript-only imports.
- Ran a Shared Drive dry-run with de-dupe enabled and media matching disabled.
- Saved dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-shared-drives-dry-run-2026-05-12.json`.
- Applied the deduped Shared Drive import with Ollama/Nomic embeddings and no media matching.
- Saved apply output to `/home/ecochran76/.local/state/transcribe-audio/legacy-shared-drives-apply-2026-05-12.json`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Shared Drive dry-run selected 65 candidates: 25 `convert`, 22 `duplicate_in_batch`, and 18 `duplicate_existing`.
- Shared Drive apply inserted 25 new transcripts and skipped 40 duplicates.
- Store verification showed 79 total transcript documents using Ollama embeddings, 70 documents marked `legacy_import.needs_enrichment=true`, and 36 total blobs.
- Duplicate source-hash verification returned no duplicate legacy source hashes.
- `transcript_store.py search "Ryan Jaggar" --kind transcript --limit 3` returned imported Shared Drive legacy hits.
- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 31 tests.
- `.venv/bin/python -m pytest -q` passed with 97 tests.

Next:

- Add a targeted blob-linking pass for Shared Drive legacy transcripts that need media matching, then add the enrichment queue/list command for the 70 legacy rows marked `needs_enrichment`.

## Turn 51 | 2026-05-12

Summary: Added the legacy enrichment queue and targeted media-linking pass.

Action:

- Added `transcript_store.py legacy-enrichment-queue` to list legacy rows marked `legacy_import.needs_enrichment=true`.
- Queue output supports text, JSON, compact JSON, and runnable `summarize_transcript.py` commands.
- Queue entries de-dupe same-hash and same-title rows by default to avoid duplicate provider calls.
- Added `legacy_media_link.py` to link already-imported legacy transcript sidecars to recordings from explicit media indexes or targeted media roots.
- Built a targeted SoyLei Shared Drive media index at `/home/ecochran76/.local/state/transcribe-audio/soylei-shared-media-index-2026-05-12.txt`.
- Saved media-link dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-media-link-dry-run-2026-05-12.json`.
- Applied matched media links and saved output to `/home/ecochran76/.local/state/transcribe-audio/legacy-media-link-apply-2026-05-12.json`.
- Attempted one first-pass enrichment smoke with the default OpenAI-compatible config and one with `/home/ecochran76/.auracall/api.env`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Targeted media index found 312 media files under the named SoyLei Shared Drive/shortcut roots.
- Media-link dry-run selected 26 unlinked legacy rows: 16 `link` and 10 `no_match`.
- Media-link apply updated 16 linked rows and skipped the 10 unmatched rows.
- Store verification showed 79 transcript documents, 70 legacy rows still marked `needs_enrichment`, and 60 transcript documents linked to source-recording blobs.
- The de-duped enrichment queue showed 68 pending first-pass readouts, 58 with blobs and 10 without blobs.
- Default OpenAI-compatible enrichment failed with `429 insufficient_quota`.
- AuraCall-compatible enrichment reached `127.0.0.1:18095` but timed out after 120 seconds; `summarize_transcript.py` now reports provider request failures as clean `TranscriptionError` messages instead of stack traces.
- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_legacy_media_link.py tests/test_transcript_store.py -q` passed with 38 tests.
- `.venv/bin/python -m pytest tests/test_legacy_media_link.py tests/test_transcript_store.py tests/test_legacy_transcript_import.py -q` passed with 32 tests.

Next:

- Repair or tune the AuraCall/OpenAI-compatible readout path for long legacy transcripts, then run a small enrichment batch from `transcript_store.py legacy-enrichment-queue --format commands`.

## Turn 52 | 2026-05-12

Summary: Proved the repaired AuraCall OpenAI-compatible path on one legacy enrichment smoke and ingested the readout.

Action:

- Loaded `/home/ecochran76/.auracall/api.env` and reran `summarize_transcript.py`
  on the prior failed Scott/gener8or legacy transcript using the AuraCall
  OpenAI-compatible endpoint.
- Used `--timeout 300` because browser-backed AuraCall requests can exceed the
  old 120 second client timeout under load.
- Wrote non-mutating smoke outputs under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/`.
- Ingested the generated readout JSON directly through
  `transcript_store.ingest_artifact` to avoid a duplicate provider call.
- Updated
  `docs/dev/notes/2026-05-12-auracall-legacy-enrichment-handoff.md` so the
  handoff no longer describes the AuraCall failure as current.

Validation:

- The source smoke transcript exists at
  `/home/ecochran76/.transcripts/legacy-artifacts/07/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.transcript.json`.
- The source artifact is 15,415 bytes with 14,237 transcript text characters.
- `summarize_transcript.py` returned successfully and wrote:
  - `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.json`
  - `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.md`
- The readout contained a non-empty summary plus participants, topics, action
  items, matter candidates, memory candidates, and next steps.
- Store ingest inserted readout id `017a8ffe7173998ba82d`.
- `transcript_store.py legacy-enrichment-queue --format compact-json` now
  reports 67 pending de-duped first-pass readouts, and the Scott/gener8or item
  is no longer pending.

Next:

- Run a bounded AuraCall-backed small batch from
  `transcript_store.py legacy-enrichment-queue --format commands --limit 3`
  before expanding to the remaining legacy queue.

## Turn 53 | 2026-05-12

Summary: Ran a bounded three-item AuraCall enrichment batch; two succeeded and one failed with provider error content.

Action:

- Selected three pending legacy transcript rows from `transcript_store.py legacy-enrichment-queue --format compact-json --provider openai-compatible --limit 3`.
- Saved the selected queue to `/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-queue-2026-05-12.json`.
- Loaded `/home/ecochran76/.auracall/api.env` and ran `summarize_transcript.py --provider openai-compatible --timeout 300 --store` sequentially for the selected items.
- Captured per-item logs and summary under `/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-retry-2026-05-12/`.
- Made a raw diagnostic chat-completions call for the failed item and saved the response body/content in the same batch directory.
- Updated `docs/dev/notes/2026-05-12-auracall-legacy-enrichment-handoff.md`.

Validation:

- Item 1, `20250417-142659-Ambient Workshop Recording (2025-04-17) - Non-Verbal Audio`, succeeded and stored a readout.
- Item 2, `2025-07-17 Shuana Sofia MacGill`, succeeded and stored a readout.
- Item 3, `2025-06-06 Breakfast with Nacu My recording 9`, failed twice with `OpenAI-compatible readout did not return valid JSON`.
- The raw diagnostic response for item 3 was HTTP 200 but the assistant content was provider error text: `Something went wrong. If this issue persists please contact us through our help center at help.openai.com.`
- Store verification showed 10 readout documents after the partial batch.
- The de-duped pending legacy enrichment queue reported 63 items after the partial batch.

Next:

- Do not run the full queue blindly. Either skip/quarantine the failed Nacu breakfast item and continue with a small batch of later queue entries, or add a provider fallback/condensation path for transcripts that trigger AuraCall internal-error content.

## Turn 54 | 2026-05-12

Summary: Tested a transcript excerpt-budget workaround, then identified it as
the wrong layer. Kept readout-shape validation and moved the real fix back to
AuraCall.

Action:

- Started the next five-item AuraCall-backed legacy enrichment batch.
- Tried and reverted a transcript-length limiting workaround. This repo should
  send the full transcript; AuraCall should handle large OpenAI-compatible
  browser-backed requests without reducing caller capability.
- Added OpenAI-compatible readout-shape validation so echoed prompt/input JSON
  cannot be stored as an empty `Transcript Readout`.
- Removed four bad empty readout rows created during an initial 90k-budget
  attempt: `afcd2217031c97899dcd`, `a00e322b974f8e424548`,
  `cd57df8c9935867d05ed`, and `79e84e2d09d64e6a1e6c`.
- Removed the stale local empty JSON/Markdown output for the still-pending SBIR
  item.

Validation from the workaround experiment:

- `2025-06-06 Breakfast with Nacu My recording 9` produced a valid readout.
- `2025-07-31 Nacu Breakfast My recording 17` produced a valid readout.
- `2025-04-24 Nacu Meeting USDA Grant and SoyLei Matters` produced a valid
  readout.
- `2025-07-29 Dr Warmbe Meniscus Tear consult` produced a valid readout.
- `2025-04-17 Nacu Eric Call SoyLei SBIR Matters` remains pending: one retry
  returned an empty response, and the next retry hit the 300 second client
  timeout. AuraCall reported no recent stuck runtime runs afterward.
- The pending de-duped queue now reports 59 items.
- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_transcript_store.py -q`
  passed with 38 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py transcript_store.py tests/test_readouts.py tests/test_transcript_store.py`
  passed.
- `.venv/bin/python -m pytest -q` passed with 105 tests.

Next:

- Fix AuraCall so a large full-transcript OpenAI-compatible request is
  transported through the browser service as an attachment when needed, and so
  failed AuraCall runs return an API error instead of HTTP 200 with empty
  assistant content. Then retry the SBIR item with the full transcript.

## Turn 55 | 2026-05-12

Summary: Removed the transcript-length workaround from this repo, verified the
fix belongs in AuraCall, and completed the pending SBIR readout with the full
transcript.

Action:

- Removed the `--max-transcript-chars` workaround from the readout CLI,
  generated queue commands, and tests.
- Kept readout-shape validation so malformed/empty AuraCall responses do not
  get stored as readouts.
- Updated the AuraCall handoff note to state that transcript truncation was a
  reverted experiment, not the path forward.
- Rebuilt/reinstalled AuraCall and retried the pending SBIR readout without
  transcript truncation.

Validation:

- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_transcript_store.py -q`
  passed with 37 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py transcript_store.py tests/test_readouts.py tests/test_transcript_store.py`
  passed.
- `.venv/bin/python -m pytest -q` passed with 104 tests.
- Full-transcript SBIR retry through `agent:instant-chatgpt-soylei` failed
  honestly with AuraCall HTTP 502 after AuraCall detected non-parseable JSON.
- Full-transcript SBIR retry through `agent:pro-extended-chatgpt-soylei`
  succeeded and wrote:
  `/home/ecochran76/.transcripts/legacy-artifacts/28/28d268e46f590765c413-2025-04-17 Nacu Eric Call SoyLei SBIR Matters.readout.json`
- The de-duped pending legacy enrichment queue now reports 58 items, and the
  SBIR item is no longer pending.

Next:

- Continue legacy enrichment in bounded batches using the full transcript path.
  Prefer stronger/project-specific AuraCall agents for long readout jobs when
  JSON completeness matters.

## Turn 56 | 2026-05-13

Summary: Added an AuraCall response-batch path for legacy transcript readouts
using a project-bound SoyLei Pro Extended transcripts agent.

Action:

- Added `scripts/auracall_legacy_enrichment_batch.py`.
- Added `write_readout_from_payload` so synchronous and batched readouts share
  the same JSON/Markdown materialization path.
- Added a dry-run test that verifies the batch payload uses
  `agent:pro-extended-chatgpt-soylei-transcripts`, JSON response-format
  metadata, and the SoyLei `wsl-chrome-3` runtime hints.
- Created registry agent `pro-extended-chatgpt-soylei-transcripts` with
  `projectName=Transcripts`, `service=chatgpt`,
  `runtimeProfile=wsl-chrome-3`, and `modelSelector=chatgpt:pro-extended`.
- Issued scoped client env:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- Restarted `auracall-api.service` and confirmed `/v1/models` includes
  `agent:pro-extended-chatgpt-soylei-transcripts`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_legacy_enrichment_batch_dry_run_writes_manifest tests/test_transcript_store.py::test_legacy_enrichment_queue_lists_pending_legacy_imports -q`
  passed with 2 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py scripts/auracall_legacy_enrichment_batch.py tests/test_transcript_store.py`
  passed.
- Live dry-run built:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-062622.json`.
- Live one-item enqueue created and completed
  `batch_0db1883c7905471c83d807411cfdee33` with
  `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`, and child
  response `resp_1a4b0915303848a6ab68a48e286e563f`.
- `status --materialize --store` wrote and ingested:
  `/home/ecochran76/.transcripts/legacy-artifacts/29/29ed3d64cca92a7cf5f5-2025-08-15 Dr Stefl Knee Replacement Consult.readout.json`.
- The de-duped pending legacy enrichment queue now reports 57 items.

Note:

- The earlier `POST /v1/projects/ensure` `button-missing` failure was repaired
  in AuraCall on 2026-05-13. The ChatGPT provider project now exists as
  `g-p-6a04628762ac8191894b16cfaddfd126`, and the transcript agent is bound to
  that provider project id.

## Turn 57 | 2026-05-13

Summary: Revalidated the AuraCall scoped client path for transcript readout
bursts after returning from live-follow work.

Validation:

- Scoped env:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- The scoped key can read `/v1/models` and sees
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- The running AuraCall registry shows
  `pro-extended-chatgpt-soylei-transcripts` bound to ChatGPT project
  `Transcripts` with provider project id
  `g-p-6a04628762ac8191894b16cfaddfd126`.
- Live scoped-env smoke passed:
  `pnpm run smoke:scoped-client-env -- /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env --prompt 'Reply exactly: auracall transcribe env ok' --expect-output 'auracall transcribe env ok' --timeout-ms 180000`.
- Response id `resp_45008e83347940909bcdba697b91fa2c` read back as
  `completed` with output `auracall transcribe env ok`.

Next:

- Resume bounded legacy readout batches through
  `scripts/auracall_legacy_enrichment_batch.py`.
- Keep concurrency and browser interaction limits in the AuraCall batch request;
  do not limit transcript length in this repo.

## Turn 58 | 2026-05-13

Summary: Added a repo-local handoff note so `transcribe-audio` can retake
ownership of the AuraCall-backed legacy readout batch workflow.

Action:

- Added
  `docs/dev/notes/2026-05-13-auracall-transcribe-ownership-handoff.md`.
- Recorded the current AuraCall transcript agent binding, scoped env path,
  live smoke evidence, one-item batch evidence, and next owner actions.
- Reiterated the policy boundary: Transcribe Audio owns queue selection,
  prompt construction, materialization, and store ingestion; AuraCall owns
  large prompt transport, browser execution, project binding, queueing, and
  rate limiting.

Validation:

- `graphiti-runtime doctor` reported healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main "AuraCall transcript readout batch scoped env next steps" --max-facts 5`
  returned older P03/readout context; the new handoff therefore relies on
  current repo runbook entries and live AuraCall evidence.
- `git diff --check` passed.

Next:

- Resume with the handoff note's three-item dry run, then one three-item live
  batch, then `status --materialize --store`.

## Turn 59 | 2026-05-13

Summary: Started the first three-item AuraCall response batch under
`transcribe-audio` ownership; batch remains in progress.

Action:

- Read `docs/dev/notes/2026-05-13-auracall-transcribe-ownership-handoff.md`
  and followed its owner actions.
- Ran the three-item dry run:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env enqueue --limit 3 --store --dry-run`.
- Dry-run manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092116.json`.
- Inspected the dry-run manifest and confirmed the expected model,
  JSON response-format metadata, full prompt payloads, and limits.
- Ran the first live three-item batch:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env enqueue --limit 3 --store --max-concurrent-runs 2 --max-browser-interactions-per-minute 8`.
- Live manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`.
- Polled with `status --materialize --store` several times; no children were
  complete yet, so no readouts were materialized.

Validation:

- Dry run selected 3 requests for
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- Dry-run prompt lengths were approximately 137447, 33426, and 12107
  characters; transcript payloads were not truncated.
- Dry-run limits were `maxConcurrentRuns=2` and
  `maxBrowserInteractionsPerMinute=8`.
- Live enqueue returned batch id `batch_bd9a400d785f4eeeaecf986621597091`.
- Current batch status is `running` with counts:
  `total=3`, `in_progress=3`, `completed=0`, `failed=0`, `cancelled=0`,
  `missing=0`.
- Child response ids:
  `resp_ad243a3df5bc4d61ac7934e144f4352b`,
  `resp_b35c7e03a57d4d11ad3d081d77277404`,
  `resp_9d59ac43f87f460081a187fa28c4bf49`.

Next:

- Re-run:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env status /home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json --materialize --store`.
- If children complete, verify readout artifacts and the pending queue count.
- If children fail or remain stuck, preserve the manifest and response ids and
  diagnose AuraCall rather than shortening transcripts in this repo.

## Turn 60 | 2026-05-13

Summary: Polled the first three-item AuraCall batch; it is not materializable
because one completed response has empty output, one child is still running,
and one child failed in AuraCall.

Action:

- Re-ran `status --materialize --store` for
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`.
- Materialization failed with `OpenAI-compatible readout returned an empty response`.
- Re-ran `status` without materialization to capture current batch state.
- Saved raw response diagnostics under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135-diagnostics/`.
- Re-checked the de-duped pending queue.

Validation:

- Batch id remains `batch_bd9a400d785f4eeeaecf986621597091`.
- Current counts are `total=3`, `completed=1`, `in_progress=1`,
  `failed=1`, `cancelled=0`, `missing=0`.
- Index 0, `resp_ad243a3df5bc4d61ac7934e144f4352b`, is marked completed by
  AuraCall but `/v1/responses/...` returns `output: []`, so there is no
  readout JSON to materialize.
- Index 1, `resp_b35c7e03a57d4d11ad3d081d77277404`, is still `in_progress`.
- Index 2, `resp_9d59ac43f87f460081a187fa28c4bf49`, failed with
  `runner_execution_failed: connect ETIMEDOUT 127.0.0.1:9222`.
- No readouts were materialized from this batch.
- The de-duped pending legacy enrichment queue still reports 57 items.

Next:

- Diagnose this as an AuraCall/runtime issue, not a transcript truncation issue:
  completed-empty output and `127.0.0.1:9222` timeout should be repaired or
  retried in AuraCall.
- After AuraCall-side diagnosis, retry a fresh bounded batch or add
  transcribe-side handling that skips failed/empty children while preserving
  their response ids for retry.

## Turn 61 | 2026-05-14

Summary: Retried the AuraCall batch path after the AuraCall upgrade; the old
batch is terminally failed and a fresh three-item retry batch is in progress.

Action:

- Re-polled old manifest
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`
  with `status --materialize --store`.
- Materialization still failed with `OpenAI-compatible readout returned an
  empty response`.
- Re-polled the old manifest without materialization and confirmed its terminal
  status.
- Re-checked the de-duped queue; it still reported 57 pending legacy readout
  items.
- Submitted a fresh three-item live batch using the same full transcript
  payloads, model, and limits.
- Polled the fresh batch twice; no children completed or failed yet.

Validation:

- Old batch `batch_bd9a400d785f4eeeaecf986621597091` is now `failed` with
  counts `total=3`, `completed=1`, `failed=2`, `in_progress=0`.
- Old index 0 `resp_ad243a3df5bc4d61ac7934e144f4352b` is completed but still
  has empty output, so no readout can be materialized.
- Old index 1 `resp_b35c7e03a57d4d11ad3d081d77277404` failed with
  `runner_execution_failed: ChatGPT response did not complete as a parseable
  JSON object.`
- Old index 2 `resp_9d59ac43f87f460081a187fa28c4bf49` failed with
  `runner_execution_failed: connect ETIMEDOUT 127.0.0.1:9222`.
- Fresh retry manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json`.
- Fresh retry batch id: `batch_e9b79b1474ec4cf8a622e52f5b8f7bce`.
- Fresh retry child response ids:
  `resp_56c5a0d25823456d99d97e50114fe887`,
  `resp_c073d5e002414a0c98f8ee0fe987470b`,
  `resp_618693902f244b8e8a777cff9fc38305`.
- Fresh retry status after several minutes remained `running` with counts
  `total=3`, `in_progress=3`, `completed=0`, `failed=0`, `cancelled=0`,
  `missing=0`.
- No readouts were materialized in this turn.

Next:

- Poll the fresh retry manifest with:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env status /home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json --materialize --store`.
- If it completes, verify stored readouts and queue count.
- If it fails or remains stuck for an unreasonable interval, diagnose AuraCall
  with the fresh batch and child response ids rather than changing transcript
  length.

## Turn 62 | 2026-05-14

Summary: Polled the fresh AuraCall retry batch; all three children failed with
partial JSON snapshots but no materializable response output.

Action:

- Re-polled fresh retry manifest
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json`
  with `status --materialize --store`.
- Materialization produced no readouts because the batch ended failed.
- Saved raw response diagnostics under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528-diagnostics/`.
- Re-checked the de-duped legacy enrichment queue.

Validation:

- Fresh retry batch `batch_e9b79b1474ec4cf8a622e52f5b8f7bce` is now failed
  with counts `total=3`, `completed=0`, `failed=3`, `in_progress=0`.
- Index 0 `resp_56c5a0d25823456d99d97e50114fe887` failed with
  `ChatGPT response did not complete as a parseable JSON object after waiting`;
  AuraCall captured a best snapshot of 22322 chars but `/v1/responses/...`
  still returned `output: []`.
- Index 1 `resp_c073d5e002414a0c98f8ee0fe987470b` failed with the same
  parseable-JSON completion issue and a best snapshot of 11918 chars; response
  output was empty.
- Index 2 `resp_618693902f244b8e8a777cff9fc38305` failed with the same issue
  and a best snapshot of 10358 chars; response output was empty.
- The de-duped pending legacy enrichment queue still reports 57 items.
- No transcript payloads were shortened and no readouts were stored from this
  retry batch.

Next:

- AuraCall should expose failed-run best snapshots as retrievable diagnostics or
  recoverable output artifacts, or complete JSON capture before marking the run
  failed.
- Transcribe-side next work can add retry/quarantine metadata around failed
  batch children, but should not treat partial snapshots as readouts unless
  AuraCall exposes a deliberate recovery contract.

## Turn 63 | 2026-05-14

Summary: Started a one-item retry against AuraCall's new recovery-artifact
contract; the run remains active with no output yet.

Action:

- Read AuraCall handoff
  `/home/ecochran76/workspace.local/auracall/docs/dev/notes/2026-05-14-chatgpt-json-artifact-handoff.md`.
- Confirmed the AuraCall fix is non-retroactive for the failed
  `batch_e9b79b1474ec4cf8a622e52f5b8f7bce`, so a fresh run is required.
- Enqueued a fresh one-item batch from the current pending queue using
  `agent:pro-extended-chatgpt-soylei-transcripts`, `maxConcurrentRuns=1`, and
  `maxBrowserInteractionsPerMinute=6`.
- Polled the manifest with `status --materialize --store` several times.
- Read the response object directly and saved a raw diagnostic snapshot.

Validation:

- Manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-171322.json`.
- Batch id: `batch_0973e70d5a1e4fa5a7f8f4c2ae7d1668`.
- Response id: `resp_723b789f244446159354a2e751dde7a0`.
- Selected transcript title:
  `2025-08-20 Nacu Eric Line of Business follow up meeting`.
- Batch status remains `running` with counts `total=1`, `in_progress=1`,
  `completed=0`, `failed=0`.
- Direct response read showed `status=in_progress`, `output_len=0`,
  `terminalStepId=null`, and a running step for
  `pro-extended-chatgpt-soylei-transcripts` on `wsl-chrome-3`.
- Response `lastUpdatedAt` was `2026-05-14T22:20:50.377Z`, proving the run was
  still active during this turn.
- Diagnostics path:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-171322-diagnostics/`.
- Queue still reports 57 pending items because no readout was materialized yet.

Next:

- Poll the one-item manifest again with `status --materialize --store`.
- If it completes with message JSON or an artifact output, materialize/store and
  verify the pending queue decreases.
- If it fails, inspect `/v1/responses/resp_723b789f244446159354a2e751dde7a0`
  for the new recovery artifact contract before changing transcribe prompts.

## Turn 64 | 2026-05-14

Summary: Exercised AuraCall recovery/output contracts and updated the batch
client to preserve full inline JSON while accepting future JSON artifact outputs.

Action:

- Polled the prior one-item recovery run and inspected its partial recovery
  artifact.
- Updated `scripts/auracall_legacy_enrichment_batch.py` so AuraCall requests no
  longer use `metadata.response_format`, because browser-backed ChatGPT runs do
  not reliably complete through that JSON-object path.
- Added `response_model_payload()` materialization support for both inline
  message JSON and future JSON artifact outputs.
- Tried the ChatGPT workspace-file contract by asking for `legacy_readout.json`.
- Observed that AuraCall completed the run with only
  `legacy_readout.json ready` in `/v1/responses/{id}` and no artifact entries in
  local `sharedState.artifacts`.
- Tried full inline JSON without length limits; AuraCall returned JSON-like text
  but with raw newlines, malformed nested sections, or truncation, so
  materialization correctly rejected it.
- A short capped prompt did materialize one readout successfully, but the cap was
  removed because full-fidelity readouts should not be product-limited just to
  work around provider transport.

Validation:

- Tests: `.venv/bin/python -m pytest tests/test_transcript_store.py
  tests/test_readouts.py -q` passed with 39 tests.
- Whitespace: `git diff --check` passed.
- Workspace artifact trial:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-175431.json`.
- Inline full-output trials:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-175902.json`
  and
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-180139.json`.
- Successful capped-output materialization:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-180342.json`,
  response `resp_19865f8e9d7046d68b523ea440a5a9be`, readout stored at
  `/home/ecochran76/.transcripts/legacy-artifacts/63/63eb9090a1dcc8e9a332-2025-08-20 Nacu Eric Line of Business follow up meeting.readout.json`.
- Current code intentionally does not keep the capped prompt; it preserves full
  detail and validates before storing.

Next:

- Fix AuraCall to expose ChatGPT workspace/file artifacts as response outputs, or
  add a durable attachment/output channel for large structured JSON.
- After AuraCall exposes the full readout artifact, retry one uncapped item and
  then resume the legacy enrichment batch.

## Turn 65 | 2026-05-14

Summary: Restored the legacy enrichment request contract to artifact-first
ChatGPT workspace output.

Action:

- Updated `scripts/auracall_legacy_enrichment_batch.py` so the SoyLei
  Transcripts AuraCall batch request instructs ChatGPT to create
  `legacy_readout.json` in its REPL/workspace and surface it as a downloadable
  artifact.
- Removed the inline JSON requirement from the AuraCall-specific prompt; the
  assistant response must now surface the actual `legacy_readout.json`
  downloadable artifact/link rather than a text-only readiness marker.
- Changed `metadata.outputContract.mode` from
  `inline_json_with_optional_workspace_artifact` to
  `chatgpt_workspace_artifact`.
- Kept `response_model_payload()` able to parse artifact outputs first after a
  non-JSON readiness message, while still tolerating parseable inline JSON for
  backward compatibility.

Validation:

- AuraCall-side live smoke `resp_db52dcf73b7d44b0abbffd327bbeac5c` now proves
  the browser run lands inside the SoyLei `Transcripts` project URL, but it
  still recorded `discovered=0 materialized=0` for the requested artifact.
- This transcribe-side change does not claim artifact extraction is fixed; it
  aligns the caller with the intended artifact contract so the next smoke tests
  the right behavior.
- Correction: the first artifact-first prompt still allowed a text-only
  readiness response. The prompt now explicitly says a text-only readiness note
  is not sufficient.
- Live retry `resp_6b10a6d743e84ec3a775060fda94b120` reached the SoyLei
  `Transcripts` project but ChatGPT returned the future-tense status sentence
  `I'll create the JSON readout...` with no artifact. The prompt was tightened
  again to forbid future-tense/status replies and require an actual
  `sandbox:/.../legacy_readout.json` link or native attachment card in the final
  response.
- Second retry `resp_48647ca8e7bc42979f89f20dd4778dee` produced a sandbox
  artifact link and AuraCall recorded `discovered=1 materialized=1`, but the
  response artifact was still exposed as `artifact_type=generated` without a
  local path, so the transcribe materializer rejected it.
- AuraCall was patched and reinstalled to preserve the materialized local path
  and JSON MIME metadata on response artifacts. The transcribe materializer was
  patched to accept generated/download JSON artifacts and prefer
  `metadata.localPath` over sandbox URIs.
- Final smoke `batch_ca0a6f46ed1844c0a789329bcc241053` /
  `resp_4235722877774ee79e158be3843de653` completed and materialized:
  - `/home/ecochran76/.transcripts/legacy-artifacts/5d/5d26c585ac566dc22c0d-2025-08-21 Lululemon JP Siddhant Xlinked HBAN  My recording 20.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/5d/5d26c585ac566dc22c0d-2025-08-21 Lululemon JP Siddhant Xlinked HBAN  My recording 20.readout.md`
  - API artifact evidence included `artifact_type=file`,
    `mime_type=application/json`, `disposition=attachment`, remote ChatGPT
    estuary URL, and `metadata.localPath`.

Next:

- Probe the project-bound ChatGPT conversation/artifact UI directly from
  AuraCall to decide whether ChatGPT generated `legacy_readout.json` and
  AuraCall missed it, or ChatGPT replied ready without creating a downloadable
  artifact.

## Turn 66 | 2026-05-14

Summary: Scaled the working AuraCall artifact path to a five-item legacy
enrichment batch.

Action:

- Ran a dry-run over five pending legacy transcript artifacts to confirm the
  selected slice.
- Submitted live batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-192515.json`.
- Batch id: `batch_4039e070788e4190b371d6e9be4a4627`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.
- Polled with `status --materialize --store` until completion.

Validation:

- Batch completed with `total=5`, `completed=5`, `failed=0`.
- Response ids:
  - `resp_e5550593a3eb46d59770fc3ae5acaa64`
  - `resp_779cbfa125b1435a86de0d2bab81d9f3`
  - `resp_0c524b251f514100bb162520ae41aa12`
  - `resp_61d874f762324a73a399491bb8269ad7`
  - `resp_bdb5f5b24ff540799ddf51e1de8ee69b`
- Materialized readouts:
  - `/home/ecochran76/.transcripts/legacy-artifacts/fe/feb0a84f7d5262804b3f-2025-08-22 Baker Pappajohn Pitch My recording 21.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/84/843ca41a06ab290c2a66-2025-08-26 Amazon SoyLei Bio My recording 20.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/1d/1dd5a4304e17b9b76f9d-2025-05-15 SoyLei USDA BPP NCAT Visit My recording 3.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/da/dae8087a62f028b8c6cf-2025-05-19 Dr Dikis Follow up My recording 4.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/a9/a9c53a94e82d1b027bf5-2025-05-20 Saber Corn Board Call Alex Buck My recording 5.readout.json`
- File verification confirmed all five JSON and Markdown outputs exist and are
  non-empty.

Next:

- Continue scaling conservatively with a 10-item batch at concurrency 1.
- If 10 items pass cleanly, consider raising batch size before raising
  concurrency; keep browser interaction rate limiting unchanged until there is
  more variability data.
