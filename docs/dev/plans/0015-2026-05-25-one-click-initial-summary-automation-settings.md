# Plan 0015 | One-Click Initial Summary And Automation Settings

State: CLOSED

Lane: P09

## Scope

Tighten the selected-conversation initial summarization workflow so an operator
can run the first-pass summary from one primary action in the conversation
workspace.

Expose a Settings surface for account/runtime status, intelligence routing, and
automation policy. Automation policy must live in user-scoped runtime state and
be readable from both CLI/library code and the web service.

This slice covers:

1. A selected-conversation `run initial summary` API action that prepares the
   scoped request manifest and submits it to the configured first-pass provider
   in one reviewed call.
2. A React first-pass summary panel with one primary action plus secondary
   status/resume controls.
3. A user-scoped automation config with stages for ingestion, transcription,
   initial summary, speaker identity, context collection, and final readout.
4. A Settings tab that exposes runtime/account status, intelligence settings,
   and automation settings without duplicating tenant secrets into the repo.

## Non-Goals

- No unattended audio ingestion, transcription, summarization, speaker
  identification, context collection, final readout, deposition, or memory
  writes.
- No production auto-run of workflow stages before every stage has targeted
  tests, browser smoke evidence, and an explicit stage-level enablement
  decision.
- No provider work from page load, settings load, config preview, config apply,
  or status inspection.
- No raw transcripts, private contact exports, credentials, calendar feed URLs,
  or tenant state in tracked repo files.
- No replacement of the existing `intelligence_config.py` task-routing contract;
  automation config controls when workflow stages may run, not which model route
  they use.

## Current State

Plan 0014 closed the conversation-scoped contact search workbench. Cached
contact selection is instant, configured-source refresh is explicit, and
operator/App Intelligence actions share local batch contracts.

Selected-conversation first-pass summary currently requires separate Prepare,
Submit, and Check actions. Prepare writes a one-request dry-run manifest under
`~/.local/state/transcribe-audio/first-pass-summary-batches/`; Submit requires
`approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`; Check can materialize
completed readouts back into the store.

`intelligence_config.py` already centralizes task-level provider/model routing
under `~/.local/state/transcribe-audio/intelligence.config.json` or
`TRANSCRIPTS_INTELLIGENCE_CONFIG`. There is not yet a matching automation
policy config for stage enablement.

The React navbar has an Intelligence tab with routing controls and a disabled
Settings tab. Production automation is still a future goal; today the safe path
is explicit operator action with persisted local manifests.

## Acceptance Criteria

- A single selected-conversation action prepares and submits an initial summary
  request, returns the manifest path, batch id/status, and explicit external
  action flags, and reuses the existing scoped-manifest validation.
- The React first-pass summary view presents `Run initial summary` as the
  primary action, leaves status/materialization available after submission, and
  does not require a backend call for merely viewing settings or already-loaded
  summary state.
- Automation settings are stored in a user-scoped JSON config outside the repo,
  with preview/apply endpoints and an approval token for apply.
- Default automation config keeps every stage disabled/manual; automatic modes
  are visible policy choices but not silently enabled.
- The Settings tab shows account/runtime status, intelligence config location
  and route summaries, and editable automation stage toggles.
- README, API docs, roadmap, and runbook describe the one-click summary and
  automation config boundaries.

## Validation

- `python -m py_compile transcript_api.py intelligence_config.py automation_config.py`
- Targeted pytest for selected first-pass summary run and automation config API.
- Existing transcript API pytest coverage for first-pass batch prepare/submit
  and intelligence config still passes.
- `npm --prefix frontend run build`
- Browser smoke verifies the Settings tab renders and the summary panel exposes
  a single primary initial-summary action without clicking a live provider job.

## Closeout Notes

- `automation_config.py` defines user-scoped automation policy at
  `~/.local/state/transcribe-audio/automation.config.json` or
  `TRANSCRIPTS_AUTOMATION_CONFIG`.
- `POST /api/conversations/<id>/first-pass-summary/run` prepares and submits a
  selected-conversation first-pass summary request in one reviewed call using
  `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`.
- `GET /api/automation/config`, `/preview`, and `/apply` expose automation
  policy without running workflow stages. Apply requires
  `approval_token=APPLY_AUTOMATION_CONFIG_UPDATE`.
- The React Settings tab now shows account/runtime status, intelligence route
  summaries, and automation stage toggles. All automation stages default to
  disabled/manual.
- Full validation passed with `243 passed`, `npm --prefix frontend run build`,
  `git diff --check`, live service health checks, and an agent-browser smoke
  screenshot at
  `~/.local/state/transcribe-audio/browser-smokes/plan-0015-settings-summary-smoke.png`.
