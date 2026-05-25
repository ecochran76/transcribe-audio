# Plan 0017 | Settings Config Workbench

State: CLOSED

Lane: P09

## Scope

Implement the Plan 0016 configuration-panel design contract in the React review
console.

This slice turns the existing Settings tab into a single config workbench for:

1. Account/runtime status.
2. Intelligence task routing.
3. Workflow automation policy.
4. Provenance source configuration and health.
5. Safety gates and approval evidence.
6. Browser-smoke and config evidence.

The implementation must keep routine edits local and zero-lag. Backend calls are
allowed only for explicit Preview, Apply, Refresh, Doctor, or smoke/evidence
actions.

## Non-Goals

- No unattended audio ingestion, transcription, summarization, speaker identity,
  context collection, final readout, deposition, memory writes, or external
  tenant writes.
- No automatic workflow-stage enablement beyond existing user-scoped automation
  config policy.
- No provider/model-turn execution from Settings.
- No contact-source refresh or calendar-source refresh caused by Settings page
  load or local edits.
- No secrets, private iCalendar URLs, credentials, raw transcripts, or tenant
  records in tracked repo files.
- No replacement of the existing `intelligence_config.py`,
  `automation_config.py`, or `provenance_config.py` user-scoped contracts.

## Current State

Plan 0016 is closed as the design authority. It defines the target aesthetics,
section model, component contract, draft/preview/apply lifecycle, and
`agent-browser` validation gate.

The current Settings tab from Plan 0015 shows account/runtime status, a compact
intelligence summary, and automation stage controls. Provenance remains a
separate top-level tab, there is no unified dirty bar, and evidence/safety
status is spread across other panels.

The current checkpoint commit before this plan is:

```text
926dc65 Checkpoint provenance and settings milestones
```

## Acceptance Criteria

- Settings renders as a workbench with Account, Intelligence, Automation,
  Provenance, Safety, and Evidence sections.
- The workbench has a compact status header, in-panel section navigation, and a
  persistent dirty/preview/apply bar.
- Automation toggles and mode/review edits update local draft state without
  backend requests until Preview or Apply.
- Intelligence route edits update local draft state without provider,
  model-turn, prompt-packet, or workflow-stage requests.
- Provenance enablement and iCalendar draft edits update local draft state
  without contact/calendar source refresh.
- Provenance source families `gog`, `gws`, `msgcli`, `odollo`, and
  `ical_calendar` are visible when configured.
- Safety and Evidence sections show approval tokens, write/read-only policy,
  config paths, doctor status, smoke evidence, and baseline browser evidence.
- Preview actions use existing preview endpoints and show redacted diff/safety
  flags.
- Apply actions remain approval-gated and write only user-scoped config.
- Mobile and desktop layouts avoid overlapping controls and keep dirty actions
  reachable.

## Closeout Notes

- The Settings tab now renders a config workbench with a status header, in-panel
  section rail, dirty/preview/apply bar, and Account, Intelligence, Automation,
  Provenance, Safety, and Evidence sections.
- Intelligence, automation, and provenance edits use local React draft state.
  `agent-browser` network checks verified automation, intelligence, and
  provenance local edits make no backend requests before explicit Preview.
- Automation Preview remains the only tested backend call during the workbench
  browser smoke and returns `will_execute_workflow_stage=false`; no Apply action
  was clicked during validation.
- Discard clears stale preview state so Apply controls do not remain available
  after local staged edits are cleared.
- Desktop and mobile screenshots were captured under
  `~/.local/state/transcribe-audio/browser-smokes/`.

## Validation

- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py`
- Focused config endpoint pytest passed with 4 tests.
- `.venv/bin/python -m pytest -q` passed with 243 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `http://transcripts.localhost/api/health`
  returned `status: ok`.
- `agent-browser` desktop screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0017-settings-workbench-desktop.png`
- `agent-browser` mobile screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0017-settings-workbench-mobile.png`
- `agent-browser` network checks verified:
  - automation toggle: no requests before Preview;
  - intelligence model edit: no requests before Preview;
  - provenance source toggle: no requests before Preview;
  - automation Preview: one `POST /api/automation/config/preview` request.
- `agent-browser` page text showed automation preview safety flags including
  `will_execute_workflow_stage=false`.
- `agent-browser` console/error checks reported no page errors.
