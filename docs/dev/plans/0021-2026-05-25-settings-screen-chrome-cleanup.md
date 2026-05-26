# Plan 0021 | Settings Screen Chrome Cleanup

State: CLOSED

Lane: P09

## Scope

Remove Library/test-status chrome from the Settings screen and keep Settings
focused on configuration controls.

This slice covers:

1. Stop rendering the operator test-status strip on Settings.
2. Remove the Settings-local API status block that duplicated frontend health
   state and could conflict with the test-status wording.
3. Hide the staged-config bar until there is an actual draft or prepared
   preview to act on.
4. Keep config paths available only inside the discrete Account runtime-roots
   disclosure.

## Non-Goals

- No change to backend health semantics.
- No change to the Library diagnostics disclosure.
- No removal of Intelligence smoke-job controls from the dedicated
  Intelligence surface.
- No external workflow execution.

## Current State

After Plan 0020, Settings still inherited the non-Library test-status strip.
When health had not resolved or the frontend fell back to fixture/default data,
that strip labeled the API state as "Preview" while the Settings status block
showed raw health as "offline". The same area also showed "Latest smoke" and a
no-op staged-edits message, occupying the first viewport before any actual
settings control.

## Acceptance Criteria

- Settings renders no `Operator test status` region.
- Settings renders no `Latest smoke` text.
- Settings renders no "API Preview" or "API offline" status block.
- Settings does not show the staged-config bar when there are no local edits or
  prepared previews.
- Settings > Intelligence profile controls appear higher in the first viewport.
- Browser inspection verifies the cleaned Settings page.

## Validation

- `python -m py_compile transcript_api.py intelligence_config.py`
- `npm --prefix frontend run build`
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q`
- `agent-browser` inspection of Settings > Intelligence on
  `http://transcripts.localhost`
- `agent-browser` console/error checks
- `git diff --check`

## Closeout

Closed on 2026-05-25.

Implemented:

- Removed the non-Library `TestStatusStrip` render path, so Settings no longer
  shows rows-in-scope, filter state, API preview, or latest-smoke diagnostics.
- Removed the Settings status card and its duplicate API/config-path summary.
- Changed the staged-config bar to render only when there is an actual local
  draft or prepared preview.
- Removed the Settings-specific left-pane card and blocked the generic Library
  kind filters from falling through into Settings.
- Kept runtime/config paths inside the Account section's collapsed runtime-roots
  disclosure.

Evidence:

- `python -m py_compile transcript_api.py intelligence_config.py` passed.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- `.venv/bin/python -m pytest -q` passed with 246 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` inspection of Settings > Intelligence verified
  `hasOperatorTestStatus=false`, `hasRowsInScope=false`,
  `hasLatestSmoke=false`, `hasApiPreview=false`, `hasApiOffline=false`,
  `hasNoStagedConfigEdits=false`, `hasDraftBar=false`, and `routeRowCount=8`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0021-settings-chrome-cleanup.png`.
- `agent-browser` console/error checks reported no page errors.
