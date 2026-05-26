# Plan 0020 | Intelligence Profile Settings Redesign

State: CLOSED

Lane: P09

## Scope

Replace the Intelligence settings section's route-by-route editing model with a
profile-first settings surface.

This slice covers:

1. A compatible intelligence config layer where named profiles define provider,
   model, base URL, timeout, and temperature once.
2. Task/component routes selecting a named profile while keeping
   component-specific policy such as fallbacks, human review, and ledger
   requirements.
3. A denser Settings > Intelligence page with profile definitions and component
   profile selections on one page.
4. Moving low-action config facts into a discrete details disclosure instead of
   large informational panels.

## Non-Goals

- No provider execution, model turns, prompt packets, ledger starts, workflow
  automation, source refresh, or external writes.
- No removal of the existing legacy `tasks` override config. Existing configs
  must continue to resolve.
- No new secret storage in tracked files.
- No replacement of the full Intelligence tab's run-ledger/operator controls.

## Current State

Plan 0017 implemented Settings as a broader config workbench, but the
Intelligence section still exposed one selected task route with provider/model
fields and a large route-map panel. That made repeated provider/model settings
look like per-component data and used large panels for facts that are better
shown as compact rows or hidden behind a disclosure.

## Acceptance Criteria

- `/api/intelligence/config` exposes named profiles and task-to-profile
  assignments in addition to resolved task routes.
- Existing task-level provider/model overrides still resolve and apply through
  the existing preview/apply endpoints.
- Profile edits and task profile assignment can be previewed and applied
  through the existing approval-gated intelligence config endpoint.
- Settings > Intelligence shows profile definitions and component profile
  selections on one page.
- Config path/provider/source facts are collapsed into a discrete details
  section.
- Browser inspection verifies the page renders the new profile surface and no
  provider/model-turn request is made from ordinary editing.

## Validation

- `python -m py_compile intelligence_config.py transcript_api.py`
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q`
- `npm --prefix frontend run build`
- `agent-browser` inspection of Settings > Intelligence on
  `http://transcripts.localhost`
- `agent-browser` console/error checks
- `git diff --check`

## Closeout

Closed on 2026-05-25.

Implemented:

- Added named intelligence profiles and task-to-profile assignments to
  `intelligence_config.py`.
- Kept legacy task-level provider/model overrides compatible while making
  profile selection clear provider/model route fields for that component.
- Extended the intelligence config preview/apply API so a profile-only edit can
  be previewed or applied without sending a task update.
- Reworked Settings > Intelligence into a profile-first page: profile
  definitions at the top, component profile selections below, and low-action
  config facts in collapsed details.
- Replaced the oversized Settings status panel with compact pills and a closed
  config-path disclosure.

Evidence:

- `python -m py_compile intelligence_config.py transcript_api.py` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- `.venv/bin/python -m pytest -q` passed with 246 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` inspection of Settings > Intelligence verified:
  `title=Settings`, `routeRowCount=8`, `compactSectionCount=2`,
  `detailsOpen=false`, `statusDetailsOpen=false`, `hasOldRouteMapText=false`,
  and `hasRuntimePathCards=false`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0020-intelligence-settings-profile-page.png`.
- `agent-browser` console/error checks reported no page errors.
