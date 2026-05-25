# Plan 0019 | One-Click Summary Workflow Prep Polish

State: CLOSED

Lane: P09

## Scope

Polish the selected-conversation First-pass summary stage so the operator sees
one dominant workflow-prep action instead of a flat cluster of internal
prepare, submit, and status controls.

This slice covers:

1. A First-pass summary action panel that presents one primary next action:
   run initial summary, submit a prepared request, check status, or acknowledge
   that the summary is already ready.
2. Secondary prepare/submit/check controls preserved behind an advanced
   disclosure for recovery, testing, and explicit resume cases.
3. Browser-smoke coverage that verifies the one-click summary surface is not
   regressed into a multi-button primary workflow.

## Non-Goals

- No backend provider contract changes.
- No new automation mode or unattended workflow-stage execution.
- No change to `automation_config.py`, `intelligence_config.py`, provenance,
  contact selection, context workbench, or final-readout gating.
- No live provider submission during browser validation.

## Current State

Plan 0015 closed the one-click backend/API path and automation settings:
`POST /api/conversations/<id>/first-pass-summary/run` prepares and submits a
selected-conversation request with `SUBMIT_FIRST_PASS_SUMMARY_BATCH`, and the
Settings workbench exposes automation policy with all stages disabled/manual by
default.

The React First-pass summary tab still shows `Run initial summary`, `Prepare
only`, `Submit`, and `Check` as peer buttons in the main action row. This keeps
the recovery controls available, but it dilutes the one-click workflow and does
not visually distinguish the normal next action from low-level batch controls.

## Acceptance Criteria

- The First-pass summary tab shows one primary next action inside a clear
  workflow-prep card.
- Prepare-only, Submit, and Check remain available but are grouped as advanced
  summary controls.
- If a summary is already linked, the primary action is disabled and reports
  that the summary is ready.
- If a prepared-only manifest exists, the primary action submits that prepared
  request.
- If a submitted manifest exists, the primary action checks status and
  materializes completed output.
- Browser smoke verifies the one-click workflow-prep card, exactly one primary
  summary action, and advanced controls.

## Validation

- `npm --prefix frontend run build`
- `python -m py_compile scripts/smoke_conversation_review_loop_ui.py`
- `scripts/smoke_conversation_review_loop_ui.py` against
  `http://transcripts.localhost`
- `agent-browser` console/error checks for the smoke session
- `git diff --check`

## Closeout Notes

- The First-pass summary tab now shows an `Initial summary prep` card with one
  primary next-action button.
- The primary action advances through the summary workflow state: run initial
  summary when no manifest exists, submit an already prepared request, check a
  submitted manifest, or stay disabled when a summary is ready.
- Prepare only, Submit, and Check remain available under `Advanced summary
  controls`.
- The conversation review loop smoke now asserts the prep card, exactly one
  primary summary action, advanced controls, and no direct inline summary
  button cluster.

## Validation Evidence

- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_conversation_review_loop_ui.py` passed.
- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py participant_identity.py` passed.
- `.venv/bin/python -m pytest
  tests/test_transcript_api.py::test_selected_first_pass_summary_run_endpoint_prepares_and_submits
  -q` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `scripts/smoke_conversation_review_loop_ui.py` passed against
  `http://transcripts.localhost` with summary checks:
  `summary_hasSummaryPrepCard=true`, `summary_summaryPrimaryActionCount=1`,
  `summary_hasAdvancedSummaryControls=true`, and
  `summary_hasInlineSummaryButtonCluster=false`.
- Smoke report:
  `~/.local/state/transcribe-audio/browser-smokes/20260525T224138Z-conversation-review-loop-smoke.json`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/20260525T224138Z-conversation-review-loop-smoke.png`.
- `agent-browser` console/error checks reported no page errors.
