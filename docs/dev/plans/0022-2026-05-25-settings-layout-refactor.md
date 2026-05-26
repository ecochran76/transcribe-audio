# Plan 0022 | Settings Layout Refactor

State: CLOSED

Lane: P09

Design track: Product UI

## Scope

Refactor Settings from a narrow panel inside the transcript review shell into a
space-efficient configuration workspace.

This slice covers:

1. Give Settings a dedicated workspace layout that removes or collapses
   transcript-specific left and right panes by default.
2. Replace the current in-panel settings rail with a compact responsive section
   navigator that does not collide with labels or status text.
3. Make Intelligence profile routing fit inside the available content width
   without clipped columns.
4. Remove global Library/workflow summary chrome from the Settings first
   viewport.
5. Rename Settings evidence copy away from operator-test terminology and toward
   validation/dogfood evidence.
6. Preserve the existing user-scoped preview/apply safety model.

## Non-Goals

- No backend config schema changes.
- No changes to provider, model-turn, contact refresh, workflow-stage,
  deposition, memory, or tenant-write behavior.
- No unattended automation enablement.
- No removal of the dedicated Intelligence surface or Library diagnostics.
- No tracked secrets, private calendar URLs, raw transcripts, private contact
  records, or tenant records.

## Current State

The live Settings page now has the right conceptual model: Account,
Intelligence, Automation, Provenance, Safety, and Evidence live in one
configuration surface; intelligence profiles are defined once; components
select profiles; and local edits stay local until Preview/Apply.

The UI audit found that the settings surface is still constrained by the global
transcript shell:

- Desktop `1440x960` leaves the Settings workbench at roughly 666px wide and
  the actual detail surface at roughly 433px because the transcript left pane
  and inspector remain visible.
- The Intelligence component/profile/policy matrix requires more width than
  that detail surface and clips the policy column.
- Mobile section navigation overlaps label and meta text because the rail
  becomes a two-column grid while each button still uses two internal columns.
- Global summary chips consume first-viewport height on Settings even though
  they are not the operator's main task there.
- The Evidence section still uses "smoke" terminology and exposes raw artifact
  paths too prominently.

## Design Direction

Settings should feel like a precise local operations console. It should remain
dense, quiet, and utility-first.

Use the existing dark operator shell, but make the work surface dominant:

- Keep the top app bar and account chip.
- Use a Settings-specific workspace variant that hides transcript-only panes.
- Use one flat workbench surface with compact section navigation and a broad
  detail region.
- Prefer row groups, matrices, and native controls over card mosaics.
- Use 8px radius for panels, 6px for controls, and 4px for small chips.
- Keep the lime accent only for active states, primary actions, and focus.
- Use semantic colors only for status: ready, warning, blocked, info.
- Keep labels and headings sentence-case, compact, and operational.

## Refactor Plan

### 1. Settings Workspace Shell

Add an `activeNav === "Settings"` workspace variant.

Desktop target:

- The Settings route keeps the global topbar.
- The transcript left pane is not rendered or is fully hidden, not merely
  collapsed to a 58px rail.
- The transcript inspector is not rendered by default because selected
  transcript metadata is not part of routine configuration.
- The center pane becomes a full-width settings workspace with a constrained
  inner width, approximately `min(1180px, 100%)`.
- Runtime roots stay in the Account disclosure, not in a permanent side pane.

Mobile target:

- Topbar remains compact.
- Settings content starts above the fold.
- Section navigation sits immediately above the active section as tabs or a
  single-row horizontal scroller.

### 2. Settings Header

Replace the global summary strip on Settings with a compact settings-only
status row.

Include only:

- Active profile.
- Intelligence profile count.
- Automation enabled count.
- Provenance enabled source count.
- Dirty/preview/apply state.

Hide:

- Conversation count.
- Artifact count.
- Open review count.
- Library filter or row-scope diagnostics.

### 3. Section Navigation

Replace the current two-column rail behavior with a responsive section
navigator.

Desktop:

- Use a slim vertical rail only when there is enough room.
- Each item has a label and a small status chip.
- Label and chip must never share a fixed two-column layout narrower than their
  text.

Tablet/mobile:

- Convert to horizontal tabs or a single-column list.
- Stack label and meta if the tab is narrow.
- The active tab is visible without wrapping over adjacent tabs.

### 4. Intelligence Routing Matrix

Make component profile selection the primary Settings table.

Desktop:

- Show columns for Component, Profile, and Policy when the surface is wide
  enough.
- Allow the table to use the full settings detail width.

Narrow/mobile:

- Render each route as a compact row with Component and Profile on the first
  line and Policy/ledger/review metadata on the second line.
- Avoid horizontal clipping for policy text.
- Keep the selected component editor below the matrix.

### 5. Evidence To Validation Copy

Rename the Settings "Evidence" section to "Validation" or "Dogfood evidence".

Replace visible copy:

- "latest smoke" -> "latest validation"
- "Smoke report" -> "Validation report"
- "Smoke screenshot" -> "Browser check screenshot"

Move raw local artifact paths into a details disclosure. The first view should
show status and recency, not filesystem internals.

### 6. Safety And Functionality Guardrails

Preserve existing behavior:

- Selecting sections is client-only.
- Editing already-loaded controls is client-only.
- Preview calls only preview endpoints.
- Apply calls only reviewed config-write endpoints.
- Apply remains disabled until a preview exists.
- Automation apply must continue to prove
  `will_execute_workflow_stage=false`.
- Provenance config edits must not refresh contacts, calendar events, or
  Odollo sources.

## Acceptance Criteria

- At desktop `1440x960`, Settings uses the full app work area and the detail
  surface is at least 850px wide.
- At desktop, the Intelligence route matrix shows Component, Profile, and
  Policy without clipping.
- At mobile `390x844`, no settings navigation labels overlap status/meta text.
- At mobile, the active Settings section's first actionable controls appear in
  the first viewport after the topbar and compact section navigation.
- Settings no longer renders conversation/artifact/review summary chips.
- Settings no longer renders visible "latest smoke", "Smoke report", or "Smoke
  screenshot" labels.
- Account, Intelligence, Automation, Provenance, Safety, and Validation remain
  reachable by keyboard and pointer.
- No backend request is made when merely selecting sections, selecting
  already-loaded contacts/config rows, or toggling local draft controls.
- Preview/Apply behavior and approval-gate semantics are unchanged.

## Validation

- `npm --prefix frontend run build`
- `git diff --check`
- Focused frontend/source inspection for accidental provider/workflow calls
  from Settings selection and local draft edits.
- `agent-browser` desktop inspection at `1440x960`:
  - open Settings > Intelligence;
  - assert no transcript left pane or inspector consumes Settings workspace
    width;
  - assert route matrix policy text is not clipped;
  - capture screenshot.
- `agent-browser` mobile inspection at `390x844`:
  - open Settings > Account and Settings > Intelligence;
  - assert section navigation has no text overlap;
  - assert first actionable controls are visible without excessive scrolling;
  - capture screenshots.
- `agent-browser` console/error check.
- Existing focused tests for config endpoints if source code changes touch API
  call wiring:
  - `.venv/bin/python -m pytest tests/test_intelligence_config.py
    tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
    tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
    -q`

## Implementation Order

1. Add the Settings-specific workspace shell and suppress transcript panes on
   Settings.
2. Replace the Settings summary strip with settings-only status.
3. Refactor Settings section navigation for desktop and mobile.
4. Make the Intelligence route matrix responsive.
5. Rename Evidence to Validation and tuck raw paths behind details.
6. Run browser inspections and tighten spacing/overflow issues found by the
   screenshots.

## Closeout

Closed on 2026-05-25.

Implemented:

- Added a Settings-specific workspace layout that hides the transcript left pane
  and inspector and gives Settings a centered `min(1180px, 100%)` work area.
- Removed the global conversation/artifact/review summary strip from Settings.
- Added a compact Settings status row for active profile, profile count,
  automation enabled count, provenance source count, and saved/staged state.
- Refactored Settings section navigation so labels and status chips stack on
  desktop and become a horizontal mobile tab scroller without label/meta
  collision.
- Made the Intelligence component/profile/policy route matrix fit the widened
  Settings detail surface and collapse without horizontal overflow on mobile.
- Renamed the Settings Evidence section to Validation, replaced visible smoke
  copy with validation/browser-check wording, and moved raw artifact paths into
  a disclosure.

Evidence:

- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- `transcripts.service` was restarted and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- Source inspection verified Settings section selection and already-loaded
  draft controls use local React state setters; preview/apply still go through
  the existing reviewed endpoints.
- `agent-browser` desktop `1440x960` inspection of Settings > Intelligence
  measured `center.width=1180`, `settingsSurface.width=952`,
  `leftPane.display=none`, `rightPane.display=none`, `routeOverflow=false`,
  and no visible conversation/artifact/open-review/latest-smoke/smoke-report
  text.
- `agent-browser` mobile `390x844` inspection of Settings > Account and
  Settings > Intelligence measured no section-nav text overlap, no route-row
  overflow, first actionable controls inside the first viewport, and no visible
  conversation/artifact/open-review/latest-smoke/smoke-report text.
- `agent-browser` keyboard reachability inspection verified every Settings
  section button is enabled with `tabIndex=0`.
- `agent-browser` network capture after local section selection and local draft
  interaction reported `No requests captured`.
- `agent-browser` console and page-error checks reported no output.
- Screenshots:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-desktop-intelligence.png`,
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-mobile-account.png`,
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-mobile-intelligence.png`.
