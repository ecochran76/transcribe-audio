# Plan 0016 | Config Panel Design Path

State: CLOSED

Lane: P09

Design track: Product UI

## Scope

Define the full design contract for the account/config panel before writing
implementation code.

The panel should become the operator's durable configuration workbench for:

1. Account and runtime profile status.
2. Intelligence task routing.
3. Workflow automation policy.
4. Provenance source configuration and health.
5. Safety, approval, and evidence controls.

This plan covers aesthetics, interaction design, information architecture,
configuration ergonomics, and browser-based inspection requirements. It permits
planning documents, read-only API checks, and `agent-browser` inspection
evidence. It does not authorize React, CSS, Python, API, or schema changes.

## Non-Goals

- No implementation code in this design slice.
- No live provider submissions, workflow-stage execution, external writes,
  deposition writes, memory writes, contact-source refreshes, or tenant writes.
- No secrets, iCalendar URLs, credentials, raw transcripts, private contact
  exports, or tenant-specific records in tracked files.
- No replacement of the existing user-scoped config files:
  `intelligence.config.json`, `automation.config.json`, and
  `provenance.config.json`.
- No new automatic stage enablement. Every workflow stage remains disabled or
  manual until a later implementation slice explicitly changes it.

## Current State

Plan 0015 added a functional Settings tab. The current surface exposes:

- Runtime profile and account/status fields.
- An intelligence-route summary with a link back to the Intelligence tab.
- Six automation stage rows with enabled, mode, review, and capability fields.
- Preview and apply controls for automation config updates.

The current Settings layout is still a first-pass settings page, not a complete
configuration workbench. It does not yet give provenance sources equal status
with automation and intelligence, does not provide a staged-change ledger, does
not expose config evidence in one place, and still relies on broad card
containers that make dense operational settings harder to scan.

Read-only `agent-browser` baseline evidence was captured before design:

- Desktop Settings screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-desktop.png`
- Mobile Settings screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-mobile.png`

## Aesthetic Direction

The panel should feel like a precise local operations console, not a marketing
settings page.

Use the existing dark operator shell as the foundation, but tighten it:

- Keep a graphite/charcoal base with warm off-white text.
- Use the existing lime accent only for primary action, active state, and focus.
- Add a small semantic palette: blue for information, green for ready, amber for
  warning/review, red for blocked/danger.
- Remove decorative radial gradients and heavy card gradients from the config
  workbench. Routine settings should use restrained surfaces and borders.
- Use one radius system: 8px for cards and panels, 6px for controls, 4px for
  compact chips and table cells.
- Use a 4px spacing grid. Dense rows should use 8px or 12px gaps, not large
  promotional spacing.
- Avoid nested cards. Use page bands, split panes, tables, and row groups.
- Keep letter spacing at 0 for ordinary text. Reserve uppercase labels for very
  short metadata, and avoid wide tracking in dense rows.
- Use fixed type sizes, not viewport-scaled `clamp()` headings inside the
  settings surface.

Suggested token intent:

```text
surface.base        #12120f
surface.raised      #1b1d18
surface.subtle      #20231c
border.subtle       rgba(244, 236, 216, 0.10)
border.strong       rgba(244, 236, 216, 0.18)
text.primary        #f4ecd8
text.secondary      #c9c0aa
text.muted          #948d7d
accent.primary      #d7ff73
status.info         #7fb7ff
status.ready        #76dfaa
status.warning      #ffbf66
status.danger       #ff6f61
```

## Information Architecture

The config panel should be a single Settings workbench with a left in-panel
navigation rail and a primary detail area.

Primary sections:

1. Account
2. Intelligence
3. Automation
4. Provenance
5. Safety
6. Evidence

The top of the panel should always show a compact config header:

- Active profile.
- API health.
- Store path.
- Config roots with redacted/user-scoped paths.
- Dirty state.
- Last loaded and last saved timestamps when available.
- A single status summary: `Defaults`, `Saved`, `Staged changes`, `Previewed`,
  `Apply failed`, or `Blocked`.

The left rail should show section labels with small state chips:

- Account: `ok`, `defaults`, or `blocked`.
- Intelligence: task count plus readiness state.
- Automation: enabled stage count.
- Provenance: enabled source count plus doctor state.
- Safety: pending approvals or `clear`.
- Evidence: latest smoke/check timestamp.

On mobile, the rail becomes compact tabs above the detail surface, and the dirty
action bar becomes sticky at the bottom.

## Functional Design

### Account

Purpose: orient the operator to the current profile and local runtime state.

Required content:

- Runtime profile name and selector.
- API status.
- Transcript store path.
- Runtime state root.
- Active config paths for intelligence, automation, and provenance.
- Environment override indicators when a config path comes from an env var.
- Read-only badges for `repo`, `user state`, `env override`, and `default`.

Ergonomics:

- Profile changes are staged locally until preview/apply.
- Paths must wrap cleanly and provide copy controls.
- Secrets and private feed URLs are never rendered.

### Intelligence

Purpose: show model/provider routing without turning the Settings panel into
the full App Intelligence console.

Required content:

- Compact route table with task, provider, model, timeout, fallback, source, and
  readiness.
- Filters for task family: summary, contextualization, identity, routing,
  memory, embedding, app-supervisor.
- A detail drawer for a selected task route.
- Link to the existing Intelligence tab for run ledgers and full controls.

Ergonomics:

- Editing a route stages a local config diff only.
- Preview must call only the intelligence config preview endpoint.
- Apply must require approval and write only user-scoped intelligence config.
- No provider, model-turn, prompt-packet, or workflow-stage action can be
  triggered from route editing.

### Automation

Purpose: make the production automation ladder understandable before any stage
is allowed to run automatically.

Stages:

1. Ingest audio
2. Transcribe audio
3. Initial summary
4. Speaker identity
5. Context collection
6. Final readout

Each stage row should show:

- Stage label.
- Current mode: manual, one-click, automatic.
- Enabled toggle.
- Requires review toggle.
- Preconditions.
- Capability state: unavailable, one-click ready, automation blocked, or
  automatic ready.
- External action flag.
- Last validation evidence.
- Blocked reason.
- Next safe action.

The rows should read like an operations matrix, not like isolated cards.

Ergonomics:

- Toggle and select changes are instant local state.
- No backend call is needed just to select, toggle, or edit an already-loaded
  stage value.
- The Preview action computes and displays the staged diff.
- The Apply action is disabled until preview succeeds.
- Apply writes config only and must show
  `will_execute_workflow_stage=false`.

### Provenance

Purpose: make context and contact source configuration a first-class part of
Settings, not a separate afterthought.

Required source families:

- `gog`
- `gws`
- `msgcli`
- `odollo`
- iCalendar feeds

Required content:

- Enabled source count by family.
- Active profile mapping.
- Redacted source identifiers.
- Per-source health/doctor status.
- Surface coverage: calendar lookup, contact identity, message affinity,
  context provenance, readout bundle.
- Last cache refresh status where applicable.
- Source-specific blocked reasons.

Source rows should show enough to answer: "Will this source be used, where, and
why is it safe?"

Ergonomics:

- Adding an iCalendar feed should support either an env ref or a redacted local
  secret reference. Direct private feed URLs should be allowed only in ignored
  user-scoped config, never in docs or tracked samples.
- Odollo tenant rows should show profile labels and readiness without exposing
  credentials.
- Contact-source refresh remains explicit and separate from config editing.
- Provenance preview/apply writes only config and never pulls fresh contacts or
  calendar events.

### Safety

Purpose: keep the automation boundary visible as production toggles are added.

Required content:

- Approval tokens required by config apply operations.
- External action summary.
- Write-bearing versus read-only operation legend.
- Current pending staged changes.
- Stage gates that must pass before automatic mode can be enabled.
- Human review requirements by stage.

Ergonomics:

- Dangerous choices need visible disabled reasons, not silent disabled buttons.
- Automatic mode should be visually allowed but operationally blocked when
  validation evidence is missing.
- The panel should explain policy through status labels and row metadata, not
  long instructional text.

### Evidence

Purpose: make validation and browser evidence visible to the operator.

Required content:

- Latest config doctor results.
- Latest API health result.
- Latest frontend build or smoke reference when available.
- Latest `agent-browser` screenshot/report paths.
- Console/network smoke summary for the config panel.

Ergonomics:

- Evidence rows should link to local artifacts when safe.
- Evidence should distinguish baseline, preview smoke, apply smoke, and mobile
  smoke.
- Missing evidence should show `not yet run`, not appear successful.

## Interaction Model

Use a draft/preview/apply lifecycle across all config sections.

1. Load current config from user-scoped runtime state.
2. Edits update local React draft state only.
3. Dirty bar summarizes staged changes.
4. Preview sends staged changes to preview endpoints and receives a redacted
   diff plus safety flags.
5. Apply requires the relevant approval token and writes only the matching
   user-scoped config.
6. After apply, reload the authoritative config and clear local preview state.

The dirty bar should be persistent on desktop and sticky on mobile. It should
show:

- Number of changed sections.
- Whether preview is stale.
- Primary action: Preview changes.
- Secondary action: Discard draft.
- Apply action only after successful preview.

All routine selection/editing controls must be local and zero-lag. Network work
belongs to explicit `Refresh`, `Doctor`, `Preview`, or `Apply` actions.

## Component Contract

Recommended component structure for a later implementation slice:

```text
SettingsWorkbench
  ConfigStatusHeader
  SettingsSectionRail
  ConfigDirtyBar
  AccountConfigSection
  IntelligenceConfigSection
  AutomationMatrixSection
  ProvenanceSourcesSection
  SafetyGatesSection
  EvidenceSection
  RedactedDiffPanel
```

Avoid a card mosaic. Use section layouts and row groups:

- `ConfigStatusHeader`: one horizontal status band.
- `SettingsSectionRail`: quiet rail with chips.
- `AutomationMatrixSection`: table/matrix rows.
- `ProvenanceSourcesSection`: grouped source rows.
- `RedactedDiffPanel`: one preview pane, reused by intelligence, automation,
  and provenance.

## Accessibility And Responsiveness

Acceptance requirements:

- Every input has a visible label.
- Toggle state is text-visible, not color-only.
- Focus rings are visible on keyboard navigation.
- Buttons have stable dimensions and do not resize when status text changes.
- Compact rows preserve at least 24px hit targets, and mobile controls preserve
  at least 44px hit targets.
- Long paths, model ids, source ids, and email-like identifiers wrap without
  overlapping adjacent controls.
- The mobile layout keeps selected section controls, dirty state, and apply
  status reachable without horizontal scrolling.

## Agent-Browser Inspection Requirement

This plan requires `$agent-browser` for both design inspection and later
implementation validation.

Baseline inspection already captured:

- Desktop Settings screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-desktop.png`
- Mobile Settings screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-mobile.png`

Before any code is written, implementation planning should inspect the baseline
with `agent-browser snapshot -i` and confirm the target design differences.

After implementation, validation must use `agent-browser` to capture:

- Desktop screenshot at 1440x1100.
- Mobile screenshot at about 390x844.
- Interactive accessibility snapshot.
- Console error check.
- Network request log for settings interactions.

Required browser interaction checks:

- Open Settings and verify Account, Intelligence, Automation, Provenance,
  Safety, and Evidence sections are reachable.
- Toggle an already-loaded automation stage and verify no backend request is
  made before Preview.
- Edit a provider/model route and verify no provider or model-turn route is
  called before Preview.
- Edit provenance source enablement and verify no contact/calendar refresh is
  started before Preview.
- Preview automation and verify the response shows
  `will_execute_workflow_stage=false`.
- Apply automation with approval and verify only the config apply endpoint is
  called.
- Use mobile viewport and verify rows do not overlap, paths wrap, and sticky
  dirty actions are reachable.
- Return to the selected conversation workflow and verify `Run initial summary`
  remains separate from Settings edits.

Evidence should be written under:

```text
~/.local/state/transcribe-audio/browser-smokes/
```

## Acceptance Criteria

- The design plan defines the settings workbench aesthetics, section model,
  interaction model, component contract, accessibility expectations, and
  browser-inspection requirements.
- The design keeps all runtime configuration user-scoped and redacted.
- The design explicitly separates local draft edits, preview, apply, source
  refresh, provider work, and workflow-stage execution.
- The design includes provenance sources as first-class Settings content.
- The design requires `agent-browser` evidence before and after implementation.
- No implementation code is changed in this planning slice.

## Closeout Notes

- Plan 0016 is complete as a design-only slice. It defines the configuration
  workbench contract before implementation code changes.
- The required baseline `agent-browser` inspection has desktop and mobile
  screenshot evidence.
- The post-implementation `agent-browser` checks remain mandatory for the next
  build slice that changes the React/CSS/API surface; they are not implementation
  work inside this design plan.
- No React, CSS, Python, API, schema, provider, source-refresh, or workflow-stage
  code was changed for this plan.

## Validation

- Read repo planning, runtime-state, architecture, and memory/context policies.
- Ran Graphiti discovery for `transcribe_audio_main`; results were stale for
  recent P09 settings work, so current repo files are the authority.
- Reviewed `ROADMAP.md`, `RUNBOOK.md`, Plan 0015, current Settings component,
  and current Settings CSS.
- Used `agent-browser` against the live local UI at `http://transcripts.localhost`
  to capture desktop and mobile baseline screenshots.
- Verified the baseline screenshots exist as PNG files at 1440x1100 and
  390x844.
- Ran `git diff --check` for tracked docs and a trailing-whitespace check for
  the new plan, roadmap, and runbook files.
