# Plan 0018 | Landing Page Navigation Redesign

State: CLOSED

Lane: P09

## Scope

Refactor the root Library landing page into a calmer transcript review
workspace with product-standard navigation.

This slice addresses the current first viewport at `/`:

1. Replace the crowded top-level nav with workflow-oriented primary navigation.
2. Move account, settings, provenance, intelligence, and automation surfaces
   behind an upper-right account/avatar chip and menu.
3. Keep search as a Library/workbench control, not a top-level destination.
4. Rework the landing layout so the main Library work surface appears in the
   first viewport on desktop and mobile.
5. Make filters contextual and collapsible so they support the selected work
   without dominating the page.
6. Preserve URL-addressable Library/search/selection/workspace state.

## Non-Goals

- No backend API changes.
- No changes to intelligence, provenance, automation, contact-search,
  speaker-identity, context, deposition, or provider execution contracts.
- No account-auth implementation beyond local UI routing/menu affordances.
- No unattended workflow automation.
- No change to stored transcript, blob, review-queue, or config schemas.
- No removal of existing surfaces; admin surfaces are relocated or grouped,
  not discarded.

## Current State

`agent-browser` inspection of `http://transcripts.localhost/` on 2026-05-25
showed the root landing page is functionally rich but visually and
ergonomically overgrown.

Desktop at 1440x1100:

- Topbar contains brand, eight primary nav pills, disabled planned nav pills,
  and global search.
- First viewport shows filters, Library table, operator status strip, and
  inspector with roughly equal visual weight.
- The selected document state pushes the URL to a `selected=` parameter after
  load, so landing state is tied to row selection immediately.
- Screenshot evidence:
  `~/.local/state/transcribe-audio/ui-audits/2026-05-25-landing-desktop.png`.

Mobile at 390x844:

- Topbar height is about 196px.
- The horizontally scrolling nav extends beyond the visible viewport and hides
  later actions such as Provenance, Intelligence, and Settings.
- The left filter pane occupies the first viewport; the main Library work
  surface begins around y=1014 and the inspector begins around y=9923.
- Screenshot evidence:
  `~/.local/state/transcribe-audio/ui-audits/2026-05-25-landing-mobile.png`.

Source hotspots:

- `frontend/src/main.jsx` defines `NAV_ITEMS` with workflow, admin, planned,
  and settings destinations mixed together.
- `frontend/src/main.jsx` renders topbar brand, primary nav, and global search
  in one row.
- `frontend/src/styles.css` uses a three-column workspace by default and
  stacks panes as filters, center, inspector on mobile.

## Audit Findings

- `nav-primary-too-broad`: Primary nav includes admin/configuration surfaces
  (`Provenance`, `Intelligence`, `Settings`) and disabled future destinations,
  so the landing page reads like a development console instead of a transcript
  workbench.
- `nav-account-actions-misplaced`: Settings are exposed as a primary
  destination instead of living behind a predictable account/avatar chip.
- `layout-mobile-main-content-buried`: On mobile, filters render before the
  Library content and push the actual work surface below the first viewport.
- `layout-first-viewport-overloaded`: Desktop first viewport gives comparable
  emphasis to chrome, filters, status diagnostics, data table, and inspector.
- `type-dashboard-heading-too-large`: Pane headings and Library headings use
  hero-scale treatment inside a dense workbench.
- `copy-dev-status-too-prominent`: "redacted preview mode", planned nav labels,
  operator test status, and developer raw-context affordances are exposed too
  prominently for a landing page.
- `interaction-hidden-horizontal-nav`: Mobile primary nav relies on horizontal
  overflow with hidden scrollbar, making later destinations discoverable only
  by guessing.

## Target Design

The root page should open as a focused transcript review workbench:

1. Topbar:
   - Left: compact brand and live/local API status.
   - Center: primary workflow nav limited to `Library`, `Review Queue`, and
     optionally `Runs`.
   - Right: account/avatar chip with menu entries for Settings, Account,
     Integrations/provenance, Intelligence, Automation, Runtime status, and
     About/debug.
2. Library toolbar:
   - Search input, kind segmented control, sort/status filters, and share link
     sit above the Library table as Library controls.
   - Search remains URL-addressable through `q=`.
3. Desktop layout:
   - Default to a two-column work surface: main Library list plus right
     selected-conversation inspector.
   - Move filters into a compact toolbar/drawer or a narrow secondary rail only
     when active.
   - Move operator test status to a collapsible diagnostics row or footer.
4. Mobile layout:
   - Header should fit within one compact bar plus optional search row.
   - Main Library content appears before filters and inspector.
   - Filters open as a sheet/drawer or inline collapsible panel below the
     Library toolbar.
   - Inspector becomes a details drawer opened from the selected row.
5. Visual style:
   - Preserve the dark operator-console theme, but reduce rounded-pill chrome,
     gradients, and high-contrast borders.
   - Use 8px-radius panels and compact controls for product surfaces.
   - Reduce hero-scale headings inside pane chrome.

## Acceptance Criteria

- Root landing page shows the Library work surface above the fold on desktop
  and mobile.
- Primary nav contains only workflow destinations and no disabled planned
  pills.
- Settings are reachable from an upper-right account/avatar chip menu.
- Provenance, Intelligence, Automation, and Runtime status are reachable from
  the account/settings menu without appearing as peer workflow destinations.
- Search is visually and semantically a Library control, not a nav item.
- Mobile view does not require horizontal nav scrolling to find Settings.
- Mobile view does not place filters above the main Library content by default.
- Existing URL state for `view`, `kind`, `q`, `selected`, `conversation`, and
  `workflow` continues to work.
- Existing `agent-browser` deep-link/share smoke remains valid or is updated
  with equivalent coverage.
- New `agent-browser` desktop and mobile screenshots verify the landing layout.
- Console/error checks report no page errors.

## Validation

- `npm --prefix frontend run build`
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q`
- Existing or updated Library deep-link/share smoke:
  `scripts/smoke_library_deeplink_share_ui.py`
- `agent-browser` desktop screenshot at 1440x1100 for `/`.
- `agent-browser` mobile screenshot at 390x844 for `/`.
- `agent-browser` snapshot confirms the account/avatar chip exposes Settings
  and admin destinations.
- `agent-browser` metrics confirm mobile center Library content begins in the
  first viewport before filters and inspector.

## Closeout Notes

- The root Library page now uses workflow-only primary navigation: `Library`
  and `Review Queue`.
- Settings, Account management, Integrations/provenance, Intelligence,
  Automation, and Runtime status moved behind the upper-right account chip.
- Search moved out of the topbar and into a Library toolbar with artifact-kind
  segmented controls and the workspace link action.
- Library filters are hidden by default and can be opened from the toolbar;
  desktop starts as a Library-plus-inspector work surface.
- Mobile now renders the compact topbar, Library work surface, toolbar, table
  header, and first row before filters or inspector.
- The Library deep-link/share smoke was updated to look for the new Library
  search and kind controls.

## Validation Evidence

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  52 tests.
- `.venv/bin/python -m pytest -q` passed with 243 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `scripts/smoke_library_deeplink_share_ui.py` passed against
  `http://transcripts.localhost` and wrote
  `~/.local/state/transcribe-audio/browser-smokes/20260525T213630Z-library-share-ui-smoke.json`.
- `agent-browser` desktop screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-landing-desktop-final.png`.
- `agent-browser` mobile screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-landing-mobile-final.png`.
- `agent-browser` account-menu screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-account-menu.png`.
- `agent-browser` metrics verified:
  - desktop topbar height 63px, first row y=387, primary nav only `Library`
    and `Review Queue`, no global search, Library search present, filters
    hidden by default;
  - mobile topbar height 113px, center pane y=125, table y=599, first row
    y=661, primary nav only `Library` and `Review Queue`, no global search,
    Library search present, filters hidden by default.
- `agent-browser` account-menu snapshot showed Settings, Account management,
  Integrations/provenance, Intelligence, Automation, and Runtime status menu
  items.
- `agent-browser` filter interaction showed the Library filter pane opens on
  demand.
- `agent-browser` console/error checks reported no page errors.
