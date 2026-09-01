# Plan 0080 | Flat Approval Row View

State: CLOSED

Lane: P09

Date: 2026-09-01

Related authority: Plans 0078-0079, `VISION.md`, `CONTEXT.md`, and
`docs/adr/0004-derive-affiliations-from-role-appointments.md`

## Scope

Replace the misleading actionable-contact filter with a true peer view of
flat approval rows. Every unreviewed organization or role hypothesis must be a
directly actionable, single-line top-level row. The full contact and
organization directory must remain equally visible through a compact mode
toggle and must never inherit a hidden approval filter.

## Vision outcomes and maturity movement

This plan advances the review/acceptance stage, provenance, organization/role
context, and self-feeding knowledge loop in `VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Approval workflow | Level 2 actions still require expanding one of 52 contact rows | Level 2 direct queue of all 57 unreviewed hypotheses | Default Contacts view contains 57 directly actionable top-level rows and zero contact expand controls |
| Directory access | Level 2 full data exists but an actionable filter makes non-review data appear hidden | Level 2 explicit peer Directory mode with no implicit approval filter | Directory toggle restores all 208 contacts and all 40 organizations |
| Judgment density | Level 2 evidence and controls are split across parent and expanded rows | Level 2 one-line approval row carries the person, proposal, source basis, activity history, targets, title, and actions | Desktop browser measurement shows one row per hypothesis with no detail panels or disclosures |

This remains explicit human review. It does not establish automated acceptance,
speaker-identification lift, or truth for any unreviewed hypothesis.

## Current State

The installed Contacts page defaults to 52 contact rows selected because they
contain unreviewed leads. Each lead is still hidden until its contact row is
expanded. The filter also removes 156 non-actionable directory rows from the
default surface, making the rest of the directory appear permanently hidden.
The live payload currently contains 57 unreviewed leads across those 52
contacts, 208 total contacts, and 40 organization rows.

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 contract and red proof | This plan | Freeze 52 parent rows versus 57 hidden actions | plan and browser evidence | Current default has expand controls and no top-level approval rows |
| P1 flat projection | P0 | Add tested deterministic flattening of unreviewed leads | frontend utility and unit tests | One stable row is emitted per unreviewed hypothesis |
| P2 single-line review table | P1 | Add sortable/resizable approval table with direct controls and compact evidence/history | React, SVG icons, CSS | All judgment fields and actions fit one line; no cards, panels, or expansion are present |
| P3 honest mode toggle | P1-P2 | Make Approval rows and Directory peer views; remove directory row filtering | React and CSS | Approval rows default to 57; Directory shows 208 contacts and 40 organizations |
| P4 installed validation | P2-P3 | Build, test, restart, and visually inspect without approving anything | service and private QA artifacts | Desktop/mobile QA and live readback pass with review-state counts unchanged |

The packets are a single tightly coupled frontend path and do not justify
subagent fan-out.

## Acceptance criteria

- The Contacts surface defaults to `Approval rows 57` and offers a compact
  peer `Directory 208` toggle.
- Approval mode renders exactly one top-level row for every unreviewed
  hypothesis. It contains no contact expand button, expanded row, evidence
  disclosure, activity table, or source table.
- Each approval row exposes in one line: contact, proposed affiliation or role,
  compact source/match evidence, transcript/calendar/email history, explicit
  person target, explicit organization target, editable role title when
  applicable, and the three existing SVG actions.
- Approval columns are visibly sortable where meaningful and operator
  resizable. Text truncates with full judgment context available through
  accessible labels and native title text.
- Directory mode always renders the complete selected directory response. Its
  `All contacts`, `Organizations`, and `Unresolved` views retain their current
  sortable/resizable table and optional detail expansion.
- Existing stale refresh, in-flight gate, idempotency, explicit-target,
  provider-write, and authority semantics remain unchanged.
- Frontend red/green tests, focused Python tests, comprehensive provider-free
  tests, production build, planning audits, desktop/mobile Agent Browser
  checks, installed service health, and live decision-count preservation pass.

## Non-goals and effect boundaries

- No provider read/write, corpus refresh, Graphiti write, speaker assignment,
  person merge, automatic acceptance, schema migration, or backend authority
  relaxation.
- No review action during automated or visual validation.
- No large card, panel, pill, button, modal, or second-line metadata block.

## Current state after opening

P0 is complete. P1 begins with a deterministic flattening test. P2-P4 remain
open.

## Current state after completion

P1-P4 are complete. Contacts now defaults to a peer `Approval rows` mode that
deterministically flattens all 57 unreviewed hypotheses into 57 directly
actionable top-level rows. Each 33-pixel row contains the contact, proposed
affiliation or role, compact provider and exact-match basis, transcript,
calendar, and email counts with SVG channel icons, both explicit authority
targets, the editable role title when applicable, and three SVG decision
actions. There are no contact expanders, expanded detail rows, evidence
disclosures, activity tables, or source tables in this mode.

The peer `Directory 208` mode restores the complete selected response without
an implicit approval filter. Its All contacts view renders 208 rows and its
Organizations view renders 40 rows; the existing optional detail expansion is
retained only there. Meaningful approval columns are visibly sortable and all
column boundaries are pointer- and keyboard-resizable.

Frontend unit tests, 15 focused backend/API tests, the full 1,283-test
provider-free suite, production build, diff hygiene, installed service
restart, and desktop/mobile Agent Browser review passed. At 1440 by 900 the
entire eight-column judgment row fits without table overflow. At 390 by 844
the page has no horizontal overflow and the wide table owns its internal
scroll. The exact QA session was closed.

The installed service remains active with zero restarts. Final readback is
five accepted and 57 unreviewed leads, seven accepted person targets, and one
accepted organization target; QA made no review decision or provider change.
