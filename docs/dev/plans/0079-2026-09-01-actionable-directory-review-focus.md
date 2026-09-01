# Plan 0079 | Actionable Directory Review Focus

State: CLOSED

Lane: P09

Date: 2026-09-01

Related authority: Plan 0078, `VISION.md`, `CONTEXT.md`, and
`docs/adr/0004-derive-affiliations-from-role-appointments.md`

## Scope

Repair the installed Contacts review workflow so its default people view shows
only rows with an unreviewed organization or role lead, while retaining a
compact operator-selectable path to all rows. Keep review actions above
collapsed activity/source/affiliation evidence. Label accepted target counts
explicitly, and prevent a rapid duplicate submission from making a successful
first decision appear to fail with a stale-projection error.

## Vision outcomes and maturity movement

This plan advances the review/acceptance stage, provenance, deterministic
replay, and self-feeding knowledge loop in `VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Review focus | Level 2 actions exist inside a 208-row mixed directory | Level 2 operator-focused queue with an actionable default and configurable broader scope | Live default renders only rows containing unreviewed leads; all-row scope remains available |
| Evidence density | Level 2 evidence is present but automatically expands before actions | Level 2 progressive disclosure keeps actions first and evidence one compact disclosure away | Browser expansion shows review controls before a closed evidence/history disclosure |
| Submission feedback | Backend is stale-safe, but the UI can send two requests before React disables a row | Level 2 fail-closed concurrency with honest recovery | A synchronous client gate suppresses duplicate in-flight submits; a 409 refreshes current state instead of masking a committed first decision |
| Organization targets | Accepted targets are safe but presented as an unexplained count | Level 2 authority-aware labelling | Target control says accepted organizations and proposed organization creation is explicit |

This slice improves the usability and truthfulness of explicit human review. It
does not claim automatic identity resolution, speaker-identification lift, or
quality of unreviewed leads.

## Current evidence

- The installed `/api/people?limit=500&view=people` response contains 208 rows,
  but only 52 rows contain an unreviewed review lead.
- The browser has no row-scope selector and expands a 428-entry activity
  timeline for the first non-actionable row.
- The same live response contains 40 proposed organizations and one accepted
  organization target; the target selector does not distinguish those states.
- Scott Roberts is accepted at projection version 2. The current React handler
  has no synchronous in-flight guard and generates a new idempotency key for
  every invocation, so two rapid invocations can yield one successful commit
  followed by one correct stale rejection.

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 reproduce and contract | This plan | Freeze live counts and UX failure without review mutation | plan, red browser evidence | The 208/52 row mismatch and automatic evidence expansion are measured |
| P1 pure row-scope behavior | P0 | Add tested actionable/decided/all filtering semantics | frontend utility and tests | Node tests prove actionable is default logic without hiding the all-row path |
| P2 dense review UI | P1 | Add compact scope selector, action-first expansion, and authority labels | React and CSS | Review controls precede a closed evidence disclosure; no large card or button is added |
| P3 stale feedback repair | P1 | Gate duplicate in-flight action and refresh on stale response | React and tests/build | One row cannot dispatch twice concurrently and 409 causes readback rather than a false terminal failure |
| P4 installed validation | P2-P3 | Build, test, restart, and visually verify the live service without taking an action | service and private QA artifacts | Live default shows 52 actionable rows, all rows remain selectable, service is healthy, and accepted/unreviewed counts are preserved |

P0-P3 are one tightly coupled frontend repair and do not justify subagent
fan-out. P4 joins the tested behavior with installed readback.

## Acceptance criteria

- People and Unresolved views default to `Needs review`; the operator can select
  `All rows` or `Reviewed / deferred` from one compact control.
- `Needs review` includes only rows with at least one `unreviewed` directory
  review lead. Accepted, rejected, deferred, and no-action rows do not pollute
  the default view.
- Organization browsing remains available and is not incorrectly filtered by
  person-review state.
- Expanding an actionable row shows organization/role actions first. Activity,
  source identities, affiliations, and reconciliation caveats are closed under
  one evidence/history disclosure by default.
- Create options explicitly say they create from the reviewed contact or
  proposal; existing target options explicitly say they use accepted people or
  organizations.
- A synchronous in-flight gate prevents two requests from one rapid UI action.
  A backend 409 triggers current projection readback and does not claim the
  rejected second request changed authority.
- Existing backend stale, idempotency, explicit-target, and provider-write
  boundaries remain unchanged.
- Focused tests, Python compilation, production build, comprehensive
  provider-free tests, planning audits, desktop/mobile Agent Browser checks,
  installed service health, and live count preservation pass.

## Non-goals and effect boundaries

- No provider read/write, corpus refresh, mail/calendar mutation, Graphiti
  write, speaker assignment, person merge, or automatic acceptance.
- No backend relaxation of optimistic concurrency or idempotency semantics.
- No schema migration and no new authoritative organization class.
- No review action during automated or visual validation.

## Current state after opening

P0 is complete. P1 starts with a frontend regression test for row-scope
classification. P2-P4 remain open.

## Current state after completion

P1-P4 are complete. The installed Contacts page defaults to 52 actionable rows
from the current 208-row contact directory and exposes reviewed/deferred and
all-row scopes through one compact select. The first tab now truthfully reads
`All contacts 208`; the Organizations view still lists all 40 proposed or
accepted organization rows independently of person-review state.

Expanded actionable rows render the compact review table first, followed by a
closed evidence/history disclosure. Existing target optgroups explicitly say
they use seven accepted people or one accepted organization, while create
options say they create from the reviewed contact or proposal. No authority
semantics changed.

The action handler now has a synchronous in-flight gate. HTTP errors preserve
their status, and a stale 409 triggers current projection readback instead of
showing a terminal false failure. A browser-intercepted synthetic double click
produced exactly one POST, refreshed the row on the synthetic 409, and created
no live effect.

Frontend unit tests, 77 focused Python tests, the full 1,283-test provider-free
suite, production build, diff hygiene, desktop/mobile Agent Browser review,
installed service restart, and live readback passed. The live review states
remain five accepted, 57 unreviewed, zero rejected, and zero deferred.
