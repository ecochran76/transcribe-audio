# Plan 0077 | Multi-Organization Role Affiliations

State: CLOSED

Lane: P09

Date: 2026-09-01

Related authority: Plan 0076, Plan 0072 A1, `CONTEXT.md`,
`docs/adr/0004-derive-affiliations-from-role-appointments.md`, and `VISION.md`

## Scope

Make the canonical People directory correctly represent a person who belongs
to several organizations and holds several concurrent or historical roles at
the same organization. Preserve every role assertion by its durable `role_id`,
derive one affiliation group per person/organization pair for presentation,
and keep provider organization strings visibly provisional.

Add an intentional compact projection: one ranked current affiliation in the
People row, an explicit `+N organizations` summary, and an expanded table that
lists every organization and every role appointment with its review state,
validity interval, and evidence count. Expose accepted, temporally valid role
appointments through the shared evidence fabric without mixing proposed roles
into contextual authority.

## Vision outcomes and maturity movement

This plan advances speaker grounding, relationship/history context,
provenance, temporal integrity, and reusable accepted knowledge from
`VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Multi-organization identity | Level 1 ledger rows exist, but the directory overwrites earlier roles for the same organization | Level 2 deterministic affiliation groups over every preserved role appointment | Adversarial two-organization/three-role fixture, replay-equal semantic hash, and API readback |
| Human review projection | Level 2 compact directory, but only the first organization/role is legible | Level 2 compact primary projection plus complete inline affiliation expansion | Desktop/mobile Agent Browser evidence with no cards, large pills, or text icons |
| Temporal contextualization | Level 1 relationships and activities are as-of filtered, but role appointments are not retrievable | Level 2 accepted-only, as-of role retrieval | Before/during/after validity tests, proposal exclusion, and current-conversation exclusion |

The measurable change is zero collapsed role appointments in a person/org
group and complete compact readback of additional organizations. This plan
does not claim automatic role discovery, accepted live role authority,
speaker-identity lift, or Level 3 automation.

## Current State

Schema v9 already stores each role assertion separately in
`knowledge_identity_role_projection` with a stable `role_id`, person,
organization, scope, validity dates, status, evidence IDs, and metadata.
`people_organization_activity.build_directory_index` currently removes every
existing affiliation for the same `organization_id` before appending the next
role, so only the last role survives in the directory. The React row then
reads `organizations[0]`, and expanded affiliation rows use only
`organization_id` as their key.

The live schema-v9 ledger currently has zero role rows. That means this defect
can be corrected without migrating or rewriting live role evidence. Current
provider-derived organization strings remain proposed observations, not
accepted role appointments.

## Domain and architecture contract

- A `role appointment` is one independently reviewable temporal assertion,
  identified by `role_id`; exact duplicate role IDs replay idempotently.
- An `affiliation` is a deterministic read-model grouping of one person and
  one organization. It contains zero or more role appointments and is not a
  second mutable source of truth.
- Distinct role IDs are never deduplicated merely because person,
  organization, or title match. Rebuild deduplicates only the ledger's exact
  durable assertion identity.
- Provider organization strings may create a proposed affiliation with no
  role appointments. They never become accepted employment or membership.
- The compact `primary_affiliation` is a display projection, never authority.
  Rank accepted/reviewed and active appointments first, then proposed active
  appointments, then the most recently ended appointment; use stable IDs as
  the final tie-breaker.
- Every role remains independently correctable or reversible through the
  existing append-only ledger.
- Context retrieval includes only accepted/reviewed appointments effective at
  the request `as_of` time and accepted no later than that time unless
  hindsight is explicitly allowed.

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 contract | This plan | Freeze terminology and derived-affiliation decision | `CONTEXT.md`, ADR, plan | Person, affiliation, role appointment, and primary projection are unambiguous |
| P1 projection | P0 | Preserve and group multiple roles across multiple organizations | projector and focused tests | Two orgs and three roles survive replay with stable grouping/order |
| P2 contextual retrieval | P1 | Add accepted temporal role appointments to the evidence bundle | evidence fabric and tests | Proposed, future, ended, hindsight, and self-originated roles obey policy |
| P3 API and compact UI | P1 | Render primary affiliation plus `+N`, and every role in expansion | existing People API/React surface | Dense desktop/mobile UI exposes all appointments and unique row identity |
| P4 installed validation | P1-P3 | Rebuild/restart/read back current live directory without accepting evidence | local store/service and private QA artifacts | Schema remains v9/clean, zero role rows remain truthful, full tests/build/browser QA pass |

P0-P1 are the critical path. P2 and P3 may follow independently after the
grouped response shape is fixed, then join at P4. Each packet has at most two
implementation attempts and one focused repair cycle before reframe. No
subagent is needed for this tightly coupled path.

## Acceptance Criteria

- One person with two organizations and two roles at one organization returns
  two affiliation groups and three role appointments; no appointment is lost.
- Reversing or correcting one role affects only that role and rebuilds to an
  identical semantic hash on replay.
- Provider string affiliations stay proposed and can coexist with authority
  roles without overwriting them.
- The API supplies `primary_affiliation`, `additional_organization_count`,
  grouped `organizations[].roles[]`, stable affiliation IDs, and role IDs.
- People rows show a compact organization/name and role summary plus `+N
  organizations`; expanded detail lists every role, date range, state, and
  evidence count.
- Role context retrieval is accepted-only, anchor-scoped, temporal,
  budget-bounded, and excludes evidence originating in the current
  conversation.
- Focused regression tests demonstrate the pre-fix collapse, then pass; the
  comprehensive provider-free suite and Vite build pass without retry.
- Agent Browser confirms the existing dense sortable/resizable table remains
  usable at desktop and mobile widths with SVG controls and no large cards,
  panels, or pill buttons.
- Installed service readback remains healthy and reports the truthful live
  role/affiliation count. No live role acceptance is manufactured for QA.

## Non-Goals and effect boundaries

- No provider reads or writes, contact refresh, mail/calendar mutation, or
  Graphiti write.
- No automatic role or organization acceptance, person merge, or speaker
  assignment.
- No schema-v10 migration: the existing role projection is already the
  durable role authority; affiliation is deliberately derived.
- No ontology redesign for employment/ownership/advisory relation types in
  this slice.
- No claim of contextual or speaker-quality improvement without a later
  accepted live cohort and measured chronological comparison.

## Validation

- Focused projector, ledger, evidence-fabric, workflow/API tests.
- Deterministic rebuild and semantic-hash equality over adversarial roles.
- `python -m py_compile` for changed Python modules.
- `npm --prefix frontend run build`.
- Comprehensive provider-free `pytest -q` with exact count and duration.
- Planning audit in active-only and goal-only modes.
- Installed `transcripts.service` state, schema, directory/API, and process
  restart readback.
- Named Agent Browser desktop/mobile session, screenshots, DOM/accessibility
  evidence, and exact-session closeout.

## Current State After Closure

P0-P4 are complete at implementation checkpoint `85aa922`. Schema v9 remains
the authority: no migration or accepted-evidence rewrite occurred. The v2
directory projection preserves every stable role ID, derives deterministic
person/organization affiliation groups, exposes a compact primary-plus-count
summary, and lists all role appointments in the inline expansion. The evidence
fabric now retrieves accepted/reviewed roles only when anchor, effective-time,
acceptance-time, budget, and current-conversation boundaries permit them.

The adversarial two-organization/three-role fixture returns two affiliation
groups and three role rows through the real API. Focused projector, ledger,
evidence, workflow, and API checks pass; the provider-free comprehensive suite
passes 1,276 tests in 75.03 seconds without retry; Python compilation and the
Vite production build pass. Agent Browser confirmed desktop and 390-by-844
mobile layouts, SVG controls, explicit sort state, keyboard resizing from 20
to 22 percent, zero directory buttons taller than 48 pixels, and three unique
expanded role rows. The named browser session and synthetic server were
closed; private screenshots remain under the Plan 0077 operator-state folder.

Installed readback after restarting `transcripts.service` reports PID 26539,
`NRestarts=0`, schema v9 sidecar authority, `dirty=0`, and database integrity
`ok`. The v2 API is byte-stable and reports 59 displayed provider-observed
affiliations but zero displayed or authority role rows. Those affiliations
remain proposed; no live role acceptance was manufactured. The plan therefore
advances the intended Level 2 representation and review/context seams without
claiming Level 3 discovery automation or measured speaker-quality lift.
