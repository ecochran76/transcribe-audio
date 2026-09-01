# Plan 0078 | Organization And Role Review Workflow

State: CLOSED

Lane: P09

Date: 2026-09-01

Related authority: Plans 0075-0077, `CONTEXT.md`, `VISION.md`, and
`docs/adr/0004-derive-affiliations-from-role-appointments.md`

## Scope

Restore organization and role leads to the canonical dense Contacts directory
and make them directly reviewable. A reviewer may accept, reject, or defer one
source-backed affiliation or role lead. Acceptance must show and bind an exact
canonical person target and organization target; each target may be an
existing accepted entity or an explicit deterministic creation from the
selected source record. A role acceptance creates one independent temporal
role appointment. An affiliation acceptance creates one accepted
`AFFILIATED_WITH` relationship without inventing a title.

Keep the interface compact: one sortable/resizable review-lead table inside
the existing inline person expansion, small SVG action controls, and no cards,
large panels, pills, or modal approval ceremony.

## Vision outcomes and maturity movement

This plan advances the review/acceptance stage, organization and role context,
provenance, deterministic replay, and the self-feeding knowledge loop in
`VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Organization/role review | Level 1 hypotheses exist, but the canonical directory dropped their review surface | Level 2 dense operator review over the live proposed cohort | Current 59 affiliation and 3 role leads appear with source-backed status and compact controls |
| Accepted identity graph | Level 1 ledger schemas exist, with zero live people, organizations, roles, or relationships | Level 2 explicit operator decisions can create/link person and organization authority and one role or affiliation assertion | Hermetic accept/reject/defer tests, deterministic rebuild, and one isolated synthetic browser acceptance |
| Context reuse | Level 2 accepted roles are temporally retrievable, but no review path can produce them | Level 2 accepted review output immediately enters the canonical directory and evidence fabric | API readback and accepted-only evidence-fabric test |

This slice proves the review and authority transition. It does not prove
automatic acceptance, speaker-identity lift, or the quality of any unreviewed
live lead.

## Current State

The installed v2 directory is healthy but carries only activities, source
records, and affiliations into its inline expansion. The earlier
`HypothesisTable` is no longer mounted. The source projection currently holds
59 provider organization-string affiliation leads and three exact-email role
leads; all remain unreviewed. The immutable mail cohort contains 120 additional
relationship leads, but those are outside this plan except that their existing
review contract must not regress.

The authority ledger currently contains zero people, organization, role, and
relationship rows. A local source contact is not itself a canonical person.
Therefore accepting an organization string cannot silently imply person
identity: the submission must name an existing canonical person or explicitly
create one from the exact reviewed local contact record.

## Domain and interface contract

- A `directory review lead` is an immutable, content-hashed projection of one
  contextual-role or affiliation hypothesis plus its source contact.
- `accept`, `reject`, and `defer` are stale-safe and idempotent. A reused
  idempotency key with different content fails closed.
- Reject and defer append review history but create no person, organization,
  role, relationship, provider, speaker, or Graphiti effect.
- Acceptance requires exact person and organization target modes. `create`
  uses stable IDs derived from the reviewed source identity and normalized
  organization name; `existing` must resolve to current authority.
- Creating a person also links the exact local contact source record through a
  bounded local-source observation. It does not merge same-name contacts or
  infer provider identity.
- Role acceptance preserves the reviewed title as an independently
  correctable `role_id`; affiliation acceptance preserves membership without
  inventing a role type.
- Multiple ledger events for one acceptance are inserted as one atomic batch
  before deterministic projection rebuild.
- Accepted `AFFILIATED_WITH` relationships participate in the derived
  person/organization affiliation read model with `roles: []`.

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 contract | This plan | Freeze review lead and explicit target semantics | plan, roadmap, runbook | No acceptance path silently resolves person identity |
| P1 projection | P0 | Carry affiliation/role leads and review history into v3 directory output | discovery, directory projector, tests | Live-shaped fixture exposes all leads without duplicating people |
| P2 authority transition | P1 | Add one stale-safe review interface and atomic ledger batch | review module, ledger, tests | Accept/reject/defer and replay/collision behavior pass through public workflow |
| P3 API and compact UI | P1-P2 | Add endpoint and compact inline target/action controls | API, React, CSS, tests | Reviewer can choose/create targets and submit without cards or modal ceremony |
| P4 installed validation | P1-P3 | Build, test, restart, and read back without reviewing live leads | service and private QA artifacts | Schema remains clean; live leads remain unreviewed; synthetic review proves effects end to end |

P0-P2 are the critical path. P3 follows the frozen receipt contract. P4 joins
all packets. Each implementation packet has at most two attempts and one
focused repair cycle before reframe. No subagent is needed for this tightly
coupled authority path.

## Acceptance Criteria

- The canonical directory exposes all current affiliation and role leads with
  stable hypothesis IDs, content hashes, projection versions, and review state.
- A local-contact acceptance explicitly creates or selects a canonical person,
  explicitly creates or selects an organization, links only the reviewed local
  source record, and creates exactly one role or affiliation assertion.
- A canonical-person row defaults to that person but never silently chooses a
  same-name candidate for an unresolved contact.
- Accept is atomic, idempotent, stale-safe, and replay-deterministic. Reject and
  defer create zero accepted graph effects. Reversal remains possible through
  existing ledger event history.
- Accepted affiliation relationships render as accepted organization groups
  with no invented role. Accepted roles render as independent role rows and
  enter accepted temporal evidence retrieval.
- The API returns a bounded receipt with exact person, organization, role, and
  relationship effect counts and zero provider/speaker effects.
- Desktop and mobile Agent Browser checks show the compact table, explicit
  target controls, SVG actions, working sort/resize, and no large cards,
  panels, pills, or buttons.
- Focused tests reproduce the missing-lead and non-atomic preconditions, then
  pass; Python compilation, Vite build, comprehensive provider-free tests,
  planning audits, installed service health, and byte-stable live readback pass
  without retry.
- No live lead is accepted during installation or QA. Live before/after counts
  remain 59 affiliation leads, three role leads, and zero accepted authority
  rows unless the operator later reviews them.

## Non-Goals and effect boundaries

- No provider reads or writes, corpus refresh, mail/calendar mutation, or
  Graphiti write.
- No automatic acceptance, same-name merge, speaker assignment, biometric
  effect, or contextual quality claim.
- No review of correspondence/co-participation mail relationships; the Plan
  0075 review contract remains unchanged.
- No schema-v10 migration. The existing ledger and projection tables remain
  sufficient.
- No bulk accept or hidden default target selection.

## Validation

- Public-workflow RED/GREEN tests for projection and accept/reject/defer.
- Atomic ledger-batch and deterministic replay checks.
- Directory/API readback with existing and newly created target paths.
- Accepted affiliation and role evidence-fabric checks.
- `python -m py_compile` for changed Python modules.
- `npm --prefix frontend run build`.
- Comprehensive provider-free `pytest -q`, without retries.
- Planning audits in active-only and goal-only modes.
- Synthetic isolated API/browser fixture at desktop and 390-by-844 mobile.
- Installed `transcripts.service`, schema, authority counts, live lead counts,
  and byte-stable API readback.

## Current State After Opening

P0 is frozen. P1 begins with one public directory regression showing that the
current v2 response drops all affiliation and role review leads. P2-P4 remain
open. The plan grants only explicit operator-reviewed local ledger mutation;
it grants no provider or automatic acceptance authority.

## Current State After Completion

P1-P4 are complete. The installed v3 directory exposes all 62 current review
leads with immutable content hashes, projection versions, decision history,
and explicit person/organization target choices. The compact inline table is
sortable and resizable, uses 25-by-25 SVG accept/reject/defer controls, and
keeps narrow-screen overflow inside the directory surface.

The review endpoint commits each accepted person/source/organization and role
or affiliation transition as one atomic ledger batch. Hermetic public-workflow
tests cover accept, reject, defer, stale submissions, idempotent replay,
conflicting replay, accepted affiliation rendering, and HTTP behavior. The
installed live cohort remains unchanged at 59 affiliation plus three role
leads, all unreviewed; installation and Agent Browser QA created no accepted
authority, provider, speaker, or Graphiti effect.

Validation passed 96 focused tests and the full 1,283-test provider-free suite,
Python compilation, the Vite production build, diff hygiene, planning audits,
desktop and 390-by-844 Agent Browser inspection, service restart, health
readback, and stable runtime checks. `transcripts.service` is active with
`NRestarts=0`.
