# Plan 0084 | Person Name And Organization Reconciliation

State: CLOSED

Lane: P09

Date: 2026-09-04

Related authority: Plans 0076, 0077, and 0083; `VISION.md`;
`docs/correction-first-identity-learning-contracts.md`

## Scope

Extend the installed correction-first identity workflow with two related,
operator-controlled reconciliation queues:

- person candidates that recognize compatible preferred-name, middle-name,
  and initial variants across accepted and unresolved source clusters; and
- organization candidates that keep name equivalence, organizational
  containment, and other institutional relationships as separate decisions.

The first person fixture is the live `R. Chris Williams` / `Chris Williams`
case. The first organization fixture is the live Iowa State family: short and
full institutional names, named departments or offices, and related entities
must not be flattened into one string or one undifferentiated organization.

Every action remains append-only, idempotent, stale-safe, and reversible under
the existing identity ledger. Candidate generation and browser QA perform zero
identity mutation.

## Vision Alignment

- Advances north-star outcomes 3, 4, 6, 7, and 8 by improving person identity,
  institutional context, provenance, accepted knowledge, and later retrieval.
- Moves person reconciliation from Level 2 exact-display-name review to Level
  2 structured name-variant review with explicit negative decisions.
- Moves organization authority from Level 1 populated names with empty
  hierarchy metadata to Level 2 operator-reviewed aliases, containment, and
  typed relationships.
- Evidence is an installed compact queue, public-interface red/green tests,
  deterministic replay, live read-only accounting, and desktop/mobile Agent
  Browser proof. This plan does not claim Level 3 automatic reconciliation or
  measured speaker-identification lift.

## Current State

- Schema 10 contains 224 immutable identity events after the operator's latest
  reviews. The exact-display duplicate queue is empty; 19 canonical-name
  corrections and five identity ambiguities remain.
- The accepted `Chris Williams` person retains `R. Chris Williams` as an alias.
  A separate unresolved `R. Chris Williams` cluster and a separate unresolved
  `Chris Williams` cluster remain. Exact display-name grouping cannot surface
  the preferred-middle-name case safely.
- The ledger already supports person/organization corrections, aliases,
  merges, reconciliation decisions, roles, and generic typed relationships.
  Person repair discovery currently compares only accepted people with equal
  display names and offers no persistent distinct-person decision.
- Thirty-eight accepted organizations exist, but zero currently have aliases,
  organization types, parent organizations, merge redirects, or
  organization-to-organization relationships.
- Reviewed organization rows include `Iowa State University`, `The Iowa State
  University of Science and Technology`, `Iowa State University Department of
  Chemical and Biological Engineering`, and `IPRT, ISU`; their intended
  equivalence or relationship has not been reviewed.
- Graphiti is healthy but returned no useful current recall for this slice.
  Current source, Plans 0076/0077/0083, and live readback are authoritative.

## Domain Contract

### Person names

- Preserve every original label as evidence. Parse a review-only name shape
  containing ordered given tokens/initials, preferred-name candidate, family
  name, prefixes, and suffixes without asserting that token position proves a
  legal first or middle name.
- A missing initial and a supplied initial may be compatible. Two supplied,
  conflicting initials are contradictory. Compatible name shapes create a
  review lead only.
- Exact person-specific external identity remains the only existing automatic
  source-link rule. Name, organization, activity, and relationship agreement
  may rank a review candidate but never merge people automatically.
- A reviewed `distinct` decision must suppress the exact candidate pair until
  its evidence fingerprint changes; it must not erase either source cluster.

### Organizations

- `alias` means two labels identify the same organization. A reviewed merge
  preserves both labels, source records, affiliations, and redirects.
- `unit_of` means two distinct organizations or organizational units have a
  reviewed containment relationship. It does not merge their identities.
- `related_to`, `predecessor_of`, and `successor_of` are typed, temporal
  organization relationships. They do not imply containment or equivalence.
- A reviewed `distinct` decision suppresses the exact pair at the current
  evidence fingerprint.
- Acronyms, shared domains, string containment, and institutional prefixes are
  candidate evidence only. Provider strings never become accepted aliases or
  relationships without an operator action.

## Execution

### P0 | Person name-variant tracer

- Add one public-interface failing test for the preferred-middle-name fixture.
- Implement structured compatibility and emit one ranked
  `name_variant_candidate` across accepted and unresolved directory entities.
- Add conflicting-initial and unrelated-same-name negative fixtures.

### P1 | Persistent person reconciliation decisions

- Extend the repair submission interface with an explicit `distinct` action.
- Reuse the immutable reconciliation-decision ledger and suppress only the
  exact pair plus bound evidence watermark.
- Preserve existing correction and merge idempotency, stale checks, adoption,
  replay, and redirects.

### P2 | Organization reconciliation module

- Add a deep module exposing one queue projection and one exact review method.
- Generate alias, containment, and ambiguous-related candidates from accepted
  organization evidence without mutation.
- Apply explicit merge, unit-of, typed-related, or distinct decisions through
  existing ledger primitives; preserve aliases and relationship provenance.

### P3 | Compact installed workflow

- Add organization reconciliation beside person repairs with one-line,
  sortable, resizable rows and compact SVG-only actions.
- Keep candidate type, both entities, evidence summary, decision selector, and
  action in one row; no cards, panels, large pills, or hidden expansion gate.
- Install after disposable replay and backup, then run read-only desktop/mobile
  Agent Browser QA without clicking a repair action.

## Parallel And Critical Path

- P0-P1 are the critical path because the established person-repair seam and
  negative-decision semantics define the reusable review contract.
- After that contract is green, organization queue projection and frontend
  rendering can be developed independently, but ledger action semantics and
  API installation remain serialized.
- One primary agent owns the critical path. No subagent is authorized or
  required by this execution.

## Acceptance Criteria

- The live unresolved `R. Chris Williams` cluster is offered as a review
  candidate for the compatible accepted person, while the separate SoyLei
  `Chris Williams` remains separate unless its evidence independently supports
  a candidate.
- Compatible missing/full initials and preferred-middle-name forms produce
  review candidates; conflicting supplied initials do not.
- Person and organization name similarity never performs an automatic merge.
- Reviewers can record `same` or `distinct` for a person pair. The decision is
  immutable, idempotent, stale-safe, and prevents an unchanged rejected pair
  from recurring.
- Iowa State short/full-name equivalence can be reviewed separately from
  department/unit containment and other institutional relationships.
- Organization merge preserves aliases and retargets accepted roles,
  relationships, activities, coverage, and source evidence. Unit/related
  decisions preserve distinct organization IDs.
- The Repairs surface remains dense, searchable, sortable, resizable, and
  directly actionable with SVG controls on desktop and narrow screens.
- Installation and visual QA change no repair decision or identity event.

## Validation

- Run one red/green public-interface test per behavior before widening.
- Focused backend: person/organization reconciliation, identity ledger,
  directory projection, workflow, and transcript API tests.
- Focused frontend tests for candidate rendering, decision selection, counts,
  compact controls, and refresh behavior.
- Provider-free presubmit suite, Python compilation, frontend production build,
  planning audit, CodeGraph status, and diff hygiene.
- Disposable migration/replay and live pre/post event accounting with backup.
- Agent Browser inspection at 1440 by 900 and 390 by 844 with console/page
  errors, row height, sort/resize controls, SVG actions, and overflow measured.

## Non-Goals And Effect Boundaries

- No automatic fuzzy/name/acronym/domain/organization merge or relationship.
- No provider, mailbox, calendar, CRM, Graphiti, speaker, biometric, or public
  write.
- No attempt to populate organizations absent from retained evidence in this
  slice.
- No claim about the legal relationship among ISU, OIPTT/OIC, ISURF, or ISUF
  without reviewed source evidence.
- No deletion or in-place rewrite of identity history.

## Definition Of Done

The live dashboard exposes compact person-name and organization reconciliation
rows, every action is an explicit append-only review decision, the Iowa State
cases can be classified without flattening distinct entities, current review
state and identity-event accounting are preserved during installation/QA, and
all bounded validation gates pass.

## Outcome

- The installed live queue contains 20 actionable person repairs and three
  actionable organization repairs. Its first person row is `Chris Williams /
  R. Chris Williams`; the unrelated SoyLei Chris cluster is not offered by
  this evidence. The first organization rows separately offer Iowa State
  formal-name equivalence, IPRT relatedness, and department containment.
- Explicit person and organization `distinct` decisions are persisted through
  immutable reconciliation events and suppress only an unchanged candidate
  fingerprint. No name or acronym heuristic applies a merge automatically.
- Organization merge replay now retargets source records, roles, activities,
  activity coverage, relationships, and child-unit parent pointers while
  preserving the source name as an alias.
- `GET` and `POST /api/organization-repairs` expose the same bounded,
  stale-safe, idempotent contract as person repairs. Installation added no
  schema migration.
- The compact installed view puts the three organization decisions and the
  Williams decision first. Rows are 31 pixels high, headers visibly sort,
  pointer and keyboard resizing work, actions contain SVGs without text, and
  `Actionable only` hides 61 historical accepted rows by default.
- The provider-free suite passed 1,305 tests in 97.67 seconds. The focused
  backend/API selection passed 96 tests, frontend tests passed eight, the
  production build and Python compilation passed, and the planning audit and
  diff hygiene were green.
- Before restart, the live database was backed up to
  `/home/ecochran76/.transcripts/backups/transcripts-pre-plan0084.sqlite3`.
  The installed service is healthy with zero restarts. Desktop and 390-pixel
  Agent Browser inspection found no console or page errors and no page-level
  horizontal overflow; narrow table overflow remains locally scrollable.
- Installation and browser QA left the identity ledger at 224 events. No
  repair action, provider action, or external write was performed.
