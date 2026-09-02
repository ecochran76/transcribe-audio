# Plan 0083 | Accepted Identity Repair Queue

State: CLOSED

## Scope

Build a compact, typed repair workflow for already accepted directory decisions.
The workflow derives repair findings from current evidence and accepted identity
authority, exposes them beside accepted organization/role decisions, and records
corrections as immutable stale-safe ledger events.

The first supported repair actions are:

- correct a canonical person's primary name to a human-name candidate already
  present in retained source evidence;
- merge one of two explicitly selected accepted people after an operator judges
  the evidence to represent the same person;
- reject or defer an accepted organization/role decision through the existing
  correction path.

The same slice prevents future create-person approvals from using an email,
incomplete label, or organization label as the canonical person name. Person
display projection excludes exact organization labels while preserving them as
source evidence.

## Non-Goals

- No automatic repair, inferred merge, bulk acceptance, or live ledger mutation
  during installation or QA.
- No provider, mailbox, calendar, CRM, speaker, biometric, Graphiti, or external
  write.
- No deletion or in-place rewrite of historical identity events.
- No assertion that a role mailbox such as `research@...` belongs to a named
  person without a separate operator identity decision.
- No automatic merge based on equal display names alone.

## Vision Alignment

- Advances north-star outcomes 3, 6, 7, and 8 by improving reusable person and
  relationship authority while retaining provenance and uncertainty.
- Advances identity quality and knowledge integrity from Level 2 shadow review
  with hidden repair debt to Level 2 operator-ready correction with deterministic
  replay and explicit unresolved cases.
- Target evidence is the installed repair queue, stale/idempotency tests,
  deterministic replay tests, live read-only counts, and desktop/mobile visual
  proof that no unrelated directory rows consume the repair surface.
- This does not prove Level 3 automatic identity resolution or contextual lift.

## Current State

- The installed schema is version 10 and the live repair projection is version
  1. The directory projection is version 5.
- The live default shows 23 actionable identity repairs: 21 evidence-backed
  canonical-name corrections and two possible-duplicate decisions. A compact
  `Actionable only` toggle reveals five additional identity-ambiguity findings.
- Fifty-nine accepted organization and role decisions remain visible as flat
  correction rows below the identity repairs; two unreviewed approval rows
  remain in the separate approval mode.
- The immutable identity ledger still contains 211 events. Installation and
  browser QA submitted no repair or review action.
- The ledger now supports typed append-only person correction. Person merges
  retarget roles, activity subjects and coverage, relationship endpoints,
  sources, and external identities during deterministic replay.
- Exact organization labels are excluded from person-name candidates. A
  company-only contact projects as `Unknown person`, and future create-person
  approvals require an explicit complete human name.
- Graphiti was healthy but returned no relevant recent Plan 0081/0082 repair
  memory. Current source, live readback, and closed plan artifacts remained the
  authority for this slice.

## Execution

### P0 | Regression-first contract

- Add focused failing tests for organization-label exclusion, explicit safe
  create-person names, person correction replay, stale/idempotent repair
  submissions, duplicate repair projection, and API behavior.

### P1 | Append-only person correction authority

- Add a `person_corrected` identity event with the same allowlisted correction
  semantics as organization correction.
- Build deterministic typed repair findings and stale-safe correction receipts.
- Keep incomplete/shared-mailbox findings visible when no safe corrective action
  is supported.

### P2 | Compact repair UI and future-review guard

- Add a peer `Repairs` work mode that defaults to actionable repair findings
  and accepted decision rows, with one compact control to reveal unresolved
  non-actionable issues.
- Keep rows dense, sortable, resizable, and directly actionable with compact SVG
  controls.
- Require an explicit evidence-backed person name whenever `Create person` is
  selected and reject company/identifier/incomplete names server-side.

### P3 | Validation and install

- Run focused backend/frontend tests, the provider-free presubmit suite,
  compilation, production build, planning audit, and diff hygiene.
- Install the frontend/backend, restart the user service, and perform read-only
  desktop/mobile Agent Browser QA. QA must not invoke a repair or review action.
- Confirm accepted counts and event count are unchanged by installation and QA.

## Acceptance Criteria

- Accepted name, duplicate, ambiguity, and accepted-decision repair rows are
  visible without expanding ordinary contacts.
- Exact organization labels cannot become derived person display names.
- A new create-person approval cannot persist an identifier-only, incomplete, or
  exact organization name as `primary_name`.
- A reviewed name correction or merge is append-only, deterministic,
  idempotent, and stale-safe.
- Existing accepted relationship/role decisions can be rejected or deferred from
  the flat repair surface.
- No repair occurs merely because the queue was generated, installed, or viewed.

## Validation

- Focused: `python -m pytest tests/test_people_organization_activity.py tests/test_identity_learning_ledger.py tests/test_person_identity_repair.py tests/test_directory_hypothesis_review.py tests/test_transcript_api.py -q`
- Frontend: `npm --prefix frontend test`
- Build: `npm --prefix frontend run build`
- Compile: `python -m py_compile identity_learning_ledger.py people_organization_activity.py person_identity_repair.py directory_hypothesis_review.py identity_review_workflow.py transcript_api.py`
- Presubmit: repository provider-free test suite.
- Governance: `python .codex/skills/repo-policy-selector/scripts/audit_planning_contract.py --repo-root . --active-only --json`
- Live: health, repair counts, immutable ledger count before/after, service
  status, and Agent Browser desktop/mobile screenshots with console inspection.

## Definition of Done

The live dashboard provides a compact repair mode backed by immutable correction
semantics, new approvals cannot silently create company/identifier names as
people, validation passes, installation is healthy, visual QA is clean, and the
live identity event count remains unchanged until the operator deliberately
chooses a repair action.

## Completion Evidence

- Focused backend selections passed 116 tests and the full provider-free suite
  passed 1,296 tests in 101.02 seconds.
- Frontend tests passed eight tests. The production build, Python compilation,
  planning audit, and diff hygiene passed.
- The live database migrated from schema 9 to 10 with a pre-migration backup at
  `/home/ecochran76/.transcripts/backups/transcripts-pre-migrate-v9-61ebbf2ce4b6.sqlite3`.
- `transcripts.service` restarted healthy with `NRestarts=0`. Accepted, rejected,
  deferred, and unreviewed review-state counts remained 59, 1, 0, and 2.
- Agent Browser inspected the installed view at 1440 by 900 and 390 by 844. The
  default contained 23 repair rows and 59 accepted-decision rows; toggling the
  filter exposed all 28 repair findings and restored the default. Repair rows
  measured about 31 pixels high, both tables retained horizontal overflow within
  their wrappers on the narrow viewport, four sort controls and four resize
  handles were exposed, all 23 actionable controls used SVG icons, and console
  and page errors were empty.
