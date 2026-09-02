# Plan 0083 | Accepted Identity Repair Queue

State: OPEN

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

- The live ledger contains 59 accepted directory decisions affecting 57 accepted
  people.
- Twenty-one accepted people currently have a better derived human display name
  than their stored primary name; five accepted people remain identity-ambiguous;
  two
  possible duplicate-name pairs require human evidence.
- Thirteen accepted contacts carry an organization label in their source alias
  pool. Two unreviewed contacts currently display an organization label as the
  person, demonstrating that presentation filtering must precede more reviews.
- The ledger supports organization, source, role, and relationship corrections,
  merges, splits, and reversals, but has no typed person correction event or
  compact accepted-repair surface.
- Graphiti is healthy but returned no relevant recent Plan 0081/0082 repair
  memory. Current source, live readback, and closed plan artifacts are authority.

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

- Add a peer `Repairs` work mode containing only repair findings and accepted
  decision rows.
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
