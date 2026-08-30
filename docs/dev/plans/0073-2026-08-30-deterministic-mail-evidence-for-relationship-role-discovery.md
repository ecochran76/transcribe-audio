# Plan 0073 | Deterministic mail evidence for relationship and role discovery

State: PLANNED

Planning boundary: this turn authorizes planning artifacts only. It does not
authorize private mailbox reads, provider calls, live backfill, schema
migration, service deployment, accepted graph writes, speaker assignment,
contact/person merge, or Graphiti publication.

Lane: P09/P10

Branch: `plan-0037-campaign`

Target: `main`

Depends on: Plan 0025 provider evidence contracts, Plan 0029 conversation
knowledge authority, and Plan 0072 A6-R2/A6-R3 contact and shadow-graph
projections.

Critical-Path Owner: primary agent

## Scope

Add a bounded, deterministic Mail Receipts evidence lane to the existing
relationship and role discovery pipeline. The lane will query the installed,
user-scoped `mail_receipts` operator-lite workbench as the authority for owned
mail evidence and durable mail-corpus state. It will normalize exact
participant, direction, thread, time, and structured signature observations;
group duplicate copies of the same interaction; and produce replayable,
review-only relationship and contextual-role hypotheses for the Contacts
workspace and later evidence bundles.

The first implementation target is a Level 2 shadow projection. It must improve
the evidence available for human review without turning correspondence into
proof of identity, employment, meeting attendance, or a particular speaker.
Only a later reviewed decision may create an accepted graph fact.

The implementation should extend the current seams rather than create a second
mail index or graph authority:

- `conversation_evidence_adapters.py` remains the bounded source-envelope and
  redaction contract;
- a new Mail Receipts adapter uses only the public operator-lite service
  profile and workbench tools (`search_mail`, selected-result context, people,
  neighborhood, and relationship-path reads) and follows returned opaque
  evidence links rather than private storage paths;
- configured GWS and Odollo adapters remain supplemental source-specific seams
  when a later packet explicitly needs direct provider metadata unavailable in
  the owned corpus;
- `relationship_role_discovery.py` remains the deterministic shadow projection;
- the user-scoped conversation knowledge store remains the local authority for
  source records, evidence snapshots, decisions, and rebuildable projections;
- `/api/people` and the compact Contacts detail remain the operator read surface;
  and
- Graphiti remains an optional projection for reviewed facts only, never raw
  messages or unreviewed hypotheses.

## Vision outcomes and maturity movement

| Capability | Current maturity | Target for this plan | Evidence |
| --- | --- | --- | --- |
| Relationship discovery | Level 2 calendar/provider shadow hypotheses | Level 2 multi-source shadow hypotheses with deterministic mail interaction evidence | Stable replay, source and independence receipts, temporal bounds, and review outcomes |
| Contextual role discovery | Level 1/2 sparse provider-declared roles | Level 2 structured mail-signature and exact-source role observations with conflict preservation | Provenance-complete role candidates, effective-time labels, and stale/conflict handling |
| Speaker deduction support | Level 2 contextual evidence with low coverage | Level 2 mail-assisted candidate/context packets, still review-only | Candidate recall, appropriate abstention, contradiction visibility, and zero automatic assignments |
| Conversation contextualization | Level 2 local transcript/calendar retrieval | Level 2 exact-first mail context usable through the immutable evidence-bundle seam | Bounded retrieval receipts, as-of enforcement, and useful reviewed context findings |
| Provenance and reusable knowledge | Level 2 rebuildable local observations | Level 2 cross-source independence and deterministic shadow graph replay | Input watermark equality, duplicate grouping, tenant isolation, and accepted-effect count of zero |

This advances VISION outcomes 3, 4, 6, 7, and 8: speaker understanding,
conversation context, provenance, knowledge projection, and retrieval. It does
not advance automatic identity acceptance beyond Level 2.

## Current State

- Plan 0072 A6-R2 accounts for 186 recording-associated attendee emails and
  enriches them through exact-email, read-only GWS/Odollo contact observations.
- Plan 0072 A6-R3 deterministically exposes 3 contextual-role, 59 affiliation,
  and 279 recurring calendar co-invitation hypotheses across 103 contacts. All
  are `Needs review`; accepted role and relationship projections are empty.
- `GwsEvidenceAdapter` already requires explicit capabilities and query terms,
  enforces record/character/page budgets, normalizes `as_of` timestamps, and
  reports partial failures instead of treating missing results as negative
  evidence.
- `BoundedProviderRecord` already carries provider/source IDs, bounded snippets,
  structured metadata, event time, independence group, freshness, redaction,
  and truncation controls. Raw body keys are rejected.
- Plan 0025 already makes exact attendee email the first reverse-lookup lane and
  calls for limit-enforced Gmail and Odollo `mail.message` evidence without
  retaining full bodies.
- The installed Mail Receipts workbench provides an operator-lite, read-only
  public contract for mail search, selected-result context, people resolution,
  relationship neighborhoods/paths, and safe previews. It does not expose
  mailbox mutation or corpus-operation execution, and its claims remain
  evidence rather than provider identity truth.
- The existing relationship projection has stable content-derived IDs, an input
  watermark, shared-address exclusion, symmetric calendar-pair deduplication,
  and explicit zero-effect counters. It does not yet consume message-level
  observations or expose mail-specific evidence.

## Deterministic evidence contract

### Query inputs

Each query is host-built from prepared, attributable values only:

- exact normalized contact email addresses already present in the local contact
  projection;
- exact configured provider IDs or tenant-local contact IDs when available;
- the recording/conversation ID and its `recorded_at` or event time;
- explicit source profile, provider kind, account, tenant, and capability; and
- an `as_of` cutoff and bounded lookback selected by the execution packet.

The query planner must first resolve exact attendee/contact emails through Mail
Receipts people/search surfaces, then follow only returned opaque result and
evidence references for selected hits. Mail Receipts relationship paths and
neighborhoods are candidate evidence, not accepted graph input; their
supporting message references must pass the same local normalization,
independence, temporal, and privacy checks as search results.

Names, organizations, subjects, transcript terms, and model-proposed text must
not enter this deterministic lane as broad search keys. They may be used by a
separately governed lexical/semantic retrieval stage after exact candidates are
prepared, but cannot masquerade as deterministic evidence.

### Normalized mail observation

Every retained observation must use a versioned schema and include:

- stable observation ID and input/query fingerprint;
- source profile, provider, account, tenant, capability, provider record ID,
  source record ID, and optional source URI;
- message/thread identity or a privacy-preserving stable digest when the raw ID
  is not appropriate for projection;
- source event time, retrieval time, `as_of`, freshness, redaction, and
  truncation state;
- exact normalized `from`, `to`, and `cc` participants, with the configured
  account's direction made explicit;
- contact candidate IDs joined only by exact email/provider ID;
- structured signature title, organization, and department observations only
  when the provider or an allowlisted deterministic parser supplies them; and
- an independence-group ID that collapses duplicate provider copies of one
  underlying interaction.

The initial projection stores no full message body, attachment content, quoted
reply history, or unrestricted snippet. Subject text is excluded from the
relationship edge contract; a later bounded context packet may carry a
redacted subject or snippet under the existing character-budget contract.

### Derived shadow hypotheses

The host may deterministically derive only the following proposal classes:

| Observation | Proposed hypothesis | Boundary |
| --- | --- | --- |
| One unique message from contact A to contact B | `SENT_MAIL_TO`, directional | Evidence of one transmission, not evidence it was read or answered |
| Unique messages in both directions across at least two independent threads | `CORRESPONDED_WITH`, symmetric | Interaction history, not a personal/professional relationship classification |
| Two contacts recur on at least two independent threads | `MAIL_THREAD_COPARTICIPANT_WITH`, symmetric | Shared thread participation, not direct interaction or organizational membership |
| Structured title/organization/department for an exact-email contact | Contextual role or affiliation proposal | May be stale, self-authored, delegated, or context-specific; retain conflicts and time |

Shared/role addresses, mailing lists, automated senders, and unresolved contact
classes are excluded from person-to-person relationship proposals and reported
with explicit reason codes. Multiple messages in one thread count as one
independent thread for recurrence thresholds. A duplicate Gmail/Odollo copy of
the same underlying interaction counts once, not once per provider.

The initial default thresholds above are part of the versioned rubric, not an
acceptance policy. They may change only through a reviewed plan revision with
before/after replay evidence.

## Implementation packets

### P0 | Freeze contracts and redacted fixtures

- Add versioned JSON schemas for query receipts, mail observations,
  independence groups, and mail-derived relationship/role hypotheses.
- Freeze reason codes for excluded shared/role/list/automated/unresolved
  addresses, temporal rejection, invalid provider shape, duplication, budget
  exhaustion, and partial source failure.
- Build synthetic and redacted fixtures covering one-way mail, bidirectional
  correspondence, thread coparticipation, duplicate cross-provider copies,
  role conflicts, shared addresses, and historical `as_of` cutoff behavior.
- Record exact numerical provider, record, character, time-window, retry, and
  pilot-cohort bounds in the future live execution packet before any private
  query. No bound may default to unlimited.

Terminal condition: schemas validate, fixtures contain no private content, and
the deterministic expected outputs are frozen before adapter implementation.

### P1 | Add the read-only Mail Receipts adapter capability

- Extend the common evidence adapter contract with explicit mail-metadata
  capabilities instead of bypassing it with direct provider calls.
- Implement a `MailReceiptsEvidenceAdapter` over the installed user-scoped
  operator-lite public contract. Discover the current service profile, use the
  narrowest search/selected-evidence/people surface, preserve opaque IDs and
  cursors, and never infer or enumerate private artifact paths.
- Map Mail Receipts evidence into the common bounded source envelope. Preserve
  original Mail Receipts evidence IDs and namespace attribution without
  surfacing raw provider IDs or recipient lists in ordinary API/UI responses.
- Keep direct GWS Gmail and Odollo `mail.message` readers out of the first
  implementation unless the P0 capability matrix proves a required structured
  field is absent from Mail Receipts and a revised execution packet explicitly
  authorizes that supplemental read.
- Emit immutable query/exchange receipts with request hash, selected profile,
  counts, truncation, warnings, failures, and result hashes.
- Fail closed on namespace/tenant/account mismatch, unsupported capability, missing
  `as_of`, malformed timestamps, raw-body fields, or unresolved query scope.

Terminal condition: mocked adapter contract tests prove read-only, bounded,
tenant-isolated, deterministic normalization with visible partial failures.

### P2 | Normalize, deduplicate, and group evidence

- Add pure functions for address normalization, configured-account direction,
  message/thread identity, structured-signature observations, and source event
  temporal classification.
- Create a deterministic cross-source independence key from provider-stable
  interaction identifiers where available and a documented conservative
  fallback otherwise.
- Preserve contradictory titles, organizations, and dates as separate source
  observations; do not select a winning value during ingestion.
- Keep the projection rebuildable from retained normalized source observations
  and receipts.

Terminal condition: reordered input and identical replay yield the same
observation IDs, independence groups, output ordering, and watermark.

### P3 | Extend shadow relationship and role discovery

- Extend `relationship_role_discovery.py` with mail-derived proposals while
  retaining current provider-role, affiliation, and calendar hypotheses.
- Keep mail observations and semantic relationship labels distinct: the
  deterministic host may say `CORRESPONDED_WITH`, but it may not infer
  colleague, client, vendor, friend, manager, employee, or family.
- Include evidence counts, independent thread counts, first/last observed time,
  directionality, counterparts, source references, conflicts, and
  `why_not_accepted` on every proposal.
- Preserve accepted-effect, provider-write, person-merge, and
  speaker-assignment counters at zero.

Terminal condition: a full redacted fixture replay produces stable, auditable,
review-only hypotheses without changing accepted graph projections.

### P4 | Expose compact review evidence

- Extend `/api/people` with summary counts and bounded evidence detail for the
  new hypothesis kinds.
- Add compact, sortable relationship/role rows to Contacts; keep mail evidence
  behind row expansion, use human-friendly labels and metadata, and avoid new
  dashboard panels or oversized controls.
- Use high-quality SVG icons for actions and status, accessible labels,
  keyboard navigation, resizable columns where tabular, and responsive dense
  layout consistent with the Review Queue direction.
- Show source class, direction, independent-thread count, time range,
  conflicts, and the explicit reason the proposal is still unaccepted.

Terminal condition: desktop and mobile Agent Browser review proves the live
shape is compact, understandable, accessible, and does not expose message body
or provider identifiers unnecessarily.

### P5 | Run one explicitly authorized private shadow pilot

- Stop for a checkpoint that names the exact provider profiles, accounts,
  Mail Receipts namespace, tenants, capabilities, numerical budgets, as-of
  rule, and a cohort of no more than 25 already-queued conversations.
- Preview the query plan and expected local write surface before the first
  owned-corpus read. Run at most one preview and one apply for the authorized
  pilot; allow one retry only for a transient, idempotent read. Do not switch to
  a mailbox-operator profile or direct provider merely because an operator-lite
  capability is absent.
- Persist private receipts under the existing user-scoped state boundary, not
  the repository, and publish only aggregate/redacted validation evidence.
- Replay the same normalized inputs without provider access to prove equality.

Terminal condition: every selected contact/query is accounted for, provider
writes and accepted effects remain zero, replay is deterministic, and the
operator can review the resulting hypotheses in Contacts.

### P6 | Measure usefulness before adding consumers

- Compare the mail-assisted shadow output with reviewed relationship/role
  decisions and speaker-review outcomes without changing the decisions.
- Report coverage, candidate recall, contradictions, false relationship leads,
  shared-address exclusions, duplicate-control rate, temporal leakage, review
  load, provider yield/failure, and appropriate abstention.
- Only propose wiring accepted mail-derived graph facts into speaker deduction
  or conversation contextualization after source-disjoint review evidence shows
  material benefit and no unacceptable high-strength failure.

Terminal condition: an evidence-backed follow-up decision either opens a
separate consumer/acceptance plan or leaves the mail lane shadow-only.

## Dependency and execution order

P0 is the critical-path gate. P1 and the P4 fixture-only UI contract may proceed
against P0 fixtures; P2 follows the frozen schemas; P3 follows P1/P2; P4
integrates after P3. P5 is a separate private/provider checkpoint after P0-P4
pass. P6 follows reviewed P5 output.

Each packet permits at most two implementation attempts and one closed-world
rework cycle. A repeated contract, privacy, tenant, or determinism failure ends
the packet for local reframe rather than widening scope. Each packet ends with
a planning-ledger checkpoint before the next begins.

## Explicit gates

Separate user authority is required before:

- querying owned private Mail Receipts evidence, a mailbox, or a provider
  during this plan's execution;
- selecting or widening provider accounts, tenants, capabilities, time ranges,
  or pilot cohorts beyond an approved preview;
- retaining message subjects/snippets or adding body/attachment retrieval;
- migrating the live knowledge schema or scheduling recurring/background work;
- accepting, correcting, rejecting, or superseding graph hypotheses;
- merging contacts or people, applying speaker identities, or updating voice
  profiles;
- letting any mail-derived hypothesis influence an automatic speaker or context
  outcome; or
- publishing reviewed facts to Graphiti or writing any external provider.

## Non-Goals

- No replacement for Mail Receipts, Gmail, Odollo, or their search/index stores.
- No broad mailbox ingest, semantic body mining, attachment processing, or
  generative relationship inference in the deterministic lane.
- No inference that a domain proves employer, a signature proves current role,
  a recipient read a message, or correspondence proves a named relationship.
- No name-only or fuzzy contact/person linking.
- No automatic acceptance, graph mutation, contact/person merge, speaker
  assignment, transcript rewrite, biometric/profile effect, or provider write.
- No raw or unreviewed private mail in repository fixtures, logs, screenshots,
  prompts, Graphiti, or public artifacts.
- No change to the Review Queue's core recording-review workflow in this plan.

## Acceptance Criteria

- Exact identifiers are queried before any broader retrieval, and every query
  is bound to explicit source profile, provider, account, tenant, capability,
  budget, and `as_of` state.
- The normalized observation schema rejects raw bodies and records complete
  provenance, redaction, truncation, freshness, temporal, and independence
  metadata.
- Identical normalized inputs produce identical observation IDs, independence
  groups, hypothesis IDs, ordering, watermarks, and aggregate counts.
- Duplicate provider copies and repeated messages in one thread do not inflate
  independent interaction counts.
- Historical evaluation excludes observations after the conversation cutoff;
  later evidence is allowed only when explicitly labeled as hindsight for a
  present-day operational workflow.
- Shared/role/list/automated addresses and unresolved contacts cannot create
  person-to-person relationship proposals.
- Conflicting role, organization, and time observations remain visible and
  source-attributable rather than collapsing to a last-write-wins fact.
- Provider failure is partial and visible; it never becomes negative evidence
  or erase successful evidence from another source.
- Contacts renders the result as a compact, sortable, expandable review surface
  with no oversized panels/buttons and SVG action/status icons.
- Every hypothesis states why it is not accepted. Accepted relationship, role,
  person-merge, provider-write, biometric, and speaker-apply counters remain
  zero throughout P0-P5.
- No speaker or conversation-context consumer is enabled until P6 produces a
  separately reviewed, source-disjoint utility and safety decision.

## Validation

- Schema and fixture validation for every accepted/rejected observation shape
  and reason code.
- Unit tests for exact address joins, direction, thread grouping, stable IDs,
  source independence, shared/list/automated exclusions, temporal cutoffs,
  conflict preservation, truncation, and deterministic replay.
- Mocked Mail Receipts operator-lite adapter contract tests covering service
  profile mismatch, namespace/tenant/account mismatch, opaque cursor handling,
  budgets, pagination, transient retry, partial failure, malformed responses,
  unavailable evidence links, and raw-body rejection. Supplemental direct GWS
  or Odollo tests are required only if a later revised packet authorizes those
  readers.
- Integration tests from normalized observations through
  `discover_relationship_roles`, `/api/people`, and the Contacts projection,
  asserting all accepted-effect counters remain zero.
- Frontend component tests, production build, and desktop/mobile Agent Browser
  proof of compact sorting, expansion, keyboard use, SVG controls, dense
  metadata, and privacy-safe evidence display.
- For an explicitly authorized P5 only: private before/after aggregate receipt,
  provider write readback, corpus/query accounting, offline replay equality,
  service/process readback, and no private artifacts in git.
- At each packet closeout: focused tests, relevant wider regression suite,
  active-only planning audit, CodeGraph status/readback, `git diff --check`,
  clean commit, and explicit branch/upstream state.

## Rollback and recovery

Mail observations and hypotheses are replaceable projections over immutable,
bounded source receipts. Disable the capability/feature flag and rebuild the
projection without mail inputs to return to the Plan 0072 A6-R3 state. Provider
systems require no rollback because the adapter is read-only. Any future
accepted-decision ledger is outside this plan and requires its own reversal and
rebuild contract.

## First recommended execution slice

Open P0 only. Freeze the schemas, reason codes, exact deterministic thresholds,
and synthetic/redacted fixtures. Do not call a provider or inspect private mail
until P0-P4 pass and the user approves the exact P5 preview.
