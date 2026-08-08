# Plan 0059 | Speaker identity foundation and shadow orchestration

State: PLANNED

Checkpoint: P0 not activated

Lane: P09

Cross-lane dependency: P10 acoustic evidence from closed Plans 0057 and 0058

## Scope

Build the first selected-conversation, end-to-end shadow workflow that combines
source-bound acoustic evidence, bounded context collection, and canonical-person
candidate retrieval without mutating live identity state. Rehearse the
conversation knowledge schema and legacy projection on a private copy of the
live transcript database, freeze the minimal contact/role/relationship
contracts, join the evidence pillars under one immutable identity case, expose
the joined result in the existing review console, and compare context-only,
acoustic-only, and combined outcomes on one frozen chronological cohort.

The architecture authority is
[Note 0055](../notes/0055-2026-08-08-speaker-identity-pillar-integration-architecture.md).
The canonical-person and relationship decisions remain in
[Note 0052](../notes/0052-2026-08-05-contact-role-relationship-sequencing.md),
and the evergreen storage authority remains
[Conversation knowledge storage and retrieval](../../conversation-knowledge-storage-and-retrieval.md).

This plan is planning authority only. It must transition to `OPEN` in a
separate activation checkpoint before any implementation, provider retrieval,
private database copy, model execution, or runtime artifact creation begins.

## Vision outcomes and maturity movement

| Capability | Current | Target | Required evidence |
| --- | --- | --- | --- |
| Speaker identification | Separate context and acoustic shadow paths | Level 2 joined selected-conversation shadow inference | Exact context-only, acoustic-only, and combined evaluations over one frozen cohort |
| Context and provenance | Mixed Level 1/2 bounded retrieval and participant bundles | Level 2 immutable context bundles with explicit scope, time, budgets, warnings, and replay | Bundle receipts, partial-failure tests, and source-level inclusion/exclusion evidence |
| Canonical-person knowledge | Level 1 source implementation; live schema version 0 and sidecar authority | Level 2 private-copy projection and reconciliation preview | Migration, backup, projection, count reconciliation, export, replay, and rollback receipts |
| Roles and relationships | Affinity summaries and a durable domain decision | Level 1 typed temporal contract plus Level 2 read-only accepted evidence in candidate ranking | Contract tests, evidence lineage, and no circular-current-run support |
| Human identity review | Level 2 acoustic review surface and existing contact review | Level 2 joined evidence review with alternatives, contradictions, and unresolved outcomes | Browser smoke, strict decisions, and immutable shadow decision ledger |
| Automatic assignment and live knowledge writes | Level 0 | Level 0 unchanged | All live assignment, person, contact, role, relationship, profile, provider, and authority mutation counters remain zero |

This advances north-star outcomes 3, 4, 6, 7, and 8: identify speakers;
retrieve relevant relationships and history; preserve provenance,
contradictions, and uncertainty; prepare accepted conversation knowledge for a
durable store; and make reviewed evidence reusable. It does not claim an
operational or self-feeding knowledge loop.

## Measurable outcome

One selected-conversation workflow can freeze an exact identity case, produce
or load both pillar bundles, retrieve canonical-person candidates from a
private shadow store, join the evidence without provenance loss, present the
result for review, and replay every artifact deterministically. The evaluation
must show whether combined evidence improves over each single pillar on the
same cohort without increasing wrong or high-confidence-wrong proposals.

## Non-goals

- No live transcript database migration or conversation-knowledge authority
  cutover.
- No default watcher enqueueing, background identity timer, or unattended
  provider/model execution.
- No automatic or human-applied live speaker assignment.
- No live creation or mutation of people, contacts, aliases, roles,
  relationships, profiles, references, or provider records.
- No raw audio, clips, embeddings, model-private acoustic features, private
  transcripts, credentials, or provider bodies in repository files or the
  contact/person schema.
- No new acoustic enrollment, profile learning, reference supersession, or
  expansion beyond explicitly allowlisted subjects.
- No autonomous role or relationship inference, authoritative relationship
  edge apply, bounded multi-hop ranking promotion, Graphiti write, or provider
  write-back.
- No historical reprocessing or automatic-confirmation threshold selection.
- No rewrite of the frozen Plans 0056 through 0058 evidence or decisions.

## Current state

Current readback on 2026-08-08 shows:

- `transcribe-watch.service` is active and running with zero restarts;
- the live conversation knowledge schema is version 0, authority mode is
  `sidecar`, and the dirty flag is false;
- the user-scoped transcript database contains 466 documents, 2 contacts, and
  3 speaker assignments;
- closed Plan 0057 provides validated non-authoritative acoustic bundles in
  the ordinary identity-review read path;
- closed Plan 0058 provides a reusable strict review surface with lazy audio
  and importer-compatible decisions;
- closed Plan 0029 provides schema migration, projection, evidence,
  retrieval, profiles, bundle integration, and rollback machinery in source;
- `participant_identity.py` and the P09 workbench provide compatibility
  contact candidates and reviewed merge/split decisions; and
- the watcher does not run acoustic analysis, joined identity evaluation, or
  automatic assignment for completed transcripts.

The main gap is not a new speech backend or a second contact store. The gap is
a bounded orchestration and authority layer that makes the existing pillars
interoperate without promoting shadow evidence into live identity state.

## Frozen contracts

P0 must freeze and test these contracts before any real-data execution:

- stable identifiers for conversation, recording, document, recording-local
  speaker, acoustic subject, canonical person, source record, evidence bundle,
  evaluation, and decision;
- immutable `AcousticEvidenceBundle`, `ContextEvidenceBundle`,
  `CanonicalCandidateSnapshot`, and `IdentityCaseEvaluation` schemas;
- one state machine from `pending` through evidence collection, proposal or
  abstention, review, and shadow decision;
- exact source, account, tenant, capability, as-of-time, model, retrieval,
  ranking, and policy versions;
- evidence lineage and independence groups that prevent circular or duplicate
  support;
- reason-coded partial failure, confidence caps, unresolved outcomes, and hard
  stops; and
- an all-false negative action vector covering live assignments, identities,
  contacts, roles, relationships, acoustic profiles, provider records,
  Graphiti, defaults, authority changes, and historical processing.

## Execution graph

| Unit | Depends on | Outcome | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| A0 activation and current-state freeze | User activation, clean owned worktree, current repo/runtime readback | Transition to `OPEN`; freeze versions, counts, selectors, privacy boundary, cohort envelope, and accepted finding ledger | Plan, Roadmap, Runbook, private activation receipt | Authority is exact, or stop before implementation |
| P0 contracts and fixtures | A0 | Add schema contracts, state transitions, negative actions, and redacted deterministic fixtures | Focused root modules and tests | Contract and adversarial tests pass |
| P1 canonical-person private shadow | P0 | Migrate a private live-database copy, project eligible sidecars/legacy state, preview reconciliation, export, replay, and rollback | Private runtime database and receipts; focused store/projection code only if required | Counts and domain meaning reconcile, or stop/refine |
| P2A acoustic adapter | P0, closed Plan 0057 evidence contract | Normalize source-bound acoustic evidence into the frozen pillar contract without model execution first | Focused adapter and tests | Exact bundles pass; forged, stale, or mismatched bundles fail closed |
| P2B context and relationship adapter | P0, P1 | Produce bounded context and canonical candidate bundles from explicit scopes, accepted history, and provider/local evidence | Focused adapter/orchestrator code, private snapshots, tests | Scope-safe replay passes; partial failure remains visible |
| P3 identity-case join | P1, P2A, P2B | Join exact bundles, preserve separate factors, cap confidence, and produce proposal, alternatives, contradiction, or abstention | One focused orchestration module, immutable private evaluations, tests | Deterministic join passes or any binding/circularity hard stop fires |
| P4 selected-conversation review | P3 | Add prepare/status/read actions and joined evidence to the existing review UI; record shadow-only decisions | `transcript_api.py`, focused frontend components, private decision ledger, tests | Browser and API smoke pass without live domain mutation |
| P5 chronological comparison | P4, human review gate | Compare context-only, acoustic-only, and combined paths on one exact cohort | Private evaluation and review receipts | Complete denominator and independent recomputation, or stop/refine |
| P6 terminal audit | P5 | Recompute acceptance, rollback, privacy, mutation, quality, and burden evidence | Plan, Roadmap, Runbook, optional source-backed Graphiti closeout | `advance_to_live_shadow_plan`, `refine`, or `stop` |

Critical path: A0 -> P0 -> P1 -> P2B -> P3 -> P4 -> P5 -> P6.

P2A may run in parallel with P1 and P2B after P0 because it consumes only the
frozen acoustic contract. P2B must wait for the private canonical-person shadow
interface. Both branches join once at P3. No other fan-out is justified.

Delegation decision: `not_spawned` for plan creation. When activated, the
primary orchestrator must record whether independent P2A and P2B ownership is
safe under the then-current runtime policy and worktree state.

## Data and authority boundaries

- Repository files contain reusable code, schemas, redacted fixtures, plans,
  and tests only.
- Raw transcripts, audio, clips, embeddings, provider snapshots, human labels,
  and review receipts remain under an explicit user-scoped private runtime
  root.
- The private shadow database is disposable and never replaces the live
  database in this plan.
- Provider records retain source profile, account, tenant, external reference,
  relationship scope, observed time, and valid time.
- Acoustic subject IDs remain opaque evidence identities. A separately
  reviewed mapping may associate one with a canonical person, but the acoustic
  branch cannot create that mapping.
- Role-only labels remain unresolved. Accepted role and relationship evidence
  may rank a candidate; current-run proposals cannot create support for
  themselves.
- Graphiti remains a reviewed discovery projection and receives no write in
  this plan.

## Cohort and comparison contract

Activation must freeze one bounded chronological cohort before any new model
output or human gold. The cohort must contain enough eligible enrolled and
non-enrolled speakers to measure proposal, abstention, and contradiction
behavior without rewriting the earlier Plan 0057 denominator. Exact size and
strata belong in A0 after current private population readback.

Every eligible speaker must receive all three blinded conditions over the same
frozen inputs:

1. context-only;
2. acoustic-only; and
3. combined acoustic plus context.

Human gold remains unavailable until all three condition outputs are frozen.
The comparison must report candidate recall, top-person correctness, enrolled
recall, proposal precision, wrong and high-confidence-wrong proposals,
appropriate abstention, unresolved rate, duplicate-person forks, provenance
completeness, provider failures, and review burden.

## Acceptance criteria

- The activated plan freezes exact identifier, evidence, decision, lineage,
  failure, and negative-action contracts before real-data execution.
- A private copy of the live database migrates from the frozen source schema to
  the current knowledge schema with backup, integrity check, projection,
  count/domain reconciliation, round-trip export, replay, and rollback.
- Re-running the exact projection and reconciliation preview is idempotent;
  ambiguous people remain separate and reversible merge/split/redirect actions
  remain review-only.
- Raw audio, clips, embeddings, model-private acoustic features, and display
  names never enter canonical-person, contact, role, or relationship fields.
- Each acoustic bundle binds exact source media, execution, identity-state,
  recording, document, and recording-local speaker references.
- Each context bundle binds exact source/account/tenant/capability/as-of scopes,
  budgets, inclusion/exclusion reasons, warnings, and independence groups.
- The join rejects mismatched or stale bundles, preserves separate evidence
  factors, prevents circular support, and produces an explicit abstention when
  policy cannot support a person.
- A non-required provider failure yields a partial reviewable case with visible
  warnings; a required binding failure stops the identity case without failing
  transcript completion.
- The review console shows candidates, alternatives, contradictions, source
  scopes, acoustic evidence, confidence caps, and unresolved outcomes without
  preselecting an identity.
- Shadow decisions are immutable and replayable but do not invoke the live
  speaker-assignment, contact, person, role, relationship, profile, reference,
  provider, Graphiti, default, or authority mutation paths.
- The frozen cohort has complete condition and review denominators, and an
  independent recomputation confirms all reported quality, failure, burden,
  privacy, and mutation counters.
- Focused tests, relevant frontend checks, browser smoke, Python compilation,
  `git diff --check`, the planning-contract audit, and the full repository test
  suite pass before terminal closeout.

## Validation

- Unit tests for every schema, state transition, identifier scope, lineage,
  confidence-cap, partial-failure, and negative-action invariant.
- Migration, integrity, projection, reconciliation, round-trip, replay, backup,
  restore, and rollback tests against disposable databases.
- Adversarial binding tests for forged IDs, hashes, source scopes, tenants,
  times, stale bundles, duplicate evidence, and circular relationship support.
- API and frontend tests for prepare/status/read, no-write review decisions,
  unresolved outcomes, and disabled apply controls.
- Browser smoke over a redacted or private selected-conversation surface with
  explicit evidence and decision inspection.
- Exact three-condition chronological evaluation with blind outputs, complete
  human gold, and independent metric recomputation.
- Current runtime readback proving the watcher and transcript pipeline continue
  to operate when identity enrichment is disabled, partial, failed, or absent.
- Full `.venv/bin/python -m pytest -q --tb=short` before terminal judgment.

## Safeguards and hard stops

- Stop on any live database migration, authority change, assignment, identity,
  contact, role, relationship, profile, reference, provider, Graphiti, default,
  or historical mutation.
- Stop on any raw private payload, credential, human label, audio, clip, or
  embedding entering the repository, shared memory, or wrong tenant scope.
- Stop on any name, email, provider record, role label, diarization label, or
  evaluation-only ID promoted directly to `person_id`.
- Stop on any acoustic subject outside the activated allowlist.
- Stop on mismatched source hashes, conversation/recording/document bindings,
  speaker-reference sets, tenant/account scopes, or as-of times.
- Stop on an incomplete cohort denominator, gold leakage, non-replayable
  artifact, duplicate-person fork, or high-confidence wrong combined proposal.
- Stop rather than weakening prepared-reference validation, confidence caps,
  evidence independence, human review, privacy, or rollback requirements.
- Split the plan if full relationship inference, live schema activation,
  accepted assignment projection, or automatic confirmation becomes necessary
  to continue.

## Local goal bounds

`max_work_unit_attempts: 2`

`max_review_rework_cycles: 1`

`max_hardening_checkpoints: 2`

`checkpoint_interval: 1 completed execution unit`

`authorization_gate: significant_departure_only`

`retry_budget_mode: renewable_execution_window`

`review_discovery_passes: 1`

`review_verification_mode: closed_world`

`review_finding_fields: criterion, evidence, consequence, reproducer, confidence, suggested_disposition`

`review_disposition_values: blocking | nonblocking_backlog | rejected | needs_evidence`

`checkpoint_record_fields: plan_version, state_transition, progress_classification, evidence, subagent_status, authority_classification, review_disposition_summary, next_action_or_stop_reason`

## Activation and authority

Plan creation authorizes only durable planning and documentation. A separate
activation checkpoint must record the exact current worktree, runtime service,
live database schema/count readback, private cohort envelope, provider and
acoustic permissions, expected write surfaces, and human gates before changing
the plan to `OPEN`.

After activation, standing authority covers ordinary repo-local implementation,
private-copy rehearsal, selected-conversation shadow execution, validation,
repair, retest, commit, push, and closeout within this scope. New authority is
required for live database migration, live assignment or knowledge mutation,
new private-data classes, provider writes, Graphiti writes, unattended watcher
integration, new enrollment, automatic confirmation, public publication, or
another significant departure.

## Terminal decision

- `advance_to_live_shadow_plan`: every acceptance criterion has current
  evidence, combined evaluation is no less safe than each single pillar,
  rollback passes, and all forbidden mutation counters remain zero.
- `refine`: the contracts and selected-conversation shadow join are safe and
  replayable, but a bounded quality, provider-yield, review, or cohort criterion
  remains unmet without weakening safeguards.
- `stop`: any privacy, tenant, binding, circularity, mutation, replay,
  duplicate-person, incomplete-denominator, or high-confidence-wrong hard stop
  occurs.

Even `advance_to_live_shadow_plan` authorizes only a separate successor plan.
It does not authorize live schema migration, watcher enablement, assignment or
relationship apply, provider write-back, profile learning, historical
reprocessing, or automatic speaker identity.
