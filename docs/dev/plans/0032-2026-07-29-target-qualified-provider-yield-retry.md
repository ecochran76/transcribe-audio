# Plan 0032 | Target-qualified provider yield retry

State: OPEN

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Objective:

> Execute one served default immutable retrieval attempt on a preflight-proven
> non-frozen, calendar-associated conversation with a nonempty deterministic
> query plan, then record whether restored GWS authorization produces at least
> one included normalized provider snapshot.

This is the single target-selection remediation justified by Plan 0031's
terminal `refine`. It does not reopen Plans 0030 or 0031 and does not reset or
repeat either plan's immutable request.

## Scope

- Use document `158fe299a59444821675`, selected deterministically as the first
  recent non-frozen conversation whose normalized transcript has calendar
  attendees and a nonempty exact-first query plan.
- Revalidate immediately before execution that it remains outside the frozen
  cohort and deterministically produces six query terms from six calendar
  attendees.
- Execute one served
  `speaker-preprocessing/prepare-evaluation` request in default `retrieval`
  mode with an empty but schema-valid Clue Discovery readout.
- Validate the new private request, query-plan, projection, bundle, and
  retrieval receipts and record one terminal `pass`, `refine`, or `stop`.

## Non-Goals

- No App Intelligence/model call, clue-generation pass, frozen-cohort
  prediction, gold review/read, or evidence-family scoring.
- No legacy rollback, speaker assignment, contact merge, CRM mutation,
  external write, automatic confirmation, or database-authority cutover.
- No second default retrieval attempt, target substitution after execution, or
  source-code remediation in this plan.

## Current State

Plan 0031 proved GWS authorization with a metadata-only calendar call, but its
single default attempt used an input with zero query terms and therefore
issued no provider query. A provider-free scan of the twelve most recent
conversations found eleven non-frozen candidates with nonempty deterministic
calendar query plans.

The selected candidate is the first eligible item in that stable newest-first
listing. Its preflight records six calendar attendees, six query terms, 270
utterances, and four anonymous diarization labels. It is not a member of
freeze `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`.

## Authority And Bounds

Authority order:

1. this Plan 0032 and its private receipt;
2. Plan 0031 terminal receipt and target-eligibility evidence;
3. Plan 0030 terminal/J2 receipts;
4. current source, installed config, live service, and private receipt
   readbacks;
5. roadmap/runbook; Graphiti remains advisory.

Bounds:

- `max_default_retrieval_attempts: 1`;
- `max_source_scope_attempts: 1` for each configured source within that
  request;
- `max_target_substitutions_after_execution: 0`;
- `max_source_code_remediations: 0`;
- `max_model_calls: 0`;
- `max_frozen_cohorts_consumed: 0`;
- `max_external_writes: 0`, excluding the product's private local immutable
  retrieval, projection, and shadow artifacts.

## Execution Packet

### P1 | Qualified immutable retry and terminal gate

Owner: primary agent

Write surface:

- private shadow artifacts and receipts under
  `~/.local/state/transcribe-audio/`;
- this plan, `ROADMAP.md`, and `RUNBOOK.md`.

Steps:

1. Revalidate target eligibility, frozen state, GWS authorization evidence,
   pushed source, explicit scopes, and authority modes.
2. Execute the one served default immutable retrieval request.
3. Validate hashes, permissions, explicit source accounting, failure
   semantics, included evidence controls, and authority invariants.
4. Record one terminal decision and push the reconciled repo authorities.

Delegation:

- `not_spawned`: one serialized immutable live attempt has no independent
  pre-execution lane. Neutral review is unnecessary because the gate is the
  deterministic included-snapshot count plus authority checks.

## Acceptance Criteria

- The immutable request contains exactly three configured source scopes and a
  nonempty query plan.
- The default path attempts each configured source through the bounded
  adapters and does not fall back to legacy collection.
- At least one normalized provider snapshot is included in the immutable
  bundle.
- Every failure, warning, inclusion, and exclusion remains reason-coded;
  missing data is not negative identity evidence.
- Receipt hashes and `0600`/`0700` protections validate without exposing raw
  provider bodies.
- Frozen predictions remain 10/10 `not_started`, ground truth remains 10/10
  `not_reviewed`, and gold content remains absent.
- Sidecars remain authoritative; live database authority and automatic
  confirmation remain disabled; external writes remain zero.

## Terminal Decisions

- `pass`: a nonempty query plan executes and at least one provider snapshot is
  included with all authority and safety checks intact.
- `refine`: the immutable attempt is safe but yields no included snapshot or
  cannot establish trustworthy receipt completeness.
- `stop`: scope, privacy, evidence integrity, frozen-cohort, gold, or
  unexpected-write safety is violated.

## Validation

- Deterministic target preflight and frozen-cohort hash/state checks.
- Served API response and private retrieval/shadow database reconciliation.
- Receipt hash and permission checks.
- Live database knowledge-table count and service/Git readbacks.
- Active planning audit, `git diff --check`, focused documentation commit,
  push verification, and served-source verification.

## Definition Of Done

Plan 0032 is done when one immutable terminal receipt records `pass`, `refine`,
or `stop`; this plan, `ROADMAP.md`, and `RUNBOOK.md` agree; all bounds and
authority states are explicit; and the documentation commit is pushed.

