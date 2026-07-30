# Plan 0034 | GWS capability budget fairness

State: OPEN

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Objective:

> Prevent one high-yield GWS capability from consuming the complete adapter
> record budget, preserve access for later configured capabilities through the
> existing public adapter interface, and prove one final served immutable
> retrieval includes normalized `gws-default` evidence.

## Scope

- Add one public-interface regression test at
  `GwsEvidenceAdapter.retrieve`.
- Implement the smallest adapter-local adaptive per-capability budget that
  preserves the existing global record and character caps.
- Keep the provider reader, request interface, snapshot contract, temporal
  policy, source order, and host-owned retrieval control unchanged.
- Run focused and joined regression suites.
- Restart `transcripts.service` once from the pushed repair and execute one
  final immutable request on the fixed non-frozen target.
- Document both the installed GWS PATH requirement and capability-fairness
  behavior.

## Non-Goals

- No provider reordering, temporal-policy relaxation, hindsight enablement,
  schema change, credential/config change, target substitution, or additional
  provider.
- No model call, clue-generation pass, frozen-cohort prediction, gold
  review/read, or evidence-family scoring.
- No legacy rollback, speaker assignment, contact merge, CRM mutation,
  automatic confirmation, database-authority cutover, or provider write.

## Current State

Plans 0031 through 0033 successively proved GWS authorization, selected a
nonempty deterministic query plan, included four Odollo snapshots, and repaired
the installed service PATH. The remaining defect is now isolated to
`GwsEvidenceAdapter.retrieve`: all configured capabilities share one inspected
record budget, so a high-yield first capability can prevent every later
capability from being queried.

The Plan 0033 request ordered
`calendar, drive, people, gmail, contacts, leads, log_notes`, capped provider
records at twenty, returned `provider_records_truncated`, and rejected twenty
GWS snapshots as outside the historical scope before any GWS evidence control
could be included. The service is healthy with the PATH repair active.

## Design And Test Contract

Module: `GwsEvidenceAdapter`

Interface: existing `retrieve(ProviderRetrievalRequest) ->
ProviderRetrievalResult`

Seam: injected `GwsProviderReader`, with the existing in-memory fake as the
true-external test adapter.

Required observable behavior:

- Under a global record budget and multiple configured capabilities, a
  provider page that exceeds the first capability's fair share is truncated,
  later capabilities are still queried, and the result contains their valid
  normalized snapshots.
- Unused share from an earlier low-yield capability remains available to later
  capabilities.
- The global record and character caps, pagination bounds, failure semantics,
  source scope, and deterministic snapshot identities remain unchanged.

The implementation remains private to the adapter. No new public method,
request field, configuration value, or caller knowledge is permitted.

## Authority And Bounds

Authority order:

1. this plan and its private receipt;
2. the red/green public-interface test and current source;
3. Plan 0033 immutable receipt and installed runtime evidence;
4. focused/joined tests and served immutable readback;
5. roadmap/runbook; Graphiti remains advisory.

Bounds:

- `max_red_green_cycles: 1`;
- `max_source_work_units: 1`;
- `max_review_rework_cycles: 0`;
- `max_service_restarts: 1`;
- `max_default_retrieval_attempts: 1`;
- `max_target_substitutions: 0`;
- `max_model_calls: 0`;
- `max_frozen_cohorts_consumed: 0`;
- provider access remains read-only; no external write is authorized.

## Execution Packet

### P1 | Test-first fairness repair and final GWS proof

Owner: primary agent

Write surface:

- `tests/test_conversation_evidence_gws.py`;
- `conversation_evidence_gws.py`;
- README;
- private Plan 0034 and product receipts;
- this plan, `ROADMAP.md`, and `RUNBOOK.md`.

Steps:

1. Add one test in which an oversized first-capability page cannot starve a
   later capability; run it and record RED.
2. Implement adaptive per-capability shares behind the existing adapter
   interface; run the test and record GREEN.
3. Run the full GWS adapter suite and the joined adapter, retrieval, policy,
   workflow, and API suites plus Python compilation.
4. Document the runtime PATH and budgeting contract, commit, and push.
5. Restart the service once, verify health/PATH/source, and execute one final
   served immutable request.
6. Require at least one included `gws-default` evidence control, revalidate all
   safety/authority invariants, and close with one terminal decision.

Delegation:

- `not_spawned`: the single test/code pair and its dependent live proof share
  one narrow adapter write surface and one serialized critical path.

## Acceptance Criteria

- The new public-interface test fails before implementation for capability
  starvation and passes after the smallest adapter-local repair.
- Existing pagination, record, character, raw-body, failure, explicit-scope,
  temporal-normalization, and deterministic-replay tests remain green.
- The served request retains six query terms, three explicit source scopes,
  and no legacy fallback.
- At least one included evidence control has source profile `gws-default`.
- GWS failures/exclusions remain explicit and Odollo yield cannot substitute
  for the GWS-specific gate.
- The installed PATH drop-in remains active; service health and restart counts
  are stable.
- Frozen predictions remain 10/10 `not_started`, ground truth remains 10/10
  `not_reviewed`, and gold remains absent.
- Sidecars remain authoritative, database authority and automatic
  confirmation remain disabled, and external writes remain zero.

## Terminal Decisions

- `pass`: tests, service, and immutable receipt prove included GWS evidence
  with all safety checks intact.
- `refine`: the bounded repair is safe but the sole final request still has no
  included GWS evidence.
- `stop`: scope, privacy, evidence integrity, frozen-cohort, gold,
  unexpected-write, or service-stability safety is violated.

## Validation

- Exact RED and GREEN pytest commands and outputs.
- Full GWS adapter and joined host-safe suites.
- Python compile and `git diff --check`.
- Active planning audit, focused commit, and push verification.
- Merged service/process PATH, PID, `NRestarts`, API/provider readiness, and
  served-source proof.
- Immutable request/bundle/projection/retrieval hashes and permissions.
- Live knowledge-table count and frozen-cohort hash/state.

## Definition Of Done

Plan 0034 is done when one immutable terminal receipt records `pass`, `refine`,
or `stop`; tests, pushed source, installed runtime, and repo authorities agree;
all bounds and authority states are explicit; and the closeout commit is
pushed.

