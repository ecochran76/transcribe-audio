# Plan 0031 | Provider yield retry

State: OPEN

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Objective:

> Use the operator-confirmed restored GWS authorization to execute one fresh,
> bounded read-only provider retry through the default immutable
> selected-conversation retrieval path, prove whether at least one normalized
> provider snapshot is included, and record a terminal readiness decision
> without consuming the frozen cohort.

This is the bounded successor authorized by the Plan 0030 `refine` decision.
It resolves only the named provider-yield residual class. It does not reopen
Plan 0030 or inherit its exhausted attempt counters.

## Scope

- Verify the configured GWS scope with one metadata-only authorization probe.
- Execute the served default `retrieval` path on the same non-frozen
  conversation used for the Plan 0030 runtime smoke.
- Exercise every configured GWS and Odollo source scope through that immutable
  retrieval request.
- Inspect the private request, query-plan, bundle, projection, and retrieval
  receipts without copying raw provider content into Git or aggregate
  receipts.
- Record one terminal `pass`, `refine`, or `stop` decision.

## Non-Goals

- No frozen-cohort prediction, gold review, gold-body read, prompt tuning, or
  evidence-family scoring.
- No speaker assignment, contact merge, CRM mutation, Graphiti provider-data
  write, automatic confirmation, or database-authority cutover.
- No legacy evidence rollback and no automatic fallback.
- No adapter, schema, or retrieval redesign unless the first attempt exposes
  one bounded correctness defect that prevents trustworthy receipt
  interpretation.

## Current State

Plan 0030 closed `refine` after all authorized provider attempts were
exhausted with zero included snapshots. Its implementation, focused tests,
served default route, private shadow replay, restore, rollback, and authority
proofs passed. The frozen ten-case cohort remains unconsumed, every prediction
is `not_started`, and gold content remains unread.

The operator reports that GWS authorization is restored. Current live
configuration resolves one explicit GWS source and two explicit Odollo tenant
sources without warnings. `transcripts.service` is active from the pushed
Plan 0030 source.

## Authority And Bounds

Authority order:

1. this Plan 0031 version and its private receipts;
2. Plan 0030 terminal and J2 receipts;
3. current repo source, tests, installed config, and live readbacks;
4. `ROADMAP.md` and `RUNBOOK.md`;
5. Graphiti as advisory discovery only.

Bounds:

- `max_gws_live_calls: 2`: one authorization probe and one default retrieval
  attempt;
- `max_odollo_live_calls_per_scope: 1`: the default retrieval attempt;
- `max_default_retrieval_attempts: 1`;
- `max_bounded_code_remediations: 1`;
- `max_model_calls: 0`;
- `max_frozen_cohorts_consumed: 0`;
- `max_external_writes: 0`, excluding private local receipts and shadow
  artifacts created by the product's read-only retrieval path.

The authorization probe must emit only shape/status metadata. The default
retrieval attempt is immutable and is not repeated merely to improve yield.

## Execution Packet

### P1 | Readiness, immutable retry, and terminal gate

Owner: primary agent

Write surface:

- private state under
  `~/.local/state/transcribe-audio/conversation-identity-shadow/`;
- private Plan 0031 receipt under
  `~/.local/state/transcribe-audio/plan-0031/`;
- this plan, `ROADMAP.md`, and `RUNBOOK.md`.

Steps:

1. Verify the Plan 0030 terminal receipt, clean pushed source, service
   identity, explicit source scopes, and unchanged authority modes.
2. Perform one metadata-only GWS authorization probe with private payload
   suppressed.
3. POST the same non-frozen conversation to the served
   `speaker-preprocessing/prepare-evaluation` route with
   `evidence_mode=retrieval`.
4. Validate request, query-plan, projection, bundle, and retrieval hashes;
   permissions; source-scope accounting; failure semantics; and included
   snapshot count.
5. Record one terminal decision and reconcile repo authorities.

Delegation:

- `not_spawned`: this packet is a short serialized live-operation critical
  path. Independent exploration would duplicate context, and no code or
  neutral scoring lane exists before the immutable receipt is available.

## Acceptance Criteria

- GWS authorization is proven by a successful bounded metadata-only call.
- The served default path creates a new immutable retrieval receipt for the
  non-frozen conversation with explicit GWS and both Odollo source scopes.
- At least one normalized provider snapshot is included in the immutable
  bundle.
- Zero-yield or failed sources remain explicitly unavailable, empty, or
  partial and are never interpreted as negative identity evidence.
- The frozen cohort remains unchanged and unconsumed; gold remains unread.
- Sidecars remain authoritative, database authority and automatic
  confirmation remain disabled, and no external write occurs.
- Receipt paths, hashes, and permission checks are recorded without raw
  provider bodies or private transcript content.

## Terminal Decisions

- `pass`: GWS authorization succeeds, the served immutable retrieval includes
  at least one provider snapshot, and all safety/authority checks pass.
- `refine`: privacy and authority remain intact, but the bounded retry cannot
  prove included provider yield or exposes one unresolved non-safety defect.
- `stop`: scope, privacy, gold, frozen-cohort, evidence-integrity, or
  unexpected-write safety is violated.

Any later gold preparation or blind comparison requires a separate explicitly
authorized plan. A `pass` here proves only that the Plan 0030 provider-yield
entry gate is now available.

## Validation

- Current Git, service, provider, and provenance-config readbacks.
- Plan 0030 terminal-receipt hash and frozen-cohort state checks.
- Served API response plus private retrieval/projection/bundle receipt
  validation.
- Focused host-safe tests if source changes; otherwise no test rerun is
  substituted for the live yield proof.
- `git diff --check`, planning audit, commit, push, and served-source
  verification.

## Definition Of Done

Plan 0031 is done when one immutable terminal receipt records `pass`, `refine`,
or `stop`; this plan, `ROADMAP.md`, and `RUNBOOK.md` agree; all bounds and
authority states are explicit; and the resulting documentation commit is
pushed.

