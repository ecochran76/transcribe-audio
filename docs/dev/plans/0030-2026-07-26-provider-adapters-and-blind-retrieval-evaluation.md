# Plan 0030 | Provider adapters and blind retrieval evaluation

State: OPEN

Lane: P09

## Scope

Complete the bounded refinement selected by Plan 0029 C7. Implement concrete
host-owned GWS and Odollo adapters that emit the versioned evidence-snapshot
contract, make the selected-conversation Identity Evaluation caller consume an
explicit scoped retrieval bundle by default, and run the already frozen unseen
chronological cohort through the five-family blind comparison.

The private freeze is
`evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`. It contains ten cases at
chronological ranks 25 through 39. It must not be regenerated, reordered, or
predicted before adapter, shadow-read, and operator-gold gates pass.

## Non-Goals

- No new storage schema or evidence-bundle redesign.
- No raw provider body in SQLite, model packets, Git, or aggregate receipts.
- No model-controlled provider access.
- No automatic speaker confirmation, contact merge, CRM mutation, external
  write, or database-authority cutover.
- No tuning on the frozen cohort after its predictions are captured.
- No extension beyond one adapter/caller/evaluation slice.

## Current State

Plan 0029 is closed with a `refine` decision. Versioned storage, sidecar shadow
projection, immutable observations and profiles, bounded evidence snapshots,
exact-first hybrid retrieval, immutable bundles, and speaker-review bundle
adaptation all pass in source.

The frozen cohort remains unseen with every prediction state `not_started`.
Production source has only the `HostEvidenceAdapter` protocol; the current
selected-conversation API still calls `collect_configured_identity_evidence`.
The live transcript store has no knowledge-schema tables and sidecars remain
authoritative. The last private retrieval preview produced three valid empty
bundles with eleven calendar candidates and no fabricated provider evidence.

## Execution Packets

### R1 | Concrete bounded provider adapters

Outcome:

- Implement GWS and Odollo adapters behind `HostEvidenceAdapter`.
- Translate only bounded, normalized provider results into
  `EvidenceSnapshotRecord`.
- Preserve source profile, account, tenant, capability, timestamps, temporal
  class, freshness, content hash, redaction, truncation, and independence
  group.
- Convert provider failure into allowlisted partial-result records.

Gate:

- Exact scope, time, freshness, capability, raw-body rejection, and
  provider-failure tests pass.
- No live provider write and no raw provider-body persistence.

### R2 | Default caller and private shadow evaluation store

Outcome:

- Build an explicit retrieval policy from validated user-scoped provenance
  Source Context.
- Make selected-conversation Identity Evaluation use
  `prepare_identity_evidence(...)` and the immutable bundle by default.
- Retain the legacy collector only as an observable rollback path with a
  warning and receipt.
- Project the frozen cohort inputs into a private shadow store and prove
  read agreement, backup/restore, rollback, and unchanged sidecar authority.

Gate:

- No silent fallback.
- No automatic assignment or external write.
- A zero-yield provider result remains labeled unavailable/empty rather than
  negative evidence.

### R3 | Gold review and five-family blind comparison

Outcome:

- Complete operator gold review for the frozen cohort without exposing it to
  prediction prompts.
- Capture all predictions before reveal for calendar-only, transcript-only,
  provenance-only, accumulated-history, and combined retrieval.
- Measure candidate recall, top identity correctness, correct-person presence,
  High/Very High correct and wrong proposals, diarization findings, validation
  yield, provider yield, latency, and packet size separately.
- Record one accept, refine, reject, or stop decision.

Gate:

- Stop immediately on invented accepted references, gold leakage, unexpected
  write, duplicate-evidence inflation, or source-scope violation.
- Automatic confirmation and database authority remain disabled regardless of
  result; either requires a separate explicit authority plan.

## Acceptance Criteria

- At least one production adapter is exercised successfully or returns a
  correctly labeled bounded failure for every configured source scope.
- The default Identity Evaluation receipt identifies request and bundle hashes,
  source failures, warnings, included/excluded reasons, freshness, and temporal
  class.
- The frozen cohort remains unchanged and blind until all predictions exist.
- Every family reports exact denominators and nulls for unmeasurable metrics.
- Combined retrieval is never inferred from missing provenance/history input.
- The terminal decision names residual risk and leaves authority states
  explicit.

## Validation

- Focused adapter, retrieval, workflow, API, projection, and campaign tests.
- Full host-safe test inventory and Python compile checks.
- Private shadow projection/reconciliation plus backup/restore/rollback.
- Receipt hashes and `0600`/`0700` permissions.
- `git diff --check`, focused commits, push verification, and served-runtime
  verification before any live evaluation.

## Stop Condition

Stop after R3 records an accept, refine, reject, or stop decision. Do not
consume another chronological cohort or add an authority cutover to this plan.
