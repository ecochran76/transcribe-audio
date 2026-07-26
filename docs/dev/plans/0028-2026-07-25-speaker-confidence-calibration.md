# Plan 0028 | Speaker Confidence Calibration

State: OPEN

Lane: P09

## Scope

Add a host-owned confidence-calibration layer for speaker identity proposals
after strict output validation and factor scoring. Preserve the model's
proposal and factor assessments, but cap the actionable confidence band when
the same proposal is unlisted, structurally unresolved or conflicting,
contains a Strong/Decisive speaker-mixing contradiction, or carries a
material identity uncertainty flag.

Replay the deterministic calibration over the complete twenty-case reviewed
corpus produced by Plan 0026 and Plan 0027. Keep validation yield, proposal
ordering, and top-person correctness unchanged while measuring the change in
High/Very High wrong proposals.

## Non-Goals

- No new model prompt, provenance retrieval, candidate generation, or
  transcript interpretation.
- No deletion or rewriting of Plan 0027's immutable predictions,
  comparisons, or rejected refinement receipt.
- No automatic confirmation of calibrated proposals in this plan.
- No new chronological review batch or use of unreviewed cases as gold.
- No speaker assignment, contact mutation, CRM write, deposition, or memory
  harvest.

## Current State

Plan 0027's bounded reference repair raised validation from 2/10 to 7/10 on
the original regression cohort and from 2/10 to 8/10 on the reviewed holdout
replay. It also exposed unsafe confidence behavior: High/Very High wrong
speaker proposals rose from 0 to 8 on regression and from 3 to 4 on holdout.

Factor-level inspection shows that every newly exposed High/Very High wrong
top proposal has at least one host-visible safety condition: an unlisted,
unresolved, or conflicting status; Strong/Decisive speaker-mixing
contradiction; or an explicit material identity warning such as mixed
diarization, unverified full identity, missing verified identifier, or
first-name-only matching. The current `speaker-identity.v1` score sums
supporting factor strength but does not limit confidence when those conditions
are present.

## Acceptance Criteria

- The original factor-derived numeric score and band remain preserved as
  uncapped evidence-strength metadata.
- Host calibration emits deterministic reason codes and a capped numeric value
  plus plain-English band.
- Unlisted/unresolved/conflicting assignments and Strong/Decisive mixing
  contradictions cannot remain High or Very High.
- Material identity uncertainty flags are normalized into a bounded,
  versioned host rule rather than treated as arbitrary prose.
- Validation yield, proposal ordering, and top-person correctness remain
  unchanged.
- On the complete twenty-case reviewed corpus, High/Very High wrong top
  proposals fall to zero without converting any currently correct proposal
  into an incorrect proposal.
- Safe bulk confirmation continues to require Very High calibrated confidence
  and no review flags.

## Validation

- TDD for each calibration reason, non-triggering advisory flags, preserved
  uncapped score, and safe-bulk behavior.
- Deterministic replay receipt over both Plan 0027 comparison cohorts.
- Focused identity, workflow, campaign, and API tests plus the full suite.
- `git diff --check`, planning audit, clean commit/push verification, and live
  local-service health readback.

## Stop Condition

Close after one deterministic twenty-case replay is accepted or rejected.
Keep automatic confirmation disabled until a future chronological holdout
validates the calibrated Very High band.
