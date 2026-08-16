# Correction-first identity shadow campaign

Plan 0072 A6 has a bounded execution harness, but no private shadow campaign
has been activated. The harness is Level 1 (built and tested in isolation).
The A6 target is Level 2 only after an explicitly authorized run processes the
25 oldest eligible historical conversations and observes a complete seven-day
new-arrival window. This advances pipeline yield, identity quality, review
load, knowledge integrity, and replayability; it does not authorize identity
acceptance, profile activation, provider writes, deletion, deployment, or
background scheduling.

## Lifecycle and authority

1. `preview` accepts redacted candidate descriptors and is read-only. It sorts
   historical work by `conversation_at`, selects at most the oldest 25
   eligible records, and freezes a half-open seven-day window.
2. `activate` requires the reviewed preview SHA-256 and the exact
   `ACTIVATE_PLAN_0072_A6_SHADOW` checkpoint token. It creates a private 0700
   campaign directory with immutable 0600 manifest and activation receipts.
3. `register-arrival` appends a stabilized eligible conversation only when its
   `artifact_stabilized_at` is inside the frozen window. The conversation time
   and stabilization time deliberately remain separate.
4. The existing A4 supervisor performs evidence processing. `record-case`
   accepts only a terminal A4 run (`stage=complete`, `state=complete`) whose
   effects are all zero, binds its content hash and terminal event, enforces at
   most one transient retry, and optionally projects an A5 review queue item.
5. `finalize` requires every frozen or registered case to have an immutable
   terminal receipt and requires `observed_through` to reach the window end.
   The exact `FINALIZE_PLAN_0072_A6_SHADOW` token is required. Unavailable
   review metrics must carry an explicit reason and create a closed
   `shadow_window_complete_pending_review` window scorecard. A later call with
   the required reviewed metrics creates a distinct terminal receipt citing
   that scorecard; the pending receipt is never overwritten.
6. `replay` verifies the activation manifest, arrival registrations, every
   case receipt, and their recorded hashes before returning the terminal
   scorecard.

Activation and finalization tokens are mechanical fail-closed guards, not
authority by themselves. The operator must first grant the Plan 0072 A6
private historical/new-conversation checkpoint described in the plan and
RUNBOOK.

## Portable candidate contract

Each candidate contains only:

- conversation and recording IDs;
- the actual original recording filename, without a filesystem path;
- source artifact and source media SHA-256 values;
- `conversation_at` and `artifact_stabilized_at` UTC timestamps;
- cohort, eligibility, and disposition.

Raw transcripts, provider payloads, and source/stored filesystem paths are
rejected. Private evidence stays in the campaign ledger and the governed local
knowledge store. Nothing from the campaign is written to Graphiti.

## Zero-effect and measurement contract

Every layer repeats these counters and requires all of them to remain zero:

```json
{
  "accepted_identity_effect_count": 0,
  "accepted_profile_effect_count": 0,
  "provider_write_count": 0,
  "raw_deletion_count": 0
}
```

The window and terminal scorecards derive pipeline yield, dispositions,
provider success/failure/retry counts, latency, duplicate suppression,
knowledge integrity, and queue projection load from case receipts. The
operator-supplied evaluation block must report candidate recall, correctness,
calibration, high-strength errors, abstention, review load, and workflow
usability as either measured or explicitly unavailable. A measured
calibration result does not bypass Plan 0072's 30-outcome minimum, and this
shadow harness cannot enable automatic acceptance.

## Redacted readiness exercise

The following command reads only the tracked redacted fixture and emits a
deterministic preview; it writes no runtime or knowledge-store state:

```bash
.venv/bin/python scripts/plan0072_a6_shadow.py preview \
  docs/dev/fixtures/plan-0072-a6/redacted-candidates.json \
  --activated-at 2026-08-16T12:00:00Z
```

The expected fixture result selects 25 historical cases, includes one
in-window arrival, ends at `2026-08-23T12:00:00Z`, and has campaign ID
`identity-shadow-062175592588b3529e27376b`. The pending-review evaluation
shape is in
`docs/dev/fixtures/plan-0072-a6/evaluation-metrics-pending-review.json`.

The remaining Level 2 proof is representative private execution plus
authenticated operator usability evidence. Unit tests and redacted fixtures
prove the bounds and replay mechanics, not real-corpus quality or usability.
