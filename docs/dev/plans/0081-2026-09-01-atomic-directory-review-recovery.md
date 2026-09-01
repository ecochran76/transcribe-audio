# Plan 0081 | Atomic Directory Review Recovery

State: OPEN

Lane: P09

Date: 2026-09-01

Related authority: Plans 0078-0080, `VISION.md`, and the operator report that
every attempted review displayed a red warning triangle.

## Scope

Make one directory review an atomic, unambiguous operation; prevent an
accepted canonical person from defaulting to a duplicate creation; and
reconcile the exact live decisions already recorded during the failed UI
session. No review action will be retried.

## Vision outcomes and maturity movement

This plan advances the review/acceptance stage, provenance, organization/role
context, and self-feeding knowledge loop in `VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Review integrity | Level 1/2 boundary: raw effects commit before projection failure, so the UI reports failure after acceptance | Level 2 atomic review receipt | A forced projection failure leaves neither raw events nor projection effects; a valid same-time dependency batch returns success |
| Identity targeting | Level 2 selectors default unresolved rows to creating a person even when one unique accepted person matches the row's names | Level 2 deterministic existing-target suggestion | Ecochran selects the accepted Eric Cochran target and Ken Anderson selects the unique accepted Ken Anderson target without automatic acceptance |
| Operator feedback | Level 1 red triangle does not distinguish committed from rejected | Level 2 decision-aware reconciliation | Client refreshes and recognizes its exact idempotency key before presenting terminal failure |
| Live authority | Level 1/2 projection is stale behind immutable recorded clicks | Level 2 reconciled projection and explicit correction queue | Every recorded click is inventoried once; valid effects project, and mistaken identities are corrected only with operator-backed authority |

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 freeze and inventory | This plan | Preserve exact clicks and stop retries | plan plus read-only live evidence | Every recent decision/idempotency key and effect is enumerated |
| P1 red regression | P0 | Reproduce generated-ID dependency inversion and post-commit failure | tests | Current code fails the same-time batch test |
| P2 atomic ledger | P1 | Replay by chronological insertion order and commit raw events with projections in one transaction | ledger | Valid dependent batch succeeds; forced rebuild failure rolls back all changes |
| P3 safe target and response | P2 | Suggest only a unique accepted person name/alias match and reconcile ambiguous responses by exact idempotency key | frontend/backend/tests | No unique match defaults to create; exact committed decision renders success |
| P4 live recovery | P2-P3 | Rebuild the recorded valid effects and apply only explicitly authorized identity correction events | private ledger | Projection matches immutable history; Ecochran is linked to accepted Eric Cochran; uncertain duplicate targets remain reviewable |
| P5 installed validation | P4 | Test, build, restart, inspect, and visually validate without a new review action | service and private QA artifacts | Full suite, installed readback, desktop/mobile Agent Browser, and session close pass |

The packets share one transaction and review contract, so no subagent split is
appropriate.

## Acceptance criteria

- Ledger events with one timestamp replay in insertion order, not UUID order.
- Raw event insertion and the resulting projection replacement commit or roll
  back together.
- API success means the decision and its current projection are both durable;
  API failure cannot leave a newly committed decision behind.
- A unique case-insensitive accepted-person match across primary names and
  aliases is suggested as the existing target. Ambiguous or absent matches
  continue to default to explicit creation.
- On an ambiguous client failure, the UI reloads once and treats only the same
  hypothesis and idempotency key as committed success.
- All recent operator clicks are reconciled without replaying them.
- The operator assertion `Ecochran = Eric Cochran = ecochran@iastate.edu` is
  represented through immutable corrective identity events. No other person
  merge is inferred from a name alone.
- No provider write, speaker assignment, biometric effect, mailbox mutation,
  corpus refresh, or automatic hypothesis acceptance occurs.

## Current state after opening

P0 found five accepted decisions and one rejection after the previous clean
baseline. Their raw ledger events are durable, but projection rebuild stopped
on a source record sorted before its newly created person. Direct idempotent
replay returns the recorded receipt, confirming that retrying is unsafe. P1 is
next.
