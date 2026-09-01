# Plan 0075 | Mail Hypothesis Contacts Review Bridge

State: CLOSED

Lane: P09

Date: 2026-09-01

## Scope

Project the completed, immutable Plan 0073 P5 mail-relationship hypotheses into
the live compact Contacts review surface through an explicit hash-pinned source
locator. Add a stale-safe human decision path for accept, reject, and defer.
Only an explicit accepted decision may append an accepted relationship to the
identity-learning ledger and become eligible for the shared evidence fabric.

No new Mail Receipts query is required: the source is the already completed
25-conversation, 57-query pilot containing 120 proposed-only hypotheses.

## Vision outcomes and maturity movement

This plan advances the `VISION.md` outcomes for real-person grounding,
relationship/history context, reviewable uncertainty, provenance, temporal
integrity, deterministic replay, and accepted knowledge improving later
conversations.

- Mail relationship discovery: current Level 2 private proposed-only artifact;
  target Level 2 live review projection with exact source provenance.
- People/relationship loop: current Level 1 accepted handoff proven only on
  disposable state; target Level 2 operator-ready decision path on the live
  Contacts surface.
- Conversation-understanding loop: current Level 1 shared accepted read seam;
  target unchanged until the operator explicitly accepts a live hypothesis.
- This plan does not claim automatic acceptance, speaker-identification lift,
  or Level 3 unattended operation.

## Non-goals and authority boundaries

- No Mail Receipts provider call, mailbox write, backfill, corpus mutation, or
  message-body read.
- No automatic graph acceptance, person merge, speaker assignment, biometric
  effect, Graphiti write, or background watcher.
- Browser validation must not decide a real hypothesis. Mutation tests use a
  disposable store and synthetic artifacts.
- The live source locator is explicit and content-pinned; missing, ambiguous,
  incomplete, effectful, or hash-drifted artifacts fail closed.

## Current State

The live dashboard now loads the immutable Plan 0073 aggregate through an
explicit preview/content-hash locator and exposes all 120 hypotheses as 238
contact-facing rows. Every live row remains `unreviewed`. Contacts supports
compact inline evidence and explicit accept/reject/defer actions; only accept
enters the existing identity relationship projection and shared evidence
fabric. Reject/defer history remains outside accepted conversation knowledge.

## Bounded implementation packet

| Field | Control |
| --- | --- |
| Outcome | Review-ready live Contacts projection for the recovered mail hypotheses |
| Owner | Primary agent in `plan-0037-campaign` |
| Write surface | One artifact-projection module, review workflow/API integration, compact Contacts UI, focused tests, plan/roadmap/runbook docs, user-scoped source locator |
| Inputs | Plan 0073 aggregate and per-conversation hypothesis artifacts, Contacts records, identity-learning ledger, shared evidence fabric |
| Attempts | At most two bounded implementation attempts before local reframe |
| Review cycles | One visual remediation cycle after the first live browser inspection |
| Terminal condition | Source-pinned 120-hypothesis live readback, disposable decision proof, compact visual acceptance, full provider-free validation, deployment/readback, commit/push/upstream parity |

The critical path is serialized: freeze this plan, add failing artifact and
decision behaviors, implement the projection/review seam, integrate API and UI,
validate on disposable state, install the source locator, restart the local
dashboard, visually inspect it, then reconcile and publish evidence.

## Behaviors to test

1. A source locator pins the absolute artifact root, preview ID, and aggregate
   content SHA-256; loading revalidates the aggregate and every listed
   hypothesis artifact.
2. Only complete proposed-only aggregates with zero accepted/provider/person/
   speaker/biometric/Graphiti effects load; drift or malformed data fails
   closed without weakening the existing Contacts route.
3. The exact 120 hypotheses attach to matching local contact IDs, appear in
   reverse-time review order, retain evidence counts/time range/basis/conflicts,
   and expose source status without duplicating a symmetric hypothesis.
4. Accept, reject, and defer require an exact hypothesis ID, source hash,
   optimistic version, actor, timestamp, and idempotency key.
5. Repeated identical submissions are idempotent; stale or conflicting
   submissions fail without a partial ledger/projection update.
6. Reject and defer record durable review history but never enter accepted
   relationship projections. Accept appends one reviewed relationship event,
   rebuilds the identity projection, and is retrievable through the shared
   evidence fabric only under its existing temporal/circularity rules.
7. The Contacts UI remains compact: no large cards or pill controls, SVG icons,
   visible sortable headers, resizable columns, inline evidence expansion, and
   small action controls with explicit accepted/rejected/deferred state.

## Validation

- TDD behavior tests for artifact integrity, projection, stale/idempotent
  decisions, accepted-only handoff, and fail-closed loading.
- Existing relationship, identity-ledger, identity-review, API, and evidence-
  fabric regression suites.
- Frontend unit/build checks and Python compilation for touched modules.
- Disposable HTTP mutation smoke using synthetic artifacts only.
- Installed `/api/people` readback showing the exact live mail count and source
  hash, followed by named-session `agent-browser` visual inspection without a
  real decision.
- Full provider-free pytest suite, active-only planning audit, link check when
  available, `git diff --check`, CodeGraph sync/status, committed checkpoint,
  push, and upstream equality.

## Rollback

Remove the user-scoped source locator and restart the dashboard to restore the
pre-Plan-0075 read projection. Reverting the implementation restores the prior
read-only hypothesis surface. Disposable decision fixtures need no rollback;
any real review remains reversible only through an explicit ledger reversal.

## Closeout evidence

- Source validation pins preview
  `plan0073-p5-139eea68bfb7e6929e4e22115458e35e`, aggregate SHA-256
  `031a31ec68eab795a2412d99293b159c5b1a640ca48074c6a3478dd3d21d456d`,
  every manifest artifact hash, proposed-only status, and zero upstream
  effects before projection.
- The installed API reports 120 hypotheses: 3 correspondence, 70 sent-mail,
  and 47 thread-coparticipation leads. They attach as 238 reciprocal
  contact-facing rows across the existing local directory without embedding
  the full source artifact in the response.
- Synthetic-store tests prove exact projection, hash-drift refusal,
  content-sensitive idempotency, optimistic stale rejection, durable
  accept/reject/defer history, accepted-only identity projection, reversal of
  a prior acceptance, HTTP receipts, and evidence-fabric retrieval.
- The focused regression set passes at 93 tests; the complete provider-free
  suite passes at 1,267 tests. Python compilation, `git diff --check`, and the
  Vite production build pass.
- Service-owned Agent Browser QA at 1440x1000 confirms the dense two-pane
  Contacts surface, visible sort/resizer controls, inline mail evidence, and
  three small square SVG decision controls. QA made no decision.
- The final live readback reports 187 local contacts, 120 configured mail
  hypotheses, 238 contact-facing mail rows, and 238 `unreviewed` states.
  `transcripts.service` is active/running with `NRestarts=0`.
- Provider reads/writes, mailbox mutations, person merges, speaker assignments,
  biometric effects, Graphiti writes, and live accepted relationships remained
  zero in this slice.
