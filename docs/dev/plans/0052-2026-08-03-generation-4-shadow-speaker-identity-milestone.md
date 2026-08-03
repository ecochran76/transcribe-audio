# Plan 0052 | Generation-4 Shadow Speaker Identity Milestone

State: CLOSED

Lane: P10

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

Optimization: balanced for correctness, evidence quality, and time

## Goal Contract

Build and execute one frozen Generation-4 evaluation that compares the
existing context-only speaker workflow with the same workflow plus a
separately visible, calibration-selected acoustic evidence factor on unseen
real conversations. Measure both the complete frozen acoustic matrix and the
combined shadow outcome, then record one immutable terminal decision without
enabling automatic assignment, enrollment, or production integration.

Recommended invocation:

```text
/goal Execute docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md from G0 through one immutable terminal decision. Treat Plan Version 1 as execution authority, preserve prediction blindness and the qualified-media authority, obey every bound and human gate, checkpoint every packet, delegate only the named disjoint lanes, and do not redefine completion around implementation, cohort freeze, acoustic scores, or passing tests.
```

The meaningful milestone is a replayable answer to this product question:
does frozen acoustic evidence improve the existing contextual workflow on
unseen real conversations without introducing unsafe confident errors?

## Hard Bounds And Checkpoint Contract

- At most two execution attempts for any provider- or model-bearing packet.
- At most one review-rework cycle per join gate and two hardening checkpoints.
- Persist a checkpoint after every packet and at least every 60 minutes during
  a long-running packet.
- At most three concurrent agents, one subagent per named parallel lane, and
  delegation depth one. The primary agent owns joins and canonical files.
- Use exactly the Plan 0051 qualified-media authority. One supplemental pool
  of at most 12 explicit candidates is permitted only if G1 proves population
  infeasibility and only before cohort freeze.
- Freeze one cohort, reveal gold once, and run the acoustic matrix once.
- Capture exactly two contextual prediction families per case: context-only
  and context plus separately visible acoustic evidence.
- Permit at most one reference-repair attempt per model phase; never repair
  evaluation evidence into training evidence.
- After prediction freeze: zero prompt, rubric, policy, cohort, gold,
  threshold, or candidate substitutions.
- Zero automatic identity assignment, automatic enrollment, profile mutation,
  default integration, or historical reprocessing.
- On exhausted attempts or a hard gate failure, record `stop`; do not silently
  narrow the objective to the last successful packet.

## Vision Outcomes And Maturity Movement

This plan advances the canonical product vision by testing voice as one
inspectable factor inside conversation-level identity inference, while keeping
review, abstention, provenance, and conflict handling intact.

| Capability | Current | Target at this milestone | Evidence |
| --- | --- | --- | --- |
| Acoustic speaker evidence | Level 1, built but no valid unseen evaluation | Level 2, valid shadow evidence | Complete nine-unit metrics and replay |
| Contextual speaker proposals | Level 2, shadow/reviewed | Level 2, measured baseline | Frozen context-only outcomes |
| Combined voice and context | Level 0, unmeasured | Level 2, bounded shadow comparison | Paired baseline/augmented outcomes |
| Automatic speaker identity | Level 0 | Level 0 | Remains prohibited |
| Reviewed profile learning | Level 0 | Level 0 | Deferred to a later governed plan |

Passing this plan is not population proof and does not promote any production
capability beyond shadow evidence.

## Authority And Inputs

- `VISION.md` is the product north star.
- Plan 0051 is the sole initial media authority. Its preview hash is
  `af5bcf2d8e60b811bcddbb875dd1044f69a090346c6118525c5c5dd80bc49974`
  and its 10-recording qualified-set hash is
  `e3c908f80c922365ead50795728feb959d8aa93e542ee2882be79efc456e48be`.
- The six active profiles, nine frozen calibration thresholds, and their
  source lineage remain fixed inputs; evaluation media cannot mutate them.
- Plan 0025 supplies the closed, reusable two-pass contextual speaker-clue
  workflow. This plan may version its input contract, not rewrite its accepted
  historical outputs.
- Generation 3's media-integrity `STOP` remains authoritative and is not a
  retryable result.
- Private gold, raw transcripts, raw audio, embeddings, and private provider
  bodies remain in private governed storage, never broad memory surfaces.

## Current State

Closed with immutable terminal decision `stop` at G3 blind preparation. G0,
G1A, G1B, and G1C completed. The single supplemental pool produced a valid
seven-recording cohort with nine people, 17 same-person session pairs, both
enrolled people in two independent sessions, complete gold, zero overlap, and
all sources within the exact 10-original plus 12-supplemental authority.

The first J1 review found that G1A trusted caller-injected combined authority.
The one permitted rework removed those inputs, pinned and validated both media
manifests, recomputed the disjoint 22-source union, bound every member to its
origin, and added tamper-negative replay tests. Replacement G1A preview
`9648201f4d3e70f65d396bb1fe82fb9aad57603ff077ccd99b7d61532a0889d7`
and private manifest
`fed08c49b26024b041774b9df7b067ae9d156022fd8a7a16dbf0df8b85451c0f`
replay exactly. Independent J1 then signed acceptance.

G2 froze the exact cohort, private-gold commitment, selected factor, complete
nine-unit contract, contextual prompt/rubric, negative actions, metrics, and
terminal policy. Its preview is
`cc3668c01d7f731ddc340c6bf39dd4983a4bb63b26da9314a6a1da14e14198ee`
and private manifest is
`ccd46ba08725ccccaaabc35e41bd1f56db8cae44e4afaa294150951de2e70a41`.

G3 executed one frozen prediction-blind preparation attempt. Three cases
completed all 15 P1/P2 method cells. The fourth P1 source duration was
`5138.648667` seconds, while the frozen decode produced `5138.4746875`
seconds: drift `0.17397950000031415` exceeded the immutable `0.05`-second
tolerance. Cohort substitution and recipe relaxation were forbidden after G2,
so the remaining three cases were not attempted. Terminal preview
`fd1a1bed2c6b300e7bc829277465b7820aee8d3c7a7f7d570683ab1f3823333c`
and private manifest
`b75aefda1bbf2c38c6c33c7054fc39d05f3631777e4409bc248411f32b0f47ea`
replay full-body. Gold was never revealed to prediction workers; no contextual
prediction, biometric model, acoustic score, profile mutation, integration,
or reprocessing action ran. A successor requires fresh authority.

## Scope

- Replay and verify the Plan 0051 authority before deriving membership.
- Establish whether its qualified recordings can support a valid cohort and
  complete private gold; if not, use the one bounded supplemental-pool option.
- Freeze a source-, conversation-, and derivative-disjoint cohort and gold
  commitment before any evaluation-model access.
- Select exactly one acoustic factor for the contextual comparison using only
  existing calibration evidence. Evaluate all nine frozen acoustic units after
  gold reveal, but do not cherry-pick the combined factor from holdout results.
- Version the Plan 0025 clue contract so acoustic evidence is separately
  cited, missing voice is not negative evidence, and conflicts remain visible.
- Freeze prompts, rubrics, candidate-union rules, temporal cutoffs, and
  terminal policy before contextual predictions.
- Capture context-only and context-plus-acoustic predictions before gold
  reveal; then run preparation, conditions, windows, exact trials, the full
  acoustic matrix, paired scoring, and the terminal decision.

## Non-Goals

- No automatic assignment, contact mutation, profile enrollment, provisional
  profile learning, integration default, or historical backfill.
- No tuning against Generation-4 gold, post-reveal prompt changes, hidden
  fusion score, or substitution of easier cases after freeze.
- No claim of demographic fairness, open-world population performance, or
  general biometric suitability from this bounded cohort.
- No raw private evidence in Git, Graphiti, logs, or portable receipts.

## Stable Decisions

- Gold construction is independent from both contextual prediction families.
- Both contextual families are frozen before gold reveal.
- Acoustic evidence is visible as its own cited factor, never blended into an
  opaque score.
- Calibration evidence selects the combined acoustic factor; Generation-4
  holdout evidence evaluates it.
- Candidate generation and candidate-union ranking are separate measurements.
- Context is time-bounded to evidence available at the conversation timestamp.
- Missing or unusable voice evidence is neutral, not evidence against a person.
- Holdout cases cannot become profile-learning evidence within this plan.
- Partial failures, abstentions, review decisions, and conflicts are outcomes,
  not rows to discard.

## Population And Blindness Gates

Before cohort freeze, prove all of the following or use the bounded supplement
and re-run the gate once:

- At least seven independent conversations or recordings.
- Both enrolled people appear in at least two independent conversations each.
- At least five total gold people are represented.
- At least four independent same-person session pairs exist.
- Every eligible speaker label has complete private gold.
- There is zero source, conversation, recording, derivative, profile-training,
  or previously revealed evaluation overlap.
- Every source still passes the complete Plan 0051 media policy.

Before any evaluation model loads, every acoustic unit must have at least 20
genuine, 100 known-impostor, and 20 open-set exact trials; P1/P2 preparation
must pass for all cohort members; every frozen condition must contain at least
two observed values with zero missing assignments; exact-trial children must
replay; both contextual prediction families must be immutable; and gold must
remain unread by prediction workers.

## Frozen Terminal Decision Policy

Apply these outcomes in order:

1. `stop` on authority drift, blindness breach, privacy failure, incomplete
   denominators, exhausted attempts, replay failure, or any safety-invalid run.
2. `reject_acoustic_factor` if the augmented workflow adds any high-confidence
   wrong identity, reduces assignment correctness, or reduces candidate recall
   relative to context-only.
3. `advance_to_limited_pilot_plan` only if every gate passes, augmented
   correctness and recall are no worse, there are zero augmented
   high-confidence wrong identities, and augmentation either fixes at least
   one baseline error or safely converts at least two baseline
   review/abstentions into correct ready-to-confirm proposals.
4. `keep_shadow_and_refine` for a valid run that is safe but does not meet the
   advance or reject rule.

Only the terminal decision is the plan finish line. All four outcomes are
truthful completion states; none authorizes production mutation by itself.

## Work Graph

```text
G0 authority replay and campaign freeze
  -> G1A cohort and private-gold feasibility
  -> G1B acoustic-factor and evidence-contract design
  -> G1C contextual visibility and temporal contract
G1A + G1B + G1C -> J1 independent design reconciliation
  -> G2 cohort, gold commitment, and frozen policy envelope
  -> G3 prediction-blind preparation and context-only predictions
  -> G4 acoustic execution and augmented predictions
G3 + G4 -> J2 blindness and completeness audit
  -> G5 gold reveal, exact trials, acoustic and paired shadow scoring
  -> J3 independent result audit
  -> G6 immutable terminal decision and closeout
```

G1A, G1B, and G1C are the only pre-authorized parallel lanes. G2 through G6
are critical-path packets and remain sequential. An independent reviewer must
own J1, J2, and J3; the primary agent reconciles findings and alone updates
canonical authority.

## Execution Packets

### G0 | Authority Replay And Campaign Freeze

Owner: primary agent. Replay the exact Plan 0051 authority, verify clean pushed
repository identity and runtime permissions, resolve all inherited hashes and
negative-action vectors, and publish the plan-version checkpoint. Terminal:
continue only on exact replay; otherwise `stop`.

Outcome: complete. Clean pushed commit `5117e7e` supplied the implementation
authority. Production preview, immutable apply, permissions, and full
inherited-authority replay passed under the hashes recorded in Current State.
The negative action vector remains false for J1 through G6, reveal, profile
mutation, default integration, and historical reprocessing.

### G1A | Cohort And Gold Feasibility

Owner: bounded evidence worker. Write only private feasibility artifacts and
aggregate portable receipts. Determine conversation identity, enrolled and
open-set coverage, same-person sessions, label completeness, and every overlap
dimension without exposing gold to prediction workers. Terminal: passing
population proposal, one supplemental-pool request, or `stop`.

Outcome: complete. All nine transcript-linked original-pool recordings and
three recordings from the single consumed supplemental pool are
operator-reviewed in immutable private GOLD_SCHEMA packets. The combined
12-recording population passes every gate with 15 people, 47 same-person
session pairs, both enrolled people represented in two independent sessions,
complete gold, zero overlap, and all sources within authority. The proposed
seven-recording cohort independently passes with nine people and 17
same-person session pairs. The proposal and manifest are content-addressed and
replay-idempotent. Only submission to J1 is true; cohort/gold freeze, reveal,
model sends, prediction, mutation, integration, and reprocessing remain false.

### G1B | Acoustic Evidence Contract

Owner: acoustic-contract worker. Write only the versioned evidence schema,
calibration-only factor-selection rule, condition taxonomy, denominator proof,
and exact-trial/replay specification. Terminal: complete contract or `stop`.

Outcome: complete. Persisted Generation-3 calibration scores replayed without
audio, models, Generation-4 gold, or holdout access. The frozen calibration
objective selected one opaque factor contract, all nine factor units remain in
the evidence matrix, and denominator minima, condition taxonomy, and exact
trial replay contracts are content-addressed. Only submission to J1 is true;
reveal, prediction, mutation, integration, and reprocessing remain false.

### G1C | Contextual Visibility Contract

Owner: contextual-contract worker. Write only the Plan 0025-compatible schema,
temporal evidence cutoff, prompt/rubric hashes, candidate-union policy,
conflict representation, and paired prediction output contract. Terminal:
complete contract or `stop`.

Outcome: complete. The contract freezes the existing Plan 0025 two-phase
workflow, exactly two paired prediction families, recording-start temporal
cutoffs, stable context-first candidate union, separate cited acoustic factors,
neutral missing voice evidence, visible conflicts, one prompt/rubric pair, and
host-computed confidence. Only submission to J1 is true; model sends, reveal,
assignments, mutations, integration, and reprocessing remain false.

### J1 | Independent Design Reconciliation

Owner: independent reviewer. Confirm lane outputs are mutually compatible,
gold remains isolated, calibration selects the acoustic factor, and no lane
silently broadens authority. One bounded rework cycle is allowed. Terminal:
signed design acceptance or `stop`.

Outcome: accepted after the one allowed rework cycle. The first review stopped
on caller-injected source authority. The repaired full-body replay derives both
immutable media manifests, exact counts and set hashes, disjoint union, and
per-member origin. The resubmission signed G2-only authorization.

### G2 | Cohort, Gold Commitment, And Policy Envelope

Owner: primary agent. Freeze exact cohort membership, private gold commitment,
population proof, all overlap proofs, selected comparison factor, full
nine-unit matrix, thresholds, conditions, window/trial rules, prompts,
rubrics, candidate union, metrics, negative actions, and terminal policy.
Terminal: immutable pre-model authority or `stop`.

Outcome: complete. The exact seven-recording cohort, private-gold commitment,
population proof, selected acoustic factor, nine-unit matrix contract,
contextual contract, metrics, negative actions, and terminal precedence are
frozen and replay-idempotent. Only G3 was authorized.

### G3 | Blind Preparation And Context Baseline

Owner: primary agent with stateless leaf workers where safe. Run P1/P2 on all
members, validate media drift, produce replayable windows and exact-trial
children without scores, and freeze context-only predictions while gold is
unread. Terminal: complete blinded baseline or `stop`.

Outcome: terminal `stop`. Three cases completed all P1/P2 methods. The fourth
case reproducibly exceeded frozen P1 duration tolerance: `0.1739795` seconds
observed versus `0.05` allowed. No context provider turn was sent. The
remaining cases were not attempted after the hard gate failed.

### G4 | Acoustic Evidence And Augmented Predictions

Owner: primary agent. Execute the selected acoustic factor without gold,
render separately cited acoustic evidence, and freeze augmented predictions
under the unchanged G2 prompt and rubric. Do not score or tune. Terminal:
complete blinded paired packet or `stop`.

Outcome: `not_run_dependency_g3_stop`.

### J2 | Blindness And Completeness Audit

Owner: independent reviewer. Verify prediction timestamps and hashes precede
gold access; inspect coverage, missingness, temporal cutoffs, candidate unions,
and no-change guarantees. One bounded rework cycle is permitted only for
non-semantic packaging defects; predictions cannot be regenerated. Terminal:
reveal authorization or `stop`.

Outcome: `not_run_dependency_g3_stop`; reveal remained false.

### G5 | Reveal, Exact Trials, And Scoring

Owner: primary agent. Reveal gold once, construct and replay the frozen exact
trials, execute all nine acoustic units once, calculate per-condition acoustic
metrics, and score paired contextual outcomes including errors, abstentions,
review, conflict, and candidate recall. Terminal: immutable result packet or
`stop`.

Outcome: `not_run_dependency_g3_stop`.

### J3 | Independent Result Audit

Owner: independent reviewer. Recompute aggregate metrics from immutable child
evidence, verify every denominator and terminal-policy input, and reject any
post-reveal mutation or omitted outcome. No model/prompt rerun is permitted.
Terminal: signed result acceptance or `stop`.

Outcome: `not_run_dependency_g3_stop`.

### G6 | Terminal Decision And Closeout

Owner: primary agent. Apply the frozen precedence mechanically; record exactly
one of `advance_to_limited_pilot_plan`, `keep_shadow_and_refine`,
`reject_acoustic_factor`, or `stop`; update plan, roadmap, runbook, receipts,
and durable memory when appropriate; validate; commit; push; and prove clean
upstream parity. A successor plan requires fresh authority.

Outcome: complete with immutable `stop` under the first terminal-policy rule.
All mutation and integration actions remain false; fresh successor authority
is required for any new evaluation generation.

## Validation

- Adversarial tests cover overlap, gold leakage, temporal leakage, stale
  authority, missing conditions, incomplete trial denominators, profile
  mutation, factor-selection leakage, prediction drift, replay drift, and
  terminal-policy precedence.
- Focused tests precede full repository tests, compilation, and
  `git diff --check`.
- Every private authority supports no-write preview, immutable apply, exact
  readback, permission checks, and full-body replay.
- Portable receipts expose only hashes, counts, reason codes, aggregate
  metrics, actions, and privacy flags.
- Independent J1, J2, and J3 findings are resolved before the next gate.
- Final closeout requires canonical plan/roadmap/runbook agreement, a clean
  worktree, pushed commit, and upstream-even verification.

## Acceptance Criteria

- A valid, disjoint Generation-4 cohort and complete private gold exist, or a
  truthful terminal `stop` explains why the milestone cannot be evaluated.
- Context-only and augmented predictions are immutable before gold reveal and
  are comparable case-for-case under one frozen rubric.
- Acoustic evidence is separately visible and its comparison factor was
  selected without Generation-4 holdout evidence.
- All nine acoustic units have complete frozen trial matrices and reported
  aggregate/per-condition outcomes.
- Paired shadow metrics include correctness, candidate recall,
  high-confidence errors, review, abstention, and conflict handling.
- Every artifact replays from exact authority without leaking private content.
- Exactly one frozen terminal decision is recorded without enabling excluded
  mutations.

## Definition Of Done

This plan is complete only when G6 records one immutable terminal decision and
all canonical authorities agree with it. Implementation, a passing cohort
gate, frozen predictions, successful acoustic execution, passing tests, or a
metric report alone are not completion.

## Revision History

- Version 1, 2026-08-03: opened the campaign-level Generation-4 shadow
  milestone from qualified media through one terminal product decision.
