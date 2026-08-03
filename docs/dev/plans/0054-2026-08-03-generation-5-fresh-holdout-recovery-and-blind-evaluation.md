# Plan 0054 | Generation-5 Fresh-Holdout Recovery And Blind Evaluation

State: OPEN

Lane: P10

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Recover from Plan 0053's J2 evidence defect without reusing its revealed
holdout, validate the already J1-accepted sample-preservation rule on a fresh
positive holdout and a separately seeded, fully predeclared negative family,
then execute the still-unmet Generation-5 blind comparison of context-only
speaker identification against context plus separately visible voice evidence.

Success remains the complete paired evaluation and immutable terminal decision,
not merely repairing the fixture, passing another holdout, or freezing a cohort.

## Vision Outcomes And Maturity Movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Audio-content preservation | Level 1; rule supported but Plan 0053 validation denominator invalid | Level 2 validated shadow contract | Fresh 7-source holdout, fixed predeclared negatives, J0/J2 review |
| Acoustic speaker evidence | Level 1 without valid unseen outcome evidence | Level 2 unseen shadow evidence | Complete nine-unit score matrices |
| Contextual speaker proposals | Level 2 shadow/reviewed | Level 2 measured on fresh cases | Frozen context-only predictions |
| Combined voice and context | Level 0 unmeasured | Level 2 bounded paired comparison | Frozen augmented predictions and paired metrics |
| Automatic assignment/profile learning | Level 0 | Level 0 | Every mutation action remains false |

## Inherited Evidence And Non-Authority

- Plan 0053 is immutable terminal `STOP`; neither its seven positive holdouts
  nor any of its development or negative fixtures may be reused.
- The inherited Plan 0053 contract is pinned to plan version 2 and file SHA-256
  `4ff5b5673bdefb7b61025691ad89c1f79b008887e07382c83588a6250c297073`.
  R0 must also bind the immutable J2 STOP preview and manifest hashes recorded
  after its terminal module is applied; a mutable path is not authority.
- J1 accepted content-preservation contract
  `2b3c988ffedebb8a0070499cc779795bea8bd44236b1234128e18859a6d8b7e9`.
  It uses AAC access-unit/sample accounting, skip/discard reconciliation,
  decode-to-PCM reference evidence, exact output PCM identity, a one-output-
  sample resampling bound, and rejection at the first complete missing AAC
  interval. Container duration is diagnostic only.
- Plan 0053 G2's 7/7 positive results and ten fixed negatives are historical
  evidence only. They cannot satisfy this plan's fresh denominators.
- The repaired harness predeclares `measurement_error` for corrupt-source-tail
  construction before any fresh holdout access. An observed reason may never
  populate its own expected field.

## Current State

Plan 0053 is closed at J2. The accepted content-preservation rule is unchanged,
the corrupt-tail expected reason is now literal in code, and focused recovery
tests pass. No Plan 0054 recovery authority, fresh holdout membership,
validation result, candidate pool, gold, prediction, model score, metric, or
terminal decision exists. R0 is the only next packet; new audio decode remains
unauthorized until J0 accepts its proposed membership and fault contract.

## Freshness, Selection, And Privacy Contract

- Source scope remains the operator-authorized `Documents/Sound Recordings`
  corpus.
- Build one exclusion union covering training, calibration, every prior
  evaluation pool/cohort/candidate (revealed or not), every Plan 0053
  development/holdout/fixture source, derivatives, and hash-equivalent media.
- R0 enumerates only top-level, regular, non-symlink `.m4a` files under exact
  root `/mnt/c/Users/ecoch/Documents/Sound Recordings`. Each must have exactly
  one top-level JSON sidecar of schema 1 or 2 whose resolved
  `source_media_path` equals the media path and whose `recording_start` is a
  valid offset-aware RFC 3339 timestamp. Missing, duplicate, ambiguous, or
  unparsable sidecars fail eligibility; there is no filesystem-time fallback.
- Pre-J0 probing is metadata-only: exactly one AAC audio stream, one or two
  channels, positive sample rate, and declared duration at least 60 seconds.
  Probe failure rejects the item into the private ledger. It does not decode
  audio or inspect transcript utterance text.
- After the complete exclusion union and same-run byte-hash deduplication,
  sort eligible rows by normalized UTC `recording_start`, source SHA-256, then
  transcript SHA-256. Freeze the exact inventory, rejection ledger, ordered
  membership, and hashes at R0.
- The first eligible row is the recovery-negative construction source. The
  next seven are the positive holdout. J0 signs these exact eight memberships
  and roles before decode; R1 may only verify them. Fewer than eight records
  `stop`.
- The negative source and its seed-derived segment are disjoint from all seven
  positives and both Plan 0053 segments. All eight sources are permanently
  excluded from evaluation, training, calibration, enrollment, and learning.
- After J2 only, enumerate at most 20 further fresh candidates under the same
  ordering and exclusions. Select the lexicographically first seven-candidate
  combination satisfying the population gates; no hand substitution.
- Raw audio, transcripts, paths, identities, gold, embeddings, PCM hashes, and
  provider bodies stay in `0700`/`0600` private runtime state. Portable receipts
  contain hashes, counts, reason codes, aggregate metrics, and negative actions.

## Hard Bounds

- J0 must accept the corrected fixture contract and deterministic fresh-holdout
  selection before any new holdout source is decoded.
- The fresh negative seed is
  `generation5-duration-recovery-holdout-v1`; it uses a source segment disjoint
  from both Plan 0053 seeds and predeclares every expected reason.
- A separately hashed R0 map predeclares every reason. In particular,
  `corrupt_source_tail = measurement_error` on both exception and non-exception
  branches; execution records observed reasons but cannot create expectations.
- Positive holdout and negatives execute once. Failure records terminal `stop`;
  no same-holdout rework, regeneration, substitution, or rule change.
- Candidate pool is at most 20; evaluation cohort is exactly seven.
- Both enrolled people must occur in at least two independent recordings; the
  cohort must contain at least five people and four same-person session pairs.
- Context-only and augmented predictions use separate stateless sessions with
  identical frozen provider/model/revision/settings/prompt/rubric/candidate
  order. Acoustic evidence is the sole paired difference.
- All nine acoustic units run gold-blind across the full candidate Cartesian
  inventory. Gold is revealed once to the scoring custodian only after J4.
- At most two provider/model attempts per authorized packet.
- Zero automatic assignment, enrollment, profile/reference mutation, default
  integration, or historical reprocessing.

## Work Graph And Packets

```text
R0 Plan-0053 terminal replay, corrected-fixture freeze, exclusion union
  -> J0 independent pre-holdout review
  -> R1 deterministic fresh positive-holdout freeze
  -> R2 one-pass positive and negative validation
  -> J2 independent validation audit
  -> E1 fresh evaluation authority and private gold feasibility
  -> J3 independent pre-model reconciliation
  -> E2 cohort/gold commitment and blind P1/P2 preparation
  -> E3 gold-blind acoustic matrices and isolated paired predictions
  -> J4 blindness/completeness audit
  -> E4 one reveal, outcome labeling, and paired scoring
  -> J5 independent result audit
  -> E5 immutable terminal decision and closeout
```

### R0 | Recovery Authority

Replay Plan 0053's terminal evidence, bind the unchanged accepted rule, freeze
the literal corrupt-tail expected reason and all other negative expectations,
build the comprehensive exclusion union, and enumerate only metadata needed to
propose the deterministic fresh holdout. No new audio decode. Terminal: J0
packet or `stop`.

### J0 | Independent Pre-Holdout Review

Verify Plan 0053 non-reuse, exclusion completeness, selection determinism,
predeclared negative reasons, tool/seed/segment binding, privacy, and absence of
rule changes. Terminal: R1/R2-only acceptance or `stop`.

### R1-R2 | Fresh Holdout Freeze And Validation

Freeze the first seven eligible sources before decode. Then run the accepted
validator once on all seven and instantiate the recovery-seeded negative grid
once. Every positive must pass and every negative must reject for its literal
predeclared reason. Terminal: immutable J2 packet or `stop`.

### J2 | Independent Validation Audit

Recompute complete denominators and exact expected reasons; verify no
substitution, regeneration, leakage, or same-holdout rework. Terminal: E1-only
acceptance or `stop`.

### E1-J5 | Fresh Blind Paired Evaluation

Apply Plan 0053's G3-J5 population, exclusion, gold-custody, worker-isolation,
complete-score-matrix, denominator, reveal, metric, and audit contracts without
narrowing them. Diagnostic media remain excluded. Predictions and all acoustic
scores freeze before the one scoring-custodian reveal.

### E5 | Terminal Decision

Apply in order: `stop` for gate/evidence failure; `reject_acoustic_factor` for
any augmented high-confidence wrong identity or reduced correctness/recall;
`advance_to_limited_pilot_plan` only for no regression, zero augmented
high-confidence errors, and either one fixed baseline error or two safely
resolved baseline review/abstentions; otherwise `keep_shadow_and_refine`.

## Scope

- Preserve and replay Plan 0053's terminal result.
- Freeze and validate the non-circular recovery fixture contract on fresh media.
- Select, prepare, and score one fresh Generation-5 blind paired evaluation.
- Produce independent reviews and one immutable terminal decision.

## Non-Goals

- No reinterpretation, repair, retry, or reuse of Plan 0053 holdout evidence.
- No change to the J1-accepted content-preservation scientific rule.
- No diagnostic source in evaluation, enrollment, calibration, or learning.
- No automatic identity assignment, profile/reference mutation, production
  integration, historical reprocessing, or population-wide fairness claim.

## Validation

- Targeted tests prove literal expected-reason mapping on every control-flow
  branch, exact one-packet and one-sample boundaries, selection ordering,
  sidecar ambiguity rejection, complete exclusions, role disjointness,
  one-pass apply, stale authority rejection, privacy, and replay.
- Every packet receives focused tests, full `pytest`, compilation,
  `audit_planning_contract.py`, `git diff --check`, clean/upstream parity, and
  the named independent review before its successor action.
- R2/J2 validate exact 7/7 positive and 11/11 fixed negative denominators.
  E2-E4 validate complete seven-case paired predictions and all nine acoustic
  score matrices before and after the single reveal as applicable.

## Acceptance Criteria

- Fresh J0 and J2 reviewers accept the corrected, non-circular validation.
- Seven entirely fresh positive holdouts pass and every recovery-seeded
  negative rejects for a reason frozen before execution.
- A new seven-recording, population-valid, overlap-zero evaluation cohort and
  complete private gold are frozen under independent J3 acceptance.
- Context-only and voice-augmented predictions plus all nine acoustic matrices
  are frozen gold-blind, then scored after one reveal and independently audited.
- Exactly one terminal decision exists; every mutation/integration action is
  still false.

## Stop Conditions

Stop on authority drift, insufficient fresh media, overlap, any validation
failure, circular expected evidence, population infeasibility, privacy or
blindness breach, missing denominator, replay failure, or exhausted attempt
bound. An early `stop` is truthful administrative closure, not achievement of
the paired-evaluation milestone.

## Definition Of Done

Administrative closure requires an immutable replaying terminal receipt, plan,
roadmap, runbook, tests, clean pushed worktree, and exact runtime agreement.
Product milestone success additionally requires completion through E4 and J5:
a fresh population-valid seven-recording cohort, frozen gold-blind context-only
and voice-augmented predictions, complete nine-unit acoustic matrices, one gold
reveal, paired metrics, and an independently accepted terminal decision.
An earlier `stop` may close the plan truthfully but does not satisfy the user’s
requested Generation-5 evaluation milestone.
