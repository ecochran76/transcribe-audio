# Plan 0053 | Generation-5 Duration Validation And Blind Evaluation

State: OPEN

Lane: P10

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

Optimization: balanced for correctness, evidence quality, and time

## Goal Contract

Determine why the frozen P1 decode differed from container-reported duration,
replace the invalid duration comparison with a technically justified
content-preservation rule, validate that rule on a sealed diagnostic holdout,
and then execute one new blind Generation-5 comparison of context-only speaker
identity against context plus separately visible acoustic evidence. Finish
with one immutable terminal decision.

The meaningful milestone is not a larger tolerance. It is a replayable answer
to two questions:

1. Did P1 preserve the source's decodable audio content under a rule derived
   from audio timing and sample semantics rather than fitted to the failed
   recording?
2. On a new unseen cohort, does frozen acoustic evidence improve the existing
   contextual workflow without adding unsafe confident identity errors?

## Vision Outcomes And Maturity Movement

This plan advances the vision's speaker, provenance, uncertainty, and replay
outcomes. It repairs a preparation gate that currently prevents trustworthy
shadow measurement, then carries the repaired contract through a complete
conversation-level identity comparison.

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Audio-content preservation | Level 1, metadata-duration gate is invalid for some real media | Level 2, validated shadow contract | Development diagnosis, sealed holdout, adversarial truncation fixtures, exact replay |
| Acoustic speaker evidence | Level 1, built without valid unseen outcome evidence | Level 2, valid unseen shadow evidence | Complete frozen acoustic matrix |
| Contextual speaker proposals | Level 2, shadow/reviewed | Level 2, measured on a new cohort | Frozen context-only outcomes |
| Combined voice and context | Level 0, unmeasured | Level 2, bounded paired comparison | Context-only versus augmented results |
| Automatic assignment or profile learning | Level 0 | Level 0 | Explicitly prohibited action vector |

Passing tests, accepting the 0.174-second case, or reaching cohort freeze does
not achieve the milestone. Representative end-to-end paired evidence and one
terminal decision are required.

## Authority And Inputs

- `VISION.md` remains the product north star.
- Plan 0052 is immutable terminal `stop`; this plan does not repair or retry it.
- Plan 0052's replay-stable G2 preview
  `6d6e86094c809c34c45694c311063c06570020348eccd6f65a420535167e3d41`
  and terminal preview
  `2f7f228189072dfb90344c916c2e104d0d4836ea613cd0f081f7e9109e33fc17`
  are diagnostic inputs only.
- The Generation-3 preparation failure is a second known positive control; its
  terminal `STOP` remains unchanged.
- Existing active profiles, calibration-only thresholds, Plan 0025 contextual
  contract, and private gold authorities may be replayed but not mutated.
- Fresh Generation-5 candidates may be selected only from the already
  operator-authorized `Documents/Sound Recordings` corpus. Every selected
  source must be disjoint from training, calibration, every prior revealed
  evaluation, every prior frozen, qualified, or privately labelled evaluation
  candidate regardless of reveal status, all diagnostic development and
  holdout media, and all prior Generation-5 candidates. The exclusion union
  explicitly includes the complete Generation-3 and Plan 0052 cohort and pool
  authorities, not only their failed members.
- Raw audio, raw transcripts, gold, paths, embeddings, provider bodies, and
  identities remain private runtime state. Portable receipts contain only
  hashes, counts, reason codes, aggregate metrics, and negative actions.

## Current State

Plan 0052 closed before prediction or biometric execution. Three cohort cases
completed P1/P2. The fourth source reported `5138.648667` seconds through
ffprobe metadata while the deterministic 16 kHz PCM decode contained
`5138.4746875` seconds. Absolute drift `0.17397950000031415` exceeded the
frozen `0.05`-second rule. Generation 3 previously lost `89.776791` seconds,
showing that a real truncation detector remains necessary even if container
duration is not the correct reference.

The present implementation compares output WAV frame count divided by 16 kHz
to a single ffprobe stream-or-format duration. That metadata may represent a
container timeline rather than the exact count of decodable audio samples.
The cause has not yet been proven. No Generation-5 diagnostic source set,
rule, holdout result, evaluation cohort, prediction, score, or decision exists.
Independent plan reviewer `/root/g5_plan_review` initially returned `STOP` on
seven scientific, ordering, exclusion, worker-isolation, and completion
defects. Version 1 was repaired before execution, and the reviewer then issued
`PASS` for plan design only. No execution packet is implied complete.

## Hard Bounds And Checkpoint Contract

- Persist a checkpoint after every packet and at least every 60 minutes during
  long-running execution.
- At most three concurrent agents, delegation depth one, and one reviewer per
  named independent gate. The primary agent owns joins and canonical files.
- Freeze the diagnostic membership before collecting new measurements.
- Diagnostic development contains exactly the two known duration failures plus
  the three lowest-hash Plan 0051-qualified sources as healthy controls.
- Diagnostic holdout contains exactly the other seven Plan 0051-qualified
  sources. Development and holdout are disjoint, and no new-rule measurement
  may be collected from the seven holdout sources before J1 acceptance.
- Define the replacement rule from standards, tool semantics, and development
  evidence only. It may not use the maximum observed drift, a fitted
  percentile, the Generation-5 cohort, or a constant chosen to admit a named
  recording.
- Independent J1 must accept the frozen diagnosis and rule before any holdout
  measurement is revealed. There is no same-holdout rule rework: failure
  records terminal `stop` or requires a future fresh plan and fresh holdout.
- The rule must reject deterministic truncation and undecodable-tail fixtures,
  beginning at the smallest unacceptable loss immediately beyond the derived
  permissible bound, plus larger packet-loss, corrupt-tail, timestamp-gap, and
  wrong-stream variants.
- Diagnostic sources can never enter Generation-5 evaluation or profile
  learning.
- Enumerate at most 20 fresh evaluation candidates once, oldest-forward by
  recording start with source SHA-256 tie-breaking after exact exclusions.
  Before transcript or gold access, freeze label eligibility, technical-
  failure treatment, population scan order, and this cohort rule: choose the
  lexicographically first seven-candidate combination in candidate order that
  passes every population gate; if none passes, record `stop`. No hand
  selection or candidate substitution is permitted after qualification.
- Freeze one evaluation cohort, reveal gold once, run each acoustic unit once,
  and create exactly two prediction families per case.
- At most two provider/model attempts per authorized packet. Zero post-reveal
  prompt, threshold, candidate, rule, cohort, or prediction changes.
- Zero automatic assignment, enrollment, profile/reference mutation, default
  integration, or historical reprocessing.
- A hard gate failure records `stop`; it cannot redefine completion around the
  last successful implementation or test.

## Technical Decision Criteria

The diagnosis must distinguish at least these hypotheses:

- container or stream timeline duration differs from decodable sample extent;
- non-zero start time, edit list, codec priming, discard padding, or packet
  timestamp discontinuity explains the difference;
- resampling introduces only mathematically bounded rounding/filter delay;
- corrupt, missing, or undecodable packets cause real content loss;
- the decoder or validation code selected the wrong stream or ended early.

The replacement contract must bind exact tool versions, stream selection,
timestamp policy, resampling arithmetic, rounding rule, allowed sample error,
and failure reason taxonomy. Its oracle must be non-circular: at least one
source-side packet/timestamp accounting path and one decode-to-null input-frame
sample count are computed without reading the produced WAV, then reconciled
against codec priming, discard padding, edit lists, and discontinuities.
Output WAV frame-count arithmetic is checked independently. Where applicable,
an aligned canonical-PCM fingerprint or bounded content fingerprint must show
that the right content, not merely equal duration, survived. Reference-path
disagreement, ambiguous source intent, decode corruption, or unexplained
packet loss fails closed rather than becoming a larger tolerance. Passing
requires content-equivalent sample extent within the derived bound, no decode
errors, correct stream/channel selection, packet/content integrity, and
adversarial truncation rejection. Container-duration agreement may remain a
reported diagnostic but cannot by itself decide preservation.

The seven Plan 0051 sources are explicitly a new-rule-blinded positive
holdout, not a population-naive diagnostic sample: they previously passed the
old decode check. G0 also freezes two disjoint negative-variant families. G1
may instantiate only the development family. After J1, G2 instantiates the
held-out family from different seeds and source segments under frozen
container/codec families and severities. The held-out grid begins at the
derived permissible bound plus one output sample and includes larger bounded
tail loss, packet removal, corrupt tail, timestamp discontinuity, and wrong
stream. Variant hashes are derived and excluded from all later evaluation.

## Gold Custody And Paired Worker Isolation

- A private host-side gold custodian may construct and commit private gold in
  G3. This is not a reveal to prediction or acoustic workers.
- Gold paths and contents are excluded from prediction/model worker tool
  surfaces and packets. Workers receive only content-addressed redacted inputs.
- Context-only and augmented proposals run in separate new stateless sessions;
  no session, transcript, hidden state, or tool result is shared between them.
- Freeze provider, model/revision, inference settings, system instructions,
  prompt/rubric, candidate ordering, evidence ordering, temporal cutoff, tool
  allowlist, and output schema before either family. The augmented packet's
  separately marked acoustic factor is the only paired difference.
- Record session/run handles, timestamps, input/output hashes, and tool-access
  receipts. Both families must finish before the scoring custodian receives
  one reveal authorization in G6.
- Reveal means first access by the isolated scoring custodian to the committed
  gold body. Prediction and acoustic workers never receive gold, including
  after scoring.

## Population, Blindness, And Evaluation Gates

Before evaluation freeze:

- the new rule passes every sealed positive holdout case and rejects every
  disjoint held-out negative variant at its predeclared reason boundary;
- J2 independently reproduces the rule, holdout classification, and replay;
- at least seven fresh independent recordings qualify;
- both enrolled people appear in at least two independent conversations;
- at least five total gold people and four same-person session pairs exist;
- every eligible speaker label has complete private gold;
- all source, conversation, recording, derivative, diagnostic, training,
  calibration, and prior-evaluation overlap dimensions are zero.

Before reveal or outcome scoring:

- P1/P2 passes and replays for every evaluation member under the new rule;
- conservative pre-reveal denominator proofs show all nine acoustic units can
  satisfy 20 genuine, 100 known-impostor, and 20 open-set minima;
- condition assignments and the gold-blind full Cartesian score inventory
  replay with no missing data;
- context-only and augmented predictions are frozen in isolated stateless
  sessions under one provider/model/inference/prompt/rubric contract;
- prediction workers have no gold access and receive acoustic evidence only in
  the augmented family as a separately cited, conflict-visible factor.

## Frozen Terminal Decision Policy

Apply in order:

1. `stop` for authority drift, invalid diagnosis, holdout failure, blindness or
   privacy breach, incomplete denominators, replay failure, or exhausted bounds.
2. `reject_acoustic_factor` if augmentation adds a high-confidence wrong
   identity or reduces assignment correctness or candidate recall.
3. `advance_to_limited_pilot_plan` only if every gate passes, augmented
   correctness and recall are no worse, there are zero augmented
   high-confidence errors, and voice either fixes one baseline error or safely
   converts two baseline review/abstentions into correct proposals.
4. `keep_shadow_and_refine` for a valid, safe run that does not meet advance or
   reject criteria.

No outcome authorizes production mutation by itself.

## Work Graph

```text
G0 current-authority replay and sealed diagnostic membership
  -> G1 development diagnosis and proposed sample-preservation contract
  -> J1 independent pre-holdout contract review
  -> G2 sealed diagnostic holdout and adversarial validation
  -> J2 independent validation audit
  -> G3 fresh media qualification, private gold, and evaluation design
  -> J3 independent pre-model reconciliation
  -> G4 frozen cohort, gold commitment, policy, and blind preparation
  -> G5 gold-blind acoustic execution and paired prediction freeze
  -> J4 blindness and completeness audit
  -> G6 single gold reveal, outcome labeling, and paired scoring
  -> J5 independent result audit
  -> G7 immutable terminal decision and closeout
```

G0 through G2 are serialized because the rule must precede holdout access.
Fresh-candidate file discovery may run as a disjoint read-only sidecar only
after G2 passes, but membership and gold remain on the critical path.

## Execution Packets

### G0 | Authority Replay And Diagnostic Freeze

Replay Plan 0052, Plan 0051, and the Generation-3 failure authority; derive
the exact five-source development and seven-source holdout sets without
decoding; prove
disjointness; freeze membership, tool identities, measurements to collect,
privacy flags, and negative actions. Terminal: immutable diagnostic authority
or `stop`.

### G1 | Development Diagnosis And Contract

Collect packet, timestamp, stream, container, decoded-sample, resampling,
content-fingerprint, and error evidence for development only. Explain both
known failures, implement a non-circular content-preservation validator, and
test it against the frozen development negative family. Do not instantiate or
read positive-holdout measurements or held-out negative variants. Terminal:
frozen diagnosis and proposed contract or `stop`.

### J1 | Independent Pre-Holdout Review

An independent reviewer checks source derivation, holdout isolation, causal
support, tool semantics, mathematical error bounds, fixture strength, privacy,
and absence of case-fitted constants. One implementation-only correction is
allowed before holdout access; changing the scientific rule requires a new
review. Terminal: signed G2-only acceptance or `stop`.

### G2 | Sealed Holdout Validation

Run the frozen validator once on every positive-holdout source and instantiate
and test the separately seeded held-out negative family. Report every outcome,
including failures and the old-rule-positive limitation. No rule changes,
variant regeneration, or source substitutions. Terminal: immutable passing
validation or `stop`.

### J2 | Independent Validation Audit

Recompute holdout results and confirm that the validation was sealed before
access, every denominator is complete, negative fixtures fail for the expected
reason, and replay is exact. Terminal: G3 authorization or `stop`.

### G3 | Fresh Evaluation Authority And Gold Feasibility

Freeze the exclusion union and deterministic selection algorithm, then
enumerate the bounded oldest-forward fresh candidate pool and qualify it with
the new rule. The private host custodian establishes conversation identity and
gold; the algorithm selects the first passing seven-candidate combination.
Prove every overlap dimension and freeze the contextual/acoustic design and
worker-isolation contract. If population is infeasible, record `stop`; do not
reuse diagnostic, prior frozen/labeled, or revealed media. Terminal: complete
proposal or `stop`.

### J3 | Independent Pre-Model Reconciliation

Verify media authority, population, gold isolation, profile/calibration
lineage, prompt/rubric, exact trials, conditions, metrics, action vector, and
terminal policy. One bounded non-semantic rework cycle is allowed. Terminal:
G4-only acceptance or `stop`.

### G4 | Freeze And Blind Preparation

Freeze cohort and gold commitment, then run and replay P1/P2 for every member
under the accepted content-preservation rule. Produce conditions, windows,
the full candidate Cartesian inventory, and conservative denominator proofs
without reading gold or running acoustic models. Terminal: complete blind
preparation or `stop`.

### G5 | Gold-Blind Acoustic Execution And Paired Prediction Freeze

Freeze context-only predictions first. Execute each of the nine acoustic units
exactly once against the complete candidate Cartesian inventory while gold
remains unread. Render only the calibration-selected factor as separately
cited evidence, then freeze exactly one voice-augmented prediction per case
under the same Plan 0025-compatible prompt/rubric. Acoustic similarity scores
are inference evidence, not correctness labels or outcome metrics. Do not
reveal gold or calculate genuine/impostor correctness. Terminal: immutable
acoustic score matrices and paired prediction packet or `stop`.

### J4 | Blindness And Completeness Audit

Verify isolated stateless session handles, tool-access receipts, provider and
inference identity, packet hashes, and prediction timestamps precede reveal;
inspect candidate unions, temporal cutoffs, acoustic visibility, conflicts,
missingness, and case-for-case completeness. Predictions cannot be
regenerated. Terminal: scoring-custodian-only reveal authorization or `stop`.

### G6 | Reveal, Outcome Labeling, And Scoring

Reveal gold once, label exact trials from the already-frozen Cartesian score
matrices, and calculate all nine acoustic units' outcomes plus paired
contextual metrics including correctness, candidate recall, high-confidence
errors, review, abstention, and conflict. Do not rerun any acoustic model.
Terminal: immutable result packet or `stop`.

### J5 | Independent Result Audit

Recompute aggregate and per-condition metrics from immutable child evidence;
verify all denominators, failure inclusion, and terminal inputs. No model,
prompt, rule, or prediction rerun is permitted. Terminal: signed result
acceptance or `stop`.

### G7 | Terminal Decision And Closeout

Apply the frozen precedence mechanically; update plan, roadmap, runbook,
receipts, tests, and durable memory when appropriate; commit and push; prove
clean upstream parity. Terminal: exactly one of
`advance_to_limited_pilot_plan`, `keep_shadow_and_refine`,
`reject_acoustic_factor`, or `stop`.

## Scope

- Diagnose metadata duration versus decodable content duration.
- Implement and validate a sample-preservation contract without case fitting.
- Qualify a fresh disjoint Generation-5 cohort under that contract.
- Execute the complete blind paired context/voice evaluation and terminal rule.

## Non-Goals

- No rewrite, reopening, or reinterpretation of Plan 0052 or Generation 3.
- No tolerance increase justified only by observed drift.
- No diagnostic source in evaluation, enrollment, calibration, or learning.
- No automatic identity assignment, profile/reference mutation, default
  integration, historical reprocessing, or production promotion.
- No demographic fairness or open-world population claim from one cohort.

## Validation

- Targeted tests cover container/stream duration disagreement, non-zero start,
  timestamp gaps, codec delay/padding, sample-rate conversion rounding,
  deterministic tail truncation, corrupt media, wrong stream, stale authority,
  holdout leakage, overlap, blindness, and terminal precedence.
- Each authority supports preview, immutable apply, `0600` private body,
  aggregate-only portable receipt, and full-body replay.
- Focused tests precede full `pytest`, compilation, planning audit,
  `git diff --check`, clean worktree, pushed commit, and upstream parity.

## Acceptance Criteria

- A source-backed diagnosis explains the 0.174-second discrepancy and
  distinguishes it from the 89.777-second Generation-3 loss.
- A technically justified non-circular content-preservation rule is frozen
  before holdout access, passes the complete sealed positive holdout, and
  rejects the disjoint held-out negative family at the derived boundary.
- Independent J1 and J2 accept the rule and validation evidence.
- If execution reaches G6, a deterministically selected, fresh, disjoint,
  population-valid Generation-5 cohort and complete private gold exist;
  isolated context-only and voice-augmented predictions are frozen before
  reveal; and all nine gold-blind acoustic score matrices, exact-trial labels,
  and paired metrics replay without a post-reveal model run.
- Independent J3 through J5 accept every gate reached by execution.
- Exactly one immutable terminal decision is recorded with every excluded
  mutation remaining false. A hard-gate `stop` may truthfully close this
  bounded plan, but only completion through G6 and J5 achieves the requested
  unseen paired-evaluation milestone and target maturity movement. An earlier
  stop must explicitly record objective steps 5-6 and that maturity movement
  as unachieved.

## Definition Of Done

The bounded plan is administratively complete only when G7 records one
immutable terminal decision and the plan, roadmap, runbook, runtime receipts,
tests, Git state, and durable memory agree. Diagnosis, implementation, holdout
success, cohort freeze, predictions, model scores, or passing tests alone are
not closure. Product-milestone success is stricter: only accepted execution
through G6/J5 proves the requested unseen voice-versus-context comparison; an
earlier terminal `stop` is valid evidence of non-achievement, not success.

## Revision History

- Version 1, 2026-08-03: opened the Generation-5 diagnostic repair and complete
  unseen paired speaker-identity evaluation under fresh authority.
