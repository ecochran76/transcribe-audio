# Plan 0071 | Corrected joined residual shadow validation

State: OPEN

Active packet: D0 corrected authority freeze

Checkpoint: Plan 0070 terminal content
`35cf5ae92835962df1e6fe23062dd3578118a2ffb0d23e8519e53c22d58555f9`
closed `withhold` after both D0 attempts failed before artifact creation. The
failure was isolated to authority-harness field shape: the Plan 0065 exposure
manifest stores collections, not integer counts. The inherited evidence and
read-only 12-recording/39-slot joined counterfactual did not drift.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064-0070

Critical-Path Owner: primary agent

## Current State

Plan 0065's corrected acoustic evidence has 10 correct and zero wrong
candidates. Plan 0069's six filename-bearing context cases complete a
12-recording/39-slot development reconstruction with five correct and one
wrong context proposal. The join accepts five correct pillar agreements, zero
wrong combined identities, and zero residual identities. The wrong context-only
proposal remains unaccepted. The existing population cannot exercise the
two-known-plus-one independently supported residual rule, so a structurally
selected supplemental development cohort remains required before any fresh
evaluation can open.

## Scope

Advance VISION outcomes 3 and 6 by first freezing and reproducing the reviewed
joined evidence, then proving the residual rule on real reviewed recordings,
and only then running one source-disjoint blind shadow evaluation. Preserve the
original recording filename in every activation, case, and human-review
artifact. Current maturity is Level 2 for context-only shadow inference and
Level 1 for the unexercised residual path. Target maturity is Level 2 measured
joined/residual shadow behavior plus one source-disjoint blind result. Evidence
is immutable private receipts, exact code/artifact hashes, literal blinded
human gold, and one terminal measurement. This plan does not implement VISION
outcomes 7 or 8.

## Non-Goals

- Do not apply speaker assignments, enroll voices, group people, change
  thresholds, or write accepted conversation knowledge.
- Do not mutate source transcripts, the transcript index, identity/biometric
  state, provider records, Graphiti, or any external system.
- Do not reuse exposed development recordings as fresh evaluation evidence.
- Do not select a cohort using identity gold, predictions, or likely pass
  status.
- Do not infer human gold, preselect answers, or expose predictions in review.
- Do not weaken the residual rule or infer a person only by elimination.
- Do not retry or retune after fresh gold is revealed.

## Execution Graph

| Packet | Depends on | Bounded outcome | Writes | Terminal |
| --- | --- | --- | --- | --- |
| D0 | activation, Plan 0070 terminal | Freeze exact inherited artifacts, exposure collections and lengths, code, filenames, and zero effects | Private manifest/receipt | Exact replay or fail closed |
| D1 | D0 | Reconstruct and measure the 12 recordings/39 slots | Private resolution/gate receipt | Five safe agreements and residual gap, or withhold |
| D2 | D1 | Freeze at most six structurally selected supplemental development recordings, predictions, and filename-bearing direct-audio review | Product modules/tests and private artifacts | Literal gold complete or await review |
| D3 | D2 | Require one correct actual residual acceptance and zero joined/residual wrong identities | Private development terminal | Pass or `residual_population_infeasible` |
| E0 | D3 | Freeze at most 12 next-oldest source-disjoint recordings before prediction or gold | Private cohort manifest/receipt | Exact denominator/exclusion replay |
| E1 | E0 | Run acoustic/context/joined/residual shadow predictions without gold | Private evidence/receipt | Complete or reason-coded failure |
| E2 | E1 | Publish direct-audio review with original filenames and no predictions | Private review assets/export | Literal gold complete or await review |
| E3 | E2 | Reveal and measure once | Private terminal and tracked closeout | Separate acceptance readiness or withhold |

Delegation is `not_spawned`: current system authority prohibits proactive
subagents, and the evidence path is tightly coupled.

## Corrected D0 Contract

- Bind the exact Plan 0070 terminal before reading inherited authority.
- Read the exact frozen Plan 0065 D1 `policy.json` content hash rather than
  expecting a source constant.
- Require list-valued Plan 0065 exposure fields with exact lengths:
  `document_ids=44`, `full_recordings=12`, `recording_hashes=23`,
  `probe_hashes=39`, `source_windows=63`, `review_clips=39`, and
  `decision_rows=39`.
- Bind the exposure object's own content hash and all inherited artifact file
  hashes. Do not transform, coerce, or reinterpret collection values.
- Require 12 Plan 0065 D2 cases, six Plan 0069 cases, six nonempty original
  recording filenames, 12 P1 recordings, 39 human-gold decisions, clean
  upstream-even source authority, zero model turns, and zero effects.

## Bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_policy_revisions`: 0.
- `max_supplemental_development_conversations`: 6.
- `max_fresh_evaluation_conversations`: 12.
- `max_primary_model_turns_per_new_case`: 1 per phase.
- `max_fallback_model_turns_per_new_case`: 1 configured fallback per phase.
- `max_reference_repairs_per_new_case`: 1 reference-only repair per phase.
- `max_fresh_evaluation_runs`: 1.
- `max_review_rework_cycles`: 1 closed-world cycle.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after every packet and before model calls, human
  review, gold reveal, or terminal measurement.
- `review_discovery_passes`: 0; the goal-level pass was previously consumed.
- `review_verification_mode`: closed_world.

## Acceptance Criteria

- D0 exactly binds Plan 0065 acoustic policy/evidence, Plan 0069 terminal and
  six filename-bearing cases, Plan 0064 P1/gold, Plan 0070 terminal, all
  exposure collections, current code, and clean upstream-even authority.
- D1 independently reproduces 12 recordings/39 slots, five correct pillar
  agreements, zero combined wrong identities, and zero residual acceptances;
  the context-only wrong proposal remains unaccepted.
- Supplemental selection is structural, joins the development exclusion set,
  and cannot consume the later oldest-forward fresh set.
- D3 requires at least one correct `pillar_agreement`, one correct
  `two_known_plus_one_independently_supported_residual`, complete lineage,
  zero joined/residual wrong identities, and complete literal human gold.
- E0 freezes code, policy, profile/person bindings, route readiness, source
  hashes, original filenames, and the complete exclusion set before E1.
- E1 predictions exist before E2 gold and preserve inspectable independent
  pillar evidence plus exact joined/residual lineage.
- E2 shows each hash-bound original filename, direct audio, no predictions,
  and no preselected decision; incomplete review cannot export.
- E3 measures once. A pass may emit
  `ready_for_separate_local_acceptance_plan`; it cannot apply any result.
- Every source/store/index/identity/knowledge/biometric/provider-write/
  Graphiti/external effect count remains zero.

## Validation

Run focused packet/resolver tests, Plan 0065/0069/0070 regressions, Python
compilation, full pytest, active/goal planning audits, CodeGraph post-edit
readback, `git diff --check`, exact private replay, mode checks, clean commits,
push, and upstream equality. Direct-audio review must be checked for filename,
playback, answerability, prediction blindness, and disabled incomplete export.
Done is a replayable readiness/withhold terminal; test success alone is not
completion.
