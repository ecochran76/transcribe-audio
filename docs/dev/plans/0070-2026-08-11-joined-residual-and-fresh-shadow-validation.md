# Plan 0070 | Joined residual and fresh shadow validation

State: OPEN

Active packet: D0 authority freeze

Checkpoint: Plan 0069 closed PASS after validating all six retained context
outputs, preserving all six original recording filenames, and measuring five
correct context candidates with zero wrong candidates under its six-case gate.
A read-only D3 counterfactual over the full 39-slot reviewed development set
now yields five correct pillar-agreement candidates and zero combined wrong
identities. It yields no actual residual acceptance. The only three-speaker
recording in the existing set has one canonical-person gold slot and therefore
cannot exercise the two-known-plus-one residual rule.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064-0069

Critical-Path Owner: primary agent

## Current State

Plan 0065's corrected acoustic policy retains 10 correct candidates and zero
wrong candidates. Plan 0069 supplies six schema-valid context cases. Combining
those with the remaining Plan 0065 context dispositions covers all 12 reviewed
recordings and 39 slots: acoustic has 10 correct/zero wrong candidates,
context has five correct/one wrong proposal, and the join accepts five correct
pillar agreements while safely withholding the context-only wrong proposal.
No reviewed case contains the population needed for a genuine residual
acceptance, so a bounded supplemental development cohort is required before a
fresh evaluation can open.

## Scope

Advance VISION outcomes 3 and 6 by proving the existing independent-pillar and
residual inference contract on reviewed real recordings, then evaluating the
pre-frozen path once on a new chronological source-disjoint cohort. Preserve
original recording filenames in every activation, case, and human-review
artifact. Current maturity is Level 2 for context-only shadow inference and
Level 1 for the unexercised residual path. Target maturity is Level 2 measured
joined/residual shadow behavior plus one source-disjoint blind result. This
plan prepares but does not implement VISION outcomes 7 or 8.

## Non-Goals

- Do not apply speaker assignments, enroll voices, group people, change default
  thresholds, or write accepted conversation knowledge.
- Do not mutate source transcripts, the transcript index, identity/biometric
  state, provider records, Graphiti, or any external system.
- Do not reuse exposed development recordings as fresh evaluation evidence.
- Do not select the fresh cohort using identity gold, model predictions, or
  likely pass status.
- Do not infer human gold, preselect answers, or expose predictions in review.
- Do not weaken the residual rule or infer a person merely by elimination.
- Do not retry or retune after fresh gold is revealed.

## Execution Graph

| Packet | Depends on | Bounded outcome | Writes | Terminal |
| --- | --- | --- | --- | --- |
| D0 | activation | Freeze Plan 0065 D1, Plan 0069, Plan 0064 gold/P1, exposure set, code, and zero-effect vector | Private manifest/receipt | Exact replay or fail closed |
| D1 | D0 | Reconstruct the 12-case corrected joined result and measure all 39 reviewed slots | Private resolution/gate receipt | Pillar agreement passes and residual gap proven, or withhold |
| D2 | D1 | Freeze at most six structurally selected supplemental development recordings, predictions, and filename-bearing direct-audio review | Product modules/tests and private artifacts | Literal gold complete or await review |
| D3 | D2 | Measure supplement and require one correct actual residual acceptance with zero joined/residual wrong identities | Private development terminal | Pass or `residual_population_infeasible` |
| E0 | D3 | Freeze at most 12 next-oldest source-disjoint recordings before prediction or gold | Private cohort manifest/receipt | Exact denominator/exclusion replay |
| E1 | E0 | Run acoustic/context/joined/residual shadow predictions without gold | Private evidence/receipt | Complete predictions or reason-coded failure |
| E2 | E1 | Publish direct-audio review with original filename for every slot and no predictions | Private review assets/export | Complete literal gold or await review |
| E3 | E2 | Reveal and measure once | Private terminal and tracked closeout | Readiness for separate acceptance plan or withhold |

Delegation is `not_spawned`: current system authority prohibits proactive
subagents, and the critical evidence path is tightly coupled.

## Bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_policy_revisions`: 0; the corrected policy is already frozen.
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
- `review_discovery_passes`: 0; the goal-level pass was consumed previously.
- `review_verification_mode`: closed_world.

## Acceptance Criteria

- D0 binds exact Plan 0065 acoustic policy/evidence, Plan 0069 terminal and six
  filename-bearing cases, Plan 0064 P1/gold, all exposure hashes, and current
  clean upstream-even source authority.
- D1 independently reproduces 12 recordings/39 slots, five correct pillar
  agreements, zero combined wrong identities, and zero residual acceptances;
  the context-only wrong proposal remains unaccepted.
- Supplemental selection is structural, permanently joins the development
  exclusion set, and cannot consume the later oldest-forward evaluation set.
- D3 requires at least one correct `pillar_agreement` and one correct
  `two_known_plus_one_independently_supported_residual`, complete lineage,
  zero joined/residual wrong identities, and complete human gold.
- E0 freezes code, policy, profile/person bindings, route readiness, source
  hashes, original filenames, and the complete exclusion set before E1.
- E1 predictions exist before E2 gold; independent pillar evidence and exact
  joined/residual lineage remain inspectable.
- E2 review shows each hash-bound original recording filename, direct audio,
  no model prediction, and no preselected decision.
- E3 measures exactly once. A pass may emit
  `ready_for_separate_local_acceptance_plan`; it cannot apply any result.
- Every source/store/index/identity/knowledge/biometric/provider-write/
  Graphiti/external effect count remains zero.

## Validation

Run focused packet and resolver tests, Plan 0065/0069 regressions, Python
compilation, full pytest, active/goal planning audits, CodeGraph post-edit
readback, `git diff --check`, exact private replay, mode checks, clean commits,
push, and upstream equality. Direct-audio review must be checked for filename,
playback, answerability, prediction blindness, and disabled incomplete export.
Done is a replayable readiness/withhold terminal; test success alone is not
completion.

