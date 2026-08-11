# Plan 0071 | Corrected joined residual shadow validation

State: OPEN

Active packet: D2 literal supplemental human-gold review

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

Corrected D0 passed on its first attempt. Activation content
`27b011dcce9b3df6922ae0d1f91b077249c2ca991e804e0ada0d40c5713ac931`
and receipt content
`ef0c0e8af6b54701668847ec22a407e71226b4aaf4a26e97b5a9ff14c6b79a69`
replay from a private 0700/0600 tree. They bind all seven exposure
collections at their exact lengths and hashes, 12 Plan 0065 cases, six Plan
0069 cases, six original recording filenames, the Plan 0070 terminal, and
clean upstream-even source authority. Model turns and every effect count are
zero.

D1 passed on its second and final attempt and replays exactly. Its first
attempt wrote no artifact because Plan 0069's nominal original-filename field
contains enriched transcript-artifact names, while Plan 0064's bound review
authority contains the actual `.m4a` recording filenames. D1 preserves both
lineages and treats the Plan 0064 names as authoritative. Resolution content
`a235ddcdbfad57915ad05a10c25f22756715798d7ccb995a46cd00e0752f83fa`,
measurement content
`32b7598262e9d0cf8baf9396bf1b7176b9e9747a39e62c02e8d3d140c05f3a9a`,
and receipt content
`13c6f879c0297b9fcdc53841954ec320f57b56d4905bcf3f4b379194300863c7`
bind all 12 original filenames and the exact 39-slot result. Acoustic is
10/0/29, context 5/1/33, combined 5/0/34, and residual-policy 5/0/34 for
correct/wrong/abstained. The one wrong context-only proposal is safely
unaccepted; actual residual acceptances remain zero. Fresh evaluation is still
closed.

D2's structural cohort is frozen before prediction or gold. Manifest content
`ea37ea3879f467ce6604df53da55c184088e3a6a9accc21abf49eeb154b8f6c2`
and receipt content
`d438ec1061b264dff0233ca6956c2d2bf6532b4bfab4fcf969c1e4b8edcfefe2`
bind six next-oldest source-disjoint recordings, exactly three diarized labels
per recording, 18 slots, six actual original `.m4a` filenames, and hash-matched
transcript/media artifacts. Prediction count is zero, human gold was not read,
fresh evaluation is disallowed, and every effect count is zero.

D2 prediction is now terminal after its two bounded runtime attempts. Attempt
one stopped before any model turn because its isolated stored-transcript path
was outside the private store root; immutable receipt content
`94458b21dceabab024f7deed59544d1d0c696bbbddb2b7d94dfa05b6a61ca217`
records zero turns and zero effects. Corrected attempt two replays with receipt
content `8de26c83af3a2dc1da7c04633fad4c698adcccf3972d42d56f7a8aecf86971b6`,
manifest content
`71fc568512b5a0c24319445df3ffe0bdbc89957b94005cd46af6acc6b182ffd2`,
and resolution content
`bf1876e0610f668ea8eaa4f5a0c4f3748540df36523e39b4410eb8428ebfe931`.
Acoustic prediction produced seven review candidates and 11 abstentions.
Context produced 18 unavailable outcomes, so joined/residual prediction also
contains seven review candidates, 11 abstentions, and zero accepted residual
identities. Human gold remained unread and every mutation/effect count is
zero.

The prediction-blind private D2 review is published under authenticated
Previews session `6b97d7a8da32`. Review authority content
`3aac595b1dce3ba6b8e41d2de653fb399b56cd21841e782e505bbf2cf34c91ba`
and receipt content
`996e92c9abe5e4c394b0f7291d32901b1bb68d7c943e07447b4ec9a645d9cabf`
bind 18 direct-audio clips, six original recording filenames, six reviewed
canonical-person options plus `not_listed` and `unresolved`, no visible model
predictions, and zero preselected decisions. Export remains disabled until all
18 literal decisions are complete. D3 and fresh evaluation remain closed.

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
