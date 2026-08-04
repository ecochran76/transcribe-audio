# Plan 0055 | Generation-5 Source-Expanded Blind Evaluation

State: OPEN

Lane: P10

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Goal Contract

Recover the still-unmet Generation-5 paired speaker-identification milestone
after Plan 0054's truthful population-infeasible stop. Freeze a new,
prior-disjoint evaluation authority that includes two operator-identified
recordings containing both enrolled people, obtain complete private gold, and
measure context-only identification against context plus separately visible
voice evidence without profile learning or post-outcome substitution.

Success is a complete seven-recording blind paired evaluation and independently
accepted terminal decision. Merely locating media, transcribing it, or proving
population feasibility is not completion.

## Vision Outcomes And Maturity Movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Speaker identity | Level 1 built, with no valid fresh Generation-5 outcome | Level 2 measured shadow evidence | Seven fresh recordings, complete gold, correctness and abstention metrics |
| Acoustic speaker evidence | Level 1 validated components | Level 2 unseen shadow evidence | Complete nine-unit acoustic score matrices |
| Combined voice and context | Level 0 unmeasured | Level 2 bounded paired comparison | Frozen context-only and augmented predictions plus paired metrics |
| Source provenance | Level 2 for top-level recordings only | Level 2 across an explicitly expanded private source route | Hash-bound bastion and archive-source receipts with exclusion proof |
| Automatic assignment/profile learning | Level 0 | Level 0 | Every mutation and integration action remains false |

This advances the north-star speaker and provenance dimensions by measuring
whether accepted voice evidence improves a grounded conversation-level
identity proposal. It does not make automatic assignment operational.

## Current State

Plan 0054 closed at `population_infeasible_stop` after all 29 private labels
were completed and all 330 seven-recording combinations failed only the gate
requiring both enrolled people in two recordings. No Plan 0054 model,
prediction, cohort/gold freeze, profile mutation, integration, or reprocessing
action ran.

The operator has now authorized one exact Zoom recording through the existing
bastion-mounted SyncThing route. Metadata-only inspection found a 110-minute
mono AAC stream and a separate audio-only artifact. The audio-only artifact is
bound by SHA-256
`06ff1b6b21736d3bb47c2d2789f30c5ae0e9c9998788f93d72cc54ce46840b12`;
its parent MP4 is bound by SHA-256
`e770dbd5eadd51185d76653c95e38a3990a2417efa43b2919133a62ba84476f0`.
Neither hash nor the source identity appears in current repo or private runtime
manifests. A separately located, previously unused archived recording is bound
by SHA-256
`cc0cd45469d3de0d9e336dbdd4abba2458bd555916328f06115008aed1ff913b`.
The operator states that both recordings contain the two enrolled speakers.
That statement authorizes stratified candidate inclusion; it is not gold and
must be confirmed by private listening review.

## Source, Freshness, And Selection Contract

- Required source A is the exact audio-only Zoom artifact at the
  operator-authorized bastion SyncThing route and the hashes above.
- Required source B is the exact previously unused archived recording bound by
  the hash above.
- Build one comprehensive exclusion union covering every prior training,
  calibration, enrollment/reference, development, diagnostic, evaluation,
  candidate, cohort, fixture, and hash-equivalent source or derivative through
  Plan 0054 closure.
- Reject either required source on byte drift, prior-source or derivative
  overlap, missing media, more than one audio stream, unsupported codec,
  unsupported channel count, duration below 60 seconds, or unreadable media.
- Required sources are predeclared population strata, not post-outcome
  replacements. Both must remain in the final seven-recording cohort.
- Enumerate additional media only under the already authorized recordings
  archive. Accept regular, non-symlink `.m4a`, `.mp4`, or `.wav` media with one
  audio stream, one or two channels, and duration at least 60 seconds.
- Sort additional eligible sources by normalized archive-relative path and
  source SHA-256. Freeze at most ten additional prior-disjoint candidates
  before any content decode, transcription, diarization, identity review, or
  model execution.
- After complete private identity review, select the lexicographically first
  five-candidate combination which, together with both required sources,
  satisfies the inherited seven-recording population and overlap gates. There
  is no hand substitution after predictions or gold reveal.
- Raw audio, transcripts, paths, identities, gold, embeddings, and provider
  bodies remain in `0700`/`0600` private runtime state. Portable receipts
  contain only hashes, counts, reason codes, aggregate metrics, and negative
  actions.

## Execution Packets

### S0 | Expanded Source Authority

Copy the required audio artifact into private runtime state, verify exact byte
identity, freeze the comprehensive exclusion union, enumerate the bounded
additional pool, and emit one immutable no-decode proposal. Terminal: J0
review or `stop`.

### J0 | Independent Source Review

Recompute hashes, exclusions, exact required membership, candidate ordering,
privacy flags, and all negative actions. Terminal: transcription-only
acceptance or `stop`.

### S1 | Bounded Transcription And Private Review

Transcribe and diarize only the accepted frozen candidates. Materialize
speaker-specific listening clips and transcript clues in a private HTML review
surface. The operator assigns every identity or stable alias and confirms that
both required recordings contain the enrolled speakers. No acoustic identity
prediction may assist gold creation. Terminal: population proposal or `stop`.

### J1 | Independent Population And Gold Review

Recompute complete labels, exact first passing combination, seven distinct
recordings/conversations, minimum five people, minimum four same-person session
pairs, both enrolled people in at least two recordings, zero overlap, and
private-gold custody. Freeze only after acceptance. Terminal: blind worker
authorization or `stop`.

### E2 | Gold-Blind Paired Prediction

Freeze context-only and voice-augmented predictions plus every required
acoustic score matrix before gold reveal. Workers receive no gold, answer
hashes, population-derived identities, or competing worker outputs. Terminal:
scoring-custodian handoff or `stop`.

### E3 | One Reveal And Paired Scoring

Reveal gold once to the scoring custodian. Compute complete denominators,
correctness, recall, abstention/review resolution, high-confidence wrong
identity counts, and paired deltas. Terminal: independent audit or `stop`.

### J2/E4 | Independent Audit And Terminal Decision

Recompute membership, matrices, predictions, reveal count, metrics, and action
vector. Apply in order: `stop` for gate/evidence failure;
`reject_acoustic_factor` for any augmented high-confidence wrong identity or
reduced correctness/recall; `advance_to_limited_pilot_plan` only for no
regression, zero augmented high-confidence errors, and either one corrected
baseline error or two safely resolved baseline review/abstentions; otherwise
`keep_shadow_and_refine`.

## Parallel And Critical-Path Design

- Hash/exclusion reconciliation and provider/tool readiness checks may proceed
  independently before J0.
- Content transcription, private review, population selection, prediction,
  reveal, and terminal audit remain serialized on the critical path.
- No subagent receives raw private identity labels or gold before its assigned
  independent review boundary.

## Scope

- Bind the exact operator-authorized Zoom audio and one exact unused archived
  recording as required candidate strata.
- Freeze a bounded prior-disjoint archive expansion and complete the new blind
  paired evaluation.
- Preserve all prior duration-validation evidence and Plan 0054's terminal
  stop without reusing its revealed candidates.

## Non-Goals

- No reuse of Plan 0054 candidates, gold, or revealed membership.
- No use of renamed, transcoded, or copied forms of prior evidence as fresh
  sources.
- No automatic identity assignment, profile/reference learning, production
  integration, historical reprocessing, or population-wide fairness claim.
- No claim that a provider transcript, successful decode, or feasible cohort
  alone realizes the product milestone.

## Acceptance Criteria

- Independent J0 accepts exact source hashes, complete exclusions, bounded
  deterministic selection, and privacy.
- Both required sources are confirmed to contain the two enrolled speakers and
  remain in a population-valid, overlap-zero seven-recording cohort.
- Complete context-only and augmented predictions and all nine acoustic
  matrices freeze before one gold reveal.
- Paired metrics and exactly one terminal decision replay immutably under J2.
- Every mutation, integration, and reprocessing action remains false.

## Validation

- Focused tests cover source drift, prior/derivative overlap, required-source
  retention, deterministic archive ordering, complete labels, population
  selection, privacy, worker isolation, one reveal, and replay.
- Run focused tests, full `pytest`, compilation, the planning-contract audit,
  `git diff --check`, clean/upstream parity, and each named independent review.
- Manually verify the private HTML review players, transcript clues, copy
  fallback, and exact source-to-card binding.

## Stop Conditions

Stop on source drift, prior or derivative overlap, missing required media,
insufficient usable speech, failure to confirm both enrolled speakers in both
required recordings, insufficient fresh additional media, population
infeasibility, privacy/blindness breach, incomplete denominator, replay
failure, or exhausted attempt bound.

## Definition Of Done

Administrative closure requires an immutable terminal receipt, reconciled
plan/roadmap/runbook, tests, clean pushed worktree, and exact runtime replay.
Product milestone success additionally requires a fresh population-valid
seven-recording cohort, frozen gold-blind paired predictions and complete score
matrices, one reveal, paired metrics, and an independently accepted terminal
decision. An earlier `stop` is truthful closure but does not satisfy the
Generation-5 paired-evaluation goal.
