# Plan 0065 | Speaker identity recovery and fresh source-disjoint validation

State: PLANNED

Checkpoint: Plan 0064 is closed fail-safe at P4. Its 39 human-gold decisions
are now development/hindsight evidence; its acoustic band produced one
high-support wrong identity, context produced zero candidates, and neither the
reviewed-development nor source-disjoint corpus exercised a correct residual
acceptance. Plan 0065 has not been activated and authorizes no provider/model
execution, identity mutation, knowledge write, or external effect.

Lane: P09/P10

Cross-lane dependency: closed Plans 0063 and 0064

Critical-Path Owner: primary agent

## Scope

Recover the bounded speaker-identity shadow path disproved by Plan 0064 and
evaluate the frozen correction exactly once on a new chronological
source-disjoint cohort. Diagnose the acoustic false acceptance, repair the
verified contextual evidence failures, demonstrate the actual constrained
residual rule on reviewed development evidence, and freeze the corrected
policy before any new evaluation prediction is produced.

Plan 0064 recordings, predictions, review artifacts, decisions, and operator
notes are development-only after the P4 reveal. They may explain or test a
correction but can never regain unseen-evaluation status. The fresh evaluation
continues the established oldest-forward corpus after every recording exposed
by Plans 0063, 0064, and Plan 0065 development work.

This plan remains shadow-only. A passing result may produce
`ready_for_separate_local_acceptance_plan`; it cannot apply an identity,
enroll a voice, enrich conversation knowledge, create a provider proposal, or
authorize an external write.

The durable inference contract remains
[Note 0056](../notes/0056-2026-08-09-context-assisted-automatic-speaker-recognition.md).
Private names, notes, audio, provider payloads, person IDs, and biometric
values remain in mode-`0600` user-scoped artifacts.

## Vision outcomes and maturity movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Acoustic identity safety | Level 2 shadow exists, but its proposed high-support band is disproved by one source-disjoint wrong identity | Level 2 corrected shadow with a pre-frozen policy and zero high-support wrong or unverifiable identities on a fresh cohort | Development diagnosis plus one blind chronological evaluation with complete human gold |
| Context/acoustic join | Level 1 implementation replays, but Plan 0064 produced zero source-disjoint context or combined candidates | Level 2 measured shadow with at least one correct, lineage-complete combined acceptance | Validated contextual factors, independent pillars, joined metrics, and explicit conflicts/abstentions |
| Residual speaker assignment | Level 1 implementation exists, but no real reviewed acceptance exercised its rule | Level 2 measured shadow only if both reviewed development and fresh evaluation contain a correct actual residual acceptance | Two-known-plus-one independently supported residual lineage, counterexamples, and human gold |
| Local identity and knowledge acceptance | Level 0 automatic apply; Plan 0064 withheld P5 | Remain Level 0 in this plan | Terminal readiness-or-withhold receipt with every action counter zero |
| External provider write-back | Level 0 | Remain Level 0 | No proposal apply and zero external provider writes |

This advances VISION outcomes 3 and 6 by making speaker inference measurable,
evidence-preserving, and safely abstaining. It prepares outcomes 7 and 8 but
does not claim them because this plan writes no accepted observation or
reusable conversation knowledge.

## Current State

Plan 0064 terminal content
`f178f4187d0e8c877362310563738144854508fb4acba8b3ea227b79e829d5b6`
replays `withhold_p5` with all six action counters at zero. The 39-row gold
contains 11 canonical-person, 24 not-listed, and four unresolved decisions.

Acoustic-only produced 12 candidates: 11 correct and one wrong. Every
candidate had two-model support. The wrong case was a not-listed person and
overlapped the lower edge of the correct score-margin/threshold-surplus band,
so a threshold-only repair is not presumed safe. Probe purity, diarization
contamination, model agreement, calibration, and profile/source overlap must
be diagnosed separately before choosing a correction.

P2 produced four complete contextual cases, four
`identity_evaluation_validation` failures, and four
`provider_routes_unavailable` cases. The validation failures all report
`calendar_association factor must cite prepared evidence`. Across 39 slots,
context produced 13 reason-coded abstentions and 26 unavailable outcomes, with
zero prepared candidate matches. The provider-capacity errors are historical
Plan 0064 evidence; current route readiness must be checked when this plan is
activated rather than assumed.

The resolver already requires two distinct accepted combined identities plus
one remaining context candidate with transcript clues, provenance, no provider
failure, and no strong contradiction. The reviewed development replay and
source-disjoint cohort produced zero acceptances through that exact rule.

## Non-Goals

- Do not tune on, relabel, or reclassify the Plan 0064 cohort as unseen.
- Do not weaken the residual rule, infer a person by elimination, or count
  duplicated provider records as independent support.
- Do not use operator notes as inference evidence or model input.
- Do not select a fresh evaluation recording because its speakers, outcome, or
  likelihood of passing is known.
- Do not overwrite Plan 0064 schemas or immutable artifacts. Version successor
  policy and receipts so the Plan 0064 terminal still replays exactly.
- Do not apply speaker assignments, create enrollments, mutate profiles or
  references, write conversation knowledge, change default thresholds, or run
  historical reprocessing.
- Do not create or apply contact-enrichment proposals and do not mutate Google,
  Odollo, or another external provider.
- Do not store raw transcripts, audio, private identity values, provider
  payloads, or human decisions in tracked files or Graphiti.
- Do not claim Level 3 operational or Level 4 dependable identity from one
  corrected shadow evaluation.

## Authority and activation

- `PLANNED` is non-executing. Activation requires a separate explicit operator
  instruction and a clean, upstream-even repository readback.
- Activation freezes the exact Plan 0064 terminal/gold lineage, current active
  profile inventory, current provider-route readiness, code authority, bounds,
  and zero-effect vector before model or provider calls.
- Ordinary repair, testing, and shadow work may proceed only after activation.
  Human gold remains a hard gate; no agent may infer or prefill it.
- Any live identity, knowledge, provider, Graphiti, or external mutation is a
  significant departure and requires a different bounded plan.

## Execution graph

| Packet | Depends on | Bounded outcome | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| D0 diagnostic authority | activation | Reclassify every Plan 0064 exposed source as development and freeze the exact false-acceptance/context-failure denominator | New Plan 0065 module, tests, private diagnostic manifest/receipt | Exact replay or fail closed on drift |
| D1 acoustic safety recovery | D0 | Separate probe/diarization contamination, calibration, margin, and model-consensus hypotheses; freeze one correction policy | Versioned acoustic policy module, tests, private development evidence | Development safety gate passes or `acoustic_recovery_failed` |
| D2 contextual evidence recovery | D0 | Repair prepared-evidence citation validity, verify route readiness, and produce safe candidates or reason-coded unavailability | Versioned context adapter/policy, tests, private provider receipts | Context development gate passes, awaits capacity, or `context_recovery_failed` |
| D3 joined/residual development gate | D1 + D2 | Demonstrate pillar agreement and the actual residual rule on reviewed development evidence | Versioned resolver/gate, tests, private review and gate receipts | Non-vacuous development gate passes or terminal withhold |
| E0 fresh authority freeze | D3 | Freeze corrected code/policy and the next oldest-forward cohort before prediction or gold | New cohort module, tests, private manifest/receipt | Exact exclusion and denominator replay |
| E1 blind shadow execution | E0 | Run independent acoustic/context lanes and the joined/residual resolver without gold | Versioned execution modules, private evidence/receipt | Complete predictions or one reason-coded terminal failure |
| E2 human-gold review | E1 | Publish one blind direct-audio decision for every slot, with the original recording basename present in review v1 | Review module/assets, tests, private clips/authority | Complete literal export or await human review |
| E3 terminal measurement | E2 | Measure the pre-frozen policy once and emit readiness or withhold | Measurement module, tests, private terminal receipt | `ready_for_separate_local_acceptance_plan` or `withhold` |

D1 and D2 are logically independent after D0. Under the current system they
run serially with intended active-agent concurrency `1`; no subagent is
authorized. D3 is the integration join and E0-E3 are the serialized critical
path. A provider-capacity wait does not authorize a retry beyond the configured
route budget.

## Packet requirements

### D0 | Development-only diagnostic freeze

- Bind Plan 0064 authority bridge `031ce0f0...`, human-gold content
  `1645c31a...`, P1 evidence `b6a87465...`, P2 receipt `50a7f4fd...`, P3
  receipt `b630d12d...`, measurement `baa26f05...`, and terminal
  `f178f418...`.
- Add every Plan 0064 recording hash, source window, clip, review, prediction,
  and decision reference to the development/exposure exclusion set.
- Recompute the acoustic and context failure classifications without copying
  private identities or notes into tracked artifacts.
- Preserve the existing Plan 0064 terminal replay as a mandatory regression.

### D1 | Acoustic safety recovery

- Reproduce the one wrong and 11 correct candidates from frozen evidence before
  changing policy.
- Test probe purity and diarization contamination independently from threshold,
  score-margin, model-consensus, profile, and source-overlap hypotheses.
- Do not select a correction merely because it separates the single wrong row.
  Require a reusable, reason-coded safety feature with synthetic counterexamples
  and reviewed-development coverage.
- The development gate requires zero high-support wrong identities, no new
  wrong candidate, at least 10 of the 11 previously correct candidates retained
  as candidates, and every demotion explained by the frozen policy.

### D2 | Contextual evidence recovery

- Correct the four prepared-evidence citation validation failures without
  inventing citations or turning an unsupported calendar factor into support;
  neutralize the factor when prepared evidence is absent.
- Run current route readiness before provider work. Permit at most one primary
  request and one configured fallback per case; capacity exhaustion produces
  an await/withhold receipt, not a third route or silent retry.
- Preserve provider, account, tenant, retrieval, and as-of-time lineage.
- The development gate requires zero schema/citation violations, one terminal
  disposition for every slot, at least one correct prepared candidate match,
  complete candidate provenance, and zero person inference from an unavailable
  workflow.

### D3 | Non-vacuous joined/residual development gate

- Require at least one correct pillar-agreement candidate and one correct
  `two_known_plus_one_independently_supported_residual` candidate with complete
  lineage and zero high-support wrong identities.
- If the existing development corpus contains no genuine residual population,
  freeze at most six additional development conversations and obtain complete
  direct-audio human gold before using them. Those recordings immediately join
  the permanent evaluation exclusion set.
- Selection of additional development cases may use known structure because
  they are explicitly development data; it must not consume or preview the
  later E0 cohort.
- If the bounded development corpus still cannot exercise the rule, terminate
  `residual_population_infeasible` and do not open E0.

### E0-E3 | One fresh blind evaluation

- Continue oldest-forward after the complete exposure set. Structural
  selection may use transcript/audio availability and diarization cardinality,
  never identity gold, speaker names, model outcome, or likely pass status.
- Freeze at most 12 recordings, their exact source hashes/windows, current
  profile/person bindings, corrected policy, code authority, provider
  readiness, and all exclusion hashes before E1.
- E1 writes predictions before any E2 gold exists. It must retain independent
  acoustic/context evidence and exact joined/residual lineage.
- E2 review v1 includes the hash-bound original recording basename, direct
  per-speaker audio, no model prediction, no preselected answer, and a disabled
  export until every slot has one literal decision.
- E3 is the only reveal/measurement. The cohort becomes development evidence
  immediately after reveal and cannot be retried under a new policy.

## Execution bounds

- `max_work_unit_attempts`: 2 for D0-D3 and E0 construction/validation.
- `max_policy_revisions`: 2 before the fresh E0 freeze; 0 after E0.
- `max_review_rework_cycles`: 1 closed-world cycle for accepted blocking review
  findings.
- `max_additional_development_conversations`: 6.
- `max_fresh_evaluation_conversations`: 12.
- `max_provider_requests_per_case`: 1 primary plus 1 configured fallback.
- `max_fresh_evaluation_runs`: 1; no retry after gold reveal.
- `max_automatic_policy_bands`: 0; this plan cannot enable apply.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after every packet, before any provider/model call,
  before human review, and before terminal measurement.
- `review_discovery_passes`: 1 inherited bounded discovery pass.
- `review_verification_mode`: closed-world after candidate adjudication.

Delegation receipt: `not_spawned`. Current system authority forbids proactive
subagents unless the user explicitly requests them.

## Acceptance Criteria

- Plan 0064 artifacts continue to replay byte/hash exactly under their original
  schemas and terminal decision.
- Every Plan 0064 and Plan 0065 development source is excluded from fresh
  evaluation by recording hash and overlapping source window.
- The frozen development policy produces zero high-support wrong identities,
  retains at least 10 of 11 prior correct acoustic candidates, and explains
  every review/abstention.
- All contextual factors cite prepared evidence or are neutral; no missing,
  post-as-of, duplicated, or unavailable source becomes support.
- Reviewed development contains at least one correct pillar agreement and one
  correct actual residual-rule acceptance with complete lineage and zero
  high-support wrong identities.
- The fresh cohort is selected chronologically without identity/outcome
  leakage, scored before gold, and receives one complete literal decision per
  slot.
- Fresh evaluation has zero high-support wrong or human-unverifiable
  identities, at least one correct lineage-complete combined acceptance, and
  at least one correct lineage-complete actual residual acceptance.
- Ambiguous, contradictory, mixed-speaker, background-noise, missing-context,
  and unavailable-provider cases abstain or route to useful review.
- The terminal result is non-applying. Speaker assignment, enrollment, profile,
  reference, threshold-default, knowledge, Graphiti, provider, and external
  write counters all remain zero.

If any non-vacuous development or fresh criterion is absent, the correct
terminal result is `withhold`; a zero-candidate result cannot pass vacuously.

## Validation

- Red-capable unit tests for mixed/background probes, diarization contamination,
  close-margin impostors, cross-model disagreement, and correct-candidate
  retention.
- Exact prepared-reference tests for calendar-factor citations, neutralization,
  route exhaustion, candidate provenance, temporal bounds, and duplicate
  provider records.
- Resolver tests for pillar agreement/conflict, one-to-one and multi-label
  handling, residual acceptance, elimination-only abstention, provider failure,
  and material contradiction.
- Byte/hash replay of Plan 0064 plus independent recomputation of every Plan
  0065 development, cohort, execution, review, and terminal receipt.
- Private-mode, containment, symlink, source-hash, exposure-set, and
  source-window overlap checks.
- Browser proof that every E2 card shows the correct original recording
  basename beside playable direct audio and the corresponding blank decision;
  desktop and narrow-mobile export behavior must match.
- Focused tests, full pytest, Python compilation, active/goal planning audits,
  CodeGraph post-edit readback, `git diff --check`, clean commits, push, and
  exact upstream equality.
- A transcription/DOCX manual smoke is required only if the implementation
  changes normalized transcription or export behavior.

## Definition of done

Plan 0065 is complete when the Plan 0064 blockers have been corrected under a
versioned no-apply policy, the actual residual rule passes reviewed development,
and one new oldest-forward blind cohort produces complete human-gold metrics
with zero high-support wrong/unverifiable identities plus non-vacuous correct
combined and residual acceptances. Completion emits only
`ready_for_separate_local_acceptance_plan`; otherwise the plan closes with a
reason-coded `withhold`. No local or external effect is part of done.
