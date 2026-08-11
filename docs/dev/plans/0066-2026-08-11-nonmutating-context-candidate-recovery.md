# Plan 0066 | Non-mutating prepared context candidate recovery

State: CLOSED

Active packet: none. A0 froze and replayed activation
`8b1580e69c281e61...`; A1 froze and replayed manifest `400f67864170dbd0...`;
A2 consumed exactly six primary turns with zero fallback and zero retry; and A3
sealed terminal `c5a843f80939972f...` with reason
`evidence_reference_compliance_failed`. Every A1 packet contained the exact
six-person reviewed roster, retained its original recording basename only in
the private receipt, and left all selected source/stored/index evidence exact.
The full 12-document binding hash remained `95533131a221486a...`.

Execution outcome: the non-mutating preparation and reviewed-roster product
repair passed, advancing source integrity and candidate preparation to the
target Level 2 shadow capability. Candidate recovery did not pass: all six
model readouts cited calendar evidence IDs that the frozen second-pass
validator did not authorize, so strict validation failed before persistence.
The final measurement is zero correct, zero wrong, 22 abstained slots, six
unavailable cases, six validation failures, and zero incomplete candidate-
provenance records. Context evidence quality remains Level 1. No assignment
was applied and no joined/residual or fresh gate opened.

Post-closeout correction: Plan 0067 established that all seven rejected
citation occurrences were present in their case-local, host-validated
first-pass `calendar_clue_ids`. They were not invented by the model. The
second-pass builder dropped the explicit `calendar_evidence` catalog, leaving
the validator unable to recognize those otherwise prepared IDs. This corrects
the diagnosis but does not reopen, rewrite, or relabel terminal
`c5a843f80939972f...`.

Checkpoint: Plan 0065 is closed `withhold` at terminal `e73e2ebc...` because
its D2 context gate produced zero correct prepared candidates. All 11 completed
Plan 0065 packets contained zero prepared people even when bounded provenance
was available. D2 also exposed that retrieval-mode preparation lazily mutated
three legacy transcript containers; reconciliation `a916fca4...` restored the
five affected copies and three index rows exactly. The operator instructed the
agent to plan and execute the successor on 2026-08-11 from clean,
upstream-even commit `8bc6b01e91c848c9e2752358c83dea7b170ed8ae`.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064 and 0065

Critical-Path Owner: primary agent

## Scope

Repair the host-owned retrieval preparation seam so a legacy transcript can be
prepared without changing its source/stored container or transcript index row,
and so the private shadow store contains the complete bounded roster of current
reviewed people before identity evaluation. Then perform one bounded
development-only model pass against previously exposed Plan 0064/0065 evidence
to prove or disprove at least one correct, lineage-complete prepared context
candidate.

All Plan 0064 decisions and Plan 0065 packets remain development/hindsight
evidence. Candidate-roster construction must be independent of gold: it uses
the complete current reviewed-person set, not the identities present in a
selected case. Development case selection may use known gold because it is
explicitly diagnostic, but every selected case and recording stays permanently
excluded from unseen evaluation.

The successor is shadow-only and non-applying. It may write content-addressed
private transcript snapshots, shadow knowledge rows, model-run artifacts, and
immutable receipts under the user-scoped Plan 0066 runtime. It may not mutate
source/stored transcript containers, the live transcript index, speaker
assignments, enrollments, biometric profiles/references, accepted conversation
knowledge, Graphiti, provider records, or another external system.

## Vision outcomes and maturity movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Contextual speaker candidate preparation | Level 1 implementation retrieves provenance but produced zero prepared people across Plan 0065 | Level 2 shadow preparation with a complete reviewed-person roster and at least one correct prepared candidate on development evidence | Source-bound roster manifest, prepared packets, direct human-gold comparison |
| Replay and source integrity | Level 1 retrieval preparation can mutate legacy transcript containers | Level 2 content-addressed private preparation with exact source/store/index no-diff proof | Before/after hashes, row-byte snapshot, private preparation receipt |
| Context evidence quality | Level 1 schema-valid evidence produced only abstentions | Level 2 only if one correct candidate has complete temporal/provider/account/tenant/evidence lineage and zero wrong candidates | Bounded model evidence and non-vacuous gate |
| Joined/residual identity | Plan 0065 D3 never opened | Remain unchanged | Explicit terminal packet state; this plan does not claim joined or residual acceptance |
| Local/external acceptance | Level 0 automatic apply | Remain Level 0 | Zero-effect terminal receipt |

This advances VISION outcomes 3 and 6 by ensuring the model receives the
host-reviewed candidate set and provenance required for grounded speaker
inference while preserving source evidence byte-for-byte. It enables, but does
not itself complete, outcomes 7 and 8 because no observation is accepted or
projected into live conversation knowledge.

## Current State

- Plan 0065 D2 contains 12 development cases and 39 terminal slot
  dispositions. Eleven cases completed and one was unavailable. Thirty-five
  slots ended `no_prepared_candidate_match` and four ended
  `context_workflow_failed`.
- Every completed Plan 0065 identity packet contains zero people. Packets carry
  between zero and 15 provenance sources, so evidence retrieval and candidate
  preparation are observably distinct blockers.
- The live user-scoped knowledge store contains six reviewed current people.
  All have primary names; four have verified email identities. Their nine
  source records are operator-reviewed and explicitly scoped.
- Retrieval policy currently accepts provider scopes but does not import
  reviewed people/source identities into its private shadow store. Its adapted
  candidate display also falls back to an opaque person ID when no exact email
  is present.
- Retrieval-mode API preparation currently calls durable identity backfill
  before preparing evidence. Plan 0065 proved that this can rewrite legacy
  transcript JSON and synchronize the live index even though the caller asked
  only to prepare shadow evidence.

## Accepted finding ledger

| Finding | Criterion | Evidence | Disposition |
| --- | --- | --- | --- |
| F1 source-container mutation | Retrieval preparation must be read-only for source/store/index | A1 changed zero source/stored/index bytes or rows and replayed exactly | `resolved` |
| F2 zero prepared people | Development packet must contain the bounded reviewed roster | All six A1 packets contain the exact six-person reviewed roster | `resolved` |
| F3 opaque candidate display | Prepared people must carry reviewed primary names without inventing identity | A1 packets carry host-reviewed primary names through the retrieval adapter | `resolved` |
| F4 evidence may remain insufficient | At least one correct prepared candidate is non-vacuous | All six A2 outputs cited valid first-pass calendar IDs omitted from second-pass reference authority, so their candidates could not be measured | `withhold`; corrected by Plan 0067 |

The goal-level broad drift-discovery pass is consumed by this ledger. After A0,
review is closed-world against F1-F4 plus critical regressions introduced by
their remediation.

## Non-Goals

- Do not reopen Plan 0065, relabel its terminal, or treat Plan 0064/0065 data as
  unseen.
- Do not use human-gold identities, operator notes, or acoustic predictions to
  construct or narrow the prepared candidate roster.
- Do not infer a person by elimination or count duplicate source records as
  independent support.
- Do not weaken evidence citation, temporal, scope, or human-confirmation
  rules to make a candidate pass.
- Do not persist a shadow evaluation into a live processing sidecar.
- Do not apply assignments, create people/enrollments, mutate biometric state,
  write live conversation knowledge, change defaults, or write Graphiti.
- Do not open a joined/residual gate or fresh blind cohort in this plan.
- Do not publish private names, emails, person IDs, transcripts, provider
  payloads, or original recording basenames in tracked files.

## Authority and activation

- The explicit operator instruction `plan and execute` activates this bounded
  successor from clean, upstream-even commit
  `8bc6b01e91c848c9e2752358c83dea7b170ed8ae`.
- A0 freezes the Plan 0065 terminal/reconciliation, source/store/index hashes,
  reviewed-person roster membership, provider readiness, code authority,
  development denominator, and zero-effect vector before any model turn.
- The complete reviewed-person roster is host-owned and gold-independent.
  Private labels and external identities stay in mode-`0600` artifacts.
- A significant departure, including any live identity/knowledge/provider
  mutation or fresh cohort, requires a new plan.

## Execution graph

| Packet | Depends on | Bounded outcome | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| A0 activation and diagnosis freeze | activation | Bind exact Plan 0065 failure, reviewed roster, source hashes, and route readiness | Plan module/tests plus private manifest/receipt | Exact replay or fail closed |
| A1 non-mutating preparation seam | A0 | Create content-addressed private transcript identity snapshots and mirror reviewed people into the private shadow store | Product modules/tests and private A1 receipt | All source/store/index bytes exact and six reviewed people prepared, or `preparation_recovery_failed` |
| A2 bounded development inference | A1 | Run one primary model turn for at most six development cases selected before execution | Private prompts/readouts/case receipts only | One terminal disposition per slot; no retry |
| A3 measurement and terminal | A2 | Compare predictions with frozen human gold and close pass/withhold | Private measurement/terminal receipts and tracked closeout docs | `context_candidate_recovered` or reason-coded `withhold` |

Intended active-agent concurrency is `1`. Delegation receipt is `not_spawned`:
current system authority prohibits proactive subagents. The primary agent owns
the critical path, model budget, reconciliation, and completion claim.

## Packet requirements

### A0 | Activation and diagnosis freeze

- Bind Plan 0065 terminal `e73e2ebc...`, terminal file `0f32d0ac...`, D2 receipt
  `8d65f6be...`, and reconciliation `a916fca4...`.
- Freeze the exact 12-case/39-slot development denominator and complete
  six-person reviewed roster without copying private values into tracked files.
- Freeze source/stored artifact hashes and the relevant live transcript-index
  row serialization for byte-level post-A1 comparison.
- Check route readiness without starting a session or sending a model turn.

### A1 | Non-mutating preparation and candidate projection

- For a schema-1 legacy transcript, materialize a deterministic, private,
  content-addressed schema-2 snapshot with derived conversation/recording IDs.
  Never overwrite or synchronize the source/stored artifact.
- Record original transcript path/hash, snapshot path/hash, ID derivation
  policy, document binding, and private containment.
- Mirror only current `reviewed` person snapshots selected by the host into the
  private shadow store, preserving source/account/tenant/identity authority.
- Add every mirrored source scope explicitly to exact-identity lookup without
  creating a provider adapter for the local reviewed scope.
- Carry reviewed primary names into prepared people. Empty/invalid labels fail
  closed; no model or client may invent labels.
- The A1 gate requires exactly six gold-independent reviewed people in every
  prepared development packet, zero source/store/index changes, exact replay,
  and zero model turns.

### A2 | Bounded development inference

- Freeze no more than six Plan 0064/0065 development cases before the first
  turn. Selection may use known development gold but roster construction may
  not; record the distinction explicitly.
- Retain exact prior provenance and add only the host-reviewed candidate roster
  plus source-bound snapshot lineage.
- Permit one primary request per case, zero fallback and zero retry. A provider
  failure becomes a terminal unavailable case.
- Store the original recording basename only in the private case receipt so a
  later human review, if needed under a successor, is answerable.

### A3 | Non-vacuous measurement and terminal

- Require zero wrong prepared candidates, zero schema/citation violations,
  complete candidate provenance, and no candidate from an unavailable flow.
- `context_candidate_recovered` additionally requires at least one correct
  prepared candidate match. A zero-candidate result cannot pass.
- Regardless of outcome, do not open joined/residual or fresh-evaluation work.
  Close with the exact next gate or stop reason.

## Execution bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_policy_revisions`: 1 before A2; 0 after the first model turn.
- `max_development_cases`: 6.
- `max_primary_model_turns`: 6 total and one per case.
- `max_fallback_model_turns`: 0.
- `max_provider_retries`: 0.
- `max_fresh_evaluation_runs`: 0.
- `max_review_rework_cycles`: 1 closed-world cycle for F1-F4.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after every packet and before the first model turn.
- `review_discovery_passes`: 1, already consumed.
- `review_verification_mode`: closed-world.

## Acceptance Criteria

- Retrieval-mode preparation leaves every selected source/stored transcript and
  corresponding live index row byte/hash-identical.
- Every prepared packet contains the exact six-person reviewed roster selected
  by the host, with primary labels and explicit source-scope lineage.
- Candidate roster membership is independent of Plan 0064/0065 human gold.
- Private snapshots and shadow stores are content-addressed, mode-contained,
  replayable, and bound to original source hashes.
- At least one development slot produces a correct prepared candidate with
  complete evidence lineage and zero wrong prepared candidates.
- Missing/contradictory evidence, mixed speakers, background noise, and provider
  failure remain abstentions or unavailable outcomes.
- Speaker assignment, enrollment, biometric, default-threshold, live knowledge,
  Graphiti, provider-write, and external-write counters remain zero.

If the source-integrity or roster gate fails, A2 does not open. If the A2
candidate gate has zero correct matches or any wrong match, the plan closes
`withhold`; it cannot pass vacuously.

## Validation

- Red-capable tests at the real API/retrieval seam for legacy source mutation,
  reviewed-person mirroring, local reviewed scope, primary-name propagation,
  content-addressed replay, and live-index no-diff.
- Existing transcript-artifact, identity-policy/retrieval, preprocessing,
  API, Plan 0065 reconciliation, and terminal regression suites.
- Immutable A0-A3 receipt replay and independent human-gold recomputation.
- Private mode, containment, symlink, source-hash, and exact row-byte checks.
- Focused tests, full pytest, Python compilation, active/goal planning audits,
  CodeGraph post-edit readback, `git diff --check`, clean commits, push, and
  exact upstream equality.
- A transcription/DOCX smoke is required only if normalized transcription or
  export behavior changes.

## Definition of done

Plan 0066 is complete when A1 proves retrieval preparation is source/index
read-only with the exact reviewed roster, A2 consumes no more than the frozen
model budget, and A3 emits either `context_candidate_recovered` with at least
one correct lineage-complete candidate and zero wrong candidates, or a
reason-coded `withhold`. No live identity, knowledge, biometric, provider, or
external effect is part of done.
