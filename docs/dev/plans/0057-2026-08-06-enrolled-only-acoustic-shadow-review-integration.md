# Plan 0057 | Enrolled-only acoustic shadow review integration

State: OPEN

Lane: P10

## Scope

Integrate the existing two-subject acoustic proposal contract into the ordinary
transcript identity-review flow as non-authoritative shadow evidence, then run
one bounded three-recording fresh batch with complete eligibility, proposal,
review, correctness, and stop-reason denominators. The slice may display and
privately review acoustic subject-ID proposals, but it may not apply a speaker
assignment or mutate identity, contact, relationship, profile, reference, or
provider state.

The exact preflight cohort is limited to three already-ingested recordings:
one later 2026-08-05 recording and two 2026-08-06 recordings. They span at
least two meeting contexts, were recorded after the Plan 0056 source, and their
media hashes are absent from retained Plan 0037 and Plan 0056 JSON evidence.
Private paths, transcript text, human labels, audio, scores, and clips remain
outside the repository and are bound only in user-scoped runtime authority.

## Vision Outcomes And Maturity Movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Acoustic speaker identity | Level 2 isolated bounded-pilot evidence | Level 2 integrated-shadow evidence | Acoustic subject-ID evidence appears in the ordinary identity-review payload for every eligible batch conversation |
| Pipeline yield | One isolated recording with no ordinary-flow denominator | Exact three-recording batch with complete stage and stop denominators | Frozen cohort, execution receipt, per-recording stage ledger, and terminal audit |
| Identity quality | One two-speaker reviewed pilot | Fresh-batch proposal precision, enrolled recall, wrong/high-confidence-wrong rates, abstention, and review burden | Complete private human review plus independent recomputation |
| Knowledge integrity | Stable enrolled IDs and unchanged state proved once | Stable IDs and unchanged state proved across the integrated flow | Hash-bound evidence, deterministic replay, and identical before/after/current state snapshots |
| Automatic assignment/profile learning | Level 0 | Level 0 | Every assignment, identity, provider, profile, and default-integration mutation flag remains false |

This advances the north-star speaker and provenance outcomes by making bounded
acoustic evidence available where operators already resolve diarized speakers.
It does not complete automatic contextualization, canonical-person resolution,
relationship inference, knowledge-store acceptance, or production speaker
assignment.

## Non-Goals

- No automatic or human-applied speaker assignment in this plan.
- No creation or mutation of people, contacts, aliases, roles, relationships,
  speaker assignments, acoustic profiles, references, or provider records.
- No GWS, Odollo, receipts-repository, calendar, local-contact, Graphiti, or
  other provider write.
- No profile learning, enrollment, supersession, threshold selection, default
  integration, production enablement, or historical reprocessing.
- No promotion of names, diarization labels, role labels, provider IDs, or
  evaluation-only IDs into canonical identity.
- No expansion beyond the two enrolled acoustic subject IDs or the exact
  three-recording cohort.
- No implementation of the deferred P09 contact, role, relationship, or
  bounded multi-hop retrieval contract.

## Current State

Plan 0056 is closed at terminal
`plan_next_bounded_integration_milestone`. Its isolated two-speaker pilot
produced one confirmed and one rejected proposal, zero wrong or
high-confidence-wrong assignments, and identical identity state before and
after execution. It applied no assignment and authorized only a separate
bounded successor.

The ordinary review flow is exposed by `conversation_identity_review()` in
`transcript_api.py`. Its current confirmed/deferred action path persists
speaker-assignment and, for confirmation, contact rows. That mutating action is
outside this plan. The integration seam is therefore the read-side identity
review payload: a dedicated acoustic shadow-evidence module will validate and
attach immutable private evidence without calling
`record_speaker_identity_review()` or writing the transcript database.

Read-only preflight found three available source hashes with zero overlap in
retained Plan 0037 and Plan 0056 evidence. The exact paths and document IDs are
private runtime inputs. No Plan 0057 authority, model execution, review, or
audit receipt exists yet.

## Execution Graph

| Unit | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 plan authority | Plan 0056 closure, current repo, exact cohort preflight | Freeze cohort, allowlist, evidence-flow contract, bounds, and negative actions | Private Plan 0057 P0 runtime tree | Replay succeeds or a cohort/privacy/disjointness hard stop fires |
| P1 integration module | P0 | Add validated read-only acoustic evidence to the ordinary identity-review payload | Product code and focused tests | Every forged, mismatched, or mutation-bearing artifact fails closed |
| P2 batch execution | P0, P1, clean pushed authority | Run local diarization, bounded clips/transcription, nine acoustic units, and proposal preparation once per recording | Private Plan 0057 P2 runtime tree | Exact batch completes, or one explicit stop receipt freezes |
| G1 human review | P2 | Record one decision for every eligible diarized speaker without applying it | Private review receipt and review artifact | Complete decisions or awaiting-human-review stop |
| P3 independent audit | G1 | Recompute yield, correctness, burden, mutations, replay, and terminal decision | Private terminal receipt plus repo closeout docs | `stop`, `refine`, or `plan_next_bounded_milestone` |

The critical path is P0 -> P1 -> P2 -> G1 -> P3. There is no useful disjoint
implementation lane in the current environment because proactive delegation is
disabled and the integration, runner, and audit contracts share one authority
surface. Delegation receipt: `not_spawned`; reason:
`runtime_policy_disables_proactive_delegation`.

## Exact Population And Freshness Contract

- Exactly 3 source recordings and exactly 3 already-ingested transcript
  document IDs.
- At least 2 distinct conversation or meeting-context identifiers.
- Every recording start is later than the frozen Plan 0056 source start.
- Every media content hash is unique within the cohort and absent from the
  retained Plan 0037 and Plan 0056 evidence roots.
- The cohort and transcript bindings freeze before local diarization or any
  acoustic model loads.
- Plan 0056 outcomes and Plan 0057 human gold are unavailable to proposal
  construction.
- Every source either enters the acoustic shadow path or records one explicit,
  enumerated stop reason.

## Evidence Flow Contract

- Machine identity is either one of the exact two enrolled subject IDs or
  `null` for abstention.
- Each evidence row binds the transcript document ID, conversation key, source
  hash, recording-local speaker reference, disposition, confidence band,
  supporting/opposing unit counts, evidence hash, and negative action vector.
- Display labels, if used in the private review surface, are review attributes
  only and never machine identity.
- The ordinary identity-review payload may expose validated evidence, its
  freshness, and its review status. It may not translate evidence into a
  contact, assignment, canonical person, relationship, or provider record.
- Missing, stale, mismatched, non-allowlisted, name-derived, or mutation-bearing
  evidence fails closed and is omitted with an explicit read-side reason.
- Cache fingerprints include the accepted acoustic evidence hash so a new
  evidence receipt cannot leave a stale identity-review payload.

## Acceptance Criteria

- P0 freezes and replays the exact three-recording, two-context,
  post-Plan-0056, prior-disjoint cohort before model execution.
- Exactly the two existing enrolled subject IDs are accepted as non-null
  machine proposals; names, contacts, roles, provider IDs, and new IDs fail
  closed.
- Every eligible recording and every diarized speaker with at least six usable
  seconds has a proposal, review, or abstention row in the integrated evidence
  ledger.
- `conversation_identity_review()` exposes the validated shadow evidence for
  the matching transcript/conversation and exposes none for mismatched or
  unbound artifacts.
- No Plan 0057 path invokes the mutating speaker-review action or changes
  primary/knowledge/profile/reference state.
- Every proposal receives an explicit human confirmation, rejection,
  neither-enrolled, or unknown decision before terminal scoring.
- The terminal audit reports eligible and entered recordings, eligible and
  covered speakers, proposals, confirmations, rejections, abstentions,
  reviews, correct and wrong proposal dispositions, high-confidence wrongs,
  enrolled recall, proposal precision, review burden, and every stop reason.
- P0, execution, review, evidence integration, and terminal receipts replay
  deterministically with private directories `0700` and retained files `0600`.
- Focused tests and the full repository suite pass, and a human-facing review
  artifact is inspected through the configured preview workflow.
- The terminal decision is frozen under this rule: `stop` on any mutation,
  non-allowlisted identity, incomplete denominator, source overlap, gold
  leakage, non-replayable artifact, or high-confidence wrong proposal;
  `refine` on any other wrong proposal or incomplete enrolled recall; otherwise
  `plan_next_bounded_milestone`.

## Validation

- Planning-contract audit before and after opening the plan.
- Focused unit tests at the acoustic shadow-evidence module interface,
  transcript API read-side seam, batch runner, review parser, and independent
  audit.
- Read-only before/after/current snapshots of primary contacts and speaker
  assignments, conversation knowledge identity tables, acoustic profiles, and
  biometric references.
- Exact private file-mode audit, canonical JSON/body hashes, source hashes,
  cohort hashes, and idempotent replays.
- One bounded local GPU execution after clean pushed P0/P1 authority; no retry
  after partial model output without a new recorded disposition.
- Browser/preview inspection of the ordinary review flow and the complete
  private decision surface.
- Full `.venv/bin/python -m pytest -q` before terminal closeout.

## Safeguards And Hard Stops

- Stop on any proposal containing a name, contact/provider identifier, role
  label, diarization label, or non-allowlisted ID as canonical identity.
- Stop on any identity, relationship, assignment, profile, reference,
  knowledge-store, Graphiti, or provider mutation.
- Stop on an unclean or upstream-divergent authority freeze, source overlap,
  cohort drift, transcript/source mismatch, gold leakage, private permission
  failure, incomplete speaker/review denominator, stale cache, non-replayable
  receipt, or high-confidence wrong proposal.
- Stop rather than widening the cohort, adding people, relaxing six-second
  eligibility, changing calibrated thresholds, or building P09 relationship
  work inside this plan.

## Local Goal Bounds

`max_work_unit_attempts: 2`

`max_acoustic_execution_attempts: 1`

`max_review_rework_cycles: 1`

`max_hardening_checkpoints: 2`

`checkpoint_interval: 1 completed execution unit`

`authorization_gate: significant_departure_only`

`retry_budget_mode: renewable_execution_window`

`review_discovery_passes: 1`

`review_verification_mode: closed_world`

`review_finding_fields: criterion, evidence, consequence, reproducer, confidence, suggested_disposition`

`review_disposition_values: blocking | nonblocking_backlog | rejected | needs_evidence`

`checkpoint_record_fields: plan_version, state_transition, progress_classification, evidence, subagent_status, authority_classification, review_disposition_summary, next_action_or_stop_reason`

The one broad drift-discovery pass occurs after P1 implementation and before
P2 execution. Later review is closed-world against accepted blocking findings
and critical regressions introduced by remediation.

## State And Authority

Plan states: `ready`, `active`, `awaiting-review`, `awaiting-gate`, `blocked`,
`complete`, `failed`, `cancelled`.

The user-approved goal supplies standing authority for ordinary in-envelope
implementation, local private execution, validation, repair, retest, and
bounded successor work. New authorization is required only for a significant
departure described by repo policy or an explicit safeguard/hard stop. Human
identity decisions remain a literal G1 gate; standing authority cannot invent
or infer them.

## Terminal Decision

Pending. One of `stop`, `refine`, or `plan_next_bounded_milestone` will be
frozen only after complete human review and independent audit. Opening and
implementing this plan does not authorize automatic assignment, production
integration, profile learning, identity creation, provider write-back,
relationship inference, or historical reprocessing.
