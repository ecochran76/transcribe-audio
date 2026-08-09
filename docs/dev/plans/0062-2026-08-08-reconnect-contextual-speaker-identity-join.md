# Plan 0062 | Reconnect contextual speaker identity to the canonical/acoustic join

State: OPEN

Checkpoint: P1-P2 complete; P3 ready

Lane: P09

Cross-lane dependency: closed Plans 0025, 0036, 0059, 0060, and 0061

Critical-Path Owner: primary agent

## Scope

Reconnect the implemented Plan 0025 two-phase, speaker-specific identity
workflow to the canonical-person and acoustic evidence contracts completed by
Plans 0059-0060. Execute the existing Clue Discovery and Identity Evaluation
workflow on the exact three recent Plan 0060 conversations, preserve
calendar/transcript/provider evidence per diarized speaker, translate validated
candidate matches into review-only canonical-person proposals, retain named
unlisted-person suggestions, and join acoustic evidence only through explicit
acoustic-subject-to-person bindings.

The exact execution cohort is:

- `8232481d6076282d7a8e` with four speaker labels;
- `47ea79857aa1ac2d1d79` with three speaker labels; and
- `92d2cd3ed6fc6c1275ca` with three speaker labels.

This plan supersedes the Plan 0060 inference behavior but does not rewrite its
frozen receipts or the Plan 0061 comparison. The Plan 0036 baseline and
prediction bodies remain sealed. Reusable source code, tests, and redacted
contracts belong in the repository. Model ledgers, provider evidence,
transcript content, names, email addresses, review clips, decisions, and
candidate payloads remain private under user-scoped runtime storage.

This document is planning authority only. A distinct A0 checkpoint must move
it to `OPEN` before model execution, provider retrieval, private proposal
preparation, source changes, or preview publication.

Planning authority is committed and pushed at
`9e6fd82f8858aadfd6c30da4d0996ffe88c28507`. The user's `plan and execute`
instruction activated A0 after the exact readbacks below.

## Vision outcomes and maturity movement

| Capability | Current | Target | Required evidence |
| --- | --- | --- | --- |
| Contextual speaker inference | Level 2 two-phase workflow exists and has a sealed ten-case baseline, but is not used by the recent join | Level 2 replayable speaker-specific inference on the exact three-conversation cohort | Ten speaker slots receive validated per-speaker assignments or explicit unresolved/unlisted outcomes with cited clues and bounded provenance |
| Canonical-person join | Level 2 storage/retrieval contracts exist; Plan 0060 repeated one recording-level compatibility snapshot | Level 2 explicit proposal-to-canonical binding | Every proposed `person_id` is present in the frozen candidate authority; unmapped and unlisted people remain review-only suggestions |
| Acoustic/context combination | Level 2 acoustic evidence and joined contracts exist; all Plan 0060 conditions were hard-coded abstentions | Level 2 evidence-preserving joined proposals or reason-coded abstentions | Context-only, acoustic-only, and combined evaluations preserve separate factors, alternatives, contradictions, caps, and exact bindings |
| Human identity review | Level 2 direct-audio review works, but the recent packet contained no useful name inference | Level 2 direct-audio review of actual contextual proposals | Authenticated review surface shows one audio clip, proposal, alternatives, evidence, and explicit unresolved/not-listed choices per speaker |
| Automatic assignment and profile learning | Level 0 | Level 0 unchanged | No apply path; assignment, contact, person, relationship, profile, reference, provider, watcher, and Graphiti mutation counts remain zero |

This advances VISION outcomes 3, 4, 6, 7, and 8. It makes the prior contextual
reasoning work part of the joined product loop instead of treating biometric
evidence as a replacement. It deliberately does not establish Level 3
automatic identity or authorize biometric enrollment.

## Current State

At planning freeze, branch `plan-0037-campaign` is clean and upstream-even at
`81ef37b6251a96247be64a068723489e73e2f7a7`. The installed transcript service
is active/running with zero restarts. The user-scoped store passes SQLite
`quick_check` with 466 documents and 3 speaker assignments. All three exact
cohort documents return HTTP 200 from the installed speaker-preprocessing
status endpoint and report `not_started`, zero evaluations, and zero review
decisions.

The transcript artifacts contain 232, 27, and 194 utterances respectively and
carry primary-event participant plus matching-calendar structures. The closed
Plan 0061 audit proved that Plan 0060 instead froze two compatibility contacts,
zero canonical calendar candidates, and empty clue IDs, then repeated that
recording-level set for every speaker. `execute_blinded_join` explicitly emitted
`outcome="abstained"` and `proposed_person_id=None` for all conditions.

Plan 0025 already provides the required deep module interfaces:

- `build_clue_discovery_packet` creates per-speaker utterance clues and calendar
  context;
- `prepare_transcript_identity_evidence` performs host-owned, scope-bound
  canonical-person retrieval;
- `prepare_identity_evaluation` prepares the second reviewed model phase; and
- `validate_and_score_identity_evaluation` rejects unprepared references and
  attaches host-owned confidence.

The missing implementation is the reusable seam that consumes that validated
result and produces speaker-specific canonical/acoustic joined evaluations.

## A0 activation checkpoint

Branch `plan-0037-campaign` and its upstream were exact at
`9e6fd82f8858aadfd6c30da4d0996ffe88c28507` with a clean worktree. Both
`transcripts.service` and `transcribe-watch.service` were active/running with
zero restarts. The user-scoped database passed SQLite `quick_check` with 466
documents, 2 contacts, and 3 speaker assignments.

All three cohort documents independently returned HTTP 200 from the installed
speaker-preprocessing status endpoint with `not_started`, zero evaluations,
and zero review decisions. The first status call required a longer provider-
aware response window but completed unchanged on retry; this is observed
latency, not evidence of a missing document.

The Plan 0036 superseding baseline remains `predictions_complete` with 10/10
captured predictions, `gold_content_included=false`, source commit `fee6ef6`,
and a next gate requiring independent operator gold. No prediction body or
partial gold was opened. Plan 0061's terminal audit remains mode `0600`, status
`complete`, terminal comparison SHA-256
`12a45055b7c3e9fc15af0e297af4b4decde67c32603c981642857678c476f4fd`,
and `live_mutation_count=0`.

A0 therefore transitions `PLANNED/A0 -> OPEN/P1-P2-ready`. Authority is the
exact three-conversation cohort, six primary reviewed model turns with the
bounded reference-repair allowance, inherited provider budgets, private
evaluation sidecars, repository implementation/tests, and all negative actions
preserved. Progress classification is `outcome_progress`; accepted finding
ledger remains empty; delegation remains `not_spawned` under current system
authority.

## P1-P2 execution evidence

The existing two-phase workflow completed on all three exact conversations:
three Clue Discovery turns, three Identity Evaluation turns, and one bounded
reference-only repair. All 10 source speaker labels were covered. The model
returned seven `unlisted` suggestion records, proving that the pre-biometric
calendar/context workflow does recover useful names even when those people are
absent from the current canonical-person candidate set. No proposal was
applied and all three evaluation sidecars remain `awaiting_human_confirmation`.

Two cohort-specific incompatibilities were found and resolved fail closed in
the reusable join:

- one repaired readout covered three source labels through four overlapping
  assignments (six label appearances), so duplicate coverage now becomes
  `context_duplicate_speaker_coverage` while preserving its suggestions; and
- the existing workflow names source diarization labels `A/B/C/D`, whereas the
  acoustic contract uses `SPEAKER_1/...`; the join now requires and hashes a
  complete one-to-one label binding rather than assuming equivalence.

The prepared prompt also separates operator-authored owner/relationship scope
from the redacted explicit retrieval scope. The adapter rejoins those records
only by their shared prepared `source_id`, matching the existing provider-scope
normalization contract and preserving account, tenant, capability, and budget
boundaries.

The real three-conversation smoke produced 30 evaluations over 10 speakers.
One enrolled acoustic subject appeared in three clips but had zero explicit
bindings to a prepared canonical person; those acoustic and combined lanes
therefore abstained instead of guessing. The contextual suggestions remain in
the review outcome and are the evidence needed for the next human binding
decision. Focused join, preprocessing, and orchestration tests pass 40/40.

## Authority and non-goals

- Do not reveal, modify, or reuse Plan 0036 prediction bodies or partial gold.
- Do not rewrite or rerun Plan 0060 or Plan 0061 frozen receipts.
- Do not infer identity from calendar membership, an acoustic subject, a
  filename, or a compatibility contact alone.
- Do not create or update contacts, canonical people, speaker assignments,
  roles, relationships, biometric references, profiles, embeddings, provider
  records, Graphiti memory, watcher state, or historical artifacts.
- Do not send raw audio, biometric features, raw provider payloads, or full
  transcripts to App Intelligence. Existing bounded prompt contracts remain
  authoritative.
- Do not automatically accept a proposal. Every proposal and every unlisted
  suggestion remains subject to explicit human confirmation.
- Do not use a current-run proposal as evidence for itself.

Allowed effects are repository source/tests/docs, private App Intelligence run
ledgers, bounded read-only provider retrieval receipts and shadow database
copies, conversation processing evaluation sidecars, and an authenticated
minimum-copy review packet. The review packet may contain the already-frozen
per-speaker clips, candidate display labels, bounded cited evidence summaries,
and client-only controls; it may not contain full recordings or raw transcript
bodies.

## Execution bounds

- `max_work_unit_attempts`: 2 per implementation or runtime unit.
- `max_primary_model_turns`: 6, one Clue Discovery and one Identity Evaluation
  turn for each of three conversations.
- `max_reference_repairs`: 1 per phase per conversation, reference-only and
  substantive-conclusion preserving.
- `max_provider_calls`: inherited per-request limits from the reviewed
  `conversation_identity_policy`; no override or expansion.
- `max_review_rework_cycles`: 1 for the final authenticated review surface.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after each conversation execution and each validated
  implementation packet.
- `review_discovery_passes`: 1 broad fresh-context pass for the goal; later
  verification is closed-world against accepted findings and critical
  regressions introduced by remediation.

Delegation receipt: `not_spawned`. Current system authority forbids proactive
subagents unless the user explicitly requests them; the primary agent owns the
critical path, all writes, validation, and reconciliation.

## Execution graph

| Unit | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| A0 activation | User `plan and execute`, clean upstream-even repo, current runtime readback | Freeze exact cohort, source/model/provider permissions, non-effects, and inherited hashes | Plan, ROADMAP, RUNBOOK, private activation receipt only | `OPEN` only if current store, services, cohort, Plan 0036 seal, and Plan 0060/0061 authority are intact |
| P1 contextual execution | A0 | Run the existing two-phase workflow independently for all three conversations | Private App Intelligence/retrieval ledgers and review-gated processing sidecars | 3/3 conversations and 10/10 speaker labels yield validated assignments or explicit unresolved/unlisted outcomes; no apply |
| P2 reusable join module | A0 and existing orchestration contracts | Add one deep interface that validates contextual output, requires explicit canonical/acoustic bindings, and returns immutable three-condition evaluations | Focused Python module and tests | Candidate matches may propose; unmapped, unlisted, conflict, and missing-evidence cases abstain with stable reasons |
| P3 joined cohort packet | P1 and P2 | Bind the exact P1 proposals to existing Plan 0060 acoustic evidence without reading Plan 0061 gold | Private immutable joined manifest and receipt | Exactly 30 evaluations over the 10 speaker slots; context and acoustic factors remain separate; replay is exact |
| P4 authenticated review | P3 | Publish one direct-audio, non-applying worksheet containing actual proposals and named unlisted suggestions | Minimum-copy private packet and authenticated Previews artifact | Ten blank decisions, playable clips, complete evidence/proposal display, strict export, no POST/apply path |
| P5 human decision and comparison | Literal operator export from P4 | Freeze named human decisions, compare contextual and combined outcomes, and decide refine/advance | Private decision/comparison receipts and durable closeout docs | Complete exact denominator, independent recomputation, unchanged forbidden mutations, explicit terminal decision |

P1 and P2 may proceed independently after A0, but P3 is forbidden until both
are complete and replayable. P4 stops at the human gate. P5 may not infer,
repair, or synthesize a person decision from silence, earlier `not_listed`
choices, calendar attendance, model output, or biometric evidence.

## Acceptance Criteria

- The exact three-conversation/ten-speaker denominator is preserved.
- Clue Discovery receives each speaker's own bounded utterance references and
  the transcript's available calendar participant/matching-calendar evidence.
- Host-owned retrieval executes only after validated discovery and preserves
  account, tenant, capability, as-of-time, budget, warning, failure, and
  lineage fields.
- Identity Evaluation returns one reviewable status for every speaker label;
  omissions, duplicate label coverage, invented references, and candidate IDs
  outside the prepared authority fail closed.
- Candidate matches can become review-only canonical proposals only through an
  explicit prepared-person-to-canonical-person binding.
- Unlisted suggested people survive into review with their bounded name,
  organization, or email hints but cannot become canonical people or biometric
  identities without a separate human decision and later mutation authority.
- Acoustic evidence can support a canonical person only through an explicit
  reviewed acoustic-subject-to-person binding. A context/acoustic disagreement
  abstains and exposes both alternatives.
- Context-only, acoustic-only, and combined evaluations preserve their own
  factors, cited evidence, independence groups, confidence, cap reasons,
  alternatives, contradictions, source failures, and stable abstention reason.
- The review UI exposes actual speaker-specific proposals rather than one
  recording-level candidate pool, plays the exact per-speaker clip remotely,
  starts with zero selections, and contains no network write or apply path.
- Plan 0036 remains sealed; Plan 0060/0061 replay and live identity-state
  baselines remain unchanged.

## Validation

- Focused unit tests through the new join module interface for candidate match,
  unlisted suggestion, unmapped person, missing speaker output, acoustic-only
  mapping, corroboration, contradiction, required-provider failure, invalid
  reference, and duplicate-speaker coverage.
- Existing speaker preprocessing, retrieval, orchestration, shadow review,
  transcript API, and Plan 0061 comparison tests.
- Python compilation, frontend production build if P4 changes frontend code,
  deterministic planning audit, CodeGraph post-edit readback, `git diff
  --check`, and full pytest suite.
- Installed-service status/API smoke for all three cohort conversations without
  exposing private proposal bodies in tracked artifacts or terminal output.
- Immutable private receipt replay, `0700` directory and `0600` file checks,
  SQLite `quick_check`, exact live identity-state comparison, service restart
  counts, and all forbidden mutation counters.
- One fresh-context drift review followed by closed-world remediation
  verification of accepted blocking findings only.

## Definition of done

Plan 0062 is complete only when the existing contextual inference path has run
on all ten recent speaker slots, its validated results drive real
speaker-specific canonical/acoustic shadow evaluations, the direct-audio review
surface presents those actual proposals without applying them, literal human
decisions are independently scored, and every forbidden live mutation remains
zero. Passing synthetic tests or merely preparing prompts does not complete the
plan.
