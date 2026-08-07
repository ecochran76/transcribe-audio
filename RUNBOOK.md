# Runbook

## Turn 319: Open Plan 0057 acoustic shadow review integration (2026-08-06)

Summary: Opened the separate bounded P10 successor authorized by Plan 0056
closure. Plan 0057 integrates enrolled-only acoustic proposals into the
ordinary transcript identity-review read path as non-authoritative evidence
and binds one exact three-recording fresh batch without authorizing any
assignment or identity/profile/provider mutation.

Plan: `docs/dev/plans/0057-2026-08-06-enrolled-only-acoustic-shadow-review-integration.md`.

Authority and discovery:

- Current branch began clean and upstream-even at `6cfe127`.
- Graphiti runtime and MCP were healthy. Repo group `transcribe_audio_main`
  returned the sourced Plan 0056 closeout episode
  `47d49786-d95a-49e0-810d-e7200d956aa4`, which confirms a separate successor
  is required.
- The deterministic policy selector returned `already-aligned` for the
  `skill-repo-maintainer` profile with the full planning, memory, CodeGraph,
  validation, and preview module set.
- The active planning-contract audit returned `ok: true` before Plan 0057 was
  created.
- CodeGraph was healthy with 238 indexed files. Structural inspection located
  the read-side integration seam in `conversation_identity_review()` and
  confirmed that `record_speaker_identity_review()` is mutating and therefore
  outside this plan.

Population preflight:

- Read-only transcript-store and media inspection found three already-ingested
  recordings: one later 2026-08-05 recording and two 2026-08-06 recordings.
- The three media hashes are unique, postdate the Plan 0056 source recording,
  span at least two meeting contexts, and have zero overlap in retained Plan
  0037 and Plan 0056 JSON evidence.
- Exact paths, document IDs, transcript content, private audio, and later human
  labels remain outside the repository and will be bound only in private P0
  authority.

Implementation checkpoint:

- Added a fail-closed, content-addressed acoustic shadow-evidence projection
  for the ordinary conversation identity-review read path. Machine-readable
  evidence carries only recording-local speaker references and the exact two
  enrolled subject IDs; it remains non-authoritative and requires human review.
- Added atomic batch activation. Individual immutable bundles remain invisible
  to ordinary review reads until one complete three-document activation index
  binds every document, conversation, bundle, and execution hash.
- Added separate P0 and execution authorities. P0 cannot decode or run models;
  the execution authority permits only local decode, diarization, transcription,
  proposal generation, and read-only publication while every identity,
  assignment, profile, provider, default, and historical mutation stays false.
- Added the bounded Plan 0057 runner, complete-denominator manifest and receipt,
  deterministic replay, unchanged-state guard, private human-review HTML and
  answer template, and a hard stop for any abandoned partial execution tree.
- Added the ordinary review UI card and cache-fingerprint binding. No acoustic
  evidence is silently aligned to Assembly transcript speaker labels; this
  recording-local-label limitation is explicit and nonblocking for this shadow
  milestone.
- Delegation receipt: `not_spawned`; proactive delegation is disabled by the
  current runtime policy and this authority/integration slice is tightly
  coupled.

Closed drift ledger:

- `F0057-01`, blocking, fixed: the original P0 negative action vector could not
  authorize the runner's local model work. A separately frozen execution
  authority now grants only the bounded local/read-only actions.
- `F0057-02`, blocking, fixed: per-document publication could expose a partial
  batch. One immutable complete-batch activation index is now required before
  any bundle is review-visible.
- `F0057-03`, blocking, fixed: a crash before the final receipt could invite an
  implicit second model attempt. A nonempty partial execution tree now stops
  rather than retries.
- `F0057-04`, blocking, fixed: a rehashed activation index could drift its
  conversation binding. Review loading now validates the activation filename,
  execution hash, complete unique binding denominator, document key,
  conversation key, and bundle/execution relationship before exposure.
- `F0057-05`, nonblocking repair, fixed after execution: the review HTML's clip
  URLs were relative to the batch root rather than its `review/` directory.
  Future artifacts now use the required parent-relative URL, with a regression
  test; the frozen artifact is preserved and receives a review-only clip
  derivative for publication.
- `F0057-06`, nonblocking repair, fixed after execution: the CLI exposed batch
  replay but no distinct execution-authority replay verb. Added
  `replay-execution-authority`; the frozen authority had already replayed
  idempotently through the same public function.

Validation:

- `.venv/bin/python -m pytest -q --tb=short` passed after the review-shell
  repairs: 881 tests in 72.83 seconds.
- The Plan 0057, acoustic shadow-evidence, and transcript API focused set passed:
  81 tests.
- `.venv/bin/python -m py_compile acoustic_plan0057.py
  acoustic_shadow_evidence.py transcript_api.py` passed.
- `git diff --check` passed.

Execution state:

- Commit `e087b45189a8257c5ea9da4ca71a160255f6506e` supplied the clean,
  upstream-even execution authority.
- P0 hash `4fe89d673771af9ae51ab278a31215e07f24fb7fd1041fe20be82e3c09a90682`
  froze 3 recordings across 2 contexts with zero prior overlap and unchanged
  identity-state hash
  `64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`.
- Execution authority hash
  `42a443a1185b31e494562a060129fae03e11e0b1a800f0863352380cd256094e`
  authorized exactly one local attempt and replayed idempotently.
- Execution content hash
  `089d0213153bd001a86669141e3b7a0a72b7b7aa8638d71e3d8f8dc5c32b41e4`
  completed 3/3 recordings and 15/15 eligible speakers with zero stop reasons.
  Per-recording speaker counts are 3, 6, and 6.
- The shadow result contains 2 medium-confidence enrolled-subject proposals and
  13 abstentions. These are evidence dispositions, not human-confirmed
  identities or applied assignments.
- Three projections became visible together under activation hash
  `244ed8c07da429fa21cbb2c27a00c218c4777a124e2e67ac45b1ab0e374b9a76`.
  Replay succeeded; all three ordinary review reads report `available`, and
  identity state remains unchanged.
- Plan 0057 is `OPEN` at G1. Next: publish the private 15-card audio/transcript
  review session, collect one literal decision per card, then run P3 independent
  audit and freeze the terminal decision.

G1/P3 gate readiness:

- Published one authenticated private review session, ID `488e06d2f6da`, with
  all 15 playable cards and exact decision instructions. A fresh feedback read
  returned zero entries, so no identity decision was inferred or recorded.
- Added `acoustic_plan0057_review.py`, which accepts exactly the 15 frozen card
  IDs and four literal identity outcomes, validates the nine-unit proposal
  evidence independently of display labels, and freezes/replays a private
  complete-review receipt without applying assignments.
- Added `acoustic_plan0057_audit.py`, which independently recomputes the exact
  recording/speaker/decision denominator, proposal confirmations and
  rejections, abstention correctness, wrong and high-confidence-wrong
  dispositions, enrolled recall, proposal precision, review burden, stop
  reasons, mutation guards, and the frozen terminal rule.
- Closed-world hardening rejects rehashed mutation-bearing review or audit
  action vectors, inconsistent support-unit counts, identity-state drift,
  incomplete decisions, and non-allowlisted identities.
- Live preflight through both new interfaces validates all 15 frozen proposal
  cards and zero stop reasons without reading or inventing human decisions.
- Focused G1/P3 tests pass: 13 tests. The joined Plan 0057 and transcript API
  suite passes: 94 tests.
- Full `.venv/bin/python -m pytest -q --tb=short` passes: 894 tests in
  71.29 seconds. Python compilation and live 15-card evidence validation pass.
- Commit `5632cc1c46fde061749a4fd1e1f329695d66cf55` contains the gate
  implementation and is pushed clean/upstream-even. Both review and audit
  repository authorities bind that commit, and live execution replay still
  validates the complete 15-speaker manifest.
- A post-commit feedback read still returned zero entries. Next: wait for all
  15 literal session decisions before any review receipt, correctness score,
  or terminal decision is frozen.
- A third consecutive goal-turn read of authenticated session `488e06d2f6da`
  also returned zero feedback entries. The live batch still replays at 15/15
  covered speakers with unchanged identity state, and the branch remains clean
  and upstream-even. This satisfies the goal-level blocked threshold: no safe
  implementation, replay, audit, or documentation work can replace the 15
  literal operator decisions. Plan 0057 remains `OPEN` at G1 and resumes from
  the existing review session when those decisions arrive.

## Turn 318: Close Plan 0056 enrolled-only acoustic pilot (2026-08-06)

Summary: Completed the human-confirmed two-speaker shadow pilot, independently
recomputed its identity guard and metrics, and closed Plan 0056 at terminal
decision `plan_next_bounded_integration_milestone`.

Actions:

- Recorded Speaker 1 as neither enrolled person. The supplied non-enrolled
  name is retained only as a private review display label and did not create or
  merge an identity.
- Confirmed Speaker 2 as the proposed enrolled subject.
- Froze and replayed human-review content hash
  `6e900e6ef73520d11487840ece2ff1c40336af1e22024a4568069b64322aa399`.
- Froze independent audit hash
  `b53fb1b545b54525ea64916fb85cd274f7cb7a890c03c721a76d6a01a21c3107`
  under terminal preview hash
  `77b900f2245eaea73ea9f92a2f618a57164a139b137562f9470633f447c9d870`.
- Preserved the contact, role, relationship, and canonical-person model as a
  deferred P09 concern; Plan 0056 made no such writes.

Results:

- 2 proposals: 1 confirmed and 1 rejected.
- 1/1 enrolled speaker correctly assigned as a disposition; enrolled recall
  `1.0` and proposal precision `0.5`.
- 0 wrong assignments, 0 high-confidence wrong assignments, 1 review, and 0
  abstentions.
- 0 identity creations and 0 profile/reference mutations.
- The frozen before, after, and current identity-state snapshot hash is
  `64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`.
- No speaker assignment was applied; no provider write, default integration,
  profile learning, relationship inference, or historical reprocessing ran.

Validation:

- P0, P1 execution, human-review, and terminal-audit receipts replayed
  idempotently.
- Private directories remained `0700` and retained files `0600`.
- `.venv/bin/python -m pytest -q tests/test_acoustic_plan0056_review.py
  tests/test_acoustic_plan0056_audit.py tests/test_acoustic_plan0056_runner.py
  tests/test_acoustic_plan0056_pilot.py` passed: 15 tests.
- `.venv/bin/python -m pytest -q` passed: 859 tests.
- The active planning-contract audit returned `ok: true` with Plan 0056
  correctly excluded as closed.

Next:

- Open a separate bounded plan for the next shadow integration milestone. It
  must define its own authority and must not infer automatic assignment or
  profile learning from this pilot result.

## Turn 317: Implement Plan 0056 P0 identity guard (2026-08-05)

Summary: Implemented and validated the pre-model authority needed to run the
enrolled-only acoustic pilot without creating duplicate identities or mutating
contact, relationship, profile, reference, or provider state.

Actions:

- Added `acoustic_plan0056_pilot.py` with a small public seam for previewing,
  freezing, replaying, and portably reviewing the Plan 0056 authority.
- Enforced exact matching against the two stable enrolled subject IDs. Names,
  aliases, titles, role labels, and provider identifiers cannot be used as
  machine identities; abstentions carry no identity.
- Added read-only SQLite cardinality and generation snapshots for the primary
  transcript store, conversation-identity shadow store, acoustic profile
  registry, and biometric reference registry.
- Bound the existing nine-unit threshold application and predeclared a
  conservative assignment rule: at least six supporting units across at least
  two candidate families, with zero opposing units; every result still requires
  human review and no assignment is applied.
- Added private immutable authority receipts under a `0700` runtime tree with
  `0600` files and replay-time source hash verification.
- The first clean live-preview attempt exposed a CLI definition-order defect
  before source inspection or artifact creation; moved the entry point after
  all public definitions and reran the focused suite successfully.
- Proved that all five user-named Chris/Eric recordings already occur in prior
  acoustic evidence and therefore cannot be called fresh. Identified one
  2026-08-05 business recording as hash-fresh; it remains unfrozen pending the
  clean committed authority gate.

Validation:

- `.venv/bin/python -m pytest -q tests/test_acoustic_plan0056_pilot.py` passed:
  7 tests.
- `.venv/bin/python -m pytest -q` passed: 847 tests.
- `.venv/bin/python -m py_compile acoustic_plan0056_pilot.py
  tests/test_acoustic_plan0056_pilot.py` passed.
- `git diff --check` passed.
- The active planning audit returned `ok: true` and recognized Plan 0056 as an
  `OPEN` P10 plan.

Safety state:

- No pilot audio was decoded, transcribed, diarized, or scored.
- No pilot outcome gold was read or created.
- No person, contact, alias, role, relationship, profile, reference, speaker
  assignment, or provider record was created or mutated.

Next:

- Commit and push P0, then freeze the exact hash-fresh source and before-state
  inventory from that clean upstream-even revision before any model action.

## Turn 316: Activate Plan 0056 enrolled-only pilot (2026-08-05)

Summary: Activated the bounded Plan 0056 execution authority without running
audio decode, models, identity creation, or mutations.

Authority Consulted:

- `VISION.md`, Plan 0056, the P10 roadmap lane, Turn 315, the contact/role/
  relationship sequencing note, and the relevant runtime, memory, planning,
  validation, worktree, commit, and preview policies.
- Graphiti group `transcribe_audio_main`; current sourced facts confirmed the
  Plan 0055 planning-only terminal decision, human-confirmation requirement,
  and prohibition on autonomous learning.

Decisions And Changes:

- Changed Plan 0056 from `PLANNED` to `OPEN` and wired P10 to the active plan.
- Required the first implementation packet to freeze the exact two-subject
  allowlist, pre-execution store/profile cardinalities, prior-evidence
  exclusions, fresh source set, scoring policy, private paths, and every
  negative action before pilot audio decode or model execution.
- Preserved the scope boundary: no GWS/Odollo/receipts reconciliation,
  relationship inference, graph mutation, profile learning, provider write,
  automatic assignment, integration, or historical reprocessing.

Validation Evidence:

- Starting repository state was clean and upstream-even at `d139ba7`.
- Graphiti runtime doctor was healthy.
- CodeGraph index was healthy with 228 files, 7,025 nodes, and 23,164 edges.

State Movement:

- Plan 0056: `PLANNED` -> `OPEN`.
- No runtime pilot authority or execution artifact exists yet.

Subagent Status And Reconciliation:

- No subagent was started; one critical-path owner remains active.

Graphiti Write Status:

- No write for activation alone; the existing Plan 0056 planning episode
  remains current until a material runtime checkpoint is accepted.

Next:

- Implement and independently test the pre-execution identity guard and fresh
  pilot authority, then freeze it from a clean pushed commit before any model
  run.

## Turn 315: Contact, role, relationship, and pilot sequencing (2026-08-05)

Summary: Memorialized the canonical identity and relationship-graph contract
and separated the minimum acoustic-pilot identity guard from later P09
conversation-knowledge productization.

Action:

- Confirmed the Plan 0055 selected gold has 22 speaker rows, 11
  evaluation-only person IDs, 9 enrolled-subject bindings, and zero contact
  IDs. No contact, person, profile/reference, relationship, or integration
  mutation occurred.
- Extended the evergreen conversation-knowledge architecture to make provider
  contacts source affinities rather than canonical people, model roles as
  temporal relationships, store relationships as evidence-backed graph edges,
  support bounded multi-hop discovery, and require host validation before App
  Intelligence relationship proposals become durable observations.
- Added the dated sequencing decision in
  `docs/dev/notes/0052-2026-08-05-contact-role-relationship-sequencing.md`.
- Created Plan 0056 as `PLANNED`: an enrolled-only, human-confirmed,
  non-mutating acoustic pilot whose machine identities are restricted to the
  two existing subject IDs.
- Deferred GWS/Odollo/receipts contact reconciliation, reviewed merges and
  splits, full role/relationship graph population, App Intelligence
  relationship-inference evaluation, multi-hop retrieval, live authority
  cutover, and historical backfill to the natural P09 productization path.

Validation:

- Repo-local Plan 0055 authority and private gold were read without mutation.
- Graphiti discovery was healthy; repo-local architecture, plans, artifacts,
  and source remained operational authority.
- `git diff --check` passed.
- The repository-wide planning audit still reports 35 pre-existing structural
  issues across historical roadmap lanes, runbook ordering, and five older
  active plans. It reports no issue against Plan 0056 or the new sequencing
  note; this slice does not broaden into repairing that unrelated baseline.

Next:

- Review and activate Plan 0056, then implement only its identity guard before
  running the bounded acoustic pilot. Do not absorb the deferred P09 graph and
  cross-provider contact work into the pilot.

## Turn 314: Plan 0055 paired evaluation terminal PASS (2026-08-04)

- Closed
  `docs/dev/plans/0055-2026-08-04-generation-5-source-expanded-blind-evaluation.md`
  after the complete paired milestone and independent terminal audit.
- E2 froze the exact 22-speaker context-only prediction at
  `3bea4134ab9ebc67970c1266e9ce98d648f1453624a0c196d4a9e0b7161740d4`,
  then completed all nine acoustic matrices and 396/396 unique
  model/profile trials under matrix set
  `8b52e50baa3c3541a0bb56460c20fc39df226b83b55c2c4fb3a834fa1a016164`.
  The separately isolated augmented prediction is
  `c96bbd56cbce52c3eda352c2f3c34747f061b5d6286b91cb18ac45327962afb2`.
  Neither worker received gold or competing output.
- The direct OpenAI worker route returned HTTP 429 with no output. A reviewed
  successor authority bound exact OpenRouter model `openai/gpt-5.2`, disabled
  tools and provider fallbacks, requested no provider storage, and preserved
  the completed context prediction across two local pre-model custody defects.
- E3 revealed gold exactly once to the scoring custodian. Context-only made
  0/22 correct assignments. Voice augmentation made 6/22 correct assignments,
  including 6/9 enrolled-speaker appearances, with 0 wrong assignments, 0
  high-confidence wrong identities, 12 abstentions, and 4 reviews. It corrected
  six baseline errors and introduced none. Score
  `2aa5943aff2a7d72e1bc090347a517e3afa10df479422c0007aa372bcb309450`
  replays idempotently.
- Independent J2 recomputed the seven-recording membership, 22 speakers, nine
  matrices, all 396 trials, prediction hashes, exactly one reveal, metrics,
  privacy, permissions, negative actions, and replay, then returned PASS.
  Terminal preview
  `7a93a9e318889e061ceff7498cb147f9ee589bb1cb7fb4f12364bf5a7b9e366a`
  freezes `advance_to_limited_pilot_plan`.
- This decision authorizes only planning the next bounded pilot. Automatic
  assignment, profile/reference mutation, default integration, and historical
  reprocessing remain false. The full suite passed `834` tests before the
  terminal module; the final terminal-focused set passed 18 tests.

Next:

- Open a new bounded limited-pilot plan that preserves human confirmation and
  defines conservative use of the acoustic factor. Do not enable production
  defaults or learn from these evaluation predictions.

## Turn 313: Plan 0055 private gold and cohort frozen (2026-08-04)

- Imported 39 complete operator labels and resolved the one remaining card,
  Candidate 8 / Speaker C, as context-derived `Mark Mba-Wright`. The frozen
  transcript directly addresses that speaker as Mark; later project evidence
  and Iowa State's faculty record align the LCA researcher role with Mark
  Mba-Wright. No acoustic identity model assisted gold creation.
- Canonicalized only explicit aliases and misspellings: Jeffrey Dikis,
  Dr. Dikis' Nurse, and Alexandra Hoen. The immutable population proposal is
  `9cd1a5c41920de2f0dc562c868268c6eaa9091be9cd7e88794969e460858f971`
  with manifest
  `b4cadac5f76d3279f9c48ae7559fc37ab3071be043dcc889c8a92c7b6b21cde5`.
- The first permitted combination passed immediately: Required A, Required B,
  and Candidates 3–7. It has seven distinct recordings/conversations, 11
  people, 25 same-person session pairs, both enrolled people in at least two
  recordings, and zero overlap.
- Independent J1 recomputed the complete 12-case/40-label denominator,
  selection ordering, population gates, private modes, and replay, then
  returned PASS. J1 freeze preview
  `b0c642d5989df72e876abbbf10427148e72c1cf3b2c8fac69eaf90e5062ff3a3`
  and private-gold manifest
  `617b98be57f28770e1b22ecaaf29568518806c73b0906c4c3abd1f84493c0aac`
  are applied.
- Gold remains unrevealed. Models, predictions, profile/reference mutation,
  integration, and historical reprocessing remain false. Current gate: prepare
  isolated gold-blind context-only and voice-augmented worker inputs before
  either worker runs.

## Turn 312: Plan 0055 S0 accepted and S1 review opened (2026-08-04)

- S0 froze the two required and ten deterministic additional recordings under
  ordered source-set hash
  `a66ba8bc5d7358bf9b831ff08d07707e87be8ea8973e08b252a1db940db19733`.
  The comprehensive exclusion union covered 2,648 prior JSON artifacts and
  3,017 evidence hashes. Independent J0 accepted the repaired source packet.
- The first S0 apply exposed that `copy2` retained the SMB source mode. The
  exact private Zoom copy was corrected to `0600`, the importer now enforces
  that mode, and immutable preview
  `7e2a99d8957b3e952c45454ac13fd4033f0b004e258c1700446f93a7b79c8f07`
  replays idempotently with byte identity intact.
- S1 transcribed and diarized only the twelve frozen candidates. Provider job
  records and full results remain private and source-hash-bound. No acoustic
  identity model, gold access, profile/reference mutation, integration, or
  historical reprocessing ran.
- Materialized 40 non-empty private listening clips and a browser review page
  under preview
  `5a3f9fc9848a5e0b669bc37796e5a55b4f9dcd7bf0f55609aefa886e4caabcf9`.
  All 40 HTTP players returned 200 with non-empty content; the full suite
  passed `815 passed in 59.39s`.
- Current gate: the operator must identify all 40 speaker cards and confirm
  Chris Williams and Eric Cochran in both Required A and Required B. Then J1
  can independently select and freeze the first population-valid cohort.

## Turn 311: Open Plan 0055 source-expanded blind successor (2026-08-04)

- Resolved the operator's exact Zoom path through the existing bastion-mounted
  SyncThing route after the local Cloud disk was unavailable. Metadata-only
  inspection found a 110-minute mono AAC stream plus a separate audio-only
  artifact.
- Bound the audio-only artifact, its parent MP4, and one separately located
  unused archived recording by SHA-256. None of the three hashes or source
  identities appears in current repo or private runtime manifests.
- The operator states that both recordings contain the two enrolled speakers.
  Plan 0055 treats that as pre-model stratified candidate authority, not gold;
  private listening review must confirm both people in both recordings.
- Opened Plan 0055 to freeze those exact required sources plus a deterministic,
  bounded, prior-disjoint archive expansion before transcription or decode.
  The new plan excludes every Plan 0054 candidate and retains the full blind
  paired-evaluation milestone, worker isolation, one reveal, and all negative
  mutation actions.
- Current gate: commit and independently accept S0 source authority before
  copying or decoding content, transcription, diarization, review, model work,
  or prediction.

## Turn 310: Plan 0054 E1 population-infeasible stop (2026-08-04)

- Imported the complete 29-speaker private operator review. The operator then
  narrowed the one initially unresolved organization representative to the
  named meeting facilitator, superseding the provisional alias packet with a
  new content-addressed corrected packet.
- Recomputed every exact seven-recording subset in frozen order. All 330
  combinations failed only `both_enrolled_people_have_two_recordings`; the
  maximum enrolled count meeting two-recording coverage was one. The other six
  population gates were satisfiable.
- Applied immutable E1 stop preview
  `e72b56b061af7e3acaf34e7ce03fda26c4d7f2da68b79b68af68eb8e9e2ce1ac`
  and manifest
  `64ad7fee943da484b015ba25ceb56509acb1da129f47e441cd48c56e7581369a`;
  exact replay is idempotent.
- Plan 0054 closes at `population_infeasible_stop`. Cohort/gold freeze, J3,
  models, predictions, profile/reference mutation, integration, and historical
  reprocessing remain false.
- A read-only availability check found a larger nested corpus inside the
  already authorized recordings folder and at least one previously unused
  recording contextually associated with both enrolled people. That is only a
  successor lead: the current plan permits top-level media only, and a valid
  successor still needs prior-source exclusion, deterministic enumeration,
  transcript authority, and evidence of a second prior-disjoint recording for
  the underrepresented enrolled person before any new review or model work.

## Turn 309: Plan 0054 recovery PASS and E1 private review (2026-08-03)

- R0 froze a fresh eight-record diagnostic membership; J0 accepted the exact
  selection, exclusions, reason contract, tool binding, and privacy boundary.
- The one-pass R2 run passed all 7 fresh positives and rejected all 11 recovery
  negatives for their literal predeclared reasons. Independent J2 recomputed
  the full denominator and returned `PASS`.
- E1-only acceptance preview
  `52321890681eb56a5ee515aae5abcf708984ac7fa80f0e5886953d7a480b7a54`
  and manifest
  `2ee51c793fa8ac349e5ab41a50841c19bd311419ff8e04dbf242d38c042c30e8`
  replay idempotently.
- E1 enumerated all 12 remaining fresh ordered records. One failed private-gold
  feasibility because it had no usable speech utterance; 11 records remain,
  represented by 29 private listening cards. Preview
  `eab13de1ac89ffafef8dca228368ddbe792341b9796d4a0f503ce3d5405dd6e1`
  and manifest
  `63e62c3199c6e2c02688ff365fc29c9926f40202fe03c8c49b1a2646912f2700`
  replay idempotently.
- Current gate: complete the private operator identity review, then select the
  first seven-recording combination passing every population rule and submit
  the frozen cohort/gold feasibility packet to J3. No model, prediction,
  profile/reference mutation, integration, or reprocessing action ran.
- Commit `20516a0` prepares that post-review step without consuming answers:
  it requires all 29 labels, rejects uncertain placeholders, binds identities
  to exact cards and the two prior enrolled subjects, checks every exact
  seven-recording combination in frozen order, and emits only a J3 proposal.
  Full validation passed with `808 passed`; no gold packet exists yet.

## Turn 308: Open Plan 0054 fresh-holdout recovery (2026-08-03)

- Opened
  `docs/dev/plans/0054-2026-08-03-generation-5-fresh-holdout-recovery-and-blind-evaluation.md`
  after Plan 0053's mandatory J2 STOP.
- The successor retains the J1-accepted sample-preservation rule but cannot
  reuse Plan 0053 development, positive-holdout, or negative media.
- R0 must freeze a comprehensive exclusion union, a deterministic new
  seven-source holdout proposal, the separately seeded recovery negatives, and
  literal expected reasons before J0 can authorize any new decode.
- The end state remains the full new blind context-only versus voice-augmented
  evaluation; passing recovery validation alone is not milestone completion.

## Turn 307: Plan 0053 J1 pass, G2 evidence, and J2 terminal STOP (2026-08-03)

- The first J1 review stopped on an exact one-packet discontinuity boundary and
  two weak adversarial implementations. A new G1 packet repaired the comparator,
  exercised the exact boundary, removed a real compressed packet, invoked the
  validator for wrong-stream input, and retained full private evidence.
- Fresh J1 review passed preview
  `3e66a93feb5a826680025135c6b60e0e541ed724a639146c1df126f403242919`;
  manifest
  `4d5ad0f08ba14ca20bd489530cc8919ab0d2defa9bb0dfaecfba5a761a303f0f`.
  G2-only acceptance preview is
  `1ccb0b9b747760b6202827f849d2575956a67ad03dd3e374d169b1777292eeda`.
- G2 measured its holdout once: 7/7 positive recordings passed and 11/11
  adversaries appeared rejected. Preview
  `fc0d0dca9eec248df2e8dccdd262a3062ca2e5deca03aaa99799e4253e647c83`
  and manifest
  `c59771c6d6054be0384956beaa7a7908dca1fe6cbdffb93bf8fe2d00469c69f2`
  replay without re-decoding.
- Independent J2 reproduced 7/7 positives and ten fixed negatives, but found
  `corrupt_source_tail` copied the observed reason into `expected_reason`.
  The negative denominator was therefore circular and J2 returned `STOP`.
- Plan 0053 is closed. No candidate, gold, model, prediction, mutation,
  integration, or reprocessing action ran. The code now predeclares
  `measurement_error`, but the revealed Plan 0053 holdout cannot be reused.
- The terminal result is now immutable: J2 STOP preview
  `058a36ebebc5a9b743f6db6d856ea5cc2e0c0e123bc8e936bbd1e8597cf5fb3e`
  and manifest
  `4033452c29812527209b478b5218b9e773e76726deeffc42a004183b78a994a5`
  replay full-body with every downstream action false.

## Turn 306: Generation-5 development diagnosis measured (2026-08-03)

- Ran the committed content-preservation validator from `3d8071d` against only
  the five G0 development members; the seven positive-holdout members remained
  unmeasured.
- All five inputs reconciled `packet_count * 1024` to native decoded samples,
  native samples to the mathematically expected 16 kHz frame count with zero
  error, and reference PCM to production PCM by exact fingerprint.
- The Generation-3 source was rejected for one 4,311,372-tick timestamp
  discontinuity. The Plan 0052 source passed with no discontinuity or content
  loss, proving that its approximately 0.174-second difference was container
  clock cadence rather than truncation.
- The complete suite passed: `770 passed in 46.13s`.
- G1 is not frozen and J1 has not accepted the contract. Do not measure the
  seven-member positive holdout or instantiate the held-out negative family.

`RUNBOOK.md` is the dated execution log for this repo. Use it to record policy adoption, roadmap changes, implementation slices, validation evidence, and operational incidents that should survive chat history.

## Turn 305 | 2026-08-03

Plan:
`docs/dev/plans/0053-2026-08-03-generation-5-duration-validation-and-blind-evaluation.md`,
version 1

Packet: P10-G5-G0 | Sealed duration-diagnostic authority

State transition: `OPEN/Plan-0053-G0 -> OPEN/Plan-0053-G1`.

- Replayed the immutable Generation-3 preparation STOP and Plan 0052 terminal
  STOP before deriving successor authority.
- Frozen G0 preview
  `5f765a67810bc4cb58c9c3a8d78aaa25aba4a67650e4fd242d456b9a54d55096`
  and `0600` private manifest
  `4c85505c18b12d6acf939d1cbe2dfa1f5d1e37de03fba40c99fd1b86d37dd818`
  replay idempotently.
- Development contains two known failures plus the three lowest-hash Plan 0051
  qualified controls under set hash
  `349079a147343b5653a4b1e1454392b57f4b70a0b80a8dbc3cce4edb53eb1944`.
  The other seven Plan 0051-qualified sources remain sealed and unmeasured
  under holdout-set hash
  `696bde15e5a1d5f006fb596dee49d55c704c1d2e61ae9e2b8bd544413f29ccae`.
- G0 performed no decode. Only G1 development diagnosis is true; holdout
  measurement, candidate enumeration, predictions, gold reveal, biometric
  models, scoring, mutation, integration, and reprocessing remain false.

Next:

- Diagnose development media only, implement the non-circular packet/decode-
  to-null/output/content validator and development negative family, then stop
  at independent J1 before any holdout access.

## Turn 304 | 2026-08-03

Plan:
`docs/dev/plans/0053-2026-08-03-generation-5-duration-validation-and-blind-evaluation.md`,
version 1

Packet: P10-G5-PLAN | Fresh diagnostic and full blind-evaluation authority

State transition: `CLOSED/Plan-0052-stop -> OPEN/Plan-0053-G0`.

- Opened a fresh successor rather than modifying Plan 0052. The new plan must
  explain the `0.1739795`-second Generation-4 discrepancy and distinguish it
  from the `89.776791`-second Generation-3 loss.
- The diagnostic design freezes a five-source development set containing both
  known failures and the three lowest-hash Plan 0051 healthy controls, plus a
  disjoint holdout containing the other seven Plan 0051-qualified sources.
  The replacement rule must be based on sample/timestamp/resampling semantics,
  not an observed maximum or a constant chosen to admit a named case.
- Diagnostic media are permanently excluded from Generation-5 evaluation.
  After independent J1/J2 acceptance only, the plan may enumerate one bounded
  oldest-forward fresh pool and carry a population-valid cohort through blind
  context-only and separately voice-augmented predictions, one reveal, full
  acoustic scoring, independent audit, and one terminal decision.
- Automatic assignment, enrollment, profile/reference mutation, default
  integration, and historical reprocessing remain unauthorized.
- Fresh independent reviewer `/root/g5_plan_review` first returned `STOP` on
  acoustic ordering, incomplete prior-evaluation exclusions, cohort
  hand-selection risk, weak held-out negatives, circular duration evidence,
  undefined gold/worker isolation, and ambiguous milestone success. The plan
  now freezes gold-blind acoustic scores before augmented prediction, complete
  exclusions, deterministic cohort construction, disjoint seeded negative
  families at the derived boundary, non-circular packet/decode/output/content
  evidence, isolated stateless paired workers, and separate administrative
  closure versus product success. The re-review returned `PASS` for design
  only; G0 and later execution remain unproven.

Next:

- Execute G0: replay current authorities, derive exact diagnostic development
  and holdout membership without decoding, freeze the measurements and
  negative actions, then begin development-only diagnosis.

## Turn 303 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-J1-G2-G3-G6 | Authority repair, pre-model freeze, and terminal preparation stop

State transition: `OPEN/J1-independent-design-reconciliation -> CLOSED/G6-stop`.

- Independent J1 first returned bounded `STOP`: G1A accepted caller-injected
  authority and membership, so replay proved caller self-consistency rather
  than exact source lineage. The one allowed rework removed those public
  inputs, pinned and validated original and supplemental manifests, recomputed
  their exact disjoint 22-source union, bound every member to its authority
  origin, and added manifest, set-hash, cap, origin, replay-broadening, and
  transcript-drift tests. Commit `0f793fb` is pushed.
- Replacement G1A preview
  `9648201f4d3e70f65d396bb1fe82fb9aad57603ff077ccd99b7d61532a0889d7`
  and `0600` manifest
  `fed08c49b26024b041774b9df7b067ae9d156022fd8a7a16dbf0df8b85451c0f`
  replay idempotently. The exact authority is 10 original plus 12
  supplemental sources with zero intersection and combined set
  `460fa3dd3befa17e249860b70477474580202577a599a357e7b7c641609cd4c2`.
  Independent J1 resubmission signed design acceptance and authorized only G2.
- G2 froze the exact seven-recording cohort, private-gold commitment,
  population proof, selected acoustic factor, full nine-unit contract,
  contextual contract, metrics, negative actions, and terminal policy at
  commit `473a67c`. Replay authority was subsequently repaired at `dc069a6`
  without changing any G2 semantics: the generator now binds its last module
  commit instead of moving `HEAD`. Renewed preview
  `6d6e86094c809c34c45694c311063c06570020348eccd6f65a420535167e3d41`
  and private manifest
  `20cb311ebf436ffdd382c4715de2987d854d1d5b5c56974739d1c4f94c96ae61`
  replay idempotently. Gold reveal and biometric models remained false.
- G3 ran one frozen prediction-blind preparation attempt. Three cases
  completed and replayed all 15 P1/P2 method cells. The fourth source probed
  at `5138.648667` seconds and decoded at `5138.4746875` seconds. Drift
  `0.17397950000031415` exceeded the immutable `0.05`-second P1 tolerance.
  The remaining three cases were not attempted after the hard gate failed.
- G6 mechanically applied terminal precedence one. Commit `ea058de` preserves
  the original G3 execution authority while binding the renewed,
  semantically-identical G2 receipt. It freezes terminal preview
  `2f7f228189072dfb90344c916c2e104d0d4836ea613cd0f081f7e9109e33fc17`,
  failure evidence
  `53915d76835849db729b380095e2043cc23af8ee3f03aece1ac2d3bc9793b1ed`,
  and `0600` private manifest
  `7600629721bfcedcf3e6a1f708164fe4600441f8241874873a473e00080fe702`.
  Full-body replay returns terminal decision `stop`.
- The full suite passed `764` tests. No contextual or augmented prediction
  turn, gold reveal, biometric execution, acoustic score, profile/reference
  mutation, default integration, or historical reprocessing occurred.

Next:

- Plan 0052 is terminal. Any new evaluation generation requires a fresh plan
  and authority; do not retry, substitute the cohort, or relax the frozen P1
  recipe under this plan.

## Turn 302 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-PASS | Complete supplemental import and passing cohort proposal

State transition: `OPEN/G1A-supplemental-second-session-review -> OPEN/J1-independent-design-reconciliation`.

- Imported the final three-label supplemental response under private packet
  content
  `d1575a9bcdca79f7cb1193fb936382bdc5a4153ae11bb16aaad43c5e6ba3d64d`;
  its file hash is
  `a6cf9837101fe24049635461db641c7d25af102e9884e5c573fa35ef5c2c5c8d`.
  Combined 12-case private GOLD_SCHEMA content is
  `37f3a2da83cdbecaa936fbad477490c234c3d25ba55243f68a413f06dee4557a`;
  its `0600` file hash is
  `5b43119baddb24c794ebfe3224b735d182bc7b10e374166d71f68c9ea61ef65d`.
- The full population passes every gate: 12 recordings, 15 people, 47
  same-person session pairs, both enrolled people in two independent sessions,
  complete gold, zero overlap, and all sources within authority. The proposed
  seven-recording cohort also passes with nine people and 17 same-person
  session pairs.
- Sealed combined qualified-set hash
  `460fa3dd3befa17e249860b70477474580202577a599a357e7b7c641609cd4c2`,
  G1A preview hash
  `1bd697995ebfcfa3d8018f77b99c4a1c28e82ce4df32b86dc514cc8d3557ee76`,
  and replay-idempotent private manifest hash
  `08d134ada150775e5da64b3dc18ab5a59a8ad25bf90524e738257d80042d7a98`.
- No acoustic model, prediction, private-gold reveal, cohort/gold freeze,
  profile mutation, new supplemental candidate, or retained decoded audio
  occurred. Only submission to independent J1 is now true.

Next:

- Run independent J1 reconciliation of the completed G1A, G1B, and G1C
  contracts. Do not authorize G2 freeze unless J1 accepts the exact combined
  source authority, private-gold isolation, calibration-only acoustic factor,
  and negative action vector.

## Turn 301 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-SUPPLEMENT-REVIEW | First supplemental import and narrowed second tranche

State transition: `OPEN/G1A-supplemental-review-required -> OPEN/G1A-supplemental-second-session-review`.

- Imported the first two-conversation, seven-label supplemental response under
  private packet content
  `a19d7c59b74a5760196ee18a599d04f94e762a77ce1d3c6cbc643c992a0b3005`.
  Combined 11-case private GOLD_SCHEMA content is
  `baf0fa93dee6b936564fa3827a735e9ba10304cd4d2f3b418b81a17300bdc6ca`;
  its `0600` file hash is
  `9a3f9d1157be49ca74ffaaa9fa7b28617a8bb50ae9bd711909dcebf961bd5be2`.
- Eleven conversations now represent 15 people and 37 same-person session
  pairs with complete gold, zero overlap, and all sources inside the combined
  original-plus-supplemental authority. The only failing gate remains the
  second enrolled person's coverage, which improved from zero to one session.
- Inspected only transcript context inside the already frozen supplemental
  pool and selected one independent three-speaker recording that explicitly
  introduces the missing enrolled person into the call. Generated review plan
  `1387295c900b32f92c11af1c3733af2aa50abae79ad1615e694d90e7cf1de2d0`
  and published preview session `f8a5bca4e1d0`.
- No acoustic model, prediction, private-gold reveal, cohort/gold freeze,
  profile mutation, new supplemental candidate, or retained decoded audio
  occurred. J1 remains false.

Next:

- Operator returns the three labels by copying the entire filled page. Import
  them immutably and re-run the combined G1A population gate; submit to J1 only
  if the second enrolled person reaches two independent sessions.

## Turn 300 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-SUPPLEMENT | Complete original gold and consume bounded supplemental pool

State transition: `OPEN/G1A-original-remainder-review-required -> OPEN/G1A-supplemental-review-required`.

- Imported all five remaining original-pool labels, applying the operator's
  immediate spelling correction before the first immutable write. Combined
  nine-case private GOLD_SCHEMA content is
  `2876ae8dacba1311968386abab9b0adfa325dce35f8ca719250c4ebc1a925b6a`;
  its `0600` file hash is
  `eab00331c61182f4a86d1d12dfbdff4bdfb3cab81f1bc62fac3d39faec6e49d9`.
- Complete-original-pool evaluation has nine conversations, ten people, 22
  same-person session pairs, complete gold, zero overlap, and all sources
  inside authority. It still fails only the second enrolled person's
  two-session requirement. G1A supplemental-request preview is
  `96b62b5254917f958440efa7c2b239c96f3596cb9cd12d6744b8a4d15c91e281`.
- Read-only calendar reconciliation found one strong eligible meeting but did
  not by itself prove a second independent session. Consumed the one authorized
  12-candidate supplemental pool using context-ranked, prior-disjoint sources.
  All 12 qualified under preview
  `cc405f40414f69bea012559d5ca4c10098ed4ab0d4e4efc37264a361c26f82d9`,
  manifest
  `c34a4ebd2d78fef8193aec18f15c97146f06f99c32d2c29d81066719954ab677`,
  and qualified-set hash
  `09ae99141880df95b3531563b484008ea411ccafab411b4da311627d5e16d994`.
- Generated a first two-conversation, seven-label supplemental review under
  plan hash
  `ec029c56462ea749b4d801a8347414c0e2e008cf5cfb75a464cc483cff035f21`
  and published preview session `1979c73bd128`. Full-page copied text is now an
  explicit supported fallback because clipboard JavaScript remains unreliable.
- Supplemental page support and fallback guidance pass 9 focused tests and the
  full 754-test suite in 78.29 seconds. Commit `ab885d7` is pushed and
  upstream-even.
- No acoustic model, prediction, private-gold reveal, cohort/gold freeze,
  profile mutation, or retained decoded audio occurred. J1 remains false.

Next:

- Operator reviews the seven supplemental labels and returns either the answer
  block or the full copied page. Import immutably and re-run G1A; continue
  within the already frozen supplemental pool only if enrolled-session coverage
  is still incomplete.

## Turn 299 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-GOLD | Complete gold import and original-pool remainder gate

State transition: `OPEN/G1A-corrected-review-awaiting-answers -> OPEN/G1A-original-remainder-review-required`.

- Normalized the operator's copied page text into the exact 12-reference answer
  block. Both later immutable corrections took precedence over stale values in
  the copied original-page text.
- Applied private, unfrozen GOLD_SCHEMA content
  `4534fbb173c5ae8d261d3df6783e3f8908d48ef514979dff800bb24db6af7534`;
  its `0600` private file hash is
  `02624aec6bca8106a18f508b57bcfd7f3250e0760a36cc8565ecaca0f813022a`.
- Population evaluation passed six gates with seven conversations, seven
  people, 11 same-person session pairs, complete gold, zero overlap, and all
  sources inside authority. It failed only because one of the two enrolled
  people has the required two sessions; the other has none in reviewed gold.
- Because two transcript-linked original-pool recordings remain unreviewed,
  the bounded supplemental pool is not yet available or consumed. Generated a
  five-label remainder review under plan hash
  `eef4a765aaf1c95979a0df1225c3d05365f60054450eebae4e00956660f92e5a`
  and published private preview session `7aecb6685519`.
- Fixed the browser copy action so clipboard denial reveals a selectable answer
  block and attempts the legacy copy fallback. Focused tests pass 9 tests; the
  full suite passes 754 tests in 65.55 seconds. Commit `a4d1748` is pushed and
  upstream-even.
- No acoustic model, prediction, private-gold reveal, cohort/gold freeze,
  profile mutation, or supplemental-media use occurred. J1 remains false.

Next:

- Operator reviews the five labels in the remaining two original-pool
  recordings and returns the copied or manually selected answer block. Merge
  those labels immutably with the existing private gold and re-run G1A. Request
  the one bounded supplemental pool only if the complete original pool still
  cannot satisfy enrolled-session coverage.

## Turn 298 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-CORRECTION | Immutable operator label corrections

State transition: `OPEN/G1A-import-ready-awaiting-answers -> OPEN/G1A-corrected-review-awaiting-answers`.

- Corrected both prior operator assertions without mutating the original gap
  packet. Immutable correction content hashes are
  `c0e2c19a9d61f5e20f43e59c1a88804f673f68b05515c0a59204820832ecc178`
  and
  `66a3a684fe6e15f85fac50f16aaf2e9d7fdd202cfc1b0d0c13705d10b8ab56be`.
- Added fail-closed correction validation over the base gap hash, speaker
  reference and label, prior identity hash, and correction content hash.
- Corrected private review bundle
  `c826f642f47e915ff3afd5476615e342f53b8ff5981b8a6ec0ddac1b641d3d2d`
  retains 12 private clips, two corrected prefills, ten blanks, and the two
  enrolled-identity hints. One corrected prefill matches an enrolled identity.
- Published corrected private preview session `d5a26f851bfe`; the previous
  preview is superseded. No raw share-link token is persisted here.
- Focused correction tests pass, and the full suite passes 754 tests in
  61.75 seconds. Commit `dc0c6c5` is pushed and upstream-even.

Next:

- Operator completes the ten blank labels in the corrected preview, uses
  **Copy all answers**, and returns the copied block. Import only that complete
  block, then run the G1A population gates before any J1 action.

## Turn 297 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-IMPORT | Fail-closed private-gold answer importer

State transition: `OPEN/G1A-operator-review-ready -> OPEN/G1A-import-ready-awaiting-answers`.

- Added exact parsing for the review page's copied answer block. Every speaker
  reference must appear once; blanks, unknowns, missing lines, duplicates,
  unknown references, and control-bearing values fail before any write.
- Repeated normalized names or stable aliases map to one opaque person ID.
  Conversation and recording IDs are deterministic and distinct from private
  source paths.
- Added exact loading of the two enrolled identity bindings from immutable
  Generation-3 private gold manifest
  `5e91c62985d137ca64689e6cd49872b92ebce1051689d62f43e32d000824495e`.
  The loader validates private mode, manifest hash/schema/status, name hashes,
  and opaque person references.
- A live read proved the two previously supplied case identities are not the
  two enrolled profiles. Review bundle
  `09a4718d04e0d931daad9e8bcb38e46b0fa524b9c53cd88797473202bafbc3ec`
  therefore shows both enrolled names only inside the private local page as
  people to look for. It retains 12 labels, two prefilled answers, ten blanks,
  12 private clips, and no external publication.
- A complete answer block can now create one content-addressed `0600` private
  GOLD_SCHEMA packet marked operator-review-complete but explicitly not frozen.
  It contains no transcript text, audio, acoustic score, model execution, or
  prediction-worker reveal.
- Focused tests cover passing population construction, repeated-person
  identity, both enrolled identities, four same-person session pairs, stale
  enrollment authority, incomplete answers, immutable apply, and modes. The
  full suite passes 753 tests. Commit `3b2b433` is pushed and upstream-even.
- Graphiti runtime doctor reported MCP HTTP degraded while direct read
  discovery succeeded. Discovery had no authoritative Plan-0052 episode and
  returned a stale Plan-0025 status; current repo/runtime evidence prevailed.

Next:

- Operator completes and copies the ten remaining page answers. Import them,
  run G1A population evaluation, and continue only if both enrolled identities
  have two sessions, at least five people and four same-person pairs exist, and
  all overlap/completeness gates pass. Otherwise use the one bounded
  replacement/supplement decision without starting J1.

## Turn 296 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1A-REVIEW | Private operator listening gate

State transition: `OPEN/G1A-private-gold-review-required -> OPEN/G1A-operator-review-ready`.

- Conservative transcript/calendar review supported 2 of 15 labels in the
  original proposed subset. It created no gold authority and correctly left
  population passing false.
- A bounded two-for-two substitution within the transcript-linked original
  pool preserves seven recordings, zero source/derivative overlap, and
  structural people/session-pair capacity while reducing unresolved labels
  from 13 to 10. Supplemental media remains unconsumed and unnecessary.
- Added a content-addressed private review-bundle generator. It produces one
  local HTML sheet, short per-label audio excerpts, bounded transcript clues,
  and a copyable batch response. It does not run acoustic models or expose
  predictions/scores.
- Production bundle
  `41294254962762e700af78273c4367c59a02718c2e91be1632ab1330a4e34f58`
  contains seven cases and 12 labels; the two prior operator assertions are
  prefilled, leaving ten manual labels. All page, receipt, and clip artifacts
  are `0600`.
- The page was intentionally kept out of the external Previews service because
  it contains private transcript clues and audio excerpts. A local browser
  smoke proved the page, 12 audio controls, 12 inputs, and prefilled labels
  render. All 12 clips replay as positive 25-second WAV excerpts.
- Implementation commits `7a5a6e5`, `3b0ff5d`, and `ed6d23d` are pushed and
  upstream-even. The full suite passed 748 tests before the two focused
  production corrections; four focused tests, compilation, and diff checks
  pass afterward.

Next:

- Operator completes the ten remaining labels in the single private review
  page and pastes its copied response. G1A then builds and validates the exact
  private GOLD_SCHEMA packet. Do not run J1 or consume supplemental media
  before that result.

## Turn 295 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1AB | Feasibility and acoustic-contract checkpoint

State transition: `OPEN/G1AB-running -> OPEN/G1A-private-gold-review-required`.

- G1A now proposes seven original-pool recordings from nine transcript-linked
  qualified recordings, with 15 speaker slots and theoretical capacity for 21
  pairs. This is a feasibility result, not a frozen cohort or gold set.
- G1A preview
  `b59c9f6e665f8ff238362b411ed7317764d5a62fae21016983201601b6ec2af3`
  and manifest
  `e5dea7c7d0ede0478ce97c5152e3e2e15122cbdc0aa57001c0cb4e51bc6bc487`
  replay exactly. Runtime directories are `0700` and files are `0600`.
- The supplemental-pool option remains unconsumed. G1A authorizes only the
  private gold review and exposes no portable paths, membership, or gold.
- G1B replayed the persisted calibration authority without audio, model load,
  Generation-4 gold, or holdout access. Acoustic-contract hash is
  `eae21ec7842803a8cf6aa695b5146927ee9da33e2133ab542cd446fcdc039aab`;
  selected opaque factor contract is
  `4cebdb5140cae4c592d99622447b39ae60d04e428d06e376094685b20a886a54`.
- G1B retained all nine factor units and enables only J1 design
  reconciliation. Its returned evidence hash is
  `66d9c3385d8a1cdd02779225787ab42e236a32c54197914a8e41b4881197e1ef`.
- The two bounded workers reached their declared terminals and the primary
  reconciled both outputs. G1C remains complete. J1 and G2 through G6 remain
  false pending passing private-gold population evidence.
- The committed implementation authority is `71b7ce8`; the full suite passed
  745 tests before the production replays recorded here.

Next:

- Complete the isolated private gold review for the proposed seven-recording
  subset, then either publish a passing population proposal or consume the one
  supplemental-pool option. Do not start J1 until G1A passes.

## Turn 294 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G1C | Contextual visibility contract

State transition: `OPEN/G1C-authorized -> OPEN/G1C-complete-awaiting-J1`.

- Preserved the closed Plan 0025 two-phase clue-discovery, host-retrieval, and
  identity-evaluation workflow while adding a path-free shadow contract.
- Froze exactly two prediction families, one prompt hash
  `4afbdde84707b2cf2308535a8b4e01b1878c7f3065f2f954dc9180bfc01f5584`,
  and one rubric hash
  `a624429790a6a4868c295d3188a033157f9314bdeeedb21d3bf794c44e8370cc`.
- Froze recording-start temporal filtering, context-first stable candidate
  union, separate cited acoustic factors, neutral missing acoustic evidence,
  visible conflicts, and distinct context versus union candidate recall.
- Contract hash is
  `f539146dfccc3a8025d20713b5cf02762d7d5a5d25cb01f4886f6dedda44bb18`.
  It contains no paths, private membership, transcript text, audio, embeddings,
  thresholds, or biometric scores and did not send a model turn.
- G1C delegation is `not_spawned`: the primary owned this disjoint lane while
  G1A and G1B occupied the other two active-agent slots, preserving the
  campaign concurrency cap.
- Validation passed 34 focused tests, compilation, and `git diff --check`.
  Only submission to J1 is true; all later execution and mutation actions are
  false.

Next:

- Await G1A and G1B, reconcile their exact runtime and code evidence, then run
  independent J1 design review before any G2 freeze.

## Turn 293 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G0 | Production campaign checkpoint

State transition: `OPEN/G0-implementation-prepared -> OPEN/G0-complete`.

- Clean pushed implementation commit `5117e7e` passed 24 focused tests, 725
  full repository tests, compilation, diff validation, and the active planning
  audit.
- Production no-write preview hash is
  `aa179741e735247e87cc6143c6526669670734c8c562ed166160eb0c6d605010`.
  It binds the exact Plan 0051 media hashes, six-profile and model-asset sets,
  score matrix, nine-threshold set, pinned runtimes, plan, repository, privacy
  flags, delegation decision, and negative actions.
- Immutable G0 manifest hash is
  `ad9e26b59502508c8810e11648d519d99860579aea1ca731445459b196836d22`.
  Full replay independently re-decoded all frozen sources and recomputed the
  thresholds without gold, retained audio, or model execution.
- Runtime directories are `0700`; manifest and receipt are `0600`, regular,
  and not symlinks.
- Only G1A cohort/gold feasibility, G1B acoustic evidence contract, and G1C
  contextual visibility contract are true. J1 through G6, reveal, profile
  mutation, integration, and historical reprocessing remain false.

Next:

- Spawn only the three named, disjoint G1 lanes with bounded write surfaces
  and runtime handles. Reconcile them at J1 before any G2 authority.

## Turn 292 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G0 | Authority replay implementation checkpoint

State transition: `OPEN/G0-not-started -> OPEN/G0-implementation-prepared`.

- Graphiti discovery was healthy and returned 10 facts, five nodes, and five
  episodes from `transcribe_audio_main`. Its Plan 0025 active-state wording is
  stale; the current repo's CLOSED plan remains authoritative.
- Replayed the exact Plan 0051 authority full-body with source re-decode and no
  retained audio. All counts, reason codes, media hashes, privacy flags, and
  the negative action vector match the frozen receipt.
- Confirmed the first inherited calibration replay failure came from invoking
  system Python, where the pinned biometric distributions are absent. The
  repository `.venv` has exact SpeechBrain 1.1.0 and ONNX Runtime 1.24.4; the
  complete nine-threshold replay then passed without audio or model execution.
- Added a portable G0 preview/apply/replay authority that binds the plan,
  repository, qualified media, six-profile set, model assets, score matrix,
  nine-threshold set, pinned runtimes, privacy flags, and negative actions.
- Reconciled Plan 0050 from stale `OPEN` to `CLOSED`; its independently audited
  Generation-3 media-integrity `STOP` remains unchanged and non-retryable.
- G0 delegation receipt is `not_spawned`: this is the primary-owned critical
  authority path. The three named G1 design lanes remain unauthorized until
  the production G0 checkpoint is applied and replayed.

Next:

- Validate, commit, and push the G0 implementation; then run its production
  no-write preview, immutable apply, permission check, and exact replay.

## Turn 291 | 2026-08-03

Plan:
`docs/dev/plans/0052-2026-08-03-generation-4-shadow-speaker-identity-milestone.md`,
version 1

Packet: P10-G4-G0 | Campaign plan opening

State transition: `PLANNED/Generation-4-cohort-design ->
OPEN/Generation-4-shadow-speaker-identity-milestone`.

- Opened a `/goal`-compatible campaign plan whose finish line is one
  immutable product decision, not implementation, cohort freeze, acoustic
  execution, a metric report, or passing tests.
- Bound the campaign to the Plan 0051 10-recording qualified-media authority,
  six active profiles, nine frozen calibration thresholds, and the closed
  Plan 0025 contextual speaker-clue workflow.
- Froze the product comparison as paired context-only and
  context-plus-separately-visible-acoustic predictions on one unseen cohort,
  with both prediction families immutable before one gold reveal.
- Defined G1A cohort/gold feasibility, G1B acoustic evidence contract, and G1C
  contextual visibility contract as the only parallel lanes. G2 through G6
  form the sequential critical path, with independent reconciliation at J1,
  J2, and J3.
- Kept automatic assignment, enrollment, profile mutation, default
  integration, and historical reprocessing outside plan authority.
- Graphiti discovery was healthy and surfaced the relevant Plan 0025 context;
  its advisory active-state wording was stale, so the repository's CLOSED
  authority controls.

Next:

- Invoke the exact `/goal` command recorded in Plan 0052 and begin G0 authority
  replay. Do not access private gold or run prediction/model work before its
  frozen gates authorize those actions.

## Turn 290 | 2026-08-02

Plan:
`docs/dev/plans/0051-2026-08-02-generation-4-media-qualification.md`,
version 1

Packet: P10-G4-A2 | Qualified media-pool freeze

State transition: `OPEN/Generation-4-media-qualification ->
CLOSED/qualified-pool-frozen`.

- Added the Generation-4 qualification authority and 14 focused adversarial
  tests at clean pushed commit `6d7ad4c`.
- Production no-write preview evaluated 12 explicit top-level candidates and
  left the Generation-4 runtime unchanged. Preview hash is
  `af5bcf2d8e60b811bcddbb875dd1044f69a090346c6118525c5c5dd80bc49974`.
- Ten candidates qualified; two were below the 60-second minimum. No candidate
  failed overlap, duplicate, stream, decode, or duration-drift checks.
- Applied manifest hash is
  `8b115bb92930916b087f114ab396f43f08d40b39f5faff8e1254d30a709c29fe`;
  qualified-set hash is
  `e3c908f80c922365ead50795728feb959d8aa93e542ee2882be79efc456e48be`.
- Authority-driven replay recovered the private candidate list and fully
  re-decoded every source without retaining audio. Runtime permissions are
  `0700`/`0600`.
- Only `build_generation4_cohort_preview` is newly true. Gold, preparation,
  models, trials, scores, metrics, selection, profile mutation, integration,
  and reprocessing remain false.

Next:

- Open a separate Generation-4 cohort/gold feasibility plan over the qualified
  pool. Require conversation identity and enrolled-speaker coverage before
  cohort freeze.

## Turn 289 | 2026-08-02

Plan:
`docs/dev/plans/0051-2026-08-02-generation-4-media-qualification.md`,
version 1

Packet: P10-G4-A1 | Media qualification plan opening

State transition: `CLOSED/Generation-3-terminal-stop ->
OPEN/Generation-4-media-qualification`.

- Recorded the durable multi-stage speaker-identity roadmap in
  `docs/dev/notes/0051-2026-08-02-speaker-identity-product-roadmap.md`.
- Graphiti discovery was healthy and returned 10 facts across five episodes;
  it surfaced the existing Plan 0025 speaker-clue workflow. Current repository
  authority confirms Plan 0025 is closed and reusable as the future contextual
  integration boundary.
- File-searcher bounded refresh confirmed 276 audio files under the explicit
  `Documents/Sound Recordings` root.
- Opened Plan 0051 to fully decode and qualify at most 12 explicit top-level
  candidates before any Generation-4 cohort, gold, or biometric work.

Next:

- Implement, test, freeze, and replay the private media-qualification
  authority. Authorize cohort preview only if at least seven candidates pass.

## Turn 288 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-F1 | Prediction-blind preparation terminal STOP

State transition: `OPEN/preflight-passed -> CLOSED/terminal-stop`.

- Committed and pushed the independently audited preparation authority at
  `e884a94`; its no-write preview hash was
  `73d24e5bb30529be35057045a431fbd3557b5314b62d49c71eaa2a796e00c7e4`.
- The single production attempt completed and replayed six P1 units and six P2
  units with 30/30 successful method cells. The seventh P1 failed before any
  application/receipt, windows, trials, evaluation models, scores, or metrics.
- Decode measured 3468.565313 seconds against frozen source duration
  3558.342104 seconds: 89.776791 seconds drift versus the immutable
  0.05-second tolerance.
- Commit `944e554` adds the separately sealed terminal recorder. Independent
  audit is `PASS`; portable STOP preview hash is
  `8cb99b0a28cdf1982e735c53490246d90b4eb25a1240aff779c5a8731121a95c`.
- Applied manifest hash
  `b0f34f5b5e90a4ff483fed0bc7544455b830c6d29ee83aa9cb45fed9c3209d37`
  replays full-body. Runtime directories/files are `0700`/`0600`; every
  post-STOP action is false.
- Validation: six focused tests, complete 708-test suite, compilation, diff
  check, independent re-audit, clean push, and live replay pass.

Next:

- Do not retry or repair Generation 3. Any new acoustic evaluation requires a
  new generation, new cohort/media authority, and separately reviewed plan.

## Turn 287 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-E1 | Reveal and structural denominator preflight

State transition: `OPEN/pre-reveal-envelope-frozen ->
OPEN/preflight-passed`, without audio preparation or model execution.

- Added a self-bound reveal/preflight implementation at clean pushed commit
  `3437d27`. Independent audit drove repairs so preview cannot read private
  gold, reveal authority is written first, parent manifest/receipt retain exact
  private-file validation, and pass/STOP actions are machine-specific.
- Production no-write reveal preview hash is
  `165b03f8838b5d496c317c2269f749b7d4d9d6a98f5aaf5c3e3b5fc0a1820e9b`;
  the 23-entry runtime snapshot remained unchanged.
- Reveal authority
  `fd0d4ec4826ed22ce073f8e65a410d1205e07b04288d739839397b3cbac3dcd5`
  was persisted before private gold was read. Exact revealed outcomes are 10
  enrolled, 10 open-set, and 8 mixed/unknown excluded label instances.
- All nine units pass structural preflight: conservative per-unit maxima are
  120 genuine, 120 known-impostor, and 240 open-set trials against required
  20/100/20. Preflight hash is
  `1f9a388ea8f26b8239f009bc1af984c6a8402d6b8e467c5ccbcbd298aa3b6126`.
- Full structural replay passed without audio, preparation, models, or scores.
  Only `run_prediction_blind_p1_p2` is newly true. Conditions, windows, exact
  trials, models, scores, metrics, decision, mutation, integration, and
  reprocessing remain false.
- Validation: seven focused tests, 102 impact tests, and the complete 695-test
  suite pass; final independent re-audit is `PASS`.

Next:

- Run isolated prediction-blind P1/P2 over the seven frozen recordings, then
  measure and freeze all five condition dimensions. Stop before windows unless
  every dimension has at least two observed values and zero missing recordings.

## Turn 286 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-D1 | Independently audited pre-reveal envelope

State transition: `OPEN/recalibration-thresholds-frozen ->
OPEN/pre-reveal-envelope-frozen`, without reveal or evaluation execution.

- Added a separate pre-reveal authority and Generation-3 terminal policy at
  clean pushed commit `da4acdc`. Required independent audit found and drove
  repairs to exact condition-function identity, terminal STOP semantics, and
  unique finite candidate-matrix coverage; final re-audit is `PASS`.
- The production no-write preview independently reproduced hash
  `c9db91fb9ed2d69055893ded7a9f987f641b3962364bef6de66f88061f968797`
  and left the complete 19-entry runtime snapshot unchanged.
- The envelope binds seven conversations, 28 labels, 12 known subjects, 24
  same-person subject/session pairs, six profiles, nine threshold/temperature
  units, five exact condition dimensions/algorithms, the 12-window evaluation
  cap, exact preparation/trial/score/metric/minimum-evidence rules, and terminal
  precedence `stop`, `reject`, `select`, `refine`.
- Applied and full-body replayed authority
  `generation3-pre-reveal-2dac320b6577456bd38a281b`; manifest hash is
  `98aa3a077eac3932e43a1938aaccbd42d9fa19b7c8f2f36c9b5625b86f959d6c`.
- Only envelope completion and the separate reveal action are true.
  Denominator preflight, preparation, conditions, windows, exact trials,
  models, scores, metrics, decision, mutation, integration, and historical
  reprocessing remain false.
- Validation: 11 focused tests, 149 impact tests, and the complete 688-test
  suite pass.

Next:

- Implement the separately self-bound reveal and structural denominator
  preflight. A failed preflight must freeze terminal `STOP` before P1/P2 or
  model loading; a pass may authorize prediction-blind P1/P2 only.

## Turn 285 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-B3 | Successor scoring and threshold freeze

State transition: `OPEN/recalibration-authority-frozen ->
OPEN/recalibration-thresholds-frozen`, without Generation-3 reveal or
evaluation execution.

- Added a separate self-bound executor in commit `7d5b535`, pushed clean and
  upstream-even before production model load. No module sealed by the pre-score
  authority changed.
- The aggregate-only execution preview hash is
  `6a6367554826ca76731b68e1b9b99e752268ecd1cb7f01e060ef8c39b69cfeba`.
  Execution authority
  `39298c74aab4a773945268cd73fbaabccf88e8e3026a4041ea6eeed29b715b4f`
  was written before any adapter loaded.
- Completed the exact 396-trial private calibration matrix across three pinned
  models, three preparation methods, 22 windows, and six successor profiles.
  Every one of nine units has exactly 44 total, 9 genuine, 35 impostor, and 26
  open-set trials. Score matrix hash is
  `3fb983b06b1984724c2f0e3e3c01f55065ff755e36416260c33fe0f2649201c2`.
- Structural score replay passed without audio or model execution. All nine
  deterministic threshold/temperature units then froze and recomputed exactly
  from persisted scores. Threshold application hash is
  `308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1`;
  threshold-set hash is
  `a927b0d9752d4b79ec42f5248afd2028db1c44414ff2d733c46c7b01b6d16759`.
- Abstention remains exactly zero. Portable receipts contain no paths,
  profile/subject IDs, biometric scores, or threshold values. Only pre-reveal
  envelope construction is newly authorized; reveal and every evaluation,
  mutation, integration, or reprocessing action remain false.
- Independent audit result is `PASS`: full live score/threshold recomputation
  matched, fail-fast PCM/model sentinels proved structural replay performs no
  acoustic execution, all nine pairs are finite and policy-candidate-bound,
  and runtime modes are exact `0700/0600`.
- Validation: seven new focused tests, 104 impact tests, and the complete
  677-test suite pass.

Next:

- Build the independently audited pre-reveal envelope binding the frozen
  cohort, gold, six profiles, score matrix, nine threshold/temperature pairs,
  preparation/window/trial/metric/decision policies, repository authority, and
  negative action gates before any Generation-3 reveal.

## Turn 284 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-B2 | Successor recalibration pre-score freeze

State transition: `OPEN/recalibration-authority-built ->
OPEN/recalibration-authority-frozen`, without calibration scoring or evaluation
reveal.

- Independently audited implementation commit `cacf58e` was pushed clean and
  upstream-even after 169 impact tests and the complete 670-test suite passed.
- Applied and idempotently replayed exact authority
  `generation3-recalibration-99fcabf628404df4940f2be0` with manifest hash
  `a87d873e79d1a859d45734e85e1b02524495915126ce02fd57cff499f7046e53`.
- The applied aggregate receipt preserves the 22 windows, six profiles, two
  subjects, three candidates, three methods, nine units, exact derived
  44/9/35/26 denominators, and zero overlap in all four semantic dimensions.
- Only recalibration-authority freeze and calibration-model execution are true.
  Threshold/temperature freeze, pre-reveal construction, evaluation reveal,
  preparation, windows, exact trials, evaluation scores/metrics/decision,
  profile mutation, integration, and reprocessing remain false.

Next:

- Add a separately self-bound score/threshold executor without changing any
  frozen authority module, commit and push it before model load, then run the
  exact 396-trial successor calibration matrix and deterministically freeze all
  nine threshold/temperature pairs.

## Turn 283 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-B1 | Successor recalibration pre-score authority

State transition: `OPEN/gold-frozen -> OPEN/recalibration-authority-built`,
without calibration scoring or evaluation reveal.

- Added a separate preview/apply/replay authority that binds the exact
  historical 22-window calibration membership, six active successor profiles,
  three candidates, three score methods, and nine candidate-method units before
  any model can load.
- Replayed the active training authority and all three prior evaluation corpus
  authorities. Calibration-to-training and calibration-to-Generation-3 overlap
  is zero independently for source, recording, conversation, and derivative
  semantic identities; every prior evaluation source has validated private,
  hash-matched semantic transcript lineage.
- Derived, rather than declared, the exact per-unit calibration denominators
  from frozen windows joined to the two active subjects: 44 total, 9 genuine,
  35 impostor, and 26 open-set trials.
- Enforced the complete candidate-by-subject Cartesian profile inventory, one
  coherent P3 profile/generation lineage per subject, current lifecycle and
  descendant eligibility, exact model revisions/assets, preprocessing and
  eight module hashes, and current clean upstream-even replay.
- Cross-checked the immutable 28-label gold receipt against the frozen cohort
  membership without reading Generation-3 gold or audio. The portable
  projection contains only aggregate counts, hashes, zero-overlap evidence,
  and action flags.
- Independent audit found and drove repairs for false declared denominators,
  incomplete profile shape, missing module/current-repository/gold bindings,
  mismatched semantic namespaces, and unvalidated prior-evaluation/training
  transcript lineage. Final re-audit result is `PASS`.
- Validation: 169 impact tests and the complete 670-test repository suite pass;
  no-write live preview content hash is
  `930dd537819dbefd2bead697fef3d930c1bb768f9ae8efac59c87fd515ed6ec9`.

Next:

- Commit and push the audited implementation, apply/replay the exact pre-score
  authority, then run and freeze the complete successor calibration score
  matrix and nine threshold/temperature pairs before pre-reveal construction.

## Turn 282 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-C2 | Exact private gold freeze

State transition: `OPEN/gold-implementation -> OPEN/gold-frozen`, without
evaluation reveal.

- Commit `43fcced` passed 65 focused/impact tests and the complete 664-test
  repository suite, was pushed upstream-even, and became the executable gold
  authority checkpoint.
- The exact private preview contained all 28 frozen labels: 10 enrolled, 10
  open-set, 2 mixed, and 6 unknown outcomes across 12 known subjects. The two
  enrolled subjects cover three and seven independent conversations.
- Independent no-write reproduction matched the membership, gold-body, and
  preview hashes exactly and returned `PASS` after checking every cited known
  identity against its target diarized label and both active-P3 lineage
  bindings.
- Gold authority `generation3-gold-5f60fa794c40c8fa5a2c5cb0` was applied and
  replayed idempotently. Its manifest hash is
  `5e91c62985d137ca64689e6cd49872b92ebce1051689d62f43e32d000824495e`;
  its gold-body hash is
  `29166e3874a152d5254007c05af97abf2d8ddfcbc97615a96784a5b4751d5399`.
- Runtime directories remain `0700` and both immutable files remain `0600`.
  The aggregate receipt contains no names, subject IDs, source membership,
  paths, transcript text, audio, embeddings, or biometric scores.
- Only gold freeze and successor-recalibration-authority construction are true.
  Reveal, audio preparation, windows, exact trials, model execution, scoring,
  metrics, decision, profile mutation, integration, and historical reprocessing
  remain false.

Next:

- Freeze and independently audit successor recalibration across the exact six
  active profiles and nine candidate-method threshold/temperature pairs before
  constructing the pre-reveal envelope.

## Turn 281 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-C1 | Exact private gold authority implementation

State transition: `OPEN/exact-preview -> OPEN/gold-implementation`, without
gold freeze or evaluation reveal.

- Added a separate private gold preview/apply/replay authority. It requires one
  outcome for each of the exact 28 frozen diarized labels and permits only
  enrolled, cohort-local open-set, mixed, or unknown outcomes.
- Both active opaque P3 subject IDs are independently bound through exact Plan
  0049 training sources, speaker-label IDs, and direct-address/response
  utterance evidence before evaluation mappings can use those subject IDs.
- Known evaluation labels require operator confirmation containing the complete
  identity or cited transcript utterances containing the complete open-set
  identity. Enrolled direct-address evidence may use only the complete first
  name or full identity because the separate active-P3 training binding supplies
  the full identity authority.
- The authority enforces both enrolled subjects across at least two independent
  conversations, at least five total known subjects, and at least one open-set
  label. Mixed and unknown labels carry no identity or evidence body.
- The portable projection contains counts, hashes, and action flags only. An
  initial subject-ID-key leak was removed; names, subject IDs, source membership,
  transcript text, paths, audio, embeddings, and scores are absent.
- Independent audit required three repair rounds: exact 28-label enforcement,
  explicit identity-token evidence, and complete-token rather than substring
  matching. Final code-audit result is `PASS`. Focused tests pass and the prior
  complete repository run reported 664 passed.

Next:

- Commit and push the independently audited implementation, construct the exact
  production private gold preview, audit its evidence body, then freeze/replay
  gold without reveal. Recalibration remains the only newly eligible successor.

## Turn 280 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-A1 | Generation-3 exact cohort preview authority

State transition: `OPEN/inventory -> OPEN/exact-preview`, without reveal or
acoustic execution.

- Implemented a separate Generation-3 preview/apply/replay authority with an
  exact-seven cohort contract, immutable private membership, aggregate-only
  receipts, and an all-negative pre-apply action vector.
- The first production preview failed closed because the former fourth source
  already occurred in a frozen corpus. Per the operator's replacement rule, a
  distinct transcribed conversation replaced it; the rejected source is not in
  the Generation-3 cohort.
- The repaired lineage authority replays all three prior corpora and transcript
  derivatives, the active Plan 0049 intake, both active P3 generations, and
  their current source/recording/conversation/derivative lineage.
- Source bytes plus independent semantic recording, conversation, and
  transcript-derivative identities all report zero overlap for the replacement
  seven-conversation cohort. Missing recording or conversation identity fails
  closed; media re-encoding does not bypass the semantic dimensions.
- The exact private preview contains 7 conversations and 28 diarized-label
  instances. Non-acoustic evidence still proposes 10 enrolled outcomes; the
  replacement raises proposed other outcomes from 16 to 18 pending exact gold.
- Applied receipts truthfully authorize only membership freeze and construction
  of the private gold-review packet. Reveal, preparation, windows, models,
  scores, metrics, decisions, profile mutation, integration, and historical
  reprocessing remain false.
- Repository replay binds the Generation-3 module, training/P3 dependencies,
  and the private immutable-write/permission dependency. Focused negative tests
  cover all four overlap dimensions, stale previews, portable privacy, applied
  actions, dependency binding, and missing identity evidence.
- Independent audit passed, 661 repository tests passed, and commit `65733dd`
  is pushed and upstream-even. Exact membership authority
  `generation3-cohort-714fb3cf3f881b8bad6757ed` applied and replayed with
  manifest hash `7e2ede46f554507583032dd15f7eb3fa5c2b1449dfb36cf622371bca8ef010db`.
  Its directory is `0700`, files are `0600`, and every action after private
  gold-packet construction remains false.

Next:

- Build and independently audit the private per-label gold review packet, then
  freeze exact gold and successor recalibration before any reveal. Do not
  prepare audio or execute models.

## Turn 279 | 2026-08-02

Plan:
`docs/dev/plans/0050-2026-08-02-generation-3-acoustic-evaluation.md`,
version 1

Packet: P10-G3-A | Generation-3 inventory and authority design

State transition: `CLOSED/Plan-0049 -> OPEN/Plan-0050`, without changing the
Generation-2 terminal STOP.

- Reopened `VISION.md`, repo policy, Plans 0043-0049, live profile receipts,
  CodeGraph, and advisory Graphiti context. Current repo/runtime artifacts
  remain authoritative.
- Independent read-only audit confirmed Generation 2 stopped on zero overlap
  between five evaluation subjects and two active profile subjects. No audio,
  windows, models, scores, metrics, or selection ran.
- Six successor profiles are active across three pinned models. Generation 3
  must recalibrate these successor artifacts before reveal; predecessor-bound
  thresholds are not silently reused.
- The requested file-searcher workflow refreshed only the named
  `Documents/Sound Recordings` folder. It surfaced novel, already-transcribed
  candidates with non-acoustic evidence proposing enrolled and open-set label
  mappings. These remain private selection leads pending exact per-recording
  gold; they are not established identity facts.
- Plan 0050 requires a new source/recording/conversation/derivative-disjoint
  cohort, active-profile subject IDs in gold, seven conversations, both enrolled
  subjects across two conversations, five total subjects, and 20 genuine / 100
  impostor / 20 open-set trials per unit.
- Condition dimensions, measurement algorithms, and coverage minima freeze
  before reveal; actual noise/usable-duration coverage is measured through
  prediction-blind P1/P2 after reveal and must pass before the exact-trial child
  or any model execution.
- The selected seven-conversation inventory proposes 10 enrolled and 16 other
  diarized label/conversation outcomes pending exact gold. The inherited cap
  of three windows would permit only 30 impostor trials with two profiles, so
  Generation 3
  precommits an evaluation-only cap of 12 before reveal. This permits a
  structural maximum of 120 genuine and 120 impostor trials per unit while
  retaining all overlap, duration, timestamp, and same-window-set rules.
- Generation-2 code and receipts remain immutable. Separate Generation-3
  authority/evaluation modules will add successor recalibration, positive-path
  execution, and the missing exact-trial child.
- Corrected Plan 0045's stale top-level state to `CLOSED` and marked its
  seven-direct-attestation route `SUPERSEDED`; Plan 0047 closed the device gate
  under a different two-direct plus five-metadata authority and composite.

Next:

- Finish the private candidate inventory and exact disjointness/denominator
  projection, then implement and independently audit the Generation-3
  pre-reveal authority. Do not reveal or run models before that commit is clean
  and pushed.

## Turn 278 | 2026-08-02

Plan:
`docs/dev/plans/0049-2026-08-02-additional-acoustic-training-conversations.md`,
version 3

Packet: P10-T1 | Additional acoustic training-conversation intake

State transition: `CLOSED/TERMINAL-STOP/Plan-0037 -> CLOSED/Plan-0049`, without
reopening or changing the stopped evaluation.

- Graphiti discovery was healthy but returned no current Plan 0048/0049
  authority, so pushed plans and private runtime manifests remain primary.
- File-searcher resolved `Documents/Sound Recordings` to the live Windows path.
  Five distinct, already-transcribed conversations were selected; their source
  hashes do not occur in any prior Plan 0037 corpus.
- Filename-derived selection leads remain private, and identity remains
  unconfirmed for the exact diarized labels. Filenames are not enrollment
  authority.
- Plan 0049 defines sufficiency as at least two confirmed people with at least
  two independent sessions and six eligible windows each. It preserves the
  Plan 0048 STOP and forbids evaluation reuse or automatic promotion.
- The exact-five intake applied and replayed under private `0700/0600` state.
  Its portable receipt contains no paths, names, transcript text, raw audio,
  embeddings, or scores and authorizes preparation only.
- The first P1 apply failed closed on duration drift. A no-output diagnostic
  decoded all five sources cleanly and measured 0.0006–0.0944 seconds of AAC
  stream-versus-PCM timing drift. Plan version 3 keeps the shared 0.050-second
  P1 module byte-exact and uses an isolated training worker whose module
  instance alone sees 0.100 seconds; concurrent ordinary callers remain at
  0.050 seconds. The training recipe/run identity changes and the failed dry
  run is not reused.
- All five admitted conversations then completed P1 with active full-body
  replay. All five P2 comparisons completed with every required method
  successful: 25 attempted, 25 successful, 0 failed, and 0 blocked.
- A private review packet freezes 14 diarized-label instances and 40 clean
  candidate clips selected inside transcript-labeled/Silero speech while
  excluding pyannote overlap and speaker-change boundaries.
- Exact-recording readouts provide direct-address or role evidence for every
  mapping in four conversations. The remaining label has a unanimous
  three-model match to an operator-enrolled profile plus same-day direct-address
  continuity. The operator confirmed all 14 mappings against evidence preview
  `e8063c5786d0ed28abd345f2f069483f6be3dd1db446153838c38f5892a8904c`;
  the private immutable confirmation receipt is
  `b9eb1e64e2dcbb20484e240fd765ac81de9c44b1731f666f2f738fadea59224c`.
- Two successor P3 generations applied and replayed: 10 eligible windows over
  four independent conversations for one confirmed person and 15 windows over
  five for the other. Both respect the three-window-per-conversation cap.
- P3 parent supersession exposed a fail-closed lifecycle ordering gap. P4 now
  acknowledges the parent-owned invalidation before successor promotion and
  resumes an already-promoted staged descendant deterministically after an
  interrupted apply.
- Six successor P4 profiles are active across all three pinned models, and all
  six predecessors are superseded. Private application receipt
  `29d1ec10bee8ee009d63e907e7b2f0c0c881c4e8693ef8dbf3312dc76e27d19b`
  records `training_sufficiency_met=true`. Evaluation was not opened.

Next: none for Plan 0049. Any new evaluation generation requires a separate
bounded plan and authority; Plan 0048's terminal STOP remains unchanged.

## Turn 277 | 2026-08-01

Plan:
`docs/dev/plans/0048-2026-08-01-plan-0037-generation-2-evaluation-execution.md`,
version 2

Packet: P4E2-E | Generation-2 evaluation reveal and terminal preflight

State transition: `CLOSED/P4E2-D4 -> CLOSED/TERMINAL-STOP/Plan-0037`.

- The authorized private reveal bound the exact two-record successor
  evaluation split. It contains five opaque evaluation subjects and has zero
  overlap with the two subjects represented by the six frozen profiles.
- All nine candidate-by-method units therefore have a structural maximum of
  zero genuine and zero impostor trials, below the frozen 20/100 minima. The
  terminal reason is `trial_class_denominator_below_policy`.
- Independent audit found one portable-receipt revocation gap. The repair
  added explicit false flags for preparation, window freeze, exact-child
  construction, models, scoring, metrics, and selection plus binding/tamper
  tests; targeted re-audit returned `PASS`.
- Nine focused and all 639 repository tests passed; compilation and
  `git diff --check` passed. Reviewed implementation commit `937ca21` is pushed.
- Applied run `generation-2-evaluation-stop-5945db0810a482bbbe80db74`
  (content SHA-256
  `5945db0810a482bbbe80db746a4851863ea89a6fa2da4f068aaf0155dd1989c9`)
  replays full-body with exact `0700` directories and `0600` files.
- No audio preparation, windows, exact-trial child, models, scores, metrics,
  model/method selection, P5 integration, or P6 historical reprocessing ran.
  Plan 0037 closes unsuccessfully at the evidence gate; nothing is promoted.

Next:

- Keep the acoustic path shadow-only and Plan 0036 sealed. Any future acoustic
  attempt requires a new plan and an evaluation cohort with sufficient frozen
  profile coverage to satisfy every trial class before model execution.

## Turn 276 | 2026-08-01

Plan:
`docs/dev/plans/0046-2026-08-01-plan-0037-p4e2-generation-2-authority.md`,
version 3

Packet: P4E2-D4 | Generation-2 pre-reveal production freeze

State transition: `OPEN/P4E2-R3-PARTIAL+P4E2-D3 -> CLOSED/P4E2-R3+CLOSED/P4E2-D4`.

- The operator confirmed frozen cases 2 and 4 used the same webcam microphone.
  A private sparse operator authority preserved exactly those two facts without
  advancing or fabricating the original sequential ledger. The augmented
  composite merged them with five source manufacturer facts: `7 / 7`
  authoritative, `2` distinct devices, `0` missing, no blockers.
- Independent audit and re-audit closed row-level generation-2 evidence,
  legacy compatibility, exact apply-manifest reconstruction, and partial
  authority discovery defects. All 630 tests, compilation, and
  `git diff --check` passed.
- Commits `aa368ab`, `32644df`, and `ed20f8b` are pushed and upstream-even.
  Applied authority
  `generation-2-pre-reveal-authority-e36736a176600d5536c7c668` replays
  full-body at reviewed preview content SHA-256
  `b83368b7bca2c5634f98c511844e82d78e87a954e99468a611b23efc5c0ff169`
  with exact private modes.
- Evaluation reveal, offline preparation, and window freeze are authorized but
  remain `not_run`. Models, scoring, metrics, and terminal decisions remain
  forbidden until the exact post-window child authority exists and replays.

Next:

- Open a bounded generation-2 evaluation execution packet: reveal only the
  sealed two-record evaluation split, freeze immutable windows, construct and
  independently review the exact trial child, then execute the frozen model ×
  method matrix once.

## Turn 274 | 2026-08-01

Plan:
`docs/dev/plans/0047-2026-08-01-plan-0037-source-device-metadata.md`,
version 1

Packet: P4E2-R3 | Source-embedded device metadata supplement

State transition: `OPEN/P4E2-R1-CASE-1+P4E2-D3 -> OPEN/P4E2-R3-PARTIAL+P4E2-D3`.

- Operator-authorized indexed search recovered all seven frozen sources with
  exact SHA-256 matches, so the conditional replacement path did not run.
- A separate immutable authority admits only the exact manufacturer hardware
  tag `Samsung:SamsungModel`; it does not alter or masquerade as Plan 0045
  direct-operator testimony. Five cases identify one opaque physical device;
  cases 2 and 4 explicitly remain unavailable.
- Independent audit exposed and closed campaign/body detachment, duplicate
  cases, receipt path disclosure, distribution drift, and extraction-time
  source-swap risks. Final targeted verification returned `PASS`; nine focused
  and all 622 tests, compilation, and `git diff --check` passed.
- Implementation commit `90c62e38f59eb2d970640593d5678f58880115b4`
  is pushed. Production authority
  `source-device-metadata-e9c6839faeaa1bdfd6bfe842` applied and replayed with
  full-body equality and exact private modes.

Next:

- Keep generation-2 apply/reveal blocked. Composite integration remains open,
  but cannot pass with two missing recordings and only one distinct device.
  Obtain genuine device evidence for cases 2 and 4 or define and review a new
  cohort/replacement packet; do not infer from ineligible metadata.

## Turn 273 | 2026-08-01

Plan:
`docs/dev/plans/0045-2026-08-01-plan-0037-p4e2-device-provenance-refinement.md`,
version 2

Packet: P4E2-R2 | Device-campaign descendant continuity

State transition: `OPEN/P4E2-R1+P4E2-D3 -> OPEN/P4E2-R1-CASE-1+P4E2-D3`.

- A final blocker audit exposed that the frozen exact-seven device campaign
  rebuilt identity from current `HEAD`, making it stale after reviewed Plan
  0046 descendant commits even though its frozen module had not changed.
- Replay now reconstructs the original body with its frozen repository
  authority while requiring a clean current checkout, ancestor proof, and the
  exact historical module blob from Git. Descendant reapply reuses the sole
  campaign for the exact predecessor pair and fails closed on duplicate valid
  campaigns.
- One bounded audit repair added ambiguity rejection. Final independent
  re-audit returned `PASS` on scoped code/test diff SHA-256
  `3f6cbebdd0f945453775b4d065801e88784f11df213f3abe1f0664da900d96c1`.
  Twelve focused, 40 joined, and all 613 repository tests passed; compilation
  and `git diff --check` passed.
- Repair commit `bb975ebe5e46f880cefadf4267d03e2b5d7ede83` is pushed and
  upstream-even. Production full-body replay, singleton descendant reapply,
  exact private modes, and idempotent case-1 reopen passed with unchanged file
  hashes and mtimes. The campaign still has zero attestations.

Next:

- Record only direct operator knowledge of the physical device for case 1, or
  record `unavailable` if the operator genuinely does not know. Do not infer
  from identity, filename, codec, container, path, or blanket authorization.

## Turn 272 | 2026-08-01

Plan:
`docs/dev/plans/0046-2026-08-01-plan-0037-p4e2-generation-2-authority.md`,
version 2

Packet: P4E2-D3 | Generation-2 pre-reveal authority

State transition: `OPEN/P4E2-R1+P4E2-D2 -> OPEN/P4E2-R1+P4E2-D3`.

- Added deterministic generation-2 preview/replay with exact successor seal,
  condition, historical calibration, profile/model, threshold/margin,
  candidate-matrix, terminal-policy, and post-window exact-trial-child
  bindings. Preview authorizes no reveal, model, score, metric, decision, or
  write.
- Added exact read-only Plan 0044 historical replay. It rehashes all seven P1
  audio artifacts, requires the ordered five-method success set, and rehashes
  all 35 P2 outputs. Production replay returned frozen safe projection
  `bbadd46c...befbd` and full-body equality.
- One bounded audit repair added full calibration-body identity, a frozen
  successor projection, canonical composite content/ID recomputation, and P1
  artifact integrity. Final independent re-audit returned `PASS` on scoped
  code/test/policy diff SHA-256
  `25b8e01d46f61195d5205947dc5378b7f03e3094f98dbe69c890f31bf59bf529`.
- Twenty-two focused tests, 56 joined predecessor tests, and all 607 repository
  tests passed. Compilation, production historical replay, and
  `git diff --check` passed.

Next:

- Continue the Plan 0045 one-case-at-a-time device campaign. Keep production
  generation-2 apply, reveal, scoring, and selection `not_run` until the exact
  seven direct attestations yield at least two opaque devices and a passing
  replayed composite authority.

## Turn 271 | 2026-08-01

Plan:
`docs/dev/plans/0046-2026-08-01-plan-0037-p4e2-generation-2-authority.md`,
version 1

Packet: P4E2-D2 | Historical calibration replay seam

State transition: `OPEN/P4E2-R1 -> OPEN/P4E2-R1+P4E2-D2`.

- Isolated the archived calibration replay failure to one exact field: P2
  module `467627bc...10bdc` versus the reviewed evaluation-split seam
  `700e10d8...bd595`; every other authority field matched.
- Added an exact replay-only compatibility contract bound to calibration
  authority `0fe6009b...fae5`. Default replay remains strict, and the contract
  is unavailable to build, reveal, preparation, selection, and apply paths.
- Added dedicated read-only validation for the archived split reveal,
  preparation, and window selection before existing profile, descendant,
  396-trial score, and nine-threshold replay checks.
- Production replay succeeded with an immutable-writer spy and unchanged hash
  plus mtime state for all six archived authority/stage/application artifacts.
- One bounded audit repair removed write-capable stage routing and restored two
  accidentally touched inference exception boundaries. Final independent
  re-audit returned `PASS` on scoped code/test diff SHA-256
  `9dae572b4683ea54176dd3f9fc750b8a84ad4f1262f20d84da4eb964671ea6b7`.
  Twenty-two historical tests, 73 joined verification tests, and all 585
  repository tests passed; compilation and `git diff --check` passed.

Next:

- Build and independently audit the deterministic no-write generation-2
  authority preview. Keep production apply, reveal, scoring, and selection
  stopped until Plan 0045 yields seven direct device attestations, at least two
  opaque devices, and a passing replayed composite condition authority.

## Turn 270 | 2026-08-01

Plan:
`docs/dev/plans/0045-2026-08-01-plan-0037-p4e2-device-provenance-refinement.md`,
version 1

Packet: P4E2-R1 | Capture-device provenance refinement

State transition: `CLOSED/TERMINAL-STOP/P4E2-D1 -> OPEN/P4E2-R1`.

- Graphiti is healthy but contains no current Plan 0044 refinement evidence;
  current repo/runtime authorities control.
- Read-only inventory of all seven current transcript structures found no
  device, recorder, microphone, hardware, manufacturer, model, or capture-app
  fields. Authoritative M4A metadata exposes only generic audio-handler labels;
  source copies have no extended attributes, and original recorded paths are
  absent.
- Opened a bounded exact-seven operator-attestation packet. It will bind the
  corpus and Plan 0044 condition authority, enforce one-at-a-time hash-chained
  review, preserve append-only corrections, and accept only direct operator
  knowledge as physical-device evidence.

Next:

- Implement and independently audit the private attestation authority, commit
  and push it, freeze/replay the exact campaign, then open the first factual
  device-provenance case.

## Turn 269 | 2026-08-01

Plan:
`docs/dev/plans/0044-2026-08-01-plan-0037-p4e2-condition-measurement.md`,
version 2

Packet: P4E2-D1 | Successor condition measurement

State transition: `CLOSED/P4E2-C2 -> CLOSED/TERMINAL-STOP/P4E2-D1`.

- Opened the bounded condition packet against the exact replayed successor
  corpus. Graphiti was healthy but returned only older Plan 0037/P2 context;
  current plan, corpus, source, code, and runtime evidence control.
- A metadata-only source probe found six mono and one stereo recording, source
  rates of 16, 22.05, 44.1, and 48 kHz, and multiple encoding profiles.
  Physical device identity is absent and must not be inferred from encoding.
- Added and independently audited the exact condition preview/apply/replay
  orchestrator. The audit repaired an actual split-recount gap and extended the
  one-attempt failure boundary through final manifest/receipt writes. Six
  focused tests and all 557 repository tests passed. Commit `837edf0` is pushed.
- Production preview `successor-conditions-b76095fdaf488f41930cc1f4` executed
  exactly 7 P1 successes and 35 P2 method successes. Full-body replay matched
  content SHA-256
  `3ef3bcdabc776dfd80fb2002fa0b29377008c08ae9b2dc5f715e6155eb0f1a5e`.
  All runtime directories/files are private `0700`/`0600`.
- Channel, noise, telephone-bandwidth, and usable-duration each passed with two
  observed values and no missing recordings. Physical device had zero observed
  values and seven missing recordings, the sole blocker. Independent runtime
  audit returned `PASS`; no biometric scoring or terminal split reveal ran.

Next:

- Keep generation-2 authority construction and terminal execution stopped.
  A separately reviewed refinement must obtain explicit physical capture-device
  provenance or freeze a genuinely eligible new cohort; encoding profiles may
  not satisfy the device gate.

## Turn 268 | 2026-08-01

Plan:
`docs/dev/plans/0043-2026-07-31-plan-0037-p4e2-successor-evaluation.md`,
version 7

Packet: P4E2-C1/P4E2-C2 | Operator gold and successor cohort freeze

State transition: `OPEN/REVIEW-TRANCHE/Plan-0037-P4E2 -> OPEN/GENERATION-2-AUTHORITY/Plan-0037-P4E2`.

- Completed all seven one-at-a-time, prediction-excluded operator reviews.
  Append-only corrections identify Jordan Katz in case 1, Nacu Hernandez in
  case 4, and unify the recurring Eric identity. Superseding gold freeze
  `7870394e-417f-40f0-8e04-3de5e1fa130b` is current; the earlier freeze remains
  audit history.
- Rejected the legacy hash split because it measured `6 / 1 / 0`. Added the
  successor-only, model-independent chronological `3 / 2 / 2` policy with
  exact-seven membership and unchanged legacy behavior.
- Added exact prior-corpus, gold-freeze, clean-repository, and module authority
  bindings; live source/transcript/gold/index drift checks; reviewed-hash apply;
  readiness-before-write; private idempotent freeze; and read-only exact-body
  replay with tamper tests.
- Independent review returned `PASS` on diff SHA-256
  `fdf8cded9c96926aae03bbacc2e88cade05e550d1430bd37d85ed62a3afd0c6f`.
  Five focused and 551 full tests passed; compilation and diff checks passed.
  Commit `50f34ab` is pushed.
- Applied private corpus `acoustic-corpus-4a2b13e7bdc201f694af2f43` at content
  SHA-256 `4a2b13e7bdc201f694af2f43d4ab845749eeeb3ea06c7a97a40164cab40b83fe`.
  Manifest SHA-256 is
  `4b77479d25d7b248cc62d500ed84c1604f105848da25ecef53661c5d9ea05a30`;
  replay returned `full_body_match=true`; files are `0600` under `0700`.
- Frozen evidence is 7 recordings/conversations, 18 labels, 10 subjects, 3
  recurrent subjects, 23 same-person pairs, and 114 different-person pairs.
  It is `ready_for_p1_measurement` but explicitly not promotion eligible.

Next:

- Build the bounded P1/P2 condition-measurement packet and bind its private
  derivative/comparison receipts before independently reviewing generation-2
  reveal authority. Missing real device or other two-value condition coverage
  must stop or refine; it cannot be relabeled as terminal-select evidence.

## Turn 267 | 2026-07-31

Plan:
`docs/dev/plans/0043-2026-07-31-plan-0037-p4e2-successor-evaluation.md`,
version 5

Packet: P4E2-C1 | Successor operator-review tranche

State transition: `OPEN/BLOCKED-COHORT/Plan-0037-P4E2 -> OPEN/REVIEW-TRANCHE/Plan-0037-P4E2`.

- Verified the parent campaign already freezes all 375 current transcript rows;
  a new unfiltered campaign would not create new evidence.
- Its frozen future pool has 236 unreviewed reviewable rows. Seven current rows
  have durable recording/conversation IDs, accessible sources, current artifact
  hashes, full prior-corpus disjointness, and pairwise pool disjointness. They
  cover 18 opaque speaker labels across seven multi-label recordings.
- The other 229 lack durable identities. P4E2-C1 excludes them without rewriting
  historical transcript artifacts or inventing grouping semantics.
- Opened a bounded seven-case successor campaign packet. Selection freezes
  before review; cases will be presented one at a time; only explicit operator
  decisions may become prediction-excluded gold.
- Implemented a private content-addressed identity-scalar projection, exact
  frozen-parent membership, a non-configurable seven-case selector, reviewed-
  preview hash binding before first apply, full-body replay, and exact-schema
  chained cursor receipts. A guard test prevents selector access to transcript
  payload/text columns, the legacy full-store preview, or transcript artifacts.
- Validation: 16 campaign tests and 549 full repository tests passed;
  compilation and `git diff --check` passed. Independent read-only review
  returned `PASS` after two bounded repair cycles.

Next:

- Implement, independently audit, and apply the deterministic private successor
  campaign; then present the first case in plain English.
- After all seven reviews, measure trial feasibility without weakening the
  terminal policy.

## Turn 266 | 2026-07-31

Plan:
`docs/dev/plans/0043-2026-07-31-plan-0037-p4e2-successor-evaluation.md`,
version 3

Packet: P4E2-A/P4E2-B | Successor inventory and pre-freeze seam

State transition: `CLOSED/STOP/Plan-0037-P4E -> OPEN/BLOCKED-COHORT/Plan-0037-P4E2`.

- Opened the successor packet without reusing the revealed generation-1
  cohort. A read-only inventory found 24 latest eligible recordings, all 24
  overlapping the original P0 corpus, and zero fully disjoint candidates.
- New operator-confirmed recordings are therefore required before a new cohort,
  authority, split reveal, audio/model execution, or terminal selection.
- Proceeding with the authority-independent P2 evaluation seam and a
  privacy-safe readiness receipt so code hashes can stabilize before any future
  terminal authority freezes.
- Added the P2 `evaluation` later-split seam across dry-run, apply, replay, and
  lineage. It requires an exact 64-hex authority, preserves the actual split in
  every receipt, keeps development authority-free, and rejects unknown splits.
  Pre-authority P2 module SHA-256 is
  `700e10d802a6443eab9d2bb9c6b9a7519cff26021ffec23acbdb767f12bcd595`.
- Authoritative private readiness receipt SHA-256
  `cae5de01dd91d7f620b17071dd87dbc4ae991793f07a4257186d18d3e587287d`
  records 24 overlaps in every identity dimension and zero fully disjoint
  candidates. It uses metadata only, is `0600` under `0700`, and ran no source-
  blob/transcript-body read, model, split reveal, or external write.
- Invalidated first implementation receipt
  `85769c302b1e6e762d77b8ab809850cb688aa7d10cfa843acd7eb627e7bd010d`
  as evidence because its reused collector rehashed source blobs while claiming
  no audio read. It remains non-authoritative audit history and is superseded
  by the receipt above.
- Validation: 27 focused tests and 547 full repository tests passed;
  compilation and `git diff --check` passed.
- Independent read-only audit returned `PASS` after verifying the metadata-only
  collector, exact authoritative receipt/module hashes, private modes, guarded
  no-blob-open test, full suite, compilation, and diff integrity.

Next:

- Checkpoint `939005c` is committed, pushed, clean, and upstream `0/0`.
- Resume cohort freeze only after the governed campaign contains genuinely new
  document-, recording-, conversation-, and source-disjoint eligible evidence.

## Turn 265 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 10

Packet: P4E | Terminal evaluation generation 1

State transition: `OPEN/Plan-0037-P4E -> CLOSED/STOP/Plan-0037-P4E`.

- Corrected terminal-evaluation authority
  `9cb2ae3700846f8137f0c48de4d3aa2133009738b5e8d591f469cb5e8edef485`
  passed independent pre-reveal review. It binds the complete P4D2 chain,
  terminal policy, nine frozen thresholds and zero margins, per-unit/global
  decision semantics, exact runtime order, privacy denials, and five-record
  evaluation split.
- Reveal `99c28df0d50610523845684878cdeea05428451f3bc63af855011a6b40efa0d9`
  confirmed 5/5 disjoint recordings/conversations, 13 eligible opaque person
  labels, and one same-person label pair.
- The first preparation call failed before audio access because the frozen P2
  module did not support `intended_split=evaluation`. The attempted seam changed
  module SHA from authority-bound `467627bc...` to `96946fcc...`; it was
  immediately reverted byte-for-byte. Calibration and P4E authorities replay.
- Per terminal policy and independent review, no post-reveal replacement
  authority, wrapper, fallback, or calibration mislabel can rehabilitate this
  generation. Terminal STOP application
  `3ee5593c52a9193e056f99002fadd86c10b29bf1f84461b196132fc3d1222c41`
  records integrity failure and the required preparation path not run.
- Exact execution counts: 0 preparation receipts, 0 selected windows, 0 audio
  preparation, 0 model executions, 0 trials/scores, and 0 evaluation metrics.
  Replay and repeat are metadata-only, do not reopen the split receipt body,
  and return the same decision hash.
- Validation: 51 focused verification tests and 542 full repository tests
  passed; compilation and diff checks passed. Independent read-only audit
  returned `PASS` after exact receipt reconstruction, guarded no-split-body
  replay, stage-tree checks, and idempotent repeat.

Next:

- Design a successor P4E generation with the evaluation seam present before
  authority freeze and a genuinely new sealed conversation-disjoint cohort.
- Treat this revealed cohort only as nonblind diagnostic/refinement evidence.
- Do not select a model/method, integrate defaults, or resume historical
  reprocessing from this stopped generation.

## Turn 264 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 9

Packet: P4D2 | Held-out calibration and threshold freeze

State transition: `OPEN/Plan-0037-P4D2 -> CLOSED/Plan-0037-P4D2`.

- Preserved blocked calibration authority generation 1 after stereo source
  discovery, then built generation-2 authority
  `0fe6009bef2adfc9c48d87eea7d4ac15c00734ec45376ba3dbba45952e42fae5`
  with an exact mono/stereo channel policy and unchanged evaluation seal.
- Prepared all 3 calibration recordings through all 5 P1/P2 methods (15/15),
  receipt `8dc66610d82dd3545cc11998e7441c172a4d44dd8bd8e3edc4d57ab102397ae9`.
  Window receipt `8798e234ac2aacf57369f4e1e50ca2a1715fb7242c752348fef1f08fa7afd5f9`
  froze 22 clean windows across 8 opaque speakers before scoring and excluded
  mixed/unknown gold.
- Score matrix `9bca1c323a4681536dffada1399fe591152c132e9e9073d299531d7ebed6fccb`
  completed 396/396 finite trials: 81 genuine, 315 impostor, and 234 open-set
  trials across three models and no-enhancement/DeepFilterNet/RNNoise.
- Threshold application
  `c00df454c799e5afa3993dec01c4f021e9236ced109b9bfcd6a44685a3f6a05b`
  froze nine thresholds and descriptive metrics using the authority's exact
  selection order. Replay recomputed the same thresholds/metrics from scores;
  repeat apply returned the same hash without audio or model execution.
- Every current metadata receipt is `0600`, has no forbidden private payload,
  and records evaluation unread. Evaluation remains sealed and has no artifact.
- Validation: 95 focused tests and 540 full repository tests passed;
  compilation and diff checks passed. Independent read-only audit returned
  `PASS` after recomputing all stage identities, coverage, threshold/metric
  replay, condition/open-set reporting, permissions, and the evaluation seal.

Next:

- Build the separately exact P4E terminal-evaluation authority. Do not reveal
  evaluation before its frozen policy and apply receipt replay.

## Turn 263 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 8

Packet: P4D | Development resubstitution diagnostics

State transition: `OPEN/Plan-0037-P4D -> CLOSED/Plan-0037-P4D`.

- Added exact development trial authority, apply, and semantic replay seams.
  Authority SHA-256
  `b2bc390d71ef51230a81a2c1e1896f916be60f3e68bfffdcd17ff0dac329fd65`
  binds the P4C application/proposal, development split, P2 comparison,
  sources, method outputs/equivalence classes, profiles, and model revisions.
- Labeled the packet as non-held-out resubstitution evidence because all probe
  segments overlap enrollment. The receipts prohibit generalization,
  accuracy/FAR/FRR/EER, threshold, calibration, and model-selection claims.
- Live application SHA-256
  `6b1a06971279785b02d99e3e42f09536c6c0e85e634a949801cfeb6e7c1d5f8a`
  completed all 450 logical trials: 225 genuine, 225 impostor, 45 unique probe
  waveforms, and 270 unique waveform/model/profile combinations. Duplicate PCM
  across no-enhancement, Silero VAD, and Pyannote is not counted as independent
  acoustic evidence.
- Replay rechecks exact trial identities/coverage, finite score bounds, current
  P4/P3 eligibility, and all sealing/privacy flags. Numeric scores are
  structurally replayed, not recomputed. A repeat apply reused the same receipt
  without reopening PCM or rerunning models.
- Receipts are private `0600`, explicitly contain derived biometric scores,
  and contain no raw audio, transcript text, embeddings, vectors, or portable
  raw biometric values. Calibration and evaluation were not read.
- Validation: 68 focused P3/P4 tests and 535 full repository tests passed.
  Compilation and diff checks passed. Independent checkpoint re-audit returned
  `PASS` with no remaining finding.

Next:

- Build the exact P4D2 held-out calibration apply authority from the verified
  P4D receipt. Keep evaluation sealed.

## Turn 262 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 7

Packet: P4C | Exact P3 references and real profile apply

State transition: `READY_FOR_REVIEW/Plan-0037-P4C -> CLOSED/Plan-0037-P4C`.

- Resolved the operator's blanket proceed instruction into exact P3 create
  approvals for only the two frozen operator-gold candidate/source sets. Both
  production references are active, replay-eligible, and non-synthetic.
- Ready preview SHA-256
  `ca9a8ee8daa2ebe2e3b466dcb676fe4de472a5604af7ec3496aa541215ce0f93`
  and apply-authority SHA-256
  `ac7884adacb9acb665cc1de3686e7bc6bed227a567a60f3451bb1b5f5eb77614`
  bind the exact P3 generations, 15 source segments, three pinned models, and
  no-enhancement development-only preparation.
- Added a real enrollment apply/replay seam. It validates P2 lineage and method
  receipts, PCM path/hash/format/bounds, current P3 generations, model
  revisions, and descendant promotion before persisting metadata-only
  receipts. It refuses custom adapters outside deterministic tests.
- Live application SHA-256
  `9ec7fe5abc04461a740e224ef2c239760b4ffdf34654b4a845aea9f0a608953a`
  activated six profiles for two people across SpeechBrain ECAPA, WeSpeaker
  CAM++, and WeSpeaker ResNet34. Exact replay and a repaired idempotent rerun
  returned the same application and profile IDs.
- Application replay reconstructs the full authority-bound receipt, requires
  exact ordered person/model/P3/preparation coverage, and rejects independently
  rehashed proposal, preview, split, trial, calibration/evaluation, external,
  and raw-biometric semantic forgeries.
- The first rerun exposed replay-only fields in the application identity. The
  repaired core returns one stable public profile shape. The noncanonical
  duplicate receipt was preserved under private `rejected-noncanonical/`
  quarantine; no profile or biometric bytes were removed.
- No trial, verification score, calibration/evaluation read, external write,
  raw-biometric portability, or default integration occurred.
- Validation: 67 focused P3/P4 tests and 534 full repository tests passed;
  compilation and diff checks passed. Independent checkpoint re-audit returned
  `PASS` after reproducing the prior semantic-forgery rejection and exact live
  canonical replay.

Next:

- Run the independently reviewed P4D development packet across the three
  pinned models and authorized preparation paths. Keep calibration and
  evaluation sealed.

## Turn 261 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 6

Packet: P4C | Reviewed-clue continuity recovery

State transition: `BLOCKED/Plan-0037-P4C-CANDIDATES -> READY_FOR_REVIEW/Plan-0037-P4C-CANDIDATES`.

- Recovered the two raw-file hash drifts without weakening identity evidence.
  A committed metadata-only continuity authority at SHA-256
  `4c952608568edea918265f0851e89f4abfec2f41ac3faf590aaca20cb10da868`
  independently binds the frozen campaign authorities, blind predictions,
  completed run ledgers, prompts, statuses, and clue-discovery packets.
  Candidate preparation accepts only
  clue utterances whose ordinal ID, speaker label, start/end timestamp, and
  bounded text exactly match the current transcript.
- Proposal schema v2 persists only witness hashes and clue-projection hashes;
  no transcript text enters the proposal. Semantic replay recomputes the full
  witness selection and rejects packet, timestamp, speaker, text, or receipt
  drift.
- Live proposal SHA-256
  `aaec42150a2cc9f81212b7d965682a220202a71af3ad203fae0d7f122c6583a4`
  is `ready_for_operator_review`: all three selected development recordings,
  two opaque candidates, five sessions, 15 windows, and 180.755531 selected
  seconds. The `0600` artifact remains explicitly non-authorizing.
- No audio, P3 store/reference, profile, embedding, model inference, trial,
  calibration/evaluation read, or external write occurred.
- Validation: 62 focused P3/P4 tests and 529 full repository tests passed;
  compilation and diff checks passed. Tests reject pre-build clue-packet,
  blind-prediction, and run-ledger drift plus post-build replay drift.
  Independent checkpoint re-audit returned `PASS` and reproduced the focused
  and full validation plus the exact live replay.

Next:

- Real P3/P4 apply still requires an exact biometric-purpose decision over the proposed
  opaque people and source-set hashes; calibration and evaluation remain
  sealed.

## Turn 260 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 5

Packet: P4C | Metadata-only candidate enrollment proposal

State transition: `OPEN/Plan-0037-P4C-CANDIDATES -> CLOSED/Plan-0037-P4C-CANDIDATES`.

- Replayed the exact frozen development split/P0 manifest, P2 v5 joined
  receipt, no-enhancement lineage, frozen reviewed/current transcript artifact
  hashes, operator-gold person rows, and Pyannote speech/overlap/change metadata
  without opening audio or retaining transcript text.
- Added a deterministic private candidate-proposal contract. Windows are
  development-only, non-overlapped, outside speaker-change regions, 0.75-15
  seconds, capped at three per person/conversation after all labels are grouped,
  and require at least two conversations per candidate. P3 source-set
  validation replays every proposed production lineage receipt.
- Live proposal SHA-256
  `9bf0bcc08b2855ffaa1413d61d8015af4ed529ed712e8c6c7334f3b3b43bf2ce`
  is truthfully `blocked`: two of three selected current transcript artifacts
  differ from their frozen operator-reviewed hashes, and the one exact record
  cannot supply a multi-session candidate. The `0600` artifact is explicitly
  non-authorizing and contains zero candidates, transcript text, raw biometric
  values, or enrollment audio.
- No P3 store/reference, profile, embedding, model inference, trial,
  calibration/evaluation read, or external write occurred.
- Validation: 61 focused P3/P4 tests and 528 full repository tests passed;
  compilation and diff checks passed. Read-only checkpoint audit is pending.

Next:

- Independently audit and commit/push the candidate proposal. The next human
  evidence gate is review of the two changed current transcript artifacts;
  exact candidate/source-set approval remains a later biometric-purpose gate.
  Calibration and evaluation remain sealed.

## Turn 259 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 4

Packet: P4C | No-audio real-enrollment preview

State transition: `OPEN/Plan-0037-P4C-PREVIEW -> CLOSED/Plan-0037-P4C-PREVIEW`
with a persisted blocked outcome.

- Added a private content-addressed enrollment-preview and replay contract. A
  ready development preview binds exact P3 generation/source-set/approval,
  segment/lineage hashes, opaque identifiers, all three model revisions, and
  the exact frozen split/P0-manifest hashes. Every recording/conversation pair
  must prove membership in the hashed development set. Build and replay use
  one strict semantic validator; synthetic fixtures, forged content-addressed
  payloads, and calibration/evaluation scope fail closed.
- Live readback found no canonical P3 reference database and no requested
  approved opaque people. Persisted the truthful no-audio blocker at SHA-256
  `30b6f33fb280daa8020fc79fcec4e82fe6c2a8930fc920399f31b0f13ff1e1a3`.
  No corpus audio, model inference, profile materialization, reference
  registration, trial, calibration, evaluation, or external write occurred.
- Validation: 56 focused P3/P4 tests and 523 full repository tests passed;
  compilation and diff checks passed. The read-only checkpoint re-audit
  returned `PASS` after the split-membership, semantic-replay, and exact
  reason/fact repairs.

Next:

- Review and commit/push the P4C preview checkpoint. P4C apply and P4D stay
  gated until real sources have explicit P3 biometric-purpose approval and an
  exact reviewed enrollment manifest/apply authority is supplied.

## Turn 258 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 4

Packet: P4B | Host adapters and synthetic profile lifecycle

State transition: `OPEN/Plan-0037-P4B -> CLOSED/Plan-0037-P4B`.

- Added one lazy host adapter contract for the exact acquired SpeechBrain
  ECAPA-TDNN, WeSpeaker CAM++, and WeSpeaker ResNet34 snapshots plus a
  deterministic fake adapter. Waveforms and vectors must be finite, bounded,
  dimensionally exact, and L2-normalized; short, silent, unavailable-model,
  OOM, and malformed inputs fail closed.
- Made SpeechBrain loading fully local by overriding the upstream YAML's remote
  pretrained path. `HF_HUB_OFFLINE=1` synthetic smoke loaded all three real
  adapters and returned 192/512/256-dimensional normalized vectors.
- Added private profile materialization bound to a replay-eligible P3 generation,
  exact model/preprocessing revisions, opaque sessions, window hashes, counts,
  and dispersion. The aggregate is a `0600` binary under `0700`; raw vectors,
  waveforms, names, email, and transcript text are excluded from receipts.
- Enforced `stage -> P3 register -> P3 promote -> P4 active`. Added a narrow P3
  descendant-invalidation request so P4 supersede/withdraw can disable scoring
  first and then complete P3 request/ack. Delete verifies ineligibility, removes
  private bytes, and retains only a non-biometric tombstone.
- Scoring validates immutable lifecycle evidence and the private artifact hash,
  then checks both P4 state and live P3 eligibility before and after inference.
  Normal replay is idempotent; blob/state tamper and mid-score revocation fail.
- The first P4B checkpoint audit returned `REFINE` on four exact gaps. Repaired
  them by replaying the exact P4A manifest and file hashes before every model
  load; making the public P4B materializer synthetic-fixture-only; adding an
  immutable full profile manifest and six metadata-tamper cases; and deriving
  withdraw/supersede acknowledgment receipts deterministically for replay and
  partial-failure recovery. Real-P3 retry smoke passed.
- Joined synthetic P3/P4 smoke ran all three real adapters without real audio:
  SpeechBrain score `0.892918`, CAM++ `0.994622`, ResNet34 `0.995596`. These are
  execution-only synthetic values and do not establish thresholds or quality.
- Validation: 47 focused P3/P4 tests, 101 joined acoustic tests, and 514 full
  tests passed; compilation and diff checks passed. The read-only checkpoint
  re-audit returned `PASS` on all four repaired blockers.

Next:

- Commit/push the reconciled P4B slice, then build P4C's no-audio exact
  real-enrollment preview. Do not access real source audio or
  materialize a real profile until that separately reviewed manifest is
  explicitly authorized.

## Turn 257 | 2026-07-31

Plan:
`docs/dev/plans/0042-2026-07-31-plan-0037-p4-verification-calibration.md`,
version 3

Packet: P4A | Verification/calibration design and acquisition readiness

State transition: `NOT-STARTED/Plan-0037-P4 -> OPEN/Plan-0037-P4`.

- Re-anchored the resumed Plan 0037 goal on `VISION.md`, Plan 0037, current
  policies, ROADMAP/RUNBOOK, pushed commit `70f3f24`, local runtime evidence,
  CodeGraph, and advisory Graphiti recall. P0-P3 are closed; P4 is the next
  serialized join and P5-P7 remain dependency-blocked.
- Opened Plan 0042 for private model-specific profile materialization,
  conversation-separated verification/calibration, and a sealed terminal
  model decision. Reconciled stale parent text that still described P2 as open
  or implied its deferred downstream measurements had run.
- Selected SpeechBrain ECAPA-TDNN, WeSpeaker CAM++, and WeSpeaker ResNet34 as
  the required first comparison. Pinned SpeechBrain package/model, WeSpeaker
  code, and both WeSpeaker model revisions plus upstream-published LFS SHA-256
  values in
  `docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json`.
- Kept code and checkpoint terms distinct: official WeSpeaker documentation
  says VoxCeleb checkpoints follow CC-BY-4.0 even though the hosting model cards
  display Apache-2.0. Missing upstream SHA-256 values for small configs must be
  computed and bound before load.
- Preserved the consequential authority boundary: the standing Plan 0037 grant
  covers acquisition/install and bounded development processing, but ordinary
  gold/contact/calendar/speaker confirmation is not biometric enrollment
  consent. Real reference registration remains behind an exact separately
  reviewed enrollment manifest/apply authorization.
- The initial read-only design audit returned `REFINE`. Repaired its four
  blockers by hash-binding WeSpeaker code and checkpoint-terms authorities,
  freezing hashed split-access transitions, freezing a pre-reveal terminal
  decision policy, and specifying P4 supersede/withdraw/delete byte-removal,
  non-biometric tombstone, and dual P4/P3 score-time checks. Calibration and
  evaluation remain unauthorized and sealed pending their exact later apply
  authorities.
- Added the immutable, side-effect-free `acoustic_verification.py` acquisition
  dry-run/replay surface. It rejects spec drift, mutable or incorrect authority
  bindings, enrollment authority injection, side-effect flags, and private-mode
  violations. The reviewed live dry-run will be regenerated after design
  approval because the repaired acquisition fixture has a new hash.
- The repaired independent design audit passed. Generated and replayed live
  dry-run `acquire-verification-489bdb9d743f150f1909e042`, binding spec
  SHA-256 `c6cc78b265eed77b5b52637765dc3cde07a74e99b1ef7fde6328a15ae1345c1c`
  and dry-run SHA-256
  `6b47964dea6a2caa65a73a23c1561267d1a22f89575a29685465ebada580af8c`.
- Acquired all three public, revision-pinned model snapshots plus the exact
  SpeechBrain wheel and immutable WeSpeaker code/terms authorities under the
  private P4 runtime. Installed SpeechBrain `1.1.0`; ONNX Runtime remains
  `1.24.4`. Acquisition manifest SHA-256 is
  `6470ecc8591fd8a40f8d788ba9a3edddc37a508cc54d47800037ab594b957ebe`.
  Independent readback verified 12 artifact hashes and private modes. No audio
  was read, no profile/embedding was materialized, and no trial ran.

Next:

- Add fail-closed acquisition readiness plus the P4B host adapter/profile seam,
  then prove it with deterministic fake models before any real enrollment.

## Turn 256 | 2026-07-31

Plan:
`docs/dev/plans/0040-2026-07-31-plan-0037-p2-speech-preparation-comparison.md`,
version 3

Packet: P2D-P2E | Community-1 acquisition and joined terminal comparison

State transition: `OPEN/Plan-0037-P2 -> CLOSED/Plan-0037-P2`.

- The operator identified `~/credentials/API-keys.env` as the Hugging Face
  credential source. Authentication and gated repository access were verified
  without printing or persisting token values.
- Acquired the complete private Community-1 snapshot at exact revision
  `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`. Immutable acquisition manifest
  SHA-256 is
  `b3fd1614b3f233fa0b2e0bece0dfd88aaa9063e6f864b5298a7cf86effdaca10`;
  readiness re-hashes all ten assets and verifies private modes before use.
- Added the real host-owned Community-1 adapter. It loads only the pinned local
  snapshot, supplies in-memory 16 kHz PCM to bypass unavailable TorchCodec
  decoding, and exports only normalized speech, overlap, and speaker-change
  regions. Provider labels, objects, waveforms, and credentials do not escape
  into portable receipts.
- The second development recording exposed a provider segment extending past
  the authoritative P1 duration. One bounded repair clips provider turns to
  `[0, duration]` before union/overlap normalization; the failed recording and
  the remaining cohort then passed.
- Final joined receipt:
  `~/.local/state/transcribe-audio/plan-0037/speech-preparation/development-comparison-20260731-v5/development-comparison.json`,
  SHA-256
  `0b3c68a31cbf7bc7f80d5302a52c8c7630414ca198cef78223b63baedbfd0ac3`.
  The deterministic three-shortest development slice totals 2,892 seconds;
  calibration/evaluation selected counts are zero; all 15/15 preparation
  attempts and all private comparison/output replays succeeded.
- Community-1 measured `167/15/61`, `202/8/82`, and `192/5/61`
  speech/overlap/change regions across the three recordings. These are
  preparation observations only; downstream transcription, diarization
  evaluation, verification, and method selection remain
  `blocked/not_run_downstream_measurements` with zero attempts.
- Independent review found v4's per-recording transcription/diarization fields
  still used the obsolete `not_run_dependency_real_methods` reason. Source and
  regression coverage were repaired, all three immutable comparisons were
  regenerated under v5, and the v5 aggregate above supersedes v4. All 19 v5
  files are `0600` and all 26 directories are `0700`.

Validation and closeout:

- Focused speech-preparation tests: 21 passed; the declared joined focused
  command passed 51 tests.
- Full repository suite: 488 passed. Focused compile and `git diff --check`
  also passed. Reused read-only reviewer `/root/p1_review_final`; the terminal
  audit returned `PASS` after the v5 reason-code and directory-mode repairs,
  with no reviewer edits.

## Turn 255 | 2026-07-31

Plan:
`docs/dev/plans/0040-2026-07-31-plan-0037-p2-speech-preparation-comparison.md`,
version 2

Packet: P2C-P2D | Standing authorization and open-candidate execution

State transition: `OPEN/Plan-0037-P2 -> OPEN/Plan-0037-P2`.

- The operator rejected per-run authorization incantations and supplied a
  blanket grant for the full bounded Plan 0037 scope. Dry-run and content
  hashes remain audit/integrity evidence, but no P2 acquisition, apply, or
  rollback token is required.
- Downloaded pinned Silero VAD 6.2.1, DeepFilterNet/DeepFilterLib 0.5.6,
  DeepFilterNet3, and signed RNNoise v0.2 artifacts into the private P2 root.
  Verified published hashes and bound locally computed SHA-256 values
  `49c52edc...84d2` for DeepFilterNet3 and `90fce4b0...0d37` for RNNoise.
- Built DeepFilterLib for CPython 3.12 and RNNoise into private runtime paths.
  Bound every download, model, config, and RNNoise library in immutable
  acquisition manifest SHA-256
  `fc28406a6c2a8a84763a238940d0cec29a414e1d7952d74d69c9f597fdbe1d13`.
  This v2 manifest supersedes the first manifest after source inspection proved
  Silero opset 16 loads `silero_vad.onnx` (SHA-256 `1a153a22...8e3`), not the
  initially bound op15 file.
- Added real host-owned Silero, DeepFilterNet, and RNNoise adapters. They decode
  PCM directly, keep provider objects out of receipts, preserve full original
  time, write private content-addressed enhanced WAVs, and require verified
  model or library hashes before readiness succeeds.
- Production synthetic run `speech-prep-8bec5774c352cf95296b213b` completed
  with no-enhancement, DeepFilterNet, and RNNoise successful; Silero truthfully
  abstained on non-speech tone; pyannote remained unattempted. Replay verified
  comparison SHA-256
  `8e18c736cae9535d7a0be37c6042a815a1e1fd56c719289e822a40e3e88921a7`.
- Ran a deterministic development-only slice: the three shortest frozen
  development recordings (2,892 seconds total). Calibration/evaluation
  selected counts remained zero. All 12 open-method attempts succeeded:
  no-enhancement, Silero, DeepFilterNet, and RNNoise on each recording;
  Silero measured 184, 248, and 321 speech regions.
- The second recording exposed DeepFilterNet's unsupported one-shot CUDA GRU
  path on a 19-minute tensor. One permitted repair changed enhancement to
  contiguous 60-second chunks with full-length concatenation; the retry and
  the third 19.5-minute recording then passed. Immutable aggregate receipt
  SHA-256 is
  `81aa1b407798409f2b4871f3eb5f0673de540ebab537ee6acb51570e09ce21fc`.
- Terminal review found enhanced-output replay originally trusted receipt
  hashes without reopening WAV bytes. The repaired v2 adapters now store
  outputs at SHA-256-addressed paths and replay requires private containment,
  `0600` mode, file existence, content-address equality, and byte-hash match.
  The v3 receipt above supersedes the stale `a7304d...` receipt and binds the
  corrected Silero manifest plus all regenerated comparisons.
- Community-1 terms/contact sharing and bounded development processing are
  covered by the standing grant. The Hugging Face client is unauthenticated
  with no token, and the cached Community-1 snapshot contains only two PLDA
  files, so P2D is `blocked/provider_auth_required` pending provider access.

Next:

- Acquire and run Community-1 immediately if provider authentication becomes
  available, then complete the five-method joined comparison and terminal
  review. Calibration/evaluation remain sealed.

## Turn 254 | 2026-07-31

Plan:
`docs/dev/plans/0040-2026-07-31-plan-0037-p2-speech-preparation-comparison.md`,
version 1

Packet: P2C | Open-candidate acquisition planning

State transition: `OPEN/Plan-0037-P2 -> OPEN/Plan-0037-P2`.

- Added the exact open-candidate acquisition spec at
  `docs/dev/fixtures/plan-0037-p2/open-candidate-acquisition-plan.json` for
  Silero VAD `6.2.1`, DeepFilterNet/DeepFilterLib `0.5.6`, DeepFilterNet3, and
  signed RNNoise `v0.2`.
- Bound official package URLs, versions, sizes, SHA-256 values, source commits,
  RNNoise signed tag identity, licenses, and terms sources. DeepFilterNet3 and
  RNNoise source lack upstream SHA-256 values, so the spec requires computing
  and content-addressing them after authorized download and before build/use.
- Recorded the live Python 3.12 compatibility constraint: DeepFilterLib has no
  CPython 3.12 wheel at `0.5.6`, while the host has the Rust/native build tool
  chain needed for an explicitly authorized private source build.
- Added immutable private dry-run/replay functions. They exclude pyannote
  terms acceptance, contact sharing, private audio, development-cohort apply,
  and biometrics; every mutation flag is false. Replay requires the originally
  reviewed byte-level plan SHA-256 before returning the same approval token.
- Persisted and replayed plan `acquire-open-585ef49febe61caf5a3d99b1`, SHA-256
  `d4b2a4c800b10cd8604b4e2f73ac553a097652f0bc1271ff27def5628c9ac836`,
  under the private P2 runtime root. No package/model/source archive was
  downloaded, installed, built, or loaded.

Validation and review:

- 66 focused acoustic P1/P2/P3 and identity-contract tests passed.
- The full repository suite passed with 482 tests.
- `python -m py_compile` and `git diff --check` passed.
- Read-only reviewer `/root/p1_review_final` returned terminal planner `PASS`
  after official-metadata, exclusion, permission, spec-drift, timestamp-tamper,
  serialization-tamper, and no-side-effect review.
- Graphiti was healthy but returned no useful P2C recall; current plans,
  installed-state readbacks, official package/release metadata, and the new
  repo fixture controlled.

Next (superseded by Turn 255):

- The operator replaced the former token gate with standing Plan 0037
  authorization; Turn 255 records the executed acquisition.

## Turn 253 | 2026-07-31

Plan:
`docs/dev/plans/0041-2026-07-31-plan-0037-p3-biometric-reference-library.md`,
version 2

Packet: P3 | Biometric reference library

State transition: `OPEN/Plan-0037-P3 -> CLOSED/Plan-0037-P3`.

- Added `acoustic_biometric_references.py`, a private reference-only authority
  that stores opaque identities, exact original-time source segments,
  biometric-purpose approvals, immutable generations/events, source claims,
  action-specific approval tokens, CAS heads, and minimized deletion
  tombstones without reading audio or creating scoring profiles.
- Added metadata-only P1/P2 lineage resolvers. Production sources must bind
  replay-validated source hash, original duration, quality evidence, and
  derivative/comparison receipts; synthetic inputs require explicit test-only
  fixture authority and cannot enter production mode.
- Added immutable run-local source manifests, staged-before-commit and
  authoritative-after-commit receipt publication with recovery, append-only
  inventory checks, exact schema replay, coordinated manifest/tombstone tamper
  detection, and filesystem-read-only public eligibility queries.
- Added an independent P4 authority contract for materialization, promotion,
  and invalidation receipts. Withdrawal is required before deleting a profile
  with descendants, and deletion blocks until every invalidation is
  acknowledged.
- Kept the P3/P4 boundary exact: P3 exposes only
  `eligible_for_materialization`; P4 exclusively owns embeddings, dispersion,
  calibration, scoring, and `biometric_profile.v1`.

Validation and review:

- 61 focused acoustic P0/P1/P2/P3 and identity-contract tests passed.
- The full repository suite passed with 477 tests.
- `python -m py_compile` and `git diff --check` passed.
- Persistent synthetic smoke root:
  `~/.local/state/transcribe-audio/plan-0037/p3-smoke-final-v2/`; create,
  register/promote, supersede/invalidate, withdraw/invalidate,
  delete/tombstone, and replay passed. All 37 files are `0600`; all seven
  governed directories are `0700`.
- Reused read-only reviewer `/root/p1_review_final` returned terminal `PASS`
  after three bounded repair audits. No private corpus/audio/model/embedding
  asset was accessed and no real source was enrolled.

Next:

- Keep P4 blocked until P2 supplies an approved successful preparation method
  and acquisition/development-cohort gates are explicitly satisfied. Advance
  the next independent Plan 0037 lane only if its dependencies do not cross
  those gates.

## Turn 252 | 2026-07-31

Plan:
`docs/dev/plans/0041-2026-07-31-plan-0037-p3-biometric-reference-library.md`,
version 2

Packet: P3 | Biometric reference library

State transition: `NOT-STARTED/Plan-0037-P3 -> OPEN/Plan-0037-P3`.

- Opened P3 after the pushed P2B checkpoint because P3 depends on P0/P1 and
  can advance without crossing P2 model-acquisition or pyannote human gates.
- Scoped P3 to a restricted user-scoped reference store and synthetic
  lifecycle proof. No P0 corpus audio, real source registration, embeddings,
  named-person
  scoring, model acquisition, or external write is authorized.
- The design audit found the frozen P0 `biometric_profile.v1` cannot represent
  reference-only state: every non-deleted object requires model/preprocessing
  revisions and a private embedding, while active means scoring-eligible.
- Revised Plan 0041 to version 2 with a distinct
  `biometric-reference-profile.v1` authority. P3 owns biometric-purpose
  approval, source references, immutable generations, CAS, lifecycle, and P4
  descendant invalidation; P4 alone owns embeddings, dispersion,
  materialized scoring profiles, and calibration.
- Graphiti runtime and MCP were healthy. Repo group `transcribe_audio_main`
  returned advisory Plan 0025 facts that model speaker output is human-review
  only; current Plan 0037, P0 contracts, source, and tests control.
- CodeGraph showed the frozen scoring-profile validator and confirmed that no
  distinct reference schema, durable generation/head store, biometric-purpose
  approval, CAS, resolver, or P3-to-P4 invalidation contract exists.
- Reused read-only reviewer `/root/p1_review_final` for one P3 design report
  and the later terminal audit. The primary owns every edit and synthetic
  private artifact.

Next:

- Implement the private synthetic reference create/replay/supersede/withdraw/
  delete lifecycle, descendant invalidation contract, CAS, and adversarial
  tests.

## Turn 251 | 2026-07-31

Plan:
`docs/dev/plans/0040-2026-07-31-plan-0037-p2-speech-preparation-comparison.md`,
version 1

Packet: P2 | Speech preparation comparison

State transition: `CLOSED/Plan-0037-P1 -> OPEN/Plan-0037-P2`.

- Opened a bounded P2 packet for internal no-enhancement, Silero VAD,
  DeepFilterNet, RNNoise, and pyannote preparation adapters plus replayable
  private comparison evidence.
- Frozen calibration/evaluation recordings, biometric enrollment/verification,
  historical reprocessing, default-pipeline changes, App Intelligence, and
  external writes remain outside P2.
- Current venv readback has PyTorch 2.11.0, torchaudio 2.11.0, onnxruntime
  1.24.4, and pyannote.audio 4.0.4. Silero/DeepFilterNet packages and an RNNoise
  executable are absent.
- The P0 inventory says all P2 model assets are unacquired. Existing incomplete
  pyannote cache fragments are not acquisition or authorization evidence;
  Community-1 remains behind an explicit human terms/contact-sharing gate.
- Graphiti was healthy but returned only advisory older preprocessing facts.
  Current plans, inventory, installed state, hashes, and receipts control.
- Reused read-only reviewer `/root/p1_review_final` for P2 design and terminal
  audit; the primary owns all edits, downloads, and private processing.

P2B checkpoint:

- Added `acoustic_speech_preparation.py` with a normalized five-method
  readiness matrix and stable no-enhancement, fake-test, dry-run, apply,
  replay, and rollback interfaces.
- Bound P2 plans and receipts to replay-verified P1 manifests, artifacts,
  timestamp maps, readiness assets, acquisition-manifest hashes, adapter
  identities, and run-specific approval tokens.
- Kept lifecycle status separate from comparison outcome: apply records
  `success/applied`, while the comparison remains
  `blocked/required_real_comparisons_not_run` with denominator
  `methods=5, attempted=1, success=1, failure=0, blocked=4`.
- Enforced private immutable artifacts, finite and role-specific timing proof,
  forbidden portable payload families, test-only fake adapters, tamper-aware
  replay, idempotence, and non-destructive revocation.
- Added a public P1 active-derivative resolver whose manifest is returned by
  the same replay operation that validates it, avoiding a validation/reopen
  race.

Validation and review:

- 40 focused P0/P1/P2 tests passed; the full repository suite passed with 456
  tests.
- Final synthetic smoke root:
  `~/.local/state/transcribe-audio/plan-0037/p2b-smoke-final`; P1 run
  `audio-run-b37e04a6706ef0ab4e3099ea`; P2 run
  `speech-prep-f312247c2ba9a601ac38a9a8`.
- First apply, idempotent apply, active replay, rollback, and inactive replay
  passed. All 15 files are `0600`; all 10 directories are `0700`.
- Read-only reviewer `/root/p1_review_final` returned terminal P2B PASS after
  independent adversarial timing, readiness, privacy, and lifecycle probes.
- No model was downloaded or installed and no P0 private corpus recording was
  read. P2 remains open.

Next:

- Persist and push this truthful P2B checkpoint. Then advance an independent
  Plan 0037 lane while P2C open-model acquisition and P2D pyannote execution
  remain behind their explicit approval gates.

## Turn 250 | 2026-07-31

Plan:
`docs/dev/plans/0039-2026-07-31-plan-0037-p1-audio-derivatives-quality.md`,
version 1

Packet: P1 | Audio derivatives and quality

State transition: `OPEN/Plan-0037-P1 -> CLOSED/Plan-0037-P1`.

- Added a private audio-derivative module with source blob/hash binding,
  resolved `ffmpeg`/`ffprobe` paths and full versions, a canonical no-shell
  decode recipe, mono-only policy, 16 kHz signed-16 PCM output, and bounded
  execution.
- Added no-clobber content-addressed publication, full source-to-output
  timestamp coverage, deterministic peak/RMS/DC/clipping/exact-zero-silence
  measures, and explicit `usable_speech_not_assessed_until_p2` abstention.
- Dry-run, apply, replay, and rollback now enforce exact tokens, immutable
  evidence bindings, source preservation, runtime containment, symlink
  rejection, `0600` files, `0700` directories, idempotence, and revocation.
  Rollback deletes nothing and a revoked run cannot reactivate.
- Kept VAD, enhancement, diarization, biometric models, enrollment, frozen
  corpus processing, App Intelligence, and provider writes outside P1.

Terminal synthetic smoke:

- Root:
  `/home/ecochran76/.local/state/transcribe-audio/plan-0037/p1-smoke-terminal`.
- Run: `audio-run-7895c2d83afd287f79855eaa`; manifest SHA-256:
  `696b02837b9c12877263dab493661cc668ab73c6bc6742ac97664bca3f147299`.
- Source/output SHA-256:
  `6eb50df72cf53112487c47897be623c1a867fa4aef8b6855ffb74b614f174d32`
  and
  `4c1a1a10f5ccc84fbb302bece43de92db4c71c02ef93c0d10ddce7e87bad05fe`.
- Apply, repeated apply, active replay, rollback, repeated rollback, and
  inactive replay passed. The map was 2.0 seconds to 2.0 seconds with zero
  drift. The source stayed unchanged; the retained run is inactive.
- All 11 retained files are `0600`; all 7 directories are `0700`.

Validation and review:

- 30 focused contract/audio-derivative tests passed.
- 146 joined artifact/store, evaluation, identity-preparation, and workflow
  tests passed.
- Full suite: 446 passed.
- Active planning audit and `git diff --check` passed.
- Graphiti terminal readback was healthy and returned only advisory generic
  repo-planning facts; current repo authorities and runtime evidence controlled.
- Read-only reviewer `/root/p1_review_final` returned PASS after independent
  synthetic-smoke replay and adversarial permission, tamper, reuse, and
  rollback checks. No frozen private corpus material was inspected.

Next:

- Derive and execute Plan 0037 P2 for bounded VAD, enhancement, and
  diarization-preparation comparison without processing the frozen corpus by
  default.

## Turn 249 | 2026-07-31

Plans:

- `docs/dev/plans/0037-2026-07-31-audio-enhancement-biometric-speaker-identity.md`,
  P0 closed; P1 next.
- `docs/dev/plans/0038-2026-07-31-plan-0037-p0-contract-evaluation-freeze.md`,
  version 1, closed.

State transition: `OPEN/Plan-0037-P0 -> CLOSED/Plan-0037-P0`.

P0 contract and privacy freeze:

- Added six versioned acoustic artifact contracts. Portable artifacts reject
  recursive embedding/vector/audio payloads; biometric lifecycle, timestamp
  mapping, verification split, and reprocessing approval/non-overwrite rules
  fail closed.
- Added a seven-candidate code/checkpoint inventory. All dependencies and
  checkpoints remain unacquired, and exact revisions, hashes, dataset terms,
  and gated-access conditions remain promotion blockers.
- Found that the user-scoped transcript store contradicted the prior private
  blob premise: selected blobs were `0755`/`0777`, the root was `0755`, and the
  database was `0644`. Hardened the complete store to `0700` directories and
  `0600` files and made those modes persistent for new database, artifact, and
  blob writes.
- Bound corpus gold paths to the campaign gold directory and matched schema,
  gold/document/campaign/manifest identities, chronological rank, reviewed
  artifact hash, current transcript lineage, and immutable source-blob hash.

Private corpus:

- Corpus: `acoustic-corpus-1f93d1405f82676420571e1b`.
- Denominators: 24 recordings, 24 conversations, 35 pseudonymous subjects,
  105 labels, 293 feasible same-person pairs, and 2,042 feasible
  different-person pairs.
- Splits: 16 development, 3 calibration, and 5 evaluation recordings; no
  conversation crosses a split.
- Manifest SHA-256:
  `73f0e04aab0274ddfeaa7f6b1567ecb135eebc0a0d6e5818cb3bd2ee5535dabf`.
- Manifest and freeze receipt are `0600`; scoped directories are `0700`.
  Unchanged replay returned the same corpus identity and receipt.
- Acoustic conditions are explicit `unassessed_until_p1` or
  `unassessed_until_p2`; the corpus is not eligible for model promotion.

Delegation:

- Spawned read-only reviewer `/root/p0_audit`. The first pass found contract,
  privacy, idempotence, and benchmark gaps. The terminal pass found gold
  provenance, reprocessing binding, and planning-wiring blockers. The primary
  agent repaired and tested each finding before closure.

Validation:

- 39 focused contract/corpus/store tests passed.
- 122 transcript-artifact, speaker-evaluation, identity-preprocessing, and
  workflow regressions passed.
- Full suite: 423 passed in 28.15 seconds.
- Graphiti discovery returned no facts because FalkorDB on `127.0.0.1:6389`
  was down; repo authorities and live readbacks controlled.
- No acoustic model ran, no Plan 0036 prediction was revealed, no biometric
  enrollment occurred, and no external write occurred.

Branch:

- Created `plan-0037-campaign` from synchronized commit `8b4a1b1` so the branch
  scope matches the active campaign.

Next:

- Derive and execute Plan 0037 P1 for immutable audio derivatives, quality
  measurement, exact timestamp maps, and dry-run/apply/replay/rollback
  receipts.

## Turn 248 | 2026-07-31

Plans:

- `docs/dev/plans/0036-2026-07-30-literal-fts-blind-speaker-rerun.md`,
  version 1, checkpointed.
- `docs/dev/plans/0037-2026-07-31-audio-enhancement-biometric-speaker-identity.md`,
  version 1, opened.

State transition: `OPEN/Plan-0036-P3 -> PAUSED-CHECKPOINT/Plan-0036-P3` and
`unplanned acoustic prototype -> OPEN/Plan-0037-P0`.

Plan 0036 checkpoint:

- Recorded current gold for chronological ranks 25 through 29: five of ten
  cases in the sealed superseding baseline.
- The rank-29 gold ID is
  `bb899562-58cb-49b0-b5fd-b292ccb20e6c`. It records the operator-confirmed
  six-label mapping and wrong calendar association without exposing the sealed
  prediction.
- The next untouched review is rank 30, document
  `fd6d03afd42775704dc6`.
- Campaign status reports `gold_content_included=false`,
  `will_execute_app_intelligence=false`, and
  `will_perform_external_write=false`.
- Current gold artifacts remain private mode `0600`. The baseline remains
  complete and sealed. No prediction body was read, and no comparison was
  revealed.
- The operator paused the remaining five reviews before conversation 6.

Acoustic research and plan:

- Added a durable research note covering Silero VAD, DeepFilterNet, RNNoise,
  pyannote.audio, SpeechBrain ECAPA-TDNN, WeSpeaker, and NVIDIA TitaNet.
- Selected a host-owned `AcousticIdentityAnalyzer` direction. It returns
  bounded evidence and keeps model adapters, preprocessing, calibration, and
  biometric storage behind one deep interface.
- Required immutable original audio, versioned timestamp-aligned derivatives,
  local calibration, explicit abstention, and private raw embeddings.
- Opened Plan 0037 for contracts, audio derivatives, speech preparation,
  biometric enrollment, model bake-off, pipeline integration, historical
  reprocessing, and blind outcome measurement.
- Added P10 to the roadmap as the immediate dependency before context-assisted
  speaker identity continues.

Validation:

- Live campaign readback over `127.0.0.1:18876` confirmed the next review and
  excluded gold content.
- Verified private gold artifact permissions.
- Graphiti discovery was unavailable because FalkorDB on `127.0.0.1:6389`
  was down. Repo plans, runtime readbacks, and private artifacts remained the
  authority.

Next:

- Execute Plan 0037 P0. Freeze the schemas, private evaluation corpus,
  checkpoint-license inventory, privacy contract, and promotion metrics before
  adding processing dependencies or enrolling biometric references.

## Turn 247 | 2026-07-30

Plan:
`docs/dev/plans/0036-2026-07-30-literal-fts-blind-speaker-rerun.md`
version 1

Packet: P0 | Literal FTS query

State transition: `CLOSED/Plan-0035-REFINE -> OPEN/Plan-0036-P0`

Plan 0035 terminal evidence:

- P0 added and tested the evaluation-freeze bridge; 164 joined tests passed.
- P1 preflight receipt:
  `~/.local/state/transcribe-audio/plan-0035/preflight-9c2585df-ee7f-45be-add6-c32ed4f4277c.json`
  (`0600`), SHA-256
  `520b636db36f9bf0cf51fc5219f1f806971daeb9ad809c853eb07a257ca20360`.
- P2 baseline:
  `baseline-f77e1874-fbfb-4ff3-87fa-9b57e2de197f`.
  Four predictions are immutable and six cases remain unstarted.
- Case 3 first failed with `no such column: member`; the sole unchanged retry
  succeeded. Case 5 then failed with `no such column: like`.
- Both failures originate from unquoted hyphen-bearing tokens in the shared
  FTS prefix-query builder. The service stayed active with `NRestarts=0`.
- P3/P4 did not run. Gold-index hash
  `6560591461573bf08d50dd110c031d56f287ea570563b9ae0bfdae691d48d3d8`
  is unchanged, gold was not read or written, and no prediction was revealed.
- Terminal receipt:
  `~/.local/state/transcribe-audio/plan-0035/terminal-refine-2dd137bf-2575-4095-a645-0bc8d6d70fe7.json`
  (`0600`), SHA-256
  `afb1886c9965c6e8ce74fe0db12b45e8b664993bffdd1b9a8c0206fa15e752c9`.

Plan 0036 bounds:

- Quote tokens only; do not change token selection, retrieval policy,
  provider order, prompts, model, confidence, or candidates.
- Preserve the partial baseline without reveal.
- Create one explicitly linked superseding baseline for the exact same unseen
  cohort after pushed and served validation.
- One total unchanged case retry is available on the superseding baseline.

Next:

- Reproduce the hyphenated-term SQLite failure through the public evidence
  search interface.
- Quote FTS5 prefix tokens, prove GREEN, and run the joined regression suite.
- Add the approval-gated explicit supersession path before any new model turn.

P0/P1 implementation checkpoint:

- FTS RED:
  `.venv/bin/python -m pytest
  tests/test_conversation_knowledge_evidence.py::test_evidence_query_treats_hyphenated_prefix_terms_as_literals
  -q` failed with `sqlite3.OperationalError: no such column: member`.
- FTS GREEN: the same test passes after `transcript_store.fts_query` quotes
  every normalized token as an FTS5 literal phrase before applying `*`.
- Supersession RED:
  `.venv/bin/python -m pytest
  tests/test_conversation_knowledge_evaluation.py::test_evaluation_holdout_explicitly_supersedes_one_partial_baseline
  -q` failed because the public interface rejected
  `supersedes_baseline_id`.
- Supersession GREEN: the same test passes after the interface verifies one
  matching partially captured, unrevealed baseline and links the replacement
  through `parent_baseline_id`. Exact replay returns the same replacement.
- The focused transcript, evidence, retrieval, evaluation, campaign, and
  runner suite passes 64 tests.
- The joined transcript-store, adapter, retrieval, knowledge, workflow,
  campaign, and transcript API suite passes 194 tests.
- Python compilation and `git diff --check` pass.

P2 blind-rerun checkpoint:

- Pushed source commit:
  `fee6ef624e4449f15753074e8c0e292150cfd0b5`.
- Restarted `transcripts.service` once from pushed source. It is active with
  PID `2334963`, `NRestarts=0`, and `/api/health` reports `status=ok`.
- Created superseding baseline
  `baseline-65fdc53f-fc1a-4534-a88d-cf4b0563fbcc`, linked to partial baseline
  `baseline-f77e1874-fbfb-4ff3-87fa-9b57e2de197f`.
- The exact ten-case cohort completed 10/10 blind predictions with zero
  infrastructure retries. The formerly failing hyphenated-query case
  completed through the served literal-FTS repair.
- Completed baseline SHA-256:
  `6f86a58d74899d0de834a9d03e75585c696e02d4d4fcf8f659f2c11912036cdd`.
  Baseline and prediction files are mode `0600`.
- Gold-index SHA-256 remains
  `6560591461573bf08d50dd110c031d56f287ea570563b9ae0bfdae691d48d3d8`.
  No gold or prediction body was read and no prediction was revealed.
- Private completion receipt:
  `~/.local/state/transcribe-audio/plan-0036/predictions-complete-07432082-0974-4250-8da5-f8e9a34e497e.json`
  (`0600`), SHA-256
  `19e0b6dc6b5c6b5692e078669014caf83f6e5aaa7fcfd7c4346581518dff257e`.

State transition: `OPEN/Plan-0036-P2 -> OPEN/Plan-0036-P3`

Next:

- Present the ten prediction-excluded review packets in chronological order.
- Collect one independent operator gold record for each case.
- Reveal and score only after all ten current reviews exist.

## Turn 246 | 2026-07-30

Plan:
`docs/dev/plans/0035-2026-07-30-blind-combined-speaker-outcome-measurement.md`
version 1

Packet: P0 | Freeze bridge and public-interface proof

State transition: `CLOSED/Plan-0034-PASS -> OPEN/Plan-0035-P0`

Vision outcome:

- Measure the current default combined speaker-identification path on the next
  frozen chronological cohort.
- Move from maturity `2 — Shadow` toward an evidence-backed decision about
  `3 — Operational`.
- Do not equate provider readiness, service health, or prediction completion
  with identity quality.

Authority and bounds:

- User authorized the successor with `plan and execute`.
- Exact freeze:
  `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`, ten cases at
  chronological ranks 25 through 39.
- Live readback before implementation: 10/10 predictions `not_started`,
  10/10 ground truth `not_reviewed`, gold absent, and freeze hash-bound.
- Plan 0034 proves included read-only GWS and Odollo evidence and preserves
  sidecar authority, zero live `knowledge_*` tables, and disabled automatic
  confirmation.
- The App Intelligence readiness helper reports Codex app-server healthy with
  stdio, schema-generation, and TypeScript-generation capabilities.
- No subagent is spawned because the single bridge/test pair and dependent
  live run form one serialized private-authority path.

Correction:

- The cohort is frozen and unseen, but its gold is not frozen. Independent
  operator gold must be written after prediction completion and before reveal.
  Plan 0035 preserves this gate instead of scoring against inferred labels.

Next:

- Add one test-first, idempotent bridge from the conversation-evaluation
  freeze to the existing private holdout-baseline interface.
- Push and install the source before starting the sole blind run.
- Capture all ten predictions, then pause at the independent operator-review
  gate without exposing predictions.

P0 implementation checkpoint:

- RED:
  `.venv/bin/python -m pytest
  tests/test_conversation_knowledge_evaluation.py::test_evaluation_freeze_starts_one_idempotent_blind_holdout
  -q` failed because the public freeze-to-holdout interface did not exist.
- GREEN: the same command passes after adding
  `start_evaluation_holdout_baseline`.
- The interface verifies the conversation-freeze schema and identity,
  blindness flags, `not_started`/`not_reviewed` states, cohort size, campaign
  manifest hash, gold-index hash, and every document/artifact pair before any
  baseline write.
- The bridge writes one deterministic private holdout freeze, reuses the
  existing baseline/capture/reveal machinery, and returns the same baseline
  on exact replay. A conflicting or non-blind replay fails closed.
- A second public-interface test proves a non-blind case is rejected before
  either a freeze or baseline directory is written.
- Focused evaluation, campaign, and baseline-runner validation passes 22
  tests. The joined adapter, identity retrieval, knowledge, workflow,
  campaign, and transcript API validation passes 164 tests.
- Python compilation and `git diff --check` pass.
- The live campaign manifest and gold-index hashes exactly match the frozen
  source hashes. No frozen document appears in the current gold index.
- The served speaker-disambiguation profile is `codex_supervisor` using
  provider `codex-app-server` and model `gpt-5.6-sol`; provider readiness is
  green on Codex CLI `0.146.0`.

## Turn 245 | 2026-07-30

Scope: establish the durable product north star and route recurring planning
surfaces to it.

State transition: `distributed product intent -> canonical VISION.md`

Outcome:

- Added `VISION.md` as the canonical authority for the intended automatic
  transcript-to-contextual-readout-to-knowledge loop.
- Defined the required readout dimensions, evidence and authority boundaries,
  calibrated-confidence behavior, maturity scale, outcome measures, planning
  contract, and conditions for realizing the vision.
- Linked `AGENTS.md` to the vision for non-trivial planning, architecture,
  prioritization, and goal execution.
- Linked `ROADMAP.md` to the vision and required roadmap lanes and bounded plans
  to report measurable vision progress rather than infrastructure completion
  alone.

Validation:

- `git diff --check`
- Relative-link target existence check for all links in `VISION.md`
- `.venv/bin/python
  .codex/skills/repo-policy-selector/scripts/audit_planning_contract.py
  --repo . --active-only --json` (`ok: true`)

Residual:

- Existing open plans predate the new planning contract. Apply the contract
  when each plan is next revised rather than rewriting their historical state
  in this documentation-only slice.

## Turn 244 | 2026-07-29

Plan:
`docs/dev/plans/0034-2026-07-29-gws-capability-budget-fairness.md`
version 1

Packet: P1 | Test-first fairness repair and final GWS proof

State transition: `CLOSED/Plan-0033-REFINE -> CLOSED/Plan-0034-PASS`

Progress classification: `bounded_remediation`; GWS now executes inside the
service and the remaining zero-yield cause is isolated to adapter-local global
budget starvation.

Design and validation authority:

- Public interface: `GwsEvidenceAdapter.retrieve`.
- External seam: injected `GwsProviderReader`; the existing in-memory fake is
  the test adapter.
- Required behavior: an oversized first-capability page is truncated to a fair
  share while a later configured capability is still queried and can return a
  normalized snapshot.
- No new interface, request field, config option, capability reorder, or
  temporal-policy change is permitted.
- TDD and codebase-design skills were read; CodeGraph impact limits the blast
  radius to the GWS adapter, its tests, policy construction, and transcript API
  caller.

Authority and bounds:

- One red/green cycle, one source work unit, one service restart, and one final
  immutable request on the fixed Plan 0032 target are authorized.
- Model calls, target substitution, credentials/config changes,
  frozen-cohort consumption, gold reads, predictions, legacy rollback,
  automatic confirmation, database authority, and external writes remain
  prohibited.
- Delegation: `not_spawned`; the test/code pair and dependent live proof are
  one narrow serialized critical path.

Next:

- RED:
  `.venv/bin/python -m pytest tests/test_conversation_evidence_gws.py::test_gws_adapter_preserves_later_capability_access_under_record_budget -q`
  failed because four Calendar snapshots consumed the record budget and
  People was never queried.
- GREEN: the same command passed after the adapter-local implementation.
- The adapter now derives each capability's share from the remaining global
  record budget and remaining configured capabilities. It truncates only the
  current capability at that share, continues to later capabilities, and
  leaves unused share available downstream.
- The full GWS suite passes 15 tests. The joined adapters, identity retrieval,
  policy, evidence, projection, provenance, workflow, and transcript API suite
  passes 143 tests.
- README now documents the installed GWS service PATH and adaptive
  capability-budget contract.

Terminal:

- Committed and pushed the adapter repair as
  `bb0f0813b0036e4df779e8ad03b666fc834d5144`, restarted the service once,
  and verified PID `3775047`, `NRestarts=0`, the active GWS PATH, green API
  health, and ready default `codex-app-server` supervision.
- The sole final immutable request
  `5bc9882e-71b8-43ca-b45c-10a9505772d2` retained six terms, three source
  scopes, and seven capabilities. It included two `gws-default` People
  controls and four Odollo controls.
- The evidence bundle remains correctly `partial`: 18 historical provider
  results were explicitly excluded as outside the shared temporal scope and
  `provider_records_truncated` remained visible.
- Retrieval receipt:
  `~/.local/state/transcribe-audio/conversation-identity-shadow/identity-retrieval-receipts/5bc9882e-71b8-43ca-b45c-10a9505772d2.json`
  (`0600`), SHA-256
  `028ecd3c488ddf5fdf6051b3247794447111ff4d88a6aac3346e7b86068daa4b`.
- Frozen and authority states remain unchanged: 10/10 predictions
  `not_started`, 10/10 ground truth `not_reviewed`, gold absent/unseen, zero
  live `knowledge_*` tables, sidecar authority, and automatic confirmation
  disabled.
- Terminal receipt:
  `~/.local/state/transcribe-audio/plan-0034/terminal-pass-79e7c595-bdfc-4c21-af67-9df4fb2e6d71.json`
  (`0600`), SHA-256
  `c569c6422ac23c822c06d6dde091f944503c6f38efdb3276cce170f63cd103f5`.

Next:

- Treat provider readiness as proven. A blind frozen-cohort prediction or
  human gold-review campaign is separate work and requires its own explicit
  authorization.

## Turn 243 | 2026-07-29

Plan: `docs/dev/plans/0033-2026-07-29-gws-service-path-repair.md`
version 1

Packet: P1 | Installed runtime repair and GWS-inclusive proof

State transition: `CLOSED/Plan-0032-PASS -> OPEN/Plan-0033-P1`

Progress classification: `bounded_remediation`; the general provider-yield
gate passed, and the remaining GWS defect is now attributed to the installed
service PATH rather than authorization, adapter semantics, or target input.

Preflight evidence:

- Interactive `gws` resolves to `/home/ecochran76/.cargo/bin/gws`; the
  metadata-only calendar probe already proved its restored authorization.
- The user manager PATH and merged `transcripts.service` unit omit
  `/home/ecochran76/.cargo/bin`.
- The Plan 0032 immutable receipt reports
  `provider_unavailable/gws executable unavailable` for all four GWS
  capabilities while four Odollo snapshots were included.
- Existing `10-codex-bin.conf` and `20-odollo-env.conf` drop-ins remain loaded
  and are outside the repair write surface.

Authority and bounds:

- One `30-gws-path.conf` drop-in, one daemon reload/restart, and one served
  immutable retry on the fixed Plan 0032 target are authorized.
- Model calls, source changes, credential changes, target substitution,
  frozen-cohort consumption, gold reads, predictions, legacy rollback,
  automatic confirmation, database authority, and external writes remain
  prohibited.
- Delegation: `not_spawned`; installed mutation and dependent live proof are a
  serialized critical path.

Next:

- Terminal `refine`.
- Installed `30-gws-path.conf`, verified the merged unit, reloaded, and
  restarted once. PID changed from 2447995 to 3519614, `NRestarts=0`, the
  running process PATH includes `/home/ecochran76/.cargo/bin`, and API health
  is green.
- The immutable six-term request executed GWS and returned twenty snapshots,
  but the shared historical policy excluded all twenty. The adapter emitted
  `provider_records_truncated`, no GWS evidence control was included, and four
  Odollo snapshots remained included.
- Retrieval receipt:
  `conversation-identity-shadow/identity-retrieval-receipts/d702aa25-b5bb-49df-bfae-30acdad33e37.json`
  (`0600`), SHA-256
  `90ab6a9a513d3ea71cefa9d01bc18977301e63f96af34966f5bfe98721c360bd`.
- Source tracing confirms the GWS adapter processes capabilities in request
  order under one global inspected-record budget. The first high-yield
  capability can therefore starve all later capabilities even when its
  snapshots will be excluded by the host retrieval policy.
- Terminal receipt:
  `~/.local/state/transcribe-audio/plan-0033/terminal-refine-818efd3e-7a4f-4dc5-98f1-ce8dfb7d24b4.json`
  (`0600`), SHA-256
  `98a34d358971b0a7f7804c7be556412e3459a2b16a2b1daf60d29ff06d1c1e82`.
- Frozen and authority states remain unchanged: 10/10 predictions
  `not_started`, 10/10 ground truth `not_reviewed`, gold absent/unread, zero
  live `knowledge_*` tables, sidecar authority, and automatic confirmation
  disabled.

Next:

- Add one public-interface regression test for cross-capability budget
  fairness, implement the adapter-local fix, and execute one final immutable
  GWS proof.

## Turn 242 | 2026-07-29

Plan:
`docs/dev/plans/0032-2026-07-29-target-qualified-provider-yield-retry.md`
version 1

Packet: P1 | Qualified immutable retry and terminal gate

State transition: `CLOSED/Plan-0031-REFINE -> OPEN/Plan-0032-P1`

Progress classification: `bounded_remediation`; a provider-free eligibility
scan removed the verified zero-term input-selection blocker without relaxing
the immutable-attempt gate.

Authority and bounds:

- The fixed target is document `158fe299a59444821675`, the first recent
  non-frozen candidate with a nonempty deterministic query plan.
- Preflight: six calendar attendees, six exact-first query terms, 270
  utterances, four anonymous diarization labels, and no frozen-cohort
  membership.
- One served default retrieval attempt and one attempt per configured source
  scope are authorized. No target substitution or code remediation is
  permitted after execution.
- Model calls, frozen-cohort consumption, gold reads, predictions, legacy
  rollback, automatic confirmation, database authority, and external writes
  remain prohibited.
- Delegation: `not_spawned`; this is one serialized immutable live attempt with
  a deterministic terminal gate.

Next:

- Terminal `pass`.
- The immutable request had three explicit scopes, seven capabilities, and six
  query terms. It included four normalized supporting-evidence snapshots: one
  Soylei contact, two Soylei leads, and one Saber log note.
- GWS remained explicitly partial with four
  `provider_unavailable/gws executable unavailable` failures. Interactive GWS
  authorization is valid; the service PATH omits
  `/home/ecochran76/.cargo/bin`.
- Retrieval receipt:
  `conversation-identity-shadow/identity-retrieval-receipts/54732bd0-41e9-4acd-9c65-4bc50f41ab21.json`
  (`0600`), SHA-256
  `eb395d24f2955720eb8a76d77b5b3552d2ae490d002de72b2d0964ad3b99c784`.
- Terminal receipt:
  `~/.local/state/transcribe-audio/plan-0032/terminal-pass-1827a286-99fe-42ca-a0c4-3812a4d72f79.json`
  (`0600`), SHA-256
  `75b264ed287a8d65c42204a5234a8b730b0f7af8b259e6dd652030098f16037f`.
- Frozen predictions remain 10/10 `not_started`; ground truth remains 10/10
  `not_reviewed`; gold remains absent and unread. The live database still has
  zero `knowledge_*` tables; sidecar authority and disabled automatic
  confirmation are unchanged.

Next:

- Repair only the installed service PATH, then prove one GWS snapshot through
  a fresh bounded immutable request.

## Turn 241 | 2026-07-29

Plan: `docs/dev/plans/0031-2026-07-29-provider-yield-retry.md` version 1

Packet: P1 | Readiness, immutable retry, and terminal gate

State transition: `CLOSED/Plan-0030-REFINE -> OPEN/Plan-0031-P1`

Progress classification: `feature_progress`; the operator reports restored
GWS authorization and explicitly authorizes the fresh bounded provider retry
required by the Plan 0030 successor entry condition.

Authority and bounds:

- Plan 0031 authorizes one metadata-only GWS authorization probe, one served
  default immutable retrieval attempt, and one Odollo call per configured
  scope as part of that attempt.
- The prior non-frozen smoke conversation remains the only evaluation target.
- Model calls, frozen-cohort consumption, gold reads, predictions, legacy
  rollback, automatic confirmation, database authority, and external writes
  remain prohibited.
- Delegation: `not_spawned`; the live retry is a short serialized critical
  path and no independent write or review lane exists before its receipt.

Preflight evidence:

- Git starts clean on `plan-0026-campaign`; local and upstream HEAD are
  `eeeb0e083ca9135880e15619cd460d4442400576`.
- CodeGraph is current with zero pending changes.
- Graphiti runtime is healthy; repo-group discovery returned only older
  unrelated operational facts, so repo files and live readbacks remain
  authority.
- `transcripts.service` is active from the pushed repo source.
- Provenance resolution returns one explicit GWS source and two explicit
  Odollo tenant sources with no warnings.

Next:

- Terminal `refine`.
- The metadata-only GWS calendar probe succeeded and emitted only authorization
  and response-shape status.
- The served request created immutable retrieval receipt
  `conversation-identity-shadow/identity-retrieval-receipts/c6e118c1-66a7-4fc6-a56b-4834a8309c74.json`
  (`0600`), SHA-256
  `46a7834826d890b7b6d2a8586071ecd6b9ca85c8db7635c8dce99ebf220c9f76`.
- The request had three explicit scopes, seven capabilities, two anonymous
  speaker labels, zero clue IDs, zero prepared people, and zero query terms.
  Every adapter returned `provider_query_failed/query terms are required`
  without issuing a provider query. The partial bundle had zero items and
  warning `no_bounded_evidence`.
- A provider-free scan of twelve recent conversations found eleven eligible
  non-frozen calendar-associated inputs. The first has six calendar attendees,
  six deterministic query terms, 270 utterances, and four anonymous labels.
- The frozen cohort remains 10/10 `not_started`, 10/10 `not_reviewed`, gold
  absent, and unconsumed. The live database still has zero `knowledge_*`
  tables; sidecar authority and disabled automatic confirmation are unchanged.
- Terminal receipt:
  `~/.local/state/transcribe-audio/plan-0031/terminal-refine-5c5cdc9a-cb28-45d7-8e8f-00dca005c262.json`
  (`0600`), SHA-256
  `0f5dad7093c23ba3ea8ec758c8c7f1ce3bcdadf054ff07bb6abb9b7e80199c31`.

Next:

- Execute Plan 0032's one target-qualified immutable retrieval attempt.

## Turn 240 | 2026-07-29

Plan: Plan 0030 version 2

Packet: J2 | Integrated readiness and terminal decision

State transition: `OPEN/J2 -> CLOSED/REFINE`

Progress classification: `terminal`; the bounded implementation and authority
proof are preserved, while R3 is correctly not run because provider yield did
not satisfy the explicit gate.

Evidence:

- Neutral J2 receipt:
  `~/.local/state/transcribe-audio/plan-0030/j2-39c5aeab-9b54-4bf4-8530-2c4fb1183e02.json`
  (`0600`), SHA-256
  `c872dbb4917f6bbab404cac8c89deebba656615b8b192da5d867c791d40a5891`.
- J2 passed silent-fallback, explicit-scope, zero-yield semantics,
  family-label, default-caller, rollback, and private-shadow gates.
- J2 failed the required yield gate: the immutable runtime bundle contained
  zero evidence controls and zero included provider snapshots after all
  permitted source attempts.
- Current source passes 142 host-safe focused tests, Python compilation, and
  `git diff --check`.
- Commit `febed062dd3908295db8d74d770866be048a46ab` is pushed to
  `origin/plan-0026-campaign`.
- `transcripts.service` was restarted from that current source and is active;
  Codex App Server reports ready. The served speaker prepare-evaluation route
  rejects an invalid legacy rollback token with HTTP 400 before provider
  access.
- Immutable terminal `refine` receipt:
  `~/.local/state/transcribe-audio/plan-0030/terminal-refine-6c2c6298-37dc-42bc-84c1-4a933554479d.json`
  (`0600`), SHA-256
  `e191559dc01fa6abf3204a47cf0126b14c047e9683842185d76e19da05bac5ee`.

Delegation:

- `/root/j1_neutral_review` executed the read-only J2 review with no children,
  edits, provider/model calls, predictions, or gold reads; terminal `refine`.
- Primary reconciliation accepted the finding because it follows the plan's
  explicit included-snapshot prerequisite and exhausted attempt bounds.

Bounds:

- Work-unit attempts: R2A `2/2`, R2B `2/2`.
- Review rework cycles: `1/1`.
- Hardening checkpoints: `1/2`.
- Provider attempts: GWS `2/2`; each Odollo scope `2/2`.
- Reference-repair turns: `0/1`.
- Frozen cohorts consumed: `0/1`.

Actions and authority:

- R3A, R3B, R3C, and R3D: `not_run`,
  reason `blocked_by_j2_no_included_provider_snapshot`.
- Model calls: 0; predictions: 0; gold-body reads: 0; external writes: 0;
  automatic confirmations: 0.
- Frozen predictions remain 10/10 `not_started`; ground truth remains 10/10
  `not_reviewed`; gold content remains absent.
- Sidecar authority remains active. Live database authority is false, schema
  version remains 0, and zero `knowledge_*` tables exist.

Residual risk and successor entry:

- GWS refresh authorization must be restored.
- A successor must explicitly authorize a new bounded provider-attempt packet
  and prove at least one included provider snapshot before any provenance or
  combined prediction.
- The current frozen cohort must not be reused for tuning; any successor
  authority must explicitly decide whether its still-unseen state permits
  continued evaluation.

## Turn 239 | 2026-07-29

Plan: Plan 0030 version 2

Packets: R2A, R2B | Default immutable retrieval and private shadow authority

State transition: `OPEN/R2A+R2B -> OPEN/J2`

Progress classification: `feature_progress` with verified J2 readiness
blockers; the default caller and shadow-authority gates are implemented, while
live provider yield remains unproven within the attempt bound.

Evidence:

- R2A private receipt:
  `~/.local/state/transcribe-audio/plan-0030/r2a-6ca3f875-8998-4e0c-95a8-04d425b175cb.json`
  (`0600`), SHA-256
  `1d19f0e076abec2844bf3cae679de665c084c168df0119d218693315120c0df5`.
- The selected Identity Evaluation API now defaults to an explicit
  source-profile/account/tenant/capability policy and an immutable retrieval
  bundle. Exact calendar identifiers lead a query plan capped at 24 terms;
  validated Clue Discovery person hints and retrieval terms are included
  before transcript-derived tokens.
- Partial provider failures remain on the retrieval-bundle path. Legacy
  collection requires `evidence_mode=legacy_rollback`, the exact approval
  token, an operator, a visible warning, and a durable private receipt.
- Default receipts include request/query-plan and bundle hashes, failures,
  warnings, included/excluded reasons, freshness, temporal class, and
  independence groups without raw provider bodies.
- A non-frozen direct runtime smoke produced a correctly labeled partial
  bundle and private receipt
  `conversation-identity-shadow/identity-retrieval-receipts/8e0454a0-4d09-46bf-82fd-73cc658aa1c7.json`
  (`0600`), SHA-256
  `5a569ebb2c63fe687ff45f7c67aae31b9143202350a5511074d5ce4da736dda9`.
- GWS attempt two remained `provider_auth_failed`: the persisted refresh token
  is expired or revoked even after a local `gws auth sync-gog`. The two Odollo
  attempts exposed later-retrieved/UTC-naive timestamp handling and then an
  unlinked-provider-record foreign key. Both code defects now have regression
  fixes, but a third live attempt is forbidden. No included provider snapshot
  is therefore proven.
- R2B remediation attempt `2/2` passed. Ten frozen inputs received only
  deterministic SHA-256-bound UUIDv5 private overlays; 10/10 source hashes,
  overlay replays, projections, sidecar read agreements, restored agreements,
  and unchanged replays passed. Rollback v3 to v0 and independent v3 restore
  passed.
- R2B private receipt:
  `~/.local/state/transcribe-audio/plan-0030/r2b-e436aa63-f5c3-40a4-9bb0-e30d9b823cab/receipt.json`
  (`0600`, private tree `0700`), SHA-256
  `c5082f098ed3bf7ffecc50c57c70f233430eb6cea8964cc381be2da68fb8d8d9`.
- The final host-safe focused run passes 142 tests; compilation and
  `git diff --check` pass.
- `transcripts.service` was restarted from the current worktree and is active.
  Its served prepare-evaluation route rejects an invalid legacy rollback token
  with HTTP 400 before provider access.

Delegation:

- `/root/r2b_shadow_authority`: first rehearsal terminal `refine`; bounded
  remediation attempt `2/2` terminal `pass`, no children or tracked edits.
- Primary reconciliation implemented the deterministic overlay contract,
  validated the joined source, retained provider-attempt accounting, and did
  not rerun an exhausted source.

Bounds:

- R2A work-unit attempts: `2/2`.
- R2B work-unit attempts: `2/2`.
- Review rework cycles: `1/1` (already consumed at J1).
- Hardening checkpoints: `1/2`.
- Provider attempts: GWS `2/2`; each Odollo scope `2/2`.
- Reference-repair turns: `0/1`.
- Frozen cohorts consumed: `0/1`.

Actions and authority:

- Model calls: 0; external writes: 0; predictions: 0; gold-body reads: 0.
- Local writes were limited to the private shadow/receipts, non-frozen durable
  transcript identity synchronization, local GWS credential sync, and service
  restart.
- Frozen predictions remain `not_started`; gold remains unread.
- Live database remains schema v0 with zero `knowledge_*` tables. Sidecar
  authority remains active; database authority and automatic confirmation are
  disabled.
- Source commit and push are pending the J2 checkpoint.

Next:

- Obtain neutral J2 readiness review. Because no included provider snapshot is
  proven and provider attempts are exhausted, provenance/combined predictions
  remain blocked and J2 must select the plan's bounded terminal path.

## Turn 238 | 2026-07-29

Plan: Plan 0030 version 2

Packets: R1B, R1C, J1 | Concrete provider adapters and integration review

State transition: `OPEN/R1B+R1C+J1 -> OPEN/R2A+R2B`

Progress classification: `feature_progress`; both configured provider families
now have concrete read-only bounded evidence adapters and the neutral join gate
passes.

Evidence:

- Private receipt:
  `~/.local/state/transcribe-audio/plan-0030/j1-3b947d28-45f1-4b3e-9a0e-fea6afdf7286.json`
  (`0600`, parent `0700`).
- Receipt content hash:
  `4fe89e18d4182263cd8b97772277416f74b1215e9c37630c522a1fc83b9c2004`.
- GWS supports configured `people`, `gmail`, `drive`, and `calendar`
  capabilities through a concrete timeout-bounded CLI reader. Gmail requests
  metadata and snippets only; Drive and Calendar use bounded field lists.
- Odollo supports configured `contacts`, `leads`, and `log_notes` through
  tenant-isolated read-only record searches. Log-note bodies may be searched
  for a match but are never requested in returned fields or persisted.
- Both adapters preserve explicit source profile/account/tenant scope and
  delegate all temporal, ID, hash, and content-boundary enforcement to the
  shared normalizer. Neither groups people nor infers identity.
- Primary reconciliation removed raw GWS diagnostic text from durable failure
  detail before neutral review.
- Neutral review initially found three issues: raw/unbounded redaction and
  truncation metadata, unbounded advancing GWS pagination/invalid failures,
  and GWS/Odollo failure semantic drift. The single allowed rework cycle
  closed all three; the same reviewer verified closure with 43 focused tests.
- The final combined adapter/evidence/retrieval run passed 53 tests in 0.35
  seconds. Compilation and `git diff --check` pass.

Delegation:

- `/root/r1b_gws`: terminal `complete`, work-unit attempts `2/2`, no children.
  Attempt two added the missing concrete GWS CLI reader.
- `/root/r1c_odollo`: terminal `complete`, work-unit attempts `1/2`, no
  children.
- `/root/j1_neutral_review`: terminal `pass_after_rework`, review-rework cycles
  `1/1`, no children.
- Primary reconciliation reviewed both disjoint write surfaces, applied the
  one bounded J1 rework, and ran the combined suites.

Bounds:

- R1B work-unit attempts: `2/2`.
- R1C work-unit attempts: `1/2`.
- Review rework cycles: `1/1`.
- Hardening checkpoints: `0/2`.
- Provider attempts per scope: `0/2`.
- Reference-repair turns: `0/1`.
- Frozen cohorts consumed: `0/1`.

Actions and authority:

- Live provider calls: 0; model calls: 0; external writes: 0; predictions: 0;
  gold-body reads: 0.
- Sidecar authority retained; live database authority and automatic
  confirmation disabled.
- Base source and remote commit were both `ad3d7d3`.

Next:

- Commit and push the adapter join, then execute R2A default-caller
  integration and R2B private sidecar provenance proof.

## Turn 237 | 2026-07-29

Plan: Plan 0030 version 2

Packet: R1A | Shared adapter contract and fixtures

State transition: `OPEN/R1A -> OPEN/R1B+R1C`

Progress classification: `feature_progress`; created the shared bounded
normalization seam required by both provider-specific lanes.

Evidence:

- Private receipt:
  `~/.local/state/transcribe-audio/plan-0030/r1a-a3e131da-c37f-4830-99be-ab053ea4fe0b.json`
  (`0600`, parent `0700`).
- Receipt content hash:
  `7a1ac31493101a2bea0e49b52fae00c21b045581f65fecfa8167bde9991c9f23`.
- `conversation_evidence_adapters.py` now owns explicit provider scope,
  allowlisted normalization, stable snapshot identities/hashes, temporal
  classification, raw-body rejection, size limits, and fixed adapter
  failure/warning codes.
- Contract TDD recorded four red stages: missing module, missing scope
  validation, missing record allowlist validation, and missing failure schema.
- Eighteen focused contract tests pass.
- Existing evidence repository and identity retrieval tests remain green; the
  combined focused run passed 28 tests in 0.35 seconds.
- Python compilation and `git diff --check` pass.

Delegation:

- `not_spawned`; R1A defined the shared seam required by both later provider
  lanes and could not be partitioned without creating competing contracts.
- Handle: none. Terminal status: `not_spawned`. Reconciliation: primary agent
  implemented and validated the shared public contract.

Bounds:

- Work-unit attempts: `1/2`.
- Review rework cycles: `0/1`.
- Hardening checkpoints: `0/2`.
- Provider attempts per scope: `0/2`.
- Reference-repair turns: `0/1`.
- Frozen cohorts consumed: `0/1`.

Actions and authority:

- Provider calls: 0; model calls: 0; external writes: 0; predictions: 0;
  gold-body reads: 0.
- Sidecar authority retained; live database authority and automatic
  confirmation disabled.
- Base source and remote commit were both `5ccc251`.

Next:

- Commit and push R1A, then execute GWS R1B and Odollo R1C as disjoint bounded
  provider lanes.

## Turn 236 | 2026-07-29

Plan: Plan 0030 version 2

Packet: P0 | Authority, freeze, and runtime preflight

State transition: `OPEN/P0 -> OPEN/R1A`

Progress classification: `feature_progress`; removed the named authority,
freeze-integrity, source-scope, and runtime-readiness gate without consuming
the cohort.

Evidence:

- Private receipt:
  `~/.local/state/transcribe-audio/plan-0030/preflight-a4fb020d-4bae-5ec3-8fc0-de8f743f34e4.json`
  (`0600`, parent `0700`).
- Receipt content hash:
  `06d270faceaa4b34e32891ab11225d295944b6b904654e660f6467f27e0e905f`.
- Freeze
  `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b` retains ten cases at ranks
  25, 26, 27, 28, 29, 30, 31, 34, 35, and 39; all predictions are
  `not_started`, all ground-truth markers are `not_reviewed`, gold content is
  absent, and source hashes match.
- Plan 0029's `refine` receipt hash recomputes; sidecar authority remains true,
  while database authority and automatic confirmation remain false.
- Installed provenance validates three scoped sources: one GWS profile with
  explicit empty account/tenant scope and two tenant-explicit Odollo profiles.
  GWS and both configured Odollo executables are present.
- Current source has only the `HostEvidenceAdapter` protocol and the selected
  caller remains legacy. The live database has zero knowledge-schema tables.
- Graphiti runtime was healthy but returned no Plan 0030-specific authority;
  repo and private receipts remained authoritative.

Delegation:

- `not_spawned`; P0 directly gated every later lane and required sensitive
  authority/freeze inspection. A parallel worker would have duplicated the
  critical path.
- Handle: none. Terminal status: `not_spawned`. Reconciliation: primary agent
  verified all source and runtime evidence directly.

Bounds:

- Work-unit attempts: `1/2`.
- Review rework cycles: `0/1`.
- Hardening checkpoints: `0/2`.
- Provider attempts per scope: `0/2`.
- Reference-repair turns: `0/1`.
- Frozen cohorts consumed: `0/1`.

Actions and authority:

- Provider calls: 0; model calls: 0; external writes: 0; predictions: 0;
  gold-body reads: 0.
- Sidecar authority retained; live database authority and automatic
  confirmation disabled.
- Source commit and remote were both `e0abd37`; no served behavior changed.

Next:

- Execute R1A with one public-interface tracer test for bounded adapter
  normalization, then proceed one behavior at a time.

## Turn 235 | 2026-07-29

Summary: Upgraded Plan 0030 to a bounded `/goal` execution authority after a
one-question-at-a-time design grilling session.

Decisions:

- Partial provider failure remains on the immutable-bundle path; the legacy
  collector has no automatic fallback and requires explicit approval, warning,
  and receipt.
- Provider adapters normalize bounded records but never group people or infer
  identity.
- Source profile, account, and tenant are always explicit, including
  intentionally empty values.
- Historical evidence retrieved later is labeled `later_retrieved`; current
  undated contacts cannot establish contemporaneous topic or relationship
  context.
- Query plans are frozen before provider access with exact identifiers first
  and capped transcript-derived terms. Identity Evaluation cannot retrieve.
- Exact verified identifiers permit deterministic internal grouping; softer
  grouping remains confidence-scored and reversible, with no upstream contact
  mutation.
- Evaluation gold requires independent operator confirmation. Each evidence
  family reports both family-specific candidate generation and ranking over a
  frozen union candidate set.

Plan contract:

- Added explicit objective fidelity, authority order, execution bounds,
  checkpoint fields, bounded packet inputs/write surfaces/validation/terminal
  conditions, execution graph, delegation and reconciliation rules, neutral
  review, hard stops, acceptance criteria, and definition of done.
- The only parallel implementation lanes are the disjoint GWS/Odollo adapters,
  private shadow validation, and operator-owned gold review.
- Plan 0030 must close on one immutable `accept`, `refine`, `reject`, or `stop`
  decision; retry or hardening bounds cannot be extended inside the plan.
- Graphiti discovery was healthy but returned no Plan 0030-specific memory;
  current repo authorities and the Plan 0029 receipts remained authoritative.

Delegation:

- `spawned`; lane `read-only neutral Plan 0030 review`; handle
  `/root/review_plan0030`; terminal status `cancelled_after_timeout`.
- The reviewer returned no findings after the initial wait, an explicit
  conclude request, and one partial-result request. The primary agent cancelled
  it, inferred no review result, and retained final reconciliation ownership.
- Deterministic plan-field validation and direct primary policy reconciliation
  remain the closeout evidence. Plan execution must obtain a fresh neutral
  review at J1 and R3D when runtime capacity is healthy.

Next:

- Execute P0 in
  `docs/dev/plans/0030-2026-07-26-provider-adapters-and-blind-retrieval-evaluation.md`.
  Do not access gold bodies or consume a frozen prediction.

## Turn 234 | 2026-07-27

Summary: Reconciled the concurrently completed Plan 0029 closeout and restored
its C7 freeze and decision receipts to the documented durable private root.

Implemented:

- Confirmed the concurrent C6 commit had the exact same tree as the local C6
  commit and rebased without duplicating it.
- Recreated the C7 freeze deterministically from the unchanged campaign
  manifest and gold index. It retained the exact freeze ID, ten chronological
  ranks, four incomplete exclusions, two duplicate-member exclusions, and
  `not_started` prediction state without reading gold bodies.
- Persisted a durable aggregate `refine` receipt with all five evidence
  families explicitly `not_run` and every requested comparison metric null.
- Added explicit `0700` enforcement for the evaluation root as well as its
  freeze directory; freeze and decision files remain `0600`.
- Kept sidecars authoritative and automatic confirmation disabled.

Validation:

- Freeze ID:
  `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`.
- Durable decision content hash:
  `08ace0d9c00278ddb97cd1fffb7b59e01af67228198af0bd605106f0e150f05f`.
- Campaign-manifest hash:
  `e387ea4fdd51bf9cfb336cacf918aa782cb8e16b931ba8e022218a3e542e5aa4`.
- Gold-index hash:
  `6560591461573bf08d50dd110c031d56f287ea570563b9ae0bfdae691d48d3d8`.
- Live transcript database still contains zero knowledge-schema tables.
- Current source still has no concrete production `HostEvidenceAdapter`, and
  the selected-conversation API still uses the legacy collector by default.

Next:

- Execute
  `docs/dev/plans/0030-2026-07-26-provider-adapters-and-blind-retrieval-evaluation.md`
  R1. Do not read or predict the frozen cohort until its adapter,
  default-caller, shadow-store, and operator-gold gates pass.

## Turn 233 | 2026-07-26

Summary: Closed Plan 0029 with an explicit bounded-refinement decision after
freezing the next unseen chronological cohort and stopping before misleading
model work.

Implemented:

- Completed C6 retrieval-bundle adaptation in the existing Identity Evaluation
  workflow while preserving exact prepared-reference validation, factor
  scoring, confidence calibration, and mandatory human confirmation.
- Included evidence becomes model-visible bounded provenance; excluded
  evidence exposes only reason, scope, freshness, temporal, ranking, and hash
  metadata, never excluded content.
- Persisted retrieval request, bundle, warning, failure, allowlist,
  independence, freshness, temporal, and inclusion metadata in review
  sidecars.
- Added `conversation_knowledge_evaluation.py` with approval-gated,
  deterministic private cohort freezes and immutable aggregate readiness
  decisions.
- Froze ten unseen cases at chronological ranks 25, 26, 27, 28, 29, 30, 31,
  34, 35, and 39. Four incomplete and two duplicate-member rows remained
  explicitly counted rather than silently disappearing.
- Recorded C7 decision `refine` before starting predictions. Production source
  has no concrete GWS/Odollo `HostEvidenceAdapter`, the default transcript API
  caller still uses the legacy collector, and the live store intentionally has
  no knowledge-schema tables.
- Opened Plan 0030 for the bounded adapter, default-caller, private-shadow, and
  blind five-family evaluation slice.

Validation:

- C6 focused coverage passes with 42 tests.
- The complete C6 inventory passes in host-safe partitions: 339
  non-participant tests and all 12 participant-identity tests.
- Three C7 freeze/decision tests pass; compile checks and `git diff --check`
  pass.
- Private freeze:
  `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`.
- Immutable decision content hash:
  `08ace0d9c00278ddb97cd1fffb7b59e01af67228198af0bd605106f0e150f05f`.
- The frozen cohort remains unseen with every prediction `not_started`.
- No new model call, provider call, source mutation, speaker assignment,
  external write, live migration, database-authority cutover, automatic
  confirmation, or Graphiti write occurred.

Next:

- Execute Plan 0030 R1 with concrete bounded GWS/Odollo evidence-snapshot
  adapters. Do not consume the frozen cohort until adapter, default-caller,
  shadow-read, and operator-gold gates pass.

## Turn 232 | 2026-07-26

Summary: Completed Plan 0029 C5 with host-owned, exact-first, bounded identity
evidence retrieval and immutable bundle receipts.

Implemented:

- Added `conversation_identity_retrieval.py` with
  `prepare_identity_evidence(...)` and one explicit policy for source tuples,
  capabilities, temporal/hindsight/freshness rules, provider calls, packet
  records/characters, per-source limits, and relationship hops.
- Added a bounded host-adapter protocol; provider results must already satisfy
  the evidence snapshot contract and are rejected if their scope, capability,
  time, or freshness falls outside policy.
- Resolved calendar attendee emails and authoritative identifiers before
  lexical, semantic, source-record, and relationship retrieval.
- Preserved every permitted account/tenant source affinity for grouped people
  while filtering relationship summaries that belong to non-permitted scopes.
- Ranked supporting and contradicting evidence with persisted raw features,
  enforced evidence-independence groups and total packet budgets, and retained
  explicit inclusion/exclusion reason codes.
- Persisted immutable retrieval requests and content-hashed bundles before
  returning any model-consumable packet.
- Preserved calendar-only and prepared-person fallback behavior.
- Converted provider exceptions, partial results, and call-budget exhaustion
  into labeled partial bundles and warnings rather than negative evidence.

Validation:

- Four C5 tests pass for exact-first lookup, bounded hybrid retrieval,
  contradiction ranking, independence/freshness/record/character/per-source
  budgets, multi-database grouping, calendar/prepared fallback, partial
  provider failure, out-of-scope rejection, call limits, and replay.
- C1-C5 focused coverage passes with 53 tests.
- The complete 349-test inventory passes in host-safe partitions: 337
  non-participant tests and the 12 participant tests already split across
  explicit empty/configured provenance lanes.
- The private isolated preview over the 3 current sidecars persisted 3
  replayable bundles, retained 11 calendar candidates, included no unavailable
  evidence, and labeled the empty bounded scope with
  `no_bounded_evidence`.
- `py_compile` and `git diff --check` pass.
- No live provider call, model call, live database migration, source artifact
  mutation, speaker assignment, external write, or Graphiti write occurred.

Next:

- Execute C6 one-caller-at-a-time integration into the speaker workflow while
  preserving Clue Discovery, exact prepared-reference validation, confidence
  calibration, review gates, and all existing fallbacks.

## Turn 231 | 2026-07-26

Summary: Completed Plan 0029 C4 with immutable reviewed observations and
deterministic person/affinity projections.

Implemented:

- Added schema version 3 with replaceable current-person and typed-affinity
  profile tables, supporting-observation IDs, deterministic input watermarks,
  and transactional rollback to the version-2 evidence schema.
- Added `conversation_knowledge_profiles.py` to append confirmed, rejected,
  deferred, superseded, split-speaker, mixed-speaker, and reviewer-asserted
  identity observations from immutable processing history.
- Added versioned source-record affinity and concept-mention observations;
  source updates append a new content-hash-keyed observation instead of
  rewriting the prior record.
- Built deterministic current person, interaction, organization, project,
  topic, terminology, and source-relationship profiles.
- Preserved same-name ambiguous people as separate profiles and retained all
  account/tenant/source affinities after grouping.
- Kept materialized profile rebuilds independent from the immutable
  observation ledger.

Validation:

- Three C4 tests pass for all required outcome types, immutable re-append,
  same-name ambiguity, every source affinity, supporting observation IDs and
  watermark, deterministic delete/rebuild, and version-3 rollback.
- C1-C4 focused coverage passes with 49 tests.
- The complete 345-test inventory passes in host-safe partitions: 333 tests
  excluding `test_participant_identity.py`, 9 participant tests against an
  explicit empty provenance profile, and 3 configured-source participant tests
  using their own fixture profiles.
- Partitioning avoids unrelated default live-provider calls while the host
  filesystem journal remains degraded; it covers every collected test.
- The private isolated preview over the 3 current processing sidecars appended
  3 split/mixed diarization observations. No person or affinity profile was
  expected because those sidecars have no review decisions or linked contacts;
  the second rebuild was unchanged.
- `py_compile` and `git diff --check` pass.
- No live database migration, provider retrieval, model call, source artifact
  mutation, speaker assignment, external write, or Graphiti write occurred.

Next:

- Execute C5 `prepare_identity_evidence(...)` with exact-first bounded
  retrieval, explicit temporal/source policies, ranking, budgets, immutable
  bundle receipts, and labeled partial-provider failure.

## Turn 230 | 2026-07-26

Summary: Completed Plan 0029 C3 with bounded, source-scoped evidence storage
and replayable retrieval records while keeping the live store unmigrated.

Implemented:

- Added schema version 2 with evidence-independence groups, bounded evidence
  snapshots, FTS5 evidence/concept indexes, embedding-profile indexes,
  immutable retrieval requests, content-hashed evidence bundles, and
  reason-coded bundle items.
- Added `conversation_knowledge_evidence.py` as the focused repository
  interface for exact external-identity lookup, lexical and semantic evidence
  search, typed concepts and mentions, and request/bundle replay.
- Required exact source-profile, account, tenant, capability, `as_of`, and
  hindsight-policy filters for every evidence query.
- Preserved source-event, observed, retrieved, expiry, temporal-class,
  freshness, redaction, truncation, independence-group, content-hash, and
  provider-failure fields.
- Rejected snippets and structured metadata above their caps and prohibited
  raw provider-body fields from the structured metadata surface.
- Added transactional version-2 migration and rollback without weakening
  version-1 conversation/person/processing interfaces.

Validation:

- Six C3 behavior tests pass for bounded content, tenant/account/capability/time
  isolation, exact and FTS5 lookup, bounded vector ranking, immutable concepts
  and mentions, request/bundle hashes, provider failures, migration failure,
  and rollback.
- The complete 342-test inventory passes in isolated partitions: 330 tests
  excluding `test_participant_identity.py`, then all 12 participant tests
  split between configured-source and no-live-provider lanes.
- Partitioning was required because the degraded host filesystem journal left
  unrelated live Odollo subprocesses in uninterruptible
  `jbd2_log_wait_commit`; memory-backed temp roots avoided misattributing that
  infrastructure fault to C3.
- A consistent private copy of the live version-0 transcript database migrated
  through versions 1 and 2, preserved legacy document counts, rolled version 2
  back to version 1, and reapplied version 2. Authority remained `sidecar`.
- `py_compile` and `git diff --check` pass.
- No provider retrieval, model call, source artifact mutation, speaker
  assignment, external write, live database migration, or Graphiti write
  occurred in C3.

Next:

- Execute C4 immutable reviewed outcomes and deterministic current person,
  interaction, organization, project, topic, and terminology projections.

## Turn 229 | 2026-07-26

Summary: Completed Plan 0029 C2 as a sidecar-authoritative, hash-bound shadow
projection without migrating the live user store.

Implemented:

- Added `conversation_knowledge_projection.py` with read-only preview,
  deterministic source watermarking, explicit apply approval, source-change
  rejection, idempotent shadow apply, reconciliation, and sidecar export.
- Projected normalized conversation, recording, utterance, evaluation,
  decision, linked legacy contact, and speaker-assignment records through
  repository interfaces rather than exposing storage SQL to callers.
- Preserved legacy contact and assignment identifiers as source provenance;
  deterministic opaque IDs represent knowledge-store people, utterances, and
  diarized speakers.
- Added immutable assignment observations, projection-state repository
  methods, and private `0700`/`0600` reconciliation receipts.
- Kept source transcripts and processing sidecars byte-for-byte unchanged and
  retained explicit `sidecar` authority.

Validation:

- Eight focused C1/C2 tests pass, including preview/apply, source drift,
  round-trip, receipt permission, idempotence, migration, and rollback cases.
- Transcript-store, processing, C1, and C2 focused coverage passes with 40
  tests.
- The full suite passes with 336 tests; `py_compile` and `git diff --check`
  pass.
- A private isolated preview over the 3 current Voice Recordings processing
  sidecars reconciled 3 conversations, 3 recordings, 245 utterances, 3
  evaluations, 11 proposals, and 0 decisions.
- All 3 live-source records exported to semantically equivalent sidecars in
  the isolated store. The live `~/.transcripts/transcripts.sqlite3` schema
  remained unmigrated.
- No provider access, model call, speaker assignment, external write, source
  artifact mutation, live database migration, or Graphiti write occurred.

Next:

- Execute C3 source, evidence, concept, retrieval-request, bundle, and
  isolation records plus exact, FTS5, relationship, timestamp, scope, and
  embedding indexes.

## Turn 228 | 2026-07-26

Summary: Completed Plan 0029 C1 as an additive, sidecar-authoritative storage
foundation without migrating the live user store.

Implemented:

- Added `conversation_knowledge_store.py` with one deep interface for schema
  lifecycle and normalized domain persistence.
- Added versioned transactional migration, private `0700` backup directories
  and `0600` integrity-checked SQLite backups, rollback to schema version 0,
  and explicit `sidecar` authority state.
- Added idempotent conversation, recording, utterance, person, source-record,
  external-identity, evaluation, and review-history interfaces.
- Added v1 relationship, concept, observation, claim, and projection-state
  tables for later Plan 0029 milestones.
- Preserved tenant/account Source Context and source affinities when several
  provider records represent one person.

Validation:

- Six C1 behavior tests pass for legacy-store compatibility, idempotent
  snapshots, cross-source identity context, immutable processing history,
  backup/rollback, and transactional migration failure.
- Transcript-store, conversation-processing, and C1 focused tests pass with
  37 tests.
- The final full suite passes with 334 tests.
- `py_compile` and `git diff --check` pass.
- No live user-store migration, speaker behavior change, provider access,
  assignment, external write, or Graphiti write occurred.

Next:

- Execute C2 shadow projection from hash-verified transcript and processing
  sidecars, with idempotent reconciliation and round-trip export receipts.

## Turn 227 | 2026-07-26

Summary: Established the durable architecture and staged implementation plan
for accumulating and retrieving conversation knowledge.

Decisions:

- The existing user-scoped transcript home is the target local storage
  authority: SQLite stores normalized records and indexes, while
  content-addressed files retain audio and immutable artifacts.
- Processing sidecars remain authoritative during shadow projection and
  become portable database exports only after explicit reconciliation,
  backup/restore, rollback, and authority-cutover gates pass.
- The domain model separates observations, claims, evaluations, review
  decisions, source records, people, external identities, relationships,
  concepts, and derived profiles.
- Evidence retrieval remains host-owned, tenant- and account-scoped,
  temporal, budgeted, duplicate-aware, and immutable before App Intelligence
  reasoning.
- Graphiti remains a reviewed compact projection, not the authority for raw
  transcripts, provider evidence, or processing history.

Durable authorities:

- `docs/adr/0002-use-a-user-scoped-conversation-knowledge-store.md`
- `docs/conversation-knowledge-storage-and-retrieval.md`
- `docs/dev/plans/0029-2026-07-26-conversation-knowledge-storage-retrieval.md`
- `CONTEXT.md` for the new retrieval, temporal, observation, claim, external
  identity, and derived-profile language.

Next:

- Execute Plan 0029 C1 as a schema-and-interface slice. Do not migrate live
  authority, change speaker behavior, or resume chronological identity
  spending until its compatibility and rollback gates pass.

## Turn 226 | 2026-07-25

Summary: Rejected Plan 0027 as a complete identity-quality repair, then
implemented and accepted Plan 0028's host-owned confidence calibration.

Implemented:

- Ran Plan 0027's bounded invalid-reference correction over the original
  regression cohort and the frozen reviewed holdout without weakening the
  prepared-evidence validators.
- Preserved the factor-derived evidence score as uncapped metadata and added a
  deterministic Medium cap for unlisted, unresolved, conflicting, mixed, or
  materially unverified speaker identities.
- Added reason-coded calibration metadata, retained the Very High plus
  no-review-flags safe-bulk gate, and added an immutable private replay receipt
  that does not mutate sealed predictions.

Outcome:

- Reference repair improved validation from 2/10 to 7/10 on regression and
  from 2/10 to 8/10 on reviewed holdout replay, but was rejected because
  High/Very High wrong identity proposals rose to 8 and 4 respectively.
- Calibration replay covered 53 reviewed person-label outcomes. Top-person
  correctness remained 17/53; High/Very High wrong proposals fell from 12 to
  0, while High/Very High correct proposals fell from 15 to 10 because five
  correct but materially uncertain proposals were capped at Medium.
- Source predictions, validation totals, and proposal ordering were unchanged.
  Automatic confirmation remains disabled until a future unseen chronological
  holdout validates the calibrated Very High band.

Validation:

- Focused identity, workflow, campaign, baseline, and API tests pass with 105
  tests; the final full suite passes with 328 tests, including the
  terminal-state replay compatibility coverage.
- Calibration receipt:
  `calibration-replay-f26f95c0-a451-451a-b561-954714085b68`.
- Accepted replay algorithm commit:
  `2d0ac75ceb8a07b4ea4574fe49e298eddf4466c8`.
- No speaker assignment, external contact mutation, CRM write, deposition, or
  memory write occurred.

## Turn 225 | 2026-07-25

Summary: Completed Plan 0026 C3-C6 through the first immutable chronological
holdout, rejected the first refinement, and paused corpus expansion at a
bounded systemic repair gate.

Implemented:

- Reviewed and privately froze the first ten eligible chronological gold
  cases, preserving transcript, participant, and correction details outside
  Git.
- Added the serial blind baseline executor, immutable per-case prediction
  capture, post-completion gold reveal, aggregate comparison metrics, failure
  taxonomy, and explicit refinement-decision receipts.
- Tested one prompt-locality refinement against the complete frozen batch,
  rejected it because its target failure count did not improve and
  High/Very High wrong proposals increased, then reverted the prompt change.
- Isolated the reserved ten-document holdout, captured all predictions before
  operator review, required post-prediction gold for reveal, and excluded one
  reviewed duplicate from quality denominators.
- Opened planned Plan 0027 for one host-mediated invalid-reference corrective
  turn without fuzzy remapping or validation weakening.

Holdout evidence:

- 10/10 predictions were captured before gold reveal; 9 cases were scorable
  after excluding one duplicate.
- Host validation completed for 2/10 predictions and rejected 8/10: four at
  Clue Discovery reference validation and four at Identity Evaluation
  reference validation.
- Calendar association was exact for 2/9 scorable cases with zero High/Very
  High wrong calendar matches.
- Speaker identity was top-correct for 5/23 known labels and present anywhere
  for 6/23; three wrong top proposals carried High/Very High confidence.
- Neither reviewed mixed label was detected.

Validation:

- Focused campaign, speaker preprocessing, and identity tests pass with 34
  tests.
- Holdout predictions record commit `377d955`; the duplicate-safe scorer is
  commit `40ce913`.
- Comparison receipt:
  `comparison-b8c1198d-c5ff-4bd9-9bd1-c36376afd37f`.
- No speaker assignment, external contact mutation, CRM write, deposition, or
  memory write occurred.

Next:

- Execute Plan 0027 once against the accumulated regression and frozen
  holdout. Resume Plan 0026 C7 only if that bounded repair materially improves
  validation yield without weakening gates or regressing identity metrics.

## Turn 224 | 2026-07-24

Summary: Implemented the private C3 campaign/gold review infrastructure for
Plan 0026; chronological operator review remains active.

Implemented:

- Added approval-gated campaign manifest apply with private `0700` directory
  and `0600` JSON permissions.
- Added private case-review packets that expose bounded transcript/calendar
  clues while explicitly never reading gold records or running App
  Intelligence.
- Added validated append-only gold records for dispositions, reviewed calendar
  association, people, per-label outcomes, same-person label groups, reviewer
  attribution, notes, and explicit supersession.
- Added campaign status without gold content and a batch freeze that requires
  exactly `K` current eligible-known records and reserves the next
  chronological holdout candidates.
- Added local API apply/status/review-packet/gold/freeze routes.

Validation:

- Targeted campaign/API tests pass for approval gates, private modes, review
  separation, tamper-safe archived fallback, append-only corrections, batch
  freeze, and the HTTP operator surface.
- No test or API path sends a prompt, runs App Intelligence, assigns a speaker,
  or performs an external write.

Next:

- Apply the live `K=10` campaign manifest from a clean commit, review the
  oldest seed cases with Eric, and freeze batch 1 before any blind baseline.

## Turn 223 | 2026-07-24

Summary: Implemented and deployed C2 of Plan 0026: safe archived-transcript
fallback with synchronized durable identities.

Implemented:

- Added a source-first transcript resolver with a stored-copy fallback that
  requires the exact DB-recorded path, confinement beneath the transcript
  store's `artifacts/` tree, the transcript suffix, and a matching SHA-256.
- Kept read-only speaker state inspection non-mutating.
- Made preparation and reviewed speaker write flows lazily add durable
  conversation/recording IDs and atomically synchronize artifact JSON, copied
  artifact, SQLite JSON/hash metadata, and the preserved original source path.
- Added the selected artifact location and hashes to preparation responses.
- Added regressions for stored fallback, tampered copies, path escape, source
  synchronization, and the selected-conversation preprocessing path.

Validation:

- `.venv/bin/python -m pytest -q tests/test_transcript_artifact_access.py
  tests/test_transcript_api.py tests/test_transcript_store.py` passed with 88
  tests.
- Python compilation and `git diff --check` passed.
- Restarted `transcripts.service`; it is active and `/api/health` returns OK.
- Read-only speaker state for document `654972c990225cc7b4f8` now returns
  `not_started` rather than the prior inaccessible-artifact HTTP 400.
- Clue Discovery preparation selected the verified `stored` artifact and
  created local run `20260725T010342Z-speaker-preprocessing-cd691372` with
  `will_send_prompt=false` and no external write.
- SQLite retains the historical `/mnt/e/...` source path while its JSON
  payload has durable conversation/recording IDs and its artifact hash matches
  the synchronized copied file.

Next:

- Implement C3's private gold-record schema and operator review surface, then
  classify seed rows and freeze the first `K=10` eligible gold cases.

## Turn 222 | 2026-07-24

Summary: Implemented C1 of Plan 0026: a deterministic, read-only
oldest-forward speaker identity campaign preview.

Implemented:

- Added `speaker_evaluation_campaign.py preview` with configurable batch size,
  store/state/runtime roots, chronological ordering, stable cursoring, and
  explicit per-row dispositions.
- Added artifact availability reporting, incomplete-artifact quarantine,
  exact normalized-transcript duplicate clustering, and candidate roles for
  the first `K` gold-review and next `K` blind-holdout rows.
- Recorded algorithm, model route, rubric, schema, and redacted
  provenance-config fingerprints while keeping transcript text out of the
  manifest.
- Kept preview strictly read-only: it neither creates campaign state nor
  executes App Intelligence or external writes.

Live evidence:

- Previewed 375 live transcript rows: 105 incomplete, 9 duplicate members, and
  261 pending operator classification across 11 duplicate clusters.
- Reserved 10 gold-review and 10 blind-holdout candidates.
- Cursor begins at chronological rank 2,
  document `654972c990225cc7b4f8`; rank 1 remains counted as incomplete.
- The first 13 legacy-path rows selected their copied `stored` artifacts in
  the preview, while later accessible rows continued to select `source`.
- Resolved App Intelligence to `codex-app-server` with model `gpt-5.6-sol`.

Next:

- Implement C2's store-bounded, hash-verified original/stored artifact
  resolver and synchronize durable transcript identities across artifact and
  store state.

## Turn 221 | 2026-07-24

Summary: Opened Plan 0026 for a chronological, oldest-forward speaker identity
test campaign and grounded its first packet in the live transcript corpus.
Canonical plan:
`docs/dev/plans/0026-2026-07-24-oldest-forward-speaker-identity-test-campaign.md`.

Current evidence:

- The live store contains 375 transcripts from 2019 through 2026; 92 retain
  legacy `/mnt/e/...` source paths.
- Copied `stored_path` artifacts exist for the oldest rows, but the current
  selected-conversation preprocessing endpoint rejects them when the original
  `source_path` is unavailable. A read-only request for the second-oldest
  substantial transcript returned HTTP 400 with
  `Selected conversation does not have an accessible transcript artifact.`
- Only three Plan 0025 processing sidecars exist, each with one evaluation and
  no review decision. The older assignment store has two confirmed labels for
  one 2026 conversation and one deferred label for another, so calendar data
  alone cannot be treated as historical ground truth.
- The first thirteen chronological rows include two one-utterance stubs,
  likely repeated imports, multi-party interviews, a canceled-event match,
  generic calendar associations, owner-address aliases, and cases where
  diarized-label count exceeds attendee count.

Plan:

- Use a configurable operational batch size, initially `K=10`, with the next
  `K` eligible known conversations reserved as a blind chronological holdout.
- Require Eric-reviewed private ground truth and keep it outside model prompts,
  retrieval queries, Git, and ordinary processing sidecars.
- Account for every chronological row with an explicit disposition rather than
  silently excluding duplicates, incomplete recordings, spurious audio, or
  unknown cases.
- Measure calendar association, speaker identity, person grouping,
  split/mixed diarization, evidence independence, retrieval yield,
  latency/cost, and reviewer workload separately.
- Refine one failure hypothesis at a time, replay preserved evidence to isolate
  algorithm changes, and accept changes only after accumulated-gold regression
  comparison.

First executable unit:

- Add a deterministic corpus enumerator and dry-run campaign manifest, then
  implement a store-bounded fallback from unavailable original transcript
  paths to verified copied artifacts before asking Eric to freeze the first
  gold batch.

## Turn 220 | 2026-07-24

Summary: Closed Plan 0025 with a two-pass, provenance-aware speaker identity
workflow and reviewed dogfood across three real conversations.

Implemented:

- Added durable conversation/recording identities and append-only
  `.processing.json` sidecars with immutable evaluations, current-evaluation
  pointers, attributable confirm/reject/defer decisions, reviewer assertions,
  and supersession history.
- Added required Source Context validation for personal GWS and
  organization-owned Odollo sources, attendee-email-first retrieval, bounded
  evidence snapshots, cross-source Person Candidate grouping, and explicit
  evidence-independence keys.
- Replaced the prototype with Clue Discovery, host retrieval, and Identity
  Evaluation phases under reviewed App Intelligence ledgers and prompt packets.
- Added separate calendar-association, person-link, and speaker-identity
  rubrics. The host derives numeric `0..100` scores plus Low, Medium, High, and
  Very High labels from cited categorical factors.
- Added selected-conversation API and React Speakers controls for both phases,
  captured evaluation persistence, evidence/warning inspection, individual
  review decisions, and safe confirmation of ready proposals only.
- Updated the App Server client for the installed 0.145 protocol and direct
  stdio lifecycle, including the current handshake, completion streaming,
  full turn-item reads, thread resume, assistant-only output extraction, and
  real timeout enforcement.

Dogfood:

- Calendar-associated case: calendar association `80 / High`; candidate and
  unlisted speaker proposals remained pending because review flags were
  present; a cross-source person grouping proposal was preserved separately.
- Split-diarization case: seven proposals represented cross-label B+E
  grouping, mixed-label exclusions, unresolved candidates, unlisted attendees,
  and a `100 / Very High` calendar match without rewriting diarization.
- Spurious/unresolved case: Speaker B persisted as `0 / Low` and unresolved;
  Speaker A's `100 / Very High` evidence score did not bypass mixed-speaker
  review flags.

Validation:

- Full backend suite: `295 passed`.
- Production frontend build passed.
- Live runs resolved through `codex-app-server / gpt-5.6-sol`; provider
  retrieval was read-only and no speaker assignment or external mutation was
  applied.

Next:

- Use the reviewed speaker identities as inputs to a separately planned
  full-conversation contextual interpretation pass.

## Turn 219 | 2026-07-24

Summary: Completed the Plan 0025 speaker-preprocessing design grill and
reconciled the resulting domain model into repo authority.

Decisions:

- Split App Intelligence preprocessing into host-governed Clue Discovery and
  Identity Evaluation phases, with bounded provenance retrieval between them.
- Separate Calendar Association Confidence, Person Link Assessment, and
  Speaker Identity Confidence under task-specific versioned evidence rubrics.
  App Intelligence returns cited factor judgments; the host validates them and
  derives evidence-strength scores and human-readable bands.
- Group duplicate cross-source people through confidence-bearing inference
  while preserving Source Records, Source Context, source affinities, and
  evidence-independence boundaries.
- Represent unlisted people, split-speaker groups, mixed-speaker findings, and
  utterance-level identity proposals without rewriting diarization.
- Keep lightweight human confirmation mandatory for speaker assignments while
  supporting safe bulk confirmation of ready proposals and structured
  calibration outcomes from corrections.
- Store immutable evaluation history in one conversation-owned JSON sidecar
  for now, with durable conversation and recording identities designed for
  later migration to central user-scoped database storage.

Documentation:

- Added `CONTEXT.md` as the conversation-intelligence ubiquitous-language
  glossary.
- Added ADR 0001 for durable opaque conversation identities.
- Revised Plan 0025 and the P09 roadmap focus to reflect the two-phase design
  and distinguish it from the implemented v1 single-packet prototype.

Next:

- Implement the bounded Plan 0025 successor contracts from Clue Discovery
  through reviewed Identity Evaluation, beginning with durable IDs, Source
  Context validation, and sidecar schema tests.

## Turn 218 | 2026-07-21

Summary: Opened Plan 0025 for App Intelligence speaker preprocessing.

Current evidence:

- Participant identity already extracts calendar attendees and queries Google
  People plus Odollo contacts, but every anonymous speaker receives the same
  candidate pool and there is no transcript-clue LLM pass.
- General provenance has Calendar/Drive and Odollo log-note adapters, while the
  identity path does not yet include Gmail, Odollo leads, or log-note evidence.
- Codex app-server readiness is healthy on local Codex CLI 0.144.6. The live
  workstation Codex model is `gpt-5.6-sol`; the repo's Codex supervisor profile
  previously left its model unspecified.

Direction:

- Keep provenance acquisition deterministic and host-owned.
- Prioritize exact calendar attendee emails for reverse lookup.
- Give App Intelligence a bounded transcript/evidence packet and require cited
  speaker proposals under a strict schema.
- Require human review before any speaker assignment and defer the broader
  contextual conversation pass.

Validation started:

- Added a red default-routing test, reproduced the empty Codex model, then
  pinned speaker disambiguation and the app supervisor to `gpt-5.6-sol`; the
  focused test now passes.
- Added TDD coverage for speaker-specific clue packets, attendee-email-first
  candidate ordering, strict model-reference validation, JSON-only prompt
  rendering, configured provenance collection, and Odollo lead promotion.
- Added metadata/snippet-only Gmail evidence collection through the inspected
  `gws gmail users messages list/get` contracts and read-only `crm.lead`
  provenance through configured Odollo profiles.
- Shared provenance config now maps GWS Gmail limits and Odollo lead model
  selection into the source adapters.
- Focused backend validation passed: `44 passed`; the full backend suite passed
  with `267 passed`; touched modules/tests compile, `git diff --check` passes,
  and live config resolution reports
  `speaker_disambiguation -> codex-app-server / gpt-5.6-sol` with a required
  ledger and low-confidence review.
- Applied a reviewed user-scoped provenance update that enables `crm.lead`
  between `res.partner` and `mail.message` for both existing Odollo tenant
  sources. Redacted readback confirms contacts, leads, and log notes are all
  enabled for both profiles.
- Restarted `transcripts.service`; it is active with `NRestarts=0`, and the live
  `/api/intelligence/config` readback reports `gpt-5.6-sol` for both speaker
  disambiguation and the App Intelligence supervisor.

Next:

- Add the conversation/API action that prepares this packet as a reviewed App
  Intelligence run/prompt artifact, then dogfood it on calendar-associated
  transcripts before enabling any automation.

## Turn 217 | 2026-07-20

Summary: Closed Plan 0024 after cutting Voice Recordings over to the restored
D: Syncthing root and completing the real backlog.

Changes:

- Updated `syncthing-voice-recordings` from the degraded `/mnt/e` root to
  `/mnt/d/SyncThing/Voice Recordings`.
- Reconciled 89 moved recordings through existing successful watcher records
  without retranscribing them.
- Processed five genuinely new valid recordings through transcription,
  calendar matching, transcript-store ingestion, and participant identity
  warming.
- Added `--path-prefix-remap OLD=NEW` to calendar repair so directory moves
  rebase only artifact media/output path fields, never transcript content.
- Applied the previously pending calendar repair against the restored
  authoritative artifact and reconciled the transcript store to one canonical
  row.
- Excluded Syncthing `.stversions` paths from recursive intake after the only
  remaining candidate proved to be an incomplete historical version rather
  than live backlog.

Validation:

- The pre-cutover watcher doctor reproduced one `unavailable_watch_dir`
  warning for `/mnt/e`; the post-cutover doctor returned `status: ok` with no
  issues.
- The path-remap and `.stversions` regressions failed before their fixes and
  passed afterward.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed:
  `32 passed`.
- `.venv/bin/python -m py_compile watch_transcriptions.py
  repair_calendar_metadata.py tests/test_transcript_artifacts.py` passed.
- The repair dry run reported one matched artifact and zero without an event;
  apply/readback confirmed D: media/output paths and event metadata.
- Final live heartbeat reported `candidates=0 attempted=0 successes=0
  failures=0 blocked=none`; `transcribe-watch.service` was active with
  `NRestarts=0`.
- `git diff --check` passed.

## Turn 216 | 2026-07-20

Summary: Closed Plan 0023 after recovering the live watcher from a stale
optional watch mount and preserving the confirmed calendar repair safely.

Current evidence:

- Windows currently has no `E:` filesystem drive, while WSL retains a stale
  `/mnt/e` drvfs mount whose Voice Recordings path raises `OSError: No such
  device`.
- `transcribe-watch.service` was in an `ExecStartPre` auto-restart loop because
  readiness checks treated the one unavailable Voice Recordings root as a
  global service failure before the three healthy watcher jobs could run.
- Three recent stored artifacts have `event: null`. Current primary-calendar
  replay gives one a strong overlapping event; the other two have no
  overlapping timed Google Calendar event. Exact private repair evidence is
  retained only in user-scoped runtime state.

Changes:

- Added a red regression test for one unavailable root among healthy jobs.
- Made unavailable-root readiness and scans job-local while preserving a fatal
  result when every configured watch root is unavailable.
- Successful transcriptions that continue after calendar lookup failures now
  persist `warning_kind=calendar_metadata_failed` and a bounded
  `warning_reason` across watcher state save/load.
- Updated README readiness/recovery guidance and the P06 roadmap state.
- Recorded the exact confirmed repair as a user-scoped pending repair because
  its authoritative drive remains absent; no store-only rewrite was attempted.

Validation:

- The deterministic unavailable-root regression failed with the production
  `OSError: No such device` before the fix and passed afterward.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed:
  `30 passed`.
- `.venv/bin/python -m py_compile watch_transcriptions.py
  tests/test_transcript_artifacts.py` passed.
- `git diff --check` passed.
- `watch_transcriptions.py --check --check-json` returned `status: ok` with one
  `unavailable_watch_dir` warning.
- `transcribe-watch.service` is active with `NRestarts=0`; its live heartbeat
  reports `candidates=0 attempted=0 successes=0 failures=0
  blocked=unavailable_watch_dir=1`.

Residual:

- Windows still has no `E:` filesystem drive. The confirmed metadata repair
  remains gated on restoring the authoritative Voice Recordings files, running
  a fresh dry-run, and reconciling the transcript store after apply.

## Turn 215 | 2026-07-18

Summary: Migrate the transcripts service Codex readiness path to the stable
user-scoped standalone command.

Changes:

- The manual service authority is `~/.local/bin/codex`, which resolves through
  the standalone Codex installer rather than an NVM version tree.
- Update only
  `~/.config/systemd/user/transcripts.service.d/10-codex-bin.conf`; preserve
  the rest of the service environment and unit contract.
- Validate readiness with `codex --version`, `codex app-server --help`,
  `codex app-server generate-json-schema --help`, and
  `codex app-server generate-ts --help`. These probes do not start a model turn
  or send transcript/private input.
- Rollback restores the protected pre-migration drop-in, runs one user daemon
  reload, and restarts `transcripts.service`.

Validation:

- `systemd-analyze --user verify` passed before daemon reload.
- `transcripts.service` restarted from PID 1070 to PID 98440 and remained
  active with `NRestarts=0`.
- `GET /api/health` returned `status: ok`.
- `GET /api/intelligence/providers` reported the Codex app-server ready on
  `/home/ecochran76/.local/bin/codex` at `codex-cli 0.144.5`; version,
  app-server help, JSON-schema help, and TypeScript-generation help checks all
  returned success without a model turn or private payload.
- The workstation Node checker fell from 43 to 42 findings, with versioned NVM
  references falling from 30 to 29 and zero incomplete coverage.

## Turn 214 | 2026-05-30

Summary: Added AuraCall agent selection to Intelligence settings.

Changes:

- Extended the redacted AuraCall choices payload with `agent_options` so the
  review console can list runtime-advertised agents without exposing API
  secrets.
- Settings > Intelligence now renders an AuraCall agent selector instead of a
  free-form model field when the profile is AuraCall-backed or already uses an
  `agent:<id>` model.
- The selected agent summary shows the runtime profile, browser profile,
  project binding, binding key, readiness, validation state, and selected
  `agent:<id>` model.
- Updated P09 roadmap text for the agent-selector settings behavior.

Validation:

- `.venv/bin/python -m py_compile auracall_choices.py transcript_api.py tests/test_transcript_store.py tests/test_transcript_api.py`
- `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_choices_readiness_validates_dispatch_team_members tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing tests/test_transcript_api.py::test_intelligence_config_endpoint_exposes_auracall_agent_options -q`
- `npm --prefix frontend run build`

## Turn 213 | 2026-05-29

Summary: Added AuraCall agent-choice readiness to first-pass summary prep.

Changes:

- Added a redacted AuraCall choices reader for
  `GET /v1/config/agent-choices`.
- First-pass summary prepare/enqueue now prefers `AURACALL_AGENT_ID` for
  single-agent runs, preserves `AURACALL_DISPATCH_TEAM` dispatch-pool routing,
  and writes `auracall_readiness` into manifests.
- `/api/intelligence/config` now exposes the same redacted readiness for the
  review console without exposing API secrets.
- Updated `README.md` and `ROADMAP.md` with the stable-agent and readiness
  contract.
- Live scoped AuraCall choices readback now sees
  `transcribe-audio-chatgpt-pro-pool` with three ready members after the
  AuraCall runtime was rebuilt/restarted and the registry team was restored.
- Live provider submit/materialize was not run because
  `transcript_store.py first-pass-summary-queue --format compact-json --limit 5`
  returned zero queued first-pass items.

Validation:

- `.venv/bin/python -m py_compile auracall_choices.py scripts/auracall_legacy_enrichment_batch.py transcript_api.py tests/test_transcript_store.py tests/test_transcript_api.py`
- `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest tests/test_transcript_store.py::test_auracall_first_pass_prepare_can_use_dispatch_team tests/test_transcript_store.py::test_auracall_first_pass_prepare_prefers_stable_agent_id tests/test_transcript_store.py::test_auracall_choices_readiness_validates_dispatch_team_members -q`
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints tests/test_transcript_api.py::test_selected_first_pass_summary_prepare_is_conversation_scoped tests/test_transcript_api.py::test_first_pass_summary_submit_and_status_use_prepared_manifest -q`

## Turn 212 | 2026-05-26

Summary: Tidied planning-contract workspace state.

Changes:

- Normalized `docs/dev/plans/0013-2026-05-24-user-scoped-provenance-config.md`
  from `State: COMPLETE` to the repo's deterministic `State: CLOSED`
  vocabulary.
- Added the exact
  `docs/dev/plans/0020-2026-05-25-intelligence-profile-settings-redesign.md`
  filename to the existing Plan 0020 runbook closeout so the planning audit can
  verify it is wired.

Validation:

- `python repo-policy-selector/scripts/audit_planning_contract.py ... --json`
  passed after the tidy.
- `git diff --check` passed.

## Turn 211 | 2026-05-26

Summary: Installed Graphiti and codegraph policy guidance.

Changes:

- Extended `docs/dev/policies/0005-memory-and-context-routing.md` with the
  shared `graph-backed-memory-usage` discipline and the repo-local Graphiti
  group `transcribe_audio_main`.
- Added `docs/dev/policies/0007-codegraph-usage.md` for the installed
  CodeGraph CLI, sibling `../codegraph` checkout, and local `.codegraph/`
  index.
- Wired the new codegraph policy into `AGENTS.md`.

Validation:

- Used `$repo-policy-selector` and enumerated the installed policy catalog from
  `repo-policy-selector/policy-library/catalog.yaml`; the requested modules map
  to `graph-backed-memory-usage` and `codegraph-usage`.
- `~/.local/bin/graphiti-runtime doctor` reported healthy, and Graphiti
  discovery found existing repo guidance for `transcribe_audio_main`.
- Initial `python repo-policy-selector/scripts/select_policy.py ... --json`
  output found `graph-backed-memory-usage` semantically covered and
  `codegraph-usage` missing. The post-patch selector re-run reports both modules
  in `already_adopted_modules` with `validation_problems=[]`.
- `codegraph status . --json` reported this repo is initialized with
  `.codegraph/codegraph.db`, 65 indexed files, and no skipped files. After
  `codegraph sync .`, pending added/modified/removed files were all zero.
- `python repo-policy-selector/scripts/audit_planning_contract.py ... --json`
  initially reported pre-existing planning-contract issues unrelated to this
  policy install: Plan 0013 used a nonstandard state value, and Plan 0020 was
  described without the exact plan filename expected by the audit.

## Turn 210 | 2026-05-25

Summary: Closed Plan 0022, the Settings layout refactor.

Changes:

- Added a Settings-specific workspace class that hides the transcript left pane
  and inspector and gives Settings a centered 1180px work area.
- Removed global conversation/artifact/review summary chips from Settings and
  replaced them with a compact settings-only status row.
- Refactored the Settings section navigation so labels and status chips do not
  collide on desktop or mobile.
- Made the Intelligence route matrix fit the widened Settings detail surface
  and collapse without horizontal overflow on mobile.
- Renamed the Settings Evidence section to Validation, replaced visible smoke
  wording with validation/browser-check wording, and moved raw artifact paths
  into a disclosure.
- Updated Plan 0022 and P09 roadmap state.

Validation:

- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- Restarted `transcripts.service`; `http://transcripts.localhost/api/health`
  returned `status: ok`.
- Source inspection showed Settings section selection and already-loaded draft
  edits use local React state setters, while preview/apply keep the reviewed
  endpoint boundaries.
- `agent-browser` desktop `1440x960` inspection of Settings > Intelligence
  measured `center.width=1180`, `settingsSurface.width=952`,
  `leftPane.display=none`, `rightPane.display=none`, and
  `routeOverflow=false`; visible text checks found no
  conversation/artifact/open-review/latest-smoke/smoke-report labels.
- `agent-browser` mobile `390x844` inspection of Settings > Account and
  Settings > Intelligence measured no section-nav text overlap, no route-row
  overflow, and first actionable controls inside the first viewport.
- `agent-browser` keyboard reachability inspection verified every Settings
  section button is enabled with `tabIndex=0`.
- `agent-browser` network capture after local section selection and local draft
  interaction reported `No requests captured`.
- `agent-browser` console and page-error checks reported no output.
- Screenshots:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-desktop-intelligence.png`,
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-mobile-account.png`,
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0022-settings-mobile-intelligence.png`.

## Turn 209 | 2026-05-25

Summary: Planned Plan 0022, the Settings layout refactor based on the live UI audit.

Findings:

- The UI audit showed Settings still constrained by the transcript three-pane
  shell: transcript filters/runtime chrome on the left, transcript inspector on
  the right, and a narrow settings detail surface in the center.
- The Intelligence component/profile/policy matrix clipped columns at desktop
  width because fixed minimum columns exceeded the available Settings surface.
- Mobile Settings navigation overlapped label and meta text when the section
  rail became two columns.
- Global conversation/artifact/review summary chips still consumed
  first-viewport space on a configuration screen.
- The Settings Evidence section still used user-visible smoke terminology even
  after Plan 0021 removed the main diagnostic strip from Settings.

Changes:

- Added `docs/dev/plans/0022-2026-05-25-settings-layout-refactor.md`.
- Wired Plan 0022 into the P09 roadmap plan list and milestone focus.

Validation:

- Used the `ui-design` Product UI track for the refactor direction.
- `~/.local/bin/graphiti-runtime doctor` was healthy; discovery returned stale
  early-roadmap context for this query, so repo docs and live UI audit evidence
  were used as authority.
- Reviewed planning/productization policies before creating the plan.

## Turn 208 | 2026-05-25

Summary: Closed Plan 0021, removing Library/test-status chrome from Settings.

Findings:

- Settings still inherited the non-Library `TestStatusStrip`, so it showed
  Library-oriented rows-in-scope/filter copy and "Latest smoke" on a
  configuration screen.
- The apparent "API Preview" versus "API offline" contradiction came from the
  same frontend health state being rendered by two components: the test strip
  mapped non-`ok` health to "Preview", while the Settings status card printed
  the raw fallback status.
- The no-staged-edits bar was also taking space when it offered no action.
- Graphiti discovery was healthy but stale for current Plan 0021 state, so repo
  docs, source, and live browser evidence were authoritative.

Changes:

- Added and closed
  `docs/dev/plans/0021-2026-05-25-settings-screen-chrome-cleanup.md`.
- Removed the non-Library `TestStatusStrip` render path so Settings no longer
  shows rows-in-scope, filter state, API preview, or latest-smoke diagnostics.
- Removed the Settings status card and duplicate API/config-path summary.
- Made the staged-config bar render only when there is a local draft or
  prepared preview.
- Removed the Settings-specific left-pane card and prevented generic Library
  kind filters from falling through into Settings.
- Updated P09 roadmap state.

Validation:

- `~/.local/bin/graphiti-runtime doctor` was healthy; discovery for current
  settings/status strip context returned stale facts.
- `python -m py_compile transcript_api.py intelligence_config.py` passed.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- `.venv/bin/python -m pytest -q` passed with 246 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` inspection of Settings > Intelligence verified
  `hasOperatorTestStatus=false`, `hasRowsInScope=false`,
  `hasLatestSmoke=false`, `hasApiPreview=false`, `hasApiOffline=false`,
  `hasNoStagedConfigEdits=false`, `hasDraftBar=false`, and `routeRowCount=8`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0021-settings-chrome-cleanup.png`.
- `agent-browser` console/error checks reported no page errors.

## Turn 207 | 2026-05-25

Summary: Closed Plan 0020, a profile-first redesign of Intelligence settings.

Findings:

- The Settings > Intelligence surface still treated provider/model selection as
  per-component route data, even though the product model should define named
  profiles once and let components select those profiles.
- The broader Settings header used too much first-viewport space for
  inactionable config paths and status facts. Long paths wrapped into narrow
  columns and hid the actual workbench controls.
- `agent-browser` confirmed the original profile page direction but also
  exposed the wasteful status/header layout, so the slice included that
  compaction instead of stopping at the route editor.
- Graphiti discovery was healthy but stale for this recent P09 settings work,
  so repo docs, code, and live UI evidence remained authoritative.

Changes:

- Added named intelligence profiles and `task_profiles` assignments to
  `intelligence_config.py`.
- Kept legacy task-level provider/model overrides compatible, but profile
  selection now clears component provider/model route fields so the profile is
  the source of truth.
- Extended `/api/intelligence/config/preview` and `/apply` so profile-only
  edits can be previewed/applied without requiring a task update.
- Reworked Settings > Intelligence into one page with profile definitions,
  component profile selections, and closed config/resolved-route details.
- Compacted the Settings status area into pills plus a closed
  "Config paths and runtime facts" disclosure, and changed the page heading from
  "Account settings" to "Settings".
- Closed `docs/dev/plans/0020-2026-05-25-intelligence-profile-settings-redesign.md`
  and updated P09 roadmap state.

Validation:

- `python -m py_compile intelligence_config.py transcript_api.py` passed.
- `~/.local/bin/graphiti-runtime doctor` was healthy; discovery for current
  P09/Plan 0020 context returned older state.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py
  tests/test_transcript_api.py::test_intelligence_config_endpoint_returns_task_routing
  tests/test_transcript_api.py::test_intelligence_config_preview_and_apply_endpoints
  -q` passed with 12 tests.
- `.venv/bin/python -m pytest -q` passed with 246 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` inspection of Settings > Intelligence verified
  `title=Settings`, `routeRowCount=8`, `compactSectionCount=2`,
  `detailsOpen=false`, `statusDetailsOpen=false`,
  `hasOldRouteMapText=false`, and `hasRuntimePathCards=false`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0020-intelligence-settings-profile-page.png`.
- `agent-browser` console/error checks reported no page errors.

## Turn 206 | 2026-05-25

Summary: Closed Plan 0019, a polish slice for the one-click initial-summary
workflow prep surface.

Findings:

- Plan 0015 had already created the backend one-click run action and automation
  settings, but the First-pass summary tab still rendered Run initial summary,
  Prepare only, Submit, and Check as peer buttons.
- The correct follow-up was UI ergonomics and browser-smoke protection, not a
  new provider or automation contract.
- Graphiti discovery was healthy but stale for current P09 state, so
  `ROADMAP.md`, `RUNBOOK.md`, and the numbered plan files remained the
  authority.

Changes:

- Added and closed
  `docs/dev/plans/0019-2026-05-25-one-click-summary-workflow-prep-polish.md`.
- Replaced the summary-stage peer button row with an `Initial summary prep`
  card that exposes one primary next action.
- The primary action now advances based on state: run initial summary, submit a
  prepared-only manifest, check a submitted manifest, or stay disabled when a
  summary is already ready.
- Moved Prepare only, Submit, and Check into `Advanced summary controls`.
- Extended `scripts/smoke_conversation_review_loop_ui.py` to assert the
  summary prep card, exactly one primary summary action, advanced controls, and
  no inline summary button cluster.
- Updated P09 roadmap state.

Validation:

- `~/.local/bin/graphiti-runtime doctor` was healthy.
- `~/.local/bin/graphiti-runtime discover --group-id transcribe_audio_main
  "one click first pass summary automation settings workflow prep Plan 0015
  current status"` returned stale facts, so repo docs were used.
- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_conversation_review_loop_ui.py` passed.
- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py participant_identity.py` passed.
- `.venv/bin/python -m pytest
  tests/test_transcript_api.py::test_selected_first_pass_summary_run_endpoint_prepares_and_submits
  -q` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `scripts/smoke_conversation_review_loop_ui.py` passed against
  `http://transcripts.localhost` and wrote
  `~/.local/state/transcribe-audio/browser-smokes/20260525T224138Z-conversation-review-loop-smoke.json`.
- Smoke checks included `summary_hasSummaryPrepCard=true`,
  `summary_summaryPrimaryActionCount=1`,
  `summary_hasAdvancedSummaryControls=true`, and
  `summary_hasInlineSummaryButtonCluster=false`.
- Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/20260525T224138Z-conversation-review-loop-smoke.png`.
- `agent-browser` console/error checks reported no page errors.

## Turn 205 | 2026-05-25

Summary: Fixed the Library conversation table width regression from the Plan
0018 landing-page refactor.

Findings:

- At wide desktop viewports, `.conversation-table` kept `width: max-content`
  from the resizable-column implementation, so the table stayed near the
  default column-width total while its containing section grew wider.
- `agent-browser` measured a 1920px viewport with a 1458px table shell and a
  1090px table before the fix.

Changes:

- Kept the Library table at `width: 100%` while setting its minimum width from
  the current resizable column-width total.
- Made `.table-shell` explicitly fill its containing section.
- Added a Library share/deep-link smoke assertion that the rendered table fills
  the table shell.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_library_deeplink_share_ui.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` measured the fixed 1920px layout with a 1458px bordered
  table shell and a 1456px table content width.
- `scripts/smoke_library_deeplink_share_ui.py --viewport 1920x1100` passed
  with `layout_tableShellWidth=1456`, `layout_tableWidth=1456`, and
  `layout_tableFillsShell=true`; report:
  `~/.local/state/transcribe-audio/browser-smokes/20260525T223249Z-library-share-ui-smoke.json`.
- `agent-browser` console/error checks reported no page errors for the smoke
  session.

## Turn 204 | 2026-05-25

Summary: Completed Plan 0018, the landing page navigation and layout redesign.

Findings:

- The Plan 0018 audit was accurate: root navigation mixed workflow and admin
  concepts, search belonged to the Library workbench, and mobile buried the
  table behind header and filters.
- The refactor did not require backend API changes; existing URL state and
  conversation search endpoints could be reused.
- `agent-browser` initially captured fallback rows during a service restart;
  recapturing after health stabilized produced live Library counts and row
  evidence.

Changes:

- Reduced primary nav to workflow destinations: `Library` and `Review Queue`.
- Added an upper-right account chip/menu with Settings, Account management,
  Integrations/provenance, Intelligence, Automation, and Runtime status.
- Moved search and artifact-kind controls into a Library toolbar.
- Collapsed Library filters by default and made them open from the toolbar.
- Reworked responsive layout so mobile shows the Library content before
  filters and inspector.
- Updated `scripts/smoke_library_deeplink_share_ui.py` for the new Library
  search and kind controls.
- Closed Plan 0018 and updated P09 roadmap state.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  52 tests.
- `.venv/bin/python -m pytest -q` passed with 243 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `scripts/smoke_library_deeplink_share_ui.py` passed and wrote
  `~/.local/state/transcribe-audio/browser-smokes/20260525T213630Z-library-share-ui-smoke.json`.
- `agent-browser` captured:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-landing-desktop-final.png`,
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-landing-mobile-final.png`,
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0018-account-menu.png`.
- Browser metrics verified desktop first row at y=387, mobile topbar 113px,
  mobile center pane y=125, mobile table y=599, mobile first row y=661, no
  global search, Library search present, filters hidden by default, and primary
  nav limited to `Library` and `Review Queue`.
- Account-menu snapshot showed Settings, Account management,
  Integrations/provenance, Intelligence, Automation, and Runtime status.
- Library filter interaction showed the filter pane opens on demand.
- `agent-browser` console/error checks reported no page errors.

## Turn 203 | 2026-05-25

Summary: Audited the current root landing page and opened Plan 0018 for navigation/layout redesign.

Findings:

- The current root Library page is a dense console surface: brand, eight nav
  pills, disabled planned destinations, global search, filters, Library table,
  operator status, and inspector all compete in the first desktop viewport.
- Settings are exposed as a top-level nav item, while app-standard account and
  configuration affordances are missing from the upper-right chrome.
- Search is implemented as a global topbar control, but the actual user model
  is Library/workbench search.
- At 390x844, `agent-browser` measured the topbar at about 196px high, the
  main Library work surface starting around y=1014, and the inspector around
  y=9923; the first mobile viewport is mostly header and filters.
- Graphiti discovery was healthy but stale for this recent P09 UI work, so the
  current source, roadmap, runbook, and browser screenshots are authoritative.

Changes:

- Added `docs/dev/plans/0018-2026-05-25-landing-page-navigation-redesign.md`.
- Wired Plan 0018 into P09 in `ROADMAP.md`.

Validation:

- Read the UI audit and agent-browser skills plus the repo planning,
  architecture, and memory/context policies.
- Ran `~/.local/bin/graphiti-runtime doctor`; Graphiti was healthy.
- Ran `~/.local/bin/graphiti-runtime discover --group-id transcribe_audio_main`
  for P09 navigation/settings/search context; useful results were older than
  current repo planning state.
- Used `agent-browser` against `http://transcripts.localhost/` and captured:
  `~/.local/state/transcribe-audio/ui-audits/2026-05-25-landing-desktop.png`
  and
  `~/.local/state/transcribe-audio/ui-audits/2026-05-25-landing-mobile.png`.
- Verified the screenshots are PNG files at 1440x1100 and 390x844.
- `agent-browser` console/error checks reported no page errors.
- No frontend implementation code was changed.

## Turn 202 | 2026-05-25

Summary: Completed Plan 0017, the Settings config workbench implementation.

Findings:

- Plan 0016's design contract could be implemented without new backend
  endpoints by reusing the existing intelligence, automation, and provenance
  preview/apply APIs.
- Initial browser smoke found a stale Apply affordance after Discard; clearing
  preview action state on draft edits and discard fixed it.
- Browser network logs confirmed that automation, intelligence, and provenance
  local edits stay local until explicit Preview.

Changes:

- Replaced the simple Settings tab with a config workbench: status header,
  section rail, dirty/preview/apply bar, and Account, Intelligence, Automation,
  Provenance, Safety, and Evidence sections.
- Added local draft dirty detection for intelligence, automation, and provenance
  settings.
- Added Discard behavior that resets local drafts and clears stale preview
  action state.
- Added restrained Settings-specific CSS for the workbench layout, row groups,
  sticky dirty bar, mobile section rail, status chips, and evidence/safety rows.
- Closed Plan 0017 and updated P09 roadmap state.

Validation:

- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py` passed.
- Focused config endpoint pytest passed with 4 tests.
- `.venv/bin/python -m pytest -q` passed with 243 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; service was active and
  `http://transcripts.localhost/api/health` returned `status: ok`.
- `agent-browser` captured desktop and mobile screenshots:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0017-settings-workbench-desktop.png`
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0017-settings-workbench-mobile.png`.
- `agent-browser` network checks showed no backend requests for automation
  toggle, intelligence model edit, or provenance source toggle before Preview.
- `agent-browser` automation Preview made one
  `POST /api/automation/config/preview` request and page text showed
  `will_execute_workflow_stage=false`.
- `agent-browser` console/error checks reported no page errors.

## Turn 201 | 2026-05-25

Summary: Checkpointed Plans 0013-0016 and opened Plan 0017 for the Settings config workbench.

Findings:

- The dirty worktree contained the accumulated Plan 0013-0016 milestone work:
  shared user-scoped provenance config, contact-search workbench, one-click
  initial summary automation settings, and the config-panel design contract.
- Graphiti was healthy, but discovery for Plan 0017/P09 work returned stale
  repo-memory entries, so the current roadmap, runbook, and plan files are the
  authority.
- Plan 0016 defines the implementation target and requires `agent-browser`
  evidence that routine local edits do not make backend calls before explicit
  Preview/Apply.

Changes:

- Created checkpoint commit
  `926dc65 Checkpoint provenance and settings milestones`.
- Added `docs/dev/plans/0017-2026-05-25-settings-config-workbench.md`.
- Wired Plan 0017 into P09 in `ROADMAP.md`.

Validation:

- Secret scan of checkpoint candidates found only documentation, placeholders,
  env refs, and redaction tests; no live private iCalendar URL surfaced.
- `git diff --check` passed before checkpoint.
- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py provenance_config.py participant_identity.py
  transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py
  repair_calendar_metadata.py route_transcript.py watch_transcriptions.py`
  passed before checkpoint.
- `npm --prefix frontend run build` passed before checkpoint.
- `.venv/bin/python -m pytest -q` passed with 243 tests before checkpoint.

## Turn 200 | 2026-05-25

Summary: Closed Plan 0016 as a completed design-only configuration panel slice.

Findings:

- Plan 0016's scope is explicitly design-only: it authorizes planning docs,
  read-only API checks, and `agent-browser` inspection evidence, but no React,
  CSS, Python, API, schema, provider, source-refresh, or workflow-stage changes.
- The plan already defines the required aesthetics, information architecture,
  section model, draft/preview/apply lifecycle, component contract,
  accessibility expectations, and required browser inspection gates.
- The baseline browser evidence exists as PNG files at the expected desktop and
  mobile sizes.

Changes:

- Marked Plan 0016 `CLOSED`.
- Added Plan 0016 closeout notes clarifying that post-implementation browser
  checks remain required for the next build slice.
- Updated P09 roadmap text to show Plan 0016 closed as the design authority.

Validation:

- Re-read Plan 0016 and the repo planning, runtime-state, architecture, and
  memory/context policies.
- Ran `~/.local/bin/graphiti-runtime doctor`; Graphiti was healthy.
- Ran `~/.local/bin/graphiti-runtime discover --group-id transcribe_audio_main`
  for Plan 0016/config-panel context; results remained older than the current
  repo docs, so current files are authoritative.
- Verified the baseline screenshots exist:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-desktop.png`
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-mobile.png`.
- Verified those screenshots are PNG files at 1440x1100 and 390x844.
- Ran `git diff --check` for tracked docs and a trailing-whitespace check for
  the new plan, roadmap, and runbook files.
- No implementation code was changed.

## Turn 199 | 2026-05-25

Summary: Planned the design path for the full configuration panel before code.

Findings:

- The current Settings tab from Plan 0015 is functional but still a first-pass
  page: account/runtime status, intelligence route summary, and automation rows
  are present, while provenance, safety gates, staged-change evidence, and
  browser-validation expectations are not yet unified into one workbench.
- The correct design track is dense Product UI: a calm operations console with
  restrained surfaces, semantic status colors, row-based matrices, and
  user-scoped runtime paths rather than marketing-style panels.
- Graphiti discovery was healthy but stale for the recent P09 settings work, so
  current `ROADMAP.md`, `RUNBOOK.md`, Plan 0015, and source files remain the
  authority.

Changes:

- Added `docs/dev/plans/0016-2026-05-25-config-panel-design-path.md`.
- Wired Plan 0016 into P09 in `ROADMAP.md`.
- Defined the Settings workbench target: Account, Intelligence, Automation,
  Provenance, Safety, and Evidence sections with a persistent dirty
  preview/apply lifecycle.
- Required `$agent-browser` for baseline inspection and later implementation
  validation.

Validation:

- Read the repo planning, runtime-state, architecture, and memory/context
  policies.
- Ran `~/.local/bin/graphiti-runtime doctor`; Graphiti was healthy.
- Ran `~/.local/bin/graphiti-runtime discover --group-id transcribe_audio_main`
  for config-panel/P09 context; useful results were older than current repo
  planning state.
- Reviewed `ROADMAP.md`, `RUNBOOK.md`, Plan 0015, current Settings component,
  and current Settings CSS.
- Used `agent-browser` against `http://transcripts.localhost` to inspect the
  current Settings UI and captured baseline screenshots:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-desktop.png`
  and
  `~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-mobile.png`.
- No implementation code was changed.

## Turn 198 | 2026-05-25

Summary: Opened Plan 0015 for one-click initial summary and automation settings.

Findings:

- The selected-conversation first-pass summary workflow already has scoped
  prepare, submit, and status operations, but the operator UI still requires
  separate clicks for the normal initial-summary path.
- Intelligence task routing already has a user-scoped config, while workflow
  stage automation needs a separate user-scoped policy config so provider choice
  and auto-run policy do not collapse into one setting.
- Production automation should remain disabled/manual by default until every
  stage from ingestion through final readout has targeted tests and browser
  smoke evidence.

Changes:

- Added `docs/dev/plans/0015-2026-05-25-one-click-initial-summary-automation-settings.md`.
- Wired Plan 0015 into P09 in `ROADMAP.md`.
- Added `automation_config.py` with user-scoped stage policy defaults,
  preview/apply support, CLI inspection/update commands, and
  `APPLY_AUTOMATION_CONFIG_UPDATE` gating.
- Added `POST /api/conversations/<id>/first-pass-summary/run` so the selected
  initial summary prepares and submits in one reviewed action using the existing
  `SUBMIT_FIRST_PASS_SUMMARY_BATCH` token.
- Added `GET /api/automation/config`, `/preview`, and `/apply`; reads and
  previews never run workflow stages, and apply writes only user-scoped config.
- Enabled the React Settings tab with account/runtime status, intelligence route
  summaries, and automation stage toggles. The conversation summary view now
  exposes `Run initial summary` as the primary action while keeping Prepare only,
  Submit, and Check as secondary/resume controls.
- Updated README and API docs for the one-click initial summary and automation
  config boundaries.

Validation:

- `python -m py_compile transcript_api.py intelligence_config.py
  automation_config.py` passed.
- `.venv/bin/python -m pytest
  tests/test_transcript_api.py::test_selected_first_pass_summary_run_endpoint_prepares_and_submits
  tests/test_transcript_api.py::test_automation_config_endpoint_defaults_preview_and_apply
  -q` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py
  tests/test_intelligence_config.py -q` passed with 59 tests.
- `.venv/bin/python -m pytest -q` passed with 243 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `systemctl --user is-active
  transcripts.service` returned `active`.
- Direct and ingress health checks returned HTTP 200 with `status: ok`.
- Live `GET /api/automation/config` returned
  `schema_version=transcribe-audio.automation-config.v1`, `exists=false`, and
  every workflow stage disabled/manual.
- Live `GET /api/conversations/6e8eee4f19a1d5a9b23f/first-pass-summary`
  returned `status=needs_summary` and the
  `SUBMIT_FIRST_PASS_SUMMARY_BATCH` future token.
- `agent-browser` smoke on the live UI verified `Run initial summary` appears
  in the selected conversation summary tab and the Settings tab shows Account,
  Intelligence, and six disabled/manual automation stages. No live provider
  submission button was clicked. Screenshot:
  `~/.local/state/transcribe-audio/browser-smokes/plan-0015-settings-summary-smoke.png`.

## Turn 197 | 2026-05-25

Summary: Completed Plan 0014, the contact search workbench.

Findings:

- The remaining Plan 0014 gaps were explicit refresh/job APIs, relationship
  affinity ranking, merge/split persistence, richer candidate evidence in the
  UI, and browser proof that selection stays local until Save.
- Cache-only search and explicit source refresh needed to stay separate:
  typing/search ranking can use local state and cached affinity, while provider
  refresh is an operator action with read-only source/job records.
- Broad searches such as `chris` work better when deterministic ranking uses
  communication recency/frequency and calendar/local transcript overlap without
  treating affinity as identity proof.

Changes:

- Added user-scoped contact refresh job, contact affinity cache, and
  merge/split decision state under `~/.local/state/transcribe-audio/`.
- Added contact refresh preview/refresh/job-read endpoints, contact-affinity
  read/refresh endpoints, and `contact-merge-batch`.
- Ranked search results with deterministic text, conversation, affinity,
  source-quality, and operator-history score components plus visible reasons.
- Made reviewed merge/split decisions feed deterministic candidate generation.
- Extended the React context workbench with cache/affinity/merge status,
  Refresh ranking, explicit source refresh, ranking reason chips, source
  details, and Merge/Split controls.
- Documented the new API endpoints and closed Plan 0014 in `ROADMAP.md`.

Validation:

- `python -m py_compile transcript_api.py participant_identity.py
  provenance_config.py` passed.
- `.venv/bin/python -m pytest -q` passed with 241 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; direct and ingress health checks returned
  HTTP 200 with `status: ok`.
- Live API smoke against conversation `6e8eee4f19a1d5a9b23f` showed
  `contact-affinity/refresh` ranks `Chris Williams` first for query `chris`
  with `contacted 13 days ago`, `5 calendar overlaps`, and
  `5 interactions in 90d`; follow-up cached search returned
  `affinity_cache_status=hit`.
- `agent-browser` smoke on the Context workbench showed the selected strip
  remains visible while searching, ranking reasons render, Use causes zero
  contact-selection requests, and Save choices causes exactly one
  `/context-workbench/contact-selection-batch` request. The temporary smoke
  selection was cleared afterward.

## Turn 196 | 2026-05-24

Summary: Planned relationship-affinity ranking for contact search.

Findings:

- Broad first-name queries such as `chris` should not rank equal text matches
  randomly or only alphabetically.
- Recency and frequency of communication are useful ranking signals but should
  not be treated as proof that a contact attended or spoke in the selected
  meeting.
- Plan 0013 already makes `gws`, `msgcli`, Odollo, calendar, and iCalendar
  sources configurable through the shared user-scoped provenance profile.

Changes:

- Added a Relationship Affinity Ranking section to Plan 0014.
- Defined `relationship_affinity` fields such as last contact date, message
  counts, calendar overlap counts, transcript overlap counts, prior selected
  count, prior exclusion count, and compact ranking evidence.
- Planned user-scoped affinity caching with no raw message bodies, private
  contact exports, full unrelated attendee lists, attachment bytes, or
  credentials.
- Added planned contact-affinity refresh/read endpoints and a deterministic
  scoring model combining text relevance, conversation relevance, relationship
  affinity, source quality, and operator history.
- Updated P09 roadmap text to include recency/frequency relationship affinity
  as an explainable ranking signal.

Validation:

- Planning update was checked against repo planning and memory/context policy.
- Graphiti discovery found only general provider/calendar context; the concrete
  affinity plan was grounded in current Plan 0014 and the shared provenance
  configuration contract.

## Turn 195 | 2026-05-24

Summary: Added explicit configured-source contact search for cache misses.

Findings:

- Searching `chris` returned zero results because the workbench search was
  cache-only and `Chris` did not exist in the selected conversation's cached
  candidates, local contacts table, or existing participant identity caches.
- The cache-only behavior is still the right default for zero-lag typing, but
  the UI needed an explicit path to query configured read-only contact sources.

Changes:

- Added a user-scoped conversation contact-search cache under
  `~/.local/state/transcribe-audio/conversation-context-contact-search-cache/`.
- Extended `/api/conversations/<id>/context-workbench/contact-search` with
  `mode=refresh` to query configured read-only contact sources, cache compact
  candidates, and return redacted source profile/warning metadata.
- Included cached source-search candidates in context contact lookup and
  selection state so refreshed contacts can be selected and batch-saved.
- Added a `Search sources` button to the context workbench while preserving
  cache-only typing and instant selection.
- Moved Plan 0014 to `OPEN` and updated P09 roadmap state.

Validation:

- `python -m py_compile transcript_api.py` passed.
- Targeted tests for source-search caching and batch selection passed.
- `.venv/bin/python -m pytest -q tests/test_transcript_api.py` passed with 47
  tests.
- `npm --prefix frontend run build` passed.
- Restarted `transcripts.service`; API health returned `status: ok`.
- Live cache-only search for `chris` initially returned zero results.
- Live `mode=refresh` source search for `chris` queried the configured read-only
  `gws` and Odollo profiles, cached 18 refreshed candidates, and returned 10
  matching compact candidates without warnings.
- Follow-up cache-only search for `chris` returned the cached 10 candidates.
- Headless Chromium CDP smoke verified the rendered workbench shows `Search
  sources`, keeps the selected strip visible, and filters the grid to `Chris`
  candidates after entering `chris`.

## Turn 194 | 2026-05-24

Summary: Planned the Contact Search Workbench.

Findings:

- P09 remains open after M1/M2/Plan 0013 for richer contact and provenance
  ergonomics.
- The current contact picker has the right initial direction: local staging,
  batch save, selected-contact strip, deterministic merge keys, and
  user-scoped provenance config.
- The next work needs a bounded contract separating instant cached selection
  from explicit slow source refresh/search.

Changes:

- Added `docs/dev/plans/0014-2026-05-24-contact-search-workbench.md`.
- Wired Plan 0014 into P09 in `ROADMAP.md`.
- Defined the workbench UX contract: selected strip, cached search, candidate
  grid, source controls, manual add, natural-language instructions, merge
  review, dirty state, batch save, and final-preview flush.
- Defined the backend contract for cache-only search by default,
  `contact-selection-batch`, explicit source refresh endpoints, merge/split
  persistence, and App Intelligence structured decisions.

Validation:

- Planning artifacts were checked against repo policy, P09 roadmap state, and
  closed Plans 0010, 0012, and 0013.
- Graphiti discovery for `transcribe_audio_main` returned only general
  planning-surface facts, so the plan was grounded in current repo files and
  live runbook/roadmap state.

## Turn 193 | 2026-05-24

Summary: Removed contact-picking lag and repaired context contact search display.

Findings:

- The context workbench called the backend `contact-selection` endpoint on
  every Use/Exclude/Clear click, so picking already-fetched candidates waited
  for a full selection write/refresh path.
- The search display swapped to backend search results and could hide already
  selected contacts while a query was active.

Changes:

- Made contact Use/Exclude/Clear update local React state immediately.
- Added a persistent selected-contact strip that stays visible while searching.
- Added a Save choices action and a `contact-selection-batch` backend endpoint
  so persistence is a single explicit backend action instead of one request per
  click.
- Final preview queueing flushes any unsaved local contact choices before it
  queues the preview.
- De-duplicated UI contact display by candidate ids, dedupe keys, and email so
  stale selected duplicates do not clutter the selected strip.

Validation:

- `python -m py_compile transcript_api.py` passed.
- `.venv/bin/python -m pytest -q tests/test_transcript_api.py` passed with 46
  tests.
- `npm --prefix frontend run build` passed.
- Restarted `transcripts.service`; API health returned `status: ok`.
- Headless Chromium CDP smoke against
  `/?selected=6e8eee4f19a1d5a9b23f&conversation=1&workflow=context#raw-audio`
  verified the selected strip showed Eric Cochran, Sean Solberg, Michael, and
  Baker Kuehl; clicking a contact action produced zero
  `/context-workbench/contact-selection*` network requests; typing `solberg`
  filtered the grid to Sean Solberg while the selected strip remained visible.

## Turn 192 | 2026-05-24

Summary: Strengthened deterministic participant contact de-duplication.

Findings:

- The F&B/SABER transcript contact candidates were still too fragmented when a
  person had multiple email addresses or inverted display-name formats.
- Email-first de-duplication merged exact addresses and configured aliases, but
  kept `Baker Kuehl <baker@saberchemical.com>` separate from
  `Baker Kuehl <bwkuehl@iastate.edu>` and kept `Sean Solberg` separate from
  `Solberg, Sean <ssolberg@fredlaw.com>`.

Changes:

- Added deterministic full-name merge keys to `participant_identity.py` for
  two-token-or-stronger person names, including inverted `Last, First` labels.
- Kept weak single-token names such as `Michael` out of automatic name merges
  to avoid broad false positives.
- Reused the same merge logic in context-workbench candidate de-duplication.
- Bumped the participant identity cache algorithm key so stale bundles rebuild.

Validation:

- `python -m py_compile participant_identity.py transcript_api.py` passed.
- `.venv/bin/python -m pytest -q tests/test_participant_identity.py tests/test_transcript_api.py`
  passed with 57 tests.
- Rebuilt the identity cache for source document `6e8eee4f19a1d5a9b23f`.
- Restarted `transcripts.service` and `transcribe-watch.service`; both are
  active.
- Live API candidates for `6e8eee4f19a1d5a9b23f` now show four top-level
  proposals: Eric Cochran, Sean Solberg, Michael, and Baker Kuehl. Sean merges
  the Fredrikson email source, Baker merges the SABER and Iowa State email
  sources, and no proposed candidate label/email matches `ecochran76`, Gmail,
  or Facebook aliases.

## Turn 191 | 2026-05-24

Summary: Implemented the shared user-scoped provenance configuration system.

Findings:

- The live user-scoped provenance config now resolves the migrated SABER Zoho
  iCalendar feed, SABER/SoyLei shared calendar IDs, one `gws` contact profile,
  disabled `msgcli` placeholder settings, and two Odollo tenant profiles.
- Direct calendar repair can find the selected F&B/SABER iCalendar attendee
  evidence with only `--provenance-profile default`.
- The ignored watcher config no longer stores private iCalendar feed URLs or
  duplicated shared-calendar provenance IDs; jobs point at the shared
  `default` provenance profile instead.

Changes:

- Added `provenance_config.py` with config loading, validation, redaction,
  doctor, sample initialization, preview/apply mutation, atomic writes, audit
  writes, and adapters for calendar metadata, participant identity, and
  context-workbench source config.
- Wired `--provenance-config` and `--provenance-profile` into both
  transcription CLIs, `repair_calendar_metadata.py`, and `route_transcript.py`.
- Made direct `--use-calendar` resolve configured `gog`/`gws` providers,
  shared calendar IDs, and iCalendar feeds by default while retaining explicit
  one-off calendar flags.
- Made `participant_identity.py` prefer the shared provenance config and fall
  back to legacy `contact-provenance.config.json` only when needed.
- Added transcript API endpoints for redacted provenance config GET, doctor,
  preview, and apply.
- Enabled the React Provenance tab with source toggles, active-profile display,
  redacted iCalendar feed display, iCal add/update controls, preview/apply
  actions, and inspector diagnostics.
- Migrated the live user-scoped config to
  `~/.local/state/transcribe-audio/provenance.config.json` and moved the
  pre-migration watcher backup under user-scoped runtime backups.
- Updated README, `ROADMAP.md`, the sample watcher config, and plan 0013.

Validation:

- `.venv/bin/python -m py_compile provenance_config.py transcript_api.py participant_identity.py assembly_transcribe.py faster_whisper_transcribe.py repair_calendar_metadata.py route_transcript.py watch_transcriptions.py tests/test_provenance_config.py tests/test_transcript_api.py`
  passed.
- Focused pytest passed: 14 tests across provenance config, participant
  identity, transcript API provenance endpoints, and watcher calendar config
  expansion.
- `.venv/bin/python -m pytest -q` passed: 227 tests.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python provenance_config.py doctor` reported `status: ok`.
- `watch_transcriptions.py --check --check-json` reported `status: ok`.
- Live repair of conversation `6e8eee4f19a1d5a9b23f` with only
  `--provenance-profile default` found one iCalendar provenance match and four
  Google calendar matches, then re-ingested the stored transcript.
- Refreshed source/store artifacts and ignored watcher config do not contain
  the private Zoho URL; the artifacts retain a hashed iCalendar calendar id and
  two attendee emails.
- Live API provenance config and doctor endpoints returned redacted, valid
  payloads; the selected context workbench returned three calendar attendees,
  20 proposed contacts, and GWS/Odollo sources.
- Browser smoke for `/?view=Provenance` rendered six source controls, showed
  `SABER Zoho=[redacted]`, previewed a no-write config update, and did not leak
  the private Zoho URL.
- Restarted `transcripts.service` and `transcribe-watch.service`; both are
  active and API health returned `status: ok`.

Next:

- Dogfood the shared context-workbench source profile over more recent
  recordings and decide which provenance-source readiness controls should be
  promoted into the UI before any future external apply contracts.

## Turn 190 | 2026-05-24

Summary: Defined the user-scoped provenance configuration system.

Findings:

- Provenance settings are currently split across watcher calendar config,
  direct CLI flags, and user-scoped `contact-provenance.config.json`.
- Direct CLI `--use-calendar` should resolve the same configured shared
  calendars and private iCalendar feeds that the watcher and web service use.
- The web console needs a shared config surface it can inspect and mutate
  without writing private operator settings into the repo.

Changes:

- Added open plan
  `docs/dev/plans/0013-2026-05-24-user-scoped-provenance-config.md`.
- Added redacted schema sample `provenance.config.json.sample`.
- Wired plan 0013 into P09 in `ROADMAP.md`.
- The defined config contract lives at
  `~/.local/state/transcribe-audio/provenance.config.json` by default, with
  `TRANSCRIPTS_PROVENANCE_CONFIG` and explicit CLI/API overrides.
- The schema covers `gog`, `gws`, `msgcli`, multiple `odollo` tenant profiles,
  and private/shared `ical_calendar` feeds.
- The plan defines redaction, secret refs, local preview/apply mutation,
  workflow profiles, direct CLI integration, watcher migration, and web API
  endpoints.

Validation:

- `python -m json.tool provenance.config.json.sample` parsed the sample
  configuration successfully.

Next:

- Implement `provenance_config.py` and migrate the live watcher/contact
  settings into the new user-scoped config so direct `--use-calendar` and the
  web service resolve the same SABER/SoyLei/Zoho provenance profile.

## Turn 189 | 2026-05-24

Summary: Added private iCalendar feed support to calendar provenance.

Findings:

- SABER's authoritative attendee feed is available through a private Zoho
  iCalendar export rather than only the imported Google calendar.
- The selected F&B/SABER transcript already had Google calendar overlap
  provenance, but the imported SABER Google calendar did not expose attendee
  emails.

Changes:

- Added repeated `--calendar-provenance-ical-url` support to both
  transcription CLIs and `repair_calendar_metadata.py`.
- Added watcher `calendar.provenance_ical_urls` support. Entries may be plain
  URL strings or redacted-label objects such as
  `{"label": "SABER Zoho", "url": "https://..."}`.
- Implemented standard-library iCalendar parsing for VEVENT summaries,
  DTSTART/DTEND, attendees, organizer, EXDATE, and common daily/weekly/monthly
  RRULE expansion.
- iCalendar provenance writes only a stable hashed `ical:<hash>` calendar id to
  artifacts; private feed URLs stay in ignored runtime config.
- Updated the ignored live `watch_transcriptions.json` so both active watcher
  jobs include the SABER Zoho feed.
- Refreshed and re-ingested selected conversation `6e8eee4f19a1d5a9b23f`.

Validation:

- Watcher config readiness passed with status `ok`.
- Live Zoho provenance smoke found one matching F&B/SABER event for the
  selected recording and two attendee emails without printing or storing the
  private feed URL in transcript artifacts.
- Refreshed source and store artifacts do not contain the private feed URL.
- The conversation API now shows a hashed `SABER Zoho` iCalendar provenance
  event and the context workbench identity bundle includes three calendar
  attendees, including the two SABER attendees from Zoho.
- `.venv/bin/python -m py_compile transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py repair_calendar_metadata.py watch_transcriptions.py tests/test_transcript_artifacts.py`
  passed.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_participant_identity.py -q`
  passed: 33 tests.
- `.venv/bin/python -m pytest -q` passed: 219 tests.
- Restarted `transcribe-watch.service`; it is active.
- `curl http://127.0.0.1:18876/api/health` returned `status: ok`.

Next:

- Dogfood the refreshed F&B/SABER context workbench and decide whether the two
  Zoho attendee identities should be preselected for speaker assignment or left
  as reviewed contact candidates.

## Turn 188 | 2026-05-23

Summary: Added shared-calendar IDs to calendar provenance.

Findings:

- The active calendar provider sees the SABER imported calendar
  `48gt5h6avb4222kf8r2a8mh1tvp5gqq4@import.calendar.google.com` as
  `Eric - SABER`.
- The active calendar provider sees the SoyLei shared calendar
  `eric.cochran@soylei.com` as `Eric - SoyLei`.
- Existing `event.matching_calendars` entries recorded overlap metadata but
  did not carry attendee emails from shared-calendar events into downstream
  identity evidence.

Changes:

- Added repeated `--calendar-provenance-calendar-id` support to both
  transcription CLIs and `repair_calendar_metadata.py`.
- Added watcher `calendar.provenance_calendar_ids` support, with
  `shared_calendar_ids` accepted as an alias.
- Updated the live ignored `watch_transcriptions.json` to include the SABER
  and SoyLei shared calendar IDs for both active recording jobs.
- Matching-calendar provenance now carries `attendees` and `attendee_emails`
  when provider event payloads include them.
- Updated README and the sample watcher config.

Validation:

- `gog calendar calendars --json --results-only --no-input` found
  `Eric - SABER` and `Eric - SoyLei` in the active provider calendar list.
- Focused tests passed:
  `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_participant_identity.py -q`
  and `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q`.
- `.venv/bin/python -m py_compile transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py repair_calendar_metadata.py watch_transcriptions.py tests/test_transcript_artifacts.py tests/test_participant_identity.py`
  passed.
- `.venv/bin/python -m pytest -q` passed: 217 tests.
- Refreshed and re-ingested selected conversation
  `6e8eee4f19a1d5a9b23f` with explicit SABER/SoyLei provenance calendar IDs.
- Live SoyLei overlap smoke returned attendee names/emails under
  `event.matching_calendars` when the shared calendar provided them.
- Restarted `transcribe-watch.service`; readiness check passed and the service
  is active.
- `curl http://127.0.0.1:18876/api/health` returned `status: ok`.

Next:

- Tighten Odollo identity candidate generation so broad event-title terms do
  not appear as speaker/contact proposals without direct attendee evidence.

## Turn 187 | 2026-05-23

Summary: Added context-workbench contact visibility and selection controls.

Findings:

- The context workbench carried the participant identity bundle but only showed
  counts and source-profile chips.
- Proposed contacts were visible only from the Speakers tab, and there was no
  context-specific API control for an operator or App Intelligence worker to
  select or exclude contacts before context/readout preparation.

Changes:

- `GET /api/conversations/<id>/context-workbench` now includes
  `proposed_contact_candidates` plus a `contact_selection` state.
- Added `POST /api/conversations/<id>/context-workbench/contact-selection` for
  local `select`, `exclude`, and `clear` decisions with
  `actor_type=operator|app_intelligence`.
- Context contact decisions are stored under user-scoped runtime state in
  `~/.local/state/transcribe-audio/conversation-context-contact-selections/`.
- Context workbench preview/queue manifests now include the contact-selection
  state alongside the participant identity bundle.
- The React Context workbench tab now renders proposed contacts with source,
  profile, confidence, selected chips, and `Use`/`Exclude` controls.
- Updated README and API docs for the new local selection endpoint.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_participant_identity.py -q`
  passed: 47 tests.
- `.venv/bin/python -m py_compile transcript_api.py participant_identity.py tests/test_transcript_api.py tests/test_participant_identity.py`
  passed.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest -q` passed: 215 tests.
- Restarted `transcripts.service`; `curl http://127.0.0.1:18876/api/health`
  returned `status: ok`.
- Live API smoke selected an Odollo contact candidate for conversation
  `6e8eee4f19a1d5a9b23f` with `actor_type=app_intelligence`; the response
  recorded local contact-selection state and reported no external write.
- Browser smoke on the selected local context workbench found
  `11 proposed`, Odollo sources, selected contact text, `Use` and `Exclude`
  buttons, and no `401` or `Unauthorized` text.

Next:

- Queue a context preview/manifest from the selected conversation and verify
  the contact-selection state carries into the App Intelligence handoff bundle.

## Turn 186 | 2026-05-23

Summary: Repaired Odollo contact provenance in the live transcript console.

Findings:

- The selected conversation
  `6e8eee4f19a1d5a9b23f` surfaced Odollo `401 Unauthorized` warnings in the
  participant identity bundle.
- Interactive Odollo auth succeeded for `soylei-prod` and `saber-prod`, but
  `transcripts.service` did not inherit `ODOO_SOYLEI_API_KEY` or
  `ODOO_SABER_API_KEY`.
- After the env repair, Odollo returned contacts, but the identity bundle still
  displayed only Google candidates because anonymous speaker labels `A/B/C/D`
  were used as broad contact-system search terms and the 20-candidate cap was
  filled by noisy Google results.

Changes:

- Wrote a user-scoped secret env file at
  `~/.local/state/transcribe-audio/odollo.env` with redacted Odoo API-key
  variables and mode `0600`.
- Added `~/.config/systemd/user/transcripts.service.d/20-odollo-env.conf` with
  `EnvironmentFile=/home/ecochran76/.local/state/transcribe-audio/odollo.env`.
- Restarted `transcripts.service`.
- Updated identity query building to ignore anonymous speaker labels such as
  `A`, `B`, `Speaker C`, and `SPEAKER_00`.
- Updated Odollo term building to include readout participant names/emails.
- Updated candidate ranking to preserve representation from configured source
  profiles so Odollo candidates cannot be completely crowded out by broad
  Google results.
- Documented the systemd Odollo environment requirement in README.

Validation:

- `odollo.cli ... odoo auth test` passed for `soylei-prod` and `saber-prod`
  from the interactive shell.
- `systemctl --user show transcripts.service` reports the Odollo environment
  file.
- `curl http://127.0.0.1:18876/api/health` returned `status: ok`.
- The selected conversation now has query terms limited to
  `ecochran76@gmail.com`, no Odollo `401` warnings, and 11 compact contact
  candidates: 1 `gws_contact`, 2 `gws_other_contact`, and 8 `odollo_contact`.
- Browser check on the selected conversation found
  `odollo_contact · soylei-prod · 0.4` in the speaker candidate UI and no
  `401`/`Unauthorized` text.
- `.venv/bin/python -m pytest tests/test_participant_identity.py tests/test_context_sources.py tests/test_transcript_api.py -q`
  passed: 61 tests.
- `.venv/bin/python -m py_compile participant_identity.py context_sources.py transcript_api.py tests/test_participant_identity.py tests/test_context_sources.py tests/test_transcript_api.py`
  passed.

Next:

- Review whether low-confidence self-email-derived Odollo candidates should be
  visually separated from direct attendee/name matches before scaling identity
  review to more raw transcripts.

## Turn 185 | 2026-05-23

Summary: Completed M2 speaker deanonymization and participant-aware context
workbench.

Changes:

- Added `participant_identity.py` with
  `transcribe-audio.participant-identity-bundle.v1`.
- Added configured read-only `gws` People/Contacts provenance for grouped
  contacts, Other Contacts, and optional directory people.
- Promoted Odollo `res.partner` contacts into the identity candidate pool while
  keeping log-note provenance out of speaker identity matching.
- Added `contact-provenance.config.json.sample` and ignored real
  `contact-provenance.config.json` files.
- Wrote the live user-scoped config at
  `~/.local/state/transcribe-audio/contact-provenance.config.json` for the
  default `gws` profile plus `soylei-prod` and `saber-prod` Odollo profiles.
- Extended conversation identity-review, context-workbench, first-pass summary,
  contextual reread, AuraCall batch, and readout prompt paths to carry the
  participant identity bundle.
- Updated the React conversation workspace with calendar evidence, source
  profile chips, provenance candidate details, manual contact entry controls,
  and blocked final-preview messaging when identity/context warnings remain.
- Updated the conversation review-loop browser smoke to wait for slow live
  contact-provenance detail loading and validate the M2 identity path.
- Updated README, API docs, Plan 0012, and ROADMAP.

Validation:

- Count-only live contact-provenance check loaded 3 source profiles, found 6
  compact contact-provenance sources for the Tempo query, and returned 0
  warnings without printing contact records.
- `.venv/bin/python -m pytest -q` passed: 213 tests.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m py_compile participant_identity.py context_sources.py transcript_api.py summarize_transcript.py contextual_reread.py scripts/auracall_legacy_enrichment_batch.py scripts/smoke_conversation_review_loop_ui.py tests/test_participant_identity.py tests/test_context_sources.py tests/test_transcript_api.py tests/test_readouts.py` passed.
- Restarted `transcripts.service`; `systemctl --user is-active
  transcripts.service` returned `active` and
  `curl http://127.0.0.1:18876/api/health` returned `status: ok`.
- Browser smoke passed:
  `~/.local/state/transcribe-audio/browser-smokes/20260523T181351Z-conversation-review-loop-smoke.json`.
  The live Tempo conversation showed 4 speakers, 14 contact candidates, 6
  calendar attendees, 3 source profiles, 3 pending assignments, manual contact
  controls, context identity chips, and final-preview blocking with 5 unresolved
  identity/context warnings.

Next:

- Dogfood identity decisions on more recordings and tune contact-source
  quality before reopening P05 external deposition apply work.

## Turn 184 | 2026-05-23

Summary: Corrected M2 contact matching to use configured provenance sources.

Changes:

- Updated Plan 0012 so contact matching comes from user-scoped provenance,
  not from calendar attendees alone.
- Named `gws people` as the Google Contacts/Other Contacts/Directory surface
  and Odollo tenant contacts as the Odoo contact provenance source.
- Clarified that calendar attendees and `event.matching_calendars`
  participants are deterministic matching evidence against contact provenance.
- Updated P09/ROADMAP text so the participant identity bundle records source
  profile, evidence, confidence, unresolved ambiguity, and operator decisions.

Validation:

- Re-read planning, architecture/productization, and memory/context-routing
  policies.
- Ran Graphiti discovery against `transcribe_audio_main`; results were broad
  prior roadmap facts, so repo files and this operator correction remain
  authoritative.
- Checked local `gws` help; Google contact access is under the `people` service
  with grouped contacts, Other Contacts, and directory people surfaces.

Next:

- Implement Plan 0012 slice 1 as a provenance-source configuration and
  normalization pass: `gws` People/Contacts adapter, Odollo contact candidate
  promotion, and identity-bundle API exposure.

## Turn 183 | 2026-05-23

Summary: Reprioritized the next roadmap slice around speaker deanonymization
and context-workbench inputs.

Changes:

- Added `docs/dev/plans/0012-2026-05-23-speaker-deanonymization-context-workbench.md`
  as the next P09 milestone.
- Deferred P05 deposition apply work until participant identity and
  context-workbench evidence are stronger.
- Updated the P09 plan so the workflow order is first-pass summary,
  speaker/contact identity, participant-aware context workbench, final readout,
  and only then deposition/memory preview.
- Recorded calendar invite attendees and `event.matching_calendars`
  participants as the first deterministic speaker/contact candidate source.
- Recorded that high-powered readout providers such as AuraCall/Extended Pro
  ChatGPT should receive an explicit participant/context bundle rather than
  infer identities from anonymous speaker labels alone.

Validation:

- Re-read planning, architecture/productization, and memory/context-routing
  policies.
- Ran Graphiti discovery against `transcribe_audio_main`; results were broad
  prior roadmap facts, so current repo files and the operator correction remain
  authoritative.

Next:

- Implement Plan 0012 slice 1: participant identity bundle schema,
  deterministic calendar-attendee extraction, and conversation API exposure.

## Turn 182 | 2026-05-23

Summary: Completed P06 service reliability and observability.

Changes:

- Added watcher readiness checks for `ffprobe`, configured watch directories,
  backend scripts, and readout script availability.
- Added `watch_transcriptions.py --check` and `--check --check-json` for
  service doctor/readiness checks without scanning or transcribing files.
- Extended watcher candidate state with `blocked_kind`, `blocked_reason`, and
  `blocked_since`.
- Extended failed processed records with `failure_kind` and `failure_reason` so
  retry backoff explains the prior failure.
- Heartbeats now include `blocked=kind=count` summaries.
- Updated README systemd guidance with `ExecStartPre=... --check`, service
  health commands, and blocked-reason interpretation.
- Installed live user-service readiness drop-in at
  `~/.config/systemd/user/transcribe-watch.service.d/10-readiness-check.conf`.
- Closed `docs/dev/plans/0006-2026-05-04-service-reliability-observability.md`
  and updated P06 in `ROADMAP.md`.

Validation:

- `python -m py_compile watch_transcriptions.py tests/test_transcript_artifacts.py`
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q`
- `.venv/bin/python watch_transcriptions.py --check --check-json`
- `systemctl --user daemon-reload && systemctl --user restart transcribe-watch.service && systemctl --user is-active transcribe-watch.service`
- `systemctl --user cat transcribe-watch.service --no-pager`
- `journalctl --user -u transcribe-watch.service -n 80 --no-pager` showed
  readiness preflight, the loaded jobs, and a heartbeat with `blocked=none`.

Next:

- Keep future service/UI status aggregation under P09 Settings or a new bounded
  plan; core P06 readiness and blocked-state requirements are satisfied.

## Turn 181 | 2026-05-23

Summary: Completed Plan 0011 P04 provenance calibration.

Changes:

- Added `p04-source-quality-v1` as the explicit source-quality profile for
  route provenance filtering.
- Route decisions now write `provenance_pack.quality_profile`; contextual
  rereads propagate the same profile into supporting-context and
  contextualization metadata.
- Added `scripts/evaluate_provenance_calibration.py` for reviewed manifest
  evaluation using the existing `context_sources.py` scorer.
- Added repo-safe calibration docs and fixtures under
  `docs/dev/fixtures/p04-calibration/`, including a manifest schema and
  synthetic 12-decision smoke corpus.
- Created the private accepted reviewed corpus at
  `~/.local/state/transcribe-audio/p04-calibration/manifests/2026-05-23-p04-source-quality-v1-reviewed.json`.
- Wrote the sanitized accepted report to
  `~/.local/state/transcribe-audio/p04-calibration/reports/2026-05-23-p04-source-quality-v1-reviewed-report.json`.
- Generated route/contextual dry-run evidence under
  `~/.local/state/transcribe-audio/p04-calibration/dry-runs/2026-05-23-source-quality-v1/`.
- Closed `docs/dev/plans/0011-2026-05-23-p04-provenance-calibration.md`.

Calibration Evidence:

- Reviewed corpus: 12 source decisions, 1 reviewed case, 4 source families.
- Result: 12 pass, 0 false positives, 0 false negatives.
- Family coverage: Calendar 3, Drive/Docs 3, Graphiti 3, Odollo 3.
- Dry-runs: known-good case retained 5 included sources and 0 excluded sources;
  weak-source case retained 2 calendar sources, excluded 3 weak non-calendar
  sources, and carried profile `p04-source-quality-v1` into route and
  contextual dry-run metadata.

Validation:

- `python -m py_compile context_sources.py route_transcript.py contextual_reread.py scripts/evaluate_provenance_calibration.py transcript_api.py`
- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_readouts.py tests/test_provenance_calibration.py tests/test_transcript_api.py::test_conversation_detail_includes_identity_and_context_state -q`
- `.venv/bin/python scripts/evaluate_provenance_calibration.py docs/dev/fixtures/p04-calibration/synthetic-manifest.json --fail-on-mismatch --require-decision-count 12 --require-source-families 4 --include-passed`
- `.venv/bin/python scripts/evaluate_provenance_calibration.py --manifest-dir ~/.local/state/transcribe-audio/p04-calibration/manifests --output ~/.local/state/transcribe-audio/p04-calibration/reports/2026-05-23-p04-source-quality-v1-reviewed-report.json --fail-on-mismatch --require-decision-count 12 --require-source-families 4 --include-passed`
- `.venv/bin/python -m pytest -q`
- `git diff --check`

Next:

- Keep P04 open for deeper Drive/Docs content fetch and target/depositor
  contracts. The local transcript-index source adapter remains deferred until a
  reviewed corpus shows it improves route decisions without leaking transcript
  content.

## Turn 180 | 2026-05-23

Summary: Re-planned P04 provenance calibration as a bounded implementation
slice.

Changes:

- Added `docs/dev/plans/0011-2026-05-23-p04-provenance-calibration.md`
  for source-quality calibration, reviewed corpus handling, threshold evidence,
  profile metadata, and validation requirements.
- Kept Plan 0004 as the parent matter-routing/contextual-reread plan and
  moved the local-index adapter and threshold-calibration work into the new
  bounded Plan 0011 slice.
- Updated `ROADMAP.md` so P04 points at Plan 0011 as the active calibration
  work before deeper Drive/Docs content fetch expands.

Validation:

- Re-read repo planning policy and architecture/memory policies for P04 scope.
- Ran Graphiti discovery for group `transcribe_audio_main`; returned only broad
  P04 planning facts, so repo-local roadmap/plan files remain authoritative.
- `git diff --check`

Next:

- Implement Plan 0011 slice 1: calibration manifest schema, repo-safe example
  README, and local-state directory contract for reviewed live manifests.

## Turn 179 | 2026-05-23

Summary: Closed Plan 0010 as the M1 dogfoodable conversation review loop.

Changes:

- Added selected-conversation first-pass summary actions under
  `/api/conversations/<id>/first-pass-summary/prepare`, `/submit`, and
  `/status`; manifests are source-transcript scoped, provider submit still
  requires `SUBMIT_FIRST_PASS_SUMMARY_BATCH`, and status can materialize
  completed readouts.
- Added durable `contacts`, `speaker_assignments`, and
  `speaker_assignment_audits` tables plus conversation speaker confirm/defer
  actions. Deferred speakers write local Review Queue items.
- Extended `/api/conversations/<id>` with first-pass summary state,
  speaker/contact review state, context-workbench provenance, final
  deposition/memory preview state, and review counters.
- Wired the React conversation workspace to selected first-pass summary actions,
  speaker/contact review, context workbench preview/queue, final preview queue,
  and Review Queue links back to conversation workflow tabs.
- Added `scripts/smoke_conversation_review_loop_ui.py` for an `agent-browser`
  M1 happy-path smoke.
- Updated README, API docs, ROADMAP, P09, and closed Plan 0010.

Dogfood Evidence:

- Historical representative: contextual conversation `fb3b9d11aea3ecb56e3d`
  now has blob-backed audio, transcript turns, first-pass summary, four speaker
  labels, context status `contextual_readout_ready`, seven included and 35
  excluded provenance sources, one deposition-preview action, and six memory
  candidates.
- Recent watcher-ingested pass: AssemblyAI transcript `2596e459aeb3812de321`
  from May 22, 2026 is non-legacy watcher output with stored media blob
  `cc16442b324c767ff111`. It verifies the source-audio/transcript workspace and
  selected first-pass summary preparation path with one dry-run request at
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-selected-2596e459aeb3812de321-20260523-104736.json`;
  it does not yet have a contextual readout.
- Local review records created during dogfooding included one speaker review
  item and one deposition/memory preview item, both linked back to conversation
  workflow stages.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py scripts/smoke_conversation_review_loop_ui.py tests/test_transcript_api.py`
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_selected_first_pass_summary_prepare_is_conversation_scoped tests/test_transcript_api.py::test_conversation_detail_includes_identity_and_context_state -q`
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q`
- `npm --prefix frontend run build`
- `git diff --check`
- `systemctl --user restart transcripts.service && systemctl --user is-active transcripts.service transcribe-watch.service`
- `python scripts/smoke_conversation_review_loop_ui.py --base-url http://127.0.0.1:18876 --session transcript-conversation-review-loop-smoke-final3 --profile /tmp/transcript-conversation-review-loop-smoke-final3-profile --viewport 1360x860`
- Browser smoke evidence:
  `~/.local/state/transcribe-audio/browser-smokes/20260523T154741Z-conversation-review-loop-smoke.json`
  and `.png`, status `pass`, no missing checks.

Next:

- Keep P09 open for broader console productization: auth/share links, richer
  contact merge workflows, provenance/deposition aggregate sections, and future
  external apply contracts.

## Turn 91 | 2026-05-23

Summary: Reframed P09 around the M1 dogfoodable conversation review loop.

Changes:

- Added `docs/dev/plans/0010-2026-05-23-dogfoodable-conversation-review-loop.md`
  as the bounded milestone plan for an end-to-end operator workflow.
- Updated `ROADMAP.md` to list the new plan under P09 and make M1 the active
  milestone focus.
- Defined the milestone as one selected conversation moving through source
  audio, transcript review, first-pass summary, speaker/contact review, context
  gathering, contextual readout, and deposition/memory preview review.
- Kept unattended external writes, broad provider expansion, and unrelated App
  Intelligence branch/fork/rollback polish out of scope for this milestone.

Validation:

- Re-read planning, architecture/productization, and memory/context-routing
  policies before editing.
- Ran Graphiti discovery against `transcribe_audio_main`; results were
  advisory and verified against current repo planning files.

Next:

- Start the M1 conversation contract audit by comparing
  `/api/conversations/<id>` with the workflow fields required by the new plan.

## Turn 1 | 2026-05-04

Summary: Adopted repo-local policy and planning surfaces for the expanded transcription platform scope.

Changes:

- Added `docs/dev/policies/` with operations-platform-oriented local policy.
- Added `ROADMAP.md` with six lanes.
- Added bounded plan files under `docs/dev/plans/`.
- Reclassified `docs/platform-expansion-plan.md` as background architecture notes rather than planning authority.
- Wired plans: `0001-2026-05-04-normalize-transcript-artifacts.md`, `0002-2026-05-04-calendar-provider-config.md`, `0003-2026-05-04-intelligence-readouts.md`, `0004-2026-05-04-matter-routing-contextual-reread.md`, `0005-2026-05-04-deposition-memory-harvest.md`, `0006-2026-05-04-service-reliability-observability.md`.
- Added Graphiti repo memory guidance for group `transcribe_audio_main`.

Policy Decision:

- Deterministic selector first pass recommended `standalone-library` from current repo shape.
- Maintainer scope requires operations-platform policy because the target system includes tenant-aware calendar access, runtime state, intelligence providers, matter routing, deposition, Graphiti/OpenClaw memory, and service reliability.

Validation:

- Ran `select_policy.py` against the local policy library.
- Ran `audit_planning_contract.py`; initial audit reported missing `ROADMAP.md`, `RUNBOOK.md`, and `docs/dev/plans/`.
- Re-ran `audit_planning_contract.py` after adoption; final result passed with `ok: true`.

Next:

- Start P01 by implementing transcript artifact sidecars.

## Turn 2 | 2026-05-04

Summary: Bootstrapped Graphiti repo memory for `transcribe-audio`.

Changes:

- Added concrete Graphiti discovery guidance to `AGENTS.md`.
- Added repo memory group guidance to `docs/dev/policies/0005-memory-and-context-routing.md`.
- Seeded Graphiti group `transcribe_audio_main` from curated repo authorities only.

Seeded Episodes:

- `9c987621-2252-45b4-a9ea-c98f9d6aff17`: `transcribe-audio: bootstrap: policy roadmap runbook`.
- `dbaa9e6b-1e6f-4929-91f7-9d5547a2923c`: `transcribe-audio: roadmap lanes and next implementation slices`.
- `d9a28757-ca32-4382-b8fb-fd18e3d689f0`: `transcribe-audio: notable event: systemd ffprobe path stall`.
- `6b698756-bcdb-4111-a873-5200ab6e7940`: `transcribe-audio: memory and context routing contract`.

Validation:

- `graphiti-runtime status` reported healthy runtime, HTTP, and FalkorDB.
- `graphiti-runtime discover --group-id transcribe_audio_main ...` returned 4 episodes and 10 facts.
- `graphiti-runtime queue` showed `transcribe_audio_main` queue size 0 after writes completed.

Next:

- Use `graphiti-discovery` against `transcribe_audio_main` before future non-trivial planning, debugging, architecture, routing, memory, or handoff work.

## Turn 3 | 2026-05-04

Summary: Started P01 by adding transcript artifact sidecars.

Changes:

- Added `transcript_artifacts.py` with `TranscriptArtifact` JSON serialization.
- Updated shared transcript output processing to write `*.transcript.json` sidecars.
- Sidecars include transcript text and selected structured utterances for downstream automation.
- Backend CLIs now emit `TRANSCRIPT_ARTIFACT_JSON=<path>` lines through the shared output path.
- Watcher state now preserves `artifact_paths` for successful processed records.
- Added focused tests under `tests/test_transcript_artifacts.py`.
- Documented the sidecar output contract in `README.md`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed with 3 tests.
- `python -m py_compile transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py` passed.
- Graphiti event `ca8c403c-73b2-47ca-ae40-3b777d81c605` recorded the P01 artifact sidecar implementation and smoke search returned 5 facts.

Next:

- Run a manual short-recording smoke with `--text-output --use-calendar` when a suitable non-sensitive clip is available.
- Continue P01 by running manual smoke and deciding whether P01 can close or needs richer artifact schema fields.

## Turn 4 | 2026-05-04

Summary: Started P02 by making calendar provider selection explicit and tenant-aware.

Changes:

- Added `CalendarProviderConfig` and explicit provider ordering.
- Changed default calendar lookup order to `gog`, then `gws`, then built-in `google-api` fallback.
- Made Google API fallback lazy so OAuth is not triggered unless that provider is reached.
- Added CLI flags for `--calendar-providers`, `--calendar-gog-account`, `--calendar-gog-client`, and `--calendar-gws-config-dir`.
- Added watcher `calendar` config expansion into backend CLI args.
- Documented the provider config in `README.md` and `watch_transcriptions.json.sample`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q` passed with 7 tests.
- `python -m py_compile transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py` passed.
- Graphiti event `350f35aa-1570-42e1-b84a-3abde59256ed` recorded the P02 calendar provider implementation.

Next:

- Run a manual watcher `--run-once` calendar lookup test against a non-sensitive short clip.
- Continue P03 after P01/P02 manual smoke gates are handled or intentionally deferred.

## Turn 5 | 2026-05-04

Summary: Closed P01 and P02 with a temp-location watcher smoke.

Smoke Setup:

- Copied `/home/ecochran76/Downloads/You Have Been Banned.mp3` to `/tmp/transcribe-watch-smoke.TFGX7X/watch/`.
- Created temp watcher config at `/tmp/transcribe-watch-smoke.TFGX7X/watch_config.json`.
- Used backend `faster_whisper` with `tiny.en`, CPU `int8`, `--text-output`, and `--no-speaker-labels`.
- Enabled structured watcher calendar config with providers `gog,gws` to avoid touching Google OAuth fallback during smoke.
- Used temp state file `/tmp/transcribe-watch-smoke.TFGX7X/state.json`.

Validation:

- First `watch_transcriptions.py --run-once` pass tracked the candidate.
- Second `watch_transcriptions.py --run-once` pass processed the stable file successfully.
- Calendar lookup tried `gog` and returned 8 events.
- The temp media was renamed from matched event metadata.
- Outputs were written under `/tmp/transcribe-watch-smoke.TFGX7X/out/`.
- Watcher state recorded the artifact path and backend success.
- Transcript text: `Speaker [0.00s - 1.74s]: You have been banned.`

Evidence:

- Artifact path: `/tmp/transcribe-watch-smoke.TFGX7X/out/2026-05-04 19-30 choir concert You Have Been Banned Transcript.transcript.json`.
- State command recorded `--use-calendar --calendar-providers gog,gws --calendar-id primary --calendar-window 24`.
- P01 and P02 are now closed in `ROADMAP.md` and their plan files.

Notes:

- Because the file was copied into `/tmp` with a fresh mtime, calendar matching used the temp copy timestamp and selected a nearby event. This was sufficient for provider/watcher validation, not a semantic event-match quality test.

Next:

- Start P03 by defining readout schemas and the first OpenAI-compatible intelligence provider seam.

## Turn 6 | 2026-05-04

Summary: Started P03 by implementing structured intelligence readouts.

Changes:

- Added `readout_artifacts.py` with readout JSON schema and Markdown rendering.
- Added `summarize_transcript.py` to read transcript sidecars and call an OpenAI-compatible chat completions API.
- Added provider seams for `openai-compatible`, `codex-exec`, `auracall`, and `openclaw`; only `openai-compatible` is implemented in this slice.
- Added watcher `readout` config and post-processing after successful transcription.
- Watcher state now preserves `readout_paths` when readout generation succeeds.
- Readout failures are logged as warnings and do not mark transcription as failed.
- Documented readout usage in `README.md`, `api_keys.json.sample`, and `watch_transcriptions.json.sample`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 12 tests.
- `python -m py_compile readout_artifacts.py summarize_transcript.py transcript_artifacts.py transcribe_common.py assembly_transcribe.py faster_whisper_transcribe.py watch_transcriptions.py tests/test_transcript_artifacts.py tests/test_readouts.py` passed.
- Manual local OpenAI-compatible smoke generated readout JSON and Markdown from the temp transcript artifact at `/tmp/transcribe-watch-smoke.TFGX7X/out/2026-05-04 19-30 choir concert You Have Been Banned Transcript.transcript.json`.
- Graphiti event `a7dd576c-a25d-426b-8754-f24c8178030e` recorded the P03 readout implementation.

Evidence:

- Readout JSON: `/tmp/transcribe-watch-smoke.TFGX7X/readouts/2026-05-04 19-30 choir concert You Have Been Banned Transcript.readout.json`.
- Readout Markdown: `/tmp/transcribe-watch-smoke.TFGX7X/readouts/2026-05-04 19-30 choir concert You Have Been Banned Transcript.readout.md`.
- Smoke summary: `The short clip says the listener has been banned.`

Next:

- Smoke P03 against a real configured provider/key, then close P03 or move to P04 routing schemas if local-compatible validation is accepted.

## Turn 7 | 2026-05-08

Summary: Repaired recent eventless transcripts and fixed service calendar provider PATH.

Cause:

- The watcher config requested calendar lookup, but the long-running service process used older provider behavior and then the systemd user PATH did not include `~/.local/bin` or `~/.cargo/bin`.
- As a result, child transcription processes could not find `gog` or `gws` and fell through to the built-in Google API provider, which failed because OAuth client secrets were not configured.

Changes:

- Updated `watch_transcriptions.json` to use structured `calendar` config with provider order `gog,gws,google-api`.
- Updated `~/.config/systemd/user/transcribe-watch.service` and `.openclaw/transcribe-watch.service` PATH to include `%h/.local/bin` and `%h/.cargo/bin`.
- Added `repair_calendar_metadata.py` for dry-run/apply backfills using each artifact's recorded recording window.
- Restarted `transcribe-watch.service` after `systemctl --user daemon-reload`.

Backfill:

- Repaired 9 recent eventless transcript artifacts: recordings 115 through 123.
- Regenerated TXT/DOCX transcript outputs with event details and participants.
- Renamed transcript artifacts and media files to calendar-based names.
- Updated `.openclaw/watch_transcriptions_state.json` so renamed recordings are not reprocessed.

Validation:

- `gog status` showed an authenticated `ecochran76@gmail.com` account.
- Repair dry run matched 8 artifacts before apply; follow-up repair matched recording 123 created during the service restart window.
- Recent sidecar scan reported `missing_recent_event_count=0`.
- Sample repaired transcript `2026-05-08 14-00 Timothy Clark My recording 122 Transcript.txt` includes event details and participants.
- `python -m py_compile repair_calendar_metadata.py watch_transcriptions.py assembly_transcribe.py faster_whisper_transcribe.py transcribe_common.py` passed.
- Planning audit passed with `ok: true`.

Notes:

- `My recording 123 (1).m4a` remains a live watcher candidate but is currently incomplete/corrupt (`moov atom not found`) and still changing. The watcher is correctly waiting rather than processing it.

Next:

- After `My recording 123 (1).m4a` finishes syncing, confirm the next successful service transcript logs `Calendar lookup: trying provider gog...`.

## Turn 8 | 2026-05-08

Summary: Confirmed the live service now uses `gog` calendar mode.

Validation:

- `My recording 123 (1).m4a` finished syncing and was processed by `transcribe-watch.service`.
- The journal showed `Calendar lookup: trying provider gog...` followed by `Calendar lookup: provider gog returned 7 event(s).`
- The service matched event `SIP WMA Hamburg` and renamed media/output files to `2026-05-08 15-00 SIP WMA Hamburg My recording 123 (1) ...`.
- The repaired transcript text includes event details, location, and participants.
- Recent sidecar scan still reports `missing_recent_event_count=0`.

Next:

- Leave the service running and monitor the next normal mobile recording only if another calendar miss appears.

## Turn 9 | 2026-05-08

Summary: Added accessible-calendar overlap context to calendar event metadata.

Changes:

- Extended calendar lookup to list accessible calendars for the active provider and query each calendar for overlapping events.
- Added `event.matching_calendars` to transcript sidecar metadata.
- Kept primary event selection and filename behavior stable; matching-calendar context is for downstream readout/routing.
- Updated `repair_calendar_metadata.py` so existing event sidecars can be refreshed with `matching_calendars` without renaming files again.
- Refreshed recent repaired transcripts to include `matching_calendars`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 17 tests.
- `python -m py_compile transcribe_common.py repair_calendar_metadata.py tests/test_transcript_artifacts.py` passed.
- Recent event sidecar scan reported `recent_event_sidecars_without_matching_calendars=0`.
- Sample `SIP WMA Hamburg` sidecar has `matching_calendars` entries for overlapping accessible calendars.
- `transcribe-watch.service` was restarted and is active.

Next:

- Use `event.matching_calendars` in the P03/P04 readout and routing prompts so overlapping calendar context helps identify the meeting matter.

## Turn 10 | 2026-05-08

Summary: Fed matching-calendar overlap context into intelligence readout prompts.

Changes:

- Added a dedicated `calendar_context` prompt block in `summarize_transcript.py`.
- Included the primary event summary, primary event participants, and `event.matching_calendars` in readout requests.
- Updated system guidance so calendar names and overlapping event summaries can inform meeting type, participant context, matter candidates, and memory candidates as evidence rather than proof.
- Documented the prompt contract in `README.md`, `ROADMAP.md`, and the P03 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_readouts.py -q` passed with 7 tests after adding prompt coverage.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.
- A local OpenAI-compatible smoke generated readout JSON/Markdown from the repaired `SIP WMA Hamburg` artifact and confirmed `matching_calendars` reached matter-candidate evidence.
- `transcribe-watch.service` remained active.

Next:

- Decide whether to close P03 after a live-provider readout smoke or move directly into P04 routing with the current local-compatible validation.

## Turn 11 | 2026-05-08

Summary: Verified recent transcripts have calendar-context metadata.

Action:

- Ran `repair_calendar_metadata.py` against `/mnt/c/Users/ecoch/Downloads/*.transcript.json` for the last 8 days with `gog,gws`.
- Applied the repair command with `--no-rename-media`; it was a no-op because all recent event sidecars already had `event.matching_calendars`.

Validation:

- Recent scan found 10 transcript artifacts, 10 with event metadata, and 0 missing `matching_calendars`.
- Calendar-context counts ranged from 1 to 5 matching calendar entries per recent transcript.
- `transcribe-watch.service` remained active.

Next:

- Run a real-provider readout smoke on one of these updated artifacts, then decide whether to close P03 or move directly into P04 routing.

## Turn 12 | 2026-05-10

Summary: Validated AuraCall as an OpenAI-compatible chat provider.

Action:

- Loaded API parameters from `/home/ecochran76/.auracall/api.env` without printing secrets.
- Sent a minimal `/v1/chat/completions` request to the AuraCall local endpoint using model `agent:instant-chatgpt-ecochran76`.

Validation:

- AuraCall returned HTTP 200 in 7.239 seconds.
- The response content matched the requested smoke phrase: `auracall smoke ok`.
- The response included a completion ID; usage accounting was not returned.

Next:

- Run `summarize_transcript.py` against one updated transcript artifact using the AuraCall OpenAI-compatible endpoint and confirm readout JSON/Markdown generation.

## Turn 13 | 2026-05-10

Summary: Tried AuraCall SoyLei ChatGPT agents for the P03 readout smoke.

Action:

- Selected `agent:pro-extended-chatgpt-soylei` from `/home/ecochran76/.auracall/config.json`.
- Ran `summarize_transcript.py` against the SoyLei/Tempo transcript artifact using the AuraCall OpenAI-compatible endpoint and model `agent:pro-extended-chatgpt-soylei`.
- Probed nearby SoyLei ChatGPT agent modes to separate profile readiness from pro/extended selector behavior.

Validation:

- `agent:pro-extended-chatgpt-soylei` returned HTTP 200 but an empty assistant message; no readout JSON/Markdown was written.
- AuraCall response metadata reported runner failure: `Unable to find the Thinking time dropdown menu.`
- `agent:instant-chatgpt-soylei` succeeded with non-empty content in 24.713 seconds.
- `agent:pro-standard-chatgpt-soylei` timed out after 109.183 seconds.
- `agent:thinking-extended-chatgpt-soylei` timed out after 109.175 seconds.

Next:

- Repair or adjust AuraCall's ChatGPT pro/thinking selector handling for the SoyLei profile, or run the P03 readout smoke with `agent:instant-chatgpt-soylei` as a temporary provider.

## Turn 14 | 2026-05-10

Summary: Retried AuraCall SoyLei pro-extended ChatGPT.

Action:

- Retried a direct one-line OpenAI-compatible smoke using `agent:pro-extended-chatgpt-soylei`.
- Kept the scope to a short smoke rather than resubmitting the full transcript, because the prior full readout failed with an empty assistant message.

Validation:

- The retry timed out after 163.74 seconds without a usable assistant response.
- `transcribe-watch.service` remained active.

Next:

- Fix AuraCall's SoyLei ChatGPT pro/thinking selector automation before using pro-extended for readouts, or temporarily use `agent:instant-chatgpt-soylei` to complete the P03 readout validation.

## Turn 15 | 2026-05-10

Summary: Passed AuraCall SoyLei pro-extended P03 readout smoke and closed P03.

Action:

- Retried a direct one-line smoke using `agent:pro-extended-chatgpt-soylei`; it succeeded with non-empty content in 15.898 seconds.
- Retried the SoyLei/Tempo transcript readout through `summarize_transcript.py`.
- Hardened `summarize_transcript.py` by duplicating the JSON-only response contract inside the user payload, because the browser-backed AuraCall path ignored the system-message-only contract on the prior attempt.

Validation:

- The hardened AuraCall readout generated valid JSON and Markdown under `/tmp/transcribe-readout-auracall-soylei-proextended-hardened.Q2cus3/`.
- Generated readout provider metadata records model `agent:pro-extended-chatgpt-soylei` and base URL `http://127.0.0.1:18095/v1`.
- The readout contains 4 participants, 15 topics, 9 action items, 5 matter candidates, and 5 memory candidates.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 19 tests.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.
- P03 is now closed in `ROADMAP.md` and `docs/dev/plans/0003-2026-05-04-intelligence-readouts.md`.

Next:

- Start P04 by defining the route decision schema and review queue for matter routing/contextual rereads.

## Turn 16 | 2026-05-10

Summary: Implemented and validated `codex-exec` readouts.

Action:

- Verified `codex exec` can return JSON through `--output-last-message` under read-only sandboxing with approval policy `never`.
- Added `codex-exec` support to `summarize_transcript.py`.
- `codex-exec` runs `codex --ask-for-approval never exec --sandbox read-only --ephemeral --output-last-message ...`, feeds the readout prompt over stdin, and validates the final message as JSON.
- Added unit coverage for command construction and JSON parsing.
- Ran the SoyLei/Tempo transcript readout through `codex-exec` with model `gpt-5.5`.

Validation:

- The direct `codex exec` smoke returned `{"ok":true,"provider":"codex-exec"}`.
- The SoyLei/Tempo `codex-exec` smoke generated valid readout JSON and Markdown under `/tmp/transcribe-readout-codex-exec-soylei.klslWj/`.
- Generated readout provider metadata records provider `codex-exec`, model `gpt-5.5`, and read-only/no-approval execution.
- The readout contains 4 participants, 13 topics, 8 action items, 4 matter candidates, and 5 memory candidates.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py -q` passed with 20 tests.
- `python -m py_compile summarize_transcript.py tests/test_readouts.py` passed.

Next:

- Start P04 by defining the route decision schema and review queue for matter routing/contextual rereads.

## Turn 17 | 2026-05-11

Summary: Started P04 with dry-run route decisions and local review queue support.

Action:

- Added `routing_artifacts.py` with `ContextProvenancePack`, `ProvenanceSource`, `RouteCandidate`, `RouteDecision`, and `ReviewQueueItem`.
- Added `route_transcript.py`, a dry-run CLI that reads existing transcript/readout artifacts and writes `*.route.json`.
- Current provenance extraction uses transcript calendar metadata, including `event.matching_calendars`, as the first `gws`-shaped context source.
- Current route candidates come from structured readout `matter_candidates`.
- Low-confidence route decisions write a local review queue item unless `--no-review-queue` is passed.
- Documented the dry-run route command in `README.md`.
- Moved P04 from PLANNED to OPEN in `ROADMAP.md` and `docs/dev/plans/0004-2026-05-04-matter-routing-contextual-reread.md`.

Validation:

- Added `tests/test_routing_artifacts.py` for provenance extraction, route selection, low-confidence review behavior, and CLI output.
- Manual route dry-run against the SoyLei/Tempo transcript and `codex-exec` readout selected `SoyLei Tempo Chemical technical collaboration` at confidence `0.95`.
- The manual route output had 4 route candidates, 3 rejected alternatives, and 4 provenance sources: one primary calendar event plus three calendar-overlap records.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py -q` passed with 24 tests.
- `python -m py_compile routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_routing_artifacts.py` passed.

Next:

- Add the live `gws` provenance adapter for Drive/Docs/Calendar context packs, then add Graphiti/OpenClaw candidate lookup as an advisory source.

## Turn 18 | 2026-05-11

Summary: Added live read-only `gws` provenance for P04 route decisions.

Action:

- Added `context_sources.py` with a `GwsProvenanceConfig` and read-only `gws` collection helpers.
- `route_transcript.py --gws-provenance` now collects live Calendar event details and Drive metadata search results into the route decision `provenance_pack`.
- Added `--gws-config-dir`, `--gws-drive-query`, `--gws-drive-page-size`, `--gws-timeout`, `--no-gws-calendar-details`, and `--no-gws-drive` flags.
- Default generated Drive queries now use precise filename-term intersections; operators can pass `--gws-drive-query` for broader full-text searches.
- Documented the live `gws` provenance mode in `README.md`.

Validation:

- Added `tests/test_context_sources.py` for generated Drive query behavior, `gws` response conversion, and `route_transcript.py` integration.
- Live read-only `gws` smoke against the SoyLei/Tempo transcript/readout selected `SoyLei Tempo Chemical technical collaboration` and produced 7 provenance sources: one stored calendar event, three calendar overlaps, and three live `gws_calendar_event_detail` records.
- No noisy Drive full-text hits were included with the default generated query after tightening to filename intersections.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 27 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py` passed.

Next:

- Add Graphiti/OpenClaw candidate lookup as an advisory routing source and keep it source-cited alongside `gws` provenance.

## Turn 19 | 2026-05-11

Summary: Started P07 with portable OpenClaw workspace files for the `transcripts` agent.

Action:

- Read OpenClaw docs for agent config, `openclaw agents`, channel routing, Slack channel behavior, and workspace Markdown templates.
- Confirmed the live OpenClaw Slack setup has a configured `default` account and a separate `soylei` account; the `transcripts` agent must target `slack/default`.
- Added portable workspace Markdown files under `openclaw/agents/transcripts/workspace/`.
- Added `openclaw/agents/transcripts/INSTALL.md` documenting the safe install flow and exact Slack channel-peer binding shape.
- Added `scripts/install_openclaw_transcripts_agent.py`, a dry-run-first installer scaffold that copies Markdown files to `~/.openclaw/workspace-transcripts` and runs safe agent creation/identity commands only with `--apply`.
- Opened P07 in `ROADMAP.md` and added `docs/dev/plans/0007-2026-05-11-openclaw-transcripts-agent.md`.

Validation:

- Graphiti runtime doctor reported healthy before the planning slice.
- Graphiti discovery against `transcribe_audio_main` returned existing repo planning and policy context, but no prior `transcripts` agent install routine.
- `openclaw agents list --json` showed no existing `transcripts` agent.
- `openclaw channels status --json` showed Slack account `default` is configured and running.
- `openclaw directory peers list --channel slack --query oc-transcripts --limit 20 --json` returned no matching peer, so the private channel still needs to be created or made visible to the Slack app.
- `python -m py_compile scripts/install_openclaw_transcripts_agent.py` passed.
- `scripts/install_openclaw_transcripts_agent.py` dry-run showed the expected Markdown copy targets, safe `openclaw agents add` command, identity command, and route-binding patch shape.
- Planning audit passed with `ok: true`; open lanes are P04, P06, and P07.

Next:

- Resolve or create private Slack channel `oc-transcripts`, obtain its Slack conversation id, then extend/apply the installer to add the exact `slack/default` channel-peer route binding.

## Turn 20 | 2026-05-11

Summary: Installed and live-verified the OpenClaw `transcripts` agent on Slack.

Action:

- Consulted the existing `gpod` OpenClaw agent for the channel-binding workflow; it confirmed the safe pattern of exact Slack conversation id, one route binding, channel allowlist, and post-restart validation.
- Created private Slack channel `oc-transcripts` on the default Slack tenant.
- Resolved Slack conversation id `C0B3WDRN38Q`.
- Invited the OpenClaw bot to the private channel.
- Applied the portable `transcripts` workspace files to `~/.openclaw/workspace-transcripts`.
- Created OpenClaw agent `transcripts` and set identity.
- Added exactly one route binding for `transcripts`: `slack accountId=default peer=channel:C0B3WDRN38Q`.
- Added Slack per-channel config for `C0B3WDRN38Q` with `enabled: true`, `requireMention: false`, and user allowlist `UEGM25PMG`.
- Updated the installer so `--slack-channel-id` can apply the binding idempotently and so identity setup no longer depends on indirect `IDENTITY.md` resolution.
- Closed P07 in `ROADMAP.md` and `docs/dev/plans/0007-2026-05-11-openclaw-transcripts-agent.md`.

Validation:

- `openclaw config validate` passed after binding.
- `openclaw gateway status --deep --require-rpc` passed after restart with read probe `ok` and capability `admin-capable`.
- `openclaw agents list --bindings --json` shows `transcripts` with one binding: `slack accountId=default peer=channel:C0B3WDRN38Q`.
- Slack API confirmed user and OpenClaw bot membership in `oc-transcripts`.
- A live Slack smoke message in `oc-transcripts` routed to `transcripts`; the bot replied `TRANSCRIPTS_BINDING_SMOKE_OK`.
- `python -m py_compile scripts/install_openclaw_transcripts_agent.py` passed.

Notes:

- OpenClaw directory search did not list the new private channel immediately, but Slack API, OpenClaw session state, and the live routed Slack response verified the channel and route.
- OpenClaw session created: `agent:transcripts:slack:channel:c0b3wdrn38q`.

Next:

- Continue P04 by adding Graphiti/OpenClaw advisory candidate lookup for route decisions, now using `transcripts` as the Slack operational surface for review and routing work.

## Turn 21 | 2026-05-11

Summary: Added read-only Graphiti/OpenClaw advisory provenance for P04 route decisions.

Action:

- Added `GraphitiProvenanceConfig` and read-only `graphiti-runtime discover` helpers in `context_sources.py`.
- `route_transcript.py --graphiti-provenance` now queries compact calendar/readout terms rather than raw transcript text.
- Added `--graphiti-group`, `--graphiti-command`, `--graphiti-timeout`, `--graphiti-max-facts`, `--graphiti-max-nodes`, `--graphiti-max-episodes`, and `--no-graphiti-candidates`.
- Graphiti facts and episodes are stored as provenance evidence only.
- Graphiti nodes can add low-confidence advisory `RouteCandidate` entries; they do not override high-confidence readout candidates.
- Documented Graphiti routing usage in `README.md`.
- Updated P04 roadmap and plan text to mark Graphiti/OpenClaw advisory lookup implemented.

Validation:

- `graphiti-runtime doctor` reported healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing advisory planning facts and confirmed the repo memory policy, but no prior route adapter implementation.
- Added tests for Graphiti query construction, discovery-payload conversion, and route integration.
- `.venv/bin/python -m pytest tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 10 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py tests/test_context_sources.py tests/test_routing_artifacts.py` passed.
- Live read-only Graphiti route smoke on the SoyLei/Tempo transcript/readout selected the same readout candidate, `SoyLei Tempo Chemical technical collaboration`, at confidence `0.95`.
- Live smoke output included 27 Graphiti provenance sources and 10 node-based advisory Graphiti candidates; the route remained `status=selected` and `review_required=false`.

Next:

- Add local candidate index support or move into contextual reread source fetching for the selected route.

## Turn 22 | 2026-05-11

Summary: Added read-only Odollo/Odoo provenance for P04 route decisions.

Action:

- Added `OdolloProvenanceConfig` and `route_transcript.py --odollo-provenance`.
- Default Odollo profiles are the two configured production tenants: `soylei-prod` and `saber-prod`.
- Added repeated `--odollo-profile` selectors plus command, repo, config, timeout, limit, contact, and log-note flags.
- Odollo contact and log-note searches use compact meeting/readout/attendee terms, not raw transcript text.
- Route provenance can now include `odollo_contact` and `odollo_log_note` sources.
- Odoo log-note bodies may be searched for matches but are not stored in route provenance snippets or metadata.
- Updated README, roadmap, and P04 plan text to mark Odollo provenance as implemented and evidence-only.

Validation:

- Graphiti runtime doctor was healthy before the routing change.
- Graphiti discovery against `transcribe_audio_main` returned existing repo routing context and no prior Odollo route adapter implementation.
- Odollo config inspection found the production profiles `soylei-prod` and `saber-prod`; the repository also has dev/test profiles that are not used by default.
- Added tests for Odollo compact query terms, Odollo source conversion, and route integration.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 33 tests.
- `python -m py_compile context_sources.py routing_artifacts.py route_transcript.py tests/test_context_sources.py tests/test_routing_artifacts.py` passed.
- Live Odollo doctor check reported SoyLei production Odoo readiness `ready`.
- Live Odollo doctor check reported Saber production Odoo readiness `degraded` because Amazon product fields/views are missing, but the contact/log-note read path used by this adapter worked.
- Live read-only Odollo route smoke on the SoyLei/Tempo transcript/readout selected `SoyLei Tempo Chemical technical collaboration`, had `review_required=false`, and added 12 Odollo provenance sources across `soylei-prod` and `saber-prod`.

Next:

- Move into contextual reread source fetching for selected routes, using calendar, gws, Graphiti, and Odollo provenance packs as cited inputs.

## Turn 23 | 2026-05-11

Summary: Added contextual reread generation for selected P04 routes.

Action:

- Added `contextual_reread.py` to generate upgraded readouts from transcript, prior readout, and route decision artifacts.
- Contextual rereads reuse the existing `openai-compatible` and `codex-exec` provider paths.
- Supporting context is built from the selected candidate's cited provenance sources plus calendar context unless `--all-provenance` is passed.
- `Readout` JSON now records `contextualization.supporting_context_sources`.
- Readout Markdown now renders supporting context sources.
- The readout prompt now accepts `prior_readout`, `route_decision`, and `supporting_context` blocks and instructs providers to cite source labels or ids in evidence fields.
- Updated README, roadmap, and P04 plan text.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned older P04 planning context; repo files remained the authority for this slice.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py -q` passed with 37 tests.
- `python -m py_compile context_sources.py contextual_reread.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- Full `codex-exec` contextual reread smoke on the SoyLei/Tempo transcript/readout/route succeeded.
- Smoke output: `/tmp/transcribe-contextual-reread-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Smoke output included 10 supporting context sources: 4 `odollo_contact`, 3 `odollo_log_note`, and 3 `gws_calendar_overlap` sources.

Next:

- Add deeper Google Drive/Docs content fetch for selected route sources or move to P05 deposition/memory harvest preview contracts.

## Turn 24 | 2026-05-11

Summary: Started P05 with a no-write deposition and memory-harvest preview contract.

Action:

- Added `deposition_artifacts.py` with `DepositAction`, `MemoryHarvestCandidate`, and `DepositPreview`.
- Added `deposition_preview.py` to generate `*.deposit-preview.json` from readout/contextual-readout artifacts.
- Preview actions can describe local filesystem, Google Drive, and Odoo targets but always use `status=preview` and `writes_enabled=false`.
- Memory harvest candidates are extracted only from structured readout `memory_candidates`.
- Transcript artifact paths are excluded from deposition action source paths unless `--include-transcript` is explicitly passed.
- Raw transcript text is never harvested into memory candidates.
- Moved P05 to OPEN in roadmap and plan docs.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` confirmed P05 was still a planned lane and returned repo memory policy facts.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py -q` passed with 41 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- No-write preview smoke over the SoyLei/Tempo contextual readout wrote `/tmp/transcribe-deposition-preview-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json`.
- Smoke output contained 3 preview actions: local filesystem, Google Drive, and Odoo record.
- Smoke output contained 7 memory candidates, all with `status=preview`.
- Smoke output did not include the transcript artifact path in deposition action source paths by default.

Next:

- Add the first apply path for local filesystem deposition, keeping Drive/Odoo/Graphiti as preview-only until their write contracts are explicitly selected.

## Turn 25 | 2026-05-11

Summary: Added local filesystem apply for reviewed deposition previews.

Action:

- Added `DepositApplyResult`, `AppliedDepositAction`, and `AppliedDepositFile` result schemas.
- Added `deposition_apply.py` to consume `*.deposit-preview.json` artifacts.
- Local apply handles only `local_filesystem` actions.
- Google Drive, Odoo, and other non-local actions are skipped with `writes_enabled=false`.
- Apply refuses previews with `review_required=true` unless `--allow-review-required` is passed.
- Local copies are idempotent: same destination hash is skipped, conflicting filenames are versioned.
- Updated README, roadmap, and P05 plan text.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing P05 planning and memory-policy facts, but no newer local apply contract.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py -q` passed with 45 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py` passed.
- Local apply smoke over the SoyLei/Tempo preview wrote `/tmp/transcribe-deposition-apply-smoke/2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-apply.json`.
- First local apply copied the preview's readout and route artifacts into the preview local target.
- Second local apply skipped both local files by same-hash idempotency.
- Google Drive and Odoo preview actions remained skipped during apply.

Next:

- Add a reviewed Graphiti memory-harvest apply path with duplicate preflight, keeping raw transcript text excluded and Drive/Odoo still preview-only.

## Turn 26 | 2026-05-11

Summary: Added user-scoped transcript/readout store and search.

Action:

- Added `transcript_store.py` with a SQLite runtime store under `~/.transcripts`.
- Store artifacts are copied under `~/.transcripts/artifacts/`.
- The database indexes transcript artifacts, first-pass readouts, and contextual readouts.
- Added SQLite FTS5 lexical search.
- Added deterministic local token-hash embeddings for semantic-style ranking without an external provider dependency.
- Added `summarize_transcript.py --store` and `contextual_reread.py --store`.
- Added transcription opt-in with `TRANSCRIPTS_STORE=true` and optional `TRANSCRIPTS_STORE_DIR`.
- Added P08 roadmap and plan docs for the user-scoped store/search lane.

Validation:

- Graphiti runtime doctor was healthy before implementation.
- Graphiti discovery against `transcribe_audio_main` returned existing artifact/readout facts but no prior store implementation.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 49 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_store.py` passed.
- Temp-store smoke ingested the SoyLei/Tempo transcript, first-pass readout, and contextual readout into `/tmp/transcripts-store-smoke`.
- Temp-store search for `Tempo Chemical concrete sealer` returned 3 results: readout, contextual readout, and transcript.
- Initialized the actual user-scoped store at `/home/ecochran76/.transcripts/transcripts.sqlite3`.
- Ingested the same three validated SoyLei/Tempo artifacts into `/home/ecochran76/.transcripts`.
- Real-store search for `Tempo Chemical concrete sealer` returned 3 results.

Next:

- Add watcher config support for automatic store ingestion and then backfill recent artifacts from Downloads into `/home/ecochran76/.transcripts`.

## Turn 27 | 2026-05-11

Summary: Replaced token-hash semantics with provider-backed embeddings.

Action:

- Inspected adjacent `../imcli` and `../ragmail` embedding patterns.
- Updated `transcript_store.py` to default to local Ollama embeddings with `ollama/nomic-embed-text`.
- Added `openai-compatible` embedding support with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- Kept `debug-hash` and `hash` only as explicit test/offline fallbacks.
- Stored embedding provider/model metadata on documents and filtered semantic search to matching provider/model rows.
- Added `nomic-embed-text` `search_document:` and `search_query:` formatting for ingest and search.
- Added long-document chunking with averaged document vectors so full transcripts do not overflow the embedding provider context.
- Updated README, roadmap, and P08 plan docs to reflect the real embedder path.

Validation:

- Graphiti runtime doctor was healthy.
- Graphiti discovery against `transcribe_audio_main` returned older artifact/readout facts but no store/embedder implementation.
- `ollama list` showed `nomic-embed-text:latest` available locally.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 7 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 52 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_store.py` passed.
- Initial live Ollama smoke failed on the full transcript with `the input length exceeds the context length`; chunking fixed that failure.
- Temp-store smoke ingested the SoyLei/Tempo transcript, first-pass readout, and contextual readout into `/tmp/transcripts-store-ollama-smoke` with Ollama/Nomic vectors.
- Temp-store search for `Tempo Chemical concrete sealer` returned 3 results with `embedding_provider=ollama` and `embedding_model=ollama/nomic-embed-text`, ranking the contextual readout first.
- Re-ingested the same three validated SoyLei/Tempo artifacts into `/home/ecochran76/.transcripts` with Ollama/Nomic vectors.
- Real-store search for `Tempo Chemical concrete sealer` returned 3 Ollama/Nomic-backed results, ranking the contextual readout first.

Next:

- Backfill recent Downloads artifacts into `/home/ecochran76/.transcripts`, then add watcher config support for automatic store ingestion.

## Turn 28 | 2026-05-11

Summary: Added watcher store ingestion and backfilled recent artifacts.

Action:

- Added a watcher job `store` config block with `enabled`, `store_dir`, `embedding_provider`, and `embedding_model`.
- The watcher now ingests successful transcript artifacts and generated readouts into the transcript store after transcription/readout processing.
- Watcher state now preserves `store_paths` for auditability.
- Updated `watch_transcriptions.json.sample`, README, roadmap, and P08 plan docs.
- Enabled store ingestion in the live `watch_transcriptions.json` job for `downloads-mobile-recordings`.
- Tightened embedding chunk size to avoid local Ollama/Nomic context overflow during larger recent-artifact backfills.
- Backfilled 9 recent Downloads transcript artifacts into `/home/ecochran76/.transcripts`.
- Backfilled 4 recent readout/contextual-readout artifacts from temp smoke locations into `/home/ecochran76/.transcripts`.
- Restarted `transcribe-watch.service` so the new watcher store config is active.

Validation:

- Graphiti runtime doctor was healthy.
- Graphiti discovery against `transcribe_audio_main` confirmed watcher artifact-path facts but no prior store-ingestion implementation.
- Watcher config parsed with `store_enabled=True`, `store_dir=/home/ecochran76/.transcripts`, `embedding_provider=ollama`, and `embedding_model=ollama/nomic-embed-text`.
- `/home/ecochran76/.transcripts/transcripts.sqlite3` now has Ollama/Nomic rows for 9 transcripts, 3 readouts, and 1 contextual readout.
- Store search for `SIP WMA Hamburg` returned Ollama/Nomic-backed results from the recent Downloads backfill.
- `systemctl --user restart transcribe-watch.service` succeeded.
- `systemctl --user status transcribe-watch.service --no-pager` showed the service active with PID 78937 and loaded job `downloads-mobile-recordings`.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 55 tests.
- `python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.

Next:

- Add a store backfill command that can enumerate artifact paths, dry-run counts, and skip/ingest deterministically without shell one-liners.

## Turn 29 | 2026-05-11

Summary: Added deterministic transcript-store backfill command.

Action:

- Added `transcript_store.py backfill` for files or directories.
- Backfill discovers `*.transcript.json`, `*.readout.json`, and `*.contextual.readout.json` by default.
- Added `--dry-run`, `--modified-within-days`, repeated `--kind`, repeated `--pattern`, `--recursive`, `--limit`, and `--force`.
- Backfill reports selected counts by kind and status.
- Current artifacts with matching source path, artifact hash, embedding provider, and embedding model are skipped without re-embedding unless `--force` is used.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was degraded: MCP HTTP was down, while FalkorDB and local inspector were healthy. Work continued from repo authority.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 10 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live dry-run over `/mnt/c/Users/ecoch/Downloads --modified-within-days 14` selected 9 transcript artifacts and reported all 9 as `skip`.
- Live apply over the same Downloads path selected 9 transcript artifacts and skipped all 9 without re-embedding.

Next:

- Add chunk-level storage/retrieval so long transcripts can return precise segment hits instead of only averaged document-level semantic scores.

## Turn 30 | 2026-05-11

Summary: Added chunk-level semantic retrieval.

Action:

- Added `document_chunks` table with per-document chunk text, vector JSON, provider, and model metadata.
- Ingest now embeds chunks and stores both averaged document vectors and per-chunk vectors.
- Search now scores document-level semantic matches and best chunk-level semantic matches.
- Search results include `chunk_semantic_score` and `best_chunk` with chunk index, score, and snippet.
- Backfill status now treats documents without matching chunk rows as `update`, so older rows can be migrated by re-running backfill.
- Backfill now reports invalid matching files as `error` instead of aborting the whole scan.
- Added default excludes for copied store internals and support for additive `--exclude` patterns.
- Cleaned the live store after an overly broad recursive `/tmp` apply ingested pytest fixture artifacts; removed 54 unintended rows and restored the live store to the intended 13 documents.
- Rebackfilled recent Downloads transcripts to populate chunk rows.
- Rebackfilled known recent `transcribe-*` readout/contextual artifacts to populate chunk rows.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no chunk-level store implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 15 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live Downloads dry-run selected 9 transcripts as `update` before chunk migration.
- Live Downloads apply updated 9 transcripts and populated chunk rows.
- Live store now has 13 documents and 369 chunk rows: 9 transcripts, 3 readouts, and 1 contextual readout.
- Search for `SIP WMA Hamburg` returned `best_chunk` snippets and chunk semantic scores.
- Broad `/tmp` dry-run with `--exclude '*/pytest-of-*/*'` selected only 5 non-pytest transcribe/readout candidates; 4 were already current and 1 reviewed-preview duplicate would insert if applied.

Next:

- Add transcript-aware chunk metadata for utterance timestamp ranges and speaker spans so `best_chunk` can point back to recording time, not just text.

## Turn 31 | 2026-05-12

Summary: Added transcript timestamp and speaker metadata to store chunks.

Action:

- Added chunk character offsets during text chunking.
- Added `metadata_json` to `document_chunks`.
- Transcript chunk metadata now includes `char_start`, `char_end`, `start_seconds`, `end_seconds`, `speakers`, and `utterance_count` when structured utterances are available.
- `best_chunk` search results now surface timestamp range, speaker list, utterance count, and full metadata.
- Backfill status now marks transcript rows as `update` when matching chunks exist but lack timestamp/speaker metadata.
- Updated README, roadmap, and P08 plan docs.
- Rebackfilled 9 recent Downloads transcripts to populate timestamp/speaker metadata in live chunk rows.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no transcript chunk metadata implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 16 tests.
- `python -m py_compile transcript_store.py tests/test_transcript_store.py` passed.
- Live Downloads dry-run selected 9 transcripts as `update` before metadata migration.
- Live Downloads apply updated 9 transcripts.
- Live store now has 13 documents, 369 chunk rows, and 247 transcript chunks with non-empty metadata.
- Live search for `Hamburg sample` returned `best_chunk` with `start_seconds`, `end_seconds`, `speakers`, and `utterance_count`.
- Follow-up live Downloads dry-run reported all 9 transcripts as `skip`.

Next:

- Add a user-facing `open`/`context` command that can take a search result document/chunk and show nearby transcript context or media timestamp instructions.

## Turn 32 | 2026-05-12

Summary: Added transcript-store context view for search hits.

Action:

- Added `transcript_store.py context <document-id>` with optional `--chunk-index` and `--context-chunks`.
- The context view prints document metadata, artifact path, media path, timestamp range, speaker list, an `ffplay -ss` seek hint, and nearby chunk text.
- Added JSON-format support through `--format json` and a `TRANSCRIPT_CONTEXT_JSON=` sentinel for CLI automation.
- Added tests for timestamp/media context extraction and CLI output.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older artifact/readout facts, but no prior context command implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 18 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 66 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live search for `Hamburg sample --kind transcript` returned document `1711b25666b79b3142d1` with `best_chunk.chunk_index=5`.
- Live context smoke for document `1711b25666b79b3142d1` chunk `5` printed chunks 4-6, timestamp `08:02.96 - 10:04.86`, speakers `A, B`, and a media seek hint for the original `.m4a`.

Next:

- Add a convenience flow that pipes a search result directly into `context`, so operators do not have to copy the document id and chunk index manually.

## Turn 33 | 2026-05-12

Summary: Added direct search-to-context shortcut.

Action:

- Added `transcript_store.py search --context`.
- Added `--context-rank` to choose a 1-based search hit and `--context-chunks` to control the nearby transcript window.
- Kept normal `search` output unchanged unless `--context` is passed.
- Added CLI test coverage for opening the best search chunk directly.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts, but no prior search-to-context implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 19 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 67 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-chunks 1` opened document `1711b25666b79b3142d1`, chunk `5`, timestamp `08:02.96 - 10:04.86`, and printed the media seek hint.

Next:

- Add compact JSON output for machine consumers of the context view, including selected search metadata when `search --context` is used.

## Turn 34 | 2026-05-12

Summary: Added compact JSON context output.

Action:

- Added `context --format compact-json` for pure single-line JSON without a sentinel.
- Added `search --context --context-format compact-json`.
- Search-to-context compact output includes `query`, `result_count`, `selected_rank`, `selected_result`, and the full `context` payload.
- Added test coverage for both compact JSON entrypoints.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts, but no prior compact context JSON implementation.
- `.venv/bin/python -m pytest tests/test_transcript_store.py -q` passed with 21 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py -q` passed with 69 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py tests/test_context_sources.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | jq ...` parsed cleanly and returned document `1711b25666b79b3142d1`, chunk `5`, timestamp `08:02.96`, and the media seek hint.
- Live smoke `transcript_store.py context 1711b25666b79b3142d1 --chunk-index 5 --format compact-json | jq ...` parsed cleanly and returned the expected document, chunk, and timestamp.

Next:

- Add a small CLI recipe for piping compact context JSON into downstream routing/readout tools.

## Turn 35 | 2026-05-12

Summary: Added compact context downstream recipe helper.

Action:

- Added `scripts/context_packet_recipe.py`.
- The helper reads direct `context --format compact-json` packets or `search --context --context-format compact-json` packets from a file or stdin.
- The helper validates `context.document.source_path` and prints non-mutating shell commands for `summarize_transcript.py`, `route_transcript.py`, and `contextual_reread.py`.
- Added `--readout`, `--route`, `--provider`, `--model`, `--store`, and `--with-provenance` options.
- Added tests for stdin packets and explicit readout/route paths.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned existing readout/routing facts but no prior compact context recipe helper.
- `.venv/bin/python -m pytest tests/test_context_packet_recipe.py -q` passed with 2 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py -q` passed with 71 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live pipe smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | scripts/context_packet_recipe.py - --store --with-provenance` printed a valid downstream recipe for document `1711b25666b79b3142d1`, chunk `5`.

Next:

- Add an apply-style helper that can optionally execute the recipe steps while preserving preview/apply boundaries.

## Turn 36 | 2026-05-12

Summary: Added preview-first context packet apply helper.

Action:

- Added `scripts/context_packet_apply.py`.
- The helper reads compact context packets from stdin or a file.
- Preview is the default; downstream commands execute only when `--apply` is present.
- Execution runs first-pass readout when `--readout` is absent, routing when `--route` is absent, then contextual reread with the resolved artifact paths.
- The helper captures `READOUT_JSON=...`, `ROUTE_DECISION_JSON=...`, and `CONTEXTUAL_READOUT_JSON=...` stdout sentinels.
- Added tests for preview mode, existing-path skips, and fake-runner execution.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Re-read runtime/tenant, memory/routing, and git/validation policies before implementing the apply boundary.
- Graphiti runtime doctor was healthy and discovery returned existing readout/routing facts but no prior preview/apply helper.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py tests/test_context_packet_recipe.py -q` passed with 5 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 74 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Live preview smoke `transcript_store.py search "Hamburg sample" --kind transcript --context --context-format compact-json --context-chunks 1 | scripts/context_packet_apply.py - --store --with-provenance` printed a preview plan for document `1711b25666b79b3142d1`, chunk `5`, without executing downstream commands.

Next:

- Add an artifact manifest for executed context-packet apply runs so generated readout, route, and contextual-readout paths are captured in one durable JSON record.

## Turn 37 | 2026-05-12

Summary: Added executed-run manifests for context packet apply.

Action:

- Updated `scripts/context_packet_apply.py` to write a manifest after successful `--apply` runs.
- Default manifest directory is `~/.local/state/transcribe-audio/context-packet-runs/`, keeping live operator state out of the repo.
- Added `--manifest-dir` and `--no-manifest`.
- Manifest schema captures transcript path, query, selected store document/chunk, generated readout/route/contextual-readout paths, and sanitized step metadata.
- Manifest intentionally omits raw context chunks and command stdout/stderr.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Re-read runtime/tenant, memory/routing, and git/validation policies before adding runtime manifests.
- Graphiti runtime doctor was healthy and discovery returned older artifact-path facts but no prior context-packet apply manifest implementation.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py -q` passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 74 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Temp live `--apply` smoke used fake downstream scripts from a temp directory, wrote a manifest under the temp manifest dir, captured fake readout/route/contextual-readout paths, and confirmed the manifest step records omit raw `stdout`.

Next:

- Add a manifest listing command so operators can inspect recent context-packet apply runs without browsing the runtime directory manually.

## Turn 38 | 2026-05-12

Summary: Added manifest listing for context packet apply runs.

Action:

- Added `scripts/context_packet_apply.py --list-manifests`.
- Added `--limit` and JSON/text output support for manifest lists.
- Listing reads sanitized manifest summaries from `--manifest-dir` without reading raw context chunks or command stdout/stderr.
- Added test coverage for recent-manifest listing.
- Updated README, roadmap, and P08 plan docs.

Validation:

- Graphiti runtime doctor was healthy and discovery returned older repo facts but no prior context-packet manifest listing implementation.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py -q` passed with 4 tests.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_context_sources.py tests/test_deposition_preview.py tests/test_deposition_apply.py tests/test_transcript_store.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py -q` passed with 75 tests.
- `.venv/bin/python -m py_compile context_sources.py contextual_reread.py deposition_apply.py deposition_artifacts.py deposition_preview.py readout_artifacts.py routing_artifacts.py route_transcript.py summarize_transcript.py transcript_store.py transcribe_common.py watch_transcriptions.py scripts/context_packet_recipe.py scripts/context_packet_apply.py tests/test_context_sources.py tests/test_context_packet_recipe.py tests/test_context_packet_apply.py tests/test_deposition_apply.py tests/test_deposition_preview.py tests/test_readouts.py tests/test_routing_artifacts.py tests/test_transcript_artifacts.py tests/test_transcript_store.py` passed.
- Temp manifest-list smoke read one demo manifest through `context_packet_apply.py --list-manifests --format json` and returned count `1`, run id `demo`, and the contextual readout path.

Next:

- Review P08 definition-of-done and decide whether to close the transcript store/search lane or keep it open for UI/operator polish.

## Turn 39 | 2026-05-12

Summary: Closed P08 transcript store/search lane.

Action:

- Reviewed P08 definition of done against current implementation and runbook validation.
- Closed P08 in `ROADMAP.md`.
- Closed `docs/dev/plans/0008-2026-05-11-transcript-store-search.md`.
- Recorded that future UI/operator polish should be tracked separately rather than holding the core store/search lane open.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded and returned only older roadmap/source facts.
- P08 definition of done is satisfied: user-scoped store initializes without repo secrets, transcript/readout/contextual-readout artifacts ingest and copy into the store, lexical/semantic search returns ranked JSON, and watcher/service flows can opt into automatic ingestion.
- Latest focused suite evidence remains Turn 38: 75 tests passed plus py_compile.

Next:

- Move back to P04/P05 integration work: run one real reviewed context-packet apply on a known transcript/readout pair, then feed the generated contextual readout into deposition preview.

## Turn 40 | 2026-05-12

Summary: Ran a real P04/P05 context-packet apply through deposition preview.

Action:

- Selected the stored SoyLei/Tempo transcript/readout pair from `~/.transcripts`.
- Generated a compact context packet from `transcript_store.py search "Tempo Chemical concrete sealer" --kind transcript --context --context-format compact-json --context-chunks 1`.
- Fixed `scripts/context_packet_apply.py` child interpreter selection so downstream commands default to repo `.venv/bin/python` when present instead of the parent `sys.executable`.
- Added `--python` for explicit child interpreter override.
- Added `--provider-timeout` so `codex-exec` provider calls can have a longer request timeout than the default 120 seconds.
- Re-ran the reviewed context-packet apply with read-only `gws`, Graphiti, and Odollo provenance and an existing readout.
- Fed the generated contextual readout and route into `deposition_preview.py` with no writes enabled.
- Updated README, ROADMAP, and P04/P05 plan docs with the integration evidence and remaining provenance-quality risk.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- Initial apply exposed two integration issues: child Python lacked `requests`, and `codex-exec` contextual reread hit its default 120-second timeout.
- `.venv/bin/python -m pytest tests/test_context_packet_apply.py` passed with 4 tests after the wrapper fixes.
- Full `.venv/bin/python -m pytest` passed with 75 tests.
- Successful apply manifest: `/home/ecochran76/.local/state/transcribe-audio/context-packet-runs/2026-05-12T16-04-03Z-930f0fc94f9abe19d050.json`.
- Generated route: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.route.json`.
- Generated contextual readout: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Deposition preview: `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json`.
- Route selected `SoyLei Tempo Chemical technical collaboration` with confidence `0.95` and `review_required=false`.
- Deposition preview produced one local-filesystem copy action with `writes_enabled=false` and six Graphiti memory-harvest candidates.
- Integration risk: some retrieved supporting provenance was broad/noisy, so source-quality filtering should precede unattended deposition or live memory harvest.

Next:

- Add provenance-source quality filtering and route/readout warnings so irrelevant Graphiti/Odollo context cannot silently flow into contextual rereads or memory-harvest previews.

## Turn 41 | 2026-05-12

Summary: Added provenance-source quality filtering for routing and contextual rereads.

Action:

- Added compact provenance quality terms derived from calendar summary, participants, readout title/topics, and matter candidates without using raw transcript text.
- Added `filter_provenance_sources()` to retain calendar sources by default and require non-calendar sources to match enough meeting-specific terms.
- Ignored retrieval-control metadata such as Graphiti query strings, Odollo matched-term lists, and quality annotations during source scoring.
- Added `--provenance-quality-threshold` and `--no-provenance-quality-filter` to `route_transcript.py`.
- Extended route provenance packs with `excluded_sources` and `warnings`.
- Added quality metadata to retained/excluded provenance sources: `quality_status`, `quality_score`, `quality_matched_terms`, and `quality_reason`.
- Propagated route warnings and excluded-source counts into contextual reread support packets and readout contextualization metadata.
- Rendered contextual warning sections in readout Markdown.
- Updated README, ROADMAP, and P04/P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_readouts.py` passed with 23 tests.
- `.venv/bin/python -m py_compile context_sources.py route_transcript.py routing_artifacts.py contextual_reread.py readout_artifacts.py tests/test_context_sources.py tests/test_readouts.py` passed.
- Live route smoke over the SoyLei/Tempo transcript/readout with `--gws-provenance --graphiti-provenance --odollo-provenance` wrote `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.route.json`.
- The live route retained 7 calendar-derived sources and excluded 35 weak Graphiti/Odollo sources below threshold 2.
- The live route warning is `Excluded 35 provenance source(s) below quality threshold 2.`
- A direct contextual support build from the live route carried `excluded_source_count=35` and the same warning forward.

Next:

- Add source-type-specific provenance scoring so true Odollo/Drive/Graphiti hits can be retained on stronger evidence than generic term overlap, then rerun a contextual reread/deposition preview smoke with the warning surface visible.

## Turn 42 | 2026-05-12

Summary: Added source-type-specific provenance scoring and deposition preview warnings.

Action:

- Tightened provenance quality scoring to use source-type-specific identity fields.
- Drive sources now score on file name/snippet plus limited file identity metadata, not the Drive query.
- Odollo contacts now score on contact label/snippet/email/company, not the search matched-term list.
- Odollo log notes now score on note label/snippet and related record identifiers, not author/date/body or matched-term metadata.
- Graphiti sources now score on labels/previews and limited source descriptions, not the original discovery query.
- Added quality reason profiles such as `drive_file_identity`, `odollo_contact_identity`, `odollo_log_note_subject`, and `graphiti_label_or_preview`.
- Added deposition preview `warnings` so route/contextual warnings remain visible at the no-write deposition review point.
- Updated README, ROADMAP, and P04/P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded.
- `.venv/bin/python -m pytest tests/test_context_sources.py` passed with 12 tests after the scoring change.
- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_deposition_preview.py` passed with 16 tests after adding preview warnings.
- `.venv/bin/python -m py_compile context_sources.py deposition_artifacts.py deposition_preview.py tests/test_context_sources.py tests/test_deposition_preview.py` passed.
- Live SoyLei/Tempo route smoke retained 7 calendar-derived sources and excluded 35 weak Graphiti/Odollo sources with source-profile quality reasons.
- Live contextual reread with `codex-exec --timeout 600` regenerated `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.contextual.readout.json`.
- Contextual readout metadata carries `excluded_source_count=35`, 7 supporting context sources, and warning `Excluded 35 provenance source(s) below quality threshold 2.`
- Contextual Markdown renders `## Context Warnings`.
- Regenerated deposition preview `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.deposit-preview.json` now includes preview warnings.

Next:

- Add a reviewed memory-harvest approval/apply contract for Graphiti candidates, keeping live writes disabled until a preview artifact is explicitly approved.

## Turn 43 | 2026-05-12

Summary: Added reviewed Graphiti memory-harvest apply contract.

Action:

- Added `memory_harvest_apply.py`.
- Default mode is a dry-run preview that writes `*.memory-harvest-apply.json`.
- Live Graphiti writes require both `--apply` and `--approval-token APPROVE_GRAPHITI_MEMORY_HARVEST`.
- The CLI refuses `review_required` previews unless `--allow-review-required` is passed after review.
- The CLI refuses warning-bearing previews unless `--allow-warnings` is passed after review.
- Added repeated `--candidate-id` filtering for reviewed subsets.
- Apply command bodies are built only from structured deposition-preview `memory_candidates`; raw transcript text is not read or harvested.
- Dry-run mode does not write temporary memory body files.
- Added `AppliedMemoryHarvestCandidate` and `MemoryHarvestApplyResult` schemas.
- Updated README, ROADMAP, and P05 plan docs.

Validation:

- Graphiti `doctor` was degraded only because Inspector ingress/Traefik was down, but MCP and FalkorDB were healthy; `graphiti-runtime discover --group-id transcribe_audio_main` succeeded and returned older P05 policy/planning facts.
- `.venv/bin/python -m pytest tests/test_memory_harvest_apply.py` passed with 5 tests.
- `.venv/bin/python -m py_compile memory_harvest_apply.py deposition_artifacts.py tests/test_memory_harvest_apply.py` passed.
- Live dry-run refusal over the SoyLei/Tempo deposit preview failed as intended because the preview carries warnings.
- Live dry-run with `--allow-warnings` wrote `/home/ecochran76/.transcripts/artifacts/22/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-apply.json`.
- The live dry-run result has `mode=preview`, six planned candidates, warnings carried forward, and planned `graphiti-runtime benchmark-write` commands.
- No live Graphiti memory write was executed in this slice.

Next:

- Review the six planned SoyLei/Tempo memory candidates and, if acceptable, run one approved single-candidate Graphiti apply with queue/readback verification.

## Turn 44 | 2026-05-12

Summary: Applied one reviewed SoyLei/Tempo memory candidate to Graphiti.

Action:

- Reviewed the six structured memory candidates from the SoyLei/Tempo deposition preview.
- Selected only candidate `3a80941071fe9036` for the first live smoke because it captures durable relationship/matter context without pricing details.
- Ran a duplicate-oriented Graphiti discovery preflight for the Tempo/SoyLei matter query; it returned only existing repo/project memories and no existing Tempo/SoyLei matter episode.
- Applied the selected candidate with `memory_harvest_apply.py --apply --approval-token APPROVE_GRAPHITI_MEMORY_HARVEST --allow-warnings`.
- Wrote the live audit artifact to `/home/ecochran76/.local/state/transcribe-audio/memory-harvest-runs/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-apply.json`.

Validation:

- Graphiti `doctor` was degraded only because Inspector API was down; MCP HTTP and FalkorDB were healthy.
- `graphiti-runtime queue` was empty/unlocked before the apply.
- The live apply returned `episode_uuid=a39add00-d8d7-4d76-a1f9-855802ffb680` in group `transcribe_audio_main`.
- Post-apply `graphiti-runtime queue` returned empty/unlocked with the transcribe-audio worker running.
- Post-apply `graphiti-runtime discover --group-id transcribe_audio_main` found the new episode plus extracted facts and nodes for Tempo Chemical, SoyLei SIP 1132 concentrate, and SoyLei 1119 emulsion.

Next:

- Add an operator review workflow that can approve or reject individual memory candidates from a preview artifact, then batch-apply only accepted candidates with duplicate checks and per-candidate audit status.

## Turn 45 | 2026-05-12

Summary: Added per-candidate memory-harvest review files and duplicate audit status.

Action:

- Added `memory_harvest_apply.py --init-review` to create `*.memory-harvest-review.json` templates from deposition previews.
- Added `--review-file` support so only `approved` candidates are eligible for live Graphiti writes.
- Recorded `rejected`, `pending`, and missing-review candidates as non-written candidate statuses in the apply audit.
- Added default per-candidate Graphiti discovery duplicate preflights during `--apply`.
- Exact same-candidate replays are skipped with status `duplicate_skipped`; possible duplicate metadata is retained under each candidate's `duplicate_check`.
- Failed duplicate preflights stop that candidate with `duplicate_check_failed` before any write attempt.
- Extended `AppliedMemoryHarvestCandidate` with review decision/reason and duplicate-check metadata.
- Updated README, ROADMAP, and the P05 plan.

Validation:

- Graphiti `doctor` was degraded only because Inspector API was down; MCP HTTP and FalkorDB were healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main` found the prior live SoyLei/Tempo memory episode, confirming the duplicate-check surface is visible to Graphiti reads.
- `.venv/bin/python -m pytest tests/test_memory_harvest_apply.py -q` passed with 9 tests.
- `.venv/bin/python -m py_compile memory_harvest_apply.py deposition_artifacts.py tests/test_memory_harvest_apply.py` passed.
- `.venv/bin/python -m pytest -q` passed with 87 tests.
- Live no-write `--init-review` over the SoyLei/Tempo deposition preview wrote `/home/ecochran76/.local/state/transcribe-audio/memory-harvest-runs/22739745e0ee248ed0e2-2026-05-06 13-15 Soylei and Tempo Chemical Technical discussion My recording 116 Transcript.memory-harvest-review.json`.
- The live review template contains six candidates, all initialized to `pending`.

Next:

- Review the generated SoyLei/Tempo memory-harvest review file, approve/reject candidates, then run a batch apply for approved non-duplicate candidates.

## Turn 46 | 2026-05-12

Summary: Planned the React + Vite transcript review console.

Action:

- Inspected repo planning, runtime/tenant, architecture, and memory-routing policies before changing the roadmap.
- Ran Graphiti discovery for existing frontend/review-console context in `transcribe_audio_main`.
- Reviewed `../previews` access-control and review-sharing docs for single-operator login, scoped share links, and feedback semantics.
- Reviewed `../buffer-cli/frontend` for the sticky navbar, animated left pane, central table viewport, and animated right inspector pane pattern.
- Added `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md`.
- Added P09 to `ROADMAP.md`.

Validation:

- Graphiti `doctor` was healthy.
- Graphiti discovery returned existing repo/project memory and no conflicting frontend plan.
- The P09 plan includes the required navbar, left pane, central viewport, and right pane responsibilities.
- The plan keeps blobs, share tokens, tenant credentials, and live runtime state under `~/.transcripts` or `~/.local/state/transcribe-audio/`.

Next:

- Start P09 slice 1 by adding the backend read API contract and minimal local API service for library/search/detail plus blob range playback before scaffolding the React shell.

## Turn 47 | 2026-05-12

Summary: Added the first local transcript review API and blob playback contract.

Action:

- Added blob storage tables to the transcript store: `blobs` and `document_blobs`.
- Updated transcript ingestion to copy existing source recordings into `~/.transcripts/blobs/` when the transcript artifact points to an available source/working media file.
- Added compact `media_blob` metadata with playback/download URLs for frontend use.
- Added `transcript_api.py`, a read-only local HTTP API over the user-scoped store.
- Implemented `/api/health`, `/api/library`, `/api/search`, `/api/documents/<id>`, `/api/documents/<id>/context`, and range-capable `/api/blobs/<blob_id>`.
- Added `docs/dev/transcript-review-api.md` and updated README, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with 3 tests.
- `.venv/bin/python -m py_compile transcript_api.py transcript_store.py tests/test_transcript_api.py` passed.
- `.venv/bin/python -m pytest -q` passed with 90 tests.
- Read-only live-store smoke through `transcript_api.list_documents(root=~/.transcripts, limit=5)` returned 14 total documents and 5 listed documents.
- The live-store smoke found no media blobs among the first 5 existing documents because older rows predate blob registration; re-backfill/update is needed to populate blob links for already-ingested transcripts.

Next:

- Add a safe store migration/backfill path that updates existing transcript rows with blob pointers, then run a dry-run/apply over recent transcripts before scaffolding the React shell.

## Turn 48 | 2026-05-12

Summary: Added dry-run-first legacy transcript import for historical transcripts without sidecars.

Action:

- Added `legacy_transcript_import.py`.
- The importer discovers legacy `*Transcript.txt` and `*Transcript.docx` outputs.
- Dry-run is the default; `--apply` is required to write synthesized sidecars and ingest them.
- Synthesized sidecars are written under `~/.transcripts/legacy-artifacts/`, not the repo or source folders.
- Legacy sidecars use the normal transcript artifact shape, set `backend=legacy-import`, preserve the source TXT/DOCX path under `output_paths`, and mark `legacy_import.needs_enrichment=true`.
- The importer attempts to match nearby source recordings by basename and passes matched media paths into the normal blob registration path.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 27 tests.
- `.venv/bin/python -m py_compile legacy_transcript_import.py transcript_api.py transcript_store.py tests/test_legacy_transcript_import.py` passed.
- `.venv/bin/python -m pytest -q` passed with 93 tests.
- A bounded `find` check over `~/Downloads` showed it is a symlink to `/mnt/c/Users/ecoch/Downloads` and found zero top-level legacy transcript candidates matching the default patterns.
- A Python dry-run against `~/Downloads` was killed after slow mount behavior; no writes were made.

Next:

- Point `legacy_transcript_import.py` at the real historical transcript root with `--recursive` and run a dry-run inventory, then apply the import with production embeddings and run the context/readout enrichment pipeline over rows marked `legacy_import.needs_enrichment=true`.

## Turn 49 | 2026-05-12

Summary: Imported historical transcript DOCX/TXT outputs from Downloads and Sound Recordings.

Action:

- Counted 45 legacy transcript candidates and 262 possible media files under `~/Downloads` and `/mnt/h/My Drive/Documents/Sound Recordings`.
- Added `--media-index-file` to `legacy_transcript_import.py` after direct Python media-root walks proved too slow on mounted folders.
- Generated transcript and media indexes with `find`, then ran a dry-run inventory from exact candidate paths.
- Saved dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-import-dry-run-2026-05-12.json`.
- Applied the import with `--embedding-provider ollama --embedding-model ollama/nomic-embed-text`.
- Saved apply output to `/home/ecochran76/.local/state/transcribe-audio/legacy-import-apply-2026-05-12.json`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Dry-run found 45 candidates, all `convert`, with 44 media matches.
- The only unmatched media item was `2025-07-09 Lululemon Summary and Transcript.docx`.
- Apply inserted 45 legacy transcript sidecars under `~/.transcripts/legacy-artifacts/`.
- Store verification showed 54 total transcript documents using Ollama embeddings, 45 documents marked `legacy_import.needs_enrichment=true`, 44 legacy documents linked to blobs, and 36 total blobs.
- Read-only API smoke listed 54 transcript documents.
- `transcript_store.py search "Scott Roberts" --kind transcript --limit 3` returned legacy transcript hits from the imported set.
- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 28 tests.
- `.venv/bin/python -m pytest -q` passed with 94 tests.

Next:

- Add an enrichment queue/list command for documents marked `legacy_import.needs_enrichment=true`, then run first-pass summaries and calendar/context enrichment in small batches.

## Turn 50 | 2026-05-12

Summary: Searched SoyLei Shared Drives and imported deduped additional legacy transcripts.

Action:

- Confirmed the prior `Sound Recordings` search was recursive and included subfolders such as `Transcribed/`.
- Searched these additional roots:
  - `/mnt/h/Shared drives/SoyLei Officers`
  - `/mnt/h/Shared drives/SoyLei Core Team`
  - `/mnt/h/.shortcut-targets-by-id/0B1xe-E5-InccUThWQ0QyY0Z0Tms/Documents/Corkboard/Clients/SoyLei`
- Found 65 additional transcript candidates in those SoyLei Shared Drive/shortcut roots.
- Added de-dupe behavior to `legacy_transcript_import.py` using existing source transcript hashes and normalized titles, plus within-batch duplicate detection.
- Added `--no-dedupe` for diagnostics and `--no-media-match` for fast mounted-drive transcript-only imports.
- Ran a Shared Drive dry-run with de-dupe enabled and media matching disabled.
- Saved dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-shared-drives-dry-run-2026-05-12.json`.
- Applied the deduped Shared Drive import with Ollama/Nomic embeddings and no media matching.
- Saved apply output to `/home/ecochran76/.local/state/transcribe-audio/legacy-shared-drives-apply-2026-05-12.json`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Shared Drive dry-run selected 65 candidates: 25 `convert`, 22 `duplicate_in_batch`, and 18 `duplicate_existing`.
- Shared Drive apply inserted 25 new transcripts and skipped 40 duplicates.
- Store verification showed 79 total transcript documents using Ollama embeddings, 70 documents marked `legacy_import.needs_enrichment=true`, and 36 total blobs.
- Duplicate source-hash verification returned no duplicate legacy source hashes.
- `transcript_store.py search "Ryan Jaggar" --kind transcript --limit 3` returned imported Shared Drive legacy hits.
- `.venv/bin/python -m pytest tests/test_legacy_transcript_import.py tests/test_transcript_api.py tests/test_transcript_store.py -q` passed with 31 tests.
- `.venv/bin/python -m pytest -q` passed with 97 tests.

Next:

- Add a targeted blob-linking pass for Shared Drive legacy transcripts that need media matching, then add the enrichment queue/list command for the 70 legacy rows marked `needs_enrichment`.

## Turn 51 | 2026-05-12

Summary: Added the legacy enrichment queue and targeted media-linking pass.

Action:

- Added `transcript_store.py legacy-enrichment-queue` to list legacy rows marked `legacy_import.needs_enrichment=true`.
- Queue output supports text, JSON, compact JSON, and runnable `summarize_transcript.py` commands.
- Queue entries de-dupe same-hash and same-title rows by default to avoid duplicate provider calls.
- Added `legacy_media_link.py` to link already-imported legacy transcript sidecars to recordings from explicit media indexes or targeted media roots.
- Built a targeted SoyLei Shared Drive media index at `/home/ecochran76/.local/state/transcribe-audio/soylei-shared-media-index-2026-05-12.txt`.
- Saved media-link dry-run output to `/home/ecochran76/.local/state/transcribe-audio/legacy-media-link-dry-run-2026-05-12.json`.
- Applied matched media links and saved output to `/home/ecochran76/.local/state/transcribe-audio/legacy-media-link-apply-2026-05-12.json`.
- Attempted one first-pass enrichment smoke with the default OpenAI-compatible config and one with `/home/ecochran76/.auracall/api.env`.
- Updated README, ROADMAP, and the P09 plan.

Validation:

- Targeted media index found 312 media files under the named SoyLei Shared Drive/shortcut roots.
- Media-link dry-run selected 26 unlinked legacy rows: 16 `link` and 10 `no_match`.
- Media-link apply updated 16 linked rows and skipped the 10 unmatched rows.
- Store verification showed 79 transcript documents, 70 legacy rows still marked `needs_enrichment`, and 60 transcript documents linked to source-recording blobs.
- The de-duped enrichment queue showed 68 pending first-pass readouts, 58 with blobs and 10 without blobs.
- Default OpenAI-compatible enrichment failed with `429 insufficient_quota`.
- AuraCall-compatible enrichment reached `127.0.0.1:18095` but timed out after 120 seconds; `summarize_transcript.py` now reports provider request failures as clean `TranscriptionError` messages instead of stack traces.
- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_legacy_media_link.py tests/test_transcript_store.py -q` passed with 38 tests.
- `.venv/bin/python -m pytest tests/test_legacy_media_link.py tests/test_transcript_store.py tests/test_legacy_transcript_import.py -q` passed with 32 tests.

Next:

- Repair or tune the AuraCall/OpenAI-compatible readout path for long legacy transcripts, then run a small enrichment batch from `transcript_store.py legacy-enrichment-queue --format commands`.

## Turn 52 | 2026-05-12

Summary: Proved the repaired AuraCall OpenAI-compatible path on one legacy enrichment smoke and ingested the readout.

Action:

- Loaded `/home/ecochran76/.auracall/api.env` and reran `summarize_transcript.py`
  on the prior failed Scott/gener8or legacy transcript using the AuraCall
  OpenAI-compatible endpoint.
- Used `--timeout 300` because browser-backed AuraCall requests can exceed the
  old 120 second client timeout under load.
- Wrote non-mutating smoke outputs under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/`.
- Ingested the generated readout JSON directly through
  `transcript_store.ingest_artifact` to avoid a duplicate provider call.
- Updated
  `docs/dev/notes/2026-05-12-auracall-legacy-enrichment-handoff.md` so the
  handoff no longer describes the AuraCall failure as current.

Validation:

- The source smoke transcript exists at
  `/home/ecochran76/.transcripts/legacy-artifacts/07/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.transcript.json`.
- The source artifact is 15,415 bytes with 14,237 transcript text characters.
- `summarize_transcript.py` returned successfully and wrote:
  - `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.json`
  - `/home/ecochran76/.local/state/transcribe-audio/auracall-smokes/0711bf42d7771e63b44d-2025-07-28 Scott gener8or imPETus SABER.readout.md`
- The readout contained a non-empty summary plus participants, topics, action
  items, matter candidates, memory candidates, and next steps.
- Store ingest inserted readout id `017a8ffe7173998ba82d`.
- `transcript_store.py legacy-enrichment-queue --format compact-json` now
  reports 67 pending de-duped first-pass readouts, and the Scott/gener8or item
  is no longer pending.

Next:

- Run a bounded AuraCall-backed small batch from
  `transcript_store.py legacy-enrichment-queue --format commands --limit 3`
  before expanding to the remaining legacy queue.

## Turn 53 | 2026-05-12

Summary: Ran a bounded three-item AuraCall enrichment batch; two succeeded and one failed with provider error content.

Action:

- Selected three pending legacy transcript rows from `transcript_store.py legacy-enrichment-queue --format compact-json --provider openai-compatible --limit 3`.
- Saved the selected queue to `/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-queue-2026-05-12.json`.
- Loaded `/home/ecochran76/.auracall/api.env` and ran `summarize_transcript.py --provider openai-compatible --timeout 300 --store` sequentially for the selected items.
- Captured per-item logs and summary under `/home/ecochran76/.local/state/transcribe-audio/legacy-enrichment-batch-3-retry-2026-05-12/`.
- Made a raw diagnostic chat-completions call for the failed item and saved the response body/content in the same batch directory.
- Updated `docs/dev/notes/2026-05-12-auracall-legacy-enrichment-handoff.md`.

Validation:

- Item 1, `20250417-142659-Ambient Workshop Recording (2025-04-17) - Non-Verbal Audio`, succeeded and stored a readout.
- Item 2, `2025-07-17 Shuana Sofia MacGill`, succeeded and stored a readout.
- Item 3, `2025-06-06 Breakfast with Nacu My recording 9`, failed twice with `OpenAI-compatible readout did not return valid JSON`.
- The raw diagnostic response for item 3 was HTTP 200 but the assistant content was provider error text: `Something went wrong. If this issue persists please contact us through our help center at help.openai.com.`
- Store verification showed 10 readout documents after the partial batch.
- The de-duped pending legacy enrichment queue reported 63 items after the partial batch.

Next:

- Do not run the full queue blindly. Either skip/quarantine the failed Nacu breakfast item and continue with a small batch of later queue entries, or add a provider fallback/condensation path for transcripts that trigger AuraCall internal-error content.

## Turn 54 | 2026-05-12

Summary: Tested a transcript excerpt-budget workaround, then identified it as
the wrong layer. Kept readout-shape validation and moved the real fix back to
AuraCall.

Action:

- Started the next five-item AuraCall-backed legacy enrichment batch.
- Tried and reverted a transcript-length limiting workaround. This repo should
  send the full transcript; AuraCall should handle large OpenAI-compatible
  browser-backed requests without reducing caller capability.
- Added OpenAI-compatible readout-shape validation so echoed prompt/input JSON
  cannot be stored as an empty `Transcript Readout`.
- Removed four bad empty readout rows created during an initial 90k-budget
  attempt: `afcd2217031c97899dcd`, `a00e322b974f8e424548`,
  `cd57df8c9935867d05ed`, and `79e84e2d09d64e6a1e6c`.
- Removed the stale local empty JSON/Markdown output for the still-pending SBIR
  item.

Validation from the workaround experiment:

- `2025-06-06 Breakfast with Nacu My recording 9` produced a valid readout.
- `2025-07-31 Nacu Breakfast My recording 17` produced a valid readout.
- `2025-04-24 Nacu Meeting USDA Grant and SoyLei Matters` produced a valid
  readout.
- `2025-07-29 Dr Warmbe Meniscus Tear consult` produced a valid readout.
- `2025-04-17 Nacu Eric Call SoyLei SBIR Matters` remains pending: one retry
  returned an empty response, and the next retry hit the 300 second client
  timeout. AuraCall reported no recent stuck runtime runs afterward.
- The pending de-duped queue now reports 59 items.
- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_transcript_store.py -q`
  passed with 38 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py transcript_store.py tests/test_readouts.py tests/test_transcript_store.py`
  passed.
- `.venv/bin/python -m pytest -q` passed with 105 tests.

Next:

- Fix AuraCall so a large full-transcript OpenAI-compatible request is
  transported through the browser service as an attachment when needed, and so
  failed AuraCall runs return an API error instead of HTTP 200 with empty
  assistant content. Then retry the SBIR item with the full transcript.

## Turn 55 | 2026-05-12

Summary: Removed the transcript-length workaround from this repo, verified the
fix belongs in AuraCall, and completed the pending SBIR readout with the full
transcript.

Action:

- Removed the `--max-transcript-chars` workaround from the readout CLI,
  generated queue commands, and tests.
- Kept readout-shape validation so malformed/empty AuraCall responses do not
  get stored as readouts.
- Updated the AuraCall handoff note to state that transcript truncation was a
  reverted experiment, not the path forward.
- Rebuilt/reinstalled AuraCall and retried the pending SBIR readout without
  transcript truncation.

Validation:

- `.venv/bin/python -m pytest tests/test_readouts.py tests/test_transcript_store.py -q`
  passed with 37 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py transcript_store.py tests/test_readouts.py tests/test_transcript_store.py`
  passed.
- `.venv/bin/python -m pytest -q` passed with 104 tests.
- Full-transcript SBIR retry through `agent:instant-chatgpt-soylei` failed
  honestly with AuraCall HTTP 502 after AuraCall detected non-parseable JSON.
- Full-transcript SBIR retry through `agent:pro-extended-chatgpt-soylei`
  succeeded and wrote:
  `/home/ecochran76/.transcripts/legacy-artifacts/28/28d268e46f590765c413-2025-04-17 Nacu Eric Call SoyLei SBIR Matters.readout.json`
- The de-duped pending legacy enrichment queue now reports 58 items, and the
  SBIR item is no longer pending.

Next:

- Continue legacy enrichment in bounded batches using the full transcript path.
  Prefer stronger/project-specific AuraCall agents for long readout jobs when
  JSON completeness matters.

## Turn 56 | 2026-05-13

Summary: Added an AuraCall response-batch path for legacy transcript readouts
using a project-bound SoyLei Pro Extended transcripts agent.

Action:

- Added `scripts/auracall_legacy_enrichment_batch.py`.
- Added `write_readout_from_payload` so synchronous and batched readouts share
  the same JSON/Markdown materialization path.
- Added a dry-run test that verifies the batch payload uses
  `agent:pro-extended-chatgpt-soylei-transcripts`, JSON response-format
  metadata, and the SoyLei `wsl-chrome-3` runtime hints.
- Created registry agent `pro-extended-chatgpt-soylei-transcripts` with
  `projectName=Transcripts`, `service=chatgpt`,
  `runtimeProfile=wsl-chrome-3`, and `modelSelector=chatgpt:pro-extended`.
- Issued scoped client env:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- Restarted `auracall-api.service` and confirmed `/v1/models` includes
  `agent:pro-extended-chatgpt-soylei-transcripts`.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_legacy_enrichment_batch_dry_run_writes_manifest tests/test_transcript_store.py::test_legacy_enrichment_queue_lists_pending_legacy_imports -q`
  passed with 2 tests.
- `.venv/bin/python -m py_compile summarize_transcript.py scripts/auracall_legacy_enrichment_batch.py tests/test_transcript_store.py`
  passed.
- Live dry-run built:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-062622.json`.
- Live one-item enqueue created and completed
  `batch_0db1883c7905471c83d807411cfdee33` with
  `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`, and child
  response `resp_1a4b0915303848a6ab68a48e286e563f`.
- `status --materialize --store` wrote and ingested:
  `/home/ecochran76/.transcripts/legacy-artifacts/29/29ed3d64cca92a7cf5f5-2025-08-15 Dr Stefl Knee Replacement Consult.readout.json`.
- The de-duped pending legacy enrichment queue now reports 57 items.

Note:

- The earlier `POST /v1/projects/ensure` `button-missing` failure was repaired
  in AuraCall on 2026-05-13. The ChatGPT provider project now exists as
  `g-p-6a04628762ac8191894b16cfaddfd126`, and the transcript agent is bound to
  that provider project id.

## Turn 57 | 2026-05-13

Summary: Revalidated the AuraCall scoped client path for transcript readout
bursts after returning from live-follow work.

Validation:

- Scoped env:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env`.
- The scoped key can read `/v1/models` and sees
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- The running AuraCall registry shows
  `pro-extended-chatgpt-soylei-transcripts` bound to ChatGPT project
  `Transcripts` with provider project id
  `g-p-6a04628762ac8191894b16cfaddfd126`.
- Live scoped-env smoke passed:
  `pnpm run smoke:scoped-client-env -- /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env --prompt 'Reply exactly: auracall transcribe env ok' --expect-output 'auracall transcribe env ok' --timeout-ms 180000`.
- Response id `resp_45008e83347940909bcdba697b91fa2c` read back as
  `completed` with output `auracall transcribe env ok`.

Next:

- Resume bounded legacy readout batches through
  `scripts/auracall_legacy_enrichment_batch.py`.
- Keep concurrency and browser interaction limits in the AuraCall batch request;
  do not limit transcript length in this repo.

## Turn 58 | 2026-05-13

Summary: Added a repo-local handoff note so `transcribe-audio` can retake
ownership of the AuraCall-backed legacy readout batch workflow.

Action:

- Added
  `docs/dev/notes/2026-05-13-auracall-transcribe-ownership-handoff.md`.
- Recorded the current AuraCall transcript agent binding, scoped env path,
  live smoke evidence, one-item batch evidence, and next owner actions.
- Reiterated the policy boundary: Transcribe Audio owns queue selection,
  prompt construction, materialization, and store ingestion; AuraCall owns
  large prompt transport, browser execution, project binding, queueing, and
  rate limiting.

Validation:

- `graphiti-runtime doctor` reported healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main "AuraCall transcript readout batch scoped env next steps" --max-facts 5`
  returned older P03/readout context; the new handoff therefore relies on
  current repo runbook entries and live AuraCall evidence.
- `git diff --check` passed.

Next:

- Resume with the handoff note's three-item dry run, then one three-item live
  batch, then `status --materialize --store`.

## Turn 59 | 2026-05-13

Summary: Started the first three-item AuraCall response batch under
`transcribe-audio` ownership; batch remains in progress.

Action:

- Read `docs/dev/notes/2026-05-13-auracall-transcribe-ownership-handoff.md`
  and followed its owner actions.
- Ran the three-item dry run:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env enqueue --limit 3 --store --dry-run`.
- Dry-run manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092116.json`.
- Inspected the dry-run manifest and confirmed the expected model,
  JSON response-format metadata, full prompt payloads, and limits.
- Ran the first live three-item batch:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env enqueue --limit 3 --store --max-concurrent-runs 2 --max-browser-interactions-per-minute 8`.
- Live manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`.
- Polled with `status --materialize --store` several times; no children were
  complete yet, so no readouts were materialized.

Validation:

- Dry run selected 3 requests for
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- Dry-run prompt lengths were approximately 137447, 33426, and 12107
  characters; transcript payloads were not truncated.
- Dry-run limits were `maxConcurrentRuns=2` and
  `maxBrowserInteractionsPerMinute=8`.
- Live enqueue returned batch id `batch_bd9a400d785f4eeeaecf986621597091`.
- Current batch status is `running` with counts:
  `total=3`, `in_progress=3`, `completed=0`, `failed=0`, `cancelled=0`,
  `missing=0`.
- Child response ids:
  `resp_ad243a3df5bc4d61ac7934e144f4352b`,
  `resp_b35c7e03a57d4d11ad3d081d77277404`,
  `resp_9d59ac43f87f460081a187fa28c4bf49`.

Next:

- Re-run:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env status /home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json --materialize --store`.
- If children complete, verify readout artifacts and the pending queue count.
- If children fail or remain stuck, preserve the manifest and response ids and
  diagnose AuraCall rather than shortening transcripts in this repo.

## Turn 60 | 2026-05-13

Summary: Polled the first three-item AuraCall batch; it is not materializable
because one completed response has empty output, one child is still running,
and one child failed in AuraCall.

Action:

- Re-ran `status --materialize --store` for
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`.
- Materialization failed with `OpenAI-compatible readout returned an empty response`.
- Re-ran `status` without materialization to capture current batch state.
- Saved raw response diagnostics under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135-diagnostics/`.
- Re-checked the de-duped pending queue.

Validation:

- Batch id remains `batch_bd9a400d785f4eeeaecf986621597091`.
- Current counts are `total=3`, `completed=1`, `in_progress=1`,
  `failed=1`, `cancelled=0`, `missing=0`.
- Index 0, `resp_ad243a3df5bc4d61ac7934e144f4352b`, is marked completed by
  AuraCall but `/v1/responses/...` returns `output: []`, so there is no
  readout JSON to materialize.
- Index 1, `resp_b35c7e03a57d4d11ad3d081d77277404`, is still `in_progress`.
- Index 2, `resp_9d59ac43f87f460081a187fa28c4bf49`, failed with
  `runner_execution_failed: connect ETIMEDOUT 127.0.0.1:9222`.
- No readouts were materialized from this batch.
- The de-duped pending legacy enrichment queue still reports 57 items.

Next:

- Diagnose this as an AuraCall/runtime issue, not a transcript truncation issue:
  completed-empty output and `127.0.0.1:9222` timeout should be repaired or
  retried in AuraCall.
- After AuraCall-side diagnosis, retry a fresh bounded batch or add
  transcribe-side handling that skips failed/empty children while preserving
  their response ids for retry.

## Turn 61 | 2026-05-14

Summary: Retried the AuraCall batch path after the AuraCall upgrade; the old
batch is terminally failed and a fresh three-item retry batch is in progress.

Action:

- Re-polled old manifest
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260513-092135.json`
  with `status --materialize --store`.
- Materialization still failed with `OpenAI-compatible readout returned an
  empty response`.
- Re-polled the old manifest without materialization and confirmed its terminal
  status.
- Re-checked the de-duped queue; it still reported 57 pending legacy readout
  items.
- Submitted a fresh three-item live batch using the same full transcript
  payloads, model, and limits.
- Polled the fresh batch twice; no children completed or failed yet.

Validation:

- Old batch `batch_bd9a400d785f4eeeaecf986621597091` is now `failed` with
  counts `total=3`, `completed=1`, `failed=2`, `in_progress=0`.
- Old index 0 `resp_ad243a3df5bc4d61ac7934e144f4352b` is completed but still
  has empty output, so no readout can be materialized.
- Old index 1 `resp_b35c7e03a57d4d11ad3d081d77277404` failed with
  `runner_execution_failed: ChatGPT response did not complete as a parseable
  JSON object.`
- Old index 2 `resp_9d59ac43f87f460081a187fa28c4bf49` failed with
  `runner_execution_failed: connect ETIMEDOUT 127.0.0.1:9222`.
- Fresh retry manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json`.
- Fresh retry batch id: `batch_e9b79b1474ec4cf8a622e52f5b8f7bce`.
- Fresh retry child response ids:
  `resp_56c5a0d25823456d99d97e50114fe887`,
  `resp_c073d5e002414a0c98f8ee0fe987470b`,
  `resp_618693902f244b8e8a777cff9fc38305`.
- Fresh retry status after several minutes remained `running` with counts
  `total=3`, `in_progress=3`, `completed=0`, `failed=0`, `cancelled=0`,
  `missing=0`.
- No readouts were materialized in this turn.

Next:

- Poll the fresh retry manifest with:
  `.venv/bin/python scripts/auracall_legacy_enrichment_batch.py --env-file /home/ecochran76/.local/state/transcribe-audio/auracall-transcripts.env status /home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json --materialize --store`.
- If it completes, verify stored readouts and queue count.
- If it fails or remains stuck for an unreasonable interval, diagnose AuraCall
  with the fresh batch and child response ids rather than changing transcript
  length.

## Turn 62 | 2026-05-14

Summary: Polled the fresh AuraCall retry batch; all three children failed with
partial JSON snapshots but no materializable response output.

Action:

- Re-polled fresh retry manifest
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528.json`
  with `status --materialize --store`.
- Materialization produced no readouts because the batch ended failed.
- Saved raw response diagnostics under
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-151528-diagnostics/`.
- Re-checked the de-duped legacy enrichment queue.

Validation:

- Fresh retry batch `batch_e9b79b1474ec4cf8a622e52f5b8f7bce` is now failed
  with counts `total=3`, `completed=0`, `failed=3`, `in_progress=0`.
- Index 0 `resp_56c5a0d25823456d99d97e50114fe887` failed with
  `ChatGPT response did not complete as a parseable JSON object after waiting`;
  AuraCall captured a best snapshot of 22322 chars but `/v1/responses/...`
  still returned `output: []`.
- Index 1 `resp_c073d5e002414a0c98f8ee0fe987470b` failed with the same
  parseable-JSON completion issue and a best snapshot of 11918 chars; response
  output was empty.
- Index 2 `resp_618693902f244b8e8a777cff9fc38305` failed with the same issue
  and a best snapshot of 10358 chars; response output was empty.
- The de-duped pending legacy enrichment queue still reports 57 items.
- No transcript payloads were shortened and no readouts were stored from this
  retry batch.

Next:

- AuraCall should expose failed-run best snapshots as retrievable diagnostics or
  recoverable output artifacts, or complete JSON capture before marking the run
  failed.
- Transcribe-side next work can add retry/quarantine metadata around failed
  batch children, but should not treat partial snapshots as readouts unless
  AuraCall exposes a deliberate recovery contract.

## Turn 63 | 2026-05-14

Summary: Started a one-item retry against AuraCall's new recovery-artifact
contract; the run remains active with no output yet.

Action:

- Read AuraCall handoff
  `/home/ecochran76/workspace.local/auracall/docs/dev/notes/2026-05-14-chatgpt-json-artifact-handoff.md`.
- Confirmed the AuraCall fix is non-retroactive for the failed
  `batch_e9b79b1474ec4cf8a622e52f5b8f7bce`, so a fresh run is required.
- Enqueued a fresh one-item batch from the current pending queue using
  `agent:pro-extended-chatgpt-soylei-transcripts`, `maxConcurrentRuns=1`, and
  `maxBrowserInteractionsPerMinute=6`.
- Polled the manifest with `status --materialize --store` several times.
- Read the response object directly and saved a raw diagnostic snapshot.

Validation:

- Manifest:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-171322.json`.
- Batch id: `batch_0973e70d5a1e4fa5a7f8f4c2ae7d1668`.
- Response id: `resp_723b789f244446159354a2e751dde7a0`.
- Selected transcript title:
  `2025-08-20 Nacu Eric Line of Business follow up meeting`.
- Batch status remains `running` with counts `total=1`, `in_progress=1`,
  `completed=0`, `failed=0`.
- Direct response read showed `status=in_progress`, `output_len=0`,
  `terminalStepId=null`, and a running step for
  `pro-extended-chatgpt-soylei-transcripts` on `wsl-chrome-3`.
- Response `lastUpdatedAt` was `2026-05-14T22:20:50.377Z`, proving the run was
  still active during this turn.
- Diagnostics path:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-171322-diagnostics/`.
- Queue still reports 57 pending items because no readout was materialized yet.

Next:

- Poll the one-item manifest again with `status --materialize --store`.
- If it completes with message JSON or an artifact output, materialize/store and
  verify the pending queue decreases.
- If it fails, inspect `/v1/responses/resp_723b789f244446159354a2e751dde7a0`
  for the new recovery artifact contract before changing transcribe prompts.

## Turn 64 | 2026-05-14

Summary: Exercised AuraCall recovery/output contracts and updated the batch
client to preserve full inline JSON while accepting future JSON artifact outputs.

Action:

- Polled the prior one-item recovery run and inspected its partial recovery
  artifact.
- Updated `scripts/auracall_legacy_enrichment_batch.py` so AuraCall requests no
  longer use `metadata.response_format`, because browser-backed ChatGPT runs do
  not reliably complete through that JSON-object path.
- Added `response_model_payload()` materialization support for both inline
  message JSON and future JSON artifact outputs.
- Tried the ChatGPT workspace-file contract by asking for `legacy_readout.json`.
- Observed that AuraCall completed the run with only
  `legacy_readout.json ready` in `/v1/responses/{id}` and no artifact entries in
  local `sharedState.artifacts`.
- Tried full inline JSON without length limits; AuraCall returned JSON-like text
  but with raw newlines, malformed nested sections, or truncation, so
  materialization correctly rejected it.
- A short capped prompt did materialize one readout successfully, but the cap was
  removed because full-fidelity readouts should not be product-limited just to
  work around provider transport.

Validation:

- Tests: `.venv/bin/python -m pytest tests/test_transcript_store.py
  tests/test_readouts.py -q` passed with 39 tests.
- Whitespace: `git diff --check` passed.
- Workspace artifact trial:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-175431.json`.
- Inline full-output trials:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-175902.json`
  and
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-180139.json`.
- Successful capped-output materialization:
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-180342.json`,
  response `resp_19865f8e9d7046d68b523ea440a5a9be`, readout stored at
  `/home/ecochran76/.transcripts/legacy-artifacts/63/63eb9090a1dcc8e9a332-2025-08-20 Nacu Eric Line of Business follow up meeting.readout.json`.
- Current code intentionally does not keep the capped prompt; it preserves full
  detail and validates before storing.

Next:

- Fix AuraCall to expose ChatGPT workspace/file artifacts as response outputs, or
  add a durable attachment/output channel for large structured JSON.
- After AuraCall exposes the full readout artifact, retry one uncapped item and
  then resume the legacy enrichment batch.

## Turn 65 | 2026-05-14

Summary: Restored the legacy enrichment request contract to artifact-first
ChatGPT workspace output.

Action:

- Updated `scripts/auracall_legacy_enrichment_batch.py` so the SoyLei
  Transcripts AuraCall batch request instructs ChatGPT to create
  `legacy_readout.json` in its REPL/workspace and surface it as a downloadable
  artifact.
- Removed the inline JSON requirement from the AuraCall-specific prompt; the
  assistant response must now surface the actual `legacy_readout.json`
  downloadable artifact/link rather than a text-only readiness marker.
- Changed `metadata.outputContract.mode` from
  `inline_json_with_optional_workspace_artifact` to
  `chatgpt_workspace_artifact`.
- Kept `response_model_payload()` able to parse artifact outputs first after a
  non-JSON readiness message, while still tolerating parseable inline JSON for
  backward compatibility.

Validation:

- AuraCall-side live smoke `resp_db52dcf73b7d44b0abbffd327bbeac5c` now proves
  the browser run lands inside the SoyLei `Transcripts` project URL, but it
  still recorded `discovered=0 materialized=0` for the requested artifact.
- This transcribe-side change does not claim artifact extraction is fixed; it
  aligns the caller with the intended artifact contract so the next smoke tests
  the right behavior.
- Correction: the first artifact-first prompt still allowed a text-only
  readiness response. The prompt now explicitly says a text-only readiness note
  is not sufficient.
- Live retry `resp_6b10a6d743e84ec3a775060fda94b120` reached the SoyLei
  `Transcripts` project but ChatGPT returned the future-tense status sentence
  `I'll create the JSON readout...` with no artifact. The prompt was tightened
  again to forbid future-tense/status replies and require an actual
  `sandbox:/.../legacy_readout.json` link or native attachment card in the final
  response.
- Second retry `resp_48647ca8e7bc42979f89f20dd4778dee` produced a sandbox
  artifact link and AuraCall recorded `discovered=1 materialized=1`, but the
  response artifact was still exposed as `artifact_type=generated` without a
  local path, so the transcribe materializer rejected it.
- AuraCall was patched and reinstalled to preserve the materialized local path
  and JSON MIME metadata on response artifacts. The transcribe materializer was
  patched to accept generated/download JSON artifacts and prefer
  `metadata.localPath` over sandbox URIs.
- Final smoke `batch_ca0a6f46ed1844c0a789329bcc241053` /
  `resp_4235722877774ee79e158be3843de653` completed and materialized:
  - `/home/ecochran76/.transcripts/legacy-artifacts/5d/5d26c585ac566dc22c0d-2025-08-21 Lululemon JP Siddhant Xlinked HBAN  My recording 20.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/5d/5d26c585ac566dc22c0d-2025-08-21 Lululemon JP Siddhant Xlinked HBAN  My recording 20.readout.md`
  - API artifact evidence included `artifact_type=file`,
    `mime_type=application/json`, `disposition=attachment`, remote ChatGPT
    estuary URL, and `metadata.localPath`.

Next:

- Probe the project-bound ChatGPT conversation/artifact UI directly from
  AuraCall to decide whether ChatGPT generated `legacy_readout.json` and
  AuraCall missed it, or ChatGPT replied ready without creating a downloadable
  artifact.

## Turn 66 | 2026-05-14

Summary: Scaled the working AuraCall artifact path to a five-item legacy
enrichment batch.

Action:

- Ran a dry-run over five pending legacy transcript artifacts to confirm the
  selected slice.
- Submitted live batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-192515.json`.
- Batch id: `batch_4039e070788e4190b371d6e9be4a4627`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.
- Polled with `status --materialize --store` until completion.

Validation:

- Batch completed with `total=5`, `completed=5`, `failed=0`.
- Response ids:
  - `resp_e5550593a3eb46d59770fc3ae5acaa64`
  - `resp_779cbfa125b1435a86de0d2bab81d9f3`
  - `resp_0c524b251f514100bb162520ae41aa12`
  - `resp_61d874f762324a73a399491bb8269ad7`
  - `resp_bdb5f5b24ff540799ddf51e1de8ee69b`
- Materialized readouts:
  - `/home/ecochran76/.transcripts/legacy-artifacts/fe/feb0a84f7d5262804b3f-2025-08-22 Baker Pappajohn Pitch My recording 21.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/84/843ca41a06ab290c2a66-2025-08-26 Amazon SoyLei Bio My recording 20.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/1d/1dd5a4304e17b9b76f9d-2025-05-15 SoyLei USDA BPP NCAT Visit My recording 3.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/da/dae8087a62f028b8c6cf-2025-05-19 Dr Dikis Follow up My recording 4.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/a9/a9c53a94e82d1b027bf5-2025-05-20 Saber Corn Board Call Alex Buck My recording 5.readout.json`
- File verification confirmed all five JSON and Markdown outputs exist and are
  non-empty.

Next:

- Continue scaling conservatively with a 10-item batch at concurrency 1.
- If 10 items pass cleanly, consider raising batch size before raising
  concurrency; keep browser interaction rate limiting unchanged until there is
  more variability data.

## Turn 67 | 2026-05-14

Summary: Scaled AuraCall legacy enrichment to a ten-item batch and preserved
partial materialization across provider/runtime failures.

Action:

- Submitted live batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-194324.json`.
- Batch id: `batch_e4aee9e4995d427980782ab7493af600`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.
- Polled with `status --materialize --store`.
- Hardened `scripts/auracall_legacy_enrichment_batch.py` so materialization
  records per-item extraction failures instead of aborting the whole batch and
  dropping later successful readouts.

Validation:

- Batch terminal state was not clean: `total=10`, `completed=9`, `failed=1`,
  `cancelled=0`, `missing=0`.
- The failed runner item was index 4,
  `resp_5abdf807f712471bbfb0c89f171631c8`, for
  `/home/ecochran76/.transcripts/legacy-artifacts/e4/e4443a54bbb79a9a2e48-2025-07-30 Nacu Breakfast with Nacu My recording 17.transcript.json`.
- Failure cause: ChatGPT browser auth preflight reported
  `chatgpt_account_session_drift`; expected `eric.cochran@soylei.com`, found
  `consult@polymerconsultinggroup.com`.
- One completed response still failed extraction: index 8,
  `resp_7c642b430b694b9fa4bdd59bcd087f35`, for
  `/home/ecochran76/.transcripts/legacy-artifacts/f6/f6d8ca6ef3bc0eecc682-2024-10-07 Vigil Cochran Performance Review My recording 5.transcript.json`.
- Extraction failure cause: AuraCall response did not include parseable readout
  JSON text or artifact output; observed assistant text began with a
  future-tense status sentence rather than the required downloadable artifact.
- Eight readouts were materialized and verified non-empty:
  - `/home/ecochran76/.transcripts/legacy-artifacts/ce/ceb04ee51746c31abd78-2025-06-05 Alireza CTE grade discussion My recording 7.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/7a/7aeed5ead7566ebc4412-2025-06-05 Iowa Energy Center Deicing Preproposal Discussion My recording 8.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/c1/c196f97cd0fbe68a669f-2025-07-31 Schulman Mac Visit My recording 15 (1).readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/c1/c18c6a8155bb03044dcd-2025-07-30 Green Dot CB2 project discussion  My recording 17 (1).readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/6d/6dee455fc63b5f406d4a-2025-07-30 Rudrapatna Follow up  My recording 16.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/bc/bc670708734fee1060a4-20240918-123353-Executive Licensing and Patent Strategy Meeting between Soilay and ACS on Brazilian Operations.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/8e/8e320b2f55cc9dcc11ec-2025-09-04 UL EPD Discussion SIP-1111 SIP-1132 My recording 20.readout.json`
  - `/home/ecochran76/.transcripts/legacy-artifacts/b9/b9d3d649b18165bcd9bd-2024-08-14 Meeting eith Chris and Scott Roberts Recording (1).readout.json`
- Manifest now records both `materialized` and `materialization_errors` so the
  retry boundary is deterministic.

Next:

- Fix or refresh the SoyLei Transcripts ChatGPT runtime profile so the bound
  account and browser session both use `eric.cochran@soylei.com`.
- Retry only the two missing items, not the whole ten-item batch.
- Keep batch size at 10 and concurrency at 1 until the account-drift and
  text-only artifact noncompliance paths are resolved.

## Turn 68 | 2026-05-14

Summary: Retried the two missing AuraCall legacy enrichment items after
verifying the SoyLei ChatGPT runtime identity.

Action:

- Verified Graphiti runtime health with `graphiti-runtime doctor`.
- Queried Graphiti group `transcribe_audio_main`; no newer AuraCall retry fact
  superseded the repo runbook/manifest evidence.
- Confirmed AuraCall profile identity using
  `auracall --profile wsl-chrome-3 profile identity-smoke --target chatgpt --prune-browser-state --json`.
- Identity preflight passed for runtime profile `wsl-chrome-3`: expected and
  actual ChatGPT identity were both `eric.cochran@soylei.com`.
- Dry-run queue
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-210644.json`
  confirmed the first two pending items were exactly the two missing artifacts
  from Turn 67.
- Submitted live retry batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260514-210745.json`.
- Batch id: `batch_9613778ae9fb4dd7a8257eed82375a23`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.

Validation:

- Batch completed with `total=2`, `completed=2`, `failed=0`, `cancelled=0`,
  `missing=0`.
- Materialization failed for both completed responses with
  `AuraCall response did not include parseable readout JSON text or artifact output`.
- Response `resp_1d69ad3a4036474c8434e6d9929ff5af` for
  `/home/ecochran76/.transcripts/legacy-artifacts/e4/e4443a54bbb79a9a2e48-2025-07-30 Nacu Breakfast with Nacu My recording 17.transcript.json`
  returned one assistant message only, beginning:
  `I'll create the requested legacy_readout.json artifact...`.
- Response `resp_1a585eb1c56f42c4982cf8df41f8b5b4` for
  `/home/ecochran76/.transcripts/legacy-artifacts/f6/f6d8ca6ef3bc0eecc682-2024-10-07 Vigil Cochran Performance Review My recording 5.transcript.json`
  returned one assistant message only, beginning:
  `I'm reading the attachment as an instruction packet...`.
- Neither response included any `artifact` output objects, local artifact path,
  downloadable JSON metadata, or parseable inline JSON.
- This retry proves the prior account-session drift was fixed before enqueue,
  but the browser runner can still stop at a future-tense/status-only ChatGPT
  reply even though the prompt forbids that shape.

Next:

- Fix AuraCall/OpenAI-compatible response handling or browser completion
  criteria so a ChatGPT response is not marked complete until the required
  `legacy_readout.json` artifact is present.
- Keep the two failed retry response ids and manifest as the current diagnostic
  boundary; do not keep re-enqueueing these two transcripts until the runner
  distinguishes "started working" status replies from completed artifacts.

## Turn 69 | 2026-05-15

Summary: Retried the two missing AuraCall legacy enrichment items after the
installed AuraCall runtime upgrade; the retry did not produce artifacts and was
cancelled after live browser/runtime diagnostics showed dispatch instability.

Action:

- Verified Graphiti runtime health and queried repo group `transcribe_audio_main`;
  no newer Graphiti fact superseded the repo runbook/manifest evidence.
- Confirmed the installed AuraCall service contained the required-artifact guard,
  then restarted `auracall-api.service` because it had been running since before
  the upgraded artifact-contract code was installed.
- Submitted live retry batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260515-055623.json`.
- Batch id: `batch_88aa13c439f44ea3a62aa9e636a5ddcc`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.
- Response ids:
  - `resp_971323e460de434d93bf2b44941091d7`
  - `resp_f047ef728c7e456ab2f7e8f03289fc52`

Validation:

- Initial batch execution was blocked by stale ChE grading response runs holding
  the same `wsl-chrome-3` ChatGPT browser runner. Cancelled:
  - `resp_6998b1db0f744932832054674dc17e65`
  - `resp_634a607998244cde84139e505600aa1f`
  - `resp_348547c367514c08899e2ea345ec3e63`
- First transcript response `resp_971323e460de434d93bf2b44941091d7` reached
  running state, but the managed Chrome renderer became unresponsive to direct
  CDP `Runtime.evaluate`. The managed Chrome process was terminated to release
  the browser lock, which stranded that response; it was cancelled at
  `2026-05-15T11:23:03Z`.
- Second transcript response `resp_f047ef728c7e456ab2f7e8f03289fc52` was then
  drained directly. Live CDP inspection showed the `wsl-chrome-3` ChatGPT tab
  idle at `https://chatgpt.com/`, not in the Transcripts project and not
  generating/submitting transcript work. It was cancelled at
  `2026-05-15T11:30:56Z`.
- Final batch status: `total=2`, `cancelled=2`, `completed=0`, `failed=0`,
  `materialized=0`, `materialization_errors=0`.
- This retry did not prove the ChatGPT artifact contract; it exposed a lower
  AuraCall browser dispatch/runtime problem before artifact generation could be
  tested.

Next:

- Fix AuraCall browser dispatch for project-bound `agent:pro-extended-chatgpt-soylei-transcripts`
  so it reliably opens the Transcripts project, submits the attached request,
  and reports a failure when navigation/submission does not happen.
- Add or use a small project-bound artifact smoke before retrying private
  transcript payloads again.
- After that smoke passes, retry only the two missing legacy items from the
  current manifest.

## Turn 70 | 2026-05-15

Summary: Verified the strengthened artifact-surfacing prompt against the
project-bound AuraCall ChatGPT agent, then retried the two pending legacy
transcript enrichments. The prompt now reaches ChatGPT and produces visible
`legacy_readout.json` artifact references, but AuraCall skips downloading those
`sandbox:` artifacts for the real transcript runs, so no local readouts were
materialized.

Action:

- Verified Graphiti runtime health and queried repo group `transcribe_audio_main`;
  no newer Graphiti fact superseded repo/runtime evidence.
- Confirmed `auracall-api.service` was active and the runtime env targeted
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- Ran a non-private artifact smoke through `/v1/response-batches` using the same
  Transcripts project-bound agent. Smoke batch `batch_5243d5260bb94feabb28b55e5f13e9e3`
  completed with response `resp_3202e87977554bd8b2f4990a47614c03` and produced a
  downloaded `legacy_readout.json` file under the AuraCall ChatGPT attachment
  cache.
- Submitted live legacy retry batch
  `/home/ecochran76/.local/state/transcribe-audio/auracall-batches/legacy-enrichment-20260515-091327.json`.
- Batch id: `batch_fdedf5abec1f496987d3a7a5769fe1b4`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Limits: `maxConcurrentRuns=1`, `maxBrowserInteractionsPerMinute=6`.
- Response ids:
  - `resp_0597befb3edf4e7fb6e201557f633b41`
  - `resp_7c146c4c0712414a891933cb24acd150`

Validation:

- The live batch completed: `total=2`, `completed=2`, `failed=0`, `cancelled=0`,
  `missing=0`.
- Both real transcript requests contained explicit instructions to create a
  ChatGPT REPL/workspace file named `legacy_readout.json`, surface it as a
  downloadable artifact/link/card, avoid text-only readiness replies, and avoid
  compressing the JSON for chat-message length.
- Response `resp_0597befb3edf4e7fb6e201557f633b41` returned an assistant status
  message plus an artifact object:
  `uri=sandbox:/mnt/data/legacy_readout.json`, `title=legacy_readout.json`,
  `artifact_type=generated`.
- Response `resp_7c146c4c0712414a891933cb24acd150` returned the same artifact
  shape: `uri=sandbox:/mnt/data/legacy_readout.json`, `title=legacy_readout.json`,
  `artifact_type=generated`.
- AuraCall artifact-fetch manifests for conversations
  `6a072a0d-abd8-83ea-9fe7-ffbc28a4a522` and
  `6a072a9c-9b1c-83ea-823d-b11a2da12fa1` show `artifactCount=1`,
  `materializedCount=0`, and the `legacy_readout.json` `sandbox:` entry marked
  `status=skipped`.
- The transcribe-audio materializer therefore reported both completed responses
  as `AuraCall response did not include parseable readout JSON text or artifact
  output` and wrote no local readout JSON files.

Next:

- Treat the prompt and project-bound dispatch as proven enough for the current
  boundary: the real failure is that AuraCall exposes only a generated
  `sandbox:/mnt/data/legacy_readout.json` artifact reference for these transcript
  jobs and skips downloading it into a file/artifact output.
- Fix AuraCall artifact fetching so generated `sandbox:` download artifacts are
  fetched or surfaced with a retrievable local path/content, matching the smoke
  behavior that produced a downloaded cache file.
- After that fix, rerun the same two-item manifest or enqueue another two-item
  bounded retry; do not widen the batch until materialization succeeds.

## Turn 71 | 2026-05-15

Summary: Repaired the live auto-transcription watcher so it sees current `.m4a`
files in Downloads and Syncthing Sound Recordings, keeps calendar mode enabled,
and drains newest recordings first.

Action:

- Verified `transcribe-watch.service` was active but reporting
  `candidates=0` for hours.
- Found the runtime config only watched `~/Downloads` with glob
  `My Recording*.m4a`, which missed dated Windows recorder files such as
  `2026-05-13 ... My recording 129.m4a`.
- Updated ignored runtime config `watch_transcriptions.json` to watch all
  `*.m4a` files in `~/Downloads` and added a recursive
  `~/SyncThing/Documents/Sound Recordings` job.
- Updated tracked sample config to use `*.m4a` for Downloads examples.
- Updated watcher candidate ordering to newest-first so recent recordings are
  not blocked behind older backlog.
- Restarted `transcribe-watch.service`.

Validation:

- Service restarted cleanly and loaded two jobs:
  `downloads-mobile-recordings` and `syncthing-sound-recordings`.
- Heartbeat changed from `candidates=0` to `candidates=88`.
- The live child process began transcribing
  `/mnt/c/Users/ecoch/Downloads/2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129.m4a`
  via AssemblyAI.
- The live transcription command includes calendar mode:
  `--use-calendar --calendar-providers gog,gws,google-api --calendar-id primary --calendar-window 24`.
- State after restart showed `downloads-mobile-recordings` with 77 pending
  candidates and `syncthing-sound-recordings` with 2 pending candidates while
  the newest file was actively transcribing.

Next:

- Keep monitoring `journalctl --user -u transcribe-watch.service -f` until the
  first broadened-glob transcription completes and store ingest succeeds.
- If AssemblyAI fails, confirm the faster-whisper fallback completes before
  changing service scope.

## Turn 72 | 2026-05-15

Summary: Continued live watcher monitoring after broadening the watch scope;
confirmed end-to-end transcription, calendar context, store ingest,
faster-whisper fallback, Syncthing ingestion, and then hardened the watcher so
new recordings remain responsive while historical backlog drains.

Action:

- Monitored `transcribe-watch.service` after enabling `*.m4a` matching and the
  Syncthing Sound Recordings job.
- Confirmed the broadened Downloads job completed
  `2026-05-13 13-00 Kiddie training and 1 other(s) My recording 129.m4a` via
  AssemblyAI with calendar context and stored two transcript artifacts.
- Confirmed AssemblyAI rejected the long NCAT/MnROAD recording as
  `Audio duration is too long`; the watcher correctly fell back to
  faster-whisper and stored the resulting transcript artifact.
- Confirmed Syncthing Sound Recordings ingestion completed
  `2022-08-10 Drive Time Live on KSCJ.m4a` via AssemblyAI and stored the
  transcript artifact.
- Changed the watcher to process at most one file per job per scan, forcing
  state saves between files and allowing newly arrived recordings to be noticed
  between backlog items.
- Added duplicate-rename protection: if a calendar-renamed media file has the
  same size and mtime as a successful prior record, the watcher records the new
  path as already processed instead of transcribing it again.
- Restarted `transcribe-watch.service` to load both changes.

Validation:

- `python -m py_compile watch_transcriptions.py` passed.
- `git diff --check` passed.
- After restart, the watcher marked renamed aliases for already-processed files
  instead of reprocessing them.
- The service advanced to the next real unprocessed Downloads file:
  `2026-05-01 12-00 ChE 4470 and 1 other(s) My recording 111.m4a` via AssemblyAI.
- Live service remained active under systemd with calendar flags present on
  child transcription commands.

Next:

- Continue periodic monitoring until the backlog is drained enough that the
  newest incoming recordings are handled immediately.
- Fix the underlying filename duplication behavior separately so calendar-aware
  output names do not repeat an already-calendar-prefixed title.
- Repair Slack notification PATH or command availability if success/failure
  alerts are still desired; current logs show `openclaw not found on PATH`.

## Turn 73 | 2026-05-15

Summary: Continued watcher monitoring, found that calendar-renamed media could
still be retried after success, and hardened retry detection so renamed/same-size
successful media are treated as already processed.

Action:

- Monitored the current Downloads backlog after the one-file-per-job scan change.
- Confirmed `2026-05-01 12-00 ChE 4470 and 1 other(s) My recording 111.m4a`
  completed via AssemblyAI, wrote calendar-context artifacts, and stored them in
  `~/.transcripts`.
- Confirmed both Syncthing Sound Recordings files completed and stored; the
  Syncthing job later showed `candidates=0`.
- Found two retry gaps caused by calendar renames:
  - same media path could be retried when mtime/fingerprint changed after a
    successful rename;
  - renamed paths could be retried when the new filename embedded the old media
    filename but mtime was not stable enough for the previous equivalence check.
- Hardened watcher retry logic:
  - a successful processed record with the same file size is not retried just
    because the fingerprint changed;
  - renamed media is matched against successful prior records by same size and
    embedded prior filename.
- Restarted `transcribe-watch.service` to load the patch and cancel duplicate
  in-flight retries.

Validation:

- `python -m py_compile watch_transcriptions.py` passed.
- `git diff --check` passed.
- After restart, the watcher skipped duplicate renamed media and advanced to
  `2026-05-01 10-00 BOTTLE update meeting - Sweeney My recording 110.m4a`.
- `110.m4a` completed via AssemblyAI, matched calendar event
  `BOTTLE update meeting - Sweeney`, wrote a transcript artifact, and stored it
  under `~/.transcripts/artifacts/56/...`.
- Final observed service state: active under systemd, no child process running,
  Downloads state had 45 processed path records and 60 candidates, Syncthing
  state had 5 processed path records and 0 candidates.

Next:

- Fix the root filename construction so calendar prefixes are not prepended to
  filenames that already contain the calendar title/date.
- Keep the watcher running; continue periodic monitor passes while historical
  Downloads backlog drains.

## Turn 74 | 2026-05-15

Summary: Fixed calendar-aware filename construction so already-prefixed media
does not produce duplicate date/title prefixes in transcript outputs.

Action:

- Added prefix stripping in `transcribe_common.py` before constructing
  calendar-aware output basenames.
- Covered both exact repeated `YYYY-MM-DD HH-MM Event` prefixes and cases where
  prior overlapping-calendar titles left a leading `and N other(s)` fragment.
- Added a regression test for already-calendar-prefixed Downloads recordings.
- Restarted `transcribe-watch.service` after the patch.

Validation:

- `python -m pytest tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile transcribe_common.py watch_transcriptions.py` passed.
- `git diff --check` passed.
- After restart, the watcher completed
  `2026-04-09 13-00 Andersons-Saber Update My recording 87.m4a` via AssemblyAI
  and wrote clean output names without a repeated calendar prefix.
- Final observed service state: active under systemd, Downloads had 75 processed
  path records and 45 candidates, Syncthing had 5 processed path records and 0
  candidates.

Next:

- Continue monitoring until Downloads backlog drains and new recordings are
  picked up immediately.
- Decide whether to clean already-created duplicate-prefixed transcript outputs
  from before this fix.
- Repair the Slack notification runtime path if watcher alerts should resume;
  current service logs still show `openclaw not found on PATH`.

## Turn 75 | 2026-05-16

Summary: Fixed a watcher liveness bug where already-processed filesystem
matches were counted as queued candidates, causing repeated no-progress
systemd restarts after the backlog drained.

Action:

- Checked Graphiti advisory memory for watcher backlog and service-state
  guidance, then verified against local systemd logs and state files.
- Found persisted watcher state had no queued candidates, while the live
  heartbeat still reported 88 candidates and eventually exited for no progress.
- Updated `scan_job` so `candidate_count` means pending/queued work after
  processed-file and equivalent-renamed-file skip checks, not every matching
  file on disk.
- Added a regression test confirming an already-successful media file reports
  zero queued candidates.
- Restarted `transcribe-watch.service` to load the fix.

Validation:

- `python -m pytest tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile watch_transcriptions.py transcribe_common.py` passed.
- `git diff --check` passed.
- After restart, the service heartbeat reported
  `candidates=0 attempted=0 successes=0 failures=0`.
- Historical duplicate-prefix inventory found 130 matching files under
  Downloads and 4 under Syncthing Sound Recordings; bulk cleanup was deferred
  because many names reflect overlapping calendar matches with different event
  titles, and renaming live media without state/sidecar updates could trigger
  re-transcription.

Next:

- Implement a bounded historical cleanup tool that updates media filenames,
  transcript sidecar paths, store metadata, and watcher state together.
- Keep the watcher running and confirm it immediately handles the next newly
  arriving recording.
- Repair Slack notification PATH if service notifications are still wanted.

## Turn 76 | 2026-05-16

Summary: Added and exercised a bounded historical filename cleanup tool for
calendar-prefixed transcript artifacts.

Action:

- Added `cleanup_transcript_filenames.py`.
- The tool derives canonical names from each transcript sidecar's calendar
  event metadata, defaults to dry-run, refuses to apply while the watcher is
  active unless `--manage-service` is used, rewrites transcript JSON path
  fields, updates watcher state, and can refresh `~/.transcripts` rows with
  `--refresh-store`.
- Added regression tests for planning, applying, sidecar rewrites, and watcher
  state rewrites.
- Documented the cleanup command in `README.md`.
- Applied three bounded live cleanup batches against Downloads/Sound Recordings
  using `--apply --manage-service --refresh-store`.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- `git diff --check` passed.
- Live cleanup applied 20 actionable plans, 19 file rename operations, watcher
  state updates, and 20 store refreshes.
- After each apply, `transcribe-watch.service` restarted successfully and
  heartbeated with `candidates=0 attempted=0 successes=0 failures=0`.
- A shared-media guard was added after dry-run exposed overlapping calendar
  artifacts that referenced the same recording but wanted different event
  titles; those media renames are now suppressed unless the current media name
  itself has cleanup noise such as `(1)`, duplicate date prefixes, or
  `and N other(s)`.

Next:

- Continue cleanup in small batches with:
  `.venv/bin/python cleanup_transcript_filenames.py ~/Downloads ~/SyncThing/Documents/"Sound Recordings" --recursive --limit 10 --apply --manage-service --refresh-store`
- Add a review/export mode for skipped conflicts so overlapping calendar
  artifacts can be resolved deliberately instead of guessed.

## Turn 77 | 2026-05-16

Summary: Continued historical cleanup batches until no safe automatic filename
cleanup remained.

Action:

- Ran additional bounded cleanup batches with `--apply --manage-service
  --refresh-store`.
- Applied 32 more actionable plans, 32 file rename operations, watcher state
  updates, and 32 transcript-store refreshes.
- Hardened `cleanup_transcript_filenames.py` so media renames are suppressed
  when the computed target still contains cleanup noise, preventing overlapping
  event artifacts from being treated as safe automatic cleanup.
- Added a regression test for that skipped/review-needed case.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- `git diff --check` passed.
- After each apply, `transcribe-watch.service` restarted successfully and
  heartbeated with `candidates=0 attempted=0 successes=0 failures=0`.
- Final dry-run reported `scanned=21 actionable=0 operations=0 skipped=21`.

Next:

- Add a skipped-conflict export/review workflow for the 21 remaining
  overlapping/ambiguous artifacts.
- Keep watcher monitoring in place and confirm next newly arriving recording is
  transcribed immediately.

## Turn 78 | 2026-05-16

Summary: Added skipped-conflict review export for historical transcript filename
cleanup.

Action:

- Added `cleanup_transcript_filenames.py --export-review <path>`.
- Review exports are JSON files with schema version, summary counts,
  per-reason counts, event metadata, source/working media paths, output paths,
  proposed operations, existing target conflicts, replacements, and suggested
  manual actions.
- Added tests for review payload construction and writing nested review paths.
- Documented the `--export-review` workflow in `README.md`.
- Generated the live review artifact at
  `.openclaw/reviews/transcript-filename-cleanup-review-2026-05-16.json`.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- `git diff --check` passed.
- Live export dry-run reported `scanned=21 actionable=0 operations=0
  skipped=21` and wrote the review file.

Next:

- Review the 21 conflict items and decide whether to merge/delete duplicate
  outputs or keep them as separate overlapping-calendar artifacts.
- Keep watcher monitoring in place and confirm next newly arriving recording is
  transcribed immediately.

## Turn 79 | 2026-05-16

Summary: Added and applied a conservative resolver for content-identical
filename cleanup conflicts.

Action:

- Added `cleanup_transcript_filenames.py --resolve-identical-conflicts`.
- The resolver compares transcript JSON text, TXT text, and DOCX paragraph
  text rather than relying on byte-identical files.
- Old redundant conflict files are moved to
  `~/.local/state/transcribe-audio/filename-cleanup-quarantine/` instead of
  being deleted.
- The resolver updates watcher state and can refresh `~/.transcripts` just like
  regular cleanup applies.
- Applied the resolver to live Downloads/Sound Recordings conflicts.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- `git diff --check` passed.
- Live resolver quarantined 12 redundant old conflict files, refreshed 4 store
  records, and restarted `transcribe-watch.service`.
- After restart, the watcher heartbeated with
  `candidates=0 attempted=0 successes=0 failures=0`.
- Regenerated review dry-run reported `scanned=17 actionable=0 operations=0
  skipped=17`.

Next:

- Review the remaining 17 non-equivalent conflicts manually; they are not safe
  for automatic merge because visible transcript/output content differs.
- Keep watcher monitoring in place and confirm next newly arriving recording is
  transcribed immediately.

## Turn 80 | 2026-05-16

Summary: Added privacy-conscious diff summaries to filename cleanup review
exports and classified the remaining conflicts.

Action:

- Added `cleanup_transcript_filenames.py --include-diff-summary` for
  `--export-review`.
- Diff summaries include line counts, body-line counts, similarity ratios,
  changed-line span counts, and a conservative classification, without storing
  transcript excerpts in the review JSON.
- Regenerated the live review artifact at
  `.openclaw/reviews/transcript-filename-cleanup-review-2026-05-16.json`.
- Remaining live conflicts classify as 7 metadata/format-only candidates, 2
  high-overlap review items, 6 partial-overlap distinct-content items, and 2
  preserve-both distinct-content items.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- Live review regeneration reported `scanned=17 actionable=0 operations=0
  skipped=17`.

Next:

- Add a reviewed apply path for metadata/format-only candidates that quarantines
  old files and rewrites pointers only after the review classification is
  explicitly selected.
- Keep the partial/distinct-content conflicts preserved until a human chooses
  whether each represents a separate overlapping-calendar artifact or a stale
  duplicate.

## Turn 81 | 2026-05-16

Summary: Added a reviewed apply path for metadata-only filename cleanup
conflicts.

Action:

- Added `cleanup_transcript_filenames.py --resolve-reviewed-conflicts`.
- Dry-run mode reports eligible reviewed resolutions without moving files.
- Apply mode only resolves conflicts whose computed diff summary classifies
  every target conflict as `metadata_or_format_only_candidate`.
- Reviewed resolution quarantines old conflict files, moves non-conflicting
  outputs, rewrites transcript sidecar pointers, updates watcher state, and can
  refresh the transcript store.

Validation:

- `python -m pytest tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile cleanup_transcript_filenames.py` passed.
- Live dry-run reported 7 eligible reviewed resolutions out of 17 skipped
  conflicts; each would quarantine one old DOCX conflict.

Next:

- Apply the 7 reviewed metadata-only resolutions with service management and
  store refresh.
- Regenerate the review export and preserve the remaining high-overlap,
  partial-overlap, and distinct-content conflicts for human review.

## Turn 82 | 2026-05-16

Summary: Applied the reviewed metadata-only filename cleanup resolutions.

Action:

- Ran `cleanup_transcript_filenames.py --apply --manage-service
  --refresh-store --resolve-reviewed-conflicts` against Downloads and
  Syncthing Sound Recordings.
- Resolved 7 reviewed metadata-only conflicts.
- Quarantined 7 old DOCX conflict files under
  `~/.local/state/transcribe-audio/filename-cleanup-quarantine/`.
- Moved 20 non-conflicting output/media paths, rewrote watcher state, and
  refreshed 7 transcript-store rows.
- Regenerated the live review artifact at
  `.openclaw/reviews/transcript-filename-cleanup-review-2026-05-16.json`.

Validation:

- Apply output reported `reviewed_resolved_count=7`, `state_updated=true`, and
  `store_refreshed_count=7`.
- Post-apply review reported `scanned=10 actionable=0 operations=0 skipped=10`.
- Remaining conflict classes: 2 high-overlap review items, 6 partial-overlap
  distinct-content items, and 2 preserve-both distinct-content items.
- `transcribe-watch.service` restarted successfully and heartbeated with
  `candidates=0 attempted=0 successes=0 failures=0`.

Next:

- Do not auto-merge the remaining 10 conflicts; they contain meaningful
  transcript/output differences under the current classifier.
- Build a small human-review report or UI slice that lets the operator compare
  the remaining pairs and choose preserve, quarantine old, or keep both.

## Turn 83 | 2026-05-16

Summary: Added a filename-conflict human review report generator.

Action:

- Added `transcript_filename_conflict_review.py`.
- The generator reads a cleanup `--export-review --include-diff-summary` JSON
  file and writes a user-scoped review template plus Markdown report under
  `~/.local/state/transcribe-audio/filename-conflict-reviews/` by default.
- Each review item has a stable id, `pending` decision, recommended decision,
  allowed decisions, conflict paths, diff metrics, event metadata, and planned
  non-conflicting operations.
- Decision choices are `preserve_both`, `quarantine_old`, `keep_target`, and
  `needs_investigation`.
- The report remains local runtime state and does not commit private paths or
  transcript content.

Validation:

- `python -m pytest tests/test_transcript_filename_conflict_review.py tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile transcript_filename_conflict_review.py cleanup_transcript_filenames.py` passed.
- Live generation wrote
  `~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review-20260516-153723.json`
  and matching Markdown.
- Live review summary: 10 items; 2 `distinct_content_preserve_both`, 2
  `high_overlap_needs_review`, and 6 `partial_overlap_distinct_content`.
- Recommended decisions: 2 `preserve_both`, 8 `needs_investigation`.

Next:

- Use that template as the apply boundary for any future human-selected
  conflict resolution.
- Add an apply path that consumes the review template and only acts on explicit
  non-pending decisions.

## Turn 84 | 2026-05-16

Summary: Added an explicit-decision apply path for filename-conflict review
templates.

Action:

- Extended `transcript_filename_conflict_review.py` with `--apply-review`.
- Review-template applies are dry-run by default.
- Live mutation requires `--apply --approval-token
  APPLY_FILENAME_CONFLICT_REVIEW`.
- `pending` and `needs_investigation` decisions are skipped.
- `preserve_both` and `keep_target` are recorded no-op decisions.
- Only `quarantine_old` moves files: old conflict paths are quarantined,
  planned non-conflicting operations are moved, sidecar pointers are rewritten,
  watcher state can be updated, and transcript-store rows can be refreshed.

Validation:

- `python -m pytest tests/test_transcript_filename_conflict_review.py tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile transcript_filename_conflict_review.py cleanup_transcript_filenames.py` passed.
- Live dry-run over
  `~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review-20260516-153723.json`
  reported 10 skipped items and 0 mutating decisions because all decisions are
  still `pending`.

Next:

- Operator should edit the review JSON decisions for any of the remaining 10
  conflicts, then rerun `--apply-review` as dry-run.
- After dry-run confirms the intended mutations, apply with
  `--apply --approval-token APPLY_FILENAME_CONFLICT_REVIEW --manage-service
  --refresh-store`.

## Turn 85 | 2026-05-16

Summary: Marked the two distinct-content filename conflicts as preserve-both
and added review audit output support.

Action:

- Updated the live operator review JSON under
  `~/.local/state/transcribe-audio/filename-conflict-reviews/` for the two
  `distinct_content_preserve_both` items.
- Set both decisions to `preserve_both` with reviewer metadata and decision
  reasons.
- Added `transcript_filename_conflict_review.py --audit-output` so dry-run or
  apply results can be written as durable local runtime audit JSON.

Validation:

- Live dry-run over the edited review JSON reported 2 `recorded_noop`, 8
  `skipped`, and 0 mutating decisions.
- `python -m pytest tests/test_transcript_filename_conflict_review.py tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile transcript_filename_conflict_review.py cleanup_transcript_filenames.py` passed.
- `git diff --check` passed.
- Wrote local audit JSON:
  `~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review-20260516-153723-preserve-both-audit.json`.

Next:

- Continue with investigation support for the remaining 8 pending conflicts.

## Turn 86 | 2026-05-16

Summary: Added bounded local investigation reports for pending filename
conflicts.

Action:

- Extended `transcript_filename_conflict_review.py` with
  `--investigate-review`.
- The investigation report includes only pending review items.
- Each conflict includes bounded first-difference hunks, line counts, paths,
  diff summary, and event/review context.
- Investigation artifacts are explicitly local/private because they may contain
  transcript snippets.

Validation:

- `python -m pytest tests/test_transcript_filename_conflict_review.py tests/test_cleanup_transcript_filenames.py tests/test_transcript_artifacts.py -q` passed.
- `python -m py_compile transcript_filename_conflict_review.py cleanup_transcript_filenames.py` passed.
- Live generation wrote
  `~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-investigation-20260516-160221.json`
  and matching Markdown.
- Live report contains 8 pending items: 2 `high_overlap_needs_review` and 6
  `partial_overlap_distinct_content`.

Next:

- Review the local Markdown investigation report and set explicit decisions in
  the review JSON for any items that are now clear.
- Keep generated investigation reports out of git because they may contain raw
  transcript snippets.

## Turn 87 | 2026-05-16

Summary: Closed the remaining filename-conflict review as no-mutation decisions.

Action:

- Operator reviewed the Previews-published investigation report and determined
  the remaining 8 pending conflicts were basically identical except for timezone
  spelling (`CDT` spelled out in one output and not the other).
- Updated the local review JSON under
  `~/.local/state/transcribe-audio/filename-conflict-reviews/` to set those 8
  items to `keep_target`.
- The review now has 8 `keep_target` decisions and 2 previously recorded
  `preserve_both` decisions.
- Wrote final no-mutation audit JSON:
  `~/.local/state/transcribe-audio/filename-conflict-reviews/filename-conflict-review-20260516-153723-final-keep-target-audit.json`.

Validation:

- `transcript_filename_conflict_review.py --apply-review ... --audit-output ...`
  reported 10 `recorded_noop` items and 0 mutating decisions.
- No transcript files were moved or quarantined in this step.

Next:

- Leave the reviewed conflicts in place; the target outputs are operator truth.
- Return to the broader review-console/UI work or service maintenance.

## Turn 88 | 2026-05-16

Summary: Started P09 with the first React + Vite review-console shell.

Action:

- Added `frontend/` with a Vite React app.
- Implemented sticky top navigation for Library, Review Queue, Context Runs,
  Contacts, Provenance, Intelligence, Depositions, and Settings.
- Added animated left filter pane, central table/review viewport, and right
  inspector pane.
- Wired the shell to `/api/health` and `/api/library` through a Vite dev proxy
  to `transcript_api.py`.
- Added redacted fallback fixture rows when the local API is unavailable.
- Surfaced the filename-conflict review queue as closed/no-mutation state.
- Kept generated frontend dependencies and builds ignored under
  `frontend/node_modules/` and `frontend/dist/`.

Validation:

- `npm install` completed with 0 vulnerabilities.
- `npm run build` passed in `frontend/`.
- `python -m py_compile transcript_api.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_filename_conflict_review.py -q` passed.

Next:

- Add read-only API support for review queue manifests under
  `~/.local/state/transcribe-audio/`, then replace the frontend's hard-coded
  queue summary with live review queue data.

## Turn 104 | 2026-05-17

Summary: Cancelled the one-item first-pass retry after runtime evidence showed
the active AuraCall lease was attached to the wrong ChatGPT target.

Action:

- Monitored one-item retry manifest
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260517-204443.json`.
- Batch id: `batch_b233f1defa434225abc95acf46fac534`.
- Response id: `resp_801d7fae735e4a348460029d8ca95ef0`.
- The retry stayed `running` for more than 90 minutes with repeating
  `chatgpt-passive-dom-probe` and `browser-runtime-hint` lease heartbeats.
- A materialize/read poll while the run was still active returned
  `AuraCall read failed (400): Unexpected end of JSON input`, so subsequent
  monitoring used plain batch status until terminal state.
- CDP target inspection showed the lease target
  `2DD81FEB230FEF239857872E722DEB56` was currently on
  `https://chatgpt.com/library`, and no open page target matched conversation
  `6a0a6f14-7a80-83ea-a77b-81f654b709aa`.
- Official AuraCall runtime inspection confirmed the active browser diagnostic
  target was `ChatGPT - Library`, with `modelResponses=0`, even though the
  lease heartbeat still renewed.
- Cancelled the run through `POST /status` with `runControl.cancel-run`.

Validation:

- Cancellation result: `status=cancelled`, `cancelled=true`.
- Final retry batch status: `cancelled`.
- Final retry counts: `total=1`, `completed=0`, `failed=0`, `cancelled=1`,
  `missing=0`, `in_progress=0`.
- AuraCall recovery after cancellation: `reclaimable=0`, `activeLease=0`,
  `recoverableStranded=0`, `stranded=0`.
- `systemctl --user is-active transcripts.service auracall-api.service`
  returned `active` and `active`.
- Live first-pass summary queue still reports 15 pending items, with
  `2026-02-12 Scott Roberts Call 2 Recording` first.

Notes:

- This is not a transcript-store failure and not a time-only timeout decision.
  The stop condition was contradictory runtime evidence: the active lease kept
  renewing while the associated browser target was no longer the transcript
  conversation.
- Do not submit another private first-pass summary batch until AuraCall rejects
  lease renewal when the active ChatGPT target is on Library/project/root
  instead of the expected conversation URL, and until stale completed/failed
  transcript tabs are cleaned or isolated from running prompt ownership.

Next:

- Return to AuraCall and fix the ChatGPT browser lifecycle/evidence boundary:
  one running prompt must have one live conversation target, and passive DOM
  lease evidence must be tied to that target instead of stale stored metadata.

## Turn 105 | 2026-05-18

Summary: Reopened AuraCall transcript-intake testing after the target-bound
browser fix and completed one first-pass readout.

Action:

- Confirmed `auracall-api.service` was systemd-active but `/status` initially
  hung; inspected service state, CDP targets, runtime records, and logs before
  restarting the user service.
- Restarted `auracall-api.service` after confirming the previous private run
  was already `cancelled` and the later non-private smoke had `succeeded`.
- Verified `/status` returned `ok: true` after restart.
- Ran a non-private direct `/v1/responses` target-bound artifact smoke through
  `agent:pro-extended-chatgpt-soylei-transcripts`.
- Left the stale ChatGPT tabs in place for the smoke so the target-binding
  evidence was tested with `ChatGPT - Library` and older transcript targets
  still present.
- Submitted a one-item first-pass summary batch after the non-private gate
  passed.

Validation:

- Non-private smoke response: `resp_797f2f89d22845789ca01148a5f4713d`.
- Non-private smoke conversation:
  `6a0b143a-acb4-83ea-aa81-183b642eb46b`.
- Non-private smoke target: `58BD9E2D464AC4FD51DB3523E5A6DB0D`.
- Non-private smoke completed with a surfaced
  `sandbox:/mnt/data/first_pass_readout.json` link and a materialized local
  artifact under AuraCall's ChatGPT cache.
- The materialized smoke artifact parsed as valid JSON and contained
  `summary=AURACALL_TRANSCRIPT_INTAKE_ARTIFACT_OK`.
- Private batch manifest:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-083505.json`.
- Private batch id: `batch_cc2e6e20733844f48e6351dcc6283026`.
- Private response id: `resp_2fd5d079dd7246d2956328503cfab449`.
- Private response completed with target
  `83E199BFB1A6E3506ABF70C0A2B4C075` and conversation
  `6a0b1594-8fc8-83ea-98be-2e6460310bcc`.
- Materialized readouts:
  `~/.transcripts/legacy-artifacts/ac/acdee7fa22751e3a64e2-2026-02-12 Scott Roberts Call 2 Recording.readout.json`
  and
  `~/.transcripts/legacy-artifacts/ac/acdee7fa22751e3a64e2-2026-02-12 Scott Roberts Call 2 Recording.readout.md`.
- The JSON readout parsed successfully and included populated participants,
  topics, decisions, action items, matter candidates, memory candidates, risks,
  and next steps.

Notes:

- The successful private run took about eight minutes from enqueue to
  completion, so long-running `Pro Extended` transcript readouts should not be
  treated as stalled solely because the browser log has already reported a
  recovered response.
- The ChatGPT profile still contains stale Library and older transcript tabs;
  the passing target-bound smoke proves the current run can bind to the correct
  conversation target in their presence, but cleanup/isolation is still useful
  operational hygiene.

Next:

- Scale cautiously to another small first-pass batch, preferably 2-3 items, and
  continue requiring surfaced `first_pass_readout.json` artifacts before
  materialization counts as successful.

## Turn 106 | 2026-05-18

Summary: Scaled AuraCall transcript-intake testing to three items; one readout
materialized, then the AuraCall service restarted and left the remaining two
responses non-terminal.

Action:

- Ran Graphiti discovery setup for repo group `transcribe_audio_main`; the
  runtime doctor was healthy, but discovery returned no current repo facts, so
  repo files and live runtime state remained authoritative for this slice.
- Confirmed `auracall-api.service`, `transcribe-watch.service`, and
  `transcripts.service` were active before submitting more transcript work.
- Prepared dry-run manifest
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-084520.json`
  with three pending first-pass transcript readouts.
- Enqueued manifest
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-085533.json`.
- Batch id: `batch_baec0e7666d143a283a01a4f4828507d`.
- Response ids:
  `resp_81b34938e9694aa9aaf19198cfe8cf89`,
  `resp_7504789aba714119b23903bfbbed4adf`, and
  `resp_3c57d99d2e5e4601970149345dfab749`.

Validation:

- Response `resp_81b34938e9694aa9aaf19198cfe8cf89` completed through
  conversation `6a0b1a5d-e0f8-83ea-bd83-3f1fc76890f8` and target
  `B0AEEA52DCF94C4E9A4C415569D29ABE`.
- The first response surfaced `sandbox:/mnt/data/first_pass_readout.json`,
  AuraCall downloaded it into its ChatGPT cache, and transcript tooling
  materialized:
  `~/.transcripts/legacy-artifacts/ff/ff13b2fc131bfca2eb12-2026-02-20 13-30 Meet with Eric (ryan jaggar) My recording 60.readout.json`
  and
  `~/.transcripts/legacy-artifacts/ff/ff13b2fc131bfca2eb12-2026-02-20 13-30 Meet with Eric (ryan jaggar) My recording 60.readout.md`.
- Batch status after materialization reported `total=3`, `completed=1`,
  `in_progress=2`, `failed=0`, `cancelled=0`, and one materialized readout.
- `resp_7504789aba714119b23903bfbbed4adf` remains API-visible as
  `in_progress` with no output; its runtime record is `running` with a
  `lease expired` event and no active lease.
- `cancel-run` for `resp_7504789aba714119b23903bfbbed4adf` returned HTTP 409:
  `run has no active lease to cancel`.
- `resp_3c57d99d2e5e4601970149345dfab749` remains API-visible as
  `in_progress` with no output; its runtime record is `running` even after a
  `lease released: cancelled` event.
- `auracall-api.service` is active again under PID `3179893`, entered active
  state at `Mon 2026-05-18 09:08:48 CDT`, and has `NRestarts=0` for the current
  unit instance.

Notes:

- AuraCall logs during the second response included `Received SIGTERM; leaving
  Chrome running (assistant response pending)` and `Session still in flight; use
  your reattach command to continue`; the service then restarted while the
  batch was active.
- A later batch materializer/status poll briefly failed with
  `Expected double-quoted property name in JSON at position 1048544`, but the
  batch status endpoint later recovered enough to report counts and the one
  materialized artifact.
- The current failure mode is not prompt quality or missing artifact surfacing.
  It is a runtime recovery boundary: responses can remain `running` /
  `in_progress` without an active lease after service restart or cancellation.
- Do not scale transcript-intake batches further until AuraCall can reconcile,
  cancel, or recover no-active-lease running responses and keep response-batch
  status durable across service restarts.

Next:

- Take the stranded-response evidence to the AuraCall repo: response batches
  need restart-safe child-run reconciliation, a cancellation path for
  no-active-lease `running` runs, and batch-status hardening around corrupted or
  partially written child state. After that, retry the two unfinished transcript
  items in a fresh one- or two-item batch.

## Turn 107 | 2026-05-18

Summary: Rechecked the three-item AuraCall transcript-intake batch after the
handoff note started; the two previously non-terminal responses had completed
and all three readouts materialized.

Action:

- Re-read AuraCall runtime records for
  `resp_7504789aba714119b23903bfbbed4adf` and
  `resp_3c57d99d2e5e4601970149345dfab749`.
- Re-read direct `/v1/responses/{id}` status through the
  `auracall-transcripts.env` scoped key.
- Re-ran the batch status/materialization command for
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-085533.json`
  with `--materialize --store`.
- Left an AuraCall handoff note at
  `../auracall/docs/dev/notes/2026-05-18-transcribe-batch-restart-recovery-handoff.md`.

Validation:

- Batch id `batch_baec0e7666d143a283a01a4f4828507d` now reports
  `status=completed`.
- Final counts: `total=3`, `completed=3`, `in_progress=0`, `failed=0`,
  `cancelled=0`, and `missing=0`.
- `resp_7504789aba714119b23903bfbbed4adf` later recorded `step-succeeded` at
  `2026-05-18T14:23:50.930Z` and direct response status `completed`.
- `resp_3c57d99d2e5e4601970149345dfab749` later recorded `step-succeeded` at
  `2026-05-18T14:19:48.438Z` and direct response status `completed`.
- Materialized readouts now include:
  `~/.transcripts/legacy-artifacts/c7/c72a9a2433cfe9027b83-2025-08-20 Nacu Eric Line of Business follow up meeting My recording 19.readout.json`
  and
  `~/.transcripts/legacy-artifacts/62/62b0e1928e29f2f6e4db-2025-09-26 SoyLei Scott Roberts Nacu Austin My recording 32.readout.json`.
- `materialization_errors=[]`.

Notes:

- Turn 106 captured a real intermediate recovery ambiguity, but it is not the
  final batch outcome.
- The remaining AuraCall issue is still worth fixing: callers should not see
  contradictory `in_progress`, no-active-lease, cancel-released, or transient
  malformed-batch-read states when the child browser work is still capable of
  completing.

Next:

- Use the AuraCall handoff note as the next owner boundary, then continue
  transcript-intake scaling only after deciding whether the current intermediate
  recovery ambiguity is acceptable for another small batch or needs an
  AuraCall-side status/cancel fix first.

## Turn 108 | 2026-05-18

Summary: Ran the first controlled one-item transcript-intake retry after the
AuraCall restart recovery fix; the batch survived a recovery window,
materialized, and passed the readout quality gate.

Action:

- Ran Graphiti discovery for `transcribe_audio_main`; it returned only older
  repo/bootstrap facts, so current repo files and the AuraCall handoff note
  remained the authority.
- Prepared and submitted a one-item first-pass summary batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-115121.json`.
- Batch id: `batch_83c0ab506c434c6db2ce0cc1f59fc601`.
- Child response id: `resp_5f8d3719d8784a6baf74b40bdbdc95c3`.
- Limits: `maxConcurrentRuns=1`,
  `maxBrowserInteractionsPerMinute=4`.
- Polled AuraCall status without submitting additional work while the child ran.
- Materialized the completed readout with `--materialize --store`.

Validation:

- During the run, AuraCall recorded target-bound passive evidence for
  conversation `6a0b43c0-8570-83ea-9ecc-0879a1eb94ee` on submitted Chrome
  target `B7E8181F5B74AF85EDD01AEDB267F833`.
- The run exercised restart/recovery behavior: the initial browser session
  received `SIGTERM`, AuraCall reattached to the submitted tab, and the batch
  still converged to `completed`.
- Final counts: `total=1`, `completed=1`, `in_progress=0`, `failed=0`,
  `cancelled=0`, and `missing=0`.
- Final diagnostics reported `terminalTransitionSource=step-succeeded`.
- Materialized readout:
  `~/.transcripts/legacy-artifacts/ac/acf05a3ca0b499dc2e9b-2024-10-02 Rich Sean Scott Colorbiotics.readout.json`.
- `materialization_errors=[]`.
- `python scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-115121.json --format text`
  passed with `1 pass, 0 warn, 0 fail`.
- Live review queue now reports 10 pending first-pass summaries.
- `transcripts.service` and `auracall-api.service` are both active.

Notes:

- This smoke confirms the fixed AuraCall installed path can recover a
  browser-backed transcript readout after service interruption and still
  materialize a valid artifact.
- The terminal batch row still reports `leaseState=expired` because the final
  lease had already expired by the time status was read; the important
  convergence signal is `completed` plus `terminalTransitionSource=step-succeeded`.

Next:

- Continue with another small controlled batch before scaling. Keep batch
  concurrency at 1 until multiple consecutive materialized readouts complete
  without contradictory operator-facing status.

## Turn 109 | 2026-05-18

Summary: Ran the second controlled one-item transcript-intake retry after the
AuraCall restart recovery fix; the batch completed, materialized, and passed
the readout quality gate.

Action:

- Rechecked repo policy, service health, and Graphiti discovery for
  `transcribe_audio_main`; Graphiti again returned older bootstrap facts only.
- Prepared and submitted a one-item first-pass summary batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-121059.json`.
- Batch id: `batch_c5b71dbdeb7a4b09b295172911c8fea1`.
- Child response id: `resp_a66384f1a6f24b2b89d91299ad63007f`.
- Limits: `maxConcurrentRuns=1`,
  `maxBrowserInteractionsPerMinute=4`.
- Polled AuraCall status without submitting additional work while the child ran.
- Materialized the completed readout with `--materialize --store`.

Validation:

- During the run, AuraCall recorded target-bound passive evidence for
  conversation `6a0b4849-8328-83ea-bc16-f8db5c026eac` on submitted Chrome
  target `6D4DFF018704B4450120A8A84B79B312`.
- The run progressed from `thinking` to `response-complete`, briefly reported
  `leaseState=expired` while still `in_progress`, then converged to
  `completed`.
- Final counts: `total=1`, `completed=1`, `in_progress=0`, `failed=0`,
  `cancelled=0`, and `missing=0`.
- Final diagnostics reported `terminalTransitionSource=step-succeeded`.
- Materialized readout:
  `~/.transcripts/legacy-artifacts/63/636b92e150f41aab214a-2025-10-16 Scott Eric Chris response to Wittmack Demand Letter.readout.json`.
- `materialization_errors=[]`.
- `python scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-121059.json --format text`
  passed with `1 pass, 0 warn, 0 fail`.
- Live review queue now reports 9 pending first-pass summaries.
- `transcripts.service` and `auracall-api.service` are both active.

Notes:

- This is the second consecutive one-item transcript-intake success on the
  fixed installed AuraCall path.
- The residual operator-facing rough edge is still the transient
  `response-complete` plus expired-lease interval before the batch row flips to
  `completed`; it resolved without cancellation or manual repair.

Next:

- Continue with a small controlled batch of two or three items at concurrency
  1, or fix the transient finalization display in AuraCall before scaling if
  operator clarity is more important than throughput.

## Turn 110 | 2026-05-19

Summary: Dogfooded AuraCall dispatch-pool routing for first-pass summaries with
three ChatGPT Pro transcript agents bound to the `Transcripts` project.

Action:

- Created AuraCall dispatch team `transcribe-audio-chatgpt-pro-pool` with
  `next_available` dispatch and `projectSync=none`.
- Bound one ChatGPT Pro transcript agent per AuraCall runtime profile:
  `wsl-chrome-2`, `wsl-chrome-3`, and `wsl-chrome-4`.
- Updated the transcribe-audio scoped AuraCall client env to use
  `AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool`,
  `AURACALL_MODEL=gpt-5.2-pro`, and
  `AURACALL_DISPATCH_MODEL=gpt-5.2-pro`.
- Updated `scripts/auracall_legacy_enrichment_batch.py` so dispatch-team
  batches put the team on the top-level AuraCall batch payload, keep child
  requests provider-model based, and avoid pinning child `agent:` models or a
  single runtime profile.
- Prepared a three-item dry-run manifest:
  `~/.local/state/transcribe-audio/auracall-batches/dispatch-team-prepare-20260518-194519.json`.
- Submitted a three-item live batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-194548.json`.
- Batch id: `batch_4f25217d09324207a48607164dcb5451`.

Validation:

- The dry-run manifest used model `gpt-5.2-pro`, top-level dispatch team
  `transcribe-audio-chatgpt-pro-pool`, `projectSync=none`, and child
  AuraCall hints containing only `service=chatgpt`, `transport=browser`, and
  the team id.
- The live batch dispatched one child to each member:
  `pro-extended-chatgpt-consult-transcripts` on `wsl-chrome-2`,
  `pro-extended-chatgpt-soylei-transcripts` on `wsl-chrome-3`, and
  `pro-extended-chatgpt-ecochran76-personal-transcripts` on `wsl-chrome-4`.
- `wsl-chrome-2` and `wsl-chrome-3` completed with AuraCall
  `response-complete` browser evidence and materialized readouts:
  `~/.transcripts/legacy-artifacts/de/de3853512ea62d394316-2026-03-03 Danielle re Nacu Soylei impersonation.readout.json`
  and
  `~/.transcripts/legacy-artifacts/43/438732c3e310b43f7994-2026-02-12 Jason Potter Eagle Outdoor Recording (11).readout.json`.
- `python scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-194548.json --format text`
  passed with `2 pass, 0 warn, 0 fail`.
- Focused tests passed:
  `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest tests/test_transcript_store.py::test_auracall_first_pass_prepare_can_use_dispatch_team -q`.
- `py_compile` passed for the batch script and test module.

Notes:

- The batch status is `failed` overall because the `wsl-chrome-4` child failed
  before submit with `Unable to find the Thinking time dropdown menu`.
- AuraCall diagnostics showed `wsl-chrome-4` selected `Pro / Standard` and did
  not expose the expected `Extended` thinking-time control, while the other two
  accounts selected `Pro / Extended`.
- Project sync is intentionally disabled for this team type, so differences in
  existing `Transcripts` project instructions, files, or settings are an
  operator-visible consistency risk, not an execution error.

Next:

- Fix or quarantine the `wsl-chrome-4` Pro Extended selector mismatch before
  scaling first-pass summary batches through the three-account pool. Until
  then, use the two proven agents or a selector all three accounts expose
  consistently.

## Turn 111 | 2026-05-19

Summary: Resumed first-pass summary scaling on the known-good SoyLei
Transcripts agent after validating AuraCall's response-batch recovery
diagnostics.

Action:

- Verified current AuraCall commits include the restart/cancel recovery fix and
  finalizing-state surfacing:
  `fb2bd271 Fix response batch restart recovery semantics` and
  `f14ac6c6 Surface finalizing response-batch state`.
- Ran targeted AuraCall regression tests:
  `pnpm vitest run tests/runtime.store.test.ts tests/runtime.responseBatchService.test.ts --maxWorkers 1`,
  `pnpm vitest run tests/runtime.serviceHost.test.ts -t "cancels a running run after its lease was already released|cancels a planned run before it has an active lease|cancels an active run owned" --maxWorkers 1`,
  and
  `pnpm vitest run tests/runtime.responsesService.test.ts -t "projects lease and passive runtime diagnostics|finalizing|detached browser response" --maxWorkers 1`.
- Re-read batch `batch_baec0e7666d143a283a01a4f4828507d`; all three jobs now
  report `runtimeState=terminal` with bounded lease/provider diagnostics.
- Re-materialized the dispatch-pool dogfood batch
  `batch_4f25217d09324207a48607164dcb5451`; the two successful children stayed
  materialized and the `wsl-chrome-4` child remained failed on the Pro Extended
  selector mismatch.
- Prepared a two-item single-agent dry-run with dispatch-team env disabled:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-195603.json`.
- Submitted the corresponding two-item live batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-195628.json`.
- Batch id: `batch_9771bfe9f76f418bbbc0b8cde0918973`.
- Model: `agent:pro-extended-chatgpt-soylei-transcripts`.
- Dispatch team: none.
- Limits: `maxConcurrentRuns=1`,
  `maxBrowserInteractionsPerMinute=8`.

Validation:

- AuraCall targeted tests passed: 10 response-batch/store tests, 3
  service-host cancel tests, and 2 response-service diagnostics tests.
- The live two-item batch progressed with explicit operator states:
  job 0 moved from `running` to `finalizing` to `terminal`; job 1 moved from
  `queued` to `running` to `finalizing` to `terminal`.
- During finalization, jobs surfaced `browserTaskState=response-complete` and
  high-confidence `chatgpt-response-finished` provider evidence instead of a
  bare ambiguous `in_progress` state.
- Final counts for `batch_9771bfe9f76f418bbbc0b8cde0918973`:
  `total=2`, `completed=2`, `in_progress=0`, `failed=0`, `cancelled=0`, and
  `missing=0`.
- Materialized readouts:
  `~/.transcripts/legacy-artifacts/d6/d651c4ad15666ba39b1e-IMCD call.readout.json`
  and
  `~/.transcripts/legacy-artifacts/c4/c4f9ece2a93d4397b3c9-2026-01-13 11-00 Meet with Eric (ryan jaggar) My recording 150.readout.json`.
- `scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-195628.json --format text`
  passed with `2 pass, 0 warn, 0 fail`.
- `auracall-api.service`, `transcribe-watch.service`, and
  `transcripts.service` were all active after the run.

Notes:

- The AuraCall recovery/finalization status surface is now good enough for
  cautious single-agent scaling: active work is distinguishable from queued,
  finalizing, and terminal work without raw runtime-record inspection.
- The dispatch-pool path remains blocked by `wsl-chrome-4` model/depth selector
  parity, not by response-batch recovery.

Next:

- Continue first-pass summaries in small single-agent batches or a two-member
  pool excluding `wsl-chrome-4`. Do not use the three-member Pro Extended pool
  until `wsl-chrome-4` proves the same Extended thinking-control surface or is
  moved to a model selector it actually exposes.

## Turn 112 | 2026-05-19

Summary: Retried the three-member AuraCall dispatch pool after the
`wsl-chrome-4` Pro Extended selector fix; all three first-pass summaries
completed and materialized.

Action:

- Verified AuraCall tenant limits before submission:
  `maxConcurrentChats=4`, `maxChatsPerHour=120`, `maxChatsPerDay=240`, and
  `activeChats=0`.
- Confirmed the scoped client env still targets
  `AURACALL_DISPATCH_TEAM=transcribe-audio-chatgpt-pro-pool` with
  `AURACALL_MODEL=gpt-5.2-pro` and `AURACALL_DISPATCH_MODEL=gpt-5.2-pro`.
- Submitted a three-item live batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-210945.json`.
- Batch id: `batch_6a07fcef576343a1a6c053ba849f2029`.
- Limits: `maxConcurrentRuns=3`,
  `maxBrowserInteractionsPerMinute=6`.

Validation:

- The batch dispatched one child to each member:
  `pro-extended-chatgpt-consult-transcripts` on `wsl-chrome-2`,
  `pro-extended-chatgpt-soylei-transcripts` on `wsl-chrome-3`, and
  `pro-extended-chatgpt-ecochran76-personal-transcripts` on `wsl-chrome-4`.
- AuraCall status showed one Chrome target and one ChatGPT conversation URL per
  running prompt, with passive DOM evidence driving runtime state:
  `thinking`, then `response-complete` or `response-incoming`.
- Final counts: `total=3`, `completed=3`, `in_progress=0`, `failed=0`,
  `cancelled=0`, `missing=0`.
- Materialized readouts:
  `~/.transcripts/legacy-artifacts/04/045cc4c8de501a0c1b56-20250811_Reynolds_Transcript.readout.json`,
  `~/.transcripts/legacy-artifacts/14/14826df1928d0956a900-2026-04-07 13-30 Scott SoyLei 2026-04-07 Scott Roberts Nacu Austin Matter.readout.json`,
  and
  `~/.transcripts/legacy-artifacts/2f/2fad3db6017f321f7350-2026-04-14 Scott Chris C-D letter response follow up.readout.json`.
- `scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-210945.json --format text`
  passed with `3 pass, 0 warn, 0 fail`.
- Live first-pass summary queue now reports 2 pending items.
- While verifying post-run tenant usage, AuraCall initially counted four chat
  starts for the three-request batch because `wsl-chrome-3` recovery replay
  wrote a second `step-started` event for the same response step. The recovered
  run reattached to the same Chrome target and same conversation URL, then
  succeeded. AuraCall was patched and reinstalled so tenant and batch
  rate-limit usage dedupe repeated `step-started` events for the same response
  step.
- After reinstall, AuraCall `/status?tenantExecutionLimits=usage` reported
  `activeChats=0`, `chatsLastHour=4` total, and per-runtime last-hour starts:
  `wsl-chrome-2=1`, `wsl-chrome-3=1`, `wsl-chrome-4=2` including the earlier
  selector smoke.

Notes:

- The dispatch-pool path is now viable for cautious three-account dogfooding on
  the `Transcripts` project.
- `projectSync=none` remains an intentional consistency risk: AuraCall does not
  reconcile project instructions, files, or settings between the tenants.

Next:

- Clear the final two pending first-pass summaries with the dispatch pool in a
  small controlled batch, then re-run the quality gate and record the final
  queue state.

## Turn 113 | 2026-05-19

Summary: Cleared the final first-pass summaries through the AuraCall
dispatch-pool path and recorded the AuraCall lease/finalization repair exposed
by the retry.

Action:

- Submitted a two-item dispatch-pool batch:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-213213.json`.
- Batch id: `batch_984676dc867047ed8b49d0f86d8304c8`.
- Limits: `maxConcurrentRuns=2`,
  `maxBrowserInteractionsPerMinute=6`.
- Materialized the completed child from that batch.
- Submitted a one-item dispatch-pool retry for the remaining failed readout:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-215319.json`.
- Batch id: `batch_47814beb6daf4478aebfcc0e32130f84`.
- Limits: `maxConcurrentRuns=1`,
  `maxBrowserInteractionsPerMinute=6`.
- Materialized the retry output and stored the readout artifacts.

Validation:

- The two-item batch dispatched through
  `transcribe-audio-chatgpt-pro-pool` with `projectSync=none`.
- Final two-item batch counts: `total=2`, `completed=1`, `failed=1`,
  `cancelled=0`, `missing=0`, `in_progress=0`.
- The completed two-item child ran on
  `pro-extended-chatgpt-soylei-transcripts` / `wsl-chrome-3` and
  materialized:
  `~/.transcripts/legacy-artifacts/23/23f91fc16f5906129168-2026-04-24 16-00 Soylei 2026-04-24 Eric Cara conference.readout.json`.
- The failed two-item child ran on
  `pro-extended-chatgpt-consult-transcripts` / `wsl-chrome-2` and failed under
  the older AuraCall runtime with
  `runner_execution_failed: Stale ChatGPT assistant response detected after send.`
- The one-item retry dispatched to
  `pro-extended-chatgpt-consult-transcripts` / `wsl-chrome-2` and completed.
- Final one-item retry counts: `total=1`, `completed=1`, `failed=0`,
  `cancelled=0`, `missing=0`, `in_progress=0`.
- Retry materialized:
  `~/.transcripts/legacy-artifacts/e0/e067625711f67972d5de-2026-04-14 Joe Domino Follow up SABER Chemical.readout.json`.
- `scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-213213.json --format text`
  passed with `1 pass, 0 warn, 0 fail`.
- `scripts/check_readout_quality.py --manifest ~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260518-215319.json --format text`
  passed with `1 pass, 0 warn, 0 fail`.
- `transcript_store.py first-pass-summary-queue --format compact-json --limit 5`
  reports `selected_count=0`.
- `transcripts.service`, `transcribe-watch.service`, and
  `auracall-api.service` are active.

Notes:

- The final one-item retry succeeded, but it exposed an AuraCall artifact
  finalization window where provider evidence was `response-complete` and the
  run later succeeded after the lease had already expired. AuraCall was patched
  and reinstalled afterward to allocate restart-safe unique lease ids and emit
  `browser-response-artifact-finalizing` runtime evidence while generated
  artifacts are materializing.
- The dispatch-pool team still uses `projectSync=none`, so project instruction
  or file drift between the three ChatGPT tenants remains an operator-visible
  consistency risk rather than a runtime error.

Next:

- Move from first-pass generation to review/triage of the materialized
  readouts. Keep any future transcript batches small enough to observe the
  newly installed AuraCall finalization heartbeat before raising concurrency.

## Turn 114 | 2026-05-19

Summary: Wrote the backend roadmap handoff and audited the live P09 backend
state after the first-pass summary queue cleared.

Action:

- Added
  `docs/dev/notes/2026-05-19-backend-roadmap-audit-handoff.md`.
- Updated `ROADMAP.md` so P09 reflects the current backend API surface,
  first-pass summary batch actions, live store counts, and cleared first-pass
  queue.
- Updated `docs/dev/plans/0009-2026-05-12-react-vite-review-console.md` so
  the P09 implementation slice includes first-pass summary
  prepare/submit/status operations rather than only read-only queue
  aggregation.
- Fixed the transcript API first-pass prepare endpoint so its internal
  AuraCall batch namespace includes the newer dispatch-team field and honors
  dispatch-team defaults from the configured env file.

Validation:

- Graphiti discovery was healthy but returned mostly older roadmap facts, so
  repo docs, source, live API responses, and SQLite counts were used as the
  authority.
- Live `GET /api/health` returned `status=ok` for
  `/home/ecochran76/.transcripts/transcripts.sqlite3`.
- Live store counts: 240 documents, 164 transcripts, 74 readouts,
  2 contextual readouts, 122 blobs, 144 document-blob links, and
  6560 chunk rows.
- Live `GET /api/review-queue?limit=100` returned `total_open=0`.
- Live `transcript_store.py first-pass-summary-queue --format compact-json --limit 5`
  returned `selected_count=0`.
- Live `GET /api/search?q=SoyLei&limit=3` returned ranked readout results.
- `transcripts.service` and `transcribe-watch.service` were active.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py tests/test_review_queue_maintenance.py -q`
  passed.
- `python -m py_compile transcript_api.py transcript_store.py review_queue_maintenance.py`
  passed.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Browser smoke against `http://transcripts.localhost/` opened the conversation
  workspace for the 2026-04-24 Cara readout, confirmed the modal title,
  section nav labels, human summary text, inherited audio source
  `/api/blobs/f363a21cdc716795bd3c`, planned disabled action labels, and Escape
  close behavior.
- `git diff --check` passed.

Next:

- Start the P09 contact/speaker backend slice: add contact, identity,
  speaker-assignment, and merge-audit tables plus reviewed API contracts before
  adding more UI chrome.

## Turn 115 | 2026-05-19

Summary: Added Codex app-server as the supervised App Intelligence surface for
the transcript console.

Action:

- Added a read-only intelligence provider registry at
  `GET /api/intelligence/providers`.
- Added `codex-app-server` readiness checks for the configured Codex binary,
  `codex --version`, `codex app-server --help`, and protocol generation help.
- Marked `codex-app-server` as the default supervised control plane for
  persistent, branchable, replayable App Intelligence runs while preserving
  `codex exec` as the stateless leaf-job provider.
- Added `--codex-bin` and `TRANSCRIPTS_CODEX_BIN` so systemd services do not
  depend on interactive-shell PATH.
- Installed a user-service override at
  `~/.config/systemd/user/transcripts.service.d/10-codex-bin.conf` pointing to
  the current Codex CLI.
- Updated `README.md`, `ROADMAP.md`, the P09 plan, and
  `docs/dev/transcript-review-api.md`.

Validation:

- `graphiti-runtime doctor` returned healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main ...` returned
  older intelligence-provider memory, so repo docs and live checks remained
  authoritative.
- `python app-intelligence-automation/scripts/check_codex_app_server.py --json`
  reported `codex-cli 0.131.0` with app-server, schema generation, TypeScript
  generation, Unix transport, WebSocket transport, and WebSocket auth support.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed.
- `.venv/bin/python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; live
  `GET /api/intelligence/providers` returned `default_supervisor=codex-app-server`
  and `codex-app-server.status=ready` using
  `/home/ecochran76/.nvm/versions/node/v24.13.0/bin/codex`.

Next:

- Add the P09 App Intelligence run-ledger schema under
  `~/.local/state/transcribe-audio/` before enabling app-server-backed branch,
  rollback, or write-bearing workflow phases from the UI.

## Turn 116 | 2026-05-19

Summary: Added prepared App Intelligence run ledgers for future
app-server-backed workflows.

Action:

- Added `app_intelligence_ledger.py` with a user-scoped run directory contract
  under `~/.local/state/transcribe-audio/app-intelligence-runs/<run_id>/`.
- Prepared ledgers now include `run.json`, `events.jsonl`,
  `codex_events.jsonl`, `branches/`, `artifacts/`, and `diffs/`.
- `run.json` records schema version, workflow, purpose, document id, provider,
  phase/status, branch placeholders, Codex thread placeholders, RNG seed
  ledger, allowed actions, approval policy, eval policy, artifact registry, and
  final decision slot.
- Added `GET /api/intelligence/runs`, `GET /api/intelligence/runs/<run_id>`,
  and `POST /api/intelligence/runs/prepare` to `transcript_api.py`.
- The prepare endpoint creates only the local ledger. It does not start
  `codex app-server`, create Codex threads, run model turns, fork branches, or
  perform external writes.
- Updated `README.md`, `ROADMAP.md`, the P09 plan, and
  `docs/dev/transcript-review-api.md`.

Validation:

- `graphiti-runtime doctor` returned healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main ...` returned
  older roadmap/intelligence facts only, so repo docs and skill references
  remained authoritative.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `.venv/bin/python -m py_compile app_intelligence_ledger.py transcript_api.py`
  passed.
- `git diff --check` passed.
- Temp-state CLI smoke created and listed a prepared `smoke-run` ledger.
- Restarted `transcripts.service`; live
  `GET /api/intelligence/runs?limit=5` returned the user-scoped runs directory
  and `total=0` without creating live state.
- Live `GET /api/intelligence/providers` still reported
  `codex-app-server.status=ready`.

Next:

- Add structured-decision validation and event-append endpoints so future
  app-server turns can record accepted/rejected decisions before the host
  executes any fork, rollback, write, or external apply action.

## Turn 117 | 2026-05-20

Summary: Centralized intelligence task selection into one library.

Action:

- Added `intelligence_config.py` as the central task-based intelligence routing
  library.
- Defined task ids for `first_pass_summary`, `contextual_reread`,
  `context_source_ranking`, `route_selection`, `speaker_disambiguation`,
  `memory_harvest_review`, `embedding`, and `app_supervisor`.
- Added resolution order: built-in defaults, optional
  `~/.local/state/transcribe-audio/intelligence.config.json` or
  `TRANSCRIPTS_INTELLIGENCE_CONFIG`, per-task environment variables, then
  explicit CLI/API overrides.
- Wired `summarize_transcript.py` and `contextual_reread.py` through the new
  library while preserving current default provider behavior.
- Added `GET /api/intelligence/config` so the operator UI can show resolved
  task routing.
- Updated `README.md`, `ROADMAP.md`, the P09 plan, and
  `docs/dev/transcript-review-api.md`.

Validation:

- `graphiti-runtime doctor` returned healthy.
- `graphiti-runtime discover --group-id transcribe_audio_main ...` returned
  older intelligence-provider facts only, so repo docs and current source were
  used as authority.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py tests/test_readouts.py tests/test_transcript_api.py -q`
  passed.
- `.venv/bin/python -m py_compile intelligence_config.py summarize_transcript.py contextual_reread.py transcript_api.py`
  passed.
- `git diff --check` passed.
- `python intelligence_config.py show` printed all resolved default task routes.
- Restarted `transcripts.service`; live `GET /api/intelligence/config` returned
  the resolved default routing from the user-scoped config path.

Next:

- Add write-preview/update endpoints for `intelligence.config.json` so the UI
  can edit task routing safely with validation, diffs, and rollback metadata.

## Turn 118 | 2026-05-20

Summary: Added reviewed preview/apply updates for intelligence task routing.

Action:

- Added validated task update preview/apply helpers to `intelligence_config.py`.
- Added CLI commands:
  - `python intelligence_config.py preview-update ...`
  - `python intelligence_config.py apply-update ... --approval-token APPLY_INTELLIGENCE_CONFIG_UPDATE`
- Added API endpoints:
  - `POST /api/intelligence/config/preview`
  - `POST /api/intelligence/config/apply`
- Apply writes only to the resolved user-scoped intelligence config path and
  requires `approval_token=APPLY_INTELLIGENCE_CONFIG_UPDATE`.
- Preview returns before/after config, resolved task values, and rollback
  metadata without writing.
- Made `intelligence_config.py` standalone by removing the `transcribe_common`
  import so config inspection works outside the full virtualenv dependency set.
- Updated `README.md` and `docs/dev/transcript-review-api.md`.

Validation:

- `graphiti-runtime doctor` returned healthy.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile intelligence_config.py` and
  `.venv/bin/python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- CLI preview smoke returned `will_write=false`, rollback metadata, and
  `resolved_after.provider=codex-exec`.
- Restarted `transcripts.service`; live
  `POST /api/intelligence/config/preview` returned a non-mutating preview.
- Confirmed no live `~/.local/state/transcribe-audio/intelligence.config.json`
  file was created by preview.

Next:

- Wire the React Intelligence panel to list providers, show resolved task
  routing, preview edits, and require explicit apply approval for config writes.

## Turn 119 | 2026-05-20

Summary: Wired the React Intelligence panel to the central provider and task
routing APIs.

Action:

- Loaded `GET /api/intelligence/providers` and `GET /api/intelligence/config`
  during review-console startup.
- Added an Intelligence left-pane task selector, central task editor, provider
  status cards, resolved-route table, and right-pane inspector.
- Added reviewed config editing from the UI:
  - Preview calls `POST /api/intelligence/config/preview` and displays
    rollback metadata without writing config.
  - Apply prompts the operator and calls `POST /api/intelligence/config/apply`
    with `approval_token=APPLY_INTELLIGENCE_CONFIG_UPDATE`.
- Kept fallback demo data for offline preview mode and improved API error
  handling for JSON error bodies.
- Added focused CSS for task routing, provider status, preview metadata, and
  responsive Intelligence layouts.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery for this slice returned
  only older repo facts, so repo source and live API evidence were used.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_intelligence_config.py tests/test_transcript_api.py -q`
  passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active.
- Live `GET /api/intelligence/config` returned
  `schema_version=transcribe-audio.intelligence-config.v1` with
  `first_pass_summary.provider=openai-compatible` and
  `app_supervisor.provider=codex-app-server`.
- Live `POST /api/intelligence/config/preview` returned a non-writing preview
  with rollback metadata and `resolved_after.provider=codex-exec`.
- Confirmed preview did not create
  `~/.local/state/transcribe-audio/intelligence.config.json`.

Next:

- Add provider-specific readiness/detail affordances and a UI action to prepare
  supervised app-intelligence run ledgers from selected tasks.

## Turn 120 | 2026-05-20

Summary: Added provider-detail and App Intelligence ledger controls to the
React Intelligence panel.

Action:

- Extended the Intelligence panel startup load to include
  `GET /api/intelligence/runs?limit=8`.
- Added provider-detail rendering for control plane, readiness, version,
  capabilities, provider notes, and readiness check names.
- Added a `Prepare run ledger` action that calls
  `POST /api/intelligence/runs/prepare` with the selected task, selected
  document id when present, and `created_by=review-console`.
- Refreshed the recent run-ledger list after a successful prepare.
- Added recent prepared-ledger cards to the Intelligence center viewport.
- Updated P09 roadmap/plan text to record the Intelligence UI state and the
  boundary that ledger preparation starts no provider work.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_intelligence_config.py tests/test_transcript_api.py -q`
  passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active.
- Live `GET /api/intelligence/providers` reported
  `codex-app-server.status=ready`, `ready=true`, `version=codex-cli 0.131.0`,
  and 4 readiness checks.
- Live `GET /api/intelligence/runs?limit=3` returned the user-scoped runs
  directory with `total=0` before any UI-prepared ledger was created.
- Live `/` served the rebuilt assets
  `index-D-TOAr_Q.js` and `index-BWeLD49c.css`.

Next:

- Add a selected-run inspector/detail view that can open one prepared ledger,
  show events/policy/paths, and establish the next explicit approval gate
  before any app-server session starts.

## Turn 121 | 2026-05-20

Summary: Added a read-only selected-run inspector for App Intelligence ledgers.

Action:

- Added selected-run state to the React Intelligence panel.
- Fetches `GET /api/intelligence/runs/<run_id>?event_limit=12` when a prepared
  run is selected.
- Turns recent run ledger rows into selectable controls.
- Shows selected ledger details in the right inspector:
  - workflow, run id, phase, provider, linked document, and ledger path;
  - allowed actions and remote-transport policy;
  - approval/eval policy JSON for the next gate;
  - recent host ledger events.
- Preserved the boundary that inspection is read-only and starting an
  app-server session remains a future separate approval-gated action.
- Updated P09 roadmap/plan text for the selected-run inspector state.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active after readiness polling.
- Live `GET /api/intelligence/runs?limit=8` returned the user-scoped runs
  directory with `total=0`.
- Live `/` served the rebuilt assets
  `index-D4eizsUg.js` and `index-DICDSuDL.css`.

Next:

- Add the backend/session-start preflight contract as a disabled or dry-run
  surface first: validate provider readiness, ledger phase, approval token
  shape, and event append semantics without starting Codex app-server work.

## Turn 122 | 2026-05-20

Summary: Added non-starting App Intelligence session-start preflight.

Action:

- Added `session_start_preflight()` to `app_intelligence_ledger.py`.
- Added `POST /api/intelligence/runs/<run_id>/session-start-preflight`.
- Dry-run preflight validates:
  - ledger exists and remains in `phase=prepared`;
  - provider is `codex-app-server`;
  - provider readiness is true;
  - `start_app_server_session` is allowed by the ledger policy;
  - host-owned control flow and structured decisions are required;
  - approval-token shape matches the future session-start or preflight-event
    token.
- Optional event-append mode records only `session_start_preflight` and
  requires `approval_token=APPEND_SESSION_START_PREFLIGHT_EVENT`.
- The future session-start token is surfaced as
  `START_APP_SERVER_SESSION`, but no endpoint starts app-server sessions yet.
- Wired the React selected-run inspector with `Dry-run preflight` and
  `Record preflight event` controls.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active.
- Live `/` served rebuilt assets `index-BlMXf_Sl.js` and
  `index-DICDSuDL.css`.
- Live `GET /api/intelligence/providers` reported
  `codex-app-server.status=ready`, `ready=true`, and
  `version=codex-cli 0.131.0`.
- Live `GET /api/intelligence/runs?limit=8` returned the user-scoped runs
  directory with `total=0`; no live ledger was created for this smoke.

Next:

- Add the first real session-start implementation behind
  `approval_token=START_APP_SERVER_SESSION`, restricted to prepared ledgers,
  stdio/unix transport only, and host-owned event capture before any model turn.

## Turn 123 | 2026-05-20

Summary: Added approved App Intelligence control-plane session start.

Action:

- Added ledger helpers to record app-server session start request, failure, and
  started events.
- Added `POST /api/intelligence/runs/<run_id>/session-start`.
- Session start requires:
  - prepared ledger phase;
  - `approval_token=START_APP_SERVER_SESSION`;
  - provider readiness passing;
  - `transport` set to `stdio` or `unix`;
  - host-owned control-flow and structured-decision policy.
- The endpoint writes `app_server_session_start_requested` before starting
  anything.
- The endpoint starts only the managed Codex app-server control-plane daemon
  with `codex app-server daemon start`.
- The endpoint records daemon version metadata, updates the ledger to
  `phase=session_started`, and appends `app_server_session_started`.
- No Codex thread id is created and `will_start_model_turn=false` remains part
  of the API/ledger boundary.
- Added a React `Start control plane` action behind an explicit confirmation.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live CLI/API evidence were used.
- `codex app-server --help`, `codex app-server daemon --help`, and
  `codex app-server proxy --help` showed local daemon/proxy support.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active after readiness polling.
- Live `/` served rebuilt assets `index-B6GvXCMc.js` and
  `index-DICDSuDL.css`.
- Live `GET /api/intelligence/providers` reported
  `codex-app-server.status=ready`, `ready=true`, and
  `version=codex-cli 0.131.0`.
- Live `GET /api/intelligence/runs?limit=8` returned the user-scoped runs
  directory with `total=0`; the live session-start endpoint was not invoked to
  avoid creating operator state or starting the daemon outside a selected
  ledger.

Next:

- Add the first model-turn preflight, not execution: generate the initial
  app-server prompt packet from a selected document plus task route and require
  review before any prompt is sent.

## Turn 124 | 2026-05-20

Summary: Added reviewed App Intelligence model-turn prompt-packet preflight.

Action:

- Added `prepare_model_turn_packet()` to `app_intelligence_ledger.py`.
- Added `POST /api/intelligence/runs/<run_id>/model-turn-preflight`.
- Model-turn preflight requires:
  - `phase=session_started`;
  - ledger policy allowing `prepare_prompt`;
  - `approval_token=PREPARE_MODEL_TURN_PREFLIGHT`;
  - a stored transcript/readout document id.
- The endpoint resolves the selected task route from `intelligence_config.py`,
  compacts the selected stored document, and builds an initial prompt packet.
- Prompt packet artifacts are written under
  `artifacts/prompt-packets/<packet_id>.json` and
  `artifacts/prompt-packets/<packet_id>.prompt.txt`.
- The ledger records `prompt_packets[]` metadata and appends
  `model_turn_preflight_prepared`.
- The packet surfaces future `SEND_APP_SERVER_MODEL_TURN` approval but does not
  use it; `will_send_prompt=false`.
- Added a React `Prepare prompt packet` action and prompt-packet list in the
  selected-run inspector.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active after readiness polling.
- Live `/` served rebuilt assets `index-9Z0DRraC.js` and
  `index-DICDSuDL.css`.
- Live `GET /api/intelligence/providers` reported
  `codex-app-server.status=ready`, `ready=true`, and
  `version=codex-cli 0.131.0`.
- Live `GET /api/intelligence/runs?limit=8` returned the user-scoped runs
  directory with `total=0`; no live model-turn preflight was invoked to avoid
  creating operator prompt artifacts outside a selected run.

Next:

- Add a prompt-packet review/apply surface that can inspect packet JSON/text
  and require `approval_token=SEND_APP_SERVER_MODEL_TURN` before any app-server
  prompt send implementation is added.

## Turn 125 | 2026-05-20

Summary: Added read-only App Intelligence prompt-packet review.

Action:

- Added `read_model_turn_packet()` to `app_intelligence_ledger.py`.
- Added `GET /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>`.
- The endpoint reads existing packet JSON and prompt text from the run's
  `artifacts/prompt-packets/` directory only.
- The review payload returns `will_send_prompt=false` and surfaces the future
  `SEND_APP_SERVER_MODEL_TURN` token without using it.
- The React selected-run inspector now selects prompt packets, loads packet
  metadata, and shows the full prompt text in a review preview.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active after readiness polling.
- Live `/` served rebuilt assets after restart.
- Live `GET /api/intelligence/providers` reported the configured providers.
- Live `GET /api/intelligence/runs?limit=8` returned the user-scoped runs
  directory; no live prompt packet was created.

Next:

- Introduce `SEND_APP_SERVER_MODEL_TURN` as a dry-run send preflight first, or
  add the actual send implementation only after packet review approval.

## Turn 126 | 2026-05-20

Summary: Added non-sending App Intelligence model-turn send preflight.

Action:

- Added `model_turn_send_preflight()` to `app_intelligence_ledger.py`.
- Added `send_model_turn` to the default allowed action list for newly
  prepared ledgers.
- Added
  `POST /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>/send-preflight`.
- The endpoint requires `approval_token=SEND_APP_SERVER_MODEL_TURN`, validates
  ledger phase, host-owned policy, structured decisions, packet/run match,
  review requirement, unsent state, and prompt text presence.
- The endpoint returns `will_send_prompt=false` and `will_write_event=false`;
  it does not start a Codex thread, send a prompt, mutate packet state, or
  append a ledger event.
- Added a React `Dry-run send preflight` action in the packet review surface.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and live API evidence were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.

Next:

- Add the real model-turn send implementation behind the existing
  `SEND_APP_SERVER_MODEL_TURN` gate, with event capture and no write-bearing
  downstream action until structured-decision validation exists.

## Turn 127 | 2026-05-20

Summary: Added gated Codex app-server model-turn send.

Action:

- Added `codex_app_server_client.py`, a minimal JSON-RPC stdio/proxy client
  for `initialize`, `thread/start`, and `turn/start`.
- Added ledger helpers for captured Codex events, model-turn started state,
  and model-turn send failure events.
- Added
  `POST /api/intelligence/runs/<run_id>/prompt-packets/<packet_id>/send`.
- The send endpoint requires `approval_token=SEND_APP_SERVER_MODEL_TURN`, runs
  the existing send preflight, starts or reuses a Codex thread, starts one
  turn with the reviewed packet prompt, records captured app-server events in
  `codex_events.jsonl`, marks the packet sent, and stores Codex thread/turn ids
  in `run.json`.
- The endpoint returns `will_execute_downstream_action=false`; it does not
  parse model output into host decisions, fork, rollback, write memory, apply a
  route, write a repository, or perform deposition actions.
- Added a React `Send reviewed packet` action behind the packet review surface.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- Used the `app-intelligence-automation` skill and generated local Codex
  app-server protocol schemas under `/tmp` to confirm JSON-RPC method shapes.
- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source, generated protocol schemas, and local skill
  references were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` are active.
- Live `/api/review-queue?limit=10` returned
  `app_intelligence_human_review` with `count=0`,
  `pending_apply_count=0`, and `needs_review_count=0`.

Next:

- Add a turn-status/readout endpoint that can inspect a started Codex turn and
  capture completion/output without executing structured decisions.

## Turn 128 | 2026-05-20

Summary: Added Codex model-turn status capture.

Action:

- Added `inspect_model_turn()` to `codex_app_server_client.py` using
  `thread/read`, `thread/turns/list`, and `thread/turns/items/list`.
- Added `record_model_turn_status()` to `app_intelligence_ledger.py`.
- Added `POST /api/intelligence/runs/<run_id>/turn-status`.
- The endpoint requires `approval_token=CAPTURE_MODEL_TURN_STATUS`, reads the
  active Codex thread/turn from the run ledger by default, captures status and
  output into `artifacts/model-turn-readouts/<turn_id>.status.json`, appends a
  `model_turn_status_captured` event, and records any app-server events.
- The endpoint returns `will_execute_structured_decision=false`; it does not
  parse output into a host decision, fork, rollback, route, memory write,
  repository write, or deposition action.
- Added a React `Capture turn status` action and latest-status display in the
  packet review surface.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source, generated protocol schemas, and local skill
  references were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Add structured-decision schema validation for captured turn output before
  enabling any host action from model results.

## Turn 129 | 2026-05-20

Summary: Added structured-decision validation for captured Codex output.

Action:

- Added `STRUCTURED_DECISION_VALIDATE_TOKEN=VALIDATE_STRUCTURED_DECISION`.
- Added host-owned decision parsing and validation to
  `app_intelligence_ledger.py`.
- Added
  `POST /api/intelligence/runs/<run_id>/structured-decision/validate`.
- The validator reads the latest captured model-turn status artifact, extracts
  a JSON object, validates `action`, `rationale`, `confidence`,
  `review_flags`, and fork-specific fields, writes a validation artifact under
  `artifacts/structured-decisions/`, appends a
  `structured_decision_validated` event, and records accepted/rejected decision
  metadata in `run.json`.
- Allowed decision actions are `continue_current_branch`, `fork_branches`,
  `rollback`, `stop`, and `ask_for_human_review`.
- The endpoint returns `will_execute_host_action=false`; no fork, rollback,
  write, memory harvest, route apply, or deposition action is executed.
- Added a React `Validate structured decision` action and result display.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned only older
  broad repo facts, so repo source and local skill references were used.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 24 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Add explicit apply endpoints for individual validated decision actions,
  starting with no-op `stop` and `ask_for_human_review` records before any
  fork/rollback/write-bearing host action.

## Turn 130 | 2026-05-20

Summary: Added ledger-only structured-decision apply records.

Action:

- Added `STRUCTURED_DECISION_APPLY_TOKEN=APPLY_STRUCTURED_DECISION`.
- Added `apply_validated_structured_decision()` to require an already validated
  decision and accept only `stop` or `ask_for_human_review`.
- Added
  `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/apply`.
- The apply path writes
  `artifacts/structured-decisions/<decision_id>.apply.json`, updates the local
  run ledger, appends `structured_decision_applied`, and marks the run
  `stopped` or `needs_human_review`.
- The apply path returns explicit non-action flags:
  `will_execute_external_action=false`,
  `will_execute_write_bearing_action=false`, and
  `will_fork_or_rollback=false`.
- Added a React `Apply ledger-only decision` control for validated `stop` and
  `ask_for_human_review` decisions.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned older broad
  repo facts only, so repo source remained authoritative.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Make `ask_for_human_review` visible in the Review Queue before adding any
  branch, rollback, or write-bearing apply action.

## Turn 131 | 2026-05-20

Summary: Surfaced App Intelligence human-review decisions in the Review Queue.

Action:

- Added a read-only App Intelligence human-review bucket to
  `/api/review-queue`.
- The bucket scans user-scoped App Intelligence run ledgers under
  `~/.local/state/transcribe-audio/app-intelligence-runs/` and includes
  validated or ledger-applied `ask_for_human_review` decisions.
- Mixed review items now include App Intelligence rows with run id, document
  id, decision id, decision status, and validation/apply artifact paths.
- Updated the React Review Queue surface so the central list is not route-only
  and the right inspector reports App Intelligence review counts.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned older broad
  repo facts only, so repo source remained authoritative.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Add a review-detail/action surface for App Intelligence human-review items so
  the operator can resolve, annotate, or reopen a ledger-only decision without
  enabling fork, rollback, memory, routing, or repository writes.

## Turn 132 | 2026-05-20

Summary: Added local App Intelligence human-review actions.

Action:

- Added `HUMAN_REVIEW_DECISION_TOKEN=RECORD_HUMAN_REVIEW_DECISION`.
- Added `record_human_review_decision()` for `annotate`, `resolve`, and
  `reopen` actions on `ask_for_human_review` decisions.
- Added
  `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/human-review`.
- Human-review actions update only `run.json`, append
  `human_review_decision_recorded`, and return explicit no-action flags for
  external actions, write-bearing actions, fork, and rollback.
- Updated `/api/review-queue` so resolved App Intelligence human-review items
  remain visible as resolved rows but do not count as open.
- Added Review Queue UI buttons for annotating, resolving, and reopening App
  Intelligence human-review items.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned older broad
  repo facts only, so repo source remained authoritative.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Add a guarded branch/fork preflight endpoint for App Intelligence that
  computes what would happen for `fork_branches` without creating threads,
  modifying branches, or running provider work.

## Turn 133 | 2026-05-20

Summary: Added App Intelligence fork-branches preflight.

Action:

- Added `FORK_BRANCHES_PREFLIGHT_TOKEN=PREVIEW_FORK_BRANCHES`.
- Added `preflight_fork_branches()` for validated `fork_branches` decisions.
- Added
  `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/fork-preflight`.
- The preflight writes
  `artifacts/structured-decisions/<decision_id>.fork-preflight.json`, appends
  `fork_branches_preflight`, and returns planned branch records.
- The preflight explicitly returns `will_create_thread=false`,
  `will_modify_branches=false`, `will_run_provider=false`, and
  `will_execute_write_bearing_action=false`.
- Added an Intelligence inspector `Preview fork plan` button for the latest
  validated `fork_branches` decision.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` returned healthy; discovery returned older broad
  repo facts only, so repo source remained authoritative.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.

Next:

- Add rollback preflight for validated `rollback` decisions, still preview-only
  and without modifying branch state.

## Turn 134 | 2026-05-20

Summary: Added App Intelligence rollback preflight.

Action:

- Added `ROLLBACK_PREFLIGHT_TOKEN=PREVIEW_ROLLBACK`.
- Added `preflight_rollback()` for validated `rollback` decisions.
- Added
  `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/rollback-preflight`.
- The preflight writes
  `artifacts/structured-decisions/<decision_id>.rollback-preflight.json`,
  appends `rollback_preflight`, and returns the current branch, target branch,
  optional target event/turn ids, and warnings for advisory-only targets.
- The preflight explicitly returns `will_modify_branches=false`,
  `will_revert_artifacts=false`, `will_create_thread=false`,
  `will_run_provider=false`, and
  `will_execute_write_bearing_action=false`.
- Added an Intelligence inspector `Preview rollback plan` button for the
  latest validated `rollback` decision.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 28 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.

Next:

- Add a safe no-op path for validated `continue_current_branch` decisions so
  every non-write-bearing structured decision has a host-owned record before
  enabling real branch, rollback, memory, route, or deposition apply actions.

## Turn 135 | 2026-05-20

Summary: Added ledger-only continue-current-branch apply records.

Action:

- Added `continue_current_branch` to the ledger-only structured-decision apply
  allowlist.
- The existing
  `POST /api/intelligence/runs/<run_id>/structured-decisions/<decision_id>/apply`
  endpoint can now record validated `continue_current_branch` decisions with
  `approval_token=APPLY_STRUCTURED_DECISION`.
- Continue apply writes
  `artifacts/structured-decisions/<decision_id>.apply.json`, appends
  `structured_decision_applied`, marks the decision `applied`, records
  `latest_continuation`, and leaves the run open as
  `phase=current_branch_continued` and `status=running`.
- Continue apply still returns no external action, no write-bearing action, no
  fork, and no rollback.
- Updated the React Intelligence inspector so the `Apply ledger-only decision`
  button is enabled for validated `continue_current_branch` decisions.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.

Next:

- Add structured decision history/replay visibility in the Intelligence
  inspector so operators can inspect more than the latest decision before real
  fork, rollback, memory, route, or deposition apply actions.

## Turn 136 | 2026-05-20

Summary: Added read-only structured-decision history in the Intelligence inspector.

Action:

- Added a Decision History card to the React Intelligence inspector using the
  existing selected-run ledger payload.
- The history shows validated, rejected, and applied structured decisions with
  decision id, validation artifact, Codex turn id, apply artifact, apply event
  id, action, status, and no-action flags.
- The newest decision is highlighted, but older decisions remain inspectable so
  operators can replay ledger state before future fork, rollback, memory,
  route, or deposition apply actions.
- This is read-only UI visibility; it does not add any new apply, preflight,
  provider, branch, memory, route, or external-write behavior.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add artifact-open/read endpoints for App Intelligence run artifacts so the
  inspector can show validation/apply/preflight JSON contents without exposing
  arbitrary filesystem paths.

## Turn 137 | 2026-05-20

Summary: Added registered App Intelligence artifact reads.

Action:

- Added
  `GET /api/intelligence/runs/<run_id>/artifacts?path=<path>` for reading
  App Intelligence run artifacts.
- The endpoint only serves files that resolve inside the selected run directory
  and are already referenced by the run ledger or event log.
- The endpoint returns parsed JSON when possible, text content, byte size,
  relative path, and explicit no-action flags.
- The Decision History card can now open validation and apply JSON artifacts
  in-place.
- Preflight artifacts are also readable when their `artifact_path` is recorded
  in the run event log.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.

Next:

- Add a small artifact picker for preflight artifacts in the Decision History
  panel, using event-log artifact records rather than guessing filenames.

## Turn 138 | 2026-05-20

Summary: Added event-log preflight artifact picker.

Action:

- Added a Preflight Artifacts picker to the React Decision History card.
- The picker derives rows only from selected-run events whose event type
  includes `preflight` and whose payload contains `artifact_path`.
- Picker buttons reuse the existing guarded registered-artifact reader; they
  do not guess filenames or introduce a new filesystem read path.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add run-artifact affordances for prompt/status artifacts in the inspector so
  every existing App Intelligence artifact class can be opened from the UI
  through the same registered-read endpoint.

## Turn 139 | 2026-05-20

Summary: Added prompt/status artifact opener controls.

Action:

- Added prompt-packet buttons for opening packet JSON and prompt text through
  the registered artifact reader.
- Added a latest turn-status button for opening the captured status JSON
  artifact through the same guarded endpoint.
- Kept the UI read-only; these buttons reuse recorded ledger paths and do not
  introduce a new filesystem read route.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add a run replay manifest endpoint that returns ordered prompt, status,
  decision, and preflight artifacts from the ledger/event log so the UI no
  longer has to derive replay structure from scattered fields.

## Turn 140 | 2026-05-21

Summary: Added App Intelligence replay manifests.

Action:

- Added `GET /api/intelligence/runs/<run_id>/replay-manifest`.
- The endpoint returns ordered registered artifact metadata for prompt packets,
  prompt text, turn status, structured-decision validation/apply artifacts,
  and preflight artifacts without reading artifact contents.
- Wired the React Intelligence inspector to load the replay manifest with run
  detail and display it as the primary replay artifact picker.
- Kept artifact content reads behind the existing registered artifact endpoint.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `python -m py_compile app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add a focused live UI smoke that selects a run, loads the replay manifest,
  opens one manifest artifact, and records the observed no-write response.

## Turn 141 | 2026-05-21

Summary: Added replay-manifest live smoke helper.

Action:

- Added `scripts/smoke_app_replay_manifest.py`.
- The smoke creates a disposable user-scoped App Intelligence run, prepares a
  local prompt packet without sending it, calls the live `/replay-manifest`
  endpoint, opens one artifact through `/artifacts?path=...`, and checks the
  no-write response flags.
- Added `--cleanup` so routine validation can leave no disposable run behind;
  omitting it leaves a selectable run for manual UI review.
- Documented the command in README and API docs, and updated the P09/ROADMAP
  current-state notes.

Validation:

- `python scripts/smoke_app_replay_manifest.py --cleanup` passed against the
  live local API and confirmed `will_execute_write_bearing_action=false`.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_app_replay_manifest.py app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add a browser-assisted operator smoke that drives the Intelligence panel
  itself against a disposable run and captures the visible replay-manifest
  result.

## Turn 142 | 2026-05-21

Summary: Added browser-assisted replay-manifest UI smoke.

Action:

- Added `scripts/smoke_app_replay_manifest_ui.py`.
- The smoke creates or refreshes a disposable App Intelligence run, opens the
  live React console with `agent-browser`, selects the Intelligence panel,
  selects the disposable run, clicks a Replay Manifest artifact, and verifies
  the visible no-write registered-reader response.
- The smoke writes JSON and screenshot evidence under
  `~/.local/state/transcribe-audio/browser-smokes/`.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `python scripts/smoke_app_replay_manifest.py --cleanup` passed against the
  live local API.
- `python scripts/smoke_app_replay_manifest_ui.py --run-id smoke-replay-manifest-ui-review --session transcript-replay-ui-smoke --cleanup`
  passed and wrote JSON/screenshot evidence under
  `~/.local/state/transcribe-audio/browser-smokes/`.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_app_replay_manifest.py scripts/smoke_app_replay_manifest_ui.py app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add an explicit operator cleanup command for disposable App Intelligence
  smoke runs and browser-smoke evidence retention.

## Turn 143 | 2026-05-21

Summary: Added App Intelligence smoke cleanup command.

Action:

- Added `scripts/cleanup_app_smokes.py`.
- The cleanup command reports disposable App Intelligence smoke run dirs and
  browser-smoke evidence files under the user-scoped runtime state.
- The command is dry-run by default and requires `--apply` before deleting any
  matching run directory or evidence file.
- Documented the command in README and API docs, and updated the P09/ROADMAP
  current-state notes.

Validation:

- Temp-state `python scripts/cleanup_app_smokes.py --apply` removed only
  matching disposable smoke dirs and expired evidence while preserving kept
  items.
- Live `python scripts/cleanup_app_smokes.py --keep-runs 1 --keep-evidence 10 --evidence-days 14`
  was dry-run only and reported `delete_run_count=0`,
  `delete_evidence_count=0`.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 30 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/cleanup_app_smokes.py scripts/smoke_app_replay_manifest.py scripts/smoke_app_replay_manifest_ui.py app_intelligence_ledger.py transcript_api.py codex_app_server_client.py`
  passed.
- `git diff --check` passed.

Next:

- Add a small smoke-status panel or endpoint that surfaces the latest API/UI
  smoke result paths in the Intelligence inspector.

## Turn 144 | 2026-05-21

Summary: Added App Intelligence smoke status endpoint and panel.

Action:

- Added `GET /api/intelligence/smokes`.
- The endpoint reports latest App Intelligence browser-smoke report metadata,
  screenshot path/existence, smoke check booleans, and disposable smoke run
  summaries without reading screenshot bytes or artifact contents.
- Added a Smoke Status card to the React Intelligence panel.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 31 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py app_intelligence_ledger.py codex_app_server_client.py scripts/cleanup_app_smokes.py scripts/smoke_app_replay_manifest.py scripts/smoke_app_replay_manifest_ui.py`
  passed.
- `git diff --check` passed.

Next:

- Add operator affordances for running smoke commands from the UI as queued,
  approval-gated local jobs rather than separate shell commands.

## Turn 145 | 2026-05-21

Summary: Added approval-gated smoke job queueing to the review console.

Action:

- Added `GET/POST /api/intelligence/smoke-jobs`.
- Smoke jobs are allowlisted to API replay smoke, browser replay smoke, and
  smoke cleanup; they require explicit approval tokens and never accept
  arbitrary shell input.
- Job records, stdout paths, and stderr paths are stored under the user-scoped
  state directory at `~/.local/state/transcribe-audio/smoke-jobs/`.
- Updated the React Smoke Status card with queue buttons and recent job
  status rows.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `python -m py_compile transcript_api.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_intelligence_smoke_jobs_endpoint_queues_allowlisted_command -q`
  passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 32 tests.
- `npm --prefix frontend run build` passed.
- Live `POST /api/intelligence/smoke-jobs` queued
  `api_replay_smoke-20260521T234447Z-c59848a0`, and
  `GET /api/intelligence/smoke-jobs?limit=1` reported `status=succeeded`,
  `returncode=0`, `will_execute_external_action=false`, and
  `will_execute_write_bearing_action=false`.

Next:

- Run a live UI-queued browser smoke through the service, then add automatic
  short polling while a queued smoke job is running.

## Turn 146 | 2026-05-21

Summary: Added smoke-job polling and verified a queued browser smoke.

Action:

- Added a React polling effect for the Smoke Status card. While any loaded
  smoke job is `queued` or `running`, the console refreshes smoke evidence,
  smoke jobs, and recent App Intelligence runs every 2 seconds.
- Updated queued smoke copy so operators know polling continues until the job
  finishes.
- Hardened `scripts/smoke_app_replay_manifest_ui.py` so service-launched
  browser smoke can resolve `agent-browser` through `AGENT_BROWSER_BIN`,
  `PATH`, `~/.local/bin/agent-browser`, or the pnpm shim.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile scripts/smoke_app_replay_manifest_ui.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active.
- Live `POST /api/intelligence/smoke-jobs` queued
  `browser_replay_smoke-20260522T024851Z-75f5bb4f`.
- Live `GET /api/intelligence/smoke-jobs?limit=1` reported
  `status=succeeded`, `returncode=0`, `will_execute_external_action=true`,
  and `will_execute_write_bearing_action=false`.
- Live `GET /api/intelligence/smokes?limit=1` reported latest browser smoke
  `status=pass`, all checks true, and screenshot evidence at
  `/home/ecochran76/.local/state/transcribe-audio/browser-smokes/20260522T024858Z-smoke-replay-manifest-ui-review.png`.

Next:

- Add a small read-only stdout/stderr tail endpoint for smoke jobs so failed
  jobs can be diagnosed from the UI without arbitrary artifact reads.

## Turn 147 | 2026-05-22

Summary: Added read-only smoke-job stdout/stderr diagnostics.

Action:

- Added `GET /api/intelligence/smoke-jobs/<job_id>/tail`.
- The endpoint validates the job id as one path segment, restricts `stream` to
  `stdout` or `stderr`, caps tail length, and rejects output paths outside
  `~/.local/state/transcribe-audio/smoke-jobs/`.
- Added stdout/stderr tail buttons and a diagnostic preview to the React Smoke
  Status card.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_intelligence_smoke_job_tail_endpoint_is_path_confined -q`
  passed.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 33 tests.
- `python -m py_compile transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were both active.
- Live stderr tail for failed job
  `browser_replay_smoke-20260522T024747Z-3e2989e6` returned the
  `agent-browser` PATH error with `will_read_arbitrary_file=false`.
- Live stdout tail for successful job
  `browser_replay_smoke-20260522T024851Z-75f5bb4f` returned the browser-smoke
  JSON tail with `will_execute_write_bearing_action=false`.

Next:

- Add an optional UI affordance for cleanup apply with a stronger confirmation
  gate, then use it to prune stale smoke failures while retaining recent pass
  evidence.

## Turn 148 | 2026-05-22

Summary: Added typed cleanup-apply gating for smoke jobs.

Action:

- Added an `Apply cleanup` button to the React Smoke Status card.
- Cleanup apply now requires typing `CLEANUP_APP_SMOKE_ARTIFACTS` in the UI
  before queueing the allowlisted cleanup job.
- Added a regression test that cleanup apply rejects the normal smoke-job
  token and records `--apply` only with the cleanup approval token.
- Fixed smoke-job JSON writes to use atomic replacement so background workers
  cannot race list/enqueue responses into empty job summaries.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token tests/test_transcript_api.py::test_smoke_job_write_is_atomic -q`
  passed with 2 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `/api/health` returned `status=ok`.
- Live cleanup dry-run job `cleanup_smokes-20260522T113924Z-7c0bd0d3`
  succeeded with `delete_run_count=0`, `delete_evidence_count=0`, and
  `will_execute_write_bearing_action=false`.
- Live cleanup apply job `cleanup_smokes-20260522T113940Z-d4008bbc`
  succeeded as a no-op with `required_approval_token_checked=CLEANUP_APP_SMOKE_ARTIFACTS`,
  `will_execute_write_bearing_action=true`, `delete_run_count=0`, and
  `delete_evidence_count=0`.

Next:

- Add a small operator-visible smoke artifact retention summary so the cleanup
  dry-run counts are visible without opening stdout tails.

## Turn 149 | 2026-05-22

Summary: Added operator-visible cleanup retention summaries.

Action:

- Added backend parsing for the structured `APP_SMOKE_CLEANUP_JSON=` line
  stored in each smoke job's bounded `stdout_tail`.
- Exposed a redacted `cleanup_summary` on smoke-job summaries with matched,
  kept, and delete counts but without delete-path lists.
- Updated the React Smoke Status recent-job list to show cleanup dry-run/apply
  retention counts inline.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token tests/test_transcript_api.py::test_intelligence_smoke_jobs_endpoint_queues_allowlisted_command -q`
  passed with 4 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py scripts/cleanup_app_smokes.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- Live `GET /api/intelligence/smoke-jobs?limit=3` returned
  `cleanup_summary` for recent cleanup jobs with apply/dry-run mode,
  matched/delete counts, and no delete-path lists.

Next:

- Add a compact visual distinction for write-bearing smoke jobs so cleanup
  apply entries stand out from dry-runs without requiring JSON inspection.

## Turn 150 | 2026-05-22

Summary: Added write-bearing smoke-job visual badges.

Action:

- Updated recent Smoke Status job rows with `write-bearing` or `read-only`
  risk badges based on `will_execute_write_bearing_action`.
- Added an orange row treatment for write-bearing cleanup apply jobs and a
  quieter green read-only treatment for dry-runs and read-only smoke jobs.
- Updated API docs, ROADMAP, and the P09 plan to document the operator-visible
  distinction.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned the rebuilt console HTML with the new
  asset names.
- Live `GET /api/intelligence/smoke-jobs?limit=1` returned
  `will_execute_write_bearing_action=true` for the latest cleanup apply job,
  which drives the write-bearing badge.

Next:

- Add a small legend to the Smoke Status card so the risk badges are
  self-explanatory for operators who have not read the docs.

## Turn 151 | 2026-05-22

Summary: Added a Smoke Status risk legend.

Action:

- Added an inline legend above recent smoke-job rows explaining
  `write-bearing` and `read-only` badges.
- Reused the existing badge colors so the legend matches the row treatments:
  write-bearing cleanup apply can delete allowlisted smoke artifacts after
  typed approval, while read-only jobs inspect status/tails or dry-run counts.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.

Next:

- Continue with a small smoke-status ergonomics pass, likely grouping recent
  jobs by action type or adding timestamps in a friendlier format.

## Turn 152 | 2026-05-22

Summary: Added friendly smoke-job timing.

Action:

- Added queued/finished/runtime timing text to recent Smoke Status job rows.
- Reused the existing local date formatter and left the backend smoke-job API
  contract unchanged.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.

Next:

- Continue smoke-status ergonomics by grouping recent jobs by action type when
  the recent list grows beyond a few rows.

## Turn 153 | 2026-05-22

Summary: Grouped recent smoke jobs by action type.

Action:

- Added frontend grouping for loaded recent smoke jobs by `job_type`.
- Rendered group headings with loaded counts while preserving newest-first
  order within each group.
- Removed the previous three-row cap so all loaded jobs from the existing
  `limit=5` request are visible.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.

Next:

- Add a small "loaded count vs total jobs" hint so operators understand when
  the grouped list is only showing the current API page.

## Turn 154 | 2026-05-22

Summary: Added smoke-job loaded-vs-total hint.

Action:

- Added a Smoke Status hint showing how many smoke jobs are loaded from the
  current API page versus the total retained smoke-job records.
- Reused the existing `/api/intelligence/smoke-jobs` `total` field; no backend
  contract change was required.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_cleanup_smoke_job_summary_is_exposed_from_stdout_tail tests/test_transcript_api.py::test_cleanup_smoke_job_summary_tolerates_bad_count_fields tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.

Next:

- Start the next operator workflow slice outside smoke-status polish, likely
  review-console support for first-pass summary batch visibility.

## Turn 155 | 2026-05-22

Summary: Added first-pass batch visibility.

Action:

- Preserved the full first-pass summary batch response in Review Queue action
  state after prepare, submit, and status checks.
- Added a structured batch status panel showing request count, batch id/status,
  provider counts, materialized count, and materialization error count.
- Kept the existing manifest-scoped backend contract unchanged.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_prepare_first_pass_summary_endpoint_writes_dry_run_manifest tests/test_transcript_api.py::test_first_pass_summary_submit_and_status_use_prepared_manifest tests/test_transcript_api.py::test_batch_status_counts_prefers_provider_aggregate_counts -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 37 tests.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.

Next:

- Add a read-only recent first-pass batch manifest list so operators can
  resume status checks after a page reload.

## Turn 156 | 2026-05-22

Summary: Added recent first-pass batch manifests.

Action:

- Added `GET /api/review-queue/first-pass-summaries/manifests` to list recent
  first-pass summary batch manifests under the user-scoped state directory.
- Manifest summaries report request count, batch id/status, provider counts,
  materialized count, and materialization error count without exposing request
  payloads or transcript content.
- Updated the Review Queue to load recent manifests on page load, refresh them
  after prepare/submit/status actions, and select a saved manifest for status
  checks after reload.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_first_pass_summary_manifests_endpoint_lists_redacted_summaries tests/test_transcript_api.py::test_prepare_first_pass_summary_endpoint_writes_dry_run_manifest tests/test_transcript_api.py::test_first_pass_summary_submit_and_status_use_prepared_manifest -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 38 tests.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` were active.
- `curl http://127.0.0.1:18876/api/health` returned `status=ok`.
- `GET /` from the live service returned rebuilt console HTML with the new
  asset names.
- Live `GET /api/review-queue/first-pass-summaries/manifests?limit=3`
  returned 3 redacted summaries out of 9 total manifests with
  `will_read_request_payloads=false` and `will_read_transcript_content=false`.

Next:

- Add a small reload-resume smoke that exercises selecting a saved manifest and
  polling status from the Review Queue UI.

## Turn 157 | 2026-05-22

Summary: Added first-pass reload-resume UI smoke.

Action:

- Added `scripts/smoke_first_pass_batch_resume_ui.py`.
- The smoke creates a disposable prepared first-pass summary manifest under the
  user-scoped runtime state, opens the Review Queue, selects the saved manifest,
  and clicks `Check and materialize`.
- The disposable manifest has no submitted batch, so the status check returns
  `prepared` without contacting AuraCall or materializing readouts.
- The smoke stores JSON and screenshot evidence under
  `~/.local/state/transcribe-audio/browser-smokes/` and supports `--cleanup`.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `python -m py_compile scripts/smoke_first_pass_batch_resume_ui.py` passed.
- Live `python scripts/smoke_first_pass_batch_resume_ui.py --base-url http://127.0.0.1:18876 --cleanup`
  passed with all checks true after tightening rendered-text assertions.
- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_first_pass_summary_manifests_endpoint_lists_redacted_summaries tests/test_transcript_api.py::test_prepare_first_pass_summary_endpoint_writes_dry_run_manifest tests/test_transcript_api.py::test_first_pass_summary_submit_and_status_use_prepared_manifest -q`
  passed with 3 tests.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 38 tests.
- `python -m py_compile scripts/smoke_first_pass_batch_resume_ui.py scripts/smoke_app_replay_manifest_ui.py transcript_api.py`
  passed.
- `git diff --check` passed.

Next:

- Add this smoke to the allowlisted smoke-job queue if we want operators to
  run it from the Smoke Status card instead of the terminal.

## Turn 158 | 2026-05-22

Summary: Queued first-pass resume UI smoke from the Smoke Status card.

Action:

- Added `first_pass_resume_ui_smoke` as an allowlisted smoke-job type.
- The fixed command runs `scripts/smoke_first_pass_batch_resume_ui.py` with the
  configured base URL, user-scoped state root, and `--cleanup`.
- Marked the job as external-action because it uses `agent-browser`, but not
  write-bearing because it only creates and removes disposable runtime smoke
  artifacts.
- Added a Smoke Status button for queueing the resume UI smoke.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_intelligence_smoke_jobs_endpoint_queues_allowlisted_command tests/test_transcript_api.py::test_first_pass_resume_ui_smoke_job_is_allowlisted tests/test_transcript_api.py::test_cleanup_smoke_job_apply_requires_cleanup_token -q`
  passed with 3 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py scripts/smoke_first_pass_batch_resume_ui.py`
  passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `POST /api/intelligence/smoke-jobs` queued
  `first_pass_resume_ui_smoke-20260522T224946Z-a6184a73`.
- Live `GET /api/intelligence/smoke-jobs?limit=1` reported
  `status=succeeded`, `returncode=0`, and
  `available_job_types` including `first_pass_resume_ui_smoke`.
- Live stdout tail included `FIRST_PASS_RESUME_UI_SMOKE_JSON=` with
  `"status":"pass"` and all Review Queue checks true.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 39 tests.
- `git diff --check` passed.

Next:

- Add a compact UI affordance to open the latest resume-smoke screenshot/report
  from the Smoke Status job row.

## Turn 159 | 2026-05-22

Summary: Started a UI-audit polish pass for operator testing.

Action:

- Used the `ui-audit` skill against the live React console and current source.
- Found that the Library kind filter controls looked interactive but did not
  actually scope the table.
- Added functional Library kind filters with row counts and active
  `aria-pressed` state.
- Added a compact operator test-status strip to the center pane showing API
  state, rows in scope, active filter/search state, latest smoke status, and a
  next testing action for the active workspace.
- Added visible keyboard focus styling for buttons, inputs, selects, and links.
- Updated ROADMAP and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed before the runbook/roadmap documentation update.
- Live browser check at `http://127.0.0.1:18876/` confirmed the status strip
  renders without crowding the center pane.
- Live browser interaction confirmed the Library `Transcripts` filter changes
  the status strip from `25 / 249` rows in scope with filter `all` to
  `9 / 249` rows in scope with filter `transcript`.

Next:

- Add direct report/screenshot links for smoke-job evidence and improve the
  Smoke Status card hierarchy so failed jobs are easier to debug.

## Turn 160 | 2026-05-22

Summary: Added direct Smoke Status evidence links.

Action:

- Added parsing for known browser-smoke stdout JSON prefixes into smoke-job
  `evidence_summary` metadata.
- Added `GET /api/intelligence/smoke-evidence?path=...` to serve JSON reports
  and PNG screenshots only from the user-scoped `browser-smokes` directory.
- Added Smoke Status row affordances for `Open report JSON` and
  `Open screenshot` when a completed job exposes known evidence.
- Updated API docs, ROADMAP, and the P09 plan.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py::test_smoke_job_summary_exposes_known_evidence_paths tests/test_transcript_api.py::test_smoke_evidence_endpoint_is_path_confined tests/test_transcript_api.py::test_intelligence_smoke_job_tail_endpoint_is_path_confined -q`
  passed with 3 tests after tightening URL path encoding.
- `.venv/bin/python -m pytest tests/test_app_intelligence_ledger.py tests/test_transcript_api.py -q`
  passed with 41 tests.
- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/intelligence/smoke-jobs?limit=1` returned
  `evidence_summary` for
  `first_pass_resume_ui_smoke-20260522T224946Z-a6184a73`, including encoded
  report and screenshot URLs.
- Live `GET /api/intelligence/smoke-evidence` returned the report JSON with
  `status=pass` and returned a PNG screenshot; `/etc/passwd` was rejected with
  HTTP 400.
- Live browser check on the Intelligence tab found both `Open report JSON` and
  `Open screenshot`.

Next:

- Add a failed-job-first grouping or alert band so the Smoke Status card makes
  failures more prominent than older successful jobs.

## Turn 161 | 2026-05-22

Summary: Added failed-job-first Smoke Status alerting.

Action:

- Increased the Smoke Status job page from 5 to 20 jobs so recent failures do
  not disappear behind newer successes.
- Added a dedicated failed smoke-job alert band before the normal grouped job
  history.
- Failed jobs are removed from the normal grouped history to avoid duplicate
  rows and keep diagnostic actions focused.
- Updated ROADMAP and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- `systemctl --user is-active transcripts.service transcribe-watch.service`
  returned both services active.
- Live `GET /api/intelligence/smoke-jobs?limit=20` returned 7 loaded jobs
  including failed job `browser_replay_smoke-20260522T024747Z-3e2989e6`.
- Live browser check on the Intelligence tab found the failed smoke alert text
  and the failed job id.

Next:

- Add filter toggles for the Smoke Status card so operators can switch between
  failed-only, write-bearing, and evidence-bearing jobs without reading the
  whole history.

## Turn 162 | 2026-05-22

Summary: Added Smoke Status job filters.

Action:

- Added local Smoke Status filter toggles for all, failed, write-bearing, and
  evidence-bearing jobs.
- Filter counts are computed from the loaded 20-job smoke page and use
  `aria-pressed` active state.
- The failed alert band remains first when failures match the current filter,
  and non-failed grouped history is filtered separately to avoid duplicates.
- Updated ROADMAP and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live browser check on the Intelligence tab showed filter counts
  `All 7`, `Failed 1`, `Write-bearing 1`, and `Evidence 2`.
- Live browser check confirmed the write-bearing filter shows the cleanup apply
  job and hides the failed browser job.
- Live browser check confirmed the evidence filter shows evidence links and
  hides the failed browser job.

Next:

- Add a compact selected-job detail drawer or inline expansion so one job's
  command, risk flags, evidence, and tails are easier to inspect together.

## Turn 163 | 2026-05-22

Summary: Reframed the frontend around dogfoodable UX.

Action:

- Paused additive UX polish to audit what is wired versus placeholder.
- Added a P09 dogfooding UX direction: every visible control must either
  execute a real local action, open a real read-only route/artifact, or be
  explicitly disabled with planned-state copy.
- Disabled planned navbar sections instead of letting them switch to an
  unrelated Library-like viewport.
- Wired the Library inspector `Open context JSON` action to the real
  `/api/documents/<id>/context` endpoint.
- Marked share-link and speaker-review inspector actions as planned/disabled.
- Updated ROADMAP and the P09 plan.

Validation:

- `npm --prefix frontend run build` passed.
- `python -m py_compile transcript_api.py` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/library?limit=1` returned selected transcript
  `2596e459aeb3812de321`.
- Live `GET /api/documents/2596e459aeb3812de321/context?context_chunks=2`
  returned a read-only context response for the selected document.
- Live browser check confirmed only Library, Review Queue, and Intelligence are
  enabled; Context Runs, Contacts, Provenance, Depositions, and Settings are
  disabled with `PLANNED` labels.
- Live browser check confirmed the selected Library artifact exposes
  `/api/documents/2596e459aeb3812de321/context?context_chunks=2` as
  `Open context JSON`, while share-link and speaker-review actions are disabled
  as planned.

Next:

- Continue wiring the beginning of the dogfooding path from Library detail:
  add document JSON/context preview inside the inspector instead of opening a
  raw JSON tab.

## Turn 164 | 2026-05-22

Summary: Added a modal conversation workspace for Library artifacts.

Action:

- Kept the right inspector as a compact selected-artifact preview and launcher.
- Added `Open conversation workspace` from the inspector.
- Added double-click row opening from the Library table.
- Added an accessible modal dialog with Escape and backdrop close behavior.
- Organized the conversation workflow into addressable sections: raw audio,
  re-transcription, raw summary, context workbench, speaker/contact identity,
  and final readout.
- Surfaced linked source audio and download in the modal, including inherited
  source-transcript blobs for readouts.
- Kept unwired mutation paths disabled and labelled as planned for
  re-transcription, context runs, contact linking, final readout generation,
  and final readout sharing.
- Updated `ROADMAP.md` and the P09 plan to make the modal workspace the primary
  conversation lifecycle surface.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main "modal inspector
  conversation workflow audio retranscription raw summary context workbench
  speakers final readout UX"` returned repo memory and no conflicting UX
  direction.
- `npm --prefix frontend run build` passed.

Next:

- Add a dedicated related-document API endpoint and wire the modal's
  re-transcription/context/final-readout actions to reviewed backend dry-run
  contracts one stage at a time.

## Turn 165 | 2026-05-22

Summary: Added read-only related-document lookup for the conversation workspace.

Action:

- Added `GET /api/documents/<document_id>/related`.
- The endpoint resolves readout/contextual-readout `source_artifact_path`
  metadata back to a stored source transcript when present.
- The endpoint lists readouts/contextual readouts derived from a selected
  transcript's source path.
- Updated the React Library detail loader to fetch document detail and related
  documents together.
- Updated the compact inspector and conversation modal to prefer
  `source_document` from the related endpoint when selecting inherited source
  audio, with the current loaded-library scan retained only as fallback.
- Documented the endpoint in README and `docs/dev/transcript-review-api.md`.
- Updated ROADMAP and the P09 plan to record related-document lookup as the
  source-audio resolution path.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main "related
  document API readout source transcript audio modal workflow dry run
  contracts"` returned older advisory memory and no conflicting direction.
- Added and passed
  `tests/test_transcript_api.py::test_related_documents_links_readout_to_source_transcript`.
- Full `tests/test_transcript_api.py` passed.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/documents/e2a67052289d1ad1d891/related` resolved source
  transcript `646f58f133108230a67c` and blob `f363a21cdc716795bd3c`.
- Live `GET /api/documents/646f58f133108230a67c/related` listed derived
  readout `e2a67052289d1ad1d891`.
- Browser smoke on `transcripts.localhost` confirmed the modal fetches
  `/api/documents/e2a67052289d1ad1d891/related` and still displays inherited
  audio `/api/blobs/f363a21cdc716795bd3c`.

Next:

- Wire the modal's first real action contract: a retranscription dry-run
  preflight that reports source blob, backend, output paths, and write plan
  before any transcription job can be queued.

## Turn 166 | 2026-05-23

Summary: Added the re-transcription dry-run preflight contract.

Action:

- Added `POST /api/documents/<document_id>/retranscription/preflight`.
- The endpoint resolves readouts back to their source transcript when possible,
  then resolves the linked `source_recording` blob.
- The response reports selected backend, source blob metadata, planned output
  paths, command preview, blocking checks, and the future
  `QUEUE_RETRANSCRIPTION_JOB` queue token.
- The contract is dry-run only and returns `will_queue=false`,
  `will_run_transcription=false`, and `will_write_files=false`.
- Wired the conversation modal's re-transcription panel to call the preflight
  endpoint, show a structured summary, and keep actual queueing/diff actions
  disabled as planned.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main "retranscription
  dry run preflight modal source blob backend planned outputs"` returned older
  advisory memory and no conflicting direction.
- Added focused API/helper tests for re-transcription preflight dry-run
  behavior.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  33 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `POST /api/documents/e2a67052289d1ad1d891/retranscription/preflight`
  returned source transcript `646f58f133108230a67c`, source blob
  `f363a21cdc716795bd3c`, and all no-queue/no-run/no-write flags false.
- Browser smoke on `transcripts.localhost` confirmed the Cara readout modal
  displays the backend selector, `Preview re-transcription`, the preflight
  success message, output folder, command preview, and safety flags.

Next:

- Add the reviewed queue endpoint behind `QUEUE_RETRANSCRIPTION_JOB`, writing a
  job record first and running no speech backend until the operator approves
  the dry-run plan.

## Turn 167 | 2026-05-23

Summary: Added reviewed re-transcription queue manifests.

Action:

- Added `POST /api/documents/<document_id>/retranscription/queue`.
- The endpoint requires `approval_token=QUEUE_RETRANSCRIPTION_JOB`.
- Queueing reuses the preflight resolver, blocks when the source blob is
  missing, and writes a durable job manifest under
  `~/.local/state/transcribe-audio/retranscription-jobs/`.
- The queued manifest records the source document, source blob, selected
  backend, planned outputs, command preview, and future
  `RUN_RETRANSCRIPTION_JOB` gate.
- Queueing remains non-executing: it returns `will_start_background_job=false`,
  `will_run_transcription=false`, and `will_write_files=false`.
- Wired the conversation modal to enable `Queue manifest` only after a
  successful preflight, then show job id, manifest path, run gate, and safety
  flags.
- Updated README, API docs, ROADMAP, and the P09 plan.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main
  "retranscription queue endpoint approval token job manifest no backend
  execution modal"` returned older advisory memory and no conflicting
  direction.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  35 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live queue API smoke without the approval token returned HTTP 400.
- Live `POST /api/documents/e2a67052289d1ad1d891/retranscription/queue`
  with `approval_token=QUEUE_RETRANSCRIPTION_JOB` wrote queued job manifest
  `retranscription-20260523T032352Z-435317b4`, source blob
  `f363a21cdc716795bd3c`, `RUN_RETRANSCRIPTION_JOB`, and all no-run/no-write
  flags false.
- Browser smoke on `transcripts.localhost` confirmed the Cara readout modal
  enables `Queue manifest` after preflight and then displays the queued
  manifest result with `RUN_RETRANSCRIPTION_JOB` and no-run/no-write flags.

Next:

- Add the separate `RUN_RETRANSCRIPTION_JOB` execution preflight/status model,
  with explicit safeguards around output overwrites and backend-specific
  command construction.

## Turn 168 | 2026-05-23

Summary: Overhauled the conversation workspace UX.

Action:

- Changed the Library table from artifact rows to conversation rows grouped by
  transcript/source path.
- Added workflow progress icons for transcript, first-pass summary, and
  contextual readout readiness.
- Updated the inspector to show useful selected-conversation metadata, audio,
  and a `Conversation summary` panel instead of labelling raw transcript text
  as a readout.
- Replaced the modal card grid with a full-viewport conversation workspace.
- Split the workspace into selectable workflow views: Transcript,
  First-pass summary, Context workbench, Speakers, and Final readout.
- Added source-transcript detail loading in the workspace so readout-selected
  conversations can display the real transcript.
- Added transcript text parsing for legacy transcript artifacts without
  structured utterances.
- Rendered transcript turns in a scrollable frame with deterministic
  speaker-color delineation.
- Kept re-transcription preflight and queue-manifest controls in the workspace
  rail without starting backend work.
- Updated ROADMAP and the P09 plan.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main "conversation
  workspace full viewport transcript speaker colors library grouped
  transcript readout inspector summary"` returned older advisory memory and no
  conflicting direction.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  35 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Browser smoke on `transcripts.localhost` with the Cara search confirmed one
  grouped conversation row, the inspector `Conversation summary` region, the
  full-viewport workspace, source audio controls, selectable workflow tabs, a
  scrollable transcript turn frame, first-pass summary view, and separate
  context workbench view.

Next:

- Add conversation-level API grouping so the UI does not need to infer
  transcript/readout relationships from paged client-side rows.

## Turn 169 | 2026-05-23

Summary: Added resizable Library columns and media play actions.

Action:

- Added draggable and keyboard-adjustable resize handles to the Library table
  column headers.
- Replaced media status text such as `Blob linked`/`Source blob` with a
  source-audio `Play` button.
- The `Play` button opens the conversation workspace when audio is available
  and renders disabled/greyed out when no source recording is linked.
- Updated ROADMAP and the P09 plan.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  35 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Browser smoke on `transcripts.localhost` with the Cara search confirmed
  accessible column resize controls in the table header and an enabled `Play`
  button that opens the full conversation workspace.

Next:

- Add conversation-level API grouping so table grouping and workflow progress
  are server-backed rather than inferred from the current client-side page.

## Turn 170 | 2026-05-23

Summary: Added server-backed conversation grouping for the Library table.

Action:

- Added `/api/conversations` to group transcript, first-pass readout, and
  contextual readout artifacts by source transcript.
- The endpoint returns representative/source/latest artifacts, workflow flags,
  calendar labels, linked media metadata, and artifact membership without
  returning artifact text or reading artifact files.
- Updated the React Library to prefer `/api/conversations`, keep client-side
  grouping only as an offline fallback, and show conversation counts in the
  operator status strip.
- Documented the endpoint in the review API docs, README, ROADMAP, and P09
  plan.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  36 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/conversations?query=Cara&limit=5` returned 6 grouped
  conversations; the first returned transcript=true, summary=true,
  contextual_readout=false, and media_ready=true.
- Browser smoke on `transcripts.localhost` confirmed the built UI loads against
  the live local API, shows conversation rows with workflow icons, and renders
  enabled/disabled `Play` buttons from server-backed media metadata.

Next:

- Add a conversation detail endpoint so the modal can load transcript,
  summaries, contextual readouts, related contacts, and media in one request
  instead of stitching several document-level calls together.

## Turn 171 | 2026-05-23

Summary: Added a conversation workspace detail endpoint.

Action:

- Added `GET /api/conversations/<document_id>` to return one selected
  conversation payload with grouped metadata, selected document, source
  transcript, first-pass summary, contextual readout, extracted participants,
  and linked media.
- Updated the conversation workspace modal to fetch the detail payload when it
  opens and use it for transcript, summary, final-readout, artifact membership,
  participant, and source-media views.
- Kept existing document-level detail calls as inspector/fallback paths while
  moving the modal toward the conversation-level contract.
- Documented the detail endpoint in README, ROADMAP, the P09 plan, and the
  review API contract.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/conversations/<document_id>` for the Cara conversation
  returned the expected detail schema with selected document, source transcript,
  first-pass summary, 4 participants, media_ready=true, and no arbitrary
  artifact-file reads.
- Browser smoke through `agent-browser` was attempted, but the default runtime
  profile was locked by an existing reachable browser and the CLI refused to
  launch a second profile without closing the active runtime.

Next:

- Make the global search/filter path call `/api/conversations` with the active
  query/kind instead of loading a fixed 500-row conversation page and filtering
  client-side.

## Turn 172 | 2026-05-23

Summary: Made Library search and filters server-backed.

Action:

- Removed the fixed startup fetch of `/api/conversations?limit=500` from the
  general app bootstrap.
- Added a debounced Library-only conversation fetch that calls
  `/api/conversations` with active `query`, `kind`, and `limit=100`.
- Updated the Library table path so API-backed conversation results are used
  directly instead of being re-filtered client-side.
- Kept client-side grouping only for offline/redacted fallback rows.
- Documented the server-backed search/filter behavior in README, ROADMAP, and
  the P09 plan.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/conversations?query=Cara&kind=readout&limit=3` returned
  schema `transcribe-audio.conversations.v1`, total=3, first row with
  transcript=true, summary=true, media_ready=true, and readout/transcript
  artifact membership.

Next:

- Add loading/empty-result affordances to the Library table so no-match server
  searches are visually distinct from API fallback or startup loading.

## Turn 173 | 2026-05-23

Summary: Added explicit Library table loading, empty, and error states.

Action:

- Added `conversationSearchStatus` state for Library conversation searches.
- Rendered a loading row while `/api/conversations` is in flight.
- Rendered an empty-result row when a server-backed search returns zero
  matching conversations.
- Rendered an API-error/fallback row when conversation search fails and the UI
  is using stale or local rows.
- Styled table states with distinct loading, empty, and warning treatments.
- Updated ROADMAP and the P09 plan to record the UX behavior.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/conversations?query=Cara&kind=readout&limit=3` returned
  total=3.
- Live `GET /api/conversations?query=zzzz-no-match-for-table-state&limit=3`
  returned total=0 and zero items, exercising the UI's empty-state input.

Next:

- Add pagination or "load more" for server-backed Library conversations so the
  table is not limited to the first 100 matches.

## Turn 174 | 2026-05-23

Summary: Added server-backed Library conversation pagination.

Action:

- Added a shared conversation page size and first-page `/api/conversations`
  loading with explicit `offset=0`.
- Added a `Load more` action that requests the next offset page, appends
  de-duplicated conversation rows, and preserves the selected conversation.
- Added a Library pagination footer with loaded/total counts and
  loading/disabled button states.
- Updated ROADMAP and the P09 plan to record paginated Library results.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /api/conversations?limit=2&offset=0` returned total=173 and two
  conversation keys.
- Live `GET /api/conversations?limit=2&offset=2` returned total=173 and two
  different conversation keys, exercising offset pagination.

Next:

- Add URL-state/deep-linking for Library query, kind, selected conversation,
  and workspace tab so dogfooding reports can point to reproducible UI states.

## Turn 175 | 2026-05-23

Summary: Added Library URL-state deep links.

Action:

- Added initial URL parsing for `view`, `kind`, `q`, `selected`,
  `conversation=1`, and `workflow`.
- Promoted the conversation workspace tab selection into App state so workflow
  tabs can be represented in the URL.
- Added URL replacement when Library search/filter/selection/workspace state
  changes.
- Preserved a URL-selected document even when it is not present on the first
  paginated conversation page.
- Updated ROADMAP and the P09 plan to record URL-addressable dogfooding state.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /?kind=readout&q=Cara&conversation=1&workflow=context&selected=test-doc`
  served the built React app through `transcripts.service`.
- Live `GET /api/conversations?query=Cara&kind=readout&limit=1` returned
  total=3 and one selected representative id for deep-link construction.

Next:

- Add an explicit copy/share affordance for the current Library workspace URL
  so dogfooding feedback does not rely on manually copying the address bar.

## Turn 176 | 2026-05-23

Summary: Added a Library copy-workspace-link affordance.

Action:

- Added a `Copy workspace link` control to the Library summary strip.
- Added clipboard-copy behavior for the current deep-linked URL.
- Added a prompt-based manual-copy fallback when the browser blocks or lacks
  clipboard write support.
- Added compact success/blocked status feedback below the Library heading.
- Updated ROADMAP and the P09 plan to record the share affordance.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- Live `GET /?kind=readout&q=Cara&conversation=1&workflow=context&selected=test-doc`
  served the built React app through `transcripts.service`.
- Live built asset contained `Copy workspace link` and the clipboard fallback
  text, confirming the deployed frontend includes the share affordance.

Next:

- Add a lightweight browser smoke for the Library deep-link/share flow so URL
  state and copy affordance regressions are caught automatically.

## Turn 177 | 2026-05-23

Summary: Used agent-browser to tighten Library and conversation workspace
density.

Action:

- Ran `agent-browser` against the live `transcripts.service` UI at desktop and
  narrower three-pane widths.
- Captured before/after screenshots for the Library and deep-linked
  conversation workspace under `/tmp/transcripts-ux-density-*.png`.
- Reduced topbar, pane, card, test-strip, workflow-tab, and modal chrome
  spacing so useful controls appear earlier in the viewport.
- Reduced Library, inspector, and workflow heading sizes for the constrained
  three-pane layout.
- Hid the topbar horizontal scrollbar while preserving horizontal nav
  scrollability.
- Replaced the blocking JavaScript prompt share fallback with an inline
  selectable URL field after `agent-browser` exposed the prompt as automation-
  and operator-hostile.
- Updated ROADMAP and the P09 plan with the browser-reviewed density pass.

Validation:

- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; `transcripts.service` and
  `transcribe-watch.service` both reported active.
- `agent-browser` captured and reviewed
  `/tmp/transcripts-ux-density-library-1280-after.png`,
  `/tmp/transcripts-ux-density-modal-after.png`, and
  `/tmp/transcripts-ux-density-share-final.png`.
- `agent-browser` click on `Copy workspace link` produced a non-blocking
  `Current workspace link` textbox instead of a JavaScript prompt.
- Live built assets contained the short clipboard fallback text and
  `scrollbar-width:none` for the compact topbar navigation.

Next:

- Add a committed `agent-browser` smoke script for the Library deep-link/share
  flow so the exact browser checks from this turn are repeatable.

## Turn 178 | 2026-05-23

Summary: Added a repeatable Library deep-link/share browser smoke.

Action:

- Added `scripts/smoke_library_deeplink_share_ui.py`.
- The smoke uses an isolated `agent-browser` profile by default, opens
  `/?kind=readout&q=Cara&conversation=1&workflow=context`, verifies Library
  URL state and the Context workbench deep link, closes the workspace, clicks
  `Copy workspace link`, and accepts either clipboard success or the
  non-blocking manual-copy URL field.
- The smoke writes JSON and screenshot evidence under
  `~/.local/state/transcribe-audio/browser-smokes/`.
- Updated README, ROADMAP, and the P09 plan to record the repeatable smoke.

Validation:

- `python -m py_compile scripts/smoke_library_deeplink_share_ui.py scripts/smoke_app_replay_manifest_ui.py transcript_api.py transcript_store.py` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed with
  37 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- `python scripts/smoke_library_deeplink_share_ui.py --base-url http://127.0.0.1:18876 --session transcript-library-share-ui-smoke-final --profile /tmp/transcript-library-share-ui-smoke-final-profile --viewport 1280x820` passed.
- The live smoke wrote report
  `~/.local/state/transcribe-audio/browser-smokes/20260523T144114Z-library-share-ui-smoke.json`
  and screenshot
  `~/.local/state/transcribe-audio/browser-smokes/20260523T144114Z-library-share-ui-smoke.png`.
- `transcripts.service` and `transcribe-watch.service` both reported active.

Next:

- Decide whether this Library deep-link/share smoke should become an
  allowlisted UI-triggered smoke job beside the existing replay and first-pass
  resume browser smokes.

## Turn 103 | 2026-05-17

Summary: Retried first-pass summaries after the AuraCall lease-heartbeat fix;
two readouts materialized and one browser child failed on stale-response
detection.

Action:

- Confirmed `transcribe-watch.service` and `auracall-api.service` were active.
- Ran Graphiti discovery against `transcribe_audio_main`; it returned only
  older broad repo facts, so live repo/API state remained the authority.
- Confirmed the live first-pass summary queue had 17 pending items before
  submitting.
- Submitted a conservative three-item AuraCall batch with
  `--max-concurrent-runs 1`, `--max-browser-interactions-per-minute 8`, and
  `--store`.
- Manifest:
  `~/.local/state/transcribe-audio/auracall-batches/first-pass-summary-20260517-201320.json`.
- Batch id: `batch_4201009fb3e84b498957ae992866191e`.
- AuraCall runner topology stayed healthy during the run: no active-lease
  health warning, fresh runner heartbeats, and runner activity updated from
  browser runtime evidence.

Validation:

- Final batch status: `failed`.
- Final counts: `total=3`, `completed=2`, `failed=1`, `cancelled=0`,
  `missing=0`, `in_progress=0`.
- Materialized readouts: 2; materialization errors: 0.
- Failed child: `resp_3168e7286aa94bef85f02eaca860e58f` with
  `runner_execution_failed: Stale ChatGPT assistant response detected after send.`
- Materialized readouts:
  `~/.transcripts/legacy-artifacts/ce/cebc3de9804d0276e862-2026-01-07 Scott Roberts Charlie Nacu Austin update Recording (10).readout.json`.
- Materialized readouts:
  `~/.transcripts/legacy-artifacts/37/37a7dc67cc5cb5870a95-2026-02-13 14-45 SoyLei - Discussion of Infringement & C&D Letters My recording 53 (1).readout.json`.
- `scripts/check_readout_quality.py` passed for both new readouts: 2 pass,
  0 warn, 0 fail.
- Live first-pass summary queue now reports 15 pending items.

Notes:

- The prior suspicious-idle/connection-loss failure did not recur in this run.
- The remaining failed child is an AuraCall browser-response freshness issue,
  not a transcript-store or prompt-materialization failure.
- During polling, AuraCall status reads intermittently returned JSON parse
  errors, but direct batch/response reads recovered and final materialization
  succeeded for completed children.

Next:

- Hand the new `Stale ChatGPT assistant response detected after send` failure
  to AuraCall, then retry the failed Scott Roberts Call 2 item as a one-item
  batch before scaling the remaining 15 pending summaries.

## Turn 102 | 2026-05-17

Summary: Retried the remaining stalled readouts after AuraCall fixes; one more
materialized, two still failed on browser connection loss.

Action:

- Verified `transcripts.service` and `auracall-api.service` were active.
- Verified AuraCall recovery status had no active leases, stale heartbeats,
  reclaimable runs, or recoverable stranded runs before submitting.
- Submitted three-item retry manifest
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260517-155332.json`.
- Batch id: `batch_78c87360d55449feaf56ca961861a0da`.
- Polled through the transcript API status/materialize endpoint.
- One readout completed and materialized.
- Two jobs failed with AuraCall `runner_execution_failed` at stage
  `connection-lost`; both reported "Chrome window closed before auracall
  finished. Please keep it open until completion."
- Submitted a one-item retry for the first remaining pending readout:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260517-155802.json`.
- Retry batch id: `batch_84c2166157194de9a249d192306f524d`.
- AuraCall restarted during the one-item retry. The response became
  `recoverableStranded`, was claimed back to the local runner, made no provider
  progress, and was cancelled to avoid leaving active work hanging.

Validation:

- Three-item retry final status: `failed`.
- Three-item retry final counts: `total=3`, `completed=1`, `failed=2`,
  `cancelled=0`, `missing=0`, `in_progress=0`.
- Three-item retry materialization: `materialized=1`,
  `materialization_errors=0`.
- Quality gate on the three-item manifest passed for the materialized readout:
  1 pass, 0 warn, 0 fail.
- Materialized readout:
  `~/.transcripts/legacy-artifacts/a0/a0cf796472e3426a594f-2026-01-23 10-30 Soylei--Ingevity review Recording (10).readout.json`.
- One-item retry final status: `cancelled`.
- One-item retry final counts: `total=1`, `completed=0`, `failed=0`,
  `cancelled=1`, `missing=0`, `in_progress=0`.
- Live review queue now reports 17 pending first-pass summaries.
- Graphiti discovery was healthy but only returned older repo memory; live
  runbook/API state remained the authority.

Next:

- Return to AuraCall for the remaining blocker: browser-backed response runs
  still lose the Chrome connection or strand after service restart under the
  `wsl-chrome-3` ChatGPT runtime profile. Do not submit another batch until a
  single non-private artifact smoke survives the restart/connection-loss path.

## Turn 101 | 2026-05-17

Summary: Ran the next first-pass batch; AuraCall stalled after two valid
readouts.

Action:

- Submitted five-item manifest
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260517-075535.json`.
- Batch id: `batch_6a27f88c9abe4d71b16090f7f53efc5a`.
- Polled through the transcript API status/materialize endpoint.
- Materialized two completed readouts, then recovered/cancelled stranded
  AuraCall runs that were stuck as `in_progress` without producing artifacts.
- Prepared an exact three-item retry manifest for the unmaterialized readouts:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260517-083532.json`.
- Retry batch id: `batch_fcc01c7655d440d0aaabfe7e0125686e`.
- The retry made no provider progress after recovery claims; all three retry
  runs were cancelled to avoid leaving active batch work hanging.

Validation:

- Original final status: `cancelled`.
- Original final counts: `total=5`, `completed=2`, `cancelled=3`,
  `failed=0`, `missing=0`, `in_progress=0`.
- Original materialization: `materialized=2`, `materialization_errors=0`.
- Retry final status: `cancelled`.
- Retry final counts: `total=3`, `completed=0`, `cancelled=3`, `failed=0`,
  `missing=0`, `in_progress=0`.
- Retry materialization: `materialized=0`, `materialization_errors=0`.
- `scripts/check_readout_quality.py` on the original manifest passed for the
  two materialized readouts: 2 pass, 0 warn, 0 fail.
- `transcripts.service` remained active.
- `auracall-api.service` was active at closeout, but had restarted during the
  batch window and left multiple runs with recoverable or suspiciously idle
  lease states.
- Live review queue now reports 18 pending first-pass summaries.

Next:

- Fix the AuraCall provider-progress/restart issue before submitting another
  multi-item batch. Use a single transcript smoke on
  `agent:pro-extended-chatgpt-soylei-transcripts`, require a surfaced
  `first_pass_readout.json` artifact, and confirm local materialization before
  retrying the three cancelled readouts.

## Turn 100 | 2026-05-17

Summary: Completed the one-item retry for the failed second-batch transcript.

Action:

- Polled retry batch `batch_d2dbf05f502e489b9cec3ee8c873f61d` through the
  transcript API status/materialize endpoint.
- Materialized the completed retry readout from manifest
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-220424.json`.
- Ran the readout quality gate on the retry manifest.

Validation:

- Retry final status: `completed`.
- Retry final counts: `total=1`, `completed=1`, `failed=0`, `cancelled=0`,
  `missing=0`, `in_progress=0`.
- Materialization: `materialized=1`, `materialization_errors=0`.
- Quality gate: 1 pass, 0 warn, 0 fail.
- Materialized readout:
  `~/.transcripts/legacy-artifacts/b0/b0fce70ac392889ad41f-2025-12-02 08-00 NCAT My recording 120.readout.json`.
- Live review queue now reports 20 pending first-pass summaries.

Next:

- Prepare and submit the next five-item first-pass summary batch, then run the
  same status/materialize and quality-gate loop before scaling.

## Turn 99 | 2026-05-17

Summary: Submitted the next five-item batch; four materialized and one item is
under single-item retry.

Action:

- Submitted prepared manifest
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-200445.json`.
- Batch id: `batch_aa65f65dff894602a30c5a7cc3dca9d1`.
- Polled with `materialize=true` until the batch reached a terminal state.
- The batch completed four readouts and failed one job.
- Failure details: job index 4 failed with `runner_execution_failed` at stage
  `connection-lost`; AuraCall reported "Chrome window closed before auracall
  finished. Please keep it open until completion."
- The four successful readouts passed `scripts/check_readout_quality.py` with
  4 pass, 0 warn, 0 fail.
- Prepared and submitted a one-item retry for the failed queue item:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-220424.json`.
- Retry batch id: `batch_d2dbf05f502e489b9cec3ee8c873f61d`.

Validation:

- Second batch final counts: `total=5`, `completed=4`, `failed=1`,
  `cancelled=0`, `missing=0`, `in_progress=0`.
- Second batch materialization: `materialized=4`,
  `materialization_errors=0`.
- Retry batch latest status: `running`, `total=1`, `in_progress=1`,
  `completed=0`, `failed=0`.
- Live review queue reports 21 pending first-pass summaries after the four
  new materialized readouts.

Next:

- Poll retry batch `batch_d2dbf05f502e489b9cec3ee8c873f61d` until it completes
  or fails. Do not start another batch while this retry is in progress.

## Turn 98 | 2026-05-16

Summary: Added a reusable quality gate for materialized first-pass readouts and
prepared the next five-item batch.

Action:

- Added `scripts/check_readout_quality.py`.
- The quality gate checks readout JSON shape, `schema_version=1`, non-empty
  title/summary/source/generated timestamp, source-artifact existence, paired
  Markdown existence, summary length, and core list fields for participants,
  topics, action items, matter candidates, and memory candidates.
- Added focused tests in `tests/test_readout_quality.py`.
- Documented the quality gate in README and `docs/dev/transcript-review-api.md`.
- Ran the gate against the completed five-item manifest:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-191244.json`.
- Prepared the next dry-run five-item manifest:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-200445.json`.

Validation:

- `python scripts/check_readout_quality.py --manifest ... --format text`
  passed for the completed batch with 5 pass, 0 warn, 0 fail.
- `.venv/bin/python -m pytest tests/test_readout_quality.py -q` passed.
- `python -m py_compile scripts/check_readout_quality.py tests/test_readout_quality.py` passed.
- The next prepared manifest has `request_count=5`, `dry_run=true`,
  `batch=null`, workflow `transcribe-audio-first-pass-summary`, and artifact
  `first_pass_readout.json`.

Next:

- Submit the next prepared five-item manifest if the operator wants to continue
  AuraCall execution, then run the same status/materialize loop and quality gate
  before scaling beyond five.

## Turn 97 | 2026-05-16

Summary: Completed and materialized the first five-item first-pass summary
batch.

Action:

- Continued polling AuraCall batch `batch_73f8ae99132741f2ba30a19905587b2c`
  through the transcript API status/materialize endpoint.
- Materialized completed readouts as they became available.
- Verified all five completed readouts wrote both JSON and Markdown artifacts
  beside their source transcript artifacts and ingested into the transcript
  store.

Validation:

- Final batch status: `completed`.
- Final counts: `total=5`, `completed=5`, `failed=0`, `cancelled=0`,
  `missing=0`, `in_progress=0`.
- Materialization: `materialized=5`, `materialization_errors=0`.
- Readout JSON validation showed each materialized file exists, each Markdown
  file exists, each readout has `schema_version=1`, and each readout preserves
  `source_artifact_path`.
- Live review queue now reports 25 pending first-pass summaries.

Next:

- Inspect a small sample of the five readouts in the review UI, then decide
  whether to submit the next prepared batch at size five or add an automatic
  post-materialization quality check before scaling.

## Turn 96 | 2026-05-16

Summary: Submitted the first live five-item first-pass summary batch through
the gated API.

Action:

- Checked Graphiti discovery, repo status, `transcripts.service`, and
  `auracall-api.service`.
- Verified the latest prepared manifest:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-191244.json`.
- Verified AuraCall `/status` returned `ok=true` and authenticated
  `/v1/models` returned HTTP 200 with the scoped transcript env key.
- Submitted the prepared manifest through
  `POST /api/review-queue/first-pass-summaries/submit`.
- Batch id: `batch_73f8ae99132741f2ba30a19905587b2c`.
- Fixed the review API's neutral status count adapter so provider aggregate
  counts are not double-counted with per-job statuses.

Validation:

- Submit returned HTTP 202 with `request_count=5`, `dry_run=false`, and the
  batch id above.
- Live status through `127.0.0.1:18876` returned `status=running`,
  `total=5`, `in_progress=5`, `completed=0`, `failed=0`, `cancelled=0`, and
  no materialized readouts yet.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest -q` passed.
- `python -m py_compile transcript_api.py tests/test_transcript_api.py` passed.
- `git diff --check` passed.

Next:

- Poll `batch_73f8ae99132741f2ba30a19905587b2c` until at least one child
  completes or fails, then materialize completed readouts and inspect any
  artifact extraction failures before scaling beyond five items.

## Turn 95 | 2026-05-16

Summary: Added gated submit and status actions for prepared first-pass summary
batch manifests.

Action:

- Added manifest-scoped submit and status endpoints under
  `/api/review-queue/first-pass-summaries/`.
- Submit requires an existing prepared manifest under
  `~/.local/state/transcribe-audio/first-pass-summary-batches/` and
  `approval_token=SUBMIT_FIRST_PASS_SUMMARY_BATCH`.
- Status polls an already submitted manifest and can materialize completed
  readouts when requested.
- Updated the React Review Queue action notice with submit and
  check/materialize controls after a batch is prepared.
- Documented the expanded batch action contract in README and
  `docs/dev/transcript-review-api.md`.

Validation:

- Graphiti discovery ran and repo files remained the implementation authority.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest -q` passed.
- `python -m py_compile transcript_api.py tests/test_transcript_api.py scripts/auracall_legacy_enrichment_batch.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Added a fake-provider API test proving submit requires the approval token,
  posts the prepared manifest payload, records the batch id, and polls status
  without touching a live provider.
- Restarted `transcripts.service`; live
  `POST http://transcripts.localhost/api/review-queue/first-pass-summaries/status`
  against the latest prepared manifest returned HTTP 200 with
  `status=prepared`, `request_count=5`, `dry_run=true`, and `batch_id=null`.

Next:

- Decide whether to submit the latest five-item prepared batch from the UI or
  keep batch execution paused until the next provider-readiness check.

## Turn 111 | 2026-05-22

Summary: Fixed Library pane ergonomics and readout/audio inspection.

Action:

- Replaced text-only pane collapse controls with accessible SVG icon controls.
- Added mouse drag and keyboard arrow resizing for the left filter pane and
  right inspector pane.
- Added selected-document detail loading in the Library inspector.
- Changed the Library inspector from raw JSON-first to human-readable readout
  preview cards with people, topics, actions, and risks when present.
- Kept raw context JSON behind a developer-labelled secondary link.
- Followed readout `source_artifact_path` metadata to the matching source
  transcript so readouts can surface source-transcript audio even when the
  readout itself has no blob.
- Increased the initial Library load to the API cap of 200 rows so current
  source-link resolution can find older source transcripts until a dedicated
  related-document endpoint exists.
- Verified the 2026-04-24 Cara/ColorBiotics readout points to source transcript
  `646f58f133108230a67c`, which has blob `f363a21cdc716795bd3c` linked to the
  original `.m4a`.
- Updated `ROADMAP.md` and the P09 plan for the UX/data-model correction.

Validation:

- `graphiti-runtime doctor` passed before the slice.
- `graphiti-runtime discover --group-id transcribe_audio_main "pane resize
  collapse controls human readable context audio missing Cara conference UX"`
  returned repo memory and no conflicting UX direction.
- `curl http://127.0.0.1:18876/api/documents/e2a67052289d1ad1d891`
  confirmed the readout has `source_artifact_path` pointing at the transcript
  sidecar.
- `curl http://127.0.0.1:18876/api/documents/646f58f133108230a67c`
  confirmed source recording playback/download URLs for blob
  `f363a21cdc716795bd3c`.
- `python -m py_compile transcript_api.py transcript_store.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Browser smoke against `http://transcripts.localhost/` selected the Cara row,
  verified the table shows `Source blob`, the inspector includes the human
  summary text, the raw JSON link is labelled `Developer: raw context JSON`,
  the old `Open context JSON` primary link is gone, the audio element points to
  `/api/blobs/f363a21cdc716795bd3c`, and both pane resize handles plus collapse
  controls are present with accessible labels.

Next:

- Add a dedicated related-document API endpoint for readout-to-source transcript
  resolution, then wire playback speed controls and richer transcript/readout
  tabs in the selected-artifact inspector.

## Turn 94 | 2026-05-16

Summary: Added a provider-neutral Review Queue action for first-pass summary
batch preparation.

Action:

- Added `POST /api/review-queue/first-pass-summaries/prepare`.
- The endpoint creates a dry-run first-pass summary manifest under
  `~/.local/state/transcribe-audio/first-pass-summary-batches/` and returns
  action metadata without exposing provider-specific UI details.
- Wired the React Review Queue first-pass summary card to call the endpoint
  with a five-item default and show the prepared manifest path.
- Documented the endpoint in README and `docs/dev/transcript-review-api.md`.
- Restarted `transcripts.service` so `transcripts.localhost` serves the new
  code.

Validation:

- Graphiti discovery ran and repo files remained the implementation authority.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest -q` passed.
- `python -m py_compile transcript_api.py scripts/auracall_legacy_enrichment_batch.py tests/test_transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Live `POST http://transcripts.localhost/api/review-queue/first-pass-summaries/prepare`
  returned HTTP 201 with `request_count=5`, `dry_run=true`,
  `batch_id=null`, workflow `transcribe-audio-first-pass-summary`, and artifact
  `first_pass_readout.json`.
- Live manifest:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-191159.json`.

Next:

- Add a provider-neutral submit/status action for prepared manifests, gated so
  the UI can start provider work only from an already prepared manifest and can
  poll/materialize results without exposing AuraCall internals.

## Turn 93 | 2026-05-16

Summary: Added a dry-run prepare path for bounded first-pass summary batches.

Action:

- Added `prepare` to `scripts/auracall_legacy_enrichment_batch.py`.
- `prepare` builds the same batch manifest as enqueue, but always leaves
  `dry_run=true`, `batch=null`, and does not submit provider work.
- Updated batch metadata and default manifest names from legacy-enrichment
  framing to first-pass summary framing.
- Updated the artifact contract from `legacy_readout.json` to
  `first_pass_readout.json`.
- Prepared a five-item first-pass summary batch manifest from the live queue:
  `~/.local/state/transcribe-audio/first-pass-summary-batches/first-pass-summary-prepare-20260516-230748.json`.
- Updated README to show `prepare` as the safe first command.

Validation:

- Graphiti discovery ran and repo files remained the implementation authority.
- `.venv/bin/python -m pytest tests/test_transcript_store.py::test_auracall_first_pass_prepare_writes_manifest tests/test_transcript_store.py::test_first_pass_summary_queue_lists_pending_imports tests/test_transcript_api.py -q` passed.
- `python -m py_compile scripts/auracall_legacy_enrichment_batch.py transcript_store.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- The prepared manifest has `request_count=5`, `dry_run=true`, `batch_id=null`,
  workflow `transcribe-audio-first-pass-summary`, and artifact file
  `first_pass_readout.json`.

Next:

- Add a provider-neutral backend/API prepare endpoint or action record for
  first-pass summary batches, so the UI can expose “Prepare batch” without
  knowing AuraCall or script internals.

## Turn 92 | 2026-05-16

Summary: Removed legacy framing from the first-pass summary queue.

Action:

- Updated the review API bucket from `legacy_enrichment` to
  `first_pass_summaries`.
- Updated the React Review Queue UI and inspector to show `First-pass
  summaries`.
- Added `transcript_store.py first-pass-summary-queue` as the preferred CLI
  command, leaving `legacy-enrichment-queue` as a compatibility alias.
- Updated the first-pass queue text output and stdout sentinel to use neutral
  naming for the preferred command.
- Kept implementation compatibility seams where renaming files/functions would
  create unnecessary churn.
- Wrote a neutral queue snapshot:
  `~/.local/state/transcribe-audio/first-pass-summary-queues/first-pass-summary-queue-20260516-230347.json`.

Validation:

- Graphiti discovery was healthy but returned only older unrelated facts; repo
  files were used as authority.
- `.venv/bin/python -m pytest tests/test_transcript_api.py tests/test_transcript_store.py -q` passed.
- `python -m py_compile transcript_api.py transcript_store.py scripts/auracall_legacy_enrichment_batch.py` passed.
- `npm --prefix frontend run build` passed.
- The queue snapshot contains 29 selected first-pass summary items and 1
  de-duped duplicate.
- Live `curl http://transcripts.localhost/api/review-queue?limit=5` reports
  `first_pass_summaries`, label `First-pass summaries`, count 29.

Next:

- Prepare a bounded first-pass summary batch from the neutral queue snapshot,
  without adding provider-specific or historical/import-specific language to
  the operator UI.

## Turn 91 | 2026-05-16

Summary: Added and applied a reviewed archive workflow for stale route reviews.

Action:

- Added `review_queue_maintenance.py`.
- The maintenance workflow plans stale local `*.route-review.json` files whose
  referenced route-decision JSON no longer exists.
- Dry-run mode is the default.
- Apply mode requires
  `--apply --approval-token ARCHIVE_STALE_ROUTE_REVIEWS`.
- Apply mode moves stale review files into
  `~/.local/state/transcribe-audio/review-queue-archive/<run-id>/` and writes
  an audit JSON.
- Updated README, P09 docs, API docs, and roadmap notes.
- Applied the reviewed live cleanup for stale pytest/temp route-review
  references.

Validation:

- Graphiti discovery was healthy but returned only older roadmap/runbook facts
  for this query; repo-local files were used as the implementation authority.
- Dry run reported 48 stale route-review candidates and 48 planned archive
  moves.
- Live apply archived 48 stale route-review files under
  `~/.local/state/transcribe-audio/review-queue-archive/20260516-223347/`.
- Live audit was written to
  `~/.local/state/transcribe-audio/review-queue-archive/stale-route-review-archive-20260516-223347.json`.
- `curl http://127.0.0.1:18876/api/review-queue?limit=100` now reports route
  reviews as `clear`, 0 route-review items, and 29 pending first-pass summary
  items.
- `.venv/bin/python -m pytest tests/test_review_queue_maintenance.py tests/test_transcript_api.py -q` passed.
- `python -m py_compile review_queue_maintenance.py transcript_api.py` passed.
- `npm --prefix frontend run build` passed.

Next:

- Add a first-pass summary queue action surface that can prepare a reviewed
  bounded batch from the 29 pending readouts without starting provider work
  from the UI.

## Turn 90 | 2026-05-16

Summary: Added live read-only review queue data to the transcript console.

Action:

- Added `--state-dir` to `transcript_api.py`, defaulting to
  `~/.local/state/transcribe-audio`.
- Added `GET /api/review-queue` to aggregate route-review files,
  filename-conflict decisions, and first-pass summary queue counts.
- Route-review items now report whether their referenced route-decision JSON
  still exists; stale temp/pytest references are surfaced as
  `stale_reference` instead of hidden or deleted.
- Replaced hard-coded React review cards with live `/api/review-queue` data.
- Updated the Review Queue inspector to show the runtime state root and live
  route, filename-conflict, and first-pass summary queue summaries.
- Updated `ROADMAP.md`, the P09 plan, README, and API docs for the endpoint.

Validation:

- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed.
- `python -m py_compile transcript_api.py` passed.
- `npm --prefix frontend run build` passed.
- Live `curl http://127.0.0.1:18876/api/review-queue?limit=100` returned
  buckets for 48 stale route-review references, 0 open filename conflicts, and
  29 pending first-pass summary items.
- Live `curl http://transcripts.localhost/api/review-queue?limit=5` returned
  HTTP 200 through local ingress.

Next:

- Add a safe local cleanup or archive workflow for stale route-review files so
  pytest/temp references do not dominate the Review Queue surface.

## Turn 89 | 2026-05-16

Summary: Pinned the transcript review console for cooper ingress.

Action:

- Selected port `18876` as the fixed transcript-console upstream port.
- Updated `transcript_api.py` to default to port `18876`.
- Added static frontend serving from `frontend/dist/` at `/` while keeping API
  routes under `/api`.
- Ensured unknown `/api/...` routes return JSON `404` instead of falling
  through to the SPA.
- Updated frontend dev proxy and docs to use port `18876`.
- Installed and enabled the user systemd service
  `~/.config/systemd/user/transcripts.service`.
- Registered cooper ingress inventory for `transcripts.localhost` and the
  Authelia-gated external host `transcripts.ecochran.dyndns.org`.
- Published the bastion Traefik route and added the matching bastion Authelia
  `one_factor` access rule.

Validation:

- `npm --prefix frontend run build` passed.
- `.venv/bin/python -m pytest tests/test_transcript_api.py -q` passed.
- `python -m py_compile transcript_api.py` passed.
- `curl http://127.0.0.1:18876/api/health` returned HTTP 200 against
  `/home/ecochran76/.transcripts/transcripts.sqlite3`.
- `curl http://transcripts.localhost/api/health` returned HTTP 200.
- `curl http://transcripts.localhost/` returned HTTP 200 for the built review
  console.
- Static HTML/assets were checked for raw-port leakage; no
  `localhost:18876`, `127.0.0.1:18876`, `localhost:5174`, or
  `127.0.0.1:5174` references were found.
- `curl https://transcripts.ecochran.dyndns.org/` returned HTTP 302 to
  `https://auth.ecochran.dyndns.org/...`, confirming unauthenticated external
  access is Authelia-gated.
- `systemctl --user restart transcripts.service` completed and
  `systemctl --user is-active transcripts.service` returned `active`.

Next:

- Add read-only API support for review queue manifests under
  `~/.local/state/transcribe-audio/`, then replace the frontend's hard-coded
  queue summary with live review queue data.
