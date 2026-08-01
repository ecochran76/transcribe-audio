# Plan 0043 | Plan 0037 P4E2 successor terminal evaluation

State: OPEN — P4E2-R1 device provenance refinement

Lane: P10

Plan Version: 10

Parent: Plan 0037 P4

Owner: primary agent

Expected Write Surface: `acoustic_speech_preparation.py`,
`speaker_evaluation_campaign.py`, `acoustic_evaluation_corpus.py`, focused
tests, privacy-safe successor
readiness/review-tranche receipts, this plan, Plan 0037, `ROADMAP.md`, and
`RUNBOOK.md`; private receipts only under user-scoped Plan 0037 and speaker-
evaluation campaign roots.

## Vision alignment

P4E2 advances the north-star outcome of calibrated, safely abstaining acoustic
speaker evidence. Current maturity remains `2 - Shadow`: development and
held-out calibration evidence exist, but no valid unseen terminal evaluation
exists. The target remains `2 - Shadow` with a replayed terminal
`select`/`refine`/`reject`/`stop` decision; only a later plan may advance toward
`3 - Operational`.

Progress is measured by a valid pre-reveal preparation seam, a genuinely new
conversation/source-disjoint cohort, exact same-person/different-person/open-set
trial minima, condition slices, terminal decision replay, and zero leakage of
raw audio, transcripts, names, gold bodies, or embeddings into portable
evidence.

## Current state

P4E generation 1 is closed with terminal `STOP`. Its five revealed evaluation
recordings are nonblind and permanently ineligible for terminal selection.
Generation 1 executed zero preparation, audio, model, score, trial, or metric
operations.

A read-only 2026-07-31 inventory of the source campaign found 24 latest
eligible operator-confirmed recordings, all 24 already present in the original
P0 corpus. After excluding overlap by document, recording, conversation, and
source SHA-256, the successor candidate count is zero. The inventory did not
read model predictions or expose gold, names, transcript text, or source audio.

The same frozen campaign contained 236 unreviewed reviewable items. Seven had
current durable conversation/recording IDs, accessible source blobs, no overlap
with either prior corpus by document, conversation, recording, or source
SHA-256, and no duplicate identity within the pool. Those seven were frozen as
successor campaign `campaign-a2165fb6568ca0e9c40d`; the operator completed all
seven prediction-excluded reviews, including append-only identity corrections.
The superseding gold freeze is
`7870394e-417f-40f0-8e04-3de5e1fa130b`, SHA-256
`70ca36436a41a0a16c37eb295783e82f48cf8b2b57735c6d6db64c1e150d7d13`.
Its current gold-index SHA-256 is
`59e443b41ea2b2fa9f4e1d7c33df3e80988750993cedb0ca4b99efb1c70e83df`.
The other 229 lack durable IDs and remain outside this packet.

## Scope

- Add `evaluation` as an explicit later-split mode in the P2 dry-run, apply,
  replay, and lineage contract. It must require an exact 64-hex split-access
  authority just like calibration and preserve the actual split name in every
  receipt.
- Add focused positive and negative tests proving evaluation succeeds only
  with an exact authority, replays deterministically, and cannot be confused
  with calibration or development.
- Persist a private metadata-only readiness receipt that compares the latest
  eligible operator-gold records with all prior frozen corpus recordings and
  reports only counts, split counts, opaque set hashes, readiness status, and
  blockers.
- Freeze one deterministic successor operator-review campaign from the seven
  currently durable, source-available, fully disjoint candidates. Bind the
  parent campaign/manifest/gold-index hashes, all prior corpus hashes, current
  artifact/source hashes, durable recording/conversation IDs, chronological
  selection order, exact membership in the frozen parent's unreviewed future
  pool, and non-configurable seven-item denominator before any case review.
  First apply must name the independently reviewed preview content hash and
  manifest ID and reject any intervening parent/corpus/store drift.
- Present the seven cases one at a time in concise plain English. Each gold
  record must be an attributable operator decision with prediction visibility
  excluded; the agent may not infer person identity, calendar outcome, or an
  eligible-known disposition from model output, existing suggestions, or
  silence.
- After reviews, measure eligible recordings, opaque subjects, speaker labels,
  same-person session pairs, different-person session pairs, and achievable
  frozen trial counts. Proceed only if the unchanged terminal minima are
  feasible; otherwise record the exact evidence shortfall.
- Treat the seven records as an availability-conditioned census of the complete
  technically eligible pool, not as representative population evidence. The
  20/100/20 window/profile trial counts are computational coverage only. A
  terminal `select` additionally requires seven distinct recordings and
  conversations, at least two operator-confirmed subjects each present in at
  least two conversations, at least four independent same-person subject-
  session pairs, at least five distinct known subjects, and measured (not
  `unassessed`) channel, device, noise, telephone-bandwidth, and usable-duration
  slices with at least two observed values in every decision-relevant condition
  dimension. Missing independent or condition coverage forces `stop` or
  `refine`, never `select`.
- Keep authority construction, split reveal, audio access, model execution,
  scoring, and terminal selection stopped until a genuinely new cohort exists
  and a reviewed generation-2 authority binds the already-implemented seam.
- Freeze the new cohort with the precommitted chronological-rank quota policy
  `3 development / 2 calibration / 2 evaluation`, prediction-excluded operator
  gold, and exact trial manifest before any model output. This deterministic
  census split is independent of model output. The default hash policy is
  rejected for this cohort because its measured result is `6 / 1 / 0`, leaving
  evaluation empty. The primary
  may execute the sealed packet only after independent metadata review; the
  reviewer may inspect hashes, schemas, counts, and receipts but not prediction
  bodies, gold bodies, raw audio, transcript text, names, or embeddings.
- Precommit the candidate matrix, trial-construction policy, terminal decision
  policy, and global stop precedence before reveal. No threshold, margin,
  candidate unit, trial, grouping rule, metric rule, or decision rule may change
  after reveal in the same generation.
- When new evidence exists, freeze the successor cohort before reading its
  terminal split and require at least 20 genuine, 100 impostor, and 20 open-set
  trials for every frozen model-by-method unit, plus explicit condition slices.

## Non-goals

- Do not reuse any original 24-recording corpus item for terminal selection.
- Do not weaken trial minima to fit the currently empty cohort.
- Do not reveal a successor split, read audio, run models, change thresholds,
  select defaults, integrate App Intelligence, or reprocess history in the
  seam/readiness packet.
- Do not put raw gold, person identifiers, names, transcript text, source
  paths, audio, embeddings, or credential values in repo or portable receipts.
- Do not normalize or rewrite the 229 legacy transcript artifacts that lack
  durable identities in this packet. A future identity-migration packet must
  preserve original evidence and independently review its grouping semantics.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P4E2-A inventory and design | primary plus read-only reviewer | P4E generation-1 STOP | disjoint-count receipt and reviewed plan exist |
| P4E2-B P2 evaluation seam | primary | P4E2-A | dry-run/apply/replay/lineage tests pass for all three split modes |
| P4E2-C1 successor review tranche | primary plus read-only reviewer | P4E2-B plus durable disjoint candidates | immutable seven-case campaign exists and operator reviews proceed one case at a time |
| P4E2-C2 successor cohort freeze | primary plus read-only reviewer | P4E2-C1 operator gold | no overlap with any prior corpus and evidence-feasible terminal policy |
| P4E2-D authority and reveal | primary plus read-only reviewer | P4E2-C2, complete replayed authority chain | independent `READY_TO_REVEAL` audit |
| P4E2-E terminal execution | primary plus read-only reviewer | P4E2-D | exact frozen matrix completes or fails closed, then one replayed terminal decision |

Intended concurrency is two agents. The primary owns writes and private
execution. The existing read-only reviewer audits design and terminal evidence.
One bounded repair-and-rerun cycle is allowed per audit; a residual finding
keeps the affected unit open.

## Gates and stop conditions

- The operator's blanket authorization removes per-command authorization
  rituals; content-addressed authorities remain integrity and scope controls.
- P4E2-B may run now because it reads no sealed split, audio, gold body, or
  model. The P2 module hash must stabilize before P4E2-D authority creation.
- P4E2-C1 is closed. All seven fully disjoint candidates now have attributable,
  prediction-excluded operator gold and an immutable superseding gold freeze.
  P4E2-C2 may construct and freeze the exact successor census; existing
  revealed evidence still cannot substitute.
- P4E2-C1 may freeze the seven current candidates before operator review. It
  first projects only `conversation_id` and `recording_id` scalars for the
  frozen parent's unreviewed future pool into a private, content-addressed
  authority. Campaign selection then reads that projection plus current
  document/artifact/blob identity metadata; it cannot query `json_payload` or
  `text_content`, invoke the legacy full-store preview, or open a transcript
  artifact. A transcript body may open only for the single current review
  packet. It must not execute App Intelligence, open any prediction, preassign
  gold, or expose another case concurrently.
- The child campaign binds the selector module SHA-256, selection schema and
  rule version, exact metadata-projection hash, content hash, and stored
  manifest SHA-256. Apply and replay require full manifest-body equality apart
  from the declared application timestamp; manifest-ID equality alone is
  insufficient.
- Opening a successor review packet requires a manifest/current-gold-index-
  bound immutable cursor receipt. Only the next unreviewed manifest item may
  open; the same outstanding case may reopen idempotently, any other case is
  rejected, and the cursor advances only after an attributable gold record for
  the outstanding document exists. Every open receipt has an exact schema,
  content hash, previous-receipt link, and fail-closed privacy flags; added,
  removed, or changed fields invalidate the cursor history.
- The generation-2 authority must bind and replay the new corpus manifest,
  split policy and seal, prediction-excluded frozen gold authority, P1
  derivative/quality manifests, P2 dry-run/comparison/apply/replay manifests and
  module hash, channel/downmix and window-selection rules, active P3/P4 profile
  and lifecycle eligibility, exact model revisions/assets, all nine frozen
  threshold and zero-margin values, the complete model-by-method candidate
  matrix, the exact trial manifest/construction rules, and terminal metric and
  decision policies.
- Every frozen candidate unit runs on the same frozen trial set. A missing unit,
  incomplete trial class, non-finite score or metric, absent required condition
  slice, denominator below policy, or runtime/integrity failure produces the
  precommitted global `stop`; it cannot be silently dropped or replaced.
- Any overlap by document, recording, conversation, or source SHA-256, missing
  trial denominator, hash drift, private-mode failure, or residual independent
  review finding stops before reveal or execution.
- Hugging Face credentials from `~/credentials/API-keys.env` may be loaded only
  if a later in-scope model operation requires them; values must never be
  printed, persisted in receipts, or sent to a reviewer.

## Acceptance and validation

- P2 supports `development`, `calibration`, and `evaluation` with fail-closed
  field schemas and exact replay/lineage binding.
- The readiness receipt is deterministic apart from its declared timestamp,
  private (`0600` under `0700`), metadata-only, and reports the current zero
  disjoint-candidate blocker truthfully.
- Before reveal, the exact frozen matrix demonstrates all model-by-method units
  can receive at least 20 genuine, 100 impostor, and 20 open-set trials from the
  same trial manifest. Terminal execution records attempted/success/failed/
  blocked denominators for every unit, finite-score and finite-metric coverage,
  required condition slices, and global stop reduction.
- The sole terminal outcome is replayed as `select`, `refine`, `reject`, or
  `stop`. `Select` requires the complete matrix and every frozen safety rule;
  incomplete evidence can only `stop`, never preselect a surviving unit.
- Focused P2/readiness tests, the full repository suite, compilation, and
  `git diff --check` pass.
- Independent review returns `PASS` for this bounded packet before commit/push.
- Plan 0037 remains open. P4E2-C1 is closed; P4E2-C2 implementation and audit
  are active; P4E2-D and P4E2-E remain `not_run`. Missing measured-condition
  evidence remains a pending gate, not a failed model.
- The successor campaign preview/apply is deterministic and idempotent, fails
  closed on parent/corpus/store drift, uses `0600` files under `0700`
  directories, and proves all seven candidates are pairwise and prior-corpus
  disjoint before operator review starts.
- A guarded selector test rejects any access to `documents.json_payload`,
  `text_content`, the legacy full-store preview, or transcript-artifact bodies;
  the separately frozen identity-scalar projection is hash-bound to the parent
  manifest, parent gold index, module, repository state, and exact future pool.
- Selector replay recomputes the full current preview from its bound parent,
  corpus, store, policy/schema, and module authorities and matches the stored
  child manifest body and SHA-256 exactly. Cursor tests cover first open,
  idempotent reopen, out-of-order/concurrent rejection, gold-bound advance, and
  manifest/index tamper rejection.

## P4E2-A/P4E2-B checkpoint

State: CLOSED; P4E2-C through P4E2-E are `not_run`.

- The P2 preparation contract now accepts `evaluation` as an exact-authority
  later split across dry-run, apply, replay, and lineage. Development remains
  authority-free, calibration behavior remains compatible, and unknown split
  names fail closed. The resulting pre-authority P2 module SHA-256 is
  `700e10d802a6443eab9d2bb9c6b9a7519cff26021ffec23acbdb767f12bcd595`.
- Authoritative private readiness receipt
  `cae5de01dd91d7f620b17071dd87dbc4ae991793f07a4257186d18d3e587287d`
  records 24 latest eligible candidates, 24 overlaps in each of the document,
  recording, conversation, and source-hash dimensions, and zero fully disjoint
  candidates. It uses only campaign, gold-index, prior-corpus, and transcript-
  store metadata; it does not open transcript bodies or source blobs. It is
  `0600` under `0700` and performs no split reveal, model execution, or external
  write.
- Earlier receipt
  `85769c302b1e6e762d77b8ab809850cb688aa7d10cfa843acd7eb627e7bd010d`
  is invalidated as evidence because its collector rehashed source blobs while
  claiming no audio read. It is retained only as non-authoritative audit
  history; the metadata-only receipt above supersedes it.
- The authoritative receipt blockers are
  `no_fully_disjoint_operator_confirmed_candidates`,
  `same_person_pair_feasibility_not_demonstrated`, and
  `different_person_pair_feasibility_not_demonstrated`. They stop cohort freeze
  without converting absent evidence into model failure.
- Validation: 27 focused tests and 547 full repository tests passed; compilation
  and `git diff --check` passed.
- Independent read-only audit returned `PASS` after verifying no transcript/body
  or source-blob dereference, exact receipt/module hashes and private modes,
  focused/full tests, compilation, and diff integrity.

## P4E2-C1 close / P4E2-C2 opening checkpoint

State: P4E2-C1 CLOSED; P4E2-C2 OPEN.

- The parent campaign already freezes all 375 transcript rows, so a second
  unfiltered oldest-forward campaign would repeat the same source authority.
- Of 236 unreviewed reviewable rows, seven current records satisfy the bounded
  successor gate: durable recording/conversation IDs, current artifact hash,
  accessible source blob, full prior-corpus disjointness, and pairwise pool
  disjointness. They contain 18 opaque speaker labels. The remaining 229 lack
  durable IDs and are excluded without mutation.
- The exact-seven child campaign was frozen and replayed at commit `2a432b4`.
  All seven cases were reviewed one at a time with prediction visibility
  excluded. Append-only corrections unified the recurring Eric identity and
  corrected the first case's Jordan Katz display identity; the earlier gold
  freeze remains audit history and the superseding freeze above is current.
- This is an availability-conditioned census, not a representativeness claim.
  Independent recording/conversation/subject-session and measured-condition
  gates remain separate from combinatorial window trial counts and can force
  `stop`/`refine` after review.
- Implementation validation is complete before runtime apply: the private
  identity-scalar projection, exact-seven selector, reviewed-hash-bound apply,
  full-body replay, and chained one-case cursor passed all 16 campaign tests
  and all 549 repository tests, plus compilation and `git diff --check`.
  Independent read-only review returned `PASS` after two bounded repair cycles.
- Current C2 evidence is 7 recordings, 7 conversations, 7 source hashes, 18
  speaker labels, 10 known subjects, 3 subjects recurring across conversations,
  and 23 independent same-person subject-session pairs. Both prior frozen
  corpora have zero overlap in all four governed identity dimensions.
- The successor corpus implementation must preserve legacy hash-split behavior
  while adding exact-seven chronological `3 / 2 / 2` assignment, superseding-
  gold-freeze and prior-corpus hash bindings, clean repository/module binding,
  reviewed preview-hash apply, full-body replay, idempotence, private modes, and
  tamper rejection. Passing C2 means only `ready_for_p1_measurement`; terminal
  selection remains pending until P1/P2 replace every decision-relevant
  `unassessed` condition and prove at least two observed values per dimension.

## P4E2-C2 close / P4E2-D opening checkpoint

State: P4E2-C2 CLOSED; P4E2-D OPEN.

- Commit `50f34ab3fd36f7b00ece776c35c9d9e05c3571f3` added the
  successor-only chronological `3 / 2 / 2` split, exact-two prior-corpus and
  superseding-gold-freeze bindings, live source/transcript/gold/index drift
  validation, reviewed-content-hash apply, fail-before-write readiness, and
  read-only exact-manifest/receipt replay. Legacy corpus content identity is
  preserved.
- Independent read-only review returned `PASS` on exact diff SHA-256
  `fdf8cded9c96926aae03bbacc2e88cade05e550d1430bd37d85ed62a3afd0c6f`
  after all authority, replay, and compatibility findings were repaired.
  Five focused tests and all 551 repository tests passed; compilation and
  `git diff --check` passed.
- The private corpus is `acoustic-corpus-4a2b13e7bdc201f694af2f43`, content
  SHA-256 `4a2b13e7bdc201f694af2f43d4ab845749eeeb3ea06c7a97a40164cab40b83fe`,
  manifest SHA-256
  `4b77479d25d7b248cc62d500ed84c1604f105848da25ecef53661c5d9ea05a30`.
  Exact replay returned `full_body_match=true`; manifest and receipt are `0600`
  under a `0700` corpus directory.
- Frozen denominators are 7 recordings, 7 conversations, 18 speaker labels,
  10 known subjects, 3 recurrent subjects, 23 feasible same-person pairs, and
  114 feasible different-person pairs. Split counts are 3 development, 2
  calibration, and 2 evaluation. Both prior corpus overlap counts remain zero.
- C2 status is only `ready_for_p1_measurement`; `promotion_eligible=false`.
  P4E2-D now delegates its bounded condition-measurement packet to Plan 0044,
  which must measure and bind channel, device, noise, telephone-bandwidth, and
  usable-duration evidence before constructing the independently reviewed
  generation-2 terminal authority. No biometric score or terminal selection
  has run.

## P4E2-D condition close / terminal stop checkpoint

State: P4E2-D CLOSED with terminal `STOP`; P4E2-E is `not_run`.

- Plan 0044 executed its reviewed private condition authority from clean pushed
  commit `837edf02e67d113d38819937acf5833a2fbd0db3`. Exact execution
  completed 7 P1 successes and 35 P2 method successes; full-body replay passed.
- Channel, noise, telephone-bandwidth, and usable-duration each have two
  observed values with zero missing recordings. Device has zero observed values
  and seven missing recordings and is the sole terminal blocker.
- Independent metadata-only audit returned `PASS`, including exact `3 / 2 / 2`
  split membership, all method/replay hashes, current readiness/module/repo
  bindings, and private modes.
- Because the precommitted decision policy requires two genuine observed
  physical-device values with no missing recordings, generation-2 authority
  construction, split reveal, biometric scoring, and terminal selection remain
  stopped. Encoding profiles cannot satisfy this gate. A later refinement must
  obtain explicit capture-device provenance or freeze a genuinely eligible new
  cohort; it may not reinterpret this result as model failure or `select`.

## P4E2-R1 device provenance refinement opening checkpoint

State: Plan 0045 OPEN; the Plan 0044 terminal `STOP` remains in force.

- Read-only structural, container, extended-attribute, and original-path
  inventory confirmed that the exact seven recordings have no explicit
  physical capture-device provenance available to the current runtime.
- Plan 0045 owns a separate append-only, exact-seven, one-case-at-a-time
  operator-attestation authority. It preserves the frozen Plan 0044 result and
  may overlay only directly known physical-device facts.
- Generation-2 authority construction, evaluation reveal, biometric scoring,
  and selection remain `not_run` until the composite authority independently
  proves seven nonmissing direct attestations and at least two distinct devices.

## P4E2-D2 historical calibration replay checkpoint

State: Plan 0046 Unit A CLOSED; Plan 0045 and Plan 0046 remain OPEN.

- The sole generation-2 descendant replay mismatch was isolated to the
  archived calibration authority's P2 module hash versus the later reviewed
  evaluation-split seam. Plan 0046 now admits only that exact transition under
  a replay-only contract; normal calibration replay remains strict.
- The compatibility route cannot enter calibration build, reveal, preparation,
  selection, or apply functions. Dedicated read-only stage validation preserves
  the existing profile, descendant, 396-trial score, and nine-threshold checks.
- Production calibration replay passed with no immutable writer call and no
  hash or mtime change across the six archived authority/stage/application
  artifacts. Independent re-audit returned `PASS`; all 585 tests passed.
- This closes only the calibration descendant seam. Generation-2 preview is
  next, while production authority apply, reveal, scoring, and selection remain
  stopped behind the Plan 0045 composite device-condition gate.

## P4E2-D3 generation-2 pre-reveal authority checkpoint

State: Plan 0046 Units A through C CLOSED; Plan 0045 and Unit D remain OPEN.

- The independently audited, deterministic generation-2 pre-reveal preview
  binds the exact successor seal, historical condition and calibration chains,
  current model/profile assets, nine frozen thresholds and margins, candidate
  matrix, terminal policy, and mandatory exact-trial child contract.
- Exact production historical-condition replay rehashed all seven P1 artifacts
  and all 35 P2 outputs. Frozen successor projection `bbadd46c...befbd` and
  canonical composite content/ID recomputation prevent detached caller input.
- Independent re-audit returned `PASS`; 22 focused, 56 joined, and all 607
  repository tests passed. No evaluation reveal, model execution, score,
  terminal metric, decision, or external write occurred.
- Production generation-2 freeze remains `not_run` until Plan 0045 produces a
  passing replayed composite with seven direct nonmissing device attestations
  and at least two distinct opaque devices.
