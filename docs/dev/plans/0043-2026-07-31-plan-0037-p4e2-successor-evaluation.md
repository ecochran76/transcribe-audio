# Plan 0043 | Plan 0037 P4E2 successor terminal evaluation

State: OPEN

Lane: P10

Plan Version: 5

Parent: Plan 0037 P4

Owner: primary agent

Expected Write Surface: `acoustic_speech_preparation.py`,
`speaker_evaluation_campaign.py`, focused tests, privacy-safe successor
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

The same frozen campaign contains 236 unreviewed reviewable items. Current
store metadata proves seven have current durable conversation/recording IDs,
accessible source blobs, no overlap with any prior corpus by document,
conversation, recording, or source SHA-256, and no duplicate identity within
the pool. Those seven contain 18 opaque speaker labels across seven multi-label
recordings. The other 229 lack durable identities and are outside this packet;
they must not be silently assigned IDs or rewritten merely to enlarge the
terminal cohort.

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
- Freeze the new cohort, conversation-disjoint split policy, prediction-excluded
  operator gold, and exact trial manifest before any model output. The primary
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
- P4E2-C is blocked while the fully disjoint candidate count is zero. New
  operator-confirmed recordings must be incorporated through the governed
  speaker-evaluation campaign; existing revealed evidence cannot substitute.
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
- Plan 0037 remains open. P4E2-C2 through P4E2-E remain `not_run` until new
  eligible source evidence exists; absence of evidence is not a failed model.
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

## P4E2-C1 opening checkpoint

State: OPEN; successor campaign not yet applied and no case review opened.

- The parent campaign already freezes all 375 transcript rows, so a second
  unfiltered oldest-forward campaign would repeat the same source authority.
- Of 236 unreviewed reviewable rows, seven current records satisfy the bounded
  successor gate: durable recording/conversation IDs, current artifact hash,
  accessible source blob, full prior-corpus disjointness, and pairwise pool
  disjointness. They contain 18 opaque speaker labels. The remaining 229 lack
  durable IDs and are excluded without mutation.
- P4E2-C1 will freeze those seven into a child campaign before opening the first
  review packet. No prediction, prior gold body, transcript text, person name,
  or audio is part of the portable selection evidence.
- This is an availability-conditioned census, not a representativeness claim.
  Independent recording/conversation/subject-session and measured-condition
  gates remain separate from combinatorial window trial counts and can force
  `stop`/`refine` after review.
- Implementation validation is complete before runtime apply: the private
  identity-scalar projection, exact-seven selector, reviewed-hash-bound apply,
  full-body replay, and chained one-case cursor passed all 16 campaign tests
  and all 549 repository tests, plus compilation and `git diff --check`.
  Independent read-only review returned `PASS` after two bounded repair cycles.
