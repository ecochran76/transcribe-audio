# Plan 0043 | Plan 0037 P4E2 successor terminal evaluation

State: OPEN

Lane: P10

Plan Version: 3

Parent: Plan 0037 P4

Owner: primary agent

Expected Write Surface: `acoustic_speech_preparation.py`, focused tests, one
privacy-safe successor-readiness module and receipt, this plan, Plan 0037,
`ROADMAP.md`, and `RUNBOOK.md`; private receipts only under
`~/.local/state/transcribe-audio/plan-0037/verification-calibration/`.

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

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P4E2-A inventory and design | primary plus read-only reviewer | P4E generation-1 STOP | disjoint-count receipt and reviewed plan exist |
| P4E2-B P2 evaluation seam | primary | P4E2-A | dry-run/apply/replay/lineage tests pass for all three split modes |
| P4E2-C successor cohort freeze | primary plus read-only reviewer | new operator-confirmed recordings | no overlap with any original P0 item and evidence-feasible terminal policy |
| P4E2-D authority and reveal | primary plus read-only reviewer | P4E2-C, complete replayed authority chain | independent `READY_TO_REVEAL` audit |
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
- Plan 0037 remains open. P4E2-C through P4E2-E remain `not_run` until new
  eligible source evidence exists; absence of evidence is not a failed model.

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
