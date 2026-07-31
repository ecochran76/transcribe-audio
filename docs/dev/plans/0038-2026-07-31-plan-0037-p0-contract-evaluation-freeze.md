# Plan 0038 | Plan 0037 P0 contract and evaluation freeze

State: CLOSED

Lane: P10

Plan Version: 1

Parent: Plan 0037 P0

Owner: primary agent

Expected Write Surface: `acoustic_identity_contracts.py`,
`acoustic_evaluation_corpus.py`, `docs/dev/fixtures/plan-0037-p0/`, focused
tests, Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`; private receipts only under
`~/.local/state/transcribe-audio/plan-0037/`.

## Vision alignment

This packet advances trustworthy speaker inference and reusable source
evidence. Acoustic identity is currently `1 - Built/experimental`; P0 keeps
the target at `1` while making the contracts and blind corpus reproducible
enough for P1-P4 to pursue `2 - Shadow`.

Measured effect: later acoustic comparisons share one conversation-separated,
operator-confirmed corpus and cannot silently leak raw biometric material into
portable evidence.

Evidence: schema-contract tests, a private frozen-corpus receipt, permission
readback, a reviewed model/license inventory, and an independent P0 audit.

This packet does not process audio, enroll a biometric profile, choose a model,
calibrate a threshold, reveal Plan 0036 predictions, or promote acoustic
evidence into App Intelligence.

## Scope

- Freeze versioned contracts for derived audio, quality, biometric profiles,
  verification trials, acoustic evidence, and historical reprocessing.
- Implement fail-closed checks that portable artifacts cannot contain raw
  embeddings or unrestricted enrollment audio.
- Build an immutable private corpus manifest from current operator-confirmed
  gold references and stored source-audio blobs without reading sealed
  prediction bodies.
- Enforce conversation-level split ownership and record incomplete condition
  metadata explicitly for later quality measurement.
- Inventory candidate code and checkpoint terms, acquisition state, revision,
  and hash requirements before dependencies are added.
- Record delegation and validation evidence.

## Non-goals

- No new audio/model dependency installation.
- No audio decoding, VAD, enhancement, diarization, embedding extraction, or
  identity scoring.
- No new gold review or change to the sealed Plan 0036 baseline.
- No raw transcript, speaker name, email, embedding, or audio content in the
  repository.

## Current state

Plan 0037 and its research authority remain open. P0 is complete. The current
private corpus is
`acoustic-corpus-1f93d1405f82676420571e1b`; it contains 24 latest eligible
operator-confirmed recordings in 24 disjoint conversations. Source blobs,
transcript-store files, the manifest, and the receipt now satisfy the private
mode contract. Acoustic condition measurements remain explicitly unassessed
until P1/P2 and therefore cannot support model promotion.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P0A contract freeze | primary | none | schemas and privacy invariants pass |
| P0B license inventory | primary | research note | every candidate has explicit code/checkpoint posture |
| P0C corpus freeze | primary | P0A | private immutable manifest, conversation splits, hashes, and modes pass |
| P0D neutral audit | delegated read-only reviewer | P0A-P0C | findings reconciled or packet remains open |

Intended concurrency is two active agents: the primary owns all writes; one
read-only reviewer independently audits P0. The join is P0D. No retry loop may
extend past one repair-and-rerun cycle without recording the remaining defect.

## Acceptance criteria

- Six versioned artifact families have explicit required fields, privacy
  classes, and validators.
- Portable acoustic evidence and reprocessing artifacts reject raw embedding
  or unrestricted enrollment-audio fields recursively.
- The private corpus contains only latest `eligible_known` operator gold
  references with accessible, hash-matched source blobs.
- Each durable conversation belongs to exactly one split; no recording/window
  from that conversation can cross splits.
- The corpus records exact denominators, source hashes, gold references,
  prediction exclusion, and incomplete acoustic-condition metadata without
  fabricating labels.
- Private directories are `0700` and files are `0600`; selected source blobs
  are no broader than `0600`.
- Model and processing candidates record code license, checkpoint/data terms,
  acquisition status, pinned revision/hash state, source URL, and promotion
  blocker.
- No dependency, enrollment, prediction reveal, prompt integration, or
  external write occurs.

## Validation

- `pytest -q tests/test_acoustic_identity_contracts.py tests/test_acoustic_evaluation_corpus.py`
- Generate the private corpus freeze against current `~/.transcripts` and the
  current speaker-evaluation campaign.
- Re-run the freeze to prove idempotence and immutable-conflict detection.
- Verify manifest/source modes and hashes from live filesystem readback.
- Run the repo planning-contract audit and the joined regression suite relevant
  to transcript artifacts and speaker evaluation.
- Reconcile the delegated review receipt in `RUNBOOK.md`.

## Terminal condition

Close this packet only when storage, privacy, benchmark/corpus, and
model-license inventory evidence all pass. Otherwise leave it `OPEN` and name
the exact failed gate; P1 must not begin.

## Closeout

State transition: `OPEN/Plan-0037-P0 -> CLOSED/Plan-0037-P0`.

- Frozen six versioned artifact families with lifecycle, timestamp-map,
  conversation-split, portable-biometric exclusion, approval, replay, and
  non-overwrite validation.
- Added a reviewed seven-candidate code/checkpoint license inventory. No
  candidate was acquired; unresolved revisions, hashes, dataset terms, and
  gated access remain fail-closed promotion blockers.
- Frozen 24 recordings, 24 conversations, 35 pseudonymous subjects, 105
  speaker labels, 293 feasible same-person pairs, and 2,042 feasible
  different-person pairs. Split counts are 16 development, 3 calibration,
  and 5 evaluation.
- Hardened the complete user-scoped transcript store to `0700` directories and
  `0600` files and made those modes persistent for new database, artifact, and
  source-blob writes.
- Private manifest SHA-256:
  `73f0e04aab0274ddfeaa7f6b1567ecb135eebc0a0d6e5818cb3bd2ee5535dabf`.
  Unchanged replay returned the same corpus identity and receipt.
- Delegation receipt: `/root/p0_audit`, read-only independent audit. Its first
  pass found lifecycle, timestamp, split, store-boundary, idempotence,
  permission, benchmark, and recursive-leak gaps. Its terminal pass found
  three blockers: gold provenance binding, reprocessing apply binding, and
  planning wiring. All three were repaired before closure.
- Validation: 39 focused contract/corpus/store tests, 122 joined regression
  tests, and 423 full tests passed. No model ran, no prediction was revealed,
  no enrollment occurred, and no external write occurred.

Next: derive and execute the bounded P1 audio-derivative and quality packet.
