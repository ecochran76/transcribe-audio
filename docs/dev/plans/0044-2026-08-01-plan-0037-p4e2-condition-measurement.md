# Plan 0044 | Plan 0037 P4E2 successor condition measurement

State: OPEN

Lane: P10

Plan Version: 1

Parent: Plan 0043 P4E2-D

Owner: primary agent

Expected Write Surface: `acoustic_successor_conditions.py`, focused tests,
this plan, Plan 0043, `ROADMAP.md`, and `RUNBOOK.md`; private P1/P2 artifacts
and condition receipts only under the user-scoped Plan 0037 runtime root.

## Vision alignment

This packet advances calibrated, safely abstaining acoustic speaker evidence.
Current and target maturity remain `2 - Shadow`: the successor corpus is
frozen, but its decision-relevant acoustic conditions are still unmeasured.
Progress is proved by replayed private P1/P2 evidence for all seven recordings,
truthful measured condition values, and an explicit pass/block result for every
terminal condition dimension. It does not prove speaker-model accuracy or
authorize terminal selection.

## Current State

Plan 0043 C2 froze and replayed successor corpus
`acoustic-corpus-4a2b13e7bdc201f694af2f43` at content SHA-256
`4a2b13e7bdc201f694af2f43d4ab845749eeeb3ea06c7a97a40164cab40b83fe`.
It has seven conversation/source-disjoint recordings, exact `3 / 2 / 2`
splits, 10 known subjects, 3 recurrent subjects, and 23 feasible same-person
pairs. Channel, device, noise, telephone-bandwidth, and usable-duration fields
remain explicitly unassessed, so the corpus is not terminal-selection eligible.

A metadata probe shows six mono sources and one stereo source, four distinct
source sample rates, and multiple encoding profiles. Physical capture-device
identity is not present in source metadata. That absence must remain explicit;
an encoding profile may be reported separately but cannot be mislabeled as a
measured physical device.

## Scope

- Add a private preview/apply/replay orchestrator bound to the exact successor
  corpus manifest SHA/content, P1/P2 module hashes, clean repository state,
  readiness assets, and a reviewed preview content hash.
- Process exactly all seven frozen records through P1 decode/quality and all
  five P2 preparation methods, using the corpus content hash as the stereo
  downmix and later-split access authority.
- Derive condition evidence only from measured source/P1/P2 values:
  source channel count/layout, source sample-rate telephone-bandwidth evidence,
  Silero usable-speech duration, speech/non-speech RMS-based noise evidence,
  and source metadata device reporting. Preserve encoding profile as a separate
  proxy field; do not call it physical-device evidence.
- Freeze exact per-record and aggregate condition denominators, observed-value
  counts, missingness, and terminal blockers. Require at least two observed
  values for every decision-relevant dimension before allowing a later
  terminal `select` authority.
- Keep gold bodies, names, transcript text, raw audio, embeddings, and biometric
  scores out of portable receipts. Runtime paths and P1/P2 artifacts remain
  private `0600` under `0700`.

## Non-Goals

- Do not create or score biometric trials, change the nine frozen thresholds,
  select a model/method, enable defaults, or reprocess history.
- Do not infer physical capture devices from codec, bitrate, container brand,
  filename, participant identity, or calendar context.
- Do not weaken the two-value condition rule when a source field is absent.
- Do not use model predictions or operator identities in condition measurement.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A contract and tests | primary | Plan 0043 C2 | deterministic preview/apply/replay and fail-closed tests pass |
| B independent audit | read-only reviewer | A | `PASS`, or one bounded repair and re-audit |
| C private execution | primary | B plus clean pushed commit | 7 P1 and 35 P2 attempts replay, or exact failure receipt |
| D condition decision | primary plus reviewer | C | every dimension records pass/block with exact observed values |

Critical path is serialized. The existing read-only reviewer audits metadata,
hashes, schemas, counts, and receipts but does not inspect gold bodies, names,
transcript text, raw audio, embeddings, or biometric values.

## Gates and stop conditions

- Preview/apply/replay must revalidate the live corpus manifest and every bound
  source hash. Apply stops before writes on repository, manifest, readiness,
  module, membership, or reviewed-hash drift.
- Exactly seven P1 runs and 35 P2 method attempts are required. A missing,
  blocked, failed, non-replayed, or non-private unit prevents a complete
  condition receipt.
- `device` passes only from an explicit source device field. Encoding-derived
  profiles are diagnostic proxies and cannot satisfy that gate.
- Missing physical-device evidence or fewer than two observed values in any
  required dimension makes terminal selection ineligible. The result is a
  truthful blocker, not a fabricated category or failed speaker model.
- Hugging Face credentials from `~/credentials/API-keys.env` may be loaded only
  if the already-authorized local pyannote adapter actually requires them;
  values must never be printed or persisted.

## Acceptance Criteria

- Preview is deterministic apart from no timestamp, names the exact seven
  opaque source units, predicts `7 x 5` attempts, and performs no writes or
  model execution.
- Apply requires the reviewed preview hash, runs exactly the frozen units,
  records P1/P2 replay hashes, and writes one immutable private condition
  manifest and receipt.
- Replay is read-only and verifies the complete manifest, receipt, P1/P2
  lineage, current source/corpus/module/readiness hashes, and modes.
- Condition aggregation reports exact per-dimension observed-value sets and
  counts, including explicit physical-device missingness; no proxy silently
  satisfies the device gate.
- Focused tests cover deterministic preview, exact denominator, drift,
  fail-before-write, condition classification, missing-device blocking,
  tamper, idempotence, replay, and private modes.
- Focused and full repository tests, compilation, `git diff --check`, and one
  independent read-only audit pass before runtime apply.

## Validation

- `python -m pytest -q tests/test_acoustic_successor_conditions.py`
- `python -m pytest -q`
- `python -m py_compile acoustic_successor_conditions.py`
- `git diff --check`
- Private runtime preview/apply/replay with exact hash and permission readback.
