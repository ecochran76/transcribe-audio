# Plan 0045 | Plan 0037 P4E2 capture-device provenance refinement

State: CLOSED

Lane: P10

Plan Version: 2

Parent: Plan 0043 P4E2-D refinement

Owner: primary agent

Expected Write Surface: `acoustic_device_provenance.py`, focused tests, this
plan, Plans 0037 and 0043, `ROADMAP.md`, and `RUNBOOK.md`; private campaign,
cursor, attestation, correction, replay, and composite-condition artifacts only
under the user-scoped Plan 0037 runtime root.

## Vision alignment

This packet advances calibrated and safely abstaining speaker evidence while
preserving missing-evidence truth. Current and target maturity remain
`2 - Shadow`. Progress is measured by an immutable exact-seven capture-device
provenance authority that either supplies genuine operator-known device facts
for every recording or preserves the terminal blocker. It does not prove model
accuracy, authorize evaluation reveal, or advance the acoustic path to
operational use.

The packet improves automatic contextualization by resolving a specific
provenance gap that currently prevents any acoustic model comparison from
becoming valid terminal evidence. It keeps the original condition campaign and
all source evidence immutable.

## Current State

Plan 0044 completed exact 7 P1 and 35 P2 method successes and replayed its full
private lineage. Channel, noise, telephone-bandwidth, and usable-duration each
passed with two observed values and zero missing recordings. Physical capture
device has zero observed values and seven missing recordings and is the sole
terminal blocker.

The initial read-only inventory found no eligible device field because it did
not query the manufacturer-specific metadata namespace and the recorded
original source paths appeared absent. A later operator-authorized indexed
search recovered all seven exact source byte streams. Plan 0047 then found an
allowlisted `Samsung:SamsungModel` hardware-model tag in five exact sources;
cases 2 and 4 contain no such tag. Generic handler, encoding, container,
filename, source folder, channel, and sample-rate profiles remain ineligible.
The five source facts are held in a separate authority and are not mislabeled
or inserted into this direct-operator attestation history.

After the campaign froze and its first case opened, reviewed Plan 0046
descendant commits advanced repository `HEAD` without changing the frozen
device-provenance module. The original replay path rebuilt campaign identity
from current `HEAD`, so the valid frozen campaign became stale. This is a
continuity defect in replay, not source or operator evidence drift.

## Scope

- Add a private preview/apply/replay device-provenance campaign bound to the
  exact successor corpus manifest, Plan 0044 condition manifest and receipt,
  all seven source hashes, clean repository commit, and module hash.
- Freeze exactly seven opaque recording cases in corpus order. Apply requires
  the independently reviewed preview content hash and writes no device claim.
- Preserve a frozen campaign across later clean descendant commits by proving
  the frozen commit remains an ancestor and the exact historical module blob
  still hashes to its stored authority. Reconstruct the frozen body with its
  original repository binding; do not rewrite or reissue it.
- Reapplying after a descendant commit must find and replay the one existing
  campaign with the same exact corpus and condition authorities. Multiple or
  malformed matches fail closed instead of creating a duplicate campaign.
- Open exactly one case at a time through an immutable hash-chained cursor.
  A private operator packet may show recording date/title and source-location
  context needed for recall, but portable receipts contain only opaque IDs,
  hashes, counts, and policy flags.
- Accept a physical-device value only as `direct_operator_knowledge`. Unknown,
  uncertain, inferred, filename-derived, folder-derived, codec-derived, or
  calendar-derived answers remain unavailable and cannot satisfy coverage.
- Store attestations and corrections append-only. Each record binds the exact
  recording/source/corpus/condition/cursor authority, operator identifier,
  casefolded/whitespace-normalized private device label, opaque device ID,
  basis, timestamp, and
  previous record hash. A correction supersedes but never rewrites history.
- Build a read-only composite condition preview and immutable private authority
  only after all seven latest attestations are direct and replayable. It may
  replace only the `device` missing value; every measured Plan 0044 condition
  and blocker rule remains unchanged.
- Require seven nonmissing device attestations and at least two distinct opaque
  physical-device IDs before `terminal_selection_eligible` can become true.

## Non-Goals

- Do not infer capture device from handler labels, codec, bitrate, sample rate,
  channels, container, filename, path, event, participant, transcript, or
  calendar context.
- Do not rerun P1/P2, modify the Plan 0044 manifest, read biometric scores,
  construct generation-2 trials, reveal evaluation, or select a model/method.
- Do not treat an authorization statement as a factual device attestation.
  Blanket execution authorization removes command rituals but does not create
  missing evidence.
- Do not place raw audio, transcript text, names, source paths, private device
  labels, credentials, embeddings, or biometric values in portable receipts.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A inventory and contract | primary | Plan 0044 terminal STOP | exact absence evidence and bounded plan exist |
| B implementation and tests | primary | A | preview/apply/cursor/attest/correct/replay tests pass |
| C independent audit | existing read-only reviewer | B | `PASS`, or one bounded repair and re-audit |
| D campaign freeze | primary | C plus clean pushed commit | exact-seven private campaign replays |
| E attestations | operator plus primary | D | seven one-at-a-time direct facts or truthful missing receipt |
| F composite authority | primary plus reviewer | E | device coverage passes or terminal STOP remains |

The critical path is serialized. Active-agent concurrency is at most two: the
primary owns all writes and the existing reviewer performs bounded read-only
audits. Review retries are limited to one repair-and-re-audit cycle per unit.

## Gates and stop conditions

- Preview, apply, open, attest, correct, compose, and replay revalidate every
  bound manifest, receipt, module, repository, source, cursor, and prior-record
  hash before writes.
- Descendant replay requires a clean current checkout, exact frozen repository
  keys, an ancestor relationship from frozen commit to current `HEAD`, and the
  historical device-provenance module hash read from the frozen Git tree.
- A campaign with an outstanding case may only reopen that case idempotently.
  It cannot skip, reorder, or concurrently open cases.
- An attestation must be an explicit operator fact about the physical capture
  device. `unknown` closes no gate; uncertainty or indirect recollection must be
  recorded unavailable.
- Device label normalization is limited to whitespace/case normalization for
  opaque equality. It cannot merge different labels through heuristics.
- Fewer than seven direct attestations, any missing source binding, fewer than
  two distinct opaque device IDs, replay drift, or residual audit finding keeps
  the Plan 0044 terminal STOP in force.

## Acceptance Criteria

- Deterministic no-write preview binds exact corpus/condition/source/repository
  authorities and predicts seven ordered cases.
- Reviewed-hash apply creates one immutable private campaign without opening a
  case or asserting device evidence.
- Hash-chained cursor enforces one-at-a-time order and idempotent reopen.
- Append-only attestations and corrections bind the current case and preserve
  superseded history; forged, inferred, stale, reordered, or duplicate records
  fail closed.
- Full replay recomputes campaign, cursor, latest-record reduction, opaque
  device IDs, exact denominators, privacy flags, and composite coverage.
- A clean descendant commit preserves the original campaign ID/content and
  reapply remains singleton-idempotent; dirty checkout, non-ancestor history,
  historical module drift, duplicate campaign, or predecessor drift fails.
- Portable receipts contain no raw/private content and all runtime files are
  `0600` under `0700` directories.
- Focused and full tests, compilation, `git diff --check`, independent audit,
  clean pushed commit, private campaign apply, and read-only replay pass before
  the first operator case opens.

## Validation

- `.venv/bin/python -m pytest -q tests/test_acoustic_device_provenance.py`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python -m py_compile acoustic_device_provenance.py`
- `git diff --check`
- Private preview/apply/replay and exact permission readback.

## Descendant replay continuity repair

State: CLOSED; Unit E attestations remained open at case 1 at this historical
checkpoint.

Unit E's original seven-direct-attestation route is `SUPERSEDED`, not completed.
Plan 0047 closed the device gate through its separately governed two-direct plus
five manufacturer-metadata authority and composite. This plan is closed and is
not a Generation-3 blocker.

- Frozen campaign `device-provenance-07f1509cf8657c793777e386` remains the
  sole exact-seven authority. No campaign, cursor, or attestation artifact is
  rewritten by this repair.
- Replay now validates the frozen commit/module from Git history under a clean
  descendant checkout and reconstructs the original body with its frozen
  repository authority. Apply reuses the single campaign bound to the same
  corpus and condition instead of deriving a duplicate from the later `HEAD`.
- Tests cover clean descendant replay and reapply plus dirty checkout,
  historical module drift, non-ancestor rejection, and two valid campaigns for
  one predecessor pair. Independent re-audit returned `PASS` on scoped
  code/test diff SHA-256
  `3f6cbebdd0f945453775b4d065801e88784f11df213f3abe1f0664da900d96c1`.
  Twelve focused tests, 40 joined device/condition/generation-2 tests, and all
  613 repository tests passed; compilation and `git diff --check` passed.
  Repair commit `bb975ebe5e46f880cefadf4267d03e2b5d7ede83` is pushed with
  upstream ahead/behind `0 / 0`.
- Clean production replay returned full-body equality with one open case and
  zero initial records. Descendant reapply returned the original campaign, the
  case reopened idempotently, every private path retained exact `0700`/`0600`
  mode, and every campaign file retained identical SHA-256 and mtime state.
