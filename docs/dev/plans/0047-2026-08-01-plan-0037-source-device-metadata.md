# Plan 0047 | Plan 0037 source-embedded device metadata supplement

State: OPEN

Lane: P10

Plan Version: 2

Parent: Plan 0045 P4E2 capture-device provenance refinement

Owner: primary agent

Expected Write Surface: `acoustic_source_device_metadata.py`,
`acoustic_generation2_authority.py`, focused tests, this plan, Plan 0045,
Plan 0046, `ROADMAP.md`, and `RUNBOOK.md`; immutable
private source-metadata manifests and receipts only beneath the existing
user-scoped Plan 0037 device-provenance campaign.

## Vision alignment

This packet advances calibrated and safely abstaining speaker evidence while
preserving missing-evidence truth. Current and target maturity remain
`2 - Shadow`. Progress is measured by an independently replayable authority
that admits only manufacturer-owned physical-device model metadata from exact
frozen source bytes. It does not establish acoustic accuracy, authorize an
evaluation reveal, or advance the product to operational use.

## Current state and target

Operator-authorized indexed search recovered all seven frozen source
recordings with exact source SHA-256 and byte-count matches, so the conditional
case-replacement path did not activate. Five sources expose the exact
`Samsung:SamsungModel` value `SM-S908U`; two expose no allowlisted hardware
model. Plan 0045 currently admits only direct operator knowledge, so these
source facts must not be inserted into its attestation history under a false
basis.

The target is a separate immutable supplement that records five observed
source-metadata facts and two explicit unavailable results, then lets the
composite authority merge those facts without modifying Plan 0045 history.

## Scope

- Bind all seven operator-selected source paths to the frozen campaign case,
  source SHA-256, and byte count before extracting metadata.
- Admit only the exact allowlisted manufacturer tag
  `Samsung:SamsungModel`; generic handler, Android, codec, container,
  filename, folder, or capture-application values remain ineligible.
- Record the extraction command/tool version, source binding, raw private model
  label, normalized label, opaque physical-device ID, and observed/unavailable
  status in a private immutable manifest.
- Keep portable receipts limited to opaque IDs, hashes, counts, policy flags,
  and absence reasons; exclude source paths and raw model labels.
- Replay every source byte and allowlisted tag before consuming the supplement.
- Extend composite preview/apply/replay so direct operator facts take
  precedence and source metadata fills only otherwise-missing recordings.
- Accept a sparse direct-operator supplement for exactly cases 2 and 4 because
  the operator confirmed both used the same webcam microphone. Bind a reviewed
  opaque device ID at apply time; keep the raw label and attestor private.
  Do not advance the original sequential Plan 0045 ledger or manufacture facts
  for cases 1, 3, 5, 6, and 7.
- Allow generation-2 pre-reveal validation to consume the augmented composite
  only when it proves exactly seven authoritative rows: two direct-operator
  observations, five manufacturer-metadata observations, zero missing rows,
  and at least two distinct opaque devices.

## Non-goals

- Do not infer device identity from non-allowlisted metadata or translate a
  manufacturer model code into a marketing name.
- Do not replace a case: every frozen source was found exactly.
- Do not treat metadata absence in cases 2 and 4 as evidence of a device.
- Do not rewrite the Plan 0045 campaign, cursor, records, or receipts.
- Do not reveal evaluation labels, run models, or perform biometrics.

## Gates and acceptance criteria

- Preview is deterministic and no-write; apply requires the reviewed content
  hash and a complete exact-seven source mapping.
- Any missing path, source hash/size mismatch, extractor failure, unallowlisted
  field, duplicate recording, or replay drift fails closed.
- Exactly five observed and two unavailable results are expected for the
  recovered production sources; evidence remains insufficient for terminal
  selection until all seven recordings and at least two distinct devices are
  authoritative.
- Focused/full tests, compilation, `git diff --check`, independent read-only
  audit, clean pushed commit, production preview/apply/replay, and exact private
  permissions must pass before composite consumption.

## Source authority checkpoint

State: CLOSED; composite-consumption unit remains OPEN.

- Operator-authorized indexed search recovered all seven exact frozen source
  byte streams; no case replacement was needed.
- The applied authority is
  `source-device-metadata-e9c6839faeaa1bdfd6bfe842`, content SHA-256
  `e9c6839faeaa1bdfd6bfe8420c0cff13c42f8d1743b4f3ce4539e1c75afa98a6`.
  It contains five observed results, two explicit unavailable results, and one
  distinct opaque device ID.
- Independent review found and closed manifest/body detachment, duplicate-case,
  portable-path disclosure, result-distribution, and extraction-time source
  swap risks. Targeted final verification returned `PASS`.
- Nine focused and all 622 repository tests passed; compilation and
  `git diff --check` passed. Implementation commit
  `90c62e38f59eb2d970640593d5678f58880115b4` is pushed.
- Production apply and full-body replay passed with exact `0700` directory and
  `0600` manifest/receipt modes. This evidence does not clear the terminal
  gate: cases 2 and 4 remain unavailable and the five observed rows identify
  only one distinct device.

## Operator and composite amendment

State: IMPLEMENTED; independent review and production apply remain OPEN.

- The operator confirmed that cases 2 and 4 were captured through the same
  webcam microphone. The raw device label is not committed to source or
  portable receipts.
- A sparse exact-two operator authority binds those facts to the frozen cases,
  source hashes, campaign manifest/state, attestor, reviewed opaque device ID,
  and append-only private manifest.
- The augmented composite accepts operator facts only where the source
  manufacturer tag was absent. Its expected evidence partition is `2 direct +
  5 source metadata = 7`, with two distinct opaque device IDs and zero missing
  recordings. All non-device Plan 0044 condition fields remain byte-derived
  predecessor values and may not change.
- Generation-2 binding recognizes the new manifest/replay schemas and requires
  the exact `2 / 5 / 7` partition before treating the composite as passing.
