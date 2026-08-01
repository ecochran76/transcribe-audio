# Plan 0047 | Plan 0037 source-embedded device metadata supplement

State: OPEN

Lane: P10

Plan Version: 1

Parent: Plan 0045 P4E2 capture-device provenance refinement

Owner: primary agent

Expected Write Surface: `acoustic_source_device_metadata.py`, focused tests,
this plan, Plan 0045, Plan 0046, `ROADMAP.md`, and `RUNBOOK.md`; immutable
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

