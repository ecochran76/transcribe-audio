# Plan 0027 | Speaker Output Reference Repair

State: OPEN

Lane: P09

## Scope

Add one bounded, ledger-backed corrective turn when Clue Discovery or Identity
Evaluation returns an otherwise structured result whose evidence references
are outside the host-prepared allowlist. The host must preserve the rejected
result, identify only the invalid reference fields, present the exact allowed
IDs for those fields, request a corrected result, and validate the correction
through the same existing schema and allowlist gates.

Use the frozen Plan 0026 accumulated regression batch and first chronological
holdout as the evaluation surface. Compare the repair-enabled algorithm with
the immutable baseline and holdout receipts; do not overwrite either prior
run.

## Non-Goals

- No fuzzy or positional remapping of invented references.
- No weakening of host validation or acceptance of unprepared evidence.
- No additional provenance retrieval during a reference-only repair.
- No speaker assignment, external contact mutation, CRM write, deposition, or
  memory harvest.
- No prompt tuning for identity reasoning beyond the invalid-reference repair.
- No advance to another chronological campaign batch in this plan.

## Current State

Plan 0026's first untouched holdout captured all ten predictions before gold
review. Eight results were rejected by host validation: four during Clue
Discovery and four during Identity Evaluation. The failures are concentrated
in invented or unprepared transcript-clue, provenance-source, or
utterance-evidence references. The two results that passed validation still
produced three High/Very High wrong speaker proposals, so a repair must report
reasoning metrics separately and cannot be described as an identity-quality
fix.

An earlier one-shot prompt clarification did not reduce Clue Discovery
validation failures and increased High/Very High wrong identity proposals; it
was explicitly rejected and reverted. This successor therefore tests a
host-mediated corrective turn rather than another broad prompt rewrite.

The bounded repair implementation is now complete in the working tree. Each
phase first enters the unchanged validator. A reference-only failure may
prepare one separate App Intelligence run whose packet preserves the rejected
JSON, identifies each invalid reference field, and lists the exact prepared
IDs allowed for that field. The campaign runner retries the failed phase once
with the corrected JSON and records both original and repair run IDs. A second
invalid result remains an explicit model-output rejection.

Synthetic workflow, API, ledger, and runner coverage passes for invented
transcript-clue IDs, provenance-source IDs, utterance IDs, repeated invalid
repair output, valid first-pass bypass, and immutable original input
artifacts. Live immutable regression and holdout comparisons remain pending.

## Acceptance Criteria

- The original rejected model result remains immutable and ledger-addressable.
- A corrective packet names the invalid fields and exact allowed IDs without
  adding transcript, gold, or provenance content that was absent from the
  original prepared packet.
- At most one corrective model turn is attempted per failed phase.
- Corrected output passes the unchanged schema and prepared-reference
  validation or remains an explicit model-output rejection.
- Synthetic tests cover invented transcript-clue IDs, provenance-source IDs,
  utterance-evidence IDs, repeated invalid repair output, and valid first-pass
  output that must not trigger repair.
- The repair-enabled run is compared against the complete accumulated gold
  regression set and the frozen holdout as a new immutable run.
- Validation-yield improvement is reported separately from calendar, speaker,
  diarization, and High/Very High wrong-proposal metrics.
- The change is rejected if it weakens validation, performs an external write,
  or introduces an unexplained metric regression.

## Validation

- TDD for corrective-packet construction, one-retry enforcement, immutable
  rejected-result preservation, and unchanged allowlist validation.
- Preserved-evidence replay where the frozen evidence artifacts support it;
  otherwise a separately labeled fresh-retrieval comparison.
- Focused campaign, workflow, API, and App Intelligence ledger tests.
- `git diff --check`, active planning audit, clean commit/push verification,
  and live local-service readback before promotion.

## Stop Condition

Close this plan after one repair-enabled accumulated-regression comparison is
accepted or rejected with a durable receipt. Resume Plan 0026 C7 only if the
result justifies spending another chronological review batch; otherwise open a
new hypothesis-specific successor or retain the campaign at its recorded
cursor.
