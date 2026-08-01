# Plan 0046 | Plan 0037 P4E2 generation-2 terminal authority

State: OPEN — Unit A closed; Unit B next

Lane: P10

Plan Version: 1

Parent: Plan 0043 P4E2-D

Owner: primary agent

Expected Write Surface: `acoustic_verification.py`, focused tests, this plan,
Plans 0037 and 0043, `ROADMAP.md`, and `RUNBOOK.md`; private pre-reveal
authority artifacts only under the user-scoped Plan 0037 runtime root after
all predecessor gates pass.

## Vision alignment

This packet advances calibrated and safely abstaining speaker evidence by
making the already-frozen calibration chain replayable from a newer, explicit
evaluation-split seam and by constructing the successor pre-reveal authority.
Current and target maturity remain `2 - Shadow`. Progress is measured by exact
replay of the archived calibration application, independently reviewed
generation-2 authority construction, and preservation of the sealed evaluation
split. This packet does not establish model accuracy, select a default, or
advance acoustic evidence to operational use.

## Current state

The generation-1 calibration authority and its 396 finite trials, nine frozen
thresholds, profiles, and descendant eligibility remain intact. Current replay
fails at one exact field: the archived calibration authority binds P2 module
SHA-256 `467627bc3452863c996b81e4aada0b5d62d0b7350064c5adc6132666b8410bdc`,
while the reviewed evaluation-split seam has module SHA-256
`700e10d802a6443eab9d2bb9c6b9a7519cff26021ffec23acbdb767f12bcd595`.
Every other reconstructed authority field matches. The archived module was not
mutated; the successor module added the explicit `evaluation` later-split mode.

Plan 0045 is concurrently collecting seven direct physical-device
attestations. Its composite condition authority is a hard predecessor of any
production generation-2 authority apply, reveal, audio access, model execution,
or scoring.

## Scope

- Add a typed, exact historical-P2 replay contract binding the archived and
  current module hashes and the sole permitted evolution reason.
- Permit that contract only on read-only replay of already-persisted
  calibration stages. Missing reveal, preparation, window, score, or threshold
  artifacts fail closed instead of being created.
- Preserve exact full-body validation for every non-P2 calibration authority
  field, every descendant authority and profile, the persisted 396-trial score
  matrix, and all nine recomputed thresholds.
- Build deterministic no-write generation-2 authority preview/replay logic
  that binds the successor corpus, split/seal, prediction-excluded gold,
  measured composite conditions, exact historical calibration replay, current
  P1/P2 authorities, profiles, model assets, frozen thresholds and margins,
  candidate matrix, trial construction, terminal metrics, and decision policy.
- Keep production apply unavailable until Plan 0045 produces a passing,
  independently reviewed composite condition authority.

## Non-goals

- Do not alter, overwrite, or reissue the archived calibration authority.
- Do not broadly ignore module hashes, accept arbitrary compatibility ranges,
  monkeypatch hashing, or permit the compatibility contract in any calibration
  apply/write path.
- Do not reveal evaluation, read successor audio or gold bodies, run models,
  score trials, calculate terminal metrics, or select a model/method.
- Do not infer missing capture-device evidence or treat execution authorization
  as a factual device attestation.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A replay seam | primary | archived calibration chain plus reviewed P2 seam | exact compatibility replay passes and adversarial drift fails |
| B authority implementation | primary | A plus frozen successor authorities | deterministic no-write generation-2 preview/replay tests pass |
| C independent audit | existing read-only reviewer | A and B | `PASS`, or one bounded repair and re-audit |
| D production freeze | primary | C plus passing Plan 0045 composite and clean pushed commit | private authority applies and replays without reveal |

Units A and B may proceed without operator device facts because they use only
fixtures or already-persisted calibration evidence. Unit D is serialized behind
Plan 0045. Active-agent concurrency is at most two; the primary owns writes and
the existing reviewer performs read-only audit.

## Gates and stop conditions

- Compatibility is valid only when the stored authority contains the exact
  archived hash, the live P2 module has the exact reviewed successor hash, the
  policy ID and reason are exact, and reconstructing with the archived hash
  yields full authority equality.
- Replay compatibility cannot be passed to authority build/apply functions.
  When active, every downstream calibration artifact must already exist as a
  private file and retain its exact content identity.
- Any additional authority drift, current-module drift, missing stage,
  descendant/profile ineligibility, score/trial drift, threshold drift, or
  nonprivate path stops replay.
- Production generation-2 apply requires seven nonmissing direct device
  attestations, at least two opaque physical-device IDs, a replayed composite
  condition authority, independent audit `PASS`, clean pushed commit, and an
  evaluation split that remains sealed.
- No preview, test, audit, or blanket authorization satisfies a content or
  integrity gate by itself.

## Acceptance and validation

- Default calibration replay behavior remains strict and continues rejecting
  the historical/current module mismatch without the exact contract.
- Exact compatibility replay validates the archived authority, score matrix,
  and recomputed thresholds without creating or changing any runtime artifact.
- Tests reject wrong archived/current hashes, unknown policy/reason, extra
  authority drift, missing persisted stages, score/threshold tamper, and any
  attempt to use compatibility in a write path.
- Generation-2 authority preview is deterministic, private-data free,
  evaluation-sealed, and fully bound to all required predecessor authorities.
- Focused and full tests, compilation, `git diff --check`, independent audit,
  clean pushed commit, and private replay pass before production authority
  apply. Reveal and terminal execution remain `not_run`.

## Unit A close checkpoint

State: CLOSED; Units B through D remain `not_run`.

- The exact replay-only contract binds calibration authority
  `0fe6009bef2adfc9c48d87eea7d4ac15c00734ec45376ba3dbba45952e42fae5`,
  archived P2 module
  `467627bc3452863c996b81e4aada0b5d62d0b7350064c5adc6132666b8410bdc`,
  and current reviewed evaluation seam
  `700e10d802a6443eab9d2bb9c6b9a7519cff26021ffec23acbdb767f12bcd595`.
  Default replay remains strict; any different authority, module, policy,
  reason, extra contract field, or non-P2 authority drift fails closed.
- Compatibility is absent from every build, reveal, preparation, selection,
  and apply signature. Historical score replay preflights all persisted stages
  and uses dedicated read-only reveal/preparation/window structural validation.
- Production replay recomputed all nine thresholds from the persisted
  396-trial score matrix. A failing immutable-writer spy was not called, and
  authority, reveal, preparation, window, score, and application artifacts
  retained identical SHA-256 and mtime state.
- Independent re-audit returned `PASS` on scoped code/test diff SHA-256
  `9dae572b4683ea54176dd3f9fc750b8a84ad4f1262f20d84da4eb964671ea6b7`
  after one bounded repair cycle. Twenty-two historical tests, 73 joined
  verification tests, and all 585 repository tests passed; compilation and
  `git diff --check` passed.
- Unit B may implement the no-write generation-2 authority preview against
  fixture and replayed predecessor summaries. Production apply remains blocked
  on Plan 0045's seven direct attestations and passing composite authority.
