# Scoped speaker comparison process bridge

State: CLOSED

## Current State

Plan 0085 supplies reviewed private enrollment and immutable profile heads. Speech Receipts #37 now needs a supported read-side comparison boundary. Existing verification adapters and cosine scoring provide the computation; this packet adds only speaker_match_bridge.py, focused tests and source documentation. The critical path owner is the parent Speech matching integrator; this worker owns the non-overlapping upstream process seam.

## Vision and maturity

Advances vision outcomes 3, 6 and 8 through bounded acoustic evidence with exact provenance and explicit uncertainty. Current cross-project comparison maturity is Level 0, target Level 1 built with synthetic hermetic tests. Scores remain uncalibrated; no numeric threshold, accepted identity, quality qualification or automatic contextualization is established.

## Scope and acceptance

One bounded normalized probe is compared only with up to 32 explicitly selected current reviewed profile heads from the same scoped enrollment store. Bind person revision, adapter/model and private profile SHA256. Check profile head, A3 active state, current authority revision and object hash before and after inference. Preserve source timestamps, sample hashes and immutable profile versions. Return uncalibrated cosine evidence with identity_assigned/enrollment_performed/qualified all false. No enrollment, new sample custody, identity assignment, Contacts mutation, provider call, campaign authority reuse or runtime installation is permitted.

## Stable seam and validation

The public ComparisonBridge.request and CLI JSON protocol prove scope, current-profile authority, hash binding, provenance, response privacy and no profile/head changes. Direct class tests inject the existing deterministic VerificationAdapter; the CLI registry has no fake mode. Red: missing module collection failure. Green: focused pytest, joined enrollment/custody regression, Python compile and targeted Ruff checks. No transcription/DOCX path changes or real-model tests are part of this bounded slice. Parent separately owns process cancellation, Speech durable intent, exact reconciliation and UI acceptance.

## Completion gate

Focused and joined tests pass, source docs and compact roadmap/runbook projections agree, no source claim implies installed/qualified inference. Preserve pre-existing dirty work. No commit, push, real media/model or shared service effects.

## Source acceptance

Twelve focused comparison tests pass in 4.28 seconds; joined comparison, enrollment, custody and acoustic verification validation passes 90 tests in 15.53 seconds. Cached offline Ruff formatting/correctness and Python compile checks pass. The production CLI smoke reports absent genuine assets without loading models and rejects a fake registry selection. Current and replaced profile heads, changed bytes, withdrawal, scope and model/hash/authority drift are tested; invalidation during embedding yields no scores. Exact red/green commands are `.venv/bin/python -m pytest tests/test_speaker_match_bridge.py -q`; the initial collection failure was followed by the implemented boundary. One test setup correction supplied the existing ledger's required superseded event; a genuine failing review-authority test drove the missing guard. Joined command adds `tests/test_speaker_enrollment_bridge.py tests/test_biometric_custody_ledger.py tests/test_acoustic_verification.py`.

This closes the delegated Level 1 source packet only. Speech durable-operation/client/HTTP/UI acceptance, live model qualification and calibration remain separate. Memory disposition: forbidden; this delegated packet does not authorize a separate cross-project memory write, and the parent orchestrator owns the joint closeout.
