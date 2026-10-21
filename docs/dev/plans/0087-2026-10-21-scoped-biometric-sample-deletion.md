# Scoped biometric sample deletion

State: CLOSED (source qualification only)

## Current State

Enrollment/comparison bridges are committed and pushed at 46beba552d1de5bddd28d3e67eaf0d00afb44b44. Speech Receipts #38 requires exact selected-sample deletion across all immutable profile versions, private copied audio/profile bytes and interrupted operation custody. Existing A3 effects can delete one sample but sequential sample effects sharing a profile cannot safely remove the same private object twice. This bounded packet adds an aggregate sample-set resolver over the existing effect protocol and supported process actions, plus Speech-owned participant hooks outside this upstream checkout.

## Scope

Scoped deletion preview/apply/status, private configured deletion authority, exact dependency fingerprint, durable operation and reference fences, union of affected profiles and copied private objects, and tombstone/physical cleanup reconciliation. Preserve prior effects and unknown outcomes. Keep source-recording deletion separate. Parent owns Speech coordinator, UI and root integration.

## Non-Goals

No live/private recording or biometric deletion, model inference, campaign edits, shared service changes, force pushes or implicit identity/enrollment authority. Source tests use private temporary synthetic stores only. No blanket person/recording delete is substituted for selected samples.

## Vision and maturity

Advances outcomes 3, 6, 7 and 8 through replayable correction and trustworthy acoustic custody. Current deletion integration maturity is Level 1 with scoped hermetic process/custody evidence. This does not establish real recognition quality, live deletion acceptance or automatic identity.

## Acceptance Criteria

Preview names selected samples, all affected immutable profiles and relevant operations. Exact confirmed intent and fences precede effects. Aggregate deletion uses existing A3 sample/profile events and tombstones without deleting originals. Pending/deleted references cannot be enrolled, compared or reactivated. Unknown operations reconcile proven receipts and exact private-object/quarantine state without blind A3 replay. Actor, fingerprint, missing authority and unexpected filesystem changes fail closed.

## Validation

Focused aggregate/shared-profile, scoped replay, stale preview, cleanup-failure/lost-ack and interrupted-unknown tests, plus existing enrollment/comparison/custody/verification regression. Record source hashes and exact commands. No transcription or DOCX path changes, so model/provider/manual transcription smoke is outside this packet.


## Result and evidence

Implemented supported deletion actions, aggregate native A3 resolution, pending/deleted enrollment/comparison fences, and exact cleanup reconciliation. Five focused tests cover shared profile versions, sequential overlapping dependencies, scoped authority/stale intent, lost native acknowledgement, unconfirmed native effect without replay, and interrupted pre-registration orphan-only cleanup. Joined enrollment/comparison/custody/acoustic/deletion suite: **95 passed in 20.74 seconds** using `.venv/bin/python -m pytest -q tests/test_speaker_sample_deletion.py tests/test_speaker_enrollment_bridge.py tests/test_speaker_match_bridge.py tests/test_biometric_custody_ledger.py tests/test_acoustic_verification.py`. Offline Ruff formatting and targeted E9/F63/F7/F82 checks passed.

The first focused run had three failures and one pass: the proposed new `sample_set` native target type violated the existing SQLite tombstone CHECK constraint. A synthetic direct trace established that schema incompatibility. The correction retains the existing `sample` target type with an explicit sample-set envelope/resolver; no schema migration or generic single-sample contract was changed. Subsequent focused five tests and joined 95 tests passed. Parent bounded review additionally identified orphan-only interrupted cleanup; the durable-intent/hash-guard regression now proves exact status recovery without an invented native receipt.

Original recordings and metadata history remain; physical private copies/profile vectors are removed only under confirmed scoped intent. No model download, embedding, private corpus/campaign action, shared runtime change or live deletion acceptance occurred. Speech integration, installation and deployed acceptance remain separate parent-owned gates. Protocol: `docs/speaker-sample-deletion-process.md`.
