# Scoped speaker enrollment process bridge

State: CLOSED

## Current State

Speech Receipts #36 lacks a supported enrollment integration for newly curated samples. Existing campaign P3/P4 authority must not be borrowed. The A3 custody ledger and pinned lazy verification adapters now support a separate additive process boundary in speaker_enrollment_bridge.py. Upstream starts clean at 5202ca13df2f57423eeadc8c39f83504b3192bbd. This packet owns only a new root module, focused tests/documentation and compact roadmap/runbook projections.

## Scope

Provide a bounded JSON-stdin/stdout enrollment bridge using a dedicated explicit tenant/application/private custody directory, reviewed person-binding and enrollment-authority configuration, exact model pins, durable operation reconciliation, cancellation, private retained samples and immutable profile versions. Reuse BiometricCustodyLedger and genuine VerificationAdapter implementations. Bind source/sample/review/recipe provenance. Preserve the previous selected profile on a failed or interrupted rebuild. Root integrates the separate Speech HTTP/UI client.

## Vision and maturity

Advances outcomes 3, 6, 7 and 8 by allowing reviewed acoustic evidence to become reproducible private knowledge without automatic identity assignment. Current integration maturity is Level 0 (absent), target Level 1 (built and hermetically tested). Evidence is exact new-sample enrollment with deterministic adapters, scoped access/authority failures, failed rebuild preservation, interruption and cancellation. No claim about general identity quality, calibration or automatic contextualization follows.

## Non-Goals

No private recording/campaign reads, real enrollment, model acquisition or loading, installed service changes, live store migration, provider calls, publication, commit or push. Existing frozen campaign plans and authority remain unchanged. Local source tests do not qualify a recognition threshold or establish identity truth.

## Acceptance Criteria

Configured authorities bind exact person revision/review/enrollment IDs independently from optional Contacts association. Only bounded hash-verified 16 kHz mono PCM WAV enters a pinned adapter. Durable intent precedes effects; unknown outcomes never replay. Private bytes remain outside portable responses. Exact sample/model/recipe/profile versions remain immutable. Cancellation waits for adapter return or confirmed parent-owned process termination; readiness is separate from qualification. A checked rebuild selects a new head without deleting the predecessor.

## Validation

Focused hermetic pytest and isolated process CLI smoke; no audio transcription or DOCX paths are changed, so upstream transcription manual-smoke requirement is not triggered. Reuse custody ledger regression coverage. Preserve first failure and distinguish source-only acceptance from installation or real inference.

## Source acceptance

The additive bridge, protocol documentation and eight focused test cases are complete. Joined validation passed 77 tests in 11.60 seconds using the existing `.venv/bin/python`: new process contracts plus existing biometric custody and acoustic verification suites. Metadata-only production CLI readiness uses an absent temporary model snapshot and reports unavailable without loading/downloading a model. Compile, targeted Ruff correctness checks, formatting and diff checks pass. Root independently verified the preceding 76-case checkpoint; the final extra parameter case proves the same promotion reconciliation from an already-unknown journal state.

First failures were test construction issues: the existing FakeVerificationAdapter requires an explicit candidate_id, and a broad centroid substring assertion incorrectly matched the public recipe name. Both were corrected at the test seam; no protection was removed. The recovery path now verifies exact custody activation event, candidate version and object hash without rerunning embeddings. Parent-death protection and concurrent cancellation retain unknown or waiting states until quiescence is proved.

This closes the source bridge prerequisite at Level 1. Speech HTTP/UI integration, actual configured biometric model readiness, reviewed enrollment mappings, real enrollment, calibrated quality and installation remain separate. No existing campaign plan/runtime, private corpus or installed service changed. All source remains uncommitted under current authority. Memory disposition: forbidden; this delegated source packet does not grant a separate cross-project memory write, and the primary orchestrator owns the joint closeout.
