# 0099: Classify retention eligibility and reserve bounded scratch

State: OPEN
Status: ready-for-agent
Owner: unassigned
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Classify retention eligibility and reserve bounded scratch, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

None (can start immediately)

## Acceptance Criteria

- [ ] Expose a public versioned policy decision/reservation interface for all accepted storage classes.
- [ ] Enforce 48h completed grace, 7d inactive unresolved cold movement, pins and quarantine without source deletion or retries.
- [ ] Apply 20GiB transient and 100GiB background reserve plus 16GiB/30minute pass limits.
- [ ] Trusted external time observations qualify age; missing/unreliable time holds age actions.

## Current State

Pending implementation. Source acceptance, integration validation and native effects remain distinct.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
