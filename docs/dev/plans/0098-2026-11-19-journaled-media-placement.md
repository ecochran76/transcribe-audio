# 0098: Journaled media placement and verified migration

State: CLOSED
Status: complete
Owner: primary
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Journaled media placement and verified migration, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

None (can start immediately)

## Acceptance Criteria

- [x] Register and resolve stable media identities in the existing store without changing source provenance.
- [x] Copy, fully verify, publish and retire per-object placements with resumable durable states.
- [x] Changed source, destination conflicts, source erasure, pins, hardlinks and missing independent source proof prevent retirement.
- [x] Public API tests prove actual bytes/access and restart behavior.

## Current State

Implemented and integrated at 7043574b5f776d92eb2ceb58f89ceb5ae30b4fdd; all 38 focused storage checks passed. Native installation/migration remains ticket0103.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
