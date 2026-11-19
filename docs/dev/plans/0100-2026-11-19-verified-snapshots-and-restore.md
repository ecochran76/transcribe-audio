# 0100: Verified compressed snapshots, rotation and isolated restore

State: OPEN
Status: ready-for-agent
Owner: unassigned
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Verified compressed snapshots, rotation and isolated restore, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

None (can start immediately)

## Acceptance Criteria

- [ ] Back up real SQLite consistently, verify integrity and compressed content, and restore to an isolated destination.
- [ ] Implement 24 hourly, seven daily and four weekly recovery-point selection, without confusing frozen evidence snapshots.
- [ ] Expose a monthly restore-drill public interface with private receipts and tombstone/recovery awareness.
- [ ] Wrong mount, corruption and untrusted rotation time preserve the live store and last good restore points.

## Current State

Pending implementation. Source acceptance, integration validation and native effects remain distinct.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
