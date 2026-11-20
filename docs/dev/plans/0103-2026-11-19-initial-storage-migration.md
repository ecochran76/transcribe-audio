# 0103: Install and account for direct initial migration

State: OPEN
Status: in-progress
Owner: /root/storage_native_recovery (native); storage_finish (integration)
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Install and account for direct initial migration, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

- 0102: Operator CLI and bounded recurring storage maintenance

## Acceptance Criteria

- [ ] Validate one integrated source generation and install the scoped maintenance configuration/entrypoint.
- [ ] Inventory approved Transcribe roots and verify native HDD identity, independent replicas and active-consumer compatibility.
- [ ] Execute eligible direct journaled migrations without canary/hold, and account for every selected object as migrated or held with exact cause.
- [ ] Verify actual application access, timers, private receipts and exclusive block recovery; never lower policy to force migration or touch Speech.

## Current State

Integrated source fcb0338f198bddd22f47200cc822a07e2e437f1e is installed.
The focused local selection passed 105 tests. Two bounded native passes retired
347 of 405 qualified WAV placements, recovering 34,194,436,096 exclusive NVMe
bytes. The remaining pass, native API access, backup/restore and hourly timer
acceptance remain in progress. Canonical sources without independent durable
replica proof remain held; no new recovery volume was provisioned. Private native
receipts are outside Git. This is not final native acceptance.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
