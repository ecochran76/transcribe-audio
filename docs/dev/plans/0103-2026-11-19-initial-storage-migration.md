# 0103: Install and account for direct initial migration

State: CLOSED
Status: complete
Owner: /root/storage_native_recovery (native); storage_finish (integration)
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Install and account for direct initial migration, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

- 0102: Operator CLI and bounded recurring storage maintenance

## Acceptance Criteria

- [x] Validate one integrated source generation and install the scoped maintenance configuration/entrypoint.
- [x] Inventory approved Transcribe roots and verify native HDD identity, independent replicas and active-consumer compatibility.
- [x] Execute eligible direct journaled migrations without canary/hold, and account for every selected object as migrated or held with exact cause.
- [x] Verify actual application access, timers, private receipts and exclusive block recovery; never lower policy to force migration or touch Speech.

## Current State

Integrated source fcb0338f198bddd22f47200cc822a07e2e437f1e is installed.
The focused local selection passed 105 tests. Three bounded native passes retired
all 405 qualified WAV placements, recovering 38,512,484,352 exclusive NVMe bytes.
Actual cold-media API range access, exact WAV recovery and all 896 frozen identity
checks passed. Strict backup acceptance initially detected 96 pre-existing derived
chunks without parent documents. Their full rows/schema were preserved privately;
only those orphans were removed transactionally. Document and linked-chunk payload
hashes were unchanged, and foreign-key/SQLite integrity checks passed. A verified
compressed snapshot and isolated monthly restore passed (864 catalog references,
832 restored canonical paths, two of two representative media hashes). The hourly
refresh-plus-maintenance timer is enabled and active; its first calendar run is
pending, distinct from the successful native controller commands. Canonical sources without independent durable replica proof remain
held. No new recovery volume was provisioned; all native receipts remain private.
Native acceptance passed. All 48 unqualified WAV placements remain present with
per-path causes, including shared logical IDs; all 832 canonical NVMe paths remain
protected. Prepared future managed admission remains unqualified and disabled.
Private evidence: native-acceptance.json, native-access-verification.json,
native-backup-restore-verification.json, native-hourly-verification.json and
native-orphan-repair-verification.json under the operator storage runtime home.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
