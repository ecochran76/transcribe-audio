# 0102: Operator CLI and bounded recurring storage maintenance

State: OPEN
Status: in-progress
Owner: /root/storage_maintenance_0102
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Operator CLI and bounded recurring storage maintenance, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

- 0098: Journaled media placement and verified migration
- 0099: Classify retention eligibility and reserve bounded scratch
- 0100: Verified compressed snapshots, rotation and isolated restore
- 0101: Preserve playback and processing across tier migration

## Acceptance Criteria

- [ ] Provide inventory, plan, migrate/reconcile, status, backup/restore and maintenance commands with private configuration/receipts.
- [ ] Use StableTime supported client/API and validated mount metadata, without fallback to guest time or absent mountpoint directory.
- [ ] Run one hourly worker with scope, bytes/time, retries, reservations, pins and per-item held states.
- [ ] Provide installable user service/timer and source-generation/config identity without activating private stores in tests.

## Current State

Pending implementation. Source acceptance, integration validation and native effects remain distinct.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
