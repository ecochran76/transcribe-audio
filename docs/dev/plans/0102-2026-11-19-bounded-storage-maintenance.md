# 0102: Operator CLI and bounded recurring storage maintenance

State: CLOSED
Status: complete
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

- [x] Provide inventory, plan, migrate/reconcile, status, backup/restore and maintenance commands with private configuration/receipts.
- [x] Use StableTime supported client/API and validated mount metadata, without fallback to guest time or absent mountpoint directory.
- [x] Run one hourly worker with scope, bytes/time, retries, reservations, pins and per-item held states.
- [x] Provide installable user service/timer and source-generation/config identity without activating private stores in tests.

## Current State

Validated integrated controller, hourly units and producer admission at5f08a65.75focused checks passed atb879664;4producer checks passed after final path guard, including actual bounded CPU publication and unchanged resume. Native installation and migration remain0103. Producer policy is explicit store_root/storage-maintenance.json outside frozen processing config.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
