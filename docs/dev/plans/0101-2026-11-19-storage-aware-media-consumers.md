# 0101: Preserve playback and processing across tier migration

State: OPEN
Status: in-progress
Owner: /root/storage_consumers_0101
Spec: 0097-2026-11-19-recording-storage-spec.md

## What to build

Preserve playback and processing across tier migration, as a complete operator-visible capability governed by accepted storage policy.

## Blocked by

- 0098: Journaled media placement and verified migration

## Acceptance Criteria

- [ ] Integrate catalog resolution into ingest/playback, corpus/derivative access and managed recording resume.
- [ ] Preserve original recording IDs, expected hashes and frozen checkpoint/provenance semantics.
- [ ] HDD absence permits text/index operations; no symlink bypass or erasure resurrection.
- [ ] Existing public consumer tests and cross-tier access/resume behavior pass without inference.

## Current State

Pending implementation. Source acceptance, integration validation and native effects remain distinct.

## Scope and Non-Goals

Own this ticket only; no shared tracker writes by workers. No Speech, provider/inference
operations, source erasure, new test framework or changes to unrelated dirty work.

## Validation

Existing pytest public seams with real files/SQLite; focused red/green receipts and
local validation on the integration tip. No required skip is a passing result.
