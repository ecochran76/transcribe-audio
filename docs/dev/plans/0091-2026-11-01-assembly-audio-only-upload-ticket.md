# Plan 0091 | Audio-only uploads with original recording provenance

State: OPEN
Status: ready-for-agent
Scope: One complete vertical slice implementing Plan0090.
Blocked by: None (can start immediately).
Spec: Plan0090 audio-only AssemblyAI upload specification.

## What to Build

Prepare and upload lossless audio from video recordings, preserve original
recording/calendar provenance and exports, clean up working files on all paths,
and retain audio-only behavior and the watcher backend fallback contract.

## Current State

Operator approved the single-ticket scope and recording-processing test seam.
Primary coordinator implementing sequentially on the integration branch.

## Acceptance Criteria

- [ ] Video uploads contain decodable audio only, with lossless preparation.
- [ ] Original recording content and calendar/export provenance are retained.
- [ ] Audio-only inputs retain direct upload behavior.
- [ ] Working files are cleaned on success and failure.
- [ ] Preparation failures are visible and never upload original video.
- [ ] Local focused/regression tests and synthetic-media CLI smoke pass.
- [ ] Standards and Spec review completed with blockers resolved.

## Non-Goals

No provider smoke, memory-incident diagnosis, backlog reset, main merge,
installed-service changes, remote publication or release.

## Validation

Pending; commands and results will be recorded in the parent execution ledger.
Tracker resolution remains separate from implemented/integrated/validated state.
