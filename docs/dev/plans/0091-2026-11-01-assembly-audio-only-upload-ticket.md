# Plan 0091 | Audio-only uploads with original recording provenance

State: CLOSED
Status: resolved (source acceptance)
Scope: One complete vertical slice implementing Plan0090.
Blocked by: None (can start immediately).
Spec: Plan0090 audio-only AssemblyAI upload specification.

## What to Build

Prepare and upload lossless audio from video recordings, preserve original
recording/calendar provenance and exports, clean up working files on all paths,
and retain audio-only behavior and the watcher backend fallback contract.

## Current State

Operator approved the single-ticket scope and recording-processing test seam.
Primary completed implementation and local validation on the integration branch.

## Acceptance Criteria

- [x] Video uploads contain decodable audio only, with lossless preparation.
- [x] Original recording content and calendar/export provenance are retained.
- [x] Audio-only inputs retain direct upload behavior.
- [x] Working files are cleaned on success and failure.
- [x] Preparation failures are visible and never upload original video.
- [x] Local focused/regression tests and synthetic-media CLI smoke pass.
- [x] Standards and Spec review completed with blockers resolved.

## Non-Goals

No provider smoke, memory-incident diagnosis, backlog reset, main merge,
installed-service changes, remote publication or release.

## Validation

Plan0090 records commands, outcomes, initial failures and Standards/Spec review.
Source result: `1dcea2797704827a5c022da972158a1ad1315f79`.
Implemented: yes. Integrated on feature branch: yes. Validated locally: yes.
Repo-native tracker: resolved CLOSED for source acceptance. Main integration and
installed-service activation: not performed.
