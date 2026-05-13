# Plan 0006 | Service Reliability And Observability

State: OPEN

Lane: P06

## Scope

Make the unattended watcher easier to diagnose and harder to silently stall.

## Non-Goals

- No changes to transcription provider behavior unless needed for observability.
- No new daemon supervisor beyond systemd.

## Current State

The watcher has heartbeats and no-progress restart behavior. A missing `ffprobe` in the systemd PATH caused queued candidates to remain unattempted while the service looked healthy.

## Work Items

- Add startup dependency checks for `ffprobe` and configured backend commands.
- Include blocked-reason summaries in heartbeat logs.
- Distinguish incomplete media, missing tools, auth/config failures, and retry backoff in state.
- Add a `doctor` or `--check` command for service readiness.
- Document service health and recovery commands.

## Acceptance Criteria

- Missing `ffprobe` fails loudly at startup or appears explicitly in heartbeat.
- Queued candidates show actionable reasons.
- `systemctl --user status transcribe-watch.service` has enough log context to diagnose common stalls.

## Validation

- Unit tests for readiness classification.
- Manual service restart and heartbeat check.
