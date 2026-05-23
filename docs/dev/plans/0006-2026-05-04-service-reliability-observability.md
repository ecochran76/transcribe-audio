# Plan 0006 | Service Reliability And Observability

State: CLOSED

Lane: P06

## Scope

Make the unattended watcher easier to diagnose and harder to silently stall.

## Non-Goals

- No changes to transcription provider behavior unless needed for observability.
- No new daemon supervisor beyond systemd.

## Current State

The watcher has heartbeats, no-progress restart behavior, startup readiness checks, and an explicit `--check` doctor path. Missing `ffprobe`, missing watch directories, missing backend scripts, and missing readout scripts are reported before the service loop starts; `--check --check-json` provides a machine-readable readiness payload. Candidate state now records `blocked_kind`, `blocked_reason`, and `blocked_since`, and heartbeat logs summarize queued work as `blocked=kind=count`. Backend failures record `failure_kind` and `failure_reason`, so retry backoff remains visible in state and heartbeat diagnostics.

## Work Items

- Done: add startup dependency checks for `ffprobe`, watch directories, configured backend scripts, and readout scripts.
- Done: include blocked-reason summaries in heartbeat logs.
- Done: distinguish incomplete media, missing tools, auth/config failures, and retry backoff in state.
- Done: add `--check` and `--check --check-json` service readiness commands.
- Done: document service health and recovery commands.

## Acceptance Criteria

- Missing `ffprobe` fails loudly during `--check` and service startup.
- Queued candidates show actionable reasons in state and heartbeat logs.
- `systemctl --user status transcribe-watch.service` plus recent journal lines have enough log context to diagnose common stalls.

## Validation

- Unit tests for readiness classification, blocked-state persistence, retry backoff visibility, media-probe classification, and heartbeat summary formatting.
- Manual `--check` readiness smoke.
- Manual service restart and heartbeat/journal check.

## Closure Notes

Closed on 2026-05-23 after watcher readiness, blocked-state, heartbeat, tests,
README documentation, and live service restart validation were completed.
