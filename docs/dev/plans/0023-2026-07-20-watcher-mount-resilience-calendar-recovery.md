# Plan 0023 | Watcher Mount Resilience And Calendar Recovery

State: CLOSED

Lane: P06

## Scope

Keep healthy watcher jobs running when one configured watch root is temporarily
unavailable, expose that degraded job in readiness and heartbeat diagnostics,
restore the live service, and recover the confirmed calendar match for recent
Voice Recordings artifacts when their authoritative files are reachable.

## Non-Goals

- Do not create, remap, or assign a Windows drive letter.
- Do not guess calendar matches for recordings without an overlapping event.
- Do not rewrite private transcript content into repo-local fixtures or logs.
- Do not weaken fatal readiness checks for shared dependencies such as Python,
  `ffprobe`, or configured backend scripts.

## Current State

The Windows `E:` drive is absent while WSL retains a stale `/mnt/e` drvfs
mount. `Path.exists()` raises `OSError` for the Voice Recordings watch root,
which escapes the readiness preflight and places `transcribe-watch.service` in
an auto-restart loop even though three other configured watch roots remain
available. Three recent stored artifacts have no event metadata; current
calendar evidence gives one artifact a strong overlapping match, while the
other two do not have an overlapping timed event. Exact private repair evidence
is retained only in user-scoped runtime state.

## Acceptance Criteria

- A deterministic regression test reproduces an `OSError` from one watch root
  while another configured root remains healthy.
- Readiness reports the unavailable job as a warning and returns success when
  at least one watch job is usable.
- Runtime scans skip the unavailable job, include its reason in heartbeat
  blocked diagnostics, and continue scanning healthy jobs.
- A configuration in which every watch root is unavailable still fails
  readiness.
- Successful transcriptions that continue after a calendar lookup failure
  retain a structured warning kind and bounded reason in watcher state.
- The live watcher remains active with the missing Voice Recordings root
  reported as degraded.
- The confirmed artifact is repaired only against the authoritative Voice
  Recordings artifact, or is explicitly left pending when that drive remains
  absent.

## Validation

- Targeted watcher readiness and scan regression tests.
- Full `tests/test_transcript_artifacts.py` suite.
- Python compilation and `git diff --check`.
- Live `--check --check-json`, systemd restart/status, and heartbeat readback.
- Dry-run calendar repair/readback for the confirmed artifact once the
  authoritative artifact is reachable.

## Closure Notes

Closed on 2026-07-20 after the watcher was made resilient to one unavailable
watch root, structured successful-run calendar warnings were persisted, tests
passed, and the live service returned to an active degraded state with zero
restarts. The authoritative Voice Recordings drive remains absent, so the
confirmed calendar repair is intentionally pending in user-scoped runtime
state rather than being applied to the transcript-store copy alone.
