# Plan 0024 | Voice Recordings D Drive Cutover And Catch-up

State: CLOSED

## Scope

Move the `syncthing-voice-recordings` watcher job from the degraded `/mnt/e`
root to the user-restored `/mnt/d/SyncThing/Voice Recordings` directory,
reconcile moved files against existing watcher state without duplicate
transcription, process genuinely new stable recordings, and complete the
pending confirmed calendar repair against the restored authoritative files.

## Non-Goals

- Do not remount, repair, or assign a drive letter to the degraded source disk.
- Do not broaden the job beyond its existing `*.m4a` intake policy.
- Do not retranscribe files already represented by successful equivalent
  watcher records.
- Do not guess calendar events for recordings without a timed overlap.

## Current State

The live job still targets `/mnt/e/SyncThing/Voice Recordings`, so readiness
reports `unavailable_watch_dir`. The replacement directory exists on `/mnt/d`.
The initial state comparison observed 89 moved recordings with equivalent
successful records and 6 genuinely unmatched `.m4a` files. The previously
confirmed calendar repair is present on the restored filesystem but its
embedded artifact paths still reference `/mnt/e`.

## Acceptance Criteria

- The configured Voice Recordings watch root is
  `/mnt/d/SyncThing/Voice Recordings`.
- Watcher readiness is green without an unavailable-root warning.
- Existing moved recordings are reconciled through the watcher state
  equivalence path rather than retranscribed.
- Genuinely new stable `.m4a` recordings are processed successfully or have a
  concrete recorded failure/block reason.
- The confirmed calendar repair is dry-run against the restored authoritative
  artifact and applied only when the match and destination paths are safe.
- The restarted systemd service remains active with no restart loop.

## Validation

- `.venv/bin/python watch_transcriptions.py --config watch_transcriptions.json --check --check-json`
- Focused watcher state/candidate comparison against the restored directory.
- Live `transcribe-watch.service` state and bounded journal monitoring through
  the catch-up cycle.
- Targeted dry-run and apply/readback for the confirmed calendar repair.
- `.venv/bin/python -m pytest tests/test_transcript_artifacts.py -q`
- `git diff --check`

## Closure Notes

Closed on 2026-07-20 after the live job moved to the healthy D: Syncthing
root. The first scan reconciled 89 moved recordings without retranscription.
Five genuinely new, valid recordings then completed transcription, calendar
matching, transcript-store ingestion, and participant identity warming. The
sixth unmatched path was an incomplete historical file under Syncthing's
`.stversions` directory; the watcher now excludes that archive by contract.

The pending eventless artifact was rebased from E: to D:, matched in a targeted
dry run, repaired against the authoritative files, and reconciled to one
canonical transcript-store row. Final readiness was green, the live heartbeat
reported `candidates=0 attempted=0 successes=0 failures=0 blocked=none`, and
the service remained active with zero restarts.
