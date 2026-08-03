# Plan 0051 | Generation-4 Media Qualification

State: CLOSED

Lane: P10

## Vision outcomes

This plan advances trustworthy speaker inference and pipeline yield from Level
1 toward Level 2 by preventing malformed or misleading media from entering a
frozen acoustic evaluation. It enables, but does not prove, acoustic identity
quality or combined voice-and-context identification.

## Scope

- Resolve and bounded-refresh `Documents/Sound Recordings` through
  file-searcher.
- Freeze an explicit candidate list of at most 12 top-level source recordings.
- Reject symlinks, duplicate bytes, prior Plan-0037 source overlap, invalid or
  multiple audio streams, unsupported channels, files shorter than 60 seconds,
  decode failure, and decoded-duration drift above 0.05 seconds.
- Fully decode each candidate through the frozen FFmpeg path without retaining
  decoded audio.
- Persist an immutable private qualification manifest and aggregate-only
  portable receipt.
- Authorize only a separate Generation-4 cohort-preview plan when at least
  seven unique candidates qualify.

## Non-Goals

- No cohort membership, speaker gold, profile, threshold, window, trial,
  biometric model, score, metric, identity assignment, enrollment, integration,
  or historical reprocessing.
- No mutation, repair, transcoding, replacement, or deletion of source media.
- No claim that qualified recordings contain the required people or labels.

## Current State

Generation 3 is closed with terminal `STOP` because one frozen M4A decoded
89.776791 seconds shorter than its advertised duration. File-searcher bounded
refresh confirms 276 audio files under the source folder. No Generation-4
runtime authority exists.

## Acceptance Criteria

- Candidate selection is explicit, bounded, deterministic, top-level, and
  private.
- Every source is byte-hashed and compared with prior Plan-0037 evidence.
- Full decode measures duration without retaining decoded audio.
- At least seven unique candidates pass every frozen rule, or the plan closes
  truthfully without authorizing cohort construction.
- Preview, apply, and replay bind exact source bytes, tool revisions, rules,
  repository authority, candidate outcomes, and negative action vector.
- Portable output contains only counts, hashes, reason codes, actions, and
  privacy flags.

## Validation

- Focused adversarial tests for overlap, duplicates, symlinks, stream shape,
  short duration, decode failure, drift, stale preview, and replay mutation.
- Full repository test suite, compilation, and `git diff --check`.
- Clean pushed repository authority before production preview/apply.
- Live no-write qualification preview, immutable apply, `0700`/`0600`
  permission check, and full-body replay.

## Execution packets

1. Durable vision and governed plan wiring.
2. Qualification preview/apply/replay implementation and tests.
3. Clean pushed production preview and reviewed apply.
4. Canonical closure in this plan, `ROADMAP.md`, and `RUNBOOK.md`.

## Terminal conditions

- Fewer than seven passing candidates closes without cohort authorization.
- Any source or repository drift stops before authority write.
- Any partial, nonprivate, or non-replayable authority is a plan failure.

## Outcome

Implemented preview/apply/replay authority at clean pushed commit `6d7ad4c`.
The complete 12-candidate no-write preview left the Generation-4 runtime empty
and froze preview hash
`af5bcf2d8e60b811bcddbb875dd1044f69a090346c6118525c5c5dd80bc49974`.
Ten candidates passed every rule; two were rejected only as shorter than 60
seconds. There were zero prior-evidence overlaps, duplicate-byte candidates,
stream-shape failures, decode failures, or duration-drift failures.

The immutable private manifest hash is
`8b115bb92930916b087f114ab396f43f08d40b39f5faff8e1254d30a709c29fe`;
qualified-set hash is
`e3c908f80c922365ead50795728feb959d8aa93e542ee2882be79efc456e48be`.
Authority-driven full-body replay re-decodes the exact private candidates while
retaining no audio. Runtime directories and files are `0700` and `0600`.
Only a separate Generation-4 cohort preview is authorized. This plan does not
prove speaker coverage or acoustic identity quality.
