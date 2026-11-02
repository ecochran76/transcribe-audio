# Plan 0092 | Preserve literal recording paths before wildcard expansion

State: CLOSED (source acceptance)
Scope: Fix existing bracketed recording paths shared by both speech backends.
Owner: Primary; sequential work, no delegation
Branch: fix/literal-media-input
Base: 07b0e6b505254f7eec0bcd1711da01118140b2df

## Current State

The watcher repeatedly fails existing converted recordings whose names contain
`[raw recording]`. A provider-free reproduction fails identically for a tiny
local file named `meeting [raw recording].mp4` and for the minimized `[a]`;
`plain.mp4` succeeds. Failure is deterministic across two baseline runs.
A correct public seam exists: shared input discovery called by both CLIs.
The regression test was observed red before the fix, then green afterward.

## Diagnosis

Ranked predictions shown to operator before probing:
1. Glob syntax consumes bracketed literals: exact path precedence or escaping
   must restore selection.
2. Filesystem access fails: direct reads must fail too.
3. Watcher command construction alters the argument: direct resolution should work.

Direct reads succeed; unescaped glob expansion returns nothing; escaped
expansion returns the exact file. The shared implementation checks a wildcard
regex before existence. Hypothesis1 is confirmed; hypotheses2/3 do not explain
the local reproduction. No debug logging or private-provider probes were needed.

## Scope And Acceptance Criteria

- Existing literal files win over glob interpretation, including bracketed
  filenames/directories; do not silently select another glob-matching recording.
- Both backends retain the shared resolver; no file renaming or state reset.
- Intentional wildcard, character-class and recursive patterns still expand.
- Home expansion, deduplication, directory filtering and unmatched warnings remain.
- Original and minimized repro, focused regressions and synthetic CLI export pass.
- Verify installed-source selection of affected real paths without provider calls.

## Vision Outcomes

Restores automatic transcript yield for previously ineligible recording names.
This repair moves bracketed-path input discovery from0(absent/broken) to1(built)
on source acceptance; installed path selection and subsequent watcher work give
separate operational evidence. It does not prove speaker/calendar accuracy or
completion of the conversation knowledge loop.

## Non-Goals

No unrelated memory-pressure repair, transcription-model changes, review
assignment, recording rename, backlog reset, forced retry, biometric work or
legal-matter changes. Existing scheduled retry policy remains intact.

## Validation

Primary executes focused input-discovery tests, then selected shared-output,
calendar, watcher and audio-upload regression tests. Hosted CI excluded by
operator cost policy. Short synthetic media CLI smoke uses local fake provider
responses, not private recordings or paid API calls.

Feedback command from either checkout:
`python ~/.local/state/transcribe-audio/diagnostics/literal-media-input/repro.py`
Imports the current checkout's resolver; checks bracketed original, minimal
bracketed path and plain-path control. Baseline exits1; fixed checkout exits0.

Bounds: one broad primary Standards/Spec review, one bounded remediation pass;
no new workers, no provider retries for diagnosis. Checkpoint on source acceptance
and installed selection. Preserve the pre-existing watcher history.

Memory disposition: forbidden; no explicit request to update agent memory.

## Acceptance Evidence | 2026-11-01

- Original provider-free feedback command: `python
  ~/.local/state/transcribe-audio/diagnostics/literal-media-input/repro.py`.
  Before: bracketed original and `[a]` resolved false while plain control passed;
  exit1 with the exact no-files-matched warning. After: all three resolve true,
  exit0. Two initial baseline runs reproduced the same bracket failure.
- Regression command: `python -m pytest -q tests/test_media_input_discovery.py`.
  Observed red: expected the existing bracketed path, got an empty list.
  First green:1passed0.62s.
- Selected regression command: `python -m pytest -q
  tests/test_media_input_discovery.py tests/test_assembly_audio_upload.py
  tests/test_transcript_artifacts.py tests/test_readouts.py`.
  Final result: **67passed1.79s**. One new test initially expected the pre-existing
  directory warning on stdout instead of stderr; corrected test expectation,
  not product behavior. No flaky retry or hidden failure.
- CLI smoke: `python ~/.local/state/transcribe-audio/diagnostics/literal-media-input/cli_smoke.py`.
  Short synthetic bracketed MKV successfully emitted DOCX/TXT/JSON through the
  actual AssemblyAI main interface with fake provider responses. Audio-only
  upload and unchanged original source hash checked. No private/provider work.
- Exact previously failed real inputs: **24/24 resolve correctly**, zero read
  errors. This is input-selection proof, not a claim of completed transcription.
  Private path readback and synthetic smoke receipts remain user-scoped under
  `~/.local/state/transcribe-audio/diagnostics/literal-media-input/`.
- `python -m py_compile transcribe_common.py tests/test_media_input_discovery.py`
  and `git diff --check`: pass. No debug instrumentation introduced or retained.

Environment: existing Python3.12 project environment, Linux/WSL, ffmpeg for
synthetic smoke. PATH explicitly includes host ffmpeg. Focused plus selected
presubmit only; no comprehensive/live-provider/hosted-CI claim. All validation
performed by primary, no delegation. Hosted Actions skipped by operator policy.

## Standards Review

Frozen comparison: base07b0e6b to this exact working patch. Primary checked the
shared-module boundary, literal path precedence, no recording/state mutation,
minimal source diff and behavior-oriented tests against repo guidance. No
accepted blocking findings. One broad review; no remediation pass needed.

## Spec Review

Original symptom is reproduced and fixed at the shared seam used by both CLIs.
Literal path precedence also prevents selecting a different character-class
match. Intentional class/recursive globs, home paths, deduplication and warnings
remain covered. No unrelated product behavior was added. No blocking findings.

## Integration Boundary

Deploy this ordinary repair to the same already-authorized watched-folder
service without forcing retries or widening provider activity. The watcher
launches fresh backend processes, so the next scheduled attempts load the fixed
resolver without interrupting the parent or rewriting its state. Existing retry
history/backoff is preserved. Main reconciliation and installed-source evidence
will be appended after the coherent source commit.
