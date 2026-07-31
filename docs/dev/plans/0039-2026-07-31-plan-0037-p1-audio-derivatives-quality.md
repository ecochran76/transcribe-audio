# Plan 0039 | Plan 0037 P1 audio derivatives and quality

State: CLOSED

Lane: P10

Plan Version: 1

Parent: Plan 0037 P1

Owner: primary agent

Expected Write Surface: `acoustic_audio_derivatives.py`, focused tests, Plan
0037, `ROADMAP.md`, and `RUNBOOK.md`; private artifacts and receipts only under
`~/.local/state/transcribe-audio/plan-0037/audio-derivatives/`.

## Vision alignment

P1 advances durable transcript evidence by making source audio reproducibly
decodable and measurable without changing the original. Acoustic identity
remains maturity `1 - Built`; this packet supplies the replayable audio seam
needed for later P2-P4 shadow evaluation toward maturity `2 - Shadow`.

Measured effect: every promoted P1 derivative has an immutable source hash, a
complete recipe, an output hash, an original-time map, deterministic signal
quality metrics, and verifiable operation receipts.

Evidence: interface and edge-case tests, content-addressed private artifacts,
an unchanged replay, a non-destructive rollback receipt, a short manual audio
smoke, joined regressions, and independent review.

This packet does not prove that cleaned audio improves identity, does not run
VAD or enhancement, and does not change the default transcription or speaker
pipeline.

## Scope

- Add one focused internal audio-derivative module using the installed local
  `ffmpeg`/`ffprobe` tools behind a narrow Python interface.
- Hash and inspect the immutable source before processing.
- Decode to a versioned PCM WAV baseline with explicit sample-rate, sample
  format, and channel policy.
- Write derived audio by output content hash under a private user-scoped root.
- Compute deterministic duration, peak, RMS, DC-offset, clipping, and digital
  silence metrics from the derived PCM stream.
- Preserve an exact source-to-output timestamp map for the identity transform.
- Provide dry-run, apply, replay, and rollback receipts. Apply and rollback
  require explicit tokens. Rollback revokes the derived run without deleting
  source or derived evidence.
- Fail closed on unsupported, corrupt, empty, silent, clipped, multi-channel,
  or hash-conflicting inputs while recording quality warnings where decoding
  remains valid.

## Non-goals

- No Silero VAD, silence removal, DeepFilterNet, RNNoise, pyannote, or model
  checkpoint acquisition.
- No destructive trim, rewrite, normalization, or replacement of source audio.
- No biometric enrollment, speaker embedding, verification score, threshold,
  or App Intelligence integration.
- No bulk processing of the frozen P0 corpus.

## Current state

P1 closed with a private, content-addressed baseline decoder and deterministic
quality assessment. Dry-run, apply, replay, and rollback bind source blob/hash,
exact tool paths and versions, the canonical command recipe, full timestamp
coverage, evidence hashes, and restrictive modes. Usable speech remains
explicitly unassessed until P2; no identity promotion is possible from P1
alone.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P1A design audit | delegated read-only reviewer | P0 | risks and validation contract returned |
| P1B derivative implementation | primary | P0 | dry-run/apply/replay/rollback and private writes implemented |
| P1C quality and edge cases | primary | P1B | deterministic metrics and fail-closed cases pass |
| P1D smoke and neutral review | primary plus delegated reviewer | P1B-P1C | short clip, replay, rollback, regressions, and findings pass |

Intended concurrency is two active agents. The primary owns all writes; one
read-only reviewer owns P1 design and terminal audit. The join is P1D. One
repair-and-rerun cycle is allowed before a remaining defect must keep P1 open.

## Acceptance criteria

- Source bytes and source mode remain unchanged across dry-run, apply, replay,
  and rollback.
- The recipe records source hash, ffmpeg/ffprobe versions, decode policy,
  channel policy, sample rate, sample format, parameters, and model revisions.
- The WAV derivative is stored by verified output SHA-256 with private `0700`
  directories and `0600` files.
- Timestamp maps are monotonic, non-overlapping, nonnegative, and bounded by
  measured source/output duration.
- Quality metrics distinguish measured values from warnings/abstention and
  cover silent, clipped, empty, corrupt, unsupported, and multi-channel input.
- Apply is explicit and idempotent; immutable conflicts fail closed.
- Replay re-hashes source, derivative, manifest, recipe, and quality evidence.
- Rollback is explicit, auditable, and non-destructive; a revoked run cannot be
  treated as active.
- No P2 processing, biometric material, model execution, prompt change, or
  external provider write occurs.

## Validation

- `.venv/bin/python -m pytest -q tests/test_acoustic_audio_derivatives.py tests/test_acoustic_identity_contracts.py`
- Joined transcript artifact, source-blob, speaker-evaluation, identity
  preprocessing, and workflow regression tests.
- Short manual audio smoke through dry-run, apply, unchanged replay, and
  rollback; record hashes, modes, durations, and receipt paths.
- Active planning-contract audit and full repository test suite.
- Reconcile `/root/p1_review` in the closeout.

## Terminal condition

Close only when original-to-derived replay and timestamp mapping pass, quality
edge cases are covered, private mode and immutable-conflict evidence pass, and
the independent P1 review has no unresolved blocker. Otherwise keep P1 open;
P2 and P3 must not begin.

## Closure evidence

- Focused contracts and derivative lifecycle: 30 passed.
- Joined transcript artifact/store, speaker evaluation, identity preparation,
  and workflow regressions: 146 passed.
- Full repository suite: 446 passed.
- Synthetic-only terminal smoke:
  `audio-run-7895c2d83afd287f79855eaa`, manifest SHA-256
  `696b02837b9c12877263dab493661cc668ab73c6bc6742ac97664bca3f147299`.
  Apply, idempotent apply, active replay, rollback, idempotent rollback, and
  inactive replay passed with 2.0 seconds mapped to 2.0 seconds and zero drift.
- Source/output SHA-256 values were
  `6eb50df72cf53112487c47897be623c1a867fa4aef8b6855ffb74b614f174d32`
  and
  `4c1a1a10f5ccc84fbb302bece43de92db4c71c02ef93c0d10ddce7e87bad05fe`.
  The retained synthetic evidence has 11 files at `0600` and 7 directories at
  `0700`; the run is revoked and inactive.
- Active planning audit and `git diff --check` passed. Graphiti was healthy at
  terminal readback; its retrieved facts were advisory and added no P1-specific
  authority.
- Independent reviewer `/root/p1_review_final` returned PASS after adversarial
  permission, tamper, replay, reuse, and rollback checks.
- No frozen corpus audio, VAD, enhancement, diarization, biometric model,
  provider write, or App Intelligence path ran in P1.
