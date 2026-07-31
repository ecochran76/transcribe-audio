# Plan 0040 | Plan 0037 P2 speech preparation comparison

State: OPEN

Lane: P10

Plan Version: 2

Parent: Plan 0037 P2

Owner: primary agent

Expected Write Surface: one focused speech-preparation module and tests,
dependency/model inventory updates, Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`;
private assets, derived audio, and comparison receipts only under
`~/.local/state/transcribe-audio/plan-0037/speech-preparation/`.

## Vision alignment

P2 advances durable diarized transcript evidence and speaker inference by
making speech-region, enhancement, overlap, and speaker-change preparation
reproducible without changing source audio. Speech preparation is currently
maturity `1 - Built` through P1's decode seam. This packet targets maturity
`2 - Shadow` for preparation on a small development-only cohort; the overall
acoustic identity path remains `1 - Built` until P3/P4 provide reviewed
profiles and calibrated verification evidence.

Measured effect: every attempted method has an exact code/model revision,
asset hash, timing map, output/abstention status, resource cost, signal-quality
delta, and comparison denominator. The packet measures preparation yield and
timing integrity; it does not claim identity improvement.

Evidence: interface and adversarial tests, pinned license/acquisition
inventory, synthetic fixtures, private unchanged replay, a development-only
comparison receipt, joined regressions, and independent terminal review.

## Scope

- Define internal host-owned adapters for no enhancement, Silero VAD,
  DeepFilterNet, RNNoise, and pyannote diarization preparation. Provider types
  must not escape the adapter seam.
- Bind every run to a replay-verified P1 derivative, exact source/output hashes,
  recipe, tool/package/model revisions, asset hashes, parameters, and private
  paths.
- Preserve original-time mappings for speech regions and any enhanced output;
  reject gaps, inversions, ambiguous bases, or drift beyond declared bounds.
- Normalize every terminal status to `success`, `failure`, or `blocked`, with a
  reason code such as `not_acquired`, `human_gate`, `not_run_dependency`,
  `abstained_no_speech`, `timeout`, or `asset_hash_mismatch`. Missing evidence
  is never interpreted as model failure.
- Compare no enhancement, DeepFilterNet, and RNNoise signal/timing evidence;
  record Silero speech regions and pyannote speech/overlap/change/diarization
  preparation where exact assets are authorized and available.
- Use synthetic clips first. After all non-gated checks pass, process only a
  bounded set of P0 development-split recordings. Calibration and evaluation
  splits remain sealed; no bulk corpus processing is authorized in P2.
- Provide dry-run, explicit apply, replay, and non-destructive rollback for
  private P2 artifacts and the aggregate comparison receipt.

## Non-goals

- No biometric enrollment, embeddings, named-person scoring, verification
  threshold, calibration, or identity promotion.
- No P0 calibration/evaluation audio, Plan 0036 prediction read, historical
  reprocessing batch, transcription replacement, or default pipeline change.
- No acceptance of gated model terms, contact-information sharing, token use,
  or gated checkpoint download without an explicit operator gate.
- No claim that cleaner audio improves transcription, diarization, or identity
  unless a later packet measures that downstream outcome.
- No external provider write or App Intelligence invocation.

## Current state

P1 is closed and pushed at `20c67d7`. The host can replay immutable 16 kHz
mono PCM derivatives with complete timestamp and quality evidence. The venv
contains PyTorch 2.11.0, torchaudio 2.11.0, onnxruntime 1.24.4, and
pyannote.audio 4.0.4, but no Silero/DeepFilterNet package or RNNoise executable
is installed. The P0 inventory records every P2 checkpoint as `not_acquired`.
A Hugging Face cache contains incomplete/unreviewed pyannote-related snapshots;
their presence does not prove a complete model, accepted current terms, pinned
inventory, or authorization to use/download them.

P2B is implemented and terminally reviewed. The host-owned seam now publishes
a normalized five-method readiness matrix, run-bound dry-run/apply/rollback
tokens, replay-verified P1 input binding, strict private immutable receipts,
finite and method-specific timing validation, deterministic fake adapters only
under explicit test mode, and a no-enhancement baseline that reuses the P1
artifact without rewriting audio. A lifecycle apply can succeed while the
aggregate comparison truthfully remains
`blocked/required_real_comparisons_not_run`; the current denominator is one
successful no-enhancement method and four blocked real methods. P2C-P2E and
the parent P2 remain open.

P2C now has a terminally reviewed acquisition planner and an exact
repo-owned open-candidate spec. Silero VAD `6.2.1`, DeepFilterNet `0.5.6`,
DeepFilterLib `0.5.6`, DeepFilterNet3, and signed RNNoise `v0.2` source/tag
identities are pinned to official URLs, revisions, sizes, and SHA-256 values
where upstream publishes them. The two artifacts without official SHA-256
must be hashed and content-addressed immediately after an authorized download
and before build/use. The planner records that Python 3.12 requires a local
DeepFilterLib source build, excludes pyannote terms/contact sharing and all
audio/model execution, and requires replay to supply its originally reviewed
byte hash. The operator supplied a standing blanket grant on 2026-07-31 for
the bounded Plan 0037 scope; the dry-run hash remains audit evidence, but no
per-run approval phrase or token is required.

Under that grant, P2C acquired all five open artifacts into private storage,
verified published SHA-256 values, computed the two missing SHA-256 values
before use, built DeepFilterLib 0.5.6 for CPython 3.12, built RNNoise v0.2 into
a private prefix, and bound installed assets in an immutable private
acquisition manifest. A synthetic production smoke ran no-enhancement,
DeepFilterNet, and RNNoise successfully; Silero correctly abstained on a
non-speech tone. Community-1 terms/contact sharing and bounded development
processing are also covered by the grant, but the local Hugging Face client
currently has no authenticated identity or access token.

The deterministic bounded development slice selected the three shortest
development recordings (2,892 seconds) and selected zero calibration or
evaluation recordings. All 12 open-method attempts succeeded, including
Silero speech-region evidence and full-length DeepFilterNet/RNNoise outputs.
One long-form DeepFilterNet failure was repaired by bounded contiguous
60-second processing and passed its single retry. Aggregate receipt SHA-256 is
`81aa1b407798409f2b4871f3eb5f0673de540ebab537ee6acb51570e09ce21fc`.
It binds corrected acquisition manifest SHA-256
`fc28406a6c2a8a84763a238940d0cec29a414e1d7952d74d69c9f597fdbe1d13`;
enhanced outputs are SHA-256-addressed and replay reopens and re-hashes each
private WAV. Earlier `a7304d...` evidence is explicitly superseded.
P2 remains open solely because Community-1 cannot be acquired/run without a
provider-authenticated Hugging Face identity.

Graphiti was healthy at P2 opening but returned only advisory older
speaker-preprocessing facts. Current repo plans, installed-package readbacks,
the P0 inventory, and hashed runtime assets control.

P2B checkpoint evidence: 40 focused tests and 456 full-repository tests pass.
The terminal synthetic lifecycle smoke is retained at
`~/.local/state/transcribe-audio/plan-0037/p2b-smoke-final` with P2 run
`speech-prep-f312247c2ba9a601ac38a9a8`, comparison denominator
`methods=5, attempted=1, success=1, failure=0, blocked=4`, active and inactive
replay proof, and all 15 files/10 directories at `0600`/`0700`. Read-only
reviewer `/root/p1_review_final` returned P2B PASS after adversarial timing,
readiness, privacy, tamper, and lifecycle review. No model download,
environment installation, or private corpus access occurred.

P2C planner checkpoint evidence: 66 focused acoustic tests and 482
full-repository tests pass. The persisted plan is
`acquire-open-585ef49febe61caf5a3d99b1`, SHA-256
`d4b2a4c800b10cd8604b4e2f73ac553a097652f0bc1271ff27def5628c9ac836`,
under the private P2 runtime root. Read-only reviewer
`/root/p1_review_final` returned terminal planner `PASS` after official
metadata, scope-exclusion, spec-drift, timestamp-tamper, serialization-tamper,
permission, and no-side-effect review.

## Standing authorization and fail-closed gates

- The operator's 2026-07-31 blanket grant authorizes all bounded Plan 0037
  acquisitions, installs/builds, gated terms/contact sharing, and development
  processing. Persisted hashes and dry runs are evidence controls, not
  authorization ceremonies; do not request per-run phrases or tokens.
- Open-license acquisition may proceed after selecting an immutable code
  or package revision, verifying official terms, and recording all acquired
  asset hashes. Downloads stay in private user-scoped caches/runtime paths.
- pyannote Community-1 is authorized but remains `status=blocked`,
  `reason_code=provider_auth_required` until the local Hugging Face client has
  an authenticated identity with repository access. Existing partial cache
  fragments do not constitute a complete pinned snapshot.
- A method without a complete pinned asset set emits `status=blocked` with
  `reason_code=not_acquired` or `provider_auth_required`; it must not silently fall back to
  another model.
- Development-cohort apply requires a persisted, hash-bound dry run and the
  standing grant; no per-run token is required.
  Synthetic tests may use deterministic fake adapters without model downloads.
- Stop and keep P2 open if an acquisition changes license posture, requires
  credentials/privilege, cannot be content-hashed, or if timing/source
  integrity cannot be proved.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P2A design/readiness audit | delegated read-only reviewer | P1 | gates, installed state, contracts, and validation matrix returned |
| P2B host seam and no-enhancement baseline | primary | P1 | deterministic fake/no-op adapters and lifecycle tests pass |
| P2C open candidate acquisition/adapters | primary | P2A-P2B | Silero, DeepFilterNet, and RNNoise revisions/assets are pinned or truthfully `blocked/not_acquired` |
| P2D gated diarization adapter | primary | P2B plus provider-authenticated Hugging Face access | pyannote runs from a complete pinned snapshot or records `blocked/provider_auth_required` |
| P2E bounded comparison and review | primary plus delegated reviewer | P2C-P2D | every required real method runs on the approved development cohort and receipt replay has no unresolved blocker |

Intended concurrency is two active agents. The primary owns all writes,
downloads, environment changes, and private processing. One read-only reviewer
owns design and terminal audit and may inspect synthetic receipts only. The
join is P2E. One repair-and-rerun cycle is allowed per terminal review; a
remaining blocker keeps P2 open.

## Acceptance criteria

- Adapters expose stable host objects and never leak provider-native objects,
  waveforms, embeddings, credentials, or raw private audio into portable
  receipts.
- Readiness distinguishes installed code, complete model assets, license/terms
  review, authorization, and runnable state with exact revisions and SHA-256.
- Every speech/overlap/change segment is monotonic, non-overlapping within its
  class, nonnegative, bounded, and mapped to original P1 time.
- Enhanced tracks are content-addressed, private, replayable, and retain a
  complete source-to-output timing proof. Source/P1 artifacts remain unchanged.
- No-enhancement, Silero, DeepFilterNet, RNNoise, and pyannote each appear in
  the comparison receipt with `success`, `failure`, or `blocked`, a reason code,
  and a truthful denominator. P3/P4 identity-effect measurement remains
  `blocked/not_run_dependency_p3_p4` rather than a fabricated zero.
- Poor quality, no speech, all speech, overlap, short audio, corrupt/tampered
  evidence, model failure, OOM/timeout, absent assets, and human-gated assets
  fail closed or abstain without fabricated zeros.
- Apply/replay/rollback are standing-grant governed, idempotent, no-clobber, mode-correct,
  tamper-evident, non-destructive, and prevent revoked-run reuse.
- Development comparisons cannot read calibration/evaluation splits, reveal
  Plan 0036 predictions, enroll biometrics, score names, or write externally.

## Validation

- `.venv/bin/python -m pytest -q tests/test_acoustic_speech_preparation.py tests/test_acoustic_audio_derivatives.py tests/test_acoustic_identity_contracts.py`
- Joined transcript artifact/store, speaker-evaluation, identity-preparation,
  and workflow regressions.
- Synthetic tone/noise/silence/overlap smoke through dry-run, apply, replay,
  rollback, and mode/hash audit.
- Development-only private comparison receipt with explicit per-method
  denominators and `status=blocked` plus precise `not_run_*` reason codes
  preserved.
- Active planning-contract audit, `git diff --check`, and full repository suite.
- Reconcile `/root/p1_review_final` P2 design and terminal reports.

## Terminal condition

Close only after the real no-enhancement, Silero, DeepFilterNet, RNNoise, and
pyannote preparation methods all run on the explicitly approved development
cohort with pinned/hash-verified assets, replayable timing evidence, synthetic
and bounded development validation, and no unresolved independent-review
blocker. Provider-authenticated Hugging Face access remains the unresolved
pyannote dependency; the operator authorization gate is already satisfied.
P2B/P2C implementation checkpoints may still commit and push truthful blocked
receipts without claiming the parent comparison complete.
