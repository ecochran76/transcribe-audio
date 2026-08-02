# Plan 0037 | Audio enhancement and biometric speaker identity

State: CLOSED — TERMINAL STOP; product acceptance not met

Lane: P10

Plan Version: 7

Execution Mode: high-level campaign plan with bounded implementation packets

Critical-Path Owner: primary agent

## Vision alignment

This plan makes source audio reusable evidence for speaker identity. It adds
versioned speech cleanup, calibrated speaker verification, and a private
biometric reference library before context-assisted identity resumes.

Current maturity: `1 — Built` for acoustic speaker identity.

Target maturity: `2 — Shadow` after local blind comparison, followed by an
evidence-backed decision about advancement toward `3 — Operational`.

Measurable vision effect: increase correct and safely abstained speaker
assignments while reducing false high-confidence identity support. Preserve
the original evidence so later models can replay every recording.

Evidence gate: no acoustic result enters the default App Intelligence packet
until a blind local evaluation proves its calibration, false-acceptance,
abstention, and diarization-label grouping behavior.

## Scope

This campaign covers the complete local acoustic-evidence path:

- Preserve immutable original audio and create content-addressed derived audio.
- Add voice activity detection and non-destructive silence exclusion.
- Evaluate noise suppression and speech enhancement without assuming that
  cleaner-sounding audio improves identity evidence.
- Produce quality-scored, timestamp-preserving speaker windows.
- Compare purpose-built speaker-verification models on operator-confirmed
  local recordings.
- Add a private, provenance-backed biometric speaker-reference library.
- Calibrate open-set speaker verification and explicit abstention.
- Detect diarization labels that likely represent the same person.
- Feed bounded acoustic evidence into the existing host-prepared speaker clue
  packet before App Intelligence reasoning.
- Reprocess eligible historical audio through a dry-run-first, versioned,
  resumable workflow.
- Preserve Plan 0036 and resume its remaining gold review only after this
  acoustic foundation reaches the agreed gate.

## Non-goals

The campaign does not expand voice evidence into authentication or unattended
identity authority:

- No destructive replacement or rewriting of original audio.
- No identity enrollment from calendar membership, transcript inference, or
  an unreviewed model proposal.
- No use as authentication, authorization, liveness proof, fraud proof, or
  synthetic-audio detection.
- No claim that a voice match alone establishes identity.
- No raw biometric embeddings in portable transcript sidecars, prompts,
  Graphiti, email, Drive, Odollo, or external providers.
- No unattended bulk enrollment or historical reprocessing before a reviewed
  manifest and explicit apply gate exist.
- No automatic speaker confirmation until a later unseen evaluation satisfies
  the existing confidence and review policy.
- No reveal of the sealed Plan 0036 predictions during this plan's model or
  threshold development.

## Current state

P0 is closed through Plan 0038, P1 through Plan 0039, P2 through Plan 0040,
and P3 through Plan 0041. The repo now has frozen contracts and a private
24-recording conversation-disjoint corpus; immutable content-addressed PCM and
quality evidence; five-method development-only speech preparation with 15/15
successful attempts; and a synthetic private biometric-reference authority
with CAS lifecycle and P4 descendant invalidation.

P4 is open through
[Plan 0043](0043-2026-07-31-plan-0037-p4e2-successor-evaluation.md). P4C has
six active real profiles, P4D development diagnostics are complete, and P4D2
has frozen nine model-by-method thresholds from 396 held-out calibration
trials. P4E generation 1 ended in a terminal `STOP` after reveal but before
audio/model execution because its authority-bound P2 module lacked the required
evaluation split seam. That cohort is no longer blind terminal evidence; a new
sealed cohort/generation is required. The seam is now implemented and replay-
tested before any successor authority freeze. Seven fully disjoint candidates
were subsequently operator-reviewed and frozen into a replayed `3 / 2 / 2`
successor corpus. Exact private condition execution completed 7 P1 runs and 35
P2 method results. Channel, noise, telephone-bandwidth, and usable-duration
coverage passed, but every recording lacks explicit physical capture-device
provenance. Plan 0044 therefore closed with terminal `STOP` before generation-2
authority construction or biometric scoring. Plan 0045 is open to collect an
exact-seven, append-only, direct-operator device provenance authority without
inferring from encoding or rewriting the measured condition evidence. No
generation-2 production authority may apply yet. Plan 0046 Units A through C
have closed the exact archived-to-current P2 replay seam and independently
audited deterministic generation-2 pre-reveal preview without weakening
default replay or entering a writer. Its production freeze remains `not_run`
while the device composite is a hard gate. No
model/method selection, App
Intelligence integration, or historical reprocessing exists. Plan 0036 remains
sealed and paused after five of ten current gold reviews.

Plan 0045's already-frozen exact-seven device campaign now replays across
reviewed clean descendant commits without changing its original repository
binding. Production replay at `bb975ebe5e46f880cefadf4267d03e2b5d7ede83`
returned full-body equality and idempotently reopened case 1 with zero recorded
attestations. The next required evidence remains the direct physical-device
fact for that open case; no identity, filename, codec, or authorization fact
can substitute for it.

The host continues to build a bounded speaker clue packet from transcript,
calendar, contact, relationship, GWS, and Odollo evidence. App Intelligence
proposes identities only from prepared candidates and cited evidence.

A review-only experiment used WavLM Base Plus embeddings and cosine similarity
over short windows. It supplied useful supporting evidence, including evidence
that diarization can split one person across labels. It did not provide
speaker-verification training, calibrated probabilities, durable enrollment,
or an abstention contract.

The research authority is
[Acoustic processing and speaker verification research](../notes/2026-07-31-acoustic-processing-and-speaker-verification-research.md).

Plan 0036 is paused after five of ten current gold reviews. Its superseding
baseline remains complete and sealed. No prediction body has been read, and
no comparison has been revealed.

## Architecture

Add one deep `AcousticIdentityAnalyzer` module at the host-prepared evidence
seam. Its interface accepts source audio, diarized turns, eligible enrolled
people, and policy. It returns an `AcousticEvidenceBundle` without selecting a
final identity.

The implementation owns these internal steps:

```text
immutable source audio
  -> decode and channel policy
  -> voice activity and quality analysis
  -> optional versioned enhancement
  -> diarization and clean-window preparation
  -> speaker embeddings and enrollment comparison
  -> normalization, calibration, and abstention
  -> same-person label evidence
  -> bounded AcousticEvidenceBundle
  -> host-prepared App Intelligence clue packet
```

Model adapters are internal seams. Start with at least two real adapters so
model variation is justified: SpeechBrain ECAPA-TDNN and WeSpeaker. Keep a
deterministic fake adapter for interface-level tests. Treat pyannote, Silero,
DeepFilterNet, and RNNoise as internal processing implementations rather than
types exposed to callers.

## Storage and privacy contract

The original blob remains authoritative. Each derived track records its source
hash, recipe, model revisions, parameters, timestamps, output hash, and
creation audit. Timestamp maps must preserve citations to the original audio.

Store reference registrations and materialized biometric profiles under
separate private user-scoped runtime roots. P3 reference generations record
biometric-purpose approval, confirmed source segments, quality, session
diversity, lifecycle, and descendant invalidation without embeddings or
scoring eligibility. P4 profiles bind an immutable P3 generation and add exact
embedding model/preprocessing revisions, private aggregate representation,
dispersion, calibration eligibility, and lifecycle audit. Files and database
rows use restrictive permissions.

Portable sidecars may store derived scores, confidence bands, evidence IDs,
model revisions, quality summaries, and reference-profile IDs. They must not
store raw embeddings or unrestricted enrollment audio. Model prompts receive
only the bounded evidence bundle.

Support explicit profile supersession, withdrawal, and deletion. A deleted or
withdrawn profile must stop contributing to future evidence while historical
audits retain non-biometric references needed to explain past decisions.

## Work graph

The campaign has one serialized critical path with bounded comparison work
inside it.

| Packet | Outcome | Dependency | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 | Freeze contracts and evaluation corpus | None | plan, schemas, private manifests, tests | Reviewed storage, privacy, benchmark, and model-license inventory |
| P1 | Build immutable audio-derivative and quality module | P0 | focused audio module, artifact schema, tests | Original-to-derived replay and timestamp mapping pass |
| P2 | Add VAD, enhancement, and diarization preparation | P1 | internal adapters and tests | No-enhancement, Silero, DeepFilterNet, RNNoise, and diarization comparison receipt |
| P3 | Add private biometric reference authority | P0, P1 | user-scoped reference store, approval/lifecycle interfaces, tests | Reviewed reference registration, supersession, withdrawal, deletion, CAS, and permissions pass |
| P4 | Materialize profiles and run verification calibration | P2, P3 | model adapters, profile materializer, evaluation runner, private results | One selected or rejected model decision with exact metrics |
| P5 | Integrate bounded acoustic evidence | P4 | speaker preprocessing and workflow modules, sidecar schema, tests | Host validation, abstention, and no-raw-biometric prompt proof pass |
| P6 | Reprocess a staged historical cohort | P5 | reprocessing workflow and private manifests | Dry run, reviewed apply, idempotent replay, and rollback pass |
| P7 | Measure identity effect and resume context path | P6 | evaluation artifacts and planning authorities | Blind comparison decides accept, refine, reject, or stop |

P2 model comparisons can run independently after P1, but P4 is the join.
P3 owns reference authority; P4 alone owns embeddings, scoring profiles, and
calibrated named-person scoring. P3 must complete before P4 materialization.
No parallel work may share live enrollment or reprocessing write surfaces.

## Packet details

Each packet produces one bounded outcome on the campaign's critical path.

### P0 | Contracts and evaluation corpus

State: CLOSED via Plan 0038.

Define versioned schemas for derived audio, quality, enrollment profiles,
verification trials, acoustic evidence, and reprocessing manifests. Inventory
code and checkpoint licenses. Select operator-confirmed source recordings
without reading Plan 0036 predictions.

Split trials by conversation, not by window. Include same-person and
different-person pairs across telephone, room, device, noise, overlap, and
usable-duration conditions. Keep all source and evaluation artifacts private.

### P1 | Audio derivatives and quality

State: CLOSED via Plan 0039.

Create a deep module that decodes audio, records channel policy, computes
quality measures, and writes content-addressed derived artifacts. Preserve an
exact timestamp map to the original recording. Provide deterministic dry-run,
apply, replay, and rollback receipts.

### P2 | Speech preparation

State: CLOSED via Plan 0040.

Use Silero VAD as the initial speech detector. Compare no enhancement,
DeepFilterNet, and RNNoise. Evaluate pyannote Community-1 for diarization,
overlap, and speaker-change preparation.

P2 proved preparation yield and timing integrity only. Downstream
transcription, diarization, and verification behavior plus enhancement
selection join P4/P7; no cleaner-audio outcome claim was made at P2 closure.

### P3 | Biometric reference library

State: CLOSED via Plan 0041.

Register only biometric-purpose-approved speaker-segment references with exact
provenance. Support multiple sessions per opaque person reference and preserve
session/device/acoustic variation metadata. Add reviewed create, supersede,
withdraw, and delete operations with audits, CAS, descendant invalidation, and
restrictive permissions. Do not create embeddings or scoring-eligible profiles
in P3.

Closed at synthetic maturity `2 - Shadow`: the private reference-only
authority, mandatory lineage/test-fixture split, immutable lifecycle replay,
action-specific approvals, deletion tombstones, and independently anchored P4
descendant invalidation passed terminal review. No real enrollment occurred.

### P4 | Verification and calibration

State: CLOSED with terminal `STOP` via Plans 0042, 0043, and 0048.

Materialize model-specific profiles from immutable eligible P3 reference
generations. Benchmark SpeechBrain ECAPA-TDNN, WeSpeaker CAM++, and one WeSpeaker ResNet or
ECAPA checkpoint. Add NVIDIA TitaNet only if the first comparison leaves a
specific quality or deployment question unanswered.

Compare raw and enhanced audio, multiple aggregation strategies, candidate
margin, score normalization, and quality-aware calibration. Select operating
thresholds from local held-out trials. Do not reuse public benchmark thresholds
as confidence values.

### P5 | Pipeline integration

State: `not_run`; closed by the P4/P7 evidence gate.

Add `AcousticEvidenceBundle` to host-prepared identity evidence. Require cited
acoustic evidence IDs and prepared candidate IDs. Keep final scoring and
confidence host-owned. Acoustic evidence may support, contradict, or abstain;
it may not create an unprepared person.

Use within-recording similarity to propose same-person diarization-label
groups. Preserve mixed and unresolved labels rather than forcing merges.

### P6 | Historical reprocessing

State: `not_run`; closed by the P4/P7 evidence gate.

Inventory eligible audio and report missing, corrupt, unsupported, already
processed, and policy-excluded items. Prepare a small reviewed cohort before
larger batches. Bind every result to source hash, recipe revision, model
revision, and previous artifact lineage.

Reprocessing must be resumable and idempotent. It must never overwrite the
original transcript or audio. New derived artifacts remain distinguishable
from the historical production result.

### P7 | Outcome measurement and continuation

State: CLOSED with terminal `STOP` via Plan 0048.

Measure the complete acoustic path on an unseen chronological cohort. Report
speaker accuracy, false acceptance, false rejection, abstention, calibration,
same-person label grouping, transcription change, diarization change, and
coverage.

After P7, resume the paused gold-review authority or create an explicit
successor evaluation if the changed review method would invalidate the old
comparison. Continue context-assisted speaker identity only after that
decision is recorded.

## Acceptance criteria

The campaign must satisfy all of these conditions before closure:

- Original audio remains immutable and independently addressable.
- Derived tracks are content-addressed, reproducible, timestamp-aligned, and
  versioned by complete processing recipe.
- Silence exclusion, enhancement, and diarization preparation report quality
  and abstention reasons.
- At least two purpose-built speaker-verification models are compared on the
  same private, conversation-separated trial set.
- Thresholds and confidence bands are calibrated on held-out local evidence.
- Biometric enrollment requires operator-confirmed provenance and supports
  supersession, withdrawal, and deletion.
- Raw embeddings remain in private user-scoped storage and never enter model
  prompts or portable sidecars.
- Acoustic evidence can identify likely split diarization labels without
  forcing ambiguous merges.
- App Intelligence receives only host-prepared candidate and evidence IDs.
- Historical reprocessing is dry-run-first, approval-gated, resumable,
  idempotent, and non-destructive.
- A blind evaluation records exact denominators and a terminal decision before
  the acoustic path becomes default.

## Validation

Validation covers module behavior, private runtime safety, and measured
identity outcomes:

- Interface-level tests with deterministic fake audio and model adapters.
- Golden artifact and schema tests for source hashes, timestamp maps, recipes,
  permissions, and replay.
- Corrupt, silent, clipped, overlapped, short, multi-channel, and unsupported
  audio tests.
- Enrollment provenance, withdrawal, deletion, tenant/user isolation, and
  no-prompt-leak tests.
- Model license and checkpoint hash inventory.
- Private blind evaluation with frozen trial manifests and prediction-excluded
  gold.
- Manual audio smoke on short clips for each promoted processing path.
- Joined transcription, calendar, watcher, artifact, speaker-preprocessing,
  store, API, and planning audits.

## Definition of done

Plan 0037 is done when versioned audio cleanup and a private biometric library
are integrated as bounded host evidence, eligible historical audio can be
reprocessed safely, and an unseen evaluation supports an explicit accept,
refine, reject, or stop decision. Completion does not enable automatic speaker
confirmation unless the existing authority and confidence gates separately
approve it.

## Terminal campaign closeout

Plan 0037 is closed as an unsuccessful, fail-closed campaign rather than a
successful Definition-of-Done completion. The generation-2 successor reveal
proved zero evaluation/profile subject overlap. All nine frozen model-by-method
units therefore have zero possible genuine and impostor trials, below the
precommitted 20/100 minima. Applied Plan 0048 run
`generation-2-evaluation-stop-5945db0810a482bbbe80db74` records terminal
`STOP` with full-body replay.

P5 integration and P6 historical reprocessing did not run. No evaluation
audio was prepared, no windows or exact-trial child were created, no model or
score ran, no terminal metrics were calculated, and no candidate or method was
selected. The acoustic path remains shadow-only and unavailable to default App
Intelligence or automatic speaker confirmation. A future attempt requires a
new plan and a cohort whose frozen profile coverage can satisfy every required
trial class before model execution.
