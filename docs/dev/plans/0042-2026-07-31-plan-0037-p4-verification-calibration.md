# Plan 0042 | Plan 0037 P4 verification and calibration

State: OPEN

Lane: P10

Plan Version: 3

Parent: Plan 0037 P4

Owner: primary agent

Expected Write Surface: one focused acoustic-verification module and tests,
one pinned acquisition fixture, Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`;
private models, profiles, trials, and receipts only under
`~/.local/state/transcribe-audio/plan-0037/verification-calibration/`.

## Vision alignment

P4 advances the north-star speaker outcome by converting reviewed reference
segments into private model-specific evidence and measuring when that evidence
should support, contradict, or abstain. Acoustic identity is currently maturity
`1 - Built`: preparation is shadow-validated and the reference authority is
synthetically shadow-validated, but no real profile, verification score, or
calibration exists. This packet targets maturity `2 - Shadow` for calibrated
verification on conversation-separated local evidence.

Measured effect: same-person and different-person trial yield, false acceptance,
false rejection, equal-error diagnostic, open-set rejection, abstention,
calibration error, candidate margin, condition slices, enhancement sensitivity,
and same-person label-grouping precision/recall. Evidence is immutable model and
profile manifests, private calibration/evaluation receipts, replay, adversarial
tests, and an independently reviewed select/refine/reject/stop decision.

This packet does not prove that voice establishes identity, authorize automatic
speaker confirmation, or make acoustic evidence part of the default pipeline.

## Scope

- Add host-owned adapters for SpeechBrain ECAPA-TDNN, WeSpeaker CAM++, and
  WeSpeaker ResNet34. Provider-native objects and raw vectors never escape the
  private materialization/scoring module.
- Pin SpeechBrain `1.1.0`, exact upstream model revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, WeSpeaker code revision
  `dfa741957e5c11f477623b6e583d67d0af25ee88`, CAM++ revision
  `acf623ad8ca746e50baa432255cf8fc57c669c45`, and ResNet34 revision
  `ff1ac5bca8ef11e90662b879aa923979e0bd277b`.
- Review and bind code plus checkpoint terms separately. The official
  WeSpeaker pretrained-model authority says VoxCeleb checkpoints follow
  CC-BY-4.0 even where a hosting model card reports Apache-2.0. Bind the code
  archive and terms document at commit
  `dfa741957e5c11f477623b6e583d67d0af25ee88` by URL, byte size, and SHA-256;
  a mutable branch URL is not acquisition evidence.
- Materialize `transcribe-audio.biometric-profile.v1` artifacts only from a
  replay-validated, active P3 generation. Bind exact P3 generation/hash,
  preprocessing and model revisions, private aggregate representation,
  per-session/window dispersion, and calibration eligibility.
- Stage each descendant, register it with P3, then promote it through the
  existing independent P4 authority receipts. Withdrawal/deletion invalidation
  must make future resolution fail before any score is used.
- P4 separately owns profile states `staged`, `active`, `superseded`,
  `withdrawn`, and `deleted`. Supersede/withdraw/delete first disable scoring,
  then invalidate the P3 descendant. Delete removes all private embedding and
  aggregate bytes while retaining only a non-biometric tombstone containing
  opaque IDs, transition reason/time, prior artifact hash, and receipt hashes.
  Every score resolves both an active P4 profile and currently eligible P3
  descendant immediately before model execution and again before persistence.
- Compare original/no-enhancement, DeepFilterNet, and RNNoise paths. Use Silero
  and Community-1 preparation evidence to choose clean, timestamp-preserving,
  non-overlapped speaker windows without treating diarization labels as names.
- Freeze conversation-separated development, calibration, and evaluation trial
  manifests. Development may guide aggregation and features; calibration may
  select thresholds; evaluation remains sealed until the terminal run.
- Bind each split to the frozen P0 manifest with exact recording-set and
  conversation-set hashes in
  `docs/dev/fixtures/plan-0037-p4/split-access-policy.json`. Development is
  currently authorized; calibration and evaluation remain unauthorized states.
  The reviewed v1 policy SHA-256 is
  `41808c1b654b20ea8b395f65757db0ffc9f1a79862b31a6a2770268be1083467`.
  Calibration requires a separate exact apply authority after verified
  development receipts. Evaluation requires verified development/calibration
  receipt hashes, the frozen decision-policy hash, and a separate exact
  terminal-evaluation apply authority.
- Compare centroid and score-level aggregation, raw cosine scores, optional
  normalization, quality features, window agreement, candidate margin, and
  explicit abstention.
- Produce one terminal `select`, `refine`, `reject`, or `stop` decision with
  exact denominators and no fabricated zeros for missing evidence. Freeze and
  hash `docs/dev/fixtures/plan-0037-p4/terminal-decision-policy.json` before any
  evaluation reveal; precedence is `stop`, `reject`, `select`, then `refine`,
  and this evaluation generation cannot retroactively change the policy. The
  reviewed v1 policy SHA-256 is
  `98eadfd2a3a55a77d873ff0f3efbf7f2e75e296915d89777c7243a9b7ff373d8`.

## Non-goals

- No biometric authentication, authorization, liveness, fraud, or synthetic-
  audio claim; no voice-only identity authority.
- No real reference registration without a separately reviewed, exact
  biometric-enrollment manifest and apply authorization. Existing gold,
  calendar/contact confirmation, transcript inference, or model output is not
  enrollment consent.
- No raw embedding, centroid, model tensor, enrollment audio, name, email,
  transcript text, or provider-native object in portable receipts, prompts,
  Graphiti, or external systems.
- No Plan 0036 prediction reveal, default pipeline integration, App Intelligence
  call, historical reprocessing, automatic speaker confirmation, or external
  provider write.
- No NVIDIA TitaNet acquisition unless the initial three-model comparison leaves
  one specific recorded quality or deployment question unanswered.

## Current state

P0 and P1 are closed with frozen contracts, a private 24-recording
conversation-disjoint corpus, immutable PCM derivatives, and replayable quality
evidence. P2 is closed at development-only preparation shadow maturity: all
five methods completed 15/15 attempts on the three-recording development slice,
but downstream verification and method selection were deliberately not run. P3
is closed at synthetic reference-authority shadow maturity with exact P1/P2
lineage, CAS lifecycle, and independent descendant invalidation; it contains no
real enrollment or embedding.

The current environment has PyTorch `2.11.0`, torchaudio `2.11.0`, ONNX Runtime
`1.24.4`, and SpeechBrain `1.1.0`. P4A acquired all three exact public model
snapshots plus the pinned WeSpeaker code/terms authorities into the private
runtime. The exact acquisition authority is
`docs/dev/fixtures/plan-0037-p4/verification-model-acquisition-plan.json`.
Its reviewed v1 SHA-256 is
`c6cc78b265eed77b5b52637765dc3cde07a74e99b1ef7fde6328a15ae1345c1c`.

Graphiti was healthy at opening. Its only current P4-specific recall is the
source-backed P2 closeout directing Plan 0037 to the next bounded
downstream/profile packet; current repo and runtime evidence control.

## Authorization and fail-closed gates

- The operator's 2026-07-31 standing Plan 0037 grant authorizes bounded model
  acquisition, package installation/build, terms/contact sharing where needed,
  and development processing. Persisted hashes are audit evidence, not per-run
  authorization rituals.
- That standing acquisition/development grant does not itself create a
  biometric-purpose approval for any person or source segment. Real enrollment
  requires an exact P3 approval object plus a reviewed P4 enrollment manifest
  naming opaque people, source generations, segment hashes, and intended split.
- P4A/P4B may acquire models and prove synthetic interfaces without opening
  private corpus audio. P4C may prepare a no-audio enrollment preview. Audio and
  real profile materialization remain fail-closed until the enrollment manifest
  is explicitly authorized.
- Development/calibration/evaluation split access is serialized. Evaluation
  predictions and gold stay sealed. The standing grant authorizes development
  only. Calibration stays fail-closed until verified development receipts plus
  an exact calibration apply authority are persisted; evaluation stays
  fail-closed until verified development/calibration receipts, the frozen
  decision-policy hash, and exact terminal-evaluation apply authority are
  persisted.
- Any unpinned code/model revision, unresolved checkpoint terms, hash mismatch,
  private-mode failure, stale P3 generation, missing descendant receipt,
  duplicate conversation leakage, or raw-biometric portability attempt blocks
  the affected unit without fallback to a different model.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| P4A design and acquisition | primary plus read-only reviewer | P2, P3 | exact models, terms, revisions, hashes, split gates, and P3/P4 bindings reviewed |
| P4B host seam and synthetic profiles | primary | P4A | three adapters, private profile lifecycle, fake-model trials, and revocation tests pass |
| P4C real enrollment preview/apply | primary | P4B plus explicit enrollment authority | exact approved P3 generations exist or a truthful blocker is persisted |
| P4D development | primary | P4C | all selected models/preparation paths run on the hashed development split and receipts verify |
| P4D2 calibration | primary | P4D plus exact calibration apply authority | hashed calibration split runs, thresholds freeze, and receipts verify |
| P4E sealed evaluation and decision | primary plus read-only reviewer | P4D2 plus frozen decision-policy hash and exact terminal-evaluation apply authority | one replayed terminal decision with exact metrics and no unresolved review blocker |

Intended concurrency is two active agents. The primary owns all code, downloads,
private artifacts, and model/audio execution. Reused reviewer
`/root/p1_review_final` owns read-only design and terminal audits and may inspect
metadata/receipts but not credential values, raw embeddings, or sealed gold
before the terminal gate. Each audit allows one bounded repair-and-rerun cycle;
a residual blocker keeps P4 open.

## Acceptance criteria

- Three real adapters run from exact hash-verified assets on the same private
  conversation-separated trials; deterministic fake adapters cover interface
  and failure behavior.
- Every model-specific profile binds one active P3 generation/hash, exact
  preprocessing/model revisions, private aggregate representation, session and
  window counts, dispersion, lifecycle, and calibration eligibility.
- P3 registration/promotion/invalidation is replayed; stale, superseded,
  withdrawn, deleted, or unacknowledged descendants cannot score.
- P4 supersede/withdraw/delete is fail-closed and ordered; deletion removes
  private biometric bytes, retains only a non-biometric tombstone, and scoring
  checks both active P4 state and live P3 eligibility twice.
- Original and enhanced paths preserve original timestamps and source lineage.
  Enhancement disagreements and quality degradation can trigger abstention.
- Development, calibration, and evaluation conversations do not overlap.
  Their record/conversation-set hashes match the frozen split policy.
  Thresholds are selected only from separately authorized held-out calibration
  evidence, and evaluation cannot unseal without the frozen terminal policy.
- Metrics include exact attempted/success/failure/blocked denominators, false
  acceptance/rejection, EER diagnostic, open-set rejection, abstention,
  calibration error, candidate recall/margin, condition slices, and
  same-person grouping behavior.
- Raw embeddings and model tensors remain private user-scoped files at `0600`
  under `0700` directories and never enter portable receipts or prompts.
- The terminal decision is `select`, `refine`, `reject`, or `stop`; it does not
  enable default integration or automatic confirmation.

## Validation

- Focused profile/materialization, model-adapter, calibration, P3/P4 lineage,
  acoustic-contract, and P1/P2 interface tests.
- Synthetic short/silent/overlap/poor-quality, single-session, multi-session,
  stale-generation, withdrawal/deletion, tamper, NaN/Inf, duplicate-conversation,
  unavailable-model, OOM/timeout, and raw-biometric-leak tests.
- Private manual model smoke with in-memory or direct PCM input; no provider
  decoding dependency.
- Development/calibration/evaluation manifest disjointness and sealed-read audit.
- Joined transcript artifact/store, speaker-evaluation/preprocessing, workflow,
  planning-contract, `python -m py_compile`, `git diff --check`, and full suite.
- Reconcile `/root/p1_review_final` design and terminal reports.

## Terminal condition

Close only after exact hash-verified SpeechBrain ECAPA, WeSpeaker CAM++, and
WeSpeaker ResNet34 profiles run on authorized conversation-separated local
evidence; held-out calibration and sealed evaluation report all required
metrics; lifecycle invalidation and privacy replay; and independent review has
no unresolved blocker. Closure records a model/preparation/threshold decision
but does not itself enable P5 integration or automatic speaker confirmation.

## P4A checkpoint

State: CLOSED; P4B is next.

- Independent design review passed after immutable authority, split-access,
  terminal-decision, and profile-lifecycle repairs.
- Live dry-run/replay bound spec SHA-256
  `c6cc78b265eed77b5b52637765dc3cde07a74e99b1ef7fde6328a15ae1345c1c`
  and dry-run SHA-256
  `6b47964dea6a2caa65a73a23c1561267d1a22f89575a29685465ebada580af8c`.
- Private acquisition manifest SHA-256 is
  `6470ecc8591fd8a40f8d788ba9a3edddc37a508cc54d47800037ab594b957ebe`;
  12 acquired files passed size/hash and `0600` readback under `0700`
  directories. SpeechBrain and ONNX Runtime imports passed. No audio read,
  enrollment, profile, embedding, trial, calibration, or evaluation occurred.
