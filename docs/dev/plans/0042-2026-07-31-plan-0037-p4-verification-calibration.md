# Plan 0042 | Plan 0037 P4 verification and calibration

State: OPEN

Lane: P10

Plan Version: 6

Parent: Plan 0037 P4

Owner: primary agent

Expected Write Surface: one focused acoustic-verification module and tests,
one narrow P3 descendant-invalidation request seam, one pinned acquisition
fixture, Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`;
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

P4B now has three offline, lazy host adapters; deterministic fake-model
materialization/scoring; a private model-specific profile store; P3 descendant
registration/promotion/invalidation; and staged, active, superseded, withdrawn,
and deleted lifecycle replay. Real adapters have run only on generated
synthetic waveforms. No real reference generation or corpus audio was opened.

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

## P4B checkpoint

State: CLOSED; the read-only checkpoint re-audit passed and P4C preview is next.

- SpeechBrain ECAPA-TDNN, WeSpeaker CAM++, and WeSpeaker ResNet34 implement one
  host adapter contract. Load is lazy and SpeechBrain overrides its upstream
  remote `pretrained_path`, so an `HF_HUB_OFFLINE=1` synthetic smoke loaded all
  three acquired snapshots and returned finite L2-normalized vectors of 192,
  512, and 256 dimensions. Immediately before construction, every adapter
  replays the exact P4A manifest/spec/false-gate/runtime bindings and verifies
  every candidate file's path, size, and SHA-256.
- Private profiles bind the replay-validated P3 profile/generation/hash, exact
  model and preprocessing revisions, window/session counts, dispersion, and a
  `0600` aggregate under a `0700` tree. Portable profile/trial receipts contain
  hashes and opaque IDs, never raw vectors, tensors, names, email, transcript
  text, or waveform content.
- P4B materialization is mechanically synthetic-only: every P3 source must
  carry the synthetic-fixture authority and preprocessing must exactly match
  `{method_id: synthetic_raw, revision: <opaque>}`. Real P3 generations cannot
  reach this API; P4C remains responsible for a separate manifest-authorized
  real-enrollment entry point.
- Materialization is ordered `stage -> P3 register -> P3 promote -> P4 active`.
  Scoring replays the current content-addressed P4 lifecycle receipt and checks
  live P3 descendant eligibility both before and after model execution.
- Added the narrow P3 request needed for P4-owned invalidation without changing
  an otherwise eligible parent reference. Supersede and withdraw disable P4
  first, then request and acknowledge P3 invalidation. Delete requires that
  acknowledgment, verifies the descendant remains ineligible, removes private
  aggregate bytes, and retains only a non-biometric tombstone and audit hashes.
- The immutable profile manifest binds P3 lineage, model/preprocessing,
  artifact path/hash, vector dimension, counts, dispersion, window hashes, and
  opaque sessions. Lifecycle receipts bind that manifest hash. Transition
  retries reuse the persisted lifecycle receipt and P3's original request time,
  making withdraw/supersede and partial-ack recovery deterministic.
- Real P3/P4 synthetic smoke activated and scored all three acquired adapters:
  SpeechBrain `0.892918`, CAM++ `0.994622`, ResNet34 `0.995596`. These synthetic
  scores prove execution and lifecycle wiring only; they are not quality,
  threshold, enrollment, or identity evidence.
- Validation: 47 focused P3/P4 tests, 101 joined acoustic tests, 514 full tests,
  offline three-adapter smoke, `py_compile`, and `git diff --check` passed.
- The independent read-only checkpoint re-audit returned `PASS`: exact P4A
  replay, the synthetic-only P4B gate, immutable metadata coverage, and
  deterministic invalidation retry semantics all satisfied the repaired
  acceptance criteria.

## P4C preview checkpoint

State: CLOSED with a truthful blocker; real enrollment apply remains gated.

- Added a private, content-addressed
  `transcribe-audio.biometric-enrollment-preview.v1` builder and replay seam.
  A ready preview is development-only and must bind requested opaque people,
  exact replay-eligible production P3 profile/generation/source-set/approval
  hashes, every source-segment and lineage-receipt hash, and all three pinned
  model revisions. It also replays the exact frozen split-policy and parent-P0
  manifest hashes and proves every recording/conversation pair is a member of
  the hashed development set.
- Synthetic P3 fixture authority, missing production lineage, duplicate or
  non-opaque people, and calibration/evaluation scope fail closed. Preview and
  replay set every audio, embedding, registration, trial, and external-write
  flag false; they cannot invoke the P4B synthetic materializer or create an
  apply authority.
- The live no-audio preview found no canonical P3 reference store and no
  requested opaque people. It therefore persisted `status=blocked` with
  `p3_reference_store_unavailable` and `no_requested_people`, SHA-256
  `30b6f33fb280daa8020fc79fcec4e82fe6c2a8930fc920399f31b0f13ff1e1a3`,
  under the private P4 runtime. No smoke-only P3 reference was promoted into
  real enrollment.
- Build and replay share one strict semantic validator. Forged split labels,
  model inventories, status/reason combinations, P3 unit/source shapes, and
  out-of-development recording/conversation pairs fail even when their forged
  JSON is itself content-addressed.
- Validation: 56 focused P3/P4 tests and 523 full repository tests passed;
  `py_compile` and `git diff --check` passed.
- The independent read-only checkpoint re-audit returned `PASS` after split
  membership, strict semantic replay, and exact reason/fact consistency were
  verified.
- P4D development remains dependency-blocked until an exact reviewed real
  enrollment manifest names approved P3 generations and an explicit biometric
  enrollment apply authority permits audio access and profile materialization.

## P4C candidate-proposal checkpoint

State: CLOSED; exact private candidates are ready for operator review while
real enrollment apply remains gated.

- Added a content-addressed, metadata-only
  `transcribe-audio.biometric-enrollment-candidate-proposal.v2` builder and
  semantic replay. It consumes the exact frozen development split/P0 corpus,
  the hash-verified P2 v5 joined receipt, replay-validated no-enhancement
  lineage, frozen operator-gold person rows, and Pyannote Community-1
  speech/overlap/change metadata. Timestamp candidates come from either the
  exact reviewed artifact or a committed metadata-only continuity authority
  at SHA-256
  `4c952608568edea918265f0851e89f4abfec2f41ac3faf590aaca20cb10da868`.
  That authority independently binds the frozen campaign, blind prediction,
  completed run ledger, prompt, status, and clue packet before field-for-field
  matching against the current artifact.
- Candidate windows are selected without opening audio: transcript millisecond
  bounds are intersected with P2 speech regions, overlap and speaker-change
  regions are removed, windows are bounded to 0.75-15 seconds and three per
  conversation after every same-person label is grouped, and candidates
  require at least two conversations. Existing gold supplies candidate
  identity evidence only; the proposal explicitly sets biometric authorization
  false and requires a separate exact apply manifest.
- Live semantic replay recovered the two raw-file hash drifts through the
  committed reviewed-clue authority. All three recordings are eligible and the
  proposal is `ready_for_operator_review` with two opaque candidate people,
  five sessions, 15 windows, and 180.755531 selected seconds.
  Canonical proposal SHA-256 is
  `aaec42150a2cc9f81212b7d965682a220202a71af3ad203fae0d7f122c6583a4`;
  the private artifact is `0600`. No transcript text, audio, name/email,
  embedding/vector, P3 mutation, model inference, trial, or external write was
  persisted or performed.
- The next human evidence gate is review of the exact opaque candidate and
  source-set hashes for biometric enrollment. The canonical P3 store remains
  absent and P4D remains blocked until that distinct approval and apply packet.
- Validation: 62 focused P3/P4 tests and 529 full repository tests passed;
  `py_compile` and `git diff --check` passed. Negative coverage rejects
  pre-build clue-packet, blind-prediction, and run-ledger drift plus post-build
  replay drift. Independent checkpoint re-audit returned `PASS` and reproduced
  the focused/full validation and exact live replay.
