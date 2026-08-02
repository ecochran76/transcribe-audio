# Plan 0050 | Generation-3 acoustic evaluation

State: OPEN

Lane: P10

Plan Version: 1

Parent: Plan 0049 successor profiles and Plan 0048 terminal STOP

Owner: primary agent

Expected Write Surface: `acoustic_generation3_authority.py`,
`acoustic_generation3_evaluation.py`, focused tests, this plan, `ROADMAP.md`,
and `RUNBOOK.md`; sealed corpus, calibration, review, preparation, trial,
score, metric, and decision artifacts only beneath the user-scoped Plan 0037
runtime root.

## Vision alignment

This packet advances speaker identity at maturity `2 - Shadow`. Current
maturity is two active operator-enrolled subjects with six successor profiles
but no valid terminal evaluation result. Target maturity remains `2 - Shadow`,
with a source-disjoint, replayable Generation-3 result that either selects,
refines, rejects, or stops under policy frozen before reveal. Evidence is the
exact cohort and gold lineage, recalibrated successor thresholds, per-unit
20 genuine / 100 impostor / 20 open-set minima, condition slices, exact trials,
scores, metrics, and terminal decision—not service health or file counts.

No result from this packet enables default integration or historical
reprocessing. Those remain separately governed P5/P6 decisions.

## Current state

Generation 2 stopped before audio preparation or model execution. Its two
revealed evaluation recordings contained five opaque subjects with no overlap
with the two active profile subjects, making zero genuine and zero impostor
trials possible for every one of nine model-method units.

Plan 0049 subsequently created two active P3 generations and six active P4
profiles across `speechbrain_ecapa_tdnn`, `wespeaker_campplus`, and
`wespeaker_resnet34`. The two subjects have 10 windows across four training
conversations and 15 windows across five. Every Plan 0049 training source is
excluded from Generation 3 by exact source SHA-256, recording, conversation,
and derivative lineage.

A bounded file-searcher refresh of `Documents/Sound Recordings` found novel,
already-transcribed candidate conversations. Non-acoustic transcript/readout
evidence proposes enrolled and open-set label mappings for review. These are
candidate proposals only until exact per-recording gold is confirmed. Candidate
filenames and paths are private selection leads only; they are not identity
truth and are not committed.

The Generation-1 and Generation-2 evaluation sources are permanently nonblind
and cannot be reused. The Generation-2 evaluator is intentionally STOP-only
and generation-bound; it remains byte-exact. Lower-level P1/P2, P3/P4,
historical-calibration replay, adapter, and scoring seams are reusable through
new Generation-3 authorities.

## Scope

- Freeze a new sealed cohort whose source, recording, conversation, and
  derivative identities are disjoint from every source used for development,
  calibration, any prior evaluation, or any current-generation P3/P4
  enrollment. This includes every split in all prior corpus manifests, all
  Plan 0049 sources/segments/derivatives, current P3 generation manifests, and
  current P4 profile lineage.
- Bind gold for enrolled speakers directly to the active P3 person reference
  IDs. Give open-set speakers new cohort-local opaque subject IDs. Never derive
  a second incompatible subject namespace.
- Require at least seven independent conversations, both enrolled subjects in
  at least two conversations each, at least five total gold subjects, and
  sufficient clean speech to support 20 genuine, 100 impostor, and 20 open-set
  trials per model-method unit.
- Recalibrate the six successor profiles before evaluation reveal using the
  existing held-out calibration sources, only after proving those sources are
  disjoint from Plan 0049 training and Generation-3 evaluation. Freeze the new
  nine thresholds and temperatures without reading Generation-3 gold or audio.
- Preserve the earlier population and condition gates. Before reveal, require
  at least four independent same-person subject/session pairs and freeze the
  exact condition dimensions, measurement algorithms, minimum of two observed
  values per dimension, and zero-missing-row rule. After prediction-blind P1/P2
  and before exact-trial/model execution, require the observed condition
  coverage to pass. The five-subject gate means five distinct evaluation gold
  subjects total; enrolled and open-set subjects both count, while
  mixed/unknown labels do not.
- Freeze an independently audited pre-reveal envelope binding the cohort seal,
  gold commitment, active generations/profiles, recalibration application,
  three-model by three-method matrix, preparation/window policy, minimum
  evidence, condition slices, metrics, and terminal decision policy.
- After authorized reveal, perform a structural denominator preflight before
  P1/P2 or model loading. A failed preflight records terminal `STOP` without
  acoustic execution.
- If preflight passes, run isolated P1/P2 preparation, freeze clean windows,
  and create a separately replayable exact-trial child binding every window,
  profile artifact, candidate, method, threshold, trial ID, gold class, and
  denominator before model execution.
- Execute the full matrix, persist private scores and portable aggregate
  receipts, compute frozen metrics and condition slices, and apply precedence
  `stop`, `reject`, `select`, `refine` without changing thresholds or policy.

## Successor recalibration authority

Before any calibration score, freeze exact calibration source/window
membership and prove it disjoint from current profile training, all evaluation
generations, and the Generation-3 cohort. Bind all six active profile IDs,
generation/profile/artifact/manifest hashes, model revisions and assets,
adapter and preprocessing module hashes, the same three preparation score
methods, window policy, raw-cosine centroid scoring, and aggregation rules.

The authority freezes exactly nine `(candidate_id, method_id)` units. Each
produces one threshold/temperature pair under the historical objective and
denominators: threshold selection and temperature fitting replay the prior
held-out calibration algorithm; ties use its frozen deterministic ordering;
missing denominators, failed units, nonfinite values, or fallback attempts stop
the campaign. Abstention margin remains exactly zero and cannot be calibrated.
The nine pairs, score-matrix hash, calibration membership hash, and complete
profile/model/preprocessing bindings must replay before the evaluation envelope
can be built.

## Per-recording gold provenance

Every selected diarized label receives one immutable private outcome bound to
the exact source hash, transcript hash, recording ID, conversation ID, speaker
label, and opaque speaker-label ID. Allowed outcomes are an active P3 person
reference ID, a cohort-local open-set subject ID, `mixed`, or `unknown`.

An enrolled/open-set identity outcome requires operator confirmation for this
exact recording/label or independently authoritative per-recording
non-acoustic evidence that explicitly binds the same label. Prior confirmation
from another recording, filenames, participant lists, transcript name mentions,
and acoustic/model output cannot substitute. Gold freezes before evaluation
scoring and cannot be revised in this generation.

## Pre-reveal envelope and negative authorization

The envelope binds the cohort/gold commitments; calibration membership,
score-matrix, and nine threshold/temperature pairs; active generations and six
profiles; exact model revisions/assets; adapter, authority, P1, P2, and
preprocessing module hashes; candidate matrix; preparation, window,
trial-construction, score-aggregation, metrics, condition, minimum-evidence,
and terminal-resolution policies; repository commit/module authority; and the
portable privacy projection.

Its action vector explicitly sets reveal authorization true only for the later
reviewed apply, while preparation, window freeze, exact-child construction,
model loading/execution, scoring, metrics, terminal decision, profile mutation,
default integration, and historical reprocessing remain false. Those actions
become eligible only in their sequenced descendants; no missing child result is
interpreted as permission.

## Pre-reveal trial-cap rationale

The historical three-window-per-speaker-per-conversation cap cannot satisfy the
unchanged 100-impostor minimum with only two enrolled profiles. The selected
seven-conversation candidate inventory proposes 10 enrolled
speaker/conversation instances: both enrolled subjects in three conversations,
and one enrolled subject in four more. If exact gold confirms that proposal, a
cap of three permits only 30 known-speaker windows and therefore only 30
different-person trials against the other profile.

Generation 3 therefore freezes an evaluation-only maximum of 12 nonoverlapping
clean windows per speaker per conversation before reveal. Conditional on exact
gold and clean-window availability, its structural maximum is 120 genuine and
120 impostor trials per model-method unit. The 18 proposed remaining diarized
label/conversation instances can structurally supply up to 216 open-set windows
before profile expansion. Minimum/maximum window duration,
original timestamps, overlap/change exclusion, mixed/unknown exclusion, and
the same-window-set rule remain unchanged. The cap cannot change after reveal,
and actual clean-window availability must still pass the 20/100/20 preflight.

## Non-goals

- Do not reuse or reinterpret Generation-1 or Generation-2 revealed audio or
  gold as blind terminal evidence.
- Do not use any source, segment, derivative, or review clip from development,
  current enrollment, Plan 0049 training, prior evaluation, or Generation-3
  evaluation in successor recalibration.
- Do not infer identity from a filename, calendar entry, participant list,
  transcript name mention, or model score. Direct transcript evidence may
  propose a label mapping; each exact recording/label requires the provenance
  defined above.
- Do not modify `acoustic_generation2_evaluation.py` or any historical receipt.
- Do not calibrate, tune, choose margins, remove hard cases, or change window
  rules after Generation-3 reveal.
- Do not promote a default method, expose biometric artifacts, or begin P5/P6.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A plan and inventory | primary | Plans 0048/0049 and live profiles | exact disjoint candidate inventory and reviewed plan |
| B successor recalibration | primary | A plus historical calibration disjointness | nine successor-profile thresholds replay before reveal |
| C cohort/gold authority | primary plus operator | A | exact sealed cohort and confirmed subject bindings |
| D pre-reveal envelope | primary | B and C | independent `PASS`, clean pushed implementation, exact replay |
| E reveal and preflight | primary | D | denominators pass or immutable terminal `STOP` |
| F preparation, conditions, and trials | primary | E pass | P1/P2, frozen condition coverage, windows, and exact-trial child replay |
| G model execution | primary | F | complete nine-unit scores and metrics or fail-closed stop |
| H independent audit/closeout | existing reviewer plus primary | G | audited terminal receipt, tests, commit, push, and live replay |

The critical path is serialized across reveal. The primary owns writes. The
existing reviewer is read-only and may perform one bounded audit followed by
one repair-and-re-audit cycle per authority boundary.

## Delegation receipt

- Agent: `/root/p1_review_final`
- Scope: read-only audit of Plans 0043-0049, live profile state, held-out source
  eligibility, Generation-2 STOP cause, and API reuse.
- Result: `FAIL` for unchanged API reuse. The reviewer confirmed zero subject
  overlap caused the Generation-2 STOP, all six successor profiles are active,
  prior evaluation sources are nonblind, and new Generation-3 authority,
  evaluation, recalibration, and exact-trial seams are required.
- Writes: none.
- Generation-3 cohort authority audit: initial `FAIL` found byte-only
  disjointness, an untruthful applied action projection, and an unbound private
  I/O dependency. The repair added independent semantic dimensions, exact
  applied actions, and dependency replay. Re-audit found and removed the last
  derivative-derived identity fallback. Final result: `PASS`; writes: none.

## Gates and stop conditions

- Stop before cohort freeze on any source/transcript drift, duplicate bytes,
  prior-corpus or training overlap, missing conversation identity, symlink,
  or nonprivate derived artifact.
- Stop before reveal unless both enrolled subjects have confirmed gold in at
  least two independent conversations and at least five gold subjects exist.
- Stop before reveal unless successor recalibration replays for all nine units
  and its inputs are evaluation- and training-disjoint.
- Stop before reveal unless at least four independent same-person
  subject/session pairs exist and the exact prediction-blind condition
  measurement/minimum policy is frozen.
- Stop after P1/P2 and before exact-trial/model execution unless every declared
  condition dimension has at least two observed values and zero missing rows.
- Stop after reveal but before audio execution if any unit cannot structurally
  reach 20 genuine, 100 impostor, or 20 open-set trials.
- Stop before model loading unless the exact-trial child replays full-body and
  binds every input, denominator, profile, candidate, method, and threshold.
- Any incomplete matrix cell, nonfinite score, required missing denominator,
  profile lifecycle drift, module drift, source mutation, or private-artifact
  leakage resolves to global `STOP`.
- Evaluation output cannot change thresholds, margins, features, methods,
  window rules, gold, cohort membership, or policy.

## Acceptance criteria

- A new source-, recording-, conversation-, and derivative-disjoint cohort is
  frozen with at least seven conversations, both enrolled subjects across at
  least two conversations, and at least five total gold subjects.
- Evaluation gold uses the exact active P3 person reference IDs for enrolled
  subjects and cohort-local opaque IDs for open-set subjects.
- Six successor profiles and nine recalibrated thresholds replay before reveal.
- The pre-reveal envelope binds exact repository, module, model, asset, adapter,
  preprocessing, policy, profile, calibration, cohort, gold-commitment, and
  privacy authorities plus an explicit fail-closed negative action vector.
- Structural preflight proves 20/100/20 minima per unit or records a truthful
  pre-audio terminal `STOP`.
- On a passing preflight, every P1/P2 result, clean window, exact trial, score,
  metric, condition slice, and terminal decision is private, content-addressed,
  immutable, and replayable.
- Portable authorities and receipts contain only counts, hashes, reason codes,
  and action flags. They contain no paths, names, subject IDs, gold bodies,
  source membership, device labels, private lineage, transcript text, raw
  audio, clips, embeddings, vectors, or biometric scores.
- Focused/full tests, compilation, `git diff --check`, independent audit,
  clean pushed commits, runtime permissions, and live full-body replay pass.

## Validation

- Adversarial tests cover prior-source/training overlap, subject-namespace
  mismatch, unconfirmed gold, insufficient conversations/subjects/trials,
  calibration/profile drift, post-reveal policy changes, exact-child omission,
  incomplete matrix cells, nonfinite values, and portable privacy leakage.
- Run focused Generation-3, calibration, P1/P2, P3/P4, and scoring tests before
  the complete repository suite.
- Compile new modules, run `git diff --check`, and verify all private runtime
  directories are `0700` and files are `0600`.
- Verify commit state, upstream push state, and installed/live replay state
  separately before closeout.

## Current execution checkpoint

Unit A is in progress. File-searcher completed a bounded refresh of the named
Sound Recordings folder. The first formal preview rejected the former fourth
candidate because its exact source hash was already frozen; it was replaced
with a separately transcribed, source-disjoint conversation. The current exact
seven-conversation preview contains 28 diarized label/conversation instances.
Non-acoustic evidence proposes 10 enrolled and 18 other outcomes for exact
review. Its exclusion authority binds all prior corpus manifests and transcript
derivatives, the active Plan 0049 training intake, both active P3 generations,
and semantic recording/conversation/derivative fingerprints. The current
preview reports zero overlap independently for source, recording, conversation,
and derivative identities. Implementation commit `65733dd` is pushed. Exact
membership authority `generation3-cohort-714fb3cf3f881b8bad6757ed` applied and
replayed full-body with private `0700/0600` modes. Its receipt authorizes only
membership freeze plus private gold-packet construction. The separate gold
preview/apply/replay implementation now independently binds both active P3
subject IDs back to exact Plan 0049 training-label evidence, requires exactly
28 per-recording label outcomes, enforces explicit identity-token evidence,
and passed independent code audit. Commit `43fcced` is pushed upstream-even and
the exact production 28-label preview passed independent no-write reproduction
with matching membership, gold-body, and preview hashes. Gold authority
`generation3-gold-5f60fa794c40c8fa5a2c5cb0` is frozen and idempotently
replayed with private `0700/0600` modes. Its aggregate-only receipt authorizes
successor recalibration construction and keeps evaluation reveal, audio
preparation, windows, exact trials, models, scores, metrics, and decision false.

The successor recalibration pre-score preview/apply/replay implementation now
binds the exact historical 22-window calibration membership, all three prior
evaluation corpus authorities, full active-training replay, the complete six
successor-profile Cartesian inventory, exact model/assets and preprocessing,
and all nine candidate-method units. Its per-unit 44 total, 9 genuine, 35
impostor, and 26 open-set denominators are derived from the frozen window and
active-subject join. Independent audit drove repairs across denominator,
profile-shape, repository, gold/cohort, semantic-namespace, and transcript
lineage validation; final re-audit is `PASS`. Live no-write preview hash is
`930dd537819dbefd2bead697fef3d930c1bb768f9ae8efac59c87fd515ed6ec9`.
Commit `cacf58e` is pushed and exact authority
`generation3-recalibration-99fcabf628404df4940f2be0` is applied and replayed
idempotently. Its manifest hash is
`a87d873e79d1a859d45734e85e1b02524495915126ce02fd57cff499f7046e53`.
Only calibration-model execution is newly authorized; no calibration score has
run, threshold freeze remains false, and evaluation reveal remains false.

The separately self-bound score/threshold executor is committed and pushed at
`7d5b535` without changing any module sealed by the pre-score authority. Its
aggregate-only production preview hash is
`6a6367554826ca76731b68e1b9b99e752268ecd1cb7f01e060ef8c39b69cfeba`.
Execution authority
`39298c74aab4a773945268cd73fbaabccf88e8e3026a4041ea6eeed29b715b4f`
was persisted before model load. The exact three-model by three-method run
completed 396 private trials with 44 total, 9 genuine, 35 impostor, and 26
open-set trials in each unit. Score matrix
`3fb983b06b1984724c2f0e3e3c01f55065ff755e36416260c33fe0f2649201c2`
replays structurally without audio or models. All nine deterministic
threshold/temperature units froze successfully under application
`308f326d3fe9baa175ed32c90df4255a8d4bfc1924c6f925eab490ae2832f4d1`
and threshold-set hash
`a927b0d9752d4b79ec42f5248afd2028db1c44414ff2d733c46c7b01b6d16759`.
Abstention remains exactly zero. The aggregate receipt now authorizes only
pre-reveal-envelope construction; reveal, evaluation preparation/windows,
exact trials, evaluation scoring/metrics/decision, profile mutation, default
integration, and historical reprocessing remain false.

## Exact-trial child binding

The child must bind the pre-reveal parent content hash, revealed-gold authority
hash, preparation receipt hashes, complete window-manifest hash, full candidate
matrix, exact model/profile/preprocessing artifacts, every trial ID and class
membership, all policy hashes, and exact genuine/impostor/open-set denominators
for each of nine units. Replay proves every parent rule is copied unchanged and
that the child introduces no threshold, margin, method, feature, membership,
window, score, metric, or terminal-policy change. Model loading remains false
until this child replays full-body.
