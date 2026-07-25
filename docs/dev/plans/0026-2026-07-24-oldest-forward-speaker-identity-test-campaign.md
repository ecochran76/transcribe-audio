# Plan 0026 | Oldest-Forward Speaker Identity Test Campaign

State: OPEN

Lane: P09

## Scope

Run a chronological, review-backed test campaign for the Plan 0025 speaker
identity workflow. Start with the oldest conversations that Eric can identify
reliably, freeze their operator-reviewed ground truth outside the model prompt,
measure the current algorithm, refine one bounded failure class at a time, and
then work forward through the corpus in chronological batches.

The campaign uses an operational batch size `K`, initially `K=10`. This is a
review-workload bound, not an evidence threshold or statistical claim. Every
campaign manifest records its actual `K`, chronological cursor, eligibility
decisions, algorithm commit, model route, provenance snapshot, and rubric
versions so later runs remain comparable.

Private ground truth, transcript excerpts, provenance evidence, and model
outputs live under the user-scoped runtime, initially:

`~/.local/state/transcribe-audio/speaker-evaluation-campaigns/`

Git retains only schemas, synthetic fixtures, aggregate metrics, failure
taxonomies, and sanitized campaign summaries.

## Non-Goals

- No automatic speaker assignment, external contact mutation, CRM write,
  deposition, or memory harvest.
- No use of calendar attendees as ground truth merely because they were
  invited.
- No disclosure of private participant identities, transcript excerpts,
  mailbox evidence, or tenant records in Git.
- No interpretation of the evidence-strength score as a probability.
- No tuning on a chronological holdout before it has been scored blindly.
- No broad contextual conversation readout; this campaign evaluates calendar
  association, person grouping, speaker identity, and diarization findings.
- No silent exclusion of difficult recordings. Ineligible, duplicate,
  spurious, short, and inaccessible cases remain represented with explicit
  disposition reasons.

## Current State

Plan 0025 is closed and live. It implements Clue Discovery, host-owned
GWS/Odollo retrieval, Identity Evaluation, host-derived `0..100` evidence
scores and bands, append-only `.processing.json` evaluations, and mandatory
review.

The live transcript store currently contains 375 transcript documents spanning
2019 through 2026. Ninety-two transcript rows retain legacy `/mnt/e/...`
source paths. Their copied `stored_path` artifacts are present, but the
selected-conversation preprocessing path currently accepts only an accessible
`source_path`. A read-only request for the second-oldest substantial
conversation returned:

`400 Selected conversation does not have an accessible transcript artifact.`

The campaign therefore cannot begin chronologically until artifact resolution
falls back safely to the stored transcript copy and keeps file, store, and
durable-ID state synchronized.

Only three Plan 0025 processing sidecars exist, each with one evaluation and
no review decision. The older SQLite speaker-assignment surface contains two
confirmed labels for one 2026 conversation and one deferred label for another.
Calendar metadata and filenames provide useful ground-truth leads for older
conversations, but Eric's review is still required before any case is scored.

C1 is implemented on the campaign branch. The read-only preview deterministically
enumerates the live corpus, records explicit dispositions and artifact
availability for all 375 rows, forms 11 exact normalized-transcript duplicate
clusters, and reserves 10 gold-review plus 10 blind-holdout candidates. Its
live manifest starts the cursor at chronological rank 2 because rank 1 is an
explicit incomplete case.

C2 is now implemented and live. Selected-conversation preprocessing resolves
the source artifact first, then accepts only the exact DB-recorded copied
artifact beneath the transcript store whose SHA-256 matches the indexed hash.
Durable-ID backfill synchronizes the selected artifact, copied artifact,
SQLite JSON/hash state, and preserved original source-path provenance. The
second-oldest substantial transcript now opens read-only and prepares Clue
Discovery from its stored copy without sending a prompt or performing an
external write. C3 private gold review is the active unit.

The first thirteen chronological transcript rows already expose useful
campaign strata:

- two one-utterance or sub-250-character artifacts that must be quarantined as
  incomplete while remaining counted;
- repeated recordings around the same event window;
- multi-party interviews with several invitees but fewer diarized labels;
- a canceled calendar event associated with a substantive recording;
- generic calendar titles such as availability blocks or broad labels that
  may be temporally close but contextually wrong;
- two-attendee calls with more diarized labels than people, which are useful
  split/mixed-speaker tests;
- owner identity represented by personal, institutional, imported `.ogcs`,
  group, and alias addresses; and
- historical source paths that are unavailable even though the copied
  transcript artifacts are intact.

## Campaign Principles

### Ground Truth Is Separate And Blind

An operator-reviewed gold record maps each diarized label to one of:

- a reviewed Person Ground Truth identity;
- `mixed`, with reviewed utterance-level notes when feasible;
- `non_person_audio`;
- `unknown_to_reviewer`; or
- `insufficient_transcript`.

It also records the reviewed calendar association as `correct`, `partial`,
`wrong`, `none`, or `uncertain`, plus any reviewed cross-label person grouping.
The model never receives the gold record. Prediction capture completes before
the evaluator opens ground truth for comparison.

Calendar attendees, filenames, contacts, and conversation clues are evidence
for preparing ground truth; none is independently authoritative.

### Chronological Cursor And Cohorts

The enumerator orders unique conversation candidates by recording time, then
stable document ID. It never advances the campaign cursor past an unclassified
row.

Each row receives one disposition:

- `eligible_known`;
- `eligible_unknown`;
- `duplicate_member`;
- `incomplete`;
- `spurious_or_non_conversation`;
- `artifact_unavailable`; or
- `needs_operator_classification`.

`eligible_known` rows fill the next batch of size `K`. Other rows remain in the
campaign manifest and contribute to edge-case counts. The next `K`
`eligible_known` rows are reserved as a blind chronological holdout while the
current batch drives refinement.

The initial seed packet is:

| Chronological rank | Document ID | Initial campaign role | Structural profile |
|---:|---|---|---|
| 1 | `f64009e6c854f8578a6f` | quarantine candidate | one utterance, very short text |
| 2 | `654972c990225cc7b4f8` | gold-review candidate 1 | multi-party interview |
| 3 | `305865c319c6329153e3` | gold-review candidate 2 | multi-party interview |
| 4 | `30cf9becc80c99781e80` | gold-review candidate 3 | canceled-event association |
| 5 | `6e3ee9f759b2aac76c18` | gold-review candidate 4 | two attendees, three labels |
| 6 | `079c3c359ac2c18ac713` | quarantine/duplicate candidate | one utterance, very short text |
| 7 | `4cfebf5bc08a39be6d4a` | gold-review candidate 5 | two attendees, six labels |
| 8 | `61509c7a8d7b30d00f04` | gold-review candidate 6 | generic calendar association |
| 9 | `70377a70bc9ba1988268` | gold-review candidate 7 | multi-party interview |
| 10 | `34f610e61f59ad32ac5d` | gold-review candidate 8 | single-attendee generic event |
| 11 | `8216f2b3fd78cff9a7d7` | gold-review candidate 9 | single-attendee generic event |
| 12 | `40f5232aefe6b4680b77` | gold-review candidate 10 | interview, multiple calendar matches |
| 13 | `c317f856d35c2a763ff8` | first blind holdout candidate | multi-party interview |

These are candidate roles, not asserted ground truth. Eric's review may
reclassify any row, after which the chronological enumerator fills the batch or
holdout from the next row without losing the original disposition. The table
shows only the first holdout candidate; the enumerator continues forward until
the full next `K` eligible-known holdout rows are reserved.

### Immutable Run Provenance

Every baseline or refinement run records:

- campaign, batch, case, conversation, recording, document, evaluation, and
  App Intelligence run IDs;
- chronological rank and duplicate-cluster ID;
- transcript artifact hash and selected artifact location;
- algorithm commit and dirty-tree flag;
- clue, identity, and processing schema versions;
- calendar-association, person-link, and speaker-identity rubric versions;
- provider, model, reasoning settings, prompt hash, and ledger references;
- provenance-config fingerprint, retrieval timestamp, source availability,
  and the bounded evidence-snapshot IDs actually supplied;
- prediction captured-at time and ground-truth reveal time; and
- reviewer identity, review method, correction reason, and supersession links.

Historical mail/contact availability can change. The campaign measures the
algorithm with the evidence available at run time and must not imply that the
same evidence existed at the conversation date. Re-runs use preserved bounded
evidence snapshots when isolating prompt/rubric changes; fresh-retrieval runs
are labeled separately.

## Metrics

### Calendar Association

- reviewed outcome: correct, partial, wrong, none, or uncertain;
- score and band distribution by reviewed outcome;
- count and inspection of High/Very High wrong associations;
- title, time-range, participant, and topic factor agreement;
- correct handling of canceled, generic, overlapping, and multiple-event
  candidates; and
- unresolved coverage instead of forced matching.

Because the score is evidence strength rather than probability, report
band-wise correctness and ordinal separation, not Brier scores or probability
calibration.

### Speaker Identity

- label-level top-proposal correctness;
- correct-person presence anywhere in the candidate set;
- Candidate Match versus Unlisted Person Suggestion correctness;
- appropriate unresolved/conflicting rate;
- High/Very High wrong-proposal count;
- per-person grouping across duplicate source records;
- reviewer correction rate and review time; and
- coverage by calendar-only, transcript-only, single-source, and independent
  multi-source evidence.

### Diarization Interpretation

- pairwise precision/recall for cross-label same-person group proposals;
- reviewed correctness of mixed-speaker findings;
- utterance-level identity coverage where the reviewer can establish it; and
- false grouping or false split cases.

### Evidence And Retrieval

- prepared-reference validation failures;
- attendee-email exact-lookup yield before broader searches;
- candidate recall by GWS/Odollo source and Source Context affinity;
- duplicate-evidence independence violations;
- useful versus misleading evidence factor counts by type;
- provenance/tool warning rates;
- model latency, retrieval latency, prompt size, and provider-call counts; and
- result stability under preserved-evidence replay.

## Failure Taxonomy

Every wrong, unresolved, or corrected result receives at least one primary
failure class:

- `artifact_resolution`;
- `duplicate_conversation`;
- `calendar_time_drift`;
- `calendar_generic_or_canceled`;
- `calendar_candidate_omission`;
- `owner_or_alias_normalization`;
- `contact_person_link`;
- `retrieval_query`;
- `source_affinity`;
- `evidence_independence`;
- `transcript_clue_discovery`;
- `candidate_generation`;
- `unlisted_person_handling`;
- `prompt_reasoning`;
- `rubric_factor_assessment`;
- `host_scoring`;
- `split_label_grouping`;
- `mixed_label_detection`;
- `insufficient_or_spurious_audio`;
- `ground_truth_uncertain`; or
- `campaign_harness`.

Secondary classes and a bounded evidence note may be attached. Reviewer notes
must not copy raw private content into tracked summaries.

## Refinement Loop

1. Freeze the batch, gold records, algorithm commit, rubrics, model route, and
   provenance snapshot policy.
2. Run every eligible case blindly and preserve its evaluation.
3. Reveal gold records and compute aggregate plus case-level results.
4. Cluster failures using the taxonomy and select one bounded hypothesis.
5. Classify the proposed change as artifact/data normalization, retrieval,
   prompt/schema, rubric/scoring, diarization interpretation, or campaign
   harness.
6. Add a synthetic or redacted regression for the failure before changing the
   algorithm.
7. Re-run preserved-evidence cases to isolate algorithm changes. Run a
   separately labeled fresh-retrieval comparison only when retrieval itself
   is the hypothesis.
8. Compare the refined run with the baseline across the complete accumulated
   gold regression set.
9. Accept the refinement only when it fixes or explains its target cases,
   preserves reference validation and no-write invariants, and introduces no
   unexplained regression. Otherwise reject or revise it without overwriting
   the prior run.
10. Score the untouched chronological holdout once. After review, promote it
    into the accumulated regression set and advance the cursor by `K`.

Immediately stop a run and open a repair packet if any of these occur:

- invented references pass host validation;
- an external or speaker-assignment write occurs without review;
- private raw evidence escapes its permitted runtime artifact;
- duplicate source records inflate independent corroboration;
- an archival-path fallback mutates the wrong artifact or loses provenance;
- a gold record is exposed to the prediction prompt; or
- a prior evaluation or review decision is overwritten.

## Initial Refinement Opportunities

The corpus inspection identifies these first hypotheses, in critical-path
order:

1. **Archived artifact resolution.** Resolve an inaccessible original
   `source_path` to the verified copied `stored_path`, record which alias was
   used, and keep durable-ID backfill plus store metadata synchronized.
2. **Conversation deduplication.** Cluster near-identical transcript content,
   overlapping recording windows, and same-event imports before allocating
   chronological batch slots. Preserve every source row as an alias.
3. **Incomplete-artifact handling.** Keep one-utterance and tiny transcripts
   as explicit incomplete/spurious cases instead of letting confident calendar
   evidence masquerade as speaker identity evidence.
4. **Calendar-negative reasoning.** Treat canceled events, generic titles,
   availability blocks, and multiple overlapping events as explicit negative
   or ambiguity factors. Topic agreement must be able to outweigh time-only
   proximity.
5. **Owner and imported-email normalization.** Group personal,
   institutional, group, alias, and `.ogcs` forms without treating them as
   independent people or independent evidence.
6. **Invitee versus speaker separation.** Measure candidate recall without
   assuming every invitee spoke or every speaker was invited.
7. **Split/mixed diarization.** Stress cases where label count exceeds likely
   people count and where one person may span labels.
8. **Temporal evidence drift.** Distinguish preserved-evidence replay from
   fresh current-state provenance so changes in mail/contact systems do not
   look like algorithm improvements.
9. **Calibration harness.** Add deterministic comparison and aggregate
   reporting rather than inferring quality from individual sidecars or
   confidence scores.

## Work Units And Dependencies

The campaign is a serialized critical path. No parallel agent execution is
required for the initial campaign.

| Unit | Outcome | Write surface | Depends on | Terminal evidence |
|---|---|---|---|---|
| C1 | deterministic oldest-forward corpus enumerator and manifest preview | repo scripts/tests; user runtime only on apply | none | preview lists all rows, dispositions, duplicate clusters, and cursor without writes |
| C2 | safe original/stored artifact resolver and synchronized durable IDs | API/artifact/store modules and tests | C1 | oldest stored transcript opens and prepares without original mount |
| C3 | private gold-record schema and operator review surface | repo schema/UI tests; user runtime | C1-C2 | seed rows classified and `K` eligible gold cases frozen |
| C4 | blind baseline executor and comparison report | repo harness; ledgers, sidecars, campaign runtime | C3 | baseline metrics and per-case taxonomy for batch 1 |
| C5 | first bounded refinement packet | smallest implicated modules/tests | C4 | target failure regression fails before and passes after |
| C6 | accumulated regression plus blind holdout | tests and campaign runtime | C5 | no unexplained regression; holdout scored once |
| C7 | repeated chronological batches | campaign runtime; sanitized summaries | C6 | cursor advances until corpus exhaustion or explicit campaign stop |

## Acceptance Criteria

- The campaign begins at the oldest transcript row and accounts for every row
  as eligible, unknown, duplicate, incomplete, spurious, unavailable, or
  pending review.
- Inaccessible original paths do not block a verified copied artifact, and the
  resolver cannot escape the transcript store or silently change provenance.
- Duplicate imports do not consume multiple gold-batch slots or count as
  independent successes.
- Eric-reviewed ground truth is stored privately and cannot enter the model
  prompt or provenance retrieval query.
- The initial batch contains `K` eligible known conversations and the next
  `K` remain blind until the batch refinement decision is frozen.
- Every run is reproducible from its algorithm/rubric/model/config/evidence
  fingerprints and preserves prior runs.
- Calendar, identity, diarization, evidence, retrieval, cost, and reviewer
  workload metrics are reported separately.
- Every corrected or unresolved proposal receives a failure classification.
- Refinements are hypothesis-specific, regression-tested, compared over the
  accumulated gold set, and accepted or rejected explicitly.
- High/Very High errors are inspected individually; confidence bands are never
  presented as probabilities.
- No automatic identity assignment or external write occurs anywhere in the
  campaign.
- A sanitized per-batch summary records improvements, regressions, accepted or
  rejected refinements, and the next chronological cursor.

## Validation

- TDD for chronological ordering, stable cursoring, dispositions, duplicate
  clustering, artifact fallback, store-boundary enforcement, blind gold
  separation, preserved-evidence replay, metric computation, and append-only
  run history.
- Synthetic fixtures for canceled/generic events, owner aliases, imported
  `.ogcs` addresses, missing invitees, extra invitees, unlisted people,
  split labels, mixed labels, duplicate records, and incomplete transcripts.
- Read-only inventory dry run against the live store before any campaign apply.
- Reviewed manifest apply requiring an explicit approval token.
- App Intelligence dry-run inspection proving the gold record is absent from
  prompt and retrieval artifacts.
- Blind baseline and holdout receipts tied to immutable evaluation/run IDs.
- Full backend tests, frontend build when UI changes, `git diff --check`,
  active planning audit, and live API/browser smoke for the oldest accessible
  stored transcript.

## Campaign Stop And Closeout

The campaign stops only when:

- every transcript row through the chosen end cursor has an explicit
  disposition;
- every eligible known case has a preserved blind result and reviewed
  comparison;
- accepted refinements pass the accumulated regression corpus;
- unresolved systemic failures have bounded successor plans; and
- ROADMAP, RUNBOOK, campaign state, tests, commit, push, and live readbacks
  agree.

If Eric pauses the campaign before corpus exhaustion, retain the plan as OPEN
and record the exact chronological cursor, pending gold-review rows, active
algorithm/rubric versions, last accepted refinement, and next executable unit.
