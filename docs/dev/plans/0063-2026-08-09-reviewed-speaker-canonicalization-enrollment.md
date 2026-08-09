# Plan 0063 | Canonicalize reviewed speakers and prepare biometric enrollment

State: CLOSED

Checkpoint: exact A1 authority accepted and terminal live apply completed;
six canonical people, nine reviewed slot bindings, one voice/person binding,
five biometric references, fifteen profiles, and twenty-three enrollment
sources now replay exactly with zero rollback and zero unauthorized effects

Lane: P10

Cross-lane dependency: P09 canonical-person knowledge plus closed Plans 0055,
0056, 0057, 0058, 0059, 0060, 0061, and 0062

Critical-Path Owner: primary agent

## Scope

Turn the exact reviewed Plan 0062 human gold into a deduplicated, reviewable
canonical-person binding packet; make available calendar-title evidence citable
by the speaker clue workflow without inventing calendar provenance; qualify
source-bound speech for the genuinely new
reviewed people; and prepare a direct-audio review packet for person grouping,
existing-voice binding, and new-speaker enrollment sources.

The source authority is the immutable Plan 0062 P5 submission
`5c2ca66fbc25689da8838b65d587fb7f3a5be778a2579f756b8f91526756cdea`,
comparison
`372cc17d31c16cdaa4deda47dd8c9fe7cbb057e62f1c6802395fc1dba8d7c84f`,
and terminal
`971c5896eaa595069f0387b5f48e5765c1d83e457478f018380ab30534e1f49c`.
Names, emails, organizations, source audio, decision bodies, biometric values,
and candidate mappings remain private under user-scoped runtime storage.

This plan has two authority stages:

- A0 may change repository code/tests/docs and create immutable private
  reconciliation, audio-feasibility, and review artifacts. It may use read-only
  live state and private copies. It may not create or update live people,
  contacts, assignments, observations, reference generations, profiles, or
  embeddings.
- A1 is a separate significant mutation gate. It requires the exact reviewed
  person-group and source-window hashes, current backup/integrity/state
  evidence, a proven private-copy apply and rollback, and literal operator
  authorization before any live canonical or biometric state changes.

The user's standing `plan and execute` goal is ordinary successor authority
for A0 preparation and implementation. It does not silently satisfy A1 or
widen the all-false mutation boundary inherited from Plan 0062.

## Vision outcomes and maturity movement

| Capability | Current | Target | Required evidence |
| --- | --- | --- | --- |
| Reviewed identity canonicalization | Level 2 human gold names nine of ten slots, but repeated people are still slot-local labels | Level 2 replayable private canonical-person reconciliation | Every named slot maps to one provisional person or one explicit merge decision; the role-only placeholder remains unresolved; exact identifiers and name-only proposals stay distinguishable |
| Calendar identity evidence | Level 1 calendar fields can enter the packet but the clue prompt permits citations only to transcript utterances, and no-calendar recordings require an explicit negative state | Level 2 host-bound, citable calendar clues with absence preserved | Stable calendar evidence IDs are prepared and validated when a calendar event exists; the corrected no-calendar case exposes no calendar candidate and never treats a title as proof that a person spoke |
| Biometric enrollment preparation | Level 2 profiles exist for enrolled subjects; five likely reviewed people have no voice profile | Level 2 reviewed, source-bound enrollment preview | Each proposed new person has replay-validated P1/P2 lineage, qualified speech windows, source-claim conflict checks, and a literal include/exclude decision |
| Existing voice/person binding | One P5 decision explicitly joins an enrolled subject to a contextual identity | Level 2 exact reviewed binding authority | The acoustic subject, contextual external identity, person proposal, source bundles, and review decision remain hash-bound without label-only reconstruction |
| Automatic identity application | Level 0 | Level 0 through A0 | No live people, contacts, assignments, profiles, references, provider records, Graphiti projections, watcher changes, or historical reprocessing |

This advances VISION outcomes 3, 6, 7, and 8. It converts reviewed speaker
labels into reusable person and voice-learning authority while preserving
ambiguity and provenance. It deliberately does not prove Level 3 automatic
speaker assignment or permit routine enrollment without review.

## Current State

At planning freeze, `plan-0037-campaign` is clean and upstream-even at
`8c4ff1a4b1b58f1df7b24405a0992ea61b63bfbc`. Plan 0062 P5 exactly replays ten
decisions: nine named slots and one role-only placeholder. The reviewed names
form six likely person groups. One repeated group has the same external email
in two slots and one member carries the explicit enrolled-voice/context link.
Two further repeated-name groups lack a shared exact external identity and
must remain human-reviewed merge proposals rather than automatic merges.

Four distinct contextual email identities are present and none matches the two
legacy live contacts. The live transcript database passes SQLite
`quick_check=ok`, contains two contacts and three speaker assignments, and its
conversation-knowledge schema remains version 0 in sidecar authority mode.
The acoustic identity-state snapshot is
`64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`.

When present, a calendar title already enters `build_clue_discovery_packet`, but
`build_clue_discovery_prompt` instructs the model to cite only prepared
utterance IDs. The host validator likewise understands transcript clue IDs but
has no calendar-evidence citation contract. This leaves real calendar evidence
underused while requiring the host to preserve an explicit no-calendar state.

Existing biometric-reference creation already requires source segments with
replay-validated P1 derivative or P2 speech-preparation lineage, immutable
quality evidence, non-overlapping source claims, and an explicit dry-run token.
The older real-enrollment apply path is bound to its own historical candidates
and does not authorize these reviewed speakers.

## A0 activation checkpoint

A0 activated at `2026-08-09T15:09:17Z` against exact pushed plan commit
`60079787d501df4b56ca3b3a225918f7dc064bbe`. The private activation content is
`3c84d2eff1469509184dacf9bbcd163a51953100e3396a7f1a54a8bf614a0139`, its
manifest file is
`20b1632b464e131fd7dfe7e207b02c248e3654b7d3646c6e85eb3a63bfc0d13c`, and the
receipt file is
`1972efc48b9888406be62c1c63481a5a3f21c37534e41c54978c4002ce0569e8`.
The private directory is `0700` and both files are `0600`.

Activation replay preserved the exact 10-slot source denominator, recorded
zero live mutations, and left A1 required. A fresh read-only check after the
freeze still reports SQLite `quick_check=ok`, 466 documents, two contacts,
three speaker assignments, zero knowledge tables, identity state
`64e0a7f4...`, and both transcript services active/running with zero restarts.
P1 and P2 may now proceed; live person, assignment, reference, profile, and
embedding changes remain unauthorized.

## P1-P4 execution checkpoint

P1 is implemented and pushed. Available calendar titles, descriptions, attendees, and
secondary matching-calendar titles now receive stable packet-bound evidence
IDs; invented and cross-packet IDs fail closed; calendar-only clues remain
candidate evidence rather than proof of speech. The synthetic title fixture
still verifies correctly spelled candidate preparation, but the operator has
confirmed that document `47ea79857aa1ac2d1d79` had no calendar appointment at
all. Dr. Stefl was identified by listening review, not calendar evidence, and
Michael Forrester was not present. That recording is therefore an explicit
no-calendar negative control rather than a missing-source case. Provider command
execution now has a bounded timeout.

P2 private reconciliation is frozen at content
82a6834165b20e9457536fbbe67e1540a583ee6dd72374296de55e5b6ccf7f05,
manifest file 4843b56e..., and receipt file 06ff9400.... It covers all ten
slots: nine named, one role placeholder, six person proposals, three merge
proposals, one existing voice/person binding, and five new enrollment
candidates. Unselected Plan 0062 biometric display options are not promoted to
bindings. Replay is idempotent and records zero mutation.

P3 replays P2 plus the three exact P1 audio lineages and is frozen at content
99078e24c28cc94727eda8a05147f7cd533def6069f2dea370978d31376bfb1c,
manifest file 5b78daa9..., and receipt file 133f923a.... All five proposed new
people are source-feasible pending review across 26 exact 3-15 second windows.
Every window is tagged as a development/training candidate and all three
source recordings are excluded from future holdouts. Device metadata remains
explicitly unverified; no enrollment is authorized.

The original P4 was frozen at content
bf53f4bff7f50c0ddc73277bc2500f513c19a6bd3004d5361efa73e4018893ac,
manifest file 7f60e0a5..., HTML file a9555394..., and receipt file
b4e82a8e.... It is now superseded because its calendar-source notice asserted
provenance the recording did not have. Its private 0700 review contains 26
mode-0600 WAV clips and 30
blank decisions: three groupings, one enrolled-voice/context binding, and 26
source include/exclude choices. Authenticated remote Previews session
cb529a51053b was browser-validated before supersession: its notice and all
players render; audio reaches ready state 4 with no media error; incomplete
export is refused; complete export contains 30 decision rows plus four exact
headers; clipboard denial selects the full fallback block; and no POST request
occurs. That browser defect was implementation validation, not a semantic
operator review cycle. The single allowed review rework cycle is now consumed
by the operator's no-calendar and participant correction. Corrected v2 P4
content `486dce6804021314565b5b9c21aeeb58b92529e4a4d4f727324c8106f5753a8a`
supersedes the old hash while preserving the unchanged P2/P3 bindings. Its
manifest is `613f660d...`, HTML is `13ce1e2a...`, and receipt is `6f8c91ad...`.
Authenticated Previews session `b18f14803bd7`, artifact `e7ffdb4c90a5`, shows
that Dr. Stefl came from listening review while Michael Forrester is absent
from the recording. Browser proof covers 26/26 finite-duration WAVs, 30 blank
choices, incomplete-export refusal, a 34-line v2 answer block, selected
manual-copy fallback, and zero POST requests. Operator review then exposed a
second blocking defect: each grouping card showed two raw slot IDs without the
two audio samples needed to decide whether they are the same person, and its
placement made the Michael Forrester cross-recording question appear attached
to the separate Dr. Stefl correction. The v2 surface and submission are
therefore superseded, not accepted. Review schema v3 requires six hash-bound
Plan 0062 comparison clips, two visibly labeled samples inside each grouping
card, direct-WAV fallbacks, and a separate no-answer-required Recording 2
context notice. A first prepublication v3 freeze `2ae49329...` was rejected by
strict replay because the new comparison-clips intermediate directory was
0755; it was never published. The repaired immutable v3 review is content
`d782cd7df6805abd0216fb002ac5133d7ca0c5f9825d074e761a702bc219a479`,
manifest `7528a641...`, HTML `54a0f25a...`, and receipt `cdb5633a...`.
Idempotent replay verifies 26 source clips plus six comparison clips, 30 blank
decisions, all 35 files at 0600, every directory at 0700, and zero live
mutations. Authenticated Previews session `4ac17bd09f2f`, artifact
`93bf4884a21e`, serves 32/32 unique WAV URLs with HTTP 200 WAV responses. Remote
HTML readback confirms the separate no-answer notice, the Michael Recording 1
versus Recording 3 pair, v3 submission binding, 30 blank decisions, and no
POST or fetch path. The operator returned the complete v3 block; its immutable
submission content is `937817fb...` and accepts all three merge proposals, the
single voice/person binding, 23 source windows, and excludes three source
windows. A separate operator-requested full-name correction was independently
verified against governed email and calendar context and is bound to its exact
reviewed slot and external identity. Private values remain outside this plan.
All live canonical and biometric changes remain unauthorized pending A1.

## P5 private-rehearsal implementation checkpoint

The deterministic post-review transition and conversation-knowledge rehearsal
are implemented and the one exact production-mode private-copy rehearsal is
complete. A complete
P4 submission resolves accepted merges to one stable canonical person, keeps
rejected merges separate, preserves all nine named speaker-slot bindings and
the reviewed voice/context outcome, and carries only literal included source
windows into at most five enrollment units. A grouping result above that bound
fails closed before rehearsal.

The knowledge rehearsal backs up the live transcript SQLite database to a
private run, migrates only that copy from knowledge schema 0 to 3, writes the
reviewed people, source records, deduplicated external identities, slot and
voice observations through the governed store interfaces, rebuilds current
person profiles, reconciles table counts and hashes, rolls the schema back to
0, and restores the exact baseline bytes.

The biometric rehearsal copies only the governed reference and profile state,
validates every selected file and SQLite table against immutable baseline
inventories, resolves each included source through its exact P1 derivative
lineage, registers one reference generation per reviewed enrollment unit, and
materializes one profile for each of the three standard production adapters.
It then exercises governed profile and reference withdrawal/deletion,
reconciles logical rollback state, restores the exact baseline bytes, and
proves the live reference/profile state unchanged. Custom adapters are allowed
only in explicit test mode and can never make a receipt A1-ready.

Replay rechecks the immutable transition, manifests, receipts, private state
snapshots, and unchanged live snapshots. Transition `75166646...` resolves six
canonical people, nine slot bindings, one active voice/person binding, five
source-feasible enrollment units, 23 included sources, three excluded sources,
and one provider-backed name correction. Complete rehearsal receipt
`7fe33287...` proves one logical apply and rollback across the knowledge and
biometric copies, exact baseline restoration, `a1_request_ready=true`, and zero
live mutations; idempotent replay passes against the current live stores. The
first invocation selected the acquisition subdirectory instead of the governed
profile-store root and stopped before biometric copying; its empty partial
directory was removed after exact inspection. The preflight order now validates
both live biometric roots before creating a run directory. A1 is not authorized.
Eighteen focused transition/rehearsal/A1/live-driver tests and the 999-test full
suite pass after this closeout.

## A1 authority implementation checkpoint

The post-rehearsal A1 request and literal-authorization contract is implemented
without preparing a request or granting authority. A request can be frozen only
after replay of a production-mode complete rehearsal whose exact transition,
review, rehearsal receipt, current live knowledge/reference/profile snapshots,
clean upstream-even repository commit, and committed mutation modules all still
match. Test-mode rehearsal receipts are permanently ineligible.

The request renders one exact five-line answer block binding its request,
transition, and rehearsal hashes plus the literal
`authorize_exact_live_apply` decision. Any changed, missing, duplicated, or
extra field fails closed. A matching response freezes a private authorization
receipt but performs zero mutations. Its requested scope covers only the
reviewed knowledge migration, people and observations, biometric references
and profiles, plus quiescing and restoring the two transcript services needed
for cross-store rollback safety. Provider writes, Graphiti writes, external
writes, and historical reprocessing remain false.

Five focused tests cover exact private request/replay, test-mode rejection,
live-state and repository drift, literal authorization/replay, and altered or
extra answer fields. The combined acoustic/rehearsal/authority set passes 110
tests and the then-current full suite passes 989 tests. No real A1 request
exists yet and no literal A1 authorization has been supplied.

## P5 live-apply implementation checkpoint

The A1-gated one-shot live driver is implemented but has not been invoked. It
accepts only the exact literal authority and unchanged request baseline,
requires both transcript services active/running, then stops only those two
services before reading the final baseline or making a backup. The knowledge
database and selected biometric state are copied byte-for-byte only while
quiesced; the profile backup excludes acquisition and calibration corpora that
the transition cannot mutate.

The driver migrates and writes the reviewed knowledge records through the same
governed store interfaces used by rehearsal, registers each reviewed reference
through the P3 lifecycle, and materializes each standard model profile through
the validated P4 core. It reconciles the exact authorized counts before
restoring both services. A terminal private receipt prevents a second apply and
replays against the resulting state.

Any failure after backup forces both services quiescent, exact-restores all
three selected state surfaces, verifies them against the byte-bound baseline,
restores both services, and freezes a terminal failure receipt that also
prevents retry. Four disposable-store tests cover successful one-shot replay,
injected mid-biometric failure and three-store restore, refusal of custom
production controls, and refusal to aim test mode at production roots. The
combined authority/rehearsal/acoustic set passes 114 tests and the full suite
passes 993 tests. This proves the driver and rollback seam on disposable state;
the one real rehearsal, A1 request, literal A1 authorization, and live apply all
still await the current 30 decisions.

The knowledge and biometric mutation bodies are no longer duplicated between
the two workflows. The private-copy rehearsal and terminal driver both call
`apply_reviewed_knowledge_transition` and
`apply_reviewed_biometric_transition`; consequently, the rehearsal's real
governed-store execution is direct evidence for the exact domain helpers the
driver will invoke after A1. Workflow-specific copy/rollback proof and
A1/service/backup/terminal orchestration remain separate. Ten focused tests and
the unchanged 993-test full suite pass after this consolidation. No real
rehearsal or live mutation occurred during the refactor.

## Terminal A1 and live-apply checkpoint

The operator's explicit `okay go` instruction authorized the exact frozen A1
transition without any further approval round. The literal authority is
content `f5f39a495b332f27246a0eca85621985365ed3ce653ff50c5b6da3e4e4e3cb6b`
and remains scoped to transition
`75166646421378e2fce4aee1e21c35a6d73fdfdbdb5b37297e4c13fc1b8663dc`.

The guarded production driver completed once. Terminal receipt
`259ea605015ecd6b681140e529002c23e131b6e5cada0d1cdd62fc2b151e3dd5`
records six canonical people, nine slot bindings, one voice/person binding,
five references, fifteen model profiles, and twenty-three selected sources.
It records one logical apply, zero rollbacks, one authorized live mutation,
and zero unauthorized effects. Both transcript services were restored to
`active/running` with zero restarts. A second invocation is an idempotent
replay of the terminal receipt and does not mutate state.

This closes Plan 0063. It establishes reusable canonical-person and acoustic
learning state; it does not yet make incoming-conversation identity evaluation
automatic, infer an unrecognized residual speaker by elimination alone, or
write enriched data back to external contact providers. Those outcomes are
owned by Plan 0064.

## Authority and non-goals

- Do not infer that two same-name decisions are the same person. Present a
  merge proposal and preserve separate provisional people until reviewed.
- Do not turn a calendar title, attendee, contact, or acoustic subject into
  proof that the person spoke. Calendar evidence creates a candidate; human or
  policy-qualified identity evidence establishes the binding.
- Do not create a canonical person for the role-only placeholder.
- Do not reuse a review clip, source window, or profile in a future holdout
  without recording it as development/training data and excluding overlap.
- Do not store raw audio, embeddings, biometric values, names, emails, or
  private decision bodies in tracked files, logs, Graphiti, or public previews.
- Do not migrate the live conversation-knowledge schema, create/update live
  contacts or people, apply speaker assignments, materialize profiles, register
  references, write providers, restart watchers, or reprocess history under A0.
- Do not reuse the historical P4C blanket authority or its candidate set for
  these new reviewed identities.

Allowed A0 effects are focused repository modules/tests/docs; immutable private
reconciliation, calendar-clue, lineage, qualification, and direct-audio review
artifacts; read-only live-state receipts; and private shadow database copies.

## Execution bounds

- `max_work_unit_attempts`: 2 per implementation or runtime unit.
- `max_review_rework_cycles`: 1 for the combined grouping/source review;
  consumed by the operator's no-calendar and participant correction.
- `max_calendar_regression_cases`: the exact no-calendar correction plus 3
  redacted controls covering a real title-person hint, attendee conflict, and
  ambiguous title text.
- `max_enrollment_candidates`: 6 provisional reviewed people before grouping;
  no more than 5 new people after accepted grouping and existing-voice binding.
- `max_source_windows_per_person`: 6.
- `max_audio_preparation_methods`: 2 (`no_enhancement` and the already pinned
  speech-region method selected by source qualification).
- `max_private_copy_apply_attempts`: 1 apply plus 1 rollback rehearsal.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after each committed implementation packet, private
  freeze, human gate, and any apply/rollback attempt.
- `review_discovery_passes`: inherited goal-level discovery is already spent;
  verification is closed-world against the worksheet-export repair,
  calendar-title recall, duplicate-person grouping, source lineage, and live
  no-mutation invariants.

Delegation receipt: `not_spawned`. Current system authority forbids proactive
subagents unless the user explicitly requests them. The primary agent owns all
discovery, writes, validation, reconciliation, and final judgment.

## Execution graph

| Unit | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| A0 activation | Committed plan, exact P5 replay, clean upstream-even branch, current live baseline | Freeze source hashes, authority stages, bounds, findings, and negative actions | Plan, ROADMAP, RUNBOOK, private activation receipt | `OPEN` only if P5, live integrity, privacy modes, and all-false mutations remain exact |
| P1 calendar evidence | A0 | Add prepared, stable calendar clue IDs and citation validation to the two-phase workflow | `speaker_identity_preprocess.py`, focused tests | Title, description, and attendees are bounded separately; citations outside the prepared set fail closed; title is candidate evidence, never speaker proof |
| P2 person reconciliation | A0 | Convert P5 decisions into provisional people and explicit grouping/binding proposals | One focused module, tests, private manifest/receipt | 10/10 decisions covered; role placeholder excluded; exact-email and name-only evidence remain distinguishable; no provider/live write |
| P3 enrollment feasibility | P2 and existing P1/P2 acoustic lineage | Resolve source-bound speech windows and quality/source-claim checks for reviewed new-person proposals | One focused module, private manifests/clips/receipts | Every candidate is eligible with exact lineage or reason-coded ineligible; future-holdout exclusion set frozen; no embedding materialization |
| P4 combined human review | P1-P3 | Direct-audio review of grouping, existing-voice binding, and new-person source inclusion | Private minimum-copy worksheet and authenticated Previews artifact | Literal decision for every merge, binding, and source; no-calendar state and participant exclusion shown; no apply endpoint |
| A1 mutation gate | P4 and private-copy apply/rollback proof | Bind one exact reviewed state transition | Private authority receipt only | Literal operator authorization matches exact person/source/apply hashes and current baseline |
| P5 reviewed apply | A1 | Create reviewed person sidecar bindings and approved biometric generations/profiles, then replay | Existing governed stores plus immutable receipts | Exact counts/hashes, backup, integrity, idempotency, rollback, and zero unauthorized effects; otherwise fail closed |

P1 and P2 may proceed independently after A0. P3 waits for provisional person
authority and source lineage. P4 joins all three. A1 and P5 are forbidden until
the review denominator is complete and the private-copy apply/rollback proves
the exact mutation.

## Accepted finding ledger

| ID | Criterion | Evidence | Consequence | Disposition |
| --- | --- | --- | --- | --- |
| F1 | Calendar evidence must be citable only when it exists | P1 synthetic calendar controls plus the operator correction that the clinical recording had no appointment | Treating the listened name as calendar-derived invented provenance and obscured a true no-calendar case | `resolved` in P1 contract; `blocking` for corrected P4 |
| F2 | Repeated reviewed names must not fork or silently merge people | P5 has two name-only repeated groups and one exact-external-identity repeated group | Name-only merge can conflate people; slot-local storage can duplicate them | `blocking` for P2/P4 |
| F3 | Existing voice and context agreement must preserve both authorities | One P5 linked decision carries exact acoustic subject and contextual suggestion | Label-only application would lose the reviewed cross-pillar binding | `blocking` for P2/P4 |
| F4 | New enrollment must use governed source lineage | Existing reference API accepts only replay-validated P1/P2 sources; P4 clips alone are not that authority | Direct clip-to-profile shortcuts would create untraceable training data | `blocking` for P3 |
| F5 | Live mutation is a significant scope expansion | Plan 0062 closed with every mutation false and the knowledge schema is still sidecar version 0 | Applying people/profiles without exact review, backup, and rollback would exceed standing authority | `blocking` for A1/P5 |

## Acceptance Criteria

- The Plan 0062 P3/P4/P5 hashes and 10-slot denominator replay exactly before
  any successor artifact is frozen.
- Calendar title, description, and attendee evidence receive stable prepared
  IDs. Clue Discovery may cite those IDs separately from transcript utterance
  IDs; invented or cross-packet citations fail closed.
- The corrected clinical case records no calendar event, attributes Dr. Stefl
  only to operator listening review, excludes Michael Forrester from that
  recording, and emits no calendar-derived person hint. Synthetic calendar
  controls still prevent title-only speaker assignment.
- Reconciliation covers all ten decisions, excludes the role-only placeholder
  from person creation, and emits deterministic provisional person IDs.
- One exact-external-identity repeated group, two name-only merge proposals,
  four distinct contextual email identities, and the explicit acoustic/context
  binding are preserved without exposing their private values.
- No name-only candidate merge becomes authoritative without a literal review
  decision. Rejected merges remain separate people.
- Each proposed new biometric person has replay-validated source lineage,
  quality evidence, unique non-overlapping source claims, development-split
  labeling, and a frozen future-holdout exclusion.
- The review surface plays every exact source, starts blank, exports a complete
  hash-bound decision block, works through remote Previews ingress, and has no
  POST/apply path.
- A private-copy apply and rollback reconcile every table/file/hash/count before
  A1 can be prepared.
- Until A1, SQLite stays `ok` with two contacts and three speaker assignments,
  identity state stays `64e0a7f4...`, services retain zero restarts, and all
  live mutation counters remain zero.

## Validation

- Focused tests for calendar evidence ID stability, citation allowlists,
  title-only abstention, attendee/title conflict, name correction, exact-email
  grouping, name-only merge proposals, role-placeholder exclusion, linked
  acoustic/context preservation, source-lineage drift, claim overlap, and
  future-holdout exclusion.
- Exact P5 replay and deterministic reconciliation/enrollment preview replay.
- Real browser validation of direct audio, strict incomplete-export refusal,
  the no-calendar correction and participant exclusion, merge/binding choices,
  copy fallback, and no client/network apply.
- Private mode checks (`0700` directories and `0600` files), SQLite
  `quick_check`, identity-state hash, service restart counts, immutable receipt
  hashes, and forbidden-mutation counters.
- Python compilation, full pytest, deterministic active planning audit,
  CodeGraph post-edit readback, `git diff --check`, clean commit/push, and exact
  upstream equality.

## Definition of done

Plan 0063 is complete only when the reviewed Plan 0062 names become an exact,
deduplicated and human-approved provisional person map; calendar-title evidence
is citable and measured without becoming identity proof; every genuinely new
speaker has reviewed, replay-valid enrollment sources or an explicit
ineligibility reason; the existing voice/person agreement is preserved; and
any approved canonical/biometric apply is independently replayed with proven
backup and rollback under A1. Passing tests, preparing clips, or creating
profiles without reusable canonical bindings does not complete the plan.
