# Plan 0061 | Plan 0060 human gold and comparison

State: CLOSED

Checkpoint: P5 complete; terminal `refine`

Lane: P09

Cross-lane dependency: closed Plan 0060 `review_ready` and P10 acoustic evidence

## Scope

Complete the literal human-decision boundary left by closed Plan 0060 without
reopening or rewriting its receipts. Revalidate the exact sealed
3-recording/10-speaker/30-condition packet and unchanged live baselines; render
one authenticated, non-applying browser worksheet from that packet; accept
exactly one explicit operator decision per speaker; then independently score
context-only, acoustic-only, and combined outcomes on the same denominator.

The Plan 0060 P4 packet remains immutable input authority. This successor may
copy only the minimum review fields needed by the authenticated Previews
session: recording-local speaker references, candidate labels and opaque IDs,
joined condition evidence, and exactly one already-frozen P2A per-speaker WAV
clip for each of the 10 decision slots. It may not copy full source recordings,
raw transcripts, provider bodies, model-private features, or credentials, and
it may not regenerate clips or rerun transcription, diarization, acoustic, or
identity models. The page keeps choices client-side and exports an exact
decision block; it has no API write or apply control.

This document is planning authority only. A distinct A0 checkpoint must move
it to `OPEN` before source changes, private review preparation, preview
publication, human-decision capture, or comparison execution.

The planning-only checkpoint was committed and pushed at
`bb0f86cb46e0779483a2b9b493f83d2d8d22adfd`. The operator's `ok go`
instruction then activated A0 after the exact readbacks below.

## Current state

Plan 0061 is closed at terminal `refine`. The operator returned the exact
13-line export, strict parsing accepted all 10 decisions, and P3 froze the
private immutable human-gold receipt. P4 independently scored all 30 frozen
views and P5 replayed the comparison against the unchanged Plan 0060 packet and
live baselines. No review choice was inferred, repaired, or applied.

The operator's instruction, "I'd prefer to simply listen directly through the
previews pages as we have before", is explicit authority for this narrow media
expansion. It opens one final bounded review-surface cycle. Sessions
`c9c4d5d5fbd0` and `03268b59db56` are superseded for decision work. Direct-audio
session `11aabed660d2` is the sole active review surface. None of the sessions
contains the frozen human-gold receipt, which remains only in the private
Plan 0061 runtime.

## P3-P5 terminal closeout

The exact submission content SHA-256 is
`30b0a3769ff501126b2ba3090cd9ebdd0e8e8359e00dca0867f88309cf91ccd9`.
It contains 3 canonical-person decisions, 7 `not_listed` decisions, and zero
`unresolved` decisions over the complete 10-slot denominator. The private
human-gold manifest SHA-256 is
`e330189d0edb7a4795c87f735cb96030d1afebcc37ca0718629b4801bd3886cb`.

All three conditions abstained on all ten slots. Each therefore achieved 7/7
appropriate abstentions, zero wrong or high-confidence-wrong proposals, 10/10
provenance-complete evaluations, zero provider failures, and zero duplicate-
person forks. Each also missed all 3 canonical-person decisions, so
top-person correctness and enrolled recall are both 0/3. Candidate recall is
3/10 because seven literal decisions establish that the true speaker was not
in the frozen candidate list. Context-only, acoustic-only, and combined deltas
are zero. The comparison content SHA-256 is
`12a45055b7c3e9fc15af0e297af4b4decde67c32603c981642857678c476f4fd`;
its manifest SHA-256 is
`0e27804629ca803e2f4d926d33bfd1c951bc175f23bb9ce3fab39ebb97e721cd`.

The unexpected second candidate was traced to the frozen compatibility-contact
snapshot, not to calendar or transcript-context inference. P2B recorded two
compatibility candidates for every recording, `calendar_candidate_count=0`,
and empty clue IDs. The join then exposed that recording-level set for every
speaker slot. This explains both the irrelevant option and the absence of more
useful calendar/context candidates.

The terminal audit selected `refine`: the candidate pool missed 7/10 reviewed
speakers, every condition missed all three known-person slots, and no condition
improved recall. Terminal manifest SHA-256 is
`565f9646b38fa4e37d05687fe70b0c8456ac23ffa3e055c55250f329ca28e592`.
Replay is exact; directories/files remain `0700`/`0600`; SQLite, identity state,
and both services remain at their frozen baselines; live mutation count is
zero. Implementation authority is commit `8bd4fe4`; focused validation passed
11 tests and the full suite passed 937 tests. Compilation, active planning
audit, CodeGraph readback, diff validation, clean commit, and upstream push
passed.

The seven `not_listed` decisions do not identify seven people and therefore
cannot authorize new biometric profiles or references. A successor must first
bind each useful speaker slot to an explicit named/canonical person using
calendar participants and reviewed context, then obtain a distinct biometric-
purpose enrollment authority. Plan 0061 creates no contact, person, profile,
reference, assignment, provider record, Graphiti memory, or historical run.

## P2 direct-audio closeout

The final review implementation landed through `af0f70b`. The publishable
directory contains only `review.html` and the exact 10 frozen P2A per-speaker
WAV clips; private manifests and receipts remain outside it. Immutable replay
passed with worksheet SHA-256
`0afd218aa74116595d14bd59c7d939ff4b4889608f478d9977060d8d6b25fcf8`,
manifest SHA-256
`5748e66248da8933e05f7aab94ad9cabac17575db715eb77ded5d9b2c43f344a`,
and receipt content SHA-256
`227f1e09070025209e8c8adec773e7024550cc5a346ad699a33cd93202d9c332`.
The bundle has 10 clips totaling 8,319,480 bytes, zero full recordings, zero
raw transcripts, zero decisions, zero preselections, zero apply, and zero live
mutations.

Authenticated Previews session `11aabed660d2` contains directory artifact
`3afb4a96364a`. Browser proof found 10 audio controls, 10 direct WAV fallbacks,
10 blank identity controls, and 30 condition views. All ten media resources
returned HTTP 200 as `audio/x-wav`, reported `readyState=4`, no media error, and
finite durations from 18.690 to 30.779 seconds. A reviewed Range request was
served as the complete 200 response with no `Accept-Ranges`; this is recorded
as actual Previews behavior and is nonblocking for these 0.6-1.0 MB clips.
There were zero page errors and no artifact-scoped POST requests. Session
feedback was empty. The browser session was closed without entering choices.

Focused validation passed 8 tests; the cache-cleared full suite passed 934
tests twice after the final bundle-scope adjustment. Python compilation,
planning audit, CodeGraph health, diff validation, clean commit, and upstream
push passed.

## A0 activation checkpoint

At activation, branch and upstream were exact at
`bb0f86cb46e0779483a2b9b493f83d2d8d22adfd` with a clean worktree. CodeGraph
was healthy with 258 indexed files, 7,890 nodes, and 26,325 edges and returned
no pending-sync warning. Repository policy selection remained
`already-aligned`; the planning audit passed. Graphiti was advisory and
returned no current Plan 0060 P5 authority.

Plan 0060 P4 replay was idempotent and exact: content SHA-256
`6f6bb30f9073ad706c45561bbf56311457f53e714743d4d905469508ecb82320`,
manifest SHA-256
`e4883c01af517ee5db4387bdf01ddebd5d876158f7a05478a0968bab3e2808f4`,
3 recordings, 10 speaker slots, 30 condition views, zero human decisions, zero
preselections, apply disabled, gold unread, and zero live mutations. The Plan
0060 terminal manifest SHA-256 remained
`f0eaac827ba19fc3b8bbd94dbe40b1efa4c525f5d351ba540238524767798a8d`.

Independent live readback remained SQLite quick-check `ok` with 466 documents,
2 contacts, 3 speaker assignments, and zero knowledge-schema tables. Identity
state remained
`64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`.
Both transcript services were active/running with zero restarts. Previews raw
health and authentication configuration passed; the public review surface is
authenticated. No preview session or private artifact was created during A0.

Progress classification: `outcome_progress`; authority classification:
activated inside the minimum-copy private/read-only envelope; accepted finding
ledger: empty; next ready unit: P1 review preparation.

## P1-P2 human-review checkpoint

Reusable review authority and tests landed at `9b935e7`. The renderer validates
the exact P4 denominator, strips candidate email and raw media/transcript,
exposes candidate labels plus condition evidence, initially linked each
recording to the local transcript console, and keeps all choices in browser
memory.
The strict parser accepts only the exact 3 hash headers and 10 ordered,
allowlisted decision rows. Partial, duplicate, stale, out-of-set, or preselected
input fails closed. Focused review/join/API validation passed 16 tests; the full
suite passed 934 tests; compilation, planning audit, and diff check passed.

The first immutable worksheet attempt had worksheet SHA-256
`1c03c0e577cee2a21751a2efdd0c3dfa05f055ad3dcbfc338fb38c8228d64266`.
Browser smoke found that two Python string literals emitted raw newlines into
the inline JavaScript, causing a syntax error: controls could change but the
progress counter stayed 0/10. No decision was submitted or frozen, no preview
was published, and no live state changed. That private attempt is retained as
failed evidence. The second and final bounded attempt fixed the escaping and
added a Node syntax regression check at commit `8f3469e`.

The successful private worksheet has SHA-256
`297212523e319006468023ef54af70178325fcfd5fc4164c7130bb36c3e471bc`,
receipt content SHA-256
`3f1a45860cdddf9633764f6d09860b645a7626d861475f6465a36df5a631f021`,
and manifest SHA-256
`1a998821b7dcc568a1d6dd3b4d4b3b1cfea652616ef0c8d4c3af9f08ca2efe7f`.
Replay is idempotent; private directories are `0700`, files are `0600`; raw
audio, raw transcript, candidate email, human decisions, preselections, apply,
and live mutations are all absent.

The second browser smoke verified 3 local recording links, 10 blank controls,
30 condition headings, disabled incomplete export, 10/10 progress after sample
browser-local choices, a 13-line hash-bound export, enabled copy/download,
zero page errors, zero POST requests, and a clear action that restored 0/10 and
an empty export. The smoke choices were cleared and the browser was closed;
they are not human gold.

The operator's remote review found that the three recording links in
authenticated Previews session `c9c4d5d5fbd0` used
`http://transcripts.localhost` and were unusable off-host. This was accepted as
the single blocking review-surface finding and consumed the one allowed review
rework cycle. No decision was entered and old-session feedback remained empty.

The repair landed at `cf16631`. It uses the existing
`https://transcripts.ecochran.dyndns.org` ingress, which currently redirects
unauthenticated requests to Authelia, and binds that base URL into the private
manifest and receipt. The replacement worksheet SHA-256 is
`1a3fc08833753b992dd4558555b1b4709e83762ae7d4a04e8b850a4928a206bb`;
receipt content SHA-256 is
`34125e1e4fe9716fd6b2a2eafc5857ab9240067c58a3cd2d9479418ca67a69e8`;
manifest SHA-256 is
`36dd1cae886caf16145ca02b1f707129455fd3f49003b91c36cd43f25e4f7014`.
It still contains zero raw audio, raw transcript, preselection, decision, apply,
or live mutation.

Browser verification found 3 exact HTTPS recording deep links, 10 blank
controls, and 30 condition views. Opening the first link preserved its exact
document query and reached the Authelia login; the worksheet produced zero page
errors and zero POST requests. Focused validation passed 8 tests and the
cache-cleared full suite passed 934 tests. Replacement authenticated Previews
session `03268b59db56` contains worksheet artifact `f96d6a27e109` and instructions
`ad57b6bc5ff1`. Session `c9c4d5d5fbd0` is superseded and must not be used.
The replacement is the P2 approval surface, not a decision receipt. P3 cannot
start until the operator returns the complete exported block.

Progress classification: `outcome_progress`; authority classification:
private/authenticated minimum-copy review only; finding ledger: the blocking
JavaScript escape defect and blocking remote-link defect are both
`accepted/fixed`, respectively within the work-unit attempt bound and the one
review-rework-cycle bound; next action: literal operator review or stop.

## Vision outcomes and maturity movement

| Capability | Current | Target | Required evidence |
| --- | --- | --- | --- |
| Human identity review | Level 2 sealed joined packet; 0/10 decisions | Level 2 usable, exact operator review over all 10 slots | Authenticated browser proof, complete decision export, strict parser rejection of partial/stale input |
| Speaker identification | Level 2 joined shadow evidence; no measured gold comparison | Level 2 gold-bound three-condition measurement | One explicit canonical-person/not-listed/unresolved decision per slot and hash-bound comparison receipt |
| Comparison | 30 frozen abstentions, no human gold | Level 2 measured candidate recall, correctness, abstention, unresolved, provenance, failure, and burden metrics | Independent recomputation agrees on the exact 10-speaker denominator |
| Automatic assignment and live knowledge writes | Level 0 | Level 0 unchanged | Every forbidden mutation counter remains zero; no apply path exists |

This advances VISION outcomes 3, 4, 6, 7, and 8 by turning the existing joined
speaker evidence into a reviewable, measurable shadow result while preserving
uncertainty, provenance, and human control. It does not claim or authorize an
operational identity loop.

## Inherited authority and activation requirements

- Plan 0060 activation content SHA-256:
  `08afc1b021a30f2a06f6e45bac88cec1b343def65b4e02261845ddff8667cf77`.
- P3 blinded-join content SHA-256:
  `10e203d34f922b894b18096b3196974d8c0c419509387ec4f90852bb3fbda026`.
- P4 sealed-packet content SHA-256:
  `6f6bb30f9073ad706c45561bbf56311457f53e714743d4d905469508ecb82320`;
  manifest SHA-256:
  `e4883c01af517ee5db4387bdf01ddebd5d876158f7a05478a0968bab3e2808f4`.
- Plan 0060 terminal `review_ready` content SHA-256:
  `396f386300dc9b23ce3882a55b76254cfa496cf599689cb412b449305e4cae96`;
  manifest SHA-256:
  `f0eaac827ba19fc3b8bbd94dbe40b1efa4c525f5d351ba540238524767798a8d`.
- Frozen live baseline: SQLite quick-check `ok`, 466 documents, 2 contacts,
  3 speaker assignments, no knowledge-schema tables, identity-state SHA-256
  `64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`,
  and both transcript services active/running with zero restarts.

A0 must replay or independently re-read all of these. Any drift stops this
plan; it cannot select another cohort, refresh provider evidence, rerun acoustic
models, or substitute a new live snapshot.

## Execution graph

| Unit | Depends on | Outcome | Terminal condition |
| --- | --- | --- | --- |
| A0 activation | User `ok go`, clean upstream-even repo, exact inherited replay, current live readback | Freeze successor authority and private/authenticated review boundary | `OPEN` only if every packet, runtime, identity, privacy, and non-effect binding remains exact |
| P1 review preparation | A0 and exact P4 packet | Build tested client-only worksheet plus strict complete-decision parser | 10 unique slots, allowlisted per-case choices, zero preselection, no network write/apply code |
| P2 preview gate | P1 | Publish one authenticated Previews directory and verify direct listening | All 10 controls render with their exact frozen speaker WAV; page and media load remotely; export works; user receives one session URL |
| P3 human gold | Literal operator export from P2 | Validate and freeze 10/10 immutable decisions | Partial, duplicate, stale, out-of-set, or inferred decisions fail closed |
| P4 comparison | P3 | Score all three frozen conditions and freeze independent metrics | Exact denominator and evaluation hashes; recomputation agrees; all forbidden mutations remain zero |
| P5 terminal audit | P4 | Recheck packet, preview provenance, private modes, runtime, identity state, and metrics | `complete`, `refine`, or `stop` with explicit evidence |

The present execution stops at P2 until the operator supplies the literal
10-line decision export. Preview approval is not a substitute for identity
decisions, and silence is not approval.

Delegation receipt: `not_spawned`. Current runtime policy forbids proactive
subagents, and the primary agent owns all writes and recomputation.

## Data, privacy, and authority boundary

- Reusable code, redacted fixtures, tests, and durable status records may enter
  git; private candidate labels, opaque IDs, decisions, and review output may
  not.
- Full recordings, raw transcripts, provider bodies, model-private features,
  and private decision receipts remain under mode-`0700` local runtime roots
  with mode-`0600` files. The only media exception is the exact set of 10
  frozen P2A per-speaker WAV clips copied into the authenticated temporary
  Previews bundle with source hashes and sizes recorded in its private
  manifest.
- The authenticated Previews copy is a temporary human-review surface
  containing the minimum worksheet and exactly 10 bound per-speaker clips. It
  may not contain a full recording or raw transcript.
- The worksheet performs no POST/PUT/PATCH/DELETE request. Its only output is a
  client-side copied/downloaded exact decision block.
- Live database/schema, assignments, people, contacts, roles, relationships,
  profiles, references, defaults, watchers, providers, Graphiti, and history
  remain read-only and unchanged.

## Acceptance criteria

- A0 proves the exact Plan 0060 P4 and terminal manifests, live counters,
  identity state, service continuity, privacy modes, and zero mutation.
- The renderer exposes every frozen candidate, alternative, contradiction,
  warning, cap, source failure, and abstention for all 10 speaker slots without
  selecting a decision.
- The worksheet gives one direct authenticated audio player and fallback WAV
  link per speaker slot, requires one allowlisted choice per slot, reports
  completion progress, and exports nothing until 10/10 choices exist.
- Browser validation proves 10 audio elements, nonzero media duration for each,
  successful WAV responses with actual range behavior recorded, zero page
  errors, and zero artifact-scoped POST/PUT/PATCH/DELETE requests.
- The strict parser binds the Plan 0060 P4 content and manifest hashes and
  rejects partial, duplicate, unknown, stale, or syntactically ambiguous input.
- The frozen human receipt and comparison are immutable and private, contain no
  apply action, and preserve every negative action.
- Condition metrics cover candidate recall, top-person correctness, enrolled
  recall, precision, wrong and high-confidence-wrong proposals, appropriate
  abstention, unresolved rate, duplicate-person forks, provenance completeness,
  provider failure, and review burden.
- Focused/full tests, browser smoke, compilation, frontend build when affected,
  planning audit, `git diff --check`, upstream equality, and final live readback
  pass before closeout.

## Hard stops

- Stop on any inherited packet, cohort, evaluation, identity, service, privacy,
  or live-counter drift.
- Stop before human-gold capture if the authenticated preview cannot preserve
  the minimum-copy boundary or if any review control is preselected.
- Stop on incomplete, inferred, reused, duplicate, out-of-set, or stale human
  decisions; do not repair them by guessing.
- Stop on any full recording, unbound clip, raw transcript publication, or any live identity,
  knowledge, assignment, profile, reference, provider, Graphiti, watcher,
  default, authority, or historical mutation.
- Stop rather than treating a candidate label, email, provider ID, diarization
  label, acoustic subject, or evaluation ID as canonical person authority.

## Local goal bounds

`max_work_unit_attempts: 2`

`max_review_rework_cycles: 2`

The second cycle exists only because the operator explicitly authorized the
bounded direct-audio replacement after the first remote-link repair proved
insufficient. It is the final review-surface cycle for this plan.

`max_hardening_checkpoints: 2`

`checkpoint_interval: 1 completed execution unit`

`authorization_gate: significant_departure_only`

`retry_budget_mode: renewable_execution_window`

`review_discovery_passes: 1`

`review_verification_mode: closed_world`

`review_finding_fields: criterion, evidence, consequence, reproducer, confidence, suggested_disposition`

`review_disposition_values: blocking | nonblocking_backlog | rejected | needs_evidence`
