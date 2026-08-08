# Plan 0061 | Plan 0060 human gold and comparison

State: OPEN

Checkpoint: A0 activated; P1 review preparation ready

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
joined condition evidence, and local transcript-console deep links. It may not
copy raw audio, raw transcripts, provider bodies, model-private features, or
credentials. The page keeps choices client-side and exports an exact decision
block; it has no API write or apply control.

This document is planning authority only. A distinct A0 checkpoint must move
it to `OPEN` before source changes, private review preparation, preview
publication, human-decision capture, or comparison execution.

The planning-only checkpoint was committed and pushed at
`bb0f86cb46e0779483a2b9b493f83d2d8d22adfd`. The operator's `ok go`
instruction then activated A0 after the exact readbacks below.

## Current state

A0 is complete and Plan 0061 is `OPEN` for P1 review preparation. The sealed
Plan 0060 packet still contains exactly 3 recordings, 10 empty decision slots,
and 30 condition views; no review choice is preselected and apply remains
disabled. No source or private review artifact existed at activation. P1 may
now add reusable renderer/parser code and a private minimum-copy worksheet.
P3 human-gold capture remains blocked on a literal complete operator export.

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
| P2 preview gate | P1 | Publish one authenticated Previews session and verify browser usability | All 10 controls render; local deep links and export work; user receives one session URL |
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
- Raw audio, raw transcripts, clips, provider bodies, model-private features,
  and private decision receipts remain under mode-`0700` local runtime roots
  with mode-`0600` files.
- The authenticated Previews copy contains only the minimum review worksheet
  and is a temporary human-review surface. It may link to the existing local
  transcript console but may not embed or duplicate its raw media/transcript.
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
- The worksheet gives one clear local listening/transcript route per recording,
  requires one allowlisted choice per slot, reports completion progress, and
  exports nothing until 10/10 choices exist.
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
- Stop on raw private media/transcript publication or any live identity,
  knowledge, assignment, profile, reference, provider, Graphiti, watcher,
  default, authority, or historical mutation.
- Stop rather than treating a candidate label, email, provider ID, diarization
  label, acoustic subject, or evaluation ID as canonical person authority.

## Local goal bounds

`max_work_unit_attempts: 2`

`max_review_rework_cycles: 1`

`max_hardening_checkpoints: 2`

`checkpoint_interval: 1 completed execution unit`

`authorization_gate: significant_departure_only`

`retry_budget_mode: renewable_execution_window`

`review_discovery_passes: 1`

`review_verification_mode: closed_world`

`review_finding_fields: criterion, evidence, consequence, reproducer, confidence, suggested_disposition`

`review_disposition_values: blocking | nonblocking_backlog | rejected | needs_evidence`
