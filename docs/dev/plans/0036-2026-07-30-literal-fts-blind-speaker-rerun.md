# Plan 0036 | Literal FTS blind speaker rerun

State: OPEN

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Vision alignment

This plan continues the speaker-identification and pipeline-yield measurement
from Plan 0035. It removes one proven retrieval robustness blocker and reruns
the exact unseen cohort through a consistent algorithm.

Current maturity: `2 — Shadow`.

Target maturity: complete the blind current-path evidence needed to accept or
reject advancement toward `3 — Operational`. The repair and a green run do not
themselves establish identity quality.

## Scope

- Quote every token produced by the shared FTS prefix-query builder as a
  literal FTS5 phrase before appending the prefix operator.
- Prove hyphenated and underscore-bearing model terms cannot become FTS5
  column or boolean expressions.
- Add an approval-gated supersession parameter to the existing
  evaluation-holdout interface.
- Preserve the partial Plan 0035 baseline without reveal.
- Create one new baseline for the exact same ten document/hash pairs, linked
  to the partial baseline it supersedes.
- Execute, review, reveal, score, and decide under the remaining Plan 0035
  safety and quality contract.

## Non-goals

- No change to token selection, retrieval ranking, scope, provider order,
  temporal policy, evidence budgets, prompt, model, confidence policy, or
  candidate logic.
- No retry or continuation inside the partial Plan 0035 baseline.
- No reading of its prediction bodies for tuning or candidate generation.
- No gold review before the superseding baseline completes.
- No five-family ablation, automatic confirmation, database-authority cutover,
  contact mutation, provider write, Graphiti write, or external deposition.
- No second retrieval repair or second superseding baseline in this plan.

## Current state

Plan 0035 proved the freeze bridge and started the default combined path on
Codex app-server with model `gpt-5.6-sol`. Four cases were captured before the
second infrastructure failure exhausted the run bound.

Both failures originated at
`ConversationEvidenceRepository.search_snapshots`. The shared
`transcript_store.fts_query` function retains hyphens in tokens and emits
unquoted expressions such as `board-member*`. FTS5 interprets the hyphen as an
operator and the following word as a column reference. The observed failures
were `no such column: member` and `no such column: like`.

The service stayed active with `NRestarts=0`. Gold remains absent for all ten
cases. No prediction has been revealed.

P0 and P1 are complete in source. The public evidence-search regression
reproduced `no such column: member` before the repair and passed after every
normalized token was quoted as a literal prefix phrase. The explicit
supersession regression failed before the interface accepted the exact
partial baseline ID, then passed after the minimal linked replacement path.
The focused suite passes 64 tests; the joined host-safe suite passes 194
tests.

P2 is complete. Commit
`fee6ef624e4449f15753074e8c0e292150cfd0b5` is pushed and served by
`transcripts.service`. Superseding baseline
`baseline-65fdc53f-fc1a-4534-a88d-cf4b0563fbcc` captured all ten immutable
blind outcomes with zero infrastructure retries and is linked to the partial
Plan 0035 baseline. Its completed baseline hash is
`6f86a58d74899d0de834a9d03e75585c696e02d4d4fcf8f659f2c11912036cdd`.
All baseline and prediction artifacts are private mode `0600`. The gold-index
hash remains
`6560591461573bf08d50dd110c031d56f287ea570563b9ae0bfdae691d48d3d8`;
no gold or prediction body was read, no prediction was revealed, and no
external write occurred. P3 independent operator review is the active packet.

## Authority and bounds

Authority order:

1. this plan and its private receipts;
2. Plan 0035 terminal receipt and partial baseline;
3. exact conversation-evaluation freeze;
4. current source, tests, and served runtime;
5. later independent operator gold;
6. vision, roadmap, and runbook; Graphiti remains advisory.

Bounds:

- `max_red_green_cycles: 2`;
- `max_source_work_units: 2`:
  one FTS literalization and one explicit baseline supersession;
- `max_service_restarts: 1`;
- `max_superseding_baselines: 1`;
- `max_frozen_cohorts_consumed: 0` additional;
- `max_case_infrastructure_retries: 1` total on the new baseline;
- `max_prompt_or_policy_changes_after_prediction_start: 0`;
- `max_target_substitutions: 0`;
- provider access remains read-only and external writes remain zero.

## Execution packets

### P0 | Literal FTS query

Owner: primary agent

Write surface:

- `transcript_store.py`;
- public-interface tests in the existing transcript/evidence suites.

Outcome:

- Quote each normalized token as an FTS5 literal phrase while retaining prefix
  matching and OR semantics.
- Preserve empty-query behavior and current alphanumeric search behavior.

Validation:

- RED reproduces SQL execution failure for a hyphenated model term.
- GREEN returns the expected scoped snapshot without an operational error.
- Existing transcript-store and knowledge-evidence searches remain green.

### P1 | Explicit partial-baseline supersession

Owner: primary agent

Dependency: P0

Write surface:

- `speaker_evaluation_campaign.py`;
- `tests/test_conversation_knowledge_evaluation.py`.

Outcome:

- Require the exact partial baseline ID before creating a replacement.
- Verify its freeze identity, source hash, partial status, captured count, and
  absence of comparison.
- Link the new baseline with `parent_baseline_id`.
- Return the same replacement on exact replay and fail conflicting
  supersession closed.

### P2 | Push, serve, and blind rerun

Owner: primary agent

Dependency: P0 and P1

Outcome:

- Push the repair, restart the service once, and verify effective source,
  health, route, model, freeze, gold absence, and authority state.
- Start one superseding baseline and capture all ten outcomes serially.
- Preserve every infrastructure and validation failure with exact reason.

Terminal condition:

- Retry one infrastructure-failed case once with unchanged inputs.
- Record `refine` on the next infrastructure failure.
- Record `stop` for gold, privacy, scope, external-write, or frozen-input
  safety violations.

### P3 | Independent review, reveal, and decision

Owner: operator for gold; primary agent for packets, reveal, and scoring

Dependency: complete P2

Outcome:

- Collect ten independent post-prediction reviews without exposing model
  output.
- Reveal once, score exact denominators, and record `accept`, `refine`,
  `reject`, or `stop`.

The goal remains active at the operator gate. No model, calendar match, or
provider record may substitute for the operator's knowledge.

## Critical path and delegation

The repair, supersession, rerun, and reveal are one serialized authority path.
No subagent is spawned under the current collaboration policy.

## Acceptance criteria

- Hyphenated and underscore-bearing query terms execute through SQLite FTS5
  as literals and retain prefix search.
- Ordinary current search tests remain green.
- The partial Plan 0035 baseline is unchanged and unrevealed.
- One new baseline explicitly names the partial baseline it supersedes and
  includes all ten original document/hash pairs.
- Ten outcomes are immutable before any gold review.
- Gold is independent and post-prediction.
- The comparison reports exact identity, calendar, confidence, validation,
  abstention, grouping, retry, and exclusion measures.
- Authority and external-write states remain unchanged.

## Validation

- Exact RED/GREEN commands.
- Transcript store, knowledge evidence, retrieval, campaign, workflow, and API
  suites.
- Python compilation, `git diff --check`, and planning audit.
- Partial and superseding baseline hashes and permissions.
- Freeze, gold-index, model-route, service, and authority readbacks.
- Focused commits and push verification.

## Definition of done

Plan 0036 is done when the FTS defect is fixed and served, one consistent
superseding baseline has ten immutable outcomes, ten later independent gold
reviews exist, reveal and scoring are complete, one terminal decision is
recorded, and repo/runtime authorities agree.
