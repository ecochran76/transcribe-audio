# Plan 0011 | P04 Provenance Calibration

State: CLOSED

Lane: P04

Parent Plan: `docs/dev/plans/0004-2026-05-04-matter-routing-contextual-reread.md`

## Scope

Turn P04 provenance quality scoring from source-type heuristics into a repeatable,
reviewable calibration workflow. The workflow should evaluate included and
excluded route/context sources against reviewed expectations, produce sanitized
threshold evidence, and make the active calibration profile visible in route and
contextual reread artifacts.

## Non-Goals

- No unattended external writes, deposition apply, Odoo writes, Drive moves, or
  Graphiti memory writes.
- No raw audio, raw transcript text, private Drive document contents, API keys,
  OAuth tokens, or unredacted meeting artifacts in the repo.
- No broad Drive/Docs content ingestion beyond bounded read-only metadata or
  snippet-limited evidence needed to calibrate selected route sources.
- No P09 console productization except small metadata surfacing needed to expose
  the active calibration profile or calibration warnings.

## Current State

`context_sources.py` now assigns source-type-specific quality profiles and
minimum scores to non-calendar provenance sources. `route_transcript.py` applies
that filter before contextual reread support is selected, and excluded weak
sources remain auditable under `provenance_pack.excluded_sources` with warnings.
Plan 0011 closed with profile `p04-source-quality-v1`, a repo-safe manifest
schema and synthetic fixture, a repeatable evaluator, a private reviewed corpus
under `~/.local/state/transcribe-audio/p04-calibration/manifests/`, and a
sanitized accepted report under
`~/.local/state/transcribe-audio/p04-calibration/reports/2026-05-23-p04-source-quality-v1-reviewed-report.json`.
The accepted corpus evaluated 12 reviewed source decisions across Calendar,
Drive/Docs, Graphiti, and Odollo source families with zero false positives and
zero false negatives. Route decisions and contextual rereads now carry the
active source-quality profile in artifact metadata.

## Calibration Model

- Calibration unit: one source decision for one transcript/readout route, not
  only the whole meeting route.
- Ground truth: reviewed `expected_include` or `expected_exclude` decisions with
  a short rationale, source type, source identifier, and privacy-safe evidence
  label.
- Profile: a named calibration profile such as `p04-source-quality-v1` that
  records source-type thresholds and scoring terms used for a run.
- Output: a sanitized report with include/exclude counts, false-positive and
  false-negative review lists, source-type summaries, and the profile version
  evaluated.

## Work Items

- Done: define a redacted calibration manifest format and a short repo README for it.
  Keep reviewed live manifests under
  `~/.local/state/transcribe-audio/p04-calibration/`; keep only schema,
  synthetic examples, or sanitized summaries in the repo.
- Done: add an evaluation harness,
  `scripts/evaluate_provenance_calibration.py`, that loads calibration manifests,
  calls the existing source-quality functions, and writes a sanitized report.
- Done: seed the initial corpus from reviewed route/context artifacts, including the
  SoyLei/Tempo context-packet path, the recent watcher-ingested conversation once
  contextualized, and enough reviewed source decisions to cover Calendar,
  Drive/Docs metadata, Graphiti advisory sources, and Odollo evidence where
  available.
- Deferred: add a metadata/snippet-only local transcript-store candidate source adapter
  only if calibration shows the local index improves route decisions without
  leaking raw transcript content into durable repo artifacts. The accepted
  corpus did not require this adapter.
- Done: make the active source-quality profile explicit in route/context output
  metadata so later review can tell which threshold set produced a decision.
- Done: tune source-type thresholds in one place and update fixture tests for known
  include/exclude cases, including weak Graphiti/Odollo/Drive examples.
- Done: record a dated sanitized calibration report path in `RUNBOOK.md` when the
  first profile is accepted.

## Acceptance Criteria

- A calibration manifest schema exists, with clear separation between repo-safe
  examples and live reviewed manifests under user-local state.
- The first accepted corpus covers at least twelve reviewed source decisions
  across at least four source families, with both expected includes and expected
  excludes represented where data exists.
- The evaluation report lists false positives and false negatives by source
  type, and gives enough rationale to adjust thresholds without exposing raw
  private content.
- Route and contextual reread artifacts identify the source-quality profile used
  for inclusion/exclusion decisions.
- Known weak Graphiti, Odollo, and Drive/Docs metadata sources stay excluded
  from contextual support unless their reviewed evidence satisfies the calibrated
  threshold.
- No raw transcripts, raw audio, credentials, token files, or unreviewed private
  source contents are added to the repo.

## Validation

- `.venv/bin/python -m pytest tests/test_context_sources.py tests/test_readouts.py -q`
- `python -m py_compile context_sources.py route_transcript.py contextual_reread.py scripts/evaluate_provenance_calibration.py`
- Run the calibration harness against the reviewed local manifest directory and
  write a sanitized report under
  `~/.local/state/transcribe-audio/p04-calibration/reports/`.
- Run a route/contextual reread dry-run for at least one known-good case and one
  weak-source case, confirming included sources, excluded sources, warnings, and
  profile metadata.

## Closure Notes

Closed on 2026-05-23 after profile `p04-source-quality-v1` had a reviewed
corpus, repeatable harness, sanitized report, route/context profile metadata,
and tests protecting selected threshold behavior. Deeper Drive/Docs content
fetch and external apply contracts remain in their own P04/P05 follow-up scope.
