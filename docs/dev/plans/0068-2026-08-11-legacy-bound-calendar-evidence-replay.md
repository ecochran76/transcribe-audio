# Plan 0068 | Legacy-bound calendar evidence replay

State: OPEN

Active packet: A0 mode-preserving authority freeze.

Checkpoint: Plan 0067 closed `withhold` at terminal `8c11ed7c...` before
writing A0 evidence or changing the product. The original Plan 0066 diagnosis
remains verified: its seven first-failure calendar IDs were explicit,
host-prepared evidence omitted from the second-pass reference catalog. Six
exact model outputs remain privately retained and no new turn is needed.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064-0067

Critical-Path Owner: primary agent

## Current State

Plan 0067 has a replayed zero-effect terminal, the six Plan 0066 outputs and
their original recording filenames remain intact, and no product fix has yet
been applied. A0 is authorized to freeze those inherited artifacts under the
uniform legacy-input rule below.

## Scope and vision outcomes

Advance VISION outcomes 2, 3, 4, and 6 by preserving host-prepared calendar
evidence across the identity-evaluation boundary, measuring speaker proposals
against exact human gold, and retaining auditable filenames and provenance.
Current maturity is Level 1: prepared context exists but six outputs are
invalid under a defective host projection. Target maturity is Level 2 only if
the repaired, zero-turn replay yields at least one correct candidate, zero
wrong candidates, zero invalid cases, and complete provenance. This plan does
not advance live observation or knowledge-reuse outcomes 7 or 8.

## Authority correction

Inherited Plan 0064/0066 artifacts are inputs, not newly created Plan 0068
evidence. Each must be a regular non-symlinked file contained beneath its exact
authorized root and match a frozen SHA-256. Their existing modes are evidence
and must not be changed. Every new Plan 0068 directory/file must remain
`0700`/`0600`. Plan 0066 A2 packet lineage is proven by reconstructing
`build_a2_packet` from the frozen A1 and prior packets, not by byte-equating A1
and A2.

## Execution graph

| Packet | Depends on | Outcome | Writes | Terminal |
| --- | --- | --- | --- | --- |
| A0 | activation | Freeze six packet/status/output pairs, original filenames, seven rejected IDs, explicit calendar catalogs, gold, terminal, and code | Private manifest/receipt only | Exact replay or fail closed |
| A1 | A0 | Propagate explicit calendar catalog, group same-event independence, reject calendar-only candidate binding | Product module and tests | Red-to-green focused tests |
| A2 | A1 | Add only the deterministic catalog field to each retained packet and validate exact retained output text | Private case/measurement receipts | Six terminal dispositions, zero turns |
| A3 | A2 | Measure frozen gold and close pass/withhold | Private terminal and tracked closeout | Non-vacuous pass or reason-coded withhold |

Delegation remains `not_spawned`: current system authority prohibits proactive
subagents, and the evidence seam is tightly coupled.

## Bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_policy_revisions`: 0.
- `max_development_cases`: 6 exact inherited cases.
- `max_primary_model_turns`: 0.
- `max_fallback_model_turns`: 0.
- `max_provider_retries`: 0.
- `max_reference_repairs`: 0.
- `max_fresh_retrievals`: 0.
- `max_fresh_evaluation_runs`: 0.
- `review_discovery_passes`: 0; the goal-level pass was consumed by Plan 0066.
- `review_verification_mode`: closed_world.

## Acceptance

- A0 replays exactly and binds six original recording filenames, six complete
  model status/output artifacts, six reconstructed Plan 0066 A2 packets, and
  seven rejected IDs that are members of explicit case-local calendar catalogs.
- Second-pass packets retain the exact first-pass `calendar_evidence` catalog.
- Catalog IDs validate; absent IDs remain rejected. Same-event calendar clues
  share one stable independence group.
- Every `candidate_match` factor set cites a prepared transcript clue; calendar
  evidence remains candidate-only and cannot establish who spoke.
- A2 alters no retained output and adds only the deterministic calendar catalog
  field to each packet. It performs zero turns, retries, repairs, retrievals,
  or fresh evaluations.
- Pass requires at least one correct, zero wrong, zero invalid, and zero
  incomplete-provenance candidates. Zero candidates cannot pass.
- All live/source/store/index/identity/knowledge/biometric/provider/Graphiti/
  external effects remain zero.

Any invalid retained output closes `withhold`; no output repair or retry is
allowed. Joined/residual, fresh evaluation, live apply, and memory publication
remain closed.

## Validation and done

Run focused A0/A1/A2/A3 and preprocessing tests, Plan 0066 regressions, full
pytest, Python compilation, active/goal planning audits, CodeGraph post-edit
readback, `git diff --check`, exact terminal replay, clean commits, push, and
upstream equality. A transcription/DOCX smoke is not applicable unless
transcription or export behavior changes. Done is an immutable pass/withhold
terminal with the product safety contract repaired and no unauthorized effect.
