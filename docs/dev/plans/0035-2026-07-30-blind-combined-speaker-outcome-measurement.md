# Plan 0035 | Blind combined speaker outcome measurement

State: CLOSED — REFINE

Lane: P09

Plan Version: 1

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

## Vision alignment

This plan advances the speaker-identification, provenance, uncertainty, and
pipeline-yield outcomes in [VISION.md](../../../VISION.md). It measures the
current default combined App Intelligence path on the next frozen
chronological cohort instead of treating provider readiness as product
success.

Current maturity: `2 — Shadow`. The workflow has run on real conversations,
but its unseen quality, validation yield, review burden, and confidence safety
are not established for the current retrieval path.

Target maturity: produce the representative blind evidence needed to accept
or reject advancement toward `3 — Operational`. This plan does not claim
level 3 merely because the run completes.

Measurable effect:

- ten frozen chronological cases receive immutable predictions or explicit
  validation-failure records;
- every eligible reviewed speaker label has an exact scoring denominator;
- calendar association, candidate recall, top identity correctness,
  high-confidence errors, validation yield, and abstention are reported;
- the terminal decision names the next dominant failure class or records that
  the current combined path is ready for a separate automation-policy plan.

## Scope

- Bind the existing conversation-knowledge evaluation freeze
  `evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b` to one compatible private
  speaker-evaluation holdout baseline without opening gold records.
- Add the smallest host-owned interface needed to validate the frozen
  document/hash pairs and create the baseline.
- Use the served default combined speaker-preprocessing workflow through
  Codex app-server on the configured current model route.
- Capture all ten results serially before any reveal.
- Present the ten conversations for independent operator review only after
  prediction capture is complete.
- Reveal and score only after every case has a current post-prediction gold
  review.
- Persist private inputs, outputs, hashes, run references, and comparison
  artifacts under the user-scoped runtime.

## Non-goals

- No prompt, schema, confidence, adapter, retrieval, calendar-matching, or
  candidate-ranking refinement during the blind run.
- No five-family attribution experiment. This plan measures the current
  production-intent combined path; family ablation remains a later diagnostic
  when the combined result requires it.
- No use of model output, calendar attendees, provider contacts, or prior
  predictions as evaluation gold.
- No automatic speaker confirmation, contact merge, CRM mutation, provider
  write, Graphiti write, knowledge-store authority cutover, or external
  deposition.
- No second cohort, target substitution, reviewed-holdout replay, or
  unbounded retry loop.
- No claim about full contextual-readout quality beyond the speaker and
  calendar dimensions measured here.

## Current state

Plan 0029 froze ten unseen chronological cases at ranks 25 through 39. The
freeze contains exact transcript artifact hashes, reports every prediction as
`not_started`, every ground-truth record as `not_reviewed`, and contains no
gold content.

Plans 0030 through 0034 implemented the default immutable-bundle caller,
private shadow projection proof, concrete GWS and Odollo adapters, restored
GWS execution inside the served runtime, and proved included normalized
evidence from GWS and both Odollo scopes. The Plan 0034 terminal state kept
the cohort unconsumed and required a separately authorized successor before
prediction.

The existing `speaker_evaluation_baseline.py` runner already executes the
served two-phase App Intelligence workflow, permits at most one
reference-repair turn per failed phase, captures validated failures as
outcomes, and writes immutable predictions before reveal. Its baseline-start
interface accepts only the older speaker-campaign gold freeze, so it cannot
yet bind the newer conversation-evaluation freeze without a small validated
bridge.

The ten cases do not yet have independent gold reviews. This is intentional:
prediction capture must complete before the operator sees any model outcome,
and the model must never receive the later gold records.

P0 completed in source. The new public interface validates the evaluation
freeze, current manifest and gold-index hashes, case identities, blindness,
and review state before writing one deterministic compatible holdout. Exact
replay is idempotent, drift fails closed, the focused suite passes 22 tests,
and the joined host-safe suite passes 164 tests.

P1 passed and created
`baseline-f77e1874-fbfb-4ff3-87fa-9b57e2de197f` from the exact freeze after a
pushed-source service restart and clean preflight. P2 captured four immutable
predictions. Case 3 first failed because an unquoted FTS term was parsed as a
column; the sole unchanged retry succeeded. Case 5 then failed in the same
class with a different term. The total retry bound was exhausted, so no later
case started.

The plan closed `refine` before gold review or reveal. Four predictions remain
private, six cases remain unstarted, the gold-index hash is unchanged,
automatic confirmation and database authority remain disabled, and external
writes remain zero. The terminal receipt is
`~/.local/state/transcribe-audio/plan-0035/terminal-refine-2dd137bf-2575-4095-a645-0bc8d6d70fe7.json`
(`0600`), SHA-256
`afb1886c9965c6e8ce74fe0db12b45e8b664993bffdd1b9a8c0206fa15e752c9`.
Plan 0036 owns the bounded literal-FTS repair and a new superseding baseline.

## Authority and bounds

Authority order:

1. this plan and its private receipts;
2. `VISION.md` and the exact conversation-evaluation freeze;
3. current source, tests, and served runtime;
4. Plan 0034 terminal receipt;
5. operator-authored append-only gold records;
6. roadmap and runbook; Graphiti remains advisory.

Bounds:

- `max_frozen_cohorts_consumed: 1`;
- `max_cases: 10`;
- `max_primary_model_phases_per_case: 2`;
- `max_reference_repairs_per_failed_phase: 1`;
- `max_case_infrastructure_retries: 1`;
- `max_target_substitutions: 0`;
- `max_prompt_or_policy_changes_after_prediction_start: 0`;
- `max_service_restarts: 1`, only if installed source changes;
- provider access is read-only and external writes remain zero.

Gold bodies, prediction bodies, transcript content, provider payloads, and
private person identities remain outside Git. Repo artifacts may contain only
schemas, tests, aggregate metrics, reason codes, hashes, and sanitized
receipts.

## Execution packets

### P0 | Freeze bridge and public-interface proof

Owner: primary agent

Write surface:

- `conversation_knowledge_evaluation.py`;
- `tests/test_conversation_knowledge_evaluation.py`;
- private preflight receipt;
- documentation required by the resulting interface.

Outcome:

- Add one deep host-owned interface that validates the evaluation freeze,
  campaign manifest identity, exact document/hash pairs, blindness flags, and
  `not_started`/`not_reviewed` states before creating a compatible holdout
  baseline.
- Make exact replay idempotent and conflicting replay fail closed.
- Read gold-index metadata only to prove that the frozen cases remain
  unreviewed. Read no gold body while creating the baseline.

Validation:

- One public-interface behavior test fails before implementation and passes
  after the minimal bridge.
- A second vertical test proves artifact-hash drift or non-blind state is
  rejected without a runtime write.
- Existing campaign and conversation-evaluation tests remain green.

Terminal condition:

- Record `stop` for gold exposure, freeze drift, or a privacy violation.
- Record `refine` if the bridge requires redesigning the campaign or
  prediction schemas.

### P1 | Live preflight and prediction start

Owner: primary agent

Dependency: P0

Outcome:

- Verify the pushed source, clean worktree, service source/PID/restart state,
  Codex app-server readiness, current configured speaker-disambiguation route,
  freeze and Plan 0034 receipt hashes, runtime permissions, and zero
  `knowledge_*` tables.
- Create exactly one private combined holdout baseline from the frozen cohort.

Gate:

- All ten document/hash pairs match the live transcript store.
- The cohort remains unseen and has no current gold records.
- GWS and both Odollo scopes are explicit.
- The default Codex app-server route is ready.

### P2 | Blind combined prediction capture

Owner: primary agent

Dependency: P1

Outcome:

- Run the ten cases serially through the served default combined path.
- Capture every valid prediction or host-validated model failure before any
  gold review.
- Record model route, App Intelligence run references, validation outcome,
  latency, provider-bundle references, and completion hashes.

Gate:

- Ten of ten cases are immutable and accounted for.
- No gold record was read or written during prediction.
- No prompt, threshold, retrieval, or candidate change occurred after the
  first model turn.

Terminal condition:

- Retry one infrastructure-failed case once with unchanged inputs.
- Record `refine` after a second infrastructure failure.
- Record `stop` for a gold leak, accepted invented reference, external write,
  cross-tenant scope, or frozen-input mutation.

### P3 | Independent operator gold

Owner: operator; primary agent prepares and validates review packets

Dependency: complete P2 manifest

Outcome:

- Review each conversation without exposing its captured prediction.
- Append a gold record for every case, retaining duplicate, unknown,
  incomplete, and unscorable dispositions in the cohort.
- Bind the completed gold-index hash before reveal.

Gate:

- Every gold record has a review time after prediction completion.
- Model and provider outputs are not accepted as gold.
- Prediction processes cannot read gold bodies.

Terminal condition:

- This is a consequential human gate. Pause with the goal active until the
  ten independent reviews exist; do not infer or fabricate them.

### P4 | Reveal, score, and decide

Owner: primary agent

Dependency: complete P3 gold index

Outcome:

- Reveal the completed baseline exactly once.
- Report exact case and label denominators, calendar association, candidate
  recall, top correctness, correct-person presence, validation failures,
  high/very-high errors, abstention, grouping findings, and exclusions.
- Classify the dominant residual as transcription/diarization, calendar
  association, candidate generation, retrieval/provenance, reasoning,
  validation, confidence, or review ergonomics.
- Record one terminal decision and its next bounded recommendation.

Terminal decisions:

- `accept`: the current combined path has sufficient measured evidence to
  enter a separate level-3 automation-policy plan.
- `refine`: one dominant bounded failure class is named for a successor.
- `reject`: the combined path is unsuitable and its production-intent
  promotion is withdrawn.
- `stop`: privacy, gold, scope, evidence-integrity, or unexpected-write safety
  failed.

## Critical path and delegation

P0 through P4 are serialized because one frozen cohort, one prediction
baseline, and one reveal gate share the same private authority. No subagent is
spawned: the current collaboration policy does not authorize delegation, and
independent file exploration would duplicate CodeGraph.

## Acceptance criteria

- The exact ten-case evaluation freeze is bound without reading or deriving
  gold.
- The binding rejects manifest, artifact-hash, blindness, prediction-state,
  or ground-truth-state drift.
- Codex app-server is the served default and its effective model route is
  recorded.
- Every case produces one immutable prediction or explicit validation failure
  before review.
- Independent review occurs only after prediction completion.
- Reveal cannot occur until every case has a current post-prediction gold
  record.
- Metrics use exact denominators and retain difficult or excluded cases with
  reason codes.
- Automatic confirmation, database authority, and external writes remain
  disabled regardless of quality.
- The terminal decision states what the result proves and deliberately does
  not prove.

## Validation

- Exact RED and GREEN commands for each bridge behavior.
- Focused conversation-evaluation, campaign, baseline-runner, retrieval,
  workflow, and API tests.
- Python compilation and `git diff --check`.
- Active planning-contract audit.
- Freeze, manifest, Plan 0034 receipt, baseline, prediction-completeness, and
  gold-index hashes.
- Private directory/file permission audit.
- Served source, health, PID, restart, Codex app-server, and effective model
  route readbacks.
- Live `knowledge_*` table count, sidecar authority, automatic-confirmation
  state, and external-write count.
- Focused commits, push verification, and repo/runbook reconciliation.

## Definition of done

Plan 0035 is done when the exact cohort has ten immutable blind outcomes,
ten later independent review decisions, one reveal comparison with exact
denominators, and one terminal decision; repo and runtime authorities agree;
all safety states are explicit; validation passes; and the closeout commit is
pushed.

If the operator review gate is not yet complete, the plan and thread goal
remain open. Prediction completion alone is not success.
