# Plan 0030 | Provider adapters and blind retrieval evaluation

State: OPEN

Lane: P09

Plan Version: 2

Execution Mode: `/goal`-compatible, bounded, checkpointed

Critical-Path Owner: primary agent

Optimization Bias: balanced wall-clock speed and reconciliation cost

## Goal Contract

Objective:

> Implement production bounded GWS and Odollo evidence-snapshot adapters,
> make immutable scoped retrieval bundles the default speaker Identity
> Evaluation input, prove the private shadow path, and complete the preserved
> five-family blind comparison without changing processing authority or
> enabling automatic confirmation.

The executing goal must keep this objective intact. Passing unit tests,
finishing one adapter, producing empty bundles, or preparing predictions does
not redefine completion.

Terminal success requires one immutable `accept`, `refine`, `reject`, or
`stop` decision after the applicable execution and safety gates below. Every
terminal decision closes Plan 0030. Any further implementation belongs in a
successor plan.

Recommended invocation:

> `/goal Execute docs/dev/plans/0030-2026-07-26-provider-adapters-and-blind-retrieval-evaluation.md from P0 through one immutable terminal decision. Treat Plan Version 2 as execution authority, obey every bound and human gate, preserve the frozen cohort, checkpoint each packet, delegate only the named disjoint lanes, and do not redefine completion around partial implementation or passing tests.`

### Hard execution bounds

- `max_work_unit_attempts: 2`
- `max_review_rework_cycles: 1`
- `max_hardening_checkpoints: 2`
- `checkpoint_interval: after every execution packet or 60 minutes`
- `max_active_agents: 3`, including the primary agent
- `max_subagents_per_lane: 1`
- `max_subagent_depth: 1`; delegated workers may not spawn children
- `max_live_provider_attempts_per_source_scope: 2`
- `max_reference_repair_turns_per_model_phase: 1`
- `max_frozen_cohorts_consumed: 1`

Exhausting a bound is not permission to loosen a gate. Record `refine`,
`reject`, or `stop`, preserve the evidence, open a successor plan only when
the terminal decision justifies one, and close this plan.

### Checkpoint record

Every checkpoint goes in `RUNBOOK.md` and contains:

- plan version and active packet;
- state transition and progress classification: `feature_progress`,
  `verified_blocker`, `bounded_remediation`, `validation_only`, or `terminal`;
- source, test, runtime, and receipt evidence;
- frozen-cohort prediction and gold-visibility states;
- delegation receipt: `spawned` or `not_spawned`, lane, run/session handle,
  terminal status, returned evidence, and reconciliation decision;
- attempts and review-rework cycles consumed;
- current authority modes and external-write state;
- commit, push, and served-runtime state when applicable;
- next packet or exact stop reason.

Readiness, tests, or documentation alone cannot be classified as feature
progress unless they remove a named gate or satisfy an acceptance criterion.

## Authority Order

When sources conflict, use this order:

1. this Plan 0030 version and its immutable checkpoints;
2. the Plan 0029 C7 freeze and decision receipts;
3. `docs/conversation-knowledge-storage-and-retrieval.md` and ADR 0002;
4. `ROADMAP.md`, then `RUNBOOK.md`;
5. repo policies, current source, tests, installed config, and live readbacks;
6. Graphiti memory as advisory discovery only.

The private freeze is
`evaluation-53f5e11d-fee5-51ed-9f8a-aba36834b95b`. Its manifest and gold-index
hashes, case order, artifact hashes, evidence-family names, and `not_started`
prediction states must match the Plan 0029 receipt before any packet proceeds.

## Scope

Complete the bounded refinement selected by Plan 0029 C7. Implement concrete
host-owned GWS and Odollo adapters that emit the versioned evidence-snapshot
contract, make the selected-conversation Identity Evaluation caller consume an
explicit scoped retrieval bundle by default, and run the already frozen unseen
chronological cohort through the five-family blind comparison.

It contains ten cases at chronological ranks 25 through 39. It must not be
regenerated, reordered, replaced, or predicted before adapter, shadow-read,
query-plan, and operator-gold gates pass.

## Non-Goals

- No new storage schema or evidence-bundle redesign.
- No raw provider body in SQLite, model packets, Git, or aggregate receipts.
- No model-controlled provider access.
- No automatic speaker confirmation, contact merge, CRM mutation, external
  write, or database-authority cutover.
- No tuning on the frozen cohort after its predictions are captured.
- No extension beyond one adapter/caller/evaluation slice.

## Current State

Plan 0029 is closed with a `refine` decision. Versioned storage, sidecar shadow
projection, immutable observations and profiles, bounded evidence snapshots,
exact-first hybrid retrieval, immutable bundles, and speaker-review bundle
adaptation all pass in source.

The frozen cohort remains unseen with every prediction state `not_started`.
Production source has only the `HostEvidenceAdapter` protocol; the current
selected-conversation API still calls `collect_configured_identity_evidence`.
The live transcript store has no knowledge-schema tables and sidecars remain
authoritative. The last private retrieval preview produced three valid empty
bundles with eleven calendar candidates and no fabricated provider evidence.

P0 is complete. The immutable private receipt
`preflight-a4fb020d-4bae-5ec3-8fc0-de8f743f34e4` verifies the unchanged ten-case
freeze, matching campaign/gold-index and Plan 0029 decision hashes,
`not_started` predictions, unread gold bodies, three validated configured
source scopes, ready GWS/Odollo executables, zero live knowledge-schema tables,
and sidecar authority. It records zero provider calls, model calls, external
writes, predictions, or gold-body reads.

R1A is complete. The shared `conversation_evidence_adapters.py` boundary now
requires explicit source profile/provider/account/tenant scope, stable provider
record identity, allowlisted capability/source type/metadata, bounded snippets
and metadata, timezone-aware timestamps, deterministic snapshot IDs and hashes,
and fixed failure/warning codes. It rejects raw provider bodies and assigns
`contemporaneous`, `later_retrieved`, or `hindsight` from source-event,
retrieval, and evaluation time. Eighteen focused contract tests and the
existing evidence-store/retrieval suites pass (28 tests total). Private receipt
`r1a-a3e131da-c37f-4830-99be-ab053ea4fe0b` records no provider/model calls,
external writes, predictions, or gold-body reads. R1B and R1C are now eligible
to execute as disjoint provider lanes.

## Revision History

- Version 1, 2026-07-26: bounded successor scope created from Plan 0029's
  `refine` decision.
- Version 2, 2026-07-29: added the `/goal` execution contract, grilling
  decisions, family/candidate/temporal semantics, bounded work graph,
  checkpoint and delegation receipts, neutral review, hard stops, and
  definition of done. The frozen cohort and product scope did not change.

## Stable Decisions From Design Review

These decisions are fixed for this plan:

1. **Partial results stay canonical.** A failed source returns a labeled
   partial immutable bundle. There is no automatic legacy fallback. The legacy
   collector requires an explicit operator action, warning, and receipt.
2. **Adapters normalize; they do not infer.** GWS and Odollo adapters preserve
   bounded provider records, IDs, scope, timestamps, hashes, redaction,
   truncation, and independence groups. Person grouping and identity inference
   remain in the shared retrieval and evaluation layers.
3. **Scope is always explicit.** Every request carries source profile,
   account, and tenant. An intentionally empty account or tenant is retained
   as meaningful data; display labels and runtime defaults cannot supply it.
4. **Later evidence remains distinguishable.** Evidence first retrieved after
   a historical conversation may be used when its source event predates the
   conversation, but it is `later_retrieved`, never `contemporaneous`.
   Results must report with and without later-retrieved evidence.
5. **Undated current contacts are later evidence.** They may generate or
   strengthen candidates through exact provider IDs or verified email, but do
   not establish contemporaneous topic or relationship context without dated
   corroboration.
6. **Retrieval plans are frozen before provider access.** Exact identifiers
   run first, followed by a capped host-approved set of Clue
   Discovery-derived terms. Identity Evaluation cannot launch searches. A
   follow-up search is a new linked request and bundle, never an invisible
   extension.
7. **Exact internal grouping is deterministic and non-mutating.** Source
   records sharing an exact verified normalized email or provider identity may
   share an internal person ID. Name, organization, topic, terminology, or
   role similarity remains a reversible, confidence-scored inference. Neither
   path mutates upstream contacts.
8. **Evaluation gold is independent.** Operator-confirmed identities define
   gold. Calendar attendees, provenance candidates, or model inference cannot
   silently become gold. Unresolved speakers remain unresolved and stay in
   the denominator appropriate to each metric.
9. **Candidate generation and ranking are measured separately.** Each family
   generates its own candidates for recall measurement. After all family
   candidate lists are frozen, every family also ranks the same frozen union
   candidate set to isolate ranking value.

## Evidence-Family Contract

All five families receive only durable case identity, diarization topology,
and the minimum schema needed to return a validated prediction. A
versioned visibility manifest must prove the remaining allowlist:

- `calendar_only`: event title, description, timing, attendee identities,
  response status, and exact attendee-identity registry matches; no utterance
  text, provider snapshots, or accumulated profiles.
- `transcript_only`: utterance text, speaker labels, direct identifiers, and
  Clue Discovery output; no calendar fields, provider snapshots, or
  accumulated profiles.
- `provenance_only`: bounded provider snapshots selected by the frozen
  host-owned query plan; no calendar fields, utterance text, or accumulated
  profiles in the evaluation prompt. Its receipt must state that query terms
  were derived from the transcript so this is not misreported as a
  transcript-independent retrieval baseline.
- `accumulated_history`: reviewed observations and profiles whose supporting
  evidence is permitted by the case `as_of` and temporal policy; no current
  calendar, current transcript text, or live provider snapshots.
- `combined`: all permitted calendar, transcript, provider, and accumulated
  evidence with independence-group de-duplication.

For `provenance_only`, `accumulated_history`, and `combined`, report two
temporal strata:

- strict `as_of` without `later_retrieved`;
- practical accumulated knowledge with labeled `later_retrieved` evidence.

For every family and stratum, report:

- family-specific candidate recall and exact denominator;
- top identity correctness on its own candidate list;
- top identity correctness on the frozen union candidate list;
- correct-person presence;
- High/Very High correct and wrong proposals;
- diarization findings;
- validation yield and failure stage;
- provider yield and partial/unavailable counts;
- latency and packet bytes;
- null plus a reason code for every unmeasurable metric.

## Execution Packets

### P0 | Authority, freeze, and runtime preflight

Owner: primary agent

Required inputs:

- this plan version;
- Plan 0029 C7 freeze and decision receipts;
- current branch, remote, worktree, installed provenance config, and adapter
  runtime readiness.

Write surface:

- one private preflight receipt;
- one `RUNBOOK.md` checkpoint.

Outcome:

- Verify freeze identity, hashes, permissions, ranks, `not_started`
  predictions, and absent gold content.
- Inventory configured GWS and Odollo scopes without exposing credentials.
- Record current source, installed, and live behavior separately.
- Make and record the packet-level delegation decision.

Validation:

- Planning audit passes.
- Freeze and decision hashes recompute.
- No provider, model, or external write occurs.

Terminal condition:

- Proceed only with an unchanged freeze and explicit scopes.
- Record `stop` on gold exposure, freeze drift, or missing authority.
- Record `refine` when a configured source scope cannot be made explicit
  within one bounded remediation without evidence leakage or scope guessing.

### R1A | Shared adapter contract and fixtures

Owner: primary agent

Required inputs:

- `HostEvidenceAdapter`, `ProviderRetrievalRequest`, and
  `EvidenceSnapshotRecord`;
- validated source-scope and temporal contracts.

Expected write surface:

- one shared adapter contract/helper module if needed;
- synthetic adapter fixtures and focused tests;
- sample/config documentation only if a public config contract changes.

Outcome:

- Define one allowlisted normalization boundary shared by provider adapters.
- Reject raw bodies, missing scope, unbounded snippets/metadata, unsupported
  capability, invalid timestamps, and unstable record identities.
- Freeze failure and warning schemas.

Validation:

- Contract tests fail first, then pass.
- Existing evidence-store and retrieval tests remain green.

Terminal condition:

- Record `refine` and split/reframe in a successor if the shared boundary
  requires storage-schema or evidence-bundle redesign.

### R1B | GWS evidence-snapshot adapter

Owner: one bounded implementation worker or primary agent

Dependency: R1A

Expected write surface:

- GWS adapter module;
- GWS-only tests and bounded fixtures.

Outcome:

- Convert permitted People/Contacts, Gmail, Drive, and Calendar results into
  normalized snapshots under explicit source profile/account/tenant and
  capability scope.
- Preserve exact provider identifiers and dated source events.
- Classify undated current contacts as `later_retrieved`.
- Return allowlisted partial failures without switching retrieval paths.

Validation:

- Mocked exact-scope, capability, time, freshness, pagination/budget, raw-body
  rejection, stable-ID, and failure tests.
- Read-only live readiness or bounded retrieval smoke only after mock coverage;
  no provider write.

Terminal condition:

- Complete with at least one bounded success or correctly labeled configured
  failure per exercised GWS scope.

### R1C | Odollo evidence-snapshot adapter

Owner: one bounded implementation worker or primary agent

Dependency: R1A

Expected write surface:

- Odollo adapter module;
- Odollo-only tests and bounded fixtures.

Outcome:

- Convert permitted contacts, leads, and log notes into normalized snapshots
  under explicit source profile/account/tenant and capability scope.
- Preserve the company relationship implied by the tenant without treating it
  as identity proof.
- Classify undated current contacts as `later_retrieved`.
- Return allowlisted partial failures without switching retrieval paths.

Validation:

- Mocked tenant isolation, model/capability, time, freshness, pagination/
  budget, raw-body rejection, stable-ID, and failure tests.
- Read-only live readiness or bounded retrieval smoke only after mock coverage;
  no tenant write.

Terminal condition:

- Complete with at least one bounded success or correctly labeled configured
  failure per exercised Odollo scope.

### J1 | Adapter reconciliation and neutral review

Owner: primary agent integrates; a fresh read-only reviewer challenges the
combined result when runtime capacity is available

Dependencies: R1B and R1C

Write surface:

- integration fixes within one review-rework cycle;
- checkpoint and delegation receipts.

Gate:

- Shared tests plus both provider suites pass.
- Reviewer finds no scope mixing, provider-specific identity inference,
  raw-body persistence, unstable IDs, or automatic legacy fallback.
- Provider results with different source databases preserve all affinities
  without duplicating independent evidence.

Terminal condition:

- One bounded rework cycle, then `refine`, `reject`, or `stop`.

### R2A | Explicit policy and default immutable-bundle caller

Owner: primary agent

Dependencies: J1

Expected write surface:

- provenance-to-retrieval policy builder;
- selected-conversation API/workflow wiring;
- explicit legacy rollback action and receipt;
- API/workflow/config tests and user-facing documentation.

Outcome:

- Build explicit source profile/account/tenant/capability and temporal policy
  from validated user-scoped Source Context.
- Freeze exact-first and capped clue-term query plans before provider calls.
- Make `prepare_identity_evidence(...)` plus its immutable bundle the default
  Identity Evaluation path.
- Keep partial bundles on that path.
- Permit the legacy collector only through an explicit operator action,
  approval token, warning, and durable rollback receipt.

Gate:

- Default receipts contain request/bundle hashes, source failures, warnings,
  included/excluded reasons, freshness, temporal class, independence groups,
  and query-plan identity.
- Identity Evaluation still performs no provider search.
- Exact-reference validation, factor records, confidence calibration,
  split/mixed findings, and human-review gates remain unchanged.
- No automatic assignment, fallback, provider write, or external write.

Terminal condition:

- Caller tests and a private non-frozen smoke prove default and explicit
  rollback behavior without touching the frozen cohort.

### R2B | Private shadow evaluation store and authority proof

Owner: primary agent or one bounded validation worker

Dependency: R1A; joins R2A before J2

Expected write surface:

- private runtime projection/backup/restore/rollback receipts;
- focused projection and authority tests;
- no live database migration.

Outcome:

- Project the frozen cohort inputs into a private shadow store and prove
  artifact/sidecar read agreement, deterministic replay, backup/restore,
  migration rollback, and unchanged sidecar authority.
- Keep raw private evaluation material outside Git with `0700` directories and
  `0600` files.

Gate:

- Sidecar/database identities and counts reconcile.
- Restored and rolled-back copies revalidate.
- Live database still has no authority-mode change.

Terminal condition:

- One failed rehearsal may receive one bounded remediation. A second failure
  records `refine` without consuming the cohort.

### J2 | Integrated readiness gate

Owner: primary agent

Dependencies: R2A and R2B

Required evidence:

- adapter and default-caller receipts;
- private shadow agreement and rollback receipts;
- full focused and host-safe regression results;
- served-runtime proof for the default caller;
- provider-yield coverage by source scope.

Gate:

- No silent fallback or missing source scope.
- Zero-yield sources are unavailable/empty, never negative evidence.
- `provenance_only` or `combined` cannot be labeled measured when their
  required provider/history inputs are absent.
- At least one included provider snapshot must exist before a provenance or
  combined prediction can start; otherwise record `refine`.

Terminal condition:

- Emit a signed/read-only readiness receipt. Prediction remains blocked until
  R3A and R3B also pass.

### R3A | Independent operator gold

Owner: operator; primary agent prepares and validates packets only

Dependency: P0; may proceed alongside R1/R2 but never expose gold to their
workers, model prompts, retrieval query plans, or prediction stores

Write surface:

- private append-only gold records and index;
- no Git content beyond aggregate counts.

Outcome:

- Confirm each scorable speaker identity independently.
- Retain unresolved and unscorable speakers with explicit reasons.
- Freeze the gold-index hash before prediction.

Gate:

- Calendar/provenance candidates are not accepted as gold without independent
  operator confirmation.
- Prediction processes cannot read gold bodies.

Terminal condition:

- Human gate. Pause without consuming predictions until the operator gold
  index is complete and hash-bound.

### R3B | Family visibility, query-plan, and candidate freeze

Owner: primary agent

Dependencies: J2 and R3A

Write surface:

- private visibility manifests, query plans, evidence bundles, per-family
  candidate lists, union candidate lists, and hashes.

Outcome:

- Build each family under the Evidence-Family Contract.
- Freeze family-specific candidates before creating the union.
- Freeze the strict-`as_of` and later-retrieved strata.
- Prove gold absence and prohibited-field absence.

Gate:

- Exact allowlist tests pass for every family.
- Query plans contain exact identifiers first, capped terms, scopes, budgets,
  and no gold-derived terms.
- Duplicate evidence cannot inflate independence groups.

Terminal condition:

- Any gold leak, scope violation, or family contamination records `stop`.

### R3C | Blind prediction capture

Owner: primary agent

Dependencies: R3B

Write surface:

- immutable private App Intelligence inputs, outputs, validation records,
  timing/size receipts, and prediction-completeness manifest.

Outcome:

- Capture every required family, temporal stratum, and candidate-mode
  prediction before reveal.
- Apply at most one existing reference-repair turn per failed phase.
- Do not tune prompts, thresholds, adapters, candidates, or retrieval after
  the first frozen prediction starts.

Gate:

- All predictions and validation failures are accounted for.
- No accepted invented references, unexpected write, or evidence-family drift.

Terminal condition:

- On partial infrastructure failure, retry the affected work unit once without
  changing inputs. A second failure records `refine`.
- A safety violation records `stop`.

### R3D | Reveal, score, and terminal decision

Owner: primary agent; fresh neutral reviewer checks aggregates and leakage

Dependencies: complete R3C prediction manifest

Write surface:

- private per-case scoring;
- sanitized aggregate metrics and immutable decision receipt;
- Plan 0030, `ROADMAP.md`, and `RUNBOOK.md` closeout.

Outcome:

- Reveal gold only after prediction completeness is immutable.
- Score every required family, stratum, and candidate mode with exact
  denominators and reason-coded nulls.
- Record one accept, refine, reject, or stop decision.

Gate:

- Automatic confirmation and database authority remain disabled regardless of
  result; either requires a separate explicit authority plan.
- Decision distinguishes adapter/candidate/retrieval yield, ranking quality,
  confidence safety, validation yield, latency, and packet size.

Terminal condition:

- `accept`: the bounded adapter/default-caller/evaluation path is supported by
  complete evidence; Plan 0030 closes with authority unchanged.
- `refine`: exactly one bounded residual class is named and moved to a
  successor plan; Plan 0030 closes.
- `reject`: default promotion is withdrawn or rolled back with a receipt;
  Plan 0030 closes.
- `stop`: privacy, scope, gold, unexpected-write, or evidence-integrity
  violation; preserve artifacts and close Plan 0030.

## Execution Graph

| From | To | Transition |
|---|---|---|
| P0 | R1A | Freeze and explicit source scopes pass |
| P0 | R3A | Independent operator packets may begin |
| R1A | R1B and R1C | Provider lanes may run in parallel |
| R1A | R2B | Private shadow proof may run without live adapters |
| R1B and R1C | J1 | Both adapter lanes reach terminal evidence |
| J1 | R2A | Integrated adapter contract passes neutral review |
| R2A and R2B | J2 | Default caller and shadow authority proofs join |
| J2 and R3A | R3B | Readiness and independent gold are hash-bound |
| R3B | R3C | Family inputs, candidates, and query plans are frozen |
| R3C | R3D | Every prediction or validation failure is immutable |
| Any gate | terminal decision | Bound exhausted or hard stop reached |

Only R1B/R1C, R2B, and the operator-owned R3A lane are legitimate concurrent
work. R2A remains on the critical path. Collapse a lane back to the primary
agent if its write surface overlaps shared retrieval, workflow, plan, or
runtime authority files.

## Delegation And Reconciliation

- The primary agent owns Plan 0030, shared contracts, integration, runtime
  mutation, frozen-cohort controls, terminal synthesis, and every commit/push.
- R1B and R1C are preferred disjoint implementation lanes. R2B may be a
  bounded validation lane after R1A stabilizes.
- J1 and R3D should use a fresh read-only reviewer when an agent slot is
  available. Reviewer output is evidence, not authority.
- Delegated workers receive no gold bodies, credential material, live-write
  tools, session-management tools, or permission to consume the cohort.
- Each worker has one packet, one owned write surface, no child agents, and a
  stop condition. Its completion receipt includes status, result, notes,
  run/session handle, timestamps, and available model/token metadata.
- The primary agent inspects source diffs plus critical reviewer transcripts
  or logs, records the reconciliation decision, and rejects output that
  crosses scope even when tests pass.
- Shared-worktree overlap triggers explicit reconciliation using Git history
  and worker provenance. Do not silently choose the newest edit.
- If no useful independent lane exists or runtime capacity is unavailable,
  record `not_spawned` and the concrete reason. Do not manufacture delegation
  solely to satisfy the receipt.

## Hard Stops

Immediately stop model/provider execution and record `stop` for:

- gold content or operator decisions entering retrieval, candidates, prompts,
  predictions, or worker context;
- an invented reference accepted by host validation;
- missing or cross-boundary source profile/account/tenant scope;
- provider or model control of retrieval;
- raw provider bodies outside the bounded snapshot contract;
- unexpected provider, CRM, contact, speaker-assignment, Graphiti, or other
  external write;
- mutation, replacement, or premature reveal of the frozen cohort;
- evidence-family contamination or duplicate-independence inflation;
- database-authority or automatic-confirmation enablement.

Record `refine` rather than `stop` for a bounded implementation/readiness
failure that preserves privacy, scope, evidence integrity, and the unseen
cohort.

## Acceptance Criteria

- Every configured source scope is exercised successfully or returns a
  correctly labeled bounded failure; at least one included provider snapshot
  is required before provenance/combined prediction.
- Adapters contain no person-grouping or identity inference.
- Every request records explicit source profile, account, tenant, capability,
  `as_of`, freshness, temporal, query-plan, and budget policy.
- The default Identity Evaluation receipt identifies request and bundle hashes,
  source failures, warnings, included/excluded reasons, freshness, and temporal
  class.
- Partial provider failure remains on the immutable-bundle path. Any legacy
  use has an explicit approval, warning, and receipt.
- Exact verified-identity grouping is deterministic and non-mutating; softer
  grouping remains evidence-cited and confidence-scored.
- Current undated contacts and later-fetched historical evidence are never
  mislabeled contemporaneous.
- The frozen cohort remains unchanged and blind until all predictions exist.
- Gold is independently operator-confirmed, hash-bound, and absent from every
  prediction surface.
- Every family reports family-specific candidate recall, own-list and
  frozen-union ranking, both temporal strata where applicable, exact
  denominators, and reason-coded nulls.
- Combined retrieval is never inferred from missing provenance/history input.
- Private shadow read agreement, backup/restore, rollback, and sidecar
  authority are proven before prediction.
- The terminal decision names residual risk and leaves authority states
  explicit.

## Validation

- Plan-version, freeze-hash, gold-visibility, delegation, attempt-bound, and
  checkpoint audits.
- Focused adapter, retrieval, workflow, API, projection, and campaign tests.
- Exact evidence-family visibility and prohibited-field tests.
- Candidate-list/union freeze, temporal-stratum, independence, and query-plan
  receipt tests.
- Full host-safe test inventory and Python compile checks.
- Private shadow projection/reconciliation plus backup/restore/rollback.
- Receipt hashes and `0600`/`0700` permissions.
- `git diff --check`, focused commits, push verification, and served-runtime
  verification before any live evaluation.

## Definition Of Done

Plan 0030 is done only when:

1. every execution packet has a terminal checkpoint and delegation receipt;
2. all applicable acceptance criteria have direct current evidence;
3. the frozen cohort is either still unseen after an early `refine`/`stop`, or
   fully captured before reveal;
4. one immutable `accept`, `refine`, `reject`, or `stop` receipt exists;
5. sidecar authority, database authority, automatic confirmation, and external
   write states are explicit;
6. Plan 0030, `ROADMAP.md`, and `RUNBOOK.md` agree on the terminal state;
7. focused and full validation, commit, push, and relevant served-runtime
   proof are recorded.

## Stop Condition

Stop after R3D records an accept, refine, reject, or stop decision, or earlier
when a bounded readiness failure or hard stop requires the same terminal
decision. Do not consume another chronological cohort, extend retry/rework
bounds, add an authority cutover, or start an unbounded hardening loop in this
plan.
