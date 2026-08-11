# Plan 0067 | Calendar evidence contract reconciliation and zero-turn replay

State: CLOSED

Active packet: None. Terminal A0 authority gate exhausted.

Checkpoint: Plan 0066 remains immutably closed `withhold` at terminal
`c5a843f80939972f...`. A post-closeout source/runtime audit proved its six
validation failures were misdiagnosed in tracked prose: all seven rejected
calendar citation occurrences were present in the corresponding frozen,
host-validated first-pass `calendar_clue_ids`. The second-pass packet builder
validated those IDs against `calendar_evidence`, then retained the discovery
readout but dropped the explicit calendar evidence catalog. Its strict
reference collector therefore could not authorize the same prepared IDs. The
raw model outputs remain privately retained in the six Plan 0066 model-turn
status artifacts and can be replayed without another model or provider call.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064-0066

Critical-Path Owner: primary agent

## Scope

Correct the Plan 0066 diagnosis without reopening or rewriting its terminal,
repair the host-owned first-pass-to-second-pass calendar evidence contract,
and replay the six exact retained Plan 0066 outputs under the repaired
validator with zero model turns. Measure those validated proposals against the
same frozen Plan 0064 human gold and close pass/withhold at the same
non-vacuous development gate.

The repair must carry the validated `calendar_evidence` catalog into the
identity-evaluation packet and derive allowlisted IDs and independence groups
from that explicit host catalog. It must not authorize IDs merely because a
model repeated them inside a discovery or evaluation readout. Calendar title,
attendee, and matching-event evidence remains candidate-only: a
`candidate_match` speaker assignment must cite a prepared transcript clue in
its factor evidence and cannot be supported by calendar evidence alone.

Plan 0067 may write product code/tests, tracked correction/plan/runbook docs,
and content-addressed private replay receipts under the user-scoped Plan 0067
runtime. It may read the frozen Plan 0066 packets, model-turn statuses, terminal,
and human gold. It may not edit Plan 0066 private artifacts, change a retained
model output, send a model/provider turn, retrieve fresh evidence, mutate a
source/stored transcript or live index row, apply an assignment, change a
person/enrollment/biometric record, write accepted conversation knowledge,
write Graphiti, or perform another external effect.

## Vision outcomes and maturity movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Calendar/conversation association evidence | Level 1 first-pass IDs are validated but disappear from second-pass authority | Level 2 explicit, source-bound calendar catalog survives second-pass shadow evaluation | Packet contract tests and six-case exact replay |
| Contextual speaker candidate quality | Level 1 Plan 0066 proposals could not be measured because the host rejected valid prepared references | Level 2 only if repaired replay yields at least one correct candidate, zero wrong candidates, and complete lineage | Frozen-output validation plus literal human-gold comparison |
| Source integrity and replay | Level 2 Plan 0066 private preparation is exact and non-mutating | Remain Level 2 while reusing frozen packets and outputs without a model call | Hash bindings, private containment, zero-turn/effect receipt |
| Local/external acceptance | Level 0 automatic apply | Remain Level 0 | Terminal negative-action vector |

This advances VISION outcomes 2, 3, and 6 by preserving calendar association
evidence across the host boundary, measuring speaker proposals against their
actual prepared citations, and retaining calibrated provenance without
granting model output authority. It does not advance outcomes 7 or 8 because
no observation is accepted or projected into live knowledge.

## Current State

- `build_identity_evaluation_packet` reconstructs and validates the first-pass
  discovery packet, but copies only `calendar_context` into the second-pass
  packet and omits `calendar_evidence` and its stable evidence IDs.
- `_prepared_identity_references` recognizes conversation, recording,
  transcript, attendee, person, source-record, source-context, and provenance
  IDs, but cannot authorize the omitted calendar evidence catalog.
- Across the six Plan 0066 cases, seven rejected calendar citation occurrences
  are members of their case-local validated discovery `calendar_clue_ids`.
  None is a newly invented ID.
- Each Plan 0066 model-turn status is a private `0600` artifact retaining
  `output_text`, raw turn metadata, thread/turn IDs, and completion state.
- Plan 0066 used six primary turns, zero fallback turns, and zero retries. This
  successor authorizes zero additional turns and zero reference repair.

## Accepted finding ledger

| Finding | Criterion | Evidence | Disposition |
| --- | --- | --- | --- |
| F1 calendar catalog projection gap | Every host-prepared evidence ID shown to the evaluator must remain valid in strict second-pass validation | Builder drops `calendar_evidence`; all seven rejected IDs are present in validated case-local discovery citations | `blocking` |
| F2 calendar-only speaker-binding risk | Calendar evidence may generate candidates but cannot prove who spoke | First-pass policy is candidate-only; second-pass assignment validation does not require factor-level transcript support for `candidate_match` | `blocking` |
| F3 Plan 0066 narrative drift | Durable closeout prose must distinguish model invention from a host-contract defect | Plan/roadmap/runbook currently call the seven IDs invented or absent from the packet | `blocking` |
| F4 retained-output replay viability | Re-evaluation must not consume another model turn or alter output | Six private completed status artifacts retain exact `output_text` and raw turn bindings | `needs_evidence` until A0 freeze/replay |

The goal-level broad drift-discovery pass was already consumed by Plan 0066.
Plan 0067 verification is closed-world against F1-F4 plus critical regressions
introduced by their remediation.

## Non-Goals

- Do not reopen, replace, or relabel the Plan 0066 terminal.
- Do not repair, normalize, regenerate, or ask a model to restate any retained
  output.
- Do not allow evidence based only on its presence in model-authored readout
  fields; authorization comes from the explicit host calendar catalog.
- Do not weaken unknown-ID rejection, evidence citation, temporal, scope,
  human-confirmation, mixed-speaker, or abstention controls.
- Do not use human gold to change packets, candidates, outputs, or validation.
- Do not open joined/residual work or a fresh source-disjoint cohort here.
- Do not publish private names, emails, person IDs, transcripts, output bodies,
  provider payloads, or original recording basenames in tracked files.

## Authority and activation

- The standing operator objective `plan and execute` activates this bounded
  successor from clean, upstream-even commit
  `ff9deb07a982d2b33813ed245cbe66760dc90e41`.
- Plan 0066 remains historical execution authority. Plan 0067 may correct its
  tracked explanation but cannot mutate its terminal or private evidence.
- A0 freezes exact Plan 0066 terminal, packet, status/output, gold, source, and
  code bindings before the product fix or replay.
- A significant departure, including a new model turn, fresh retrieval,
  reference repair, live apply, or fresh evaluation cohort, requires a new
  plan.

## Execution graph

| Packet | Depends on | Bounded outcome | Expected write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| A0 authority/correction freeze | activation | Bind the six packet/status pairs, seven valid calendar citations, Plan 0066 terminal, gold, code authority, and zero-effect budget | Plan module/tests plus private A0 manifest/receipt | Exact replay or fail closed |
| A1 product contract repair | A0 | Carry explicit validated calendar evidence into the second pass, authorize only catalog IDs, and forbid calendar-only candidate matches | `speaker_identity_preprocess.py` and focused tests | Red-to-green contract and safety tests |
| A2 zero-turn replay | A1 | Parse and validate each exact retained output against a deterministically rebuilt repaired packet | Private case/measurement receipts only | Six terminal case dispositions, zero model turns |
| A3 measurement/terminal | A2 | Compare only validated assignments with frozen gold and close pass/withhold | Private terminal receipt and tracked closeout docs | `context_candidate_recovered` or reason-coded `withhold` |

Intended active-agent concurrency is `1`. Delegation receipt is `not_spawned`:
current system authority prohibits proactive subagents, and the authority,
implementation, and replay steps share one tightly coupled evidence seam.

## Execution bounds

- `max_work_unit_attempts`: 3 for A0 and 2 for each remaining packet.
- `max_policy_revisions`: 1 before A2; 0 after replay begins.
- `max_review_rework_cycles`: 1 closed-world F1-F4 cycle.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `max_development_cases`: 6 exact inherited cases.
- `max_primary_model_turns`: 0.
- `max_fallback_model_turns`: 0.
- `max_provider_retries`: 0.
- `max_reference_repairs`: 0.
- `max_fresh_evaluation_runs`: 0.
- `checkpoint_interval`: after each packet and before replay.
- `review_discovery_passes`: 0 additional; inherited goal total remains 1.
- `review_verification_mode`: closed_world.

Revision 1 (pre-A2): A0's first two fail-closed executions exposed two
authority-check assumptions rather than evidence drift: legacy transcript
store inputs are hash-bound regular files but not necessarily mode `0600`, and
Plan 0066 A2 intentionally transforms A1 packets by carrying forward prior
provenance, retrieval, and source contexts. No A0 artifact was written. The
single allowed policy revision adds one A0 attempt so the freeze can validate
the documented `build_a2_packet` transformation and preserve strict private
mode requirements for newly written Plan 0067 artifacts. Model, retrieval,
repair, evaluation, and effect budgets remain zero.

## Acceptance Criteria

- The second-pass identity packet contains the exact first-pass
  `calendar_evidence` catalog already used to validate discovery citations.
- Strict reference collection authorizes each catalog `evidence_id`, rejects an
  ID absent from that catalog, and assigns one stable calendar independence
  group without double-counting same-event clues.
- A `candidate_match` assignment must cite a prepared transcript clue in its
  factor evidence; calendar-only speaker binding fails closed.
- All six Plan 0066 outputs replay byte/text-exactly with zero model turns,
  fallback turns, retries, reference repairs, or fresh retrieval.
- The replay emits at least one correct prepared candidate, zero wrong prepared
  candidates, zero schema/reference violations, and complete provenance to
  pass. A zero-candidate result cannot pass.
- Source/stored transcripts, index rows, assignments, identities, accepted
  knowledge, biometrics, providers, Graphiti, and external systems remain
  unchanged.

If F1 or F2 cannot be fixed without broadening evidence authority, A2 does not
open. If any retained output remains invalid, produces a wrong candidate, or
lacks required lineage, Plan 0067 closes `withhold`; no output repair or model
retry is allowed.

## Validation

- Red-capable product tests for calendar catalog propagation, exact-ID
  acceptance, unknown-ID rejection, stable independence grouping, and
  calendar-only candidate-match rejection.
- Plan tests for exact status/output binding, six-case denominator, seven
  prepared citation occurrences, zero-turn replay, human-gold measurement,
  private containment/modes, and immutable terminal replay.
- Existing preprocessing, workflow, Plan 0066, contextual join, API, transcript
  artifact, and knowledge retrieval regressions.
- Full pytest, Python compilation, active/goal planning audits, CodeGraph
  post-edit readback, `git diff --check`, clean commits, push, exact upstream
  equality, and post-commit receipt replay.
- A transcription/DOCX smoke is not applicable unless normalized transcription
  or export behavior changes.

## Definition of done

Plan 0067 is complete when the host calendar catalog survives the second-pass
boundary with candidate-only safety intact, all six exact retained outputs
receive terminal zero-turn dispositions, and A3 emits either a non-vacuous
`context_candidate_recovered` result with zero wrong candidates or a
reason-coded `withhold`. No live or external effect is part of done.

## Terminal closeout

Plan 0067 closed `withhold` before A0 wrote an artifact. Its three bounded
attempts exposed inherited-artifact mode assumptions in the freeze harness and
the intentional Plan 0066 A1-to-A2 provenance transformation; the product
calendar contract was not changed and no retained output was replayed.
Terminal content `8c11ed7c3cd0f8bbd1185f299f9d9f6a81fee28f8612609f6b7a21ff990c47c4`
and file `a834a55b3bdf1f609a277be7a6d9c46b9f60df0531557f2d0b1f81170405ccef`
replay with reason `a0_legacy_artifact_mode_contract_mismatch`. All model,
retrieval, repair, source/store/index, identity, knowledge, biometric,
provider, Graphiti, and external effect counts are zero.
