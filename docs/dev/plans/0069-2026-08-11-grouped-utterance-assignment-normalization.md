# Plan 0069 | Grouped utterance-assignment normalization

State: CLOSED

Active packet: None. Terminal replayed `pass`.

Checkpoint: Plan 0068 closed `withhold` after strict validation reported three
retained outputs with an empty singular `utterance_id`. Read-only inspection
now shows that those outputs are not empty assignments: they contain 10
substantive legacy grouped objects with `utterance_ids` arrays covering 28
prepared utterances. A deterministic counterfactual that expands each group
into otherwise-identical singular objects validates all six retained outputs
and passes frozen human-gold measurement.

Lane: P09/P10

Cross-lane dependency: closed Plans 0064-0068

Critical-Path Owner: primary agent

## Current State

Plan 0069 is closed with an immutable PASS terminal. Its six retained output
texts remain byte-exact, all six original recording filenames are bound in A0,
A2, case, and terminal evidence, and deterministic normalization reconciled
the exact 10-group/28-utterance legacy serialization without any model, live,
or external effect.

## Scope

### Vision outcomes

Advance VISION outcomes 2, 3, 4, and 6 by accepting one unambiguous legacy
serialization of already-prepared utterance assignments, retaining the six
original recording filenames, and producing auditable no-apply measurement.
Current maturity is Level 1 because three otherwise substantive retained
outputs fail the singular schema boundary. Target maturity is Level 2 only if
all six outputs validate after deterministic normalization and frozen human
gold reports at least one correct candidate, zero wrong candidates, zero
unavailable cases, and complete provenance. This plan does not advance live
observation or knowledge-reuse outcomes 7 or 8.

## Non-Goals

- Do not call a model, provider, retrieval source, or fresh evaluation.
- Do not change any retained Plan 0068 or Plan 0066 artifact.
- Do not infer, apply, publish, enroll, group, or write speaker identities.
- Do not open joined/residual evaluation, unseen evaluation, or live workflows.
- Do not accept arbitrary output repair; only the exact fail-closed grouped
  serialization normalization described below is in scope.

## Authority and normalization contract

Plan 0068 A0, A2, and terminal artifacts are immutable inputs. A0 must bind
their paths, content/file hashes, source commit, the six exact retained output
text hashes, all six original recording filenames, and the exact grouped
assignment inventory before product work begins. Every new Plan 0069
directory/file remains private `0700`/`0600`.

Normalization may replace an object containing `utterance_ids: [id, ...]`
and no `utterance_id` with one deep-copied object per listed ID, removing only
`utterance_ids` and adding the corresponding singular `utterance_id`. It must
reject non-object assignments, non-list/empty/blank/duplicate IDs, mixed
singular/plural shape, and any grouped ID outside the packet's prepared
utterance allowlist. It must not mutate its input or alter person IDs,
statuses, rationales, factors, review flags, speaker labels, provenance, or
any other field.

## Execution graph

| Packet | Depends on | Outcome | Writes | Terminal |
| --- | --- | --- | --- | --- |
| A0 | activation | Freeze Plan 0068 authority, six filenames, and exact 10-group/28-utterance inventory | Private manifest/receipt only | Exact replay or fail closed |
| A1 | A0 | Add fail-closed grouped-to-singular normalization | Product module and tests | Red-to-green focused tests |
| A2 | A1 | Normalize and validate the six exact retained output texts | Private case/measurement receipts | Six dispositions, zero model turns |
| A3 | A2 | Recompute frozen gold and close pass/withhold | Private terminal and tracked closeout | Non-vacuous pass or reason-coded withhold |

Delegation remains `not_spawned`: current system authority prohibits proactive
subagents, and the evidence seam is tightly coupled.

## Bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_policy_revisions`: 0.
- `max_development_cases`: 6 exact inherited cases.
- `max_grouped_objects`: 10 exact inherited objects.
- `max_expanded_utterance_assignments`: 28 exact inherited assignments.
- `max_primary_model_turns`: 0.
- `max_fallback_model_turns`: 0.
- `max_provider_retries`: 0.
- `max_model_reference_repairs`: 0.
- `max_fresh_retrievals`: 0.
- `max_fresh_evaluation_runs`: 0.
- `review_discovery_passes`: 0; the goal-level pass was consumed by Plan 0066.
- `review_verification_mode`: closed_world.

## Acceptance Criteria

- A0 binds six exact retained outputs, all six original recording filenames,
  and exactly 10 grouped objects covering 28 unique prepared utterance IDs.
- Focused tests prove the exact expansion, non-mutation, field preservation,
  and fail-closed rejection of blank, duplicate, mixed, or unknown IDs.
- A2 changes only the in-memory serialization of grouped utterance assignments;
  retained output text and every substantive field remain unchanged.
- All six outputs validate against their repaired Plan 0068 packets.
- Pass requires at least one correct, zero wrong, zero invalid/unavailable, and
  zero incomplete-provenance candidates. Zero candidates cannot pass.
- A2 performs zero model turns, retries, model repairs, retrievals, or fresh
  evaluations and retains the original filename on every case artifact.
- All live/source/store/index/identity/knowledge/biometric/provider/Graphiti/
  external effects remain zero.

Any ambiguity, authority drift, normalization outside the frozen inventory,
invalid output after normalization, or failed measurement closes `withhold`.
Joined/residual, fresh evaluation, live apply, and memory publication remain
closed.

## Validation

Run focused A0/A1/A2/A3 and preprocessing tests, Plan 0068 regressions, full
pytest, Python compilation, active/goal planning audits, CodeGraph post-edit
readback, `git diff --check`, exact terminal replay, mode checks, clean commits,
push, and upstream equality. A transcription/DOCX smoke is not applicable
because transcription and export behavior do not change. Done is an immutable
pass/withhold terminal with filenames retained and no unauthorized effect.

## Terminal closeout

A0 content `fd05ffce34c0fcb6dbbba88e203843636a1a3fbf6f509997e9eabb545fd04db2`
binds all six original recording filenames, six exact retained output hashes,
and exactly 10 grouped objects covering 28 prepared utterance IDs. A1 added an
explicit fail-closed normalizer that deep-copies the readout, preserves every
substantive field, and rejects blank, duplicate, mixed, unknown, or repeated
utterance IDs.

A2 manifest content
`c069bd52c0d747b9d075ae355ec001a33d85c24f1a85974b6042c4e758647471`
replays all six exact output texts. All six validate after in-memory
normalization; frozen human-gold measurement reports five correct candidates,
zero wrong, 17 abstained, zero incomplete provenance, zero unavailable, and
zero validation failures. Retained output changes, model turns, retries,
model repairs, retrievals, and fresh evaluations are all zero.

Terminal content
`2d53d01fbe9b4d953bb79ba0192206ed67f207d8c4c194fa18c2831e74770cfe`
and file `f3962ebbdd913cf6064b7ef45b5322759da0b45267a2f09a2ab37f741681fd78`
close `plan0069_closed_pass` with reason
`grouped_assignment_schema_reconciled`. Joined/residual, fresh evaluation,
live apply, knowledge, Graphiti, provider-write, and external gates remain
closed, and every effect count is zero.
