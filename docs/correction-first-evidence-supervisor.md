---
last_updated: 2026-08-16
applies_to: Plan 0072 A4 and knowledge schema v7
---

# Correction-first evidence supervisor

Plan 0072 A4 adds a replayable, zero-effect supervisor for identity evidence.
It binds each run to exact capabilities, source scopes, budgets, input hashes,
rubric versions, and stage history before recording hypotheses or scores.

> **Note:** This component is experimental and non-live. It doesn't authorize
> provider access, historical processing, live migration, accepted identity or
> profile effects, review publication, scheduling, or deployment.

This packet advances VISION outcomes 2, 3, 4, 5, 6, 7, and 8. It moves the
evidence supervisor from a Level 1 contract to a Level 2 replayable component
on synthetic inputs. A5-A9 retain the gates for review-product acceptance,
private shadow operation, reviewed learning, promotion, and automation.

## Public interface

`IdentityEvidenceSupervisor` in `identity_evidence_supervisor.py` owns the A4
boundary:

- `start_run(...)` validates the frozen A0 processing-run artifact and appends
  its initial stage.
- `advance_stage(...)` permits only the next supervisor stage and records
  outputs, partial failures, and zero effect counts.
- `record_adapter_exchange(...)` validates one read-only request and result,
  binds its tenant/account/capability scope, enforces cumulative budgets, and
  permits one retry only after a named transient failure.
- `record_conversation_candidate(...)` stores a calendar-association candidate
  without accepting it as the represented conversation.
- `record_purpose_hypothesis(...)` stores a purpose label, alternatives, and
  evidence without treating it as a reviewed readout.
- `record_participant_hypothesis(...)` stores a suspected participant without
  establishing presence or speech.
- `score_candidate(...)` records four visible evidence pillars and computes a
  deterministic combined ranking score.
- `record_calibration_outcome(...)` appends one reviewed, source-disjoint
  outcome.
- `calibrated_likelihood(...)` returns `insufficient_data` below 30 outcomes.
  At or above 30, it records the observed rate and a 95% Wilson interval.

## Knowledge schema v7

`ConversationKnowledgeStore.migrate()` adds immutable supervisor runs, stage
events, adapter exchanges, calendar candidates, purpose and participant
hypotheses, score batches, pillar assessments, calibration outcomes, and
calibration snapshots.

SQLite triggers reject `UPDATE` and `DELETE` on every authoritative v7 row.
Rollback from v7 removes only v7 objects and restores schema v6. It preserves
A1 identity/contact history, A2 transcript corrections, and A3 biometric
custody.

## Run and adapter boundaries

Every run retains the original recording filename, conversation and recording
IDs, source hashes, as-of time, operation mode, policy version, capabilities,
exact source scopes, budgets, model/rubric/profile versions, inputs, failures,
and effect counts. A4 accepts only `contract_fixture` and `shadow` modes with
zero effects.

An adapter request must use a configured provider kind, profile, account,
tenant, and capability. The result must bind to the same request, run, and
source scope. It must report zero provider writes and consumed record,
character, call, and latency budgets.

Budget consumption accumulates across the run. The supervisor refuses the
exchange that would exceed any run limit. A partial or unavailable result
remains visible without deleting prior successful observations. One retry is
available only for `transient_timeout`, `transient_unavailable`, or
`rate_limited`; authorization, scope, schema, and privacy failures do not retry.

## Hypotheses remain nonbinding

Calendar association, conversation purpose, and suspected participants are
separate records. Calendar attendees and event fit can generate hypotheses,
but none of these artifacts assigns a person to a voice or accepts a calendar
event as the represented conversation.

Each hypothesis carries alternatives and evidence lineage. Later review APIs
may project these records, but A4 creates no decision or effect.

## Evidence strength and lineage

The host records four 0-100 Evidence Strength pillars:

- calendar association;
- person link;
- contextual speaker; and
- acoustic speaker.

Each pillar retains positive and negative factors, evidence IDs, evidence-
independence groups, and a material-contradiction flag. The supervisor records
calendar, person, context, and acoustic values separately before taking their
equal-weight mean as the combined ranking score.

The combined score is not a probability. A material contradiction caps it at
49 and requires review. Reuse of one evidence-independence group across
pillars receives the same conservative cap, so duplicated evidence cannot
create false corroboration.

Every re-score cites the exact predecessor assessment, rubric version, and
model version. Historical scores remain immutable and may rise or fall.

## Calibrated likelihood

Calibration outcomes bind correctness to a reviewed decision, score band,
pillar, evaluation version, and source-disjoint identifier. Reusing one
source-disjoint identifier with a different outcome fails closed.

The supervisor shows no empirical likelihood until the relevant band has 30
source-disjoint outcomes. Once eligible, it returns the observed rate, sample
size, 95% Wilson interval, evaluation version, and exact input watermark. This
history is measurement evidence only; it doesn't promote an automatic band.

## Redacted replay evidence

The synthetic [supervisor replay fixture](dev/fixtures/plan-0072-a4/supervisor-replay.json)
binds one contract-fixture run to redacted source scopes and four independent
pillars. It replays a combined score of 75 with zero provider writes and zero
accepted identity or profile effects.

The fixture contains no provider body, credential, private transcript, real
tenant selector, raw media, biometric payload, or real identity.

## Current authority boundary

A4 tests use only committed redacted JSON and pytest temporary stores. No
adapter calls a provider. No historical or new conversation is processed, no
queue is published, and no runtime service changes.

A5 may build local review APIs and Authelia-protected preview views over these
records. Provider access and private live-shadow processing remain closed
until their later packet gates.

## Related documents

- [Correction-first identity-learning contracts](correction-first-identity-learning-contracts.md)
- [Correction-first identity ledger](correction-first-identity-ledger.md)
- [Correction-first transcript learning](correction-first-transcript-learning.md)
- [Correction-first biometric custody](correction-first-biometric-custody.md)
- [Plan 0072](dev/plans/0072-2026-08-16-correction-first-speaker-contact-learning.md)
- [Plan 0072 grilled architecture decisions](dev/notes/0058-2026-08-16-plan0072-grilled-architecture.md)
