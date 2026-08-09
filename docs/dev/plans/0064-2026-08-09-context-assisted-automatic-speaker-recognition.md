# Plan 0064 | Context-assisted automatic speaker recognition and enrichment

State: OPEN

Checkpoint: Plan 0063 terminal live learning state is installed and replayed;
P0 contract and corpus freeze are ready

Lane: P09/P10

Cross-lane dependency: closed Plans 0025, 0029, 0030, 0059, 0060, 0062, and
0063

Critical-Path Owner: primary agent

## Scope

Implement the incoming-conversation path that loads the current governed
biometric profile inventory, runs the existing bounded contextual identity
workflow, joins both evidence pillars through canonical-person bindings, and
solves speaker identity at conversation level. Prove the specific reusable
pattern in which two enrolled voices are recognized and a third speaker is
assigned only when one independently supported context candidate remains.

Persist accepted speaker observations and provenance-backed source affinities
to the user-scoped conversation knowledge/contact stores. Produce
non-destructive provider-enrichment proposals for configured contact sources;
keep direct external provider mutation behind a later field-ownership and
effect-receipt packet.

The durable design authority is
[Note 0056](../notes/0056-2026-08-09-context-assisted-automatic-speaker-recognition.md).
Private names, emails, provider IDs, audio, and biometric values remain in
user-scoped runtime artifacts.

## Vision outcomes and maturity movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Active biometric recognition | Level 2 reviewed profiles exist but new conversations do not consume them automatically | Level 3 automatic profile inventory and calibrated per-slot candidates with safe abstention | Replayable scoring receipts on the reviewed corpus plus a source-disjoint chronological cohort |
| Context/acoustic join | Level 2 branches and schemas exist; Plan 0063 installed bindings but no incoming orchestration | Level 3 automatic evidence-preserving join | Separate acoustic/context factors, agreement/conflict handling, exact canonical bindings, deterministic replay |
| Residual speaker assignment | Level 0 | Level 2 measured shadow, then Level 3 only inside a validated policy band | Correct two-known-plus-one-residual outcomes, explicit counterexamples, zero unacceptable high-confidence errors |
| Contact/provenance enrichment | Level 2 canonical source affinities exist for reviewed people | Level 3 accepted local observations enrich later retrieval automatically | Round-trip person profiles, source lineage, no duplicate provider votes, measurable retrieval improvement |
| External provider write-back | Level 0 | Level 1 proposal contract only | Field-level proposal and rollback/effect design; zero external writes in this plan |

This advances VISION outcomes 3, 6, 7, and 8 and the automatic knowledge loop.
It moves the product from reviewed learning-state preparation toward correct,
low-touch use on future conversations.

## Current State

Plan 0063 terminal receipt
`259ea605015ecd6b681140e529002c23e131b6e5cada0d1cdd62fc2b151e3dd5`
replays one completed live apply with six canonical people, nine reviewed slot
bindings, one active voice/person binding, five references, fifteen profiles,
and twenty-three enrollment sources. Both transcript services are
`active/running` with zero restarts.

The repository already has per-speaker Clue Discovery and Identity Evaluation,
bounded provider retrieval, canonical-person evidence bundles, acoustic shadow
evidence, and joined evaluation schemas. The missing runtime seam is dynamic
active-profile selection plus a conversation-level resolver that consumes
those existing components and projects accepted outcomes back into reusable
knowledge.

## Non-Goals

- Do not identify a residual speaker by elimination alone.
- Do not count duplicate provider/contact records as independent support.
- Do not enroll a new voice from an automatic identity proposal.
- Do not overwrite provider-authoritative Google, Odollo, or other external
  contact fields in this plan.
- Do not place raw transcripts, audio, embeddings, private provider payloads,
  or private identity values in tracked artifacts or Graphiti.
- Do not make the watcher perform provider retrieval or model work inline.
- Do not claim dependable Level 4 identity from the reviewed training corpus.

## Execution packets

| Packet | Outcome | Terminal evidence |
| --- | --- | --- |
| P0 contract/corpus freeze | Bind active profile inventory, person bindings, reviewed development corpus, and a chronological source-disjoint evaluation set | Immutable private manifest; no overlap; current store/service hashes |
| P1 dynamic acoustic evidence | Score each eligible diarized slot against only active governed profiles and translate subjects through reviewed person bindings | Per-slot candidates/abstentions, calibration version, model/source/profile hashes |
| P2 contextual evidence reuse | Run the existing two-phase clue and identity workflow with current canonical/source affinities | Per-slot candidates, alternatives, contradictions, provider failures, as-of-time boundaries |
| P3 conversation resolver | Join both pillars and solve globally, including reason-coded residual inference | Deterministic assignments/abstentions; one-to-one and multi-label cases; no hidden fusion score |
| P4 measured shadow | Compare context-only, acoustic-only, combined, and residual-policy outcomes on both corpora | Correctness, high-confidence error, abstention, review rate, candidate recall, lineage completeness |
| P5 local acceptance/enrichment | Enable policy-qualified local speaker observations and canonical profile/source-affinity refresh for the validated band | Idempotent apply/rollback, knowledge round trip, improved later retrieval, zero external writes |
| P6 provider proposal handoff | Prepare field-owned, deduplicated external contact enrichment proposals | Reviewable proposal contract and successor plan; no provider mutation |

P1 and P2 may run independently after P0. P3 joins them. P4 must pass before
P5 can enable any automatic local acceptance. P6 does not authorize an
external effect.

## Execution bounds

- `max_work_unit_attempts`: 2 per packet.
- `max_review_rework_cycles`: 1 closed-world cycle for accepted blocking
  findings.
- `max_development_conversations`: the three Plan 0063 source conversations.
- `max_evaluation_conversations`: 12 chronological source-disjoint recordings.
- `max_profile_models`: the three governed active adapters installed by Plan
  0063.
- `max_automatic_policy_bands`: 1 initial high-support band.
- `max_hardening_checkpoints_without_outcome_progress`: 2.
- `checkpoint_interval`: after each packet and before any local apply.
- `authorization_gate`: significant departure only; ordinary implementation,
  testing, repair, and bounded local shadow progression do not create new
  approval rituals.

Delegation receipt: `not_spawned`. Current system authority forbids proactive
subagents unless the user explicitly requests them.

## Acceptance Criteria

- Every eligible speaker slot receives independent acoustic and contextual
  evidence bundles or a reason-coded unavailable state.
- Active acoustic subjects resolve to canonical people only through reviewed
  voice/person bindings; agreement and conflict with context remain visible.
- The resolver operates over the full speaker-slot set and supports both
  one-to-one coverage and same-person multi-label cases.
- Residual assignment requires two accepted known-person bindings, exactly one
  independently supported remaining canonical candidate, relevant transcript
  support, complete provenance, and no material contradiction.
- Ambiguous, incomplete, duplicate, or conflicting cases abstain and produce a
  useful audio-linked review rather than a context-free question.
- Development sources never enter evaluation metrics as unseen evidence.
- The source-disjoint evaluation has zero unacceptable high-confidence wrong
  identities before any automatic local acceptance is enabled.
- Accepted local observations round-trip through canonical profiles and
  improve candidate retrieval for a later conversation without circular
  self-support.
- Existing legacy contacts and provider source records remain deduplicated and
  provenance-preserving.
- External provider write count remains zero; P6 produces proposals only.

## Validation

- Focused tests for dynamic profile inventory, inactive/withdrawn profiles,
  missing person bindings, context/acoustic agreement, conflict, duplicate
  provider records, one-to-one assignment, multi-label identity, residual
  acceptance, and residual abstention.
- Exact replay of the Plan 0063 reviewed corpus plus a frozen chronological
  source-disjoint evaluation.
- Condition comparison for context-only, acoustic-only, combined, and
  residual-policy outcomes.
- Private artifact modes, input/output hashes, SQLite `quick_check`, knowledge
  round-trip, rollback, service isolation, and zero external/provider effects.
- Direct-audio browser proof for every routed review case with the question
  adjacent to its evidence.
- Python compilation, focused and full pytest, active/goal planning audits,
  CodeGraph post-edit readback, `git diff --check`, clean commit/push, and exact
  upstream equality.

## Definition of done

Plan 0064 is complete when future eligible conversations automatically use the
reviewed biometric identities and the existing contextual workflow together;
the measured high-support band can correctly accept known speakers and a
context-supported residual speaker without elimination-only guessing; accepted
local observations enrich canonical/contact provenance for later retrieval;
all other cases abstain usefully; and external provider writes remain zero
pending a separately validated write-back plan.
