# Product vision

Transcribe Audio turns every eligible recording into a trustworthy,
fully contextualized readout. It also adds the conversation's accepted
knowledge to a private, growing, interconnected body of conversation
intelligence. That knowledge improves future transcript processing and gives
authorized agents grounded context about people, organizations, projects,
matters, topics, and relationships.

Fully contextualized does not mean falsely certain. The product must preserve
ambiguity, competing explanations, and missing evidence when the available
sources do not support a reliable conclusion.

## North-star outcome

The intended product is an automatic knowledge loop, not a collection of
independent transcription utilities. For each eligible recording, the system
must:

1. Produce a durable transcript and diarized utterances.
2. Determine which conversation the recording represents, including ambiguous
   or drifting calendar matches.
3. Identify the speakers from transcript clues, calendar attendees, contacts,
   email, files, log notes, and previously accepted conversation knowledge.
4. Establish the relevant topics, matters, relationships, terminology, and
   history that were available at the time of the conversation.
5. Generate a useful readout of what happened, why it matters, what was
   decided, what remains open, and what should happen next.
6. Attach evidence, provenance, calibrated confidence, alternatives, and
   unresolved questions to contextual claims.
7. Preserve the source artifacts and project accepted observations into the
   user-scoped conversation knowledge store.
8. Make that knowledge retrievable for later transcripts and for authorized
   agents seeking context about a person or matter.

## Product loop

Each accepted conversation should make the next relevant conversation easier
to understand without weakening source boundaries or temporal integrity.

```text
Recording
  -> transcript and diarization
  -> first-pass readout
  -> speaker and provenance inference
  -> topic, relationship, and history retrieval
  -> grounded contextual readout
  -> review or policy-qualified acceptance
  -> conversation knowledge store
  -> context for future transcripts and authorized agents
```

The loop retains the original evidence so that later improvements to
transcription, identity resolution, retrieval, or reasoning can replay the
same conversation. Derived conclusions may improve; the historical evidence
must not be silently rewritten.

## Contextualized readout

A completed contextual readout answers the following questions while showing
the basis for its answers:

| Dimension | Required answer |
| --- | --- |
| Conversation | What happened, what was decided, which actions were assigned, and which questions remain open? |
| Topics and matters | Which subjects, projects, organizations, entities, and specialized terms are involved, and why? |
| Speakers | Who does each diarized label represent, with what confidence, alternatives, and supporting or conflicting evidence? |
| Relationships | What roles, organizations, commercial or personal relationships, and prior interactions explain the exchange? |
| History | Which earlier conversations, messages, documents, events, or log notes are relevant as of the conversation time? |
| Provenance | Which source, account, tenant, record, and retrieval time supports each contextual claim? |
| Uncertainty | What remains unresolved, contradictory, or too weakly supported to accept automatically? |

Speaker identity is a conversation-level inference. Multiple diarized labels
may belong to the same person, and a person may appear under different labels
across recordings. Calendar attendees provide strong candidates but are not
proof: event-title fit, timing, transcript clues, known relationships, topics,
and contradictory evidence all affect the result.

## Knowledge and authority

The product separates evidence from interpretation and broad retrieval from
private authority.

- Transcript artifacts, audio references, provider snapshots, and immutable
  observations preserve what the system received or directly observed.
- Derived speaker assignments, entity links, topic associations, relationship
  histories, and readouts preserve the evidence and policy version that
  produced them.
- The user-scoped conversation knowledge store is the durable authority for
  conversation-derived knowledge. Provider systems remain authoritative for
  their full external records.
- Graphiti may receive compact, reviewed, source-backed projections for broad
  agent discovery. It is not the primary store for raw transcripts, private
  provider payloads, or conversation artifacts.
- App Intelligence reasons over bounded evidence prepared by the host. The
  host owns authorization, tenant routing, temporal filtering, person
  grouping, retrieval budgets, and durable writes.

The same person may appear in Google Workspace, one or more Odollo tenants,
and conversation history. Identity resolution should group those records when
the evidence supports it while retaining every source affinity. Those
affinities are themselves useful relationship evidence: a personal Google
Workspace contact and a company CRM contact can provide different, compatible
views of the same person.

## Operating principles

The automatic loop depends on the following non-negotiable behaviors:

- Prefer grounded inference to deterministic matching, but never hide the
  evidence or force a conclusion.
- Use numeric confidence for filtering and agent decisions, paired with
  plain-English confidence bands for human interpretation.
- Calibrate confidence against measured outcomes. A number has no operational
  meaning until its thresholds and consequences are validated.
- Allow obvious, well-supported cases to proceed automatically. Route
  ambiguous, contradictory, novel, or high-risk cases to review or defer them
  without treating review as the default for every conversation.
- Do not count duplicate provider records as independent corroboration.
- Preserve source, account, tenant, authorization, and as-of-time boundaries
  through retrieval and projection.
- Treat model output as a reasoned proposal until an acceptance policy or
  human decision gives it authority.
- Keep processing replayable, auditable, reversible, and idempotent.
- Keep raw transcripts and private provider content out of broadly shared
  memory surfaces.
- Prevent hindsight leakage when evaluating historical conversations. Later
  knowledge may help current processing, but historical quality tests must use
  only evidence available at the relevant time.

## Measures of progress

Progress is measured by outcomes across the whole loop, not by service health,
provider availability, or isolated examples. Plans and evaluations use this
shared maturity scale:

| Level | Band | Meaning |
| --- | --- | --- |
| 0 | Absent | The capability does not exist. |
| 1 | Built | The capability is implemented and tested in isolation. |
| 2 | Shadow | The capability runs manually or in shadow on real artifacts. |
| 3 | Operational | The capability runs automatically, is measured, and has a safe fallback. |
| 4 | Dependable | The capability is a reliable, self-feeding part of the production knowledge loop. |

The principal measures are:

- Pipeline yield: the share of eligible recordings that reach every processing
  stage and the explicit reason each remaining item stops.
- Identity quality: candidate recall, assignment correctness,
  high-confidence error rate, calibration, and human-review rate.
- Readout quality: coverage and correctness of decisions, actions, topics,
  relationships, and history, plus citation and appropriate-abstention rates.
- Knowledge integrity: projection coverage, duplicate-person resolution,
  temporal correctness, provenance completeness, and deterministic replay.
- Retrieval utility: how often future transcript processing and authorized
  agents retrieve useful, correct, source-supported context.
- Autonomy: latency, human touches per conversation, retry burden, and the
  proportion of cases safely completed without intervention.

Provider yield and runtime health are prerequisites. They do not, by
themselves, demonstrate progress toward the product outcome.

## Planning contract

Every substantive roadmap lane or bounded execution plan must state:

1. Which north-star outcomes it advances.
2. The current and target maturity levels for those outcomes.
3. The measurable user or system outcome it is expected to change.
4. The evidence that will prove the change.
5. What the work deliberately does not prove or complete.
6. How it improves automatic contextualization or reusable knowledge.

Infrastructure work may be necessary, but it must name the downstream product
capability it enables. Passing tests, reaching a provider, or closing a
bounded plan cannot be reported as realizing the vision unless representative
end-to-end evidence supports that claim.

[ROADMAP.md](ROADMAP.md) organizes the work toward this outcome.
[RUNBOOK.md](RUNBOOK.md) records execution evidence and current state.
Bounded plans define authorized slices. This document remains the authority
for the intended product outcome when a local milestone or implementation
detail could otherwise narrow the objective.

## Definition of realized

The vision is realized when representative eligible recordings automatically
become contextual readouts that correctly identify the conversation,
speakers, topics, relationships, and relevant history with calibrated
confidence and source-level provenance.

Well-supported cases must proceed without routine human approval. Unresolved
cases must defer safely and explain why. Accepted observations must update the
private conversation knowledge store deterministically, improve later
transcript processing, and remain available to authorized agents through
grounded retrieval. The complete loop must meet validated quality and autonomy
thresholds over a representative chronological corpus while preserving
privacy, tenant isolation, auditability, replay, and rollback.

## Related architecture

The vision is implemented through the following durable authorities:

- [Conversation knowledge storage and retrieval](docs/conversation-knowledge-storage-and-retrieval.md)
- [ADR 0002: Use a user-scoped conversation knowledge store](docs/adr/0002-use-a-user-scoped-conversation-knowledge-store.md)
- [Repository roadmap](ROADMAP.md)
- [Repository agent guidance](AGENTS.md)
