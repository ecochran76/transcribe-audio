# Note 0061 | Two-loop evidence-fabric architecture

Date: 2026-09-01

Status: accepted architecture direction

## Decision

Calendar, Mail Receipts, Contacts, Google Drive, SysRAG, messages, CRM, prior
transcripts, and future corpora are evidence sources. They are not separate
product architectures. The product is organized around two parallel learning
loops joined by one host-owned evidence fabric and one append-only knowledge
authority.

```text
governed evidence sources
  -> capability-scoped Evidence Fabric
       -> People and Relationship Discovery
       -> Conversation Understanding
  -> proposals and cited readouts
  -> review or policy-qualified acceptance
  -> append-only knowledge ledger and rebuildable projections
  -> bounded context for later conversations
```

People and Relationship Discovery is a first-class product loop. It builds
temporal, evidence-backed knowledge about people, external identities,
organizations, roles, affiliations, relationships, interaction history, and
conflicts. It is not merely a prerequisite for speaker identification.

Conversation Understanding is the per-conversation loop. It determines which
conversation a recording represents, who participated and spoke, what the
exchange meant, which projects or matters explain it, what was decided, and
what remains open. It consumes accepted people/relationship knowledge plus
bounded direct evidence and returns new claims for review.

## Evidence-fabric seam

The external interface is capability-based rather than provider-based:

```python
collect(EvidenceRequest) -> EvidenceBundle
```

An evidence request carries the requesting purpose, conversation/person/
organization/matter anchors, exact source scopes, allowed capabilities,
`as_of` and hindsight policy, freshness policy, record/character/provider-call
budgets, and bounded relationship hops. An evidence bundle carries normalized
observations, bounded provider snapshots, accepted local-knowledge context,
source failures, unresolved gaps, independence groups, the accepted-knowledge
watermark, and a deterministic content hash.

Adapters advertise capabilities. Provider-specific query syntax, pagination,
rate limits, privacy shaping, and error handling remain behind each adapter.
The fabric owns common scope, capability, temporal, freshness, budget,
provenance, and failure semantics.

Initial capability families are:

- identity directories: contacts, people, organizations, and CRM records;
- interaction history: calendar, mail, messages, and prior conversations;
- content context: Drive documents, SysRAG evidence, log notes, and other
  governed corpora;
- accepted knowledge: people, role, relationship, organization, matter, and
  terminology projections; and
- acoustic evidence: private recording samples and governed profiles.

Google Drive is a document/context adapter. SysRAG is a governed semantic
filesystem-recall adapter returning cited evidence. Neither source becomes the
authority for derived people, relationship, speaker, or conversation claims,
and neither may directly mutate the local knowledge ledger.

## Authority and circularity

Evidence never flows directly from a provider or model into projected truth:

```text
source -> immutable observation -> attributed hypothesis
       -> review or qualified acceptance -> authoritative projection
```

Each processing run freezes the accepted-knowledge watermark it used. A role,
relationship, speaker assignment, transcript correction, or purpose proposed
from the current conversation cannot support itself in that processing
version. Accepted findings may influence a later conversation or a separately
versioned rerun. Historical evaluation excludes later evidence unless the run
is explicitly labeled as hindsight.

Records derived from one underlying interaction share an independence group,
even when Calendar, Mail, CRM, and a copied note repeat the same fact.

## Review projections

Identity Review remains conversation-first. People remains entity-first. They
are projections over the same claim IDs, evidence IDs, decision state, and
watermark. Contacts are source records attached to provisional or canonical
people, not a second flat authority.

The first walking skeleton must prove, on a disposable store, that one reviewed
relationship originating in conversation A becomes cited accepted context for
later conversation B, is not available to A itself, respects `as_of`, and
replays byte-equivalently. It must also preserve existing speaker-identity
retrieval through the same evidence-fabric seam. Live/private relationship
acceptance remains outside that first implementation milestone.

## Maturity movement

- Current evidence adapters: Level 1 built with bounded Level 2 source-specific
  shadow executions.
- Current people/relationship discovery: Level 2 proposed-only shadow with no
  accepted relationship learning loop.
- Current conversation understanding: Level 1 isolated readout and retrieval
  capabilities without one automatic coordinator.
- First target: Level 1 integrated two-loop walking skeleton with deterministic
  replay and no live/private effects.
- Later target: Level 2 chronological shadow proving review usefulness and
  contextual lift on representative conversations before any Level 3
  automation.
