# Plan 0074 | Two-Loop Evidence-Fabric Walking Skeleton

State: CLOSED

Lane: P09

Date: 2026-09-01

## Scope

Implement the first vertical slice of the architecture frozen in Note 0061:
one capability-based evidence fabric used by both people/relationship discovery
and conversation understanding. Preserve the current speaker-identity entry
point while moving its provider collection and accepted-relationship read to
the shared seam.

On a disposable schema-v8 store, append a reviewed relationship derived from
conversation A and prove that a later conversation B can retrieve it as cited
accepted context. The same relationship must be excluded from A to prevent
self-corroboration, excluded before its temporal/acceptance validity, and
included deterministically for B under the frozen knowledge watermark.

## Vision outcomes and maturity movement

This plan advances speaker grounding, relationship/history context,
provenance, uncertainty, deterministic replay, and reusable accepted knowledge
in `VISION.md`.

- Evidence fabric: current Level 1 source-specific contracts; target Level 1
  shared deep module exercised by two product purposes.
- People/relationship loop: current Level 2 proposed-only source shadows;
  target Level 1 accepted-knowledge handoff proven on disposable state.
- Conversation-understanding loop: current Level 1 isolated retrieval/readout;
  target Level 1 retrieval of accepted relationship context through the same
  fabric.
- This plan does not claim Level 2 representative-corpus lift, Level 3
  automatic processing, or a live accepted relationship.

## Non-Goals

- No provider, mailbox, Drive, SysRAG, CRM, Graphiti, or private-corpus read.
- No live-store migration, accepted live relationship, person merge, speaker
  assignment, background worker, dashboard deployment, or provider write.
- No one-off Mail Receipts artifact loader in the Contacts route.
- No replacement of source-specific adapter behavior with a lowest-common-
  denominator schema.
- No full conversation-understanding model call or claim that the product loop
  is operational.

## Current State

`prepare_identity_evidence` already carries scopes, capabilities, `as_of`,
hindsight policy, freshness policy, provider-call/record/character budgets, and
bounded relationship hops. It invokes GWS, Odollo, and Mail Receipts adapters,
persists bounded snapshots, and then prepares an identity-specific bundle.

The provider request/result/protocol seam is currently owned by
`conversation_identity_retrieval.py`, so other consumers must depend on an
identity-named module. Relationship retrieval reads affinity profiles rather
than the reviewed relationship projection. No public source-agnostic fabric
returns accepted relationship context with an explicit knowledge watermark,
originating-conversation exclusion, and deterministic bundle hash.

## Bounded implementation packet

| Field | Control |
| --- | --- |
| Outcome | Shared evidence fabric and disposable two-loop walking skeleton |
| Owner | Primary agent in the current `plan-0037-campaign` worktree |
| Write surface | One focused product module, identity retrieval integration, focused tests, architecture/plan/roadmap/runbook docs |
| Inputs | `VISION.md`, Note 0061, conversation knowledge architecture, schema-v8 ledger/evidence contracts |
| Attempts | At most two implementation attempts before local reframe |
| Review cycles | One closed-world remediation cycle if current validation finds a blocking defect |
| Terminal condition | All acceptance criteria pass with zero live/private/provider/external effects |

Critical path is serialized: freeze docs, add one behavior test, implement the
fabric, integrate identity retrieval, add temporal/circularity/replay behaviors,
then run focused and broad validation. No parallel agent lane is used.

## Behaviors to test

1. One public `collect` interface accepts a purpose, anchors, exact scopes,
   capabilities, temporal policy, freshness policy, and explicit budgets.
2. Existing adapters can satisfy the shared interface without provider names
   entering the fabric interface.
3. Provider exceptions and out-of-scope snapshots remain bounded visible
   failures while valid snapshots persist immutably.
4. Only reviewed/accepted relationships are returned as accepted knowledge.
5. A relationship whose originating conversation equals the request
   conversation is excluded with a visible reason.
6. Relationship start/end and acceptance time obey `as_of`; hindsight requires
   an explicit policy.
7. The bundle exposes the exact identity-ledger projection watermark and a
   deterministic semantic content hash.
8. `prepare_identity_evidence` uses the shared fabric while preserving its
   existing caller-visible behavior and persisted bundle contract.

## Acceptance Criteria

- The evidence fabric is a deep module with one small source-independent
  interface and no direct dependency on GWS, Odollo, Mail Receipts, Drive, or
  SysRAG implementations.
- The same interface supports at least `people_relationship_discovery`,
  `speaker_identity`, and `conversation_understanding` purposes.
- A disposable-store test proves reviewed A-to-B relationship context is
  unavailable to its originating conversation, available to a later
  conversation, temporally filtered, cited, and bound to an exact watermark.
- Proposed, conflicted, rejected, and temporally unavailable relationship
  claims never appear as accepted knowledge.
- Reordered adapter outputs and repeated collection produce the same semantic
  bundle hash and no duplicate evidence snapshot.
- Existing identity retrieval tests remain green after provider collection and
  relationship context cross the new seam.
- Repository planning and architecture authorities describe the two loops,
  expandable source capabilities, Drive/SysRAG extension posture, and the
  anti-circularity rule.
- All effect counters for live/private/provider/Graphiti/person/speaker/
  biometric/deployment actions remain zero.

## Validation

- Focused red/green tests for the public fabric interface and disposable
  two-loop walking skeleton.
- Existing evidence-adapter and identity-retrieval regression tests.
- Python compilation for touched modules.
- Full provider-free pytest suite when focused checks pass.
- Active-only planning audit, internal-link check when available,
  `git diff --check`, CodeGraph sync/status, clean committed checkpoint, push,
  and upstream equality.

## Rollback

The new seam is behavior-preserving for existing callers. Reverting the
implementation commit restores provider collection to the identity module; the
disposable walking-skeleton state has no live rollback requirement. No schema
migration is introduced.

## Closeout Evidence

- `conversation_evidence_fabric.py` now owns the source-independent request,
  adapter, relationship, and bundle contracts. Its public `collect` seam
  supports people/relationship discovery, speaker identity, and conversation
  understanding without importing a provider implementation.
- Existing GWS, Odollo, and Mail Receipts adapters now implement that shared
  protocol directly. `prepare_identity_evidence` delegates bounded provider
  collection and accepted-relationship retrieval through the fabric while
  preserving its caller-visible identity bundle.
- Four disposable-store behavior tests prove the reviewed A-to-B handoff,
  self-corroboration exclusion, `as_of` exclusion, explicit hindsight opt-in,
  one/two-hop expansion, exact watermark, deterministic replay, immutable
  scoped provider evidence, bounded failures, and identity-facade compatibility.
- Red/green implementation evidence included initial missing-module, missing
  provider-collection, missing facade-relationship, and missing second-hop
  failures before each bounded implementation step passed.
- Validation passed: 9 focused fabric/identity tests; 52 evidence-adapter and
  identity regressions; 37 workflow/store/ledger integration tests; compilation
  of every touched Python module; and the full provider-free suite at 1,259
  tests.
- The active-only planning audit and `git diff --check` pass. CodeGraph was
  synchronized after the implementation with zero pending changes; its
  pre-existing full-reindex recommendation is recorded in the runbook.
- Live/private/provider/Graphiti/person/speaker/biometric/deployment effect
  counters remained zero. The milestone changes no live schema or runtime
  configuration and makes no representative-corpus usefulness claim.
