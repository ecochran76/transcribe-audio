# Context-assisted automatic speaker recognition after reviewed enrollment

Date: 2026-08-09

## Decision

The product must use the reviewed voice learning completed by Plan 0063 as an
input to the existing contextual speaker-identity workflow, not as a parallel
or replacement identity system. The intended future three-person flow is:

1. diarization produces recording-local speaker slots;
2. the acoustic branch independently recognizes two enrolled voices;
3. the context branch independently retrieves calendar, email, contact,
   transcript-clue, and accepted-history candidates;
4. the host joins acoustic subjects to canonical people through reviewed
   voice/person bindings and records agreement or conflict with context;
5. the remaining voice may be assigned to the remaining context-supported
   person only through a constrained, evidence-backed residual decision; and
6. an accepted decision enriches that canonical person's provenance and source
   affinities in the user-scoped conversation/contact stores so future
   conversations have better evidence.

The reviewed example contains private names and provider identifiers, so this
tracked note describes the reusable contract without copying those values.
The exact full-name correction and source bindings remain in the immutable
private Plan 0063 transition and terminal live receipt.

## What Plan 0063 now provides

Plan 0063 completed the missing learning-state application:

- canonical people and reviewed recording-local slot bindings;
- one explicit enrolled-voice-to-canonical-person binding;
- five provenance-backed biometric reference generations;
- fifteen active model profiles over twenty-three reviewed source windows; and
- provider-backed external-identity/source affinities for the reviewed people.

This is durable identity substrate. It does not yet cause a new transcript to
load the active profile inventory, score its speakers, run the contextual
candidate workflow, solve the conversation-level assignment, or persist an
automatic decision.

## Required inference contract

### Independent evidence first

Acoustic recognition and contextual retrieval run independently against the
same frozen conversation and speaker-slot set. An enrolled voice match cannot
invent a contact identity; a calendar attendee or email contact cannot prove
that the person spoke. Agreement raises support only when the acoustic subject
already has an authoritative person binding. Conflict remains visible and
caps or blocks automatic acceptance.

### Conversation-level assignment

Identity is solved across all speaker slots together. The resolver must
enforce one-to-one coverage where appropriate, preserve the possibility that a
person occupies multiple diarization labels, and retain unresolved people.
It may use a residual third-person inference only when all of the following are
true:

- the recognized voice/person bindings for the other slots meet calibrated
  acceptance thresholds;
- the context candidate set is complete enough for the conversation and has
  exactly one supported, still-unassigned person;
- transcript clues and source provenance support that person in the relevant
  slot and contain no material contradiction;
- candidate records are deduplicated to canonical people rather than counted
  as independent provider votes; and
- the policy version records why the residual result was accepted instead of
  treating elimination alone as identity evidence.

If any condition fails, the third slot remains unresolved or is routed to a
single useful review with its audio and evidence in the right place.

### Provenance-backed contact enrichment

An accepted speaker observation should add conversation-derived evidence to
the canonical person's profile and preserve every contributing provider,
account, tenant, source record, as-of time, and decision lineage. The first
implementation writes the user-scoped conversation knowledge/contact stores
and can project a non-destructive enrichment proposal for existing configured
contact sources. Direct mutation of Google, Odollo, or another external
provider remains a distinct bounded apply capability with deduplication,
field-level ownership, rollback, and effect receipts; local acceptance must
not silently overwrite provider-authoritative data.

## Product behavior to build

```text
new recording
  -> transcript + diarized slots
  -> active enrolled-profile inventory
  -> acoustic candidates per slot
  -> bounded context candidates per slot
  -> canonical-person evidence join
  -> constrained conversation assignment
       -> two bound voice matches
       -> one supported residual candidate, or abstention
  -> policy-qualified acceptance or useful review
  -> speaker observations + provenance/contact enrichment
  -> stronger evidence for the next conversation
```

The first measured milestone must replay this path on the reviewed
three-conversation corpus and then evaluate it on chronological recordings not
used for biometric enrollment. Automatic acceptance is enabled only for a
measured policy band with zero unacceptable high-confidence identity errors;
the remainder abstains without reopening routine approval rituals.

## Planning authority

[Plan 0064](../plans/0064-2026-08-09-context-assisted-automatic-speaker-recognition.md)
closed fail-safe after source-disjoint measurement found one high-support wrong
acoustic identity, zero context/combined candidates, and no actual residual
acceptance. [Plan 0065](../plans/0065-2026-08-11-speaker-identity-recovery-fresh-validation.md)
is the planned no-apply recovery and fresh-validation successor on the P09/P10
critical path. This note remains the durable architecture decision; the plans
own their bounded implementation, validation, and maturity evidence.
