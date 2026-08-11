# Plan 0064 | Context-assisted automatic speaker recognition and enrichment

State: CLOSED

Checkpoint: Plan 0063 terminal live learning state is installed and replayed;
P0 is terminal on immutable private corpus
`f24722166f5f147ee6b26b13bba87d1f12ab60530c3ca0add3d8687046c5675a`
with zero identity or external effects; P1 and P2 are terminal and replayed;
P3 is terminal and replayed; all 39 literal P4 human-gold decisions are frozen
and measured; the source-disjoint and reviewed-development gates failed; Plan
0064 therefore closed fail-safe at P4 with P5/P6 withheld and zero effects

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
replays one completed live apply that created six canonical people, nine
reviewed slot bindings, one active voice/person binding, five references,
fifteen profiles, and twenty-three selected enrollment sources. Those are
Plan 0063 apply deltas, not the complete governed biometric inventory.

Fresh read-only P0 discovery found seven active reference heads and twenty-one
active model profiles across seven acoustic subjects and three governed
adapters. The reference store also retains two superseded generations, and
the profile store retains six superseded profiles. The active references bind
sixty-three source claims over eleven distinct recording hashes. Five active
subjects use canonical-person IDs directly, one older acoustic subject has an
accepted explicit canonical-person binding, and one older active subject is
not canonically bound. Its three profiles remain visible for lifecycle and
drift accounting but are ineligible to emit a person candidate.

The transcript document store currently contains 390 transcript artifacts:
371 retain a source-recording blob, 320 have non-empty diarized utterances,
276 have at least two diarized speaker labels, and 312 retain local calendar
event context. The schema-v3 conversation sidecar has no projected
conversations yet, so P0 must enumerate the authoritative document/blob store
rather than treating `knowledge_conversations` as the historical corpus. Both
transcript services are `active/running` with zero restarts.

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

### P1 and P2 terminal checkpoint

- P1 preview `8dc9ef4e6cde703869480409b9f7f0f3bda1489636671dfb27555c04369006e2`
  scored all 39 speaker slots in the 12-recording P0 cohort against the active
  governed profile matrix. Its immutable evidence content is
  `b6a87465ddcdef0a781554c56cf1fe8bdad6b86c8ac7b5ae2905300db320bbb4`,
  and its terminal receipt content is
  `d0e2441adbbaadf22fd401946a36a858317d3f0f36a26a72607e7f2973407a30`.
  The run completed 819 model/profile trials and produced 12 multi-model
  candidates, one single-model review, and 26 abstentions.
- P2 preview `d6014903bf89a4398d3fd392b9feae65d9105c093f21264d954a2649c5253a23`
  preserved the same 12 recordings and 39 slots. Its terminal receipt content
  is `50a7f4fd15b8c65c1faf4628309e72796661ac7760651eb7c9666d9117d9bd6b`.
  Four contextual workflows completed, while eight became reason-coded
  unavailable. The final four use `provider_routes_unavailable` after the
  primary route reached `usageLimitExceeded` and the configured fallback
  returned `credit_balance_exhausted`; no later retry occurred.
- P2 validated and recorded the workflow-owned transcript identity hydration
  in bridge content
  `fc0f3a506492741623516f5aff7d7a5674f797a72a4f1bbb3aac18480cdae222`.
  The bridge permits only synchronized conversation ID, recording ID, schema,
  and artifact-hash changes on the selected transcript rows. It rejects any
  semantic cohort drift.
- Both packets replay idempotently. Speaker assignment, enrollment, profile,
  knowledge, provider, and external-write counters remain zero.

### P3 and P4 review checkpoint

- P3 preview `2ec73512fc8122efd79201471473b9ac6f5e7f1197f4a5a9c644eebe1537a55b`
  binds the exact P1 and P2 receipts over all 39 slots. Resolution content
  `2f55e7adb9a48e44073e402bd3bc802ddc10c518cdb3d158d00f5a5058492dcb`
  and receipt `b630d12d6ce21804d8cd0ad4e24ff6f22730ad365c0ea271f9e2db6d661d115e`
  replay idempotently.
- Acoustic-only produces 12 candidates, one review, and 26 abstentions. The
  combined and residual conditions produce 13 reviews and 26 abstentions,
  with zero candidates and zero residual acceptances because P2 produced no
  canonical context candidate. This preserves the independent evidence
  pillars and does not manufacture agreement.
- P4 review v2 receipt
  `22bbfd4eb3559801af97dab4f94dbb1d79c559820e1297a74299e77a79612680`
  binds 39 private speaker-specific clips, review authority
  `e2df49c9fb081ea50d17d77a09b8c26a577b0e6f3cb3b64d8acb580e7b8a0daf`,
  and the 12-source original-recording filename set
  `97eeb3ed600491117ebb889ce6476735575050d75f48428cca36a34959764ca5`.
  Each card shows only the basename from its hash-bound source media path; no
  private parent directory is exposed. The review hides model predictions and
  requires one literal decision for every slot before it exports the unchanged
  decision contract.
- Browser checks found 39 cards, 39 original-filename labels, 39 playable audio
  controls, and 39 selectors; the export stays disabled at 0/39 and becomes
  enabled only at 39/39. Desktop and 390-pixel mobile screenshots show the
  filename and question adjacent to the correct audio and decision control,
  with no mobile horizontal overflow. Authenticated Previews artifact
  `dbfba8a5b884` supersedes the filename-free review artifact.
- The operator supplied all 39 literal decisions against the filename-free v1
  authority `6df988b11c152b78f9da59ab6d2324516082196d70d0340ecba2298051582f67`.
  Exact bridge content
  `031ce0f0f2864e3c34a1ff081644629557395f4a948f06b0c2d9c2f3179ea67d`
  proved that v2 changed only the review authority hash by adding the 12
  display-only basenames. It rebound the unchanged 39 decision rows, hash
  `707ce0b9bb0d2b147b4ec09d8325e56bfa78c2e13541202ab1a6e654b8f5f58d`,
  to v2 authority `e2df49c9fb081ea50d17d77a09b8c26a577b0e6f3cb3b64d8acb580e7b8a0daf`.
  The gold contains 11 canonical-person, 24 not-listed, and four unresolved
  decisions. No model output, decision, or action counter changed.

### P4 measurement and reviewed-development gate checkpoint

- `speaker_identity_plan0064_p4_measurement.py` accepts only the exact
  authority-bound 39-row decision contract. It rejects missing, reordered,
  duplicate, extra, or non-allowlisted decisions and freezes private
  human-gold, four-condition metrics, a terminal gate, and an exact replay
  receipt. Human `unresolved` decisions are measured as unverifiable rather
  than silently counted as correct or wrong.
- The P4 gate is intentionally non-vacuous. It requires complete gold, zero
  high-support wrong or unverifiable identities, at least one correct combined
  acceptance, at least one correct acceptance whose reason is the specific
  two-known-plus-one-independently-supported-residual rule, complete lineage,
  and a passing reviewed-development gate. It never grants apply authority.
- Reviewed-development preview
  `a2a3e65cd1a35531348c179583c0fd667af98ee01b7a1e6636c8f1911412542b`
  replayed the exact three Plan 0063 conversations, ten speaker slots, and
  current twenty-one-profile inventory. Evidence content
  `d5097df904f25989395eb02facd62e2e0df9743275a90f7fc230d073c5c9a7c2`
  and receipt content
  `878c1cbff9b6e6bd280cb780757d2dc9bdda1ca0067197079659108dfb5275db`
  are immutable and replay exactly.
- The reviewed replay produced three correct combined candidates, five
  reviews, two abstentions, and zero high-support wrong identities. Its
  residual-policy view has the same three pillar-agreement candidates but zero
  candidates produced by the actual residual rule. Gate content
  `cb942cbd9efea0bdfc64a633a8e8aa179149d6ed67beb1dbd780fb01e132b0c1`
  therefore records `quality_gate_passed=false`.
- Private P4 measurement content
  `baa26f05bee01165ddf9f5dd77de39b47cc1da9be71fbda5568a73673f8c09c7`
  evaluated all 39 rows and replays exactly. Acoustic-only produced 12
  candidates: 11 correct and one wrong, including one high-support wrong
  identity; precision is `0.916667` and recall across known gold is `1.0`.
  Context produced zero candidates. Combined and residual-policy each produced
  zero candidates, 13 reviews, and 26 abstentions, with zero residual-rule
  acceptances.
- Terminal content
  `f178f4187d0e8c877362310563738144854508fb4acba8b3ea227b79e829d5b6`
  records `withhold_p5`. The failed checks are zero high-support wrong,
  combined and residual acceptance/lineage, the specific residual-rule
  acceptance, and the reviewed-development gate. P5 and P6 were not run; all
  assignment, enrollment, profile, knowledge, provider, and external-write
  counters remain zero.

### Terminal disposition

Plan 0064 execution is closed because its bounded P0-P4 campaign reached a
deterministic terminal gate and no in-plan apply authority remains. Product
acceptance and the target maturity movement were not achieved. A successor
must treat these 39 decisions as development/hindsight evidence, correct the
acoustic false acceptance and missing context/residual evidence, and validate
the revised policy on a fresh source-disjoint cohort before requesting any
local acceptance or enrichment authority.

### P0 corrected freeze contract

- Inventory the complete live governed stores. Keep Plan 0063-created counts
  separate from whole-store active, superseded, and withdrawn counts.
- Reconcile every active reference head, eligible descendant, active model
  profile, model revision, source generation, and state receipt. Any drift or
  repeated profile identity fails closed.
- Resolve acoustic subjects to canonical people only by direct canonical
  person identity or an accepted explicit voice/person binding. Preserve
  unbound subjects in the inventory with `missing_canonical_person_binding`;
  they cannot become identity candidates.
- Build the development exclusion from every source claim reachable from an
  active reference, not only the twenty-three sources added by Plan 0063.
  Exclude the whole recording hash and retain the exact source windows so
  partial-window reuse cannot masquerade as unseen evidence.
- Continue the established oldest-forward chronological corpus by recording
  time and stable document ID. Exclude recordings already exposed through
  prior speaker-identity gold, review, or prediction artifacts from the unseen
  denominator; do not silently restart at chronological rank one.
- Retain one reason-coded row for every recording considered through the
  twelfth eligible selection. Structural eligibility requires a readable,
  hash-matching transcript, a readable hash-matching source recording,
  non-empty diarization with at least two labels, no repeated recording hash,
  and no development overlap. Calendar context is classified independently as
  locally available or requiring bounded P2 retrieval; lack of a calendar
  event alone does not erase an otherwise eligible conversation.
- Freeze at most twelve selected recordings in chronological order. P0 reads
  no gold, scores no speaker, performs no historical reprocessing, and records
  zero assignments, enrollments, provider writes, Graphiti writes, or other
  external effects.

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
- Active profiles whose acoustic subject lacks a reviewed canonical-person
  binding remain inventoried but can only yield an unavailable/abstention
  state, never a person candidate.
- The resolver operates over the full speaker-slot set and supports both
  one-to-one coverage and same-person multi-label cases.
- Residual assignment requires two accepted known-person bindings, exactly one
  independently supported remaining canonical candidate, relevant transcript
  support, complete provenance, and no material contradiction.
- Ambiguous, incomplete, duplicate, or conflicting cases abstain and produce a
  useful audio-linked review rather than a context-free question.
- Development sources and recordings previously exposed through identity gold,
  review, or prediction never enter evaluation metrics as unseen evidence.
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

The following product outcome remains unmet; terminal fail-safe closure above
does not claim it. Future work is complete only when eligible conversations
use the reviewed biometric identities and the existing contextual workflow together;
the measured high-support band can correctly accept known speakers and a
context-supported residual speaker without elimination-only guessing; accepted
local observations enrich canonical/contact provenance for later retrieval;
all other cases abstain usefully; and external provider writes remain zero
pending a separately validated write-back plan.
