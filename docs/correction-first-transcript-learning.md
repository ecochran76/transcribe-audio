# Correction-first transcript learning

Plan 0072 A2 adds a non-live transcript correction layer to knowledge schema
v5. It preserves immutable raw ASR and diarization, records reviewed
span-level corrections, creates versioned normalized transcripts, indexes raw
and selected normalized text, and binds transcript-only semantic claims back
to exact raw spans.

This packet advances VISION outcomes 2, 3, 4, 6, 7, and 8 from a Level 1/2
design seam to a Level 2 replayable component. It does not establish Level 3
historical operation, review-product acceptance, or automatic learning. Those
later maturity claims require private shadow evidence and the A4-A9 gates in
Plan 0072.

## Public interface

`TranscriptCorrectionLedger` in `transcript_correction_ledger.py` owns the A2
boundary:

- `register_terminology(...)` appends one versioned terminology registry;
- `resolve_terminology(...)` applies exact scope precedence and exposes
  equal-scope conflicts for review;
- `terminology_hints(...)` returns only reviewed, applicable vocabulary with
  its exact version and content hash, without calling a provider;
- `record_raw_transcript(...)` freezes raw text, source hash, and diarization;
- `propose_correction(...)` binds a proposal to an exact raw span and evidence;
- `decide_correction(...)` appends an idempotent review decision and requires
  explicit supersession after the first decision;
- `normalize(...)` applies only the current accepted, in-scope decisions and
  creates a new normalized generation plus a two-layer reindex receipt;
- `search_transcripts(...)` returns raw and normalized matches with layer and
  generation provenance;
- `record_semantic_map(...)` accepts transcript-only topics, terms, entities,
  and questions with exact normalized spans and raw-generation lineage; and
- `record_identity_cascade(...)` permits one identity requeue per processing
  version, then records `manual_resolution_required` and refuses a third.

## Knowledge schema v5

`ConversationKnowledgeStore.migrate()` now advances ordinary stores through
v5. The additive migration creates immutable terminology versions/entries,
raw generations, correction proposals/decisions, normalized generations,
semantic maps, correction-run and identity-cascade ledgers, and reindex
receipts. It also creates a replaceable selected-normalized projection and an
FTS5 index for raw and normalized transcript layers.

SQLite triggers reject `UPDATE` and `DELETE` on every authoritative v5 ledger
table. Only the current-normalized projection and derived FTS index are
replaceable. Rollback from v5 removes v5 objects and restores schema v4 while
preserving the A1 identity/contact ledger and older transcript data.

## Terminology and correction rules

Scope precedence is conversation, project/matter, organization, domain, then
global. Only reviewed terminology versions and reviewed entries can resolve
terms or enter a backend hint bundle. Equal-scope entries with different
canonical values produce `review_required`; they are never silently ordered.
The redacted SESO fixture therefore treats `CISO` as a chemistry-scoped ASR
confusion, not a global synonym or replacement.

Raw transcript text is never rewritten. A correction records its original
text, replacement, raw span hash, scope, evidence, processing version, pass,
and cascade count. Normalization uses only the current accepted decision for a
proposal and only when its scope applies. The selected normalized generation
is the downstream readout; raw comparison and citation lineage remain
available.

Each processing version permits one `pre_identity` and one `post_identity`
pass. One material transcript/identity cascade records
`identity_requeue_required`; the second records
`manual_resolution_required`; any further cascade is rejected.

## Semantic and search lineage

Normalization replaces both search layers atomically and records the raw and
normalized SHA-256 values, index version, and layer count. Search results name
the layer, generation, conversation, and recording so callers cannot mistake
normalized text for source ASR.

The A2 semantic map is deliberately transcript-only. Claims may contain only
a label, exact normalized span, non-empty raw lineage, and optional metadata.
Calendar, contact, relationship, speaker, or other enrichment fields are
rejected; enriched drafts and accepted readouts remain later packets.

## Redacted replay evidence

The synthetic [terminology](dev/fixtures/plan-0072-a2/terminology-registry.json),
[correction replay](dev/fixtures/plan-0072-a2/transcript-correction-replay.json),
and [semantic map](dev/fixtures/plan-0072-a2/semantic-map.json) fixtures replay
a reviewed scoped term, immutable raw generation, accepted span correction,
normalized generation, exact semantic lineage, and two-layer reindex receipt.
They contain no private transcript, contact, provider, or biometric data.

## Current authority boundary

A2 is exercised only against pytest temporary stores and committed synthetic
fixtures. It did not process a historical or new private conversation, migrate
a live store, call or write a provider, activate backend hints, collect or
derive biometric material, publish a dashboard, deploy a service, schedule a
worker, or write Graphiti memory.
