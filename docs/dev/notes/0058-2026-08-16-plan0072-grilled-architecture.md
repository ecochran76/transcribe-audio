# Note 0058 | Plan 0072 grilled architecture decisions

Date: 2026-08-16

Status: accepted planning authority for Plan 0072

Boundary: architecture and future packet design only. This note authorizes no
provider access, historical processing, biometric collection, identity or
contact mutation, dashboard publication, background service, or deployment.

## Product posture

Plan 0072 builds an evidence-gathering and correction loop for speaker
identification. It does not perform authentication. Early incorrect guesses
are expected, so every observation, score, proposal, decision, merge, split,
transcript correction, and derived profile must be versioned, attributable,
replayable, and reversible.

The first live state is shadow operation. Every named speaker assignment is
reviewed. Later automatic acceptance is available only to a measured,
source-disjoint band that satisfies the promotion gate below.

## Contact baseline and person reconciliation

- Ingest every configured, authorized, read-only contact directory before
  historical speaker processing. People/Contacts, Odollo contact directories,
  and equivalent stores form the baseline; Calendar, Gmail, Drive, local
  conversation history, and configured messages or receipts are enabled for
  later targeted conversation enrichment. Account, tenant, capability, budget,
  and privacy scope remain explicit. No provider write is implied.
- Store bounded normalized identity/contact fields and opaque source
  references. Full messages, documents, CRM bodies, and provider records stay
  provider-side; retain only bounded cited snapshots needed for evidence.
- Preserve each provider/account/tenant/type/record ID as an immutable,
  source-scoped record with append-only observations. A record missing from a
  later read is not automatically a tombstone.
- Reconcile on ingest in three levels:
  1. auto-deduplicate the exact same provider/account/type/record ID;
  2. auto-link records to one provisional person only for an exact
     person-specific email or verified phone with no conflicting evidence;
  3. route name, organization, role, address, or fuzzy similarity to a ranked
     merge proposal for review.
- Shared or role email addresses never auto-link to a person. When an event
  meets the association threshold or is reviewed into participant gathering,
  its attendees become source records and participant hypotheses; they become
  provisional people only when an exact stable person-specific identifier
  passes the same no-conflict rule. Attendance never proves presence or speech.
- Clean records ingest without per-contact review. Queue conflicts, ambiguous
  merges, and material data-quality problems.
- Reviewed local overrides never overwrite provider source fields. Provider
  write-back is outside the initial release and, if later added, is a
  field-level previewed proposal workflow.
- Display name precedence is reviewed preferred name, then a deterministic
  highest-authority person-specific source. Preserve all aliases and history.

## Roles, relationships, organizations, and purpose

- One person may have many simultaneous roles and relationships. Conflicting
  assertions coexist with sources, effective times, and conflict links until
  reviewed; no last-write-wins truth field is permitted.
- Use a versioned hierarchical ontology. Top-level relationship families
  include `FAMILY`, `PROFESSIONAL`, `MEDICAL`, `LEGAL`, `COMMERCIAL`,
  `EDUCATIONAL`, `PROJECT`, and `SOCIAL`. Leaf types include `PARENT_OF`,
  `CHILD_OF`, `SPOUSE_OF`, `PHYSICIAN_FOR`, and `EMPLOYEE_OF`, with optional
  details such as father, mother, guardian, or department. Each type defines
  direction, inverse, and symmetry behavior.
- Keep two assertion forms: a typed entity-to-entity relationship edge and a
  contextual role in a conversation, event, organization, project, matter, or
  time interval.
- Organizations are hierarchical and multi-affiliation through edges such as
  `PART_OF`, `SUBSIDIARY_OF`, and `DEPARTMENT_OF`.
- Reviewers may propose new ontology types. The workflow detects near
  duplicates, requires a parent mapping, and versions accepted additions.
- Provider-declared roles, organization membership, CRM facts, and calendar
  metadata may contribute immediately as source-weighted observations.
  Transcript/model-inferred roles and relationships remain hypotheses and
  cannot corroborate themselves before review.
- Communication frequency, recency, direction, meeting count, and shared
  projects may create derived relationship observations, never identity proof.
- Relationship effective time defaults to the conversation date when that is
  the observed context. Review time is separate; an absent end date means no
  known end, not an eternal fact.
- Conversation purpose is a structured claim with primary and secondary
  purposes, project/matter/organization links, evidence, alternatives, and
  score history. Calendar-derived purpose cannot strengthen its originating
  calendar candidate.
- The speaker-review flow supports lightweight role and relationship
  confirmation. Authoritative graph editing lives in the People view.

## Acoustic subjects, samples, profiles, and deletion

- Retain unreviewed voice samples and embeddings indefinitely for now in
  private, person-unbound storage, until explicit deletion or a later policy
  change. This is a current operator decision, not a claim that indefinite
  retention is universally appropriate.
- Automatically form anonymous recurring-voice clusters. Cluster membership
  remains soft and reversible: preserve primary and alternative memberships,
  pairwise evidence, and unclustered cases.
- A person may have multiple acoustic profile families for device, channel,
  environment, language, health, or speaking-style conditions.
- Identifying one sample does not assign every cluster member. Review controls
  distinguish current speaker identity, cluster membership confirmation, and
  a separately previewed bulk assignment.
- A confirmed cluster sample re-scores related unreviewed samples and requeues
  only material changes. It never auto-assigns them.
- A reviewed identity makes quality-qualified samples eligible for a pending
  profile pool. Candidate profiles may build and evaluate automatically, but
  initial activation requires ordinary dashboard confirmation. No unreviewed
  prediction may enroll or extend a named profile.
- Person merge and split always require review, even after any future
  automatic speaker-acceptance band is enabled.
- Per-person and per-recording `do_not_use_for_biometric_matching` controls
  exclude and invalidate dependent samples, embeddings, profiles, and external
  benchmark material while preserving a minimal audit record.
- Initial deletion scope includes sample, cluster, person profile, recording,
  and person. Preview downstream effects. Delete active data immediately and
  exclude it from future backups; encrypted historical backups expire on
  their schedule unless a later design proves cryptographic shredding.
- Raw audio and biometric processing are local by default. Architecture may
  support an opt-in, bounded external challenger benchmark using pseudonymous,
  reviewed samples. Routine cloud audio processing remains prohibited until a
  challenger proves measurable lift in diarization, identification accuracy,
  abstention, latency, and cost.
- Short, noisy, overlapping, or background-only recordings receive explicit
  reason codes. Context review may continue while unusable acoustic samples
  remain excluded. Transcript-only records are context-only with audio marked
  unavailable; audio-only historical records first enter the ordinary pinned,
  replayable transcription pipeline.

## Evidence strength, calibrated likelihood, and promotion

- Show a numeric score that can rise or fall as evidence or scoring versions
  change. Preserve the original score and add immutable re-scores; distinguish
  new evidence from a new rubric or model.
- The host computes a 0-100 `Evidence Strength Score`. It measures support
  under a versioned rubric and is not a probability.
- Show separate calendar-association, person-link, contextual-speaker,
  acoustic, and combined scores. The combined score is a ranking aid; all
  pillars remain visible. Material contradiction caps the combined score and
  forces review.
- Show empirical `Calibrated Likelihood` only after at least 30 reviewed,
  source-disjoint outcomes exist in the relevant score band. Display sample
  size, interval, and evaluation version; otherwise show insufficient data.
- Device/account ownership is a meaningful positive prior because the owner is
  very likely, but not guaranteed, to be present. Filename topics, calendar
  attendees, event fit, self-introductions, roles, and semantic clues generate
  or strengthen hypotheses but do not independently bind a voice.
- Candidate sets remain open and include `not_listed`, `unresolved`, and
  `mixed_or_background` where applicable.
- Reviewed outcomes accumulate continuously, but learning is batch controlled.
  Evaluate weekly and propose a candidate rubric/model only after at least 25
  new reviewed speaker decisions or a material correction. Freeze train,
  calibration, and source-disjoint evaluation sets and promote deliberately.
- Automatic named acceptance requires at least 100 varied, source-disjoint
  reviewed speaker outcomes, at least 99% precision in the proposed automatic
  band, safe abstention/fallback, and no systematic high-strength error. Recall
  may remain modest. Evaluation review is prediction-blind; ordinary
  operational review shows the best guess, alternatives, and evidence.

## Transcript correction and semantic processing

- Preserve raw ASR and diarization. Store span-level correction proposals with
  original and replacement text, utterance/time, domain context, evidence,
  confidence/version, and review state. Accepted corrections create a
  versioned normalized transcript used downstream; no destructive rewrite.
- Maintain a versioned, scope-aware terminology registry containing canonical
  spelling, expansion, definition, aliases, ASR confusions, pronunciation
  hints, supporting conversations, dates, and domain/project scope.
- `CISO` to `SESO` for semi-epoxidized soybean oil is a SoyLei/chemistry ASR
  confusion, not a global replacement and not a synonym.
- Scope precedence is conversation, project/matter, organization, domain, then
  global. Equal-scope conflicts require review.
- Show corrected text by default with marked spans and raw comparison. Search
  both raw and normalized text; summaries and retrieval use the selected
  normalized version while citations retain raw lineage.
- Reviewed terminology may feed backend prompts, hotwords, or vocabulary hints
  where supported, with the terminology version recorded. A terminology change
  first re-normalizes existing transcripts; targeted re-transcription is for
  poor confidence, high correction density, diarization/timestamp defects, or
  high-value conversations.
- Use two bounded correction passes: pre-identity with domain/acoustic/general
  context, then post-identity with reviewed person/role/organization/project
  context. If the second pass materially changes identity evidence, invalidate
  and requeue identity once. A second cascade in the same processing version
  stops as `manual_resolution_required`.
- Unreviewed corrections may generate retrieval candidates and alternatives,
  but cannot strengthen corroborating identity evidence until reviewed or
  policy-qualified.
- Produce three readout stages: transcript-only semantic map; enriched draft
  using calendar, contacts, relationships, history, and speaker hypotheses;
  accepted readout after review. The first two are provisional processing
  artifacts; the third is reviewed knowledge.

## Dashboard and review semantics

- Use the existing Authelia-protected route as the sole authentication gate at
  initial launch. Do not add Google OAuth, local login, or step-up
  authentication at this time. Preserve the dashboard's existing request
  protections plus stale-write rejection and non-disclosure of raw paths or
  unrestricted audio; this plan adds no second security layer.
- Provide separate `Identity Review` and `People` tabs. Identity Review is
  conversation-first with cross-conversation cluster context. People is the
  authoritative view of canonical/provisional people, source records, aliases,
  organizations, roles, relationships, anonymous clusters, profiles, and
  score/correction/merge history.
- The People view may visualize at most two relationship hops in the initial
  release. Tables and explicit forms remain the authoritative editing surface;
  no unrestricted graph editor is included.
- Every relevant view displays the actual original recording filename.
- Show up to three calendar candidates plus `no matching event`, with search
  for lower-ranked candidates. Candidate-event attendees remain evidence
  snapshots; broader contact gathering/participant hypotheses begin only when
  the candidate meets the minimum association threshold or is reviewed.
- Record mentioned people as `mentioned_person` hypotheses, distinct from
  participants and speakers.
- Record grouping is proposal-only. Exact media hashes auto-link source
  occurrences while preserving filenames and paths; near-duplicates and
  overlaps require review.
- Support whole-label fast decisions plus utterance-level corrections. A queue
  item is complete only when every speaker has an explicit disposition;
  `unresolved` is valid.
- Requeue an unresolved or reviewed item only after a material new exact fact,
  confirmed cluster, corrected event/transcript, meaningful score delta, or
  new rubric/model threshold crossing. Preserve the prior decision.
- Freeform comments are immutable semantic-correction input. App Intelligence
  may propose structured derivatives in a secondary correction queue; only a
  confirmed structured derivative becomes a learning label.
- Portable review exports/imports are schema-versioned, content-hashed, bound
  to exact review authority, and reject stale or altered decisions.
- Default priority is new unreviewed work, high-strength quick confirmations,
  high-downstream-impact cases, high-learning uncertainty, then historical
  backlog. Filter by date, confidence, person, cluster, calendar, and impact.
- The scorecard reports correctness, corrections, unresolved/not-listed,
  precision/recall, high-strength errors, calibration, pillar contribution,
  clusters resolved, duplicates prevented, review time, pipeline yield,
  provider failure, and backlog.

## Processing stages, budgets, and launch gates

- Start asynchronously only after transcript artifacts stabilize. The watcher
  must not run provider/model enrichment inline. New conversations process
  immediately; historical backfill is oldest-first on a separate throttled
  queue.
- Provider failures are partial. Retry one bounded transient idempotent read,
  then continue with visible missing-provider state. Recovery appends evidence,
  re-scores, and requeues only material changes.
- Enforce hard daily provider/model budgets and reserve capacity for new
  conversations. Pause rather than silently reducing evidence. Above 500
  actionable conversations, continue cheap normalization, metadata, sample
  extraction, and clustering while throttling expensive enrichment.
- Launch in three stages:
  1. live shadow: process and populate the Authelia-protected queue, but apply
     no identity/contact/profile decisions;
  2. reviewed learning: explicit decisions update local people,
     relationships, assignments, and candidate profiles;
  3. policy-qualified automation: enable only the accepted automatic band.
- Stage 2 requires at least 25 historical conversations plus seven days of new
  conversations, replayable queue state, no cross-tenant or destructive
  correction failures, and a usable review workflow. High accuracy is not a
  Stage 2 prerequisite because all effects are explicit.
- Reprocessing after a new rubric/model first re-scores existing factor
  records, then reprocesses stale, unresolved, contradictory, low-strength, or
  materially affected cases. Full replay is bounded maintenance.
- Initial implementation order is: identity/contact/relationship/ontology
  ledger and baseline directory ingestion; terminology/transcript correction
  and semantic map; voice custody and anonymous clustering; evidence
  supervisor and confidence history; APIs plus Identity Review and People;
  shadow processing; reviewed learning; evaluation and promotion.
- No external notifications are part of the initial release; the dashboard is
  the only review surface.
