# Plan 0025 | App Intelligence Speaker Preprocessing

State: CLOSED

Lane: P09

## Scope

Add a reviewed post-transcription preprocessing stage that uses App
Intelligence to identify anonymous diarization speakers from transcript clues
and bounded, source-attributed provenance.

Calendar invitee email addresses are the strongest deterministic starting
point. The host resolves those addresses and names against configured Google
Workspace and Odollo sources, gathers bounded related evidence, and gives the
transcript plus cited evidence to a Codex app-server run. The model returns
speaker proposals and the clues supporting them; it does not write speaker
assignments or mutate any external system.

The first provenance contract covers:

- Google Workspace Calendar invitees and event metadata.
- Google Workspace People reverse lookup for attendee emails and names.
- Bounded Gmail and Drive search evidence for resolved or candidate people.
- Odollo tenant contacts, CRM leads, and related log-note evidence.

Every enabled provenance source used for speaker preprocessing declares Source
Context: its owning person or organization, relationship scope, account or
tenant label, available evidence capabilities, and any authoritative
identifier role. App Intelligence must not infer those semantics from source
names or credentials.

A source with missing or invalid Source Context is excluded from speaker
preprocessing with an explicit configuration warning. The failure is scoped to
this stage and does not disable that source for established workflows outside
speaker preprocessing.

The prompt receives only the reviewed semantic Source Context needed for
reasoning. Credentials, executable commands, local paths, internal
configuration details, and unrelated tenant metadata remain host-only.

## Non-Goals

- No second-pass interpretation or full contextual readout of the
  conversation; that depends on reviewed speaker identities and is later work.
- No unattended speaker assignment, contact merge, CRM mutation, email/Drive
  mutation, deposition, or memory harvest.
- No automatic assembly of multiple recordings into one conversation; the
  identity and sidecar contracts permit it, but grouping remains explicit
  review work in this slice.
- No raw private transcripts, mail bodies, contact exports, or log-note bodies
  in Git.
- No model-controlled provider calls. The host owns provenance adapters,
  limits, redaction, run ledgers, and validation.

## Current State

`participant_identity.py` already extracts calendar attendees, prioritizes
email/name query terms, gathers Google People and Odollo contact candidates,
and emits review-gated participant identity bundles. It currently gives every
anonymous speaker the same ranked candidate pool and does not run a dedicated
LLM clue-analysis pass.

The two-phase workflow is implemented. `speaker_preprocessing_workflow.py`
prepares reviewed Clue Discovery and Identity Evaluation App Intelligence
runs; `speaker_identity_preprocess.py` validates prepared references, groups
person records, and derives host-owned scores under three versioned rubrics.
Selected-conversation API and React Speakers controls expose preparation,
capture, evidence, warnings, individual decisions, and safe ready-only
confirmation.

Normalized transcripts now carry durable conversation and recording IDs.
`conversation_processing.py` appends immutable evaluations and attributable
review decisions to a conversation-owned `.processing.json` sidecar without
rewriting transcript diarization. Required Source Context is validated for the
configured personal Google Workspace and company-owned Odollo sources.

The Codex App Server client is aligned with the installed protocol: it omits
the on-wire JSON-RPC header, completes the `initialize`/`initialized`
handshake, uses direct stdio turns, waits for `turn/completed`, reads full turn
items through `thread/turns/list`, and enforces non-blocking timeouts.

## Evidence And Reasoning Contract

Speaker preprocessing has two App Intelligence phases.

### Phase 1 | Clue Discovery

The host prepares a clue-discovery packet containing:

- durable conversation, recording, and source-artifact references;
- per-diarized-speaker utterance excerpts with stable utterance references;
- normalized calendar association candidates and attendees, with emails first;
- bounded semantic Source Context for each eligible provenance source; and
- explicit exclusions for full-conversation interpretation and direct provider
  access.

App Intelligence returns structured names, affiliations, relationships,
topics, forms of address, search terms, possible unlisted people, possible
split-speaker groups, possible mixed-speaker utterances, and cited clues. The
host validates the output before using it for retrieval.

### Host Retrieval And Person Grouping

The host performs bounded source-affinity-first retrieval across eligible GWS
and Odollo sources. It groups source records into Person Candidates using
confidence-bearing Person Link Assessments rather than exact-match rules
alone. Every Source Record retains its Source Context and tenant/account
attribution. Duplicate records belong to one evidence-independence group and
do not inflate corroboration.

The host preserves exactly the bounded snippets and metadata supplied to the
next phase as immutable Evidence Snapshots. Full message, document, transcript,
contact-export, and log-note bodies remain outside the packet.

### Phase 2 | Identity Evaluation

App Intelligence returns:

- a Calendar Association Confidence assessment separate from every Speaker
  Identity Confidence assessment;
- Candidate Matches, Unlisted Person Suggestions, Unresolved Proposals, and
  Conflicting Proposals;
- Speaker Group Proposals for one person split across diarized labels;
- Mixed Speaker Findings and cited Utterance Identity Proposals when one label
  appears to contain multiple people;
- structured Evidence Factor Assessments with factor type, direction,
  categorical strength, clue/source citations, independence group, bounded
  rationale, and credible alternatives; and
- warnings and review flags.

Calendar association, person linking, and speaker identity use separate
versioned Evidence Rubrics. App Intelligence assesses factors; the host
validates prepared references and computes the Evidence Strength Score and
Confidence Band. The score represents rubric-based evidence strength, not a
probability. Validated factors are preserved independently of derived scores
so historical assessments can be re-scored without recollecting provenance or
rerunning App Intelligence. New provenance requires an explicit re-evaluation.

The authoritative domain language is maintained in `CONTEXT.md`. Host
validation rejects invented speaker, candidate, clue, source, or evaluation
references and never applies an identity readout directly.

## Storage Contract

For this bounded slice, conversation processing metadata is stored in one
conversation-owned JSON sidecar alongside the conversation's other
transcription artifacts. The sidecar retains immutable evaluation history,
identifies the current evaluation explicitly, and owns references to
evaluation inputs and outputs, bounded evidence snapshots, validated
evidence-factor assessments, derived confidence assessments, rubric versions,
and review state. Re-evaluation appends a new record rather than overwriting a
prior assessment, and the sidecar does not retain full source bodies.

Each new conversation receives a durable opaque conversation ID when its
normalized transcript artifact is created, whether or not preprocessing runs.
Existing artifacts receive an ID lazily during first processing or migration.
The processing sidecar and derived artifacts reference that ID. File paths,
existing conversation keys, and content fingerprints remain aliases or
integrity evidence; they do not define conversation identity.

A recording and a conversation have distinct identities. The ordinary
creation path starts one conversation for one recording, while the storage
contract permits multiple recording IDs to belong to one conversation ID.

The JSON sidecar is a provisional persistence boundary. A future storage slice
will migrate transcription artifacts and processing records into a central
database under the user-scoped storage location without changing the domain
meaning of the records.

## Work Items

- Pin the built-in Codex app-server supervisor and speaker-disambiguation
  routing to the current workstation model while preserving user overrides.
- Replace the v1 single-packet prototype with reviewed clue-discovery and
  identity-evaluation packets plus strict output schemas.
- Make attendee-email exact matches the first reverse-lookup lane across
  Google People and configured Odollo tenants.
- Add read-only, limit-enforced Gmail/Drive/Calendar evidence collection for
  candidate people without retaining full message or document bodies.
- Add Odollo `crm.lead` identity candidates and relevant `mail.message`
  evidence with tenant/profile attribution.
- Extend provenance profiles with required Source Context metadata and surface
  missing or invalid declarations through configuration validation.
- Implement confidence-bearing Person Link Assessments, source-affinity-first
  retrieval, and evidence-independence grouping across duplicate records.
- Prepare both phases as reviewed App Intelligence run/prompt artifacts;
  preserve the existing approval gates for sending and capturing each turn.
- Add task-specific versioned rubrics and host scoring for calendar
  association, person linking, and speaker identity.
- Validate structured model output against prepared speaker, candidate, clue,
  source, factor, and evaluation IDs.
- Support Speaker Group Proposals, Mixed Speaker Findings, and cited
  utterance-level identity proposals without rewriting diarization.
- Persist conversation processing metadata in a JSON sidecar beside the
  transcription artifacts, retaining bounded history suitable for later
  migration to user-scoped central storage.
- Assign and preserve a durable opaque conversation ID independently of
  artifact paths and content fingerprints during normalized artifact creation,
  with lazy backfill for existing artifacts.
- Surface preprocessing state, evidence, warnings, and proposed identities in
  the conversation speaker-review workflow.
- Support individual confirm, reject, and defer decisions plus a safe
  conversation-level action that confirms only ready-to-confirm proposals;
  preserve every individual decision in processing history.
- Preserve structured reviewer corrections and rejection reasons as
  calibration outcomes without automatically training a model, mutating
  contacts, or writing graph memory.
- Allow a reviewer to confirm a provisional person even when bounded
  provenance enrichment finds no supporting record; record it as a
  reviewer-asserted identity without a fabricated evidence score.
- Require every persisted review decision to identify the reviewer, timestamp,
  decision method, proposal and evaluation IDs, and any superseded decision;
  retain an optional reviewer note.
- Dogfood on calendar-associated transcripts before considering automation.

## Acceptance Criteria

- Speaker preprocessing resolves through `codex-app-server` on the explicit
  current default model unless a user-scoped override is configured.
- A calendar attendee email is reverse-looked up before broad name or free-text
  searches and remains traceable to its calendar evidence.
- Clue Discovery runs before enriched person-specific provenance retrieval, and
  the host—not the model—executes all provider calls.
- Sources lacking valid Source Context are excluded with a visible warning
  without being disabled for unrelated workflows.
- Each anonymous speaker receives its own transcript clue set and model
  proposal rather than a shared undifferentiated candidate list.
- Calendar Association Confidence remains distinct from Speaker Identity
  Confidence, and calendar-only evidence may produce a reviewable proposal
  without pretending to identify a diarized speaker conclusively.
- Every model-supported identity claim cites prepared transcript clues,
  provenance sources, and structured evidence factors.
- Cross-source duplicate people can be grouped by contextual inference while
  retaining every Source Record and avoiding duplicate-evidence inflation.
- Split and merged diarization cases are representable without changing the
  original diarized labels.
- Invalid or invented speaker, candidate, clue, source, factor, or evaluation
  references fail validation.
- Evidence scores and bands are host-derived under named rubric versions, and
  preserved factor records can be deterministically re-scored.
- Low-confidence, conflicting, or unresolved results enter human review; no
  speaker assignment or external write occurs automatically.
- High-strength proposals may be presented as ready to confirm, but every
  speaker assignment still requires lightweight human confirmation.
- The later full-conversation contextual pass is not invoked by this stage.

## Validation

- TDD coverage for default model routing, packet construction, evidence
  ordering and limits, two-phase sequencing, strict output validation,
  rubric scoring, sidecar history, durable IDs, and review gating.
- Focused adapter tests with mocked Google Workspace and Odollo command output;
  no private fixtures in the repo.
- `python -m py_compile` for touched Python modules and tests.
- Dry-run App Intelligence ledger/prompt-packet inspection for both phases,
  proving no prompt was sent and no downstream action was executed.
- Reviewed local dogfood on at least one calendar-associated transcript with
  attendee emails, one transcript with split diarized speakers, and one
  transcript that remains unresolved or spurious.

## Closeout Evidence

- Backend regression: `295 passed`.
- Frontend production build: Vite build completed successfully.
- Calendar-associated dogfood: separate Calendar Association and Speaker
  Identity scores persisted; high-strength proposals retained review flags and
  were not auto-confirmed.
- Split-diarization dogfood: cross-label speaker grouping, mixed-label review
  flags, unlisted people, and unresolved candidates persisted without changing
  original diarization.
- Spurious/unresolved dogfood: one proposal remained explicitly unresolved at
  `0 / Low`; a high-scoring proposal with mixed-speaker flags remained pending.
- All three cases used `codex-app-server` with `gpt-5.6-sol`, host-owned
  GWS/Odollo retrieval, bounded evidence snapshots, and no external mutation.
