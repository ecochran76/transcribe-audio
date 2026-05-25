# Plan 0014 | Contact Search Workbench

State: CLOSED

Lane: P09

## Scope

Build the Contact Search Workbench as the operator-facing surface for selecting,
finding, merging, and annotating participant contacts for one conversation.

The workbench should sit inside the conversation Context workflow first, with a
future Contacts navbar page acting as an aggregate/index view rather than the
primary review path. It must make already-fetched candidates instant to select,
show selected contacts regardless of active search text, preserve provenance
evidence, and allow both operator and App Intelligence decisions through the
same local state contract.

This plan turns the current contact picker into a durable workbench with four
clear phases:

1. Cached candidate hydration from deterministic provenance.
2. Instant local staging for select/exclude/clear/merge/split.
3. Explicit batch persistence to user-scoped runtime state.
4. Optional reviewed refresh/search against configured source providers.

## Non-Goals

- No unattended writes to Google Contacts, Odoo/Odollo, Outlook, Graphiti,
  Drive, or deposition targets.
- No live provider call on every checkbox, Use, Exclude, or Clear action.
- No LLM-only identity claim treated as authoritative contact selection.
- No raw contact exports, private calendar feeds, tenant credentials, or raw
  transcript content in tracked repo files.
- No broad CRM enrichment workflow in this slice; enrichment can be represented
  as provenance evidence or a future reviewed action, not automatic mutation.
- No global contact dedupe database beyond the user-scoped candidate/contact
  cache needed for transcript workbench review.

## Current State

Plan 0012 closed the speaker deanonymization and participant-aware context
milestone. `participant_identity.py` builds a participant identity bundle from
calendar attendees, readout participants, local contacts, configured `gws`
People/Contacts provenance, configured Odollo contacts, and operator hints.

Plan 0013 closed the shared user-scoped provenance configuration milestone.
Calendar, contact, Odollo, `msgcli`, and iCalendar source definitions now live
under `~/.local/state/transcribe-audio/provenance.config.json` by default, and
the CLI, watcher, API, and React console resolve the same profile.

The completed context workbench shows proposed contacts, cached/source search,
manual contacts, natural-language instructions, selected/excluded state,
dedupe clusters, merge/split controls, source evidence, relationship-affinity
ranking reasons, and cache/merge status in one conversation-scoped surface.
Use/Exclude/Clear interactions update local React state immediately and persist
only through Save choices, automatic final-preview flush, reviewed merge/split,
or explicit source refresh.

Cached search remains the default and does not call external providers.
Configured-source refresh is explicit, writes user-scoped cache/job records,
and keeps provider calls read-only. Relationship-affinity refresh writes a
user-scoped compact cache using local transcript/contact decision history and
calendar overlap metadata; broad-name searches rank by deterministic text,
conversation, affinity, source-quality, and operator-history scores with
visible reasons. App Intelligence decisions use the same local batch contracts
as operator decisions.

## User Experience Contract

The first screen of the workbench is the selected conversation, not a generic
contacts landing page.

The contact panel should have stable regions:

- **Selected strip**: always visible, even when search is active. Shows selected
  canonical people as removable chips. Selection state updates immediately.
- **Search bar**: filters cached candidates locally as the user types. It never
  hides the selected strip.
- **Candidate grid**: shows matching candidates with label, email, source,
  confidence, merge count, and concise provenance badges.
- **Source controls**: explicit Refresh buttons for cached sources such as
  calendar attendees, `gws`, Odollo tenants, `msgcli`, and local contacts.
  Refresh is the only normal UI path that may call external provenance tools.
- **Manual add**: name/email form plus natural-language instruction/context
  text area. Manual contacts are staged immediately and persisted in the same
  batch as other choices.
- **Merge review**: shows canonical contact, merged emails/names, source
  evidence, and Split/Merge controls when deterministic dedupe is uncertain.
- **Save state**: Save choices persists all staged changes in one batch. Final
  readout/deposition preview automatically flushes pending local choices first.

Selection interactions must be UI-local:

- clicking Use, Exclude, Clear, selected chips, or merge/split controls updates
  React state synchronously;
- no backend request is required to select already-fetched candidates;
- backend persistence is explicit batch save, automatic final-preview flush, or
  explicit source refresh;
- unsaved local choices show a small dirty-state indicator and survive search
  text changes while the modal remains open.

## Search Semantics

Search is split into two modes.

### Cached Search

Default typing searches already-known state only:

- participant identity bundle candidates;
- selected/excluded candidates from prior decisions;
- local `contacts` table rows;
- manual contacts staged in the current session;
- recently refreshed provider results stored in the participant identity cache
  or future contact-search cache.

Cached search runs locally in the browser for the fetched candidate list. If the
candidate list is larger than the current UI payload, the backend cached-search
endpoint may be used, but it must read only local cache/state and must not call
external provider CLIs.

Selected contacts remain visible outside the filtered grid. A search with zero
grid hits should say that no cached contacts matched and offer explicit source
refresh/search actions.

### Source Refresh / Expanded Search

Provider-backed source searches are explicit actions:

- `Refresh calendar attendees`
- `Refresh GWS contacts`
- `Refresh Odollo tenant contacts`
- `Search all configured sources`
- future `Search msgcli/mail contacts`

These actions may be slow and should show progress, source status, warnings,
and timeout behavior. Results are written into user-scoped cache/state, then
the candidate grid updates from cache. The user can continue selecting already
fetched candidates while refresh jobs run.

## Candidate Model

Every candidate should be normalized into a compact UI record:

- `contact_id`: stable local id for selecting this candidate.
- `canonical_key`: optional alias/person key for same-person merge.
- `dedupe_key`: deterministic grouping key.
- `merge_keys`: secondary email/name/alias keys.
- `label`, `email`, `organization`, `role`, `phone` where known.
- `source_type`: calendar attendee, `gws_contact`, `gws_other_contact`,
  `gws_directory_person`, `odollo_contact`, `msgcli_contact`, local contact,
  manual contact, or operator participant hint.
- `source_profile`: user-scoped profile/tenant/source id.
- `confidence`: match score with reason, not an opaque ranking.
- `evidence`: attendee ids, provider ids, matched query terms, calendar event,
  tenant/profile, manual note, or operator/App Intelligence decision.
- `merged_sources`: all source rows that contributed to the candidate.
- `review_state`: selected, excluded, staged, saved, conflict, or needs review.

The UI should never display raw secret refs, iCalendar URLs, tenant credentials,
or large raw contact payloads. It should display source labels and stable
redacted identifiers.

## Relationship Affinity Ranking

Search ranking should use communication recency and frequency as explainable
ranking signals, especially for broad first-name queries such as `chris`.
Affinity is not identity proof: a recent/frequent correspondent should rank
higher in search results, but the system should not claim they attended or
spoke in a meeting without attendee, transcript, operator, or other reviewed
evidence.

Each candidate may carry a `relationship_affinity` block:

- `last_contacted_at`: latest email, message, calendar, or reviewed transcript
  interaction timestamp known from configured read-only sources.
- `interaction_count_30d`, `interaction_count_90d`, and
  `interaction_count_365d`: bounded counts by normalized person/contact key.
- `last_calendar_overlap_at` and `calendar_overlap_count_365d`: calendar
  co-attendance signals.
- `message_count_30d` and `message_count_365d`: email/message signals from
  `gws` Gmail, future `msgcli`, or other configured mail provenance.
- `transcript_overlap_count_365d`: prior selected-speaker/contact or
  participant-context appearances in the local transcript store.
- `prior_selected_count` and `prior_excluded_count`: operator/App Intelligence
  decision history in this workbench.
- `evidence`: compact reason strings such as `emailed 3 days ago`,
  `6 calendar overlaps`, `selected before`, or `no recent communication`.

The affinity cache should live under user-scoped runtime state, not tracked
repo files. It should store compact contact ids, source profile labels,
timestamps, counts, and redacted evidence summaries. It should not store raw
message bodies, private contact exports, full attendee lists unrelated to the
conversation, attachment bytes, or provider credentials.

Initial source coverage should be incremental:

1. Local transcript/contact decision history from `~/.transcripts`.
2. Calendar co-attendance from transcript event metadata and configured
   calendar provenance.
3. `gws` Gmail/People metadata if the active profile exposes read-only message
   search/count access.
4. `msgcli` mail metadata after its installed command surface can provide
   bounded sender/recipient/date counts for a query.
5. Odollo communication metadata only when enabled as read-only provenance and
   clearly separated from contact identity.

The scoring model should be deterministic and inspectable. A first pass can use
weighted sub-scores:

- `text_score`: exact name/email prefix, full-name match, token overlap, or
  substring match.
- `conversation_score`: current attendee, same domain/org, participant hint, or
  source-context relevance.
- `affinity_score`: recency decay plus log-scaled communication frequency.
- `source_quality_score`: reviewed local contact, Google Contacts, Other
  Contacts, directory, Odollo, or manual source.
- `operator_history_score`: prior selected/excluded/merge/split decisions.

For broad search results, sort by final `rank_score`, then confidence, then
label. The UI should show the strongest two or three ranking reasons next to
each candidate so the operator can understand why `Chris A` outranks
`Chris B`.

## Merge And Dedupe Policy

The resolver should work hard to merge obvious same-person contacts while
keeping weak matches reviewable.

Automatic merge is allowed when one or more strong deterministic keys agree:

- exact normalized email or configured email alias;
- Gmail plus/dot canonicalization where configured;
- configured canonical alias in user-scoped provenance contact settings;
- strong two-token-or-better full-name match, including `Last, First`
  normalization;
- provider resource id already linked to a local reviewed contact.

Automatic merge is not allowed for weak evidence alone:

- single-token names such as `Michael`;
- only shared email domain;
- only company/organization match;
- only loose transcript text token overlap;
- low-confidence Odollo or Other Contacts fallback hits.

The workbench should expose merge clusters:

- show canonical display label and primary email;
- show alternate names/emails/source rows;
- allow operator Split when the automatic merge is wrong;
- allow operator Merge when two candidates are clearly the same person;
- persist merge/split decisions as user-scoped local policy or contact alias
  records, not repo files.

## State And Persistence

Runtime state remains user-scoped:

- `~/.local/state/transcribe-audio/participant-identity-bundles/` for cached
  identity bundles.
- `~/.local/state/transcribe-audio/conversation-context-contact-selections/`
  for selected/excluded/manual/staged contact decisions.
- `~/.local/state/transcribe-audio/provenance.config.json` for source
  definitions, canonical aliases, and participant hints.
- `~/.transcripts/transcripts.sqlite3` for local reviewed contacts and
  persistent contact rows.

State layers:

1. **Fetched state**: API payload from the conversation detail.
2. **Staged state**: local unsaved React state for instant interactions.
3. **Saved state**: batch-persisted decisions under user-scoped runtime state.
4. **Learned state**: operator-approved aliases/merge/split rules that affect
   future deterministic candidate generation.

Saved decisions must include actor type: `operator` or `app_intelligence`.
App Intelligence can propose selections, exclusions, instructions, or merges,
but those proposals remain local review records unless a workflow explicitly
allows automatic save.

## API Contract

Existing endpoints should be refined rather than replaced:

- `GET /api/conversations/<id>` returns the workbench's initial fetched state.
- `GET /api/conversations/<id>/context-workbench` returns the same contact,
  instruction, context, warning, and source status state without full document
  reload when needed.
- `GET /api/conversations/<id>/context-workbench/contact-search` searches
  cached local state by default. It must not call external providers unless an
  explicit `mode=refresh` or dedicated refresh endpoint is added.
- `POST /api/conversations/<id>/context-workbench/contact-selection-batch`
  persists select/exclude/clear/manual actions in one local batch.

New/refined endpoints:

- `POST /api/conversations/<id>/context-workbench/contact-refresh/preview`
  returns which configured sources would be searched and why.
- `POST /api/conversations/<id>/context-workbench/contact-refresh`
  refreshes selected configured sources with approval token or explicit
  operator action.
- `GET /api/conversations/<id>/context-workbench/contact-refresh/<job_id>`
  reports source-level progress, warnings, counts, and cache path.
- `POST /api/conversations/<id>/context-workbench/contact-merge-batch`
  records merge/split/alias decisions locally.
- `POST /api/conversations/<id>/context-workbench/contact-affinity/refresh`
  refreshes bounded recency/frequency metadata for the active query and source
  set.
- `GET /api/conversations/<id>/context-workbench/contact-affinity`
  returns cached, redacted affinity facts for current candidates.
- `POST /api/conversations/<id>/context-workbench/instructions` continues to
  save natural-language operator context.

All endpoints should report:

- `will_execute_external_action`;
- `will_perform_external_write`;
- source counts and warnings;
- cache status: hit, miss, stale, refreshing, or error;
- redacted source profile labels.

## App Intelligence Contract

App Intelligence receives the same compact workbench bundle an operator sees:

- selected contacts;
- unresolved candidates;
- excluded contacts and reasons;
- merge clusters;
- calendar attendee evidence;
- operator instructions;
- source refresh status and warnings.

It may return structured decisions:

- select candidate;
- exclude candidate with reason;
- add manual candidate;
- request source refresh;
- propose merge/split;
- ask for operator clarification;
- add natural-language context/instructions.

The backend should validate those decisions through the same batch endpoints as
operator actions. It must not call external write-bearing systems from contact
workbench decisions.

## Implementation Slices

1. **Workbench state contract audit**: document current conversation payload,
   contact-selection files, local contacts table, identity bundle cache, and
   missing fields for the UI candidate model.
2. **Cached search contract**: make the backend contact-search endpoint
   explicitly cache-only by default, add `cache_status`, and ensure selected
   contacts are always included in search payloads.
3. **UI layout pass**: separate selected strip, search, candidate grid, source
   controls, manual add, merge review, instructions, and save state.
4. **Source refresh path**: add preview and explicit refresh endpoints for
   configured `gws`, Odollo, calendar, and future `msgcli` sources, with
   source-level progress and no secret leakage.
5. **Affinity ranking path**: build a user-scoped relationship-affinity cache,
   collect bounded local/calendar/message counts, compute deterministic
   `rank_score`, and expose compact ranking reasons in the API/UI.
6. **Merge/split decisions**: persist operator-approved alias/merge/split
   choices and feed them into deterministic candidate generation.
7. **App Intelligence handoff**: define structured decision schema for
   contact-workbench actions and validate it through batch endpoints.
8. **Browser smoke and performance guard**: add a rendered smoke proving
   selection causes zero contact-selection network requests until Save, search
   keeps selected contacts visible, and source refresh is explicit.

## Acceptance Criteria

- Selecting, excluding, clearing, or removing an already-fetched contact updates
  the UI with no backend request and no visible lag.
- Selected contacts remain visible while search is active.
- Search text filters cached candidates correctly and does not erase staged
  selections.
- Backend cached search does not call `gws`, Odollo, `msgcli`, calendar, or any
  external provider unless the operator chooses an explicit refresh action.
- Save choices persists all staged selection changes in one batch request.
- Final preview queueing flushes pending contact choices before queueing.
- Manual contacts and natural-language instructions are visible in the same
  workbench state used for readout handoff.
- Broad-name searches such as `chris` use recency/frequency affinity signals
  when available, show ranking reasons, and still distinguish ranking from
  identity proof.
- Merge clusters show provenance from every contributing source and support
  reviewed split/merge decisions.
- App Intelligence can use the same local batch contracts as the operator.
- API responses and UI never expose private feed URLs, credentials, raw contact
  exports, or tenant secrets.

## Validation

- `python -m py_compile transcript_api.py participant_identity.py
  provenance_config.py`.
- Focused `pytest` coverage for cache-only search, batch contact selection,
  selected-contact inclusion during search, merge/split persistence, and source
  refresh preview.
- Focused `pytest` coverage for relationship-affinity scoring: recent
  communication outranks stale communication for the same text score, frequent
  communication outranks one-off weak matches, prior exclusions reduce rank,
  and missing affinity does not hide exact text matches.
- `npm --prefix frontend run build`.
- Browser/CDP or `agent-browser` smoke for:
  - page loads selected conversation context workflow;
  - selected strip is visible before and after entering search text;
  - clicking Use/Exclude/Clear causes zero selection network requests;
  - Save choices causes exactly one batch request;
  - search for a known candidate filters the grid but keeps selected contacts
    visible;
  - explicit source refresh shows progress and does not leak secrets.
  - an affinity-backed `chris` search shows communication recency/frequency
    reasons and orders recent/frequent contacts above stale equal-text matches.
- Manual dogfood pass on:
  - one calendar-backed transcript with deterministic attendee matches;
  - one transcript requiring operator manual contacts;
  - one transcript with same-person multi-email merge candidates.

## Closeout Evidence

- `python -m py_compile transcript_api.py participant_identity.py
  provenance_config.py` passed.
- `.venv/bin/python -m pytest -q` passed with 241 tests.
- `npm --prefix frontend run build` passed.
- `git diff --check` passed.
- Restarted `transcripts.service`; direct and ingress health checks returned
  HTTP 200 with `status: ok`.
- Live API smoke against conversation `6e8eee4f19a1d5a9b23f` verified
  `contact-affinity/refresh` ranks `Chris Williams` first for `chris` with
  `contacted 13 days ago`, `5 calendar overlaps`, and `5 interactions in 90d`
  reasons, and follow-up cache search returned `affinity_cache_status=hit`.
- `agent-browser` smoke against
  `http://transcripts.localhost/?selected=6e8eee4f19a1d5a9b23f&conversation=1&workflow=context#raw-audio`
  verified the selected strip stayed visible during search, `Search sources`
  and `Refresh ranking` rendered, ranking reasons appeared in the candidate
  grid, clicking Use caused zero contact-selection network requests, and Save
  choices caused exactly one `/context-workbench/contact-selection-batch`
  request. The temporary smoke selection was cleared afterward.
