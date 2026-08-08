# Plan 0058 | Review-surface reliability

State: CLOSED

Checkpoint: P4 complete; terminal decision `complete`

Lane: P10

## Scope

Build and browser-prove a reusable, deterministic human-review surface that
collects one explicit allowlisted speaker-identity decision per card, exports
the exact line-oriented answer contract already accepted by the strict Plan
0057 parser, and loads audio on demand without triggering the public preview
path's concurrent-media failure. Validate the surface with a non-sensitive
15-card synthetic fixture through the configured Previews public ingress.

This plan may add product code, focused tests, a private synthetic runtime
fixture, and preview/browser validation evidence. It may not rerun or amend
Plan 0057, execute a fresh acoustic cohort, apply a speaker assignment, or
modify the Previews repository, reverse proxy, or installed service.

## Vision Outcomes And Maturity Movement

| Capability | Current | Target | Evidence |
| --- | --- | --- | --- |
| Human identity review | Level 1 display-only artifact requiring chat-side answers | Level 2 synthetic-shadow review workflow with complete card controls and strict export | 15/15 decisions round-trip through the existing parser in tests and a browser fixture |
| Review media reliability | Level 1 eager-load surface with reproducible ingress failures | Level 2 on-demand synthetic-browser proof with direct-file fallback | Every synthetic card loads and seeks serially through the public preview path without media error or HTTP 502 |
| Acoustic speaker identity | Level 2 integrated-shadow evidence | Level 2 unchanged | No new acoustic execution, score, proposal, assignment, or identity evidence |
| Knowledge integrity | Frozen Plan 0057 state unchanged | Unchanged with explicit negative actions | Runtime receipt and tests prove zero assignment, identity, contact, relationship, profile, reference, provider, Graphiti, integration, or historical mutation |

This advances the north-star speaker-review and calibrated-uncertainty outcomes
by making literal human decisions collectable and replayable at the point where
non-authoritative acoustic evidence is shown. It improves the automatic
contextualization loop's safe fallback; it does not advance automatic identity
acceptance or the conversation knowledge store.

## Measurable Outcome

- `15/15` synthetic cards expose one keyboard-operable decision group.
- The four permitted outcomes are the two exact enrolled subject IDs,
  `neither_enrolled`, and `unknown`; display labels never become machine IDs.
- Complete exports parse without reformatting through
  `acoustic_plan0057_review.parse_review_answers()`.
- Partial, duplicate, unknown-card, inexact-identity, unsafe-label, and
  non-allowlisted decisions fail closed.
- All 15 synthetic audio controls start with `preload="none"`, load and seek
  serially through the public preview path, and expose a direct-file fallback.
- Browser/network evidence contains zero media errors and zero HTTP 502
  responses for the passing fixture.

## Non-Goals

- No fresh recording selection, diarization, transcription, acoustic model
  execution, proposal construction, gold collection, or correctness scoring.
- No speaker assignment, identity/contact/person/alias/role/relationship
  creation or mutation, profile/reference learning, provider write, Graphiti
  write, default integration, or historical reprocessing.
- No change to Plan 0057's frozen modules, receipts, result, or terminal
  decision.
- No Previews code, proxy, route, configuration, credential, or installed
  service mutation.
- No claim that range-serving behavior is fixed; on-demand loading and the
  direct-file link are the supported repo-owned mitigation.
- No production/default enablement and no automatic assignment threshold.

## Current State

Plan 0057 is closed at `plan_next_bounded_milestone` with complete 3/3
recording and 15/15 speaker denominators, correct two enrolled proposals and
13 abstentions, and unchanged identity state. Its published HTML uses
`preload="metadata"` for all 15 audio controls and has no decision controls.

Fresh browser diagnosis reproduced the operator's media symptom. Chromium
disabled cards after receiving intermittent `502 text/plain` responses during
the eager 15-request load. The corresponding retained and published WAVs are
byte-identical, non-empty, and valid mono 16 kHz PCM according to `ffprobe`.
This rejects missing publication and malformed source media as causes. The
public transport is outside this repo; the eager-load trigger and absent form
controls are inside the generated review surface.

The accepted drift-discovery findings are:

- `F0058-01`, `blocking`: no per-card decision controls or browser-generated
  importer-compatible answer block.
- `F0058-02`, `blocking`: eager metadata loading launches the complete media
  set concurrently and can receive ingress 502 responses that Chromium reports
  as format errors.

No other discovery finding is accepted. Remediation verification is
closed-world against these two findings plus critical regressions introduced
by their fixes.

## Execution Graph

| Unit | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 diagnosis and contract | Plan 0057 closeout, retained session `488e06d2f6da`, artifact `afff342eb85c` | Freeze the exact form and media failure contract without private payloads or tokens | Plan, corrected Note 0054, Roadmap, Runbook | Findings accepted or stop as `needs_evidence` |
| P1 reusable renderer | P0 | Add accessible controls, fail-closed export, lazy audio, status, and direct fallback | New root Python module and focused tests | Static, parser round-trip, escaping, and negative tests pass |
| P2 deterministic fixture authority | P1, clean pushed code | Create/replay one private 15-card synthetic WAV/HTML fixture with hashes and negative actions | New Plan 0058 module plus user-scoped runtime tree | Apply and replay are byte-stable with `0700`/`0600` modes |
| P3 public browser proof | P2 | Publish the synthetic directory and verify controls, export, serial media load/seek, fallback, and network status | Previews session plus ephemeral revocable access | 15/15 cards pass or one bounded rework/terminal stop |
| P4 terminal audit | P3 | Recompute acceptance evidence, close or stop, and update durable authorities | Plan, Roadmap, Runbook, optional Graphiti closeout | `complete`, `refine`, or `stop` |

Critical path: P0 -> P1 -> P2 -> P3 -> P4. Delegation receipt:
`not_spawned`; reason: the active runtime policy disables proactive
multi-agent delegation, and the renderer, fixture, browser proof, and closeout
share one tightly coupled authority path.

## Acceptance Criteria

- One semantic `fieldset`/`legend` decision group exists for every exact card,
  with keyboard-operable controls and no preselected identity.
- The only machine outcomes are the exact supplied enrolled subject IDs,
  `neither_enrolled`, and `unknown`; review display labels remain attributes.
- The browser exporter refuses incomplete cards and produces one ordered,
  duplicate-free `card_id=identity` line per card.
- The exported block is accepted unchanged by the existing strict Plan 0057
  parser; malformed and adversarial inputs remain rejected.
- Every audio element uses `preload="none"`, reports its load/error state, and
  has a direct-file fallback without embedding audio bytes in HTML.
- A deterministic non-sensitive 15-card fixture is private at rest, publishes
  without private transcripts or identities, and replays with stable hashes.
- In the target Chromium browser through public Previews ingress, all 15 media
  files load and seek serially, no media element reports an error, and captured
  media responses contain no 502.
- The browser surface exposes all 15 decision groups, rejects incomplete
  export, and produces a complete parser-compatible answer block.
- No forbidden mutation occurs, and no credential or share-link token is
  written to the repo, Graphiti, retained receipts, or closeout logs.

## Validation

- Focused pytest coverage for renderer validation, escaping, accessibility,
  lazy media, decision completeness, ordered export, parser round-trip, and
  fail-closed identity/card cases.
- Plan 0058 preview/apply/replay tests with deterministic synthetic WAV hashes,
  private modes, no-clobber behavior, and negative action vector.
- Planning-contract audit and `git diff --check` before authority freeze and
  closeout.
- Clean, pushed, upstream-even repository authority before private fixture
  apply.
- Previews doctor plus one published synthetic session.
- Target-browser DOM, media-state, seek, export, accessibility, and network
  evidence through the public share route; revoke diagnostic share access at
  closeout.
- Full `.venv/bin/python -m pytest -q --tb=short` before terminal closeout.

## Safeguards And Hard Stops

- Stop on any non-allowlisted or display-name-derived machine identity.
- Stop on incomplete, duplicate, unknown-card, inexact, reordered, or
  non-parser-compatible export.
- Stop on any raw private transcript, human label, source audio, credential, or
  share token entering repo files, fixtures, Graphiti, or retained logs.
- Stop on any assignment, identity/contact/relationship, profile/reference,
  provider, Graphiti, integration, or historical mutation.
- Stop rather than modifying the Previews repo/service/proxy or widening into a
  fresh acoustic cohort.
- Stop or split after the single allowed P3 rework if the same 502/media-error
  invariant remains.

## Local Goal Bounds

`max_work_unit_attempts: 2`

`max_review_rework_cycles: 1`

`max_hardening_checkpoints: 2`

`checkpoint_interval: 1 completed execution unit`

`authorization_gate: significant_departure_only`

`retry_budget_mode: renewable_execution_window`

`review_discovery_passes: 1`

`review_verification_mode: closed_world`

`review_finding_fields: criterion, evidence, consequence, reproducer, confidence, suggested_disposition`

`review_disposition_values: blocking | nonblocking_backlog | rejected | needs_evidence`

`checkpoint_record_fields: plan_version, state_transition, progress_classification, evidence, subagent_status, authority_classification, review_disposition_summary, next_action_or_stop_reason`

## State And Authority

Plan states: `ready`, `active`, `awaiting-review`, `awaiting-gate`, `blocked`,
`complete`, `failed`, `cancelled`.

The user-approved goal supplies standing authority for ordinary in-envelope
implementation, private synthetic fixture generation, Previews publication,
browser validation, repair, retest, commit, push, and closeout. New authority
is required for Previews code/runtime/proxy mutation, private-data expansion,
fresh acoustic execution, identity or assignment mutation, provider writes,
production enablement, publication outside the private preview workflow, or
another significant departure.

## Terminal Decision

- `complete`: both accepted findings pass closed-world verification, every
  acceptance criterion has current evidence, and all mutation/privacy guards
  remain negative.
- `refine`: the bounded implementation is safe and parser-compatible but one
  non-safety browser criterion remains unmet after the allowed rework.
- `stop`: any hard stop, privacy breach, forbidden mutation, parser mismatch,
  repeated 502/media failure, or unowned cross-repo repair is required.

Even `complete` authorizes only a later bounded P10 milestone. It does not
authorize fresh acoustic execution, automatic assignment, profile learning,
production integration, provider write-back, or historical reprocessing.

## Checkpoint 2026-08-07 | P1 and P2 implementation

- Progress: `P1 complete`; `P2 implementation complete`, with the committed
  fixture apply/replay proof still pending.
- Evidence: focused renderer/fixture tests `11 passed`; complete repository
  suite `905 passed`; Python compilation and `git diff --check` passed.
- Authority: implementation remains inside the approved repo-local renderer,
  tests, and private synthetic fixture scope. No runtime fixture, preview
  publication, acoustic execution, assignment, identity, provider, or
  integration mutation has occurred at this checkpoint.
- Review disposition: both accepted findings have implementation coverage;
  browser/public-ingress verification remains open for `F0058-01` and
  `F0058-02`.
- Delegation: `not_spawned`; the active runtime policy disallows proactive
  delegation and the execution path remains tightly coupled.
- Next action: commit and push the implementation authority, then execute P2
  apply/replay and P3 public-browser proof against that exact commit.

## Terminal Checkpoint 2026-08-07 | P2 through P4 complete

- State transition: `OPEN` -> `CLOSED`; terminal decision `complete`.
- Implementation authority: clean, pushed, upstream-even commit `d381744`.
  Focused renderer/fixture tests passed `11/11`; the complete repository suite
  passed `905/905` in 88.06 seconds; compilation and `git diff --check` passed.
  The active-plan planning-contract audit returned `ok: true`; the unscoped
  audit continues to report only inherited missing-state defects in older
  plans, not Plan 0058.
- Private fixture: preview content
  `91fc2edf7dd503f49d3a97d8cc9d936420939552f6c90578f1acf8a0f77dd953`
  produced 15 deterministic WAVs and index hash
  `d3aa45a1463798408a781ca02365f651a08e0d49c85f62c159d5504e0dd5367e`.
  Apply/replay was idempotent with `0700` directories, `0600` files, and every
  negative action still false.
- Public proof: Previews session `e3d074abfa25`, artifact `151d48a5038b`.
  Chromium found 15 cards, 15 fieldsets, 60 radios, no preselection, 15 lazy
  audio controls, and 15 direct-file fallbacks. Incomplete export returned no
  block; a complete 15-line ordered export was accepted unchanged by
  `acoustic_plan0057_review.parse_review_answers()` with all four allowlisted
  machine outcomes represented and one review-only label retained separately.
- Media proof: before explicit loading the page emitted zero media requests.
  Serial on-demand validation loaded metadata, sought, and advanced playback
  for `15/15` clips. The captured network log contained 15 Media responses at
  HTTP 200, zero failed responses, and zero HTTP 502 responses.
- Visual proof: the complete full-page Chromium capture was inspected for all
  15 consistently rendered cards, decision controls, lazy audio rows,
  fallbacks, and export controls; the temporary capture was then trashed.
- Access cleanup: diagnostic read-only share link `2b294327ba44` was revoked at
  `2026-08-08T04:28:35Z`; the ephemeral browser profiles and screenshot were
  trashed. No credential or share token entered the repository or retained
  evidence.
- Finding disposition: `F0058-01` and `F0058-02` are `resolved`. Closed-world
  verification found no critical regression and used no rework cycle.
- Progress classification: `outcome_progress`; human-review and review-media
  reliability reached the targeted Level 2 synthetic-shadow evidence. Acoustic
  speaker identity remains Level 2 integrated-shadow and unchanged.
- Authority classification: all work remained inside Plan 0058. No fresh
  cohort, assignment, identity/contact/relationship, profile/reference,
  provider, Graphiti, integration, historical, or Previews-runtime mutation
  occurred.
- Next action: any further P10 advancement requires a separate bounded plan;
  this closure grants no automatic assignment or production authority.
