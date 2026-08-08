# Plan 0057 closeout and Plan 0058 review-surface handoff

Date: 2026-08-07

Purpose: Give a fresh-context agent the minimum durable orientation needed to
consider the next bounded P10 milestone after Plan 0057, with special attention
to the human-review entry controls and unreliable audio playback reported by
the operator.

## Correction After Fresh Browser Diagnosis

Fresh Plan 0058 startup work on 2026-08-07 reproduced the audio failure in the
retained authenticated review surface and narrows this handoff in two material
ways:

- the failed media are not missing or malformed: the retained and published
  copies are byte-identical, `ffprobe` accepts them as mono 16 kHz PCM WAV, and
  their sizes and durations are complete;
- Chromium captured intermittent `502 text/plain` responses at the public
  Previews ingress while the page eagerly issued 15 concurrent
  `Range: bytes=0-` media requests from `preload="metadata"`. The corresponding
  controls became disabled with media error code 4 and `Format error`.

`F0057-08` is therefore no longer merely `needs_evidence`. It is an accepted
blocking review-surface finding for the successor. The failing transport lies
outside this repository, but the concurrency trigger is owned by the generated
HTML. Plan 0058 may remain repo-local by rendering on-demand audio
(`preload="none"`) with an explicit direct-file fallback and proving the result
through the same external ingress. It may diagnose but must not modify the
Previews repository, proxy, or installed service.

Plan 0058 should contain only the review-surface reliability slice. A fresh
acoustic cohort is deferred to a later bounded successor after the synthetic
review fixture passes. Graphiti is healthy at this correction checkpoint, but
its focused repo recall still lacks the Plan 0057 closeout and remains advisory.

## Outcome First

Plan 0057 is closed at terminal decision `plan_next_bounded_milestone`. Its
three-recording integrated-shadow batch covered all 15 eligible speakers. Both
enrolled-subject proposals and all 13 abstentions were correct; enrolled recall
and proposal precision were each `1.0`. Identity state was unchanged and no
speaker assignment, identity, contact, relationship, profile, reference,
provider, default-integration, Graphiti, or historical-reprocessing mutation
occurred.

This is Level 2 integrated-shadow evidence, not authorization for automatic
assignment. Review burden was `1.0` and manual-resolution burden was `13/15`.
The next agent should consider a separate bounded Plan 0058 that makes the
review surface reliable before expanding the acoustic pilot.

## Authority Order

Read these in order before planning or changing code:

1. `AGENTS.md`
2. `VISION.md`
3. Relevant current files under `docs/dev/policies/`
4. `ROADMAP.md`, lane P10
5. `docs/dev/plans/0057-2026-08-06-enrolled-only-acoustic-shadow-review-integration.md`
6. `RUNBOOK.md`, Turns 319 and 320
7. `docs/dev/notes/0053-2026-08-06-plan0056-closeout-successor-handoff.md`
8. Graphiti group `transcribe_audio_main`, advisory only

Repo files, frozen private receipts, tests, and current runtime readback are
authoritative. Do not use this handoff or Graphiti to override current evidence.

## Repository And Git State

- Repository: `/home/ecochran76/workspace.local/transcribe-audio`
- Branch: `plan-0037-campaign`
- Plan 0057 closeout commit: `46bb144`
- At handoff creation, the branch began clean and upstream-even.
- Plan 0057 is `CLOSED`; do not reopen it or change its frozen result.
- P10 remains open only at roadmap altitude. A new numbered plan is required
  before another acoustic execution or review-surface implementation slice.

Do not overwrite unrelated work if the worktree has changed since this note.
Re-establish current state before proceeding.

## Frozen Plan 0057 Evidence

The canonical narrative and full acceptance evidence live in the closed plan
and Runbook. The essential immutable handles are:

- P0 authority SHA-256:
  `4fe89d673771af9ae51ab278a31215e07f24fb7fd1041fe20be82e3c09a90682`
- Execution authority SHA-256:
  `42a443a1185b31e494562a060129fae03e11e0b1a800f0863352380cd256094e`
- Execution content SHA-256:
  `089d0213153bd001a86669141e3b7a0a72b7b7aa8638d71e3d8f8dc5c32b41e4`
- Human-review content SHA-256:
  `5e12d4fb2bf332e370b38a5888ce26bfe483355425e728db9924f05705c4fcee`
- Terminal preview SHA-256:
  `c859b3d217f027ddf14c4630a283a3aa111e2e87ab5a18609a9d92ef9b99f85a`
- Independent audit content SHA-256:
  `f8402069597495a9eddce9dafb4dd1a2baf53ed8c324811fc9b06b20c9dfecc5`
- Frozen/current identity-state SHA-256:
  `64e0a7f44f59563ee848212a93d00e817be59c5471f035a96db7a75f8810924a`

Private receipts live below
`~/.local/state/transcribe-audio/plan-0057/`. Human-review and terminal-audit
directories were verified `0700`; retained receipts and manifests were `0600`.
Do not copy private labels, raw transcripts, audio, answer files, or provider
data into repository documentation, Graphiti, logs, or test fixtures.

Useful replay commands:

```bash
.venv/bin/python acoustic_plan0057_review.py replay \
  --review-content-sha256 5e12d4fb2bf332e370b38a5888ce26bfe483355425e728db9924f05705c4fcee

.venv/bin/python acoustic_plan0057_audit.py replay \
  --audit-content-sha256 c859b3d217f027ddf14c4630a283a3aa111e2e87ab5a18609a9d92ef9b99f85a
```

Do not rerun the expensive acoustic/model batch merely to rediscover these
results.

## Measured Result

- Eligible/entered recordings: 3/3
- Eligible/covered speakers: 15/15
- Enrolled speakers in human review: 2
- Proposals: 2
- Confirmed correct proposals: 2
- Abstentions: 13
- Correct abstentions: 13
- Wrong proposal dispositions: 0
- High-confidence wrong dispositions: 0
- Unknown identities: 0
- Enrolled recall: `1.0`
- Proposal precision: `1.0`
- Review burden: `1.0`
- Manual-resolution burden: `13/15`
- Identity creations, applied assignments, profile/reference mutations, and
  provider writes: 0

The result supports another bounded shadow milestone. It is too small and too
review-heavy to justify production or automatic assignment.

## Review-Surface Findings To Consider

### F0057-07: no decision entry controls

Disposition: `nonblocking_backlog` for Plan 0057; expected to be blocking
before another human review gate.

The published HTML showed card IDs, proposals, audio, transcripts, and text
instructions but no per-card form controls. The operator had to return 15
ordered answers in chat. A successor should provide one explicit identity
control per card and generate an importer-compatible answer block without
turning display names into machine identity.

### F0057-08: eager media loading overloads the public preview path

Disposition: `blocking` for Plan 0058 until a browser-validated lazy-loading
surface passes; nonblocking for the already closed Plan 0057 result.

The operator reported that audio was missing for many cards. Current evidence
rules out simple publication loss:

- the retained review tree contains 15 non-empty WAV files;
- the published Previews artifact also contains all 15 files;
- every referenced authenticated artifact path returned HTTP 200 with
  `audio/x-wav`;
- a diagnostic request carrying `Range: bytes=0-15` returned HTTP 200 and the
  full file rather than HTTP 206;
- fresh Chromium reproduction observed disabled controls with media error code
  4 and captured intermittent 502 responses only after the page launched its
  eager 15-file metadata load;
- the failed published files hash-identically match the retained files and
  decode successfully with `ffprobe`.

The authenticated review session is `488e06d2f6da`; its directory artifact is
`afff342eb85c`. Do not persist credentials or share-link tokens while testing.
If the root cause is in the separate Previews service rather than this repo's
generated HTML, stop after diagnosis and request/establish separate authority
for that repo or runtime. Do not silently widen a transcribe-audio plan into a
cross-repo service mutation.

## Recommended Next Bounded Decision

Plan 0058 is a review-surface reliability milestone with a synthetic browser
proof, not a fresh shadow or automatic-assignment plan.

Suggested execution graph:

| Unit | Outcome | Terminal condition |
| --- | --- | --- |
| P0 reproduce and freeze | Freeze the reproduced missing-controls finding and the eager-load/502 browser evidence without retaining credentials or share tokens | Exact reproducer and repo-owned mitigation boundary are recorded |
| P1 entry workflow | Render one accessible allowlisted decision control per card and export exact importer-compatible answers | All cards round-trip without name-to-identity promotion or mutation |
| P2 media workflow | Render lazy on-demand audio plus an explicit direct-file fallback and prove it under the configured public preview path | All synthetic cards load and seek serially without a media error or 502 |
| P3 synthetic inspection | Publish and inspect a non-sensitive 15-card fixture through Previews | Browser inspection passes before any fresh human gate |

Fresh acoustic execution is explicitly outside Plan 0058. The public-ingress
failure belongs outside this repository, so the bounded repo-owned result is a
lazy-loading mitigation plus a documented fallback. Any proxy/service repair
or fresh acoustic cohort requires separate authority.

Minimum acceptance evidence for the review-surface slice should include:

- exact card-to-field binding and complete/duplicate/unknown-card rejection;
- accessible keyboard-operable controls for enrolled subject, neither
  enrolled, and unknown outcomes;
- export that the existing strict importer accepts without manual reformatting;
- authenticated browser proof that every expected audio control loads, plays,
  and seeks or uses a documented reliable fallback;
- privacy proof that raw audio, transcripts, and review labels remain private;
- deterministic artifact generation and tests for the failing cases;
- no mutation to assignment, identity, contact, relationship, profile,
  reference, provider, Graphiti, integration, or historical state.

## Hard Stops And Deferred Work

Unless a separate authorized plan deliberately changes the boundary, do not:

- apply speaker assignments or enable automatic/default integration;
- learn, replace, or expand acoustic profiles from review outcomes;
- create or merge people, contacts, aliases, roles, or relationships;
- turn non-enrolled display labels into canonical identities or enrollments;
- write to GWS, Odollo, Graphiti, receipts repositories, or other providers;
- rerun Plan 0057's model batch or amend its frozen receipts;
- begin historical reprocessing;
- widen into the P09 canonical-person and relationship-graph work;
- modify the separate Previews codebase or installed service without explicit
  cross-repo/runtime authority.

## Fresh-Context Startup

```bash
cd /home/ecochran76/workspace.local/transcribe-audio
git status --short --branch
git rev-list --left-right --count HEAD...@{upstream}
git log -3 --oneline --decorate
~/.local/bin/graphiti-runtime doctor
```

Then:

1. Invoke `graphiti-discovery` and query `transcribe_audio_main`, but prefer the
   repo if Graphiti lacks a current Plan 0057 closeout fact.
2. Read the authority files listed above and replay the immutable review/audit
   receipts rather than rerunning audio models.
3. Use CodeGraph before changing the review generator or tracing how its HTML
   is produced; do not reconstruct structural call paths with grep.
4. Use the Previews and browser skills to reproduce the authenticated media
   behavior. Keep diagnosis read-only until ownership and authority are clear.
5. Draft a bounded Plan 0058 with vision outcome, current/target maturity,
   measurable user outcome, write surface, dependencies, explicit limits,
   review findings, safeguards, and terminal decisions.
6. Run the planning-contract audit, commit, push, and verify a clean,
   upstream-even branch before freezing any new runtime authority.

Planning validation command:

```bash
.venv/bin/python .codex/skills/repo-policy-selector/scripts/audit_planning_contract.py \
  --repo-root . --active-only --json
git diff --check
```

At handoff creation, `graphiti-runtime doctor` reported `mcp_http: down` while
FalkorDB and Inspector ingress were healthy and the in-session Graphiti MCP
search still returned results. At Plan 0058 startup the doctor was fully
healthy, but focused search still produced no current Plan 0057 closeout fact.
Treat Graphiti as stale/advisory for this decision.

## Suggested Skills

- `graphiti-discovery` for advisory prior-decision routing.
- `repo-policy-selector` before opening a new bounded plan.
- `codebase-investigator` or CodeGraph MCP tools for the review-generator and
  media-serving call paths.
- `diagnosing-bugs` for the audio playback reproducer and ownership boundary.
- `previews` for publishing and inspecting the complete review artifact.
- `agent-browser` or `dev-browser` for authenticated browser-level playback,
  accessibility, and form validation.
- `codebase-design` if Plan 0058 needs a narrow reusable review-form contract.

## Best Recommendation

Open Plan 0058 as the bounded repo-owned review-surface slice now that browser
evidence identifies eager loading as the trigger. Make entry controls,
importer-compatible export, lazy audio, and complete synthetic browser proof
the terminal gates. Preserve Plan 0057's no-mutation boundary and defer another
acoustic cohort until the human review workflow passes without chat-side repair.
