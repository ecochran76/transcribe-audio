# Plan 0064 fresh-context implementation handoff

Date: 2026-08-09

Purpose: Preserve the exact historical authority, live-state baseline,
operator expectations, startup checks, and first bounded implementation packet
that began Plan 0064 without relying on chat history or reopening an approval
ritual.

## Current-status correction

This is the historical P0 startup handoff, not the current execution pointer.
P0 through P3 are now terminal, P4 direct-audio review is published, and the
strict P4 ingestion/measurement implementation is ready. A complete literal
39-row human-gold export has not been received, so source-disjoint measurement
cannot run. The separate reviewed Plan 0063 development replay completed with
three correct combined acceptances and zero high-support wrong identities, but
it produced zero acceptances through the specific residual rule and therefore
failed its non-vacuous quality gate. P5 remains unauthorized and P6 remains
withheld. Use the active plan and latest Runbook turn for current execution;
the P0 instructions below are retained as historical evidence.

## Outcome first

Plan 0063 is closed and its exact reviewed canonical-person plus biometric
transition has been applied successfully to the live user-scoped stores. Plan
0064 is now `OPEN` and is the current P09/P10 implementation path for making
that learning state useful on future conversations.

The intended behavior is already fixed by
[Note 0056](0056-2026-08-09-context-assisted-automatic-speaker-recognition.md):
recognize enrolled voices through active governed profiles, independently run
the existing contextual speaker workflow, join both through reviewed
voice/person bindings, solve identities across the whole conversation, and
accept a residual speaker only when one independently supported canonical
candidate remains without material contradiction. Accepted local observations
then enrich canonical/contact provenance for future retrieval.

The next ready unit is Plan 0064 P0. Freeze the complete active
reference/profile/person-binding inventory and one oldest-forward,
source-disjoint, previously unexposed evaluation cohort of at most twelve
conversations. Then P1 dynamic acoustic evidence and P2 contextual evidence
reuse may begin independently. No new operator approval is needed for ordinary
implementation, testing, repair, private shadow artifacts, or bounded
progression inside Plan 0064.

## Authority order

Read these in order before changing code or runtime state:

1. `AGENTS.md`
2. `VISION.md`
3. relevant current files under `docs/dev/policies/`
4. `ROADMAP.md`, Plan 0064 paragraphs under P10
5. `docs/dev/plans/0064-2026-08-09-context-assisted-automatic-speaker-recognition.md`
6. `docs/dev/notes/0056-2026-08-09-context-assisted-automatic-speaker-recognition.md`
7. `docs/dev/plans/0063-2026-08-09-reviewed-speaker-canonicalization-enrollment.md`, especially the terminal A1 checkpoint
8. `RUNBOOK.md`, Turns 357 through 359
9. Graphiti group `transcribe_audio_main`, advisory only

The pushed repository, immutable private Plan 0063 receipts, tests, and fresh
runtime readbacks are authoritative. This handoff and Graphiti are routing
surfaces, not substitutes for those sources.

## Repository and git state

- Repository: `/home/ecochran76/workspace.local/transcribe-audio`
- Branch: `plan-0037-campaign`
- Plan 0063 closeout commit:
  `d4d61a68af2996c8256eb4d6b1fa0484d8000ea6`
- This handoff's pushed commit:
  `ea1ba2f8194cf2b9c974a000ff1cc2ddfbc458bb`
- Both commits are historical anchors. Require them to remain ancestors of the
  current clean, upstream-even branch; do not require HEAD to stay frozen at
  either commit while Plan 0064 progresses.
- Plan 0063: `CLOSED`
- Plan 0064: `OPEN`
- No code changed in the Plan 0063 closeout/Plan 0064 planning commit.

Do not overwrite unrelated work if the worktree, HEAD, or upstream state has
changed. Reconcile current evidence before proceeding.

## Installed live state

The exact Plan 0063 transition is
`75166646421378e2fce4aee1e21c35a6d73fdfdbdb5b37297e4c13fc1b8663dc`.
The A1 authority content is
`f5f39a495b332f27246a0eca85621985365ed3ce653ff50c5b6da3e4e4e3cb6b`.
The terminal live receipt content is
`259ea605015ecd6b681140e529002c23e131b6e5cada0d1cdd62fc2b151e3dd5`.

Private authority and receipt paths:

- `/home/ecochran76/.local/state/transcribe-audio/plan-0063/a1-75166646421378e2fce4/private-authority.json`
- `/home/ecochran76/.local/state/transcribe-audio/plan-0063/live-apply-75166646421378e2fce4/receipt.json`

The terminal receipt records:

- six reviewed canonical people;
- nine reviewed speaker-slot bindings;
- one active enrolled-voice/person binding;
- five newly created biometric reference generations;
- fifteen newly created active model profiles;
- twenty-three selected enrollment sources;
- one logical/live apply;
- zero rollbacks; and
- zero unauthorized effects.

Its replayed post-apply snapshot hashes are:

- knowledge:
  `a7a97fa5fea5ca74f607bd71516f81ad36d09c91208e4d82bc87c8abc9b2e2e9`
- profiles:
  `474fd70b144d8017a27913fecf76b83f215de518bad7a5b45cb2db7f2678bae8`
- references:
  `b117238c325de8d9f5fdffab3e82bfaf7b513dfbc629b79a93d2594389ac64bf`

Complete live inventory discovered after the handoff commit:

- seven active reference heads over seven acoustic subjects;
- twenty-one active model profiles over three governed adapters;
- two superseded reference generations and six superseded model profiles;
- sixty-three active-reference source claims over eleven distinct recording
  hashes;
- five subjects directly identified by canonical-person IDs, one older subject
  resolved by the accepted explicit voice/person binding, and one older active
  subject with no reviewed canonical binding; and
- eighteen identity-ready active profiles plus three unbound profiles that
  must remain visible but cannot emit a person candidate.

The Plan 0063-created 5/15/23 counts are deltas inside this 7/21 whole-store
state. P0 must not freeze only the delta.

Live roots:

- conversation knowledge: `/home/ecochran76/.transcripts`
- biometric references:
  `/home/ecochran76/.local/state/transcribe-audio/plan-0037/biometric-references`
- verification profile store:
  `/home/ecochran76/.local/state/transcribe-audio/plan-0037/verification-calibration`

SQLite `quick_check` passed after the apply. Both `transcripts.service` and
`transcribe-watch.service` were `active/running` with `NRestarts=0`. The six
live reviewed canonical names include the provider-backed corrected full name;
do not regress it to a truncated first-name-only record. Exact private identity
values stay out of tracked docs and broad memory surfaces.

## Current validation evidence

- Plan 0063's final implementation checkpoint reported a 999-test full-suite
  pass before the exact A1 request and live apply. No source code changed in
  the subsequent apply/closeout slice.
- The live apply returned `live_apply_completed`; immediate replay returned
  `idempotent_replay=true` against all three current snapshot hashes.
- Post-apply SQLite `quick_check=ok`.
- Post-apply canonical-person count: six.
- Legacy `contacts` count remained two and legacy `speaker_assignments` count
  remained three; Plan 0063 added governed knowledge records rather than
  rewriting those legacy tables.
- Both exact transcript services returned to `active/running` with zero
  restarts.
- Active and goal-only planning audits passed after Plan 0064 was opened.
- `git diff --check` passed and commit `d4d61a6` was pushed upstream.

Do not rerun the Plan 0063 live apply. Its terminal receipt makes repeat calls
read-only idempotent replay. Do not rerun expensive acoustic enrollment merely
to rediscover the installed profile inventory.

## Operator expectations that must survive context reset

- `Okay go` and the standing Plan 0064 objective are execution authority for
  ordinary in-envelope work. Do not create repeated approval prompts for each
  bounded packet, test, repair, or private shadow step.
- A genuinely significant departure still stops: external provider mutation,
  a new tenant or private-data class, destructive action, weakened safety
  control, or materially changed Plan 0064 objective/non-goals.
- If human review is necessary later, place each question beside the correct
  audio and supporting evidence. A context-free question or a question in the
  wrong recording/card is not answerable.
- Review audio must work through authenticated remote Previews pages. Do not
  give the operator `localhost` media links when they are viewing remotely.
- Preserve explicit no-calendar evidence. A listened identity must not be
  relabeled as calendar-derived merely because a calendar title would have
  been convenient.
- Use the provider-backed full name already stored in the canonical record.
- When acoustic and contextual evidence point to the same person, preserve
  both as agreeing evidence; do not force the operator to choose one source.

## Plan 0064 P0: exact next packet

Goal: freeze the reusable live identity inventory and a valid evaluation
denominator before adding automatic scoring or inference behavior.

Required outputs:

1. A private, content-addressed inventory of active biometric references and
   profiles, including subject IDs, model/adaptor versions, source-generation
   bindings, active/withdrawn state, and current state hashes.
2. A private, content-addressed inventory of authoritative voice-subject to
   canonical-person bindings and current canonical source affinities.
3. An exact development exclusion set containing every source claim reachable
   from an active reference, including every Plan 0063 enrollment recording
   and source window.
4. An oldest-forward candidate evaluation cohort of at most twelve eligible
   recordings that is source-disjoint from all enrollment/development media
   and excludes prior speaker-identity gold, review, and prediction exposure.
5. Eligibility, exclusion, overlap, media-availability, diarization, and
   context-availability reason codes for every considered recording.
6. A frozen P0 receipt that records zero speaker assignments, new enrollments,
   provider writes, Graphiti writes, or historical reprocessing.
7. Focused tests for whole-store versus transition-delta accounting, inventory
   drift, withdrawn profiles, missing voice/person bindings, source overlap,
   prior-evaluation exposure, repeated recording hashes, and incomplete
   candidate denominators.

P0 may read live user-scoped state and create private manifests/receipts. It
may add focused repository code/tests/docs needed for deterministic replay. It
must not score the evaluation cohort, infer identities, apply observations,
create new voice profiles, or mutate external providers.

## Implementation map

Use CodeGraph before editing or tracing these structural seams. Likely
authoritative components are listed for routing; verify current symbols in the
index rather than assuming names or signatures from this note.

- `acoustic_biometric_references.py`: governed reference generations and
  active reference inventory.
- `acoustic_verification.py`: profile lifecycle, active model profiles, and
  scoring adapters.
- `acoustic_shadow_evidence.py`: source-bound non-authoritative acoustic
  evidence contract.
- `speaker_identity_preprocess.py`: existing per-speaker Clue Discovery and
  Identity Evaluation preparation/validation.
- `conversation_identity_retrieval.py`: bounded contextual candidate evidence.
- `conversation_knowledge_store.py`: canonical people, source records,
  observations, and current person profiles.
- `speaker_identity_plan0063_transition.py` and shared Plan 0063 apply helpers:
  exact installed binding/reference lineage; replay/reference only, not a new
  Plan 0064 orchestrator.

Plan 0064 should add a focused P0 inventory/cohort module rather than growing
the watcher or API transport into the identity domain.

## Safety and effect boundaries

Plan 0064 explicitly permits bounded repo implementation, tests, private
shadow artifacts, and later policy-qualified local knowledge acceptance after
the source-disjoint quality gate passes. It does not permit:

- residual identity by elimination alone;
- automatic voice enrollment from context or inferred identity;
- counting duplicate provider records as independent corroboration;
- external Google, Odollo, or other provider writes;
- raw transcript, audio, embedding, provider payload, or private identity
  values in git or Graphiti;
- watcher-inline provider retrieval or acoustic model work;
- using Plan 0063 development media as unseen evaluation evidence;
- historical reprocessing during P0; or
- claiming dependable Level 4 identity from this bounded cohort.

The eventual residual-speaker rule requires two calibrated accepted known
voice/person bindings, exactly one independently supported remaining canonical
candidate, slot-relevant transcript/context support, complete provenance, and
no material contradiction. Otherwise abstain or route one useful audio-linked
review.

## Fresh-context startup

```bash
cd /home/ecochran76/workspace.local/transcribe-audio
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count @{upstream}...HEAD
git log -3 --oneline --decorate
/home/ecochran76/.local/bin/graphiti-runtime doctor
/home/ecochran76/.local/bin/graphiti-runtime discover \
  --group-id transcribe_audio_main \
  --max-facts 6 --max-nodes 4 --max-episodes 4 \
  "Plan 0064 active profile inventory source-disjoint cohort speaker identity"
systemctl --user show transcripts.service transcribe-watch.service \
  --property=Id,ActiveState,SubState,NRestarts --no-pager
sqlite3 -readonly /home/ecochran76/.transcripts/transcripts.sqlite3 \
  "PRAGMA quick_check; SELECT COUNT(*) FROM knowledge_people;"
```

Historical handoff baseline:

- branch `plan-0037-campaign` with `d4d61a6...` and `ea1ba2f...` as pushed
  ancestors;
- upstream counts `0 0`;
- Graphiti runtime healthy but focused recall stale for Plan 0064;
- both services active/running with zero restarts;
- SQLite `ok`; and
- six canonical people.

Proceed with P0 when the current branch remains clean and upstream-even and
both historical anchors remain ancestors. Record the current HEAD in each new
P0 checkpoint rather than editing this handoff after every commit.

Planning validation:

```bash
.venv/bin/python .codex/skills/repo-policy-selector/scripts/audit_planning_contract.py \
  --repo-root . --active-only --json
.venv/bin/python .codex/skills/repo-policy-selector/scripts/audit_planning_contract.py \
  --repo-root . --goal-only --json
git diff --check
```

## Historical blockers and residual risk

P0 had no blocker when this handoff was issued and is now complete. The
principal risks identified at that point were confusing Plan 0063 apply deltas
with the complete live inventory, training/evaluation or prior
evaluation overlap, using inactive profiles, allowing the one unbound active
subject to emit a person candidate, losing authoritative voice/person
bindings, and treating a context candidate as residual proof without
independent slot support. Freeze and test those conditions before implementing
automatic scoring or acceptance.

Graphiti discovery was healthy at handoff creation but returned older Plan
0025 context and no Plan 0064 fact. Use it only as advisory routing until it is
refreshed through a separately authorized memory-write workflow.

## Suggested skills

- `graphiti-discovery` for advisory repo-memory routing at startup.
- `repo-policy-selector` before modifying the active plan or execution bounds.
- `codegraph-workspace` and CodeGraph MCP tools for structural code discovery.
- `codebase-design` when defining the focused P0 inventory/cohort module.
- `faster-whisper` only if P0 eligibility checks expose missing diarization or
  transcript preparation, not as a reason to widen the packet.
- `previews` plus `agent-browser` for any later human audio review surface;
  verify authenticated remote playback and avoid localhost links.
- `app-intelligence-automation` when P2 begins reusing the reviewed two-phase
  contextual workflow.

## Historical recommendation

The recommendation at handoff time was to execute Plan 0064 P0 immediately
under standing authority. That packet and P1 through P3 are now complete. The
current next evidence boundary is the complete 39-row human-gold export; even
after it arrives, P5 must remain disabled unless the source-disjoint and
reviewed-development gates both demonstrate the plan's non-vacuous acceptance
requirements.
