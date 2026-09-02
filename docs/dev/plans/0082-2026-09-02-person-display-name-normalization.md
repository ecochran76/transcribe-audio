# Plan 0082 | Person Display-Name Normalization

State: CLOSED

Lane: P09

Date: 2026-09-02

Related authority: Plan 0081, `VISION.md`, and the operator report that accepted
person targets mix `First Last`, `Last, First`, and incomplete titled labels.

## Scope

Add one deterministic presentation projection for person names. Preserve every
accepted and provider-observed label unchanged as identity evidence, while the
directory and review selectors receive a human-friendly display name, a stable
sort key, and an explicit completeness state. Install and visually validate the
result against the current live target cohort.

## Non-Goals

- Do not merge people, change canonical person IDs, or rewrite immutable
  identity events.
- Do not infer a missing given name, expand initials, or claim that a title-only
  label such as `Dr. Stefl` is a complete identity.
- Do not read or mutate providers, accept directory hypotheses, alter speaker
  assignments, or change biometric authority.
- Do not treat display normalization as evidence that two same-name records are
  one person.

## Vision outcomes and maturity movement

This plan advances speaker review, identity quality, provenance, and the
review/acceptance stage of the knowledge loop in `VISION.md`.

| Capability | Current maturity | Target | Outcome evidence |
| --- | --- | --- | --- |
| Person-name presentation | Level 1: identity authority exists, but provider formatting leaks directly into review controls | Level 2 deterministic presentation projection | Public directory tests prove `Family, Given` renders as `Given Family`, raw labels remain intact, and incomplete names are explicit |
| Review usability | Level 2 review workflow with inconsistent target labels | Level 2 normalized, alphabetized targets | Installed selectors show normalized complete names and visibly qualify incomplete names without changing target IDs |
| Identity provenance | Level 2 retained source records | Level 2 retained evidence plus derived presentation | Live API readback shows unchanged primary/source labels beside derived display metadata |

## Execution graph

| Packet | Depends on | Outcome | Write surface | Terminal condition |
| --- | --- | --- | --- | --- |
| P0 contract freeze | This plan | Freeze source-preserving display behavior | plan and tests | Examples and non-goals are explicit |
| P1 projection tracer | P0 | Add one failing public projection regression | tests | Current v3 output fails the normalized-target expectation |
| P2 name presentation | P1 | Derive display name, sort key, and completeness behind the directory interface | projection and tests | Complete names normalize; incomplete names remain truthful; raw labels are byte-preserved |
| P3 UI consumption | P2 | Render derived names and incomplete-state context in compact selectors/directory rows | frontend and tests | Both approval and expanded selectors use the same display helper |
| P4 installed validation | P3 | Build, test, install, read back, and visually inspect | service and private QA artifacts | Focused/full tests, live API, desktop/mobile Agent Browser, and no-effect accounting pass |

The packets are serialized around one small projection interface; no subagent
split is appropriate.

## Current State

The installed v4 directory projection preserves accepted `primary_name`,
member labels, aliases, and source records while deriving `display_name`,
`sort_name`, and `name_completeness`. Both compact and expanded review selectors
consume that presentation layer. Live examples now display `Zachary Gates` and
`Basia Cienkosz`; `Dr. Stefl` remains unchanged and is labeled `incomplete
name` rather than being expanded without evidence.

## Acceptance Criteria

- Accepted/source `primary_name`, aliases, member labels, and source-record
  labels remain unchanged.
- A reversible `Family, Given` label is displayed as `Given Family`.
- A complete untitled alias can supply the display form when the primary label
  is incomplete or provider-formatted.
- Honorifics are omitted only when a complete name remains; `Dr. Stefl` is not
  silently presented as a complete `Stefl` identity.
- Every person entity and accepted target exposes `display_name`, `sort_name`,
  and `name_completeness`; accepted targets are sorted deterministically.
- Compact selectors use `display_name` and append a concise incomplete-name cue
  when necessary.
- Existing target IDs, target suggestion behavior, review effects, and provider
  write count remain unchanged.

## Validation

- One red-to-green public projection test for normalization and preservation.
- One red-to-green frontend behavior test for target-label presentation.
- Focused directory, review workflow, API, and frontend test selections.
- Frontend production build, Python compilation, planning audit, and diff
  hygiene.
- Installed `/api/people` readback for the current examples and accepted-target
  count/ID preservation.
- Desktop and narrow Agent Browser inspection with no review action invoked.

## Closeout Evidence

- Public projection and frontend red-to-green tests cover reversible
  comma-form normalization, whitespace normalization, source-label retention,
  incomplete-name disclosure, deterministic target ordering, and preservation
  of raw-name suggestion matching.
- Focused directory/API tests pass 11 tests; the broader directory and review
  selection passes 45 tests; frontend tests pass 6 tests; the provider-free
  suite passes 1,289 tests in 91.82 seconds.
- The frontend production build, Python compilation, planning audit, and diff
  hygiene pass. The direct `.venv/bin/pytest` retry was invalid in this repo
  because it omitted the repository import path; `.venv/bin/python -m pytest`
  is the passing authoritative invocation. A later frontend retry mistakenly
  passed Node's unsupported bare `--run` flag; canonical `npm test` passes all
  6 tests.
- Installed `/api/people` returns schema v4 with the same 15 accepted target
  IDs and unchanged review counts: 10 accepted, 1 rejected, and 51 unreviewed.
  The inspected source/display pairs are `Gates, Zachary` / `Zachary Gates`,
  `Cienkosz, Basia` / `Basia Cienkosz`, and `Dr. Stefl` / `Dr. Stefl` with
  `name_completeness=incomplete`.
- Agent Browser inspected the live Contacts approval table at 1440 by 900 and
  390 by 844. The person selector contained all three expected display labels,
  neither comma-form label, no console or page errors, and the narrow table used
  contained horizontal scrolling. No review action was invoked; the named QA
  browser was closed.
- Implementation checkpoint `7160a99` is pushed. `transcripts.service` is
  active/running at PID 14110 with `NRestarts=0`.
