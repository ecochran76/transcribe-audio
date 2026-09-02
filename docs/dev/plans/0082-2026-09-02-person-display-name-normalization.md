# Plan 0082 | Person Display-Name Normalization

State: OPEN

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

The source projection copies accepted `primary_name` directly into
`review_targets.people[].label`, and React renders that label verbatim. The only
client normalization trims and lowercases names for exact-match suggestions.
Live examples show `Gates, Zachary` from Calendar/GWS, `Cienkosz, Basia` with a
better `Basia Cienkosz` alias, and incomplete `Dr. Stefl` from human review.

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

