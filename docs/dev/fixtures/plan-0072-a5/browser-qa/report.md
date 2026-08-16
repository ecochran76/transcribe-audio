# Dogfood Report: Plan 0072 A5 identity review

| Field | Value |
|-------|-------|
| **Date** | 2026-08-16 |
| **App URL** | `http://127.0.0.1:18972/?view=Identity+Review` |
| **Session** | `plan72-a5` |
| **Scope** | Disposable redacted queue/People store; desktop and mobile; audio, effect preview, decision record, and tab transition |

## Summary

| Severity | Open | Resolved during audit |
|----------|------|-----------------------|
| Critical | 0 | 0 |
| High | 0 | 0 |
| Medium | 0 | 1 |
| Low | 0 | 0 |
| **Total** | **0** | **1** |

The final audited build has no browser errors or console messages. It exposes
the actual original filename, bounded source audio, all frozen decision
actions, exact zero-effect preview, stale-safe decision recording, and the
separate People projection at both 1440x1000 and 390x844.

## Resolved issue

### ISSUE-001: People retained the Identity Review state filter

| Field | Value |
|-------|-------|
| **Severity** | medium |
| **Category** | functional |
| **Status** | resolved and re-verified |
| **URL** | `http://127.0.0.1:18972/?view=People` |
| **Repro Video** | N/A; the faulty bundle was replaced before closeout |

Switching from Identity Review to People initially retained the
`unreviewed` state and requested `/api/people?status=unreviewed`, hiding a
valid reviewed person. The view now resets its query, filter, and selection
when the mode changes. The same Identity Review to People transition loads
one reviewed person and all three authoritative tables.

Verification: [People desktop](screenshots/people-desktop.png) and
[People mobile](screenshots/people-mobile.png).

## Acceptance evidence

- [Identity Review desktop](screenshots/identity-review-desktop.png) shows the
  original `.m4a` filename, calendar alternatives, participant hypotheses,
  evidence pillars, speaker proposal, and source-bound audio control.
- [Exact effect preview](screenshots/identity-review-preview.png) shows
  preview-only scope, zero provider writes, zero raw deletions, and no applied
  identity, contact, relationship, profile, provider, or deletion effect.
- [Recorded decision](screenshots/identity-review-recorded.png) shows the
  unresolved queue state at projection v2 and zero accepted identity effects.
- [Identity Review mobile](screenshots/identity-review-mobile.png) and
  [People mobile](screenshots/people-mobile.png) remain readable and operable
  at 390x844.
- Browser playback loaded the 8-second WAV through the source blob range route
  and advanced to 7.05 seconds. Final browser errors and console messages were
  both empty.

No private corpus, provider, live store, accepted identity/profile effect,
public route, deployment, or second authentication layer was used.
