# Plan 0048 | Plan 0037 generation-2 evaluation execution

State: IMPLEMENTED; production apply remains OPEN

Lane: P10

Plan Version: 1

Parent: Plan 0046 generation-2 terminal authority

Owner: primary agent

Expected Write Surface: `acoustic_generation2_evaluation.py`, focused tests,
this plan, Plan 0037, `ROADMAP.md`, and `RUNBOOK.md`; immutable private reveal
and terminal-stop artifacts only beneath the user-scoped Plan 0037 runtime
root after review and clean pushed commit.

## Vision alignment

This packet advances calibrated and safely abstaining speaker evidence by
executing the already-authorized successor reveal and enforcing the frozen
minimum-evidence policy before audio preparation, biometric models, or
scoring. Current and target maturity remain `2 - Shadow`. Progress is measured
by exact replay of the applied pre-reveal parent, a private prediction-excluded
gold reveal, and a deterministic terminal outcome. A STOP is a valid measured
outcome and does not authorize operational speaker identity.

## Current state and target

Applied parent authority
`generation-2-pre-reveal-authority-e36736a176600d5536c7c668`, content SHA-256
`e36736a176600d5536c7c6688ce00d04165955cf09d69cd67d2bb1b082ef61ad`,
authorizes only evaluation reveal, offline P1/P2 preparation, and immutable
window freeze. It forbids models, scores, metrics, and decisions until a valid
exact-trial child exists.

The authorized private reveal establishes that the two evaluation recordings
contain six opaque person labels but zero subjects represented by the six
frozen profiles. Therefore every candidate-by-method unit has an absolute
maximum of zero genuine and zero impostor trials, below the frozen minima of
20 genuine and 100 impostor. No windowing choice can repair that class absence.

The target is an immutable, replayable reveal/preflight authority that records
the private opaque gold only in user-scoped storage and emits a portable
count/hash-only receipt with terminal `STOP`. P1/P2 preparation, window freeze,
exact-trial child construction, model execution, scoring, metrics, and
selection remain `not_run` because the frozen policy requires a global stop
before model execution.

## Scope

- Replay and bind the exact applied generation-2 parent manifest and reviewed
  successor corpus bytes.
- Reveal only the two frozen evaluation records and validate their source,
  transcript lineage, gold structure, membership hash, and disjointness.
- Compare opaque evaluation subject IDs with frozen profile person references
  before reading or deriving audio.
- Calculate only structural maximum class feasibility. Do not construct
  windows, embeddings, scores, or data-dependent thresholds.
- Persist one immutable private manifest containing the opaque reveal and one
  portable receipt containing hashes, counts, reason codes, action flags, and
  no subject IDs, names, paths, transcript text, audio, embeddings, or scores.
- Replay full bodies, exact paths, content identities, permissions, and
  singleton directory inventory.

## Non-goals

- Do not weaken the 20 genuine / 100 impostor / 20 open-set per-unit minima.
- Do not reinterpret open-set trials as impostor trials.
- Do not prepare audio or freeze windows after a proof that a required class
  has zero eligible subjects.
- Do not create an incomplete exact-trial child merely to pass the child gate.
- Do not run any model, score, metric, candidate reduction, or terminal model
  selection.
- Do not place raw private gold or paths in Git or portable receipts.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A contract | primary | closed Plans 0046 and 0047 | plan freezes exact reveal/preflight semantics |
| B implementation | primary | A | deterministic preview/apply/replay and adversarial tests pass |
| C independent audit | existing read-only reviewer | B | `PASS`, or one bounded repair and re-audit |
| D production apply | primary | C plus clean pushed commit | private reveal and portable STOP receipt replay full-body |
| E closeout | primary | D | Plan 0037 records terminal STOP and all forbidden work as `not_run` |

## Gates and stop conditions

- Parent authority ID/content, preview content, successor corpus file SHA-256,
  corpus content identity, evaluation membership, profiles, candidate matrix,
  terminal policy, and minimum-evidence policy must match exactly.
- Any private path, file type, permission, source byte, transcript byte, gold
  structure, duplicate membership, or cross-split overlap drift fails closed.
- Only `person` gold with a valid opaque subject ID can contribute to class
  feasibility. Mixed, unknown, or missing subject gold is excluded.
- With no evaluation subject represented by any frozen profile, maximum
  genuine and impostor denominators are both zero for every matrix unit.
- A required denominator below policy yields
  `trial_class_denominator_below_policy` and global STOP before preparation or
  model execution. Blanket authorization cannot override this content gate.

## Acceptance and validation

- Preview is deterministic and no-write; it proves the expected terminal stop
  from fixture/private-safe inputs.
- Apply requires the independently reviewed content hash, a clean upstream-even
  commit, and the exact production parent/corpus.
- Replay reconstructs the full private manifest and portable receipt, rehashes
  source and transcript artifacts, and rejects tamper, extra keys, path swaps,
  partial directories, duplicates, parent drift, and policy drift.
- Tests prove that any profile/evaluation subject overlap changes feasibility
  and therefore conflicts with the reviewed STOP preview.
- Focused and full tests, compilation, `git diff --check`, independent audit,
  clean push, private apply/replay, and exact `0700`/`0600` modes pass before
  closeout.

## Unit A through Unit C checkpoint

State: CLOSED; Unit D production apply remains OPEN.

- The production-safe preflight is
  `generation-2-evaluation-preflight-76159d12cb5d6e5272a11b72`, content
  SHA-256
  `76159d12cb5d6e5272a11b7219188a75b1872a6168ea92d00b3e535eeddd0104`.
  It binds two evaluation recordings, five opaque evaluation subjects, two
  frozen profile subjects, zero overlap, and all nine candidate units.
- Every unit has a structural maximum of zero genuine and zero impostor trials
  because no evaluation subject has a frozen profile. The frozen minima are 20
  genuine and 100 impostor per unit, so the exact outcome is global STOP with
  `trial_class_denominator_below_policy` before audio preparation.
- One audit repair added the full downstream-action revocation vector to the
  portable receipt and expanded parent-receipt, corpus-path, transcript-byte,
  and receipt-escalation tamper coverage. Independent targeted re-audit
  returned `PASS`.
- Nine focused tests and all 639 repository tests passed; compilation and
  `git diff --check` passed. No audio, window, exact-trial child, model, score,
  terminal metric, or model/method selection was run.
