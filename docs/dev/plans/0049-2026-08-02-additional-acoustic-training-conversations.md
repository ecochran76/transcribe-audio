# Plan 0049 | Additional acoustic training conversations

State: OPEN

Lane: P10

Plan Version: 1

Parent: Plan 0037 terminal closeout and Plan 0048 STOP authority

Owner: primary agent

Expected Write Surface: `acoustic_training_expansion.py`, focused tests, this
plan, `ROADMAP.md`, and `RUNBOOK.md`; private intake, preparation, review, and
reference artifacts only beneath the user-scoped Plan 0037 runtime root.

## Vision alignment

This packet advances speaker identity at maturity `2 - Shadow` by adding up to
five independent, replayable, operator-reviewable conversations to the private
biometric training pool. It improves the chance that enrolled profiles cover
real session and capture variation while preserving abstention and source
boundaries. Progress is measured by exact source/transcript bindings, clean
speaker windows, independent session coverage, and operator-confirmed identity
provenance—not merely by counting files.

This packet does not reopen or rewrite the stopped generation-2 evaluation.
Training from newly selected conversations cannot retroactively satisfy that
generation's frozen 20 genuine / 100 impostor evaluation minima.

## Current state

Plan 0048 closed with terminal `STOP` because the revealed evaluation subjects
did not overlap the frozen profile-subject namespace. Current real profiles
cover two people across three pinned models and were built from two independent
development sessions per person. The user authorized processing up to five
additional conversations and identified `Documents/Sound Recordings` as a
source containing webcam-captured recordings.

File-searcher resolved the live user-named folder. Five already-transcribed,
distinct conversations were selected from it. Each has 2–3 diarized labels,
and its source SHA-256 occurs in none of the three frozen Plan 0037 corpus
manifests. Exact paths and filename-derived selection leads remain private in
the runtime intake authority; they are not committed. Filename metadata is not
enrollment truth, and transcript diarization labels remain unconfirmed for
these exact sources.

## Scope

- Freeze exactly the five selected source and transcript paths, SHA-256 values,
  sizes, durations, utterance counts, and diarized-label sets in a private
  deterministic intake authority.
- Reject duplicate bytes, any overlap with prior Plan 0037 corpora, missing or
  drifted source/transcript bytes, symlinks, nonprivate derived state, and more
  than five conversations.
- Run the existing P1 PCM/quality path and the reviewed P2 preparation path
  without modifying source audio or transcript artifacts.
- Build compact private review packets with timestamped candidate windows for
  each diarized label. Do not treat filenames, calendar data, transcript clues,
  or model output as confirmed identity.
- After operator label confirmation, register only exact reviewed segments in
  the P3 reference lifecycle and build successor profiles through the existing
  staged descendant workflow.
- Define training sufficiency as at least two operator-confirmed people, each
  represented by at least two independent conversations and six eligible
  windows total, with no more than three windows per person per conversation.

## Non-goals

- Do not use the revealed generation-2 evaluation audio or gold as training.
- Do not change the terminal STOP receipt, frozen evaluation policy,
  thresholds, margins, or candidate decision.
- Do not infer speaker identity from a filename, attendee list, transcript
  content, or acoustic model proposal.
- Do not claim every Sound Recordings file used the webcam; the user's folder
  description does not attest each selected file's exact capture device.
- Do not run terminal evaluation, promote a default acoustic path, or begin
  historical reprocessing.

## Execution graph

| Unit | Owner | Dependency | Terminal condition |
| --- | --- | --- | --- |
| A intake authority | primary | exact five live source/transcript pairs | deterministic private preview/apply/replay passes |
| B intake audit | existing read-only reviewer | A | `PASS`, or one bounded repair and re-audit |
| C preparation | primary | B plus clean pushed implementation | exact P1/P2 results replay for every admitted conversation |
| D speaker review | primary plus operator | C | every admitted label is confirmed, mixed, unknown, or excluded |
| E references/profiles | primary | D | sufficiency contract passes or truthful residual blocker is recorded |

The critical path is serialized. The reviewer performs read-only audit only;
the primary owns all writes.

## Gates and stop conditions

- Intake stops on any missing source/transcript, path swap, byte drift,
  duplicate source, prior-corpus overlap, unsupported channel layout, or
  transcript/source mismatch.
- Preparation stops on a failed method, timestamp drift, source mutation,
  nonprivate output, or replay mismatch.
- Enrollment stops until the operator confirms the exact speaker-label mapping
  for each selected conversation. Blanket execution authorization does not
  substitute for an identity fact.
- A person with fewer than two independent confirmed sessions or fewer than
  six eligible windows remains insufficient and is not enrolled from this
  packet.
- No result from this packet changes the closed Plan 0048 terminal outcome.

## Acceptance criteria

- Between one and five, and no more than five, novel conversations are frozen
  with exact full-body replay.
- Every admitted source and transcript remains byte-identical; original audio
  is never overwritten.
- P1/P2 preparation succeeds and replays for every admitted source, or the
  packet stops with per-recording failure evidence.
- Review packets expose bounded timestamps and label IDs without portable raw
  transcript text, names, embeddings, or scores.
- At least two people meet the exact two-session/six-window training
  sufficiency contract before the goal is called complete.
- Focused/full tests, compilation, `git diff --check`, independent audit,
  clean push, private apply/replay, and exact permissions pass.

## Validation

- Adversarial tests cover source/transcript drift, duplicate source bytes,
  prior-corpus overlap, more-than-five input, path swaps, partial runtime
  directories, portable-private-data leakage, and self-rehashed extra keys.
- Run the complete repository test suite after focused validation.
- Verify live source hashes before and after preparation and exact `0700`
  directory / `0600` private-file modes.
- Record operator confirmations and training denominators separately from
  preparation success.
