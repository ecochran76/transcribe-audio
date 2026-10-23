# Shared Speech sample harvesting consumer

State: SOURCE_QUALIFIED
Lane: P09
Critical-Path Owner: Speech graph primary agent
Work item: CochranResearchGroup/speech-receipts#53
Branch: plan-0037-campaign
Base: 5a9c14e5939ebd18972bfce40d54cdc58b9a4f31

## Current State

The existing authenticated meeting console owns recording blobs, normalized
transcript/diarization and participant corrections. Speech Receipts owns generic
harvesting, retained samples and sample associations. No shared harvester
consumer exists in this checkout. This packet implements an optional configured
consumer over supported HTTP and existing public conversation/blob helpers.

## Scope and maturity

Advance VISION outcomes 3, 6, 7 and 8 from separate Level 1 source capabilities to
Level 2 deterministic cross-service fixtures and reviewable meeting controls.
Exchange exact owner/source/revision/hash and operation IDs, stream original
media only under admitted short-lived authenticated source references, show
playable candidates and explicit retention/attributed association proposals.
Keep meeting context and participant assignment authority in Transcribe Audio.
No duplicate canonical recording library or local reimplementation of candidate
selection is permitted.

## Boundaries

No frozen Plan 0072/0064 pilot, private evaluation corpus, existing enrollment
sample import, provider/model call, automatic person acceptance, campaign
mutation, shared service restart, live deployment, commit or push. Tests use
synthetic fixtures and an owned loopback Speech process. Outer meeting ingress
remains the existing authenticated boundary; the source-export route additionally
requires an independent service token and exact configured owner affinity.

## Acceptance and validation

Persist exact request intent before outbound effects; reconcile missing responses
and interrupted journals without re-extracting or repeating retention. Reject
foreign ownership, stale source/timing and unconfigured/unauthorized export.
Test real authenticated cross-service HTTP, source drift, outage, cancellation,
reopen and exact retention/proposal correlation. Add rendered synthetic meeting
candidate playback and explicit keep/propose controls. Preserve historical
curation and speaker-assignment workflows. Focused Python tests target 15 seconds,
one CPU process and no model/provider. Frontend tests/build and affected API
regressions form the source integration lane; installed acceptance is separate.

## Source result

Implemented optional `speech_harvesting.py`, supported `transcript_api.py` routes,
meeting Speakers panel and independent source/observation transports. See
[operator contract](../../shared-speech-harvesting.md). Foreign observations are
append-only proposed/conflicting evidence; original recordings and meeting
assignments remain unchanged. Physical original erasure is unsupported.

Actual cross-service opt-in validation: 7 passed (final duration in private source manifest), including lost
ACK on extraction/retention/proposal and independent observation receipt access.
Default tests skip those three explicit integration cases. Frontend 8 tests and
production build pass. Rendered synthetic meeting decoded a 7.5-second candidate
and explicitly kept it; the original pending participant assignment remained.
Qualification artifacts are user-scoped outside Git. Primary source acceptance was confirmed by the Speech graph owner. Shared integration
and publication remain separate; no installed/live effect or commit is claimed.

## Publication | 2026-10-23

The operator approved publication. Primary verified unchanged qualified source
hashes and published commit `1d2b45f4cd3e9a07683c14d46a723b337063d119` to
`origin/plan-0037-campaign`, with exact remote ref readback. Speech ticket53 is
closed on source/fixture acceptance. This branch remains separate from main;
Transcribe Audio service deployment, real media, original-owner purge and frozen
identity pilots were not performed. Speech's independent installed cutover is
tracked in its plan0048 and ticket50; it does not install this consumer here.
